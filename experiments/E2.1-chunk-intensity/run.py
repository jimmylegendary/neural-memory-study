#!/usr/bin/env python3
"""
E2.1 chunkwise scan vs sequential update -- arithmetic-intensity curve (CPU microbench)

Part III (D4 workload-split pair thesis) supporting experiment.

Claim under test (ch09/ch10): chunk size C is simultaneously a *semantic*
hyperparameter AND the x-axis of the roofline. As C grows, the effective
arithmetic intensity of a chunkwise linear-attention / DeltaNet-style scan
rises from the memory-bound floor (C=1, per-token rank-1 state RMW, the TTT
decode regime) up toward the compute-bound plateau (large C, the prefill/
training regime). This is an *algorithmic* curve -- hardware-independent in
SHAPE -- so it must reproduce on a commodity CPU even though the absolute
ridge coordinate differs ~100x from an H100.

Corrected ch10 closed form being checked (fp32, per chunk of C tokens, dim d):

    AI(C) = (4 C^2 d + 6 C d^2) / (4 d^2 + 8 C d)          [FLOP / byte]

    numerator  = intra-chunk attention (4 C^2 d)  +  three d^2-scaling GEMMs
                 (Q@S inter, K@S delta-correction, K^T@(V-KS) state RMW; 6 C d^2)
    denominator= state matrix touched (4 d^2 B)   +  C x d input/output tiles
                 (K,V,Q,O -> 8 C d B in fp32)

The kernel implemented below is a numerically-real DeltaNet chunkwise scan
(delta rule  S <- S + K^T (V - K S)) whose true FLOP count matches the
numerator exactly; we time it and divide analytic FLOPs by wall time to get
effective GFLOP/s, then place each (C,d) point on this host's measured
roofline.

HONESTY (plan section 5, item 5): CPU measurement validates the *shape* of
AI(C) and the memory->compute crossover, NOT any H100 absolute number. The
host ridge coordinate sits well below H100's ~295 FLOP/B (gap factor is
host-dependent and reported at runtime); no numeric transport to accelerator
silicon is claimed.
"""
import os, sys, json, time, statistics, platform, subprocess

# OpenBLAS thread count must be fixed before numpy import.
THREADS = int(os.environ.get("E21_THREADS", "8"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(THREADS))
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DTYPE = np.float32
DTYPE_BYTES = 4

# ---- H100 reference coordinate (dgx_h100_x4.json twin, hat-schema SoT) ----
# imported only to state, in the output, how far the host ridge sits from the
# accelerator ridge -- makes the "shape only, not absolute" caveat quantitative.
H100_PEAK_FLOPS = 989_472e9          # bf16 MXU, 4xH100 twin value used in E3
H100_BW_HBM = 3.35e12                # HBM3
H100_RIDGE = H100_PEAK_FLOPS / H100_BW_HBM   # ~295 FLOP/B


# ----------------------------------------------------------------------------
# Analytic AI(C) -- the ch10 formula, evaluated directly.
# ----------------------------------------------------------------------------
def ai_ch10(C, d):
    num = 4.0 * C * C * d + 6.0 * C * d * d
    den = (4.0 * d * d + 8.0 * C * d) * (DTYPE_BYTES / 4.0)  # formula written for fp32
    return num / den

def flops_per_chunk(C, d):
    return 4.0 * C * C * d + 6.0 * C * d * d

def bytes_per_chunk(C, d):
    # ch10 denominator, in fp32 bytes
    return (4.0 * d * d + 8.0 * C * d) * (DTYPE_BYTES / 4.0)


# ----------------------------------------------------------------------------
# Roofline calibration for THIS host.
# ----------------------------------------------------------------------------
def calibrate_gemm_peak(N=4096, reps=5):
    """Effective compute peak: large square fp32 GEMM, 2 N^3 FLOPs."""
    a = np.random.rand(N, N).astype(DTYPE)
    b = np.random.rand(N, N).astype(DTYPE)
    c = np.empty((N, N), dtype=DTYPE)
    np.dot(a, b, out=c)  # warm
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        np.dot(a, b, out=c)
        ts.append(time.perf_counter() - t0)
    t = min(ts)  # best-case peak
    gflops = (2.0 * N**3) / t / 1e9
    return gflops, N, t

def calibrate_stream_bw(nbytes_target=512 * 1024 * 1024, reps=7):
    """Peak sustained DRAM BW via STREAM-triad-like RMW: a[:] = b + s*c.
    Traffic = 3 arrays (read b, read c, write a) x 4 B x n."""
    n = nbytes_target // DTYPE_BYTES
    a = np.empty(n, dtype=DTYPE)
    b = np.random.rand(n).astype(DTYPE)
    c = np.random.rand(n).astype(DTYPE)
    s = DTYPE(1.0001)
    np.multiply(c, s, out=a); np.add(a, b, out=a)  # warm
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        np.multiply(c, s, out=a)
        np.add(a, b, out=a)
        ts.append(time.perf_counter() - t0)
    t = min(ts)
    gbps = (3.0 * DTYPE_BYTES * n) / t / 1e9
    return gbps, n, t


# ----------------------------------------------------------------------------
# DeltaNet-style chunkwise scan.  Delta rule:  S <- S + K^T (V - K S)
#   inter   = Q @ S                (2 C d^2)
#   correct = K @ S                (2 C d^2)   predicted values (delta rule)
#   update  = K^T @ (V - correct)  (2 C d^2)   state RMW
#   intra   = tril(Q K^T) @ (V-correct)        (4 C^2 d)
# FLOP count == ch10 numerator 4C^2 d + 6 C d^2 exactly.
# ----------------------------------------------------------------------------
def make_chunk_pool(C, d, pool_chunks):
    rng = np.random.default_rng(0)
    scale = 1.0 / (d ** 0.5)
    Q = (rng.standard_normal((pool_chunks, C, d)) * scale).astype(DTYPE)
    K = (rng.standard_normal((pool_chunks, C, d)) * scale).astype(DTYPE)
    V = (rng.standard_normal((pool_chunks, C, d)) * scale).astype(DTYPE)
    return Q, K, V

def run_chunkwise(Q, K, V, S, C, d, nc, mask):
    """Process nc chunks in-place through state S. Returns nothing (timed by caller)."""
    pool = Q.shape[0]
    for i in range(nc):
        j = i % pool
        Qc = Q[j]; Kc = K[j]; Vc = V[j]
        correct = Kc @ S                 # C x d   (2 C d^2)  delta correction
        resid = Vc - correct             # C x d   elementwise
        inter = Qc @ S                   # C x d   (2 C d^2)  inter-chunk (state read)
        A = Qc @ Kc.T                     # C x C   (2 C^2 d)  intra scores
        if mask is not None:
            A *= mask                     # causal
        O = A @ resid                     # C x d   (2 C^2 d)  intra output
        O += inter                        # combine (cheap)
        S += Kc.T @ resid                 # d x d   (2 C d^2)  state RMW update


def measure(C, d, target_flops=8e8, reps=5):
    fpc = flops_per_chunk(C, d)
    nc = int(max(1, round(target_flops / fpc)))
    nc = min(nc, 20000)                    # loop-overhead safety cap
    pool = min(nc, 32)                     # bounded working set of input tiles
    Q, K, V = make_chunk_pool(C, d, pool)
    mask = np.tril(np.ones((C, C), dtype=DTYPE)) if C > 1 else None
    # warm
    S = np.zeros((d, d), dtype=DTYPE)
    run_chunkwise(Q, K, V, S, C, d, min(nc, pool), mask)
    ts = []
    for _ in range(reps):
        S = np.zeros((d, d), dtype=DTYPE)
        t0 = time.perf_counter()
        run_chunkwise(Q, K, V, S, C, d, nc, mask)
        ts.append(time.perf_counter() - t0)
    t = statistics.median(ts)
    total_flops = fpc * nc
    gflops = total_flops / t / 1e9
    return {
        "C": C, "d": d, "nc_chunks": nc,
        "flops_per_chunk": fpc,
        "bytes_per_chunk": bytes_per_chunk(C, d),
        "ai_ch10_flop_per_byte": ai_ch10(C, d),
        "median_time_s": t,
        "measured_gflops": gflops,
        "state_bytes": d * d * DTYPE_BYTES,
    }


def main():
    Cs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    ds = [512, 1024, 2048]

    print(f"# E2.1 chunkwise arithmetic-intensity curve")
    print(f"# host    : {platform.platform()}")
    print(f"# cpu     : AMD Ryzen 7 255 (8c/16t, L1d 32K/c, L2 1M/c=8M, L3 16M)")
    print(f"# numpy   : {np.__version__}  (OpenBLAS, threads={THREADS})")
    print(f"# dtype   : fp32")
    print(f"# ch10 AI : AI(C) = (4C^2 d + 6 C d^2) / (4 d^2 + 8 C d)")
    print()

    print("## roofline calibration (this host)")
    gemm_gflops, gemm_N, gemm_t = calibrate_gemm_peak()
    bw_gbps, bw_n, bw_t = calibrate_stream_bw()
    ridge = gemm_gflops / bw_gbps
    print(f"  peak GFLOP/s (GEMM {gemm_N}^3)        : {gemm_gflops:8.1f}   (t={gemm_t*1e3:.2f} ms)")
    print(f"  peak GB/s   (STREAM-triad {bw_n} elt): {bw_gbps:8.1f}   (t={bw_t*1e3:.2f} ms)")
    print(f"  ridge point (peak/BW)                : {ridge:8.2f} FLOP/B")
    print(f"  [ref] H100 twin ridge                : {H100_RIDGE:8.2f} FLOP/B  ({H100_RIDGE/ridge:.0f}x host)")
    print()

    print("## AI(C) sweep")
    hdr = f"{'d':>5} {'C':>5} {'nc':>7} {'AI_ch10':>9} {'meas_GF/s':>10} {'roof_GF/s':>10} {'bound':>7} {'stateMB':>8}"
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for d in ds:
        for C in Cs:
            m = measure(C, d)
            ai = m["ai_ch10_flop_per_byte"]
            roof = min(gemm_gflops, ai * bw_gbps)   # roofline-predicted attainable
            bound = "compute" if ai >= ridge else "memory"
            m["roofline_gflops"] = roof
            m["bound"] = bound
            m["frac_of_roofline"] = m["measured_gflops"] / roof if roof > 0 else 0.0
            m["frac_of_peak"] = m["measured_gflops"] / gemm_gflops
            rows.append(m)
            print(f"{d:5d} {C:5d} {m['nc_chunks']:7d} {ai:9.2f} "
                  f"{m['measured_gflops']:10.1f} {roof:10.1f} {bound:>7} "
                  f"{m['state_bytes']/1e6:8.2f}")
        print()

    # crossover chunk C* per d: smallest C with AI(C) >= host ridge
    cross = {}
    for d in ds:
        cstar = None
        for C in Cs:
            if ai_ch10(C, d) >= ridge:
                cstar = C
                break
        # also solve continuously: AI(C)=ridge
        # (4C^2 d + 6Cd^2) = ridge*(4d^2 + 8Cd)
        # 4d C^2 + (6d^2 - 8 d ridge) C - 4 d^2 ridge = 0
        aq = 4.0 * d
        bq = 6.0 * d * d - 8.0 * d * ridge
        cq = -4.0 * d * d * ridge
        disc = bq * bq - 4 * aq * cq
        c_cont = (-bq + disc ** 0.5) / (2 * aq)
        cross[str(d)] = {"C_star_discrete": cstar, "C_star_continuous": round(c_cont, 2)}

    print("## memory->compute crossover chunk C* (AI(C*) = host ridge)")
    for d in ds:
        c = cross[str(d)]
        print(f"  d={d:5d}:  C* (discrete grid) = {c['C_star_discrete']}, "
              f"C* (continuous) = {c['C_star_continuous']}")
    print()

    results = {
        "experiment": "E2.1-chunk-intensity",
        "workload": "DeltaNet-style chunkwise linear-attention scan (delta rule S<-S+K^T(V-KS)); "
                    "C=1 is the per-token rank-1 state RMW = TTT decode regime",
        "purpose": "empirically verify the SHAPE of the corrected ch10 AI(C) curve and the "
                   "memory->compute crossover on a commodity CPU roofline",
        "formula_ch10": "AI(C) = (4C^2 d + 6 C d^2) / (4 d^2 + 8 C d)   [fp32 FLOP/byte]",
        "dtype": "float32",
        "host": {
            "platform": platform.platform(),
            "cpu": "AMD Ryzen 7 255 (8 core / 16 thread; L1d 32KB/core, L2 1MB/core (8MB), L3 16MB shared)",
            "numpy": np.__version__,
            "blas": "scipy-openblas 0.3.33 (Haswell/AVX2 kernels)",
            "threads": THREADS,
        },
        "roofline_host": {
            "peak_gflops": gemm_gflops,
            "peak_gbps": bw_gbps,
            "ridge_flop_per_byte": ridge,
            "gemm_calib_N": gemm_N,
            "stream_calib_elems": bw_n,
        },
        "roofline_h100_ref": {
            "peak_flops_s": H100_PEAK_FLOPS,
            "bw_hbm_bps": H100_BW_HBM,
            "ridge_flop_per_byte": H100_RIDGE,
            "host_ridge_gap_factor": H100_RIDGE / ridge,
            "note": "shape-only transport; host ridge sits ~%.0fx below H100 ridge" % (H100_RIDGE / ridge),
        },
        "sweep": {"C": Cs, "d": ds, "T_note": "per-chunk timing; nc chunks per point at ~constant FLOP budget"},
        "curve": rows,
        "crossover_C_star": cross,
        "caveats": [
            "CPU measurement validates the SHAPE of AI(C) (how intensity scales with chunk C) and the "
            "memory->compute crossover ONLY; it does NOT transport to H100 absolute numbers -- the host "
            f"ridge (~{ridge:.1f} FLOP/B) sits ~{H100_RIDGE/ridge:.0f}x below the H100 twin ridge ({H100_RIDGE:.0f} FLOP/B).",
            "C=1 (sequential per-token) path is rank-1 GEMV-dominated; OpenBLAS may drop such small ops to "
            "1-2 threads, so the C=1 point is confounded by threading. Both threads=8 and threads=1 runs are "
            "provided (E21_THREADS env) so the algorithmic vs thread effect can be separated.",
            "DRAM traffic is computed analytically from the ch10 byte model (4 d^2 + 8 C d, fp32), not measured "
            "with perf/LLC counters (perf not available in this sandbox). Effective GFLOP/s IS measured (wall clock).",
            "measured_gflops is a real wall-clock throughput; roofline_gflops is the analytic attainable "
            "min(peak, AI*BW). Their ratio (frac_of_roofline) shows how tightly the CPU tracks the predicted "
            "roofline, not a silicon-accurate efficiency.",
            "State matrix S is reused across the nc timed chunks, so it stays cache-resident when it fits "
            "(d=512 -> 1MB in L2; d=2048 -> 16MB at the L3 edge); the resulting cache-residency effect on the "
            "low-C memory-bound points is the local analogue of the E1.2/E2.2 residency cliff, not a DRAM-only rate.",
        ],
    }
    out_name = "results.json" if THREADS == 8 else f"results_threads{THREADS}.json"
    with open(os.path.join(HERE, out_name), "w") as f:
        json.dump(results, f, indent=2)
    print(f"# wrote {out_name} ({len(rows)} curve points)")

    # headline summary line
    d_ref = 2048
    lowC = next(r for r in rows if r["d"] == d_ref and r["C"] == 1)
    hiC = next(r for r in rows if r["d"] == d_ref and r["C"] == 1024)
    print(f"# HEADLINE (d={d_ref}): AI rises {lowC['ai_ch10_flop_per_byte']:.2f} -> "
          f"{hiC['ai_ch10_flop_per_byte']:.1f} FLOP/B as C 1->1024 "
          f"({hiC['ai_ch10_flop_per_byte']/lowC['ai_ch10_flop_per_byte']:.0f}x); "
          f"measured GFLOP/s {lowC['measured_gflops']:.0f} -> {hiC['measured_gflops']:.0f}; "
          f"host ridge {ridge:.1f} FLOP/B")


if __name__ == "__main__":
    main()
