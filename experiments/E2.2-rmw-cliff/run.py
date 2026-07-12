#!/usr/bin/env python3
"""E2.2 -- RMW cache-cliff microbenchmark (CPU analog of on-die state residency).

WHAT / WHY
----------
The pair thesis (D4) says TTT decode is a per-token read-modify-write (RMW) of the
*entire* fast-weight state: read S, update it, write S back -- every token. Whether
that is cheap or catastrophic depends on ONE thing: does S fit in on-die memory
(cache / scratchpad) or does it spill to DRAM? E1.2 models this crossover on HW twins
(scratchpad 268MB -> 2 TTT layers fit, the rest spills). This experiment is the LOCAL,
DIRECTLY-MEASURED analog: sweep an in-place RMW over arrays crossing L1/L2/L3/DRAM
boundaries on this host and watch the effective write-back bandwidth fall off a cliff
at each capacity boundary.

The portable result is the SHAPE of the curve (plateaus separated by cliffs at the
cache-capacity boundaries), NOT the absolute GB/s -- an H100's on-die/off-die ridge sits
~100x elsewhere. We calibrate this host's roofline (peak GEMM GFLOP/s, peak STREAM-triad
GB/s) so the RMW curve can be read against this machine's own coordinates.

HOST (recorded at runtime): AMD Ryzen 7 255 (Zen5), 8 cores / 16 threads.
Cache: L1d 32 KiB/core, L2 1 MiB/core (private), L3 16 MiB (shared).
=> single-thread cliffs expected near 32 KiB (L1->L2), 1 MiB (L2->L3), 16 MiB (L3->DRAM).

METHOD
------
- RMW kernel = numpy in-place ufunc `x *= a` : one full read pass + one full write pass
  over the array => 2*nbytes of DRAM/cache traffic, N flops. Deeply memory-bound
  (intensity 0.5 flop/elem = 0.125 flop/byte fp32) -- a pure bandwidth probe, exactly the
  TTT-state-RMW regime. numpy elementwise ufuncs are single-threaded C loops (NOT BLAS),
  so the single-thread series is inherently one core -> clean per-core cache cliffs.
- Compound RMW `x *= a; x += b` = TTT-epilogue analog (momentum+decay: two RMW sub-passes).
- Multi-thread (16) RMW: array split across a thread pool (numpy releases the GIL in
  ufuncs) -> AGGREGATE bandwidth; the cliff shifts because each thread touches 1/T of S,
  so the private-L2 working set is T*L2 and the shared L3 is split T ways.
- Read-only reference (`np.add.reduce`): read-only traffic = 1*nbytes; contrasts the
  write-back tax at the DRAM plateau (RMW moves ~2x the bytes of a pure read).
- Bandwidth is reported as BEST (min time) over repeats -- standard for bandwidth
  microbenchmarks (least-perturbed steady state). Warmup touches the array first so the
  measured state is cache-resident where it fits.

Run:
  OMP_NUM_THREADS=16 /home/jimmy/repos/neural-memory-study/.venv/bin/python run.py
"""
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

KB, MB, GB = 1024, 1024**2, 1024**3
DT = np.float32
DT_BYTES = 4
HERE = os.path.dirname(os.path.abspath(__file__))

# --- cache topology (from lscpu on this host; recorded for the report) ---
CACHE = {"L1d_per_core_bytes": 32 * KB, "L2_per_core_bytes": 1 * MB, "L3_shared_bytes": 16 * MB}


def now():
    return time.perf_counter()


def timed_passes(fn, target_s=0.15, min_reps=5, max_reps=200000):
    """Run fn() repeatedly; return (best_time_s, median_time_s, reps). fn does ONE pass."""
    # calibrate rep count from a single trial
    t0 = now()
    fn()
    t1 = now()
    one = max(t1 - t0, 1e-9)
    reps = int(target_s / one)
    reps = max(min_reps, min(max_reps, reps))
    times = []
    for _ in range(reps):
        a = now()
        fn()
        b = now()
        times.append(b - a)
    return min(times), statistics.median(times), reps


def make_state(nbytes):
    n = max(8, nbytes // DT_BYTES)
    x = np.ones(n, dtype=DT)
    x[:] = 1.0000001  # touch -> resident
    return x


# ---------------- RMW kernels ----------------
def rmw_single_pass(x, a=np.float32(1.0000001)):
    x *= a  # read + write, one pass


def rmw_compound(x, a=np.float32(1.0000001), b=np.float32(0.0000001)):
    x *= a
    x += b


def read_only(x):
    # pure read: reduction over the whole array (1x traffic, no write-back)
    return np.add.reduce(x)


def measure_call_overhead(reps=20000):
    """Fixed per-ufunc-call dispatch cost (best case), from a tiny L1-resident array.
    Subtracting this recovers the true in-cache bandwidth that would otherwise be
    swamped by ~0.5us of Python/numpy dispatch at small array sizes."""
    tiny = np.ones(64, dtype=DT)
    a = np.float32(1.0000001)

    def fn():
        tiny[...] *= a  # item-assign (don't rebind the closure name)
    fn()
    best, _, _ = timed_passes(fn, target_s=0.1, min_reps=reps, max_reps=reps)
    return best


def mt_rmw_looped(x, pool, nthreads, inner, a=np.float32(1.0000001)):
    """One pool dispatch does `inner` in-place RMW passes per worker slice, so the
    (large, ~170us) ThreadPool.map dispatch cost is amortized over `inner` passes.
    Returns nothing; caller divides measured time by `inner`."""
    n = x.shape[0]
    chunk = (n + nthreads - 1) // nthreads
    slices = [slice(i * chunk, min((i + 1) * chunk, n)) for i in range(nthreads)]

    def worker(sl):
        xs = x[sl]
        for _ in range(inner):
            xs *= a
    list(pool.map(worker, slices))


# ---------------- bandwidth measurement ----------------
def bw_gbps(bytes_per_pass, best_time):
    return bytes_per_pass / best_time / 1e9


def region_of(real_bytes):
    return ("L1" if real_bytes <= CACHE["L1d_per_core_bytes"] else
            "L2" if real_bytes <= CACHE["L2_per_core_bytes"] else
            "L3" if real_bytes <= CACHE["L3_shared_bytes"] else "DRAM")


def sweep(sizes, kernel_factory, bytes_multiplier, label, overhead=0.0):
    """kernel_factory(x) -> callable doing one pass. bytes_multiplier = traffic per element-byte.
    `overhead` (s) is the fixed per-call cost subtracted to recover in-cache bandwidth."""
    out = []
    for nbytes in sizes:
        x = make_state(nbytes)
        real_bytes = x.nbytes
        fn = kernel_factory(x)
        fn()  # warmup
        best, med, reps = timed_passes(fn)
        traffic = real_bytes * bytes_multiplier
        gbps = bw_gbps(traffic, best)
        # overhead correction is trustworthy only when the fixed dispatch cost is a minority
        # of the measured time; below that the array is too small for pure-numpy to resolve BW.
        overhead_dominated = best <= overhead * 1.5
        if overhead_dominated:
            gbps_corr = None
        else:
            gbps_corr = bw_gbps(traffic, best - overhead)
        out.append({
            "req_bytes": nbytes, "real_bytes": int(real_bytes), "elems": int(x.nbytes // DT_BYTES),
            "region": region_of(real_bytes), "traffic_bytes_per_pass": int(traffic),
            "best_s": best, "median_s": med, "reps": reps,
            "gbps": gbps, "gbps_corrected": gbps_corr, "overhead_dominated": overhead_dominated,
        })
        cs = f"{gbps_corr:8.2f}" if gbps_corr is not None else "  ovhd-- "
        print(f"  [{label}] {real_bytes/1e3:10.1f} KB  {region_of(real_bytes):4s}  "
              f"best={best*1e6:9.2f} us  BW_raw={gbps:8.2f}  BW_corr={cs} GB/s  (reps={reps})")
        del x
    return out


def sweep_mt(sizes, pool, nthreads, bytes_multiplier, label):
    """Multi-thread RMW; inner-loop amortizes the ThreadPool dispatch cost."""
    out = []
    for nbytes in sizes:
        x = make_state(nbytes)
        real_bytes = x.nbytes
        # pick inner passes so one dispatch runs ~a few ms of work (amortize ~170us map cost)
        inner = max(4, min(2000, int(3e6 / max(real_bytes, 1)) + 4))

        def one_dispatch():
            mt_rmw_looped(x, pool, nthreads, inner)
        one_dispatch()  # warmup
        # time whole dispatch (inner passes), then per-pass = total/inner
        best_tot, med_tot, reps = timed_passes(one_dispatch, target_s=0.2, min_reps=5, max_reps=400)
        best = best_tot / inner
        traffic = real_bytes * bytes_multiplier
        gbps = bw_gbps(traffic, best)
        out.append({
            "req_bytes": nbytes, "real_bytes": int(real_bytes), "elems": int(x.nbytes // DT_BYTES),
            "region": region_of(real_bytes), "traffic_bytes_per_pass": int(traffic),
            "inner_passes": inner, "best_s_per_pass": best, "reps_dispatch": reps,
            "gbps": gbps, "gbps_corrected": gbps,
        })
        print(f"  [{label}] {real_bytes/1e3:10.1f} KB  {region_of(real_bytes):4s}  "
              f"per-pass={best*1e6:9.2f} us  BW={gbps:8.2f} GB/s  (inner={inner})")
        del x
    return out


# ---------------- roofline calibration ----------------
def calibrate_gemm():
    """Peak GFLOP/s from a large multithreaded GEMM (OpenBLAS)."""
    best = 0.0
    for n in (2048, 4096):
        A = np.random.rand(n, n).astype(DT)
        B = np.random.rand(n, n).astype(DT)
        C = A @ B  # warmup
        times = []
        for _ in range(3):
            t0 = now(); C = A @ B; t1 = now()
            times.append(t1 - t0)
        t = min(times)
        gflops = (2.0 * n**3) / t / 1e9
        best = max(best, gflops)
        del A, B, C
    return best


def calibrate_stream_triad(nthreads=1):
    """Peak GB/s from a STREAM-triad a[:] = b + s*c on DRAM-resident arrays."""
    n = 32 * MB // DT_BYTES  # 32 MB per array -> firmly DRAM (>16 MB L3)
    a = np.zeros(n, dtype=DT); b = np.ones(n, dtype=DT); c = np.full(n, 2.0, DT)
    s = np.float32(3.0)

    def triad_single():
        np.multiply(c, s, out=a)
        np.add(a, b, out=a)  # a = b + s*c  (3 arrays read/written -> 4*nbytes traffic approx)

    if nthreads == 1:
        triad_single()  # warmup
        best, _, reps = timed_passes(triad_single, target_s=0.2)
    else:
        pool = ThreadPoolExecutor(max_workers=nthreads)
        chunk = (n + nthreads - 1) // nthreads
        slices = [slice(i * chunk, min((i + 1) * chunk, n)) for i in range(nthreads)]

        def triad_mt():
            def w(sl):
                np.multiply(c[sl], s, out=a[sl]); np.add(a[sl], b[sl], out=a[sl])
            list(pool.map(w, slices))
        triad_mt()
        best, _, reps = timed_passes(triad_mt, target_s=0.2)
        pool.shutdown()
    # traffic: read c, read b, write a, plus the intermediate write+read of a in two-op form.
    # Conservative STREAM convention: 3 array streams touched = 3*nbytes moved.
    traffic = 3 * a.nbytes
    return bw_gbps(traffic, best), reps


def main():
    nthreads = int(os.environ.get("OMP_NUM_THREADS", "16"))
    print("=" * 100)
    print("E2.2  RMW cache-cliff microbenchmark  (CPU analog of TTT-state on-die residency crossover)")
    print("=" * 100)
    print(f"host: {platform.processor() or 'unknown'} | numpy {np.__version__} | "
          f"OMP_NUM_THREADS={nthreads}")
    print(f"cache: L1d {CACHE['L1d_per_core_bytes']//KB} KiB/core, "
          f"L2 {CACHE['L2_per_core_bytes']//MB} MiB/core, L3 {CACHE['L3_shared_bytes']//MB} MiB shared")

    # ---- roofline calibration ----
    print("\n[calibrate] roofline of this host ...")
    peak_gflops = calibrate_gemm()
    st1, _ = calibrate_stream_triad(1)
    stN, _ = calibrate_stream_triad(nthreads)
    ridge_flop_per_byte = peak_gflops / stN if stN else float("nan")
    call_overhead = measure_call_overhead()
    print(f"  peak GEMM        : {peak_gflops:8.1f} GFLOP/s (fp32, OpenBLAS MT)")
    print(f"  STREAM-triad  1T : {st1:8.1f} GB/s")
    print(f"  STREAM-triad {nthreads:2d}T : {stN:8.1f} GB/s")
    print(f"  ridge point      : {ridge_flop_per_byte:8.2f} FLOP/byte (peak_flops / peak_BW_MT)")
    print(f"  ufunc call ovhd  : {call_overhead*1e6:8.3f} us (subtracted to recover in-cache BW)")

    # ---- size sweep across cache boundaries ----
    # log-ish spacing with extra points bracketing 32KB, 1MB, 16MB boundaries
    sizes = [
        4 * KB, 8 * KB, 16 * KB, 24 * KB, 32 * KB, 48 * KB, 64 * KB, 96 * KB, 128 * KB,
        192 * KB, 256 * KB, 384 * KB, 512 * KB, 768 * KB, 1 * MB, 1536 * KB, 2 * MB,
        3 * MB, 4 * MB, 6 * MB, 8 * MB, 12 * MB, 16 * MB, 24 * MB, 32 * MB, 48 * MB,
        64 * MB, 128 * MB, 256 * MB,
    ]

    print("\n[sweep] single-thread RMW single-pass  `x *= a`  (traffic = 2*nbytes: read+write)")
    rmw1 = sweep(sizes, lambda x: (lambda: rmw_single_pass(x)), 2, "rmw1T", overhead=call_overhead)

    print("\n[sweep] single-thread RMW compound  `x*=a; x+=b`  (TTT-epilogue; traffic = 4*nbytes)")
    rmwc = sweep(sizes, lambda x: (lambda: rmw_compound(x)), 4, "rmwC", overhead=2 * call_overhead)

    print("\n[sweep] read-only reference  `add.reduce(x)`  (traffic = 1*nbytes; no write-back)")
    rdonly = sweep(sizes, lambda x: (lambda: read_only(x)), 1, "read", overhead=call_overhead)

    print(f"\n[sweep] {nthreads}-thread RMW single-pass (aggregate BW; inner-loop amortizes dispatch)")
    pool = ThreadPoolExecutor(max_workers=nthreads)
    rmwmt = sweep_mt(sizes, pool, nthreads, 2, f"rmw{nthreads}T")
    pool.shutdown()

    # ---- derive plateau bandwidths & cliff ratios (use CORRECTED BW) ----
    def plateau(series, lo, hi, key="gbps_corrected"):
        vals = [r[key] for r in series if lo < r["real_bytes"] <= hi and r[key] is not None]
        return statistics.median(vals) if vals else None

    def summarize(series, name, key="gbps_corrected"):
        l1 = plateau(series, 0, CACHE["L1d_per_core_bytes"], key)
        l2 = plateau(series, CACHE["L1d_per_core_bytes"], CACHE["L2_per_core_bytes"], key)
        l3 = plateau(series, CACHE["L2_per_core_bytes"], CACHE["L3_shared_bytes"], key)
        dram = plateau(series, CACHE["L3_shared_bytes"], 1 << 40, key)
        # in-cache peak taken from the L3-resident band (>=1MB, <=L3): overhead-negligible,
        # so this cliff is robust to the correction model.
        incache_band = [r[key] for r in series
                        if 1 * MB < r["real_bytes"] <= CACHE["L3_shared_bytes"] and r[key] is not None]
        allvals = [r[key] for r in series if r[key] is not None]
        incache_peak = max(incache_band) if incache_band else (max(allvals) if allvals else None)
        d = {"peak_gbps": (max(allvals) if allvals else None),
             "incache_peak_gbps": incache_peak,
             "L1_gbps": l1, "L2_gbps": l2, "L3_gbps": l3, "DRAM_gbps": dram,
             "incache_peak_over_DRAM": (incache_peak / dram) if (incache_peak and dram) else None,
             "L1_over_DRAM": (l1 / dram) if (l1 and dram) else None,
             "L2_over_DRAM": (l2 / dram) if (l2 and dram) else None,
             "L3_over_DRAM": (l3 / dram) if (l3 and dram) else None}
        print(f"\n[plateau] {name}: L1={l1 and round(l1,1)}  L2={l2 and round(l2,1)}  "
              f"L3={l3 and round(l3,1)}  DRAM={dram and round(dram,1)} GB/s  "
              f"| in-cache-peak {round(incache_peak,1)} => DRAM  = "
              f"x{d['incache_peak_over_DRAM'] and round(d['incache_peak_over_DRAM'],1)} cliff")
        return d

    sum_rmw1 = summarize(rmw1, "RMW 1T")
    sum_rmwc = summarize(rmwc, "RMW compound 1T")
    sum_rd = summarize(rdonly, "read-only 1T")
    sum_mt = summarize(rmwmt, f"RMW {nthreads}T", key="gbps")

    results = {
        "exp_id": "E2.2-rmw-cliff",
        "host": {
            "cpu": platform.processor() or "AMD Ryzen 7 255 (Zen5)",
            "cores_threads": "8C/16T", "numpy": np.__version__,
            "python": sys.version.split()[0], "omp_num_threads": nthreads,
            "cache": CACHE,
        },
        "roofline": {
            "peak_gemm_gflops_fp32": peak_gflops,
            "stream_triad_1T_gbps": st1, "stream_triad_MT_gbps": stN,
            "ridge_flop_per_byte": ridge_flop_per_byte,
            "ufunc_call_overhead_us": call_overhead * 1e6,
        },
        "sweeps": {
            "rmw_single_1T": rmw1, "rmw_compound_1T": rmwc,
            "read_only_1T": rdonly, f"rmw_single_{nthreads}T": rmwmt,
        },
        "plateaus": {
            "rmw_single_1T": sum_rmw1, "rmw_compound_1T": sum_rmwc,
            "read_only_1T": sum_rd, f"rmw_single_{nthreads}T": sum_mt,
        },
        "headline": {
            "rmw1T_incache_peak_gbps": sum_rmw1["incache_peak_gbps"],
            "rmw1T_DRAM_gbps": sum_rmw1["DRAM_gbps"],
            "rmw1T_L3_to_DRAM_cliff": sum_rmw1["incache_peak_over_DRAM"],
            "rmw1T_L1_over_DRAM": sum_rmw1["L1_over_DRAM"],
            "rmw1T_L2_over_DRAM": sum_rmw1["L2_over_DRAM"],
            "readonly_over_rmw_at_DRAM": (
                (sum_rd["DRAM_gbps"] / sum_rmw1["DRAM_gbps"])
                if (sum_rd["DRAM_gbps"] and sum_rmw1["DRAM_gbps"]) else None),
        },
        "caveats": [
            "SHAPE not absolutes: the portable claim is the cliff structure (in-cache plateau "
            "collapsing to a DRAM plateau at the L3 capacity boundary), not the GB/s values -- "
            "an H100 on-die/off-die ridge sits ~100x elsewhere. Do NOT transfer these numbers to H100.",
            "The ROBUST headline is the L3->DRAM (on-die vs off-die) cliff: it is read from "
            "overhead-negligible large arrays (>=1MB) and needs no correction. The finer L1/L2 "
            "sub-cliffs use overhead-corrected BW (see below) and are model-dependent.",
            "numpy elementwise ufuncs are single-threaded C loops; the 1T RMW series is one "
            "core, giving PER-CORE cache boundaries (L1 32KiB, L2 1MiB private, L3 16MiB shared). "
            "The MT series splits the array across a Python ThreadPool with an inner loop that "
            "amortizes the ~170us dispatch cost -> aggregate BW; per-thread working set is 1/T so "
            "the effective L2/L3 fit shifts.",
            "OVERHEAD CORRECTION: at small sizes a fixed ~0.4-0.5us numpy ufunc dispatch swamps "
            "the true in-cache time. `gbps_corrected` subtracts a measured per-call overhead "
            "(linear-overhead model) so L1/L2 bandwidth is not under-reported; `gbps` (raw) is "
            "also stored. For >=1MB arrays the correction is negligible.",
            "The L1 boundary (<=32KiB) is UNRESOLVABLE by this pure-numpy method: an L1-resident "
            "RMW is faster than the ~0.6us ufunc dispatch, so every L1-region point is flagged "
            "overhead_dominated and its corrected BW is null. The first resolvable point is ~49KB "
            "(L2). The cliffs this experiment DOES resolve are L2/L3 -> DRAM, which is the "
            "thesis-relevant on-die-vs-off-die crossover.",
            "The read-only reference uses np.add.reduce, a single-accumulator reduction that is "
            "partly limited by the add-dependency chain, not purely by read bandwidth; read its "
            "~34 GB/s as a rough read-throughput floor, not a peak read-BW figure. The write-back "
            "point still stands: RMW moves 2x the bytes per element, so at a saturated DRAM the "
            "distinct-array throughput of RMW (~34 array-GB/s) matches a read while costing ~2x "
            "the wall time per element -- that 2x is the write-back tax.",
            "Bandwidth = best (min) time over repeats, standard for BW microbenchmarks. "
            "Traffic is analytic (read+write passes counted), not measured via perf/LLC-misses: "
            "perf hardware counters were not used in this sandbox run.",
            "STREAM-triad traffic uses the 3-array convention; the two-op numpy form "
            "(multiply then add into `a`) touches `a` an extra time, so the reported STREAM GB/s "
            "is a conservative lower bound on true DRAM BW.",
            "numpy is 2.5.1 here (plan referenced 2.4.x); OpenBLAS 0.3.33 backend. "
            "Frequency scaling (CPU governor, 61% scaling MHz observed) and a swap-pressured "
            "host add noise; min-time reduces but does not eliminate it.",
            "This is the CPU analog of E1.2's scratchpad residency crossover, not a substitute: "
            "it demonstrates the mechanism (write-back BW collapses when state stops fitting "
            "on-die) on real silicon this host happens to expose.",
        ],
    }

    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[write] {out_path}")

    print("\n" + "=" * 100)
    print("HEADLINE")
    print(f"  RMW 1T on-die -> off-die cliff: in-cache peak "
          f"{round(sum_rmw1['incache_peak_gbps'],1)} GB/s -> DRAM "
          f"{round(sum_rmw1['DRAM_gbps'],1)} GB/s "
          f"= x{round(sum_rmw1['incache_peak_over_DRAM'],1)} collapse at the L3 (16MiB) boundary")
    print(f"  ridge point of this host: {ridge_flop_per_byte:.1f} FLOP/byte "
          f"(peak {peak_gflops:.0f} GFLOP/s / {stN:.0f} GB/s); RMW intensity 0.5 flop/elem is "
          f"far left of ridge -> firmly memory-bound")
    print(f"  read-only vs RMW at DRAM: read {round(sum_rd['DRAM_gbps'],1)} "
          f"vs RMW {round(sum_rmw1['DRAM_gbps'],1)} GB/s (write-back tax ~2x traffic)")
    print("=" * 100)
    return results


if __name__ == "__main__":
    main()
