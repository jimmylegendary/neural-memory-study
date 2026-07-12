#!/usr/bin/env python3
"""E3.1 — Analytical crossover model for TTT/neural-memory decode.

Part III (D4 workload-split pair thesis) analytical backbone. A single closed-form cost
model, parameterised by (d, m, L, chunk C, batch B, context S) and grounded on the
dgx_h100_x4.json HW twin (SoT for HBM3 BW/energy/capacity and the compute roofline),
computes three crossovers reused across the monograph:

  (a) S*   — context length at which the KV-cache read per token equals the TTT fast-weight
             state read-modify-write (RMW) per token. Below S* the transformer's KV traffic
             is cheaper; above it the transformer pays more per token than the TTT recurrence.
  (b) C*   — chunk size at which the chunkwise inner-loop training/prefill kernel crosses the
             HW ridge (compute-bound above C*, memory-bound below). The prefill/training half
             of the pair thesis: enlarge the chunk and you move into accelerator territory.
  (c) B_max — batch size at which unshared per-sequence state either (i) saturates HBM BW under
             a per-token latency target, or (ii) exceeds the on-die / HBM capacity budget.

State-size multipliers m (state bytes/layer = m * d^2 * dtype_bytes) per method, from the six
dossiers in ../../notes/*.json:
  Titans (2501.00663): neural memory = 2-layer MLP (W1 d x 4d, W2 4d x d = 8 d^2 params) PLUS a
      momentum/surprise state S_t of equal shape -> m = 16.  (m=8 without momentum.)
  Miras/Moneta/Yaad (2504.13173): 2-layer residual MLP memory, retention regulariser; m in [8,16].
  Atlas (2505.23735): deep (2-layer) memory + Omega rule with a Muon (2nd-order) preconditioner
      state -> m = 16.
  TNT (2511.07343): 2-layer MLP memory, chunkwise; single active fast-weight m = 8; the paper's
      chunk sets {8,16,32,64,128} are the C* sweep.
  HOPE / Nested Learning (2512.24695): CMS = chain of k MLP memory blocks with different update
      frequencies. Only the fastest level RMWs every token (m_fast ~ 8); total resident state
      across ~6 levels is larger (reported separately, most of it low-frequency).
  Sleep (2606.03979): same CMS substrate; consolidation is OFFLINE (the P -> infinity row).
      Per-token decode cost = HOPE fast level; +5 MLP blocks of dim 64 are negligible per token.

HONESTY (mirrors ../../notes/local-experiments-plan.md §5):
  * hatir/twin numbers are exploration-grade / RELATIVE — for ranking + crossover, not
    silicon-accurate absolutes. We only promote ratios and crossover *positions*.
  * The six papers do NOT publish H100 wall-clock decode; absolute us/token here is a roofline
    lower bound, unverifiable externally, deferred to the in-house A100 runbook (Part III-a).
  * Per-kernel roofline only: no kernel-launch / scheduling / tail overheads.
  * KV head config assumed GQA-8 / head-dim 128 (kv_dim = Hkv*hd = 1024) unless a row says else.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TWIN_PATH = "/home/jimmy/repos/hat-schema/twins/dgx_h100_x4.json"
KB, MB, GB = 1024, 1024**2, 1024**3


# ---------------------------------------------------------------------------
# HW constants — imported from the dgx_h100_x4 twin (SoT). Explicit datasheet
# fallback with provenance if the twin is unavailable, so the model is reproducible.
# ---------------------------------------------------------------------------
def load_hw(twin_path=TWIN_PATH):
    prov = "dgx_h100_x4.json twin (hat-schema SoT)"
    try:
        t = json.load(open(twin_path))

        def find(o, role):
            if isinstance(o, dict):
                if o.get("role") == role or o.get("level_id") == role:
                    return o
                for c in o.get("children", []):
                    r = find(c, role)
                    if r:
                        return r
            return None

        hbm = find(t, "hbm")
        l2 = find(t, "l2")
        mxu = find(t, "compute") or find(t, "mxu")
        peak_macs = mxu["peak_macs_per_s"] * mxu.get("instances", 1)
        return {
            "bw_hbm_bps": hbm["bandwidth_bps"],
            "epb_hbm": hbm["energy_pj_per_byte"],
            "hbm_cap_bytes": hbm["capacity_bytes"],
            "ondie_bytes": l2["capacity_bytes"],       # L2 as the on-die residency budget
            "peak_macs_s": peak_macs,
            "peak_flops_s": 2 * peak_macs,
            "provenance": prov,
        }
    except Exception as e:  # pragma: no cover — datasheet fallback
        return {
            "bw_hbm_bps": 3.35e12, "epb_hbm": 7.0, "hbm_cap_bytes": 80 * GB,
            "ondie_bytes": 50 * MB, "peak_macs_s": 4.947e14, "peak_flops_s": 9.894e14,
            "provenance": f"H100 SXM5 datasheet fallback (twin unreadable: {e})",
        }


HW = load_hw()
RIDGE = HW["peak_flops_s"] / HW["bw_hbm_bps"]   # FLOP/byte at the roofline knee


# ---------------------------------------------------------------------------
# Closed-form per-token TTT decode cost (one layer, then x L).
# state bytes/layer = m * d^2 * dtype;  RMW traffic = 2x state (read S + write S back).
# ---------------------------------------------------------------------------
def state_bytes_layer(d, m, dtype_bytes=2):
    return m * d * d * dtype_bytes


def ttt_rmw_per_token(d, m, L, dtype_bytes=2):
    """All-layer read-modify-write traffic for one decode token (bytes)."""
    return 2 * state_bytes_layer(d, m, dtype_bytes) * L


def ttt_decode(d, m, L, dtype_bytes=2, batch=1):
    traffic = ttt_rmw_per_token(d, m, L, dtype_bytes) * batch
    time_s = traffic / HW["bw_hbm_bps"]
    energy_pj = traffic * HW["epb_hbm"]
    # per-token compute: gradient GEMVs (~2-3 O(d^2) MACs) + O(m d^2) elementwise epilogue.
    macs = (3 * d * d + m * d * d) * L * batch
    compute_s = macs / HW["peak_macs_s"]
    return {
        "state_MB_layer": state_bytes_layer(d, m, dtype_bytes) / 1e6,
        "rmw_GB_token": traffic / 1e9,
        "rmw_ms_token": time_s * 1e3,
        "rmw_uJ_token": energy_pj / 1e6,
        "compute_ms_token": compute_s * 1e3,
        "arith_intensity_flopB": (2 * macs) / traffic,
        "bound": "memory" if time_s > compute_s else "compute",
    }


# ---------------------------------------------------------------------------
# (a) S* — KV read == TTT RMW crossover context length (tokens).
# KV read/token/layer = 2 * kv_dim * S * kv_dtype ; TTT RMW/token/layer = 2 * m d^2 * ttt_dtype.
# S* = m d^2 (ttt_dtype/kv_dtype) / kv_dim.
# ---------------------------------------------------------------------------
def s_star(d, m, kv_dim, ttt_dtype=2, kv_dtype=2):
    return m * d * d * (ttt_dtype / kv_dtype) / kv_dim


# ---------------------------------------------------------------------------
# (b) C* — chunkwise arithmetic-intensity crossover (chunk of C tokens, matrix memory W in R^{dxd}).
# FLOPs(C) = 4 C d^2 (retrieval Q W + compression K^T V) + 4 C^2 d (intra-chunk causal).
# Bytes(C) = 4 d^2 (state RMW) + 8 C d (chunk Q,K,V,O IO), bf16.
# I(C) = C(d+C)/(d+2C).  Solve I(C*) = RIDGE  ->  C^2 + C(d-2R) - R d = 0.
# ---------------------------------------------------------------------------
def chunk_intensity(C, d):
    return C * (d + C) / (d + 2 * C)


def c_star(d, ridge=RIDGE):
    a, b, c = 1.0, (d - 2 * ridge), -ridge * d
    disc = b * b - 4 * a * c
    return (-b + math.sqrt(disc)) / (2 * a)


# ---------------------------------------------------------------------------
# (c) B_max — the batch ceiling from unshared per-sequence state.
#   BW-bound (latency target t): aggregate RMW must fit the budget -> B <= BW*t / rmw_per_token.
#   Capacity-bound: one resident state copy per sequence -> B <= budget / state_per_seq.
# ---------------------------------------------------------------------------
def b_max_bw(d, m, L, t_target_s, dtype_bytes=2):
    return HW["bw_hbm_bps"] * t_target_s / ttt_rmw_per_token(d, m, L, dtype_bytes)


def state_per_seq(d, m, L, dtype_bytes=2):
    return state_bytes_layer(d, m, dtype_bytes) * L   # one resident copy, all layers


def b_max_cap(d, m, L, budget_bytes, dtype_bytes=2):
    return budget_bytes / state_per_seq(d, m, L, dtype_bytes)


# ---------------------------------------------------------------------------
# Config rows: canonical (d, L) per param scale + hypothetical scaling rows.
# kv_dim = Hkv*hd (GQA-8, hd=128 => 1024) held fixed per the plan's validated anchor.
# ---------------------------------------------------------------------------
CONFIGS = [
    # name,        params,  d,    L,   kv_dim(Hkv*hd)
    ("Titans-170M", "170M",  768, 12, 1024),
    ("Titans-340M", "340M", 1024, 24, 1024),
    ("Titans-760M", "760M", 1536, 24, 1024),
    ("neural-mem-1.3B", "1.3B", 2048, 24, 1024),   # <- plan validation anchor (d=2048, L=24)
    ("hypo-7B",      "7B",  4096, 32, 1024),
    ("hypo-70B",     "70B", 8192, 80, 1024),
]

# method -> per-token state multiplier m (fast/high-frequency portion), + note.
METHODS = {
    "Titans":  {"m": 16, "note": "2-layer MLP memory (8 d^2) + momentum/surprise state (8 d^2)"},
    "Miras":   {"m": 8,  "note": "2-layer residual MLP memory (m 8-16 w/ retention state)"},
    "Atlas":   {"m": 16, "note": "deep memory + Omega/Muon 2nd-order preconditioner state"},
    "TNT":     {"m": 8,  "note": "2-layer MLP memory, chunkwise (chunk set {8..128} = C* sweep)"},
    "HOPE":    {"m": 8,  "note": "CMS fastest level RMWs/token; total resident ~6 levels (below)"},
    "Sleep":   {"m": 8,  "note": "HOPE substrate; consolidation OFFLINE (P->inf); +5x64 MLP negligible"},
}

LATENCY_TARGETS_MS = [10.0, 50.0]
DTYPES = {"bf16": 2, "fp8": 1}


def build_results():
    out = {
        "hw": {**HW, "ridge_flop_per_byte": RIDGE},
        "workload": "TTT/neural-memory per-token fast-weight state RMW at decode",
        "crossovers": {},
        "validation": {},
        "sweeps": {},
        "caveats": [],
    }

    # ---- validation: reproduce the plan's hand-calc (d=2048, m=16, L=24) ----
    v = ttt_decode(2048, 16, 24)
    sstar_anchor = s_star(2048, 16, 1024)
    out["validation"] = {
        "config": "d=2048, m=16, L=24, GQA-8 (kv_dim=1024), bf16",
        "state_MB_per_layer": round(v["state_MB_layer"], 1),
        "rmw_GB_per_token": round(v["rmw_GB_token"], 2),
        "rmw_ms_per_token_HBM3": round(v["rmw_ms_token"], 2),
        "S_star_tokens": round(sstar_anchor),
        "plan_expected": {"state_MB_layer": 134, "rmw_GB_token": 6.4,
                          "rmw_ms_token": 1.9, "S_star_tokens": 65000},
        "checks": {
            "state_MB_~134": abs(v["state_MB_layer"] - 134.2) < 1.0,
            "rmw_GB_~6.4": abs(v["rmw_GB_token"] - 6.44) < 0.1,
            "rmw_ms_~1.9": abs(v["rmw_ms_token"] - 1.9) < 0.1,
            "S_star_~65k": abs(sstar_anchor - 65536) < 2000,
        },
    }
    out["validation"]["all_pass"] = all(out["validation"]["checks"].values())

    # ---- main table: method x config crossovers (bf16) ----
    table = []
    for cname, params, d, L, kv_dim in CONFIGS:
        for mname, mspec in METHODS.items():
            m = mspec["m"]
            dec = ttt_decode(d, m, L)
            row = {
                "config": cname, "method": mname, "params": params,
                "d": d, "L": L, "m": m, "kv_dim": kv_dim,
                "state_MB_layer": round(dec["state_MB_layer"], 1),
                "rmw_GB_token": round(dec["rmw_GB_token"], 3),
                "rmw_ms_token": round(dec["rmw_ms_token"], 3),
                "rmw_uJ_token": round(dec["rmw_uJ_token"], 1),
                "compute_ms_token": round(dec["compute_ms_token"], 4),
                "arith_intensity_flopB": round(dec["arith_intensity_flopB"], 2),
                "bound": dec["bound"],
                "S_star_tokens": round(s_star(d, m, kv_dim)),
                "C_star_tokens": round(c_star(d), 1),
                "B_max_bw_10ms": round(b_max_bw(d, m, L, 0.010), 2),
                "B_max_bw_50ms": round(b_max_bw(d, m, L, 0.050), 2),
                "B_max_cap_HBM80GB": round(b_max_cap(d, m, L, HW["hbm_cap_bytes"]), 1),
                "B_max_cap_ondie_L2": round(b_max_cap(d, m, L, HW["ondie_bytes"]), 4),
                "state_per_seq_GB": round(state_per_seq(d, m, L) / 1e9, 3),
            }
            table.append(row)
    out["crossovers"]["table"] = table

    # ---- (a) S* sweep: S* vs (d, m) and dtype-mismatch (fp8 KV vs bf16 TTT) ----
    sstar_sweep = []
    for d in (1024, 2048, 4096, 8192):
        for m in (8, 16):
            sstar_sweep.append({
                "d": d, "m": m, "kv_dim_1024_bf16": round(s_star(d, m, 1024)),
                "kv_dim_1024_fp8KV": round(s_star(d, m, 1024, ttt_dtype=2, kv_dtype=1)),
                "kv_dim_d_MHA_bf16": round(s_star(d, m, d)),
            })
    out["sweeps"]["S_star"] = {
        "formula": "S* = m*d^2*(ttt_dtype/kv_dtype)/kv_dim  [tokens]",
        "note": "GQA-8 kv_dim=1024 fixed; fp8-KV column halves S* (cheaper KV pushes crossover in); "
                "MHA (kv_dim=d) column shows S* collapses to 2m*d for the momentum case.",
        "rows": sstar_sweep,
    }

    # ---- (b) C* sweep: intensity curve + crossover per d ----
    cstar_sweep = []
    for d in (512, 1024, 2048, 4096, 8192):
        cs = c_star(d)
        curve = {C: round(chunk_intensity(C, d), 2) for C in (1, 4, 16, 64, 256, 1024)}
        cstar_sweep.append({
            "d": d, "C_star_tokens": round(cs, 1),
            "I_at_C1_seq": round(chunk_intensity(1, d), 3),
            "intensity_curve_flopB": curve,
        })
    out["sweeps"]["C_star"] = {
        "ridge_flop_per_byte": round(RIDGE, 1),
        "formula": "I(C)=C(d+C)/(d+2C); C* solves I(C*)=ridge -> C^2+C(d-2R)-Rd=0",
        "note": "Sequential (C=1) sits at I~1 flop/B => deep memory-bound plateau. C* rises with "
                "small d (asymptote 2*ridge) and approaches ridge for large d. TNT's chunk set "
                "{8..128} straddles this: small chunks stay memory-bound, C~128 approaches compute.",
        "rows": cstar_sweep,
    }

    # ---- (c) B_max sweep: BW-bound vs capacity-bound, which binds ----
    bmax_sweep = []
    for cname, params, d, L, kv_dim in CONFIGS:
        m = 16  # Titans/Atlas worst case
        bw10 = b_max_bw(d, m, L, 0.010)
        bw50 = b_max_bw(d, m, L, 0.050)
        cap_hbm = b_max_cap(d, m, L, HW["hbm_cap_bytes"])
        cap_ondie = b_max_cap(d, m, L, HW["ondie_bytes"])
        bmax_sweep.append({
            "config": cname, "d": d, "L": L, "m": m,
            "state_per_seq_GB": round(state_per_seq(d, m, L) / 1e9, 3),
            "B_max_bw_10ms": round(bw10, 2),
            "B_max_bw_50ms": round(bw50, 2),
            "B_max_cap_HBM80GB": round(cap_hbm, 1),
            "B_max_cap_ondie_L2_50MB": round(cap_ondie, 4),
            "binding_at_10ms": "BW" if bw10 < cap_hbm else "capacity",
            "ondie_holds_one_seq": cap_ondie >= 1.0,
        })
    out["sweeps"]["B_max"] = {
        "latency_targets_ms": LATENCY_TARGETS_MS,
        "note": "For latency-sensitive decode BW binds far before HBM capacity (opposite of the "
                "KV cache, which is capacity-bound). On-die (L2 50MB) cannot hold even one "
                "sequence's full-model state at these sizes => TTT state cannot be pinned on-die "
                "whole-model; residency must be per-layer/streamed (motivates E1.2).",
        "rows": bmax_sweep,
    }

    # ---- dtype sensitivity on the anchor ----
    out["sweeps"]["dtype_anchor_d2048_m16_L24"] = {
        db: {"rmw_GB_token": round(ttt_decode(2048, 16, 24, dtype_bytes=b)["rmw_GB_token"], 3),
             "rmw_ms_token": round(ttt_decode(2048, 16, 24, dtype_bytes=b)["rmw_ms_token"], 3)}
        for db, b in DTYPES.items()
    }

    out["caveats"] = [
        "hatir/twin numbers are exploration-grade/RELATIVE (ranking + crossover, not "
        "silicon-accurate absolutes); only ratios and crossover positions are promoted to text.",
        "The 6 papers do NOT publish H100 wall-clock decode; absolute us/token is a roofline "
        "lower bound, externally unverifiable, deferred to the in-house A100 runbook (Part III-a).",
        "Per-kernel roofline only: no kernel-launch/scheduling/tail-effect overheads.",
        "KV comparison assumes GQA-8 / head-dim 128 (kv_dim=1024); MHA and other GQA groups shift "
        "S* linearly (see S_star.kv_dim_d_MHA column).",
        "State multipliers m are from published architecture descriptions (2-layer MLP=8d^2, "
        "+momentum=16d^2), not from released decode traces; HOPE m is the fastest-level per-token "
        "portion, total resident state across ~6 CMS levels is larger and mostly low-frequency.",
        "C* uses a matrix-memory (W in R^{dxd}) chunkwise intensity model; a 2-layer MLP memory "
        "scales FLOPs and bytes together, preserving the curve SHAPE but not the exact C* value.",
        "(d,L) per param scale are canonical transformer shapes, not the exact per-paper configs "
        "(most papers report params, not full width/depth); the 1.3B anchor matches the plan.",
    ]
    return out


def fmt(x, w=8, p=2):
    return f"{x:{w}.{p}f}"


def main():
    r = build_results()
    print("=" * 100)
    print("E3.1  ANALYTICAL CROSSOVER MODEL — TTT/neural-memory decode (Part III D4 pair thesis)")
    print("=" * 100)
    hw = r["hw"]
    print(f"HW SoT: {hw['provenance']}")
    print(f"  HBM3 BW {hw['bw_hbm_bps']/1e12:.2f} TB/s | {hw['epb_hbm']:.0f} pJ/B | cap {hw['hbm_cap_bytes']/GB:.0f} GB"
          f" | on-die(L2) {hw['ondie_bytes']/MB:.0f} MB")
    print(f"  peak {hw['peak_flops_s']/1e12:.0f} TFLOP/s | RIDGE = {RIDGE:.1f} FLOP/B\n")

    v = r["validation"]
    print("-- VALIDATION (reproduce plan hand-calc: " + v["config"] + ") " + "-" * 20)
    print(f"   state/layer {v['state_MB_per_layer']} MB (exp 134) | RMW {v['rmw_GB_per_token']} GB/tok (exp 6.4)"
          f" | {v['rmw_ms_per_token_HBM3']} ms/tok (exp 1.9) | S* {v['S_star_tokens']} (exp ~65k)")
    print(f"   ALL CHECKS PASS: {v['all_pass']}   {v['checks']}\n")

    print("-- CROSSOVER TABLE (bf16) — per-token TTT decode + S*/C*/B_max " + "-" * 30)
    hdr = (f"  {'config':16s} {'meth':7s} {'d':>5s} {'L':>3s} {'m':>3s} {'RMW GB':>7s} {'ms/tok':>7s} "
           f"{'uJ/tok':>8s} {'bound':>7s} {'S* tok':>8s} {'C*':>6s} {'Bmax10ms':>9s} {'BmaxHBM':>8s}")
    print(hdr)
    for row in r["crossovers"]["table"]:
        print(f"  {row['config']:16s} {row['method']:7s} {row['d']:5d} {row['L']:3d} {row['m']:3d} "
              f"{row['rmw_GB_token']:7.3f} {row['rmw_ms_token']:7.3f} {row['rmw_uJ_token']:8.1f} "
              f"{row['bound']:>7s} {row['S_star_tokens']:8d} {row['C_star_tokens']:6.1f} "
              f"{row['B_max_bw_10ms']:9.2f} {row['B_max_cap_HBM80GB']:8.1f}")

    print("\n-- (a) S* SWEEP (context where KV read == TTT RMW) " + "-" * 30)
    print(f"  {'d':>5s} {'m':>3s} {'GQA8/bf16':>10s} {'GQA8/fp8KV':>11s} {'MHA/bf16':>9s}")
    for s in r["sweeps"]["S_star"]["rows"]:
        print(f"  {s['d']:5d} {s['m']:3d} {s['kv_dim_1024_bf16']:10d} {s['kv_dim_1024_fp8KV']:11d} "
              f"{s['kv_dim_d_MHA_bf16']:9d}")

    print("\n-- (b) C* SWEEP (chunk where intensity crosses ridge=" + f"{RIDGE:.0f}" + " FLOP/B) " + "-" * 20)
    print(f"  {'d':>5s} {'C*':>7s}   intensity I(C) at C=[1,4,16,64,256,1024]")
    for s in r["sweeps"]["C_star"]["rows"]:
        curve = " ".join(f"{s['intensity_curve_flopB'][c]:7.1f}" for c in (1, 4, 16, 64, 256, 1024))
        print(f"  {s['d']:5d} {s['C_star_tokens']:7.1f}   {curve}")

    print("\n-- (c) B_max SWEEP (m=16; which constraint binds) " + "-" * 30)
    print(f"  {'config':16s} {'state/seq GB':>12s} {'Bmax@10ms':>10s} {'Bmax@50ms':>10s} "
          f"{'BmaxHBM80':>10s} {'ondie/seq':>9s} {'binds@10ms':>11s}")
    for s in r["sweeps"]["B_max"]["rows"]:
        print(f"  {s['config']:16s} {s['state_per_seq_GB']:12.3f} {s['B_max_bw_10ms']:10.2f} "
              f"{s['B_max_bw_50ms']:10.2f} {s['B_max_cap_HBM80GB']:10.1f} "
              f"{s['B_max_cap_ondie_L2_50MB']:9.4f} {s['binding_at_10ms']:>11s}")

    print("\n-- dtype sensitivity (anchor d=2048,m=16,L=24) " + "-" * 30)
    for db, val in r["sweeps"]["dtype_anchor_d2048_m16_L24"].items():
        print(f"   {db:5s}: {val['rmw_GB_token']:6.3f} GB/tok | {val['rmw_ms_token']:5.3f} ms/tok")

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(r, f, indent=2)
    print("\n[written] results.json")
    print("=" * 100)
    assert v["all_pass"], "VALIDATION FAILED — model does not reproduce the plan hand-calc"
    print("PAIR-THESIS TAKEAWAYS:")
    print("  decode side: TTT state RMW is memory-bound at every scale (intensity ~O(m+3) FLOP/B <<"
          f" ridge {RIDGE:.0f}); the transformer's KV read only overtakes it past S*~65k tok (d=2048).")
    print("  prefill/training side: the SAME recurrence turns compute-bound once chunk C>C*~"
          f"{c_star(2048):.0f} (d=2048) — enlarge the chunk and you cross into accelerator territory.")
    print("  serving side: unshared write-heavy state makes decode BW-bound on batch (B_max~"
          f"{b_max_bw(2048,16,24,0.010):.0f}@10ms) long before HBM capacity, and too big to pin on-die.")


if __name__ == "__main__":
    main()
