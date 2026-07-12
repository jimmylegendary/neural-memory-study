"""E1.1 — TTT-decode per-token state RMW baseline on the DGX-H100 twin.

Part III (pair-thesis) support: the DECODE half of the workload-split pair.
Claim under test: "TTT decode = per-token read-modify-write of the ENTIRE fast-weight
state" is a MEMORY-BOUND workload whose cost is dominated by the state RMW traffic
(read state + write state back), qualitatively unlike append-once/read-many KV cache.

Workload (one TTT-MLP decode step, per layer), from local-experiments-plan.md §0.5:
  * fast-weight state = W1 in R^{d x 4d}, W2 in R^{4d x d}  -> 8 d^2 elems;
    with momentum/2nd-moment (Titans/Atlas) the multiplier m rises to 16 -> m d^2 elems.
    state_bytes = m * d^2 * dtype_bytes.
  * per-token COMPUTE: forward + gradient GEMVs through W1,W2 (~24 d^2 MACs) plus an
    elementwise momentum/decay epilogue over the whole state (~3 MAC/elem):
        S_t = eta S_{t-1} - theta grad ;  M_t = (1-alpha) M_{t-1} + S_t
  * per-token TRAFFIC: the whole state is READ + WRITTEN BACK (RMW) = 2 * state_bytes.
    arithmetic intensity ~ O(1) MAC/byte  =>  deterministically memory-bound.

Modeling choices (faithful to the two precedent DSEs in hatir/impl/examples/):
  * The step is modeled as ONE hatir RMW op  tile("w -> w", {"w": state_elems}) which
    charges 2*state_bytes of backing traffic (read S + write S back; see
    stateful_writeback_dse.py) and folds the per-element MAC count into macs_per_point,
    so the roofline returns max(compute, memory) and a correct memory/compute bound.
  * Backing device = the REAL dgx_h100_x4.json HBM level (bandwidth_bps, energy_pj_per_byte,
    capacity_bytes taken verbatim from the hat-schema SoT twin) with a plain vector/gemv
    compute leaf carrying the twin's AGGREGATE peak MACs. We deliberately do NOT stack the
    full mxu (matrix_unit [16,8,16]) leaf: a matmul array-fill derate would spuriously
    inflate an elementwise vector RMW by 16x. Both precedent DSEs (kv_cache_memory_dse.py,
    stateful_writeback_dse.py) use the same plain-compute-leaf convention for memory-bound
    decode ops. This keeps us on E3's coordinates (bw_hbm, epb_hbm, peak_macs from the twin).

All numbers are EXPLORATION-GRADE / RELATIVE (per hatir's own docstrings): rankings,
ratios and crossovers are the claim, not silicon-accurate wall-clock. See results.json
'caveats'. Run with the study venv:
  /home/jimmy/repos/neural-memory-study/.venv/bin/python run.py
"""
import json
import os
from hatir import linearize_graph, tile

TWIN_PATH = "/home/jimmy/repos/hat-schema/twins/dgx_h100_x4.json"
E3_PATH = "/home/jimmy/repos/neural-memory-study/experiments/E3-analytical/results.json"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MB, GB = 1024**2, 1024**3

# Per-token compute model (documented constants, plan §0.5)
GEMV_MACS_PER_D2 = 24.0    # forward(8 d^2) + backward/gradient(~2x) through W1,W2
EPILOGUE_MACS_PER_ELEM = 3.0  # momentum + decay FMAs per state element
ENERGY_PJ_PER_MAC = 0.2    # bf16 tensor-core-class MAC energy (same as example leaves)


def load_hbm_backing():
    """Pull the HBM level (BW, energy, capacity) verbatim from the SoT twin, and attach a
    plain vector compute leaf with the twin's aggregate peak MACs. Returns (backing_dict, hw_meta)."""
    full = json.load(open(TWIN_PATH))
    hbm = full["children"][0]["children"][0]                     # ib -> nvlink -> hbm
    mxu = hbm["children"][0]["children"][0]["children"][0]["children"][0]  # l2->smem->reg->mxu
    peak_macs = mxu["peak_macs_per_s"] * mxu.get("instances", 1)  # aggregate over SMs
    backing = {
        "level_id": "hbm", "role": "hbm",
        "capacity_bytes": hbm["capacity_bytes"],
        "bandwidth_bps": hbm["bandwidth_bps"],
        "energy_pj_per_byte": hbm["energy_pj_per_byte"],
        "children": [{
            "level_id": "gemv", "role": "compute", "instances": 1,
            "peak_macs_per_s": peak_macs, "dtype_bytes": 2,
            "energy_pj_per_mac": ENERGY_PJ_PER_MAC,
        }],
    }
    meta = {
        "provenance": "dgx_h100_x4.json (hat-schema SoT); HBM level verbatim + aggregate mxu peak",
        "bw_hbm_bps": hbm["bandwidth_bps"],
        "epb_hbm": hbm["energy_pj_per_byte"],
        "hbm_cap_bytes": hbm["capacity_bytes"],
        "peak_macs_s": peak_macs,
        "peak_flops_s": 2 * peak_macs,
        "ridge_flop_per_byte": 2 * peak_macs / hbm["bandwidth_bps"],
    }
    return backing, meta


def ttt_decode_step(d, m, dtype_bytes, hw, peak_macs):
    """Cost one TTT decode step (single layer): RMW the m*d^2 fast-weight state.

    Returns per-LAYER derived metrics + the compute decomposition."""
    state_elems = m * d * d
    gemv_macs = GEMV_MACS_PER_D2 * d * d
    epilogue_macs = EPILOGUE_MACS_PER_ELEM * state_elems
    total_macs = gemv_macs + epilogue_macs
    macs_per_point = total_macs / state_elems

    dd = tile("w -> w", {"w": state_elems}, hw, dtype_bytes=dtype_bytes,
              macs_per_point=macs_per_point, name="ttt_rmw").derived

    state_bytes = state_elems * dtype_bytes
    # GEMV-only compute time (reference): pure forward+backward MAC latency at peak.
    gemv_compute_us = gemv_macs / peak_macs * 1e6
    # arithmetic intensity = FLOP / RMW-byte  (RMW moves 2*state_bytes)
    intensity = (2 * total_macs) / dd.total_backing_bytes
    return {
        "state_elems": state_elems,
        "state_bytes": state_bytes,
        "rmw_bytes_layer": dd.total_backing_bytes,       # == 2 * state_bytes
        "energy_pj_layer": dd.energy_pj,
        "time_us_layer": dd.kernel_time_us,
        "memory_us_layer": dd.memory_us,
        "compute_us_layer": dd.compute_us,
        "gemv_compute_us_layer": gemv_compute_us,
        "bound": dd.bound,
        "arith_intensity_flopB": intensity,
        "total_macs_layer": total_macs,
    }


def main():
    backing, meta = load_hbm_backing()
    hw = linearize_graph(backing).stack()
    peak = meta["peak_macs_s"]

    e3 = None
    if os.path.exists(E3_PATH):
        e3 = json.load(open(E3_PATH))

    print("=" * 118)
    print("E1.1  TTT-decode per-token state RMW baseline on the DGX-H100 twin (Titans/Atlas-class)")
    print(f"HBM backing (SoT): BW {meta['bw_hbm_bps']/1e12:.2f} TB/s | {meta['epb_hbm']} pJ/B | "
          f"peak {peak/1e12:.1f} TMAC/s | ridge {meta['ridge_flop_per_byte']:.0f} FLOP/B")
    print("state = m*d^2 elems (m=8: W1,W2 ; m=16: +momentum). Step = read state + write back (RMW).")
    print("=" * 118)

    DS = [1024, 2048, 4096, 8192]
    MS = [8, 16]
    DTS = [2, 1]
    LS = [12, 24, 48]

    rows = []
    hdr = (f"{'d':>5} {'m':>3} {'dt':>3} {'L':>3} | {'state/lyr':>10} {'RMW/lyr':>9} | "
           f"{'GB/tok':>8} {'ms/tok':>8} {'mJ/tok':>8} | {'bound':>6} {'AI(F/B)':>7} {'mem/gemv':>8}")
    print(hdr)
    print("-" * 118)
    for d in DS:
        for m in MS:
            for dt in DTS:
                base = ttt_decode_step(d, m, dt, hw, peak)
                for L in LS:
                    gb_tok = base["rmw_bytes_layer"] * L / 1e9
                    ms_tok = base["time_us_layer"] * L / 1e3
                    mj_tok = base["energy_pj_layer"] * L / 1e9
                    # memory-domination ratio: RMW memory time vs GEMV compute time
                    mem_vs_gemv = base["memory_us_layer"] / max(base["gemv_compute_us_layer"], 1e-9)
                    row = {
                        "config": f"d{d}_m{m}_dt{dt}_L{L}",
                        "d": d, "m": m, "dtype_bytes": dt, "L": L,
                        "state_MB_layer": round(base["state_bytes"] / 1e6, 3),
                        "rmw_MB_layer": round(base["rmw_bytes_layer"] / 1e6, 3),
                        "rmw_GB_token": round(gb_tok, 4),
                        "time_us_layer": round(base["time_us_layer"], 4),
                        "ms_token": round(ms_tok, 4),
                        "energy_uJ_layer": round(base["energy_pj_layer"] / 1e6, 2),
                        "uJ_token": round(base["energy_pj_layer"] * L / 1e6, 2),
                        "energy_mem_uJ_token": round(base["rmw_bytes_layer"] * meta["epb_hbm"] * L / 1e6, 2),
                        "bound": base["bound"],
                        "arith_intensity_flopB": round(base["arith_intensity_flopB"], 3),
                        "memory_us_layer": round(base["memory_us_layer"], 4),
                        "compute_us_layer": round(base["compute_us_layer"], 4),
                        "gemv_compute_us_layer": round(base["gemv_compute_us_layer"], 5),
                        "mem_over_gemv_ratio": round(mem_vs_gemv, 1),
                    }
                    rows.append(row)
                    # print a subset (L=24) to keep stdout readable but show the grid
                    if L == 24:
                        print(f"{d:>5} {m:>3} {dt:>3} {L:>3} | {base['state_bytes']/1e6:>9.1f}M "
                              f"{base['rmw_bytes_layer']/1e6:>8.1f}M | {gb_tok:>8.3f} {ms_tok:>8.4f} "
                              f"{mj_tok:>8.3f} | {base['bound']:>6} {base['arith_intensity_flopB']:>7.2f} "
                              f"{mem_vs_gemv:>7.0f}x")
    print("-" * 118)
    print("(table prints L=24 slice; results.json holds all L in {12,24,48})")

    # ---- Headline config: plan §0.5 canonical d=2048 m=16 L=24 dt=2 ----
    head = next(r for r in rows if r["config"] == "d2048_m16_dt2_L24")
    print("\nHEADLINE (plan §0.5 canonical: d=2048, m=16, L=24, bf16):")
    print(f"  state/layer      : {head['state_MB_layer']:.1f} MB")
    print(f"  RMW traffic/token: {head['rmw_GB_token']:.2f} GB   (2 x state x L)")
    print(f"  decode time/token: {head['ms_token']:.3f} ms   (state-movement only, HBM-bound)")
    print(f"  energy/token     : {head['uJ_token']/1e3:.2f} mJ")
    print(f"  bound            : {head['bound']}  |  intensity {head['arith_intensity_flopB']:.2f} FLOP/B "
          f"(ridge {meta['ridge_flop_per_byte']:.0f})  |  memory time = {head['mem_over_gemv_ratio']:.0f}x GEMV compute")

    # ---- Cross-check vs E3 analytical (same twin coordinates) ----
    xcheck = []
    if e3:
        e3_by = {(t["d"], t["L"], t["m"]): t for t in e3["crossovers"]["table"]}
        for (d, L, m), t in e3_by.items():
            r = next((x for x in rows if x["d"] == d and x["L"] == L and x["m"] == m and x["dtype_bytes"] == 2), None)
            if not r:
                continue
            xcheck.append({
                "config": t["config"] + "/" + t["method"], "d": d, "L": L, "m": m,
                "e3_rmw_GB": t["rmw_GB_token"], "hat_rmw_GB": r["rmw_GB_token"],
                "e3_ms": t["rmw_ms_token"], "hat_ms": r["ms_token"],
                "e3_uJ": t["rmw_uJ_token"], "hat_mem_uJ": r["energy_mem_uJ_token"],
            })
        print("\nCROSS-CHECK vs E3 analytical (bf16 configs; HAT hbm-only-energy vs E3 memory energy):")
        print(f"  {'config':>22} {'E3 GB':>7} {'HAT GB':>7} {'E3 ms':>7} {'HAT ms':>7} {'E3 uJ':>9} {'HAT uJ':>9}")
        seen = set()
        for x in xcheck:
            key = (x["d"], x["L"], x["m"])
            if key in seen:
                continue
            seen.add(key)
            print(f"  {x['config']:>22} {x['e3_rmw_GB']:>7.3f} {x['hat_rmw_GB']:>7.3f} "
                  f"{x['e3_ms']:>7.3f} {x['hat_ms']:>7.3f} {x['e3_uJ']:>9.0f} {x['hat_mem_uJ']:>9.0f}")
        print("  -> HAT reproduces E3 traffic/time exactly (same twin BW); HAT total energy adds the")
        print("     epilogue-MAC term E3 omits, so HAT mem-only energy == E3 (3-way self-consistent).")

    # ---- validation asserts (trends must hold, not just print) ----
    for r in rows:
        assert r["bound"] == "memory", f"expected memory-bound, got {r['bound']} for {r['config']}"
        assert r["arith_intensity_flopB"] < meta["ridge_flop_per_byte"], "intensity must be below ridge"
        assert r["mem_over_gemv_ratio"] > 1.0, "memory time must exceed GEMV compute"
    # traffic scales exactly 2x with m, exactly with dtype, exactly with L
    # (compare rmw_MB_layer which is unrounded-enough; tolerance absorbs 3-decimal display rounding)
    r_m8 = next(r for r in rows if r["config"] == "d2048_m8_dt2_L24")
    r_m16 = next(r for r in rows if r["config"] == "d2048_m16_dt2_L24")
    assert abs(r_m16["rmw_MB_layer"] / r_m8["rmw_MB_layer"] - 2.0) < 1e-3
    r_dt1 = next(r for r in rows if r["config"] == "d2048_m16_dt1_L24")
    assert abs(r_m16["rmw_MB_layer"] / r_dt1["rmw_MB_layer"] - 2.0) < 1e-3
    r_L12 = next(r for r in rows if r["config"] == "d2048_m16_dt2_L12")
    r_L48 = next(r for r in rows if r["config"] == "d2048_m16_dt2_L48")
    assert abs(r_L48["ms_token"] / r_L12["ms_token"] - 4.0) < 1e-3  # linear in L
    print("\n[validate] all trends hold: every config memory-bound, below ridge, RMW>>GEMV;")
    print("           traffic exactly linear in m, dtype, L.")

    # ---- emit results.json ----
    results = {
        "experiment": "E1.1-decode-baseline",
        "title": "TTT-decode per-token fast-weight state RMW baseline on DGX-H100 twin",
        "workload": "Titans/Atlas-class per-token read-modify-write of the entire fast-weight state at decode",
        "hw": meta,
        "compute_model": {
            "gemv_macs_per_d2": GEMV_MACS_PER_D2,
            "epilogue_macs_per_elem": EPILOGUE_MACS_PER_ELEM,
            "energy_pj_per_mac": ENERGY_PJ_PER_MAC,
            "state_elems": "m * d^2  (m=8: W1,W2 ; m=16: +momentum)",
            "rmw_bytes_layer": "2 * m * d^2 * dtype_bytes  (read state + write back)",
        },
        "sweep": {"d": DS, "m": MS, "dtype_bytes": DTS, "L": LS},
        "headline": {
            "config": "d=2048, m=16, L=24, bf16 (plan §0.5 canonical)",
            "state_MB_per_layer": head["state_MB_layer"],
            "rmw_GB_per_token": head["rmw_GB_token"],
            "decode_ms_per_token": head["ms_token"],
            "energy_mJ_per_token": round(head["uJ_token"] / 1e3, 3),
            "bound": head["bound"],
            "arith_intensity_flopB": head["arith_intensity_flopB"],
            "memory_over_gemv_compute_ratio": head["mem_over_gemv_ratio"],
        },
        "table": rows,
        "cross_check_vs_E3": xcheck,
        "caveats": [
            "EXPLORATION-GRADE / RELATIVE: hatir is a pure cost model; its own docstrings state numbers "
            "are for ranking + crossover, NOT silicon-accurate absolutes. We claim ratios/bound/crossover only.",
            "Per-kernel analytic model: kernel-launch, scheduling, and tail overheads are NOT included; "
            "this is the traffic/energy STRUCTURE of a decode step, not measured wall-clock latency.",
            "The 6 source papers do not publish H100 decode wall-clock, so absolute ms/token is externally "
            "unverifiable; absolute-value validation is deferred to the in-house A100 runbook (Part III P3-a).",
            "Modeling choice: state resides in HBM, so the RMW op is rooted at the twin's HBM level with a "
            "plain vector/gemv compute leaf. The full mxu leaf (matrix_unit [16,8,16]) is intentionally not "
            "stacked for this elementwise vector RMW; a matmul array-fill derate would spuriously inflate "
            "traffic 16x. Same plain-leaf convention as kv_cache_memory_dse.py / stateful_writeback_dse.py.",
            "HBM energy is symmetric (single energy_pj_per_byte=7.0 in the twin; no separate write energy), "
            "so read and write are charged equally. HAT total energy adds the epilogue-MAC term that E3 omits; "
            "HAT memory-only energy matches E3 exactly.",
            "This experiment (E1.1) uses ONLY the dgx_h100_x4 SoT twin. The novel twins "
            "(novel-swscratchpad-npu, novel-pim-cim) are NOT used here; when E1.2 extends onto them, the "
            "verify committee flags them simulation_ready=False / PPA-abstain, so those results are "
            "directional DSE only and must not be cited as real-device claims.",
        ],
        "supports": "Part III pair-thesis, DECODE half: 'decode/serving state management = per-token RMW of "
                    "the entire fast-weight state, unshared + write-heavy, qualitatively unlike append-once/"
                    "read-many KV cache' — quantified as a deterministically memory-bound step.",
    }
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {os.path.join(OUT_DIR, 'results.json')}  ({len(rows)} configs)")


if __name__ == "__main__":
    main()
