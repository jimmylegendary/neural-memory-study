"""E1.2 — State-placement DSE for the TTT/neural-memory decode step (Part III, pair thesis D4).

WHERE do you put the per-token fast-weight state? At decode, a Titans/Atlas/HOPE step READS the whole
per-layer state S, updates it, and WRITES S back (a full read-modify-write, unshared per sequence,
write == read traffic — qualitatively unlike the append-once/read-many KV cache). Unlike the KV cache
S does NOT grow with context: its size is fixed by (d, m), so on-chip RESIDENCY is a *designable*
choice. This DSE sweeps placing S in HBM3 / on-die SRAM (L2-class) / a novel sw-managed scratchpad
(268 MB @ 60 TB/s) / a novel compute-in-memory (PIM) bank, and answers:

  Q1  Rank the memory devices for a fixed per-layer state RMW step.
  Q2  RESIDENCY CROSSOVER: at what state size does the on-chip tier spill to HBM? (capacity knob)
  Q3  PIM's energy-vs-latency trade for the WRITE-HEAVY epilogue, swept over epilogue compute intensity.
  Q4  Cross-check on the ACTUAL novel .hw twins (full hierarchy) — provenance-tagged, exploration-grade.

Direct extension of hatir/impl/examples/stateful_writeback_dse.py (write-heavy state RMW = the direct
prototype of TTT decode) + kv_cache_memory_dse.py (twin loading / output conventions). Device params
are lifted from the real hat-schema twins (dgx-h100-x4, novel-swscratchpad-npu, novel-pim-cim) so the
flat single-tier comparison stays byte-comparable (2*S traffic) across candidates.

All numbers exploration-grade / RELATIVE (rankings + crossovers, not silicon-accurate absolutes). The
two novel twins are provenance=analytical -> the verify committee flags simulation_ready=False; their
results are directional DSE only, never a claim about a shipping device.

Run: /home/jimmy/repos/neural-memory-study/.venv/bin/python run.py
"""
import json
import os

import hatir
from hatir import linearize_graph, tile

# decimal units (matches E3-analytical's convention: state_MB_layer = 2*m*d^2 / 1e6) ------------------
MB, GB = 1e6, 1e9
HERE = os.path.dirname(os.path.abspath(__file__))
TWIN_DIR = "/home/jimmy/repos/hat-schema/examples"


def state_bytes_of(d, m):
    """Per-layer TTT state bytes: W1(d x 4d)+W2(4d x d)=8d^2, +momentum => m*d^2 elems, bf16 (2 B)."""
    return 2 * m * d * d


# ---- device params lifted from the real hat-schema twins (SoT), so a flat single-tier compare stays
#      byte-comparable (each candidate reads S + writes S back = 2*S at its own bw/energy). ------------
#   capacity in real bytes (the twins' capacity_bytes); everything DISPLAYED as /1e6 (decimal MB).
HBM3 = dict(epb=7.0, epb_wr=7.0, bw=3.35e12, cap=80 * GB)                    # dgx-h100-x4 hbm
ONDIE_L2 = dict(epb=0.4, bw=10.0e12, cap=52428800)                          # dgx-h100-x4 L2 (50 MiB)
SCRATCHPAD = dict(epb=0.9, bw=60.0e12, cap=268435456)                       # novel-swscratchpad-npu (256 MiB)
PIM_BANK = dict(epb=3.9, bw=8.2e12, cap=64 * GB, macs=1.2e12, epm=0.6)      # novel-pim-cim pim_bank


def dev_hbm(epb, epb_wr, bw, cap=80 * GB, peak_pe=3.748e13, epm=0.2):
    """Flat backing memory (r/w-asymmetric) feeding a separate fast PE — state crosses the bus."""
    pe = {"level_id": "pe", "role": "compute", "instances": 1, "peak_macs_per_s": peak_pe,
          "dtype_bytes": 2, "energy_pj_per_mac": epm}
    return {"level_id": "dev", "role": "hbm", "capacity_bytes": cap, "bandwidth_bps": bw,
            "energy_pj_per_byte": epb, "energy_pj_per_byte_wr": epb_wr, "children": [pe]}


def dev_sram(epb, bw, cap, peak_pe=3.748e13, epm=0.2):
    """Flat on-die SRAM (symmetric r/w) feeding a PE — the residency candidate that avoids HBM."""
    pe = {"level_id": "pe", "role": "compute", "instances": 1, "peak_macs_per_s": peak_pe,
          "dtype_bytes": 2, "energy_pj_per_mac": epm}
    return {"level_id": "sram", "role": "sram", "capacity_bytes": cap, "bandwidth_bps": bw,
            "energy_pj_per_byte": epb, "children": [pe]}


def dev_pim(epb, bw, cap, macs, epm):
    """Compute-in-memory bank: the update runs WHERE S lives (0 off-die movement); movement stays
    local at the bank's energy, compute at the bank's (slow) scalar MAC lane."""
    return {"level_id": "pim", "role": "hbm", "capacity_bytes": cap, "bandwidth_bps": bw,
            "energy_pj_per_byte": epb, "energy_pj_per_byte_wr": epb,
            "peak_macs_per_s": macs, "dtype_bytes": 2, "energy_pj_per_mac": epm}


def writeback(state_bytes, twin, macs_per_elem=1.0):
    """Cost one read-S + update + write-S-back step (the einsum w -> w == 2*S movement)."""
    elems = state_bytes // 2
    hw = linearize_graph(twin).stack()
    return tile("w -> w", {"w": elems}, hw, dtype_bytes=2, macs_per_point=macs_per_elem,
                name="rmw").derived


def report(tag, d, fits=None):
    fit = "" if fits is None else ("  [fits on-die]" if fits else "  [SPILLS to HBM]")
    print(f"  {tag:34s} traffic {d.total_backing_bytes/1e6:8.1f} MB | energy {d.energy_pj/1e6:9.1f} uJ "
          f"| {d.kernel_time_us:9.1f} us ({d.bound}){fit}")


# ---- E3 configs — the residency crossover is measured against these (per-layer state derived exactly,
#      matching E3-analytical/results.json crossovers.table: 18.9 / 33.6 / 75.5 / 134.2 / 536.9 / 2147.5 MB)
#   (config, d, m, L)
CONFIGS = [
    ("Titans-170M",   768,  16, 12),
    ("Titans-340M",   1024, 16, 24),
    ("Titans-760M",   1536, 16, 24),
    ("neural-mem-1.3B", 2048, 16, 24),
    ("hypo-7B",       4096, 16, 32),
    ("hypo-70B",      8192, 16, 80),
]
EPILOGUE_MACS_PER_ELEM = 3      # S=eta*S - theta*grad ; M=(1-a)M + S  -> a few MACs/elem (plan 0.5)

results = {"experiment": "E1.2-state-placement",
           "workload": "TTT/neural-memory per-token fast-weight state RMW at decode",
           "twin_provenance": {
               "HBM3/on-die": "dgx-h100-x4.hw (hat-schema SoT)",
               "scratchpad": "novel-swscratchpad-npu.hw (provenance=analytical, simulation_ready=False)",
               "PIM": "novel-pim-cim.hw (provenance=analytical, simulation_ready=False)"}}


def main():
    print("=" * 116)
    print("E1.2 STATE-PLACEMENT DSE — where to put the per-token fast-weight state at TTT decode.")
    print("A decode step reads S + writes S back (RMW). S is FIXED-size (independent of context) => on-chip")
    print("residency is a designable choice, unlike the context-growing KV cache.\n")

    # ================= Q1: device ranking for a fixed per-layer state RMW =================
    S = state_bytes_of(2048, 16)     # neural-mem-1.3B per-layer state (134 MB) — the anchor from E3
    print(f"Q1) Rank memory devices for a {S/1e6:.0f} MB per-layer state RMW step "
          f"(epilogue {EPILOGUE_MACS_PER_ELEM} MAC/elem):")
    cands = {
        "HBM3 3.35TB/s 7pJ (baseline)": (dev_hbm(**{k: HBM3[k] for k in ("epb", "epb_wr", "bw", "cap")}),
                                         HBM3["cap"]),
        "on-die SRAM 10TB/s 0.4pJ 50MB": (dev_sram(ONDIE_L2["epb"], ONDIE_L2["bw"], ONDIE_L2["cap"]),
                                          ONDIE_L2["cap"]),
        "sw-scratchpad 60TB/s 0.9pJ 256MB": (dev_sram(SCRATCHPAD["epb"], SCRATCHPAD["bw"], SCRATCHPAD["cap"]),
                                             SCRATCHPAD["cap"]),
        "PIM in-bank 8.2TB/s 3.9pJ": (dev_pim(PIM_BANK["epb"], PIM_BANK["bw"], PIM_BANK["cap"],
                                              PIM_BANK["macs"], PIM_BANK["epm"]), PIM_BANK["cap"]),
    }
    q1 = {}
    for name, (twin, cap) in cands.items():
        d = writeback(S, twin, macs_per_elem=EPILOGUE_MACS_PER_ELEM)
        fits = S <= cap
        report(name, d, fits)
        q1[name] = {"traffic_MB": round(d.total_backing_bytes / 1e6, 1),
                    "energy_uJ": round(d.energy_pj / 1e6, 2),
                    "time_us": round(d.kernel_time_us, 2), "bound": d.bound,
                    "cap_MB": round(cap / 1e6, 1), "fits_one_layer": fits}
    best_e = min(q1, key=lambda k: q1[k]["energy_uJ"])
    best_t = min(q1, key=lambda k: q1[k]["time_us"])
    print(f"     -> lowest ENERGY: {best_e}  |  lowest TIME: {best_t}")
    print("     -> NOTE: the 50 MB L2 posts the lowest roofline energy but a 134 MB state SPILLS out of it")
    print("        (the flat-twin cost path does not enforce residency — that is Q2's job); the REALIZABLE")
    print("        winner that actually holds S is the 268 MB scratchpad (both fits AND lowest time).")
    print("     -> on-chip residency wins energy AND time by ~1 order of magnitude over HBM3 when S fits.\n")
    results["Q1_device_ranking"] = {"state_MB": S / 1e6, "epilogue_macs_per_elem": EPILOGUE_MACS_PER_ELEM,
                                    "devices": q1, "lowest_energy": best_e, "lowest_time": best_t}

    # ================= Q2: residency crossover — capacity is the knob =================
    print("Q2) RESIDENCY CROSSOVER: sweep on-chip capacity vs per-layer state; when does S spill to HBM?")
    caps = [ONDIE_L2["cap"], 64 * MB, 128 * MB, SCRATCHPAD["cap"], 512 * MB, 1 * GB]
    print(f"     Per-layer state that FITS on-chip (crossover = capacity). '#L fit' = layers that co-reside.")
    header = "     {:16s}".format("cap \\ config") + "".join(f"{c[0][:11]:>13s}" for c in CONFIGS)
    print(header)
    q2_grid = []
    for cap in caps:
        row = {"cap_MB": round(cap / 1e6, 1), "cells": {}}
        cells = []
        for (cfg, d_, m, L) in CONFIGS:
            s_bytes = state_bytes_of(d_, m)
            one_fits = s_bytes <= cap
            n_fit = int(cap // s_bytes)
            cells.append(f"{('Y' if one_fits else 'n')}/{n_fit}L".rjust(13))
            row["cells"][cfg] = {"one_layer_fits": one_fits, "layers_that_fit": n_fit}
        print(f"     {cap/1e6:6.0f} MB      " + "".join(cells))
        q2_grid.append(row)
    # residency crossover state size for each cap = cap itself (S<=cap fits). Report the model boundary.
    print("     -> crossover state = the capacity itself. For the 268 MB scratchpad, single-layer state")
    print("        fits up to neural-mem-1.3B (134 MB, d=2048) and SPILLS by hypo-7B (537 MB, d=4096):")
    # find the config crossover for the 256MB scratchpad
    scr_cap = SCRATCHPAD["cap"]
    last_fit, first_spill = None, None
    for (cfg, d_, m, L) in CONFIGS:
        s_mb = state_bytes_of(d_, m) / 1e6
        if state_bytes_of(d_, m) <= scr_cap:
            last_fit = (cfg, round(s_mb, 1), d_)
        elif first_spill is None:
            first_spill = (cfg, round(s_mb, 1), d_)
    # closed-form d at which one layer == scratchpad cap (m=16, bf16): 2*m*d^2 = cap
    import math
    d_cross = math.sqrt(scr_cap / (2 * 16))
    print(f"        crossover width d* ~= {d_cross:.0f} (m=16, bf16) : d<=2048 resident, d>=4096 spills.")
    results["Q2_residency_crossover"] = {
        "scratchpad_cap_MB": scr_cap / 1e6, "grid": q2_grid,
        "last_config_that_fits_one_layer": {"config": last_fit[0], "state_MB": last_fit[1], "d": last_fit[2]},
        "first_config_that_spills": {"config": first_spill[0], "state_MB": first_spill[1], "d": first_spill[2]},
        "crossover_width_d_star": round(d_cross, 0),
        "note": "on-die 50MB L2 holds no single layer past Titans-340M; 256MB scratchpad holds up to 1.3B."}

    # ---- Q2b: hatir.residency.live_set_spill_bytes over a 2-TOKEN decode window. Decode cycles through
    #      ALL L layers every token, so each layer-state Si is re-used token-to-token => to stay resident
    #      the whole L-layer set must co-reside; overflow spills (writeback+refetch) to HBM per token.
    print("\n     Q2b) whole-model resident set over a 2-token decode window (each layer-state re-used every")
    print("          token) spill to HBM, via hatir.residency.live_set_spill_bytes (scratchpad = 268 MB):")
    q2b = {}
    for (cfg, d_, m, L) in CONFIGS:
        s_bytes = state_bytes_of(d_, m)
        whole_MB = L * s_bytes / 1e6
        # 2 tokens x L layers: Si read+written at token0 node i AND token1 node (L+i) => Si stays live
        # across the whole first token, so peak working set = sum over L layers (the resident requirement).
        nodes = ([{"id": i, "reads": [f"S{i}"], "writes": [f"S{i}"]} for i in range(L)]
                 + [{"id": L + i, "reads": [f"S{i}"], "writes": [f"S{i}"]} for i in range(L)])
        tbytes = {f"S{i}": s_bytes for i in range(L)}
        spill, peak = hatir.live_set_spill_bytes(nodes, tbytes, scr_cap,
                                                 transient={f"S{i}" for i in range(L)}, policy="largest")
        print(f"        {cfg:16s} L={L:2d}  whole-model {whole_MB:9.1f} MB  peak-live {peak/1e6:9.1f} MB  "
              f"spill/tok {spill/2/1e6:9.1f} MB")
        q2b[cfg] = {"L": L, "whole_model_state_MB": round(whole_MB, 1),
                    "peak_live_MB": round(peak / 1e6, 1),
                    "spill_MB_per_token": round(spill / 2 / 1e6, 1),
                    "whole_model_fits_268MB": whole_MB * 1e6 <= scr_cap}
    print("     -> only the smallest config (Titans-170M, 226 MB whole-model) fits the 268 MB scratchpad;")
    print("        every larger config's L-layer state overruns it and the residency module reports a per-")
    print("        token writeback+refetch spill to HBM => at scale, state placement is per-layer / streamed.")
    results["Q2b_whole_model_spill"] = {"scratchpad_cap_MB": scr_cap / 1e6, "policy": "largest",
                                        "window_tokens": 2, "per_config": q2b,
                                        "caveat": "residency.py excludes PERSISTENT tensors from the "
                                        "transient set; TTT state is a 3rd class (persistent BUT mutated "
                                        "every token) so we tag it activation-class/transient to force the "
                                        "spill sim (plan limit #4); eviction-policy precision not claimed."}

    # ================= Q3: PIM energy-vs-latency trade for the write-heavy epilogue =================
    print("\nQ3) PIM (in-bank) vs HBM+PE over EPILOGUE COMPUTE INTENSITY (MAC/elem), fixed 134 MB state:")
    print(f"     {'MAC/elem':>9} | {'HBM uJ':>9} {'PIM uJ':>9} {'E win':>6} | {'HBM us':>9} {'PIM us':>9} {'T cost':>7}")
    hbm_twin = dev_hbm(**{k: HBM3[k] for k in ("epb", "epb_wr", "bw", "cap")})
    pim_twin = dev_pim(PIM_BANK["epb"], PIM_BANK["bw"], PIM_BANK["cap"], PIM_BANK["macs"], PIM_BANK["epm"])
    q3 = []
    pim_time_crossover = None
    for mpe in (1, 3, 8, 32, 128, 512, 1024):
        hbm = writeback(state_bytes_of(2048, 16), hbm_twin, macs_per_elem=mpe)
        pim = writeback(state_bytes_of(2048, 16), pim_twin, macs_per_elem=mpe)
        ew = hbm.energy_pj / pim.energy_pj
        tc = pim.kernel_time_us / hbm.kernel_time_us
        marker = " <- TTT epilogue" if mpe == EPILOGUE_MACS_PER_ELEM else ""
        print(f"     {mpe:9d} | {hbm.energy_pj/1e6:9.0f} {pim.energy_pj/1e6:9.0f} {ew:5.1f}x | "
              f"{hbm.kernel_time_us:9.0f} {pim.kernel_time_us:9.0f} {tc:6.1f}x{marker}")
        q3.append({"macs_per_elem": mpe, "hbm_uJ": round(hbm.energy_pj / 1e6, 1),
                   "pim_uJ": round(pim.energy_pj / 1e6, 1), "energy_win_x": round(ew, 2),
                   "hbm_us": round(hbm.kernel_time_us, 1), "pim_us": round(pim.kernel_time_us, 1),
                   "time_cost_x": round(tc, 2), "hbm_bound": hbm.bound, "pim_bound": pim.bound})
        if pim_time_crossover is None and tc > 1.0:
            pim_time_crossover = mpe
    print(f"     -> movement-bound (low MAC/elem, incl the ~{EPILOGUE_MACS_PER_ELEM}-MAC/elem TTT epilogue): PIM is a")
    print("        near-pure ENERGY win (cheaper local movement, no bus). Compute-bound (high MAC/elem): PIM's")
    print(f"        slow scalar lane (1.2 vs 37 TMAC/s) costs latency; time crossover ~ {pim_time_crossover} MAC/elem.")
    results["Q3_pim_energy_latency"] = {"state_MB": round(state_bytes_of(2048, 16) / 1e6, 1), "ttt_epilogue_macs_per_elem": EPILOGUE_MACS_PER_ELEM,
                                        "sweep": q3, "pim_time_crossover_macs_per_elem": pim_time_crossover}

    # ================= Q4: cross-check on the ACTUAL novel twins (full hierarchy) =================
    print("\nQ4) Cross-check on the REAL novel .hw twins (full memory hierarchy loaded from hat-schema):")
    q4 = {}
    S4 = state_bytes_of(2048, 16)
    for label, path, engine in [("novel-swscratchpad-npu", f"{TWIN_DIR}/novel-swscratchpad-npu.hw", None),
                                ("novel-pim-cim", f"{TWIN_DIR}/novel-pim-cim.hw", "mac_lane"),
                                ("dgx-h100-x4 (ref)", f"{TWIN_DIR}/dgx-h100-x4.hw", None)]:
        try:
            lg = hatir.load_twin(path)
            hw = lg.stack(engine=engine) if engine else lg.stack()
            d = tile("w -> w", {"w": S4 // 2}, hw, dtype_bytes=2,
                     macs_per_point=EPILOGUE_MACS_PER_ELEM, name="rmw").derived
            print(f"     {label:26s} traffic {d.total_backing_bytes/1e6:9.1f} MB | "
                  f"energy {d.energy_pj/1e6:9.1f} uJ | {d.kernel_time_us:9.1f} us ({d.bound})")
            q4[label] = {"traffic_MB": round(d.total_backing_bytes / 1e6, 1),
                         "energy_uJ": round(d.energy_pj / 1e6, 2), "time_us": round(d.kernel_time_us, 2),
                         "bound": d.bound, "engine": engine}
        except Exception as e:  # noqa: BLE001
            print(f"     {label:26s} LOAD/COST FAILED: {e}")
            q4[label] = {"error": str(e)}
    print("     -> full-hierarchy traffic counts movement at EVERY tier (hbm->scratchpad->reg), so it is")
    print("        NOT byte-comparable to the flat single-tier Q1 numbers; reported as a directional check.")
    print("     -> dgx-h100-x4 shows 0 uJ because that twin omits per-tier energy_pj_per_byte (structural")
    print("        check is traffic/time); both novel twins are provenance=analytical => sim_ready=False.")
    results["Q4_actual_novel_twins"] = {
        "note": "full-hierarchy load; traffic sums all tiers (not comparable to flat Q1); directional only. "
                "dgx-h100-x4 reports 0 uJ because that twin omits per-tier energy_pj_per_byte annotations.",
        "twins": q4,
        "simulation_ready": False,
        "provenance": "analytical (novel-swscratchpad-npu, novel-pim-cim) — exploration-grade, needs silicon."}

    print("=" * 116)
    print("Decision surface (pre-silicon): the TTT/neural-memory state is FIXED-size, so on-chip residency")
    print("is a real design knob — but a 268 MB scratchpad holds only ~2 layers of a 1.3B state and no")
    print("whole-model state past the smallest (170M) config, so placement is per-layer/streamed. PIM buys")
    print("movement energy at the cost of compute latency (fine at TTT's ~3 MAC/elem). This is the memory-")
    print("centric half of the pair thesis: decode state management is a placement/residency problem.")

    _headline_and_caveats()
    _validate()

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[wrote] {os.path.join(HERE, 'results.json')}")


def _headline_and_caveats():
    q1 = results["Q1_device_ranking"]["devices"]
    hbm_e = q1["HBM3 3.35TB/s 7pJ (baseline)"]["energy_uJ"]
    scr_e = q1["sw-scratchpad 60TB/s 0.9pJ 256MB"]["energy_uJ"]
    hbm_t = q1["HBM3 3.35TB/s 7pJ (baseline)"]["time_us"]
    scr_t = q1["sw-scratchpad 60TB/s 0.9pJ 256MB"]["time_us"]
    q3 = results["Q3_pim_energy_latency"]
    ttt_row = next(r for r in q3["sweep"] if r["macs_per_elem"] == EPILOGUE_MACS_PER_ELEM)
    results["headline_numbers"] = [
        f"On-chip scratchpad (268MB/256MiB @60TB/s, 0.9pJ/B) vs HBM3 for a 134MB per-layer state RMW: "
        f"{hbm_e/scr_e:.1f}x lower energy, {hbm_t/scr_t:.1f}x lower time — WHEN the state fits on-chip.",
        f"Residency crossover: a 268MB scratchpad holds a SINGLE layer's state up to d=2048 "
        f"(neural-mem-1.3B, 134MB) and SPILLS to HBM by d=4096 (hypo-7B, 537MB); crossover width "
        f"d*~={results['Q2_residency_crossover']['crossover_width_d_star']:.0f} (m=16, bf16).",
        f"Whole-model residency: only Titans-170M (L=12, 226MB total state) fits the 268MB scratchpad; "
        f"every larger config (>=340M) overruns it and spills per token => at scale state placement is "
        f"per-layer / streamed, not pinned whole-model.",
        f"PIM for the write-heavy epilogue: at TTT's ~{EPILOGUE_MACS_PER_ELEM} MAC/elem it is a "
        f"{ttt_row['energy_win_x']:.1f}x energy win at {ttt_row['time_cost_x']:.2f}x time; the latency cost "
        f"only bites past ~{q3['pim_time_crossover_macs_per_elem']} MAC/elem (compute-bound epilogue).",
    ]
    results["caveats"] = [
        "hatir/twin numbers are exploration-grade/RELATIVE (rankings + crossover positions, not "
        "silicon-accurate absolutes); only ratios and crossover positions are promoted to text.",
        "The two novel twins (novel-swscratchpad-npu, novel-pim-cim) are provenance=analytical => the "
        "verify committee flags simulation_ready=False / PPA abstain; PIM & scratchpad results are "
        "DIRECTIONAL DSE only, never a claim about a shipping device.",
        "Q1/Q2/Q3 use FLAT single-tier device twins (params lifted from the real twins) so traffic is "
        "byte-comparable (2*S). Q4 loads the FULL-hierarchy novel twins whose traffic sums every tier "
        "(hbm->scratchpad->reg) and is therefore NOT comparable to Q1 — reported as a directional check.",
        "residency.py excludes persistent tensors from the transient working set; TTT state is a 3rd "
        "class (persistent BUT mutated every token), so Q2b tags it activation-class/transient to force "
        "the spill sim — eviction-policy-level precision is not claimed (plan limit #4).",
        "Per-kernel roofline only: no kernel-launch / scheduling / tail overheads; per-token ABSOLUTE "
        "latency is a lower bound, deferred to the in-house A100 runbook (Part III-a).",
        "Per-layer state sizes (m=16, bf16) are canonical transformer shapes from E3-analytical, not "
        "released decode traces; PIM epilogue intensity (~3 MAC/elem) is from the published update rule.",
    ]


def _validate():
    """Trend checks — the DSE's rankings/crossovers must HOLD, not just print."""
    S = state_bytes_of(2048, 16)
    hbm = writeback(S, dev_hbm(7.0, 7.0, 3.35e12), macs_per_elem=EPILOGUE_MACS_PER_ELEM)
    scr = writeback(S, dev_sram(SCRATCHPAD["epb"], SCRATCHPAD["bw"], SCRATCHPAD["cap"]),
                    macs_per_elem=EPILOGUE_MACS_PER_ELEM)
    l2 = writeback(S, dev_sram(ONDIE_L2["epb"], ONDIE_L2["bw"], ONDIE_L2["cap"]),
                   macs_per_elem=EPILOGUE_MACS_PER_ELEM)
    # on-chip SRAM strictly cheaper in energy AND time than HBM for the RMW step
    assert scr.energy_pj < hbm.energy_pj and scr.kernel_time_us < hbm.kernel_time_us
    assert l2.energy_pj < hbm.energy_pj
    # residency crossover: 134MB fits the 256MB scratchpad, not the 50MB L2
    assert state_bytes_of(2048, 16) <= SCRATCHPAD["cap"] and state_bytes_of(2048, 16) > ONDIE_L2["cap"]
    # PIM: movement-bound epilogue is an energy win; compute-bound epilogue costs time
    pim_lo = writeback(S, dev_pim(**{"epb": PIM_BANK["epb"], "bw": PIM_BANK["bw"], "cap": PIM_BANK["cap"],
                                     "macs": PIM_BANK["macs"], "epm": PIM_BANK["epm"]}), macs_per_elem=3)
    hbm_lo = writeback(S, dev_hbm(7.0, 7.0, 3.35e12), macs_per_elem=3)
    pim_hi = writeback(S, dev_pim(**{"epb": PIM_BANK["epb"], "bw": PIM_BANK["bw"], "cap": PIM_BANK["cap"],
                                     "macs": PIM_BANK["macs"], "epm": PIM_BANK["epm"]}), macs_per_elem=1024)
    hbm_hi = writeback(S, dev_hbm(7.0, 7.0, 3.35e12), macs_per_elem=1024)
    assert pim_lo.energy_pj < hbm_lo.energy_pj                    # energy win at low intensity
    assert pim_hi.kernel_time_us > hbm_hi.kernel_time_us          # latency cost at high intensity
    print("[validate] all state-placement DSE trends hold (SRAM residency win, crossover, PIM energy/latency).")


if __name__ == "__main__":
    main()
