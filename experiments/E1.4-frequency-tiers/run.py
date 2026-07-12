#!/usr/bin/env python3
"""E1.4 — Frequency-tiered placement DSE for the Nested-Learning / Sleep update-frequency continuum.

Part III (D4 workload-split pair thesis), memory-hierarchy half. The neural-memory literature
(Titans -> HOPE/Nested-Learning -> Sleep) exposes not ONE fast-weight state but a CONTINUUM of
states with different UPDATE CADENCES:

    per-token fast weights  ->  chunk-cadenced CMS levels  ->  sleep-cadenced experts  ->  frozen weights
        (update every tok)        (update every ~64 tok)        (consolidated offline)      (never updated)

This DSE directly translates that update-frequency continuum into a MEMORY-HIERARCHY PLACEMENT
SPEC. The governing physics (per the plan, §E1.4):

    A state of size S that is ACCESSED (read and/or read-modify-written) once every P tokens
    contributes an amortised traffic of  ~S / P  bytes per token. Placed on a tier of bandwidth
    BW_tier, it adds  (S / P) / BW_tier  to the per-token critical path. As long as that amortised
    time stays within the per-token budget t_tok (set by the per-token fast level on HBM), the
    state can live on the CHEAPEST / highest-capacity / coldest tier that still meets its deadline.
    Its per-tier LATENCY TOLERANCE is therefore  P * t_tok  — a window that widens linearly with
    the update period. Slow-cadence state tolerates slow, cheap memory; per-token state does not.

The honest subtlety this DSE surfaces (and does NOT paper over): what gates a tier is the
FASTER of a state's READ cadence and WRITE cadence.
  * Fast weights: read AND written every token  -> HBM/SRAM (per-token READ pins it).
  * A CMS level QUERIED every token but UPDATED every 64 tokens is still READ-gated to HBM;
    the slow write only relaxes its ENERGY tier, not the resident copy's placement.
  * A genuinely multi-timescale (down-sampled) CMS level, and sleep-consolidated experts, are
    ACCESSED at their own low cadence -> the whole read-modify-write amortises -> DDR / CXL.
  * Frozen base weights are read every token, written never -> HBM (read-gated, like fast weights,
    but on the *read-heavy / accelerator* side of the pair thesis, not the write-heavy side).
So the memory-centric opportunity is precisely the band of write-heavy RMW states whose ACCESS
cadence is sub-per-token: those are the ones a placement engine can push down the hierarchy.

Method: cost each level's RMW with hatir.tile("w -> w") (2*S traffic, r/w-charged) on a per-tier
device twin, exactly the stateful_writeback_dse.py precedent. HBM + on-die (L2) numbers are pulled
from the dgx_h100_x4.json twin (hat-schema SoT, same source as E3); DDR5 + CXL are the plan's
documented datasheet tiers; the on-die SRAM/scratchpad tier is the NOVEL sw-scratchpad twin
(268 MB @ 60 TB/s) used exploration-grade only.

All numbers exploration-grade / RELATIVE (ranking + crossover + admissibility, not silicon-accurate
absolutes), per notes/local-experiments-plan.md §5. Run with the study venv:
  /home/jimmy/repos/neural-memory-study/.venv/bin/python run.py
"""
import json
import math
import os

from hatir import linearize_graph, tile

HERE = os.path.dirname(os.path.abspath(__file__))
TWIN_PATH = "/home/jimmy/repos/hat-schema/twins/dgx_h100_x4.json"
E3_RESULTS = "/home/jimmy/repos/neural-memory-study/experiments/E3-analytical/results.json"
KB, MB, GB = 1024, 1024**2, 1024**3
INF = math.inf


# ---------------------------------------------------------------------------
# HW SoT — HBM + on-die (L2) numbers from the dgx_h100_x4 twin, like E3.
# DDR5 / CXL are the plan's documented datasheet tiers (memory_device_dse.py numbers).
# The SRAM/scratchpad tier is the NOVEL sw-scratchpad twin (exploration-grade).
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
        return {
            "bw_hbm_bps": hbm["bandwidth_bps"], "epb_hbm": hbm["energy_pj_per_byte"],
            "hbm_cap_bytes": hbm["capacity_bytes"],
            "l2_bw_bps": l2["bandwidth_bps"], "l2_cap_bytes": l2["capacity_bytes"],
            "provenance": prov,
        }
    except Exception as e:  # pragma: no cover — datasheet fallback
        return {"bw_hbm_bps": 3.35e12, "epb_hbm": 7.0, "hbm_cap_bytes": 80 * GB,
                "l2_bw_bps": 1.0e13, "l2_cap_bytes": 50 * MB,
                "provenance": f"H100 datasheet fallback (twin unreadable: {e})"}


HW = load_hw()

# Memory tiers, ordered COLDEST/CHEAPEST -> HOTTEST (the order a placement engine tries).
# (bw_bps, energy pJ/B, capacity, provenance). SRAM = novel sw-scratchpad (268 MB @ 60 TB/s,
# 0.9 pJ/B) — pre-silicon / simulation_ready=False, used exploration-grade.
TIERS = [
    ("CXL",  {"bw": 0.5e12,          "epb": 15.0, "cap": 512 * GB,
              "prov": "plan datasheet (CXL pool 0.5 TB/s 15 pJ/B, +lat)"}),
    ("DDR5", {"bw": 0.4e12,          "epb": 12.0, "cap": 256 * GB,
              "prov": "plan datasheet (DDR5 ~0.4 TB/s 12 pJ/B)"}),
    ("HBM3", {"bw": HW["bw_hbm_bps"], "epb": HW["epb_hbm"], "cap": HW["hbm_cap_bytes"],
              "prov": "dgx_h100_x4 twin (SoT)"}),
    ("SRAM", {"bw": 6.0e13,          "epb": 0.9,  "cap": 268 * MB,
              "prov": "NOVEL novel-swscratchpad-npu twin (pre-silicon, simulation_ready=False)"}),
]
TIER_ORDER = [name for name, _ in TIERS]          # coldest -> hottest
TIER = dict(TIERS)


def _dev(bw, epb, cap):
    """A single-tier backing-memory twin feeding a decode PE, per stateful_writeback_dse._dev."""
    pe = {"level_id": "pe", "role": "compute", "instances": 1, "peak_macs_per_s": 1e13,
          "dtype_bytes": 2, "energy_pj_per_mac": 0.2}
    return {"level_id": "mem", "role": "hbm", "capacity_bytes": cap, "bandwidth_bps": bw,
            "energy_pj_per_byte": epb, "energy_pj_per_byte_wr": epb, "children": [pe]}


def rmw(state_bytes, tier_name, macs_per_elem=1.0):
    """Cost one read-S + update + write-S-back over a tier twin (hatir tile 'w -> w' == 2*S traffic)."""
    spec = TIER[tier_name]
    elems = int(state_bytes) // 2
    hw = linearize_graph(_dev(spec["bw"], spec["epb"], spec["cap"])).stack()
    return tile("w -> w", {"w": elems}, hw, dtype_bytes=2,
                macs_per_point=macs_per_elem, name="update").derived


# ---------------------------------------------------------------------------
# Model + update-frequency continuum. Anchor = neural-mem-1.3B (d=2048, L=24, bf16), matching E3
# and the plan hand-calc. One 2-layer-MLP memory block = 8*d^2 params (m_block=8).
# ---------------------------------------------------------------------------
D, L, DTYPE = 2048, 24, 2
M_BLOCK = 8                                    # one 2-layer MLP memory block = 8 d^2 elems
BLOCK_BYTES = M_BLOCK * D * D * DTYPE          # per layer
BLOCK_ALL = BLOCK_BYTES * L                    # all layers
FROZEN_PARAMS = 1.3e9                          # neural-mem-1.3B base weights
FROZEN_BYTES = int(FROZEN_PARAMS * DTYPE)      # read every token, written never

# Nested-Learning / Sleep continuum. P_read / P_write = cadence (in tokens) of read / write access.
#   blocks = # of memory blocks resident at this level (sizes the state).
LEVELS = [
    dict(name="fast-weights",   P_read=1,      P_write=1,      blocks=1,
         desc="neural-memory fast level: read+RMW every token (Titans core)"),
    dict(name="CMS-chunk",      P_read=1,      P_write=64,     blocks=1,
         desc="HOPE CMS level QUERIED/token, UPDATED/chunk (read-gated)"),
    dict(name="CMS-mid",        P_read=4096,   P_write=4096,   blocks=1,
         desc="multi-timescale down-sampled CMS level (access cadenced)"),
    dict(name="sleep-experts",  P_read=262144, P_write=262144, blocks=3,
         desc="Sleep-consolidated experts: offline / idle-window cadence"),
    dict(name="frozen-weights", P_read=1,      P_write=INF,    blocks=None,
         desc="base model params: read/token, written NEVER (read-heavy side)"),
]


def level_bytes(lv):
    if lv["blocks"] is None:
        return FROZEN_BYTES
    return BLOCK_ALL * lv["blocks"]


def us(seconds):
    return seconds * 1e6


def main():
    print("=" * 112)
    print("E1.4  FREQUENCY-TIERED PLACEMENT DSE — Nested-Learning/Sleep update cadence -> memory hierarchy")
    print("=" * 112)
    print(f"HW SoT: {HW['provenance']}")
    print(f"  HBM3 {HW['bw_hbm_bps']/1e12:.2f} TB/s / {HW['epb_hbm']:.0f} pJ/B / {HW['hbm_cap_bytes']/GB:.0f} GB"
          f" | L2(on-die) {HW['l2_cap_bytes']/MB:.0f} MB @ {HW['l2_bw_bps']/1e12:.0f} TB/s")
    def _capstr(b):
        return f"{b/GB:.0f}GB" if b >= GB else f"{b/MB:.0f}MB"
    print(f"  tiers (cold->hot): " + " -> ".join(
        f"{n}({TIER[n]['bw']/1e12:.2f}TB/s,{TIER[n]['epb']:.0f}pJ/B,{_capstr(TIER[n]['cap'])})"
        for n in TIER_ORDER))
    print(f"  model anchor: d={D}, L={L}, bf16, 1 memory block = {M_BLOCK}*d^2 = {BLOCK_BYTES/1e6:.1f} MB/layer"
          f" -> {BLOCK_ALL/1e9:.2f} GB all-layer\n")

    # ---- t_tok: per-token critical-path budget = fast-weight all-layer RMW on HBM ----
    fast = rmw(BLOCK_ALL, "HBM3")
    t_tok_us = fast.kernel_time_us
    print(f"-- per-token budget t_tok = fast-weight all-layer RMW on HBM3 --")
    print(f"   state {BLOCK_ALL/1e9:.2f} GB | RMW traffic {fast.total_backing_bytes/1e9:.2f} GB"
          f" | t_tok = {t_tok_us:.1f} us/token ({fast.bound})\n")

    # ---- E3 cross-check (state size must match the E3 analytical backbone) ----
    e3_note = "E3 results not found (skipped cross-check)"
    try:
        e3 = json.load(open(E3_RESULTS))
        row = next(r for r in e3["crossovers"]["table"]
                   if r["config"] == "neural-mem-1.3B" and r["method"] == "HOPE")
        e3_state_MB = row["state_MB_layer"]
        ours_MB = BLOCK_BYTES / 1e6
        agree = abs(e3_state_MB - ours_MB) < 1.0
        e3_note = (f"E3 neural-mem-1.3B/HOPE state/layer={e3_state_MB} MB vs ours={ours_MB:.1f} MB "
                   f"-> {'MATCH' if agree else 'MISMATCH'}")
        print(f"-- E3 cross-check: {e3_note}\n")
    except Exception as e:
        print(f"-- E3 cross-check: {e3_note} ({e})\n")

    # ---- build the placement table ----
    # For each level, the resident copy is READ-gated: it must serve its amortised per-token read
    #   (state_bytes / P_read) within t_tok. Latency tolerance on a tier = P_read * t_tok.
    # We recommend the COLDEST/CHEAPEST admissible tier (first in cold->hot order that (a) has
    #   capacity and (b) meets the read deadline). Write cadence sets the amortised write ENERGY.
    print("-- PLACEMENT TABLE (per-level amortised per-token cost by tier; * = READ-admissible & fits) " + "-" * 8)
    hdr = (f"  {'level':15s} {'state':>8s} {'P_read':>8s} {'P_wr':>8s} {'tol(us)':>10s} | "
           + " ".join(f"{n:>11s}" for n in TIER_ORDER) + "  -> place")
    print(hdr)

    table = []
    for lv in LEVELS:
        sb = level_bytes(lv)
        p_read = lv["P_read"]
        # latency tolerance = window (in us) available to serve one full read of the resident copy
        tol_us = (t_tok_us * p_read) if math.isfinite(p_read) else INF
        cells = {}
        admissible = []
        for tname in TIER_ORDER:
            spec = TIER[tname]
            fits = sb <= spec["cap"]
            read_time_us = us(sb / spec["bw"])                      # one full read of the copy
            amort_read_us = (read_time_us / p_read) if math.isfinite(p_read) else 0.0
            # write energy amortised over its own cadence
            d = rmw(sb, tname)
            write_energy_uJ = (d.energy_pj / 2.0) / 1e6             # ~half the RMW energy is the write
            amort_wr_energy_uJ = (write_energy_uJ / lv["P_write"]) if math.isfinite(lv["P_write"]) else 0.0
            ok = fits and (amort_read_us <= t_tok_us + 1e-9)
            cells[tname] = {
                "fits_capacity": fits,
                "read_us_full": round(read_time_us, 3),
                "amort_read_us_per_tok": round(amort_read_us, 4),
                "rmw_us_full": round(d.kernel_time_us, 3),
                "rmw_energy_uJ_full": round(d.energy_pj / 1e6, 1),
                "amort_write_energy_uJ_per_tok": round(amort_wr_energy_uJ, 4),
                "bound": d.bound,
                "read_admissible": ok,
            }
            if ok:
                admissible.append(tname)
        # coldest/cheapest admissible tier = first in cold->hot order
        place = admissible[0] if admissible else "NONE(spill)"
        row = {
            "level": lv["name"], "desc": lv["desc"],
            "state_bytes": sb, "state_GB": round(sb / 1e9, 3),
            "P_read": (None if p_read == INF else p_read),
            "P_write": (None if lv["P_write"] == INF else lv["P_write"]),
            "latency_tolerance_us": (None if tol_us == INF else round(tol_us, 1)),
            "tiers": cells,
            "recommended_tier": place,
            "read_gated": (p_read == 1),
        }
        table.append(row)
        tol_s = "inf" if tol_us == INF else f"{tol_us:.0f}"
        pr_s = "inf" if not math.isfinite(p_read) else str(p_read)
        pw_s = "inf" if not math.isfinite(lv["P_write"]) else str(lv["P_write"])
        cellstrs = []
        for tname in TIER_ORDER:
            c = cells[tname]
            mark = "*" if c["read_admissible"] else (" " if c["fits_capacity"] else "x")
            cellstrs.append(f"{c['amort_read_us_per_tok']:10.3f}{mark}")
        print(f"  {lv['name']:15s} {sb/1e9:6.2f}GB {pr_s:>8s} {pw_s:>8s} {tol_s:>10s} | "
              + " ".join(cellstrs) + f"  -> {place}")
    print("   (cell = amortised per-token READ us on that tier; * READ-admissible & fits,"
          " ' ' fits but too slow, x capacity-exceeded)\n")

    # ---- headline placement summary ----
    print("-- PLACEMENT SUMMARY (update-frequency continuum -> memory hierarchy) " + "-" * 20)
    for row in table:
        gate = "READ every token (per-token-gated)" if row["read_gated"] else \
               f"accessed every {row['P_read']} tok (cadence-gated)"
        print(f"   {row['level']:15s} -> {row['recommended_tier']:5s}  [{gate}]")
    print()

    # ---- the pair-thesis payoff: which band is the memory-centric opportunity ----
    # amortised per-token write ENERGY of the coldest-placed slow levels vs the hot fast level.
    fast_row = next(r for r in table if r["level"] == "fast-weights")
    fast_wr = fast_row["tiers"]["HBM3"]["amort_write_energy_uJ_per_tok"]
    mid_row = next(r for r in table if r["level"] == "CMS-mid")
    mid_place = mid_row["recommended_tier"]
    mid_wr = mid_row["tiers"][mid_place]["amort_write_energy_uJ_per_tok"]
    print("-- PAIR-THESIS READOUT --")
    print(f"   fast weights: read+write EVERY token -> pinned to HBM3, amort write energy"
          f" {fast_wr:.2f} uJ/tok — the hot, write-heavy, unshared RMW state (memory-centric core).")
    print(f"   CMS-mid (cadence {mid_row['P_read']}): read amortises {mid_row['tiers'][mid_place]['amort_read_us_per_tok']:.3f} us/tok"
          f" << t_tok {t_tok_us:.0f} us -> relaxes to {mid_place}"
          f" ({mid_row['tiers'][mid_place]['state_GB'] if False else mid_row['state_GB']:.2f} GB at"
          f" {TIER[mid_place]['epb']:.0f} pJ/B), amort write energy {mid_wr:.3f} uJ/tok.")
    print(f"   frozen weights: read/token, write NEVER -> HBM3 too, but READ-heavy (accelerator side"
          f" of the pair thesis), NOT the write-heavy memory-centric opportunity.")
    print(f"   => the placeable band = write-heavy RMW states with SUB-per-token access cadence;"
          f" the update-frequency continuum IS a memory-hierarchy spec.\n")

    # ---- per-layer fast weights on the NOVEL on-die scratchpad (exploration-grade) ----
    per_layer_fits = BLOCK_BYTES <= TIER["SRAM"]["cap"]
    sram_pl = rmw(BLOCK_BYTES, "SRAM")
    hbm_pl = rmw(BLOCK_BYTES, "HBM3")
    print("-- (novel/exploration-grade) per-LAYER fast weights on the on-die scratchpad --")
    print(f"   1 layer fast weights = {BLOCK_BYTES/1e6:.1f} MB, scratchpad cap {TIER['SRAM']['cap']/1e6:.0f} MB"
          f" -> fits={per_layer_fits}")
    print(f"   per-layer RMW: scratchpad {sram_pl.kernel_time_us:.2f} us / {sram_pl.energy_pj/1e6:.2f} uJ"
          f"  vs HBM {hbm_pl.kernel_time_us:.2f} us / {hbm_pl.energy_pj/1e6:.2f} uJ"
          f"  -> {hbm_pl.energy_pj/sram_pl.energy_pj:.1f}x cheaper energy on-die (per layer, streamed).")
    print(f"   NOTE: all-layer state {BLOCK_ALL/1e9:.2f} GB does NOT fit on-die -> whole-model on-die"
          f" pinning infeasible; residency must be per-layer/streamed (agrees with E3 B_max_ondie).\n")

    # ---- validation: the placement trends must HOLD, not just print ----
    _validate(table, t_tok_us, sram_pl, hbm_pl, per_layer_fits)

    # ---- assemble results.json ----
    results = {
        "experiment": "E1.4-frequency-tiers",
        "title": "Frequency-tiered placement DSE (Nested-Learning/Sleep update cadence -> memory hierarchy)",
        "supports": "Part III D4 pair thesis — 'update-frequency continuum = memory-hierarchy spec' section",
        "hw": {**HW, "e3_cross_check": e3_note},
        "tiers": {n: {"bw_bps": TIER[n]["bw"], "energy_pj_per_byte": TIER[n]["epb"],
                      "capacity_bytes": TIER[n]["cap"], "provenance": TIER[n]["prov"]}
                  for n in TIER_ORDER},
        "tier_order_cold_to_hot": TIER_ORDER,
        "model": {"d": D, "L": L, "dtype_bytes": DTYPE, "m_block": M_BLOCK,
                  "block_bytes_per_layer": BLOCK_BYTES, "block_bytes_all_layer": BLOCK_ALL,
                  "frozen_params": FROZEN_PARAMS, "anchor": "neural-mem-1.3B (matches E3)"},
        "t_tok_us": round(t_tok_us, 3),
        "t_tok_definition": "per-token critical-path budget = fast-weight all-layer RMW time on HBM3",
        "placement_table": table,
        "per_layer_scratchpad": {
            "block_bytes_per_layer": BLOCK_BYTES,
            "scratchpad_cap_bytes": TIER["SRAM"]["cap"],
            "fits_per_layer": per_layer_fits,
            "sram_rmw_us": round(sram_pl.kernel_time_us, 3),
            "sram_rmw_energy_uJ": round(sram_pl.energy_pj / 1e6, 3),
            "hbm_rmw_us": round(hbm_pl.kernel_time_us, 3),
            "hbm_rmw_energy_uJ": round(hbm_pl.energy_pj / 1e6, 3),
            "energy_win_x": round(hbm_pl.energy_pj / sram_pl.energy_pj, 2),
            "all_layer_fits_on_die": BLOCK_ALL <= TIER["SRAM"]["cap"],
        },
        "headline": {
            "fast_weights_tier": next(r["recommended_tier"] for r in table if r["level"] == "fast-weights"),
            "cms_chunk_tier": next(r["recommended_tier"] for r in table if r["level"] == "CMS-chunk"),
            "cms_mid_tier": next(r["recommended_tier"] for r in table if r["level"] == "CMS-mid"),
            "sleep_experts_tier": next(r["recommended_tier"] for r in table if r["level"] == "sleep-experts"),
            "frozen_weights_tier": next(r["recommended_tier"] for r in table if r["level"] == "frozen-weights"),
            "latency_tolerance_widens_with_cadence": "P_read * t_tok (linear in update period)",
            "gating_rule": "resident copy is gated by max(read-cadence, write-cadence) i.e. the FASTER access; "
                           "sub-per-token access cadence is what lets a write-heavy RMW state drop to DDR/CXL",
            "per_layer_scratchpad_energy_win_x": round(hbm_pl.energy_pj / sram_pl.energy_pj, 2),
        },
        "caveats": [
            "hatir/twin numbers are exploration-grade/RELATIVE (ranking + crossover + admissibility, "
            "not silicon-accurate absolutes); only ratios, tier orderings and admissibility flags are "
            "promoted to text (notes/local-experiments-plan.md §5.1).",
            "The SRAM tier is the NOVEL novel-swscratchpad-npu twin (268 MB @ 60 TB/s, 0.9 pJ/B, "
            "provenance=analytical): the hat-schema verify committee marks such pre-silicon designs "
            "simulation_ready=False / PPA-abstain (CATALOG.md). Its scratchpad results are DIRECTIONAL "
            "DSE only, not claims about a real device (§5.2).",
            "DDR5 and CXL tiers are the plan's documented datasheet order-of-magnitude numbers "
            "(0.4 TB/s/12 pJ/B and 0.5 TB/s/15 pJ/B), not from a verified twin; CXL access latency "
            "(~2 us) is NOT added to the amortised model here (traffic/energy structure only).",
            "Per-kernel roofline only: no kernel-launch / scheduling / tail overhead; the amortised "
            "per-token time is a lower bound on the critical-path contribution, not a wall-clock (§5.3).",
            "The READ-cadence of CMS levels is architecture-dependent and NOT published as a decode "
            "trace: we model two honest regimes (queried-every-token => read-gated to HBM; genuinely "
            "down-sampled multi-timescale => access-cadenced => DDR/CXL) rather than assert one. The "
            "6 papers do not publish H100 wall-clock decode, so t_tok is a roofline budget (§5).",
            "Write energy is approximated as ~half the symmetric RMW energy (read pJ/B == write pJ/B on "
            "these tiers); real r/w asymmetry (see stateful_writeback_dse.py) would shift write-energy "
            "figures but not the read-gated placement.",
            "State sizes assume one 2-layer-MLP memory block = 8*d^2 elems per layer (m_block=8), "
            "matching the E3 analytical backbone (cross-checked above); sleep-experts uses 3 resident "
            "blocks as an illustrative low-frequency footprint, not a per-paper measured value.",
        ],
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("[written] results.json")
    print("=" * 112)
    print("TAKEAWAY: the Nested-Learning/Sleep update-frequency continuum maps cleanly onto the memory")
    print("hierarchy — per-token fast weights pin to HBM (or per-layer on-die SRAM), chunk/query-cadenced")
    print("levels stay read-gated to HBM while their SLOW writes amortise, and genuinely low-cadence")
    print("(multi-timescale / sleep-consolidated) state relaxes to DDR/CXL with tolerance P*t_tok. The")
    print("placeable band = write-heavy RMW states with sub-per-token access — the memory-centric core")
    print("of the D4 pair thesis; frozen read-only weights sit on the read-heavy accelerator side.")


def _validate(table, t_tok_us, sram_pl, hbm_pl, per_layer_fits):
    """Trend checks — the placement rankings must hold, not merely print."""
    by = {r["level"]: r for r in table}
    order = {n: i for i, n in enumerate(TIER_ORDER)}   # cold=0 ... hot=3
    # 1) fast weights (read+write/token) are pinned to the HOT end (HBM), not a cold tier.
    assert by["fast-weights"]["recommended_tier"] == "HBM3", by["fast-weights"]["recommended_tier"]
    # 2) frozen weights (read/token) are ALSO read-gated to HBM (read-heavy side of the thesis).
    assert by["frozen-weights"]["recommended_tier"] == "HBM3", by["frozen-weights"]["recommended_tier"]
    # 3) a genuinely cadenced level (CMS-mid) relaxes to a strictly COLDER tier than the fast level.
    assert order[by["CMS-mid"]["recommended_tier"]] < order["HBM3"], by["CMS-mid"]["recommended_tier"]
    # 4) sleep-consolidated experts relax at least as cold as CMS-mid (lower cadence -> colder-or-equal).
    assert order[by["sleep-experts"]["recommended_tier"]] <= order[by["CMS-mid"]["recommended_tier"]]
    # 5) latency tolerance widens monotonically with read period (P_read * t_tok).
    tols = [(r["P_read"], r["latency_tolerance_us"]) for r in table
            if r["P_read"] is not None and r["latency_tolerance_us"] is not None]
    tols.sort(key=lambda x: x[0])
    assert all(tols[i][1] <= tols[i + 1][1] + 1e-6 for i in range(len(tols) - 1)), tols
    # 6) per-token read gating: the fast level's amortised read on a cold tier EXCEEDS t_tok
    #    (that's WHY it can't go cold), while on HBM it's within budget.
    fw = by["fast-weights"]["tiers"]
    assert fw["CXL"]["amort_read_us_per_tok"] > t_tok_us
    assert fw["HBM3"]["amort_read_us_per_tok"] <= t_tok_us + 1e-6
    # 7) on-die scratchpad is a real per-layer energy win but whole-model does NOT fit on-die.
    assert per_layer_fits and sram_pl.energy_pj < hbm_pl.energy_pj
    assert BLOCK_ALL > TIER["SRAM"]["cap"]
    print("[validate] all frequency-tier placement trends hold "
          "(read-gating, cadence->tier relaxation, tolerance monotonicity, on-die per-layer win).\n")


if __name__ == "__main__":
    main()
