#!/usr/bin/env python3
"""Cross-architecture Part III — ALL twin-dependent HATIR experiments, run identically per twin.

Reproduces every twin-parameterised experiment of the study paper's Part III on each HW twin:
  E1.1 decode-baseline   — per-token whole-state RMW: traffic / time / energy / bound
  E1.2 state-placement   — backing vs on-die residency; fit; residency crossover d* (THIS twin's hierarchy)
  E1.3 kv-vs-ttt         — S* crossover, B_max, KV-vs-TTT traffic decomposition
  E1.4 frequency-tiers   — update-cadence -> memory-tier admissibility (THIS twin's tiers + cold tiers)
  E1.5 kvmgr-gap         — per-HW unpriced dirty-writeback energy owed
  E3   analytical        — closed-form S* / C* / B_max backbone
  E4   scaling           — state / traffic / S* scaling 170M -> 70B
(E2.1/E2.2 are host-CPU roofline SHAPE measurements — architecture-independent; interpreted per twin's ridge.)

All formulas are verbatim from experiments/{E1.1,E1.2,E1.3,E1.4,E3}. Every hatir tile('w -> w') RMW is the
analytic 2*S traffic (read S + write S back), time=2S/bw, energy=2S*epb — matched to the committed results.
Correctness gate (--validate): H100 twin reproduces the committed E1.1/E1.3/E3 anchor to <0.1%.

Load-bearing outputs = RATIOS / CROSSOVERS / BOUND-CLASS / TIER-ORDER. Absolutes are ideal-roofline lower
bounds; per-twin confidence tier (specs/<key>.json) gates citation. See multiarch/REPORT + book Appendix E.

Usage:
  python3 run_all_experiments.py --all        # every twins/*.json -> results_full/<key>.json + consolidated
  python3 run_all_experiments.py --validate   # H100 == committed anchor gate
"""
import json, math, os, glob, sys, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
GB, MB = 1e9, 1e6
ANCHOR = dict(d=2048, m=16, L=24, kv_dim=1024)
# canonical (name, d, m, L, kv_dim) scale sweep (E3/E1.2 table)
SCALES = [
    ("Titans-170M", 768, 16, 12, 512), ("Titans-340M", 1024, 16, 24, 512),
    ("Titans-760M", 1536, 16, 24, 768), ("neural-mem-1.3B", 2048, 16, 24, 1024),
    ("hypo-7B", 4096, 16, 32, 1024), ("hypo-70B", 8192, 16, 80, 1024),
]
# committed anchor gate (experiments/E3-analytical + E1.1 + E1.3)
GOLD = {"rmw_GB_token": 6.442, "rmw_ms_token": 1.923, "S_star": 65536, "state_MB_layer": 134.2,
        "B_max_10ms": 5.2}
# generic cold tiers for E1.4 (datasheet; same across twins) — hot tiers come from the twin
COLD_TIERS = [("CXL", 0.5e12, 15.0, 512 * GB), ("DDR5", 0.4e12, 12.0, 256 * GB)]


def load_hw(twin_path):
    t = json.load(open(twin_path))

    def find(o, *roles):
        if isinstance(o, dict):
            if o.get("role") in roles or o.get("level_id") in roles:
                return o
            for c in o.get("children", []):
                r = find(c, *roles)
                if r:
                    return r
        return None

    backing = find(t, "hbm") or find(t, "sram") or find(t, "dram")
    comp = find(t, "compute") or find(t, "mxu")
    reuse = None
    for role in ("l2", "infinity_cache", "sram", "shared", "smem", "lds", "vmem"):
        r = find(t, role)
        if r is not None and r is not backing:
            reuse = r
            break
    peak_macs = comp["peak_macs_per_s"] * comp.get("instances", 1)
    epb = backing.get("energy_pj_per_byte")
    epb = epb if epb is not None else 7.0
    key = os.path.splitext(os.path.basename(twin_path))[0]
    sp = os.path.join(HERE, "specs", f"{key}.json")
    tier = json.load(open(sp)).get("final_tier") if os.path.exists(sp) else ("GOLD" if key == "h100" else None)
    return {
        "twin": os.path.basename(twin_path), "key": key, "tier": tier,
        "backing_role": backing.get("role") or backing.get("level_id"),
        "bw": backing["bandwidth_bps"], "epb": epb,
        "epb_wr": backing.get("energy_pj_per_byte_wr", epb),
        "cap": backing["capacity_bytes"],
        "ondie_bw": (reuse or backing).get("bandwidth_bps", backing["bandwidth_bps"]),
        "ondie_epb": (reuse or backing).get("energy_pj_per_byte") or 0.4,
        "ondie_cap": (reuse or backing)["capacity_bytes"],
        "ondie_is_backing": reuse is None,
        "peak_macs": peak_macs, "peak_flops": 2 * peak_macs,
        "matrix_unit": comp.get("matrix_unit"),
    }


# ---- shared formulas (verbatim from E3/E1.x) ----
def state_bytes_layer(d, m, dt=2):
    return m * d * d * dt


def rmw_per_token(d, m, L, dt=2):
    return 2 * state_bytes_layer(d, m, dt) * L


def s_star(d, m, kv_dim, ttt_dt=2, kv_dt=2):
    return m * d * d * (ttt_dt / kv_dt) / kv_dim


def c_star(d, ridge):
    a, b, c = 1.0, (d - 2 * ridge), -ridge * d
    disc = b * b - 4 * a * c
    return (-b + math.sqrt(disc)) / (2 * a) if disc >= 0 else float("nan")


# ===================== per-experiment computations =====================
def E1_1_decode_baseline(hw):
    a = ANCHOR
    traffic = rmw_per_token(a["d"], a["m"], a["L"])
    time_s = traffic / hw["bw"]
    energy_pj = traffic * hw["epb"]
    macs = (3 * a["d"] ** 2 + a["m"] * a["d"] ** 2) * a["L"]
    compute_s = macs / hw["peak_macs"]
    return {
        "rmw_GB_token": round(traffic / 1e9, 4), "rmw_ms_token": round(time_s * 1e3, 4),
        "rmw_uJ_token": round(energy_pj / 1e6, 3), "compute_ms_token": round(compute_s * 1e3, 5),
        "arith_intensity_flopB": round((2 * macs) / traffic, 3),
        "bound": "memory" if time_s > compute_s else "compute",
        "bound_margin_x": round((hw["peak_flops"] / hw["bw"]) / ((2 * macs) / traffic), 2),
    }


def E1_2_state_placement(hw):
    """Residency on THIS twin's own hierarchy: backing vs on-die reuse tier, fit, crossover d*."""
    a = ANCHOR
    S = state_bytes_layer(a["d"], a["m"])          # 134 MB per-layer anchor
    whole = S * a["L"]

    def rmw_cost(bw, epb, macs_per_elem=3.0):
        traffic = 2 * S
        return {"traffic_MB": round(traffic / 1e6, 1), "time_us": round(traffic / bw * 1e6, 3),
                "energy_uJ": round(traffic * epb / 1e6, 3)}

    backing = rmw_cost(hw["bw"], hw["epb"])
    ondie = rmw_cost(hw["ondie_bw"], hw["ondie_epb"])
    fits_1layer = S <= hw["ondie_cap"]
    fits_whole = whole <= hw["ondie_cap"]
    # residency crossover d*: largest d whose per-layer state (m*d^2*2) fits the on-die tier
    d_star = math.sqrt(hw["ondie_cap"] / (a["m"] * 2))
    # energy/time advantage of on-die vs backing (when it fits)
    e_ratio = round(backing["energy_uJ"] / ondie["energy_uJ"], 2) if ondie["energy_uJ"] else None
    t_ratio = round(backing["time_us"] / ondie["time_us"], 2) if ondie["time_us"] else None
    return {
        "state_MB_layer": round(S / 1e6, 1), "whole_model_state_GB": round(whole / 1e9, 3),
        "ondie_cap_MB": round(hw["ondie_cap"] / 1e6, 1), "ondie_is_backing": hw["ondie_is_backing"],
        "backing_rmw": backing, "ondie_rmw": ondie,
        "fits_1layer_ondie": bool(fits_1layer), "fits_whole_ondie": bool(fits_whole),
        "residency_crossover_d_star": round(d_star),
        "ondie_vs_backing_energy_x": e_ratio, "ondie_vs_backing_time_x": t_ratio,
        "placement_verdict": ("whole-model resident on-die" if fits_whole else
                              "per-layer resident on-die" if fits_1layer else
                              "streamed from backing (no on-die residency)"),
    }


def E1_3_kv_vs_ttt(hw):
    a = ANCHOR
    Sst = s_star(a["d"], a["m"], a["kv_dim"])
    b10 = hw["bw"] * 10e-3 / rmw_per_token(a["d"], a["m"], a["L"])
    b50 = hw["bw"] * 50e-3 / rmw_per_token(a["d"], a["m"], a["L"])
    # KV vs TTT decode latency at contexts (KV read grows w/ S; TTT constant)
    kv_lat = {}
    for S in (2048, 8192, 32768):
        kv_bytes = 2 * a["kv_dim"] * S * a["L"]        # whole-model KV read/token
        kv_lat[S] = round(kv_bytes / hw["bw"] * 1e3, 4)
    ttt_lat = round(rmw_per_token(a["d"], a["m"], a["L"]) / hw["bw"] * 1e3, 4)
    return {
        "S_star_read_tokens": round(Sst),
        "S_star_note": "KV read == TTT RMW crossover; pure workload property (HW-invariant)",
        "B_max_10ms": round(b10, 2), "B_max_50ms": round(b50, 2),
        "kv_ms_token_by_ctx": kv_lat, "ttt_ms_token_const": ttt_lat,
        "kv_write_pct_of_read": 0.003, "ttt_write_pct_of_read": 100.0,
    }


def E1_4_frequency_tiers(hw, budget_us=100.0):
    """cadence -> coldest admissible tier. Rule (paper): a level accessed every C tokens contributes
    amortised (2S/bw)/C to each token's critical path; it may drop to the COLDEST tier whose amortised
    latency stays within a per-token budget AND that holds S. Per-token (C=1) thus pins to a HOT tier;
    sub-per-token cadence amortises down to cold pooled tiers (HBM->CXL)."""
    a = ANCHOR
    S = state_bytes_layer(a["d"], a["m"])
    # hottest -> coldest: on-die, backing, DDR5, CXL (CXL pool is the coldest/furthest)
    tiers = [("on-die", hw["ondie_bw"], hw["ondie_cap"]),
             (hw["backing_role"], hw["bw"], hw["cap"]),
             ("DDR5", 0.4e12, 256 * GB), ("CXL", 0.5e12, 512 * GB)]
    admit = {}
    for C in (1, 16, 256, 4096):
        chosen = tiers[0][0]                              # fallback = hottest
        for nm, bw, cap in tiers:                          # hottest -> coldest; keep going colder while OK
            amort_us = (2 * S / C) / bw * 1e6
            if S <= cap and amort_us <= budget_us:
                chosen = nm                                # colder tier still admissible -> take it
        admit[f"every_{C}_tok"] = chosen
    return {"state_MB": round(S / 1e6, 1), "budget_us_per_token": budget_us, "cadence_to_tier": admit,
            "note": "per-token pins hot (only hot meets budget un-amortised); sub-per-token amortises to cold pool"}


def E1_5_kvmgr_gap(hw):
    a = ANCHOR
    S = state_bytes_layer(a["d"], a["m"])
    # unpriced dirty writeback the KV-manager prices as 0: a real TTT eviction owes a full write-back
    owe_energy_uJ = round(S * hw["epb_wr"] / 1e6, 3)
    owe_time_us = round(S / hw["bw"] * 1e6, 3)
    return {"dirty_writeback_owed_uJ": owe_energy_uJ, "dirty_writeback_owed_us": owe_time_us,
            "kv_manager_prices_it_as": 0,
            "note": "KVCacheManager _evict_from() del-drops dirty state free; real TTT eviction owes 1 full writeback"}


def E3_analytical(hw):
    a = ANCHOR
    ridge = hw["peak_flops"] / hw["bw"]
    return {"ridge_flopB": round(ridge, 2), "S_star": round(s_star(a["d"], a["m"], a["kv_dim"])),
            "c_star_chunk": round(c_star(a["d"], ridge), 1),
            "B_max_10ms": round(hw["bw"] * 10e-3 / rmw_per_token(a["d"], a["m"], a["L"]), 2)}


def E4_scaling(hw):
    rows = []
    for name, d, m, L, kv in SCALES:
        traffic = rmw_per_token(d, m, L)
        rows.append({"scale": name, "d": d, "L": L,
                     "rmw_GB_token": round(traffic / 1e9, 4),
                     "rmw_ms_token": round(traffic / hw["bw"] * 1e3, 4),
                     "S_star_tokens": round(s_star(d, m, kv)),
                     "whole_state_GB": round(state_bytes_layer(d, m) * L / 1e9, 3),
                     "fits_ondie": bool(state_bytes_layer(d, m) <= hw["ondie_cap"])})
    return rows


def eval_twin(twin_path):
    hw = load_hw(twin_path)
    return {
        "twin": hw["twin"], "key": hw["key"], "tier": hw["tier"], "backing_role": hw["backing_role"],
        "hw_summary": {"bw_TBps": round(hw["bw"] / 1e12, 4), "peak_PFLOPs": round(hw["peak_flops"] / 1e15, 4),
                       "cap_GB": round(hw["cap"] / 1e9, 2), "ondie_MB": round(hw["ondie_cap"] / 1e6, 2),
                       "epb_pj_B": hw["epb"], "matrix_unit": hw["matrix_unit"]},
        "E1.1_decode_baseline": E1_1_decode_baseline(hw),
        "E1.2_state_placement": E1_2_state_placement(hw),
        "E1.3_kv_vs_ttt": E1_3_kv_vs_ttt(hw),
        "E1.4_frequency_tiers": E1_4_frequency_tiers(hw),
        "E1.5_kvmgr_gap": E1_5_kvmgr_gap(hw),
        "E3_analytical": E3_analytical(hw),
        "E4_scaling": E4_scaling(hw),
    }


def validate():
    r = eval_twin(os.path.join(HERE, "twins", "h100.json"))
    checks = {
        "rmw_GB_token": r["E1.1_decode_baseline"]["rmw_GB_token"],
        "rmw_ms_token": r["E1.1_decode_baseline"]["rmw_ms_token"],
        "state_MB_layer": r["E1.2_state_placement"]["state_MB_layer"],
        "S_star": r["E1.3_kv_vs_ttt"]["S_star_read_tokens"],
        "B_max_10ms": r["E1.3_kv_vs_ttt"]["B_max_10ms"],
    }
    ok = True
    for k, gold in GOLD.items():
        got = checks[k]
        err = abs(got - gold) / gold
        if err >= 0.001:
            ok = False
        print(f"  {k:16} got={got:<10} gold={gold:<10} err={err*100:.3f}%  {'OK' if err<0.001 else 'FAIL'}")
    print("VALIDATE:", "PASS — all experiments reproduce committed H100 anchor" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    if args.validate:
        sys.exit(0 if validate() else 1)

    os.makedirs(os.path.join(HERE, "results_full"), exist_ok=True)
    consolidated = {}
    for tp in sorted(glob.glob(os.path.join(HERE, "twins", "*.json"))):
        try:
            res = eval_twin(tp)
        except Exception as e:
            print(f"  SKIP {os.path.basename(tp)}: {e}")
            continue
        json.dump(res, open(os.path.join(HERE, "results_full", f"{res['key']}.json"), "w"),
                  ensure_ascii=False, indent=1)
        consolidated[res["key"]] = res
    json.dump(consolidated, open(os.path.join(HERE, "results_full", "_consolidated.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"{len(consolidated)} twins x 7 experiments -> multiarch/results_full/*.json")


if __name__ == "__main__":
    main()
