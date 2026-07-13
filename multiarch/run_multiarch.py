#!/usr/bin/env python3
"""Cross-architecture Part III cost model — the SAME closed-form experiment (E3 backbone),
parameterised by an arbitrary HW twin, run identically across every architecture.

The load-bearing outputs are RATIOS / CROSSOVERS / BOUND-CLASS (the pair thesis), never
silicon-accurate absolutes (see multiarch/AUTHORING-SPEC.md + book Appendix E). Absolutes are
ideal-roofline lower bounds; per-twin confidence tier gates how they may be cited.

This driver replicates experiments/E3-analytical/run.py's formulas VERBATIM but reads HW
constants from any twin (HBM GPUs and SRAM-only accelerators alike). Correctness gate:
`--validate` reruns the H100 twin and asserts it reproduces the committed E3 anchor to <0.1%.

Usage:
  python3 run_multiarch.py --twin twins/h100.json            # one twin -> results/<key>.json
  python3 run_multiarch.py --all                             # every twins/*.json + REPORT table
  python3 run_multiarch.py --validate                        # H100 == committed E3 anchor gate
"""
import json, math, os, glob, sys, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
GB, MB = 1e9, 1e6

# committed E3 anchor (experiments/E3-analytical/results.json) — the reproduction gate
H100_GOLD = {"rmw_GB_token": 6.442, "rmw_ms_token": 1.923, "S_star_read_tokens": 65536,
             "state_MB_layer": 134.2}
ANCHOR = dict(d=2048, m=16, L=24, kv_dim=1024)      # neural-mem-1.3B, GQA-8, bf16
# scale sweep (d, L, m) — canonical per param scale (E3 table)
SCALES = [
    ("Titans-170M", dict(d=1024, L=12, m=16, kv_dim=512)),
    ("340M",        dict(d=1024, L=24, m=16, kv_dim=512)),
    ("1.3B-anchor", dict(d=2048, L=24, m=16, kv_dim=1024)),
    ("7B",          dict(d=4096, L=32, m=16, kv_dim=1024)),
    ("70B",         dict(d=8192, L=80, m=16, kv_dim=1024)),
]


def load_hw(twin_path):
    """Twin-structure-agnostic constant extractor. Works on the deep H100 twin, flat
    single-accelerator twins, and SRAM-only accelerators (Groq/Cerebras: no HBM)."""
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

    backing = find(t, "hbm") or find(t, "sram") or find(t, "dram")   # HBM else on-chip SRAM
    comp = find(t, "compute") or find(t, "mxu")
    if backing is None or comp is None:
        raise ValueError(f"{twin_path}: missing backing memory or compute leaf")
    # reuse tier = an on-chip level distinct from backing (L2/SRAM/shared); else backing itself
    reuse = None
    for role in ("l2", "sram", "shared", "smem"):
        r = find(t, role)
        if r is not None and r is not backing:
            reuse = r
            break
    peak_macs = comp["peak_macs_per_s"] * comp.get("instances", 1)
    epb = backing.get("energy_pj_per_byte")
    return {
        "twin": os.path.basename(twin_path),
        "backing_role": backing.get("role") or backing.get("level_id"),
        "bw_bps": backing["bandwidth_bps"],
        "epb": epb if epb is not None else float("nan"),
        "cap_bytes": backing["capacity_bytes"],
        "ondie_bytes": reuse["capacity_bytes"] if reuse else backing["capacity_bytes"],
        "ondie_is_backing": reuse is None,          # True for SRAM-only (whole memory is on-die)
        "peak_macs_s": peak_macs,
        "peak_flops_s": 2 * peak_macs,
        "matrix_unit": comp.get("matrix_unit"),
        "tier": t.get("confidence") or (t.get("meta") or {}).get("confidence"),
    }


# ---- E3 formulas, verbatim, parameterised by hw ----
def state_bytes_layer(d, m, dt=2):
    return m * d * d * dt


def ttt_rmw_per_token(d, m, L, dt=2):
    return 2 * state_bytes_layer(d, m, dt) * L


def ttt_decode(hw, d, m, L, dt=2, batch=1):
    traffic = ttt_rmw_per_token(d, m, L, dt) * batch
    time_s = traffic / hw["bw_bps"]
    energy_pj = traffic * hw["epb"]
    macs = (3 * d * d + m * d * d) * L * batch
    compute_s = macs / hw["peak_macs_s"]
    return {
        "state_MB_layer": round(state_bytes_layer(d, m, dt) / 1e6, 3),
        "rmw_GB_token": round(traffic / 1e9, 4),
        "rmw_ms_token": round(time_s * 1e3, 4),
        "rmw_uJ_token": round(energy_pj / 1e6, 3) if not math.isnan(energy_pj) else None,
        "compute_ms_token": round(compute_s * 1e3, 5),
        "arith_intensity_flopB": round((2 * macs) / traffic, 3),
        "bound": "memory" if time_s > compute_s else "compute",
    }


def s_star(d, m, kv_dim, ttt_dt=2, kv_dt=2):
    return m * d * d * (ttt_dt / kv_dt) / kv_dim


def chunk_intensity(C, d):
    return C * (d + C) / (d + 2 * C)


def c_star(d, ridge):
    a, b, c = 1.0, (d - 2 * ridge), -ridge * d
    disc = b * b - 4 * a * c
    return (-b + math.sqrt(disc)) / (2 * a) if disc >= 0 else float("nan")


def b_max_bw(hw, d, m, L, t_target_s, dt=2):
    return hw["bw_bps"] * t_target_s / ttt_rmw_per_token(d, m, L, dt)


def eval_twin(twin_path):
    hw = load_hw(twin_path)
    ridge = hw["peak_flops_s"] / hw["bw_bps"]
    a = ANCHOR
    dec = ttt_decode(hw, a["d"], a["m"], a["L"])
    Sst = s_star(a["d"], a["m"], a["kv_dim"])
    res = {
        "twin": hw["twin"], "backing_role": hw["backing_role"], "tier": hw["tier"],
        "hw": {"bw_TBps": round(hw["bw_bps"] / 1e12, 4), "peak_PFLOPs": round(hw["peak_flops_s"] / 1e15, 4),
               "cap_GB": round(hw["cap_bytes"] / 1e9, 2), "ondie_MB": round(hw["ondie_bytes"] / 1e6, 2),
               "epb_pj_B": hw["epb"], "matrix_unit": hw["matrix_unit"], "ridge_flopB": round(ridge, 2),
               "ondie_is_backing": hw["ondie_is_backing"]},
        "anchor_1p3B": {
            **dec, "S_star_read_tokens": round(Sst),
            "c_star_chunk": round(c_star(a["d"], ridge), 1),
            "B_max_10ms": round(b_max_bw(hw, a["d"], a["m"], a["L"], 10e-3), 2),
            "B_max_50ms": round(b_max_bw(hw, a["d"], a["m"], a["L"], 50e-3), 2),
            "state_fits_ondie": bool(state_bytes_layer(a["d"], a["m"]) <= hw["ondie_bytes"]),
        },
        "scaling": [],
    }
    for name, s in SCALES:
        d0 = ttt_decode(hw, s["d"], s["m"], s["L"])
        res["scaling"].append({
            "scale": name, "d": s["d"], "L": s["L"],
            "rmw_GB_token": d0["rmw_GB_token"], "rmw_ms_token": d0["rmw_ms_token"],
            "S_star_read_tokens": round(s_star(s["d"], s["m"], s["kv_dim"])),
            "bound": d0["bound"],
            "whole_model_state_GB": round(state_bytes_layer(s["d"], s["m"]) * s["L"] / 1e9, 3),
            "fits_ondie": bool(state_bytes_layer(s["d"], s["m"]) <= hw["ondie_bytes"]),
        })
    return res


def validate():
    r = eval_twin(os.path.join(HERE, "twins", "h100.json"))["anchor_1p3B"]
    ok = True
    for k, gold in H100_GOLD.items():
        got = r[k]
        err = abs(got - gold) / gold
        flag = "OK" if err < 0.001 else "FAIL"
        if err >= 0.001:
            ok = False
        print(f"  {k:20} got={got:<12} gold={gold:<10} err={err*100:.3f}%  {flag}")
    print("VALIDATE:", "PASS — driver reproduces committed E3 anchor" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--twin")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    if args.validate:
        sys.exit(0 if validate() else 1)

    twins = ([args.twin] if args.twin else
             sorted(glob.glob(os.path.join(HERE, "twins", "*.json"))))
    table = []
    for tp in twins:
        try:
            res = eval_twin(tp)
        except Exception as e:
            print(f"  SKIP {os.path.basename(tp)}: {e}")
            continue
        key = os.path.splitext(os.path.basename(tp))[0]
        with open(os.path.join(HERE, "results", f"{key}.json"), "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        an = res["anchor_1p3B"]
        table.append((key, res["tier"], res["backing_role"], res["hw"]["bw_TBps"],
                      res["hw"]["peak_PFLOPs"], res["hw"]["ridge_flopB"],
                      an["rmw_ms_token"], an["S_star_read_tokens"], an["c_star_chunk"],
                      an["B_max_10ms"], an["bound"]))
    # console table
    print(f"\n{'twin':12}{'tier':7}{'back':6}{'BW(TB/s)':>10}{'PF':>7}{'ridge':>8}"
          f"{'ms/tok':>9}{'S*':>9}{'C*':>8}{'Bmax':>7}{'bound':>8}")
    for r in table:
        print(f"{r[0]:12}{str(r[1]):7}{r[2]:6}{r[3]:>10}{r[4]:>7}{r[5]:>8}"
              f"{r[6]:>9}{r[7]:>9}{r[8]:>8}{r[9]:>7}{r[10]:>8}")
    print(f"\n{len(table)} twins evaluated -> multiarch/results/*.json")


if __name__ == "__main__":
    main()
