#!/usr/bin/env python3
"""HOPE / Sleep applied-model decode-cost scaling to 1T params, per HW twin.

Architecture (grounded, notes/2512.24695 + notes/2606.03979):
  HOPE block = self-modifying Titans (6 test-time memories: k,v,q,eta,alpha,+ — projections ARE memory)
               feeding a CMS chain (MLP^(f1..fk), each updated at frequency f_l; fast level RMWs every
               token, slow levels every chunk C^(l)). y_t = MLP^(fk)(...MLP^(f1)(o_t)).
  Sleep      = wake/sleep lifecycle. WAKE = HOPE. SLEEP (offline, no input) = consolidation (fast block's
               knowledge appended as a NEW low-rank EXPERT to the slower block via GKD+RL; expert pool
               GROWS each sleep; structural CF prevention) + dreaming. Slow-level & expert updates are
               ENTIRELY OFFLINE -> NOT on the decode critical path.

The question: scale total model size to 1T by growing FFN + self-modifying Titans + the (Sleep-appended)
expert pool. Does per-token DECODE cost stay bounded? Decode cost is dominated by the memory-bound
per-token RMW traffic (Part III). Three regimes for what is RMW'd per token:
  dense_ttt : whole model is test-time-learned  -> RMW ∝ P_total   (strawman upper bound)
  hope      : self-mod + ALL CMS levels update at inference (fast C=1 + slow amortized 1/C)
  sleep     : self-mod + FAST CMS level only (slow-level consolidation + expert growth are OFFLINE)

All numbers exploration-grade / RELATIVE (per-token RMW state, decode ms/token as ideal roofline lower
bound). Absolutes NOT asserted. Twin BW from multiarch/twins (2-round-verified). See REPORT / Appendix E.
"""
import json, os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_all_experiments import load_hw

HERE = os.path.dirname(os.path.abspath(__file__))
BYTES = 2  # bf16

# --- architecture constants (grounded) ---
M_FAST = 8           # fast CMS level = 2-layer MLP memory (Titans-style, 8 d^2 params)
SELFMOD_N = 6        # self-modifying Titans: 6 memories (k,v,q,eta,alpha,+)
SELFMOD_C = 0.5      # each self-mod memory ~ 0.5 d^2 (small projection-sized state)
SLOW_LEVELS = [16, 256, 4096, 65536]   # slow CMS chunk sizes C^(l) (nested frequencies)
M_SLOW = 8           # each slow level = 2-layer MLP memory (8 d^2)
K_ACTIVE = 2         # MoE top-k active experts per token (Sleep expert pool is routed)

# two scaling families to 1T:
#  WIDTH  — grow d,L (dense backbone) + some experts. Fast-level state ∝ d^2 L, so even Sleep decode grows.
#  EXPERT — HOLD the fast backbone (d,L fixed), grow TOTAL only by the Sleep-appended expert pool (offline,
#           MoE-routed). Fast-level RMW state is CONSTANT -> Sleep decode is FLAT while total -> 1T.
TARGET_P = {"1.3B": 1.3e9, "7B": 7e9, "70B": 70e9, "175B": 175e9, "405B": 405e9, "1T": 1.0e12}
ORDER = ["1.3B", "7B", "70B", "175B", "405B", "1T"]
RECIPE_WIDTH = {"1.3B": (2048, 24), "7B": (3072, 32), "70B": (4096, 48),
                "175B": (6144, 64), "405B": (8192, 80), "1T": (8192, 96)}
RECIPE_EXPERT = {k: (2048, 24) for k in ORDER}   # fixed decode-affordable backbone; grow experts only


def params(d, L):
    """Per-model parameter decomposition (all ×L layers)."""
    p_attn = 4 * d * d * L                       # QKVO
    p_ffn_shared = 8 * d * d * L                 # one always-on FFN (d_ff=4d, 2 matrices)
    p_selfmod = SELFMOD_N * SELFMOD_C * d * d * L
    p_cms_fast = M_FAST * d * d * L
    p_cms_slow = len(SLOW_LEVELS) * M_SLOW * d * d * L
    p_expert_each = 8 * d * d * L                 # each MoE/CMS expert = FFN-sized
    return dict(attn=p_attn, ffn=p_ffn_shared, selfmod=p_selfmod,
                cms_fast=p_cms_fast, cms_slow=p_cms_slow, expert_each=p_expert_each)


def build(label, d, L):
    P = params(d, L)
    backbone = P["attn"] + P["ffn"] + P["selfmod"] + P["cms_fast"] + P["cms_slow"]
    target = TARGET_P[label]
    n_experts = max(0, round((target - backbone) / P["expert_each"]))
    total = backbone + n_experts * P["expert_each"]
    # active params per token (forward): backbone(minus dormant experts) + top-k experts
    active = P["attn"] + P["ffn"] + P["selfmod"] + P["cms_fast"] + P["cms_slow"] + min(K_ACTIVE, n_experts) * P["expert_each"]
    # per-token RMW STATE bytes (the memory-bound decode driver) by regime -----------------
    selfmod_B = P["selfmod"] * BYTES
    fast_B = P["cms_fast"] * BYTES
    # slow levels amortized: state/C summed
    slow_amort_B = sum((M_SLOW * d * d * L) * BYTES / C for C in SLOW_LEVELS)
    S_dense = total * BYTES                        # whole model is TTT state
    S_hope = selfmod_B + fast_B + slow_amort_B     # self-mod + fast(C=1) + slow amortized
    S_sleep = selfmod_B + fast_B                   # fast level + self-mod ONLY (rest offline)
    return dict(label=label, d=d, L=L, n_experts=n_experts, total=total, active=active,
                S_dense=S_dense, S_hope=S_hope, S_sleep=S_sleep, P=P)


def decode_ms(state_bytes, bw):
    """ideal roofline lower bound: RMW traffic = 2*state (read+write), time = traffic/bw."""
    return (2 * state_bytes) / bw * 1e3


def family(recipe):
    return [build(l, *recipe[l]) for l in ORDER]


def main():
    twins = {}
    for k in ("h100", "mtia2", "wse3", "b100", "vr100"):
        p = os.path.join(HERE, "twins", f"{k}.json")
        if os.path.exists(p):
            twins[k] = load_hw(p)
    fams = {"WIDTH (grow d,L)": family(RECIPE_WIDTH),
            "EXPERT (fixed backbone, grow experts offline)": family(RECIPE_EXPERT)}

    for fname, rows in fams.items():
        print("=" * 104)
        print(f"HOPE/Sleep scaling to 1T — family: {fname}")
        print("=" * 104)
        print(f"{'scale':6}{'d':6}{'L':4}{'#exp':7}{'total P':>10}{'active P':>10}"
              f"{'S_dense':>11}{'S_hope':>11}{'S_sleep':>11}")
        for r in rows:
            print(f"{r['label']:6}{r['d']:<6}{r['L']:<4}{r['n_experts']:<7}"
                  f"{r['total']/1e9:>8.0f}B{r['active']/1e9:>8.1f}B"
                  f"{r['S_dense']/1e9:>9.1f}GB{r['S_hope']/1e9:>9.2f}GB{r['S_sleep']/1e9:>9.2f}GB")
        print(f"\n  decode ms/tok on H100 (BW 3.35 TB/s), ideal lower bound:")
        print(f"  {'scale':6}{'sleep':>10}{'hope':>10}{'dense':>12}{'dense/sleep':>13}")
        for r in rows:
            s = decode_ms(r["S_sleep"], 3.35e12); h = decode_ms(r["S_hope"], 3.35e12); dn = decode_ms(r["S_dense"], 3.35e12)
            print(f"  {r['label']:6}{s:>10.3f}{h:>10.3f}{dn:>12.1f}{dn/s:>12.0f}x")
        print()

    out = {"arch": "HOPE/Sleep (notes 2512.24695 + 2606.03979)", "bytes": BYTES,
           "constants": {"m_fast": M_FAST, "selfmod_n": SELFMOD_N, "slow_levels": SLOW_LEVELS, "k_active": K_ACTIVE},
           "families": {}}
    for fname, rows in fams.items():
        out["families"][fname] = {
            "scales": [{k: v for k, v in r.items() if k != "P"} for r in rows],
            "decode_ms": {tk: {r["label"]: {"sleep": round(decode_ms(r["S_sleep"], hw["bw"]), 5),
                                            "hope": round(decode_ms(r["S_hope"], hw["bw"]), 5),
                                            "dense": round(decode_ms(r["S_dense"], hw["bw"]), 3)}
                               for r in rows} for tk, hw in twins.items()}}
    json.dump(out, open(os.path.join(HERE, "results_full", "_hope_sleep_scaling.json"), "w"),
              ensure_ascii=False, indent=1)
    print("-> multiarch/results_full/_hope_sleep_scaling.json")


if __name__ == "__main__":
    main()
