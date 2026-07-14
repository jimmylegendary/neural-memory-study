#!/usr/bin/env python3
"""Cross-architecture figure suite for the multiarch Part III report.

Reproduces the study-paper figure TYPES across all 8 twins + consolidated "한눈에" views:
  fig1  decode ms/token per HW (log)                 -- E1.1
  fig2  roofline ridge vs decode AI + knee           -- E1.1/E3
  fig3  B_max@10ms per HW (log)                       -- E1.3
  fig4  S* invariance across HW (flat)               -- E1.3
  fig5  state placement: on-die budget vs state size -- E1.2 (tier residency)
  fig6  RMW scaling: ms/token vs model size per HW   -- E4
  fig7  consolidated 4-panel master                  -- 한눈에
Reads multiarch/results_full/*.json. Saves multiarch/figures/*.png (150 dpi, PDF-embeddable).
"""
import json, os, glob
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_CJK):
    fm.fontManager.addfont(_CJK)
    matplotlib.rcParams["font.family"] = "Noto Sans CJK KR"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
AI = 0.594  # decode arithmetic intensity (anchor)

TIER_COLOR = {"GOLD": "#c9a227", "SILVER": "#8a8f98", "BRONZE": "#b06a3b", None: "#444"}
LABEL = {"h100": "H100", "b100": "B100", "mi355x": "MI355X", "tpu-v7": "TPU v7",
         "vr200": "Rubin*", "mtia2": "MTIA v2", "groq-lpu": "Groq LPU", "wse3": "WSE-3"}


def load():
    d = {}
    for f in sorted(glob.glob(os.path.join(HERE, "results_full", "*.json"))):
        if os.path.basename(f).startswith("_"):
            continue
        r = json.load(open(f))
        d[r["key"]] = r
    return d


def order_by_bw(D):
    return sorted(D, key=lambda k: D[k]["hw_summary"]["bw_TBps"])


def barcolors(keys, D):
    return [TIER_COLOR[D[k]["tier"]] for k in keys]


def annotate_tier(ax, keys, D):
    from matplotlib.patches import Patch
    seen = {}
    for k in keys:
        seen[D[k]["tier"]] = TIER_COLOR[D[k]["tier"]]
    ax.legend(handles=[Patch(color=c, label=t) for t, c in seen.items()],
              title="confidence", fontsize=7, title_fontsize=7, loc="best")


def fig1(D):
    ks = order_by_bw(D)
    v = [D[k]["E1.1_decode_baseline"]["rmw_ms_token"] for k in ks]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar([LABEL[k] for k in ks], v, color=barcolors(ks, D))
    ax.set_yscale("log")
    for i, x in enumerate(v):
        ax.text(i, x * 1.15, f"{x:g}", ha="center", fontsize=7)
    ax.set_ylabel("decode ms / token (ideal 하한, log)")
    ax.set_title("fig1 · 토큰당 decode 비용 (E1.1) — 5자릿수 스프레드, state 배치가 축")
    annotate_tier(ax, ks, D)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1-decode-ms.png"), dpi=150); plt.close(fig)


def fig2(D):
    ks = order_by_bw(D)
    v = [D[k]["E3_analytical"]["ridge_flopB"] for k in ks]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar([LABEL[k] for k in ks], v, color=barcolors(ks, D))
    ax.set_yscale("log")
    ax.axhline(AI, color="crimson", ls="--", lw=1.3, label=f"decode AI = {AI}")
    for i, (k, x) in enumerate(zip(ks, v)):
        m = x / AI
        ax.text(i, x * 1.15, f"{x:g}\n({m:.0f}×)" if m >= 10 else f"{x:g}\n({m:.2f}×)",
                ha="center", fontsize=6.5)
    ax.set_ylabel("roofline ridge (FLOP/B, log)")
    ax.set_title("fig2 · ridge vs decode AI (E1.1/E3) — 7/8 memory-bound, Cerebras만 knee")
    ax.legend(fontsize=7)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2-ridge-vs-ai.png"), dpi=150); plt.close(fig)


def fig3(D):
    ks = order_by_bw(D)
    v = [D[k]["E1.3_kv_vs_ttt"]["B_max_10ms"] for k in ks]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar([LABEL[k] for k in ks], v, color=barcolors(ks, D))
    ax.set_yscale("log")
    for i, x in enumerate(v):
        ax.text(i, x * 1.15, f"{x:g}", ha="center", fontsize=7)
    ax.set_ylabel("B_max @ 10 ms/token (seqs, log)")
    ax.set_title("fig3 · 동시 시퀀스 상한 B_max (E1.3) — BW에 비례 (BW-bound)")
    annotate_tier(ax, ks, D)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3-bmax.png"), dpi=150); plt.close(fig)


def fig4(D):
    ks = order_by_bw(D)
    v = [D[k]["E1.3_kv_vs_ttt"]["S_star_read_tokens"] for k in ks]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot([LABEL[k] for k in ks], v, "o-", color="#2a6f97", lw=2, ms=7)
    ax.axhline(65536, color="green", ls=":", lw=1)
    ax.set_ylim(0, 90000)
    ax.text(0.02, 0.9, "S* = 65536 tokens — 8개 전부 동일 (하드웨어 무관 workload 속성)",
            transform=ax.transAxes, fontsize=8, color="green")
    ax.set_ylabel("S* (KV↔TTT crossover, tokens)")
    ax.set_title("fig4 · S* 불변성 (E1.3) — 상쇄 논증의 직접 실증")
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4-sstar-invariance.png"), dpi=150); plt.close(fig)


def fig5(D):
    ks = order_by_bw(D)
    cap = [D[k]["hw_summary"]["ondie_MB"] for k in ks]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar([LABEL[k] for k in ks], cap, color=barcolors(ks, D))
    ax.set_yscale("log")
    ax.axhline(134.2, color="crimson", ls="--", lw=1.2, label="1-layer state 134 MB")
    ax.axhline(3221, color="purple", ls="-.", lw=1.2, label="whole-model state 3.2 GB")
    for i, (k, c) in enumerate(zip(ks, cap)):
        v = D[k]["E1.2_state_placement"]["placement_verdict"]
        tag = ("whole" if "whole" in v else "layer" if "per-layer" in v else "spill")
        ax.text(i, c * 1.2, tag, ha="center", fontsize=6.5)
    ax.set_ylabel("on-die/on-chip 예산 (MB, log)")
    ax.set_title("fig5 · state 배치·tier 잔류 (E1.2) — on-chip 예산이 클수록 state 상주")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5-state-placement.png"), dpi=150); plt.close(fig)


def fig6(D):
    ks = order_by_bw(D)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    xs = [row["d"] for row in D["h100"]["E4_scaling"]]
    names = [row["scale"].replace("neural-mem-", "").replace("Titans-", "") for row in D["h100"]["E4_scaling"]]
    for k in ks:
        ys = [row["rmw_ms_token"] for row in D[k]["E4_scaling"]]
        ax.plot(xs, ys, "o-", lw=1.5, ms=4, label=LABEL[k], color=TIER_COLOR[D[k]["tier"]]
                if D[k]["tier"] != "SILVER" else None)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("model width d (→ 규모)"); ax.set_ylabel("decode ms/token (log)")
    ax.set_title("fig6 · RMW 비용 스케일링 (E4) — ∝ m·d²·L, 기울기는 HW 무관 (오프셋만 BW)")
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(ls=":", alpha=0.4)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6-scaling.png"), dpi=150); plt.close(fig)


def fig7(D):
    ks = order_by_bw(D)
    fig, axs = plt.subplots(2, 2, figsize=(9.0, 6.4))
    L = [LABEL[k] for k in ks]; col = barcolors(ks, D)
    # (a) decode ms
    a = axs[0, 0]; a.bar(L, [D[k]["E1.1_decode_baseline"]["rmw_ms_token"] for k in ks], color=col)
    a.set_yscale("log"); a.set_title("(a) decode ms/tok (E1.1)", fontsize=9); a.tick_params(labelsize=6.5)
    a.grid(axis="y", ls=":", alpha=0.4)
    # (b) ridge vs AI
    b = axs[0, 1]; b.bar(L, [D[k]["E3_analytical"]["ridge_flopB"] for k in ks], color=col)
    b.set_yscale("log"); b.axhline(AI, color="crimson", ls="--", lw=1); b.set_title("(b) ridge vs AI (knee=Cerebras)", fontsize=9)
    b.tick_params(labelsize=6.5); b.grid(axis="y", ls=":", alpha=0.4)
    # (c) S* invariance
    c = axs[1, 0]; c.plot(L, [D[k]["E1.3_kv_vs_ttt"]["S_star_read_tokens"] for k in ks], "o-", color="#2a6f97")
    c.set_ylim(0, 90000); c.set_title("(c) S* = 65536 불변 (E1.3)", fontsize=9); c.tick_params(labelsize=6.5)
    c.grid(axis="y", ls=":", alpha=0.4)
    # (d) on-die budget vs state
    d = axs[1, 1]; d.bar(L, [D[k]["hw_summary"]["ondie_MB"] for k in ks], color=col)
    d.set_yscale("log"); d.axhline(134.2, color="crimson", ls="--", lw=1); d.axhline(3221, color="purple", ls="-.", lw=1)
    d.set_title("(d) on-chip 예산 vs state (E1.2)", fontsize=9); d.tick_params(labelsize=6.5)
    d.grid(axis="y", ls=":", alpha=0.4)
    fig.suptitle("fig7 · 한눈에 — 크로스아키텍처 pair thesis (BW 오름차순; 색=신뢰도 등급)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(FIG, "fig7-consolidated.png"), dpi=150); plt.close(fig)


def main():
    D = load()
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        f(D)
    print("figures ->", FIG)
    for p in sorted(glob.glob(os.path.join(FIG, "*.png"))):
        print("  ", os.path.basename(p))


if __name__ == "__main__":
    main()
