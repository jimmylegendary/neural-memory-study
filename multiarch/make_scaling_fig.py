#!/usr/bin/env python3
"""fig8 — HOPE/Sleep decode-cost scaling to 1T (two families). Reads _hope_sleep_scaling.json."""
import json, os
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_CJK):
    fm.fontManager.addfont(_CJK); matplotlib.rcParams["font.family"] = "Noto Sans CJK KR"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "results_full", "_hope_sleep_scaling.json")))
fams = D["families"]
FA = "WIDTH (grow d,L)"
FB = "EXPERT (fixed backbone, grow experts offline)"

fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
for ax, fname, title in [(axs[0], FA, "(a) WIDTH — d,L 성장"),
                         (axs[1], FB, "(b) EXPERT — backbone 고정, expert offline 성장")]:
    rows = fams[fname]["scales"]
    dm = fams[fname]["decode_ms"]["h100"]
    tot = [r["total"] / 1e9 for r in rows]
    labels = [r["label"] for r in rows]
    dense = [dm[r["label"]]["dense"] for r in rows]
    hope = [dm[r["label"]]["hope"] for r in rows]
    sleep = [dm[r["label"]]["sleep"] for r in rows]
    ax.plot(tot, dense, "o-", color="#c1121f", lw=2, label="Dense-all-TTT")
    ax.plot(tot, hope, "s--", color="#5a189a", lw=1.6, label="HOPE (inference 갱신)")
    ax.plot(tot, sleep, "D-", color="#1a759f", lw=2, label="Sleep (offline consolidation)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("total params (B, log)")
    ax.set_title(title, fontsize=10)
    ax.grid(ls=":", alpha=0.4)
    for x, y, l in zip(tot, sleep, labels):
        if l in ("1.3B", "1T"):
            ax.annotate(l, (x, y), fontsize=7, xytext=(0, -12), textcoords="offset points", ha="center")
axs[0].set_ylabel("decode ms/token @ H100 (ideal 하한, log)")
axs[0].legend(fontsize=7, loc="upper left")
axs[1].annotate("Sleep 평탄:\n1T도 6.4B backbone처럼 decode\n(dense 대비 149×; floor=M3)", (600, 20), fontsize=7.5,
                color="#1a759f", ha="center",
                bbox=dict(boxstyle="round", fc="#e8f0f2", ec="#1a759f", alpha=0.9))
fig.suptitle("fig8 · HOPE/Sleep decode 비용 1T 스케일링 — Sleep은 expert 성장을 offline로 밀어 decode를 묶는다", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(HERE, "figures", "fig8-hope-sleep-scaling.png"), dpi=150)
print("-> figures/fig8-hope-sleep-scaling.png")
