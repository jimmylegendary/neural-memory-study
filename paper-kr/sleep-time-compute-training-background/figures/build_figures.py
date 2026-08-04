#!/usr/bin/env python3
"""Generate the original vector figures used by the STC training background."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


OUT = Path(__file__).parent
NAVY = "#17263C"
BLUE = "#2F6690"
TEAL = "#3A7D78"
AMBER = "#A46719"
RED = "#8C3B3B"
MIST = "#EEF2F5"
SAND = "#F4EFE6"
INK = "#22262C"
LINE = "#BAC4CE"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK KR", "DejaVu Sans"],
        "axes.unicode_minus": False,
        # Matplotlib 3.11 Type42 subsetting over the Noto CJK TTC can emit a
        # post-subset glyph id above 0xFFFF and fail while writing CIDToGIDMap.
        # Type3 keeps the figure vector-based and avoids that backend defect;
        # searchable Korean alternatives remain in the LaTeX captions.
        "pdf.fonttype": 3,
        "figure.facecolor": "white",
    }
)


def box(ax, xy, width, height, title, subtitle="", color=BLUE, fill=MIST):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.5,
        edgecolor=color,
        facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.62,
        title,
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
        color=NAVY,
    )
    if subtitle:
        ax.text(
            x + width / 2,
            y + height * 0.28,
            subtitle,
            ha="center",
            va="center",
            fontsize=8.4,
            color=INK,
        )
    return patch


def arrow(ax, start, end, color=LINE, label=None, curve=0.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.6,
        color=color,
        connectionstyle=f"arc3,rad={curve}",
    )
    ax.add_patch(patch)
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2 + (0.04 if curve >= 0 else -0.04)
        ax.text(mx, my, label, ha="center", va="center", fontsize=8, color=INK)


def setup(width=12, height=5.2):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def learning_map():
    fig, ax = setup(12, 4.8)
    steps = [
        ("데이터", "예시·경험", BLUE),
        ("순전파", "현재 모델의 예측", BLUE),
        ("손실", "목표와 오차", AMBER),
        ("역전파", "파라미터별 기울기", AMBER),
        ("옵티마이저", "상태를 포함한 갱신", TEAL),
        ("검증", "보지 않은 데이터", TEAL),
        ("출판", "버전·롤백·배포", NAVY),
    ]
    xs = [0.02, 0.16, 0.30, 0.44, 0.58, 0.72, 0.86]
    for idx, ((title, subtitle, color), x) in enumerate(zip(steps, xs)):
        box(ax, (x, 0.42), 0.115, 0.23, title, subtitle, color=color)
        if idx < len(steps) - 1:
            arrow(ax, (x + 0.115, 0.535), (xs[idx + 1], 0.535), color=color)
    arrow(ax, (0.775, 0.40), (0.635, 0.26), color=RED, label="실패 시 재학습", curve=0.18)
    arrow(ax, (0.635, 0.26), (0.075, 0.40), color=RED, label="데이터·목표 수정", curve=0.13)
    ax.text(0.02, 0.88, "학습은 하나의 커널이 아니라 검증 가능한 상태 전환 파이프라인이다", fontsize=16, weight="bold", color=NAVY)
    ax.text(0.02, 0.80, "Sleep-time compute도 동일한 파이프라인을 wake latency 바깥에서 수행한다.", fontsize=10, color=INK)
    save(fig, "learning-map.pdf")


def training_regimes():
    fig, ax = setup(10.5, 6.3)
    ax.set_xlim(-0.1, 1.05)
    ax.set_ylim(-0.1, 1.05)
    ax.arrow(0, 0, 0.98, 0, head_width=0.025, head_length=0.025, color=NAVY, length_includes_head=True)
    ax.arrow(0, 0, 0, 0.98, head_width=0.025, head_length=0.025, color=NAVY, length_includes_head=True)
    ax.text(0.50, -0.075, "배포 이후 시간  →", ha="center", fontsize=11, weight="bold", color=NAVY)
    ax.text(-0.08, 0.50, "바뀌는 상태의 크기  →", ha="center", va="center", rotation=90, fontsize=11, weight="bold", color=NAVY)
    regions = [
        (0.08, 0.82, "사전학습", "전체 가중치", BLUE),
        (0.24, 0.70, "지속 사전학습", "전체/대규모 가중치", BLUE),
        (0.34, 0.56, "Full fine-tuning", "전체 가중치", AMBER),
        (0.43, 0.35, "LoRA / Adapter", "작은 parametric delta", TEAL),
        (0.61, 0.62, "Test-time training", "wake-path fast state", AMBER),
        (0.78, 0.45, "Parametric sleep", "deferred adapter/weights", RED),
        (0.80, 0.16, "External-memory sleep", "text/vector/graph", TEAL),
        (0.62, 0.08, "Retrieval / cache", "비학습 외부 상태", BLUE),
    ]
    for x, y, title, sub, color in regions:
        ax.scatter([x], [y], s=125, color=color, zorder=3)
        ax.text(x + 0.018, y + 0.016, title, fontsize=9.5, weight="bold", color=NAVY)
        ax.text(x + 0.018, y - 0.025, sub, fontsize=7.8, color=INK)
    ax.axvline(0.53, linestyle="--", color=LINE, linewidth=1.2)
    ax.text(0.52, 0.99, "배포 경계", ha="right", va="top", fontsize=9, color=INK)
    ax.text(0.02, 1.02, "언제 배우는가 × 무엇을 바꾸는가", fontsize=16, weight="bold", color=NAVY)
    save(fig, "training-regimes.pdf")


def continual_learning_map():
    fig, ax = setup(12, 6.2)
    ax.text(0.02, 0.92, "안정성-가소성 문제를 푸는 네 가지 레버", fontsize=16, weight="bold", color=NAVY)
    columns = [
        (0.02, "Replay", "과거 데이터/생성 샘플을 다시 본다", ["buffer", "coreset", "generative replay"], BLUE),
        (0.27, "Regularize", "중요한 방향의 변화를 벌점화한다", ["EWC", "distillation", "functional constraint"], AMBER),
        (0.52, "Constrain/Isolate", "허용된 방향·모듈만 바꾼다", ["GEM", "LoRA", "routing"], TEAL),
        (0.77, "Grow/Compact", "용량을 늘리고 나중에 압축한다", ["experts", "adapter bank", "prune/merge"], RED),
    ]
    for x, title, desc, items, color in columns:
        box(ax, (x, 0.54), 0.21, 0.24, title, desc, color=color, fill="white")
        for j, item in enumerate(items):
            ax.text(x + 0.02, 0.44 - j * 0.075, f"• {item}", fontsize=9, color=INK)
        arrow(ax, (x + 0.105, 0.30), (0.50, 0.14), color=color)
    box(ax, (0.37, 0.03), 0.26, 0.15, "검증 가능한 기억", "retention · transfer · access · deletion", color=NAVY, fill=MIST)
    ax.text(0.02, 0.83, "어느 한 방법도 용량·재현·롤백·보안을 동시에 해결하지 않는다.", fontsize=10, color=INK)
    save(fig, "continual-learning-map.pdf")


def rl_sleep_policy():
    fig, ax = setup(12, 5.7)
    ax.text(0.02, 0.91, "Sleep controller를 정책으로 보면", fontsize=16, weight="bold", color=NAVY)
    box(ax, (0.03, 0.54), 0.17, 0.22, "상태 $s_t$", "새 경험·staleness·capacity\nSLA·energy·risk", color=BLUE)
    box(
        ax,
        (0.28, 0.54),
        0.17,
        0.22,
        r"정책 $\pi(a|s)$",
        "admit · defer · route\nevict · consolidate",
        color=TEAL,
    )
    box(ax, (0.53, 0.54), 0.17, 0.22, "행동 $a_t$", "skip · summarize · graph\nadapter · full update", color=AMBER)
    box(ax, (0.78, 0.54), 0.17, 0.22, "다음 wake", "quality · latency\nforgetting · audit", color=NAVY)
    arrow(ax, (0.20, 0.65), (0.28, 0.65), color=BLUE)
    arrow(ax, (0.45, 0.65), (0.53, 0.65), color=TEAL)
    arrow(ax, (0.70, 0.65), (0.78, 0.65), color=AMBER)
    box(ax, (0.28, 0.15), 0.42, 0.18, "보상 $r_t$", "later utility − sleep cost − staleness − interference − governance risk", color=RED, fill=SAND)
    arrow(ax, (0.865, 0.53), (0.70, 0.33), color=RED, label="지연된 관측")
    arrow(ax, (0.28, 0.24), (0.15, 0.53), color=RED, label="credit assignment", curve=0.20)
    ax.text(0.03, 0.05, "핵심 난점: 보상은 늦고, counterfactual은 관측되지 않으며, online 탐색은 사용자 상태를 손상할 수 있다.", fontsize=9.3, color=INK)
    save(fig, "rl-sleep-policy.pdf")


def training_memory_cost():
    fig, ax = plt.subplots(figsize=(11, 6.0))
    categories = ["Inference", "LoRA sleep", "Full FT sleep"]
    weights = [1.0, 1.0, 1.0]
    gradients = [0.0, 0.08, 1.0]
    optim = [0.0, 0.16, 2.0]
    activ = [0.25, 0.45, 0.80]
    import numpy as np

    x = np.arange(len(categories))
    bottom = np.zeros(len(categories))
    for values, label, color in [
        (weights, "model weights", NAVY),
        (gradients, "gradients", AMBER),
        (optim, "optimizer state", RED),
        (activ, "saved activations", TEAL),
    ]:
        ax.bar(x, values, bottom=bottom, label=label, color=color, width=0.62)
        bottom += np.array(values)
    ax.set_xticks(x, categories, fontsize=11)
    ax.set_ylabel("상대 메모리(개념적)")
    ax.set_title("학습이 시작되면 HBM에는 가중치 외 상태가 생긴다", loc="left", fontsize=16, weight="bold", color=NAVY)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=LINE, alpha=0.5)
    ax.text(0.02, -0.18, "정확한 배수는 optimizer, precision, sharding, recomputation에 따라 달라진다.", transform=ax.transAxes, fontsize=9, color=INK)
    fig.tight_layout()
    save(fig, "training-memory-cost.pdf")


def main():
    learning_map()
    training_regimes()
    continual_learning_map()
    rl_sleep_policy()
    training_memory_cost()
    print("generated 5 background figures")


if __name__ == "__main__":
    main()
