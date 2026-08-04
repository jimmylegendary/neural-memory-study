#!/usr/bin/env python3
"""Generate the fifteen canonical original figures for the STC Study Paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


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
WHITE = "#FFFFFF"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK KR", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 3,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "axes.edgecolor": LINE,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    }
)


def canvas(width=12, height=5.4, axis=False):
    fig, ax = plt.subplots(figsize=(width, height))
    if not axis:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, subtitle="", edge=BLUE, fill=MIST, fs=10):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=fill,
        edgecolor=edge,
        linewidth=1.5,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", color=NAVY, weight="bold", fontsize=fs)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", color=INK, fontsize=fs - 2)
    return patch


def arrow(ax, p1, p2, color=BLUE, label="", curve=0):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.5,
            color=color,
            connectionstyle=f"arc3,rad={curve}",
        )
    )
    if label:
        ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + 0.04, label, ha="center", fontsize=8, color=INK)


def title(ax, text, subtitle=""):
    ax.text(0.02, 0.95, text, ha="left", va="top", fontsize=15, weight="bold", color=NAVY)
    if subtitle:
        ax.text(0.02, 0.89, subtitle, ha="left", va="top", fontsize=9, color=INK)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def f001_lifecycle():
    fig, ax = canvas(height=4.8)
    title(ax, "Wake–test–sleep lifecycle", "경계는 clock time이 아니라 response-critical dependency다")
    stages = [
        (0.03, "WAKE", "query·tool·fast state", BLUE),
        (0.27, "CAPTURE", "episode·feedback·trace", TEAL),
        (0.51, "SLEEP", "replay·synthesis·training", AMBER),
        (0.75, "PUBLISH", "validate·version·rollback", RED),
    ]
    for i, (x, a, b, c) in enumerate(stages):
        box(ax, x, 0.42, 0.19, 0.25, a, b, edge=c, fill=WHITE)
        if i < 3:
            arrow(ax, (x + 0.19, 0.545), (stages[i + 1][0], 0.545), c)
    arrow(ax, (0.845, 0.39), (0.125, 0.39), RED, "reusable state for later wake", curve=-0.20)
    ax.text(0.125, 0.23, "latency-critical", ha="center", weight="bold", color=BLUE)
    ax.text(0.62, 0.23, "deferred transformation", ha="center", weight="bold", color=AMBER)
    ax.plot([0.24, 0.24], [0.18, 0.76], linestyle="--", color=LINE, linewidth=1.2)
    save(fig, "stc-f001.pdf")


def f002_taxonomy():
    fig, ax = canvas(height=5.8)
    title(ax, "두 개의 독립 축: update timing × memory medium")
    cols = ["Within-query", "Post-response", "Periodic sleep"]
    rows = ["Ephemeral / state", "External text·vector·graph", "Parametric weights·adapter"]
    examples = [
        ["KV / recurrent state", "session fast state", "state compaction"],
        ["retrieval write", "Letta / Mem0", "Dreaming / graph rebuild"],
        ["TTT / neural memory", "async adapter", "replay / distill / refresh"],
    ]
    x0, y0, cw, rh = 0.23, 0.16, 0.245, 0.20
    for j, c in enumerate(cols):
        ax.text(x0 + j * cw + cw / 2, 0.82, c, ha="center", weight="bold", color=NAVY, fontsize=10)
    for i, r in enumerate(rows):
        ax.text(0.20, y0 + (2 - i) * rh + rh / 2, r, ha="right", va="center", color=INK, fontsize=9)
        for j in range(3):
            edge = [BLUE, TEAL, AMBER][j]
            fill = [MIST, "#E8F2F0", SAND][j]
            box(ax, x0 + j * cw, y0 + (2 - i) * rh, cw - 0.02, rh - 0.025, examples[i][j], edge=edge, fill=fill, fs=9)
    ax.text(0.60, 0.08, "같은 medium도 timing이 다르면 SLA·data horizon·validation이 달라진다", ha="center", fontsize=9, color=INK)
    save(fig, "stc-f002.pdf")


def f003_chronology():
    fig, ax = canvas(width=13, height=4.7)
    title(ax, "1995–2026: 이름보다 먼저 존재한 계보")
    events = [
        (1995, "CLS", "fast/slow learning"),
        (2017, "EWC·DGR·GEM", "continual learning"),
        (2020, "PAD", "wake/NREM/REM"),
        (2023, "MemGPT", "external hierarchy"),
        (2025, "Titans·HOPE", "multi-rate memory"),
        (2026, "Need Sleep", "parametric consolidation"),
        (2026.4, "Dreaming·Mem0", "product background synthesis"),
    ]
    ax.plot([0.06, 0.95], [0.46, 0.46], color=NAVY, linewidth=2)
    lo, hi = 1995, 2027
    for idx, (year, name, note) in enumerate(events):
        x = 0.06 + (year - lo) / (hi - lo) * 0.89
        y = 0.62 if idx % 2 == 0 else 0.25
        ax.plot([x, x], [0.46, y], color=[BLUE, TEAL, AMBER, RED][idx % 4], linewidth=1.4)
        ax.scatter([x], [0.46], s=55, color=[BLUE, TEAL, AMBER, RED][idx % 4], zorder=3)
        ax.text(x, y + (0.03 if y > 0.46 else -0.03), f"{int(year)}  {name}\n{note}", ha="center", va="bottom" if y > 0.46 else "top", fontsize=8.5, color=INK, weight="bold" if idx in (5, 6) else "normal")
    save(fig, "stc-f003.pdf")


def f004_problem_matrix():
    fig, ax = canvas(width=12, height=6.3, axis=True)
    problems = ["freshness", "personalization", "agent experience", "bounded capacity", "delete/rollback", "repeated long context"]
    methods = ["long context", "retrieval", "model editing", "continual FT", "external sleep", "parametric sleep"]
    data = np.array([
        [1, 3, 2, 2, 3, 1],
        [1, 3, 2, 2, 3, 3],
        [1, 3, 1, 2, 3, 2],
        [0, 1, 0, 1, 3, 2],
        [0, 3, 1, 0, 3, 0],
        [3, 2, 1, 1, 2, 3],
    ])
    cmap = mpl.colors.ListedColormap(["#F2F3F4", "#DCE7EF", "#A8C8D7", "#3A7D78"])
    ax.imshow(data, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(len(methods)), methods, rotation=22, ha="right", fontsize=9)
    ax.set_yticks(range(len(problems)), problems, fontsize=9)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, ["weak", "niche", "useful", "strong"][data[i, j]], ha="center", va="center", fontsize=7.5, color=NAVY)
    ax.set_title("문제별 strongest default와 STC 기여는 다르다", loc="left", color=NAVY, fontsize=14, weight="bold", pad=16)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, "stc-f004.pdf")


def f005_promisingness():
    fig, ax = canvas(width=11, height=6.3, axis=True)
    points = [
        (4.6, 4.6, "external synthesis", TEAL),
        (3.4, 4.4, "hybrid promotion", BLUE),
        (1.8, 4.5, "per-user full weights", RED),
        (3.8, 3.4, "periodic global refresh", AMBER),
        (4.2, 3.8, "index / graph compaction", TEAL),
        (2.2, 3.9, "on-device consolidation", BLUE),
    ]
    ax.set_xlim(0, 5.2); ax.set_ylim(0, 5.2)
    ax.axvspan(0, 2.6, color=SAND, alpha=0.45); ax.axvspan(2.6, 5.2, color=MIST, alpha=0.6)
    ax.axhline(2.6, color=LINE, linewidth=1); ax.axvline(2.6, color=LINE, linewidth=1)
    for x, y, label, color in points:
        ax.scatter(x, y, s=160, color=color, edgecolor=WHITE, linewidth=1.5)
        ax.text(x + 0.08, y + 0.10, label, fontsize=9, color=INK)
    ax.set_xlabel("Evidence maturity →", fontsize=10); ax.set_ylabel("Expected value →", fontsize=10)
    ax.set_title("조건부 promisingness: maturity와 value를 분리한다", loc="left", color=NAVY, fontsize=14, weight="bold")
    ax.text(0.2, 4.95, "research option", color=AMBER, fontsize=9); ax.text(3.0, 4.95, "near-term priority", color=TEAL, fontsize=9)
    ax.grid(alpha=0.15)
    save(fig, "stc-f005.pdf")


def f006_hybrid():
    fig, ax = canvas(width=13, height=5.7)
    title(ax, "Hybrid: external system-of-record → selective reversible promotion")
    box(ax, 0.03, 0.42, 0.17, 0.23, "Episode ledger", "source·scope·time", BLUE, WHITE)
    box(ax, 0.25, 0.42, 0.18, 0.23, "External memory", "text·vector·graph", TEAL, "#E8F2F0")
    box(ax, 0.49, 0.42, 0.18, 0.23, "Promotion gate", "reuse·stability·cost", AMBER, SAND)
    box(ax, 0.73, 0.42, 0.20, 0.23, "Parametric cache", "LoRA·adapter·bounded state", RED, "#F5EAEA")
    for a, b, c in [(0.20, 0.25, BLUE), (0.43, 0.49, TEAL), (0.67, 0.73, AMBER)]:
        arrow(ax, (a, 0.535), (b, 0.535), c)
    arrow(ax, (0.83, 0.39), (0.12, 0.39), RED, "rollback / rebuild from lineage", curve=-0.20)
    ax.text(0.34, 0.23, "authoritative, deletable, temporal", ha="center", fontsize=9, color=TEAL, weight="bold")
    ax.text(0.82, 0.23, "serving accelerator, not sole truth", ha="center", fontsize=9, color=RED, weight="bold")
    save(fig, "stc-f006.pdf")


def f007_evidence_types():
    fig, ax = canvas(width=12, height=5.8)
    title(ax, "수식의 지위를 섞지 않는다")
    items = [
        (0.03, "MEASURED", "trace·quality·bytes\n직접 관측", TEAL),
        (0.27, "IDENTITY", "cost = compute + I/O\n항등식", BLUE),
        (0.51, "FIT CANDIDATE", "restricted surface\n데이터로 적합", AMBER),
        (0.75, "HYPOTHESIS", "capacity knee·cadence\n반증 전 가설", RED),
    ]
    for i, (x, a, b, c) in enumerate(items):
        box(ax, x, 0.43, 0.19, 0.25, a, b, c, WHITE, 10)
        if i < 3:
            arrow(ax, (x + 0.19, 0.555), (items[i + 1][0], 0.555), LINE)
    ax.text(0.50, 0.25, "universal ‘more sleep FLOPs → better memory’ law: not established", ha="center", color=RED, weight="bold", fontsize=10)
    save(fig, "stc-f007.pdf")


def f008_break_even():
    fig, ax = canvas(width=10.5, height=6.0, axis=True)
    n = np.linspace(0, 120, 241)
    external = 8 + 0.52 * n
    promote = 42 + 0.13 * n
    ax.plot(n, external, color=TEAL, linewidth=2.5, label="external retrieval cumulative cost")
    ax.plot(n, promote, color=RED, linewidth=2.5, label="sleep promotion + cheap reuse")
    cross = (42 - 8) / (0.52 - 0.13)
    ax.axvline(cross, color=AMBER, linestyle="--", linewidth=1.5)
    ax.scatter([cross], [8 + 0.52 * cross], s=90, color=AMBER, zorder=4)
    ax.text(cross + 2, 15, f"break-even ≈ {cross:.0f} reuses", color=AMBER, weight="bold")
    ax.set_xlabel("future valid reuses N"); ax.set_ylabel("cumulative lifecycle cost")
    ax.set_title("Promotion은 reuse가 one-time sleep cost를 회수할 때만 이긴다", loc="left", color=NAVY, fontsize=14, weight="bold")
    ax.legend(frameon=False, loc="upper left"); ax.grid(alpha=0.18)
    save(fig, "stc-f008.pdf")


def f009_capacity_knee():
    fig, ax = canvas(width=10.5, height=6.0, axis=True)
    x = np.linspace(0, 1, 300)
    utility = 1 - np.exp(-4.2 * x)
    interference = 0.05 + 0.12 * x + 1.8 * np.maximum(0, x - 0.68) ** 2
    marginal = 4.2 * np.exp(-4.2 * x)
    ax.plot(x, utility, color=TEAL, linewidth=2.5, label="retained utility")
    ax.plot(x, interference, color=RED, linewidth=2.5, label="interference + compaction cost")
    ax.plot(x, marginal / marginal.max(), color=BLUE, linewidth=1.8, linestyle="--", label="normalized marginal gain")
    ax.axvspan(0.68, 1.0, color=SAND, alpha=0.55)
    ax.axvline(0.68, color=AMBER, linestyle="--")
    ax.text(0.70, 0.85, "capacity knee", color=AMBER, weight="bold")
    ax.set_xlabel("occupied effective capacity"); ax.set_ylabel("normalized metric")
    ax.set_title("Finite memory: bytes가 남아도 utility knee가 먼저 올 수 있다", loc="left", color=NAVY, fontsize=14, weight="bold")
    ax.legend(frameon=False); ax.grid(alpha=0.18)
    save(fig, "stc-f009.pdf")


def f010_cadence():
    fig, ax = canvas(width=10.5, height=6.0, axis=True)
    interval = np.logspace(-1, 2, 300)
    staleness = 0.36 * interval ** 0.55
    batching = 2.2 / np.sqrt(interval) + 0.15
    total = staleness + batching
    optimum = interval[np.argmin(total)]
    ax.plot(interval, staleness, color=RED, label="staleness cost")
    ax.plot(interval, batching, color=BLUE, label="launch / batching cost")
    ax.plot(interval, total, color=TEAL, linewidth=2.8, label="total")
    ax.axvline(optimum, color=AMBER, linestyle="--")
    ax.text(optimum * 1.12, total.min() + 0.12, "workload-conditioned cadence", color=AMBER, weight="bold")
    ax.set_xscale("log"); ax.set_xlabel("sleep interval / batch window"); ax.set_ylabel("normalized lifecycle cost")
    ax.set_title("Cadence frontier: 더 자주도, 더 오래 모으기도 항상 최적은 아니다", loc="left", color=NAVY, fontsize=14, weight="bold")
    ax.legend(frameon=False); ax.grid(alpha=0.18, which="both")
    save(fig, "stc-f010.pdf")


def f011_roofline():
    fig, ax = canvas(width=10.5, height=6.2, axis=True)
    intensity = np.logspace(-2, 3, 400)
    for bw, peak, color, label in [(1.0, 100, BLUE, "wake cluster"), (0.55, 100, AMBER, "remote sleep state")]:
        perf = np.minimum(peak, bw * intensity)
        ax.plot(intensity, perf, color=color, linewidth=2.2, label=label)
    pts = [(0.08, 0.08, "index/graph"), (0.7, 0.7, "optimizer offload"), (8, 8, "adapter FT"), (90, 80, "dense GEMM")]
    for x, y, label in pts:
        ax.scatter(x, y, s=80, color=TEAL if x < 10 else RED)
        ax.text(x * 1.18, y * 0.9, label, fontsize=8.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity [FLOP / state-migration byte]"); ax.set_ylabel("effective throughput")
    ax.set_title("Wake–sleep 분리의 숨은 roofline: state migration", loc="left", color=NAVY, fontsize=14, weight="bold")
    ax.legend(frameon=False); ax.grid(alpha=0.18, which="both")
    save(fig, "stc-f011.pdf")


def f012_device_map():
    fig, ax = canvas(width=11, height=6.3, axis=True)
    opportunities = [
        (4.5, 4.7, 210, "snapshot/COW", TEAL),
        (4.3, 4.2, 190, "provenance store", TEAL),
        (4.0, 3.8, 180, "index/graph tier", BLUE),
        (3.4, 4.1, 160, "near-data compaction", BLUE),
        (3.0, 3.5, 170, "adapter fabric", AMBER),
        (2.6, 3.6, 160, "CXL pooling", AMBER),
        (2.1, 3.8, 150, "neural-state cache", RED),
        (1.8, 3.6, 140, "on-device sleep", RED),
    ]
    for x, y, s, label, color in opportunities:
        ax.scatter(x, y, s=s, color=color, alpha=0.88, edgecolor=WHITE)
        ax.text(x + 0.06, y + 0.06, label, fontsize=8.5)
    ax.set_xlim(1, 5); ax.set_ylim(2.8, 5.1)
    ax.set_xlabel("evidence maturity →"); ax.set_ylabel("cross-scenario residual value →")
    ax.set_title("Memory-device 기회: broad parametric sleep 없이도 남는가?", loc="left", color=NAVY, fontsize=14, weight="bold")
    ax.grid(alpha=0.18)
    save(fig, "stc-f012.pdf")


def f013_failures():
    fig, ax = canvas(width=13, height=6.0)
    title(ax, "Failure mode마다 lifecycle control을 붙인다")
    failures = ["poisoning", "stale fact", "forgetting", "capacity", "privacy", "model collapse"]
    controls = ["provenance", "temporal version", "replay slices", "admit/evict", "scope/delete", "real-data anchor"]
    ys = np.linspace(0.75, 0.18, 6)
    for i, (f, c, y) in enumerate(zip(failures, controls, ys)):
        box(ax, 0.03, y - 0.045, 0.20, 0.085, f, edge=RED, fill="#F5EAEA", fs=8.5)
        box(ax, 0.77, y - 0.045, 0.20, 0.085, c, edge=TEAL, fill="#E8F2F0", fs=8.5)
        arrow(ax, (0.23, y), (0.43, 0.50), RED)
        arrow(ax, (0.57, 0.50), (0.77, y), TEAL)
    box(ax, 0.43, 0.39, 0.14, 0.22, "Candidate gate", "validate·shadow\nversion·rollback", edge=NAVY, fill=MIST, fs=10)
    save(fig, "stc-f013.pdf")


def f014_landscape():
    fig, ax = canvas(width=11, height=6.3, axis=True)
    points = [
        (4.6, 3.2, "OpenAI Dreaming", TEAL, "industry"),
        (4.4, 3.4, "Mem0 Dream", TEAL, "industry"),
        (3.9, 3.1, "Letta sleep agent", BLUE, "industry"),
        (3.8, 2.6, "Zep/Graphiti", BLUE, "industry"),
        (2.0, 4.5, "Language Models Need Sleep", RED, "academia"),
        (2.4, 4.1, "Nested Learning/HOPE", AMBER, "academia"),
        (2.8, 3.8, "Memory Caching", AMBER, "academia"),
        (3.0, 3.3, "ReasoningBank", BLUE, "academia"),
    ]
    for x, y, label, color, kind in points:
        marker = "s" if kind == "industry" else "o"
        ax.scatter(x, y, s=130, marker=marker, color=color, edgecolor=WHITE)
        ax.text(x + 0.06, y + 0.06, label, fontsize=8.3)
    ax.set_xlim(1.4, 5.1); ax.set_ylim(2.1, 4.9)
    ax.set_xlabel("deployment evidence →"); ax.set_ylabel("parametric / learning depth →")
    ax.set_title("2026-08-05 landscape: products are external-first; parametric work is research-heavy", loc="left", color=NAVY, fontsize=13, weight="bold")
    ax.grid(alpha=0.18)
    save(fig, "stc-f014.pdf")


def f015_agenda():
    fig, ax = canvas(width=13, height=5.7)
    title(ax, "반증 가능한 research → system → device agenda")
    stages = [
        (0.02, "TRACE", "wake episodes\nreuse·staleness"),
        (0.21, "MATCHED BASELINES", "RAG·long context\nedit·global refresh"),
        (0.40, "MULTI-CYCLE", "retention·plasticity\ncapacity knee"),
        (0.59, "END-TO-END", "compute·I/O\npublication·rollback"),
        (0.78, "DEVICE", "migration bytes\nendurance·SLA"),
    ]
    colors = [BLUE, TEAL, AMBER, RED, NAVY]
    for i, (x, a, b) in enumerate(stages):
        box(ax, x, 0.42, 0.17, 0.25, a, b, colors[i], WHITE, 9)
        if i < len(stages) - 1:
            arrow(ax, (x + 0.17, 0.545), (stages[i + 1][0], 0.545), colors[i])
    ax.text(0.50, 0.24, "report non-win regions, uncertainty, provenance and deletion—not only peak quality", ha="center", fontsize=9.5, color=INK, weight="bold")
    save(fig, "stc-f015.pdf")


def main():
    functions = [
        f001_lifecycle,
        f002_taxonomy,
        f003_chronology,
        f004_problem_matrix,
        f005_promisingness,
        f006_hybrid,
        f007_evidence_types,
        f008_break_even,
        f009_capacity_knee,
        f010_cadence,
        f011_roofline,
        f012_device_map,
        f013_failures,
        f014_landscape,
        f015_agenda,
    ]
    for function in functions:
        function()


if __name__ == "__main__":
    main()
