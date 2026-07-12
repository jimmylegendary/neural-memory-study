#!/usr/bin/env python3
"""Render Part III experiment figures (PNG) from experiments/*/results.json.

Reproduce: /home/jimmy/repos/neural-memory-study/.venv/bin/python render_experiments.py
Outputs into /home/jimmy/repos/neural-memory-study/figures/:
  exp-a-kv-ttt-crossover.png   (S* traffic crossover; E1.3/E3)
  exp-b-state-placement.png    (energy/latency device tradeoff + residency; E1.2)
  exp-c-chunk-roofline.png     (AI(C) on host roofline; E2.1)
  exp-d-frequency-tiers.png    (cadence->tier admissibility table; E1.4)
  exp-e-rmw-cliff.png          (on/off-die RMW bandwidth cliff; E2.2)

Exploration-grade: ratios / crossovers / shapes are the load-bearing claims (see each caption).
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

EXP = "/home/jimmy/repos/neural-memory-study/experiments"
FIG = "/home/jimmy/repos/neural-memory-study/figures"

def load(e):
    return json.load(open(os.path.join(EXP, e, "results.json")))

E3, E12, E13, E14, E21, E22 = (load("E3-analytical"), load("E1.2-state-placement"),
    load("E1.3-kv-vs-ttt"), load("E1.4-frequency-tiers"),
    load("E2.1-chunk-intensity"), load("E2.2-rmw-cliff"))

# ---- shared style ------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
})
C_TTT, C_KV, C_MARK, C_ACC = "#c0392b", "#2471a3", "#7f8c8d", "#1e8449"
BW = 3.35e12  # HBM3 bytes/s from twin

def save(fig, name):
    p = os.path.join(FIG, name)
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)
    return p

# =============================================================================
# (a) KV vs TTT per-token traffic crossover S*  (E1.3 anchor + E3 scaling)
# =============================================================================
def fig_a():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    # left: anchor neural-mem-1.3B, KV read grows w/ S, TTT const
    kvS = E13["kv_B_max_vs_S"]
    S = np.array([r["S_tokens"] for r in kvS], float)
    kv_gb = np.array([r["kv_ms_token"] for r in kvS]) * (BW/1e12)  # ms * TB/s = GB
    ttt_read = E13["panelA_S_star"][2]["ttt_read_GB_token"]   # 3.2212
    ttt_rmw  = E13["panelA_S_star"][2]["ttt_rmw_GB_token"]    # 6.4425
    Sstar_r  = E13["panelA_S_star"][2]["S_star_read_tokens"]  # 65471
    Sstar_w  = E13["panelA_S_star"][2]["S_star_rmw_tokens"]   # 130942

    Sfine = np.logspace(np.log10(2048), np.log10(2_200_000), 200)
    kv_fine = np.interp(np.log10(Sfine), np.log10(S), np.log10(kv_gb))
    kv_fine = 10**kv_fine
    ax.loglog(Sfine, kv_fine, color=C_KV, lw=2.2, label="KV read (append-once/read-many)  grows ∝ S")
    ax.scatter(S, kv_gb, color=C_KV, zorder=5, s=28)
    ax.axhline(ttt_read, color=C_TTT, lw=2.0, ls="--", label="TTT read half  (const in S)")
    ax.axhline(ttt_rmw,  color=C_TTT, lw=2.4, label="TTT full RMW = read+write  (const in S)")
    for Sx, lab, y in [(Sstar_r, "S*ᵣₑₐd\n65k", ttt_read), (Sstar_w, "S*ᵣₘw\n131k", ttt_rmw)]:
        ax.axvline(Sx, color=C_MARK, ls=":", lw=1.3)
        ax.scatter([Sx], [y], color=C_MARK, s=55, zorder=6, marker="D")
        ax.annotate(lab, (Sx, y), textcoords="offset points", xytext=(6, 8),
                    fontsize=8.5, color="#333", ha="left")
    ax.set_xlabel("context length  S  (tokens)")
    ax.set_ylabel("per-token traffic  (GB/token)")
    ax.set_title("(a) KV vs TTT traffic crossover — anchor 1.3B\n"
                 "below S* KV is cheaper; above S* TTT wins", fontsize=10)
    ax.legend(fontsize=7.7, loc="upper left")
    ax.grid(True, which="both", alpha=0.2)

    # right: S* read-crossover scaling with model size (E3 closed form)
    ss = E3["sweeps"]["S_star"]["rows"]
    confs = [(r["d"], r["m"], r["kv_dim_1024_bf16"]) for r in ss if r["m"] == 16]
    ds = [c[0] for c in confs]; sstar = [c[2] for c in confs]
    ax2.loglog(ds, sstar, "o-", color=C_TTT, lw=2, ms=6, label="S* (GQA-8, bf16, m=16)")
    ssf8 = [r["kv_dim_1024_fp8KV"] for r in ss if r["m"] == 16]
    ax2.loglog(ds, ssf8, "s--", color=C_KV, lw=1.6, ms=5, label="S* (fp8 KV → 2×)")
    for d, s in zip(ds, sstar):
        ax2.annotate(f"{s//1024}k", (d, s), textcoords="offset points", xytext=(4, -12), fontsize=8)
    ax2.set_xlabel("hidden dim  d  (∝ model size)")
    ax2.set_ylabel("crossover context  S*  (tokens)")
    ax2.set_title("S* grows ∝ m·d² : bigger memory ⇒ TTT wins\n"
                  "only at longer context", fontsize=10)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, which="both", alpha=0.2)
    fig.text(0.5, -0.02,
        "Source: E1.3 (hatir) + E3 (closed form), agree <1%. EXPLORATION-GRADE: crossover POSITION is the claim, "
        "not absolute GB (papers publish no H100 wall-clock).",
        ha="center", fontsize=7.3, color="#555")
    return save(fig, "exp-a-kv-ttt-crossover.png")

# =============================================================================
# (b) state placement: energy/latency device tradeoff + residency crossover
# =============================================================================
def fig_b():
    dev = E12["Q1_device_ranking"]["devices"]
    order = ["HBM3 3.35TB/s 7pJ (baseline)", "PIM in-bank 8.2TB/s 3.9pJ",
             "on-die SRAM 10TB/s 0.4pJ 50MB", "sw-scratchpad 60TB/s 0.9pJ 256MB"]
    short = {"HBM3 3.35TB/s 7pJ (baseline)": "HBM3\n(off-die,\nbaseline)",
             "PIM in-bank 8.2TB/s 3.9pJ": "PIM\nin-bank*",
             "on-die SRAM 10TB/s 0.4pJ 50MB": "on-die SRAM\n(50MB,\nspills)",
             "sw-scratchpad 60TB/s 0.9pJ 256MB": "sw-scratchpad*\n(268MB,\nfits)"}
    en = [dev[k]["energy_uJ"] for k in order]
    tm = [dev[k]["time_us"]  for k in order]
    fits = [dev[k].get("fits_one_layer", None) for k in order]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))
    # left: scatter energy vs time, log-log; annotate wins vs HBM
    cols = [C_KV, "#8e44ad", C_ACC, C_TTT]
    # per-device label offsets + alignment so no label overlaps a point or the title
    off = {"HBM3 3.35TB/s 7pJ (baseline)":        ((-10, -6), "right", "top"),
           "PIM in-bank 8.2TB/s 3.9pJ":           ((10, 6),   "left",  "bottom"),
           "on-die SRAM 10TB/s 0.4pJ 50MB":       ((10, -4),  "left",  "top"),
           "sw-scratchpad 60TB/s 0.9pJ 256MB":    ((10, 10),  "left",  "bottom")}
    for k, e, t, c in zip(order, en, tm, cols):
        axL.scatter(t, e, s=140, color=c, zorder=5, edgecolor="white", lw=1)
        (dx, dy), ha, va = off[k]
        axL.annotate(short[k], (t, e), textcoords="offset points", xytext=(dx, dy),
                     fontsize=8, color="#222", ha=ha, va=va)
    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlim(3.5, 320)          # room for right/left labels
    axL.set_ylim(1.1e2, 3.2e3)      # headroom so the HBM3 label clears the title
    axL.set_xlabel("time per 134MB-layer RMW  (µs)")
    axL.set_ylabel("energy per RMW  (µJ)")
    hb = dev[order[0]]
    axL.annotate(f"scratchpad vs HBM (when resident):\n6.8× energy · 14.9× time",
                 (tm[3], en[3]), textcoords="offset points", xytext=(10, -38),
                 fontsize=8, color=C_TTT,
                 arrowprops=dict(arrowstyle="->", color=C_TTT, lw=1))
    axL.set_title("(b) State placement: energy/latency tradeoff\n134MB per-layer RMW, 4 device twins", fontsize=10)
    axL.grid(True, which="both", alpha=0.2)

    # right: residency crossover -- layers-that-fit vs scratchpad capacity
    grid = E12["Q2_residency_crossover"]["grid"]
    caps = [g["cap_MB"] for g in grid]
    configs = ["Titans-170M", "Titans-340M", "Titans-760M", "neural-mem-1.3B", "hypo-7B"]
    ccol = plt.cm.viridis(np.linspace(0.1, 0.85, len(configs)))
    for conf, c in zip(configs, ccol):
        y = [g["cells"][conf]["layers_that_fit"] for g in grid]
        axR.plot(caps, y, "o-", color=c, lw=1.8, ms=5, label=conf)
    axR.axvline(268.4, color=C_MARK, ls=":", lw=1.4)
    axR.annotate("novel scratchpad\n268MB", (268.4, axR.get_ylim()[1]*0.8),
                 fontsize=7.5, color=C_MARK, ha="left")
    axR.axhline(1, color="#aaa", lw=0.8, ls="--")
    axR.set_xscale("log")
    axR.set_xlabel("on-chip capacity  (MB)")
    axR.set_ylabel("# whole layers of state that fit")
    axR.set_title("Residency crossover: 268MB holds one layer\n"
                  "up to 1.3B (d=2048), spills by 7B (d=4096)", fontsize=10)
    axR.legend(fontsize=7.3, loc="upper left")
    axR.grid(True, which="both", alpha=0.2)
    fig.text(0.5, -0.02,
        "Source: E1.2 (hatir). *PIM & scratchpad twins are provenance=analytical, simulation_ready=False → "
        "DIRECTIONAL DSE, not device claims. Ratios/crossover only.",
        ha="center", fontsize=7.3, color="#555")
    return save(fig, "exp-b-state-placement.png")

# =============================================================================
# (c) chunkwise AI(C) curve on the host roofline  (E2.1)
# =============================================================================
def fig_c():
    rh = E21["roofline_host"]
    peak, bw_gbps, ridge = rh["peak_gflops"], rh["peak_gbps"], rh["ridge_flop_per_byte"]
    curve = E21["curve"]
    ds = sorted({c["d"] for c in curve})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))

    # roofline backbone (AI vs attainable GFLOP/s)
    ai = np.logspace(-0.2, 3.1, 300)
    roof = np.minimum(peak, ai * bw_gbps)
    for ax in (axL, axR):
        ax.loglog(ai, roof, color="#222", lw=1.6, label="host roofline")
        ax.axvline(ridge, color=C_MARK, ls=":", lw=1.3)
        ax.axhline(peak, color="#999", ls="--", lw=0.8)
    axL.annotate(f"ridge {ridge:.0f} FLOP/B", (ridge, peak*0.25), rotation=90,
                 fontsize=7.8, color=C_MARK, va="center")

    dcol = {512: C_KV, 1024: "#e67e22", 2048: C_TTT}
    for d in ds:
        pts = [c for c in curve if c["d"] == d]
        x = [p["ai_ch10_flop_per_byte"] for p in pts]
        y = [p["measured_gflops"] for p in pts]
        axL.loglog(x, y, "o-", color=dcol[d], lw=1.5, ms=4.5, label=f"d={d} measured")
        # mark C=1 (decode regime) and C=32 (crossover)
        for p in pts:
            if p["C"] == 1:
                axL.scatter(p["ai_ch10_flop_per_byte"], p["measured_gflops"],
                            s=90, facecolor="none", edgecolor=dcol[d], lw=1.8, zorder=6)
    axL.annotate("C=1 sequential\n(= TTT decode regime)\nAI≈1, memory-bound",
                 (1.0, 4.7), textcoords="offset points", xytext=(14, -6),
                 fontsize=8, color="#333")
    axL.set_xlabel("arithmetic intensity  AI(C)  (FLOP/byte)")
    axL.set_ylabel("measured throughput  (GFLOP/s)")
    axL.set_title("(c) Chunkwise scan on the host roofline\nmeasured GFLOP/s vs AI(C)", fontsize=10)
    axL.legend(fontsize=7.5, loc="lower right")
    axL.grid(True, which="both", alpha=0.2)

    # right: measured GFLOP/s vs C (the algorithmic knob), C* marker
    for d in ds:
        pts = sorted([c for c in curve if c["d"] == d], key=lambda p: p["C"])
        axR2x = [p["C"] for p in pts]; y = [p["measured_gflops"] for p in pts]
        axR.loglog(axR2x, y, "o-", color=dcol[d], lw=1.6, ms=4.5, label=f"d={d}")
    Cstar = E21["crossover_C_star"]["2048"]["C_star_discrete"]
    axR.axvline(Cstar, color=C_MARK, ls=":", lw=1.4)
    axR.annotate(f"C*≈{Cstar} (host)\nmemory→compute", (Cstar, 8), fontsize=8, color=C_MARK, ha="left")
    axR.set_xlabel("chunk size  C  (tokens)")
    axR.set_ylabel("measured throughput  (GFLOP/s)")
    axR.set_title("Growing C moves the SAME algorithm\nfrom memory-bound to compute-bound", fontsize=10)
    axR.legend(fontsize=8, loc="lower right")
    axR.grid(True, which="both", alpha=0.2)
    fig.text(0.5, -0.02,
        "Source: E2.1 (CPU numpy). SHAPE-only: host ridge 34 FLOP/B sits ~9× below the H100 twin ridge 295; "
        "H100 C* is ~306-430 (E3). Curve shape transports, absolute C* does not.",
        ha="center", fontsize=7.3, color="#555")
    return save(fig, "exp-c-chunk-roofline.png")

# =============================================================================
# (d) frequency-tier placement table  (E1.4)
# =============================================================================
def fig_d():
    rows = E14["placement_table"]
    tiers = ["SRAM", "HBM3", "DDR5", "CXL"]  # hot -> cold
    tier_lab = ["SRAM*\n60TB/s", "HBM3\n3.35TB/s", "DDR5\n0.4TB/s", "CXL\n0.5TB/s"]
    lvl_lab = {
        "fast-weights": "fast-weights\nR+W / token",
        "CMS-chunk": "CMS-chunk\nR/tok, W/64",
        "CMS-mid": "CMS-mid\nR+W / 4096",
        "sleep-experts": "sleep-experts\nR+W / 262k",
        "frozen-weights": "frozen-weights\nR/tok, W never",
    }
    levels = [r["level"] for r in rows]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ncol, nrow = len(tiers), len(levels)
    for i, r in enumerate(rows):          # rows top->bottom
        yi = nrow - 1 - i
        rec = r["recommended_tier"]
        for j, t in enumerate(tiers):
            cell = r["tiers"][t]
            adm = cell["read_admissible"]
            fitc = cell["fits_capacity"]
            amort = cell["amort_write_energy_uJ_per_tok"]
            # color: recommended green; admissible-but-not-chosen pale; inadmissible grey; no-capacity hatched
            if t == rec:
                fc = "#1e8449"; tc = "white"
            elif adm and fitc:
                fc = "#a9dfbf"; tc = "#154360"
            elif not fitc:
                fc = "#e5e7e9"; tc = "#909497"
            else:
                fc = "#f2f4f4"; tc = "#b3b6b7"
            ax.add_patch(plt.Rectangle((j, yi), 1, 1, facecolor=fc, edgecolor="white", lw=2))
            if not fitc:
                txt = "no cap"
            else:
                txt = f"{amort:.2f}\nµJ/tok" if amort >= 0.01 else f"{amort:.3f}\nµJ/tok"
            mark = "★ " if t == rec else ""
            ax.text(j+0.5, yi+0.5, mark+txt, ha="center", va="center", fontsize=7.6, color=tc)
    ax.set_xlim(0, ncol); ax.set_ylim(0, nrow)
    ax.set_xticks([j+0.5 for j in range(ncol)]); ax.set_xticklabels(tier_lab, fontsize=8.5)
    ax.set_yticks([nrow-1-i+0.5 for i in range(nrow)])
    ax.set_yticklabels([lvl_lab.get(l, l) for l in levels], fontsize=8.5)
    ax.set_title("(d) Update cadence → memory-tier assignment (★ = recommended)\n"
                 "amortised write energy per token; gated by the FASTER of read/write cadence", fontsize=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # legend
    from matplotlib.patches import Patch
    leg = [Patch(fc="#1e8449", label="recommended tier"),
           Patch(fc="#a9dfbf", label="admissible (not cheapest gated)"),
           Patch(fc="#e5e7e9", label="capacity overflow")]
    ax.legend(handles=leg, fontsize=7.6, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    fig.text(0.5, -0.10,
        "Source: E1.4 (hatir). *SRAM tier = novel scratchpad twin (simulation_ready=False, directional). "
        "Sub-per-token cadence lets a write-heavy RMW state legally drop to DDR/CXL without lengthening the critical path.",
        ha="center", fontsize=7.2, color="#555")
    return save(fig, "exp-d-frequency-tiers.png")

# =============================================================================
# (e) on-die/off-die RMW bandwidth cliff  (E2.2)  -- bonus, complements (b)
# =============================================================================
def fig_e():
    sw = E22["sweeps"]["rmw_single_1T"]
    ro = E22["sweeps"]["read_only_1T"]
    cache = E22["host"]["cache"]
    def series(s):
        x = np.array([r["real_bytes"]/1e6 for r in s])  # MB
        g = np.array([(r.get("gbps_corrected") or r["gbps"]) for r in s], float)
        dom = np.array([r.get("overhead_dominated", False) for r in s])
        return x, g, dom
    x, g, dom = series(sw)
    xr, gr, _ = series(ro)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(x[~dom], g[~dom], "o-", color=C_TTT, lw=2, ms=5, label="RMW (read+write, 1 thread)")
    ax.plot(xr[~dom], gr[~dom], "s--", color=C_KV, lw=1.6, ms=4.5, label="read-only reference")
    # cache boundaries
    for cap, lab in [(cache["L2_per_core_bytes"]/1e6, "L2 1MB"),
                     (cache["L3_shared_bytes"]/1e6, "L3 16MB")]:
        ax.axvline(cap, color=C_MARK, ls=":", lw=1.3)
        ax.annotate(lab, (cap, ax.get_ylim()[1]*0.9 if lab=="L2 1MB" else 250),
                    fontsize=8, color=C_MARK, rotation=90, va="top", ha="right")
    pl = E22["plateaus"]["rmw_single_1T"]
    ax.axhline(pl["incache_peak_gbps"], color="#aaa", ls="--", lw=0.8)
    ax.axhline(pl["DRAM_gbps"], color="#aaa", ls="--", lw=0.8)
    ax.annotate(f"in-cache ≈{pl['incache_peak_gbps']:.0f} GB/s", (x[3], pl["incache_peak_gbps"]),
                fontsize=8, color="#555", xytext=(0,4), textcoords="offset points")
    ax.annotate(f"DRAM ≈{pl['DRAM_gbps']:.0f} GB/s", (x[-4], pl["DRAM_gbps"]),
                fontsize=8, color="#555", xytext=(0,-14), textcoords="offset points")
    ax.annotate(f"{pl['incache_peak_over_DRAM']:.1f}× cliff\n(on-die vs off-die)",
                (18, (pl['incache_peak_gbps']+pl['DRAM_gbps'])/2), fontsize=8.5, color=C_TTT,
                ha="left", va="center",
                arrowprops=None)
    ax.set_xscale("log")
    ax.set_xlabel("state size  (MB)")
    ax.set_ylabel("effective bandwidth  (GB/s)")
    ax.set_title("(e) RMW bandwidth cliff at the on-die/off-die boundary\n"
                 "local analogue of the E1.2 scratchpad fit→spill", fontsize=10)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(True, which="both", alpha=0.2)
    fig.text(0.5, -0.02,
        "Source: E2.2 (CPU numpy). SHAPE-only: the 3.9× L3→DRAM cliff structure transports; GB/s values do not "
        "(H100 on/off-die ridge ~100× elsewhere). RMW ≈0.5× read throughput = the write-back tax KV avoids.",
        ha="center", fontsize=7.3, color="#555")
    return save(fig, "exp-e-rmw-cliff.png")

if __name__ == "__main__":
    paths = [fig_a(), fig_b(), fig_c(), fig_d(), fig_e()]
    print("\nRENDERED:", len(paths))
    for p in paths:
        print(" ", p)
