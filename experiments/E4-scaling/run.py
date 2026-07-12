"""E4 — scaling-law digitization + fit (dossier angle A3 quantitative backbone).

Part III (ch19) support. NO GPU / no model training: this fits PUBLISHED numbers
(digitized in data.csv, per-row table provenance) with numpy/scipy on CPU.

The three fits the dossier (A3) asks for, each honesty-gated to ONE setting_group
(same paper + tokenizer + data + metric — the only legal fit boundary; the six papers
use DIFFERENT tokenizers/data so cross-paper is ordinal-only):
  (a) ppl vs params, WITHIN one line (Miras Moneta / GDN). Power-law fit + R^2.
      HONEST CONFOUND: params N and tokens D co-scale (15/30/100B) across these points,
      so the fitted exponent is a COMPUTE-scaling descriptor, not a pure param law;
      N and D are NOT separable from 3 points. Reported as descriptive only.
  (b) ppl vs state-bytes AND vs capacity-proxy, CROSS-ARCH at fixed 1.3B (Atlas Table 2).
      Spearman rank corr. Tests whether capacity O(d_k^p) adds predictive power over raw
      state-bytes for PERPLEXITY.
  (c) long-context retention length vs capacity-proxy vs state-bytes (BABILong). Spearman.
      Tests the same on the LONG-CONTEXT axis.

A3 falsification rule (dossier §5): "capacity proxy adds no predictive power over raw
state size" => A3 falsified; we report exactly where it does and does not add power.

State-bytes model (bf16): state_bytes = m * d^2 * L_layer * 2, with m the fast-weight
multiplier (matrix~1, deep-MLP 8, +momentum/2nd-moment 16, +poly features ~24 directional;
attention fast-weight m=0, KV priced separately). All state numbers are EXPLORATION-GRADE
(ordinal/ratio load-bearing, not silicon-accurate) — carried from the E1.x/E3 convention.

Reproduce: /home/jimmy/repos/neural-memory-study/.venv/bin/python run.py
"""
import csv, json, os, sys
import numpy as np
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data.csv")
OUT = os.path.join(HERE, "results.json")
FIG = "/home/jimmy/repos/neural-memory-study/figures/exp-f-scaling-fits.png"
DTYPE_BYTES = 2  # bf16


# ----------------------------------------------------------------------------- IO
def load_rows():
    rows = []
    with open(DATA) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            if line.startswith("source,"):
                header = [h.strip() for h in line.strip().split(",")]
                continue
            parts = [p.strip() for p in line.rstrip("\n").split(",")]
            rows.append(dict(zip(header, parts)))
    # numeric coercions
    for r in rows:
        for k in ("params_M", "tokens_B", "value", "capacity_ordinal",
                  "state_mult_m", "d_model", "L_layer", "context_train"):
            try:
                r[k] = float(r[k])
            except (ValueError, KeyError):
                r[k] = np.nan
    return rows


def group(rows, g):
    return [r for r in rows if r["setting_group"] == g]


def state_bytes(m, d, L):
    if m <= 0 or d <= 0 or L <= 0:
        return np.nan
    return m * d * d * L * DTYPE_BYTES


# --------------------------------------------------------------------- fit helpers
def powerlaw_fit(x, y):
    """Fit y = A * x^(-b) via least squares in log-log; return params, R^2 (on log y)."""
    lx, ly = np.log(x), np.log(y)
    slope, intercept, r, p, se = stats.linregress(lx, ly)
    b = -slope
    A = float(np.exp(intercept))
    return {"A": A, "exponent_b": float(b), "r2_logy": float(r * r),
            "n_points": int(len(x))}


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return {"rho": None, "p": None, "n": int(mask.sum())}
    rho, p = stats.spearmanr(x[mask], y[mask])
    return {"rho": float(rho), "p": float(p), "n": int(mask.sum())}


# =============================================================================
def main():
    rows = load_rows()
    results = {
        "experiment": "E4-scaling",
        "purpose": ("digitize every published (model, params, tokens, context, ppl/score) "
                    "point across the six papers and fit candidate scaling laws; test whether "
                    "Atlas' O(d_k^p) capacity axis adds predictive power over raw state-bytes "
                    "(dossier angle A3, ch19)."),
        "no_gpu": True,
        "host": {"numpy": np.__version__, "scipy": __import__("scipy").__version__,
                 "python": sys.version.split()[0]},
        "dtype_bytes": DTYPE_BYTES,
        "honesty": ("Fits are gated to one setting_group (same paper+tokenizer+data+metric). "
                    "The six papers use different tokenizers/data (Llama-2/FineWeb, T5, 32k), so "
                    "cross-paper comparisons are ORDINAL/directional only. Absolute exponents and "
                    "state-bytes are exploration-grade; orderings and rank-correlations are the "
                    "load-bearing claims."),
    }

    # ---------- state-bytes table for the four families at 340/760/1.3B ----------
    fam = {"matrix (GDN)": 1, "deep-MLP (Titans/TTT)": 8,
           "deep-MLP+momentum (Titans/Atlas)": 16, "deep-MLP+poly (Atlas, directional)": 24,
           "attention (fast-weight)": 0}
    sizes = {"340M": (1024, 24), "760M": (1536, 24), "1.3B": (2048, 24)}
    sb_table = {}
    for fname, m in fam.items():
        sb_table[fname] = {}
        for sname, (d, L) in sizes.items():
            b = state_bytes(m, d, L)
            sb_table[fname][sname] = None if np.isnan(b) else {
                "bytes": float(b), "MB_per_model": round(b / 1e6, 2),
                "MB_per_layer": round(b / L / 1e6, 3) if L else None}
    results["state_bytes_model"] = {
        "formula": "state_bytes = m * d^2 * L_layer * dtype_bytes (bf16=2)",
        "m_by_family": fam,
        "size_configs_d_Llayer": {k: {"d": v[0], "L_layer": v[1]} for k, v in sizes.items()},
        "table": sb_table,
        "capacity_proxy_ordinal": {
            "matrix O(d_k)": 1, "deep_mlp subquadratic": 2,
            "deep_mlp_poly O(d_k^p)": 3, "attention exp-phi* unbounded": 4},
        "note": ("(d,L_layer) are the standard Llama-family shapes for each size "
                 "(12*L*d^2 ~ param count); per-size configs are not published per-model in the "
                 "notes, so state-bytes are directional. Attention has NO fast-weight state "
                 "(m=0); its KV cache grows with context and is priced separately (see E1.3)."),
    }

    # =========================================================================
    # (a) ppl vs params WITHIN a line (Miras). Power law + compute confound.
    # =========================================================================
    a = {"description": "ppl vs params within one line; N and D CO-SCALE (confound flagged)."}
    for g in ("miras_moneta_wikippl", "miras_gdn_wikippl"):
        gr = sorted(group(rows, g), key=lambda r: r["params_M"])
        N = np.array([r["params_M"] for r in gr])
        D = np.array([r["tokens_B"] for r in gr])
        ppl = np.array([r["value"] for r in gr])
        C = 6.0 * (N * 1e6) * (D * 1e9)  # ~ FLOPs (Chinchilla), the axis N and D actually move on
        a[g] = {
            "points": [{"params_M": float(n), "tokens_B": float(d), "wiki_ppl": float(p)}
                       for n, d, p in zip(N, D, ppl)],
            "fit_ppl_vs_params": powerlaw_fit(N, ppl),
            "fit_ppl_vs_compute_6ND": powerlaw_fit(C, ppl),
            "confound": ("tokens scale 15->30->100B alongside params 340->760->1300M, so the "
                         "params-only exponent is NOT identifiable as pure param-scaling; the "
                         "compute (6ND) fit is the honest axis. Reported as descriptive."),
        }
    results["fit_a_ppl_vs_params"] = a

    # =========================================================================
    # (b) ppl vs state-bytes vs capacity, cross-arch @ 1.3B (Atlas Table 2).
    # =========================================================================
    gr = group(rows, "atlas_1p3B_ppl")
    for r in gr:
        r["_state_bytes"] = state_bytes(r["state_mult_m"], r["d_model"], r["L_layer"])
    # parametric-memory family only (well-defined fast-weight state): GDN, Titans, Atlas(++)
    fam_models = {"GatedDeltaNet", "Titans_LMM", "Atlas", "Atlas++"}
    param_fam = [r for r in gr if r["model"] in fam_models]
    attn = [r for r in gr if r["arch_class"] == "attention"]

    def pack(rs):
        return [{"model": r["model"], "arch": r["arch_class"], "wiki_ppl": r["value"],
                 "capacity_ordinal": r["capacity_ordinal"],
                 "state_bytes": (None if np.isnan(r["_state_bytes"]) else float(r["_state_bytes"])),
                 "state_MB": (None if np.isnan(r["_state_bytes"]) else round(r["_state_bytes"]/1e6, 1))}
                for r in rs]

    b = {
        "description": "cross-architecture ppl at fixed 1.3B (same tokenizer/data/tokens).",
        "parametric_family_points": pack(param_fam),
        "attention_points": pack(attn),
        # within parametric family
        "spearman_ppl_vs_statebytes__parametric": spearman(
            [r["_state_bytes"] for r in param_fam], [r["value"] for r in param_fam]),
        "spearman_ppl_vs_capacity__parametric": spearman(
            [r["capacity_ordinal"] for r in param_fam], [r["value"] for r in param_fam]),
        # including the attention corner (state m=0)
        "spearman_ppl_vs_statebytes__incl_attention": spearman(
            [(0.0 if np.isnan(r["_state_bytes"]) else r["_state_bytes"]) for r in param_fam + attn],
            [r["value"] for r in param_fam + attn]),
        "spearman_ppl_vs_capacity__incl_attention": spearman(
            [r["capacity_ordinal"] for r in param_fam + attn],
            [r["value"] for r in param_fam + attn]),
    }
    results["fit_b_ppl_vs_state_capacity"] = b

    # =========================================================================
    # (c) long-context retention vs capacity vs state-bytes (BABILong, directional).
    # =========================================================================
    gr = group(rows, "babilong_retention")
    for r in gr:
        r["_state_bytes"] = state_bytes(r["state_mult_m"], r["d_model"], r["L_layer"])
    param_ret = [r for r in gr if r["arch_class"] != "attention"]  # exclude bounded-ctx attention
    c = {
        "description": ("BABILong sustained-accuracy length (order-of-magnitude bins) vs capacity. "
                        "CROSS-PAPER, ordinal only. Attention (GPT-4) excluded from the parametric "
                        "correlation: its limit is bounded training context (~128-256K), not capacity "
                        "-- the retrieval gap the papers themselves flag."),
        "points": [{"model": r["model"], "arch": r["arch_class"],
                    "retention_tokens": r["value"], "capacity_ordinal": r["capacity_ordinal"],
                    "state_MB": (None if np.isnan(r["_state_bytes"]) else round(r["_state_bytes"]/1e6, 1))}
                   for r in gr],
        "spearman_retention_vs_capacity__parametric": spearman(
            [r["capacity_ordinal"] for r in param_ret], [r["value"] for r in param_ret]),
        "spearman_retention_vs_statebytes__parametric": spearman(
            [r["_state_bytes"] for r in param_ret], [r["value"] for r in param_ret]),
        "titans_vs_atlas_leap": {
            "state_bytes_ratio_atlas_over_titans": None,
            "retention_ratio_atlas_over_titans": None,
        },
    }
    # the diagnostic ratio: Titans->Atlas retention leap vs its modest state-byte increase
    t = next((r for r in gr if r["model"] == "Titans_MAC"), None)
    at = next((r for r in gr if r["model"] == "Atlas_MAC"), None)
    if t and at and np.isfinite(t["_state_bytes"]) and np.isfinite(at["_state_bytes"]):
        c["titans_vs_atlas_leap"]["state_bytes_ratio_atlas_over_titans"] = round(
            at["_state_bytes"] / t["_state_bytes"], 2)
        c["titans_vs_atlas_leap"]["retention_ratio_atlas_over_titans"] = round(
            at["value"] / t["value"], 2)
        c["titans_vs_atlas_leap"]["reading"] = (
            "the ~%.1fx retention leap is far larger than the ~%.1fx state-byte increase, so the "
            "capacity-CLASS change (deep-MLP -> deep-MLP+poly, ordinal 2->3) explains the leap that "
            "raw bytes underpredict." % (
                at["value"] / t["value"], at["_state_bytes"] / t["_state_bytes"]))
    results["fit_c_retention_vs_capacity"] = c

    # =========================================================================
    # Supporting: state COUNT vs ppl within TNT (monotone), and the chunk-mismatch U-curve.
    # =========================================================================
    gr = sorted(group(rows, "tnt_localmem"), key=lambda r: r["value"], reverse=True)
    n_mem = [0, 1, 2, 3, 4]
    ppl_mem = [23.53, 21.04, 20.74, 20.47, 20.15]
    results["support_tnt_localmem_state_count_vs_ppl"] = {
        "description": "within TNT (150M, same setting): more local memories (more state) -> lower avg ppl.",
        "n_local_memories": n_mem, "avg_ppl": ppl_mem,
        "spearman_count_vs_ppl": spearman(n_mem, ppl_mem),
        "note": "monotone; but gain saturates (21.04 -> 20.15 over +1..+4), diminishing returns in state count.",
    }
    gr = group(rows, "tnt_infer_chunk")
    infer_C = [8, 16, 32, 64, 128, 256, 512]
    infer_ppl = [r["value"] for r in gr]
    results["support_tnt_infer_chunk_mismatch"] = {
        "description": ("550M Titans pre-trained at C=64: inference-chunk vs ppl is a U-curve minimised "
                        "at the TRAIN chunk (C=64, ppl 13.78), NOT a scaling law -- train/serve resolution "
                        "mismatch (TNT Challenge 3). Included as a published-data point, not fit."),
        "inference_chunk": infer_C, "ppl": infer_ppl,
        "argmin_chunk": infer_C[int(np.argmin(infer_ppl))], "min_ppl": float(min(infer_ppl)),
    }

    # =========================================================================
    # A3 VERDICT (the honest bottom line).
    # =========================================================================
    b = results["fit_b_ppl_vs_state_capacity"]
    c = results["fit_c_retention_vs_capacity"]
    results["A3_verdict"] = {
        "headline": ("state-capacity earns its keep as a LONG-CONTEXT axis, NOT a perplexity axis. "
                     "A3 is partially supported and sharply scoped -- not falsified, but narrowed."),
        "on_perplexity_axis_b": {
            "within_parametric_family": ("state-bytes and capacity-ordinal are rank-EQUIVALENT and both "
                "perfectly anti-correlate with ppl (rho=%s), so capacity adds NOTHING over raw state-bytes "
                "for ppl." % b["spearman_ppl_vs_capacity__parametric"]["rho"]),
            "including_attention": ("adding the attention corner INVERTS capacity vs ppl (unbounded capacity "
                "but WORST ppl): capacity rho=%s while raw state-bytes (attn=0) stays consistent rho=%s. "
                "=> capacity-proxy is actively MISLEADING for ppl once attention is in the picture; ppl "
                "rewards compression, not raw retrieval capacity." % (
                    b["spearman_ppl_vs_capacity__incl_attention"]["rho"],
                    b["spearman_ppl_vs_statebytes__incl_attention"]["rho"])),
        },
        "on_longcontext_axis_c": ("capacity-ordinal tracks BABILong retention within the parametric family "
            "(rho=%s) and specifically explains the Titans->Atlas retention leap that the modest state-byte "
            "increase underpredicts; this is the ONE axis where capacity adds predictive power over raw "
            "state-bytes. Attention is separately bounded by training context (the retrieval gap)." % (
                c["spearman_retention_vs_capacity__parametric"]["rho"],)),
        "for_ch19": ("propose state-bytes and capacity as first-class scaling axes ONLY for the long-context "
                     "/ retention target, and explicitly WARN against fitting ppl vs capacity cross-architecture "
                     "(measures the wrong thing). Any single-tokenizer cross-arch ppl-vs-params exponent is a "
                     "compute-scaling descriptor with N,D confounded -- not evidence for a state axis."),
    }

    results["caveats"] = [
        "Exploration-grade: rankings, rank-correlations, ratios and crossovers are load-bearing; "
        "absolute exponents and state-bytes are directional (per-size (d,L) assumed standard Llama shapes).",
        "ppl-vs-params fits (a) have N and D co-scaling (15/30/100B); the exponent is a compute-scaling "
        "descriptor, not a pure param law, and is NOT identifiable from 3 points.",
        "Cross-paper points (BABILong retention) use DIFFERENT tokenizers/data/backbones and coarse "
        "order-of-magnitude accuracy bins -- ordinal/directional ONLY, never fit as continuous law.",
        "state_mult m per family is directional (matrix~1, deep-MLP 8, +momentum 16, +poly ~24); attention "
        "fast-weight m=0 with KV priced separately (E1.3).",
        "N is tiny per setting_group (3-8 points), so Spearman rho is used as the honesty-robust statistic "
        "rather than parametric CIs; p-values are indicative only.",
    ]

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("wrote", OUT)

    render_figure(rows, results)
    return results


# ============================================================================= FIGURE
def render_figure(rows, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
        "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    C_A, C_B, C_C, C_ATT, C_MUT = "#2471a3", "#c0392b", "#1e8449", "#7f8c8d", "#8e44ad"
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.2))

    # (a) ppl vs params, within-line power law -----------------------------------
    ax = axs[0, 0]
    for g, col, mk, lab in (("miras_moneta_wikippl", C_A, "o", "Moneta"),
                            ("miras_gdn_wikippl", C_B, "s", "Gated DeltaNet")):
        gr = sorted(group(rows, g), key=lambda r: r["params_M"])
        N = np.array([r["params_M"] for r in gr]); ppl = np.array([r["value"] for r in gr])
        ax.plot(N, ppl, mk, color=col, ms=7, label=lab)
        fit = results["fit_a_ppl_vs_params"][g]["fit_ppl_vs_params"]
        xs = np.linspace(N.min(), N.max(), 50)
        ax.plot(xs, fit["A"] * xs ** (-fit["exponent_b"]), "-", color=col, lw=1.4, alpha=0.8)
        yo = 26 if col == C_A else -20   # Moneta label up, GDN label down (avoid overlap)
        ax.annotate(r"%s: $b=%.2f,\ R^2=%.3f$" % (lab, fit["exponent_b"], fit["r2_logy"]),
                    (N[0], ppl[0]), color=col, fontsize=7.5,
                    xytext=(6, yo), textcoords="offset points", ha="left")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("params (M)  [tokens co-scale 15/30/100B]"); ax.set_ylabel("WikiText ppl")
    ax.set_title("(a) ppl vs params, within Miras (N,D confounded)")
    ax.legend(frameon=False, fontsize=8)

    # (b) ppl vs state-bytes, cross-arch @1.3B -----------------------------------
    ax = axs[0, 1]
    b = results["fit_b_ppl_vs_state_capacity"]
    for pt in b["parametric_family_points"]:
        ax.plot(pt["state_MB"], pt["wiki_ppl"], "o", color=C_C, ms=8)
        ax.annotate(pt["model"], (pt["state_MB"], pt["wiki_ppl"]), fontsize=7.5,
                    xytext=(6, 2), textcoords="offset points", color=C_C)
    # attention points at ~0 fast-weight state (drawn at left edge marker)
    for pt in b["attention_points"]:
        ax.plot(1.0, pt["wiki_ppl"], "^", color=C_ATT, ms=8)
        ax.annotate(pt["model"], (1.0, pt["wiki_ppl"]), fontsize=7.5,
                    xytext=(6, 0), textcoords="offset points", color=C_ATT)
    ax.set_xscale("log")
    ax.set_xlabel("fast-weight state (MB/model, bf16)  |  attention=0 (drawn at 1MB)")
    ax.set_ylabel("WikiText ppl @1.3B")
    rho_par = b["spearman_ppl_vs_capacity__parametric"]["rho"]
    rho_att = b["spearman_ppl_vs_capacity__incl_attention"]["rho"]
    ax.set_title("(b) ppl vs state @1.3B: capacity rho=%.2f (fam) -> %.2f (+attn)" % (rho_par, rho_att))
    ax.text(0.5, 0.04, "more parametric state -> lower ppl (green);\nunbounded-capacity attention "
            "= WORST ppl (grey) -> capacity misleads for ppl",
            transform=ax.transAxes, fontsize=7.3, color="#555", ha="center")

    # (c) retention vs capacity --------------------------------------------------
    ax = axs[1, 0]
    c = results["fit_c_retention_vs_capacity"]
    # stagger points that share (capacity_ordinal, retention) so labels don't collide
    seen = {}
    for pt in c["points"]:
        att = pt["arch"] == "attention"
        col = C_ATT if att else C_C
        key = (pt["capacity_ordinal"], round(np.log10(pt["retention_tokens"]), 1))
        k = seen.get(key, 0); seen[key] = k + 1
        dx = 0.13 * k                     # spread x for co-located points
        dyoff = 6 + 11 * k                # stagger label vertically
        ax.plot(pt["capacity_ordinal"] + dx, pt["retention_tokens"], "^" if att else "o",
                color=col, ms=8)
        ax.annotate(pt["model"], (pt["capacity_ordinal"] + dx, pt["retention_tokens"]),
                    fontsize=7, xytext=(7, dyoff - 6), textcoords="offset points", color=col)
    ax.set_xlim(0.6, 4.7)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["matrix\nO(d_k)", "deep-MLP", "deep-MLP\n+poly", "attention\nunbounded"], fontsize=7.5)
    ax.set_xlabel("capacity-proxy ordinal (Atlas)")
    ax.set_ylabel("BABILong sustained length (tokens)")
    rho_c = c["spearman_retention_vs_capacity__parametric"]["rho"]
    ax.set_title("(c) retention vs capacity: rho=%.2f (parametric)" % rho_c)
    ax.text(0.5, 0.06, "capacity earns its keep HERE (green);\nattention unbounded but ctx-bounded "
            "(grey) = retrieval gap", transform=ax.transAxes, fontsize=7.3, color="#555", ha="center")

    # (d) TNT support: chunk-mismatch U + local-mem monotone ----------------------
    ax = axs[1, 1]
    s = results["support_tnt_infer_chunk_mismatch"]
    ax.plot(s["inference_chunk"], s["ppl"], "o-", color=C_MUT, ms=6, label="infer-chunk ppl (550M)")
    ax.axvline(64, color=C_MUT, ls="--", lw=1, alpha=0.6)
    ax.annotate("train C=64\n(min ppl 13.78)", (64, 13.78), fontsize=7.5, color=C_MUT,
                xytext=(10, 20), textcoords="offset points")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("inference chunk C (train C=64)")
    ax.set_ylabel("ppl")
    ax.set_title("(d) TNT train/serve chunk mismatch (U-curve, not a law)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle("E4 — scaling-law digitization + fits (dossier A3): capacity is a long-context axis, "
                 "not a perplexity axis", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(FIG, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG)


if __name__ == "__main__":
    main()
