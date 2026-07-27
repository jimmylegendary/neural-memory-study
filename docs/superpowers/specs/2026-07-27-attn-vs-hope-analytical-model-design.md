# Full Attention + MoE vs HOPE: One-Block Analytical Model Design

**Status:** Approved for implementation by the user's 2026-07-27 resume directive.

## 1. Goal

Build an auditable, parameter-driven analytical model for:

1. one full-attention Transformer block with a top-k MoE FFN; and
2. one HOPE block, modeled as self-modifying Titans followed by a sequential Continuum Memory System (CMS).

The model covers inference prefill and autoregressive decode. It reports FLOPs, effective HBM reads and writes, persistent state, arithmetic intensity, compute latency, memory latency, and stagewise roofline latency. The canonical local implementation must remain useful without Google Sheets access; a connected Desktop session will import or apply the verified workbook source to the user's existing `Attn vs HOPE` Sheet.

## 2. Honesty Contract

- One multiply-accumulate counts as two FLOPs.
- Latencies are analytical roofline lower bounds, not measured wall-clock performance.
- Effective HBM bytes count traffic crossing the HBM boundary. On-chip/register/shared-memory traffic is excluded.
- Kernel fusion removes HBM traffic only when the producer-consumer intermediate is actually kept on chip.
- FlashAttention removes the quadratic score/probability HBM materialization. It does **not** remove the mandatory attention-output write, KV-cache append, or later consumer traffic.
- Classical Flash-Decoding uses split-local partial outputs and log-sum-exp values followed by a reduction. Those temporary global-memory writes/reads are modeled when `decode_splits > 1`.
- HOPE's public paper does not define a paper-exact GPU implementation, deep-MLP DGD primitive, gate output width, complete momentum placement, or decode update schedule. These are explicit inputs or scenarios, never hidden constants.
- CMS read cadence and update cadence are distinct. The paper's sequential CMS reads every enabled level every token even when a level updates rarely.
- Distributed communication is outside the baseline roofline. Tensor/expert/context parallelism may scale local FLOPs and bytes, but collectives require a separate network model.

## 3. Architecture

The implementation has three independent layers.

### 3.1 Analytical engine

`experiments/E5-attn-vs-hope/model.py` contains typed input records, stage records, primitive cost functions, Attention+MoE builders, HOPE builders, and roofline aggregation. Every stage exposes:

`phase, component, stage, executions, flops, hbm_read_bytes, hbm_write_bytes, mandatory_write_bytes, temporary_write_bytes, persistent_state_delta_bytes, compute_efficiency, bandwidth_efficiency, notes`.

The engine returns both:

- `stagewise_latency = sum(max(stage_compute, stage_memory) + launch)`, the primary sequential lower bound; and
- `aggregate_latency = max(sum(FLOPs)/P_eff, sum(bytes)/BW_eff)`, an optimistic diagnostic.

### 3.2 Reproducible experiment output

`run.py` emits:

- `results.json`, containing inputs, every stage, totals, crossovers, and QA checks;
- `stdout.txt`, a compact human-readable report; and
- workbook handoff data consumed by the Desktop builder.

Reference scenarios include prefill and decode on the repository's H100 anchor, plus a legacy HOPE proxy cross-check.

### 3.3 Spreadsheet builder

`sheet/build_workbook.mjs` is a Google Sheets-ready `.xlsx` builder using only `@oai/artifact-tool`. It creates:

- `00_Guide`
- `01_Inputs`
- `02_HW`
- `10_AttnMoE_Prefill`
- `11_AttnMoE_Decode`
- `20_HOPE_Prefill`
- `21_HOPE_Decode`
- `30_Compare`
- `40_Sweeps`
- `90_QA`
- `99_Sources`

All derived cells are formulas. The local CLI currently lacks the required artifact-tool runtime, so this source is verified structurally here and executed/rendered in the connected Desktop session. `DESKTOP-HANDOFF.md` gives exact commands, expected paths, inspections, renders, and Google Sheet transfer steps.

The [native one-tab Sheet archive](../../../research/attn-vs-hope/README.md) is the user-reviewed operational model. The [generated 11-tab workbook](../../../experiments/E5-attn-vs-hope/sheet/Attn-vs-HOPE.xlsx) is its reproducible analytical companion and must remain a separate artifact. The [Google meeting package](../../../research/google-meeting/README.md) links questions and device follow-ups to both roles without treating them as interchangeable.

## 4. Shared Variables and Roofline

Let:

- \(B\): request batch
- \(Q\): query tokens per sequence in this invocation
- \(K\): keys available per query sequence
- \(N=BQ\): processed tokens
- \(D\): model width
- \(b_a,b_w,b_{kv},b_s,b_o\): bytes per activation, static weight, KV element, mutable state, and optimizer state
- \(P_{\mathrm{peak}}\): hardware peak FLOP/s
- \(BW_{\mathrm{HBM}}\): hardware HBM byte/s
- \(\epsilon_{c,i},\epsilon_{b,i}\): stage-specific compute and bandwidth efficiencies

For stage \(i\):

\[
P_{\mathrm{eff},i}=\epsilon_{c,i}P_{\mathrm{peak}},
\qquad
BW_{\mathrm{eff},i}=\epsilon_{b,i}BW_{\mathrm{HBM}}
\]

\[
t_{\mathrm{comp},i}=\frac{F_i}{P_{\mathrm{eff},i}},
\qquad
t_{\mathrm{mem},i}=\frac{R_i+W_i}{BW_{\mathrm{eff},i}}
\]

\[
AI_i=\frac{F_i}{R_i+W_i},
\qquad
t_{\mathrm{roof},i}=\max(t_{\mathrm{comp},i},t_{\mathrm{mem},i})+t_{\mathrm{launch},i}
\]

`read_latency` and `write_latency` are explanatory decompositions. Because reads and writes share the HBM interface, the stage memory lower bound is `(read + write) / bandwidth`, not `max(read, write)`.

## 5. Full Attention + MoE

### 5.1 Shapes

\[
D_q=H_qd_h,\qquad D_{kv}=H_{kv}d_h
\]

`D_kv` is the width of **one** K tensor (and one V tensor), never the combined K+V width.

Attention pair count is:

\[
\Pi =
\begin{cases}
B H_q QK, & \text{general/full}\\
B H_q Q(Q+1)/2, & \text{causal self-prefill}
\end{cases}
\]

### 5.2 Projection and attention FLOPs

\[
F_{QKV}=2ND(D_q+2D_{kv})
\]

\[
F_{\mathrm{attn}}=4\Pi d_h+c_{\mathrm{softmax}}\Pi
\]

\[
F_O=2ND_qD
\]

The FlashAttention core writes no \(QK^\top\) scores or softmax probabilities to HBM. Its required traffic still includes Q and K/V reads and the final O write:

\[
R_{\mathrm{FA}}
=ND_qb_a
+2BKD_{kv}b_{kv}\,m_{\mathrm{KV}}
\]

\[
W_{\mathrm{FA}}=ND_qb_a
\]

where \(m_{\mathrm{KV}}\) is an explicit effective HBM reload multiplier. `1.0` is the ideal one-pass lower bound; tile/L2 behavior may make the measured value larger.

Q materialization and O materialization are separate fusible traffic terms. KV append remains:

\[
W_{\mathrm{KV,append}}=2B QD_{kv}b_{kv}.
\]

### 5.3 Flash-Decode temporary traffic

For \(G>1\) KV splits, the split kernels emit one partial output and one log-sum-exp scalar per query head and split:

\[
B_{\mathrm{partial}}
=B QH_qG(d_hb_{\mathrm{partial}}+b_{\mathrm{LSE}}).
\]

The reduction reads these partials and writes the final output. For \(G=1\), partial traffic is zero.

### 5.4 MoE

For \(E\) experts, top-k \(r\), expert hidden width \(F\), and \(m_{\mathrm{ffn}}\) expert matrices (`3` for SwiGLU, `2` for a conventional two-matrix FFN):

\[
F_{\mathrm{router}}=2NDE+c_{\mathrm{router}}NE
\]

\[
F_{\mathrm{experts}}=2N r\,m_{\mathrm{ffn}}DF.
\]

If the number of active unique experts is not supplied, the uniform-routing expectation is:

\[
A=E\left(1-\left(1-\frac1E\right)^{Nr}\right).
\]

The ideal effective expert-weight read is:

\[
R_{\mathrm{expert\ weights}}
=A\,m_{\mathrm{ffn}}DFb_w\,f_{\mathrm{HBM}},
\]

where \(f_{\mathrm{HBM}}\) is the fraction served from HBM after cache/residency effects. Dispatch, SwiGLU intermediate, down-projection, and combine traffic are separate stages so fusion assumptions remain visible.

Persistent per-block KV state is:

\[
S_{\mathrm{KV}}=2BKD_{kv}b_{kv}.
\]

## 6. HOPE

### 6.1 Shipped-credible default

The default follows the paper prose:

- static \(q=xW_q\);
- mutable \(\{M_k,M_v,M_\eta,M_\alpha,M_{\mathrm{mem}}\}\);
- self-modifying Titans followed by sequential CMS.

The paper's update-set equations and ablation also mention \(M_q\), so `adaptive_q` is an explicit optional variant rather than silently chosen.

### 6.2 Memory primitive

For memory \(r\) with input \(I_r\), hidden width \(H_r\), and output \(O_r\):

\[
P_r=I_rH_r+H_rO_r
\]

\[
F^{\mathrm{fwd}}_r(n)=2nP_r+n(c_\sigma H_r+c_{\mathrm{res}}O_r).
\]

The weight-gradient backward cost without an input gradient is:

\[
F^{\mathrm{bwd,W}}_r(n)=2nI_rH_r+4nH_rO_r.
\]

Because deep-MLP DGD is underspecified, the implementation reports both the explicit `target + prediction + weight backward + DGD-extra` decomposition and a calibrated multiplier:

\[
F^{\mathrm{update}}_r=\mu_r F^{\mathrm{fwd}}_r.
\]

The repository's legacy proxy uses \(\mu_r=4\).

### 6.3 Mutable state traffic

For \(m_r\) parameter-sized momentum/optimizer buffers and one update event:

\[
R^{\mathrm{apply}}_r
=B(1+m_r)P_rb_s,
\qquad
W^{\mathrm{apply}}_r
=B(1+m_r)P_rb_s.
\]

Forward/update-compute weight reloads are controlled separately by `state_weight_read_count`. Fast weights are request-local, so decode traffic scales with \(B\); they do not receive shared-weight batch amortization.

The paper uses two update chunks:

- \(C_{\mathrm{mem}}\) for \(M_{\mathrm{mem}}\);
- \(C_{\mathrm{aux}}\) for \(M_k,M_v,M_\eta,M_\alpha\) (and optional \(M_q\)).

At phase position \(p\), the number of boundaries crossed by \(Q\) tokens is:

\[
U_r=
\left\lfloor\frac{p+Q}{C_r}\right\rfloor
-\left\lfloor\frac{p}{C_r}\right\rfloor.
\]

### 6.4 Decode timing

HOPE decode reports:

- normal-token latency;
- boundary/update-event latency; and
- amortized latency.

\[
t_{\mathrm{avg}}
=\frac{(C_r-1)t_{\mathrm{normal}}+t_{\mathrm{boundary}}}{C_r}.
\]

Two schedules remain explicit:

- eager gradient: target/prediction/backward every token, apply at the boundary;
- deferred replay: retain token/activation data, batch update work at the boundary.

The paper publishes neither an official decode kernel nor wall-clock trace, so neither schedule is labeled canonical.

### 6.5 CMS

For level \(\ell\):

\[
P_\ell=I_\ell H_\ell+H_\ell O_\ell,
\qquad
F^{\mathrm{CMS,fwd}}_\ell=2N P_\ell.
\]

The default read period is one token. Update period \(C_\ell\), BPTT span \(G_\ell\), and optimizer slots \(m_\ell\) are independent:

\[
F^{\mathrm{CMS,grad}}_\ell
=\kappa_\ell 2BG_\ell P_\ell
\]

\[
R^{\mathrm{CMS,apply}}_\ell
=s_\ell(1+m_\ell)P_\ell b_s,
\qquad
W^{\mathrm{CMS,apply}}_\ell
=s_\ell(1+m_\ell)P_\ell b_s,
\]

where \(s_\ell=1\) for shared/frozen state and \(s_\ell=B\) for personalized mutable state.

Persistent HOPE state is context-length independent but batch dependent:

\[
S_{\mathrm{HOPE/request}}
=\sum_r P_r(b_s+m_rb_o)
+\sum_\ell P_\ell(b_s+m_\ell b_o)
+S_{\mathrm{chunk\ cache}}.
\]

## 7. Required Scenarios

The output must distinguish:

1. `paper_equation_lower_bound`: static q, no momentum slots;
2. `shipped_credible_momentum`: static q with configurable Titans-style momentum;
3. `legacy_repo_proxy`: exact algebraic reproduction of `multiarch/hope_block_dag.py`;
4. `adaptive_q_hypothetical`: optional sixth adaptive memory.

M3 and Delta Momentum are sensitivity variants, not shipped-HOPE facts.

## 8. Workbook Design

Each stage table includes:

`Order | Phase | Component | Stage | Symbolic equation | Executions/cadence | FLOPs | Required read B | Mandatory write B | Temporary/fusible write B | Effective read B | Effective write B | Persistent state B | AI | Compute ms | HBM ms | Roofline ms | Bound | Evidence/assumption`.

The comparison sheet shows prefill and decode:

- total FLOPs;
- HBM read/write/total;
- stagewise and aggregate latency;
- arithmetic intensity and ridge classification;
- KV/state footprint and HBM-fit status;
- Attention/HOPE ratios;
- normal vs boundary HOPE decode;
- context-length and batch crossovers.

Charts are limited to:

1. stagewise latency comparison;
2. roofline scatter; and
3. context-length sweep of decode latency and persistent state.

## 9. Validation

Automated tests must prove:

- K and V both contribute factor two to KV traffic;
- causal self-prefill pair count is \(Q(Q+1)/2\);
- FlashAttention score/probability write is zero while O write remains;
- FlashDecode split temporary bytes are zero at one split and positive above one;
- decode KV read is linear in context; KV append is context independent;
- MoE active-weight bytes depend on unique active experts, not token count times full expert weights;
- HOPE state is independent of context and linear in batch;
- CMS forward reads are not divided by update cadence;
- stagewise roofline is no smaller than the aggregate optimistic bound;
- the legacy HOPE closed form reproduces:

\[
F=194Cd^2+20Cd+64d^2+8d,
\qquad
B_{\mathrm{HBM}}=b(81d^2+8d);
\]

- at \(d=2048,C=1,b=2\), the legacy proxy yields `1,082,187,776 FLOPs` and `679,510,016 bytes`;
- workbook formulas contain no hardcoded hardware/model constants in calculation ranges and every cross-sheet reference is quoted.

## 10. Deliverables

- design and implementation plan;
- tested analytical engine;
- reproducible reference results and report;
- workbook builder source and formula map;
- Desktop execution/Google Sheets handoff;
- fresh repository verification;
- committed, merged, and pushed changes, preserving unrelated files.
