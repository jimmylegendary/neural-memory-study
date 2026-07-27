# Attn-vs-HOPE workbook formula map

This document maps the generated 11-tab workbook to the verified E5 engine,
the approved analytical design, and the workbook cells that hold each live
assumption. The workbook is a formula-driven companion: `results.json` supplies
raw inputs, stage order, provenance, and QA references; calculation values are
reconstructed by spreadsheet formulas.

## Workbook-wide roofline convention

The four stage sheets use the design-section 8 schema in columns `A:S`:

| Column | Meaning | Workbook formula / lineage |
|---|---|---|
| A | Order | Formula-generated row order |
| B:E | Phase, component, stage, symbolic equation | Text metadata; stage order is asserted against `results.json` during build |
| F | Executions/cadence | Formula from `01_Inputs`, including fractional amortized update cadence |
| G | FLOPs | Formula from the relevant design equation and `01_Inputs` |
| H | Required read B | Per-execution effective HBM read formula |
| I | Mandatory write B | Per-execution mandatory HBM write formula |
| J | Temporary/fusible write B | Per-execution temporary HBM write formula |
| K | Effective read B | `F × H` |
| L | Effective write B | `F × (I + J)` |
| M | Persistent state B | Formula for current request-resident state; not traffic |
| N | Arithmetic intensity | `(F × G) / (K + L)` when traffic is nonzero |
| O | Compute ms | `F × G / (01_Inputs!compute_efficiency × 02_HW!peak_flops) × 02_HW!ms_per_second` |
| P | HBM ms | `(K + L) / (01_Inputs!bandwidth_efficiency × 02_HW!hbm_bandwidth) × 02_HW!ms_per_second` |
| Q | Roofline ms | `MAX(O, P) + F × 02_HW!launch_seconds × 02_HW!ms_per_second` |
| R | Bound | `compute`, `memory`, or `balanced` from `O` versus `P` |
| S | Evidence/assumption | Visible interpretation of the stage equation and traffic convention |

Every cross-sheet reference in these formulas is single-quoted. The builder
contains no literal H100 or model-dimension values in calculation formulas.

Each stage sheet has the same formula summary in `A2:N3`:

| Cell | Summary |
|---|---|
| A3 | Total FLOPs |
| B3 | Effective HBM read B |
| C3 | Effective HBM write B |
| D3 | Total effective HBM B |
| E3 | Persistent state B |
| F3 | Aggregate arithmetic intensity |
| G3 | Sum of stage compute ms |
| H3 | Sum of stage HBM ms |
| I3 | `MAX(G3,H3)`, the optimistic aggregate roofline |
| J3 | `SUM(stage roofline ms)`, the primary sequential lower bound |
| K3 | Aggregate bound |
| L3 | HBM fit: `E3 <= '02_HW'!$C$8` |
| M3 | Ridge AI: `'02_HW'!$C$6 / '02_HW'!$C$7` |
| N3 | Compute-side or memory-side ridge classification |

## `00_Guide`

- `A4:B13` explains the analytical scope, stagewise versus aggregate latency,
  HBM-fit interpretation, the three approved charts, and rebuild workflow.
- The guide explicitly distinguishes this generated 11-tab workbook from the
  user's native one-tab live Sheet under `research/attn-vs-hope/`.

## `01_Inputs`

### Shared Attention/MoE and HOPE inputs (`A4:F43`)

The live numeric assumption cells are `C5:C43`:

| Cells | Inputs |
|---|---|
| C5:C9 | Batch; prefill Q/K; decode Q/K |
| C10:C13 | Model width `D`; head width `d_h`; query and KV head counts |
| C14:C18 | Activation, weight, KV, partial-output, and LSE bytes |
| C19:C22 | Decode splits, tokens/split, KV reload multiplier, softmax FLOPs/pair |
| C23:C30 | Expert count, top-k, hidden width, matrix count, router cost, route metadata bytes, active-expert override, expert HBM fraction |
| C31:C34 | Compute/bandwidth efficiencies, expert activation FLOPs, fusion flag |
| C35:C39 | Mutable-state/optimizer bytes, sigma/residual FLOPs, chunk-cache bytes |
| C40:C43 | Prefill/decode positions and main/auxiliary memory chunks |

### Mutable memories (`A47:H52`)

Rows `48:52` are `M_k`, `M_v`, `M_eta`, `M_alpha`, and `M_mem`.
Columns `B:H` hold `I`, `H`, `O`, update chunk, momentum slots, update
multiplier `μ`, and state-weight read count.

### CMS levels (`A56:I59`)

Rows `57:59` are `cms_l1:cms_l3`. Columns `B:I` hold `I`, `H`, `O`, update
period, BPTT span, optimizer slots, personalized flag, and gradient multiplier.

### Sweep controls (`A63:G65`)

- `B64:G64`: context sweep points.
- `B65:G65`: batch sweep points.

## `02_HW`

The hardware and unit-conversion source cells are:

| Cell | Value |
|---|---|
| C5 | Hardware name |
| C6 | Peak FLOP/s |
| C7 | HBM byte/s |
| C8 | HBM capacity B |
| C9 | Kernel-launch seconds/execution |
| C10 | Milliseconds/second |
| C11 | Bytes/GiB |
| C12 | FLOPs/TFLOP |

`C6:C9` come from `results.json.hardware`, itself derived from
`multiarch/twins/h100.json`. Unit conversions are visible rather than embedded
as unexplained constants in formulas.

## `10_AttnMoE_Prefill`

- Stage table: `A5:S17`; formula rows `6:17`.
- Query/KV widths: `D_q = H_q d_h`, `D_kv = H_kv d_h`.
- Causal pairs: `Π = B H_q Q(Q+1)/2`.
- QKV FLOPs: `2ND(D_q+2D_kv)`.
- FlashAttention FLOPs: `4Πd_h + c_softmax Π`.
- KV reads: `2BKD_kv b_kv m_KV`; no quadratic score/probability HBM write.
- KV append/state: `2BQD_kv b_kv` / `2BKD_kv b_kv`.
- Output projection: `2ND_qD`.
- Router: `2NDE + c_router NE`.
- Active experts: override from `01_Inputs!C29`, otherwise
  `E[1-(1-1/E)^(Nr)]`.
- Expert matrices: per-matrix `2NrDF`; expert-weight HBM read uses active
  experts and `01_Inputs!C30`.
- Dispatch, SwiGLU intermediate, down projection, and combine traffic remain
  separate so `01_Inputs!C34` controls only eligible fusion traffic.

## `11_AttnMoE_Decode`

- Stage table: `A5:S18`; formula rows `6:18`.
- Decode pairs: `Π = B H_q QK` with `Q=1` from `01_Inputs!C8`.
- Flash-Decode partial bytes for `G>1`:
  `BQH_qG(d_h b_partial + b_LSE)`.
- `flash_decode_reduce` cadence becomes zero when `G=1`; otherwise it reads
  partials and writes the final output.
- Decode KV read is linear in `K`; KV append remains context-independent.
- MoE suffix equations are identical to the prefill sheet at decode token
  count.

## `20_HOPE_Prefill`

- Stage table: `A5:S39`; formula rows `6:39`.
- Static q: `2ND²`, with shared `W_q` read and temporary q output.
- Per-memory parameters: `P = IH + HO` from `01_Inputs!B48:H52`.
- Forward FLOPs:
  `2NP + N(c_sigma H + c_res O)`.
- Weight-gradient FLOPs: `2NIH + 4NHO`.
- Update decomposition: target forward + prediction forward + weight backward
  + `μF_fwd - (2F_fwd + F_bwd,W)`.
- Boundary count:
  `INT((position+Q)/chunk) - INT(position/chunk)`.
- State apply read/write per event:
  `B(1+momentum_slots)P b_state` in each direction.
- CMS forward: `2NP_l`; personalized-state and optimizer-slot state bytes are
  driven by `01_Inputs!B57:I59`.

## `21_HOPE_Decode`

- Timing summary: `A5:E8`.
  - Row 6: normal token, excluding state-apply rows.
  - Row 7: simultaneous boundary, converting fractional apply cadence to one
    event per memory.
  - Row 8: per-memory-period amortized timing.
- Amortized stage table: `A11:S45`; formula rows `12:45`.
- Eager-gradient target/prediction/backward work occurs every token.
- State-apply cadence is `1/update_chunk` for the amortized rows.
- Persistent state is context-independent and batch-dependent:
  memory/CMS parameter state plus optimizer slots and chunk cache.

## `30_Compare`

- `A5:I9`: formula links to the four stage-sheet summaries for stagewise and
  aggregate latency, FLOPs, HBM GiB, AI, state GiB, HBM fit, and bound.
- `A12:C17`: Attention/HOPE ratios plus normal, boundary, and amortized HOPE
  decode timing.
- `K5:M9`: formula-backed AI/effective-TFLOP helper table.
- Native chart 1: stagewise versus aggregate latency.
- Native chart 2: roofline scatter, AI versus effective TFLOP/s.

## `40_Sweeps`

- Context sweep: `A5:F11`; source controls `01_Inputs!B64:G64`.
- Attention context-stage helpers: `H5:T11`.
- Batch sweep: `A16:F22`; source controls `01_Inputs!B65:G65`.
- Attention batch-stage helpers: `H16:T22`.
- Attention state: `2BKD_kv b_kv / bytes_per_GiB`.
- HOPE context latency/state remain constant because the recurrent state does
  not grow with context.
- HOPE batch state scales with request count.
- HOPE batch latency recomputes the shared static-q projection at each batch;
  only request-local stages are scaled from the baseline. This preserves the
  non-batch-linear shared `W_q` HBM read.
- Native chart 3 uses `A5:E11`: context-length decode latency and persistent
  state.

## `90_QA`

- `A5:G21`: 16 formula-versus-engine reconciliation rows.
  - Column B: workbook formula result.
  - Column C: clearly labeled imported engine reference.
  - Column D: formula delta.
  - Column E: tolerance.
  - Column F: formula-generated PASS/REVIEW.
- Checks cover baseline FLOPs/HBM/stagewise latency, all three HOPE decode
  timing modes, both context-sweep endpoints, and B=1/B=32 HOPE batch points.
- `A25:C37`: the 12 upstream boolean invariants imported from
  `results.json.qa_checks`.

Imported numerical outputs are confined to the clearly labeled engine-reference
column on this sheet; calculation, comparison, and sweep sheets contain formulas.

## `99_Sources`

`A5:E13` records repository paths, their role in the workbook, the verified
E5 interface, the H100 twin derivation, the separate live-Sheet archive, and the
plain-text HOPE paper URL.
