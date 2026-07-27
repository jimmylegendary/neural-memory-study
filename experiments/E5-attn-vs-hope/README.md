# E5 — Full Attention + MoE versus HOPE

This directory is the canonical, standard-library numerical engine for the
one-block comparison. It produces analytical roofline lower bounds, not
measured GPU wall-clock results. One multiply-accumulate is two FLOPs, and
effective bytes count traffic that crosses the HBM boundary only.

## Paper facts represented directly

- Full attention uses `D_q = H_q d_h`, `D_kv = H_kv d_h`, causal
  `B H_q Q(Q+1)/2` prefill pairs, and linear `B H_q QK` decode pairs.
- Decode is exactly one next-token invocation (`Q=1`), and the KV HBM reload
  multiplier is a lower-bound multiplier no smaller than one.
- FlashAttention does not materialize the quadratic score or probability
  matrix in HBM. It still accounts for Q/K/V reads and the final output write.
- Split Flash-Decode materializes partial outputs and LSE records only for more
  than one split.
- Shipped HOPE has static `q=xW_q` and mutable
  `{M_k,M_v,M_eta,M_alpha,M_mem}`. `M_q` appears only in the explicitly
  hypothetical adaptive-q scenario.
- The shipped dependency chain is `static W_q -> M_mem -> CMS L1/L2/L3`, then
  all auxiliary target/prediction loss preparation, then all backward/update
  work. `M_k`, `M_v`, `M_eta`, and `M_alpha` are not direct block-forward
  outputs. CMS online gradients/state apply remain excluded from the baseline.

## Implementation choices

- Reference inputs mirror the user-reviewed live Sheet. Full Attention TTFT is
  one causal `Q=K=131072` invocation. A HOPE prefill operation row is one
  2,048-token chunk; only the TTFT metrics multiply forward and online groups
  by 64. Decode/ITL is exactly one `Q=1` token and never multiplies OSL.
- The reference timing authority is the live-Sheet analytical scenario:
  `GPU_FLOPS=4.614e15 FLOP/s` and `HBM_BW=7.4e12 B/s`. The repository H100 twin
  supplies HBM capacity and provenance only; those sources are deliberately
  mixed and labeled. The lower bound uses zero launch overhead because neither
  authority supplies a reference launch constant.
- Static/shared weights are read once per invocation. Request-local HOPE state
  scales with batch. The configured state-weight reload count is independent of
  update-apply RMW traffic.
- CMS capacity is never flattened into a dense hidden width. Each level has a
  dense router and top-k=1 active rank-64 expert A/B matrices. FLOPs use active
  experts; persistent state uses the complete 64/128/256 pool. Prefill HBM
  reads use the personalized per-request upper bound `min(capacity,Q*top_k)`,
  while decode reads top-k active experts. Update periods remain 1K/5K/10K.
- The shipped reference uses eager Titans loss/backward and amortizes state
  apply by each memory's update chunk for ITL. Group summaries expose Titans
  forward, CMS forward, forward total, loss, and backward+update separately and
  conserve the complete stage totals without inserting synthetic stages.
  CMS optimizer slots are independent and default to zero.
- Router metadata uses 8 bytes per selected route (index plus score). Dispatch,
  expert up projection, gate projection, SwiGLU activation/intermediate,
  down-projection, combine, Q, and other potentially fusible intermediates stay
  explicit. Every expert GEMM carries its own active-expert weight bytes and
  FLOPs in one stage. `fuse_expert_intermediates` removes only the internal
  up/gate/SwiGLU HBM materializations; dispatch and combine remain explicit.

## Scenarios and sensitivities

`results.json` distinguishes four HOPE scenarios:

1. `paper_equation_lower_bound` — static q and no momentum slots.
2. `shipped_credible_momentum` — static q and one configurable
   parameter-sized momentum slot.
3. `legacy_repo_proxy` — exact closed-form reproduction of the legacy script.
4. `adaptive_q_hypothetical` — adds `M_q`; deliberately excluded from defaults.

Sensitivity inputs include batch, context, prefill chunk, head/KV geometry,
Flash-Decode splits, KV reload multiplier, expert count/top-k/hidden width,
unique active experts, HBM-resident expert fraction, mutable-state byte widths,
expert-activation FLOPs, expert-intermediate fusion, memory update chunks,
Titans momentum slots, CMS capacity/LRD/update period/BPTT span, independent CMS
optimizer slots, update schedule, state reload count, and stage efficiencies.

## Stagewise versus aggregate roofline

The primary sequential lower bound is the sum of each stage's roofline time:

`sum(max(stage_compute, stage_memory) + launches)`.

The aggregate diagnostic is:

`max(sum(stage_compute), sum(stage_memory))`.

The aggregate form permits compute-heavy and memory-heavy stages to overlap
optimistically, so it cannot exceed the stagewise sum. Neither number models
distributed collectives or claims measured wall-clock performance.

Schema-v3 `results.json` additionally exposes
`attention_moe.metrics.ttft_ms/itl_ms` and, for each HOPE scenario, per-chunk
prefill, chunk-multiplied TTFT, and per-token
forward/online-overhead/including-online ITL metrics. Each context/batch sweep
row stores `hope_forward_decode_seconds` as the primary serving comparison and
`hope_including_online_decode_seconds` as a separate online-learning
sensitivity, with distinct predicates and first-satisfying-grid-point
summaries. Those grid points are not interpolated crossover thresholds.

The reference results do not claim exact equality with the native Sheet
because the engine keeps the sparse-CMS active-weight upper bound, explicit
MoE traffic, and stagewise roofline assumptions auditable.

## Reproduce

From the repository root, run the exact task commands:

```bash
python -m unittest -v experiments/E5-attn-vs-hope/test_model.py
python experiments/E5-attn-vs-hope/run.py
sha256sum experiments/E5-attn-vs-hope/results.json experiments/E5-attn-vs-hope/stdout.txt
python experiments/E5-attn-vs-hope/run.py
sha256sum experiments/E5-attn-vs-hope/results.json experiments/E5-attn-vs-hope/stdout.txt
```

On systems that install only `python3` (including the current worker image),
substitute `python3` for `python`; the target and outputs are identical.

## Provenance

- [Approved analytical design](../../docs/superpowers/specs/2026-07-27-attn-vs-hope-analytical-model-design.md)
- [Repository H100 twin](../../multiarch/twins/h100.json)
- [Legacy HOPE baseline](../../multiarch/hope_block_dag.py)
- [User-reviewed live-sheet archive](../../research/attn-vs-hope/Attn-vs-HOPE.xlsx)
- [Google meeting package](../../research/google-meeting/)

`model.py` is the source of equations, `run.py` is the deterministic reference
driver, `results.json` is the complete machine-readable record, and `stdout.txt`
is the compact human-readable report.
