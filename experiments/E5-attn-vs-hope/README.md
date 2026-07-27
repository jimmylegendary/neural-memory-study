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
- HOPE executes all self-modifying Titans forward/loss/backward/update work
  before entering the sequential CMS chain. CMS read cadence is one token even
  when a level's update period is long.

## Implementation choices

- Reference model inputs mirror the user-reviewed live-sheet values. A prefill
  row is one 2,048-token chunk; the recorded count of 64 chunks belongs in TTFT
  aggregation and is not multiplied into an operation row.
- The hardware peak is derived from the H100 twin as
  `2 FLOP/MAC * 132 instances * 3.748e12 MAC/s`; HBM bandwidth and capacity are
  read from the twin. The twin has no kernel-launch constant, so the reference
  lower bound uses zero rather than inventing one.
- Static/shared weights are read once per invocation. Request-local HOPE state
  scales with batch. The configured state-weight reload count is independent of
  update-apply RMW traffic.
- CMS low-rank capacity is flattened into an equivalent aggregate hidden width:
  `H_level = LRD * capacity`, preserving `P=I*H+H*O`. Capacities are 64, 128,
  and 256; update periods are 1K, 5K, and 10K tokens.
- The shipped reference uses eager Titans gradients and reports normal,
  simultaneous-boundary, and per-memory-period amortized decode. CMS online
  backward/update is excluded from this baseline, while every CMS forward read
  remains present. CMS optimizer slots are an independent input and default to
  zero; they are never inferred from Titans momentum slots.
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
