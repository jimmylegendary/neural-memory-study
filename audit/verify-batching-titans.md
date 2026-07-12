# Verify: batching/tensorize claims against ORIGINAL Titans (arXiv:2501.00663)

Scope (per task): judge only the claims that concern the **original Titans paper** —
the **Titans portion of C6** and the **tensorize principle of C1**. C2–C5, C7–C10 are
about *Self-Modifying / Self-Referential Titans* and *Hope/Nested Learning* (later papers),
so they are **out of scope** for verification against 2501.00663 and are marked as such.

Primary source: `papers/2501.00663.txt` (§3.2 "How to Parallelize the Long-term Memory
Training", lines 412–480; Appendix C, lines 2592–2647; abstract/intro line 24, 119–121).

---

## Core judgment questions

### Q-A. Does Titans reconstruct test-time inner-loop mini-batch GD via (i) chunk gradient as matmul, (ii) momentum recurrence as parallel associative scan?

**CONFIRMED (both).**

(i) matmul reconstruction — §3.2:
> "calculating the weights in the inner loop with mini-batch gradient descent, data-dependent
> learning rate, and weight decay can be reformulated so that it uses only matmuls and sum.
> We build upon the work of Yu Sun et al. (2024) that shows forward pass of a model optimizing
> with the mini-batch gradient descent (with constant learning rate) can be calculated using
> matmuls." (lines 421–424)

Closed form (Eq. 16): `M_t = (1-α_t)M_{t-1} - θ_t ∇ℓ(M_{t-1};x_t) = β_t M_0 - Σ_{i=1}^t θ_i (β_t/β_i) ∇ℓ(M_{t'};x_i)`,
with `t' = t - mod(t,b)`, `β_i = Π_{j=1}^i (1-α_j)` (lines 426–441).
Tensorized gradient (Eq. 17): `Σ θ_i (β_b/β_i) ∇ℓ(W_0;x_i) = Θ_b B_b (W_0 X - X) X^⊤`,
`Θ_b = diag(θ_1..θ_b)`, `B_b` analogous on `β_b/β_i` (lines 444–462).

(ii) momentum via parallel associative scan — §3.2:
> "S_t = η_t S_{t-1} - θ_t u_t ... we can compute all u_t at the same time, and so Equation 18
> is a linear recurrence with u_t as an input, S_t as the hidden state, and η_t as input-dependent
> transition value. Accordingly, we can use parallel associative scan (J. T. Smith, Warrington,
> and Linderman 2023) to calculate S_t s in this chunk." (lines 466–472)

Both mechanisms are stated exactly as C6/Q-A describe.

### Q-B. Exact form of chunkwise-parallel training

**CONFIRMED, with the paper's own shorthand caveat.**

- Sequence split into chunks of size `b ≥ 1`; the update is written as **mini-batch** GD
  (line 424), i.e. within a chunk gradients are taken at the **chunk-start weights** `M_{t'}`,
  `t' = t - mod(t,b)` (Eq. 16, line 441). This is the "gradients at batch-start weights"
  tensorization inherited from TTT (Yu Sun et al. 2024), extended with data-dependent lr +
  weight decay.
- Linear-memory gradient: `∇ℓ(W_0;x_t) = (W_0 x_t - x_t) x_t^⊤` (line 444). **Caveat:** the
  paper writes `x_t` as shorthand for the (k_t, v_t) pair; with the stated associative loss
  `ℓ = ‖M(k_t)-v_t‖²` the literal form is `(W_0 k_t - v_t) k_t^⊤` (cf. Appendix C, Eq. 32–34,
  lines 2592–2610). A re-implementer must undo this shorthand — noted in the study's own
  `assumptions_and_scope`. Does not affect the tensorization structure.
- MLP memory (`L_M ≥ 2`): "The process for MLPs with N_p ≥ 2 is similar" (line 442) — only
  asserted, not written out.
- Recurrence class: **intra-chunk linear, inter-chunk non-linear** — Appendix C:
  > "our LMM is using inter-chunk non-linear recurrence and intra-chunk linear recurrence."
  (lines 2619–2620). Matches C5's dual/chunk-wise description.
- Optional LTI/global-convolution fast path when gates are made per-chunk constant (lines
  473–480) — present but explicitly **not used in experiments**.

### Q-C. Batch-direction independence

**CONFIRMED as structurally inherent, but NOT explicitly discussed as a serving mechanism in this paper.**

- The tensorization in §3.2 is over the **sequence/chunk** dimension. The memory state
  (`M_t`, `S_t`) is per-sequence; training runs over batches of sequences (batch size 0.5M
  tokens, line 750), so independent per-sequence states across the batch is the standard,
  inherent setup — each sequence carries its own optimizer trajectory.
- The paper does **not** contain any explicit cross-request / serving-time batched-decode
  (BMM / grouped-GEMM per-sample-weights) discussion. That framing is a sound extrapolation
  (and appears in the study note's `systems_implications`, which explicitly calls it a
  "batched-per-sample-weights (grouped GEMM / bmm) workload ... the same serving pattern TTT
  imposes"), but it is the note's inference, not text of 2501.00663.

---

## Per-claim verdicts

### C1 — tensorize principle (add batch dim → BMM/grouped-GEMM/batched rank-update; structurally like attention per-request K/V batching)

**CONFIRMED (tensorize principle) / extrapolation (cross-request batching).**

- The load-bearing principle — that the per-request inner update is expressible as
  matmul/sum (and a rank-1 outer-product `(W k_t - v_t) k_t^⊤` in the linear case, i.e. a
  rank-update) — is directly in Eq. 16–18. Adding a request/batch axis makes these batched
  matmuls; nothing in the derivation couples distinct sequences, so batch-independence holds.
- **Not literally in the paper:** the specific serving vocabulary (BMM, grouped GEMM,
  continuous batching, "structurally similar to attention's per-request K/V batching"). This
  is a correct but external systems reading. So C1's *math* is CONFIRMED; its *serving claim*
  is a valid extrapolation, not a paper quote.

### C6 — Titans reconstructs inner-loop mini-batch GD as matmul+sum + parallel associative scan for momentum

**CONFIRMED (Titans portion).**

- "matmul + sum" reconstruction: lines 421–424, Eq. 16–17. Exact.
- momentum recurrence = parallel associative scan: lines 466–472, Eq. 18. Exact.
- Labeling caveat: the §3.2 heading is literally **"How to Parallelize the Long-term Memory
  Training"**; the abstract/intro says the paper presents "a **fast and parallelizable**
  algorithm to train our deep neural long-term memory" (lines 120–121, echoed line 281). So
  the descriptor "Fast and Parallelizable Training" is a faithful paraphrase of the paper's
  own wording. The attribution of that exact phrase to **"NL" (Nested Learning / Hope)** is a
  *different paper* and is **out of scope** here — not verified against 2501.00663.

### C2, C3, C4, C5, C7, C8, C9, C10 — OUT OF SCOPE for this paper

These concern Self-Modifying / Self-Referential Titans and Hope/Nested Learning (later works),
not 2501.00663. Two incidental cross-checks against the original that bear on them:

- **C5** (batch + intra-chunk token parallel; chunk-boundary recurrence M0→M1→M2; "dual/chunk-wise
  form"): the intra-chunk-linear / inter-chunk-nonlinear structure it invokes is **CONFIRMED**
  in the original (lines 2619–2620, Eq. 16). The remainder (self-modifying weights) is a
  later-paper construct — not judged here.
- **C2** (`M_t = M_{t-1} A_t − η_t v̂_t k_t^⊤` with data-dependent transition `A_t`): the original's
  linear-memory special case (Appendix C, Eq. 32) is `S_{t+1} = S_t(I − θ_t k_t k_t^⊤) + θ_t v_t k_t^⊤`
  plus a diagonal weight-decay `(1−α_t)` — a **diagonal/identity-minus-rank-1** transition, not a
  general dense data-dependent `A_t:[B,Din,Din]`. So C2's general `A_t` is a *generalization beyond*
  the original Titans update, consistent with it being a later "self-modifying" variant. Flag, not a
  contradiction of 2501.00663.

---

## Bottom line

- **C6 (Titans part): CONFIRMED** — the original paper explicitly reconstructs the inner-loop
  mini-batch GD into (i) matmul+sum (Eq. 16–17) and (ii) a parallel associative scan for the
  momentum recurrence (Eq. 18), under a "fast and parallelizable" framing (§3.2 / abstract).
- **C1 (tensorize principle): CONFIRMED for the math**; the cross-request BMM/grouped-GEMM
  serving analogy is a sound extrapolation, not text of the paper.
- Chunkwise form and intra/inter-chunk linearity: **CONFIRMED**, with the paper's `x_t`↔(k_t,v_t)
  shorthand and the "MLP case is similar" hand-wave as the only imprecisions.
- Batch-direction independence: **inherent/CONFIRMED structurally**, but the serving-time batched
  state manager is not a topic of this paper.
