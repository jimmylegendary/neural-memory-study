# PRE-RESEARCH — The Google TTT / Neural-Memory Line (Behrouz et al., 2024–2026)

**Synthesis over six deep-read analyses** (`notes/2501.00663.json` … `notes/2606.03979.json`),
the Part-I prerequisite curriculum (`notes/prereq-curriculum.md`), and the Veridraft
tooling plan (`notes/veridraft-tooling.md`). Target artifact: a 150–200 pp. study paper
for transformer-inference systems experts with zero training background.

Papers, chronological:

| # | arXiv | Short name | Date | One-line role |
|---|-------|-----------|------|----------------|
| 1 | 2501.00663 | **Titans** | 2024-12 | Deep memory + GD-with-momentum-and-decay as the sequence layer |
| 2 | 2504.13173 | **Miras** | 2025-04 | The design-space taxonomy (attentional bias / retention / architecture / algorithm) |
| 3 | 2505.23735 | **Atlas** | 2025-05 | Optimal memorization: windowed objective (Omega), capacity theory, Muon inner optimizer |
| 4 | 2511.07343 | **TNT** | 2025-11 | Training economics: chunk hierarchy, resets, two-stage train-big/serve-small |
| 5 | 2512.24695 | **Nested Learning** | 2025-12 (NeurIPS'25) | The ontology: model+optimizer = nested associative memories at update frequencies; Hope |
| 6 | 2606.03979 | **Sleep** | 2026-06 | The lifecycle: wake/sleep, offline consolidation, dreaming, capacity growth |

---

## 1. Chronological arc — one continuous research story

The line is best read as a single program executed in six installments: **first make an
optimizer be the sequence layer (Titans), then name the design space that move opens
(Miras), then push each axis of that space to its optimum (Atlas), then pay the training
bill (TNT), then declare the move universal — everything, including the outer optimizer,
is the same object (Nested Learning) — and finally delete the train/test boundary
entirely (Sleep).** Each paper's "future work" section is, almost verbatim, the next
paper's abstract.

### Bridge 1: Titans → Miras

**What Titans leaves open.** Titans ships one point design: a deep MLP memory whose
weights are updated per token by gradient descent with data-dependent momentum
("past surprise") and weight decay ("forgetting") on an L2 associative loss, composed
with attention three ways (MAC/MAG/MAL). Its own Appendix C shows Gated DeltaNet,
Longhorn, RWKV-7, and TTT are special cases — which raises the question the paper never
answers: *why these choices?* Why L2? Why momentum+decay and not another optimizer? Is
"forgetting" the right abstraction at all? Titans names "better inner objectives beyond
L2 regression" and "better inner optimizers" explicitly as open.

**How Miras answers.** Miras re-derives the entire field — Transformers, all gated
linear RNNs, the delta-rule family, TTT, Titans — as associative memories minimizing an
internal objective (the **attentional bias**) online, subject to a **retention gate**
(the re-theorized forget gate: models never erase, they decline to retain). The
observation that everything before used only two objectives (dot-product, L2) and one
regularizer (L2) converts Titans' point design into a 4-axis design space (**Miras**:
memory architecture × attentional bias × retention gate × learning algorithm) and
populates two new axes with Moneta (Lp/Lq), Yaad (Huber), Memora (KL/simplex,
retention-by-renormalization).

**Carried intact:** the associative k→v inner loss; deep 2-layer-MLP memory; chunkwise
gradient-at-chunk-start training; outer-learned data-dependent gates; the bilevel
(meta-learning) framing. **Generalized:** Titans' surprise → gradient of an *arbitrary*
attentional bias; forget gate → retention regularization (L2 is one point in a
Bregman/f-divergence/elastic-net family); Titans itself becomes one row of Table 1.
**Discarded/deferred:** momentum — all three shipped Miras models use plain GD; the
memory-*algorithm* axis is deliberately left open ("Newton/other optimizers as future
work"), the clean setup for Atlas.

### Bridge 2: Miras → Atlas

**What Miras leaves open.** Three things, stated in its own open questions: (i) the
optimizer axis untouched (plain GD everywhere); (ii) per-token single-pair objectives
only — no model asks whether the last c tokens are *jointly* well-stored; (iii) no
theory of memory capacity for any variant.

**How Atlas answers.** All three at once. (i) **Muon in the inner loop** — momentum over
gradients, Newton–Schulz-orthogonalized: the first parallelizable recurrent model with
an approximately second-order ("locally optimal") memory update, with NS iteration count
as a test-time-compute dial. (ii) The **Omega rule** — a sliding-window inner loss over
the last c tokens with learned per-token γ gates ("surprise of the context" instead of
surprise of the token); c=1 recovers the delta rule/Titans, c=L recovers the Mesa-layer.
(iii) A formal **capacity theory**: matrix memory stores O(d_k) pairs, deep MLPs stay
subquadratic, degree-p polynomial features buy O(d_k^p), and the exponential map φ*
shows softmax attention is an associative memory with *unbounded* capacity — the root
explanation of the retrieval gap, plus DeepTransformers/Dot as strict generalizations of
(unnormalized) attention.

**Carried intact:** the full Miras vocabulary and 4-axis frame (Atlas' Table 1 is Miras'
taxonomy plus two columns); deep MLP memory; chunkwise training (window costs only a
banded mask); Titans' momentum/decay gates re-enter. **Generalized:** per-token loss →
windowed loss; momentum → orthogonalized momentum; keys → featurized keys.
**Discarded:** the "test-time *training*" name — Atlas insists on "test-time
*memorization*" (nothing survives a state reset), a terminological move that quietly
sets up the continual-learning turn: if in-context adaptation isn't learning, real
continual learning still needs a mechanism, which is NL's and Sleep's territory.
**Honest wrinkle the study paper must keep:** Atlas' own ablation shows removing Muon
*improves* perplexity while helping reasoning — the optimizer axis's value is
equivocal at 760M, and attention still wins in-context recall (53.6 vs 43.7).

### Bridge 3: Atlas → TNT

**What Atlas leaves open.** The whole line, through Atlas, reports **zero wall-clock
numbers**. Everything rests on the chunkwise trick — inner gradients at a stale
chunk-start state — with an unquantified approximation, a fixed chunk size C serving as
both a throughput knob and (unnoticed) a semantic hyperparameter, and <5–10% FLOPs
utilization for deep memories at quality-optimal small chunks. Atlas also never asks
what happens if you *serve* at a different chunk size than you trained.

**How TNT answers.** TNT is the systems installment: not a new architecture but a
training paradigm. It names three challenges — (1) no efficient small-chunk training,
(2) write-with-keys/read-with-queries domain mismatch, (3) the newly *discovered*
train/inference chunk-size mismatch (a 550M Titans trained at C=64 goes from ppl 13.78
at C=64 to 36.45 at C=8) — and fixes them with: a **hierarchical memory** (one global
deep memory at C_G=2048 for long range + N local memories periodically **reset to a
learned W_init**, which severs the nonlinear recurrence and buys true context
parallelism — the only known way to parallelize a nonlinear deep-memory recurrence);
**Q-K projection** onto the running key subspace at retrieval; and **two-stage
training** — pre-train at big chunks, fine-tune ~5% at small chunks, making chunk-1
decode the *quality-optimal* operating point. Result: up to 17.4× faster to target loss
than the accurate Titans baseline *while improving* perplexity; plain-JAX TNT beats
FlashAttention per step at 32K.

**Carried intact:** the compression/retrieval two-op abstraction; chunk-start gradient
anchoring; meta-learned initial state (now load-bearing: W_init is what makes resets
survivable). **Generalized:** one memory → a *hierarchy of memories at different
timescales* — the engineering prefiguration of CMS; chunk size → two independent knobs
(train-time throughput, serve-time resolution). **Discarded (temporarily):** momentum,
forget gating, Muon — stripped "for clarity" (App. D), so TNT validates the training
economics on a simplified Titans and defers composition with the full line.

### Bridge 4: TNT → Nested Learning

**What TNT leaves open.** TNT's global/local hierarchy is a trick justified by
throughput. But it plants the decisive idea: *components updating at different
frequencies, with knowledge parked at slower timescales surviving resets at faster
ones.* Nothing yet says why that should be the organizing principle of the whole model.

**How NL answers.** Nested Learning declares it the ontology. Any model *together with
its training procedure* is one system of nested optimization problems, each an
associative memory compressing its own context flow (tokens, gradients, higher signals)
at its own **update frequency**. Under this lens: backprop is a self-referential
associative memory over local surprise signals; momentum is a Hebbian memory over
gradients; **Adam is the optimal memory for an element-wise L2 objective**; Muon's NS
iterations are an inner optimization level; attention is the non-parametric
Nadaraya–Watson solution; recurrent memory layers are "MLP blocks with one extra
level." The framework is then used generatively: new optimizers (Delta Momentum, DMGD,
DGD, M3), the **Continuum Memory System** (a chain of MLP blocks at geometrically spaced
update frequencies — TNT's hierarchy made architectural, the short/long-term dichotomy
made a spectrum), and **self-modifying Titans** (the k/v/η/α projections themselves
become test-time-updated memories that *generate their own targets* — Schmidhuber's
self-referential line made chunk-parallelizable). The combination, **Hope**, beats
Titans/Transformer++/RWKV-7/etc. at 760M and 1.3B, holds BABILong to 10M tokens, and
solves parity/aⁿbⁿcⁿ with length generalization while training in parallel.

**Carried intact:** Miras' Definition 1 (verbatim, as NL's Definition 1); the Omega
rule (imported); chunkwise stale-snapshot parallelization (now also the compromise that
makes *self-reference* trainable); meta-learned inits (now one of five knowledge-transfer
mechanisms, MAML-classified). **Generalized:** 2 levels → K levels; Titans'
persistent/long/short memory triad → a frequency continuum; the forget gate → DGD's
data-dependent decay *at the optimizer level*; TNT's reset → Nested-CMS re-initialization.
**Discarded:** the architecture/optimizer distinction itself — the paper's title move
("the illusion of deep learning architecture"). Also newly *motivating* rather than
technical: continual learning and the anterograde-amnesia analogy, with online
consolidation delivered and **offline consolidation explicitly deferred** — the
hand-off sentence to Sleep.

### Bridge 5: Nested Learning → Sleep

**What NL leaves open.** By its own list: catastrophic forgetting is NOT solved (only
reshaped by CMS); offline consolidation (replay/sleep) is "explicitly out of scope";
capacity is fixed, so forgetting is inevitable under finite compression; and everything
adaptive still happens *while input flows*.

**How Sleep answers.** Abolish the train/test split in favor of a **wake/sleep
lifecycle**. Wake = NL's CMS running online consolidation. Sleep = two offline
processes: (1) **Memory Consolidation** — just before a fast block's scheduled update
would overwrite it, distill its knowledge *upward* (Knowledge Seeding: GKD from the
smaller pre-expansion self, plus a Learning-to-Imitate RL reward) into a **freshly
activated low-rank MoE expert** in the next slower block, then reset the fast block's
old experts (synaptic pruning) — catastrophic forgetting reframed as a *capacity*
problem answered by structural growth, not regularization; (2) **Dreaming** — a
SEAL-style RL loop generating synthetic data with deliberately randomized MoE routing
for novelty, filtered by gradient-based importance, LoRA-applied in isolated instances,
reinforced by ReST^EM. Results: beats SEAL on knowledge incorporation and ARC, beats
GRPO on AIME with Qwen3-8B, near-perfect BABILong to 10M tokens.

**Carried intact:** CMS and the update-frequency definition (verbatim import);
Hope; the chunk-boundary schedule (sleep fires exactly at CMS update boundaries).
**Generalized:** reset-to-init → reset-after-consolidation (TNT's throughput trick
becomes a memory-hygiene principle); self-modification of weights-by-rule (NL) →
self-modification of weights-by-self-generated-*data*; the inner/outer loop dichotomy →
three regimes (wake token/chunk updates, sleep consolidation jobs, sleep dreaming jobs).
**Discarded:** the last remnant of "test time." Also a quiet methodological break the
study paper must flag: Sleep's machinery is *grafted onto pre-trained Llama/Qwen
backbones*, not meta-learned end-to-end — the first paper in the line whose core
mechanism is an algorithmic wrapper rather than a differentiated-through inner loop.

### Verdict on the author's thesis ("after all six, Google's next step is visible as a completed form")

**Substantially supported, with two honest qualifications.** Supported: the six papers
systematically open and then close every axis of one design space — objective, retention,
architecture, optimizer, timescale, training parallelization, lifecycle — and their
open-questions sections *converge* (see §3): scale, serving economics, learned
schedules, task-free sleep, safety. The completed form (§3.1) is genuinely derivable
from the texts, not speculation. Qualification 1: completion is *conceptual*; the
empirical ceiling is 1.3B/100B tokens from scratch, and the in-context-retrieval gap to
attention is measured and unclosed in both Atlas and NL — the completed form may still
be a hybrid with attention, which the line's own MAG/MAC results quietly concede.
Qualification 2: the "one story" reading is partly retrospective — TNT is a training
paper that strips the very mechanisms (momentum, gating) the story says are essential,
and Sleep abandons end-to-end meta-learning; the arc has real seams, and the study
paper will be more credible for showing them.

---

## 2. Cumulative concept ledger

| Concept | Born in | Evolved in | Final form in |
|---|---|---|---|
| Surprise metric (gradient of inner loss as novelty) | Titans (momentary surprise = ∇ℓ) | Miras (gradient of arbitrary attentional bias); Atlas (surprise **of the context** = windowed gradient) | NL (Local Surprise Signal; backprop itself = memory over surprise) |
| Momentum-as-memory | Titans (S_t = past surprise; first momentum in a linear-recurrent-style model) | dropped by Miras' shipped models; Atlas (momentum → Muon: NS-orthogonalized) | NL (momentum = Hebbian memory over gradients; Delta Momentum, DMGD, **M3** multi-timescale momentum) |
| Forgetting / retention gate | Mamba-2/GDN (pre-line); Titans (data-dependent weight decay α_t generalizes them to deep memory) | Miras (retention re-theorization: local+global, KL/simplex, elastic-net hard/soft, retention-by-renormalization); Atlas (γ window gates = in-context pruning); TNT (periodic reset to W_init = hard scheduled retention) | Sleep (synaptic-pruning reset + **capacity expansion**: don't overwrite — grow) |
| Attentional bias (inner objective as design axis) | Miras (named; implicit L2 in Titans/TTT) | Atlas (Omega windowed bias; dot-product vs L2 unified with feature maps) | NL (every level has its own objective over its own context flow) |
| Omega rule (windowed inner objective) | Atlas | — | NL (imported verbatim; window retention in Generalized GD) |
| Deep memory module | Titans (MLP L_M≥2; depth ablation) | Miras (2-layer residual MLP becomes the standard); Atlas (capacity theorem for depth; gated MLP in Atlas++); TNT (formal compression/retrieval two-op abstraction) | Sleep (MoE MLP chain with grown low-rank experts) |
| Memory capacity (formal) | Atlas (matrix O(d_k) → poly O(d_k^p) → φ* unbounded = attention) | — | Sleep (catastrophic forgetting *reframed as capacity*; growth as the remedy) |
| Persistent memory | Titans (N_p learnable prefix tokens) | NL (gating corollary when init isn't meta-learned) | NL/CMS (persistent weights = the frequency-0 point of a continuum) |
| Hybrid composition (attention + memory) | Titans (MAC/MAG/MAL; MAL indicted) | Miras-H (Samba-style); Atlas (MAG/MAL, MAC for BABILong) | NL (**Hope** = self-modifying Titans + CMS; Hope-Attention variant) |
| Chunkwise-parallel inner loop | TTT (external), adopted in Titans (+ scan for momentum) | Miras (smoothed surrogates, lag token); Atlas (banded window mask, batched NS-5) | **TNT** (the subject: hierarchy + resets + two-stage; chunk economics quantified); per-level chunking in NL/Sleep |
| Chunk size as a *semantic* knob (train/serve mismatch) | TNT (Challenge 3 / Fig. 2 discovery) | — | TNT (Stage-2 fine-tune makes chunk-1 decode optimal) |
| Context parallelism for nonlinear recurrence (reset to learned init) | TNT | NL (Nested-CMS re-initialization at context end) | Sleep (reset after upward consolidation) |
| Meta-learned initial memory state | Titans (M_0, implicit) | TNT (W_init load-bearing for resets); NL (initialization = one of 5 knowledge-transfer mechanisms, MAML class) | Sleep (teacher = the smaller pre-expansion self) |
| Q-K projection (write/read domain alignment) | TNT | — | TNT (not yet re-adopted downstream — open thread) |
| Multi-timescale / nested levels | Titans (implicit: persistent/long/short triad) | TNT (global/local chunk hierarchy) | NL (K levels, update-frequency Def., **CMS**) → Sleep (wake/sleep atop CMS) |
| Optimizer-as-memory / optimizer-as-architecture | Titans (update rule *is* SGD+momentum+decay, implicit) | Miras (learning-algorithm axis); Atlas (Muon transplanted into the layer) | NL (Adam = optimal L2 memory; momentum/AdaGrad/Muon reverse-engineered; M3/DMGD/DGD designed from the frame) |
| Second-order / "locally optimal" inner updates | Atlas (Muon NS-k; k = test-time-compute dial) | NL (NS iterations = an inner optimization level) | NL |
| Self-modification / self-reference | (Schmidhuber 1992/93, external) | NL (self-modifying Titans: projections+gates as memories generating own targets; backprop shown self-referential) | Sleep (Dreaming: the model edits its own weights via self-generated data + RL) |
| Consolidation | NL (online consolidation; CMS knowledge cycling; amnesia framing) | — | Sleep (**offline**: Knowledge Seeding upward distillation + LTI + pruning) |
| Wake/sleep lifecycle (no train/test split) | Sleep | — | Sleep |
| Expressivity beyond TC⁰ | Titans (Thm 4.1, asserted, unproven) | — | NL (empirical: parity, aⁿbⁿcⁿ, Shuffle-2 at 100% incl. OOD lengths, with parallelizable training; still no theorem) |

---

## 3. Convergence analysis

### 3.1 The completed form, argued strictly from the six papers

Composing only what the papers themselves built and validated, the end state of this
line is:

1. **State = weights, at every timescale.** The KV cache is replaced (or minimized to a
   sliding window in hybrids) by a spectrum of parameter blocks indexed by update
   frequency — per-token fast memories (Titans/Atlas), chunk-cadenced CMS levels (NL),
   sleep-cadenced grown experts (Sleep), frequency-0 persistent weights. Titans →
   Sleep is one long widening of this spectrum from {∞, 0} to a continuum.
2. **Every block is the same object**: an associative memory defined by (architecture,
   attentional bias, retention, inner optimizer, frequency) — Miras' four axes plus
   NL's fifth (level). Attention itself sits inside the frame as the non-parametric,
   unbounded-capacity, frequency-∞ corner (Miras' NW row; Atlas' φ*; NL Part 5).
3. **The inner optimizer is a design surface equal to the architecture** — GD → +momentum
   (Titans) → objective zoo (Miras) → windowed+Muon (Atlas) → DGD with self-generated
   targets (NL). Symmetrically, the *outer* optimizer got redesigned from the same
   frame (M3), and NL names "architecture-specific optimizers" as the undelivered
   promise.
4. **Training is chunk-anchored everywhere and hierarchical where it must be.** The
   stale-snapshot chunk trick is the universal enabler (all six), TNT gives its
   economics and the reset mechanism, and the two-stage recipe reconciles training
   throughput with chunk-1 decode.
5. **The lifecycle is wake/sleep.** In-context (parametric) learning during wake;
   scheduled offline jobs that consolidate upward into grown capacity and self-generate
   training data during sleep. The model is never "done training"; serving fleets run
   recurring fine-tuning jobs by design.

Read as a product, this is a **continually-learning LLM whose serving state is
per-session/per-tenant mutable weights, trained with TNT economics on TPU-shaped
matmuls, with a sleep service attached to the deployment**. That is the "next step"
that becomes visible — and it is visible precisely because the six papers' own
open-questions sections all point at the same five gaps (below), i.e., the remaining
work is over-determined by the texts.

### 3.2 The obvious next open problems (from the papers' own lists, deduplicated)

1. **Scale.** Every quality claim ceilings at 1.3B/100B (TNT at 150M); Titans promised
   larger models "in the next version" in Dec 2024 and the line never delivered them.
   Whether momentum/deep-memory/self-modification advantages survive 7B+, SFT/RLHF, and
   production data is the single biggest unknown.
2. **The retrieval gap.** Attention still wins in-context recall in Atlas (53.6 vs 43.7)
   and NL (FDA 67.3 vs 41.9). Capacity theory says why (unbounded φ*). Is the gap
   closable parametrically, or is the completed form necessarily a hybrid?
3. **Serving economics and kernels.** Zero decode-throughput/latency numbers in the
   entire line; no fused deep-memory kernel exists; per-request mutable weights break
   shared-weight batching (grouped-GEMM decode, state snapshot/rollback for speculative
   decoding, numerics-drift of optimizer-trajectory state, multi-tenant isolation,
   privacy of weight-absorbed context). Named but unanswered in Titans, Miras, Atlas,
   NL, Sleep alike.
4. **Learned schedules.** Chunk sizes, window c, CMS frequencies, level count, sleep
   timing — all hand-set; NL shows a wrongly-placed level can hurt (inner-q ablation);
   Sleep hard-wires sleep to chunk boundaries. Nothing learns *when to update what*.
5. **Task-free, safe self-modification.** Dreaming needs an explicit (context, metric)
   pair; the reward-model dependence is unexamined; recursive weight self-editing has
   no safety analysis. Plus the deferred theory: no regret/capacity/expressivity results
   for anything past the linear special cases, no error bound on chunkwise staleness
   anywhere in six papers.

---

## 4. Table of contents (as assembled — 25 chapters + front/back, ≈ 204 pp.)

**Status (2026-07-12): reconciled to the built files** under `study-kr/`. This supersedes
the draft ToC. Per-chapter page budgets follow `style/STYLE-NOTATION.md` §4.3 (v1.1), the
authoritative budget; Part III budgets are this dossier's originals. The min-path column of
`prereq-curriculum.md` (48 pp. Part I) remains the compression valve toward ≈ 155 pp.

Two structural changes vs the draft: (a) the standalone **Ch. 0 introduction** was folded
into front matter (`front/00-preface.md`) plus **ch18**, which now opens Part III with the
convergence + pair-thesis program; (b) the **B9 continual-learning interlude** became a
first-class Part I chapter (**ch11**), so Part II is a clean six-chapter run (ch12–ch17)
and every downstream chapter number shifted +1 from the draft.

### Front matter (`study-kr/front/`, ≈ 4 pp.)
`00-preface.md` (audience, why these six, how to read + bypass route, pair-thesis preview,
honesty contract, notation & 3-layer convention), `00-toc.md`, and the three Part openers
(`01/02/03-partN-opener.md`).

### Part I — Background for inference engineers (`study-kr/part1/`, ≈ 78 pp.)
Every concept as a (state, update, cost) object; each chapter clears one systems bridge.

| Ch. | File / title | pp. |
|---|---|---|
| 1 | ch01 — Orientation: inference/learning Rosetta stone; two loops ($W$ vs $\Theta$); master eq. (M) | 4 |
| 2 | ch02 — Training as a system: backprop + optimizers (momentum/Adam/decay/Muon) as objects | 10 |
| 3 | ch03 — Online learning: OGD, regret, FTRL | 6 |
| 4 | ch04 — Meta-learning / bilevel: inner vs outer loop, MAML, $W_{\mathrm{init}}$ | 6 |
| 5 | ch05 — Associative memory: Hopfield → delta rule; crosstalk, capacity | 8 |
| 6 | ch06 — Linear attention & FWP: DeltaNet, Gated DeltaNet, Longhorn, RWKV-7 | 10 |
| 7 | ch07 — SSM lineage: S4 → Mamba → Mamba-2, SSD duality | 6 |
| 8 | ch08 — TTT lineage: test-time adaptation → TTT-Linear/MLP, dual form | 7 |
| 9 | ch09 — Chunkwise-parallel training: one scheme, four instances; stale-snapshot, semantic-knob | 10 |
| 10 | ch10 — Consolidated systems bridge: cost-model cheat sheet, roofline vs chunk size, glossary | 5 |
| 11 | ch11 — Continual learning, complementary learning systems, distillation, RL-lite | 4–7 |

### Part II — The six papers in depth (`study-kr/part2/`, ≈ 57 pp.)
Uniform 8-section grid; all math reduced to the unified notation / eq. (M); honesty caveats in-body.

| Ch. | File / title | pp. |
|---|---|---|
| 12 | ch12 — Titans: mechanism, chunkwise math, MAC/MAG/MAL, per-mechanism S-NIAH, systems reading | 10 |
| 13 | ch13 — Miras: the 4-axis taxonomy, FTRL/Learning–Retaining, Moneta/Yaad/Memora | 8 |
| 14 | ch14 — Atlas: capacity theory, Omega, Muon, DeepTransformers/Dot; equivocal Muon ablation + retrieval gap | 10 |
| 15 | ch15 — TNT: chunk economics, resets/context parallelism, Q-K projection, chunk-size mismatch; only wall-clock | 7 |
| 16 | ch16 — Nested Learning: levels/frequencies, optimizers-as-memories, CMS, self-modifying Titans, Hope | 13 |
| 17 | ch17 — Sleep: wake/sleep, Knowledge Seeding, Dreaming; the graft-vs-cotrain seam | 9 |

### Part III — Original contribution (`study-kr/part3/`, ≈ 60 pp.)
Spine = the **D4 workload-split pair thesis**, formalized in ch18 and quantified by 8 exploration-grade experiments.

| Ch. | File / title | pp. |
|---|---|---|
| 18 | ch18 — Convergence & the contribution program: completed form + two qualifications, D4 pair thesis, 8-claim map | 8 |
| 19 | ch19 — Scaling analysis & candidate scaling laws: state-bytes and capacity ($O(d_k^p)$) as first-class axes | 9 |
| 20 | ch20 — Hardware bottleneck: decode roofline per architecture; backward-pass-at-decode as a new serving primitive | 10 |
| 21 | ch21 — Lessons from the Transformer/NVIDIA era: the hardware-lottery reading | 7 |
| 22 | ch22 — Player-strategy: Google / NVIDIA / open-source kernel ecosystem; each player's next rational move | 6 |
| 23 | ch23 — Large-scale training & serving projections: TNT economics to 7–70B; per-session weights as a cache class | 8 |
| 24 | ch24 — Proposals: algorithm-level experiments + HW-level frequency-tiered placement / fused-kernel pair | 9 |
| 25 | ch25 — Conclusion & research agenda | 3 |

**Back matter** — bibliography (Veridraft/S2-verified pool), glossary index (≈ 5 pp.)

Total: 4 (front) + 78 (I) + 57 (II) + 60 (III) + 5 (back) ≈ **204 pp.**
(min-path compression → ≈ 155 pp.)

---

## 5. Candidate original-contribution angles, ranked by defensibility

Ranked for a top-tier venue (MLSys/ISCA/NeurIPS-datasets-or-position/ICML depending on
angle). Each: evidence in the papers / small supporting experiment / falsifier.

**A1. The missing serving-cost characterization of test-time-memorization models.**
*Most defensible.* Evidence: the entire line publishes no decode throughput, latency, or
memory-crossover measurements (explicitly absent in Titans, Miras, Atlas, NL, Sleep;
TNT gives training wall-clock only); each paper's state math (2·P_M buffers, 8–16 d²
Miras states, 6-memory Hope blocks) is stated but never costed against a served KV-cache
baseline. Experiment: implement (or adapt flash-linear-attention/TTT kernels for)
Titans-LMM and one Atlas variant decode paths; measure tokens/s, per-request state
residency, and quality-vs-context on one H100/TPU vs FlashAttention + paged KV at
4K–1M context; validate a published roofline model. Falsifier: if measurements simply
track naive FLOP/byte counting with no surprises (no crossover shift from RMW traffic,
no batching cliff from per-request weights), the paper reduces to arithmetic anyone can
do; also falsified as *novel* if a 2026 systems paper already benchmarks this family
end-to-end (must re-verify literature at writing time).

**A2. Train/serve chunk-consistency: replicate and extend TNT's Challenge 3.**
Evidence: the chunk-size-mismatch phenomenon rests on ONE figure (550M Titans, one
training chunk, no gating/momentum, no explanation); TNT App. D explicitly defers
gated/momentum variants; no theory anywhere. Experiment: 100–150M models, grid over
(train chunk × inference chunk) × {plain GD, +momentum, +forget gate}, plus a
loss-landscape probe of *why* over-specialization happens. Cheap (≪ TNT's 10B-token
budget at reduced scale). Falsifier: the mismatch vanishes with gating/momentum or at
smaller scale — which would itself be a publishable negative result against TNT's
generality claim, so the angle is robust either way; weak falsification if effect
reproduces but resists any mechanistic account (then it's a replication note, not a
paper).

**A3. State-capacity as a scaling-law axis.** Evidence: Atlas' capacity theorems give a
theoretical axis (O(d_k) / subquadratic / O(d_k^p) / unbounded) that cleanly orders the
architectures; Miras/Atlas/NL publish enough (params, tokens, context, ppl) points to
attempt cross-architecture fits; the line itself never fits a law. Experiment: 3–4
sizes × 3 architecture classes (matrix, deep-MLP, deep+featurized) trained on identical
tokens; fit ppl vs (params, state-bytes, capacity-proxy) and test whether capacity
predicts long-context benchmarks better than state-bytes. Falsifier: published points
too heterogeneous (different tokenizers/data — they are: Llama-2 vs T5 vocab across the
line) and own runs too small for stable exponents; capacity proxy adds no predictive
power over raw state size.

**A4. Per-session-weights serving architecture (design/position contribution).**
Evidence: Titans' systems notes (grouped-GEMM decode, snapshot/rollback for speculative
decoding, numerics drift of optimizer-trajectory state), NL's continual-serving
questions (per-tenant divergence, rollback, provenance), Sleep's fleet-lifecycle
implications (versioning every sleep, 5-instance dream fan-out) — all raised, none
designed. Contribution: a coherent serving-stack design ("session state = weights":
paging, eviction=retention-gate cooperation, checkpoint format, multi-tenant isolation,
sleep-as-background-compaction) with a simulator-level evaluation. Falsifier: without
at least simulation numbers it's an opinion piece; defensibility depends on A1's
measurements feeding it.

**A5. The hardware-lottery / player-strategy analysis of the line.** Evidence: every
paper visibly negotiates its math with accelerators (chunkwise = trade exactness for
GEMMs; Atlas "tensorize and maximize matmuls"; TNT manufactures arithmetic intensity;
MAL wins throughput only via FlashAttention maturity). Strong as a *chapter* and as
framing; weak as a standalone top-tier submission (not falsifiable enough). Keep in the
book; possibly a workshop/position spin-off.

**A6. Memory-device / memory-centric architecture argument — see verdict below.**
Rank: mid, and only in the two load-bearing forms identified; as a general "this line
needs PIM" claim it would be rejected on the papers' own evidence.

### Honest verdict: where the memory-centric thread is real vs forced

**Genuinely load-bearing (two places):**

1. **Decode-side state traffic.** These models' decode step reads *and rewrites* the
   entire fast-weight state every token (Miras: the full 8–16 d² state per layer;
   Titans: M_t + momentum S_t; Hope: six memories) — read-modify-write, per-request,
   unshared, at O(1) reuse. That is a qualitatively different memory workload from a
   KV cache (append-once, read-many, shareable pages): bandwidth-bound, write-heavy,
   residency-critical. A memory-centric argument here — high-RMW-bandwidth state
   residency, near-memory elementwise update engines, per-tenant state placement — is
   grounded in the papers' own cost accounting, not imposed on it.
2. **Frequency-tiered placement.** NL/Sleep define components *by update frequency*
   (per-token fast weights → chunk-cadenced CMS levels → sleep-cadenced experts →
   frozen weights). That is natively a memory-hierarchy specification: hot small
   fast-weights in SRAM/HBM, slow CMS levels tolerating DDR/CXL-class latency, sleep
   consolidation as an explicit data-migration schedule. The mapping is almost literal
   and no prior work states it; this is the freshest defensible memory-architecture
   claim in the study.

**Forced (do not argue):**

3. **"Training this line needs new memory devices."** The evidence runs the other way:
   six consecutive papers *reshape the algorithm into dense matmuls* to fit existing
   accelerators (chunk anchoring, banded masks, batched NS-5, resets-for-parallelism);
   TNT beats FlashAttention in plain JAX on a stock TPU. Training/prefill is and will
   remain tensor-core territory; claiming device-level necessity there contradicts the
   line's central engineering pattern.
4. **Generic PIM advocacy.** The per-token update is gradient computation through an
   MLP — GEMV/GEMM-shaped, not the sparse/irregular access pattern that motivates most
   PIM proposals. Only the elementwise state-update epilogue (AXPY, decay, renorm) is
   PIM-shaped, and it is a minority of the FLOPs.

**Comprehensive stance for the book (per the author's requirement not to neglect
accelerators):** present both sides of one workload split — decode/serving state
management is the genuine memory-centric opportunity (angles 1–2); training/prefill and
the NS/deep-memory FLOPs are the accelerator opportunity (fused chunk kernels,
grouped-GEMM decode engines, backward-capable serving kernels). The interesting HW
proposal is the *pair*, not either alone.

---

## 6. Open decisions needing a user interview

1. **Venue & format.** A 150–200 pp. artifact fits no conference. Options: (a) arXiv
   monograph + 2–3 extracted venue papers (A1→MLSys, A2/A3→ICML/NeurIPS, survey
   core→TMLR or FnT-ML); (b) Foundations-and-Trends-style commissioned survey; (c) book
   with a publisher. Tradeoff: (a) maximizes citations and career signal per unit time
   but fragments the narrative; (b)/(c) preserve the single artifact but review cycles
   are 6–12 months and top-tier "venue" credit is weaker.
2. **Language.** English-only vs English + Korean edition (precedent: the HATIR
   paper/patent line was dual-language). English maximizes reach and matches the
   job-search targets; a Korean edition serves SAIT-internal influence and domestic
   seminar reuse but roughly +15–20% effort even with AI-assisted translation, and
   drift between editions becomes a maintenance liability.
3. **Experiment scope & compute budget.** Angles A1–A3 range from a few hundred GPU-hours
   (A1 kernel benchmarking, A2 at 100–150M) to several thousand (A3 scaling fits at
   3–4 sizes). Need: what hardware is actually available (personal, SAIT, cloud
   credits?) and whether SAIT IP policy permits publishable runs. Tradeoff: with zero
   experiments the book is still viable (survey+analysis) but Part III drops to
   position-paper defensibility; with A1+A2 (~modest budget) two chapters gain
   first-party numbers and one venue paper becomes extractable.
4. **Positioning: study paper vs survey vs position paper.** Study (pedagogical, Part I
   heavy) serves the inference-engineer audience and is unique — no such on-ramp
   exists; survey (exhaustive related-work, Veridraft snowballing) competes with
   inevitable 2026 TTT surveys on speed; position (the completed-form + hardware
   thesis) is highest-risk/highest-distinctiveness. The current draft ToC is
   study-first with a position-paper Part III; confirm that blend and the Part I/III
   ratio (62:52 as drafted).
5. **Publication strategy & sequencing for a 150–200 p. artifact.** Serial release
   (Part I as a standalone tutorial early → feedback → Parts II–III) builds audience
   and de-risks, but exposes the framing to be scooped, and the line itself is moving
   (a 7th paper would force revision — note Titans' promised "larger models" paper is
   still outstanding). Single big-bang release is coherent but delays any visibility
   6–9 months. Also decide: engage Veridraft gating from day one (claims/bundle.json
   per §`veridraft-tooling.md`) or only for Part III quantitative claims.
6. **Relationship to the career track.** The topic sits exactly on the ML-systems
   profile being marketed (memory-centric systems + inference infra). Decide whether
   the artifact is optimized as a portfolio centerpiece (favors English, arXiv-first,
   A1 systems angle, fast partial releases) or as a long-term research program opener
   (favors completeness, experiments, venue papers). These pull scheduling in opposite
   directions and should be settled before the ToC is frozen.

---

*Sources: the six deep-read JSONs in `notes/` (all claims above trace to their
`tldr`/`core_mechanism`/`connections`/`open_questions` fields), `notes/prereq-curriculum.md`
(Part I budgets), `notes/veridraft-tooling.md` (gating workflow). No external claims
introduced beyond the six papers except explicitly flagged strategy context.*
