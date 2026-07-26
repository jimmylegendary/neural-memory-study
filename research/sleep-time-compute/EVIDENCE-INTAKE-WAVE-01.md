# Sleep-Time Compute Evidence Intake — Wave 01

- Intake date: 2026-07-25
- Scope: high-leverage sources discovered or promoted after the pre-research
  dossier
- Status vocabulary:
  - `A-ABS`: primary abstract and bibliographic record verified
  - `A-HTML`: primary HTML body inspected
  - `FULL`: full paper and supplement read
  - `REPRO`: associated result reproduced
- Rule: no `A-ABS` item supports a claim more specific than its abstract.

## Chronological intake

| ID | Earliest verified public date | Source | Evidence status | Why it enters the core |
|---|---:|---|---|---|
| W1-001 | 2016-10-03 | [Computational principles of synaptic memory consolidation](https://doi.org/10.1038/nn.4401) | A-HTML | Multi-timescale synaptic consolidation is a biological and mathematical capacity prior, not an LLM implementation recipe. |
| W1-002 | 2018-10-22 | [Prioritized memory access explains planning and hippocampal replay](https://doi.org/10.1038/s41593-018-0232-z) | A-HTML | Supplies the normative gain-times-need replay principle that the project generalizes to future-use-conditioned memory routing. |
| W1-003 | 2020-06-09 | [Optimal Continual Learning has Perfect Memory and is NP-hard](https://proceedings.mlr.press/v119/knoblauch20a.html) | A-ABS | Establishes an impossibility/complexity boundary against universal perfect bounded-memory claims. |
| W1-004 | 2020-10-15 | [Optimal forgetting: Semantic compression of episodic memories](https://doi.org/10.1371/journal.pcbi.1008367) | A-HTML | Gives a rate-distortion account of graceful, gist-directed forgetting rather than random decay. |
| W1-005 | 2022-04-22 | [Memory Bounds for Continual Learning](https://arxiv.org/abs/2204.10830) | A-ABS | In the PAC framework, the abstract states a linear-in-\(k\) memory lower bound and a logarithmic-pass multiplicative-weights result, with improper learning necessary for the latter performance. |
| W1-015 | 2024-11-08 | [Generative Adapter: Contextualizing Language Models in Parameters with A Single Forward Pass](https://arxiv.org/abs/2411.05877) | A-ABS | Maps context directly to generated low-rank adapters, preempting a generic claim that document or user context can first be compiled into parametric state. |
| W1-016 | 2025-04-17 | [Sleep-time Compute: Beyond Inference Scaling at Test-time](https://arxiv.org/abs/2504.13171) | A-ABS | Defines query-anticipating offline compute and multi-query amortization; central prior for separating deferred computation from memory consolidation. |
| W1-006 | 2025-06-06 | [Cartridges: Lightweight and general-purpose long context representations via self-study](https://arxiv.org/abs/2506.06266) | A-ABS | Direct latent-memory baseline: an offline-trained corpus-specific KV cache, synthetic self-study data, and context distillation with reuse amortization. |
| W1-007 | 2025-09-13 | [Scaling Law for Catastrophic Forgetting via Gradient Products](https://openreview.net/forum?id=68TggRP3Bb) | A-ABS | Candidate parametric-interference law; the reported \(1/d\) relation depends on orthogonal output heads and teacher-student assumptions. |
| W1-017 | 2025-09-19 OpenReview | [Language Models Need Sleep: Learning to Self Modify and Consolidate Memories](https://openreview.net/forum?id=iiZy6xyVVE) | A-ABS | Direct parametric-sleep prior with upward distillation, parameter expansion, and RL-generated dream curricula. |
| W1-008 | 2025-09-19 | [Continual Learning via Sparse Memory Finetuning](https://arxiv.org/abs/2510.15103) | A-ABS | Direct sparse-parametric destination baseline using selectively activated memory-layer slots; the ICLR 2026 record is a desk-rejected submission, not an accepted paper. |
| W1-018 | 2026-02-03 | [LatentMem: Customizing Latent Memory for Multi-Agent Systems](https://arxiv.org/abs/2602.03036) | A-ABS | Learns an agent-conditioned composer from a raw experience bank into compact latent memory using task-level optimization signals. |
| W1-019 | 2026-02-05 | [Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory](https://arxiv.org/abs/2602.06025) | A-ABS | BudgetMem supplies an RL-trained cost/performance router across runtime memory-module budget tiers, but not across durable memory media. |
| W1-020 | 2026-02-13 | [Doc-to-LoRA: Learning to Instantly Internalize Contexts](https://arxiv.org/abs/2602.15902) | A-ABS | Meta-learns single-pass context-to-LoRA compilation, a direct baseline for fast parametric destination cost. |
| W1-009 | 2026-04-29 | [When Continual Learning Moves to Memory: A Study of Experience Reuse in LLM Agents](https://arxiv.org/abs/2604.27003) | A-ABS | Shows the stability-plasticity problem can reappear as representation and retrieval competition in external memory. |
| W1-010 | 2026-05-06 | [Sharp Capacity Thresholds in Linear Associative Memory: From Winner-Take-All to Listwise Retrieval](https://arxiv.org/abs/2605.05189) | A-ABS | Gives restricted but sharp capacity transitions: \(d^2\asymp n\log n\) for top-1 and \(d^2\asymp n\) for its listwise criterion under isotropic Gaussian assumptions. |
| W1-021 | 2026-05-08 | [Human-Inspired Memory Architecture for LLM Agents](https://arxiv.org/abs/2605.08538) | A-ABS | Combines an explicit sleep phase, interference forgetting, maturation, reconsolidation, graphs, and hybrid retrieval in one external-memory pipeline. |
| W1-022 | 2026-05-15 | [RecMem: Recurrence-based Memory Consolidation for Efficient and Effective Long-Running LLM Agents](https://arxiv.org/abs/2605.16045) | A-ABS | Directly studies when to invoke costly LLM consolidation and reports recurrence-triggered external-memory construction savings. |
| W1-023 | 2026-05-20 | [Auto-Dreamer: Learning Offline Memory Consolidation for Language Agents](https://arxiv.org/abs/2605.20616) | A-ABS | Learned periodic external consolidation over typed, provenance-linked memory with task-reward training and supersession of the original region. |
| W1-024 | 2026-05-25 | [Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference](https://arxiv.org/abs/2605.26099) | A-ABS | Converts recent context into persistent fast weights by repeated offline passes before clearing the KV cache, directly covering a latent/fast-weight sleep operator. |
| W1-025 | 2026-06-03 | [Scaling Self-Evolving Agents via Parametric Memory](https://arxiv.org/abs/2606.04536) | A-ABS | TMEM combines explicit compressed memory with online fast-LoRA absorption and RL-trained extraction, strongly narrowing external-to-parametric novelty. |
| W1-026 | 2026-06-03 | [Cartridges at Scale: Training Modular KV Caches over Large Document Collections](https://arxiv.org/abs/2606.04557) | A-ABS | Scales modular latent/KV artifacts with distractor mixing and a GPU/persistent-storage budget manager. |
| W1-012 | 2026-06-04 | [Agent Memory: Characterization and System Implications of Stateful Long-Horizon Workloads](https://arxiv.org/abs/2606.06448) | A-HTML | Closest systems neighbor: profiles construction, retrieval, generation, freshness, footprint, energy, and scaling for ten agent-memory systems, including a long-context baseline with no external representation. |
| W1-011 | 2026-06-07 (CVPR day; earlier public date unresolved) | [Smart Replay: Adaptive Scheduling of Memory Rehearsal for Computational Resource-Aware Incremental Learning](https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_Smart_Replay_Adaptive_Scheduling_of_Memory_Rehearsal_for_Computational_Resource-Aware_CVPR_2026_paper.pdf) | A-ABS | Compute-budgeted replay scheduling via an optimal-control formulation; a direct comparison point for adaptive sleep allocation. |
| W1-027 | 2026-06-09 | [Learning What to Remember: Observability-Safe Memory Retention via Constrained Optimization for Long-Horizon Language Agents](https://arxiv.org/abs/2606.10616) | A-ABS | OSL-MR models budget feasibility, query-conditioned evidence utility, and delayed miss, reacquisition, and staleness costs under an observable/offline label split. |
| W1-013 | 2026-06-23 | [TRUSTMEM: Learning Trustworthy Memory Consolidation for LLM Agents with Long-Term Memory](https://arxiv.org/abs/2606.25161) | A-HTML | Transition-level coverage, preservation, and faithfulness verification plus preference-guided RL; direct safety and compiler-training baseline. |
| W1-014 | 2026-07-20 | [Retain or Consolidate? Budget-Dependent Operator Selection for Language Agent Memory](https://arxiv.org/abs/2607.17545) | FULL | Closest operator-selection neighbor: reports a relative-budget crossover and learns among retain/merge/abstract/rewrite actions. |

## Evidence-verified claim cards

### W1-001 — Benna and Fusi

- Domain: biological/theoretical consolidation.
- Core question supported: how multiple timescales can extend memory lifetime
  under bounded synaptic resources.
- Permitted current use: motivation and a candidate multiscale prior.
- Not yet permitted: transferring its capacity exponent directly to neural
  language models.
- Next action: read the mathematical model, its assumptions, and the capacity
  comparison in full.

### W1-002 — Mattar and Daw

- Domain: normative hippocampal replay and reinforcement learning.
- Primary result inspected: replay is prioritized by expected utility for future
  decisions, balancing imminent need and gain from propagating information.
- Project extension: replace state-action backup value with destination-specific
  future task gain minus compilation, storage, interference, staleness, and
  governance cost.
- Evidence boundary: the extension is H1; the source does not study LLM memory,
  sleep clusters, or cross-medium promotion.

### W1-003 — Knoblauch et al.

- Domain: general continual-learning theory.
- Abstract-supported result: generally optimal continual learning requires
  perfect memory and solves an NP-hard problem.
- Use: impossibility boundary and explanation for why replay or stored exemplars
  remain competitive.
- Caution: “generally” and the formal assumptions must be recovered from the
  paper before stating a theorem in the manuscript.

### W1-004 — Nagy et al.

- Domain: cognitive memory and information theory.
- Inspected result: rate-distortion theory yields a continuum from verbatim to
  gist-like memory; optimal forgetting moves toward lower rate while preserving
  task-relevant structure according to the distortion function.
- Use: basis for future-query distortion and graceful external-memory
  compression.
- Caution: the paper models human memory distortions with generative models; it
  does not establish an agent-memory policy.

### W1-005 — Chen, Papadimitriou, and Peng

- Domain: PAC continual learning and communication complexity.
- Abstract-supported result: in the PAC framework, the authors state that any
  continual learner, even an improper one, needs memory growing linearly with
  \(k\); with logarithmically many passes they give a multiplicative-weights
  algorithm whose memory scales well, and state that improper learning is
  necessary for that performance.
- Use: repeated sleep passes may change theoretical feasibility, but do not
  imply free compression.
- Caution: not a universal lower bound for every practical agent workload.

### W1-006 — Cartridges

- Domain: latent corpus memory.
- Abstract-supported method: train a smaller corpus-specific KV cache offline.
  Naive next-token training is not competitive; self-study generates synthetic
  conversations and applies context distillation.
- Abstract-reported result: matches in-context learning on evaluated
  long-context tasks while using 38.6x less memory and enabling 26.4x higher
  throughput.
- Use: a direct sleep-time latent destination and a concrete
  distillation/dataset-augmentation recipe.
- Caution: corpus-level repeated queries are not automatically equivalent to
  cross-session continually changing personal memory.

### W1-007 — Gradient-products scaling

- Domain: parametric forgetting theory.
- Abstract-supported result: a gradient-product proxy tracks forgetting in
  linear and nonlinear teacher-student models; orthogonal output heads underpin
  a reported \(1/d\) scaling, while other factors modulate it.
- Use: candidate mechanism for the parametric tier.
- Caution: ICLR 2026 submission, restricted model, and no warrant for calling
  \(1/d\) a universal LLM law.

### W1-008 — Sparse Memory Finetuning

- Domain: sparse parametric consolidation.
- Abstract-supported method: update memory-layer slots highly activated by new
  knowledge relative to pretraining use.
- Abstract-reported result: at matched new-knowledge acquisition in two QA
  settings, NaturalQuestions F1 drops 89% for full tuning, 71% for LoRA, and 11%
  for the proposed sparse method.
- Use: a parametric baseline and evidence that destination sparsity can reduce
  interference.
- Caution: two QA tasks, an ICLR 2026 desk-rejected submission/preprint, and no
  demonstrated lifelong capacity, eviction, rollback, or per-user serving
  story. The venue status comes from the
  [OpenReview record](https://openreview.net/submissions?page=127&venue=ICLR.cc%2F2026%2FConference),
  not from arXiv.

### W1-009 — Continual learning moves to memory

- Domain: external procedural memory.
- Abstract-supported result: under a limited context, old and new experiences
  compete during retrieval; abstract procedural memories can transfer more
  reliably than detailed trajectories, and negative transfer harms difficult
  cases.
- Use: external memory does not remove interference; it relocates it to
  representation and access.
- Caution: the record labels itself “working in progress.”

### W1-010 — Sharp linear-memory capacity

- Domain: associative-memory theory.
- Abstract-supported result: in an isotropic Gaussian key-value model, top-1
  retrieval has a sharp \(d^2\asymp n\log n\) scale while the paper's listwise
  tail-average-margin criterion has \(d^2\asymp n\).
- Use: motivates capacity knees and retrieval-criterion dependence.
- Caution: Gaussian data, linear memory, and the specified decoding criteria;
  the exponent is not imported into natural-language memory without testing.

### W1-011 — Smart Replay

- Domain: compute-aware incremental learning.
- Abstract-supported method: optimize replay ratio as an optimal-control problem
  balancing new-task and memory losses under a compute budget, with a practical
  adaptive heuristic.
- Abstract-reported result: adaptive replay-ratio scheduling outperforms
  fixed-replay baselines under the same computational budget.
- Use: direct baseline for sleep-compute allocation.
- Caution: computer-vision incremental learning rather than deployed LLM memory.

### W1-012 — Agent Memory systems characterization

- Domain: agent-memory systems, including an external-representation-free
  long-context baseline.
- Inspected results:
  - LLM-mediated construction can exceed total query-phase energy over 300
    queries;
  - median decode share of construction tokens is 4.6%, making construction a
    repeated long-read, short-write workload;
  - if cumulative construction and retrieval exceed the session interval, a
    system cannot simultaneously keep zero staleness and hide that latency;
  - none of the ten evaluated systems prunes or forgets by default;
  - agentic construction token cost can grow super-linearly with history while
    indexed retrieval remains nearly flat.
- Use: closest systems baseline and a strong reason to separate latency-critical
  wake scheduling from throughput-oriented construction.
- Novelty boundary: this project must add explicit sleep, cross-medium routing,
  parametric/latent tiers, lifetime capacity, deletion lineage, and learned
  utility—not repeat external-memory profiling.

### W1-013 — TrustMem

- Domain: trustworthy external consolidation.
- Inspected method: a frozen-LLM Memory Transition Verifier evaluates coverage,
  preservation, and faithfulness; candidate transitions produce preference
  pairs under the same state for Transition-Ranked GRPO. Its reward also
  includes terminal task utility, compactness, action executability, and content
  specificity.
- Use: verifier, reward, and safety baselines for the sleep compiler.
- Inspected training scope: 562 balanced instances from a 4,139-instance pool,
  with 463 held-out validation instances; the method is evaluated on
  MemoryAgentBench, HaluMem, and the Mem-\(\alpha\) validation set.
- Reliability boundary: transition omission/corruption/hallucination rates are
  judged by GPT-4o-mini at temperature zero; these are not exact ground truth or
  an independent human safety certificate.
- Comparator boundary: the appendix mixes Qwen2.5-7B, Qwen3-4B/8B/32B, and
  system frameworks with unfixed backbones across applicability-dependent
  tables. Its headline ranking is therefore not a matched-backbone lifecycle
  comparison.
- Permitted numbers after table inspection: HaluMem extraction F1 is 69.45,
  12.14 points above the strongest table comparator; removing transition
  verification changes the reported aggregate validation score from 66.3 to
  58.7. These are source-reported, not reproduced.

### W1-014 — Retain or Consolidate?

- Domain: external-memory operator selection.
- Fully inspected theory: utility decomposes into coverage of evidence
  omitted by retention and signed replacement of raw evidence that already
  fits, but exactly only for an idealized surrogate under a localized retrieval
  assumption—not as an identity for binary LLM-judge accuracy.
- Inspected result: tight budgets favor consolidation while loose
  budgets favor retention on LongMemEval, with a smaller crossover on LoCoMo;
  merge and cross-note abstraction generally beat local rewrite when compression
  is required.
- Controlled protocol: all actions receive the same gold evidence, candidate
  cluster, answer model, and grader. LongMemEval uses 32/64/128/256-token
  budgets and three paired realizations for the primary DeepSeek-V3.2
  evaluation; LoCoMo uses 16/32/64/128-token budgets. The GLM-5.2 replication
  and full-history diagnostics use one realization. Question-level
  bootstrap/randomization and BM25-retrieval checks test robustness.
- Concrete anchor: at 32 tokens, Abstract is 0.520 versus retention 0.040,
  a +0.480 paired gain with 95% interval [+0.373, +0.582]. At 256 tokens
  Abstract is -0.080 relative to retention, and the cross-model replication
  gives -0.147 [-0.267, -0.027].
- Router boundary: grouped cross-fitting supports the within-benchmark
  when/which mechanism, but on the independent full-history split fixed Merge
  reaches 0.457 macro accuracy, direct ridge 0.447, and the frozen safe router
  0.393; the paper explicitly says transferable end-to-end routing is not
  established.
- Safety boundary: zero observed harm on a calibration split is a conservative
  operating rule, not a population safety certificate.
- Use: direct evidence for a pressure-conditioned crossover and a required
  baseline for operator selection.
- Novelty boundary: extend from external operators to latent/parametric
  destinations, full lifecycle and rollback cost, volatile memory, queue
  stability, and hundreds of wake-sleep cycles.

### W1-015 — Generative Adapter

- Domain: amortized context-to-parameter compilation.
- Abstract-supported method: a self-supervised adapter generator maps an unseen
  context to a low-rank adapter in one forward pass rather than running
  per-context finetuning.
- Use: compilation-cost baseline for turning documents, demonstrations, or user
  context into parametric state.
- Boundary: one-shot contextualization is not a versioned lifetime policy for
  promotion, interference, supersession, rollback, or deletion.

### W1-016 — Sleep-time Compute

- Domain: anticipatory deferred computation.
- Abstract-supported result: anticipating what queries users might ask and
  precomputing useful quantities before the real query reduces required
  test-time compute; multiple related queries amortize the offline work.
- Use: founding operational prior and the reason the project must measure query
  predictability, reuse, and sleep-to-wake cost transfer.
- Boundary: precomputation about a fixed context is not automatically
  consolidation into a durable external, latent, or parametric memory medium.

### W1-017 — Language Models Need Sleep

- Domain: parametric consolidation and synthetic dreaming.
- Abstract-supported method: a parameter-expansion stage uses RL-based upward
  distillation (“Knowledge Seeding”), followed by an RL-generated synthetic
  curriculum for rehearsal and self-improvement.
- Use: direct comparator for parametric sleep, capacity expansion, distillation,
  and dream-data construction.
- Boundary: the OpenReview record is an ICLR 2026 submission; the abstract does
  not establish matched external-memory baselines, lifelong deletion, or
  lifecycle serving cost.

### W1-018 — LatentMem

- Domain: task-trained latent memory.
- Abstract-supported method: retain raw trajectories in an experience bank,
  then use an agent-conditioned composer to synthesize compact latent memories;
  LMPO sends task-level optimization signals into the composer.
- Use: learned external-to-latent destination and task-relevant
  rate-distortion baseline.
- Boundary: multi-agent customization is not yet a many-cycle promotion,
  eviction, or cross-medium routing study.

### W1-019 — BudgetMem

- Domain: query-aware runtime memory budgeting.
- Abstract-supported method: a compact RL policy routes among low/mid/high
  budget tiers across memory modules to control an accuracy–construction-cost
  frontier.
- Use: required learned-router control.
- Boundary: it routes runtime effort inside an external-memory pipeline rather
  than deciding where a memory item should persist across external, latent, and
  parametric media.

### W1-020 — Doc-to-LoRA

- Domain: meta-learned context distillation.
- Abstract-supported method: a hypernetwork performs approximate context
  distillation in one forward pass by generating a LoRA adapter for an unseen
  prompt.
- Use: fast user-local parametric compiler and wake/sleep placement baseline.
- Boundary: the abstract does not establish bounded lifelong adapter capacity,
  interference across sequential contexts, or deletion propagation.

### W1-021 — Human-Inspired Memory Architecture

- Domain: biologically inspired external agent memory.
- Abstract-supported method: combines sleep-phase consolidation,
  interference-based forgetting, engram maturation, reconsolidation, entity
  graphs, and hybrid retrieval.
- Use: system-level baseline for an explicit sleep phase and forgetting.
- Boundary: a pipeline with many cognitive labels is not evidence for a
  cross-medium destination oracle or a universal biological mapping.

### W1-022 — RecMem

- Domain: adaptive external-memory consolidation trigger.
- Abstract-supported method: store interactions in a lightweight subconscious
  embedding layer and invoke an LLM to extract episodic/semantic memory only
  after sustained semantic recurrence.
- Abstract-reported result: up to 87% lower construction-token cost for three
  compared memory systems while exceeding their accuracy.
- Boundary: recurrence narrows “when should memory sleep?” novelty, but the
  destination remains external and the abstract does not give a matched
  lifecycle router across media.

### W1-023 — Auto-Dreamer

- Domain: learned offline external consolidation.
- Abstract-supported method: inspect a read-only typed working region and its
  provenance-linked trajectories, then synthesize a compact replacement that
  supersedes the sources; GRPO uses downstream agent performance as reward.
- Use: strongest direct baseline for cross-session external sleep, provenance,
  replacement, and reward-trained consolidation.
- Boundary: it does not compare external abstraction with reusable latent/KV or
  user-parametric destinations under one lifecycle budget.

### W1-024 — Offline Recurrence

- Domain: architecture-native fast-weight sleep.
- Abstract-supported method: periodically run \(N\) offline recurrent passes
  over recent context, update persistent SSM fast weights using a learned local
  rule, and then clear the KV cache while preserving wake prediction latency.
- Use: direct latent/fast-weight operator and sleep-duration scaling baseline.
- Boundary: architecture-specific evicted-context reasoning is not a
  versioned, deletable personal-memory system or a cross-medium router.

### W1-025 — TMEM

- Domain: explicit plus parametric within-episode memory.
- Abstract-supported method: compress history explicitly and absorb distilled
  supervision into online fast LoRA weights; an RL-trained extraction policy
  improves both task actions and the data used for adaptation.
- Use: strongest threat to a generic external-to-parametric-memory novelty
  claim.
- Boundary: the reported formulation is online and within one episode; it does
  not establish deferred many-cycle destination choice with full archival,
  serving, rollback, staleness, and deletion accounting.

### W1-026 — Cartridges at Scale

- Domain: scalable modular latent/KV memory.
- Abstract-supported method: dynamic distractor mixing trains composable
  per-document cartridges, while a budget manager rotates hundreds of modules
  between GPU and persistent storage.
- Use: direct baseline for latent capacity, retrieval-to-latent composition,
  residency, and offload.
- Boundary: static document collections do not cover volatile, user-scoped
  lifetime memory or parametric promotion.

### W1-027 — OSL-MR

- Domain: observability-safe external retention.
- Abstract-supported method: constrained stochastic optimization combines
  budget feasibility, evidence utility, delayed miss/reacquisition/staleness
  penalties, and strict separation between online-observable features and
  offline supervision.
- Use: required future-use/lifecycle-cost policy baseline and leakage-control
  template.
- Boundary: it decides external retention, not semantic consolidation or
  destination among raw external, abstract, latent, and parametric memory.

## Immediate full-read order

1. W1-023, W1-027, and W1-014 — closest external consolidation, retention
   policy, and operator-selection overlap;
2. W1-025, W1-017, W1-020, and W1-015 — parametric sleep and
   context-to-adapter compilation;
3. W1-024, W1-026, W1-018, and W1-006 — latent/fast-weight destination,
   capacity, and amortization;
4. W1-022, W1-016, and W1-019 — trigger, anticipatory compute, and learned
   budget routing;
5. W1-012 and W1-013 — systems accounting, verifier, and RL data construction;
6. W1-021, W1-009, and W1-011 — integrated external architecture,
   representation interference, and compute-aware replay;
7. W1-008 — sparse parametric baseline and forgetting protocol;
8. W1-005 and W1-003 — impossibility boundaries;
9. W1-004 and W1-002 — normative derivation;
10. W1-010 and W1-007 — capacity and interference scaling.

## Wave-01 novelty decision

The defensible paper claim is not that consolidation is useful under pressure:
W1-014 already directly addresses that question for external memory. It is not
that memory construction has a lifecycle cost: W1-012 already measures it.

As of the intake date, the safer remaining gap is:

> We found no primary study that jointly learns and evaluates, under one matched
> lifetime cost model, routing of the same memory items among raw external
> retention, external abstraction, reusable latent/KV state, and user-local or
> shared parametric state over many wake–sleep cycles, while preserving version
> lineage and charging verification, staleness, serving, data movement,
> rollback, and deletion.

Each component has substantial prior art. The proposed novelty is their
controlled cross-medium comparison and lifecycle-aware routing, not
future-use-aware retention, learned consolidation, sleep, dreaming, or
external-to-parametric transfer individually. This question remains conditional
on full-paper reads and code inspection; if any neighbor already performs the
matched comparison, G1 narrows or kills the claim before confirmatory work.
