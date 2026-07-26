# Company-Lineage Primary-Source Audit for Sleep-Time Compute

- Audit date: 2026-07-25
- Scope: Google/DeepMind, Meta/FAIR, Microsoft Research, and the deployed
  agent-memory lineages most often cited as sleep-time systems
- Sources: papers, proceedings, OpenReview records, official repositories,
  tagged source, release records, and official product documentation
- Status: pre-ingestion review artifact; numerical claims must still be split
  into atomic evidence records before registry promotion
- Companion:
  [`PRIMARY-SOURCE-AUDIT-AGENT-MEMORY-SYSTEMS.md`](PRIMARY-SOURCE-AUDIT-AGENT-MEMORY-SYSTEMS.md)
  contains the implementation-level product audit from which the concise
  Letta/Mem0/Zep/LangMem entries below are derived

## 1. Audit contract

### 1.1 Strict lifecycle criteria

A system is strict sleep-time compute (`STC`) only if all three conditions are
demonstrated:

| Criterion | Required evidence |
|---|---|
| `C1` | Accumulated wake experience is an input to the transform. A static pretraining or benchmark corpus is not deployed wake experience. |
| `C2` | The transform runs after the current visible action, outside its causal response path, or against an isolated asynchronous snapshot. Merely using `async def` or a method named “background” is insufficient. |
| `C3` | The transformed state is durable and is read by a later wake. Same-token or same-call hidden state does not suffice. |

Cell values are `Y` (shown), `P` (partly shown or only at a task/buffer
boundary), `N` (not shown), and `N/A`. A `PASS` is a lifecycle result, not a
claim of production readiness or biological fidelity.

This audit distinguishes two PASS evidence tiers. `EXPERIMENTAL` means a
controlled sequential stream instantiates wake, off-path transformation, and
later reuse. In applied-product rows, unqualified `PASS` is shorthand for
`IMPLEMENTED`: released code or versioned product documentation implements
that lifecycle. Neither tier alone proves production deployment, workload
generality, or operational reliability. A static pretraining corpus remains
`C1=N`; a benchmark stream counts only when the method actually accumulates
state from its sequential wake episodes.

### 1.2 Phase labels

| Label | Meaning |
|---|---|
| `Q` | query-time or within-response transformation |
| `W` | live wake-time/per-token/per-step update |
| `B` | task, buffer, overflow, or compaction boundary |
| `S` | query-independent deferred/background transform with later reuse |
| `P` | proposal, static substrate, benchmark, or predeployment training |

Task-boundary post-training can be a useful `B/S` precursor without proving a
recurring deployed-lifetime sleep service. “Persistent” also has two meanings
that must not be conflated: persistent for later tokens in one sequence, and
persistent for a later user/agent wake across a lifecycle boundary.

### 1.3 Evidence hygiene

1. Earliest-public date means the earliest primary record found in this audit.
   Where an anonymous OpenReview submission predates attribution, both dates
   are stated.
2. Venue status is reported as of the audit date. A submission is not described
   as accepted.
3. Company lineage means at least one author affiliation or an official product
   owner. It does not imply that the company deployed the method.
4. Capacity, deletion, rollback, provenance, privacy, and production scheduling
   are recorded as separate claims. A bounded prompt, active cache, or top-k
   result is not a bounded lifetime store.
5. “No paper found” is scoped negative evidence, not proof that no internal or
   unpublished system exists.

## 2. Executive result

The audited evidence does **not** support the claim that all three companies
already operate a general sleep-time learning stack.

- **Google/DeepMind** supplies the strongest parametric research line. The
  decisive strict paper is
  [*Language Models Need Sleep*](https://openreview.net/forum?id=iiZy6xyVVE):
  wake-derived fast memory is consolidated off-wake into newly activated
  low-rank experts and reused later. The paper is an ICLR 2026 submission, not
  an accepted ICLR paper in the public record inspected here. The implementation
  preallocates a finite expert pool and does not solve lifetime reclamation,
  deletion, rollback, or safe promotion. Google's later NSTM work decouples
  periodic fast-weight updates from per-frame application, but remains live
  video-stream TTT and does not add a deferred cross-session sleep phase.
- **Meta/FAIR** supplies strong continual-learning regularizers and high-capacity
  sparse parametric memory substrates, but this targeted audit found no public
  Meta paper satisfying `C1+C2+C3` for a recurring deployed sleep phase.
  PAHF is persistent personalization, but its update is foreground
  write-through. A joint FAIR/Google DeepMind/Cornell/NVIDIA study estimates
  GPT-style memorization capacity under a synthetic no-generalization
  protocol; it is a capacity anchor, not a sleep lifecycle or a universal
  bits/parameter law.
- **Microsoft** supplies the strongest explicit scheduled external-memory
  lifecycle architecture:
  [HIMA](https://arxiv.org/abs/2605.08538) includes a scheduled consolidation
  pipeline and passes the strict lifecycle test at the architecture level.
  Its “every six hours” setting is a design default, not an empirically executed
  six-hour deployment. Only four of its six named mechanisms receive meaningful
  ablation evidence. Three Microsoft-linked systems add narrower
  task/session-linked evidence:
  [H-EPM](https://arxiv.org/abs/2512.07287) connects successful test
  trajectories, a hybrid tool graph, and a separate GRPO loop, but does not
  disclose the ACEBench boundary ordering needed for strict sleep; by contrast,
  [PlugMem](https://arxiv.org/abs/2603.03296) inserts completed WebArena task
  trajectories into a reusable knowledge graph after the environment is done,
  while [MemMA](https://arxiv.org/abs/2603.18718) tests and repairs provisional
  memory after each completed LoCoMo session. PlugMem and MemMA are experimental
  boundary passes, not periodic production sleep services. MemMA's released
  quick start also depends on a pre-generated probe file whose generation
  pipeline is not yet public.
- Microsoft
  [MEMENTO](https://arxiv.org/abs/2604.09852) adds a different, strictly
  **within-query** branch: predeployment SFT/RL teaches a model to summarize
  and physically evict completed reasoning blocks during one generation. It
  is not sleep-time learning, but its restart and passcode-probe experiments
  prove that the visible summary text is not the complete retained state:
  block-conditioned summary KV states carry an implicit information channel.
- The independent **Memento project** is a different lineage.
  [Memento 2](https://arxiv.org/abs/2512.22716) and the
  [Memento-Skills paper](https://arxiv.org/abs/2603.18743) are wake-time
  case/skill learning; a post-paper repository release later adds a real
  DreamDaemon. None of these is Microsoft MEMENTO, and the daemon cannot be
  used retroactively as the paper's evaluated mechanism.
- [Retrospective Harness Optimization
  (RHO)](https://arxiv.org/abs/2606.05922) adds a later control-plane
  `EXPERIMENTAL PASS`: completed trajectories are re-solved and diagnosed
  off-path, a persistent instructions/skills/tools harness is selected by
  self-preference, and held-out future tasks consume it. Its one-round,
  same-model, label-free gate is not a safe autonomous promotion service, and
  neither the harness nor its retained run/version archive has a lifetime cap.
- **Adjacent learned-memory control** predates M★:
  [MemEvolve](https://arxiv.org/abs/2512.18746) evolves executable
  encode/store/retrieve/manage providers after task batches,
  [ALMA](https://arxiv.org/abs/2602.07755) searches database/update/retrieval
  designs and tests a post-task dynamic update loop, and
  [MemSkill](https://arxiv.org/abs/2602.02474) learns selection over a bounded
  natural-language skill bank. Their experimental or predeployment passes do
  not establish an autonomous governed production scheduler.
- **Applied memory vendors/frameworks** demonstrate that strict non-parametric
  sleep is already implementable. Letta API, Letta Code reflection, Letta AI
  Memory SDK, Zep Cloud, Mem0 Platform V3, and LangMem's durable background
  executor pass the lifecycle test in their audited configurations. None
  continually trains shared base-model weights in production.

The practical novelty boundary is therefore not “invent sleep.” It is a
versioned, validated, reversible routing system that decides **when**, **where**,
and **how much** wake experience to move among raw episode, text/vector/graph,
latent/KV, user-parametric, and shared-parametric media.

## 3. Company-lineage chronology

This table orders sources by earliest public primary evidence, not by the date
of the latest arXiv revision.

| Earliest public | Lineage | Primary source and final/current status | Phase | Strict result |
|---:|---|---|:---:|---|
| 2016-12-02 | DeepMind | [EWC](https://arxiv.org/abs/1612.00796), PNAS 2017 | `B` | precursor |
| 2017-06-26 | Meta/FAIR | [GEM](https://arxiv.org/abs/1706.08840), NeurIPS 2017 | `W` | PARTIAL |
| 2017-11-27 | KU Leuven + Meta/FAIR | [MAS](https://arxiv.org/abs/1711.09601), ECCV 2018 | `W/B` | PARTIAL |
| 2019-07-10 | Meta/FAIR | [Product-Key Memory](https://papers.nips.cc/paper_files/paper/2019/hash/9d8df73a3cfbf3c5b47bc9b50f214aff-Abstract.html), NeurIPS 2019 | `P` | FAIL |
| 2022-01-28 | Google | [Memorizing Transformers](https://openreview.net/forum?id=TrjbxzRcnf-), ICLR 2022 Spotlight; arXiv followed 2022-03-16 | `W` | FAIL |
| 2023-06-12 | UCSB + Microsoft | [LongMem](https://arxiv.org/abs/2306.07174), NeurIPS 2023 | `W` | FAIL |
| 2023-08-09 | LangChain | [LangGraph repository](https://github.com/langchain-ai/langgraph) begins | `P` | storage alone FAIL |
| 2023-10-12 | MemGPT/Letta lineage | [MemGPT](https://arxiv.org/abs/2310.08560), preprint | `W/Q` | PARTIAL |
| 2024-07-12 | Mem0 | [Mem0 repository](https://github.com/mem0ai/mem0) lineage begins in the renamed Embedchain repository | `W/Q` | product-dependent |
| 2024-08-13 | Zep | [Graphiti repository](https://github.com/getzep/graphiti) begins | `W/Q` | OSS PARTIAL |
| 2024-10-10 | UW + Microsoft | [GenerativeAdapter](https://arxiv.org/abs/2411.05877), AFM 2024 Oral, then ICLR 2025 Poster | `Q/W` | FAIL |
| 2024-12-12 | Meta/FAIR | [Memory Layers at Scale](https://arxiv.org/abs/2412.09764), ICML 2025 Poster | `P` | FAIL |
| 2024-12-31 | Google | [Titans](https://arxiv.org/abs/2501.00663), NeurIPS 2025 Poster | `W` | FAIL |
| 2025-01-20 | Zep | [Zep paper](https://arxiv.org/abs/2501.13956), preprint | `W/S` | Cloud PASS; OSS PARTIAL |
| 2025-01-21 | LangChain | [LangMem repository](https://github.com/langchain-ai/langmem) begins; `ReflectionExecutor` added 2025-02-05 | `S` | PASS with durable executor |
| 2025-04-08 | Google/DeepMind | [Lattice](https://arxiv.org/abs/2504.05646), ES-FoMo III workshop; ICLR 2026 submission remains unaccepted in the record inspected | `W` | FAIL |
| 2025-04-17 | Letta | [Sleep-time Compute](https://arxiv.org/abs/2504.13171), preprint | `S` | PARTIAL |
| 2025-04-17 | Google | [MIRAS](https://arxiv.org/abs/2504.13173), ICLR 2026 Poster | `W` | FAIL |
| 2025-04 (lineage) | Letta | [Sleeptime Agent implementation](https://github.com/letta-ai/letta/blob/0.16.8/letta/groups/sleeptime_multi_agent_v4.py), audited tag `0.16.8` | `S` | PASS |
| 2025-04-28 | Mem0 | [Mem0 paper](https://arxiv.org/abs/2504.19413), ECAI 2025 | `W/S` | PARTIAL |
| 2025-05-29 | Google | [ATLAS](https://arxiv.org/abs/2505.23735), ICML 2026 | `W` | FAIL |
| 2025-05-30 | Meta FAIR + Google DeepMind + Cornell + NVIDIA | [How much do language models memorize?](https://arxiv.org/abs/2505.24832), preprint | `P` | conditional parametric-capacity measurement; strict FAIL |
| 2025-06-12 | MIT, adjacent comparator | [SEAL](https://arxiv.org/abs/2506.10943), NeurIPS 2025 Spotlight | `Q/B` | precursor |
| 2025-07-08 | Google | [Trellis](https://openreview.net/forum?id=r61s1FNYlj), COLM 2025 | `W` | FAIL |
| 2025-08-30 | Letta | [AI Memory SDK](https://github.com/letta-ai/ai-memory-sdk) begins | `S` | PASS for per-turn extraction |
| 2025-09-18 | Microsoft-majority | [ACON](https://arxiv.org/abs/2510.00615), ICLR 2026 submission withdrawn; ICML 2026 | `P+Q` | precursor/FAIL |
| 2025-09-18 | Google | [Nested Learning](https://arxiv.org/abs/2512.24695), NeurIPS 2025 Poster | `W/P` | FAIL |
| 2025-09-19 | Google + Cornell | [Language Models Need Sleep](https://openreview.net/forum?id=iiZy6xyVVE), submitted to ICLR 2026; arXiv v1 2026-06-02 | `S` | EXPERIMENTAL PASS in benchmark lifecycle |
| 2025-09-19 | Meta/FAIR | [Sparse Memory Finetuning](https://openreview.net/forum?id=LGo7U1m24L), ICLR 2026 desk-rejected submission | `B/P` | precursor |
| 2025-09-20 | Google | [Memory Caching](https://openreview.net/forum?id=B5SkWFRE8U), ICLR 2026 submission, later ICML 2026 | `W` | FAIL |
| 2025-09-20 | Google-majority | [TNT](https://arxiv.org/abs/2511.07343), anonymous ICLR record precedes attributed arXiv v1; ICLR 2026 Poster | `P` | precursor |
| 2025-09-29 | Google Cloud + academia | [ReasoningBank](https://arxiv.org/abs/2509.25140), ICLR 2026 Poster | `B` | EXPERIMENTAL PASS at query/task boundary |
| 2025-10-06 | Microsoft | [LEGOMem](https://arxiv.org/abs/2510.04851), AAMAS 2026 | `P` | precursor |
| 2025-10-24 | Letta | [Letta Code repository](https://github.com/letta-ai/letta-code) becomes public | `S` | PASS |
| 2025-12-08 | HKUST + Microsoft Research Asia | [H-EPM](https://arxiv.org/abs/2512.07287), ICML 2026 | `W/P` | PARTIAL/ordering unresolved; RL is training-loop learning |
| 2025-12-21 | NUS/BUPT/CUHK/OPPO/ByteDance collaboration | [MemEvolve](https://arxiv.org/abs/2512.18746), ICML 2026 | `B/P` | outer architecture EXPERIMENTAL PASS; inner content FAIL |
| 2025-12-27 | independent Memento project | [Memento 2](https://arxiv.org/abs/2512.22716), single-author theory preprint | `W` | FAIL; external case/policy/value wake learning, not Microsoft MEMENTO |
| 2026-02-02 | academic collaboration | [MemSkill](https://arxiv.org/abs/2602.02474), preprint | `P/Q` | predeployment learned skill bank; runtime content FAIL |
| 2026-02-03 | Microsoft | [MEMORA](https://arxiv.org/abs/2602.03315), ICML 2026 | `W/B` | PARTIAL, not strict |
| 2026-02-06 | UIUC + Tsinghua + Microsoft Research | [PlugMem](https://arxiv.org/abs/2603.03296), ICML 2026 | `B/P` | EXPERIMENTAL PASS at WebArena task boundary |
| 2026-02-08 | UBC + Vector Institute | [ALMA](https://arxiv.org/abs/2602.07755), ICLR 2026 workshop oral | `B/P` | outer design and dynamic task-loop EXPERIMENTAL PASS |
| 2026-02-18 | Meta | [PAHF](https://arxiv.org/abs/2602.16173), preprint | `W` | PARTIAL |
| 2026-03-19 | Penn State + Amazon + Microsoft | [MemMA](https://arxiv.org/abs/2603.18718), preprint | `B` | EXPERIMENTAL PASS at LoCoMo session boundary |
| 2026-03-19 | Memento project collaboration | [Memento-Skills](https://arxiv.org/abs/2603.18743), preprint | `W` | FAIL for paper loop; same-question skill rewrite/retry |
| 2026-04-06 | independent SMF successor | [Improving Sparse Memory Finetuning](https://arxiv.org/abs/2604.05248), preprint | `P/B` | open reimplementation/extension, not strict sleep |
| 2026-04-08 | Microsoft | [MEMENTO](https://arxiv.org/abs/2604.09852), official article; arXiv followed 2026-04-10 | `Q/W` | FAIL; learned within-query state compaction |
| 2026-04-10 | City University of Hong Kong + Microsoft | [M★](https://arxiv.org/abs/2604.11811), preprint | `P` | memory-harness search precursor |
| 2026-04-14 | Microsoft-heavy collaboration | [WebXSkill](https://arxiv.org/abs/2604.13318), preprint | `P` | precursor |
| 2026-04-22 | Memento project repository | [Memento-Skills DreamDaemon](https://github.com/Memento-Teams/Memento-Skills/tree/71ac933ea1381d53389a2426f59634e0182071b8), post-paper code | `S` | code-level PASS; no quantitative Dream evidence or transactional publication |
| 2026-04-25 | independent | [Evolve](https://arxiv.org/abs/2604.23424), single-author preprint | `Q/W+S` | strict external consolidation; identical-query evaluation and cross-store durability gaps |
| 2026-05-04 | independent SMF successor | [Sparse Memory Finetuning as a Low-Forgetting Alternative](https://arxiv.org/abs/2605.03229), preprint | `P/B` | controlled reimplementation/extension, not strict sleep |
| 2026-05-08 | USTC + HUST + Microsoft Research + collaborators | [MemCompiler](https://arxiv.org/abs/2605.07594), preprint | `Q/W` | FAIL; learned memory-delivery comparator |
| 2026-05-08 | Microsoft | [HIMA](https://arxiv.org/abs/2605.08538), preprint | `S` | PASS pipeline; mechanisms PARTIAL |
| 2026-05-25 | CMU + UMD, adjacent comparator | [Do Language Models Need Sleep?](https://arxiv.org/abs/2605.26099), preprint | `B/S` | PASS in benchmark lifecycle |
| 2026-06-04 | City University of Hong Kong + Microsoft Research Asia | [RHO](https://arxiv.org/abs/2606.05922), preprint | `B/P` | EXPERIMENTAL PASS for retrospective harness update |
| 2026-06-04 | Microsoft-majority collaboration | [MAGE](https://arxiv.org/abs/2606.06090), preprint | `W/B` | FAIL |
| 2026-07-16 | University of Washington + Google | [Online Neural Space Time Memory](https://arxiv.org/abs/2607.15271), preprint | `W` | FAIL; periodic live TTT update, not sleep |

Current managed products without a stable paper publication date are dated and
versioned in their detailed entries rather than assigned a guessed first-public
date.

## 4. Google and DeepMind lineage

### G1. Elastic Weight Consolidation

- **Primary/status:** Kirkpatrick et al.,
  [*Overcoming Catastrophic Forgetting in Neural Networks*](https://arxiv.org/abs/1612.00796),
  first public 2016-12-02; PNAS 2017; DeepMind authors.
- **Operator:** at a task switch, estimate a diagonal Fisher importance and
  penalize movement of parameters important to old tasks.
- **Training basis:** sequential supervised/RL task data; no replay dataset is
  required for the protected task after its Fisher is computed.
- **Anchor:** ten Atari games are learned sequentially; the Fisher is recomputed
  at task switches. The paper explicitly frames the model as fixed-capacity.
- **Criteria:** `C1=P`, `C2=P`, `C3=Y`; `B` precursor, not demonstrated
  deployed recurring sleep.
- **Capacity/governance nonclaim:** EWC allocates stability inside fixed
  parameters; it neither measures remaining semantic capacity nor supplies
  item-level provenance, deletion, unlearning, or rollback. Accumulating
  quadratic penalties can eventually constrain plasticity.

### G2. Memorizing Transformers

- **Primary/status:** Wu et al.,
  [*Memorizing Transformers*](https://openreview.net/forum?id=TrjbxzRcnf-),
  public ICLR record 2022-01-28, ICLR 2022 Spotlight; all Google.
- **Operator:** append 512-token chunks of internal key/value representations
  to a non-differentiable kNN memory and retrieve at the next layer.
- **Anchor:** C4 perplexity improves from `17.20` to `14.42` with an `8K`
  memory and `14.04` with `65K + XL`; arXiv improves `3.29 -> 2.49 -> 2.31`.
  The study used 32 TPUs.
- **Criteria:** `C1=Y`, `C2=N`, `C3=P`; `W`, strict FAIL. Writes occur while
  processing the sequence and the memory is cleared between documents.
- **Capacity/governance nonclaim:** the evaluated FIFO memory grows to at most
  262K tokens in experiments, but this is a cache setting, not lifetime memory.
  There is no semantic compaction, deletion lineage, rollback, privacy policy,
  or cross-session scheduler.

### G3. Titans

- **Primary/status:** Behrouz, Zhong & Mirrokni,
  [*Titans: Learning to Memorize at Test Time*](https://arxiv.org/abs/2501.00663),
  first public 2024-12-31; NeurIPS 2025 Poster; Google Research.
- **Operator:** a deep neural memory is updated token/chunk-wise at test time
  using surprise, momentum, and weight decay; attention and persistent learned
  tokens complement this fast state.
- **Anchor:** the paper demonstrates needle-in-a-haystack operation beyond a
  2M-token context and evaluates 340M/400M/760M scales. These are sequence
  results, not deployed multi-session learning.
- **Criteria:** `C1=Y`, `C2=N`, `C3=P`; `W`, strict FAIL.
- **Capacity/governance nonclaim:** fixed-shape neural fast state and decay make
  active state bounded, but no lifetime capacity, session durability, off-path
  snapshot, selective deletion, provenance, validation, or rollback is shown.

### G4. Lattice

- **Primary/status:** Karami, Pascanu & Mirrokni,
  [*Lattice*](https://arxiv.org/abs/2504.05646), first public 2025-04-08;
  ES-FoMo III workshop; subsequently submitted to ICLR 2026, with no acceptance
  shown in the inspected record; Google Research + Google DeepMind.
- **Operator:** fixed memory slots receive only the component of an incoming
  representation orthogonal to current slot state, derived as online
  optimization.
- **Anchor:** at 760M scale its common-sense average is `54.03` versus TTT
  `52.42`; with only 16 slots its perplexity is `15.52`, versus TTT with 64
  slots at `15.60`. Lattice reaches `14.71` at 192 slots.
- **Criteria:** `C1=Y`, `C2=N`, `C3=N`; `W`, strict FAIL.
- **Capacity/governance nonclaim:** fixed slots bound physical fast state but
  force interference/forgetting. The paper does not implement cross-session
  publication, delete, rollback, provenance, or a background service.

### G5. MIRAS

- **Primary/status:** Behrouz et al.,
  [*It's All Connected*](https://arxiv.org/abs/2504.13173), first public
  2025-04-17; ICLR 2026 Poster; Google.
- **Operator:** a design framework parameterizes memory architecture,
  attentional-bias objective, retention gate, and online learning rule;
  Moneta, Yaad, and Memora are example recurrent instances.
- **Anchor:** the paper trains 340M through 1.3B models on 4K contexts and
  evaluates up to 32K training contexts and 8K RULER tasks. It supplies a theory
  of retention regularization, not an idle consolidation scheduler.
- **Criteria:** `C1=Y`, `C2=N`, `C3=N`; `W`, strict FAIL.
- **Capacity/governance nonclaim:** the framework explains fixed-state
  interference and forgetting, but not total lifetime capacity, versioned
  promotion, deletion, provenance, or rollback.

### G6. ATLAS

- **Primary/status:** Behrouz et al.,
  [*ATLAS*](https://arxiv.org/abs/2505.23735), first public 2025-05-29; ICML
  2026; Google.
- **Operator:** the Omega rule updates neural memory against a local window of
  current and past tokens rather than one token, with context pruning and
  higher-order optimization approximations.
- **Anchor:** ATLAS retains approximately `80%` accuracy on the 10M-token
  BABILong setting where the cited Titans configuration drops sharply.
- **Criteria:** `C1=Y`, `C2=N`, `C3=N`; `W`, strict FAIL. The paper explicitly
  states that state does not carry to a new independent global context.
- **Capacity/governance nonclaim:** fixed fast state and pruning address
  sequence capacity only. No deployed lifetime store, delete fan-out,
  provenance, canary, or rollback is implemented.

### G7. Trellis

- **Primary/status:** Karami et al.,
  [*Trellis*](https://openreview.net/forum?id=r61s1FNYlj), public
  2025-07-08; COLM 2025; all Google.
- **Operator:** a two-pass recurrent key/value regression update compresses the
  stream into a fixed number of slots, using gradient descent, a forget gate,
  and weight decay.
- **Anchor:** at 790M scale the reported average is `48.38` versus TTT `46.71`;
  RULER average is `79.8` versus Gated DeltaNet `75.8` and TTT `66.1`. Removing
  forgetting worsens perplexity from `10.87` to `11.28`.
- **Criteria:** `C1=Y`, `C2=N`, `C3=N`; `W`, strict FAIL.
- **Capacity/governance nonclaim:** Trellis bounds within-sequence state, not
  cross-session memory. No scheduler, publication, delete, provenance, or
  rollback is evaluated.

### G8. Nested Learning

- **Primary/status:** Behrouz et al.,
  [*Nested Learning*](https://arxiv.org/abs/2512.24695), public NeurIPS record
  2025-09-18; NeurIPS 2025 Poster; Google.
- **Operator:** interprets architectures and optimizers as nested associative
  memories updating at different frequencies; the Hope architecture separates
  fast and slow update levels.
- **Anchor:** the important contribution for this program is the explicit
  frequency hierarchy and optimization-as-memory formalism, not a measured
  user-lifetime background service.
- **Criteria:** `C1=Y`, `C2=N`, `C3=P`; `W/P`, strict FAIL.
- **Capacity/governance nonclaim:** multiple update clocks do not by themselves
  supply isolation, bounded lifetime capacity, atomic promotion, user deletion,
  provenance, or rollback.

### G9. Language Models Need Sleep

- **Primary/status:** Behrouz, Hashemi, Javanmard & Mirrokni,
  [arXiv](https://arxiv.org/abs/2606.03979) v1 2026-06-02, v2 2026-07-10;
  earliest public anonymous
  [OpenReview](https://openreview.net/forum?id=iiZy6xyVVE) record
  2025-09-19, modified 2026-02-11. The inspected record says “Submitted to ICLR
  2026,” not accepted. Affiliations are Google/Cornell.
- **Operator — Knowledge Seeding:** use the current fast/high-frequency model
  as teacher and a prospective slow model plus a newly unmasked low-rank expert
  as student; optimize only the new expert with generalized knowledge
  distillation, publish the sender update, reset old high-frequency experts, and
  activate the slow expert.
- **Operator — Dreaming:** generate task-conditioned candidates; introduce
  novelty through a random irrelevant MoE expert; rank candidates with
  gradient-based utility plus random exploration; LoRA-SFT on selected dreams;
  train the dream generator with a binary downstream reward through ReSTEM.
- **Training/data basis:** generated self-edit curricula, generalized
  distillation, LoRA SFT, RL, semantic rewards, and token-level Levenshtein
  reward. This is the clearest audited combination of distillation, RL, and
  synthetic sleep data.
- **Anchors:**
  - Qwen3-1.7B AIME24/AIME25/HMMT:
    base `49.8/34.5/25.7`, sleep `53.2/40.2/29.3`.
  - Qwen3-8B:
    base `73.8/68.1/42.4`, sleep `79.2/69.0/46.1`.
  - Knowledge incorporation, SQuAD no-context:
    base `31.9/31.9`, SEAL `46.7/43.2`, sleep four-level `48.9/46.2`.
  - Few-shot ARC success:
    ICL `0`, TTT `10`, SEAL `72.5`, Sleep `80`.
  - At matched target performance, ordinary SFT takes `4.3x`, `3.6x`, and
    `4.8x` the wall time on the three reasoning benchmarks.
- **Criteria:** `C1=Y`, `C2=Y`, `C3=Y`; `S`, strict `EXPERIMENTAL` PASS in the
  paper's sequential benchmark lifecycle. The wake state is produced from
  benchmark episodes rather than a static pretraining corpus, but no production
  deployment or recurring service is evidenced.
- **Capacity boundary:** the audited experimental implementation preallocates
  and masks five low-rank MLP blocks of dimension 64 while holding active
  parameter count constant relative to the reported models. This is an
  implementation ceiling, not evidence for five successful lifetime
  consolidations or five independently usable memory slots. The experiments
  do not measure usable pool capacity or unbounded expansion.
- **Governance/reproducibility nonclaim:** no pool-exhaustion experiment,
  expert merge/evict policy, router-saturation study, user deletion, provenance,
  rollback, privacy contract, production scheduler, released code, hardware
  accounting, seeds, or confidence intervals was found. “NREM/REM” is an
  analogy; random-expert mixing does not establish biological REM fidelity.

### G10. Memory Caching

- **Primary/status:** Behrouz et al.,
  [OpenReview](https://openreview.net/forum?id=B5SkWFRE8U), first public
  2025-09-20 as an ICLR 2026 submission; arXiv v1
  [2026-02-27](https://arxiv.org/abs/2602.24281); ICML 2026; Google.
- **Operator:** cache recurrent hidden-state checkpoints and retrieve/aggregate
  them through GRM, Soup, or sparse selective caching (`SSC`), interpolating
  fixed recurrent state and growing attention memory.
- **Anchor:** on 16K NIAH, Titans `75.4/21.2` becomes GRM `88.2/32.2`; at 1.3B,
  the reported aggregate rises `56.82 -> 58.33`.
- **Criteria:** `C1=Y`, `C2=N`, `C3=N`; `W`, strict FAIL.
- **Capacity/governance nonclaim:** SSC can bound the active/load set, but total
  cached checkpoints still grow. No lifetime eviction/deletion, cross-session
  scheduler, provenance, or rollback is shown.

### G11. TNT

- **Primary/status:** Li et al.,
  [*TNT*](https://arxiv.org/abs/2511.07343), attributed arXiv v1
  2025-11-10; the anonymous ICLR submission was public 2025-09-20; ICLR 2026
  Poster; Google-majority + USC.
- **Operator:** stage one pretrains a global memory with large chunks and
  periodically reset parallel local memories; stage two briefly fine-tunes
  locals at the small inference chunk size.
- **Anchor:** Titans with chunk 8 takes `19.48 h`; TNT with chunk 64 takes
  `1.12 h`, a reported `17.37x` speedup. Stage one `3.06 h` plus stage two
  `0.15 h`; perplexity `24.10 -> 23.99`.
- **Criteria:** `C1=N`, `C2=Y`, `C3=N`; `P`, pretraining-infrastructure
  precursor, not deployed sleep.
- **Capacity/governance nonclaim:** periodic local reset is a training
  parallelization device, not semantic forgetting or a user-memory lifecycle.
  No deployment publication, deletion, lineage, or rollback is shown.

### G12. ReasoningBank

- **Primary/status:** Ouyang et al.,
  [*ReasoningBank*](https://arxiv.org/abs/2509.25140), first public
  2025-09-29; ICLR 2026 Poster; Google Cloud + UIUC/Yale.
  [Code](https://github.com/google-research/reasoning-bank) is Apache-2.0.
- **Operator:** after a task, a judge and extractor produce at most three
  structured successful/failed reasoning memories; JSON entries are appended,
  embedded, retrieved top-1 on a later task, and expanded through
  memory-aware test-time scaling (`MaTTS`).
- **Anchor:** WebArena with Gemini Flash improves `40.5 -> 48.8 -> 51.8`
  for base/ReasoningBank/MaTTS while average steps fall `9.7 -> 8.3 -> 7.9`.
  Gemini Pro improves `46.7 -> 53.9 -> 56.3`; Claude `41.7 -> 46.3 -> 48.8`.
  Judge accuracy is `72.7%`; tokens per task rise `50,847 -> 53,055` (`+4.3%`).
- **Criteria:** `C1=Y`, `C2=Y`, `C3=Y`; `B`, strict PASS at the
  query/task-completion boundary. Extraction and append occur after the visible
  task response and the resulting entry is read on a later task. This does not
  establish an independently scheduled asynchronous or idle-time service:
  cadence, queue ownership, and a scheduler outside the episode loop are
  separate infrastructure nonclaims.
- **Capacity/governance nonclaim:** storage is append-only. Increasing retrieval
  from top-1 to top-2/3/4 reduces one reported score `49.7 -> 46.0 -> 45.5 ->
  44.4`, direct evidence of retrieval competition. There is no prune, merge,
  delete, rollback, claim provenance, or privacy lifecycle.

### G13. Online Neural Space Time Memory

- **Identity:** Elmieh et al.,
  [*Online Neural Space Time Memory for Dynamic Novel View
  Synthesis*](https://arxiv.org/abs/2607.15271), first public 2026-07-16;
  University of Washington and Google, with ten of eleven authors carrying the
  Google affiliation.
- **Operator:** predeployment outer-loop training alternates an isolated memory
  readout loss with synthesis supervision. At runtime a gradient-based
  fast-weight memorization step runs periodically, while every video frame
  applies the current memory with cross-view attention. A running average over
  cached historical fast-weight states regularizes drift.
- **Result anchors:** the audited paper reports approximately `58 ms` for one
  update-plus-apply step at 256×256 on H100 versus `27 ms` for apply alone,
  and demonstrates a 29-apply-to-one-update schedule. On its 390-scene memory
  stress test, the reported memory-view PSNR is `21.46/20.86/20.71` at
  timesteps 4/30/60.
- **Criteria:** current video observations update fast weights that later
  frames reuse, so `C1=Y,C3=Y`; the transform is a live periodic online TTT
  step, not a query-independent snapshot job, so `C2=N` and the phase is `W`.
- **Capacity and nonclaims:** the active fast-weight shape and one running
  average are constant, which is useful update/apply-frequency evidence. The
  experiment covers minute-scale video streams, not cross-session user memory;
  it does not measure finite-capacity overwrite, privacy deletion, rollback,
  offload/migration, task drift, or a sleep scheduler. The official arXiv
  record and project page expose the paper and result gallery; no public code
  release was linked in the audited record.

This is a strong **wake-plane frequency-decoupling analogue**. It does not
change the conclusion about direct Google sleep research: Language Models Need
Sleep remains the explicit benchmark-lifecycle parametric pass, while NSTM is
online TTT.

### G14. Conditional memorization-capacity measurement

- **Primary/status:** Morris et al.,
  [*How much do language models memorize?*](https://arxiv.org/abs/2505.24832),
  first public 2025-05-30; the author affiliations span Meta FAIR, Google
  DeepMind, Cornell, and NVIDIA.
- **Operator:** this is not a deployed memory operator. The study separates
  unintended memorization from generalization and uses synthetic uniform
  sequences to eliminate generalization when estimating retained information.
- **Anchor:** hundreds of GPT-style transformers from roughly 500K to 1.5B
  parameters show a bfloat16 plateau around `3.5–3.6 bits/parameter`; the
  reported fp32 mean is about `3.83`. Capacity saturation is also connected to
  grokking/double descent and membership-inference scaling in the declared
  experiments.
- **Criteria:** `C1=N`, `C2=N/A`, `C3=N`; `P`, strict FAIL.
- **Capacity boundary:** the estimate is a measured lower bound in specified
  data, architecture, optimization, and precision settings. It is not a
  theoretical maximum, a natural-language fact count, or a sequential
  sleep-update capacity. Auxiliary state and pretrained priors must be counted
  in any transfer to the present program.

The value is retained as a controlled opaque-payload benchmark anchor. It may
not be pooled as if it measured the same object as the roughly two
knowledge-bits/parameter factual-tuple result of Allen-Zhu and Li.

### G15. Negative conclusion for the Google line

The sequence-model line establishes fast neural state, forgetting gates,
fixed-versus-growing memory trade-offs, and multi-frequency learning. Only the
Google/Cornell sleep paper demonstrates a strict parametric `C1+C2+C3`
transition in its benchmark lifecycle. It does **not** demonstrate a
production sleep cluster or unlimited lifetime adaptation.

## 5. Meta and FAIR lineage

### M1. Gradient Episodic Memory

- **Primary/status:** Lopez-Paz & Ranzato,
  [*Gradient Episodic Memory*](https://arxiv.org/abs/1706.08840), first public
  2017-06-26; NeurIPS 2017; all FAIR.
- **Operator:** retain examples per prior task; at every new update, solve a
  quadratic program so the current gradient does not increase stored-task loss.
- **Anchor:** on CIFAR-100, accuracy rises with episodic memory size:
  `0.487/0.579/0.633/0.654` at `200/1,280/2,560/5,120` examples. With five
  epochs per rotated-MNIST task, GEM reports `ACC/BWT 0.89/-0.02` versus EWC
  `0.61/-0.11`.
- **Criteria:** `C1=Y`, `C2=N`, `C3=Y`; `W`, PARTIAL but not sleep.
- **Capacity/governance nonclaim:** the episodic buffer is finite, but quality
  is explicitly memory-size dependent. Per-task descriptors and lab task
  boundaries are assumed; no background scheduler, semantic deletion,
  provenance, or rollback is supplied.

### M2. Memory Aware Synapses

- **Primary/status:** Aljundi et al.,
  [*Memory Aware Synapses*](https://arxiv.org/abs/1711.09601), first public
  2017-11-27; ECCV 2018; KU Leuven-led with FAIR coauthors.
- **Operator:** accumulate unsupervised output-sensitivity importance from
  observed inputs, then penalize changing important weights during a new task.
- **Anchor:** the paper argues explicitly that fixed capacity and unlimited
  information require selective erasure. Reported object-recognition forgetting
  is about `0.49%`; adapting importance to a chosen user subset preserves that
  subset while leaving more capacity for later tasks (`84%` versus `69%` in the
  cited adaptation example).
- **Criteria:** `C1=Y`, `C2=N`, `C3=Y`; `W/B`, PARTIAL but not sleep.
- **Capacity/governance nonclaim:** MAS offers a plasticity allocation signal,
  not free-space accounting or causal erasure. It has no item-to-weight
  provenance, exact delete, rollback, or deployment scheduler.

### M3. Large Memory Layers with Product Keys

- **Primary/status:** Lample et al.,
  [NeurIPS primary page](https://papers.nips.cc/paper_files/paper/2019/hash/9d8df73a3cfbf3c5b47bc9b50f214aff-Abstract.html),
  first public arXiv 2019-07-10; NeurIPS 2019; FAIR.
- **Operator:** train a large static key/value layer with exact product-key
  lookup; only a small selected set of values receives each training update.
- **Anchor:** `512^2 = 262K` slots, four heads, top-32 reads; scaling from 16K
  to 1M slots reduces test perplexity `22.8 -> 18.0` without changing reported
  inference time. A 12-layer model with memory outperforms a 24-layer dense
  model while running nearly twice as fast.
- **Criteria:** `C1=N`, `C2=N/A`, `C3=N`; `P`, strict FAIL.
- **Capacity/governance nonclaim:** this is an efficient parametric destination,
  not a deployed memory writer. Usage imbalance is visible—without query batch
  normalization only `25.8%` of 1M values are used. No lifetime allocation,
  compaction, deletion, provenance, or rollback exists.

### M4. Memory Layers at Scale

- **Primary/status:** Berges et al.,
  [*Memory Layers at Scale*](https://openreview.net/forum?id=ATqGm1WyDj),
  arXiv first public 2024-12-12; ICML 2025 Poster; Meta FAIR.
  [Code](https://github.com/facebookresearch/memory) is public.
- **Operator:** pretrain shared sparse key/value pools as substitutes for
  selected feed-forward layers.
- **Anchor:** up to `64M` keys / `128B` memory parameters, trained on `1T`
  tokens; a 1.3B base with 128B memory approaches Llama-2 7B factual
  performance despite the latter using roughly `10x` FLOPs and `2x` tokens.
- **Criteria:** `C1=N`, `C2=N/A`, `C3=N`; `P`, strict FAIL.
- **Capacity/governance nonclaim:** the work is a scaling law for a static
  pretrained destination. It does not show post-deployment writes, sleep
  scheduling, slot reclamation, per-user isolation, deletion, provenance, or
  rollback.

### M5. Continual Learning via Sparse Memory Finetuning

- **Primary/status:** Lin et al.,
  [OpenReview](https://openreview.net/forum?id=LGo7U1m24L), public
  2025-09-19; arXiv v1 2025-10-16; the public status is ICLR 2026
  **desk-rejected submission**, not an accepted paper; Meta FAIR + Berkeley.
- **Operator:** rank memory rows by task access relative to a background corpus
  using TF-IDF, then update only the most task-specific rows.
- **Anchor:** for matched new-knowledge acquisition, NaturalQuestions F1 drops
  `89%` after full fine-tuning, `71%` with LoRA, and `11%` with sparse memory
  fine-tuning. The experiments use a 1M-row pool; typical fact “core sets” span
  roughly 100–500 indices.
- **Criteria:** `C1=N`, `C2=Y`, `C3=Y` only in the benchmark's
  predeployment/task-boundary sense; `B/P` precursor, not strict deployed STC.
- **Capacity/governance nonclaim:** locality reduces interference but the 1M
  pool is a finite substrate. There is no deployed slot-exhaustion test,
  consolidation cadence, exact deletion, item provenance, or rollback.

#### M5a. Independent SMF reimplementations and scope

Two 2026 preprints test the substrate outside the original Meta/Berkeley
implementation:

- Goyal et al.,
  [*Improving Sparse Memory Finetuning*](https://arxiv.org/abs/2604.05248),
  first public 2026-04-06, retrofit Qwen-2.5-0.5B with additive/replacement
  sparse memory modules and introduce KL-divergence row selection as an
  alternative to TF-IDF.
- Gupta et al.,
  [*Sparse Memory Finetuning as a Low-Forgetting Alternative to LoRA and Full
  Finetuning*](https://arxiv.org/abs/2605.03229), first public 2026-05-04,
  compare the same Qwen scale across seeds on MedMCQA, WikiText, and TriviaQA.
  Their abstract reports `+2.5 pp` MedMCQA for the KL sparse variant while the
  two forgetting probes remain within roughly one point of the base; LoRA and
  full fine-tuning gain more target utility but drift more. The body reports
  target gains of about `+4.6 pp` and `+5.4 pp` for those two comparators.

These are independent open reimplementations/extensions, not exact replications
of the original model, data stream, optimizer, or headline `89/71/11%`
comparison. They strengthen the narrow claim that sparse row-local updates can
occupy a different stability--plasticity frontier, while also showing that
row-selection rule and architecture alter which forgetting probe is preserved.
They do not establish off-path scheduling, lifetime pool exhaustion, deletion,
rollback, or deployed sleep. They must receive their own source/version records
rather than being counted as two independent confirmations of the original
numerical claim.

### M6. PAHF

- **Primary/status:** Liang et al.,
  [*Learning Personalized Agents from Human Feedback*](https://arxiv.org/abs/2602.16173),
  first public 2026-02-18; preprint; work done at Meta Superintelligence Labs.
- **Operator:** retrieve explicit per-user preference notes; ask pre-action
  clarification when needed; immediately parse post-action correction and
  add/update the persistent memory.
- **Anchor:** after preference drift, embodied success is `32.3%` for no
  memory, `67.9%` post-action-only, and `70.5%` for PAHF; shopping reaches
  `70.3%` for PAHF versus `27.0%` no-memory.
- **Criteria:** `C1=Y`, `C2=N`, `C3=Y`; `W`, PARTIAL. The memory mutation is
  part of the live feedback loop, not deferred sleep.
- **Capacity/governance nonclaim:** the paper deliberately uses a simple
  dense-retrieval backend and delegates scalable memory architecture to future
  work. No cap, decay, delete/RTBF, source lineage, rollback, or privacy
  contract is established.

### M7. Joint parametric-capacity boundary

The Morris et al. study audited in `G14` also belongs to the public Meta/FAIR
line. It is important negative structure for sleep scaling: a fixed GPT-style
parameter state saturates under independent synthetic information, and the
measured plateau depends on precision and protocol. It does not provide
deferred updates, per-user state, source-to-weight lineage, deletion, rollback,
or an external-memory comparison. Counting the same multi-affiliation paper in
both company lineages does not make it independent evidence.

### M8. Scoped negative result

As of 2026-07-25, this targeted primary-source audit found **no public Meta
paper** that demonstrates all of:

1. deployed accumulated wake input;
2. transformation off the current user-visible path or on an async snapshot;
3. durable transformed state consumed by a later wake.

GEM and MAS are online continual-learning mechanisms; PKM and Memory Layers are
static substrates; Sparse Memory Finetuning is task-boundary adaptation; PAHF
is foreground write-through memory; the Morris et al. study is static
capacity measurement. This is evidence about the audited public corpus, not a
universal claim about Meta's internal systems.

## 6. Microsoft lineage

### MS1. LongMem

- **Primary/status:** Wang et al.,
  [*Augmenting Language Models with Long-Term Memory*](https://arxiv.org/abs/2306.07174),
  first public 2023-06-12; NeurIPS 2023; UCSB + Microsoft Research.
- **Operator:** a frozen encoder writes past hidden-state key/value pairs to a
  memory bank; a trainable residual SideNet retrieves and reads them.
- **Anchor:** the evaluated bank holds `65,536` entries/tokens and uses FIFO;
  the paper reports long-form memory to 65K tokens.
- **Criteria:** `C1=Y`, `C2=N`, `C3=P`; `W`, strict FAIL. Persistence is for
  later positions in the evaluated context, not a scheduled later user wake.
- **Capacity/governance nonclaim:** FIFO bounds the cache but does not perform
  semantic consolidation. There is no cross-session publication, deletion
  lineage, rollback, or privacy model.

### MS2. GenerativeAdapter

- **Primary/status:** Chen et al.,
  [*GenerativeAdapter*](https://arxiv.org/abs/2411.05877), earliest attributable
  public record 2024-10-10 at AFM 2024 Oral; ICLR 2025 Poster; UW + Microsoft.
- **Operator:** a pretrained generator converts past-context hidden states into
  additive adapters for a frozen base LM in one forward pass.
- **Anchor:** evaluated on Mistral-7B and Llama-2-7B for document knowledge,
  demonstrations, and personalization; the claimed benefit is avoiding
  gradient-based adaptation rather than creating an offline lifecycle.
- **Criteria:** `C1=Y`, `C2=N`, `C3=P`; `Q/W`, strict FAIL.
- **Capacity/governance nonclaim:** generated fast adapters are not given a
  lifetime store budget, merge/evict policy, source provenance, deletion,
  validation, or rollback protocol.

### MS3. ACON

- **Primary/status:** Kang et al.,
  [*ACON*](https://arxiv.org/abs/2510.00615), first public on OpenReview
  2025-09-18; arXiv 2025-10-01; ICLR 2026 submission withdrawn; ICML 2026;
  KAIST/Microsoft/Cambridge, with first-author work during a Microsoft
  internship. [Code](https://github.com/microsoft/acon) is MIT.
- **Operator:** offline natural-language optimization learns compression
  guidelines and distils a smaller compressor; at runtime, observations/history
  are compressed when a threshold is crossed.
- **Anchor:** on AppWorld the paper reports `26–54%` lower peak tokens and up
  to `46%` improvement for smaller models; these combine offline controller
  training with wake-path compression.
- **Criteria:** offline learner `C1=N,C2=Y,C3=Y` only as a predeployment
  artifact; runtime operator `C1=Y,C2=N,C3=P`. Overall `P+Q`, strict FAIL.
- **Capacity/governance nonclaim:** a shorter current context is not a bounded
  lifetime memory. No raw canonical log, durable cross-session destination,
  semantic deletion, provenance, or rollback is demonstrated.

### MS4. LEGOMem

- **Primary/status:** Han et al.,
  [*LEGOMem*](https://arxiv.org/abs/2510.04851), first public 2025-10-06; AAMAS
  2026; all Microsoft.
- **Operator:** mine successful training trajectories into full-task and
  subtask procedural memories, then compose those fixed modules at deployment.
- **Anchor:** `93` successful full-task memories and `250` subtask memories are
  constructed from `148` training tasks.
- **Criteria:** `C1=N`, `C2=Y`, `C3=Y` in a predeployment benchmark;
  `P`, precursor rather than recurring deployed sleep.
- **Capacity/governance nonclaim:** the evaluated bank is fixed. Failures,
  continual growth, conflict resolution, deletion, source lineage, and rollback
  are not evaluated.

### MS5. H-EPM

- **Primary/status:** Li et al.,
  [*Experience-Evolving Multi-Turn Tool-Use Agent with Hybrid
  Episodic–Procedural Memory*](https://arxiv.org/abs/2512.07287), first public
  2025-12-08; ICML 2026 / PMLR 306; HKUST and Microsoft Research Asia, with
  first-author work during a Microsoft internship.
  [Code](https://github.com/LISijia-dev/H-EPM) is public.
- **External operator:** successful tool-use trajectories become a directed
  tool graph. Nodes are tools; an edge stores both a normalized
  frequency/efficiency weight and zero or more LLM-produced state summaries.
  At inference, the current tool selects adjacent candidates; the agent either
  compares current-state similarity against episodic edge annotations or uses
  procedural edge weights.
- **Boundary evidence:** the main inference experiments build the graph from
  a training split. The ACEBench appendix instead says that memory is “updated
  online using test-time data” and that an LLM judge identifies successful
  trajectories in real time. It provides no ordering algorithm showing that
  the response/task is first finalized, that graph insertion starts only
  afterward, or that insertion cannot affect the same task. The released
  repository contains no ACEBench runner. Its
  [standalone builder](https://github.com/LISijia-dev/H-EPM/blob/4ad8ad335fe63d365f4baf97df77e19055c79f1c/tau-bench/tool_graph_builder.py#L62-L141)
  transforms already reward-labelled trajectory files, while the closest
  [ToolSandbox online path](https://github.com/LISijia-dev/H-EPM/blob/4ad8ad335fe63d365f4baf97df77e19055c79f1c/ToolSandbox/tool_sandbox/openai_api_agent.py#L380-L404)
  updates after tool-call segments inside execution. Conservative criteria are
  therefore `C1=Y,C2=P,C3=P`; `W/P`, PARTIAL and **not** a strict PASS.
  Within-task state summarisation and retrieval are also `W`.
- **Parametric operator:** during a separate GRPO experiment, the same graph
  guides exploration and is updated with successful rollouts while LoRA/model
  policy weights are trained. Training uses one epoch, batch size `8`, eight
  rollouts per task, terminal success reward, and KL coefficient `.001`;
  memory-guidance content is masked from the policy loss. This is a
  memory-guided post-training loop, not evidence of a deployed off-path weight
  trainer.
- **Anchors:**
  - Inference on \(\tau^2\)-Bench reaches `0.512` with GPT-4.1 versus `0.358`
    for the base model; the largest tabled relative gain is `+53.7%` for
    Qwen3-8B (`0.201 -> 0.309`).
  - ACEBench online-memory end-to-end accuracy is `0.533` versus `0.400`
    for GPT-4.1-mini, `0.567` versus `0.503` for GPT-4.1, and `0.503` versus
    `0.484` for GPT-4o.
  - GRPO-guided training on \(\tau^2\)-Bench reaches `0.223` versus base
    `0.158` and ordinary GRPO `0.178`.
  - Initial graph construction uses three rollouts per training task and keeps
    the shortest successful trajectory; about `2.2K` \(\tau^2\)-Bench tasks
    take roughly `15 h` on one A100. One RL epoch takes about `20 h` on eight
    H100s.
- **Capacity/governance nonclaim:** each edge may accumulate multiple state
  summaries, and no hard cap, merge, eviction, expiry, physical delete,
  provenance/version protocol, or rollback is evaluated. Keeping only
  successful trajectories is an admission filter, not a lifetime capacity
  mechanism; it can also preserve reward/judge error. The implementation
  timing finding is pinned to repository commit
  `4ad8ad335fe63d365f4baf97df77e19055c79f1c`.

### MS6. MEMORA

- **Primary/status:** Xia et al.,
  [*MEMORA*](https://arxiv.org/abs/2602.03315), first public 2026-02-03,
  v2 2026-07-02; ICML 2026 / PMLR 306; all Microsoft/M365 Research.
  Public [repository](https://github.com/microsoft/Memora) is MIT.
- **Operator:** map each segment to a primary abstraction and concrete value;
  retrieve related entries; use an LLM to merge or create; add cue anchors;
  retrieve semantically or with a REFINE/EXPAND/STOP policy optionally trained
  by GRPO.
- **Anchors:**
  - LoCoMo LLM-judge overall: RAG `0.633`, Mem0 `0.653`, MEMORA policy `0.863`.
  - LongMemEval average: `87.4%`.
  - Construction: `1322.0 s` per conversation; offset optimization `739.9 s`
    (`45%` faster) while score changes `0.863 -> 0.860`.
  - No updates scores `0.795`; threshold `0.8` performs 88 updates (`21.0%`)
    and scores `0.801`; threshold `0.6` causes `299` updates (`68.2%`) and
    scores `0.799`.
  - A conversation yields on average 432 candidates: `344` new entries and
    `88` updates. Update ratio remains `16.5–22.2%` across store-size buckets.
- **Criteria:** `C1=Y`, `C2=N`, `C3=Y`; strict FAIL, adjacent PARTIAL. The
  construction is persistent, but no owned post-response/async scheduler is
  demonstrated.
- **Capacity boundary:** the paper interprets the roughly stable measured
  update fraction across its store-size buckets as evidence that merge/update
  work remains approximately linear over that observed range. That is an
  author inference from a finite range, not an asymptotic scaling law. It also
  does **not** imply a bounded store: the measured average still contains
  `344` new entries per conversation, direct counterevidence to “merge solves
  capacity.”
- **Governance nonclaim:** no hard cap, eviction, user deletion, rollback,
  source-claim provenance, or privacy lifecycle. Policy retrieval also costs
  `5.697 s` mean end-to-end versus `1.062 s` for semantic retrieval.

### MS7. PlugMem

- **Primary/status:** Yang et al.,
  [*PlugMem: A Task-Agnostic Plugin Memory Module for LLM
  Agents*](https://arxiv.org/abs/2603.03296), first public 2026-02-06; ICML
  2026 according to the
  [Microsoft Research record](https://www.microsoft.com/en-us/research/publication/plugmem-a-task-agnostic-plugin-memory-module-for-llm-agents/);
  UIUC, Tsinghua, and Microsoft Research.
  [Code and data](https://github.com/TIMAN-group/PlugMem) are public.
- **Operator:** heterogeneous dialogue, document, and action traces are
  standardised into episodic tuples. LLM extraction then creates
  provenance-linked proposition/concept and prescription/intent graphs.
  Retrieval alternates low-level knowledge candidates with high-level routing
  nodes, and a reasoning model compresses the result for the base agent.
- **Boundary evidence:** LongMemEval and HotpotQA primarily use a prebuilt
  index. In WebArena's online split, the system records the current action
  trace but inserts and abstracts the episodic sequence only **at the end of
  each task**. In public repository commit
  `3b2ce75257d40bca8fac3f78e54e22ea41d92529`,
  the
  [pinned WebArena path](https://github.com/TIMAN-group/PlugMem/blob/3b2ce75257d40bca8fac3f78e54e22ea41d92529/src/eval/webarena/plugmem_agent.py#L328-L447)
  exits
  `while not env.done()`, freezes `final_status`, calls `memory.close()` to
  perform semantic/procedural extraction, inserts the result, and returns the
  already computed status. Insert/close failures are caught rather than used
  to revise that task's result. Subsequent online tasks and a held-out
  retrieval-only split read the graph. For that implemented benchmark path,
  `C1=Y,C2=Y,C3=Y`: `EXPERIMENTAL PASS` at a task boundary, not an autonomous
  periodic scheduler. It is synchronous post-task tail work, not a queued or
  background job; stepwise append and retrieval remain on the wake path.
- **Anchors:**
  - LongMemEval-S accuracy is `75.1` with `362.58` average injected memory
    tokens, versus Zep `71.2`/`1600` and LiCoMemory `73.0`/`5914.85` in the
    paper's table; some baseline values come from prior work or a subset, so
    this is not a fully matched systems comparison.
  - HotpotQA is `61.4` EM / `74.1` F1 with `81.6` average memory tokens.
  - WebArena offline success is `58.4%` Shopping, `55.2%` GitLab, and `21.6%`
    Multi-site, using an online/offline template split and injected human
    demonstrations.
  - A controlled HotpotQA merge at threshold `.7` reduces active semantic
    nodes `3413 -> 3242` (`-5.0%`), attachments `23230 -> 20604` (`-11.3%`),
    and mean tag-induced candidate fan-out `38.36 -> 31.28` (`-18.5%`);
    EM/F1 changes `61.00/74.39 -> 62.00/74.65`.
- **Capacity/governance qualification:** the appendix implements
  LLM-synthesised merge plus **soft** deactivation and describes create,
  retrieve, update, and delete. The main benchmarks primarily test
  create/retrieve, and the merge experiment is a one-shot finite-graph study.
  It does not demonstrate a total byte/object cap, repeated-lifetime
  equilibrium, physical erasure, user-scoped delete fan-out, versioned
  provenance, or rollback. Its own utility–cost sweeps show that extra retrieved
  memory can saturate and then become counterproductive.

### MS8. MemMA

- **Primary/status:** Lin et al.,
  [*MemMA: Coordinating the Memory Cycle through Multi-Agent Reasoning and
  In-Situ Self-Evolution*](https://arxiv.org/abs/2603.18718), first public
  2026-03-19; arXiv preprint; Penn State, Amazon, and Microsoft affiliations.
  The [Microsoft Research record](https://www.microsoft.com/en-us/research/publication/memma-coordinating-the-memory-cycle-through-multi-agent-reasoning-and-in-situ-self-evolution/)
  and [public code](https://github.com/ventr1c/memma) were both available at
  audit time.
- **Forward operator:** a Meta-Thinker supplies construction guidance to a
  Memory Manager as dialogue chunks arrive and answerability guidance to a
  Query Reasoner during iterative retrieval. The persistent backend can be a
  single-agent store, A-Mem, or LightMem.
- **Backward operator:** after each dialogue session, the provisional store is
  tested with \(J=5\) synthetic probes spanning single-session,
  cross-session, and temporal questions. Failed probes produce
  evidence-grounded repair facts; semantic consolidation chooses
  `SKIP`, `MERGE`, or `INSERT`, and the repaired state is written back before
  later sessions and final QA use it.
- **Boundary evidence:** in repository commit
  `51d463ad24f8b4f63633d12b32df7c64ed285fff`, the
  [released LightMem runner](https://github.com/ventr1c/memma/blob/51d463ad24f8b4f63633d12b32df7c64ed285fff/scripts/run_memma_self_refine_lightmem.py#L4108-L4289)
  first processes every turn in a session, then queues and runs that session's
  self-refinement. A session with no extracted entry can be deferred and
  replayed after a later extraction; after all sessions, a separate LightMem
  offline update runs. The
  [refinement implementation](https://github.com/ventr1c/memma/blob/51d463ad24f8b4f63633d12b32df7c64ed285fff/scripts/run_memma_self_refine_lightmem.py#L2009-L2326)
  freezes the pre-repair entry set for evaluation, applies repair actions, and
  then retests the same probes. Thus the evaluated session lifecycle has
  `C1=Y,C2=Y,C3=Y`: `B`, an **EXPERIMENTAL PASS at a session boundary**.
  It is synchronous benchmark construction, not a queued production daemon,
  and turn-level construction remains `W`.
- **Anchors:** evaluation uses only LoCoMo's first conversation, `conv-26`
  (`19` sessions, `419` turns, `152` non-adversarial final QA pairs), top-30
  retrieval, \(H=3\) retrieval-refinement steps, and one answer/judge setup
  anchored by GPT-4o-mini. With GPT-4o-mini, MemMA+LightMem reports
  `49.40` F1 / `38.28` BLEU-1 / `81.58` accuracy versus LightMem
  `44.58 / 36.66 / 75.66`. Backend accuracy changes are
  `52.60 -> 84.87` for Single-Agent, `52.63 -> 78.29` for A-Mem, and
  `75.66 -> 81.58` for LightMem. Removing iterative retrieval lowers
  Single-Agent MemMA accuracy `84.87 -> 70.39`; removing self-evolution lowers
  it to `73.68`.
- **Reproducibility qualification:** the released quick start obtains the
  backward-path probes from committed
  `data/memory_rl_train_locomo_conv_26.parquet`, not from a documented
  generator. The
  [README at the pinned commit](https://github.com/ventr1c/memma/blob/51d463ad24f8b4f63633d12b32df7c64ed285fff/README.md#L128-L161)
  states that the generation pipeline will be documented later. The file
  inspected here is `905,574` bytes, has SHA-256
  `0bdf4164657da33b5c6bd786483b4bac507fb712f90213abba529a86a3485c8e`,
  and contains `76` session rows. The runner also supports real-time probe
  generation, but that does not repair the missing provenance for the
  paper-aligned quick start. No final-QA leakage was identified from the
  inspected rows, yet a complete leakage and selection audit is impossible
  without the generator and its inputs.
- **Capacity/governance nonclaim:** retrieval exposes a bounded top-\(k\) view,
  not a bounded durable store. Construction supports delete actions, but the
  repair path only skips, merges, or inserts, and the paper reports no hard
  byte/object cap, lifetime growth curve, physical erasure, versioned
  provenance, rollback, privacy lifecycle, scheduler, seeds, or confidence
  intervals. “Compact” is therefore qualitative.

### MS9. MEMENTO

- **Primary/status:** Kontonis et al.,
  [*MEMENTO: Teaching LLMs to Manage Their Own
  Context*](https://arxiv.org/abs/2604.09852), announced in an
  [official Microsoft Research article](https://www.microsoft.com/en-us/research/articles/memento-teaching-llms-to-manage-their-own-context/)
  on 2026-04-08 and first posted to arXiv on 2026-04-10; all authors list
  Microsoft affiliations. The
  [code](https://github.com/microsoft/memento) was audited at commit
  `d8c10e66ff313844b4b5f063dd4c075d3ea56aab`; the
  [OpenMementos dataset](https://huggingface.co/datasets/microsoft/OpenMementos)
  reports exactly `228,557` records.
- **Predeployment data and training:** OpenThoughts-v3 reasoning traces are
  split into sentence/code/equation atoms; a GPT-5.x model scores every
  boundary from zero to three; dynamic programming chooses coherent,
  size-balanced blocks; another GPT-5.x call compresses each block; and a
  separate GPT-5.x judge scores six fidelity dimensions. A score below eight
  triggers up to two feedback rounds, changing reported acceptance from `28%`
  after one pass to `92%`. The final pool is `54%` math, `19%` code, and `27%`
  science and compresses roughly `10.9K` reasoning tokens to `1.85K` memento
  tokens per trace. Reasoning checkpoints use two SFT stages—ordinary full
  attention and then block-masked attention—on `31K` samples, `32K` sequence
  length, five epochs per stage, learning rate \(8\times10^{-5}\), batch 512,
  and 32 B200 GPUs. A non-reasoning Qwen2.5-7B model first receives ordinary
  reasoning SFT. The Qwen3-8B RL extension uses `29,670` DolciMath prompts,
  CISPO with rule-based SymPy reward, group size eight, batch 240, learning
  rate \(10^{-6}\), clipping `.2`, KL `.001`, 24 GPUs, and selects step 350.
- **Runtime operator:** during one response, the model emits a reasoning block
  and a textual memento. Once the memento closes, the vLLM extension
  [compacts the request](https://github.com/microsoft/memento/blob/d8c10e66ff313844b4b5f063dd4c075d3ea56aab/vllm/vllm/v1/core/block_masking/processor.py#L45-L162);
  its KV manager
  [copies retained entries and frees trailing blocks](https://github.com/microsoft/memento/blob/d8c10e66ff313844b4b5f063dd4c075d3ea56aab/vllm/vllm/v1/core/kv_cache_manager.py#L430-L456).
  Later tokens in the **same generation** read prior memento text and KV.
  Runtime therefore has `C1=Y,C2=N,C3=N`: `Q/W`, strict FAIL. SFT/RL has
  `C1=N,C2=Y,C3=Y`: a predeployment learned compactor, not consolidation of
  deployed wake experience.
- **Accuracy and systems anchors:** for Qwen3-8B, MEMENTO versus the matched
  OpenThoughts control changes AIME'26 `64.7 -> 57.3`, competition math
  `49.2 -> 45.1`, MATH-500 `89.7 -> 90.1`, GPQA-D `57.8 -> 55.8`, and
  LiveCodeBench v6 `70.0 -> 66.5`, while peak KV ratios range from `.32` to
  `.47`. RL raises the five MEMENTO scores to `64.9/49.4/91.0/62.9/68.8`,
  but also raises competition-math peak KV from `1.08` to `1.48 GB`.
  Uniform-attention models obtain about `2–3x` lower peak KV and up to `3.5x`
  lower KV area-under-curve; hybrid sliding-window OLMo-3 obtains only
  `.85–.93x` peak ratios and its AIME'26 KV AUC is `1.02x`, showing that peak
  reduction does not imply lower total memory-time. On one B200 with 240
  Qwen3-8B requests and a 32K maximum, the patched engine reports
  `4,290` versus `2,447` token/s and `693` versus `1,096 s` batch completion.
- **Complete-state finding:** a restart ablation keeps identical memento text
  but recomputes its KV without the evicted block, lowering AIME24 pass@1
  `66.1 -> 50.8` (`-15.3 pp`). Random five-digit probes recover masked-block
  information from downstream memento KV at `26.7%` for Qwen3-8B and `23.0%`
  for Qwen3-32B versus `10%` chance; the signal is stronger in deeper layers
  and persists seven hops in a controlled toy model. Thus a memento is
  **text plus block-conditioned KV**, not a portable text record. Physical
  deletion of the original block is neither semantic erasure nor proof that a
  restart, migration, audit, or privacy deletion preserves behavior.
- **Capacity/governance nonclaim:** the work bounds one request's active KV
  footprint, not a durable cross-session store. It reports no tenant-scoped
  provenance, deletion contract, persistence, later-wake reuse, rollback, or
  lifetime capacity curve. The hidden KV channel must nevertheless count in
  answer-reachable state, privacy testing, and total memory-time accounting.

### MS10. M★

- **Primary/status:** Pan et al.,
  [*M★: Every Task Deserves Its Own Memory
  Harness*](https://arxiv.org/abs/2604.11811), first public 2026-04-10;
  preprint/under review; City University of Hong Kong and Microsoft. A
  NeurIPS-formatted manuscript is not acceptance evidence; the
  [Microsoft Research record](https://www.microsoft.com/en-us/research/publication/mstar-every-task-deserves-its-own-memory-harness/)
  also labels it a preprint.
  [Code](https://github.com/wbopan/mstar) is public.
- **Operator:** represent a memory harness as executable Python whose
  `Schema`, write/read `Logic`, and agent `Instruction` jointly determine what
  is stored and how it is used. A coding-agent reflector consumes failed and
  successful validation traces, patches a parent program, runs compile/runtime
  checks, and adds valid children to a score-biased population.
- **Training basis:** each of seven task configurations uses past episodes,
  a static validation set, a rotating validation subset, and a held-out test
  set. Twenty evolution iterations repeatedly re-ingest the episodes and score
  candidate programs. This changes the **memory control program**, not base
  model weights or one user's durable memory.
- **Anchors:** the selected programs lead seven of eight reported columns and
  improve up to `31%` relative to the strongest per-column baseline.
  LoCoMo token F1 is `0.459` versus Mem0 `0.373` and GEPA+Vector Search
  `0.300`; ALFWorld unseen is `0.881`; PRBench Finance is `0.586` versus GEPA
  `0.449`. Evolution costs range roughly `$12–90` per configuration; reported
  20-iteration wall time reaches about `100 h` for ALFWorld. Across the
  paper's three five-seed robustness groups, `14/15` evolved programs exceed
  the strongest fixed baseline.
- **Criteria:** `C1=N,C2=Y,C3=Y`; `P`, predeployment memory-harness
  optimisation. It is evidence that the sleep **operator and destination
  policy may themselves be learned per task**, not evidence of recurring
  sleep over deployed wake experience.
- **Historical boundary:** M★ did not originate memory-architecture evolution.
  [MemEvolve](https://arxiv.org/abs/2512.18746) had already evolved executable
  `Encode/Store/Retrieve/Manage` providers from completed task batches, and
  [ALMA](https://arxiv.org/abs/2602.07755) had already searched executable
  database, update, and retrieval designs. M★'s narrower distinction is joint
  task-specific evolution of `Schema`, memory `Logic`, and agent
  `Instruction`. [MemSkill](https://arxiv.org/abs/2602.02474) is another,
  narrower branch: it PPO-trains a selector over an at-most-30-item
  natural-language skill bank and periodically edits that bank before
  deployment.
- **Capacity/governance nonclaim:** search has a finite 20-iteration evaluation
  budget, but evolved stores have no common hard lifetime cap or deletion
  contract. The paper warns that programs can encode task-specific heuristics
  and carry wrong, private, or biased information; it proposes deployment
  review rather than implementing provenance, revocation, or rollback.
  Repository commit `09a8d34f69588a85fde78b20702d14f056267956`
  keeps program ancestry/checkpoints for optimiser resume, not versioned
  deployed user-memory rollback.

### MS11. WebXSkill

- **Primary/status:** Wang et al.,
  [*WebXSkill*](https://arxiv.org/abs/2604.13318), first public 2026-04-14;
  preprint; Microsoft-heavy collaboration.
- **Operator:** mine reusable executable action subsequences from synthetic
  training trajectories, index them in a URL graph, and deploy in grounded or
  guided mode.
- **Anchor:** reported gains are up to `+9.8` points on WebArena and `+12.9`
  points on WebVoyager.
- **Criteria:** `C1=N`, `C2=Y`, `C3=Y` for a predeployment skill bank; `P`,
  precursor rather than deployed recurring sleep.
- **Capacity/governance nonclaim:** no online skill-bank growth policy,
  deduplication law, delete propagation, provenance/rollback contract, or
  multi-tenant scheduler is evaluated.

### MS12. MemCompiler

- **Primary/status:** Ding et al.,
  [*MemCompiler: Compile, Don't Inject—State-Conditioned Memory for Embodied
  Agents*](https://arxiv.org/abs/2605.07594), first public 2026-05-08, revised
  2026-05-14; arXiv preprint; USTC, HUST, Microsoft Research, Nanjing
  University, and Tsinghua/AIR affiliations. A
  [Microsoft Research record](https://www.microsoft.com/en-us/research/publication/memcompiler-compile-dont-inject-state-conditioned-memory-for-embodied-agents/)
  is public.
- **Runtime operator:** retrieve a read-only task memory once at episode start.
  At each step, a learned Memory Compiler reads the current observation and a
  structured Brief State, then emits executable text guidance plus \(N=16\)
  latent Soft-Mem tokens. `Create`, `Update`, `Delete`, and `Fold` operations
  update only Brief State; the retrieved episode memory does not consolidate.
- **Training basis:** a frontier teacher traverses the training tasks while
  maintaining a task memory of successful and failed trajectories. Successful
  traces become compiler/executor SFT data. The released specification
  fine-tunes a Qwen2.5-VL-7B compiler with LoRA on `q_proj`/`v_proj`
  (\(r=16,\alpha=32,\mathrm{dropout}=.1\)) for five epochs at learning rate
  \(10^{-5}\), while freezing the executor. GRPO then updates only the compiler
  and Soft-Mem projection for `500` steps with binary episode-success reward,
  no shaping or KL penalty, and the executor still frozen.
- **Anchors:** Qwen2.5-14B changes from `46.27 -> 82.16` on AlfWorld and
  `31.11 -> 46.67` on ScienceWorld; Qwen3.5-27B changes
  `61.19 -> 91.45` and `21.11 -> 48.44`. On AlfWorld, the paper reports
  compiler latency `.093 s`, executor latency `.12 s`, and `1003.2` executor
  input tokens, versus `.30 s` and `2481.1` tokens for ahead-of-time
  monolithic injection. The advertised `60%` latency reduction refers to the
  executor; summing reported compiler and executor components gives about
  `.213 s`, approximately `29%` below `.30 s`.
- **Criteria:** `C1=Y,C2=N,C3=N/P`; `Q/W`, strict FAIL. Compilation is part of
  every action's causal path and its output is consumed immediately. This is a
  useful learned **delivery/router** and text-plus-latent destination
  comparator, not a sleep-time memory transform.
- **Capacity/governance nonclaim:** the at-most-ten-item Brief State before
  folding and sixteen soft tokens bound live compiled state only. The read-only
  task-memory bank contains raw successful trajectories plus rationales and has
  no published lifetime cap, eviction, scoring hierarchy, deletion lineage,
  versioning, rollback, or privacy contract.

### MS13. Human-Inspired Memory Architecture

- **Primary/status:** Kerestecioglu et al.,
  [*Human-Inspired Memory Architecture for LLM Agents*](https://arxiv.org/abs/2605.08538),
  first public 2026-05-08; preprint; all Microsoft. No public code or data was
  found in this audit.
- **Operator:** hot events flow into warm episodic memory and permanent
  semantic/knowledge-graph memory. A scheduled sleep pipeline scores,
  deduplicates/clusters, promotes, retains, prunes, and builds gists/graph
  entries. Forgetting, maturation, reconsolidation, and interference are
  separate stages.
- **Configured mechanisms:**
  - default sleep cadence: every six hours, declared domain-tunable;
  - importance weights: recency `.25`, frequency `.25`, surprise `.20`,
    entity `.15`, outcome `.15`;
  - top `20%` promote, middle `60%` retain, bottom `20%` prune;
  - quarantine TTL `15 min`;
  - decay `lambda=.001`, half-life approximately 29 days;
  - fidelity levels from full `L0=100%` through summary `L2=50%`, gist
    `L3=25%`, tombstone `L5=0`;
  - maturation half-life `168 h`, slope `48`; reconsolidation window `60 min`.
- **Anchors:**
  - VSCode workload: `13,127` issues / `120K` events; deduplication reaches
    `97.2%` retention precision with `58%` store reduction, `+21.8 pp` over
    keep-all `75.4%`; the store stabilizes at roughly 300–500 events.
  - LongMemEval-S overall:
    raw `78.4 [74.8,82.0]`; dedup `76.8 [73.0,80.4]`; adaptive-50K `76.2`;
    adaptive-10K `57.0`; aggressive consolidation `48.4`.
  - LongMemEval-M:
    raw `71.2 [67.2,75.0]`; 200K budget `70.1 [66.0,74.2]`; 115K `65.6`;
    50K `49.2`; 25K `38.8`.
- **Criteria:** `C1=Y`, `C2=Y`, `C3=Y`; architecture-level strict PASS.
- **Crucial qualification:** the paper does not report running a recurring
  every-six-hours deployment. Evaluations use quarterly VSCode windows or
  “every N sessions” with `N` not specified. Four mechanisms have ablations;
  maturation is forced to availability one in LongMemEval, graph/maturation are
  absent in VSCode, and reconsolidation is not tested under the contradictions
  needed to exercise it. The complete six-mechanism claim is therefore PARTIAL.
- **Capacity/governance nonclaim:** the budget curve shows that aggressive
  compaction destroys factual recall. Tombstone fidelity is representational
  degradation, not physical erasure. No user-scoped delete fan-out, exact
  rollback, source-claim provenance, or privacy/tenant-isolation contract is
  demonstrated despite enterprise motivation.

### MS14. MAGE

- **Primary/status:** Chen et al.,
  [*Beyond Semantic Organization*](https://arxiv.org/abs/2606.06090), first
  public 2026-06-04; preprint; Microsoft-majority collaboration.
- **Operator:** a hierarchical execution-state tree supports Grow, Compress,
  Maintain, and Revise. The active root-to-current path is the current state;
  erroneous work is isolated on an inactive branch.
- **Anchor:** across MemoryArena, MAGE improves average success by `7.8 pp`
  over long context and reduces token use by `55.1%`. Web-shopping success is
  `0.3933` versus long context `0.3333`, with `1015K` versus `1528K` tokens.
  Removing Compress raises tokens to `2469K`.
- **Criteria:** `C1=Y`, `C2=N`, `C3=P`; `W/B`, strict FAIL. Operations run as
  part of task execution; durable cross-session later-wake reuse is not shown.
- **Capacity/governance nonclaim:** active context is bounded, but the tree can
  grow. Branch revision provides logical error isolation and local rollback, not
  user deletion, physical erasure, full provenance, or lifetime compaction.

### MS15. Retrospective Harness Optimization

- **Primary/status:** Pan et al.,
  [*Retrospective Harness Optimization: Improving LLM Agents via
  Self-Preference over Trajectory
  Rollouts*](https://arxiv.org/abs/2606.05922), first public 2026-06-04;
  arXiv preprint; City University of Hong Kong and Microsoft Research Asia.
  A later manuscript/repository presentation uses the subtitle *Evolving
  Agents in the Dark*; the source registry must preserve title/version
  identity rather than silently overwrite it. The
  [Microsoft Research record](https://www.microsoft.com/en-us/research/publication/retrospective-harness-optimization-improving-llm-agents-via-self-preference-over-trajectory-rollouts/)
  and MIT-licensed [code](https://github.com/wbopan/retro-harness) are public;
  code was audited at commit
  `e5f2d1a8a06ab3523ab42e0042d2fa13d9acb701`.
- **Operator:** score past task trajectories for difficulty; choose a
  difficulty-diverse \(k=10\) coreset with greedy DPP; re-solve each task
  \(G=3\) times; derive self-validation and cross-rollout self-consistency
  diagnoses; sample \(N=3\) candidate harnesses; re-solve the coreset with
  each; and keep the candidate whose mean pairwise self-preference score is
  strictly positive and largest. The learned artifact is a persistent
  directory of Markdown instructions/skills plus executable tool scripts—not
  base-model weights and not only a memory bank.
- **Boundary evidence:** the paper separates trajectory/training pools from
  held-out pools. Completed original trajectories and fixed task outcomes feed
  one retrospective pass; only after selection does the persistent harness
  serve the held-out tasks. In the pinned implementation,
  [`run_round`](https://github.com/wbopan/retro-harness/blob/e5f2d1a8a06ab3523ab42e0042d2fa13d9acb701/src/rho/loop.py#L200-L440)
  performs before-rollouts, candidate construction, after-rollouts, ranking,
  and positive-score acceptance, while
  [`run_evolution`](https://github.com/wbopan/retro-harness/blob/e5f2d1a8a06ab3523ab42e0042d2fa13d9acb701/src/rho/loop.py#L443-L477)
  advances the selected harness only after the round. This yields
  `C1=Y,C2=Y,C3=Y`: `B/P`, an **EXPERIMENTAL PASS**. The repository's separate
  [Codex retrospection workflow](https://github.com/wbopan/retro-harness/blob/e5f2d1a8a06ab3523ab42e0042d2fa13d9acb701/codex/retrospection.py#L440-L644)
  can mine real completed sessions, stage candidates in isolated worktrees,
  retain a backup, and apply a positive winner manually; it still supplies no
  autonomous periodic/pressure scheduler.
- **Anchors:** with Codex GPT-5.5 at high reasoning effort, held-out pass rate
  changes `0.59 -> 0.78` on 100 SWE-Bench Pro tasks, `0.71 -> 0.76` on 59
  Terminal-Bench 2 tasks, and `0.29 -> 0.37` on 100 GAIA-2 tasks. The matched
  Sleep-time Compute baseline reaches `0.64/0.73/0.32`. Removing
  self-consistency yields `0.56/0.75/0.27`; removing self-validation yields
  `0.70/0.73/0.30`; raw-trajectory proposal yields `0.60/0.75/0.29`.
  Best-of-three preference does not identify the hidden test optimum:
  SWE-Bench candidate mean is `.79`, chosen `.78`, and lowest `.73`.
- **Cost/reproducibility qualification:** for SWE-Bench Pro, the optimization
  phase uses `30` before-rollouts, `10` diagnoses, `3` proposals, `30`
  after-rollouts, and `30` ranks: `103` agent invocations before the shared
  100-task test. Difficulty selection additionally uses 100 auxiliary judge
  calls plus one batched embedding call. Reported RHO wall time including
  evaluation is `23.1 h` summed agent time and `3.2 h` elapsed with ten-way
  concurrency. The study reports one optimization round, deterministic splits
  but provider-default sampling, no multi-run confidence intervals, and no
  independent judge; solver, optimizer, and ranker share GPT-5.5. SWE grading
  prevents test-directory access by prompt convention rather than sandbox.
- **Capacity/governance nonclaim:** \(k,G,N\) bound one update's selection and
  search cost, not lifetime state. Harness files, executable scripts,
  content-addressed versions, trajectories, diagnoses, and run logs can grow.
  The Codex variant notes a 32 KB `AGENTS.md` limit and asks the optimizer to
  prune stale skills, but skills/scripts and historical runs remain uncapped.
  Filesystem digests and backups provide useful version/recovery primitives,
  not source-level deletion, authorization, tenant isolation, or semantic
  rollback. The paper explicitly warns that adversarial content in past
  trajectories can become persistent harness behavior; self-preference alone
  is therefore not a production promotion attestation.

### MS16. Negative conclusion for the Microsoft line

HIMA remains the only audited Microsoft paper with an explicitly scheduled
cross-wake external sleep pipeline. PlugMem provides a narrower task-boundary
`EXPERIMENTAL PASS`, MemMA provides a session-boundary `EXPERIMENTAL PASS`,
and RHO provides a completed-trajectory-to-persistent-harness
`EXPERIMENTAL PASS`; none owns a periodic or pressure-triggered production
sleep service. MemMA's paper-aligned quick start also relies on probe data with
an unpublished generation pipeline. RHO consumes completed unlabeled
trajectories and publishes a full instruction/skill/tool harness, but its
same-model self-preference gate is not an independent safety attestation and
its executable artifact has no lifetime cap. H-EPM connects successful
experience, external graph memory, and parametric RL, but its paper leaves the
ACEBench boundary order unresolved and its closest released online writer is
inline. MEMORA, MAGE, MemCompiler, LongMem, and GenerativeAdapter provide
valuable memory destinations, compilers, and update primitives but remain
foreground or within-task. MEMENTO adds learned within-query text/KV
compaction and a direct warning that visible artifacts can omit
answer-relevant and privacy-relevant latent state; it is not durable sleep.
ACON, LEGOMem, M★, and WebXSkill are predeployment/offline-control precedents.
M★ is part of the earlier MemEvolve/ALMA/MemSkill control-program lineage and
is important because it searches memory representation, write/read policy,
and agent instruction jointly; RHO is the later deployed-session-facing
variant that retrospectively revises the whole harness, while H-EPM connects
external experience reuse to a separate parametric RL loop.

The
[STATE-Bench v0.8.1](https://github.com/microsoft/STATE-Bench/releases/tag/v0.8.1)
release (2026-07-16) adds 450 stateful enterprise tasks and an Agent Learning
track, but it is a benchmark rather than evidence of a sleep operator. It does
not prescribe artifact format, capacity, physical erasure, or full accounting
for offline artifact-construction cost.

## 7. Applied system lineages

The detailed implementation anchors are in the companion audit. This section
keeps the company chronology self-contained.

### A1. MemGPT and Letta

#### MemGPT 2023

- **Primary/status:** Packer et al.,
  [*MemGPT*](https://arxiv.org/abs/2310.08560), first public 2023-10-12.
- **Operator:** recall store, core memory, archival memory, context-pressure
  warnings, and recursive summary/paging inside the foreground agent loop.
- **Anchor:** Deep Memory Retrieval improves GPT-4 `32.1 -> 92.5` and GPT-4
  Turbo `35.3 -> 93.4`.
- **Criteria:** `C1=Y`, `C2=N`, `C3=Y`; PARTIAL.
- **Nonclaim:** recall storage can be indefinite; no global capacity,
  transactional rollback, causal deletion, or claim lineage.

#### Letta Sleep-time Compute paper

- **Primary/status:** Lin et al.,
  [*Sleep-time Compute*](https://arxiv.org/abs/2504.13171), first public
  2025-04-17; preprint.
- **Operator:** compile shared context `C` into `C'` before unknown future
  queries, amortizing offline work over later queries.
- **Anchor:** approximately `5x` less test-time compute at matched accuracy,
  up to `+13 pp` on Stateful GSM-Symbolic, `+18 pp` on AIME, and about `2.5x`
  lower average per-query cost.
- **Criteria:** `C1=P`, `C2=Y`, `C3=Y`; PARTIAL because the artifact is a
  one-context compilation study, not a full lifetime memory service.
- **Nonclaim:** no weight update, RL, distillation, capacity policy, deletion,
  source lineage, or rollback.

#### Letta API Sleeptime Agent

- **Primary/version:**
  [`sleeptime_multi_agent_v4.py`](https://github.com/letta-ai/letta/blob/0.16.8/letta/groups/sleeptime_multi_agent_v4.py)
  at audited Letta tag `0.16.8` (commit
  `1131535716e8a31c9a437f8695e25ac98f203a24`), Apache-2.0; lineage begins
  in 2025-04.
- **Operator:** foreground response returns; every configured number of turns
  (default creation path: five), unprocessed message pairs are dispatched via a
  separate background run; the sleeper edits persistent blocks shared with the
  wake agent.
- **Criteria:** `C1=Y`, `C2=Y`, `C3=Y`; PASS.
- **Nonclaim:** blocks have local token limits, but there is no global learned
  allocator, claim-level source lineage, or automatic transactional rollback.

#### Letta Code Dream/Reflection

- **Primary/version:** [Letta Code tag
  `v0.28.18`](https://github.com/letta-ai/letta-code/tree/v0.28.18),
  repository public 2025-10-24; Apache-2.0.
- **Operator:** post-turn transcript delta triggers a silent reflection
  subagent at compaction events or after 25 steps; it works in an isolated MemFS
  Git worktree, commits memory/skill changes, merges them, and recompiles later
  context.
- **Criteria:** `C1=Y`, `C2=Y`, `C3=Y`; PASS.
- **Capacity/governance:** strongest audited open pattern for explicit tier
  movement, archive/delete policy, conflict isolation, version provenance, and
  rollback. It still does not prove physical erase across all replicas,
  model-weight unlearning, or a global capacity law.

#### Letta AI Memory SDK

- **Primary/version:** [repository tag
  `v0.2.0`](https://github.com/letta-ai/ai-memory-sdk/tree/v0.2.0)
  begins 2025-08-30; Apache-2.0.
- **Operator:** one subconscious agent per subject; `add_messages` launches an
  asynchronous persistent run; callers can wait for completion.
- **Criteria:** `C1=Y`, `C2=Y`, `C3=Y`; PASS for per-turn extraction.
- **Nonclaim:** “offline collective revisioning of all data” is a roadmap item,
  not implemented evidence.

### A2. Mem0

#### Paper and open-source library

- **Primary/status:** [Mem0 paper](https://arxiv.org/abs/2504.19413), first
  public 2025-04-28; ECAI 2025. Audited OSS `mem0ai 2.0.13`, Apache-2.0.
- **Operator:** extract candidate facts from summary/recent messages; retrieve
  ten similar memories; an LLM chooses ADD, UPDATE, DELETE, or NOOP. OSS
  `Memory.add` blocks until mutation finishes.
- **Anchor:** LoCoMo overall Mem0 `66.88`, Mem0 Graph `68.44`, full context
  `72.90`; p95 latency `1.440/2.590/17.117 s`; memory tokens
  `1,764/3,616/26,031`.
- **Criteria:** paper `C1=Y,C2=P,C3=Y`, PARTIAL; OSS
  `C1=Y,C2=N,C3=Y`, PARTIAL.
- **Nonclaim:** `AsyncMemory` is syntax, not an owned queue. History and
  explicit update/delete APIs exist, but expiration hiding is not proven
  physical erase and there is no one-click transaction rollback.

#### Managed Platform V3

- **Primary/current documentation:**
  [V3 Add API](https://docs.mem0.ai/api-reference/memory/add-memories),
  retrieved 2026-07-25. This mutable managed-product page requires a captured
  snapshot and digest before registry promotion.
- **Operator:** return `event_id` and `PENDING`, process one extraction in a
  background queue, append ADD-only memory, expose it to later search.
- **Criteria:** `C1=Y`, `C2=Y`, `C3=Y`; PASS.
- **Capacity/governance nonclaim:** current default is ADD-only and grows;
  explicit user/batch delete and history exist, but causal erasure across
  derived artifacts and transactional rollback are not established.

#### OpenClaw auto-dream

- **Primary/source:**
  [`dream-gate.ts`](https://github.com/mem0ai/mem0/blob/v2.0.13/integrations/openclaw/dream-gate.ts)
  at audited tag `v2.0.13` (commit
  `ca2abca2b884e038d3e525070e79d3057ef2012c`).
- **Operator:** after defaults of 24 hours, five sessions, and 20 memories,
  inject a dream before the next prompt to merge, rewrite, remove noise/secrets,
  or call add/update/delete.
- **Criteria:** `C1=Y`, `C2=N`, `C3=Y`; PARTIAL because the next user waits for
  the dream.

### A3. Zep Cloud and Graphiti

- **Primary/status:** [Zep paper](https://arxiv.org/abs/2501.13956), first
  public 2025-01-20; Graphiti repository begins 2024-08-13; audited Graphiti
  `v0.29.2`, Apache-2.0.
- **Representation:** raw episode subgraph, temporal semantic entity/fact
  graph, and community summaries. Edges retain source episode IDs and separate
  event time from transaction time.
- **Graphiti OSS criteria:** `C1=Y`, `C2=N`, `C3=Y`; PARTIAL.
  `add_episode` performs extraction/dedup/embedding/invalidation before return;
  documentation recommends an application queue but does not own one.
- **Zep Cloud criteria:** `C1=Y`, `C2=Y`, `C3=Y`; PASS.
  Official [batch ingestion](https://help.getzep.com/adding-batch-data) and
  [webhooks](https://help.getzep.com/v3/webhooks) establish asynchronous
  construction and eventual next-wake visibility. Both mutable product pages
  were retrieved 2026-07-25 and require captured snapshots and digests before
  registry promotion.
- **Anchor:** DMR `94.8%` versus MemGPT `93.4%`; LongMemEval GPT-4o-mini full
  context `55.4%/31.3 s` versus Zep `63.8%/3.20 s`; average context
  `115K -> 1.6K` tokens.
- **Capacity/governance nonclaim:** the Cloud docs state no hard graph-count or
  size limit. Deletion APIs exist, but deleting an episode does not rebuild a
  shared entity summary or automatically restore an edge invalidated by that
  episode. This is not full causal erase/rollback.

### A4. LangMem and LangGraph

- **Primary/version:** [LangMem](https://github.com/langchain-ai/langmem)
  begins 2025-01-21; `ReflectionExecutor` 2025-02-05; audited `0.0.30`, MIT.
  [LangGraph](https://github.com/langchain-ai/langgraph) begins 2023-08-09.
- **Operator:** `ReflectionExecutor.submit` schedules local debounced worker
  jobs or durable remote LangGraph runs; a manager semantically retrieves and
  emits insert/update/delete calls into a persistent store.
- **Criteria:** durable `ReflectionExecutor`
  `C1=Y,C2=Y,C3=Y`, PASS; inline manager
  `C1=Y,C2=N,C3=Y`, PARTIAL; `BaseStore` alone
  `C1=N,C2=N/A,C3=Y`, FAIL because storage is not transformation.
- **Capacity/governance nonclaim:** `query_limit=5` bounds the consolidation
  working set, not total memory. Deletes default off; TTL is not semantic
  consolidation; default provenance and rollback are weak; local threads are
  not durable in serverless deployments.

## 8. Direct non-company control: offline recurrence

Lee et al.,
[*Do Language Models Need Sleep? Offline Recurrence for Improved Online
Inference*](https://arxiv.org/abs/2605.26099), first public 2026-05-25, is a
CMU/UMD preprint and must not be attributed to Google, Meta, or Microsoft.

- At a hard/sliding-window eviction boundary, the model performs `N` recurrent
  passes over accumulated context, updates fixed-size SSM fast weights through a
  learned local rule, clears the KV cache, and later predicts from the refined
  weights in one pass.
- Training backpropagates end-to-end through the sleep passes. This is learned
  offline recurrence rather than gradient descent during deployment.
- On GSM-Infinite, Jet six-loop improves six-operation accuracy
  `0.742 -> 0.812` and eight-operation `0.351 -> 0.388`; one reported
  sliding-window setting improves `0.596 -> 0.905`.
- **Criteria:** `C1=Y`, `C2=Y`, `C3=Y`; strict PASS in the benchmark
  lifecycle. It remains a boundary-triggered research prototype, not a
  production asynchronous cluster.
- Fixed fast-weight shape bounds active state, but the paper does not provide
  selective delete, item provenance, rollback, multi-user isolation, or
  long-term semantic capacity. Training cost scales roughly linearly with sleep
  passes `N`.

This title must also not be confused with the earlier Google/Cornell
[*Language Models Need Sleep*](https://openreview.net/forum?id=iiZy6xyVVE):
the former uses repeated learned forward-pass recurrence into fixed fast state;
the latter uses parametric expansion/distillation plus RL-generated dreams.

## 9. Cross-company strict matrix

| Source/system | C1 | C2 | C3 | Destination | Verdict |
|---|:---:|:---:|:---:|---|---|
| EWC | P | P | Y | shared weights + Fisher | task-boundary precursor |
| Memorizing Transformers | Y | N | P | external K/V cache | FAIL |
| Titans/MIRAS/ATLAS/Lattice/Trellis/Memory Caching | Y | N | N/P | neural fast state | FAIL |
| Google Language Models Need Sleep | Y | Y | Y | new low-rank slow experts | EXPERIMENTAL PASS; no deployed lifecycle shown |
| ReasoningBank | Y | Y | Y | structured text + embeddings | EXPERIMENTAL PASS at query/task boundary; no autonomous async scheduler shown |
| GEM/MAS | Y | N | Y | weights + importance/buffer | PARTIAL, online |
| PKM/Memory Layers at Scale | N | N/A | N | sparse parametric pool | FAIL, static substrate |
| Sparse Memory Finetuning | N | Y | Y | selected memory rows | predeployment precursor |
| PAHF | Y | N | Y | per-user text/vector memory | PARTIAL, foreground |
| LongMem/GenerativeAdapter | Y | N | P | K/V bank or generated adapter | FAIL |
| ACON/LEGOMem/M★/WebXSkill | N/P | Y | Y/P | compressor, memory program, or procedural skill | predeployment precursor |
| MemEvolve outer / inner | Y / Y | Y / N | Y / Y | executable provider / content store | outer EXPERIMENTAL PASS; inner FAIL |
| MemSkill learned bank / episode content | N / Y | Y / N | Y / P | controller+30-skill bank / temporary memory | predeployment FAIL / runtime FAIL |
| ALMA outer / dynamic content | Y | Y | Y | executable design / external task memory | EXPERIMENTAL PASS; no autonomous scheduler |
| H-EPM online ACEBench claim/code | Y | P | P | episodic/procedural tool graph | PARTIAL; boundary order unresolved, closest code inline |
| MEMORA | Y | N | Y | abstraction/cue memory | PARTIAL, not strict |
| PlugMem WebArena online path | Y | Y | Y | episodic/semantic/procedural graph | EXPERIMENTAL PASS at task boundary |
| MemMA LoCoMo construction path | Y | Y | Y | repaired text/vector memory | EXPERIMENTAL PASS at session boundary; probe generator unavailable |
| MEMENTO runtime / SFT-RL | Y / N | N / Y | N / Y | within-call summary text + block-conditioned KV / compactor weights | runtime FAIL / predeployment precursor |
| MemCompiler | Y | N | N/P | current text guidance + latent soft tokens | FAIL, per-step delivery compiler |
| HIMA | Y | Y | Y | tiered vector/text/graph memory | PASS pipeline; mechanism evidence PARTIAL |
| MAGE | Y | N | P | execution-state tree | FAIL |
| RHO | Y | Y | Y | persistent instructions, skills, and executable tools | EXPERIMENTAL PASS at completed-trajectory boundary; no autonomous scheduler or independent promotion gate |
| Letta API / Letta Code / Letta AI SDK | Y | Y | Y | prompt blocks/files/skills | PASS |
| Mem0 Platform V3 | Y | Y | Y | text/vector memory | PASS |
| Mem0 OSS / OpenClaw dream | Y | N | Y | text/vector memory | PARTIAL |
| Zep Cloud | Y | Y | Y | raw episodes + temporal graph | PASS |
| Graphiti OSS | Y | N | Y | temporal graph | PARTIAL |
| LangMem durable ReflectionExecutor | Y | Y | Y | JSON/vector/prompt memory | PASS |
| Do Language Models Need Sleep? | Y | Y | Y | fixed SSM fast weights | PASS, research lifecycle |

## 10. Capacity and governance synthesis

### 10.1 Five distinct notions of “bounded”

1. **Prompt boundedness:** ACON, MemGPT paging, Letta blocks, MAGE active path.
2. **Active-state boundedness:** Titans-family fixed fast state, Trellis/Lattice
   slots, HIMA token target.
3. **Physical-store boundedness:** only shown when total retained objects are
   actually capped/reclaimed; top-k retrieval does not count.
4. **Parametric capacity boundedness:** a fixed model/pool can saturate even
   when FLOPs per query remain constant.
5. **Governance boundedness:** deletion, revocation, provenance, and rollback
   work must have an SLA; tombstoning or down-ranking alone does not bound it.

These five rows distinguish what a company artifact means when it says
“bounded”; they are not the complete behavioral-capacity taxonomy. Retention,
addressability/application, and plasticity/acquisition remain separate
endpoints even when prompt, active state, or physical bytes are capped.

### 10.2 Direct overload evidence

- Google Sleep uses a finite preallocated expert pool and has no exhaustion
  experiment.
- ReasoningBank degrades as more memories are retrieved despite an append-only
  store.
- HIMA's LongMemEval curve collapses under aggressive budgets; compaction has a
  distortion floor.
- MEMORA continues adding about 344 net new entries per conversation despite a
  stable merge rate.
- PlugMem's retrieval-budget sweep rises, saturates, and can decline; its
  finite HotpotQA merge reduces active nodes by only `5.0%` at threshold `.7`
  and uses soft deactivation rather than physical reclamation.
- MemMA limits the retrieved view and probe count, but reports neither total
  store bytes nor a repeated-lifetime growth curve; repair may still insert new
  facts and “compact” has no measured storage endpoint.
- MemCompiler bounds Brief State and Soft-Mem while leaving its read-only task
  memory bank outside that bound.
- MEMENTO bounds active within-request KV, but its restart and passcode probes
  show that the visible memento text understates retained semantic state. Peak
  KV can fall while KV area-under-curve rises on a hybrid-attention model.
- MemEvolve and ALMA bound search rounds, not the stores created by generated
  code; MemEvolve's timestamped backups and both systems' design archives can
  grow. MemSkill genuinely caps its reusable skill bank at 30, but its
  per-episode content store remains outside that cap.
- RHO bounds one retrospective search by \(k=10\), \(G=3\), and \(N=3\), not
  the lifetime of the promoted harness, executable tools, content-addressed
  versions, trajectories, diagnoses, backups, or run logs. A 32 KB
  `AGENTS.md` limit constrains one prompt file rather than total procedural
  memory.
- H-EPM admits only judged-successful trajectories, yet every repeated tool
  edge may accumulate multiple state summaries and no lifetime cap is tested.
- Memory Caching bounds selected/active checkpoints, not the total cache.
- The joint FAIR/DeepMind memorization study measures about 3.6
  bits/parameter only in its synthetic no-generalization protocol; it does not
  bound natural-language or deployed sequential capacity.
- Product systems generally bound prompt or retrieval output while leaving the
  durable store unbounded.

The evidence therefore rejects both simple slogans: “just keep everything and
retrieve” and “always consolidate aggressively.”

### 10.3 Candidate lifetime balance law

The following is a synthesis hypothesis, not an established empirical law:

\[
\frac{dM_{\mathrm{durable}}}{dt}
=
\lambda_e\bar b_{\mathrm{admit}}\,[1-d(M_{\mathrm{durable}},Q)]
-
r_{\mathrm{expire}}\bar b_{\mathrm{delete}}
-
r_{\mathrm{compact}}\bar b_{\mathrm{reclaimed}},
\]

where:

- \(M_{\mathrm{durable}}\) is physically retained durable state in bytes;
- \(\lambda_e\) is admitted actionable-event arrival in events/s and
  \(\bar b_{\mathrm{admit}}\) is admitted bytes/event;
- \(d(M_{\mathrm{durable}},Q)\) is the fraction of incoming bytes removed by
  validated deduplication without unacceptable distortion for future query
  distribution \(Q\);
- the final two products are physically deleted and physically reclaimed
  bytes/s.

Every term therefore has unit byte/s. Promotion, tier movement, logical
invalidation, or an updated merge ratio is not reclamation unless bytes leave
the declared total-state boundary. Governance constraints determine which
deletions or compactions are admissible, but do not themselves subtract bytes.

A stable sleep service also requires:

\[
N\lambda_u\,\mathbb E[F_{\mathrm{event}}] < G_s,
\]

for \(N\) active users, per-user admitted rate
\(\lambda_u\,[\mathrm{event}/(\mathrm{user}\cdot\mathrm{s})]\), expected
sleep work \(F_{\mathrm{event}}\,[\mathrm{FLOP}/\mathrm{event}]\), and effective
aggregate service \(G_s\,[\mathrm{FLOP}/\mathrm{s}]\). If it fails,
publication staleness grows without bound even if serving latency remains low.

### 10.4 Destination-routing hypothesis

Promote item `e` into destination `j` only when:

\[
v_U\,\mathbb{E}[\text{reuse}(e)]\Delta Q_j(e)
>
C_{\text{train}}+C_{\text{validate}}+C_{\text{move}}
+C_{\text{interference}}+C_{\text{stale}}+C_{\text{revoke}}.
\]

Here \(v_U\) is a frozen lifecycle-value conversion per unit of future-query
utility; every \(C_\cdot\) term is converted into that same declared value
unit. Without a preregistered conversion or an explicitly constrained
multi-objective comparison, the inequality is dimensionally inadmissible.

The revoke term makes volatile, personal, or legally deletable facts poor
shared-weight candidates. Stable procedures repeatedly validated across
contexts are better candidates for user adapters or shared parameters. Raw
episodes should remain canonical while consolidated memories are versioned,
rebuildable views whenever retention policy permits.

## 11. Infrastructure consequence

The evidence motivates five logical responsibilities rather than one
undifferentiated memory service. A deployment may group snapshot with the
memory fabric and validation/promotion with control, yielding the four service
planes used in the main research design:

1. **Wake plane:** latency-critical inference, retrieval, and minimal write-ahead
   logging.
2. **Snapshot plane:** immutable transcript/experience versions, tombstones,
   source IDs, and change-data capture.
3. **Sleep plane:** queued extraction, replay, distillation, RL/dream
   generation, compaction, and destination routing on isolated state.
4. **Validation/promotion plane:** shadow evaluation, remember/update/forget
   tests, poisoning and privacy checks, canary publication, atomic pointer
   swap, and rollback.
5. **Long-term memory plane:** raw episode store plus text/vector/graph,
   latent/KV, per-user adapter, and shared-weight tiers with independent
   retention policies.

The missing systems primitive is a transactional sleep commit:

1. read immutable snapshot `v`;
2. produce candidate `v+1` with source IDs and model/prompt/config hashes;
3. validate utility, interference, poisoning, deletion, and regressions;
4. atomically publish the candidate to later wake;
5. retain prior state for rollback;
6. propagate tombstones through every derived view;
7. rebuild or quarantine parametric destinations when exact unlearning is
   unavailable.

This minimizes wake/sleep data movement by co-locating immutable snapshots with
sleep workers and moving validated deltas or content-addressed artifacts rather
than complete histories.

## 12. Research questions opened by the audit

1. **Trigger:** Which policy should start sleep—idle time, novelty, uncertainty,
   error, store pressure, expected reuse, or a learned benefit-per-FLOP score?
2. **Stage semantics:** Can “NREM” be operationalized as high-fidelity replay
   and deduplication, and “REM” as counterfactual recombination, with measurable
   rather than metaphorical distinctions?
3. **Destination:** Which evidence should remain raw, become text/vector/graph,
   compile into latent/KV state, enter a user adapter, or enter shared weights?
4. **Capacity:** What is the correct capacity unit across tokens, objects,
   graph edges, latent rank, expert slots, interference, retrieval competition,
   and delete/rebuild workload?
5. **Scaling:** What function relates future wake loss to sleep FLOPs, memory
   bytes, staleness, data movement, and reversibility? Where are the
   destination-specific crossovers?
6. **Safety:** How should a candidate memory be validated against poisoning,
   self-training drift, mis-consolidation, and preference change before
   publication?
7. **Governance:** Can one deletion request be proven to remove or quarantine
   its effects from raw, semantic, graph, latent, adapter, and shared-weight
   replicas?
8. **Longitudinal evaluation:** Which benchmark streams real users/tasks long
   enough to expose saturation, stale retrieval, compaction damage, router
   collapse, and sleep-queue backlog?
9. **Economics:** When is parametric promotion cheaper than repeated external
   retrieval after training, validation, interference, rollback, and revocation
   costs are included?
10. **Systems:** What consistency model best trades wake latency against sleep
    freshness, and which snapshot/publication protocol prevents concurrent wake
    writes from being lost?

## 13. Defensible thesis boundary

Prior art already establishes periodic replay, fast-to-slow consolidation,
learned retention, learned replay selection, synthetic dreaming,
external-to-parametric distillation, fixed and growing memory architectures,
and several working background memory products.

The defensible research target is:

> a versioned lifetime memory state machine that learns when and where to route
> wake-derived information across raw, external structured, latent, user-
> parametric, and shared-parametric media under joint accuracy, latency, energy,
> storage, staleness, data-movement, interference, reversibility, privacy, and
> deletion constraints.

That target is broader than any one audited paper and narrower than claiming
the invention of sleep-time compute itself.
