# Sleep-Time Training and Data Methods Atlas

- Audit cutoff: 2026-07-25
- Status: pre-registry synthesis; not a canonical evidence ledger
- Scope: sleep-time data construction, replay, distillation, regularisation,
  parameter isolation, synthetic dreaming, reinforcement learning, external
  memory transformation, and promotion verification
- Companion audits:
  [biological/computational sleep](PRIMARY-SOURCE-AUDIT-BIOLOGICAL-COMPUTATIONAL-SLEEP.md),
  [early wake--sleep algorithms](PRIMARY-SOURCE-AUDIT-EARLY-PRECURSORS.md),
  [agent-memory systems](PRIMARY-SOURCE-AUDIT-AGENT-MEMORY-SYSTEMS.md),
  [company lineages](PRIMARY-SOURCE-AUDIT-COMPANY-LINEAGES.md),
  [capacity theory](PRIMARY-SOURCE-AUDIT-CAPACITY-CONSOLIDATION-THEORY.md), and
  [July 2026 counterevidence](PRIMARY-SOURCE-AUDIT-JULY-2026.md)

This document answers a narrower question than the general survey:

> If an AI system is allowed to learn after a wake period, exactly what data
> should it construct, what objective should it optimize, what state should it
> modify, and what evidence is required before that state is reused?

The literature does not support one universal “sleep loss.” It supports a
factorization of the sleep operator.

```text
wake events
→ admission and snapshot
→ replay/dream/abstraction corpus
→ training or transformation operator
→ candidate external/latent/parametric state
→ retention + acquisition + governance validation
→ publish, quarantine, or roll back
```

## 1. Classification boundary

A deployed method is called strict sleep-time compute here only when:

1. it consumes accumulated wake experience;
2. the transformation runs after that experience and off the user-visible
   action path or against an immutable snapshot; and
3. the resulting state persists and is reused in a later wake period.

This separates four objects that papers often place under one label.

| Object | Updated state | Timing | Durable across later wake? |
|---|---|---|---|
| inference-time reasoning | transient tokens | after a query | normally no |
| test-time training | weights/fast state | on the query or execution path | sometimes |
| token-space sleep compilation | text/context artifact | before a future query | yes for that context |
| lifetime sleep learning | external, latent, adapter, or slow weights | delayed/off-path | yes |

The [Letta sleep-time compute paper](https://arxiv.org/abs/2504.13171) primarily
establishes the third object. It rewrites a known context into a query-useful
text representation and reports an inference-cost/accuracy frontier; it does
not train model weights. SIESTA, Auto-Dreamer, Microsoft HIMA, and Google
*Language Models Need Sleep* are closer examples of the fourth object in,
respectively, vision, external agent memory, tiered external memory, and
parametric fast-to-slow consolidation.

## 2. Chronological training lineage

Dates below refer to earliest public versions where known; later archival
venues are named separately.

| Date | Work | Sleep corpus | Operator | Destination | Main boundary |
|---|---|---|---|---|---|
| 1989–1990 | McCloskey--Cohen; Ratcliff | old and new examples | interleaved training analysis | shared weights | establishes catastrophic interference, not a sleep system |
| 1995 | CLS; Robins pseudorehearsal | reinstated episodes or pseudo-items | slow interleaved learning | slow distributed memory | theory and small networks |
| 2016–2017 | EWC, SI | current data plus parameter-importance state | regularised gradient descent | shared weights | protects directions but can consume plastic subspace |
| 2017 | Deep Generative Replay | current real plus generated old inputs | solver distillation and generator replay | generator + solver weights | generator/teacher carry the old information |
| 2017 | DGDMN | generated samples from full short-term modules | bounded replay mixture | long-term generator/classifier | explicit sleep trigger, but task batches and weak reproducibility |
| 2017–2018 | GEM, MAS | episodic examples or unlabeled activation gradients | projected/regularised online updates | shared weights | update occurs on the wake path |
| 2017–2018 | FearNet | Gaussian old latent samples plus all recent real latents | reconstruction + classification | mPFC network and class statistics | fixed cadence; class statistics grow |
| 2018 | Progress & Compress | current-task inputs/states with active-column outputs | KL/cross-entropy distillation + online EWC | fixed-size knowledge base | known task boundary; not autonomous sleep |
| 2019 | CLEAR | recent on-policy and replayed old trajectories | off-policy RL + policy/value cloning | policy/value weights | strong wake/replay baseline, not delayed sleep |
| 2020 | Brain-Inspired Replay | generated old plus current samples | replay with feedback connections | classifier/generator | generator quality and task setting remain limiting |
| 2022 | PAD | replayed, occluded, mixed, and perturbed latents | NREM reconstruction and REM separation losses | visual representation | biological role assignment is an engineered analogy |
| 2022 | Singh--Norman--Schapiro | autonomous hippocampal/neocortical attractor transitions seeded by random activity | contrastive Hebbian/error-driven NREM/REM sleep updates | neocortical weights | strict experimental sleep; stage schedule is fixed and tasks are synthetic categories |
| 2022 | Golden et al. PLOS | spontaneous Poisson activation from mean firing rates | unsupervised STDP | SNN synapses | two tasks, 10 simulated networks, residual trace required |
| 2022 | Sleep Replay Consolidation | stored activation means plus tiny real rehearsal | Poisson replay + local Hebbian plasticity | ANN/SNN-converted weights | pure intrinsic replay remains far below joint training |
| 2023 | SIESTA | balanced OPQ-compressed latents, optionally MixUp/CutMix | supervised sleep SGD | upper 97.81% of network | bounded latent store; large sleep-update count |
| 2023–2025 | WSCL | STM/LTM samples plus external disjoint labeled data | alternating NREM/REM supervised batches | selected network layers | evaluated “dreams” are real external images |
| 2024–2025 | PCMC | raw STM/LTM image patches | SimCLR-style sleep retraining, re-indexing, stochastic pruning | encoder + centroid stores | memory reduction, not a bounded lifetime store |
| 2025 | Letta sleep-time compute | known context without future query | prompted inference/rewrite | reusable text context | no gradient training or lifetime capacity test |
| 2025 | SEAL | model-generated self-edits, augmentations, and update directives | SFT inner update; downstream reward trains self-edit policy | model weights | adaptation framework; not by itself a delayed lifecycle |
| 2025 | Spens et al. | recent memories, world-model abstractions, generated samples | recurrent PPO selects offline action and replay value | task-specific learner/model | explicit learned sleep control in bounded toy domains |
| 2025 | Cartridges | source-conditioned synthetic conversations and QA | full-context teacher distillation | compact recurrent/KV state | corpus-specific compilation and teacher-coverage risk |
| 2025 | H-EPM | successful tool trajectories with episodic state summaries | online graph update is claimed; graph-guided GRPO rollouts in a separate experiment | external tool graph; optionally policy weights | ACEBench boundary order unresolved and closest released online writer inline; parametric update remains a training loop |
| 2025 | MemEvolve | completed task traces, success, token cost, and latency | fixed-LLM Pareto evolutionary search over encode/store/retrieve/manage code | executable memory architecture plus provider-specific content | outer batch-boundary experimental pass; inner content update remains wake-blocking and backups can grow |
| 2025 | Memento 2 | cases, outcomes, feedback, policy/value estimates | foreground Parzen-policy and \(Q\) update around a frozen LLM | external cases and controller state | wake-time same-task learning; convergence requires finite current memory but supplies no fixed-budget coverage mechanism |
| 2026 | MemSkill | offline task episodes and periodically collected hard cases | PPO skill selector plus fixed-LLM designer editing a natural-language skill bank | controller weights plus at-most-30 reusable skills | predeployment learning, not deployed wake sleep; training snapshots are not semantic rollback |
| 2026 | ALMA | completed memory-collection/deployment tasks and design scores | open-ended fixed-LLM code search over database, update, and retrieval | executable memory design and task-boundary external content | outer design and dynamic task loops pass experimentally; no autonomous scheduler or online design evolution |
| 2026 | PlugMem | dialogue, document, or completed action traces | LLM standardisation, proposition/prescription induction, merge/soft deactivate | provenance-linked external knowledge graph | strict experimental WebArena task boundary; no periodic scheduler or lifetime cap |
| 2026 | MSSR | buffered old samples plus per-sample loss/retention state | memory-strength-aware interval, ratio, and sample scheduling with LoRA updates | model LoRA + replay-state scalars | adaptive continual-training replay, but updates remain in the training loop rather than a distinct off-path sleep service |
| 2026 | MemMA | completed dialogue session plus five synthetic single/cross-session/temporal probes | frozen-state verification, failure-to-fact repair, semantic skip/merge/insert, write-back | repaired external text/vector memory | strict experimental session boundary; paper-aligned probe generator unavailable and one LoCoMo conversation only |
| 2026 | Memento-Skills | failed task, ground-truth-assisted judge, reflection, synthetic router goals | same-question skill rewrite/retry; separate InfoNCE router fitting | external Markdown/code skills plus embedding router | paper loop is wake-time; growth 5→41/235 has no cap, versioned deletion, or sleep scheduler |
| 2026 | Microsoft MEMENTO | OpenThoughts-v3 traces segmented by atomization, LLM boundary scores, dynamic programming, GPT-5.x compression, and independent rubric feedback | full-attention SFT → block-masked SFT; optional rule-reward CISPO | compactor weights plus transient textual mementos and block-conditioned KV | predeployment training followed by within-query compaction; hidden KV is part of the effective state and physical source-block eviction is not semantic erasure |
| 2026 | M★ | task episodes plus static/rotating validation failures | population-based reflective code evolution | executable memory schema, write/read logic, and instructions | predeployment search over the memory harness itself, not deployed wake consolidation |
| 2026 | Memento-Skills DreamDaemon | post-response session summaries staged in Markdown | quick async or 600-second daemon scan; LLM create/update/delete of topic views | external topic Markdown plus index | post-paper code-level strict pass; no Dream experiment, hard cap, provenance, transactional journal, or rollback |
| 2026 | Evolve | query-derived teacher sections, TTL, category, staging/canonical overlap | nightly/manual deterministic orchestration plus teacher compile at similarity ≥.85 | external vector sections plus SQLite metadata | strict external pass, but all phases repeat identical questions; cross-store publication is non-atomic and canonical lifetime is unbounded |
| 2026 | MemCompiler | successful teacher trajectories plus current observation, Brief State, and read-only episode memory | compiler SFT then GRPO; per-step state-conditioned text and latent-token compilation | compiler/Soft-Mem parameters and transient Brief State | learned delivery policy on the wake path; not durable memory consolidation |
| 2026 | Retrospective Harness Optimization | completed unlabeled task trajectories, difficulty-diverse coreset, repeated re-solves, and self-diagnoses | DPP selection, group rollout, self-validation/consistency, best-of-\(N\) full-harness proposal, and pairwise self-preference | persistent instructions, natural-language skills, and executable tools | strict experimental retrospective boundary; one-round search and same-model gate do not bound or safely attest the lifetime harness |
| 2026 | Surprise-Gated Memory | labeled frozen-encoder image embeddings or one-shot image/text facts admitted by calibrated prediction error | periodic full interleaved replay into a slow linear readout, or fast-to-slow prototype/familiarity migration | frozen backbone + episodic embeddings + slow readout/prototype store | strict proof-of-concept sleep; surprise halves one buffer without measured gain, recent-window replay is destructive, and code/full settings/lifetime cap are absent |
| 2026 | Auto-Dreamer | typed memory region plus provenance-linked trajectories | GRPO using later agent performance | compact external procedural memory | strict external sleep; bounded working region/tool budget and empirically compact active bank, but no hard lifetime cap and archive retention remains unresolved |
| 2026 | Google *Language Models Need Sleep* | knowledge-seeding targets and generated dream questions | GKD/SFT plus ReSTEM dream-generator learning | newly activated slow expert | strict experimental benchmark STC; finite evaluated preallocation, but lifetime slot count and exhaustion behavior are untested |
| 2026 | Facts in Weights | bare facts or 24-item study packets | SFT, offline/online distillation, periodic redistillation, GRPO probes | LoRA-merged weights | retention plateaus; recursive teacher drift |
| 2026 | TrustMem | old memory, new evidence, and candidate transformations | preference-guided policy over write/revise/prune/no-op | external memory | transition-quality learning; hard lifetime cap unresolved |
| 2026 | Microsoft HIMA | accumulated events selected by salience and time | deterministic dedup, abstraction, tier promotion, decay | hot/episodic/KG stores | strict external pipeline; default six-hour cadence not empirically tested |

The important chronology is not that “sleep appeared in 2025.” Explicit
wake--sleep training, generated replay, fixed-cadence transfer, and pruning
predate the LLM term; learned offline orchestration appeared independently in
the 2025 Spens work. The 2025–2026 shift is toward query-independent LLM
context compilation, agent memory, parametric experts, and lifecycle
infrastructure.

## 3. What should be in a sleep corpus?

### 3.1 Canonical evidence

Canonical wake records are the only representation that can support exact
correction, replay regeneration, deletion fan-out, and factual audit. They can
be text, tool results, environment transitions, user corrections, or
privacy-scoped event records.

Raw replay is expensive, but the literature does not justify eliminating it
unconditionally:

- [Tiny Episodic Memories](https://arxiv.org/abs/1902.10486) reports material
  benefit from as little as one old exemplar per class in its evaluated
  continual-learning benchmarks.
- Sleep Replay Consolidation improves markedly when only 0.75% real rehearsal
  is added to intrinsic replay.
- *Facts in Weights* shows that a forgotten weight-stored fact can often be
  recovered when the fact is supplied in context, and argues for an external
  canonical copy.
- Auto-Dreamer retains provenance-linked source trajectories while replacing
  only an active memory region.

Therefore this program uses a conservative experimental baseline—not a
demonstrated safety guarantee—consisting of a hybrid with a small stratified
canonical anchor set rather than generator-only replay.

### 3.2 Bounded episodic or coreset replay

Selection policies include:

| Policy | Example | Benefit | Failure mode |
|---|---|---|---|
| uniform reservoir | WSCL LTM | unbiased stream sample under fixed bytes | rare high-utility events may be omitted |
| class-balanced | SIESTA | controls majority-class domination | requires known labels/classes |
| most-populated-class eviction | SIESTA | caps latent count | ignores future utility within a class |
| recent plus uniform historical | CLEAR | mixes plasticity and stability | old distribution and policy may be stale |
| gradient constraint buffer | GEM | explicitly protects observed old losses | cost grows with constraints/tasks |
| novelty/utility admission | PCMC, HIMA | avoids redundant writes | estimator error becomes permanent selection bias |
| value plus diversity | Spens image/maze branches | prioritizes useful non-duplicate episodes | demonstrated only in selected action spaces |

A sleep paper must report both the active replay bytes and the canonical bytes
that remain elsewhere. “No raw replay in the update” is not the same claim as
“the system stores no raw information.”

### 3.3 Latent replay

Latent replay stores a frozen encoder output or sufficient statistic instead
of the original input.

- SIESTA stores `14 x 14 x 80` MobileNetV3 latents compressed with OPQ.
- FearNet samples Gaussian class statistics in its learned mPFC latent space.
- PAD constructs occluded or mixed latent episodes.
- Generative Adapter accumulates a streaming hidden-state statistic from which
  a hypernetwork produces an adapter.
- Cartridges compile a source corpus into compact recurrent/KV state.

This saves I/O and sometimes privacy surface, but introduces representation
expiry. If the encoder changes, old latents can become incompatible; if the
encoder is frozen forever, future-domain plasticity may suffer. A credible
system records the encoder/checkpoint digest and either re-embeds from
canonical data or retires incompatible latents.

The inverse mismatch is equally important for text/vector memory: if sleep
rewrites the text while preserving its old vector, the latent key is now a
stale representation of a new payload. Payload digest, vector digest, encoder
digest, provenance, and authorization state must be published as one
generation. With a fixed encoder, refresh cost scales with changed items; an
incompatible encoder update can force a whole-store re-embed.

### 3.4 Generated replay

Generated replay has three distinct forms.

1. **Pseudo-input replay.** DGR and DGDMN generate old-domain inputs and use an
   old solver or stored classifier as teacher.
2. **Distributional latent replay.** FearNet samples class-conditional
   Gaussians; PLOS 2022 drives spontaneous activity from stored mean firing
   rates.
3. **Query or curriculum generation.** SEAL restructures an input into its own
   training examples; Google Sleep generates candidate questions; Spens lets a
   controller select generated experience.

Generated replay does not remove memory. It moves memory into generator
parameters, teacher checkpoints, stored statistics, prompts, random seeds, and
generation compute. Recursive use adds a distinct corruption channel:

```text
teacher error
→ generated target bias
→ student state drift
→ next-cycle teacher error
```

The *Facts in Weights* recursive self-merged teacher is direct warning
evidence: the frozen original teacher condition is much stronger than using
the repeatedly modified model as its own teacher. Every recursive sleep system
therefore needs a frozen or independently refreshed anchor and a cycle-depth
ablation.

### 3.5 Dream data and augmentation

The word “dream” hides at least four mechanisms.

| Mechanism | Concrete operator | Evidence boundary |
|---|---|---|
| corruption/occlusion | PAD NREM, SIESTA MixUp/CutMix | improves a chosen invariance; does not discover new facts |
| interpolation/recombination | PAD REM latent mixing | may improve separation; generator bias can create impossible samples |
| counterfactual question generation | Google Sleep dreaming | downstream-filtered curriculum; reward can overfit benchmark verifier |
| cross-session procedural abstraction | Auto-Dreamer | synthesizes compact replacement memory, not new model knowledge |

WSCL is a caution against biological relabeling: its evaluated REM phase uses
an external real labeled dataset with disjoint classes. The final paper does
not establish internally generated dreams.

Good dream data should satisfy:

1. coverage of old, rare, failed, corrected, and recent cases;
2. novelty beyond paraphrase without losing source grounding;
3. explicit mixture weights for canonical, generated, adversarial, and random
   controls;
4. a verifier independent enough not to reproduce the generator's errors;
5. held-out future-task utility rather than dream likelihood as the endpoint;
6. lineage from every generated item to source state, generator, prompt,
   decoding configuration, and verifier;
7. a rejection path for privacy, poison, contradiction, and impossible
   counterfactuals.

## 4. What does sleep optimize?

No audited paper implements the following complete objective. It is a proposed
comparison scaffold that makes each paper's omitted terms visible:

\[
\begin{aligned}
\mathcal J_{\text{sleep}} ={}&
\lambda_{\text{new}}\mathcal L_{\text{new}}
+\lambda_{\text{replay}}\mathcal L_{\text{old}}
+\lambda_{\text{KD}}D(p_{\text{teacher}}\Vert p_{\text{candidate}})\\
&+\lambda_{\text{stability}}R(\theta,\theta_{\text{incumbent}})
+\lambda_{\text{repr}}\mathcal L_{\text{representation}}
-\lambda_{\text{future}}\widehat U_{\text{future}}\\
&+\lambda_{\text{cost}}C_{\text{lifecycle}}
+\lambda_{\text{gov}}C_{\text{governance}} .
\end{aligned}
\]

Each loss, utility, and lifecycle/governance component is normalized by a
preregistered scale. The corresponding \(\lambda_\cdot\) coefficient converts
it into one common dimensionless optimization unit; raw currency, FLOPs, bytes,
seconds, and task loss may not be added directly. A deployment may instead
retain the vector objective and enforce explicit resource constraints.

The candidate is publishable only under separate constraints:

\[
\Delta U_{\text{protected}}\ge -\epsilon_{\text{ret}},
\quad
g_{\text{acq}}\ge g_{\min},
\quad
C_{\text{delete}}\le B_{\text{delete}},
\quad
\rho_{\text{sleep}}<1.
\]

### 4.1 Supervised replay

SIESTA, WSCL, and ordinary rehearsal minimize supervised cross-entropy over
current and replayed samples. This is the strongest simple baseline because it
uses labels directly. Any dream or distillation method should be compared at
the same:

- canonical/replay bytes;
- optimizer updates and accelerator time;
- old/new sample ratio;
- augmentation budget;
- model and pretraining checkpoint;
- class/task information;
- lifetime rather than per-sleep compute.

### 4.2 Distillation

Distillation transfers behavior rather than exact source records.

- Progress & Compress uses the plastic active column as teacher for a stable
  knowledge base while online EWC protects older knowledge.
- DGR uses the previous solver's outputs on generated old inputs.
- Cartridges uses a full-context teacher to train a compact state.
- Google Sleep Knowledge Seeding uses generalized knowledge distillation from
  the pre-transfer fast state into a newly unmasked slow expert, then resets
  old high-frequency experts.
- *Facts in Weights* compares offline and online KL directions, frozen versus
  recursively updated teachers, and periodic full redistillation.

The teacher must be named as part of the stored state. “No replay data” is not
a complete accounting if a frozen full model or prior adapter supplies the old
targets.

At minimum, report:

```text
teacher identity/version and whether it has already self-updated
input distribution used for distillation
KL direction, temperature, and coefficient
hard-label versus soft-target mixture
teacher/student capacity and active parameters
reset/reinitialization sequence
old/new/protected-slice outcomes
cycle-depth drift
```

### 4.3 Regularisation and gradient constraints

EWC and SI penalize movement in parameter directions estimated to be important.
GEM projects a proposed gradient so episodic losses do not increase. These
methods preserve old behavior differently:

```text
EWC/SI: store importance + reference parameters
GEM: store old examples + solve constraints
P&C: distill current behavior + regularise old knowledge base
```

All three can defer rather than eliminate saturation. Protected directions
accumulate, the replay buffer grows or becomes more selective, and acquisition
may degrade before aggregate old-task retention. That is why the present
program separates `H-STC-007` from ordinary forgetting.

### 4.4 Isolation, sparse write, and growth

Adapters, experts, masks, sparse memory rows, and frozen columns reduce
interference by assigning new updates to less-shared state.

- Progressive Networks allocate and freeze columns.
- PackNet repeatedly prunes and freezes weights.
- HAT learns task masks.
- Sparse Memory Finetuning updates selected memory rows.
- Google Sleep activates a previously masked slow expert and trains only that
  new expert during Knowledge Seeding.

These are capacity allocation mechanisms, not infinite memory. Report total,
active, free, recyclable, quarantined, and archived parameters separately.
Google Sleep's evaluated implementation preallocates a finite pool; activating
one expert does not demonstrate an exhaustion, merge, eviction, or routing
policy after the pool is full.

### 4.5 Contrastive and representation objectives

PCMC retrains its encoder with a patch-level SimCLR-style contrastive loss,
then re-embeds centroids and prunes raw patches. PAD assigns reconstruction to
its NREM-like phase and separation under mixed/perturbed replay to its REM-like
phase.

These objectives can improve retrieval geometry without preserving exact
facts. Evaluation therefore needs:

- exact and semantic recall;
- temporal and relational queries;
- nearest-neighbor confusion and false association;
- representation rank and cluster separation;
- re-index cost and compatibility across versions;
- reconstruction from canonical source after encoder change.

### 4.6 External-memory transformation

External consolidation may use no gradient at all:

- Mem0 extracts candidate facts and applies add/update/delete/no-op actions.
- Zep/Graphiti constructs bi-temporal entities, facts, and invalidation edges.
- MEMORA merges an abstraction with concrete episodic values.
- H-EPM compiles successful tool trajectories into state-annotated procedural
  graph edges for later tasks.
- PlugMem standardises raw traces, induces provenance-linked propositions and
  prescriptions, and can merge then soft-deactivate redundant semantic nodes.
- MemMA tests a provisional session store with synthetic questions and converts
  failures into skip/merge/insert repairs before later sessions use it.
- HIMA scores, deduplicates, summarizes, promotes, decays, and prunes across
  hot, episodic, and knowledge-graph tiers.
- Auto-Dreamer learns how to replace a selected typed-memory region.

The training question moves from “which weight loss?” to:

```text
which events are admitted?
which records may be coalesced?
what exact evidence must remain?
what future-query utility justifies an abstraction?
which contradiction closes or versions a fact?
how does deletion reach every derived node?
```

An active index can be bounded while the raw archive, invalidated graph,
provenance ledger, and generated candidates grow without bound. Both active and
total state must be measured.

PlugMem supplies unusually clear task-boundary evidence. In repository commit
`3b2ce75257d40bca8fac3f78e54e22ea41d92529`, its WebArena loop first exits
`while not env.done()` and freezes `final_status`; only then does
`memory.close()` run semantic/procedural extraction and insert the graph before
the already computed status is returned. The transform output is therefore not
needed by the completed task (`C1=C2=C3=Y`). This is synchronous post-task tail
work, not a queued daemon, while stepwise trace append and retrieval remain
inline. Its HotpotQA merge result uses soft deactivation and does not establish
physical reclamation or repeated-lifetime equilibrium.

MemMA supplies a complementary session-boundary pattern. At repository commit
`51d463ad24f8b4f63633d12b32df7c64ed285fff`, the released LightMem path
ingests all turns in one LoCoMo session, then tests the provisional store with
five synthetic probes, derives repair facts from failed probes, consolidates
them semantically, writes them back, and retests before the next session. This
is `C1=C2=C3=Y` at the session boundary, although turn-level construction is
still wake-time. The paper-aligned quick start uses a committed 76-row probe
parquet whose generation pipeline is promised but not released. Its
single-conversation evaluation, same-model answer/judge coupling, and lack of
seeds, confidence intervals, total-storage accounting, physical erase, and
rollback make it a narrow experimental pass rather than a production
control-plane precedent.

M★ belongs to an emerging memory-harness evolution lineage rather than
originating it. It uses task episodes, static and rotating validation queries,
successful/failed traces, and a coding-agent reflector to evolve the executable
`Schema`, write/read `Logic`, and `Instruction` of a memory system. Twenty
population-search iterations repeatedly re-ingest episodes and score
candidates. This is strong evidence that a sleep control plane may learn the
**operator and destination policy**, but it is predeployment program search:
it neither consumes one deployed user's wake stream nor trains base-model
weights. Its finite search budget also does not bound the lifetime store
created by the selected program.

### 4.7 Learning the memory architecture itself

Five papers now form a chronological control-program lineage, and their learned
objects must not be collapsed:

| First public | System | Learned object | Boundary verdict | Capacity/recovery boundary |
|---:|---|---|---|---|
| 2025-12-21 | [MemEvolve](https://arxiv.org/abs/2512.18746) | executable `Encode/Store/Retrieve/Manage` provider plus evolving content | outer architecture evolution `C1=C2=C3=Y`, experimental `B/P`; inner content update is wake-blocking `W` | no common cap; per-round active store may be renamed to timestamped backup; optimiser checkpoint is not semantic rollback |
| 2026-02-02 | [MemSkill](https://arxiv.org/abs/2602.02474) | PPO selector plus natural-language memory skills edited by a fixed LLM | learned artifact `C1=N,C2=Y,C3=Y`, predeployment `P`; episode content is `Q/W` and resets | normal released cap is 30 skills, replacing lowest average-reward used skill or least-used fallback; snapshots/checkpoints are training recovery |
| 2026-02-08 | [ALMA](https://arxiv.org/abs/2602.07755) | executable database/schema, update, and retrieval design \(M=(U,D,R)\) | outer search and dynamic post-task content loop are experimental passes; online evolution of design code is future work | archive and arbitrary generated runtime stores have no common cap, deletion contract, or safe promotion protocol |
| 2026-04-10 | [M★](https://arxiv.org/abs/2604.11811) | task-specific schema, write/read logic, and agent instruction | `C1=N,C2=Y,C3=Y`, predeployment `P` | 20 search iterations bound evaluation, not the selected store; no deployed revocation or rollback |
| 2026-06-04 | [Retrospective Harness Optimization](https://arxiv.org/abs/2606.05922) | complete persistent harness: instructions, skills, and executable tool scripts | completed trajectories are transformed and a positive-scoring winner is published before held-out tasks, `C1=C2=C3=Y`, experimental `B/P` | \(k=10,G=3,N=3\) bound one search; files, versions, trajectories, diagnoses, backups, and logs remain uncapped, while self-preference is not independent promotion attestation |

MemEvolve predates M★ by almost four months and evolves the full modular memory
provider from completed execution batches. In repository commit
`6035d5659d7a092dbfa6a87b1a32a3cee652ba54`, provider runs complete before the
meta-evolver consumes their logs and promotes code for the next round. Its
inner `take_in_memory` call, however, occurs after answer generation but before
the task function returns, so architecture evolution passes the boundary test
while content evolution does not. The accepted ICML 2026 camera-ready record
and the arXiv v1 have different author lists; the source registry must preserve
both versions rather than silently merge them.

ALMA is the closest direct precursor to M★. A Meta Agent discovers executable
memory designs over four sequential-decision domains. In dynamic deployment,
the environment action and reward are finalized, then `general_update` mutates
memory for the next task. The learned design reaches `53.9%` overall with
GPT-5-mini versus `41.1%` no-memory, but the design archive itself is
unbounded, runtime governance depends on arbitrary generated code, and the
paper does not evolve the design online during deployment.

MemSkill contributes a different mechanism: a small PPO controller selects
skills and a fixed-LLM designer edits the bank every 100 training steps. Its
released default cap is **30**, not the `OperationBank` fallback of 20. At
capacity, replacement uses lowest average reward among used skills or a
least-used fallback; a 2,000-item hard-case pool and training snapshots are
also bounded. These are valuable eviction and recovery primitives, but the
learned bank is produced before deployment and per-episode memory has no total
cap.

RHO is the first item in this lineage whose released workflow directly mines
completed agent sessions and can replace the entire persistent harness for
later work. It selects a difficulty-diverse ten-task coreset, re-solves each
three times, derives self-validation and cross-rollout-consistency diagnoses,
proposes three harnesses, and accepts only the best candidate with positive
mean pairwise self-preference. Its reported one-round SWE-Bench Pro
optimization requires 103 agent invocations after selection; difficulty
selection adds 100 judge calls and one batched embedding call. This is a real
sleep-cost term, not free reflection. The same GPT-5.5 family solves,
diagnoses, proposes, and ranks, the chosen candidate is not always the hidden
test optimum, and adversarial trajectory content can become executable
persistent behavior. Thus none of the five alone demonstrates a deployed,
autonomous, independently attested, capacity-bounded, reversible sleep control
plane.

### 4.8 Teaching within-query state compaction

[Microsoft MEMENTO](https://arxiv.org/abs/2604.09852) is not sleep-time
learning, but it is unusually direct evidence that **memory-management
behavior can be taught by changing training data and attention constraints**.
Its released data pipeline at commit
`d8c10e66ff313844b4b5f063dd4c075d3ea56aab` performs:

```text
OpenThoughts-v3 trace
→ indivisible sentence/code/equation atoms
→ GPT-5.x local boundary scores in {0,1,2,3}
→ dynamic-programming global segmentation
→ GPT-5.x block compression
→ independent six-dimension judge
→ up to two feedback-guided retries
```

The
[pipeline implementation](https://github.com/microsoft/memento/blob/d8c10e66ff313844b4b5f063dd4c075d3ea56aab/data/pipeline/run_full_pipeline.py#L205-L330)
separates local semantic judgement from the combinatorial partition; this is a
general sleep-data design pattern. The released
[OpenMementos](https://huggingface.co/datasets/microsoft/OpenMementos) record
contains `228,557` annotated traces (`54%` math, `19%` code, `27%` science).
One-pass compression passes the paper's rubric only `28%` of the time; two
feedback rounds raise the reported rate to `92%`. Thus generated summaries
need explicit downstream-sufficiency tests, exact-value/formula checks, and
rejected-candidate lineage rather than a generic “summary quality” score.

For already-reasoning checkpoints, training first uses ordinary full
attention and then block-masked attention, five epochs per stage on 31K
examples at 32K sequence length. The first stage teaches format; the second
removes prior reasoning blocks from future attention and forces the textual
memento to carry forward sufficient state. A non-reasoning checkpoint needs
an additional ordinary reasoning-SFT stage. Optional CISPO on DolciMath uses a
rule-based SymPy reward and improves single-sample accuracy while weakening
some KV savings. This is a useful `SFT → constrained SFT → RL` curriculum for
a **future sleep consolidator**, but MEMENTO itself trains before deployment
and compacts state inside the current generation.

Most importantly, its restart ablation shows that the output artifact is not
only text. Re-prefilling identical memento text without the original
block-conditioned KV lowers AIME24 pass@1 from `66.1%` to `50.8%`. Downstream
KV probes recover random five-digit information from already masked blocks
above chance. A training manifest must therefore declare both explicit and
implicit state channels:

```text
visible summary text
block-conditioned KV or recurrent state
source block and attention mask
restart/migration semantics
privacy and deletion residual probes
```

Calling the text “self-contained,” evicting the source tokens, or measuring
only peak KV would otherwise hide capacity, portability, and erasure costs.

### 4.9 Surprise-gated admission with full interleaved replay

[Surprise as a Signal for Plasticity and
Metacognition](https://arxiv.org/abs/2606.31495) supplies a deliberately small,
low-evidence but useful data-policy ablation. A prediction-error residual over
a frozen representation gates whether an embedding enters an episodic buffer;
periodic sleep replays old and new embeddings into a slow linear readout. On
the 1000-class DINOv2 arm, surprise-gated admission halves stored examples and
scores `84.2` versus full replay `83.7`, but both are single runs and the paper
explicitly treats the difference as unmeasured variation. The supported claim
is storage reduction without observed loss, not superior learning.

The negative control is more important for lifetime design. Replaying only
the latest five tasks yields `41.2` retention versus `67.0` for no replay on
DINOv2 and `0.0` versus `25.9` on I-JEPA. A recent-only bounded buffer can
therefore intensify interference by repeatedly training the slow state on the
wrong mixture. Full interleaving recovers old retention but makes per-cycle
replay grow with retained classes; classwise cores such as three examples per
class still grow linearly with an open-ended label space.

Its vision-language proof of concept uses a second path: a calibrated frozen
detector admits a one-shot embedding/text fact, and a manual or
fast-store-full sleep trigger migrates it into prototypes/familiarity state
before clearing the fast store. This is external/light-module consolidation,
not base-VLM weight learning. The paper gives no executable code, complete
training hyperparameters, trigger threshold, total slow-store cap, or
deletion/rollback protocol. It should be replicated as a
`surprise admission × replay mixture` experiment, not adopted as a learned
production scheduler.

### 4.10 From wake-time skill editing to deferred knowledge compilation

Three similarly named but operationally different mechanisms sharpen the
training boundary.

[Memento 2](https://arxiv.org/abs/2512.22716) updates external cases, a Parzen
policy, and \(Q\) after task feedback while a frozen language model remains
fixed. [Memento-Skills](https://arxiv.org/abs/2603.18743) judges a failed
attempt with the current question's ground truth, rewrites a Markdown/code
skill, and retries that same question. Both learn persistent external control
state, but neither defers transformation beyond the current task. Their data
belongs in a **wake-learning baseline**, not a sleep-training arm.

Memento-Skills separately offers a useful predeployment router recipe:

```text
8K public skill descriptions
→ sample approximately 3K
→ generate synthetic intent/goal queries
→ judge-filter relevance
→ InfoNCE embedding training
→ held-out skill-retrieval and downstream execution tests
```

This is best described as supervised contrastive contextual-bandit fitting,
despite the paper's “single-step offline RL” label. The actual training
checkpoint, split artifacts, and router code were not released in the closest
paper-time tree, so the recipe is a proposed replication target rather than a
reproducible training baseline.

The later
[Memento-Skills `v0.3.0` DreamDaemon](https://github.com/Memento-Teams/Memento-Skills/tree/71ac933ea1381d53389a2426f59634e0182071b8)
does cross a sleep boundary: a post-response summary enters `_staging.md`; a
quick asynchronous path or periodic daemon asks an LLM to create, update, and
delete topic documents, then rebuilds an index for a later wake. This is
inference-based external consolidation, not the paper's router learning or
base-model training. Its separate file mutations and staging clear should be
replaced in experiments by candidate-manifest creation, validation, atomic
pointer swap, and rollback.

[Evolve](https://arxiv.org/abs/2604.23424) makes the cold--warm--sleep--post
cycle explicit. A Qwen3.5 2B model queries staging/canonical section memory;
GLM-4.7 acquires misses and compiles same-category overlaps at cosine
similarity at least `.85`; an optional 03:00 scheduler runs the consolidation
outside the query path. The reported sleep pass shrinks canonical section
counts by about 31–34% on three 250-question benchmarks, but the released
answer CSVs do not include section snapshots or compile logs needed to
independently recompute those counts.

The evaluation corpus, however, repeats exactly the same 250 questions in the
same order during cold, warm, and post-sleep phases. Its lower teacher-call
rate is therefore a **repeated-key amortization** result, not evidence that a
compiled section answers unseen related questions. Cold-to-post score changes
are only `+1.0/-0.6/+1.2 pp` for custom/NQ/TriviaQA, with no parallel no-sleep
control; they do not identify a causal accuracy benefit from consolidation.
The reported teacher-call counter also excludes provider retries, and Evolve's
latency timer performs both suppress and augment generations versus one
baseline generation. A sleep-data benchmark
must split:

```text
acquisition query
same-key cache probe
held-out paraphrase
held-out compositional query
contradiction/correction query
unrelated negative-control query
```

and must release the acquired sections, source spans, candidate/replacement
lineage, compile logs, vector/metadata snapshots, and item-level paired
outcomes. Without them, answer CSVs can verify output scoring but not the
claimed knowledge lifecycle.

### 4.11 Functional neural state, reconstructive narrative, and learned calls

Recent work uses “consolidation” for three distinct training objects.

**Mela** learns a hierarchical functional neural-memory update rule during
pretraining. At test time, surprise-driven fast and slow modules update inside
the current sequence and reconstruct multi-granularity features. This is a
useful wake/test-time parametric-state arm, but the released wrapper does not
persist the functional state across calls. Its training corpus and perplexity
objective teach an update architecture; they do not construct a deployed
sleep corpus.

**MIRROR** uses no gradient training for consolidation. After each response,
inference-only worker threads update Goals, Reasoning, and Memory and a
Controller fully regenerates a bounded narrative. This is reconstructive
external transformation. A faithful baseline must attach a monotonically
increasing turn generation to the snapshot and publish only if:

\[
g_{\text{candidate}} >
g_{\text{published}}
\quad\land\quad
v_{\text{parent}}=v_{\text{expected}}.
\]

Without this conditional publication, an older slow worker can finish after a
newer one and regress memory state even when every individual synthesis is
valid.

**HeLa-Mem** proposes a hub-triggered reflective distillation from episodic
graph to semantic memory. Its released evaluator also performs periodic LLM
fact extraction independent of hub reflection, while decay and adaptive
forgetting are not called. The benchmark must therefore separate:

```text
embedding + temporal graph only
periodic extraction only
Hebbian edge reinforcement only
hub-triggered reflection only
all components with active decay/forgetting
```

Each arm needs multiple later queries per constructed graph; reinforcement
after the sole query cannot demonstrate a learning effect.

**LMLM and LaCy** learn a destination decision during pretraining rather than
sleep. LMLM annotates factual spans with database calls and masks returned
values from the language-model loss. LaCy uses token loss plus lightweight
grammatical factuality to replace selected factual targets with `<CALL>`.
They motivate a sleep-data routing target:

```text
<KEEP_RAW> <WRITE_TEXT> <WRITE_GRAPH> <CALL_EXTERNAL>
<COMPILE_ADAPTER> <COMPILE_SHARED> <DELETE_OR_NOOP>
```

The label cannot be generated from current token loss alone. It must reflect
future reuse, downstream generalization, retrieval failure, volatility,
delete fan-out, interference, and lifecycle cost. LaCy's null NLU result under
factual offloading is a required negative control: fewer factual targets in
weights do not automatically create usable reasoning capacity.

### 4.12 Internalization curricula, module routing, and temporal transforms

The final 2026 audit adds four training objects that must be separated.

**Capability-complete internalization data.** ImprintBench-style streams
require examples and future-query holdouts for:

```text
direct acquisition
temporal replacement
reference resolution
cross-memory composition
implicit relevance
boundary awareness / justified abstention
```

A sleep corpus made only from source→answer pairs can train direct recall while
leaving update, composition, and scope behavior unmeasured. Each source event
should therefore generate a grouped probe family, with the entire family kept
on one side of the train/holdout boundary to prevent paraphrase leakage.

**Document-to-adapter curricula.** LoRA-as-memory experiments show that raw
next-token data, QA pairs, summaries, rewrites, and mixtures yield different
usable memories from the same document. The parametric arm should cross:

```text
source transform
  ∈ {raw, QA, summary, rewrite, mixed}

module topology
  ∈ {single adapter, isolated adapters, routed top-1, routed top-k}

composition
  ∈ {none, linear, scaled concatenation, TIES}
```

Ranks and total trainable parameters must be matched. An oracle router is a
diagnostic upper bound, not a deployable result. Router training needs
document/query negatives, cross-document composition queries, out-of-catalog
queries, and correction/version pairs; merge training or selection needs
sign-conflict and incompatible-module stress cases.

**Feedback-to-memory data.** MemoryBench-style explicit verbal/action and
implicit feedback belongs in the wake baseline. The same feedback stream can
then be staged for a sleep arm, but the future query and reward must be
identical. This produces a clean:

```text
immediate wake edit
vs staged deterministic transform
vs staged learned transform
vs no write
```

comparison rather than confounding phase with evidence quality.

**Multi-timescale external transforms.** TiMem, RecMem, GAM, and MEMORA provide
four distinct recipes:

- close temporal windows into session/day/week/month views;
- delay expensive synthesis until semantic similarity and recurrence cross a
  threshold;
- flush a bounded event-progression buffer at pause/end/overflow into a topic
  graph while retaining raw evidence;
- edit embodied entity/action state online, then abstract cross-video
  habits, workflows, and preferences offline.

These recipes require negative controls that the source papers do not jointly
provide:

```text
one clock vs multiple clocks
recurrence-only vs surprise/utility/volatility admission
bounded live buffer vs bounded total state
local region maintenance vs global rebuild
raw archive retained vs removed vs tiered
text-only vs multimodal provenance/delete propagation
```

The raw/reversible tier must stay available for correction, rare one-off
events, and causal evaluation. Reducing construction tokens or recalled
context is not evidence that total information or lifetime work was reduced.

## 5. Reinforcement learning for sleep

RL appears at four different control levels and should not be reported as one
method.

### 5.1 Data-generation policy

[SEAL](https://arxiv.org/abs/2506.10943) trains the model to emit a self-edit:
synthetic data plus, optionally, optimization directives. SFT applies the
self-edit to the model; downstream post-update performance becomes reward for
the self-edit policy. This establishes learned data construction, not an
always-safe continual update. Reward evaluation can be expensive and
benchmark-specific, and the evaluated adaptation horizon is much shorter than
a lifetime wake--sleep stream.

Google Sleep's dreaming stage generates candidate questions from accumulated
context/task state, introduces an otherwise irrelevant expert to encourage
novel combinations, and filters candidates by whether an SFT gradient predicts
usefulness. Downstream improvement supplies a binary reward, and the paper's
ReSTEM loop trains the generator. This is the
closest audited parametric NREM/REM-like pairing, but the biological names are
analogies and random expert injection is not evidence about REM physiology.

### 5.2 Replay-selection policy

[Spens, Burgess, and Behrens](https://papers.nips.cc/paper_files/paper/2025/hash/d7e5870810331da5a8ac8bd16d42e074-Abstract-Conference.html)
use a recurrent PPO meta-controller whose sleep action can replay recent
memories, train a world model, learn from generated data, or do nothing,
depending on the experiment. Future wake reward trains the controller. Some
branches also estimate episode value and combine it with diversity.

This preempts claims that learned sleep scheduling, action selection, or replay
valuation begins with LLM agents. It does not establish one universal
controller: the action spaces and value mechanisms differ across image,
maze, and relational tasks.

#### MSSR: adaptive LLM replay without a sleep boundary

[MSSR](https://arxiv.org/abs/2603.09892), first public 2026-03-10, maintains a
loss-updated memory-strength and stability scalar per buffered sample. Its
schedule expands replay intervals as modeled stability grows, decays the replay
ratio toward a floor, and samples low-retention items more often; selected
replay and current samples jointly update LoRA weights. It evaluates three
backbones and a sequence of up to 11 tasks. The paper reports roughly `3–5%`
normalized wall-clock and `4–6%` peak-memory overhead relative to fixed replay
for its 7B setting.

This is a close comparator for pressure-aware cadence and sample selection, but
not strict STC: loss observation, scalar refresh, selection, and LoRA update
remain inside the continual fine-tuning loop. The interval rule is an engineered
memory-dynamics schedule, not a learned off-path scheduler, and no
before-versus-after recoverability-boundary estimand, versioned publication,
causal deletion, rollback, or multi-tenant queue is tested.

#### H-EPM: online external memory plus memory-guided RL

[H-EPM](https://arxiv.org/abs/2512.07287), first public 2025-12-08 and
published at ICML 2026, separates two relevant operators.

1. In its ACEBench no-training-set condition, the paper says an LLM judge
   identifies successful test trajectories and memory is updated online. It
   does not state that task output is finalized before insertion begins or
   provide an ordering algorithm. The repository has no ACEBench path; its
   closest ToolSandbox online writer updates after tool-call segments within
   execution. This is `C1=Y,C2=P,C3=P`, not a strict PASS.
2. In its parametric experiment, the graph guides GRPO exploration while new
   successful rollouts update the graph. One epoch uses batch size `8`, eight
   rollouts per task, terminal success reward, KL coefficient `.001`, and masks
   memory-guidance tokens from the policy loss. On \(\tau^2\)-Bench the reported
   result is `0.223`, versus `0.178` for ordinary GRPO and `0.158` for the base
   policy.

This is a concrete bridge between external consolidated experience and
parametric post-training, but not yet an alternating deployed wake/sleep
trainer. Initial memory construction itself takes three rollouts per task and
keeps the shortest success; about `2.2K` tasks require roughly `15 h` on one
A100. Each repeated graph edge can accumulate multiple state summaries, with no
merge, cap, expiry, causal delete, or rollback mechanism.

### 5.3 Consolidation-operator policy

[Auto-Dreamer](https://arxiv.org/abs/2605.20616) applies GRPO to an external
memory consolidator. The policy reads a bounded region and provenance-linked
source trajectories, uses tools under a budget, and proposes a fresh
replacement set. End-to-end later agent performance is the reward. Its
cross-domain results make it a stronger direct precedent for learned external
sleep than a prompt-only summarizer. The selected working region and tool
rollout are bounded, and the active bank is empirically compact; the method
does not establish a hard global lifetime cap, while provenance-linked source
trajectories remain a persistent archive cost.

TrustMem learns a policy over memory transition candidates and evaluates
coverage, preservation, and faithfulness. It is most relevant to the promotion
gate, although a learned prune action and bloat penalty are not a proof of
bounded lifetime storage or correct physical deletion.

### 5.4 Retrieval and delivery policy

MEMORA's optional GRPO policy iteratively chooses `REFINE`, `EXPAND`, or `STOP`
during retrieval. This can improve how stored memory is used, but it is a
wake/read policy, not evidence that the memory write/consolidation phase is
sleep-time. Similarly, query-time organization or compression should not be
counted as an offline scheduler.

[MemCompiler](https://arxiv.org/abs/2605.07594) shows that the delivery policy
itself can be learned across text and latent channels. A teacher first
traverses training tasks with a memory of successful and failed trajectories;
successful traces yield compiler and executor SFT examples. A
Qwen2.5-VL-7B compiler is then LoRA-tuned for five epochs, followed by
`500`-step GRPO that updates only the compiler and Soft-Mem projection with
binary episode-success reward while the executor remains frozen. At deployment,
however, the compiler consumes observation, mutable Brief State, and read-only
episode memory at **every step** and immediately emits guidance plus sixteen
soft tokens. The result is `Q/W`, not sleep. Its folded ten-item Brief State
and fixed soft-token count bound current compiled state, not the underlying
trajectory bank.

### 5.5 Reward design

A deployable sleep reward is delayed, partially observed, and confounded by
future workload. A one-step benchmark reward can select transformations that
erase rare or long-horizon utility. The benchmark should therefore compare:

\[
R =
w_u U_{\text{future}}
- w_f F_{\text{protected}}
- w_h H_{\text{false-memory}}
- w_c C_{\text{lifecycle}}
- w_g G_{\text{governance-debt}}.
\]

This is a proposed reward interface, not a result from the cited papers.
Every component is normalized by a frozen scale and each \(w_\cdot\) maps it
into one declared reward unit; otherwise the expression is only a vector of
objectives. Weights and thresholds must be frozen before confirmatory evaluation.
Counterfactual evaluation requires logged propensities or a simulator; absent
that support, off-policy estimates are exploratory.

## 6. Cadence, trigger, and data freshness

Audited triggers include:

| Trigger | Work | What it establishes | What it does not |
|---|---|---|---|
| short-term slots full | DGDMN | capacity-triggered transfer | optimal trigger |
| every 10 class sessions | FearNet | cadence trade-off | task-free deployment |
| task boundary | P&C, WSCL | stable boundary consolidation | idle-time scheduler |
| every 120K samples | SIESTA | compute-limited periodic sleep | pressure adaptation |
| midpoint of each task | PCMC | controlled before/after comparison | production cadence |
| completed task | PlugMem WebArena | post-task external abstraction and later reuse | autonomous periodic or pressure-aware scheduler |
| same-task feedback | Memento 2; Memento-Skills paper | persistent wake learning and later reuse | deferred sleep; current-task independence |
| claimed successful test trajectory | H-EPM ACEBench | motivates outcome-gated online graph evolution | post-task start, later-only reuse, or deployed async ownership |
| fixed sleep-action count | Spens image/maze | learned operator within a bounded window | when to open the window |
| modeled memory strength / expanding intervals | MSSR | adaptive replay cadence inside continual LLM fine-tuning | separate off-path service or recoverability-boundary causality |
| selected working region | Auto-Dreamer | per-job bounded transactional transform | global active-store or backlog boundedness |
| multi-frequency transfer schedule | Google Sleep | fast-to-slow hierarchy | pool exhaustion policy |
| default six hours | HIMA architecture | explicit batch lifecycle | empirically superior six-hour cadence |
| post-response quick path; 600-second scan; 24-hour gate | Memento-Skills DreamDaemon | executable deferred topic-memory consolidation | tested cadence optimum, durable queue, or atomic publication |
| manual or default 03:00, staging count ≥1 | Evolve | explicit scheduler ownership and later persistent reuse | concurrent-query isolation, adaptive pressure trigger, or queue stability |

Clock time alone cannot be the research conclusion. A candidate trigger should
use a pressure vector:

```text
novelty arrival
duplicate/rewrite rate
retrieval competition
interference and acquisition loss
free slot/expert fraction
staleness and correction rate
sleep backlog
privacy/delete deadline
idle-window and accelerator availability
```

The strict queue condition is a necessary, not sufficient, bound:

\[
\lambda_{\text{admitted}}\,
\mathbb E[F_{\text{verify+transform+publish+recover}}]
<
G_{\text{sleep}}.
\]

Here \(\lambda_{\text{admitted}}\) is event/s, \(F\) is FLOP/event including
verification, transformation, publication, and recovery work, and
\(G_{\text{sleep}}\) is effective aggregate FLOP/s. The inequality is
dimensionally a necessary offered-load condition; batching, priority, retries,
and multi-resource bottlenecks require stronger tests.

If it fails, the memory system is unstable even if every individual sleep
update is accurate.

## 7. Failure and capacity audit by training method

| Method | Capacity moved into | Characteristic failure | Required control |
|---|---|---|---|
| raw replay | archive bytes and I/O | privacy/storage/backlog | TTL, tiering, stratified coreset |
| latent replay | latent bytes and encoder compatibility | stale/unreadable latents | encoder digest, re-embed/rebuild |
| generated replay | generator/teacher state and generation compute | compounding hallucination or mode loss | canonical anchors, frozen teacher, cycle-depth audit |
| distillation | teacher and student capacity | unreachable behavior, teacher drift | hard-label anchors, KL audit, future-query probes |
| regularisation | reference/importance state and protected directions | loss of plasticity | acquisition probe, recycle/reset path |
| gradient projection | exemplars and constraint solve | compute growth, conflicting constraints | bounded coreset, infeasibility policy |
| sparse isolation/growth | free slots, experts, masks, routing | pool exhaustion and routing collision | free-fraction threshold, merge/retire/rebuild |
| contrastive abstraction | raw source and index versions | semantic collision, exact-detail loss | source-linked rollback, query-type tests |
| RL data/operator policy | reward model, rollout data, exploration budget | reward hacking and short-horizon bias | held-out reward, negative controls, OPE limits |
| external merge/rewrite | archive, graph, lineage, verifier calls, and refreshed retrieval keys | destructive context collapse or key--payload drift | immutable source, generation-bound payload+key, versioned view, no-force baseline |
| memory-program search | candidate code, repeated re-ingestion, validation budget | benchmark heuristics and expensive re-evaluation | held-out transfer, runtime checks, store-level governance contract |
| retrospective harness search | trajectories, diagnoses, candidate files/tools, versions, and rank calls | correlated self-judge, executable poisoning, unbounded procedural growth | independent canary, sandboxed capabilities, artifact cap, provenance, atomic promotion, and exact rollback |
| surprise-gated replay | retained embedding/prototype cores and replay compute | threshold drift, rare-known misclassification, recent-window displacement, cumulative full replay | calibrated OOD holdout, no-replay/full/recent/coreset controls, class-open stream, complete replay-state and lifetime-work accounting |
| learned context compaction | summary text, block-conditioned KV, compactor weights | restart loss, latent leakage, long-generation memory-time increase | stateful-versus-restart ablation, KV probes, peak plus AUC accounting, source-linked privacy tests |
| wake-time skill/case editing | skill catalog, cases, controller state, retry budget | same-task label leakage, catalog-wide prompt growth, unbounded procedural variants | strict later-query split, fixed artifact budget, dedup/version/delete policy, cost-inclusive baseline |
| staged topic/section compilation | canonical files, vector index, metadata, expired state, compile logs | repeated-key overclaim, cross-store divergence, partial staging clear, orphaned expiry | held-out semantic probes, immutable source lineage, journaled manifest, atomic pointer swap, crash/rebuild test |
| within-call functional neural memory | learned update rule, current-call state, KV | phase-name overclaim, no cross-call persistence, opaque information capacity | explicit state handoff/restart test, cross-session probe, all-state byte accounting |
| reconstructive bounded narrative | prior narrative, thread outputs, per-turn LLM work | cumulative rewrite error, stale worker overwrite, later-wake blocking | generation-tagged snapshot/CAS publication, rapid-turn stress, no-rewrite/raw-history controls |
| episodic-to-semantic graph reflection | graph nodes/edges, extracted facts, reflection calls | inactive decay/forgetting, current-query-only reinforcement, unbounded facts | call-path instrumentation, multi-query graph, active mechanism ablations, hard total cap |
| learned external-call routing | call targets, external DB/LLM, annotator/parser state | delegation overuse, lookup outage, domain-specific labels, assumed freed capacity | call-cost/reliability sweep, NLU null control, opaque-bit accounting, dynamic rerouting |
| modular LoRA memory | adapter ranks/count, router, merge workspace, GPU cache, source lineage | rank knee, catalog error, incompatible merges, serving miss, undeletable derived weights | matched total parameters, oracle/learned routing gap, top-k merge ablation, versioned adapter registry, correction/unlearning trial |
| temporal/recurrence graph consolidation | raw store, hierarchy levels, graphs, summaries, scheduled calls | clock-amplified service load, rare one-off suppression, bounded-live/unbounded-archive illusion | call/byte accounting by level, rare-event holdout, one-clock control, archive cap/tiering, local-versus-global maintenance |
| embodied offline consolidation | video/action evidence, entity histories, habits, workflows, preferences | temporal leakage, identity/consent propagation, stale physical state, multimodal erase gaps | time-anchored snapshots, participant isolation, evidence-level consent and tombstones, derived-view rebuild, OOD planning holdout |

Four comparisons are mandatory.

1. **Retention versus plasticity.** Old-task accuracy can pass while
   fixed-budget acquisition has already collapsed.
2. **Active versus total state.** A compact prompt or index can hide a growing
   archive, checkpoint cache, graph, or expert pool.
3. **Per-cycle versus lifetime compute.** Replaying all prior tasks or
   redistilling all prior facts produces quadratic cumulative work.
4. **Behavioral forgetting versus physical deletion.** Failure to retrieve is
   not proof that source information was erased from weights, summaries,
   embeddings, checkpoints, or backups.

## 8. Minimum sleep-data manifest

Every sleep corpus release should freeze:

```text
wake snapshot and event-range digest
source/privacy/tenant scope
admission, sampling, dedup, and exclusion policy
canonical/latent/generated item counts and bytes
generator, teacher, verifier, tokenizer, prompt, and decoding versions
label source and uncertainty
old/new/rare/failure/correction mixture
augmentation and counterfactual operators
per-item provenance and delete dependency
train/validation/future-query holdout split
poison, contradiction, and impossible-dream filters
optimizer, objective weights, cadence, and compute budget
candidate destination and rollback artifact
```

Generated data may not be promoted to independent evidence for the same claim
that generated it. Teacher, generator, and judge sharing a model family must be
reported as correlated evidence, not three independent validators.

## 9. Matched benchmark matrix

The central experiment should give every method the same frozen wake stream,
future queries, canonical byte budget, and lifetime resource envelope.

### 9.1 Data arms

```text
D0  current wake data only
D1  current + uniform raw replay
D2  current + stratified/utility coreset
D3  current + latent replay
D4  current + generator-only replay
D5  current + generator replay + small canonical anchors
D6  D5 + grounded counterfactual/dream data
D7  D6 + failure/correction/adversarial mixture
D8  grouped internalization curriculum: acquire/update/reference/compose/scope
D9  explicit-action/verbal + implicit feedback stream
D10 multimodal entity/action evidence with temporal and consent anchors
```

### 9.2 Operator arms

```text
O0  no sleep
O1  deterministic merge/summary
O2  supervised replay
O3  distillation
O4  regularised shared-weight update
O5  sparse adapter/expert update
O6  external learned consolidator
O7  learned router across external/latent/parametric/no-op
O8  single vs isolated/routed/merged LoRA knowledge modules
O9  temporal-hierarchical or recurrence-gated graph consolidation
```

### 9.3 Cadence arms

```text
T0  immediate online update
T1  fixed event count
T2  fixed wall clock
T3  capacity/queue pressure
T4  novelty/surprise/interference
T5  learned trigger
T6  nested temporal closure: session/day/week/month
```

The full Cartesian product is unnecessary and wasteful. A blinded pilot should
screen dominated arms; the confirmatory set must retain at least:

- a strong raw-replay baseline;
- a generator-only failure control;
- a small-anchor hybrid;
- external-only, parametric-only, and hybrid destinations;
- immediate wake update and delayed sleep at equal total compute;
- fixed cadence and pressure trigger;
- incumbent-candidate canary, rollback, and delete trials.

### 9.4 Endpoints

Primary endpoints:

```text
held-out lifetime utility
protected-slice retention
fresh-normalized fixed-budget acquisition gain
lifecycle cost and bytes moved
delete completeness and rollback recovery
```

Secondary diagnostics:

```text
exact/semantic/temporal/relational/procedural recall
acquisition/update/reference/composition/relevance/boundary-awareness profile
writer/retriever/reader error
destination-router and within-destination router/merge error
false memory and calibration
representation rank, dormant units, gradient conflict
active/total/archive/catalog/graph/auxiliary bytes
free/recyclable expert or slot fraction
sleep backlog, per-clock calls, and service utilization
promotion latency and wake P99 interference
```

## 10. Research conclusions supported now

1. Replay, generated replay, distillation, parameter isolation, sleep
   augmentation, and learned offline control all have direct pre-2025
   precedents. Novelty cannot rest on inventing any one of them.
2. No single audited training method jointly solves future-query utility,
   bounded lifetime capacity, plasticity, correction, deletion, and rollback.
3. Generated replay trades raw storage for teacher/generator state, compute,
   and error debt; it is not memory-free.
4. External consolidation and parametric consolidation are complementary
   destinations. External memory is the safer canonical evidence layer;
   adapter/weight state is best treated as a high-reuse compiled cache until
   stronger unlearning and reachability evidence exists.
5. The most important missing learner is not another summarizer. It is a
   lifecycle router that chooses data, operator, destination, cadence, and
   deliberate forgetting under a joint utility/cost/governance constraint.
6. The most important missing early warning is acquisition loss before old
   retention loss. This is the basis of `H-STC-007`.
7. Test-time neural state, asynchronous reconstructive state, reflective graph
   distillation, and pretraining-time external-call learning are four different
   method classes. Their results cannot be pooled under one “consolidation”
   label.
8. Parametric capacity estimates and factual offloading studies justify a
   controlled cross-medium allocator experiment; they do not establish a
   universal bits/parameter law or automatic reasoning-capacity gain.
9. A parametric adapter pool does not eliminate retrieval. Router error,
   merge interference, catalog growth, and serving residency determine how
   much stored adapter content is usable.
10. Recurrence and temporal closure are useful admission/cadence signals, but
    neither bounds raw archives, derived graphs, or cumulative construction
    work.
11. Direct recall is an incomplete internalization target. Update,
    composition, relevance, and boundary-awareness curricula and holdouts are
    required before claiming that sleep data entered a model.

The paper-level opportunity is therefore a matched, transactional,
cross-substrate sleep compiler—not a claim that “models need sleep” by analogy.
