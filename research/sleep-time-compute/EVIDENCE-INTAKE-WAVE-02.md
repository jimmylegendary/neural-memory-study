# Sleep-Time Compute Evidence Intake — Wave 02

- Intake date: 2026-07-25
- Scope: strict wake/sleep precedents, 2026-07 update, negative evidence, and
  systems/capacity implications
- Status: intake-local IDs only; canonical `SRC-STC-*`, `CL-STC-*`, and
  `EV-STC-*` IDs are allocated after Wave 01 ingestion
- Evidence rule: a paper's name, biological metaphor, or offline-trained
  component does not determine its phase. The actual deployed execution point
  does.

## 1. Operational phase test

A method is **strict sleep-time compute** only when it satisfies all three
conditions:

1. it runs outside the current wake action's causal critical path;
2. it transforms accumulated lifetime experience rather than merely processing
   a static predeployment corpus;
3. it publishes durable state that is reused in a later wake phase.

The following labels are used in this intake:

| Label | Meaning |
|---|---|
| `Q` | query-time action or within-query context/KV transformation |
| `W` | live wake-time or per-step online learning |
| `B` | task, write, or overflow boundary; sleep-adjacent but not autonomous idle compute |
| `S` | query-independent background/offline durable transformation |
| `P` | proposal, theory, or benchmark rather than an implemented lifecycle operator |

An offline-trained router used at query time is `Q`, not `S`. A one-shot
predeployment distillation is not recurring deployed sleep. A prompt budget is
not a persistent-memory capacity bound.

## 2. Foundational precedents promoted to the core

| Earliest public date | Source and status | Actual wake/sleep mechanism | Capacity and boundary |
|---:|---|---|---|
| 2017-10-28 | Kamra, Gupta & Liu, [Deep Generative Dual Memory Network](https://arxiv.org/abs/1710.10368), ICLR 2018 submission/preprint | Task-specific short-term generative memories absorb wake data. When the STM set is exhausted, they generate samples into an LTM, which consolidates them with deep generative replay. | Uses a finite generated-sample budget \(N_{\max}\) and deliberately allocates a minimum fraction to new data, producing gradual forgetting. Small image tasks and task descriptors; no verified ICLR acceptance, deletion lineage, or rollback. |
| 2017-11-28 | Kemker & Kanan, [FearNet](https://openreview.net/forum?id=SJ1Xmf-Rb), ICLR 2018 | Recent exemplars enter an HC-like store. By default every ten study sessions, autoencoder pseudo-examples from old classes are mixed with HC data to fine-tune an mPFC-like network for 60 epochs; HC is then cleared. | Strict periodic parametric sleep. Uses pretrained image/audio embeddings and class-incremental sessions. Stores per-class Gaussian statistics; deletion means clearing recent exemplars, not causal erasure from weights. |
| 2018-05-16 | Schwarz et al., [Progress & Compress](https://proceedings.mlr.press/v80/schwarz18a.html), ICML 2018 | An active column learns the current task; a compression phase distils it into a fixed-size knowledge base while online EWC protects prior skills. | Constant parameter count, no old-task data, and no task-specific parameters, but requires task boundaries and current-task inputs. Its decay factor permits graceful forgetting but gives no item-level delete/rollback. |
| 2019-02-01 | Kaplanis, Shanahan & Clopath, [Policy Consolidation](https://proceedings.mlr.press/v97/kaplanis19a.html), ICML 2019 | A cascade of hidden policies evolves at multiple timescales and regularises the current policy with bidirectional KL distillation. | `W`, not `S`: consolidation runs continuously at policy updates and requires no task boundaries. It is a multi-timescale stability prior, not an offline sleep scheduler. |
| 2023-03-19 | Harun et al., [SIESTA](https://openreview.net/forum?id=MqDVlBWRRV), TMLR 2023 | Wake uses no rehearsal or backpropagation: a frozen lower encoder produces product-quantised features, a bounded buffer stores them, and output class means update online. Sleep uses balanced reconstructed latent replay to backpropagate through the upper network. | Strict wake/sleep precursor. ImageNet-1K without augmentation: final top-5 83.59 versus offline 83.31, 2.02 GB replay store, 1.9 h on one A5000. The lower representation is assumed reusable and frozen. |
| 2023-12-06 | Sorrenti et al., [Wake-Sleep Consolidated Learning](https://arxiv.org/abs/2401.08623), IEEE TNNLS 2025 | One wake epoch uses dynamic parameter freezing and STM. Ten sleep epochs alternate NREM replay from STM/LTM and REM exposure to a disjoint dream dataset. | REM in the main experiment is external, unseen, labelled image data—not a model-generated dream. Vision benchmarks and task boundaries only. The previous dossier attribution to “Jie et al.” was wrong. |
| 2024-09-24 | Taylor, Vassiliades & Dovrolis, [PCMC](https://proceedings.mlr.press/v274/taylor25a.html), CoLLAs 2024 / PMLR 2025 | Wake clusters patch embeddings into fixed STM and promoted LTM centroids. Periodic sleep retrains the encoder for 300 epochs using stored raw patches, re-embeds centroids, and prunes redundant memory. | On ImageNet40, the reported configuration reaches 59.7±1.34 with maximum LTM equivalent to 8,819 images versus 60.0±1.21 and 12,715 without consolidation. Strong direct precursor, but sleep is expensive and scope is 40-class vision. |
| 2025-09-18 | Spens, Burgess & Behrens, [Modelling the Control of Offline Processing with Reinforcement Learning](https://papers.nips.cc/paper_files/paper/2025/hash/d7e5870810331da5a8ac8bd16d42e074-Abstract-Conference.html), NeurIPS 2025 | A recurrent PPO meta-controller chooses from a different offline action set in each toy task. In image and maze only, a value estimator learns from Shapley-style marginal utility and deterministic MMR then reranks for diversity; neither applies to the relational task. | Direct novelty collision: adaptive sleep curricula already exist, and learned replay selection exists in two setups. Only the relational world model resets each episode; the image experiment assumes a validation set. No LLM, cross-medium lifetime state, deletion, rollback, or production scheduler. |

## 3. 2026 evidence that changes the thesis

### 3.1 Negative evidence for irreversible consolidation

[Useful Memories Become Faulty When Continuously Updated by
LLMs](https://arxiv.org/abs/2605.12978) reports non-monotone utility under
repeated natural-language memory rewriting:

- in WebShop, AWM rises to 0.64 at eight examples and falls to 0.20 at 128;
- GPT-5.4 solves a 19-item ARC slice without memory, yet streamed
  consolidation from ground-truth solutions reduces solved-item performance to
  roughly 53–54%;
- retaining raw episodes is competitive with or better than forced
  consolidation in the reported ARC-AGI Stream regimes.

This is counterevidence to always-consolidate, not evidence that every
abstraction is harmful. It supports treating raw episodic records as the
canonical truth and text/vector/graph/latent/parametric memories as versioned,
rebuildable views.

### 3.2 Mechanistic external fast/slow memory

[Memini](https://arxiv.org/abs/2605.05097) places coupled fast/slow
Benna–Fusi dynamics on each edge of an external directed memory graph. Repeated
old pairs retain higher fast weight than a single-timescale baseline in a
124-event Wikipedia-history toy stream. This is an online mechanistic prior,
not sleep: nodes and edges are not physically reclaimed, and retrieval quality,
delete propagation, provenance, and rollback are not evaluated.

### 3.3 Low-evidence explicit sleep prototype

[Sleep-Consolidated Memory](https://arxiv.org/abs/2604.20943) is a single-author
research preview using a NetworkX/SQLite semantic graph, Llama 3.2 extraction,
NREM co-occurrence strengthening, REM random walks, and threshold pruning.
Its ten-turn synthetic fact benchmark, fixed-seed zero variance, tiny
360-concept latency test, unavailable code, and internally inconsistent
time-decay equations make it directional evidence only.

### 3.4 KV micro-cycle adjacency

[SleepGate](https://arxiv.org/abs/2603.14517) trains a small synthetic
transformer with conflict tags and an attention gate. The reported experiment
uses soft attention biasing; cache entries are not actually evicted or
compressed. Accuracy degrades from 99.5% at interference depth five to 16.5%
at depth thirty. The paper's `793,344` count is the base transformer; the full
SleepGate model has `917,313` parameters. It is a query-time/inference
micro-cycle, not demonstrated cross-session sleep or a measured
capacity-reduction system.

### 3.5 Surprise-gated replay

[Surprise as a Signal for Plasticity and
Metacognition](https://arxiv.org/abs/2606.31495) is a single-author research
note with two strict proof-of-concept sleep paths. A frozen-encoder
prediction-error signal gates episodic writes; periodic full interleaved
replay updates a slow linear readout after completed image tasks. In a second
system, a manual or fast-store-full trigger migrates one-shot image/text facts
into slow prototype/familiarity state, clears the fast store, and supports
later recall after dialogue reset. Both satisfy `C1=C2=C3=Y`, but neither
updates the base encoder/VLM.

The strongest result is paired with a warning: full replay raises oldest-task
retention from `65.8→83.5` on DINOv2 and `25.9→77.2` on I-JEPA, whereas
replaying only the latest five tasks falls to `41.2` and `0.0`. Surprise-gated
admission halves one buffer with `84.2` versus full replay `83.7`, but these
are single-seed runs. The note releases no code or complete optimizer,
trigger, buffer, or lifecycle settings; full-history replay and classwise
prototypes grow with the stream, and no deletion, provenance, or rollback
contract is evaluated.

### 3.6 External knowledge and skill lifecycles

Three April-era artifacts separate persistent learning from actual sleep.

- [Memento 2](https://arxiv.org/abs/2512.22716) keeps the LLM frozen and
  updates cases, a Parzen policy, and \(Q\) in the same task's
  read--act--feedback--write loop. Its finite-memory convergence assumptions
  and value-error bound are useful theory, but the operational phase is `W`,
  not `S`, and no fixed-budget mechanism reconciles bounded memory with dense
  coverage.
- [Memento-Skills](https://arxiv.org/abs/2603.18743) uses a
  ground-truth-assisted judge to rewrite external Markdown/code skills and
  retry the same question. The paper loop is `W`. A separate repository
  release added `DreamDaemon` 34 days later: post-response summaries enter
  staging, and a quick async path or periodic daemon compiles topic Markdown
  for later wakes. That code path is `S`, but it has no Dream experiment,
  transaction journal, hard cap, provenance, or rollback.
- [Evolve](https://arxiv.org/abs/2604.23424) provides a strict
  cold--warm--sleep--post section-memory cycle and an optional 03:00 scheduler.
  Its reported canonical stores shrink by about 31–34% after one pass, but
  cold, warm, and post phases repeat the same 250 questions in the same order.
  Cold-to-post score moves only `+1.0/-0.6/+1.2 pp` on
  custom/NQ/TriviaQA and has no no-sleep control. The released artifacts
  therefore establish repeated-query amortization, not unseen-query transfer
  or a causal sleep-accuracy gain; they also omit the stores/logs needed to
  recompute the reported compaction.

Evolve's vector store and SQLite metadata are updated in separate operations,
the vector snapshot persists only at normal shutdown, and there is no global
foreground-query lock or cross-store journal. Memento-Skills
similarly mutates topic files, index, and staging separately. Both motivate a
hard publication requirement: “sleep completed” means a validated,
content-addressed manifest became atomically visible—not merely that each
individual storage API returned successfully.

### 3.7 Reconstructive state and capacity-allocation boundaries

Three further systems sharpen the phase boundary.

- [Mela](https://arxiv.org/abs/2605.10537) applies fast/slow functional neural
  memory updates within a sequence. The released model does not return or
  resupply that state across calls; it therefore belongs to `Q`, despite using
  “test-time consolidation” terminology.
- [MIRROR](https://openreview.net/forum?id=IviO4bIZc7) is a direct `S`
  external-state example: after a Talker responds, asynchronous worker threads
  and a Controller regenerate a bounded narrative for the next turn. Its
  `O(1)` claim bounds state/per-turn work rather than lifetime work. The
  released concurrency path lacks turn-generation compare-and-swap, so an
  older completion can overwrite newer state.
- [HeLa-Mem](https://aclanthology.org/2026.acl-long.625/) proposes reflective
  episodic-graph-to-semantic distillation, but the released LongMemEval default
  does not enable that consolidation and the paper's LoCoMo path is not
  released. Decay/forgetting functions are present but not invoked in the
  evaluated path. It is paper-level mechanism evidence, not a verified
  reflective-sleep reproduction.

Capacity work also changes the interpretation of the parametric/external
axis. Controlled tuple experiments report about two knowledge
bits/parameter, while a different synthetic no-generalization protocol reports
about 3.6 memorized bits/parameter. These are workload- and definition-specific
anchors, not universal constants. LMLM and LaCy train models during
predeployment to delegate factual tokens to external calls. LaCy improves
FactScore in a cascade but does not find a significant NLU gain from freeing
tested factual targets, contradicting the unqualified claim that factual
offload automatically creates usable reasoning capacity.

### 3.8 Last-mile destination, cadence, and total-state evidence

Eight late-discovered works sharpen the thesis without changing its operational
phase test.

- [ImprintBench](https://openreview.net/forum?id=QIJgTW3Qd2) shows that
  continual learning must be evaluated beyond direct recall: acquisition,
  temporal update, reference resolution, composition, implicit relevance, and
  boundary awareness can rank context, auxiliary-parameter, and
  model-parameter methods differently.
- [Understanding LoRA as Knowledge
  Memory](https://arxiv.org/abs/2603.01097) shows a low-rank capacity knee and
  exposes routing and merge interference. On the 64K-token PhoneBook corpus,
  oracle-routed `8×rank-4` modules beat one rank-32 adapter under a matched
  trainable-parameter budget, but a learned router can lose that advantage.
  Parametric modular memory therefore retains a retrieval problem.
- [MemoryBench](https://arxiv.org/abs/2510.17281) supplies an explicit/implicit
  user-feedback simulator over declarative and procedural workloads. It is a
  wake-learning baseline, not a sleep benchmark.
- [Agent-Native Memory](https://arxiv.org/abs/2606.24775) finds no single
  external system dominates and reports large utility/latency differences.
  Localized maintenance must be a scheduler action rather than treating every
  sleep pass as a global rebuild.
- [TiMem](https://aclanthology.org/2026.findings-acl.1091/) implements
  segment/session/day/week/month consolidation, reducing recalled context but
  increasing construction calls by roughly 25–30% in its reported
  comparisons.
- [MEMORA](https://arxiv.org/abs/2607.14252) is a strict experimental
  multimodal pass: online entity/action edits are followed by offline
  participant/video consolidation into habits, workflows, and preferences.
- [RecMem](https://aclanthology.org/2026.findings-acl.1619/) uses recurrence as
  a promotion signal and sharply reduces construction tokens, but relies on an
  unbounded verbatim store; removing that store causes a large accuracy loss.
- [GAM](https://aclanthology.org/2026.acl-long.1600/) bounds its live
  progression buffer at 2,048 tokens while its event/topic graphs and archive
  grow with sessions. It is a direct counterexample to equating bounded hot
  state with bounded total lifetime state.

The shared consequence is a two-stage routing problem: first select a
destination, then retrieve a document/item, adapter/module, or shared-weight
behavior inside that destination. Every capacity result must include the
catalog, graph, archive, recovery, and cumulative rewrite state that the live
bound omits.

## 4. July 2026 cadence matrix

| Date | Source | Phase | Durable destination and decisive boundary |
|---:|---|:---:|---|
| 2026-06-28 | [Discovery by Dreaming](https://arxiv.org/abs/2607.16256) | `S` | Synthetic cross-domain replay/counterfactuals into LoRA. Narrow positive result for Llama-8B rank 256; within-domain and 70B/72B tests are null or inconclusive, and overtraining reverses the gain. |
| 2026-06-30 | [Surprise as a Signal for Plasticity and Metacognition](https://arxiv.org/abs/2606.31495) | `B/S` | Surprise-gated episodic writes are periodically fully replayed into a slow readout; a VLM proof of concept empties fast facts into prototypes. Strict experimental boundary, single-seed core evidence, no code/full settings, and no lifetime cap. |
| 2026-07-01 | [Procedural Memory Distillation](https://arxiv.org/abs/2607.01480) | `W` | External experience/insight/behaviour scaffold into policy weights during a post-training loop. Not deployed idle sleep; deleting the scaffold does not unlearn the weights. |
| 2026-07-02 | [Episodic-to-Semantic Consolidation Without Identity Drift](https://arxiv.org/abs/2607.01988) | `B/S` primitive | Append-only per-identity log into deterministic SQLite fact upserts with checkpoints and provenance. Synthetic rule-based prototype; no cap, decay, RTBF, or selective rollback. |
| 2026-07-04 | [ReCoLoRA](https://arxiv.org/abs/2607.07719) | `B` | Task-boundary SVD separates slow principal, frozen residual, and fresh fast adapter without replay. Six-task, backbone-sensitive evidence; no autonomous sleep. |
| 2026-07-07 | [MemDefrag](https://arxiv.org/abs/2607.05969) | `W/Q` | Persistent latent prefix is reordered per query and permanently pruned on overflow at 12,800 latent tokens. Bounded latent capacity, but no semantic deletion, provenance, privacy, or rollback. |
| 2026-07-08 | [Intrinsic-Noise Consolidation](https://arxiv.org/abs/2607.06924) | `B+W` | Fisher/anchor state is computed at a boundary; a barrier/noise regulariser runs on every later-task update. Not sleep; hardware-in-loop net result is not a demonstrated overall win. |
| 2026-07-09 | [Rate–Distortion View of Memory Compaction](https://arxiv.org/abs/2607.08032) | `P` | Cross-layer rate–distortion taxonomy, lower bound, reversible tiering, and async sleep are proposed. The cross-substrate router, state machine, revocation, rollback, and COMPACT-Bench are not implemented. |
| 2026-07-13 | [Can a Language Model Learn Facts Continually in Its Weights?](https://arxiv.org/abs/2607.11020) | `S` experiment | Every 20 writes, all accumulated facts are distilled into a fresh base copy. At 100 writes retention is 25% versus 28% without consolidation, although capability is protected by 12 points. |
| 2026-07-14 | [MemOps](https://arxiv.org/abs/2607.12893) | `P` | Remember/Forget/Update/Reflect/Trajectory benchmark. Behavioural forgetting is not physical erase or replica deletion. |
| 2026-07-15 | [MemCon](https://arxiv.org/abs/2607.13591) | `W/Q` | A tabular-UCB controller selects memory actions at steps/queries. Raw trajectories remain; hard capacity, RTBF, and rollback are absent. |
| 2026-07-15 | [MEMORA](https://arxiv.org/abs/2607.14252) | `S` experimental | Online edits over egocentric action evidence are consolidated offline at participant/video boundaries into durable habits, workflows, and preferences. Strict multimodal pass; short participant histories, no lifetime cap, consent workflow, or causal erase. |
| 2026-07-15 | [PReM](https://arxiv.org/abs/2607.14327) | `Q` | Within-query KV preservation/refresh from the original 32K prompt. Not persistent long-term memory. |
| 2026-07-16 | [NSTM](https://arxiv.org/abs/2607.15271) | `W` | Live dynamic neural visual state: 1-FPS gradient write and 30-FPS read. An update/apply decoupling analogue, not Google sleep research. |
| 2026-07-20 | [Retain or Consolidate?](https://arxiv.org/abs/2607.17545) | `Q` | Only the ridge utility estimator is trained offline; retain/merge/abstract/rewrite happens per query. Tight-budget abstraction and loose-budget retention cross over, but the persistent store is unbounded. |
| 2026-07-20 | [EAR](https://arxiv.org/abs/2607.17879) | `W/Q` | Each query triggers exploration, feedback, buffer replay, and reranker-adapter update. No buffer cap or sleep phase. |
| 2026-07-22 | [World Model Remembers, Actor Forgets](https://arxiv.org/abs/2607.19749) | `B/S` | Every 2,000 environment steps, frozen-world-model dreams train a forgotten actor for 50 updates per prior task. MiniGrid \(n=3\); raw buffer never clears and per-cycle cost grows linearly with tasks. |

Strict deployed-sleep candidates in this July set are therefore limited to the
deterministic external semantic transform, the failed periodic fact
distillation, cross-domain offline dreaming, and periodic world-model dream
rehearsal. None demonstrates a production sleep cluster, atomic publication,
delete fan-out, rollback, and multi-tenant scheduling together.

## 5. Microsoft and adjacent systems corrections

| Source | Correct status | Permitted use | Nonclaim |
|---|---|---|---|
| [LEGOMem](https://arxiv.org/abs/2510.04851) | Earliest public 2025-10-06; AAMAS 2026 final; all authors Microsoft | One-shot construction of 93 full-task memories and 250 subtask memories from successful trajectories over 148 training tasks | Fixed bank; no recurring deployed cycle, failures, growth, conflict, deletion, or rollback |
| [ACON](https://arxiv.org/abs/2510.00615) | Earliest public 2025-09-18 on OpenReview; arXiv 2025-10-01; ICML 2026 / PMLR 306; KAIST, Microsoft, and University of Cambridge affiliations; first-author work performed during a Microsoft internship | Offline natural-language guideline optimisation and compressor distillation; runtime threshold compression | Compressor-policy training plus wake-path replacement, not durable memory sleep |
| [H-EPM](https://arxiv.org/abs/2512.07287) | Earliest public 2025-12-08; ICML 2026 / PMLR 306; HKUST + Microsoft Research Asia | Test-time online tool-graph update is claimed for ACEBench; separate graph-guided GRPO loop | ACEBench boundary order is undisclosed and closest released online writer is inline; not a strict pass; graph has no cap, merge, expiry, delete, or rollback |
| [PlugMem](https://arxiv.org/abs/2603.03296) | Earliest public 2026-02-06; ICML 2026 according to Microsoft Research; UIUC + Tsinghua + Microsoft Research | Completed WebArena trajectories become provenance-linked episodic/semantic/procedural graph memory for later tasks | Experimental task-boundary pass; HotpotQA merge uses soft deactivation and does not establish lifetime equilibrium or physical erasure |
| [MemMA](https://arxiv.org/abs/2603.18718) | Earliest public 2026-03-19; preprint; Penn State + Amazon + Microsoft | After each completed LoCoMo session, five synthetic probes verify provisional memory; failures become evidence-grounded repairs that are consolidated and written back before later sessions | Experimental session-boundary pass in one 19-session conversation; paper quick start uses a 76-row pre-generated probe file whose generation pipeline is not public; no total-store cap, physical erase, versioning, or rollback |
| [MEMENTO](https://arxiv.org/abs/2604.09852) | Microsoft Research article 2026-04-08; arXiv 2026-04-10; all Microsoft | Predeployment staged SFT and optional CISPO teach a model to emit block summaries and physically compact KV during one generation | Runtime is `Q/W`, not sleep; identical summary text loses 15.3 pp after KV restart, so hidden block-conditioned KV is answer-relevant state and source-token eviction is not semantic erasure |
| [Retrospective Harness Optimization](https://arxiv.org/abs/2606.05922) | Earliest public 2026-06-04; City University of Hong Kong + Microsoft Research Asia; later repository/manuscript subtitle *Evolving Agents in the Dark* | Completed unlabeled trajectories drive difficulty-diverse coreset selection, repeated re-solves, diagnosis, three full-harness proposals, and positive self-preference promotion for held-out tasks | Experimental completed-trajectory pass; one update costs 103 agent invocations after selection, same-model preference is not independent attestation, and executable harness/version/log lifetime is uncapped |
| [MemEvolve](https://arxiv.org/abs/2512.18746) | Earliest public 2025-12-21; ICML 2026; multi-institution/OPPO/ByteDance lineage | A fixed LLM evolves executable encode/store/retrieve/manage providers after completed task batches; promoted code is reused next round | Outer architecture experimental pass, but inner content update blocks task-function completion; no common store cap, and per-round storage backups may accumulate |
| [MemSkill](https://arxiv.org/abs/2602.02474) | Earliest public 2026-02-02; preprint; academic collaboration | PPO learns skill selection and a fixed-LLM designer edits a reusable natural-language skill bank every 100 training steps | Predeployment corpus, not deployed wake experience; released normal skill cap is 30, while episode memory is uncapped and training rollback is not semantic rollback |
| [ALMA](https://arxiv.org/abs/2602.07755) | Earliest public 2026-02-08; ICLR 2026 workshop oral; UBC + Vector Institute | Open-ended code search discovers database, update, and retrieval designs; dynamic evaluation updates external memory after a task reward is fixed | Outer and dynamic content loops pass experimentally; no autonomous scheduler, common generated-store cap, safe promotion, or online evolution of design code |
| [M★](https://arxiv.org/abs/2604.11811) | Earliest public 2026-04-10; preprint; City University of Hong Kong + Microsoft | Twenty-iteration reflective population search jointly evolves memory schema, write/read logic, and agent instructions | Predeployment memory-control-program search, not deployed wake consolidation; finite search budget does not bound the selected store |
| [MemCompiler](https://arxiv.org/abs/2605.07594) | Earliest public 2026-05-08; preprint; USTC + HUST + Microsoft Research + collaborators | Teacher-derived SFT and GRPO train a compiler to emit per-step text guidance and 16 latent Soft-Mem tokens from current Brief State and read-only episode memory | Wake-path memory delivery, not consolidation; bounded Brief State/soft tokens do not bound the trajectory bank; no deletion lineage or rollback |
| [STATE-Bench v0.8.1](https://github.com/microsoft/STATE-Bench/releases/tag/v0.8.1) | Microsoft software/benchmark release, 2026-07-16; no paper | 450 stateful enterprise tasks and an Agent Learning track with train/held-out split | Does not prescribe artifact format, capacity, forgetting, or physical erase; offline artifact-building cost may lie outside reported run cost |
| [Agent Memory: Characterization and System Implications of Stateful Long-Horizon Workloads](https://arxiv.org/abs/2606.06448) | 2026-06-04 preprint; Stanford + independent + KU Leuven + MIT | Ten-system suite—including a long-context baseline with no external representation—shows that construction can dominate energy, is long-read/short-write, and creates a sync-latency versus async-staleness constraint | Each isolated local SLURM job receives one H100 80GB and six Intel Xeon Platinum 8480C CPU cores, while remote-construction runs use OpenAI APIs; no distributed consistency, cross-substrate routing, or audited privacy deletion |

LEGOMem, ACON, H-EPM, PlugMem, MemMA, MEMENTO, RHO, M★, MemCompiler, and
STATE-Bench are Microsoft or mixed-Microsoft precedents. MemEvolve, MemSkill,
ALMA, and Agent Memory are independent adjacent control/system studies.
PlugMem instantiates event-triggered task-boundary external consolidation, and
MemMA instantiates probe-verified session-boundary repair, both in benchmark
streams. RHO turns completed agent trajectories into a later persistent
instruction/skill/tool harness, but its one-round same-model self-preference
gate is not a production promotion certificate. H-EPM is an adjacent
online-memory/RL bridge whose strict boundary remains unresolved; MemCompiler
learns memory delivery but runs on the action path. MEMENTO learns
within-response state management and reveals an implicit KV information
channel; it is neither durable memory nor a sleep operator. Only the separate
Human-Inspired Memory Architecture paper is direct Microsoft evidence of an
explicitly named periodic external sleep phase. MemEvolve and ALMA also show
that the executable memory architecture itself can be evolved at an
experimental batch/task boundary; this control-program lineage predates M★,
while RHO extends it to retrospective completed-session evidence.

## 6. Intake-local claim set

| Claim | Class | Intake statement |
|---|---|---|
| `W2-C01` | `SOURCE-SUMMARY` | Strict sleep requires both deferred scheduling and durable cross-timescale transformation; NSTM, EAR, MemCon, and OAS are online/query-time adjacent methods. |
| `W2-C02` | `SYNTHESIS` | External-memory construction is a distinct high-cost service class; a stable sleep queue requires arrival work below service capacity and must trade synchronous latency against asynchronous staleness. |
| `W2-C03` | `SYNTHESIS` | Raw episodes should remain canonical truth and consolidated products should be versioned derived views because repeated LLM rewriting can reverse memory utility. |
| `W2-C04` | `SOURCE-SUMMARY` | Memory addition is not monotone-beneficial; reliable utility labels can make selective addition/deletion better than add-all. |
| `W2-C05` | `SYNTHESIS` | LEGOMem and ACON are offline-control precedents, not recurring deployed sleep. |
| `W2-C06` | `SOURCE-SUMMARY` | OAS establishes a relative-budget crossover, not a general persistent sleep consolidator. |
| `W2-C07` | `SOURCE-SUMMARY` | Weight-written facts can preserve content yet lose addressability; periodic full-fact distillation protected capability but not 100-write retention. |
| `W2-C08` | `SOURCE-SUMMARY` | ReCoLoRA is a task-boundary parametric consolidation method with backbone-sensitive six-task evidence. |
| `W2-C09` | `SOURCE-SUMMARY` | Cross-domain dreaming has a narrow positive rank-256 result and substantial null, dose, module, and judge counterevidence. |
| `W2-C10` | `SOURCE-SUMMARY` | Frozen-world-model dreams can recover a forgotten MiniGrid actor, but the tested replay store is unbounded and rehearsal cost grows with prior tasks. |
| `W2-C11` | `SOURCE-SUMMARY` | NSTM demonstrates live read/write frequency decoupling and constant-shape fast state, while retaining finite-capacity and missed-event limits. |
| `W2-C12` | `SYNTHESIS` | Prompt-budget boundedness, persistent-store boundedness, and reversible deletion are distinct properties. |
| `W2-C13` | `ANALYTIC-DERIVATION` | Re-distilling all accumulated facts every \(k\) writes costs \(O(n^2/k)\); replaying all prior tasks each interval costs \(O(T^2)\) over a \(T\)-task lifetime. |
| `W2-C14` | `SOURCE-SUMMARY` | MemOps and STATE-Bench improve diagnosis but do not demonstrate physical erasure, bounded-memory policy, or full offline-construction cost accounting. |
| `W2-C15` | `HYPOTHESIS` | Promotion into shared weights should require high expected reuse, low volatility, and low deletion risk; volatile or identifying facts should remain versioned external state. |
| `W2-C16` | `SOURCE-SUMMARY` | PlugMem demonstrates external consolidation and later reuse after a completed WebArena task; H-EPM claims online test-memory evolution but does not disclose the boundary order required for strict sleep. |
| `W2-C17` | `SOURCE-SUMMARY` | H-EPM joins external successful-experience memory to a parametric GRPO loop, while leaving graph lifetime capacity and governance unresolved. |
| `W2-C18` | `SOURCE-SUMMARY` | M★ treats the memory harness itself—schema, write/read logic, and instructions—as a learnable program, but does so in predeployment validation search rather than a deployed wake stream. |
| `W2-C19` | `SOURCE-SUMMARY` | MemMA verifies and repairs provisional external memory after a completed dialogue session, but its narrow strict pass depends on one conversation and an undocumented pre-generated probe pipeline. |
| `W2-C20` | `SOURCE-SUMMARY` | MemCompiler demonstrates learned text-plus-latent memory delivery and bounded live compiled state, not deferred durable consolidation or bounded lifetime storage. |
| `W2-C21` | `SOURCE-SUMMARY` | MemEvolve's outer architecture evolution passes at a completed-batch boundary while its inner content update remains wake-blocking; neither finite search rounds nor backup rotation bounds total retained state. |
| `W2-C22` | `SOURCE-SUMMARY` | MemSkill learns a bounded 30-item skill bank before deployment, providing concrete eviction and training-recovery primitives but not deployed sleep or bounded episode memory. |
| `W2-C23` | `SOURCE-SUMMARY` | ALMA predates M★ as open-ended executable memory-design search and also demonstrates post-task dynamic content updates, but not autonomous scheduling, online design evolution, or governed generated stores. |
| `W2-C24` | `SOURCE-SUMMARY` | MEMENTO teaches within-query compaction through staged SFT and optional RL, but restart and passcode probes show that textual summaries omit answer-relevant and privacy-relevant state retained in their block-conditioned KV. |
| `W2-C25` | `SOURCE-SUMMARY` | RHO passes the strict boundary experimentally by promoting a full harness only after completed trajectories, but a positive correlated self-preference score neither proves safe promotion nor bounds executable procedural memory over repeated rounds. |
| `W2-C26` | `SOURCE-SUMMARY` | Surprise-gated full replay passes the boundary experimentally and can halve one episodic buffer without observed loss, but recent-only replay is worse than no replay and full-history rehearsal remains an unbounded lifetime-cost strategy. |
| `W2-C27` | `SOURCE-SUMMARY` | Memento 2 and the Memento-Skills paper learn persistent external controller/skill state, but current-task feedback changes the current-task loop; neither paper demonstrates deferred sleep. |
| `W2-C28` | `SOURCE-SUMMARY` | Memento-Skills DreamDaemon is a later code-level strict external sleep pass, not the mechanism evaluated by the earlier paper, and it lacks quantitative Dream evidence and transactional recovery. |
| `W2-C29` | `SOURCE-SUMMARY` | Evolve demonstrates an explicit external consolidation cycle and scheduler, but identical ordered questions, small mixed-sign pre/post score changes, and no no-sleep control make its result repeated-key reuse rather than held-out transfer or a causal sleep-accuracy effect. |
| `W2-C30` | `SYNTHESIS` | A sleep update spanning files, vector state, metadata, indexes, and staging is not atomic unless one versioned manifest or serving pointer commits the complete derived state and supports rebuild/rollback. |
| `W2-C31` | `SOURCE-SUMMARY` | Mela's released functional neural memory is recomputed inside a model call and is not returned or persisted across calls; it is a test-time architecture, not a durable sleep operator. |
| `W2-C32` | `SOURCE-SUMMARY` | MIRROR is strict asynchronous external consolidation with bounded published narrative state, but bounded per-turn state/work does not imply constant lifetime compute. |
| `W2-C33` | `SOURCE-SUMMARY` | MIRROR's public concurrency path lacks enforced worker bounds and generation-ordered publication, allowing later-wake blocking and stale completion overwrite. |
| `W2-C34` | `SOURCE-SUMMARY` | HeLa-Mem's paper proposes reflective episodic-to-semantic consolidation, while its released default evaluation does not execute the principal consolidation/forgetting path or reproduce the LoCoMo ablation. |
| `W2-C35` | `SYNTHESIS` | Published LLM bits-per-parameter estimates are conditional on what is stored, how it is queried, precision, model, data, and training protocol; no reported value is a universal sleep-memory capacity constant. |
| `W2-C36` | `SOURCE-SUMMARY` | LMLM and LaCy learn parametric-versus-external allocation during pretraining; they are destination-policy precedents rather than deployed sleep-time learning. |
| `W2-C37` | `SOURCE-SUMMARY` | LaCy improves cascade factuality, but factual offloading does not significantly improve tested NLU; “externalize facts to free reasoning capacity” remains unsupported. |
| `W2-C38` | `SYNTHESIS` | An asynchronous consolidator requires monotonic generation IDs and conditional publication in addition to storage atomicity; otherwise completion order can regress the served memory version. |
| `W2-C39` | `SOURCE-SUMMARY` | ImprintBench shows that continual internalization must be tested across acquisition, temporal update, reference resolution, composition, implicit relevance, and boundary awareness rather than direct recall alone. |
| `W2-C40` | `SOURCE-SUMMARY` | LoRA knowledge memory has a finite rank-dependent capacity knee; modular low-rank memories can beat one larger adapter under oracle routing, but router error and merge interference can erase that advantage. |
| `W2-C41` | `SOURCE-SUMMARY` | Raw, QA, summary, rewrite, and mixed supervision produce materially different usable LoRA memories from the same source documents; sleep-data transformation is part of the parametric operator. |
| `W2-C42` | `SOURCE-SUMMARY` | MemoryBench makes explicit and implicit user-feedback learning a necessary wake baseline, while supplying no sleep phase, lifetime bound, or deletion/rollback contract. |
| `W2-C43` | `SOURCE-SUMMARY` | Agent-native memory systems occupy a broad utility–latency frontier and no system dominates all workloads; localized maintenance can be preferable to global reorganization. |
| `W2-C44` | `SOURCE-SUMMARY` | TiMem's session/day/week/month hierarchy reduces recalled context but adds roughly 25–30% construction calls in the reported comparisons and does not bound total stored state. |
| `W2-C45` | `SOURCE-SUMMARY` | MEMORA is a strict experimental multimodal sleep pass: offline consolidation of accumulated egocentric action evidence produces durable habits, workflows, and preferences, with time-scoped anti-leakage controls. |
| `W2-C46` | `SOURCE-SUMMARY` | RecMem sharply reduces eager construction through recurrence-gated promotion, but its accuracy depends on an unbounded verbatim recovery store; recurrence is an admission signal rather than a total-capacity bound. |
| `W2-C47` | `SOURCE-SUMMARY` | GAM's 2,048-token live buffer coexists with growing topic/event graphs and a raw archive; bounded hot state does not imply bounded lifetime state or maintenance work. |
| `W2-C48` | `SYNTHESIS` | Modular parametric memory retains a retrieval layer. A project-defined diagnostic — normalized isolated-module utility × router hit rate × composition-retention ratio × serving hit rate — measures end-to-end usability; it is not stored bits, independent capacity, or a source-authored scaling law. |

## 7. Exact evidence-anchor map

The canonical ingester must split compound numerical statements into separate
records even when this intake lists them together.

| Evidence | Source version and anchor | Relation |
|---|---|---|
| `W2-E01` | NSTM v1, §3.1 and Figs. 1–2 | `QUALIFIES C01`; `SUPPORTS C11` |
| `W2-E02` | EAR v1, §3, Algorithm 1, Appendix A | `QUALIFIES C01` |
| `W2-E03` | MemCon v1, §§3.1–3.2, Fig. 1, Appendix C Table 7 | `QUALIFIES C01` |
| `W2-E04` | OAS v2, Fig. 1 and §3 | `QUALIFIES C01/C06` |
| `W2-E05` | Agent Memory v1, §§4.2–4.3, Figs. 3–5, Table 3 | `SUPPORTS C02` |
| `W2-E06` | Agent Memory v1, §3.1 and §4.6 Fig. 8 | `SUPPORTS C02` |
| `W2-E07` | Useful Memories v1, ARC-AGI and ARC-AGI Stream experiments | `CONTRADICTS` always-consolidate; `SUPPORTS C03` |
| `W2-E08` | Episodic-to-Semantic v1, §§3.1–3.4, Figs. 1–2, Lemma 1, §5 | `SUPPORTS C03` |
| `W2-E09` | [ACL 2026 long paper](https://aclanthology.org/2026.acl-long.27/), Table 2 addition ablation | `SUPPORTS C04` |
| `W2-E10` | Same ACL paper, Table 2 deletion ablation | `SUPPORTS C04` |
| `W2-E11` | LEGOMem v1, §3.2.1, §4.1.3, Table 1 | `QUALIFIES C05` |
| `W2-E12` | ACON v3, §3, Algorithm 1, §4 | `QUALIFIES C05` |
| `W2-E13` | OAS v2, Fig. 2, Table 2, §4 | `SUPPORTS C06` |
| `W2-E14` | OAS v2, Table 12 and Limitations | `QUALIFIES C06`; `NON_CLAIM` |
| `W2-E15` | Facts in Weights v1, §4, Fig. 5b, Figs. 9–11 | `SUPPORTS C07` |
| `W2-E16` | Facts in Weights v1, §4.2, Fig. 7 right | `CONTRADICTS` naive periodic distillation; `SUPPORTS C07/C13` |
| `W2-E17` | ReCoLoRA v1, §3, Fig. 1, Table 1 | `SUPPORTS C08` |
| `W2-E18` | ReCoLoRA v1, Tables 5–6, §6.2 | `QUALIFIES C08` |
| `W2-E19` | Discovery by Dreaming v2, §3.4 Table 2 | `SUPPORTS C09` |
| `W2-E20` | Discovery by Dreaming v2, §3.2, Tables 1–2 and 6 | `QUALIFIES/CONTRADICTS C09` |
| `W2-E21` | Discovery by Dreaming v2, §§4–5 and judge audit appendix | `QUALIFIES C09` |
| `W2-E22` | World Model v1, §5 Fig. 1 and §6 Tables 1–3 | `SUPPORTS C10` |
| `W2-E23` | World Model v1, §3, Algorithm 1, Table 4, §9 | `QUALIFIES C10`; `SUPPORTS C13` |
| `W2-E24` | NSTM v1, §4.4 and Appendix Table 5 | `SUPPORTS C02/C11` |
| `W2-E25` | NSTM v1, §4.4 limitations | `QUALIFIES C11/C12` |
| `W2-E26` | MemDefrag v1, §§4.2.1–4.2.2, Algorithm 2, Appendix D, §5 | `SUPPORTS C12`; `NON_CLAIM` |
| `W2-E27` | PReM v1, §3, Fig. 1, §4 Table 6 | `QUALIFIES C01` |
| `W2-E28` | PMD v1, §§2–3, Fig. 1, Algorithm 1, hyperparameter appendix | `QUALIFIES C01/C07`; `NON_CLAIM` |
| `W2-E29` | MemOps v1, §3.2, Fig. 5, Tables 2–4 | `SUPPORTS C14`; `NON_CLAIM` |
| `W2-E30` | STATE-Bench v0.8.1 commit `4efcbf2d…`, `docs/AGENT_LEARNING_TRACK.md` | `SUPPORTS/QUALIFIES C14` |
| `W2-E31` | [Agent-Memory Protocol](https://proceedings.mlr.press/v317/wu26a.html), protocol section | `SUPPORTS C15`; `NON_CLAIM` |
| `W2-E32` | Mem0 v1, method, architecture, and update algorithm | `SUPPORTS C15`; `NON_CLAIM` |
| `W2-E33` | H-EPM v3, §§4.1–4.3, Appendix C.1 Table 6, Appendix D–E; repository commit `4ad8ad335fe63d365f4baf97df77e19055c79f1c` | `QUALIFIES C16`; `SUPPORTS C17`; closest online code contradicts a strict off-path reading |
| `W2-E34` | PlugMem v1, §§3.1–3.4, Table 5, Appendix C.5 and E.3; repository commit `3b2ce75257d40bca8fac3f78e54e22ea41d92529`, `plugmem_agent.py:328-447` | `SUPPORTS C16`; implemented post-task result-independence; `QUALIFIES` merge/delete/capacity claims |
| `W2-E35` | M★ v2, §§2–5, Algorithm 1, Appendix D.1 and J; repository commit `09a8d34f69588a85fde78b20702d14f056267956` | `SUPPORTS C18`; optimiser lineage is not deployed memory rollback; `NON_CLAIM` for deployed wake learning and lifetime capacity |
| `W2-E36` | MemMA v1, §§4.2–5.5, Tables 2–3, Appendix F–G; repository commit `51d463ad24f8b4f63633d12b32df7c64ed285fff`, `run_memma_self_refine_lightmem.py:2009-2326,4108-4289`, README lines 128–161 | `SUPPORTS C19`; implemented session ordering; `QUALIFIES` reproducibility, leakage-audit, capacity, and production-scheduler claims |
| `W2-E37` | MemCompiler v2, §§3–4, Tables 2–5, Appendix C.3 and Table 7 | `SUPPORTS C20`; training and per-step runtime ordering; `NON_CLAIM` for sleep-time consolidation and lifetime-store boundedness |
| `W2-E38` | MemEvolve ICML 2026 camera-ready, §§3–4, Tables 2–3, Appendix B.1; repository commit `6035d5659d7a092dbfa6a87b1a32a3cee652ba54`, `auto_evolver.py:813-990`, `run_provider.py:48-84`, GAIA runner lines 293–357 and 593–624 | `SUPPORTS C21`; outer/inner lifecycle split; `QUALIFIES` latency objective, storage backup, cap, and rollback claims |
| `W2-E39` | MemSkill v2, §§3.2, 4.5, Tables 1–3 and 7, Appendix B.1/B.4; repository commit `9907c35f8cc71684d06a1f00e0b9c5c4a7b12c4`, config line 24, trainer lines 1112–1182 and 1304–1490, operation bank lines 205–280 | `SUPPORTS C22`; exact cap/eviction/cadence; `NON_CLAIM` for deployed sleep and semantic rollback |
| `W2-E40` | ALMA v1, §§3–4, Table 1, Figs. 4–6, Appendix A.2/B.3; repository commit `7f78ac89da5ca1f0d2ada635be93e0ba8fdf2c54`, `agent_workflow.py:51-257`, `meta_agent.py:216-307` | `SUPPORTS C23`; exact post-reward update ordering; `QUALIFIES` scheduler, capacity, promotion, and online-design-evolution claims |
| `W2-E41` | MEMENTO v1, §§3–6, Tables 1, 3–4, Figs. 2–9, Appendix A.2–A.4; repository commit `d8c10e66ff313844b4b5f063dd4c075d3ea56aab`, `run_full_pipeline.py:205-330`, `processor.py:45-162`, `kv_cache_manager.py:430-456`; OpenMementos dataset card | `SUPPORTS C24`; exact data/training/runtime path; `QUALIFIES` sleep, portability, capacity, privacy-erasure, and peak-only efficiency claims |
| `W2-E42` | RHO v1, §§3–5, Tables 1–3 and 8, Appendix A–C; repository commit `e5f2d1a8a06ab3523ab42e0042d2fa13d9acb701`, `src/rho/loop.py:200-477`, `src/rho/stores/harness.py:46-75`, `codex/retrospection.py:440-644` | `SUPPORTS C25`; exact retrospective ordering, content-addressed promotion, measured call/time cost, and held-out results; `QUALIFIES` scheduler, judge independence, poisoning, capacity, authorization, and rollback claims |
| `W2-E43` | Surprise-Gated Memory v1, System 1 Figs. 1–4, System 2 Figs. 5–10, Limitations; no released code/data or complete training configuration in the paper/arXiv record | `SUPPORTS C26`; strict fast-clear/slow-reuse and full-versus-recent replay ablations; `QUALIFIES` uncertainty, reproducibility, threshold robustness, scaling, lifetime cost, deletion, and production claims |
| `W2-E44` | Memento 2 v3, Algorithms and Theorems 8/10, Corollary 15, assumptions and limitations; no released experiment/dataset/checkpoint | `SUPPORTS C27`; exact wake loop and conditional value bound; `QUALIFIES` sleep, empirical, finite-capacity, and dense-coverage claims |
| `W2-E45` | Memento-Skills v1, §§3–4, router experiment, Tables 1–4, appendices; paper-time repository commit `52c75ebb6eaa422168c8cd473cc9ed34c71165ba` | `SUPPORTS C27`; same-question ground-truth-assisted rewrite/retry and router results; `QUALIFIES` sleep, release completeness, variance, cost, capacity, and rollback claims |
| `W2-E46` | Memento-Skills post-paper commit `71ac933ea1381d53389a2426f59634e0182071b8`, `infra/memory/consolidation/engine.py:115-234`, `daemon/dream/consolidator.py:21-90`, `daemon/dream/loop.py:31-81`, configuration model lines 263–310 | `SUPPORTS C28`; exact staging/background/later-read path; `SUPPORTS C30`; `NON_CLAIM` for paper results or quantitative Dream benefit |
| `W2-E47` | Evolve v1, §§3–6, Tables 2–9, appendices; released answer/judge CSVs and Wilson-score script | `SUPPORTS C29`; answer and logical-call rates; `QUALIFIES` causal sleep effect, provider-call/cost accounting, mismatched latency workload, single-run inference, standard-RAG comparison, and unreleased store-compression artifacts |
| `W2-E48` | Evolve repository commit `edbf8adaec3506f2b2372f1913e887ef07b542f0`, `SleepConsolidation.java:46-174`, `SleepScheduler.java:18-150`, `Orchestrator.java:75-245`, `SectionRetrievalStore.java:150-194`, all four benchmark question resources and section schema | `SUPPORTS C29/C30`; exact lifecycle and identical-phase questions; `QUALIFIES` disabled/local scheduler, category routing, TTL refresh, cross-store atomicity/durability, provenance, capacity, and exact paper-time source claims |
| `W2-E49` | Mela v1, §§3–5 and appendices; repository commit `fa74b2b21c34915df016977ff6546b0a7f0bc545`, HMM forward/model wrapper/generation paths | `SUPPORTS C31`; within-call functional update and missing cross-call state; `NON_CLAIM` for deployed persistence, sleep scheduling, or lifetime capacity |
| `W2-E50` | MIRROR MemAgents workshop paper, §§3–5, Tables 1–2, ablations and latency appendix | `SUPPORTS C32`; asynchronous prior-turn-to-next-turn lifecycle and bounded narrative; `QUALIFIES` lifetime work, benchmark breadth, judge dependence, and error compounding |
| `W2-E51` | MIRROR repository commit `78c6383a3d538250990a1edf1a3dbf7a1aef2ada`, standard/production orchestration and Controller state paths | `SUPPORTS C33/C38`; exact wait, thread, shared-state, and completion-order behavior; `NON_CLAIM` for generation-ordered atomic publication |
| `W2-E52` | HeLa-Mem ACL 2026, §§3–5, Tables 1–3, equations and reproducibility checklist | `SUPPORTS` proposed reflective mechanism and reported ablations; `QUALIFIES C34` through equation/parameter inconsistency, single-value reporting, and unreleased principal benchmark |
| `W2-E53` | HeLa-Mem repository commit `19adb2d709aa74db759835dbc78bc6fa4f4349ef`, LongMemEval scripts, graph update/retrieval, decay and forgetting paths | `SUPPORTS C34`; exact default and call graph; `CONTRADICTS` an executable causal reading of the released consolidation/forgetting result |
| `W2-E54` | Allen-Zhu & Li v1, controlled tuple experiments and 12 capacity results; Morris et al. v3, definitions, synthetic uniform-sequence experiments, precision ablation and limitations | `SUPPORTS C35`; `QUALIFIES` universal-capacity extrapolation |
| `W2-E55` | LMLM v3 / ICLR 2026, data annotation, special-token/loss masking, offload and edit/delete experiments | `SUPPORTS C36`; `NON_CLAIM` for deployed sleep or sequential lifetime allocation |
| `W2-E56` | LaCy v4, §§3–5, FactScore/call-rate/NLU experiments, compute appendix and limitations | `SUPPORTS C36/C37`; `CONTRADICTS` automatic reasoning-capacity gain; `QUALIFIES` domain, scale, and unmeasured freed-bit claims |
| `W2-E57` | ImprintBench official OpenReview `QIJgTW3Qd2`, author-profile records for CTB@ICML 2026 and CompLearn 2026, and indexed workshop-PDF metadata; the exact PDF could not be locally vendored because the official endpoint returned a Cloudflare challenge | `SUPPORTS C39` at record/taxonomy level; numerical promotion remains quarantined until the exact venue-specific PDF is pinned |
| `W2-E58` | LoRA as Knowledge Memory v4, §§4–6, Figs. 4, 6–8, Tables 1–2, master configuration and long-context appendices | `SUPPORTS C40/C41/C48`; exact data-format, rank, matched-budget, routing, merge, and long-context results; `NON_CLAIM` for deployed lifecycle, delete, or unlearning |
| `W2-E59` | MemoryBench v7, §§2.1–3.3, Figs. 1, 3–4, Tables 1–3 and 10–17, feedback-simulator appendices | `SUPPORTS C42`; exact memory/feedback taxonomy, off/on-policy setup, results, and time exclusions; `NON_CLAIM` for sleep or governed lifetime state |
| `W2-E60` | Agent-Native Memory v1, taxonomy §§3.1–3.4, RQ1–RQ5, Figs. 2–8, effectiveness/fidelity/efficiency tables and release statements | `SUPPORTS C43`; utility, evidence-gap, construction/query-cost, and localized-maintenance evidence; `QUALIFIES` release state at the audit cutoff |
| `W2-E61` | TiMem ACL Findings 2026, §§3.1–3.2 and 4.2–4.6, Tables 1–2, 4, 6–7, Limitations | `SUPPORTS C44`; exact temporal hierarchy, call counts, context reduction, latency, and no-storage-forgetting boundary |
| `W2-E62` | MEMORA v1, §§III–V, Fig. 1, Tables I–III, planning and controlled-ablation appendices | `SUPPORTS C45`; exact online-edit/offline-consolidate/later-use order, 45-hour/18-participant scope, time-restricted evaluation, and held-out controls; `NON_CLAIM` for lifetime governance |
| `W2-E63` | RecMem ACL Findings 2026, §§3.1–3.4 and 4.1–4.3, Tables 1–2, Figs. 1 and 3–5, Limitations | `SUPPORTS C46`; recurrence thresholds, construction-token accounting, subconscious-store ablation, and rare-event/capacity limitations |
| `W2-E64` | GAM ACL 2026, §§3.2–3.4 and 4.1–4.4, Tables 1–3 and 10, Algorithms 1–2 | `SUPPORTS C47`; exact 2,048-token live buffer, boundary trigger, graph/archive path, performance and progressive-session growth |
| `W2-E65` | Joint LoRA v4 and GAM/RecMem/TiMem state accounting | `SUPPORTS C48`; project synthesis that adapter catalogs and external archives belong in total-state and retrieval accounting, not a source-authored theorem |

## 8. Systems and scaling implications

### 8.1 Three clocks

- **Wake clock:** milliseconds to seconds; query response, retrieval, and small
  online updates.
- **Consolidation clock:** minutes to hours; batch extraction, replay,
  distillation, dreaming, validation, and publication.
- **Governance clock:** retention and deletion SLA; tombstone propagation,
  revocation, audit, and rebuild.

These clocks require separate queues and SLOs. Combining them in one scheduler
allows long consolidation jobs to violate wake latency or delete deadlines.
Within the consolidation clock, session/day/week/month rollups are distinct
job classes rather than free substeps. Their requested, coalesced, and
completed calls and source generations must be reported separately.

### 8.2 Queue stability

For \(N\) active users, per-user admitted event rate
\(\lambda_u\,[\mathrm{event}/(\mathrm{user}\cdot\mathrm{s})]\), expected
consolidation work \(F_{\mathrm{event}}\,[\mathrm{FLOP}/\mathrm{event}]\), and
aggregate effective sleep service \(G_s\,[\mathrm{FLOP}/\mathrm{s}]\), a
minimal stability condition is

\[
N\lambda_u\mathbb E[F_{\mathrm{event}}] < G_s.
\]

If it fails, staleness grows without bound. Admission priority should be based
on predicted reusable benefit minus harm, normalised by marginal sleep work,
not age alone.

### 8.3 Promotion threshold

Promote an item \(e\) to destination \(j\) only when

\[
v_U\,\mathbb{E}[\text{future reuse}(e)]\,\Delta Q_j(e)
>
C_{\text{train}}+C_{\text{validate}}+C_{\text{move}}
+C_{\text{interference}}+C_{\text{stale}}+C_{\text{revoke}}.
\]

\(v_U\) converts one unit of future-query utility into the same frozen
lifecycle-value unit used by every \(C_\cdot\) term. Without that conversion,
the comparison remains a constrained multi-objective decision rather than a
scalar inequality.

The revocation term makes volatile, private, or user-identifying facts poor
shared-weight candidates. Stable procedures with repeated, independently
observed benefit are stronger adapter or weight candidates.

### 8.4 Publication protocol

1. read immutable snapshot version \(v\);
2. produce candidate \(v+1\) with source IDs, model/prompt/config hashes, and
   transformation lineage;
3. shadow-evaluate remember/update/forget, poisoning, privacy, and regression;
4. atomically swap the serving pointer;
5. retain the prior view/checkpoint for rollback;
6. propagate tombstones to semantic, vector, graph, latent, and adapter views;
7. quarantine or rebuild parametric destinations when exact unlearning is
   unavailable.

For a multi-store external memory, step 2 produces all vector, metadata, graph,
text, and index artifacts under one candidate manifest. Individual database
transactions are not enough: foreground readers must resolve one immutable
manifest version, and recovery must either complete or discard that version
without clearing its source staging range.

### 8.5 Revised novelty boundary

The project must not claim the first:

- sleep phase;
- periodic generative replay;
- fast-to-slow consolidation;
- learned replay selector;
- learned sleep action or curriculum;
- rate–distortion framing;
- external-to-parametric distillation;
- scheduled external section or topic consolidation.

The defensible target is a **versioned memory object and executable lifetime
state machine that learns where and when to route the same items across raw,
semantic/vector/graph, latent/KV, user-parametric, and shared-parametric media
under matched accuracy, latency, energy, storage, staleness, movement,
reversibility, and deletion costs**.
