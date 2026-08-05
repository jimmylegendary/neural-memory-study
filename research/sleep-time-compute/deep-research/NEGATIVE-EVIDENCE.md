# Sleep-Time Compute 반증·실패·운영 공백 레지스터

- source freeze: 2026-08-05
- 목적: STC를 지지하는 논문 수를 세는 대신, broad adoption을 막거나 조건부로 만드는 evidence를 분리한다.
- claim labels: `DIRECT`는 source가 직접 보인 결과, `INFERENCE`는 여러 source에서 도출한 해석, `GAP`은 필요한 공개 evidence가 없다는 뜻이다.

## 1. 요약

| Risk family | 가장 강한 현재 evidence | STC에 주는 제약 | 상태 |
|---|---|---|---|
| catastrophic forgetting | sequential fact writes, CL theory | sleep update가 old behavior를 보존한다는 검증이 매 cycle 필요 | DIRECT |
| recursive self-distillation | Nature model collapse + accumulated-real-data qualifier | dream data에 provenance와 real anchor가 필요 | DIRECT |
| replay scaling | generative replay/generative distillation | replay cost와 teacher/generator drift가 sleep FLOPs와 함께 증가 | DIRECT + INFERENCE |
| data poisoning | AgentPoison | durable memory write는 security-critical publish operation | DIRECT |
| stale memory | OpenAI/Mem0 product motivation | append-only memory는 long horizon에서 contradiction과 staleness를 누적 | DIRECT |
| deletion | OpenAI controls; Mem0 non-destructive Dream | source delete와 every derived state delete를 구분해야 함 | DIRECT + GAP |
| benchmark leakage | ACL 2026 and search-time contamination | anticipated-query/dream data가 eval과 겹치면 gain이 과대평가될 수 있음 | DIRECT + INFERENCE |
| rollback | Letta merge-on-success는 pattern, broad product evidence는 부족 | bad sleep state를 atomic publish하고 reversible해야 함 | GAP with positive pattern |
| capacity exhaustion | continual facts, rate-distortion, memory-statistics tradeoff | physical bits보다 behavioral/retrieval/governance capacity가 먼저 병목 가능 | DIRECT + THEORY |
| operational incidents | public product docs contain no incident corpus | absence of reports is not evidence of absence | GAP |

## 2. Catastrophic forgetting and behavioral reachability

### NE-01 — sequential factual writes can make old facts unreachable

- type: `DIRECT`
- source: `SRC-STC-0036`, *Can a Language Model Learn Facts Continually in Its Weights?*
- fixed locator: results comparing bare facts and broad study data; 20 sequential writes; recovery interventions; latest-fact error analysis
- observed boundary:
  - broad study material substantially narrows the gap between recitation and composition.
  - nevertheless, after twenty writes, earlier facts are frequently not usable in normal answering.
  - prompts can recover many apparently forgotten facts, so the issue is not equivalent to physical erasure.
- implication: capacity must include **behavioral reachability**. A sleep job can reduce training loss yet still make older knowledge inaccessible to the normal query policy.
- required gate: after each parametric publish, evaluate old facts, new facts, composed use, calibration, and prompted recoverability separately.

### NE-02 — regularization reduces forgetting but spends memory/statistical budget

- type: `DIRECT/THEORY`
- source: `SRC-STC-0050`, *Memory-Statistics Tradeoff in Continual Learning with Structural Regularization*
- fixed locator: OpenReview abstract; theorem statements and upper/lower bounds
- observed boundary: richer structural state can improve joint excess risk but consumes more memory complexity; the published theory uses two linear-regression tasks.
- implication: “consolidation metadata is cheap” is not generally true. Fisher/Hessian sketches, replay exemplars, gradients, and adapters are capacity consumers.
- extrapolation warning: the theorem is not an LLM scaling law.

### NE-03 — small sleep experiments are not unbounded lifelong proofs

- type: `INFERENCE`
- sources: `SRC-STC-0011`, `SRC-STC-0012`
- fixed locator: PLOS task protocol and result figures; Nature Communications task sequence and replay ablations
- boundary: both support sleep-like replay in controlled tasks, but neither tests hundreds of heterogeneous writes, deletion, tenant isolation, or model-version churn.
- implication: cite them as mechanism evidence, not broad deployment proof.

## 3. Recursive self-distillation and synthetic dream collapse

### NE-04 — recursive generated-data replacement can collapse distribution tails

- type: `DIRECT`
- source: `SRC-STC-0041`, *AI Models Collapse When Trained on Recursively Generated Data*
- fixed locator: Nature version-of-record abstract; Methods; Figures 1–4
- observed boundary: successive models trained on generated replacements can lose information about the original distribution, with low-probability regions failing early.
- STC relevance: a dream curriculum that repeatedly consumes prior dream outputs without preserved real evidence can form the same recursive loop.
- prohibited overclaim: the paper does not show that all synthetic augmentation is harmful.

### NE-05 — accumulating real and synthetic data can avoid the specific replacement-collapse regime

- type: `DIRECT/QUALIFYING`
- source: `SRC-STC-0069`, *Is Model Collapse Inevitable?*
- fixed locator: abstract; accumulated-data experiments; main theorem and figures
- observed boundary: replacing real data with each synthetic generation tends toward collapse, whereas retaining accumulated real data alongside generated samples can avoid that regime in tested settings.
- design rule: immutable wake evidence or statistically representative replay anchors must survive compaction; dream data needs generation-depth metadata.

### NE-06 — synthetic replay needs a teacher-quality and denoising gate

- type: `DIRECT`
- source: `SRC-STC-0070`, *Continual Learning of Diffusion Models with Generative Distillation*
- fixed locator: abstract; catastrophic-denoising failure; proposed distillation and experiments
- observed boundary: naive generative replay can fail because the old generator is not a clean substitute for original denoising targets; distillation changes the target construction.
- implication: “generate old-like samples and fine-tune” is not a universal replay recipe. The target, teacher, confidence, and retained-real-data mix matter.

## 4. Replay scaling cost

### NE-07 — replay traffic grows unless selection/compaction is explicit

- type: `INFERENCE grounded in operators`
- sources: `SRC-STC-0003`, `0007`, `0008`, `0021`
- fixed locator: each paper’s method and training-cost discussion; `Language Models Need Sleep` Knowledge Seeding and Dreaming stages
- mechanism:

```text
naive retained replay bytes after T windows:       B_replay(T) = Σ_t B_t
naive sleep training FLOPs:                        C_sleep(T) ∝ epochs × tokens(B_replay(T)) × model cost
bounded replay with selection budget M:            B_active(T) ≤ M
but omitted evidence creates distortion:            D_task(M, selector) > 0
```

- implication: a credible STC system must expose selection, dedup, representative sampling, provenance, and distortion validation. Background time does not make compute or bytes free.

### NE-08 — replay generator is itself a memory that can forget

- type: `DIRECT/INFERENCE`
- sources: `SRC-STC-0003`, `0005`, `0008`, `0070`
- boundary: generative replay moves the old-data dependency into generator quality; it does not eliminate state.
- implication: account for generator parameters, checkpoints, seeds, teacher versions, and validation—not only replay-token bytes.

## 5. Data poisoning and memory integrity

### NE-09 — persistent agent memory is a backdoor surface

- type: `DIRECT`
- source: `SRC-STC-0042`, *AgentPoison*
- fixed locator: NeurIPS abstract; attack construction in Section 3; evaluations in Sections 4–5 and Tables 1–4
- observed boundary: poisoning long-term memory or a RAG knowledge base can trigger targeted agent behavior while preserving benign utility enough to evade casual inspection.
- implication: sleep consolidation amplifies write authority. A malicious episode can be summarized, generalized, and republished into many later sessions.
- required controls: source trust, tenant ACL, anomaly detection, signed lineage, quarantine, canary evaluation, atomic publish, rollback.

### NE-10 — internal memory channels create privacy exposure unseen by output-only audit

- type: `DIRECT`, preprint maturity
- source: `SRC-STC-0072`, *AgentLeak*
- fixed locator: abstract; 32-class attack taxonomy; internal-channel results
- observed boundary: shared memory, inter-agent messages, and tool arguments create exposure paths that output-only checking misses.
- implication: sleep cluster traffic must be included in privacy threat models and DLP—not treated as internal trusted traffic by default.

## 6. Stale memory, contradiction, and temporal validity

### NE-11 — long-lived saved notes become stale

- type: `DIRECT product motivation`
- source: `SRC-STC-0052`, OpenAI Dreaming
- fixed locator: frozen official page lines 39–52 and 241–284
- observed boundary: saved memories can become incorrect or irrelevant as time passes; background synthesis is introduced partly to improve freshness and scalability across multi-year histories.
- implication: memory value is time-dependent. `event_time`, `valid_from`, `valid_to`, and current-location/context cannot be collapsed into a timeless summary.

### NE-12 — soft decay is ranking, not deletion or truth maintenance

- type: `DIRECT`
- source: `SRC-STC-0062`, Mem0 Memory Decay
- fixed locator: frozen official page lines 81–120, 232–271
- observed boundary: decay reorders candidates within a bounded factor; it does not remove memories and is explicitly a search-time concern.
- implication: access-frequency decay can manage a hot set, but it cannot by itself resolve contradiction, legal deletion, or rare-but-critical facts.

### NE-13 — supersede and merge preserve history, which is useful and costly

- type: `DIRECT`
- source: `SRC-STC-0063`, Mem0 Dream
- fixed locator: frozen official page lines 133–145, 150–161, 219–226
- observed boundary: superseded and merged records are retained; synthesized patterns are additive and traceable.
- implication: non-destructive history enables audit/rollback but does not bound physical storage. A separate retention policy is still necessary.

## 7. Deletion and unlearning

### NE-14 — deleting conversation and deleting saved memory are distinct actions

- type: `DIRECT product control`
- source: `SRC-STC-0053`, OpenAI memory controls
- fixed locator: official control documentation covering saved-memory deletion, chat deletion, memory disable, and Temporary Chat
- boundary: user-visible storage surfaces can have separate deletion semantics.
- implication: a publish/delete protocol must enumerate every derived state: raw episode, extracted fact, summary, graph edge, embedding, cache, adapter, optimizer snapshot, evaluation trace, and backup.

### NE-15 — public end-to-end derived-state deletion proof is missing

- type: `GAP`
- audited entities: Google/DeepMind, Meta, Microsoft, OpenAI, Letta/MemGPT, Mem0, Zep
- evidence available: user controls, source links, history flags, repository-level snapshots.
- missing evidence: a public test that deletes one source event and demonstrates bounded-time removal or invalidation from every summary, index, cache, adapter, checkpoint, and replica.
- implication: weight-space STC cannot be the source-of-truth for deletable personal facts until exact or practically certified unlearning/rollback exists.

## 8. Benchmark leakage and evaluation contamination

### NE-16 — benchmark exposure inflates evaluation

- type: `DIRECT`
- source: `SRC-STC-0071`, *When Benchmarks Leak*
- fixed locator: ACL 2026 pages 44743–44760; abstract and contamination experiments
- observed boundary: test set contamination can inflate scores; attempts to detect/remove contaminated items or suppress contaminated behavior have their own reliability/utility costs.
- STC relevance: anticipated-query generation can accidentally reproduce public benchmark items, and later sleep training can turn them into parametric shortcuts.

### NE-17 — retrieval agents can contaminate themselves at evaluation time

- type: `DIRECT`, preprint maturity
- source: `SRC-STC-0073`, *Search-Time Contamination in Deep Research Agents*
- fixed locator: abstract; three-type contamination taxonomy; six-benchmark evaluation
- observed boundary: a searching agent may retrieve benchmark metadata, question context, or answers from the public web, inflating measured performance.
- implication: external-memory STC evaluation needs sandboxed sources, traceable retrieval, time-split tasks, secret holdouts, and exact train/dream/eval lineage.

### NE-18 — current STC papers rarely close the full contamination loop

- type: `INFERENCE/GAP`
- audited artifacts: `SRC-STC-0014`, `0021`, `0043`, `0051`
- missing artifact: cryptographic or exact dataset lineage proving that generated sleep queries, retrieved web context, reward-model data, and final evaluation items are non-overlapping.
- implication: results remain useful, but broad scaling-law fits must include a contamination audit field.

## 9. Rollback and publication atomicity

### NE-19 — merge-on-success is a promising implementation pattern, not yet an industry standard

- type: `DIRECT positive pattern + GAP`
- source: `SRC-STC-0060`, Letta Code Reflection Launcher
- fixed locator: tagged code lines 2207–2234 for isolated worktree/snapshot; 2281–2285 for merge on successful subagent result
- implication: sleep mutation should happen on an isolated candidate state, be evaluated, and become visible atomically.
- gap: public products do not generally publish MTTR, rollback success, state rebase behavior, or consistency across caches/replicas.

### NE-20 — parametric rollback is multi-object rollback

- type: `INFERENCE`
- objects that must stay consistent:

```text
base model version
+ adapter/expert delta
+ optimizer state
+ replay/dream dataset version
+ tokenizer/retriever/embedding model
+ evaluation verdict
+ routing and cache epoch
```

- implication: restoring only a weight file can leave routing, embeddings, caches, and summaries inconsistent.

## 10. Capacity exhaustion

### NE-21 — physical storage, usable knowledge, and active retrieval have different capacities

- type: `SYNTHESIS grounded in DIRECT/THEORY`
- sources: `SRC-STC-0034`, `0035`, `0036`, `0050`, `0062`, `0063`
- fixed locators: capacity estimates; rate-distortion formulation; continual-fact write curves; memory-statistics bounds; product lifecycle docs
- five distinct capacities:

| Capacity | Definition | First visible failure |
|---|---|---|
| physical | bits/parameters/bytes that can be stored | OOM, storage exhaustion, cost |
| behavioral reachability | stored knowledge usable by normal policy | correct fact encoded but not answered |
| retrieval | candidates found within latency/top-k budget | miss, ranking crowding |
| governance | items that can remain consented, current, auditable | stale/undeletable/unauthorized state |
| working set | state resident near compute within SLA | migration latency, cache thrash |

- implication: adding memory until disk or parameters fill is not the relevant naive baseline. Effective capacity can be exceeded much earlier.

### NE-22 — expansion postpones but does not solve routing and governance

- type: `DIRECT/INFERENCE`
- source: `SRC-STC-0049`, MaRS
- fixed locator: OpenReview abstract; slot-expansion method; results
- boundary: controlled capacity expansion can preserve old knowledge, but each new slot still needs routing, versioning, placement, validation, and retirement.
- implication: parameter/expert addition changes the capacity slope; it does not make accumulation free.

### NE-23 — query-agnostic compaction can discard future-relevant rare state

- type: `THEORY/PROPOSAL`
- source: `SRC-STC-0035`, rate-distortion memory compaction
- fixed locator: unified formulation; critique of attention magnitude/recency; repeated-compaction benchmark gap
- implication: compaction must state its query/task distribution. Recency alone is not a sufficient importance estimator.

## 11. Operational evidence and incidents

### NE-24 — no public operational incident corpus was found

- type: `GAP`
- audited official sources: `SRC-STC-0052`–`0067`
- searched categories: failed background run, partial publish, stale summary incident, cross-user contamination, deletion lag, schedule starvation, sleep job wake-SLA interference, corrupted snapshot.
- result: official pages document features and some controls, but no structured incident dataset or reliability curve for background consolidation was found.
- interpretation: this does **not** show that incidents do not occur. It prevents estimating failure rate.

### NE-25 — foreground isolation is stated, but end-to-end interference is unmeasured

- type: `DIRECT + GAP`
- source: `SRC-STC-0063`
- fixed locator: Mem0 Dream lines 204–216, which state scheduled background processing and no live add/search latency addition
- missing: shared accelerator/storage/network P99 traces, preemption latency, queue backlog, and load-shedding behavior.
- implication: logical background execution and physical resource isolation are different guarantees.

## 12. Disconfirmation matrix for the five program hypotheses

| Hypothesis | Evidence against | What would falsify it more strongly |
|---|---|---|
| H-STC: broad parametric STC becomes a primary scaling axis | continual fact reachability failure, deletion/rollback gap, recursive-data risk | fleet-scale per-user weight sleep with repeated-cycle retention, exact rollback, favorable cost |
| H-EXT: external memory dominates | retrieval poisoning, stale state, context pollution, learned skill may need repeated retrieval | external-only systems fail on skill transfer while reversible adapter promotion succeeds broadly |
| H-HYBRID: external system-of-record + selective promotion dominates | operational complexity, dual-state inconsistency, scarce crossover data | a single-medium design matches quality, governance, and cost across facts and skills |
| H-REFRESH: periodic global refresh remains default | personalization and fast-changing local experience miss batch cadence | cheap global refresh plus retrieval matches per-user learning at required freshness |
| H-NICHE: STC remains specialized | OpenAI and Mem0 E4 product evidence already contradicts “no product STC” | broad independent deployments with common scheduler/benchmark would refute niche-only framing further |

## 13. Required negative controls for future STC experiments

1. same total FLOPs without sleep scheduling
2. external RAG/text/graph memory at matched storage and validation cost
3. periodic global refresh at matched data freshness
4. online TTT/fast-state baseline at matched wake latency
5. real-only, synthetic-only, and anchored mixed replay
6. time-split and secret-holdout evaluation
7. repeated consolidation depth, not single cycle
8. old/new/compositional/rare-tail/deletion tasks
9. poisoned and contradictory episodes
10. failed-job, partial-publish, rollback, and rebase injection
11. HBM/DRAM/SSD/network bytes and energy
12. foreground P50/P95/P99 TTFT and ITL during background load

## 14. Bottom line

Negative evidence does not say “sleep-time compute cannot work.” It changes the engineering claim from

> more background training produces more durable intelligence

to the conditional form

> background transformation is valuable when evidence is high quality and reusable, the destination is appropriate, capacity is actively managed, and candidate state can be validated, published, deleted, and rolled back without violating wake SLA.
