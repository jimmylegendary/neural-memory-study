# Mainstream Scenarios for Sleep-Time Compute

- source freeze: 2026-08-05
- horizon: next several model/product generations; no calendar probability is asserted
- current base case: **external-memory dominant in production, evolving toward hybrid promotion where reuse justifies compilation**

## 1. Scenario logic

The scenarios are not mutually exclusive across all workloads. A consumer assistant can be external-memory dominant while an enterprise coding agent uses hybrid promotion and a foundation-model vendor uses periodic-refresh dominant global training.

Each scenario records:

- triggers: technical/economic conditions that make it viable
- leading indicators: observable signals before full adoption
- falsifiers: evidence that should lower confidence
- system shape: wake/sleep/memory architecture
- device implications: workloads that remain if the scenario occurs

## 2. Scenario A — external-memory dominant

### Definition

Persistent learning is implemented mainly by text/event/vector/graph memory. Background jobs synthesize, deduplicate, supersede, merge, decay, and index state; the foundation model remains fixed between global releases.

### Why it is credible now

- OpenAI Dreaming and Mem0 Dream are E4 external background-synthesis evidence.
- Letta provides tagged asynchronous sleep code.
- Zep/Graphiti provides temporal graph and episode provenance.
- sequential factual weight writes show behavioral reachability risk.

### Triggers

1. retrieval and memory-writing quality keeps improving faster than exact model unlearning.
2. base models upgrade frequently, making per-user adapters expensive to migrate.
3. regulation/customer controls require concrete provenance and bounded deletion.
4. context costs fall enough that retrieved memory is affordable.
5. learned external memory policies deliver most skill/strategy benefit without main-model writes.

### Leading indicators

- more product APIs expose pattern, supersede, merge, latest-only, provenance, cadence, and memory events.
- agent benchmarks score reliability and cost with pluggable memory rather than only long recall.
- vector stores evolve into temporal/event systems with learned writer/router models.
- memory state becomes portable across model vendors.
- incident response focuses on poisoned/stale memory and derived-state delete.

### Falsifiers

- retrieval/context overhead remains dominant even at high reuse.
- external summaries systematically lose procedural skill that adapters retain.
- learned memory writers fail to generalize beyond narrow tasks.
- a safe per-user parametric system demonstrates superior cost, delete, and rollback at fleet scale.

### System shape

```text
wake inference cluster
    ↕ retrieval / append
provenance memory fabric
    ↕ scheduled compaction/synthesis/indexing
background memory service
```

### Device implications

- large capacity for immutable episodes and versions
- random-read bandwidth for vector/graph traversal
- sequential scan bandwidth for batch synthesis
- snapshot/COW and content-addressed dedup
- secure deletion/tombstone propagation
- endurance for recurring index/graph rewrites

This scenario strongly benefits memory/storage vendors even without parametric training.

## 3. Scenario B — hybrid promotion dominant

### Definition

External memory is the canonical system-of-record. Background policy promotes only stable, consented, high-reuse knowledge/skill into reversible adapters, experts, latent caches, or compiled prompts; demotion/rebuild occurs when validity or base version changes.

### Triggers

1. a calibrated reuse predictor estimates whether compilation will amortize.
2. adapter/expert artifacts become cheap to validate, route, and atomically publish.
3. provenance links connect each compiled behavior to external evidence.
4. deletion can immediately deny serving and asynchronously rebuild affected artifacts.
5. learned routing recovers most oracle module-isolation benefit.

### Leading indicators

- vendors expose adapter catalogs tied to document/user/cohort lineage.
- memory systems add explicit `promote`, `demote`, `rebuild`, and base-compatibility operations.
- benchmarks sweep query reuse and correction rate, not only one-time accuracy.
- sleep schedulers co-locate state with adapter training while wake serving loads only hot deltas.
- papers report total external + parametric + auxiliary + recovery bytes.

### Falsifiers

- dual-state consistency and model-version churn erase serving savings.
- most promoted items become stale before reaching break-even.
- routing catalog memory/latency grows faster than compiled-state savings.
- delete/rollback cannot be bounded across external and parametric copies.

### System shape

```text
canonical episodes/text/graph
    → external synthesis and validation
    → reuse/volatility/governance decision
    → candidate adapter/expert/latent artifact
    → canary + atomic publish
    → later wake routing
    ↘ demote/rebuild on drift or delete
```

### Device implications

- hot/cold tiering for many small versioned artifacts
- high-bandwidth checkpoint delta build and validation
- metadata/lineage lookup with low tail latency
- atomic multi-object publish and rollback
- near-data preprocessing/embedding generation
- HBM pooling or fast adapter paging to limit base-model replicas

### Assessment

This is the most attractive medium-term architecture because it preserves external governance while allowing parametric amortization. Its evidence maturity is below external-only production systems because end-to-end promotion/demotion is not yet publicly validated at scale.

## 4. Scenario C — periodic-refresh dominant

### Definition

Most durable learning remains centralized in scheduled global/cohort retraining. Per-session memory is limited to retrieval and short-lived state; dedicated per-user sleep jobs remain uncommon.

### Triggers

1. global updates cover most economically valuable changes.
2. frontier training/validation efficiency improves faster than personalized scheduling.
3. privacy rules discourage retaining rich per-user histories.
4. base-model releases are frequent and adapter compatibility poor.
5. retrieval handles freshness between releases.

### Leading indicators

- model vendors shorten post-training release cadence.
- synthetic-data and evaluation factories become centralized platform infrastructure.
- personalization remains prompt/profile based.
- per-tenant adapters are regenerated from source on release rather than continuously updated.

### Falsifiers

- persistent agents need sub-release adaptation to local tools/procedures.
- personal/enterprise data cannot join central training but carries high value.
- global refresh cost or safety validation grows too large.
- external background memory substantially improves reliability between releases.

### System shape

```text
global data lake → offline pre/post-training → versioned base release
local memory     → query-time retrieval only
```

### Device implications

- large centralized training storage/bandwidth remains dominant
- less per-user adapter fan-out
- continuing demand for retrieval index and cache tiers
- snapshot/lineage remains important for release rollback

Even here, background external-memory maintenance survives; only the claim of a distinct personalized sleep-training cluster weakens.

## 5. Scenario D — STC niche

### Definition

Explicit sleep jobs are valuable only in narrow high-reuse domains—coding repositories, enterprise procedures, robotics fleets, or offline/on-device personalization. General assistants rely on retrieval and global refresh.

### Triggers

1. future-query predictability is high only in specialized domains.
2. compile cost is large and reuse distribution heavy-tailed.
3. general personal facts remain too volatile/sensitive for parametric storage.
4. lifecycle validation cost dominates for low-value memories.

### Leading indicators

- STC vendors/products target enterprise corpus compilation rather than universal user learning.
- deployment uses cohort/domain adapters, not one adapter per user.
- sleep runs trigger on capacity/ROI thresholds rather than every night.
- research gains cluster in repeated-task benchmarks.

### Falsifiers

- consumer-scale external dreaming continues to expand and becomes a standard agent primitive—this already weakens the strictest “niche-only” view.
- cheap learned memory policies generalize broadly.
- multi-user batching makes personalized sleep economical.

### System shape

Specialized sleep clusters sit beside ordinary inference fleets and accept only high-value admitted jobs. Most memory remains external.

### Device implications

- opportunity concentrates in enterprise/on-device SKUs
- workload is bursty and state-affine rather than universally high volume
- heterogeneous CPU/NPU/GPU plus large local memory may matter more than maximum training FLOPs
- smaller addressable market but higher willingness to pay for privacy/reliability

## 6. Scenario E — STC broad adoption

### Definition

Persistent AI systems routinely allocate a named background learning budget per user/agent/cohort. External consolidation and selective parametric updates operate continuously, and sleep compute becomes a standard scaling/serving metric alongside pretraining and test-time compute.

### Necessary triggers

1. repeated-cycle retention/plasticity curves remain stable across hundreds of updates.
2. valid-reuse prediction makes compilation economics reliable.
3. recursive dream-data collapse is controlled by canonical real anchors and quality gates.
4. delete/rollback operates end to end across derived external and parametric state.
5. sleep queue stays stable without violating wake P99 SLA.
6. state migration and model residency costs do not erase compute savings.
7. common benchmarks measure later-wake utility per lifecycle cost.

### Leading indicators

- cloud platforms expose separate wake and sleep quotas/SLOs.
- model APIs return durable state/version handles and publish transactions.
- hardware roadmaps discuss consolidation/compaction traces explicitly.
- benchmarks publish curves over sleep FLOPs, evidence, reuse, capacity, cycles, and migration bytes.
- adapter/expert marketplaces/catalogs support automatic promotion and retirement.

### Falsifiers

- no stable capacity knee improvement over external memory at matched total state.
- sleep validation and rollback cost grows faster than saved inference cost.
- per-user state fan-out makes base residency/adapter paging inefficient.
- privacy/security incidents force retention minimization that removes the training corpus.
- model releases invalidate compiled state faster than it pays back.

### System shape

```text
                         ┌─ wake inference / TTT cluster
event + state fabric ────┼─ sleep training / synthesis cluster
                         └─ validation, publish, rollback control plane
                                  ↓
                  external + latent + parametric memory tiers
```

### Device implications

- sustained mixed read/write bandwidth rather than inference read-mostly traffic
- large capacity for canonical evidence, candidate, predecessor, and recovery state
- high endurance for checkpoint/index rewrite and compaction
- low-latency metadata/lineage operations
- atomic snapshot/clone/publish primitives
- memory encryption, per-tenant keys, secure erase, attestation
- accelerator-memory composability to avoid duplicating base weights

## 7. Cross-scenario signal dashboard

| Signal | External dominant | Hybrid promotion | Periodic refresh | STC niche | STC broad |
|---|---|---|---|---|---|
| product background synthesis count | high | high | medium | specialized | high |
| public per-user weight updates | low | selective | low | specialized | high |
| provenance/delete APIs | high | very high | central | domain-specific | very high |
| adapter catalog growth | low | high | cohort | domain | very high |
| base-version churn tolerance | high | medium | high | medium | high after tooling |
| reuse predictor quality | optional | critical | low need | critical | critical |
| dedicated sleep cluster | optional | likely | central training only | domain | standard |
| memory-device mixed writes | high | very high | medium | domain | very high |

## 8. What should be monitored quarterly

1. OpenAI/Mem0/Letta/Zep memory release notes and deletion semantics
2. Google/Meta/Microsoft production announcements, not only research papers
3. learned memory writer/router benchmarks beyond games and web agents
4. repeated-cycle continual fact/skill results
5. exact unlearning and derived-state delete protocols
6. adapter compatibility across base-model upgrades
7. STATE-Bench and other realistic memory benchmark submissions
8. public cost/latency/bytes traces for background synthesis
9. on-device idle-window training and endurance measurements
10. hardware/software support for snapshot/COW/tiered parameter state

## 9. Current scenario judgment

### Most consistent with observed evidence

```text
near term:  external-memory dominant
evolution:  toward hybrid promotion dominant for high-reuse stable skills
coexistence: periodic-refresh dominant for global common knowledge
```

### Less supported today

- STC niche in the sense of “background memory transformation never reaches general assistants” is already weakened by OpenAI and Mem0.
- STC broad adoption in the sense of continuous per-user parametric learning lacks E3–E5 evidence.

The likely mainstream outcome is therefore not a binary win for weights or retrieval. It is a multi-clock system in which external consolidation becomes standard and parametric promotion earns admission case by case.

