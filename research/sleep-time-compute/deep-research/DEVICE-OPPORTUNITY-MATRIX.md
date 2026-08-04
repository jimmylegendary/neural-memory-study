# Memory-Device Opportunity Matrix for Wake–Sleep AI Systems

- source freeze: 2026-08-05
- target reader: memory device, system architecture, controller/firmware, accelerator-memory co-design teams
- evidence boundary: workload opportunities are derived from observed lifecycle primitives; no dedicated public STC device market-size or fleet trace is claimed.

## 1. Core judgment

The durable opportunity is broader than “sell more HBM for night-time training.” Persistent AI creates a state lifecycle:

```text
append evidence
→ index/graph/update metadata
→ scan and synthesize
→ train/compile candidate state
→ validate against old/new/canary data
→ snapshot and atomic publish
→ hot-state serve
→ expire/delete/rollback/rebuild
```

This lifecycle needs capacity, bandwidth, endurance, latency, atomicity, and security across HBM, system memory, pooled memory, SSD/NAND, and controller metadata. Many opportunities survive if STC does not become mainstream because RAG, agent memory, checkpointing, and model-serving already use the same primitives.

## 2. Workload primitives

| Primitive | Dominant operation | Access pattern | State lifetime | Likely bottleneck |
|---|---|---|---|---|
| episode append | small/medium sequential writes | per interaction, tenant-local | days–years | durability, metadata, encryption |
| embedding/index build | model inference + bulk writes | batch scan, vector write | model/index version | compute then write amplification |
| graph maintenance | random read-modify-write | small objects, neighborhood traversal | long | random latency, consistency |
| summary/pattern synthesis | scan + LLM inference | sequential/semantic reads, small output | cadence | source movement, inference cost |
| LoRA/adapter training | GEMM + optimizer writes | repeated dataset read, parameter/optimizer RMW | versioned artifact | HBM capacity/bandwidth, compute |
| replay/distillation | large read + training | old/new mixed stream | many cycles | archive bandwidth, total FLOPs |
| canary validation | parallel inference | model + benchmark reads | per candidate | base residency, scheduling |
| snapshot/COW | metadata clone + changed-page write | sparse delta, version tree | rollback horizon | atomicity, metadata scalability |
| publish | pointer/epoch switch | tiny control write with fleet fan-out | instantaneous/logged | atomic visibility, cache invalidation |
| delete/rebuild | lineage traversal + compaction | sparse logical delete, later bulk reclaim | deadline-bound | secure erase, amplification |
| hot adapter/expert paging | small model-state read | bursty, reuse-skewed | request/session | tail latency, cache hit rate |
| recurrent/KV state reuse | large read-mostly tensor state | prefix/session key lookup | minutes–days | capacity, compatibility, movement |

## 3. Roofline and latency accounting

For a stage with (F) FLOPs, HBM bytes (B_h), host/device bytes (B_{hd}), network/storage bytes (B_n), and random operations (N_{io}):

\[
t_{stage}\ge
\max\left(
\frac{F}{P_{eff}},
\frac{B_h}{BW_h},
\frac{B_{hd}}{BW_{hd}},
\frac{B_n}{BW_n},
\frac{N_{io}}{IOPS_{eff}}
\right)+t_{sync}+t_{queue}.
\]

Operational intensity at the HBM boundary is:

\[
AI_h=\frac{F}{B_h}\quad[\mathrm{FLOP/byte}],
\qquad
P_{roof}=\min(P_{peak},AI_h\,BW_h).
\]

Important consequences:

- adapter GEMM may be compute-bound while optimizer/checkpoint and small-rank kernels are bandwidth/launch-bound.
- graph/index maintenance is typically random-access and metadata-bound rather than FLOP-bound.
- remote sleep clusters can become link-bound even if their GPU is faster.
- validation may double model residency or traffic unless base weights and caches are shared.
- compaction reports must count read + rewritten output + index rebuild + temporary candidate/predecessor bytes.

## 4. Opportunity matrix

| Opportunity | Product/system concept | capacity | bandwidth | endurance | latency | atomicity | security | survives if STC does not become mainstream | Evidence / confidence |
|---|---|---|---|---|---|---|---|---|---|
| O1 provenance-first episode store | content-addressed, append-optimized encrypted object/log tier with event-time and lineage metadata | very high, multi-year | high sequential ingest + scan | high append; low overwrite preferred | moderate ingest, predictable scan | append + manifest commit | per-tenant key, immutable audit, tombstone | **yes**—agent memory/RAG needs it | E4 external memory lifecycle; high |
| O2 snapshot/COW memory state | cheap clone of index/graph/adapter state; page-granular delta; version DAG | candidate + predecessor + recovery reserve | burst write/read during validation/publish | high metadata/delta cycles | low clone/publish tail | **critical** epoch switch and crash consistency | signed snapshot, rollback ACL | **yes**—checkpointing/databases | Letta code pattern E3; medium-high |
| O3 high-endurance mutable-index tier | SSD/NAND/controller path optimized for vector/graph rewrite and compaction | large | mixed random + sequential | **critical** under rewrite amplification | low P99 random read/write | journaled multi-object update | tenant isolation, poisoning audit | **yes**—RAG/vector DB | product docs/repositories E3–E4; high |
| O4 near-data compaction/preprocessing | computational storage or host-side engine for dedup, filter, sampling, tokenize/embed pre-stage | large source-local | high internal scan; reduced uplink | workload-dependent | background deadline, not token latency | checkpointed job progress | attested code/data boundary | **yes**—data pipelines | C10/C12 measurement gap; medium expected value |
| O5 adapter/expert artifact fabric | versioned small-delta store with hot paging to HBM and compatibility digest | many artifacts/cohorts | burst read, multicast, prefetch | mostly read, periodic rebuild | **very low P99** at route/activation | catalog + artifact epoch | tenant ACL, signed lineage | conditional—also serves static PEFT | hybrid scenario; medium |
| O6 CXL/pooled memory for sleep state | shared base/cache/candidate pool to reduce GPU-local duplication | high pooled capacity | coherent/interconnect bandwidth | DRAM-like | low enough for training/validation, not always token hot path | namespace/snapshot consistency | isolation, encrypted links | **yes**—model serving/cache pools | systems hypothesis; medium |
| O7 state-local sleep scheduling telemetry | controller exposes bytes, heat, wear, locality, queue pressure to scheduler | metadata scale | telemetry + placement control | helps wear balancing | low control latency | versioned placement intent | attested counters | **yes**—storage-aware schedulers | no standard trace; research opportunity |
| O8 secure delete/rebuild accelerator | lineage scan, crypto-erase, tombstone compaction, affected-artifact enumeration | whole derived-state graph | scan/rewrite heavy | high during reclamation | logical deny immediate; physical deadline | delete transaction + audit proof | **core requirement** | **yes**—privacy/regulation | current product gap; high strategic value |
| O9 recurrent/KV/neural-memory cache tier | keyed state store with model/version/tokenizer digest and reuse policy | high hot/warm state | read-heavy, large tensor transfer | medium | TTFT-critical | immutable generation + invalidate | tenant-bound encryption | **yes**—TTT/RNN serving | Memory Caching/NSTM frontier; medium-high |
| O10 on-device consolidation window | LPDDR/NAND/NPU co-design for charge/idle/thermal gated local summary/adapter build | bounded local | efficient local scan | NAND wear critical | wake state load low; sleep completion soft | power-loss-safe candidate publish | secure enclave, local-only keys | **yes**—local personalization | C12 measurement gap; high upside/low maturity |
| O11 validation-resident memory pool | share base weights and canary corpus across many candidate jobs; isolate deltas | base + many deltas | high read multicast, delta R/W | mostly read | batch throughput + deadline | candidate result binding | canary confidentiality | conditional but also useful for model CI | inferred systems need; medium |
| O12 lifecycle benchmark appliance/tooling | trace capture and replay for wake/sleep bytes, energy, queue, wear, delete | trace/history | sustained telemetry | trace-store writes | nonintrusive | signed measurement epochs | privacy-preserving aggregation | **yes**—industry lacks evidence | C10/C12 gaps; high first-mover value |

## 5. Opportunity details

### O1. Provenance-first episode store

External product evidence converges on retaining source episodes and deriving summaries/patterns with links. The device/system should support:

- append-only canonical segments
- content hash and source/tenant/time metadata
- separate active view vs retained archive
- fast sequential scan for background synthesis
- selective random fetch for provenance fallback
- logical tombstone with physical reclamation deadline
- generation-aware index pointers

**KPI:** useful canonical bytes/TB, scan GB/s/W, append write amplification, tombstone-to-reclaim time, metadata overhead, corruption detection.

### O2. Snapshot/COW and atomic publish

Sleep jobs should never mutate the active state in place. Required transaction:

```text
clone active generation
→ write candidate deltas
→ validate old/new/poison/delete canaries
→ atomically publish generation pointer
→ retain predecessor for rollback horizon
→ reclaim after readers and policy allow
```

Device/controller support for reflink-like clone, delta maps, epoch fencing, checksums, and crash-consistent manifests can reduce both bytes and publish risk.

**KPI:** clone latency vs state size, changed-byte amplification, publish latency, rollback latency, maximum concurrent generations, recovery success after injected power loss.

### O3. High-endurance index/graph tier

Temporal graph and vector memory updates can generate small random writes plus periodic bulk rebuilds. The device opportunity is not peak sequential bandwidth alone.

**KPI:** random-read P99 under concurrent compaction, bytes written per admitted memory, index rebuild amplification, steady-state endurance, QoS isolation between wake lookup and sleep rewrite.

### O4. Near-data preprocessing

The most plausible near-data work is not full LLM backpropagation. It is low-risk data reduction before expensive transport:

- metadata filter and authorization check
- dedup/minhash/signature
- stratified/reservoir sampling
- compression/decompression
- tokenization or embedding when supported and version-pinned
- graph neighborhood extraction

**Crossover condition:** source bytes are much larger than output delta, privacy/locality prevents movement, and local compute energy plus model-state movement is lower than remote scan traffic.

### O5. Adapter/expert artifact fabric

Hybrid promotion can create a large catalog of small artifacts. Challenges:

- base model/version/tokenizer compatibility
- tenant/cohort routing
- hot-set prediction and prefetch
- HBM fragmentation
- inactive artifact retention and delete
- merge/rebuild history

**KPI:** adapter cache hit rate, added TTFT on miss, artifacts/GB, routing metadata bytes, base upgrade rebuild cost, secure erase proof.

### O6. Pooling and residency

A separate sleep cluster can duplicate huge base weights even when only small adapters change. Pooled memory or regional co-location can share base/canary state, but link latency/bandwidth and fault domain matter.

Compare:

```text
duplicate base in every worker
vs
remote pooled base/cache
vs
shard-local base with remote archive
vs
time-share wake replicas during slack
```

No architecture wins universally. Required input is measured model residency, source shard size, delta size, batch opportunity, and wake interference.

### O8. Delete and secure reclamation

Delete is both metadata and data movement:

1. immediate deny/tombstone
2. stop future retrieval/training
3. identify all descendants
4. invalidate active summary/index/adapter
5. rebuild or rebase unaffected state
6. reclaim snapshots/backups under policy
7. verify no authorized path remains

Device features that expose crypto-erase domains, secure namespace rotation, proof-carrying erase logs, and efficient lineage-index scans can differentiate a memory solution.

### O9. Neural-memory state cache

RNN/TTT models replace or supplement KV with recurrent/neural state. Serving efficiency depends on reuse identity:

```text
(model digest, memory-module version, prefix/session digest,
 tokenizer, precision, update-rule version, tenant policy)
```

State cache must reject incompatible hits and account for hidden source-conditioned state. The opportunity exists even if sleep training remains niche because recurrent/long-context serving still benefits.

### O10. On-device consolidation

Potential trigger gates:

- charging or energy budget available
- thermal headroom
- minimum new evidence count
- privacy policy permits retained local state
- source and base version compatible
- enough free/reclaimable storage

Potential outputs, in increasing risk:

1. deduped/summarized local text
2. rebuilt vector/graph index
3. trained memory writer/router
4. local LoRA/adapter
5. base-weight update

Start from 1–2 because they need less training support and are easier to reverse.

## 6. Scenario robustness

| Opportunity | external-memory dominant | hybrid promotion dominant | periodic-refresh dominant | STC niche | STC broad adoption |
|---|---:|---:|---:|---:|---:|
| provenance episode store | very high | very high | medium | high | very high |
| snapshot/COW | high | very high | very high | high | very high |
| mutable-index endurance | very high | high | high | high | very high |
| near-data preprocessing | high | high | high | domain-dependent | very high |
| adapter/expert fabric | low | very high | medium | high | very high |
| pooled state memory | high | very high | high | medium | very high |
| secure delete/rebuild | very high | very high | high | high | very high |
| recurrent/KV state cache | high | high | high | high | high |
| on-device consolidation | medium | high | low | high | high |
| lifecycle benchmark tooling | high | very high | high | high | very high |

The most robust device bets are provenance storage, snapshot/COW, high-endurance index tiers, secure delete, state cache, and measurement tooling.

## 7. Minimum device requirements by state class

| State class | capacity | bandwidth | endurance | latency | atomicity | security |
|---|---|---|---|---|---|---|
| raw canonical evidence | largest | scan-heavy | append + delete compaction | moderate | segment manifest | encryption, retention ACL |
| external vector/graph | large | random read + rebuild | high RMW | low P99 lookup | index generation | poison/provenance tags |
| latent/KV/recurrent | hot/warm large | very high tensor transfer | mostly read | TTFT-critical | immutable version | tenant isolation |
| adapter/expert | many small/medium | burst load | periodic rewrite | route-critical | artifact epoch | signature, compatibility ACL |
| optimizer/teacher | candidate-only large | HBM/DRAM high | checkpoint write | background deadline | checkpoint consistency | restricted access |
| version/recovery | proportional to rollback depth | moderate | version churn | fast rollback | core requirement | tamper-evident log |

## 8. Reference architecture for a device solution

```text
Wake GPUs / NPUs
  ├─ HBM: base hot layers, active KV/neural state, hot adapters
  ├─ pooled/system memory: warm adapters, candidate deltas, validation cache
  └─ low-latency fabric to memory service

Memory service
  ├─ metadata/lineage DB: tenant, source, version, authorization
  ├─ vector/graph tier: active external memory
  ├─ canonical object/log tier: immutable wake evidence
  ├─ snapshot/COW tier: candidates, predecessors, rollback
  └─ secure reclamation/compaction controller

Sleep workers
  ├─ source-local scan/filter/dedup
  ├─ synthesis/index/replay/distillation/LoRA jobs
  ├─ canary validation
  └─ transactional publish
```

## 9. Proposed benchmark suite for memory-device teams

### Trace families

1. personal assistant: many small events, weekly synthesis, temporal corrections
2. enterprise agent: stateful tasks, strategy memory, daily consolidation
3. code repository: large source scan, repeated queries, adapter candidate
4. recurrent serving: prefix/state reuse with model-version churn
5. delete storm: many descendants, rebuild and secure reclamation
6. poison event: quarantine and rollback
7. base upgrade: index re-embed and adapter compatibility fan-out
8. on-device idle window: energy/thermal/endurance constrained consolidation

### Metrics

- HBM/DRAM/SSD capacity by state class
- read/write bytes and IOPS by phase
- device and link utilization
- write amplification and projected endurance
- energy per admitted event and useful later reuse
- wake TTFT/ITL/P99 during sleep load
- sleep completion age and missed deadline
- snapshot clone/publish/rollback latency
- delete-to-deny and delete-to-reclaim time
- crash recovery and atomicity failures
- state migration bytes and model-residency duplication

## 10. Go-to-market/research prioritization without market hype

### Tier 1 — useful under every plausible scenario

- provenance-aware append and tiering
- snapshot/COW + atomic generation publish
- high-endurance vector/graph/index QoS
- secure delete and lineage-based reclamation
- recurrent/KV/adaptor cache compatibility metadata
- lifecycle telemetry and benchmark tooling

### Tier 2 — high upside if hybrid grows

- adapter/expert paging fabric
- pooled base/candidate state
- near-data preprocessing/embedding
- validation-resident shared memory

### Tier 3 — option value, evidence still sparse

- on-device gradient-based sleep training
- computational-storage model training
- specialized NAND/HBM modes explicitly optimized for per-user continual updates

## 11. Final opportunity verdict

For a memory-device company, the safest thesis is not that sleep FLOPs will simply add another giant GPU training market. It is:

> Persistent AI converts model serving from read-mostly checkpoint execution into a versioned state lifecycle with append, scan, random retrieval, background rewrite, candidate duplication, atomic publish, rollback, and secure reclamation.

The resulting opportunity is strongest where device capabilities expose lifecycle guarantees—not just peak bandwidth. Capacity, bandwidth, endurance, latency, atomicity, and security must be measured together. The opportunity survives if STC does not become mainstream because external agent memory, RAG maintenance, recurrent state caching, checkpointing, and model CI already require most of the same primitives.

