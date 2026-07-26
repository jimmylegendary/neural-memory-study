# Systems-Infrastructure Primary-Source Adjacency Audit for Sleep-Time Compute

- Audit cutoff: 2026-07-25
- Audit scope: linearizability and MVCC; atomic distributed publication;
  LSM merge and compaction; dynamic vector-index maintenance; LoRA/adaptor
  serving; model and cluster scheduling; disaggregated memory and storage
- Evidence policy: original papers on official proceedings, publisher,
  institutional, project, or author-hosted pages only
- Status: bounded prior-art audit, not a patent search and not a declaration of
  legal novelty
- Companion architecture:
  [`SYSTEM-INFRA-BLUEPRINT.md`](SYSTEM-INFRA-BLUEPRINT.md)

## 1. Audit contract

### 1.1 Unit of analysis

The unit of novelty is an **element or an explicit composition of elements**,
not the project name and not the biological metaphor. This audit therefore asks
three different questions:

1. Is the systems primitive already established?
2. Is the proposed use a Sleep-Time Compute (`STC`) lifecycle integration of
   established primitives?
3. Is there evidence that the exact integration is new?

The answers use the following labels.

| Label | Meaning | Permitted manuscript language |
|---|---|---|
| `KNOWN PRIMITIVE` | A primary source directly establishes the mechanism or a materially equivalent systems contract. | “We adopt,” “we instantiate,” or “we apply”; do not claim primitive novelty. |
| `STC-SPECIFIC INTEGRATION` | The primitive is known, but the STC design binds it to wake/sleep phases, AI-memory media, or STC-specific safety invariants not evaluated together in the cited source. | A systems-integration contribution may be claimed only after implementation and end-to-end evaluation. |
| `NOVELTY UNRESOLVED` | This bounded audit found no primary source establishing the exact conjunction. Non-discovery is not proof of novelty. | “We investigate” or “we hypothesize”; no “first,” “novel,” or exclusivity claim without a broader claim chart and search. |

An element can be a `KNOWN PRIMITIVE` while the stricter STC composition that
contains it remains `NOVELTY UNRESOLVED`.

### 1.2 Claim-boundary rules

1. A database transaction paper does not establish semantic correctness of a
   generated memory, an adapter, or an unlearning operation.
2. A snapshot mechanism does not by itself establish deletion precedence,
   authorization freshness, or garbage-collection safety.
3. An ANN update paper does not establish correctness across embedding-model
   changes or atomicity across vector, graph, text, and model stores.
4. A multi-adapter serving paper does not establish how adapters are trained,
   admitted, deleted, merged safely, or remembered across a user lifecycle.
5. An inference or training scheduler does not establish a joint wake-SLO and
   sleep-staleness policy.
6. A far-memory or disaggregated-storage paper does not establish provenance,
   retention, correction, or cross-substrate capacity accounting.
7. A reported performance result is bounded by the source's hardware,
   workload, scale, and objective. It is not imported as an STC performance
   prediction.

### 1.3 Search limitation

The audit deliberately uses a compact set of canonical and directly adjacent
primary sources. It does not cover patents, proprietary deployments, every
database or vector engine, every adapter-serving system, or unpublished work.
`NOVELTY UNRESOLVED` means that an exact composition was not established by
this source set; it never means that no such prior art exists.

## 2. Executive result

The infrastructure primitives in the STC blueprint are not individually new:

- linearizable publication, multiversion reads, snapshot isolation,
  conditional transactions, and externally consistent distributed
  transactions are established database and distributed-systems mechanisms;
- immutable runs, background merge, tombstones, and compaction are core LSM
  mechanisms;
- incremental ANN insertion, deletion, local maintenance, and rebuild
  avoidance have direct vector-index prior art;
- frozen-base low-rank adaptation, multi-adapter batching, adapter paging,
  merge/unmerge, and request-adapter migration have direct model-serving prior
  art;
- deadline-aware model serving, iteration-level generation scheduling,
  training-job scheduling, and co-adaptive resource allocation are established
  scheduling mechanisms;
- remote/far memory, heterogeneous GPU--CPU--NVMe offload, hardware-resource
  disaggregation, and compute/storage separation are established.

The defensible STC contribution boundary is narrower: one lifecycle and
control-plane contract that jointly handles heterogeneous AI-memory roots,
wake-derived evidence, deferred transforms, version-pinned serving, deletion
and authorization epochs, validation, promotion, rollback, and complete
capacity/resource accounting. This is an `STC-SPECIFIC INTEGRATION`; the exact
composition's novelty remains `NOVELTY UNRESOLVED` until a broader database,
IR, ML-systems, continual-learning, product, patent, and implementation audit
is frozen.

## 3. Primary-source register

### 3.1 Linearizability, MVCC, and atomic distributed state

| ID | Primary source, authors, venue, year, exact URL | What the source establishes | Claim boundary for STC |
|---|---|---|---|
| `DB-01` | Herlihy and Wing, [*Linearizability: A Correctness Condition for Concurrent Objects*](https://cs.brown.edu/people/mph/HerlihyW90/p463-herlihy.pdf), *ACM Transactions on Programming Languages and Systems* 12(3), 1990. Author-hosted by Maurice Herlihy at Brown. | Defines linearizability: each concurrent operation appears to take effect atomically between invocation and response while respecting real-time order. It also establishes locality of the property across objects. | It specifies histories of abstract operations. It does not make a multi-store AI-memory update linearizable, define the STC state object, or prove deletion and authorization semantics. |
| `DB-02` | Bernstein and Goodman, [*Multiversion Concurrency Control—Theory and Algorithms*](https://doi.org/10.1145/319996.319998), *ACM Transactions on Database Systems* 8(4), 1983. | Formalizes multiversion histories in which writes create new versions and analyzes serializability of MVCC algorithms. | It does not select an STC snapshot boundary, govern derived AI artifacts, or define safe reclamation of a version still pinned by a wake request. |
| `DB-03` | Berenson, Bernstein, Gray, Melton, O'Neil, and O'Neil, [*A Critique of ANSI SQL Isolation Levels*](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-95-51.pdf), ACM SIGMOD, 1995; MSR-TR-95-51. Author-hosted by Microsoft Research. | Defines Snapshot Isolation and identifies anomalies not captured by the ANSI phenomena. A transaction reads from a committed snapshot, while write-conflict rules do not make SI equivalent to serializability. | A version-pinned wake view is adjacent to SI, but SI alone does not prove cross-store referential completeness, deletion precedence, or a linearizable manifest publication. |
| `DB-04` | Aguilera, Merchant, Shah, Veitch, and Karamanolis, [*Sinfonia: A New Paradigm for Building Scalable Distributed Systems*](https://research.google/pubs/sinfonia-a-new-paradigm-for-building-scalable-distributed-systems-2/), ACM SOSP, 2007. Official research publication page. | Introduces minitransactions over distributed memory nodes with declared compare, read, and write sets, providing conditional atomic updates for building distributed infrastructure. | It establishes conditional multi-item atomicity, not an STC artifact model, learned-candidate validation, semantic rollback, or a heterogeneous text/vector/graph/adapter commit protocol. |
| `DB-05` | Corbett et al., [*Spanner: Google's Globally-Distributed Database*](https://research.google/pubs/spanner-googles-globally-distributed-database-2/), USENIX OSDI, 2012. Official Google Research page. | Establishes a globally distributed, synchronously replicated, multiversion database with externally consistent transactions, timestamped snapshot reads, and TrueTime-backed ordering. | It does not establish that a model checkpoint, ANN root, graph root, and authorization graph form one correct AI-memory generation, nor does it validate generated memory before commit. |
| `DB-06` | Michael, [*Hazard Pointers: Safe Memory Reclamation for Lock-Free Objects*](https://research.ibm.com/publications/hazard-pointers-safe-memory-reclamation-for-lock-free-objects), *IEEE Transactions on Parallel and Distributed Systems* 15(6), 2004. Official IBM Research page. | Establishes a reader-announced mechanism that prevents removed objects from being reclaimed while they can still be dereferenced, without relying on garbage collection. | It applies to lock-free dynamic objects in one memory-safety model. It does not specify distributed manifest leases, durable rollback windows, authorization after deletion, or object-store garbage collection. |

### 3.2 LSM merge and compaction

| ID | Primary source, authors, venue, year, exact URL | What the source establishes | Claim boundary for STC |
|---|---|---|---|
| `LSM-01` | O'Neil, Cheng, Gawlick, and O'Neil, [*The Log-Structured Merge-Tree (LSM-Tree)*](https://doi.org/10.1007/s002360050048), *Acta Informatica* 33(4), 1996, pp. 351–385. Official publisher record. | Establishes deferred and batched movement of inserts and deletes from an in-memory component through disk components by merge-like operations, with data remaining queryable during maintenance. It explicitly exposes an update/read trade-off. | It does not perform semantic summarization, distillation, graph repair, or learned admission. Ordinary key tombstones are not proof of dependency-complete deletion from derived AI state. |
| `LSM-02` | Chang et al., [*Bigtable: A Distributed Storage System for Structured Data*](https://research.google/pubs/bigtable-a-distributed-storage-system-for-structured-data/), USENIX OSDI, 2006. Official Google Research page. | Implements WAL-backed memtables, immutable SSTables, minor compaction, merging compaction, and major compaction in a distributed serving system. | It establishes production-oriented background compaction of typed records, not utility-aware STC abstraction, cross-medium validation, or atomic promotion of learned model artifacts. |

### 3.3 Dynamic vector-index updates

| ID | Primary source, authors, venue, year, exact URL | What the source establishes | Claim boundary for STC |
|---|---|---|---|
| `ANN-01` | Malkov and Yashunin, [*Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs*](https://arxiv.org/abs/1603.09320), *IEEE Transactions on Pattern Analysis and Machine Intelligence* 42(4), 2020; author preprint first posted 2016. | Establishes hierarchical proximity-graph ANN search and an incremental insertion/construction procedure. | The paper is not a complete high-churn lifecycle protocol. It does not establish authorization-aware deletion, encoder-version migration, or transactionality with payload and graph stores. |
| `ANN-02` | Singh, Subramanya, Krishnaswamy, and Simhadri, [*FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search*](https://arxiv.org/abs/2105.09613), arXiv primary preprint, 2021. | Demonstrates a disk-resident graph index supporting concurrent real-time inserts, deletes, and searches and compares maintenance cost with periodic rebuild practice. | Its vectors live in a fixed representation space for the experiment. It does not solve embedding-model replacement, source-lineage deletion, cross-store snapshots, or semantic correctness of changed payloads. |
| `ANN-03` | Xu et al., [*SPFresh: Incremental In-Place Update for Billion-Scale Vector Search*](https://www.microsoft.com/en-us/research/publication/spfresh-incremental-in-place-update-for-billion-scale-vector-search/), ACM SOSP, 2023. Official Microsoft Research page. | Introduces locality-aware incremental rebalancing and partition splitting so a billion-scale vector index can absorb updates without periodic global rebuilding under the evaluated workloads. | The result is not an unconditional lower bound against rebuild. It does not cover arbitrary embedding-space drift, retained old encoders, version-federated indices, or STC deletion and publication invariants. |

### 3.4 Adapter and LoRA serving

| ID | Primary source, authors, venue, year, exact URL | What the source establishes | Claim boundary for STC |
|---|---|---|---|
| `ADP-01` | Hu et al., [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685), ICLR, 2022. Author preprint. | Freezes pretrained weights and trains low-rank update matrices, reducing trainable and stored task-specific parameters; an adapter can be merged into the base for serving. | The experiments are predeployment task adaptation, not wake-derived recurring consolidation. Fixed rank does not bound catalog size, optimizer state, replicas, merge workspace, or rollback versions. |
| `ADP-02` | Chen, Ye, Wu, Zhuo, Ceze, and Krishnamurthy, [*Punica: Multi-Tenant LoRA Serving*](https://proceedings.mlsys.org/paper_files/paper/2024/hash/054de805fcceb78a201f5e9d53c85908-Abstract-Conference.html), MLSys, 2024. Official proceedings. | Shows heterogeneous batching of requests for different LoRA adapters on one shared base-model copy and cluster consolidation through a serving scheduler. | It serves already-created adapters. It does not establish their training trigger, evidence lineage, validation, deletion, rollback, or long-term catalog policy. |
| `ADP-03` | Sheng et al., [*SLoRA: Scalable Serving of Thousands of LoRA Adapters*](https://proceedings.mlsys.org/paper_files/paper/2024/hash/906419cd502575b617cc489a1a696a67-Abstract-Conference.html), MLSys, 2024. Official proceedings. | Stores adapters in host memory, fetches active adapters to GPU memory, and jointly pages dynamic adapter weights and KV tensors while batching heterogeneous LoRA computation. | It establishes a serving residency hierarchy, not durable STC memory semantics, provenance-aware eviction, selective deletion after merging, or wake/sleep promotion. |
| `ADP-04` | Wu, Zhu, Zhang, Sun, Liu, and Jin, [*dLoRA: Dynamically Orchestrating Requests and Adapters for LoRA LLM Serving*](https://www.usenix.org/conference/osdi24/presentation/wu-bingyang), USENIX OSDI, 2024. | Dynamically merges and unmerges adapters and co-migrates requests and adapters between replicas to respond to workload skew and replica imbalance. | Merge/unmerge here is a serving optimization, not semantic memory consolidation. The paper does not establish reversible source deletion, multi-adapter interference gates, or transactional promotion with external memory. |

### 3.5 Model and cluster scheduling

| ID | Primary source, authors, venue, year, exact URL | What the source establishes | Claim boundary for STC |
|---|---|---|---|
| `SCH-01` | Gu, Chowdhury, Shin, Zhu, Jeon, Qian, Liu, and Guo, [*Tiresias: A GPU Cluster Manager for Distributed Deep Learning*](https://www.usenix.org/conference/nsdi19/presentation/gu), USENIX NSDI, 2019. | Schedules and places distributed training jobs with unknown execution times to reduce job-completion time, including attained-service-based policies. | Its objective is training JCT. It does not jointly protect interactive inference tail latency, price memory staleness, or schedule a typed extract/compact/rebuild/verify DAG. |
| `SCH-02` | Gujarati et al., [*Serving DNNs like Clockwork: Performance Predictability from the Bottom Up*](https://www.usenix.org/conference/osdi20/presentation/gujarati), USENIX OSDI, 2020. | Centralizes admission, model loading, and execution decisions to provide predictable DNN inference and tight request-level SLOs across many models. | It does not schedule training or maintenance jobs and does not establish that variable-duration generative wake traffic can safely share a fleet with STC transforms. |
| `SCH-03` | Qiao et al., [*Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning*](https://www.usenix.org/conference/osdi21/presentation/qiao), USENIX OSDI, 2021. | Co-optimizes cluster-wide resource allocation with per-training-job configuration using a goodput objective. | Goodput is not STC lifecycle utility. The source does not model protected-memory retention, deletion deadlines, candidate validation, or wake p99 interference. |
| `SCH-04` | Yu, Jeong, Kim, Kim, and Chun, [*Orca: A Distributed Serving System for Transformer-Based Generative Models*](https://www.usenix.org/conference/osdi22/presentation/yu), USENIX OSDI, 2022. | Introduces iteration-level scheduling and selective batching for autoregressive transformer serving, improving utilization while managing request latency. | It schedules one serving lifecycle. It does not coordinate background training, index compaction, validation, publication, or later-wake memory freshness. |

### 3.6 Disaggregated memory and storage

| ID | Primary source, authors, venue, year, exact URL | What the source establishes | Claim boundary for STC |
|---|---|---|---|
| `DIS-01` | Dragojević, Narayanan, Hodson, and Castro, [*FaRM: Fast Remote Memory*](https://www.usenix.org/conference/nsdi14/technical-sessions/dragojevi%C4%87), USENIX NSDI, 2014. | Exposes cluster memory as a shared address space with RDMA reads and transactions over allocated objects, demonstrating low-latency key-value and graph stores. | It is a remote-memory transaction substrate, not a heterogeneous AI-memory governance or consolidation system. Volatile object access does not establish durable model publication. |
| `DIS-02` | Gu, Lee, Zhang, Chowdhury, and Shin, [*Efficient Memory Disaggregation with Infiniswap*](https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/gu), USENIX NSDI, 2017. | Transparently exposes idle remote memory as distributed swap over RDMA, using decentralized placement and eviction without application changes. | Page-level remote capacity has no knowledge of tenant provenance, memory utility, model compatibility, retention, or semantic deletion. |
| `DIS-03` | Shan, Huang, Chen, and Zhang, [*LegoOS: A Disseminated, Distributed OS for Hardware Resource Disaggregation*](https://www.usenix.org/conference/osdi18/presentation/shan), USENIX OSDI, 2018. | Separates processor, memory, and storage components and manages them with a splitkernel abstraction in an evaluated prototype. | It does not establish an accelerator-aware wake/sleep deployment, an AI-memory object model, or atomic promotion across model and retrieval artifacts. |
| `DIS-04` | Ruan, Schwarzkopf, Aguilera, and Belay, [*AIFM: High-Performance, Application-Integrated Far Memory*](https://www.usenix.org/conference/osdi20/presentation/ruan), USENIX OSDI, 2020. | Uses application-level object semantics to migrate, prefetch, and evacuate data between near and far memory while avoiding limitations of page-only swapping. | It provides placement mechanisms, not STC value estimation, provenance, durable versioning, or source-dependent reclamation. |
| `DIS-05` | Rajbhandari, Ruwase, Rasley, Smith, and He, [*ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning*](https://www.microsoft.com/en-us/research/publication/zero-infinity-breaking-the-gpu-memory-wall-for-extreme-scale-deep-learning/), SC, 2021. Official Microsoft Research page. | Coordinates model-state partitioning and movement across GPU, CPU, and NVMe to train models larger than accelerator memory while overlapping transfer and computation. | It addresses one training job's model and optimizer state. It does not manage semantic memory, user-scoped adapters, wake retrieval, deletion, or lifecycle promotion. |
| `DIS-06` | Dageville et al., [*The Snowflake Elastic Data Warehouse*](https://doi.org/10.1145/2882903.2903741), ACM SIGMOD, 2016. | Establishes a multi-cluster shared-data architecture that separates elastic compute from persistent cloud storage and uses local caching. | It targets analytical database workloads. It does not establish latency-sensitive model serving, accelerator-state migration, or AI-memory provenance and promotion. |
| `DIS-07` | Verbitski et al., [*Amazon Aurora: Design Considerations for High Throughput Cloud-Native Relational Databases*](https://www.amazon.science/publications/amazon-aurora-design-considerations-for-high-throughput-cloud-native-relational-databases), ACM SIGMOD, 2017. Official Amazon Science page. | Moves redo processing into a distributed, multi-tenant storage service, reduces compute-to-storage network traffic, and provides durable, replicated database state and recovery. | It establishes a database log/storage split, not STC-derived artifact validation, model/index compatibility, or semantics of deleting knowledge compiled into parameters. |

## 4. Element-by-element prior-art chart

### 4.1 Concurrency, snapshots, and publication

| Proposed STC element | Classification | Direct adjacency | Exact claim boundary |
|---|---|---|---|
| One request reads one pinned immutable generation. | `KNOWN PRIMITIVE` | `DB-02`, `DB-03`, `DB-05` | Multiversion snapshot reads are established. STC must still define whether authorization is evaluated only at pin time or continuously and how deletion during a pinned read behaves. |
| A background job writes a private candidate rather than mutating served state. | `KNOWN PRIMITIVE` | `DB-02`, `LSM-01`, `LSM-02` | Copy-on-write versions and immutable runs are known. Generated candidate semantics and validation are outside those sources. |
| Publication has one linearization point. | `KNOWN PRIMITIVE` | `DB-01`, `DB-04`, `DB-05` | Linearizable or transactional publication is established. Merely calling a pointer swap “atomic” is insufficient; the complete STC state object and history must be specified. |
| Conditional publication compares the parent generation before installing a candidate. | `KNOWN PRIMITIVE` | `DB-01`, `DB-04` | Compare-and-conditional-write semantics are known. Parent-only CAS does not cover a deletion or authorization change unless those epochs participate in the same atomic comparison. |
| One manifest names immutable text, vector, graph, latent, adapter, policy, and lineage roots. | `STC-SPECIFIC INTEGRATION` | `DB-04`, `DB-05`, `LSM-02` | Atomic multi-item state and immutable storage components are known. Their composition as one heterogeneous AI-memory generation is an STC contract, not a new transaction primitive. |
| The publication register atomically compares `{parent_generation, parent_manifest_digest, deletion_epoch, authorization_epoch}`. | `STC-SPECIFIC INTEGRATION`; exact contribution `NOVELTY UNRESOLVED` | `DB-01`, `DB-04`, `DB-05` | Conditional transactions can implement the comparison. The STC-specific safety property—no generated or parametric descendant of stale authorization becomes answer-reachable—needs a formal history and failure proof. |
| A deleted source invalidates every derived view before candidate publication. | `STC-SPECIFIC INTEGRATION`; `NOVELTY UNRESOLVED` | Database sources establish atomic state, not AI derivation semantics. | Novelty cannot be claimed until adjacent provenance, data-deletion, machine-unlearning, and materialized-view-maintenance work is audited. |
| Pinned predecessors are reclaimed only after reader quiescence and rollback constraints. | `KNOWN PRIMITIVE` at the reclamation level; `STC-SPECIFIC INTEGRATION` at policy level | `DB-02`, `DB-06` | MVCC retains versions and hazard pointers establish one safe-reclamation pattern, but the STC design must add explicit distributed read leases/epochs or another mechanism and distinguish authorization from physical liveness. |
| Cross-store crash recovery restores one semantically compatible generation. | `STC-SPECIFIC INTEGRATION`; `NOVELTY UNRESOLVED` | `DB-04`, `DB-05`, `DIS-07` | Distributed transactions and database recovery are known. Recovery of a mutually compatible base model, tokenizer, adapter set, encoder, ANN root, graph, and policy is not established by these sources. |

### 4.2 Merge, compaction, and capacity

| Proposed STC element | Classification | Direct adjacency | Exact claim boundary |
|---|---|---|---|
| Append wake events and batch maintenance later. | `KNOWN PRIMITIVE` | `LSM-01`, `LSM-02` | Deferred maintenance and batching are foundational LSM properties. Calling the batch “sleep” does not create novelty. |
| Maintain immutable levels and merge them in the background while reads continue. | `KNOWN PRIMITIVE` | `LSM-01`, `LSM-02` | The storage mechanism is established. STC semantic merge quality and protected-memory retention remain separate empirical questions. |
| Represent deletion with tombstones until compaction makes it physical. | `KNOWN PRIMITIVE` | `LSM-01`, `LSM-02` | Tombstone propagation is not equivalent to deleting facts from summaries, graph dependencies, adapters, optimizer state, or backups. |
| Choose between local maintenance and global rebuild. | `KNOWN PRIMITIVE` for storage/index maintenance; `STC-SPECIFIC INTEGRATION` for the decision objective | `LSM-01`, `LSM-02`, `ANN-02`, `ANN-03` | Locality and rebuild trade-offs are known. A policy using future utility, deletion risk, model compatibility, and wake SLOs is not established here. |
| Use semantic `MERGE`, `ABSTRACT`, `REWRITE`, or `DISTILL` as compaction operators. | `STC-SPECIFIC INTEGRATION`; operator safety `NOVELTY UNRESOLVED` | `LSM-01` and `LSM-02` provide only structural adjacency. | Storage compaction preserves database semantics by defined key rules; learned compression can change truth conditions. It requires acquisition/retention/error gates and cannot inherit LSM correctness. |
| Account for retained and transient candidate, predecessor, migration, workspace, and replica bytes. | `STC-SPECIFIC INTEGRATION`; accounting theory `NOVELTY UNRESOLVED` | `LSM-02`, `ADP-03`, `DIS-05` expose individual hidden-state classes. | The complete cross-substrate ledger is not established by one source. Its novelty and minimal sufficient form require a separate resource-accounting audit. |

### 4.3 Dynamic vector maintenance

| Proposed STC element | Classification | Direct adjacency | Exact claim boundary |
|---|---|---|---|
| Incrementally insert vectors into a graph index. | `KNOWN PRIMITIVE` | `ANN-01`, `ANN-02` | HNSW construction and FreshDiskANN updates directly establish this. |
| Process concurrent insert, delete, and search traffic. | `KNOWN PRIMITIVE` | `ANN-02` | Streaming update support is established for the evaluated fixed embedding space. |
| Avoid periodic full rebuild with local in-place rebalancing. | `KNOWN PRIMITIVE` | `ANN-03` | SPFresh directly establishes one such design. Therefore “all index updates require global rebuild” is false. |
| Use an LSM-like secondary delta index and periodic merge. | `KNOWN PRIMITIVE` | `LSM-01`, `ANN-02`, `ANN-03` | This is an established design family, not an STC novelty. |
| Bind every vector key to payload, encoder, provenance, and authorization versions. | `STC-SPECIFIC INTEGRATION` | `DB-02`, `DB-05`, `ANN-02`, `ANN-03` | Version fields are ordinary systems mechanisms. The cross-artifact invariant is STC-specific and must be tested under concurrent payload and encoder changes. |
| Re-embed changed payloads before the new generation becomes visible. | `STC-SPECIFIC INTEGRATION` | Dynamic ANN papers maintain vectors, not payload semantics. | The rule prevents stale-key publication but is not shown as a new ANN algorithm. |
| Upgrade an encoder by rebuilding affected keys or dual-reading versioned roots. | `STC-SPECIFIC INTEGRATION`; optimal policy `NOVELTY UNRESOLVED` | `DB-02`, `DB-05`, `ANN-02`, `ANN-03` | Existing update systems refute an unconditional global-rebuild lower bound. A general affected-set lower bound must state whether old encoders, federated indices, or query-time translation are permitted and must charge their residency and fan-out. |
| Atomically publish a vector root with text, graph, adapter, and authorization roots. | `STC-SPECIFIC INTEGRATION`; `NOVELTY UNRESOLVED` | `DB-04`, `DB-05`, `ANN-02`, `ANN-03` | Neither dynamic ANN source proves an atomic heterogeneous memory generation. A broader vector-database and transaction-layer audit is required before novelty language. |

### 4.4 Adapter lifecycle and serving

| Proposed STC element | Classification | Direct adjacency | Exact claim boundary |
|---|---|---|---|
| Keep a shared base fixed and store small low-rank deltas. | `KNOWN PRIMITIVE` | `ADP-01` | This is LoRA's defining mechanism. |
| Batch requests that use different adapters on one base-model copy. | `KNOWN PRIMITIVE` | `ADP-02`, `ADP-03` | Punica and SLoRA directly establish heterogeneous multi-adapter batching. |
| Keep a large adapter catalog in host memory and page the active set to GPU memory. | `KNOWN PRIMITIVE` | `ADP-03` | SLoRA directly establishes this residency pattern. |
| Dynamically merge/unmerge adapters or migrate adapter and request together. | `KNOWN PRIMITIVE` | `ADP-04` | dLoRA directly establishes both serving mechanisms. |
| Train a user/session adapter from accumulated wake evidence during a deferred sleep phase. | `STC-SPECIFIC INTEGRATION` | `ADP-01` establishes the training substrate only. | None of `ADP-01`–`ADP-04` establishes recurring wake-derived, off-path, later-wake consolidation. Learning efficacy belongs to the STC learning audit, not this serving audit. |
| Admit, quarantine, canary, publish, and roll back an adapter as memory. | `STC-SPECIFIC INTEGRATION`; `NOVELTY UNRESOLVED` | `DB-04`, `DB-05`, `ADP-02`–`ADP-04` | Transaction and serving mechanisms exist separately. The exact safety/control-plane composition is not established here. |
| Route adapters by tenant/cohort while preserving evidence and deletion lineage. | `STC-SPECIFIC INTEGRATION`; `NOVELTY UNRESOLVED` | `ADP-02`–`ADP-04` route and place adapters but do not govern source lineage. | A claim requires comparison with multi-tenant model-serving, MoE routing, personalization, access-control, provenance, and unlearning prior art. |
| Selectively delete one source after adapter merging. | `NOVELTY UNRESOLVED` | No adapter-serving source establishes selective semantic deletion. | Retaining merge lineage helps rebuild but does not itself unlearn. This requires a machine-unlearning audit and an empirical deletion test. |
| Use fixed adapter rank as a module-local cap and separately bound catalog and auxiliary state. | `STC-SPECIFIC INTEGRATION` | `ADP-01`, `ADP-03` | Rank bounds one module's parameter shape, not module count, catalog metadata, optimizer state, replicas, merge workspace, or rollback checkpoints. A rank-only lifetime-cap claim is rejected. |

### 4.5 Wake and sleep scheduling

| Proposed STC element | Classification | Direct adjacency | Exact claim boundary |
|---|---|---|---|
| Schedule latency-sensitive inference against request SLOs. | `KNOWN PRIMITIVE` | `SCH-02`, `SCH-04` | Predictable serving, iteration scheduling, and selective batching are established. |
| Schedule unknown-duration distributed training jobs for JCT or goodput. | `KNOWN PRIMITIVE` | `SCH-01`, `SCH-03` | Training cluster scheduling and co-adaptive allocation are established. |
| Load, evict, and place many model or adapter variants. | `KNOWN PRIMITIVE` | `SCH-02`, `ADP-02`–`ADP-04` | Model and adapter residency decisions are established serving mechanisms. |
| Let wake inference preempt checkpointable sleep work on a shared fleet. | `STC-SPECIFIC INTEGRATION` | `SCH-01`–`SCH-04` establish component schedulers. | The exact interference and checkpoint contract is not evaluated by these sources. It must report wake p99, lost work, checkpoint I/O, sleep staleness, and starvation. |
| Schedule a typed STC DAG: extract, compact, graph, dream, distill, rebuild, verify, publish. | `STC-SPECIFIC INTEGRATION`; `NOVELTY UNRESOLVED` | `SCH-01`, `SCH-03` | Generic job and resource scheduling are known. The STC operator graph, safety gates, and generation dependencies are a domain-specific composition. |
| Optimize wake utility/SLO, memory freshness, protected retention, deletion deadlines, and lifecycle cost jointly. | `NOVELTY UNRESOLVED` | No scheduler in the bounded set uses this objective. | A defensible contribution requires a dimensional resource vector, preregistered objective or Pareto order, baselines, workload traces, and failure/noninferiority endpoints. |
| Select among shared, time-partitioned, separate, and shard-local wake/sleep fleets. | `STC-SPECIFIC INTEGRATION`; crossover law `NOVELTY UNRESOLVED` | `SCH-01`–`SCH-04`, `DIS-03`–`DIS-07` | The deployment options use known mechanisms. Their crossover depends on model size, shard size, reuse, bandwidth, preemption, privacy, and deadline and must be measured rather than asserted. |
| Coordinate scheduler completion with transactional candidate publication. | `STC-SPECIFIC INTEGRATION`; `NOVELTY UNRESOLVED` | `DB-04`, `DB-05`, `SCH-01`–`SCH-04` | Schedulers and transactions are known separately. The invariant that a late successful worker cannot publish against a newer memory or authorization generation needs an explicit state machine and fault test. |

### 4.6 Disaggregation and data movement

| Proposed STC element | Classification | Direct adjacency | Exact claim boundary |
|---|---|---|---|
| Access remote cluster memory with low CPU overhead. | `KNOWN PRIMITIVE` | `DIS-01`, `DIS-02` | RDMA remote memory and transparent disaggregated swap are established. |
| Expose application-aware objects to a near/far-memory runtime. | `KNOWN PRIMITIVE` | `DIS-04` | AIFM directly establishes semantic hints for object migration and prefetch. |
| Disaggregate compute, memory, and storage hardware. | `KNOWN PRIMITIVE` | `DIS-03` | LegoOS directly explores this architecture. |
| Separate elastic compute from persistent shared storage. | `KNOWN PRIMITIVE` | `DIS-06`, `DIS-07` | Snowflake and Aurora establish two database-specific forms of compute/storage separation. |
| Offload model and optimizer state across GPU, CPU, and NVMe during training. | `KNOWN PRIMITIVE` | `DIS-05` | ZeRO-Infinity directly establishes heterogeneous model-state offload. |
| Move sleep computation toward tenant-local evidence and return a small compiled delta. | `STC-SPECIFIC INTEGRATION` | `DIS-03`–`DIS-07` provide placement and movement substrates. | “Compute to data” is not unconditionally optimal; moving large model weights can destroy batching and residency reuse. |
| Use one provenance-aware fabric for raw events, text, vectors, graphs, latent state, adapters, and recovery versions. | `STC-SPECIFIC INTEGRATION`; `NOVELTY UNRESOLVED` | `DIS-01`–`DIS-07` manage bytes or database objects, not this semantic hierarchy. | The integration may be a contribution after demonstrating typed compatibility, isolation, deletion, rollback, and capacity accounting. The concept alone is not evidence of novelty. |
| Minimize lifecycle data movement across wake, sleep, and memory planes. | `STC-SPECIFIC INTEGRATION`; law and mechanism `NOVELTY UNRESOLVED` | `DIS-04`, `DIS-05`, `DIS-07` expose movement/placement trade-offs. | A valid law must count model weights, canonical-source reads, network bytes, host-device bytes, index migration, replicas, and rollback objects under explicit topology and reuse assumptions. |
| Maintain separate native ledgers for retained state, peak state, cumulative work, movement, and latency. | `STC-SPECIFIC INTEGRATION` | `DIS-02`–`DIS-07` expose distinct state and transfer resources. | Text/vector/graph/adapter bytes and cumulative compute cannot be one capacity number. Retained bytes, peak transient bytes, byte-seconds, I/O, network, CPU time, GPU time, and latency remain separate unless a frozen conversion is justified. |

## 5. Novelty-safe claim chart

| Candidate manuscript claim | Audit result | Safe replacement |
|---|---|---|
| “We introduce atomic publication for sleep memories.” | Rejected: linearizability, conditional transactions, MVCC, and external consistency are known. | “We apply transactional publication to a heterogeneous STC memory generation and specify the additional deletion, authorization, compatibility, and validation invariants.” |
| “We introduce background merge and compaction for agent memory.” | Rejected at the primitive level: LSM and Bigtable establish deferred merge and compaction. | “We test whether semantic and parametric STC operators can inherit the operational benefits of background compaction without inheriting its semantic correctness.” |
| “Vector memory must be globally rebuilt after updates.” | Rejected: FreshDiskANN and SPFresh establish dynamic update paths. | “Encoder changes affect a declared key set; rebuild, local update, version federation, and translation are compared while charging all residency, fan-out, and migration costs.” |
| “A fixed-rank adapter gives bounded long-term memory.” | Rejected: rank bounds one module, not the lifetime catalog or auxiliary state. | “Adapter-local rank is fixed; total parametric capacity includes module count, catalog, optimizer, replicas, workspace, and recovery versions.” |
| “We introduce multi-tenant adapter swapping.” | Rejected: Punica, SLoRA, and dLoRA establish batching, paging, merge/unmerge, and migration. | “We integrate existing multi-adapter serving with STC evidence lineage, validation, scope, deletion, and promotion controls.” |
| “We introduce separate wake and sleep clusters.” | Rejected as a novelty claim: inference and training schedulers and disaggregated deployments are established. | “We characterize when shared, time-partitioned, separate, and shard-local deployments dominate under a joint wake-SLO and sleep-staleness workload.” |
| “No prior system integrates all audited elements.” | Unsupported: bounded non-discovery is not proof. | “The audited sources establish the primitives separately. Whether the exact STC conjunction is novel remains unresolved pending a broader element-by-element and patent audit.” |
| “The architecture is novel because it resembles biological sleep.” | Rejected: metaphor is not a systems element. | “The biological framing motivates phase separation; contribution claims are made only for implemented mechanisms, invariants, measurements, or theory.” |

## 6. Exact STC integration boundary

The following composition is the strongest contribution candidate supported by
this audit, provided it is implemented and evaluated:

```text
immutable wake evidence snapshot
  + version-pinned authorized serving view
  + typed deferred transform DAG
  + candidate-only heterogeneous writes
  + validation on acquisition, retention, safety, deletion, and resources
  + one atomic publication register containing
      manifest generation
      manifest digest
      deletion epoch
      authorization epoch
  + immutable text/vector/graph/latent/adapter/policy/lineage roots
  + reader-safe rollback and reclamation
  + native retained-state, peak-state, compute, I/O, network, and latency ledgers
```

Every line uses known mechanisms. The potentially publishable systems
contribution is the **formal and evaluated composition**:

- a state-machine specification showing what later wake requests may observe;
- a failure model covering crash, retry, late completion, concurrent wake
  append, deletion, authorization change, partial rebuild, and rollback;
- a lineage model connecting source evidence to every answer-reachable
  representation;
- a scheduler that reports both wake SLO and sleep freshness/safety outcomes;
- matched comparisons with simpler transaction, index, adapter, scheduling,
  and storage baselines;
- fault injection showing no mixed generation, stale authorization, or
  reclaimed pinned read;
- capacity accounting that includes inactive, transient, replicated, and
  recovery state.

Without those artifacts, the proposal is an architecture synthesis, not yet a
demonstrated systems contribution.

## 7. Novelty-unresolved research questions

These questions are safe to investigate but unsafe to answer affirmatively from
this audit alone.

### `N-SYS-01` — Heterogeneous AI-memory linearizability

Can a practical specification make one wake request observe a linearizable
combination of model base, routed adapters, tokenizer, embedding encoder,
vector index, temporal graph, policy, and authorization graph without forcing
all payload bytes into one database transaction?

**Required falsifier:** a crash or concurrency history in which a served answer
uses roots from different logical generations, or a proof that the proposed
manifest abstraction cannot represent one of the required read dependencies.

### `N-SYS-02` — Authorization-aware publication

Does atomically comparing generation, manifest digest, deletion epoch, and
authorization epoch prevent all stale-answer-reachable descendants from being
promoted when sleep jobs and policy changes race?

**Required falsifier:** any execution in the declared failure model in which a
candidate derived from a now-denied source becomes reachable after the denial
linearization point.

### `N-SYS-03` — Encoder migration frontier

When does rebuild, local dynamic update, dual-index federation, retained old
encoders, or query-time translation minimize lifecycle cost under a fixed
retrieval-quality and deletion-correctness constraint?

**Required falsifier for a claimed global-refresh lower bound:** one permitted
federated or translated design that satisfies quality and deletion constraints
without re-encoding all keys, after its encoder residency, query fan-out,
calibration, and deletion fan-out are charged.

### `N-SYS-04` — Selectively deletable parametric memory

Can a modular adapter catalog preserve useful cross-session consolidation while
supporting source- or tenant-scoped deletion without full retraining?

**Required falsifier:** a preregistered deletion probe remains recoverable above
the threshold after the declared delete procedure, or protected unrelated
utility falls below its noninferiority margin.

### `N-SYS-05` — Joint wake/sleep scheduler

Can one policy improve sleep freshness or lifecycle utility without violating
wake p99, protected retention, deletion deadlines, or gross resource caps
relative to static fleet partitions?

**Required falsifier:** across all preregistered load cells, every qualified
policy either misses a protected constraint or its adjusted utility/resource
confidence interval is practically equivalent to the best static baseline.

### `N-SYS-06` — Data-movement crossover law

Can the architecture predict when to move raw evidence, compiled deltas, model
weights, or computation under a topology-aware resource vector?

**Required falsifier:** the preregistered crossover ordering fails on held-out
topologies or workload scales outside the practical-equivalence margin.

## 8. Audit disposition

### 8.1 Known primitives that must not be claimed as new

- linearizability, MVCC, snapshot reads, CAS/conditional update, distributed
  transactions, and immutable copy-on-write state;
- LSM-style deferred merge, tombstones, levels, and background compaction;
- incremental ANN insertion/deletion and local update without mandatory global
  rebuild;
- LoRA, shared-base multi-adapter batching, host-to-GPU adapter paging,
  merge/unmerge, and request-adapter migration;
- inference SLO scheduling, generative iteration scheduling, distributed
  training scheduling, and resource/goodput co-optimization;
- far memory, remote memory, heterogeneous state offload, hardware-resource
  disaggregation, and compute/storage separation.

### 8.2 STC-specific integration that can be engineered and evaluated

- a single authorized generation across external and parametric memory media;
- wake/sleep lifecycle state transitions and typed operator DAGs;
- validation-gated promotion and semantic rollback;
- encoder/index/payload compatibility bound to source lineage;
- adapter serving bound to tenant scope, evidence, deletion, and recovery;
- a scheduler that jointly reports wake interference, sleep staleness, safety,
  and resource consumption;
- a complete retained/peak/work/movement ledger.

### 8.3 Novelty that remains unresolved

- the exact atomic authorization-and-deletion invariant across all
  answer-reachable memory media;
- a practical formal consistency model for heterogeneous AI memory;
- selective deletion after parametric merge without full rebuild;
- a general encoder-migration or index-refresh law that includes federation
  and translation;
- the joint wake/sleep scheduling objective and its scaling/crossover law;
- the end-to-end composition of all mechanisms under one evaluated STC
  lifecycle.

The manuscript should present these as research questions or evaluated systems
hypotheses until the broader claim chart is complete. The safe high-level
position is:

> STC infrastructure does not require a new transaction, compaction, ANN,
> adapter-serving, scheduler, or memory-disaggregation primitive. It requires a
> rigorously specified and evaluated integration that makes wake-derived,
> heterogeneous memory safe to transform, validate, publish, serve, delete,
> roll back, and account for across repeated wake and sleep phases.
