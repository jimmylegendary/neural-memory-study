# Wake--Sleep Memory System Infrastructure Blueprint

- Cutoff: 2026-07-25
- Status: research architecture, not a production reference implementation
- Scope: wake/test-time execution, delayed sleep learning, long-term memory
  fabric, promotion/control plane, data movement, capacity, failure, deletion,
  and experimental validation

## 1. Architecture claim

The defensible architecture is not “three clusters because brains sleep.” It
is a logical separation of latency-sensitive wake execution from
throughput-oriented candidate construction and transactional state promotion.
The logical planes may share or separate physical hardware according to
measured load.

```text
                         ┌──────────────────────────┐
                         │ policy + control plane   │
                         │ admit / route / schedule │
                         │ verify / publish / undo  │
                         └───────┬─────────┬────────┘
                                 │         │
             immutable snapshot │         │ atomic manifest swap
                                 ▼         ▼
┌──────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│ wake plane       │────▶│ sleep plane      │────▶│ published memory   │
│ infer / retrieve │ WAL │ replay / dream   │ cand│ fabric             │
│ fast learn       │     │ FT / RL / compact│     │ text/vector/graph  │
└───────┬──────────┘     └────────┬─────────┘     │ latent/adapter     │
        │                         │               └─────────┬──────────┘
        └──────── version-pinned read view ─────────────────┘
```

The defining sleep transaction is:

\[
T(X_v,M_v,B,P)\rightarrow \widehat M_{v+1},
\]

where \(X_v\) is an immutable wake snapshot, \(M_v\) is the published memory
manifest, \(B\) is a resource budget, \(P\) is the frozen policy/configuration,
and \(\widehat M_{v+1}\) is only a candidate. It becomes \(M_{v+1}\) after
validation and compare-and-swap publication.

## 2. State model

### 2.1 Scope axis

| Scope | Example state | Main advantage | Main risk |
|---|---|---|---|
| shared fixed | base model, tokenizer | amortized capability | cannot absorb personal updates safely |
| shared slow | globally promoted adapter/weight, common procedure | high reuse | cross-user interference and rollback blast radius |
| cohort | domain/team adapter, shared graph | narrower reuse and governance | cohort leakage and version fan-out |
| user | preference memory, user adapter, personal graph | isolation | long-tail storage/residency |
| session | fast weights, KV, working memory | low write latency | volatile and easy to lose |
| ephemeral query | prompt, scratchpad, tool trace | no durable burden | no later-wake benefit |

Scope and medium are independent. A user memory can be text, graph, latent
state, or adapter; a graph can be user-local or shared.

### 2.2 Medium axis

| Medium | Canonical? | Mutable? | Typical serving path |
|---|---|---|---|
| raw event/evidence | yes within retention policy | append/version/delete | cold or selective retrieval |
| semantic text | derived | rewrite/version | prompt injection |
| vector index | derived | rebuild/invalidate | ANN retrieval |
| temporal graph | derived | edge version/invalidate | graph traversal + prompt |
| latent/KV | derived; may contain implicit source information absent from visible tokens | checkpoint/expire/rebuild | accelerator-resident read; restart may not be equivalent |
| adapter/expert | compiled derived state | train/merge/retire | model forward |
| shared weights | compiled, hardest to reverse | train/rebuild/unlearn | model forward |
| optimizer/importance/router | auxiliary memory | update/reset | sleep/update control |

The raw event is the system of record only while policy permits its retention.
All other forms are materialized views with source lineage. If raw retention
expires, the system must state which recovery guarantees disappear.

### 2.3 Complete capacity accounting

Incremental answer-bearing memory state is not only the live prompt:

\[
B_{\text{retained}} =
B_r+B_e+B_l+B_p+B_a+B_v.
\]

The terms are mutually exclusive physical ownership classes for raw/canonical
evidence, external text/vector/graph and its index, latent/KV state, parametric
modules, auxiliary catalog/router/optimizer state, and
version/lineage/recovery state. \(B_p\) includes inactive/preallocated experts;
\(B_a\) includes Fisher/importance, router, and generator state; \(B_v\)
includes checkpoints and rollback versions. Active, durable, replicated, and
backup bytes are reported as projections or physical copies without double
counting.
Every answer-reachable KV/recurrent state counts even when its source tokens
have been physically evicted or the state has no user-visible serialization.

Retained incremental state is distinct from incremental peak state:

\[
\begin{aligned}
B_{\text{peak,state}}(t)={}&B_{\text{retained}}(t)
+B_{\text{candidate}}(t)+B_{\text{pinned}}(t)\\
&+B_{\text{migration}}(t)+B_{\text{workspace}}(t)
+B_{\text{replica}}(t).
\end{aligned}
\]

`B_candidate` includes unacknowledged staging and candidate objects;
`B_pinned` covers excess predecessor bytes retained solely by readers;
`B_migration` covers excess dual indexes/encoders; `B_workspace` covers build,
merge, tool, and validation sandboxes; and `B_replica` covers temporary copies
not already counted in retained state. Every term includes only bytes absent
from \(B_{\text{retained}}\), has a hard cap, and has a named GC owner.
Retained/peak-state bytes and cumulative work are separate ledgers.

The frozen base is excluded from the incremental information-capacity state but
not from infrastructure. Fleet placement and HBM admission use:

\[
B_{\text{peak,physical}}(t)=
B_{\text{base,resident}}(t)+B_{\text{peak,state}}(t)
+B_{\text{execution}}(t),
\]

charging every physical base-model copy on wake/sleep fleets and all
nonpersistent execution memory absent from the state ledger. Separate clusters
can reduce contention yet increase \(B_{\text{base,resident}}\); both state
and physical peaks are mandatory. The reported physical maximum is
\[
\max_t\left[
B_{\text{base,resident}}(t)+B_{\text{peak,state}}(t)
+B_{\text{execution}}(t)\right],
\]
with all components evaluated at the same maximizing timestamp, never as a sum
of independent component high-water marks.

## 3. Wake plane

### 3.1 Responsibilities

- serve inference under a latency SLO;
- pin one authorized serving head and register a bounded reader lease;
- retrieve and assemble external memory;
- maintain optional fast/session state;
- append events, tool outcomes, corrections, and consent changes to a
  write-ahead log;
- assign opaque tenant-scoped source handles;
- enforce immediate deny/tombstone checks before any memory read;
- emit lightweight pressure and utility signals;
- never publish an unvalidated sleep candidate.

Immediate denial outranks snapshot consistency. If authorization or deletion
changes during a pinned request, the request may not perform another protected
artifact read or emit an answer containing the newly denied source; it aborts,
filters, or falls back under the frozen policy. Bytes already processed are not
misreported as “unread,” and the physical-removal deadline is measured
separately.

### 3.2 Fast learning boundary

Wake may perform cheap online operations:

```text
append a raw event
update a session summary
write a bounded fast-state statistic
apply a user-local online adapter step
invalidate a contradicted edge
```

Each operation still records its parent state and can be disabled. It is
classified as wake/test-time learning, not sleep. The sleep plane can later
consolidate or undo its effect.

### 3.3 Wake SLOs

At minimum:

```text
p50/p95/p99 end-to-end latency
retrieval and prompt-assembly latency
memory-version pin time
fast-update overhead
bytes read and written
cache/KV invalidation rate
fallback and abstention rate
stale/unauthorized read count
```

Wake traffic has priority when shared resources saturate. The fallback is a
last-known-good authorized manifest, provenance-only retrieval, or
no-memory/abstention—not a poisoned or deleted prior version.

## 4. Snapshot boundary

### 4.1 Event log

The wake log is append-oriented, tenant encrypted, and ordered by a monotonic
event sequence or causally explicit partial order. A record contains:

```text
opaque source/event handle
tenant and privacy scope in deletable sidecar
event time and ingest time
payload or payload reference
tool/environment outcome
model/prompt/version that produced the event
correction, conflict, and consent status
retention/deletion epoch
content checksum inside the protected domain
```

Operational manifests outside the encrypted sidecar use random,
non-derivable handles rather than direct identifiers or reusable content
fingerprints.

### 4.2 Snapshot isolation

A tenant's publication authority is one linearizable control record:

\[
\mathrm{ServingHead}=
(\text{manifest generation},\text{manifest digest},
\text{authorization epoch},\text{deletion epoch/high-watermark}).
\]

The four fields are one compare-and-swap word, not independently checked
metadata. Every authorization or deletion change advances `ServingHead` even
when memory content has not yet been reclaimed. The manifest digest names the
exact authorization/deny root as well as the heterogeneous memory roots. A
delete prepares the new deny root and atomically installs its digest/epochs in
`ServingHead`; that CAS is the denial linearization point.

A sleep job receives:

```text
exact parent ServingHead
event range [a,b]
policy/config digest
budget and deadline
input object digests
```

Events after `b` remain in wake. A deletion after snapshot creation advances
the atomic head and quarantines affected in-flight candidates. A candidate may
publish only by comparing the exact complete parent head; a validation-to-CAS
deletion or authorization race therefore changes the CAS word and aborts.

### 4.3 Copy-on-write

Sleep never mutates the live serving tree. It writes:

```text
temporary objects
→ content checksums
→ per-artifact readiness records
→ candidate manifest
→ validation attestations
→ manifest-last atomic publish
```

Partial and orphaned candidates are disposable. Retries are idempotent by
operation digest.

## 5. Sleep plane

### 5.1 Job classes

| Class | Typical operator | Preferred hardware | Output |
|---|---|---|---|
| extract | parse, classify, redact, embed | CPU/GPU inference | events, embeddings |
| diagnose | generate provenance-bound probes, freeze/evaluate provisional state, localize failures | LLM inference + isolated retrieval | signed probe results and repair candidates |
| compact | dedup, merge, summarize | LLM inference + CPU/index | semantic view |
| temporal-rollup | close session/day/week/month windows and coalesce dependencies | LLM inference + CPU/index | hierarchy-level view |
| local-maintain | transform one tenant/subgraph/topic region | colocated CPU/RAM + LLM inference | scoped graph/index delta |
| view-compile | compile staged text into zero/one/many versioned replacements | LLM inference + vector/metadata/filesystem | candidate multi-store manifest |
| graph | entity resolution, temporal update | CPU/RAM + LLM inference | graph delta |
| embodied | edit/consolidate time-anchored video/action evidence | VLM/LLM inference + object store | entity, habit, workflow, preference views |
| replay | sample/mix canonical or latent data | storage/CPU | training shard |
| dream | generate/filter counterfactual data | GPU inference | versioned synthetic shard |
| distill | teacher→KV/adapter/expert | GPU training | compiled model delta |
| adapter-package | route, compatibility-check, merge, and cache modular deltas | CPU + GPU serving | versioned routed adapter set |
| tune | SFT/regularised update | GPU training | adapter/weight candidate |
| policy | PPO/GRPO/bandit update | rollout + GPU training | router/controller |
| program-search | reflective code evolution or retrospective trajectory diagnosis + full-harness proposal/ranking | sandboxed CPU/LLM inference | versioned memory harness |
| rebuild | re-embed/re-index/retrain from source | mixed, high I/O | clean replacement |
| verify | canary, delete, poison, performance tests | isolated mixed fleet | signed result |

Jobs declare native resource vectors rather than one synthetic FLOP estimate:

\[
\mathbf c_j =
(c_{\text{GPU-s}},c_{\text{CPU-s}},c_{\text{HBM-byte-s}},
c_{\text{RAM-byte-s}},c_{\text{I/O-byte}},c_{\text{net-byte}}).
\]

### 5.2 Scheduler action space

For one admitted memory region, the scheduler can:

```text
NO_OP
RETAIN_EXACT
EXPIRE
PROBE_REPAIR
DEDUP
MERGE
ABSTRACT
REWRITE
GRAPH_UPDATE
LATENT_COMPILE
ADAPTER_COMPILE
PARAMETRIC_PROMOTE
DEMOTE
RECYCLE
REBUILD
QUARANTINE
```

The destination, operator, scope, and cadence are distinct decisions. A learned
policy may propose them, but hard privacy, deletion, capacity, and promotion
constraints remain outside the policy.

The concrete scheduling key is:

```text
(when, tenant/cohort, evidence range, graph/topic region,
 operator, destination, resource budget, deadline)
```

TiMem-style nested clocks, RecMem-style recurrence thresholds, and GAM-style
pause/end/overflow triggers can all propose this tuple. None may silently
upgrade a local job into a global rebuild. A multi-clock planner coalesces
overlapping evidence ranges, records which lower-level generation each rollup
consumed, and invalidates descendants when a correction or deletion changes
that generation.

PlugMem shows one implemented event trigger that belongs in this interface:
completed task. MemMA adds a completed-session trigger and a
probe→failure-localization→repair sub-DAG, but its released paper-aligned probe
generator is missing; probe code, source snapshot, random seed, filter, and
evaluation-separation manifest must therefore be required inputs. H-EPM
motivates success-gated memory evolution, but its paper does not expose the
ACEBench ordering and its closest released online writer is inline; it cannot
support a strict post-task trigger claim. None owns a periodic/pressure
production scheduler.

RHO adds a concrete completed-retrospective-batch job: its Codex workflow mines
finished sessions, constructs candidate instruction/skill/tool directories in
isolation, and can apply a positive winner after a manual decision. Its
measured one-round optimization is not a cheap metadata step—103 agent
invocations after difficulty selection—so selection, diagnosis, proposal,
re-solve, ranking, and independent-canary capacity must be separately
schedulable.

Surprise-Gated Memory adds a candidate admission signal and a
fast-store-pressure trigger, not a production scheduling solution. Its
full-history replay preserves old classes while a recent-only replay window is
worse than no replay. The scheduler must therefore log novelty calibration,
false admission/rejection, replay-mixture coverage, retained support by
age/class, and marginal old/new utility. Shrinking a working set under pressure
cannot bypass the protected-old canary.

Evolve and the post-paper Memento-Skills DreamDaemon add executable schedule
points. Evolve can run manually or at 03:00 once staging contains one section.
DreamDaemon combines a quick post-response transform with a 600-second scan,
24-hour delay, and an effective five-session-or-10-KB staging gate. These
defaults should enter the experimental action space, but neither supplies
evidence that clock cadence is optimal. A scheduler must also distinguish
foreground acquisition/TTL refresh from deferred compilation: an `async`
function called before the current query finishes remains wake work.

MemEvolve, ALMA, MemSkill, M★, and RHO show that memory architecture, skill
selection, schema, write/read logic, agent instruction, and executable tools
can themselves be learned or searched artifacts. The scheduler must therefore
version both the memory contents and the **harness that interprets them**; a
harness change is a schema/code/capability migration, not an ordinary memory
write. MemCompiler adds a separate wake-plane interface: state-conditioned
text/latent delivery can be learned, but its per-step output cannot be
mislabeled as a sleep artifact. Microsoft MEMENTO adds a stricter
representation warning: within-query compaction can preserve information in
block-conditioned summary KV that is absent from the summary text. Such work
stays on the wake plane, but its artifact manifest and governance tests belong
to the same control plane.

### 5.3 Batching and affinity

Useful batching keys:

- same base model/tokenizer and adapter shape;
- same embedding/index version;
- same tenant or permitted cohort boundary;
- compatible privacy and data-residency zone;
- similar deadline and operator;
- same source shard to avoid network movement.

Batch efficiency must not merge security domains. Cross-tenant gradients or
summaries are forbidden unless a separate shared-promotion protocol has
explicit authorization and privacy review.

### 5.4 Preemption

Extract/compact jobs can often checkpoint at item boundaries. Training jobs
checkpoint optimizer and RNG state at declared intervals. A preempted job does
not hold a live manifest lock. Deadline failure produces a typed result:

```text
DEFER
CHEAPER_OPERATOR
PARTIAL_REGION
DROP_LOW_UTILITY
NO_OP
```

It never silently publishes a partial state as complete.

## 6. Promotion and control plane

### 6.1 Admission

Admission estimates:

\[
V_i =
v_U\widehat{\Delta U_i}
\int_0^H\widehat\lambda_i(t)\widehat S_i(t)e^{-\omega t}dt
- C_i
- C^{\mathrm{risk}}_i,
\]

where \(v_U\) is a frozen lifecycle-value conversion per utility unit, `C` is
lifecycle resource cost converted to that value unit, and
\(C^{\mathrm{risk}}\) is expected
volatility, privacy, poison, correction, and deletion risk in the same unit.
The horizon \(H\) is seconds, \(\widehat\lambda_i\) is reuse/s,
\(\widehat S_i\) is dimensionless survival, and \(\omega\) is s\(^{-1}\).
A cycle-indexed policy must use a separately declared per-cycle formulation.
Absent those conversions, admission is a constrained multi-objective decision
rather than a scalar score. This is a candidate decision score, not an
established scaling law.

Admission is intentionally conservative:

- retain exact externally when evidence/reuse is weak;
- compile only after repeated validated reuse;
- prefer user-local over shared promotion without cross-user evidence;
- preserve a rebuild or rollback path;
- no-op when uncertainty exceeds the expected gain.

### 6.2 Candidate validation

Every candidate is compared with its incumbent under identical probes:

```text
current-task utility
protected old-memory retention
fixed-budget acquisition on held-out new tasks
exact/semantic/temporal/relational/procedural recall
false-memory and calibration
rare/failure/correction slices
privacy, poison, and authorization
delete dependency completeness
latency, bytes, compute, and energy
rollback recovery
probe generator/data lineage and final-evaluation separation
```

Aggregate gain cannot compensate for a preregistered protected-slice violation.
Acquisition and retention are separate gates.

### 6.3 Publication transaction

```text
1. verify all candidate artifact checksums
2. read and verify the exact parent ServingHead
3. verify model/tokenizer/index compatibility
4. recompute or prove equivalence of every changed payload's retrieval key
5. verify required independent attestations
6. write every vector/graph/metadata/file/index sub-root under a candidate manifest
7. verify cross-store referential and key--payload completeness at the readiness barrier
8. compare-and-swap the complete ServingHead from the exact parent to the candidate head
9. emit publication and source-staging acknowledgement receipts
10. keep predecessor and unacknowledged source range during rollback window
11. reclaim only after reader leases, rollback, deletion, and backup constraints
```

Because manifest identity and authorization/deletion epochs occupy the same CAS
word, the CAS prevents a stale sleep job from erasing a newer wake update,
deletion, authorization change, or candidate. A pre-CAS watermark check alone
is insufficient. Rebase is explicit and revalidates the combined state.

Asynchronous completion order is distinct from scheduling order. Every job
therefore carries:

```text
tenant/session
input event high-watermark
exact parent ServingHead
candidate logical generation
idempotency token
deadline and cancellation state
```

Publication requires an exact full-head match and a strictly newer logical
generation. A successful but late worker is recorded as `STALE_COMPLETION` and
its candidate remains quarantined or is reclaimed; it may not overwrite the
served state. “Last thread to finish wins” is forbidden. If the new incumbent
contains events absent from the stale job, rebase means rebuilding and
revalidating from a new immutable snapshot, not relabelling the old output.

This requirement applies even when the destination is one bounded narrative.
MIRROR demonstrates the useful Talker/Thinker split but its public
implementation does not attach a compare-and-swap generation to Controller
completion, and a rapid later turn may wait for earlier background work. A
production service also enforces per-tenant/global worker limits, deadline
cancellation, a last-known-good fallback, and no unbounded thread retention.

This transaction is deliberately stronger than one transaction per substrate.
Evolve updates vectors and SQLite metadata sequentially; DreamDaemon writes
topic files, deletes old files, rebuilds an index, and clears staging through
separate filesystem operations. Either can leave a mixed generation after a
crash or concurrent read. The serving path therefore resolves only immutable
sub-roots named by one manifest. Staging is not cleared in place: its event
range is acknowledged after pointer publication and retained until recovery
policy permits reclamation.

Every serving read registers a renewable, bounded lease plus a non-bypassable
epoch/hazard slot for its exact head. Generation order alone is not a
reclamation proof because immutable objects may be shared across versions. GC
first computes the reachability closure of the current head, every live pinned
head, in-flight candidate/migration roots, and every rollback, backup,
legal-hold, or deletion-retention root. A physical object is a reclamation
candidate only when it is outside that union and below the minimum applicable
live generation/retire epoch.

Lease expiry requests cancellation but is not itself quiescence. Before
physical reclamation, every reader that could have acquired the retired object
must acknowledge a post-retire quiescent state, release its hazard slot, or be
terminated behind an isolation boundary that proves its old capability cannot
execute. A stalled reader with no acknowledgement keeps the object live and
causes backpressure or worker fencing; timeout alone never authorizes free.
Any resumed reader must reacquire a current fenced capability before another
object dereference.

A mid-read delete advances `ServingHead` and the immediate-deny map; deny wins
for any further read or answer emission, while the hazard/lease protocol still
prevents use-after-free until the reader is quiescent. This separates
authorization precedence from safe physical reclamation. The serving path
checks the current head/deny root before each protected artifact read and
before response emission; pinning an old content generation never pins old
authorization.

The same rule applies inside one vector backend. The audited LightMem update
path can replace the memory text while writing back the incumbent embedding
vector. A successful vector-store call can therefore publish a stale semantic
key. Each item manifest binds payload digest, key digest, encoder digest,
provenance, and authorization watermark. Changed payloads are re-embedded or
must pass a backend-specific ranking-equivalence proof; sampling may monitor
drift, but it cannot waive the generation invariant. If the encoder changes,
the control plane either rebuilds all affected keys behind a new root or
dual-reads versioned roots until migration completes. It never silently
relabels the old index as compatible.

### 6.4 Rollback

Rollback reuses only still-authorized, compatible predecessor **memory
sub-roots**. It CAS-publishes a new manifest and `ServingHead` with a newer
generation, the current deny/authorization/deletion roots, and monotonically
advanced epochs; it never reinstalls an old complete head or regresses
governance state. Cache invalidation follows that publication. This is tested
against delete-before-rollback and delete-during-rollback interleavings, not
inferred from checkpoint existence. If a deletion makes predecessor content
unauthorized, rollback uses a rebuilt clean version or no-memory fallback.

### 6.5 Harness and policy promotion

A learned router or evolved memory harness is promoted through the same
candidate transaction as data and weights, but with additional obligations:

```text
typed schema and migration digest
write/read behavioral contract
resource and output limits
sandbox and dependency allowlist
static plus rotating validation provenance
cross-task/tenant transfer tests
delete, correction, and rollback conformance
fallback to the incumbent harness
```

MemEvolve and ALMA establish earlier executable provider/design search;
MemSkill adds a bounded skill bank with usage/reward replacement and training
snapshots; M★ jointly searches schema, memory logic, and instruction. M★
reports that native task-specific programs beat transferred programs on all
three tested targets and that five of six transfers fall below its universal
seed. RHO later demonstrates that completed real-session trajectories can
produce and promote a complete persistent harness, with content-addressed
versions and an incumbent backup. Yet its solver, diagnoses, proposals, and
pairwise preference all use the same model family, and the selected proposal
is not always the hidden-test-best candidate. A positive self-preference score
therefore satisfies a task-utility gate only; it cannot substitute for an
independent poison, authorization, and capability canary.

This is evidence for specialisation, not permission to auto-deploy
benchmark-selected code. A production hierarchy should freeze a universal
governance kernel, validate tenant/task-specific operator modules, and publish
them independently of memory contents so either side can roll back without
silently reinterpreting the other. Optimizer checkpoints, design archives,
skill snapshots, content-addressed harnesses, and filesystem backups are
recovery inputs; they become semantic rollback only after the corresponding
memory state, schema, indexes, executable dependencies, and authorization
watermark are restored atomically.

## 7. Long-term memory fabric

### 7.1 Tiering

One useful logical hierarchy is:

| Tier | Contents | Relative access target | Typical policy |
|---|---|---|---|
| hot | active session state, top vectors, resident adapters | lowest/serving path | strict cap, frequent eviction |
| warm | episodic/semantic store, graph partitions | interactive retrieval | TTL, dedup, compaction |
| cold | canonical events, old versions, training shards | asynchronous/batch | retention policy, batch access |
| recovery | immutable candidate/predecessor checkpoints | exceptional | rollback window then reclaim |

The hot cap is not the lifetime cap. GAM's bounded progression buffer,
RecMem's reduced constructed view, TiMem's reduced recalled context, and a
fixed-rank adapter can all coexist with growing archives, graphs, hierarchy
levels, catalogs, or checkpoints. Capacity admission reads one complete ledger:

```text
B_retained =
  B_r raw/canonical evidence
  + B_e external text/vector/graph payload and index
  + B_l latent/KV state
  + B_p parametric modules
  + B_a auxiliary catalog/router/optimizer state
  + B_v version/lineage/recovery state

B_peak,state =
  B_retained
  + disjoint candidate/staging + reader-pinned predecessor
  + migration/dual-index overlap + build/merge workspace
  + temporary replicas
```

These are mutually exclusive physical ownership classes: a content-addressed
byte is charged once, while a physical replica is charged where it resides.
`hot`, `active`, `warm`, `cold`, and `archive` are overlapping
residency/lifecycle projections of the partition and are reported separately,
not added to it. Transient overhead is \(B_{\mathrm{peak,state}}-
B_{\mathrm{retained}}\), while the enforced peak cap applies to the total.
The incremental state cap and the separate
\(B_{\mathrm{peak,physical}}\) fleet/HBM cap are both enforced. Peak sampling
is calibrated against allocator and storage telemetry before G4.

Canonical source and derived view are orthogonal to hot/warm/cold placement.
A cold raw event can remain the authorized truth while a hot vector, warm
topic document, or adapter is rebuilt from it. A generated section identifier
without source spans, teacher/prompt digest, and parent lineage is not a
canonical anchor. Evolve's released section schema illustrates this failure:
compact serving state exists, but evidence provenance needed for correction,
rebuild, and deletion does not.

Microsoft HIMA is evidence that hot/episodic/graph lifecycle mechanisms can be
made explicit. It is not proof that this exact tier count or its nominal
six-hour cadence is optimal.

### 7.2 Index and model compatibility

Every derived artifact binds:

```text
base-model digest
tokenizer digest
encoder/embedding digest
schema and compiler version
attention/mask policy, position convention, and KV layout
source snapshot and parent manifest
privacy/deletion epoch
explicit-state digest and implicit-state completeness class
```

A base or embedding upgrade can mark latent/vector/adapter artifacts
incompatible. They are rebuilt from authorized canonical sources, converted
with an audited migration, or retired. For a stateful latent artifact,
text-identical re-prefill is not presumed equivalent: MEMENTO's restart
ablation loses `15.3 pp` while keeping memento text fixed. The registry must
declare one of `PORTABLE_TEXT`, `SERIALIZED_LATENT`, `LIVE_KV_LINEAGE`, or
`REBUILD_ONLY`, and validation must test the claimed restart and migration
semantics.

### 7.3 Parametric registry

Adapter/expert records additionally track:

```text
shape and parameter count
rank, module family, and catalog generation
scope and authorization
active/free/recyclable/quarantined status
training data and teacher digests
optimizer and objective
canary result
parent and merge lineage
router/index digest and oracle-versus-deployed route metrics
compatibility matrix and permitted merge operator
residency/offload location
load/swap cost and resident-cache priority
rollback expiry
```

Inactive masked experts count toward physical capacity. Merged adapters retain
lineage to all inputs and cannot be advertised as selectively deletable unless
the rebuild/unlearning path is tested. A fixed adapter rank is a module-local
bound only: catalog metadata, module count, merge workspace, optimizer state,
resident copies, and predecessor checkpoints remain in total-state accounting.

### 7.4 Temporal and multimodal lineage

Every temporal rollup records:

```text
tenant and time-window identity
lower-level source generations
window-close trigger and clock
raw/reversible evidence roots
summary/graph payload and retrieval-key roots
operator/model/prompt digest
consent and deletion epoch
superseded and dependent rollups
```

Every embodied-derived record additionally points to participant/video/time
spans and typed observation/action provenance. Entity state, inferred
knowledge, habit, workflow, preference, benchmark item, and parametric
descendants share one authorization graph. A time-restricted evaluation
resolves a manifest at the query anchor; post-anchor products cannot be hidden
by retrieval filters after they have already influenced a derived view.

## 8. Minimize data movement

### 8.1 Principle

Move computation toward the data or move sufficient, provenance-bearing
deltas—not full histories—when the privacy and hardware boundary permits.

Candidate techniques:

- shard sleep work by tenant/source locality;
- embed and redact near ingest;
- use content-addressed immutable objects with deduplicated transfer;
- send event ranges and object digests rather than copies;
- train on local shards and publish small LoRA/adapter deltas;
- keep graph partition and entity-resolution work colocated;
- cache teacher logits or sufficient statistics only when their storage and
  expiry are cheaper than recomputation;
- pin model weights near repeated sleep jobs;
- use delta checkpoints and manifest roots;
- rebuild indices locally from a canonical shard.

### 8.2 Movement accounting

For each candidate:

\[
C_{\text{move}} =
\sum_{\ell}
\left(
B_{\ell}\cdot p_{\ell}
+ T_{\ell}\cdot v_{\text{delay}}
\right),
\]

where \(B_\ell\) is bytes moved across link class \(\ell\), \(T_\ell\) is its
transfer delay in seconds, and \(p_\ell\) and \(v_{\text{delay}}\) convert byte
and second into the same frozen lifecycle-value unit. Energy is either
converted through that declared price or reported as a separate vector
coordinate; it is not silently added to currency. Report source read, network,
accelerator host-device, replica, and publication bytes separately.

### 8.3 When not to move compute

Moving an accelerator job to each data shard can lose batching and weight
reuse. The design must measure the crossover among:

```text
model weight size
source shard size
number of candidate events
expected repeated jobs per checkpoint
network bandwidth
accelerator utilization
privacy/data-residency constraints
deadline
```

Data locality is a hypothesis, not an unconditional rule.

## 9. Physical deployment options

### Option A — One shared fleet, priority isolation

Wake and sleep share accelerators; wake preempts sleep.

**Best when:** sleep load is small/bursty, models dominate transfer cost, and
idle inference capacity is available.

**Risks:** wake P99 interference, repeated preemption, optimizer/checkpoint
thrash, unpredictable sleep completion.

### Option B — Time-partitioned fleet

The same hardware changes role by region/time window.

**Best when:** usage has predictable diurnal valleys and model/data residency
can persist across the switch.

**Risks:** timezone diversity, unfinished jobs crossing wake peaks, capacity
stranding.

### Option C — Separate wake and sleep fleets

Latency-optimized serving and throughput-optimized training are physically
separate.

**Best when:** sustained sleep load, different accelerator types, hard
isolation, or predictable batching offsets transfer.

**Risks:** state/network movement, duplicate model residency, slow promotion,
split operational ownership.

### Option D — Regional/shard-local hybrid

Cheap extract/embed/compact work stays near memory shards; heavy distillation
uses a pooled sleep fleet; published deltas return to the shard.

**Best when:** raw data is large/private and compiled outputs are small.

**Risks:** complex orchestration, heterogeneous reproducibility, cross-shard
entity resolution.

The benchmark should identify the load/size/interference crossover rather than
selecting Option C by analogy.

## 10. Queueing and capacity control

### 10.1 Necessary stability

For job classes \(k\), keep compute service, byte flow, and occupancy stock
dimensionally separate:

\[
\sum_k\lambda_k\mathbb E[w^{\mathrm{gpu}}_k] < m_{\mathrm{gpu}},
\qquad
\sum_k\lambda_k\mathbb E[w^{\mathrm{cpu}}_k] < m_{\mathrm{cpu}},
\]

\[
\sum_k\lambda_k\mathbb E[d^{\mathrm{io}}_k] < C_{\mathrm{io}},
\qquad
\sum_k\lambda_k\mathbb E[d^{\mathrm{net}}_k] < C_{\mathrm{net}},
\]

\[
\sum_k\lambda_k\mathbb E[a^{\mathrm{ram}}_k] < B_{\mathrm{ram,cap}},
\qquad
\sum_k\lambda_k\mathbb E[a^{\mathrm{hbm}}_k] < B_{\mathrm{hbm,cap}},
\]

where \(w\) is GPU-s/job or core-s/job, \(d\) is byte/job, and \(a\) is
byte-s/job. The occupancy necessities also require sample-path stock caps
\(\sup_tB_{\mathrm{ram,resident}}(t)\le B_{\mathrm{ram,cap}}\) and
\(\sup_tB_{\mathrm{hbm,resident}}(t)\le B_{\mathrm{hbm,cap}}\).
RAM/HBM bytes are never compared with byte/s bandwidth.
This is necessary but not sufficient under setup costs, correlated bursts,
deadlines, affinity, multi-resource blocking, and retries.

Nested temporal clocks are separate job classes in each typed sum. The controller
does not estimate them as one event rate multiplied by one average cost:

\[
\sum_{j\in\{\text{session,day,week,month}\}}
\lambda_j\mathbb E[w^x_j]
+\lambda_{\text{pressure}}\mathbb E[w^x_{\text{pressure}}]
<m_x,\quad x\in\{\mathrm{gpu},\mathrm{cpu}\},
\]

\[
\sum_j\lambda_j\mathbb E[d^y_j]
+\lambda_{\text{pressure}}\mathbb E[d^y_{\text{pressure}}]
<C_y,\quad y\in\{\mathrm{io},\mathrm{net}\},
\]

with an analogous byte-s/job occupancy sum against each RAM/HBM byte cap plus
the hard instantaneous stock cap.

Jobs sharing evidence ranges are coalesced when model, tenant, generation, and
deadline permit. The scheduler reports raw requested jobs, coalesced jobs,
provider/model calls, and completed publications separately; hiding added
clock demand through one aggregated call count is forbidden.

### 10.2 Operational gates

```text
p99 queue age
per-clock requested/coalesced/completed calls and oldest-generation lag
deadline miss rate
oldest privacy/delete job age
backlog work by resource
retry and poison-quarantine rate
novelty-gate calibration drift and protected-support coverage
tenant fairness
sleep-to-wake preemption cost
candidate waste before publish
local-maintenance touched-state fraction and boundary-repair debt
```

If pressure binds, the control plane can lower admission, choose a cheaper
operator, shrink the working region, defer low-value work, or retain exact
external state. It must not allow silent unbounded backlog.

### 10.3 Nine load indicators

```text
rho_sleep    sleep service
rho_store    durable byte flow
rho_read     live context/application bandwidth
rho_repr     representational associations
rho_hot      accelerator residency
rho_gov      deletion/governance service
rho_plastic  fresh-normalized acquisition
rho_route    item/adapter catalog selection and merge service
rho_clock    temporal-rollup demand after coalescing
```

These are diagnostic coordinates, not proved phase transitions. A load near or
above one triggers a candidate regime change and additional validation.

## 11. Lifecycle cost

The comparison unit is gross lifetime resource consumption, not only tokens at
query time:

\[
\begin{aligned}
C_{\text{gross}} ={}&
C_{\text{wake}}
+C_{\text{capture}}
+C_{\text{store}}
+C_{\text{move}}
+C_{\text{sleep}}\\
&+C_{\text{verify}}
+C_{\text{serve}}
+C_{\text{govern}}
+C_{\text{failure}}.
\end{aligned}
\]

`C_failure` includes rollback, poisoned candidate waste, recovery, and
unavailability. `C_govern` includes provenance, correction, deletion,
authorization, audit, and backup expiry. A method cannot claim lower cost by
excluding its offline construction while charging the baseline's online
retrieval. Reuse belongs in the separately normalized future-query utility;
subtracting \(V_{\text{reuse}}\) here would count it twice. Each \(C_\cdot\) is
a disjoint gross native-resource charge converted with a source-, date-,
currency-, and amortization-bound price manifest frozen before TEST. The full
native vector is always published. If any consumed resource lacks a defensible
conversion, the scalar efficiency branch is ineligible and the system reports
only constrained Pareto comparisons.

## 12. Deletion, correction, and unlearning

### 12.1 Immediate deny and eventual removal

Deletion is a multi-stage transaction:

```text
1. authorization deny/tombstone
2. quarantine in-flight descendants
3. remove online payload and identifier mapping
4. invalidate/rebuild semantic, vector, graph, and latent views
5. retire/rebuild/unlearn affected adapters
6. expire backups/keys under declared maximum
7. probe surviving KV/recurrent descendants for opaque-source residuals
8. audit residual authorization, bytes, lineage, and behavior
```

Report each stage independently. A tombstone or inability to retrieve is not
physical deletion. Nor is physical removal of source tokens proof of semantic
erasure: MEMENTO recovers injected passcode information from downstream KV
after the source block is masked and freed.

### 12.2 Shared weights

Exact item-level removal from shared weights is not assumed. Available
responses are:

- block promotion of high-delete-risk personal data;
- isolate it in a user adapter;
- quarantine/retire an affected adapter;
- rebuild from remaining canonical data;
- run a separately validated unlearning method;
- report unresolved behavioral residual.

### 12.3 Correction

Corrections preserve temporal lineage:

```text
old assertion valid interval
new assertion valid interval
source and confidence
derived artifacts affected
serving precedence
rebuild or invalidation receipt
```

Blindly overwriting a graph node or summary destroys evidence needed for
temporal queries and audit.

The same rule applies to generated skill and topic memory. Removing a visible
Markdown file is not complete deletion until its source sessions, catalog and
index descendants, generated variants, backups, and any later skills derived
from it are traversed or rebuilt. Operational “traceability” to a generated
section ID is insufficient when the source URL/span never entered the
artifact.

## 13. Security and failure model

Threats:

```text
poisoned wake events
prompt injection in replay/dream data
cross-tenant leakage
stale snapshot publication
authorization/deletion changes between validation and publication CAS
out-of-order background completion overwrites a newer turn
unbounded background-worker creation or later-wake join
partial object/index write
cross-store generation divergence
staging clear before durable multi-store publication
worker loss after write before receipt
model/index incompatibility
rollback to a deleted version
generated-data privacy leakage
implicit KV/recurrent-state leakage after explicit source eviction
judge/reward manipulation
lineage or tombstone omission
same-query benchmark leakage presented as future-query transfer
```

Controls:

- immutable inputs and content checksums;
- schema validation and typed manifests;
- tenant isolation and minimum-privilege credentials;
- sandboxed tool execution for dream/consolidation workers;
- independent canaries and red-team probes;
- one CAS over the complete parent `ServingHead`;
- immediate-deny precedence plus bounded reader leases/epoch-safe GC;
- idempotent operation tokens;
- manifest-last publication;
- signed promotion attestations;
- fault injection and recovery drills;
- no automatic shared-weight promotion for personal data in the minimum
  viable system.

## 14. Observability

### 14.1 Per memory

```text
source and derivation lineage
visible artifact plus latent-state completeness class
restart/serialization/model-migration equivalence status
reuse count and utility estimate
last validation and correction
active/archive tier
bytes and replicas
promotion/demotion history
delete/retention state
affected model/index versions
exact-key, paraphrase, composition, and correction reuse counts
```

### 14.2 Per sleep job

```text
input snapshot and output candidate digest
incumbent/candidate harness digest and complete transitive file/tool manifest
queue/start/end/publish time
scheduled/input/published logical generation and parent digest
resource vector and bytes moved
data mixture and objective
staging event range and every output sub-root
candidate-versus-incumbent metrics
validation and failure code
retry/preemption count
cancel/deadline/stale-completion disposition and next-wake wait
trajectory/diagnosis/proposal/rank counts and model identities
```

### 14.3 Fleet

```text
wake SLO and interference
sleep utilization and backlog
active/total memory growth
active/archived harness versions, executable bytes, and retained run records
promotion/rollback rate
delete debt
capacity-pressure vector
utility per resource
tenant fairness
exact-key versus unseen-semantic hit and teacher-call rates
flat-catalog selection tokens/latency versus injected-artifact tokens
```

## 15. Minimum viable system

The first credible system should stay deliberately narrow:

1. fixed shared base model;
2. per-user canonical event log;
3. external semantic/vector memory with versioned lineage;
4. optional user-isolated adapter as the only parametric destination;
5. fixed and pressure-based sleep triggers;
6. deterministic baseline plus one learned router;
7. immutable candidate, canary, atomic publish, rollback;
8. immediate deny and tested external/adapter rebuild;
9. active and total byte accounting;
10. trace-driven shared-versus-separate fleet simulator.

Shared-weight promotion, cross-user graph learning, and exact parametric
unlearning remain later extensions.

## 16. Systems experiment matrix

### 16.1 Fleet factor

```text
shared priority fleet
time-partitioned fleet
separate wake/sleep fleets
regional hybrid
```

### 16.2 Workload factor

```text
steady low novelty
bursty session close
nested session/day/week/month close
high duplicate rate
rare one-off and recurrence-threshold skew
high correction/volatility
long-tail tenants
adapter catalog growth and cross-document composition
embodied participant/video stream
privacy/delete burst
model upgrade/rebuild
poisoned event injection
```

### 16.3 State factor

```text
external only
user adapter only
external + adapter
routed/merged adapter pool + external canonical source
temporal hierarchy + raw recovery archive
active-index bounded / archive unbounded
fully bounded with expiry
```

### 16.4 Required outputs

```text
wake P99 and throughput
sleep completion/deadline distribution
per-clock requested/coalesced/completed calls
GPU/CPU/I/O/network utilization
bytes moved and stored by class
graph/index/catalog nodes, bytes touched, and rewrite amplification
adapter route/merge/cache hit, miss, load, and swap
peak resident state and memory-time AUC by explicit/implicit class
utility/retention/acquisition
internalization capability profile and rare-event survival
promotion and rollback latency
delete completion by stage
multimodal consent/tombstone fan-out
cost per useful later-wake reuse
```

## 17. Candidate systems hypotheses

These remain exploratory until frozen with exact estimands and thresholds.

1. **Separation crossover.** Separate sleep hardware dominates shared
   priority scheduling only above a sleep-load and wake-interference boundary.
2. **Delta-locality crossover.** Moving adapters/index deltas instead of raw
   histories improves cost only when source size and repeated reuse exceed
   model/residency coordination overhead.
3. **Transactional tail-risk advantage.** Snapshot-plus-canary promotion
   reduces worst-case utility loss more than it adds median lifecycle cost.
4. **Governance-aware routing.** Higher volatility/delete pressure shifts the
   destination from shared parameters toward user-local or external state.
5. **Plasticity service alarm.** Acquisition loss and free-capacity pressure
   predict the need for recycle/rebuild before old-retention canaries fail.
6. **Queue-before-memory failure.** Under high novelty arrival, backlog age
   violates freshness SLO before physical store capacity is exhausted unless
   admission adapts.
7. **Semantic-amortization gap.** Exact repeated-query teacher-call savings
   exceed savings on held-out paraphrase/composition/correction queries until
   the consolidator and router are explicitly trained for semantic transfer.
8. **Catalog-selection crossover.** A hierarchical bounded skill router beats
   prompting every skill description only beyond a catalog-size threshold
   after router-build, miss-recovery, and governance cost are charged.
9. **Generation-ordering advantage.** Generation-tagged conditional
   publication eliminates stale-overwrite failures under bursty turns at lower
   lifecycle cost than serially blocking every next wake, above a measurable
   background-work/inter-arrival boundary.
10. **Parametric-routing crossover.** Modular low-rank adapters beat one
    adapter only below a router/merge/cache-miss load boundary.
11. **Clock-count queue crossover.** More temporal rollup levels improve
    utility per active byte only until added calls and abstraction debt exceed
    available sleep service.
12. **Localized-maintenance frontier.** Region-local jobs beat global rebuilds
    above a graph-size threshold and below a cross-boundary dependency rate.
13. **Recovery-substrate floor.** Removing raw/reversible state cannot meet
    rare-event, correction, and rollback SLOs below a measurable archive
    reserve, even when the active view is smaller.

## 18. Evidence boundary

Pieces of this architecture exist separately:

- Letta demonstrates pre-query token-space compilation and explicit
  primary/sleeper roles in later products;
- Google Sleep demonstrates benchmark fast-to-slow parametric transfer and
  dream-data learning;
- HIMA demonstrates explicit external tiering and scheduled consolidation;
- PlugMem and MemMA demonstrate completed-task/session external transforms,
  with MemMA also exposing probe-guided repair and its data-lineage risk;
- Auto-Dreamer demonstrates learned offline external-memory replacement;
- MemEvolve, ALMA, MemSkill, and M★ demonstrate learned/searched memory
  architecture or control artifacts at experimental/predeployment boundaries;
- RHO demonstrates an experimental completed-trajectory-to-persistent-harness
  boundary and content-addressed versions, while exposing correlated
  self-preference, executable poisoning, search-cost, and lifetime-growth gaps;
- Surprise-Gated Memory demonstrates experimental periodic replay and
  fast-store clearing, while showing that naive recent-window rehearsal can be
  more destructive than no replay and that full replay has growing lifetime
  cost;
- Evolve demonstrates an explicit external section-memory cycle and scheduler,
  while revealing identical-query evaluation, uncapped canonical growth,
  missing source provenance, and non-atomic vector/metadata publication;
- Memento 2 and the Memento-Skills paper demonstrate persistent wake-time
  case/skill learning, while the later DreamDaemon independently demonstrates
  a code-level external sleep path without quantitative Dream evidence or a
  transactional file/index/staging commit;
- MemCompiler demonstrates learned state-conditioned text/latent delivery on
  the wake path;
- MEMENTO demonstrates learned within-query compaction and shows that
  block-conditioned KV can be both answer-relevant and secret-bearing after
  source-block eviction, so visible text and peak KV are incomplete state and
  cost boundaries;
- Zep/Graphiti demonstrates temporal graph lineage and contradiction handling;
- LoRA-as-memory demonstrates that modular parametric storage needs a catalog,
  router, compatibility-aware merge path, and serving cache;
- TiMem demonstrates multi-clock temporal rollups and their added construction
  calls; RecMem demonstrates recurrence-gated synthesis plus dependence on a
  raw recovery tier;
- GAM demonstrates that a bounded live buffer can feed growing graph/archive
  state;
- the agent-native systems study demonstrates a broad utility–latency frontier
  and motivates localized rather than universal global maintenance;
- MEMORA demonstrates strict experimental offline consolidation of embodied
  action evidence and the need for time-anchored multimodal lineage;
- continual-learning systems demonstrate replay, regularisation, isolation,
  reset, and capacity trade-offs.

The current STC corpus has not found one source that demonstrates all planes
with cross-medium routing, transactional promotion, complete retained/peak
capacity accounting, deletion propagation, and fleet scaling. Non-discovery is
not novelty evidence. Routing/gating, multi-version/atomic state, and
background merge/compaction are established adjacent primitives; the
[systems infrastructure primary-source audit](SYSTEMS-INFRA-PRIMARY-SOURCE-AUDIT.md)
maps those precedents element by element.

The bounded contribution hypothesis is therefore an STC-specific integration
and matched systems benchmark, not invention of those primitives. The linked
audit and claim chart cover selected primary sources for dynamic vector-index
maintenance, adapter serving, scheduling, and disaggregated memory, but that
bounded pass does not establish priority. Novelty remains unresolved pending
the broader database, IR, ML-systems, continual-learning, product, patent, and
implementation audit specified by that audit.
