# Sleep-Time Compute Systems Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate the lifecycle accounting, versioned memory fabric, sleep queue, deletion/rollback semantics, design-space simulator, and A100 measurement path needed to turn the sleep-time compute theory into an executable systems claim.

**Architecture:** The benchmark owns observable events, query records, method decisions, and result semantics. This systems slice owns resource accounting and the runtime that safely converts immutable wake snapshots into verified, versioned memory artifacts. Every operation emits native-unit charges; no scalar cost is formed unless a preregistered valuation is supplied. The analytical DSE, trace simulation, and measured accelerator paths produce distinct evidence classes and immutable bundles.

**Tech Stack:** Python 3.14, uv, pytest, Hypothesis, Pydantic 2, NumPy, SciPy, PyYAML, PyTorch for the accelerator-only extra, NVIDIA Management Library/DCGM, the CUDA CUPTI Profiling API for A100 HBM counters, Linux `perf_event_open` uncore-IMC counters for host DRAM, Linux cgroups and `/proc`, Docker, `torchrun`, NCCL, NFS, InfiniBand.

## Ownership and Boundaries

- Package root is `research/sleep-time-compute/`; all paths below are relative to it unless stated otherwise.
- `stc_research.benchmark.schemas` owns `ObservableEvent`, `QueryObservable`, and `RunResult`.
- `stc_research.methods.base` owns the method-facing compiler and policy protocols.
- `stc_research.systems.schemas` is the sole definition site for lifecycle phase, resource, energy, storage, and queue records.
- `RunResult.cost_summary` imports `LifecycleCostSummary`; it never introduces a second accounting type.
- Systems code never edits evidence, claims, manuscripts, or the presentation handoff. It emits bundles that the artifact DAG registers.
- `TRACE-SIMULATION`, `ANALYTIC-DERIVATION`, `ORIGINAL-MEASUREMENT`, and
  `INDEPENDENT-REPLICATION` remain distinct. `SYNTHETIC` is a test-only class
  that can never support a research claim.
- Raw archives, lineage, indexes, optimizer state, checkpoints, and active memory are separate storage classes. A reduction in active memory is never described as total-storage reduction.
- GPU-board, host, and modeled storage/network energy stay separate. They are not summed into an empirical “total energy” without a frozen boundary and calibration.
- Immediate deny records override already pinned content generations. One
  linearizable `ServingHead` CAS word binds generation, manifest digest,
  authorization epoch, and deletion epoch; publication, rollback,
  authorization changes, and deletion all compare-and-swap that exact head.
  Safe physical reclamation additionally requires reachability closure and
  acknowledged reader quiescence/hazard release or proven worker fencing.
- The existing `runbook/hope-reproduction-a100.md` and existing `experiments/` tree are read-only historical inputs.

## Command-Root Contract

Every `bash` block starts in a fresh shell at
`research/sleep-time-compute/`, unless that block itself resolves the
repository root and changes directory. No block inherits a prior working
directory or shell variable. Cluster blocks revalidate explicitly named
operator inputs such as `STC_BOOTSTRAP_ROOT` and `STC_NODE_RANK`, then source
the checksum-verified environment in that same block. Repository-level Git
commands use `git -C ../..`.

## Shared Runtime Contract

Create `src/stc_research/runtime/contracts.py` with:

```python
class MethodRuntime(Protocol):
    def answer(
        self, query: QueryObservable, view: PublishedView
    ) -> AnswerOutcome: ...

    def plan_sleep(
        self,
        events: Sequence[ObservableEvent],
        view: PublishedView,
        cutoff: datetime,
    ) -> Sequence[ArtifactProposal]: ...

    def materialize(
        self,
        proposal: ArtifactProposal,
        snapshot: SnapshotRef,
        candidate_dir: Path,
    ) -> ShadowArtifact: ...

    def validate(
        self,
        shadow: ShadowArtifact,
        snapshot: SnapshotRef,
    ) -> VerificationReport: ...
```

`QueryTarget`, `OracleAnnotation`, later outcomes, and hindsight utility are not
accepted by this protocol. Type checks and runtime guards fail closed if an
oracle-side object crosses the boundary.

## Command Surface

The shared `src/stc_research/cli.py` exposes exactly:

```text
stc systems dse
stc systems deletion-audit
stc systems index-refresh materialize
stc systems index-refresh run
stc systems report
stc runtime lifecycle
stc runtime run-matrix
stc runtime import-matrix
stc runtime freeze-contract
stc accelerator probe
stc accelerator reconcile-probes
stc accelerator run
stc accelerator validate
stc accelerator runbook-smoke
stc prereg validate
stc results validate
stc gate verify-chain
stc gate inputs assemble
stc gate evaluate
stc dag import-bundles
stc dag check
stc validate
```

This plan owns the `systems`, `runtime`, and `accelerator` commands and modifies
`cli.py` plus `tests/test_cli.py` in the introducing task. The preregistration
and result validators are theory-owned; gate, DAG, and generic `validate`
commands are evidence/program-owned. They are listed because this plan invokes their exact
public contracts, but this plan does not reimplement them. The master plan uses
no alias outside this table.

## Generated Bundle Contract

```text
results/system-runs/{run_id}/
  run-spec.json
  metrics.jsonl
  lifecycle-cost.json
  manifest.json

results/a100-runs/{run_id}/
  probes/
    node-0.json
    node-1.json
    cluster.json
  raw-metrics/
    rank-0000.jsonl ... rank-0007.jsonl
  rank-manifests/
    rank-0000.json ... rank-0007.json
  host-counters/
    node-0.jsonl
    node-1.jsonl
  host-manifests/
    node-0.json
    node-1.json
  summary.json
  manifest.json

results/{confirmatory-primary,confirmatory-clean-rerun,scaling}/
  workers/rank-{global_rank:04d}/children/{child_key}/attempt-{attempt_id}/
  logical-references.jsonl
  manifest.json

manifests/systems/runs/{run_id}.json
manifests/accelerator/runs/{run_id}.json
```

The three post-G4 matrix roots are canonical imported experiment trees, not
single local `results/system-runs/{run_id}` bundles. Each has the rank/child
tree, complete logical-to-physical reference map, and one top-level
manifest-last inventory described in Task 6. Their typed distributed receipts
make them valid inputs to `dag import-bundles`; no code treats them as the
four-file local system-run layout.

Every manifest records schema version, producer command, Git revision, resolved
configuration digest, environment/container digest, input digests, output
digests, evidence class, measurement boundary, per-metric provenance, and
measurement admissibility. Accelerator manifests distinguish
`SYNTHETIC|PARTIAL|ADMISSIBLE`. The mapping is exact:
`SYNTHETIC -> evidence_class=SYNTHETIC`,
`PARTIAL -> evidence_class=null, claim_admissible=false`, and
`ADMISSIBLE -> evidence_class=ORIGINAL-MEASUREMENT`. Generated payloads are
ignored by Git; their small manifests and checksums are retained under the two
`manifests/` paths above.

`stc dag import-bundles` accepts either one bundle directory or a directory
whose immediate children are bundles. It revalidates every payload digest,
copies only each small manifest into the class-appropriate manifest directory,
and rejects an overwrite, duplicate run ID with different bytes, missing
payload, or evidence-class/measurement-status upgrade.

---

### Task 1: Implement native-unit lifecycle accounting

**Files:**
- Create: `src/stc_research/systems/__init__.py`
- Create: `src/stc_research/systems/schemas.py`
- Create: `src/stc_research/systems/accounting.py`
- Create: `src/stc_research/runtime/__init__.py`
- Create: `src/stc_research/runtime/contracts.py`
- Create: `configs/systems/storage-accounting.yaml`
- Create: `tests/systems/test_accounting.py`
- Create: `tests/systems/test_storage_telemetry.py`
- Create: `tests/runtime/test_contracts.py`

**Interfaces:**
- Produces: `LifecyclePhase`, `NativeResourceTotals`, `EnergyBreakdown`,
  `StorageBreakdown`, `LatencyBreakdown`, `ThroughputSummary`,
  `FreshnessSummary`, `ResidencySummary`, `BudgetUtilization`,
  `FeasibilityRecord`, `LifecycleCharge`, `LifecycleCostSummary`, and the
  neutral `MethodRuntime`/artifact/view protocol used by theory methods.
- Consumed by: benchmark results, queue simulation, runtime, accelerator
  harness, statistics, and scaling analysis.

- [ ] **Step 1: Write the failing conservation tests**

Test these exact invariants:

```text
sum(per_phase.native_resources) == total.native_resources
gpu_board_j, host_j, storage_network_modeled_j remain separate
active, durable, index, lineage, latent, optimizer_checkpoint bytes remain separate
B_retained == B_r + B_e + B_l + B_p + B_a + B_v
B_peak_state(t) == B_retained(t) + B_candidate(t) + B_pinned(t)
             + B_migration(t) + B_workspace(t) + B_replica(t)
incremental_peak_state_bytes == max_t B_peak_state(t)
physical_peak_bytes ==
  max_t [B_base_resident(t) + B_peak_state(t) + B_execution(t)]
physical_peak_components_at_t_star sum exactly to physical_peak_bytes
independent base/state/execution component maxima are diagnostics, not summands
transient_overhead_peak_bytes == max_t(B_peak_state(t) - B_retained(t))
every physical object has exactly one ownership class, cap, and GC owner
ledger high-water sampling reconciles with allocator/storage telemetry
zero future reuse never divides by zero
JSON round-trip preserves units and phase labels
raw archive bytes remain charged when active memory is compacted
```

Also reject negative quantities, unknown units, nonfinite values, and a charge
whose end timestamp precedes its start.

- [ ] **Step 2: Run the targeted test and confirm failure**

```bash
uv run pytest tests/systems/test_accounting.py tests/runtime/test_contracts.py -q
```

Expected: import failure because the systems package does not exist.

- [ ] **Step 3: Implement immutable schemas and rollups**

Use frozen records. `NativeResourceTotals` contains:

```text
input_tokens
output_tokens
training_tokens
flops
gpu_seconds
cpu_seconds
hbm_read_bytes
hbm_write_bytes
host_dram_read_bytes
host_dram_write_bytes
disk_read_bytes
disk_write_bytes
network_read_bytes
network_write_bytes
```

`LifecyclePhase` is exactly:

```text
WAKE, QUEUE, SLEEP, VERIFY, PUBLISH, ROLLBACK, DELETE
```

`StorageBreakdown` is not a bag of overlapping labels. Its retained partition
has exactly six mutually exclusive ownership classes:

```text
B_r raw/canonical evidence
B_e external text/vector/graph payload and index
B_l latent/KV state
B_p parametric modules
B_a auxiliary catalog/router/optimizer state
B_v version/lineage/recovery state
```

Each object-level `StorageObjectCharge` binds an opaque physical-object handle,
exactly one retained or transient ownership class, physical byte count, hard
cap, named reclamation owner, allocation/retire epochs, and replica identity.
Content-addressed aliases do not duplicate a physical charge; physical copies
do. The five disjoint transient additions are `candidate`, `pinned`,
`migration`, `workspace`, and `replica`; incremental peak state adds them to
retained state. `hot`, `active`, `warm`, `cold`, and `archive` are
separate overlapping projection tags and never enter a sum.

`LifecycleCostSummary` includes totals, per-phase totals,
`retained_storage_bytes`, the typed retained partition,
`incremental_peak_state_bytes`, all five typed transient additions,
`transient_overhead_peak_bytes`, `physical_peak_timestamp`,
`base_model_resident_bytes_at_physical_peak`,
`incremental_state_bytes_at_physical_peak`,
`execution_bytes_at_physical_peak`, `physical_peak_bytes`, independent
component-peak diagnostics, final storage,
wake latency p50/p95/p99, queue age p50/p95/p99, deadline misses, dropped jobs,
and the three energy boundaries. Quantiles over empty collections serialize as
`null`, not zero.

The scalar base/state/execution fields at `physical_peak_timestamp` are aligned
samples whose sum equals `physical_peak_bytes`; independent component
high-waters are separately named diagnostics and are never summed as an
identity. `storage-accounting.yaml` freezes event-driven high-water sampling, periodic
sampling cadence, allocator/storage telemetry sources, byte rounding, and the
maximum reconciliation error. Before G4,
`test_storage_telemetry.py` injects allocation/free races, noncoincident
component maxima, and short-lived
spikes and requires the ledger high-water plus frozen calibration bound to
cover allocator/storage high-water while absolute reconciliation error stays
within `max(4096 bytes, 1%)`. A missing telemetry source, duplicate ownership,
unowned transient object, cap exceedance, or undercount outside that bound is a
typed G4 failure, not a null metric.

It also contains typed, non-collapsed outcomes for TTFT; retrieval, queue,
prefill, decode, writeback, publish, and adapter-swap latency; accepted
requests/s and tokens/s; SLO violations; cache/adapter hit and miss; MFU and
achieved HBM bandwidth; HBM residency/offload; freshness/staleness; unused
resource caps and feasibility masks; and economic cost per useful answer only
when the frozen G2 cost profile is complete. Missing measurements serialize as
`null` with a reason, never zero.

Implement `runtime/contracts.py` with `from __future__ import annotations` and
`TYPE_CHECKING` imports for benchmark records so importing the neutral protocol
does not import the benchmark implementation. This module owns
`ArtifactProposal`, `PublishedView`, `SnapshotRef`, `ShadowArtifact`,
`VerificationReport`, and `AnswerOutcome`; benchmark schemas and method
implementations import them rather than redefining them.

- [ ] **Step 4: Verify, format, and commit**

```bash
uv run pytest tests/systems/test_accounting.py \
  tests/systems/test_storage_telemetry.py tests/runtime/test_contracts.py -q
uv run ruff check src/stc_research/systems src/stc_research/runtime/contracts.py \
  tests/systems tests/runtime/test_contracts.py
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: add lifecycle resource accounting"
```

Expected: all accounting tests pass and Ruff reports no errors.

---

### Task 2: Build the versioned memory fabric

**Files:**
- Create: `src/stc_research/runtime/versioning.py`
- Create: `src/stc_research/runtime/memory_fabric.py`
- Create: `src/stc_research/runtime/reference_store.py`
- Create: `tests/runtime/test_versioning.py`

**Interfaces:**
- Consumes: contract-owned `PublishedView`, `SnapshotRef`, and `ShadowArtifact`
  from `runtime/contracts.py`; `versioning.py` and `memory_fabric.py` neither
  define nor re-export competing records with those names.
- Produces: `ArtifactManifest`, `ShadowCandidate`, `ReadyCandidate`,
  `ServingHead`, `ReadCapability`, `PublishOutcome`, `ReachabilityRootSet`,
  `ReclamationDecision`, and `MemoryFabric`.
- Requires: opaque tenant handles and benchmark/method contracts.

- [ ] **Step 1: Write failure-first publication tests**

Cover:

```python
MemoryFabric.load_head(tenant_handle) -> ServingHead
MemoryFabric.pin_read(auth, tenant_handle) -> ReadCapability
MemoryFabric.snapshot(tenant_handle, information_cutoff) -> SnapshotRef
MemoryFabric.stage(snapshot, artifacts) -> ShadowCandidate
MemoryFabric.mark_ready(candidate, verification) -> ReadyCandidate
MemoryFabric.publish(
    candidate,
    expected_head: ServingHead,
) -> PublishOutcome
MemoryFabric.advance_authorization(
    tenant_handle, expected_head: ServingHead, new_deny_root
) -> ServingHead
MemoryFabric.delete(
    tenant_handle, expected_head: ServingHead, deletion_request
) -> ServingHead
MemoryFabric.rollback(
    tenant_handle, expected_head: ServingHead,
    authorized_predecessor_memory_roots, reason
) -> ServingHead
MemoryFabric.reclaim(
    object_handle, root_set: ReachabilityRootSet, retire_epoch
) -> ReclamationDecision
```

Assert that:

- readers observe one complete immutable generation;
- a candidate is invisible until every artifact and manifest digest validates;
- manifest publication happens last;
- `ServingHead` atomically binds generation, manifest digest, authorization
  epoch, deletion epoch, and deny-root digest;
- any stale field in the exact expected head rejects publication, rollback,
  authorization change, or deletion;
- publication revalidates the exact head after candidate verification, so a
  delete/authorization change before validation, between validation and CAS,
  or after CAS has one linearizable outcome;
- immediate deny is checked before every protected artifact read and before
  answer emission, even when content is pinned to an older generation;
- retrying the same completed operation is idempotent;
- base-model or tokenizer mismatch invalidates latent/KV and parametric
  artifacts;
- rollback can reuse only authorized last-known-good predecessor **memory
  roots**. It publishes a new manifest/head with monotonically advanced
  generation/epochs and the current authorization/deletion/deny root; it never
  reinstalls an old full `ServingHead` or regresses governance state;
- every read registers a renewable bounded lease and non-bypassable
  epoch/hazard slot for its exact head;
- lease timeout requests cancellation but never by itself proves quiescence or
  permits physical free;
- GC traverses the reachability closure of current and pinned heads,
  in-flight candidate/migration roots, rollback roots, backups, legal holds,
  and deletion-retention roots, including shared objects across versions;
- an unreachable retired object is reclaimed only after every capable reader
  acknowledges post-retire quiescence/releases its hazard, or after an
  isolation boundary proves the old capability fenced and unable to execute;
- a stalled unacknowledged reader causes backpressure/fencing, and a resumed
  reader must acquire a current capability before another dereference.

`ArtifactManifest` records opaque tenant-scoped artifact/source/parent handles,
snapshot version, generation, base-model and tokenizer hashes, compiler
name/version/config digest, artifact class, privacy scope, deletion epoch,
authorization epoch, encrypted object location, integrity value, byte counts,
readiness, and lineage
edges. Operational metadata never stores a raw-payload or reversible content
hash. Integrity uses a tenant-keyed MAC or a digest of encrypted/packed artifact
bytes; the separately encrypted deletable sidecar owns any payload mapping.

- [ ] **Step 2: Confirm failure and implement a filesystem reference store**

```bash
uv run pytest tests/runtime/test_versioning.py -q
```

Use temporary directories, atomic rename on one filesystem, tenant-keyed or
encrypted-object integrity values, and one compare-and-swap `ServingHead`
record per tenant. Candidate sub-roots are immutable and are served only
through the manifest digest in that head. Deny lookup is a separate
immediate-read path whose digest/epochs advance through the same head CAS.
Implement explicit `ReadCapability` acquisition, renewal, hazard release,
quiescence acknowledgement, worker fencing, root-set construction, and
reachability-aware reclamation. Do not claim the reference store implements a
distributed consensus protocol.

- [ ] **Step 3: Run property tests**

Generate arbitrary valid stage/ready/publish/authorization/delete/rollback/read
/cancel/fence/reclaim sequences. The invariants are that a pinned reader sees
either the old or new complete content generation, never a partial mixture;
current deny always precedes further protected reads/emission; and no object
reachable from any declared root or possibly executable reader is reclaimed.
The state-machine suite includes deterministic schedules for delete before
validation, between validation and head CAS, after CAS, and during a pinned
read; delete immediately before and during rollback; plus shared-object
reclamation before/after quiescence and worker fencing.

```bash
uv run pytest tests/runtime/test_versioning.py -q
```

Expected: all deterministic linearization/reclamation schedules and randomized
operation sequences pass; a generation-only CAS, lease-timeout free, or
minimum-generation-only GC mutant is killed.

- [ ] **Step 4: Commit**

```bash
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: implement versioned memory publication"
```

---

### Task 3: Implement wake, sleep, memory, and control planes

**Files:**
- Create: `src/stc_research/runtime/wake.py`
- Create: `src/stc_research/runtime/sleep.py`
- Create: `src/stc_research/runtime/control.py`
- Create: `src/stc_research/runtime/coordinator.py`
- Create: `tests/runtime/test_planes.py`

**Interfaces:**
- Wake path: authorize against current `ServingHead` → acquire
  `ReadCapability`/hazard → recheck immediate deny before each protected read
  and emission → answer → acknowledge quiescence/release hazard → append
  observable event → charge latency.
- Sleep path: freeze cutoff → snapshot → plan → materialize shadow → verify →
  canary → exact-`ServingHead` CAS publish.
- Control path: admission, cadence, deadline, tier movement, overload fallback.

- [ ] **Step 1: Write boundary and cutoff tests**

Test:

- an answer call receives `QueryObservable` and a pinned `PublishedView`;
- all four inline/deferred × identity/consolidating cells share the exact
  enqueue-time cutoff digest;
- inline and deferred differ only in scheduling;
- identity and consolidating differ only in operator semantics;
- no oracle record is importable through the runtime input protocol;
- failed validation leaves the current head unchanged;
- a mid-answer authorization/delete advance blocks the next protected
  dereference or emission while retaining the old object until reader
  quiescence/fencing;
- a late sleep worker whose expected head is stale returns
  `STALE_COMPLETION` and never relabels or publishes its candidate.

- [ ] **Step 2: Implement the four planes**

`control.py` records every admission, deferral, downgrade, drop, and deadline
decision with observable features and the decision-time cutoff. An overload
path must delay, choose a cheaper registered operator, or drop a low-utility job
instead of allowing an unbounded queue.

- [ ] **Step 3: Verify and commit**

```bash
uv run pytest tests/runtime/test_planes.py -q
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: add wake and sleep runtime planes"
```

Expected: at least seven plane tests pass.

---

### Task 4: Implement the resource-vector queue simulator and DSE

**Files:**
- Create: `src/stc_research/systems/queue.py`
- Create: `src/stc_research/systems/dse.py`
- Modify: `src/stc_research/cli.py`
- Modify: `tests/test_cli.py`
- Create: `configs/systems/dse-smoke.yaml`
- Create: `tests/systems/test_queue.py`
- Create: `tests/systems/test_dse.py`

**Interfaces:**
- Produces: `ResourceDemand`, `ResourceCapacity`, `SleepJob`,
  `QueueTraceRecord`, and `QueueSimulationResult`.
- Consumes: service-time/resource distributions calibrated by derivation,
  trace, or measurement with evidence class retained.

- [ ] **Step 1: Write deterministic scheduling tests**

Cover setup plus per-event service, batching, tenant affinity, priority,
preemption, deadlines, and simultaneous GPU/CPU/storage-I/O/network caps.
Assert that wake work preempts best-effort sleep work under pressure.
Freeze the event ordering rule as:

```text
(timestamp, event_priority, tenant_handle, job_id)
```

Use a local seeded generator; global random state is forbidden.

- [ ] **Step 2: Implement the discrete-event simulator**

The simulator reports utilization by resource, queue-age distribution,
deadline-miss rate, wake p99 impact, completed/delayed/downgraded/dropped jobs,
and every mitigation. A cell is stable only if the post-warmup queue has no
positive linear-growth slope within its preregistered tolerance.

The queue configuration schema freezes simulation horizon, warm-up,
post-warmup fit window, Theil–Sen slope estimator, block-bootstrap uncertainty,
slope tolerance, p99-age SLO, deadline-miss threshold, drain-time limit,
arrival/service distributions, and resource capacities. Classification first
checks analytical offered load for each resource, then finite-horizon slope
with its confidence interval, then drain time. A finite trace with a small
observed queue is not called stable when offered load or the upper slope bound
violates the frozen rule.

- [ ] **Step 3: Add an eight-cell DSE smoke fixture**

The fixture contains four stable low-arrival cells and four overloaded cells.
Every overloaded cell must record at least one mitigation.

```bash
uv run pytest tests/systems/test_queue.py tests/systems/test_dse.py -q
uv run stc systems dse \
  --config configs/systems/dse-smoke.yaml \
  --output /tmp/stc-dse-smoke
sha256sum /tmp/stc-dse-smoke/manifest.json
uv run stc systems dse \
  --config configs/systems/dse-smoke.yaml \
  --output /tmp/stc-dse-smoke-rerun
sha256sum /tmp/stc-dse-smoke-rerun/manifest.json
cmp /tmp/stc-dse-smoke/metrics.jsonl \
  /tmp/stc-dse-smoke-rerun/metrics.jsonl
cmp /tmp/stc-dse-smoke/lifecycle-cost.json \
  /tmp/stc-dse-smoke-rerun/lifecycle-cost.json
```

Expected: all tests pass, semantic payloads are byte-identical, both manifests
carry the same payload digest despite their different output roots, and the CLI reports
`stable_cells=4 overloaded_cells=4 unmitigated_overload=0`.

- [ ] **Step 4: Commit**

```bash
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: add sleep queue design-space simulator"
```

---

### Task 5: Implement deletion, rollback, and fault injection

**Files:**
- Create: `src/stc_research/runtime/deletion.py`
- Create: `src/stc_research/runtime/failures.py`
- Modify: `src/stc_research/cli.py`
- Modify: `tests/test_cli.py`
- Create: `configs/systems/deletion-profile.yaml`
- Create: `tests/runtime/test_deletion.py`
- Create: `tests/runtime/test_reclamation.py`
- Create: `tests/runtime/test_deletion_audit.py`
- Create: `tests/runtime/test_failures.py`
- Create: `tests/runtime/fixtures/deletion-audit/*`

**Interfaces:**
- Produces: `DeletionRequest`, `DeletionReport`, deny records, lineage
  invalidation, candidate quarantine, and typed fault scenarios.
- `stc systems deletion-audit` emits one system-run bundle containing both the
  deletion conformance dimensions and the frozen fault-injection outcomes used
  by G4; neither is recoverable only from prose or a test exit code.

- [ ] **Step 1: Write deletion conformance tests**

Test access denial, online removal, backup/key expiry, lineage invalidation,
adapter rebuild/retirement, and behavioral residual as independently reported
dimensions. An old pin must stop returning newly denied content before physical
cleanup completes.

The final tombstone tuple is exactly a random deletion-transaction handle,
coarse time bucket, policy/result code, artifact-class counts, and proof
digest. Reject tenant, user, source, raw locator, content hash, embedding
fingerprint, or any reusable cross-system join key.

- [ ] **Step 2: Write fault-injection tests**

Inject:

```text
worker death
truncated object
stale exact-ServingHead CAS
corrupt index
storage-node loss
poison/canary failure
deletion before candidate validation
deletion between candidate validation and head CAS
deletion after successful publication CAS
deletion during a pinned read
reclaim attempt before reader quiescence
lease expiry without hazard release
worker fence followed by capability replay
shared object reachable from another live/rollback/backup root
duplicate event delivery
lost event before acknowledgement
```

Assert no corrupt/truncated candidate publishes, retries are idempotent,
rollback does not revive denied material, immediate deny precedes further
protected reads/emission, no reachable or possibly reader-accessible object is
freed, and RTO/RPO are measured rather than assumed.

- [ ] **Step 3: Implement deny-first deletion and failure harness**

Lineage traversal must cover raw, abstract, graph/index, latent/KV,
user-parametric, optimizer, checkpoint, cache, and candidate artifacts.
Separate cryptographic erasure from physical deletion and behavioral
unlearning; do not treat one as proof of the others.

Reclamation constructs one `ReachabilityRootSet` from the current head, all
live pinned heads, in-flight candidates and migrations, rollback roots,
backups, legal holds, and deletion-retention roots, then traverses immutable
sub-root sharing. An object outside the closure is still not freeable until
every reader that could have acquired it acknowledges a post-retire quiescent
state/releases its hazard, or a tested isolation boundary fences that worker
and rejects replay of its old capability. Lease expiry only requests
cancellation. The audit reports roots, closure digest, retire epoch, reader
acknowledgements/fence proofs, reclaim owner, and outcome for every candidate.

`deletion-profile.yaml` freezes ledger retention, maximum backup retention,
key-destruction latency, legal-hold states, declared replica/storage classes,
RTO, RPO, duplicate/lost-event tolerance, maximum stale read, and the
behavioral-residual metric/threshold. Legal hold is a separate non-deleted
outcome. The audit inventories every replica, replays lineage, tests access
denial, scans retained metadata for identifiers/content fingerprints, and runs
a linkability attack with every still-retained key. Physical-removal and
behavioral-unlearning claims pass or fail independently.

- [ ] **Step 4: Verify and commit**

```bash
uv run pytest \
  tests/runtime/test_deletion.py \
  tests/runtime/test_reclamation.py \
  tests/runtime/test_deletion_audit.py \
  tests/runtime/test_failures.py -q
uv run stc systems deletion-audit \
  --profile configs/systems/deletion-profile.yaml \
  --fixture tests/runtime/fixtures/deletion-audit \
  --output /tmp/stc-deletion-audit
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: validate deletion rollback and runtime failures"
```

Expected: deletion, reclamation, deletion-audit, and failure tests pass; the
audit reports each completion dimension, every named injected-fault outcome,
every root/reader reclamation proof, and zero undeclared replica, unsafe free,
or retained linkable key.

---

### Task 6: Run the four-cell lifecycle integration protocol

**Files:**
- Create: `configs/systems/lifecycle-smoke.yaml`
- Create: `configs/systems/index-refresh.yaml`
- Create: `configs/systems/capacity-production-phases.yaml`
- Create: `schemas/runtime/capacity-production-phases-v1.schema.json`
- Modify: `src/stc_research/cli.py`
- Create: `src/stc_research/runtime/index_refresh.py`
- Modify: `tests/test_cli.py`
- Create: `tests/runtime/test_lifecycle_integration.py`
- Create: `tests/runtime/test_index_refresh.py`
- Create: `tests/runtime/test_manifest_runner.py`
- Create: `tests/runtime/test_distributed_manifest_runner.py`
- Create: `tests/runtime/test_matrix_import.py`
- Create: `tests/runtime/test_capacity_phase_plan.py`
- Create: `tests/runtime/fixtures/manifest-smoke.jsonl`
- Create: `manifests/systems/runtime-G2.json`
- Create: `manifests/systems/index-refresh-G2.json`
- Modify: `src/stc_research/runtime/coordinator.py`

**Interfaces:**
- Consumes: benchmark fixture, one frozen destination, accounting, memory
  fabric, queue, and runtime.
- Produces: four immutable lifecycle bundles plus a G2-bound, five-arm
  representation-refresh protocol.

- [ ] **Step 1: Freeze the exact 2×2 contract**

The four cells are:

```text
inline × identity
deferred × identity
inline × semantic consolidation
deferred × semantic consolidation
```

Use the same events, destination, enqueue-time information cutoff, caps,
deadline, seed, and probe queries. The identity operator validates and
republishes a byte-equivalent record; semantic consolidation may change its
representation.

- [ ] **Step 2: Execute and validate**

```bash
uv run stc runtime lifecycle \
  --config configs/systems/lifecycle-smoke.yaml \
  --output /tmp/stc-lifecycle-smoke
uv run pytest tests/runtime/test_lifecycle_integration.py -q
```

Expected:

```text
logical_cells=4
cutoff_digest_count=1
unvalidated_publications=0
charge_conservation_errors=0
bundle_digest_errors=0
```

At least five integration tests pass.

- [ ] **Step 2A: Freeze and execute the representation-refresh protocol**

This is a `SYSTEMS_CHARACTERIZATION`/conformance track, not a powered
confirmatory scaling claim. Freeze active-population sizes
`{1K,2K,4K,8K,16K,32K,64K,128K,256K,512K}`; the first eight may fit
descriptive curves and the last two are untouched extrapolation stress sizes.
Run five deterministic paired trace seeds for each of three frozen workload
traces (uniform change, locality-skewed change, and delete-heavy change) in
these five arms:

```text
fixed encoder + local changed-key update
versioned encoder/index federation
query-time old→new representation translation
dual-index background migration
single-current-encoder full replacement
```

All arms receive byte-identical payload changes, incompatible-encoder events
where applicable, query/probe order, delete/correction trace, quality floor,
freshness deadline, hardware allocation, and failure policy. The manifest
freezes the declared affected-key set
\(\mathcal K_c^{\mathrm{aff}}\) before execution. Instrument:

```text
encoding/publication item operations
retained and incremental peak-state bytes by disjoint ownership class
resident frozen-base copies, execution bytes, and physical HBM/fleet peak
resident encoder/index versions
query fan-out, translation and calibration work
migration bytes and completion lag
delete fan-out and residual checks
retrieval quality, p50/p95/p99 query latency, freshness misses
```

The fixed-encoder arm's affected set is only changed payloads. Whole-store
work is a lower-bound candidate only for the single-current full-replacement
arm with no federation or translation. Federation, translation, and dual-index
arms must charge displaced encoder/index residency, query fan-out/calibration,
migration, peak overlap, and deletion fan-out; hiding any term invalidates that
arm. Every arm must meet the same retrieval-quality and deletion-residual
constraints before cost comparison.

```bash
uv run stc systems index-refresh materialize \
  --config configs/systems/index-refresh.yaml \
  --output manifests/systems/index-refresh-G2.json
uv run stc systems index-refresh run \
  --manifest manifests/systems/index-refresh-G2.json \
  --output /tmp/stc-index-refresh
uv run pytest tests/runtime/test_index_refresh.py -q
```

Tests recompute \(\sum_c\sum_{i\in\mathcal K_c^{\mathrm{aff}}}e_{ic}\), reject
an undeclared whole-store affected set, inject an encoder incompatibility,
exercise deletes during migration, and fail every arm that omits residency,
fan-out, translation, migration, peak, or delete cost. A held-out size reversal
or quality/deletion failure is a reported characterization boundary; no
alternative is declared universally optimal. The manifest/result bundle carry
`evidence_role=SYSTEMS_CHARACTERIZATION` and
`claim_admissibility=CONFORMANCE_ONLY`. G4 may use them to establish
accounting, atomicity, quality-floor, and deletion conformance, but not a
slope, universal lower bound, scaling law, or comparative superiority.
Promotion to a scientific claim requires a separate pre-TEST amendment with
variance pilot, power, exact estimand/model, prediction tolerance,
multiplicity, held-out decision, and result-analysis contract.

- [ ] **Step 3: Implement the general manifest runner**

`stc runtime run-matrix` reads a frozen logical manifest, resolves the method
registry, and executes every cell through the same wake/sleep coordinator,
versioned fabric, accounting hooks, exact-`ServingHead` publication and
deny/reclamation protocol, and bundle writer used
by the four-cell smoke. It never invokes a method-only shortcut. Fixture mode
may consume a standalone test manifest. There are exactly three
G1-descended, pre-TEST calibration authorizations:

- `scope=core` permits only `--mode training-calibration` and the typed
  TRAIN-only pilot;
- `scope=capacity-selection` permits only
  `--mode capacity-calibration --calibration-role selection`;
- `scope=capacity-power` permits only
  `--mode capacity-calibration --calibration-role power`.

Each requires its exact readiness, development budget, TRAIN/CAL manifest and
cohort index. Scope/role mismatch, any TEST URI/ID/derivative/capability, or
unit/seed/derivative overlap between capacity selection and power cohorts is a
prelaunch failure.

Core confirmatory production requires only the core
`--preregistration`, PASS G2 chain, and
`--execution-snapshot manifests/stc/g4-confirmatory-execution.json`, plus
`--budget manifests/stc/confirmatory-budget.json` and the closed
`--budget-role primary|clean`. The primary role can debit only
`Q_core,primary`; the clean role can debit only `Q_core,clean`, and neither can
borrow or reduce the powered cohort. Capacity
production (`scaling`, `coverage`, `parametric-information`) instead requires
`--capacity-preregistration`, PASS
`--capacity-gate manifests/capacity-gates/G2-CAP.json`, and
`--execution-snapshot manifests/stc/g4-capacity-execution.json`. The capacity
snapshot binds PASS core G4/G2, G4 inputs, both preregistrations, G2-CAP,
capacity stakes/configs/powers/manifests/cohort indexes/subbudgets, aggregate
budget, both core/capacity selection bundles, disjoint power-calibration
receipt/validation/summary, checkout, target environment, and the exact
CAL→TEST transport predicate. It contains no confirmatory outcome, analysis,
primary receipt, or clean-rerun receipt. A core and capacity snapshot are
separate typed documents; neither may substitute for the other.

Core confirmatory launch order is not manifest row order or a child-key hash.
The runner consumes the final run index's frozen randomized-complete-block/
Williams schedule and enforces its node/rank/wave/thermal/cache assignments,
order seed/digest, and balance tolerances. Core preregistration, G2, the core
G4 snapshot, every shard manifest, and the distributed receipt bind those same
bytes. Hybrid-all-last, one method confined to a rank/time block, or an order
seed drift fails before launch; a file-row permutation with identical frozen
schedule is invariant.

Every capacity mode requires `--budget`, `--aggregate-budget`,
`--actual-ledger results/capacity/actual-rre-ledger.jsonl`, and the node-0
canonical budget coordinator. It computes
dominant-resource RRE from the frozen native reference vector, rejects a
missing counter, and enforces both the track quota, sponsor-frozen
\(Q_{\rm capacity,max}\), and every absolute native/carbon ceiling
before reservation, launch, resume, verify, and merge. Node 0 durably appends
one hash-chained reservation to
`results/capacity/actual-rre-ledger.jsonl` and atomically replaces
`results/capacity/actual-rre-ledger.head.json` with the new row count and
terminal digest before
issuing a cohort token; v1 forbids concurrent track reservations. Admission
uses `completed_actual + all_unreconciled_reservations + next_reservation`,
not completed actual alone. An abandoned reservation remains charged until
typed no-launch reconciliation; actual above reservation is a system-safety
failure and halts the program. Native cgroup/device/I/O/network caps include a
frozen counter-latency margin. Track ledgers are deterministic views only.
Tests crash between reservation and launch, race two stale reservations, delay
counters, exceed a reservation, omit retry/CPU/I/O work, reuse a token, and
drift a derived view.

Scaling also requires `--scaling-config` and `--hardware-config`; coverage and
information require their exact G2-CAP configs. Scaling and coverage expose
the closed CLI enum `--execution-role fit|holdout`; scaling maps it exactly to
manifest enum `a100_fit|a100_holdout`. Hyphenated manifest aliases,
underscored CLI aliases, mixed roles, or a role mismatch fail parsing. A
holdout role requires the matching immutable provisional artifact and
one-time digest-bound `--holdout-unlock`; the runner proves no holdout result
was readable or schedulable before the seal. Post-unlock anchor reruns are
score/control only and cannot refit. Parametric information has
`--execution-role all` only and verifies fresh process/base, unique child
payload sub-seed under its master seed family, frozen decoder, and
complete-state inventory for every child.
The `confirmatory-clean-rerun` run ID additionally requires
`--rerun-request manifests/stc/confirmatory-clean-rerun-request.json`. That
typed evidence-owned request binds the validated primary receipt/result tree,
analysis preparation, requested checkout, isolation policy, and exact
confirmatory inputs. The runner rejects a missing, stale, pre-primary, or
already-consumed request, and the clean-rerun receipt binds its digest.
Capacity scheduling order is frozen before outcomes and does not wait for,
inspect, or bind confirmatory result bytes. Primary/clean confirmatory receipts
belong only to the PAPER-C independent-reproduction chain; presenting either
receipt to a capacity runner is rejected as an undeclared dependency.

`capacity-production-phases.yaml` is a preregistered wrapper over this same
runner, not a second execution path. Its closed ordered phase IDs and mappings
are:

```text
scaling-fit:
  mode=scaling, cli_role=fit, manifest_role=a100_fit
scaling-holdout:
  mode=scaling, cli_role=holdout, manifest_role=a100_holdout
coverage-cardinality-fit:
  mode=coverage, arm=cardinality, cli_role=fit
coverage-fixed-bits-fit:
  mode=coverage, arm=fixed-bits, cli_role=fit
coverage-cardinality-holdout:
  mode=coverage, arm=cardinality, cli_role=holdout
coverage-fixed-bits-holdout:
  mode=coverage, arm=fixed-bits, cli_role=holdout
parametric-information-all:
  mode=parametric-information, cli_role=all
```

Each row freezes its manifest, cohort index, track/aggregate budget, canonical
ledger, output/staging path, receipt, validation path, selection/config
digests, snapshot, required predecessor phase, `schedule_digest`, canonical
ordered child-key vector, node/rank/wave/thermal/cache assignments, and
per-block balance tolerances. The same schedule object is bound by G2-CAP,
the capacity G4 snapshot, every rank shard manifest, and the final receipt;
workers must launch owned children in the canonical subsequence induced by
that global order. `stc runtime
run-capacity-phase --phase-plan ... --phase-id ...` resolves the row and calls
`run-matrix`; it cannot override a mapped field. Scaling and coverage holdout
rows additionally require the exact provisional and one-time unlock artifacts.
Tests mutate every field, reorder phases, mix an arm/role, omit a predecessor,
alias an output, pass a confirmatory receipt, or attempt a holdout before
unlock; all fail before worker launch.

The optional `--launcher torchrun` path reuses the accelerator package's tested
container launcher. The cohort index expands logical rows to unique physical
child keys; `sha256(child_key) mod world_size` assigns ownership only and may
not define or change launch order. A rank executes its owned keys in their
frozen global-order subsequence and writes only
`workers/rank-{global_rank:04d}/children/{child_key}/attempt-{attempt_id}/`,
and each child manifest is atomically renamed last. Global rank zero waits at a
barrier, verifies the complete expected child/reference incidence map and all
digests, writes the merged result index, then atomically renames the top-level
manifest last. `--resume verify-complete` reuses only checksum-valid children
bound to the same snapshot and inputs, retains incomplete attempts for audit,
and starts a new attempt for every missing or invalid child. A present
top-level manifest makes a non-`--verify-only` invocation fail rather than
overwrite a completed run.

`stc runtime run-matrix --verify-only --receipt <path>` emits a typed receipt
only after the top-level bundle passes. The receipt binds the source tree
digest, snapshot, manifest/index/budget, all rank and child manifests, logical
reference map, environment, and measurement-status digests.
`stc runtime import-matrix --receipt <path> --source <dir> --output <dir>`
revalidates those bytes, copies into a temporary sibling of the canonical
output (using a verified reflink only when available), rejects symlinks and
external references, and atomically renames the canonical top-level manifest
last. It refuses an existing output and never invokes a child workload.

```bash
uv run pytest \
  tests/runtime/test_manifest_runner.py \
  tests/runtime/test_distributed_manifest_runner.py \
  tests/runtime/test_capacity_phase_plan.py \
  tests/runtime/test_matrix_import.py -q
uv run stc runtime run-matrix \
  --manifest tests/runtime/fixtures/manifest-smoke.jsonl \
  --mode fixture \
  --output /tmp/stc-manifest-runner
```

Expected: all fixture cells have lifecycle charges and valid manifests; a
method returning an uncharged or cutoff-mismatched artifact fails closed.
`tests/runtime/test_distributed_manifest_runner.py` exercises an eight-rank
fake launcher and rejects overlapping shard ownership, a missing/extra child,
logical-only placeholders, duplicate physical execution, cross-rank writes,
snapshot/preregistration drift, unsafe resume, and a top-level manifest written
before every child manifest verifies. Capacity-only scheduler fixtures reject
monotone load or \(M\) order, a substrate isolated in one wave,
method×rank confounding, thermal/cache imbalance beyond tolerance, and
order-seed/digest drift. Permuting input file rows while preserving the
canonical schedule is invariant, whereas using the ownership hash as launch
order fails. Scaling tests also reject a missing or
drifted program envelope/reference profile, any planned or actual committed
RRE above \(Q_{\rm capacity,max}\), or any native-resource/carbon overrun.
Import tests interrupt the copy, mutate one source byte, inject a symlink, and
precreate the destination; no case may publish a canonical manifest or execute
a child. Clean-rerun tests reject a missing/stale/replayed rerun request or one
that does not bind the imported primary receipt/tree. CLI tests own every
distributed, receipt, rerun-request, and import option used by Task 8.
Capacity tests prove the inverse dependency boundary: supplying any
confirmatory primary/clean receipt or validation report to scaling, coverage,
or information prevents worker launch. Mutating a phase predecessor,
arm/role, manifest, output, receipt, or order byte stales the runtime contract.

- [ ] **Step 4: Freeze the runtime contract and register manifests**

Register only the small manifests with the artifact DAG. Large generated
payloads remain outside Git.

```bash
uv run stc runtime freeze-contract \
  --index-refresh manifests/systems/index-refresh-G2.json \
  --capacity-phase-plan configs/systems/capacity-production-phases.yaml \
  --output manifests/systems/runtime-G2.json
uv run stc dag import-bundles \
  --input /tmp/stc-lifecycle-smoke \
  --manifest-dir manifests/systems/runs
uv run stc dag check
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: add matched lifecycle integration protocol"
```

---

### Task 7: Build the A100 measurement harness

**Files:**
- Create: `src/stc_research/accelerator/__init__.py`
- Create: `src/stc_research/accelerator/schemas.py`
- Create: `src/stc_research/accelerator/probe.py`
- Create: `src/stc_research/accelerator/counters.py`
- Create: `src/stc_research/accelerator/energy.py`
- Create: `src/stc_research/accelerator/kernels.py`
- Create: `src/stc_research/accelerator/runner.py`
- Create: `src/stc_research/accelerator/validate.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `src/stc_research/cli.py`
- Modify: `tests/test_cli.py`
- Create: `configs/accelerator/a100-smoke.yaml`
- Create: `configs/accelerator/a100-confirmatory.yaml`
- Create: `configs/accelerator/counter-profile.yaml`
- Create: `schemas/accelerator/counter-profile-v1.schema.json`
- Create: `schemas/accelerator/a100-runbook-smoke-v1.schema.json`
- Create: `containers/a100/Dockerfile`
- Create: `containers/a100/toolchain.lock.json`
- Create: `tests/accelerator/test_probe.py`
- Create: `tests/accelerator/test_counters.py`
- Create: `tests/accelerator/test_energy.py`
- Create: `tests/accelerator/test_runner.py`
- Create: `tests/accelerator/test_distributed_bundle.py`
- Create: `tests/accelerator/test_bundle_validation.py`
- Create: `tests/accelerator/test_dag_import.py`
- Create: `tests/accelerator/test_runbook_smoke.py`
- Create: `tests/accelerator/fixtures/runbook-smoke.md`

**Interfaces:**
- Measures: synthetic KV read, fast-state read/modify/write, delayed-write
  cadence, adapter load/apply/update, checkpoint, storage/network transfer, and
  wake/sleep contention.
- Produces: structurally valid `SYNTHETIC` bundles or target-A100 bundles whose
`measurement_status` is fail-closed as `PARTIAL|ADMISSIBLE`.
- Owns the closed stage names
  `single-gpu-semantic-deletion-rollback-gate`,
  `single-gpu-resource-path-smoke`, `single-node-contention-gate`,
  `nccl-ib-nfs-gate`, and `confirmatory`, plus
  `accelerator runbook-smoke --runbook PATH --config PATH --backend fake
  --output PATH`. The latter executes all gate command contracts through the
  fake backend and emits a small `SYNTHETIC` command-surface receipt; it is
  never target-hardware evidence.

- [ ] **Step 1: Write fake-backend and admissibility tests**

Reject `measurement_status=ADMISSIBLE` if any of these are missing:

```text
two host identities and exactly eight unique A100 80 GB GPU UUIDs
driver/CUDA/container digest
power cap and clock policy
energy counter source and sampling interval
30-second warm-up
120-second measurement window
pre/post idle measurement
resolved configuration and code digest
raw metric checksums
one checksum-valid raw shard and manifest-last completion record for every global rank
one checksum-valid host-counter shard and manifest-last completion record per node
CUPTI HBM read/write counter availability and calibration
host uncore-IMC DRAM read/write counter availability and calibration on both nodes
```

A fake backend must validate structurally but carry
`measurement_status: SYNTHETIC` and `evidence_class: SYNTHETIC`; it cannot
support an absolute hardware claim.

`a100-confirmatory.yaml` also freezes repetitions, minimum successful sample
count for p99, paired-window/block-bootstrap uncertainty, CUDA-event timing plus
host synchronization, kernel numerical oracle/readback, anti-elision checksum,
logical tensor-shape-derived byte count, NVML/DCGM sampling resolution and
counter-wrap/reset handling, the counter-profile digest, counter calibration
tolerances, and maximum permitted power/clock/temperature drift. Any drift,
sample-count, shard, or counter-calibration violation demotes the bundle to
`PARTIAL`.

Runner/parser tests require the stage-specific world sizes and checks:

```text
single-gpu-semantic-deletion-rollback-gate  ranks=1
single-gpu-resource-path-smoke              ranks=1
single-node-contention-gate                 ranks=4 on one node
nccl-ib-nfs-gate                            ranks=8 across two nodes
confirmatory                                ranks=8 across two nodes
```

The first stage must report identity semantics, deletion non-resurrection, and
authorized rollback. The second must exercise KV read, fast-state RMW, adapter
load/apply/update, checkpoint, and storage/network transfer. The third must
report concurrent wake/sleep interference and deadline accounting. An unknown
stage, wrong world size, omitted stage check, or attempt to launch
`confirmatory` without checksum-valid PASS manifests for all four prior gates
fails before worker launch.

`accelerator run` owns a repeatable `--require-gate PATH` option. Each worker
rehashes the referenced manifest-last bundle before container launch and binds
the ordered prerequisite digests into its own run manifest. A missing,
non-PASS, wrong-stage, stale, or duplicate prerequisite aborts every rank
before measurement. `accelerator validate` owns repeatable
`--require-check CODE`; it fails unless the stage manifest contains each exact
PASS check code.

For target-A100 `PARTIAL|ADMISSIBLE` bundles, per-metric provenance is exact:

```text
logical_hbm_read_bytes/logical_hbm_write_bytes  DERIVED_FROM_SHAPE
hbm_read_bytes/hbm_write_bytes                  DIRECT_COUNTER
host_dram_read_bytes/host_dram_write_bytes      DIRECT_COUNTER
gpu_board_j                                     DIRECT_COUNTER
host_j                                          DIRECT_COUNTER | UNAVAILABLE
storage_network_modeled_j                       MODELED
```

For a `SYNTHETIC` fake bundle, logical byte fields may remain
`DERIVED_FROM_SHAPE`, modeled storage/network fields remain `MODELED`, and
every physical HBM, host-DRAM, GPU-energy, or host-energy value is `null` with
`provenance=UNAVAILABLE` and `reason=SYNTHETIC_BACKEND`. A fake value may
never carry `DIRECT_COUNTER`, even when it exists only to exercise schema
shape.

GPU HBM uses the CUDA-toolkit-matched CUPTI Profiling API counters
`dram__bytes_read.sum` and `dram__bytes_write.sum`; on A100 those counters map
to device HBM and never to host DRAM. Host DRAM v1 uses Linux
`perf_event_open` over discovered `uncore_imc_*` `cas_count_read` and
`cas_count_write` PMU events. The checked-in `counter-profile.yaml`
preregisters the accepted semantic event names, required permissions,
bytes-per-CAS derivation rule, scaling/reset policy, calibration fixture
digest, and tolerances. Each node probe records its actual kernel/CPU,
`/sys/bus/event_source/devices/uncore_imc_*` instances, event/format strings,
resolved `perf_event_attr` type/config values, cpumasks, bytes-per-CAS,
time-enabled/time-running scaling, CUPTI/CUDA library hashes, permissions,
calibration result, and container digest. The cluster probe and run manifest
bind both resolved probe digests and the counter-profile digest. If either node
lacks those PMU events or permission, host DRAM is `null` with a reason and the
G8 target-measurement requirement remains blocked; logical bytes are never
substituted.

- [ ] **Step 2: Implement probe reconciliation, counters, runner, and validator**

Define `[project.optional-dependencies].accelerator` with exact direct
requirements `torch==2.13.0` and `nvidia-ml-py==13.610.43`, then update and
commit `uv.lock`. `containers/a100/toolchain.lock.json` additionally binds the
CUDA-toolkit-matched CUPTI/NCCL shared libraries and the resolved accelerator
wheel hashes; the runner rejects a wheel/library mismatch. The accelerator
tests assert those exact package versions, and `uv sync --frozen --extra
accelerator` must leave the lockfile unchanged.

Capture latency p50/p95/p99, achieved bandwidth, effective batch, logical and
direct-counter bytes, FLOPs, GPU seconds, host time, GPU-board joules, host
joules where directly measured, and modeled storage/network joules in separate
fields.

`stc accelerator reconcile-probes` accepts exactly two node probe records,
verifies matching code/container/config and complementary node ranks, selects
only mutually present NFS/NIC/HCA/counter capabilities, and writes both a typed
cluster probe and a shell-quoted environment file. Tests reject a single probe,
duplicate rank/GPU UUID, mismatched image or checkout, non-shared filesystem,
unreachable master route, unsafe shell bytes, unsupported PMU, and socket-only
NCCL fallback.

Each kernel performs a seeded numerical correctness/readback check before
timing. Measurement uses synchronized CUDA events for device work and monotonic
host clocks for end-to-end latency, at least three paired idle/work windows,
and stores raw NVML/DCGM samples so energy integration can be independently
recomputed. Each global rank writes only
`raw-metrics/rank-{global_rank:04d}.jsonl`, fsyncs it, and atomically renames
its own `rank-manifests/rank-{global_rank:04d}.json.tmp` completion record
last. Each node-local rank zero similarly writes and fsyncs its own host-counter
shard before atomically publishing its node host manifest. After a distributed
barrier, global rank zero verifies all eight rank manifests/shard checksums,
both host manifests/shards, both probe records, and all counter calibrations,
then writes the summary and atomically renames `manifest.json.tmp` to
`manifest.json` last. No rank may write another rank's files; no nonzero global
rank may write the summary or global manifest.
Rank/host completion records bind run and measurement-window IDs, global/local
rank or node rank, host/GPU UUIDs, shard digest and row count, resolved
config/code/container/probe/counter-profile digests, and terminal success.
They cannot assign the bundle's measurement status or evidence class.

- [ ] **Step 3: Run the hardware-free test path**

```bash
uv lock
uv sync --frozen --extra accelerator
uv run pytest \
  tests/accelerator/test_probe.py \
  tests/accelerator/test_counters.py \
  tests/accelerator/test_energy.py \
  tests/accelerator/test_runner.py \
  tests/accelerator/test_distributed_bundle.py \
  tests/accelerator/test_bundle_validation.py \
  tests/accelerator/test_dag_import.py \
  tests/accelerator/test_runbook_smoke.py -q
uv run stc accelerator run \
  --config configs/accelerator/a100-smoke.yaml \
  --backend fake \
  --output /tmp/stc-a100-fake
uv run stc accelerator validate /tmp/stc-a100-fake \
  --require-measurement-status SYNTHETIC
uv run stc dag import-bundles \
  --input /tmp/stc-a100-fake \
  --manifest-dir manifests/accelerator/runs
uv run stc dag check
```

Expected: tests pass; validator reports `PASS measurement_status=SYNTHETIC`;
the small fake manifest is imported with its class intact; claim export rejects
`ORIGINAL-MEASUREMENT`. Tests also reject a missing rank, two ranks writing the
same shard, a rank or host manifest written before its shard is durable, a
global manifest written before the barrier, missing HBM/host-DRAM counters, an
uncalibrated counter, and a fake bundle relabeled as measured. Missing direct
counters fail a target-A100 `ADMISSIBLE` bundle; they do not prevent an
honestly labeled `SYNTHETIC` structural bundle from validating. Tests reject
any synthetic physical metric labeled `DIRECT_COUNTER`.
The DAG-import test covers both `SYNTHETIC` and `PARTIAL` fixture manifests,
recomputes payload digests, and rejects the wrong manifest namespace, an
overwrite, a duplicate run ID with different bytes, an unregistered payload,
and any evidence-class or measurement-status upgrade.

- [ ] **Step 4: Commit**

```bash
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: add accelerator lifecycle measurement harness"
```

---

### Task 8: Create and exercise the new A100 runbook

**Files:**
- Create: `runbook/sleep-time-compute-a100.md`
- Create: `research/sleep-time-compute/manifests/a100-runbook-smoke.json`
- Create when target hardware succeeds:
  `research/sleep-time-compute/manifests/accelerator/runs/{a100-single-gpu-semantic,a100-single-gpu-resource-smoke,a100-single-node-contention,a100-two-node-gate,a100-confirmatory}.json`
- Modify when target hardware succeeds:
  `research/sleep-time-compute/registry/artifacts.jsonl`
- Create after post-G4 execution:
  `research/sleep-time-compute/manifests/systems/distributed-confirmatory-receipt.json`
- Create after post-G4 execution:
  `research/sleep-time-compute/manifests/systems/distributed-confirmatory-clean-rerun-receipt.json`
- Create after post-G4 execution:
  `research/sleep-time-compute/manifests/systems/distributed-scaling-fit-receipt.json`
- Create after post-G4 execution:
  `research/sleep-time-compute/manifests/systems/distributed-scaling-holdout-receipt.json`
- Create after post-G4 execution:
  `research/sleep-time-compute/manifests/systems/distributed-coverage-{cardinality,fixed-bits}-{fit,holdout}-receipt.json`
- Create after post-G4 execution:
  `research/sleep-time-compute/manifests/systems/distributed-parametric-information-receipt.json`

**Interfaces:**
- Target: two manually managed nodes, each with four A100 80 GB GPUs, NFS, and
  InfiniBand.
- Produces: environment probe, single-GPU gate, two-node gate, confirmatory
  microbenchmark bundle, and artifact-DAG handoff before G4.
- Documents but does not pre-authorize: the post-G4 distributed
  `runtime run-matrix` path for every unique physical child behind the 297
  confirmatory rows and the capacity program's 120 scaling, 160 coverage, and
  48 parametric-information logical rows, with their independently powered
  child counts.
- `manifests/a100-runbook-smoke.json` is a schema-validated,
  hardware-free command-surface receipt generated by Task 7's fake backend
  from the final runbook. It records the runbook/config/toolchain digests,
  exact ordered gate names/world sizes/prerequisites, parsed command argv, fake
  bundle digests, `evidence_class=SYNTHETIC`, and
  `measurement_status=SYNTHETIC`. It contains no physical metric and cannot
  satisfy the target-A100 branch of G4 or any hardware claim.

- [ ] **Step 1: Document discovery before execution**

The runbook obtains hosts, mount points, device identities, power caps, clock
policy, CUDA/NCCL/driver versions, and network interfaces from
`stc accelerator probe`. It must not invent IP addresses or cluster policy.
Both nodes start from the same clean checkout and the same image ID from
`containers/a100/Dockerfile`; the image ID, exported image checksum, checkout
SHA/tree, `uv.lock`, and `toolchain.lock.json` digests are checked before any
measurement. `stc accelerator run --launcher torchrun --container-ref ...` is
the public launcher: it constructs the container invocation from validated
probe records and then launches the internal distributed worker. The runbook
never asks an operator to call an internal Python module directly. The launcher
passes only the reconciled NCCL variables, validated GPU/InfiniBand devices,
read-only checkout, shared result mount, and `CAP_PERFMON` needed for the frozen
counter profile; it records the exact container argv/environment and rejects a
broader privilege request.

On node 0, after the operator sets the reviewed shared NFS bootstrap path:

```bash
: "${STC_BOOTSTRAP_ROOT:?set this to the reviewed shared NFS directory}"
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$STC_REPO_ROOT"
docker build \
  --iidfile "$STC_BOOTSTRAP_ROOT/stc-a100.iid" \
  --tag stc-a100:stc-v1 \
  --file research/sleep-time-compute/containers/a100/Dockerfile \
  .
docker save \
  --output "$STC_BOOTSTRAP_ROOT/stc-a100.tar" \
  stc-a100:stc-v1
sha256sum "$STC_BOOTSTRAP_ROOT/stc-a100.tar" \
  > "$STC_BOOTSTRAP_ROOT/stc-a100.tar.sha256"
```

On node 1:

```bash
: "${STC_BOOTSTRAP_ROOT:?set this to the reviewed shared NFS directory}"
sha256sum -c "$STC_BOOTSTRAP_ROOT/stc-a100.tar.sha256"
docker load --input "$STC_BOOTSTRAP_ROOT/stc-a100.tar"
docker image inspect stc-a100:stc-v1 \
  --format '{{.Id}}' \
  > "$STC_BOOTSTRAP_ROOT/stc-a100-node1.iid"
cmp "$STC_BOOTSTRAP_ROOT/stc-a100.iid" \
  "$STC_BOOTSTRAP_ROOT/stc-a100-node1.iid"
```

- [ ] **Step 2: Document the staged gates**

Use this order:

1. build container and record resolved digest;
2. probe both nodes;
3. single-GPU semantic, deletion, and rollback gate;
4. KV/RMW/adapter/checkpoint/transfer smoke;
5. four-GPU single-node contention gate;
6. two-node NCCL/InfiniBand and NFS gate;
7. full confirmatory microbenchmark measurement;
8. immutable validation and artifact-DAG import;
9. after typed G4 and its execution snapshot only, distributed confirmatory
   and scaling production matrices.

Record exact failure recovery and resume commands. Never silently reuse a
partial measurement window.

- [ ] **Step 3: Run when the target hardware is available**

An operator first sets `STC_BOOTSTRAP_ROOT` on both nodes to the same reviewed
NFS directory. That value is the only pre-existing cluster path. On node 0:

```bash
: "${STC_BOOTSTRAP_ROOT:?set this to the reviewed shared NFS directory}"
export STC_CONTAINER_REF=stc-a100:stc-v1
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$STC_REPO_ROOT/research/sleep-time-compute"
uv sync --frozen --extra accelerator
mkdir -p "$STC_BOOTSTRAP_ROOT/probes"
uv run stc accelerator probe \
  --node-rank 0 \
  --container-ref "$STC_CONTAINER_REF" \
  --output "$STC_BOOTSTRAP_ROOT/probes/node-0.json"
```

On node 1:

```bash
: "${STC_BOOTSTRAP_ROOT:?set this to the reviewed shared NFS directory}"
export STC_CONTAINER_REF=stc-a100:stc-v1
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$STC_REPO_ROOT/research/sleep-time-compute"
uv sync --frozen --extra accelerator
mkdir -p "$STC_BOOTSTRAP_ROOT/probes"
uv run stc accelerator probe \
  --node-rank 1 \
  --container-ref "$STC_CONTAINER_REF" \
  --output "$STC_BOOTSTRAP_ROOT/probes/node-1.json"
```

After both probe commands finish, node 0 reconciles them:

```bash
: "${STC_BOOTSTRAP_ROOT:?set this to the reviewed shared NFS directory}"
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$STC_REPO_ROOT/research/sleep-time-compute"
uv run stc accelerator reconcile-probes \
  --probe "$STC_BOOTSTRAP_ROOT/probes/node-0.json" \
  --probe "$STC_BOOTSTRAP_ROOT/probes/node-1.json" \
  --nfs-root "$STC_BOOTSTRAP_ROOT" \
  --counter-profile configs/accelerator/counter-profile.yaml \
  --output-probe "$STC_BOOTSTRAP_ROOT/probes/cluster.json" \
  --output-env "$STC_BOOTSTRAP_ROOT/cluster.env"
sha256sum "$STC_BOOTSTRAP_ROOT/cluster.env" \
  > "$STC_BOOTSTRAP_ROOT/cluster.env.sha256"
```

Reconciliation fails unless host routes are mutually reachable, the NFS mount
has the same filesystem identity on both nodes, the container/checkouts match,
there are exactly eight unique A100 80 GB UUIDs, and the reviewed NIC/HCA and
counter backends exist on both nodes. It emits shell-quoted `export` statements
for `STC_MASTER_ADDR`, `STC_MASTER_PORT`, `STC_NFS_ROOT`,
`STC_CONTAINER_REF`, `STC_COUNTER_PROFILE`,
`NCCL_SOCKET_IFNAME`, and `NCCL_IB_HCA`; the last two are the actual NCCL
variables, not presentation-only aliases. Both operators inspect the file and
run `sha256sum -c "$STC_BOOTSTRAP_ROOT/cluster.env.sha256"` before sourcing the
same bytes.

Node 0 now runs and validates the three mandatory local gates in order. First,
the one-rank semantic/deletion/rollback gate:

```bash
: "${STC_BOOTSTRAP_ROOT:?set this to the reviewed shared NFS directory}"
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$STC_REPO_ROOT/research/sleep-time-compute"
sha256sum -c "$STC_BOOTSTRAP_ROOT/cluster.env.sha256"
. "$STC_BOOTSTRAP_ROOT/cluster.env"
uv run stc accelerator run \
  --backend a100 \
  --launcher torchrun \
  --nnodes 1 --nproc-per-node 1 --node-rank 0 \
  --master-addr "$STC_MASTER_ADDR" --master-port "$STC_MASTER_PORT" \
  --container-ref "$STC_CONTAINER_REF" \
  --probe "$STC_BOOTSTRAP_ROOT/probes/cluster.json" \
  --counter-profile "$STC_COUNTER_PROFILE" \
  --stage single-gpu-semantic-deletion-rollback-gate \
  --config configs/accelerator/a100-smoke.yaml \
  --output "$STC_NFS_ROOT/results/a100-runs/a100-single-gpu-semantic"
uv run stc accelerator validate \
  "$STC_NFS_ROOT/results/a100-runs/a100-single-gpu-semantic" \
  --require-stage single-gpu-semantic-deletion-rollback-gate \
  --require-ranks 1 \
  --require-check SEMANTIC_IDENTITY \
  --require-check DELETION_NON_RESURRECTION \
  --require-check AUTHORIZED_ROLLBACK
```

Second, the one-rank resource-path smoke, which cannot launch unless the first
gate manifest remains checksum-valid:

```bash
: "${STC_BOOTSTRAP_ROOT:?set this to the reviewed shared NFS directory}"
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$STC_REPO_ROOT/research/sleep-time-compute"
sha256sum -c "$STC_BOOTSTRAP_ROOT/cluster.env.sha256"
. "$STC_BOOTSTRAP_ROOT/cluster.env"
uv run stc accelerator run \
  --backend a100 \
  --launcher torchrun \
  --nnodes 1 --nproc-per-node 1 --node-rank 0 \
  --master-addr "$STC_MASTER_ADDR" --master-port "$STC_MASTER_PORT" \
  --container-ref "$STC_CONTAINER_REF" \
  --probe "$STC_BOOTSTRAP_ROOT/probes/cluster.json" \
  --counter-profile "$STC_COUNTER_PROFILE" \
  --stage single-gpu-resource-path-smoke \
  --config configs/accelerator/a100-smoke.yaml \
  --require-gate \
    "$STC_NFS_ROOT/results/a100-runs/a100-single-gpu-semantic" \
  --output "$STC_NFS_ROOT/results/a100-runs/a100-single-gpu-resource-smoke"
uv run stc accelerator validate \
  "$STC_NFS_ROOT/results/a100-runs/a100-single-gpu-resource-smoke" \
  --require-stage single-gpu-resource-path-smoke \
  --require-ranks 1 \
  --require-check KV_READ \
  --require-check FAST_STATE_RMW \
  --require-check ADAPTER_LOAD_APPLY_UPDATE \
  --require-check CHECKPOINT \
  --require-check STORAGE_NETWORK_TRANSFER
```

Third, the four-rank single-node wake/sleep contention gate:

```bash
: "${STC_BOOTSTRAP_ROOT:?set this to the reviewed shared NFS directory}"
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$STC_REPO_ROOT/research/sleep-time-compute"
sha256sum -c "$STC_BOOTSTRAP_ROOT/cluster.env.sha256"
. "$STC_BOOTSTRAP_ROOT/cluster.env"
uv run stc accelerator run \
  --backend a100 \
  --launcher torchrun \
  --nnodes 1 --nproc-per-node 4 --node-rank 0 \
  --master-addr "$STC_MASTER_ADDR" --master-port "$STC_MASTER_PORT" \
  --container-ref "$STC_CONTAINER_REF" \
  --probe "$STC_BOOTSTRAP_ROOT/probes/cluster.json" \
  --counter-profile "$STC_COUNTER_PROFILE" \
  --stage single-node-contention-gate \
  --config configs/accelerator/a100-smoke.yaml \
  --require-gate \
    "$STC_NFS_ROOT/results/a100-runs/a100-single-gpu-semantic" \
  --require-gate \
    "$STC_NFS_ROOT/results/a100-runs/a100-single-gpu-resource-smoke" \
  --output "$STC_NFS_ROOT/results/a100-runs/a100-single-node-contention"
uv run stc accelerator validate \
  "$STC_NFS_ROOT/results/a100-runs/a100-single-node-contention" \
  --require-stage single-node-contention-gate \
  --require-ranks 4 \
  --require-check WAKE_SLEEP_INTERFERENCE \
  --require-check DEADLINE_ACCOUNTING
```

Each command publishes its top-level manifest last. A failed validation leaves
the later stage unlaunchable; recovery creates a new measurement-window ID and
reruns the failed gate rather than editing or overwriting its manifest.

For the two-node NCCL/InfiniBand/NFS gate, launch the following concurrently.
On node 0:

```bash
: "${STC_BOOTSTRAP_ROOT:?set this to the reviewed shared NFS directory}"
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$STC_REPO_ROOT/research/sleep-time-compute"
sha256sum -c "$STC_BOOTSTRAP_ROOT/cluster.env.sha256"
. "$STC_BOOTSTRAP_ROOT/cluster.env"
uv run stc accelerator run \
  --backend a100 \
  --launcher torchrun \
  --nnodes 2 --nproc-per-node 4 --node-rank 0 \
  --master-addr "$STC_MASTER_ADDR" --master-port "$STC_MASTER_PORT" \
  --container-ref "$STC_CONTAINER_REF" \
  --probe "$STC_BOOTSTRAP_ROOT/probes/cluster.json" \
  --counter-profile "$STC_COUNTER_PROFILE" \
  --stage nccl-ib-nfs-gate \
  --config configs/accelerator/a100-confirmatory.yaml \
  --require-gate \
    "$STC_NFS_ROOT/results/a100-runs/a100-single-gpu-semantic" \
  --require-gate \
    "$STC_NFS_ROOT/results/a100-runs/a100-single-gpu-resource-smoke" \
  --require-gate \
    "$STC_NFS_ROOT/results/a100-runs/a100-single-node-contention" \
  --output "$STC_NFS_ROOT/results/a100-runs/a100-two-node-gate"
```

On node 1:

```bash
: "${STC_BOOTSTRAP_ROOT:?set this to the reviewed shared NFS directory}"
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$STC_REPO_ROOT/research/sleep-time-compute"
sha256sum -c "$STC_BOOTSTRAP_ROOT/cluster.env.sha256"
. "$STC_BOOTSTRAP_ROOT/cluster.env"
uv run stc accelerator run \
  --backend a100 \
  --launcher torchrun \
  --nnodes 2 --nproc-per-node 4 --node-rank 1 \
  --master-addr "$STC_MASTER_ADDR" --master-port "$STC_MASTER_PORT" \
  --container-ref "$STC_CONTAINER_REF" \
  --probe "$STC_BOOTSTRAP_ROOT/probes/cluster.json" \
  --counter-profile "$STC_COUNTER_PROFILE" \
  --stage nccl-ib-nfs-gate \
  --config configs/accelerator/a100-confirmatory.yaml \
  --require-gate \
    "$STC_NFS_ROOT/results/a100-runs/a100-single-gpu-semantic" \
  --require-gate \
    "$STC_NFS_ROOT/results/a100-runs/a100-single-gpu-resource-smoke" \
  --require-gate \
    "$STC_NFS_ROOT/results/a100-runs/a100-single-node-contention" \
  --output "$STC_NFS_ROOT/results/a100-runs/a100-two-node-gate"
```

Node 0 validates the completed gate before either node starts confirmatory
measurement:

```bash
: "${STC_BOOTSTRAP_ROOT:?set this to the reviewed shared NFS directory}"
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$STC_REPO_ROOT/research/sleep-time-compute"
sha256sum -c "$STC_BOOTSTRAP_ROOT/cluster.env.sha256"
. "$STC_BOOTSTRAP_ROOT/cluster.env"
uv run stc accelerator validate \
  "$STC_NFS_ROOT/results/a100-runs/a100-two-node-gate" \
  --require-stage nccl-ib-nfs-gate \
  --require-ranks 8 \
  --require-check NCCL_NET_IB \
  --require-check NFS_ATOMIC_RENAME \
  --require-check NFS_SUSTAINED_THROUGHPUT \
  --counter-profile "$STC_COUNTER_PROFILE"
```

After that validation passes, launch confirmatory measurement concurrently. On
node 0:

```bash
: "${STC_BOOTSTRAP_ROOT:?set this to the reviewed shared NFS directory}"
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$STC_REPO_ROOT/research/sleep-time-compute"
sha256sum -c "$STC_BOOTSTRAP_ROOT/cluster.env.sha256"
. "$STC_BOOTSTRAP_ROOT/cluster.env"
uv run stc accelerator run \
  --backend a100 \
  --launcher torchrun \
  --nnodes 2 --nproc-per-node 4 --node-rank 0 \
  --master-addr "$STC_MASTER_ADDR" --master-port "$STC_MASTER_PORT" \
  --container-ref "$STC_CONTAINER_REF" \
  --probe "$STC_BOOTSTRAP_ROOT/probes/cluster.json" \
  --counter-profile "$STC_COUNTER_PROFILE" \
  --stage confirmatory \
  --config configs/accelerator/a100-confirmatory.yaml \
  --require-gate \
    "$STC_NFS_ROOT/results/a100-runs/a100-single-gpu-semantic" \
  --require-gate \
    "$STC_NFS_ROOT/results/a100-runs/a100-single-gpu-resource-smoke" \
  --require-gate \
    "$STC_NFS_ROOT/results/a100-runs/a100-single-node-contention" \
  --require-gate \
    "$STC_NFS_ROOT/results/a100-runs/a100-two-node-gate" \
  --output "$STC_NFS_ROOT/results/a100-runs/a100-confirmatory"
```

On node 1:

```bash
: "${STC_BOOTSTRAP_ROOT:?set this to the reviewed shared NFS directory}"
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$STC_REPO_ROOT/research/sleep-time-compute"
sha256sum -c "$STC_BOOTSTRAP_ROOT/cluster.env.sha256"
. "$STC_BOOTSTRAP_ROOT/cluster.env"
uv run stc accelerator run \
  --backend a100 \
  --launcher torchrun \
  --nnodes 2 --nproc-per-node 4 --node-rank 1 \
  --master-addr "$STC_MASTER_ADDR" --master-port "$STC_MASTER_PORT" \
  --container-ref "$STC_CONTAINER_REF" \
  --probe "$STC_BOOTSTRAP_ROOT/probes/cluster.json" \
  --counter-profile "$STC_COUNTER_PROFILE" \
  --stage confirmatory \
  --config configs/accelerator/a100-confirmatory.yaml \
  --require-gate \
    "$STC_NFS_ROOT/results/a100-runs/a100-single-gpu-semantic" \
  --require-gate \
    "$STC_NFS_ROOT/results/a100-runs/a100-single-gpu-resource-smoke" \
  --require-gate \
    "$STC_NFS_ROOT/results/a100-runs/a100-single-node-contention" \
  --require-gate \
    "$STC_NFS_ROOT/results/a100-runs/a100-two-node-gate" \
  --output "$STC_NFS_ROOT/results/a100-runs/a100-confirmatory"
```

Node 0 seals and imports only after both launchers exit:

```bash
: "${STC_BOOTSTRAP_ROOT:?set this to the reviewed shared NFS directory}"
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$STC_REPO_ROOT/research/sleep-time-compute"
sha256sum -c "$STC_BOOTSTRAP_ROOT/cluster.env.sha256"
. "$STC_BOOTSTRAP_ROOT/cluster.env"
uv run stc accelerator validate \
  "$STC_NFS_ROOT/results/a100-runs/a100-confirmatory" \
  --require-target a100-80gb \
  --require-ranks 8 \
  --counter-profile "$STC_COUNTER_PROFILE" \
  --require-measurement-status ADMISSIBLE
for STC_A100_RUN_ID in \
  a100-single-gpu-semantic \
  a100-single-gpu-resource-smoke \
  a100-single-node-contention \
  a100-two-node-gate \
  a100-confirmatory
do
  uv run stc dag import-bundles \
    --input "$STC_NFS_ROOT/results/a100-runs/$STC_A100_RUN_ID" \
    --manifest-dir manifests/accelerator/runs
done
uv run stc dag check
```

The runbook verifies collective numerical correctness, confirms NCCL `NET/IB`
transport rather than socket fallback, tests NFS atomic rename and sustained
throughput, verifies the eight rank shard/manifest pairs and two host-counter
shard/manifest pairs, and provides exact node-specific resume commands keyed by
validated stage manifests. Resume creates a new measurement-window ID; it
never appends to or silently reuses a partial window.

Expected: each node reports four A100 80 GB devices; validation reports zero
missing rank/host shards, digest errors, short measurement windows, uncalibrated
HBM/host-DRAM counters, or mixed energy boundaries; and all four prerequisite
gate manifests plus `a100-confirmatory` exist under
`manifests/accelerator/runs/`. Each imported confirmatory manifest retains the
ordered prerequisite run IDs and digests. If hardware or either
required counter backend is unavailable, keep this gate `BLOCKED-EXTERNAL`; do
not replace it with synthetic or shape-derived evidence.

- [ ] **Step 4: Document the post-G4 production matrix path**

This step is part of the runbook but must not execute until Task 9 records PASS
G4. The evidence controller emits two non-substitutable typed snapshots.
`manifests/stc/g4-confirmatory-execution.json` binds only the core
preregistration/G2, confirmatory manifest/run index, analysis plan, G4
decision/input index, checkout, and target environment.
`manifests/stc/g4-capacity-execution.json` binds PASS core G4/G2 plus the
capacity preregistration/G2-CAP, pre-G2-CAP configs/stakes, disjoint capacity
CAL receipts/validations/selections, all capacity powers/manifests/cohort
indexes/budgets, canonical aggregate-ledger contract, checkout, and exact
target-A100 CAL→TEST transport predicate. It binds no confirmatory outcomes or
receipts.

On node 0, freeze the exact control bytes on NFS:

```bash
: "${STC_BOOTSTRAP_ROOT:?set this to the reviewed shared NFS directory}"
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$STC_REPO_ROOT/research/sleep-time-compute"
sha256sum -c "$STC_BOOTSTRAP_ROOT/cluster.env.sha256"
. "$STC_BOOTSTRAP_ROOT/cluster.env"
uv run stc gate verify-chain --through G4 --root manifests/gates
uv run stc prereg validate \
  --manifest manifests/stc/preregistration.json \
  --fail-on-upstream-digest-change
uv run stc prereg validate \
  --manifest manifests/stc/capacity-preregistration.json \
  --fail-on-upstream-digest-change
mkdir -p "$STC_NFS_ROOT/control"
sha256sum \
  manifests/stc/g4-confirmatory-execution.json \
  manifests/stc/g4-capacity-execution.json \
  manifests/stc/confirmatory.jsonl \
  manifests/stc/confirmatory-run-index.json \
  manifests/stc/capacity-preregistration.json \
  manifests/capacity-gates/G2-CAP.json \
  manifests/stc/capacity-aggregate-budget.json \
  configs/systems/capacity-production-phases.yaml \
  manifests/stc/scaling-cells.jsonl \
  manifests/stc/scaling-cohort-index.json \
  manifests/stc/scaling-budget.json \
  manifests/stc/coverage-cardinality-cells.jsonl \
  manifests/stc/coverage-fixed-bits-cells.jsonl \
  manifests/stc/coverage-scaling-cohort-index.json \
  manifests/stc/coverage-scaling-budget.json \
  manifests/stc/parametric-information-cells.jsonl \
  manifests/stc/parametric-information-cohort-index.json \
  manifests/stc/parametric-information-budget.json \
  configs/stc/design/scaling.yaml \
  configs/stc/design/scaling-hardware.yaml \
  manifests/stc/preregistration.json \
  > "$STC_NFS_ROOT/control/post-g4-inputs.sha256"
```

Both nodes must have the same clean checkout and must pass this exact block
before either launches a production child. Export `STC_NODE_RANK=0` on node 0
and `STC_NODE_RANK=1` on node 1:

```bash
: "${STC_BOOTSTRAP_ROOT:?set this to the reviewed shared NFS directory}"
: "${STC_NODE_RANK:?set to 0 on node 0 and 1 on node 1}"
case "$STC_NODE_RANK" in 0|1) ;; *) exit 2 ;; esac
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$STC_REPO_ROOT/research/sleep-time-compute"
sha256sum -c "$STC_BOOTSTRAP_ROOT/cluster.env.sha256"
. "$STC_BOOTSTRAP_ROOT/cluster.env"
sha256sum -c "$STC_NFS_ROOT/control/post-g4-inputs.sha256"
uv run stc gate verify-chain --through G4 --root manifests/gates
uv run stc prereg validate \
  --manifest manifests/stc/preregistration.json \
  --fail-on-upstream-digest-change
uv run stc prereg validate \
  --manifest manifests/stc/capacity-preregistration.json \
  --fail-on-upstream-digest-change
```

Launch the following confirmatory command concurrently on both nodes:

```bash
: "${STC_BOOTSTRAP_ROOT:?set this to the reviewed shared NFS directory}"
: "${STC_NODE_RANK:?set to 0 on node 0 and 1 on node 1}"
case "$STC_NODE_RANK" in 0|1) ;; *) exit 2 ;; esac
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$STC_REPO_ROOT/research/sleep-time-compute"
sha256sum -c "$STC_BOOTSTRAP_ROOT/cluster.env.sha256"
. "$STC_BOOTSTRAP_ROOT/cluster.env"
sha256sum -c "$STC_NFS_ROOT/control/post-g4-inputs.sha256"
uv run stc runtime run-matrix \
  --manifest manifests/stc/confirmatory.jsonl \
  --cohort-index manifests/stc/confirmatory-run-index.json \
  --preregistration manifests/stc/preregistration.json \
  --execution-snapshot manifests/stc/g4-confirmatory-execution.json \
  --budget manifests/stc/confirmatory-budget.json \
  --budget-role primary \
  --gate-chain manifests/gates \
  --mode confirmatory \
  --run-id confirmatory-primary \
  --backend a100 \
  --launcher torchrun \
  --nnodes 2 --nproc-per-node 4 --node-rank "$STC_NODE_RANK" \
  --master-addr "$STC_MASTER_ADDR" --master-port "$STC_MASTER_PORT" \
  --container-ref "$STC_CONTAINER_REF" \
  --probe "$STC_BOOTSTRAP_ROOT/probes/cluster.json" \
  --counter-profile "$STC_COUNTER_PROFILE" \
  --shard-count 8 \
  --resume verify-complete \
  --output "$STC_NFS_ROOT/staging/confirmatory-primary"
```

Assignment is over unique physical child keys, not the 297 logical parent
rows. The runner must resolve exactly the preregistered \(297n\) logical child
references, execute each deduplicated physical child at most once, and retain
the frozen logical-to-physical incidence/covariance map. After both launchers
exit, node 0 verifies, semantically validates, and imports the manifest-last
bundle:

```bash
: "${STC_BOOTSTRAP_ROOT:?set this to the reviewed shared NFS directory}"
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$STC_REPO_ROOT/research/sleep-time-compute"
sha256sum -c "$STC_BOOTSTRAP_ROOT/cluster.env.sha256"
. "$STC_BOOTSTRAP_ROOT/cluster.env"
sha256sum -c "$STC_NFS_ROOT/control/post-g4-inputs.sha256"
uv run stc runtime run-matrix \
  --manifest manifests/stc/confirmatory.jsonl \
  --cohort-index manifests/stc/confirmatory-run-index.json \
  --preregistration manifests/stc/preregistration.json \
  --execution-snapshot manifests/stc/g4-confirmatory-execution.json \
  --budget manifests/stc/confirmatory-budget.json \
  --budget-role primary \
  --gate-chain manifests/gates \
  --mode confirmatory \
  --run-id confirmatory-primary \
  --verify-only \
  --receipt manifests/systems/distributed-confirmatory-receipt.json \
  --output "$STC_NFS_ROOT/staging/confirmatory-primary"
uv run stc runtime import-matrix \
  --receipt manifests/systems/distributed-confirmatory-receipt.json \
  --source "$STC_NFS_ROOT/staging/confirmatory-primary" \
  --output results/confirmatory-primary
uv run stc results validate \
  --preregistration manifests/stc/preregistration.json \
  --manifest manifests/stc/confirmatory.jsonl \
  --cohort-index manifests/stc/confirmatory-run-index.json \
  --results results/confirmatory-primary \
  --output manifests/stc/confirmatory-result-validation.json
uv run stc dag import-bundles \
  --input results/confirmatory-primary \
  --manifest-dir manifests/systems/runs
uv run stc dag check
```

Stop here while the evidence controller analyzes/prepares the imported primary
tree and writes
`manifests/stc/confirmatory-clean-rerun-request.json`. Node 0 freezes that
request only after it validates against the primary receipt and result-tree
digests:

```bash
: "${STC_BOOTSTRAP_ROOT:?set this to the reviewed shared NFS directory}"
sha256sum -c "$STC_BOOTSTRAP_ROOT/cluster.env.sha256"
. "$STC_BOOTSTRAP_ROOT/cluster.env"
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$STC_REPO_ROOT/research/sleep-time-compute"
uv run stc runtime run-matrix \
  --manifest manifests/stc/confirmatory.jsonl \
  --cohort-index manifests/stc/confirmatory-run-index.json \
  --preregistration manifests/stc/preregistration.json \
  --execution-snapshot manifests/stc/g4-confirmatory-execution.json \
  --budget manifests/stc/confirmatory-budget.json \
  --budget-role clean \
  --gate-chain manifests/gates \
  --mode confirmatory \
  --run-id confirmatory-clean-rerun \
  --rerun-request manifests/stc/confirmatory-clean-rerun-request.json \
  --verify-request-only \
  --primary-receipt manifests/systems/distributed-confirmatory-receipt.json \
  --primary-results results/confirmatory-primary
mkdir -p "$STC_NFS_ROOT/control/clean-rerun"
STC_CONTROL_FILE="$STC_NFS_ROOT/control/clean-rerun/confirmatory-clean-rerun-request.json"
if test -e "$STC_CONTROL_FILE"
then
  cmp manifests/stc/confirmatory-clean-rerun-request.json "$STC_CONTROL_FILE"
else
  cp manifests/stc/confirmatory-clean-rerun-request.json "$STC_CONTROL_FILE"
fi
sha256sum \
  "$STC_NFS_ROOT/control/clean-rerun/confirmatory-clean-rerun-request.json" \
  > "$STC_NFS_ROOT/control/clean-rerun/checksums.sha256"
```

Both nodes verify the frozen request checksum from their controller package
root before proceeding; the exact command is included below. G5 also requires
a full clean confirmatory rerun. On each node, create a
distinct detached checkout and initially empty dependency/model caches. The
per-node control file makes a failed clean rerun resumable in the same isolated
environment; it is not permission to reuse the primary run:

```bash
: "${STC_BOOTSTRAP_ROOT:?set this to the reviewed shared NFS directory}"
: "${STC_NODE_RANK:?set to 0 on node 0 and 1 on node 1}"
sha256sum -c "$STC_BOOTSTRAP_ROOT/cluster.env.sha256"
. "$STC_BOOTSTRAP_ROOT/cluster.env"
STC_CONTROLLER_REPO="$(git rev-parse --show-toplevel)"
STC_CONTROLLER_ROOT="$STC_CONTROLLER_REPO/research/sleep-time-compute"
cd "$STC_CONTROLLER_ROOT"
sha256sum -c \
  "$STC_NFS_ROOT/control/clean-rerun/checksums.sha256"
STC_CLEAN_ROOT_FILE="$STC_NFS_ROOT/control/clean-root-node-$STC_NODE_RANK.txt"
if test -f "$STC_CLEAN_ROOT_FILE"
then
  read -r STC_CLEAN_ROOT < "$STC_CLEAN_ROOT_FILE"
else
  STC_CLEAN_ROOT="$(mktemp -d \
    "/var/tmp/stc-confirmatory-clean-rank-$STC_NODE_RANK.XXXXXX")"
  printf '%s\n' "$STC_CLEAN_ROOT" > "$STC_CLEAN_ROOT_FILE"
  git -C "$STC_CONTROLLER_REPO" worktree add --detach \
    "$STC_CLEAN_ROOT/repo" \
    "$(git -C "$STC_CONTROLLER_REPO" rev-parse HEAD)"
  mkdir -p "$STC_CLEAN_ROOT/caches/uv" \
    "$STC_CLEAN_ROOT/caches/xdg" \
    "$STC_CLEAN_ROOT/caches/torch" \
    "$STC_CLEAN_ROOT/caches/huggingface"
  test -z "$(find "$STC_CLEAN_ROOT/caches" -mindepth 2 -print -quit)"
fi
export UV_CACHE_DIR="$STC_CLEAN_ROOT/caches/uv"
export UV_PROJECT_ENVIRONMENT="$STC_CLEAN_ROOT/venv"
export XDG_CACHE_HOME="$STC_CLEAN_ROOT/caches/xdg"
export TORCH_HOME="$STC_CLEAN_ROOT/caches/torch"
export HF_HOME="$STC_CLEAN_ROOT/caches/huggingface"
cd "$STC_CLEAN_ROOT/repo/research/sleep-time-compute"
git diff --quiet
git diff --cached --quiet
uv sync --frozen --extra accelerator
```

Launch this command concurrently from those clean checkouts:

```bash
: "${STC_BOOTSTRAP_ROOT:?set this to the reviewed shared NFS directory}"
: "${STC_NODE_RANK:?set to 0 on node 0 and 1 on node 1}"
case "$STC_NODE_RANK" in 0|1) ;; *) exit 2 ;; esac
sha256sum -c "$STC_BOOTSTRAP_ROOT/cluster.env.sha256"
. "$STC_BOOTSTRAP_ROOT/cluster.env"
STC_CONTROLLER_REPO="$(git rev-parse --show-toplevel)"
STC_CONTROLLER_ROOT="$STC_CONTROLLER_REPO/research/sleep-time-compute"
cd "$STC_CONTROLLER_ROOT"
sha256sum -c "$STC_NFS_ROOT/control/post-g4-inputs.sha256"
sha256sum -c \
  "$STC_NFS_ROOT/control/clean-rerun/checksums.sha256"
STC_CLEAN_ROOT_FILE="$STC_NFS_ROOT/control/clean-root-node-$STC_NODE_RANK.txt"
read -r STC_CLEAN_ROOT < "$STC_CLEAN_ROOT_FILE"
export UV_CACHE_DIR="$STC_CLEAN_ROOT/caches/uv"
export UV_PROJECT_ENVIRONMENT="$STC_CLEAN_ROOT/venv"
export XDG_CACHE_HOME="$STC_CLEAN_ROOT/caches/xdg"
export TORCH_HOME="$STC_CLEAN_ROOT/caches/torch"
export HF_HOME="$STC_CLEAN_ROOT/caches/huggingface"
cd "$STC_CLEAN_ROOT/repo/research/sleep-time-compute"
git diff --quiet
git diff --cached --quiet
uv run --no-sync stc runtime run-matrix \
  --manifest "$STC_CONTROLLER_ROOT/manifests/stc/confirmatory.jsonl" \
  --cohort-index "$STC_CONTROLLER_ROOT/manifests/stc/confirmatory-run-index.json" \
  --preregistration "$STC_CONTROLLER_ROOT/manifests/stc/preregistration.json" \
  --execution-snapshot "$STC_CONTROLLER_ROOT/manifests/stc/g4-confirmatory-execution.json" \
  --budget "$STC_CONTROLLER_ROOT/manifests/stc/confirmatory-budget.json" \
  --budget-role clean \
  --gate-chain "$STC_CONTROLLER_ROOT/manifests/gates" \
  --mode confirmatory \
  --run-id confirmatory-clean-rerun \
  --rerun-request "$STC_NFS_ROOT/control/clean-rerun/confirmatory-clean-rerun-request.json" \
  --backend a100 \
  --launcher torchrun \
  --nnodes 2 --nproc-per-node 4 --node-rank "$STC_NODE_RANK" \
  --master-addr "$STC_MASTER_ADDR" --master-port "$STC_MASTER_PORT" \
  --container-ref "$STC_CONTAINER_REF" \
  --probe "$STC_BOOTSTRAP_ROOT/probes/cluster.json" \
  --counter-profile "$STC_CONTROLLER_ROOT/configs/accelerator/counter-profile.yaml" \
  --shard-count 8 \
  --require-clean-worktree \
  --fresh-cache-root "$STC_CLEAN_ROOT/caches" \
  --resume verify-complete \
  --output "$STC_NFS_ROOT/staging/confirmatory-clean-rerun"
```

After both clean launchers exit, node 0 returns to the controller checkout and
creates a distinct receipt and canonical import:

```bash
: "${STC_BOOTSTRAP_ROOT:?set this to the reviewed shared NFS directory}"
sha256sum -c "$STC_BOOTSTRAP_ROOT/cluster.env.sha256"
. "$STC_BOOTSTRAP_ROOT/cluster.env"
STC_CONTROLLER_REPO="$(git rev-parse --show-toplevel)"
STC_CONTROLLER_ROOT="$STC_CONTROLLER_REPO/research/sleep-time-compute"
cd "$STC_CONTROLLER_ROOT"
uv run stc runtime run-matrix \
  --manifest manifests/stc/confirmatory.jsonl \
  --cohort-index manifests/stc/confirmatory-run-index.json \
  --preregistration manifests/stc/preregistration.json \
  --execution-snapshot manifests/stc/g4-confirmatory-execution.json \
  --budget manifests/stc/confirmatory-budget.json \
  --budget-role clean \
  --gate-chain manifests/gates \
  --mode confirmatory \
  --run-id confirmatory-clean-rerun \
  --rerun-request "$STC_NFS_ROOT/control/clean-rerun/confirmatory-clean-rerun-request.json" \
  --verify-only \
  --receipt manifests/systems/distributed-confirmatory-clean-rerun-receipt.json \
  --output "$STC_NFS_ROOT/staging/confirmatory-clean-rerun"
uv run stc runtime import-matrix \
  --receipt manifests/systems/distributed-confirmatory-clean-rerun-receipt.json \
  --source "$STC_NFS_ROOT/staging/confirmatory-clean-rerun" \
  --output results/confirmatory-clean-rerun
uv run stc results validate \
  --preregistration manifests/stc/preregistration.json \
  --manifest manifests/stc/confirmatory.jsonl \
  --cohort-index manifests/stc/confirmatory-run-index.json \
  --results results/confirmatory-clean-rerun \
  --output manifests/stc/confirmatory-clean-rerun-result-validation.json
uv run stc dag import-bundles \
  --input results/confirmatory-clean-rerun \
  --manifest-dir manifests/systems/runs
uv run stc dag check
```

The receipt records the detached checkout and the four initially empty cache
roots. Its run ID, child bundle IDs, result-tree digest, and output path must
differ from the primary run. Evidence-owned confirmatory and clean-rerun
controllers consume these receipts and canonical imported trees; a valid
receipt disables child execution in those wrappers.

Capacity production is independent of confirmatory outcomes. Once the
capacity snapshot exists, execute the seven frozen phases in their
preregistered global-ledger order. On both nodes, the exact launcher surface is:

```bash
: "${STC_BOOTSTRAP_ROOT:?set this to the reviewed shared NFS directory}"
: "${STC_NODE_RANK:?set to 0 on node 0 and 1 on node 1}"
: "${STC_CAPACITY_PHASE:?set to one frozen capacity phase ID}"
case "$STC_NODE_RANK" in 0|1) ;; *) exit 2 ;; esac
sha256sum -c "$STC_BOOTSTRAP_ROOT/cluster.env.sha256"
. "$STC_BOOTSTRAP_ROOT/cluster.env"
sha256sum -c "$STC_NFS_ROOT/control/post-g4-inputs.sha256"
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$STC_REPO_ROOT/research/sleep-time-compute"
uv run stc runtime run-capacity-phase \
  --phase-plan configs/systems/capacity-production-phases.yaml \
  --phase-id "$STC_CAPACITY_PHASE" \
  --capacity-preregistration manifests/stc/capacity-preregistration.json \
  --capacity-gate manifests/capacity-gates/G2-CAP.json \
  --execution-snapshot manifests/stc/g4-capacity-execution.json \
  --aggregate-budget manifests/stc/capacity-aggregate-budget.json \
  --actual-ledger results/capacity/actual-rre-ledger.jsonl \
  --backend a100 --launcher torchrun \
  --nnodes 2 --nproc-per-node 4 --node-rank "$STC_NODE_RANK" \
  --master-addr "$STC_MASTER_ADDR" --master-port "$STC_MASTER_PORT" \
  --container-ref "$STC_CONTAINER_REF" \
  --probe "$STC_BOOTSTRAP_ROOT/probes/cluster.json" \
  --counter-profile "$STC_COUNTER_PROFILE" \
  --shard-count 8 --resume verify-complete \
  --staging-root "$STC_NFS_ROOT/staging/capacity"
```

Run both-node `scaling-fit`, then node 0 verifies/imports the mapped receipt
and seals the fit:

```bash
uv run stc runtime run-capacity-phase \
  --phase-plan configs/systems/capacity-production-phases.yaml \
  --phase-id scaling-fit \
  --capacity-preregistration manifests/stc/capacity-preregistration.json \
  --capacity-gate manifests/capacity-gates/G2-CAP.json \
  --execution-snapshot manifests/stc/g4-capacity-execution.json \
  --aggregate-budget manifests/stc/capacity-aggregate-budget.json \
  --actual-ledger results/capacity/actual-rre-ledger.jsonl \
  --verify-only --staging-root "$STC_NFS_ROOT/staging/capacity"
uv run stc runtime import-matrix \
  --receipt manifests/systems/distributed-scaling-fit-receipt.json \
  --source "$STC_NFS_ROOT/staging/capacity/scaling-fit" \
  --output results/scaling/a100-fit
uv run stc results validate \
  --capacity-preregistration manifests/stc/capacity-preregistration.json \
  --capacity-gate manifests/capacity-gates/G2-CAP.json \
  --execution-role fit \
  --manifest manifests/stc/scaling-cells.jsonl \
  --cohort-index manifests/stc/scaling-cohort-index.json \
  --budget manifests/stc/scaling-budget.json \
  --aggregate-budget manifests/stc/capacity-aggregate-budget.json \
  --actual-ledger results/capacity/actual-rre-ledger.jsonl \
  --results results/scaling/a100-fit \
  --output manifests/stc/scaling-fit-result-validation.json
uv run stc scaling fit \
  --stage provisional \
  --config configs/stc/design/scaling.yaml \
  --hardware-config configs/stc/design/scaling-hardware.yaml \
  --cohort-index manifests/stc/scaling-cohort-index.json \
  --fit-results results/scaling/a100-fit \
  --validation-report manifests/stc/scaling-fit-result-validation.json \
  --output manifests/stc/scaling-provisional.json
uv run stc scaling holdout-unlock \
  --capacity-preregistration manifests/stc/capacity-preregistration.json \
  --capacity-gate manifests/capacity-gates/G2-CAP.json \
  --provisional-artifact manifests/stc/scaling-provisional.json \
  --fit-receipt manifests/systems/distributed-scaling-fit-receipt.json \
  --fit-validation manifests/stc/scaling-fit-result-validation.json \
  --output manifests/stc/scaling-holdout-unlock.json
```

Now run both-node `scaling-holdout`, verify/import it, and validate it without
refitting:

```bash
uv run stc runtime run-capacity-phase \
  --phase-plan configs/systems/capacity-production-phases.yaml \
  --phase-id scaling-holdout \
  --capacity-preregistration manifests/stc/capacity-preregistration.json \
  --capacity-gate manifests/capacity-gates/G2-CAP.json \
  --execution-snapshot manifests/stc/g4-capacity-execution.json \
  --aggregate-budget manifests/stc/capacity-aggregate-budget.json \
  --actual-ledger results/capacity/actual-rre-ledger.jsonl \
  --provisional-artifact manifests/stc/scaling-provisional.json \
  --holdout-unlock manifests/stc/scaling-holdout-unlock.json \
  --verify-only --staging-root "$STC_NFS_ROOT/staging/capacity"
uv run stc runtime import-matrix \
  --receipt manifests/systems/distributed-scaling-holdout-receipt.json \
  --source "$STC_NFS_ROOT/staging/capacity/scaling-holdout" \
  --output results/scaling/a100-holdout
uv run stc results validate \
  --capacity-preregistration manifests/stc/capacity-preregistration.json \
  --capacity-gate manifests/capacity-gates/G2-CAP.json \
  --execution-role holdout \
  --manifest manifests/stc/scaling-cells.jsonl \
  --cohort-index manifests/stc/scaling-cohort-index.json \
  --budget manifests/stc/scaling-budget.json \
  --aggregate-budget manifests/stc/capacity-aggregate-budget.json \
  --actual-ledger results/capacity/actual-rre-ledger.jsonl \
  --provisional-artifact manifests/stc/scaling-provisional.json \
  --holdout-unlock manifests/stc/scaling-holdout-unlock.json \
  --results results/scaling/a100-holdout \
  --output manifests/stc/scaling-holdout-result-validation.json
```

Coverage follows the same physical launcher exactly, first for
`coverage-cardinality-fit` and `coverage-fixed-bits-fit`. Node 0 maps each
phase to its frozen receipt, imports to
`results/coverage-scaling/{cardinality,fixed-bits}/fit`, and writes the two
immutable fit validations. It then executes:

```bash
uv run stc coverage fit \
  --stage provisional \
  --capacity-preregistration manifests/stc/capacity-preregistration.json \
  --capacity-gate manifests/capacity-gates/G2-CAP.json \
  --config configs/stc/design/coverage-scaling.yaml \
  --cohort-index manifests/stc/coverage-scaling-cohort-index.json \
  --cardinality-fit-results results/coverage-scaling/cardinality/fit \
  --fixed-bits-fit-results results/coverage-scaling/fixed-bits/fit \
  --cardinality-fit-validation manifests/stc/coverage-cardinality-fit-result-validation.json \
  --fixed-bits-fit-validation manifests/stc/coverage-fixed-bits-fit-result-validation.json \
  --output manifests/stc/coverage-provisional.json
uv run stc coverage holdout-unlock \
  --capacity-preregistration manifests/stc/capacity-preregistration.json \
  --capacity-gate manifests/capacity-gates/G2-CAP.json \
  --provisional-artifact manifests/stc/coverage-provisional.json \
  --cardinality-fit-receipt manifests/systems/distributed-coverage-cardinality-fit-receipt.json \
  --fixed-bits-fit-receipt manifests/systems/distributed-coverage-fixed-bits-fit-receipt.json \
  --output manifests/stc/coverage-holdout-unlock.json
```

Only then run `coverage-cardinality-holdout` and
`coverage-fixed-bits-holdout`, import to the corresponding `/holdout` roots,
and write the two holdout validations, each binding the provisional/unlock,
aggregate ledger, and mapped holdout receipt. A combined arm or combined
fit+holdout validation is forbidden.

Finally run `parametric-information-all`, verify
`distributed-parametric-information-receipt.json`, import to
`results/parametric-information`, and validate against its manifest/cohort,
track/aggregate budgets, canonical ledger/head, fresh child seeds/base
restores, frozen decoder, and complete-state inventory.

The executor resolves exactly \(120n_{\rm scaling}\),
\(160n_{\rm coverage}\), and \(48n_{\rm information}\) logical child
references plus the four charged post-unlock scaling anchors. The scaling
split is 96 fit and 24 holdout logical configurations; coverage has four
immutable arm×stage validations. No H100 child runs. If a node fails before a
top-level manifest is published, both operators rerun only the same phase with
`--resume verify-complete`; no manual shard copy/merge or outcome-driven phase
reordering is allowed.

Authoring the commands above is a pre-G4 documentation action; executing them
remains post-G4. Before the runbook commit, exercise its exact stage/option
surface through the fake backend and create the previously declared small
receipt:

```bash
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$STC_REPO_ROOT/research/sleep-time-compute"
uv run pytest tests/accelerator/test_runbook_smoke.py tests/test_cli.py -q
uv run stc accelerator runbook-smoke \
  --runbook "$STC_REPO_ROOT/runbook/sleep-time-compute-a100.md" \
  --config configs/accelerator/a100-smoke.yaml \
  --backend fake \
  --output manifests/a100-runbook-smoke.json
uv run stc validate --root .
```

Expected: the receipt resolves the final runbook digest, contains all five
closed pre-G4 hardware stage names, both core production stage names, and all
seven capacity phase IDs in their prerequisite order. It encodes hardware
world sizes 1/1/4/8/8, production world size 8, both snapshot types, every
gate edge, and is explicitly `SYNTHETIC`. Mutation tests remove/reorder a
stage, alter a world size, substitute a snapshot, inject a confirmatory receipt
into capacity, change a phase-plan mapping, drop a prerequisite, inject an
unknown CLI option, relabel a fake metric, or change the runbook after receipt
creation; every case fails. This command validates syntax/lineage only and
does not execute any post-G4 matrix.

- [ ] **Step 5: Commit the runbook and small manifest**

```bash
git -C ../.. add runbook/sleep-time-compute-a100.md \
  research/sleep-time-compute/manifests/a100-runbook-smoke.json
for STC_A100_RUN_ID in \
  a100-single-gpu-semantic \
  a100-single-gpu-resource-smoke \
  a100-single-node-contention \
  a100-two-node-gate \
  a100-confirmatory
do
  STC_A100_MANIFEST="research/sleep-time-compute/manifests/accelerator/runs/$STC_A100_RUN_ID.json"
  if test -f "../../$STC_A100_MANIFEST"
  then
    git -C ../.. add "$STC_A100_MANIFEST"
  fi
done
git -C ../.. add \
  research/sleep-time-compute/registry/artifacts.jsonl
git -C ../.. commit -m "docs: add sleep-time compute A100 runbook"
```

---

### Task 9: Evaluate G4 systems correctness

**Files:**
- Create: `research/sleep-time-compute/src/stc_research/systems/report.py`
- Create: `research/sleep-time-compute/schemas/systems/g4-evidence-v1.schema.json`
- Create: `research/sleep-time-compute/tests/systems/test_g4_report.py`
- Create: `research/sleep-time-compute/reports/g4-tests.xml`
- Create: `research/sleep-time-compute/reports/g4-systems-evidence.json`
- Create: `research/sleep-time-compute/reports/g4-systems-audit.md`
- Create: `research/sleep-time-compute/manifests/systems/g4-inputs.json`
- Create: `research/sleep-time-compute/manifests/gate-inputs/G4.json`
- Create: `research/sleep-time-compute/manifests/gates/G4.json`
- Create after PASS G4:
  `research/sleep-time-compute/manifests/stc/g4-confirmatory-execution.json`
- Create after PASS G4 and PASS G2-CAP:
  `research/sleep-time-compute/manifests/stc/g4-capacity-execution.json`
- Create: `research/sleep-time-compute/schemas/systems/execution-snapshot-v1.schema.json`
- Create: `research/sleep-time-compute/systems/INFRASTRUCTURE-BLUEPRINT.md`
- Modify: `research/sleep-time-compute/registry/artifacts.jsonl`
- Modify: `research/sleep-time-compute/src/stc_research/cli.py`
- Modify: `research/sleep-time-compute/tests/test_cli.py`

**Interfaces:**
- Consumes: a JUnit test report plus registered lifecycle, DSE,
  deletion-with-fault-outcomes, representation-refresh, fake-accelerator, and optional
  measured-accelerator manifests. Selecting `a100-confirmatory` additionally
  requires the four fixed registered prerequisite run IDs and their ordered
  digest chain; a confirmatory manifest alone is inadmissible.
- Produces: immutable `G4SystemsEvidence`, a rendered Markdown view, a frozen
  systems input index, a generic typed gate-input manifest, a typed G4
  decision, two non-substitutable execution snapshots, and explicit
  hardware-claim blockers.

- [ ] **Step 1: Run the complete local suite**

```bash
mkdir -p results/system-runs results/a100-runs reports
uv run pytest tests/systems tests/runtime tests/accelerator tests/test_cli.py -q \
  --junitxml=reports/g4-tests.xml
uv run stc runtime lifecycle \
  --config configs/systems/lifecycle-smoke.yaml \
  --output results/system-runs/g4-lifecycle
uv run stc systems dse \
  --config configs/systems/dse-smoke.yaml \
  --output results/system-runs/g4-dse
uv run stc systems deletion-audit \
  --profile configs/systems/deletion-profile.yaml \
  --fixture tests/runtime/fixtures/deletion-audit \
  --output results/system-runs/g4-deletion
uv run stc systems index-refresh run \
  --manifest manifests/systems/index-refresh-G2.json \
  --output results/system-runs/g4-index-refresh
uv run stc accelerator run \
  --config configs/accelerator/a100-smoke.yaml \
  --backend fake \
  --output results/a100-runs/g4-fake
uv run stc accelerator validate results/a100-runs/g4-fake \
  --require-measurement-status SYNTHETIC
uv run stc dag import-bundles \
  --input results/system-runs \
  --manifest-dir manifests/systems/runs
uv run stc dag import-bundles \
  --input results/a100-runs/g4-fake \
  --manifest-dir manifests/accelerator/runs
uv run stc dag check
```

Expected: all tests pass; `g4-lifecycle`, `g4-dse`, `g4-deletion`, and
`g4-index-refresh` exist in `manifests/systems/runs/`; `g4-fake` exists in
`manifests/accelerator/runs/` with `measurement_status=SYNTHETIC`; and every
registered manifest resolves to a checksum-valid bundle.

- [ ] **Step 2: Audit claim boundaries**

The G4 report explicitly verifies:

- no simulator output is labeled hardware measurement;
- no fake accelerator output is claim-admissible as measurement;
- the retained six-class ownership partition, all five transient additions,
  incremental peak state, transient overhead, resident base copies,
  non-state execution bytes, total physical HBM/fleet peak, per-term caps/GC owners, and
  allocator/storage-telemetry reconciliation remain visible and pass;
- the `ServingHead` state machine passes every delete-vs-validation/CAS/read
  interleaving, immediate-deny check, reachability-root closure, reader
  quiescence/hazard, and worker-fencing test; timeout or generation order alone
  never authorizes reclamation;
- energy boundaries remain visible in every table;
- deletion claims distinguish deny, online removal, backup/key expiry, lineage
  invalidation, and behavioral residual;
- the five representation-refresh arms share payload/query/delete traces and
  quality constraints; the report recomputes affected-set work and charges
  federation/translation/migration/peak/delete fan-out without treating
  whole-store replacement as universal, and enforces
  `claim_admissibility=CONFORMANCE_ONLY`;
- all four factorial cells share the cutoff digest;
- no runtime method can access oracle-only records.

`G4SystemsEvidence` is validated against `g4-evidence-v1.schema.json` and
contains the exact `manifests/gates/G3.json` digest, test-report digest and
counts, selected run IDs and manifest/payload digests, artifact-DAG digest,
queue stability/SLO results, lifecycle charge conservation, every fault
outcome, all deletion completion dimensions, tombstone/linkability results,
the typed retained/peak storage ledger with per-term caps and GC owners,
telemetry calibration source/error/bound, every `ServingHead` transition and
reclamation root/reader proof,
accelerator evidence classes and per-metric provenance for KV read, fast-state
RMW, batching, adapter load/apply/update, direct-counter HBM bytes,
direct-counter host-DRAM bytes, GPU-board energy, host energy or its typed
unavailability, and modeled storage/network energy; unresolved blockers,
evaluator version, and environment digest. Every accelerator record binds its
registered run ID plus bundle, manifest, probe, counter-profile,
container/environment, and raw-shard digests. The Markdown audit is generated
from this JSON and is never accepted as gate input.
`manifests/systems/g4-inputs.json` freezes the G3 predecessor and exact selected
IDs/digests so extra files in either manifest directory cannot alter the gate.
For the measured branch it freezes
`a100-single-gpu-semantic`, `a100-single-gpu-resource-smoke`,
`a100-single-node-contention`, `a100-two-node-gate`, and
`a100-confirmatory`, along with every prerequisite edge and the probe,
counter-profile, config, check-result, and manifest digest needed to rehash
the chain offline.

- [ ] **Step 3: Generate the infrastructure blueprint**

Document the wake inference/training plane, sleep training plane, versioned
memory fabric, control plane, and raw/external/latent/user-parametric storage
tiers. Include lifecycle sequence, data movement, queue admission and overload,
snapshot/version publication, fast-path pinning, rollback, deletion, offload,
cluster co-location/separation DSE inputs, failure domains, SLOs, and the exact
metrics needed to choose among shared and separated clusters. Every arrow
identifies the object type, consistency boundary, and charged bytes.

- [ ] **Step 4: Build typed evidence before recording the gate**

```bash
uv run pytest tests/systems/test_g4_report.py tests/test_cli.py -q
uv run stc systems report \
  --predecessor manifests/gates/G3.json \
  --test-report reports/g4-tests.xml \
  --storage-accounting-config configs/systems/storage-accounting.yaml \
  --system-manifest-dir manifests/systems/runs \
  --system-run-id g4-lifecycle \
  --system-run-id g4-dse \
  --system-run-id g4-deletion \
  --system-run-id g4-index-refresh \
  --accelerator-manifest-dir manifests/accelerator/runs \
  --accelerator-run-id g4-fake \
  --input-index manifests/systems/g4-inputs.json \
  --json reports/g4-systems-evidence.json \
  --markdown reports/g4-systems-audit.md
uv run stc gate inputs assemble G4 \
  --root . \
  --predecessor manifests/gates/G3.json \
  --systems-evidence reports/g4-systems-evidence.json \
  --systems-input-index manifests/systems/g4-inputs.json \
  --output manifests/gate-inputs/G4.json
uv run stc gate evaluate G4 \
  --inputs manifests/gate-inputs/G4.json \
  --output manifests/gates/G4.json
uv run stc execution snapshot build-core \
  --gate manifests/gates/G4.json \
  --g4-input-index manifests/systems/g4-inputs.json \
  --preregistration manifests/stc/preregistration.json \
  --design manifests/stc/confirmatory.jsonl \
  --cohort-index manifests/stc/confirmatory-run-index.json \
  --analysis manifests/stc/analysis-plan.json \
  --confirmatory-budget manifests/stc/confirmatory-budget.json \
  --program-budget-envelope configs/stc/design/program-budget-envelope.yaml \
  --rre-reference configs/stc/design/rre-reference-native.yaml \
  --coarse-feasibility manifests/stc/capacity-coarse-feasibility.json \
  --environment manifests/accelerator/runs/a100-confirmatory.json \
  --output manifests/stc/g4-confirmatory-execution.json
uv run stc execution snapshot build-capacity \
  --gate manifests/gates/G4.json \
  --g4-input-index manifests/systems/g4-inputs.json \
  --core-preregistration manifests/stc/preregistration.json \
  --capacity-preregistration manifests/stc/capacity-preregistration.json \
  --capacity-gate manifests/capacity-gates/G2-CAP.json \
  --capacity-phase-plan configs/systems/capacity-production-phases.yaml \
  --capacity-stakes manifests/stc/deployment-stakes-G2-CAP.json \
  --capacity-calibration-summary manifests/stc/capacity-calibration-summary.json \
  --aggregate-budget manifests/stc/capacity-aggregate-budget.json \
  --program-budget-envelope configs/stc/design/program-budget-envelope.yaml \
  --rre-reference configs/stc/design/rre-reference-native.yaml \
  --coarse-feasibility manifests/stc/capacity-coarse-feasibility.json \
  --target-environment manifests/accelerator/runs/a100-confirmatory.json \
  --require-exact-calibration-transport \
  --output manifests/stc/g4-capacity-execution.json
uv run stc dag check
```

Expected: the report command re-resolves every frozen digest and fails on a
missing, stale, overwritten, class-upgraded, unregistered, or G3-mismatched
input. The generic assembler and gate evaluator independently validate the
predecessor and JSON/schema/systems-index/DAG digests. Tests mutate the G3
record, systems evidence, systems input index, and assembled G4 input manifest;
every mutation blocks evaluation with a stable code.
Snapshot tests swap the core/capacity types, inject a confirmatory result
receipt into the capacity snapshot, mutate any capacity phase/config/stakes/
power/manifest/budget/CAL lineage edge, or change A100 container,
driver/CUDA, clock/power, serving stack, model/tokenizer, or cost-profile
identity relative to CAL; every mutation prevents snapshot creation or
prelaunch rederivation.
Local implementation evidence may pass without cluster access, but every paper
claim that requires measured A100 HBM, host-DRAM, latency, or energy remains
explicitly blocked until an `ADMISSIBLE` target manifest exists.

When Task 8 succeeds, rebuild the same typed evidence and G4 decision with the
fixed run ID `a100-confirmatory` explicitly selected. The report never scans
the accelerator manifest directory and silently promotes whatever it finds:

```bash
uv run stc systems report \
  --predecessor manifests/gates/G3.json \
  --test-report reports/g4-tests.xml \
  --storage-accounting-config configs/systems/storage-accounting.yaml \
  --system-manifest-dir manifests/systems/runs \
  --system-run-id g4-lifecycle \
  --system-run-id g4-dse \
  --system-run-id g4-deletion \
  --system-run-id g4-index-refresh \
  --accelerator-manifest-dir manifests/accelerator/runs \
  --accelerator-run-id g4-fake \
  --accelerator-run-id a100-confirmatory \
  --accelerator-prerequisite-run-id a100-single-gpu-semantic \
  --accelerator-prerequisite-run-id a100-single-gpu-resource-smoke \
  --accelerator-prerequisite-run-id a100-single-node-contention \
  --accelerator-prerequisite-run-id a100-two-node-gate \
  --input-index manifests/systems/g4-inputs.json \
  --json reports/g4-systems-evidence.json \
  --markdown reports/g4-systems-audit.md
uv run stc gate inputs assemble G4 \
  --root . \
  --predecessor manifests/gates/G3.json \
  --systems-evidence reports/g4-systems-evidence.json \
  --systems-input-index manifests/systems/g4-inputs.json \
  --output manifests/gate-inputs/G4.json
uv run stc gate evaluate G4 \
  --inputs manifests/gate-inputs/G4.json \
  --output manifests/gates/G4.json
uv run stc dag check
```

The second branch fails unless all four named prerequisite runs and
`a100-confirmatory` are registered and digest-valid, their stage order,
world-size, check codes, and prerequisite edges match exactly, and the final
run is `measurement_status=ADMISSIBLE` and target-matched. Parser tests require
all four distinct `--accelerator-prerequisite-run-id` occurrences when the
target run is selected; mutation tests remove, reorder, duplicate, or alter a
gate manifest/edge/check and fail before evidence generation. When the target
run is absent, the first branch remains the authoritative typed decision and
carries an explicit measured-hardware blocker.

- [ ] **Step 5: Commit**

```bash
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: audit sleep-time systems correctness"
```

## Dependency Graph

```text
shared schemas
├── Task 1 accounting ─┬── Task 4 queue/DSE
│                     └── Task 7 accelerator harness ─── Task 8 pre-G4 A100 measurement
└── Task 2 versioning ─── Task 3 planes ─┐
                       └── Task 5 faults ├── Task 6 lifecycle integration
                                        ┘
all local tasks ───────────────────────────── Task 9 G4
Task 9 PASS + core snapshot ───────────────── Task 8 core production
Task 9 PASS + G2-CAP + capacity snapshot ──── Task 8 capacity phases
```

Tasks 4 and 7 may run in parallel after Task 1. Task 8 is the only task that
requires the external A100 cluster. Its microbenchmark branch may feed G4; its
297n primary and 297n clean-rerun branches require PASS G4 plus the core
snapshot. Its \(120n_{\rm scaling}+4\) anchors,
\(160n_{\rm coverage}\), and \(48n_{\rm information}\) capacity branches
require PASS G4, PASS G2-CAP, and the separate capacity snapshot; they never
depend on confirmatory outcome receipts.

## Validity Traps

- Never compute lifecycle cost from sleep compute alone; charge construction,
  verification, publication, retrieval/application, migration, invalidation,
  deletion, rollback, lineage, and raw retention.
- Match resource caps, not forced resource consumption. Record unused budgets
  and infeasibility masks.
- A queue that grows indefinitely is not a valid deployment point even if its
  average service cost is low.
- Wake latency must include contention from sleep jobs and data movement.
- Manifest-last atomic publication in one reference store is not evidence of a
  distributed transaction across independent stores.
- Any stale `ServingHead` field, deny-before-emission failure, governance epoch
  regression, or reclamation before reachability and quiescence/fencing proof
  is a correctness failure even when artifact content validates.
- Cryptographic erasure, physical removal, and behavioral unlearning answer
  different questions.
- H100 projections are separate DSE points; they never inherit A100
  measurements.
- The systems paper extension is allowed only if it owns a distinct runtime or
  scheduler claim, failure-injection study, and accelerator result set.
