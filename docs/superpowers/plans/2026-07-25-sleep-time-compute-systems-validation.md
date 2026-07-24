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
- Immediate deny records override already pinned reads. Publication uses compare-and-swap on both parent generation and deletion watermark.
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
- Create: `tests/systems/test_accounting.py`
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
peak_storage_bytes >= final_storage_bytes >= 0
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

`LifecycleCostSummary` includes totals, per-phase totals, peak/final storage,
wake latency p50/p95/p99, queue age p50/p95/p99, deadline misses, dropped jobs,
and the three energy boundaries. Quantiles over empty collections serialize as
`null`, not zero.

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
uv run pytest tests/systems/test_accounting.py tests/runtime/test_contracts.py -q
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
  `PublishOutcome`, and `MemoryFabric`.
- Requires: opaque tenant handles and benchmark/method contracts.

- [ ] **Step 1: Write failure-first publication tests**

Cover:

```python
MemoryFabric.pin_read(auth, tenant_handle) -> PublishedView
MemoryFabric.snapshot(tenant_handle, information_cutoff) -> SnapshotRef
MemoryFabric.stage(snapshot, artifacts) -> ShadowCandidate
MemoryFabric.mark_ready(candidate, verification) -> ReadyCandidate
MemoryFabric.publish(
    candidate,
    expected_parent_generation,
    expected_deletion_watermark,
) -> PublishOutcome
MemoryFabric.rollback(tenant_handle, bad_generation, reason) -> PublishedView
```

Assert that:

- readers observe one complete immutable generation;
- a candidate is invisible until every artifact and manifest digest validates;
- manifest publication happens last;
- stale parent generation or deletion watermark rejects publication;
- retrying the same completed operation is idempotent;
- base-model or tokenizer mismatch invalidates latent/KV and parametric
  artifacts;
- rollback can select only an authorized last-known-good generation.

`ArtifactManifest` records opaque tenant-scoped artifact/source/parent handles,
snapshot version, generation, base-model and tokenizer hashes, compiler
name/version/config digest, artifact class, privacy scope, deletion epoch,
encrypted object location, integrity value, byte counts, readiness, and lineage
edges. Operational metadata never stores a raw-payload or reversible content
hash. Integrity uses a tenant-keyed MAC or a digest of encrypted/packed artifact
bytes; the separately encrypted deletable sidecar owns any payload mapping.

- [ ] **Step 2: Confirm failure and implement a filesystem reference store**

```bash
uv run pytest tests/runtime/test_versioning.py -q
```

Use temporary directories, atomic rename on one filesystem, tenant-keyed or
encrypted-object integrity values, and compare-and-swap metadata. Do not claim
the reference store implements a distributed consensus protocol.

- [ ] **Step 3: Run property tests**

Generate arbitrary valid stage/ready/publish/rollback sequences. The invariant
is that a pinned reader sees either the old or new complete generation and
never a partial mixture.

```bash
uv run pytest tests/runtime/test_versioning.py -q
```

Expected: at least eight tests pass, including randomized operation sequences.

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
- Wake path: authorize → pin generation → answer → append observable event →
  charge latency.
- Sleep path: freeze cutoff → snapshot → plan → materialize shadow → verify →
  canary → publish.
- Control path: admission, cadence, deadline, tier movement, overload fallback.

- [ ] **Step 1: Write boundary and cutoff tests**

Test:

- an answer call receives `QueryObservable` and a pinned `PublishedView`;
- all four inline/deferred × identity/consolidating cells share the exact
  enqueue-time cutoff digest;
- inline and deferred differ only in scheduling;
- identity and consolidating differ only in operator semantics;
- no oracle record is importable through the runtime input protocol;
- failed validation leaves the current generation unchanged.

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
stale generation CAS
stale deletion-watermark CAS
corrupt index
storage-node loss
poison/canary failure
deletion during a pinned read
duplicate event delivery
lost event before acknowledgement
```

Assert no corrupt/truncated candidate publishes, retries are idempotent,
rollback does not revive denied material, and RTO/RPO are measured rather than
assumed.

- [ ] **Step 3: Implement deny-first deletion and failure harness**

Lineage traversal must cover raw, abstract, graph/index, latent/KV,
user-parametric, optimizer, checkpoint, cache, and candidate artifacts.
Separate cryptographic erasure from physical deletion and behavioral
unlearning; do not treat one as proof of the others.

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
  tests/runtime/test_deletion_audit.py \
  tests/runtime/test_failures.py -q
uv run stc systems deletion-audit \
  --profile configs/systems/deletion-profile.yaml \
  --fixture tests/runtime/fixtures/deletion-audit \
  --output /tmp/stc-deletion-audit
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: validate deletion rollback and runtime failures"
```

Expected: deletion, deletion-audit, and failure tests pass; the audit reports
each completion dimension, every named injected-fault outcome, and zero
undeclared replica or retained linkable key.

---

### Task 6: Run the four-cell lifecycle integration protocol

**Files:**
- Create: `configs/systems/lifecycle-smoke.yaml`
- Modify: `src/stc_research/cli.py`
- Modify: `tests/test_cli.py`
- Create: `tests/runtime/test_lifecycle_integration.py`
- Create: `tests/runtime/test_manifest_runner.py`
- Create: `tests/runtime/test_distributed_manifest_runner.py`
- Create: `tests/runtime/test_matrix_import.py`
- Create: `tests/runtime/fixtures/manifest-smoke.jsonl`
- Create: `manifests/systems/runtime-G2.json`
- Modify: `src/stc_research/runtime/coordinator.py`

**Interfaces:**
- Consumes: benchmark fixture, one frozen destination, accounting, memory
  fabric, queue, and runtime.
- Produces: four immutable system-run bundles.

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

- [ ] **Step 3: Implement the general manifest runner**

`stc runtime run-matrix` reads a frozen logical manifest, resolves the method
registry, and executes every cell through the same wake/sleep coordinator,
versioned fabric, accounting hooks, deletion watermark, and bundle writer used
by the four-cell smoke. It never invokes a method-only shortcut. Fixture mode
may consume a standalone test manifest. Every smoke, confirmatory, scaling, or
public production mode also requires `--preregistration`; the runner verifies
that the supplied logical manifest and cohort-index/public-replay digest are
members of that G2 envelope before expanding child `RunSpec` records.
Confirmatory and scaling modes additionally require a PASS
`--execution-snapshot` whose G4, G4-input-index, preregistration, logical
confirmatory manifest/run index, scaling manifest/cohort index/budget,
scaling design/hardware configurations, analysis-plan, checkout, and
environment digests all resolve before any child starts. The scaling runner
also requires `--budget`, `--scaling-config`, and `--hardware-config`, rejects
any digest other than the ones in the snapshot, and enforces the frozen
135-RRE hard ceiling before launch and again during resume/merge.
The `confirmatory-clean-rerun` run ID additionally requires
`--rerun-request manifests/stc/confirmatory-clean-rerun-request.json`. That
typed evidence-owned request binds the validated primary receipt/result tree,
analysis preparation, requested checkout, isolation policy, and exact
confirmatory inputs. The runner rejects a missing, stale, pre-primary, or
already-consumed request, and the clean-rerun receipt binds its digest.
Scaling additionally requires `--primary-receipt`, `--primary-validation`,
`--clean-receipt`, and `--clean-validation`. Before any scaling worker starts,
the runner proves that both confirmatory trees passed, cover the same frozen
\(297n\) logical child references, and have distinct run IDs, child bundles,
environment/cache identities, and result-tree digests.

The optional `--launcher torchrun` path reuses the accelerator package's tested
container launcher. The cohort index expands logical rows to unique physical
child keys; `sha256(child_key) mod world_size` assigns each key to exactly one
global rank. A rank writes only
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
before every child manifest verifies. Scaling tests also reject a missing or
drifted budget and any planned or actual RRE above 135.
Import tests interrupt the copy, mutate one source byte, inject a symlink, and
precreate the destination; no case may publish a canonical manifest or execute
a child. Clean-rerun tests reject a missing/stale/replayed rerun request or one
that does not bind the imported primary receipt/tree. CLI tests own every
distributed, receipt, rerun-request, and import option used by Task 8.
Scaling tests also prove that omitting or mutating either confirmatory receipt
or validation report prevents worker launch, not merely final merge.

- [ ] **Step 4: Freeze the runtime contract and register manifests**

Register only the small manifests with the artifact DAG. Large generated
payloads remain outside Git.

```bash
uv run stc runtime freeze-contract \
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
  `research/sleep-time-compute/manifests/systems/distributed-scaling-receipt.json`

**Interfaces:**
- Target: two manually managed nodes, each with four A100 80 GB GPUs, NFS, and
  InfiniBand.
- Produces: environment probe, single-GPU gate, two-node gate, confirmatory
  microbenchmark bundle, and artifact-DAG handoff before G4.
- Documents but does not pre-authorize: the post-G4 distributed
  `runtime run-matrix` path for every unique physical child behind the 297
  confirmatory and 112 scaling logical rows.
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
G4 and the evidence controller produces the combined typed snapshot at
`manifests/stc/g4-confirmatory-execution.json`. Despite its legacy filename,
that snapshot explicitly binds both the confirmatory manifest/run index and the
scaling manifest/cohort index/budget. It also binds the preregistration,
analysis plan, G4 decision/input index, checkout, and target environment.

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
mkdir -p "$STC_NFS_ROOT/control"
sha256sum \
  manifests/stc/g4-confirmatory-execution.json \
  manifests/stc/confirmatory.jsonl \
  manifests/stc/confirmatory-run-index.json \
  manifests/stc/scaling-cells.jsonl \
  manifests/stc/scaling-cohort-index.json \
  manifests/stc/scaling-budget.json \
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
mkdir -p "$STC_NFS_ROOT/control/scaling-prereqs"
STC_CONTROL_FILE="$STC_NFS_ROOT/control/scaling-prereqs/distributed-confirmatory-receipt.json"
if test -e "$STC_CONTROL_FILE"
then
  cmp manifests/systems/distributed-confirmatory-receipt.json "$STC_CONTROL_FILE"
else
  cp manifests/systems/distributed-confirmatory-receipt.json "$STC_CONTROL_FILE"
fi
STC_CONTROL_FILE="$STC_NFS_ROOT/control/scaling-prereqs/confirmatory-result-validation.json"
if test -e "$STC_CONTROL_FILE"
then
  cmp manifests/stc/confirmatory-result-validation.json "$STC_CONTROL_FILE"
else
  cp manifests/stc/confirmatory-result-validation.json "$STC_CONTROL_FILE"
fi
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
STC_CONTROL_FILE="$STC_NFS_ROOT/control/scaling-prereqs/distributed-confirmatory-clean-rerun-receipt.json"
if test -e "$STC_CONTROL_FILE"
then
  cmp manifests/systems/distributed-confirmatory-clean-rerun-receipt.json \
    "$STC_CONTROL_FILE"
else
  cp manifests/systems/distributed-confirmatory-clean-rerun-receipt.json \
    "$STC_CONTROL_FILE"
fi
STC_CONTROL_FILE="$STC_NFS_ROOT/control/scaling-prereqs/confirmatory-clean-rerun-result-validation.json"
if test -e "$STC_CONTROL_FILE"
then
  cmp manifests/stc/confirmatory-clean-rerun-result-validation.json \
    "$STC_CONTROL_FILE"
else
  cp manifests/stc/confirmatory-clean-rerun-result-validation.json \
    "$STC_CONTROL_FILE"
fi
uv run stc dag import-bundles \
  --input results/confirmatory-clean-rerun \
  --manifest-dir manifests/systems/runs
uv run stc dag check
sha256sum \
  "$STC_NFS_ROOT/control/scaling-prereqs/distributed-confirmatory-receipt.json" \
  "$STC_NFS_ROOT/control/scaling-prereqs/confirmatory-result-validation.json" \
  "$STC_NFS_ROOT/control/scaling-prereqs/distributed-confirmatory-clean-rerun-receipt.json" \
  "$STC_NFS_ROOT/control/scaling-prereqs/confirmatory-clean-rerun-result-validation.json" \
  > "$STC_NFS_ROOT/control/scaling-prereqs/checksums.sha256.tmp"
if test -e "$STC_NFS_ROOT/control/scaling-prereqs/checksums.sha256"
then
  cmp "$STC_NFS_ROOT/control/scaling-prereqs/checksums.sha256.tmp" \
    "$STC_NFS_ROOT/control/scaling-prereqs/checksums.sha256"
  rm "$STC_NFS_ROOT/control/scaling-prereqs/checksums.sha256.tmp"
else
  mv "$STC_NFS_ROOT/control/scaling-prereqs/checksums.sha256.tmp" \
    "$STC_NFS_ROOT/control/scaling-prereqs/checksums.sha256"
fi
```

The receipt records the detached checkout and the four initially empty cache
roots. Its run ID, child bundle IDs, result-tree digest, and output path must
differ from the primary run. Evidence-owned confirmatory and clean-rerun
controllers consume these receipts and canonical imported trees; a valid
receipt disables child execution in those wrappers.

After both confirmatory trees validate, launch scaling concurrently on both
nodes from the reviewed controller checkout with the same `STC_NODE_RANK`
assignments:

```bash
: "${STC_BOOTSTRAP_ROOT:?set this to the reviewed shared NFS directory}"
: "${STC_NODE_RANK:?set to 0 on node 0 and 1 on node 1}"
case "$STC_NODE_RANK" in 0|1) ;; *) exit 2 ;; esac
sha256sum -c "$STC_BOOTSTRAP_ROOT/cluster.env.sha256"
. "$STC_BOOTSTRAP_ROOT/cluster.env"
sha256sum -c "$STC_NFS_ROOT/control/scaling-prereqs/checksums.sha256"
unset UV_CACHE_DIR UV_PROJECT_ENVIRONMENT XDG_CACHE_HOME TORCH_HOME HF_HOME
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
STC_CONTROLLER_ROOT="$STC_REPO_ROOT/research/sleep-time-compute"
cd "$STC_CONTROLLER_ROOT"
uv sync --frozen --extra accelerator
uv run stc runtime run-matrix \
  --manifest manifests/stc/scaling-cells.jsonl \
  --cohort-index manifests/stc/scaling-cohort-index.json \
  --budget manifests/stc/scaling-budget.json \
  --scaling-config configs/stc/design/scaling.yaml \
  --hardware-config configs/stc/design/scaling-hardware.yaml \
  --primary-receipt "$STC_NFS_ROOT/control/scaling-prereqs/distributed-confirmatory-receipt.json" \
  --primary-validation "$STC_NFS_ROOT/control/scaling-prereqs/confirmatory-result-validation.json" \
  --clean-receipt "$STC_NFS_ROOT/control/scaling-prereqs/distributed-confirmatory-clean-rerun-receipt.json" \
  --clean-validation "$STC_NFS_ROOT/control/scaling-prereqs/confirmatory-clean-rerun-result-validation.json" \
  --preregistration manifests/stc/preregistration.json \
  --execution-snapshot manifests/stc/g4-confirmatory-execution.json \
  --gate-chain manifests/gates \
  --mode scaling \
  --run-id scaling \
  --backend a100 \
  --launcher torchrun \
  --nnodes 2 --nproc-per-node 4 --node-rank "$STC_NODE_RANK" \
  --master-addr "$STC_MASTER_ADDR" --master-port "$STC_MASTER_PORT" \
  --container-ref "$STC_CONTAINER_REF" \
  --probe "$STC_BOOTSTRAP_ROOT/probes/cluster.json" \
  --counter-profile "$STC_COUNTER_PROFILE" \
  --shard-count 8 \
  --resume verify-complete \
  --output "$STC_NFS_ROOT/staging/scaling"
```

The scaling runner resolves exactly the preregistered \(112n\) logical child
references, preserves the 96-fit/16-holdout roles, executes no H100 child, and
fails before launch or merge if the planned or actual RRE exceeds the frozen
135 ceiling. After both launchers exit, node 0 runs:

```bash
: "${STC_BOOTSTRAP_ROOT:?set this to the reviewed shared NFS directory}"
sha256sum -c "$STC_BOOTSTRAP_ROOT/cluster.env.sha256"
. "$STC_BOOTSTRAP_ROOT/cluster.env"
sha256sum -c "$STC_NFS_ROOT/control/scaling-prereqs/checksums.sha256"
STC_REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$STC_REPO_ROOT/research/sleep-time-compute"
uv run stc runtime run-matrix \
  --manifest manifests/stc/scaling-cells.jsonl \
  --cohort-index manifests/stc/scaling-cohort-index.json \
  --budget manifests/stc/scaling-budget.json \
  --scaling-config configs/stc/design/scaling.yaml \
  --hardware-config configs/stc/design/scaling-hardware.yaml \
  --primary-receipt "$STC_NFS_ROOT/control/scaling-prereqs/distributed-confirmatory-receipt.json" \
  --primary-validation "$STC_NFS_ROOT/control/scaling-prereqs/confirmatory-result-validation.json" \
  --clean-receipt "$STC_NFS_ROOT/control/scaling-prereqs/distributed-confirmatory-clean-rerun-receipt.json" \
  --clean-validation "$STC_NFS_ROOT/control/scaling-prereqs/confirmatory-clean-rerun-result-validation.json" \
  --preregistration manifests/stc/preregistration.json \
  --execution-snapshot manifests/stc/g4-confirmatory-execution.json \
  --gate-chain manifests/gates \
  --mode scaling \
  --run-id scaling \
  --verify-only \
  --receipt manifests/systems/distributed-scaling-receipt.json \
  --output "$STC_NFS_ROOT/staging/scaling"
uv run stc runtime import-matrix \
  --receipt manifests/systems/distributed-scaling-receipt.json \
  --source "$STC_NFS_ROOT/staging/scaling" \
  --output results/scaling
uv run stc results validate \
  --preregistration manifests/stc/preregistration.json \
  --manifest manifests/stc/scaling-cells.jsonl \
  --cohort-index manifests/stc/scaling-cohort-index.json \
  --results results/scaling \
  --output manifests/stc/scaling-result-validation.json
uv run stc dag import-bundles \
  --input results/scaling \
  --manifest-dir manifests/systems/runs
uv run stc dag check
```

If a node fails before the top-level manifest is published, both operators
rerun the same track command with `--resume verify-complete`; no manual shard
copy or merge is allowed. Rank ownership remains the same, verified complete
children are reused, invalid/incomplete attempts remain auditable, and global
rank zero alone publishes the merged index and manifest after the barrier.

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
closed stage names in order, encodes world sizes 1/1/4/8/8 and every
`--require-gate` edge, and is explicitly `SYNTHETIC`. Mutation tests remove a
stage, reorder stages, alter a world size, drop a prerequisite, inject an
unknown CLI option, relabel a fake metric, or change the runbook after receipt
creation; every case fails. This command does not execute the post-G4
confirmatory/scaling matrices.

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
- Create: `research/sleep-time-compute/systems/INFRASTRUCTURE-BLUEPRINT.md`
- Modify: `research/sleep-time-compute/registry/artifacts.jsonl`
- Modify: `research/sleep-time-compute/src/stc_research/cli.py`
- Modify: `research/sleep-time-compute/tests/test_cli.py`

**Interfaces:**
- Consumes: a JUnit test report plus registered lifecycle, DSE,
  deletion-with-fault-outcomes, fake-accelerator, and optional
  measured-accelerator manifests. Selecting `a100-confirmatory` additionally
  requires the four fixed registered prerequisite run IDs and their ordered
  digest chain; a confirmatory manifest alone is inadmissible.
- Produces: immutable `G4SystemsEvidence`, a rendered Markdown view, a frozen
  systems input index, a generic typed gate-input manifest, a typed G4
  decision, and explicit hardware-claim blockers.

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

Expected: all tests pass; `g4-lifecycle`, `g4-dse`, and `g4-deletion` exist in
`manifests/systems/runs/`; `g4-fake` exists in
`manifests/accelerator/runs/` with `measurement_status=SYNTHETIC`; and every
registered manifest resolves to a checksum-valid bundle.

- [ ] **Step 2: Audit claim boundaries**

The G4 report explicitly verifies:

- no simulator output is labeled hardware measurement;
- no fake accelerator output is claim-admissible as measurement;
- storage and energy boundaries remain visible in every table;
- deletion claims distinguish deny, online removal, backup/key expiry, lineage
  invalidation, and behavioral residual;
- all four factorial cells share the cutoff digest;
- no runtime method can access oracle-only records.

`G4SystemsEvidence` is validated against `g4-evidence-v1.schema.json` and
contains the exact `manifests/gates/G3.json` digest, test-report digest and
counts, selected run IDs and manifest/payload digests, artifact-DAG digest,
queue stability/SLO results, lifecycle charge conservation, every fault
outcome, all deletion completion dimensions, tombstone/linkability results,
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
  --system-manifest-dir manifests/systems/runs \
  --system-run-id g4-lifecycle \
  --system-run-id g4-dse \
  --system-run-id g4-deletion \
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
uv run stc dag check
```

Expected: the report command re-resolves every frozen digest and fails on a
missing, stale, overwritten, class-upgraded, unregistered, or G3-mismatched
input. The generic assembler and gate evaluator independently validate the
predecessor and JSON/schema/systems-index/DAG digests. Tests mutate the G3
record, systems evidence, systems input index, and assembled G4 input manifest;
every mutation blocks evaluation with a stable code.
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
  --system-manifest-dir manifests/systems/runs \
  --system-run-id g4-lifecycle \
  --system-run-id g4-dse \
  --system-run-id g4-deletion \
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
Task 9 PASS + combined execution snapshot ─── Task 8 post-G4 matrices
```

Tasks 4 and 7 may run in parallel after Task 1. Task 8 is the only task that
requires the external A100 cluster. Its microbenchmark branch may feed G4; its
297n primary, 297n clean-rerun, and 112n scaling child branches are forbidden
until PASS G4 and the combined execution snapshot exist.

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
- A deletion watermark race is a correctness failure even when the artifact
  content itself validates.
- Cryptographic erasure, physical removal, and behavioral unlearning answer
  different questions.
- H100 projections are separate DSE points; they never inherit A100
  measurements.
- The systems paper extension is allowed only if it owns a distinct runtime or
  scheduler claim, failure-injection study, and accelerator result set.
