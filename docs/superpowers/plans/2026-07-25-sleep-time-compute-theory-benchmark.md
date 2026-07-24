# Sleep-Time Compute Theory and Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the normative theory, leakage-safe lifetime benchmark, matched memory baselines, preregistered 297-cell design, statistical decision rules, and held-out scaling-law tests required by the approved sleep-time compute research design.

**Architecture:** A deterministic generator first creates a hidden oracle world and then emits chronologically available observables into a physically separate stream. Deployable methods receive only past-observable features and compile the same events into raw external, abstract external, latent/KV, user-parametric, or forgotten states under a common resource vector. Theory, experiment design, results, and claims remain separate artifacts connected by digests.

**Tech Stack:** Python 3.14, uv, Pydantic 2, pytest, Hypothesis, NumPy, SciPy `milp`, scikit-learn, statsmodels, PyArrow, Matplotlib, PyYAML, pinned Qwen3-4B, Cartridges, PEFT, and rank_bm25 for execution-only extras.

## Ownership and Scientific Boundaries

- Package root is `research/sleep-time-compute/`; all paths below are relative
  to it unless stated otherwise.
- Command-root contract: every `bash` block starts in a fresh shell whose
  working directory is `research/sleep-time-compute/`; blocks never inherit a
  prior `cd`, shell variable, or activation. Package paths (`configs/`,
  `manifests/`, `results/`, `tests/`) are relative to that directory,
  repository paths use `../../`, and repository Git commands use
  `git -C ../..` with repository-root-relative pathspecs. An executor that
  cannot set the working directory must prepend
  `cd research/sleep-time-compute` to each block.
- `stc_research.models` continues to own evidence/control-plane records.
- `stc_research.benchmark.schemas` owns scientific event, query, run, and
  preregistration records.
- `stc_research.systems.schemas` owns `LifecycleCostSummary`; benchmark code
  imports it and never duplicates lifecycle accounting.
- `stc_research.runtime.contracts.MethodRuntime` is the concrete runtime
  boundary.
- The existing `experiments/E4-scaling/` implementation is historical,
  exploration-grade evidence and is never imported into confirmatory analysis.
- Artifact bytes are an empirical rate proxy, not mutual information.
- Transcript reconstruction is a diagnostic, not the primary future-task
  distortion.
- Public benchmarks supply external validation but are
  `scaling_fit_eligible=false`.
- Any post-G2 change to method, threshold, seed, comparator, scorer, stopping
  rule, or model demotes the affected analysis to exploratory.

## Typed Scientific Contracts

Use frozen Pydantic v2 models and string enums. Export deterministic JSON
Schemas.

```python
class MemoryAction(StrEnum):
    KEEP_RAW = "keep_raw"
    COMPRESS_EXTERNAL = "compress_external"
    COMPILE_LATENT_KV = "compile_latent_kv"
    UPDATE_USER_LORA = "update_user_lora"
    FORGET = "forget"


class EventType(StrEnum):
    EXACT_FACT = "exact_fact"
    DRIFTING_PREFERENCE = "drifting_preference"
    PROCEDURE = "procedure"
    SKILL = "skill"
    RELATION = "relation"
    NEGATIVE_FEEDBACK = "negative_feedback"
    CORRECTION = "correction"
    CONTRADICTION = "contradiction"
    PRIVACY_DELETION = "privacy_deletion"
    OBSERVATION = "observation"


class ProbePhase(StrEnum):
    PRE_SLEEP = "pre_sleep"
    POST_SLEEP = "post_sleep"
    NEXT_WAKE_DELAYED = "next_wake_delayed"


class StreamSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
```

`ObservableEvent` contains:

```text
schema_version, event_id, user_id, source_id, session_id,
cycle_index, event_index, event_type, payload, timestamp_s,
confidence_at_write, privacy_scope, provenance, supersedes
```

All operational IDs are randomly generated opaque UUID handles. They are never
content hashes, names, email addresses, or reusable source identifiers.

`OracleAnnotation`, stored in a separate directory and never imported by a
deployable method, contains:

```text
schema_version, event_id, future_reuse, valid_to_s, later_outcomes,
poisoned, deleted_at_s, ground_truth_dependencies,
hindsight_action_utility
```

`poisoned` and the future role “one-off distractor” exist only on the oracle
side. Adversarial content is emitted as an ordinary observable event; the
generator never reveals a `POISON` or `DISTRACTOR` type to the policy.

`QueryObservable` contains:

```text
schema_version, query_id, user_id, session_id, cycle_index,
timestamp_s, probe_phase, query_type, prompt
```

`QueryTarget`, withheld from methods, contains:

```text
schema_version, query_id, answer, answer_type,
dependency_event_ids, target_version, validity_state,
scorer_kind, abstention_expected
```

`StreamManifest`, also unavailable to method code, maps each opaque user/stream
handle to `workload_family_id`, `stream_split`, grammar/template-family digest,
paired seed family, and generator digest. Analysis and the runner join through
this sidecar. Workload-family and split labels never appear in
`ObservableEvent`, `QueryObservable`, `PolicyFeatureRow`, compiler inputs, or a
deployable cache/config; adversarial leakage tests cover every boundary.

`PolicyFeatureRow` stops at `decision_time_s` and contains event/source type,
age, confidence, privacy scope, past valid-hit count/rate, past contradiction
count, past-only reuse/invalidation estimates and uncertainty, active/durable
occupancy, sleep queue load, and a write-time frozen observable embedding.
Forbidden fields include true reuse, true invalidation, future timestamps,
future labels, answers, dependencies, poison labels, hindsight utility, and any
embedding computed after the cutoff.

The method layer exposes:

```python
class DestinationPolicy(Protocol):
    def choose(
        self,
        features: PolicyFeatureRow,
        feasible_actions: Sequence[MemoryAction],
    ) -> MemoryAction: ...


class MemoryCompiler(Protocol):
    def compile(
        self,
        events: Sequence[ObservableEvent],
        destination: MemoryAction,
        cutoff_s: float,
    ) -> ArtifactProposal: ...
```

## Command Surface

`src/stc_research/cli.py` and `tests/stc/test_cli_contract.py` own these exact
commands; the task that introduces each command wires and tests it before use:

```text
stc schema export
stc theory rate-distortion
stc theory verify-hypotheses
stc theory render
stc benchmark generate
stc benchmark audit
stc benchmark property-check
stc benchmark freeze
stc adapter materialize
stc adapter validate
stc adapter build-replay-manifest
stc upstream fetch
stc methods check
stc pilot run
stc pilot summarize
stc design materialize
stc design power
stc design finalize
stc analyze confirmatory
stc scaling materialize
stc scaling fit
stc prereg freeze
stc prereg validate
stc results validate
stc gate inputs assemble
stc gate evaluate
```

The two `stc gate` commands are consumed from the shared evidence/control-plane
CLI; this plan owns only the typed G2 input-assembly contract and its domain
tests.

## Frozen Confirmatory Constants

```text
deployable methods = 6
oracle methods = 1
R = {1, 4, 16}
phi = {1, 1/4, 1/16}
paired seed families = 3
routing deployable = 6 × 3 × 3 × 3 = 162
routing oracle = 1 × 3 × 3 × 3 = 27
schedule/operator = 2 × 2 × 3 × 3 × 3 = 108
logical total = 297
default horizon = 128 cycles
diagnostic horizons = {8, 32, 128, 512}
retrieval lags = {1, 8, 32, 128}
```

`R=0`, no-memory, and full-history are diagnostics and are excluded from the
297 logical confirmatory runs.

---

### Task 1: Extend the package for scientific schemas and commands

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/stc_research/cli.py`
- Create: `src/stc_research/benchmark/__init__.py`
- Create: `src/stc_research/benchmark/schemas.py`
- Create: `schemas/stc/observable-event-v1.json`
- Create: `schemas/stc/oracle-annotation-v1.json`
- Create: `schemas/stc/query-observable-v1.json`
- Create: `schemas/stc/query-target-v1.json`
- Create: `schemas/stc/stream-manifest-v1.json`
- Create: `schemas/stc/adapter-manifest-v1.json`
- Create: `schemas/stc/run-spec-v1.json`
- Create: `schemas/stc/run-result-v1.json`
- Create: `schemas/stc/logical-cell-v1.json`
- Create: `schemas/stc/confirmatory-summary-v1.json`
- Create: `schemas/stc/scaling-cohort-index-v1.json`
- Create: `schemas/stc/scaling-budget-v1.json`
- Create: `schemas/stc/preregistration-v1.json`
- Create: `tests/stc/test_schemas.py`
- Create: `tests/stc/test_cli_contract.py`

**Interfaces:**
- Produces: the typed contracts above and `stc schema export --check`.
- Requires: Systems-validation Task 1 has already created the canonical
  `LifecycleCostSummary` imported by `RunResult`.

- [ ] **Step 1: Write schema round-trip and boundary tests**

Test deterministic schema export, forbidden extra fields, invalid IDs,
nonfinite metrics, negative resources, timestamp ordering, and physical
separation of observables/oracle records.

- [ ] **Step 2: Confirm failure**

```bash
uv run pytest tests/stc/test_schemas.py -q
```

- [ ] **Step 3: Add the scientific dependency group**

Add frozen-compatible ranges for:

```text
pydantic>=2.11,<3
scikit-learn>=1.7,<2
statsmodels>=0.14,<1
pyarrow>=21,<22
```

Add this execution-only optional extra; the lockfile pins every transitive
version:

```toml
[project.optional-dependencies]
models = [
  "torch>=2.8,<3",
  "transformers>=4.55,<5",
  "safetensors>=0.6,<1",
  "rank-bm25==0.2.2",
  "peft @ git+https://github.com/huggingface/peft.git@051b2c5d9f2a94413418e6a8f65881bb2e31bc71",
  "cartridges @ git+https://github.com/HazyResearch/cartridges.git@ef34ba97a06049c34820506e2c283746284ae5f0",
]
```

It is not installed for evidence/theory-only fixture validation.

- [ ] **Step 4: Implement models and deterministic export**

```bash
uv lock
uv sync --group dev
uv run stc schema export
uv run stc schema export --check
uv run pytest tests/stc/test_schemas.py tests/stc/test_cli_contract.py -q
```

Expected: `schemas_current=true`; a second export changes no file.

- [ ] **Step 5: Commit**

```bash
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: define sleep-time scientific schemas"
```

---

### Task 2: Implement the constrained semi-Markov evaluator

**Files:**
- Create: `src/stc_research/theory/__init__.py`
- Create: `src/stc_research/theory/semimarkov.py`
- Create: `tests/stc/theory/test_semimarkov.py`

**Interfaces:**
- Produces: `evaluate_policy(trajectory, constraints, discount_rate_s)`.
- Consumes: dimensionless task distortion and native-unit lifecycle charges.

- [ ] **Step 1: Write dimensional and feasibility tests**

Verify:

- discounting uses `exp(-rho_s * elapsed_wall_seconds)`, not decision index;
- resource caps are undiscounted native-unit totals;
- privacy, deletion, p99 wake latency, active/durable capacity, queue stability,
  and cross-user interference are hard constraints;
- mixed-unit scalar addition raises an error;
- economic scalarization is disabled without a frozen valuation and every
  resource price.

- [ ] **Step 2: Implement and test**

```bash
uv run pytest tests/stc/theory/test_semimarkov.py -q
```

Expected: elapsed-time and decision-index fixtures differ; infeasible
trajectories remain infeasible regardless of utility.

- [ ] **Step 3: Commit**

```bash
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: implement constrained lifetime objective"
```

---

### Task 3: Implement empirical rate-distortion and H-STC-001

**Files:**
- Create: `src/stc_research/theory/rate_distortion.py`
- Modify: `src/stc_research/cli.py`
- Modify: `tests/stc/test_cli_contract.py`
- Create: `tests/stc/theory/test_rate_distortion.py`
- Create: `tests/stc/fixtures/rate-distortion-supported.jsonl`
- Create: `tests/stc/fixtures/rate-distortion-null.jsonl`

**Interfaces:**
- Produces: nondominated artifact-bytes/held-out-task-loss frontiers and a
  query-distribution reversal test.
- Owns and parser-tests the exact command
  `theory rate-distortion --fixture PATH`.

- [ ] **Step 1: Write frontier and non-claim tests**

Reject claims that serialized bytes estimate \(I(E;Z)\). Keep transcript
reconstruction under `secondary_diagnostics`. Require held-out future-query
loss for the primary distortion.

- [ ] **Step 2: Implement monotone nondominated frontier extraction**

```bash
uv run pytest tests/stc/theory/test_rate_distortion.py \
  tests/stc/test_cli_contract.py -q
uv run stc theory rate-distortion \
  --fixture tests/stc/fixtures/rate-distortion-supported.jsonl
```

Expected: the fixture shows a representation-order reversal when the frozen
future-query distribution changes. The null fixture yields
`H-STC-001=FALSIFIED_OR_NARROWED`, not a rewritten hypothesis.

- [ ] **Step 3: Commit**

```bash
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: test future-task rate distortion"
```

---

### Task 4: Implement H-STC-002 through H-STC-005

**Files:**
- Create: `src/stc_research/theory/promotion.py`
- Create: `src/stc_research/theory/cadence.py`
- Create: `src/stc_research/theory/capacity.py`
- Create: `src/stc_research/theory/hypotheses.py`
- Create: `src/stc_research/theory/render.py`
- Modify: `src/stc_research/cli.py`
- Modify: `tests/stc/test_cli_contract.py`
- Create: `tests/stc/theory/test_hypothesis_laws.py`
- Create: `tests/stc/theory/test_render.py`
- Create: `configs/stc/theory/laws.yaml`

**Interfaces:**
- Produces: discounted reuse, break-even, cadence, promotion/demotion,
  multi-capacity, and reversibility-reserve diagnostics.
- Owns and parser-tests `theory verify-hypotheses CONFIG --output PATH` and
  `theory render --laws PATH --output PATH` before either command is invoked.

- [ ] **Step 1: Test H-STC-002 reuse and promotion conditions**

Numerically integrate:

```text
∫[0,H] lambda_i(t) S_i(t) exp(-omega t) dt
```

Check the stationary Poisson/exponential infinite-horizon special case
`lambda / (mu + omega)`. A destination-region threshold may be emitted only
after verifying monotone single crossing of destination gain/cost curves.

- [ ] **Step 2: Test H-STC-003 hysteresis conditions**

Derive promotion and demotion thresholds for a two-state switching-cost model.
If sufficient conditions fail, return `classification=empirical_policy_surface`
rather than asserting universal hysteresis. Compare with a one-threshold
thrashing baseline.

- [ ] **Step 3: Test cadence laws**

For the general candidate:

```text
C_bar(tau) = C_s/tau + a*tau^p
tau_star = (C_s/(a*p))^(1/(p+1))
```

For the linear delay special case:

```text
J(tau) = F/tau + h*r_e*tau/2
tau_star = sqrt(2F/(h*r_e))
```

Verify units, numerical minima, and counterexamples under burst/drift arrival
processes.

- [ ] **Step 4: Test H-STC-004 and H-STC-005**

Compute the six dimensionless loads:

```text
rho_sleep, rho_store, rho_read, rho_repr, rho_hot, rho_gov
```

Test binding-resource saturation and the byte-flow equilibrium for every
store. An archive with no expiry, deletion, or physical reclamation must be
classified unbounded. Test both non-monotone and monotone archive-depth
fixtures; do not force an optimum when the data do not support one.

- [ ] **Step 5: Run and commit**

```bash
uv run pytest \
  tests/stc/theory/test_hypothesis_laws.py \
  tests/stc/theory/test_render.py \
  tests/stc/test_cli_contract.py -q
uv run stc theory verify-hypotheses \
  configs/stc/theory/laws.yaml \
  --output /tmp/stc-theory-laws-smoke.json
uv run stc theory render \
  --laws /tmp/stc-theory-laws-smoke.json \
  --output /tmp/stc-theory-render-smoke
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: implement sleep-time candidate laws"
```

Expected: all five H-STC IDs receive a machine-readable supported,
falsified/narrowed, or empirical-curve outcome; dimensional checks pass.
The rendered derivation/equation bundle records the law-config, source, and
unit/property-test digests; this task owns and tests the `stc theory render`
command later used by the master plan.

---

### Task 5: Build the deterministic controlled lifetime generator

**Files:**
- Create: `src/stc_research/benchmark/generator.py`
- Create: `src/stc_research/benchmark/state_machine.py`
- Create: `src/stc_research/benchmark/protocol.py`
- Create: `src/stc_research/benchmark/property_check.py`
- Create: `configs/stc/benchmark/controlled-128.yaml`
- Create: `manifests/stc/generator-property-report.json`
- Create: `tests/stc/benchmark/test_generator_properties.py`
- Create: `tests/stc/benchmark/test_state_machine.py`
- Create: `tests/stc/benchmark/test_property_check_cli.py`
- Modify: `src/stc_research/cli.py`
- Modify: `tests/stc/test_cli_contract.py`

**Interfaces:**
- Produces: separate observable, oracle, query-observable, query-target, and
  manifest files.

- [ ] **Step 1: Freeze the random hierarchy and state machine**

Use:

```text
master SeedSequence
  → TRAIN/VALIDATION/TEST split
    → workload grammar/template family
      → paired seed family
        → user
          → cycle
            → event/query component
```

Generate the hidden oracle plan first, then emit observables chronologically.
Freeze disjoint grammar/template families across splits, with exactly sixteen
preregistered primary TEST workload families (any extra diagnostic families
are ineligible for primary inference) and no template or lexical-entity pool
shared with TRAIN/VALIDATION. Create one ex-ante paired event cohort before assigning R,
future queries, deletion, volatility, or poison. Conditional on randomized R,
assign future queries independently and apply the same administrative
censoring rule across cells. Never condition admission on future validity or
horizon position.

- [ ] **Step 2: Generate all controlled event families**

Include exact facts, drifting preferences, procedures, skills, relations,
negative feedback, correction, contradiction, deletion, poison, and one-off
distractors. Independently vary reuse, live-memory pressure, volatility,
paraphrase difficulty, poison rate, event rate, and idle slack.

- [ ] **Step 3: Implement the three frozen transition probes**

Each cycle emits exactly pre-sleep, post-sleep, and delayed-next-wake probes.
The state machine enforces supersession, validity intervals, deletion
precedence, abstention, and dependency closure.

Property tests compare event type, age, payload length, source, volatility,
deletion, poison, and admission marginals across R cells; only assigned future
reuse may differ. They also prove user IDs and workload-family IDs are disjoint
across TRAIN/VALIDATION/TEST.

- [ ] **Step 4: Run generator and properties**

```bash
uv run stc benchmark generate \
  --config configs/stc/benchmark/controlled-128.yaml \
  --output /tmp/stc-controlled-smoke
uv run stc benchmark property-check \
  --config configs/stc/benchmark/controlled-128.yaml \
  --seeds 0:99 \
  --output manifests/stc/generator-property-report.json
uv run pytest \
  tests/stc/benchmark/test_generator_properties.py \
  tests/stc/benchmark/test_state_machine.py \
  tests/stc/benchmark/test_property_check_cli.py -q
```

Expected:

```text
cycles=128
probe_transition_types=3
orphan_query_dependencies=0
future_leakage=0
invalid_supersession=0
deletion_precedence_errors=0
replay_digest_mismatches=0
non_R_marginal_failures=0
split_identity_or_template_overlap=0
property_seed_count=100
```

`stc benchmark property-check` is the sole production property-sweep command.
It writes a typed report containing the config/code digests, exact inclusive
seed range, per-property failure IDs, and aggregate counts. The CLI test proves
that `0:99` means exactly 100 seeds, a partial/duplicate seed set fails, and a
nonzero property count exits nonzero. This task owns the command used by the
master plan.

- [ ] **Step 5: Commit**

```bash
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: add controlled lifetime stream generator"
```

---

### Task 6: Implement leakage audit, scorer, and exact oracle

**Files:**
- Create: `src/stc_research/benchmark/leakage.py`
- Create: `src/stc_research/benchmark/scoring.py`
- Create: `src/stc_research/benchmark/oracle_candidates.py`
- Create: `src/stc_research/benchmark/oracle.py`
- Create: `src/stc_research/benchmark/freeze.py`
- Create: `tests/stc/benchmark/test_leakage.py`
- Create: `tests/stc/benchmark/test_scoring.py`
- Create: `tests/stc/benchmark/test_oracle_candidates.py`
- Create: `tests/stc/benchmark/test_benchmark_freeze.py`
- Create: `tests/stc/methods/test_oracle.py`
- Modify: `src/stc_research/cli.py`
- Modify: `tests/stc/test_cli_contract.py`

**Interfaces:**
- Produces: deterministic exact/temporal scores and a constrained
  multiple-choice oracle over five actions.

- [ ] **Step 1: Write adversarial leakage fixtures**

Attempt to leak future reuse, valid-to time, answer, dependency, poison label,
future embedding, or hindsight utility through payload, metadata, feature
builder, cache, and config. Every attempt must fail closed.

- [ ] **Step 2: Implement exact and temporal scoring**

Score at query time with target version, supersession, deletion, abstention,
poison, and dependency closure. Semantic judges remain secondary and cannot
override the exact/temporal endpoint.

- [ ] **Step 3: Implement the constrained oracle**

Use `scipy.optimize.milp` for the same five actions and the same storage,
compute, latency, privacy, and deletion caps. The oracle may use the sidecar
only through an explicitly nondeployable interface and cannot violate caps.
For PAPER-C1, each oracle cell also emits counterfactual optima for four
medium-restricted clairvoyant policies—raw external, compressed external,
latent/KV, and user-LoRA, each plus deliberate forgetting—under the identical
sidecar and caps. These are counterfactual result blocks inside the same 27
oracle runs, not extra logical cells.

Freeze the finite candidate table and time-expanded mixed-integer program
rather than treating “oracle” as an informal best run. Before solving, run the
actual deterministic compilers on every preregistered source/session cluster,
action, eligible transition cycle, cutoff, and finite retrieval-bundle
candidate. Store and digest each artifact's bytes, lineage, availability,
resource vector, deletion behavior, and the actual deterministic scorer's
`temporal_exact` coefficient for every query/retrieval bundle. Lossy
compressed/KV/LoRA artifacts therefore receive their measured candidate score,
not perfect dependency credit. Restricted and unrestricted oracles must consume
the identical candidate-table digest.

Binary `x[group, action, cycle]` variables select create/update/migrate/forget
transitions, `y[artifact, cycle]` variables encode active time-indexed
availability after deadlines and before eviction/deletion, `z[group, action,
cycle]` variables activate shared compiler/index overheads, and
`r[query, retrieval_bundle]` variables select a pre-scored feasible bundle.
State-balance constraints link every transition to the previous cycle and
permit retain-then-evict, migrate, and forget trajectories. A retrieval bundle
may be selected only when all of its named artifacts are active; query score is
the candidate table's integer `temporal_exact` coefficient. The
integer-scaled objective is the exact equal-cycle/equal-probe score defined in
Task 10.

Constraints encode per-cycle active/durable bytes, live-token/application
budget, sleep compute and queue deadlines, privacy scope, deletion/reclamation,
artifact availability, and the medium restriction. Shared-overhead, state
transition, and retrieval-bundle ANDs use explicit binary linearizations with
recorded finite bounds; the solver may not discover new clusters/bundles or use
an unrecorded big-M.

Pin SciPy/HiGHS versions and all options, including presolve, deterministic
seed/thread count, `mip_rel_gap=0`, feasibility tolerances, and a 3,600-second
limit. Apply a frozen lexicographic tie-break after maximizing task score:
minimize lifecycle resources and then the canonical action vector. Only a
solver status proving global optimality is an oracle result. A limit,
numerical warning, or nonzero gap produces `ORACLE_UNRESOLVED`, makes the
corresponding C1 contrast ineligible, and can never be relabeled an upper
bound. Exhaustively enumerate small fixtures and require exact agreement in
objective, feasibility, and tie-broken assignment; add fixtures with shared
overhead, lossy compilation, retain-then-evict, migration, deletion, deadline,
and medium-restriction interactions.

- [ ] **Step 4: Verify**

```bash
uv run stc benchmark audit /tmp/stc-controlled-smoke
uv run pytest \
  tests/stc/benchmark/test_leakage.py \
  tests/stc/benchmark/test_scoring.py \
  tests/stc/benchmark/test_oracle_candidates.py \
  tests/stc/benchmark/test_benchmark_freeze.py \
  tests/stc/methods/test_oracle.py -q
```

Expected:

```text
oracle_columns_in_policy=0
future_timestamp_violations=0
invalid_dependency_edges=0
oracle_resource_violations=0
```

Implement `stc benchmark freeze` here as a content-addressed G3 packager. Its
production form requires `--controlled`, `--property-report`, `--public-replay`,
`--preregistration`, and `--output`; it verifies every supplied digest is a
member of the frozen G2 envelope, copies no restricted data, and writes a typed
inventory of schemas, generator, scorer, oracle candidate table, adapter
transformations, licenses, and validation reports. The command fails on a
missing/stale G2 edge or an unresolved oracle/property failure. The freeze test
uses signed fixture digests, mutates each input in turn, and proves a second
freeze is byte-identical.

The exact deferred production invocation, run only after G2 and the controlled
corpus exist, is:

```bash
uv run stc benchmark freeze \
  --controlled generated/stc/controlled \
  --property-report manifests/stc/generator-property-report.json \
  --public-replay manifests/stc/public-replay.jsonl \
  --preregistration manifests/stc/preregistration.json \
  --output generated/stc/benchmark
```

`test_benchmark_freeze.py` and `test_cli_contract.py` invoke this same option
set against signed fixture paths; omitting or renaming any producer argument is
a failing CLI-contract test.

- [ ] **Step 5: Commit**

```bash
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: add leakage audit scorer and oracle"
```

---

### Task 7: Implement deterministic public benchmark adapters

**Files:**
- Create: `src/stc_research/benchmark/adapters/__init__.py`
- Create: `src/stc_research/benchmark/adapters/base.py`
- Create: `src/stc_research/benchmark/adapters/longmemeval.py`
- Create: `src/stc_research/benchmark/adapters/memoryagentbench.py`
- Create: `src/stc_research/benchmark/adapters/halumem.py`
- Create: `configs/stc/upstreams.lock.yaml`
- Create: `configs/stc/public/longmemeval.yaml`
- Create: `configs/stc/public/memoryagentbench.yaml`
- Create: `configs/stc/public/halumem.yaml`
- Create: `manifests/stc/public-replay.jsonl`
- Modify: `src/stc_research/cli.py`
- Modify: `tests/stc/test_cli_contract.py`
- Create: `tests/stc/benchmark/test_public_adapters.py`
- Create: `tests/stc/fixtures/longmemeval-record.json`
- Create: `tests/stc/fixtures/memoryagentbench-record.json`
- Create: `tests/stc/fixtures/halumem-user.json`

**Interfaces:**
- Produces: transformation manifests with source revision, source SHA,
  transformation SHA, license, row/order rule, boundary rule, withholding rule,
  and dropped-ID ledger.
- Owns and parser-tests the exact commands
  `adapter materialize --config-dir PATH --output PATH`,
  `adapter validate --config-dir PATH`, and
  `adapter build-replay-manifest --inputs PATH --output PATH`.

- [ ] **Step 1: Pin and verify upstream identities**

Freeze repository/data revisions only after fetching the authoritative
upstreams and verifying them. The starting candidates from the design audit are
recorded as candidates, not trusted until resolved by the implementation task.

- [ ] **Step 2: Implement LongMemEval**

Preserve `haystack_session_ids` order; map one original session to one wake
session; sleep after each session; release the question only after all history.
Move `answer`, `answer_session_ids`, and `has_answer` to the oracle sidecar.

- [ ] **Step 3: Implement MemoryAgentBench**

Preserve row/context order; reproduce the pinned 4,096-token sentence/token
chunker; sleep after every upstream chunk; withhold questions, answers, and
gold metadata until ingestion completes.

- [ ] **Step 4: Implement HaluMem Medium**

Preserve user/session timestamps; expose dialogue only; sleep after each source
session; keep memory points, answers, and evidence oracle-only. Because of
CC BY-NC-ND constraints, publish adapter code, revisions, and transformation
digests but do not redistribute transformed data without license review.

- [ ] **Step 5: Validate and commit**

```bash
uv run pytest tests/stc/benchmark/test_public_adapters.py \
  tests/stc/test_cli_contract.py -q
uv run stc adapter materialize \
  --config-dir configs/stc/public \
  --output results/public-adapters
uv run stc adapter validate --config-dir configs/stc/public
uv run stc adapter build-replay-manifest \
  --inputs results/public-adapters \
  --output manifests/stc/public-replay.jsonl
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: add versioned memory benchmark adapters"
```

Expected: source/output counts match each declared transformation; all public
tracks report `scaling_fit_eligible=false`; the replay manifest includes exact
upstream/transformation/license/order/boundary/withholding digests. Downloaded
or transformed data remain outside Git when redistribution is not permitted.

---

### Task 8: Implement the four single-medium compilers

**Files:**
- Create: `src/stc_research/methods/__init__.py`
- Create: `src/stc_research/methods/base.py`
- Create: `src/stc_research/methods/raw_external.py`
- Create: `src/stc_research/methods/compressed_external.py`
- Create: `src/stc_research/methods/latent_kv.py`
- Create: `src/stc_research/methods/user_lora.py`
- Create: `configs/stc/methods/raw-external.yaml`
- Create: `configs/stc/methods/compressed-external.yaml`
- Create: `configs/stc/methods/latent-kv.yaml`
- Create: `configs/stc/methods/user-lora.yaml`
- Create: `configs/stc/models.lock.yaml`
- Create: `manifests/stc/model-upstreams.json`
- Modify: `src/stc_research/cli.py`
- Modify: `tests/stc/test_cli_contract.py`
- Create: `tests/stc/methods/test_method_contract.py`
- Create: `tests/stc/methods/test_model_upstreams.py`

**Interfaces:**
- Every compiler implements `MethodRuntime`, preserves lineage and cutoff, and
  emits measured serialized bytes and lifecycle charges.
- Owns and parser-tests
  `upstream fetch --lock PATH --cache PATH --output PATH` and
  `methods check --config-dir PATH`.

- [ ] **Step 1: Write shared method-contract tests**

Require:

```text
artifact lineage
information cutoff
base/tokenizer/compiler/config digests
active/durable/index/lineage byte accounting
compile/verify/application compute
privacy scope and deletion path
deterministic artifact digest
validation-tuned admission/forget action
```

- [ ] **Step 2: Implement raw and compressed external methods**

Use the same deterministic BM25 retrieval backend and live-token accounting.
Compression retains source-to-summary lineage and confidence/verifier results.
Neither method may access future query data during compilation.

- [ ] **Step 3: Implement the frozen latent/KV method**

Pin a real Cartridges-compatible local checkpoint path and use its existing
`TrainConfig`, `KVFromText.Config`, `HFModelConfig`,
`FlexQwen3ForCausalLM`, `TrainDataset.Config`, `DataSource`,
`TrainableCache.save/from_pretrained`, and `flex_generate` interfaces. Do not
invent an undocumented loader or depend on mutable W&B state.

- [ ] **Step 4: Implement user-local LoRA**

Use pinned PEFT `LoraConfig`/`get_peft_model`, opaque user-isolated adapter IDs,
exact serialized adapter and optimizer/checkpoint bytes, and explicit base
revision compatibility. Cross-user batching cannot merge user state.

- [ ] **Step 5: Materialize and verify model upstreams**

`models.lock.yaml` freezes every repository commit, base-model and tokenizer
revision, exact downloaded-file SHA-256, expected size, license identifier and
license-text digest, and the Cartridges/PEFT compatibility tuple. Fetch into
the repo-local ignored cache `.cache/stc/models/`; reject mutable branch names,
floating Hub revisions, size-only checks, and files not named by the lock.
Generate `model-upstreams.json` from verified bytes.

```bash
uv sync --extra models --group dev
uv run stc upstream fetch \
  --lock configs/stc/models.lock.yaml \
  --cache .cache/stc/models \
  --output manifests/stc/model-upstreams.json
uv run pytest tests/stc/methods/test_model_upstreams.py \
  tests/stc/test_cli_contract.py -q
```

Run the same fetch and a one-example save/load/generate capability test from a
clean cache. G2 later freezes both the lock and generated-manifest digests;
changing a model, tokenizer, license, or upstream commit stales the gate.
`test_model_upstreams.py` mutates each lock field and each resolved manifest
digest independently, proves `stc upstream fetch` rejects the mismatched edge
in both directions, and verifies regeneration is byte-identical only for
unchanged verified bytes. Task 13's preregistration mutation test repeats these
edges at the frozen-envelope boundary.

- [ ] **Step 6: Verify**

```bash
uv run stc methods check --config-dir configs/stc/methods
uv run pytest \
  tests/stc/methods/test_method_contract.py \
  tests/stc/methods/test_model_upstreams.py \
  tests/stc/test_cli_contract.py -q
```

Expected: all four methods pass identical contracts. A high-pressure cell may
forget/admit selectively; it must not be forced to store until failure.

- [ ] **Step 7: Commit**

```bash
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: add matched single-medium compilers"
```

---

### Task 9: Implement static mixture, deployable hybrid, and oracle routing

**Files:**
- Create: `src/stc_research/methods/static_mixture.py`
- Create: `src/stc_research/methods/hybrid.py`
- Create: `src/stc_research/methods/oracle.py`
- Create: `configs/stc/methods/static-mixture.yaml`
- Create: `configs/stc/methods/hybrid.yaml`
- Create: `configs/stc/methods/oracle.yaml`
- Create: `tests/stc/methods/test_static_mixture.py`
- Create: `tests/stc/methods/test_hybrid.py`
- Modify: `tests/stc/methods/test_oracle.py`

**Interfaces:**
- Static mixture: validation-frozen content-free allocation.
- Hybrid: cost-sensitive past-only destination router trained by oracle
  imitation on training users.
- Oracle: nondeployable upper bound.

- [ ] **Step 1: Implement the static mixture**

Freeze action proportions and cadence per resource/workload cell on validation
users. Hash the generator-assigned opaque event UUID with SHA-256 or HMAC and a
published manifest key to assign the frozen allocation bucket; the method
never creates or replaces operational IDs. Python `hash()` is forbidden.
Changing payload text while holding the existing UUID constant must not change
allocation.

- [ ] **Step 2: Implement the hybrid**

Train a cost-sensitive multinomial policy with resource shadow prices and
uncertainty from past-only features. Charge feature construction, controller
training, tuning, and inference. Training users/workload families, validation
users/workload families, and test users/workload families are all disjoint.
Hyperparameters, feature selection, thresholds, and cadence never consume a
TEST family.

- [ ] **Step 3: Enforce oracle isolation**

The oracle consumes only the oracle-side interface. Import-time, type-level,
and runtime tests reject oracle access from every deployable method.

- [ ] **Step 4: Verify and commit**

```bash
uv run pytest \
  tests/stc/methods/test_static_mixture.py \
  tests/stc/methods/test_hybrid.py \
  tests/stc/methods/test_oracle.py -q
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: add static hybrid and oracle routers"
```

Expected: `deployable_methods=6 oracle_methods=1`.

---

### Task 10: Materialize the 297 logical confirmatory runs

**Files:**
- Create: `src/stc_research/analysis/__init__.py`
- Create: `src/stc_research/analysis/estimands.py`
- Create: `src/stc_research/design/__init__.py`
- Create: `src/stc_research/design/confirmatory.py`
- Create: `src/stc_research/methods/identity_external.py`
- Create: `src/stc_research/methods/consolidating_external.py`
- Create: `configs/stc/design/analysis.yaml`
- Create: `configs/stc/design/confirmatory.yaml`
- Create: `manifests/stc/deployment-stakes.json`
- Create: `manifests/stc/analysis-plan-skeleton.json`
- Create: `manifests/stc/confirmatory-skeleton.jsonl`
- Modify: `src/stc_research/cli.py`
- Modify: `tests/stc/test_cli_contract.py`
- Create: `tests/stc/analysis/test_estimands.py`
- Create: `tests/stc/design/test_confirmatory.py`
- Create: `tests/stc/methods/test_factorial_operators.py`

**Interfaces:**
- Consumes: method IDs, seed hierarchy, controlled generator, resource caps,
  and the blinded-pilot rule for `B_ref`.
- Produces: one immutable pre-power row per logical cell. Task 11 fills the
  powered user count and freezes the final manifest; no run may start from the
  skeleton. It also freezes the estimand functions that both power and final
  inference must import.
- Owns and parser-tests the complete `design materialize` option set shown in
  Step 5, including all three output paths.

- [ ] **Step 1: Write exact enumeration tests**

Assert:

```text
routing_deployable=162
routing_oracle=27
schedule_operator=108
logical_total=297
duplicate_logical_cells=0
```

Also assert that R=0, no-memory, full-history, additional latent
implementations, graph memory, and vendor systems are absent.

- [ ] **Step 2: Implement the two factorial operators**

The identity operator verifies and republishes a byte-equivalent versioned
external record. The consolidating operator applies one frozen compressor to
canonicalize and merge the identical declared source set. Both implement
`MethodRuntime`, receive the same enqueue-time snapshot, destination, cap, and
deadline, and differ only in transformation semantics. Write tests for byte
equivalence, source-set equality, cutoff equality, and validation-before-
publication.

- [ ] **Step 3: Implement the manifest compiler**

The 297-row skeleton stores `LogicalCellSpec`: logical-cell ID, the frozen TEST
workload-family-set digest, method, R, phi, paired seed family, schedule,
operator, cap rule, and all generator/adapter/model/method/config digests. It
does not multiply the scientific cell count by workload family or user.
Task 11's powered count and frozen seed table deterministically expand each
logical cell into child `RunSpec` records. Each `RunSpec` contains run and
parent-logical-cell IDs, singular `workload_family_id`, `stream_split`, user
and paired-seed IDs, method, destination, R, phi, schedule, operator,
resource-cap fields, `enqueue_snapshot_digest`, `information_cutoff_s`,
`deadline_s`, `source_set_digest`, `compressor_digest`, and every upstream
digest. Physical-run deduplication may reference an identical
full-configuration hash but cannot remove a logical row or contrast.

Within every `(workload_family_id, R, phi, paired_seed_family)` factorial
block, the four schedule×operator rows must have byte-identical source-set,
enqueue-snapshot, destination, information-cutoff, deadline, compressor,
generator, model, and cap digests. Only `schedule`, `operator`, and their
derived runtime action may differ. Compilation rejects a partial four-row
block or any equality-invariant violation.

The pre-power skeleton has `powered_user_count: null` and
`active_byte_cap: null` with `execution_status: BLOCKED_PRE_POWER`. Only Task 11
may produce the final `confirmatory.jsonl` with a positive count, a frozen
absolute per-user active-byte cap for every phi, and
`execution_status: FROZEN`.

- [ ] **Step 4: Freeze estimands, missingness, and resampling before power**

Implement the following functions once in `analysis/estimands.py`; Task 11
imports them for power and Task 12 imports the same symbols for inference.
Duplicated formulas are a gate failure.

For query \(q\), define

```text
temporal_exact(q) =
  1{canonical answer is exact AND target version is valid at query time
    AND deletion, supersession, poison, dependency-closure, and required
    abstention semantics are all satisfied}
```

For user stream \(u\), cell \(c\), cycle \(t=1,\ldots,128\), and each of the
three frozen probes \(p\), let \(Q_{uctp}\) be the scheduled primary queries.
The normalized equal-cycle/equal-probe lifetime AUC is exactly

```text
AUC(u,c) = (1 / (3 * 128)) *
           sum_p sum_t [(1 / |Q(u,c,t,p)|) *
                        sum_q_in_Q temporal_exact(q)]
```

Every primary user/cell/cycle/probe must contain at least one query. A zero
denominator, absent probe, malformed target, checksum mismatch, or protocol
violation invalidates the complete paired block; it is never imputed as zero.
Late-horizon events with no scheduled query are administratively censored and
do not enter the denominator. Their admitted-cohort retention is a named
secondary endpoint.

Freeze operational failure handling: at most two attempts may execute the
same immutable run spec; the first checksum-valid complete attempt is accepted,
all attempts remain logged, and an outcome cannot trigger a retry. The G2
manifest contains an ordered reserve-seed list used only for pre-score
infrastructure failures. Replacement occurs for the whole paired
method/regime block before outcomes are unblinded. A protocol-valid method
timeout, compiler exception, deadline miss, cap violation, or invalid artifact
is a method outcome, not missing data: charge the work, force a failure answer
at affected probes, and score zero except where the frozen target itself
requires abstention. Only shared infrastructure corruption that prevents
scoring any method in a paired block is missing.

For unresolved infrastructure missingness, report completion by method/cell/
family, the cell-balanced method-minus-comparator completion-risk difference,
its adjusted 95% interval, and a frozen 0.01 absolute differential-completion
tolerance. Primary point estimates use complete paired blocks, but every
support decision must also survive worst-case MNAR bounds that assign missing
directional outcomes adversarially in `[0,1]` for the proposed method and its
comparator. Support is prohibited if any family completes below 95%, the
absolute differential completion exceeds 0.01, its adjusted interval excludes
zero, or the worst-case bound loses support. A cap-respecting method may choose
`FORGET` and abstain, which remains a scored feasible outcome. A structurally
infeasible required cell is reported as infeasible, never scored as zero, and
prevents support for a contrast that needs it.

Primary uncertainty uses 20,000 deterministic hierarchical paired bootstrap
replicates. Resample the 16 TEST workload families with replacement, then
resample paired user streams within each selected family; retain every method,
R, phi, schedule, operator, cycle, and probe for a selected stream. Studentize
each contrast with a workload-family-clustered CR2 sandwich standard error and
Satterthwaite small-sample degrees of freedom, and use the
replicate-wise maximum absolute t statistic across the single ordered global
family `PAPER-C-GLOBAL-75` defined below. The 95% and 90% critical values are
the corresponding empirical quantiles of that same replicate-wise maximum;
claim-specific critical values are forbidden. Report adjusted 95% intervals,
adjusted 90% intervals for TOST, all unadjusted intervals, and
leave-one-paired-seed-family-out sensitivity. Prevalidate CR2 coverage in the
power simulation and require a 99,999-draw Webb six-point wild-cluster bootstrap
sensitivity at the workload-family level; a claim cannot be supported when the
primary and wild-cluster decisions or seed-family sensitivity disagree.
Generalization is scoped to the frozen workload-family generator population,
not arbitrary real-world tasks.

Freeze these exact primary contrast functions in
`analysis-plan-skeleton.json`:

- C1: nine unrestricted-clairvoyant minus replicate-wise
  medium-restricted-oracle-envelope contrasts.
- C2: a 1/9 equal-cell mean over the nine R×phi cells. In every bootstrap and
  within each cell, first form the four-single-medium utility envelope, compare
  that envelope with static mixture, and carry the cost of the same cellwise
  selected comparator; only then take the equal-cell mean. Benefit support requires
  point \(\Delta U\ge0.02\) and adjusted 95% LCB \(>0\). Efficiency support
  requires utility LCB \(>-0.01\), point lifecycle-cost reduction
  \(1-\bar C_h/\bar C_b\ge0.10\), and its adjusted LCB \(>0\). Independently
  freeze full-oracle-minus-hybrid regret per cell and its 1/9 equal-cell mean.
  Optional oracle-gap capture fraction
  \((U_h-U_b)/(U_o-U_b)\) is admissible only when the paired denominator is
  positive and at least 0.01; otherwise report `CAPTURE_FRACTION_INADMISSIBLE`.
- C3: for each of the two adjacent pairs
  `(external-envelope, latent/KV)` and `(latent/KV, user-LoRA)`, freeze one
  low-promotion and one high-promotion corner from H-STC-002 plus
  validation-only feasibility, before pilot condition labels or TEST outcomes.
  The deployment-stakes record fixes \(\delta_{cross}=0.01\). Let
  \(D_j^L\) and \(D_j^H\) be the first-minus-second utility difference at the
  two corners and \(I_j=D_j^L-D_j^H\). The primary sign-reversal test is the
  intersection-union pair \(D_j^L>\delta_{cross}\) and
  \(D_j^H<-\delta_{cross}\); its four oriented component margins are members of
  `PAPER-C-GLOBAL-75` and use the same global critical values as every other
  primary margin. Report both \(I_j\). No separate ordinal interaction may
  rescue this test.
- C4: enumerate, in every one of nine R×phi regimes, two operator-specific
  schedule simple effects, two schedule-specific operator simple effects, and
  the schedule×operator interaction, plus the schedule-marginal operator main
  effect: 54 reported contrasts total. Those rows plus the five aggregate
  boundary rows listed below are the 59 C4 members of
  `PAPER-C-GLOBAL-75`; no C4-local critical value exists.

Freeze this exact ordered multiplicity table:

```text
PAPER-C-GLOBAL-75
  C1: 9 cellwise unrestricted-minus-restricted-envelope margins
  C2: 3 margins
      benefit utility against 0
      utility noninferiority against -0.01
      lifecycle-cost reduction against 0
  C3: 4 oriented crossover component margins
  C4: 59 margins
      54 cellwise schedule/operator/main/interaction contrasts
      2 cell-balanced identity-schedule TOST boundaries
      2 cell-balanced interaction TOST boundaries
      1 cell-balanced lifecycle-cost-reduction boundary against 0.10
```

The ordered contrast IDs, orientations, null boundaries, studentizer, and
`critical_95`/`critical_90` bootstrap quantiles are serialized in
`analysis-plan-skeleton.json`. Point-materiality checks, oracle-regret/capture
diagnostics, \(I_j\), native resource vectors, and hypervolume are reported but
cannot enter or alter `PAPER-C-GLOBAL-75`. Missingness completion-risk
contrasts live in a separately frozen fail-only validity family: they may veto
support but can never create it. Task 11 power and Task 12 inference import the
same ordered table and critical-value function; tests permute, omit, and add a
contrast and require an implementation-digest failure.

Power is evaluated at deployment-stake alternatives fixed before the blinded
pilot, not estimated effects: C2 benefit \(\Delta U=0.03\); C2 efficiency
\(\Delta U=0\) with 0.15 lifecycle-cost reduction; each C3 powered pair has
\(D^L=\delta_{cross}+0.01\) and
\(D^H=-(\delta_{cross}+0.01)\); C4 has one frozen operator simple effect of
0.03, identity schedule and marginal interaction exactly zero, and 0.15
identity deferral cost reduction. These one-point utility and five-point cost
buffers are justified in a pre-data deployment-stakes record and receive
this exact nonbinding sensitivity grid:

```text
utility gain or crossover buffer: {0.005, 0.010, 0.015}
cost-reduction buffer above 0.10:  {0.025, 0.050, 0.100}
equivalence true effect:           {-0.005, 0, +0.005}
infrastructure missing rate:       {0, 0.025, 0.05}
```

The selection-binding effects are the central alternatives stated above,
equivalence effect zero, and—for each contrast—the hierarchical covariance
model with every variance component at its blinded-pilot upper 95% bound and
each frozen correlation endpoint chosen to maximize that contrast's variance,
followed by a recorded nearest-PSD projection. The binding missing rate is
`max(0.05, blinded pilot upper 95% bound)`. Other grid points are reported
diagnostics and cannot increase or decrease selected n. These choices do not
change the support thresholds above; changing them after pilot labels or TEST
results is a new G2 design.

The skeleton also freezes the full C1–C4 status/reason table, contrast-family
membership, cost denominators, feasibility rules, bootstrap seed/count,
studentization, multiplicity, and the prohibition on TEST-family tuning.

- [ ] **Step 5: Verify and commit**

```bash
uv run stc design materialize \
  --config configs/stc/design/confirmatory.yaml \
  --analysis-config configs/stc/design/analysis.yaml \
  --output manifests/stc/confirmatory-skeleton.jsonl \
  --analysis-output manifests/stc/analysis-plan-skeleton.json \
  --deployment-stakes-output manifests/stc/deployment-stakes.json
uv run pytest \
  tests/stc/analysis/test_estimands.py \
  tests/stc/design/test_confirmatory.py \
  tests/stc/methods/test_factorial_operators.py \
  tests/stc/test_cli_contract.py -q
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: freeze 297-cell confirmatory design"
```

---

### Task 11: Implement blinded paired power simulation

**Files:**
- Create: `src/stc_research/design/power.py`
- Create: `configs/stc/design/power.yaml`
- Create: `configs/stc/design/cost-profile.yaml`
- Create: `manifests/stc/blinded-pilot-summary.json`
- Create: `manifests/stc/power.json`
- Create: `manifests/stc/confirmatory.jsonl`
- Create: `manifests/stc/confirmatory-run-index.json`
- Create: `manifests/stc/confirmatory-summary.json`
- Create: `manifests/stc/analysis-plan.json`
- Modify: `src/stc_research/cli.py`
- Modify: `tests/stc/test_cli_contract.py`
- Create: `tests/stc/design/test_power.py`
- Create: `tests/stc/fixtures/pilot-summary.json`

**Interfaces:**
- Selects one user-stream count before G2 from a blinded pilot.
- Freezes the exact lifecycle-cost functional used by PAPER-C2/C4.
- Imports every utility, contrast, feasibility, and bootstrap function from
  `analysis/estimands.py`; simulation-only reimplementations are forbidden.
- Imports the ordered `PAPER-C-GLOBAL-75` membership table, orientations, null
  boundaries, studentizer, and shared critical-value function from
  `analysis-plan-skeleton.json`; a local or claim-specific multiplicity table
  is forbidden.
- Owns and parser-tests `pilot run`, `pilot summarize`, `design power`, and
  `design finalize` with the exact options shown below.

- [ ] **Step 1: Write power-selection tests**

Simulate candidate user counts 16 through 256 in steps of 16, with 10,000 paired
Monte Carlo experiments per candidate and the exact Task 10 estimands and
hierarchical analysis. A chosen count means user streams per logical row; paired
seed family is already one of the 297 logical factors and is not multiplied a
second time. Because every candidate is divisible by sixteen, allocate it
equally across the 16 frozen TEST workload families. The same user stream is
paired across every method/regime contrast for its seed family. Select the
maximum count needed by:

- PAPER-C2 two-point benefit or one-point noninferiority plus 10% cost branch;
- PAPER-C3 two-corner material sign reversal for either adjacent-medium pair;
- PAPER-C4 operator effect, identity deferral TOST/cost, and marginal
  interaction TOST.

For each candidate, compute a binomial 95% Monte Carlo lower confidence bound
for every required selection-binding branch under the exact Task 10
covariance/missingness rule. Select the smallest
candidate whose every lower bound strictly exceeds 0.90. If none through 256
passes, write `power_status=FAIL_MAX_N`, leave both confirmatory and analysis
plans unfinalized, and fail G2; increasing the range is a new preregistration,
not an implicit fallback.

`cost-profile.yaml` defines `lifecycle_cost_usd` as a complete, non-overlapping
sum of native lifecycle resources times frozen deployment prices. It names the
price source/date/currency, amortization boundary, GPU/CPU time, active and
durable byte-seconds, index/lineage/latent/checkpoint byte-seconds, disk I/O,
and network transfer. Energy is reported separately when already included in
server prices, preventing double counting. Privacy, deletion, latency, and
capacity remain hard constraints and cannot be purchased away. If any consumed
resource lacks a price, the economic cost branch is inadmissible rather than
silently dropping it.

For PAPER-C2, percent cost reduction is recomputed in every paired bootstrap as
`1 - mean(C_hybrid) / mean(C_stronger_baseline)`. For PAPER-C4 identity,
it is `1 - cell_balanced_mean(C_deferred) /
cell_balanced_mean(C_inline)`. Denominators must be positive. The price profile,
functional, cell weights, and denominators are frozen before G2 and reused
unchanged in power and inference; the full native resource vector is always
reported beside the USD contrast.

- [ ] **Step 2: Implement deterministic simulation**

Record pilot digest, covariance model, seed, simulation count, candidate grid,
contrast-specific power and uncertainty, sensitivity grid, chosen count, and
no-optional-stopping rule. The power engine imports `auc_128`,
`paper_c2_contrasts`, `paper_c3_sign_reversal`, `paper_c4_contrasts`, and the
hierarchical resampling configuration from `analysis/estimands.py`, and imports
the exact ordered `PAPER-C-GLOBAL-75` table and critical-value function from
`analysis-plan-skeleton.json`. Tests mutate each shared estimand function,
permute/add/drop one global-family member, and change each 90%/95% critical
quantile rule; both power and inference must reject the changed implementation
digest.

The test fixture verifies code paths only. Before production power selection,
run the separately budgeted pilot from training users and export a blinded
summary that contains pooled variance/covariance and failure-rate information
but no condition labels or held-out test outcomes. It also computes `B_ref` as
the preregistered 95th percentile across pilot/training users of peak serialized
bytes for the raw full-valid-history store over 128 cycles. The summary records
the quantile estimator, user inclusion rule, serialized-byte measurement
version, source run digests, estimate, and uncertainty. No held-out user
contributes.

```bash
uv run stc pilot run \
  --config configs/stc/design/power.yaml \
  --output results/pilot
uv run stc pilot summarize \
  --input results/pilot \
  --output manifests/stc/blinded-pilot-summary.json \
  --blind-condition-labels
```

- [ ] **Step 3: Verify and commit**

```bash
uv run pytest tests/stc/design/test_power.py \
  tests/stc/test_cli_contract.py -q
uv run stc design power \
  --config configs/stc/design/power.yaml \
  --cost-profile configs/stc/design/cost-profile.yaml \
  --analysis manifests/stc/analysis-plan-skeleton.json \
  --deployment-stakes manifests/stc/deployment-stakes.json \
  --pilot manifests/stc/blinded-pilot-summary.json \
  --output manifests/stc/power.json
uv run stc design finalize \
  --skeleton manifests/stc/confirmatory-skeleton.jsonl \
  --analysis-skeleton manifests/stc/analysis-plan-skeleton.json \
  --power manifests/stc/power.json \
  --output manifests/stc/confirmatory.jsonl \
  --run-index-output manifests/stc/confirmatory-run-index.json \
  --summary-output manifests/stc/confirmatory-summary.json \
  --analysis-output manifests/stc/analysis-plan.json
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: implement blinded confirmatory power design"
```

The chosen count \(n\) is intentionally unknown until this task runs; no
arbitrary number is inserted in the plan. Finalization refuses any nonpassing
power manifest and must report:

```text
logical_total = 297
logical_child_references = 297 * n
child_references_per_logical = n
child_references_per_logical_per_family = n / 16
duplicate_child_keys = 0
missing_child_keys = 0
extra_child_keys = 0
blocked_pre_power = 0
missing_user_count = 0
missing_B_ref = 0
heldout_cap_recomputation = 0
```

Every held-out run reuses the exact cap `phi * B_ref` from the frozen pilot
manifest. The compact run index freezes the disjoint workload-family allocation,
ordered primary/reserve seeds, child-`RunSpec` expansion algorithm, and a
logical-child-reference→physical-execution map. Several logical child references
may share one physical result only when their complete configuration hash is
identical; `unique_physical_executions <= 297 * n`, every parent reference
remains explicit, and shared results retain one covariance identity rather than
becoming pseudo-replicates. The runtime logs every expanded child spec and
verifies the index digest. A logical cell is complete only when all \(n\) child
references resolve to checksum-valid results or preregistered failure outcomes;
297 placeholder/logical-level records can never satisfy the summary.

---

### Task 12: Implement primary statistics and claim decisions

**Files:**
- Consume read-only: `src/stc_research/analysis/estimands.py`
- Create: `src/stc_research/analysis/statistics.py`
- Create: `src/stc_research/benchmark/result_validation.py`
- Create: `tests/stc/analysis/test_statistics.py`
- Create: `tests/stc/benchmark/test_result_validation.py`
- Create: `tests/stc/fixtures/supported-results.jsonl`
- Create: `tests/stc/fixtures/narrowed-results.jsonl`
- Create: `tests/stc/fixtures/falsified-results.jsonl`
- Create: `tests/stc/fixtures/infeasible-results.jsonl`
- Create: `tests/stc/fixtures/result-validation/{preregistration.json,manifest.jsonl,cohort-index.json,results/}`
- Create: `manifests/stc/confirmatory-result-validation.json`
- Modify: `src/stc_research/cli.py`
- Modify: `tests/stc/test_cli_contract.py`

**Interfaces:**
- Produces: result blocks and machine decisions for PAPER-C1–C4.
- Produces: `stc results validate`, the fail-closed validator used before every
  smoke, public, confirmatory, or scaling analysis.
- Consumes: the finalized G2 analysis plan and the exact Task 10 estimand
  functions; it cannot infer or retune a rule from results.
- Owns and parser-tests the exact production contracts
  `results validate --preregistration PATH --manifest PATH
  [--cohort-index PATH] --results PATH --output PATH` and
  `analyze confirmatory --manifest PATH --results PATH --validation-report PATH
  --output PATH`. The analyzer cannot open the result root until the supplied
  report is schema-valid, `PASS`, and digest-matched.

- [ ] **Step 1: Write paired-resampling tests**

Recompute the Task 10 equal-cycle/equal-probe AUC, hierarchical
workload-family→paired-user bootstrap, cluster studentization, 20,000
replicates, the one ordered `PAPER-C-GLOBAL-75` max-|t| family, missingness
rules, and seed-family sensitivity by importing—not copying—
`analysis/estimands.py` and the finalized analysis plan. Recompute
baseline envelopes and ratio denominators inside every replicate. Tests reject
a different function digest, cycle-level resampling, TEST-family tuning,
unpaired method rows, empty probes, silent infeasibility, or a locally
reconstructed membership/critical-value table.

`test_result_validation.py` covers all four runtime modes. In production mode
the command requires `--preregistration`, the frozen logical manifest, its
cohort/run index, and the result root. It verifies envelope membership, exact
logical-child coverage, logical→physical deduplication, attempt/retry status,
checksums, lifecycle charges, information cutoffs, and upstream digests before
writing a typed validation report. A logical-only 297-row fixture, a duplicated
physical result presented as independent, an unindexed child, and a stale G2
digest all fail.

- [ ] **Step 2: Implement PAPER-C1**

Within each of the nine preregistered reuse-pressure cells, compare the
unrestricted clairvoyant oracle against the envelope of the four
medium-restricted clairvoyant counterfactuals generated with the same future
sidecar, caps, and forgetting option. Recompute the restricted-oracle envelope
inside each paired whole-user bootstrap replicate. Read the nine C1 adjusted
intervals from the shared maximum across all 75 `PAPER-C-GLOBAL-75` members;
the nine rows never receive a claim-specific maximum. This isolates strict
cross-medium heterogeneity from the separate value of future-aware
admission/forgetting.

`PAPER-C1=SUPPORTED` only if at least one adjusted 95% lower bound is strictly
above zero. If point estimates are positive but no adjusted interval excludes
zero, return `NARROWED_INSUFFICIENT_PRECISION`; if no cell has positive
heterogeneity value, return `FALSIFIED`. Report all nine estimates and never
use oracle weak dominance as support. Any unresolved/nonoptimal oracle cell
returns `NARROWED_ORACLE_UNRESOLVED`.

- [ ] **Step 3: Implement PAPER-C2**

Use the 1/9 equal-cell R×phi estimand. Within every bootstrap replicate and
every R×phi cell, first select the stronger of the four single-medium methods
to form that cell's single-medium envelope; then select the stronger of that
envelope and static mixture. Carry the paired lifecycle cost of the same
cellwise selected comparator and only then average the nine cell contrasts.
A single global baseline winner is forbidden because medium rankings may
cross by resource tuple. The benefit, noninferiority, and cost boundaries are
the three C2 members of `PAPER-C-GLOBAL-75` and use its global critical values.
Support only if either:

1. point hybrid utility gain is at least 0.02 and the adjusted 95% utility LCB
   is above zero; or
2. the adjusted 95% utility LCB is above -0.01, point lifecycle-cost reduction
   is at least 0.10, and its adjusted 95% LCB is above zero.

The alternatives are disjunctive. Secondary hypervolume cannot overturn a
failed primary decision. Freeze this exhaustive priority-ordered reason table:

```text
SUPPORTED_BENEFIT
SUPPORTED_EFFICIENCY
FALSIFIED_UTILITY_HARM
  := benefit UCB < 0.02 and utility-noninferiority UCB < -0.01
FALSIFIED_NO_EFFICIENT_BRANCH
  := benefit UCB < 0.02 and cost-reduction UCB < 0.10
NARROWED_POINT_GAIN_BELOW_0_02
NARROWED_COST_POINT_BELOW_0_10
NARROWED_INSUFFICIENT_PRECISION
NARROWED_SEED_SENSITIVITY
NARROWED_MISSING_OR_INFEASIBLE
```

Each machine reason carries `decision_class=FALSIFIED|NARROWED` but maps to one
of the two canonical evidence statuses `SUPPORTED` or
`FALSIFIED/NARROWED`; the evidence registry never invents separate release
statuses for the two non-support classes.

Every C2 result block, including a null/narrowed branch, also reports cellwise
and 1/9 equal-cell `oracle_regret = U_unrestricted_oracle - U_hybrid` with its
paired simultaneous interval. If the optional capture fraction is reported,
recompute the oracle, baseline envelope, and denominator admissibility inside
every bootstrap replicate. Regret and capture diagnose how much past-only
routing leaves on the table; neither changes the fixed C2 support predicate.

- [ ] **Step 4: Implement PAPER-C3**

Define the external comparator as the replicate-wise stronger of raw and
compressed external memory. Before G2, freeze one low-promotion and one
high-promotion corner from H-STC-002 and a materiality margin
`delta_crossover=0.01` from the pre-data deployment tolerance. Pilot variance
affects power only, not the materiality margin. TEST users and TEST workload
families play no role in corner or margin selection.

For adjacent pair \(j\) in `(external-envelope, latent/KV)` and
`(latent/KV, user-LoRA)`, compute \(D_j^L\), \(D_j^H\), and
\(I_j=D_j^L-D_j^H\) exactly as frozen in Task 10. `SUPPORTED` requires, for at
least one pair, simultaneous adjusted LCBs above zero for both
\(D_j^L-\delta_{cross}\) and \(-D_j^H-\delta_{cross}\), plus stable direction
under leave-one-seed-family-out sensitivity. The four component margins are
named members of `PAPER-C-GLOBAL-75` and use its global 75-member critical
values; \(I_j\) is reported but no categorical or ordinal interaction is an
extra hurdle or rescue.

If both adjacent pairs each have at least one required component whose adjusted
UCB is below zero, return `FALSIFIED_PREDICTED_REVERSAL`. Otherwise return
`NARROWED_INSUFFICIENT_PRECISION`, `NARROWED_SIGN_CHANGE_BELOW_MARGIN`, or
`NARROWED_SEED_SENSITIVITY` as applicable. A pair with any structurally
infeasible corner is ineligible for support and reported with its feasibility
mask; if both are ineligible, return `NARROWED_INFEASIBLE`, never a fabricated
low utility.

- [ ] **Step 5: Implement PAPER-C4 exactly**

For every regime \(r\) in the nine R×phi cells, enumerate:

```text
S_identity(r)       = deferred - inline
S_consolidating(r)  = deferred - inline
O_inline(r)         = consolidating - identity
O_deferred(r)       = consolidating - identity
O_main(r)           = 0.5 * (O_inline(r) + O_deferred(r))
I(r)                = S_consolidating(r) - S_identity(r)
```

These are 54 mandatory reported main/simple effects/interactions whose rows,
plus five aggregate-boundary rows, are the 59 C4 members of
`PAPER-C-GLOBAL-75`; all use the global 75-member critical values. The positive
branch requires all three:

1. at least one of the nine `O_main(r)` or 18 `O_schedule(r)` effects has
   point gain at least 0.02 and adjusted 95% LCB above zero;
2. the 1/9 cell-balanced `S_identity` adjusted 90% interval lies wholly in
   `[-0.01,+0.01]`, and the adjusted 95% lifecycle-cost-reduction LCB is
   strictly above 0.10;
3. the 1/9 cell-balanced interaction adjusted 90% interval lies wholly in
   `[-0.01,+0.01]`.

Always publish cellwise interactions and intervals. An interval overlapping
the equivalence margin and null is `NARROWED_INSUFFICIENT_PRECISION`, never
evidence of equivalence. A significant interaction cannot support
separability. Freeze the exhaustive priority-ordered non-support reasons:

```text
FALSIFIED_NO_OPERATOR_EFFECT
  := no adjusted 95% LCB is above zero among all O_main(r) and O_schedule(r)
NARROWED_GAIN_BELOW_0_02
NARROWED_IDENTITY_SCHEDULE_OUTSIDE_OR_UNPROVEN_EQUIVALENCE
NARROWED_NO_DEFERRED_COST_BENEFIT
NARROWED_SCHEDULE_CONTINGENT_OR_INTERACTION_INCONCLUSIVE
NARROWED_EQUIVALENCE_IMPRECISE
NARROWED_SEED_SENSITIVITY
NARROWED_MISSING_OR_INFEASIBLE
```

Only `FALSIFIED_NO_OPERATOR_EFFECT` maps to
`decision_class=FALSIFIED`. Every other missed support condition maps to
`decision_class=NARROWED`, including a confidently nonequivalent schedule
effect, cost LCB at or below 0.10, or nonseparable interaction. The recorded
decision includes every satisfied/failed predicate even though the canonical
reason uses this deterministic priority.

- [ ] **Step 6: Verify all decision branches**

```bash
uv run stc results validate \
  --preregistration tests/stc/fixtures/result-validation/preregistration.json \
  --manifest tests/stc/fixtures/result-validation/manifest.jsonl \
  --cohort-index tests/stc/fixtures/result-validation/cohort-index.json \
  --results tests/stc/fixtures/result-validation/results \
  --output /tmp/stc-confirmatory-validation.json
uv run stc analyze confirmatory \
  --manifest tests/stc/fixtures/result-validation/manifest.jsonl \
  --results tests/stc/fixtures/result-validation/results \
  --validation-report /tmp/stc-confirmatory-validation.json \
  --output /tmp/stc-confirmatory-analysis
uv run pytest \
  tests/stc/analysis/test_statistics.py \
  tests/stc/benchmark/test_result_validation.py \
  tests/stc/test_cli_contract.py -q
```

Expected: supported, narrowed, and falsified fixtures exercise every branch;
the infeasible fixture proves that no zero substitution occurs. Outputs contain
adjusted intervals, cellwise effects, failed-run accounting, canonical
`SUPPORTED|FALSIFIED/NARROWED` mappings, and exact result-block IDs.

After production confirmatory execution, the exact fail-closed invocation is:

```bash
uv run stc results validate \
  --preregistration manifests/stc/preregistration.json \
  --manifest manifests/stc/confirmatory.jsonl \
  --cohort-index manifests/stc/confirmatory-run-index.json \
  --results results/confirmatory-primary \
  --output manifests/stc/confirmatory-result-validation.json
```

`stc analyze confirmatory` must consume that typed validation report before it
reads outcomes; a missing/nonpassing report, 297 logical-only rows without all
\(297n\) children, or a covariance/dedup mismatch blocks analysis. This task
does not authorize production execution: the evidence-owned
`stc experiments snapshot-g4` controller must first bind a PASS G4 execution
snapshot at `manifests/stc/g4-confirmatory-execution.json`, and production
confirmatory commands may run only through that snapshot.

- [ ] **Step 7: Commit**

```bash
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: implement preregistered claim statistics"
```

---

### Task 13: Implement dense scaling and model competition

**Files:**
- Create: `src/stc_research/analysis/scaling.py`
- Create: `src/stc_research/preregistration.py`
- Create: `tests/stc/analysis/test_scaling.py`
- Create: `tests/stc/analysis/test_scaling_materialize.py`
- Create: `tests/stc/test_preregistration.py`
- Create: `configs/stc/design/scaling.yaml`
- Create: `configs/stc/design/scaling-hardware.yaml`
- Create: `manifests/stc/theory-laws-G2.json`
- Create: `manifests/stc/scaling-cells.jsonl`
- Create: `manifests/stc/scaling-cohort-index.json`
- Create: `manifests/stc/scaling-budget.json`
- Create: `manifests/stc/scaling-summary.json`
- Create: `manifests/stc/scaling-result-validation.json`
- Create: `manifests/stc/preregistration.json`
- Create: `manifests/gate-inputs/G2.json`
- Create: `manifests/gates/G2.json`
- Create: `tests/stc/test_g2_gate.py`
- Create: `tests/stc/fixtures/scaling-invalid.jsonl`
- Create: `tests/stc/fixtures/scaling-confirmatory-three-point.jsonl`
- Create: `tests/stc/fixtures/scaling-a100-fit.jsonl`
- Create: `tests/stc/fixtures/scaling-a100-holdout.jsonl`
- Modify: `src/stc_research/cli.py`
- Modify: `tests/stc/test_cli_contract.py`

**Interfaces:**
- Produces: `scaling_law` only when every validity condition passes; otherwise
  `empirical_curve`; an incomplete pre-hardware fit is explicitly
  `empirical_curve_provisional`.
- Produces: a 112-logical-cell scaling manifest, exact
  logical→physical/cohort index, prospective budget plus append-only actual
  budget-ledger contract, and a
  typed G2 `GateEvaluationInput`/`GateRecord` pair through the shared gate
  engine; this task never hand-writes a gate record.
- Produces: `stc prereg freeze` and `stc prereg validate`; the former writes one
  canonical content-addressed envelope over every named input below, and the
  latter re-resolves every path/digest edge before G2 or result execution.

- [ ] **Step 1: Freeze the dense scale axes**

For static mixture and hybrid only, use:

```text
{1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8}
```

Fit on the first eight scales and reserve the largest two as extrapolation
holdouts. The configuration declares which physical quantity the normalized
factor multiplies.

Freeze the core one-axis-at-a-time resources in the configuration before G2:
active-byte capacity, live-token application budget, total sleep FLOPs, and
latent/adapter representation capacity. Materialize:

```text
one-axis cells   = 4 axes × 10 factors × 2 methods = 80
joint-axis cells = 16 frozen D-optimal log-resource tuples × 2 methods = 32
candidate total  = 112
```

Choose the 16 joint tuples deterministically from the Cartesian product of the
**first eight fit factors only**. Use D-optimality on the 11-column
intercept+four-main-effect+six-pairwise-interaction log-resource matrix; freeze
the candidate lattice, selection seed, tie-break, selected-row order, rank,
condition number, and matrix digest. Factors 4 and 8 are prohibited from every
joint training tuple and from every inner/outer fitting fold. They occur only in
the named one-axis extrapolation rows and their outcomes can never choose a
model, threshold, interaction term, or hyperparameter.

Keep all 112 logical rows, but do not turn duplicate resource tuples into
pseudo-replicates. The materializer computes a
`physical_configuration_id = SHA256(method, complete resource tuple, hardware,
paired_user_cohort_id, generator/model/runtime digests)`, where the paired-user
cohort identity is assigned before any logical-axis label and therefore does
not contain the logical parent ID. Thus the four one-axis factor-1 rows per
method—and any joint tuple that is byte-identical to an existing
configuration—reference one physical execution for the same paired cohort.
The cohort index stores the logical→physical incidence matrix; fitting
collapses identical physical results to one response and propagates their
perfect covariance while retaining every logical axis label and holdout
obligation.

Import the powered confirmatory count \(n\) and the same 16 TEST workload
families from `confirmatory-summary.json`. Each logical scaling row has exactly
\(n\) A100 child references, \(n/16\) per family. Freeze
`NVIDIA-A100-SXM4-80GB` as the sole required `hardware_id`, including exact
driver, CUDA, clock/power, container, serving-stack, model, and tokenizer
fingerprints. Assign the 64 first-eight-factor one-axis rows plus all 32 joint
rows to `a100_fit`; assign the 16 factor-4/factor-8 one-axis rows to
`a100_holdout`. The holdout rows are untouched configurations, not an extra
hardware population, and cannot fit a coefficient, select a competitor, tune
a threshold, or choose a hyperparameter:

```text
a100_fit_logical_rows = 96
a100_holdout_logical_rows = 16
a100_fit_logical_child_references = 96 * n
a100_holdout_logical_child_references = 16 * n
total_logical_child_references = 112 * n
child_references_per_logical_per_family = n / 16
unique_physical_executions <= 112 * n
```

`scaling-cohort-index.json` freezes every child key, workload family, panel seed,
hardware role, logical parent, physical configuration, deduplication edge, and
execution order. Missing/duplicate/extra child keys, a fit/holdout role swap,
an environment-fingerprint mismatch, or a holdout result used in fitting is a
gate failure.

H100 is strictly an optional future `external_transport_validation` track.
It is recorded in a separate extension manifest/result block, never merged
with A100 observations, and is excluded from the 112 logical cells, child
coverage, G2/G8 inputs, 135-RRE budget, model selection, and the minimum
`scaling_law` predicate. Its absence cannot narrow an A100-supported result,
and its presence cannot promote an unsupported A100 result.

Cell count is not compute budget. For every **unique physical cohort**, use the
blinded pilot cost model's upper 95% projected accelerator-seconds, including
all fresh-snapshot load panels and the conservative frozen hardware multiplier,
and divide by the similarly upper-bounded accelerator-seconds of one powered
128-cycle confirmatory logical row. `scaling-budget.json` records each projected
weight, covariance/source digest, and the total. G2 passes only if the
prospective upper-bound total is at most 135.

The runtime also enforces an actual cumulative
135-reference-run-equivalent ceiling in the frozen cohort order. Before a
launch it requires `actual_rre_so_far + frozen_hard_quota(next) <= 135`; a
resource supervisor terminates the cohort at its non-borrowable quota or at
the remaining global quota, whichever comes first, so measured cumulative RRE
can never exceed 135 even when a duration model underpredicts. It records
measured accelerator-seconds and actual RRE after every cohort. A quota
termination, missing required A100 holdout, or required child left unexecuted
yields `BUDGET_EXHAUSTED_OR_INCOMPLETE` and forces `empirical_curve`; no
outcome-informed reordering/substitution, quota borrowing, or confirmatory
surplus is allowed.
The append-only runtime record is
`results/scaling/actual-rre-ledger.jsonl`; every row binds the physical cohort,
before/after cumulative RRE, hard quota, measured resource counters, and result
digest. `stc results validate` recomputes it from raw lifecycle records and
rejects a missing/reordered row or any instantaneous/cumulative overrun.

- [ ] **Step 2: Estimate a capacity knee at each resource tuple**

Do not infer a knee from a monotonically increasing lifetime trajectory.
Instead, every 128-cycle scaling child contains eight independent 16-cycle
fresh-snapshot panels with

```text
N_assoc = {8, 16, 32, 64, 128, 256, 512, 1024}.
```

At the start of each panel, restore the same canonical empty-memory/base-model
snapshot, ingest exactly the assigned number of generator-controlled
associations, and issue the same preregistered age/lag/probe mix. Association
sets are disjoint across panels. A seed-derived balanced Latin square assigns
panel order across paired users, so load is orthogonal to wall-clock order,
cycle phase, event mix, and query age. Snapshot restore is orchestration rather
than a free method action; every compiler/retrieval/serving resource consumed
inside all eight panels is charged and included in RRE. Tests reject state
carry-over, monotone panel order, unequal probe schedules, or a load bin with
fewer than the frozen user/family replicates.

Estimate \(K_{eff}(f,r,m)\) separately for workload family \(f\), resource tuple
\(r\), and method \(m\) using the preregistered binomial logistic retention
curve. The knee is the association load where predicted exact/temporal
retention crosses the frozen deployment threshold. Each family-level fit
requires all eight bins, observations on both sides of the threshold,
nonseparation, and an identifiable finite knee.

Leave-workload-family-out evaluation refits the **entire** first and second
stages inside each outer fold. No statistic containing the held-out family's
queries, knee, bootstrap draws, threshold tuning, or failure mask may enter its
training fold. After predictions are locked, estimate the held-out family's
knee solely as an evaluation target. Inner folds tune only on the remaining
families. The hierarchical workload-family→paired-user bootstrap repeats the
panel-level first-stage fits and second-stage surface fit in every replicate;
pooled all-family knees and point-estimate plug-in fits are forbidden.

The three R values or three phi values in the confirmatory manifest are never
a scaling dataset. The fitter must reject
`scaling-confirmatory-three-point.jsonl` with
`classification=INVALID_SCALING_INPUT`.

- [ ] **Step 3: Implement preregistered resource-surface competitors**

Fit the following competitors to \(K_{eff}\) with its propagated uncertainty:

```text
constant/null
normalized Cobb-Douglas
min-bottleneck
generalized mean
interaction
segmented load-at-one
```

Fit a separate resource surface for static mixture and hybrid; a pooled surface
without preregistered method×resource interactions is forbidden. Use nested
grouped cross-validation, the leak-free leave-workload-family-out procedure
above, prediction intervals, and A100 configuration holdouts. Group splits are
based on `workload_family_id` and may never mix TRAIN/VALIDATION/TEST templates.
The 64 first-eight-factor one-axis rows identify marginal shapes; the 32
joint-axis rows identify and test interaction structure. The 16 factor-4/8
A100 rows are never used in model selection or coefficient fitting: they
evaluate frozen A100-surface predictions and the exact preregistered
sign/regime-reversal predicate only after the provisional fit is sealed.

- [ ] **Step 4: Implement the scaling-law gate**

A relation may be labeled `scaling_law` only if:

- each fitted axis has at least eight fit points;
- every included family×resource×method tuple has a valid first-stage
  \(K_{eff}\);
- joint-axis rank and condition-number checks support the claimed interaction
  terms;
- factors 4 and 8 are both predicted out of sample on every axis and method;
- all factor-4/factor-8 A100 holdout outcomes are absent from every
  training/tuning digest;
- nested model competition beats the frozen tolerance;
- exponent signs and model selection are stable;
- interval coverage passes;
- all required logical-child references resolve, actual RRE is at most 135, and
  the frozen logical→physical covariance map is honored;
- leave-family-out and heldout-A100 configuration validation do not reverse the claimed
  regime;
- resources are normalized to explicit reference units.

Otherwise return `empirical_curve`, even if an in-sample power law has high
\(R^2\). A malformed/three-point input returns `invalid_scaling_input`, which
is distinct from an honest empirical curve. A fit without all required A100
fit and holdout results or the frozen A100 environment fingerprints is always
`empirical_curve_provisional`.

The final A100 hardware procedure is fixed as follows. Refit the resource
surface from raw results for the 96 A100 fit rows only and require its training
input digest, selected competitor, hyperparameters, coefficients, and exponent
signs to reproduce the sealed provisional fit. Then evaluate the 16 untouched
A100 factor-4/8 rows without refitting or calibration. For each method, the
hardware gate requires family-level normalized-log-\(K_{eff}\) RMSE at most
0.10, at least 90% coverage by the preregistered 90% prediction intervals, and
no holdout family×axis point whose simultaneous interval lies wholly in the
opposite predicted bottleneck regime. These predicates, including handling of
an invalid family-level knee, are serialized before G2. Optional H100 transport
measurements use a separate post hoc transport predicate/result block and do
not enter this procedure.

- [ ] **Step 5: Materialize the scaling track without running it**

```bash
uv run stc scaling materialize \
  --config configs/stc/design/scaling.yaml \
  --hardware-config configs/stc/design/scaling-hardware.yaml \
  --confirmatory-summary manifests/stc/confirmatory-summary.json \
  --pilot manifests/stc/blinded-pilot-summary.json \
  --cost-profile configs/stc/design/cost-profile.yaml \
  --output manifests/stc/scaling-cells.jsonl \
  --cohort-index-output manifests/stc/scaling-cohort-index.json \
  --budget-output manifests/stc/scaling-budget.json
```

Expected:

```text
logical_total = 112
one_axis_logical_total = 80
joint_axis_logical_total = 32
d_opt_model_columns = 11
d_opt_selected_rows = 16
d_opt_rank = 11
joint_factor_4_or_8_rows = 0
a100_fit_logical_rows = 96
a100_holdout_logical_rows = 16
a100_fit_logical_child_references = 96 * n
a100_holdout_logical_child_references = 16 * n
total_logical_child_references = 112 * n
child_references_per_logical_per_family = n / 16
duplicate_child_keys = 0
missing_child_keys = 0
extra_child_keys = 0
unique_physical_executions <= 112 * n
prospective_upper_95_rre <= 135
hard_actual_rre_ceiling = 135
```

The three outputs are one atomic materialization. The cells file records all
112 logical identities; the cohort index records all \(112n\) required A100
child references, the logical→physical incidence/covariance map, and the
96-fit/16-holdout role split. The budget records every unique physical cohort's
blinded upper-bound weight and the hard actual ceiling.
`test_scaling_materialize.py` independently
reconstructs the fit-only Cartesian candidate lattice and D-opt matrix,
verifies the exact child equations and deduplication map, and rejects a
materializer that emits only 112 logical placeholders or inserts an H100 child.
No output may execute until its digest is in the G2 preregistration below.

- [ ] **Step 6: Verify and freeze G2/G3 inputs**

```bash
uv run pytest \
  tests/stc/analysis/test_scaling.py \
  tests/stc/analysis/test_scaling_materialize.py \
  tests/stc/test_preregistration.py \
  tests/stc/test_g2_gate.py \
  tests/stc/test_cli_contract.py -q
uv run stc scaling fit \
  --stage provisional \
  --config configs/stc/design/scaling.yaml \
  --hardware-config configs/stc/design/scaling-hardware.yaml \
  --cohort-index manifests/stc/scaling-cohort-index.json \
  --budget manifests/stc/scaling-budget.json \
  --fit-results tests/stc/fixtures/scaling-a100-fit.jsonl \
  --output /tmp/stc-scaling-provisional.json
uv run stc scaling fit \
  --stage provisional \
  --config configs/stc/design/scaling.yaml \
  --hardware-config configs/stc/design/scaling-hardware.yaml \
  --cohort-index manifests/stc/scaling-cohort-index.json \
  --budget manifests/stc/scaling-budget.json \
  --fit-results tests/stc/fixtures/scaling-invalid.jsonl \
  --output /tmp/stc-scaling-invalid.json
uv run stc scaling fit \
  --stage final \
  --config configs/stc/design/scaling.yaml \
  --hardware-config configs/stc/design/scaling-hardware.yaml \
  --cohort-index manifests/stc/scaling-cohort-index.json \
  --budget manifests/stc/scaling-budget.json \
  --fit-results tests/stc/fixtures/scaling-a100-fit.jsonl \
  --holdout-results tests/stc/fixtures/scaling-a100-holdout.jsonl \
  --output /tmp/stc-scaling-final.json
uv run stc theory verify-hypotheses \
  configs/stc/theory/laws.yaml \
  --scaling-config configs/stc/design/scaling.yaml \
  --output manifests/stc/theory-laws-G2.json
uv run stc prereg freeze \
  --design manifests/stc/confirmatory.jsonl \
  --run-index manifests/stc/confirmatory-run-index.json \
  --power manifests/stc/power.json \
  --analysis manifests/stc/analysis-plan.json \
  --deployment-stakes manifests/stc/deployment-stakes.json \
  --confirmatory-summary manifests/stc/confirmatory-summary.json \
  --scaling manifests/stc/scaling-cells.jsonl \
  --scaling-config configs/stc/design/scaling.yaml \
  --scaling-hardware-config configs/stc/design/scaling-hardware.yaml \
  --scaling-cohort-index manifests/stc/scaling-cohort-index.json \
  --scaling-budget manifests/stc/scaling-budget.json \
  --cost-profile configs/stc/design/cost-profile.yaml \
  --generator-property-report manifests/stc/generator-property-report.json \
  --hypothesis-registry manifests/hypotheses-G2.json \
  --theory-laws manifests/stc/theory-laws-G2.json \
  --runtime manifests/systems/runtime-G2.json \
  --upstreams configs/stc/upstreams.lock.yaml \
  --model-lock configs/stc/models.lock.yaml \
  --model-upstreams manifests/stc/model-upstreams.json \
  --public-replay manifests/stc/public-replay.jsonl \
  --output manifests/stc/preregistration.json
uv run stc prereg validate \
  --manifest manifests/stc/preregistration.json \
  --fail-on-upstream-digest-change
uv run stc gate inputs assemble G2 \
  --root . \
  --predecessor manifests/gates/G1.json \
  --preregistration manifests/stc/preregistration.json \
  --hypothesis-registry manifests/hypotheses-G2.json \
  --theory-laws manifests/stc/theory-laws-G2.json \
  --runtime manifests/systems/runtime-G2.json \
  --confirmatory-summary manifests/stc/confirmatory-summary.json \
  --analysis-plan manifests/stc/analysis-plan.json \
  --model-lock configs/stc/models.lock.yaml \
  --model-upstreams manifests/stc/model-upstreams.json \
  --upstreams configs/stc/upstreams.lock.yaml \
  --public-replay manifests/stc/public-replay.jsonl \
  --generator-property-report manifests/stc/generator-property-report.json \
  --scaling-config configs/stc/design/scaling.yaml \
  --scaling-hardware-config configs/stc/design/scaling-hardware.yaml \
  --scaling-manifest manifests/stc/scaling-cells.jsonl \
  --scaling-cohort-index manifests/stc/scaling-cohort-index.json \
  --scaling-budget manifests/stc/scaling-budget.json \
  --output manifests/gate-inputs/G2.json
uv run stc gate evaluate G2 \
  --inputs manifests/gate-inputs/G2.json \
  --output manifests/gates/G2.json
```

Expected: the provisional supported fixture is
`empirical_curve_provisional`, the malformed fixture is
`invalid_scaling_input`, and only the final fixture can be `scaling_law`.
Every fit reports outer-family-fold RMSE/MAE/coverage, exponents and signs,
model-selection stability, untouched A100 holdout metrics, child
coverage, covariance-map compliance, projected RRE, and actual RRE. A
`--stage final` call without the A100 holdout input, with a nonreproducible
fit-stage digest, or with an A100 holdout row in training fails closed.

The G2 input assembler validates the 297-logical/\(297n\)-child confirmatory
summary, the 112-logical/\(112n\)-child scaling index, fit-only D-opt digest,
global multiplicity table, prospective RRE bound, and hard actual ceiling. It
then freezes all confirmatory/scaling cells, hypothesis equations, runtime
semantics, public/model upstreams, cost prices, retry reserves, and analysis
code digests before any production run. The generic gate engine emits the
canonical typed `GateRecord` with `gate_id=G2`, predecessor/input digests,
named check results, `gate_status=PASS|FAIL`, frozen evaluation timestamp, and
record digest; a domain command may not hand-write it.

`test_g2_gate.py` mutates each named input path and digest one at a time,
including both sides of the model-lock→model-upstreams and
upstreams-lock→public-replay edges, every preregistration constituent,
hypothesis registry, executable theory laws, confirmatory summary, scaling
config/hardware/cells/cohort/budget, runtime, generator property report, and
G1 predecessor. Every mutation stales the preregistration or returns a typed
G2 `FAIL`; no stale artifact may produce `PASS`.

`test_preregistration.py` separately mutates the bytes, recorded digest, and
path of every `stc prereg freeze` argument, adds an unregistered input, removes
one registered input, and permutes invocation argument order. Validation must
fail for every semantic mutation; argument-order permutations and two freezes
over byte-identical inputs must be byte-identical and retain the canonical
ordered artifact table.

After all required production A100 fit and holdout cohorts have been measured,
and without waiting for optional H100 work, the systems
execution plan must invoke this exact final-refit hook before G5:

```bash
uv run stc results validate \
  --preregistration manifests/stc/preregistration.json \
  --manifest manifests/stc/scaling-cells.jsonl \
  --cohort-index manifests/stc/scaling-cohort-index.json \
  --results results/scaling \
  --output manifests/stc/scaling-result-validation.json
uv run stc scaling fit \
  --stage final \
  --config configs/stc/design/scaling.yaml \
  --hardware-config configs/stc/design/scaling-hardware.yaml \
  --cohort-index manifests/stc/scaling-cohort-index.json \
  --budget manifests/stc/scaling-budget.json \
  --preregistration manifests/stc/preregistration.json \
  --validation-report manifests/stc/scaling-result-validation.json \
  --fit-results results/scaling/a100-fit \
  --holdout-results results/scaling/a100-holdout \
  --output manifests/stc/scaling-summary.json
```

Missing measured A100 holdout validation, incomplete required child coverage,
or actual RRE above 135 forces `empirical_curve`; it cannot be waived into a
hardware scaling law. Optional H100 external-transport results, if later
collected, are validated and published under a separate manifest/result block
and never modify `scaling-summary.json`.

- [ ] **Step 7: Commit**

```bash
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: add held-out sleep-time scaling analysis"
```

## Dependency Graph

```text
Task 1 schemas
├── Task 2 semi-Markov
├── Task 3 rate-distortion
├── Task 4 hypothesis laws
├── Task 5 generator
└── Task 7 public adapters

Task 5 ─── Task 6 scorer/oracle
Task 1 + Task 6 ─── Task 8 single-medium methods
Task 4 + Task 8 ─── Task 9 routers
Task 5 + Task 8 + Task 9 ─── Task 10 confirmatory manifest
Task 5 + Task 10 ─── Task 11 power
Task 6 + Task 10 + Task 11 ─── Task 12 statistics
Task 4 + Task 10 + Task 12 ─── Task 13 scaling/preregistration
```

Tasks 2–4 and Task 7 can run in parallel after Task 1.

## Major Validity Traps

- Never leak future query structure, true reuse, `has_answer`, gold evidence,
  poison labels, later embeddings, or oracle utility into deployable routing.
- Generate R, volatility, pressure, poison, deletion, and paraphrase
  independently; otherwise PAPER-C3 is confounded.
- Correct late-horizon censoring by balanced admitted cohorts and report
  censored distractors separately.
- Compute `B_ref` only from blinded pilot/training streams and reuse the same
  absolute cap on held-out users.
- Give single-medium baselines fair admission and forgetting policies.
- Oracle weak dominance is not evidence that the deployable hybrid captures
  oracle value.
- Preserve all 297 logical cells even when physical execution is deduplicated.
- Match enqueue-time information for inline and deferred schedules.
- Treat user/stream seed—not cycle—as the top-level independent unit.
- Do not fit a scaling law to three confirmatory reuse points, four horizon
  values, or public-benchmark rankings.
- Do not use the same model to write and grade a memory.
- Keep exact replay, distillation variants, and dream augmentation in the
  preregistered extension track unless assigned before G2.
