# Sleep-Time Compute Theory and Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the normative theory, leakage-safe lifetime benchmark, matched memory baselines, preregistered 297-cell design, statistical decision rules, and held-out scaling-law tests required by the approved sleep-time compute research design.

**Architecture:** A deterministic generator first creates a hidden oracle world and then emits chronologically available observables into a physically separate stream. Deployable methods receive only past-observable features and compile the same events into raw external, abstract external, latent/KV, user-parametric, or forgotten states under a common resource vector. Theory, experiment design, results, and claims remain separate artifacts connected by digests.

**Tech Stack:** Python 3.14, uv, Pydantic 2, pytest, Hypothesis, NumPy, SciPy `milp`, scikit-learn, statsmodels, PyArrow, Matplotlib, PyYAML, pinned Qwen3-4B, Cartridges, PEFT, and rank_bm25 for execution-only extras.

## Two-pass implementation and pre-data execution law

Tasks 1–13 are first completed as **Phase A: code + fixture tests only**. No
real TRAIN, CALIBRATION, pilot, or TEST bytes are read while any imported
runtime, estimator, power, materializer, gate, or analysis module remains
mutable. Shell blocks that materialize or run real cohorts are normative
**Phase B** commands, not permission to execute them at their textual task
position.

After every bound module and container test passes, freeze a scoped transitive
code/container/config lock. Then, without a bound-code change, Phase B runs in
this order: materialize configs/skeletons/stakes; authorize and execute
disjoint core/capacity-selection CAL; freeze selections; execute and validate
the blinded core pilot and freeze \(B_{\rm ref}\); only then materialize,
authorize, and execute capacity-power CAL with that \(B_{\rm ref}\) digest;
compute all powers;
materialize/finalize core and capacity manifests; freeze core preregistration
and G2; then freeze capacity preregistration and G2-CAP. Every readiness,
receipt, summary, power, and preregistration binds the transitive imported
module closure. An imported-module digest change after CAL stales all
descendants and requires rerunning the affected Phase B branch; a proved
docs-only edit outside the closure does not. TEST remains capability-denied
until its matching G4 snapshot.

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
    CALIBRATION = "calibration"
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
stc design freeze-comparators
stc design power
stc design finalize
stc analyze confirmatory
stc coverage materialize
stc coverage power
stc coverage fit
stc information materialize
stc information power
stc information estimate
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
separation of observables/oracle records. `RunResult.status` is the closed enum
`COMPLETE|RESOURCE_FAILURE|SHARED_PROTOCOL_INVALID|ADMIN_CENSORED|
STRUCTURAL_INFEASIBLE`; failure rows require reason, attempts, affected probes,
score disposition, and the full consumed-resource ledger.

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
- privacy, deletion, p99 wake latency, active/durable retained capacity,
  transient peak-byte capacity, peak accelerator residency, queue stability,
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

### Task 3: Implement the prior-grounded future-task distortion test (H-STC-001)

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
- Treats Colaco and Lahjouji
  ([arXiv:2607.08032](https://arxiv.org/abs/2607.08032)) and semantic-compression
  work as prior grounding. The test does not claim rate-distortion framing,
  query-aware distortion, or reversible tiering as original.

- [ ] **Step 1: Write frontier and non-claim tests**

Reject claims that serialized bytes estimate \(I(E;Z)\). Keep transcript
reconstruction under `secondary_diagnostics`. Require held-out future-query
loss for the primary distortion. Record H-STC-001 as an inherited empirical
proposition; reserve novelty claims for versioned lifecycle routing and its
governance-constrained cross-substrate evaluation.

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

### Task 4: Implement the pre-data H-STC-002 through H-STC-007 contracts

**Files:**
- Create: `src/stc_research/theory/promotion.py`
- Create: `src/stc_research/theory/cadence.py`
- Create: `src/stc_research/theory/capacity.py`
- Create: `src/stc_research/theory/recoverability.py`
- Create: `src/stc_research/theory/plasticity.py`
- Create: `src/stc_research/theory/hypotheses.py`
- Create: `src/stc_research/theory/render.py`
- Modify: `src/stc_research/cli.py`
- Modify: `tests/stc/test_cli_contract.py`
- Create: `tests/stc/theory/test_hypothesis_laws.py`
- Create: `tests/stc/theory/test_render.py`
- Create: `configs/stc/theory/laws.yaml`

**Interfaces:**
- Produces: discounted reuse, break-even, cadence, promotion/demotion,
  multi-capacity, reversibility-reserve, and recoverability-margin
  diagnostics plus separate retention and fixed-budget acquisition probes.
- Produces only derivation, property-test, empirical-curve-preregistration, or
  future-preregistration status. It does not ingest outcomes and cannot emit
  `SUPPORTED`, `FALSIFIED`, or `NARROWED` for an empirical H-STC hypothesis.
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

- [ ] **Step 4: Verify the H-STC-004 extension design and H-STC-005 law**

Compute the seven dimensionless loads:

```text
rho_sleep, rho_store, rho_read, rho_repr, rho_hot, rho_gov, rho_plastic
```

Freeze exactly seven resource-targeted cells. On CALIBRATION only, a target is
eligible as uniquely binding when its load lower bound is at least `0.90` and a
named control resource's load upper bound is at most `0.70`, with cap-hit,
queue, and shadow-price direction agreeing under the frozen rule. A cell that
fails eligibility is `INCONCLUSIVE_ANTECEDENT` and cannot be relabeled after
TEST. Require at least four eligible cells or H-STC-004 is globally
inconclusive.

If this future extension is promoted, its amended G2 must materialize separate
one-at-a-time `+25%` target-capacity and `+25%` nonbinding-control
interventions on the same paired streams. Freeze
`delta_bind=0.01` and `epsilon_nonbind=0.005`. A cell supports the predicted
bottleneck only when the simultaneous adjusted target-effect LCB exceeds
`delta_bind` and the control-effect simultaneous 95% interval lies wholly inside
`[-epsilon_nonbind,+epsilon_nonbind]`. H-STC-004 is supported if at least one
of the exactly seven cells passes under the frozen `H-STC-004-21` family
(seven oriented binding margins plus fourteen nonbinding TOST boundaries),
falsified if every eligible target-effect UCB is below `delta_bind`, and
otherwise narrowed for diagnosis failure, nonbinding gain, or imprecision.
The family uses the same maximum-\(|t|\), CR2, wild-cluster, and seed
sensitivity machinery as the paper family but never enters or changes
`PAPER-C-DEPLOYABILITY-GLOBAL-131`. The current program checks this design for completeness
and records `PREREGISTERED_FUTURE`; it does not execute the intervention rows
or infer an empirical outcome.

Also test byte-flow equilibrium for every store. An archive with no expiry,
deletion, or physical reclamation must be classified unbounded. For independent
opaque payloads, freeze public side information \(\Theta_0\), represent every
incremental answer-bearing carrier in finite precision as \(M_n\), and test the
necessary bound
`sum_i R_i(D_i | Q_i, theta_0) <= I(Y_1:n; M_n | Q_1:n, theta_0)
<= H(M_n | theta_0) <= b_mutable` in bits. Include correlated-source,
lossy-distortion, excluded-archive, changed-base, and
hidden-generator/router-state counterfixtures so the implementation neither
double-counts shared structure nor declares offload to be free capacity. Test
both non-monotone and monotone archive-depth fixtures; do not force an optimum
when the data do not support one.

- [ ] **Step 5: Verify H-STC-006 boundary and future experiment contract**

Represent the complete retained-state inventory—not only base weights—and
verify the no-free-resurrection information boundary symbolically. On
controlled streams, calibrate a past-only recoverability proxy from old-task
canary margin, teacher agreement, source availability, retrieval confidence,
and representational conflict. Compare the same sleep operator before and
after a preregistered proxy threshold and against fixed cadence at matched
compute. Include:

```text
residual trace present + early intervention
residual trace present + late intervention
trace/source removed + independent dream randomness
trace removed + correlated external prior
miscalibrated proxy negative control
```

The no-free-resurrection boundary is a derivation/property-test result when the
independence fixture cannot recover task-specific information. The empirical
scheduling hypothesis remains `PREREGISTERED_FUTURE`: it may be falsified only
after a future amendment binds powered early/late/fixed-cadence manifests,
CALIBRATION thresholds, multiplicity, runtime receipts, and TEST outcomes.

- [ ] **Step 6: Verify the H-STC-007 future experiment contract**

Measure fixed-budget acquisition on held-out new-task probes against an
architecture-matched fresh control, separately from protected-slice old-memory
retention. Cover all four boundary combinations and the intervention controls:

```text
retention pass + acquisition pass
retention pass + acquisition fail
retention fail + acquisition pass
retention fail + acquisition fail
restricted low-utility recycle + canonical replay
random unrestricted recycle negative control
fresh-base upper diagnostic
```

Require held-out-seed replication, the fresh-control normalization, and
protected-slice retention noninferiority. In a future powered execution, the
ordering claim is falsified if no preregistered high-interference regime
crosses the acquisition boundary before the retention boundary. The
intervention claim is separately falsified if restricted recycling fails to
recover acquisition or violates protected-slice noninferiority. The present G2
only validates that arms, endpoints, normalization, and falsifiers are fully
specified; empirical status remains `PREREGISTERED_FUTURE`.

- [ ] **Step 7: Run and commit**

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

Expected: all seven H-STC IDs receive exactly one pre-data status from
`DERIVATION_VERIFIED`, `PROPERTY_TESTED`, `EMPIRICAL_CURVE_PREREGISTERED`, or
`PREREGISTERED_FUTURE`; dimensional checks pass. The command rejects any
pre-G2 `SUPPORTED`, `FALSIFIED`, `NARROWED`, effect estimate, confidence
interval, or TEST-result reference.
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
  → TRAIN/CALIBRATION/TEST split
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
shared with TRAIN/CALIBRATION. Create one ex-ante paired event cohort before assigning R,
future queries, deletion, volatility, or poison. Conditional on randomized R,
assign future queries independently and apply the same administrative
censoring rule across cells. Never condition admission on future validity or
horizon position.

- [ ] **Step 2: Generate all controlled event families**

Include exact facts, drifting preferences, procedures, skills, relations,
negative feedback, correction, contradiction, deletion, poison, and one-off
distractors. Independently vary reuse, live-memory pressure, volatility,
paraphrase difficulty, poison rate, event rate, and idle slack. Exact facts
include a salted opaque-key/uniform-bit-payload family with known conditional
entropy and deterministic equal-information encodings at several text lengths.
TRAIN/CALIBRATION/TEST payload seeds are disjoint.

- [ ] **Step 3: Implement the three frozen transition probes**

Each cycle emits exactly pre-sleep, post-sleep, and delayed-next-wake probes.
The state machine enforces supersession, validity intervals, deletion
precedence, abstention, and dependency closure.

Property tests compare event type, age, payload length, source, volatility,
deletion, poison, and admission marginals across R cells; only assigned future
reuse may differ. They also prove user IDs and workload-family IDs are disjoint
across TRAIN/CALIBRATION/TEST. Opaque-payload tests reject duplicate values,
compressible templates, key leakage, seed reuse across splits, and accidental
coupling between information bits and serialized length.

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

Use `scipy.optimize.milp` for the same five actions and the same active,
durable-retained, transient-peak-byte, sleep-compute, accelerator-residency,
latency, privacy, and deletion caps. The oracle may use the sidecar only
through an explicitly nondeployable interface and cannot violate caps.
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

Freeze action proportions and cadence on validation users pooled across
validation workload families. A configuration may vary with the public
resource cap \(\phi\), base/model digest, and hardware envelope, but never with
generator \(R\), TEST workload-family ID, or a future-distribution label. Its
digest must be invariant across \(R\) and workload family within the same
public envelope. Hash the generator-assigned opaque event UUID with SHA-256 or
HMAC and a published manifest key to assign the frozen allocation bucket; the
method never creates or replaces operational IDs. Python `hash()` is
forbidden. Changing payload text while holding the existing UUID constant must
not change allocation.

- [ ] **Step 2: Implement the hybrid**

Train a cost-sensitive multinomial policy with resource shadow prices and
uncertainty from past-only features. Charge feature construction, controller
training, tuning, and inference. Training users/workload families, validation
users/workload families, and test users/workload families are all disjoint.
Hyperparameters, feature selection, thresholds, and cadence never consume a
TEST family.

Add constructor-boundary tests for every deployable method: `R`,
`workload_family_id`, oracle annotations, and future-query schema are rejected
from `MethodRuntimeSpec`; mutating `R` or family ID with byte-identical
observables and public caps leaves the static/single-medium config digest
unchanged. The hybrid may react only to chronological past-observable features,
never the mutated orchestration fields.

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
- Create: `manifests/stc/deployment-stakes-core.json`
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
digest. The final run index additionally binds `randomization_block_id`,
node/rank/wave/thermal/cache-phase assignments, Williams-order position,
randomization seed, canonical order digest, and balance tolerances.
Physical-run deduplication may reference an identical
full-configuration hash but cannot remove a logical row or contrast.

`RunSpec` is an analysis/orchestration record, not a method input. The runner
constructs a separate `MethodRuntimeSpec` containing only the declared public
resource envelope, base/model/compiler digests, and chronologically observable
state; it omits `R`, `workload_family_id`, oracle fields, and future-query
schema. A type-level allowlist and runtime serialization boundary prevent a
method from retaining the parent `RunSpec`.

Within every `(workload_family_id, R, phi, paired_seed_family)` factorial
block, the four schedule×operator rows must have byte-identical source-set,
enqueue-snapshot, destination, information-cutoff, deadline, compressor,
generator, model, and cap digests. Only `schedule`, `operator`, and their
derived runtime action may differ. Compilation rejects a partial four-row
block or any equality-invariant violation.

Before outcomes, construct a randomized complete-block/Williams schedule
within each
`(workload_family_id, physical_stream_id, paired_seed_family, R, phi)` block.
It balances every deployable method and the four C4 schedule×operator cells
across node, rank, launch wave, wall-clock position, thermal state, and cache
phase. The order seed/digest and tolerances flow through final run index, G2,
G4 execution snapshot, and distributed receipt; the paired-block terms are
identical in power and inference. Assigning hybrid last, pinning one method to
one rank/time block, or drifting the seed/order fails. Permuting file rows
while preserving the frozen execution order is invariant.

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
violation invalidates the complete paired block only when it originates in the
shared generator/query/target/common-oracle artifact and prevents every method
in that block from being scored. A method-produced absent probe, checksum
failure, malformed artifact, or protocol violation is instead the observed
method outcome `RESOURCE_FAILURE`; it is never converted into shared
missingness. Late-horizon events with no scheduled query are administratively
censored and do not enter the denominator. Their admitted-cohort retention is
a named secondary endpoint.

Every result row carries exactly one typed status:

```text
COMPLETE
RESOURCE_FAILURE
SHARED_PROTOCOL_INVALID
ADMIN_CENSORED
STRUCTURAL_INFEASIBLE
```

The run-result schema requires a frozen reason code, attempt lineage, consumed
resource ledger, affected probe IDs, and score disposition. Tests reject an
unknown status, a method-only failure labeled `SHARED_PROTOCOL_INVALID`, or a
failure row with omitted resource use.

Freeze operational failure handling: a second attempt may execute the same
immutable run spec only when the first attempt ended in shared **pre-score**
infrastructure failure before any method outcome existed. The first
checksum-valid terminal `RunResult` envelope—whether `COMPLETE`,
`RESOURCE_FAILURE`, `ADMIN_CENSORED`, or
`STRUCTURAL_INFEASIBLE`—is final; all attempts remain logged, and no observed
method outcome can trigger a retry. A method-artifact checksum failure is
recorded inside a checksum-valid `RESOURCE_FAILURE` envelope. The G2 manifest
contains an ordered reserve-seed list used only for shared pre-score
infrastructure failures. Replacement occurs for the whole paired
method/regime block before outcomes are unblinded. A protocol-valid method
timeout, OOM/cgroup memory kill, compiler exception, deadline miss, cap
violation, or invalid artifact
is a typed `RESOURCE_FAILURE`, not missing data: charge the work, retain the
randomized stream and score every affected probe contribution zero regardless
of whether the target would reward an intentional abstention. Unaffected
pre-failure probes retain their frozen scores. A deliberate protocol-valid
`ABSTAIN`/`FORGET` action under status `COMPLETE` is distinct and may score
correctly when the target requires abstention. Only
shared infrastructure corruption that prevents scoring every method in a
paired block is missing.

Let \(F_{m,c}\) be the intent-to-treat `RESOURCE_FAILURE` proportion for method
\(m\) in cell \(c\). Publish it for every method×cell×workload family and
freeze \(\delta_{\mathrm{RF}}=0.01\) absolute as the one-sided
failure-noninferiority margin. Every primary claim that asserts deployable
dominance has a proposed-minus-selected-comparator failure contrast aggregated
with the same cell/family weights as its utility contrast. The separately
frozen ordered `RESOURCE-FAILURE-VETO-27` paired-score family supplies
simultaneous 95% UCBs; support is
vetoed unless every required UCB is at most
\(\delta_{\mathrm{RF}}\). This family may veto but never create scientific
support. The boundary \(\Delta F=\delta_{\mathrm{RF}}\) is used only to
calibrate one-sided type-I error and simultaneous coverage. Power and
sample-size selection use the interior deployment alternative
\(\Delta F=0\), method-specific correlated failures, and zero-inflated cells;
requiring a UCB below its true boundary is never mislabeled as power.

These paired Bernoulli rows never use an empirical-variance Wald/sandwich
studentizer as their authorization statistic. Within each fixed block, retain
the proposed/comparator failure pair and form a hierarchy-aware constrained
score for \(\Delta F=\delta_{\rm RF}\) from paired discordances and the frozen
weights. Its null covariance is the least-favorable member of the full
rare-event/discordance nuisance set and remains positive at the boundary even
when no failure is observed. Calibrate all 27 directed scores jointly with the
104 scientific row statistics in the one global boundary Monte Carlo
max-statistic. An exact paired-binomial/McNemar inversion is allowed only when
its preregistered equal-weight and independence conditions hold. All-zero and
all-one fixtures yield finite UCBs; all-zero can pass only when its
finite-sample simultaneous bound clears 0.01, never from `SE=0`. Ordinary and
wild-cluster \(t\) results for RF are diagnostics only. A boundary fixture with
comparator failure zero and proposed failure probability 0.01 must remain
within global alpha even when it produces zero observed events.

Freeze the fail-only family as an ordered companion table, with the same
cell/family weights as each scientific contrast:

```text
RF-C2-BENEFIT
RF-C2-EFFICIENCY
RF-C3-RAW-LOW
RF-C3-COMPRESSED-LOW
RF-C3-RAW-HIGH
RF-C3-COMPRESSED-HIGH
RF-C3-LATENT-LORA-LOW
RF-C3-LATENT-LORA-HIGH
RF-C4-OP-{cell_id}-{inline|deferred}       18 rows
RF-C4-IDENTITY-DEFERRED-COST               1 row
```

The exact count is 27 (2 C2 + 6 C3 + 18 C4 operator + 1 C4 cost). Every row
binds scientific-parent ID, comparator direction, harm orientation,
cell/family weights, null/margin, paired-score type, and shared critical function.
These 27 rows join the scientific rows under the global union below.

The C3 Boolean decision imports all six matching failure rows: the low
external OR may use only a utility component whose own failure row passes,
the high external AND requires both, and latent/LoRA requires both. PAPER-C4's
positive operator branch imports the failure row for the exact
cell×schedule component that supports it; the identity deferred-cost branch
imports its aggregate failure row. A missing, reordered, differently weighted,
or nonpassing companion yields
`NARROWED_RESOURCE_FAILURE_NONINFERIORITY`. ITT zero scoring does not replace
this veto.

Infrastructure missingness is an alpha-free paired-integrity/MNAR veto, not a
studentized contrast family. Under the status taxonomy, shared invalidity or
administrative censoring removes the entire paired block, so proposed and
comparator missing indicators are byte-identical and their difference is
structurally zero. Any asymmetric method event must be a scored
`RESOURCE_FAILURE`; an unknown asymmetric missing status invalidates the
schema. Never form a zero-SE `0/0` bootstrap row. Report completion by
method/cell/family and validate pair identity. Primary point estimates use
scoring-eligible paired blocks: every
method has either `COMPLETE` or scored `RESOURCE_FAILURE`, while shared
invalidity and administrative censoring are excluded and structural
infeasibility follows its explicit reason path. Every support decision must
also survive worst-case MNAR
bounds that assign missing
directional outcomes adversarially in `[0,1]` for the proposed method and its
comparator. Cost-bearing C2/C4 branches additionally freeze a strictly positive
cell/method cost floor and a reservation-derived native-cost upper envelope.
Their joint partial-identification bound assigns unresolved proposed cost to
the upper envelope and comparator cost to the lower floor (and reverses them
for the diagnostic bound) while varying utility adversarially. If that
worst-case lifecycle-cost-reduction margin does not clear 0.10, or any missing
cost lies outside the frozen support, the efficiency branch is narrowed; a
scalar complete-case cost ratio is forbidden. A missing-cost mutation must
overturn an apparent 10% saving. Support is prohibited if any family completes
below 95%, paired-integrity fails, or the worst-case bound loses support. A
cap-respecting method may choose
`FORGET` and abstain, which remains a scored feasible outcome. A structurally
infeasible required cell is reported as infeasible, never scored as zero, and
prevents support for a contrast that needs it.

The crossed 16 TEST workload families×3 paired-seed families are fixed,
predeclared finite-target blocks, not IID superpopulation clusters. Primary
uncertainty uses 20,000 deterministic blocked paired bootstrap replicates:
keep all 48 blocks once with equal workload-family and equal seed-family
weights, and resample top-level paired user/stream IDs within each block,
retaining every method, R, phi, schedule, operator, cycle, and probe for a
selected stream. Studentize each of the 104 continuous scientific rows with a
48-block paired-stream-cluster sandwich standard error; compute each of the 27
RF rows with its boundary-positive paired score above. Use the replicate-wise
maximum absolute standardized row statistic across the single ordered global
family `PAPER-C-DEPLOYABILITY-GLOBAL-131` defined below. The familywise 95%
critical value is the corresponding joint boundary-Monte-Carlo/bootstrap
quantile;
claim-specific critical values are forbidden. Report adjusted 95% intervals,
the two signed adjusted-95% equivalence margins, all unadjusted intervals, and
leave-one-paired-seed-family-out sensitivity. Prevalidate CR2 coverage in the
power simulation; both simulation and inference include the frozen
node/rank/wave/thermal/cache randomization-block terms and paired-block
covariance. Require a 99,999-draw paired-stream wild-cluster bootstrap
sensitivity stratified within workload×seed block; a claim cannot be supported when the
primary and wild-cluster decisions or seed-family sensitivity disagree.
Generalization is scoped to the equal-weight finite mixture of these 16 frozen
families, not a workload-family generator population or arbitrary real-world
tasks. Duplicating or splitting a family label without changing the finite
mixture cannot change the point estimate, standard error, or decision.

Every continuous scientific-row bootstrap statistic is explicitly centered.
For ordered transformed margin \(m_j=s_j\hat\theta_j-b_j\), replicate \(b\)
uses

```text
t[b,j] = ((s[j] * theta_hat[b,j] - b[j]) - m_hat[j]) / se[b,j]
max_t[b] = max_j abs(t[b,j])
```

where \(s_j\), boundary \(b_j\), orientation, and studentizer are frozen before
resampling. Failure noninferiority is represented by the separate constrained
paired score for
\(m_j=\delta_{\mathrm{RF}}-\Delta F_j\); each TOST row contributes its two
predeclared continuous boundary margins. Using
\((s_j\hat\theta_{b,j}-b_j)/SE_{b,j}\) without subtracting the observed
transformed estimate is forbidden because signal would inflate the critical
value. The Webb wild-cluster sensitivity uses the same centered transformed
scientific rows; its RF \(t\) analogue is diagnostic. Mutation tests omit
centering for ordinary/TOST rows or replace the RF score with zero-SE
studentization and must change the implementation digest or fail.

Freeze these exact primary contrast functions in
`analysis-plan-skeleton.json`:

- C1: for each of nine cells and each of four medium-restricted oracles \(j\),
  define \(G_{cj}=U_{\mathrm{unrestricted},c}-U_{j,c}\). The envelope gap is
  the nonsmooth functional \(D_c=\min_jG_{cj}\), so no bootstrap may resample a
  max/argmax selector. Instead, all 36 oriented component margins
  \(G_{cj}-\delta_{\mathrm{C1}}\), with
  \(\delta_{\mathrm{C1}}=0.01\), enter the global simultaneous family. Under
  their joint coverage, a valid derived interval for \(D_c\) is
  \([\min_j L_{cj},\min_j U_{cj}]\).
- C2: a 1/9 equal-cell mean over the nine R×phi cells using two G2-frozen,
  \(R\)-blind CALIBRATION comparator maps per public
  \(\phi\). The benefit map selects the highest pooled CALIBRATION utility
  among four single media and static mixture, with canonical ID for an exact
  tie. The efficiency map selects the lowest-cost member within
  \(\delta_{\mathrm{CAL}}=0.01\) of that CALIBRATION maximum under the complete
  price manifest. TEST never recomputes either max/argmax; all five TEST
  baseline comparisons are reported as diagnostics. The three ordered rows are
  benefit \(\Delta U-0.02\), utility noninferiority \(\Delta U+0.01\), and
  lifecycle-cost reduction \((1-\bar C_h/\bar C_b)-0.10\). Benefit support
  requires the adjusted 95% LCB of its margin above zero. Efficiency support
  requires both noninferiority and cost-margin adjusted LCBs above zero; point
  estimates never substitute for margin confidence. Independently
  freeze full-oracle-minus-hybrid regret per cell and its 1/9 equal-cell mean.
  Optional oracle-gap capture fraction
  \((U_h-U_b)/(U_o-U_b)\) is admissible only when the paired denominator is
  positive and at least 0.01; otherwise report `CAPTURE_FRACTION_INADMISSIBLE`.
- C3: for each of the two adjacent pairs
  `(external-envelope, latent/KV)` and `(latent/KV, user-LoRA)`, freeze one
  low-promotion and one high-promotion corner from H-STC-002 plus
  CALIBRATION-only feasibility, before pilot condition labels or TEST outcomes.
  The deployment-stakes record fixes \(\delta_{cross}=0.01\). All utilities
  come from the four tuned **deployable** single-medium method IDs
  `raw_external`, `compressed_external`, `latent_kv`, and `user_lora`;
  medium-restricted clairvoyant oracles belong only to C1 and are rejected from
  every C3 row. Define the six oriented effects
  \(g^L_r=U_r-U_l\), \(g^L_c=U_c-U_l\),
  \(g^H_r=U_l-U_r\), \(g^H_c=U_l-U_c\),
  \(g^L_{lp}=U_l-U_p\), and \(g^H_{lp}=U_p-U_l\).
  The six global-family members are exactly
  \(g-\delta_{cross}\), not merely \(g\) against zero. The low external
  condition requires at least one adjusted LCB \(>0\); its high condition
  requires both. The latent/LoRA condition requires both direct component
  LCBs \(>0\). Report the low external-envelope effect with componentwise
  maxima and the high latent-minus-envelope effect with componentwise minima,
  plus bootstrap-derived reversal spans; no ordinal interaction or selected
  TEST max may rescue the test.
- C4: enumerate, in every one of nine R×phi regimes, two operator-specific
  schedule simple effects, two schedule-specific operator simple effects, and
  the schedule×operator interaction, plus the schedule-marginal operator main
  effect: 54 reported contrasts total. Those rows plus the five aggregate
  boundary rows listed below are the 59 C4 members of
  the 104-row scientific subtable; no C4-local critical value exists.

Freeze this exact ordered multiplicity table:

```text
PAPER-C-GLOBAL-104
  C1: 36 cellwise unrestricted-minus-each-restricted component margins
      9 cells x 4 media, each G_cj - delta_C1 against 0
  C2: 3 margins
      benefit utility minus 0.02 against the G2-frozen CAL benefit map
      utility plus 0.01 against the G2-frozen CAL efficiency map
      lifecycle-cost reduction minus 0.10 against the same efficiency map
  C3: 6 oriented crossover component margins
      4 deployable raw/compressed-vs-latent g - delta_cross margins
      2 deployable latent-vs-user-LoRA g - delta_cross margins
  C4: 59 margins
      54 cellwise schedule/operator/main/interaction contrasts; every
         claim-capable operator row is O - 0.02
      2 cell-balanced identity-schedule TOST boundaries
      2 cell-balanced interaction TOST boundaries
      1 cell-balanced lifecycle-cost-reduction minus 0.10
```

`PAPER-C-GLOBAL-104` is the scientific-effect subtable. Every deployable
support decision uses the larger ordered union
`PAPER-C-DEPLOYABILITY-GLOBAL-131`:

```text
104 scientific utility/cost/equivalence margins
+ 27 RESOURCE-FAILURE-VETO companion margins
= 131 ordered margins under one joint row-specific max-stat critical value at alpha=.05
```

RF rows remain veto-only—they can never create efficacy—but sharing the
131-row critical is necessary because their truth is part of the deployability
claim. Completion integrity/MNAR remains an alpha-free hard veto. A mixed-null
simulation in which one claim is false only on efficacy and another only on RF
must keep the probability of any false `SUPPORTED` status at most 0.05.

The ordered contrast IDs, orientations, null boundaries, row-statistic types,
and single `critical_95` joint boundary quantile are serialized in
`analysis-plan-skeleton.json`. Point-materiality checks, oracle-regret/capture
diagnostics, \(I_j\), native resource vectors, and hypervolume are reported but
cannot enter or alter the 104-row scientific subtable. Task 11 power and Task
12 inference import the complete 131-row ordered union and one critical-value
function; tests permute, omit, and add a
contrast and require an implementation-digest failure.

Power is evaluated at deployment-stake alternatives fixed before the blinded
pilot, not estimated effects. For C1, place material heterogeneity
\(\Delta U=0.02=\delta_{\mathrm{C1}}+0.01\) in turn in each of the exactly nine
reuse×pressure cells, require all four component margins in that location, and
bind the maximum required \(n\) across those nine location scenarios. C2 benefit
\(\Delta U=0.03\); C2 efficiency \(\Delta U=0\) with 0.15 lifecycle-cost
reduction. For the C3 external branch, exactly one prespecified low-corner
external-minus-latent component and both high-corner
latent-minus-external components equal
\(\delta_{cross}+0.01\), while the unused low component is zero; repeat with
each possible low component and bind the larger required \(n\). For the
latent/LoRA branch, both oriented components equal
\(\delta_{cross}+0.01\). For C4, enumerate 27 claim-capable locations as
feasible underlying 2×2 cell means, not freely injected derived contrasts.
Eighteen simple-effect scenarios are the nine R×phi regimes crossed with
`{inline,deferred}`: at the active regime set both identity cells to zero, the
selected consolidating schedule cell to `0.03`, and the other consolidating
schedule cell to zero, up to a common admissible offset; all other regimes have
zero operator effect. Nine main-effect scenarios set both active-regime
consolidating cells to `0.03` and both identity cells to zero. Recompute all 54
derived contrasts in every scenario. Thus a simple-only scenario has one
selected simple effect `0.03`, `O_main=0.015`, and the algebraically implied
signed interaction, whereas a main scenario has
`O_inline=O_deferred=O_main=0.03` and `I=0`. Freeze every other
identity/operator/schedule utility and cost nuisance by one pre-data rule, pair
each scenario with 0.15 identity-deferral cost reduction, and bind the maximum
required \(n\) over all 27 scenarios. The composite-success simulation must
achieve at least 0.90 power for the simple-only mutation
`O_inline=.03, O_deferred=0`, whose nine-cell marginal interaction can still
lie inside the aggregate ±0.01 separability margin. Directly injecting an
inconsistent derived row—such as `O_main=.03` with both simple effects zero,
or `I=0` with unequal simple effects—fails. Wildcard
or unspecified locations are invalid; moving the high-variance opaque stratum
onto a powered location must preserve or increase \(n\). These one-point utility and five-point cost
buffers are justified in a pre-data deployment-stakes record and receive
this exact nonbinding sensitivity grid:

```text
utility gain or crossover buffer: {0.005, 0.010, 0.015}
cost-reduction buffer above 0.10:  {0.025, 0.050, 0.100}
equivalence true effect:           {-0.005, 0, +0.005}
infrastructure missing rate:       {0, 0.025, 0.05}
```

The selection-binding effects are the central alternatives stated above,
equivalence effect zero. Before any pilot observation, freeze one
hierarchy-aware covariance-set builder; choosing the narrowest builder,
shrinkage, or support bound after viewing pilot dispersion is forbidden. Its
center is the 131-vector paired-user influence covariance from all 48 fixed
workload-family×paired-seed-family blocks, with every block retained and users
resampled only within block. Utility and failure coordinates have support
`[0,1]`; cost coordinates use the positive denominator floor and
reservation-derived upper envelope frozen in deployment stakes. Values outside
those supports make the pilot inadmissible rather than silently winsorized.
The center uses a pre-frozen diagonal shrinkage coefficient and block-weight
rule. A block bootstrap forms simultaneous max-standardized bands for every
unique covariance entry; a pre-frozen finite-support remainder is added, and
\(\mathcal C_\Sigma\) contains every PSD matrix satisfying all bands. If a
non-PSD numerical center must be repaired, only a recorded PSD projection plus
the smallest common diagonal inflation that restores every registered
contrast-variance lower target is permitted.

Before real pilot collection, a nomination Monte Carlo screens
`n_pilot_per_block ∈ {4,8,12,16}` over near-singular and near-independent
correlations, correlated heteroskedastic block effects, rare high-variance
strata, and the frozen bounded supports. It may nominate the smallest apparent
pass, but cannot authorize it. A disjoint fixed-size verification stream then
tests only that nominated size with a candidate×stress-grid simultaneous lower
99% bound (or an equivalent preregistered alpha-spending rule); authorization
requires full covariance-object coverage at least 0.99 and familywise
false-selection probability at most 0.01 over the entire size search. Reusing
nomination draws or scanning verification until a size passes is forbidden.
This one size, shrinkage coefficient, entry studentizers, PSD/inflation rule,
support remainder, bootstrap count/seed, and stress-grid digest are then
immutable; if no size is authorized, pilot execution is blocked. A boundary
fixture with all four builders at true 0.99 object coverage must
false-authorize any undercovered choice with probability at most 0.01. The real blinded
pilot constructs one familywise 99% confidence/tolerance set
\(\mathcal C_\Sigma\) (`alpha_Sigma=0.01`); pointwise upper-95 components are
insufficient. Its schema binds the opaque variance-stratum mapping digest,
top-level paired-user hierarchy, chosen pilot size, resampling seed/count,
builder/version, simultaneous critical value, support bounds, shrinkage and
PSD/inflation diagnostics, and complete covariance-object digest.

Covariance alone does not identify finite-sample power. The same pre-pilot
contract therefore freezes a full cluster-level DGP set
\(\mathcal C_{\rm DGP}\): pilot-cluster bootstrap generators plus bounded
Gaussian, Student-\(t\), skewed-beta, beta-binomial rare-discordance,
zero-inflated cost, and copula-perturbation families whose marginals, support,
missingness, block effects, and covariance all lie in their simultaneous outer
sets. Power is a joint minimax calculation: for every
\(P\in\mathcal C_{\rm DGP}\) (and hence
\(\Sigma(P)\in\mathcal C_\Sigma\)), simulate the complete ordered 131-contrast
vector, recompute that draw's shared max-\(|t|\) critical values and all Boolean
PAPER-C decisions, and minimize composite-success probability over the
admissible set. A validated conservative global bound may replace explicit
minimax simulation only if it uniformly upper-bounds the critical value and
lower-bounds every branch's success over \(\mathcal C_{\rm DGP}\). Checking only
\(\max_{\Sigma\succeq0,\Sigma\in\mathcal C_\Sigma}c^\top\Sigma c\) one contrast
at a time is a diagnostic, not a power rule, because it loses the
cross-contrast dependence that determines the max-\(|t|\) critical value. If
an implementation builds endpoint matrices first, its PSD
repair must use recorded diagonal inflation and prove every registered
contrast variance is at least its pre-repair target; an unconstrained
nearest-PSD projection is forbidden because it may reduce a contrast
variance. Mutation tests include incompatible correlation endpoints whose
nearest-PSD projection shrinks one target contrast and a multicomponent case
where pointwise but not simultaneous 95% coverage holds. A second mutation
holds every \(c^\top\Sigma c\) fixed while changing the 131-way dependence
from near-perfect correlation to near-independence; the latter must induce the
larger global critical value or weaker composite power. A pilot mutation in
which every entry has pointwise 95% coverage but the full covariance object
misses more than 1% must enlarge the set or fail. The selection-binding
shared-infrastructure-missingness alternative is exactly `0`, because the
small deployment buffers cannot survive adversarial `[0,1]`
partial-identification at positive missingness even with infinite \(n\).
Rate `0.025` is a mandatory stress/sensitivity scenario expected to narrow
small-effect claims; `0.05` is the completion-feasibility boundary.
If the blinded-pilot upper 95% bound exceeds `0.05`, G2 returns
`FAIL_INFRASTRUCTURE_COMPLETION_FEASIBILITY`; it may narrow the claim or trigger
a prospective design amendment, but increasing \(n\) may not pretend to solve
a boundary completion probability. For the fail-only family, type-I calibration
uses \(\Delta F=\delta_{\mathrm{RF}}\) and power uses \(\Delta F=0\).
Other grid points are reported diagnostics and cannot increase or decrease
selected \(n\). These choices do not change the support thresholds above;
changing them after pilot labels or TEST results is a new G2 design.

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
  --deployment-stakes-output manifests/stc/deployment-stakes-core.json
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
- Create: `src/stc_research/design/comparators.py`
- Create: `src/stc_research/design/power.py`
- Create: `src/stc_research/design/calibration.py`
- Create: `src/stc_research/analysis/scaling.py`
- Create: `src/stc_research/analysis/coverage_scaling.py`
- Create: `src/stc_research/analysis/parametric_information.py`
- Create: `configs/stc/design/comparator-selection.yaml`
- Create: `configs/stc/design/calibration.yaml`
- Create: `configs/stc/design/calibration-environment.yaml`
- Create: `configs/stc/design/design-development-budget.yaml`
- Create: `configs/stc/design/power.yaml`
- Create: `configs/stc/design/cost-profile.yaml`
- Create: `configs/stc/design/rre-reference-native.yaml`
- Create: `configs/stc/design/program-budget-envelope.yaml`
- Create: `configs/stc/design/scaling.yaml`
- Create: `configs/stc/design/scaling-hardware.yaml`
- Create: `configs/stc/design/coverage-scaling.yaml`
- Create: `configs/stc/design/parametric-information.yaml`
- Create: `configs/stc/design/capacity-estimators.yaml`
- Create: `manifests/stc/deployment-stakes-G2-CAP.json`
- Create: `manifests/stc/scaling-contrast-family-G2-CAP.json`
- Create: `manifests/stc/coverage-contrast-family-G2-CAP.json`
- Create: `manifests/stc/calibration-baseline-summary.json`
- Create: `manifests/stc/training-calibration-core-plan.json`
- Create: `manifests/stc/training-calibration-core-cells.jsonl`
- Create: `manifests/stc/training-calibration-core-cohort-index.json`
- Create: `manifests/stc/training-calibration-core-budget.json`
- Create: `manifests/stc/capacity-selection-plan.json`
- Create: `manifests/stc/capacity-selection-cells.jsonl`
- Create: `manifests/stc/capacity-selection-cohort-index.json`
- Create: `manifests/stc/capacity-selection-budget.json`
- Create: `manifests/stc/capacity-power-calibration-plan.json`
- Create: `manifests/stc/capacity-power-calibration-cells.jsonl`
- Create: `manifests/stc/capacity-power-calibration-cohort-index.json`
- Create: `manifests/stc/capacity-power-calibration-budget.json`
- Create: `manifests/stc/design-development-budget.json`
- Create: `manifests/stc/training-calibration-core-readiness.json`
- Create: `manifests/stc/capacity-selection-readiness.json`
- Create: `manifests/stc/capacity-power-calibration-readiness.json`
- Create: `manifests/stc/training-calibration-core-result-validation.json`
- Create: `manifests/stc/capacity-selection-result-validation.json`
- Create: `manifests/stc/capacity-power-calibration-result-validation.json`
- Create: `manifests/stc/selection-bundle-G2.json`
- Create: `manifests/stc/capacity-selection-bundle-G2-CAP.json`
- Create: `manifests/stc/capacity-calibration-summary.json`
- Create: `manifests/systems/training-calibration-core-receipt.json`
- Create: `manifests/systems/capacity-selection-receipt.json`
- Create: `manifests/systems/capacity-power-calibration-receipt.json`
- Create: `manifests/systems/pilot-receipt.json`
- Create: `manifests/stc/comparator-maps-G2.json`
- Create: `manifests/stc/blinded-pilot-summary.json`
- Create: `manifests/stc/pilot-result-validation.json`
- Create: `manifests/stc/power.json`
- Create: `manifests/stc/confirmatory.jsonl`
- Create: `manifests/stc/confirmatory-run-index.json`
- Create: `manifests/stc/confirmatory-summary.json`
- Create: `manifests/stc/confirmatory-budget.json`
- Create: `manifests/stc/capacity-coarse-feasibility.json`
- Create: `manifests/stc/analysis-plan.json`
- Modify: `src/stc_research/cli.py`
- Modify: `tests/stc/test_cli_contract.py`
- Create: `tests/stc/design/test_comparators.py`
- Create: `tests/stc/design/test_calibration.py`
- Create: `tests/stc/design/test_power.py`
- Create: `tests/stc/analysis/test_scaling.py`
- Create: `tests/stc/analysis/test_scaling_materialize.py`
- Create: `tests/stc/analysis/test_coverage_scaling.py`
- Create: `tests/stc/analysis/test_parametric_information.py`
- Create: `schemas/stc/coverage-scaling-cell-v1.json`
- Create: `schemas/stc/parametric-information-cell-v1.json`
- Create: `schemas/stc/parametric-information-result-v1.json`
- Create: `tests/stc/fixtures/calibration-baselines.jsonl`
- Create: `tests/stc/fixtures/pilot-summary.json`

**Interfaces:**
- Selects one user-stream count before G2 from a blinded pilot.
- Materializes the only two claim-bearing PAPER-C2 comparator maps from
  CALIBRATION and binds their exact digest before power, finalization, and G2.
- Owns a pre-G2, G1-descended calibration authorization and execution path that
  cannot address TEST bytes. Comparator, codec/policy, and decoder freezes may
  consume only receipt-validated CALIBRATION outputs from this path.
- Freezes the exact lifecycle-cost functional used by PAPER-C2/C4.
- Imports every utility, contrast, feasibility, and bootstrap function from
  `analysis/estimands.py`; simulation-only reimplementations are forbidden.
- Imports the ordered `PAPER-C-DEPLOYABILITY-GLOBAL-131` membership table,
  orientations, null
  boundaries, studentizer, and shared critical-value function from
  `analysis-plan-skeleton.json`; a local or claim-specific multiplicity table
  is forbidden.
- Owns and parser-tests `pilot run`, `pilot summarize`, `design power`, and
  `design finalize` with the exact options shown below. It also owns
  `design freeze-comparators --config PATH --calibration-results PATH
  --selection-bundle PATH --validation-report PATH --cost-profile PATH
  --summary-output PATH --output PATH`.

Every `* power` command uses a preregistered search stream only to nominate a
candidate and a disjoint fixed-size verification stream to authorize it.
Verification computes familywise simultaneous bounds over every candidate and
selection-binding branch, or an equivalently valid monotone alpha-spending
bound with `alpha_MC=0.01`. Pointwise 95% binomial bounds after scanning candidates, reuse of search
draws, and optional continuation are invalid. The common mutation fixture
places many candidates exactly at true power 0.90 and requires the probability
that any is falsely authorized conditional on the pilot set to remain at most
0.01. Together with `alpha_Sigma=0.01`, the union-bound unconditional
false-authorization assurance is at most 0.02; both budgets and their
construction digests are serialized in the power record.

The capacity annex freezes two co-primary alpha accounts before G2-CAP:
`\(\alpha_{\rm scaling}=0.025\)` for the complete A100 scaling-law family and
`\(\alpha_{\rm coverage}=0.025\)` for the joint max-\(|t|\) family containing
`COV-CARD-SLOPE` and `COV-FIXED-BITS-PRED`. No alpha is recycled after results.
Information-capacity outputs remain estimator qualification/characterization
and consume no claim-bearing alpha. A global-null simulation must keep the
probability that at least one scaling or coverage claim is supported at most
0.05. Core PAPER-C and the capacity annex are separately named, always jointly
reported confirmatory families; neither may be selectively relabeled as a
single paper-wide “any success” endpoint.

Before any data-producing command, freeze and validate the sponsor envelope
and prove that even the optimistic mandatory design can fit it:

```bash
uv run stc budget coarse-feasibility \
  --program-envelope configs/stc/design/program-budget-envelope.yaml \
  --rre-reference configs/stc/design/rre-reference-native.yaml \
  --calibration-config configs/stc/design/calibration.yaml \
  --development-budget-config configs/stc/design/design-development-budget.yaml \
  --power-config configs/stc/design/power.yaml \
  --capacity-estimator-config configs/stc/design/capacity-estimators.yaml \
  --confirmatory-skeleton manifests/stc/confirmatory-skeleton.jsonl \
  --scaling-config configs/stc/design/scaling.yaml \
  --coverage-config configs/stc/design/coverage-scaling.yaml \
  --information-config configs/stc/design/parametric-information.yaml \
  --output manifests/stc/capacity-coarse-feasibility.json
```

The calculation prices core TRAIN/CAL and pilot; the full
capacity-selection search; the disjoint power-CAL lattice; every bracket
candidate `{2^0,...,2^24}` at all prospective tuples; every codec/policy dry
run and decoder/critic candidate; fresh-retrain calibration; the smallest
allowed \(n\) candidate for each production track; every mandatory physical
row/panel/lag/fresh retrain; and the complete 297n clean reproduction. It uses
an optimistic engineering lower cost. If any development, core-primary,
core-clean, capacity-RRE, native
resource, or carbon/energy lower bound exceeds its sponsor cap, it returns
`FAIL_DESIGN_INFEASIBLE`; no pilot or power-CAL may start. Passing is necessary
but not sufficient: the later joint upper-99 freezes can still fail. Mutation
tests omit the max-load bracket candidate, one decoder candidate, the pilot,
one eight-panel scaling component, one clean-reproduction cohort, or one native
counter and require rejection.

- [ ] **Step 0: Authorize and execute CALIBRATION without TEST access**

`calibration.yaml` materializes only TRAIN/CALIBRATION rows for hybrid
oracle-imitation, static cadence/proportion search, every single-medium
hyperparameter/early-stop choice, comparator baselines, coverage
metric/codec/policy selection, and information decoder/optional critic
selection. Capacity CALIBRATION is then split again. The
`capacity-selection-*` cohort selects coverage metric/codec/policy, the
information decoder/optional critic, and one tuple-specific eight-load
logarithmic bracket for every prospective scaling resource×method tuple. Each
bracket is chosen by the deterministic predeclared rule below from the complete
frozen candidate grid; selected loads, rejected candidates, selection
statistics, and the tie-break are bound in
`capacity-selection-bundle-G2-CAP.json`. A disjoint
`capacity-power-calibration-*` cohort applies that frozen selection and alone
estimates coverage radius/codec residual covariance, binary-channel
precision/coverage and fresh-retrain cost for every information
substrate/precision, and scaling response plus cost. The scaling calibration
contains blinded target-A100 128-cycle response panels at all eight loads and
all four lags for every one of the 120 prospective production
resource×method configurations, using disjoint CALIBRATION streams; it estimates
retention-binomial/overdispersion, paired-user and workload-family covariance,
separation/finite-knee/failure rates, resource-dependent residuals, and every
component of the eight-panel cost. A preregistered variance/cost model may
smooth within this exact lattice and must emit a simultaneous upper prediction
envelope, but no production tuple may lie outside its support. None is imputed
from the generic PAPER-C pilot or an unseen panel/load/resource cell; a missing
component, unsupported tuple, or point-only interpolation fails G2-CAP.

The core, capacity-selection, and capacity-power plans bind mutually disjoint
unit IDs, source/payload seeds, and derivative closure. The power summary may
consume only the second capacity cohort and the already-frozen selection
bundle; it cannot see a selection-cohort outcome. For every selected artifact,
the selection plan also binds the complete search space, selection functional,
canonical tie-break, maximum trials/resources, seed hierarchy, and required
losing-trial log. Every plan rejects any TEST workload, user, payload seed,
derivative, URI, path, or digest.
`design-development-budget.json` gives pilot and TRAIN/CALIBRATION
positive non-borrowable quotas in a budget class separate from both the
297-cell confirmatory design and the shared sponsor-frozen
\(Q_{\rm capacity,max}\) program; unused
work never transfers across those classes.

`stc calibration authorize` is a typed, content-addressed authorization
descended from PASS G1, not a scientific result gate. It binds the
TRAIN/CALIBRATION split/config, generator property report, logical/cohort
manifests, selection plan, budget, checkout tree, `uv.lock`, model/upstream
locks, and exact calibration environment. Its closed policy is scope-disjoint:
`scope=core` permits only `runtime mode=training-calibration` and the
separately typed TRAIN-only `pilot run`; `scope=capacity-selection` permits
only `runtime mode=capacity-calibration --calibration-role selection`; and
`scope=capacity-power` permits only
`runtime mode=capacity-calibration --calibration-role power`. All are
G1-descended pre-TEST exceptions
and deny every TEST URI, derivative, capability, or path before worker launch.
A readiness presented across these roles, a missing role, or any readiness
presented with a TEST reference is a prelaunch authorization failure. The
authorizer also fails if selection and power cohorts overlap in unit, seed, or
derivative closure. A later change to any bound
code/environment/split/model digest stales the authorization and, if it occurs
after maps are frozen, stales G2 as well.

The later G4 execution snapshot performs an exact transport check against these
readiness records. Hardware-independent generator/model/tokenizer/code fields
must be byte-identical. Every hardware-dependent tuning field—batching,
cadence, early stop, shadow price, throughput/cost, and comparator
efficiency—must come from the same target A100 model, container, driver/CUDA,
clock/power policy, serving stack, and cost-profile source/date used for TEST.
A CPU/local calibration, changed container/clock/price profile, or silent
batch-size retune stales the relevant G2 or G2-CAP selection edge and blocks
TEST. Cross-environment transport requires a separately powered, CAL-only rule
frozen before either TEST capability is issued.

```bash
uv run stc capacity stakes materialize \
  --coverage-config configs/stc/design/coverage-scaling.yaml \
  --information-config configs/stc/design/parametric-information.yaml \
  --scaling-config configs/stc/design/scaling.yaml \
  --hardware-config configs/stc/design/scaling-hardware.yaml \
  --estimator-config configs/stc/design/capacity-estimators.yaml \
  --output manifests/stc/deployment-stakes-G2-CAP.json
uv run stc calibration materialize-selection \
  --config configs/stc/design/calibration.yaml \
  --coverage-config configs/stc/design/coverage-scaling.yaml \
  --information-config configs/stc/design/parametric-information.yaml \
  --scaling-config configs/stc/design/scaling.yaml \
  --hardware-config configs/stc/design/scaling-hardware.yaml \
  --estimator-config configs/stc/design/capacity-estimators.yaml \
  --capacity-stakes manifests/stc/deployment-stakes-G2-CAP.json \
  --generator-config configs/stc/benchmark/controlled-128.yaml \
  --generator-property-report manifests/stc/generator-property-report.json \
  --development-budget-config configs/stc/design/design-development-budget.yaml \
  --core-plan-output manifests/stc/training-calibration-core-plan.json \
  --core-output manifests/stc/training-calibration-core-cells.jsonl \
  --core-cohort-index-output manifests/stc/training-calibration-core-cohort-index.json \
  --core-budget-output manifests/stc/training-calibration-core-budget.json \
  --capacity-selection-plan-output manifests/stc/capacity-selection-plan.json \
  --capacity-selection-output manifests/stc/capacity-selection-cells.jsonl \
  --capacity-selection-cohort-index-output manifests/stc/capacity-selection-cohort-index.json \
  --capacity-selection-budget-output manifests/stc/capacity-selection-budget.json \
  --development-budget-output manifests/stc/design-development-budget.json
uv run stc calibration authorize \
  --predecessor manifests/gates/G1.json \
  --plan manifests/stc/training-calibration-core-plan.json \
  --manifest manifests/stc/training-calibration-core-cells.jsonl \
  --cohort-index manifests/stc/training-calibration-core-cohort-index.json \
  --budget manifests/stc/training-calibration-core-budget.json \
  --development-budget manifests/stc/design-development-budget.json \
  --generator-property-report manifests/stc/generator-property-report.json \
  --environment configs/stc/design/calibration-environment.yaml \
  --scope core \
  --output manifests/stc/training-calibration-core-readiness.json
uv run stc calibration authorize \
  --predecessor manifests/gates/G1.json \
  --plan manifests/stc/capacity-selection-plan.json \
  --manifest manifests/stc/capacity-selection-cells.jsonl \
  --cohort-index manifests/stc/capacity-selection-cohort-index.json \
  --budget manifests/stc/capacity-selection-budget.json \
  --development-budget manifests/stc/design-development-budget.json \
  --generator-property-report manifests/stc/generator-property-report.json \
  --environment configs/stc/design/calibration-environment.yaml \
  --scope capacity-selection \
  --coverage-config configs/stc/design/coverage-scaling.yaml \
  --information-config configs/stc/design/parametric-information.yaml \
  --scaling-config configs/stc/design/scaling.yaml \
  --hardware-config configs/stc/design/scaling-hardware.yaml \
  --capacity-stakes manifests/stc/deployment-stakes-G2-CAP.json \
  --estimator-config configs/stc/design/capacity-estimators.yaml \
  --output manifests/stc/capacity-selection-readiness.json
uv run stc runtime run-matrix \
  --mode training-calibration \
  --manifest manifests/stc/training-calibration-core-cells.jsonl \
  --cohort-index manifests/stc/training-calibration-core-cohort-index.json \
  --budget manifests/stc/training-calibration-core-budget.json \
  --calibration-readiness manifests/stc/training-calibration-core-readiness.json \
  --output results/training-calibration-core-staging
uv run stc runtime run-matrix \
  --mode training-calibration \
  --manifest manifests/stc/training-calibration-core-cells.jsonl \
  --cohort-index manifests/stc/training-calibration-core-cohort-index.json \
  --budget manifests/stc/training-calibration-core-budget.json \
  --calibration-readiness manifests/stc/training-calibration-core-readiness.json \
  --verify-only \
  --receipt manifests/systems/training-calibration-core-receipt.json \
  --output results/training-calibration-core-staging
uv run stc runtime import-matrix \
  --receipt manifests/systems/training-calibration-core-receipt.json \
  --source results/training-calibration-core-staging \
  --output results/training-calibration-core
uv run stc results validate \
  --calibration-readiness manifests/stc/training-calibration-core-readiness.json \
  --manifest manifests/stc/training-calibration-core-cells.jsonl \
  --cohort-index manifests/stc/training-calibration-core-cohort-index.json \
  --results results/training-calibration-core \
  --output manifests/stc/training-calibration-core-result-validation.json
uv run stc selection freeze-core \
  --plan manifests/stc/training-calibration-core-plan.json \
  --results results/training-calibration-core \
  --validation-report manifests/stc/training-calibration-core-result-validation.json \
  --output manifests/stc/selection-bundle-G2.json
uv run stc runtime run-matrix \
  --mode capacity-calibration \
  --calibration-role selection \
  --manifest manifests/stc/capacity-selection-cells.jsonl \
  --cohort-index manifests/stc/capacity-selection-cohort-index.json \
  --budget manifests/stc/capacity-selection-budget.json \
  --calibration-readiness manifests/stc/capacity-selection-readiness.json \
  --output results/capacity-selection-staging
uv run stc runtime run-matrix \
  --mode capacity-calibration \
  --calibration-role selection \
  --manifest manifests/stc/capacity-selection-cells.jsonl \
  --cohort-index manifests/stc/capacity-selection-cohort-index.json \
  --budget manifests/stc/capacity-selection-budget.json \
  --calibration-readiness manifests/stc/capacity-selection-readiness.json \
  --verify-only \
  --receipt manifests/systems/capacity-selection-receipt.json \
  --output results/capacity-selection-staging
uv run stc runtime import-matrix \
  --receipt manifests/systems/capacity-selection-receipt.json \
  --source results/capacity-selection-staging \
  --output results/capacity-selection
uv run stc results validate \
  --calibration-readiness manifests/stc/capacity-selection-readiness.json \
  --manifest manifests/stc/capacity-selection-cells.jsonl \
  --cohort-index manifests/stc/capacity-selection-cohort-index.json \
  --results results/capacity-selection \
  --output manifests/stc/capacity-selection-result-validation.json
uv run stc selection freeze-capacity \
  --plan manifests/stc/capacity-selection-plan.json \
  --results results/capacity-selection \
  --validation-report manifests/stc/capacity-selection-result-validation.json \
  --output manifests/stc/capacity-selection-bundle-G2-CAP.json
# Phase-B dependency barrier: do not run the following capacity-power commands
# until pilot summarize below has frozen and validated B_ref.
uv run stc calibration materialize-capacity-power \
  --config configs/stc/design/calibration.yaml \
  --coverage-config configs/stc/design/coverage-scaling.yaml \
  --information-config configs/stc/design/parametric-information.yaml \
  --scaling-config configs/stc/design/scaling.yaml \
  --hardware-config configs/stc/design/scaling-hardware.yaml \
  --estimator-config configs/stc/design/capacity-estimators.yaml \
  --capacity-stakes manifests/stc/deployment-stakes-G2-CAP.json \
  --core-selection-bundle manifests/stc/selection-bundle-G2.json \
  --capacity-selection-bundle manifests/stc/capacity-selection-bundle-G2-CAP.json \
  --b-ref-summary manifests/stc/blinded-pilot-summary.json \
  --b-ref-validation manifests/stc/pilot-result-validation.json \
  --selection-cohort-index manifests/stc/capacity-selection-cohort-index.json \
  --plan-output manifests/stc/capacity-power-calibration-plan.json \
  --output manifests/stc/capacity-power-calibration-cells.jsonl \
  --cohort-index-output manifests/stc/capacity-power-calibration-cohort-index.json \
  --budget-output manifests/stc/capacity-power-calibration-budget.json
uv run stc calibration authorize \
  --predecessor manifests/gates/G1.json \
  --plan manifests/stc/capacity-power-calibration-plan.json \
  --manifest manifests/stc/capacity-power-calibration-cells.jsonl \
  --cohort-index manifests/stc/capacity-power-calibration-cohort-index.json \
  --budget manifests/stc/capacity-power-calibration-budget.json \
  --development-budget manifests/stc/design-development-budget.json \
  --generator-property-report manifests/stc/generator-property-report.json \
  --environment configs/stc/design/calibration-environment.yaml \
  --scope capacity-power \
  --coverage-config configs/stc/design/coverage-scaling.yaml \
  --information-config configs/stc/design/parametric-information.yaml \
  --scaling-config configs/stc/design/scaling.yaml \
  --hardware-config configs/stc/design/scaling-hardware.yaml \
  --capacity-stakes manifests/stc/deployment-stakes-G2-CAP.json \
  --estimator-config configs/stc/design/capacity-estimators.yaml \
  --core-selection-bundle manifests/stc/selection-bundle-G2.json \
  --capacity-selection-bundle manifests/stc/capacity-selection-bundle-G2-CAP.json \
  --b-ref-summary manifests/stc/blinded-pilot-summary.json \
  --b-ref-validation manifests/stc/pilot-result-validation.json \
  --disjoint-from manifests/stc/capacity-selection-cohort-index.json \
  --output manifests/stc/capacity-power-calibration-readiness.json
uv run stc runtime run-matrix \
  --mode capacity-calibration \
  --calibration-role power \
  --manifest manifests/stc/capacity-power-calibration-cells.jsonl \
  --cohort-index manifests/stc/capacity-power-calibration-cohort-index.json \
  --budget manifests/stc/capacity-power-calibration-budget.json \
  --calibration-readiness manifests/stc/capacity-power-calibration-readiness.json \
  --core-selection-bundle manifests/stc/selection-bundle-G2.json \
  --selection-bundle manifests/stc/capacity-selection-bundle-G2-CAP.json \
  --b-ref-summary manifests/stc/blinded-pilot-summary.json \
  --b-ref-validation manifests/stc/pilot-result-validation.json \
  --output results/capacity-power-calibration-staging
uv run stc runtime run-matrix \
  --mode capacity-calibration \
  --calibration-role power \
  --manifest manifests/stc/capacity-power-calibration-cells.jsonl \
  --cohort-index manifests/stc/capacity-power-calibration-cohort-index.json \
  --budget manifests/stc/capacity-power-calibration-budget.json \
  --calibration-readiness manifests/stc/capacity-power-calibration-readiness.json \
  --core-selection-bundle manifests/stc/selection-bundle-G2.json \
  --selection-bundle manifests/stc/capacity-selection-bundle-G2-CAP.json \
  --b-ref-summary manifests/stc/blinded-pilot-summary.json \
  --b-ref-validation manifests/stc/pilot-result-validation.json \
  --verify-only \
  --receipt manifests/systems/capacity-power-calibration-receipt.json \
  --output results/capacity-power-calibration-staging
uv run stc runtime import-matrix \
  --receipt manifests/systems/capacity-power-calibration-receipt.json \
  --source results/capacity-power-calibration-staging \
  --output results/capacity-power-calibration
uv run stc results validate \
  --calibration-readiness manifests/stc/capacity-power-calibration-readiness.json \
  --manifest manifests/stc/capacity-power-calibration-cells.jsonl \
  --cohort-index manifests/stc/capacity-power-calibration-cohort-index.json \
  --core-selection-bundle manifests/stc/selection-bundle-G2.json \
  --selection-bundle manifests/stc/capacity-selection-bundle-G2-CAP.json \
  --b-ref-summary manifests/stc/blinded-pilot-summary.json \
  --b-ref-validation manifests/stc/pilot-result-validation.json \
  --disjoint-from manifests/stc/capacity-selection-cohort-index.json \
  --results results/capacity-power-calibration \
  --output manifests/stc/capacity-power-calibration-result-validation.json
uv run stc capacity calibrate \
  --plan manifests/stc/capacity-power-calibration-plan.json \
  --results results/capacity-power-calibration \
  --validation-report manifests/stc/capacity-power-calibration-result-validation.json \
  --core-selection-bundle manifests/stc/selection-bundle-G2.json \
  --selection-bundle manifests/stc/capacity-selection-bundle-G2-CAP.json \
  --b-ref-summary manifests/stc/blinded-pilot-summary.json \
  --b-ref-validation manifests/stc/pilot-result-validation.json \
  --selection-cohort-index manifests/stc/capacity-selection-cohort-index.json \
  --power-cohort-index manifests/stc/capacity-power-calibration-cohort-index.json \
  --output manifests/stc/capacity-calibration-summary.json
```

`selection-bundle-G2.json` binds the core hybrid weights, static
proportions/cadence, single-medium hyperparameters/early-stop state, comparator
baseline table, and all input/search/output edges.
`capacity-selection-bundle-G2-CAP.json` separately binds the coverage
metric/codec/policy, information decoder/critic, and their complete search
edges. A missing/failing capacity bundle cannot fail the core bundle.
`capacity-calibration-summary.json` is computed only after that bundle exists
and only from the disjoint power-calibration cohort. It contains coverage
residual/covariance terms, bitwise-channel precision and fresh-retrain cost,
and the scaling retention-binomial/overdispersion, paired-user,
workload-family, finite-knee/separation/failure, resource-dependent residual,
and eight-panel cost components. Every component carries its sample count,
source cells, point estimate, and uncertainty. Resource reservations use one
program-wide max-stat simultaneous upper prediction envelope over every
track×cohort×native-resource cell with frozen familywise overrun
\(\alpha_{\rm budget}=0.01\), not marginal 95% bounds. Power includes the
resulting truncation/failure probability.

Capacity power uses a separate pre-CAL full-object nuisance-set contract with
`alpha_Sigma_cap=0.01`; it may not reuse a favorable point covariance from
this summary. `capacity-estimators.yaml`, before either capacity CAL cohort,
byte-freezes the block/user hierarchy, bounded supports and marginal-rate
envelopes, covariance-entry studentizers, diagonal shrinkage, simultaneous
max-entry band, PSD projection plus nonshrinking diagonal inflation, and a
cluster-level distribution family containing pilot-cluster resampling,
heteroskedastic/skewed bounded marginals, beta-binomial rare events,
zero-inflated costs, and Gaussian/t-copula dependence perturbations. A
pre-CAL nomination simulation may screen builder/sample-size candidates, but
only a disjoint fixed-size verification stream may authorize the selected
builder; its lower confidence bound is simultaneous over the entire
candidate×stress-grid object. The selected set must cover the complete
covariance-and-distribution object with familywise probability at least 0.99.
Choosing the narrowest builder, shrinkage, support, or copula after CAL fails
the digest.

Scaling, coverage, and information power minimize composite success over this
full nuisance set, not merely over matrices with the same covariance. Their
common search/selected-verification Monte Carlo uses
`alpha_MC=0.01`; the recorded design-assurance union bound is therefore at most
`alpha_Sigma_cap + alpha_MC = 0.02`, separate from scientific type-I alpha.
Correlated heteroskedastic and same-covariance/different-tail mutations must
enlarge or preserve powered \(n\), and a boundary fixture must false-authorize
any candidate/builder with probability at most the recorded union bound.
Mutation tests inject TEST or derivative-overlap
paths, omit the receipt or losing trial, change a tie-break/search budget,
overrun or borrow the development budget, drift code/environment after
execution, hand-author a selected config without its search log, or try to
freeze from an unvalidated result. Shared unit/seed/derivative closure between
the capacity cohorts, computing the summary before selection freeze, including
a selection-cohort outcome in the summary, omitting any scaling variance or
panel component, imputing an unseen operation/load/resource from the generic
pilot, or providing only a point cost all fail before power or G2-CAP.

- [ ] **Step 1: Freeze the CALIBRATION comparator maps**

`comparator-selection.yaml` fixes the five eligible baselines, the three
public \(\phi\) values, exact `delta_CAL=0.01`, equal weights over
CALIBRATION workload family and \(R\), native-cost aggregation, and canonical
method-ID tie breaking. For each \(\phi\), the benefit map selects the highest
pooled CALIBRATION utility. The efficiency map first retains every method
within 0.01 of that same maximum, then selects the lowest complete
`lifecycle_cost_usd`; an exact cost tie uses canonical method ID. A missing
price or incomplete resource ledger makes only the efficiency entry
`INADMISSIBLE_MISSING_PRICE`, never silently free.

The output schema permits keys only by public \(\phi\), never by \(R\), user,
payload seed, or TEST family. It records every eligible method's pooled
utility/cost, source and split digests, equal-weight table, selected method
IDs, tie sets, exclusion reasons, `delta_CAL`, cost-profile digest, selection
config/code digests, and a self-digest. `calibration-baseline-summary.json`
additionally publishes bootstrap selection probabilities and
leave-one-CALIBRATION-family-out identities as stability diagnostics; they
cannot change the deterministic realized map or select from TEST.

Tests require that row-order permutations and balanced renamings/permutations
of \(R\) and CALIBRATION-family labels leave the map byte-identical, while
utility, price, eligibility, tie, split, code, or source-digest mutations
change or invalidate it. A TEST path is rejected at the parser boundary;
mutating any TEST fixture cannot affect the map. The same map digest is then
required by power, finalization, preregistration, G2, and final inference.

```bash
uv run stc design freeze-comparators \
  --config configs/stc/design/comparator-selection.yaml \
  --calibration-results results/training-calibration-core/baselines.jsonl \
  --selection-bundle manifests/stc/selection-bundle-G2.json \
  --validation-report manifests/stc/training-calibration-core-result-validation.json \
  --cost-profile configs/stc/design/cost-profile.yaml \
  --summary-output manifests/stc/calibration-baseline-summary.json \
  --output manifests/stc/comparator-maps-G2.json
uv run pytest tests/stc/design/test_comparators.py \
  tests/stc/test_cli_contract.py -q
```

- [ ] **Step 2: Write power-selection tests**

Simulate candidate user counts 64 through 512 in steps of 16, with 10,000 paired
Monte Carlo experiments per candidate and the exact Task 10 estimands and
hierarchical analysis. A chosen count means user streams per logical row; paired
seed family is already one of the 297 logical factors and is not multiplied a
second time. Because every candidate is divisible by sixteen, allocate it
equally across the 16 frozen TEST workload families. The same user stream is
paired across every method/regime contrast for its seed family. The run index
must prove at least four genuinely independent physical users in every
workload-family×paired-seed-family block. The three seed families remain fixed
equal-weight blocks and are never pooled into pseudoreplicates; resample users
within each block while preserving complete logical-row covariance. The three
seed families are a fixed finite target, not three random PSUs: a common
within-family shift changes the fixed-mixture point estimate and
leave-one-seed-family-out sensitivity but does not create a seed-superpopulation
variance term. Aliased or duplicated physical streams fail the four-user floor.
Repeated probes or
cycles never count as users. A
contrast scoped inside one seed stratum is ineligible unless its own
independent-user floor is separately powered. Select the
maximum count needed by:

- PAPER-C1 one-point material oracle-heterogeneity value in at least one of
  exactly nine reuse×pressure cells, where the powered cell succeeds only when
  all four direct component LCBs clear `delta_C1`;
- PAPER-C2 two-point benefit or one-point noninferiority plus 10% cost branch;
- PAPER-C3 two-corner material sign reversal for either adjacent-medium pair,
  using the exact low-corner external Boolean OR, high-corner external Boolean
  AND, and two-component latent/LoRA rule rather than an envelope max;
- PAPER-C4 operator effect, identity deferral TOST/cost, and marginal
  interaction TOST; and
- every ordered fail-only companion required by the C2, C3, and C4 Boolean
  support rules, powered at the interior \(\Delta F=0\) alternative while the
  boundary \(\Delta F=\delta_{\mathrm{RF}}\) is used only for type-I/coverage
  calibration.

Use one preregistered search Monte Carlo stream to identify the first candidate
that appears to pass, then discard that stream for authorization. Re-evaluate
only the selected candidate with a disjoint, fixed-size, high-precision
verification Monte Carlo stream and compute simultaneous branch bounds over
the whole candidate×branch search family, or use an equivalent preregistered
monotone alpha-spending rule. The selected count is authorized only when the
verification familywise 99% lower bound on every required branch's
composite-success probability strictly exceeds 0.90. Reusing search draws,
reporting pointwise Clopper--Pearson bounds after scanning candidates, or
continuing verification until a pass is forbidden. A mutation with many
true-power-0.90 candidates must have conditional familywise false authorization
at most 0.01. If no candidate through 512 passes, write
`power_status=FAIL_MAX_N`, leave both confirmatory and analysis plans
unfinalized, and fail G2; increasing the range is a new preregistration, not an
implicit fallback.

The selection-binding shared-infrastructure missing rate is exactly `0`.
The `0.025` scenario is reported as a nonbinding MNAR stress result, and the
`0.05` boundary is a completion-gate calibration case, not a power alternative.
A zero-noise fixture with 2.5% adversarially missing blocks must narrow the
small-effect alternatives rather than falsely pass. If the blinded pilot
upper-95 bound is above `0.05`, write
`power_status=FAIL_INFRASTRUCTURE_COMPLETION_FEASIBILITY` before the candidate
search; larger \(n\) cannot rescue it.

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
`1 - mean(C_hybrid) / mean(C_G2_frozen_efficiency_map)`. For PAPER-C4 identity,
it is `1 - cell_balanced_mean(C_deferred) /
cell_balanced_mean(C_inline)`. Denominators must be positive. The price profile,
functional, cell weights, and denominators are frozen before G2 and reused
unchanged in power and inference; the full native resource vector is always
reported beside the USD contrast.

- [ ] **Step 3: Implement deterministic simulation**

Record pilot digest, covariance model, seed, simulation count, candidate grid,
contrast-specific power and uncertainty, sensitivity grid, chosen count, and
no-optional-stopping rule. The power engine imports `auc_128`,
`paper_c2_contrasts`, `paper_c3_sign_reversal`, `paper_c4_contrasts`, and the
hierarchical resampling configuration from `analysis/estimands.py`, and imports
the exact ordered `PAPER-C-DEPLOYABILITY-GLOBAL-131` table and critical-value function from
`analysis-plan-skeleton.json`. It imports—never reconstructs—the frozen
`comparator-maps-G2.json` digest and applies those identities in every C2 TEST
replicate. Conditional power uses the realized frozen map. A mandatory
end-to-end sensitivity additionally resimulates the frozen CALIBRATION
selection algorithm jointly with TEST and reports map-selection probabilities;
the selected \(n\) is the maximum required by the conditional and end-to-end
branches. No simulated or observed TEST value may select a comparator.
Tests mutate each shared estimand function,
permute/add/drop one global-family member, and change the global 95% critical
quantile rule, comparator-map identity, map digest, or selection-code digest;
both power and inference must reject the changed implementation digest.

The test fixture verifies code paths only. Before production power selection,
run the separately budgeted pilot from training users and export a blinded
summary with variance/covariance and failure-rate information by opaque
`variance_stratum_id`, but no semantic condition labels or held-out test
outcomes. The mapping from each precommitted design stratum to its opaque ID is
sealed before pilot execution and only its digest enters deployment stakes.
Power uses the worst admissible joint covariance over all anonymous strata,
so a rare high-variance cell cannot be diluted by pooled averages; an optional
trusted join may only reproduce that precommitted mapping and may not lower the
worst-case \(n\). A mutation with one high-variance stratum among many quiet
strata must increase or preserve selected \(n\), never average it away. The
summary also computes `B_ref` as
the preregistered 95th percentile across pilot/training users of peak serialized
bytes for the raw full-valid-history store over 128 cycles. The summary records
the quantile estimator, user inclusion rule, serialized-byte measurement
version, source run digests, estimate, and uncertainty. No held-out user
contributes.
This generic pilot supports PAPER-C power and `B_ref` only. Coverage estimator
error, information decoder/substrate cost, and eight-panel scaling cost must
come from the receipt-validated `capacity-calibration-summary.json`; cross-track
imputation is forbidden.

```bash
uv run stc pilot run \
  --config configs/stc/design/power.yaml \
  --calibration-readiness manifests/stc/training-calibration-core-readiness.json \
  --selection-bundle manifests/stc/selection-bundle-G2.json \
  --development-budget manifests/stc/design-development-budget.json \
  --receipt manifests/systems/pilot-receipt.json \
  --output results/pilot
uv run stc pilot summarize \
  --input results/pilot \
  --receipt manifests/systems/pilot-receipt.json \
  --selection-bundle manifests/stc/selection-bundle-G2.json \
  --output manifests/stc/blinded-pilot-summary.json \
  --blind-condition-labels
uv run stc pilot validate \
  --receipt manifests/systems/pilot-receipt.json \
  --selection-bundle manifests/stc/selection-bundle-G2.json \
  --summary manifests/stc/blinded-pilot-summary.json \
  --output manifests/stc/pilot-result-validation.json
```

- [ ] **Step 4: Verify and commit**

```bash
uv run pytest tests/stc/design/test_comparators.py \
  tests/stc/design/test_power.py \
  tests/stc/analysis/test_scaling.py \
  tests/stc/analysis/test_scaling_materialize.py \
  tests/stc/analysis/test_coverage_scaling.py \
  tests/stc/analysis/test_parametric_information.py \
  tests/stc/test_cli_contract.py -q
uv run stc design power \
  --config configs/stc/design/power.yaml \
  --cost-profile configs/stc/design/cost-profile.yaml \
  --comparator-maps manifests/stc/comparator-maps-G2.json \
  --analysis manifests/stc/analysis-plan-skeleton.json \
  --deployment-stakes manifests/stc/deployment-stakes-core.json \
  --pilot manifests/stc/blinded-pilot-summary.json \
  --pilot-validation manifests/stc/pilot-result-validation.json \
  --output manifests/stc/power.json
uv run stc design finalize \
  --skeleton manifests/stc/confirmatory-skeleton.jsonl \
  --analysis-skeleton manifests/stc/analysis-plan-skeleton.json \
  --selection-bundle manifests/stc/selection-bundle-G2.json \
  --pilot-receipt manifests/systems/pilot-receipt.json \
  --pilot-validation manifests/stc/pilot-result-validation.json \
  --comparator-maps manifests/stc/comparator-maps-G2.json \
  --power manifests/stc/power.json \
  --output manifests/stc/confirmatory.jsonl \
  --run-index-output manifests/stc/confirmatory-run-index.json \
  --summary-output manifests/stc/confirmatory-summary.json \
  --analysis-output manifests/stc/analysis-plan.json
uv run stc design budget-freeze \
  --program-envelope configs/stc/design/program-budget-envelope.yaml \
  --rre-reference configs/stc/design/rre-reference-native.yaml \
  --coarse-feasibility manifests/stc/capacity-coarse-feasibility.json \
  --power manifests/stc/power.json \
  --confirmatory-summary manifests/stc/confirmatory-summary.json \
  --run-index manifests/stc/confirmatory-run-index.json \
  --output manifests/stc/confirmatory-budget.json
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: implement blinded confirmatory power design"
```

The chosen count \(n\) is intentionally unknown until this task runs; no
arbitrary number is inserted in the plan. Finalization refuses any nonpassing
power manifest. `design budget-freeze` computes one simultaneous upper-99
native-resource reservation for all \(297n\) primary children and a distinct
all-\(297n\) clean-reproduction reservation; both must fit
`Q_core,primary`/`Q_core,clean` and every absolute native cap without
borrowing. It never samples outcomes or reduces \(n\). Finalization must
report:

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
- Consume read-only: `manifests/stc/comparator-maps-G2.json`
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
  --comparator-maps PATH --output PATH`. The analyzer cannot open the result root until the supplied
  report is schema-valid, `PASS`, and digest-matched.

- [ ] **Step 1: Write paired-resampling tests**

Recompute the Task 10 equal-cycle/equal-probe AUC, hierarchical
workload-family→paired-user bootstrap, continuous-row cluster studentization,
RF paired-boundary scores, 20,000 replicates, the one ordered
`PAPER-C-DEPLOYABILITY-GLOBAL-131` joint max-stat family, missingness
rules, and seed-family sensitivity by importing—not copying—
`analysis/estimands.py` and the finalized analysis plan. Recompute the 36 C1
and six C3 direct component margins inside every replicate, but apply the
G2-frozen C2 comparator maps without TEST selection. Tests reject a TEST
max/argmax, a different function digest, cycle-level resampling,
TEST-family tuning, unpaired method rows, empty probes, silent infeasibility,
or a locally reconstructed membership/critical-value table. They also mutate
the map bytes, self-digest, selected identity, \(\phi\)-only shape, selection
code digest, CALIBRATION source digest, and G2/preregistration binding; every
mutation fails before outcomes are read.

Every claim output additionally emits a complete-case estimate/interval that
excludes scored `RESOURCE_FAILURE` rows, labeled
`SENSITIVITY_COMPLETE_CASE_ONLY`. Tests require it beside the ITT estimate and
prove that changing, omitting, or reversing the complete-case result cannot
create, rescue, or overturn a support decision. OOM/cgroup kills are explicitly
mutated between `RESOURCE_FAILURE` and shared missingness; only the former is
accepted.

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
unrestricted clairvoyant oracle against each of the four medium-restricted
clairvoyant counterfactuals generated with the same future sidecar, caps, and
forgetting option. Estimate all 36 component gaps \(G_{cj}\) inside each paired
whole-user bootstrap replicate; never bootstrap the max/argmax envelope. Read
their adjusted intervals from the shared maximum across all 131
`PAPER-C-DEPLOYABILITY-GLOBAL-131` members; the 36 rows never receive a claim-specific
maximum. For each cell derive the envelope gap
\(D_c=\min_jG_{cj}\) and its simultaneous interval
\([\min_jL_{cj},\min_jU_{cj}]\). This isolates strict cross-medium
heterogeneity from the separate value of future-aware admission/forgetting
without invalid selection at tied media.

Freeze `delta_C1=0.01` before power. `PAPER-C1=SUPPORTED` only if at least one
of the exactly nine cells has all four component LCBs strictly above
`delta_C1`, equivalently its derived \(D_c\) LCB is above the margin. If every
cell has derived \(D_c\) UCB below `delta_C1`, return
`FALSIFIED_NO_MATERIAL_HETEROGENEITY`; otherwise return
`NARROWED_INSUFFICIENT_PRECISION`. Report all 36 component and nine derived
estimates; never use oracle weak dominance as support. Any unresolved or
nonoptimal restricted/unrestricted oracle required by a cell returns
`NARROWED_ORACLE_UNRESOLVED`.

- [ ] **Step 3: Implement PAPER-C2**

Use the 1/9 equal-cell R×phi estimand and import two comparator maps frozen on
CALIBRATION before G2. For each public \(\phi\), the benefit map is the
highest-utility baseline under frozen pooled CALIBRATION family/\(R\) weights;
the efficiency map is the lowest-cost member within
`delta_CAL=0.01` of that CALIBRATION maximum. Exact ties use canonical method
ID. Each map is constant across the three TEST \(R\) cells and every TEST
family. Within every bootstrap replicate, apply those fixed identities and
carry their paired utility/cost; never select a comparator from TEST. The
benefit, noninferiority, and cost boundaries are the three C2 members of
the 104-row scientific subtable and use the 131-row union critical value.
Support only if either:

1. the adjusted 95% LCB of `hybrid utility gain - 0.02` is above zero; or
2. the adjusted 95% LCB of `hybrid utility gain + 0.01` and the adjusted 95%
   LCB of `lifecycle-cost reduction - 0.10` are both above zero.

A point gain of 0.021 with interval `[0.001,0.041]` against zero is not
material-benefit support because its margin interval crosses zero. The same
rule applies to cost materiality.

Both branches must also pass the separately frozen failure-rate
noninferiority veto. Publish all five TEST baseline comparisons and CAL map
selection diagnostics, but neither a TEST envelope nor secondary hypervolume
can overturn a failed primary decision. Freeze this exhaustive
priority-ordered reason table:

```text
SUPPORTED_BENEFIT
SUPPORTED_EFFICIENCY
FALSIFIED_UTILITY_HARM
  := benefit UCB < 0.02 and utility-noninferiority UCB < -0.01
FALSIFIED_NO_EFFICIENT_BRANCH
  := benefit UCB < 0.02 and cost-reduction UCB < 0.10
NARROWED_BENEFIT_MARGIN_UNPROVEN
NARROWED_COST_MARGIN_UNPROVEN
NARROWED_INSUFFICIENT_PRECISION
NARROWED_SEED_SENSITIVITY
NARROWED_RESOURCE_FAILURE_NONINFERIORITY
NARROWED_MISSING_OR_INFEASIBLE
```

Each machine reason carries `decision_class=FALSIFIED|NARROWED` but maps to one
of the two canonical evidence statuses `SUPPORTED` or
`FALSIFIED/NARROWED`; the evidence registry never invents separate release
statuses for the two non-support classes.

Every C2 result block, including a null/narrowed branch, also reports cellwise
and 1/9 equal-cell `oracle_regret = U_unrestricted_oracle - U_hybrid` with its
paired simultaneous interval. If the optional capture fraction is reported,
use the frozen benefit-map baseline and recompute only the oracle and
denominator arithmetic inside every bootstrap replicate. Regret and capture
diagnose how much past-only routing leaves on the table; neither changes the
fixed C2 support predicate.

- [ ] **Step 4: Implement PAPER-C3**

PAPER-C3 is explicitly a matched-cap utility-ranking reversal, not an
economic-optimality or full-lifecycle Pareto-crossover claim. Report complete
cost/bytes/latency/governance beside it and treat a governance/feasibility veto
as ineligible, but do not imply that equal caps equal realized lifecycle cost.

Do not select an external comparator inside TEST. Before G2, freeze one
low-promotion and one high-promotion corner from H-STC-002 and a materiality margin
`delta_crossover=0.01` from the pre-data deployment tolerance. Pilot variance
affects power only, not the materiality margin. TEST users and TEST workload
families play no role in corner or margin selection.

Bind the exact deployable method IDs `raw_external`, `compressed_external`,
`latent_kv`, and `user_lora`; reject C1's medium-restricted clairvoyant method
IDs. For `(external-envelope, latent/KV)`, compute four oriented effects: at
the low corner `raw-latent` and `compressed-latent`; at the high corner
`latent-raw` and `latent-compressed`. For `(latent/KV, user-LoRA)`, compute
low `latent-LoRA` and high `LoRA-latent`. Each of the six
104-row scientific-subtable members is the oriented effect minus
`delta_crossover=0.01`, with its adjusted interval tested against zero. The
low external condition passes when at least one of its two margin LCBs exceeds
zero; the high condition passes only when both do. The direct pair requires
both margin LCBs above zero. At least one pair must pass together with stable
leave-one-seed-family-out direction; no TEST max/argmax is bootstrapped.
Each utility component is eligible in the corresponding Boolean only if its
same-weight fail-only companion UCB is at most
`delta_RF=0.01`; a failed or absent companion returns
`NARROWED_RESOURCE_FAILURE_NONINFERIORITY` before precision or seed
sensitivity reasons.

For reporting, derive the low external-envelope effect interval as
\([\max(L_{\rm raw-lat},L_{\rm comp-lat}),
\max(U_{\rm raw-lat},U_{\rm comp-lat})]\) and the high
latent-minus-envelope interval as
\([\min(L_{\rm lat-raw},L_{\rm lat-comp}),
\min(U_{\rm lat-raw},U_{\rm lat-comp})]\). Compute each reversal span directly
inside paired bootstrap replicates so covariance is retained. The external
pair is materially falsified when both low-corner margin UCBs are below zero
or either required high-corner margin UCB is below zero. The direct pair is
materially falsified when either margin UCB is below zero. If both pairs are
falsified, return
`FALSIFIED_PREDICTED_REVERSAL`. Otherwise return
`NARROWED_SIGN_CHANGE_BELOW_MARGIN` when at least one pair satisfies its
Boolean directional rule at the point-estimate level (\(g>0\)) but no pair
satisfies the same Boolean rule with every required point
\(g\ge\delta_{\mathrm{cross}}\). If a pair satisfies the material
point-estimate rule but a required adjusted margin interval does not clear
zero, return `NARROWED_INSUFFICIENT_PRECISION`; return
`NARROWED_SEED_SENSITIVITY` when only the frozen sensitivity fails. A pair
with any structurally
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
the 104-row scientific subtable; all use the global 131-member union critical
value. The positive
branch requires all three:

1. at least one of the nine `O_main(r)` or 18 `O_schedule(r)` effects has an
   adjusted 95% LCB for `O - 0.02` above zero;
2. the simultaneous two-sided 95% interval for the 1/9 cell-balanced
   `S_identity` lies wholly in `[-0.01,+0.01]`, and the adjusted 95% LCB for
   `lifecycle-cost reduction - 0.10` is above zero;
3. the simultaneous two-sided 95% interval for the 1/9 cell-balanced
   interaction lies wholly in `[-0.01,+0.01]`.

Both signed equivalence boundaries are members of the ordered global family
and use one centered signed max-statistic calibrated at the 95% familywise
level. Boundary-null simulations across every rejection-capable equivalence
row must keep the probability of any false equivalence declaration at most
0.05; an ordinary pointwise or adjusted-90% TOST interval is insufficient.

Always publish cellwise interactions and intervals. An interval overlapping
the equivalence margin and null is `NARROWED_INSUFFICIENT_PRECISION`, never
evidence of equivalence. A significant interaction cannot support
separability. Freeze the exhaustive priority-ordered non-support reasons:

```text
FALSIFIED_NO_OPERATOR_EFFECT
  := every eligible O_main(r) and O_schedule(r) simultaneous UCB is below 0.02
NARROWED_GAIN_BELOW_0_02
NARROWED_IDENTITY_SCHEDULE_OUTSIDE_OR_UNPROVEN_EQUIVALENCE
NARROWED_NO_DEFERRED_COST_BENEFIT
NARROWED_SCHEDULE_CONTINGENT_OR_INTERACTION_INCONCLUSIVE
NARROWED_EQUIVALENCE_IMPRECISE
NARROWED_SEED_SENSITIVITY
NARROWED_RESOURCE_FAILURE_NONINFERIORITY
NARROWED_MISSING_OR_INFEASIBLE
```

Only `FALSIFIED_NO_OPERATOR_EFFECT` maps to
`decision_class=FALSIFIED`. Every other missed support condition maps to
`decision_class=NARROWED`, including a confidently nonequivalent schedule
effect, a cost-margin LCB at or below zero, or nonseparable interaction. An
operator interval such as `[-0.05,0.08]` is insufficient precision and can
never be labeled falsification. The recorded
decision includes every satisfied/failed predicate even though the canonical
reason uses this deterministic priority. The exact operator simple component
used by condition 1 must pass
`RF-C4-OP-{cell_id}-{inline|deferred}`; an `O_main` component requires both
schedule companions for that cell. Condition 2 must pass
`RF-C4-IDENTITY-DEFERRED-COST`; a different cell/schedule failure row cannot
substitute.

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
  --comparator-maps manifests/stc/comparator-maps-G2.json \
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
- Create: `src/stc_research/preregistration.py`
- Create: `tests/stc/analysis/test_capacity_budget.py`
- Create: `tests/stc/test_preregistration.py`
- Create: `manifests/stc/theory-laws-G2.json`
- Create: `manifests/stc/theory-laws-G2-CAP.json`
- Create: `manifests/stc/scaling-cells.jsonl`
- Create: `manifests/stc/scaling-cohort-index.json`
- Create: `manifests/stc/scaling-power.json`
- Create: `manifests/stc/scaling-budget.json`
- Create: `manifests/stc/coverage-cardinality-cells.jsonl`
- Create: `manifests/stc/coverage-fixed-bits-cells.jsonl`
- Create: `manifests/stc/coverage-scaling-manifest-index.json`
- Create: `manifests/stc/coverage-scaling-cohort-index.json`
- Create: `manifests/stc/coverage-power.json`
- Create: `manifests/stc/coverage-scaling-budget.json`
- Create: `manifests/stc/parametric-information-cells.jsonl`
- Create: `manifests/stc/parametric-information-cohort-index.json`
- Create: `manifests/stc/parametric-information-power.json`
- Create: `manifests/stc/parametric-information-budget.json`
- Create: `manifests/stc/capacity-aggregate-budget.json`
- Create: `manifests/stc/scaling-summary.json`
- Create: `manifests/stc/coverage-scaling-summary.json`
- Create: `manifests/stc/parametric-information-summary.json`
- Create: `manifests/stc/scaling-fit-result-validation.json`
- Create: `manifests/stc/scaling-holdout-result-validation.json`
- Create: `manifests/stc/scaling-provisional.json`
- Create: `manifests/stc/scaling-holdout-unlock.json`
- Create: `manifests/stc/coverage-cardinality-fit-result-validation.json`
- Create: `manifests/stc/coverage-cardinality-holdout-result-validation.json`
- Create: `manifests/stc/coverage-fixed-bits-fit-result-validation.json`
- Create: `manifests/stc/coverage-fixed-bits-holdout-result-validation.json`
- Create: `manifests/stc/coverage-provisional.json`
- Create: `manifests/stc/coverage-holdout-unlock.json`
- Create: `manifests/stc/parametric-information-result-validation.json`
- Create: `manifests/stc/preregistration.json`
- Create: `manifests/stc/capacity-preregistration.json`
- Create: `manifests/gate-inputs/G2.json`
- Create: `manifests/gates/G2.json`
- Create: `manifests/capacity-gate-inputs/G2-CAP.json`
- Create: `manifests/capacity-gates/G2-CAP.json`
- Create: `tests/stc/test_g2_gate.py`
- Create: `tests/stc/test_g2_cap_gate.py`
- Create: `tests/stc/fixtures/scaling-invalid.jsonl`
- Create: `tests/stc/fixtures/scaling-confirmatory-three-point.jsonl`
- Create: `tests/stc/fixtures/scaling-a100-fit.jsonl`
- Create: `tests/stc/fixtures/scaling-a100-holdout.jsonl`
- Modify: `src/stc_research/cli.py`
- Modify: `tests/stc/test_cli_contract.py`

**Interfaces:**
- Produces: `a100_regime_scaling_law` only when every A100 validity condition
  passes; otherwise `empirical_curve`. An incomplete pre-holdout fit is
  explicitly
  `empirical_curve_provisional`.
- Produces: a 120-logical-cell scaling manifest, exact
  logical→physical/cohort index, prospective budget plus append-only actual
  budget-ledger contract, and a
  typed G2-CAP domain-child input/record pair through the capacity-gate
  evaluator; this task never hand-writes a gate record.
- Produces: two separately identified coverage manifests (fixed codec versus
  fixed total bits) and one fresh-retraining parametric-information manifest.
  Each has a powered child/cohort index, prospective hard budget, append-only
  actual-budget ledger contract, post-G4 execution/receipt path, and fail-closed
  result validation. They are G2-CAP-bound capacity tracks, never silently folded
  into the 120-cell resource-surface fit.
- Produces: one shared capacity-budget envelope whose three positive,
  non-borrowable track quotas and projected upper bounds sum to at most the
  pre-data sponsor ceiling \(Q_{\rm capacity,max}\),
  plus one append-only global actual ledger. No track owns an independent
  copy of that allowance.
- Produces: core `stc prereg freeze`/G2 for PAPER-C and a separate typed
  `stc prereg freeze-capacity`/`G2-CAP` child gate for scaling, coverage, and
  parametric information. `G2-CAP` has predecessor PASS G2. A capacity power or
  budget failure blocks only capacity execution/claims; it never blocks or
  rescues PAPER-C. Both validators re-resolve every path/digest edge before
  their respective execution mode.
  `G2-CAP` is a typed domain child in `manifests/capacity-gates`, not a new ID in
  the canonical linear G0–G8 `GateRecord` engine; its evaluator nevertheless
  binds and rederives the canonical PASS G2 record as its predecessor.

- [ ] **Step 1: Freeze the dense scale axes**

For static mixture and hybrid only, use:

```text
{1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8}
```

Fit on the first eight scales and reserve the largest two as extrapolation
holdouts. The configuration declares which physical quantity the normalized
factor multiplies.

Freeze four **assigned treatment caps** in the configuration before G2-CAP:
active-store byte cap \(B_{\rm active,cap}\), live-token application cap
\(T_{\rm live,cap}\), sleep-compute cap \(F_{\rm sleep,cap}\), and serialized
latent/adapter representation cap \(b_{\rm repr,cap}\) in bits. Adapter
parameters and latent slots are converted through their frozen precision and
serialization schema; counts, bytes, and bits are never added directly. Randomize and fit
on those assigned caps under intention-to-treat. Realized resident bytes,
tokens, FLOPs, and representation bits are metered outcomes/resource charges,
never substituted as regressors. The manifest freezes a prescribed schedule or
utilization tolerance for each cap; under-utilization is reported and may
narrow feasibility, but post-treatment actual work cannot move a row along an
axis. Materialize:

```text
one-axis cells        = 4 axes × 10 factors × 2 methods = 80
joint-axis fit cells  = 16 frozen D-optimal log-resource tuples × 2 methods = 32
joint-axis holdouts   = 4 frozen maximin tuples × 2 methods = 8
candidate total       = 120
```

Choose the 16 joint tuples deterministically from the Cartesian product of the
**first eight fit factors only**. Use D-optimality on the exact fitted
10-column four-main-effect+six-pairwise-interaction normalized log-resource
matrix. There is no intercept because \(g_m(r_0)=0\); a selector with an
11-column intercept targets a different estimator and fails its digest.
Optimize the **combined physical-execution information** already supplied by
the fixed OAT rows. Before scoring a candidate, apply the frozen
logical→physical incidence map and either collapse perfect aliases or compute
the equivalent GLS information \(X^\top\Sigma_{\rm alias}^{+}X\). The four
factor-1 OAT aliases have total weight one, not four, and a candidate joint
tuple byte-identical to an existing physical tuple contributes zero
incremental information. The objective, rank-10 assertion, condition number,
and determinant use the same byte-identical design matrix as the final
estimator, with frozen method/block weights.
Freeze the candidate lattice, selection seed, tie-break, selected-row order,
combined rank, condition number, D-efficiency, and matrix digest. A candidate
that maximizes the joint-only determinant but loses the combined objective is
rejected. Factors 4 and 8 are prohibited from every
joint training tuple and from every inner/outer fitting fold. They occur only in
the named one-axis extrapolation rows and the joint holdouts described next;
their outcomes can never choose a model, threshold, interaction term, or
hyperparameter.

Before any outcome exists, choose four additional joint holdout tuples
deterministically by maximin log-resource distance from the 16 D-optimal fit
tuples. The candidate set is the full four-axis lattice excluded from fitting,
and every selected tuple must vary at least two axes and contain factor 4 or 8
on at least two axes. Freeze the admissible physical-feasibility domain,
candidate set, distance metric, selection/tie-break seed, feasibility
exclusions, order, and digest before G2-CAP. If fewer than four such
multi-axis-extreme tuples exist or fit the frozen aggregate budget,
`interaction` and `segmented` branches are
`INELIGIBLE`; interior tuples cannot substitute for them, though an
additive/marginal A100 relation may remain eligible. Materialize both methods
at every tuple. These eight rows are
`a100_joint_holdout` and are opened only after the same provisional seal as the
one-axis holdouts. Selection maximizes both distance from the fit set and
pairwise MAXPRO/maximin distance among holdouts, must exceed a frozen minimum
pairwise log-distance, and must satisfy a predeclared four-row
Hadamard-style high/low directional coverage and rank rule across all axes.
An all-high corner plus near-duplicate one-axis perturbations fails even if
each row contains two extreme factors.

Keep all 120 logical rows, but do not turn duplicate resource tuples into
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
Replicating one logical alias into four rows must leave the selected joint set,
information matrix, and determinant unchanged; treating an OAT-identical joint
candidate as new information fails before G2-CAP.

Import the same 16 TEST workload families, but not PAPER-C's powered user count.
Before any capacity outcome, deterministically freeze two families as
`final_workload_holdout`; the remaining 14 alone may enter inner/outer model
selection, thresholding, and interval calibration. The dedicated
`scaling-power.json` selects \(n_{\rm scaling}\), divisible by 16. Each logical
scaling row has exactly \(n_{\rm scaling}\) A100 child references,
\(n_{\rm scaling}/16\) per family. Freeze
`NVIDIA-A100-SXM4-80GB` as the sole required `hardware_id`, including exact
driver, CUDA, clock/power, container, serving-stack, model, and tokenizer
fingerprints. Assign the 64 first-eight-factor one-axis rows plus all 32 joint
rows to `a100_fit`; assign the 16 factor-4/factor-8 one-axis rows plus eight
joint maximin rows to `a100_holdout`. The holdout rows are untouched configurations, not an extra
hardware population, and cannot fit a coefficient, select a competitor, tune
a threshold, or choose a hyperparameter:

```text
a100_fit_logical_rows = 96
a100_one_axis_holdout_logical_rows = 16
a100_joint_holdout_logical_rows = 8
a100_holdout_logical_rows = 24
a100_fit_config_fit_family_children = 96 * n_scaling * 14 / 16
family_only_holdout_children = 96 * n_scaling * 2 / 16
resource_only_holdout_children = 24 * n_scaling * 14 / 16
double_holdout_children = 24 * n_scaling * 2 / 16
total_logical_child_references = 120 * n_scaling
child_references_per_logical_per_family = n_scaling / 16
charged_post_unlock_anchor_executions = 4
unique_physical_executions <= 120 * n_scaling + 4
```

`scaling-cohort-index.json` freezes every child key, workload family, panel seed,
resource role, family role, their four-way execution role, logical parent,
physical configuration, deduplication edge, and execution order. Only
fit-resource×fit-family children are visible before the provisional seal.
Family-only, resource-only, and double-holdout children unlock together and
remain score-only. Missing/duplicate/extra child keys, choosing final families
after outcomes, a role swap, an environment-fingerprint mismatch, or any final
family/holdout result in a fit/CV digest is a gate failure.

The three validation roles are separate conjunctive estimands and are never
pooled by row count. Within each role, first compute each named family's score
using equal weights over unique physical configurations after logical-alias
collapse; then give the named families exact equal weight. Materialize distinct
family-only, resource-only, and double-holdout RMSE and PI-coverage rows in
`SCALING-A100-GLOBAL-{K}` and require all three. Duplicating a logical
configuration, splitting/duplicating a family label while preserving its fixed
weight, or permuting result rows cannot alter a score, interval, or decision.

To diagnose unavoidable fit→holdout stage drift, predeclare two representative
fit resource tuples per method as four post-unlock anchor reruns. Anchors are
separate charged physical control runs, not additional logical fit rows; they
are score-only, cannot update coefficients/intervals, and have frozen
environment/thermal/cache drift thresholds. Drift beyond threshold makes the
A100 law ineligible or narrows it; anchors may not post-hoc adjust holdout
predictions.

Within each workload-family×paired-seed block, the pre-outcome fit scheduler
interleaves only fit-resource×fit-family methods, resource axes, and factors
across node/rank, wall-clock, thermal, and cache-phase blocks. Fit completes
and seals before any holdout capability exists. After the one-time unlock, an
independently frozen blocked schedule interleaves the family-only,
resource-only, and double-holdout children; the two stages never overlap. Both
seeds, balance tolerances, block labels, order digests, and the observed stage
shift/control diagnostics are bound in the cohort index and canonical global
ledger; the power model and surface fit include fit-stage block terms and the
holdout score reports stage-shift sensitivity. Sorting by factor, running one
method before the other, scheduling any holdout before unlock, fitting on
actual rather than assigned FLOPs/tokens/bytes, or node/time imbalance beyond
the frozen tolerance fails G2-CAP or narrows the claim before outcome
inspection.

`stc scaling power` imports the complete two-stage knee→surface estimator,
nested 14-family grouped CV, all seven competitors, child-role firewall, and
one-axis/family/joint/double-holdout score predicates. It evaluates
`n_scaling in {64,96,128,160,192,224,256,288,320,352,384,416,448,480,512}`
with the common independent
search/verification Monte Carlo contract and refits both stages in every
replicate; a plug-in knee or selected-model-only simulation is forbidden.
Its simultaneous bounds spend only
\(\alpha_{\rm scaling}=0.025\). Deployment stakes freeze:

Before power, `scaling materialize-contrast-family` emits the ordered
`SCALING-A100-GLOBAL-{K}` table. It enumerates every method×axis effect margin,
every claim-capable interaction/segment/bottleneck-direction margin for every
eligible competitor, both exact/temporal knee-component margins, and all
fit/family/resource/double-holdout prediction predicates with IDs,
orientations, boundaries, roles, and row-statistic types. Every axis row binds
the complete ordered factor-doubling edge list and equal weights defined below;
every interaction row likewise binds its ordered 2×2 face list, orientations,
and equal weights. Segment/bottleneck rows bind their named hinge-side edges.
\(K\) is the byte-frozen
row count, not a value chosen after selection. Each Monte Carlo/bootstrap
replicate reruns knee estimation, all competitor fits and selection, and the
complete union max-statistic. Adding, dropping, relabeling, or changing a
boundary, or scoring only the eventually selected model, invalidates the
digest. Global-null fixtures include best-model-after-noise and require any
support probability within \(\alpha_{\rm scaling}\).

```text
tau_exact(128) = tau_temporal(128) = 0.80
admissible tau range for a prospective amendment = [0.10, 0.90]
first-stage bracket: at each tuple, simultaneous CAL LCB for BOTH components
                     at the selected low load is >= 0.90, and simultaneous
                     CAL UCB for BOTH at the selected high load is <= 0.70
axis effect:          0.10 within-family delta-log-K per resource doubling
axis null boundary:   0.02
interaction/segment effect when claimed: 0.10
interaction/segment null boundary:       0.02
holdout delta-log-K RMSE:                <= 0.10
nominal prediction interval:             95%
empirical coverage floor:                0.90
```

The selection-binding scaling DGP also freezes every validation-row interior,
not only effect sizes: true low-bracket exact and temporal retention are `0.95`,
true high-bracket values are `0.65`, true family-level
delta-log-\(K\) predictive RMSE is `0.05` (null boundary `0.10`), and true
nominal-95% prediction-interval coverage is `0.95` (null boundary `0.90`).
Family, user, first-stage-knee, stage-shift, and cross-row correlations range
over the CAL-derived nuisance covariance set, and power is minimax over that
set. Mixed-null/type-I simulations put each bracket, RMSE, coverage, and effect
row in turn on its boundary while all other rows remain interior; changing any
interior DGP or nuisance-set digest after CAL stales power and G2-CAP.

Prediction-interval coverage is a Bernoulli row and uses a hierarchy-aware
boundary-score/concentration statistic with positive null variance at
`p=0.90`, jointly calibrated with continuous rows in the global max statistic.
It never uses an empirical-variance Wald interval. Before simulation, an
all-success attainability calculation uses only independent top-level users in
the two held-out families and the allocated \(K\)-row multiplicity; repeated
resource configurations for the same users do not increase \(n_{\rm eff}\).
If no candidate can make the simultaneous coverage LCB exceed 0.90 even under
all successes, G2-CAP returns `FAIL_POWER_BOUND_UNATTAINABLE`. All-covered and
all-uncovered fixtures return finite decisions, and duplicating configurations
for the same 32 users cannot improve the bound.

For each claim-enabled competitor branch, data success requires every required
family/resource/method knee finite and bracketed, the appropriate simultaneous
effect margin above its boundary, and all resource-only, family-only, and
double-holdout predicates. The verification familywise LCB at the allocated
\(\alpha_{\rm scaling}=0.025\) of every branch must exceed 0.90; boundary
simulations must keep one-sided type-I error at most 0.025 and nominal-95
interval coverage LCB at least 0.90. The smallest passing
candidate is \(n_{\rm scaling}\). `FAIL_MAX_N` disables capacity execution at
G2-CAP without affecting core G2. A competitor lacking a defensible powered
alternative remains `EMPIRICAL_CHARACTERIZATION_ONLY` and cannot yield a law.

H100 is strictly an optional future `external_transport_validation` track.
It is recorded in a separate extension manifest/result block, never merged
with A100 observations, and is excluded from the 120 logical cells, child
coverage, G2/G8 inputs, \(Q_{\rm capacity,max}\) budget, model selection, and the minimum
`a100_regime_scaling_law` predicate. Its absence cannot narrow an
A100-regime result, and its presence cannot promote an unsupported A100 result.
The unqualified label `scaling_law` and any cross-hardware/system law remain
inadmissible unless a prospective amendment makes a second hardware family
mandatory, powers it, and freezes its transport predicate before results.

Cell count is not compute budget. Before any data, freeze the sampling-free
engineering allowance `reference_profile_id=RRE-NATIVE-A100-V1` in
`configs/stc/design/rre-reference-native.yaml`. It gives exact positive native
allowances for one reference unit and binds units, target A100 topology,
container/counter semantics, rationale, and digest. It is not estimated from a
stochastic child, pilot, CAL, or TEST and therefore has no favorable
denominator uncertainty. Changing it is a prospective budget amendment. Its
native reference vector is:

```text
x_ref = {GPU-s, CPU-core-s, DRAM-byte-s, HBM-byte-s,
         storage-I/O-byte, network-byte}
```

For every unique physical cohort with its program-wide simultaneous upper-99
resource vector
\(x\), define dominant-resource reference-run equivalents
\(\mathrm{RRE}(x)=\max_{r:x_{\rm ref,r}>0}x_r/x_{\rm ref,r}\).
A resource whose reference component is zero receives a separately frozen
native hard cap and cannot be treated as free. Missing counters are
inadmissible, never zero. Include all fresh-snapshot load panels and conservative
hardware multipliers. `scaling-budget.json` records each projected native
vector, RRE, covariance/source digest, and the total. Let the prospective total for
this track be \(P_{\rm scaling}\), including the four anchors' separate
simultaneous-envelope native vectors and fixed post-unlock order. G2-CAP
passes this track only if
\(P_{\rm scaling}\le q_{\rm scaling}\), where the positive quota
\(q_{\rm scaling}\) is frozen in the shared capacity-budget envelope below.

The runtime enforces both \(q_{\rm scaling}\) and the shared cumulative
\(Q_{\rm capacity,max}\) ceiling in frozen cohort order. Define committed
usage as completed measured actual plus every unreconciled hard reservation.
Before a launch it requires both
`track_committed + hard_reservation(next) <= q_scaling` and
`global_committed + hard_reservation(next) <= Q_capacity,max`, plus for every
native resource/carbon component \(r\),
`global_native_committed[r] + reservation_native[r] <= sponsor_cap[r]`.
A crashed or abandoned
reservation remains fully committed until a typed no-launch reconciliation;
no second token can issue meanwhile. Native GPU/CPU time, DRAM/HBM occupancy,
storage I/O, and network budgets use enforceable cgroup/device/I/O/network
limits plus a frozen counter-latency margin. Actual use above reservation is a
system-safety failure that invalidates the cohort and halts the capacity
program, never an allowed budget overrun. It records the complete measured
native vector and actual RRE after every cohort. A quota termination, missing required A100
holdout, or required child left unexecuted yields
`BUDGET_EXHAUSTED_OR_INCOMPLETE` and forces `empirical_curve`; no
outcome-informed reordering/substitution, cross-track borrowing, or
confirmatory surplus is allowed.
The scaling view
`results/scaling/actual-rre-ledger.jsonl` is derived deterministically from the
single canonical capacity ledger; every row binds the physical cohort,
before/after cumulative RRE, hard quota, measured resource counters, and result
digest. It is never an independent append target. `stc results validate`
recomputes both the view and canonical rows from raw lifecycle records and
rejects a missing/reordered row or any instantaneous/cumulative overrun.

- [ ] **Step 1A: Materialize the unconfounded coverage/codec experiment**

Create two manifests whose rows can never be pooled:

```text
prototype counts M = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}
fit counts          = first 8
extrapolation       = {2048, 4096}
prototype policies  = {uniform, value-aware-coreset, learned-quantizer, eviction}

cardinality arm       = 10 M × 4 policies = 40 logical cells
fixed-total-bits arm  = 10 M × 4 policies × 3 frozen B budgets = 120 logical cells
B candidate grid      = {2^20, 2^22, ..., 2^36} bits
B_active_repr,total   = 3 budgets selected before power by the frozen rule
```

Capacity-selection CAL performs an outcome-blind exact serialization dry run
for every candidate budget, all ten \(M\) values, and all four policies using
the selected codec/schema. It retains a candidate only when every row has
strictly positive payload allowance after all active-serving overhead and its
joint upper-99 native-resource reservation fits the sponsor envelope. The
three smallest retained candidates become the only
\(B_{\rm active\,repr,total}\) values and are bound in the capacity selection
bundle and deployment stakes before capacity-power CAL. Fewer than three
retained candidates yields `FAIL_COVERAGE_BUDGET_GRID_INFEASIBLE` at G2-CAP;
negative/zero-payload rows, dropping a large-\(M\) row, or choosing larger
budgets from TEST performance is forbidden.

In `coverage-cardinality-cells.jsonl`, codec family, precision, serialized
bytes per prototype, metric, key/value schema, and decoder are byte-identical
across \(M\); total bytes must equal fixed overhead plus \(M\) times the frozen
per-prototype bytes. This arm alone may estimate a pure
\(M^{-1/d}\) cardinality exponent.

In `coverage-fixed-bits-cells.jsonl`, each of three preregistered active
query-serving representation budgets \(B_{\rm active\,repr,total}\) is exactly
constant across \(M\), including fixed headers, keys/index,
provenance/policy metadata, decoder state, and payload.
For each row compute the achieved payload allowance
\[
b_{\rm payload}(M)=
\{B_{\rm active\,repr,total}-B_{\rm fixed}-B_{\rm index}(M)
-B_{\rm metadata}(M)\}/M
\]
from the measured serialization ledger; a negative allowance makes that row
prospectively infeasible. Raw/canonical evidence, lineage, rollback, and
recovery copies are reported separately in complete
\(B_{\rm retained}\) and \(B_{\rm peak,state}\) ledgers and remain hard
constraints. This arm supports an active-representation law, never a
fixed-total-memory law; an uncapped archive under a total-memory label fails
the claim gate. Fit the joint candidate
\(cM^{-1/d}+q(b_{\rm payload}(M))\), where \(q\) is an independently measured
codec-distortion curve rather than a free residual term. Holding only payload
bytes constant or substituting nominal \(B/M\) fails the arm invariant. Every row records
achieved covering radius, quantization distortion, retrieval error
\(\delta_M\), downstream utility, serialized total/peak bytes, and codec
digest.

TRAIN/CALIBRATION states choose metric, compact-support diagnostic, codecs,
policy hyperparameters, and tolerances. The last two \(M\) values and two
entire named workload families remain untouched TEST holdouts. A cardinality
claim requires the declared covering-number diagnostic and exact fixed-codec
invariant to pass and the branch's frozen equal-weight four-cell estimand
(two held-out sizes×two held-out families) to clear both simultaneous
prediction-error and interval-coverage margins. The four branch×size×family
results are mandatory worst-cell diagnostics but are not four separately
powered population-generalization gates. The fixed-bit arm can support only a
joint geometry--codec curve. Tests deliberately vary prototype count at fixed
total bytes, vary bytes/prototype in the cardinality arm, leak a holdout state
into codec fitting, omit codec distortion, or replace the four-cell finite
target by a best/worst-cell selector; each mutation must fail before fit.

The deployable `value-aware-coreset` policy uses only a frozen
TRAIN/CALIBRATION value estimator whose allowed-feature schema and digest
exclude future reuse, future queries, realized utility, TEST family identity,
and every holdout label. Any policy using those fields is explicitly
`ORACLE_UPPER_BOUND` and excluded from deployable slope/prediction claims.
Injecting a future/value label into a deployable coreset weight fails before
fit.

Freeze two ordered coverage-power decisions in deployment stakes. For
`COV-CARD-SLOPE`, let
`g_card = 1 - radius(2M)/radius(M)` on the fixed-codec log-linear fit:
the null boundary is `g_card=0.05` and the interior deployment alternative is
`g_card=0.15`. Data success requires the simultaneous one-sided LCB calibrated
at \(\alpha_{\rm coverage}=0.025\) for `g_card-0.05` above zero for every
required policy branch. For
`COV-FIXED-BITS-PRED`, let `g_joint` be the held-out normalized-log-radius
RMSE reduction of the frozen geometry-plus-codec predictor versus the
independently calibrated codec-only predictor: the boundary is `0.02`, the
interior alternative is `0.10`, and data success requires its simultaneous
one-sided LCB calibrated at \(\alpha_{\rm coverage}=0.025\) above the
boundary. Both IDs use one frozen joint max-stat critical function
across policies and bit budgets, never claim-specific critical values. Pilot
effects cannot set either alternative, and arms can never pool.

`coverage materialize-contrast-family` freezes `COVERAGE-GLOBAL-48`:

- 4 ordered `g_card-0.05` effect rows, one per cardinality policy;
- 12 ordered `g_joint-0.02` effect rows, one per fixed-bits
  policy×selected-budget branch; and
- for each of those 16 branches, one equal-weight four-cell validation
  estimand pooling the two held-out sizes×two named held-out families, with
  both `0.10-normalized_log_radius_RMSE` and
  `PI_empirical_coverage-0.90` margins (32 rows).

Every row binds arm, policy, selected \(B_{\rm active\,repr,total}\) digest
when applicable, the exact four-cell equal weights/role, orientation, boundary,
row-statistic type, and Boolean parent ID. Each claim requires all of its effect and validation rows;
deterministic codec/serialization invariants remain alpha-free vetoes. The
whole 48-row max-stat union spends
\(\alpha_{\rm coverage}=0.025\). Add/drop/reweight/sign/bit-budget mutations,
best-branch selection, or a mixed global-null fixture in which an effect row
fails for one claim and a validation row fails for the other must not permit
any false coverage support above that alpha.

The empirical-coverage row is Bernoulli and never uses a naive Wald
studentizer. For each branch and held-out family, first pair each within-family
top-level user across the two held-out sizes and average that complete
two-size panel; then combine the two fixed-family estimates with exact one-half
weights. Test the `p=0.90` boundary with a hierarchy-aware one-sided score statistic
whose null covariance is the least-favorable member of the frozen
block/cross-branch covariance set and whose variance floor is positive at the
boundary. Calibrate that score jointly with the continuous effect/error rows by
the same boundary Monte Carlo max statistic. Exact binomial inversion is
allowed only when a preregistered independence diagnostic validates both
within-user and cross-branch independence. All-covered and all-uncovered
fixtures must return finite score bounds; zero empirical variance may neither
auto-pass nor crash. A naive studentized max-\(t\) implementation must fail a
mutation in which every observed interval covers.

`stc coverage power` imports the final coverage estimator and uses only the
receipt-validated capacity-calibration covariance/error summary. It evaluates
independent seed counts
`{64,96,128,160,192,224,256,320,384,448,512}` with the common disjoint
search/verification Monte Carlo contract. The max-\(|t|\) family for both
coverage IDs spends only \(\alpha_{\rm coverage}=0.025\). The complete
pre-data interior DGP sets every required branch's effect to its alternative
above, the true equal-weight four-cell normalized-log-radius RMSE to `0.05`
(boundary `0.10`), and true prediction-interval coverage to `0.95` (boundary
`0.90`). Cell heterogeneity and all cross-branch/error/coverage correlations
range over the receipt-validated nuisance covariance set; the minimax branch
power, not a favorable independent Bernoulli plug-in, selects \(n\). Boundary
and mixed-null DGPs put each validation row in turn at `0.10` error or `0.90`
coverage while leaving the others interior, and must control familywise false
support at 0.025. Every required fixed-codec and fixed-bit branch must have a
verification-familywise Monte Carlo
**detection-power** LCB at the allocated
\(\alpha_{\rm coverage}=0.025\) above 0.90, a nominal 95%
prediction-interval empirical-coverage LCB at least 0.90, and a simultaneous
upper bound at that same alpha on held-out normalized-log-radius prediction
RMSE at most
0.10. Detection power and interval calibration are distinct recorded
predicates; a nominal 90% interval is never substituted for this contract or
required to have a finite-simulation
coverage LCB above its own nominal boundary. The smallest candidate passing
every branch is `n_coverage`; `FAIL_MAX_N` blocks G2-CAP.
Before simulation, an analytic attainability check uses the actual
top-level-user count in each 48-row coverage margin and its allocated
multiplicity. A candidate grid whose all-success best-case simultaneous LCB
cannot reach 0.90 is `FAIL_POWER_BOUND_UNATTAINABLE`, never passed via zero-SE
max-\(t\).

Each of the 160 logical rows has exactly `n_coverage` child references balanced
over the 16 frozen TEST workload families, with at least four independent
users per family. Scaling enforces the identical floor. A fixture with one
user per family and arbitrarily many within-user probes must fail power and
cohort validation. The same pre-outcome 14/2 split used
by scaling reserves two final workload families. The claim-bearing validation
estimand is the frozen equal-weight mixture of two held-out sizes×those two
named families; branch×size×family intervals are mandatory diagnostics but
cannot each be treated as sufficiently powered claim gates. Passing supports
transport to that four-cell finite target within the frozen equal-weight
16-family benchmark. This is not a workload-generator population
claim: family labels are fixed strata, uncertainty resamples paired users
within each stratum, and increasing seeds cannot create family-level
generalization. Duplicating or splitting either held-out family label cannot
improve power, standard errors, or claim scope. A single-family result is
labeled only for that named family.
`coverage-scaling-manifest-index.json` binds the two non-poolable arm manifests;
`coverage-scaling-cohort-index.json` binds every child, physical-deduplication
edge, arm, fit/holdout role, seed, family, codec, and budget identity. For each
named held-out family it also binds a `coverage_validation_panel_id` that pairs
the **same within-family physical user/seed** across the two held-out sizes.
The row statistic first averages/resamples those complete two-size panels
within family and then combines the two fixed-family estimates with exact
one-half weights; it never asserts one user identity across different workload
families. A missing size, independently permuted user ID in one size, or
cross-family row-count pooling fails cohort validation even when marginal cell
counts match. Production
first exposes only fit sizes and non-held-out workload families. After a
fit-only result validation passes, `stc coverage fit --stage provisional`
seals codec-distortion fits, policy hyperparameters, model identities, and
prediction intervals. `stc coverage holdout-unlock` binds that immutable
artifact and authorizes exactly the two held-out sizes plus held-out workload
children once. A combined fit+holdout run, opening any holdout result before
the seal, or refitting after unlock fails.
Within each fit or post-unlock holdout phase, pre-outcome blocked randomization
interleaves \(M\), policy, bit budget, family/seed, node/rank, and
thermal/cache-time blocks; it never runs ascending \(M\) or one policy in a
single block. The order digest is frozen in the cohort index and canonical
ledger, and block effects enter power and fit.
`coverage-scaling-budget.json` contains a positive non-borrowable per-track RRE
quota frozen at G2-CAP, every child's program-wide simultaneous-envelope
projected reservation, and the
frozen execution order.
`results/coverage-scaling/actual-rre-ledger.jsonl` is a deterministic
validated view of the canonical capacity ledger, not an append target; quota borrowing,
seed reduction, and outcome-informed reordering are forbidden.

- [ ] **Step 1B: Materialize the conditional parametric-information experiment**

Use three parametric substrates—shared-weight update, fixed-rank user adapter,
and preallocated expert/slot—at two frozen precisions and the ordered
association-load grid `{8,16,32,64,128,256,512,1024}`, for 48 logical cells.
Each association payload has exactly \(L=256\) independent uniform Bernoulli
bits. These values and the three coverage
\(B_{\rm active\,repr,total}\) values are bound in
`deployment-stakes-G2-CAP.json`; extending either range after outcomes is a
new prospective amendment. A dedicated
parametric-information power procedure—not PAPER-C's powered \(n\)—fixes the
independent seed-ensemble size. Every child run:

1. starts in a fresh process from the same public \(\theta_0\) digest;
2. draws a new independent opaque-payload seed with known
   \(H(Y\mid Q,\Theta_0)\);
3. retrains the candidate from scratch with the frozen protocol;
4. evaluates through a decoder/readout frozen on CALIBRATION before any TEST
   payload is generated; and
5. serializes a complete answer-bearing state inventory and bit count.

Within each substrate×precision stratum, the primary load sweep assigns the
same total gradient-token cap and the same sleep-FLOP cap to every \(N\).
Those assigned caps, realized gradient tokens/FLOPs, early termination, and
exposure per association are recorded; an optimizer may not silently spend
more compute at a larger load. The resulting quantity is therefore a
**protocol-conditional effective information capacity**, not an intrinsic
bits-per-parameter constant. A separately labeled characterization arm holds
exposure per association fixed over a sparse, preregistered
\(N\times\)compute grid. If an apparent high-\(N\) knee disappears there, the
primary finding is `PROTOCOL_LIMITED`, not parametric bit capacity. This
secondary arm cannot rescue a failed primary estimator-qualification gate or
promote a universal law.

Each of the \(N\) association payloads contains \(L=256\) independent uniform
Bernoulli source bits. Here \(p_{a\ell}\) means the true error probability of
the CAL-frozen decoder orientation for association \(a\), bit \(\ell\), not
block error. The population total and normalized bitwise Fano functionals are

```text
I_Fano_total = sum_a=1..N sum_l=1..L [1 - h2(p_a,l)]
I_Fano_per_payload_bit = I_Fano_total / (N * L)
```

The finite-sample inferential lower bound is never the raw plug-in
`sum[1-h2(p_hat)]`, which is upward biased near \(p=0.5\). It either maps
simultaneous upper confidence bounds \(p^U_{a\ell}\) through
`max(0, 1-h2(min(p^U,0.5)))` and sums them, or uses a validated direct
simultaneous lower-confidence procedure for the nonlinear functional. The
master seed family is the resampling cluster and coverage is simultaneous
across bits, loads, precisions, and substrates. A bias-corrected plug-in is
reported only as descriptive. A shuffled/no-signal small-\(n\) fixture must
return inferential lower bound zero and non-support even when its descriptive
plug-in is positive. Thus no signal is \(p_i=0.5\); no block-error null is
substituted.
The unnormalized information lower bound and
`I_LB / complete_mutable_state_bits` are also reported, but the latter never
omits a carrier. A variational lower bound is optional and,
if present, has a separately frozen critic family, training split, and
coverage diagnostic; it cannot replace a failed Fano track after outcomes are
seen. Conditioning is permitted only on public, payload-seed-independent
\(\Theta_0\). Every payload-, TRAIN-, CALIBRATION-, or TEST-dependent adapter,
router, optimizer, codebook, decoder delta, checkpoint, lookup table, and
sidecar is charged to complete mutable state. A theta-only bits/parameter row
is admissible only when the frozen readout consumes \(\theta\) and
\(\Theta_0\) alone.
After training, the evaluation readout has no capability to access the payload
RNG seed, generated training dataset, replay log, or a regeneration service.
If any such object is retained for reproducibility/recovery, its canonical
compressed bit length is charged to complete mutable state and the theta-only
branch is inadmissible. A fixture with \(\theta=\theta_0\) whose answers are
regenerated from a retained seed/log must be detected as a hidden carrier.

`test_parametric_information.py` independently checks a no-signal channel, a
known binary-symmetric channel, seed resampling, fresh-base restoration,
decoder immutability, and confidence-interval coverage. An adversarial fixture
places every answer bit in a router while leaving \(\theta=\theta_0\): the
theta-only estimate must be zero/inadmissible and the complete-state
denominator must include the router. Missing auxiliary state, reusing a
checkpoint across seeds, conditioning on a payload-dependent carrier, fitting
the decoder on TEST, or estimating from one checkpoint fails G2-CAP validation.

`stc information power` imports that exact estimator and evaluates independent
seed counts `{32,64,96,128,160,192,224,256}` using a preregistered search
Monte Carlo followed by the common disjoint fixed-size verification Monte
Carlo contract above (20,000 verification experiments unless the frozen
precision calculation requires more). A top-level `seed_family_id` deterministically
spawns disjoint child payload seeds for every substrate×precision×load cell;
payload seed and checkpoint reuse remain forbidden. Because bits and
associations learned in one child can be dependent, the seed family is the
resampling cluster. Power stresses the disjoint power-CAL estimates over a
frozen correlated-BSC/beta-binomial grid and inference uses a cluster
bootstrap or an equivalently valid concentration bound. Exact binomial
probabilities are permitted only if a frozen independence diagnostic passes;
otherwise nominal-binomial intervals are invalid. The
frozen channels are no-signal \(p=0.5\) and binary-symmetric channels
\(p\in\{0.10,0.25\}\). A candidate passes only when:

```text
no-signal false-positive probability upper 95% bound <= 0.05
nominal 95% interval empirical-coverage lower 95% bound >= 0.93
Pr(Fano lower bound > 0.05 bit/payload-bit at p=0.25) lower 95% bound > 0.90
95th percentile of interval half-width <= 0.05 bit/payload-bit
```

The `p=0.25` branch binds detection power, `p=0.10` binds the worst required
precision/coverage branch, and `p=0.5` binds false-positive control.
`c_min=0.05 bit/payload-bit`, \(L\), and the **population-functional**
binary-symmetric boundary
`p_boundary=0.369127748985`, satisfying
`1-h2(p_boundary)=I_Fano_per_payload_bit=0.05`, are frozen in deployment
stakes. This boundary and its digest are invariant to sample size and interval
realization; they are never solved from an inferential LCB. Boundary draws
require
`Pr(inferential_LB_per_payload_bit > c_min) <= 0.05` under the joint
seed-cluster procedure. They calibrate one-sided type-I error and are not used
as the power alternative.
The homogeneous BSC is only one boundary fixture. Uniform type-I validation
also ranges over the complete heterogeneous null
\[
\frac1{NL}\sum_{a,\ell}\{1-h_2(p_{a\ell})\}\le 0.05,
\]
including sparse near-perfect bits mixed with chance bits, heterogeneous
association difficulty, and beta-binomial/correlated seed clusters on the
boundary. The per-bit simultaneous-\(p^U\) construction may rely on its uniform
monotonic theorem; any direct nonlinear lower bound must provide an equivalent
least-favorable proof or calibration over this null set. A method calibrated
only on homogeneous `p_boundary` fails when a same-functional
sparse-signal/mixture fixture exceeds 0.05 false positives.
The smallest passing candidate is `n_information`; if none passes,
`FAIL_MAX_N` blocks G2-CAP. Each of the 48 logical rows has exactly
`n_information` fresh-retraining children. The
`parametric-information-cohort-index.json` binds every independent payload
sub-seed, its master `seed_family_id`, fresh \(\theta_0\) restore, frozen decoder/readout, complete-state
inventory, logical parent, execution order, and physical identity; seed or
checkpoint reuse is forbidden. It also freezes a pre-outcome blocked schedule
that interleaves substrate, precision, and association load within
master seed family over node/rank and thermal/cache-time blocks; monotone load
order or a whole substrate in one time block is invalid, and block effects
enter power and estimates. `parametric-information-budget.json` freezes a
positive non-borrowable track quota and every child's program-wide
simultaneous-envelope projected RRE.
`results/parametric-information/actual-rre-ledger.jsonl` is a
deterministic validated view of the canonical capacity ledger, not an
independent append target.

This power result is labeled `ESTIMATOR_QUALIFICATION_ONLY`. It supports
detection/coverage/precision of the normalized Fano estimator, not superiority
of one substrate, a universal bits-per-parameter constant, or a parametric
capacity/scaling law. Theta-only and complete-state efficiencies and all
substrate×precision×load contrasts are simultaneous-interval
`CHARACTERIZATION` outputs. Promoting any such comparison requires a separate
pre-TEST amendment with a numerical efficiency margin, ordered contrast family,
power, and multiplicity. A best-TEST-substrate selector or an omitted
answer-bearing denominator fails the claim gate.

After all three powers and projected budgets exist, `stc capacity
budget-freeze` writes `capacity-aggregate-budget.json`. It binds
\((q_{\rm scaling},q_{\rm coverage},q_{\rm information})\), requires every
quota to be positive and every track projection to fit its own quota, and
requires:

```text
q_scaling + q_coverage + q_information <= Q_capacity,max
projected_joint_upper_99_scaling
  + projected_joint_upper_99_coverage
  + projected_joint_upper_99_information <= Q_capacity,max
for every native resource/carbon component r:
  sum(projected_joint_upper_99_by_track[r]) <= sponsor_absolute_cap[r]
```

Quotas are non-borrowable.
`results/capacity/actual-rre-ledger.jsonl` is the single canonical budget
source of truth. Each row carries `track_id`, track/global before-and-after
totals, reservation ID, frozen cohort/order key, hard reserved RRE, measured
actual RRE, the projected and actual native vectors, retries/failures, result
digest, and a hash-chain predecessor. A
single node-0 budget coordinator follows the frozen global order, durably
reserves the complete next cohort and atomically replaces
`results/capacity/actual-rre-ledger.head.json` with the new row count and
terminal digest before
issuing a launch token. An abandoned reservation continues to consume quota
until a typed reconciliation proves that no work launched; tokens cannot be
reused. Concurrent track reservations are forbidden in v1.

The apparent track ledgers are deterministic validated views derived from that
canonical file, never independent append targets. Missing, reordered, forked,
or nonreconciling rows; a stale reservation; an uncharged retry; derived-view
drift; a track overrun; or a global actual total above
\(Q_{\rm capacity,max}\) fails validation. If the three
powered tracks cannot fit the shared cap, G2-CAP returns
`FAIL_CAPACITY_AGGREGATE_BUDGET`; raising the sponsor ceiling requires an explicit
prospective amendment.

Because the full-horizon surface charges eight 128-cycle panels per scaling
child, G2-CAP must dry-run this exact work in the blinded cost model. If it cannot
fit the aggregate cap, the full-horizon track remains
`FAIL_CAPACITY_AGGREGATE_BUDGET`; seeds, panels, lags, or rows are not silently
dropped. A prospective amendment may instead define a distinct short-horizon
`K_eff_short` surface using only lags `{1,8}` plus a separately powered sparse
lag-128 transport study, with correspondingly narrowed claims. Sixteen-cycle
data may never retain a lag-32/128 or full-horizon label.

- [ ] **Step 2: Estimate a capacity knee at each resource tuple**

Do not infer a knee from a monotonically increasing lifetime trajectory.
Before TEST, capacity-selection CAL evaluates the frozen candidate grid
`{2^0,2^2,...,2^24}` and chooses for each prospective
resource×method tuple a deterministic eight-point log-load subgrid. Its lowest
point must have simultaneous CAL lower bounds for both exact and temporal
retention at least \(0.80+0.10=0.90\), and its highest point must have
simultaneous CAL upper bounds for both components at most
\(0.80-0.10=0.70\). Among admissible subgrids,
choose the widest log span, then the smallest lower endpoint; record every
candidate and tie. The disjoint power-CAL cohort re-evaluates only that frozen
subgrid. Failure to find a bracket, finite fit, or budget-feasible eight-panel
subgrid makes that tuple, and therefore any law requiring it,
`INELIGIBLE_UNBRACKETED` before TEST. Interval-censored extrapolation cannot
support the law. TEST never recenters, expands, or replaces a bracket.
Exact-only bracketing with an out-of-range temporal knee (or vice versa) is
`INELIGIBLE_UNBRACKETED`; averaging components is forbidden. The common
\(\tau=0.80\) encodes the pre-data deployment requirement that at least 80% of
admitted associations remain answerable at lag 128. Changing either threshold
with raw coefficients held fixed stales the bracket, component knees,
`SCALING-A100-GLOBAL-{K}`, power, and capacity preregistration.

Every scaling child then contains eight independent
**128-post-ingest-cycle** fresh-snapshot panels with

```text
N_assoc(tuple) = the frozen eight-point capacity-selection subgrid.
```

At panel cycle 0, restore the same canonical empty-memory/base-model snapshot
and ingest exactly the assigned number of generator-controlled associations.
Then issue the globally frozen retention lags `{1,8,32,128}` with the same
probe mix during 128 post-ingest cycles. Every admitted association has its
lag-128 query at the final cycle, so it remains denominator-valid; an item
introduced after cycle 0 is a protocol error, not administrative censoring.
Association sets are disjoint across panels. A seed-derived balanced Latin
square assigns panel order across paired users, so load is orthogonal to
wall-clock order, cycle phase, event mix, and query age. Snapshot restore is
orchestration rather than a free method action; every
compiler/retrieval/serving resource consumed across all eight 128-cycle panels
is charged and included in projected and actual RRE. Tests reject a 16-cycle
panel that cannot expose lag 32/128, state carry-over, monotone panel order,
unequal probe schedules, a missing final-cycle lag-128 denominator, or a load
bin with fewer than the frozen user/family replicates.

Retention probing is a full census and is read-only against an immutable snapshot: it cannot write
memory, adapt weights, update retrieval statistics, or change the measured
state. Every query gets the same assigned \(T_{\rm live,cap}\), frozen
concurrency, batching, cache/warm-state policy, and queue class. Execute probes
serially under equal conditions; every admitted association contributes to the
lag-specific census denominator, and larger \(N\) may increase total charged
probe cost but never concurrent service pressure. Service
throughput/saturation is a distinct endpoint and cannot define \(K_{eff}\).
Query-triggered writes/cache eviction, concurrency varying with \(N\), sampling
a subset, or a non-census denominator makes the knee invalid.

Estimate \(K_{eff}(f,r,m)\) separately for workload family \(f\), resource tuple
\(r\), and method \(m\) using the preregistered binomial logistic retention
curve. The full-horizon primary uses lag 128 only. It fits separate
threshold-anchored component knees \(K_{\rm exact}^{(128)}\) and
\(K_{\rm temporal}^{(128)}\), with simultaneous uncertainty, and defines
\[
K_{\rm eff}^{(128)}
=\min\{K_{\rm exact}^{(128)},K_{\rm temporal}^{(128)}\}.
\]
Both component-knee margins belong to one simultaneous family. Derive the
nonsmooth interval without selecting a component:
`CI(K_eff)=[min(L_exact,L_temporal), min(U_exact,U_temporal)]`; never bootstrap
only the observed smaller/argmin component. Power requires both component
branches and includes tie/order-swap fixtures. Thus one component cannot mask failure of the other. Lag 1/8/32 surfaces are
secondary horizon-labeled outputs under the frozen multiplicity family; they
cannot be averaged with lag 128 or rescue the full-horizon law. CALIBRATION,
power, bootstrap, brackets, resource-surface fit, and holdouts import this exact
scalar rule and the same \(\tau_{\rm ret}\). Each family-level fit
requires all eight bins, observations on both sides of the threshold,
nonseparation, and an identifiable finite knee.

Bracketing and a finite coefficient are not sufficient. A frozen
threshold-anchored-logistic goodness-of-fit veto tests monotonic decrease,
single crossing, and a simultaneous retention-residual envelope over all eight
loads. On failure, a preregistered shape-constrained monotone crossing fit is
computed as a sensitivity and its log-knee must agree with the logistic knee
within 0.10. Multiple crossings, a violated monotonic envelope, or larger
disagreement yields `KNEE_UNIDENTIFIED` and excludes that tuple from any law.
Power includes Gompertz, mixture/two-knee, locally flat, and separated-shape
mutations; a biased finite logistic knee may not pass merely because all eight
bins exist.

Freeze one reference resource tuple \(r_0\), present in the 96 fit-resource
configurations, before G2-CAP. Define
\[
K_0(f,m)=K_{\rm eff}(f,r_0,m),\qquad
\Delta\log K(f,r,m)=\log K_{\rm eff}(f,r,m)-\log K_0(f,m).
\]
Here and everywhere in the capacity contract, `log` is the natural logarithm:
\(\Delta\log K=\ln(K/K_0)\), and each resource coordinate is
\(\ln(r_j/r_{0j})\). “Per doubling” is the finite contrast
\(g_m(\ldots,2r_j,\ldots)-g_m(\ldots,r_j,\ldots)\) on that natural-log
response scale. A coverage residual is
\(e=\ln(\widehat R/R)\), with dimensionless ratio and no hidden
standardization, but the claim-bearing accuracy functional is the nonnegative
equal-weight
\(\mathrm{RMSE}_{\log R}=\sqrt{\sum_c w_c e_c^2}\), never signed mean error.
Thus under- and overprediction cannot cancel or turn a large underprediction
into an automatic pass; a `{0.1,10}` prediction-ratio fixture has large error.
The bases and reference constants are schema-bound; switching to log2 without
transforming every threshold stales power, fit, and preregistration.

The phrase “axis effect” never permits a post-fit slice. Before power, let
\(\mathcal L_{\rm eval}\) be the named, physically feasible first-eight-factor
Cartesian lattice frozen independently of outcomes. For axis \(j\), enumerate
every directed edge
\(\mathcal E_j=\{(r,r^+):r,r^+\in\mathcal L_{\rm eval},
r^+_j=2r_j,\ r^+_{-j}=r_{-j}\}\), deduplicate physical aliases, sort by
canonical resource tuple, and assign exact equal weights. The primary axis
contrast is the finite-target mean of its edge differences; its component edge
IDs, weights, boundary `0.02`, and power value `0.10` are materialized before
CAL. For interaction \(j,k\), enumerate every admissible oriented 2×2 doubling
face and use the exact equal-weight finite difference-in-differences; segment
and bottleneck competitors analogously bind named hinge-side edge sets. Missing
edges make that branch ineligible and cannot be silently reweighted. Edge
order, a nonlinear sign-changing surface, or selection of a favorable base
tuple cannot alter the frozen contrast; row permutation is invariant and a
post-fit slice switch stales the family digest. Claims are explicitly scoped
to these finite-lattice averages, with all edgewise effects and sign
heterogeneity reported.

The primary resource law fits \(g_m(r)\), constrained by \(g_m(r_0)=0\), to
these paired within-family log-knee contrasts with **no** free family
intercept, equal weights over the 16 fixed families, and paired-user
uncertainty within family. A rank check rejects a centered design that also
adds a free family intercept. For either held-out
family, its \(r_0\) measurement is a preregistered charged
`normalization_only` anchor: it normalizes that named-family transport test but
cannot train coefficients and is excluded from every RMSE,
prediction-interval coverage, sign, multiplicity, and model-selection
denominator. Its uncertainty and shared covariance are propagated into every
other held-out \(\Delta\log K\). Including/duplicating \(r_0\), assigning it an
infinitesimal interval, or treating anchored deltas as independent must leave
the decision unchanged or fail validation.
Without such an anchor or prospectively frozen predictive family covariates,
absolute unseen-family \(K\) prediction is inadmissible. Multiplying every
\(K\) in one held-out family by a constant must leave the relative resource-law
decision invariant; any absolute-prediction branch must fail or explicitly
consume its anchor.

The claim target enum is frozen as
`future_estimated_knee_fixed_panel_v1`, not an unobserved latent threshold
knee. A holdout target is the delta-log knee that would be re-estimated from an
independent future sample under the exact frozen `n_scaling`, eight-load panel,
probe census, bracket, and first-stage estimator. Consequently RMSE compares
the sealed surface prediction with that future estimator's sampling
distribution, and each prediction interval includes both surface uncertainty
and first-stage target/normalization-anchor estimation noise. Power and every
hierarchical bootstrap replicate regenerate the entire independent target
panel and refit its knee; a plug-in squared error against one noisy
\(\widehat K\), or an interval for latent \(K\) alone, is invalid. The `0.10`
RMSE tolerance is therefore explicitly conditional on this panel design and
cannot be transported to another \(n\), load grid, or probe census. A
zero-surface-error fixture with large first-stage knee noise must worsen both
predictive RMSE and coverage. Switching to a latent-knee target, changing the
panel design, or omitting target-estimation noise changes the target digest and
stales power, provisional fit, unlock, and final validation.

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
shape-constrained nonparametric monotone lattice
```

Let \(x_j=\ln(r_j/r_{0j})\) and center every basis so \(g_m(0)=0\).
Each schema stores the following exact formula and parameter-domain digest:

```text
constant/null:
  g(x) = 0

normalized_cobb_douglas:
  g(x) = sum_j beta_j x_j
  beta_j in [0,4]

min_bottleneck:
  g(x) = min_j {a_j + beta_j x_j} - min_j a_j
  min_j a_j = 0; a_j in [0,4]; beta_j in [0,4]
  exact nonsmooth objective; no temperature/soft-min substitution

generalized_mean:
  g(x) = beta * log((sum_j w_j exp(p*x_j))^(1/p))
  p in frozen grid; beta in [0,4]
  w_j >= 0.05; sum_j w_j = 1

interaction:
  g(x) = sum_j beta_j x_j + sum_{j<k} gamma_jk x_j x_k
  beta_j in [0,4]; gamma_jk in [-4,4]
  every directed doubling edge in L_eval must be nondecreasing

segmented_load_at_one:
  g(x) = sum_j {beta_j_minus min(x_j,0)
                + beta_j_plus max(x_j,0)}
  beta_j_minus,beta_j_plus in [0,4]; continuous at factor 1

shape_constrained:
  g(x) = sum centered univariate I-spline bases
         + six centered rank-2 pairwise tensor bases
  every L_eval doubling edge nondecreasing; g(0)=0; EDF cap below
```

All fits minimize covariance-aware GLS on unique physical responses and their
full propagated first-stage/anchor covariance. Ridge acts only on
`gamma_jk`, the difference
`beta_j_plus-beta_j_minus`, or the I-spline roughness term, as applicable; it
never penalizes the anchored value. Parameter bounds, simplex order,
canonical axis order, exact-nonsmooth global-solver seed, convergence tolerance,
maximum iterations `100000`, and failure code
`INADMISSIBLE_OPTIMIZATION_OR_IDENTIFIABILITY` are schema-bound. A competitor
whose rank, monotonic-edge, convergence, or parameter-domain check fails is
excluded before selection rather than repaired.

Nominal 95% prediction intervals are equal-tail quantiles of the frozen
hierarchical bootstrap that resamples users within fixed blocks, repeats both
knee components, propagates normalization-anchor covariance, refits the
candidate/selector, and adds the independent future-target panel noise. A
delta-method-only or selected-model-conditional interval is inadmissible.
Golden synthetic fixtures recover each formula within `1e-6`; changing a sign,
basis centering, bound, penalty target, generalized-mean normalization,
min-versus-soft-min operation, or PI method changes the formula digest and
stales power/provisional/final artifacts.

The selector is a frozen functional, not “best fit” prose. For each method,
outer leave-one-family-out folds have weight `1/14`; within a fold, unique
physical configurations have equal weight after alias collapse. Inner grouped
folds choose hyperparameters by equal-family
`future_estimated_knee_fixed_panel_v1` delta-log-\(K\) RMSE. Exact grids are:

```text
generalized-mean p = {-8,-4,-2,-1,-0.5,0.5,1,2,4,8}
interaction ridge  = {0,1e-4,1e-3,1e-2,1e-1,1}
segmented ridge    = {0,1e-4,1e-3,1e-2,1e-1,1}; hinge=factor 1 exactly
I-spline roughness = {1e-4,1e-3,1e-2,1e-1,1,10,100}
optimizer tolerance= 1e-8
score-tie tolerance= 1e-12
```

For hyperparameters and then model class, apply the one-standard-error rule to
the 14 equal-weight outer-family RMSE contributions and choose the first
eligible entry in this fixed low-to-high complexity order:
`constant`, `normalized_cobb_douglas`, `min_bottleneck`,
`generalized_mean`, `interaction`, `segmented_load_at_one`,
`shape_constrained`. An A100 relation has no winner unless its simultaneous
RMSE improvement over constant is at least `0.02`; a nonadditive/segmented
branch additionally requires at least `0.02` improvement over normalized
Cobb-Douglas. The selected class and every claimed effect direction must recur
in at least 12 of 14 outer folds; otherwise return
`EMPIRICAL_CURVE_NO_STABLE_WINNER`. Exact-score ties use the complexity order,
then canonical hyperparameter serialization. The provisional/final fit and
every power/bootstrap replicate call this same selector digest. Equal-score,
near-tie, one-family reversal, model-order permutation, and hidden-grid
mutations must either reproduce the frozen winner or return the typed
no-winner state.

Fit a separate resource surface for static mixture and hybrid; a pooled surface
without preregistered method×resource interactions is forbidden. Use nested
grouped cross-validation, the leak-free leave-workload-family-out procedure
above, prediction intervals, and A100 configuration holdouts. Group splits are
based on `workload_family_id` and may never mix
TRAIN/CALIBRATION/TEST templates.
The 16 families are fixed finite-target strata. The two final families test
transport only to those named strata; neither family resampling nor a
workload-generator-population claim is permitted. Bootstrap/wild-bootstrap
uncertainty resamples paired users within strata, and duplicated/split family
labels are invariant.
The shape-constrained competitor is an estimable sparse basis, not an
\(8^4=4096\)-free-node lattice. It uses monotone additive I-splines on the same
four normalized axes plus only the six preregistered rank-2 pairwise tensor
terms. The maximum effective degrees of freedom per method is
`min(24, floor(n_independent_physical_fit_configs/3))`; every fit records
design rank, penalty nullspace, active constraints, and EDF, and is
`INADMISSIBLE_NONIDENTIFIED` if any bound fails. Knot candidates are the eight
fit factors and roughness is selected only inside inner grouped folds. It may
not see factor-4/8 or held-out-family outcomes. Its prediction interval and the
entire constrained fit are recomputed in every hierarchical bootstrap and
power replicate; a 4096-node, rank-deficient, or point-only isotonic fit is
inadmissible.
The 64 first-eight-factor one-axis rows identify marginal shapes; the 32
joint-axis fit rows identify interaction structure. The 16 factor-4/8 one-axis
and eight maximin joint A100 rows are never used in model selection or
coefficient fitting: they evaluate frozen A100-surface predictions and the
exact preregistered sign/regime-reversal predicate only after the provisional
fit is sealed. No interaction law is admissible
unless both methods pass all four joint holdout tuples, each containing factor
4 or 8 on at least two axes. If fewer than four qualifying extreme tuples lie
in the predeclared physical-feasibility domain, if one is omitted, or if the
aggregate budget cannot execute all four, the interaction, segmented, and
nonadditive branches are ineligible; the maximum admissible result
is an additive/marginal A100 relation. Interior substitutions cannot rescue
those branches.

Any contour where two fitted method surfaces happen to cross is
`EXPLORATORY_PHASE_DIAGRAM_ONLY`. This protocol freezes no boundary-location
estimand, uniqueness/transversality condition, displacement margin, or powered
holdout crossing predicate; therefore a selected model's visual contour can
never create a phase-boundary law. A future prospective amendment must define
and power those objects, including no-boundary, multiple-crossing, and
flat-boundary DGPs, before such a claim is eligible.

- [ ] **Step 4: Implement the scaling-law gate**

A relation may be labeled `a100_regime_scaling_law` only if:

- each fitted axis has at least eight fit points;
- every included family×resource×method tuple has a valid first-stage
  \(K_{eff}\);
- joint-axis rank and condition-number checks support the claimed interaction
  terms;
- factors 4 and 8 are both predicted out of sample on every axis and method;
- all four maximin multi-axis tuples are predicted out of sample for both
  methods before any interaction law is admitted;
- all factor-4/factor-8 A100 holdout outcomes are absent from every
  training/tuning digest;
- nested model competition beats the frozen tolerance;
- exponent signs and model selection are stable;
- interval coverage passes;
- all required logical-child references resolve, actual RRE is at most
  \(Q_{\rm capacity,max}\), and
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
signs to reproduce the sealed provisional fit. Then evaluate the 24 untouched
A100 one-axis/joint holdout rows without refitting or calibration. For each method, the
hardware gate requires the simultaneous LCB for
`0.10 - family_level_delta_log_K_RMSE` above zero and the preregistered
hierarchy-aware null-score/concentration simultaneous lower bound calibrated
at \(\alpha_{\rm scaling}=0.025\) of at least 0.90 on empirical coverage by
the nominal-95% prediction intervals. The score keeps workload×paired-seed
blocks fixed, resamples users within block, and uses positive boundary
variance rather than an empirical Wald denominator. In addition,
every claimed axis sign, interaction, segment, and bottleneck
direction must have its preregistered material-margin simultaneous LCB above
zero under the same `SCALING-A100-GLOBAL-{K}` critical value. A wide interval
that merely is not wholly opposite is `NARROWED_INSUFFICIENT_PRECISION`, never
support. Likewise point RMSE 0.099 whose simultaneous uncertainty crosses 0.10
cannot authorize a law. These predicates, including handling of an invalid family-level knee,
are serialized before G2-CAP. Optional H100 transport
measurements use a separate post hoc transport predicate/result block and do
not enter this procedure.
The binomial LCB is used only across independent Monte Carlo replicates in
power, never across correlated holdout points; a correlated-failure fixture
must make a naive binomial pass while the required cluster LCB fails.

Production enforces a one-shot holdout firewall, not merely a role column.
`runtime run-matrix --logical-role a100-fit` is the only first launch. Its
receipt is imported and validated into `scaling-fit-result-validation.json`;
then `scaling fit --stage provisional` writes the immutable
`scaling-provisional.json`. Only `scaling holdout-unlock`, after rehashing the
preregistration, fit receipt/validation, configuration, cohort index, track and
aggregate budgets, code, environment, selected model, hyperparameters,
coefficients, signs, and prediction-interval state, may write the one-time
`scaling-holdout-unlock.json`. The runner rejects `a100-holdout` without that
exact unlock and rejects a combined/mixed-role invocation. Final evaluation
must reproduce the provisional fit digest exactly and may only score the
holdout; model reselection, refitting, threshold tuning, or unlock reuse fails.

- [ ] **Step 5: Materialize the scaling track without running it**

```bash
uv run stc capacity stakes validate \
  --coverage-config configs/stc/design/coverage-scaling.yaml \
  --information-config configs/stc/design/parametric-information.yaml \
  --scaling-config configs/stc/design/scaling.yaml \
  --hardware-config configs/stc/design/scaling-hardware.yaml \
  --estimator-config configs/stc/design/capacity-estimators.yaml \
  --manifest manifests/stc/deployment-stakes-G2-CAP.json \
  --fail-on-config-drift
uv run stc scaling materialize-contrast-family \
  --config configs/stc/design/scaling.yaml \
  --hardware-config configs/stc/design/scaling-hardware.yaml \
  --deployment-stakes manifests/stc/deployment-stakes-G2-CAP.json \
  --alpha 0.025 \
  --output manifests/stc/scaling-contrast-family-G2-CAP.json
uv run stc coverage materialize-contrast-family \
  --config configs/stc/design/coverage-scaling.yaml \
  --selection-bundle manifests/stc/capacity-selection-bundle-G2-CAP.json \
  --deployment-stakes manifests/stc/deployment-stakes-G2-CAP.json \
  --alpha 0.025 \
  --output manifests/stc/coverage-contrast-family-G2-CAP.json
uv run stc coverage power \
  --config configs/stc/design/coverage-scaling.yaml \
  --selection-bundle manifests/stc/capacity-selection-bundle-G2-CAP.json \
  --capacity-calibration-summary manifests/stc/capacity-calibration-summary.json \
  --deployment-stakes manifests/stc/deployment-stakes-G2-CAP.json \
  --contrast-family manifests/stc/coverage-contrast-family-G2-CAP.json \
  --output manifests/stc/coverage-power.json
uv run stc coverage materialize \
  --config configs/stc/design/coverage-scaling.yaml \
  --selection-bundle manifests/stc/capacity-selection-bundle-G2-CAP.json \
  --power manifests/stc/coverage-power.json \
  --cardinality-output manifests/stc/coverage-cardinality-cells.jsonl \
  --fixed-bits-output manifests/stc/coverage-fixed-bits-cells.jsonl \
  --manifest-index-output manifests/stc/coverage-scaling-manifest-index.json \
  --cohort-index-output manifests/stc/coverage-scaling-cohort-index.json \
  --budget-output manifests/stc/coverage-scaling-budget.json
uv run stc information power \
  --config configs/stc/design/parametric-information.yaml \
  --selection-bundle manifests/stc/capacity-selection-bundle-G2-CAP.json \
  --capacity-calibration-summary manifests/stc/capacity-calibration-summary.json \
  --deployment-stakes manifests/stc/deployment-stakes-G2-CAP.json \
  --output manifests/stc/parametric-information-power.json
uv run stc information materialize \
  --config configs/stc/design/parametric-information.yaml \
  --selection-bundle manifests/stc/capacity-selection-bundle-G2-CAP.json \
  --power manifests/stc/parametric-information-power.json \
  --output manifests/stc/parametric-information-cells.jsonl \
  --cohort-index-output manifests/stc/parametric-information-cohort-index.json \
  --budget-output manifests/stc/parametric-information-budget.json
uv run stc scaling power \
  --config configs/stc/design/scaling.yaml \
  --hardware-config configs/stc/design/scaling-hardware.yaml \
  --selection-bundle manifests/stc/selection-bundle-G2.json \
  --capacity-calibration-summary manifests/stc/capacity-calibration-summary.json \
  --deployment-stakes manifests/stc/deployment-stakes-G2-CAP.json \
  --contrast-family manifests/stc/scaling-contrast-family-G2-CAP.json \
  --output manifests/stc/scaling-power.json
uv run stc scaling materialize \
  --config configs/stc/design/scaling.yaml \
  --selection-bundle manifests/stc/selection-bundle-G2.json \
  --power manifests/stc/scaling-power.json \
  --hardware-config configs/stc/design/scaling-hardware.yaml \
  --confirmatory-summary manifests/stc/confirmatory-summary.json \
  --pilot manifests/stc/blinded-pilot-summary.json \
  --capacity-calibration-summary manifests/stc/capacity-calibration-summary.json \
  --cost-profile configs/stc/design/cost-profile.yaml \
  --output manifests/stc/scaling-cells.jsonl \
  --cohort-index-output manifests/stc/scaling-cohort-index.json \
  --budget-output manifests/stc/scaling-budget.json
uv run stc capacity budget-freeze \
  --scaling-budget manifests/stc/scaling-budget.json \
  --coverage-budget manifests/stc/coverage-scaling-budget.json \
  --parametric-information-budget manifests/stc/parametric-information-budget.json \
  --program-budget-envelope configs/stc/design/program-budget-envelope.yaml \
  --output manifests/stc/capacity-aggregate-budget.json
```

Expected:

```text
coverage_cardinality_logical_total = 40
coverage_fixed_bits_logical_total = 120
coverage_fit_sizes = 8
coverage_holdout_sizes = 2
coverage_heldout_workload_families = 2
coverage_heldout_family_ids = exact final_workload_holdout IDs
coverage_heldout_family_role_and_split_digest = preregistered digest
coverage_logical_child_references = 160 * n_coverage
parametric_information_logical_total = 48
parametric_information_seed_ensemble = n_information
parametric_information_logical_child_references = 48 * n_information
logical_total = 120
one_axis_logical_total = 80
joint_axis_fit_logical_total = 32
joint_axis_holdout_logical_total = 8
d_opt_model_columns = 10
d_opt_selected_rows = 16
d_opt_rank = 10
joint_fit_factor_4_or_8_rows = 0
joint_holdout_multi_axis_rows = 8
a100_fit_logical_rows = 96
a100_one_axis_holdout_logical_rows = 16
a100_joint_holdout_logical_rows = 8
a100_holdout_logical_rows = 24
a100_fit_config_fit_family_children = 96 * n_scaling * 14 / 16
family_only_holdout_children = 96 * n_scaling * 2 / 16
resource_only_holdout_children = 24 * n_scaling * 14 / 16
double_holdout_children = 24 * n_scaling * 2 / 16
total_logical_child_references = 120 * n_scaling
child_references_per_logical_per_family = n_scaling / 16
duplicate_child_keys = 0
missing_child_keys = 0
extra_child_keys = 0
charged_anchor_keys = 4
unique_physical_executions <= 120 * n_scaling + 4
projected_joint_upper_99_scaling <= q_scaling
projected_joint_upper_99_coverage <= q_coverage
projected_joint_upper_99_information <= q_information
q_scaling + q_coverage + q_information <= Q_capacity,max
sum(projected_joint_upper_99_by_track) <= Q_capacity,max
for every native resource/carbon component r:
  sum(projected_joint_upper_99_by_track[r]) <= sponsor_absolute_cap[r]
hard_global_actual_rre_ceiling = Q_capacity,max
hard_global_native_component_ceiling[r] = sponsor_absolute_cap[r]
```

Each materializer writes its own outputs atomically. The two coverage
manifests must have disjoint arm IDs and incompatible codec-budget invariants;
the coverage manifest/cohort indexes resolve all \(160n_{\rm coverage}\)
children. The information manifest and cohort index resolve all
\(48n_{\rm information}\) children and bind fresh-retraining, payload-seed,
decoder, estimator, complete-state inventory, and dedicated power digests. The
three core scaling outputs are one atomic materialization. The cells file records all
120 logical identities; the cohort index records all
\(120n_{\rm scaling}\) required A100
child references, the logical→physical incidence/covariance map, and the
96-fit/24-holdout role split. The budget records every unique physical cohort's
blinded upper-bound weight and its non-borrowable subquota. The aggregate
envelope binds all three budget digests, quotas, frozen track/cohort order, and
the sponsor-frozen \(Q_{\rm capacity,max}\) plus componentwise absolute
GPU/CPU/memory-time/I/O/network/energy/carbon ceilings. Scalar RRE passing is
insufficient: both the summed simultaneous upper-99 native vectors and the
canonical ledger's measured-plus-unreconciled-reservation component totals
must fit every absolute cap. A fixture whose individually valid track budgets
exceed the scalar aggregate, or whose network/carbon component exceeds its cap
while RRE passes, must fail the command.
`test_scaling_materialize.py` independently
reconstructs the fit-only Cartesian candidate lattice and D-opt matrix,
verifies the exact child equations and deduplication map, and rejects a
materializer that emits only 120 logical placeholders, omits a joint holdout,
selects a joint holdout after outcomes, or inserts an H100 child.
No output may execute until its digest is in the G2 preregistration below.

- [ ] **Step 6: Verify and freeze G2/G3 inputs**

```bash
uv run pytest \
  tests/stc/analysis/test_scaling.py \
  tests/stc/analysis/test_scaling_materialize.py \
  tests/stc/analysis/test_coverage_scaling.py \
  tests/stc/analysis/test_parametric_information.py \
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
  --scope core \
  --output manifests/stc/theory-laws-G2.json
uv run stc theory verify-hypotheses \
  configs/stc/theory/laws.yaml \
  --scope capacity \
  --scaling-config configs/stc/design/scaling.yaml \
  --output manifests/stc/theory-laws-G2-CAP.json
uv run stc prereg freeze \
  --design manifests/stc/confirmatory.jsonl \
  --run-index manifests/stc/confirmatory-run-index.json \
  --power manifests/stc/power.json \
  --analysis manifests/stc/analysis-plan.json \
  --training-calibration-plan manifests/stc/training-calibration-core-plan.json \
  --training-calibration-manifest manifests/stc/training-calibration-core-cells.jsonl \
  --training-calibration-cohort-index manifests/stc/training-calibration-core-cohort-index.json \
  --training-calibration-budget manifests/stc/training-calibration-core-budget.json \
  --design-development-budget manifests/stc/design-development-budget.json \
  --calibration-readiness manifests/stc/training-calibration-core-readiness.json \
  --training-calibration-receipt manifests/systems/training-calibration-core-receipt.json \
  --training-calibration-validation manifests/stc/training-calibration-core-result-validation.json \
  --selection-bundle manifests/stc/selection-bundle-G2.json \
  --pilot-receipt manifests/systems/pilot-receipt.json \
  --pilot-validation manifests/stc/pilot-result-validation.json \
  --comparator-maps manifests/stc/comparator-maps-G2.json \
  --calibration-baseline-summary manifests/stc/calibration-baseline-summary.json \
  --comparator-selection-config configs/stc/design/comparator-selection.yaml \
  --deployment-stakes manifests/stc/deployment-stakes-core.json \
  --confirmatory-summary manifests/stc/confirmatory-summary.json \
  --confirmatory-budget manifests/stc/confirmatory-budget.json \
  --program-budget-envelope configs/stc/design/program-budget-envelope.yaml \
  --rre-reference configs/stc/design/rre-reference-native.yaml \
  --coarse-feasibility manifests/stc/capacity-coarse-feasibility.json \
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
  --confirmatory-budget manifests/stc/confirmatory-budget.json \
  --program-budget-envelope configs/stc/design/program-budget-envelope.yaml \
  --rre-reference configs/stc/design/rre-reference-native.yaml \
  --coarse-feasibility manifests/stc/capacity-coarse-feasibility.json \
  --analysis-plan manifests/stc/analysis-plan.json \
  --training-calibration-plan manifests/stc/training-calibration-core-plan.json \
  --training-calibration-manifest manifests/stc/training-calibration-core-cells.jsonl \
  --training-calibration-cohort-index manifests/stc/training-calibration-core-cohort-index.json \
  --training-calibration-budget manifests/stc/training-calibration-core-budget.json \
  --design-development-budget manifests/stc/design-development-budget.json \
  --calibration-readiness manifests/stc/training-calibration-core-readiness.json \
  --training-calibration-receipt manifests/systems/training-calibration-core-receipt.json \
  --training-calibration-validation manifests/stc/training-calibration-core-result-validation.json \
  --selection-bundle manifests/stc/selection-bundle-G2.json \
  --pilot-receipt manifests/systems/pilot-receipt.json \
  --pilot-validation manifests/stc/pilot-result-validation.json \
  --comparator-maps manifests/stc/comparator-maps-G2.json \
  --calibration-baseline-summary manifests/stc/calibration-baseline-summary.json \
  --comparator-selection-config configs/stc/design/comparator-selection.yaml \
  --model-lock configs/stc/models.lock.yaml \
  --model-upstreams manifests/stc/model-upstreams.json \
  --upstreams configs/stc/upstreams.lock.yaml \
  --public-replay manifests/stc/public-replay.jsonl \
  --generator-property-report manifests/stc/generator-property-report.json \
  --output manifests/gate-inputs/G2.json
uv run stc gate evaluate G2 \
  --inputs manifests/gate-inputs/G2.json \
  --output manifests/gates/G2.json
uv run stc prereg freeze-capacity \
  --core-preregistration manifests/stc/preregistration.json \
  --predecessor manifests/gates/G2.json \
  --capacity-selection-plan manifests/stc/capacity-selection-plan.json \
  --capacity-selection-manifest manifests/stc/capacity-selection-cells.jsonl \
  --capacity-selection-cohort-index manifests/stc/capacity-selection-cohort-index.json \
  --capacity-selection-budget manifests/stc/capacity-selection-budget.json \
  --capacity-selection-readiness manifests/stc/capacity-selection-readiness.json \
  --capacity-selection-receipt manifests/systems/capacity-selection-receipt.json \
  --capacity-selection-validation manifests/stc/capacity-selection-result-validation.json \
  --capacity-selection-bundle manifests/stc/capacity-selection-bundle-G2-CAP.json \
  --capacity-power-plan manifests/stc/capacity-power-calibration-plan.json \
  --capacity-power-manifest manifests/stc/capacity-power-calibration-cells.jsonl \
  --capacity-power-cohort-index manifests/stc/capacity-power-calibration-cohort-index.json \
  --capacity-power-budget manifests/stc/capacity-power-calibration-budget.json \
  --capacity-power-readiness manifests/stc/capacity-power-calibration-readiness.json \
  --capacity-power-receipt manifests/systems/capacity-power-calibration-receipt.json \
  --capacity-power-validation manifests/stc/capacity-power-calibration-result-validation.json \
  --capacity-calibration-summary manifests/stc/capacity-calibration-summary.json \
  --b-ref-summary manifests/stc/blinded-pilot-summary.json \
  --b-ref-validation manifests/stc/pilot-result-validation.json \
  --deployment-stakes manifests/stc/deployment-stakes-G2-CAP.json \
  --theory-laws manifests/stc/theory-laws-G2-CAP.json \
  --capacity-phase-plan configs/systems/capacity-production-phases.yaml \
  --scaling manifests/stc/scaling-cells.jsonl \
  --scaling-config configs/stc/design/scaling.yaml \
  --scaling-hardware-config configs/stc/design/scaling-hardware.yaml \
  --scaling-power manifests/stc/scaling-power.json \
  --scaling-contrast-family manifests/stc/scaling-contrast-family-G2-CAP.json \
  --scaling-cohort-index manifests/stc/scaling-cohort-index.json \
  --scaling-budget manifests/stc/scaling-budget.json \
  --coverage-cardinality manifests/stc/coverage-cardinality-cells.jsonl \
  --coverage-fixed-bits manifests/stc/coverage-fixed-bits-cells.jsonl \
  --coverage-manifest-index manifests/stc/coverage-scaling-manifest-index.json \
  --coverage-cohort-index manifests/stc/coverage-scaling-cohort-index.json \
  --coverage-power manifests/stc/coverage-power.json \
  --coverage-contrast-family manifests/stc/coverage-contrast-family-G2-CAP.json \
  --coverage-config configs/stc/design/coverage-scaling.yaml \
  --coverage-budget manifests/stc/coverage-scaling-budget.json \
  --parametric-information manifests/stc/parametric-information-cells.jsonl \
  --parametric-information-cohort-index manifests/stc/parametric-information-cohort-index.json \
  --parametric-information-power manifests/stc/parametric-information-power.json \
  --parametric-information-config configs/stc/design/parametric-information.yaml \
  --parametric-information-budget manifests/stc/parametric-information-budget.json \
  --capacity-aggregate-budget manifests/stc/capacity-aggregate-budget.json \
  --program-budget-envelope configs/stc/design/program-budget-envelope.yaml \
  --rre-reference configs/stc/design/rre-reference-native.yaml \
  --coarse-feasibility manifests/stc/capacity-coarse-feasibility.json \
  --runtime manifests/systems/runtime-G2.json \
  --output manifests/stc/capacity-preregistration.json
uv run stc prereg validate \
  --manifest manifests/stc/capacity-preregistration.json \
  --expected-predecessor manifests/gates/G2.json \
  --fail-on-upstream-digest-change
uv run stc capacity gate-input assemble G2-CAP \
  --root . \
  --predecessor manifests/gates/G2.json \
  --capacity-preregistration manifests/stc/capacity-preregistration.json \
  --capacity-aggregate-budget manifests/stc/capacity-aggregate-budget.json \
  --program-budget-envelope configs/stc/design/program-budget-envelope.yaml \
  --rre-reference configs/stc/design/rre-reference-native.yaml \
  --coarse-feasibility manifests/stc/capacity-coarse-feasibility.json \
  --runtime manifests/systems/runtime-G2.json \
  --output manifests/capacity-gate-inputs/G2-CAP.json
uv run stc capacity gate evaluate G2-CAP \
  --inputs manifests/capacity-gate-inputs/G2-CAP.json \
  --output manifests/capacity-gates/G2-CAP.json
```

Expected: the provisional supported fixture is
`empirical_curve_provisional`, the malformed fixture is
`invalid_scaling_input`, and only the final fixture can be
`a100_regime_scaling_law`.
Every fit reports outer-family-fold RMSE/MAE/coverage, exponents and signs,
model-selection stability, untouched A100 holdout metrics, child
coverage, covariance-map compliance, projected RRE, and actual RRE. A
`--stage final` call without the A100 holdout input, with a nonreproducible
fit-stage digest, or with an A100 holdout row in training fails closed.

The core G2 input assembler validates only the
297-logical/\(297n\)-child confirmatory summary; G1-descended core
TRAIN/CAL plan, readiness, manifest, budget, receipt, validation, and
selection bundle; every selected/losing-trial search-log edge; the
\(\phi\)-only comparator maps; global multiplicity table; theory/runtime
semantics; public/model upstreams; cost prices; retry reserves; and analysis
code digests. A capacity artifact in the core preregistration or G2 input is an
unexpected-input failure.

The domain child `G2-CAP` separately validates PASS G2, the capacity
TRAIN/CAL projection and selection/calibration artifacts, the
120-logical/\(120n_{\rm scaling}\)-child index, fit-only D-opt and pre-outcome
maximin holdout digests, dedicated scaling power, the 40/120 coverage manifests
with \(160n_{\rm coverage}\) children, the 48-cell information manifest with
\(48n_{\rm information}\) children, all three non-borrowable subbudgets, and
the aggregate envelope. It rejects unless every track projection fits its
quota and both quota/projection sums are at most \(Q_{\rm capacity,max}\).
`G2-CAP=FAIL` blocks only
capacity capabilities and claims; core PAPER-C remains eligible and no
capacity result can rescue it.
G2 rejects an empirical H-STC support/falsification field; later empirical
decisions require a distinct powered amendment, execution receipts, result
validation, and post-TEST decision artifact.
The generic gate engine emits the
canonical typed `GateRecord` with `gate_id=G2`, predecessor/input digests,
named check results, `gate_status=PASS|FAIL`, frozen evaluation timestamp, and
record digest; a domain command may not hand-write it.

`test_g2_gate.py` mutates each core input path and digest one at a time,
including both sides of the model-lock→model-upstreams and
upstreams-lock→public-replay edges, every preregistration constituent,
hypothesis registry, executable theory laws, confirmatory summary, runtime,
generator property report, and
G1 predecessor. It also mutates comparator-map identity/self-digest,
CALIBRATION source/split/summary, selection config/code, cost profile, and
tries to add an \(R\) or TEST-family keyed entry; every case fails. It removes
a losing trial, changes a selection functional/tie-break/search budget, injects
a TEST/derivative-overlap URI, hand-authors a selected config, drifts the
TRAIN/CAL code/environment, or borrows its development budget; every case
fails. It also injects each capacity artifact into the core envelope and
requires `CORE_G2_UNEXPECTED_CAPACITY_INPUT`.

`test_g2_cap_gate.py` mutates scaling config/hardware/cells/power/cohort/budget,
each coverage arm/config/power/manifest-index/cohort-index/budget, and every
parametric-information manifest/config/power/cohort/budget edge, including
decoder, payload generator, estimator, fresh base, auxiliary inventory, and
held-out splits. It tests three individually passing 100-RRE budgets, quota
borrowing, aggregate-ledger reorder/truncation, uncharged retries, stale
reservations, a swapped core/capacity preregistration, and a changed PASS G2
predecessor. Every case returns a typed `CAP_G2_FAIL` while rederiving the same
core G2 bytes.

`test_preregistration.py` separately mutates the bytes, recorded digest, and
path of every core or capacity freeze argument, adds an unregistered input,
removes one registered input, swaps envelope types, and permutes invocation
argument order. Validation must
fail for every semantic mutation; argument-order permutations and two freezes
over byte-identical inputs must be byte-identical and retain the canonical
ordered artifact table.

After all required production A100 fit and holdout cohorts have been measured,
and without waiting for optional H100 work, the systems
execution plan must invoke this exact final-refit hook before G5:

```bash
uv run stc results validate \
  --capacity-preregistration manifests/stc/capacity-preregistration.json \
  --capacity-gate manifests/capacity-gates/G2-CAP.json \
  --manifest manifests/stc/coverage-cardinality-cells.jsonl \
  --cohort-index manifests/stc/coverage-scaling-cohort-index.json \
  --logical-arm cardinality \
  --execution-role fit \
  --budget manifests/stc/coverage-scaling-budget.json \
  --aggregate-budget manifests/stc/capacity-aggregate-budget.json \
  --actual-ledger results/capacity/actual-rre-ledger.jsonl \
  --results results/coverage-scaling/cardinality/fit \
  --output manifests/stc/coverage-cardinality-fit-result-validation.json
uv run stc results validate \
  --capacity-preregistration manifests/stc/capacity-preregistration.json \
  --capacity-gate manifests/capacity-gates/G2-CAP.json \
  --manifest manifests/stc/coverage-fixed-bits-cells.jsonl \
  --cohort-index manifests/stc/coverage-scaling-cohort-index.json \
  --logical-arm fixed-bits \
  --execution-role fit \
  --budget manifests/stc/coverage-scaling-budget.json \
  --aggregate-budget manifests/stc/capacity-aggregate-budget.json \
  --actual-ledger results/capacity/actual-rre-ledger.jsonl \
  --results results/coverage-scaling/fixed-bits/fit \
  --output manifests/stc/coverage-fixed-bits-fit-result-validation.json
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
uv run stc results validate \
  --capacity-preregistration manifests/stc/capacity-preregistration.json \
  --capacity-gate manifests/capacity-gates/G2-CAP.json \
  --manifest manifests/stc/coverage-cardinality-cells.jsonl \
  --cohort-index manifests/stc/coverage-scaling-cohort-index.json \
  --logical-arm cardinality \
  --execution-role holdout \
  --budget manifests/stc/coverage-scaling-budget.json \
  --aggregate-budget manifests/stc/capacity-aggregate-budget.json \
  --actual-ledger results/capacity/actual-rre-ledger.jsonl \
  --provisional-artifact manifests/stc/coverage-provisional.json \
  --holdout-unlock manifests/stc/coverage-holdout-unlock.json \
  --results results/coverage-scaling/cardinality/holdout \
  --output manifests/stc/coverage-cardinality-holdout-result-validation.json
uv run stc results validate \
  --capacity-preregistration manifests/stc/capacity-preregistration.json \
  --capacity-gate manifests/capacity-gates/G2-CAP.json \
  --manifest manifests/stc/coverage-fixed-bits-cells.jsonl \
  --cohort-index manifests/stc/coverage-scaling-cohort-index.json \
  --logical-arm fixed-bits \
  --execution-role holdout \
  --budget manifests/stc/coverage-scaling-budget.json \
  --aggregate-budget manifests/stc/capacity-aggregate-budget.json \
  --actual-ledger results/capacity/actual-rre-ledger.jsonl \
  --provisional-artifact manifests/stc/coverage-provisional.json \
  --holdout-unlock manifests/stc/coverage-holdout-unlock.json \
  --results results/coverage-scaling/fixed-bits/holdout \
  --output manifests/stc/coverage-fixed-bits-holdout-result-validation.json
uv run stc coverage fit \
  --stage final \
  --capacity-preregistration manifests/stc/capacity-preregistration.json \
  --capacity-gate manifests/capacity-gates/G2-CAP.json \
  --config configs/stc/design/coverage-scaling.yaml \
  --cohort-index manifests/stc/coverage-scaling-cohort-index.json \
  --provisional-artifact manifests/stc/coverage-provisional.json \
  --holdout-unlock manifests/stc/coverage-holdout-unlock.json \
  --cardinality-fit-results results/coverage-scaling/cardinality/fit \
  --cardinality-holdout-results results/coverage-scaling/cardinality/holdout \
  --fixed-bits-fit-results results/coverage-scaling/fixed-bits/fit \
  --fixed-bits-holdout-results results/coverage-scaling/fixed-bits/holdout \
  --cardinality-fit-validation manifests/stc/coverage-cardinality-fit-result-validation.json \
  --cardinality-holdout-validation manifests/stc/coverage-cardinality-holdout-result-validation.json \
  --fixed-bits-fit-validation manifests/stc/coverage-fixed-bits-fit-result-validation.json \
  --fixed-bits-holdout-validation manifests/stc/coverage-fixed-bits-holdout-result-validation.json \
  --output manifests/stc/coverage-scaling-summary.json
uv run stc results validate \
  --preregistration manifests/stc/capacity-preregistration.json \
  --capacity-gate manifests/capacity-gates/G2-CAP.json \
  --manifest manifests/stc/parametric-information-cells.jsonl \
  --cohort-index manifests/stc/parametric-information-cohort-index.json \
  --budget manifests/stc/parametric-information-budget.json \
  --aggregate-budget manifests/stc/capacity-aggregate-budget.json \
  --actual-ledger results/capacity/actual-rre-ledger.jsonl \
  --results results/parametric-information \
  --output manifests/stc/parametric-information-result-validation.json
uv run stc information estimate \
  --preregistration manifests/stc/capacity-preregistration.json \
  --capacity-gate manifests/capacity-gates/G2-CAP.json \
  --config configs/stc/design/parametric-information.yaml \
  --manifest manifests/stc/parametric-information-cells.jsonl \
  --cohort-index manifests/stc/parametric-information-cohort-index.json \
  --budget manifests/stc/parametric-information-budget.json \
  --aggregate-budget manifests/stc/capacity-aggregate-budget.json \
  --validation-report manifests/stc/parametric-information-result-validation.json \
  --results results/parametric-information \
  --output manifests/stc/parametric-information-summary.json
uv run stc results validate \
  --preregistration manifests/stc/capacity-preregistration.json \
  --capacity-gate manifests/capacity-gates/G2-CAP.json \
  --manifest manifests/stc/scaling-cells.jsonl \
  --cohort-index manifests/stc/scaling-cohort-index.json \
  --budget manifests/stc/scaling-budget.json \
  --aggregate-budget manifests/stc/capacity-aggregate-budget.json \
  --actual-ledger results/capacity/actual-rre-ledger.jsonl \
  --execution-role fit \
  --results results/scaling/a100-fit \
  --output manifests/stc/scaling-fit-result-validation.json
uv run stc results validate \
  --capacity-preregistration manifests/stc/capacity-preregistration.json \
  --capacity-gate manifests/capacity-gates/G2-CAP.json \
  --manifest manifests/stc/scaling-cells.jsonl \
  --cohort-index manifests/stc/scaling-cohort-index.json \
  --budget manifests/stc/scaling-budget.json \
  --aggregate-budget manifests/stc/capacity-aggregate-budget.json \
  --actual-ledger results/capacity/actual-rre-ledger.jsonl \
  --execution-role holdout \
  --provisional-artifact manifests/stc/scaling-provisional.json \
  --holdout-unlock manifests/stc/scaling-holdout-unlock.json \
  --results results/scaling/a100-holdout \
  --output manifests/stc/scaling-holdout-result-validation.json
uv run stc scaling fit \
  --stage final \
  --config configs/stc/design/scaling.yaml \
  --hardware-config configs/stc/design/scaling-hardware.yaml \
  --cohort-index manifests/stc/scaling-cohort-index.json \
  --budget manifests/stc/scaling-budget.json \
  --aggregate-budget manifests/stc/capacity-aggregate-budget.json \
  --preregistration manifests/stc/capacity-preregistration.json \
  --capacity-gate manifests/capacity-gates/G2-CAP.json \
  --fit-validation-report manifests/stc/scaling-fit-result-validation.json \
  --holdout-validation-report manifests/stc/scaling-holdout-result-validation.json \
  --provisional-artifact manifests/stc/scaling-provisional.json \
  --holdout-unlock manifests/stc/scaling-holdout-unlock.json \
  --fit-results results/scaling/a100-fit \
  --holdout-results results/scaling/a100-holdout \
  --output manifests/stc/scaling-summary.json
```

Missing measured A100 holdout validation, incomplete required child coverage,
or actual RRE above \(Q_{\rm capacity,max}\) forces `empirical_curve`; it cannot be waived into a
cross-hardware or systems scaling law. Optional H100 external-transport results, if later
collected, are validated and published under a separate manifest/result block
and never modify `scaling-summary.json`.

The coverage command refuses to pool its two arms and emits no cardinality
exponent unless fixed-codec, two-size extrapolation, held-out-workload, and
codec-distortion checks pass. The information command refuses any reused
training seed/checkpoint, mutable decoder, uncharged answer-bearing auxiliary
state, or conditioning variable outside public \(\Theta_0\); it publishes
theta-only and complete-state estimands separately. A failure in either track
narrows its capacity claim and cannot be rescued by the 120-cell resource
surface.

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
