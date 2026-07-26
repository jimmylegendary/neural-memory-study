# Sleep-Time Compute Benchmark and Experiment Blueprint

- Cutoff: 2026-07-25
- Status: pre-registration blueprint; no experimental outcome is implied
- Detailed execution source:
  `docs/superpowers/plans/2026-07-25-sleep-time-compute-theory-benchmark.md`
- Research contract:
  `docs/superpowers/specs/2026-07-25-sleep-time-compute-research-design.md`

## 1. Decision the benchmark must support

Given the same accumulated wake events and future queries, should a system:

```text
retain exact external evidence
compress/merge external evidence
compile reusable latent/KV state
compile a user-local adapter
deliberately forget/no-op
route dynamically among these actions
```

The benchmark must determine whether a past-only router improves the
held-out lifetime utility/lifecycle-cost frontier over tuned single-medium and
static-mixture baselines. It must also quantify null results and boundaries.

It is not designed to prove:

- that biological sleep stages map to model operators;
- that one memory medium is universally best;
- that simulated infrastructure is measured hardware;
- that a 128-cycle stream establishes production lifetime reliability;
- that successful recall proves physical deletion or unlearning.

## 2. Unit hierarchy and cycle

```text
user
└── cycle
    ├── session
    │   └── event
    ├── pre-sleep query block
    ├── sleep candidate construction
    ├── post-sleep query block
    └── delayed next-wake query block
```

One cycle is:

```text
wake capture
→ pre-sleep probe
→ immutable snapshot
→ sleep/no-sleep action
→ candidate validation and promotion
→ post-sleep probe
→ delayed next-wake probe
```

Default horizon is 128 cycles. Horizon sensitivity uses
\(C\in\{8,32,128,512\}\). Retrieval lags are
\(\{1,8,32,128\}\) cycles where admissible.

## 3. Observable and oracle separation

The generator creates two physically separate artifacts.

### 3.1 Observable stream

Available to the method at the declared time:

```text
opaque event and source IDs
event type and payload
event/ingest time
currently visible confidence
privacy scope
declared provenance
currently known supersession
currently known delete/poison flags
past reuse and outcomes
resource and queue state
```

### 3.2 Oracle ledger

Never visible to the learner/router:

```text
true valid-from and valid-to
future reuse count and query distribution
future contradiction/correction
future deletion
poison truth
ground-truth dependencies
exact query answers
clairvoyant best destination/action
```

Files, schemas, API objects, and process permissions keep them separate.
Tests attempt direct and transitive future leakage.

Any MemMA-style diagnostic probe is a learner-visible **derived training
artifact**, not an evaluation query. Its generator receives only the immutable
observable snapshot; generator code/model/prompt, input digests, seed,
candidate count, filters, accepted probes, and rejected probes are logged.
Property tests fail the run if a probe or repair contains an oracle answer,
future event ID, evaluation-query hash, or transitive derivative of them.

### 3.3 Tuning and final-evaluation split

Before any method is fit, each
`workload-family × user × opaque-payload-seed` unit is assigned to disjoint
TRAIN, CALIBRATION, or TEST partitions. TRAIN fits learned routers and
compilers. CALIBRATION alone selects static action proportions/cadence,
single-medium hyperparameters, early stopping, router architecture and
thresholds, merge policy, residency policy, and comparator rules. All search
trials and their resource use are logged.

TEST is executed once after G2 freeze and cannot select a baseline, router,
threshold, stopping point, comparator, or claim wording. C2 uses the
CALIBRATION-selected benefit and efficiency maps frozen at G2. C1 and C3
evaluate every direct component under one simultaneous family and derive
envelope bounds algebraically; they never resample a TEST max/argmax.

Generator \(R\) and held-out workload-family identity are orchestration and
analysis fields, never deployable-method inputs. Static-mixture and
single-medium configurations are pooled across CALIBRATION families and may
vary only with the public resource envelope, base/model, and hardware digests;
their configuration digest is invariant across \(R\) and TEST family.
`MethodRuntimeSpec` strips both fields before construction. Mutation tests hold
the observable stream/caps fixed, change \(R\) or family ID, and require
identical configurations and actions; only the hybrid's chronological
past-observable features may alter its decisions.

## 4. Controlled event families

### 4.1 Stable exact facts

- unique key and exact value;
- paraphrase and application queries;
- independent frequency and delay;
- one opaque-payload subfamily whose salted keys and uniformly sampled bit
  strings make admitted conditional entropy known;
- deterministic encodings of the same payload at multiple surface lengths, so
  semantic information load is not confounded with text bytes;
- optional composition with other facts.

### 4.2 Volatile facts and preferences

- explicit validity interval;
- supersession and correction events;
- temporal queries asking old and current values;
- different volatility hazards.

### 4.3 Procedures and skills

- multi-step executable outcome;
- reusable and one-off variants;
- success, failure, and partial trajectories;
- environment version and tool dependencies.

### 4.4 Relations and graph events

- entity aliases;
- temporal edges;
- contradictions and invalidation;
- multi-hop queries.

### 4.5 Negative feedback and failures

- failed action and reason;
- corrected procedure;
- tempting but wrong repeated behavior;
- future avoidance query.

### 4.6 Privacy and deletion

- user/source scope;
- consent expiry;
- delete time and dependency fan-out;
- post-delete access, lineage, and residual probes.

### 4.7 Poison and adversarial events

- prompt injection;
- malicious preference/fact;
- conflicting high-frequency source;
- delayed poison label hidden from the method.

### 4.8 One-off distractors

- low future reuse;
- high surface salience;
- large payload variants;
- controls unnecessary consolidation.

Reuse, volatility, poison, deletion, payload size, and memory pressure vary
independently. Generator property tests reject accidental correlation.
For the opaque-payload subfamily they additionally reject duplicate payloads,
key leakage, compressible templates, and any seed reuse across train,
calibration, and final holdout streams.

## 5. Query families

| Family | Required evidence | Main failure |
|---|---|---|
| exact | canonical value | summary/detail loss |
| semantic | meaning under paraphrase | brittle string matching |
| temporal | correct value at time | overwrite/version error |
| relational | one- and multi-hop relation | graph/entity collision |
| procedural | executable later success | vague summary |
| negative/failure | what not to do and why | success-only bias |
| calibration | answer or abstain | false memory |
| deletion | no authorized access and bounded residual | tombstone-only claim |

Exact, temporal, and executable outcomes are confirmatory where deterministic.
Model-based semantic judges and human review are secondary validity tracks.
The writer and grader cannot be the same model/configuration without being
reported as correlated evidence.

Every reusable item also receives a frozen query-relationship label:

```text
ACQUISITION_KEY       exact query that caused the initial write
EXACT_REPEAT          byte/semantic duplicate of that query
HELDOUT_PARAPHRASE    unseen wording, same required memory
HELDOUT_COMPOSITION   unseen combination of two or more memories
CORRECTION_PROBE      requires supersession or temporal precedence
UNRELATED_CONTROL     shares category or vocabulary but not the answer
```

Lifecycle phases never reuse an `ACQUISITION_KEY` as evidence of semantic
transfer. Exact-repeat savings and held-out transfer are separate endpoints.

## 6. Information timing

Every method receives the same enqueue-time information cutoff. Deferred sleep
does not see events, queries, corrections, or labels that arrived after its
snapshot unless an explicitly exploratory information-accumulation condition
is run.

This distinction separates:

```text
effect of execution timing
from
effect of receiving more information
```

The primary schedule/operator factorial changes time while holding destination,
source set, resource cap, and information cutoff constant.

## 7. Confirmatory method family

### 7.1 Deployable methods

1. **Raw external:** exact versioned evidence and frozen retriever.
2. **Compressed external:** one frozen merge/abstraction operator.
3. **Latent/KV compiler:** one architecture-compatible Cartridges-style
   artifact.
4. **User-parametric compiler:** one LoRA destination.
5. **Static mixture:** tuned action proportions and cadence, assigned by an
   opaque-ID deterministic hash; no event content or future labels.
6. **Past-only hybrid router:** event-conditional action using only observable
   history.

### 7.2 Diagnostic method

7. **Clairvoyant oracle router:** future-aware upper bound for the value of
   heterogeneity. It is never described as deployable.

### 7.3 Excluded from primary dominance

- no-memory lower diagnostic;
- full-history non-budgeted upper diagnostic;
- proprietary products with unmatched information/cost boundaries;
- second latent implementation;
- graph memory;
- evolved/search-selected memory harness;
- shared-weight promotion;
- recurrent architecture-native fast weights.

They remain characterization or preregistered extensions and cannot enter the
primary comparison after results are seen.

## 8. Resource envelope

Methods receive the same upper bounds rather than being forced to waste equal
resources:

```text
live token cap
active-byte cap
durable-byte cap
total incremental peak-state byte cap
physical HBM/fleet peak cap
sleep-FLOP cap
peak accelerator cap
information cutoff
privacy/delete contract
```

The total incremental peak-state cap is
\(B_{\mathrm{peak,state}}=B_{\mathrm{retained}}+\) disjoint candidate/staging,
reader-pinned predecessor, migration, workspace, and temporary-replica
additions. Transient overhead
\(B_{\mathrm{peak,state}}-B_{\mathrm{retained}}\) is a reported diagnostic,
not a replacement cap. The physical cap is
\[
B_{\mathrm{peak,physical}}=\max_t\left[
B_{\mathrm{base,resident}}(t)+B_{\mathrm{peak,state}}(t)
+B_{\mathrm{execution}}(t)\right],
\]
so separate wake/sleep fleets charge duplicated base copies and runtime
memory. All three terms are evaluated at the same maximizing timestamp;
independent component high-water marks are never summed. Peak sampling must
be calibrated against allocator/storage telemetry before G4. Unused budget
is reported. All of the following count:

```text
search and tuning
router training and inference
teacher/judge/model calls
embedding and retrieval
data generation and curation
training and optimizer state
answer-reachable explicit tokens and implicit KV/recurrent state
index rebuild
raw archive and lineage
verification and rollback reserve
candidate/staging, reader-pinned predecessor, dual-index migration,
build/merge workspace, and temporary replicas
```

Bytes moved, latency, throughput, and energy are outcomes, not matched inputs.

## 9. Reuse and pressure factors

### 9.1 Reuse

\[
R\in\{1,4,16\}
\]

is the controlled mean number of valid future queries per admitted event before
supersession, deletion, or horizon end. \(R=0\) is a separate break-even
diagnostic.

### 9.2 Active capacity

\[
\phi =
\frac{B_{\text{active,cap}}}{B_{\text{ref}}},
\qquad
\phi\in\{1,\tfrac14,\tfrac1{16}\}.
\]

`B_ref` is frozen as the blinded-pilot 95th percentile of raw valid-history
peak bytes over the 128-cycle horizon. Held-out streams reuse the absolute cap;
they do not recalculate it.

Active bytes include every answer-reachable payload, embedding, index,
metadata, adapter delta, and wake-required state. Durable archive is accounted
separately.

## 10. Confirmatory run matrix

### 10.1 Routing and medium

```text
6 deployable methods
× 3 reuse regimes
× 3 active-capacity ratios
× 3 paired stream-seed families
= 162 logical lifetime runs

oracle on the same 27 operating cells
= 27 logical lifetime runs

subtotal = 189
```

### 10.2 Schedule and operator

```text
2 schedules
  {inline-immediate, deferred-at-cycle-boundary}
× 2 operators
  {identity, semantic consolidation}
× 3 reuse regimes
× 3 active-capacity ratios
× 3 paired stream-seed families
= 108 logical lifetime runs
```

Destination is versioned external memory. The identity operator republishes a
byte-equivalent record; semantic consolidation uses one frozen compressor.

### 10.3 Total

```text
189 + 108 = 297 logical confirmatory lifetime cells
```

The user/stream replicate count inside each cell is set by blinded paired power
simulation before G2. A byte-identical physical execution can support more than
one estimand, but logical cells, contrasts, and multiplicity remain fixed.

## 11. Primary endpoint and claim spine

### 11.1 Primary endpoint

For user \(u\), cell \(c\), cycle \(t=1,\ldots,C\), and each of the three
frozen probe blocks \(p\), let \(Q_{uctp}\) contain at least one scheduled
primary query. Each deterministic temporal/deletion-aware query score
\(s(q)\in[0,1]\) is frozen at G2. Define:

\[
U_{uctp}=\frac{1}{|Q_{uctp}|}\sum_{q\in Q_{uctp}}s(q),
\qquad
\operatorname{AUC}(u,c)=\frac{1}{3C}\sum_{p=1}^{3}\sum_{t=1}^{C}U_{uctp}.
\]

Thus cycles and probe blocks have equal weight, the horizon is normalized, and
one absolute utility point means \(0.01\) on this scale. A physical-time
alternative is secondary unless its interval weights are frozen at G2. The
primary endpoint is this lifetime utility AUC under the frozen resource tuple.
The main paired contrast is the deployable hybrid versus the G2-frozen
CALIBRATION-selected benefit or efficiency comparator map. The candidate pool
contains the four tuned single media and the tuned static mixture; all five
held-out comparisons are published, but no TEST-selected envelope supports the
claim.

Comparator maps are selected on CALIBRATION and frozen at G2; TEST never
performs a max/argmax selection. For each public resource cap \(\phi\), pooling
CALIBRATION families and generator-\(R\) strata with frozen equal weights, the
benefit map chooses the highest-AUC baseline among the four single media and
static mixture, with canonical method ID as the only exact tie-break. The
efficiency map first forms the CALIBRATION-competitive set within
\(\delta_{\mathrm{CAL}}=0.01\) of that maximum, then chooses its lowest gross
lifecycle-cost member under the complete frozen price manifest, with canonical
ID as the remaining tie-break. Both maps are invariant across \(R\) and TEST
workload family. If scalar cost is inadmissible, the efficiency branch is
ineligible. TEST reports paired comparisons against all five baselines as
diagnostics, but no TEST envelope or changed map can create support. Reuse
affects the utility queries and is not subtracted again as a monetary “reuse
value.” Lifecycle cost is the gross, non-overlapping native-resource sum under
a source/date/currency/amortization-bound price manifest; if any consumed
resource lacks a defensible price, the 10% scalar efficiency branch is
ineligible and the native resource Pareto comparison remains.

### 11.2 PAPER-C1

**Hypothesis:** oracle heterogeneity value exceeds the frozen
\(\delta_{\mathrm{C1}}=0.01\) margin in at least one of the exactly nine
reuse×pressure cells.

**Decision:** estimate all four unrestricted-minus-restricted component gaps
in every cell under the 104-row scientific subtable and the common
131-row deployability-union critical value. The derived envelope
gap is their minimum. Support requires at least one cell whose four adjusted
95% component LCBs all exceed \(\delta_{\mathrm{C1}}\). If every cell's
derived UCB is below the margin, the material heterogeneity claim is falsified.
Every other pattern is `NARROWED_INSUFFICIENT_PRECISION`; all 36 component and
nine derived estimates/intervals are published. No TEST max/argmax is
bootstrapped.

### 11.3 PAPER-C2

**Hypothesis:** the past-only router adds useful held-out value over the
G2-frozen CALIBRATION benefit/efficiency comparator maps.

Positive branch requires either:

- the simultaneous 95% LCB of `utility gain - 0.02` is above zero; or
- the simultaneous 95% LCBs of `utility gain + 0.01` and
  `lifecycle-cost reduction - 0.10` are both above zero.

Both branches must pass the separate failure-rate noninferiority veto. If
neither holds, result is `FALSIFIED/NARROWED`. All five fixed baseline
comparisons and oracle regret are reported diagnostically; neither a TEST
envelope nor secondary hypervolume can override the frozen decision.

### 11.4 PAPER-C3

**Hypothesis:** reuse and relative memory pressure predict a matched-cap
utility-ranking reversal for external→latent and/or
latent→user-parametric media. This is not an economically optimal destination
or full-lifecycle Pareto-crossover claim. Cost, bytes, latency, deletion, and
governance are reported and may make a cell infeasible, but they do not enter
the six utility decision margins.

Before G2, freeze one low-promotion and one high-promotion corner and
\(\delta_{\mathrm{cross}}=0.01\). The external→latent pair uses four direct
components from the tuned deployable method IDs `raw_external`,
`compressed_external`, and `latent_kv`: either external method must exceed
latent by more than one point at the low corner, while latent must exceed both
by more than one point at the high corner. The latent→user-parametric pair uses
`latent_kv` and deployable `user_lora`, with one one-point oriented margin at
each corner. C1's clairvoyant restricted-oracle rows are forbidden from C3.
All six `effect - delta_cross` margins enter the 104-row scientific subtable
and use the common 131-row deployability-union critical value. Support requires
one pair's full Boolean component rule plus
leave-one-seed-family-out direction stability. The low external-envelope
interval uses componentwise maxima; the high latent-minus-envelope interval
uses componentwise minima; neither resamples a max/argmax. If each pair has a
required material condition ruled out by adjusted UCBs, the reversal is
falsified. Positive but sub-one-point directions, imprecision, or infeasible
corners receive separate narrowed outcomes.

### 11.5 PAPER-C4

The positive branch is `SUPPORTED` only when all three frozen conditions hold:

1. at least one preregistered operator row has a simultaneous 95% LCB for
   `utility gain - 0.02` above zero;
2. for identity, the simultaneous two-sided 95% equal-cell marginal
   deferred-minus-inline interval lies wholly inside \([-1,+1]\) percentage
   point. Deferred execution must also have a simultaneous 95% LCB for
   `lifecycle-cost reduction - 0.10` above zero;
3. the simultaneous two-sided 95% interval for the equal-cell marginal
   schedule×operator utility interaction lies wholly inside \([-1,+1]\)
   percentage point. Both signed equivalence boundaries live in the one
   centered global max-stat family. Every cellwise
   interaction and interval is still reported.

PAPER-C4 is `FALSIFIED_NO_OPERATOR_EFFECT` only when every eligible operator
main/conditional simple-effect simultaneous UCB is below 0.02. Otherwise,
failure of any support component is `NARROWED`:
sub-two-point gains are below materiality, failure of condition 2 is
consolidation without demonstrated deferred-cost benefit, and failure of
condition 3 is schedule-contingent or interaction-inconclusive consolidation.
An interval overlapping both the null and an equivalence margin is insufficient
precision, not equivalence. A significant interaction is reported and cannot
be relabeled as support.

## 12. Retention and plasticity

Every cycle measures two distinct outcomes:

```text
retention:
  does the incumbent still answer protected old queries?

acquisition:
  how much does a fixed new-task data/step budget improve held-out new-task
  utility relative to an architecture-matched fresh control?
```

The acquisition cell is invalid when the fresh-control denominator is below a
frozen learnability threshold.

### 12.1 H-STC-007 extension

A preregistered high-interference parametric stream compares:

```text
no recycle
restricted low-utility recycle + canonical replay
random unrestricted recycle + identical replay
fresh-base diagnostic
```

Retention and acquisition are measured at every cycle. The ordering claim is
falsified if no frozen high-interference regime crosses the acquisition
boundary before aggregate retention. The intervention is separately falsified
if restricted recycle fails to recover acquisition or violates protected-slice
noninferiority.

Secondary diagnostics:

```text
dormant-unit fraction
activation and gradient stable rank
weight magnitude
free/recyclable parameter fraction
optimizer-state age
```

They never substitute for behavioral acquisition.

## 13. Scaling track

At most two methods are used for dense scaling. One axis varies at a time with
at least eight fit points and two held-out scales.

Candidate axes:

```text
cycle/event count
reuse
active/durable bytes
live tokens
latent slots
adapter/expert capacity
adapter rank, module count, router error, and resident catalog fraction
sleep FLOPs
arrival/backlog load
consolidation-clock count and local-maintenance radius
volatility/delete pressure
users/concurrency
estimated state-support covering dimension
```

The external-capacity characterization track freezes a representation/metric
and separates two experiments:

1. **cardinality:** freeze codec, precision, and bytes per prototype, allow
   total bytes to grow with \(M\), and test the \(M^{-1/d}\) term;
2. **fixed-active-representation budget:** freeze
   \(B_{\rm active\,repr,total}\), including headers, keys/index,
   provenance/policy metadata, decoder, and payload; vary \(M\); measure
   \(b_{\rm payload}(M)=
   [B_{\rm active\,repr,total}-B_{\rm fixed}-B_{\rm index}(M)
   -B_{\rm metadata}(M)]/M\); reject infeasible negative allowances; and fit
   \(cM^{-1/d}+q(b_{\rm payload}(M))\) using an independently calibrated
   distortion function.

Both compare uniform prototypes, value-aware coresets, learned quantization,
and eviction, and measure held-out coverage radius, codec distortion,
retrieval error \(\delta_M\), and downstream utility across at least eight
sizes. A cardinality exponent is accepted only in arm 1, inside a workload
where compact-support/covering-number diagnostics pass. Validation is one
frozen equal-weight finite target: two held-out sizes×two named held-out
families. Within each family the same physical user/seed is paired across the
two sizes; family estimates then receive exact one-half weights. Accuracy is
nonnegative log-radius
\(\sqrt{\sum_c w_c\{\ln(\widehat R_c/R_c)\}^2}\), so over- and
underprediction cannot cancel, and prediction-interval coverage uses the
hierarchy-aware boundary-score contract rather than a zero-variance Wald
interval. Memento 2's conditional theorem is not treated as empirical
validation of either fit.

The deployable value-aware policy uses only a frozen past-observable
TRAIN/CALIBRATION estimator; future reuse/query/utility and holdout labels make
it an oracle upper bound excluded from deployable claims. Raw evidence,
lineage, rollback, and recovery state are reported in full retained/peak-state
ledgers outside \(B_{\rm active\,repr,total}\), so arm 2 never supports a
fixed-total-memory claim.

The first-stage capacity knee is threshold-anchored logistic. The seven
second-stage resource-surface competitors are:

```text
constant/null
normalized Cobb-Douglas
min-bottleneck
generalized mean
interaction
segmented load-at-one
shape-constrained monotone I-spline/tensor basis
```

A ten-column no-intercept D-optimal design is byte-identical to the
\(g(r_0)=0\) estimator, scores unique physical configurations rather than
logical aliases, and gives an already observed tuple zero incremental
information. Exact formulas, parameter domains, GLS loss/penalties,
hyperparameter grids, one-SE/complexity tie-break selector, finite-lattice
doubling-edge/interaction-face contrasts, and
`future_estimated_knee_fixed_panel_v1` predictive target are bound by the
canonical analysis-plan digest. Family-only, resource-only, and double-holdout
RMSE/coverage are three separate conjunctions with equal named-family then
equal unique-configuration weights. A fitted crossing contour is exploratory,
not a powered phase-boundary law.

A relation is an `A100-regime scaling law` only if it passes frozen prediction
tolerances on held-out scales/resource configurations and workload families on
that frozen A100 stack. An unqualified or cross-hardware scaling law additionally
requires a prospectively mandatory, powered second-hardware holdout. Otherwise
it is a regime-specific relation, empirical curve, or null.

## 14. Exploratory budget

The dense resource surface, coverage/codec study, and parametric-information
study share a sponsor-frozen pre-data ceiling \(Q_{\rm capacity,max}\).
`program-budget-envelope.yaml` also freezes non-borrowable
`Q_core,primary`/`Q_core,clean` quotas and absolute native-resource and
carbon/energy caps. Before power-CAL, an optimistic lower-cost
coarse-feasibility calculation must cover every mandatory physical
cohort/panel or return `FAIL_DESIGN_INFEASIBLE`. After power-CAL, each track's
program-wide simultaneous upper-99 native-resource reservation envelope must
fit its quota and the three quotas/projections must fit
\(Q_{\rm capacity,max}\); rows, seeds, panels, and lags cannot be silently
reduced.

RRE is the maximum component ratio against the sampling-free engineering
profile `RRE-NATIVE-A100-V1`, not a noisy observed child. Every reference
component is positive and every native resource also has an absolute sponsor
cap; missing counters are never zero. Production checks one canonical
append-only actual ledger before every launch and merge, with committed work
equal to measured actual plus unreconciled reservations; track views are
derived from it. Core G2 and G2-CAP remain independent, and unused core,
clean-reproduction, or capacity quota never transfers.

Candidate factors:

- 512/2K/8K live tokens;
- sleep-to-wake FLOP ratio;
- volatility and deletion;
- trigger family;
- exact replay, distillation, and dream augmentation;
- asynchronous schedule;
- per-user versus batched LoRA;
- graph or shared-weight destination;
- physical cluster placement;
- provenance/verification/rollback enforcement.
- fixed universal harness versus universal kernel plus task-specific searched
  write/read module versus RHO-style full-harness rewrite, with all
  MemEvolve/ALMA/M★/RHO-style re-ingestion, code generation, repeated
  re-solves, diagnosis, ranking, debugging, and validation charged;
- fixed diagnostic probes versus generated single-session, cross-session,
  temporal, contradiction, delete, and rare-event probes, with generator
  provenance and leakage tests;
- direct retrieved-context injection versus a MemCompiler-style learned
  text/latent delivery module, reported as wake cost rather than sleep cost.
- MEMENTO-style stateful compaction versus text-identical restart/re-prefill,
  with peak KV, KV memory-time AUC, latency, passcode leakage, and downstream
  utility measured jointly.
- calibrated-surprise admission crossed with full-history, recent-window,
  fixed class-balanced coreset, reservoir, and no replay; report admitted
  support, replay work, old retention, and fresh acquisition separately.
- wake-time case/skill rewrite versus delayed topic/section compilation, with
  exact-repeat, held-out paraphrase, composition, correction, and unrelated
  query families reported separately.
- flat all-skill-description prompting versus hierarchical bounded retrieval
  across catalog size, charging router construction, prompt tokens, misses,
  retry learning, and governance work.

Successive halving or a frozen fractional factorial allocates this budget.
Exploratory findings cannot rewrite the confirmatory claim spine.

## 15. Public external-validity tracks

### 15.1 MemoryAgentBench

Use for retrieval, test-time learning, long-range understanding, and selective
forgetting characterization.

### 15.2 LongMemEval

Use for delayed recall, temporal update, preference, and abstention.

### 15.3 HaluMem

Use for writer→retriever→reader failure decomposition.

### 15.4 Procedural transfer

ScienceWorld training with held-out ALFWorld or WebArena is a stretch external
track. H-EPM is the successful-trajectory, state-annotated tool-graph
comparator; PlugMem is the task-boundary proposition/prescription-graph
comparator; MemMA is the session-boundary probe-and-repair comparator;
MemCompiler is a wake-path state-conditioned delivery comparator. Their public
code paths should be replayed as characterization tracks, not inserted post
hoc into the primary dominance family.

Each adapter:

- preserves original session order;
- declares the sleep boundary and withheld queries;
- has a version and transformation digest;
- separates train, validation, and evaluation users;
- does not fit the controlled scaling laws.

The procedural track must include success, partial-success, and failed
trajectories. An H-EPM-style success-only admission arm is matched against
outcome-calibrated and stratified failure-retaining arms so that “cleaner
memory” cannot hide loss of negative procedural knowledge. A PlugMem-style
merge arm must report physical bytes and inactive originals separately; soft
deactivation is not reclamation.

The MemMA adapter must regenerate probes from a published pipeline rather than
reuse only the committed parquet, then compare fixed hand-authored probes,
real-time generated probes, committed probes, and no repair. Final LoCoMo QA
hashes remain inaccessible to generation. The MemCompiler adapter must charge
compiler latency, tokens, parameters, and trajectory-bank bytes separately from
executor latency and live Brief State; executor-only speedup is not total
system speedup.

### 15.5 Learned within-query compaction

Microsoft MEMENTO is a characterization track, not a sleep arm. Reproduce its
block-mask runtime with four frozen conditions:

```text
uncompressed reasoning
stateful memento text + original block-conditioned KV
identical memento text + restart/re-prefill KV
text-only external summary under the same visible-token budget
```

Cross the conditions with compression pressure, reasoning length, model
family, and hop depth. Report pass@1, pass@k, peak KV, KV AUC, throughput,
batch completion, generated tokens, and restart/migration cost. A lower peak
does not count as a system win if longer trajectories raise memory-time, as in
MEMENTO's OLMo-3 AIME'26 example.

For state completeness and privacy, inject a uniformly sampled opaque
five-digit or longer bit payload into a source block that is subsequently
evicted. Keep evaluation payloads label-unique and inaccessible to training.
Probe text, downstream KV by layer and hop, serialized artifacts, and the
post-delete serving process. Include a causal pre-source control and a
text-identical restart control. Physical source-block removal passes deletion
only if explicit retrieval, latent probes, and behavioral influence all meet
their frozen chance/equivalence bounds.

### 15.6 Retrospective full-harness learning

RHO is a control-plane characterization track rather than a primary memory
destination. Freeze a past trajectory/training pool and an inaccessible
held-out pool, then compare:

```text
frozen universal harness
frozen governance kernel + generated task module
RHO-style generated complete harness
raw-trajectory proposal ablation
no-consistency and no-self-validation ablations
```

Match the optimizer model, task solver, source trajectories, coreset size,
rollout group, proposal count, and held-out budget. Preserve the paper's
difficulty/DPP selection and positive-score acceptance as one arm, but add an
independent promotion judge and a frozen safety/capability canary as separate
gates. Re-run enough independent seeds and optimization rounds to estimate
candidate-selection regret and growth rather than treating the paper's single
round as a lifetime result.

Charge every auxiliary call. The paper-aligned one-round accounting is 30
before-rollouts, 10 diagnoses, three proposals, 30 candidate re-solves, and 30
pairwise ranks—103 agent invocations—plus 100 difficulty-judge calls and one
batched embedding call for the SWE-Bench Pro selection pass. Report summed
agent time and elapsed time separately, because ten-way concurrency changes
latency but not total work.

Inject benign distractors, false successful trajectories, prompt injection,
secret-bearing traces, and capability-escalation requests. Measure held-out
utility, independent-gate disagreement, poison persistence, authorization
violations, exact rollback, source deletion fan-out, active and archived
harness bytes, executable dependency count, version/log growth, and
incumbent-to-candidate migration cost. A 32 KB instruction-file limit is not a
total artifact cap.

### 15.7 Surprise admission and replay-support geometry

Reproduce Surprise-Gated Memory as a low-evidence characterization track with
frozen features and an independently specified implementation. Cross
admission policy and replay support rather than treating them as one method:

```text
admission: all / calibrated surprise / random matched-count / oracle novelty
replay: all history / recent window / reservoir / class-balanced coreset / none
destination: slow linear readout / matched-capacity MLP
```

Use open-class streams longer than 50 tasks and at least two frozen
representations. Freeze threshold calibration before the stream, then add
distribution-shift and rare-known slices. Report old-task retention,
fresh-normalized acquisition, gate AUROC/F1, false admissions/rejections,
support bytes, class coverage, per-cycle and cumulative replay work, and
capacity knees. Multiple seeds are mandatory because the paper's central
ImageNet comparisons are single runs.

The preregistered negative control is recent-window replay: the source reports
`41.2` versus `67.0` no replay on DINOv2 and `0.0` versus `25.9` on I-JEPA.
A bounded method does not pass merely by using constant bytes; it must avoid
making old retention worse than equal-compute no replay. For the VLM path,
teach opaque and contradictory facts, sleep, clear fast memory and dialogue,
then test prototype collision, retrieval, source deletion, and slow-store
growth. Do not call frozen-VLM prototype migration base-model learning.

### 15.8 Staged external knowledge and skill compilation

Use [Evolve](https://arxiv.org/abs/2604.23424) and the post-paper
[Memento-Skills DreamDaemon](https://github.com/Memento-Teams/Memento-Skills/tree/71ac933ea1381d53389a2426f59634e0182071b8)
as characterization tracks, with Memento 2 and the Memento-Skills paper loop
as matched wake-learning comparators. Reproduce:

```text
no persistent memory
foreground acquire/update only
foreground skill rewrite + same-task retry
deferred section/topic compilation
deferred compilation with versioned transactional publication
oracle bounded external memory
```

For Evolve, preserve its 42-category classifier, `.85` overlap threshold,
teacher, local model, and explicit sleep cycle as a paper-aligned arm. Then
replace its identical cold/warm/post question sequence with disjoint
acquisition, exact-repeat, paraphrase, composition, correction, and unrelated
sets. Report answer quality, teacher calls, latency, category error, false
merge/split/delete, and store bytes by relationship label. The paper-aligned
repetition remains a diagnostic cache baseline and cannot support a
future-query-generalization claim.

For relationship class \(r\), freeze the quality-floor-conditional
amortization estimand

\[
h_r
=
\Pr(\text{no teacher call and correct}\mid r).
\]

The semantic-transfer gap is

\[
G_{\text{sem}}
=
h_{\text{exact-repeat}}
-
\sum_{r\in\mathcal R_{\text{unseen}}}\rho_r h_r,
\]

where deployment weights \(\rho_r\) are frozen before outcomes. Estimate both
terms from paired underlying memories with different surface queries; never
relabel repeated questions as independent semantic samples.

The sleep-specific causal characterization adds a parallel no-sleep lifecycle
with the same elapsed time, query families, foreground writes, and teacher
budget:

\[
\Delta_{\text{sleep}}
=
(U_{\text{post}}-U_{\text{pre}})_{\text{sleep}}
-
(U_{\text{post}}-U_{\text{pre}})_{\text{no-sleep}}.
\]

Store compaction, teacher-call avoidance, and utility each receive a separate
\(\Delta_{\text{sleep}}\); one cannot proxy for another. This corrects the
source study's lack of a no-sleep counterfactual.

For skill memory, sweep catalog size while separating the at-most-three topic
documents injected after Dream from all skill descriptions consulted during
selection. Report catalog prompt tokens, router calls, retrieved artifact
count, execution retries, skill variants, deduplication, and active/archived
bytes. A compact execution prompt does not pass the capacity endpoint if flat
catalog selection or version governance grows without bound.

Both adapters release immutable source events, source spans, candidate
sections/topics/skills, parent and replacement edges, teacher/model/prompt
digests, rejected candidates, and final sub-store roots. Fault-inject after
each vector, metadata, topic, index, and staging operation. Compare the source
implementation with a manifest-last protocol:

```text
write immutable candidate sub-roots
→ validate cross-store references and canaries
→ CAS one serving-manifest pointer
→ emit publication + staging-range acknowledgement
→ retain predecessor/source until rollback expiry
```

Required recovery outcomes are all-or-old visibility, idempotent replay, no
lost staging event, exact rollback, and bounded stale reads. Per-store
transactions without a common version root fail this endpoint.

### 15.9 LightMem sleep-causality and index-coherence track

The two direct papers called LightMem are separate characterization arms.

The ICLR 2026
[LightMem](https://arxiv.org/abs/2510.18866) arm preserves the paper's
compression, topic segmentation, soft append, timestamp-constrained update
queues, and manual offline parallel update. Freeze both arXiv v1 and the
camera-ready v4 result tables, and pin the last repository commit before the
v1 public timestamp. Its LongMemEval table already supplies an important
descriptive negative result: across the six reported GPT/Qwen configurations,
offline-update minus soft-update accuracy is
`+0.40, -2.39, -1.57, +0.39, -5.06, -1.35` percentage points. These values do
not estimate a causal sleep effect because there is no parallel no-sleep
lifetime and no uncertainty for the within-LightMem contrast.

Run the following matched arms with recorded model decisions so that a second
live LLM sample cannot masquerade as an indexing effect:

```text
soft append, no sleep
paper-aligned sleep rewrite with incumbent vector
same rewrite decisions with changed payloads re-embedded
same rewrite decisions with a manifest-last payload+key publish
oracle update decisions with manifest-last publication
```

For every rewritten item, recompute the backend-specific key from the published
payload and record key--payload discrepancy, encoder digest, top-\(k\) ranking
overlap, relevant-item recall, and stale-key hit rate. Include exact-repeat,
paraphrase, temporal correction, conflict, and unrelated queries. A payload
rewrite passes semantic publication only if payload, key, encoder,
provenance, and authorization watermark belong to the same serving
generation.

Sweep \(M\), the number of LTM entries, and separately count vector searches,
coordinator comparisons, queue elements inspected, LLM calls, embeddings,
bytes read/written, and wall time. The paper's call-count model must not be
reported as whole-system asymptotic complexity: its released update loop scans
the complete entry collection and every other entry's bounded queue for each
target. Fit linear, \(M\log M\), and quadratic candidates on held-out store
sizes.

The ACL 2026
[Lightweight LLM Agent Memory with Small Language
Models](https://aclanthology.org/2026.acl-long.588/) arm is initially
`CHARACTERIZATION`, not `REPRODUCTION`: the paper specifies a 10,000-item MTM
cap, a 10--15-turn offline cadence, approximately one new LTM node per four
turns, and a fixed retrieval budget, but provides no identified offline model,
released SLM-2 training set/checkpoint, or implementation. Recreate only a
declared compatible implementation unless those artifacts appear. Compare its
reported full-versus-no-offline DialSim contrast (`4.12` versus `3.96` F1)
with a paired no-sleep control and an LTM-capacity sweep. Charge the unbounded
shared LTM, de-identification audit, source lineage, cross-user leakage tests,
and delete fan-out; a bounded MTM or Top-\(K\) read does not establish bounded
lifetime state.

### 15.10 Functional, reconstructive, and reflective mechanism track

This track prevents three recent “consolidation” labels from being treated as
one intervention.

**Mela characterization.** Run two independent calls with the same first
sequence and a second-call probe. Compare the released wrapper against:

```text
fresh functional memory every call
explicitly serialized/reloaded HMM state extension
same extension with state reset before probe
KV-only continuation
```

The released path is a characterization control; persistence added by the
study is labeled an extension, not paper reproduction. Record every
answer-reachable tensor, initialization source, serialization bytes,
per-token update FLOPs, restart equivalence, and cross-session retention.

**MIRROR concurrency and lifetime work.** Replay identical conversations under
human-paced, bursty, and adversarial inter-arrival traces. Compare:

```text
full bounded reconstruction every turn
event-triggered reconstruction
append-only bounded raw/history baseline
generation-tagged CAS reconstruction
source release without generation ordering
```

Log scheduled turn, snapshot parent, worker start/finish/cancel, attempted and
published generations, next-turn wait, active thread count, token/FLOP cost,
state quality, and stale-overwrite incidence. Sweep horizon and pause
distribution. Constant narrative length passes a state bound only; cumulative
rewrite cost is reported separately.

**HeLa-Mem causal mechanism.** Build one memory graph per dialogue and issue
multiple later queries after each update. Factor:

```text
embedding + fixed temporal edges
periodic LLM fact extraction
Hebbian reinforcement
hub-triggered reflective distillation
decay + adaptive forgetting
```

Instrument the call graph to prove each nominal component executed. Reproduce
both the printed \((1-\lambda)w+\eta I\) update and the released
`.995 × w` retention update as separate arms; do not silently choose one.
Report graph/fact growth, retrieval context, per-query causal availability,
seeds/intervals, and a fixed-total-byte sweep. If the principal LoCoMo assets
remain unavailable, label the exercise compatible reproduction rather than
paper reproduction.

### 15.11 Conditional parametric-capacity and allocation track

Use the opaque-payload family with known conditional entropy to compare:

```text
shared-weight full fine-tuning
fixed-rank user adapter
preallocated expert/slot
external exact lookup
external lookup with a learned <CALL> router
past-only dynamic destination router
```

Sweep added parameter bits, precision, payload entropy, association count,
training tokens, update order, reuse, volatility, and delete pressure. Report
mutual-information lower bounds, exact and semantic addressability,
interference, plasticity, call latency/failure, total state, and delete
residuals. Fit the controlled Allen-Zhu/Li and Morris-style protocols
separately; pooling their bits/parameter estimates is forbidden.

For each independent opaque-payload seed, hold the public base
\(\theta_0\) fixed and retrain the candidate state from scratch. A decoder/
readout and every estimator choice are frozen on CALIBRATION before TEST.
Each of \(N\) payloads comprises \(L\) independent uniform Bernoulli bits. If
\(p_{a\ell}\) is the frozen decoder's association×bit error probability, the
required primary is
\(I_{\rm LB,total}=\sum_a\sum_\ell[1-h_2(p_{a\ell})]\) and
\(I_{\rm LB/bit}=I_{\rm LB,total}/(NL)\), with \(p_{a\ell}=0.5\) as the
no-signal null and simultaneous master-seed-family cluster intervals. Error
probabilities may not be pooled before applying \(h_2\). Block error may not replace per-bit
error. A separately preregistered variational lower bound is optional secondary
evidence only and cannot rescue a failed primary after outcomes. Conditioning
is restricted to frozen public, payload-seed-
independent side information in \(\Theta_0\). Every answer-bearing or
payload/TRAIN/CALIBRATION/TEST-dependent router, optimizer, codebook, auxiliary
decoder, and checkpoint is charged to complete mutable state and cannot be
conditioned away. A theta-only estimand is admissible only when the frozen
readout accesses \(\theta\) and public \(\Theta_0\) alone; one checkpoint
cannot identify \(I(Y;\theta\mid Q,\theta_0)\).

A LaCy/LMLM-inspired subexperiment holds gradient-bearing ground-truth tokens
constant while replacing factual targets with external calls. It measures
fresh non-factual acquisition and representation/plasticity probes, not just
FactScore or factual leakage. The simple freed-capacity hypothesis passes only
if offloading improves those endpoints at matched parameters, gradient tokens,
and external-call budget. A null result is publishable and narrows the claim
that externalization creates usable reasoning capacity.

### 15.12 Capability-complete continual internalization track

ImprintBench is the primary destination-comparison adapter once its exact
venue-specific PDF and dataset snapshot are pinned. Until then, preserve only
the record-level news, API-changelog, and personalization domain taxonomy; do
not freeze or cite the item count from unpinned bytes. After pinning, record
the exact snapshot count and report the six capabilities independently:

```text
direct acquisition
temporal update
reference resolution
composition
implicit relevance
boundary awareness
```

Compare native long context, matched context management, auxiliary parameters,
one user adapter, and external memory. Add a no-update baseline and an oracle
evidence-delivery diagnostic. Every method receives the same source events,
information cutoff, future questions, and total lifecycle envelope; prompt
tokens, retriever/index state, optimizer state, and update FLOPs are charged.
The paper's reported compression and SFT/distillation patterns are descriptive
targets, not acceptance thresholds.

MemoryBench supplies the paired wake-learning control. Reuse its declarative
and procedural workloads plus explicit action/verbal and implicit feedback,
then compare:

```text
immediate feedback-driven update
staged feedback, same operator at cycle boundary
staged learned consolidation
no write
```

The off-policy and on-policy tracks remain separate, as do dataset-native
metrics. A system omitted because of unreasonable runtime is recorded as a
resource failure, not missing-at-random accuracy. This track cannot support a
sleep claim unless information, operator, and compute are held constant while
only execution timing changes.

### 15.13 Modular LoRA knowledge-memory track

Reproduce the controlled LoRA-as-memory results with a pinned model and
document corpus, then extend them into a lifetime stream. Cross:

```text
supervision ∈ {raw, QA, summary, rewrite, mixed}
rank ∈ {2, 4, 8, 16, 32, 64}
topology ∈ {one adapter, isolated adapters, routed top-1, routed top-k}
composition ∈ {none, linear, scaled concatenation, TIES}
```

The matched-budget anchor is one rank-32 adapter versus eight rank-4 modules
over the same 64K-token PhoneBook corpus, with total trainable adapter
parameters matched. Run an oracle router first, then frozen lexical, embedding,
and learned routers. Report route top-\(k\), out-of-catalog rejection,
composition coverage, merge conflict, setup/load/merge time, resident and total
adapter bytes, and end-task utility. Oracle gains are upper bounds and cannot
be pooled with deployable routing.

The confirmatory deployable endpoint is the aggregate oracle-minus-deployment
gap. Routing, merge, and residency components belong to extension `X-STC-Q`
and remain exploratory unless a pre-TEST amendment materializes, powers, and
multiplicity-binds this complete \(2\times2\times2\)
routing×merge×residency intervention at G2:

```text
routing   ∈ {oracle module set, frozen qualified deployed router}
merge     ∈ {oracle-compatible composition, frozen deployed merge policy}
residency ∈ {selected modules pre-resident, frozen deployed cache/load policy}
```

If promoted, the factorial preserves the same queries, module pool, and
parameter budget and charges each intervention's setup and serving work.
CALIBRATION freezes numerical thresholds for route-recall LCB,
out-of-catalog-FPR UCB, and routing/load cost; at least one router must qualify
before TEST. No qualified router yields `INCONCLUSIVE`, not a post-hoc
competence exception. Among all qualified routers, simultaneous
modular-minus-single UCBs at or below the frozen materiality margin in every
cell falsify the extension claim. Without the amendment, component results are
diagnostic only and cannot qualify or rescue the PAPER-C aggregate claim.

Extend the source stream with corrections, superseded versions, tenant
isolation, cross-document composition, one-off facts, and deletion. Compare:

```text
new adapter version + old version retained
in-place adapter update
merge into a shared adapter
external canonical memory + compiled adapter cache
```

Track rank knees, adapter-count/catalog growth, GPU cache hit/swap, interference
within and across modules, source-to-weight lineage, rollback, and post-delete
behavioral residuals. A fixed rank bounds one module, not the module pool or
its recovery state.

### 15.14 Multi-clock, recurrence, and local-maintenance track

Use one frozen conversational stream and future-query set to compare four
external transformation policies:

```text
TiMem-style session/day/week/month temporal closure
RecMem-style similarity + recurrence admission
GAM-style pause/end/overflow buffer flush
global periodic rebuild versus region-local maintenance
```

Required ablations are one clock versus the full temporal hierarchy,
recurrence-only versus surprise/utility/volatility admission, 2,048-token live
buffer versus a fixed **total** state cap, and raw archive retained versus
tiered versus removed. Local and global jobs receive the same model, evidence,
and eventual query set.

Measure LoCoMo/LongMemEval-style task utility, rare one-off survival,
correction, exact-evidence loss, construction calls by clock, p50/p95 job and
query latency, active/graph/archive/index bytes, graph nodes/edges, queue age,
bytes touched, and cumulative rewrite work. Reproduce source-reported call or
growth figures only as characterization targets. A smaller recalled context or
bounded progression buffer does not pass the total-capacity endpoint.

The agent-native memory study provides an external systems envelope. Select
representatives from low-latency/lightweight, hierarchical, graph, and
high-utility regions rather than declaring one universal baseline. Plot
utility, exactness, construction/query latency, cost, and maintenance locality
jointly; a dominated system can still be useful as a mechanism control.

### 15.15 Embodied sleep and multimodal governance track

MEMORA supplies a strict experimental extension from text streams to
egocentric action evidence. Freeze participant/video order and compare:

```text
raw time-restricted evidence
online entity/action edits only
typed episodic stores without offline consolidation
full offline habit/workflow/preference consolidation
full system with post-anchor evidence deliberately injected
```

The leakage control must rebuild or roll back the memory snapshot at each query
anchor; retrieval and consolidation products after the anchor are forbidden in
all valid arms. Preserve participant isolation and report memory-assessment,
in-distribution replay, and out-of-distribution planning separately across
answer backbones.

Add corrections to object state, rare safety-relevant actions, conflicting
videos, consent withdrawal, and participant deletion. Every entity, inferred
fact, habit, workflow, preference, adapter, and benchmark item must trace to
time spans and participant authorization. Measure utility, false
generalization, temporal leakage, active/archive video-derived bytes,
construction calls, deletion fan-out, rebuild latency, and residual influence.
The paper's 45-hour/eighteen-participant horizon is external validity, not a
lifetime-capacity demonstration.

## 16. Learning and memory metrics

```text
lifetime utility AUC
current utility
retention and forgetting
fixed-budget normalized acquisition
forward transfer
memory half-life and lag survival
exact/semantic/temporal/relational/procedural recall
direct acquisition / update / reference / composition / relevance / boundary profile
calibration and false-memory rate
writer/retriever/reader errors
destination route / adapter route / merge compatibility errors
coverage/preservation/faithfulness/detail loss
stale/conflicting hits
key--payload discrepancy and ranking overlap after rewrite
poison amplification
cross-user/task interference
exact-repeat versus held-out-semantic transfer gap
rare one-off and recovery-substrate ablation gap
false merge/split/delete and orphaned canonical state
```

## 17. Governance metrics

Deletion is not one binary score:

```text
access denial
online physical artifact removal
derived-lineage invalidation
backup/key expiry
behavioral residual influence
completion latency for each stage
undeclared replica discovery
opaque-source residual in surviving KV/recurrent state
rollback after correction or poison
participant/video consent propagation to entity/habit/workflow descendants
multimodal time-anchor leakage and derived-view rebuild completeness
```

Every derived artifact exposes source dependencies or explicitly fails the
governance condition. Shared-weight unlearning is not assumed.

## 18. Systems metrics

```text
p50/p95/p99 wake latency and TTFT
accepted requests/s and tokens/s
retrieval/queue/prefill/decode/writeback decomposition
effective batch and MFU
achieved/roofline memory bandwidth
peak KV and KV occupancy-time AUC
cache/adapter hit, miss, load, and swap
adapter catalog route, merge, and compatibility-check latency
flat-catalog selection tokens/latency and injected-artifact tokens/latency
sleep throughput, per-clock calls, queue age, and deadline misses
GPU/CPU/HBM/DRAM/storage/network resource use
bytes moved, maintenance-scope bytes, and memory amplification
active/archive/index/graph/catalog state and cumulative rewrite work
freshness/staleness
version publish and rollback cost
lifetime cost and energy
```

Energy boundaries distinguish GPU-board measurement, host measurement, and
modeled storage/network. Energy-per-correct is reported only above a frozen
accuracy floor.

## 19. Statistical design

### 19.1 Independent unit

The crossed 16 workload families×3 paired-seed families are 48 fixed,
equal-weight finite-target blocks. A physical user/stream inside one such block
is the top-level resampling unit; the three seed families are not random PSUs
for a seed-generator superpopulation. Cycles and probes are repeated
measurements, not independent samples. Every powered logical row has at least
four genuinely independent users in each applicable workload×seed block.

### 19.2 Pairing

All methods receive identical user streams, event order, query order, and
resource tuple. Method order and hardware placement are randomized or blocked.

### 19.3 Power

Before G2, blinded pilot covariance feeds a paired hierarchical
power/sensitivity simulation but never changes scientific margins. A
pre-pilot, hierarchy-aware full-covariance builder retains all 48 fixed blocks,
resamples users only within block, freezes bounded supports, shrinkage,
PSD-plus-diagonal-inflation rules, and passes full-object 99% coverage on the
prospectively frozen pilot-size/stress grid. The final user count is the
maximum required across all nine possible C1 powered-cell locations,
PAPER-C2's frozen-map router contrast, PAPER-C3's direct-component crossover
rule, and 27 feasible C4 locations. C4 comprises 18 schedule-specific
simple-effect scenarios and nine schedule-marginal main-effect scenarios,
always generated from underlying identity/consolidating×inline/deferred 2×2
cell means and then algebraically transformed; inconsistent derived-effect
injection is forbidden. Opaque variance strata preserve rare high-variance
cells. The blueprint, design spec, and executable
`analysis-plan-skeleton.json` bind the same 48-block/27-scenario/covariance-
builder digest; mismatch blocks G2.

### 19.4 Inference

The 48 workload-family×paired-seed-family TEST blocks are fixed equal-weight
finite-target strata, not IID clusters. Primary uncertainty uses exactly
20,000 deterministic blocked paired bootstrap replicates. Each replicate keeps
all 48 blocks once and resamples top-level paired user/stream IDs within each,
retaining all methods,
reuse/pressure cells, schedules, operators, cycles, and probes belonging to
each selected stream.

Each of the 104 continuous scientific rows uses a 48-block-stratified
paired-stream-cluster sandwich standard error. Each of the 27 rare
paired-Bernoulli RF rows instead uses a constrained paired-discordance score
with positive least-favorable boundary variance; all-zero/all-one samples have
finite bounds and never auto-pass through `SE=0`. Multiplicity
uses the replicate-wise maximum absolute standardized row statistic over the one frozen
global contrast family `PAPER-C-DEPLOYABILITY-GLOBAL-131`; its empirical
quantile provides the adjusted 95% critical value for ordinary and both signed
equivalence boundaries. The ordered union contains the 104 scientific
utility/cost/equivalence margins and 27 same-weight resource-failure companion
margins. Operational rows can only veto, but share the critical because they
are part of a deployability claim. Shared invalid/admin censoring is a
pair-identical, alpha-free integrity/MNAR veto and never a zero-SE studentized
row. A mixed-null simulation must control any false supported claim at 0.05.
For a continuous transformed margin
\(m_j=s_j\hat\theta_j-b_j\), replicate \(b\) uses the
centered statistic
\(((s_j\hat\theta_{b,j}-b_j)-m_j)/SE_{b,j}\); ordinary,
and both TOST boundary rows use this rule. RF rows use their
constrained boundary score and join the same max-stat vector.
Uncentered \((s_j\hat\theta_{b,j}-b_j)/SE_{b,j}\) statistics and
claim-specific critical values are forbidden. Report adjusted and unadjusted
intervals plus leave-one-paired-seed-family-out sensitivity. The Webb
wild-cluster sensitivity uses the same centered transformed rows.

Before G2, power simulation must prevalidate sandwich coverage. A 99,999-draw
paired-stream wild-cluster bootstrap stratified within workload×paired-seed
block is a required
sensitivity. A claim cannot be supported when the primary decision, the
wild-cluster decision, or seed-family sensitivity disagree. Mixed-effects
models, if reported, are secondary sensitivity analyses and cannot replace or
override this primary procedure.
Claim scope is the crossed equal-weight finite mixture of 16 named families and
three fixed seed families. A common seed-family shift changes the conditional
finite-target point estimate and leave-one-seed sensitivity but does not create
a seed-superpopulation variance term. Family-label duplication/splitting that
preserves that mixture must not change uncertainty or decisions.

The frozen analysis manifest also fixes estimand/aggregation, the global
multiplicity family, equivalence/materiality margins, missing and failed-run
handling, and outlier/invalid-cell rules.
The blueprint, research-design spec, and executable theory plan all bind the
same canonical predicate/function digest from
`analysis-plan-skeleton.json`. A cross-document contract test extracts C2/C4
margin IDs, boundaries, interval level, and falsification rule from each source
and rejects any mismatch; prose cannot silently define a weaker test.

No outcome-driven optional stopping.

### 19.5 Missingness

Only shared protocol corruption that prevents every method in a paired block
from being scored invalidates that block. A zero denominator caused by the
generator, shared checksum corruption, or a missing common oracle artifact is
such infrastructure missingness and receives worst-case MNAR bounds.

A protocol-valid method timeout, OOM, compiler exception, deadline miss, cap
violation, invalid method artifact, or method-produced absent probe is instead
an observed intent-to-treat `RESOURCE_FAILURE`. Keep the randomized stream,
charge all consumed work, and score every affected probe zero even when a
deliberate abstention would have matched the target; unaffected pre-failure
probes retain their frozen scores. Only an intentional protocol-valid
`ABSTAIN`/`FORGET` under `COMPLETE` may receive abstention credit. Apply any
preregistered partial-identification bound only to its declared unresolved
cost/missingness component,
and report feasibility/failure rate by method and cell. Dominance requires
utility/cost criteria and a same-weight failure-rate-noninferiority companion
for every required C2, C3, and C4 component. The boundary
\(\Delta F=0.01\) calibrates type-I error; the interior \(\Delta F=0\)
alternative determines power. A failed companion yields
`NARROWED_RESOURCE_FAILURE_NONINFERIORITY`. Complete-case estimates are
sensitivity-only. Administrative late-horizon censoring is named and excluded
symmetrically from the primary denominator.

## 20. Reproducibility and artifact graph

Every run binds:

```text
code and repository SHA
container/environment
base checkpoint and revision
tokenizer
dataset/generator version
observable/oracle stream digests
diagnostic-probe generator/prompt/input/output/filter digests
method config and search budget
random-number hierarchy
resource/hardware profile
raw event/query/result logs
derived metric blocks
candidate/published memory manifests
source spans, parent/replacement edges, and every multi-store sub-root
figure/table IDs
```

Required trace:

```text
source/version
→ question/hypothesis
→ config/data/code/environment
→ raw checksum
→ derived result
→ claim
→ figure/table
→ manuscript sentence
```

## 21. Three evidence tracks

1. **Local reproducibility:** small open model and controlled stream for
   determinism, correctness, and qualitative effects.
2. **Analytical/trace-driven DSE:** arrival, reuse, capacity, cluster, and data
   movement beyond available hardware; always labeled simulation.
3. **A100 80 GB runbook:** measured utilization, energy, state traffic,
   training/serving contention, and tail latency.

H100 or other accelerators are separate future measurements, never inferred
from A100 or simulation.

## 22. Failure injection

Test:

```text
worker killed before/after candidate write
partial/truncated object
vector committed while metadata is absent, and the reverse
payload rewritten while the old embedding/key remains visible
new embedding/key visible while the old payload remains visible
topic delete before index publication
index publication before topic readiness
staging cleared before serving-manifest CAS
stale compare-and-swap
out-of-order asynchronous completion attempts to publish an older generation
index corruption
node loss
poisoned wake event or generated dream
deletion during pinned read
base-model/embedding upgrade
queue overload and deadline miss
rollback predecessor made unauthorized
concurrent foreground query at every multi-store publication step
```

Freeze RTO, RPO, duplicate/lost-event tolerance, deletion precedence, and
maximum stale-read bounds before systems claims.

## 23. Stop and claim-down rules

- Failed primary hypothesis follows its frozen null-paper branch.
- A detectable sub-threshold effect is not material support.
- An equivalence interval overlapping its margin is inconclusive, not
  equivalent.
- A simulation cannot support absolute hardware latency or energy.
- A product document supports product behavior, not comparative effectiveness.
- Reducing active memory while keeping an archive is not total compression.
- Behavioral forgetting is not physical deletion.
- No new confirmatory arm is added after G2.
- Unused compute cannot be spent on post-hoc hypothesis rescue.

## 24. Minimum publishable result

The benchmark succeeds as a research artifact even if the router fails, if it
produces:

1. a deterministic, leakage-tested lifetime stream;
2. matched external, latent, user-parametric, static-mixture, and hybrid arms;
3. complete lifecycle resource accounting;
4. retention and acquisition separated;
5. at least one well-estimated crossover, capacity knee, or informative null;
6. reproducible raw-to-claim provenance;
7. explicit deployment and biological nonclaims.

This makes a negative paper scientifically useful rather than a failed demo.
