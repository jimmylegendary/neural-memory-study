# Open Questions, Answerability, and Hypothesis Matrix

- Cutoff: 2026-07-25
- Status: pre-registry research control artifact
- Purpose: distinguish questions answerable from existing primary evidence,
  questions requiring the program's controlled benchmark, and questions that
  remain deployment research

## 1. Answerability levels

| Level | Meaning | Permitted output |
|---|---|---|
| `A` | primary literature and audited implementations can answer now | evidence-backed synthesis with limitations |
| `B` | existing work narrows the alternatives, but a matched controlled experiment is required | preregistered hypothesis and experiment |
| `C` | answer depends on production workload, elapsed time, privacy, or hardware fleet | systems measurement plan; no universal conclusion |
| `D` | current evidence cannot support the proposed theorem or scaling-law form | explicit non-claim and conditions for future evidence |

An answer can have more than one level. For example, existing papers establish
that generated replay is not memory-free (`A`), while the best replay mixture
for a particular long-lived agent is workload-dependent (`B/C`).

## 2. RQ1 — Is sleep a computational phase or only a scheduling label?

**Current answerability: `A/B`.**

Existing work establishes several delayed transformations:

- SIESTA updates a large upper-network fraction during explicit sleep after
  backpropagation-free wake.
- PCMC retrains a representation, re-indexes memory, and prunes patches during
  sleep.
- Spens et al. learn which offline operator to run.
- Auto-Dreamer consolidates cross-session external memory offline.
- PlugMem transforms completed WebArena trajectories after a task boundary for
  later benchmark tasks; H-EPM's claimed online test-memory path does not
  disclose the ordering needed to place it on the sleep side of that boundary.
- MemMA verifies and repairs provisional memory after a completed dialogue
  session; its paper-aligned probe generator is not public.
- MemEvolve and ALMA evolve executable memory architecture after completed
  batches/design evaluations; ALMA also updates content after a task reward is
  fixed. These are controlled experimental boundaries, not autonomous daemons.
- RHO retrospectively transforms completed trajectories into a persistent
  instruction/skill/tool harness and evaluates the promoted winner on held-out
  tasks. This is an experimental boundary pass, but only one manually launched
  optimization round rather than a production sleep service.
- Surprise-Gated Memory periodically replays completed labeled embedding
  tasks into a slow readout and separately clears one-shot facts from a fast
  store into slow prototypes. Both are strict proof-of-concept boundaries,
  though the base encoder/VLM stays frozen and code/full settings are absent.
- Evolve explicitly separates foreground acquisition/TTL refresh from a
  manual or scheduled section-compilation cycle, then reuses canonical
  sections later. Its external lifecycle passes, although its evaluation
  repeats the same questions and therefore does not establish semantic
  transfer.
- Memento-Skills' paper loop rewrites skills and retries the same current
  question, so it is wake learning. The repository's DreamDaemon appeared 34
  days later and does pass at the code level; it cannot be used retroactively
  to relabel the paper or explain its benchmark gains.
- Microsoft HIMA specifies scheduled external tier consolidation.
- Google *Language Models Need Sleep* transfers fast parametric state into
  slower experts.
- Microsoft MEMENTO compacts reasoning blocks inside one still-running
  generation. It is a useful learned-memory comparator but fails the boundary:
  same-call downstream tokens need the output and no later wake reads it.

Therefore sleep can be defined operationally without a biological claim:

\[
T(\text{immutable wake snapshot},\text{incumbent state},\text{budget})
\rightarrow
\text{candidate persistent state}.
\]

What remains untested is whether delayed execution itself improves the
lifetime frontier after total compute, data, and destination are matched.

**Resolution experiment.** Compare immediate test-time update, delayed
snapshot update, and no update with identical events, optimizer steps, replay
bytes, model state, and future queries. Charge snapshot, queue, and promotion
cost.

**Reject sleep-specific benefit if:** delayed execution has no utility,
latency, safety, or amortization advantage over an otherwise identical
immediate update.

## 3. RQ2 — What should be consolidated?

**Current answerability: `A/B`.**

The literature answers what is possible, not the globally optimal selector.
Replay can be uniform, balanced, reservoir-based, value-ranked,
novelty-filtered, generated, or source-conditioned. Rare exact facts,
procedures, preferences, failure traces, corrections, and reusable
representations have different future-query values.

The current normative candidate is prioritized expected value:

\[
\text{priority}_i
\propto
\text{gain}_i
\int_0^H
\lambda_i(t)S_i(t)e^{-\omega t}dt,
\]

subject to redundancy, risk, privacy, and capacity shadow prices. This is a
program hypothesis, not an established law. Here \(H\) is measured in seconds,
\(\lambda_i\) in reuse/s, \(S_i\) is dimensionless survival probability, and
\(\omega\) is a s\(^{-1}\) discount rate. A cycle-indexed alternative must
redeclare every rate per cycle rather than mix the two clocks.

**Resolution experiment.** Freeze a lifetime stream with rare high-value
events, frequent low-value duplicates, contradictions, failures, and
corrections. Compare uniform, recency, surprise, oracle, learned, and
gain-times-need selection at equal admitted bytes and sleep compute.

**Extension question.** Does selector error compound faster than
transformation error? A poor admission decision can make later reconstruction
impossible even when the consolidator itself is accurate.

## 4. RQ3 — Which destination should receive each memory?

**Current answerability: `A/B/C`.**

Primary evidence supports all of the candidate destinations:

```text
raw text/event archive
semantic text or summary
vector/graph index
latent/KV state
user/session adapter or expert
shared model weights
deliberate no-write/forget
```

No audited paper learns and validates a joint router across this whole set.
External memory preserves provenance and editability but suffers retrieval
competition and context cost. Parametric memory can amortize frequent reuse
but suffers interference, addressability failure, and difficult deletion.
Latent/adapters occupy the middle as serving caches.

MEMENTO makes the latent destination less simple than “a saved vector.” Its
restart ablation holds summary text constant while removing block-conditioned
KV and loses `15.3 pp` on AIME24. The effective destination is therefore a
compound state \((\text{text},\text{KV},\text{model},\text{mask history})\);
portability, deletion, and capacity comparisons that price only the visible
text are invalid.

LoRA-as-memory adds a second retrieval boundary. Isolated low-rank modules can
reduce interference under a fixed trainable-parameter budget, but the system
must select the right adapter and sometimes compose several adapters.
Oracle-routed modules outperform one larger adapter in the reported matched
setting; learned routing can remove that gain, and naive concatenation can
collapse under merge interference. Destination choice is therefore not one
decision:

```text
event → destination
destination → item/document/module/shared behavior
selected state → compatible composition → answer
```

ImprintBench further expands “successfully internalized” beyond direct recall
to temporal update, reference, composition, relevance, and scope detection.
The destination experiment must score all six capabilities before treating a
weight or adapter write as a semantic substitute for external evidence.

**Resolution experiment.** Route the same events to every destination and a
hybrid under the same information boundary, future queries, and preregistered
native-resource caps for retained/peak bytes, live tokens, sleep FLOPs, and
governance. Latency, bytes moved, and energy are measured outcomes rather than
quantities forced equal. Cross single/routed/merged adapter pools with external
retrieval, and report oracle-router versus learned-router gaps. Hold the base
checkpoint fixed where possible.

**Preregistered link:** `H-STC-001`, `H-STC-002`, and `H-STC-004`.

**Extension hypothesis.** The optimum is not a single medium but a hysteretic
promotion/demotion policy whose thresholds depend on reuse, volatility, and
recovery cost.

## 5. RQ4 — When should sleep run?

**Current answerability: `A/B/C`.**

Fixed triggers already exist: full short-term slots, every 10 classes, every
120K samples, task midpoint, known task boundary, a completed task or dialogue
session, a claimed online successful-trajectory update, fixed multi-frequency
transfer, or a nominal six-hour batch. PlugMem provides implemented
task-boundary evidence and MemMA provides implemented session-boundary
probe-and-repair; H-EPM motivates a success-gated trigger but leaves its
boundary timing unresolved. RHO adds a completed-retrospective-batch/manual
trigger: its Codex workflow can mine finished sessions, but exposes no
autonomous periodic, pressure, or governance-deadline scheduler. Spens learns
the operator within a sleep window, but does not solve production window
opening. Surprise-Gated Memory adds calibrated prediction error for admission
and either periodic or fast-store-full sleep, but does not report the actual
production threshold, queue load, or trigger false-positive cost. No audited
study establishes a universal wall-clock cadence.

Evolve contributes a concrete but unvalidated scheduler default: manual sleep
or 03:00 when staging contains at least one section. Memento-Skills
DreamDaemon combines a quick post-session path with a 600-second scan and
nominal 24-hour gate; the effective staging threshold is five sessions or
10 KB, not the separately declared `dream.min_sessions=2`. These are
implementation points for a trigger sweep, not evidence that clock time or
their defaults are optimal.

TiMem makes clock multiplicity measurable: segment/session/day/week/month
views reduce recalled context, while the full hierarchy adds roughly 25–30%
construction calls over its L1-only comparison. RecMem instead triggers
expensive transformation through similarity and recurrence; GAM uses
pause/session-end/overflow. These results turn “when” into two coupled
questions: which clock/pressure opens a job, and which local evidence region
the job is allowed to touch.

Candidate trigger inputs:

```text
novelty and reuse estimate
retrieval/interference pressure
plasticity decline
free slot/expert fraction
staleness and correction arrival
sleep backlog and retry load
privacy/delete deadline
idle and accelerator availability
```

**Resolution experiment.** Compare one clock, nested temporal clocks, fixed
event count, capacity pressure, recurrence, surprise/interference, and learned
triggers under burst and drift streams. Cross global and local maintenance.
Report task utility, construction calls, queue stability, job-scope bytes, and
distortion by abstraction depth.

**Preregistered link:** `H-STC-003` and `H-STC-006`.

## 6. RQ5 — How should sleep data be constructed?

**Current answerability: `A/B`.**

The audited options are raw replay, coreset replay, latent replay, generated
old inputs, paraphrase/application study packets, counterfactual questions,
negative/failure trajectories, and source-grounded procedural abstraction.
The strongest available warning is that generator-only replay and recursive
self-teaching can accumulate error; small canonical anchors can have
disproportionate value.

MemMA adds synthetic **diagnostic probes** rather than only replay examples:
single-session, cross-session, and temporal questions test the provisional
store and failed probes generate repairs. This creates a new audit requirement:
the probe generator, its source context, selection/filtering, and separation
from final evaluation queries are part of the sleep dataset lineage. Committing
probe outputs without their generator, as in MemMA's paper-aligned quick start,
is insufficient for a complete leakage and reproducibility audit.

MEMENTO contributes a complementary **state-sufficiency corpus** recipe:
atomize a long trace, score local semantic boundaries, solve global
segmentation with dynamic programming, compress each block, and use an
independent rubric judge with feedback retries. Its reported `28% -> 92%`
acceptance improvement shows why one-pass synthetic summaries are a weak
sleep-data baseline. Its OpenMementos scaling also shows a distribution
warning: all 1K–100K arms improve with more data, yet matched raw
OpenThoughts training is often stronger. More generated consolidation data
does not by itself identify the correct teacher or state distribution.

LoRA-as-memory supplies a direct source-transformation ablation: raw document,
QA, summary, rewrite, and mixed supervision produce different downstream
memory from the same evidence. ImprintBench supplies the missing probe
families; MemoryBench supplies explicit/implicit feedback streams; MEMORA
extends the evidence unit to time-anchored multimodal action state. These
sources require grouped train/holdout splits so paraphrases, temporal updates,
and derived embodied questions cannot leak across the boundary.

**Resolution experiment.** Use the `D0–D10` data arms in
[the training/data atlas](TRAINING-DATA-METHODS-ATLAS.md), including
generator-only, canonical-only, internalization-complete, feedback, and
multimodal controls. Measure cycle-depth corruption, rare-event survival,
temporal correction, composition, boundary awareness, and delete lineage.

**Extension hypothesis.** There is a nonzero canonical-anchor floor below
which generated replay's recoverability margin collapses abruptly, even when
short-horizon average accuracy changes smoothly.

## 7. RQ6 — Which training method fits each transformation?

**Current answerability: `A/B`.**

Existing evidence maps methods to different targets:

| Transformation | Plausible operator |
|---|---|
| exact/semantic external memory | deterministic or LLM extract/merge/version |
| cross-session procedure | prompted or RL-trained external consolidator |
| source→compact context/KV | teacher distillation |
| fast→slow parameter | GKD/SFT into isolated adapter/expert |
| shared-weight protection | replay + EWC/SI/GEM-like constraint |
| representation refresh | contrastive/reconstruction training + re-index |
| content/operator scheduling | contextual bandit, PPO/GRPO, or deterministic policy |
| memory delivery into the acting model | SFT/RL compiler over text and latent channels |
| document→modular parametric memory | transformed supervision + isolated LoRA, learned router, interference-aware merge |
| event→multi-timescale hierarchy | deterministic temporal closure or learned local summarizer |
| recurrent/rare-event admission | calibrated similarity, recurrence, surprise, utility, and recovery-value policy |
| embodied evidence→habit/workflow | typed online edit + offline multimodal abstraction with temporal provenance |
| memory architecture/control program | evolutionary code search, learned skill selection, retrospective full-harness proposal, validation-gated promotion |
| staged external knowledge view | category-aware deterministic routing plus teacher merge/split/delete and versioned index publication |

No method dominates all transformations. The benchmark must compare a method
only where its destination and objective are meaningful.

The control-plane lineage is chronological. MemEvolve first evolves executable
`Encode/Store/Retrieve/Manage` providers from completed task batches. MemSkill
learns a selector and periodically edits an at-most-30-item natural-language
skill bank. ALMA performs open-ended search over executable database, update,
and retrieval designs. M★ later jointly searches memory `Schema`, write/read
`Logic`, and agent `Instruction` from validation failures. M★'s native
task-specific program beats transferred programs in every reported target,
while five of six transfers fall below the universal seed. Together these
results support per-workload operator specialisation. RHO then moves the
lineage closer to deployment: completed unlabeled trajectories drive
difficulty-diverse selection, group re-solves, diagnoses, three full-harness
proposals, and pairwise self-preference before a winner is used later.
However, solver, diagnostician, optimizer, and ranker share the same model
family; the selected proposal is not always the hidden-test optimum, and the
paper warns that adversarial trajectories can entrench executable behavior.
None demonstrates an autonomous, independently attested, continually evolving
production control plane.

MemCompiler is orthogonal: SFT and GRPO learn how to compile a read-only memory
into immediate text guidance and latent soft tokens conditioned on current
Brief State. It improves delivery, but runs per action and therefore cannot
answer whether the underlying memory should be consolidated during sleep.

MEMENTO adds another orthogonal curriculum: full-attention SFT teaches format,
block-masked SFT creates state-sufficiency pressure, and rule-reward CISPO
partially recovers accuracy while increasing KV use. This suggests testing
`format imitation → constrained consolidation → downstream-utility RL` as
separate stages rather than attributing the result to “distillation” alone.

Surprise-Gated Memory supplies a replay-mixture counterexample. Full
interleaving protects the old readout, surprise admission halves one buffer
without measured loss, but a latest-five-task window is worse than no replay.
The method comparison must therefore factor admission from rehearsal support:
`all-history`, `recent`, `stratified coreset`, `surprise-only`, and `no replay`
are different training distributions, not interchangeable implementations of
“sleep.”

Memento 2 and the Memento-Skills paper are essential wake baselines for
external procedural learning: cases, policy/value state, or skills can improve
without any delayed phase. The later DreamDaemon and Evolve supply sleep-side
external transforms, but use LLM inference rather than an optimizer. Their
proper training analogue is to learn admission, overlap, merge/split/delete,
and trigger policies while keeping publication and governance constraints
non-learned. Evolve's fixed `.85` similarity threshold and 42-category
taxonomy are ablation factors, not a validated universal operator.

**Resolution experiment.** Factor data, objective, destination, and cadence
instead of comparing named systems as indivisible packages.

**Extension question.** Can the router learn a low-dimensional sufficient
policy over workload statistics, or does each memory class require a distinct
controller? More sharply, which components should be universal, tenant-specific,
task-specific, or re-searched after drift, and can a hierarchical harness avoid
M★'s transfer failures without multiplying validation and governance cost?

## 8. RQ7 — What bounds lifetime capacity?

**Current answerability: `A/B/D`.**

The literature establishes bounded-state overwrite, interference, finite
expert/slot pools, replay/generator state, retrieval competition, external
store growth, and loss of plasticity. It does not establish one transformer
memory-capacity exponent.

Controlled LLM studies provide useful but incompatible conditional anchors:
roughly two knowledge bits/parameter for factual-tuple storage and flexible
extraction, and roughly 3.6 memorized bits/parameter when uniform synthetic
sequences remove generalization. Different stored objects, readouts, precision,
architecture, and training protocols prevent treating either number as a
universal constant. LMLM and LaCy further show that models can learn to
externalize selected factual targets, but LaCy's factual offloading does not
significantly improve tested NLU. “Fewer facts in weights” and “more usable
reasoning capacity” are separate estimands.

Recent graph systems sharpen the external case. PlugMem's finite HotpotQA merge
soft-deactivates nodes and reduces active semantic nodes only `5.0%` at its
stricter threshold; H-EPM lets an edge accumulate multiple state summaries and
specifies no eviction. Both can improve later-task utility while leaving total
physical state monotone. Admission and active-index compactness are therefore
not evidence of lifetime equilibrium.

GAM, RecMem, TiMem, and modular LoRA make that distinction concrete in four
different media. GAM bounds a 2,048-token progression buffer while its topic
and event graphs plus raw archive grow. RecMem reduces synthesis calls but
retains an unbounded verbatim store whose removal sharply hurts accuracy.
TiMem shrinks recalled context while adding hierarchy levels and construction
calls. Low-rank adapters bound one module while adapter count, catalog state,
router, merge workspace, and GPU residency can grow. The capacity ledger must
therefore include:

```text
retained-byte ledger:
  B_r raw/canonical evidence
  + B_e external text/vector/graph payload and index
  + B_l latent/KV state
  + B_p parametric modules
  + B_a auxiliary catalog/router/optimizer state
  + B_v version/lineage/recovery state

incremental peak-state ledger:
  B_peak,state = B_retained
  + disjoint candidate/staging + reader-pinned predecessor
  + dual-index migration + build/merge workspace + temporary replicas

transient-overhead diagnostic:
  B_peak,state - B_retained

physical fleet/HBM ledger:
  B_peak,physical =
    max_t [B_base,resident(t) + B_peak,state(t) + B_execution(t)]

cumulative work ledger:
  read + rewrite + construction + validation + migration + reclamation work
```

The six retained classes are mutually exclusive physical ownership classes
for incremental memory state.
Content-addressed bytes are charged exactly once; physical replicas are
charged where they reside. `hot`, `active`, `warm`, `cold`, and `archive` are
overlapping residency/lifecycle projections of that partition, never extra
summands. Every frozen-base copy on wake/sleep fleets and non-state execution
memory is separately charged in the physical peak. The three physical terms
are aligned at the same maximizing timestamp; independent component
high-water marks are never added. Peak sampling is reconciled against
allocator and storage telemetry before any bounded-cap claim.

At least five scientific capacities must be separated:

1. physical storage;
2. retention/representation;
3. addressability/application;
4. plasticity/acquisition;
5. governance/recoverability.

Sleep service rate and hot-state residency are additional operational
bottlenecks.

There is also a restricted information-theoretic answer. Let \(\Theta_0\) be
the frozen base/public side information and \(M_n\) every finite-precision
incremental mutable carrier of the evaluated stream. If
\(H(M_n\mid\Theta_0)\le b_{\mathrm{mutable}}\) bits and the controlled answers are
conditionally independent under the declared \((Q_i,\Theta_0)\) source model,
constant-distortion joint recall requires

\[
\sum_i R_{Y_i\mid Q_i,\Theta_0}(D_i)
\le I(Y_{1:n};M_n\mid Q_{1:n},\Theta_0)
\le H(M_n\mid\Theta_0)
\le b_{\mathrm{mutable}}.
\]

The frozen base is reported separately; if it is changed to encode the stream,
its delta or checkpoint is charged to \(M_n\). An open-ended positive-entropy
stream therefore cannot remain uniformly accurately recallable in fixed
mutable state. Expansion, genuine redundancy exploitation, selective
admission, a bounded retention horizon, or explicit distortion/forgetting is
eventually necessary. Moving bytes to an external archive changes the
accounted boundary but does not make the information free.

The complete-state qualifier is empirical as well as theoretical. MEMENTO
physically frees old block KV but downstream memento KV retains decodable
random-passcode information and materially improves task accuracy. “Deleted
source tokens,” “small visible summary,” and “low peak KV” therefore do not
establish bounded semantic state, privacy erasure, or low memory-time. AUC,
implicit latent state, model dependence, and restart/migration loss must be
included.

Procedural memory has the same accounting problem. RHO fixes coreset,
rollout-group, and proposal counts for one update, while promoted files,
executable tools, content-addressed versions, source trajectories, diagnoses,
backups, and run logs can accumulate across rounds. A prompt-file byte limit
therefore does not prove bounded harness capacity; the experiment must cap and
report the complete transitive artifact graph.

Surprise gating reduces admitted samples in one finite stream but does not
solve lifetime capacity. A classwise core grows with the number of classes,
slow prototypes grow with taught facts, and full interleaved replay makes one
cycle increasingly expensive. The paper's recent-window collapse also shows
that a constant replay window can cross an interference knee before simple
retention metrics signal a safe reclamation point.

External topic and skill systems expose two additional hidden capacities.
Evolve clears staging and reduces one finite canonical store by roughly a
third, but novel topics can still grow without bound and unqueried expired
canonical entries are never reclaimed. Memento-Skills injects at most three
topic documents after Dream, yet its skill-selection path can expose the
entire catalog description set to the model. Bounded retrieved output is
therefore compatible with unbounded persistent bytes, selection work, and
governance descendants.

Memento 2 provides a conditional theory bridge but not a capacity solution.
Its value bound improves as coverage radius \(r_M\) and retrieval error
\(\delta_M\) fall, while its theorems assume finite current memory and
stationary evaluation dynamics. Under a separately verified compact metric
support with covering dimension \(d\), one may test the restricted candidate
\(r_M=\Theta(M^{-1/d})\) only when codec/precision and bytes per prototype are
fixed. A separate fixed-active-representation-bit experiment must subtract
headers, index/keys, metadata, and decoder state from its complete active
budget, measure achieved \(b_{\rm payload}(M)\), and fit geometry together with
codec distortion, for example
\(cM^{-1/d}+q(b_{\rm payload}(M))\). Raw evidence, lineage, and rollback bytes
remain separately reported, so this is not a fixed-total-memory claim. Open-ended or shifting
support, merge distortion, and retrieval competition can violate that bridge;
no transformer-wide capacity exponent follows from the theorem.

**Resolution experiment.** Sweep association load, bytes, context, latent
state, trainable parameters, sleep compute, and cycles. Measure capacity knees
separately by workload and query class. Add an opaque independent-payload arm
with known entropy so the finite-state bound is tested without relying on an
LLM judge or an unmeasurable natural-language “fact count.”

**Preregistered link:** `H-STC-004`, `H-STC-005`, `H-STC-006`, and
`H-STC-007`.

**Forbidden inference.** Benna--Fusi \(N/\log N\), Lahiri--Ganguli
\(O(\sqrt{NM})\), or bounded-synapse exponents may not be transferred directly
to transformer factual capacity.

## 9. RQ8 — What scaling laws govern the system?

**Current answerability: `B/C/D`.**

Existing papers provide isolated curves: query amortization, replay-memory
size, latent slots, expert sizes, active external budgets, and training
throughput. None establishes a deployed cross-substrate sleep scaling law.

The program will test, not assume:

- reuse break-even between live retrieval and compilation;
- repeated-key cache amortization versus held-out paraphrase/composition
  amortization;
- retrospective-harness gain against its trajectory-selection, 103-call
  one-round search, repeated-round validation, and artifact-growth cost;
- optimal cadence under fixed per-sleep cost and delay/interference cost;
- queue stability under novelty arrival;
- rate-distortion-action trade-offs for future queries;
- capacity knees and min-bottleneck versus smooth interaction surfaces;
- adapter rank/module count against router error, merge interference, and
  catalog residency;
- temporal-clock count against construction calls, queue age, and repeated
  abstraction distortion;
- local maintenance radius against downstream utility and graph rewrite cost;
- bounded hot state against total durable bytes and cumulative lifetime work;
- recoverability and plasticity boundaries.

**Scaling-law admission rule.** At least eight points per scale axis, two
held-out scales/configurations, alternative functional forms, normalized
resource units, and workload-family holdout survival admit at most an
`A100-regime scaling law` on the required A100 track. An unqualified,
cross-hardware, or systems law additionally requires a prospectively mandatory,
powered second-hardware holdout. Otherwise the result is an empirical curve.

**Extension hypothesis.** Scaling is piecewise: different capacity pressures
bind in different regimes, so a segmented/min-bottleneck model will predict
held-out workloads better than one global power law.

## 10. RQ9 — What infrastructure is required?

**Current answerability: `A/B/C`.**

Evidence supports a four-plane architecture:

```text
wake plane
  low-latency inference, fast state, append-only event/WAL

sleep plane
  immutable snapshot, replay/dream generation, FT/RL/graph compaction

promotion/control plane
  retention, acquisition, safety, privacy, lineage, atomic publish/rollback

memory plane
  raw/semantic/vector/graph/latent/adapter/checkpoint hot-warm-cold tiers
```

No paper demonstrates the integrated transactional system. The key data
movement proposal is to colocate shard-local source and embeddings, send
content-addressed deltas or sufficient statistics to sleep workers, and
publish small versioned adapters/index roots rather than repeatedly moving raw
histories.

MEMENTO adds a serving constraint: a latent artifact may be **stateful but not
re-prefillable**. A memory plane must record whether an object is portable
text, serialized latent state, or live KV lineage tied to an exact
model/tokenizer/mask history. Restarting, moving, quantizing, or upgrading it
may change behavior even when the visible tokens and digest match.

Evolve and DreamDaemon add a storage-consistency constraint. A sleep candidate
may span vectors, relational metadata, topic files, index roots, and a staging
range. Per-database atomicity does not make that candidate atomic. Foreground
readers need one immutable manifest/root version; source staging can be
acknowledged only after validation and pointer swap, and a crash must support
idempotent replay or discard.

MIRROR adds an ordering constraint even when the destination is one bounded
text object. Asynchronous turn \(g\) must publish against its expected parent
and only while its generation exceeds the served generation. A worker that
finishes successfully but late is stale, not authoritative. Worker caps,
deadline/fallback behavior, cancellation, and later-wake wait time belong in
the memory protocol rather than the application thread pool.

LightMem adds a semantic-consistency constraint within a vector store itself.
Its audited update path can rewrite the visible memory text while preserving
the old embedding vector. Payload, retrieval key, encoder digest, provenance,
and authorization watermark must therefore be one generation-bound object.
“One database write succeeded” is not enough if its key now describes an
earlier payload.

Modular parametric and multi-timescale memory add three services. First, an
adapter registry must track source lineage, rank, compatibility, tenant/version
scope, router index, merge policy, accelerator cache residency, and
revocation status. Second, a hierarchical scheduler must coalesce
session/day/week/month jobs while preventing overlapping views from rewriting
the same evidence independently. Third, embodied memory needs time-anchored
multimodal evidence and consent/tombstone propagation from video/action spans
to entity, habit, workflow, preference, and parametric descendants.

**Resolution experiment.** Compare one shared fleet, time-partitioned fleet,
and separated wake/sleep clusters under the same arrival trace. Measure wake
P99 interference, bytes moved, accelerator utilization, per-clock calls,
adapter cache/routing misses, local-versus-global rewrite scope, backlog,
promotion latency, and failure recovery.

**Extension hypothesis.** A separate sleep cluster helps only beyond a burst
and interference threshold; below it, network movement and idle capacity make
time-sharing superior.

## 11. RQ10 — How do provenance, privacy, and unlearning change the optimum?

**Current answerability: `A/B/C`.**

The answer is structural: a memory that cannot be traced, invalidated,
reconstructed, or deleted has an additional lifecycle cost. This cost is
usually omitted from model-centric comparisons. Zep/Graphiti, Auto-Dreamer,
Letta Code, and HIMA contribute partial external lifecycle mechanisms; none
proves item-level deletion across external, latent, adapter, shared-weight,
checkpoint, and backup descendants.

MEMENTO supplies a concrete privacy falsifier: place an opaque secret only in a
soon-evicted source block, then probe every surviving explicit and latent
descendant. Its above-chance downstream KV recovery shows that physical
source-block reclamation is not sufficient evidence of semantic erasure.

RHO supplies the control-plane analogue: a poisoned completed trajectory can
be summarized into persistent instructions or executable tools. A positive
score from a correlated self-preference judge demonstrates comparative task
utility, not authorization, non-escalation, source-level deletability, or
independent safety. Promotion must therefore bind candidate provenance,
capability sandboxing, canary evidence, and a recoverable incumbent digest.

Evolve supplies a provenance counterexample. Its released evaluation questions
may contain source URLs and excerpts, but acquisition sends only question and
category to the teacher, and stored sections lack source spans, parent
lineage, model/prompt digest, tenant, or authorization scope. A generated
section ID is operational traceability, not evidence provenance. DreamDaemon
has the same derived-view issue: deleting a topic file does not prove that its
source session, index descendants, backups, or later skills were invalidated.

**Resolution experiment.** Inject correction, consent withdrawal, poisoning,
and deletion events. Require source-to-derived dependency traversal, denial
before physical reclamation, bounded deletion latency, rebuild or quarantine
for parametric artifacts, and rollback after a bad promotion.

**Extension hypothesis.** Governance shadow price moves the parametric
promotion boundary toward external memory as volatility or deletion
probability increases, even when raw inference utility is unchanged.

## 12. RQ11 — Are NREM and REM useful computational distinctions?

**Current answerability: `A/B/D`.**

Neuroscience supports replay and state-dependent consolidation, but not a
one-to-one mapping from sleep stages to LLM operators. PAD and WSCL
operationalize NREM/REM-like phases by design; Google Sleep pairs fast-to-slow
knowledge seeding with synthetic dreaming. These are useful ablation
factorizations, not biological validation.

**Resolution experiment.** Compare:

```text
replay/distillation only
novel counterfactual generation only
sequential replay→dream
sequential dream→replay
interleaved joint objective
matched extra data/compute control
```

Evaluate robustness, exact retention, composition, hallucination, and drift.

**Extension hypothesis.** Stage separation helps only when conservative
replay and exploratory generation have materially conflicting gradient or
verification requirements.

## 13. RQ12 — Which state should be fixed, user-local, or shared?

**Current answerability: `A/B/C`.**

The safest current decomposition is:

```text
fixed/shared base
tenant or cohort policy modules
user/session fast adapters or experts
external canonical and derived memory
ephemeral query state
```

Shared updates amortize reuse but create cross-user interference, privacy, and
rollback blast radius. User-local state limits blast radius but creates
adapter residency, routing, cold-start, and long-tail utilization costs.

**Resolution experiment.** Compare session, user, cohort, and shared promotion
with identical event reuse and privacy labels. Measure utility, negative
transfer, HBM residency, offload, routing load, delete fan-out, and rollback
scope.

**Extension hypothesis.** Promotion scope should be selected jointly with
memory medium: a user-local adapter may dominate a shared weight even at lower
raw compression because its governance and interference costs are bounded.

## 14. Frozen preregistration candidates

These hypotheses are already promoted into the research design and must be
frozen at G2 before confirmatory results.

| ID | Hypothesis | Independent falsifier |
|---|---|---|
| `H-STC-001` | optimal representation changes with future-query distortion | no representation-order reversal under preregistered query shifts |
| `H-STC-002` | reuse and volatility define conditional destination regions | after the preregistered monotone single-crossing eligibility conditions hold, the predicted destination ordering/crossover is absent on held-out streams; failure of those conditions narrows the claim instead of falsifying its conditional antecedent |
| `H-STC-003` | switching costs can produce promotion/demotion hysteresis | after the derivable sufficient conditions hold, promotion and demotion thresholds are not separated or the two-threshold policy fails its frozen anti-thrashing contrast; unmet sufficient conditions yield an empirical policy surface, not falsification |
| `H-STC-004` | in at least one of exactly seven preregistered cells, a capacity diagnosed as binding before TEST limits utility | CALIBRATION freezes binding/nonbinding load rules and requires at least four eligible cells or the global result is `INCONCLUSIVE_ANTECEDENT`; separate \(+25\%\) interventions require binding-effect LCB \(>0.01\) and a simultaneous two-sided 95% nonbinding interval within \(\pm0.005\) under one centered 21-margin family, with boundary-null any-false-equivalence control. If every eligible binding-effect UCB is below \(0.01\), falsify; nonbinding gain or diagnostic failure makes that cell ineligible/narrowed and cannot be relabeled |
| `H-STC-005` | reversibility reserve can have a non-monotone optimum | every preregistered eligible regime spans an adequately powered archive range and remains monotone; a monotone result in only one regime or over an insufficient range narrows the claim but does not falsify its “at least one regime” form |
| `H-STC-006` | a past-only recoverability proxy can schedule intervention | proxy is uncalibrated, or the same operator immediately before the frozen boundary fails its matched-compute comparison with after-boundary intervention and fixed cadence |
| `H-STC-007` | acquisition can fail before aggregate retention | no high-interference regime shows the order; recycle fails separately |

## 15. Exploratory hypotheses not yet preregistered

These are research-generating propositions. They may become later hypotheses
only after source audit, estimand, assumptions, owner, and falsifier are
complete. They cannot be promoted after looking at confirmatory outcomes.
Routing/gating, multi-version atomic publication, and background
merge/compaction are known adjacent primitives; this list does not claim their
invention. Any novelty claim must survive an element-by-element systems
prior-art audit and be limited to an STC-specific lifecycle integration or
matched empirical result.

### X-STC-A — Transactional-compaction advantage

A delayed memory transformation with snapshot isolation, canary validation,
atomic promotion, and rollback will have a better tail-risk/lifecycle-cost
frontier than an accuracy-matched in-place update.

### X-STC-B — Canonical-anchor phase boundary

Generator replay has a minimum canonical-anchor fraction below which
recoverability and rare-event survival degrade faster than average accuracy.

### X-STC-C — Teacher-corruption exponent

Recursive self-teaching accumulates error with cycle count faster than a
frozen-teacher or periodically refreshed-teacher design at equal compute.

### X-STC-D — Stage-count crossover

Adding fast/medium/slow tiers improves lifetime utility only until transfer
error, metadata, read amplification, and governance debt exceed the next
time-scale benefit.

### X-STC-E — Data-locality crossover

Moving content-addressed deltas and trainable state to shard-local sleep
workers beats moving raw wake histories only above a reuse/size threshold;
below it, coordination overhead dominates.

### X-STC-F — Plasticity-first capacity alarm

The earliest useful system alarm is a vector combining acquisition gain,
free/recyclable capacity, representation rank, and backlog—not old-task
retention alone.

### X-STC-G — Diagnostic-probe value curve

Provenance-complete adaptive probes should show diminishing marginal
downstream failures detected per verification cost, with probe-type coverage
delaying the saturation point. The hypothesis fails if generated probes do not
improve held-out error localization over equal-cost fixed probes, or if their
gain disappears after strict evaluation-leakage controls.

### X-STC-H — Governed harness-specialisation crossover

A fixed governance kernel plus task/tenant-specific write-read modules should
outperform both a universal harness and fully task-specific generated programs
above a heterogeneity threshold, after charging re-ingestion, code generation,
sandboxing, migration, validation, version count, and rollback. A monotone
winner with no crossover would reject the proposed intermediate regime. RHO
makes this directly testable: compare its fully rewritten harness against a
frozen governance kernel with generated task modules, using independent
promotion judges, repeated optimization rounds, poison canaries, and a fixed
artifact budget.

### X-STC-I — Delivery-compilation crossover

A learned text/latent memory compiler should improve total wake-path
utility-cost only when the executor savings exceed compiler, state-maintenance,
and model-residency cost. Reporting executor latency alone will overstate the
benefit. The crossover should move with retrieved-context length, executor
size, reuse, and compiler batching.

### X-STC-J — Explicit/implicit state-completeness gap

For learned compression, the utility and privacy gap between a nominal artifact
and its complete execution state should grow with compression pressure and
stateful reuse depth. A valid portable artifact either preserves the
answer-relevant latent channel explicitly or passes restart-equivalence and
secret-residual tests. The hypothesis fails if text-identical restart,
serialization, model migration, and deletion remain equivalent across the
preregistered pressure and hop sweeps.

### X-STC-K — Repeated-key/semantic-transfer gap

The apparent amortization benefit of external consolidation will be
systematically larger on exact repeated acquisition queries than on held-out
paraphrase, composition, correction, and distribution-shift queries. The gap
should widen as the compiler overfits query wording or category boundaries.
It is rejected if equal-cost compiled memory preserves its teacher-call and
quality advantage across frozen unseen-query families with no material
reliability loss.

### X-STC-L — Catalog-selection tax

For external skills and procedures, total wake cost will eventually be
dominated by catalog selection, description prompting, compatibility checks,
and governance rather than by the few artifacts injected into execution. A
hierarchical bounded router should cross below flat catalog prompting at a
measurable catalog size. No crossover, after charging router construction and
miss recovery, rejects the claim.

### X-STC-M — Coverage-dimension capacity law

For a fixed workload representation, the fixed-codec cardinality arm should
obey an \(M^{-1/d}\)-style geometry relation while total bytes grow. A distinct
fixed-active-representation-bit arm should expose the geometry--quantization
trade-off through achieved net payload bits after
header/index/metadata/decoder overhead, using a joint candidate such as
\(cM^{-1/d}+q(b_{\rm payload}(M))\). The claim fails if the
appropriate frozen relation does not predict held-out scales/workloads or if
estimated \(\delta_M\) reverses the benefit. This is a conditional
external-memory hypothesis, not a universal LLM fact-capacity law.

### X-STC-N — Representation-refresh amplification

For an indexed external store, item-operation work should scale with the
declared affected set \(A_c\). Local fixed-encoder rewrites usually make
\(A_c\) the changed items. Whole-store refresh is a conditional lower bound
only for single-current-encoder serving with no federation or query
translation; under that contract a linearly growing store and one incompatible
encoder change per cycle imply quadratic lifetime work. Federation and
translation are valid alternatives only when resident encoder/index bytes,
query fan-out/calibration, migration, and deletion fan-out are charged. The
claim fails if measured work does not scale with \(A_c\), or an alternative
hides its displaced costs.

### X-STC-O — Bounded-state/lifetime-work separation

A consolidator can maintain constant-size published state while lifetime
transformation work grows linearly or faster with events. Reconstructing a
fixed-size narrative every turn should cross a no-rewrite or event-triggered
policy once marginal quality gain falls below rewrite, verification, and
error-recovery cost. The claim fails if a matched full-rewrite policy remains
strictly dominant across the preregistered horizon, pause, and volatility
sweeps after all LLM calls and later-wake waits are charged.

### X-STC-P — Usable-capacity release after externalization

Removing factual targets from parametric training will improve non-factual
acquisition only if the released representational/optimization capacity is
reachable by the new objective rather than remaining unused or being consumed
by call-policy learning. At matched gradient-token, parameter, and external
call budgets, the hypothesis requires a measurable improvement in fresh
non-factual acquisition or plasticity—not merely lower factual leakage. A null
result reproduces the LaCy boundary and rejects the simple “offload frees
reasoning” mechanism in that regime.

### X-STC-Q — Parametric-routing knee

At a fixed total adapter-parameter budget, modular low-rank memories should
beat one adapter only while router error, merge interference, catalog
selection, and serving misses remain below the interference saved by module
isolation. The confirmatory estimand is the aggregate oracle--deployment gap;
routing, merge, and residency attribution is extension `X-STC-Q` and remains
exploratory unless a pre-TEST amendment materializes and powers the full
\(2\times2\times2\) intervention at G2. In that promoted extension,
CALIBRATION freezes numerical route-recall LCB, out-of-catalog FPR UCB, and
serving-cost thresholds. If no router qualifies, its deployable test is
`INCONCLUSIVE`. If every qualified router's simultaneous
modular-minus-single UCB is at or below the frozen materiality margin in all
cells, the modular-region hypothesis is falsified; “competent” cannot be
redefined after TEST. Without the amendment, component effects are labeled
exploratory and neither qualify nor rescue the PAPER-C aggregate claim.

### X-STC-R — Clock-count queue crossover

Adding consolidation timescales should first improve utility per active byte,
then reduce net lifecycle utility once construction-call arrival destabilizes
the sleep queue or repeated abstraction accumulates distortion. It fails if
additional clocks remain Pareto-improving across queue load, latency, cost,
and distortion holdouts.

### X-STC-S — Internalization-completeness gap

Direct-recall parity should overstate parametric internalization; the largest
destination gaps should occur in temporal replacement, boundary awareness, or
cross-memory composition. Stable destination rankings across acquisition,
update, reference, composition, relevance, and scope under matched resources
reject the proposed gap.

### X-STC-T — Recovery-substrate floor

For volatile or low-frequency evidence, removing raw/reversible state should
create a nonzero correction and rare-event utility floor that a lossy semantic
view cannot cross. A bounded lossy view that matches archive-backed systems on
late queries, correction, one-off events, and rollback without retaining an
equivalent information-bearing state elsewhere rejects the hypothesis.

## 16. Questions the current program must not pretend to answer

- a universal biological theory of NREM and REM;
- the factual capacity of a transformer from synaptic-memory exponents;
- a universal 2- or 3.6-bits/parameter constant across natural-language,
  continual-update, and sleep workloads;
- a universal optimum number of sleep hours or cluster split;
- perfect deletion from shared foundation-model weights;
- indefinite learning from a finite expert/slot pool;
- production reliability from one benchmark lifecycle;
- novelty from database non-discovery;
- “zero forgetting” from a finite short-horizon average score.

The research can instead deliver bounded claims: operational definitions,
cross-substrate comparisons, falsifiable candidate laws, capacity knees,
transactional infrastructure requirements, and clearly scoped null results.
