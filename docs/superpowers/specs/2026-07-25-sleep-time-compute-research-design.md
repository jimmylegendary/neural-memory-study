# Sleep-Time Compute Research Design

- Date: 2026-07-25
- Status: approved high-level direction; detailed execution design awaiting
  user review
- Primary language: English paper spine; Korean NM-grade companion
- Parent evidence map: `dossier/SLEEP-TIME-COMPUTE-PRE-RESEARCH.md`

## 1. Objective

This program studies sleep-time compute as a distinct phase of lifelong machine
learning and memory management:

1. pre-training and post-training before deployment;
2. wake-time inference and test-time learning on the user-visible critical path;
3. sleep-time learning and consolidation away from that critical path.

The target is not a metaphorical survey of biological sleep. The target is a
falsifiable algorithm-and-systems theory for deciding:

- what experience should be retained, replayed, compressed, promoted, demoted,
  or forgotten;
- when consolidation should run;
- where the result should live: raw text, vector index, graph, latent/KV state,
  session adapter, sparse parametric memory, or shared weights;
- how the decision changes with future reuse, interference, staleness,
  provenance, privacy, compute, storage, and data-movement constraints.

The work succeeds only if it yields a defensible research paper, not merely an
annotated bibliography, and if every central conclusion is connected to
evidence, an experiment, a derivation, or an explicit hypothesis label.

## 2. Intended reader and decision

The primary reader is an AI systems or architecture researcher familiar with
transformer inference, retrieval, parameter-efficient tuning, and continual
learning.

The paper should let that reader decide:

> For a long-running agent with a bounded lifetime budget, which memory
> transformation should be executed on the wake path, which should be deferred
> to sleep, and which representation should serve each class of future query?

## 3. Positioning

### 3.1 What adjacent work already establishes

The field already contains separate lines of work on:

- biological replay and systems consolidation;
- continual learning, rehearsal, regularization, and parameter isolation;
- test-time training and stateful sequence models;
- external agent memory in text, vector, graph, and hybrid databases;
- offline memory construction, summarization, dream generation, and parametric
  consolidation;
- memory serving, retrieval, and long-context efficiency.

A recent systems characterization of ten external agent-memory systems already
measures construction, retrieval, generation, freshness, footprint, and energy.
It finds that LLM-mediated construction can dominate lifecycle energy, that
construction is predominantly a long-read/short-write prefill workload, that
asynchronous construction creates a freshness-latency constraint, and that
default stores grow monotonically without pruning. It does not unify external
memory with parametric consolidation or learn an explicit cross-medium sleep
policy. See [Agent Memory: Characterization and System Implications of Stateful
Long-Horizon Workloads](https://arxiv.org/abs/2606.06448).

The closest algorithmic neighbors already remove several broader novelty
claims. [Sleep-time Compute](https://arxiv.org/abs/2504.13171) anticipates
future queries and amortizes offline work;
[Retain or Consolidate?](https://arxiv.org/abs/2607.17545) selects among
external-memory operators under a token budget;
[Auto-Dreamer](https://arxiv.org/abs/2605.20616) learns provenance-linked
offline external consolidation;
[OSL-MR](https://arxiv.org/abs/2606.10616) learns observability-safe external
retention with delayed lifecycle costs;
[TMEM](https://arxiv.org/abs/2606.04536) combines explicit memory with online
LoRA absorption; and [Cartridges at
Scale](https://arxiv.org/abs/2606.04557) manages modular reusable KV artifacts
across GPU and persistent storage. The project therefore cannot claim to invent
future-use-aware retention, learned consolidation, sleep, external-to-parametric
transfer, or latent memory individually.

Continual-learning theory also rules out a casual promise of free, perfect
lifelong memory. General optimal continual learning can require perfect memory
and solve an NP-hard problem, while the PAC result in Memory Bounds states a
linear-in-task-count memory lower bound and a different logarithmic-pass result.
See
[Knoblauch et al.](https://proceedings.mlr.press/v119/knoblauch20a.html) and
[Chen et al.](https://arxiv.org/abs/2204.10830).

### 3.2 Missing synthesis

The missing unit of analysis is the **memory transformation over the full
lifetime**, not the retrieval method or update rule in isolation. Current work
rarely compares, at matched total budgets:

- retaining an event externally;
- compiling it into a compact external abstraction;
- compiling it into a reusable latent/KV artifact;
- compiling it into user-local or shared parameters;
- keeping it only in an active test-time state;
- deliberately forgetting it.

It also rarely prices accuracy, future query savings, interference, storage,
freshness, provenance, unlearning, and data movement in one objective.

The pre-data novelty claim is consequently narrow:

> As of 2026-07-25, the evidence intake found no primary study that jointly
> learns and evaluates, under one matched lifetime cost model, routing of the
> same memory items among raw external retention, external abstraction,
> reusable latent/KV state, and user-local or shared parametric state over many
> wake–sleep cycles, while preserving version lineage and charging
> verification, staleness, serving, data movement, rollback, and deletion.

This is a search result and falsifiable novelty hypothesis, not proof of
absence. G1 full-paper and code review must narrow or kill it if a counterexample
is located.

### 3.3 Central thesis

> Sleep-time compute is future-relevant, resource-bounded memory compilation: a
> deferred, versioned process that transforms an experience stream across memory
> media to minimize expected future-query loss and full-lifecycle systems risk
> and cost.

The primary-paper empirical thesis is deliberately conditional and narrower
than the whole research program:

> On long-horizon workloads with heterogeneous reuse and active-capacity
> pressure, a past-only policy that chooses among raw external retention,
> external abstraction, reusable latent/KV compilation, user-local parametric
> compilation, and deliberate forgetting should occupy a better held-out
> lifetime-utility/lifecycle-cost frontier than tuned single-medium and
> static-mixture policies. Every method receives the same upper-bound resource
> vector and information cutoff; live latency, bytes moved, verification,
> rollback, deletion, staleness, and policy overhead are measured and charged
> rather than falsely treated as equal inputs.

This thesis is falsified if the adaptive hybrid frontier remains inside the
envelope of the best eligible single-medium policies and the tuned deterministic
mixture under the common resource caps and information cutoff; if held-out tier
choices and gains are unrelated to the preregistered reuse and pressure
interventions; or if any improvement disappears when raw archives, provenance,
verification, rollback, deletion, staleness, serving, and controller overhead
are charged. Volatility remains an extension axis, so the first paper does not
use it to rescue a failed central claim.

The larger hybrid action space makes an oracle's weak dominance uninteresting.
The study therefore separates three estimands:

1. **oracle heterogeneity value:** whether the workload contains strict
   cross-medium complementarity at all;
2. **deployable routing value:** held-out lifetime utility of a past-only router
   versus the tuned single-medium envelope and a tuned deterministic mixture,
   after charging policy training, inference, and search;
3. **sleep scheduling value:** the effect of deferral with the events,
   destination, information cutoff, and consolidation operator held fixed,
   followed by the scheduling-by-operator interaction.

The deployable result, not oracle inclusion, carries the central paper claim.

## 4. Approaches considered

### A. Survey- and monograph-first

Build the most comprehensive chronological synthesis of neuroscience,
continual learning, agent memory, and systems infrastructure.

- Strength: maximum breadth and durable reference value.
- Risk: novelty becomes diffuse; evidence may not support a single publishable
  claim.

### B. Memory-compilation paper spine, then companion monograph

Develop one sharp theory, benchmark, scheduler, and systems cost model. Expand
the validated spine into the larger Korean work.

- Strength: every literature branch must support a falsifiable question or a
  design choice.
- Risk: requires disciplined matched-budget experiments and careful scope.

### C. Systems characterization first

Measure wake, sleep, storage, and data movement for existing implementations,
then infer architecture recommendations.

- Strength: concrete measurements and a natural MLSys/ASPLOS story.
- Risk: overlaps the 2026 external-memory characterization and may leave the
  parametric/external decision without a learned or normative basis.

### Decision

Use Approach B. Preserve the measurement discipline of C and the historical
coverage of A as supporting layers. The English paper is the claim spine; the
Korean monograph is the full argument and evidence atlas.

## 5. Operational definitions

### 5.1 Wake-time compute

Compute that is causally required for the current user-visible action, including
inference, retrieval, tool use, and immediate test-time updates.

### 5.2 Sleep-time compute

Two independent attributes are separated.

**Deferred/background compute** is a scheduling property: it is not required to
produce the current action and can be batched, delayed, interrupted, or rebuilt.

**Sleep consolidation** is an operator property: it consumes a versioned
snapshot of accumulated experience or replay, performs a cross-timescale
transformation under a future-task objective, validates the result, and updates
persistent state for later actions.

The strict sleep condition is both deferred and consolidating. A deferred
identity copy is background maintenance, while an on-path summarizer is
consolidation but not sleep. This includes weight updates and external-memory
transformation and excludes ordinary static-corpus preprocessing outside a
continuing deployed lifetime.

### 5.3 Memory media

The design space is two-dimensional rather than a binary “weights versus RAG”:

| Medium | Example state | Typical strength | Dominant risk |
|---|---|---|---|
| Active context | prompt/KV cache | exact, immediate | bounded window and live prefill |
| Wake-local mutable state | fast weights/TTT state | fast adaptation | interference and critical-path cost |
| Raw episodic store | event log/blob | provenance and reversibility | unbounded growth |
| Semantic external memory | text/facts/vector | editable, attributable | retrieval competition and write cost |
| Relational external memory | temporal/knowledge graph | conflict and relation handling | construction and schema cost |
| Latent external memory | learned tokens/KV/embeddings | compact serving | opacity and difficult deletion |
| User/session parameters | adapter/LoRA/sparse module | retrieval-free reuse | per-user serving and versioning |
| Shared parameters | model weights/memory layer | maximum amortization | cross-user interference and governance |

### 5.4 Consolidation is not automatically forgetting

If raw provenance is retained, consolidation can reduce active working state but
does not reduce total stored bytes. Total capacity becomes bounded only through
lossy compression, eviction, deletion, or transfer to cheaper archival media.

## 6. Research questions and answer contracts

The bullets below are screening contracts. Before confirmatory work, every
question receives separate fields for estimand, null, rejection rule, minimum
effect, inconclusive condition, and validity failure. A short horizon or broken
deletion contract invalidates or limits an experiment; it is not evidence for a
scientific null.

### RQ1 — Is sleep a computational phase or only a scheduling label?

- Answer with: a \(2\times2\) comparison of online versus deferred scheduling
  and identity/maintenance versus consolidating operators, holding the event
  cutoff and operator fixed where required. This separates deferral/batching
  value from cross-timescale transformation value.
- Null: after charging batching, peak resources, queue delay, staleness, and
  publication, neither deferral nor the consolidating operator has a material
  main or interaction effect.

### RQ2 — What should be consolidated?

- Answer with: marginal future utility of event selection under recurring,
  surprising, conflicting, and long-tail experiences.
- Null/rejection: learned or utility-based selection cannot beat recency, surprise,
  uniform replay, or reservoir sampling at matched budget.

### RQ3 — Where should each memory go?

- Answer with: crossover surfaces among raw, semantic, graph, latent, adapter,
  and shared-parametric destinations as reuse, query entropy, update frequency,
  privacy, and deletion requirements vary.
- Null/rejection: a single medium dominates across the controlled workload regimes.

### RQ4 — When should sleep run?

- First-paper status: exploratory/companion; it cannot rescue or withdraw a
  confirmatory PAPER-C claim.
- Answer with: periodic, pressure-triggered, idle-triggered, surprise-triggered,
  and learned cadence comparisons under arrival and freshness SLOs.
- Null/rejection: the learned trigger has no robust advantage over a fixed cadence
  after scheduler overhead is charged.

### RQ5 — How should sleep data be constructed?

- Answer with: controlled comparisons of exact replay, distilled targets,
  hard-negative rehearsal, counterfactual augmentation, synthetic dreams, and
  mixtures, with contamination and provenance audits.
- Null/rejection: synthetic or distilled data adds no benefit beyond replay, or its
  false-memory cost exceeds the retained utility.

### RQ6 — Which training method fits each transformation?

- Answer with: supervised distillation, contrastive/reconstruction objectives,
  regularized continual tuning, parameter isolation, contextual bandits, and
  RL scheduling evaluated on the same lifetime objective.
- Null/rejection: complexity beyond a deterministic policy yields no matched-budget
  frontier improvement.

### RQ7 — What bounds lifetime capacity?

- Answer with: memory growth, retrieval competition, gradient interference,
  consolidation amplification, and deletion lineage measured over hundreds of
  wake-sleep cycles.
- Null/rejection: no capacity knee or performance decline is detected within the
  preregistered horizon and precision.
- Inconclusive: the horizon, scale range, or stream diversity cannot identify
  saturation.

### RQ8 — What scaling laws govern the system?

- Answer with: fitted and derived relations for reuse break-even, optimal sleep
  interval, queue stability, rate-distortion, storage growth, and interference,
  including uncertainty and regime boundaries.
- Null/rejection: exponents fail out of sample across horizon, model size, and stream
  rate, or reduce to benchmark-specific curve fitting.

### RQ9 — What infrastructure is required?

- Answer with: an executable or trace-driven wake/sleep/memory architecture that
  measures p99 latency, staleness, throughput, energy, bytes moved, footprint,
  versioning, and failure recovery.
- Null/rejection: the proposed split gives no benefit over one shared cluster after
  utilization and transfer overhead are included.

### RQ10 — How do provenance, privacy, and unlearning change the optimum?

- Answer with: per-user versus shared promotion, deletion requests, poisoned or
  contradicted memories, lineage tracking, rollback, and cross-user leakage
  tests.
- Validity failure: a claimed efficiency frontier relies on irrecoverable parametric
  writes or unverifiable synthesized memories that violate the task contract.

### RQ11 — Are NREM and REM useful computational distinctions?

- Answer with: functional ablations, not terminology. “NREM” means faithful
  replay/compression of observed experience; “REM” means generative recombination
  or counterfactual rehearsal.
- Null/rejection: the two-stage process offers no benefit over a single mixed replay
  distribution, or dream generation increases contamination.

### RQ12 — Which state should be fixed, user-local, or shared?

- Answer with: reuse scope, interference, serving multiplicity, privacy, and
  amortization surfaces for base weights, adapters, and external stores.
- Boundary result: per-user parametric state misses serving or deletion SLOs,
  or shared promotion creates negative transfer; this identifies an
  inadmissible region rather than falsifying the question.

## 7. Normative model

### 7.1 Rate-distortion-action objective

The lifecycle is a constrained semi-Markov control problem. Decision epoch
\(n\) occurs at physical time \(T_n\):

\[
s_n=(M_n,V_n,Q_n,\mathcal{B}_n,\hat\lambda_n,\hat\mu_n),
\qquad
M_{n+1}\sim P(\cdot\mid s_n,a_n),
\]

where \(M_n\) is tiered memory, \(V_n\) its versions and deletion watermark,
\(Q_n\) the sleep queues, \(\mathcal{B}_n\) remaining resource budgets, and
\(\hat\lambda_n,\hat\mu_n\) past-only reuse and invalidation estimates. The
primary normative problem keeps task distortion and physical resources in
their native units:

\[
\min_\pi
\mathbb{E}_{d^\pi}
\left[
\sum_n e^{-\rho(T_n-T_0)}D_n
\right]
\quad\text{subject to}\quad
\mathbb{E}_{d^\pi}\!\left[\sum_n c_{n,r}\right]\le \mathcal{B}_r
\ \ \forall r.
\]

where \(d^\pi\) is the trajectory/occupancy distribution induced by the policy.
\(D_n\) is dimensionless task-loss distortion, \(c_{n,r}\) is consumption in
the native unit of resource \(r\), \(\mathcal{B}_r\) is its undiscounted
lifetime cap, \(\rho\) has unit s\(^{-1}\), and elapsed wall time—not decision
index—determines discounting. The optimization also has hard preregistered
bounds on p99 wake latency, active and durable capacity, data leakage, deletion
residual, queue stability, privacy scope, and cross-user interference. Privacy
and deletion are constraints, not costs a high-utility event can buy through.

An optional economic view is admissible only after a valuation and all resource
prices are frozen:

\[
J_{\$}(\pi)=
\mathbb{E}_{d^\pi}\!\left[
\sum_n e^{-\rho(T_n-T_0)}
\left(v_DD_n+\sum_r p_r c_{n,r}\right)
\right],
\]

where \(v_D\) is currency per unit task loss and \(p_r\) is currency per native
resource unit. Without those conversions, the empirical result remains a
constrained vector outcome and never adds loss, FLOPs, bytes, seconds, and
energy directly.

The rate-distortion subproblem for an event \(E\), future query \(Q\), answer or
target \(Y\), and compiled artifact \(Z\) is:

\[
R_{Q,Y}(D)=
\inf_{p(z\mid e):
\mathbb{E}[\ell(Y,\hat Y(Q,Z))]\le D}
I(E;Z).
\]

In implementation, the theory is tested through empirical
artifact-bytes-versus-held-out-task-loss curves. It does not assume that good
transcript reconstruction is the relevant distortion.

Exogenous-query benchmarks estimate this objective under a controlled query
process. Procedural agent tracks are analyzed separately because memory changes
actions and therefore changes the future occupancy measure; those tracks use
causal return and a robust or CVaR objective under distribution shift.

The rate-distortion component is grounded in the normative view that forgetting
should discard low-value detail before high-value structure, rather than degrade
randomly. See [Optimal forgetting: Semantic compression of episodic
memories](https://doi.org/10.1371/journal.pcbi.1008367).

### 7.2 Generalized replay priority

A first-order index for candidate \(i\), destination \(d\), and current selected
memory set \(M\) is:

\[
\operatorname{Priority}(i,d)=
\underbrace{\operatorname{Need}_i}_{\text{expected reuse}}
\times
\underbrace{\operatorname{Gain}_{i,d\mid M}}_{\text{per-hit conditional marginal gain}}
-
\underbrace{\operatorname{Cost}_{i,d\mid M}}_{\text{sleep, storage, movement}}
-
\underbrace{\operatorname{Risk}_{i,d\mid M}}_{\text{interference and staleness}}.
\]

The gain-times-need structure generalizes the normative account of prioritized
replay in [Mattar and Daw](https://doi.org/10.1038/s41593-018-0232-z). Its use
for LLM memory routing is an original hypothesis, not an established result.
Gain excludes reuse so need is not counted twice. Redundancy, complementarity,
graph clusters, and capacity shadow prices require subset selection or a
constrained knapsack; independent item ranking is only a diagnostic baseline.

### 7.3 Reuse break-even law

For candidate \(i\) over horizon \(H\), define present-value cost:

\[
K_i =
C_{\text{compile}}+C_{\text{verify}}+C_{\text{move}}
+\int_0^H c_{\text{store},i}(t)e^{-\omega t}\,dt
+P_{\text{fail},i}C_{\text{rollback},i},
\]

and present-value benefit:

\[
V_i =
\int_0^H
\lambda_i(t)S_i(t)e^{-\omega t}g_i(t)\,dt-K_i,
\]

where \(\lambda_i(t)\) is future hit intensity, \(S_i(t)\) survival against
invalidation, and \(g_i(t)\) per-hit systems saving in the same declared unit.
Compile only if \(V_i>0\), subject to quality non-inferiority and the hard
constraints.

For constant per-hit saving \(\Delta c_q>0\), this reduces to the diagnostic
break-even:

\[
N^\star=\frac{K_i}{\Delta c_q}.
\]

If task-quality change is monetized, a preregistered value coefficient can
enter \(g_i\); otherwise quality remains a separate endpoint. The \(R=0\)
condition is a no-reuse control and is never substituted into an amortization
ratio. The model predicts:

- raw append-only memory for low-reuse or rapidly changing events;
- external semantic memory for moderate reuse and high editability;
- user-local parameters for high reuse with privacy isolation;
- shared parameters only for high cross-user reuse and low conflict.

If the compiled path does not reduce wake cost at the required quality level,
the break-even denominator is non-positive and compilation is never
economically justified.

### 7.4 Sleep-cadence law

If per-sleep fixed overhead is \(C_s\) cost/job, the interval is \(\tau\), and
cycle-integrated delay/interference harm is
\(H_{\text{cycle}}(\tau)=a\tau^{p_c+1}\), the average cost rate is:

\[
\bar C(\tau)=\frac{C_s}{\tau}+a\tau^{p_c},
\qquad
\tau^\star =
\left(\frac{C_s}{a p_c}\right)^{1/(p_c+1)}.
\]

This is a candidate scaling law. Its exponent and even its functional form must
be tested across stream rates, memory media, and freshness constraints.

For fixed job overhead \(F\), event rate \(r_e\), and linear per-event delay harm
\(h\) cost/(event·time), the cycle harm is
\(h r_e\tau^2/2\), and the average-rate special case is:

\[
J(\tau)=\frac{F}{\tau}+\frac{h r_e\tau}{2},
\qquad
\tau^\star=\sqrt{\frac{2F}{h r_e}}.
\]

This predicts that optimal periodic cadence shortens with event rate. Burst and
drift experiments test whether pressure or surprise triggers improve around
this baseline.

### 7.5 Capacity and impossibility boundary

No claim will promise perfect bounded-memory learning over arbitrary streams.
The research instead studies task-conditioned lossy memory under a declared
future-query distribution. Lower bounds from continual learning define the
impossibility boundary; rate-distortion and utility define which errors a
bounded system should accept.

### 7.6 Parametric interference hypothesis

Recent teacher-student analysis reports a \(1/d\) forgetting relation under
orthogonal output heads, using a gradient-product proxy. This is a restricted
result, not a universal LLM scaling law. It will be treated as a candidate
mechanism and tested against task similarity, adapter rank, model width, and
update depth. See [Scaling Law for Catastrophic Forgetting via Gradient
Products](https://openreview.net/forum?id=68TggRP3Bb).

### 7.7 Dimensionless load and regime boundaries

The study will test a set of dimensionless loads rather than force all behavior
into one smooth power law:

\[
\rho_{\text{sleep}} =
\frac{r_e\bar c_s}{G_{\text{sleep}}},
\quad
\rho_{\text{store}} =
\frac{r_e\bar b\,\tau_{\text{retain}}}{B_{\text{store}}},
\quad
\rho_{\text{read}} =
\frac{b_{\text{relevant,tok}}}{B_{\text{live,tok}}},
\]

\[
\rho_{\text{repr}} =
\frac{N_{\text{assoc}}}{\kappa_{j,\mathcal T,\epsilon}},
\quad
\rho_{\text{hot}} =
\frac{U s_{\text{user,byte}}}{B_{\text{HBM,byte}}},
\quad
\rho_{\text{gov}} =
\frac{r_{\text{delete}}\bar u_{\text{ops/delete}}}
{G_{\text{gov,ops/s}}}.
\]

\(\kappa_{j,\mathcal T,\epsilon}\) is defined separately for memory medium
\(j\), task/criterion \(\mathcal T\), and tolerated distortion \(\epsilon\);
association count is never treated as a universal capacity unit across text,
graphs, latent state, and parameters.

These are load indicators, not proved phase transitions. Candidate regime
changes to test near a binding load of one are:

- \(\rho_{\text{read}}<1\) with low reuse favors exact retention;
- \(\rho_{\text{read}}>1\) and \(\rho_{\text{sleep}}<1\) favors merge or
  abstraction;
- high discounted reuse and low invalidation hazard can justify latent or
  parametric compilation;
- \(\rho_{\text{repr}}>1\) requires expansion, sparsity, eviction, or accepted
  interference;
- \(\rho_{\text{sleep}}>1\) makes the consolidation queue unstable;
- \(\rho_{\text{hot}}>1\) forces adapter/fast-state offload or
  externalization;
- \(\rho_{\text{gov}}>1\) can make parametric promotion operationally
  inadmissible.

Fits will permit segmented or phase-transition behavior. A relation will be
called a scaling law only if each fitted scale axis has at least eight points,
at least two held-out scales are predicted, candidate power, bottleneck, and
segmented models are compared with nested cross-validation, all resources are
normalized to stated reference units, and the regime interpretation survives
across workloads. Otherwise it is an empirical curve.

A capacity knee may be fit as:

\[
A(N_{\text{assoc}})=A_0-\Delta
\sigma\left(
\frac{\log N_{\text{assoc}}-\log K_{\text{eff}}}{w}
\right),
\]

\[
\frac{K_{\text{eff}}}{K_0}\propto
\left(\frac{M_{\text{store}}}{M_0}\right)^\alpha
\left(\frac{B_{\text{live,tok}}}{B_0}\right)^\beta
\left(\frac{F_{\text{sleep,tot}}}{F_0}\right)^\gamma
\left(\frac{M_{\text{latent}}}{L_0}\right)^\nu.
\]

The multiplicative expression for \(K_{\text{eff}}\) is only a candidate
empirical surface. A min-bottleneck model, normalized Cobb-Douglas model,
generalized mean, and interaction model are preregistered competitors.
Prediction-error tolerance is fixed from pilot variance, measurement
calibration, and the target decision tolerance before confirmatory data;
exponent reversal or prediction-interval failure on workload and hardware
holdouts blocks a scaling-law claim.

Bounded capacity also requires byte-flow equilibrium for every store \(j\):

\[
\frac{dB_j}{dt}=
r_{\text{admit},j}\bar b_{\text{in},j}
-r_{\text{expire},j}\bar b_{\text{delete},j}
-r_{\text{compact},j}\bar b_{\text{reclaimed},j}.
\]

An archive with no expiry, deletion, or physical reclamation is unbounded even
if its active index is compact.

The notation is fixed as follows:

| Symbol | Unit | Meaning |
|---|---|---|
| \(F_{\text{sleep,job}}\) | FLOP | compute consumed by one sleep job |
| \(G_{\text{sleep}}\) | FLOP/s | effective sleep-cluster service rate |
| \(F_{\text{sleep,tot}}\) | FLOP | total sleep compute over the declared horizon |
| \(B_{\text{live,tok}}\) | token | answer-time evidence budget |
| \(B_{\text{live,byte}}\) | byte | live state residency budget |
| \(B_{\text{active}}\) | byte | active memory artifacts |
| \(B_{\text{durable}}\) | byte | raw/deletable durable payload |
| \(B_{\text{index}}\) | byte | indices and metadata |
| \(B_{\text{lineage}}\) | byte | provenance and invalidation metadata |
| \(c_{n,r}\) | native resource unit | resource-\(r\) consumption at epoch \(n\) |
| \(T_n\) | s | physical time of decision epoch \(n\) |
| \(\rho\) | s\(^{-1}\) | continuous-time discount rate |
| \(v_D\) | currency/loss unit | frozen value of one task-loss unit |
| \(p_r\) | currency/resource unit | frozen resource price |
| \(C_{\cdot},K_{\cdot}\) | declared cost unit | fixed or present-value amortization terms |

### 7.8 Original hypothesis set

All items in this subsection are hypotheses; G2 freezes each record and digest.

1. **H-STC-001 — Future-relevant rate-distortion.** The best memory code minimizes loss on
   future tasks, not reconstruction error of the original transcript. Changing
   the query distribution changes the optimal representation of the same event.
2. **H-STC-002 — Reuse-volatility promotion cone.** General discounted reusable hits are
   \(\int_0^H\lambda_i(t)S_i(t)e^{-\omega t}dt\). Under stationary Poisson hits,
   exponential invalidation, and an infinite horizon, the special case is
   \(\lambda_i/(\mu_i+\omega)\). Piecewise external, latent, and parametric
   regions are predicted only if destination cost/gain curves satisfy monotone
   single-crossing conditions.
3. **H-STC-003 — Conditional promotion-demotion hysteresis.** A two-state switching model
   with promotion and demotion costs predicts separated thresholds under
   derivable sufficient conditions; threshold ordering is not asserted
   universally. A single threshold is tested for tier thrashing.
4. **H-STC-004 — Multi-capacity bottleneck.** Usable memory is limited by the currently
   binding storage, application bandwidth, representation, sleep service, hot
   residency, or governance capacity. Increasing only one resource should
   eventually stop improving utility.
5. **H-STC-005 — Reversibility-reserve optimum.** Archive depth is predicted to have a
   non-monotonic optimum: rollback benefit first permits more aggressive
   abstraction, while durable bytes, attack surface, and deletion workload
   eventually dominate.

## 8. Training design

### 8.1 Sleep corpus

Every training item retains:

- source event identifiers and timestamps;
- raw observation or reversible pointer;
- current query/action context;
- later reuse and outcome signals available only after the event;
- contradiction, supersession, and deletion markers;
- teacher-generated abstractions with confidence and verifier results;
- negative examples, including similar but obsolete or wrong memories.

Future information can label an offline oracle but cannot leak into the online
policy at evaluation time. The implementation physically separates:

```text
observable_event_schema:
  event_id, user_id, source_id, type, payload, timestamp,
  confidence_at_write, privacy_scope, provenance, supersedes

oracle_annotation_schema:
  future_reuse, valid_to, later_outcomes, poisoned,
  deleted_at, ground_truth_dependencies, hindsight_action_utility
```

The deployable policy receives only a versioned feature whitelist derived from
past observations, such as past hit rate, age, source type, uncertainty, and
current occupancy. It estimates \(\hat\lambda\) and \(\hat\mu\); it never
receives the generator's true reuse parameter or future-query schema. Unit tests
fail if an oracle column, future timestamp, future-derived embedding, or label
enters the policy input. The oracle router is explicitly clairvoyant and only a
non-deployable upper bound.

All identifiers in research and production schemas are opaque, randomly
generated handles rather than email addresses, account names, raw URIs, or
content-derived hashes. Re-identification maps and source locators live only in
the separately encrypted, deletable payload sidecar described in Section 10.4.

### 8.2 Compiler objectives

1. **Faithful compression:** distill facts, procedures, preferences, and
   relations while preserving source linkage.
2. **Retrieval discrimination:** contrast relevant records against hard
   negatives, stale versions, and semantic near-duplicates.
3. **Parametric consolidation:** rehearse retained examples, constrain
   interference, and isolate user-local updates with adapters or sparse modules.
4. **Dream augmentation:** generate counterfactual combinations, then filter
   them with provenance-aware validators and measure contamination separately.
5. **Policy learning:** imitate a hindsight utility oracle, then adapt using a
   contextual bandit or constrained RL objective over delayed lifetime reward.

### 8.3 Scheduler action space

For each candidate cluster:

- keep raw;
- summarize or extract facts;
- add/update/delete external semantic memory;
- construct or update graph relations;
- create latent cache;
- update user adapter;
- propose promotion to shared memory;
- demote to archival storage;
- forget while retaining a deletion tombstone.

## 9. Benchmark and experiment design

### 9.1 Lifetime stream

The unit hierarchy is:

```text
user → cycle → session → event → query
```

Each cycle is:

```text
wake capture → pre-sleep probe → sleep
→ post-sleep probe → next-wake delayed probe
```

The scoring ledger joins the separate observable and oracle schemas:

```text
event_id, user_id, source_id, type, payload, valid_from, valid_to,
confidence, privacy_scope, expected_reuse, provenance, supersedes,
poisoned, deleted_at, ground_truth_dependencies
```

The controlled benchmark uses 128 cycles by default, with
\(C\in\{8,32,128,512\}\) and retrieval lags of
\(\{1,8,32,128\}\) cycles. It independently varies:

- event arrival rate and idle slack;
- future reuse frequency and delay;
- query entropy and paraphrase;
- stable facts, volatile facts, skills, procedures, and relations;
- contradictions and supersession;
- adversarial or poisoned memories;
- deletion and unlearning requests;
- single-user and cross-user reuse;
- storage, compute, freshness, and latency budgets.

The controlled event families include exact facts, drifting preferences,
reusable procedures, negative feedback, correction and contradiction, privacy
deletion, poison, and one-off distractors. Reuse count, live-memory pressure,
volatility, and poison rate must be independently controllable.

Before G3, the benchmark specification freezes events per cycle, session-length
and payload distributions, query-generation grammar, paraphrase difficulty,
conflict-resolution state machine, deletion ground truth, random-number
hierarchy, and deterministic exact/temporal scorer. Generator property tests
check conservation, no future leakage, valid supersession, deletion precedence,
and replay determinism. Exact-oracle and executable outcomes are confirmatory;
model-based semantic judging and a human-audited subset are secondary validity
checks.

Existing long-memory suites provide external-validity tracks:

- MemoryAgentBench for retrieval, test-time learning, long-range understanding,
  and selective forgetting;
- LongMemEval for temporal updates, abstention, and delayed recall;
- HaluMem for writer-to-retriever-to-reader error propagation;
- ScienceWorld to ALFWorld or WebArena procedural transfer as a stretch track.

The project needs the controlled generator in addition to these suites so
crossover laws can be estimated rather than systems only being ranked at one
operating point. Future queries, corrections, and deletion events are hidden
from the consolidator. Policy-training users and evaluation users are disjoint,
and the same judge is not used both to write and to grade a memory.

Each public benchmark uses a versioned deterministic adapter that preserves
original session order, declares its sleep-boundary and query-withholding rules,
and publishes a transformation hash. These tracks provide external validation;
they do not fit the controlled scaling laws.

### 9.2 Baselines

The confirmatory architecture-compatible paper track is limited to:

1. raw versioned external retrieval;
2. compressed external memory;
3. one reusable latent/KV compiler compatible with the frozen base model,
   initially a Cartridges-style artifact;
4. one user-local parametric compiler, initially LoRA;
5. a tuned deterministic static mixture over the first four destinations and
   deliberate forgetting;
6. the deployable hybrid router over those actions;
7. a clairvoyant oracle router used only to estimate heterogeneity value.

The static mixture is tuned only on validation users. For each resource/workload
cell it freezes an action-proportion vector and cadence, then maps an opaque
event ID to an action by a published deterministic hash schedule. It receives no
event content, inferred reuse, future label, or test-user feedback. This controls
for the value of merely allocating some traffic to every tier; the learned
router must add event-conditional value beyond that larger action set.

No-memory and full-history are non-budgeted lower/upper diagnostics and are
excluded from matched dominance tests. A second latent/KV implementation, graph
memory, sparse memory layer, architecture-native recurrent fast weight, and
vendor/product-style Mem0, Zep/Graphiti, or Letta implementation belong to
adjacent characterization or a preregistered extension; they cannot expand the
first paper's main comparison after results are seen. The one frozen
Cartridges-style latent baseline is the only latent implementation in the
confirmatory family.

Methods operate under the same **upper-bound resource vector**, not forced
equality:

```text
live token cap, active-byte cap, durable-byte cap, sleep-FLOP cap,
peak accelerator cap, information cutoff, privacy/deletion contract
```

Unused budget is reported. Bytes moved, latency, throughput, and energy are
outcomes rather than matched inputs. Each operating point has a feasibility
mask; a method requiring sleep cannot run in the zero-sleep cell. Search,
controller training/inference, auxiliary model calls, embeddings, curation,
index rebuilds, optimizer/checkpoint state, raw archive, verification, and
recovery all count.

Every eligible confirmatory baseline freezes repository SHA, patch, container,
base checkpoint and revision, tokenizer, compiler and embedding model, index,
top-k, chunking, prompt, optimizer, trainable parameter count, training tokens,
and hyperparameter search budget at G2. Proprietary systems remain directional
characterization because their information and cost boundaries cannot be
matched.

### 9.3 Confirmatory matrix and exploratory ablations

The confirmatory reuse variable \(R\in\{1,4,16\}\) is the
generator-controlled mean count of valid future queries per admitted event
before supersession, deletion, or horizon end. Active-capacity pressure is
defined through:

\[
\phi =
\frac{B_{\text{active,cap}}}{B_{\text{ref}}},
\qquad
P_{\text{active}}=\phi^{-1},
\qquad
\phi\in\{1,\tfrac14,\tfrac1{16}\}.
\]

\(B_{\text{ref}}\) is frozen at G2 as the 95th percentile, over blinded
pilot/training streams, of the peak bytes needed by a raw full-valid-history
store during the 128-cycle horizon. Held-out streams reuse that absolute
per-user cap and never recompute it from test data. \(B_{\text{active}}\)
includes every answer-reachable payload, embedding, index, destination-specific
metadata, adapter delta, and wake-required state; the shared frozen base model
is reported separately, and durable archival payload is charged to
\(B_{\text{durable}}\). Thus \(\phi=1\) is the low-pressure reference and
\(\phi=1/16\) is the highest-pressure cell; infeasible methods remain visible
through the preregistered feasibility mask.

The routing/medium confirmatory family for PAPER-C1 through PAPER-C3 is:

```text
6 deployable methods × 3 reuse regimes R={1,4,16}
× 3 active-capacity ratios phi={1,1/4,1/16}
× 3 paired stream-seed families
= 162 lifetime runs

oracle router on the same 27 stream/operating cells
= 27 upper-bound runs

routing/medium subtotal = 189 lifetime runs
```

PAPER-C4 has its own powered factorial rather than an uncounted ablation:

```text
2 schedules {inline-immediate, deferred-at-cycle-boundary}
× 2 operators {identity, consolidating}
× 3 reuse regimes × 3 active-capacity ratios
× 3 paired stream-seed families
= 108 lifetime runs

logical confirmatory total = 189 + 108 = 297 lifetime runs
```

The PAPER-C4 destination is frozen to versioned external memory. The identity
operator validates and republishes a byte-equivalent record without semantic
compression; the consolidating operator uses one frozen compressor to
canonicalize and merge the same declared source set. The schedule changes only
execution time: both cells receive the same enqueue-time information snapshot,
resource cap, deadline, and destination. This makes the schedule main effect,
operator main effect, and schedule-by-operator interaction identifiable without
confounding deferral with extra future information. Information accumulation
during sleep is a separately labeled exploratory factor.

Each family contains a user count fixed by blinded paired power simulation
before G2; the final count is the maximum required for the PAPER-C2 paired
contrast, the PAPER-C3 interaction/crossover, and the PAPER-C4 factorial
contrasts. All methods see identical user streams. A physical execution whose
complete configuration hash is identical across families may be referenced by
both estimands, but the 297-cell logical design, contrasts, and multiplicity
family remain fixed and fully enumerated. The no-reuse \(R=0\) cell is a
separate break-even diagnostic, not part of amortized inference.
Full-history/no-memory diagnostics are also outside the matched total.

The dense scaling track varies one axis at a time with at least eight points on
at most two methods and uses at least two held-out scales. Other axes—512/2K/8K
live tokens, sleep-to-wake FLOPs, concurrency, volatility, trigger family,
dream/replay order, graph/shared weights, and cluster placement—use a
fractional-factorial or successive-halving exploratory design capped at no more
than the compute-equivalent of another 135 lifetime runs. Pilot, confirmatory,
and exploratory budgets are separate; unused capacity does not authorize
post-hoc runs.

Required causal extensions use the same events and frozen contrasts:

- continuous asynchronous scheduling as an exploratory third level beside the
  powered inline/deferred factorial, with total GPU-hours, peak GPUs, batch
  ceiling, deadline, and information cutoff declared;
- same information with raw, compressed external, and user-LoRA destinations;
- fixed cadence, tuned deterministic mixture, learned trigger, and oracle;
- sequential per-user LoRA, adapter-grouped batches, per-stream replicas, and
  heterogeneous batched-LoRA serving;
- exact replay, distillation, and dream augmentation in the extension track;
- with/without provenance, verification, rollback, and deletion enforcement.

### 9.4 Metrics

For each preregistered resource tuple, the single primary endpoint is
**lifetime utility AUC**. The primary comparison is the paired delta between the
deployable hybrid and the stronger of the tuned single-medium envelope and
static mixture on held-out users and workload families. Secondary frontier
hypervolume uses a reference point frozen at G2 and includes policy
training/tuning cost. PAPER-C2's positive benefit branch succeeds if either one
of these preregistered alternatives holds:

- hybrid utility improvement of at least two absolute percentage points with a
  user/stream hierarchical-bootstrap 95% lower bound above zero;
- no worse than one percentage point in utility with at least 10% lower
  preregistered lifecycle cost and a lower confidence bound above zero.

The alternatives are disjunctive, not cumulative. If neither holds,
PAPER-C2 is FALSIFIED/NARROWED and the null-paper branch reports the regret or
inadmissible region; secondary hypervolume cannot overturn that decision.
PAPER-C4 separately tests the two schedule simple effects, the two operator
simple effects, and their interaction under its frozen multiplicity correction.
Its positive branch is `SUPPORTED` only if all three conditions hold:

1. consolidating versus identity produces at least a two-point future-query
   utility gain with the multiplicity-adjusted 95% lower bound above zero in at
   least one preregistered reuse-pressure regime;
2. for the identity operator, the cell-balanced marginal deferred-minus-inline
   utility contrast passes two one-sided equivalence tests at familywise
   \(\alpha=0.05\): its multiplicity-adjusted 90% confidence interval lies
   wholly inside \([-1,+1]\) percentage point. Deferred execution must also
   reduce cell-balanced marginal lifecycle cost by at least 10%, with the
   multiplicity-adjusted 95% lower confidence bound exceeding 10%;
3. the cell-balanced marginal schedule-by-operator utility interaction passes
   the same familywise-\(\alpha=0.05\) equivalence procedure, with its adjusted
   90% confidence interval wholly inside \([-1,+1]\) percentage point. This
   supports only a marginal first-order separability claim; every cellwise
   interaction and its interval is still reported.

If no multiplicity-adjusted operator main or conditional simple effect survives,
PAPER-C4 is `FALSIFIED`. Any other outcome that misses one or more support
conditions is `NARROWED`: a surviving sub-two-point effect is labeled “detectable
but below the materiality threshold”; failure of condition 2 is “consolidation
without a demonstrated deferred-cost benefit”; and failure of condition 3 is
“schedule-contingent or interaction-inconclusive consolidation.” An interval
that overlaps the equivalence margin and the null is
`NARROWED—insufficient precision`, not evidence of equivalence. A significant
interaction is a result, but it cannot be relabeled as support for the
separability hypothesis.

The exact effect thresholds are recalibrated only from blinded pilot variance
and deployment tolerance before G2; any later change makes the analysis
exploratory.

#### Learning and memory

- current-task utility;
- retained utility and forgetting;
- forward transfer;
- memory half-life and retention survival by lag;
- exact, semantic, temporal, relational, and procedural recall;
- calibration and false-memory rate;
- writer, retriever, and reader error decomposition;
- consolidation coverage, preservation, faithfulness, and detail loss;
- stale/conflicting hit rate and poison amplification;
- access deletion;
- physical artifact removal;
- derived-lineage invalidation;
- behavioral residual-influence removal;
- rollback success and deletion-propagation latency;
- interference across users and tasks.

#### Systems

- p50/p95/p99 wake latency and time to first token;
- accepted requests/s, aggregate tokens/s, and SLO violation rate;
- retrieval, queue, prefill, decode, and writeback latency decomposition;
- effective-batch distribution, MFU, achieved/roofline memory bandwidth;
- cache and adapter hit/miss, HBM residency, and heterogeneous-LoRA batching;
- sleep job latency, throughput, and deadline miss rate;
- construction/retrieval/generation energy;
- tokens and FLOPs by phase;
- HBM, DRAM, disk, and network bytes moved;
- resident bytes and memory amplification;
- freshness/staleness and queue backlog;
- adapter load/swap and version-publish cost;
- lifetime cost and energy per correct or useful answer.

Energy boundaries are reported separately as GPU-board measurement, host
measurement, and modeled storage/network contribution. Runs freeze clocks or
power caps, warm-up, idle subtraction, and minimum measurement duration.
HBM-component energy remains analytical unless directly measurable. Accuracy
and energy Pareto results are primary; energy-per-correct is secondary and only
reported above a preregistered accuracy floor.

Inference uses paired method runs on identical streams. User and stream seed,
not cycle, are the top-level independent units. G2 freezes one primary
comparison family, a power/sensitivity simulation, a hierarchical bootstrap or
mixed-effects model, multiplicity correction, failed-run handling, and a fixed
run count with no outcome-driven optional stopping.

All storage-reduction claims report active, durable, index, latent,
optimizer/checkpoint, and lineage bytes separately. Reducing the active bank
while retaining a raw archive is not reported as reducing total storage.

### 9.5 Three evidence tracks

1. **Local reproducibility:** small open models and a synthetic lifetime stream;
   verifies correctness, determinism, and qualitative effects.
2. **Analytical and trace-driven DSE:** sweeps arrival, reuse, capacity, cluster,
   and data-movement parameters beyond available hardware.
3. **Accelerator runbook:** a pinned A100 80 GB reproduction for energy,
   utilization, prefill/training contention, and tail latency.

No simulated or analytical result will be presented as measured hardware data.
The accelerator sequence starts with KV reads, fast-state read-modify-write,
delayed writes, adapter load/apply/update, checkpoint, and network/storage
microbenchmarks. Small semantic and rollback gates precede any 760M/30B
reproduction. Absolute latency and energy claims require target-accelerator
measurement. Debug and confirmatory checkpoints, revisions, dtype, clock/power
configuration, and serving stack are frozen in the runbook. H100 remains a
separate future measurement or DSE point and is never conflated with A100 data.

## 10. System architecture

### 10.1 Planes

#### Wake plane

- latency-sensitive inference and retrieval;
- active context and optional lightweight test-time state;
- append-only event logging;
- immutable read view of the currently published memory version.

#### Sleep plane

- throughput-oriented prefill, embedding, distillation, replay, and tuning;
- batch construction, compaction, conflict resolution, and graph maintenance;
- preemptible jobs with deadlines derived from freshness SLOs;
- validation before state publication.

#### Memory fabric

- content-minimized, append-only operational metadata with explicit expiry,
  a non-identifying deletion ledger, and separately deletable
  tenant-encrypted payload/identity sidecars;
- text/vector/graph/latent stores;
- adapter and checkpoint registry;
- procedural lineage from every compiled item to sources and transformations,
  distinguished from unproven semantic/effect attribution inside parameters;
- versioned snapshots, deletion tombstones, rollback, and audit records.

#### Policy and control plane

- utility estimation and admission;
- cadence, resource, and deadline scheduling;
- tier promotion/demotion;
- per-user isolation and shared-memory promotion review;
- queue stability and fleet-level capacity control.

### 10.2 Data movement

The architecture should move computation toward user-sharded memory where
possible. It should publish compact deltas, indices, or adapters instead of
retransmitting full histories. Serving pins a version while sleep builds the
next one; publication is atomic after validation. Personal-data MVP promotion
stops at an external artifact or user-isolated adapter. Shared-weight promotion
and selective shared-weight unlearning remain stretch research.

### 10.3 Queue condition

For normalized-server sleep job classes \(k\), the single-resource diagnostic
condition is:

\[
\sum_k \lambda_k \mathbb{E}[S_k] < m,
\]

where \(S_k\) is server-seconds/job and \(m\) is effective parallel servers.
The actual model uses a resource-demand vector:

\[
\sum_k\lambda_k\mathbb{E}[c_{k,r}]<C_r
\quad\text{for each }r\in
\{\text{GPU, CPU, storage I/O, network}\}.
\]

Trace simulation includes setup plus per-event service, bulk batching,
priority/preemption, deadlines, and tenant affinity. Stability alone is
insufficient; p99 queue age and deadline misses are publication gates. When a
condition fails, the policy delays, compresses, chooses a cheaper operator, or
drops low-utility work rather than allowing unbounded staleness.

### 10.4 Versioning and failure contract

Every derived artifact records only opaque tenant-scoped artifact and source
handles, base-model and tokenizer hashes, snapshot version, parent handles,
checksum, compiler version, privacy scope, and deletion epoch. A
content-addressed tenant manifest atomically names the ready versions of every
vector, graph, latent, and adapter artifact.

“Append-only” means tamper-evident while a record is inside its declared
retention window; it does not mean that identifying metadata is immortal. The
deletion contract is:

- account identifiers, raw source locators, payloads, free text, embeddings,
  reversible content hashes, and the map from an opaque handle to a person live
  only in a tenant-encrypted deletable sidecar;
- operational manifests contain random non-derivable handles and the minimum
  state needed for concurrency and authorization; they never contain direct
  identifiers or content fingerprints;
- deletion first places an immediate deny record at the authorization boundary,
  then removes payloads, mappings, indices, caches, optimizer state, candidates,
  backups after their frozen maximum retention, and every reachable derived
  artifact; affected adapters are rebuilt or retired;
- the final tombstone contains only a random deletion-transaction handle,
  coarse time bucket, policy/result code, artifact-class counts, and proof
  digest. It contains no tenant, user, source, content-derived identifier, or
  reusable cross-system join key;
- ledger retention, backup expiry, key-destruction latency, and legal-hold
  exceptions are frozen for the deployment profile at G4. Legal hold is
  reported as a separate non-deleted state rather than silently counted as
  success;
- completion is reported independently for access denial, online physical
  removal, backup/key expiry, lineage invalidation, and behavioral residual.

Deletion audit replays the lineage traversal, inventories every storage class,
tests that removed handles cannot authorize a read, scans remaining metadata for
direct identifiers and content fingerprints, and performs a linkability attack
with every key the system still retains. A physical-removal claim fails if any
undeclared replica or identifiable sidecar remains; an unlearning claim fails
if behavioral residual exceeds its preregistered threshold. Aggregate,
non-identifying operational evidence may remain, but it cannot substitute for
artifact removal.

The minimum failure protocol is:

- a wake request pins one memory version;
- sleep reads an immutable snapshot and produces a shadow artifact;
- verification and canary probes run before publication;
- per-artifact readiness barriers complete before manifest-last publication;
- a compare-and-swap checks both parent generation and deletion watermark and
  then publishes one manifest pointer atomically;
- worker retries are idempotent and uncommitted candidates are disposable;
- partial storage writes use temporary objects, synchronization, checksum, and
  manifest-last publication;
- a stale snapshot causes abort or explicit rebase, never a blind merge;
- poison or canary regression quarantines the artifact and rolls the pointer
  back;
- a deny path for deletion overrides all pinned old versions immediately;
- deletion quarantines in-flight descendants, cryptographically erases or
  expires tenant payloads under declared backup/retention rules, and uses the
  lineage graph to invalidate or rebuild summaries, latent state, and adapters;
- a base-model upgrade invalidates incompatible latent or parametric artifacts
  and recompiles them from provenance;
- under memory or queue pressure, wake traffic takes priority and the fallback
  is selected by contract from a last-known-good authorized version,
  quarantined provenance-only retrieval, or no-memory/abstention; poisoned,
  stale, or deleted raw content is never a default fallback.

Fault injection kills workers, truncates objects, forces stale compare-and-swap,
corrupts indices, drops a node, injects poison, and deletes during a pinned read.
G4 freezes and tests RTO, RPO, duplicate/lost-event tolerance, deletion
precedence, and maximum stale-read bounds.

## 11. Claim contract

Every material claim is assigned one class:

- **SOURCE-SUMMARY:** bounded statement of a paper's own method or result;
- **VENDOR-BEHAVIOR:** existence, API, or documented product behavior, not an
  effectiveness generalization;
- **RERUN:** same implementation and data executed again;
- **INDEPENDENT-REPLICATION:** result regenerated through an independent
  environment or implementation path;
- **ORIGINAL-MEASUREMENT:** project result with immutable raw logs;
- **ANALYTIC-DERIVATION:** checked mathematical consequence of stated
  assumptions;
- **TRACE-SIMULATION:** analytical or simulated systems result, never a hardware
  measurement;
- **SYNTHESIS:** interpretation supported by multiple sources;
- **HYPOTHESIS:** novel, falsifiable, and not presented as established.

Each claim record includes source, exact support span, evidence tier, assumptions,
scope, counterevidence, status, artifact dependency, and last audit date.

Three orthogonal fields prevent a single confidence score from hiding different
weaknesses:

- **source grade:** A peer-reviewed primary; B research preprint with method;
  C official code or product documentation; D proposal, vendor claim, or blog;
  X retracted, contradicted, or unverifiable;
- **warrant:** MEAS, REPL, DERIV, SUMM, SYNTH, or PROP;
- **reproduction strength:** r0 none through r3 independent environment or
  second implementation path.

At G0, paper claims are frozen as `PAPER-C1` through `PAPER-C4`. At release,
each must be `SUPPORTED` or `FALSIFIED/NARROWED`; `HYPOTHESIS` and `UNRESOLVED`
cannot appear as an abstract or conclusion contribution. A failed central
hypothesis follows a preregistered null-paper branch and supports only the
negative or boundary claim actually tested.

Headline quantitative claims require an ORIGINAL-MEASUREMENT or
INDEPENDENT-REPLICATION at r2 or above. RERUN alone does not establish external
validity, product documentation supports only product behavior, and
TRACE-SIMULATION cannot support absolute hardware latency or energy.

The required trace is:

```text
source version/hash → exact anchor → question/hypothesis
→ data/config/code/environment → raw-result checksum
→ derived result block → claim → figure/table → manuscript sentence
```

Cross-paper numbers remain directional or ordinal unless their environments and
accounting boundaries can be harmonized.

## 12. Artifact graph

### 12.1 Research control artifacts

1. chronological source registry;
2. one paper card per primary work;
3. evidence ledger and claim registry;
4. open-question, hypothesis, and falsifier registry;
5. terminology and taxonomy registry;
6. contradiction and negative-results log.

`qa-memory` remains an append-only discovery inbox. An item is promoted to the
hypothesis registry only after it has a unique ID, evidence anchor, estimand,
null/rejection rule, validity-failure rule, owner, and artifact dependency.

### 12.2 Scientific artifacts

1. English paper manuscript;
2. Korean NM-grade companion monograph;
3. formal theory and scaling-law note;
4. controlled lifetime benchmark and dataset generator;
5. baseline and scheduler implementation;
6. trace-driven simulator and cost model;
7. local and accelerator experiment manifests;
8. raw result bundles, statistical analysis, and figure-ready tables;
9. wake/sleep/memory infrastructure blueprint;
10. reproducibility and limitations report.

Every experiment bundle includes config, seed, git SHA, container digest,
environment fingerprint, benchmark-generator/data digest, split manifest, base
checkpoint/tokenizer revision, upstream baseline commit/patch, exact command,
analysis/render digest, immutable stdout and raw metrics, derived result blocks,
and checksums. Missing required input automatically blocks G5.

The artifact list is backed by a machine-readable DAG. Every node records:

```text
artifact_id, schema_version, input_digests, producer_command,
output_digest, consumers, gate_status, owner
```

Changing an upstream digest automatically marks downstream results, claims,
figures, tables, equations, manuscripts, audits, release tags, and presentation
handoffs `STALE`. G7 audits only a frozen candidate tag.

Every final figure and table includes question/claim IDs, input result blocks,
generator command, caption, and reuse owner. Every analytic equation links to a
derivation and unit/property-test digest.

### 12.3 Communication artifacts

1. executive research memo;
2. publication-quality figure suite;
3. comparison tables and paper timeline;
4. presentation-handoff manifest for a separate PPTX repository/branch and
   review process;
5. optional static project page or executable appendix.

The polished PPTX is a non-blocking consumer with its own design/tooling plan and
definition of done. It consumes only versioned claims, tables, equations, and
figures exported from the research artifacts and does not become a second,
manually diverging source of truth.

Its read-only handoff manifest contains, for each slide candidate, a claim ID,
one-sentence message, figure/data asset, short and long caveats, citation, alt
text, and release tag. PPTX tooling and visual-language research can proceed in
parallel; factual slide production starts from a frozen evidence tag.

## 13. Publication strategy

### Primary paper

Working title:

> **When Should Models Sleep? Budgeted, Provenance-Linked Memory
> Compilation Across External, Latent, and Parametric Tiers**

The paper contribution must contain:

1. a constrained external-versus-latent-versus-user-parametric compilation
   formulation;
2. a 128-cycle exact-oracle lifetime benchmark, resource-cap protocol, and
   powered schedule/operator factorial;
3. a past-only deployable destination router with oracle-regret analysis;
4. held-out reuse, pressure, and capacity crossovers plus the preregistered
   positive or null/narrowed factorial result.

The first paper owns RQ1, RQ3, and the controlled part of RQ7/RQ8. RQ4 learned
cadence/trigger comparison, RQ5/RQ6 training variants, RQ9 cluster architecture,
RQ11 dream staging, RQ12 shared weights, graph and additional-latent expansion,
and full safety infrastructure remain companion or extension material unless a
separate paper is preregistered before its data is inspected. Additional latent
implementations remain extension work; the single frozen latent baseline above
is part of the first paper.

The provisional pre-data claim spine is:

- **PAPER-C1:** strict oracle heterogeneity value exists in at least one
  preregistered reuse-pressure regime;
- **PAPER-C2:** a past-only deployable router captures useful oracle value on
  held-out users/workload families, or the null branch quantifies its regret;
- **PAPER-C3:** reuse and relative memory pressure predict a preregistered
  external-to-latent and/or latent-to-user-parametric destination crossover
  under full lifecycle accounting;
- **PAPER-C4:** over the preregistered reuse-pressure grid with equal cell
  weights and a matched enqueue-time information cutoff, the marginal semantic
  consolidation contrast yields a preregistered future-query utility gain,
  while the marginal deferral contrast for the identity operator preserves
  utility and reduces lifecycle cost; the cell-balanced marginal interaction
  remains inside the preregistered equivalence margin. This claim does not imply
  that every cell benefits or that heterogeneous cellwise interactions are
  absent.

These are hypotheses, not results. G0 may narrow wording before any
confirmatory run; after G2 only SUPPORTED or FALSIFIED/NARROWED outcomes are
allowed.

The default first-paper profile is learning and benchmark work for ICLR/NeurIPS.
A later systems paper may target MLSys only if it owns a distinct executable
runtime/scheduler claim, failure-injection study, and accelerator result set.
Before G2, a branch manifest assigns each claim, experiment, and figure to one
paper; headline results cannot be duplicated across simultaneous submissions.
ASPLOS remains stretch work requiring an actual runtime/compiler/hardware
contribution.

### Companion monograph

The Korean companion preserves the full chronology, biological foundations,
corporate research lineages, training taxonomy, negative results, derivations,
system design, and research diary that cannot fit the paper.

It consumes the same claim IDs. A parity map links every paper claim to its
Korean location and caveat; monograph-only assertions become new registered
claims and pass continuity, translation, and non-claim audits.

## 14. Research phases and gates

### Phase 0 — Foundation

- freeze definitions, claim schema, and source-quality tiers;
- ingest the verified pre-research dossier;
- identify direct, adjacent, and analogical literature.

Gate: every paper named in the framing has a verified identity and evidence
status.

### Phase 1 — Evidence atlas

- read primary sources chronologically;
- produce paper cards and cross-paper matrices;
- extract training data, objective, update location, cadence, capacity, systems
  assumptions, and failure modes.

Gate: central historical and technical claims are traceable to primary sources;
blog and product claims are separately labeled.

### Phase 2 — Theory and benchmark

- formalize the memory-compilation objective;
- derive candidate laws and regime maps;
- implement the lifetime generator and oracle;
- pre-register minimum experiments and falsifiers.

Gate: every proposed law maps to measurable variables and an out-of-sample test.

### Phase 3 — Algorithmic draft and local evidence

- reproduce selected baselines;
- run matched-budget local experiments;
- fit and invalidate candidate laws;
- produce the first complete algorithmic draft with honest null results.

Gate: no central conclusion relies only on an analogy or unverified third-party
number. This is not yet the minimum publishable gate.

### Phase 4 — Systems validation

- trace-driven sweeps;
- accelerator measurements;
- wake/sleep contention and separated-cluster studies;
- capacity, data movement, freshness, rollback, and deletion experiments.

Gate: reported system recommendations remain valid after full lifecycle and
transfer costs. The minimum publishable gate is evaluated only after Phase 4.

### Phase 5 — Synthesis and publication

- finalize the English paper;
- expand the Korean monograph;
- cross-audit claims, citations, figures, equations, and reproduction;
- export stable inputs for the separate PPTX track.

Gate: every PAPER-C claim is SUPPORTED or FALSIFIED/NARROWED by admissible
evidence; hypotheses and unresolved questions remain outside the contribution
list, and all known counterevidence and limitations are visible.

### Cross-phase review gates

- **G0 — spine lock:** one-sentence thesis, at most four main claims,
  non-goals, and primary venue profile;
- **G1 — evidence readiness:** high-risk sources carded with version, exact
  anchor, review status, code/data, and non-claim; search cutoff date,
  databases/queries, inclusion criteria, and citation-snowball stopping rule are
  frozen;
- **G2 — preregistration:** falsifier, baseline, matched budget, metric, effect
  threshold, seed, stopping rule, and confirmatory/exploratory label; hypotheses
  have unique `H-STC-*` IDs in a timestamped digest, and amendments
  automatically demote affected tests to exploratory;
- **G3 — benchmark validity:** construct, oracle, transition correctness,
  leakage, license, privacy, and split audit;
- **G4 — implementation correctness:** unit and property tests, baseline parity,
  infrastructure smoke test, checkpoint, rollback, and deletion;
- **G5 — result admissibility:** immutable raw logs, clean rerun, and an
  independent repeat or second calculation path for headline results;
- **G6 — claim gate:** zero blocked main claims, exact result-block references,
  and no omitted material caveat;
- **G7 — independent audit:** evidence, coverage, notation, statistics,
  reproduction, privacy/safety, and claim-to-figure review followed by
  adjudication and re-audit; the auditor did not author or run the artifact, or
  is a fresh-context reviewer, and sees only the frozen tag; critical/major
  severity and human-signed public waiver rules are fixed;
- **G8 — release QA:** clean builds, zero unresolved citation/xref, 100% figure
  provenance, visual QA, an immutable pre-seal release-candidate
  manifest/inventory, and an artifact smoke test against that exact inventory.

G8 does not depend on a release that can exist only after G8. Its non-null
evaluator attestation is a trusted release evaluator's signature over the
complete G8 subject—candidate, G7, replay, visual approval, pre-seal inventory,
smoke, rights, and measured systems evidence—not a reuse of visual approval.
After a signature-valid G8 `PASS`, deterministic sealing may add only the
canonical G8 record and seal metadata to the pre-seal inventory; it may not
change any declared payload, rights disposition, environment, or reproduction
input. A typed `ReleaseSealRecord` proves the draft-to-release payload identity
and binds the resulting release checksums.

The paper skeleton exists from G0 and G0 binds an immutable content-addressed
snapshot of those non-assertive bytes plus a build/trace/parity report generated
from the snapshot itself; the live manuscript may later fill its stable slots
without rewriting that historical snapshot. Background chapters of the
companion may grow after G1, but original-contribution chapters are not frozen
before G5.

G0 immutability is an executable property, not only a stored digest. The
canonical `publication scaffold verify --snapshot-root PATH --output PATH
--offline` command treats the snapshot root as read-only input, requires its
report path to be outside that root, and copies only manifest-declared source
bytes into a newly created temporary tree. English and Korean builds, their
trace checks, and parity validation execute only in that temporary tree with
repository/network fallback disabled. The snapshot inventory excludes
generated build trees, caches, and generated manuscript PDFs. A `PASS` requires
the manifest byte hash and the complete declared-source inventory hash to be
recomputed before and after verification and remain identical, with the copied
inventory matching them. Parser tests forbid an in-place or caller-selected
work-root mode; an adversarial-builder mutation test proves generated output
and source writes land only in the temporary copy, and a concurrent-mutation
test proves any original-snapshot drift fails verification.

PPTX design and tool experiments may run separately; external-facing slide
claims are assembled only from a G8-tagged handoff. Any internal prerelease deck
is visibly watermarked and cannot be exported.

## 15. Minimum publishable result and extension ladder

### Minimum

- exact-oracle controlled lifetime workload with 128 wake-sleep cycles;
- one deterministic public-benchmark replay for external validity;
- no-memory and full-history/oracle bounds;
- raw external, compressed external, one reusable latent/KV artifact,
  per-user LoRA, tuned static mixture, deployable hybrid, and oracle-router
  upper bound;
- one executable inline/deferred scheduler implementing the frozen RQ1
  protocol;
- common resource caps on live tokens, active and durable bytes, sleep FLOPs,
  information cutoff, privacy/deletion, and peak accelerators; bytes moved,
  latency, and energy are reported outcomes;
- reuse, cadence, and capacity sweeps;
- the 297-cell logical confirmatory design, power-determined users, paired
  stream families, and hierarchical uncertainty;
- local reproduction plus trace-driven systems analysis;
- target-accelerator measurement of the KV/read-modify-write, batching,
  adapter-load/update, DRAM-byte, and energy anchors used for absolute claims;
- either a supported deployable hybrid claim against the stronger of the tuned
  single-medium envelope and static mixture, or the preregistered null-paper
  branch that narrows the title and central thesis;
- one held-out crossover prediction; otherwise results are called regime curves,
  not scaling laws.

### Strong extension

- real multi-session agent traces;
- a second latent implementation and graph-memory tier;
- learned dream generation with contamination controls;
- H100-class lifecycle energy;
- multi-user adapter serving and deletion;
- public benchmark package and reference scheduler.

### Stretch

- end-to-end cluster prototype;
- cross-model and multimodal transfer;
- shared promotion with governance;
- a second systems-paper extraction.

## 16. Pre-registered claim-down criteria

The central claim is withdrawn or narrowed if any of the following survives the
planned controls:

- hybrid misses the §9.4 primary lifetime-utility/non-inferiority criterion
  against the stronger of the tuned single-medium envelope and static mixture
  on held-out users;
- even the oracle router cannot beat the best single medium;
- sleep gains disappear after future leakage removal and resource-cap,
  information-cutoff, and operator harmonization;
- PAPER-C4 misses its operator-utility, deferred-cost/noninferiority, or
  interaction-equivalence gate and therefore cannot support the cell-balanced
  marginal deferral/consolidation separation claim; no regime-universal claim
  is permitted even when the marginal gate passes;
- measured promotion break-even reuse exceeds the target confirmatory range
  \(R\in[1,16]\), or
  compilation has no positive wake-cost or utility denominator;
- no reproducible capacity knee exists, or fitted exponents reverse across
  scale, dataset, or hardware;
- parametric/expert growth makes active bytes and wake latency grow
  proportionally, defeating the proposed offline-growth separation;
- analytical bytes or latency fail the G2 measurement-calibrated prediction
  interval on held-out hardware points without an explained omitted term;
- energy improvement disappears under repeated confidence intervals and idle
  subtraction;
- the sleep queue violates its G2 utilization, deadline-miss, or p99-age
  contract in the target workload, or staleness erases the quality benefit;
- consolidation amplifies corrections, deletion failures, poison, or
  cross-user leakage relative to exact external memory;
- rollback or deletion cannot propagate to derived external and user-local
  parametric state.

Null results remain publication artifacts. They are not silently replaced by a
new confirmatory hypothesis on the same data.

## 17. Non-claims

The work will not claim:

- that biological NREM/REM maps one-to-one onto machine components;
- that all offline computation is sleep;
- that external memory is free of interference;
- that parametric memory is inherently more intelligent than retrieval;
- that a finite-memory learner can preserve arbitrary streams perfectly;
- that synthetic dreams are beneficial without contamination tests;
- that a product’s current behavior proves the method in its paper;
- that an analytical or simulated cost is a hardware measurement;
- that one benchmark ranking is a universal memory scaling law.

## 18. Definition of done

The research goal is complete only when:

1. the source registry and paper cards cover the declared literature scope;
2. every `PAPER-C*` claim is SUPPORTED or FALSIFIED/NARROWED; broader companion
   questions may remain explicitly unresolved but cannot be paper contributions;
3. original hypotheses have falsifiers and experimental status;
4. scaling laws include assumptions, uncertainty, regime limits, and held-out
   validation;
5. benchmark, code, configs, raw logs, and analysis reproduce the reported
   results;
6. the infrastructure blueprint is tied to measured or clearly labeled
   analytical bottlenecks; any systems-paper claim additionally has an
   executable trace-driven prototype, failure injection, p99/freshness, and
   rollback evidence;
7. all main claims are admissible and no blocked or verification placeholder
   remains;
8. every result table and figure regenerates from a clean checkout, every
   analytic equation passes derivation/unit checks, and the held-out scale tests
   are reported;
9. the English paper and Korean companion build cleanly and pass claim,
   citation, equation, figure, statistical, privacy/safety, visual, and
   reproducibility audits;
10. an independent audit, adjudication, and re-audit close with no critical or
   major issue open;
11. a release tag, checksum manifest, environment archive, and reproduction
   bundle exist;
12. the separate PPTX track receives a frozen, versioned evidence-and-figure
   handoff; deck production and visual QA are governed by its own non-blocking
   definition of done.
