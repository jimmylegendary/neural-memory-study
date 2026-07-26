# Sleep-Time Compute Scaling-Law and Theory Agenda

- Cutoff: 2026-07-25
- Status: candidate laws and preregistration design; no deployed STC scaling
  law is claimed
- Scope: reuse, cadence, queueing, byte flow, capacity knees, addressability,
  interference, plasticity, recoverability, governance, and infrastructure
  crossovers

## 1. Evidence boundary

The audited literature contains useful local results:

- bounded synapse models derive memory traces under explicit random-update and
  readout assumptions;
- continual-learning papers measure forgetting, transfer, replay, and
  plasticity over finite task streams;
- Letta sleep-time compute measures context/query amortization;
- HIMA measures external-memory budget/retention curves;
- PlugMem measures a finite graph-merge/soft-deactivation result and shows a
  retrieved-memory utility curve that can saturate then decline;
- MemMA demonstrates five-probe session-boundary verification/repair, but
  reports no probe-count, store-growth, or lifetime scaling sweep and does not
  release the paper-aligned probe-generation pipeline;
- H-EPM claims online successful-trajectory graph accumulation, with unresolved
  boundary ordering, and does not cap the number of state summaries per edge;
- MemEvolve, MemSkill, ALMA, M★, and RHO report finite-budget
  learned/searched memory-control artifacts; only MemSkill hard-caps one
  reusable bank, while generated stores, full executable harnesses, design
  archives, versions, and run records remain outside those per-search bounds;
- MemCompiler reports compiler and executor cost separately, making clear that
  executor-only improvement is not total-system improvement;
- Microsoft MEMENTO reports lower active KV together with a text-identical
  restart loss and recoverable opaque information in downstream KV; visible
  summary bytes and peak KV are therefore incomplete state and cost measures;
- Surprise-Gated Memory reports that novelty admission halves one episodic
  buffer without measured loss, while latest-five-task replay is worse than no
  replay and full interleaving preserves retention at growing lifetime cost;
- ReasoningBank shows retrieval quality can decline when more memories are
  injected;
- MEMORA measures merge/update frequency but continues to add many new
  memories;
- Google Sleep uses a finite preallocated expert pool;
- *Facts in Weights* measures retention collapse over 20–100 sequential writes.

None establishes one power law covering deployed wake experience, external
memory, latent state, parameter updates, governance, and sleep infrastructure.
Every relation below is therefore one of:

| Label | Meaning |
|---|---|
| `IDENTITY` | accounting or information identity under stated definitions |
| `NECESSARY` | necessary condition from queue/flow/information assumptions |
| `DERIVED-CANDIDATE` | optimum or threshold derived under a restricted model |
| `EMPIRICAL-CANDIDATE` | functional form to compare against alternatives |
| `HYPOTHESIS` | falsifiable program proposition |

## 2. State and unit inventory

### 2.1 Workload

| Symbol | Unit | Meaning |
|---|---|---|
| \(\lambda_e\) | event/s | admitted novel wake-event arrival |
| \(\lambda_s\) | job/s | sleep-job arrival after batching/admission |
| \(\lambda_i(t)\) | reuse/s | future query/use intensity for memory \(i\) |
| \(\mu_i\) | s\(^{-1}\) | invalidation/volatility hazard |
| \(H\) | s | physical-time evaluation horizon used in continuous integrals |
| \(K\) | cycle | wake--sleep cycle count |
| \(N_a\) | association | admitted retrievable associations |
| \(U\) | user | concurrently resident users |
| \(q\) | query | future evaluation queries |

### 2.2 Memory

| Symbol | Unit | Meaning |
|---|---|---|
| \(B_r\) | byte | retained raw/canonical evidence |
| \(B_e\) | byte | active external text/vector/graph |
| \(B_l\) | byte | all answer-reachable latent/KV state, including implicit source-conditioned channels |
| \(B_p\) | byte | adapters/experts and trainable local state |
| \(B_a\) | byte | auxiliary optimizer/importance/router/generator state |
| \(B_v\) | byte | version, lineage, and recovery state |
| \(b_{\text{mutable}}\) | bit | entropy/description-length bound for finite-precision incremental mutable state |
| \(T_{\text{live}}\) | token | answer-time visible context budget |
| \(B_{\text{live}}\) | byte | answer-time resident state, including hidden KV/cache state |
| \(P_{\text{active}}\) | parameter | active trainable/served parameters |
| \(P_{\text{total}}\) | parameter | active plus masked/inactive pool |

### 2.3 Compute and service

| Symbol | Unit | Meaning |
|---|---|---|
| \(F_s\) | FLOP/job | sleep compute per job |
| \(G_s\) | FLOP/s | effective sleep service |
| \(S_s\) | server-s/job | normalized sleep service time |
| \(m\) | server | effective parallel normalized servers |
| \(w^{\mathrm{gpu}}_j,w^{\mathrm{cpu}}_j\) | GPU-s/job, core-s/job | class \(j\) compute service work |
| \(m_{\mathrm{gpu}},m_{\mathrm{cpu}}\) | GPU, core | available compute servers |
| \(d^{\mathrm{io}}_j,d^{\mathrm{net}}_j\) | byte/job | class \(j\) storage/network flow |
| \(C_{\mathrm{io}},C_{\mathrm{net}}\) | byte/s | sustainable storage/network bandwidth |
| \(a^{\mathrm{ram}}_j,a^{\mathrm{hbm}}_j\) | byte-s/job | class \(j\) occupancy area |
| \(B_{\mathrm{ram,cap}},B_{\mathrm{hbm,cap}}\) | byte | physical stock capacities |
| \(x_r\) | native resource | measured lifetime consumption of resource \(r\) |
| \(p_r\) | value/native resource | frozen resource price or utility conversion |
| \(\tau\) | s | physical sleep interval in the analytic cadence laws |
| \(L_w\) | s | wake response latency |
| \(B_{\text{move}}\) | byte | bytes moved across boundaries |
| \(\omega\) | s\(^{-1}\) | continuous-time future-value discount rate |

### 2.4 Outcomes

The outcome is a vector, never only QA accuracy:

```text
future utility
retention and forgetting
fixed-budget acquisition gain
addressability and retrieval competition
false memory and calibration
delete and rollback completeness
wake latency and sleep backlog
energy, bytes moved, and lifecycle cost
```

## 3. Total-state identity

`IDENTITY` for **incremental answer-bearing memory state** under the
accounting definitions:

\[
B_{\text{retained}} =
B_r+B_e+B_l+B_p+B_a+B_v.
\]

Replicas and backups are either included in their class or reported as a
separate multiplier. An experiment fails total-capacity accounting if it
reports only active prompt/index state while checkpoints, raw archive, cached
segments, masked experts, or derived versions grow.

Retained incremental state is not peak incremental state. During build,
migration, rollback, or a concurrent read, its disjoint physical ledger is:

\[
\begin{aligned}
B_{\text{peak,state}}(t)={}&B_{\text{retained}}(t)
+B_{\text{candidate}}(t)+B_{\text{pinned}}(t)\\
&+B_{\text{migration}}(t)+B_{\text{workspace}}(t)
+B_{\text{replica}}(t).
\end{aligned}
\]

Candidate/staging objects, reader-pinned predecessors, dual indexes, merge
workspace, and temporary replicas must each have a hard cap and a named
reclamation owner. Every transient term means only bytes not already included
in \(B_{\text{retained}}\): `pinned` is excess state kept solely by a reader,
`migration` is the excess dual representation, and `replica` is an additional
temporary copy. Cumulative transformation work is a separate resource ledger
and is never added to bytes.

The frozen base is public side information for the information-capacity
converse, but its physical residency is not free infrastructure. Report:

\[
B_{\text{peak,physical}}(t)=
B_{\text{base,resident}}(t)+B_{\text{peak,state}}(t)
+B_{\text{execution}}(t),
\]

where \(B_{\text{base,resident}}\) charges every physical base-model copy on
wake and sleep fleets and \(B_{\text{execution}}\) charges activations,
allocator reserve, kernels, and other nonpersistent runtime workspace absent
from the state ledger. Capacity/scientific claims use
\(B_{\text{peak,state}}\); cluster placement, HBM, duplication, and fleet-cost
claims use \(B_{\text{peak,physical}}\). Both are mandatory.
The reported physical maximum is
\[
\max_t\left[
B_{\text{base,resident}}(t)+B_{\text{peak,state}}(t)
+B_{\text{execution}}(t)\right],
\]
with the three terms aligned at the same maximizing timestamp; it is not a sum
of independent component high-water marks.

This identity does not say that every byte has equal information value. It
prevents a hidden-state “compression” claim. A visible summary of \(b\) bytes
does not imply \(B_l=b\) when its KV was computed from now-hidden source
tokens; the retained execution state, model/version dependency, and any
required restart checkpoint still count.

### 3.1 Finite-state lifetime bound

Let \(\Theta_0\) be frozen public side information, including the immutable base
checkpoint and protocol, and let \(M_n\) be **all incremental mutable state**
that can encode the admitted stream after \(n\) items. \(M_n\) includes external
records, latent state, weight deltas, generators, indexes, provenance,
optimizer/router state, and recovery copies. Suppose \(M_n\) is a
finite-precision digital state and
\(H(M_n\mid\Theta_0)\le b_{\text{mutable}}\) bits. For a controlled family of
answers \(Y_i\) that are conditionally independent under the declared
\((Q_i,\Theta_0)\) source model, separable distortion targets \(D_i\), and
conditional rate--distortion functions
\(R_{Y_i\mid Q_i,\Theta_0}(D_i)\), joint queryability requires:

\[
\sum_{i=1}^{n}R_{Y_i\mid Q_i,\Theta_0}(D_i)
\le
I(Y_{1:n};M_n\mid Q_{1:n},\Theta_0)
\le H(M_n\mid\Theta_0)
\le b_{\text{mutable}}.
\]

Label: `NECESSARY` for that declared source/query/distortion model. It is not a
universal count of natural-language facts. Correlated items, pretrained priors,
structured answers, unequal query value, and lossy goals can lower the joint
rate. The frozen base is reported separately in physical bytes. If it changes
to encode the evaluated stream, its delta or complete versioned checkpoint is
charged to \(M_n\); alternatively a total-physical version of the bound must
convert every retained byte to a declared bit representation and charge the
full base inside a total bit bound \(b_{\text{total}}\).

If novel independent items arrive at rate \(\lambda_e>0\) and each requires
stationary positive rate \(\bar R(D)>0\), constant-distortion retention over
physical time \(t\) needs at least:

\[
b_{\text{mutable}}(t)\ge \lambda_e t\,\bar R(D\mid\Theta_0).
\]

Therefore a fixed-state system cannot provide indefinite, constant-distortion
recall for an open-ended positive-entropy stream. It must eventually do at
least one of the following:

```text
expand retained physical state
exploit genuine redundancy or a correlated prior
raise distortion / forget / expire some items
restrict the retained horizon or admission rate
move information across a declared storage boundary
replace exact recall with reconstruction whose retained source is still counted
```

External offload changes where \(B\) is paid; it does not remove the bound.
Likewise, bounded hot memory plus an ever-growing cold archive bounds serving
residency but not lifetime storage. The empirical program estimates useful
capacity knees well before this ideal information limit because retrieval,
interference, plasticity, and governance can bind first.

## 4. Reuse break-even

### 4.1 Discrete special case

Suppose an item costs \(C_{\text{compile}}\) once to compile into a cheaper
serving form. Each later use saves
\(\Delta C_{\text{use}}=C_{\text{baseline}}-C_{\text{compiled}}>0\), while
maintenance and invalidation cost \(C_{\text{maint}}\). The restricted
break-even reuse count is:

\[
n_i^* =
\frac{C_{\text{compile}}+C_{\text{maint}}}
{\Delta C_{\text{use}}}.
\]

Label: `DERIVED-CANDIDATE`. It assumes stationary cost, identical uses, and no
quality difference.

### 4.2 Continuous reuse and volatility

Discounted valid reuse:

\[
R_i(H)=
\int_0^H
\lambda_i(t)S_i(t)e^{-\omega t}\,dt.
\]

If reuse is stationary Poisson and validity survival is
\(S_i(t)=e^{-\mu_i t}\), the infinite-horizon special case is:

\[
R_i(\infty)=\frac{\lambda_i}{\mu_i+\omega}.
\]

Promotion to destination \(j\) is economically admissible only when:

\[
R_i(H)\Delta U_{ij}
>
C_{\text{compile},ij}
+C_{\text{serve},ij}
+C_{\text{govern},ij}
+C_{\text{risk},ij}.
\]

All terms are converted with frozen resource prices into the same declared
utility/cost unit. This is a decision inequality, not proof that estimated
reuse or utility is calibrated.

### 4.3 Empirical test

Sweep reuse count, inter-use gap, volatility, correction probability, query
type, and destination. Fit the observed crossover without using held-out
queries to estimate \(\lambda_i\). Reject a universal threshold if destination
curves cross more than once or workload identity dominates.

## 5. Sleep cadence

### 5.1 General restricted optimum

Let one sleep run cost \(C_s\,[\mathrm{cost}/\mathrm{run}]\), with a run treated
as one invocation, and let delaying consolidation by interval \(\tau\) create
average penalty \(a\tau^p\), \(p>0\). To make \(\bar C\) a cost rate,
\(a\) has unit cost/s\(^{p+1}\):

\[
\bar C(\tau)=\frac{C_s}{\tau}+a\tau^p.
\]

The stationary optimum is:

\[
\tau^* =
\left(\frac{C_s}{ap}\right)^{1/(p+1)}.
\]

Label: `DERIVED-CANDIDATE`. Required assumptions include constant job cost,
smooth stationary delay cost, no batching threshold, no burst, and no hard
deadline.

### 5.2 Linear-staleness special case

For event rate \(r_e\), fixed run cost \(F\), and per-event per-time harm \(h\):

\[
J(\tau)=\frac{F}{\tau}+\frac{h r_e\tau}{2},
\qquad
\tau^*=\sqrt{\frac{2F}{h r_e}}.
\]

Dimensional check:

```text
F/tau                 cost/time
h [cost/(event*time)]
* r_e [event/time]
* tau [time]          cost/time
```

### 5.3 Failure regimes

The formula is not expected to hold under:

- bursty or self-exciting event arrivals;
- operator setup/batch economies;
- multi-resource blocking;
- deadline, deletion, or privacy urgency;
- non-monotone repeated-consolidation damage;
- changing teacher/generator quality;
- capacity-triggered phase changes.

The benchmark compares the analytic special case with segmented, trace-driven,
and learned policies.

### 5.4 Multiple consolidation clocks

For \(N\) tenants and temporal levels \(j=1,\ldots,J\), let
\(\tau_j\,[\mathrm{s}/\mathrm{closure}]\) be the nominal closure interval and
\(\bar F_j\,[\mathrm{work}/\mathrm{job}]\) the measured mean work after
batching, validation, publication, and retry. Under the deliberately
restricted regular-arrival approximation:

\[
\lambda_j \approx \frac{N}{\tau_j},
\qquad
\Lambda_{\mathrm{clock}}
=
\sum_{j=1}^{J}\lambda_j\bar F_j .
\]

This is an `OFFERED-LOAD-ACCOUNTING-CANDIDATE`, not a queue theorem. Session
closures depend on activity, higher levels may consume lower-level products,
and sparse tenants do not generate regular jobs. The trace-driven value uses
observed \(\lambda_j(t)\) and dependency-aware work.

Adding a level \(J+1\) is lifecycle-beneficial only when its future-query and
state-reduction value exceeds its incremental burden. A scalar diagnostic is
legal only after every term below is converted into the same frozen
lifecycle-value-per-second unit:

\[
\Delta C_{\mathrm{clock}}
=
\lambda_{J+1}p_F\bar F_{J+1}
+C_{\mathrm{overlap}}
+C_{\mathrm{distortion}}
+C_{\mathrm{queue}}
+C_{\mathrm{governance}} .
\]

Here \(p_F\) is the frozen lifecycle-value conversion per work unit; every
\(C_\cdot\) is already a rate in that same unit.

Without defensible conversion coefficients, report the native vector
\((\lambda\bar F,\ \text{bytes/s},\ \text{queue s},\ \text{distortion},
\ \text{governance failures})\) and its Pareto frontier; do not add unlike
units.

TiMem's reported roughly 25–30% call increase from L1-only to the full
hierarchy is one empirical anchor, not a universal multiplier. The planned
fit sweeps clock count, activity burstiness, abstraction depth, job
coalescing, and queue service while holding future queries fixed.

## 6. Queue stability

### 6.1 Scalar diagnostic

For normalized jobs:

\[
\rho_s =
\frac{\lambda_s\mathbb E[S_s]}{m}<1.
\]

Label: `NECESSARY` for the simplified queue. It is not a tail-latency
guarantee.

### 6.2 Dimensionally typed resource-vector diagnostics

\[
\sum_j\lambda_j\mathbb E[w^{\mathrm{gpu}}_j] < m_{\mathrm{gpu}},
\qquad
\sum_j\lambda_j\mathbb E[w^{\mathrm{cpu}}_j] < m_{\mathrm{cpu}},
\]

\[
\sum_j\lambda_j\mathbb E[d^{\mathrm{io}}_j] < C_{\mathrm{io}},
\qquad
\sum_j\lambda_j\mathbb E[d^{\mathrm{net}}_j] < C_{\mathrm{net}},
\]

and the Little-law occupancy necessities are

\[
\sum_j\lambda_j\mathbb E[a^{\mathrm{ram}}_j] < B_{\mathrm{ram,cap}},
\qquad
\sum_j\lambda_j\mathbb E[a^{\mathrm{hbm}}_j] < B_{\mathrm{hbm,cap}}.
\]

The first line has units GPU/core, the second byte/s, and the third byte.
RAM/HBM stock is never inserted into a generic bandwidth inequality. These
mean-occupancy necessities do not replace the hard sample-path constraints

\[
\sup_t B_{\mathrm{ram,resident}}(t)\le B_{\mathrm{ram,cap}},
\qquad
\sup_t B_{\mathrm{hbm,resident}}(t)\le B_{\mathrm{hbm,cap}}.
\]

Every work, flow, and occupancy term includes verify, transform, publication,
recovery reserve, and charged retries—not only the central training step.
Schemas reject byte/job×job/s compared with bytes, byte-s/job compared with
byte/s, or an independent component high-water substituted for time-aligned
occupancy.

### 6.3 Measured law candidate

An empirical backlog model may use:

\[
Q_{t+1} =
\max\{0,Q_t+A_t-S_t\},
\]

with class/deadline/affinity state. A “sleep scaling law” fails operationally
if mean utilization looks stable while p99 job age, privacy deadline, or
tenant starvation diverges.

## 7. Byte-flow equilibrium

For every physical store \(j\):

\[
\frac{dB_j}{dt}
=
r_{\text{admit},j}\bar b_{\text{in},j}
-r_{\text{expire},j}\bar b_{\text{delete},j}
-r_{\text{compact},j}\bar b_{\text{reclaimed},j}.
\]

Label: `IDENTITY` at the measured flow boundary.

An active store is not bounded if compaction only moves bytes to an undeclared
archive. A merge ratio is not sufficient; net additions matter. MEMORA's
reported update fraction, for example, does not imply bounded storage when
many candidates still become new records.

Candidate external-growth model:

\[
\frac{dM}{dt}
=
\lambda_e[1-d(M)]
-e(M,F_{\text{reclaim}}),
\]

where \(d(M)\) is the duplicate/merge probability and \(e\) is physical
expiry/reclamation in the same state unit per second as \(dM/dt\), and
\(F_{\text{reclaim}}\) is the declared sleep compute budget allocated to
reclamation. It is distinct from the cadence section's \(C_s\), the value cost
of one sleep run. This is `EMPIRICAL-CANDIDATE`; neither function is assumed
stationary.

### 7.1 Replay-support accumulation

If sleep after task \(t\) fully replays \(m_i\) retained examples from every
task \(i\le t\), its replay-example count is

\[
R_t=\sum_{i=1}^{t}m_i,
\qquad
W_T=\sum_{t=1}^{T}R_t
=\sum_{i=1}^{T}m_i(T-i+1).
\]

This is an `IDENTITY`. For constant \(m_i=m>0\), lifetime replay work is
\(mT(T+1)/2=\Theta(T^2)\), before epochs, model FLOPs, verification, or I/O.
A surprise gate that admits a constant fraction \(\alpha\) changes the
coefficient to \(\alpha m\), not the exponent. A fixed latest-\(w\) window
changes work to \(O(wT)\), but Surprise-Gated Memory supplies direct finite
counterevidence that this cheaper mixture can be worse than no replay:
DINOv2 oldest-task retention `41.2` versus `67.0`, and I-JEPA `0.0` versus
`25.9`.

The empirical question is therefore not “full or bounded replay” in isolation.
It is whether a fixed-byte coreset can preserve the protected-support geometry
needed by the slow learner. Fit retention, acquisition, and work jointly
against total support, age coverage, class/task diversity, surprise threshold,
and backbone separability. An admission rule that reduces bytes but preserves
the same quadratic replay schedule is not a lifetime scaling solution.

### 7.2 Staged-view equilibrium

For a section/topic compiler, clearing staging and shrinking one cycle do not
establish bounded memory. At a declared cycle boundary, let
\(A_c^{\mathrm{total}}\) be newly admitted retained bytes and
\(A_c^{\mathrm{active}}\) the subset admitted directly into the active view.
Let \(R_c^{\mathrm{total}}\) be bytes physically reclaimed from the complete
authorized state, \(R_c^{\mathrm{active}}\) the reclaimed bytes whose
pre-transition origin was active, \(X_c^{\mathrm{out}}\) bytes moved from
active to retained nonactive state, and \(X_c^{\mathrm{in}}\) bytes reactivated
from retained nonactive state. The transition ledger assigns every physical
byte once by its pre/post boundary state. Then

\[
B_{c+1}^{\text{total}}
=B_c^{\text{total}}+A_c^{\text{total}}-R_c^{\text{total}},
\qquad
B_{c+1}^{\text{active}}
=B_c^{\text{active}}+A_c^{\text{active}}+X_c^{\text{in}}
-R_c^{\text{active}}-X_c^{\text{out}}.
\]

These are `IDENTITY` statements at declared boundaries. A positive
active-store compression ratio can coexist with
\(B_{c+1}^{\text{total}}>B_c^{\text{total}}\). Evolve reports a 31–34%
canonical-section reduction for one cycle over fixed repeated questions, but
has no hard cap and leaves unqueried expired canonical sections resident.
Memento-Skills Dream similarly bounds selected next-wake topics, not total
topics, skills, versions, indexes, or source sessions.

A stationary bounded-store claim therefore needs evidence that

\[
\limsup_{C\rightarrow\infty}
\frac{1}{C}\sum_{c=1}^{C}
\mathbb E[A_c^{\text{total}}-R_c^{\text{total}}]\le 0
\]

under a nondegenerate stream of novel topics and corrections, while utility,
deletion, and provenance constraints remain satisfied. This is a necessary
flow condition, not a sufficient utility theorem.

### 7.3 Index--payload coherence and refresh amplification

An external memory item is not only its visible text or graph payload. At
serving generation \(g\), item \(i\) is at least the coupled object

\[
\left(
x_i^{(g)},
z_i^{(g)},
\theta_{\mathrm{enc}}^{(g)},
\pi_i^{(g)}
\right),
\qquad
z_i^{(g)}=f_{\theta_{\mathrm{enc}}^{(g)}}(x_i^{(g)}),
\]

where \(x_i\) is the canonical payload, \(z_i\) is its retrieval key,
\(\theta_{\mathrm{enc}}\) identifies the encoder, and \(\pi_i\) is provenance
and authorization state. The equality is a publication invariant, not a
statistical scaling law. A sleep operator that rewrites \(x_i\) while
retaining the old \(z_i\) has published a mixed semantic generation even if
both writes individually succeeded.

For an audited candidate generation, define the key--payload discrepancy

\[
d_i^{(g)}
=
1-\cos\left(
z_i^{(g)},
f_{\theta_{\mathrm{enc}}^{(g)}}(x_i^{(g)})
\right)
\]

and the access-weighted incoherence rate

\[
I_\tau^{(g)}
=
\frac{\sum_i w_i\mathbf 1[d_i^{(g)}>\tau]}
{\sum_i w_i}.
\]

These are `MEASUREMENT-DEFINITIONS`. They do not assume that cosine distance
is the best retrieval metric; a backend-specific exact-key or ranking
equivalence test may replace it. Encoder-digest mismatch is an unconditional
failure even when a sampled \(d_i\) happens to be small.

Let \(\mathcal K_c^{\mathrm{aff}}\) be the keys affected by cycle \(c\), and
let \(e_{ic}\) be measured encoding-and-publication work for item \(i\). The
general item-operation accounting bound is

\[
W_{\mathrm{index}}
\ge
\sum_{c=1}^{C}\sum_{i\in\mathcal K_c^{\mathrm{aff}}}e_{ic}.
\]

With a fixed encoder and \(m_c\) changed payloads,
\(\mathcal K_c^{\mathrm{aff}}\) is the changed set, so for constant per-item
work \(e\):

\[
W_{\mathrm{local-index}}
\ge
e\sum_{c=1}^{C}m_c .
\]

Under the narrower **single-current-encoder serving** contract, with no
versioned federation or query-time translation, an incompatible encoder change
makes \(\mathcal K_c^{\mathrm{aff}}\) the complete active population \(M_c\),
so:

\[
W_{\mathrm{global-index}}
\ge
e\sum_{c=1}^{C}M_c .
\]

These are `CONDITIONAL-ACCOUNTING-LOWER-BOUNDS` for a materialized dense index.
If \(M_c=\Theta(c)\), the single-current-encoder contract plus a global
incompatible change each cycle implies \(\Theta(C^2)\) lifetime refresh work
before graph repair, validation, or replication. A versioned federation,
resident old encoder, query translation, or dual index can reduce the immediate
affected set. It must instead charge every resident encoder/index byte, query
fan-out and calibration, migration work, and deletion fan-out. Those systems
are alternative cost placements, not counterexamples to the affected-set
identity.

This distinction is visible in two opposite implementation choices. PCMC
explicitly re-embeds centroids after sleep retrains its encoder. The audited
paper-time and current LightMem implementation rewrites a memory payload while
passing the incumbent vector back to the vector store. The latter is
code-level evidence of a possible coherence failure, not evidence that every
LightMem result is invalid. The benchmark must measure whether stale-key
drift changes retrieval, and the release gate must reject the mixed generation
regardless of measured average utility.

### 7.4 Bounded published state does not bound lifetime rewrite work

Let a reconstructive consolidator publish at most \(b\) tokens after cycle
\(c\), and let \(F_c(b,h_c)\) be the measured cost of reading its bounded
prior state plus eligible wake/thread state \(h_c\), generating the candidate,
verifying it, and publishing it. Then:

\[
B_{\text{published}}(c)\le b
\quad\not\Rightarrow\quad
W_{\text{lifetime}}(C)=O(1),
\]

because

\[
W_{\text{lifetime}}(C)
=
\sum_{c=1}^{C} a_c F_c(b,h_c),
\]

where \(a_c\in\{0,1\}\) indicates whether reconstruction runs. This is an
`ACCOUNTING-IDENTITY`. If every cycle rewrites and measured
\(F_c\ge F_{\min}>0\), lifetime work is \(\Omega(C)\) even though the served
state is constant-size. If eligible history grows until a cap \(H\), the
transient cost can grow faster before becoming \(O(CH)\); the exact relation
must be measured from tokens, calls, and wall time rather than inferred from
the output cap.

This is the correct reading of MIRROR-style `O(1)` reconstruction: constant
space/per-turn boundedness, not free lifetime consolidation. A trigger policy
is beneficial only if the skipped rewrite cost and error exposure exceed the
lost marginal future utility and additional staleness. The scaling experiment
therefore sweeps horizon, event novelty, pause distribution, and trigger rate,
including full-rewrite, event-triggered, and no-rewrite controls.

### 7.5 Bounded hot state is not lifetime equilibrium

For cycle \(c\), use one disjoint physical-media partition for retained bytes:

\[
\begin{aligned}
B_{\mathrm{retained},c}
=\;&B_{r,c}+B_{e,c}+B_{l,c}+B_{p,c}+B_{a,c}+B_{v,c}.
\end{aligned}
\]

The six terms are mutually exclusive byte-ownership classes: raw/canonical
evidence, external text/vector/graph (including its index), latent/KV,
parametric modules, auxiliary catalog/router/optimizer state, and
version/lineage/recovery state. Content-addressed physical bytes are charged
once to their owning class; references and replicas are charged where they
physically reside. `hot`, `active`, and `archive` are overlapping residency or
lifecycle projections of this partition, not extra summands. For example,
\(B_{\mathrm{hot},c}\le B_{\mathrm{retained},c}\), while catalog bytes belong
to \(B_a\), adapters to \(B_p\), and recovery checkpoints to \(B_v\).

Therefore:

\[
B_{\mathrm{hot},c}\le b
\quad\not\Rightarrow\quad
\sup_c B_{\mathrm{retained},c}<\infty .
\]

The six-class sum is an `ACCOUNTING-IDENTITY`; the implication failure is a
logical counterexample, not a second identity. GAM's fixed 2,048-token
progression buffer with growing event/topic graphs and raw archive is a direct
empirical example. RecMem's smaller constructed view depends on a verbatim
store whose removal harms accuracy. TiMem's smaller recalled context adds
hierarchical views and calls. An adapter's fixed rank similarly does not bound
adapter count, catalog metadata, or recovery checkpoints.

A system demonstrates bounded lifetime state only if it publishes declared
retained and peak caps, counts every term above plus the transient ledger from
Section 3, and shows:

\[
\Pr\!\left[
\substack{
\max_{c\le C}B_{\mathrm{retained},c}\le B_{\mathrm{retained,max}}\\
\text{and }\sup_{t\le H}B_{\mathrm{peak,state}}(t)
 \le B_{\mathrm{peak,state,max}}
}
\right]
\ge 1-\alpha
\]

over the preregistered finite horizon **and** specifies the admission,
replacement, tiering, or forgetting policy that could sustain the bound beyond
that horizon. A finite run below a cap is evidence for that run, not an
asymptotic guarantee.

## 8. Addressability and retrieval competition

Storage survival does not imply useful recall. Let \(M\) be total stored
memories, \(k\) retrieved memories, \(T_{\text{live}}\) the visible-token cap,
\(B_{\text{live}}\) the resident-byte cap, and \(Q\) query type:

\[
U_{\text{read}}
=
G(M_{\text{useful}},k,Q)
-I(M,k,Q)
-L(M,k)
-S_{\text{stale}}(M,Q).
\]

Label: `EMPIRICAL-CANDIDATE`.

Candidate measurements:

- oracle-store recall versus retriever recall;
- retriever recall versus reader use;
- false association as \(M\) and \(k\) grow;
- exact, temporal, relational, and procedural query classes;
- active context token and latency constraints;
- contradiction and stale-version rate.

ReasoningBank's lower performance at larger retrieved `k` and HIMA's budget
curves are anchor observations, not sufficient data for one exponent.

Query reuse also needs a relationship-conditioned decomposition. Let
\(\rho_r\) be the future-query share for relationship class \(r\), and
\(h_r(M)\) the probability that memory avoids a teacher call while preserving
the frozen quality floor:

\[
h_{\text{deploy}}(M)
=
\sum_{r\in
\{\text{exact},\text{paraphrase},\text{composition},
\text{correction},\text{shift}\}}
\rho_r h_r(M).
\]

This is an `IDENTITY` once classes and deployment shares are fixed. Repeating
the acquisition questions estimates \(h_{\text{exact}}\), not
\(h_{\text{deploy}}\). Evolve's identical cold/warm/post order therefore
cannot identify semantic amortization without held-out relationship classes.

### 8.1 Coverage-budget candidate

Memento 2 bounds its value error using a memory coverage radius \(r_M\) and
retrieval error \(\delta_M\), but assumes finite current memory and does not
supply an eviction mechanism. A restricted bridge from memory count to
coverage is possible only after declaring geometry. Suppose the relevant state
support is a compact metric set with effective covering dimension \(d>0\), and
there are constants \(a_1,a_2\) such that its covering number obeys

\[
a_1 r^{-d}\le N_{\mathcal X}(r)\le a_2 r^{-d}
\]

over the measured radius range. An approximately optimal \(M\)-prototype
cover with a fixed codec/precision—and therefore fixed bytes per
prototype—then has

\[
r_M=\Theta(M^{-1/d}).
\]

This is a `CONDITIONAL-DERIVATION`, not a universal property of language
memory. Under Memento 2's additional local-consistency, stationarity, bounded
reward, and discount assumptions, a constructive upper-bound candidate
becomes

\[
\lVert V^\star-V^{\pi_M}\rVert_\infty
\le
\frac{2R_{\max}}{(1-\gamma)^2}
\left[
\epsilon_{\mathrm{LLM}}\!\left(cM^{-1/d}\right)
+\delta_M
\right].
\]

The result does not say the right-hand side is tight or that item count alone
controls value. Distribution shift, noncompact/open-ended support, changing
policy visitation, anisotropic task relevance, merge distortion, and a
retriever whose \(\delta_M\) rises with competition can erase the nominal
coverage gain.

The experiment therefore has two non-interchangeable arms:

1. **cardinality arm:** freeze codec, precision, and bytes per prototype, allow
   total bytes to grow with \(M\), and test the \(M^{-1/d}\) geometry term;
2. **fixed-active-representation-budget arm:** freeze
   \(B_{\rm active\,repr,total}\), including headers, index/keys, metadata, decoder, and
   payload; vary \(M\); measure
   \(b_{\rm payload}(M)=
   [B_{\rm active\,repr,total}-B_{\rm fixed}-B_{\rm index}(M)
   -B_{\rm metadata}(M)]/M\);
   and fit
   \(r_{\mathrm{eff}}(M,B)\lesssim
   cM^{-1/d}+q(b_{\rm payload}(M))\). Here \(q\) is independently measured
   quantization distortion, and an infeasible negative payload allowance is a
   failed row rather than hidden extra state.

Both estimate \(d\), achieved radius, codec distortion, and \(\delta_M\) on
held-out states while comparing uniform prototypes, past-observable
value-aware coresets,
learned quantization, and eviction. Only the first arm identifies the pure
cardinality exponent.
Future-value features make a coreset an oracle-only upper bound. Raw evidence,
lineage, rollback, and recovery copies remain in separately reported complete
retained/peak-state ledgers, so the second arm is not a fixed-total-memory law.

## 9. Capacity knee

### 9.1 Threshold-anchored logistic knee

\[
\operatorname{logit}p_{\rm ret}(N_a)
=
\operatorname{logit}\tau_{\rm ret}
-\kappa\log(N_a/K_{\text{eff}}),
\qquad \kappa>0.
\]

For the confirmatory lag-128 surface,
\(\tau_{\rm exact}=\tau_{\rm temporal}=0.80\), frozen in deployment stakes
before bracket CAL. It represents the requirement that at least 80% of
admitted associations remain answerable; prospective amendments must keep
thresholds in `[0.10,0.90]`. The low/high load bracket requires simultaneous
CAL bounds for **both** components above 0.90 and below 0.70 respectively.
Changing either threshold stales the bracket, knees, contrast family, power,
and preregistration.

`K_eff` has the same association-count unit as \(N_a\), so the logarithm is
dimensionless, and the parameterization guarantees
\(p_{\rm ret}(K_{\rm eff})=\tau_{\rm ret}\) for the frozen retention threshold.
`K_eff` is task-, medium-, resource-, and tolerance-specific. It is not the
unqualified sigmoid midpoint unless \(\tau_{\rm ret}=0.5\), and it is not a
universal number of facts. Power, fixtures, and surface fitting import this
same threshold-anchored function; changing \(\tau_{\rm ret}\) while holding raw
logistic coefficients fixed must change the derived \(K_{\rm eff}\).

Eight bins and a finite coefficient do not establish model adequacy. The
confirmatory estimator applies a frozen monotonic-decrease, single-crossing,
and simultaneous residual-envelope test. If the logistic fails, a
shape-constrained monotone crossing sensitivity must agree within 0.10
log-capacity; multiple crossings, envelope failure, or larger disagreement is
`KNEE_UNIDENTIFIED`, not a usable finite knee. Power includes Gompertz,
mixture/two-knee, flat, and separated alternatives.

### 9.2 Competing capacity surfaces

Freeze a reference resource tuple \(r_0\) and define the previously implicit
normalizer exactly:

\[
K_0(f,m)=K_{\rm eff}(f,r_0,m).
\]

The primary law targets within-family
\(\Delta\log K(f,r,m)=\log K_{\rm eff}(f,r,m)-\log K_0(f,m)\), fits
\(g_m(r_0)=0\) with no free family intercept, and uses equal weights over the
16 fixed benchmark families. A centered design plus a free family intercept is
rank-invalid. A named held-out family consumes its preregistered charged \(r_0\)
`normalization_only` anchor but cannot use it to refit or score the resource
function. The anchor is excluded from every validation denominator while its
uncertainty/shared covariance propagates into all other held-out deltas.
Absolute zero-shot \(K\) prediction for an unseen family is not identified
without such an anchor or prospectively frozen predictive family covariates.

All logarithms in this contract are natural logarithms of dimensionless ratios:
\(\Delta\log K=\ln(K/K_0)\) and resource coordinate
\(x_j=\ln(r_j/r_{0j})\). A “per doubling” effect is the finite contrast between
\(r_j\) and \(2r_j\) on that response scale. Coverage uses residual
\(e=\ln(\widehat R/R)\) but claim-bearing accuracy is the nonnegative
equal-weight \(\sqrt{\sum_c w_c e_c^2}\), never a signed mean that can cancel
under- and overprediction. Log bases and references are schema fields; replacing
ln by log2 without transforming thresholds invalidates the artifact digest.

One candidate smooth surface:

\[
\frac{K_{\text{eff}}(f,r,m)}{K_0(f,m)}
\propto
\left(\frac{B_{\text{store,cap}}}{B_{\text{store,cap},0}}\right)^\alpha
\left(\frac{T_{\text{live,cap}}}{T_0}\right)^\beta
\left(\frac{F_{\text{sleep,cap}}}{F_0}\right)^\gamma
\left(\frac{b_{\text{repr,cap}}}
           {b_{\text{repr,cap},0}}\right)^\nu.
\]

The controlled input surface has exactly four assigned axes: active
external/store cap \(B_{\text{store,cap}}\), answer-time token cap
\(T_{\text{live,cap}}\), sleep-compute cap or prescribed schedule
\(F_{\text{sleep,cap}}\), and serialized latent-plus-adapter
\(b_{\rm repr,cap}\) in bits. Slot and parameter counts are converted by a
frozen precision/serialization schema; mixed counts/bytes/bits are forbidden.
The fit is intention-to-treat on assignments. Realized \(B_{\text{live}}\),
tokens, FLOPs, physical HBM residency, \(B_r\), and
\(B_{\text{peak,physical}}\) remain measured constraints/outcomes; they cannot
replace an axis or become a hidden fifth controlled input.

Required competitors:

```text
min-bottleneck
normalized Cobb--Douglas
generalized mean
segmented knee
interaction surface
nonparametric monotone fit
```

If the relation changes sign or misses held-out workloads/hardware, it remains
an empirical curve.

### 9.3 Capacity types

Separate knees are estimated for:

```text
physical storage
retention/representation
addressability/application
plasticity/acquisition
governance/recoverability
sleep service
hot-state residency
```

The first binding knee defines the current regime but may move after an
intervention.

### 9.4 Conditional parametric bit capacity

For a declared data process \(\mathcal D\), query/readout process
\(\mathcal Q\), architecture \(\mathcal A\), precision \(\pi\), training
protocol \(\mathcal T\), and distortion \(D\), define a measured efficiency:

\[
\widehat c_\theta
\left(
\mathcal D,\mathcal Q,\mathcal A,\pi,\mathcal T,D
\right)
=
\frac{
\widehat I(Y;\theta\mid Q,\theta_0)
}{
\Delta P
}
\quad
\text{bits/added parameter},
\]

where \(\theta_0\) is the frozen base state and \(\Delta P\) is the declared
added/modified parameter count. If optimizer, routing, or generator state
carries answer information, it belongs in the denominator or in a separately
reported complete-state efficiency.

Allen-Zhu and Li's roughly two knowledge bits/parameter and Morris et al.'s
roughly 3.6 memorized bits/parameter use different controlled objects and
protocols. They are conditional reference points for this surface, not two
estimates of one universal constant. Natural-language facts generally lack
known entropy and separable readouts, so the confirmatory fit uses opaque
independent payloads. Natural-language and factual QA remain external-validity
tracks.

The estimator cannot come from one checkpoint. For each independent
opaque-payload seed, freeze \(\theta_0\), retrain the candidate state from
scratch, and evaluate a decoder/readout whose bit orientation was frozen on
CALIBRATION before TEST. The source uses \(N\) associations with \(L\)
independent uniform Bernoulli payload bits each. For true association×bit
decoder error \(p_{a\ell}\), the population Fano functional is

\[
\mathcal I_{\mathrm{Fano,total}}
=\sum_{a=1}^{N}\sum_{\ell=1}^{L}
[1-h_2(p_{a\ell})],
\qquad
\mathcal I_{\mathrm{Fano/bit}}
=\frac{\mathcal I_{\mathrm{Fano,total}}}{NL}.
\]

The finite-sample inferential lower bound is not the upward-biased plug-in
\(\sum[1-h_2(\widehat p)]\). It either maps frozen-orientation simultaneous
upper confidence bounds \(p^U_{a\ell}\) through
\(\max\{0,1-h_2(\min(p^U_{a\ell},0.5))\}\), or uses a directly validated
simultaneous lower confidence procedure for the nonlinear functional. Report a
separate bias-corrected descriptive plug-in estimate. Thus no signal is
\(p_{a\ell}=0.5\); the error is never pooled before applying the nonlinear
entropy function or reinterpreted as block error, whose null would differ.
Use the master seed family as the resampling cluster and control
simultaneously across bits, loads, precision, and substrate. A shuffled
small-sample/no-signal fixture must return inferential lower bound zero and
non-support even when the descriptive plug-in is positive. Report the
unnormalized bound and complete-state bit efficiency. A separately
preregistered variational lower
bound is optional but cannot rescue the Fano track. Conditioning is allowed
only on frozen public, payload-seed-independent side information included in
\(\Theta_0\).
Every \(Y\)-, payload-, TRAIN-, CALIBRATION-, or TEST-dependent adapter,
router, optimizer, codebook, auxiliary decoder, or checkpoint is included in
the complete mutable-state bit denominator; it cannot be conditioned away.
A theta-only efficiency may be reported separately only when the frozen
readout accesses \(\theta\) and public \(\Theta_0\) alone. Changing the decoder
after TEST is forbidden. Power freezes \(L\), the minimum information margin,
its boundary channel for type-I calibration, the distinct detection
alternative, and the confidence-interval precision target; association-load
changes never change the denominator silently.

The post-training evaluation capability cannot read the payload RNG seed,
generated dataset, replay log, or a regeneration service. If retained for
reproducibility/recovery, those objects' canonical compressed bit lengths enter
complete mutable state and invalidate the theta-only branch. A
\(\theta=\theta_0\) fixture that answers by regenerating payloads from a seed
is hidden external state, not parametric information.

Externalization also has no automatic conservation law of the form “one
factual bit removed yields one reasoning bit.” Call-policy parameters,
unused capacity, optimization geometry, and data composition can absorb the
difference. The usable-release estimand is the fixed-budget improvement in
fresh non-factual acquisition after factual targets are routed externally,
holding gradient-bearing tokens and external-call resources fixed.

## 10. Parametric interference

### 10.1 Restricted geometry candidate

For sleep updates \(g_t\) and protected-task gradients \(g_o\), gradient or NTK
overlap can diagnose interference:

\[
\chi =
\frac{\langle g_t,g_o\rangle}
{\|g_t\|\|g_o\|}.
\]

The diagnostic cell is invalid when either gradient norm is zero; no epsilon
substitution is allowed in the confirmatory value. This is a local diagnostic,
not a lifetime law. *Facts in Weights* reports that
a local linearized update predictor did not predict later forgetting well,
which directly warns against promoting \(\chi\) to a controller without
held-out calibration.

### 10.2 Accumulated protected directions

Regularisation candidates imply:

```text
less raw replay
→ more reference/importance state
→ more protected directions
→ potentially less future plastic subspace
```

The final arrow is `HYPOTHESIS`, measured rather than assumed.

### 10.3 Parametric pool

For isolated experts/slots:

\[
f_{\text{free}} =
\frac{P_{\text{free}}+P_{\text{verified-recyclable}}}
{P_{\text{total}}}.
\]

Candidate regime change occurs when `f_free` falls below a frozen safety
threshold. Active parameters staying constant does not mean total physical
capacity is constant.

### 10.4 Modular-parametric routing and merge knee

For \(K\) adapters with ranks \(r_1,\ldots,r_K\), let
\(\Delta P(r_k)\) be each module's trainable parameters and \(B_{\mathrm{cat}}\)
the catalog/router/compatibility state. A matched comparison requires:

\[
\sum_{k=1}^{K}\Delta P(r_k)
=
\Delta P_{\mathrm{single}},
\]

while reporting \(B_{\mathrm{cat}}\), load/cache state, and optimizer/recovery
bytes separately. Parameter matching alone is not total-state matching.

Let \(U_{\mathrm{oracle}}\) be utility when the correct module set is supplied
and \(U_{\mathrm{deploy}}\) be utility under the deployed router, merge, and
residency policy. The confirmatory measurable tax is only the aggregate:

\[
L_{\mathrm{module}}
=
U_{\mathrm{oracle}}-U_{\mathrm{deploy}}.
\]

PAPER-C attributes none of this aggregate gap. Attribution to routing, merge,
or residency is extension `X-STC-Q` and remains exploratory unless a pre-TEST
amendment materializes, powers, and multiplicity-binds a full
\(2\times2\times2\) routing×merge×residency intervention at G2. A post-hoc
additive decomposition is forbidden.

Only in that promoted extension is router competence frozen on CALIBRATION
before TEST: route-recall lower confidence bound
\(\ge x_{\mathrm{route}}\), out-of-catalog false-positive upper bound
\(\le y_{\mathrm{ooc}}\), and routing/load cost
\(\le z_{\mathrm{serve}}\), with all three numerical thresholds frozen at G2.
At least one calibration-qualified router is required. If none qualifies, the
extension comparison is `INCONCLUSIVE`; without the amendment, every component
result is exploratory and cannot qualify or rescue the aggregate PAPER-C
claim.

The modular advantage region is the set of \((K,r,\rho_q,\chi,\gamma)\) where:

\[
\Delta U_{\mathrm{isolation}}
>
L_{\mathrm{module}}+C_{\mathrm{catalog/load}},
\]

after converting serving cost into the same frozen utility unit.
\(\rho_q\) describes query/module sparsity, \(\chi\) module-gradient overlap,
and \(\gamma\) composition demand. LoRA-as-memory's oracle `8×rank-4` versus
rank-32 result and the large gap between robust and naive merge methods are
anchors for this candidate surface, not a universal rank law.

Sweep rank, module count, source entropy, query composition, router
supervision, catalog residency, and correction depth. Fit a segmented knee and
interaction surface; reject the proposed modular region if held-out learned
routing never recovers the oracle isolation benefit at matched lifecycle
cost.

## 11. Plasticity boundary

Define fixed-budget acquisition gain:

\[
g_{\text{acq,current}}
=
\frac{
U(\theta\;\text{after fixed new-task budget})
-U(\theta\;\text{before})
}{
U(\theta_{\text{fresh}}\;\text{after same budget})
-U(\theta_{\text{fresh}}\;\text{before})
}.
\]

The ratio is valid only when the fresh-control denominator exceeds a frozen
minimum learnability threshold. An unlearnable or numerically degenerate probe
invalidates the cell rather than being imputed as zero.

The dimensionless load is:

\[
\rho_{\text{plastic}}
=
\frac{g_{\text{acq,min}}}
{\max(g_{\text{acq,current}},\epsilon_g)}.
\]

Candidate boundary: \(\rho_{\text{plastic}}>1\).

`H-STC-007` predicts that in at least one preregistered high-interference
regime this boundary occurs before aggregate old-retention failure. The
intervention prediction—restricted low-utility recycle with canonical
replay—has a separate test.

Secondary diagnostics such as dormant-unit fraction, stable rank, gradient
rank, and weight magnitude do not replace the behavioral acquisition endpoint.

## 12. Recoverability and no-free-resurrection

Let \(Z\) be a task-specific memory variable. Let retained state \(\Omega\)
contain external, latent, parameter, auxiliary, and recovery state. If:

\[
I(Z;\Omega)=0
\]

and new randomness \(R\) is independent of \(Z\) conditional on \(\Omega\),
then:

\[
I(Z;\Omega,R)=0.
\]

Label: `NECESSARY` information boundary under the stated independence. It says
that erased task-specific information cannot be reconstructed from independent
randomness. It does not say a correlated pretrained prior cannot guess the
answer, or that behavioral recall reveals every retained trace.

An empirical recoverability proxy can combine:

```text
old-task canary margin
teacher agreement
source availability
retrieval confidence
representation conflict
```

`H-STC-006` predicts that a past-only calibrated proxy identifies the
preregistered recoverability boundary. At matched compute, enqueue-time
information, and operator, intervention immediately before that boundary is
compared with intervention after it and with fixed cadence. The empirical
hypothesis fails when the proxy is uncalibrated on held-out streams or the
before-boundary action does not improve the matched frontier; either result is
separate from the information boundary.

## 13. Reversibility reserve

Let archive/recovery depth be \(D\). More reserve can enable aggressive
compaction and rollback, but also adds bytes, attack surface, delete work, and
stale versions:

\[
U_{\text{reserve}}(D)
=
V_{\text{rollback}}(D)
-C_{\text{store}}(D)
-C_{\text{attack}}(D)
-C_{\text{delete}}(D).
\]

`H-STC-005` predicts a non-monotone optimum in at least one preregistered
eligible regime. It is falsified only if every eligible regime spans an
adequately powered archive range and remains monotone. A monotone curve in one
regime, or an under-ranged curve, narrows the claim or establishes insufficient
range; the analysis does not force an interior optimum.

## 14. Promotion and demotion hysteresis

Let destination state be external \(E\) or compiled \(P\), with switching
costs \(C_{E\to P}\) and \(C_{P\to E}\). Separated thresholds may arise when
value curves satisfy monotone single crossing and switching costs are
positive.

Label: `DERIVED-CANDIDATE` under verified sufficient conditions.

If conditions fail, the output is an empirical policy surface, not a universal
hysteresis law. Compare against a one-threshold policy and measure thrashing,
stale compiled state, and missed promotion.

## 15. Data-movement and fleet crossover

### 15.1 Movement cost

\[
C_{\text{move}}
=
\sum_\ell
\left(
B_\ell p_\ell
+T_\ell v_{\text{delay}}
\right).
\]

\(B_\ell\) and \(T_\ell\) are respectively bytes and seconds for link class
\(\ell\); \(p_\ell\) and \(v_{\text{delay}}\) convert byte and second into the
same frozen lifecycle-value unit. `B` includes source read, network,
host-device, replica, checkpoint, and publication traffic by link class.

### 15.2 Fleet alternatives

```text
shared priority fleet
time-partitioned fleet
separate wake/sleep fleet
shard-local hybrid
```

Candidate crossover variables:

```text
sleep load and burstiness
wake P99 interference
model weight residency
source shard size
delta/output size
batching opportunity
network bandwidth
privacy locality
deadline
```

There is no evidence-based right to assume a separate cluster is always best.

### 15.3 Memory-harness/control-program search amortization

MemEvolve, MemSkill, ALMA, M★, and RHO expose a control-plane scaling term
that ordinary memory accounting omits: each candidate harness or skill policy
may require **re-ingesting episodes** or repeatedly re-solving a trajectory
coreset before it can be scored. For \(I_h\) evaluated mutations, \(N_e\)
episodes, and \(N_v\) validation cases, a first accounting model is:

\[
C_{\text{harness-search}}
=
\sum_{i=1}^{I_h}
\left[
N_e c_{\text{write},i}
+N_v(c_{\text{read},i}+c_{\text{agent},i}+c_{\text{judge},i})
+c_{\text{reflect},i}
+c_{\text{check},i}
\right].
\]

This is an `IDENTITY` after each measured component is defined, not a law that
cost is linear: caches, representative subsets, parallelism, variable
trajectories, and rubric calls change the terms. M★ uses 20 iterations per
configuration, reports approximately `$12–90` search cost, and reaches about
`100 h` wall time for ALFWorld; those finite points do not establish an
exponent. MemEvolve instead uses a three-iteration Pareto evolution in its
paper setting, with 60 trajectories per candidate round; ALMA reports 11
meta-learning steps and 43 discovered designs; MemSkill alternates 100 PPO
training steps with a designer cycle. These are distinct points on the same
cost surface, not interchangeable “harness training” samples.

RHO makes the retrospective-call decomposition explicit. For a trajectory pool
of \(N_p\), selected coreset \(k\), rollout group \(G\), and candidate count
\(N_h\), one paper-aligned round has the measured accounting identity

\[
\begin{aligned}
C_{\text{RHO-round}}={}&
N_p c_{\text{difficulty}} + c_{\text{embed-batch}}
+ kG c_{\text{before}} + k c_{\text{diagnose}}\\
&+N_h c_{\text{propose}}
+kN_h(c_{\text{after}}+c_{\text{rank}})
+c_{\text{independent-canary}}.
\end{aligned}
\]

The paper sets \(N_p=100,k=10,G=3,N_h=3\): excluding selection and the
independent canary that production would still need, the last five terms
produce `30 + 10 + 3 + 30 + 30 = 103` agent invocations. Its full reported
RHO run costs `23.1 h` summed agent time but `3.2 h` elapsed under ten-way
concurrency. Parallelism changes latency, not total work. Over \(R\) rounds,
cost is \(\sum_r C_{\text{RHO-round},r}\); no linear exponent may be claimed
until cache reuse, trajectory growth, early stopping, and model-call
heterogeneity are measured.

The state term must also include the whole transitive procedural artifact:

\[
B_{\text{harness,total}}(R)
=B_{\text{active}}(R)+B_{\text{versions}}(R)+B_{\text{tools}}(R)
+B_{\text{trajectories}}(R)+B_{\text{diagnoses}}(R)+B_{\text{logs}}(R).
\]

Fixed \(k,G,N_h\), or a 32 KB instruction-file limit bounds neither this sum
nor its migration, deletion, and validation debt.

Search is lifetime-admissible only if its held-out future benefit exceeds
search, migration, revalidation, and drift-triggered re-search cost. Because
five of six cross-task transfers in M★ fall below its universal seed, the
relevant sweep is hierarchical:

```text
one universal harness
shared kernel + task module
tenant-specific harness
per-task harness
continuous re-search after drift
```

The held-out crossover should predict when specialisation repays its repeated
ingestion and governance cost; in-sample validation gain is insufficient.

### 15.4 Diagnostic-probe scaling

MemMA exposes a second omitted term: provisional memory may be tested before
publication. For session \(s\) with \(J_s\) probes,

\[
C_{\text{probe},s}
=
C_{\text{generate},s}
+
\sum_{j=1}^{J_s}
\left(
C_{\text{retrieve},sj}
+C_{\text{answer},sj}
+C_{\text{judge},sj}
+\mathbb 1[\text{fail}_{sj}]C_{\text{repair},sj}
\right)
+C_{\text{retest},s}.
\]

This is an accounting identity, not evidence that \(J_s=5\) is optimal.
The verification frontier must sweep probe count, type coverage, generator
quality, failure prevalence, and state size while measuring marginal
downstream error detected per cost. Adaptive stopping is admissible only from
past probe outcomes; final evaluation questions remain hidden. A generator or
filter that memorizes held-out answers belongs in total state and invalidates
the estimate rather than providing “free” diagnostic power.

### 15.5 Wake delivery-compilation accounting

For a MemCompiler-like delivery path, compare complete action latency:

\[
L_{\text{compiled}}
=L_{\text{retrieve}}+L_{\text{compiler}}+L_{\text{executor}}
\quad\text{versus}\quad
L_{\text{inject}}
=L_{\text{retrieve}}+L_{\text{executor-with-memory}}.
\]

The paper's `.093 s` compiler plus `.12 s` executor is about `.213 s`, not the
`.12 s` executor component alone; relative to the reported `.30 s`
ahead-of-time baseline, this is approximately a `29%` complete-path reduction
under that table's omitted/common-term assumptions. The same accounting must
include compiler parameters, Soft-Mem projection, task-memory bytes, and Brief
State. This relation belongs to wake serving, but it determines whether
sleep-produced memory can be delivered cheaply enough to have positive
lifetime value.

### 15.6 Explicit/implicit state completeness and portability

Let \(X\) be the visible compressed artifact, \(Z\) its surviving latent/KV
state, \(Y\) a downstream answer, and \(S\) a source secret. A text-only
compression claim is complete only if the declared deployment semantics make
\(Z\) unnecessary:

\[
I(Y;Z\mid X,\theta,Q)=0.
\]

MEMENTO supplies direct counterevidence for its own stateful runtime:
recomputing \(Z\) from identical \(X\) without the original reasoning block
changes AIME24 pass@1 from `66.1%` to `50.8%`. Its downstream passcode probes
also imply \(I(S;Z\mid X)>0\) in the tested construction. These are
source-specific empirical results, not universal constants, but they establish
two required estimands:

\[
\Delta U_{\text{port}}
=
U(X,Z_{\text{source-conditioned}})
-U(X,Z_{\text{restart}})
\]

and

\[
L_{\text{latent-secret}}
=
\operatorname{Adv}\!\left[\widehat S(Z),S\right],
\]

where attack advantage is measured over a frozen chance baseline with
label-unique holdout secrets. Both must be swept against compression ratio,
model scale, layer, state quantization, serialization, and hop depth.

Memory efficiency also requires two non-equivalent measures:

\[
B_{\text{KV,peak}}=\max_t B_{\text{KV}}(t),
\qquad
A_{\text{KV}}=\int_0^T B_{\text{KV}}(t)\,dt.
\]

A lower \(B_{\text{KV,peak}}\) does not imply lower \(A_{\text{KV}}\): longer
generation or poor interaction with sliding-window attention can reverse the
second metric. MEMENTO's OLMo-3 AIME'26 arm has a `.91x` peak ratio but a
`1.02x` AUC ratio. A complete scaling surface therefore includes explicit
bytes, implicit bytes, peak residency, memory-time, utility, portability, and
residual-information risk rather than reporting one “compression ratio.”

Label: the equations are `IDENTITY`/estimand definitions; the conditional
information claims are empirical for the cited experiment. `X-STC-J`
hypothesizes that portability and privacy gaps increase with compression
pressure and stateful reuse depth, subject to model-family interactions.

### 15.7 External-view compilation and catalog-selection cost

For a staged external compiler, total cycle cost must include acquisition,
compilation, validation, multi-store publication, and recovery:

\[
C_{\text{view},c}
=C_{\text{acquire},c}
+C_{\text{classify/search},c}
+C_{\text{compile},c}
+C_{\text{verify},c}
+C_{\text{publish},c}
+C_{\text{recover},c}.
\]

Evolve's falling teacher calls measure only part of the first term under exact
query repetition. A positive deployment result requires
relationship-weighted future savings to exceed the whole sum plus any quality
loss:

\[
\sum_r N_{r,\text{future}}
\left(C_{\text{teacher},r}^{0}-C_{\text{teacher},r}^{M}\right)
>
C_{\text{view},c}+C_{\text{quality-loss}}.
\]

These are accounting definitions; future counts and costs are measured, not
assumed. A logical pipeline-call counter is not a provider-work counter:
malformed-output retries, HTTP/429 retries, embeddings, local generations,
judges, tokens, and failed calls must appear as separate measured terms. This
matters for Evolve because its released `teacher_calls` omits those retries and
its latency condition performs two local generations where the baseline
performs one.

Skill systems add a distinct selection term. If all \(K\) skill descriptions
are placed in the selector prompt, measured prompt tokens satisfy

\[
T_{\text{flat}}(K)=T_0+\sum_{i=1}^{K}t_i,
\]

an exact linear sum in description tokens even if the model's latency is not
linear. A hierarchical router instead has

\[
C_{\text{hier}}(K)
=
\frac{C_{\text{build}}(K)}{R_{\text{reuse}}}
+C_{\text{route}}(K)
+k\bar C_{\text{inspect}}
+p_{\text{miss}}(K)C_{\text{recover}},
\]

where \(k\) is the bounded inspected set. The crossover is empirical: index
construction, drift refresh, misses, and governance can outweigh prompt
savings at small \(K\). Memento-Skills' paper-time runtime makes this test
necessary because compact injected skills do not imply compact selection
work.

Publication has another complete-state term. If candidate validity requires
vector root \(V\), metadata root \(D\), topic/file root \(F\), index root \(I\),
and source range \(S\), the serving state is the tuple
\((V,D,F,I,S,\text{manifest generation})\), not the success status of any one
write. Reliability is measured by all-or-old visibility, no-lost-event
recovery, and bounded stale reads under faults; multiplying independent
per-write success probabilities is forbidden unless independence is tested.

### 15.8 Local-maintenance radius and service frontier

For a memory graph/store with \(M\) addressable units, let a job touch a
selected region of size \(k\le M\). Complete local-job cost is:

\[
C_{\mathrm{local}}(k,M)
=
C_{\mathrm{select}}(M)
+C_{\mathrm{read}}(k)
+C_{\mathrm{transform}}(k)
+C_{\mathrm{boundary}}(k,M)
+C_{\mathrm{verify/publish}}(k),
\]

while \(C_{\mathrm{global}}(M)\) applies the transform and validation to the
whole state. The boundary term charges missed cross-region dependencies,
duplicate summaries, index repair, and delete propagation; it prevents
subgraph size alone from being treated as job cost.

For frozen future-query mix \(Q\), define:

\[
\Delta V(k)
=
\Delta U_Q(k)
-v_C C_{\mathrm{local}}(k,M)
-v_D D_{\mathrm{boundary}}(k,Q)
-v_S S_{\mathrm{stale}}(M-k,Q).
\]

The best maintenance radius \(k^*\) is empirical and can be global. The
agent-native systems comparison motivates the candidate that localized
maintenance often occupies a better utility–latency region, but it does not
establish monotonicity or one optimum across workloads.

The planned frontier reports at least:

```text
future-query utility and exact-evidence fidelity
construction and query p50/p95/p99 latency
provider/model calls and tokens
bytes/nodes/edges read, rewritten, and published
queue age and stale-region error
recovery, delete, and cross-boundary repair work
```

A system is Pareto-dominated only under the same information, model, hardware,
governance, and missing-run policy. A protocol-valid timeout, OOM, deadline
miss, cap violation, or method-produced absent probe is an observed
intent-to-treat `RESOURCE_FAILURE`: retain the randomized stream, charge work,
and score every affected probe contribution zero even when intentional
abstention would have matched the target; unaffected pre-failure probes retain
their frozen scores. Only a protocol-valid intentional `ABSTAIN/FORGET` under
`COMPLETE` may earn abstention credit. Partial-identification bounds are
restricted to declared unresolved lifecycle cost or shared missingness, never
method failure utility. Report a feasibility endpoint. Only shared protocol corruption that prevents
all paired methods from being scored can invalidate a block. Complete-case
analysis is sensitivity-only.

## 16. Lifecycle utility frontier

The PAPER-C primary estimand has no free query-family weights. For user \(u\),
cell \(c\), cycle \(t=1,\ldots,128\), and each of the exactly three frozen
probe blocks \(p\), first average \(s(q)\in[0,1]\) over the nonempty scheduled
query set \(Q_{uctp}\), then average the resulting \(3\times128\) block means
equally:

\[
\mathrm{AUC}(u,c)=\frac{1}{3\cdot128}
\sum_{p=1}^{3}\sum_{t=1}^{128}
\frac{1}{|Q_{uctp}|}\sum_{q\in Q_{uctp}}s(q).
\]

Thus one absolute “utility point” means \(0.01\) on this normalized scale.
A separately labeled deployment-weighted or physical-time-weighted utility may
be reported only as a secondary estimand with weights, intervals, and horizon
normalization frozen at G2; it cannot alter PAPER-C decisions.

Define gross priced resource consumption, without subtracting reuse:

\[
C_{\mathrm{gross}}=\sum_r p_rx_r.
\]

Reuse already changes \(U_c\); subtracting a separate reuse value would double
count it. A normalized optional lifetime objective is:

\[
\mathcal V =
v_U U_{\text{future}}
-C_{\text{gross}}
-v_F F_{\text{protected}}
-v_H H_{\text{false-memory}}
-v_G G_{\text{governance}}
-v_R R_{\text{failure}}.
\]

Here \(x_r\) is gross consumed native resource \(r\), while \(p_r\) is its
price from a source-, date-, currency-, and amortization-bound manifest frozen
at G2. It is distinct from the service-capacity symbol \(C_r\). The complete
native resource vector is always reported. If any coefficient is absent or
indefensible, the scalar efficiency branch is ineligible and only constrained
Pareto comparisons are confirmatory.

Compare Pareto frontiers instead of collapsing everything into a score when
stakeholder exchange rates are not defensible.

Required frontier axes:

```text
utility/retention/acquisition
wake latency
sleep compute
total bytes
peak residency and memory-time AUC
bytes moved
delete/rollback completeness
```

A method dominates only when it is no worse on all frozen axes and better on
at least one within uncertainty.

## 17. Scaling experiment grid

### 17.1 Axes

Each confirmatory scale family uses at least eight ordered points where
feasible:

```text
association/event count
wake--sleep cycles
raw/external bytes
live context tokens
latent slots
active/total adapter parameters
adapter rank, module count, catalog residency, router error, merge width
added parameter bits and numerical precision
external-call rate and failure probability
sleep FLOPs
arrival/backlog load
temporal-clock count and job coalescing
local-maintenance radius / touched-state fraction
volatility/delete rate
users/tenants
```

### 17.2 Holdouts

At least two held-out scales per fitted axis, final workload-family holdouts,
and seed/user-stream replication are mandatory for an A100-regime claim.
Claim scope then determines additional prospective holdouts:

- a cross-query-type claim requires a powered query-type holdout;
- a cross-model/architecture claim requires a powered model/architecture
  holdout;
- a cross-hardware/system claim requires a powered second hardware/fleet
  holdout.

An absent higher-level holdout narrows only that scope; it does not invalidate
an otherwise qualified A100-regime relation.

### 17.3 Candidate models

```text
constant/null
linear/log-linear
power
exponential saturation
logistic knee
min-bottleneck
segmented regression
interaction surface
nonparametric monotone
```

Model selection uses nested cross-validation on fit data. Final holdouts are
opened once. Prediction intervals, parameter uncertainty, residuals, and
extrapolation range are published.

### 17.4 Claim-down rules

Call the result:

- **A100-regime scaling law** only if the four-axis frozen admission rule,
  A100 configuration holdouts, and workload-family holdouts pass;
- **cross-hardware/system scaling law** only if a prospectively mandatory,
  powered second-hardware holdout also passes;
- **regime-specific relation** if it holds only for a named family;
- **empirical curve** if fit is descriptive but not predictive;
- **null/narrowed** if alternatives are indistinguishable or precision is
  insufficient.

No exponent is silently reinterpreted after holdout failure.

## 18. Biological-theory transfer boundary

The audited bounded-synapse literature offers design intuition:

- finite mutable state is overwritten;
- multiple timescales can trade initial strength for lifetime;
- bidirectional fast/slow coupling may outperform one-way copying;
- additional internal state and precision are real capacity;
- biased updates can destroy balanced-model gains.

It does not authorize:

- mapping one synapse to one transformer parameter or memory row;
- calling \(N/\log N\) transformer factual capacity;
- applying finite-state Markov bounds to external text/graph stores;
- treating ideal-observer signal-to-noise as agent answer accuracy.

The present program uses those papers to choose variables and countermodels,
not to borrow their exponents.

## 19. Preregistered hypothesis map

| ID | Candidate relation | Required negative control |
|---|---|---|
| `H-STC-001` | future-task distortion changes the best representation | unchanged query distribution |
| `H-STC-002` | reuse/volatility define conditional promotion regions | non-single-crossing curves |
| `H-STC-003` | switching cost creates conditional hysteresis | one-threshold policy |
| `H-STC-004` | in at least one of exactly seven cells, a pre-TEST diagnosed binding capacity limits utility | paired \(+25\%\) binding/nonbinding interventions with frozen \(0.01/\pm0.005\) margins |
| `H-STC-005` | recovery reserve can have an interior optimum | monotone archive fixtures |
| `H-STC-006` | recoverability proxy improves intervention timing | miscalibrated and fixed-cadence controls |
| `H-STC-007` | acquisition can fail before retention | all four retention/acquisition boundary combinations |

For `H-STC-004`, CALIBRATION diagnoses “binding” from frozen utilization,
queue pressure, cap-hit frequency, and shadow-price criteria without TEST
utility. In exactly seven resource-targeted cells, separate \(+25\%\)
binding/nonbinding interventions test a binding-effect LCB above \(0.01\) and a
nonbinding simultaneous two-sided 95% interval inside \(\pm0.005\). Both
signed nonbinding boundaries use the same centered max-stat family, with a
boundary-null simulation controlling any false equivalence declaration at
0.05. At least one cell must
pass the frozen 21-margin family (seven binding margins plus fourteen
nonbinding TOST boundaries); all eligible binding-effect UCBs below \(0.01\)
falsify the global claim. At least four cells must be CALIBRATION-eligible;
otherwise the global result is `INCONCLUSIVE_ANTECEDENT` regardless of a
single apparent TEST pass. Eligibility and labels cannot change after TEST.

## 20. Exploratory program hypotheses; novelty not established

These hypotheses combine known ideas with the STC lifecycle. Routing/gating,
multi-version atomic state, and background merge/compaction have strong
adjacent prior art. This section claims neither invention nor priority for
those primitives. Novelty, if any, requires an adjacent-literature audit and an
element-by-element claim chart for the STC-specific integration and matched
benchmark.

### 20.1 Teacher-corruption law

Does error grow with recursive teacher depth faster for generated replay than
for canonical-anchor hybrids? Candidate outcome is a cycle-dependent
corruption curve, not assumed exponential.

### 20.2 Stage-count crossover

At fixed total bytes and compute, does adding fast/medium/slow stages help only
until copy error, metadata, and read amplification dominate?

### 20.3 Governance shadow price

Can correction/delete probability be converted into an empirical destination
cost that predicts external-versus-parametric routing?

### 20.4 Queue-before-storage failure

Under bursty novelty, does freshness/backlog violate its SLO before physical
storage fills? If so, sleep service—not memory bytes—is the first capacity
knee.

### 20.5 Plasticity-first alarm

Does a joint acquisition/free-capacity/representation/backlog signal predict
benefit from recycle or rebuild earlier than retention canaries?

### 20.6 Harness-specialisation crossover

As task heterogeneity and drift rise, when does a task-specific memory program
outperform a universal harness after charging repeated ingestion, validation,
migration, sandboxing, and rollback? Does a universal governance kernel plus
specialised write/read modules dominate both extremes, or merely move the
capacity problem into program/version count? Use RHO-style full-harness
rewrites as the high-flexibility arm, but replace correlated self-preference
with an additional independent promotion gate and estimate chosen-candidate
regret over repeated seeds.

### 20.7 Diagnostic-probe value curve

Does marginal held-out failure detection per probe saturate with \(J\), and
does typed coverage move the knee more efficiently than adding same-type
questions? Compare fixed, real-time generated, committed, oracle, and no-probe
arms under strict generator/evaluation separation.

### 20.8 Control-program version capacity

When specialist harnesses proliferate, does program count, migration debt, or
validation throughput become the binding capacity before memory bytes? The
total-state fit must count code, schemas, dependencies, test fixtures,
checkpoints, retained compatible memory versions, source trajectories,
diagnoses, candidate tools, backups, and run logs.

### 20.9 Delivery-compilation crossover

Across retrieved-context length, executor size, and reuse, where does a learned
text/latent compiler's complete-path latency and lifecycle cost cross direct
injection? A fitted relation that predicts executor-only latency but not total
latency is insufficient.

### 20.10 Repeated-key/semantic-amortization gap

How does \(h_{\text{exact}}-h_r\) scale for unseen paraphrase, composition,
correction, and distribution shift as compiler capacity, source diversity,
and category granularity change? A cache-like exact benefit that does not
predict the relationship-weighted deployment mix is not a memory scaling law.

### 20.11 Catalog-selection crossover

At what catalog size and reuse does a hierarchical skill router repay its
construction, refresh, miss recovery, and governance cost relative to flat
description prompting? Fit prompt tokens, latency, retrieval/execution
quality, active/archived artifact bytes, and drift jointly; a token-only
crossover is insufficient.

### 20.12 Coverage-dimension capacity law

Within a frozen workload family, does an estimated metric/representation
covering dimension predict held-out coverage-radius and value-error decay?
The fixed-codec cardinality arm tests the \(M^{-1/d}\) term while total bytes
grow; the fixed-active-representation-bit arm separately tests the joint
geometry--quantization trade-off
\(cM^{-1/d}+q(b_{\rm payload}(M))\). The hypothesis fails if
the appropriate frozen relation does not survive held-out scale/workload tests,
or if retrieval competition \(\delta_M\) reverses the expected gain. Failure
narrows the relation; it does not refute the conditional Memento 2 theorem.

### 20.13 Representation-refresh amplification

Under matched memory utility and deletion guarantees, does a selective payload
rewrite with generation-atomic key refresh have lifetime index work
proportional to the affected set? Under the explicitly restricted
single-current-encoder/no-federation contract, does repeatedly changing an
incompatible shared encoder make the affected set the whole store and become
quadratic in sleep cycles for linear store growth?

The current G4 conformance study crosses exactly five characterization
arms—fixed-encoder local rewrites,
old-encoder federation, query translation, dual-index migration, and full
encoder replacement—with memory growth and sleep cadence. Adapter-versioned
encoders may be an exploratory implementation inside the federation or query-
translation family. None of these curves is confirmatory without a new
pre-TEST amendment that adds power, a fitted estimand/model, prediction
tolerance, uncertainty, and held-out decision. Measure affected encoded item-operations,
resident encoder/index bytes, query fan-out/calibration, bytes duplicated,
publish lag, \(I_\tau\), retrieval recall, stale hits, deletion fan-out, and
crash recovery. The hypothesis fails if measured work does not scale with the
declared affected population after caching/batching, or if the alternative
placement is omitted from its native-resource ledger.

### 20.14 Bounded-state/lifetime-work crossover

For bounded reconstructive memory, at what novelty, volatility, pause, and
horizon does event-triggered rewriting dominate full per-turn rewriting after
charging verification, stale-state loss, worker cancellation, and later-wake
wait? A constant output-token curve alone is not evidence for the crossover.

### 20.15 Usable-capacity release

Under matched parameter, gradient-token, and external-call budgets, does
offloading selected factual targets improve fresh non-factual acquisition or
plasticity? If only factual leakage decreases while acquisition is unchanged,
the “freed reasoning capacity” mechanism is rejected in that regime even when
cascade factuality improves.

### 20.16 Parametric-routing knee

Under a fixed total adapter-parameter budget, where do low-rank module
isolation gains cross router error, merge interference, catalog selection, and
residency cost? Fit oracle and deployable router surfaces separately. The
confirmatory endpoint is the aggregate oracle--deployment gap; component
attribution is the conditional extension `X-STC-Q`. It requires a pre-TEST
amendment that freezes the \(2\times2\times2\) factorial, power, thresholds,
and multiplicity at G2. If promoted, among every CALIBRATION-qualified router,
a simultaneous adjusted UCB for modular-minus-single
\(\le\delta_{\mathrm{modular}}\) in every frozen cell falsifies the proposed
region, while no qualified router makes the extension `INCONCLUSIVE`. If not
promoted, the factorial remains exploratory and never becomes a “competent
router” post-hoc rescue.

### 20.17 Clock-count queue crossover

As session/day/week/month-style levels are added, does utility per active byte
first improve and then fall when construction calls, queue age, and repeated
abstraction distortion bind? Additional clocks remaining Pareto-improving over
burst, latency, cost, and distortion holdouts reject the crossover.

### 20.18 Internalization-completeness gap

Does direct-recall parity overstate destination equivalence? Fit
destination-specific curves for acquisition, temporal update, reference,
composition, implicit relevance, and boundary awareness. Stable rankings and
gaps across all six capabilities under matched evidence/resources reject the
candidate completeness gap.

### 20.19 Recovery-substrate floor

For rare or volatile evidence, is there a nonzero correction/late-query
utility floor after raw or reversible state is removed? Cross archive bytes,
event frequency, volatility, and abstraction depth while checking that no
equivalent trace survives in weights, summaries, backups, or generator state.
A bounded lossy view matching archive-backed correction, one-off recall, and
rollback rejects the floor in that regime.

## 21. Minimum credible scaling result

The minimum paper result is not a grand universal exponent. It is:

1. a reproducible lifetime stream;
2. matched external, parametric, and hybrid destinations;
3. at least one held-out predictive crossover or capacity knee;
4. retention and acquisition measured separately;
5. total-state and lifecycle-cost accounting;
6. a null/negative result path;
7. an executable trace from source and configuration to fit and figure;
8. no biological or deployment extrapolation beyond the tested regime.

That result would already improve on the current fragmented literature by
turning “sleep-time compute scales” into a falsifiable, cross-substrate
statement.
