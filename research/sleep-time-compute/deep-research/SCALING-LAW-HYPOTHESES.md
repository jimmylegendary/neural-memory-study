# Sleep-Time Compute Scaling-Law Hypotheses

- source freeze: 2026-08-05
- status: research agenda and preregistration candidates; **no universal deployed STC scaling law is currently established**
- strict separation: measured identity → analytical break-even → fit candidate → untested hypothesis

## 1. Why “accuracy vs sleep FLOPs” is not enough

Pretraining scaling usually relates loss to parameters, data, and compute under controlled training. Sleep-time learning adds path dependence and persistent state:

```text
previous state
+ accumulated wake evidence
+ admission policy
+ sleep operator/data
+ destination medium
+ validation/publish outcome
→ next state
→ distribution of later wake queries
```

A complete study therefore needs at least:

| Symbol | Unit | Meaning |
|---|---:|---|
| (F_s) | FLOP/job | sleep FLOPs including forward/backward/update |
| (F_v) | FLOP/job | validation/judge/canary work |
| (E_w) | evidence units or canonical bytes | admitted wake evidence |
| (R_v) | valid future reuses | later uses surviving drift/expiry |
| (B_r,B_e,B_l,B_p,B_a,B_v) | bytes | raw, external, latent, parametric, auxiliary, version/recovery state |
| (B_{move,ell}) | bytes | state-migration bytes across link (ell) |
| (d_{rec}) | generations | recursive-distillation depth |
| (N_c) | cycles | number of wake/sleep cycles |
| (Q_{age}) | seconds | sleep queue/job age |
| (L_{wake}) | seconds | TTFT/ITL/task latency impact |
| (U_{later}) | declared utility | later-wake quality/reliability |
| (D) | task distortion | loss from compaction/forgetting |

The high-level surface is only a notation:

\[
U_{later}=f(C_{pre},C_{post},C_{wake},C_{sleep},E_w,R_v,
B_{state},B_{move},d_{rec},N_c,\mathcal W,\mathcal M).
\]

Here (mathcal W) is workload/query process and (mathcal M) is memory medium/operator. This is not a fitted law.

## 2. Measured identities

These are accounting equalities or necessary dimensional conditions once every term is measured. They do not predict quality by themselves.

### ID-1 — total retained answer-bearing state

\[
B_{retained}=B_r+B_e+B_l+B_p+B_a+B_v.
\]

- (B_r): raw/canonical episodes and documents
- (B_e): external text/vector/graph/index state
- (B_l): latent, recurrent, KV, compressed execution state
- (B_p): adapters, experts, mutable weight deltas
- (B_a): optimizer, importance, router, generator, teacher-support state
- (B_v): lineage, predecessor, recovery, rollback metadata/state

Reporting only active summary bytes or adapter parameters is incomplete.

### ID-2 — peak incremental and physical state

\[
\begin{aligned}
B_{peak,state}(t)={}&B_{retained}(t)+B_{candidate}(t)+B_{pinned}(t)\\
&+B_{migration}(t)+B_{workspace}(t)+B_{replica}(t),
\end{aligned}
\]

\[
B_{peak,physical}(t)=B_{base,resident}(t)+B_{peak,state}(t)+B_{execution}(t).
\]

The physical maximum is evaluated at the same timestamp; independent component high-water marks must not be added.

### ID-3 — full replay accumulation

If sleep after task (t) replays (m_i) items from every task (i\le t):

\[
R_t=\sum_{i=1}^t m_i,
\qquad
W_T=\sum_{t=1}^T R_t=\sum_{i=1}^T m_i(T-i+1).
\]

For constant (m_i=m>0), lifetime replay examples are (mT(T+1)/2=\Theta(T^2)). Reducing a constant admission fraction changes the coefficient, not the exponent.

### ID-4 — byte-flow equilibrium

For physical store (j):

\[
\frac{dB_j}{dt}
=r_{admit,j}\bar b_{in,j}
-r_{expire,j}\bar b_{delete,j}
-r_{compact,j}\bar b_{reclaimed,j}.
\]

Moving active items to an undeclared archive is not reclamation. A bounded hot tier does not imply bounded lifetime storage.

### ID-5 — queue recursion

\[
Q_{t+1}=\max\{0,Q_t+A_t-S_t\}.
\]

Mean utilization below one is insufficient if P99 age, privacy deadline, or tenant starvation diverges.

### ID-6 — dimensionally typed offered load

\[
\sum_j\lambda_j\mathbb E[w_j^{gpu}]<m_{gpu},
\qquad
\sum_j\lambda_j\mathbb E[d_j^{io}]<C_{io},
\]

\[
\sum_j\lambda_j\mathbb E[a_j^{hbm}]<B_{hbm,cap}.
\]

GPU-s/job × job/s is compared with GPU count, byte/job × job/s with byte/s, and byte-s/job × job/s with bytes. HBM capacity is never substituted for HBM bandwidth.

### ID-7 — lifecycle work per useful reuse

For published item/artifact (i):

\[
C_{life,i}=C_{ingest}+C_{transform}+C_{train}+C_{validate}
+C_{publish}+C_{serve}+C_{maint}+C_{delete/rollback}.
\]

\[
\eta_i=\frac{\Delta U_{later,i}\,R_{valid,i}}{C_{life,i}}.
\]

The numerator/denominator require a declared utility/cost conversion; otherwise report a Pareto vector rather than a scalar.

## 3. Analytical break-even relations

These are derived under explicit restricted assumptions.

### BR-1 — reuse break-even

If compilation costs (C_{compile}), maintenance costs (C_{maint}), and each valid later use saves (Delta C_{use}>0):

\[
n^*=\frac{C_{compile}+C_{maint}}{\Delta C_{use}}.
\]

This assumes stationary identical uses and no quality difference.

With reuse intensity (lambda_i(t)), validity survival (S_i(t)), and discount (omega):

\[
R_i(H)=\int_0^H\lambda_i(t)S_i(t)e^{-\omega t}\,dt.
\]

If (lambda_i) is stationary Poisson and (S_i(t)=e^{-\mu_i t}):

\[
R_i(\infty)=\frac{\lambda_i}{\mu_i+\omega}.
\]

Promotion to destination (j) is admissible only if:

\[
R_i(H)\Delta U_{ij}
>
C_{compile,ij}+C_{serve,ij}+C_{govern,ij}+C_{risk,ij}.
\]

### BR-2 — sleep cadence under smooth staleness

For run cost (C_s), interval (	au), and delay penalty (a\tau^p):

\[
\bar C(\tau)=\frac{C_s}{\tau}+a\tau^p,
\qquad
\tau^*=\left(\frac{C_s}{ap}\right)^{1/(p+1)}.
\]

Linear-staleness special case with event rate (r_e), fixed run cost (F), per-event-time harm (h):

\[
J(\tau)=\frac{F}{\tau}+\frac{hr_e\tau}{2},
\qquad
\tau^*=\sqrt{\frac{2F}{hr_e}}.
\]

These fail under burst, hard deadlines, batch thresholds, multi-resource blocking, non-monotone consolidation damage, or capacity-triggered regime changes.

### BR-3 — local compute vs state movement

For link class (ell):

\[
C_{move}=\sum_\ell(B_\ell p_\ell+T_\ell v_{delay}).
\]

Move compute to data only when additional local execution/replication cost is less than moving source, model, optimizer, candidate, and validation state to a remote sleep cluster. A separate sleep cluster is not automatically optimal.

### BR-4 — finite-state lifetime bound

Let all incremental mutable state (M_n) contain at most (b_{mutable}) bits given frozen base/protocol (Theta_0). For a controlled conditionally independent answer family:

\[
\sum_{i=1}^{n}R_{Y_i\mid Q_i,\Theta_0}(D_i)
\le H(M_n\mid\Theta_0)
\le b_{mutable}.
\]

If novel independent evidence arrives at rate (lambda_e>0) with positive rate (ar R(D)):

\[
b_{mutable}(t)\ge \lambda_e t\,\bar R(D\mid\Theta_0).
\]

This is not a universal facts-per-parameter constant. It proves that fixed finite state cannot retain an open positive-entropy stream forever at fixed distortion.

## 4. Fit candidates

These functions are proposed to fit controlled sweeps. They become useful only if held-out workloads/hardware pass preregistered error and shape tests.

### FC-1 — diminishing sleep-compute return with evidence and valid reuse

One saturating candidate:

\[
\Delta U
=A\left[1-\exp\left(
-k F_s^{\alpha}E_{eff}^{\beta}R_v^{\gamma}
\right)\right]
-\Phi_{int}-\Phi_{stale}-\Phi_{risk}.
\]

Where:

- (E_{eff}) weights canonical evidence by confidence, diversity, and authorization.
- (R_v) counts valid later reuses, not total queries.
- (Phi_{int}) is interference/old-task loss.
- (Phi_{stale}) is validity drift.
- (Phi_{risk}) is poison, privacy, deletion, and rollback penalty.

Competitors: log-linear, power law with floor, segmented knee, generalized mean, nonparametric monotone surface. A single power law must be rejected if signs change or knees appear.

### FC-2 — resource-conditioned effective capacity knee

Define (K_{eff}) as the admitted association count where preregistered retention crosses (	au_{ret}):

\[
\operatorname{logit}p_{ret}(N_a)
=\operatorname{logit}\tau_{ret}
-\kappa\log(N_a/K_{eff}).
\]

A candidate resource surface:

\[
\frac{K_{eff}(f,r,m)}{K_0(f,m)}
\propto
\left(\frac{B_{store}}{B_{store,0}}\right)^\alpha
\left(\frac{T_{live}}{T_0}\right)^\beta
\left(\frac{F_s}{F_{s,0}}\right)^\gamma
\left(\frac{b_{repr}}{b_{repr,0}}\right)^\nu.
\]

Required competitors: min-bottleneck, Cobb–Douglas, generalized mean, segmented knee, interaction surface, monotone nonparametric fit. Separate knees are estimated for physical storage, behavioral reachability, retrieval, plasticity, governance, sleep service, and hot-state residency.

### FC-3 — valid-reuse distribution and compilation yield

If candidate memories have heavy-tailed valid reuse (R_v), fit:

\[
P(R_v\ge r)\propto r^{-\zeta}
\]

only after tail diagnostics. The economic yield of promotion budget (B_p) is:

\[
Y(B_p)=\sum_{i\in\operatorname{top}(B_p)}
\max\{0,R_{v,i}\Delta U_i-C_{life,i}\}.
\]

Hypothesis: a small high-reuse subset may capture most compilation value, favoring hybrid promotion. Falsifier: reuse is flat/unstable or prediction error destroys top-k advantage.

### FC-4 — recursive-distillation depth

Let (d_{rec}) count how many generations separate training examples from canonical wake evidence. Fit separate common and rare-tail distortions:

\[
D_{common}(d)=D_{0,c}+a_c d^{\delta_c},
\qquad
D_{tail}(d)=D_{0,t}+a_t d^{\delta_t}.
\]

Alternative: segmented collapse knee or exponential survival. The Nature collapse result motivates the factor; it does not determine its form for STC. Include an anchored-real-data fraction (q_{real}) and interaction (d\times q_{real}), informed by `SRC-STC-0069`.

### FC-5 — state-migration and fleet crossover

Fit later-wake lifecycle cost under four fleet placements:

```text
shared priority fleet
time-partitioned fleet
separate wake/sleep fleet
shard-local hybrid
```

Inputs: source shard bytes, model residency bytes, delta bytes, batch size, interconnect bandwidth, privacy locality, wake load, sleep burstiness. Outputs: total work, state-migration bytes, P99 wake interference, sleep completion age, physical peak state, cost per useful reuse.

### FC-6 — compaction distortion under repeated cycles

For cycle (c), fit:

\[
D_{c+1}=g(D_c,\rho_c,q_{canon,c},F_{s,c},\mathcal W_c),
\]

where (ho_c) is compression/reclamation ratio and (q_{canon,c}) is retained canonical support. Compare one-shot compression with repeated summary-of-summary and source-anchored recomputation.

## 5. Untested hypotheses

Each is explicitly an untested hypothesis, not a conclusion.

### H-S1 — reuse-predictability law

At matched lifecycle state/cost, STC advantage grows with calibrated valid future reuse and query predictability, but saturates after the compiled artifact covers the dominant query modes.

- prediction: high-reuse stable domains show a clear crossover; one-off/unpredictable domains do not.
- falsifier: no interaction between reuse/predictability and STC gain.

### H-S2 — external-first promotion law

For mixed volatile facts and stable skills, external-first hybrid promotion dominates direct parametric writes on a Pareto frontier of utility, deletion, and cost.

- prediction: stable procedural items promote; volatile exact facts remain external.
- falsifier: direct parametric or external-only dominates across all preregistered workloads.

### H-S3 — behavioral capacity knee precedes physical capacity

In at least one high-interference stream, normal-query reachability or new-task plasticity falls below threshold while unused physical parameter/storage capacity remains.

- evidence anchor: continual facts and CL theory.
- falsifier: behavioral retention/plasticity tracks physical occupancy monotonically without earlier knee.

### H-S4 — source-anchored dream stability

Keeping canonical wake evidence and limiting recursive depth moves the rare-tail collapse knee to larger (N_c) than summary-of-summary or synthetic-only dreaming at matched bytes/FLOPs.

- falsifier: anchored data yields no tail/retention benefit or costs erase utility.

### H-S5 — non-monotone reversibility reserve

Recovery depth initially improves rollback and permits aggressive compaction, then harms storage, delete work, attack surface, and stale-version risk; therefore at least one workload has an interior optimum.

- falsifier: adequately ranged workloads remain monotone.

### H-S6 — sleep-service knee

As admitted evidence rate rises, sleep job age/freshness fails before raw compute utilization reaches nominal saturation because state affinity, validation, and multi-resource blocking dominate.

- falsifier: trace-driven queue stays within age SLA up to the simple utilization boundary.

### H-S7 — device locality crossover

For sufficiently large source-to-delta ratio and privacy locality, shard-local or storage-near preprocessing reduces total movement/energy enough to beat a centralized sleep cluster even with lower compute efficiency.

- falsifier: model/optimizer residency dominates so strongly that centralized accelerator always wins.

### H-S8 — trained memory-policy scaling

Scaling the memory writer/router policy and downstream RL data yields more later-task utility per trainable parameter than updating the main model for experience-driven agent tasks, until memory-policy capacity saturates.

- evidence anchors: ReasoningBank, MemoPilot, PAHF, Proactive Memory.
- falsifier: main-model updates consistently dominate at matched total work/state and governance.

## 6. Training/data axes required in the scaling sweep

| Axis | Levels |
|---|---|
| canonical support | full, coreset, none |
| replay | raw, latent, generated, mixed |
| synthetic depth | 0, 1, 2, 4, 8+ generations |
| operator | SFT, LoRA/PEFT, distillation, regularization, RL memory policy, external transform |
| admission | all, surprise, uncertainty, utility/reuse predicted, fixed reservoir |
| destination | text, vector, graph, latent, adapter, expert, base weights, hybrid |
| cadence | per event, post-task, fixed interval, queue/capacity pressure, learned |
| validation | none, same-distribution, time-split, secret holdout, poison/delete canary |
| cycle count | 1, 5, 20, 100, 500+ |

## 7. Minimum reporting table

Every data point must report:

```text
pre/post/wake/sleep/validation FLOPs
canonical/replay/generated tokens and bytes
all retained-state classes and physical peak bytes
state-migration bytes by link
foreground TTFT/ITL/task latency during sleep load
sleep queue age and completion/failure/retry
old/new/compositional/rare-tail utility
plasticity and prompted recoverability
valid reuse, volatility, correction rate
delete latency, rollback success, poison result
base model/version and artifact compatibility
recursive-distillation depth
```

## 8. What would count as a real STC scaling law

A publishable law must:

1. define units and state boundary exactly.
2. fit more than one model size, workload, memory medium, and hardware setting.
3. include strongest alternatives at matched total lifecycle cost.
4. predict held-out conditions, not merely interpolate training runs.
5. expose knees/regime changes rather than force a global power law.
6. survive contamination, recursive-depth, capacity, and repeated-cycle controls.
7. report falsification when the candidate form fails.

Until then, the honest output is a family of identities, restricted break-even equations, workload-conditioned empirical surfaces, and untested hypotheses—not a universal “more sleep FLOPs always wins” curve.
