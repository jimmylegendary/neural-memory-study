# Scientific Red-Team Audit: Theory, Benchmark, and Infrastructure

- Audit date: 2026-07-25
- Scope:
  [`SCALING-LAWS-THEORY-AGENDA.md`](SCALING-LAWS-THEORY-AGENDA.md),
  [`OPEN-QUESTIONS-ANSWERABILITY-MATRIX.md`](OPEN-QUESTIONS-ANSWERABILITY-MATRIX.md),
  [`BENCHMARK-EXPERIMENT-BLUEPRINT.md`](BENCHMARK-EXPERIMENT-BLUEPRINT.md),
  [`SYSTEM-INFRA-BLUEPRINT.md`](SYSTEM-INFRA-BLUEPRINT.md), and the executable
  [publication/provenance plan](../../docs/superpowers/plans/2026-07-25-sleep-time-compute-evidence-publication.md)
- Audit mode: independent, adversarial, pre-data review
- Status: findings incorporated; the independent document-level re-audit found
  no residual HIGH/CRITICAL contradiction. Executable implementation and
  artifact re-audits remain required at G2, G4, and G6.

## 1. Audit purpose

This audit asks whether the proposed theory, experiment, and system can support
their intended claims even if the results look favorable. It searches for
hidden side information, unit errors, nonidentified mechanisms, post-hoc escape
clauses, method-dependent missingness, and race conditions. A wording change is
not considered a resolution unless the corresponding estimand, experiment, or
state machine also changes.

## 2. Finding and resolution register

| ID | Severity | Blocking failure | Incorporated resolution | Verification before G2/G4 |
|---|---|---|---|---|
| `SRT-001` | HIGH | The finite-state converse charged mutable memory but omitted the frozen base as side information, making the bound overstrong or physically ambiguous. | Define frozen public side information \(\Theta_0\), finite-precision incremental mutable state \(M_n\), and \(\sum_iR_{Y_i\mid Q_i,\Theta_0}(D_i)\le I(Y_{1:n};M_n\mid Q_{1:n},\Theta_0)\le H(M_n\mid\Theta_0)\le b_{\mathrm{mutable}}\) bits. Charge any base delta/checkpoint to \(M_n\), and report frozen-base bytes separately. | Opaque-payload generator, base digest, mutable-state inventory, and conditional-entropy calculation are frozen and independently reproduced. |
| `SRT-002` | HIGH | Varying prototype count at fixed total bytes changes bits per prototype, so the claimed \(M^{-1/d}\) cardinality exponent is confounded with quantization. | Split the study into a fixed-codec/bytes-per-prototype cardinality arm and a fixed-active-representation-bit joint geometry–codec arm \(cM^{-1/d}+q(b_{\rm payload}(M))\), with measured headers/index/metadata rather than nominal \(B/M\). | Both arms have separate manifests, at least eight fit points, two held-out sizes×two named held-out families as one equal-weight finite target, and measured codec distortion. |
| `SRT-003` | HIGH | Checking a deletion watermark before a separate manifest CAS leaves a validation-to-publication race; request pinning lacked safe reclamation and mid-read deletion semantics. | Use one linearizable `ServingHead` containing manifest generation/digest and authorization/deletion epochs. Deletes and authorization changes advance the same CAS word. Immediate deny precedes further reads/emission. GC computes reachability from current, pinned, candidate, migration, rollback, backup, legal-hold, and deletion-retention roots, then waits for acknowledged quiescence/hazard release or proven worker fencing; a lease timeout or generation threshold alone never authorizes free. | Model-check publication/deletion interleavings, shared-object reachability, reader cancellation/restart, and reclaim attempts before and after quiescence/fencing. |
| `SRT-004` | HIGH | “Lifetime utility AUC,” “utility point,” and lifecycle cost were underdefined; subtracting reuse from cost could double count the utility benefit. | Freeze equal-cycle/equal-probe \(s(q)\in[0,1]\) AUC and define one point as \(0.01\). Use gross non-overlapping native-resource cost only. Require a frozen price manifest for scalar efficiency; otherwise use constrained Pareto comparisons. | The same estimand and price-profile code is imported by power and final inference; native resource vectors are always emitted. |
| `SRT-005` | HIGH | Timeout, OOM, or method-produced missing probes could be excluded as complete-case missingness, favoring expensive or fragile methods; crash-as-abstention could even earn correctness. | Treat protocol-valid operational failures as intent-to-treat `RESOURCE_FAILURE`, retain randomized streams, charge work, and score every affected probe zero regardless of whether deliberate abstention would be correct. Only intentional `ABSTAIN/FORGET` under `COMPLETE` can earn abstention credit; only shared protocol corruption can invalidate a block. | Failure-rate noninferiority is required for dominance; deliberate-abstain-versus-OOM fixtures, worst-case MNAR, and complete-case sensitivity are published. |
| `SRT-006` | MED-HIGH | A whole-store index-refresh lower bound silently assumed one current encoder and no federation or translation. | Define an affected set \(A_c\) and charge \(\sum_c\sum_{i\in A_c}e_{ic}\). Whole-store refresh is conditional on single-current-encoder serving without federation/translation. Alternatives charge encoder/index residency, query fan-out/calibration, migration, and deletion fan-out. Current G4 runs five exact arms as `SYSTEMS_CHARACTERIZATION`/conformance evidence only; a slope or superiority claim needs a separately powered amendment. | Run fixed encoder, federation, translation, dual-index migration, and full replacement arms with the same quality/deletion constraints. Verify exact sizes/seeds/workloads and reject any report that promotes conformance curves to a scaling claim. |
| `SRT-007` | MED-HIGH | Tokens, bytes, work, latency, and value appeared in shared scalars or ledgers without conversions. | Separate \(T_{\mathrm{live}}\) tokens from \(B_{\mathrm{live}}\) bytes, retained/peak byte ledgers from cumulative work, and native resource vectors from optional priced objectives. | Dimensional analysis and schema unit checks reject mixed-unit addition. |
| `SRT-008` | MED-HIGH | Router, static cadence/proportions, baseline hyperparameters, early stopping, and comparator choice could use TEST information. | Split every workload-family×user×payload-seed unit into disjoint TRAIN, CALIBRATION, and TEST. Fit on TRAIN, select every rule and the \(R\)-blind comparator maps on CALIBRATION, freeze at G2, and execute TEST once. C1/C3 use simultaneous direct-component inference rather than a TEST max/argmax. | Split hashes, comparator-map digests, and search logs are gate inputs; tests reject seed, user, family, derivative overlap, or TEST-time comparator selection. |
| `SRT-009` | MED-HIGH | The additive router/merge/residency loss decomposition was not identified, and “competent router” was a post-hoc escape hatch. | PAPER-C confirms only the aggregate oracle–deployment gap and never attributes components. The \(2\times2\times2\) intervention and router-qualification/UCB rule belong to extension `X-STC-Q`; they become claim-bearing only through a pre-TEST amendment that binds their cells, thresholds, power, and multiplicity at G2. Without that amendment they remain explicitly exploratory and are not prerequisites or rescue paths for PAPER-C. | PAPER-C code exposes no component attribution. If `X-STC-Q` is promoted, publish all qualified routers and thresholds and bind its simultaneous UCB family before TEST; otherwise label every component result exploratory. |
| `SRT-010` | MEDIUM | Several falsifiers used “at least one,” “wrong order,” or “nonbinding” without cells, margins, or multiplicity; code/property tests could also masquerade as empirical H-STC outcomes. | PAPER-C1 now uses exactly nine cells and a one-point materiality margin. PAPER-C3 freezes two adjacent pairs, two corners, a one-point reversal margin, and sensitivity. H-STC-004 is a future extension design with exactly seven targeted cells, CALIBRATION-only eligibility, paired \(+25\%\) interventions, and frozen \(0.01/\pm0.005\) margins. Current G2 permits only derivation/property/preregistration statuses for H-STC-004/006/007; empirical decisions require a separately powered amendment and execution evidence. | PAPER-C uses global simultaneous intervals and exhaustive reasons. Pre-G2 theory artifacts reject empirical effect/interval/status fields; any promoted H-STC extension must bind manifests, power, multiplicity, receipts, validation, and post-TEST decisions. |
| `SRT-011` | MEDIUM | Capacity accounting omitted candidates, staging, reader-pinned predecessors, dual indexes, workspaces, temporary replicas, and physical duplication of the frozen base across fleets. | Separate \(B_{\mathrm{retained}}\) from incremental \(B_{\mathrm{peak,state}}(t)=B_{\mathrm{retained}}+B_{\mathrm{candidate}}+B_{\mathrm{pinned}}+B_{\mathrm{migration}}+B_{\mathrm{workspace}}+B_{\mathrm{replica}}\). Give every term a cap, disjoint rule, and GC owner. For HBM/fleet placement report \(\max_t[B_{\mathrm{base,resident}}(t)+B_{\mathrm{peak,state}}(t)+B_{\mathrm{execution}}(t)]\), with components aligned at the maximizing timestamp; never sum independent component high-waters as an identity. | Peak sampling is calibrated against allocation telemetry; noncoincident-max fixtures pass; every base copy is charged; cumulative work remains a separate ledger. |
| `SRT-012` | MEDIUM | “Novel extensions” and database non-discovery risked claiming known routing, MVCC/atomic publication, and LSM compaction primitives. | Relabel the section “Exploratory program hypotheses; novelty not established.” Limit the contribution hypothesis to STC-specific integration and matched evidence. Add an element-by-element [systems primary-source audit](SYSTEMS-INFRA-PRIMARY-SOURCE-AUDIT.md). | No “first,” “novel,” or exclusivity language passes G6 without an adjacent-literature claim chart. |
| `SRT-013` | MEDIUM | Parametric mutual information was requested without an identifiable estimator; one checkpoint cannot determine it. | Retrain from a fixed \(\theta_0\) over independent opaque-payload seed ensembles, freeze the decoder/readout before TEST, use a Fano/block-error or preregistered variational lower bound, charge auxiliary carriers, and report estimator intervals. | Simulation coverage, decoder freeze, seed independence, and auxiliary-state accounting are independently audited. |

### 2.1 Second-pass statistical and execution findings

The second pass expanded the audit from conceptual validity to executable
authorization. “Incorporated” below means the governing design/plan now states
the contract; it does not mean G2/G4 has already run.

| ID | Severity | Blocking failure | Incorporated resolution | Required mutation/verification |
|---|---|---|---|---|
| `SRT-014` | HIGH | Separate claimwise intervals and operational checks did not control the probability of any false deployability claim; zero rare failures made Wald RF rows undefined. | Freeze one `PAPER-C-DEPLOYABILITY-GLOBAL-131` joint max-stat union: 104 centered continuous scientific margins plus 27 veto-only boundary-positive paired-discordance RF scores. Shared protocol missingness is an alpha-free pair-integrity/MNAR veto, not a zero-SE completion row. | Mixed-null and zero-event boundary simulation must keep any false `SUPPORTED` decision at most 0.05; all-zero/all-one RF bounds are finite; add/drop/sign/reorder mutations stale the family digest. |
| `SRT-015` | HIGH | Treating 16 workload labels or three seed families as IID clusters created pseudoreplication and an undefined generalization target. | Define 48 fixed workload×paired-seed finite-target blocks, retain each block once, resample independent users only within block, and report leave-one-seed-family sensitivity. | Duplicating/splitting a label is invariant; cycles/probes and fixed seed families cannot increase \(n_{\rm eff}\). |
| `SRT-016` | HIGH | C4 power originally injected incompatible derived effects and later covered only nine schedule-invariant main-effect locations. | Generate 27 feasible underlying 2×2 DGPs: 18 regime×schedule simple-effect locations and nine regime main-effect locations; algebraically recompute all simple/main/interaction effects. | A simple-only `O_inline=.03, O_deferred=0` scenario must achieve composite power; inconsistent derived rows fail. |
| `SRT-017` | HIGH | Point covariance, pointwise 95% bounds, or scanning pilot sizes could underpower the 131-way Boolean decision; covariance alone does not identify finite-sample tails. | Freeze before pilot a hierarchy-aware full covariance-and-cluster-DGP outer-set builder with bounded supports, shrinkage, PSD plus nonshrinking inflation, skew/rare-event/copula stresses, nomination and disjoint selected-size verification. Allocate `alpha_Sigma=.01` and `alpha_MC=.01`. | Full-object coverage is at least 0.99; candidate×stress false authorization is controlled; same covariance/different tails can increase selected \(n\). |
| `SRT-018` | HIGH | PAPER-C budget and run order could be outcome-adapted or confounded by node/rank/time/thermal/cache state. | Freeze non-borrowable primary/clean quotas and native/carbon caps; use complete-block/Williams schedules whose assignments, order seed/digest, and balance tolerances flow G2→G4→receipt. | Hybrid-last, method×rank, thermal/cache imbalance, order drift, stale reservations, or aliasing retries fail before inference. |
| `SRT-019` | HIGH | Scaling mixed absolute family difficulty with resource effects and left log/reference units implicit. | Define \(K_0(f,m)=K_{\rm eff}(f,r_0,m)\), fit paired \(\ln(K/K_0)\) with \(g(r_0)=0\), no free intercept, and propagate the charged `normalization_only` anchor covariance. | Log-base/reference switches, duplicate anchors, free intercepts, and absolute held-out-family predictions stale the estimator. |
| `SRT-020` | HIGH | An 11-column D-optimal selector and repeated logical aliases overstated information relative to the actual anchored model. | Use the exact 10-column no-intercept main+pairwise design; collapse logical→physical aliases or use equivalent GLS, giving factor-1 aliases total weight one and existing tuples zero incremental information. | Replicating an alias 1→4 times leaves selection/determinant unchanged; rank is exactly 10. |
| `SRT-021` | HIGH | Named scaling competitors, slice effects, and a visual crossing contour allowed multiple mathematical implementations and post-fit favorable claims. | Bind exact formulas/domains/penalties/optimizer/PI construction for seven competitors; freeze nested-CV loss, numeric grids, one-SE/complexity tie rule and stability threshold. Axis/interaction claims use ordered equal-weight finite-lattice edges/faces. Phase boundaries are exploratory only. | Golden formula fits, near ties, one-family reversal, sign-changing slices, formula-digest drift, and post-fit contour promotion fail. |
| `SRT-022` | HIGH | Scaling validation pooled unequal role counts, ignored first-stage knee noise, and used unattainable/zero-SE PI-coverage bounds. | Freeze target `future_estimated_knee_fixed_panel_v1`; include target/anchor noise. Require separate family-only, resource-only, and double-holdout estimands, weighted by named family then unique physical config. Use a boundary-score statistic, all-success attainability check, and \(n_{\rm scaling}\le512\) grid. | Repeating configurations for the same users cannot improve coverage; logical aliases/family-label splits are invariant; zero surface error with noisy knees worsens RMSE/coverage. |
| `SRT-023` | HIGH | Signed log-radius error could cancel catastrophic over/underprediction, and four-cell PI coverage lacked a valid PSU. | Use nonnegative equal-weight log-radius RMSE. Within each named family, pair the same user across two held-out sizes; combine the two fixed-family estimates with one-half weights. Use a null-score statistic with positive boundary variance in `COVERAGE-GLOBAL-48`. | Ratios `{0.1,10}` cannot cancel; permuting one size's user IDs fails; all-covered/all-uncovered samples yield finite decisions. |
| `SRT-024` | HIGH | Coverage bit budgets could be chosen from TEST performance or omit infeasible large-\(M\) rows. | Dry-run the full codec/schema on the fixed candidate grid, retain only budgets feasible for every \(M\)/policy under native caps, and freeze the three smallest before power. | Negative payload, missing large-\(M\) row, larger favorable budget, holdout leakage, or arm pooling fails G2-CAP. |
| `SRT-025` | HIGH | Scaling/coverage/information shared only favorable CAL point estimates and independent budget ledgers. | Freeze a separate full-object capacity nuisance set with `alpha_Sigma_cap=.01`, disjoint verification `alpha_MC=.01`, one sponsor envelope, componentwise native/carbon caps, and one hash-chained actual ledger plus atomic head. | Builder selection after CAL, covariance-only tails, track borrowing, stale/forked ledger views, and counter omissions fail closed. |
| `SRT-026` | HIGH | Fano calibration conflated a population boundary with an inferential LCB and tested only homogeneous BSCs. | Freeze population `p_boundary=0.369127748985` satisfying \(1-h_2(p)=.05\); require \(\Pr(LB>.05)\le.05\) uniformly over heterogeneous sparse-signal/beta-binomial nulls. Charge every answer-bearing seed/log/router sidecar. | Homogeneous, sparse near-perfect+chance, correlated, no-signal, and hidden-carrier fixtures are mandatory; the boundary digest is \(n\)-invariant. |
| `SRT-027` | HIGH | One combined G4 snapshot and clean-rerun/independent pooling could alter core conclusions or bypass missing capacity evidence. | Use distinct core and capacity snapshots. Validated primary data alone set PAPER-C status; clean and independent analyses test reproducibility only. G5 binds all core/capacity preregistrations, receipts, stage validations, provisional/unlock artifacts, contrast families, budgets, canonical ledger/head, and primary/independent summaries. | Swap primary/clean pass/fail, omit one capacity stage, or force independent disagreement: no run may rescue primary; the affected capacity annex becomes typed failed/omitted. |
| `SRT-028` | HIGH | C2/C4 predicates admitted point-estimate savings, weak equivalence, or an incorrect falsification complement. | C2 requires simultaneous LCBs over benefit or utility-noninferiority+cost branches. C4 requires an operator `O-.02` LCB, simultaneous two-sided 95% identity/interaction equivalence, and deferred cost; falsification requires every eligible operator UCB below .02. | Boundary, equivalence-overlap, submaterial, and narrowed outcomes have exhaustive non-overlapping reason codes. |
| `SRT-029` | HIGH | A finite logistic coefficient could masquerade as a capacity knee without a threshold bracket or shape validity. | Require both exact/temporal component brackets, simultaneous CAL bounds, monotonic/single-crossing/residual-envelope checks, and agreement with a shape-constrained sensitivity within 0.10; derive the min-knee interval without selecting the observed argmin. | Gompertz, two-knee, flat, separation, tie/order-swap, and multiple-crossing mutations can return `KNEE_UNIDENTIFIED`. |
| `SRT-030` | HIGH | Capacity shard ownership hashing left launch order and hardware/time balance unenforced. | Bind per-track schedule assignments/tolerances/order digest through G2-CAP, capacity G4, shard manifests, canonical ledger, and receipts. Hashing assigns ownership only; each rank preserves its canonical subsequence. | Monotone load/\(M\), substrate-in-one-wave, method×rank, thermal/cache imbalance, order drift fail; file-row permutation preserving the schedule is invariant. |
| `SRT-031` | HIGH | Post-G5 manuscript filling was a prose-only instruction: a manual edit could leave `PENDING_G5`, soften a caveat, break EN/KO parity, or build different live sources while G6 hashed only the PDFs. | Add one closed `publication fill-results` command that atomically fills paired EN/KO slots from G5-bound terminal claim revisions and canonical `ResultBlock`s. Emit bibliography, live-source inventory, trace, and parity reports. G6 independently rehashes and binds the immutable G0 lineage, fill/adjudication reports, result blocks, live roots/inventories, manuscript links, parity map/report, traces, and PDFs. | Missing/duplicate/bypass CLI arguments, one-language partial updates, softened/null caveats, unregistered numbers, stale constituent digests, G0 mutation, extra live sources, report drift, and PDF/source mismatch all fail closed. |

## 3. Explicit non-finding

The full-replay identity was not rejected. If every task \(t\) triggers one
sleep job and replays a fixed \(m>0\) examples from every task seen so far,

\[
\sum_{t=1}^{T} mt = \frac{mT(T+1)}{2}=\Theta(T^2).
\]

This is correct under those stated assumptions. It must not be generalized to
bounded windows, sublinear admission, event-triggered sleep, or replay policies
whose support per prior task changes with \(T\).

## 4. Cross-document consistency contract

The following terms now have one meaning throughout the program:

```text
frozen base / public side information: Θ0
incremental answer-bearing state:      M_n
incremental information bound (bits):   b_mutable
retained incremental state bytes:      B_retained
incremental memory-state maximum:       B_peak,state(t)
fleet/HBM physical maximum:             B_peak,physical(t)
resident frozen-base copies:            B_base,resident(t)
visible answer-time context:           T_live
resident answer-time state:            B_live
gross priced resource cost:            C_gross
method operational failure:            RESOURCE_FAILURE
atomic publication authority:          ServingHead
```

Any future document or code that reintroduces token/byte ambiguity, a net cost
with reuse subtraction, a separate deletion check and manifest CAS, complete-
case method failure, or post-TEST router competence reopens this audit.

## 5. Remaining independent checks

Before G2:

1. a statistician re-audits estimands, multiplicity, missingness, power, and
   threshold reason codes;
2. an information theorist checks the conditional rate–distortion and
   parametric-information estimators;
3. an ANN/IR reviewer checks the affected-set and coverage–codec designs.

Before G4:

1. a distributed-systems reviewer checks `ServingHead`, deletion precedence,
   reader leases, rollback, and GC;
2. state-machine/property tests cover every publication/deletion interleaving;
3. retained and peak byte telemetry is reconciled with the resource ledger.

Passing this first audit means the known failure modes have explicit contracts.
It is not evidence that the hypotheses are true or that the design is already
publication-ready.
