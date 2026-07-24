# Sleep-Time Compute Research Program Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the evidence-gated theory, benchmark, systems validation, English paper, Korean companion, and non-blocking presentation handoff required to complete the approved sleep-time compute research design.

**Architecture:** A self-contained research package under `research/sleep-time-compute/` owns typed registries, immutable experiment manifests, a content-addressed artifact DAG, theory code, benchmark code, and generated reports. English, Korean, and PPTX artifacts consume the same frozen claim/result graph; none is an independent source of factual truth. Existing repository infrastructure—vendored primary papers, Veridraft, Pandoc/LuaLaTeX, A100 runbooks, and prior audit patterns—is reused without modifying the completed TTT claim bundle.

**Tech Stack:** Python 3.14, uv, Pydantic 2, pytest, Hypothesis, jsonschema, NumPy, SciPy, scikit-learn, statsmodels, PyArrow, Matplotlib, PyYAML, Veridraft 0.3.0, Pandoc 3.1.11, LuaLaTeX, TypeScript, PptxGenJS 4.0.1, LibreOffice, Poppler, Open XML SDK, veraPDF.

## Global Constraints

- The authoritative specification is `docs/superpowers/specs/2026-07-25-sleep-time-compute-research-design.md`.
- The existing `claims/bundle.json` and `.veridraft` artifact for `neural-memory-study-2026` are read-only inputs; the new program uses bundle ID `sleep-time-compute-2026`.
- The first paper has at most four main claims: `PAPER-C1` through `PAPER-C4`.
- The confirmatory design contains 297 logical lifetime runs: 162 deployable routing/medium cells, 27 oracle cells, and 108 schedule/operator factorial cells.
- Confirmatory methods are raw external, compressed external, one frozen reusable latent/KV implementation, user-local LoRA, tuned static mixture, deployable hybrid, and a non-deployable oracle.
- Reuse is \(R\in\{1,4,16\}\); active-capacity ratio is \(\phi\in\{1,1/4,1/16\}\); the default lifetime is 128 cycles.
- Observable policy inputs and oracle-only annotations are physically separate and fail closed on leakage.
- Privacy and deletion are hard constraints; direct identifiers and content-derived fingerprints never enter retained operational metadata.
- Headline quantitative claims require `ORIGINAL-MEASUREMENT` or `INDEPENDENT-REPLICATION` at reproduction strength r2 or above.
- Simulation and derivation are never labeled hardware measurement; A100 and H100 evidence remain separate.
- A changed upstream digest marks all downstream results, claims, figures, manuscripts, release tags, and presentation handoffs `STALE`.
- Existing untracked seminar previews and equation assets belong to the user and are never staged, edited, or removed by this program.
- All project changes use focused commits and push to the private `origin` only after their task-level tests pass, as required by `DECISIONS.md`.
- PPTX factual production is non-blocking and begins only from a frozen research handoff; tooling spikes may proceed earlier.
- Every master-plan command block starts at the repository root and does not
  inherit `cd` state from an earlier block. Subplans explicitly declare their
  package-relative command root.
- Every package CLI block therefore begins with
  `cd research/sleep-time-compute`; every presentation block begins with
  `cd presentation/sleep-time-compute`. A block that starts with `git` or
  document-only tooling intentionally remains at the repository root.
- **Commit arbitration:** when a master task ends with its own integration
  commit, execute the referenced subplan implementation/test steps but skip
  their local `git add`, `commit`, and `push` steps; the master commit is the
  only commit for that batch. When a master task has no integration commit, the
  referenced subplan commit remains authoritative. The only mandatory
  intermediate-identity exceptions are Evidence Task 14 Phase B.3
  (audit-control SHA), Phase B.5 (research release/tag), Phase B.6
  (presentation handoff), Presentation Task 0 Step 0 (human-provisioned
  reviewer trust root), and Presentation Task 0 Step 6 (signed capability
  outcome plus offline fixtures). Each is executed exactly once where the
  master explicitly invokes it. The two Presentation Task 0 commits must be
  distinct and the trust-root commit must precede every signed spike subject.
  This rule prevents both `nothing to commit` failures and accidental loss of
  identity-bearing release boundaries.
- Gate records are typed, content-addressed JSON at
  `research/sleep-time-compute/manifests/gates/G0.json` through `G8.json`.
  Each gate binds the exact predecessor-gate digests and fails closed if any
  predecessor, input manifest, signature, or attestation has changed.

---

## File and Interface Map

### Research package

```text
research/sleep-time-compute/
  .gitignore
  pyproject.toml
  uv.lock
  README.md
  src/stc_research/
    __init__.py
    ids.py
    models.py
    jsonl_store.py
    validate.py
    artifact_dag.py
    digest.py
    export_veridraft.py
    cli.py
    theory/
    benchmark/
    methods/
    design/
    analysis/
    systems/
    runtime/
    accelerator/
  tests/
  schemas/
    control/
    stc/
  registry/
    sources.jsonl
    evidence.jsonl
    claims.jsonl
    claim-history.jsonl
    questions.jsonl
    hypotheses.jsonl
    terminology.jsonl
    contradictions.jsonl
    negative-results.jsonl
    audit-findings.jsonl
    audit-adjudications.jsonl
    artifacts.jsonl
  cards/
  extracts/
  manifests/
    gates/
    candidates/
    releases/
    accelerator/runs/
  theory/
  benchmark/
  systems/
  reports/
  generated/
  publication/
    links/
    assets/
    handoff/
  results/
```

`stc_research.models` is the only definition site for evidence/control-plane
record field names. `stc_research.benchmark.schemas` owns event/query/run
records, and `stc_research.systems.schemas` owns lifecycle accounting records.
`stc_research.jsonl_store` performs canonical ordering and atomic writes.
`stc_research.validate` checks cross-record references and release invariants.
`stc_research.artifact_dag` computes digests and propagates `STALE`.
`stc_research.export_veridraft` maps admissible release claims into a separate
Veridraft bundle.

### Source and publication artifacts

```text
papers/sleep-time-compute/
  primary/
  text/
  checksums.sha256
claims/sleep-time-compute.bundle.json
paper-en/sleep-time-compute/
  main.tex
  sections/
  figures/
  tables/
  references.bib
  Makefile
study-kr/sleep-time-compute/
  BOOK.md
  front/
  chapters/
  back/
  build/
presentation/sleep-time-compute/
  package.json
  package-lock.json
  tsconfig.json
  src/
  tests/
  manifests/
  reports/
```

### Stable identifiers

| Object | Format | Example |
|---|---|---|
| Source | `SRC-STC-NNNN` | `SRC-STC-0001` |
| Evidence span | `EV-STC-NNNNN` | `EV-STC-00001` |
| General claim | `CL-STC-NNNN` | `CL-STC-0001` |
| Paper claim | fixed | `PAPER-C1` |
| Research question | `RQ1`–`RQ12` | `RQ3` |
| Hypothesis | fixed | `H-STC-001` |
| Artifact | `ART-STC-NNNN` | `ART-STC-0001` |
| Experiment | `EXP-STC-NNNN` | `EXP-STC-0001` |
| Result block | `RES-STC-NNNN` | `RES-STC-0001` |
| Figure/table | `FIG-STC-NNN` / `TAB-STC-NNN` | `FIG-STC-001` |
| English material block | `EN-SNNNN` | `EN-S0001` |
| Korean material block | `KO-SNNNN` | `KO-S0001` |
| Slide | `STC-SLIDE-NNN` | `STC-SLIDE-001` |

IDs are never recycled. Superseded records retain their ID and point to the
replacement.

## Subplan Boundaries

1. `2026-07-25-sleep-time-compute-evidence-publication.md` owns source intake,
   cards, evidence/claim/question registries, bibliography, manuscript
   traceability, and Korean parity.
2. `2026-07-25-sleep-time-compute-theory-benchmark.md` owns derivations,
   controlled generator, oracle, baselines, statistics, and scaling fits.
3. `2026-07-25-sleep-time-compute-systems-validation.md` owns lifecycle cost,
   queue/DSE, version/deletion/failure semantics, and accelerator runbooks.
4. `2026-07-25-sleep-time-compute-presentation.md` owns only the frozen handoff,
   rendering pipeline, editable deck, PDF, and visual/accessibility QA.

The common record schemas and artifact DAG are implemented once in Task 2 and
are consumed by every subplan.

---

### Task 1: Freeze the approved design and execution baseline

**Files:**
- Modify: `PLAN.md`
- Create: `docs/superpowers/plans/2026-07-25-sleep-time-compute-evidence-publication.md`
- Create: `docs/superpowers/plans/2026-07-25-sleep-time-compute-theory-benchmark.md`
- Create: `docs/superpowers/plans/2026-07-25-sleep-time-compute-systems-validation.md`
- Create: `docs/superpowers/plans/2026-07-25-sleep-time-compute-presentation.md`

**Interfaces:**
- Consumes: approved design and the three audited intake/design documents.
- Produces: one versioned execution baseline and four non-overlapping plans.

- [ ] **Step 1: Record the new program in the repository plan**

Add a new active phase to `PLAN.md`:

```markdown
- **P7 Sleep-Time Compute** 🟡 — approved design, evidence Wave 01, and
  execution plans under `docs/superpowers/`; research bundle ID
  `sleep-time-compute-2026`; 297-cell confirmatory design.
```

- [ ] **Step 2: Verify that only program-owned files are staged**

Run:

```bash
git add PLAN.md \
  docs/superpowers/specs/2026-07-25-sleep-time-compute-research-design.md \
  docs/superpowers/plans/2026-07-25-sleep-time-compute-*.md \
  docs/presentation/2026-07-25-pptx-toolchain-research.md \
  dossier/SLEEP-TIME-COMPUTE-PRE-RESEARCH.md \
  research/sleep-time-compute/EVIDENCE-INTAKE-WAVE-01.md
git diff --cached --name-only
```

Expected: the output contains only the explicitly listed design, plan, dossier,
intake, and `PLAN.md` paths; it contains no `seminar/_preview` or
`seminar/assets/eq` path.

- [ ] **Step 3: Run document syntax gates**

Run:

```bash
git diff --check --cached
for stc_doc in \
  docs/superpowers/specs/2026-07-25-sleep-time-compute-research-design.md \
  docs/superpowers/plans/2026-07-25-sleep-time-compute-*.md \
  docs/presentation/2026-07-25-pptx-toolchain-research.md \
  dossier/SLEEP-TIME-COMPUTE-PRE-RESEARCH.md \
  research/sleep-time-compute/EVIDENCE-INTAKE-WAVE-01.md
do
  pandoc -f gfm -t gfm "$stc_doc" -o /dev/null || exit 1
done
```

Expected: all commands exit 0 with no diagnostic.

- [ ] **Step 4: Commit and push the frozen baseline**

Run:

```bash
git commit -m "research: freeze sleep-time compute design and execution plans"
git push origin main
```

Expected: commit succeeds, push fast-forwards `origin/main`, and unrelated
untracked seminar assets remain untracked.

### Task 2: Build the shared research control plane

**Files:**
- Create: `research/sleep-time-compute/pyproject.toml`
- Create: `research/sleep-time-compute/uv.lock`
- Create: `research/sleep-time-compute/src/stc_research/{ids,models,jsonl_store,validate,digest,artifact_dag,export_veridraft,cli}.py`
- Create: `research/sleep-time-compute/schemas/control/*.schema.json`
- Create: `research/sleep-time-compute/tests/test_{ids,models,jsonl_store,validate,artifact_dag,export_veridraft}.py`
- Create: `research/sleep-time-compute/tests/test_{gate_engine,candidate_freeze,attestations,gate_release,release_assembly,presentation_handoff}.py`
- Create: `research/sleep-time-compute/registry/*.jsonl`

**Interfaces:**
- Consumes: identifier and claim contracts in the approved design.
- Produces: canonical record constructors, JSONL persistence, cross-reference
  validation, digest propagation, and Veridraft export used by all later tasks.

- [ ] **Step 1: Execute Tasks 1–6 and the early tooling phase of Evidence Task 14**

Implement the generic typed-gate engine and the candidate, release, attestation,
and presentation-handoff command surfaces now, against empty, adversarial, and
redistribution-safe fixtures. Evidence Task 14's real candidate freeze, release
assembly, and handoff export remain deferred until their scientific inputs
exist; no later audit/release/handoff task may invoke a control-plane command
from this batch that was not implemented and tested here. Domain commands owned
by later tasks are implemented and tested immediately before their first
runtime invocation.

When this subplan is executed from the master plan, execute the implementation
and verification steps of Evidence Tasks 1–6 and Task 14 Phase A but defer
their individual commit/push steps. Master Task 2 Step 3 is the sole integration
commit for this batch, so an exact run never ends with a duplicate
`nothing to commit` failure.

Run:

```bash
cd research/sleep-time-compute
uv sync --frozen
uv run pytest tests/test_ids.py tests/test_models.py \
  tests/test_jsonl_store.py tests/test_validate.py \
  tests/test_artifact_dag.py tests/test_export_veridraft.py \
  tests/test_gate_engine.py tests/test_candidate_freeze.py \
  tests/test_attestations.py tests/test_gate_release.py \
  tests/test_release_assembly.py \
  tests/test_presentation_handoff.py -q
uv run stc release verify \
  tests/fixtures/g8-release/publication/releases/stc-fixture-v1 \
  --offline --require-environment --require-sbom --require-reproduction
uv run stc presentation validate \
  tests/fixtures/g8-release/publication/handoff/stc-fixture-v1 \
  --offline
```

Expected: all control-plane tests pass. Wave 01 ingestion remains Task 3 so the
control plane can first be tested against empty and adversarial fixtures.

- [ ] **Step 2: Validate an empty but schema-complete registry**

Run:

```bash
cd research/sleep-time-compute
uv run stc validate --root .
```

Expected:

```text
PASS schema_inventory=complete cross_refs=0 duplicate_ids=0 boundary_errors=0 stale_release_nodes=0
```

- [ ] **Step 3: Commit and push the control plane**

Run:

```bash
git add research/sleep-time-compute claims/sleep-time-compute.bundle.json
git commit -m "research: add evidence control plane and artifact DAG"
git push origin HEAD
```

Expected: tests remain green from a clean checkout and the legacy claim bundle
is byte-identical.

### Task 3: Ingest and audit the chronological evidence atlas

**Files:**
- Modify: `research/sleep-time-compute/registry/*.jsonl`
- Create: `research/sleep-time-compute/cards/*.json`
- Create: `research/sleep-time-compute/cards/generated/*.md`
- Create: `research/sleep-time-compute/extracts/*.json`
- Create: `papers/sleep-time-compute/{primary,text}/`
- Create: `papers/sleep-time-compute/checksums.sha256`
- Create: `research/sleep-time-compute/reports/evidence-readiness.md`

**Interfaces:**
- Consumes: the pre-research dossier, Wave 01 intake, primary PDFs/HTML, and
  control-plane schemas.
- Produces: G1-ready primary-source cards and exact evidence anchors.

- [ ] **Step 1: Reconcile the dossier, bootstrap Wave 01, and freeze Wave 02 search**

Run:

```bash
cd research/sleep-time-compute
uv run stc ingest-dossier \
  --input ../../dossier/SLEEP-TIME-COMPUTE-PRE-RESEARCH.md
uv run stc ingest-wave01 --input EVIDENCE-INTAKE-WAVE-01.md
uv run stc validate --root .
```

Expected: dossier and Wave 01 canonical keys reconcile without duplicate works;
the Wave 01 report alone says `wave_sources=27`, `canonical_cards=27`,
`rendered_cards=27`; every card has a matching source record and no `A-ABS`
record supports a body-only claim. Then execute Evidence Task 8 to freeze the
search protocol before Wave 02 discovery and vendor the pinned sources.

- [ ] **Step 2: Execute the closest-neighbor full-read wave**

The required first wave contains the founding 2025 Sleep-time Compute paper,
Auto-Dreamer, OSL-MR, RecMem, TMEM,
Cartridges at Scale, Offline Recurrence, LatentMem, Doc-to-LoRA, Generative
Adapter, the OpenReview and arXiv v2 lineage of Language Models Need Sleep,
Agent Memory, TrustMem, and Retain or Consolidate.

Then execute Evidence Task 9 Step 6 for the explicitly requested chronology:
the 2019→2022 PLOS lineage, PAD and Nature sleep replay, DANN, MemGPT/Letta,
Mem0/Zep version lineages, and the attributed Google/DeepMind, Meta, and
Microsoft paper/product portfolios. Papers used for assertions must be FULL;
official product records are versioned A-HTML and remain behavior/existence
evidence only.

```bash
cd research/sleep-time-compute
uv run stc validate --root .
```

Expected: all 14 high-risk works and every manuscript-used paper in the
user-requested lineage are `FULL` and have version, checksum, exact
method/result/limitation anchors, code/data inspection status, non-claim
boundary, reviewer, and review date. Corporate attribution and
paper-versus-product version audits have no unresolved identity.

- [ ] **Step 3: Freeze the claim spine and pass G0/G1**

Execute Evidence Task 10 Steps 1–3 first, without running Step 4. Then execute
Phase A in the exact interleave Evidence Task 11 A.1–A.2, Task 12 A.1–A.2,
Task 11 A.3, Task 12 A.3, including scaffold tests, both placeholder builds,
snapshot-only offline verification, trace, and parity validation. Only then
execute Evidence Task 10 Step 4
exactly to freeze hypotheses/branch ownership and assemble/evaluate G0 and G1.
Freeze RQ1–RQ12, H-STC-001–005, PAPER-C1–C4 history, the publication-branch
manifest, the non-assertive paper/companion scaffold, and the final
search-log/stopping audit. After those exact steps, run this non-mutating
verification block:

```bash
cd research/sleep-time-compute
uv run stc gate verify-chain --through G1 --root manifests/gates
uv run stc validate --root . --gate G1
uv run stc branches validate manifests/publication-branches.json
uv run stc publication trace \
  --source ../../paper-en/sleep-time-compute/main.tex
uv run stc publication validate-parity publication/links/parity.jsonl
```

Expected: typed G0/G1 manifests pass, G1 embeds the exact G0,
search-protocol, chronological full-read-order, and evidence-readiness digests,
the search-protocol digest is unchanged, and no critical novelty threat or
lower-than-FULL direct-neighbor card remains. Evidence Tasks 11–12 Phase A has
also created buildable English/Korean claim-and-caveat skeletons containing
only explicitly non-assertive result placeholders. G0 binds the immutable
`publication/scaffolds/G0/manifest.json` snapshot and its snapshot-only
verification report rather than mutable live manuscript paths; Task 3's commit
preserves both the snapshot and live roots.

- [ ] **Step 4: Commit and push the evidence atlas**

Run:

```bash
git add papers/sleep-time-compute \
  paper-en/sleep-time-compute \
  study-kr/sleep-time-compute \
  research/sleep-time-compute
git commit -m "research: add sleep-time compute evidence atlas wave 02"
git push origin HEAD
```

Expected: no generated claim is accepted without at least one resolvable
evidence ID.

### Task 4: Implement and verify the theory and benchmark core

**Files:**
- Create: `research/sleep-time-compute/src/stc_research/theory/*.py`
- Create: `research/sleep-time-compute/src/stc_research/benchmark/*.py`
- Create: `research/sleep-time-compute/src/stc_research/methods/*.py`
- Create: `research/sleep-time-compute/src/stc_research/design/*.py`
- Create: `research/sleep-time-compute/src/stc_research/analysis/*.py`
- Create: `research/sleep-time-compute/tests/stc/{theory,benchmark,methods,design,analysis}/*.py`
- Create: `research/sleep-time-compute/configs/stc/*`
- Create: `research/sleep-time-compute/manifests/stc/*`
- Create: `research/sleep-time-compute/src/stc_research/gate_inputs_g3.py`
- Modify: `research/sleep-time-compute/src/stc_research/cli.py`
- Create: `research/sleep-time-compute/tests/stc/test_g3_gate.py`
- Modify: `research/sleep-time-compute/tests/stc/test_cli_contract.py`

**Interfaces:**
- Consumes: frozen hypotheses, observable/oracle schemas, and resource units.
- Produces: checked derivations, deterministic lifetime streams, exact oracle,
  method contracts, preregistration matrix, and statistical analysis.
- Owns and parser-tests the exact
  `gate inputs assemble G3 --root PATH --predecessor PATH --theory PATH
  --benchmark PATH --dag PATH --output PATH` domain adapter. It calls the
  generic Evidence Task 4 gate engine and cannot write a `GateRecord`.

- [ ] **Step 1: Build shared contracts and the pre-power scientific core**

Execute Task 1 of the systems-validation subplan first so
`LifecycleCostSummary` and `MethodRuntime` have one definition site. Then
execute Theory/Benchmark Tasks 1–10 to create schemas, theory, generator,
oracle, public adapters, methods, factorial operators, and the blocked
pre-power 297-cell skeleton.

- [ ] **Step 2: Build the runtime before the real pilot**

Execute Systems Tasks 2–7: versioned fabric, planes, queue/DSE,
deletion/failure semantics, lifecycle integration/general manifest runner, and
the hardware-free accelerator harness. Freeze
`manifests/systems/runtime-G2.json`, including code, container, base/tokenizer,
compiler, accounting, and runtime-contract digests.

- [ ] **Step 3: Run the blinded pilot and freeze G2**

Execute Theory/Benchmark Tasks 11–13 only after the runtime exists. The real
pilot flows through `runtime.coordinator`; it fixes `B_ref`, user count,
cost-profile digest, final 297-row manifest, analysis plan, 112 candidate
scaling cells, their frozen logical-to-physical execution map and complete
physical-child cohort index, conservative projected and hard actual
reference-run-equivalent ceilings, the global primary-comparison membership
table, and the G2 preregistration digest. No production result is read before
this freeze.

Run:

```bash
cd research/sleep-time-compute
uv run pytest tests/systems tests/runtime tests/accelerator tests/stc -q
uv run stc benchmark property-check \
  --config configs/stc/benchmark/controlled-128.yaml \
  --seeds 0:99 \
  --output manifests/stc/generator-property-report.json
uv run stc prereg validate \
  --manifest manifests/stc/preregistration.json \
  --fail-on-upstream-digest-change
uv run stc gate verify-chain --through G2 --root manifests/gates
```

Expected: theory unit/property tests pass; 100 generator seeds conserve events,
respect supersession/deletion, and leak no oracle field;
`confirmatory-summary.json` reports exactly 297 logical rows and
`297 * n` child references, with `n/16` children per workload family per
logical row. The scaling artifacts report 80 one-axis plus 32 joint-axis rows,
`96 * n` A100 fit and `16 * n` untouched A100 holdout child references, hence
`112 * n` total; `missing_B_ref=0`; projected and hard actual scaling budgets
are frozen. The typed G2 digest includes G1, runtime, model lock, hypotheses,
theory laws, multiplicity table, both child indexes, and every frozen
upstream/config revision. Digest-mutation tests must make validation fail.

- [ ] **Step 4: Generate the frozen G3 benchmark artifacts**

Implement and test the G3 domain-input adapter before invoking it. It binds the
canonical G2 predecessor, rendered theory manifest, frozen benchmark manifest,
and artifact-DAG digest; mutation of any one input must return typed `FAIL`.

Run:

```bash
cd research/sleep-time-compute
uv run pytest tests/stc/test_g3_gate.py tests/stc/test_cli_contract.py -q
uv run stc benchmark generate \
  --config configs/stc/benchmark/controlled-128.yaml \
  --output generated/stc/controlled
uv run stc theory render \
  --laws manifests/stc/theory-laws-G2.json \
  --output generated/theory
uv run stc benchmark freeze \
  --controlled generated/stc/controlled \
  --property-report manifests/stc/generator-property-report.json \
  --public-replay manifests/stc/public-replay.jsonl \
  --preregistration manifests/stc/preregistration.json \
  --output generated/stc/benchmark
uv run stc gate inputs assemble G3 \
  --root . \
  --predecessor manifests/gates/G2.json \
  --theory generated/theory/manifest.json \
  --benchmark generated/stc/benchmark/manifest.json \
  --dag registry/artifacts.jsonl \
  --output manifests/gate-inputs/G3.json
uv run stc gate evaluate G3 \
  --inputs manifests/gate-inputs/G3.json \
  --output manifests/gates/G3.json
uv run stc gate verify-chain --through G3 --root manifests/gates
uv run stc dag check
```

Expected: every equation has a derivation/unit-test digest; every benchmark
artifact has an input digest and `gate_status=PASS`; G3 binds G2 and both
generated manifests; no downstream node is `STALE`.

- [ ] **Step 5: Commit and push the theory/benchmark/runtime core**

Run:

```bash
git add research/sleep-time-compute
git commit -m "research: freeze lifetime benchmark runtime and preregistration"
git push origin HEAD
```

Expected: clean rerun regenerates identical controlled data and manifests.

### Task 5: Build the runtime, reproduce baselines, and run pre-G4 evidence

**Files:**
- Create: `research/sleep-time-compute/src/stc_research/methods/*.py`
- Create: `research/sleep-time-compute/src/stc_research/systems/*.py`
- Create: `research/sleep-time-compute/src/stc_research/runtime/*.py`
- Create: `research/sleep-time-compute/src/stc_research/accelerator/*.py`
- Create: `research/sleep-time-compute/tests/stc/methods/*.py`
- Create: `research/sleep-time-compute/tests/{systems,runtime,accelerator}/*.py`
- Create: `research/sleep-time-compute/configs/stc/methods/*.yaml`
- Create: `research/sleep-time-compute/manifests/stc/*.json`
- Create: `research/sleep-time-compute/results/{smoke,public-replay}/*`

**Interfaces:**
- Consumes: G2 preregistration, G3 benchmark, frozen baseline commits and
  resource caps.
- Produces: the lifecycle runtime, checked method adapters, smoke bundles, and
  deterministic public replay. It does not execute production confirmatory or
  scaling outcomes before G4.

- [ ] **Step 1: Reverify the frozen runtime and G2 digest**

Systems Tasks 1–7 and Theory Tasks 1–13 completed before this phase. Verify the
clean runtime and preregistration without changing either.

```bash
cd research/sleep-time-compute
uv run pytest tests/systems tests/runtime tests/accelerator -q
uv run stc prereg validate \
  --manifest manifests/stc/preregistration.json \
  --fail-on-upstream-digest-change
```

Expected: all local runtime and lifecycle tests pass; no confirmatory artifact
has been generated yet; the G2 digest matches.

- [ ] **Step 2: Verify baseline adapters against the runtime contract**

Run:

```bash
cd research/sleep-time-compute
uv run pytest tests/stc/methods -q
uv run stc methods check --config-dir configs/stc/methods
```

Expected: every method returns the same typed action/result interface, observes
the same information cutoff, and reports unused resources and feasibility.

- [ ] **Step 3: Run the local smoke matrix before confirmatory scale**

Run:

```bash
cd research/sleep-time-compute
uv run stc runtime run-matrix \
  --manifest manifests/stc/confirmatory.jsonl \
  --cohort-index manifests/stc/confirmatory-run-index.json \
  --preregistration manifests/stc/preregistration.json \
  --mode smoke \
  --output results/smoke
uv run stc results validate \
  --preregistration manifests/stc/preregistration.json \
  --manifest manifests/stc/confirmatory.jsonl \
  --cohort-index manifests/stc/confirmatory-run-index.json \
  --results results/smoke \
  --output manifests/stc/smoke-result-validation.json
```

Expected: one cell per method plus oracle and four factorial cells complete;
invalid deletion, future leakage, missing checksum, and over-budget fixtures
fail closed.

- [ ] **Step 4: Run the deterministic public-benchmark replay**

```bash
cd research/sleep-time-compute
uv run stc runtime run-matrix \
  --manifest manifests/stc/public-replay.jsonl \
  --preregistration manifests/stc/preregistration.json \
  --mode public-replay \
  --output results/public-replay
uv run stc results validate \
  --preregistration manifests/stc/preregistration.json \
  --manifest manifests/stc/public-replay.jsonl \
  --results results/public-replay \
  --output manifests/stc/public-replay-result-validation.json
```

Expected: at least one frozen public track completes with upstream,
transformation, order, sleep-boundary, withholding, license, and output
digests; it remains excluded from scaling-law fitting.

- [ ] **Step 5: Prove production execution is still closed**

```bash
cd research/sleep-time-compute
uv run stc prereg validate \
  --manifest manifests/stc/preregistration.json \
  --fail-on-upstream-digest-change
uv run stc gate verify-chain --through G3 --root manifests/gates
test ! -e manifests/stc/g4-confirmatory-execution.json
test ! -e results/confirmatory-primary
test ! -e results/scaling
```

Expected: G2/G3 remain valid and no production outcome exists. The only legal
producer of the G4 execution snapshot is Evidence Task 13 after typed G4
passes; the official primary, clean rerun, independent calculation, and
scaling track therefore cannot start in this task.

### Task 6: Complete systems validation and infrastructure outputs

**Files:**
- Create: `research/sleep-time-compute/src/stc_research/systems/*.py`
- Create: `research/sleep-time-compute/src/stc_research/runtime/*.py`
- Create: `research/sleep-time-compute/src/stc_research/accelerator/*.py`
- Create: `research/sleep-time-compute/tests/systems/*.py`
- Create: `research/sleep-time-compute/tests/runtime/*.py`
- Create: `research/sleep-time-compute/tests/accelerator/*.py`
- Create: `research/sleep-time-compute/systems/*.md`
- Create: `research/sleep-time-compute/manifests/systems/*.json`
- Create: `research/sleep-time-compute/manifests/a100-runbook-smoke.json`
- Create when target hardware succeeds:
  `research/sleep-time-compute/manifests/accelerator/runs/{a100-single-gpu-semantic,a100-single-gpu-resource-smoke,a100-single-node-contention,a100-two-node-gate,a100-confirmatory}.json`
- Modify: `research/sleep-time-compute/registry/artifacts.jsonl`
- Create: `research/sleep-time-compute/reports/g4-{tests.xml,systems-evidence.json,systems-audit.md}`
- Create: `research/sleep-time-compute/manifests/gates/G4.json`
- Create: `runbook/sleep-time-compute-a100.md`

**Interfaces:**
- Consumes: benchmark traces, method resource events, deletion lineage, and
  target hardware definitions.
- Produces: queue/DSE results, lifecycle accounting, failure-injection evidence,
  deletion audit, infrastructure blueprint, A100 commands and measurement
  manifests, and the typed G4 decision.

- [ ] **Step 1: Build the local G4 evidence inputs**

Tasks 1–7 were completed before the production run. Execute Systems Task 9
Step 1 exactly: run the full JUnit-emitting suite, lifecycle, DSE,
deletion/fault audit, fake accelerator, both bundle imports, and DAG check.
The fake accelerator remains `SYNTHETIC`.

Expected: stable queues meet the declared p99/deadline gates; unstable queues
are detected; stale CAS, worker death, poison, partial write, and pinned-read
deletion preserve the failure contract; no retained tombstone contains a
tenant/user/source/content-derived join key. Registered local bundle IDs are
exactly `g4-lifecycle`, `g4-dse`, `g4-deletion`, and `g4-fake`.

- [ ] **Step 2: Execute the staged A100 runbook when target hardware exists**

Execute Systems Task 8 Steps 1–4 exactly. Steps 1–3 cover
container/environment pinning, dual-node probes and reconciliation, the three
mandatory local gates, the two-node gate, calibrated CUPTI HBM and host
uncore-IMC DRAM counters, manifest-last eight-rank measurement, validation,
and DAG import on the reviewed A100 nodes. Step 4 must author the exact post-G4
production-matrix section and run its hardware-free fake-backend command
surface receipt, but must not execute those post-G4 production commands yet.

Expected: the fixed `a100-confirmatory` manifest is registered with
`measurement_status=ADMISSIBLE`, distinct GPU-HBM/host-DRAM provenance, raw
counter/calibration uncertainty, and exact rank/probe/environment digests. If
hardware or the required counter permissions are unavailable, record the
typed `BLOCKED-EXTERNAL` condition; synthetic or shape-derived bytes cannot
substitute. The four fixed prerequisite gate manifests are also registered and
their ordered digest/check closure is selected into `g4-inputs.json`; an
NFS-only prerequisite cannot support the measured branch.

- [ ] **Step 3: Build typed systems evidence, then evaluate G4**

Execute Systems Task 9 Steps 2–4. If Step 2 succeeded, use its exact measured
branch selecting both `g4-fake` and `a100-confirmatory`; otherwise use the
local branch selecting `g4-fake` and preserving the measured-hardware blocker.
The JSON report is generated before gate evaluation; Markdown is only a view.

Run:

```bash
cd research/sleep-time-compute
uv run stc gate verify-chain --through G4 --root manifests/gates
uv run stc validate --root . --gate G4
pandoc -f gfm -t gfm systems/INFRASTRUCTURE-BLUEPRINT.md -o /dev/null
pandoc -f gfm -t gfm ../../runbook/sleep-time-compute-a100.md -o /dev/null
```

Expected: canonical `manifests/gates/G4.json` binds G3 and the typed
`g4-systems-evidence.json`/input-index digests; reports distinguish
measurement, derivation, and trace simulation. The runbook contains
environment fingerprint, exact pre-G4 measurement and post-G4 production
commands, expected outputs, failure handling, and result-bundle checksums.

- [ ] **Step 4: Commit and push the runbook and G4 systems evidence**

This master task has no integration commit, so the global arbitration rule
retains the subplan boundaries. Execute Systems Task 8 Step 5 and Systems Task
9 Step 5 exactly, in that order, then publish both focused commits:

```bash
git push origin HEAD
```

Expected: the A100 runbook/small smoke manifest and the typed G4
systems-evidence/blueprint commit are both reachable from `origin`; no
confirmatory, clean-rerun, or scaling payload has been staged.

### Task 7: Build evidence-gated English and Korean publications

**Files:**
- Create: `paper-en/sleep-time-compute/*`
- Create: `study-kr/sleep-time-compute/*`
- Create: `research/sleep-time-compute/publication/links/parity.jsonl`
- Create: `research/sleep-time-compute/generated/publication/*`
- Create: `claims/sleep-time-compute.bundle.json`

**Interfaces:**
- Consumes: frozen claims, evidence, results, figures, theory, and limitations.
- Produces: English paper, Korean companion, parity map, Veridraft bundle, and
  publication-ready PDFs.

- [ ] **Step 1: Freeze the combined post-G4 execution snapshot**

Run the first command block of Evidence Task 13 Step 1:

```bash
cd research/sleep-time-compute
uv run stc experiments snapshot-g4 \
  --confirmatory-manifest manifests/stc/confirmatory.jsonl \
  --run-index manifests/stc/confirmatory-run-index.json \
  --analysis-plan manifests/stc/analysis-plan.json \
  --scaling-manifest manifests/stc/scaling-cells.jsonl \
  --scaling-cohort-index manifests/stc/scaling-cohort-index.json \
  --scaling-budget manifests/stc/scaling-budget.json \
  --scaling-config configs/stc/design/scaling.yaml \
  --scaling-hardware-config configs/stc/design/scaling-hardware.yaml \
  --preregistration manifests/stc/preregistration.json \
  --g4 manifests/gates/G4.json \
  --g4-input-index manifests/systems/g4-inputs.json \
  --output manifests/stc/g4-confirmatory-execution.json
```

Expected: the single legacy-named snapshot binds both the \(297n\)
confirmatory and \(112n\) scaling execution spaces, hard 135-RRE budget,
analysis, G2 preregistration, G4 decision/input index, checkout, and target
environment. Neither mode can substitute a different manifest.

- [ ] **Step 2: Execute and import the distributed primary run**

On the two A100 nodes execute the primary portion of Systems Task 8 Step 4
exactly: control digest copy, concurrent eight-rank `run-matrix`, manifest-last
verification, typed receipt, atomic `import-matrix`, semantic validation, DAG
import, and DAG check. Stop before the clean-rerun block.

Expected controller-owned outputs:

```text
results/confirmatory-primary/
manifests/systems/distributed-confirmatory-receipt.json
manifests/stc/confirmatory-result-validation.json
```

Every \(297n\) logical child reference resolves; deduplicated physical children
execute at most once; a valid receipt disables any second primary execution.

- [ ] **Step 3: Analyze the primary and prepare the clean rerun**

Run the evidence-owned analysis-only and typed-request commands from Evidence
Task 13:

```bash
cd research/sleep-time-compute
uv run stc experiments run-confirmatory \
  --mode analyze-imported \
  --hypotheses manifests/hypotheses-G2.json \
  --execution-snapshot manifests/stc/g4-confirmatory-execution.json \
  --validated-run results/confirmatory-primary \
  --receipt manifests/systems/distributed-confirmatory-receipt.json \
  --validation-report manifests/stc/confirmatory-result-validation.json \
  --output results/confirmatory-primary-analysis
uv run stc experiments clean-rerun prepare \
  --hypotheses manifests/hypotheses-G2.json \
  --execution-snapshot manifests/stc/g4-confirmatory-execution.json \
  --gate-chain manifests/gates \
  --primary-receipt manifests/systems/distributed-confirmatory-receipt.json \
  --primary-validation-report manifests/stc/confirmatory-result-validation.json \
  --repo-root ../.. \
  --run-id confirmatory-clean-rerun \
  --isolation-policy node-local-detached-fresh-caches \
  --output manifests/stc/confirmatory-clean-rerun-request.json
```

Expected: primary analysis is derived without relaunching children; the clean
request binds a different run ID, detached clean checkout, initially empty
caches, G4 snapshot, and primary receipt.

- [ ] **Step 4: Execute the clean rerun and A100 scaling track**

Resume Systems Task 8 Step 4 at its clean-rerun boundary. Both-node clean runs
must consume the request digest and produce the distinct clean receipt/import/
validation. Then execute the scaling portion exactly, including the frozen
96-fit/16-holdout roles, hard 135-RRE prelaunch/resume/merge enforcement,
receipt, atomic import, validation, and DAG import.

Expected controller-owned outputs include:

```text
results/confirmatory-clean-rerun/
results/scaling/
manifests/systems/distributed-confirmatory-clean-rerun-receipt.json
manifests/systems/distributed-scaling-receipt.json
manifests/stc/confirmatory-clean-rerun-result-validation.json
manifests/stc/scaling-result-validation.json
```

If target hardware is unavailable, preserve the exact external blocker; do not
run G5 or use synthetic outputs.

- [ ] **Step 5: Finalize independent calculations and pass typed G5**

Resume Evidence Task 13 Step 1 after the external-compute pause. Execute its
exact `clean-rerun finalize`, independent confirmatory calculation, primary
final scaling fit, independently implemented scaling fit, typed G5 input
assembly, and G5 evaluation commands.

```bash
cd research/sleep-time-compute
uv run stc gate verify-chain --through G5 --root manifests/gates
uv run stc validate --root . --gate G5
```

Expected: G5 binds G4, the combined execution snapshot, raw and analyzed
primary/clean trees, both receipts and validations, independent confirmatory
results, raw scaling, primary and independent scaling summaries, cohort, and
budget. Clean rerun and independent calculations agree within frozen
tolerances; incomplete children, actual RRE above 135, or disagreement blocks
headline claims.

- [ ] **Step 6: Adjudicate claims, fill both publications, and pass G6**

Execute Evidence Task 13 Step 2, then only Evidence Tasks 11–12 Phase B, then
Task 13 Step 3. Their Phase A skeleton/parity scaffold already exists from
G0. The order matters: registered result decisions and caveats replace the
non-assertive slots in both manuscripts before clean builds, parity validation,
Veridraft, G6 input assembly, and G6 evaluation.

```bash
cd research/sleep-time-compute
make -C ../../paper-en/sleep-time-compute clean all
make -C ../../study-kr/sleep-time-compute clean all
uv run stc publication validate-parity publication/links/parity.jsonl
uv run stc gate verify-chain --through G6 --root manifests/gates
uv run stc validate --root . --gate G6
```

Expected: builds exit 0; unresolved citations/xrefs, missing figures, blocked
claims, parity gaps, missing glyphs, and overfull text all equal zero. Every
PAPER-C revision is `SUPPORTED` or honestly `FALSIFIED/NARROWED` with exact
result blocks and material caveats; G6 binds G5, claim bundle, Veridraft gate,
both canonical PDFs, and trace/parity reports.

- [ ] **Step 7: Commit and push the scientific candidate**

Run:

```bash
git add claims/sleep-time-compute.bundle.json \
  paper-en/sleep-time-compute \
  study-kr/sleep-time-compute \
  research/sleep-time-compute
git commit -m "paper: freeze G6 sleep-time compute candidate"
git push origin HEAD
```

Expected: record the resulting commit as `scientific_candidate_sha`; a clean
checkout at that SHA rebuilds the same figure/table data and passes G5/G6.

### Task 8: Audit the frozen candidate and release G7–G8

**Files:**
- Create: `research/sleep-time-compute/reports/G7-independent-audit.md`
- Create: `research/sleep-time-compute/manifests/{trusted-reviewer-keys.json,G7-independence-protocol.json}`
- Create: `research/sleep-time-compute/manifests/attestations/{author-executor-roster,G7-auditor}.json`
- Create: `research/sleep-time-compute/manifests/attestations/G8-evaluator.json`
- Create: `research/sleep-time-compute/manifests/reaudits/G7.json`
- Create: `research/sleep-time-compute/publication/reviews/{visual-qa-machine,visual-qa-final,human-visual-approval}.json`
- Create: `research/sleep-time-compute/publication/reviews/all-pages-contact-sheet.png`
- Create: `research/sleep-time-compute/manifests/candidates/stc-paper-v1.0.0-rc1.json`
- Create: `research/sleep-time-compute/manifests/release-candidates/stc-paper-v1.0.0.json`
- Create: `research/sleep-time-compute/manifests/gate-inputs/G8-evaluation-subject.json`
- Create: `research/sleep-time-compute/reports/release-candidate-smoke.json`
- Create: `research/sleep-time-compute/manifests/gates/G7.json`
- Create: `research/sleep-time-compute/manifests/gates/G8.json`
- Create: `research/sleep-time-compute/publication/releases/stc-paper-v1.0.0/*`

**Interfaces:**
- Consumes: every research artifact and independent audit result.
- Produces: a research-only G8 tag that proves every scientific
  definition-of-done item and authorizes—but does not depend on—the PPTX
  handoff.

- [ ] **Step 1: Freeze the scientific candidate identity**

Execute Evidence Task 14 Phase B.1 exactly from the package root. The
release-control checkout need only have a clean tracked index/worktree;
unrelated user-owned untracked seminar files are preserved and ignored. The
candidate is a separate detached worktree at the peeled annotated RC tag and
must be completely clean, including untracked files.

Expected: `stc-paper-v1.0.0-rc1` resolves to the G6 candidate commit, and the
candidate record binds tag object, commit/tree, G0–G6, claims/results/papers,
environment, and artifact DAG without writing into the candidate.

Publish the immutable auditor input only after freeze succeeds:

```bash
git push origin stc-paper-v1.0.0-rc1
```

- [ ] **Step 2: Run independent G7 audit, adjudication, and re-audit**

Execute Evidence Task 14 Phase B.2 exactly. Audit evidence, coverage,
chronology, notation, statistics, reproduction, privacy/safety, deletion,
claim-to-figure trace, and Korean parity. Every trusted key, author/executor
roster, auditor identity/role, report, finding, adjudication, and re-audit is
signature- and subject-digest-bound.

Expected: typed G7 has zero unresolved critical/major findings, an exact G6
predecessor, verified independence, and an unchanged candidate digest. A
scientific change requires a new commit/RC and full restart.

- [ ] **Step 3: Replay, obtain visual and gate-evaluator approval, and pass G8**

Execute the first machine-replay/visual-QA block of Evidence Task 14 Phase B.3.
Then pause. An identified human reviews both exact PDF digests and every
contact-sheet page and signs `human-visual-approval.json`; automation cannot
manufacture the record or continue without it. Resume Phase B.3 to verify that
signature, finalize visual QA, freeze the immutable pre-seal inventory, run its
smoke test, and assemble the canonical G8 evaluation subject. Pause a second
time while an identified trusted `release_evaluator` reviews and signs that
exact subject. Finally verify the evaluator signature, assemble typed G8
inputs, evaluate G8, rights-scan staged audit records, and commit the
audit-control identity.

Expected: G8 binds G7, candidate, clean replay, public replay, exact-PDF human
approval, the typed G4 systems evidence/input index, and fixed measured run ID
`a100-confirmatory`. Measured KV/RMW, batching, adapter, GPU-HBM, host-DRAM,
energy, and final A100 scaling evidence must be admissible; otherwise G8 is
`BLOCKED`, never waived with synthetic data. G8 additionally binds the exact
pre-seal manifest/inventory and its smoke report, eliminating any dependency on
a not-yet-created post-G8 release. Its non-null evaluator-attestation ref is the
separate signature over the complete G8 subject, never the visual-only human
approval.

- [ ] **Step 4: Assemble and verify the redistribution-safe research release**

Execute Evidence Task 14 Phase B.4 exactly: only `stc release assemble` and
`stc release verify`. Do not execute the separate Phase B.6 presentation
handoff commands; Task 9 owns them. Assembly runs in the release-control
checkout, reads the detached candidate, and writes only
`publication/releases/stc-paper-v1.0.0`. It consumes the G8-bound draft manifest
and smoke report and emits a typed `release-seal.json`; it may add the G8 record
and seal metadata but may not alter any draft-declared payload.

Expected: deterministic source-date-normalized archives reproduce in different
controller directories. The release contains environment archive, SPDX SBOM,
container/toolchain metadata, command log, full G0–G8 chain, and a
redistribution-safe reproduction bundle; unlicensed source payloads are
excluded per artifact while URL/checksum/retrieval metadata remain. The
`ReleaseSealRecord` proves byte identity between the frozen draft inventory and
the final release payload.

- [ ] **Step 5: Commit, tag, and push the audited research release**

Execute Evidence Task 14 Phase B.5 exactly. It stages only manifests and
`publication/releases`, rights-scans, re-verifies, commits, creates the
annotated research tag, and proves that the object is a tag. Presentation
handoff validation is intentionally absent and cannot block this commit. Then
publish the already-created identities:

```bash
git push origin HEAD
git push origin stc-paper-v1.0.0-rc1
git push origin stc-paper-v1.0.0
```

Expected: the final annotated tag points to the controller-owned release
metadata commit. Its manifest separately names the immutable scientific
candidate SHA and audit-control SHA; no presentation outcome can change G8 or
the research tag.

### Task 9: Export and consume the non-blocking presentation handoff

**Files:**
- Create: `research/sleep-time-compute/publication/handoff/stc-paper-v1.0.0/presentation-handoff.json`
- Create: `research/sleep-time-compute/publication/handoff/stc-paper-v1.0.0/presentation-handoff.schema.json`
- Create: `research/sleep-time-compute/publication/handoff/stc-paper-v1.0.0/release-manifest.json`
- Create: `research/sleep-time-compute/publication/handoff/stc-paper-v1.0.0/artifact-dag.jsonl`
- Create: `research/sleep-time-compute/publication/handoff/stc-paper-v1.0.0/{source,claim,evidence,result}-index.jsonl`
- Create: `research/sleep-time-compute/publication/handoff/stc-paper-v1.0.0/schemas/*.schema.json`
- Create: `research/sleep-time-compute/publication/handoff/stc-paper-v1.0.0/checksums.sha256`
- Create: `research/sleep-time-compute/publication/handoff/stc-paper-v1.0.0/assets/*`
- Create: `presentation/sleep-time-compute/*`

**Interfaces:**
- Consumes: the frozen `stc-paper-v1.0.0` G8 claim/figure/table release only.
- Produces: editable PPTX, display-authoritative PDF, notes, reports, and sealed
  presentation checksums without changing research claims.

- [ ] **Step 1: Export, validate, and commit the frozen research handoff**

Execute Evidence Task 14 Phase B.6 exactly. Its exporter verifies the existing
annotated research tag and commits only the self-contained handoff. In
particular, run:

```bash
cd research/sleep-time-compute
uv run stc presentation export \
  --release publication/releases/stc-paper-v1.0.0 \
  --research-tag stc-paper-v1.0.0 \
  --output publication/handoff/stc-paper-v1.0.0
uv run stc presentation validate \
  publication/handoff/stc-paper-v1.0.0 \
  --offline
git -C ../.. add research/sleep-time-compute/publication/handoff
uv run stc sources staged-scan --repo-root ../.. --cached
git -C ../.. diff --cached --check
git -C ../.. commit -m "presentation: add sleep-time evidence handoff"
```

Expected: every slide candidate references resolvable claim/evidence/asset IDs
and hashes; every slide tag equals the envelope tag; no stale node or
unverified factual copy is exported; the release, DAG, schema, minimal
redistribution-safe source/claim/evidence/result indexes, assets, and bundle
checksum digests all resolve exclusively inside the handoff directory.
Any failure blocks only the presentation track and leaves the research release
commit, G8 record, and annotated tag unchanged.

- [ ] **Step 2: Scaffold, render, and machine-check the generated parent deck**

First resolve the mandatory tooling branch from the repository root with
fail-closed state transitions:

1. If no reviewer trust-root commit exists, execute Presentation Task 0 Step 0
   once. If a trust-root file or commit exists but its approval/digest is
   invalid or mutated, stop for human security remediation; never regenerate
   or overwrite it.
2. With a valid trust root, if no signed spike outcome exists, execute
   Presentation Task 0 Steps 1–6 once. If any part of an existing signed
   outcome exists but offline verification fails, stop for adjudication; never
   rerun the spike as an automatic repair.
3. The Step 0 trust-root and Step 6 outcome commits are mandatory standalone
   identities and are not absorbed into the Task 9 integration commit.

These paths pause for human key provisioning, PowerPoint operations, and
content/visual review. Do not create the presentation package or factual deck
until this command passes:

```bash
node docs/presentation/verify-pptx-spike.mjs \
  --manifest docs/presentation/pptx-spike-manifest.json \
  --fixtures docs/presentation/pptx-spike-fixtures \
  --trusted-keys docs/presentation/pptx-reviewer-trust-root.json \
  --offline
```

Then execute Presentation Tasks 1–4 and Task 5 Steps 1–3 against the read-only
handoff, skipping their local commit steps under commit arbitration. The
representative slice must pass machine QA and receive three digest-bound,
trust-root-verified role approvals before the complete-deck command can run.

```bash
cd presentation/sleep-time-compute
npm ci
npm run build
npm test
node dist/cli.js validate-handoff \
  ../../research/sleep-time-compute/publication/handoff/stc-paper-v1.0.0
node dist/cli.js build \
  --handoff-bundle ../../research/sleep-time-compute/publication/handoff/stc-paper-v1.0.0 \
  --slide-ids-file reports/representative-slice.json \
  --output build/representative.pptx
```

Pause. The trusted `POWERPOINT_OPERATOR` opens the exact representative PPTX
without repair and exports
`build/qa/representative/representative-powerpoint.pdf`. Then run:

```bash
cd presentation/sleep-time-compute
: "${STC_TRUST_ROOT_COMMIT:?set to Presentation Task 0 Step 0 commit}"
npm run qa -- \
  --handoff ../../research/sleep-time-compute/publication/handoff/stc-paper-v1.0.0 \
  --slide-ids-file reports/representative-slice.json \
  --pptx build/representative.pptx \
  --out build/qa/representative \
  --pdf-input build/qa/representative/representative-powerpoint.pdf
node dist/cli.js editorial-subject \
  --pptx build/representative.pptx \
  --powerpoint-pdf build/qa/representative/representative-powerpoint.pdf \
  --slide-ids-file reports/representative-slice.json \
  --qa build/qa/representative/report.json \
  --contact-sheet build/qa/representative/contact-sheet.png \
  --trusted-keys ../../docs/presentation/pptx-reviewer-trust-root.json \
  --trust-root-commit "$STC_TRUST_ROOT_COMMIT" \
  --output reports/representative-slice-subject.json
```

Pause for the PowerPoint operator, content owner, and independent visual editor
to sign the exact representative-slice subject as specified in Presentation
Task 5. Continue only after:

```bash
cd presentation/sleep-time-compute
node dist/cli.js verify-editorial-approval \
  --subject reports/representative-slice-subject.json \
  --powerpoint-operator-attestation \
    reports/representative-slice-powerpoint-operator-attestation.json \
  --content-owner-attestation \
    reports/representative-slice-content-owner-attestation.json \
  --visual-editor-attestation \
    reports/representative-slice-visual-editor-attestation.json \
  --trusted-keys ../../docs/presentation/pptx-reviewer-trust-root.json \
  --output reports/editorial-approval.json
```

Only now render the full deck and create a distinct full-deck review subject:

```bash
cd presentation/sleep-time-compute
: "${STC_TRUST_ROOT_COMMIT:?set to Presentation Task 0 Step 0 commit}"
node dist/cli.js build \
  --handoff-bundle ../../research/sleep-time-compute/publication/handoff/stc-paper-v1.0.0 \
  --output build/generated.pptx
npm run qa -- \
  --handoff ../../research/sleep-time-compute/publication/handoff/stc-paper-v1.0.0 \
  --pptx build/generated.pptx \
  --out build/qa/generated \
  --pdf-output build/qa/generated/generated.pdf
node dist/cli.js full-deck-subject \
  --pptx build/generated.pptx \
  --provisional-pdf build/qa/generated/generated.pdf \
  --contact-sheet build/qa/generated/contact-sheet.png \
  --qa build/qa/generated/report.json \
  --pacing-rubric reports/full-deck-review-rubric.json \
  --trusted-keys ../../docs/presentation/pptx-reviewer-trust-root.json \
  --trust-root-commit "$STC_TRUST_ROOT_COMMIT" \
  --output reports/full-deck-subject.json
```

Pause again. The content owner and visual editor review every page and sign the
full-deck subject; representative-slice approval cannot authorize unseen
slides. Verify before entering postflight:

```bash
cd presentation/sleep-time-compute
node dist/cli.js verify-full-deck-approval \
  --subject reports/full-deck-subject.json \
  --content-owner-attestation \
    reports/full-deck-content-owner-attestation.json \
  --visual-editor-attestation \
    reports/full-deck-visual-editor-attestation.json \
  --trusted-keys ../../docs/presentation/pptx-reviewer-trust-root.json \
  --output reports/full-deck-approval.json
```

Expected: `generated.pptx`, its provisional QA PDF, contact sheet, notes,
OOXML/geometry/font/accessibility/render reports, and checksums exist. The
content owner and independent visual editor approvals resolve only through the
pre-provisioned trust root and separately bind the representative slice and
full-deck pacing subjects. The full-deck approval covers every generated slide.
The provisional PDF is not `display.pdf`.

- [ ] **Step 3: Pause for target-PowerPoint postflight**

An identified human operator opens the exact `generated.pptx` digest in the
recorded desktop PowerPoint version, applies only font embedding, reading
order, object name/description, or accessibility-metadata edits, saves and
reopens `build/release.pptx` without repair/warnings, reruns Accessibility
Checker and Reading Order, and records
`reports/powerpoint-{reopen,accessibility}.json`. Do not export the release PDF
or sign the terminal subject until automated semantic postflight validation
passes. No automated worker fabricates these observations or silently proceeds
around the pause.

- [ ] **Step 4: Validate the human delta and seal the authoritative PDF**

Execute Presentation Task 6:

```bash
cd presentation/sleep-time-compute
npm run build
npm test -- \
  postflight.test.ts \
  powerpoint_attestation.test.ts \
  cli.test.ts
node dist/cli.js verify-full-deck-approval \
  --subject reports/full-deck-subject.json \
  --content-owner-attestation \
    reports/full-deck-content-owner-attestation.json \
  --visual-editor-attestation \
    reports/full-deck-visual-editor-attestation.json \
  --trusted-keys ../../docs/presentation/pptx-reviewer-trust-root.json \
  --output reports/full-deck-approval.json
node dist/cli.js validate-postflight \
  --generated build/generated.pptx \
  --release build/release.pptx \
  --report build/qa/postflight.json
```

Pause. The trusted operator now exports `build/release.pdf` from the exact
reopened, validated `release.pptx` and records the complete options in
`reports/powerpoint-export-options.json`. Then run:

```bash
cd presentation/sleep-time-compute
: "${STC_TRUST_ROOT_COMMIT:?set to Presentation Task 0 Step 0 commit}"
npm run qa -- \
  --handoff ../../research/sleep-time-compute/publication/handoff/stc-paper-v1.0.0 \
  --pptx build/release.pptx \
  --out build/qa/release \
  --pdf-input build/release.pdf
node dist/cli.js compare-renders \
  --generated build/qa/generated/renders \
  --release build/qa/release/renders \
  --output build/qa/release-render-diff.json
node dist/cli.js powerpoint-postflight-subject \
  --generated build/generated.pptx \
  --release build/release.pptx \
  --postflight build/qa/postflight.json \
  --reopen-report reports/powerpoint-reopen.json \
  --accessibility-report reports/powerpoint-accessibility.json \
  --release-pdf build/release.pdf \
  --export-options reports/powerpoint-export-options.json \
  --trusted-keys ../../docs/presentation/pptx-reviewer-trust-root.json \
  --trust-root-commit "$STC_TRUST_ROOT_COMMIT" \
  --output reports/powerpoint-postflight-subject.json
```

Pause once more. The pre-provisioned `POWERPOINT_OPERATOR` signs that exact
canonical subject. Verify the signature and seal only afterward:

```bash
cd presentation/sleep-time-compute
node dist/cli.js verify-powerpoint-operator \
  --subject reports/powerpoint-postflight-subject.json \
  --operator-attestation \
    reports/powerpoint-postflight-operator-attestation.json \
  --trusted-keys ../../docs/presentation/pptx-reviewer-trust-root.json \
  --output reports/powerpoint-postflight-qa.json
cp build/release.pdf build/display.pdf
cmp build/release.pdf build/display.pdf
node dist/cli.js seal \
  --generated build/generated.pptx \
  --release build/release.pptx \
  --release-pdf build/release.pdf \
  --display-pdf build/display.pdf \
  --qa build/qa/release \
  --postflight build/qa/postflight.json \
  --full-deck-approval reports/full-deck-approval.json \
  --powerpoint-subject reports/powerpoint-postflight-subject.json \
  --powerpoint-operator-attestation \
    reports/powerpoint-postflight-operator-attestation.json \
  --powerpoint-qa reports/powerpoint-postflight-qa.json \
  --trusted-keys ../../docs/presentation/pptx-reviewer-trust-root.json \
  --render-diff build/qa/release-render-diff.json \
  --output manifests/release.json
sha256sum build/release.pptx build/release.pdf build/display.pdf \
  > build/release-checksums.sha256
```

Expected: all factual/chart/media/geometry deltas are zero; every package
change is in the signed allowlist; Open XML, font, accessibility, render,
veraPDF, and exact-PDF human checks pass. `display.pdf` is byte-identical to
the validated `release.pdf` exported from the sealed `release.pptx`.

- [ ] **Step 5: Commit and push the presentation manifests**

```bash
git add -f \
  presentation/sleep-time-compute/build/representative.pptx \
  presentation/sleep-time-compute/build/generated.pptx \
  presentation/sleep-time-compute/build/release.pptx \
  presentation/sleep-time-compute/build/release.pdf \
  presentation/sleep-time-compute/build/display.pdf \
  presentation/sleep-time-compute/build/release-checksums.sha256 \
  presentation/sleep-time-compute/build/qa/representative \
  presentation/sleep-time-compute/build/qa/generated \
  presentation/sleep-time-compute/build/qa/release \
  presentation/sleep-time-compute/build/qa/postflight.json \
  presentation/sleep-time-compute/build/qa/release-render-diff.json
git add presentation/sleep-time-compute
for STC_DURABLE_LEAF in \
  presentation/sleep-time-compute/build/representative.pptx \
  presentation/sleep-time-compute/build/generated.pptx \
  presentation/sleep-time-compute/build/release.pptx \
  presentation/sleep-time-compute/build/release.pdf \
  presentation/sleep-time-compute/build/display.pdf \
  presentation/sleep-time-compute/build/qa/representative/report.json \
  presentation/sleep-time-compute/build/qa/generated/report.json \
  presentation/sleep-time-compute/build/qa/release/report.json \
  presentation/sleep-time-compute/build/qa/postflight.json
do
  git ls-files --error-unmatch "$STC_DURABLE_LEAF"
done
git commit -m "presentation: release audited sleep-time compute deck"
git push origin HEAD
```

Expected: deck provenance points to `stc-paper-v1.0.0`; presentation commits do
not alter the frozen research release manifest or claim bundle.
