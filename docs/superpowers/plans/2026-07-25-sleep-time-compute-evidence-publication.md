# Sleep-Time Compute Evidence and Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a primary-source evidence atlas and claim graph that can generate an evidence-gated English paper and parity-checked Korean companion without unsupported or stale claims.

**Architecture:** Typed append-only JSONL registries and one canonical JSON paper card per source are the source of truth; Markdown registries, cards, bibliographies, and assembled manuscripts are generated views linked by stable IDs. A validator enforces schema, cross-reference, evidence-boundary, release-status, manuscript-parity, and artifact-digest rules. Veridraft receives a generated, separate bundle rather than becoming the canonical registry.

**Tech Stack:** Python 3.14, uv, pytest, Hypothesis, jsonschema, PyYAML, Veridraft 0.3.0, Pandoc 3.1.11, LuaLaTeX, BibLaTeX.

## Global Constraints

- Source chronology uses the earliest verified public date and preserves venue/status separately.
- `A-ABS` evidence cannot support a method/result claim absent from the abstract.
- Every source summary identifies the source version and exact anchor.
- Direct quotation is minimized; evidence spans store a short anchor and a faithful support summary.
- Blog, product, vendor, and paper claims use distinct evidence classes.
- `PAPER-C1` through `PAPER-C4` are the only main paper claims.
- The bounded contribution hypothesis is the matched, versioned lifetime
  routing of the same items across raw external, abstract external, latent/KV,
  and user-parametric media—not any component in isolation. Novelty and
  priority remain separately audited and may remain unresolved.
- Existing `claims/bundle.json` is not modified.
- Every manuscript claim resolves through `claim_id → evidence/result IDs → source/result artifact digest`.
- The Korean companion may be broader than the paper but every added assertion receives its own registered claim.
- No paper is marked `FULL` until the methods, experiments, limitations, and supplement needed by its card have been inspected.
- Source grades are exactly `A`, `B`, `C`, `D`, or `X`; review status is exactly
  `A-ABS`, `A-HTML`, `FULL`, or `REPRO`.
- Evidence relations are exactly `SUPPORTS`, `CONTRADICTS`, `QUALIFIES`, or
  `NON_CLAIM`; there is no scalar confidence score.
- Negative results and contradictions are first-class records, not prose
  discarded during synthesis.
- Search absence is a falsifiable novelty hypothesis, never proof of novelty.
- Vendor/product versions and related papers are distinct source records.
- Withdrawn, contradicted, or unverifiable sources are grade `X` and cannot
  support release conclusions.
- Command-root contract: every `bash` block starts in a fresh shell whose
  working directory is `research/sleep-time-compute/`; blocks never inherit a
  prior `cd`, shell variable, or activation. Package paths (`registry/`,
  `manifests/`, `results/`, `publication/`) are relative to that directory,
  repository paths use `../../`, and controller Git commands use
  `git -C ../..` with repository-root-relative pathspecs. The sole exception
  is a command intentionally inspecting an explicit detached worktree, which
  uses `git -C <package-relative-worktree>`. An executor that cannot set the
  working directory must prepend `cd research/sleep-time-compute` to each block.
- G0 scaffold verification treats `publication/scaffolds/G0` as a read-only
  input. It copies only manifest-declared source bytes into a newly created
  temporary tree, performs both language builds, both trace checks, and parity
  validation there, and can pass only when pre/post rehashes of the snapshot
  manifest and its declared source inventory are identical.

---

## Record Interfaces

```python
SubjectRef(
    path: str,
    sha256: str,
)

ArtifactRef(
    artifact_id: str,
    canonical_url: str | None,
    local_path: str | None,
    sha256: str,
    media_type: str,
    availability: str,
    license_id: str,
    license_evidence_url: str,
    license_evidence_sha256: str,
    redistribution_allowed: bool,
    license_review_disposition: str,
    rights_reviewer: str,
    rights_reviewed_at: str,
    rights_review_expires_at: str | None,
)

SourceRecord(
    source_id: str,
    legacy_ids: list[str],
    canonical_key: str,
    title: str,
    authors: list[str],
    earliest_public_date: str,
    date_precision: str,
    venue_status: str,
    peer_review_status: str,
    versions: list[SourceVersion],
    source_grade: str,
    review_status: str,
    urls: list[str],
    local_artifacts: list[ArtifactRef],
    citation_key: str,
    code_records: list[ArtifactRef],
    data_records: list[ArtifactRef],
    rights_summary: str,
    topics: list[str],
    affiliations: list[str],
    inclusion_reason: str,
    non_claim: str,
    last_verified: str,
)

EvidenceRecord(
    evidence_id: str,
    source_id: str,
    source_version: str,
    source_version_digest: str,
    anchor_kind: str,
    anchor: str,
    support_span_digest: str,
    support_summary: str,
    relation: str,
    claim_ids: list[str],
    question_ids: list[str],
    hypothesis_ids: list[str],
    source_grade: str,
    warrant: str,
    reproduction_strength: str,
    assumptions: list[str],
    scope: str,
    counterevidence_ids: list[str],
    reviewer: str,
    reviewed_at: str,
)

ClaimSupportRef(
    evidence_id: str,
    source_id: str,
    source_version: str,
    source_version_digest: str,
    artifact_id: str,
    support_span_digest: str,
    relation: str,
    role: str,  # RELEASE_SUPPORT | CONTEXT | COUNTEREVIDENCE
    source_grade: str,
    warrant: str,
    reproduction_strength: str,
)

ClaimEvidenceRequirement(
    allowed_claim_classes: list[str],
    allowed_source_grades: list[str],
    allowed_warrants: list[str],
    minimum_reproduction_strength: str,
    minimum_independent_sources: int,
    requires_result_block: bool,
    minimum_independent_result_paths: int,
)

ClaimResultProvenance(
    asserted_modality: str,  # explicit result modality or NOT_APPLICABLE
    calibration_modalities: list[str],
)

ClaimRecord(
    claim_id: str,
    statement: str,
    headline_quantitative: bool,
    claim_class: str,
    result_provenance: ClaimResultProvenance,
    status: str,
    scope: str,
    assumptions: list[str],
    source_ids: list[str],
    evidence_ids: list[str],
    support_refs: list[ClaimSupportRef],
    evidence_requirement: ClaimEvidenceRequirement,
    counterevidence_ids: list[str],
    result_ids: list[str],
    artifact_dependencies: list[str],
    caveats: list[str],
    falsifier: str,
    paper_owner: str,
    gate_status: str,
    last_audit_date: str,
)

PaperCard(
    source_id: str,
    reviewed_version_digest: str,
    reviewer: str,
    reviewed_at: str,
    research_question: str,
    wake_input: str,
    trigger: str,
    operator: str,
    destination: str,
    training_data: list[str],
    objective: list[str],
    optimizer: str,
    update_location: str,
    cadence: str,
    capacity_policy: str,
    systems_assumptions: list[str],
    reported_results: list[str],
    limitations: list[str],
    failure_modes: list[str],
    evidence_ids: list[str],
    counterevidence_ids: list[str],
    code_review: str,
    data_review: str,
    non_claims: list[str],
)

DigestRef(
    artifact_id: str,
    sha256: str,
)

AuditAttestation(
    attestation_id: str,
    subject_sha256: str,
    subject_refs: list[SubjectRef],
    signer_id: str,
    signer_role: str,
    independence_mode: str,
    author_executor_roster_digest: str,
    algorithm: str,
    key_id: str,
    signature: str,
    signed_at: str,
)

GateEvaluationAttestation(
    attestation_id: str,
    gate_id: str,
    subject_sha256: str,
    subject_refs: list[SubjectRef],
    signer_id: str,
    signer_role: str,
    independence_mode: str,
    algorithm: str,
    key_id: str,
    signature: str,
    signed_at: str,
)

HumanApproval(
    approval_id: str,
    subject_sha256: str,
    pdf_digests: list[str],
    machine_visual_report_digest: str,
    contact_sheet_digest: str,
    all_pages_reviewed: bool,
    review_checks: dict[str, str],
    disposition: str,
    open_issues: list[str],
    signer_id: str,
    signer_role: str,
    algorithm: str,
    key_id: str,
    signature: str,
    signed_at: str,
)

GateRecord(
    gate_id: str,
    schema_version: str,
    evaluation_commit_sha: str,
    evaluation_tree_digest: str,
    scientific_candidate_sha: str | None,
    scientific_candidate_tree_digest: str | None,
    artifact_dag_digest: str,
    predecessor_gate_refs: list[DigestRef],
    input_refs: list[DigestRef],
    evaluator_id: str,
    evaluator_role: str,
    independence_mode: str,
    evaluator_attestation_ref: DigestRef | None,
    finding_ids: list[str],
    adjudication_ids: list[str],
    reaudit_refs: list[DigestRef],
    environment_digest: str,
    evaluated_at: str,
    status: str,  # PASS | FAIL | BLOCKED
)
```

`SubjectRef`, `SourceVersion`, `ArtifactRef`, and `ClaimResultProvenance` are
immutable nested records defined once in `models.py` and exported through
`$defs` in `common.schema.json`.
`SourceVersion` contains identifier, public date/precision, status, canonical
URL, local artifact ID, SHA-256, and supersedes/derived-from lineage.
Rights attach to each immutable `ArtifactRef`, not merely to its source. A
source-level `rights_summary` is a non-authoritative generated roll-up and
cannot authorize export. `source_ids` and `evidence_ids` in a `ClaimRecord` are
derived indexes whose exact equality with `support_refs` is validated; raw IDs
or raw anchor strings are never sufficient release support. Only
`ClaimSupportRef.role == "RELEASE_SUPPORT"` entries count toward the typed
`ClaimEvidenceRequirement`; `CONTEXT` and `COUNTEREVIDENCE` remain visible but
cannot be promoted implicitly. Scholarly author/reviewer identity is
permitted; the direct-identifier ban applies only to retained benchmark/user
operational metadata. Automated G0–G6 records identify the evaluator tool and
mode without pretending to have a human signature; G7/G8 require a non-null,
verified `evaluator_attestation_ref`. Every gate binds its evaluated commit and
tree; the final scientific-candidate fields are null before freeze and required
for G7/G8, avoiding a self-referential pre-commit SHA.

Allowed claim classes are exactly:

```text
SOURCE-SUMMARY
VENDOR-BEHAVIOR
RERUN
INDEPENDENT-REPLICATION
ORIGINAL-MEASUREMENT
ANALYTIC-DERIVATION
TRACE-SIMULATION
SYNTHESIS
HYPOTHESIS
```

Allowed release statuses are exactly:

```text
DRAFT
HYPOTHESIS
UNRESOLVED
SUPPORTED
FALSIFIED/NARROWED
BLOCKED
WITHDRAWN
```

The orthogonal warrant vocabulary is `MEAS`, `REPL`, `DERIV`, `SUMM`, `SYNTH`,
or `PROP`; reproduction strength is `r0`, `r1`, `r2`, or `r3`. Result modality
is exactly `NOT_APPLICABLE`, `SOURCE_REPORTED`, `SIMULATED`,
`MODEL_ESTIMATED`, `ANALYTICAL`, `ACTUAL_MEASUREMENT`, or `ACTUAL_HARDWARE`.
The first is permitted only when the claim has no result dependency.

---

### Task 1: Scaffold the isolated research package

**Files:**
- Create: `research/sleep-time-compute/.gitignore`
- Create: `research/sleep-time-compute/pyproject.toml`
- Create: `research/sleep-time-compute/uv.lock`
- Create: `research/sleep-time-compute/src/stc_research/__init__.py`
- Create: `research/sleep-time-compute/src/stc_research/cli.py`
- Create: `research/sleep-time-compute/tests/test_cli.py`
- Create: `research/sleep-time-compute/README.md`

**Interfaces:**
- Consumes: Python 3.14 and `uv`.
- Produces: console command `stc`.

- [ ] **Step 1: Write the failing CLI test**

```python
from stc_research.cli import main


def test_version_command(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == "stc-research 0.1.0"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest tests/test_cli.py::test_version_command -q
```

Expected: FAIL because `stc_research` does not exist.

- [ ] **Step 3: Create the package metadata**

Use this project contract:

```toml
[project]
name = "stc-research"
version = "0.1.0"
requires-python = ">=3.14"
dependencies = [
  "jsonschema>=4.25,<5",
  "numpy>=2.5,<3",
  "scipy>=1.18,<2",
  "matplotlib>=3.11,<4",
  "PyYAML>=6.0,<7",
]

[project.scripts]
stc = "stc_research.cli:main"

[dependency-groups]
dev = [
  "pytest>=8.4,<10",
  "hypothesis>=6.130,<7",
  "ruff>=0.12,<1",
]
publication = [
  "veridraft @ git+ssh://git@github.com/jimmylegendary/veridraft.git@176b46c0e29c44fec3c2863178c35dfe292aaa1e",
]

[build-system]
requires = ["hatchling>=1.27,<2"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/stc_research"]
```

The package-local `.gitignore` contains at minimum:

```text
.cache/stc/sources/
results/
build/candidate-worktrees/
build/replay-worktrees/
build/g5-clean-reruns/
.veridraft-stc/
```

The source-cache rule is a safety boundary: no later task may negate it or
force-add a cached full text. `results/` is the digest-verified large-payload
store; only its small typed receipts, validations, result blocks, and artifact
records are committed.

- [ ] **Step 4: Implement the minimal CLI**

```python
from __future__ import annotations

import argparse
from collections.abc import Sequence

from stc_research import __version__


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stc")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version")
    args = parser.parse_args(argv)
    if args.command == "version":
        print(f"stc-research {__version__}")
        return 0
    raise AssertionError(args.command)
```

`__init__.py` contains:

```python
__version__ = "0.1.0"
```

- [ ] **Step 5: Lock dependencies and rerun**

Run:

```bash
uv lock
uv sync --frozen
uv run pytest tests/test_cli.py -q
git -C ../.. check-ignore --quiet \
  research/sleep-time-compute/build/candidate-worktrees/.probe
git -C ../.. check-ignore --quiet \
  research/sleep-time-compute/build/replay-worktrees/.probe
git -C ../.. check-ignore --quiet \
  research/sleep-time-compute/build/g5-clean-reruns/.probe
git -C ../.. check-ignore --quiet \
  research/sleep-time-compute/results/.probe
```

Expected: `1 passed`; all three nested-worktree probes and the large result
store probe are ignored.

- [ ] **Step 6: Commit**

```bash
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: scaffold sleep-time evidence package"
```

### Task 2: Implement stable IDs and typed records

**Files:**
- Create: `research/sleep-time-compute/src/stc_research/ids.py`
- Create: `research/sleep-time-compute/src/stc_research/models.py`
- Create: `research/sleep-time-compute/tests/test_ids.py`
- Create: `research/sleep-time-compute/tests/test_models.py`

**Interfaces:**
- Produces: `parse_stable_id(value: str) -> StableId` and dataclasses
  `SubjectRef`, `ArtifactRef`, `SourceRecord`, `EvidenceRecord`, `ClaimSupportRef`,
  `ClaimEvidenceRequirement`, `ClaimResultProvenance`, `ClaimRecord`,
  `PaperCard`, `QuestionRecord`, `HypothesisRecord`, `ArtifactNode`,
  `DigestRef`, `AuditAttestation`, `GateEvaluationAttestation`,
  `HumanApproval`, and `GateRecord`.

- [ ] **Step 1: Write ID rejection tests**

```python
import pytest

from stc_research.ids import parse_stable_id


@pytest.mark.parametrize(
    ("value", "kind", "number"),
    [
        ("SRC-STC-0001", "source", 1),
        ("EV-STC-00001", "evidence", 1),
        ("CL-STC-0042", "claim", 42),
        ("H-STC-005", "hypothesis", 5),
        ("H-STC-006", "hypothesis", 6),
        ("H-STC-007", "hypothesis", 7),
        ("PAPER-C4", "paper_claim", 4),
        ("RQ12", "question", 12),
        ("FIG-STC-042", "figure", 42),
        ("TAB-STC-007", "table", 7),
    ],
)
def test_parse_stable_id(value, kind, number):
    parsed = parse_stable_id(value)
    assert (parsed.kind, parsed.number) == (kind, number)


@pytest.mark.parametrize("value", ["SRC-STC-1", "EV-00001", "PAPER-C5", ""])
def test_reject_noncanonical_ids(value):
    with pytest.raises(ValueError):
        parse_stable_id(value)
```

- [ ] **Step 2: Run tests and confirm failure**

```bash
uv run pytest tests/test_ids.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement exact ID patterns**

```python
PATTERNS = {
    "source": re.compile(r"^SRC-STC-(\d{4})$"),
    "evidence": re.compile(r"^EV-STC-(\d{5})$"),
    "claim": re.compile(r"^CL-STC-(\d{4})$"),
    "hypothesis": re.compile(r"^H-STC-(\d{3})$"),
    "paper_claim": re.compile(r"^PAPER-C([1-4])$"),
    "question": re.compile(r"^RQ([1-9]|1[0-2])$"),
    "artifact": re.compile(r"^ART-STC-(\d{4})$"),
    "experiment": re.compile(r"^EXP-STC-(\d{4})$"),
    "result": re.compile(r"^RES-STC-(\d{4})$"),
    "figure": re.compile(r"^FIG-STC-(\d{3})$"),
    "table": re.compile(r"^TAB-STC-(\d{3})$"),
}
```

Return an immutable `StableId(kind: str, number: int, value: str)`.
`H-STC-000` is rejected; the durable namespace permits `H-STC-001` through
`H-STC-999`, while the initial registry seeds only 001–007.

- [ ] **Step 4: Add record-construction tests**

Test that `SourceRecord.from_dict()` rejects:

```text
unknown review_status
invalid ISO date
empty title
noncanonical source_id
direct identifier in operational metadata
unknown source_grade
```

Test that `ClaimRecord.from_dict()` rejects:

```text
unknown claim_class
PAPER-C status UNRESOLVED at release
headline numeric claim with reproduction_strength below r2
missing explicit headline_quantitative classification
TRACE-SIMULATION claiming measured hardware latency
result-bearing claim using NOT_APPLICABLE result provenance
raw evidence/source IDs that do not exactly equal typed support refs
release support that fails its typed ClaimEvidenceRequirement
two versions of one source counted as two independent sources
```

- [ ] **Step 5: Implement enums and immutable dataclasses**

Use `StrEnum` for the exact vocabularies and `@dataclass(frozen=True,
slots=True)` for records. `to_dict()` must produce deterministic field order.

- [ ] **Step 6: Run and commit**

```bash
uv run pytest tests/test_ids.py tests/test_models.py -q
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: define evidence and claim record contracts"
```

Expected: all tests pass.

### Task 3: Add canonical JSONL persistence

**Files:**
- Create: `research/sleep-time-compute/src/stc_research/jsonl_store.py`
- Create: `research/sleep-time-compute/tests/test_jsonl_store.py`
- Create: `research/sleep-time-compute/registry/sources.jsonl`
- Create: `research/sleep-time-compute/registry/evidence.jsonl`
- Create: `research/sleep-time-compute/registry/claims.jsonl`
- Create: `research/sleep-time-compute/registry/claim-history.jsonl`
- Create: `research/sleep-time-compute/registry/questions.jsonl`
- Create: `research/sleep-time-compute/registry/hypotheses.jsonl`
- Create: `research/sleep-time-compute/registry/terminology.jsonl`
- Create: `research/sleep-time-compute/registry/contradictions.jsonl`
- Create: `research/sleep-time-compute/registry/negative-results.jsonl`
- Create: `research/sleep-time-compute/registry/audit-findings.jsonl`
- Create: `research/sleep-time-compute/registry/audit-adjudications.jsonl`
- Create: `research/sleep-time-compute/registry/artifacts.jsonl`

**Interfaces:**
- Produces: `read_jsonl(path, factory)`,
  `canonical_jsonl_digest(records)`,
  `write_jsonl_atomic(path, records, *, expected_current_digest=None,
  retry_token=None)`, and
  `append_unique(path, record, id_field, *, retry_token=None)`.
- `write_jsonl_atomic` creates an absent registry, treats an identical
  canonical payload as a no-op, and requires a matching
  `expected_current_digest` for every state-changing replacement of an existing
  registry. Missing or stale preconditions raise `StaleWriteError` without
  changing the destination.
- Both writers use the same persistent sibling `flock` and canonical
  operation-bound retry tokens. A post-`os.replace` directory-fsync failure
  raises `CommitOutcomeUnknownError`; only the exact canonical operation may be
  retried, and an exact retry must never erase a newer append.
- Writer destinations reserve every basename ending in `.lock` and every
  basename beginning `.stc-jsonl-`, including parent components the writer
  would create. Validation occurs before directory creation or input
  materialization. Existing coordination locks must be zero-byte regular
  files and are opened with `O_NOFOLLOW`/`O_CLOEXEC`, then revalidated through
  the opened descriptor before `flock`.
- Temporary files use a deterministic destination-specific SHA-256 namespace;
  cleanup never crosses destination namespaces. New registries use mode
  `0644`; replacements preserve ordinary permission bits, strip special mode
  bits, and apply the mode before file `fsync`. Symlink and non-regular
  registry destinations fail closed.

- [ ] **Step 1: Write atomicity, canonical-identity, CAS, and concurrency tests**

```python
def test_write_jsonl_is_canonical_and_atomic(tmp_path):
    path = tmp_path / "sources.jsonl"
    write_jsonl_atomic(path, [{"source_id": "SRC-STC-0002", "title": "B"},
                              {"source_id": "SRC-STC-0001", "title": "A"}])
    assert path.read_text().splitlines()[0].startswith(
        '{"source_id":"SRC-STC-0001"'
    )
    assert not list(tmp_path.glob("*.tmp"))


def test_append_unique_rejects_duplicate(tmp_path):
    path = tmp_path / "sources.jsonl"
    append_unique(path, {"source_id": "SRC-STC-0001"}, "source_id")
    with pytest.raises(DuplicateIdError):
        append_unique(path, {"source_id": "SRC-STC-0001"}, "source_id")
```

Also cover:

- strict JSON reads, including duplicate keys, non-object rows, and non-finite
  numbers;
- deep canonical snapshots that reject unsupported values and non-string keys,
  normalize tuples to arrays, and remain unchanged if the caller mutates the
  input while waiting for the lock;
- retry identity for list/tuple equivalence and the distinct JSON scalar
  identities `true`, `1`, `1.0`, `0.0`, and `-0.0`;
- exact-record append retries, mismatched-token conflicts, and
  post-replace/directory-fsync uncertainty;
- bulk compare-and-swap success, stale-writer preservation, and the reverse
  race in which a completed append must survive a writer holding an older
  snapshot;
- multiprocess unique appends, duplicate races, a killed lock holder, orphan
  temporary-file cleanup, and absence of torn/corrupt JSONL.
- cross-destination prefix/glob collisions, data/lock/temp namespace
  collisions, dangling lock symlinks, nonzero or non-regular lock sidecars,
  reserved missing-parent paths, permission preservation, and
  `fchmod -> file fsync -> replace -> directory fsync` ordering;
- package `.gitignore` coverage for the exact hashed registry temporary
  namespace so an orphaned full-registry snapshot cannot enter a broad stage.

- [ ] **Step 2: Confirm failure, implement, and rerun**

```bash
uv run pytest tests/test_jsonl_store.py -q
```

Expected before implementation: FAIL; after implementation: pass.

Use `tempfile.NamedTemporaryFile(dir=path.parent)`, `flush`, `os.fsync`,
`os.replace`, parent-directory `fsync`, UTF-8, `sort_keys=True`, compact JSON
separators, and one POSIX `flock` namespace shared by append and bulk writers.
The package is explicitly Linux/POSIX-only.

- [ ] **Step 3: Commit**

```bash
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: add canonical append-only registry storage"
```

### Task 4: Enforce schemas, cross-references, and evidence boundaries

**Files:**
- Create: `research/sleep-time-compute/schemas/control/common.schema.json`
- Create: `research/sleep-time-compute/schemas/control/source.schema.json`
- Create: `research/sleep-time-compute/schemas/control/claim-history.schema.json`
- Create: `research/sleep-time-compute/schemas/control/paper-card.schema.json`
- Create: `research/sleep-time-compute/schemas/control/evidence.schema.json`
- Create: `research/sleep-time-compute/schemas/control/claim.schema.json`
- Create: `research/sleep-time-compute/schemas/control/question.schema.json`
- Create: `research/sleep-time-compute/schemas/control/hypothesis.schema.json`
- Create: `research/sleep-time-compute/schemas/control/search-protocol.schema.json`
- Create: `research/sleep-time-compute/schemas/control/branch-manifest.schema.json`
- Create: `research/sleep-time-compute/schemas/control/artifact.schema.json`
- Create: `research/sleep-time-compute/schemas/control/manuscript-link.schema.json`
- Create: `research/sleep-time-compute/schemas/control/parity.schema.json`
- Create: `research/sleep-time-compute/schemas/control/publication-asset.schema.json`
- Create: `research/sleep-time-compute/schemas/control/publication-visual-review.schema.json`
- Create: `research/sleep-time-compute/schemas/control/equation.schema.json`
- Create: `research/sleep-time-compute/schemas/control/audit-finding.schema.json`
- Create: `research/sleep-time-compute/schemas/control/audit-adjudication.schema.json`
- Create: `research/sleep-time-compute/schemas/control/terminology.schema.json`
- Create: `research/sleep-time-compute/schemas/control/contradiction.schema.json`
- Create: `research/sleep-time-compute/schemas/control/negative-result.schema.json`
- Create: `research/sleep-time-compute/schemas/control/search-log.schema.json`
- Create: `research/sleep-time-compute/schemas/control/full-read-order.schema.json`
- Create: `research/sleep-time-compute/schemas/control/full-read-log.schema.json`
- Create: `research/sleep-time-compute/schemas/control/experiment-bundle.schema.json`
- Create: `research/sleep-time-compute/schemas/control/g4-execution-snapshot.schema.json`
- Create: `research/sleep-time-compute/schemas/control/distributed-rerun-request.schema.json`
- Create: `research/sleep-time-compute/schemas/control/result-block.schema.json`
- Create: `research/sleep-time-compute/schemas/control/gate-input.schema.json`
- Create: `research/sleep-time-compute/schemas/control/gate.schema.json`
- Create: `research/sleep-time-compute/schemas/control/audit-attestation.schema.json`
- Create:
  `research/sleep-time-compute/schemas/control/gate-evaluation-attestation.schema.json`
- Create: `research/sleep-time-compute/schemas/control/human-approval.schema.json`
- Create: `research/sleep-time-compute/schemas/control/trusted-key-set.schema.json`
- Create: `research/sleep-time-compute/schemas/control/independence-protocol.schema.json`
- Create: `research/sleep-time-compute/schemas/control/candidate.schema.json`
- Create: `research/sleep-time-compute/schemas/control/replay-report.schema.json`
- Create: `research/sleep-time-compute/schemas/control/release-environment.schema.json`
- Create: `research/sleep-time-compute/schemas/control/reproduction-manifest.schema.json`
- Create: `research/sleep-time-compute/schemas/control/sbom-spdx.schema.json`
- Create: `research/sleep-time-compute/schemas/control/release-manifest.schema.json`
- Create: `research/sleep-time-compute/schemas/control/handoff-index.schema.json`
- Create: `research/sleep-time-compute/schemas/control/presentation-handoff.schema.json`
- Create: `research/sleep-time-compute/src/stc_research/validate.py`
- Create: `research/sleep-time-compute/src/stc_research/gate_engine.py`
- Create: `research/sleep-time-compute/src/stc_research/source_rights.py`
- Modify: `research/sleep-time-compute/src/stc_research/cli.py`
- Modify: `research/sleep-time-compute/tests/test_cli.py`
- Create: `research/sleep-time-compute/tests/test_validate.py`
- Create: `research/sleep-time-compute/tests/test_gate_engine.py`
- Create: `research/sleep-time-compute/tests/test_source_rights.py`

**Interfaces:**
- Produces: `ValidationReport` and
  `validate_repository(root: Path, gate: str | None) -> ValidationReport`.
- Produces:
  `evaluate_gate(gate_input: GateEvaluationInput) -> GateRecord`; this is the
  only writer of canonical `manifests/gates/G0.json` through `G8.json`.
- Extends the declared `stc` CLI with `validate`, `gate evaluate`, and
  `gate verify-chain`, and `sources staged-scan`.

- [ ] **Step 1: Write failing repository fixtures**

Create tests for:

```text
evidence references missing source
claim references missing evidence
A-ABS evidence anchor outside abstract
PAPER-C release status unresolved
source marked FULL with no method/result/limitation anchors
duplicate canonical source key
source date later than an earlier registered version
claim manuscript location absent
stale artifact used by a release claim
unknown evidence relation or scalar confidence field
grade X evidence used as release support
A-ABS paper card cites a non-abstract anchor
missing canonical JSON card for a primary source
post-G2 hypothesis amendment still marked confirmatory
negative result omitted from the negative-results registry
manuscript sentence digest differs from its link record
release or presentation handoff depends on a stale artifact
redistribution-disallowed source has a committed full-text artifact
source release artifact lacks license evidence or review disposition
artifact-level rights absent even though its source-level summary allows reuse
claim has mixed-grade refs but no exact qualifying release-support subset
context/counterevidence ref incorrectly counted toward a release threshold
two versions of one work incorrectly counted as independent support
gate skips a predecessor or changes a predecessor digest
unsigned or subject-digest-mismatched audit/human attestation
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_validate.py -q
```

Expected: import or assertion failures.

- [ ] **Step 3: Implement JSON Schema and semantic validation**

`common.schema.json` is the only definition site for `$defs.SubjectRef`,
`$defs.SourceVersion`, `$defs.ArtifactRef`, `$defs.ClaimResultProvenance`,
stable IDs, digests, source grade, review status, warrant, reproduction
strength, result modality, and evidence relation. Every other control schema
references those definitions by canonical `$id`; it may not copy a nested
shape. Tests round-trip the Python models through exported schemas and mutate
every nested field to prove the schema and dataclass reject the same invalid
record.

The validator discovers every `*.schema.json` under `schemas/control/`,
requires a unique `$id` and declared owner/model, and compares the discovered
set with the model export manifest. `schema_inventory=complete` is computed
from that comparison, never from a hard-coded schema count. It validates the
full claim evidence profile rather than checking IDs alone. For each
`ClaimSupportRef`, it resolves and compares the evidence ID, source ID,
reviewed source version and digest, artifact ID, support-span digest, relation,
role, source grade, warrant, and reproduction strength. It then evaluates only
the exact `RELEASE_SUPPORT` subset against `ClaimEvidenceRequirement`,
deduplicating independence by canonical work identity. Mixed-grade support is
valid only when that qualifying subset independently meets every threshold;
raw source/evidence IDs, context, and counterevidence never fill a gap.

Task 4 owns the generic gate engine, typed gate-input/record schemas, canonical
serialization, predecessor verification, and atomic output. Domain plans own
only typed input assembly and domain checks: Task 10 owns G0/G1 inputs, the
benchmark/experiment plans own G2–G4 inputs, Task 13 owns G5/G6 inputs, and the
independent-audit/release workflow owns G7/G8 inputs. No domain command may
hand-write a gate record or bypass `evaluate_gate`.

The engine fixes the predecessor chain as `G0: []`, `G1: [G0]`, and
`G<n>: [G<n-1>]` for G2–G8. It resolves predecessors only from canonical
`manifests/gates/G*.json`, verifies their `PASS` status and digest, binds all
input artifact digests plus the evaluated commit/tree, and writes one canonical
`GateRecord` atomically. G7/G8 additionally bind the frozen scientific
candidate commit/tree; G0–G6 cannot guess that future identity. Re-evaluation
with byte-identical inputs is byte-identical when given the same frozen
evaluation timestamp; a changed input requires a new record and recursively
stales successors.
For G7/G8 the engine requires a non-null `evaluator_attestation_ref`, verifies
the trusted key/role/time/signature, and recomputes its subject digest from the
canonical domain evaluation subject. G8 specifically rejects a visual-only
`HumanApproval` in that field; it requires the separate
`GateEvaluationAttestation` over the complete G8 subject.
`stc validate --gate G1` selects validation rules only; it does not create or
pass G1.

The command:

```bash
uv run stc validate --root . --gate G1
```

must print one stable summary line:

```text
PASS schema_inventory=complete cross_refs=0 duplicate_ids=0 boundary_errors=0 stale_release_nodes=0
```

Exit 2 on validation failure and print one machine-readable JSON diagnostic per
line to stderr.

- [ ] **Step 4: Run and commit**

```bash
uv run pytest \
  tests/test_validate.py \
  tests/test_gate_engine.py \
  tests/test_source_rights.py \
  tests/test_cli.py -q
uv run stc validate --root .
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: enforce evidence and release invariants"
```

### Task 5: Implement the content-addressed artifact DAG

**Files:**
- Create: `research/sleep-time-compute/src/stc_research/digest.py`
- Create: `research/sleep-time-compute/src/stc_research/artifact_dag.py`
- Modify: `research/sleep-time-compute/src/stc_research/cli.py`
- Modify: `research/sleep-time-compute/tests/test_cli.py`
- Create: `research/sleep-time-compute/tests/test_artifact_dag.py`
- Modify: `research/sleep-time-compute/registry/artifacts.jsonl`

**Interfaces:**
- Produces: `sha256_path`, `topological_order`, `refresh_dag`, and
  `assert_release_fresh`.
- Extends the declared `stc` CLI with the exact commands `dag check` and
  `dag import-bundles --input PATH --manifest-dir PATH`. Parser tests cover the
  repeated systems/runtime invocations; semantic tests reject incomplete,
  digest-mismatched, non-manifest-last, and duplicate-ID bundles.

- [ ] **Step 1: Write staleness-propagation tests**

```python
def test_changed_source_stales_all_consumers(tmp_path):
    dag = make_chain("source", "card", "claim", "figure", "paper", "handoff")
    refresh_dag(dag, {"source": "new-digest"})
    assert [dag.nodes[n].status for n in
            ["card", "claim", "figure", "paper", "handoff"]] == ["STALE"] * 5


def test_cycle_is_rejected():
    with pytest.raises(DagCycleError):
        topological_order(make_cycle("A", "B", "A"))
```

- [ ] **Step 2: Implement canonical digests**

Directories hash the sorted tuple `(relative POSIX path, file sha256)`.
JSON hashes canonical compact JSON with sorted keys. Symlinks are rejected in
release inputs.

Every artifact node records:

```text
artifact_id
schema_version
artifact_type
path
input_digests
producer_command
output_digest
consumers
claim_ids
question_ids
gate_status
owner
frozen_release_tag
```

- [ ] **Step 3: Add CLI and run**

```bash
uv run pytest tests/test_artifact_dag.py tests/test_cli.py -q
uv run stc dag check
```

Expected: test pass and `PASS nodes=<n> stale=0 cycles=0 missing=0`.

- [ ] **Step 4: Commit**

```bash
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: add content-addressed artifact staleness graph"
```

### Task 6: Export a separate Veridraft bundle

**Files:**
- Create: `research/sleep-time-compute/src/stc_research/export_veridraft.py`
- Modify: `research/sleep-time-compute/src/stc_research/cli.py`
- Modify: `research/sleep-time-compute/tests/test_cli.py`
- Create: `research/sleep-time-compute/tests/test_export_veridraft.py`
- Create: `research/sleep-time-compute/veridraft.config.json`
- Create: `claims/sleep-time-compute.bundle.json`

**Interfaces:**
- Consumes: validated claim/evidence/result records.
- Produces: CAW-02 bundle `sleep-time-compute-2026`.
- Extends the declared `stc` CLI with the exact command
  `claims export-veridraft --output PATH`; `test_export_veridraft.py` owns its
  semantics and `test_cli.py` owns the parser contract.

- [ ] **Step 1: Write mapping tests**

Assert:

```text
ORIGINAL-MEASUREMENT → P1 with result_refs
INDEPENDENT-REPLICATION → P1 with result_refs
RERUN → P1 with result_refs, but never independent external-validity status
SOURCE-SUMMARY → P2 with source_artifact evidence
VENDOR-BEHAVIOR → P2 for documented existence/API behavior only
SYNTHESIS → P2 with at least two independent source artifacts
ANALYTIC-DERIVATION → P1 with derivation and unit/property-test artifacts
HYPOTHESIS → excluded from release bundle
TRACE-SIMULATION → P1 only when the statement says simulated/analytical
```

Also assert the exporter never reads or writes `claims/bundle.json`.
Reject product-document effectiveness claims, RERUN-only generalization claims,
and numerical cross-paper comparisons whose accounting/environment records are
not harmonized.

- [ ] **Step 2: Implement exporter and config**

Use:

```json
{
  "gate_profile": "neurips-paper",
  "data_dir": ".veridraft-stc"
}
```

Every `source_artifact` ref has the form `<repo-relative-path>@<git-sha>`.

- [ ] **Step 3: Run isolated Veridraft gate**

```bash
uv run pytest tests/test_export_veridraft.py tests/test_cli.py -q
uv run stc claims export-veridraft \
  --output ../../claims/sleep-time-compute.bundle.json
uv run --group publication veridraft \
  --config veridraft.config.json \
  --data-dir .veridraft-stc \
  import-bundle ../../claims/sleep-time-compute.bundle.json
uv run --group publication veridraft \
  --config veridraft.config.json \
  --data-dir .veridraft-stc \
  gate sleep-time-compute-2026
```

Expected during scaffolding: the exporter emits zero release claims and the
local validator passes; Veridraft import/gate is first required after supported
claims exist.

- [ ] **Step 4: Commit**

```bash
git -C ../.. add research/sleep-time-compute claims/sleep-time-compute.bundle.json
git -C ../.. commit -m "research: isolate sleep-time claim gating"
```

#### Early-tooling checkpoint — execute Task 14 Phase A now

Despite Task 14's numerical placement beside the release workflow, its Phase A
is a prerequisite for Tasks 7–13. Immediately after Task 6, implement and
commit the fixture-only candidate, release-assembly, attestation, and
presentation-handoff tooling and tests specified in Task 14 Phase A. It may
consume only synthetic fixtures and canonical schemas; it must not freeze a
real candidate or emit a public release. Do not begin Task 7 until those tests
pass. Task 14 Phase B is invoked only after the real G6 candidate, independent
G7 audit, and G8 inputs exist.

### Task 7: Bootstrap Wave 01 into typed records

**Files:**
- Create: `research/sleep-time-compute/src/stc_research/ingest_dossier.py`
- Create: `research/sleep-time-compute/src/stc_research/ingest_wave01.py`
- Modify: `research/sleep-time-compute/src/stc_research/cli.py`
- Modify: `research/sleep-time-compute/tests/test_cli.py`
- Create: `research/sleep-time-compute/tests/test_ingest_dossier.py`
- Create: `research/sleep-time-compute/tests/test_ingest_wave01.py`
- Create: `research/sleep-time-compute/registry/source-id-map.json`
- Modify: `research/sleep-time-compute/registry/sources.jsonl`
- Create: `research/sleep-time-compute/cards/SRC-STC-0001.json` through `SRC-STC-0027.json`
- Create: `research/sleep-time-compute/cards/generated/SRC-STC-0001.md` through `SRC-STC-0027.md`

**Interfaces:**
- Consumes: `dossier/SLEEP-TIME-COMPUTE-PRE-RESEARCH.md` and
  `EVIDENCE-INTAKE-WAVE-01.md`.
- Produces: a Wave-specific increment/reconciliation report covering 27
  Wave 01 rows, 27 canonical JSON cards, and 27 deterministic Markdown views,
  plus reconciled candidate source records from the dossier. Original `W1-*`
  IDs are retained as `legacy_ids`.
  Stable source IDs never encode chronology; chronological order is a generated
  view by verified earliest-public date.
- Extends the declared `stc` CLI with exact commands
  `ingest-dossier --input PATH` and `ingest-wave01 --input PATH`; the two
  ingestion tests own behavior and `test_cli.py` pins both parser surfaces.

- [ ] **Step 1: Write dossier reconciliation tests**

Parse the dossier chronology/reference blocks into candidate identities, then
reconcile DOI, arXiv, OpenReview, title/version, product, and URL aliases before
allocating IDs through `source-id-map.json`. Test that a newly discovered older
paper receives a new ID without renumbering any existing record and that the
chronological view still sorts correctly.

The dossier is an intake source, not a warrant: it can seed identity, topic,
affiliation, and inclusion reason, but cannot create an evidence span supporting
a release claim.

- [ ] **Step 2: Write Wave 01 count and fidelity tests**

```python
def test_wave01_bootstrap_count(tmp_path):
    report = ingest_wave01(WAVE01, tmp_path)
    assert report.sources == 27
    assert report.canonical_cards == 27
    assert report.rendered_cards == 27
    assert report.duplicate_canonical_keys == 0


def test_wave01_keeps_review_status(tmp_path):
    ingest_wave01(WAVE01, tmp_path)
    records = load_sources(tmp_path / "registry/sources.jsonl")
    assert by_key(records, "arxiv:2607.17545").review_status == "FULL"
    assert by_key(records, "arxiv:2605.20616").review_status == "A-ABS"
```

- [ ] **Step 3: Implement deterministic parsers**

The Wave 01 parser reads only the chronological Markdown table and card
headings. It
does not infer authors, affiliations, or body claims absent from the intake.
Unpopulated optional fields are omitted, not filled with invented text.
Canonical JSON is written first; Markdown cards are generated only from it.

- [ ] **Step 4: Run reconciliation, bootstrap, and validation**

```bash
uv run pytest tests/test_ingest_dossier.py tests/test_ingest_wave01.py \
  tests/test_cli.py -q
uv run stc ingest-dossier \
  --input ../../dossier/SLEEP-TIME-COMPUTE-PRE-RESEARCH.md
uv run stc ingest-wave01 --input EVIDENCE-INTAKE-WAVE-01.md
uv run stc validate --root .
```

Expected: dossier identities reconcile with Wave 01, and the Wave-specific
report says `wave_sources=27 canonical_cards=27 rendered_cards=27`, with zero
duplicate works, ID changes, view drift, or chronology errors.

- [ ] **Step 5: Commit**

```bash
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: structure evidence intake wave 01"
```

### Task 8: Vendor and checksum 25 high-risk works and their versions

**Files:**
- Modify: `research/sleep-time-compute/.gitignore`
- Create when redistribution is positively permitted:
  `papers/sleep-time-compute/primary/*.pdf`
- Create when redistribution is positively permitted:
  `papers/sleep-time-compute/text/*.txt`
- Create: `papers/sleep-time-compute/checksums.sha256`
- Create: `research/sleep-time-compute/manifests/source-fetch.json`
- Create: `research/sleep-time-compute/manifests/search-protocol.json`
- Create: `research/sleep-time-compute/registry/search-log.jsonl`
- Modify: `research/sleep-time-compute/registry/artifacts.jsonl`
- Create: `research/sleep-time-compute/src/stc_research/source_manifest.py`
- Modify: `research/sleep-time-compute/src/stc_research/source_rights.py`
- Modify: `research/sleep-time-compute/src/stc_research/cli.py`
- Modify: `research/sleep-time-compute/tests/test_cli.py`
- Create: `research/sleep-time-compute/tests/test_source_manifest.py`
- Modify: `research/sleep-time-compute/tests/test_source_rights.py`

**Interfaces:**
- Produces: immutable local warrants for the G1 full-read wave.
- Extends the declared `stc` CLI with `sources fetch` and `sources verify`, and
  exercises the Task 4 `sources staged-scan` command against real manifests.
  `test_cli.py` pins the exact manifest/cache/release-root options below.

- [ ] **Step 1: Freeze the formal search-reproduction protocol**

Record cutoff `2026-07-25`, databases, literal queries, inclusion/exclusion
criteria, corporate/product-document policy, citation-snowball rule, and the
stopping condition “two consecutive snowball rounds yield no new direct
neighbor.” The dossier and Wave 01/02 work performed before this protocol are
labelled exploratory seed discovery with their real timestamps; they are never
misrepresented as prospectively preregistered. Freeze and hash a separate
formal reproduction protocol before re-running the canonical searches. Formal
search rounds append query, timestamp, result count, inclusion decision, and
snowball parent to `search-log.jsonl` without changing the frozen protocol
digest.

- [ ] **Step 2: Freeze the source manifest**

The manifest contains these canonical keys:

```text
arxiv:2504.13171
arxiv:2605.20616
arxiv:2606.10616
arxiv:2605.16045
arxiv:2606.04536
arxiv:2606.04557
arxiv:2605.26099
arxiv:2602.03036
arxiv:2602.15902
arxiv:2411.05877
arxiv:2606.03979
arxiv:2606.06448
arxiv:2606.25161
arxiv:2607.17545
arxiv:1710.10368
openreview:SJ1Xmf-Rb
pmlr:schwarz18a
arxiv:2303.10725
arxiv:2401.08623
arxiv:2409.16391
neurips2025:d7e5870810331da5a8ac8bd16d42e074
arxiv:2605.08538
arxiv:2605.12978
arxiv:2607.08032
arxiv:2607.11020
```

Each work entry contains immutable version entries, and every downloadable or
generated artifact in each version has its own `ArtifactRef`: exact artifact
ID, URL/path, digest, media type, license ID, license-evidence URL,
license-evidence digest, `redistribution_allowed`, rights reviewer/date/expiry,
and review disposition. A
source-level rights summary is derived only and never authorizes a sibling
artifact. The work key `arxiv:2606.03979` has an explicit arXiv `v2` version
and the earlier OpenReview `iiZy6xyVVE` version; each artifact is fetched and
hashed separately so method changes remain auditable rather than collapsed.

Fetching and version pinning do not imply redistribution permission. Store
nonredistributable or license-unclear full texts only in the ignored,
content-addressed `.cache/stc/sources/`; commit their authoritative URLs,
expected digests, metadata, and registered short evidence summaries/anchors,
not the PDF or full-text extract. Copy a PDF/text extract into
`papers/sleep-time-compute/` only when the reviewed license disposition
explicitly permits it. Public release and presentation export independently
reapply the same allowlist.

Assert that `.cache/stc/sources/` is matched by the committed package
`.gitignore`. Tests must prove that a permitted metadata record does not
authorize an unreviewed PDF/text sibling, a source-level allow flag cannot
override an artifact-level denial, and unknown/expired rights dispositions fail
closed.

- [ ] **Step 3: Test manifest uniqueness and title verification**

```bash
uv run pytest tests/test_source_manifest.py tests/test_source_rights.py \
  tests/test_cli.py -q
git -C ../.. check-ignore \
  research/sleep-time-compute/.cache/stc/sources/rights-probe.pdf
uv run stc sources fetch \
  --manifest manifests/source-fetch.json \
  --cache .cache/stc/sources
uv run stc sources verify \
  --manifest manifests/source-fetch.json \
  --cache .cache/stc/sources \
  --release-root ../../papers/sleep-time-compute
```

Expected: 25 primary-work records, every declared version artifact and text
extract available in the verified cache, title/version lineage matches for all,
and every committed artifact has `redistribution_allowed=true` with a
resolvable license-review record. Unknown/false dispositions remain cache-only;
the manifest-aware checksum verification over the permitted release tree exits
0, including the valid zero-redistributable-artifact case.

- [ ] **Step 4: Commit**

Stage the intended paths, then scan the actual Git index rather than only the
working tree. `staged-scan` enumerates every staged blob, detects full text by
content/media type as well as extension, resolves it to the exact artifact ID
and rights record, and fails on cache paths, missing records, nonpositive
permission, missing license evidence, or stale rights review.

```bash
git -C ../.. add papers/sleep-time-compute \
  research/sleep-time-compute/manifests \
  research/sleep-time-compute/.gitignore \
  research/sleep-time-compute/registry/artifacts.jsonl \
  research/sleep-time-compute/registry/sources.jsonl
uv run stc sources staged-scan --repo-root ../.. --cached
git -C ../.. diff --cached --check
git -C ../.. commit -m "research: pin high-risk sleep-time primary sources"
```

Before commit, both `stc sources verify --release-root` and the staged-index
scan must fail if any staged or release artifact lacks positive, per-artifact
redistribution permission.

### Task 9: Complete the closest-neighbor full-read wave

**Files:**
- Modify: `research/sleep-time-compute/cards/*.json`
- Modify: `research/sleep-time-compute/cards/generated/*.md`
- Create: `research/sleep-time-compute/extracts/*.json`
- Modify: `research/sleep-time-compute/registry/sources.jsonl`
- Modify: `research/sleep-time-compute/registry/evidence.jsonl`
- Modify: `research/sleep-time-compute/registry/claims.jsonl`
- Modify: `research/sleep-time-compute/registry/contradictions.jsonl`
- Create: `research/sleep-time-compute/src/stc_research/promote_qa.py`
- Create: `research/sleep-time-compute/src/stc_research/full_read_order.py`
- Modify: `research/sleep-time-compute/src/stc_research/cli.py`
- Modify: `research/sleep-time-compute/tests/test_cli.py`
- Create: `research/sleep-time-compute/tests/test_promote_qa.py`
- Modify: `research/sleep-time-compute/registry/negative-results.jsonl`
- Create: `research/sleep-time-compute/registry/full-read-log.jsonl`
- Create: `research/sleep-time-compute/manifests/requested-lineages.json`
- Create: `research/sleep-time-compute/manifests/full-read-order.json`
- Create: `research/sleep-time-compute/reports/closest-neighbor-matrix.md`
- Create: `research/sleep-time-compute/reports/requested-lineage-matrix.md`
- Create: `research/sleep-time-compute/reports/corporate-attribution-audit.md`
- Create: `research/sleep-time-compute/reports/blog-claim-audit.md`
- Create: `research/sleep-time-compute/tests/test_full_read_order.py`

**Interfaces:**
- Produces: one independently reviewable card and evidence extract per source.
- Extends the declared `stc` CLI with `sources full-read-order build` and
  `sources full-read-order validate`, plus the read-only
  `qa promote --input PATH --output PATH` proposal command implemented by
  `promote_qa.py` and covered by `test_promote_qa.py`; `test_cli.py` pins all
  three exact parser surfaces.

- [ ] **Step 1: Use this exact canonical paper-card contract**

The canonical JSON records reviewed source/version digest, reviewer/date,
research question, wake input, trigger, operator, destination, training data,
objective, optimizer, update location, cadence, capacity policy, system
assumptions, exact results, limitations, failure modes, code/data review,
evidence IDs, counterevidence IDs, and explicit non-claims. An `A-ABS` card may
reference abstract anchors only.

Generate the following human-readable view:

Every card contains:

```markdown
# <source ID> — <title>

## Identity and versions
## Problem and estimand
## Memory medium and state transition
## Wake/sleep boundary and information cutoff
## Training data construction
## Objective, optimizer, and update location
## Evaluation protocol and exact reported results
## Capacity, forgetting, staleness, and deletion
## Systems and lifecycle accounting
## Assumptions and validity threats
## What this preempts
## What it does not establish
## Reproduction/code/data status
## Exact evidence anchors
## Open questions
```

Each exact result has an `EV-STC-*` anchor. Every “does not establish” statement
is scoped to inspected material, not absence beyond the search.

- [ ] **Step 2: Freeze and follow one chronological read manifest**

Before beginning Task 9's full-read audit, encode every required
direct-neighbor work, material version, requested-lineage paper, and captured
official product version in `requested-lineages.json`, then generate
`manifests/full-read-order.json`. Each read unit records order index, source ID,
canonical work key, version ID, artifact ID/digest, earliest verified public
date/precision, version public date, and inclusion set. A top-level
`order_payload_sha256` hashes the canonical ordered item array while excluding
the digest field itself. Sort
globally by the read unit's verified version-public date (or the work's
earliest verified public date when no distinct version date exists), then date
precision, canonical key, and version ID; topical category, later venue date,
source ID, and discovery order never override that order. Multiple versions of
a work remain separate units and need not be adjacent when another work's
public date falls between them. A pre-existing `FULL` intake label does not
exempt a unit from ordered Task 9 re-verification.

Every completed review appends its order index, `order_payload_sha256`, exact
artifact digest, reviewer, completion timestamp, and resulting card/evidence IDs to
`registry/full-read-log.jsonl`. The log must be a gap-free prefix of the
manifest while work is in progress and an exact complete sequence before G1.
Discovering a newly required older work freezes a new manifest revision before
any further reading; it never silently inserts a row into an already executed
order.

```bash
uv run pytest tests/test_full_read_order.py tests/test_promote_qa.py \
  tests/test_cli.py -q
uv run stc sources full-read-order build \
  --sources registry/sources.jsonl \
  --direct-neighbors manifests/source-fetch.json \
  --requested-lineages manifests/requested-lineages.json \
  --output manifests/full-read-order.json
uv run stc sources full-read-order validate \
  --order manifests/full-read-order.json \
  --log registry/full-read-log.jsonl \
  --allow-incomplete-prefix
```

- [ ] **Step 3: Audit external consolidation/retention coverage**

Produce audited cards for Auto-Dreamer, OSL-MR, RecMem, Agent Memory, TrustMem,
Retain or Consolidate, Useful Memories Become Faulty, and the Microsoft
Human-Inspired Memory Architecture. This and the next topical checkboxes are
completeness audits after the corresponding chronological read units, not
permission to read by topic.

- [ ] **Step 4: Audit parametric and context-to-adapter coverage**

Produce audited cards for TMEM, Doc-to-LoRA, Generative Adapter, Language
Models Need Sleep, and Can a Language Model Learn Facts Continually in Its
Weights. The Google paper compares the OpenReview manuscript with arXiv
`2606.03979` version `v2` and records every material method/result change.

- [ ] **Step 5: Audit latent/KV coverage**

Produce audited cards for Cartridges at Scale, Offline Recurrence, and
LatentMem.

- [ ] **Step 6: Audit the founding deferred-compute paper**

Produce a full card for arXiv:2504.13171, *Sleep-time Compute: Beyond Inference
Scaling at Test-time*. Separate deferred query-anticipating identity work from
semantic memory consolidation and record the exact multi-query amortization
setup. Its actual reading position is determined by the chronological manifest,
not by this plan section's placement.

- [ ] **Step 7: Audit strict predecessors and the closest theory collision**

Produce audited cards for DGDMN, FearNet, Progress & Compress, SIESTA,
Wake-Sleep Consolidated Learning, PCMC, the Spens--Burgess--Behrens offline
processing controller, and the Rate--Distortion View of Memory Compaction.
For each, record the exact wake/sleep boundary, trigger, replay selector,
destination, finite-capacity behavior, and whether control is learned. These
works explicitly preempt claims to the first wake/sleep cycle, first learned
sleep controller, first replay selector, and first cross-layer compaction
objective.

- [ ] **Step 8: Complete the user-requested chronology and product lineage**

The 25 direct neighbors are the novelty-risk core, not the whole requested
atlas. Promote every paper in these named lineages to `FULL` when its body
supports a manuscript/monograph assertion; inspect official product/research
documentation to `A-HTML` with capture date/version, and keep vendor
effectiveness claims separate:

```text
biological/early sleep:
  Golden et al. 2019 preprint → 2022 PLOS Computational Biology article
  PAD (2022)
  Tadros et al. Nature Communications sleep-like replay (2022)
  2016 synaptic consolidation and 2018 prioritized replay basis

algorithmic wake/sleep predecessors:
  DGDMN, FearNet, Progress & Compress, CLEAR, SIESTA, WSCL, PCMC
  Spens et al. offline-processing control with RL

claimed dream line:
  Dream-Augmented Neural Networks SSRN manuscript
  MyGO (registered grade `X`/withdrawn and usable only as a non-claim failure
  warning) and other replay/distillation comparators used in the synthesis

external/product line:
  MemGPT (2023)
  Letta Sleep-time Compute paper, 2025 sleep-agent article, and 2026
    context-repository/code/memory-model documents as distinct versions
  Mem0 ECAI paper and Mem0 V3 behavior as distinct records
  Zep/Graphiti paper+OSS and managed Zep Observations as distinct records

Meta line:
  Product-Key Memory, Expire-Span, Memory Layers at Scale,
  Sparse Memory Finetuning, PAHF, and any search-discovered direct Meta
  successor

Google/DeepMind line:
  complementary-learning-systems basis, Titans, Miras, Atlas, TNT,
  Nested Learning, Memory Caching, Language Models Need Sleep
  OpenReview→arXiv lineage, ReasoningBank, and NSTM as an infrastructure
  analogue

Microsoft line:
  LongMem, Generative Adapter, Memora, Human-Inspired Memory Architecture,
  LEGOMem, ACON, MAGE, and official Foundry Agent Memory documentation

capacity, negative-result, and theory line:
  Useful Memories Become Faulty, Rate–Distortion View of Memory Compaction,
  Can a Language Model Learn Facts Continually in Its Weights, MemDefrag
```

For each record, distinguish earliest public date from later venue/product
date, author affiliation from corporate ownership, peer review from preprint,
paper method from current product behavior, and actual offline/scheduled sleep
from wake-time or ingest-time updates. The DANN record remains grade D unless
new primary evidence changes it; its title is not accepted as a result.
The PLOS item is recorded as a journal article—not a “PLOS conference”—and its
2019 preprint establishes idea chronology.

`requested-lineage-matrix.md` compares wake input, trigger, operator,
destination, training data/objective, cadence, capacity/forgetting, and
systems assumptions. `corporate-attribution-audit.md` gives a source-backed
Google/Meta/Microsoft attribution and lists every “not sleep” boundary. Product
documentation may establish versioned existence/behavior only, never
independent quality.

Treat the initiating Naver post as a versioned secondary pointer, not a
scientific warrant. `blog-claim-audit.md` maps every identifiable statement
(including “2022 PLOS conference,” DANN, Meta, and product-memory claims) to
the corrected primary record, disposition
`CONFIRMED|QUALIFIED|CONTRADICTED|UNRESOLVED`, exact evidence IDs, and the
needed correction. The paper cites primary sources; the Korean companion may
discuss this audit explicitly.

- [ ] **Step 9: Validate the full-read corpus**

```bash
uv run stc sources full-read-order validate \
  --order manifests/full-read-order.json \
  --log registry/full-read-log.jsonl \
  --require-complete
uv run stc validate --root .
```

Expected: all 25 high-risk work records and every manuscript-used paper in the
named-request lineages have `FULL` status, while versioned official product
records have at least `A-HTML`; all carry exact
method/result/limitation/supplement anchors, version/checksum, reviewer,
code/data inspection status, non-claim, and counterevidence; every negative or
null result has a separate ledger record. A missing full text or uninspected
material version blocks the later G1 gate. The requested-lineage report has no
unresolved PLOS/DANN/MemGPT/Letta/Mem0/Zep/Google/Meta/Microsoft identity.
The chronological log exactly matches every ordered artifact digest with no
gap, duplicate, reordering, or topical batch substitution.

- [ ] **Step 10: Commit**

```bash
git -C ../.. add research/sleep-time-compute
git -C ../.. commit -m "research: full-read closest sleep-time compute neighbors"
```

### Task 10: Freeze the research-question, hypothesis, and novelty registries

**Files:**
- Modify: `research/sleep-time-compute/registry/questions.jsonl`
- Modify: `research/sleep-time-compute/registry/hypotheses.jsonl`
- Modify: `research/sleep-time-compute/registry/claims.jsonl`
- Modify: `research/sleep-time-compute/registry/claim-history.jsonl`
- Modify: `research/sleep-time-compute/registry/terminology.jsonl`
- Modify: `research/sleep-time-compute/registry/contradictions.jsonl`
- Create: `research/sleep-time-compute/reports/novelty-audit.md`
- Modify: `research/sleep-time-compute/manifests/search-protocol.json`
- Modify: `research/sleep-time-compute/registry/search-log.jsonl`
- Create: `research/sleep-time-compute/generated/search-protocol.md`
- Create: `research/sleep-time-compute/manifests/publication-branches.json`
- Create: `research/sleep-time-compute/manifests/gate-inputs/G0.json`
- Create: `research/sleep-time-compute/manifests/gate-inputs/G1.json`
- Create: `research/sleep-time-compute/manifests/gates/G0.json`
- Create: `research/sleep-time-compute/manifests/gates/G1.json`
- Create: `research/sleep-time-compute/src/stc_research/gate_inputs_g0_g1.py`
- Create: `research/sleep-time-compute/src/stc_research/hypotheses.py`
- Create: `research/sleep-time-compute/src/stc_research/branches.py`
- Modify: `research/sleep-time-compute/src/stc_research/cli.py`
- Modify: `research/sleep-time-compute/tests/test_cli.py`
- Create: `research/sleep-time-compute/tests/test_gate_inputs_g0_g1.py`
- Create: `research/sleep-time-compute/tests/test_hypotheses_cli.py`
- Create: `research/sleep-time-compute/tests/test_branches_cli.py`

**Interfaces:**
- Produces: typed G0/G1 input manifests and `GateRecord`s plus a timestamped
  G2-ready hypothesis digest. The domain assembler cannot write gate results;
  it calls the Task 4 engine.
- Extends the declared `stc` CLI with the exact command contracts
  `gate inputs assemble G0|G1`, `hypotheses freeze --output PATH`, and
  `branches validate MANIFEST`. The last two commands have parser-level tests
  for their exact positional/options surface plus mutation tests for post-freeze
  hypothesis edits and duplicate publication ownership.

- [ ] **Step 1: Register RQ1–RQ12 and H-STC-001–007**

First run a tested, read-only promotion scan over `../../qa-memory/qa.jsonl`.
`stc qa promote` may propose but never edit qa-memory. It rejects a proposed
hypothesis without a unique H-ID, evidence anchor, estimand, null/rejection
rule, validity-failure rule, owner, or artifact dependency, and writes a
reviewable proposal file before registry insertion.

Every question record contains:

```text
estimand
null
rejection_rule
minimum_effect
inconclusive_condition
validity_failure
owner
artifact_dependencies
paper_scope
```

Every hypothesis contains assumptions, derivation artifact, falsifier,
confirmatory cells, and amendment history.

Any amendment after the frozen G2 digest automatically changes the affected
record to exploratory and recursively stales its confirmatory result/claim
consumers.

```bash
uv run pytest tests/test_promote_qa.py -q
uv run stc qa promote \
  --input ../../qa-memory/qa.jsonl \
  --output generated/qa-promotion-proposals.json
```

- [ ] **Step 2: Register PAPER-C1–C4 verbatim from the approved design**

The C4 record must state that its inference is cell-balanced marginal and not
regime-universal.

Store their pre-data G0 revisions in `claim-history.jsonl`. Before G2,
`publication-branches.json` assigns every claim, experiment, result block,
figure, and table to exactly one paper. Headline results cannot be duplicated
across simultaneous paper branches.

- [ ] **Step 3: Verify and render the frozen search protocol**

Verify the Task 8 protocol digest is unchanged, the append-only search log
records every query and snowball round, and the declared stopping rule has
actually fired or remains explicitly open. Render the JSON protocol plus log
as a Markdown view. The novelty audit treats search absence as a falsifiable
hypothesis and lists every remaining database/coverage limitation.

- [ ] **Step 4: Validate and digest**

The G0 input manifest binds questions, pre-data claim/history revisions,
hypotheses, terminology, publication-branch ownership, the Task 11–12 Phase A
immutable paper/companion scaffold snapshot and parity-map digests, and their
artifact-DAG nodes. The scaffold command therefore runs after claim registration
but before G0 input assembly. G0 binds
`publication/scaffolds/G0/manifest.json`, its preserved bytes, and the
snapshot-only build/trace/parity report
`publication/scaffolds/G0-verification.json`, not the live manuscript paths
that Phase B intentionally fills later. The verification report is admissible
only when it records `PASS`, identical
`snapshot_manifest_sha256_before`/`snapshot_manifest_sha256_after`, identical
`snapshot_inventory_sha256_before`/`snapshot_inventory_sha256_after`, and a
`copied_source_inventory_sha256` equal to that same inventory. The G1 input
manifest binds the
canonical G0 record plus source/version
records, per-artifact rights records, cards, evidence/counterevidence,
source-fetch/search protocol and append-only log, the complete chronological
read manifest/log, novelty audit, and requested-lineage audits. Both manifests
use typed digest refs; a path or ID without its digest is invalid.

```bash
uv run pytest tests/test_gate_inputs_g0_g1.py tests/test_gate_engine.py \
  tests/test_hypotheses_cli.py tests/test_branches_cli.py tests/test_cli.py -q
uv run stc validate --root .
uv run stc hypotheses freeze \
  --output manifests/hypotheses-G2.json
uv run stc branches validate manifests/publication-branches.json
uv run stc gate inputs assemble G0 \
  --root . \
  --output manifests/gate-inputs/G0.json
uv run stc gate evaluate G0 \
  --inputs manifests/gate-inputs/G0.json \
  --output manifests/gates/G0.json
uv run stc validate --root . --gate G1
uv run stc gate inputs assemble G1 \
  --root . \
  --predecessor manifests/gates/G0.json \
  --output manifests/gate-inputs/G1.json
uv run stc gate evaluate G1 \
  --inputs manifests/gate-inputs/G1.json \
  --output manifests/gates/G1.json
uv run stc gate verify-chain --through G1 --root manifests/gates
sha256sum manifests/hypotheses-G2.json
```

Expected: unique IDs, all falsifiers executable, no branch collision, frozen
search protocol, no PAPER-C claim relies on an analogical or product-only
source, G0 and G1 are typed `PASS` records, and G1 contains the exact digest of
the canonical G0 predecessor. A validator summary alone never counts as a
passed gate.

### Task 11: Create the English paper skeleton and trace checker

**Execution timing:** Phase A runs during the G0 freeze, after claim
registration and before G0 input assembly, and creates a buildable,
non-assertive scaffold. Phase B runs only after G5 and replaces
registered result slots with adjudicated result blocks and caveats. This split
enforces the design requirement that the paper skeleton exists from G0 without
allowing pre-result prose to imply a positive finding.
The exact Phase-A interleave is Task 11 A.1–A.2, Task 12 A.1–A.2, Task 11 A.3,
then Task 12 A.3. Task 11 A.3 is the shared finalizer because it snapshots and
offline-verifies both language trees only after the Korean structure/parity
implementation exists.

**Files:**
- Create: `paper-en/sleep-time-compute/main.tex`
- Create: `paper-en/sleep-time-compute/sections/{abstract,introduction,related,method,benchmark,experiments,systems,limitations,conclusion}.tex`
- Create: `paper-en/sleep-time-compute/references.bib`
- Create: `study-kr/sleep-time-compute/back/99-bibliography.md`
- Create: `research/sleep-time-compute/publication/bibliography/manifest.json`
- Create: `paper-en/sleep-time-compute/Makefile`
- Create: `research/sleep-time-compute/publication/links/manuscript-links.jsonl`
- Create: `research/sleep-time-compute/publication/assets/figures.jsonl`
- Create: `research/sleep-time-compute/publication/assets/tables.jsonl`
- Create: `research/sleep-time-compute/publication/assets/equations.jsonl`
- Create:
  `research/sleep-time-compute/publication/scaffolds/G0/{manifest.json,english/**,korean/**,parity.jsonl}`
- Create:
  `research/sleep-time-compute/publication/scaffolds/G0-verification.json`
- Create at G8: `research/sleep-time-compute/publication/reviews/human-visual-approval.json`
- Create: `research/sleep-time-compute/src/stc_research/publication_scaffold.py`
- Create: `research/sleep-time-compute/src/stc_research/publication_fill.py`
- Create: `research/sleep-time-compute/src/stc_research/publication.py`
- Create: `research/sleep-time-compute/src/stc_research/publication_visual_qa.py`
- Modify: `research/sleep-time-compute/src/stc_research/cli.py`
- Modify: `research/sleep-time-compute/tests/test_cli.py`
- Create: `research/sleep-time-compute/tests/test_publication_scaffold.py`
- Create: `research/sleep-time-compute/tests/test_publication_fill.py`
- Create: `research/sleep-time-compute/tests/test_publication_trace.py`
- Create: `research/sleep-time-compute/tests/test_publication_visual_qa.py`
- Create: `research/sleep-time-compute/tests/fixtures/publication-qa/*`

**Interfaces:**
- Produces: buildable paper skeleton with claim markers
  `\stcclaim{PAPER-C1}{...}`, source markers `\stccite{SRC-STC-0001}`, and
  stable paired material-block markers.
- Extends the declared `stc` CLI with exact commands
  `publication scaffold materialize --claims PATH --english-root PATH
  --korean-root PATH --parity PATH --mode g0-placeholders`,
  `publication scaffold snapshot --english-root PATH --korean-root PATH
  --parity PATH --snapshot-root PATH`, and
  `publication scaffold verify --snapshot-root PATH --output PATH --offline`.
  Materialization is idempotent and refuses to overwrite a non-placeholder
  result block. Snapshot runs only after all deterministic bibliography/source
  rendering and refuses to overwrite any existing snapshot byte with a
  different digest.
- The verify parser accepts no positional arguments and requires exactly one
  `--snapshot-root`, exactly one `--output`, and `--offline`; unknown mutation
  surfaces such as `--in-place`, `--write-snapshot`, or `--work-root` are
  parser errors. At runtime, `--output` must resolve outside
  `--snapshot-root`. Exit `0` means the JSON report has `status: "PASS"` and
  all immutable-input, English build/trace, Korean build/trace, and parity
  checks passed. A malformed manifest, undeclared or missing source, pre/post
  digest drift, output path inside the snapshot, attempted snapshot write, or
  any build/trace/parity error exits nonzero and cannot emit a `PASS` report.
- Also owns `bibliography render --sources PATH --bib PATH --korean PATH
  --output PATH`, `publication inventory --root PATH --entry PATH --output
  PATH`, `publication trace --source PATH --links PATH --output PATH`, and the
  fixture/real
  `publication visual-qa` argument surface shown below. `publication.py`,
  `publication_fill.py`,
  `publication_visual_qa.py`, `test_publication_trace.py`,
  `test_publication_fill.py`, `test_publication_visual_qa.py`, and
  `test_cli.py` jointly implement and pin those exact commands.

- Post-G5 filling is a separate exact command:
  `publication fill-results --g5 PATH --claims PATH --history PATH
  --result-blocks PATH --g0-snapshot PATH --english-root PATH --korean-root
  PATH --parity PATH --links PATH --output PATH`. The parser accepts each
  option exactly once and no positional or mutation-only bypass option. The
  command rehashes G5, every terminal claim revision, every referenced
  `ResultBlock`, the immutable G0 snapshot, both live source inventories,
  parity records, and manuscript-link records before and after one atomic
  paired EN/KO replacement. It writes a machine fill report only after both
  trees and all link records pass. It cannot write inside the G0 snapshot,
  cannot mutate it, and cannot leave one language updated without the other.
  Exit zero requires no `PENDING_G5`, no unregistered numeric assertion, exact
  status/scope/interval/caveat transfer, and exact G0-lineage retention.

- [ ] **Phase A.1: Write scaffold and trace tests**

Assert every `\stcclaim{}` ID exists, every citation key maps to a source, every
figure/table ID is owned by one paper, and abstract/conclusion contain no
`HYPOTHESIS` or `UNRESOLVED` claim.

Every material block is delimited in TeX comments:

```text
% STC:BEGIN EN-S0001
...material sentence or paragraph...
% STC:END EN-S0001
```

`manuscript-links.jsonl` maps the block to claim IDs, citations, caveats,
normalized text digest, result blocks, and artifact dependencies. Duplicate,
nested, unmatched, or digest-drifted markers fail.

Pin the G0 immutability contract with these named tests:

- `test_scaffold_verify_cli_accepts_only_snapshot_root_output_offline` accepts
  the documented command and rejects a missing required option, an extra
  positional argument, duplicate `--snapshot-root`, duplicate `--output`,
  duplicate `--offline`, `--in-place`, `--write-snapshot`, and `--work-root`.
- `test_scaffold_verify_rejects_output_inside_snapshot` resolves both paths
  before comparison and exits nonzero when the report would be written at or
  below the snapshot root.
- `test_scaffold_verify_builds_in_fresh_temp_copy_without_mutating_snapshot`
  uses an adversarial fake builder that edits a source and creates a build
  directory, cache, and PDF in its working directory. It asserts every build,
  trace, and parity working path is in one newly created tree that is neither
  equal to nor a descendant of the snapshot root, while the original snapshot
  byte inventory and manifest hash remain unchanged, and expects the verifier
  to return `PASS`.
- `test_scaffold_verify_detects_snapshot_mutation_between_rehashes` changes one
  declared snapshot source through a controlled hook after the pre-hash and
  asserts nonzero exit, `status: "FAIL"`, reason
  `SNAPSHOT_MUTATED_DURING_VERIFY`, and unequal before/after inventory digests.

Pin the post-G5 chain with these named tests:

- `test_fill_results_cli_requires_the_exact_closed_argument_surface` rejects a
  missing or duplicate option, any positional argument, and unknown
  `--in-place`, `--skip-parity`, `--allow-pending`, or
  `--allow-unregistered-number` bypass.
- `test_fill_results_updates_paired_slots_atomically_from_terminal_claims`
  injects one supported, one falsified, one narrowed, and one null result and
  requires exact paired EN/KO status, estimate, interval, scope, and material
  caveat digests in one deterministic fill report.
- `test_fill_results_refuses_softened_caveat_unregistered_number_or_pending`
  mutates each failure independently and expects nonzero exit without changing
  either live tree or the link/parity files.
- `test_fill_results_rehashes_g5_results_links_live_sources_and_g0_snapshot`
  mutates each constituent through a controlled pre-commit hook and requires
  fail-closed rollback; the G0 snapshot byte inventory is unchanged in every
  branch.

- [ ] **Phase A.2: Create a buildable non-assertive skeleton**

The abstract contains only the problem and preregistered contribution slots; it
does not state unobserved positive results. The limitations section exists from
the first build.

Every figure/table record includes question/claim IDs, result blocks, generator
command, caption, reuse owner, alt text, and digest. Every equation record
includes derivation and unit/property-test artifact digests. Manually orphaned
assets are rejected.

Each PAPER-C claim has a stable result-slot ID whose content is typed
`PENDING_G5` and explicitly says that no empirical result has yet been
observed. Scaffold tests reject effect estimates, directional success language,
or `SUPPORTED` status before a matching G5 result block exists.
The snapshot command copies the exact final pre-G0 English/Korean/parity source
trees into `publication/scaffolds/G0`, records every POSIX path/digest and source
commit/tree in its manifest, and registers that immutable manifest in the DAG.
Its schema-fixed source set includes manuscript sources, bibliography,
Makefiles/templates, and declared static assets. The inventory and copied
snapshot exclude generated build trees, cache trees, and generated manuscript
PDFs; snapshot creation never copies them, and verification rejects any
unmanifested node that appears later. Manifest entries are regular files only:
symlinks, devices, sockets, and path traversal are rejected.

Verification first opens `manifest.json` and every declared source only through
read-only, no-follow file handles. It hashes the exact manifest bytes and
reconstructs the complete declared source inventory before doing work, then
copies those source bytes—not the snapshot directory wholesale—into a fresh
temporary tree. English build and trace, Korean build and trace, and parity
validation all receive paths in that temporary tree; no subprocess receives
the snapshot root as a working or output directory. Repository and network
fallback remain disabled, so an undeclared build dependency fails before G0.
After all checks, the verifier independently reopens and rehashes the original
snapshot manifest and source inventory. It writes its report outside the
snapshot only after this post-hash. `PASS` requires manifest-before equals
manifest-after, inventory-before equals inventory-after, the temporary copy
equals the pre-hash inventory, and every build/trace/parity check succeeds.

`G0-verification.json` records at least:

```text
status
snapshot_manifest_sha256_before
snapshot_manifest_sha256_after
snapshot_inventory_sha256_before
snapshot_inventory_sha256_after
copied_source_inventory_sha256
workspace_outside_snapshot
english_build_status
english_trace_status
korean_build_status
korean_trace_status
parity_status
offline
failure_reason
```

Generate the English BibLaTeX file and Korean bibliography view
deterministically from `SourceRecord` citation metadata. The manifest records
the shared source-registry digest. Duplicate citekeys, unresolved source IDs,
or hand-edited bibliography drift fail.

- [ ] **Phase A.3: Materialize, build, and validate the G0 scaffold**

```bash
uv run pytest tests/test_publication_scaffold.py \
  tests/test_publication_trace.py tests/test_cli.py -q
uv run stc publication scaffold materialize \
  --claims registry/claims.jsonl \
  --english-root ../../paper-en/sleep-time-compute \
  --korean-root ../../study-kr/sleep-time-compute \
  --parity publication/links/parity.jsonl \
  --mode g0-placeholders
uv run stc bibliography render \
  --sources registry/sources.jsonl \
  --bib ../../paper-en/sleep-time-compute/references.bib \
  --korean ../../study-kr/sleep-time-compute/back/99-bibliography.md \
  --output publication/reports/G0-bibliography-render.json
make -C ../../paper-en/sleep-time-compute clean all
uv run stc publication trace \
  --source ../../paper-en/sleep-time-compute/main.tex \
  --links publication/links/manuscript-links.jsonl \
  --output publication/reports/G0-english-trace.json
make -C ../../study-kr/sleep-time-compute clean all
uv run stc publication trace \
  --source ../../study-kr/sleep-time-compute/BOOK.md \
  --links publication/links/manuscript-links.jsonl \
  --output publication/reports/G0-korean-trace.json
uv run stc publication validate-parity publication/links/parity.jsonl \
  --english-trace publication/reports/G0-english-trace.json \
  --korean-trace publication/reports/G0-korean-trace.json \
  --output publication/reports/G0-parity-validation.json
uv run stc publication scaffold snapshot \
  --english-root ../../paper-en/sleep-time-compute \
  --korean-root ../../study-kr/sleep-time-compute \
  --parity publication/links/parity.jsonl \
  --snapshot-root publication/scaffolds/G0
uv run stc publication scaffold verify \
  --snapshot-root publication/scaffolds/G0 \
  --output publication/scaffolds/G0-verification.json \
  --offline
```

Expected: PDF builds, unresolved references are zero, and every material
sentence marker and publication asset resolves. All result slots remain
non-assertive `PENDING_G5` placeholders. The snapshot-only verifier rebuilds
and traces the copied English/Korean/parity bytes only in a fresh temporary
tree, records the exact before/copy/after manifest and source-inventory
digests, and passes before G0 input assembly. The original snapshot contains no
generated build tree, cache, or generated PDF after verification, and its
pre/post manifest and inventory digests are byte-identical.

- [ ] **Phase B.1: Fill only adjudicated post-G5 result slots**

After G5, resolve each stable slot from a digest-bound `ResultBlock` and the
corresponding claim-adjudication record. Preserve falsified, narrowed, null,
and inconclusive outcomes verbatim; do not silently delete a preregistered slot
or soften a material caveat. The live manuscripts retain their
`g0ScaffoldManifestSha256` lineage but never rewrite
`publication/scaffolds/G0`; a mutation there stales G0 and blocks G6. Re-run
bibliography rendering, both manuscript builds, trace, and parity validation.
Any remaining `PENDING_G5`, unregistered numeric result, or prose/result digest
mismatch blocks G6.

The post-G5 fill is executable rather than an editing instruction:

```bash
uv run pytest tests/test_publication_fill.py \
  tests/test_publication_trace.py tests/test_parity.py tests/test_cli.py -q
uv run stc publication fill-results \
  --g5 manifests/gates/G5.json \
  --claims registry/claims.jsonl \
  --history registry/claim-history.jsonl \
  --result-blocks manifests/stc/adjudicated-result-blocks.jsonl \
  --g0-snapshot publication/scaffolds/G0 \
  --english-root ../../paper-en/sleep-time-compute \
  --korean-root ../../study-kr/sleep-time-compute \
  --parity publication/links/parity.jsonl \
  --links publication/links/manuscript-links.jsonl \
  --output publication/reports/post-G5-fill.json
uv run stc bibliography render \
  --sources registry/sources.jsonl \
  --bib ../../paper-en/sleep-time-compute/references.bib \
  --korean ../../study-kr/sleep-time-compute/back/99-bibliography.md \
  --output publication/reports/post-G5-bibliography-render.json
make -C ../../paper-en/sleep-time-compute clean all
make -C ../../study-kr/sleep-time-compute clean all
uv run stc publication inventory \
  --root ../../paper-en/sleep-time-compute \
  --entry main.tex \
  --output publication/reports/post-G5-english-source-inventory.json
uv run stc publication inventory \
  --root ../../study-kr/sleep-time-compute \
  --entry BOOK.md \
  --output publication/reports/post-G5-korean-source-inventory.json
uv run stc publication trace \
  --source ../../paper-en/sleep-time-compute/main.tex \
  --links publication/links/manuscript-links.jsonl \
  --output publication/reports/post-G5-english-trace.json
uv run stc publication trace \
  --source ../../study-kr/sleep-time-compute/BOOK.md \
  --links publication/links/manuscript-links.jsonl \
  --output publication/reports/post-G5-korean-trace.json
uv run stc publication validate-parity publication/links/parity.jsonl \
  --english-trace publication/reports/post-G5-english-trace.json \
  --korean-trace publication/reports/post-G5-korean-trace.json \
  --output publication/reports/post-G5-parity-validation.json
```

The inventory command rejects symlinks, generated build/cache trees,
undeclared extensions, path traversal, and a source graph that escapes its
root. The trace and parity reports bind the corresponding live inventory,
result-block set, fill report, and G0 manifest digest. Re-running any report
against changed source bytes fails rather than silently refreshing lineage.

- [ ] **Shared tooling: Implement all-page publication visual QA**

`stc publication visual-qa` verifies both PDF digests, page size/count,
successful Poppler rendering of every page, embedded/subset font inventory,
missing/replacement glyph scans, extractable text, unresolved-reference and
overfull-box build-log counters, raster dimensions, and contact-sheet
generation. A deterministic Matplotlib contact sheet includes every page at a
legible index. The machine report stores tool versions, input/build-log
digests, page-level findings, contact-sheet digest, and zero-error counters.

The command also validates a separate `HumanApproval` record containing the
approval ID, canonical unsigned-payload digest, signer ID/role, exact English
and Korean PDF digests, machine-report and contact-sheet digests,
`all_pages_reviewed`, cross-page consistency, clipping/overflow,
equation/table/figure legibility, Korean glyph review, open issues,
`PASS|FAIL`, signing algorithm, trusted key ID, detached signature, and signed
timestamp. The signature is verified over canonical JSON with the signature
field omitted; a typed-but-unverified string is not approval. Automation
cannot synthesize or sign that record. At skeleton time it may be absent and
the command returns `PENDING_HUMAN`; G8 requires a signature-valid,
exact-digest `PASS`.

```bash
uv run pytest tests/test_publication_visual_qa.py -q
uv run stc publication visual-qa \
  --english tests/fixtures/publication-qa/english.pdf \
  --korean tests/fixtures/publication-qa/korean.pdf \
  --english-log tests/fixtures/publication-qa/english.log \
  --korean-log tests/fixtures/publication-qa/korean.log \
  --output /tmp/stc-publication-visual-qa.json \
  --contact-sheet /tmp/stc-publication-contact-sheet.png \
  --allow-pending-human
```

Expected: the good fixture is machine-PASS/PENDING_HUMAN; fixtures with an
unrenderable page, unembedded font, replacement glyph, overfull box, missing
page in the contact sheet, or mismatched human-review digest fail closed.

### Task 12: Create the Korean companion and parity map

**Execution timing:** Phase A is created by the same pre-G0-input scaffold
command as Task 11 and may expand background chapters after G1. Phase B runs after G5 and
fills original-contribution sections only from the same adjudicated result
records used by the English paper.

**Files:**
- Create: `study-kr/sleep-time-compute/BOOK.md`
- Create: `study-kr/sleep-time-compute/front/*.md`
- Create: `study-kr/sleep-time-compute/chapters/*.md`
- Create: `study-kr/sleep-time-compute/back/*.md`
- Create: `study-kr/sleep-time-compute/build/*`
- Create: `research/sleep-time-compute/publication/links/parity.jsonl`
- Create: `research/sleep-time-compute/src/stc_research/parity.py`
- Modify: `research/sleep-time-compute/src/stc_research/cli.py`
- Modify: `research/sleep-time-compute/tests/test_cli.py`
- Create: `research/sleep-time-compute/tests/test_parity.py`

**Interfaces:**
- Produces: Korean locations and caveats for every paper claim; monograph-only
  claims link to their own evidence.
- Extends the declared `stc` CLI with the exact positional contract
  `publication validate-parity PARITY_JSONL --english-trace PATH
  --korean-trace PATH --output PATH`; `parity.py`, `test_parity.py`, and the
  shared parser tests own this command.

- [ ] **Phase A.1: Write parity tests**

```python
def test_every_english_material_claim_has_korean_location():
    assert parity_ids() == english_material_claim_ids()


def test_monograph_only_assertions_are_registered():
    assert scan_unregistered_claim_markers() == []
```

Each parity record contains claim ID, English block IDs, Korean block IDs,
material caveats in both languages, parity status, reviewer, and review date.
Korean material blocks use stable paired HTML comments such as:

```text
<!-- STC:BEGIN KO-S0001 -->
...
<!-- STC:END KO-S0001 -->
```

Paraphrase is allowed; omitted or softened material caveats are not.

- [ ] **Phase A.2: Create the G0 chapter map and placeholders**

Use chapters for chronology, biological/theoretical basis, wake/test/sleep
framing, external/latent/parametric media, training data and objectives,
capacity/forgetting, theory/scaling, benchmark/results, systems infrastructure,
governance/deletion, open hypotheses, and limitations.

Every English result slot has a Korean partner carrying the same
`PENDING_G5` status and material caveat identifiers. Monograph-only theoretical
hypotheses are visibly labeled and cannot be promoted to paper evidence by
parity alone.

- [ ] **Phase A.3: Build and validate the placeholder companion**

```bash
uv run pytest tests/test_parity.py tests/test_cli.py -q
make -C ../../study-kr/sleep-time-compute clean all
uv run stc publication validate-parity publication/links/parity.jsonl \
  --english-trace publication/reports/G0-english-trace.json \
  --korean-trace publication/reports/G0-korean-trace.json \
  --output publication/reports/G0-parity-validation.json
```

Expected: zero parity gaps, missing glyphs, unresolved citations, or claim
markers without evidence; every empirical result slot remains non-assertive.

- [ ] **Phase B.1: Fill the companion from the shared post-G5 result map**

Replace paired placeholders from the same digest-bound result and adjudication
records as Task 11, expand interpretation only when separately registered as a
monograph claim, and re-run the Phase A.3 commands. G6 requires exact status,
scope, uncertainty, and material-caveat parity even when the Korean exposition
is longer.

### Task 13: Admit results, adjudicate claims, and pass G6

**Files:**
- Modify: `claims/sleep-time-compute.bundle.json`
- Create: `research/sleep-time-compute/src/stc_research/g4_execution_snapshot.py`
- Create: `research/sleep-time-compute/src/stc_research/experiment_replay.py`
- Create: `research/sleep-time-compute/src/stc_research/adjudicate_results.py`
- Create: `research/sleep-time-compute/src/stc_research/analysis/independent_headline.py`
- Create: `research/sleep-time-compute/src/stc_research/analysis/independent_scaling.py`
- Create: `research/sleep-time-compute/src/stc_research/analysis/independent_coverage.py`
- Create: `research/sleep-time-compute/src/stc_research/analysis/independent_information.py`
- Modify: `research/sleep-time-compute/src/stc_research/cli.py`
- Modify: `research/sleep-time-compute/tests/test_cli.py`
- Create: `research/sleep-time-compute/tests/test_g4_execution_snapshot.py`
- Create: `research/sleep-time-compute/tests/test_experiment_bundle.py`
- Create: `research/sleep-time-compute/tests/test_clean_rerun_protocol.py`
- Create: `research/sleep-time-compute/tests/test_independent_headline.py`
- Create: `research/sleep-time-compute/tests/test_independent_scaling.py`
- Create: `research/sleep-time-compute/tests/test_independent_coverage.py`
- Create: `research/sleep-time-compute/tests/test_independent_information.py`
- Create: `research/sleep-time-compute/tests/test_result_adjudication.py`
- Create: `research/sleep-time-compute/reports/G5-result-admissibility.md`
- Create: `research/sleep-time-compute/reports/G6-claim-gate.md`
- Modify: `research/sleep-time-compute/registry/audit-findings.jsonl`
- Modify: `research/sleep-time-compute/registry/audit-adjudications.jsonl`
- Create: `research/sleep-time-compute/manifests/stc/g4-confirmatory-execution.json`
- Create: `research/sleep-time-compute/manifests/stc/g4-capacity-execution.json`
- Create: `research/sleep-time-compute/manifests/stc/confirmatory-clean-rerun-request.json`
- Create after distributed verification:
  `research/sleep-time-compute/manifests/systems/distributed-confirmatory-clean-rerun-receipt.json`
- Create after distributed validation:
  `research/sleep-time-compute/manifests/stc/confirmatory-clean-rerun-result-validation.json`
- Create: `research/sleep-time-compute/manifests/stc/scaling-summary-independent.json`
- Create: `research/sleep-time-compute/manifests/stc/coverage-scaling-summary-independent.json`
- Create: `research/sleep-time-compute/manifests/stc/parametric-information-summary-independent.json`
- Create: `research/sleep-time-compute/manifests/stc/adjudicated-result-blocks.jsonl`
- Create: `research/sleep-time-compute/manifests/stc/result-adjudication-report.json`
- Create: `research/sleep-time-compute/manifests/gate-inputs/G5.json`
- Create: `research/sleep-time-compute/manifests/gate-inputs/G6.json`
- Create: `research/sleep-time-compute/manifests/gates/G5.json`
- Create: `research/sleep-time-compute/manifests/gates/G6.json`

**Interfaces:**
- Produces: admissible result blocks, result-backed claim revisions, and a
  zero-blocker G6 manuscript candidate.
- Extends the declared `stc` CLI with
  `experiments run-confirmatory`, `experiments clean-rerun prepare`, and
  `experiments clean-rerun finalize`,
  `gate inputs assemble G5|G6`, and `claims adjudicate-results`. The independent
  paths are the declared module CLIs
  `python -m stc_research.analysis.independent_headline` and
  `python -m stc_research.analysis.independent_scaling`,
  `independent_coverage`, and `independent_information`.
- Consumes the theory plan's declared `results validate` and `scaling fit`
  CLI commands for the canonical primary scaling calculation.
- Consumes the systems plan's two non-substitutable G4 snapshots, canonical
  core and capacity atomic imports, per-phase distributed receipts, four
  coverage arm×stage validations, scaling fit/holdout validations, information
  validation, capacity phase plan, aggregate budget, and canonical actual
  ledger/head.

- [ ] **Step 1: Validate every experiment bundle and pass G5**

The experiment-bundle schema requires config, seed, Git SHA, container digest,
environment fingerprint, generator/data digest, split manifest, base
checkpoint/tokenizer revision, upstream commit/patch, exact command,
analysis/render digest, immutable stdout/raw metrics, derived result blocks,
and checksums. Missing any field blocks G5.

The systems controller supplies two non-substitutable executable snapshots.
`g4-confirmatory-execution.json` binds only the core preregistration, 297-row
manifest/run index, analysis, core budget, PASS G2/G4 chain, G4 input index,
checkout, and target environment. `g4-capacity-execution.json` separately
binds PASS G2-CAP, capacity preregistration, phase plan, stakes, disjoint CAL
summary, all scaling/coverage/information configs/powers/manifests/cohort
indexes, aggregate budget, canonical-ledger contract, checkout/environment,
and exact CAL→TEST transport. It contains no confirmatory outcome or receipt.
Tests swap snapshot types, mutate any constituent, inject primary/clean
receipts into capacity, or bypass the phase plan; every case fails. G5
independently rehashes both snapshots against each result's recorded snapshot
digest.

The systems plan owns the distributed primary child execution, receipt, and
atomic import. Its canonical raw inputs are
`results/confirmatory-primary`,
`manifests/systems/distributed-confirmatory-receipt.json`, and
`manifests/stc/confirmatory-result-validation.json`. Evidence Task 13 never
launches those children again. `experiments run-confirmatory` is analysis-only:
it revalidates the receipt/import top-level manifest and validation report,
copies no raw bytes, refuses a symlink/alias, runs the canonical analysis from
the immutable import, and writes only to the disjoint
`results/confirmatory-primary-analysis` tree.

The clean rerun uses the exact same two-node/eight-A100
`runtime run-matrix → verify-only receipt → results validate → import-matrix`
protocol and all \(297n\) child/covariance semantics from the systems plan. The
controller's `clean-rerun prepare` command validates the primary import and
receipt, derives the requested checkout from the frozen execution snapshot, and
writes a one-use typed request. That request pins the same
manifest/cohort/preregistration/snapshot, launcher, probe, counter, container,
ranks, and shard count; binds run ID `confirmatory-clean-rerun`; and requires
the `node-local-detached-fresh-caches` isolation policy. Preparation does not
create a controller-local worktree or cache and never substitutes a local loop.

Cluster operators execute that typed request through the systems distributed
launcher, which stages at
`$STC_NFS_ROOT/staging/confirmatory-clean-rerun`, emits
`manifests/systems/distributed-confirmatory-clean-rerun-receipt.json`, validates
to `manifests/stc/confirmatory-clean-rerun-result-validation.json`, and
atomically imports to `results/confirmatory-clean-rerun`. On each node, the
systems runbook creates a detached worktree under
`/var/tmp/stc-confirmatory-clean-rank-$STC_NODE_RANK.XXXXXX` and four initially
empty cache roots, then passes the request with `--rerun-request`; its receipt
attests both node-local checkout identities, paths, digests, and initial cache
freshness. The controller then runs `clean-rerun finalize`, which revalidates
all four objects and writes canonical derived output only to
`results/confirmatory-clean-rerun-analysis`. Missing distributed launcher
inputs, reused primary cache/environment/run ID, fewer than \(297n\) logical
child references, a non-A100 backend, absent receipt, or non-atomic import
blocks G5. `confirmatory-independent` is a second implementation over the
**primary** immutable raw import; the clean rerun remains a separate
reproduction dataset. Each raw and analysis tree has its own
manifest-last top-level inventory and artifact-DAG record; none is committed
as a large payload or silently aliased to another tree.

`analysis.independent_headline` is a separately implemented module. An
import-graph/AST test rejects any import of the primary analysis,
result-rendering, or adjudication module, and golden tests require it to
calculate decisions/estimates directly from frozen raw fields. Compare exact
logical-cell coverage, decisions, estimates within frozen tolerance, and
raw/derived digests across all three trees. The preregistered primary analysis
is the sole source of PAPER-C scientific statuses. Primary failure plus clean
or independent pass cannot create support; primary support plus required
discordance becomes `NARROWED_REPRODUCTION_DISCORDANT` (with the primary
estimate still reported). No pooling/OR/meta-estimand is allowed without a
prospective powered amendment. A swap-pass/fail mutation must prove that only
the primary can create support. A copy, alias, symlink, shared
derived-results file, or different CLI flag into the primary function is not
an independent path.

Scaling-law outputs are headline results too. The separately implemented
`analysis.independent_scaling` recomputes the preregistered final fit, model
selection from fit data only, immutable provisional digest, untouched
score-only A100 holdout metrics, child coverage, and scalar/componentwise
resource predicates directly from the validated fit/holdout bundles. Its AST/import-graph test
rejects imports from the primary scaling fitter, renderer, or cached
`scaling-summary.json`; it treats both stage validations as digest-bound
comparison targets and recomputes every predicate. G5 binds the primary and
independent summaries, provisional/unlock, fit/holdout receipts/validations,
`SCALING-A100-GLOBAL-{K}`, budget/cohort, canonical ledger/head, and result-tree
digests. A result-backed claim requires both implementations to classify
`a100_regime_scaling_law`; `scaling_law` without hardware qualifier is
inadmissible.

`analysis.independent_coverage` separately reimplements codec accounting,
cardinality and fixed-bits estimands, the sealed fit, score-only holdout
evaluation, `COVERAGE-GLOBAL-48` max-stat decisions, and four arm×stage
validation/ledger predicates. `analysis.independent_information` independently
reconstructs the complete-state inventory, verifies fresh seeds/base restores
and decoder orientation, and recomputes the finite-sample bitwise-Fano
inferential lower bound from association×bit errors; it must return zero on
shuffled/no-signal fixtures and detect seed/log sidecars. G5 binds their
primary/independent summaries and all raw roots. Information remains
`ESTIMATOR_QUALIFICATION_ONLY`/`CHARACTERIZATION`; it cannot be promoted to a
universal capacity law.

A failed capacity track yields a typed failed/omitted capacity annex and blocks
only that track's claims. Core G5 can still pass, but no missing/failing
capacity artifact may be bypassed into a capacity statement.

```bash
uv run pytest \
  tests/test_g4_execution_snapshot.py \
  tests/test_experiment_bundle.py \
  tests/test_clean_rerun_protocol.py \
  tests/test_independent_headline.py \
  tests/test_independent_scaling.py \
  tests/test_independent_coverage.py \
  tests/test_independent_information.py \
  tests/test_cli.py -q
uv run stc execution snapshot verify \
  --expected-type core \
  --snapshot manifests/stc/g4-confirmatory-execution.json \
  --gate manifests/gates/G4.json \
  --g4-input-index manifests/systems/g4-inputs.json \
  --core-preregistration manifests/stc/preregistration.json \
  --design manifests/stc/confirmatory.jsonl \
  --cohort-index manifests/stc/confirmatory-run-index.json \
  --analysis manifests/stc/analysis-plan.json \
  --budget manifests/stc/confirmatory-budget.json \
  --program-budget-envelope configs/stc/design/program-budget-envelope.yaml
uv run stc execution snapshot verify \
  --expected-type capacity \
  --snapshot manifests/stc/g4-capacity-execution.json \
  --gate manifests/gates/G4.json \
  --g4-input-index manifests/systems/g4-inputs.json \
  --core-preregistration manifests/stc/preregistration.json \
  --capacity-preregistration manifests/stc/capacity-preregistration.json \
  --capacity-gate manifests/capacity-gates/G2-CAP.json \
  --capacity-phase-plan configs/systems/capacity-production-phases.yaml \
  --capacity-stakes manifests/stc/deployment-stakes-G2-CAP.json \
  --capacity-aggregate-budget manifests/stc/capacity-aggregate-budget.json \
  --program-budget-envelope configs/stc/design/program-budget-envelope.yaml
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

Pause at this explicit external-compute boundary. Execute the typed request
through the systems plan's exact concurrent two-node distributed launcher,
receipt, semantic validation, and atomic `import-matrix` protocol. Resume only
after all three declared clean-rerun artifacts exist and revalidate. Both
distributed launch invocations and node 0's `--verify-only` invocation must
pass
`--rerun-request manifests/stc/confirmatory-clean-rerun-request.json`; omitting
or changing that argument invalidates the receipt.

```bash
uv run stc experiments clean-rerun finalize \
  --request manifests/stc/confirmatory-clean-rerun-request.json \
  --validated-run results/confirmatory-clean-rerun \
  --receipt \
    manifests/systems/distributed-confirmatory-clean-rerun-receipt.json \
  --validation-report \
    manifests/stc/confirmatory-clean-rerun-result-validation.json \
  --output results/confirmatory-clean-rerun-analysis
uv run python -m stc_research.analysis.independent_headline \
  --hypotheses manifests/hypotheses-G2.json \
  --raw-bundles results/confirmatory-primary \
  --receipt manifests/systems/distributed-confirmatory-receipt.json \
  --validation-report manifests/stc/confirmatory-result-validation.json \
  --output results/confirmatory-independent
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
  --preregistration manifests/stc/capacity-preregistration.json \
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
uv run python -m stc_research.analysis.independent_scaling \
  --preregistration manifests/stc/capacity-preregistration.json \
  --capacity-gate manifests/capacity-gates/G2-CAP.json \
  --scaling-manifest manifests/stc/scaling-cells.jsonl \
  --cohort-index manifests/stc/scaling-cohort-index.json \
  --contrast-family manifests/stc/scaling-contrast-family-G2-CAP.json \
  --budget manifests/stc/scaling-budget.json \
  --aggregate-budget manifests/stc/capacity-aggregate-budget.json \
  --actual-ledger results/capacity/actual-rre-ledger.jsonl \
  --fit-validation manifests/stc/scaling-fit-result-validation.json \
  --holdout-validation manifests/stc/scaling-holdout-result-validation.json \
  --provisional-artifact manifests/stc/scaling-provisional.json \
  --holdout-unlock manifests/stc/scaling-holdout-unlock.json \
  --fit-results results/scaling/a100-fit \
  --holdout-results results/scaling/a100-holdout \
  --output manifests/stc/scaling-summary-independent.json
uv run stc results validate \
  --preregistration manifests/stc/capacity-preregistration.json \
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
  --preregistration manifests/stc/capacity-preregistration.json \
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
uv run stc results validate \
  --preregistration manifests/stc/capacity-preregistration.json \
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
  --preregistration manifests/stc/capacity-preregistration.json \
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
  --preregistration manifests/stc/capacity-preregistration.json \
  --capacity-gate manifests/capacity-gates/G2-CAP.json \
  --config configs/stc/design/coverage-scaling.yaml \
  --cohort-index manifests/stc/coverage-scaling-cohort-index.json \
  --contrast-family manifests/stc/coverage-contrast-family-G2-CAP.json \
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
uv run python -m stc_research.analysis.independent_coverage \
  --preregistration manifests/stc/capacity-preregistration.json \
  --capacity-gate manifests/capacity-gates/G2-CAP.json \
  --cardinality-manifest manifests/stc/coverage-cardinality-cells.jsonl \
  --fixed-bits-manifest manifests/stc/coverage-fixed-bits-cells.jsonl \
  --cohort-index manifests/stc/coverage-scaling-cohort-index.json \
  --contrast-family manifests/stc/coverage-contrast-family-G2-CAP.json \
  --budget manifests/stc/coverage-scaling-budget.json \
  --aggregate-budget manifests/stc/capacity-aggregate-budget.json \
  --actual-ledger results/capacity/actual-rre-ledger.jsonl \
  --provisional-artifact manifests/stc/coverage-provisional.json \
  --holdout-unlock manifests/stc/coverage-holdout-unlock.json \
  --cardinality-fit-validation manifests/stc/coverage-cardinality-fit-result-validation.json \
  --cardinality-holdout-validation manifests/stc/coverage-cardinality-holdout-result-validation.json \
  --fixed-bits-fit-validation manifests/stc/coverage-fixed-bits-fit-result-validation.json \
  --fixed-bits-holdout-validation manifests/stc/coverage-fixed-bits-holdout-result-validation.json \
  --results-root results/coverage-scaling \
  --output manifests/stc/coverage-scaling-summary-independent.json
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
uv run python -m stc_research.analysis.independent_information \
  --preregistration manifests/stc/capacity-preregistration.json \
  --capacity-gate manifests/capacity-gates/G2-CAP.json \
  --config configs/stc/design/parametric-information.yaml \
  --manifest manifests/stc/parametric-information-cells.jsonl \
  --cohort-index manifests/stc/parametric-information-cohort-index.json \
  --budget manifests/stc/parametric-information-budget.json \
  --aggregate-budget manifests/stc/capacity-aggregate-budget.json \
  --actual-ledger results/capacity/actual-rre-ledger.jsonl \
  --validation-report manifests/stc/parametric-information-result-validation.json \
  --results results/parametric-information \
  --output manifests/stc/parametric-information-summary-independent.json
uv run stc gate inputs assemble G5 \
  --root . \
  --predecessor manifests/gates/G4.json \
  --core-preregistration manifests/stc/preregistration.json \
  --capacity-preregistration manifests/stc/capacity-preregistration.json \
  --capacity-gate manifests/capacity-gates/G2-CAP.json \
  --core-execution-snapshot manifests/stc/g4-confirmatory-execution.json \
  --capacity-execution-snapshot manifests/stc/g4-capacity-execution.json \
  --capacity-phase-plan configs/systems/capacity-production-phases.yaml \
  --primary-raw results/confirmatory-primary \
  --primary-analysis results/confirmatory-primary-analysis \
  --primary-receipt manifests/systems/distributed-confirmatory-receipt.json \
  --primary-validation manifests/stc/confirmatory-result-validation.json \
  --clean-raw results/confirmatory-clean-rerun \
  --clean-analysis results/confirmatory-clean-rerun-analysis \
  --clean-request manifests/stc/confirmatory-clean-rerun-request.json \
  --clean-receipt \
    manifests/systems/distributed-confirmatory-clean-rerun-receipt.json \
  --clean-validation \
    manifests/stc/confirmatory-clean-rerun-result-validation.json \
  --core-independent results/confirmatory-independent \
  --scaling-fit-raw results/scaling/a100-fit \
  --scaling-holdout-raw results/scaling/a100-holdout \
  --scaling-fit-receipt manifests/systems/distributed-scaling-fit-receipt.json \
  --scaling-holdout-receipt manifests/systems/distributed-scaling-holdout-receipt.json \
  --scaling-fit-validation manifests/stc/scaling-fit-result-validation.json \
  --scaling-holdout-validation manifests/stc/scaling-holdout-result-validation.json \
  --scaling-provisional manifests/stc/scaling-provisional.json \
  --scaling-holdout-unlock manifests/stc/scaling-holdout-unlock.json \
  --scaling-contrast-family manifests/stc/scaling-contrast-family-G2-CAP.json \
  --scaling-summary manifests/stc/scaling-summary.json \
  --scaling-independent-summary manifests/stc/scaling-summary-independent.json \
  --scaling-budget manifests/stc/scaling-budget.json \
  --scaling-cohort-index manifests/stc/scaling-cohort-index.json \
  --coverage-raw-root results/coverage-scaling \
  --coverage-cardinality-fit-receipt manifests/systems/distributed-coverage-cardinality-fit-receipt.json \
  --coverage-cardinality-holdout-receipt manifests/systems/distributed-coverage-cardinality-holdout-receipt.json \
  --coverage-fixed-bits-fit-receipt manifests/systems/distributed-coverage-fixed-bits-fit-receipt.json \
  --coverage-fixed-bits-holdout-receipt manifests/systems/distributed-coverage-fixed-bits-holdout-receipt.json \
  --coverage-cardinality-fit-validation manifests/stc/coverage-cardinality-fit-result-validation.json \
  --coverage-cardinality-holdout-validation manifests/stc/coverage-cardinality-holdout-result-validation.json \
  --coverage-fixed-bits-fit-validation manifests/stc/coverage-fixed-bits-fit-result-validation.json \
  --coverage-fixed-bits-holdout-validation manifests/stc/coverage-fixed-bits-holdout-result-validation.json \
  --coverage-provisional manifests/stc/coverage-provisional.json \
  --coverage-holdout-unlock manifests/stc/coverage-holdout-unlock.json \
  --coverage-contrast-family manifests/stc/coverage-contrast-family-G2-CAP.json \
  --coverage-summary manifests/stc/coverage-scaling-summary.json \
  --coverage-independent-summary manifests/stc/coverage-scaling-summary-independent.json \
  --coverage-budget manifests/stc/coverage-scaling-budget.json \
  --coverage-cohort-index manifests/stc/coverage-scaling-cohort-index.json \
  --information-raw results/parametric-information \
  --information-receipt manifests/systems/distributed-parametric-information-receipt.json \
  --information-validation manifests/stc/parametric-information-result-validation.json \
  --information-summary manifests/stc/parametric-information-summary.json \
  --information-independent-summary manifests/stc/parametric-information-summary-independent.json \
  --information-budget manifests/stc/parametric-information-budget.json \
  --information-cohort-index manifests/stc/parametric-information-cohort-index.json \
  --capacity-aggregate-budget manifests/stc/capacity-aggregate-budget.json \
  --capacity-actual-ledger results/capacity/actual-rre-ledger.jsonl \
  --capacity-ledger-head results/capacity/actual-rre-ledger.head.json \
  --output manifests/gate-inputs/G5.json
uv run stc gate evaluate G5 \
  --inputs manifests/gate-inputs/G5.json \
  --output manifests/gates/G5.json
```

Expected: each producer records its exact command, implementation digest,
environment/container digest, and input/output inventory; both distributed raw
imports and their disjoint analysis trees plus each independent tree cover the
same frozen cells. G5 is `PASS` only after binding both non-substitutable
core/capacity execution snapshots, core and capacity preregistrations,
G2-CAP/phase plan, primary/clean requests/receipts/validations, and every
scaling/coverage/information raw root, stage receipt, stage validation,
provisional/unlock, contrast family, track/aggregate budget, cohort index,
canonical ledger/head, and primary/independent summary. Core G5 may remain
valid when a capacity annex has a typed failed/omitted state, but that state
must itself be bound and no capacity claim is eligible. Any hardware,
coverage, information, RRE, or independent-fit discrepancy blocks only its
corresponding capacity statement and cannot rescue or change PAPER-C.

- [ ] **Step 2: Append result-backed claim revisions**

`adjudicate-results` preserves every G0 hypothesis revision in
`claim-history.jsonl` and appends one evidence-bearing revision per PAPER-C
claim with exact result block, decision reason, native resource vector,
lifecycle-cost profile digest, material caveats, and status `SUPPORTED` or
`FALSIFIED/NARROWED`. Null/narrowed outcomes are not rewritten into new
confirmatory hypotheses on the same data.

```bash
uv run pytest tests/test_result_adjudication.py -q
uv run stc claims adjudicate-results \
  --core-preregistration manifests/stc/preregistration.json \
  --capacity-preregistration manifests/stc/capacity-preregistration.json \
  --capacity-gate manifests/capacity-gates/G2-CAP.json \
  --core-status-source primary-only \
  --primary-raw results/confirmatory-primary \
  --primary-analysis results/confirmatory-primary-analysis \
  --clean-rerun-raw results/confirmatory-clean-rerun \
  --clean-rerun-analysis results/confirmatory-clean-rerun-analysis \
  --core-independent results/confirmatory-independent \
  --scaling-summary manifests/stc/scaling-summary.json \
  --scaling-independent-summary manifests/stc/scaling-summary-independent.json \
  --coverage-summary manifests/stc/coverage-scaling-summary.json \
  --coverage-independent-summary manifests/stc/coverage-scaling-summary-independent.json \
  --information-summary manifests/stc/parametric-information-summary.json \
  --information-independent-summary manifests/stc/parametric-information-summary-independent.json \
  --capacity-aggregate-budget manifests/stc/capacity-aggregate-budget.json \
  --capacity-actual-ledger results/capacity/actual-rre-ledger.jsonl \
  --capacity-annex-policy typed-fail-closed \
  --g5 manifests/gates/G5.json \
  --claims registry/claims.jsonl \
  --history registry/claim-history.jsonl \
  --result-blocks-output manifests/stc/adjudicated-result-blocks.jsonl \
  --output manifests/stc/result-adjudication-report.json
```

Mutation tests swap primary and clean PASS/FAIL states, make clean disagree with
primary, and make either independent implementation disagree with its primary
capacity summary. Only validated primary core data may set PAPER-C status;
clean/independent disagreement narrows reproducibility and can never rescue or
pool with a failed primary. A failed/missing capacity pair emits a typed
failed/omitted annex and no capacity claim while leaving eligible PAPER-C
status unchanged. The command writes the terminal claim revisions,
`adjudicated-result-blocks.jsonl`, and `result-adjudication-report.json` as one
compare-and-swap transaction. The report binds G5, the before/after claims and
history digests, every consumed primary/clean/independent summary, and the
canonical ordered `ResultBlock` digest. A partial write, stale history, missing
block, or output block not referenced by a terminal revision fails closed.

- [ ] **Step 3: Export and gate G6**

```bash
uv run pytest tests/test_publication_fill.py \
  tests/test_publication_trace.py tests/test_parity.py tests/test_cli.py -q
uv run stc publication fill-results \
  --g5 manifests/gates/G5.json \
  --claims registry/claims.jsonl \
  --history registry/claim-history.jsonl \
  --result-blocks manifests/stc/adjudicated-result-blocks.jsonl \
  --g0-snapshot publication/scaffolds/G0 \
  --english-root ../../paper-en/sleep-time-compute \
  --korean-root ../../study-kr/sleep-time-compute \
  --parity publication/links/parity.jsonl \
  --links publication/links/manuscript-links.jsonl \
  --output publication/reports/post-G5-fill.json
uv run stc bibliography render \
  --sources registry/sources.jsonl \
  --bib ../../paper-en/sleep-time-compute/references.bib \
  --korean ../../study-kr/sleep-time-compute/back/99-bibliography.md \
  --output publication/reports/post-G5-bibliography-render.json
make -C ../../paper-en/sleep-time-compute clean all
make -C ../../study-kr/sleep-time-compute clean all
uv run stc publication inventory \
  --root ../../paper-en/sleep-time-compute \
  --entry main.tex \
  --output publication/reports/post-G5-english-source-inventory.json
uv run stc publication inventory \
  --root ../../study-kr/sleep-time-compute \
  --entry BOOK.md \
  --output publication/reports/post-G5-korean-source-inventory.json
uv run stc publication trace \
  --source ../../paper-en/sleep-time-compute/main.tex \
  --links publication/links/manuscript-links.jsonl \
  --output publication/reports/post-G5-english-trace.json
uv run stc publication trace \
  --source ../../study-kr/sleep-time-compute/BOOK.md \
  --links publication/links/manuscript-links.jsonl \
  --output publication/reports/post-G5-korean-trace.json
uv run stc publication validate-parity publication/links/parity.jsonl \
  --english-trace publication/reports/post-G5-english-trace.json \
  --korean-trace publication/reports/post-G5-korean-trace.json \
  --output publication/reports/post-G5-parity-validation.json
uv run stc claims export-veridraft \
  --output ../../claims/sleep-time-compute.bundle.json
uv run --group publication veridraft \
  --config veridraft.config.json \
  --data-dir .veridraft-stc \
  import-bundle ../../claims/sleep-time-compute.bundle.json
uv run --group publication veridraft \
  --config veridraft.config.json \
  --data-dir .veridraft-stc \
  gate sleep-time-compute-2026
make -C ../../paper-en/sleep-time-compute clean all
make -C ../../study-kr/sleep-time-compute clean all
uv run stc gate inputs assemble G6 \
  --root . \
  --predecessor manifests/gates/G5.json \
  --claim-bundle ../../claims/sleep-time-compute.bundle.json \
  --veridraft-data .veridraft-stc \
  --novelty-audit reports/novelty-audit.md \
  --systems-claim-chart SYSTEMS-INFRA-PRIMARY-SOURCE-AUDIT.md \
  --g0-scaffold-manifest publication/scaffolds/G0/manifest.json \
  --g0-verification publication/scaffolds/G0-verification.json \
  --post-g5-fill-report publication/reports/post-G5-fill.json \
  --result-adjudication-report manifests/stc/result-adjudication-report.json \
  --result-blocks manifests/stc/adjudicated-result-blocks.jsonl \
  --english-source-root ../../paper-en/sleep-time-compute \
  --english-source-inventory publication/reports/post-G5-english-source-inventory.json \
  --korean-source-root ../../study-kr/sleep-time-compute \
  --korean-source-inventory publication/reports/post-G5-korean-source-inventory.json \
  --manuscript-links publication/links/manuscript-links.jsonl \
  --parity-map publication/links/parity.jsonl \
  --english-trace publication/reports/post-G5-english-trace.json \
  --korean-trace publication/reports/post-G5-korean-trace.json \
  --parity-report publication/reports/post-G5-parity-validation.json \
  --bibliography-render-report publication/reports/post-G5-bibliography-render.json \
  --english-pdf ../../paper-en/sleep-time-compute/main.pdf \
  --korean-pdf ../../study-kr/sleep-time-compute/build/sleep-time-compute.pdf \
  --output manifests/gate-inputs/G6.json
uv run stc gate evaluate G6 \
  --inputs manifests/gate-inputs/G6.json \
  --output manifests/gates/G6.json
```

Expected: every PAPER-C and headline quantitative slot is fully adjudicated and
gate-valid—whether supported, falsified, narrowed, null, or typed
inconclusive—and the typed G6 record binds the canonical G5 predecessor, claim
bundle, both clean PDF builds,
the immutable G0 scaffold lineage, post-G5 fill and result-adjudication reports,
canonical result blocks, live English/Korean source inventories and roots,
manuscript links, trace/parity reports, exact qualifying support refs,
novelty/priority status, and the digest-bound adjacent-literature claim chart.
The assembler independently reopens and rehashes every inventory member and
rejects extra live source bytes, stale report digests, an altered link/parity
record, a G0 snapshot mutation, or any PDF not built from the supplied live
inventory. Hashing only the two PDFs is explicitly insufficient.

G6 requires zero blocked main claims, exact result-block references, visible
material caveats, and no hypothesis/unresolved assertion contributing to the
abstract or conclusion. A `FALSIFIED/NARROWED` main claim is admissible only
when the corresponding null/narrowed result and boundary are stated explicitly.
Every contribution claim carries `novelty_status =
ESTABLISHED|UNRESOLVED|NOT_CLAIMED` and an optional claim-chart dependency.
Any “first,” “novel,” exclusivity, or priority language fails when the status
is unresolved, the asserted element lies outside the chart, or the chart's
search scope remains bounded. The present systems audit therefore supports
prior-art-safe integration wording but cannot by itself establish priority.

- [ ] **Step 4: Rebuild and commit the G6 candidate**

```bash
git -C ../.. add claims/sleep-time-compute.bundle.json \
  paper-en/sleep-time-compute \
  study-kr/sleep-time-compute \
  research/sleep-time-compute
uv run stc sources staged-scan --repo-root ../.. --cached
git -C ../.. diff --cached --check
git -C ../.. commit -m "paper: gate English and Korean sleep-time publications"
```

Expected: no G5/G6 blocker remains and both clean builds pass; the commit is
ready to become the frozen scientific candidate for independent G7 audit.

### Task 14: Implement candidate, audited-release, and presentation tooling

**Execution timing:** Phase A is the early-tooling checkpoint immediately after
Task 6 and before Task 7. Phase B is a runtime release procedure after Task 13
and the real independent-audit inputs exist. Phase A must not freeze, tag, or
release the real study.

**Files:**
- Create: `research/sleep-time-compute/src/stc_research/candidate.py`
- Create: `research/sleep-time-compute/src/stc_research/attestations.py`
- Create: `research/sleep-time-compute/src/stc_research/release.py`
- Create: `research/sleep-time-compute/src/stc_research/presentation_handoff.py`
- Modify: `research/sleep-time-compute/src/stc_research/cli.py`
- Modify: `research/sleep-time-compute/tests/test_cli.py`
- Create: `research/sleep-time-compute/tests/test_candidate_freeze.py`
- Create: `research/sleep-time-compute/tests/test_attestations.py`
- Create: `research/sleep-time-compute/tests/test_gate_release.py`
- Create: `research/sleep-time-compute/tests/test_release_assembly.py`
- Create: `research/sleep-time-compute/tests/test_presentation_handoff.py`
- Create: `research/sleep-time-compute/tests/fixtures/g8-release/*`
- Create:
  `research/sleep-time-compute/schemas/control/release-candidate-manifest.schema.json`
- Create:
  `research/sleep-time-compute/schemas/control/release-seal.schema.json`
- Create before G7:
  `research/sleep-time-compute/manifests/trusted-reviewer-keys.json`
- Create before G7:
  `research/sleep-time-compute/manifests/attestations/author-executor-roster.json`
- Create before G7:
  `research/sleep-time-compute/manifests/attestations/G7-auditor.json`
- Create before G8:
  `research/sleep-time-compute/manifests/attestations/G8-evaluator.json`
- Create before G7:
  `research/sleep-time-compute/manifests/G7-independence-protocol.json`
- Create before G7:
  `research/sleep-time-compute/manifests/reaudits/G7.json`
- Create at runtime:
  `research/sleep-time-compute/manifests/candidates/stc-paper-v1.0.0-rc1.json`
- Create at runtime:
  `research/sleep-time-compute/manifests/gate-inputs/G7.json`
- Create at runtime:
  `research/sleep-time-compute/manifests/gate-inputs/G8.json`
- Create at runtime:
  `research/sleep-time-compute/manifests/gate-inputs/G8-evaluation-subject.json`
- Create at runtime: `research/sleep-time-compute/manifests/gates/G7.json`
- Create at runtime: `research/sleep-time-compute/manifests/gates/G8.json`
- Create at runtime:
  `research/sleep-time-compute/manifests/release-candidates/stc-paper-v1.0.0.json`
- Create at runtime:
  `research/sleep-time-compute/reports/release-candidate-smoke.json`
- Create at runtime:
  `research/sleep-time-compute/publication/releases/stc-paper-v1.0.0/release-seal.json`

**Interfaces:**
- Consumes: the Task 4 `evaluate_gate` engine and typed GateRecord schemas. This
  task implements G7/G8 input adapters; it never implements a second gate
  writer.
- Produces: a candidate record read from a detached worktree, verified
  attestations, deterministic release/environment/SBOM/reproduction bundles,
  and a self-contained evidence-owned presentation handoff.
- Extends the declared `stc` CLI with `candidate freeze`, `attest verify`,
  `gate inputs assemble G7|G8`, `gate inputs subject G8`, `release replay`,
  `release draft-manifest`, `release smoke`, `release assemble`,
  `release verify`, `presentation export`, and `presentation validate`. Parser
  tests own the exact arguments shown in Phase B, including the draft, smoke,
  and seal-output paths.

#### Phase A — early tooling, fixtures only

- [ ] **Phase A.1: Implement detached candidate/controller separation**

The controller is the ordinary package checkout and is the only place allowed
to receive candidate manifests, replay outputs, release trees, or handoff
trees. The candidate is a separate Git worktree at the exact commit peeled
from an already-existing annotated RC tag. It must have detached `HEAD`, an
empty tracked/untracked status, and the expected commit/tree digest.
Before tagging, the controller requires a clean tracked worktree and index;
unrelated user-owned untracked files are out of scope and must be ignored,
preserved, and neither staged nor deleted.

`stc candidate freeze` takes all four explicit locations: repository root, tag,
candidate-worktree root, controller package root, and output path. It treats
the candidate as read-only and opens every committed control input beneath that
root. Large ignored result payloads may resolve only through committed
receipt/validation/artifact records into an immutable content-addressed result
store; an ambient or digest-mismatched controller file is forbidden. It writes
the record atomically beneath the controller, then proves the candidate status
and tree digest are unchanged. It rejects a branch, lightweight tag, moved tag,
candidate path alias/symlink, output inside the candidate, dirty candidate,
missing G6, stale DAG, controller/candidate confusion, or any attempt to build
in the candidate. Clean rebuilds use a different disposable replay worktree;
they never reuse the frozen candidate worktree.

The candidate record binds annotated tag object/commit, commit/tree, G0–G6
chain digests, claim/result/manuscript digests, environment identity, and
artifact-DAG digest. It is immutable. A scientific correction creates a new
commit and monotonically named RC tag.

- [ ] **Phase A.2: Implement signed G7/G8 input adapters**

The trusted-key set is a reviewed manifest of signer ID, role, public key,
validity interval, and revocation status. `stc attest verify` canonicalizes the
record with only the signature field omitted, recomputes `subject_sha256` from
the separately referenced subject bytes/digest bundle, verifies the detached
signature over that canonical record, resolves the trusted non-revoked key,
and checks signer role and time. Merely populating `algorithm`, `key_id`, or
`signature` fields never passes.

The G7 adapter requires all of these exact, digest-bound inputs:

```text
candidate record
G6 predecessor record
independent audit report
audit-findings.jsonl
audit-adjudications.jsonl
re-audit manifest and re-audit attestations
signed auditor AuditAttestation
signed author/executor roster
trusted-key manifest
public independence protocol
```

The auditor attestation binds candidate commit/tree and record digest, audit
protocol/report, findings, adjudications, re-audits, roster digest, evaluator
identity/role, and `independence_mode`. A self-audit is rejected unless the
public protocol records fresh-context independence and a second named,
signature-valid reviewer. Every critical/major finding needs a typed
adjudication and independent re-audit against the exact current candidate.
Scientific changes invalidate the candidate and require a new commit/RC tag;
a waiver cannot preserve an old subject digest.

The G8 adapter requires the canonical G7 predecessor; candidate; public replay
and clean-build report; both the machine and final visual reports plus the
all-page contact-sheet digest; signature-valid exact-PDF `HumanApproval`;
complete G0–G7 chain;
the per-artifact rights scan embedded in the smoke report; the immutable
pre-seal release-candidate manifest and its artifact-smoke report; and
digest-bound measured target-A100 evidence/input
index with selected run ID `a100-confirmatory`. Missing target hardware is
`BLOCKED`, never waived or replaced by a local/synthetic result. A draft/smoke
digest mismatch, undeclared payload, or failed reproduction recipe also blocks
G8.

G8 has its own signed evaluator boundary; the visual `HumanApproval` is not
reused as a gate-evaluator signature. `stc gate inputs subject G8` first emits a
canonical `G8-evaluation-subject.json` containing digest refs for candidate,
G7, replay/build, both visual reports, HumanApproval, G0–G7, draft/smoke,
rights, and target-A100 evidence. An identified signer with trusted role
`release_evaluator` reviews and signs those exact subject bytes in
`G8-evaluator.json`. Automation cannot synthesize the signature.
`stc gate inputs assemble G8 --subject ... --evaluator-attestation ...` then
creates the final wrapper; the Task 4 engine recomputes the subject digest,
verifies signer role/key/time/signature, and writes its attestation ref into the
G8 `GateRecord`. Any changed subject requires a new signature.

- [ ] **Phase A.3: Implement deterministic release assembly**

`stc release draft-manifest` runs after G7, replay, exact-PDF approval, and
final visual QA but before G8. It freezes the exact scientific payload
inventory, per-artifact rights dispositions, environment/SBOM inputs,
reproduction recipe, expected output digests, and a closed list of permitted
control/seal additions. That list is exactly the smoke report, G8 evaluation
subject, G8 evaluator attestation, final G8 input/record, final manifest,
checksum inventory, outer archive, and seal record at the normative paths
shown below; no scientific payload path is open. `stc release smoke` resolves
that draft in a disposable
scratch worktree/cache while treating the detached candidate and controller
inputs as read-only, reruns rights and artifact-DAG checks, validates every
path/license, executes the declared non-hardware reproduction smoke, and
records the draft digest plus every checked artifact digest. Neither command
creates a final release or claims G8.

`stc release assemble` then accepts only that same draft, its passing smoke
report, and a signature-valid G8 `PASS` record that binds both. It writes into
a new, empty controller output directory. It may add only the canonical G8
record, the final manifest/checksum/archive envelope, and seal metadata named
by the draft; every draft-declared payload byte must remain identical. It never
modifies or reads undeclared uncommitted state from the controller and never
writes into the candidate. The only non-Git payload input allowed is the
ignored content-addressed result store resolved through committed receipts,
validations, and artifact digests. The release tree is:

```text
publication/releases/{release_tag}/
  pre-seal-manifest.json
  release-manifest.json
  release-seal.json
  artifact-dag.jsonl
  checksums.sha256
  release-bundle.tar.zst
  gate-chain/G0.json ... G8.json
  control/candidate.json
  audit/trusted-reviewer-keys.json
  audit/author-executor-roster.json
  audit/G7-auditor.json
  audit/G8-evaluator.json
  audit/G7-independence-protocol.json
  audit/G7-independent-audit.md
  audit/G7-reaudits.json
  audit/findings.jsonl
  audit/adjudications.jsonl
  audit/human-visual-approval.json
  audit/visual-qa-machine.json
  audit/visual-qa-final.json
  audit/all-pages-contact-sheet.png
  audit/public-replay.json
  control/G8-evaluation-subject.json
  control/G8-input.json
  control/release-candidate-smoke.json
  systems/g4-systems-evidence.json
  systems/g4-inputs.json
  systems/g4-evidence-bundle/
  papers/english.pdf
  papers/korean.pdf
  indexes/source-index.jsonl
  indexes/claim-index.jsonl
  indexes/evidence-index.jsonl
  indexes/result-index.jsonl
  environment/environment.json
  environment/toolchain.json
  environment/uv.lock
  environment/environment.tar.zst
  sbom/sbom.spdx.json
  reproduction/reproduction-manifest.json
  reproduction/reproduction-bundle.tar.zst
  reproduction/commands.jsonl
```

`environment.json` records the candidate/tag/tree, lockfile and container-image
digests, Python/uv/Pandoc/TeX/Poppler versions, OS/architecture, locale,
timezone, CPU/GPU/CUDA/driver identities, and every environment variable
allowed to affect a build. The environment schema allowlists names and
normalized values and rejects secrets, credentials, user names, and absolute
host paths. The SPDX 2.3 JSON SBOM inventories the release files, Python
dependencies, containers, system build tools, licenses, and relationships. The
reproduction manifest binds every input, command, expected output, gate,
schema, environment, and checksum; its bundle contains the candidate source
archive, locks, configs, schemas, permitted data, scripts, and replay
instructions needed to regenerate all non-hardware outputs.

Assembly uses the annotated tag timestamp as `SOURCE_DATE_EPOCH`, UTF-8
canonical JSON with final newline, sorted POSIX paths, fixed file modes,
numeric owner/group 0, empty owner/group names, normalized `LC_ALL=C` and
`TZ=UTC`, and single-threaded fixed-level zstd. Absolute paths, symlinks,
ambient timestamps, random archive order, unpinned dependencies, and host-only
cache references fail. Two assemblies of the same inputs in different
controller directories must have identical file inventories and archive
digests.

Archive/checksum boundaries are non-self-referential and schema-fixed.
`pre-seal-manifest.json` is byte-identical to the G8-bound draft.
`release-manifest.json` inventories the draft-declared payload and permitted
seal additions but not its own byte digest; `release-bundle.tar.zst` contains
that manifest and every declared payload except the outer archive,
`checksums.sha256`, and `release-seal.json`; each nested archive excludes
itself; and `checksums.sha256` covers every regular release file except itself
and `release-seal.json`. The terminal `ReleaseSealRecord` binds the draft,
smoke, G8, final manifest, archive, and checksum-inventory digests without being
included in any digest it signs. Tests reject an assembler that changes those
boundaries, changes a draft payload byte, adds an unpermitted file, or attempts
to hash an object into itself.

The `ReleaseSealRecord` fields are exactly `schemaVersion`, `releaseTag`,
`candidateRef`, `auditControlCommit`, `draftManifestRef`, `smokeReportRef`,
`g8EvaluationSubjectRef`, `g8EvaluatorAttestationRef`, `g8RecordRef`,
`finalManifestRef`, `payloadProjectionSha256`, `permittedAdditionRefs`,
`archiveRef`, `checksumInventoryRef`, `sourceDateEpoch`, `sealToolRef`, and
`sealedAt`. Every `*Ref` is a typed ID/path/SHA-256 tuple; no bare path or
ambient lookup is valid.

`stc release verify` validates the `ReleaseSealRecord`, proves that the final
payload projection is byte-identical to the draft inventory, and verifies that
the only additions are the draft-authorized G8/seal envelope. Thus G8 verifies
an executable immutable candidate release, while the later seal records final
bytes without circularly making G8 depend on itself.
In offline mode it also schema-validates the bundled smoke report, recomputes
its draft pairing, verifies the G8 evaluation subject and evaluator
attestation, and proves that `control/G8-input.json` is the exact input consumed
by `gate-chain/G8.json`; no dangling digest is accepted as a substitute for
bundled evidence.

The release manifest inventories every included artifact ID/digest plus its
license ID, license-evidence URL/digest, rights reviewer/date/expiry, and
positive redistribution disposition. The assembler reruns the artifact
allowlist and staged-content scanner logic; a source-level summary cannot
authorize a file.
Nonredistributable source text is absent from both environment/reproduction
archives and is represented only by authoritative URL plus registered short
support summary/anchor.

The release manifest also carries an explicit source-to-release byte map for
every G8-subject leaf. In particular, the frozen candidate record maps to
`control/candidate.json`, the exact signed contact sheet maps to
`audit/all-pages-contact-sheet.png`, and the selected measured systems report
and input index map to `systems/g4-systems-evidence.json` and
`systems/g4-inputs.json`. `systems/g4-evidence-bundle/` contains the
redistribution-safe selected run manifests, receipts, validations, probe/
counter/config records, and typed hardware replay digests needed to validate
those two files offline. When `a100-confirmatory` is selected, that bundle
contains the registered manifests and check-result closure for the exact
prerequisite runs `a100-single-gpu-semantic`,
`a100-single-gpu-resource-smoke`, `a100-single-node-contention`, and
`a100-two-node-gate` as well as the confirmatory manifest. Offline verification
rehashes their ordered prerequisite edges and rejects a dangling NFS-only gate
digest. Large raw measurements may remain in the immutable content store only
when the reproduction manifest records their digest and retrieval authority,
but no direct G8-subject leaf or prerequisite manifest/check closure may be
absent.

- [ ] **Phase A.4: Implement a self-contained presentation handoff**

Only the release assembler's G8-bound output may produce:

```text
publication/handoff/{release_tag}/
  presentation-handoff.json
  presentation-handoff.schema.json
  release-manifest.json
  artifact-dag.jsonl
  source-index.jsonl
  claim-index.jsonl
  evidence-index.jsonl
  result-index.jsonl
  schemas/*.schema.json
  checksums.sha256
  assets/
```

Each index is a minimal, redistribution-safe frozen projection rather than a
pointer back to the research package. The handoff exporter copies the four
byte-identical release indexes; it may not requery or reinterpret the candidate
registry:

```text
source-index: source ID, work/version identity, citation, URL, artifact digest,
  accessed date, rights disposition, short non-claim
claim-index: claim ID, exact statement/status/scope, caveats, typed support,
  counterevidence and result IDs, record digest
evidence-index: evidence/source/version/artifact IDs and digests, relation,
  role, short anchor/support summary, warrant/reproduction strength
result-index: result ID, estimand/unit, estimate/uncertainty, decision,
  logical-cell/resource vector, caveats, artifact digest
```

The envelope fields are exactly `schemaVersion`, `releaseTag`,
`releaseManifestSha256`, `artifactDagSha256`, `sourceIndexSha256`,
`claimIndexSha256`, `evidenceIndexSha256`, `resultIndexSha256`,
`schemaSha256`, and `slides`. Every slide contains exactly:

```text
id
order
kind
headline
audienceQuestion
takeaway
body
visualBrief
assets
sourceRefs
chartSpec
notes
altText
mustKeep
editorialFreedom
researchReleaseTag
```

`researchReleaseTag` equals the envelope tag. The exact nested records are:

```text
assets[]:
  assetId, sha256, sourceRefIds, license, role, crop, altText,
  researchClaimIds, evidenceIds, artifactId, relativePath, reuseOwner
sourceRefs[]:
  sourceRefId, url, citation, accessedAt, researchClaimIds, evidenceIds,
  resultIds, shortCaveat, longCaveat
chartSpec:
  null | {datasetAssetId, datasetSha256, units, transforms, filters, uncertainty}
mustKeep[]:
  {mustKeepId, field, text, canonicalTokenSha256}
editorialFreedom:
  {allowedLayoutOperations, textVariants, reorderConstraint}
editorialFreedom.allowedLayoutOperations[]:
  REFLOW | MOVE_WITHIN_SAFE_AREA | RESIZE_WITHIN_DECLARED_BOUNDS |
  SELECT_PRODUCER_VARIANT | REORDER_WITHIN_GROUP |
  CROP_TO_PRODUCER_BOX
editorialFreedom.textVariants[]:
  {variantId, field, text, canonicalTokenSha256}
editorialFreedom.reorderConstraint:
  null | {
    groupId, minOrder, maxOrder, precedesSlideIds, followsSlideIds
  }
```

Canonical token hashing normalizes Unicode to NFC and line endings to LF while
preserving token order, punctuation, numbers, and citation markers. Only this
evidence producer may author a text variant or reorder constraint. Unknown
operations, variant text/hash drift, out-of-group/precedence reordering, or
loss of any `mustKeep` field/text/hash fails handoff validation.

The envelope binds the release manifest, DAG, four indexes, all bundled
schemas, and assets by digest. Offline validation is sandboxed to the handoff
root and must resolve every slide/source/claim/evidence/result/chart reference
from those files alone; absolute paths, `..` escapes, repository fallbacks,
network fetches, or an undeclared external registry fail. Citation URLs remain
display metadata and are not resolution dependencies. Disallowed assets are
excluded and replaced only by a permitted derived asset or a source reference.

- [ ] **Phase A.5: Run and commit the fixture-only early tooling**

```bash
uv run pytest \
  tests/test_candidate_freeze.py \
  tests/test_attestations.py \
  tests/test_gate_release.py \
  tests/test_release_assembly.py \
  tests/test_presentation_handoff.py \
  tests/test_cli.py -q
uv run stc release verify \
  tests/fixtures/g8-release/publication/releases/stc-fixture-v1 \
  --offline --require-environment --require-sbom --require-reproduction
uv run stc presentation validate \
  tests/fixtures/g8-release/publication/handoff/stc-fixture-v1 \
  --offline
git -C ../.. add research/sleep-time-compute
git -C ../.. diff --cached --check
git -C ../.. commit -m "research: add audited release and handoff tooling"
```

Expected: the valid synthetic fixture passes. Tests fail on candidate writes,
controller/candidate path aliasing, unsigned or wrong-subject attestations,
unresolved major findings, absent/untrusted/wrong-subject G8 evaluator
attestations, a G8 record not binding its draft/smoke digests,
draft mutation after smoke, post-G8 payload mutation or undeclared addition,
invalid release seals, non-G8 release, nondeterministic archives,
environment/SBOM/reproduction omissions, stale DAG nodes, schema drift,
unresolved bundled IDs, any missing candidate/contact-sheet/G4 subject leaf,
repository/network fallback, digest mismatch, and any unlicensed asset. A
two-directory fixture assembly produces identical draft
payload projections, manifests, archives, checksum inventories, and seal
subject digests.

#### Phase B — invoke on the real audited candidate

- [ ] **Phase B.1: Create and freeze the detached scientific candidate**

Run these commands only after Task 13's G6 candidate commit exists. The
annotated tag is created once; never retarget it.

```bash
git -C ../.. diff --quiet
git -C ../.. diff --cached --quiet
uv run stc sources staged-scan --repo-root ../.. --cached
git -C ../.. tag -a stc-paper-v1.0.0-rc1 HEAD \
  -m "sleep-time compute scientific candidate rc1"
git -C ../.. worktree add --detach \
  research/sleep-time-compute/build/candidate-worktrees/stc-paper-v1.0.0-rc1 \
  stc-paper-v1.0.0-rc1^{commit}
git -C build/candidate-worktrees/stc-paper-v1.0.0-rc1 \
  status --porcelain=v1 --untracked-files=all
uv run stc candidate freeze \
  --repo-root ../.. \
  --tag stc-paper-v1.0.0-rc1 \
  --candidate-worktree build/candidate-worktrees/stc-paper-v1.0.0-rc1 \
  --controller-root . \
  --output manifests/candidates/stc-paper-v1.0.0-rc1.json
```

Expected: the controller has no tracked or staged change, and the staged scan
passes. Unrelated controller-untracked files are deliberately ignored,
preserved, and never staged or deleted. The detached candidate status command
emits nothing, and the candidate record binds the peeled annotated tag and
exact clean detached tree. The record is written only under the controller.

- [ ] **Phase B.2: Verify attestations and pass G7**

The independent reviewer supplies the report, typed findings/adjudications,
re-audit manifest, signed roster, auditor attestation, public independence
protocol, and trusted keys before this step.

```bash
uv run stc attest verify \
  --attestation manifests/attestations/author-executor-roster.json \
  --trusted-keys manifests/trusted-reviewer-keys.json
uv run stc attest verify \
  --attestation manifests/attestations/G7-auditor.json \
  --trusted-keys manifests/trusted-reviewer-keys.json
uv run stc gate inputs assemble G7 \
  --root . \
  --predecessor manifests/gates/G6.json \
  --candidate manifests/candidates/stc-paper-v1.0.0-rc1.json \
  --audit reports/G7-independent-audit.md \
  --findings registry/audit-findings.jsonl \
  --adjudications registry/audit-adjudications.jsonl \
  --reaudit manifests/reaudits/G7.json \
  --auditor-attestation manifests/attestations/G7-auditor.json \
  --author-executor-roster manifests/attestations/author-executor-roster.json \
  --independence-protocol manifests/G7-independence-protocol.json \
  --trusted-keys manifests/trusted-reviewer-keys.json \
  --output manifests/gate-inputs/G7.json
uv run stc gate evaluate G7 \
  --inputs manifests/gate-inputs/G7.json \
  --output manifests/gates/G7.json
```

- [ ] **Phase B.3: Replay, obtain visual and gate-evaluator approval, and pass G8**

The replay command creates a separate disposable detached worktree and isolated
caches, rebuilds there, and copies only checksummed reports/PDFs to the
controller. It cannot reuse or write into the frozen candidate worktree. After
machine QA, a human reviews every rendered page/contact sheet and supplies the
signature-valid approval record defined in Task 11.

```bash
uv run stc release replay \
  --candidate manifests/candidates/stc-paper-v1.0.0-rc1.json \
  --repo-root ../.. \
  --replay-worktree build/replay-worktrees/stc-paper-v1.0.0-rc1 \
  --controller-root . \
  --output publication/replay/stc-paper-v1.0.0-rc1
uv run stc publication visual-qa \
  --english publication/replay/stc-paper-v1.0.0-rc1/papers/english.pdf \
  --korean publication/replay/stc-paper-v1.0.0-rc1/papers/korean.pdf \
  --english-log publication/replay/stc-paper-v1.0.0-rc1/logs/english.log \
  --korean-log publication/replay/stc-paper-v1.0.0-rc1/logs/korean.log \
  --output publication/reviews/visual-qa-machine.json \
  --contact-sheet publication/reviews/all-pages-contact-sheet.png \
  --allow-pending-human
```

Pause here. A human reviews the exact two PDF digests and every contact-sheet
page, then creates and signs
`publication/reviews/human-visual-approval.json`. Automation must not continue
past this boundary until that external approval exists.

```bash
uv run stc attest verify \
  --attestation publication/reviews/human-visual-approval.json \
  --trusted-keys manifests/trusted-reviewer-keys.json
uv run stc publication visual-qa \
  --english publication/replay/stc-paper-v1.0.0-rc1/papers/english.pdf \
  --korean publication/replay/stc-paper-v1.0.0-rc1/papers/korean.pdf \
  --english-log publication/replay/stc-paper-v1.0.0-rc1/logs/english.log \
  --korean-log publication/replay/stc-paper-v1.0.0-rc1/logs/korean.log \
  --human-review publication/reviews/human-visual-approval.json \
  --output publication/reviews/visual-qa-final.json \
  --contact-sheet publication/reviews/all-pages-contact-sheet.png
uv run stc release draft-manifest \
  --candidate manifests/candidates/stc-paper-v1.0.0-rc1.json \
  --g7 manifests/gates/G7.json \
  --public-replay publication/replay/stc-paper-v1.0.0-rc1/replay-report.json \
  --machine-visual-qa publication/reviews/visual-qa-machine.json \
  --final-visual-qa publication/reviews/visual-qa-final.json \
  --contact-sheet publication/reviews/all-pages-contact-sheet.png \
  --human-approval publication/reviews/human-visual-approval.json \
  --systems-evidence reports/g4-systems-evidence.json \
  --systems-input-index manifests/systems/g4-inputs.json \
  --release-tag stc-paper-v1.0.0 \
  --output manifests/release-candidates/stc-paper-v1.0.0.json
uv run stc release smoke \
  --draft manifests/release-candidates/stc-paper-v1.0.0.json \
  --candidate-worktree build/candidate-worktrees/stc-paper-v1.0.0-rc1 \
  --controller-root . \
  --scratch-root build/release-smoke/stc-paper-v1.0.0 \
  --output reports/release-candidate-smoke.json \
  --offline
uv run stc gate inputs subject G8 \
  --root . \
  --predecessor manifests/gates/G7.json \
  --candidate manifests/candidates/stc-paper-v1.0.0-rc1.json \
  --public-replay publication/replay/stc-paper-v1.0.0-rc1/replay-report.json \
  --machine-visual-review publication/reviews/visual-qa-machine.json \
  --final-visual-review publication/reviews/visual-qa-final.json \
  --contact-sheet publication/reviews/all-pages-contact-sheet.png \
  --human-approval publication/reviews/human-visual-approval.json \
  --release-candidate-manifest manifests/release-candidates/stc-paper-v1.0.0.json \
  --release-smoke reports/release-candidate-smoke.json \
  --target-a100-evidence reports/g4-systems-evidence.json \
  --target-a100-input-index manifests/systems/g4-inputs.json \
  --target-a100-run-id a100-confirmatory \
  --trusted-keys manifests/trusted-reviewer-keys.json \
  --output manifests/gate-inputs/G8-evaluation-subject.json
```

Pause again. An identified trusted `release_evaluator` reviews the canonical
subject, candidate/replay, draft inventory, smoke details, rights, visual
approvals, and measured A100 evidence, then creates and signs
`manifests/attestations/G8-evaluator.json`. Automation must not continue if the
record is absent, invalid, or names a different subject digest.

```bash
uv run stc attest verify \
  --attestation manifests/attestations/G8-evaluator.json \
  --subject manifests/gate-inputs/G8-evaluation-subject.json \
  --trusted-keys manifests/trusted-reviewer-keys.json
uv run stc gate inputs assemble G8 \
  --root . \
  --subject manifests/gate-inputs/G8-evaluation-subject.json \
  --evaluator-attestation manifests/attestations/G8-evaluator.json \
  --trusted-keys manifests/trusted-reviewer-keys.json \
  --output manifests/gate-inputs/G8.json
uv run stc gate evaluate G8 \
  --inputs manifests/gate-inputs/G8.json \
  --output manifests/gates/G8.json
git -C ../.. add \
  research/sleep-time-compute/manifests \
  research/sleep-time-compute/reports/G7-independent-audit.md \
  research/sleep-time-compute/registry/audit-findings.jsonl \
  research/sleep-time-compute/registry/audit-adjudications.jsonl \
  research/sleep-time-compute/reports/release-candidate-smoke.json \
  research/sleep-time-compute/publication/replay \
  research/sleep-time-compute/publication/reviews
uv run stc sources staged-scan --repo-root ../.. --cached
git -C ../.. diff --cached --check
git -C ../.. commit -m "research: record independent G7 and publication G8"
```

Expected: the controller is clean at a committed audit-control SHA after G8;
the separate scientific-candidate tag/commit is unchanged. The committed
pre-seal manifest, smoke report, evaluation subject, and evaluator attestation
have the exact digests embedded in G8. Release assembly records both identities
and reads no uncommitted audit input.

- [ ] **Phase B.4: Assemble and verify the deterministic research release**

```bash
uv run stc release assemble \
  --candidate manifests/candidates/stc-paper-v1.0.0-rc1.json \
  --draft manifests/release-candidates/stc-paper-v1.0.0.json \
  --release-smoke reports/release-candidate-smoke.json \
  --g8 manifests/gates/G8.json \
  --candidate-worktree build/candidate-worktrees/stc-paper-v1.0.0-rc1 \
  --controller-root . \
  --release-tag stc-paper-v1.0.0 \
  --output publication/releases/stc-paper-v1.0.0 \
  --seal-output publication/releases/stc-paper-v1.0.0/release-seal.json \
  --source-date-epoch-from-candidate-tag \
  --archive-format tar.zst
uv run stc release verify \
  publication/releases/stc-paper-v1.0.0 \
  --offline --require-environment --require-sbom --require-reproduction
```

Expected: clean-room verification reproduces every non-hardware artifact or
matches a declared hardware replay digest, and the research-release checksum
inventory is complete. Verification also proves draft-payload byte identity,
permits only the declared G8/seal additions, and validates the terminal
`ReleaseSealRecord`. No presentation handoff is created in this phase.

- [ ] **Phase B.5: Commit and tag the controller-owned research release**

```bash
git -C ../.. add \
  research/sleep-time-compute/manifests \
  research/sleep-time-compute/publication/releases
uv run stc sources staged-scan --repo-root ../.. --cached
git -C ../.. diff --cached --check
uv run stc release verify \
  publication/releases/stc-paper-v1.0.0 \
  --offline --require-environment --require-sbom --require-reproduction
git -C ../.. commit -m "research: record audited sleep-time release"
git -C ../.. tag -a stc-paper-v1.0.0 HEAD \
  -m "audited sleep-time compute research release v1.0.0"
git -C ../.. cat-file -t stc-paper-v1.0.0
```

The release manifest names the previously committed
`scientific_candidate_sha` and later committed `audit_control_sha`; it never
claims the SHA of this still-later controller-owned release-manifest commit.
Candidate/replay/G5-clean worktrees and caches remain ignored and are never
staged.

Expected: `cat-file` prints `tag`; the annotated `stc-paper-v1.0.0` tag points
to the committed research release. Research completion is now independent of
all presentation/PPTX work.

- [ ] **Phase B.6: Export the nonblocking handoff after the research tag**

This phase is invoked only by master Task 9 after Phase B.5 and the annotated
research tag exist. Handoff failure can block presentation work but cannot
invalidate, move, or replace the research release/tag.

```bash
git -C ../.. diff --quiet
git -C ../.. diff --cached --quiet
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

Expected: the exporter verifies the annotated research tag and byte-identical
release indexes, then the four handoff indexes resolve every PPTX-facing
reference without repository or network access. Any failure leaves the
research release commit and tag untouched.
