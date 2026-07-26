from __future__ import annotations

import hashlib
import json
import re
from calendar import monthrange
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError, ValidationError
from referencing import Registry, Resource

from stc_research.ids import parse_stable_id
from stc_research.jsonl_store import read_jsonl
from stc_research.models import (
    ArtifactNode,
    AuditAttestation,
    ClaimRecord,
    EvidenceRecord,
    GateEvaluationAttestation,
    GateEvaluationInput,
    GateRecord,
    HumanApproval,
    HypothesisRecord,
    PaperCard,
    QuestionRecord,
    ReleaseStatus,
    ReviewStatus,
    SourceRecord,
    TrustedKeySet,
)

_CONTROL_SCHEMA_BASE_ID = "https://schemas.sleep-time-compute.org/control"
_DRAFT_2020_12_URI = "https://json-schema.org/draft/2020-12/schema"
_COMMON_SHARED_DEFINITION_NAMES = frozenset(
    {
        "SubjectRef",
        "SourceVersion",
        "ArtifactRef",
        "ClaimResultProvenance",
        "stableId",
        "sha256",
        "awareDateTime",
        "sourceGrade",
        "reviewStatus",
        "warrant",
        "reproductionStrength",
        "resultModality",
        "evidenceRelation",
    }
)
_REQUIRED_SHARED_DEFINITION_REFS = {
    "source.schema.json": {
        ("properties", "versions", "items", "$ref"): "SourceVersion",
        ("properties", "local_artifacts", "items", "$ref"): "ArtifactRef",
        ("properties", "code_records", "items", "$ref"): "ArtifactRef",
        ("properties", "data_records", "items", "$ref"): "ArtifactRef",
    },
    "claim.schema.json": {
        ("properties", "result_provenance", "$ref"): "ClaimResultProvenance",
    },
    "audit-attestation.schema.json": {
        ("properties", "subject_refs", "items", "$ref"): "SubjectRef",
        ("properties", "signed_at", "$ref"): "awareDateTime",
    },
    "gate-evaluation-attestation.schema.json": {
        ("properties", "signed_at", "$ref"): "awareDateTime",
    },
    "human-approval.schema.json": {
        ("properties", "signed_at", "$ref"): "awareDateTime",
    },
    "gate.schema.json": {
        ("properties", "artifact_dag_digest", "$ref"): "sha256",
        ("properties", "evaluated_at", "$ref"): "awareDateTime",
    },
    "evidence.schema.json": {
        ("properties", "relation", "$ref"): "evidenceRelation",
        ("properties", "source_grade", "$ref"): "sourceGrade",
        ("properties", "warrant", "$ref"): "warrant",
        (
            "properties",
            "reproduction_strength",
            "$ref",
        ): "reproductionStrength",
    },
    "negative-result.schema.json": {
        ("properties", "result_modality", "$ref"): "resultModality",
    },
}
_REQUIRED_SHARED_DEFINITION_REFS["source.schema.json"].update(
    {
        ("properties", "source_id", "$ref"): "sourceId",
        ("properties", "source_grade", "$ref"): "sourceGrade",
        ("properties", "review_status", "$ref"): "reviewStatus",
    }
)
_DIAGNOSTIC_ORDER = {
    code: index
    for index, code in enumerate(
        (
            "SCHEMA_INVENTORY_MISMATCH",
            "SCHEMA_ID_DUPLICATE",
            "SCHEMA_METADATA_MISSING",
            "SCHEMA_DRAFT_MISMATCH",
            "SCHEMA_SHARED_DEFINITION_INLINE",
            "SCHEMA_INVALID",
            "RECORD_SCHEMA_INVALID",
            "REFERENCED_PATH_OUTSIDE_ROOT",
            "SOURCE_CANONICAL_KEY_DUPLICATE",
            "SOURCE_VERSION_PREDATES_EARLIEST_PUBLIC_DATE",
            "SOURCE_CANONICAL_CARD_MISSING",
            "EVIDENCE_SOURCE_MISSING",
            "EVIDENCE_SOURCE_VERSION_MISMATCH",
            "EVIDENCE_SOURCE_VERSION_DIGEST_MISMATCH",
            "EVIDENCE_SOURCE_GRADE_MISMATCH",
            "CLAIM_EVIDENCE_MISSING",
            "CLAIM_SUPPORT_REF_MISMATCH",
            "CLAIM_EVIDENCE_REQUIREMENT_UNMET",
            "A_ABS_ANCHOR_OUTSIDE_ABSTRACT",
            "A_ABS_PAPER_CARD_NON_ABSTRACT_EVIDENCE",
            "POST_G2_CONFIRMATORY_AMENDMENT",
            "PAPER_C_UNRESOLVED",
            "CLAIM_MANUSCRIPT_LOCATION_MISSING",
            "MANUSCRIPT_SENTENCE_DIGEST_MISMATCH",
            "STALE_RELEASE_ARTIFACT",
            "STALE_HANDOFF_ARTIFACT",
            "NEGATIVE_RESULT_REGISTRY_OMISSION",
            "FULL_REVIEW_ANCHORS_MISSING",
        )
    )
}
_RELEASE_GATES = frozenset({"G6", "G7", "G8"})
_POST_G2_GATES = frozenset({"G3", "G4", "G5", "G6", "G7", "G8"})
_REPRODUCTION_STRENGTH_RANK = {"r0": 0, "r1": 1, "r2": 2, "r3": 3}
_FULL_ANCHOR_PATTERNS = {
    "method": re.compile(r"\b(?:method|methods|methodology)\b", re.IGNORECASE),
    "result_or_experiment": re.compile(
        r"\b(?:result|results|experiment|experiments|experimental)\b",
        re.IGNORECASE,
    ),
    "limitation": re.compile(
        r"\b(?:limitation|limitations)\b",
        re.IGNORECASE,
    ),
}


@dataclass(frozen=True, slots=True)
class ControlSchemaExport:
    filename: str
    schema_id: str
    owner: str
    model_name: str
    model_type: type[object] | None = None


def _control_schema_export(
    filename: str,
    owner: str,
    model_name: str,
    model_type: type[object] | None = None,
) -> ControlSchemaExport:
    return ControlSchemaExport(
        filename=filename,
        schema_id=f"{_CONTROL_SCHEMA_BASE_ID}/{filename}",
        owner=owner,
        model_name=model_name,
        model_type=model_type,
    )


CONTROL_SCHEMA_EXPORT_MANIFEST = (
    _control_schema_export("common.schema.json", "evidence-governance", "CommonDefinitions"),
    _control_schema_export(
        "source.schema.json",
        "evidence-governance",
        "SourceRecord",
        SourceRecord,
    ),
    _control_schema_export("claim-history.schema.json", "evidence-governance", "ClaimHistoryRecord"),
    _control_schema_export(
        "paper-card.schema.json",
        "evidence-governance",
        "PaperCard",
        PaperCard,
    ),
    _control_schema_export(
        "evidence.schema.json",
        "evidence-governance",
        "EvidenceRecord",
        EvidenceRecord,
    ),
    _control_schema_export(
        "claim.schema.json",
        "evidence-governance",
        "ClaimRecord",
        ClaimRecord,
    ),
    _control_schema_export(
        "question.schema.json",
        "research-program",
        "QuestionRecord",
        QuestionRecord,
    ),
    _control_schema_export(
        "hypothesis.schema.json",
        "research-program",
        "HypothesisRecord",
        HypothesisRecord,
    ),
    _control_schema_export("search-protocol.schema.json", "evidence-governance", "SearchProtocol"),
    _control_schema_export("branch-manifest.schema.json", "research-program", "BranchManifest"),
    _control_schema_export(
        "artifact.schema.json",
        "artifact-dag",
        "ArtifactNode",
        ArtifactNode,
    ),
    _control_schema_export("manuscript-link.schema.json", "publication", "ManuscriptLink"),
    _control_schema_export("parity.schema.json", "publication", "ParityRecord"),
    _control_schema_export("publication-asset.schema.json", "publication", "PublicationAsset"),
    _control_schema_export(
        "publication-visual-review.schema.json",
        "publication",
        "PublicationVisualReview",
    ),
    _control_schema_export("equation.schema.json", "publication", "EquationRecord"),
    _control_schema_export("audit-finding.schema.json", "independent-audit", "AuditFinding"),
    _control_schema_export(
        "audit-adjudication.schema.json",
        "independent-audit",
        "AuditAdjudication",
    ),
    _control_schema_export("terminology.schema.json", "evidence-governance", "TerminologyRecord"),
    _control_schema_export("contradiction.schema.json", "evidence-governance", "ContradictionRecord"),
    _control_schema_export("negative-result.schema.json", "research-program", "NegativeResult"),
    _control_schema_export("search-log.schema.json", "evidence-governance", "SearchLog"),
    _control_schema_export("full-read-order.schema.json", "evidence-governance", "FullReadOrder"),
    _control_schema_export("full-read-log.schema.json", "evidence-governance", "FullReadLog"),
    _control_schema_export("experiment-bundle.schema.json", "theory-benchmark", "ExperimentBundle"),
    _control_schema_export(
        "g4-execution-snapshot.schema.json",
        "theory-benchmark",
        "G4ExecutionSnapshot",
    ),
    _control_schema_export(
        "distributed-rerun-request.schema.json",
        "theory-benchmark",
        "DistributedRerunRequest",
    ),
    _control_schema_export("result-block.schema.json", "theory-benchmark", "ResultBlock"),
    _control_schema_export(
        "gate-input.schema.json",
        "gate-engine",
        "GateEvaluationInput",
        GateEvaluationInput,
    ),
    _control_schema_export(
        "gate.schema.json",
        "gate-engine",
        "GateRecord",
        GateRecord,
    ),
    _control_schema_export(
        "audit-attestation.schema.json",
        "independent-audit",
        "AuditAttestation",
        AuditAttestation,
    ),
    _control_schema_export(
        "gate-evaluation-attestation.schema.json",
        "gate-engine",
        "GateEvaluationAttestation",
        GateEvaluationAttestation,
    ),
    _control_schema_export(
        "human-approval.schema.json",
        "publication",
        "HumanApproval",
        HumanApproval,
    ),
    _control_schema_export(
        "trusted-key-set.schema.json",
        "gate-engine",
        "TrustedKeySet",
        TrustedKeySet,
    ),
    _control_schema_export(
        "independence-protocol.schema.json",
        "independent-audit",
        "IndependenceProtocol",
    ),
    _control_schema_export("candidate.schema.json", "release-engineering", "ReleaseCandidate"),
    _control_schema_export("replay-report.schema.json", "release-engineering", "ReplayReport"),
    _control_schema_export(
        "release-environment.schema.json",
        "release-engineering",
        "ReleaseEnvironment",
    ),
    _control_schema_export(
        "reproduction-manifest.schema.json",
        "release-engineering",
        "ReproductionManifest",
    ),
    _control_schema_export("sbom-spdx.schema.json", "release-engineering", "SpdxDocument"),
    _control_schema_export(
        "release-manifest.schema.json",
        "release-engineering",
        "ReleaseManifest",
    ),
    _control_schema_export("handoff-index.schema.json", "publication", "HandoffIndex"),
    _control_schema_export(
        "presentation-handoff.schema.json",
        "publication",
        "PresentationHandoff",
    ),
)


@dataclass(frozen=True, slots=True)
class ValidationDiagnostic:
    code: str
    registry: str
    record_id: str
    field: str
    related_id: str | None
    message: str


@dataclass(frozen=True, slots=True)
class ValidationReport:
    diagnostics: tuple[ValidationDiagnostic, ...]
    schema_inventory: str

    @property
    def success(self) -> bool:
        return not self.diagnostics

    @property
    def diagnostic_count(self) -> int:
        return len(self.diagnostics)


def validate_repository(root: Path, gate: str | None = None) -> ValidationReport:
    repository_root = Path(root)
    registry = repository_root / "registry"
    sources = read_jsonl(registry / "sources.jsonl", SourceRecord.from_dict)
    evidence = read_jsonl(registry / "evidence.jsonl", EvidenceRecord.from_dict)
    claims = read_jsonl(registry / "claims.jsonl", ClaimRecord.from_dict)
    artifacts = read_jsonl(registry / "artifacts.jsonl", ArtifactNode.from_dict)
    raw_negative_results = read_jsonl(
        registry / "negative-results.jsonl",
        lambda payload: payload,
    )
    hypotheses = read_jsonl(
        registry / "hypotheses.jsonl",
        HypothesisRecord.from_dict,
    )
    manuscript_links_path = (
        repository_root / "publication" / "links" / "manuscript-links.jsonl"
    )
    raw_manuscript_links = (
        read_jsonl(manuscript_links_path, lambda payload: payload)
        if manuscript_links_path.is_file()
        else []
    )

    schema_diagnostics = _validate_control_schema_inventory(repository_root)
    diagnostics: list[ValidationDiagnostic] = list(schema_diagnostics)
    record_validators = (
        _control_record_validators(repository_root)
        if not schema_diagnostics
        else {}
    )
    negative_results = _valid_control_records(
        raw_negative_results,
        validator=record_validators.get("negative-result.schema.json"),
        schema_filename="negative-result.schema.json",
        registry="registry/negative-results.jsonl",
        id_field="result_id",
        diagnostics=diagnostics,
    )
    manuscript_links = _valid_control_records(
        raw_manuscript_links,
        validator=record_validators.get("manuscript-link.schema.json"),
        schema_filename="manuscript-link.schema.json",
        registry="publication/links/manuscript-links.jsonl",
        id_field="claim_id",
        diagnostics=diagnostics,
    )
    sources_by_id = {source.source_id: source for source in sources}
    evidence_by_id = {item.evidence_id: item for item in evidence}
    artifacts_by_id = {artifact.artifact_id: artifact for artifact in artifacts}
    registered_negative_result_ids = {
        result_id
        for item in negative_results
        if isinstance(result_id := item.get("result_id"), str)
    }
    claims_with_manuscript_location = {
        claim_id
        for item in manuscript_links
        if isinstance(claim_id := item.get("claim_id"), str)
        and isinstance(location := item.get("location"), str)
        and bool(location.strip())
    }
    manuscript_links_by_claim: dict[str, list[Mapping[str, Any]]] = {}
    for item in manuscript_links:
        if not isinstance(item, Mapping):
            continue
        claim_id = item.get("claim_id")
        if isinstance(claim_id, str):
            manuscript_links_by_claim.setdefault(claim_id, []).append(item)

    canonical_key_owners: dict[str, str] = {}
    for source in sources:
        owner_id = canonical_key_owners.get(source.canonical_key)
        if owner_id is None:
            canonical_key_owners[source.canonical_key] = source.source_id
        else:
            diagnostics.append(
                ValidationDiagnostic(
                    code="SOURCE_CANONICAL_KEY_DUPLICATE",
                    registry="registry/sources.jsonl",
                    record_id=source.source_id,
                    field="canonical_key",
                    related_id=owner_id,
                    message=(
                        f"source {source.source_id} duplicates canonical_key "
                        f"{source.canonical_key!r} already used by {owner_id}"
                    ),
                )
            )

        earliest_public_date, _ = _public_date_interval(
            source.earliest_public_date,
            source.date_precision,
        )
        for version in source.versions:
            _, version_public_date = _public_date_interval(
                version.public_date,
                version.date_precision,
            )
            if version_public_date < earliest_public_date:
                diagnostics.append(
                    ValidationDiagnostic(
                        code="SOURCE_VERSION_PREDATES_EARLIEST_PUBLIC_DATE",
                        registry="registry/sources.jsonl",
                        record_id=source.source_id,
                        field="earliest_public_date",
                        related_id=version.identifier,
                        message=(
                            f"source {source.source_id} earliest_public_date "
                            f"{source.earliest_public_date} is later than version "
                            f"{version.identifier} public_date {version.public_date}"
                        ),
                    )
                )

    for item in evidence:
        source = sources_by_id.get(item.source_id)
        if source is None:
            diagnostics.append(
                ValidationDiagnostic(
                    code="EVIDENCE_SOURCE_MISSING",
                    registry="registry/evidence.jsonl",
                    record_id=item.evidence_id,
                    field="source_id",
                    related_id=item.source_id,
                    message=(
                        f"evidence {item.evidence_id} references missing source "
                        f"{item.source_id}"
                    ),
                )
            )
            continue

        source_version = next(
            (
                version
                for version in source.versions
                if version.identifier == item.source_version
            ),
            None,
        )
        if source_version is None:
            diagnostics.append(
                ValidationDiagnostic(
                    code="EVIDENCE_SOURCE_VERSION_MISMATCH",
                    registry="registry/evidence.jsonl",
                    record_id=item.evidence_id,
                    field="source_version",
                    related_id=source.source_id,
                    message=(
                        f"evidence {item.evidence_id} references source version "
                        f"{item.source_version} absent from source {source.source_id}"
                    ),
                )
            )
            continue

        if item.source_version_digest != source_version.sha256:
            diagnostics.append(
                ValidationDiagnostic(
                    code="EVIDENCE_SOURCE_VERSION_DIGEST_MISMATCH",
                    registry="registry/evidence.jsonl",
                    record_id=item.evidence_id,
                    field="source_version_digest",
                    related_id=source.source_id,
                    message=(
                        f"evidence {item.evidence_id} source version "
                        f"{item.source_version} digest {item.source_version_digest} "
                        f"does not match source {source.source_id} digest "
                        f"{source_version.sha256}"
                    ),
                )
            )

        if item.source_grade is not source.source_grade:
            diagnostics.append(
                ValidationDiagnostic(
                    code="EVIDENCE_SOURCE_GRADE_MISMATCH",
                    registry="registry/evidence.jsonl",
                    record_id=item.evidence_id,
                    field="source_grade",
                    related_id=source.source_id,
                    message=(
                        f"evidence {item.evidence_id} source_grade "
                        f"{item.source_grade.value!r} does not match source "
                        f"{source.source_id} source_grade "
                        f"{source.source_grade.value!r}"
                    ),
                )
            )

    for claim in claims:
        for evidence_id in dict.fromkeys(
            (*claim.evidence_ids, *(ref.evidence_id for ref in claim.support_refs))
        ):
            if evidence_id not in evidence_by_id:
                diagnostics.append(
                    ValidationDiagnostic(
                        code="CLAIM_EVIDENCE_MISSING",
                        registry="registry/claims.jsonl",
                        record_id=claim.claim_id,
                        field="evidence_ids",
                        related_id=evidence_id,
                        message=(
                            f"claim {claim.claim_id} references missing evidence "
                            f"{evidence_id}"
                        ),
                    )
                )

        qualifying_canonical_keys: set[str] = set()
        has_unresolved_support_ref = False
        for index, ref in enumerate(claim.support_refs):
            item = evidence_by_id.get(ref.evidence_id)
            if item is None:
                has_unresolved_support_ref = True
                continue

            source = sources_by_id.get(item.source_id)
            if source is None:
                has_unresolved_support_ref = True
                continue

            source_version = next(
                (
                    version
                    for version in source.versions
                    if version.identifier == item.source_version
                ),
                None,
            )
            if source_version is None:
                has_unresolved_support_ref = True
                continue

            material_fields = (
                ("source_id", ref.source_id, item.source_id, "source_id"),
                (
                    "source_version",
                    ref.source_version,
                    item.source_version,
                    "source_version",
                ),
                (
                    "source_version_digest",
                    ref.source_version_digest,
                    item.source_version_digest,
                    "source_version_digest",
                ),
                (
                    "artifact_id",
                    ref.artifact_id,
                    source_version.local_artifact_id,
                    "resolved source version local_artifact_id",
                ),
                (
                    "support_span_digest",
                    ref.support_span_digest,
                    item.support_span_digest,
                    "support_span_digest",
                ),
                ("relation", ref.relation, item.relation, "relation"),
                (
                    "source_grade",
                    ref.source_grade,
                    item.source_grade,
                    "source_grade",
                ),
                ("warrant", ref.warrant, item.warrant, "warrant"),
                (
                    "reproduction_strength",
                    ref.reproduction_strength,
                    item.reproduction_strength,
                    "reproduction_strength",
                ),
            )
            exact_profile = True
            for field_name, declared, canonical, canonical_label in material_fields:
                if declared == canonical:
                    continue
                exact_profile = False
                diagnostics.append(
                    ValidationDiagnostic(
                        code="CLAIM_SUPPORT_REF_MISMATCH",
                        registry="registry/claims.jsonl",
                        record_id=claim.claim_id,
                        field=f"support_refs[{index}].{field_name}",
                        related_id=item.evidence_id,
                        message=(
                            f"claim {claim.claim_id} support_refs[{index}]."
                            f"{field_name} "
                            f"{_serialized_string(declared)!r} does not match "
                            f"evidence {item.evidence_id} {canonical_label} "
                            f"{_serialized_string(canonical)!r}"
                        ),
                    )
                )

            requirement = claim.evidence_requirement
            if (
                exact_profile
                and item.source_version_digest == source_version.sha256
                and item.source_grade is source.source_grade
                and ref.role.value == "RELEASE_SUPPORT"
                and item.relation.value == "SUPPORTS"
                and item.source_grade.value != "X"
                and item.source_grade in requirement.allowed_source_grades
                and item.warrant in requirement.allowed_warrants
                and _REPRODUCTION_STRENGTH_RANK[
                    item.reproduction_strength.value
                ]
                >= _REPRODUCTION_STRENGTH_RANK[
                    requirement.minimum_reproduction_strength.value
                ]
            ):
                qualifying_canonical_keys.add(source.canonical_key)

        if gate in _RELEASE_GATES and not has_unresolved_support_ref:
            requirement = claim.evidence_requirement
            canonical_count = len(qualifying_canonical_keys)
            if canonical_count < requirement.minimum_independent_sources:
                diagnostics.append(
                    ValidationDiagnostic(
                        code="CLAIM_EVIDENCE_REQUIREMENT_UNMET",
                        registry="registry/claims.jsonl",
                        record_id=claim.claim_id,
                        field=(
                            "evidence_requirement.minimum_independent_sources"
                        ),
                        related_id=None,
                        message=(
                            f"claim {claim.claim_id} has {canonical_count} "
                            "qualifying independent canonical work(s), below "
                            "minimum_independent_sources="
                            f"{requirement.minimum_independent_sources}"
                        ),
                    )
                )
            elif (
                claim.status is ReleaseStatus.SUPPORTED
                and not qualifying_canonical_keys
            ):
                diagnostics.append(
                    ValidationDiagnostic(
                        code="CLAIM_EVIDENCE_REQUIREMENT_UNMET",
                        registry="registry/claims.jsonl",
                        record_id=claim.claim_id,
                        field="support_refs",
                        related_id=gate,
                        message=(
                            f"supported claim {claim.claim_id} has no exact "
                            "qualifying RELEASE_SUPPORT at release boundary "
                            f"{gate}"
                        ),
                    )
                )

    for item in evidence:
        source = sources_by_id.get(item.source_id)
        if (
            source is not None
            and source.review_status is ReviewStatus.A_ABS
            and not _is_abstract_anchor(item)
        ):
            diagnostics.append(
                ValidationDiagnostic(
                    code="A_ABS_ANCHOR_OUTSIDE_ABSTRACT",
                    registry="registry/evidence.jsonl",
                    record_id=item.evidence_id,
                    field="anchor",
                    related_id=source.source_id,
                    message=(
                        f"A-ABS source {source.source_id} cannot support evidence "
                        f"anchored at {item.anchor!r}"
                    ),
                )
            )

    for source in sources:
        card_relative_path = f"cards/{source.source_id}.json"
        card_path = repository_root / card_relative_path
        if not card_path.is_file():
            diagnostics.append(
                ValidationDiagnostic(
                    code="SOURCE_CANONICAL_CARD_MISSING",
                    registry="registry/sources.jsonl",
                    record_id=source.source_id,
                    field="canonical_card",
                    related_id=card_relative_path,
                    message=(
                        f"source {source.source_id} is missing canonical card "
                        f"{card_relative_path}"
                    ),
                )
            )
            continue

        card = PaperCard.from_dict(
            json.loads(card_path.read_text(encoding="utf-8"))
        )
        if source.review_status is not ReviewStatus.A_ABS:
            continue
        for field_name in ("evidence_ids", "counterevidence_ids"):
            for index, evidence_id in enumerate(getattr(card, field_name)):
                item = evidence_by_id.get(evidence_id)
                if item is None or _is_abstract_anchor(item):
                    continue
                diagnostics.append(
                    ValidationDiagnostic(
                        code="A_ABS_PAPER_CARD_NON_ABSTRACT_EVIDENCE",
                        registry=card_relative_path,
                        record_id=source.source_id,
                        field=f"{field_name}[{index}]",
                        related_id=item.evidence_id,
                        message=(
                            f"A-ABS card {source.source_id} cites evidence "
                            f"{item.evidence_id} with non-abstract anchor "
                            f"{item.anchor!r}"
                        ),
                    )
                )

    for hypothesis in hypotheses:
        for index, amendment in enumerate(hypothesis.amendment_history):
            gate_id = amendment.get("gate_id")
            if (
                not isinstance(gate_id, str)
                or gate_id not in _POST_G2_GATES
                or amendment.get("confirmatory") is not True
            ):
                continue
            diagnostics.append(
                ValidationDiagnostic(
                    code="POST_G2_CONFIRMATORY_AMENDMENT",
                    registry="registry/hypotheses.jsonl",
                    record_id=hypothesis.hypothesis_id,
                    field=f"amendment_history[{index}].confirmatory",
                    related_id=gate_id,
                    message=(
                        f"hypothesis {hypothesis.hypothesis_id} "
                        f"amendment_history[{index}] at {gate_id} cannot remain "
                        "confirmatory after G2"
                    ),
                )
            )

    if gate in _RELEASE_GATES:
        for claim in claims:
            if (
                parse_stable_id(claim.claim_id).kind == "paper_claim"
                and claim.status is ReleaseStatus.UNRESOLVED
            ):
                diagnostics.append(
                    ValidationDiagnostic(
                        code="PAPER_C_UNRESOLVED",
                        registry="registry/claims.jsonl",
                        record_id=claim.claim_id,
                        field="status",
                        related_id=gate,
                        message=(
                            f"PAPER-C claim {claim.claim_id} remains unresolved "
                            f"at release boundary {gate}"
                        ),
                    )
                )
            if (
                parse_stable_id(claim.claim_id).kind == "paper_claim"
                and claim.status
                in {
                    ReleaseStatus.SUPPORTED,
                    ReleaseStatus.FALSIFIED_NARROWED,
                }
            ):
                if claim.claim_id not in claims_with_manuscript_location:
                    diagnostics.append(
                        ValidationDiagnostic(
                            code="CLAIM_MANUSCRIPT_LOCATION_MISSING",
                            registry=(
                                "publication/links/manuscript-links.jsonl"
                            ),
                            record_id=claim.claim_id,
                            field="location",
                            related_id=gate,
                            message=(
                                f"release claim {claim.claim_id} has no manuscript "
                                "location in publication/links/"
                                f"manuscript-links.jsonl at {gate}"
                            ),
                        )
                    )
                for manuscript_link in manuscript_links_by_claim.get(
                    claim.claim_id,
                    (),
                ):
                    manuscript_path = manuscript_link.get("manuscript_path")
                    location = manuscript_link.get("location")
                    sentence_digest = manuscript_link.get("sentence_digest")
                    if not (
                        isinstance(manuscript_path, str)
                        and isinstance(location, str)
                        and isinstance(sentence_digest, str)
                    ):
                        continue
                    contained_path = _contained_repository_path(
                        repository_root,
                        manuscript_path,
                    )
                    if contained_path is None:
                        diagnostics.append(
                            ValidationDiagnostic(
                                code="REFERENCED_PATH_OUTSIDE_ROOT",
                                registry=(
                                    "publication/links/manuscript-links.jsonl"
                                ),
                                record_id=claim.claim_id,
                                field="manuscript_path",
                                related_id=manuscript_path,
                                message=(
                                    f"manuscript link {claim.claim_id} path "
                                    f"{manuscript_path} resolves outside "
                                    "repository root"
                                ),
                            )
                        )
                        continue
                    block = _manuscript_block(contained_path, location)
                    if block is None:
                        continue
                    block_digest = hashlib.sha256(
                        " ".join(block.split()).encode("utf-8")
                    ).hexdigest()
                    if block_digest == sentence_digest:
                        continue
                    diagnostics.append(
                        ValidationDiagnostic(
                            code="MANUSCRIPT_SENTENCE_DIGEST_MISMATCH",
                            registry=(
                                "publication/links/manuscript-links.jsonl"
                            ),
                            record_id=claim.claim_id,
                            field="sentence_digest",
                            related_id=location,
                            message=(
                                f"manuscript link {claim.claim_id} location "
                                f"{location} sentence_digest {sentence_digest} "
                                "does not match normalized block digest "
                                f"{block_digest}"
                            ),
                        )
                    )
                for index, artifact_id in enumerate(
                    claim.artifact_dependencies
                ):
                    artifact = artifacts_by_id.get(artifact_id)
                    if artifact is None or artifact.gate_status != "STALE":
                        continue
                    diagnostics.append(
                        ValidationDiagnostic(
                            code="STALE_RELEASE_ARTIFACT",
                            registry="registry/claims.jsonl",
                            record_id=claim.claim_id,
                            field=f"artifact_dependencies[{index}]",
                            related_id=artifact_id,
                            message=(
                                f"release claim {claim.claim_id} depends on stale "
                                f"artifact {artifact_id} at {gate}"
                            ),
                        )
                    )

    release_directory = repository_root / "publication" / "releases"
    for manifest_path in sorted(
        release_directory.glob("*/release-manifest.json")
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        relative_path = manifest_path.relative_to(repository_root).as_posix()
        record_diagnostics = _control_record_diagnostics(
            manifest,
            validator=record_validators.get("release-manifest.schema.json"),
            schema_filename="release-manifest.schema.json",
            registry=relative_path,
            id_field="release_id",
        )
        diagnostics.extend(record_diagnostics)
        if record_diagnostics:
            continue
        if not isinstance(manifest, Mapping):
            continue
        release_id = manifest.get("release_id")
        artifact_refs = manifest.get("artifact_refs")
        if not isinstance(release_id, str) or not isinstance(
            artifact_refs,
            list,
        ):
            continue
        for index, artifact_ref in enumerate(artifact_refs):
            if not isinstance(artifact_ref, Mapping):
                continue
            artifact_id = artifact_ref.get("artifact_id")
            artifact = (
                artifacts_by_id.get(artifact_id)
                if isinstance(artifact_id, str)
                else None
            )
            if artifact is None or artifact.gate_status != "STALE":
                continue
            diagnostics.append(
                ValidationDiagnostic(
                    code="STALE_HANDOFF_ARTIFACT",
                    registry=relative_path,
                    record_id=release_id,
                    field=f"artifact_refs[{index}].artifact_id",
                    related_id=artifact_id,
                    message=(
                        f"release manifest {release_id} depends on stale "
                        f"artifact {artifact_id}"
                    ),
                )
            )

    handoff_directory = repository_root / "publication" / "handoff"
    for handoff_path in sorted(
        handoff_directory.glob("*/presentation-handoff.json")
    ):
        handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
        relative_path = handoff_path.relative_to(repository_root).as_posix()
        record_diagnostics = _control_record_diagnostics(
            handoff,
            validator=record_validators.get(
                "presentation-handoff.schema.json"
            ),
            schema_filename="presentation-handoff.schema.json",
            registry=relative_path,
            id_field="presentation_id",
        )
        diagnostics.extend(record_diagnostics)
        if record_diagnostics:
            continue
        if not isinstance(handoff, Mapping):
            continue
        presentation_id = handoff.get("presentation_id")
        gate_ref = handoff.get("g8_gate_ref")
        if not isinstance(presentation_id, str) or not isinstance(
            gate_ref,
            Mapping,
        ):
            continue
        artifact_id = gate_ref.get("artifact_id")
        artifact = (
            artifacts_by_id.get(artifact_id)
            if isinstance(artifact_id, str)
            else None
        )
        if artifact is None or artifact.gate_status != "STALE":
            continue
        diagnostics.append(
            ValidationDiagnostic(
                code="STALE_HANDOFF_ARTIFACT",
                registry=relative_path,
                record_id=presentation_id,
                field="g8_gate_ref.artifact_id",
                related_id=artifact_id,
                message=(
                    f"presentation handoff {presentation_id} depends on stale "
                    f"artifact {artifact_id}"
                ),
            )
        )

    for claim in claims:
        if claim.status is not ReleaseStatus.FALSIFIED_NARROWED:
            continue
        for index, result_id in enumerate(claim.result_ids):
            if result_id in registered_negative_result_ids:
                continue
            diagnostics.append(
                ValidationDiagnostic(
                    code="NEGATIVE_RESULT_REGISTRY_OMISSION",
                    registry="registry/claims.jsonl",
                    record_id=claim.claim_id,
                    field=f"result_ids[{index}]",
                    related_id=result_id,
                    message=(
                        f"FALSIFIED/NARROWED claim {claim.claim_id} result "
                        f"{result_id} is absent from "
                        "registry/negative-results.jsonl"
                    ),
                )
            )

    evidence_by_source: dict[str, list[EvidenceRecord]] = {}
    for item in evidence:
        evidence_by_source.setdefault(item.source_id, []).append(item)
    for source in sources:
        if source.review_status is not ReviewStatus.FULL:
            continue
        anchored_categories = {
            category
            for item in evidence_by_source.get(source.source_id, ())
            for category, pattern in _FULL_ANCHOR_PATTERNS.items()
            if pattern.search(f"{item.anchor_kind} {item.anchor}")
        }
        missing_categories = tuple(
            category
            for category in _FULL_ANCHOR_PATTERNS
            if category not in anchored_categories
        )
        if missing_categories:
            diagnostics.append(
                ValidationDiagnostic(
                    code="FULL_REVIEW_ANCHORS_MISSING",
                    registry="registry/sources.jsonl",
                    record_id=source.source_id,
                    field="review_status",
                    related_id=None,
                    message=(
                        f"FULL source {source.source_id} lacks evidence anchors for "
                        f"{', '.join(missing_categories)}"
                    ),
                )
            )

    diagnostics.sort(key=_diagnostic_sort_key)
    return ValidationReport(
        diagnostics=tuple(diagnostics),
        schema_inventory="incomplete" if schema_diagnostics else "complete",
    )


def _control_record_validators(
    repository_root: Path,
) -> dict[str, Draft202012Validator]:
    schemas: dict[str, Mapping[str, Any]] = {}
    resources: list[tuple[str, Resource[Any]]] = []
    schema_directory = repository_root / "schemas" / "control"
    for path in sorted(schema_directory.glob("*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(schema, Mapping):
            return {}
        schema_id = schema.get("$id")
        if not isinstance(schema_id, str):
            return {}
        schemas[path.name] = schema
        resources.append((schema_id, Resource.from_contents(schema)))

    registry = Registry().with_resources(resources)
    format_checker = FormatChecker()
    return {
        filename: Draft202012Validator(
            schemas[filename],
            registry=registry,
            format_checker=format_checker,
        )
        for filename in (
            "manuscript-link.schema.json",
            "negative-result.schema.json",
            "presentation-handoff.schema.json",
            "release-manifest.schema.json",
        )
        if filename in schemas
    }


def _valid_control_records(
    records: list[dict[str, Any]],
    *,
    validator: Draft202012Validator | None,
    schema_filename: str,
    registry: str,
    id_field: str,
    diagnostics: list[ValidationDiagnostic],
) -> list[dict[str, Any]]:
    valid_records: list[dict[str, Any]] = []
    for record in records:
        record_diagnostics = _control_record_diagnostics(
            record,
            validator=validator,
            schema_filename=schema_filename,
            registry=registry,
            id_field=id_field,
        )
        diagnostics.extend(record_diagnostics)
        if not record_diagnostics:
            valid_records.append(record)
    return valid_records


def _control_record_diagnostics(
    record: object,
    *,
    validator: Draft202012Validator | None,
    schema_filename: str,
    registry: str,
    id_field: str,
) -> tuple[ValidationDiagnostic, ...]:
    if validator is None:
        return ()
    record_id = _control_record_id(record, id_field)
    errors = sorted(validator.iter_errors(record), key=_schema_error_sort_key)
    return tuple(
        ValidationDiagnostic(
            code="RECORD_SCHEMA_INVALID",
            registry=registry,
            record_id=record_id,
            field=_schema_error_field(error),
            related_id=schema_filename,
            message=(
                f"record {record_id} violates {schema_filename} at "
                f"{_schema_error_field(error)} ({error.validator})"
            ),
        )
        for error in errors
    )


def _control_record_id(record: object, id_field: str) -> str:
    if isinstance(record, Mapping):
        value = record.get(id_field)
        if isinstance(value, str) and value:
            return value
    return "<unknown>"


def _schema_error_sort_key(
    error: ValidationError,
) -> tuple[tuple[str, ...], str, str]:
    return (
        tuple(str(part) for part in error.absolute_path),
        str(error.validator),
        error.message,
    )


def _schema_error_field(error: ValidationError) -> str:
    field = "$"
    for part in error.absolute_path:
        if isinstance(part, int):
            field = f"{field}[{part}]"
        else:
            field = str(part) if field == "$" else f"{field}.{part}"
    return field


def _validate_control_schema_inventory(
    repository_root: Path,
) -> tuple[ValidationDiagnostic, ...]:
    schema_directory = repository_root / "schemas" / "control"
    paths = tuple(sorted(schema_directory.glob("*.schema.json")))
    manifest_by_filename = {
        entry.filename: entry for entry in CONTROL_SCHEMA_EXPORT_MANIFEST
    }
    expected_filenames = frozenset(manifest_by_filename)
    discovered_filenames = frozenset(path.name for path in paths)
    diagnostics: list[ValidationDiagnostic] = []

    if discovered_filenames != expected_filenames:
        missing = sorted(expected_filenames - discovered_filenames)
        unexpected = sorted(discovered_filenames - expected_filenames)
        diagnostics.append(
            _schema_diagnostic(
                code="SCHEMA_INVENTORY_MISMATCH",
                record_id="control",
                field="filenames",
                message=(
                    "control schema inventory differs from the export manifest; "
                    f"missing={missing!r}, unexpected={unexpected!r}"
                ),
            )
        )

    schemas: dict[str, Mapping[str, Any]] = {}
    for path in paths:
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            diagnostics.append(
                _schema_diagnostic(
                    code="SCHEMA_INVALID",
                    record_id=path.name,
                    field="$",
                    message=f"control schema is not valid UTF-8 JSON: {error}",
                )
            )
            continue
        if not isinstance(parsed, Mapping):
            diagnostics.append(
                _schema_diagnostic(
                    code="SCHEMA_INVALID",
                    record_id=path.name,
                    field="$",
                    message="control schema root must be a JSON object",
                )
            )
            continue
        schemas[path.name] = parsed

    schema_id_counts = Counter(
        schema_id
        for schema in schemas.values()
        if isinstance(schema_id := schema.get("$id"), str) and schema_id
    )
    for filename, schema in schemas.items():
        schema_id = schema.get("$id")
        if isinstance(schema_id, str) and schema_id_counts[schema_id] > 1:
            diagnostics.append(
                _schema_diagnostic(
                    code="SCHEMA_ID_DUPLICATE",
                    record_id=filename,
                    field="$id",
                    related_id=schema_id,
                    message=f"control schema ID is declared by multiple files: {schema_id}",
                )
            )

        entry = manifest_by_filename.get(filename)
        if entry is not None and schema_id != entry.schema_id:
            diagnostics.append(
                _schema_diagnostic(
                    code="SCHEMA_INVENTORY_MISMATCH",
                    record_id=filename,
                    field="$id",
                    related_id=str(schema_id) if schema_id is not None else None,
                    message=(
                        f"{filename} must declare manifest schema ID "
                        f"{entry.schema_id!r}"
                    ),
                )
            )

        if schema.get("$schema") != _DRAFT_2020_12_URI:
            diagnostics.append(
                _schema_diagnostic(
                    code="SCHEMA_DRAFT_MISMATCH",
                    record_id=filename,
                    field="$schema",
                    related_id=(
                        str(schema.get("$schema"))
                        if schema.get("$schema") is not None
                        else None
                    ),
                    message=f"{filename} must use Draft 2020-12",
                )
            )

        for metadata_field, manifest_value in (
            ("x-stc-owner", entry.owner if entry is not None else None),
            ("x-stc-model", entry.model_name if entry is not None else None),
        ):
            value = schema.get(metadata_field)
            if (
                not isinstance(value, str)
                or not value.strip()
                or (manifest_value is not None and value != manifest_value)
            ):
                diagnostics.append(
                    _schema_diagnostic(
                        code="SCHEMA_METADATA_MISSING",
                        record_id=filename,
                        field=metadata_field,
                        related_id=str(value) if value is not None else None,
                        message=(
                            f"{filename} must declare {metadata_field}="
                            f"{manifest_value!r}"
                        ),
                    )
                )

        if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
            diagnostics.append(
                _schema_diagnostic(
                    code="SCHEMA_INVALID",
                    record_id=filename,
                    field="$",
                    message=(
                        f"{filename} must be a closed root object schema"
                    ),
                )
            )
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as error:
            diagnostics.append(
                _schema_diagnostic(
                    code="SCHEMA_INVALID",
                    record_id=filename,
                    field="$",
                    message=f"{filename} is not a valid Draft 2020-12 schema: {error.message}",
                )
            )

    diagnostics.extend(_shared_definition_diagnostics(schemas))
    diagnostics.sort(key=_diagnostic_sort_key)
    return tuple(diagnostics)


def _shared_definition_diagnostics(
    schemas: Mapping[str, Mapping[str, Any]],
) -> tuple[ValidationDiagnostic, ...]:
    diagnostics: list[ValidationDiagnostic] = []
    common = schemas.get("common.schema.json")
    common_definitions = common.get("$defs") if common is not None else None
    if not isinstance(common_definitions, Mapping):
        common_definitions = {}
    missing_definitions = sorted(
        _COMMON_SHARED_DEFINITION_NAMES - set(common_definitions)
    )
    if missing_definitions:
        diagnostics.append(
            _schema_diagnostic(
                code="SCHEMA_SHARED_DEFINITION_INLINE",
                record_id="common.schema.json",
                field="$defs",
                message=(
                    "common schema is missing shared definitions: "
                    f"{missing_definitions!r}"
                ),
            )
        )

    for filename in sorted(schemas):
        if filename == "common.schema.json":
            continue
        schema = schemas[filename]
        definitions = schema.get("$defs", {})
        copied_names = (
            sorted(_COMMON_SHARED_DEFINITION_NAMES & set(definitions))
            if isinstance(definitions, Mapping)
            else []
        )
        if copied_names:
            diagnostics.append(
                _schema_diagnostic(
                    code="SCHEMA_SHARED_DEFINITION_INLINE",
                    record_id=filename,
                    field="$defs",
                    message=(
                        f"{filename} redeclares common definitions: "
                        f"{copied_names!r}"
                    ),
                )
            )
        for pointer, definition_name in _inline_shared_definition_copies(
            schema,
            common_definitions,
        ):
            diagnostics.append(
                _schema_diagnostic(
                    code="SCHEMA_SHARED_DEFINITION_INLINE",
                    record_id=filename,
                    field=pointer,
                    related_id=definition_name,
                    message=(
                        f"{filename} copies common definition {definition_name} "
                        f"inline at JSON pointer {pointer}"
                    ),
                )
            )

    common_id = (
        common.get("$id")
        if common is not None and isinstance(common.get("$id"), str)
        else f"{_CONTROL_SCHEMA_BASE_ID}/common.schema.json"
    )
    for filename, path_bindings in _REQUIRED_SHARED_DEFINITION_REFS.items():
        schema = schemas.get(filename)
        if schema is None:
            continue
        for path, definition_name in path_bindings.items():
            expected_ref = f"{common_id}#/$defs/{definition_name}"
            if _schema_value_at_path(schema, path) != expected_ref:
                diagnostics.append(
                    _schema_diagnostic(
                        code="SCHEMA_SHARED_DEFINITION_INLINE",
                        record_id=filename,
                        field=".".join(path[:-1]),
                        related_id=definition_name,
                        message=(
                            f"{filename} must bind {'.'.join(path[:-1])} "
                            f"directly to {expected_ref}"
                        ),
                    )
                )
    return tuple(diagnostics)


def _inline_shared_definition_copies(
    schema: Mapping[str, Any],
    common_definitions: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    mandated_definitions = tuple(
        (name, common_definitions[name])
        for name in sorted(_COMMON_SHARED_DEFINITION_NAMES)
        if isinstance(common_definitions.get(name), Mapping)
    )
    copies: list[tuple[str, str]] = []
    for path, node in _walk_schema_nodes(schema):
        if not isinstance(node, Mapping):
            continue
        for definition_name, definition in mandated_definitions:
            if node == definition:
                copies.append((_json_pointer(path), definition_name))
    return tuple(copies)


def _walk_schema_nodes(
    value: Any,
    path: tuple[str, ...] = (),
) -> tuple[tuple[tuple[str, ...], Any], ...]:
    nodes: list[tuple[tuple[str, ...], Any]] = [(path, value)]
    if isinstance(value, Mapping):
        for key in sorted(value):
            nodes.extend(_walk_schema_nodes(value[key], (*path, key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            nodes.extend(_walk_schema_nodes(item, (*path, str(index))))
    return tuple(nodes)


def _json_pointer(path: tuple[str, ...]) -> str:
    if not path:
        return ""
    return "/" + "/".join(
        segment.replace("~", "~0").replace("/", "~1") for segment in path
    )


def _schema_value_at_path(
    schema: Mapping[str, Any],
    path: tuple[str, ...],
) -> Any:
    value: Any = schema
    for segment in path:
        if not isinstance(value, Mapping) or segment not in value:
            return None
        value = value[segment]
    return value


def _schema_diagnostic(
    *,
    code: str,
    record_id: str,
    field: str,
    message: str,
    related_id: str | None = None,
) -> ValidationDiagnostic:
    return ValidationDiagnostic(
        code=code,
        registry="schemas/control",
        record_id=record_id,
        field=field,
        related_id=related_id,
        message=message,
    )


def _is_abstract_anchor(evidence: EvidenceRecord) -> bool:
    anchor_kind = evidence.anchor_kind.strip().casefold()
    anchor = evidence.anchor.strip().casefold()
    return anchor_kind == "abstract" or re.match(r"^abstract\b", anchor) is not None


def _manuscript_block(path: Path, location: str) -> str | None:
    if not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    begin_marker = f"% STC:BEGIN {location}"
    end_marker = f"% STC:END {location}"
    begin_index: int | None = None
    for index, line in enumerate(lines):
        marker = line.rstrip("\r\n")
        if begin_index is None:
            if marker == begin_marker:
                begin_index = index + 1
            continue
        if marker == end_marker:
            return "".join(lines[begin_index:index])
    return None


def _contained_repository_path(
    repository_root: Path,
    relative_path: str,
) -> Path | None:
    try:
        resolved_root = repository_root.resolve()
        resolved_path = (repository_root / relative_path).resolve()
        resolved_path.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved_path


def _serialized_string(value: object) -> str:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, str):
        return value
    raise TypeError(f"expected a serialized string value, got {type(value).__name__}")


def _public_date_interval(value: str, precision: str) -> tuple[date, date]:
    if precision == "YEAR":
        year = int(value)
        return date(year, 1, 1), date(year, 12, 31)
    if precision == "MONTH":
        year, month = value.split("-")
        parsed_year = int(year)
        parsed_month = int(month)
        return (
            date(parsed_year, parsed_month, 1),
            date(
                parsed_year,
                parsed_month,
                monthrange(parsed_year, parsed_month)[1],
            ),
        )
    if precision == "DAY":
        parsed = date.fromisoformat(value)
        return parsed, parsed
    raise ValueError(f"unsupported public date precision: {precision}")


def _diagnostic_sort_key(
    diagnostic: ValidationDiagnostic,
) -> tuple[int, str, str, str, str, str]:
    return (
        _DIAGNOSTIC_ORDER[diagnostic.code],
        diagnostic.registry,
        diagnostic.record_id,
        diagnostic.field,
        diagnostic.related_id or "",
        diagnostic.message,
    )
