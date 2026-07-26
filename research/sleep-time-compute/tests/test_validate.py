import hashlib
import json
import shutil
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from urllib.parse import urlparse

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from stc_research.models import (
    ArtifactNode,
    ArtifactRef,
    AuditAttestation,
    ClaimEvidenceRequirement,
    ClaimRecord,
    ClaimResultProvenance,
    ClaimSupportRef,
    DigestRef,
    EvidenceRecord,
    GateEvaluationAttestation,
    GateRecord,
    HumanApproval,
    HypothesisRecord,
    PaperCard,
    QuestionRecord,
    SourceRecord,
    SourceVersion,
    SubjectRef,
)
from stc_research.validate import CONTROL_SCHEMA_EXPORT_MANIFEST, validate_repository

PACKAGE_ROOT = Path(__file__).parents[1]
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
MANUSCRIPT_LOCATION = "EN-S0001"
MANUSCRIPT_SENTENCE = "The reviewed source supports this bounded release claim."
MANUSCRIPT_NONCANONICAL_BLOCK = (
    "The reviewed\t source\nsupports   this bounded release claim."
)
ARTIFACT_PAYLOAD = b'{"claim_support":"bounded"}\n'
RELEASE_ARTIFACT_PAYLOAD = b'{"release_payload":"bounded"}\n'
GATE_ARTIFACT_PAYLOAD = b'{"gate":"G8"}\n'
DRAFT_2020_12_URI = "https://json-schema.org/draft/2020-12/schema"
CANONICAL_REGISTRY_JSONL_FILENAMES = (
    "artifacts.jsonl",
    "audit-adjudications.jsonl",
    "audit-findings.jsonl",
    "claim-history.jsonl",
    "claims.jsonl",
    "contradictions.jsonl",
    "evidence.jsonl",
    "hypotheses.jsonl",
    "negative-results.jsonl",
    "questions.jsonl",
    "sources.jsonl",
    "terminology.jsonl",
)


def _trusted_key_set_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "key_set_id": "stc-reviewer-keys",
        "version": 1,
        "keys": [
            {
                "key_id": "release-key-z",
                "signer_id": "release-evaluator-1",
                "algorithm": "Ed25519",
                "public_key": "base64-der-spki-ed25519-public-key-z",
                "roles": ["visual_reviewer", "release_evaluator"],
                "valid_from": "2026-07-01T00:00:00Z",
                "valid_until": "2027-07-01T00:00:00Z",
                "revoked": False,
            },
            {
                "key_id": "audit-key-a",
                "signer_id": "independent-auditor-1",
                "algorithm": "Ed25519",
                "public_key": "base64-der-spki-ed25519-public-key-a",
                "roles": ["quality_reviewer", "independent_auditor"],
                "valid_from": "2026-07-01T00:00:00Z",
                "valid_until": None,
                "revoked": False,
            },
        ],
        "issued_at": "2026-07-01T00:00:00Z",
    }
    _refresh_trusted_key_set_digest(payload)
    return payload


def _refresh_trusted_key_set_digest(payload: dict[str, object]) -> None:
    unsigned = deepcopy(payload)
    unsigned.pop("set_digest", None)
    payload["set_digest"] = hashlib.sha256(
        (
            json.dumps(
                unsigned,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()


EXPECTED_CONTROL_SCHEMA_FILENAMES = frozenset(
    {
        "common.schema.json",
        "source.schema.json",
        "claim-history.schema.json",
        "paper-card.schema.json",
        "evidence.schema.json",
        "claim.schema.json",
        "question.schema.json",
        "hypothesis.schema.json",
        "search-protocol.schema.json",
        "branch-manifest.schema.json",
        "artifact.schema.json",
        "manuscript-link.schema.json",
        "parity.schema.json",
        "publication-asset.schema.json",
        "publication-visual-review.schema.json",
        "equation.schema.json",
        "audit-finding.schema.json",
        "audit-adjudication.schema.json",
        "terminology.schema.json",
        "contradiction.schema.json",
        "negative-result.schema.json",
        "search-log.schema.json",
        "full-read-order.schema.json",
        "full-read-log.schema.json",
        "experiment-bundle.schema.json",
        "g4-execution-snapshot.schema.json",
        "distributed-rerun-request.schema.json",
        "result-block.schema.json",
        "gate-input.schema.json",
        "gate.schema.json",
        "audit-attestation.schema.json",
        "gate-evaluation-attestation.schema.json",
        "human-approval.schema.json",
        "trusted-key-set.schema.json",
        "independence-protocol.schema.json",
        "candidate.schema.json",
        "replay-report.schema.json",
        "release-environment.schema.json",
        "reproduction-manifest.schema.json",
        "sbom-spdx.schema.json",
        "release-manifest.schema.json",
        "handoff-index.schema.json",
        "presentation-handoff.schema.json",
    }
)
COMMON_SHARED_DEFINITION_NAMES = frozenset(
    {
        "SubjectRef",
        "SourceVersion",
        "ArtifactRef",
        "ClaimResultProvenance",
        "stableId",
        "sha256",
        "sourceGrade",
        "reviewStatus",
        "warrant",
        "reproductionStrength",
        "resultModality",
        "evidenceRelation",
    }
)


def _manifest_value(entry: object, name: str) -> object:
    if isinstance(entry, Mapping):
        if name == "model_name":
            return entry.get("model_name", entry.get("model_type"))
        return entry.get(name)
    if name == "model_name":
        return getattr(entry, "model_name", getattr(entry, "model_type", None))
    return getattr(entry, name, None)


def _schema_directory(root: Path) -> Path:
    return root / "schemas" / "control"


def _schema_paths(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(_schema_directory(root).glob("*.schema.json")))


def _read_schema(path: Path) -> dict[str, object]:
    schema = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(schema, dict)
    return schema


def _write_schema(path: Path, schema: Mapping[str, object]) -> None:
    path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _schema_repo(tmp_path: Path) -> Path:
    root = _valid_repo(tmp_path)
    shutil.copytree(PACKAGE_ROOT / "schemas", root / "schemas")
    return root


def _schema_diagnostics(report: object) -> set[str]:
    return {
        diagnostic.code
        for diagnostic in report.diagnostics
        if diagnostic.code.startswith("SCHEMA_")
    }


def _schema_value_at_path(
    schema: Mapping[str, object], *path: str
) -> object:
    value: object = schema
    for segment in path:
        assert isinstance(value, Mapping)
        value = value[segment]
    return value


def _artifact_ref() -> dict[str, object]:
    return {
        "artifact_id": "ART-STC-0001",
        "canonical_url": "https://example.org/paper.pdf",
        "local_path": "artifacts/paper.pdf",
        "sha256": SHA_A,
        "media_type": "application/pdf",
        "availability": "VENDORED",
        "license_id": "CC-BY-4.0",
        "license_evidence_url": "https://example.org/license",
        "license_evidence_sha256": SHA_B,
        "redistribution_allowed": True,
        "license_review_disposition": "ALLOW",
        "rights_reviewer": "rights-reviewer-1",
        "rights_reviewed_at": "2026-07-25",
        "rights_review_expires_at": None,
    }


def _source() -> dict[str, object]:
    return {
        "source_id": "SRC-STC-0001",
        "legacy_ids": ["W1-001"],
        "canonical_key": "doi:10.1000/sleep-work:v1",
        "title": "A Full Sleep-Time Compute Study",
        "authors": ["Author One"],
        "earliest_public_date": "2025-04-17",
        "date_precision": "DAY",
        "venue_status": "PUBLISHED",
        "peer_review_status": "PEER_REVIEWED",
        "versions": [
            {
                "identifier": "v1",
                "public_date": "2025-04-17",
                "date_precision": "DAY",
                "status": "PUBLISHED",
                "canonical_url": "https://example.org/paper",
                "local_artifact_id": "ART-STC-0001",
                "sha256": SHA_A,
                "supersedes": None,
                "derived_from": [],
            }
        ],
        "source_grade": "A",
        "review_status": "FULL",
        "urls": ["https://example.org/paper"],
        "local_artifacts": [_artifact_ref()],
        "citation_key": "author2025sleep",
        "code_records": [],
        "data_records": [],
        "rights_summary": "Reuse allowed by the reviewed artifact record.",
        "topics": ["sleep-time-compute"],
        "affiliations": ["Example Lab"],
        "inclusion_reason": "Primary source with a complete review.",
        "non_claim": "No claim beyond the cited spans.",
        "last_verified": "2026-07-25",
    }


def _evidence(
    number: int,
    *,
    anchor: str,
    claim_ids: list[str] | None = None,
) -> dict[str, object]:
    return {
        "evidence_id": f"EV-STC-{number:05d}",
        "source_id": "SRC-STC-0001",
        "source_version": "v1",
        "source_version_digest": SHA_A,
        "anchor_kind": "SECTION",
        "anchor": anchor,
        "support_span_digest": SHA_B,
        "support_summary": f"Reviewed {anchor.lower()} span.",
        "relation": "SUPPORTS",
        "claim_ids": claim_ids or [],
        "question_ids": [],
        "hypothesis_ids": [],
        "source_grade": "A",
        "warrant": "SUMM",
        "reproduction_strength": "r0",
        "assumptions": [],
        "scope": "The cited source version only.",
        "counterevidence_ids": [],
        "reviewer": "reviewer-1",
        "reviewed_at": "2026-07-25",
    }


def _claim() -> dict[str, object]:
    support_ref = {
        "evidence_id": "EV-STC-00001",
        "source_id": "SRC-STC-0001",
        "source_version": "v1",
        "source_version_digest": SHA_A,
        "artifact_id": "ART-STC-0001",
        "support_span_digest": SHA_B,
        "relation": "SUPPORTS",
        "role": "RELEASE_SUPPORT",
        "source_grade": "A",
        "warrant": "SUMM",
        "reproduction_strength": "r0",
    }
    return {
        "claim_id": "CL-STC-0001",
        "statement": "The reviewed source describes a bounded method.",
        "headline_quantitative": False,
        "claim_class": "SOURCE-SUMMARY",
        "result_provenance": {
            "asserted_modality": "NOT_APPLICABLE",
            "calibration_modalities": [],
        },
        "status": "SUPPORTED",
        "scope": "The reviewed source version only.",
        "assumptions": [],
        "source_ids": ["SRC-STC-0001"],
        "evidence_ids": ["EV-STC-00001"],
        "support_refs": [support_ref],
        "evidence_requirement": {
            "allowed_claim_classes": ["SOURCE-SUMMARY"],
            "allowed_source_grades": ["A"],
            "allowed_warrants": ["SUMM"],
            "minimum_reproduction_strength": "r0",
            "minimum_independent_sources": 1,
            "requires_result_block": False,
            "minimum_independent_result_paths": 0,
        },
        "counterevidence_ids": [],
        "result_ids": [],
        "artifact_dependencies": ["ART-STC-0001"],
        "caveats": ["Bounded to one reviewed version."],
        "falsifier": "A full read contradicts the cited span.",
        "paper_owner": "PAPER-A",
        "gate_status": "G1",
        "last_audit_date": "2026-07-25",
    }


def _paper_card(
    source: dict[str, object],
    *,
    evidence_ids: list[str] | None = None,
) -> dict[str, object]:
    versions = source["versions"]
    assert isinstance(versions, list)
    assert versions
    selected_version = versions[-1]
    assert isinstance(selected_version, dict)
    source_id = source["source_id"]
    reviewed_version_digest = selected_version["sha256"]
    assert isinstance(source_id, str)
    assert isinstance(reviewed_version_digest, str)
    return {
        "source_id": source_id,
        "reviewed_version_digest": reviewed_version_digest,
        "reviewer": "reviewer-1",
        "reviewed_at": "2026-07-25",
        "research_question": "How is sleep-time computation represented?",
        "wake_input": "session events",
        "trigger": "offline window",
        "operator": "abstraction",
        "destination": "external memory",
        "training_data": ["past-only session events"],
        "objective": ["future-query utility"],
        "optimizer": "none",
        "update_location": "external store",
        "cadence": "daily",
        "capacity_policy": "bounded compaction",
        "systems_assumptions": ["atomic publication"],
        "reported_results": [],
        "limitations": ["Bounded to the reviewed source version."],
        "failure_modes": ["staleness"],
        "evidence_ids": [] if evidence_ids is None else evidence_ids,
        "counterevidence_ids": [],
        "code_review": "not available",
        "data_review": "not available",
        "non_claims": ["No hardware result is established."],
    }


def _hypothesis(
    *,
    amendment_history: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "hypothesis_id": "H-STC-001",
        "statement": "Reuse and pressure induce a routing crossover.",
        "evidence_ids": ["EV-STC-00001"],
        "evidence_anchor": "EV-STC-00001:Abstract",
        "estimand": "destination crossover",
        "null": "no crossover",
        "rejection_rule": "pre-registered interaction is nonzero",
        "validity_failure": "router sees future information",
        "owner": "paper-a",
        "artifact_dependencies": ["ART-STC-0001"],
        "assumptions": ["matched budgets"],
        "derivation_artifact": "ART-STC-0001",
        "falsifier": "held-out crossover is absent",
        "confirmatory_cells": ["reuse-high/pressure-high"],
        "amendment_history": amendment_history,
    }


def _artifact_node(
    *,
    claim_id: str,
    gate_status: str,
    artifact_id: str = "ART-STC-0001",
    artifact_path: str = "artifacts/claim-support.json",
    payload: bytes = ARTIFACT_PAYLOAD,
) -> dict[str, object]:
    return {
        "artifact_id": artifact_id,
        "schema_version": "1.0.0",
        "artifact_type": "claim-support",
        "path": artifact_path,
        "input_digests": [SHA_A],
        "producer_command": "stc evidence build",
        "output_digest": hashlib.sha256(payload).hexdigest(),
        "consumers": [],
        "claim_ids": [claim_id],
        "question_ids": [],
        "gate_status": gate_status,
        "owner": "evidence-governance",
        "frozen_release_tag": None,
    }


def _normalized_text_digest(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _manuscript_link(
    *,
    manuscript_path: str = "manuscript/paper-c.md",
    location: str = MANUSCRIPT_LOCATION,
    block_text: str = MANUSCRIPT_SENTENCE,
    sentence_digest: str | None = None,
) -> dict[str, object]:
    return {
        "claim_id": "PAPER-C1",
        "manuscript_path": manuscript_path,
        "location": location,
        "sentence_digest": (
            _normalized_text_digest(block_text)
            if sentence_digest is None
            else sentence_digest
        ),
        "evidence_ids": ["EV-STC-00001"],
        "artifact_dependencies": ["ART-STC-0001"],
        "last_verified": "2026-07-25",
    }


def _release_manifest(
    *,
    artifact_digest: str,
    gate_digest: str,
) -> dict[str, object]:
    return {
        "release_id": "stc-paper-v1.0.0",
        "candidate_ref": {
            "path": "control/candidate.json",
            "sha256": SHA_A,
        },
        "environment_ref": {
            "path": "environment/environment.json",
            "sha256": SHA_B,
        },
        "sbom_ref": {
            "path": "sbom/sbom.spdx.json",
            "sha256": SHA_C,
        },
        "reproduction_manifest_ref": {
            "path": "reproduction/reproduction-manifest.json",
            "sha256": SHA_A,
        },
        "artifact_refs": [
            {
                "artifact_id": "ART-STC-0001",
                "sha256": artifact_digest,
            }
        ],
        "gate_refs": [
            {
                "artifact_id": "ART-STC-0002",
                "sha256": gate_digest,
            }
        ],
        "schema_inventory_digest": SHA_B,
        "released_at": "2026-07-25T12:00:00+09:00",
    }


def _presentation_handoff(*, gate_digest: str) -> dict[str, object]:
    return {
        "presentation_id": "stc-deck-v1.0.0",
        "handoff_index_ref": {
            "path": "publication/handoff-index.json",
            "sha256": SHA_A,
        },
        "deck_ref": {
            "path": "presentation/sleep-time-compute.pptx",
            "sha256": SHA_B,
        },
        "pdf_ref": {
            "path": "presentation/sleep-time-compute.pdf",
            "sha256": SHA_C,
        },
        "claim_link_digest": SHA_A,
        "visual_review_ref": {
            "path": "publication/reviews/visual-review.json",
            "sha256": SHA_B,
        },
        "human_approval_ref": {
            "path": "publication/reviews/human-approval.json",
            "sha256": SHA_C,
        },
        "g8_gate_ref": {
            "artifact_id": "ART-STC-0001",
            "sha256": gate_digest,
        },
        "handed_off_at": "2026-07-25T12:00:00+09:00",
    }


def _negative_result() -> dict[str, object]:
    return {
        "result_id": "RES-STC-0001",
        "experiment_id": "EXP-STC-0001",
        "hypothesis_ids": [],
        "result_modality": "SOURCE_REPORTED",
        "outcome": "The bounded hypothesis was narrowed.",
        "result_digest": SHA_A,
        "artifact_dependencies": ["ART-STC-0001"],
        "registered_at": "2026-07-25T12:00:00+09:00",
    }


def _schema_catalog() -> tuple[dict[str, dict[str, object]], Registry]:
    schemas = {path.name: _read_schema(path) for path in _schema_paths(PACKAGE_ROOT)}
    resources = []
    for filename, schema in sorted(schemas.items()):
        schema_id = schema["$id"]
        assert isinstance(schema_id, str), filename
        resources.append((schema_id, Resource.from_contents(schema)))
    return schemas, Registry().with_resources(resources)


def _schema_errors(filename: str, payload: object) -> tuple[str, ...]:
    schemas, registry = _schema_catalog()
    validator = Draft202012Validator(
        schemas[filename],
        registry=registry,
        format_checker=FormatChecker(),
    )
    return tuple(
        error.message
        for error in sorted(
            validator.iter_errors(payload),
            key=lambda error: (
                tuple(str(part) for part in error.absolute_path),
                error.message,
            ),
        )
    )


def _model_payloads() -> dict[str, tuple[type[object], dict[str, object]]]:
    return {
        "source.schema.json": (SourceRecord, _source()),
        "evidence.schema.json": (EvidenceRecord, _evidence(1, anchor="Methods")),
        "claim.schema.json": (ClaimRecord, _claim()),
        "paper-card.schema.json": (
            PaperCard,
            {
                "source_id": "SRC-STC-0001",
                "reviewed_version_digest": SHA_A,
                "reviewer": "reviewer-1",
                "reviewed_at": "2026-07-25",
                "research_question": "How is sleep represented?",
                "wake_input": "session events",
                "trigger": "offline window",
                "operator": "abstraction",
                "destination": "external memory",
                "training_data": ["past-only session events"],
                "objective": ["future-query utility"],
                "optimizer": "none",
                "update_location": "external store",
                "cadence": "daily",
                "capacity_policy": "bounded compaction",
                "systems_assumptions": ["atomic publication"],
                "reported_results": [],
                "limitations": ["abstract-only review"],
                "failure_modes": ["staleness"],
                "evidence_ids": ["EV-STC-00001"],
                "counterevidence_ids": [],
                "code_review": "not available",
                "data_review": "not available",
                "non_claims": ["No hardware result is established."],
            },
        ),
        "question.schema.json": (
            QuestionRecord,
            {
                "question_id": "RQ1",
                "statement": "Does the route differ by condition?",
                "estimand": "destination crossover",
                "null": "no crossover",
                "rejection_rule": "interaction is nonzero",
                "minimum_effect": "pre-registered threshold",
                "inconclusive_condition": "interval crosses threshold",
                "validity_failure": "future information leaks",
                "owner": "paper-a",
                "artifact_dependencies": ["ART-STC-0001"],
                "paper_scope": "PAPER-A",
            },
        ),
        "hypothesis.schema.json": (
            HypothesisRecord,
            {
                "hypothesis_id": "H-STC-001",
                "statement": "Reuse and pressure induce a routing crossover.",
                "evidence_ids": ["EV-STC-00001"],
                "evidence_anchor": "EV-STC-00001:p.4",
                "estimand": "destination crossover",
                "null": "no crossover",
                "rejection_rule": "pre-registered interaction is nonzero",
                "validity_failure": "router sees future information",
                "owner": "paper-a",
                "artifact_dependencies": ["ART-STC-0001"],
                "assumptions": ["matched budgets"],
                "derivation_artifact": "ART-STC-0001",
                "falsifier": "held-out crossover is absent",
                "confirmatory_cells": ["reuse-high/pressure-high"],
                "amendment_history": [],
            },
        ),
        "artifact.schema.json": (
            ArtifactNode,
            {
                "artifact_id": "ART-STC-0002",
                "schema_version": "1.0.0",
                "artifact_type": "claim-bundle",
                "path": "generated/claims.json",
                "input_digests": [SHA_A],
                "producer_command": "stc claims build",
                "output_digest": SHA_B,
                "consumers": ["ART-STC-0003"],
                "claim_ids": ["PAPER-C1"],
                "question_ids": ["RQ1"],
                "gate_status": "CURRENT",
                "owner": "evidence",
                "frozen_release_tag": None,
            },
        ),
        "audit-attestation.schema.json": (
            AuditAttestation,
            {
                "attestation_id": "ATT-G8-001",
                "subject_sha256": SHA_C,
                "subject_refs": [{"path": "manifests/g8.json", "sha256": SHA_A}],
                "signer_id": "release-evaluator-1",
                "signer_role": "release_evaluator",
                "independence_mode": "fresh-context",
                "author_executor_roster_digest": SHA_B,
                "algorithm": "Ed25519",
                "key_id": "release-key-1",
                "signature": "base64-signature",
                "signed_at": "2026-07-25T12:00:00+09:00",
            },
        ),
        "gate-evaluation-attestation.schema.json": (
            GateEvaluationAttestation,
            {
                "attestation_id": "ATT-G8-001",
                "gate_id": "G8",
                "subject_sha256": SHA_C,
                "subject_refs": [{"path": "manifests/g8.json", "sha256": SHA_A}],
                "signer_id": "release-evaluator-1",
                "signer_role": "release_evaluator",
                "independence_mode": "fresh-context",
                "algorithm": "Ed25519",
                "key_id": "release-key-1",
                "signature": "base64-signature",
                "signed_at": "2026-07-25T12:00:00+09:00",
            },
        ),
        "human-approval.schema.json": (
            HumanApproval,
            {
                "approval_id": "APPROVAL-001",
                "subject_sha256": SHA_A,
                "pdf_digests": [SHA_B],
                "machine_visual_report_digest": SHA_C,
                "contact_sheet_digest": SHA_A,
                "all_pages_reviewed": True,
                "review_checks": {"layout": "PASS", "citations": "PASS"},
                "disposition": "PASS",
                "open_issues": [],
                "signer_id": "visual-reviewer-1",
                "signer_role": "visual_reviewer",
                "algorithm": "Ed25519",
                "key_id": "visual-key-1",
                "signature": "base64-signature",
                "signed_at": "2026-07-25T12:00:00+09:00",
            },
        ),
        "gate.schema.json": (
            GateRecord,
            {
                "gate_id": "G6",
                "gate_artifact_id": "ART-STC-9907",
                "schema_version": "1.0.0",
                "evaluation_commit_sha": "d" * 40,
                "evaluation_tree_digest": SHA_A,
                "scientific_candidate_sha": None,
                "scientific_candidate_tree_digest": None,
                "artifact_dag_digest": SHA_B,
                "predecessor_gate_refs": [],
                "input_refs": [],
                "evaluator_id": "stc-gate-engine",
                "evaluator_role": "automated",
                "independence_mode": "deterministic",
                "evaluator_attestation_ref": None,
                "finding_ids": [],
                "adjudication_ids": [],
                "reaudit_refs": [],
                "environment_digest": SHA_C,
                "evaluated_at": "2026-07-25T12:00:00+09:00",
                "status": "PASS",
            },
        ),
    }
def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _valid_repo(
    tmp_path: Path,
    *,
    sources: list[dict[str, object]] | None = None,
    evidence: list[dict[str, object]] | None = None,
    claims: list[dict[str, object]] | None = None,
    hypotheses: list[dict[str, object]] | None = None,
    artifacts: list[dict[str, object]] | None = None,
    negative_results: list[dict[str, object]] | None = None,
    manuscript_links: list[dict[str, object]] | None = None,
    artifact_payloads: Mapping[str, bytes] | None = None,
    manuscript_blocks: Mapping[tuple[str, str], str] | None = None,
) -> Path:
    root = tmp_path / "repository"
    registry = root / "registry"
    source_records = [_source()] if sources is None else sources
    evidence_records = (
        [
            _evidence(1, anchor="Methods"),
            _evidence(2, anchor="Results"),
            _evidence(3, anchor="Limitations"),
        ]
        if evidence is None
        else evidence
    )
    claim_records = [_claim()] if claims is None else claims
    hypothesis_records = [] if hypotheses is None else hypotheses
    artifact_records = [] if artifacts is None else artifacts
    negative_result_records = [] if negative_results is None else negative_results
    for claim in claim_records:
        assert ClaimRecord.from_dict(deepcopy(claim)).to_dict() == claim
    for hypothesis in hypothesis_records:
        assert HypothesisRecord.from_dict(deepcopy(hypothesis)).to_dict() == hypothesis
    for artifact in artifact_records:
        assert ArtifactNode.from_dict(deepcopy(artifact)).to_dict() == artifact
        artifact_path = artifact["path"]
        assert isinstance(artifact_path, str)
        payload = (
            ARTIFACT_PAYLOAD
            if artifact_payloads is None
            else artifact_payloads.get(artifact_path, ARTIFACT_PAYLOAD)
        )
        assert isinstance(payload, bytes)
        materialized_artifact = root / artifact_path
        materialized_artifact.parent.mkdir(parents=True, exist_ok=True)
        materialized_artifact.write_bytes(payload)
    for negative_result in negative_result_records:
        assert _schema_errors("negative-result.schema.json", negative_result) == ()
    records_by_filename = {
        "artifacts.jsonl": artifact_records,
        "claims.jsonl": claim_records,
        "evidence.jsonl": evidence_records,
        "hypotheses.jsonl": hypothesis_records,
        "negative-results.jsonl": negative_result_records,
        "sources.jsonl": source_records,
    }
    for filename in CANONICAL_REGISTRY_JSONL_FILENAMES:
        _write_jsonl(
            registry / filename,
            records_by_filename.get(filename, []),
        )

    for source in source_records:
        card = _paper_card(source)
        assert PaperCard.from_dict(deepcopy(card)).to_dict() == card
        source_id = source["source_id"]
        assert isinstance(source_id, str)
        _write_json(root / "cards" / f"{source_id}.json", card)

    if manuscript_links is not None:
        blocks_by_path: dict[str, list[tuple[str, str]]] = {}
        for link in manuscript_links:
            assert _schema_errors("manuscript-link.schema.json", link) == ()
            manuscript_path = link["manuscript_path"]
            location = link["location"]
            assert isinstance(manuscript_path, str)
            assert isinstance(location, str)
            block_text = (
                MANUSCRIPT_SENTENCE
                if manuscript_blocks is None
                else manuscript_blocks.get(
                    (manuscript_path, location),
                    MANUSCRIPT_SENTENCE,
                )
            )
            blocks_by_path.setdefault(manuscript_path, []).append(
                (location, block_text)
            )
        for manuscript_path, blocks in blocks_by_path.items():
            materialized_manuscript = root / manuscript_path
            materialized_manuscript.parent.mkdir(parents=True, exist_ok=True)
            materialized_manuscript.write_text(
                "".join(
                    f"% STC:BEGIN {location}\n"
                    f"{block_text}\n"
                    f"% STC:END {location}\n"
                    for location, block_text in blocks
                ),
                encoding="utf-8",
            )
        _write_jsonl(
            root / "publication" / "links" / "manuscript-links.jsonl",
            manuscript_links,
        )
    return root


def _diagnostic_codes(report: object) -> set[str]:
    return {diagnostic.code for diagnostic in report.diagnostics}


def _semantic_repo(
    tmp_path: Path,
    *,
    sources: list[dict[str, object]],
    evidence: list[dict[str, object]],
    hypotheses: list[dict[str, object]] | None = None,
) -> Path:
    root = _valid_repo(
        tmp_path,
        sources=sources,
        evidence=evidence,
        claims=[],
        hypotheses=hypotheses,
    )
    shutil.copytree(PACKAGE_ROOT / "schemas", root / "schemas")
    return root


def _diagnostic_contract(
    report: object,
) -> tuple[tuple[str, str, str, str, str | None, str], ...]:
    return tuple(
        (
            diagnostic.code,
            diagnostic.registry,
            diagnostic.record_id,
            diagnostic.field,
            diagnostic.related_id,
            diagnostic.message,
        )
        for diagnostic in report.diagnostics
    )


def _claim_profile_source(
    number: int,
    *,
    canonical_key: str | None = None,
    source_grade: str = "A",
) -> dict[str, object]:
    source = deepcopy(_source())
    source_id = f"SRC-STC-{number:04d}"
    artifact_id = f"ART-STC-{number:04d}"
    version_digest = SHA_A if number == 1 else SHA_C
    source["source_id"] = source_id
    source["legacy_ids"] = [f"W1-{number:03d}"]
    source["canonical_key"] = (
        canonical_key
        if canonical_key is not None
        else f"doi:10.1000/sleep-work-{number}"
    )
    source["citation_key"] = f"author2025sleep{number}"
    source["review_status"] = "A-HTML"
    source["source_grade"] = source_grade

    versions = source["versions"]
    artifacts = source["local_artifacts"]
    assert isinstance(versions, list)
    assert isinstance(versions[0], dict)
    assert isinstance(artifacts, list)
    assert isinstance(artifacts[0], dict)
    versions[0]["local_artifact_id"] = artifact_id
    versions[0]["sha256"] = version_digest
    canonical_artifact = deepcopy(artifacts[0])
    canonical_artifact["artifact_id"] = artifact_id
    canonical_artifact["canonical_url"] = (
        f"https://example.org/source-{number}.pdf"
    )
    canonical_artifact["local_path"] = f"artifacts/source-{number}.pdf"
    canonical_artifact["sha256"] = version_digest
    unrelated_artifact = deepcopy(canonical_artifact)
    unrelated_artifact["artifact_id"] = f"ART-STC-{9000 + number:04d}"
    unrelated_artifact["canonical_url"] = (
        f"https://example.org/source-{number}-supplement.pdf"
    )
    unrelated_artifact["local_path"] = (
        f"artifacts/source-{number}-supplement.pdf"
    )
    unrelated_artifact["sha256"] = SHA_B
    source["local_artifacts"] = [unrelated_artifact, canonical_artifact]
    return source


def _claim_profile_evidence(
    number: int,
    source: dict[str, object],
    *,
    relation: str = "SUPPORTS",
    source_grade: str = "A",
) -> dict[str, object]:
    evidence = _evidence(number, anchor="Reviewed section")
    versions = source["versions"]
    assert isinstance(versions, list)
    assert isinstance(versions[0], dict)
    assert isinstance(source["source_id"], str)
    evidence["source_id"] = source["source_id"]
    evidence["source_version"] = versions[0]["identifier"]
    evidence["source_version_digest"] = versions[0]["sha256"]
    evidence["relation"] = relation
    evidence["source_grade"] = source_grade
    return evidence


def _claim_profile_support_ref(
    evidence: dict[str, object],
    source: dict[str, object],
) -> dict[str, object]:
    versions = source["versions"]
    assert isinstance(versions, list)
    assert isinstance(versions[0], dict)
    return {
        "evidence_id": evidence["evidence_id"],
        "source_id": evidence["source_id"],
        "source_version": evidence["source_version"],
        "source_version_digest": evidence["source_version_digest"],
        "artifact_id": versions[0]["local_artifact_id"],
        "support_span_digest": evidence["support_span_digest"],
        "relation": evidence["relation"],
        "role": "RELEASE_SUPPORT",
        "source_grade": evidence["source_grade"],
        "warrant": evidence["warrant"],
        "reproduction_strength": evidence["reproduction_strength"],
    }


def _claim_profile_claim(
    support_refs: list[dict[str, object]],
    *,
    minimum_independent_sources: int,
) -> dict[str, object]:
    claim = _claim()
    claim["support_refs"] = support_refs
    claim["source_ids"] = list(
        dict.fromkeys(ref["source_id"] for ref in support_refs)
    )
    claim["evidence_ids"] = list(
        dict.fromkeys(ref["evidence_id"] for ref in support_refs)
    )
    claim["counterevidence_ids"] = list(
        dict.fromkeys(
            ref["evidence_id"]
            for ref in support_refs
            if ref["role"] == "COUNTEREVIDENCE"
        )
    )
    claim["artifact_dependencies"] = list(
        dict.fromkeys(ref["artifact_id"] for ref in support_refs)
    )
    requirement = claim["evidence_requirement"]
    assert isinstance(requirement, dict)
    requirement["minimum_independent_sources"] = minimum_independent_sources
    return claim


def _paper_release_records() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    source = _claim_profile_source(1)
    evidence = _claim_profile_evidence(1, source)
    evidence["claim_ids"] = ["PAPER-C1"]
    claim = _claim_profile_claim(
        [_claim_profile_support_ref(evidence, source)],
        minimum_independent_sources=1,
    )
    claim["claim_id"] = "PAPER-C1"
    claim["paper_owner"] = "PAPER-C"
    claim["gate_status"] = "G6"
    assert (
        ClaimRecord.from_dict(deepcopy(claim), for_release=True).to_dict()
        == claim
    )
    return source, evidence, claim


def _claim_profile_repo(
    tmp_path: Path,
    *,
    sources: list[dict[str, object]],
    evidence: list[dict[str, object]],
    claims: list[dict[str, object]],
    artifacts: list[dict[str, object]] | None = None,
    negative_results: list[dict[str, object]] | None = None,
    manuscript_links: list[dict[str, object]] | None = None,
    artifact_payloads: Mapping[str, bytes] | None = None,
    manuscript_blocks: Mapping[tuple[str, str], str] | None = None,
) -> Path:
    for source in sources:
        assert SourceRecord.from_dict(deepcopy(source)).to_dict() == source
    for item in evidence:
        assert EvidenceRecord.from_dict(deepcopy(item)).to_dict() == item
    for claim in claims:
        assert ClaimRecord.from_dict(deepcopy(claim)).to_dict() == claim

    root = _valid_repo(
        tmp_path,
        sources=sources,
        evidence=evidence,
        claims=claims,
        artifacts=artifacts,
        negative_results=negative_results,
        manuscript_links=manuscript_links,
        artifact_payloads=artifact_payloads,
        manuscript_blocks=manuscript_blocks,
    )
    shutil.copytree(PACKAGE_ROOT / "schemas", root / "schemas")
    return root


def test_reports_evidence_that_references_a_missing_source(tmp_path: Path) -> None:
    evidence = [_evidence(1, anchor="Methods")]
    evidence[0]["source_id"] = "SRC-STC-9999"

    report = validate_repository(
        _valid_repo(tmp_path, evidence=evidence, claims=[])
    )

    assert "EVIDENCE_SOURCE_MISSING" in _diagnostic_codes(report)


def test_reports_claim_that_references_missing_evidence(tmp_path: Path) -> None:
    claim = _claim()
    missing_evidence_id = "EV-STC-99999"
    claim["evidence_ids"] = [missing_evidence_id]
    claim["support_refs"][0]["evidence_id"] = missing_evidence_id

    report = validate_repository(_valid_repo(tmp_path, claims=[claim]))

    assert "CLAIM_EVIDENCE_MISSING" in _diagnostic_codes(report)


def test_reports_a_abs_evidence_anchor_outside_the_abstract(tmp_path: Path) -> None:
    source = _source()
    source["review_status"] = "A-ABS"

    report = validate_repository(
        _valid_repo(
            tmp_path,
            sources=[source],
            evidence=[_evidence(1, anchor="Methods section 2")],
        )
    )

    assert "A_ABS_ANCHOR_OUTSIDE_ABSTRACT" in _diagnostic_codes(report)


def test_reports_unresolved_paper_c_claim_at_release(tmp_path: Path) -> None:
    claim = _claim()
    claim["claim_id"] = "PAPER-C1"
    claim["status"] = "UNRESOLVED"

    report = validate_repository(_valid_repo(tmp_path, claims=[claim]), gate="G6")

    assert "PAPER_C_UNRESOLVED" in _diagnostic_codes(report)


def test_reports_full_source_without_method_result_or_limitation_anchors(
    tmp_path: Path,
) -> None:
    evidence = [
        _evidence(number, anchor="Abstract")
        for number in (1, 2, 3)
    ]

    report = validate_repository(_valid_repo(tmp_path, evidence=evidence))

    assert "FULL_REVIEW_ANCHORS_MISSING" in _diagnostic_codes(report)


def test_reports_duplicate_source_canonical_key_with_deterministic_owner(
    tmp_path: Path,
) -> None:
    first_source = _source()
    first_source["review_status"] = "A-ABS"
    duplicate_source = deepcopy(first_source)
    duplicate_source["source_id"] = "SRC-STC-0002"
    duplicate_source["legacy_ids"] = ["W1-002"]
    duplicate_source["citation_key"] = "author2025sleepduplicate"
    duplicate_version = duplicate_source["versions"][0]
    duplicate_artifact = duplicate_source["local_artifacts"][0]
    assert isinstance(duplicate_version, dict)
    assert isinstance(duplicate_artifact, dict)
    duplicate_version["local_artifact_id"] = "ART-STC-0002"
    duplicate_artifact["artifact_id"] = "ART-STC-0002"

    report = validate_repository(
        _semantic_repo(
            tmp_path,
            sources=[first_source, duplicate_source],
            evidence=[],
        )
    )

    assert _diagnostic_contract(report) == (
        (
            "SOURCE_CANONICAL_KEY_DUPLICATE",
            "registry/sources.jsonl",
            "SRC-STC-0002",
            "canonical_key",
            "SRC-STC-0001",
            (
                "source SRC-STC-0002 duplicates canonical_key "
                "'doi:10.1000/sleep-work:v1' already used by SRC-STC-0001"
            ),
        ),
    )


def test_reports_source_version_that_predates_earliest_public_date(
    tmp_path: Path,
) -> None:
    source = _source()
    source["review_status"] = "A-ABS"
    source["earliest_public_date"] = "2025-04-18"
    version = source["versions"][0]
    assert isinstance(version, dict)
    version["public_date"] = "2025-04-17"
    version["date_precision"] = "DAY"

    report = validate_repository(
        _semantic_repo(tmp_path, sources=[source], evidence=[])
    )

    assert _diagnostic_contract(report) == (
        (
            "SOURCE_VERSION_PREDATES_EARLIEST_PUBLIC_DATE",
            "registry/sources.jsonl",
            "SRC-STC-0001",
            "earliest_public_date",
            "v1",
            (
                "source SRC-STC-0001 earliest_public_date 2025-04-18 is later "
                "than version v1 public_date 2025-04-17"
            ),
        ),
    )


def test_source_version_month_interval_overlapping_day_is_not_predating(
    tmp_path: Path,
) -> None:
    source = _source()
    source["review_status"] = "A-ABS"
    source["earliest_public_date"] = "2025-04-17"
    source["date_precision"] = "DAY"
    version = source["versions"][0]
    assert isinstance(version, dict)
    version["public_date"] = "2025-04"
    version["date_precision"] = "MONTH"
    assert SourceRecord.from_dict(deepcopy(source)).to_dict() == source

    report = validate_repository(
        _semantic_repo(tmp_path, sources=[source], evidence=[])
    )

    assert _diagnostic_contract(report) == ()


def test_source_version_month_interval_definitely_before_day_is_predating(
    tmp_path: Path,
) -> None:
    source = _source()
    source["review_status"] = "A-ABS"
    source["earliest_public_date"] = "2025-05-01"
    source["date_precision"] = "DAY"
    version = source["versions"][0]
    assert isinstance(version, dict)
    version["public_date"] = "2025-04"
    version["date_precision"] = "MONTH"
    assert SourceRecord.from_dict(deepcopy(source)).to_dict() == source

    report = validate_repository(
        _semantic_repo(tmp_path, sources=[source], evidence=[])
    )

    assert _diagnostic_contract(report) == (
        (
            "SOURCE_VERSION_PREDATES_EARLIEST_PUBLIC_DATE",
            "registry/sources.jsonl",
            "SRC-STC-0001",
            "earliest_public_date",
            "v1",
            (
                "source SRC-STC-0001 earliest_public_date 2025-05-01 is later "
                "than version v1 public_date 2025-04"
            ),
        ),
    )


def test_reports_evidence_source_version_absent_from_resolved_source(
    tmp_path: Path,
) -> None:
    source = _source()
    source["review_status"] = "A-ABS"
    evidence = _evidence(1, anchor="Abstract")
    evidence["anchor_kind"] = "ABSTRACT"
    evidence["source_version"] = "v2"

    report = validate_repository(
        _semantic_repo(tmp_path, sources=[source], evidence=[evidence])
    )

    assert _diagnostic_contract(report) == (
        (
            "EVIDENCE_SOURCE_VERSION_MISMATCH",
            "registry/evidence.jsonl",
            "EV-STC-00001",
            "source_version",
            "SRC-STC-0001",
            (
                "evidence EV-STC-00001 references source version v2 absent "
                "from source SRC-STC-0001"
            ),
        ),
    )


def test_reports_evidence_source_version_digest_mismatch(
    tmp_path: Path,
) -> None:
    source = _source()
    source["review_status"] = "A-ABS"
    evidence = _evidence(1, anchor="Abstract")
    evidence["anchor_kind"] = "ABSTRACT"
    evidence["source_version_digest"] = SHA_B

    report = validate_repository(
        _semantic_repo(tmp_path, sources=[source], evidence=[evidence])
    )

    assert _diagnostic_contract(report) == (
        (
            "EVIDENCE_SOURCE_VERSION_DIGEST_MISMATCH",
            "registry/evidence.jsonl",
            "EV-STC-00001",
            "source_version_digest",
            "SRC-STC-0001",
            (
                f"evidence EV-STC-00001 source version v1 digest {SHA_B} does "
                f"not match source SRC-STC-0001 digest {SHA_A}"
            ),
        ),
    )


@pytest.mark.parametrize(
    ("field_name", "declared_value", "expected_message"),
    (
        pytest.param(
            "source_id",
            "SRC-STC-0002",
            (
                "claim CL-STC-0001 support_refs[0].source_id 'SRC-STC-0002' "
                "does not match evidence EV-STC-00001 source_id "
                "'SRC-STC-0001'"
            ),
            id="source-id",
        ),
        pytest.param(
            "source_version",
            "v2",
            (
                "claim CL-STC-0001 support_refs[0].source_version 'v2' does "
                "not match evidence EV-STC-00001 source_version 'v1'"
            ),
            id="source-version",
        ),
        pytest.param(
            "source_version_digest",
            SHA_C,
            (
                f"claim CL-STC-0001 support_refs[0].source_version_digest "
                f"'{SHA_C}' does not match evidence EV-STC-00001 "
                f"source_version_digest '{SHA_A}'"
            ),
            id="source-version-digest",
        ),
        pytest.param(
            "artifact_id",
            "ART-STC-9001",
            (
                "claim CL-STC-0001 support_refs[0].artifact_id "
                "'ART-STC-9001' does not match evidence EV-STC-00001 "
                "resolved source version local_artifact_id 'ART-STC-0001'"
            ),
            id="artifact-id",
        ),
        pytest.param(
            "support_span_digest",
            SHA_C,
            (
                f"claim CL-STC-0001 support_refs[0].support_span_digest "
                f"'{SHA_C}' does not match evidence EV-STC-00001 "
                f"support_span_digest '{SHA_B}'"
            ),
            id="support-span-digest",
        ),
        pytest.param(
            "relation",
            "NON_CLAIM",
            (
                "claim CL-STC-0001 support_refs[0].relation 'NON_CLAIM' does "
                "not match evidence EV-STC-00001 relation 'SUPPORTS'"
            ),
            id="relation",
        ),
        pytest.param(
            "source_grade",
            "B",
            (
                "claim CL-STC-0001 support_refs[0].source_grade 'B' does not "
                "match evidence EV-STC-00001 source_grade 'A'"
            ),
            id="source-grade",
        ),
        pytest.param(
            "warrant",
            "SYNTH",
            (
                "claim CL-STC-0001 support_refs[0].warrant 'SYNTH' does not "
                "match evidence EV-STC-00001 warrant 'SUMM'"
            ),
            id="warrant",
        ),
        pytest.param(
            "reproduction_strength",
            "r1",
            (
                "claim CL-STC-0001 support_refs[0].reproduction_strength "
                "'r1' does not match evidence EV-STC-00001 "
                "reproduction_strength 'r0'"
            ),
            id="reproduction-strength",
        ),
    ),
)
def test_claim_support_ref_fields_match_resolved_evidence_profile(
    tmp_path: Path,
    field_name: str,
    declared_value: object,
    expected_message: str,
) -> None:
    source = _claim_profile_source(1)
    evidence = _claim_profile_evidence(1, source)
    support_ref = _claim_profile_support_ref(evidence, source)
    support_ref[field_name] = declared_value
    minimum_independent_sources = 1
    if field_name == "relation":
        support_ref["role"] = "CONTEXT"
        minimum_independent_sources = 0

    claim = _claim_profile_claim(
        [support_ref],
        minimum_independent_sources=minimum_independent_sources,
    )
    requirement = claim["evidence_requirement"]
    assert isinstance(requirement, dict)
    if field_name == "source_grade":
        requirement["allowed_source_grades"] = ["A", "B"]
    if field_name == "warrant":
        requirement["allowed_warrants"] = ["SUMM", "SYNTH"]

    sources = [source]
    if field_name == "source_id":
        sources.append(_claim_profile_source(2))

    report = validate_repository(
        _claim_profile_repo(
            tmp_path,
            sources=sources,
            evidence=[evidence],
            claims=[claim],
        )
    )

    assert _diagnostic_contract(report) == (
        (
            "CLAIM_SUPPORT_REF_MISMATCH",
            "registry/claims.jsonl",
            "CL-STC-0001",
            f"support_refs[0].{field_name}",
            "EV-STC-00001",
            expected_message,
        ),
    )


@pytest.mark.parametrize(
    ("canonical_relation", "support_role"),
    (
        pytest.param("NON_CLAIM", "CONTEXT", id="context"),
        pytest.param("CONTRADICTS", "COUNTEREVIDENCE", id="counterevidence"),
    ),
)
def test_supported_release_requires_positive_release_support(
    tmp_path: Path,
    canonical_relation: str,
    support_role: str,
) -> None:
    source = _claim_profile_source(1)
    evidence = _claim_profile_evidence(
        1,
        source,
        relation=canonical_relation,
    )
    support_ref = _claim_profile_support_ref(evidence, source)
    support_ref["role"] = support_role
    claim = _claim_profile_claim(
        [support_ref],
        minimum_independent_sources=0,
    )
    root = _claim_profile_repo(
        tmp_path,
        sources=[source],
        evidence=[evidence],
        claims=[claim],
    )

    assert _diagnostic_contract(validate_repository(root)) == ()

    report = validate_repository(root, gate="G6")

    assert _diagnostic_contract(report) == (
        (
            "CLAIM_EVIDENCE_REQUIREMENT_UNMET",
            "registry/claims.jsonl",
            "CL-STC-0001",
            "support_refs",
            "G6",
            (
                "supported claim CL-STC-0001 has no exact qualifying "
                "RELEASE_SUPPORT at release boundary G6"
            ),
        ),
    )


def test_release_requirement_rejects_canonical_mixed_grade_support(
    tmp_path: Path,
) -> None:
    first_source = _claim_profile_source(1)
    second_source = _claim_profile_source(2, source_grade="X")
    first_evidence = _claim_profile_evidence(1, first_source)
    second_evidence = _claim_profile_evidence(
        2,
        second_source,
        source_grade="X",
    )
    first_ref = _claim_profile_support_ref(first_evidence, first_source)
    second_ref = _claim_profile_support_ref(second_evidence, second_source)
    second_ref["source_grade"] = "A"
    claim = _claim_profile_claim(
        [first_ref, second_ref],
        minimum_independent_sources=2,
    )

    report = validate_repository(
        _claim_profile_repo(
            tmp_path,
            sources=[first_source, second_source],
            evidence=[first_evidence, second_evidence],
            claims=[claim],
        ),
        gate="G6",
    )

    assert _diagnostic_contract(report) == (
        (
            "CLAIM_SUPPORT_REF_MISMATCH",
            "registry/claims.jsonl",
            "CL-STC-0001",
            "support_refs[1].source_grade",
            "EV-STC-00002",
            (
                "claim CL-STC-0001 support_refs[1].source_grade 'A' does not "
                "match evidence EV-STC-00002 source_grade 'X'"
            ),
        ),
        (
            "CLAIM_EVIDENCE_REQUIREMENT_UNMET",
            "registry/claims.jsonl",
            "CL-STC-0001",
            "evidence_requirement.minimum_independent_sources",
            None,
            (
                "claim CL-STC-0001 has 1 qualifying independent canonical "
                "work(s), below minimum_independent_sources=2"
            ),
        ),
    )


def test_evidence_source_grade_drift_invalidates_release_support(
    tmp_path: Path,
) -> None:
    source = _claim_profile_source(1, source_grade="X")
    evidence = _claim_profile_evidence(
        1,
        source,
        source_grade="A",
    )
    support_ref = _claim_profile_support_ref(evidence, source)
    claim = _claim_profile_claim(
        [support_ref],
        minimum_independent_sources=1,
    )

    report = validate_repository(
        _claim_profile_repo(
            tmp_path,
            sources=[source],
            evidence=[evidence],
            claims=[claim],
        ),
        gate="G6",
    )

    assert _diagnostic_contract(report) == (
        (
            "EVIDENCE_SOURCE_GRADE_MISMATCH",
            "registry/evidence.jsonl",
            "EV-STC-00001",
            "source_grade",
            "SRC-STC-0001",
            (
                "evidence EV-STC-00001 source_grade 'A' does not match source "
                "SRC-STC-0001 source_grade 'X'"
            ),
        ),
        (
            "CLAIM_EVIDENCE_REQUIREMENT_UNMET",
            "registry/claims.jsonl",
            "CL-STC-0001",
            "evidence_requirement.minimum_independent_sources",
            None,
            (
                "claim CL-STC-0001 has 0 qualifying independent canonical "
                "work(s), below minimum_independent_sources=1"
            ),
        ),
    )


@pytest.mark.parametrize(
    "canonical_relation",
    (
        pytest.param("NON_CLAIM", id="context-non-claim"),
        pytest.param("CONTRADICTS", id="counterevidence"),
    ),
)
def test_release_requirement_does_not_count_canonical_non_support_evidence(
    tmp_path: Path,
    canonical_relation: str,
) -> None:
    source = _claim_profile_source(1)
    evidence = _claim_profile_evidence(
        1,
        source,
        relation=canonical_relation,
    )
    support_ref = _claim_profile_support_ref(evidence, source)
    support_ref["relation"] = "SUPPORTS"
    claim = _claim_profile_claim(
        [support_ref],
        minimum_independent_sources=1,
    )

    report = validate_repository(
        _claim_profile_repo(
            tmp_path,
            sources=[source],
            evidence=[evidence],
            claims=[claim],
        ),
        gate="G6",
    )

    assert _diagnostic_contract(report) == (
        (
            "CLAIM_SUPPORT_REF_MISMATCH",
            "registry/claims.jsonl",
            "CL-STC-0001",
            "support_refs[0].relation",
            "EV-STC-00001",
            (
                "claim CL-STC-0001 support_refs[0].relation 'SUPPORTS' does "
                f"not match evidence EV-STC-00001 relation "
                f"'{canonical_relation}'"
            ),
        ),
        (
            "CLAIM_EVIDENCE_REQUIREMENT_UNMET",
            "registry/claims.jsonl",
            "CL-STC-0001",
            "evidence_requirement.minimum_independent_sources",
            None,
            (
                "claim CL-STC-0001 has 0 qualifying independent canonical "
                "work(s), below minimum_independent_sources=1"
            ),
        ),
    )


def test_release_requirement_deduplicates_canonical_work_identity(
    tmp_path: Path,
) -> None:
    canonical_key = "doi:10.1000/shared-sleep-work"
    first_source = _claim_profile_source(1, canonical_key=canonical_key)
    second_source = _claim_profile_source(2, canonical_key=canonical_key)
    first_evidence = _claim_profile_evidence(1, first_source)
    second_evidence = _claim_profile_evidence(2, second_source)
    claim = _claim_profile_claim(
        [
            _claim_profile_support_ref(first_evidence, first_source),
            _claim_profile_support_ref(second_evidence, second_source),
        ],
        minimum_independent_sources=2,
    )

    report = validate_repository(
        _claim_profile_repo(
            tmp_path,
            sources=[first_source, second_source],
            evidence=[first_evidence, second_evidence],
            claims=[claim],
        ),
        gate="G6",
    )

    assert _diagnostic_contract(report) == (
        (
            "SOURCE_CANONICAL_KEY_DUPLICATE",
            "registry/sources.jsonl",
            "SRC-STC-0002",
            "canonical_key",
            "SRC-STC-0001",
            (
                "source SRC-STC-0002 duplicates canonical_key "
                "'doi:10.1000/shared-sleep-work' already used by "
                "SRC-STC-0001"
            ),
        ),
        (
            "CLAIM_EVIDENCE_REQUIREMENT_UNMET",
            "registry/claims.jsonl",
            "CL-STC-0001",
            "evidence_requirement.minimum_independent_sources",
            None,
            (
                "claim CL-STC-0001 has 1 qualifying independent canonical "
                "work(s), below minimum_independent_sources=2"
            ),
        ),
    )


def test_reports_registered_source_missing_canonical_card(
    tmp_path: Path,
) -> None:
    source = _source()
    source["review_status"] = "A-ABS"
    assert SourceRecord.from_dict(deepcopy(source)).to_dict() == source
    root = _semantic_repo(tmp_path, sources=[source], evidence=[])
    card_path = root / "cards" / "SRC-STC-0001.json"
    assert card_path.is_file()
    card_path.unlink()

    report = validate_repository(root)

    assert _diagnostic_contract(report) == (
        (
            "SOURCE_CANONICAL_CARD_MISSING",
            "registry/sources.jsonl",
            "SRC-STC-0001",
            "canonical_card",
            "cards/SRC-STC-0001.json",
            (
                "source SRC-STC-0001 is missing canonical card "
                "cards/SRC-STC-0001.json"
            ),
        ),
    )


def test_reports_a_abs_card_with_non_abstract_evidence(
    tmp_path: Path,
) -> None:
    card_source = _claim_profile_source(1)
    card_source["review_status"] = "A-ABS"
    evidence = _claim_profile_evidence(1, card_source)
    evidence["anchor_kind"] = "SECTION"
    evidence["anchor"] = "Methods section 2"
    root = _semantic_repo(
        tmp_path,
        sources=[card_source],
        evidence=[evidence],
    )
    card = _paper_card(
        card_source,
        evidence_ids=["EV-STC-00001"],
    )
    assert PaperCard.from_dict(deepcopy(card)).to_dict() == card
    _write_json(root / "cards" / "SRC-STC-0001.json", card)

    report = validate_repository(root)

    assert _diagnostic_contract(report) == (
        (
            "A_ABS_ANCHOR_OUTSIDE_ABSTRACT",
            "registry/evidence.jsonl",
            "EV-STC-00001",
            "anchor",
            "SRC-STC-0001",
            (
                "A-ABS source SRC-STC-0001 cannot support evidence anchored "
                "at 'Methods section 2'"
            ),
        ),
        (
            "A_ABS_PAPER_CARD_NON_ABSTRACT_EVIDENCE",
            "cards/SRC-STC-0001.json",
            "SRC-STC-0001",
            "evidence_ids[0]",
            "EV-STC-00001",
            (
                "A-ABS card SRC-STC-0001 cites evidence EV-STC-00001 "
                "with non-abstract anchor 'Methods section 2'"
            ),
        ),
    )


def test_reports_post_g2_confirmatory_hypothesis_amendment(
    tmp_path: Path,
) -> None:
    source = _source()
    source["review_status"] = "A-ABS"
    evidence = _evidence(1, anchor="Abstract")
    evidence["anchor_kind"] = "ABSTRACT"
    hypothesis = _hypothesis(
        amendment_history=[
            {
                "amended_at": "2026-07-25T12:00:00+09:00",
                "gate_id": "G3",
                "reason": "Post-registration wording change.",
                "confirmatory": True,
            }
        ]
    )
    assert HypothesisRecord.from_dict(deepcopy(hypothesis)).to_dict() == hypothesis
    root = _semantic_repo(
        tmp_path,
        sources=[source],
        evidence=[evidence],
        hypotheses=[hypothesis],
    )

    report = validate_repository(root)

    assert _diagnostic_contract(report) == (
        (
            "POST_G2_CONFIRMATORY_AMENDMENT",
            "registry/hypotheses.jsonl",
            "H-STC-001",
            "amendment_history[0].confirmatory",
            "G3",
            (
                "hypothesis H-STC-001 amendment_history[0] at G3 cannot "
                "remain confirmatory after G2"
            ),
        ),
    )


def test_reports_release_claim_missing_manuscript_location(
    tmp_path: Path,
) -> None:
    source, evidence, claim = _paper_release_records()
    artifact = _artifact_node(
        claim_id="PAPER-C1",
        gate_status="CURRENT",
    )

    report = validate_repository(
        _claim_profile_repo(
            tmp_path,
            sources=[source],
            evidence=[evidence],
            claims=[claim],
            artifacts=[artifact],
            manuscript_links=[],
        ),
        gate="G6",
    )

    assert _diagnostic_contract(report) == (
        (
            "CLAIM_MANUSCRIPT_LOCATION_MISSING",
            "publication/links/manuscript-links.jsonl",
            "PAPER-C1",
            "location",
            "G6",
            (
                "release claim PAPER-C1 has no manuscript location in "
                "publication/links/manuscript-links.jsonl at G6"
            ),
        ),
    )


def test_reports_stale_artifact_dependency_for_release_claim(
    tmp_path: Path,
) -> None:
    source, evidence, claim = _paper_release_records()
    artifact = _artifact_node(
        claim_id="PAPER-C1",
        gate_status="STALE",
    )

    report = validate_repository(
        _claim_profile_repo(
            tmp_path,
            sources=[source],
            evidence=[evidence],
            claims=[claim],
            artifacts=[artifact],
            manuscript_links=[_manuscript_link()],
        ),
        gate="G6",
    )

    assert _diagnostic_contract(report) == (
        (
            "STALE_RELEASE_ARTIFACT",
            "registry/claims.jsonl",
            "PAPER-C1",
            "artifact_dependencies[0]",
            "ART-STC-0001",
            (
                "release claim PAPER-C1 depends on stale artifact "
                "ART-STC-0001 at G6"
            ),
        ),
    )


def test_reports_falsified_claim_missing_negative_result_registration(
    tmp_path: Path,
) -> None:
    source = _claim_profile_source(1)
    evidence = _claim_profile_evidence(1, source)
    evidence["claim_ids"] = ["CL-STC-0001"]
    claim = _claim_profile_claim(
        [_claim_profile_support_ref(evidence, source)],
        minimum_independent_sources=1,
    )
    claim["status"] = "FALSIFIED/NARROWED"
    claim["result_ids"] = ["RES-STC-0001"]
    claim["result_provenance"] = {
        "asserted_modality": "SOURCE_REPORTED",
        "calibration_modalities": [],
    }
    artifact = _artifact_node(
        claim_id="CL-STC-0001",
        gate_status="CURRENT",
    )

    report = validate_repository(
        _claim_profile_repo(
            tmp_path,
            sources=[source],
            evidence=[evidence],
            claims=[claim],
            artifacts=[artifact],
            negative_results=[],
        )
    )

    assert _diagnostic_contract(report) == (
        (
            "NEGATIVE_RESULT_REGISTRY_OMISSION",
            "registry/claims.jsonl",
            "CL-STC-0001",
            "result_ids[0]",
            "RES-STC-0001",
            (
                "FALSIFIED/NARROWED claim CL-STC-0001 result RES-STC-0001 "
                "is absent from registry/negative-results.jsonl"
            ),
        ),
    )


def test_reports_manuscript_block_digest_mismatch(
    tmp_path: Path,
) -> None:
    source, evidence, claim = _paper_release_records()
    artifact = _artifact_node(
        claim_id="PAPER-C1",
        gate_status="CURRENT",
    )
    mismatched_digest = SHA_C
    actual_digest = _normalized_text_digest(MANUSCRIPT_NONCANONICAL_BLOCK)
    canonical_digest = _normalized_text_digest(MANUSCRIPT_SENTENCE)
    raw_digest = hashlib.sha256(
        MANUSCRIPT_NONCANONICAL_BLOCK.encode("utf-8")
    ).hexdigest()
    assert actual_digest == canonical_digest
    assert actual_digest != raw_digest
    assert mismatched_digest != actual_digest

    report = validate_repository(
        _claim_profile_repo(
            tmp_path,
            sources=[source],
            evidence=[evidence],
            claims=[claim],
            artifacts=[artifact],
            manuscript_links=[
                _manuscript_link(
                    block_text=MANUSCRIPT_NONCANONICAL_BLOCK,
                    sentence_digest=mismatched_digest,
                )
            ],
            manuscript_blocks={
                (
                    "manuscript/paper-c.md",
                    MANUSCRIPT_LOCATION,
                ): MANUSCRIPT_NONCANONICAL_BLOCK,
            },
        ),
        gate="G6",
    )

    assert _diagnostic_contract(report) == (
        (
            "MANUSCRIPT_SENTENCE_DIGEST_MISMATCH",
            "publication/links/manuscript-links.jsonl",
            "PAPER-C1",
            "sentence_digest",
            MANUSCRIPT_LOCATION,
            (
                "manuscript link PAPER-C1 location EN-S0001 sentence_digest "
                f"{mismatched_digest} does not match normalized block digest "
                f"{actual_digest}"
            ),
        ),
    )


def test_reports_release_manifest_dependency_on_stale_artifact(
    tmp_path: Path,
) -> None:
    source = _claim_profile_source(1)
    evidence = _claim_profile_evidence(1, source)
    claim = _claim_profile_claim(
        [_claim_profile_support_ref(evidence, source)],
        minimum_independent_sources=1,
    )
    release_artifact = _artifact_node(
        claim_id="CL-STC-0001",
        gate_status="STALE",
        artifact_path="artifacts/release-payload.json",
        payload=RELEASE_ARTIFACT_PAYLOAD,
    )
    gate_artifact = _artifact_node(
        claim_id="CL-STC-0001",
        gate_status="CURRENT",
        artifact_id="ART-STC-0002",
        artifact_path="artifacts/g8.json",
        payload=GATE_ARTIFACT_PAYLOAD,
    )
    release_artifact_digest = release_artifact["output_digest"]
    gate_artifact_digest = gate_artifact["output_digest"]
    assert isinstance(release_artifact_digest, str)
    assert isinstance(gate_artifact_digest, str)
    manifest = _release_manifest(
        artifact_digest=release_artifact_digest,
        gate_digest=gate_artifact_digest,
    )
    assert _schema_errors("release-manifest.schema.json", manifest) == ()
    root = _claim_profile_repo(
        tmp_path,
        sources=[source],
        evidence=[evidence],
        claims=[claim],
        artifacts=[release_artifact, gate_artifact],
        artifact_payloads={
            "artifacts/release-payload.json": RELEASE_ARTIFACT_PAYLOAD,
            "artifacts/g8.json": GATE_ARTIFACT_PAYLOAD,
        },
    )
    relative_path = (
        "publication/releases/stc-paper-v1.0.0/release-manifest.json"
    )
    _write_json(root / relative_path, manifest)

    report = validate_repository(root)

    assert _diagnostic_contract(report) == (
        (
            "STALE_HANDOFF_ARTIFACT",
            relative_path,
            "stc-paper-v1.0.0",
            "artifact_refs[0].artifact_id",
            "ART-STC-0001",
            (
                "release manifest stc-paper-v1.0.0 depends on stale artifact "
                "ART-STC-0001"
            ),
        ),
    )


def test_reports_presentation_handoff_dependency_on_stale_artifact(
    tmp_path: Path,
) -> None:
    source = _claim_profile_source(1)
    evidence = _claim_profile_evidence(1, source)
    claim = _claim_profile_claim(
        [_claim_profile_support_ref(evidence, source)],
        minimum_independent_sources=1,
    )
    gate_artifact = _artifact_node(
        claim_id="CL-STC-0001",
        gate_status="STALE",
        artifact_path="artifacts/g8.json",
        payload=GATE_ARTIFACT_PAYLOAD,
    )
    gate_artifact_digest = gate_artifact["output_digest"]
    assert isinstance(gate_artifact_digest, str)
    handoff = _presentation_handoff(gate_digest=gate_artifact_digest)
    assert _schema_errors("presentation-handoff.schema.json", handoff) == ()
    root = _claim_profile_repo(
        tmp_path,
        sources=[source],
        evidence=[evidence],
        claims=[claim],
        artifacts=[gate_artifact],
        artifact_payloads={
            "artifacts/g8.json": GATE_ARTIFACT_PAYLOAD,
        },
    )
    relative_path = (
        "publication/handoff/stc-paper-v1.0.0/presentation-handoff.json"
    )
    _write_json(root / relative_path, handoff)

    report = validate_repository(root)

    assert _diagnostic_contract(report) == (
        (
            "STALE_HANDOFF_ARTIFACT",
            relative_path,
            "stc-deck-v1.0.0",
            "g8_gate_ref.artifact_id",
            "ART-STC-0001",
            (
                "presentation handoff stc-deck-v1.0.0 depends on stale "
                "artifact ART-STC-0001"
            ),
        ),
    )


def test_invalid_manuscript_link_is_diagnosed_and_excluded_from_release_semantics(
    tmp_path: Path,
) -> None:
    source, evidence, claim = _paper_release_records()
    artifact = _artifact_node(
        claim_id="PAPER-C1",
        gate_status="CURRENT",
    )
    root = _claim_profile_repo(
        tmp_path,
        sources=[source],
        evidence=[evidence],
        claims=[claim],
        artifacts=[artifact],
        manuscript_links=[],
    )
    invalid_link = _manuscript_link(
        manuscript_path="../outside.md",
        sentence_digest=SHA_C,
    )
    assert _schema_errors("manuscript-link.schema.json", invalid_link)
    _write_jsonl(
        root / "publication" / "links" / "manuscript-links.jsonl",
        [invalid_link],
    )
    (root.parent / "outside.md").write_text(
        (
            f"% STC:BEGIN {MANUSCRIPT_LOCATION}\n"
            f"{MANUSCRIPT_NONCANONICAL_BLOCK}\n"
            f"% STC:END {MANUSCRIPT_LOCATION}\n"
        ),
        encoding="utf-8",
    )

    report = validate_repository(root, gate="G6")

    assert _diagnostic_contract(report) == (
        (
            "RECORD_SCHEMA_INVALID",
            "publication/links/manuscript-links.jsonl",
            "PAPER-C1",
            "manuscript_path",
            "manuscript-link.schema.json",
            (
                "record PAPER-C1 violates manuscript-link.schema.json at "
                "manuscript_path (pattern)"
            ),
        ),
        (
            "CLAIM_MANUSCRIPT_LOCATION_MISSING",
            "publication/links/manuscript-links.jsonl",
            "PAPER-C1",
            "location",
            "G6",
            (
                "release claim PAPER-C1 has no manuscript location in "
                "publication/links/manuscript-links.jsonl at G6"
            ),
        ),
    )


def test_schema_valid_manuscript_symlink_outside_root_is_diagnosed(
    tmp_path: Path,
) -> None:
    source, evidence, claim = _paper_release_records()
    artifact = _artifact_node(
        claim_id="PAPER-C1",
        gate_status="CURRENT",
    )
    root = _claim_profile_repo(
        tmp_path,
        sources=[source],
        evidence=[evidence],
        claims=[claim],
        artifacts=[artifact],
        manuscript_links=[],
    )
    external_manuscript = root.parent / "outside.md"
    external_manuscript.write_text(
        (
            f"% STC:BEGIN {MANUSCRIPT_LOCATION}\n"
            f"{MANUSCRIPT_SENTENCE}\n"
            f"% STC:END {MANUSCRIPT_LOCATION}\n"
        ),
        encoding="utf-8",
    )
    manuscript_path = root / "manuscript" / "paper-c.md"
    manuscript_path.parent.mkdir(parents=True, exist_ok=True)
    manuscript_path.symlink_to(external_manuscript)
    link = _manuscript_link()
    assert _schema_errors("manuscript-link.schema.json", link) == ()
    _write_jsonl(
        root / "publication" / "links" / "manuscript-links.jsonl",
        [link],
    )

    report = validate_repository(root, gate="G6")

    assert _diagnostic_contract(report) == (
        (
            "REFERENCED_PATH_OUTSIDE_ROOT",
            "publication/links/manuscript-links.jsonl",
            "PAPER-C1",
            "manuscript_path",
            "manuscript/paper-c.md",
            (
                "manuscript link PAPER-C1 path manuscript/paper-c.md "
                "resolves outside repository root"
            ),
        ),
    )


def test_invalid_negative_result_is_diagnosed_and_does_not_resolve_claim(
    tmp_path: Path,
) -> None:
    source = _claim_profile_source(1)
    evidence = _claim_profile_evidence(1, source)
    evidence["claim_ids"] = ["CL-STC-0001"]
    claim = _claim_profile_claim(
        [_claim_profile_support_ref(evidence, source)],
        minimum_independent_sources=1,
    )
    claim["status"] = "FALSIFIED/NARROWED"
    claim["result_ids"] = ["RES-STC-0001"]
    claim["result_provenance"] = {
        "asserted_modality": "SOURCE_REPORTED",
        "calibration_modalities": [],
    }
    invalid_result = _negative_result()
    del invalid_result["outcome"]
    assert _schema_errors("negative-result.schema.json", invalid_result)
    root = _claim_profile_repo(
        tmp_path,
        sources=[source],
        evidence=[evidence],
        claims=[claim],
        negative_results=[],
    )
    _write_jsonl(
        root / "registry" / "negative-results.jsonl",
        [invalid_result],
    )

    report = validate_repository(root)

    assert _diagnostic_contract(report) == (
        (
            "RECORD_SCHEMA_INVALID",
            "registry/negative-results.jsonl",
            "RES-STC-0001",
            "$",
            "negative-result.schema.json",
            (
                "record RES-STC-0001 violates negative-result.schema.json "
                "at $ (required)"
            ),
        ),
        (
            "NEGATIVE_RESULT_REGISTRY_OMISSION",
            "registry/claims.jsonl",
            "CL-STC-0001",
            "result_ids[0]",
            "RES-STC-0001",
            (
                "FALSIFIED/NARROWED claim CL-STC-0001 result RES-STC-0001 "
                "is absent from registry/negative-results.jsonl"
            ),
        ),
    )


def test_invalid_release_manifest_is_excluded_from_stale_scan(
    tmp_path: Path,
) -> None:
    source = _claim_profile_source(1)
    evidence = _claim_profile_evidence(1, source)
    claim = _claim_profile_claim(
        [_claim_profile_support_ref(evidence, source)],
        minimum_independent_sources=1,
    )
    stale_artifact = _artifact_node(
        claim_id="CL-STC-0001",
        gate_status="STALE",
        artifact_path="artifacts/release-payload.json",
        payload=RELEASE_ARTIFACT_PAYLOAD,
    )
    gate_artifact = _artifact_node(
        claim_id="CL-STC-0001",
        gate_status="CURRENT",
        artifact_id="ART-STC-0002",
        artifact_path="artifacts/g8.json",
        payload=GATE_ARTIFACT_PAYLOAD,
    )
    invalid_manifest = _release_manifest(
        artifact_digest=str(stale_artifact["output_digest"]),
        gate_digest=str(gate_artifact["output_digest"]),
    )
    del invalid_manifest["released_at"]
    assert _schema_errors("release-manifest.schema.json", invalid_manifest)
    root = _claim_profile_repo(
        tmp_path,
        sources=[source],
        evidence=[evidence],
        claims=[claim],
        artifacts=[stale_artifact, gate_artifact],
        artifact_payloads={
            "artifacts/release-payload.json": RELEASE_ARTIFACT_PAYLOAD,
            "artifacts/g8.json": GATE_ARTIFACT_PAYLOAD,
        },
    )
    relative_path = (
        "publication/releases/stc-paper-v1.0.0/release-manifest.json"
    )
    _write_json(root / relative_path, invalid_manifest)

    report = validate_repository(root)

    assert _diagnostic_contract(report) == (
        (
            "RECORD_SCHEMA_INVALID",
            relative_path,
            "stc-paper-v1.0.0",
            "$",
            "release-manifest.schema.json",
            (
                "record stc-paper-v1.0.0 violates "
                "release-manifest.schema.json at $ (required)"
            ),
        ),
    )


def test_invalid_presentation_handoff_is_excluded_from_stale_scan(
    tmp_path: Path,
) -> None:
    source = _claim_profile_source(1)
    evidence = _claim_profile_evidence(1, source)
    claim = _claim_profile_claim(
        [_claim_profile_support_ref(evidence, source)],
        minimum_independent_sources=1,
    )
    stale_artifact = _artifact_node(
        claim_id="CL-STC-0001",
        gate_status="STALE",
        artifact_path="artifacts/g8.json",
        payload=GATE_ARTIFACT_PAYLOAD,
    )
    invalid_handoff = _presentation_handoff(
        gate_digest=str(stale_artifact["output_digest"])
    )
    del invalid_handoff["handed_off_at"]
    assert _schema_errors("presentation-handoff.schema.json", invalid_handoff)
    root = _claim_profile_repo(
        tmp_path,
        sources=[source],
        evidence=[evidence],
        claims=[claim],
        artifacts=[stale_artifact],
        artifact_payloads={
            "artifacts/g8.json": GATE_ARTIFACT_PAYLOAD,
        },
    )
    relative_path = (
        "publication/handoff/stc-paper-v1.0.0/presentation-handoff.json"
    )
    _write_json(root / relative_path, invalid_handoff)

    report = validate_repository(root)

    assert _diagnostic_contract(report) == (
        (
            "RECORD_SCHEMA_INVALID",
            relative_path,
            "stc-deck-v1.0.0",
            "$",
            "presentation-handoff.schema.json",
            (
                "record stc-deck-v1.0.0 violates "
                "presentation-handoff.schema.json at $ (required)"
            ),
        ),
    )


def test_control_schema_inventory_matches_declared_export_contract() -> None:
    manifest_schema_ids = {
        _manifest_value(entry, "filename"): _manifest_value(entry, "schema_id")
        for entry in CONTROL_SCHEMA_EXPORT_MANIFEST
    }
    discovered_schema_ids = {
        path.name: _read_schema(path)["$id"]
        for path in _schema_paths(PACKAGE_ROOT)
    }

    assert len(CONTROL_SCHEMA_EXPORT_MANIFEST) == len(manifest_schema_ids)
    assert set(manifest_schema_ids) == EXPECTED_CONTROL_SCHEMA_FILENAMES
    assert set(discovered_schema_ids) == EXPECTED_CONTROL_SCHEMA_FILENAMES
    assert manifest_schema_ids == discovered_schema_ids
    for entry in CONTROL_SCHEMA_EXPORT_MANIFEST:
        assert all(
            isinstance(_manifest_value(entry, field), str)
            and _manifest_value(entry, field).strip()
            for field in ("filename", "schema_id", "owner", "model_name")
        )
    for filename, schema_id in discovered_schema_ids.items():
        assert isinstance(schema_id, str)
        parsed_schema_id = urlparse(schema_id)
        assert parsed_schema_id.scheme == "https"
        assert parsed_schema_id.netloc
        assert not parsed_schema_id.query
        assert not parsed_schema_id.fragment
        assert parsed_schema_id.path.endswith(f"/{filename}")

    report = validate_repository(PACKAGE_ROOT)

    assert report.schema_inventory == "complete"
    assert not _schema_diagnostics(report)


def test_control_schema_inventory_rejects_an_unexpected_schema_at_same_count(
    tmp_path: Path,
) -> None:
    root = _schema_repo(tmp_path)
    schema_directory = _schema_directory(root)
    (schema_directory / "source.schema.json").rename(
        schema_directory / "unexpected.schema.json"
    )

    report = validate_repository(root)

    assert len(_schema_paths(root)) == len(EXPECTED_CONTROL_SCHEMA_FILENAMES)
    assert report.schema_inventory == "incomplete"
    assert "SCHEMA_INVENTORY_MISMATCH" in _schema_diagnostics(report)


def test_every_control_schema_has_canonical_metadata_and_closed_root_object() -> None:
    schemas = [_read_schema(path) for path in _schema_paths(PACKAGE_ROOT)]
    schema_ids = [schema.get("$id") for schema in schemas]

    assert schemas
    assert all(isinstance(schema_id, str) and schema_id.strip() for schema_id in schema_ids)
    assert len(schema_ids) == len(set(schema_ids))
    for schema in schemas:
        Draft202012Validator.check_schema(schema)
        assert schema["$schema"] == DRAFT_2020_12_URI
        assert isinstance(schema["x-stc-owner"], str) and schema["x-stc-owner"].strip()
        assert isinstance(schema["x-stc-model"], str) and schema["x-stc-model"].strip()
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False


def test_control_schema_metadata_mutations_report_public_diagnostics(
    tmp_path: Path,
) -> None:
    mutations = (
        ("duplicate_id", "SCHEMA_ID_DUPLICATE"),
        ("missing_owner", "SCHEMA_METADATA_MISSING"),
        ("missing_model", "SCHEMA_METADATA_MISSING"),
        ("wrong_draft", "SCHEMA_DRAFT_MISMATCH"),
    )

    for mutation, expected_code in mutations:
        root = _schema_repo(tmp_path / mutation)
        schema_directory = _schema_directory(root)
        source_path = schema_directory / "source.schema.json"
        source_schema = _read_schema(source_path)

        if mutation == "duplicate_id":
            claim_path = schema_directory / "claim.schema.json"
            claim_schema = _read_schema(claim_path)
            claim_schema["$id"] = source_schema["$id"]
            _write_schema(claim_path, claim_schema)
        elif mutation == "missing_owner":
            del source_schema["x-stc-owner"]
            _write_schema(source_path, source_schema)
        elif mutation == "missing_model":
            del source_schema["x-stc-model"]
            _write_schema(source_path, source_schema)
        else:
            source_schema["$schema"] = "https://json-schema.org/draft/2019-09/schema"
            _write_schema(source_path, source_schema)

        report = validate_repository(root)

        assert expected_code in _schema_diagnostics(report)


def test_common_schema_is_the_only_owner_of_shared_definitions_and_nested_refs(
    tmp_path: Path,
) -> None:
    root = _schema_repo(tmp_path)
    schema_directory = _schema_directory(root)
    common_path = schema_directory / "common.schema.json"
    common_schema = _read_schema(common_path)
    common_id = common_schema["$id"]
    common_definitions = common_schema["$defs"]

    assert isinstance(common_id, str) and common_id.strip()
    assert isinstance(common_definitions, Mapping)
    assert COMMON_SHARED_DEFINITION_NAMES <= set(common_definitions)
    for path in _schema_paths(root):
        if path == common_path:
            continue
        definitions = _read_schema(path).get("$defs", {})
        assert isinstance(definitions, Mapping)
        assert not (COMMON_SHARED_DEFINITION_NAMES & set(definitions))

    required_references = (
        (
            "source.schema.json",
            ("properties", "versions", "items", "$ref"),
            "SourceVersion",
        ),
        (
            "source.schema.json",
            ("properties", "local_artifacts", "items", "$ref"),
            "ArtifactRef",
        ),
        (
            "source.schema.json",
            ("properties", "code_records", "items", "$ref"),
            "ArtifactRef",
        ),
        (
            "source.schema.json",
            ("properties", "data_records", "items", "$ref"),
            "ArtifactRef",
        ),
        (
            "claim.schema.json",
            ("properties", "result_provenance", "$ref"),
            "ClaimResultProvenance",
        ),
        (
            "audit-attestation.schema.json",
            ("properties", "subject_refs", "items", "$ref"),
            "SubjectRef",
        ),
        (
            "gate.schema.json",
            ("properties", "artifact_dag_digest", "$ref"),
            "sha256",
        ),
        (
            "evidence.schema.json",
            ("properties", "relation", "$ref"),
            "evidenceRelation",
        ),
        (
            "evidence.schema.json",
            ("properties", "source_grade", "$ref"),
            "sourceGrade",
        ),
        (
            "evidence.schema.json",
            ("properties", "warrant", "$ref"),
            "warrant",
        ),
        (
            "evidence.schema.json",
            ("properties", "reproduction_strength", "$ref"),
            "reproductionStrength",
        ),
        (
            "negative-result.schema.json",
            ("properties", "result_modality", "$ref"),
            "resultModality",
        ),
        (
            "source.schema.json",
            ("properties", "source_id", "$ref"),
            "sourceId",
        ),
        (
            "source.schema.json",
            ("properties", "source_grade", "$ref"),
            "sourceGrade",
        ),
        (
            "source.schema.json",
            ("properties", "review_status", "$ref"),
            "reviewStatus",
        ),
    )
    for filename, path, definition_name in required_references:
        schema = _read_schema(schema_directory / filename)
        assert _schema_value_at_path(schema, *path) == (
            f"{common_id}#/$defs/{definition_name}"
        )

    for case_index, (filename, ref_path, definition_name) in enumerate(
        required_references
    ):
        case_root = _schema_repo(tmp_path / f"inline-{case_index:02d}")
        case_schema_path = _schema_directory(case_root) / filename
        case_schema = _read_schema(case_schema_path)
        target_parent = _schema_value_at_path(case_schema, *ref_path[:-2])
        assert isinstance(target_parent, dict)
        target_parent[ref_path[-2]] = deepcopy(common_definitions[definition_name])
        target_ref = f"{common_id}#/$defs/{definition_name}"
        case_schema["x-stc-dead-shared-definition-ref"] = {"$ref": target_ref}
        _write_schema(case_schema_path, case_schema)

        report = validate_repository(case_root)
        diagnostic_codes = _schema_diagnostics(report)

        assert "SCHEMA_SHARED_DEFINITION_INLINE" in diagnostic_codes, (
            filename,
            ".".join(ref_path[:-1]),
            definition_name,
            diagnostic_codes,
        )


@pytest.mark.parametrize(
    ("filename", "expected_model", "payload"),
    tuple(
        (filename, model_type, payload)
        for filename, (model_type, payload) in _model_payloads().items()
    ),
    ids=tuple(_model_payloads()),
)
def test_model_backed_schema_exports_round_trip_and_reject_unknown_fields(
    filename: str,
    expected_model: type[object],
    payload: dict[str, object],
) -> None:
    manifest = {entry.filename: entry for entry in CONTROL_SCHEMA_EXPORT_MANIFEST}
    exported_model = manifest[filename].model_type
    assert exported_model is expected_model, (
        f"{filename}: manifest model_type is {exported_model!r}, "
        f"expected {expected_model.__name__}"
    )

    record = exported_model.from_dict(deepcopy(payload))
    serialized = record.to_dict()
    assert _schema_errors(filename, serialized) == (), filename
    assert exported_model.from_dict(serialized).to_dict() == serialized, filename

    unknown_payload = deepcopy(serialized)
    unknown_payload["unexpected_field"] = True
    assert _schema_errors(filename, unknown_payload), filename
    with pytest.raises((TypeError, ValueError), match="unexpected_field"):
        exported_model.from_dict(unknown_payload)


def test_trusted_key_schema_uses_typed_model_and_shared_aware_datetimes() -> None:
    from stc_research import models

    schema = json.loads(
        (
            PACKAGE_ROOT
            / "schemas"
            / "control"
            / "trusted-key-set.schema.json"
        ).read_text(encoding="utf-8")
    )
    key_schema = schema["$defs"]["key"]
    aware_datetime_ref = (
        "https://schemas.sleep-time-compute.org/control/"
        "common.schema.json#/$defs/awareDateTime"
    )
    nullable_aware_datetime = {
        "oneOf": [
            {"$ref": aware_datetime_ref},
            {"type": "null"},
        ]
    }

    assert {"signer_id", "revoked"} <= set(key_schema["required"])
    assert key_schema["properties"]["signer_id"] == {
        "$ref": (
            "https://schemas.sleep-time-compute.org/control/"
            "common.schema.json#/$defs/nonEmptyString"
        )
    }
    assert key_schema["properties"]["revoked"] == {"type": "boolean"}
    assert key_schema["properties"]["valid_from"] == {
        "$ref": aware_datetime_ref
    }
    assert key_schema["properties"]["valid_until"] == nullable_aware_datetime
    assert schema["properties"]["issued_at"] == {
        "$ref": aware_datetime_ref
    }

    TrustedKeySet = models.TrustedKeySet
    export = next(
        entry
        for entry in CONTROL_SCHEMA_EXPORT_MANIFEST
        if entry.filename == "trusted-key-set.schema.json"
    )
    assert export.model_type is TrustedKeySet
    payload = _trusted_key_set_payload()
    record = TrustedKeySet.from_dict(deepcopy(payload))
    assert record.to_dict() == payload
    assert _schema_errors("trusted-key-set.schema.json", payload) == ()


@pytest.mark.parametrize(
    "invalid_case",
    [
        "empty-key-set-id",
        "zero-version",
        "boolean-version",
        "empty-keys",
        "bad-set-digest",
        "naive-issued-at",
        "unknown-root-field",
        "empty-key-id",
        "empty-signer-id",
        "empty-algorithm",
        "empty-public-key",
        "empty-roles",
        "empty-role-entry",
        "naive-valid-from",
        "naive-valid-until",
        "non-boolean-revoked",
        "integer-revoked",
        "unknown-key-field",
    ],
)
def test_trusted_key_schema_and_model_reject_every_invalid_nested_field(
    invalid_case: str,
) -> None:
    from stc_research import models

    payload = _trusted_key_set_payload()
    keys = payload["keys"]
    assert isinstance(keys, list)
    key = keys[0]
    assert isinstance(key, dict)
    refresh_digest = True

    if invalid_case == "empty-key-set-id":
        payload["key_set_id"] = ""
    elif invalid_case == "zero-version":
        payload["version"] = 0
    elif invalid_case == "boolean-version":
        payload["version"] = True
    elif invalid_case == "empty-keys":
        payload["keys"] = []
    elif invalid_case == "bad-set-digest":
        payload["set_digest"] = "not-a-digest"
        refresh_digest = False
    elif invalid_case == "naive-issued-at":
        payload["issued_at"] = "2026-07-01T00:00:00"
    elif invalid_case == "unknown-root-field":
        payload["unknown_root"] = "forbidden"
    elif invalid_case == "empty-key-id":
        key["key_id"] = ""
    elif invalid_case == "empty-signer-id":
        key["signer_id"] = ""
    elif invalid_case == "empty-algorithm":
        key["algorithm"] = ""
    elif invalid_case == "empty-public-key":
        key["public_key"] = ""
    elif invalid_case == "empty-roles":
        key["roles"] = []
    elif invalid_case == "empty-role-entry":
        key["roles"] = [""]
    elif invalid_case == "naive-valid-from":
        key["valid_from"] = "2026-07-01T00:00:00"
    elif invalid_case == "naive-valid-until":
        key["valid_until"] = "2027-07-01T00:00:00"
    elif invalid_case == "non-boolean-revoked":
        key["revoked"] = "false"
    elif invalid_case == "integer-revoked":
        key["revoked"] = 1
    else:
        key["unknown_nested"] = "forbidden"

    if refresh_digest:
        _refresh_trusted_key_set_digest(payload)

    assert _schema_errors("trusted-key-set.schema.json", payload)
    with pytest.raises((TypeError, ValueError)):
        models.TrustedKeySet.from_dict(payload)


def test_common_nested_models_round_trip_against_shared_definitions() -> None:
    schemas, registry = _schema_catalog()
    common_schema = schemas["common.schema.json"]
    common_id = common_schema["$id"]
    assert isinstance(common_id, str)
    cases = (
        (
            "SubjectRef",
            SubjectRef,
            {"path": "reports/a.json", "sha256": SHA_A},
        ),
        ("SourceVersion", SourceVersion, _source()["versions"][0]),
        ("ArtifactRef", ArtifactRef, _artifact_ref()),
        (
            "ClaimResultProvenance",
            ClaimResultProvenance,
            {
                "asserted_modality": "NOT_APPLICABLE",
                "calibration_modalities": [],
            },
        ),
    )

    for definition_name, model_type, payload in cases:
        record = model_type.from_dict(deepcopy(payload))
        serialized = record.to_dict()
        assert serialized == payload, definition_name
        validator = Draft202012Validator(
            {"$ref": f"{common_id}#/$defs/{definition_name}"},
            registry=registry,
            format_checker=FormatChecker(),
        )
        errors = tuple(
            error.message
            for error in sorted(
                validator.iter_errors(serialized),
                key=lambda error: (
                    tuple(str(part) for part in error.absolute_path),
                    error.message,
                ),
            )
        )
        assert errors == (), definition_name
        assert model_type.from_dict(serialized).to_dict() == serialized, definition_name


def _set_payload_path(
    payload: dict[str, object],
    path: tuple[str | int, ...],
    value: object,
) -> dict[str, object]:
    mutated = deepcopy(payload)
    target: object = mutated
    for segment in path[:-1]:
        if isinstance(target, dict):
            assert isinstance(segment, str)
        else:
            assert isinstance(target, list)
            assert isinstance(segment, int)
        target = target[segment]
    if isinstance(target, dict):
        assert isinstance(path[-1], str)
    else:
        assert isinstance(target, list)
        assert isinstance(path[-1], int)
    target[path[-1]] = value
    return mutated


def _common_definition_errors(
    definition_name: str,
    payload: object,
) -> tuple[str, ...]:
    schemas, registry = _schema_catalog()
    common_id = schemas["common.schema.json"]["$id"]
    assert isinstance(common_id, str)
    validator = Draft202012Validator(
        {"$ref": f"{common_id}#/$defs/{definition_name}"},
        registry=registry,
        format_checker=FormatChecker(),
    )
    return tuple(
        error.message
        for error in sorted(
            validator.iter_errors(payload),
            key=lambda error: (
                tuple(str(part) for part in error.absolute_path),
                error.message,
            ),
        )
    )


def _common_nested_field_cases() -> tuple[
    tuple[
        str,
        type[object],
        dict[str, object],
        dict[str, object],
    ],
    ...,
]:
    claim_payload = _claim()
    support_refs = claim_payload["support_refs"]
    evidence_requirement = claim_payload["evidence_requirement"]
    assert isinstance(support_refs, list)
    assert isinstance(support_refs[0], dict)
    assert isinstance(evidence_requirement, dict)

    return (
        (
            "SubjectRef",
            SubjectRef,
            {"path": "reports/a.json", "sha256": SHA_A},
            {
                "path": "../escape",
                "sha256": "bad",
            },
        ),
        (
            "SourceVersion",
            SourceVersion,
            _source()["versions"][0],
            {
                "identifier": "",
                "public_date": "2026-02-30",
                "date_precision": "WEEK",
                "status": "",
                "canonical_url": "",
                "local_artifact_id": "ART-STC-1",
                "sha256": "bad",
                "supersedes": "",
                "derived_from": [""],
            },
        ),
        (
            "ArtifactRef",
            ArtifactRef,
            _artifact_ref(),
            {
                "artifact_id": "ART-STC-1",
                "canonical_url": "",
                "local_path": "../paper.pdf",
                "sha256": "bad",
                "media_type": "",
                "availability": "",
                "license_id": "",
                "license_evidence_url": "",
                "license_evidence_sha256": "bad",
                "redistribution_allowed": "yes",
                "license_review_disposition": "",
                "rights_reviewer": "",
                "rights_reviewed_at": "not-a-date",
                "rights_review_expires_at": "2026-02-30T12:00:00+09:00",
            },
        ),
        (
            "ClaimResultProvenance",
            ClaimResultProvenance,
            {
                "asserted_modality": "NOT_APPLICABLE",
                "calibration_modalities": [],
            },
            {
                "asserted_modality": "MAGIC",
                "calibration_modalities": ["NOT_APPLICABLE"],
            },
        ),
        (
            "ClaimSupportRef",
            ClaimSupportRef,
            support_refs[0],
            {
                "evidence_id": "EV-STC-1",
                "source_id": "SRC-STC-1",
                "source_version": "",
                "source_version_digest": "bad",
                "artifact_id": "ART-STC-1",
                "support_span_digest": "bad",
                "relation": "MAGIC",
                "role": "MAGIC",
                "source_grade": "Z",
                "warrant": "OPINION",
                "reproduction_strength": "r9",
            },
        ),
        (
            "ClaimEvidenceRequirement",
            ClaimEvidenceRequirement,
            evidence_requirement,
            {
                "allowed_claim_classes": [],
                "allowed_source_grades": ["X"],
                "allowed_warrants": [],
                "minimum_reproduction_strength": "r9",
                "minimum_independent_sources": -1,
                "requires_result_block": "yes",
                "minimum_independent_result_paths": -1,
            },
        ),
        (
            "DigestRef",
            DigestRef,
            {"artifact_id": "ART-STC-0001", "sha256": SHA_A},
            {
                "artifact_id": "ART-STC-1",
                "sha256": "bad",
            },
        ),
    )


_COMMON_NESTED_FIELD_CASES = _common_nested_field_cases()


@pytest.mark.parametrize(
    ("definition_name", "model_type", "payload", "invalid_fields"),
    _COMMON_NESTED_FIELD_CASES,
    ids=tuple(case[0] for case in _COMMON_NESTED_FIELD_CASES),
)
def test_common_nested_fields_have_schema_model_rejection_parity(
    definition_name: str,
    model_type: type[object],
    payload: dict[str, object],
    invalid_fields: dict[str, object],
) -> None:
    model_field_names = {field.name for field in fields(model_type)}
    assert set(invalid_fields) == model_field_names, definition_name
    assert model_type.from_dict(deepcopy(payload)).to_dict() == payload
    assert _common_definition_errors(definition_name, payload) == ()

    for field_name, invalid_value in invalid_fields.items():
        mutated = _set_payload_path(payload, (field_name,), invalid_value)
        assert _common_definition_errors(definition_name, mutated), (
            definition_name,
            field_name,
        )
        with pytest.raises((TypeError, ValueError)):
            model_type.from_dict(mutated)


@pytest.mark.parametrize(
    ("filename", "model_type", "payload", "path", "invalid_value"),
    (
        pytest.param(
            "source.schema.json",
            SourceRecord,
            _source(),
            ("source_id",),
            "SRC-STC-1",
            id="source-stable-id",
        ),
        pytest.param(
            "source.schema.json",
            SourceRecord,
            _source(),
            ("local_artifacts", 0, "sha256"),
            "bad",
            id="source-artifact-sha256",
        ),
        pytest.param(
            "source.schema.json",
            SourceRecord,
            _source(),
            ("source_grade",),
            "Z",
            id="source-grade",
        ),
        pytest.param(
            "source.schema.json",
            SourceRecord,
            _source(),
            ("review_status",),
            "ABSTRACT",
            id="source-review-status",
        ),
        pytest.param(
            "evidence.schema.json",
            EvidenceRecord,
            _evidence(1, anchor="Methods"),
            ("relation",),
            "LIKELY",
            id="evidence-relation",
        ),
        pytest.param(
            "evidence.schema.json",
            EvidenceRecord,
            _evidence(1, anchor="Methods"),
            ("warrant",),
            "OPINION",
            id="evidence-warrant",
        ),
        pytest.param(
            "evidence.schema.json",
            EvidenceRecord,
            _evidence(1, anchor="Methods"),
            ("reproduction_strength",),
            "r9",
            id="evidence-reproduction-strength",
        ),
        pytest.param(
            "evidence.schema.json",
            EvidenceRecord,
            _evidence(1, anchor="Methods"),
            ("reviewed_at",),
            "not-a-date",
            id="evidence-reviewed-at-malformed",
        ),
        pytest.param(
            "evidence.schema.json",
            EvidenceRecord,
            _evidence(1, anchor="Methods"),
            ("reviewed_at",),
            "2026-07-25T12:00:00",
            id="evidence-reviewed-at-naive",
        ),
        pytest.param(
            "evidence.schema.json",
            EvidenceRecord,
            _evidence(1, anchor="Methods"),
            ("reviewed_at",),
            "2026-02-30T12:00:00+09:00",
            id="evidence-reviewed-at-impossible-calendar-date",
        ),
        pytest.param(
            "claim.schema.json",
            ClaimRecord,
            _claim(),
            ("result_provenance", "asserted_modality"),
            "MAGIC",
            id="claim-result-modality",
        ),
        pytest.param(
            "audit-attestation.schema.json",
            AuditAttestation,
            _model_payloads()["audit-attestation.schema.json"][1],
            ("signed_at",),
            "2026-07-25T12:00:00",
            id="audit-attestation-signed-at-naive",
        ),
        pytest.param(
            "audit-attestation.schema.json",
            AuditAttestation,
            _model_payloads()["audit-attestation.schema.json"][1],
            ("signed_at",),
            "2026-02-30T12:00:00+09:00",
            id="audit-attestation-signed-at-impossible-calendar-date",
        ),
        pytest.param(
            "gate-evaluation-attestation.schema.json",
            GateEvaluationAttestation,
            _model_payloads()["gate-evaluation-attestation.schema.json"][1],
            ("signed_at",),
            "2026-07-25T12:00:00",
            id="gate-evaluation-attestation-signed-at-naive",
        ),
        pytest.param(
            "human-approval.schema.json",
            HumanApproval,
            _model_payloads()["human-approval.schema.json"][1],
            ("signed_at",),
            "2026-07-25T12:00:00",
            id="human-approval-signed-at-naive",
        ),
        pytest.param(
            "gate.schema.json",
            GateRecord,
            _model_payloads()["gate.schema.json"][1],
            ("evaluated_at",),
            "2026-07-25",
            id="gate-evaluated-at-date-only",
        ),
        pytest.param(
            "gate.schema.json",
            GateRecord,
            _model_payloads()["gate.schema.json"][1],
            ("evaluated_at",),
            "2026-07-25T12:00:00",
            id="gate-evaluated-at-naive",
        ),
        pytest.param(
            "gate.schema.json",
            GateRecord,
            _model_payloads()["gate.schema.json"][1],
            ("evaluated_at",),
            "0000-07-25T12:00:00+09:00",
            id="gate-evaluated-at-year-zero",
        ),
        pytest.param(
            "gate.schema.json",
            GateRecord,
            _model_payloads()["gate.schema.json"][1],
            ("evaluated_at",),
            "2026-07-25T12:00:00-00:00",
            id="gate-evaluated-at-unknown-offset",
        ),
    ),
)
def test_model_backed_root_schemas_reject_invalid_shared_values(
    filename: str,
    model_type: type[object],
    payload: dict[str, object],
    path: tuple[str | int, ...],
    invalid_value: object,
) -> None:
    assert model_type.from_dict(deepcopy(payload)).to_dict() == payload

    mutated = _set_payload_path(payload, path, invalid_value)
    assert _schema_errors(filename, mutated), (filename, path)
    with pytest.raises((TypeError, ValueError)):
        model_type.from_dict(mutated)
