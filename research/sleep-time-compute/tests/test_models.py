from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import pytest

from stc_research import models
from stc_research.models import (
    ArtifactNode,
    ClaimRecord,
    ClaimResultProvenance,
    EvidenceRecord,
    GateEvaluationAttestation,
    GateRecord,
    HumanApproval,
    HypothesisRecord,
    PaperCard,
    QuestionRecord,
    ResultModality,
    SourceRecord,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def trusted_key_set_dict() -> dict[str, object]:
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
    unsigned = dict(payload)
    unsigned.pop("set_digest", None)
    canonical_body = (
        json.dumps(
            unsigned,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    payload["set_digest"] = hashlib.sha256(canonical_body).hexdigest()


class _SpoofedClaimResultProvenance(ClaimResultProvenance):
    __slots__ = ()

    def to_dict(self):
        return {
            "asserted_modality": "ACTUAL_HARDWARE",
            "calibration_modalities": [],
        }


def artifact_dict(number: int = 1) -> dict[str, object]:
    return {
        "artifact_id": f"ART-STC-{number:04d}",
        "canonical_url": "https://example.org/paper.pdf",
        "local_path": f"artifacts/paper-{number}.pdf",
        "sha256": SHA_A,
        "media_type": "application/pdf",
        "availability": "VENDORED",
        "license_id": "CC-BY-4.0",
        "license_evidence_url": "https://example.org/license",
        "license_evidence_sha256": SHA_B,
        "redistribution_allowed": True,
        "license_review_disposition": "ALLOW",
        "rights_reviewer": "reviewer-1",
        "rights_reviewed_at": "2026-07-25",
        "rights_review_expires_at": None,
    }


def source_version_dict(
    identifier: str = "v1", artifact_number: int = 1
) -> dict[str, object]:
    return {
        "identifier": identifier,
        "public_date": "2025-04-17",
        "date_precision": "DAY",
        "status": "PREPRINT",
        "canonical_url": "https://arxiv.org/abs/2504.13171",
        "local_artifact_id": f"ART-STC-{artifact_number:04d}",
        "sha256": SHA_A,
        "supersedes": None,
        "derived_from": [],
    }


def source_dict() -> dict[str, object]:
    artifact = artifact_dict()
    return {
        "source_id": "SRC-STC-0001",
        "legacy_ids": ["S-001"],
        "canonical_key": "arxiv:2504.13171:v1",
        "title": "Sleep-time Compute",
        "authors": ["Author One"],
        "earliest_public_date": "2025-04-17",
        "date_precision": "DAY",
        "venue_status": "ARXIV",
        "peer_review_status": "PREPRINT",
        "versions": [source_version_dict()],
        "source_grade": "B",
        "review_status": "A-ABS",
        "urls": ["https://arxiv.org/abs/2504.13171"],
        "local_artifacts": [artifact],
        "citation_key": "shibata2025sleeptime",
        "code_records": [],
        "data_records": [],
        "rights_summary": "Redistribution permitted by reviewed artifact record.",
        "topics": ["sleep-time-compute"],
        "affiliations": ["Example Lab"],
        "inclusion_reason": "Directly studies sleep-time compute.",
        "non_claim": "Abstract review does not establish full methods.",
        "last_verified": "2026-07-25",
    }


def support_dict(
    *,
    evidence_number: int = 1,
    source_id: str = "SRC-STC-0001",
    source_version: str = "v1",
    source_grade: str = "A",
    warrant: str = "SUMM",
    reproduction_strength: str = "r0",
) -> dict[str, object]:
    return {
        "evidence_id": f"EV-STC-{evidence_number:05d}",
        "source_id": source_id,
        "source_version": source_version,
        "source_version_digest": SHA_A,
        "artifact_id": "ART-STC-0001",
        "support_span_digest": SHA_B,
        "relation": "SUPPORTS",
        "role": "RELEASE_SUPPORT",
        "source_grade": source_grade,
        "warrant": warrant,
        "reproduction_strength": reproduction_strength,
    }


def requirement_dict(
    *,
    claim_class: str = "SOURCE-SUMMARY",
    source_grades: list[str] | None = None,
    warrants: list[str] | None = None,
    minimum_reproduction_strength: str = "r0",
    minimum_independent_sources: int = 1,
) -> dict[str, object]:
    return {
        "allowed_claim_classes": [claim_class],
        "allowed_source_grades": source_grades or ["A"],
        "allowed_warrants": warrants or ["SUMM"],
        "minimum_reproduction_strength": minimum_reproduction_strength,
        "minimum_independent_sources": minimum_independent_sources,
        "requires_result_block": False,
        "minimum_independent_result_paths": 0,
    }


def claim_dict() -> dict[str, object]:
    support = support_dict()
    return {
        "claim_id": "CL-STC-0001",
        "statement": "The reviewed source describes a bounded method.",
        "headline_quantitative": False,
        "claim_class": "SOURCE-SUMMARY",
        "status": "DRAFT",
        "scope": "The reviewed source version only.",
        "assumptions": [],
        "source_ids": ["SRC-STC-0001"],
        "evidence_ids": ["EV-STC-00001"],
        "support_refs": [support],
        "evidence_requirement": requirement_dict(),
        "counterevidence_ids": [],
        "result_ids": [],
        "artifact_dependencies": ["ART-STC-0001"],
        "caveats": ["Abstract-only evidence."],
        "falsifier": "A full read contradicts this bounded summary.",
        "paper_owner": "PAPER-A",
        "gate_status": "PRE-G0",
        "last_audit_date": "2026-07-25",
    }


def result_provenance_dict(
    asserted_modality: str,
    calibration_modalities: list[str] | None = None,
) -> dict[str, object]:
    return {
        "asserted_modality": asserted_modality,
        "calibration_modalities": calibration_modalities or [],
    }


def claim_with_class(
    claim_class: str,
    *,
    asserted_modality: str | None = None,
) -> dict[str, object]:
    data = claim_dict()
    data["claim_class"] = claim_class
    data["evidence_requirement"] = requirement_dict(claim_class=claim_class)
    if asserted_modality is not None:
        data["result_provenance"] = result_provenance_dict(asserted_modality)
    return data


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("review_status", "PARTIAL", "review_status"),
        ("earliest_public_date", "2025-02-30", "ISO date"),
        ("title", "", "title"),
        ("source_id", "SRC-STC-1", "source_id"),
        ("source_grade", "E", "source_grade"),
    ],
)
def test_source_record_rejects_invalid_contract_fields(field, value, message):
    data = source_dict()
    data[field] = value
    with pytest.raises(ValueError, match=message):
        SourceRecord.from_dict(data)


def test_source_record_rejects_direct_identifier_in_operational_metadata():
    data = source_dict()
    data["operational_metadata"] = {
        "event_handle": "evt-random",
        "email": "person@example.org",
    }
    with pytest.raises(ValueError, match="direct identifier"):
        SourceRecord.from_dict(data)


def test_source_record_is_immutable_and_serializes_in_field_order():
    record = SourceRecord.from_dict(source_dict())
    with pytest.raises(FrozenInstanceError):
        record.title = "Changed"
    assert list(record.to_dict())[:5] == [
        "source_id",
        "legacy_ids",
        "canonical_key",
        "title",
        "authors",
    ]
    assert record.to_dict()["review_status"] == "A-ABS"
    assert record.to_dict()["versions"][0]["identifier"] == "v1"


def test_source_record_rejects_nested_mutable_string_tuple_elements():
    data = source_dict()
    data["authors"] = [["mutable-author"]]
    with pytest.raises(TypeError, match="authors"):
        SourceRecord.from_dict(data)


def test_direct_constructor_rejects_mutable_string_tuple():
    record = SourceRecord.from_dict(source_dict())
    with pytest.raises(TypeError, match="authors"):
        replace(record, authors=[["mutable-author"]])


def test_source_record_detaches_input_lists():
    data = source_dict()
    authors = data["authors"]
    record = SourceRecord.from_dict(data)
    authors.append("Later Mutation")
    assert record.authors == ("Author One",)


def test_records_have_no_inherited_instance_dict_or_serializer_spoof_slot():
    record = SourceRecord.from_dict(source_dict())
    assert not hasattr(record, "__dict__")
    with pytest.raises(AttributeError):
        object.__setattr__(record, "to_dict", lambda: {"spoof": True})


def test_source_record_rejects_nested_source_version_subclass():
    class SpoofedSourceVersion(models.SourceVersion):
        __slots__ = ()

        def to_dict(self):
            return {"identifier": "spoofed"}

    version = SpoofedSourceVersion.from_dict(source_version_dict())
    data = source_dict()
    data["versions"] = [version]
    with pytest.raises(TypeError, match="versions.*SourceVersion"):
        SourceRecord.from_dict(data)

    record = SourceRecord.from_dict(source_dict())
    with pytest.raises(TypeError, match="versions.*SourceVersion"):
        replace(record, versions=(version,))


@pytest.mark.parametrize(
    ("public_date", "precision"),
    [
        ("20250417", "DAY"),
        ("2025-W16-4", "DAY"),
        ("0000", "YEAR"),
        ("0000-04", "MONTH"),
    ],
)
def test_source_version_rejects_noncanonical_public_dates(public_date, precision):
    data = source_dict()
    data["versions"][0]["public_date"] = public_date
    data["versions"][0]["date_precision"] = precision
    with pytest.raises(ValueError, match="public_date"):
        SourceRecord.from_dict(data)


@pytest.mark.parametrize(
    ("target", "public_date", "precision"),
    [
        ("source", "２０２５", "YEAR"),
        ("source", "٢٠٢٥-04", "MONTH"),
        ("version", "２０２５", "YEAR"),
        ("version", "٢٠٢٥-04", "MONTH"),
    ],
)
def test_public_dates_reject_unicode_digit_confusables(
    target,
    public_date,
    precision,
):
    data = source_dict()
    if target == "source":
        data["earliest_public_date"] = public_date
        data["date_precision"] = precision
    else:
        data["versions"][0]["public_date"] = public_date
        data["versions"][0]["date_precision"] = precision
    with pytest.raises(ValueError, match="public_date"):
        SourceRecord.from_dict(data)


def test_canonical_day_date_rejects_unicode_digit_confusable():
    data = source_dict()
    data["last_verified"] = "２０２６-07-25"
    with pytest.raises(ValueError, match="canonical YYYY-MM-DD"):
        SourceRecord.from_dict(data)


def evidence_dict() -> dict[str, object]:
    return {
        "evidence_id": "EV-STC-00001",
        "source_id": "SRC-STC-0001",
        "source_version": "v1",
        "source_version_digest": SHA_A,
        "anchor_kind": "PAGE",
        "anchor": "p. 4",
        "support_span_digest": SHA_B,
        "support_summary": "The source explicitly defines the operator.",
        "relation": "SUPPORTS",
        "claim_ids": ["CL-STC-0001"],
        "question_ids": ["RQ1"],
        "hypothesis_ids": ["H-STC-001"],
        "source_grade": "A",
        "warrant": "SUMM",
        "reproduction_strength": "r0",
        "assumptions": [],
        "scope": "Definition only.",
        "counterevidence_ids": [],
        "reviewer": "reviewer-1",
        "reviewed_at": "2026-07-25",
    }


def test_evidence_record_uses_exact_enums_and_ids():
    record = EvidenceRecord.from_dict(evidence_dict())
    assert record.to_dict()["relation"] == "SUPPORTS"
    bad = evidence_dict()
    bad["relation"] = "LIKELY_SUPPORTS"
    with pytest.raises(ValueError, match="relation"):
        EvidenceRecord.from_dict(bad)


@pytest.mark.parametrize(
    "reviewed_at",
    [
        "2026-07-25T12:00:00",
        "20260725T120000+09:00",
        "2026-W30-6T12:00:00+09:00",
        "２０２６-07-25T12:00:00+09:00",
        "2026-07-25T1２:00:00+09:00",
    ],
)
def test_general_reviewed_at_rejects_noncanonical_or_naive_datetime(reviewed_at):
    data = evidence_dict()
    data["reviewed_at"] = reviewed_at
    with pytest.raises(ValueError, match="canonical timezone-aware"):
        EvidenceRecord.from_dict(data)


@pytest.mark.parametrize("reviewed_at", ["2026-07-25", "2026-07-25T12:00:00+09:00"])
def test_general_reviewed_at_accepts_date_or_aware_datetime(reviewed_at):
    data = evidence_dict()
    data["reviewed_at"] = reviewed_at
    assert EvidenceRecord.from_dict(data).reviewed_at == reviewed_at


def test_claim_record_rejects_unknown_claim_class():
    data = claim_dict()
    data["claim_class"] = "OPINION"
    with pytest.raises(ValueError, match="claim_class"):
        ClaimRecord.from_dict(data)


def test_unresolved_paper_claim_is_allowed_before_release_but_rejected_at_release():
    data = claim_dict()
    data["claim_id"] = "PAPER-C1"
    data["status"] = "UNRESOLVED"
    ClaimRecord.from_dict(data)
    with pytest.raises(ValueError, match="release-admissible"):
        ClaimRecord.from_dict(data, for_release=True)


def test_release_accepts_narrowed_paper_claim():
    data = claim_dict()
    data["claim_id"] = "PAPER-C1"
    data["status"] = "FALSIFIED/NARROWED"
    ClaimRecord.from_dict(data, for_release=True)


def test_release_requires_persisted_headline_quantitative_classification():
    data = claim_dict()
    del data["headline_quantitative"]
    data["claim_id"] = "PAPER-C1"
    data["status"] = "SUPPORTED"
    with pytest.raises(ValueError, match="headline_quantitative"):
        ClaimRecord.from_dict(data, for_release=True)


def test_legacy_headline_argument_is_explicit_not_a_silent_false_default():
    data = claim_dict()
    del data["headline_quantitative"]
    record = ClaimRecord.from_dict(data, headline_quantitative=False)
    assert record.headline_quantitative is False


def test_direct_replace_cannot_bypass_persisted_headline_validation():
    record = ClaimRecord.from_dict(claim_dict())
    with pytest.raises(ValueError, match="headline quantitative"):
        replace(record, headline_quantitative=True)


def test_release_rejects_numerical_paper_claim_with_r1_headline_support():
    data = claim_dict()
    data["claim_id"] = "PAPER-C1"
    data["status"] = "SUPPORTED"
    data["headline_quantitative"] = True
    data["claim_class"] = "ORIGINAL-MEASUREMENT"
    data["result_provenance"] = result_provenance_dict("ACTUAL_MEASUREMENT")
    data["statement"] = "The measured p99 latency is 12 ms."
    data["support_refs"] = [
        support_dict(
            warrant="MEAS",
            reproduction_strength="r1",
        )
    ]
    data["evidence_requirement"] = requirement_dict(
        claim_class="ORIGINAL-MEASUREMENT",
        warrants=["MEAS"],
        minimum_reproduction_strength="r1",
    )
    with pytest.raises(ValueError, match="headline quantitative"):
        ClaimRecord.from_dict(data, for_release=True)


def test_grade_x_release_support_is_rejected_even_when_minimum_is_zero():
    data = claim_dict()
    data["support_refs"] = [support_dict(source_grade="X")]
    data["evidence_requirement"] = requirement_dict(minimum_independent_sources=0)
    with pytest.raises(ValueError, match="grade X.*RELEASE_SUPPORT"):
        ClaimRecord.from_dict(data)


def test_supported_release_requires_positive_qualifying_release_support():
    data = claim_dict()
    context = support_dict()
    context["role"] = "CONTEXT"
    data["support_refs"] = [context]
    data["evidence_requirement"] = requirement_dict(minimum_independent_sources=0)
    data["claim_id"] = "PAPER-C1"
    data["status"] = "SUPPORTED"
    data["headline_quantitative"] = False
    with pytest.raises(ValueError, match="qualifying RELEASE_SUPPORT"):
        ClaimRecord.from_dict(data, for_release=True)


@pytest.mark.parametrize(
    ("minimum_strength", "actual_strength"),
    [("r1", "r2"), ("r2", "r1")],
)
def test_headline_quantitative_claim_requires_typed_and_actual_r2(
    minimum_strength, actual_strength
):
    data = claim_dict()
    data["claim_class"] = "ORIGINAL-MEASUREMENT"
    data["headline_quantitative"] = True
    data["result_provenance"] = result_provenance_dict("ACTUAL_MEASUREMENT")
    data["statement"] = "The measured p99 latency is 12 ms."
    data["support_refs"] = [
        support_dict(
            warrant="MEAS",
            reproduction_strength=actual_strength,
        )
    ]
    data["evidence_requirement"] = requirement_dict(
        claim_class="ORIGINAL-MEASUREMENT",
        warrants=["MEAS"],
        minimum_reproduction_strength=minimum_strength,
    )
    with pytest.raises(ValueError, match="headline quantitative"):
        ClaimRecord.from_dict(data)


def test_headline_quantitative_claim_accepts_typed_and_actual_r2():
    data = claim_dict()
    data["claim_class"] = "ORIGINAL-MEASUREMENT"
    data["headline_quantitative"] = True
    data["result_provenance"] = result_provenance_dict("ACTUAL_MEASUREMENT")
    data["statement"] = "The measured p99 latency is 12 ms."
    data["support_refs"] = [support_dict(warrant="MEAS", reproduction_strength="r2")]
    data["evidence_requirement"] = requirement_dict(
        claim_class="ORIGINAL-MEASUREMENT",
        warrants=["MEAS"],
        minimum_reproduction_strength="r2",
    )
    ClaimRecord.from_dict(data)


def test_original_measurement_rejects_simulated_modality():
    data = claim_with_class(
        "ORIGINAL-MEASUREMENT",
        asserted_modality="SIMULATED",
    )
    with pytest.raises(ValueError, match="ORIGINAL-MEASUREMENT.*modality"):
        ClaimRecord.from_dict(data)


def test_original_measurement_missing_provenance_fails_closed():
    data = claim_with_class("ORIGINAL-MEASUREMENT")
    with pytest.raises(ValueError, match="explicit structured result_provenance"):
        ClaimRecord.from_dict(data)


@pytest.mark.parametrize("modality", ["ACTUAL_MEASUREMENT", "ACTUAL_HARDWARE"])
def test_original_measurement_accepts_only_actual_modalities(modality):
    data = claim_with_class(
        "ORIGINAL-MEASUREMENT",
        asserted_modality=modality,
    )
    ClaimRecord.from_dict(data)


@pytest.mark.parametrize("claim_class", ["RERUN", "INDEPENDENT-REPLICATION"])
def test_rerun_and_replication_require_explicit_result_provenance(claim_class):
    data = claim_with_class(claim_class)
    with pytest.raises(ValueError, match="explicit structured result_provenance"):
        ClaimRecord.from_dict(data)


@pytest.mark.parametrize(
    ("claim_class", "modality"),
    [
        ("RERUN", "NOT_APPLICABLE"),
        ("RERUN", "SOURCE_REPORTED"),
        ("INDEPENDENT-REPLICATION", "NOT_APPLICABLE"),
        ("INDEPENDENT-REPLICATION", "SOURCE_REPORTED"),
    ],
)
def test_rerun_and_replication_reject_non_result_modalities(claim_class, modality):
    data = claim_with_class(claim_class, asserted_modality=modality)
    with pytest.raises(ValueError, match=f"{claim_class}.*modality"):
        ClaimRecord.from_dict(data)


@pytest.mark.parametrize(
    ("claim_class", "modality"),
    [
        ("RERUN", "SIMULATED"),
        ("RERUN", "ANALYTICAL"),
        ("RERUN", "ACTUAL_MEASUREMENT"),
        ("INDEPENDENT-REPLICATION", "MODEL_ESTIMATED"),
        ("INDEPENDENT-REPLICATION", "ACTUAL_HARDWARE"),
    ],
)
def test_rerun_and_replication_accept_explicit_result_path_modality(
    claim_class, modality
):
    data = claim_with_class(claim_class, asserted_modality=modality)
    ClaimRecord.from_dict(data)


@pytest.mark.parametrize(
    "claim_class",
    ["SOURCE-SUMMARY", "VENDOR-BEHAVIOR", "SYNTHESIS", "HYPOTHESIS"],
)
def test_non_result_legacy_claims_default_to_not_applicable(claim_class):
    data = claim_with_class(claim_class)
    record = ClaimRecord.from_dict(data)
    assert record.result_provenance.asserted_modality is ResultModality.NOT_APPLICABLE


@pytest.mark.parametrize(
    "claim_class",
    ["SOURCE-SUMMARY", "VENDOR-BEHAVIOR", "SYNTHESIS", "HYPOTHESIS"],
)
@pytest.mark.parametrize(
    "result_signal",
    ["result_ids", "requires_result_block", "minimum_independent_result_paths"],
)
def test_result_bearing_legacy_claim_requires_explicit_provenance(
    claim_class,
    result_signal,
):
    data = claim_with_class(claim_class)
    if result_signal == "result_ids":
        data["result_ids"] = ["RES-STC-0001"]
    else:
        data["evidence_requirement"][result_signal] = (
            True if result_signal == "requires_result_block" else 1
        )

    with pytest.raises(ValueError, match="explicit structured result_provenance"):
        ClaimRecord.from_dict(data)


@pytest.mark.parametrize("claim_class", ["SOURCE-SUMMARY", "VENDOR-BEHAVIOR"])
def test_source_claim_with_result_accepts_explicit_source_reported_provenance(
    claim_class,
):
    data = claim_with_class(
        claim_class,
        asserted_modality="SOURCE_REPORTED",
    )
    data["result_ids"] = ["RES-STC-0001"]
    ClaimRecord.from_dict(data)


def test_not_applicable_provenance_rejects_result_id_from_dict():
    data = claim_with_class(
        "SOURCE-SUMMARY",
        asserted_modality="NOT_APPLICABLE",
    )
    data["result_ids"] = ["RES-STC-0001"]
    with pytest.raises(ValueError, match="NOT_APPLICABLE.*result"):
        ClaimRecord.from_dict(data)


@pytest.mark.parametrize(
    "requirement_changes",
    [
        {},
        {"requires_result_block": True},
        {"minimum_independent_result_paths": 1},
    ],
)
def test_not_applicable_provenance_rejects_result_block_on_direct_replace(
    requirement_changes,
):
    record = ClaimRecord.from_dict(claim_dict())
    requirement = replace(record.evidence_requirement, **requirement_changes)
    with pytest.raises(ValueError, match="NOT_APPLICABLE.*result"):
        replace(
            record,
            result_ids=("RES-STC-0001",),
            evidence_requirement=requirement,
        )


def test_not_applicable_provenance_rejects_calibration_dependency():
    data = claim_with_class(
        "SOURCE-SUMMARY",
        asserted_modality="NOT_APPLICABLE",
    )
    data["result_provenance"]["calibration_modalities"] = ["ACTUAL_HARDWARE"]
    with pytest.raises(ValueError, match="NOT_APPLICABLE.*result"):
        ClaimRecord.from_dict(data)


def test_headline_path_does_not_bypass_not_applicable_result_validation():
    data = claim_with_class(
        "SOURCE-SUMMARY",
        asserted_modality="NOT_APPLICABLE",
    )
    data["result_ids"] = ["RES-STC-0001"]
    data["headline_quantitative"] = True
    data["support_refs"] = [support_dict(reproduction_strength="r2")]
    data["evidence_requirement"] = requirement_dict(
        claim_class="SOURCE-SUMMARY",
        minimum_reproduction_strength="r2",
    )
    with pytest.raises(ValueError, match="NOT_APPLICABLE.*result"):
        ClaimRecord.from_dict(data)


def test_result_bearing_analytic_derivation_requires_analytical_modality():
    data = claim_with_class("ANALYTIC-DERIVATION")
    data["result_ids"] = ["RES-STC-0001"]
    with pytest.raises(ValueError, match="explicit structured result_provenance"):
        ClaimRecord.from_dict(data)

    data["result_provenance"] = result_provenance_dict("ANALYTICAL")
    ClaimRecord.from_dict(data)


def test_non_result_analytic_derivation_requires_explicit_provenance():
    data = claim_with_class("ANALYTIC-DERIVATION")
    with pytest.raises(ValueError, match="explicit structured result_provenance"):
        ClaimRecord.from_dict(data)

    data["result_provenance"] = result_provenance_dict("NOT_APPLICABLE")
    ClaimRecord.from_dict(data)


@pytest.mark.parametrize(
    "modality",
    ["NOT_APPLICABLE", "SIMULATED", "ACTUAL_MEASUREMENT"],
)
def test_result_bearing_analytic_derivation_rejects_non_analytical_modality(
    modality,
):
    data = claim_with_class(
        "ANALYTIC-DERIVATION",
        asserted_modality=modality,
    )
    data["result_ids"] = ["RES-STC-0001"]
    with pytest.raises(ValueError, match="ANALYTIC-DERIVATION.*modality"):
        ClaimRecord.from_dict(data)


def test_headline_numeric_original_measurement_cannot_use_simulated_modality():
    data = claim_with_class(
        "ORIGINAL-MEASUREMENT",
        asserted_modality="SIMULATED",
    )
    data["headline_quantitative"] = True
    data["statement"] = "The measured p99 latency is 12 ms."
    data["support_refs"] = [support_dict(warrant="MEAS", reproduction_strength="r2")]
    data["evidence_requirement"] = requirement_dict(
        claim_class="ORIGINAL-MEASUREMENT",
        warrants=["MEAS"],
        minimum_reproduction_strength="r2",
    )
    with pytest.raises(ValueError, match="ORIGINAL-MEASUREMENT.*modality"):
        ClaimRecord.from_dict(data)


def test_trace_simulation_cannot_claim_measured_hardware_latency():
    data = trace_simulation_claim(
        "We measured A100 hardware p99 latency at 12 ms.",
        asserted_modality="ACTUAL_HARDWARE",
    )
    with pytest.raises(
        ValueError,
        match="simulated, model-estimated, or analytical",
    ):
        ClaimRecord.from_dict(data)


def trace_simulation_claim(
    statement: str,
    *,
    asserted_modality: str = "SIMULATED",
    calibration_modalities: list[str] | None = None,
) -> dict[str, object]:
    data = claim_dict()
    data["claim_class"] = "TRACE-SIMULATION"
    data["statement"] = statement
    data["result_provenance"] = result_provenance_dict(
        asserted_modality,
        calibration_modalities,
    )
    data["support_refs"] = [support_dict(warrant="DERIV")]
    data["evidence_requirement"] = requirement_dict(
        claim_class="TRACE-SIMULATION", warrants=["DERIV"]
    )
    return data


def test_trace_simulation_rejects_physical_trial_result_modality():
    data = trace_simulation_claim(
        "The simulation configuration was documented; physical A100 trials "
        "returned p99 latency of 12 ms.",
        asserted_modality="ACTUAL_HARDWARE",
    )
    with pytest.raises(
        ValueError,
        match="simulated, model-estimated, or analytical",
    ):
        ClaimRecord.from_dict(data)


def test_trace_simulation_missing_structured_provenance_fails_closed():
    data = trace_simulation_claim("A typed trace result.")
    del data["result_provenance"]
    with pytest.raises(ValueError, match="structured result_provenance"):
        ClaimRecord.from_dict(data)


def test_trace_simulation_accepts_model_estimate_without_simulation_keyword():
    data = trace_simulation_claim(
        "A100 p99 latency was 12 ms in our discrete-event model.",
        asserted_modality="MODEL_ESTIMATED",
    )
    ClaimRecord.from_dict(data)


def test_trace_simulation_accepts_analytical_result():
    data = trace_simulation_claim(
        "The closed-form queueing derivation yields a 12 ms p99 bound.",
        asserted_modality="ANALYTICAL",
    )
    ClaimRecord.from_dict(data)


@pytest.mark.parametrize(
    "actual_modality",
    ["ACTUAL_MEASUREMENT", "ACTUAL_HARDWARE"],
)
def test_trace_simulation_rejects_actual_result_modalities(actual_modality):
    data = trace_simulation_claim(
        "The physical trial returned a 12 ms p99 latency.",
        asserted_modality=actual_modality,
    )
    with pytest.raises(ValueError, match="TRACE-SIMULATION"):
        ClaimRecord.from_dict(data)


def test_trace_simulation_accepts_real_calibration_for_simulated_result():
    data = trace_simulation_claim(
        "The simulation was calibrated against actual A100 measurements; "
        "simulated p99 latency is 12 ms.",
        asserted_modality="SIMULATED",
        calibration_modalities=["ACTUAL_HARDWARE"],
    )
    record = ClaimRecord.from_dict(data)
    assert record.result_provenance.to_dict() == {
        "asserted_modality": "SIMULATED",
        "calibration_modalities": ["ACTUAL_HARDWARE"],
    }


@pytest.mark.parametrize(
    "result_provenance",
    [
        {"asserted_modality": "UNKNOWN", "calibration_modalities": []},
        {"calibration_modalities": []},
        {
            "asserted_modality": "SIMULATED",
            "calibration_modalities": ["UNKNOWN"],
        },
        {
            "asserted_modality": "SIMULATED",
            "calibration_modalities": [{}],
        },
    ],
)
def test_trace_simulation_rejects_malformed_structured_provenance(
    result_provenance,
):
    data = trace_simulation_claim("A typed trace result.")
    data["result_provenance"] = result_provenance
    with pytest.raises((TypeError, ValueError), match="modalit|fields"):
        ClaimRecord.from_dict(data)


def test_trace_result_provenance_round_trips_canonical_dict():
    data = trace_simulation_claim(
        "A100 p99 latency was 12 ms in our discrete-event model.",
        asserted_modality="MODEL_ESTIMATED",
        calibration_modalities=["ACTUAL_HARDWARE"],
    )
    record = ClaimRecord.from_dict(data)
    assert ClaimRecord.from_dict(record.to_dict()).to_dict() == record.to_dict()


def test_claim_rejects_required_nested_subclass_and_canonical_round_trip_is_exact():
    spoofed = _SpoofedClaimResultProvenance.from_dict(
        {
            "asserted_modality": "ANALYTICAL",
            "calibration_modalities": [],
        }
    )
    data = trace_simulation_claim(
        "A closed-form trace analysis.",
        asserted_modality="ANALYTICAL",
    )
    data["result_provenance"] = spoofed
    with pytest.raises(TypeError, match="result_provenance.*ClaimResultProvenance"):
        ClaimRecord.from_dict(data)

    record = ClaimRecord.from_dict(
        trace_simulation_claim(
            "A closed-form trace analysis.",
            asserted_modality="ANALYTICAL",
        )
    )
    with pytest.raises(TypeError, match="result_provenance.*ClaimResultProvenance"):
        replace(record, result_provenance=spoofed)

    payload = record.to_dict()
    assert payload["result_provenance"]["asserted_modality"] == "ANALYTICAL"
    round_trip = ClaimRecord.from_dict(payload)
    assert round_trip.result_provenance.asserted_modality is ResultModality.ANALYTICAL


def test_trace_result_provenance_detaches_input_and_is_immutable():
    calibrations = ["ACTUAL_HARDWARE"]
    data = trace_simulation_claim(
        "A typed simulated output.",
        calibration_modalities=calibrations,
    )
    record = ClaimRecord.from_dict(data)
    calibrations.append("ANALYTICAL")
    assert record.result_provenance.calibration_modalities == (
        ResultModality.ACTUAL_HARDWARE,
    )
    with pytest.raises(FrozenInstanceError):
        record.result_provenance.asserted_modality = ResultModality.ACTUAL_HARDWARE


def test_direct_trace_claim_rejects_raw_or_actual_result_provenance():
    record = ClaimRecord.from_dict(trace_simulation_claim("A typed simulated output."))
    with pytest.raises(TypeError, match="result_provenance"):
        replace(
            record,
            result_provenance={
                "asserted_modality": "SIMULATED",
                "calibration_modalities": [],
            },
        )
    actual = ClaimResultProvenance.from_dict(
        {
            "asserted_modality": "ACTUAL_HARDWARE",
            "calibration_modalities": [],
        }
    )
    with pytest.raises(
        ValueError,
        match="simulated, model-estimated, or analytical",
    ):
        replace(record, result_provenance=actual)


@pytest.mark.parametrize("relation", ["CONTRADICTS", "QUALIFIES", "NON_CLAIM"])
def test_release_support_requires_supports_relation(relation):
    data = claim_dict()
    support = support_dict()
    support["relation"] = relation
    data["support_refs"] = [support]
    with pytest.raises(ValueError, match="RELEASE_SUPPORT.*SUPPORTS"):
        ClaimRecord.from_dict(data)


def test_context_support_cannot_fill_release_threshold():
    data = claim_dict()
    support = support_dict()
    support["role"] = "CONTEXT"
    data["support_refs"] = [support]
    with pytest.raises(ValueError, match="independent source"):
        ClaimRecord.from_dict(data)


def test_claim_indexes_must_equal_typed_support_refs():
    data = claim_dict()
    data["source_ids"] = ["SRC-STC-0001", "SRC-STC-0002"]
    with pytest.raises(ValueError, match="derived indexes"):
        ClaimRecord.from_dict(data)


def test_release_support_must_satisfy_typed_requirement():
    data = claim_dict()
    data["evidence_requirement"] = requirement_dict(source_grades=["B"])
    with pytest.raises(ValueError, match="evidence requirement"):
        ClaimRecord.from_dict(data)


def test_two_versions_of_one_source_are_one_independent_source():
    data = claim_dict()
    data["evidence_ids"] = ["EV-STC-00001", "EV-STC-00002"]
    data["support_refs"] = [
        support_dict(evidence_number=1, source_version="v1"),
        support_dict(evidence_number=2, source_version="v2"),
    ]
    data["evidence_requirement"] = requirement_dict(minimum_independent_sources=2)
    with pytest.raises(ValueError, match="independent source"):
        ClaimRecord.from_dict(data)


def counterevidence_claim() -> dict[str, object]:
    data = claim_dict()
    support = support_dict()
    support["role"] = "COUNTEREVIDENCE"
    support["relation"] = "CONTRADICTS"
    data["support_refs"] = [support]
    data["evidence_requirement"] = requirement_dict(minimum_independent_sources=0)
    return data


def test_counterevidence_ids_must_equal_typed_counterevidence_refs():
    data = counterevidence_claim()
    data["counterevidence_ids"] = []
    with pytest.raises(ValueError, match="counterevidence_ids"):
        ClaimRecord.from_dict(data)


def test_counterevidence_ids_follow_first_typed_ref_order():
    data = counterevidence_claim()
    second = support_dict(evidence_number=2)
    second["role"] = "COUNTEREVIDENCE"
    second["relation"] = "QUALIFIES"
    data["support_refs"] = [second, data["support_refs"][0]]
    data["evidence_ids"] = ["EV-STC-00002", "EV-STC-00001"]
    data["counterevidence_ids"] = ["EV-STC-00001", "EV-STC-00002"]
    with pytest.raises(ValueError, match="counterevidence_ids"):
        ClaimRecord.from_dict(data)


def test_exact_counterevidence_index_is_valid_but_not_release_support():
    data = counterevidence_claim()
    data["counterevidence_ids"] = ["EV-STC-00001"]
    ClaimRecord.from_dict(data)


def test_question_hypothesis_and_artifact_node_round_trip():
    question = QuestionRecord.from_dict(
        {
            "question_id": "RQ1",
            "statement": "Is sleep a computational phase?",
            "estimand": "sleep operator effect",
            "null": "no operator effect",
            "rejection_rule": "reject when the interval excludes zero",
            "minimum_effect": "0.02 utility units",
            "inconclusive_condition": "interval spans both thresholds",
            "validity_failure": "wake/sleep information cutoff differs",
            "owner": "paper-a",
            "artifact_dependencies": ["ART-STC-0001"],
            "paper_scope": "primary",
        }
    )
    hypothesis = HypothesisRecord.from_dict(
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
        }
    )
    node = ArtifactNode.from_dict(
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
        }
    )
    assert question.to_dict()["question_id"] == "RQ1"
    assert hypothesis.to_dict()["confirmatory_cells"] == ["reuse-high/pressure-high"]
    assert node.to_dict()["input_digests"] == [SHA_A]


def test_gate_evaluation_attestation_is_available_for_downstream_gate_models():
    attestation = GateEvaluationAttestation.from_dict(
        {
            "attestation_id": "ATT-G8-001",
            "gate_id": "G8",
            "subject_sha256": SHA_C,
            "subject_refs": [
                {"path": "reports/release-smoke.json", "sha256": SHA_B},
                {"path": "manifests/g8-subject.json", "sha256": SHA_A},
            ],
            "signer_id": "release-evaluator-1",
            "signer_role": "release_evaluator",
            "independence_mode": "fresh-context",
            "algorithm": "Ed25519",
            "key_id": "release-key-1",
            "signature": "base64-signature",
            "signed_at": "2026-07-25T12:00:00+09:00",
        }
    )
    assert attestation.to_dict()["gate_id"] == "G8"
    assert [ref["path"] for ref in attestation.to_dict()["subject_refs"]] == [
        "manifests/g8-subject.json",
        "reports/release-smoke.json",
    ]


def test_trusted_key_set_is_typed_deeply_immutable_and_canonical():
    TrustedKeySet = models.TrustedKeySet
    payload = trusted_key_set_dict()

    record = TrustedKeySet.from_dict(payload)

    assert record.to_dict() == payload
    assert isinstance(record.keys, tuple)
    assert isinstance(record.keys[0].roles, tuple)
    assert [key.key_id for key in record.keys] == [
        "release-key-z",
        "audit-key-a",
    ]
    assert record.keys[0].roles == (
        "visual_reviewer",
        "release_evaluator",
    )
    assert record.keys[1].roles == (
        "quality_reviewer",
        "independent_auditor",
    )
    assert record.keys[1].valid_until is None
    with pytest.raises(FrozenInstanceError):
        record.keys[0].signer_id = "changed"  # type: ignore[misc]

    revoked_payload = trusted_key_set_dict()
    revoked_keys = revoked_payload["keys"]
    assert isinstance(revoked_keys, list)
    revoked_key = revoked_keys[1]
    assert isinstance(revoked_key, dict)
    revoked_key["revoked"] = True
    _refresh_trusted_key_set_digest(revoked_payload)
    revoked_record = TrustedKeySet.from_dict(revoked_payload)
    assert revoked_record.keys[1].revoked is True
    assert revoked_record.to_dict() == revoked_payload


@pytest.mark.parametrize(
    ("invalid_case", "expected_error"),
    [
        ("duplicate-key-id", "key_id.*unique"),
        ("duplicate-public-key", "public_key.*unique"),
        ("duplicate-role", "roles.*unique"),
        ("set-digest-mismatch", "set_digest"),
        ("naive-valid-from", "timezone-aware"),
        ("naive-issued-at", "timezone-aware"),
        ("naive-valid-until", "timezone-aware"),
        ("equal-validity", "valid_until.*after"),
        ("reversed-validity", "valid_until.*after"),
    ],
)
def test_trusted_key_set_rejects_ambiguous_or_noncanonical_trust_material(
    invalid_case,
    expected_error,
):
    TrustedKeySet = models.TrustedKeySet
    payload = trusted_key_set_dict()
    keys = payload["keys"]
    assert isinstance(keys, list)
    key = keys[0]
    assert isinstance(key, dict)

    if invalid_case == "duplicate-key-id":
        keys.append({**key, "signer_id": "second-signer"})
        _refresh_trusted_key_set_digest(payload)
    elif invalid_case == "duplicate-public-key":
        keys.append(
            {
                **key,
                "key_id": "release-key-2",
                "signer_id": "second-signer",
            }
        )
        _refresh_trusted_key_set_digest(payload)
    elif invalid_case == "duplicate-role":
        key["roles"] = ["release_evaluator", "release_evaluator"]
        _refresh_trusted_key_set_digest(payload)
    elif invalid_case == "set-digest-mismatch":
        payload["set_digest"] = SHA_A
    elif invalid_case == "naive-valid-from":
        key["valid_from"] = "2026-07-01T00:00:00"
        _refresh_trusted_key_set_digest(payload)
    elif invalid_case == "naive-issued-at":
        payload["issued_at"] = "2026-07-01T00:00:00"
        _refresh_trusted_key_set_digest(payload)
    elif invalid_case == "naive-valid-until":
        key["valid_until"] = "2027-07-01T00:00:00"
        _refresh_trusted_key_set_digest(payload)
    elif invalid_case == "equal-validity":
        key["valid_until"] = key["valid_from"]
        _refresh_trusted_key_set_digest(payload)
    else:
        key["valid_until"] = "2026-06-30T23:59:59Z"
        _refresh_trusted_key_set_digest(payload)

    with pytest.raises((TypeError, ValueError), match=expected_error):
        TrustedKeySet.from_dict(payload)


def test_attestation_subject_ref_rejects_path_traversal():
    data = {
        "attestation_id": "ATT-G8-001",
        "gate_id": "G8",
        "subject_sha256": SHA_C,
        "subject_refs": [{"path": "../outside.json", "sha256": SHA_A}],
        "signer_id": "release-evaluator-1",
        "signer_role": "release_evaluator",
        "independence_mode": "fresh-context",
        "algorithm": "Ed25519",
        "key_id": "release-key-1",
        "signature": "base64-signature",
        "signed_at": "2026-07-25T12:00:00+09:00",
    }
    with pytest.raises(ValueError, match="package-root-relative"):
        GateEvaluationAttestation.from_dict(data)


@pytest.mark.parametrize(
    "signed_at",
    [
        "2026-07-25T12:00:00",
        "20260725T120000+09:00",
        "2026-W30-6T12:00:00+09:00",
    ],
)
def test_gate_attestation_requires_canonical_timezone_aware_signed_at(signed_at):
    data = {
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
        "signed_at": signed_at,
    }
    with pytest.raises(ValueError, match="timezone-aware"):
        GateEvaluationAttestation.from_dict(data)


@pytest.mark.parametrize(
    "signed_at", ["2026-07-25T03:00:00Z", "2026-07-25T03:00:00+00:00"]
)
def test_gate_attestation_normalizes_utc_to_z(signed_at):
    data = {
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
        "signed_at": signed_at,
    }
    assert GateEvaluationAttestation.from_dict(data).signed_at == (
        "2026-07-25T03:00:00Z"
    )


def human_approval_dict() -> dict[str, object]:
    return {
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
    }


@pytest.mark.parametrize(
    "review_checks",
    [
        {"layout": ["PASS"]},
        {"layout": 1},
        {"": "PASS"},
        {"layout": ""},
        {1: "PASS"},
    ],
)
def test_human_approval_review_checks_are_exact_nonempty_strings(review_checks):
    data = human_approval_dict()
    data["review_checks"] = review_checks
    with pytest.raises((TypeError, ValueError), match="review_checks"):
        HumanApproval.from_dict(data)


def test_human_approval_review_checks_are_deeply_immutable():
    data = human_approval_dict()
    checks = data["review_checks"]
    approval = HumanApproval.from_dict(data)
    checks["layout"] = "FAIL"
    assert approval.review_checks["layout"] == "PASS"
    with pytest.raises(TypeError):
        approval.review_checks["layout"] = "FAIL"


def test_direct_human_approval_snapshots_mapping_proxy_backing_dict():
    approval = HumanApproval.from_dict(human_approval_dict())
    backing = {"layout": "PASS"}
    replaced = replace(approval, review_checks=MappingProxyType(backing))
    backing["layout"] = "FAIL"
    assert replaced.review_checks["layout"] == "PASS"
    assert replaced.to_dict()["review_checks"] == {"layout": "PASS"}


def test_direct_human_approval_snapshots_plain_mapping():
    approval = HumanApproval.from_dict(human_approval_dict())
    backing = {"layout": "PASS"}
    replaced = replace(approval, review_checks=backing)
    backing["layout"] = "FAIL"
    assert replaced.review_checks["layout"] == "PASS"


def hypothesis_dict() -> dict[str, object]:
    return {
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
    }


def test_direct_hypothesis_snapshots_nested_amendment_mapping():
    hypothesis = HypothesisRecord.from_dict(hypothesis_dict())
    nested_notes = ["initial"]
    backing = {"notes": nested_notes}
    replaced = replace(
        hypothesis,
        amendment_history=(MappingProxyType(backing),),
    )
    nested_notes.append("mutated")
    backing["new"] = "late"
    assert replaced.to_dict()["amendment_history"] == [{"notes": ["initial"]}]


def test_hypothesis_from_dict_rejects_deque_in_amendment_history():
    data = hypothesis_dict()
    data["amendment_history"] = [{"payload": deque(["not", "json"])}]
    with pytest.raises(TypeError, match="canonical JSON"):
        HypothesisRecord.from_dict(data)


def test_direct_hypothesis_rejects_deque_in_amendment_history():
    hypothesis = HypothesisRecord.from_dict(hypothesis_dict())
    with pytest.raises(TypeError, match="canonical JSON"):
        replace(
            hypothesis,
            amendment_history=(MappingProxyType({"payload": deque(["not", "json"])}),),
        )


@pytest.mark.parametrize("direct", [False, True])
def test_hypothesis_rejects_arbitrary_object_in_amendment_history(direct):
    amendment = (
        MappingProxyType({"payload": object()}) if direct else {"payload": object()}
    )
    if direct:
        hypothesis = HypothesisRecord.from_dict(hypothesis_dict())
        with pytest.raises(TypeError, match="canonical JSON"):
            replace(hypothesis, amendment_history=(amendment,))
        return

    data = hypothesis_dict()
    data["amendment_history"] = [amendment]
    with pytest.raises(TypeError, match="canonical JSON"):
        HypothesisRecord.from_dict(data)


def test_hypothesis_rejects_non_string_nested_json_key():
    data = hypothesis_dict()
    data["amendment_history"] = [{"payload": {1: "not-json-object"}}]
    with pytest.raises(TypeError, match="keys must be strings"):
        HypothesisRecord.from_dict(data)


@pytest.mark.parametrize("invalid_number", [float("nan"), float("inf"), float("-inf")])
def test_hypothesis_from_dict_rejects_non_finite_json_number(invalid_number):
    data = hypothesis_dict()
    data["amendment_history"] = [{"value": invalid_number}]
    with pytest.raises(ValueError, match="finite"):
        HypothesisRecord.from_dict(data)


@pytest.mark.parametrize("invalid_number", [float("nan"), float("inf"), float("-inf")])
def test_direct_hypothesis_rejects_non_finite_json_number(invalid_number):
    hypothesis = HypothesisRecord.from_dict(hypothesis_dict())
    with pytest.raises(ValueError, match="finite"):
        replace(
            hypothesis,
            amendment_history=(MappingProxyType({"value": invalid_number}),),
        )


def test_hypothesis_json_payload_is_detached_and_serializes_canonically():
    nested_values = [1, 2.5, True, None, {"label": ["original"]}]
    amendment = {"payload": nested_values}
    data = hypothesis_dict()
    data["amendment_history"] = [amendment]

    hypothesis = HypothesisRecord.from_dict(data)
    nested_values.append("late")
    amendment["late"] = "mutation"

    payload = hypothesis.to_dict()
    assert payload["amendment_history"] == [
        {"payload": [1, 2.5, True, None, {"label": ["original"]}]}
    ]
    encoded = json.dumps(
        payload,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert json.loads(encoded) == payload


class _MalformedPayloadRecord(models._Record):
    def to_dict(self):
        return {"payload": object()}


@pytest.mark.parametrize(
    "payload_record",
    [models._Record(), _MalformedPayloadRecord()],
    ids=["base-record", "malformed-subclass"],
)
@pytest.mark.parametrize("direct", [False, True], ids=["from-dict", "direct"])
def test_hypothesis_json_payload_rejects_untyped_record_escape(
    payload_record,
    direct,
):
    amendment = {"payload": payload_record}
    if direct:
        hypothesis = HypothesisRecord.from_dict(hypothesis_dict())
        with pytest.raises(TypeError, match="canonical JSON"):
            replace(
                hypothesis,
                amendment_history=(MappingProxyType(amendment),),
            )
        return

    data = hypothesis_dict()
    data["amendment_history"] = [amendment]
    with pytest.raises(TypeError, match="canonical JSON"):
        HypothesisRecord.from_dict(data)


def test_successful_mapping_payload_records_serialize_as_strict_json():
    records = [
        HypothesisRecord.from_dict(hypothesis_dict()),
        HumanApproval.from_dict(human_approval_dict()),
    ]
    for record in records:
        json.dumps(record.to_dict(), allow_nan=False)


def gate_record_dict(gate_id: str) -> dict[str, object]:
    return {
        "gate_id": gate_id,
        "gate_artifact_id": f"ART-STC-{9901 + int(gate_id[1:]):04d}",
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
    }


def test_g8_requires_candidate_and_evaluator_attestation():
    with pytest.raises(ValueError, match="G7/G8"):
        GateRecord.from_dict(gate_record_dict("G8"))


def test_pre_g7_gate_rejects_scientific_candidate_fields():
    data = gate_record_dict("G6")
    data["scientific_candidate_sha"] = "e" * 40
    data["scientific_candidate_tree_digest"] = SHA_A
    with pytest.raises(ValueError, match="before G7"):
        GateRecord.from_dict(data)


@pytest.mark.parametrize("duplicate_field", ["finding_ids", "adjudication_ids"])
def test_gate_record_rejects_duplicate_audit_identifiers(duplicate_field):
    data = gate_record_dict("G6")
    data[duplicate_field] = ["AUDIT-001", "AUDIT-001"]

    with pytest.raises(ValueError, match=f"{duplicate_field}.*unique"):
        GateRecord.from_dict(data)


def test_g8_with_candidate_and_attestation_ref_is_valid():
    data = gate_record_dict("G8")
    data["scientific_candidate_sha"] = "e" * 40
    data["scientific_candidate_tree_digest"] = SHA_A
    data["evaluator_attestation_ref"] = {
        "artifact_id": "ART-STC-0001",
        "sha256": SHA_B,
    }
    GateRecord.from_dict(data)


def test_gate_rejects_optional_nested_digest_ref_subclass():
    class SpoofedDigestRef(models.DigestRef):
        __slots__ = ()

        def to_dict(self):
            return {"artifact_id": "ART-STC-9999", "sha256": SHA_A}

    spoofed = SpoofedDigestRef.from_dict(
        {"artifact_id": "ART-STC-0001", "sha256": SHA_B}
    )
    data = gate_record_dict("G8")
    data["scientific_candidate_sha"] = "e" * 40
    data["scientific_candidate_tree_digest"] = SHA_A
    data["evaluator_attestation_ref"] = spoofed
    with pytest.raises(TypeError, match="evaluator_attestation_ref.*DigestRef"):
        GateRecord.from_dict(data)

    valid = gate_record_dict("G8")
    valid["scientific_candidate_sha"] = "e" * 40
    valid["scientific_candidate_tree_digest"] = SHA_A
    valid["evaluator_attestation_ref"] = {
        "artifact_id": "ART-STC-0001",
        "sha256": SHA_B,
    }
    record = GateRecord.from_dict(valid)
    with pytest.raises(TypeError, match="evaluator_attestation_ref.*DigestRef"):
        replace(record, evaluator_attestation_ref=spoofed)


@pytest.mark.parametrize(
    "evaluation_commit_sha",
    ["g" * 40, "A" * 40, "a" * 39, "a" * 64, "0" * 40],
)
def test_gate_rejects_unsupported_evaluation_git_object_id(
    evaluation_commit_sha,
):
    data = gate_record_dict("G6")
    data["evaluation_commit_sha"] = evaluation_commit_sha
    with pytest.raises(ValueError, match="evaluation_commit_sha.*Git SHA-1"):
        GateRecord.from_dict(data)


@pytest.mark.parametrize(
    "scientific_candidate_sha",
    ["candidate-main", "F" * 40, "f" * 39, "f" * 64, "0" * 40],
)
def test_gate_rejects_unsupported_candidate_git_object_id(
    scientific_candidate_sha,
):
    data = gate_record_dict("G8")
    data["scientific_candidate_sha"] = scientific_candidate_sha
    data["scientific_candidate_tree_digest"] = SHA_A
    data["evaluator_attestation_ref"] = {
        "artifact_id": "ART-STC-0001",
        "sha256": SHA_B,
    }
    with pytest.raises(ValueError, match="scientific_candidate_sha.*Git SHA-1"):
        GateRecord.from_dict(data)


def paper_card_dict() -> dict[str, object]:
    return {
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
    }


def test_paper_card_rejects_empty_meaningful_text():
    data = paper_card_dict()
    data["optimizer"] = ""
    with pytest.raises(ValueError, match="optimizer"):
        PaperCard.from_dict(data)


def test_paper_card_rejects_nested_or_empty_tuple_text():
    data = paper_card_dict()
    data["training_data"] = [["mutable"]]
    with pytest.raises(TypeError, match="training_data"):
        PaperCard.from_dict(data)

    data = paper_card_dict()
    data["training_data"] = [""]
    with pytest.raises(ValueError, match="training_data"):
        PaperCard.from_dict(data)
