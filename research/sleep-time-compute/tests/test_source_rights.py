from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from datetime import date
from pathlib import Path

import pytest

from stc_research.models import SourceRecord

SHA_A = "a" * 64
SHA_B = "b" * 64
UNKNOWN_DIGEST = "0" * 64


def _rights_api():
    from stc_research.source_rights import (
        StagedBlob,
        evaluate_staged_blobs,
        scan_staged_index,
    )

    return StagedBlob, evaluate_staged_blobs, scan_staged_index


def _artifact(
    *,
    path: str,
    content: bytes,
    artifact_id: str = "ART-STC-0001",
    redistribution_allowed: bool = True,
    license_id: str = "CC-BY-4.0",
    license_evidence_url: str = "https://example.org/license",
    license_evidence_sha256: str = SHA_B,
    disposition: str = "ALLOW",
    rights_review_expires_at: str | None = None,
    media_type: str = "application/pdf",
) -> dict[str, object]:
    return {
        "artifact_id": artifact_id,
        "canonical_url": "https://example.org/paper.pdf",
        "local_path": path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "media_type": media_type,
        "availability": "VENDORED",
        "license_id": license_id,
        "license_evidence_url": license_evidence_url,
        "license_evidence_sha256": license_evidence_sha256,
        "redistribution_allowed": redistribution_allowed,
        "license_review_disposition": disposition,
        "rights_reviewer": "rights-reviewer-1",
        "rights_reviewed_at": "2026-07-20",
        "rights_review_expires_at": rights_review_expires_at,
    }


def _source(
    artifacts: list[dict[str, object]],
    *,
    source_id: str = "SRC-STC-0001",
    legacy_id: str = "W1-001",
    canonical_key: str = "doi:10.1000/rights-work:v1",
    citation_key: str = "author2025rights",
    rights_summary: str = "Reuse allowed by the source-level summary.",
) -> SourceRecord:
    source = {
        "source_id": source_id,
        "legacy_ids": [legacy_id],
        "canonical_key": canonical_key,
        "title": "A Rights-Controlled Source",
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
                "local_artifact_id": (
                    artifacts[0]["artifact_id"]
                    if artifacts
                    else "ART-STC-0001"
                ),
                "sha256": (
                    artifacts[0]["sha256"] if artifacts else SHA_A
                ),
                "supersedes": None,
                "derived_from": [],
            }
        ],
        "source_grade": "A",
        "review_status": "FULL",
        "urls": ["https://example.org/paper"],
        "local_artifacts": artifacts,
        "citation_key": citation_key,
        "code_records": [],
        "data_records": [],
        "rights_summary": rights_summary,
        "topics": ["sleep-time-compute"],
        "affiliations": ["Example Lab"],
        "inclusion_reason": "Primary source with explicit rights review.",
        "non_claim": "The source-level summary is not an export license.",
        "last_verified": "2026-07-25",
    }
    return SourceRecord.from_dict(source)


def _diagnostic_contract(
    report: object,
) -> tuple[tuple[str, str, str | None, str | None, str], ...]:
    return tuple(
        (
            diagnostic.code,
            diagnostic.path,
            diagnostic.source_id,
            diagnostic.artifact_id,
            diagnostic.message,
        )
        for diagnostic in report.diagnostics
    )


def test_disallowed_artifact_cannot_stage_full_text() -> None:
    StagedBlob, evaluate_staged_blobs, _ = _rights_api()
    content = b"extension-identified redistribution-controlled source\n"
    artifact = _artifact(
        path="papers/paper.pdf",
        content=content,
        redistribution_allowed=False,
        disposition="DENY",
        media_type="application/octet-stream",
    )

    report = evaluate_staged_blobs(
        [_source([artifact])],
        [StagedBlob(path="papers/paper.pdf", content=content)],
        as_of=date(2026, 7, 25),
    )

    assert _diagnostic_contract(report) == (
        (
            "REDISTRIBUTION_DISALLOWED",
            "papers/paper.pdf",
            "SRC-STC-0001",
            "ART-STC-0001",
            (
                "artifact ART-STC-0001 for source SRC-STC-0001 does not "
                "permit redistribution of staged full text papers/paper.pdf"
            ),
        ),
    )


def test_reviewed_artifact_with_exact_digest_allows_full_text() -> None:
    StagedBlob, evaluate_staged_blobs, _ = _rights_api()
    content = b"%PDF-1.7\npositively reviewed source\n"
    artifact = _artifact(
        path="papers/paper.pdf",
        content=content,
    )

    report = evaluate_staged_blobs(
        [_source([artifact])],
        [StagedBlob(path="papers/paper.pdf", content=content)],
        as_of=date(2026, 7, 25),
    )

    assert report.success
    assert report.diagnostics == ()


@pytest.mark.parametrize(
    ("second_redistribution_allowed", "second_disposition"),
    [(False, "DENY"), (True, "ALLOW")],
    ids=["allow-deny", "allow-allow"],
)
def test_duplicate_artifact_rights_for_one_path_fail_closed(
    second_redistribution_allowed: bool,
    second_disposition: str,
) -> None:
    StagedBlob, evaluate_staged_blobs, _ = _rights_api()
    path = "papers/shared.pdf"
    content = b"%PDF-1.7\none staged object with duplicate rights records\n"
    allowed_artifact = _artifact(
        path=path,
        content=content,
        artifact_id="ART-STC-0001",
    )
    second_artifact = _artifact(
        path=path,
        content=content,
        artifact_id="ART-STC-0002",
        redistribution_allowed=second_redistribution_allowed,
        disposition=second_disposition,
    )
    allowed_source = _source([allowed_artifact])
    second_source = _source(
        [second_artifact],
        source_id="SRC-STC-0002",
        legacy_id="W1-002",
        canonical_key="doi:10.1000/conflicting-rights-work:v1",
        citation_key="author2025conflictingrights",
    )
    expected = (
        (
            "ARTIFACT_RIGHTS_AMBIGUOUS",
            path,
            None,
            None,
            (
                f"staged full text {path} matches multiple artifact-level "
                "rights records: SRC-STC-0001/ART-STC-0001, "
                "SRC-STC-0002/ART-STC-0002"
            ),
        ),
    )

    forward_report = evaluate_staged_blobs(
        [allowed_source, second_source],
        [StagedBlob(path=path, content=content)],
        as_of=date(2026, 7, 25),
    )
    reverse_report = evaluate_staged_blobs(
        [second_source, allowed_source],
        [StagedBlob(path=path, content=content)],
        as_of=date(2026, 7, 25),
    )

    assert _diagnostic_contract(forward_report) == expected
    assert _diagnostic_contract(reverse_report) == expected


def test_artifact_rights_bind_to_exact_staged_digest() -> None:
    StagedBlob, evaluate_staged_blobs, _ = _rights_api()
    authorized_content = b"%PDF-1.7\nauthorized immutable source\n"
    staged_content = b"%PDF-1.7\ndifferent staged source bytes\n"
    artifact = _artifact(
        path="papers/paper.pdf",
        content=authorized_content,
    )
    expected_digest = hashlib.sha256(authorized_content).hexdigest()
    actual_digest = hashlib.sha256(staged_content).hexdigest()
    assert expected_digest != actual_digest

    report = evaluate_staged_blobs(
        [_source([artifact])],
        [StagedBlob(path="papers/paper.pdf", content=staged_content)],
        as_of=date(2026, 7, 25),
    )

    assert _diagnostic_contract(report) == (
        (
            "ARTIFACT_DIGEST_MISMATCH",
            "papers/paper.pdf",
            "SRC-STC-0001",
            "ART-STC-0001",
            (
                f"staged full text papers/paper.pdf digest {actual_digest} "
                f"does not match artifact ART-STC-0001 digest "
                f"{expected_digest}"
            ),
        ),
    )


def test_artifact_media_type_detects_full_text_without_extension_or_magic() -> None:
    StagedBlob, evaluate_staged_blobs, _ = _rights_api()
    content = b"opaque staged payload without PDF magic"
    pdf_artifact = _artifact(
        path="papers/paper.bin",
        content=content,
        redistribution_allowed=False,
        disposition="DENY",
        media_type="application/pdf",
    )

    report = evaluate_staged_blobs(
        [_source([pdf_artifact])],
        [StagedBlob(path="papers/paper.bin", content=content)],
        as_of=date(2026, 7, 25),
    )

    assert _diagnostic_contract(report) == (
        (
            "REDISTRIBUTION_DISALLOWED",
            "papers/paper.bin",
            "SRC-STC-0001",
            "ART-STC-0001",
            (
                "artifact ART-STC-0001 for source SRC-STC-0001 does not "
                "permit redistribution of staged full text papers/paper.bin"
            ),
        ),
    )

    metadata_artifact = _artifact(
        path="papers/paper.bin",
        content=content,
        redistribution_allowed=False,
        disposition="DENY",
        media_type="application/json",
    )
    metadata_report = evaluate_staged_blobs(
        [_source([metadata_artifact])],
        [StagedBlob(path="papers/paper.bin", content=content)],
        as_of=date(2026, 7, 25),
    )
    assert metadata_report.diagnostics == ()


def test_disallowed_plain_text_extract_cannot_be_staged() -> None:
    StagedBlob, evaluate_staged_blobs, _ = _rights_api()
    path = "papers/sleep-time-compute/text/work-v1.txt"
    content = b"generated full-text extract without PDF bytes\n"
    artifact = _artifact(
        path=path,
        content=content,
        redistribution_allowed=False,
        disposition="DENY",
        media_type="text/plain",
    )

    report = evaluate_staged_blobs(
        [_source([artifact])],
        [StagedBlob(path=path, content=content)],
        as_of=date(2026, 7, 25),
    )

    assert _diagnostic_contract(report) == (
        (
            "REDISTRIBUTION_DISALLOWED",
            path,
            "SRC-STC-0001",
            "ART-STC-0001",
            (
                "artifact ART-STC-0001 for source SRC-STC-0001 does not "
                f"permit redistribution of staged full text {path}"
            ),
        ),
    )


def test_allowed_artifact_requires_license_evidence_and_approved_review() -> None:
    StagedBlob, evaluate_staged_blobs, _ = _rights_api()
    content = b"%PDF-1.7\nlicense review pending\n"
    artifact = _artifact(
        path="papers/paper.pdf",
        content=content,
        license_evidence_url="UNKNOWN",
        license_evidence_sha256=UNKNOWN_DIGEST,
        disposition="PENDING",
    )

    report = evaluate_staged_blobs(
        [_source([artifact])],
        [StagedBlob(path="papers/paper.pdf", content=content)],
        as_of=date(2026, 7, 25),
    )

    assert _diagnostic_contract(report) == (
        (
            "LICENSE_EVIDENCE_MISSING",
            "papers/paper.pdf",
            "SRC-STC-0001",
            "ART-STC-0001",
            (
                "artifact ART-STC-0001 lacks authoritative license evidence "
                "for staged full text papers/paper.pdf"
            ),
        ),
        (
            "RIGHTS_REVIEW_NOT_APPROVED",
            "papers/paper.pdf",
            "SRC-STC-0001",
            "ART-STC-0001",
            (
                "artifact ART-STC-0001 rights disposition 'PENDING' is not "
                "ALLOW for staged full text papers/paper.pdf"
            ),
        ),
    )


def test_noassertion_is_not_authoritative_license_evidence() -> None:
    StagedBlob, evaluate_staged_blobs, _ = _rights_api()
    content = b"%PDF-1.7\nlicense identity not asserted\n"
    artifact = _artifact(
        path="papers/paper.pdf",
        content=content,
        license_id="NOASSERTION",
    )

    report = evaluate_staged_blobs(
        [_source([artifact])],
        [StagedBlob(path="papers/paper.pdf", content=content)],
        as_of=date(2026, 7, 25),
    )

    assert _diagnostic_contract(report) == (
        (
            "LICENSE_EVIDENCE_MISSING",
            "papers/paper.pdf",
            "SRC-STC-0001",
            "ART-STC-0001",
            (
                "artifact ART-STC-0001 lacks authoritative license evidence "
                "for staged full text papers/paper.pdf"
            ),
        ),
    )


def test_source_summary_cannot_authorize_unregistered_artifact() -> None:
    StagedBlob, evaluate_staged_blobs, _ = _rights_api()
    content = b"%PDF-1.7\nunregistered sibling full text\n"

    report = evaluate_staged_blobs(
        [_source([], rights_summary="Redistribution allowed for this source.")],
        [StagedBlob(path="papers/unreviewed.pdf", content=content)],
        as_of=date(2026, 7, 25),
    )

    assert _diagnostic_contract(report) == (
        (
            "ARTIFACT_RIGHTS_MISSING",
            "papers/unreviewed.pdf",
            None,
            None,
            (
                "staged full text papers/unreviewed.pdf has no exact "
                "artifact-level rights record; source summaries cannot "
                "authorize it"
            ),
        ),
    )


@pytest.mark.parametrize(
    ("rights_review_expires_at", "as_of", "effective_expiry_date"),
    [
        ("2026-07-24", date(2026, 7, 25), "2026-07-24"),
        (
            "2026-07-26T00:30:00+02:00",
            date(2026, 7, 26),
            "2026-07-25",
        ),
    ],
    ids=["date", "aware-datetime-crossing-utc-date"],
)
def test_expired_artifact_rights_fail_closed(
    rights_review_expires_at: str,
    as_of: date,
    effective_expiry_date: str,
) -> None:
    StagedBlob, evaluate_staged_blobs, _ = _rights_api()
    content = b"%PDF-1.7\nexpired rights review\n"
    artifact = _artifact(
        path="papers/paper.pdf",
        content=content,
        rights_review_expires_at=rights_review_expires_at,
    )

    report = evaluate_staged_blobs(
        [_source([artifact])],
        [StagedBlob(path="papers/paper.pdf", content=content)],
        as_of=as_of,
    )

    assert _diagnostic_contract(report) == (
        (
            "RIGHTS_REVIEW_EXPIRED",
            "papers/paper.pdf",
            "SRC-STC-0001",
            "ART-STC-0001",
            (
                "artifact ART-STC-0001 rights review expired at "
                f"{effective_expiry_date} before scan date "
                f"{as_of.isoformat()}"
            ),
        ),
    )


@pytest.mark.parametrize(
    "rights_reviewed_at",
    ["2026-07-26", "2026-07-25T23:30:00-02:00"],
    ids=["date", "aware-datetime-crossing-utc-date"],
)
def test_future_artifact_rights_review_is_not_yet_valid(
    rights_reviewed_at: str,
) -> None:
    StagedBlob, evaluate_staged_blobs, _ = _rights_api()
    content = b"%PDF-1.7\nfuture-dated rights review\n"
    artifact = _artifact(
        path="papers/paper.pdf",
        content=content,
    )
    artifact["rights_reviewed_at"] = rights_reviewed_at

    report = evaluate_staged_blobs(
        [_source([artifact])],
        [StagedBlob(path="papers/paper.pdf", content=content)],
        as_of=date(2026, 7, 25),
    )

    assert _diagnostic_contract(report) == (
        (
            "RIGHTS_REVIEW_NOT_YET_VALID",
            "papers/paper.pdf",
            "SRC-STC-0001",
            "ART-STC-0001",
            (
                "artifact ART-STC-0001 rights review starts at 2026-07-26 "
                "after scan date 2026-07-25"
            ),
        ),
    )


def test_cache_paths_cannot_be_staged_even_with_positive_rights() -> None:
    StagedBlob, evaluate_staged_blobs, _ = _rights_api()
    content = b"%PDF-1.7\ncache-only source\n"
    path = "research/sleep-time-compute/.cache/stc/sources/paper.pdf"
    artifact = _artifact(path=path, content=content)

    report = evaluate_staged_blobs(
        [_source([artifact])],
        [StagedBlob(path=path, content=content)],
        as_of=date(2026, 7, 25),
    )

    assert _diagnostic_contract(report) == (
        (
            "STAGED_CACHE_PATH",
            path,
            None,
            None,
            f"staged cache path {path} is forbidden",
        ),
    )


def _git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )


def test_staged_scan_reads_index_blobs_not_working_tree(tmp_path: Path) -> None:
    _, _, scan_staged_index = _rights_api()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tests@example.org")
    _git(repo, "config", "user.name", "STC Tests")
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-qm", "fixture root")

    staged_content = b"%PDF-1.7\nstaged controlled source\n"
    staged_artifact = _artifact(
        path="papers/paper.bin",
        content=staged_content,
        redistribution_allowed=False,
        disposition="DENY",
        media_type="application/octet-stream",
    )
    staged_source = _source([staged_artifact]).to_dict()
    source_registry = (
        repo
        / "research"
        / "sleep-time-compute"
        / "registry"
        / "sources.jsonl"
    )
    source_registry.parent.mkdir(parents=True)
    source_registry.write_text(
        json.dumps(staged_source, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    paper = repo / "papers" / "paper.bin"
    paper.parent.mkdir(parents=True)
    paper.write_bytes(staged_content)
    _git(
        repo,
        "add",
        "research/sleep-time-compute/registry/sources.jsonl",
        "papers/paper.bin",
    )

    working_source = deepcopy(staged_source)
    local_artifacts = working_source["local_artifacts"]
    assert isinstance(local_artifacts, list)
    assert isinstance(local_artifacts[0], dict)
    local_artifacts[0]["redistribution_allowed"] = True
    local_artifacts[0]["license_review_disposition"] = "ALLOW"
    source_registry.write_text(
        json.dumps(working_source, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    paper.write_bytes(b"working tree is not the staged PDF")

    report = scan_staged_index(
        repo,
        as_of=date(2026, 7, 25),
    )

    assert _diagnostic_contract(report) == (
        (
            "REDISTRIBUTION_DISALLOWED",
            "papers/paper.bin",
            "SRC-STC-0001",
            "ART-STC-0001",
            (
                "artifact ART-STC-0001 for source SRC-STC-0001 does not "
                "permit redistribution of staged full text papers/paper.bin"
            ),
        ),
    )


@pytest.mark.parametrize(
    ("invalid_case", "expected_detail"),
    [
        (
            "nested-duplicate-key",
            "contains duplicate object key 'redistribution_allowed'",
        ),
        (
            "top-level-duplicate-key",
            "contains duplicate object key 'rights_summary'",
        ),
        (
            "unknown-duplicate-key",
            "contains duplicate object key",
        ),
        ("malformed-json", "contains invalid JSON"),
        ("invalid-source-record", "is not a valid SourceRecord"),
    ],
)
def test_staged_scan_rejects_invalid_source_registry(
    tmp_path: Path,
    invalid_case: str,
    expected_detail: str,
) -> None:
    _, _, scan_staged_index = _rights_api()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tests@example.org")
    _git(repo, "config", "user.name", "STC Tests")
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-qm", "fixture root")

    staged_content = b"%PDF-1.7\nduplicate-key authorization bypass\n"
    staged_artifact = _artifact(
        path="papers/paper.pdf",
        content=staged_content,
    )
    staged_source = _source([staged_artifact]).to_dict()
    source_line = json.dumps(
        staged_source,
        sort_keys=True,
        separators=(",", ":"),
    )
    if invalid_case == "nested-duplicate-key":
        source_line = source_line.replace(
            '"redistribution_allowed":true',
            '"redistribution_allowed":false,"redistribution_allowed":true',
            1,
        )
        assert (
            '"redistribution_allowed":false,"redistribution_allowed":true'
            in source_line
        )
    elif invalid_case == "top-level-duplicate-key":
        source_line = source_line.replace(
            '"rights_summary":"Reuse allowed by the source-level summary."',
            (
                '"rights_summary":"untrusted duplicate",'
                '"rights_summary":"Reuse allowed by the source-level summary."'
            ),
            1,
        )
        assert (
            '"rights_summary":"untrusted duplicate","rights_summary":'
            in source_line
        )
    elif invalid_case == "unknown-duplicate-key":
        source_line = source_line.replace(
            '"topics":',
            (
                '"do-not-echo-secret":"first-sensitive-value",'
                '"do-not-echo-secret":"second-sensitive-value","topics":'
            ),
            1,
        )
        assert source_line.count('"do-not-echo-secret":') == 2
    elif invalid_case == "malformed-json":
        source_line = source_line[:-1]
    elif invalid_case == "invalid-source-record":
        source_line = source_line.replace(
            '"source_id":"SRC-STC-0001"',
            '"source_id":"not-a-source-id"',
            1,
        )
        assert '"source_id":"not-a-source-id"' in source_line
    else:
        raise AssertionError(invalid_case)
    source_registry = (
        repo
        / "research"
        / "sleep-time-compute"
        / "registry"
        / "sources.jsonl"
    )
    source_registry.parent.mkdir(parents=True)
    source_registry.write_text(source_line + "\n", encoding="utf-8")
    paper = repo / "papers" / "paper.pdf"
    paper.parent.mkdir(parents=True)
    paper.write_bytes(staged_content)
    _git(
        repo,
        "add",
        "research/sleep-time-compute/registry/sources.jsonl",
        "papers/paper.pdf",
    )

    report = scan_staged_index(repo, as_of=date(2026, 7, 25))

    registry_path = "research/sleep-time-compute/registry/sources.jsonl"
    assert _diagnostic_contract(report) == (
        (
            "SOURCE_REGISTRY_INVALID",
            registry_path,
            None,
            None,
            (
                f"staged source registry {registry_path} line 1 "
                f"{expected_detail}"
            ),
        ),
    )
    if invalid_case == "unknown-duplicate-key":
        rendered_diagnostics = repr(report.diagnostics)
        assert "do-not-echo-secret" not in rendered_diagnostics
        assert "first-sensitive-value" not in rendered_diagnostics
        assert "second-sensitive-value" not in rendered_diagnostics


def test_staged_scan_reports_invalid_second_source_registry_line(
    tmp_path: Path,
) -> None:
    _, _, scan_staged_index = _rights_api()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tests@example.org")
    _git(repo, "config", "user.name", "STC Tests")
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-qm", "fixture root")

    valid_content = b"%PDF-1.7\nvalid first source record\n"
    valid_source = _source(
        [
            _artifact(
                path="papers/valid.pdf",
                content=valid_content,
                artifact_id="ART-STC-0001",
            )
        ]
    )
    staged_content = b"%PDF-1.7\ninvalid second source record\n"
    invalid_source = _source(
        [
            _artifact(
                path="papers/paper.pdf",
                content=staged_content,
                artifact_id="ART-STC-0002",
            )
        ],
        source_id="SRC-STC-0002",
        legacy_id="W1-002",
        canonical_key="doi:10.1000/invalid-second-source:v1",
        citation_key="author2025invalidsecond",
    )
    valid_line = json.dumps(
        valid_source.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )
    invalid_line = json.dumps(
        invalid_source.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).replace(
        '"redistribution_allowed":true',
        '"redistribution_allowed":false,"redistribution_allowed":true',
        1,
    )
    source_registry = (
        repo
        / "research"
        / "sleep-time-compute"
        / "registry"
        / "sources.jsonl"
    )
    source_registry.parent.mkdir(parents=True)
    source_registry.write_text(
        valid_line + "\n" + invalid_line + "\n",
        encoding="utf-8",
    )
    paper = repo / "papers" / "paper.pdf"
    paper.parent.mkdir(parents=True)
    paper.write_bytes(staged_content)
    _git(
        repo,
        "add",
        "research/sleep-time-compute/registry/sources.jsonl",
        "papers/paper.pdf",
    )

    report = scan_staged_index(repo, as_of=date(2026, 7, 25))

    registry_path = "research/sleep-time-compute/registry/sources.jsonl"
    assert _diagnostic_contract(report) == (
        (
            "SOURCE_REGISTRY_INVALID",
            registry_path,
            None,
            None,
            (
                f"staged source registry {registry_path} line 2 contains "
                "duplicate object key 'redistribution_allowed'"
            ),
        ),
    )


def test_staged_scan_reports_invalid_utf8_without_echoing_payload(
    tmp_path: Path,
) -> None:
    _, _, scan_staged_index = _rights_api()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tests@example.org")
    _git(repo, "config", "user.name", "STC Tests")
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-qm", "fixture root")

    staged_content = b"%PDF-1.7\ninvalid UTF-8 registry bypass\n"
    valid_source = _source(
        [
            _artifact(
                path="papers/valid.pdf",
                content=b"%PDF-1.7\nvalid first source record\n",
            )
        ]
    )
    valid_line = json.dumps(
        valid_source.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    sensitive_invalid_payload = (
        b'"secret":"do-not-echo-' + bytes((0xFF,)) + b'"'
    )
    source_registry = (
        repo
        / "research"
        / "sleep-time-compute"
        / "registry"
        / "sources.jsonl"
    )
    source_registry.parent.mkdir(parents=True)
    source_registry.write_bytes(
        valid_line + b"\n{" + sensitive_invalid_payload + b"}\n"
    )
    paper = repo / "papers" / "paper.pdf"
    paper.parent.mkdir(parents=True)
    paper.write_bytes(staged_content)
    _git(
        repo,
        "add",
        "research/sleep-time-compute/registry/sources.jsonl",
        "papers/paper.pdf",
    )

    report = scan_staged_index(repo, as_of=date(2026, 7, 25))

    registry_path = "research/sleep-time-compute/registry/sources.jsonl"
    assert _diagnostic_contract(report) == (
        (
            "SOURCE_REGISTRY_INVALID",
            registry_path,
            None,
            None,
            (
                f"staged source registry {registry_path} line 2 contains "
                "invalid UTF-8"
            ),
        ),
    )
    assert b"do-not-echo" not in repr(report.diagnostics).encode("utf-8")
