from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, fields
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath

from stc_research.models import ArtifactRef, SourceRecord, SourceVersion

_SOURCE_REGISTRY_PATH = "research/sleep-time-compute/registry/sources.jsonl"
_UNKNOWN_LICENSE_VALUES = frozenset(
    {"", "n/a", "noassertion", "none", "unknown"}
)
_UNKNOWN_DIGEST = "0" * 64
_KNOWN_SOURCE_REGISTRY_KEYS = frozenset(
    field.name
    for model_type in (SourceRecord, SourceVersion, ArtifactRef)
    for field in fields(model_type)
)


@dataclass(frozen=True, slots=True)
class StagedBlob:
    path: str
    content: bytes


@dataclass(frozen=True, slots=True)
class SourceRightsDiagnostic:
    code: str
    path: str
    source_id: str | None
    artifact_id: str | None
    message: str


@dataclass(frozen=True, slots=True)
class SourceRightsReport:
    diagnostics: tuple[SourceRightsDiagnostic, ...]

    @property
    def success(self) -> bool:
        return not self.diagnostics


@dataclass(frozen=True, slots=True)
class _ArtifactBinding:
    source: SourceRecord
    artifact: ArtifactRef


class _DuplicateJsonKeyError(ValueError):
    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(f"duplicate object key {key!r}")


def evaluate_staged_blobs(
    sources: Iterable[SourceRecord],
    blobs: Iterable[StagedBlob],
    as_of: date,
) -> SourceRightsReport:
    artifacts_by_path = _artifacts_by_path(sources)
    diagnostics: list[SourceRightsDiagnostic] = []
    for blob in sorted(blobs, key=lambda item: item.path):
        if _is_cache_path(blob.path):
            diagnostics.append(
                SourceRightsDiagnostic(
                    code="STAGED_CACHE_PATH",
                    path=blob.path,
                    source_id=None,
                    artifact_id=None,
                    message=f"staged cache path {blob.path} is forbidden",
                )
            )
            continue

        bindings = artifacts_by_path.get(blob.path, ())
        if not _is_full_text(blob, bindings):
            continue
        if len(bindings) > 1:
            matches = ", ".join(
                f"{binding.source.source_id}/{binding.artifact.artifact_id}"
                for binding in bindings
            )
            diagnostics.append(
                SourceRightsDiagnostic(
                    code="ARTIFACT_RIGHTS_AMBIGUOUS",
                    path=blob.path,
                    source_id=None,
                    artifact_id=None,
                    message=(
                        f"staged full text {blob.path} matches multiple "
                        f"artifact-level rights records: {matches}"
                    ),
                )
            )
            continue
        if not bindings:
            diagnostics.append(
                SourceRightsDiagnostic(
                    code="ARTIFACT_RIGHTS_MISSING",
                    path=blob.path,
                    source_id=None,
                    artifact_id=None,
                    message=(
                        f"staged full text {blob.path} has no exact "
                        "artifact-level rights record; source summaries cannot "
                        "authorize it"
                    ),
                )
            )
            continue

        binding = bindings[0]
        source = binding.source
        artifact = binding.artifact
        staged_digest = hashlib.sha256(blob.content).hexdigest()
        if staged_digest != artifact.sha256:
            diagnostics.append(
                SourceRightsDiagnostic(
                    code="ARTIFACT_DIGEST_MISMATCH",
                    path=blob.path,
                    source_id=source.source_id,
                    artifact_id=artifact.artifact_id,
                    message=(
                        f"staged full text {blob.path} digest {staged_digest} "
                        f"does not match artifact {artifact.artifact_id} digest "
                        f"{artifact.sha256}"
                    ),
                )
            )
            continue

        if not artifact.redistribution_allowed:
            diagnostics.append(
                SourceRightsDiagnostic(
                    code="REDISTRIBUTION_DISALLOWED",
                    path=blob.path,
                    source_id=source.source_id,
                    artifact_id=artifact.artifact_id,
                    message=(
                        f"artifact {artifact.artifact_id} for source "
                        f"{source.source_id} does not permit redistribution of "
                        f"staged full text {blob.path}"
                    ),
                )
            )
            continue

        if not _has_authoritative_license_evidence(artifact):
            diagnostics.append(
                SourceRightsDiagnostic(
                    code="LICENSE_EVIDENCE_MISSING",
                    path=blob.path,
                    source_id=source.source_id,
                    artifact_id=artifact.artifact_id,
                    message=(
                        f"artifact {artifact.artifact_id} lacks authoritative "
                        f"license evidence for staged full text {blob.path}"
                    ),
                )
            )
        if artifact.license_review_disposition != "ALLOW":
            diagnostics.append(
                SourceRightsDiagnostic(
                    code="RIGHTS_REVIEW_NOT_APPROVED",
                    path=blob.path,
                    source_id=source.source_id,
                    artifact_id=artifact.artifact_id,
                    message=(
                        f"artifact {artifact.artifact_id} rights disposition "
                        f"{artifact.license_review_disposition!r} is not ALLOW "
                        f"for staged full text {blob.path}"
                    ),
                )
            )
        reviewed_at = _rights_review_date(artifact.rights_reviewed_at)
        if reviewed_at > as_of:
            diagnostics.append(
                SourceRightsDiagnostic(
                    code="RIGHTS_REVIEW_NOT_YET_VALID",
                    path=blob.path,
                    source_id=source.source_id,
                    artifact_id=artifact.artifact_id,
                    message=(
                        f"artifact {artifact.artifact_id} rights review starts "
                        f"at {reviewed_at.isoformat()} after scan date "
                        f"{as_of.isoformat()}"
                    ),
                )
            )
        if (
            artifact.rights_review_expires_at is not None
            and (
                expires_at := _rights_review_date(
                    artifact.rights_review_expires_at
                )
            )
            < as_of
        ):
            diagnostics.append(
                SourceRightsDiagnostic(
                    code="RIGHTS_REVIEW_EXPIRED",
                    path=blob.path,
                    source_id=source.source_id,
                    artifact_id=artifact.artifact_id,
                    message=(
                        f"artifact {artifact.artifact_id} rights review expired "
                        f"at {expires_at.isoformat()} before scan date "
                        f"{as_of.isoformat()}"
                    ),
                )
            )

    return SourceRightsReport(diagnostics=tuple(diagnostics))


def scan_staged_index(
    repo_root: Path,
    as_of: date,
) -> SourceRightsReport:
    repository = Path(repo_root)
    source_payload = _git_index_blob(repository, _SOURCE_REGISTRY_PATH)
    sources, registry_diagnostic = _parse_staged_source_registry(source_payload)
    if registry_diagnostic is not None:
        return SourceRightsReport(diagnostics=(registry_diagnostic,))
    staged_paths = tuple(
        path.decode("utf-8")
        for path in _git_output(
            repository,
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=ACMR",
            "-z",
            "--",
        ).split(b"\0")
        if path
    )
    blobs = tuple(
        StagedBlob(path=path, content=_git_index_blob(repository, path))
        for path in staged_paths
        if path != _SOURCE_REGISTRY_PATH
    )
    return evaluate_staged_blobs(sources, blobs, as_of)


def _artifacts_by_path(
    sources: Iterable[SourceRecord],
) -> dict[str, tuple[_ArtifactBinding, ...]]:
    artifacts: dict[str, list[_ArtifactBinding]] = {}
    for source in sorted(sources, key=lambda item: item.source_id):
        for artifact in (
            *source.local_artifacts,
            *source.code_records,
            *source.data_records,
        ):
            if artifact.local_path is None:
                continue
            artifacts.setdefault(artifact.local_path, []).append(
                _ArtifactBinding(source=source, artifact=artifact)
            )
    return {
        path: tuple(
            sorted(
                bindings,
                key=lambda binding: (
                    binding.source.source_id,
                    binding.artifact.artifact_id,
                ),
            )
        )
        for path, bindings in artifacts.items()
    }


def _is_full_text(
    blob: StagedBlob,
    bindings: tuple[_ArtifactBinding, ...],
) -> bool:
    path = PurePosixPath(blob.path)
    return (
        path.suffix.casefold() == ".pdf"
        or blob.content.startswith(b"%PDF-")
        or any(
            binding.artifact.media_type.partition(";")[0].strip().casefold()
            == "application/pdf"
            for binding in bindings
        )
        or (
            path.suffix.casefold() == ".txt"
            and path.is_relative_to("papers/sleep-time-compute/text")
        )
    )


def _is_cache_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return any(
        parts[index : index + 3] == (".cache", "stc", "sources")
        for index in range(len(parts) - 2)
    )


def _has_authoritative_license_evidence(artifact: ArtifactRef) -> bool:
    return (
        artifact.license_id.strip().casefold() not in _UNKNOWN_LICENSE_VALUES
        and artifact.license_evidence_url.strip().casefold()
        not in _UNKNOWN_LICENSE_VALUES
        and artifact.license_evidence_sha256 != _UNKNOWN_DIGEST
    )


def _rights_review_date(value: str) -> date:
    if "T" not in value:
        return date.fromisoformat(value)
    return datetime.fromisoformat(value).astimezone(UTC).date()


def _parse_staged_source_registry(
    payload: bytes,
) -> tuple[tuple[SourceRecord, ...], SourceRightsDiagnostic | None]:
    sources: list[SourceRecord] = []
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            return (), _invalid_source_registry(
                line_number,
                "contains invalid UTF-8",
            )
        try:
            record = json.loads(
                line,
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_nonfinite_json_constant,
                parse_float=_strict_json_float,
            )
        except _DuplicateJsonKeyError as error:
            detail = "contains duplicate object key"
            if error.key in _KNOWN_SOURCE_REGISTRY_KEYS:
                detail = f"{detail} {error.key!r}"
            return (), _invalid_source_registry(
                line_number,
                detail,
            )
        except (json.JSONDecodeError, ValueError):
            return (), _invalid_source_registry(
                line_number,
                "contains invalid JSON",
            )
        if not isinstance(record, dict):
            return (), _invalid_source_registry(
                line_number,
                "is not a valid SourceRecord",
            )
        try:
            sources.append(SourceRecord.from_dict(record))
        except (TypeError, ValueError):
            return (), _invalid_source_registry(
                line_number,
                "is not a valid SourceRecord",
            )
    return tuple(sources), None


def _strict_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _strict_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number {value!r}")
    return parsed


def _invalid_source_registry(
    line_number: int,
    detail: str,
) -> SourceRightsDiagnostic:
    return SourceRightsDiagnostic(
        code="SOURCE_REGISTRY_INVALID",
        path=_SOURCE_REGISTRY_PATH,
        source_id=None,
        artifact_id=None,
        message=(
            f"staged source registry {_SOURCE_REGISTRY_PATH} "
            f"line {line_number} {detail}"
        ),
    )


def _git_index_blob(repository: Path, path: str) -> bytes:
    return _git_output(repository, "show", f":{path}")


def _git_output(repository: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
