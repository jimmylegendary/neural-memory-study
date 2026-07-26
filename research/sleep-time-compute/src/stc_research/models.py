from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Self

from stc_research.ids import parse_stable_id

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+\d{1,3}[- .]?)?(?:\d[- .]?){8,14}\d(?!\d)")
_ISO_DATE_RE = re.compile(r"^[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])$")
_AWARE_DATETIME_RE = re.compile(
    r"^[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])"
    r"T([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]"
    r"(?:\.[0-9]{1,6})?(?:Z|[+-]([01][0-9]|2[0-3]):[0-5][0-9])$"
)
_DIRECT_IDENTIFIER_KEYS = {
    "account_id",
    "address",
    "email",
    "full_name",
    "ip_address",
    "name",
    "phone",
    "phone_number",
    "tenant_id",
    "user_id",
}
_REPRODUCTION_RANK = {"r0": 0, "r1": 1, "r2": 2, "r3": 3}


class SourceGrade(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    X = "X"


class ReviewStatus(StrEnum):
    A_ABS = "A-ABS"
    A_HTML = "A-HTML"
    FULL = "FULL"
    REPRO = "REPRO"


class EvidenceRelation(StrEnum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    QUALIFIES = "QUALIFIES"
    NON_CLAIM = "NON_CLAIM"


class ClaimClass(StrEnum):
    SOURCE_SUMMARY = "SOURCE-SUMMARY"
    VENDOR_BEHAVIOR = "VENDOR-BEHAVIOR"
    RERUN = "RERUN"
    INDEPENDENT_REPLICATION = "INDEPENDENT-REPLICATION"
    ORIGINAL_MEASUREMENT = "ORIGINAL-MEASUREMENT"
    ANALYTIC_DERIVATION = "ANALYTIC-DERIVATION"
    TRACE_SIMULATION = "TRACE-SIMULATION"
    SYNTHESIS = "SYNTHESIS"
    HYPOTHESIS = "HYPOTHESIS"


class ReleaseStatus(StrEnum):
    DRAFT = "DRAFT"
    HYPOTHESIS = "HYPOTHESIS"
    UNRESOLVED = "UNRESOLVED"
    SUPPORTED = "SUPPORTED"
    FALSIFIED_NARROWED = "FALSIFIED/NARROWED"
    BLOCKED = "BLOCKED"
    WITHDRAWN = "WITHDRAWN"


class Warrant(StrEnum):
    MEAS = "MEAS"
    REPL = "REPL"
    DERIV = "DERIV"
    SUMM = "SUMM"
    SYNTH = "SYNTH"
    PROP = "PROP"


class ReproductionStrength(StrEnum):
    R0 = "r0"
    R1 = "r1"
    R2 = "r2"
    R3 = "r3"


class ResultModality(StrEnum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    SOURCE_REPORTED = "SOURCE_REPORTED"
    SIMULATED = "SIMULATED"
    MODEL_ESTIMATED = "MODEL_ESTIMATED"
    ANALYTICAL = "ANALYTICAL"
    ACTUAL_MEASUREMENT = "ACTUAL_MEASUREMENT"
    ACTUAL_HARDWARE = "ACTUAL_HARDWARE"


_LEGACY_NO_RESULT_DEFAULT_CLASSES = frozenset(
    {
        ClaimClass.SOURCE_SUMMARY,
        ClaimClass.VENDOR_BEHAVIOR,
        ClaimClass.SYNTHESIS,
        ClaimClass.HYPOTHESIS,
    }
)
_RESULT_PATH_MODALITIES = frozenset(
    {
        ResultModality.SIMULATED,
        ResultModality.MODEL_ESTIMATED,
        ResultModality.ANALYTICAL,
        ResultModality.ACTUAL_MEASUREMENT,
        ResultModality.ACTUAL_HARDWARE,
    }
)


def _raw_claim_has_result_dependency(data: Mapping[str, Any]) -> bool:
    if bool(data.get("result_ids")):
        return True
    requirement = data.get("evidence_requirement")
    if isinstance(requirement, Mapping):
        return bool(
            requirement.get("requires_result_block")
            or requirement.get("minimum_independent_result_paths")
        )
    return bool(
        getattr(requirement, "requires_result_block", False)
        or getattr(requirement, "minimum_independent_result_paths", 0)
    )


class SupportRole(StrEnum):
    RELEASE_SUPPORT = "RELEASE_SUPPORT"
    CONTEXT = "CONTEXT"
    COUNTEREVIDENCE = "COUNTEREVIDENCE"


class GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"


def _nonempty(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _sha256(value: object, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _git_sha1(value: object, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or _GIT_SHA1_RE.fullmatch(value) is None
        or value == "0" * 40
    ):
        raise ValueError(f"{field_name} must be a lowercase 40-hex Git SHA-1 object ID")


def _iso_date(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be an ISO date")
    if _ISO_DATE_RE.fullmatch(value) is None or value.startswith("0000-"):
        raise ValueError(f"{field_name} must be a canonical YYYY-MM-DD ISO date")
    try:
        date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a valid ISO date") from error
    return value


def _aware_datetime(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a timezone-aware ISO datetime")
    if _AWARE_DATETIME_RE.fullmatch(value) is None or value.startswith("0000-"):
        raise ValueError(
            f"{field_name} must be a canonical timezone-aware ISO datetime"
        )
    if value.endswith("-00:00"):
        raise ValueError(f"{field_name} must not use the unknown -00:00 offset")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a valid ISO datetime") from error
    if parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    timespec = "seconds" if parsed.microsecond == 0 else "microseconds"
    canonical = parsed.isoformat(timespec=timespec)
    if parsed.utcoffset().total_seconds() == 0:
        canonical = f"{canonical[:-6]}Z"
    return canonical


def _iso_date_or_datetime(value: object, field_name: str) -> str:
    if isinstance(value, str) and "T" not in value:
        return _iso_date(value, field_name)
    return _aware_datetime(value, field_name)


def _public_date(value: object, precision: object, field_name: str) -> None:
    if not isinstance(value, str) or not isinstance(precision, str):
        raise TypeError(f"{field_name} and date_precision must be strings")
    if precision == "DAY":
        _iso_date(value, field_name)
        return
    if (
        precision == "MONTH"
        and re.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])", value)
        and not value.startswith("0000-")
    ):
        return
    if precision == "YEAR" and re.fullmatch(r"[0-9]{4}", value) and value != "0000":
        return
    raise ValueError(f"{field_name} must match ISO date precision YEAR, MONTH, or DAY")


def _stable_id(value: object, field_name: str, *allowed_kinds: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a canonical stable ID")
    try:
        parsed = parse_stable_id(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a canonical stable ID") from error
    if parsed.kind not in allowed_kinds:
        expected = ", ".join(allowed_kinds)
        raise ValueError(f"{field_name} must have ID kind {expected}")


def _relative_path(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a package-root-relative path")
    path = PurePosixPath(value)
    if (
        "\\" in value
        or path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError(f"{field_name} must be a package-root-relative safe path")


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _freeze_mapping(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{field_name} keys must be strings")
        if not key:
            raise ValueError(f"{field_name} keys must be non-empty")
        frozen[key] = _freeze_value(item, field_name)
    return MappingProxyType(dict(sorted(frozen.items())))


def _freeze_value(value: Any, field_name: str) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value, field_name)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item, field_name) for item in value)
    if isinstance(value, StrEnum):
        return value
    if value is None or type(value) in {str, int, bool}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field_name} JSON numbers must be finite")
        return value
    raise TypeError(
        f"{field_name} must contain only canonical JSON values; "
        f"got {type(value).__name__}"
    )


def _thaw(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, _Record):
        return _canonical_record_dict(value)
    if isinstance(value, Mapping):
        return {
            str(key): _thaw(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _canonical_record_dict(record: _Record) -> dict[str, Any]:
    return {field.name: _thaw(getattr(record, field.name)) for field in fields(record)}


def _reject_direct_identifiers_in_operational_metadata(
    data: Mapping[str, Any],
) -> None:
    for key, value in data.items():
        normalized = str(key).lower().replace("-", "_")
        if "metadata" not in normalized or not any(
            marker in normalized for marker in ("operational", "tenant", "user")
        ):
            continue
        _scan_direct_identifiers(value)


def _scan_direct_identifiers(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in _DIRECT_IDENTIFIER_KEYS:
                raise ValueError(
                    "direct identifier is forbidden in operational metadata"
                )
            _scan_direct_identifiers(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _scan_direct_identifiers(item)
        return
    if isinstance(value, str) and (
        _EMAIL_RE.search(value) is not None or _PHONE_RE.search(value) is not None
    ):
        raise ValueError("direct identifier is forbidden in operational metadata")


class _Record:
    __slots__ = ()

    _ENUM_FIELDS: ClassVar[Mapping[str, type[StrEnum]]] = {}
    _TUPLE_ENUM_FIELDS: ClassVar[Mapping[str, type[StrEnum]]] = {}
    _NESTED_FIELDS: ClassVar[Mapping[str, type[_Record]]] = {}
    _OPTIONAL_NESTED_FIELDS: ClassVar[Mapping[str, type[_Record]]] = {}
    _TUPLE_NESTED_FIELDS: ClassVar[Mapping[str, type[_Record]]] = {}
    _TUPLE_FIELDS: ClassVar[frozenset[str]] = frozenset()
    _MAPPING_FIELDS: ClassVar[frozenset[str]] = frozenset()
    _TUPLE_MAPPING_FIELDS: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls._build(data)

    @classmethod
    def _build(cls, data: Mapping[str, Any]) -> Self:
        if not isinstance(data, Mapping):
            raise TypeError(f"{cls.__name__} input must be a mapping")
        expected = tuple(field.name for field in fields(cls) if field.init)
        missing = [name for name in expected if name not in data]
        unexpected = sorted(set(data) - set(expected))
        if missing:
            raise ValueError(f"{cls.__name__} missing fields: {', '.join(missing)}")
        if unexpected:
            raise ValueError(
                f"{cls.__name__} unexpected fields: {', '.join(unexpected)}"
            )

        values: dict[str, Any] = {}
        for name in expected:
            value = data[name]
            if name in cls._ENUM_FIELDS:
                enum_type = cls._ENUM_FIELDS[name]
                try:
                    value = enum_type(value)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"unknown {name}: {value!r}") from error
            elif name in cls._TUPLE_ENUM_FIELDS:
                enum_type = cls._TUPLE_ENUM_FIELDS[name]
                value = cls._enum_tuple(value, enum_type, name)
            elif name in cls._NESTED_FIELDS:
                nested_type = cls._NESTED_FIELDS[name]
                if type(value) is nested_type:
                    pass
                elif isinstance(value, Mapping):
                    value = nested_type.from_dict(value)
                else:
                    raise TypeError(
                        f"{name} must be exactly {nested_type.__name__} "
                        "or a plain mapping"
                    )
            elif name in cls._OPTIONAL_NESTED_FIELDS:
                nested_type = cls._OPTIONAL_NESTED_FIELDS[name]
                if value is None or type(value) is nested_type:
                    pass
                elif isinstance(value, Mapping):
                    value = nested_type.from_dict(value)
                else:
                    raise TypeError(
                        f"{name} must be exactly {nested_type.__name__}, "
                        "a plain mapping, or None"
                    )
            elif name in cls._TUPLE_NESTED_FIELDS:
                nested_type = cls._TUPLE_NESTED_FIELDS[name]
                items = cls._sequence(value, name)
                converted: list[_Record] = []
                for item in items:
                    if type(item) is nested_type:
                        converted.append(item)
                    elif isinstance(item, Mapping):
                        converted.append(nested_type.from_dict(item))
                    else:
                        raise TypeError(
                            f"{name} entries must be exactly "
                            f"{nested_type.__name__} or plain mappings"
                        )
                value = tuple(converted)
            elif name in cls._TUPLE_MAPPING_FIELDS:
                items = cls._sequence(value, name)
                value = tuple(_freeze_mapping(item, name) for item in items)
            elif name in cls._TUPLE_FIELDS:
                value = cls._string_tuple(value, name)
            elif name in cls._MAPPING_FIELDS:
                value = _freeze_mapping(value, name)
            values[name] = value
        return cls(**values)

    @staticmethod
    def _sequence(value: Any, field_name: str) -> list[Any] | tuple[Any, ...]:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{field_name} must be a list")
        return value

    @classmethod
    def _string_tuple(cls, value: Any, field_name: str) -> tuple[str, ...]:
        result: list[str] = []
        for item in cls._sequence(value, field_name):
            if not isinstance(item, str):
                raise TypeError(f"{field_name} entries must be strings")
            if not item.strip():
                raise ValueError(f"{field_name} entries must be non-empty")
            result.append(item)
        return tuple(result)

    @classmethod
    def _enum_tuple(
        cls, value: Any, enum_type: type[StrEnum], field_name: str
    ) -> tuple[StrEnum, ...]:
        converted: list[StrEnum] = []
        for item in cls._sequence(value, field_name):
            try:
                converted.append(enum_type(item))
            except (TypeError, ValueError) as error:
                raise ValueError(f"unknown {field_name} value: {item!r}") from error
        return tuple(converted)

    def to_dict(self) -> dict[str, Any]:
        return _canonical_record_dict(self)

    def _validate_declared_containers(self) -> None:
        self._snapshot_declared_mappings()
        for name, enum_type in self._ENUM_FIELDS.items():
            if not isinstance(getattr(self, name), enum_type):
                raise TypeError(f"{name} must be {enum_type.__name__}")
        for name, enum_type in self._TUPLE_ENUM_FIELDS.items():
            value = getattr(self, name)
            self._require_tuple(value, name)
            if any(not isinstance(item, enum_type) for item in value):
                raise TypeError(f"{name} entries must be {enum_type.__name__}")
        for name, nested_type in self._NESTED_FIELDS.items():
            if type(getattr(self, name)) is not nested_type:
                raise TypeError(f"{name} must be exactly {nested_type.__name__}")
        for name, nested_type in self._OPTIONAL_NESTED_FIELDS.items():
            value = getattr(self, name)
            if value is not None and type(value) is not nested_type:
                raise TypeError(
                    f"{name} must be exactly {nested_type.__name__} or None"
                )
        for name, nested_type in self._TUPLE_NESTED_FIELDS.items():
            value = getattr(self, name)
            self._require_tuple(value, name)
            if any(type(item) is not nested_type for item in value):
                raise TypeError(
                    f"{name} entries must be exactly {nested_type.__name__}"
                )
        for name in self._TUPLE_FIELDS:
            value = getattr(self, name)
            self._require_tuple(value, name)
            for item in value:
                if not isinstance(item, str):
                    raise TypeError(f"{name} entries must be strings")
                if not item.strip():
                    raise ValueError(f"{name} entries must be non-empty")
        for name in self._TUPLE_MAPPING_FIELDS:
            value = getattr(self, name)
            self._require_tuple(value, name)
            for item in value:
                _validate_frozen_mapping(item, name)
        for name in self._MAPPING_FIELDS:
            _validate_frozen_mapping(getattr(self, name), name)

    def _snapshot_declared_mappings(self) -> None:
        for name in self._MAPPING_FIELDS:
            object.__setattr__(
                self,
                name,
                _freeze_mapping(getattr(self, name), name),
            )
        for name in self._TUPLE_MAPPING_FIELDS:
            value = getattr(self, name)
            self._require_tuple(value, name)
            object.__setattr__(
                self,
                name,
                tuple(_freeze_mapping(item, name) for item in value),
            )

    @staticmethod
    def _require_tuple(value: Any, field_name: str) -> None:
        if not isinstance(value, tuple):
            raise TypeError(f"{field_name} must be an immutable tuple")


def _validate_frozen_mapping(value: Any, field_name: str) -> None:
    if not isinstance(value, MappingProxyType):
        raise TypeError(f"{field_name} must be an immutable mapping")
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{field_name} keys must be strings")
        if not key:
            raise ValueError(f"{field_name} keys must be non-empty")
        _validate_deeply_immutable(item, field_name)


def _validate_deeply_immutable(value: Any, field_name: str) -> None:
    if isinstance(value, Mapping):
        _validate_frozen_mapping(value, field_name)
        return
    if isinstance(value, tuple):
        for item in value:
            _validate_deeply_immutable(item, field_name)
        return
    if isinstance(value, StrEnum):
        return
    if value is None or type(value) in {str, int, bool}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field_name} JSON numbers must be finite")
        return
    raise TypeError(
        f"{field_name} must contain only canonical JSON values; "
        f"got {type(value).__name__}"
    )


@dataclass(frozen=True, slots=True)
class SubjectRef(_Record):
    path: str
    sha256: str

    def __post_init__(self) -> None:
        self._validate_declared_containers()
        _relative_path(self.path, "path")
        _sha256(self.sha256, "sha256")


@dataclass(frozen=True, slots=True)
class SourceVersion(_Record):
    identifier: str
    public_date: str
    date_precision: str
    status: str
    canonical_url: str | None
    local_artifact_id: str
    sha256: str
    supersedes: str | None
    derived_from: tuple[str, ...]

    _TUPLE_FIELDS: ClassVar[frozenset[str]] = frozenset({"derived_from"})

    def __post_init__(self) -> None:
        self._validate_declared_containers()
        _nonempty(self.identifier, "identifier")
        _public_date(self.public_date, self.date_precision, "public_date")
        _nonempty(self.status, "status")
        if self.canonical_url is not None:
            _nonempty(self.canonical_url, "canonical_url")
        _stable_id(self.local_artifact_id, "local_artifact_id", "artifact")
        _sha256(self.sha256, "sha256")
        if self.supersedes is not None:
            _nonempty(self.supersedes, "supersedes")
        for lineage_id in self.derived_from:
            _nonempty(lineage_id, "derived_from")


@dataclass(frozen=True, slots=True)
class ArtifactRef(_Record):
    artifact_id: str
    canonical_url: str | None
    local_path: str | None
    sha256: str
    media_type: str
    availability: str
    license_id: str
    license_evidence_url: str
    license_evidence_sha256: str
    redistribution_allowed: bool
    license_review_disposition: str
    rights_reviewer: str
    rights_reviewed_at: str
    rights_review_expires_at: str | None

    def __post_init__(self) -> None:
        self._validate_declared_containers()
        _stable_id(self.artifact_id, "artifact_id", "artifact")
        if self.canonical_url is None and self.local_path is None:
            raise ValueError("ArtifactRef requires canonical_url or local_path")
        if self.canonical_url is not None:
            _nonempty(self.canonical_url, "canonical_url")
        if self.local_path is not None:
            _relative_path(self.local_path, "local_path")
        _sha256(self.sha256, "sha256")
        for field_name in (
            "media_type",
            "availability",
            "license_id",
            "license_evidence_url",
            "license_review_disposition",
            "rights_reviewer",
        ):
            _nonempty(getattr(self, field_name), field_name)
        _sha256(self.license_evidence_sha256, "license_evidence_sha256")
        if not isinstance(self.redistribution_allowed, bool):
            raise TypeError("redistribution_allowed must be boolean")
        object.__setattr__(
            self,
            "rights_reviewed_at",
            _iso_date_or_datetime(self.rights_reviewed_at, "rights_reviewed_at"),
        )
        if self.rights_review_expires_at is not None:
            object.__setattr__(
                self,
                "rights_review_expires_at",
                _iso_date_or_datetime(
                    self.rights_review_expires_at, "rights_review_expires_at"
                ),
            )


@dataclass(frozen=True, slots=True)
class SourceRecord(_Record):
    source_id: str
    legacy_ids: tuple[str, ...]
    canonical_key: str
    title: str
    authors: tuple[str, ...]
    earliest_public_date: str
    date_precision: str
    venue_status: str
    peer_review_status: str
    versions: tuple[SourceVersion, ...]
    source_grade: SourceGrade
    review_status: ReviewStatus
    urls: tuple[str, ...]
    local_artifacts: tuple[ArtifactRef, ...]
    citation_key: str
    code_records: tuple[ArtifactRef, ...]
    data_records: tuple[ArtifactRef, ...]
    rights_summary: str
    topics: tuple[str, ...]
    affiliations: tuple[str, ...]
    inclusion_reason: str
    non_claim: str
    last_verified: str

    _ENUM_FIELDS: ClassVar[Mapping[str, type[StrEnum]]] = {
        "source_grade": SourceGrade,
        "review_status": ReviewStatus,
    }
    _TUPLE_NESTED_FIELDS: ClassVar[Mapping[str, type[_Record]]] = {
        "versions": SourceVersion,
        "local_artifacts": ArtifactRef,
        "code_records": ArtifactRef,
        "data_records": ArtifactRef,
    }
    _TUPLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "legacy_ids",
            "authors",
            "urls",
            "topics",
            "affiliations",
        }
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        if not isinstance(data, Mapping):
            raise TypeError("SourceRecord input must be a mapping")
        _reject_direct_identifiers_in_operational_metadata(data)
        return cls._build(data)

    def __post_init__(self) -> None:
        self._validate_declared_containers()
        _stable_id(self.source_id, "source_id", "source")
        for field_name in (
            "canonical_key",
            "title",
            "venue_status",
            "peer_review_status",
            "citation_key",
            "rights_summary",
            "inclusion_reason",
            "non_claim",
        ):
            _nonempty(getattr(self, field_name), field_name)
        _public_date(
            self.earliest_public_date,
            self.date_precision,
            "earliest_public_date",
        )
        _iso_date(self.last_verified, "last_verified")
        if not self.authors:
            raise ValueError("authors must not be empty")
        if not self.versions:
            raise ValueError("versions must not be empty")


@dataclass(frozen=True, slots=True)
class EvidenceRecord(_Record):
    evidence_id: str
    source_id: str
    source_version: str
    source_version_digest: str
    anchor_kind: str
    anchor: str
    support_span_digest: str
    support_summary: str
    relation: EvidenceRelation
    claim_ids: tuple[str, ...]
    question_ids: tuple[str, ...]
    hypothesis_ids: tuple[str, ...]
    source_grade: SourceGrade
    warrant: Warrant
    reproduction_strength: ReproductionStrength
    assumptions: tuple[str, ...]
    scope: str
    counterevidence_ids: tuple[str, ...]
    reviewer: str
    reviewed_at: str

    _ENUM_FIELDS: ClassVar[Mapping[str, type[StrEnum]]] = {
        "relation": EvidenceRelation,
        "source_grade": SourceGrade,
        "warrant": Warrant,
        "reproduction_strength": ReproductionStrength,
    }
    _TUPLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "claim_ids",
            "question_ids",
            "hypothesis_ids",
            "assumptions",
            "counterevidence_ids",
        }
    )

    def __post_init__(self) -> None:
        self._validate_declared_containers()
        _stable_id(self.evidence_id, "evidence_id", "evidence")
        _stable_id(self.source_id, "source_id", "source")
        _nonempty(self.source_version, "source_version")
        _sha256(self.source_version_digest, "source_version_digest")
        for field_name in (
            "anchor_kind",
            "anchor",
            "support_summary",
            "scope",
            "reviewer",
        ):
            _nonempty(getattr(self, field_name), field_name)
        _sha256(self.support_span_digest, "support_span_digest")
        for claim_id in self.claim_ids:
            _stable_id(claim_id, "claim_ids", "claim", "paper_claim")
        for question_id in self.question_ids:
            _stable_id(question_id, "question_ids", "question")
        for hypothesis_id in self.hypothesis_ids:
            _stable_id(hypothesis_id, "hypothesis_ids", "hypothesis")
        for evidence_id in self.counterevidence_ids:
            _stable_id(evidence_id, "counterevidence_ids", "evidence")
        object.__setattr__(
            self,
            "reviewed_at",
            _iso_date_or_datetime(self.reviewed_at, "reviewed_at"),
        )


@dataclass(frozen=True, slots=True)
class ClaimSupportRef(_Record):
    evidence_id: str
    source_id: str
    source_version: str
    source_version_digest: str
    artifact_id: str
    support_span_digest: str
    relation: EvidenceRelation
    role: SupportRole
    source_grade: SourceGrade
    warrant: Warrant
    reproduction_strength: ReproductionStrength

    _ENUM_FIELDS: ClassVar[Mapping[str, type[StrEnum]]] = {
        "relation": EvidenceRelation,
        "role": SupportRole,
        "source_grade": SourceGrade,
        "warrant": Warrant,
        "reproduction_strength": ReproductionStrength,
    }

    def __post_init__(self) -> None:
        self._validate_declared_containers()
        _stable_id(self.evidence_id, "evidence_id", "evidence")
        _stable_id(self.source_id, "source_id", "source")
        _nonempty(self.source_version, "source_version")
        _sha256(self.source_version_digest, "source_version_digest")
        _stable_id(self.artifact_id, "artifact_id", "artifact")
        _sha256(self.support_span_digest, "support_span_digest")
        if (
            self.role is SupportRole.RELEASE_SUPPORT
            and self.relation is not EvidenceRelation.SUPPORTS
        ):
            raise ValueError("RELEASE_SUPPORT requires relation SUPPORTS")
        if (
            self.role is SupportRole.RELEASE_SUPPORT
            and self.source_grade is SourceGrade.X
        ):
            raise ValueError("grade X cannot be used as RELEASE_SUPPORT")
        if self.role is SupportRole.COUNTEREVIDENCE and self.relation not in {
            EvidenceRelation.CONTRADICTS,
            EvidenceRelation.QUALIFIES,
        }:
            raise ValueError(
                "COUNTEREVIDENCE requires relation CONTRADICTS or QUALIFIES"
            )
        if (
            self.relation is EvidenceRelation.CONTRADICTS
            and self.role is not SupportRole.COUNTEREVIDENCE
        ):
            raise ValueError("CONTRADICTS requires role COUNTEREVIDENCE")
        if (
            self.relation is EvidenceRelation.NON_CLAIM
            and self.role is not SupportRole.CONTEXT
        ):
            raise ValueError("NON_CLAIM requires role CONTEXT")


@dataclass(frozen=True, slots=True)
class ClaimEvidenceRequirement(_Record):
    allowed_claim_classes: tuple[ClaimClass, ...]
    allowed_source_grades: tuple[SourceGrade, ...]
    allowed_warrants: tuple[Warrant, ...]
    minimum_reproduction_strength: ReproductionStrength
    minimum_independent_sources: int
    requires_result_block: bool
    minimum_independent_result_paths: int

    _ENUM_FIELDS: ClassVar[Mapping[str, type[StrEnum]]] = {
        "minimum_reproduction_strength": ReproductionStrength,
    }
    _TUPLE_ENUM_FIELDS: ClassVar[Mapping[str, type[StrEnum]]] = {
        "allowed_claim_classes": ClaimClass,
        "allowed_source_grades": SourceGrade,
        "allowed_warrants": Warrant,
    }

    def __post_init__(self) -> None:
        self._validate_declared_containers()
        if not self.allowed_claim_classes:
            raise ValueError("allowed_claim_classes must not be empty")
        if not self.allowed_source_grades:
            raise ValueError("allowed_source_grades must not be empty")
        if SourceGrade.X in self.allowed_source_grades:
            raise ValueError("grade X cannot be release support")
        if not self.allowed_warrants:
            raise ValueError("allowed_warrants must not be empty")
        for field_name in (
            "minimum_independent_sources",
            "minimum_independent_result_paths",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if not isinstance(self.requires_result_block, bool):
            raise TypeError("requires_result_block must be boolean")


@dataclass(frozen=True, slots=True)
class ClaimResultProvenance(_Record):
    asserted_modality: ResultModality
    calibration_modalities: tuple[ResultModality, ...]

    _ENUM_FIELDS: ClassVar[Mapping[str, type[StrEnum]]] = {
        "asserted_modality": ResultModality,
    }
    _TUPLE_ENUM_FIELDS: ClassVar[Mapping[str, type[StrEnum]]] = {
        "calibration_modalities": ResultModality,
    }

    def __post_init__(self) -> None:
        self._validate_declared_containers()
        if ResultModality.NOT_APPLICABLE in self.calibration_modalities:
            raise ValueError("calibration_modalities cannot contain NOT_APPLICABLE")
        if len(self.calibration_modalities) != len(set(self.calibration_modalities)):
            raise ValueError("calibration_modalities must be unique")


@dataclass(frozen=True, slots=True)
class ClaimRecord(_Record):
    claim_id: str
    statement: str
    headline_quantitative: bool
    claim_class: ClaimClass
    result_provenance: ClaimResultProvenance
    status: ReleaseStatus
    scope: str
    assumptions: tuple[str, ...]
    source_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    support_refs: tuple[ClaimSupportRef, ...]
    evidence_requirement: ClaimEvidenceRequirement
    counterevidence_ids: tuple[str, ...]
    result_ids: tuple[str, ...]
    artifact_dependencies: tuple[str, ...]
    caveats: tuple[str, ...]
    falsifier: str
    paper_owner: str
    gate_status: str
    last_audit_date: str

    _ENUM_FIELDS: ClassVar[Mapping[str, type[StrEnum]]] = {
        "claim_class": ClaimClass,
        "status": ReleaseStatus,
    }
    _NESTED_FIELDS: ClassVar[Mapping[str, type[_Record]]] = {
        "evidence_requirement": ClaimEvidenceRequirement,
        "result_provenance": ClaimResultProvenance,
    }
    _TUPLE_NESTED_FIELDS: ClassVar[Mapping[str, type[_Record]]] = {
        "support_refs": ClaimSupportRef,
    }
    _TUPLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "assumptions",
            "source_ids",
            "evidence_ids",
            "counterevidence_ids",
            "result_ids",
            "artifact_dependencies",
            "caveats",
        }
    )

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        for_release: bool = False,
        headline_quantitative: bool | None = None,
    ) -> Self:
        if not isinstance(data, Mapping):
            raise TypeError("ClaimRecord input must be a mapping")
        payload = dict(data)
        if "headline_quantitative" not in payload:
            if headline_quantitative is None:
                raise ValueError(
                    "ClaimRecord requires explicit headline_quantitative classification"
                )
            if not isinstance(headline_quantitative, bool):
                raise TypeError("headline_quantitative must be boolean")
            payload["headline_quantitative"] = headline_quantitative
        elif headline_quantitative is not None:
            if not isinstance(headline_quantitative, bool):
                raise TypeError("headline_quantitative must be boolean")
            if payload["headline_quantitative"] is not headline_quantitative:
                raise ValueError(
                    "headline_quantitative argument conflicts with persisted "
                    "classification"
                )
        if "result_provenance" not in payload:
            raw_claim_class = payload.get("claim_class")
            try:
                claim_class = ClaimClass(raw_claim_class)
            except TypeError, ValueError:
                claim_class = None
            may_default_to_not_applicable = (
                claim_class in _LEGACY_NO_RESULT_DEFAULT_CLASSES
                and not _raw_claim_has_result_dependency(payload)
            )
            if claim_class is not None and not may_default_to_not_applicable:
                raise ValueError(
                    f"{claim_class.value} requires explicit structured "
                    "result_provenance"
                )
            payload["result_provenance"] = {
                "asserted_modality": ResultModality.NOT_APPLICABLE.value,
                "calibration_modalities": [],
            }
        if payload["headline_quantitative"] is True:
            cls._prevalidate_headline_strength(payload)
        record = cls._build(payload)
        record._validate_context(for_release=for_release)
        return record

    @staticmethod
    def _prevalidate_headline_strength(data: Mapping[str, Any]) -> None:
        requirement = data.get("evidence_requirement")
        support_refs = data.get("support_refs")
        if not isinstance(requirement, Mapping) or not isinstance(
            support_refs, (list, tuple)
        ):
            return
        minimum = requirement.get("minimum_reproduction_strength")
        if minimum in _REPRODUCTION_RANK and (
            _REPRODUCTION_RANK[minimum] < _REPRODUCTION_RANK["r2"]
        ):
            raise ValueError(
                "headline quantitative claim requires typed reproduction "
                "strength r2 or above"
            )
        actual_strengths = [
            ref.get("reproduction_strength")
            for ref in support_refs
            if isinstance(ref, Mapping)
            and ref.get("role") == "RELEASE_SUPPORT"
            and ref.get("relation") == "SUPPORTS"
        ]
        if actual_strengths and not any(
            strength in _REPRODUCTION_RANK
            and _REPRODUCTION_RANK[strength] >= _REPRODUCTION_RANK["r2"]
            for strength in actual_strengths
        ):
            raise ValueError(
                "headline quantitative claim requires actual RELEASE_SUPPORT "
                "at r2 or above"
            )

    def __post_init__(self) -> None:
        self._validate_declared_containers()
        _stable_id(self.claim_id, "claim_id", "claim", "paper_claim")
        if not isinstance(self.headline_quantitative, bool):
            raise TypeError("headline_quantitative must be boolean")
        for field_name in (
            "statement",
            "scope",
            "falsifier",
            "paper_owner",
            "gate_status",
        ):
            _nonempty(getattr(self, field_name), field_name)
        for source_id in self.source_ids:
            _stable_id(source_id, "source_ids", "source")
        for evidence_id in self.evidence_ids:
            _stable_id(evidence_id, "evidence_ids", "evidence")
        for evidence_id in self.counterevidence_ids:
            _stable_id(evidence_id, "counterevidence_ids", "evidence")
        for result_id in self.result_ids:
            _stable_id(result_id, "result_ids", "result")
        for artifact_id in self.artifact_dependencies:
            _stable_id(artifact_id, "artifact_dependencies", "artifact")
        _iso_date(self.last_audit_date, "last_audit_date")

        expected_source_ids = _ordered_unique(
            ref.source_id for ref in self.support_refs
        )
        expected_evidence_ids = _ordered_unique(
            ref.evidence_id for ref in self.support_refs
        )
        if (
            self.source_ids != expected_source_ids
            or self.evidence_ids != expected_evidence_ids
        ):
            raise ValueError(
                "source_ids and evidence_ids must exactly equal typed support "
                "derived indexes"
            )
        expected_counterevidence_ids = _ordered_unique(
            ref.evidence_id
            for ref in self.support_refs
            if ref.role is SupportRole.COUNTEREVIDENCE
        )
        if self.counterevidence_ids != expected_counterevidence_ids:
            raise ValueError(
                "counterevidence_ids must exactly equal typed COUNTEREVIDENCE "
                "support refs in first-reference order"
            )
        self._validate_result_provenance()
        self._validate_evidence_requirement()
        self._validate_headline_quantitative()

    def _qualifying_release_support(self) -> tuple[ClaimSupportRef, ...]:
        requirement = self.evidence_requirement
        minimum_rank = _REPRODUCTION_RANK[
            requirement.minimum_reproduction_strength.value
        ]
        return tuple(
            ref
            for ref in self.support_refs
            if ref.role is SupportRole.RELEASE_SUPPORT
            and ref.relation is EvidenceRelation.SUPPORTS
            and ref.source_grade is not SourceGrade.X
            and ref.source_grade in requirement.allowed_source_grades
            and ref.warrant in requirement.allowed_warrants
            and _REPRODUCTION_RANK[ref.reproduction_strength.value] >= minimum_rank
        )

    def _validate_evidence_requirement(self) -> None:
        requirement = self.evidence_requirement
        if self.claim_class not in requirement.allowed_claim_classes:
            raise ValueError("claim_class fails typed evidence requirement")
        qualifying = self._qualifying_release_support()
        independent_sources = {ref.source_id for ref in qualifying}
        if len(independent_sources) < requirement.minimum_independent_sources:
            raise ValueError(
                "typed evidence requirement has insufficient independent source support"
            )
        if requirement.requires_result_block and not self.result_ids:
            raise ValueError("typed evidence requirement requires a result block")
        if len(set(self.result_ids)) < requirement.minimum_independent_result_paths:
            raise ValueError(
                "typed evidence requirement has insufficient independent result paths"
            )

    def _validate_result_provenance(self) -> None:
        modality = self.result_provenance.asserted_modality
        if modality is ResultModality.NOT_APPLICABLE and self._has_result_dependency():
            raise ValueError(
                f"{self.claim_class.value}: NOT_APPLICABLE modality cannot "
                "coexist with a result block or result provenance dependency"
            )
        if self.claim_class in _LEGACY_NO_RESULT_DEFAULT_CLASSES:
            if modality not in {
                ResultModality.NOT_APPLICABLE,
                ResultModality.SOURCE_REPORTED,
            }:
                raise ValueError(
                    f"{self.claim_class.value} asserted modality must be "
                    "NOT_APPLICABLE or SOURCE_REPORTED"
                )
            return

        if self.claim_class is ClaimClass.TRACE_SIMULATION:
            if modality in {
                ResultModality.SIMULATED,
                ResultModality.MODEL_ESTIMATED,
                ResultModality.ANALYTICAL,
            }:
                return
            raise ValueError(
                "TRACE-SIMULATION may assert only simulated, model-estimated, "
                "or analytical outputs"
            )

        if self.claim_class is ClaimClass.ORIGINAL_MEASUREMENT:
            if modality in {
                ResultModality.ACTUAL_MEASUREMENT,
                ResultModality.ACTUAL_HARDWARE,
            }:
                return
            raise ValueError(
                "ORIGINAL-MEASUREMENT asserted modality must be "
                "ACTUAL_MEASUREMENT or ACTUAL_HARDWARE"
            )

        if self.claim_class in {
            ClaimClass.RERUN,
            ClaimClass.INDEPENDENT_REPLICATION,
        }:
            if modality in _RESULT_PATH_MODALITIES:
                return
            raise ValueError(
                f"{self.claim_class.value} asserted modality must describe "
                "an explicit result path"
            )

        if self.claim_class is ClaimClass.ANALYTIC_DERIVATION:
            allowed = (
                {ResultModality.ANALYTICAL}
                if self.result_ids
                else {
                    ResultModality.NOT_APPLICABLE,
                    ResultModality.ANALYTICAL,
                }
            )
            if modality in allowed:
                return
            raise ValueError(
                "ANALYTIC-DERIVATION result-bearing asserted modality "
                "must be ANALYTICAL"
            )

    def _has_result_dependency(self) -> bool:
        return bool(
            self.result_ids
            or self.evidence_requirement.requires_result_block
            or self.evidence_requirement.minimum_independent_result_paths
            or self.result_provenance.calibration_modalities
        )

    def _validate_context(self, *, for_release: bool) -> None:
        parsed = parse_stable_id(self.claim_id)
        if (
            for_release
            and parsed.kind == "paper_claim"
            and self.status
            not in {
                ReleaseStatus.SUPPORTED,
                ReleaseStatus.FALSIFIED_NARROWED,
            }
        ):
            raise ValueError("PAPER-C claim must have a release-admissible status")
        if (
            for_release
            and self.status is ReleaseStatus.SUPPORTED
            and not self._qualifying_release_support()
        ):
            raise ValueError(
                "SUPPORTED release claim requires positive qualifying RELEASE_SUPPORT"
            )

    def _validate_headline_quantitative(self) -> None:
        if not self.headline_quantitative:
            return
        if self.claim_class not in {
            ClaimClass.ORIGINAL_MEASUREMENT,
            ClaimClass.INDEPENDENT_REPLICATION,
        }:
            raise ValueError(
                "headline quantitative claim requires ORIGINAL-MEASUREMENT "
                "or INDEPENDENT-REPLICATION"
            )
        if (
            _REPRODUCTION_RANK[
                self.evidence_requirement.minimum_reproduction_strength.value
            ]
            < _REPRODUCTION_RANK["r2"]
        ):
            raise ValueError(
                "headline quantitative claim requires typed reproduction "
                "strength r2 or above"
            )
        if not any(
            ref.role is SupportRole.RELEASE_SUPPORT
            and _REPRODUCTION_RANK[ref.reproduction_strength.value]
            >= _REPRODUCTION_RANK["r2"]
            for ref in self._qualifying_release_support()
        ):
            raise ValueError(
                "headline quantitative claim requires actual RELEASE_SUPPORT "
                "at r2 or above"
            )


@dataclass(frozen=True, slots=True)
class PaperCard(_Record):
    source_id: str
    reviewed_version_digest: str
    reviewer: str
    reviewed_at: str
    research_question: str
    wake_input: str
    trigger: str
    operator: str
    destination: str
    training_data: tuple[str, ...]
    objective: tuple[str, ...]
    optimizer: str
    update_location: str
    cadence: str
    capacity_policy: str
    systems_assumptions: tuple[str, ...]
    reported_results: tuple[str, ...]
    limitations: tuple[str, ...]
    failure_modes: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    counterevidence_ids: tuple[str, ...]
    code_review: str
    data_review: str
    non_claims: tuple[str, ...]

    _TUPLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "training_data",
            "objective",
            "systems_assumptions",
            "reported_results",
            "limitations",
            "failure_modes",
            "evidence_ids",
            "counterevidence_ids",
            "non_claims",
        }
    )

    def __post_init__(self) -> None:
        self._validate_declared_containers()
        _stable_id(self.source_id, "source_id", "source")
        _sha256(self.reviewed_version_digest, "reviewed_version_digest")
        object.__setattr__(
            self,
            "reviewed_at",
            _iso_date_or_datetime(self.reviewed_at, "reviewed_at"),
        )
        for field_name in (
            "reviewer",
            "research_question",
            "wake_input",
            "trigger",
            "operator",
            "destination",
            "optimizer",
            "update_location",
            "cadence",
            "capacity_policy",
            "code_review",
            "data_review",
        ):
            _nonempty(getattr(self, field_name), field_name)
        for evidence_id in (*self.evidence_ids, *self.counterevidence_ids):
            _stable_id(evidence_id, "evidence_ids", "evidence")


@dataclass(frozen=True, slots=True)
class QuestionRecord(_Record):
    question_id: str
    statement: str
    estimand: str
    null: str
    rejection_rule: str
    minimum_effect: str
    inconclusive_condition: str
    validity_failure: str
    owner: str
    artifact_dependencies: tuple[str, ...]
    paper_scope: str

    _TUPLE_FIELDS: ClassVar[frozenset[str]] = frozenset({"artifact_dependencies"})

    def __post_init__(self) -> None:
        self._validate_declared_containers()
        _stable_id(self.question_id, "question_id", "question")
        for field_name in (
            "statement",
            "estimand",
            "null",
            "rejection_rule",
            "minimum_effect",
            "inconclusive_condition",
            "validity_failure",
            "owner",
            "paper_scope",
        ):
            _nonempty(getattr(self, field_name), field_name)
        for artifact_id in self.artifact_dependencies:
            _stable_id(artifact_id, "artifact_dependencies", "artifact")


@dataclass(frozen=True, slots=True)
class HypothesisRecord(_Record):
    hypothesis_id: str
    statement: str
    evidence_ids: tuple[str, ...]
    evidence_anchor: str
    estimand: str
    null: str
    rejection_rule: str
    validity_failure: str
    owner: str
    artifact_dependencies: tuple[str, ...]
    assumptions: tuple[str, ...]
    derivation_artifact: str
    falsifier: str
    confirmatory_cells: tuple[str, ...]
    amendment_history: tuple[Mapping[str, Any], ...]

    _TUPLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "evidence_ids",
            "artifact_dependencies",
            "assumptions",
            "confirmatory_cells",
        }
    )
    _TUPLE_MAPPING_FIELDS: ClassVar[frozenset[str]] = frozenset({"amendment_history"})

    def __post_init__(self) -> None:
        self._validate_declared_containers()
        _stable_id(self.hypothesis_id, "hypothesis_id", "hypothesis")
        for evidence_id in self.evidence_ids:
            _stable_id(evidence_id, "evidence_ids", "evidence")
        for artifact_id in self.artifact_dependencies:
            _stable_id(artifact_id, "artifact_dependencies", "artifact")
        _stable_id(self.derivation_artifact, "derivation_artifact", "artifact")
        for field_name in (
            "statement",
            "evidence_anchor",
            "estimand",
            "null",
            "rejection_rule",
            "validity_failure",
            "owner",
            "falsifier",
        ):
            _nonempty(getattr(self, field_name), field_name)
        if not self.evidence_ids:
            raise ValueError("evidence_ids must not be empty")
        if not self.artifact_dependencies:
            raise ValueError("artifact_dependencies must not be empty")


@dataclass(frozen=True, slots=True)
class ArtifactNode(_Record):
    artifact_id: str
    schema_version: str
    artifact_type: str
    path: str
    input_digests: tuple[str, ...]
    producer_command: str
    output_digest: str
    consumers: tuple[str, ...]
    claim_ids: tuple[str, ...]
    question_ids: tuple[str, ...]
    gate_status: str
    owner: str
    frozen_release_tag: str | None

    _TUPLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"input_digests", "consumers", "claim_ids", "question_ids"}
    )

    def __post_init__(self) -> None:
        self._validate_declared_containers()
        _stable_id(self.artifact_id, "artifact_id", "artifact")
        _relative_path(self.path, "path")
        for digest in self.input_digests:
            _sha256(digest, "input_digests")
        _sha256(self.output_digest, "output_digest")
        for consumer_id in self.consumers:
            _stable_id(consumer_id, "consumers", "artifact")
        for claim_id in self.claim_ids:
            _stable_id(claim_id, "claim_ids", "claim", "paper_claim")
        for question_id in self.question_ids:
            _stable_id(question_id, "question_ids", "question")
        for field_name in (
            "schema_version",
            "artifact_type",
            "producer_command",
            "gate_status",
            "owner",
        ):
            _nonempty(getattr(self, field_name), field_name)
        if self.frozen_release_tag is not None:
            _nonempty(self.frozen_release_tag, "frozen_release_tag")


@dataclass(frozen=True, slots=True)
class DigestRef(_Record):
    artifact_id: str
    sha256: str

    def __post_init__(self) -> None:
        self._validate_declared_containers()
        _stable_id(self.artifact_id, "artifact_id", "artifact")
        _sha256(self.sha256, "sha256")


@dataclass(frozen=True, slots=True)
class GateArtifactBinding:
    artifact_id: str
    relative_path: str
    resolved_path: Path
    declared_sha256: str
    loaded_sha256: str

    def __post_init__(self) -> None:
        _stable_id(self.artifact_id, "artifact_id", "artifact")
        _relative_path(self.relative_path, "relative_path")
        if not isinstance(self.resolved_path, Path) or not self.resolved_path.is_absolute():
            raise ValueError("resolved_path must be an absolute pathlib.Path")
        _sha256(self.declared_sha256, "declared_sha256")
        _sha256(self.loaded_sha256, "loaded_sha256")


@dataclass(frozen=True, slots=True)
class GateAuthorizationSnapshot:
    gate_input_sha256: str
    artifact_registry_sha256: str
    artifact_bindings: tuple[GateArtifactBinding, ...]

    def __post_init__(self) -> None:
        _sha256(self.gate_input_sha256, "gate_input_sha256")
        _sha256(
            self.artifact_registry_sha256,
            "artifact_registry_sha256",
        )
        if not isinstance(self.artifact_bindings, tuple):
            raise TypeError("artifact_bindings must be an immutable tuple")
        if any(
            type(binding) is not GateArtifactBinding
            for binding in self.artifact_bindings
        ):
            raise TypeError(
                "artifact_bindings entries must be exactly GateArtifactBinding"
            )


@dataclass(frozen=True, slots=True)
class TrustedKey(_Record):
    key_id: str
    signer_id: str
    algorithm: str
    public_key: str
    roles: tuple[str, ...]
    valid_from: str
    valid_until: str | None
    revoked: bool

    _TUPLE_FIELDS: ClassVar[frozenset[str]] = frozenset({"roles"})

    def __post_init__(self) -> None:
        self._validate_declared_containers()
        for field_name in (
            "key_id",
            "signer_id",
            "algorithm",
            "public_key",
        ):
            _nonempty(getattr(self, field_name), field_name)
        if not self.roles:
            raise ValueError("roles must not be empty")
        if len(self.roles) != len(set(self.roles)):
            raise ValueError("roles must contain unique values")
        object.__setattr__(
            self,
            "valid_from",
            _aware_datetime(self.valid_from, "valid_from"),
        )
        if self.valid_until is not None:
            object.__setattr__(
                self,
                "valid_until",
                _aware_datetime(self.valid_until, "valid_until"),
            )
            valid_from = datetime.fromisoformat(self.valid_from)
            valid_until = datetime.fromisoformat(self.valid_until)
            if valid_until <= valid_from:
                raise ValueError("valid_until must be after valid_from")
        if type(self.revoked) is not bool:
            raise TypeError("revoked must be boolean")


@dataclass(frozen=True, slots=True)
class TrustedKeySet(_Record):
    key_set_id: str
    version: int
    keys: tuple[TrustedKey, ...]
    set_digest: str
    issued_at: str

    _TUPLE_NESTED_FIELDS: ClassVar[Mapping[str, type[_Record]]] = {
        "keys": TrustedKey,
    }

    def __post_init__(self) -> None:
        self._validate_declared_containers()
        _nonempty(self.key_set_id, "key_set_id")
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError("version must be a positive integer")
        if self.version < 1:
            raise ValueError("version must be a positive integer")
        if not self.keys:
            raise ValueError("keys must not be empty")
        key_ids = tuple(key.key_id for key in self.keys)
        if len(key_ids) != len(set(key_ids)):
            raise ValueError("key_id values must be unique")
        public_keys = tuple(key.public_key for key in self.keys)
        if len(public_keys) != len(set(public_keys)):
            raise ValueError("public_key values must be unique")
        _sha256(self.set_digest, "set_digest")
        object.__setattr__(
            self,
            "issued_at",
            _aware_datetime(self.issued_at, "issued_at"),
        )
        unsigned_payload = {
            "key_set_id": self.key_set_id,
            "version": self.version,
            "keys": [key.to_dict() for key in self.keys],
            "issued_at": self.issued_at,
        }
        expected_digest = hashlib.sha256(
            _canonical_json_bytes(unsigned_payload)
        ).hexdigest()
        if self.set_digest != expected_digest:
            raise ValueError("set_digest does not match canonical key-set bytes")


@dataclass(frozen=True, slots=True)
class GateEvaluationInput(_Record):
    gate_id: str
    gate_artifact_id: str
    schema_version: str
    evaluation_commit_sha: str
    evaluation_tree_digest: str
    scientific_candidate_sha: str | None
    scientific_candidate_tree_digest: str | None
    artifact_dag_digest: str
    input_refs: tuple[DigestRef, ...]
    evaluator_id: str
    evaluator_role: str
    independence_mode: str
    evaluator_attestation_ref: DigestRef | None
    finding_ids: tuple[str, ...]
    adjudication_ids: tuple[str, ...]
    reaudit_refs: tuple[DigestRef, ...]
    environment_digest: str
    evaluated_at: str
    status: GateStatus
    repository_root: Path
    input_path: Path
    output_path: Path
    authorization_snapshot: GateAuthorizationSnapshot

    _SERIALIZED_FIELDS: ClassVar[tuple[str, ...]] = (
        "gate_id",
        "gate_artifact_id",
        "schema_version",
        "evaluation_commit_sha",
        "evaluation_tree_digest",
        "scientific_candidate_sha",
        "scientific_candidate_tree_digest",
        "artifact_dag_digest",
        "input_refs",
        "evaluator_id",
        "evaluator_role",
        "independence_mode",
        "evaluator_attestation_ref",
        "finding_ids",
        "adjudication_ids",
        "reaudit_refs",
        "environment_digest",
        "evaluated_at",
        "status",
    )
    _ENUM_FIELDS: ClassVar[Mapping[str, type[StrEnum]]] = {
        "status": GateStatus,
    }
    _OPTIONAL_NESTED_FIELDS: ClassVar[Mapping[str, type[_Record]]] = {
        "evaluator_attestation_ref": DigestRef,
    }
    _TUPLE_NESTED_FIELDS: ClassVar[Mapping[str, type[_Record]]] = {
        "input_refs": DigestRef,
        "reaudit_refs": DigestRef,
    }
    _TUPLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"finding_ids", "adjudication_ids"}
    )

    def __post_init__(self) -> None:
        self._validate_declared_containers()
        for field_name in ("finding_ids", "adjudication_ids"):
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must contain unique values")
        if re.fullmatch(r"G[0-8]", self.gate_id) is None:
            raise ValueError("gate_id must be G0 through G8")
        _stable_id(self.gate_artifact_id, "gate_artifact_id", "artifact")
        for field_name in (
            "schema_version",
            "evaluator_id",
            "evaluator_role",
            "independence_mode",
        ):
            _nonempty(getattr(self, field_name), field_name)
        _git_sha1(self.evaluation_commit_sha, "evaluation_commit_sha")
        for field_name in (
            "evaluation_tree_digest",
            "artifact_dag_digest",
            "environment_digest",
        ):
            _sha256(getattr(self, field_name), field_name)
        if self.scientific_candidate_sha is not None:
            _git_sha1(
                self.scientific_candidate_sha,
                "scientific_candidate_sha",
            )
        if self.scientific_candidate_tree_digest is not None:
            _sha256(
                self.scientific_candidate_tree_digest,
                "scientific_candidate_tree_digest",
            )
        gate_number = int(self.gate_id[1:])
        candidate_fields = (
            self.scientific_candidate_sha,
            self.scientific_candidate_tree_digest,
        )
        if gate_number >= 7 and (
            any(value is None for value in candidate_fields)
            or self.evaluator_attestation_ref is None
        ):
            raise ValueError(
                "G7/G8 require scientific candidate SHA/tree and evaluator attestation"
            )
        if gate_number < 7 and (
            any(value is not None for value in candidate_fields)
            or self.evaluator_attestation_ref is not None
        ):
            raise ValueError(
                "scientific candidate fields and evaluator attestation must be null "
                "before G7"
            )
        object.__setattr__(
            self,
            "evaluated_at",
            _aware_datetime(self.evaluated_at, "evaluated_at"),
        )
        for field_name in ("repository_root", "input_path", "output_path"):
            path = getattr(self, field_name)
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"{field_name} must be an absolute pathlib.Path")
        for field_name in ("input_path", "output_path"):
            try:
                getattr(self, field_name).relative_to(self.repository_root)
            except ValueError as error:
                raise ValueError(
                    f"{field_name} must be contained by repository_root"
                ) from error
        if type(self.authorization_snapshot) is not GateAuthorizationSnapshot:
            raise TypeError(
                "authorization_snapshot must be exactly GateAuthorizationSnapshot"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            name: _thaw(getattr(self, name))
            for name in self._SERIALIZED_FIELDS
        }


@dataclass(frozen=True, slots=True)
class AuditAttestation(_Record):
    attestation_id: str
    subject_sha256: str
    subject_refs: tuple[SubjectRef, ...]
    signer_id: str
    signer_role: str
    independence_mode: str
    author_executor_roster_digest: str
    algorithm: str
    key_id: str
    signature: str
    signed_at: str

    _TUPLE_NESTED_FIELDS: ClassVar[Mapping[str, type[_Record]]] = {
        "subject_refs": SubjectRef,
    }

    def __post_init__(self) -> None:
        self._validate_declared_containers()
        _validate_attestation_common(self)
        _sha256(
            self.author_executor_roster_digest,
            "author_executor_roster_digest",
        )


@dataclass(frozen=True, slots=True)
class GateEvaluationAttestation(_Record):
    attestation_id: str
    gate_id: str
    subject_sha256: str
    subject_refs: tuple[SubjectRef, ...]
    signer_id: str
    signer_role: str
    independence_mode: str
    algorithm: str
    key_id: str
    signature: str
    signed_at: str

    _TUPLE_NESTED_FIELDS: ClassVar[Mapping[str, type[_Record]]] = {
        "subject_refs": SubjectRef,
    }

    def __post_init__(self) -> None:
        self._validate_declared_containers()
        _validate_attestation_common(self)
        if re.fullmatch(r"G[0-8]", self.gate_id) is None:
            raise ValueError("gate_id must be G0 through G8")


def _validate_attestation_common(
    record: AuditAttestation | GateEvaluationAttestation,
) -> None:
    _nonempty(record.attestation_id, "attestation_id")
    _sha256(record.subject_sha256, "subject_sha256")
    if not record.subject_refs:
        raise ValueError("subject_refs must not be empty")
    object.__setattr__(
        record,
        "subject_refs",
        tuple(sorted(record.subject_refs, key=lambda ref: ref.path)),
    )
    paths = [ref.path for ref in record.subject_refs]
    if len(paths) != len(set(paths)):
        raise ValueError("subject_refs contain duplicate paths")
    for field_name in (
        "signer_id",
        "signer_role",
        "independence_mode",
        "algorithm",
        "key_id",
        "signature",
    ):
        _nonempty(getattr(record, field_name), field_name)
    object.__setattr__(
        record,
        "signed_at",
        _aware_datetime(record.signed_at, "signed_at"),
    )


@dataclass(frozen=True, slots=True)
class HumanApproval(_Record):
    approval_id: str
    subject_sha256: str
    pdf_digests: tuple[str, ...]
    machine_visual_report_digest: str
    contact_sheet_digest: str
    all_pages_reviewed: bool
    review_checks: Mapping[str, str]
    disposition: str
    open_issues: tuple[str, ...]
    signer_id: str
    signer_role: str
    algorithm: str
    key_id: str
    signature: str
    signed_at: str

    _TUPLE_FIELDS: ClassVar[frozenset[str]] = frozenset({"pdf_digests", "open_issues"})
    _MAPPING_FIELDS: ClassVar[frozenset[str]] = frozenset({"review_checks"})

    def __post_init__(self) -> None:
        self._validate_declared_containers()
        _nonempty(self.approval_id, "approval_id")
        _sha256(self.subject_sha256, "subject_sha256")
        for digest in self.pdf_digests:
            _sha256(digest, "pdf_digests")
        _sha256(
            self.machine_visual_report_digest,
            "machine_visual_report_digest",
        )
        _sha256(self.contact_sheet_digest, "contact_sheet_digest")
        if not isinstance(self.all_pages_reviewed, bool):
            raise TypeError("all_pages_reviewed must be boolean")
        if not self.review_checks:
            raise ValueError("review_checks must not be empty")
        for key, value in self.review_checks.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError("review_checks keys and values must be strings")
            if not key.strip() or not value.strip():
                raise ValueError(
                    "review_checks keys and values must be non-empty strings"
                )
        for field_name in (
            "disposition",
            "signer_id",
            "signer_role",
            "algorithm",
            "key_id",
            "signature",
        ):
            _nonempty(getattr(self, field_name), field_name)
        object.__setattr__(
            self,
            "signed_at",
            _aware_datetime(self.signed_at, "signed_at"),
        )


@dataclass(frozen=True, slots=True)
class GateRecord(_Record):
    gate_id: str
    gate_artifact_id: str
    schema_version: str
    evaluation_commit_sha: str
    evaluation_tree_digest: str
    scientific_candidate_sha: str | None
    scientific_candidate_tree_digest: str | None
    artifact_dag_digest: str
    predecessor_gate_refs: tuple[DigestRef, ...]
    input_refs: tuple[DigestRef, ...]
    evaluator_id: str
    evaluator_role: str
    independence_mode: str
    evaluator_attestation_ref: DigestRef | None
    finding_ids: tuple[str, ...]
    adjudication_ids: tuple[str, ...]
    reaudit_refs: tuple[DigestRef, ...]
    environment_digest: str
    evaluated_at: str
    status: GateStatus

    _ENUM_FIELDS: ClassVar[Mapping[str, type[StrEnum]]] = {
        "status": GateStatus,
    }
    _OPTIONAL_NESTED_FIELDS: ClassVar[Mapping[str, type[_Record]]] = {
        "evaluator_attestation_ref": DigestRef,
    }
    _TUPLE_NESTED_FIELDS: ClassVar[Mapping[str, type[_Record]]] = {
        "predecessor_gate_refs": DigestRef,
        "input_refs": DigestRef,
        "reaudit_refs": DigestRef,
    }
    _TUPLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"finding_ids", "adjudication_ids"}
    )

    def __post_init__(self) -> None:
        self._validate_declared_containers()
        for field_name in ("finding_ids", "adjudication_ids"):
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must contain unique values")
        if re.fullmatch(r"G[0-8]", self.gate_id) is None:
            raise ValueError("gate_id must be G0 through G8")
        _stable_id(self.gate_artifact_id, "gate_artifact_id", "artifact")
        for field_name in (
            "schema_version",
            "evaluator_id",
            "evaluator_role",
            "independence_mode",
        ):
            _nonempty(getattr(self, field_name), field_name)
        _git_sha1(self.evaluation_commit_sha, "evaluation_commit_sha")
        for field_name in (
            "evaluation_tree_digest",
            "artifact_dag_digest",
            "environment_digest",
        ):
            _sha256(getattr(self, field_name), field_name)
        if self.scientific_candidate_sha is not None:
            _git_sha1(
                self.scientific_candidate_sha,
                "scientific_candidate_sha",
            )
        if self.scientific_candidate_tree_digest is not None:
            _sha256(
                self.scientific_candidate_tree_digest,
                "scientific_candidate_tree_digest",
            )
        gate_number = int(self.gate_id[1:])
        candidate_fields = (
            self.scientific_candidate_sha,
            self.scientific_candidate_tree_digest,
        )
        if gate_number >= 7 and (
            any(value is None for value in candidate_fields)
            or self.evaluator_attestation_ref is None
        ):
            raise ValueError(
                "G7/G8 require scientific candidate SHA/tree and evaluator attestation"
            )
        if gate_number < 7 and (
            any(value is not None for value in candidate_fields)
            or self.evaluator_attestation_ref is not None
        ):
            raise ValueError(
                "scientific candidate fields and evaluator attestation must be null "
                "before G7"
            )
        object.__setattr__(
            self,
            "evaluated_at",
            _aware_datetime(self.evaluated_at, "evaluated_at"),
        )
