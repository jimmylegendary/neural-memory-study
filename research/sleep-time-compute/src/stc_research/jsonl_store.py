from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from fcntl import LOCK_EX, LOCK_UN, flock
from pathlib import Path
from typing import Any, Protocol

from stc_research.ids import parse_stable_id


class DuplicateIdError(ValueError):
    pass


class RegistryIntegrityError(ValueError):
    pass


class RetryConflictError(RuntimeError):
    pass


class StaleWriteError(RuntimeError):
    pass


class CommitOutcomeUnknownError(OSError):
    safe_to_retry = True

    def __init__(self, destination: Path, retry_token: str) -> None:
        self.destination = destination
        self.retry_token = retry_token
        super().__init__(
            f"atomic replace completed for {destination}, but directory fsync "
            "failed; commit durability is unknown and an exact-token retry is safe"
        )


class SupportsToDict(Protocol):
    def to_dict(self) -> Mapping[str, Any]: ...


type JsonlRecord = Mapping[str, Any] | SupportsToDict
type CanonicalJson = (
    dict[str, CanonicalJson] | list[CanonicalJson] | str | int | float | bool | None
)

_PRIMARY_ID_KINDS: Mapping[str, frozenset[str]] = {
    "source_id": frozenset({"source"}),
    "evidence_id": frozenset({"evidence"}),
    "claim_id": frozenset({"claim", "paper_claim"}),
    "hypothesis_id": frozenset({"hypothesis"}),
    "question_id": frozenset({"question"}),
    "artifact_id": frozenset({"artifact"}),
    "experiment_id": frozenset({"experiment"}),
    "result_id": frozenset({"result"}),
    "figure_id": frozenset({"figure"}),
    "table_id": frozenset({"table"}),
}
_DEFAULT_REGISTRY_MODE = 0o644
_REGISTRY_PERMISSION_MASK = 0o777


class _StrictJsonError(ValueError):
    pass


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _StrictJsonError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(constant: str) -> None:
    raise _StrictJsonError(f"non-finite JSON constant {constant!r}")


def _strict_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise _StrictJsonError(f"non-finite JSON number {value!r}")
    return parsed


def read_jsonl[T](path: str | Path, factory: Callable[[dict[str, Any]], T]) -> list[T]:
    source = Path(path)
    records: list[T] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(
                    line,
                    object_pairs_hook=_strict_object,
                    parse_constant=_reject_nonfinite_constant,
                    parse_float=_strict_float,
                )
            except json.JSONDecodeError as error:
                raise ValueError(f"{source}:{line_number}: invalid JSON") from error
            except _StrictJsonError as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error
            if not isinstance(payload, dict):
                raise TypeError(
                    f"{source}:{line_number}: JSONL record must be an object"
                )
            records.append(factory(payload))
    return records


def write_jsonl_atomic(
    path: str | Path,
    records: Iterable[JsonlRecord],
    *,
    expected_current_digest: str | None = None,
    retry_token: str | None = None,
) -> None:
    destination = Path(path)
    _validate_writer_destination(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_payload(records)
    expected_retry_token = _operation_retry_token(
        destination,
        "write",
        payload,
    )
    _validate_retry_token(retry_token, expected_retry_token)
    _validate_expected_current_digest(expected_current_digest)

    with _destination_lock(destination):
        if retry_token is not None and destination.exists():
            if destination.read_text(encoding="utf-8") != payload:
                raise RetryConflictError(
                    f"{destination} contains newer state; exact write retry refused"
                )
            _fsync_retry(destination, expected_retry_token)
            return
        if not destination.exists():
            if expected_current_digest is not None:
                raise StaleWriteError(
                    f"{destination} has no current state for expected-current digest"
                )
            _write_jsonl_atomic_unlocked(
                destination,
                payload,
                expected_retry_token,
            )
            return

        current_payload = _canonical_existing_payload(destination)
        if current_payload == payload:
            return
        if expected_current_digest is None:
            raise StaleWriteError(
                f"changing {destination} requires an expected-current canonical digest"
            )
        actual_current_digest = _payload_digest(current_payload)
        if not hmac.compare_digest(
            expected_current_digest,
            actual_current_digest,
        ):
            raise StaleWriteError(
                f"{destination} has a stale expected-current canonical digest"
            )
        _write_jsonl_atomic_unlocked(
            destination,
            payload,
            expected_retry_token,
        )


def append_unique(
    path: str | Path,
    record: JsonlRecord,
    id_field: str,
    *,
    retry_token: str | None = None,
) -> None:
    destination = Path(path)
    _validate_writer_destination(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _record_snapshot(record)
    record_line = _canonical_snapshot_line(payload)
    record_id = _validated_primary_id(
        payload,
        id_field,
        context="new record",
    )
    expected_retry_token = _operation_retry_token(
        destination,
        f"append:{id_field}",
        record_line,
    )
    _validate_retry_token(retry_token, expected_retry_token)

    with _destination_lock(destination):
        existing = (
            read_jsonl(destination, lambda item: item) if destination.exists() else []
        )
        rows_by_id = _validate_existing_registry(existing, id_field)
        duplicate = rows_by_id.get(record_id)
        if duplicate is not None:
            if retry_token is not None:
                if _canonical_snapshot_line(duplicate) != record_line:
                    raise RetryConflictError(
                        f"{destination} retry is not the exact canonical "
                        f"record for {id_field} {record_id}"
                    )
                _fsync_retry(destination, expected_retry_token)
                return
            raise DuplicateIdError(f"duplicate {id_field}: {record_id}")

        existing.append(payload)
        _write_jsonl_atomic_unlocked(
            destination,
            _canonical_payload(existing),
            expected_retry_token,
        )


def _validate_existing_registry(
    records: list[dict[str, Any]],
    id_field: str,
) -> dict[str, dict[str, Any]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row_number, payload in enumerate(records, start=1):
        record_id = _validated_primary_id(
            payload,
            id_field,
            context=f"preexisting row {row_number}",
        )
        if record_id in rows_by_id:
            raise RegistryIntegrityError(
                f"duplicate preexisting {id_field}: {record_id}"
            )
        rows_by_id[record_id] = payload
    return rows_by_id


def _validated_primary_id(
    payload: Mapping[str, Any],
    id_field: str,
    *,
    context: str,
) -> str:
    if id_field not in payload:
        raise RegistryIntegrityError(f"{context} must contain exactly one {id_field}")
    record_id = payload[id_field]
    try:
        parsed = parse_stable_id(record_id)
    except (TypeError, ValueError) as error:
        raise RegistryIntegrityError(
            f"{context} must contain a valid canonical {id_field}"
        ) from error
    allowed_kinds = _PRIMARY_ID_KINDS.get(id_field)
    if allowed_kinds is not None and parsed.kind not in allowed_kinds:
        expected = " or ".join(sorted(allowed_kinds))
        raise RegistryIntegrityError(
            f"{context} {id_field} must have stable ID kind {expected}"
        )
    return record_id


def _record_mapping(record: JsonlRecord) -> Mapping[str, Any]:
    if isinstance(record, Mapping):
        return record
    to_dict = getattr(record, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return payload
    raise TypeError("JSONL record must be a mapping or expose to_dict()")


def _record_snapshot(record: JsonlRecord) -> dict[str, CanonicalJson]:
    snapshot = _canonical_json_snapshot(_record_mapping(record), "record")
    if not isinstance(snapshot, dict):
        raise TypeError("JSONL record must snapshot to a canonical JSON object")
    return snapshot


def _canonical_json_snapshot(value: Any, field_path: str) -> CanonicalJson:
    if isinstance(value, Mapping):
        snapshot: dict[str, CanonicalJson] = {}
        for key in sorted(value, key=lambda item: str(item)):
            if type(key) is not str:
                raise TypeError(
                    f"{field_path} canonical JSON object keys must be strings"
                )
            snapshot[key] = _canonical_json_snapshot(
                value[key],
                f"{field_path}.{key}",
            )
        return snapshot
    if isinstance(value, (list, tuple)):
        return [
            _canonical_json_snapshot(item, f"{field_path}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or type(value) in {str, int, bool}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field_path} canonical JSON numbers must be finite")
        return value
    raise TypeError(
        f"{field_path} contains unsupported canonical JSON value {type(value).__name__}"
    )


def _canonical_line(record: JsonlRecord) -> str:
    return _canonical_snapshot_line(_record_snapshot(record))


def _canonical_snapshot_line(record: Mapping[str, CanonicalJson]) -> str:
    return json.dumps(
        record,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_payload(records: Iterable[JsonlRecord]) -> str:
    lines = sorted(_canonical_line(record) for record in records)
    payload = "\n".join(lines)
    return f"{payload}\n" if lines else ""


def canonical_jsonl_digest(records: Iterable[JsonlRecord]) -> str:
    return _payload_digest(_canonical_payload(records))


def _canonical_existing_payload(destination: Path) -> str:
    raw_payload = destination.read_text(encoding="utf-8")
    canonical_payload = _canonical_payload(read_jsonl(destination, lambda item: item))
    if raw_payload != canonical_payload:
        raise RegistryIntegrityError(
            f"{destination} is not canonical JSONL; bulk replacement refused"
        )
    return canonical_payload


def _payload_digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _operation_retry_token(
    destination: Path,
    operation: str,
    payload: str,
) -> str:
    digest = hashlib.sha256()
    for component in (
        "stc-jsonl-retry-v1",
        str(destination.absolute()),
        operation,
        payload,
    ):
        digest.update(component.encode("utf-8"))
        digest.update(b"\0")
    return f"stc-jsonl-retry-v1:{digest.hexdigest()}"


def _validate_retry_token(provided: str | None, expected: str) -> None:
    if provided is None:
        return
    if not isinstance(provided, str) or not hmac.compare_digest(provided, expected):
        raise RetryConflictError(
            "retry token does not match the exact canonical JSONL operation"
        )


def _validate_expected_current_digest(value: str | None) -> None:
    if value is None:
        return
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected_current_digest must be a lowercase SHA-256 digest")


@contextmanager
def _destination_lock(destination: Path) -> Iterator[None]:
    _validate_writer_destination(destination)
    lock_path = destination.with_name(f"{destination.name}.lock")
    _validate_existing_lock_sidecar(lock_path)
    flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        lock_descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise RegistryIntegrityError(
            f"{lock_path} coordination lock could not be opened safely"
        ) from error

    with ExitStack() as stack:
        stack.callback(os.close, lock_descriptor)
        try:
            lock_state = os.fstat(lock_descriptor)
        except OSError as error:
            raise RegistryIntegrityError(
                f"{lock_path} coordination lock could not be verified safely"
            ) from error
        _validate_lock_sidecar_state(lock_path, lock_state)
        lock_stream = stack.enter_context(
            os.fdopen(lock_descriptor, "r+b", closefd=False)
        )
        flock(lock_stream.fileno(), LOCK_EX)
        try:
            _registry_permissions(destination)
            _cleanup_stale_temporary_files(destination)
            yield
        finally:
            flock(lock_stream.fileno(), LOCK_UN)


def _validate_writer_destination(destination: Path) -> None:
    _reject_reserved_sidecar_component(destination.name, destination)
    parent = destination.parent
    while not os.path.lexists(parent):
        _reject_reserved_sidecar_component(parent.name, destination)
        next_parent = parent.parent
        if next_parent == parent:
            break
        parent = next_parent


def _reject_reserved_sidecar_component(name: str, destination: Path) -> None:
    if name.endswith(".lock") or name.startswith(".stc-jsonl-"):
        raise RegistryIntegrityError(
            f"{destination} uses reserved sidecar namespace component {name!r}"
        )


def _validate_existing_lock_sidecar(lock_path: Path) -> None:
    try:
        lock_state = lock_path.lstat()
    except FileNotFoundError:
        return
    _validate_lock_sidecar_state(lock_path, lock_state)


def _validate_lock_sidecar_state(
    lock_path: Path,
    lock_state: os.stat_result,
) -> None:
    if not stat.S_ISREG(lock_state.st_mode) or lock_state.st_size != 0:
        raise RegistryIntegrityError(
            f"{lock_path} coordination lock must be a zero-byte regular file"
        )


def _cleanup_stale_temporary_files(destination: Path) -> None:
    pattern = f"{_temporary_prefix(destination)}*.tmp"
    for temporary_path in destination.parent.glob(pattern):
        if not temporary_path.is_dir():
            temporary_path.unlink(missing_ok=True)


def _temporary_prefix(destination: Path) -> str:
    canonical_path = os.path.abspath(os.fspath(destination))
    digest = hashlib.sha256(os.fsencode(canonical_path)).hexdigest()
    return f".stc-jsonl-{digest}."


def _registry_permissions(destination: Path) -> int:
    try:
        mode = destination.lstat().st_mode
    except FileNotFoundError:
        return _DEFAULT_REGISTRY_MODE
    if not stat.S_ISREG(mode):
        raise RegistryIntegrityError(
            f"{destination} registry destination must be a regular file"
        )
    return stat.S_IMODE(mode) & _REGISTRY_PERMISSION_MASK


def _write_jsonl_atomic_unlocked(
    destination: Path,
    payload: str,
    retry_token: str,
) -> None:
    temporary_path: Path | None = None
    destination_mode = _registry_permissions(destination)
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=_temporary_prefix(destination),
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fchmod(temporary.fileno(), destination_mode)
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
        try:
            _fsync_directory(destination.parent)
        except OSError as error:
            raise CommitOutcomeUnknownError(
                destination,
                retry_token,
            ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _fsync_retry(destination: Path, retry_token: str) -> None:
    try:
        _fsync_directory(destination.parent)
    except OSError as error:
        raise CommitOutcomeUnknownError(destination, retry_token) from error


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
