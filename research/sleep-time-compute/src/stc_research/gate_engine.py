from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from stc_research.models import (
    ArtifactNode,
    AuditAttestation,
    DigestRef,
    GateArtifactBinding,
    GateAuthorizationSnapshot,
    GateEvaluationAttestation,
    GateEvaluationInput,
    GateRecord,
    GateStatus,
    HumanApproval,
    TrustedKey,
    TrustedKeySet,
)

_GATE_ID_RE = re.compile(r"G[0-8]")
_GATE_INPUT_FIELDS = frozenset(GateEvaluationInput._SERIALIZED_FIELDS)
_AUDITED_GATES = frozenset({"G7", "G8"})
_TRUSTED_KEY_SET_PATH = "manifests/trusted-reviewer-keys.json"
_ATTESTATION_PATHS = {
    "G7": "manifests/attestations/G7-auditor.json",
    "G8": "manifests/attestations/G8-evaluator.json",
}
_ATTESTATION_TYPES = {
    "G7": "audit-attestation",
    "G8": "gate-evaluation-attestation",
}
_REQUIRED_EVALUATOR_ROLES = {
    "G7": "independent_auditor",
    "G8": "release_evaluator",
}
_G8_SUBJECT_PATH = "manifests/gate-inputs/G8-evaluation-subject.json"
_ED25519_SPKI_PREFIX = bytes.fromhex("302a300506032b6570032100")
_AUDIT_ATTESTATION_FIELDS = frozenset(field.name for field in fields(AuditAttestation))
_HUMAN_APPROVAL_FIELDS = frozenset(field.name for field in fields(HumanApproval))


class GateEngineError(RuntimeError):
    def __init__(self, code: str, gate_id: str, path: str) -> None:
        self.code = code
        self.gate_id = gate_id
        self.path = path
        super().__init__(f"{code}: gate={gate_id} path={path}")


@dataclass(frozen=True, slots=True)
class GateChainDiagnostic:
    code: str
    gate_id: str
    path: str


@dataclass(frozen=True, slots=True)
class GateChainReport:
    diagnostics: tuple[GateChainDiagnostic, ...]
    stale_gate_ids: tuple[str, ...]

    @property
    def success(self) -> bool:
        return not self.diagnostics


class _DuplicateKeyError(ValueError):
    pass


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError
        result[key] = value
    return result


def _reject_json_constant(_: str) -> None:
    raise ValueError


def _strict_json_document(raw: bytes) -> object:
    text = raw.decode("utf-8", errors="strict")
    return json.loads(
        text,
        object_pairs_hook=_strict_object_pairs,
        parse_constant=_reject_json_constant,
    )


def _strict_jsonl_documents(raw: bytes) -> tuple[object, ...]:
    text = raw.decode("utf-8", errors="strict")
    records: list[object] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        records.append(
            json.loads(
                line,
                object_pairs_hook=_strict_object_pairs,
                parse_constant=_reject_json_constant,
            )
        )
    return tuple(records)


def _display_gate_input(gate_id: str) -> str:
    return f"manifests/gate-inputs/{gate_id}.json"


def _display_gate_output(gate_id: str) -> str:
    return f"manifests/gates/{gate_id}.json"


def _gate_number(gate_id: str) -> int:
    if _GATE_ID_RE.fullmatch(gate_id) is None:
        raise ValueError("gate_id must be G0 through G8")
    return int(gate_id[1:])


def _path_has_symlink(root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _relative_path_components(relative_path: str) -> tuple[str, ...] | None:
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or relative_path.startswith("/")
        or "\\" in relative_path
        or "\x00" in relative_path
    ):
        return None
    components = tuple(relative_path.split("/"))
    if any(component in {"", ".", ".."} for component in components):
        return None
    return components


def _read_rooted_regular_file(
    root: Path,
    relative_path: str,
    *,
    gate_id: str,
    invalid_code: str,
) -> bytes:
    components = _relative_path_components(relative_path)
    if components is None or not root.is_absolute():
        raise GateEngineError(invalid_code, gate_id, relative_path)

    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | close_on_exec
    leaf_flags = (
        os.O_RDONLY | os.O_NOFOLLOW | close_on_exec | getattr(os, "O_NONBLOCK", 0)
    )
    descriptors: list[int] = []
    try:
        root_descriptor = os.open(os.fspath(root), directory_flags)
        descriptors.append(root_descriptor)
        parent_descriptor = root_descriptor
        for component in components[:-1]:
            parent_descriptor = os.open(
                component,
                directory_flags,
                dir_fd=parent_descriptor,
            )
            descriptors.append(parent_descriptor)
        leaf_descriptor = os.open(
            components[-1],
            leaf_flags,
            dir_fd=parent_descriptor,
        )
        descriptors.append(leaf_descriptor)
        if not stat.S_ISREG(os.fstat(leaf_descriptor).st_mode):
            raise OSError("authorization target is not a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(leaf_descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError as error:
        raise GateEngineError(
            invalid_code,
            gate_id,
            relative_path,
        ) from error
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _load_gate_input_payload(
    root: Path,
    gate_id: str,
) -> tuple[dict[str, Any], bytes]:
    display_path = _display_gate_input(gate_id)
    raw = _read_rooted_regular_file(
        root,
        display_path,
        gate_id=gate_id,
        invalid_code="GATE_INPUT_INVALID",
    )
    try:
        payload = _strict_json_document(raw)
        if not isinstance(payload, dict):
            raise TypeError
        if set(payload) != _GATE_INPUT_FIELDS:
            raise ValueError
        if payload.get("gate_id") != gate_id:
            raise ValueError
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise GateEngineError("GATE_INPUT_INVALID", gate_id, display_path) from error
    return payload, raw


def _load_artifact_registry(
    root: Path,
    gate_id: str,
) -> tuple[tuple[ArtifactNode, ...], bytes]:
    display_path = "registry/artifacts.jsonl"
    raw = _read_rooted_regular_file(
        root,
        display_path,
        gate_id=gate_id,
        invalid_code="ARTIFACT_REGISTRY_INVALID",
    )
    try:
        payloads = _strict_jsonl_documents(raw)
        records = tuple(
            ArtifactNode.from_dict(payload)
            for payload in payloads
            if isinstance(payload, dict)
        )
        if len(records) != len(payloads):
            raise TypeError
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise GateEngineError(
            "ARTIFACT_REGISTRY_INVALID",
            gate_id,
            display_path,
        ) from error
    return records, raw


def _resolve_artifact_bindings(
    root: Path,
    gate_id: str,
    payload: dict[str, Any],
    registry: tuple[ArtifactNode, ...],
    *,
    deferred_paths: frozenset[str] = frozenset(),
) -> tuple[GateArtifactBinding, ...]:
    try:
        input_refs = tuple(DigestRef.from_dict(item) for item in payload["input_refs"])
    except (TypeError, ValueError) as error:
        raise GateEngineError(
            "GATE_INPUT_INVALID",
            gate_id,
            _display_gate_input(gate_id),
        ) from error

    bindings: list[GateArtifactBinding] = []
    bound_nodes: list[ArtifactNode] = []
    for input_ref in input_refs:
        matches = tuple(
            node for node in registry if node.artifact_id == input_ref.artifact_id
        )
        if not matches:
            raise GateEngineError(
                "INPUT_ARTIFACT_UNKNOWN",
                gate_id,
                _display_gate_input(gate_id),
            )
        if len(matches) != 1:
            raise GateEngineError(
                "INPUT_ARTIFACT_AMBIGUOUS",
                gate_id,
                "registry/artifacts.jsonl",
            )
        node = matches[0]
        if input_ref.sha256 != node.output_digest:
            raise GateEngineError(
                "INPUT_REF_DIGEST_MISMATCH",
                gate_id,
                _display_gate_input(gate_id),
            )

        artifact_path = root / node.path
        if node.path in deferred_paths:
            loaded_digest = node.output_digest
        else:
            artifact_bytes = _read_rooted_regular_file(
                root,
                node.path,
                gate_id=gate_id,
                invalid_code="INPUT_ARTIFACT_UNSAFE",
            )
            loaded_digest = _sha256(artifact_bytes)
            if loaded_digest != node.output_digest:
                raise GateEngineError(
                    "INPUT_ARTIFACT_BYTES_MISMATCH",
                    gate_id,
                    node.path,
                )
        bindings.append(
            GateArtifactBinding(
                artifact_id=node.artifact_id,
                relative_path=node.path,
                resolved_path=artifact_path,
                declared_sha256=node.output_digest,
                loaded_sha256=loaded_digest,
            )
        )
        bound_nodes.append(node)

    environment_digest = payload["environment_digest"]
    environment_matches = tuple(
        node
        for node in bound_nodes
        if node.artifact_type == "release-environment"
        and node.output_digest == environment_digest
    )
    if not environment_matches:
        raise GateEngineError(
            "ENVIRONMENT_DIGEST_UNBOUND",
            gate_id,
            _display_gate_input(gate_id),
        )
    if len(environment_matches) != 1:
        raise GateEngineError(
            "ENVIRONMENT_DIGEST_AMBIGUOUS",
            gate_id,
            _display_gate_input(gate_id),
        )

    return tuple(bindings)


def _bound_artifact_nodes(
    gate_id: str,
    registry: tuple[ArtifactNode, ...],
    bindings: tuple[GateArtifactBinding, ...],
) -> tuple[tuple[GateArtifactBinding, ArtifactNode], ...]:
    result: list[tuple[GateArtifactBinding, ArtifactNode]] = []
    for binding in bindings:
        matches = tuple(
            node for node in registry if node.artifact_id == binding.artifact_id
        )
        if len(matches) != 1:
            raise GateEngineError(
                "ARTIFACT_REGISTRY_INVALID",
                gate_id,
                "registry/artifacts.jsonl",
            )
        result.append((binding, matches[0]))
    return tuple(result)


def _load_canonical_authority_document(
    root: Path,
    gate_id: str,
    relative_path: str,
    *,
    expected_digest: str,
    invalid_code: str,
) -> tuple[dict[str, Any], bytes]:
    raw = _read_rooted_regular_file(
        root,
        relative_path,
        gate_id=gate_id,
        invalid_code=invalid_code,
    )
    try:
        payload = _strict_json_document(raw)
        if not isinstance(payload, dict):
            raise TypeError
        if _sha256(raw) != expected_digest:
            raise ValueError
        if raw != _canonical_json(payload):
            raise ValueError
    except (
        TypeError,
        ValueError,
        UnicodeError,
        json.JSONDecodeError,
    ) as error:
        raise GateEngineError(
            invalid_code,
            gate_id,
            relative_path,
        ) from error
    return payload, raw


def _strict_base64(
    value: object,
    *,
    expected_length: int | None = None,
) -> bytes:
    if not isinstance(value, str):
        raise TypeError
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError from error
    if base64.b64encode(decoded).decode("ascii") != value:
        raise ValueError
    if expected_length is not None and len(decoded) != expected_length:
        raise ValueError
    return decoded


def _ed25519_public_key_der(encoded: object) -> bytes:
    der = _strict_base64(encoded)
    if len(der) != len(_ED25519_SPKI_PREFIX) + 32 or not der.startswith(
        _ED25519_SPKI_PREFIX
    ):
        raise ValueError
    return der


def _verify_ed25519_signature(
    public_key_der: bytes,
    message: bytes,
    signature: bytes,
) -> bool:
    try:
        with tempfile.TemporaryDirectory(prefix="stc-ed25519-") as directory:
            temporary_root = Path(directory)
            public_key_path = temporary_root / "public-key.der"
            message_path = temporary_root / "message.json"
            signature_path = temporary_root / "signature.bin"
            public_key_path.write_bytes(public_key_der)
            message_path.write_bytes(message)
            signature_path.write_bytes(signature)
            completed = subprocess.run(
                [
                    "openssl",
                    "pkeyutl",
                    "-verify",
                    "-pubin",
                    "-keyform",
                    "DER",
                    "-inkey",
                    os.fspath(public_key_path),
                    "-rawin",
                    "-in",
                    os.fspath(message_path),
                    "-sigfile",
                    os.fspath(signature_path),
                ],
                check=False,
                capture_output=True,
                timeout=10,
            )
    except OSError, subprocess.SubprocessError:
        return False
    return completed.returncode == 0


def _parse_attestation(
    gate_id: str,
    relative_path: str,
    payload: dict[str, Any],
) -> AuditAttestation | GateEvaluationAttestation:
    if gate_id == "G8" and frozenset(payload) in {
        _AUDIT_ATTESTATION_FIELDS,
        _HUMAN_APPROVAL_FIELDS,
    }:
        raise GateEngineError(
            "EVALUATOR_ATTESTATION_TYPE_INVALID",
            gate_id,
            relative_path,
        )
    subject_refs = payload.get("subject_refs")
    if not isinstance(subject_refs, list):
        raise GateEngineError(
            "EVALUATOR_ATTESTATION_INVALID",
            gate_id,
            relative_path,
        )
    subject_paths = [
        ref.get("path") if isinstance(ref, dict) else None for ref in subject_refs
    ]
    if any(
        not isinstance(path, str) for path in subject_paths
    ) or subject_paths != sorted(subject_paths):
        raise GateEngineError(
            "EVALUATOR_ATTESTATION_INVALID",
            gate_id,
            relative_path,
        )

    try:
        if gate_id == "G7":
            return AuditAttestation.from_dict(payload)
        return GateEvaluationAttestation.from_dict(payload)
    except GateEngineError:
        raise
    except (TypeError, ValueError) as error:
        raise GateEngineError(
            "EVALUATOR_ATTESTATION_INVALID",
            gate_id,
            relative_path,
        ) from error


def _parse_trusted_key_set(
    gate_id: str,
    relative_path: str,
    payload: dict[str, Any],
) -> TrustedKeySet:
    try:
        return TrustedKeySet.from_dict(payload)
    except (TypeError, ValueError) as error:
        raise GateEngineError(
            "TRUSTED_KEY_SET_INVALID",
            gate_id,
            relative_path,
        ) from error


def _resolve_evaluator_authority(
    gate_input: GateEvaluationInput,
    registry: tuple[ArtifactNode, ...],
    bindings: tuple[GateArtifactBinding, ...],
) -> tuple[
    tuple[GateArtifactBinding, ArtifactNode],
    tuple[GateArtifactBinding, ArtifactNode],
]:
    gate_id = gate_input.gate_id
    input_path = _display_gate_input(gate_id)
    attestation_ref = gate_input.evaluator_attestation_ref
    if attestation_ref is None:
        raise GateEngineError(
            "EVALUATOR_ATTESTATION_UNBOUND",
            gate_id,
            input_path,
        )

    bound_nodes = _bound_artifact_nodes(gate_id, registry, bindings)
    attestation_candidates = tuple(
        pair
        for pair in bound_nodes
        if pair[0].artifact_id == attestation_ref.artifact_id
        and pair[0].declared_sha256 == attestation_ref.sha256
    )
    if not attestation_candidates:
        raise GateEngineError(
            "EVALUATOR_ATTESTATION_UNBOUND",
            gate_id,
            input_path,
        )
    if len(attestation_candidates) != 1:
        raise GateEngineError(
            "EVALUATOR_ATTESTATION_AMBIGUOUS",
            gate_id,
            input_path,
        )
    attestation_pair = attestation_candidates[0]

    trust_candidates = tuple(
        pair
        for pair in bound_nodes
        if pair[1].path == _TRUSTED_KEY_SET_PATH
        or pair[1].artifact_type == "trusted-key-set"
    )
    if not trust_candidates:
        raise GateEngineError(
            "TRUSTED_KEY_SET_UNBOUND",
            gate_id,
            input_path,
        )
    if len(trust_candidates) != 1:
        raise GateEngineError(
            "TRUSTED_KEY_SET_AMBIGUOUS",
            gate_id,
            input_path,
        )
    trust_pair = trust_candidates[0]

    attestation_node = attestation_pair[1]
    expected_attestation_path = _ATTESTATION_PATHS[gate_id]
    if attestation_node.path != expected_attestation_path:
        raise GateEngineError(
            "EVALUATOR_ATTESTATION_PATH_INVALID",
            gate_id,
            attestation_node.path,
        )
    if attestation_node.artifact_type != _ATTESTATION_TYPES[gate_id]:
        raise GateEngineError(
            "EVALUATOR_ATTESTATION_TYPE_INVALID",
            gate_id,
            attestation_node.path,
        )

    trust_node = trust_pair[1]
    if trust_node.path != _TRUSTED_KEY_SET_PATH:
        raise GateEngineError(
            "TRUSTED_KEY_SET_PATH_INVALID",
            gate_id,
            trust_node.path,
        )
    if trust_node.artifact_type != "trusted-key-set":
        raise GateEngineError(
            "TRUSTED_KEY_SET_TYPE_INVALID",
            gate_id,
            trust_node.path,
        )
    return attestation_pair, trust_pair


def _validate_attestation_subjects(
    gate_input: GateEvaluationInput,
    attestation: AuditAttestation | GateEvaluationAttestation,
    *,
    attestation_path: str,
    registry: tuple[ArtifactNode, ...],
    bindings: tuple[GateArtifactBinding, ...],
) -> None:
    gate_id = gate_input.gate_id
    bound_nodes = _bound_artifact_nodes(gate_id, registry, bindings)
    for subject_ref in attestation.subject_refs:
        if subject_ref.path == attestation_path:
            raise GateEngineError(
                "ATTESTATION_SUBJECT_INVALID",
                gate_id,
                attestation_path,
            )
        matches = tuple(
            pair
            for pair in bound_nodes
            if pair[1].path == subject_ref.path
            and pair[1].output_digest == subject_ref.sha256
            and pair[0].declared_sha256 == subject_ref.sha256
            and pair[0].loaded_sha256 == subject_ref.sha256
        )
        if len(matches) != 1:
            raise GateEngineError(
                "ATTESTATION_SUBJECT_INVALID",
                gate_id,
                attestation_path,
            )

    if gate_id == "G7":
        subject_payload = {
            "subject_refs": [
                subject_ref.to_dict() for subject_ref in attestation.subject_refs
            ]
        }
        if _sha256(_canonical_json(subject_payload)) != attestation.subject_sha256:
            raise GateEngineError(
                "ATTESTATION_SUBJECT_INVALID",
                gate_id,
                attestation_path,
            )
        return

    distinguished_refs = tuple(
        subject_ref
        for subject_ref in attestation.subject_refs
        if subject_ref.path == _G8_SUBJECT_PATH
    )
    if (
        len(distinguished_refs) != 1
        or attestation.subject_sha256 != distinguished_refs[0].sha256
    ):
        raise GateEngineError(
            "ATTESTATION_SUBJECT_INVALID",
            gate_id,
            attestation_path,
        )
    distinguished_nodes = tuple(
        pair
        for pair in bound_nodes
        if pair[1].path == _G8_SUBJECT_PATH
        and pair[1].output_digest == distinguished_refs[0].sha256
    )
    if len(distinguished_nodes) != 1:
        raise GateEngineError(
            "ATTESTATION_SUBJECT_INVALID",
            gate_id,
            attestation_path,
        )
    distinguished_node = distinguished_nodes[0][1]
    if distinguished_node.artifact_type != "gate-evaluation-subject":
        raise GateEngineError(
            "ATTESTATION_SUBJECT_INVALID",
            gate_id,
            _G8_SUBJECT_PATH,
        )
    subject_payload, _ = _load_canonical_authority_document(
        gate_input.repository_root,
        gate_id,
        _G8_SUBJECT_PATH,
        expected_digest=distinguished_node.output_digest,
        invalid_code="ATTESTATION_SUBJECT_INVALID",
    )
    if (
        set(subject_payload) != {"candidate", "gate_id", "review_scope"}
        or subject_payload.get("gate_id") != "G8"
        or any(
            not isinstance(subject_payload.get(field_name), str)
            or not subject_payload[field_name].strip()
            for field_name in ("candidate", "review_scope")
        )
    ):
        raise GateEngineError(
            "ATTESTATION_SUBJECT_INVALID",
            gate_id,
            _G8_SUBJECT_PATH,
        )


def _attestation_key(
    gate_id: str,
    attestation: AuditAttestation | GateEvaluationAttestation,
    trusted_key_set: TrustedKeySet,
    *,
    attestation_path: str,
) -> TrustedKey:
    matches = tuple(
        key for key in trusted_key_set.keys if key.key_id == attestation.key_id
    )
    if len(matches) != 1:
        raise GateEngineError(
            "ATTESTATION_KEY_INVALID",
            gate_id,
            attestation_path,
        )
    key = matches[0]
    if key.revoked or key.signer_id != attestation.signer_id:
        raise GateEngineError(
            "ATTESTATION_KEY_INVALID",
            gate_id,
            _TRUSTED_KEY_SET_PATH,
        )
    if key.algorithm != "Ed25519":
        raise GateEngineError(
            "ATTESTATION_KEY_INVALID",
            gate_id,
            _TRUSTED_KEY_SET_PATH,
        )
    if attestation.algorithm != "Ed25519":
        raise GateEngineError(
            "ATTESTATION_KEY_INVALID",
            gate_id,
            attestation_path,
        )
    return key


def _validate_attestation_identity_and_role(
    gate_input: GateEvaluationInput,
    attestation: AuditAttestation | GateEvaluationAttestation,
    trusted_key_set: TrustedKeySet,
    key: TrustedKey,
    *,
    attestation_path: str,
) -> None:
    gate_id = gate_input.gate_id
    if (
        gate_input.evaluator_role != attestation.signer_role
        or gate_input.independence_mode != attestation.independence_mode
        or gate_input.evaluator_id
        not in {trusted_key.signer_id for trusted_key in trusted_key_set.keys}
    ):
        raise GateEngineError(
            "ATTESTATION_IDENTITY_MISMATCH",
            gate_id,
            _display_gate_input(gate_id),
        )
    if (
        isinstance(attestation, GateEvaluationAttestation)
        and attestation.gate_id != gate_id
    ):
        raise GateEngineError(
            "ATTESTATION_IDENTITY_MISMATCH",
            gate_id,
            attestation_path,
        )
    required_role = _REQUIRED_EVALUATOR_ROLES[gate_id]
    if attestation.signer_role != required_role:
        raise GateEngineError(
            "ATTESTATION_ROLE_INVALID",
            gate_id,
            attestation_path,
        )
    if required_role not in key.roles:
        raise GateEngineError(
            "ATTESTATION_ROLE_INVALID",
            gate_id,
            _TRUSTED_KEY_SET_PATH,
        )


def _validate_attestation_time(
    gate_input: GateEvaluationInput,
    attestation: AuditAttestation | GateEvaluationAttestation,
    trusted_key_set: TrustedKeySet,
    key: TrustedKey,
    *,
    attestation_path: str,
) -> None:
    gate_id = gate_input.gate_id
    signed_at = datetime.fromisoformat(attestation.signed_at)
    issued_at = datetime.fromisoformat(trusted_key_set.issued_at)
    valid_from = datetime.fromisoformat(key.valid_from)
    valid_until = (
        None if key.valid_until is None else datetime.fromisoformat(key.valid_until)
    )
    evaluated_at = datetime.fromisoformat(gate_input.evaluated_at)
    if (
        issued_at > signed_at
        or valid_from > signed_at
        or (valid_until is not None and signed_at > valid_until)
    ):
        raise GateEngineError(
            "ATTESTATION_TIME_INVALID",
            gate_id,
            _TRUSTED_KEY_SET_PATH,
        )
    if signed_at > evaluated_at:
        raise GateEngineError(
            "ATTESTATION_TIME_INVALID",
            gate_id,
            attestation_path,
        )


def _validate_attestation_signature(
    gate_id: str,
    payload: dict[str, Any],
    key: TrustedKey,
    *,
    attestation_path: str,
) -> None:
    try:
        signature = _strict_base64(
            payload["signature"],
            expected_length=64,
        )
    except KeyError, TypeError, ValueError:
        raise GateEngineError(
            "ATTESTATION_SIGNATURE_INVALID",
            gate_id,
            attestation_path,
        ) from None
    try:
        public_key_der = _ed25519_public_key_der(key.public_key)
    except TypeError, ValueError:
        raise GateEngineError(
            "ATTESTATION_KEY_INVALID",
            gate_id,
            _TRUSTED_KEY_SET_PATH,
        ) from None

    unsigned_payload = dict(payload)
    unsigned_payload.pop("signature", None)
    if not _verify_ed25519_signature(
        public_key_der,
        _canonical_json(unsigned_payload),
        signature,
    ):
        raise GateEngineError(
            "ATTESTATION_SIGNATURE_INVALID",
            gate_id,
            attestation_path,
        )


def _verify_audited_gate_authorization(
    gate_input: GateEvaluationInput,
    registry: tuple[ArtifactNode, ...],
    bindings: tuple[GateArtifactBinding, ...],
) -> None:
    gate_id = gate_input.gate_id
    if gate_id not in _AUDITED_GATES:
        return
    attestation_pair, trust_pair = _resolve_evaluator_authority(
        gate_input,
        registry,
        bindings,
    )
    attestation_node = attestation_pair[1]
    trust_node = trust_pair[1]
    attestation_payload, _ = _load_canonical_authority_document(
        gate_input.repository_root,
        gate_id,
        attestation_node.path,
        expected_digest=attestation_node.output_digest,
        invalid_code="EVALUATOR_ATTESTATION_INVALID",
    )
    trust_payload, _ = _load_canonical_authority_document(
        gate_input.repository_root,
        gate_id,
        trust_node.path,
        expected_digest=trust_node.output_digest,
        invalid_code="TRUSTED_KEY_SET_INVALID",
    )
    attestation = _parse_attestation(
        gate_id,
        attestation_node.path,
        attestation_payload,
    )
    trusted_key_set = _parse_trusted_key_set(
        gate_id,
        trust_node.path,
        trust_payload,
    )

    if (
        gate_input.evaluator_role != attestation.signer_role
        or gate_input.independence_mode != attestation.independence_mode
    ):
        raise GateEngineError(
            "ATTESTATION_IDENTITY_MISMATCH",
            gate_id,
            _display_gate_input(gate_id),
        )
    if (
        isinstance(attestation, GateEvaluationAttestation)
        and attestation.gate_id != gate_id
    ):
        raise GateEngineError(
            "ATTESTATION_IDENTITY_MISMATCH",
            gate_id,
            attestation_node.path,
        )
    required_role = _REQUIRED_EVALUATOR_ROLES[gate_id]
    if attestation.signer_role != required_role:
        raise GateEngineError(
            "ATTESTATION_ROLE_INVALID",
            gate_id,
            attestation_node.path,
        )

    _validate_attestation_subjects(
        gate_input,
        attestation,
        attestation_path=attestation_node.path,
        registry=registry,
        bindings=bindings,
    )
    key = _attestation_key(
        gate_id,
        attestation,
        trusted_key_set,
        attestation_path=attestation_node.path,
    )
    _validate_attestation_identity_and_role(
        gate_input,
        attestation,
        trusted_key_set,
        key,
        attestation_path=attestation_node.path,
    )
    _validate_attestation_time(
        gate_input,
        attestation,
        trusted_key_set,
        key,
        attestation_path=attestation_node.path,
    )
    _validate_attestation_signature(
        gate_id,
        attestation_payload,
        key,
        attestation_path=attestation_node.path,
    )


def _digest_refs(
    values: object,
    *,
    gate_id: str,
) -> tuple[DigestRef, ...]:
    if not isinstance(values, list):
        raise GateEngineError(
            "GATE_INPUT_INVALID",
            gate_id,
            _display_gate_input(gate_id),
        )
    try:
        return tuple(DigestRef.from_dict(value) for value in values)
    except (TypeError, ValueError) as error:
        raise GateEngineError(
            "GATE_INPUT_INVALID",
            gate_id,
            _display_gate_input(gate_id),
        ) from error


def _string_tuple(
    value: object,
    *,
    gate_id: str,
) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise GateEngineError(
            "GATE_INPUT_INVALID",
            gate_id,
            _display_gate_input(gate_id),
        )
    return tuple(value)


def _build_gate_input(
    payload: dict[str, Any],
    *,
    root: Path,
    input_path: Path,
    output_path: Path,
    snapshot: GateAuthorizationSnapshot,
) -> GateEvaluationInput:
    gate_id = str(payload.get("gate_id", ""))
    try:
        evaluator_attestation = payload["evaluator_attestation_ref"]
        return GateEvaluationInput(
            gate_id=gate_id,
            gate_artifact_id=payload["gate_artifact_id"],
            schema_version=payload["schema_version"],
            evaluation_commit_sha=payload["evaluation_commit_sha"],
            evaluation_tree_digest=payload["evaluation_tree_digest"],
            scientific_candidate_sha=payload["scientific_candidate_sha"],
            scientific_candidate_tree_digest=payload[
                "scientific_candidate_tree_digest"
            ],
            artifact_dag_digest=payload["artifact_dag_digest"],
            input_refs=_digest_refs(payload["input_refs"], gate_id=gate_id),
            evaluator_id=payload["evaluator_id"],
            evaluator_role=payload["evaluator_role"],
            independence_mode=payload["independence_mode"],
            evaluator_attestation_ref=(
                None
                if evaluator_attestation is None
                else DigestRef.from_dict(evaluator_attestation)
            ),
            finding_ids=_string_tuple(
                payload["finding_ids"],
                gate_id=gate_id,
            ),
            adjudication_ids=_string_tuple(
                payload["adjudication_ids"],
                gate_id=gate_id,
            ),
            reaudit_refs=_digest_refs(
                payload["reaudit_refs"],
                gate_id=gate_id,
            ),
            environment_digest=payload["environment_digest"],
            evaluated_at=payload["evaluated_at"],
            status=GateStatus(payload["status"]),
            repository_root=root,
            input_path=input_path,
            output_path=output_path,
            authorization_snapshot=snapshot,
        )
    except GateEngineError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise GateEngineError(
            "GATE_INPUT_INVALID",
            gate_id,
            _display_gate_input(gate_id),
        ) from error


def _load_gate_input(
    root: Path,
    gate_id: str,
    *,
    verify_authorization: bool,
    defer_authority_bytes: bool,
) -> GateEvaluationInput:
    _gate_number(gate_id)
    repository_root = Path(root).resolve()
    input_path = repository_root / _display_gate_input(gate_id)
    output_path = repository_root / _display_gate_output(gate_id)
    payload, gate_input_bytes = _load_gate_input_payload(
        repository_root,
        gate_id,
    )
    registry, registry_bytes = _load_artifact_registry(
        repository_root,
        gate_id,
    )
    registry_digest = _sha256(registry_bytes)
    if payload["artifact_dag_digest"] != registry_digest:
        raise GateEngineError(
            "ARTIFACT_DAG_DIGEST_MISMATCH",
            gate_id,
            _display_gate_input(gate_id),
        )
    bindings = _resolve_artifact_bindings(
        repository_root,
        gate_id,
        payload,
        registry,
        deferred_paths=(
            frozenset(
                {
                    _TRUSTED_KEY_SET_PATH,
                    _ATTESTATION_PATHS[gate_id],
                }
            )
            if defer_authority_bytes and gate_id in _AUDITED_GATES
            else frozenset()
        ),
    )
    snapshot = GateAuthorizationSnapshot(
        gate_input_sha256=_sha256(gate_input_bytes),
        artifact_registry_sha256=registry_digest,
        artifact_bindings=bindings,
    )
    gate_input = _build_gate_input(
        payload,
        root=repository_root,
        input_path=input_path,
        output_path=output_path,
        snapshot=snapshot,
    )
    if verify_authorization:
        _verify_audited_gate_authorization(
            gate_input,
            registry,
            bindings,
        )
    return gate_input


def load_gate_input(root: Path, gate_id: str) -> GateEvaluationInput:
    return _load_gate_input(
        root,
        gate_id,
        verify_authorization=True,
        defer_authority_bytes=False,
    )


def _read_gate_record(
    root: Path,
    gate_id: str,
) -> tuple[GateRecord, bytes] | None:
    relative_path = _display_gate_output(gate_id)
    try:
        raw = _read_rooted_regular_file(
            root,
            relative_path,
            gate_id=gate_id,
            invalid_code="GATE_RECORD_INVALID",
        )
        payload = _strict_json_document(raw)
        if not isinstance(payload, dict):
            return None
        return GateRecord.from_dict(payload), raw
    except (
        GateEngineError,
        TypeError,
        ValueError,
        UnicodeError,
        json.JSONDecodeError,
    ):
        return None


def _derive_gate_record(
    gate_input: GateEvaluationInput,
    predecessor_refs: tuple[DigestRef, ...],
) -> GateRecord:
    return GateRecord(
        gate_id=gate_input.gate_id,
        gate_artifact_id=gate_input.gate_artifact_id,
        schema_version=gate_input.schema_version,
        evaluation_commit_sha=gate_input.evaluation_commit_sha,
        evaluation_tree_digest=gate_input.evaluation_tree_digest,
        scientific_candidate_sha=gate_input.scientific_candidate_sha,
        scientific_candidate_tree_digest=(gate_input.scientific_candidate_tree_digest),
        artifact_dag_digest=gate_input.artifact_dag_digest,
        predecessor_gate_refs=predecessor_refs,
        input_refs=gate_input.input_refs,
        evaluator_id=gate_input.evaluator_id,
        evaluator_role=gate_input.evaluator_role,
        independence_mode=gate_input.independence_mode,
        evaluator_attestation_ref=gate_input.evaluator_attestation_ref,
        finding_ids=gate_input.finding_ids,
        adjudication_ids=gate_input.adjudication_ids,
        reaudit_refs=gate_input.reaudit_refs,
        environment_digest=gate_input.environment_digest,
        evaluated_at=gate_input.evaluated_at,
        status=gate_input.status,
    )


def _is_gate_derivable(
    root: Path,
    gate_id: str,
    *,
    checked: set[str] | None = None,
    historical_authority: bool = False,
) -> bool:
    checked = set() if checked is None else checked
    if gate_id in checked:
        return False
    checked.add(gate_id)
    loaded_record = _read_gate_record(root, gate_id)
    if loaded_record is None:
        return False
    record, record_bytes = loaded_record
    try:
        gate_input = _load_gate_input(
            root,
            gate_id,
            verify_authorization=not historical_authority,
            defer_authority_bytes=historical_authority,
        )
    except GateEngineError, OSError, ValueError:
        return False

    gate_number = _gate_number(gate_id)
    predecessor_refs: tuple[DigestRef, ...] = ()
    if gate_number:
        predecessor_id = f"G{gate_number - 1}"
        if not _is_gate_derivable(
            root,
            predecessor_id,
            checked=checked,
            historical_authority=True,
        ):
            return False
        predecessor = _read_gate_record(root, predecessor_id)
        if predecessor is None:
            return False
        predecessor_record, predecessor_bytes = predecessor
        predecessor_refs = (
            DigestRef(
                artifact_id=predecessor_record.gate_artifact_id,
                sha256=_sha256(predecessor_bytes),
            ),
        )
    expected = _derive_gate_record(gate_input, predecessor_refs)
    return record == expected and record_bytes == _canonical_json(expected.to_dict())


def _resolve_predecessor(
    gate_input: GateEvaluationInput,
) -> tuple[DigestRef, ...]:
    gate_number = _gate_number(gate_input.gate_id)
    if gate_number == 0:
        return ()
    predecessor_id = f"G{gate_number - 1}"
    predecessor_path = _display_gate_output(predecessor_id)
    loaded = _read_gate_record(gate_input.repository_root, predecessor_id)
    if loaded is None:
        raise GateEngineError(
            "PREDECESSOR_MISSING",
            gate_input.gate_id,
            predecessor_path,
        )
    predecessor, predecessor_bytes = loaded
    if predecessor.status is not GateStatus.PASS:
        raise GateEngineError(
            "PREDECESSOR_NOT_PASS",
            gate_input.gate_id,
            predecessor_path,
        )
    if not _is_gate_derivable(gate_input.repository_root, predecessor_id):
        raise GateEngineError(
            "PREDECESSOR_DERIVATION_MISMATCH",
            gate_input.gate_id,
            predecessor_path,
        )
    return (
        DigestRef(
            artifact_id=predecessor.gate_artifact_id,
            sha256=_sha256(predecessor_bytes),
        ),
    )


def _reject_reused_gate_artifact_id(gate_input: GateEvaluationInput) -> None:
    gate_number = _gate_number(gate_input.gate_id)
    for earlier_number in range(gate_number):
        earlier = _read_gate_record(
            gate_input.repository_root,
            f"G{earlier_number}",
        )
        if (
            earlier is not None
            and earlier[0].gate_artifact_id == gate_input.gate_artifact_id
        ):
            raise GateEngineError(
                "GATE_ARTIFACT_ID_REUSED",
                gate_input.gate_id,
                _display_gate_input(gate_input.gate_id),
            )


def _runtime_gate_id(gate_input: GateEvaluationInput) -> str:
    expected_parent = gate_input.repository_root / "manifests" / "gate-inputs"
    if gate_input.input_path.parent == expected_parent:
        match = re.fullmatch(r"(G[0-8])\.json", gate_input.input_path.name)
        if match is not None:
            return match.group(1)
    return gate_input.gate_id


def _raise_gate_input_object_mismatch(gate_id: str) -> None:
    raise GateEngineError(
        "GATE_INPUT_OBJECT_MISMATCH",
        gate_id,
        _display_gate_input(gate_id),
    )


def _expected_snapshot_bindings(
    root: Path,
    gate_id: str,
    payload: dict[str, Any],
    registry: tuple[ArtifactNode, ...],
) -> tuple[tuple[str, str, Path, str], ...] | None:
    try:
        input_refs = tuple(DigestRef.from_dict(item) for item in payload["input_refs"])
    except KeyError, TypeError, ValueError:
        return None
    result: list[tuple[str, str, Path, str]] = []
    for input_ref in input_refs:
        matches = tuple(
            node
            for node in registry
            if node.artifact_id == input_ref.artifact_id
            and node.output_digest == input_ref.sha256
        )
        if len(matches) != 1:
            return None
        node = matches[0]
        result.append(
            (
                node.artifact_id,
                node.path,
                root / node.path,
                node.output_digest,
            )
        )
    return tuple(result)


def _precheck_gate_input_object(gate_input: GateEvaluationInput) -> str:
    canonical_gate_id = _runtime_gate_id(gate_input)
    expected_input_path = gate_input.repository_root / _display_gate_input(
        canonical_gate_id
    )
    expected_output_path = gate_input.repository_root / _display_gate_output(
        canonical_gate_id
    )
    if (
        gate_input.gate_id != canonical_gate_id
        or gate_input.input_path != expected_input_path
        or gate_input.output_path != expected_output_path
    ):
        _raise_gate_input_object_mismatch(canonical_gate_id)

    raw = _read_rooted_regular_file(
        gate_input.repository_root,
        _display_gate_input(canonical_gate_id),
        gate_id=canonical_gate_id,
        invalid_code="GATE_INPUT_CHANGED",
    )
    try:
        payload = _strict_json_document(raw)
        if (
            not isinstance(payload, dict)
            or set(payload) != _GATE_INPUT_FIELDS
            or payload.get("gate_id") != canonical_gate_id
        ):
            raise ValueError
    except (
        TypeError,
        ValueError,
        UnicodeError,
        json.JSONDecodeError,
    ):
        return canonical_gate_id

    caller_payload = gate_input.to_dict()
    current_digest = _sha256(raw)
    caller_digest = _sha256(_canonical_json(caller_payload))
    if payload != caller_payload:
        if (
            gate_input.authorization_snapshot.gate_input_sha256 == current_digest
            or gate_input.authorization_snapshot.gate_input_sha256 != caller_digest
        ):
            _raise_gate_input_object_mismatch(canonical_gate_id)
        return canonical_gate_id
    if gate_input.authorization_snapshot.gate_input_sha256 != current_digest:
        _raise_gate_input_object_mismatch(canonical_gate_id)

    registry_path = "registry/artifacts.jsonl"
    try:
        registry_raw = _read_rooted_regular_file(
            gate_input.repository_root,
            registry_path,
            gate_id=canonical_gate_id,
            invalid_code="ARTIFACT_DAG_CHANGED",
        )
    except GateEngineError:
        return canonical_gate_id
    registry_digest = _sha256(registry_raw)
    if payload.get("artifact_dag_digest") != registry_digest:
        return canonical_gate_id
    if gate_input.authorization_snapshot.artifact_registry_sha256 != registry_digest:
        _raise_gate_input_object_mismatch(canonical_gate_id)

    try:
        registry_payloads = _strict_jsonl_documents(registry_raw)
        registry = tuple(
            ArtifactNode.from_dict(item)
            for item in registry_payloads
            if isinstance(item, dict)
        )
        if len(registry) != len(registry_payloads):
            raise TypeError
    except (
        TypeError,
        ValueError,
        UnicodeError,
        json.JSONDecodeError,
    ):
        return canonical_gate_id
    expected_bindings = _expected_snapshot_bindings(
        gate_input.repository_root,
        canonical_gate_id,
        payload,
        registry,
    )
    if expected_bindings is None:
        return canonical_gate_id
    actual_bindings = gate_input.authorization_snapshot.artifact_bindings
    if len(actual_bindings) != len(expected_bindings):
        _raise_gate_input_object_mismatch(canonical_gate_id)
    for binding, expected in zip(
        actual_bindings,
        expected_bindings,
        strict=True,
    ):
        (
            artifact_id,
            relative_path,
            resolved_path,
            declared_sha256,
        ) = expected
        if (
            binding.artifact_id != artifact_id
            or binding.relative_path != relative_path
            or binding.resolved_path != resolved_path
            or binding.declared_sha256 != declared_sha256
            or binding.loaded_sha256 != declared_sha256
        ):
            _raise_gate_input_object_mismatch(canonical_gate_id)
    return canonical_gate_id


def _rehash_authorization(gate_input: GateEvaluationInput) -> None:
    snapshot = gate_input.authorization_snapshot
    input_path = _display_gate_input(gate_input.gate_id)
    input_bytes = _read_rooted_regular_file(
        gate_input.repository_root,
        input_path,
        gate_id=gate_input.gate_id,
        invalid_code="GATE_INPUT_CHANGED",
    )
    if _sha256(input_bytes) != snapshot.gate_input_sha256:
        raise GateEngineError(
            "GATE_INPUT_CHANGED",
            gate_input.gate_id,
            input_path,
        )

    registry_path = "registry/artifacts.jsonl"
    registry_bytes = _read_rooted_regular_file(
        gate_input.repository_root,
        registry_path,
        gate_id=gate_input.gate_id,
        invalid_code="ARTIFACT_DAG_CHANGED",
    )
    if _sha256(registry_bytes) != snapshot.artifact_registry_sha256:
        raise GateEngineError(
            "ARTIFACT_DAG_CHANGED",
            gate_input.gate_id,
            registry_path,
        )

    for binding in snapshot.artifact_bindings:
        artifact_bytes = _read_rooted_regular_file(
            gate_input.repository_root,
            binding.relative_path,
            gate_id=gate_input.gate_id,
            invalid_code="INPUT_ARTIFACT_DIGEST_MISMATCH",
        )
        if (
            _sha256(artifact_bytes) != binding.loaded_sha256
            or binding.loaded_sha256 != binding.declared_sha256
        ):
            raise GateEngineError(
                "INPUT_ARTIFACT_DIGEST_MISMATCH",
                gate_input.gate_id,
                binding.relative_path,
            )


def _reverify_audited_authorization(
    gate_input: GateEvaluationInput,
) -> None:
    if gate_input.gate_id not in _AUDITED_GATES:
        return
    _payload, _ = _load_gate_input_payload(
        gate_input.repository_root,
        gate_input.gate_id,
    )
    registry, _ = _load_artifact_registry(
        gate_input.repository_root,
        gate_input.gate_id,
    )
    _verify_audited_gate_authorization(
        gate_input,
        registry,
        gate_input.authorization_snapshot.artifact_bindings,
    )


def _ensure_safe_output(gate_input: GateEvaluationInput) -> None:
    output_path = gate_input.output_path
    root = gate_input.repository_root
    if _path_has_symlink(root, output_path):
        raise GateEngineError(
            "GATE_OUTPUT_UNSAFE",
            gate_input.gate_id,
            _display_gate_output(gate_input.gate_id),
        )
    if output_path.exists() and not output_path.is_file():
        raise GateEngineError(
            "GATE_OUTPUT_UNSAFE",
            gate_input.gate_id,
            _display_gate_output(gate_input.gate_id),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if _path_has_symlink(root, output_path):
        raise GateEngineError(
            "GATE_OUTPUT_UNSAFE",
            gate_input.gate_id,
            _display_gate_output(gate_input.gate_id),
        )


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".stc-gate-",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o644)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def evaluate_gate(gate_input: GateEvaluationInput) -> GateRecord:
    if type(gate_input) is not GateEvaluationInput:
        raise TypeError("gate_input must be exactly GateEvaluationInput")
    canonical_gate_id = _precheck_gate_input_object(gate_input)
    _rehash_authorization(gate_input)
    _reverify_audited_authorization(gate_input)
    canonical_input = load_gate_input(
        gate_input.repository_root,
        canonical_gate_id,
    )
    if gate_input != canonical_input:
        _raise_gate_input_object_mismatch(canonical_gate_id)
    predecessor_refs = _resolve_predecessor(gate_input)
    _reject_reused_gate_artifact_id(gate_input)
    record = _derive_gate_record(gate_input, predecessor_refs)
    _rehash_authorization(gate_input)
    _reverify_audited_authorization(gate_input)
    _ensure_safe_output(gate_input)
    _atomic_write(
        gate_input.output_path,
        _canonical_json(record.to_dict()),
    )
    return record


def verify_gate_chain(root: Path, *, through: str) -> GateChainReport:
    through_number = _gate_number(through)
    repository_root = Path(root).resolve()
    diagnostics: list[GateChainDiagnostic] = []
    stale_gate_ids: list[str] = []
    previous: tuple[GateRecord, bytes] | None = None
    upstream_stale = False

    for gate_number in range(through_number + 1):
        gate_id = f"G{gate_number}"
        path = _display_gate_output(gate_id)
        loaded = _read_gate_record(repository_root, gate_id)
        if loaded is None:
            diagnostics.append(
                GateChainDiagnostic("GATE_RECORD_MISSING", gate_id, path)
            )
            stale_gate_ids.append(gate_id)
            upstream_stale = True
            previous = None
            continue
        record, _record_bytes = loaded
        if upstream_stale:
            diagnostics.append(
                GateChainDiagnostic("UPSTREAM_GATE_STALE", gate_id, path)
            )
            stale_gate_ids.append(gate_id)
            previous = loaded
            continue

        edge_error: str | None = None
        if gate_number == 0:
            if record.predecessor_gate_refs:
                edge_error = "PREDECESSOR_CHAIN_INVALID"
        elif len(record.predecessor_gate_refs) != 1 or previous is None:
            edge_error = "PREDECESSOR_CHAIN_INVALID"
        else:
            previous_record, previous_bytes = previous
            predecessor_ref = record.predecessor_gate_refs[0]
            if predecessor_ref.artifact_id != previous_record.gate_artifact_id:
                edge_error = "PREDECESSOR_IDENTITY_MISMATCH"
            elif predecessor_ref.sha256 != _sha256(previous_bytes):
                edge_error = "PREDECESSOR_DIGEST_MISMATCH"
        if edge_error is not None:
            diagnostics.append(GateChainDiagnostic(edge_error, gate_id, path))
            stale_gate_ids.append(gate_id)
            upstream_stale = True
            previous = loaded
            continue

        if not _is_gate_derivable(
            repository_root,
            gate_id,
            historical_authority=gate_number < through_number,
        ):
            diagnostics.append(
                GateChainDiagnostic(
                    "GATE_DERIVATION_MISMATCH",
                    gate_id,
                    path,
                )
            )
            stale_gate_ids.append(gate_id)
            upstream_stale = True
        previous = loaded

    return GateChainReport(
        diagnostics=tuple(diagnostics),
        stale_gate_ids=tuple(stale_gate_ids),
    )
