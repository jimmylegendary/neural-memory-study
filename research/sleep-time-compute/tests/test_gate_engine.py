from __future__ import annotations

import base64
import hashlib
import importlib
import inspect
import json
import os
import stat
import subprocess
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).parents[1]
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
EVALUATED_AT = "2026-07-25T12:00:00+09:00"
TEST_ED25519_PRIVATE_KEY = """-----BEGIN PRIVATE KEY-----
MC4CAQAwBQYDK2VwBCIEIP5sRXbO3H0fv86OfE/AkM5XIY8IoQKMHa9Hv1G3JFKz
-----END PRIVATE KEY-----
"""
TEST_ED25519_PRIVATE_KEY_2 = """-----BEGIN PRIVATE KEY-----
MC4CAQAwBQYDK2VwBCIEIAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8g
-----END PRIVATE KEY-----
"""
TEST_ED25519_SPKI_BASE64 = (
    "MCowBQYDK2VwAyEAhfUjLKc8axRD3CbN1QpxJ7Uo//mfSs1erCkVyX6QwJA="
)
TEST_ED25519_SPKI_BASE64_2 = (
    "MCowBQYDK2VwAyEAebVWLo/mVPlAeLES6KmLp5AfhTrmlb7X4OORC60ElmQ="
)
TEST_P256_SPKI_BASE64 = (
    "MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE8k1ywUPdUBQ1fwIT348zQbzysPX0"
    "e5B8vI0XjFIjSjCU/Vbyf4g5uPBLAQ0BgjO2HgZgEJgQ1teetwu0zE1Z2w=="
)
TEST_G7_SIGNATURE_BASE64 = (
    "GMl7A2Uq4bTYWJnzF9syDwTlISZHAgPuXGgvzIJErGkSUj3KA4Yr6iAygLMReiEZ"
    "18i3CKBVRl5Uh1fL8XWVAA=="
)
TEST_G8_SIGNATURE_BASE64 = (
    "T54oEg7BORHU6osM9oabgeFQJrlvIYkVjNVmNUajyoUOB8LayVzx4kMyF2uBHN1S"
    "TYAJbK5OV0yZU2M1MmNvBg=="
)


def _gate_api():
    from stc_research.gate_engine import (
        GateEngineError,
        evaluate_gate,
        load_gate_input,
        verify_gate_chain,
    )

    return GateEngineError, evaluate_gate, load_gate_input, verify_gate_chain


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


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: object) -> bytes:
    encoded = _canonical_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return encoded


def _write_jsonl(path: Path, payloads: list[dict[str, object]]) -> bytes:
    encoded = b"".join(_canonical_json(payload) for payload in payloads)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return encoded


def _gate_artifact_id(gate_id: str) -> str:
    return f"ART-STC-{9901 + int(gate_id[1:]):04d}"


def _artifact_node(
    artifact_id: str,
    path: str,
    payload: bytes,
    *,
    artifact_type: str = "gate-input",
) -> dict[str, object]:
    return {
        "artifact_id": artifact_id,
        "schema_version": "1.0.0",
        "artifact_type": artifact_type,
        "path": path,
        "input_digests": [SHA_A],
        "producer_command": "tests: create immutable gate input",
        "output_digest": _sha256(payload),
        "consumers": [],
        "claim_ids": [],
        "question_ids": [],
        "gate_status": "PASS",
        "owner": "gate-engine-tests",
        "frozen_release_tag": None,
    }


def _gate_input_payload(
    gate_id: str,
    *,
    artifact_dag_digest: str,
    input_ref: dict[str, str],
    environment_digest: str,
    status: str = "PASS",
) -> dict[str, object]:
    gate_number = int(gate_id[1:])
    return {
        "gate_id": gate_id,
        "gate_artifact_id": _gate_artifact_id(gate_id),
        "schema_version": "1.0.0",
        "evaluation_commit_sha": f"{gate_number + 1:x}" * 40,
        "evaluation_tree_digest": SHA_A,
        "scientific_candidate_sha": None,
        "scientific_candidate_tree_digest": None,
        "artifact_dag_digest": artifact_dag_digest,
        "input_refs": [input_ref],
        "evaluator_id": "stc-gate-engine",
        "evaluator_role": "automated",
        "independence_mode": "deterministic",
        "evaluator_attestation_ref": None,
        "finding_ids": [],
        "adjudication_ids": [],
        "reaudit_refs": [],
        "environment_digest": environment_digest,
        "evaluated_at": EVALUATED_AT,
        "status": status,
    }


def _gate_repo(tmp_path: Path) -> tuple[Path, dict[str, dict[str, str]]]:
    root = tmp_path / "package"
    environment_refs: dict[str, dict[str, str]] = {}
    nodes: list[dict[str, object]] = []
    for suffix, artifact_id in (
        ("a", "ART-STC-1001"),
        ("b", "ART-STC-1002"),
    ):
        relative_path = f"inputs/environment-{suffix}.json"
        payload = _write_json(
            root / relative_path,
            {"environment": suffix, "version": 1},
        )
        digest = _sha256(payload)
        environment_refs[suffix] = {
            "artifact_id": artifact_id,
            "sha256": digest,
        }
        nodes.append(
            _artifact_node(
                artifact_id,
                relative_path,
                payload,
                artifact_type="release-environment",
            )
        )
    registry_payload = _write_jsonl(
        root / "registry" / "artifacts.jsonl",
        nodes,
    )
    artifact_dag_digest = _sha256(registry_payload)
    for gate_number in range(9):
        gate_id = f"G{gate_number}"
        _write_json(
            root / "manifests" / "gate-inputs" / f"{gate_id}.json",
            _gate_input_payload(
                gate_id,
                artifact_dag_digest=artifact_dag_digest,
                input_ref=environment_refs["a"],
                environment_digest=environment_refs["a"]["sha256"],
            ),
        )
    return root, environment_refs


def _rewrite_gate_input(
    root: Path,
    gate_id: str,
    mutate,
) -> dict[str, object]:
    path = root / "manifests" / "gate-inputs" / f"{gate_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    _write_json(path, payload)
    return payload


def _rewrite_artifact_registry(
    root: Path,
    mutate,
    *,
    gate_id: str = "G0",
) -> list[dict[str, object]]:
    path = root / "registry" / "artifacts.jsonl"
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    mutate(records)
    registry_bytes = _write_jsonl(path, records)
    _rewrite_gate_input(
        root,
        gate_id,
        lambda payload: payload.__setitem__(
            "artifact_dag_digest",
            _sha256(registry_bytes),
        ),
    )
    return records


def _error_code(error: BaseException) -> str:
    return str(error.code)  # type: ignore[attr-defined]


def _chain_contract(report: object) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (diagnostic.code, diagnostic.gate_id, diagnostic.path)
        for diagnostic in report.diagnostics
    )


def _assert_deeply_immutable(value: object, path: str = "gate_input") -> None:
    mutable_container_types = (Mapping, list, set, frozenset, bytearray)
    assert not isinstance(value, mutable_container_types), (
        f"{path} contains non-tuple container {type(value).__name__}"
    )
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            _assert_deeply_immutable(item, f"{path}[{index}]")
        return
    if is_dataclass(value) and not isinstance(value, type):
        assert type(value).__dataclass_params__.frozen  # type: ignore[attr-defined]
        for field in fields(value):
            _assert_deeply_immutable(
                getattr(value, field.name),
                f"{path}.{field.name}",
            )


def test_gate_input_and_record_schemas_close_identity_and_output_fields() -> None:
    gate_input_schema = json.loads(
        (PACKAGE_ROOT / "schemas" / "control" / "gate-input.schema.json").read_text(
            encoding="utf-8"
        )
    )
    gate_schema = json.loads(
        (PACKAGE_ROOT / "schemas" / "control" / "gate.schema.json").read_text(
            encoding="utf-8"
        )
    )

    input_output_fields = {
        "gate_artifact_id",
        "finding_ids",
        "adjudication_ids",
        "reaudit_refs",
        "environment_digest",
        "status",
    }
    assert input_output_fields <= set(gate_input_schema["required"])
    assert input_output_fields <= set(gate_input_schema["properties"])
    assert "gate_artifact_id" in gate_schema["required"]
    assert "gate_artifact_id" in gate_schema["properties"]
    assert gate_input_schema["additionalProperties"] is False
    assert gate_schema["additionalProperties"] is False
    aware_datetime_ref = (
        "https://schemas.sleep-time-compute.org/control/"
        "common.schema.json#/$defs/awareDateTime"
    )
    assert gate_input_schema["properties"]["evaluated_at"] == {
        "$ref": aware_datetime_ref
    }
    assert gate_schema["properties"]["evaluated_at"] == {
        "$ref": aware_datetime_ref
    }
    candidate_condition = {
        "if": {
            "properties": {"gate_id": {"enum": ["G7", "G8"]}},
            "required": ["gate_id"],
        },
        "then": {
            "properties": {
                "scientific_candidate_sha": {
                    "$ref": (
                        "https://schemas.sleep-time-compute.org/control/"
                        "common.schema.json#/$defs/gitSha1"
                    )
                },
                "scientific_candidate_tree_digest": {
                    "$ref": (
                        "https://schemas.sleep-time-compute.org/control/"
                        "common.schema.json#/$defs/sha256"
                    )
                },
                "evaluator_attestation_ref": {
                    "$ref": (
                        "https://schemas.sleep-time-compute.org/control/"
                        "common.schema.json#/$defs/DigestRef"
                    )
                },
            }
        },
        "else": {
            "properties": {
                "scientific_candidate_sha": {"type": "null"},
                "scientific_candidate_tree_digest": {"type": "null"},
                "evaluator_attestation_ref": {"type": "null"},
            }
        },
    }
    assert gate_input_schema["allOf"] == [candidate_condition]
    assert gate_schema["allOf"] == [candidate_condition]


@pytest.mark.parametrize("gate_id", ["G7", "G8"])
def test_loader_rejects_missing_candidate_or_attestation_for_audited_gates(
    tmp_path: Path,
    gate_id: str,
) -> None:
    GateEngineError, _, load_gate_input, _ = _gate_api()
    root, _ = _gate_repo(tmp_path)

    with pytest.raises(GateEngineError) as captured:
        load_gate_input(root, gate_id)

    assert _error_code(captured.value) == "GATE_INPUT_INVALID"
    assert captured.value.path == f"manifests/gate-inputs/{gate_id}.json"
    assert not (root / "manifests" / "gates" / f"{gate_id}.json").exists()


@pytest.mark.parametrize("gate_id", ["G7", "G8"])
@pytest.mark.parametrize(
    "missing_field",
    [
        "scientific_candidate_sha",
        "scientific_candidate_tree_digest",
        "evaluator_attestation_ref",
    ],
)
def test_public_gate_input_model_requires_complete_audited_identity(
    tmp_path: Path,
    gate_id: str,
    missing_field: str,
) -> None:
    from stc_research.models import DigestRef

    _, _, load_gate_input, _ = _gate_api()
    root, _ = _gate_repo(tmp_path)
    base = load_gate_input(root, "G0")
    valid = replace(
        base,
        gate_id=gate_id,
        scientific_candidate_sha="e" * 40,
        scientific_candidate_tree_digest=SHA_B,
        evaluator_attestation_ref=DigestRef(
            artifact_id="ART-STC-2001",
            sha256=SHA_C,
        ),
    )
    assert valid.gate_id == gate_id

    with pytest.raises(ValueError, match="G7/G8 require"):
        replace(valid, **{missing_field: None})


def test_loader_binds_one_argument_evaluation_to_canonical_paths(
    tmp_path: Path,
) -> None:
    from stc_research.models import GateEvaluationInput

    _, evaluate_gate, load_gate_input, _ = _gate_api()
    root, _ = _gate_repo(tmp_path)

    gate_input = load_gate_input(root, "G0")

    assert isinstance(gate_input, GateEvaluationInput)
    assert tuple(inspect.signature(evaluate_gate).parameters) == ("gate_input",)
    assert gate_input.repository_root == root.resolve()
    assert gate_input.input_path == (
        root / "manifests" / "gate-inputs" / "G0.json"
    ).resolve()
    assert gate_input.output_path == (
        root / "manifests" / "gates" / "G0.json"
    ).resolve()
    expected_payload = json.loads(gate_input.input_path.read_text(encoding="utf-8"))
    assert gate_input.to_dict() == expected_payload
    with pytest.raises(FrozenInstanceError):
        gate_input.repository_root = tmp_path  # type: ignore[misc]


def test_loaded_authorization_snapshot_is_public_and_deeply_immutable(
    tmp_path: Path,
) -> None:
    from stc_research.models import (
        GateArtifactBinding,
        GateAuthorizationSnapshot,
    )

    _, _, load_gate_input, _ = _gate_api()
    root, environment_refs = _gate_repo(tmp_path)

    gate_input = load_gate_input(root, "G0")
    snapshot = gate_input.authorization_snapshot

    assert isinstance(snapshot, GateAuthorizationSnapshot)
    assert isinstance(snapshot.artifact_bindings, tuple)
    assert isinstance(gate_input.input_refs, tuple)
    assert isinstance(gate_input.finding_ids, tuple)
    assert isinstance(gate_input.adjudication_ids, tuple)
    assert isinstance(gate_input.reaudit_refs, tuple)
    assert snapshot.gate_input_sha256 == _sha256(gate_input.input_path.read_bytes())
    registry = root / "registry" / "artifacts.jsonl"
    assert snapshot.artifact_registry_sha256 == _sha256(registry.read_bytes())
    assert len(snapshot.artifact_bindings) == 1
    binding = snapshot.artifact_bindings[0]
    assert isinstance(binding, GateArtifactBinding)
    assert binding.artifact_id == "ART-STC-1001"
    assert binding.relative_path == "inputs/environment-a.json"
    assert binding.resolved_path == (
        root / "inputs" / "environment-a.json"
    ).resolve()
    assert binding.declared_sha256 == environment_refs["a"]["sha256"]
    assert binding.loaded_sha256 == environment_refs["a"]["sha256"]
    _assert_deeply_immutable(gate_input)

    with pytest.raises(FrozenInstanceError):
        gate_input.input_refs[0].sha256 = SHA_C  # type: ignore[misc]
    with pytest.raises(TypeError):
        snapshot.artifact_bindings[0] = binding  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        binding.loaded_sha256 = SHA_C  # type: ignore[misc]


@pytest.mark.parametrize(
    ("invalid_binding", "expected_code", "expected_path"),
    [
        (
            "unknown-artifact",
            "INPUT_ARTIFACT_UNKNOWN",
            "manifests/gate-inputs/G0.json",
        ),
        (
            "ambiguous-artifact-id",
            "INPUT_ARTIFACT_AMBIGUOUS",
            "registry/artifacts.jsonl",
        ),
        (
            "ref-registry-digest-mismatch",
            "INPUT_REF_DIGEST_MISMATCH",
            "manifests/gate-inputs/G0.json",
        ),
        (
            "artifact-bytes-mismatch",
            "INPUT_ARTIFACT_BYTES_MISMATCH",
            "inputs/environment-a.json",
        ),
        (
            "environment-unbound",
            "ENVIRONMENT_DIGEST_UNBOUND",
            "manifests/gate-inputs/G0.json",
        ),
        (
            "environment-ambiguous",
            "ENVIRONMENT_DIGEST_AMBIGUOUS",
            "manifests/gate-inputs/G0.json",
        ),
    ],
)
def test_loader_rejects_input_not_uniquely_bound_to_canonical_artifact(
    tmp_path: Path,
    invalid_binding: str,
    expected_code: str,
    expected_path: str,
) -> None:
    GateEngineError, _, load_gate_input, _ = _gate_api()
    root, environment_refs = _gate_repo(tmp_path)

    if invalid_binding == "unknown-artifact":
        _rewrite_gate_input(
            root,
            "G0",
            lambda payload: payload.update(
                {
                    "input_refs": [
                        {
                            "artifact_id": "ART-STC-1999",
                            "sha256": SHA_C,
                        }
                    ],
                    "environment_digest": SHA_C,
                }
            ),
        )
    elif invalid_binding == "ambiguous-artifact-id":

        def duplicate_artifact_id(records: list[dict[str, object]]) -> None:
            duplicate = dict(records[1])
            duplicate["artifact_id"] = "ART-STC-1001"
            records.append(duplicate)

        _rewrite_artifact_registry(root, duplicate_artifact_id)
    elif invalid_binding == "ref-registry-digest-mismatch":
        _rewrite_gate_input(
            root,
            "G0",
            lambda payload: payload.update(
                {
                    "input_refs": [
                        {
                            "artifact_id": "ART-STC-1001",
                            "sha256": SHA_C,
                        }
                    ],
                    "environment_digest": SHA_C,
                }
            ),
        )
    elif invalid_binding == "artifact-bytes-mismatch":
        (root / "inputs" / "environment-a.json").write_bytes(
            b"changed-before-initial-load\n"
        )
    elif invalid_binding == "environment-unbound":
        _rewrite_gate_input(
            root,
            "G0",
            lambda payload: payload.__setitem__("environment_digest", SHA_C),
        )
    else:
        environment_a = (root / "inputs" / "environment-a.json").read_bytes()
        (root / "inputs" / "environment-b.json").write_bytes(environment_a)

        def make_environment_digest_ambiguous(
            records: list[dict[str, object]],
        ) -> None:
            records[1]["output_digest"] = environment_refs["a"]["sha256"]

        _rewrite_artifact_registry(root, make_environment_digest_ambiguous)
        _rewrite_gate_input(
            root,
            "G0",
            lambda payload: payload.update(
                {
                    "input_refs": [
                        environment_refs["a"],
                        {
                            "artifact_id": "ART-STC-1002",
                            "sha256": environment_refs["a"]["sha256"],
                        },
                    ],
                    "environment_digest": environment_refs["a"]["sha256"],
                }
            ),
        )

    with pytest.raises(GateEngineError) as captured:
        load_gate_input(root, "G0")

    assert _error_code(captured.value) == expected_code
    assert captured.value.gate_id == "G0"
    assert captured.value.path == expected_path
    assert str(root.resolve()) not in str(captured.value)
    assert not (root / "manifests" / "gates" / "G0.json").exists()


def test_g0_evaluation_is_canonical_self_identified_and_atomic(
    tmp_path: Path,
) -> None:
    _, evaluate_gate, load_gate_input, _ = _gate_api()
    root, environment_refs = _gate_repo(tmp_path)

    record = evaluate_gate(load_gate_input(root, "G0"))
    output = root / "manifests" / "gates" / "G0.json"

    assert output.read_bytes() == _canonical_json(record.to_dict())
    assert stat.S_IMODE(output.stat().st_mode) == 0o644
    assert record.gate_id == "G0"
    assert record.gate_artifact_id == _gate_artifact_id("G0")
    assert record.predecessor_gate_refs == ()
    assert [item.to_dict() for item in record.input_refs] == [
        environment_refs["a"]
    ]
    assert record.status.value == "PASS"


@pytest.mark.parametrize(
    ("predecessor_state", "expected_code"),
    [
        ("missing", "PREDECESSOR_MISSING"),
        ("fail", "PREDECESSOR_NOT_PASS"),
    ],
)
def test_gate_evaluation_requires_exact_pass_predecessor(
    tmp_path: Path,
    predecessor_state: str,
    expected_code: str,
) -> None:
    GateEngineError, evaluate_gate, load_gate_input, _ = _gate_api()
    root, _ = _gate_repo(tmp_path)
    if predecessor_state == "fail":
        _rewrite_gate_input(
            root,
            "G0",
            lambda payload: payload.__setitem__("status", "FAIL"),
        )
        evaluate_gate(load_gate_input(root, "G0"))

    with pytest.raises(GateEngineError) as captured:
        evaluate_gate(load_gate_input(root, "G1"))

    assert _error_code(captured.value) == expected_code
    assert captured.value.gate_id == "G1"
    assert captured.value.path == "manifests/gates/G0.json"


def test_predecessor_ref_uses_canonical_gate_identity_and_file_digest(
    tmp_path: Path,
) -> None:
    _, evaluate_gate, load_gate_input, _ = _gate_api()
    root, _ = _gate_repo(tmp_path)
    g0 = evaluate_gate(load_gate_input(root, "G0"))
    g0_bytes = (root / "manifests" / "gates" / "G0.json").read_bytes()

    g1 = evaluate_gate(load_gate_input(root, "G1"))

    assert len(g1.predecessor_gate_refs) == 1
    assert g1.predecessor_gate_refs[0].artifact_id == g0.gate_artifact_id
    assert g1.predecessor_gate_refs[0].sha256 == _sha256(g0_bytes)


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("skip", "PREDECESSOR_CHAIN_INVALID"),
        ("digest", "PREDECESSOR_DIGEST_MISMATCH"),
        ("identity", "PREDECESSOR_IDENTITY_MISMATCH"),
    ],
)
def test_verify_chain_rejects_skipped_or_changed_predecessor(
    tmp_path: Path,
    mutation: str,
    expected_code: str,
) -> None:
    _, evaluate_gate, load_gate_input, verify_gate_chain = _gate_api()
    root, _ = _gate_repo(tmp_path)
    evaluate_gate(load_gate_input(root, "G0"))
    evaluate_gate(load_gate_input(root, "G1"))
    g1_path = root / "manifests" / "gates" / "G1.json"
    g1 = json.loads(g1_path.read_text(encoding="utf-8"))
    if mutation == "skip":
        g1["predecessor_gate_refs"] = []
    elif mutation == "digest":
        g1["predecessor_gate_refs"][0]["sha256"] = SHA_C
    else:
        g1["predecessor_gate_refs"][0]["artifact_id"] = "ART-STC-9899"
    _write_json(g1_path, g1)

    report = verify_gate_chain(root, through="G1")

    assert not report.success
    assert _chain_contract(report) == (
        (expected_code, "G1", "manifests/gates/G1.json"),
    )


@pytest.mark.parametrize(
    "stale_derivation",
    ["schema-valid-gate-tamper", "gate-input-changed-without-reevaluation"],
)
def test_successor_and_chain_rederive_predecessor_from_frozen_input(
    tmp_path: Path,
    stale_derivation: str,
) -> None:
    (
        GateEngineError,
        evaluate_gate,
        load_gate_input,
        verify_gate_chain,
    ) = _gate_api()
    root, _ = _gate_repo(tmp_path)
    evaluate_gate(load_gate_input(root, "G0"))

    if stale_derivation == "schema-valid-gate-tamper":
        gate_path = root / "manifests" / "gates" / "G0.json"
        gate_record = json.loads(gate_path.read_text(encoding="utf-8"))
        gate_record["evaluator_id"] = "schema-valid-tamper"
        _write_json(gate_path, gate_record)
    else:
        _rewrite_gate_input(
            root,
            "G0",
            lambda payload: payload.__setitem__(
                "evaluator_id",
                "changed-without-reevaluation",
            ),
        )

    report = verify_gate_chain(root, through="G0")
    with pytest.raises(GateEngineError) as captured:
        evaluate_gate(load_gate_input(root, "G1"))

    assert not report.success
    assert report.stale_gate_ids == ("G0",)
    assert _chain_contract(report) == (
        (
            "GATE_DERIVATION_MISMATCH",
            "G0",
            "manifests/gates/G0.json",
        ),
    )
    assert _error_code(captured.value) == "PREDECESSOR_DERIVATION_MISMATCH"
    assert captured.value.gate_id == "G1"
    assert captured.value.path == "manifests/gates/G0.json"
    assert not (root / "manifests" / "gates" / "G1.json").exists()


@pytest.mark.parametrize(
    ("changed_target", "expected_code"),
    [
        ("input-file", "GATE_INPUT_CHANGED"),
        ("artifact", "INPUT_ARTIFACT_DIGEST_MISMATCH"),
        ("artifact-dag", "ARTIFACT_DAG_CHANGED"),
    ],
)
def test_loaded_input_rehashes_every_authorization_material_before_write(
    tmp_path: Path,
    changed_target: str,
    expected_code: str,
) -> None:
    GateEngineError, evaluate_gate, load_gate_input, _ = _gate_api()
    root, _ = _gate_repo(tmp_path)
    gate_input = load_gate_input(root, "G0")
    if changed_target == "input-file":
        _rewrite_gate_input(
            root,
            "G0",
            lambda payload: payload.__setitem__(
                "evaluator_id",
                "changed-after-load",
            ),
        )
    elif changed_target == "artifact":
        (root / "inputs" / "environment-a.json").write_bytes(b"changed\n")
    else:
        registry = root / "registry" / "artifacts.jsonl"
        registry.write_bytes(registry.read_bytes() + b"\n")

    with pytest.raises(GateEngineError) as captured:
        evaluate_gate(gate_input)

    assert _error_code(captured.value) == expected_code
    assert not (root / "manifests" / "gates" / "G0.json").exists()


@pytest.mark.parametrize(
    ("changed_target", "expected_code", "expected_path"),
    [
        (
            "gate-input",
            "GATE_INPUT_CHANGED",
            "manifests/gate-inputs/G0.json",
        ),
        (
            "artifact-registry",
            "ARTIFACT_DAG_CHANGED",
            "registry/artifacts.jsonl",
        ),
        (
            "input-artifact",
            "INPUT_ARTIFACT_DIGEST_MISMATCH",
            "inputs/environment-a.json",
        ),
    ],
)
def test_rehash_rejects_symlink_before_any_unsafe_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_target: str,
    expected_code: str,
    expected_path: str,
) -> None:
    GateEngineError, evaluate_gate, load_gate_input, _ = _gate_api()
    root, _ = _gate_repo(tmp_path)
    gate_input = load_gate_input(root, "G0")
    targets = {
        "gate-input": root / "manifests" / "gate-inputs" / "G0.json",
        "artifact-registry": root / "registry" / "artifacts.jsonl",
        "input-artifact": root / "inputs" / "environment-a.json",
    }
    target = targets[changed_target]
    original = target.read_bytes()
    outside = tmp_path / f"outside-{changed_target}"
    outside.write_bytes(original)
    target.unlink()
    target.symlink_to(outside)
    real_read_bytes = Path.read_bytes
    unsafe_reads: list[Path] = []

    def observe_read(path: Path) -> bytes:
        if path.is_symlink():
            unsafe_reads.append(path)
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", observe_read)

    with pytest.raises(GateEngineError) as captured:
        evaluate_gate(gate_input)

    assert _error_code(captured.value) == expected_code
    assert captured.value.path == expected_path
    assert unsafe_reads == []
    assert real_read_bytes(outside) == original
    assert not (root / "manifests" / "gates" / "G0.json").exists()


@pytest.mark.parametrize("race_target", ["leaf", "intermediate-directory"])
def test_rehash_uses_rooted_nofollow_dirfds_across_path_swap_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    race_target: str,
) -> None:
    GateEngineError, evaluate_gate, load_gate_input, _ = _gate_api()
    gate_engine = importlib.import_module("stc_research.gate_engine")
    root, _ = _gate_repo(tmp_path)
    gate_input = load_gate_input(root, "G0")
    artifact_path = root / "inputs" / "environment-a.json"
    artifact_bytes = artifact_path.read_bytes()
    outside_directory = tmp_path / "outside-inputs"
    outside_directory.mkdir()
    outside_file = outside_directory / "environment-a.json"
    outside_file.write_bytes(artifact_bytes)
    external_inodes = {
        (outside_directory.stat().st_dev, outside_directory.stat().st_ino),
        (outside_file.stat().st_dev, outside_file.stat().st_ino),
    }
    real_open = os.open
    open_events: list[tuple[str, int, int | None]] = []
    external_opens: list[str] = []
    race_triggered = False
    attack_component = (
        "environment-a.json"
        if race_target == "leaf"
        else "inputs"
    )

    def race_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal race_triggered
        path_text = os.fsdecode(os.fspath(path))
        open_events.append((path_text, flags, dir_fd))
        if (
            not race_triggered
            and dir_fd is not None
            and path_text == attack_component
        ):
            race_triggered = True
            if race_target == "leaf":
                artifact_path.unlink()
                artifact_path.symlink_to(outside_file)
            else:
                moved_directory = root / "inputs-before-race"
                (root / "inputs").rename(moved_directory)
                (root / "inputs").symlink_to(
                    outside_directory,
                    target_is_directory=True,
                )
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) in external_inodes:
            external_opens.append(path_text)
        return descriptor

    monkeypatch.setattr(gate_engine.os, "open", race_open)

    with pytest.raises(GateEngineError) as captured:
        evaluate_gate(gate_input)

    assert _error_code(captured.value) == "INPUT_ARTIFACT_DIGEST_MISMATCH"
    assert captured.value.path == "inputs/environment-a.json"
    assert race_triggered
    assert external_opens == []
    root_events = [
        event
        for event in open_events
        if event[0] == os.fspath(root.resolve()) and event[2] is None
    ]
    assert root_events
    assert root_events[0][1] & os.O_DIRECTORY
    assert root_events[0][1] & os.O_NOFOLLOW
    attack_events = [
        event
        for event in open_events
        if event[0] == attack_component and event[2] is not None
    ]
    assert attack_events
    assert attack_events[0][1] & os.O_NOFOLLOW
    if race_target == "intermediate-directory":
        assert attack_events[0][1] & os.O_DIRECTORY
    for path_text, _, dir_fd in open_events:
        if dir_fd is not None:
            assert not Path(path_text).is_absolute()
    assert outside_file.read_bytes() == artifact_bytes
    assert not (root / "manifests" / "gates" / "G0.json").exists()


def test_same_input_and_frozen_timestamp_reproduce_identical_gate_bytes(
    tmp_path: Path,
) -> None:
    _, evaluate_gate, load_gate_input, _ = _gate_api()
    root, _ = _gate_repo(tmp_path)

    first = evaluate_gate(load_gate_input(root, "G0"))
    first_bytes = (root / "manifests" / "gates" / "G0.json").read_bytes()
    second = evaluate_gate(load_gate_input(root, "G0"))
    second_bytes = (root / "manifests" / "gates" / "G0.json").read_bytes()

    assert first == second
    assert first_bytes == second_bytes


def test_changed_input_recursively_stales_every_successor(
    tmp_path: Path,
) -> None:
    _, evaluate_gate, load_gate_input, verify_gate_chain = _gate_api()
    root, environment_refs = _gate_repo(tmp_path)
    for gate_id in ("G0", "G1", "G2", "G3"):
        evaluate_gate(load_gate_input(root, gate_id))
    original_g1 = (
        root / "manifests" / "gates" / "G1.json"
    ).read_bytes()

    def change_input(payload: dict[str, object]) -> None:
        payload["input_refs"] = [environment_refs["b"]]
        payload["environment_digest"] = environment_refs["b"]["sha256"]

    _rewrite_gate_input(root, "G1", change_input)
    evaluate_gate(load_gate_input(root, "G1"))
    changed_g1 = (root / "manifests" / "gates" / "G1.json").read_bytes()
    report = verify_gate_chain(root, through="G3")

    assert original_g1 != changed_g1
    assert not report.success
    assert report.stale_gate_ids == ("G2", "G3")
    assert _chain_contract(report) == (
        (
            "PREDECESSOR_DIGEST_MISMATCH",
            "G2",
            "manifests/gates/G2.json",
        ),
        ("UPSTREAM_GATE_STALE", "G3", "manifests/gates/G3.json"),
    )


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "scientific_candidate_sha",
        "scientific_candidate_tree_digest",
        "evaluator_attestation_ref",
        "predecessor_gate_refs",
    ],
)
def test_pre_g7_input_rejects_future_identity_or_caller_predecessor(
    tmp_path: Path,
    forbidden_field: str,
) -> None:
    GateEngineError, _, load_gate_input, _ = _gate_api()
    root, _ = _gate_repo(tmp_path)

    def add_forbidden(payload: dict[str, object]) -> None:
        if forbidden_field == "scientific_candidate_sha":
            payload[forbidden_field] = "e" * 40
        elif forbidden_field == "scientific_candidate_tree_digest":
            payload[forbidden_field] = SHA_B
        elif forbidden_field == "evaluator_attestation_ref":
            payload[forbidden_field] = {
                "artifact_id": "ART-STC-2001",
                "sha256": SHA_B,
            }
        else:
            payload[forbidden_field] = []

    _rewrite_gate_input(root, "G6", add_forbidden)

    with pytest.raises(GateEngineError) as captured:
        load_gate_input(root, "G6")

    assert _error_code(captured.value) == "GATE_INPUT_INVALID"


@pytest.mark.parametrize("duplicate_field", ["finding_ids", "adjudication_ids"])
def test_gate_input_rejects_duplicate_audit_identifiers(
    tmp_path: Path,
    duplicate_field: str,
) -> None:
    GateEngineError, _, load_gate_input, _ = _gate_api()
    root, _ = _gate_repo(tmp_path)
    _rewrite_gate_input(
        root,
        "G0",
        lambda payload: payload.__setitem__(
            duplicate_field,
            ["AUDIT-001", "AUDIT-001"],
        ),
    )

    with pytest.raises(GateEngineError) as captured:
        load_gate_input(root, "G0")

    assert _error_code(captured.value) == "GATE_INPUT_INVALID"
    assert captured.value.path == "manifests/gate-inputs/G0.json"
    assert not (root / "manifests" / "gates" / "G0.json").exists()


@pytest.mark.parametrize(
    ("invalid_document", "expected_code", "expected_path"),
    [
        (
            "gate-input-known-duplicate",
            "GATE_INPUT_INVALID",
            "manifests/gate-inputs/G0.json",
        ),
        (
            "gate-input-unknown-duplicate",
            "GATE_INPUT_INVALID",
            "manifests/gate-inputs/G0.json",
        ),
        (
            "gate-input-invalid-utf8",
            "GATE_INPUT_INVALID",
            "manifests/gate-inputs/G0.json",
        ),
        (
            "artifact-registry-known-duplicate",
            "ARTIFACT_REGISTRY_INVALID",
            "registry/artifacts.jsonl",
        ),
        (
            "artifact-registry-unknown-duplicate",
            "ARTIFACT_REGISTRY_INVALID",
            "registry/artifacts.jsonl",
        ),
        (
            "artifact-registry-invalid-utf8",
            "ARTIFACT_REGISTRY_INVALID",
            "registry/artifacts.jsonl",
        ),
    ],
)
def test_loader_rejects_ambiguous_or_non_utf8_authorization_json(
    tmp_path: Path,
    invalid_document: str,
    expected_code: str,
    expected_path: str,
) -> None:
    GateEngineError, _, load_gate_input, _ = _gate_api()
    root, environment_refs = _gate_repo(tmp_path)
    gate_input_path = root / "manifests" / "gate-inputs" / "G0.json"
    registry_path = root / "registry" / "artifacts.jsonl"
    secret_key = b"do-not-echo-secret-key"
    secret_value = b"do-not-echo-secret-value"

    if invalid_document.startswith("gate-input"):
        raw = gate_input_path.read_bytes()
        if invalid_document == "gate-input-known-duplicate":
            digest_token = (
                b'"sha256":"' + environment_refs["a"]["sha256"].encode() + b'"'
            )
            replacement = (
                b'"sha256":"'
                + secret_value
                + b'","sha256":"'
                + environment_refs["a"]["sha256"].encode()
                + b'"'
            )
            assert raw.count(digest_token) == 1
            raw = raw.replace(digest_token, replacement, 1)
        elif invalid_document == "gate-input-unknown-duplicate":
            ref_bytes = _canonical_json(environment_refs["a"]).rstrip(b"\n")
            duplicate = (
                ref_bytes[:-1]
                + b',"'
                + secret_key
                + b'":"first","'
                + secret_key
                + b'":"second"}'
            )
            assert raw.count(ref_bytes) == 1
            raw = raw.replace(ref_bytes, duplicate, 1)
        else:
            raw = raw.rstrip(b"\n") + b"\xff\n"
        gate_input_path.write_bytes(raw)
    else:
        raw = registry_path.read_bytes()
        if invalid_document == "artifact-registry-known-duplicate":
            digest_token = (
                b'"output_digest":"'
                + environment_refs["a"]["sha256"].encode()
                + b'"'
            )
            replacement = (
                b'"output_digest":"'
                + secret_value
                + b'","output_digest":"'
                + environment_refs["a"]["sha256"].encode()
                + b'"'
            )
            assert raw.count(digest_token) == 1
            raw = raw.replace(digest_token, replacement, 1)
        elif invalid_document == "artifact-registry-unknown-duplicate":
            insertion_point = b'"output_digest":'
            duplicate = (
                b'"metadata":{"'
                + secret_key
                + b'":"first","'
                + secret_key
                + b'":"second"},'
                + insertion_point
            )
            assert raw.count(insertion_point) == 2
            raw = raw.replace(insertion_point, duplicate, 1)
        else:
            raw = raw + b"\xff"
        registry_path.write_bytes(raw)
        _rewrite_gate_input(
            root,
            "G0",
            lambda payload: payload.__setitem__(
                "artifact_dag_digest",
                _sha256(raw),
            ),
        )

    with pytest.raises(GateEngineError) as captured:
        load_gate_input(root, "G0")

    assert _error_code(captured.value) == expected_code
    assert captured.value.gate_id == "G0"
    assert captured.value.path == expected_path
    diagnostic_bytes = repr(captured.value).encode("utf-8")
    assert secret_key not in diagnostic_bytes
    assert secret_value not in diagnostic_bytes
    assert not (root / "manifests" / "gates" / "G0.json").exists()


def test_loader_rejects_referenced_artifact_symlink_escape(
    tmp_path: Path,
) -> None:
    GateEngineError, _, load_gate_input, _ = _gate_api()
    root, _ = _gate_repo(tmp_path)
    artifact_path = root / "inputs" / "environment-a.json"
    original = artifact_path.read_bytes()
    outside = tmp_path / "outside-environment.json"
    outside.write_bytes(original)
    artifact_path.unlink()
    artifact_path.symlink_to(outside)

    with pytest.raises(GateEngineError) as captured:
        load_gate_input(root, "G0")

    assert _error_code(captured.value) == "INPUT_ARTIFACT_UNSAFE"
    assert captured.value.gate_id == "G0"
    assert captured.value.path == "inputs/environment-a.json"
    assert outside.read_bytes() == original
    assert not (root / "manifests" / "gates" / "G0.json").exists()


@pytest.mark.parametrize("unsafe_kind", ["file-symlink", "directory-symlink"])
def test_canonical_gate_output_rejects_symlink_escape(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    GateEngineError, evaluate_gate, load_gate_input, _ = _gate_api()
    root, _ = _gate_repo(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.json"
    sentinel.write_bytes(b"do not overwrite\n")
    gates = root / "manifests" / "gates"
    if unsafe_kind == "directory-symlink":
        gates.parent.mkdir(parents=True, exist_ok=True)
        gates.symlink_to(outside, target_is_directory=True)
    else:
        gates.mkdir(parents=True)
        (gates / "G0.json").symlink_to(sentinel)

    with pytest.raises(GateEngineError) as captured:
        evaluate_gate(load_gate_input(root, "G0"))

    assert _error_code(captured.value) == "GATE_OUTPUT_UNSAFE"
    assert sentinel.read_bytes() == b"do not overwrite\n"


def test_atomic_replace_failure_leaves_no_gate_or_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, evaluate_gate, load_gate_input, _ = _gate_api()
    gate_engine = importlib.import_module("stc_research.gate_engine")
    root, _ = _gate_repo(tmp_path)
    gate_input = load_gate_input(root, "G0")

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("injected atomic replace failure")

    monkeypatch.setattr(gate_engine.os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected atomic replace failure"):
        evaluate_gate(gate_input)

    gates = root / "manifests" / "gates"
    assert not (gates / "G0.json").exists()
    assert not tuple(gates.glob(".stc-gate-*"))


def test_failed_atomic_reevaluation_preserves_previous_canonical_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, evaluate_gate, load_gate_input, _ = _gate_api()
    gate_engine = importlib.import_module("stc_research.gate_engine")
    root, _ = _gate_repo(tmp_path)
    evaluate_gate(load_gate_input(root, "G0"))
    gates = root / "manifests" / "gates"
    output = gates / "G0.json"
    previous_bytes = output.read_bytes()
    _rewrite_gate_input(
        root,
        "G0",
        lambda payload: payload.__setitem__(
            "evaluator_id",
            "authorized-byte-different-reevaluation",
        ),
    )
    replacement_input = load_gate_input(root, "G0")

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("injected reevaluation replace failure")

    monkeypatch.setattr(gate_engine.os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected reevaluation replace failure"):
        evaluate_gate(replacement_input)

    assert output.read_bytes() == previous_bytes
    assert not tuple(gates.glob(".stc-gate-*"))


def test_gate_artifact_identity_cannot_be_reused_by_a_successor(
    tmp_path: Path,
) -> None:
    GateEngineError, evaluate_gate, load_gate_input, _ = _gate_api()
    root, _ = _gate_repo(tmp_path)
    evaluate_gate(load_gate_input(root, "G0"))
    _rewrite_gate_input(
        root,
        "G1",
        lambda payload: payload.__setitem__(
            "gate_artifact_id",
            _gate_artifact_id("G0"),
        ),
    )

    with pytest.raises(GateEngineError) as captured:
        evaluate_gate(load_gate_input(root, "G1"))

    assert _error_code(captured.value) == "GATE_ARTIFACT_ID_REUSED"


def _refresh_trust_set_digest(payload: dict[str, object]) -> None:
    unsigned = deepcopy(payload)
    unsigned.pop("set_digest", None)
    payload["set_digest"] = _sha256(_canonical_json(unsigned))


def _sign_attestation(
    tmp_path: Path,
    payload: dict[str, object],
    *,
    private_key_pem: str = TEST_ED25519_PRIVATE_KEY,
) -> None:
    unsigned = deepcopy(payload)
    unsigned.pop("signature", None)
    private_key = tmp_path / "fixture-ed25519-private-key.pem"
    message = tmp_path / "fixture-ed25519-message.json"
    private_key.write_text(private_key_pem, encoding="ascii")
    message.write_bytes(_canonical_json(unsigned))
    completed = subprocess.run(
        [
            "openssl",
            "pkeyutl",
            "-sign",
            "-rawin",
            "-inkey",
            os.fspath(private_key),
            "-in",
            os.fspath(message),
        ],
        check=True,
        capture_output=True,
    )
    assert len(completed.stdout) == 64
    payload["signature"] = base64.b64encode(completed.stdout).decode("ascii")


def _g7_subject_digest(refs: list[dict[str, str]]) -> str:
    ordered = sorted(refs, key=lambda ref: ref["path"])
    return _sha256(_canonical_json({"subject_refs": ordered}))


def _noncanonical_base64_pad_bits(encoded: str) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    padding = len(encoded) - len(encoded.rstrip("="))
    assert padding in {1, 2}
    data_index = len(encoded) - padding - 1
    canonical_index = alphabet.index(encoded[data_index])
    unused_bits = 2 if padding == 1 else 4
    assert canonical_index % (1 << unused_bits) == 0
    alias = (
        encoded[:data_index]
        + alphabet[canonical_index + 1]
        + encoded[data_index + 1 :]
    )
    assert base64.b64decode(alias, validate=True) == base64.b64decode(
        encoded,
        validate=True,
    )
    assert base64.b64encode(base64.b64decode(alias, validate=True)).decode(
        "ascii"
    ) == encoded
    return alias


def _trust_set_payload(*, split_gate_keys: bool = False) -> dict[str, object]:
    keys: list[dict[str, object]] = [
        {
            "key_id": "reviewer-key-1",
            "signer_id": "reviewer-1",
            "algorithm": "Ed25519",
            "public_key": TEST_ED25519_SPKI_BASE64,
            "roles": [
                "release_evaluator",
                "independent_auditor",
            ],
            "valid_from": "2026-07-01T00:00:00Z",
            "valid_until": "2027-07-01T00:00:00Z",
            "revoked": False,
        }
    ]
    if split_gate_keys:
        keys.append(
            {
                "key_id": "reviewer-key-2",
                "signer_id": "reviewer-2",
                "algorithm": "Ed25519",
                "public_key": TEST_ED25519_SPKI_BASE64_2,
                "roles": ["release_evaluator"],
                "valid_from": "2026-07-01T00:00:00Z",
                "valid_until": "2027-07-01T00:00:00Z",
                "revoked": False,
            }
        )
    payload: dict[str, object] = {
        "key_set_id": "stc-reviewer-keys",
        "version": 1,
        "keys": keys,
        "issued_at": "2026-07-01T00:00:00Z",
    }
    _refresh_trust_set_digest(payload)
    return payload


def _attested_gate_repo(
    tmp_path: Path,
    *,
    split_gate_keys: bool = False,
) -> tuple[Path, dict[str, object]]:
    root, environment_refs = _gate_repo(tmp_path)
    artifacts: dict[str, dict[str, object]] = {}

    def add_artifact(
        name: str,
        artifact_id: str,
        relative_path: str,
        artifact_type: str,
        payload: bytes,
    ) -> dict[str, str]:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        ref = {
            "artifact_id": artifact_id,
            "sha256": _sha256(payload),
        }
        artifacts[name] = {
            "artifact_id": artifact_id,
            "path": relative_path,
            "artifact_type": artifact_type,
            "ref": ref,
        }
        return ref

    g7_audit_ref = add_artifact(
        "g7-audit",
        "ART-STC-2001",
        "reports/G7-independent-audit.md",
        "audit-report",
        b"# Independent G7 audit\n\nNo open critical findings.\n",
    )
    g7_findings_ref = add_artifact(
        "g7-findings",
        "ART-STC-2002",
        "reports/G7-audit-findings.jsonl",
        "audit-findings",
        b"",
    )
    g8_subject_bytes = _canonical_json(
        {
            "gate_id": "G8",
            "candidate": "stc-paper-v1.0.0-rc1",
            "review_scope": "complete release evaluation",
        }
    )
    g8_subject_ref = add_artifact(
        "g8-subject",
        "ART-STC-2003",
        "manifests/gate-inputs/G8-evaluation-subject.json",
        "gate-evaluation-subject",
        g8_subject_bytes,
    )
    g8_smoke_ref = add_artifact(
        "g8-smoke",
        "ART-STC-2004",
        "reports/release-candidate-smoke.json",
        "release-smoke",
        _canonical_json({"status": "PASS", "rights": "PASS"}),
    )

    trust_payload = _trust_set_payload(split_gate_keys=split_gate_keys)
    trust_ref = add_artifact(
        "trust",
        "ART-STC-2010",
        "manifests/trusted-reviewer-keys.json",
        "trusted-key-set",
        _canonical_json(trust_payload),
    )

    g7_subject_refs = sorted(
        [
            {
                "path": str(artifacts["g7-audit"]["path"]),
                "sha256": g7_audit_ref["sha256"],
            },
            {
                "path": str(artifacts["g7-findings"]["path"]),
                "sha256": g7_findings_ref["sha256"],
            },
        ],
        key=lambda ref: ref["path"],
    )
    g7_attestation: dict[str, object] = {
        "attestation_id": "ATT-G7-001",
        "subject_sha256": _g7_subject_digest(g7_subject_refs),
        "subject_refs": g7_subject_refs,
        "signer_id": "reviewer-1",
        "signer_role": "independent_auditor",
        "independence_mode": "fresh-context",
        "author_executor_roster_digest": SHA_B,
        "algorithm": "Ed25519",
        "key_id": "reviewer-key-1",
        "signature": "",
        "signed_at": "2026-07-25T02:00:00Z",
    }
    _sign_attestation(tmp_path, g7_attestation)
    g7_attestation_ref = add_artifact(
        "g7-attestation",
        "ART-STC-2021",
        "manifests/attestations/G7-auditor.json",
        "audit-attestation",
        _canonical_json(g7_attestation),
    )

    g8_subject_refs = sorted(
        [
            {
                "path": str(artifacts["g8-subject"]["path"]),
                "sha256": g8_subject_ref["sha256"],
            },
            {
                "path": str(artifacts["g8-smoke"]["path"]),
                "sha256": g8_smoke_ref["sha256"],
            },
        ],
        key=lambda ref: ref["path"],
    )
    g8_signer_id = "reviewer-2" if split_gate_keys else "reviewer-1"
    g8_key_id = "reviewer-key-2" if split_gate_keys else "reviewer-key-1"
    g8_attestation: dict[str, object] = {
        "attestation_id": "ATT-G8-001",
        "gate_id": "G8",
        "subject_sha256": g8_subject_ref["sha256"],
        "subject_refs": g8_subject_refs,
        "signer_id": g8_signer_id,
        "signer_role": "release_evaluator",
        "independence_mode": "fresh-context",
        "algorithm": "Ed25519",
        "key_id": g8_key_id,
        "signature": "",
        "signed_at": "2026-07-25T02:00:00Z",
    }
    _sign_attestation(
        tmp_path,
        g8_attestation,
        private_key_pem=(
            TEST_ED25519_PRIVATE_KEY_2
            if split_gate_keys
            else TEST_ED25519_PRIVATE_KEY
        ),
    )
    g8_attestation_ref = add_artifact(
        "g8-attestation",
        "ART-STC-2022",
        "manifests/attestations/G8-evaluator.json",
        "gate-evaluation-attestation",
        _canonical_json(g8_attestation),
    )

    registry_path = root / "registry" / "artifacts.jsonl"
    nodes = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    for artifact in artifacts.values():
        artifact_path = root / str(artifact["path"])
        nodes.append(
            _artifact_node(
                str(artifact["artifact_id"]),
                str(artifact["path"]),
                artifact_path.read_bytes(),
                artifact_type=str(artifact["artifact_type"]),
            )
        )
    registry_bytes = _write_jsonl(registry_path, nodes)
    artifact_dag_digest = _sha256(registry_bytes)

    for gate_number in range(9):
        gate_id = f"G{gate_number}"

        def bind_gate(
            payload: dict[str, object],
            *,
            gate_id: str = gate_id,
        ) -> None:
            payload["artifact_dag_digest"] = artifact_dag_digest
            if gate_id == "G7":
                payload.update(
                    {
                        "scientific_candidate_sha": "e" * 40,
                        "scientific_candidate_tree_digest": SHA_C,
                        "input_refs": [
                            environment_refs["a"],
                            g7_audit_ref,
                            g7_findings_ref,
                            trust_ref,
                            g7_attestation_ref,
                        ],
                        "evaluator_id": g8_signer_id,
                        "evaluator_role": "independent_auditor",
                        "independence_mode": "fresh-context",
                        "evaluator_attestation_ref": g7_attestation_ref,
                    }
                )
            elif gate_id == "G8":
                payload.update(
                    {
                        "scientific_candidate_sha": "e" * 40,
                        "scientific_candidate_tree_digest": SHA_C,
                        "input_refs": [
                            environment_refs["a"],
                            g8_subject_ref,
                            g8_smoke_ref,
                            trust_ref,
                            g8_attestation_ref,
                        ],
                        "evaluator_id": "reviewer-1",
                        "evaluator_role": "release_evaluator",
                        "independence_mode": "fresh-context",
                        "evaluator_attestation_ref": g8_attestation_ref,
                    }
                )

        _rewrite_gate_input(root, gate_id, bind_gate)

    context: dict[str, object] = {
        "artifacts": artifacts,
        "trust_payload": trust_payload,
        "g7_attestation": g7_attestation,
        "g8_attestation": g8_attestation,
        "environment_ref": environment_refs["a"],
    }
    return root, context


def _rewrite_bound_artifact(
    root: Path,
    *,
    artifact_id: str,
    relative_path: str,
    payload: bytes,
) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = _sha256(payload)
    registry_path = root / "registry" / "artifacts.jsonl"
    nodes = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    matches = [node for node in nodes if node["artifact_id"] == artifact_id]
    assert len(matches) == 1
    matches[0]["path"] = relative_path
    matches[0]["output_digest"] = digest
    registry_digest = _sha256(_write_jsonl(registry_path, nodes))

    for gate_number in range(9):
        gate_id = f"G{gate_number}"

        def rebind(payload_object: dict[str, object]) -> None:
            payload_object["artifact_dag_digest"] = registry_digest
            refs = payload_object["input_refs"]
            assert isinstance(refs, list)
            for ref in refs:
                if isinstance(ref, dict) and ref.get("artifact_id") == artifact_id:
                    ref["sha256"] = digest
            attestation_ref = payload_object["evaluator_attestation_ref"]
            if (
                isinstance(attestation_ref, dict)
                and attestation_ref.get("artifact_id") == artifact_id
            ):
                attestation_ref["sha256"] = digest

        _rewrite_gate_input(root, gate_id, rebind)


def _mutate_registry_node(
    root: Path,
    artifact_id: str,
    mutate,
) -> None:
    registry_path = root / "registry" / "artifacts.jsonl"
    nodes = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    matches = [node for node in nodes if node["artifact_id"] == artifact_id]
    assert len(matches) == 1
    mutate(matches[0])
    registry_digest = _sha256(_write_jsonl(registry_path, nodes))
    for gate_number in range(9):
        _rewrite_gate_input(
            root,
            f"G{gate_number}",
            lambda payload, digest=registry_digest: payload.__setitem__(
                "artifact_dag_digest",
                digest,
            ),
        )


def _evaluate_through(root: Path, gate_number: int) -> None:
    _, evaluate_gate, load_gate_input, _ = _gate_api()
    for number in range(gate_number + 1):
        gate_id = f"G{number}"
        evaluate_gate(load_gate_input(root, gate_id))


def _attestation_fixture(
    context: dict[str, object],
    gate_id: str,
) -> tuple[dict[str, object], dict[str, object]]:
    artifacts = context["artifacts"]
    assert isinstance(artifacts, dict)
    artifact = artifacts[
        "g7-attestation" if gate_id == "G7" else "g8-attestation"
    ]
    payload = context[
        "g7_attestation" if gate_id == "G7" else "g8_attestation"
    ]
    assert isinstance(artifact, dict)
    assert isinstance(payload, dict)
    return artifact, deepcopy(payload)


def _authority_artifact(
    context: dict[str, object],
    target: str,
) -> dict[str, object]:
    artifacts = context["artifacts"]
    assert isinstance(artifacts, dict)
    artifact = artifacts[target]
    assert isinstance(artifact, dict)
    return artifact


def _expected_attestation_path(gate_id: str) -> str:
    return (
        "manifests/attestations/G7-auditor.json"
        if gate_id == "G7"
        else "manifests/attestations/G8-evaluator.json"
    )


def _required_evaluator_role(gate_id: str) -> str:
    return "independent_auditor" if gate_id == "G7" else "release_evaluator"


def _other_evaluator_role(gate_id: str) -> str:
    return "release_evaluator" if gate_id == "G7" else "independent_auditor"


def _rewrite_attestation_payload(
    root: Path,
    artifact: dict[str, object],
    payload: dict[str, object],
) -> None:
    _rewrite_bound_artifact(
        root,
        artifact_id=str(artifact["artifact_id"]),
        relative_path=str(artifact["path"]),
        payload=_canonical_json(payload),
    )


def _rewrite_trust_payload(
    root: Path,
    artifact: dict[str, object],
    payload: dict[str, object],
) -> None:
    _rewrite_bound_artifact(
        root,
        artifact_id=str(artifact["artifact_id"]),
        relative_path=str(artifact["path"]),
        payload=_canonical_json(payload),
    )


def _add_ambiguous_subject_binding(
    root: Path,
    *,
    gate_id: str,
    source_artifact: dict[str, object],
) -> None:
    registry_path = root / "registry" / "artifacts.jsonl"
    records = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    source_path = root / str(source_artifact["path"])
    shadow_id = "ART-STC-2087" if gate_id == "G7" else "ART-STC-2088"
    records.append(
        _artifact_node(
            shadow_id,
            str(source_artifact["path"]),
            source_path.read_bytes(),
            artifact_type=str(source_artifact["artifact_type"]),
        )
    )
    registry_digest = _sha256(_write_jsonl(registry_path, records))
    for gate_number in range(9):
        current_gate_id = f"G{gate_number}"

        def bind_shadow(
            payload: dict[str, object],
            *,
            current_gate_id: str = current_gate_id,
        ) -> None:
            payload["artifact_dag_digest"] = registry_digest
            if current_gate_id == gate_id:
                refs = payload["input_refs"]
                assert isinstance(refs, list)
                refs.append(
                    {
                        "artifact_id": shadow_id,
                        "sha256": str(source_artifact["ref"]["sha256"]),
                    }
                )

        _rewrite_gate_input(root, current_gate_id, bind_shadow)


def _coherent_public_gate_input(
    root: Path,
    gate_id: str,
    loaded,
):
    from stc_research.models import (
        DigestRef,
        GateArtifactBinding,
        GateAuthorizationSnapshot,
    )

    input_path = root / "manifests" / "gate-inputs" / f"{gate_id}.json"
    registry_path = root / "registry" / "artifacts.jsonl"
    canonical_payload = json.loads(input_path.read_text(encoding="utf-8"))
    registry_records = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    input_refs = tuple(
        DigestRef.from_dict(ref)
        for ref in canonical_payload["input_refs"]
    )
    bindings: list[GateArtifactBinding] = []
    for ref in input_refs:
        matches = [
            node
            for node in registry_records
            if node["artifact_id"] == ref.artifact_id
        ]
        assert len(matches) == 1
        node = matches[0]
        relative_path = str(node["path"])
        artifact_bytes = (root / relative_path).read_bytes()
        bindings.append(
            GateArtifactBinding(
                artifact_id=ref.artifact_id,
                relative_path=relative_path,
                resolved_path=(root / relative_path).resolve(),
                declared_sha256=str(node["output_digest"]),
                loaded_sha256=_sha256(artifact_bytes),
            )
        )
    evaluator_ref = DigestRef.from_dict(
        canonical_payload["evaluator_attestation_ref"]
    )
    coherent = replace(
        loaded,
        artifact_dag_digest=canonical_payload["artifact_dag_digest"],
        input_refs=input_refs,
        evaluator_attestation_ref=evaluator_ref,
        authorization_snapshot=GateAuthorizationSnapshot(
            gate_input_sha256=_sha256(input_path.read_bytes()),
            artifact_registry_sha256=_sha256(registry_path.read_bytes()),
            artifact_bindings=tuple(bindings),
        ),
    )
    assert coherent.to_dict() == canonical_payload
    return coherent


def _write_exact_forged_gate_record(root: Path, gate_id: str) -> None:
    from stc_research.models import GateRecord

    gate_number = int(gate_id[1:])
    assert gate_number > 0
    input_payload = json.loads(
        (
            root / "manifests" / "gate-inputs" / f"{gate_id}.json"
        ).read_text(encoding="utf-8")
    )
    predecessor_path = (
        root / "manifests" / "gates" / f"G{gate_number - 1}.json"
    )
    predecessor_bytes = predecessor_path.read_bytes()
    predecessor = json.loads(predecessor_bytes)
    record_payload = dict(input_payload)
    record_payload["predecessor_gate_refs"] = [
        {
            "artifact_id": predecessor["gate_artifact_id"],
            "sha256": _sha256(predecessor_bytes),
        }
    ]
    record = GateRecord.from_dict(record_payload)
    _write_json(
        root / "manifests" / "gates" / f"{gate_id}.json",
        record.to_dict(),
    )


def test_real_ed25519_attestations_authorize_g7_and_g8(
    tmp_path: Path,
) -> None:
    _, _, _, verify_gate_chain = _gate_api()
    root, context = _attested_gate_repo(tmp_path)
    g7_attestation = context["g7_attestation"]
    g8_attestation = context["g8_attestation"]
    assert isinstance(g7_attestation, dict)
    assert isinstance(g8_attestation, dict)
    assert g7_attestation["signature"] == TEST_G7_SIGNATURE_BASE64
    assert g8_attestation["signature"] == TEST_G8_SIGNATURE_BASE64
    assert g7_attestation["subject_sha256"] == (
        "37521112f167bc3e84b2fac6cb727486cc18149f04a3e6c09a86478d63c43623"
    )

    _evaluate_through(root, 8)

    g7 = json.loads(
        (root / "manifests" / "gates" / "G7.json").read_text(encoding="utf-8")
    )
    g8 = json.loads(
        (root / "manifests" / "gates" / "G8.json").read_text(encoding="utf-8")
    )
    assert g7["evaluator_role"] == "independent_auditor"
    assert g8["evaluator_role"] == "release_evaluator"
    assert g7["evaluator_attestation_ref"]["artifact_id"] == "ART-STC-2021"
    assert g8["evaluator_attestation_ref"]["artifact_id"] == "ART-STC-2022"
    report = verify_gate_chain(root, through="G8")
    assert report.success
    assert report.diagnostics == ()


@pytest.mark.parametrize("gate_id", ["G7", "G8"])
@pytest.mark.parametrize(
    "forged_field",
    [
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
        "repository_root",
        "input_path",
        "output_path",
        "authorization_snapshot",
    ],
)
def test_evaluate_rejects_any_public_gate_input_difference_from_canonical_bytes(
    tmp_path: Path,
    gate_id: str,
    forged_field: str,
) -> None:
    from stc_research.models import (
        DigestRef,
        GateAuthorizationSnapshot,
    )

    GateEngineError, evaluate_gate, load_gate_input, _ = _gate_api()
    root, _ = _attested_gate_repo(tmp_path)
    _evaluate_through(root, int(gate_id[1:]) - 1)
    loaded = load_gate_input(root, gate_id)
    replacements: dict[str, object]
    victim: Path | None = None
    victim_before: bytes | None = None
    if forged_field == "gate_id":
        replacements = {"gate_id": "G8" if gate_id == "G7" else "G7"}
    elif forged_field == "gate_artifact_id":
        replacements = {"gate_artifact_id": "ART-STC-2099"}
    elif forged_field == "schema_version":
        replacements = {"schema_version": "secret-forged-schema"}
    elif forged_field == "evaluation_commit_sha":
        replacements = {"evaluation_commit_sha": "f" * 40}
    elif forged_field == "evaluation_tree_digest":
        replacements = {"evaluation_tree_digest": SHA_B}
    elif forged_field == "scientific_candidate_sha":
        replacements = {"scientific_candidate_sha": "f" * 40}
    elif forged_field == "scientific_candidate_tree_digest":
        replacements = {"scientific_candidate_tree_digest": SHA_B}
    elif forged_field == "artifact_dag_digest":
        replacements = {"artifact_dag_digest": SHA_A}
    elif forged_field == "input_refs":
        replacements = {"input_refs": loaded.input_refs[:-1]}
    elif forged_field == "evaluator_id":
        replacements = {"evaluator_id": "forged-reviewer"}
    elif forged_field == "evaluator_role":
        replacements = {"evaluator_role": "forged-role"}
    elif forged_field == "independence_mode":
        replacements = {"independence_mode": "forged-independence"}
    elif forged_field == "evaluator_attestation_ref":
        replacements = {
            "evaluator_attestation_ref": DigestRef(
                artifact_id="ART-STC-2010",
                sha256=next(
                    ref.sha256
                    for ref in loaded.input_refs
                    if ref.artifact_id == "ART-STC-2010"
                ),
            )
        }
    elif forged_field == "finding_ids":
        replacements = {"finding_ids": ("secret-forged-finding",)}
    elif forged_field == "adjudication_ids":
        replacements = {"adjudication_ids": ("secret-forged-adjudication",)}
    elif forged_field == "reaudit_refs":
        replacements = {"reaudit_refs": (loaded.input_refs[0],)}
    elif forged_field == "environment_digest":
        replacements = {"environment_digest": SHA_B}
    elif forged_field == "evaluated_at":
        replacements = {"evaluated_at": "2026-07-25T12:00:01+09:00"}
    elif forged_field == "status":
        replacements = {"status": type(loaded.status).FAIL}
    elif forged_field == "repository_root":
        alternate_root, _ = _attested_gate_repo(tmp_path / "alternate")
        replacements = {
            "repository_root": alternate_root.resolve(),
            "input_path": (
                alternate_root / "manifests" / "gate-inputs" / f"{gate_id}.json"
            ).resolve(),
            "output_path": (
                alternate_root / "manifests" / "gates" / f"{gate_id}.json"
            ).resolve(),
        }
    elif forged_field == "input_path":
        replacements = {
            "input_path": (root / "inputs" / "environment-a.json").resolve()
        }
    elif forged_field == "output_path":
        victim = root / "inputs" / "environment-a.json"
        victim_before = victim.read_bytes()
        replacements = {"output_path": victim.resolve()}
    else:
        replacements = {
            "authorization_snapshot": GateAuthorizationSnapshot(
                gate_input_sha256=SHA_A,
                artifact_registry_sha256=(
                    loaded.authorization_snapshot.artifact_registry_sha256
                ),
                artifact_bindings=(
                    loaded.authorization_snapshot.artifact_bindings
                ),
            )
        }
    forged = replace(loaded, **replacements)

    try:
        with pytest.raises(GateEngineError) as captured:
            evaluate_gate(forged)
    finally:
        if victim is not None:
            assert victim_before is not None
            assert victim.read_bytes() == victim_before

    assert _error_code(captured.value) == "GATE_INPUT_OBJECT_MISMATCH"
    assert captured.value.path == f"manifests/gate-inputs/{gate_id}.json"
    assert "forged" not in repr(captured.value).lower()
    assert not (root / "manifests" / "gates" / f"{gate_id}.json").exists()


@pytest.mark.parametrize("gate_id", ["G7", "G8"])
@pytest.mark.parametrize(
    ("invalid_case", "expected_code", "expected_location"),
    [
        ("evaluator-ref-unbound", "EVALUATOR_ATTESTATION_UNBOUND", "input"),
        (
            "evaluator-ref-duplicate",
            "EVALUATOR_ATTESTATION_AMBIGUOUS",
            "input",
        ),
        ("trust-ref-unbound", "TRUSTED_KEY_SET_UNBOUND", "input"),
        ("trust-ref-duplicate", "TRUSTED_KEY_SET_AMBIGUOUS", "input"),
        (
            "evaluator-wrong-path",
            "EVALUATOR_ATTESTATION_PATH_INVALID",
            "wrong-attestation",
        ),
        (
            "trust-wrong-path",
            "TRUSTED_KEY_SET_PATH_INVALID",
            "wrong-trust",
        ),
        (
            "evaluator-wrong-type",
            "EVALUATOR_ATTESTATION_TYPE_INVALID",
            "attestation",
        ),
        ("trust-wrong-type", "TRUSTED_KEY_SET_TYPE_INVALID", "trust"),
        (
            "evaluator-pretty-json",
            "EVALUATOR_ATTESTATION_INVALID",
            "attestation",
        ),
        (
            "evaluator-duplicate-json",
            "EVALUATOR_ATTESTATION_INVALID",
            "attestation",
        ),
        (
            "evaluator-invalid-utf8",
            "EVALUATOR_ATTESTATION_INVALID",
            "attestation",
        ),
        (
            "evaluator-missing-lf",
            "EVALUATOR_ATTESTATION_INVALID",
            "attestation",
        ),
        (
            "evaluator-unsorted-subject-refs",
            "EVALUATOR_ATTESTATION_INVALID",
            "attestation",
        ),
        (
            "evaluator-schema-missing",
            "EVALUATOR_ATTESTATION_INVALID",
            "attestation",
        ),
        (
            "evaluator-schema-extra",
            "EVALUATOR_ATTESTATION_INVALID",
            "attestation",
        ),
        (
            "evaluator-schema-wrong-type",
            "EVALUATOR_ATTESTATION_INVALID",
            "attestation",
        ),
        ("trust-pretty-json", "TRUSTED_KEY_SET_INVALID", "trust"),
        ("trust-duplicate-json", "TRUSTED_KEY_SET_INVALID", "trust"),
        ("trust-invalid-utf8", "TRUSTED_KEY_SET_INVALID", "trust"),
        ("trust-missing-lf", "TRUSTED_KEY_SET_INVALID", "trust"),
        ("trust-schema-missing", "TRUSTED_KEY_SET_INVALID", "trust"),
        ("trust-schema-extra", "TRUSTED_KEY_SET_INVALID", "trust"),
        ("trust-schema-wrong-type", "TRUSTED_KEY_SET_INVALID", "trust"),
        (
            "wrapper-identity-mismatch",
            "ATTESTATION_IDENTITY_MISMATCH",
            "input",
        ),
        (
            "wrapper-role-mismatch",
            "ATTESTATION_IDENTITY_MISMATCH",
            "input",
        ),
        (
            "wrapper-independence-mismatch",
            "ATTESTATION_IDENTITY_MISMATCH",
            "input",
        ),
        ("revoked-key", "ATTESTATION_KEY_INVALID", "trust"),
        ("unknown-key", "ATTESTATION_KEY_INVALID", "attestation"),
        ("signer-key-mismatch", "ATTESTATION_KEY_INVALID", "trust"),
        ("wrong-required-role", "ATTESTATION_ROLE_INVALID", "attestation"),
        ("key-missing-role", "ATTESTATION_ROLE_INVALID", "trust"),
        ("key-not-yet-valid", "ATTESTATION_TIME_INVALID", "trust"),
        ("key-expired", "ATTESTATION_TIME_INVALID", "trust"),
        ("key-set-issued-after-signing", "ATTESTATION_TIME_INVALID", "trust"),
        ("signed-after-evaluation", "ATTESTATION_TIME_INVALID", "attestation"),
        ("wrong-attestation-algorithm", "ATTESTATION_KEY_INVALID", "attestation"),
        ("wrong-key-algorithm", "ATTESTATION_KEY_INVALID", "trust"),
        ("malformed-public-key", "ATTESTATION_KEY_INVALID", "trust"),
        ("malformed-spki-der", "ATTESTATION_KEY_INVALID", "trust"),
        ("wrong-public-key-type", "ATTESTATION_KEY_INVALID", "trust"),
        ("public-key-trailing-garbage", "ATTESTATION_KEY_INVALID", "trust"),
        ("public-key-whitespace", "ATTESTATION_KEY_INVALID", "trust"),
        ("public-key-unpadded", "ATTESTATION_KEY_INVALID", "trust"),
        ("public-key-extra-padding", "ATTESTATION_KEY_INVALID", "trust"),
        ("public-key-nonzero-pad-bits", "ATTESTATION_KEY_INVALID", "trust"),
        ("public-key-trailing-der", "ATTESTATION_KEY_INVALID", "trust"),
        (
            "malformed-signature",
            "ATTESTATION_SIGNATURE_INVALID",
            "attestation",
        ),
        ("wrong-signature", "ATTESTATION_SIGNATURE_INVALID", "attestation"),
        ("short-signature", "ATTESTATION_SIGNATURE_INVALID", "attestation"),
        ("long-signature", "ATTESTATION_SIGNATURE_INVALID", "attestation"),
        (
            "signature-trailing-garbage",
            "ATTESTATION_SIGNATURE_INVALID",
            "attestation",
        ),
        (
            "signature-whitespace",
            "ATTESTATION_SIGNATURE_INVALID",
            "attestation",
        ),
        (
            "signature-unpadded",
            "ATTESTATION_SIGNATURE_INVALID",
            "attestation",
        ),
        (
            "signature-extra-padding",
            "ATTESTATION_SIGNATURE_INVALID",
            "attestation",
        ),
        (
            "signature-nonzero-pad-bits",
            "ATTESTATION_SIGNATURE_INVALID",
            "attestation",
        ),
        (
            "unsigned-attestation-id-tamper",
            "ATTESTATION_SIGNATURE_INVALID",
            "attestation",
        ),
        (
            "unsigned-independence-tamper",
            "ATTESTATION_SIGNATURE_INVALID",
            "attestation",
        ),
        ("trust-self-digest-mismatch", "TRUSTED_KEY_SET_INVALID", "trust"),
    ],
)
def test_g7_and_g8_apply_the_same_complete_authorization_contract(
    tmp_path: Path,
    gate_id: str,
    invalid_case: str,
    expected_code: str,
    expected_location: str,
) -> None:
    GateEngineError, _, load_gate_input, _ = _gate_api()
    root, context = _attested_gate_repo(tmp_path)
    attestation_artifact, attestation = _attestation_fixture(context, gate_id)
    trust_artifact = _authority_artifact(context, "trust")
    trust_payload = deepcopy(context["trust_payload"])
    assert isinstance(trust_payload, dict)
    wrong_attestation_path = (
        f"manifests/attestations/not-{gate_id}-evaluator.json"
    )
    wrong_trust_path = "manifests/not-trusted-reviewer-keys.json"

    if invalid_case in {
        "evaluator-ref-unbound",
        "evaluator-ref-duplicate",
        "trust-ref-unbound",
        "trust-ref-duplicate",
    }:
        artifact = (
            attestation_artifact
            if invalid_case.startswith("evaluator")
            else trust_artifact
        )
        artifact_id = str(artifact["artifact_id"])

        def change_binding(payload: dict[str, object]) -> None:
            refs = payload["input_refs"]
            assert isinstance(refs, list)
            matches = [
                ref
                for ref in refs
                if isinstance(ref, dict)
                and ref.get("artifact_id") == artifact_id
            ]
            assert len(matches) == 1
            if invalid_case.endswith("duplicate"):
                refs.append(deepcopy(matches[0]))
            else:
                payload["input_refs"] = [
                    ref
                    for ref in refs
                    if not (
                        isinstance(ref, dict)
                        and ref.get("artifact_id") == artifact_id
                    )
                ]

        _rewrite_gate_input(root, gate_id, change_binding)
    elif invalid_case == "evaluator-wrong-path":
        old_path = root / str(attestation_artifact["path"])
        _rewrite_bound_artifact(
            root,
            artifact_id=str(attestation_artifact["artifact_id"]),
            relative_path=wrong_attestation_path,
            payload=old_path.read_bytes(),
        )
    elif invalid_case == "trust-wrong-path":
        old_path = root / str(trust_artifact["path"])
        _rewrite_bound_artifact(
            root,
            artifact_id=str(trust_artifact["artifact_id"]),
            relative_path=wrong_trust_path,
            payload=old_path.read_bytes(),
        )
    elif invalid_case == "evaluator-wrong-type":
        _mutate_registry_node(
            root,
            str(attestation_artifact["artifact_id"]),
            lambda node: node.__setitem__(
                "artifact_type",
                (
                    "gate-evaluation-attestation"
                    if gate_id == "G7"
                    else "audit-attestation"
                ),
            ),
        )
    elif invalid_case == "trust-wrong-type":
        _mutate_registry_node(
            root,
            str(trust_artifact["artifact_id"]),
            lambda node: node.__setitem__(
                "artifact_type",
                "audit-attestation",
            ),
        )
    elif invalid_case in {
        "evaluator-schema-missing",
        "evaluator-schema-extra",
        "evaluator-schema-wrong-type",
    }:
        if invalid_case == "evaluator-schema-missing":
            attestation.pop("key_id")
        elif invalid_case == "evaluator-schema-extra":
            attestation["secret-extra-field"] = "secret-attestation-value"
        else:
            attestation["signed_at"] = {
                "secret-wrong-type": "secret-attestation-time"
            }
        _rewrite_attestation_payload(
            root,
            attestation_artifact,
            attestation,
        )
    elif invalid_case.startswith("evaluator-"):
        if invalid_case == "evaluator-pretty-json":
            encoded = (
                json.dumps(
                    attestation,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
        elif invalid_case == "evaluator-duplicate-json":
            encoded = _canonical_json(attestation).replace(
                b'"signer_id":"reviewer-1"',
                b'"signer_id":"secret-shadow","signer_id":"reviewer-1"',
                1,
            )
        elif invalid_case == "evaluator-invalid-utf8":
            encoded = _canonical_json(attestation).rstrip(b"\n") + b"\xff\n"
        elif invalid_case == "evaluator-missing-lf":
            encoded = _canonical_json(attestation).rstrip(b"\n")
        else:
            refs = attestation["subject_refs"]
            assert isinstance(refs, list)
            refs.reverse()
            _sign_attestation(tmp_path, attestation)
            encoded = _canonical_json(attestation)
        _rewrite_bound_artifact(
            root,
            artifact_id=str(attestation_artifact["artifact_id"]),
            relative_path=str(attestation_artifact["path"]),
            payload=encoded,
        )
    elif invalid_case in {
        "trust-schema-missing",
        "trust-schema-extra",
        "trust-schema-wrong-type",
    }:
        if invalid_case == "trust-schema-missing":
            trust_payload.pop("keys")
        elif invalid_case == "trust-schema-extra":
            trust_payload["secret-extra-field"] = "secret-trust-value"
        else:
            trust_payload["version"] = "secret-wrong-type"
        _refresh_trust_set_digest(trust_payload)
        _rewrite_trust_payload(root, trust_artifact, trust_payload)
    elif invalid_case.startswith("trust-") and invalid_case != (
        "trust-self-digest-mismatch"
    ):
        if invalid_case == "trust-pretty-json":
            encoded = (
                json.dumps(
                    trust_payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
        elif invalid_case == "trust-duplicate-json":
            encoded = _canonical_json(trust_payload).replace(
                b'"key_set_id":"stc-reviewer-keys"',
                (
                    b'"key_set_id":"secret-shadow",'
                    b'"key_set_id":"stc-reviewer-keys"'
                ),
                1,
            )
        elif invalid_case == "trust-invalid-utf8":
            encoded = _canonical_json(trust_payload).rstrip(b"\n") + b"\xff\n"
        else:
            encoded = _canonical_json(trust_payload).rstrip(b"\n")
        _rewrite_bound_artifact(
            root,
            artifact_id=str(trust_artifact["artifact_id"]),
            relative_path=str(trust_artifact["path"]),
            payload=encoded,
        )
    elif invalid_case == "wrapper-identity-mismatch":
        _rewrite_gate_input(
            root,
            gate_id,
            lambda payload: payload.__setitem__(
                "evaluator_id",
                "secret-wrapper-reviewer",
            ),
        )
    elif invalid_case == "wrapper-role-mismatch":
        _rewrite_gate_input(
            root,
            gate_id,
            lambda payload: payload.__setitem__(
                "evaluator_role",
                "secret-wrapper-role",
            ),
        )
    elif invalid_case == "wrapper-independence-mismatch":
        _rewrite_gate_input(
            root,
            gate_id,
            lambda payload: payload.__setitem__(
                "independence_mode",
                "secret-wrapper-independence",
            ),
        )
    elif invalid_case in {
        "unknown-key",
        "signer-key-mismatch",
        "wrong-required-role",
        "signed-after-evaluation",
        "wrong-attestation-algorithm",
        "malformed-signature",
        "wrong-signature",
        "short-signature",
        "long-signature",
        "signature-trailing-garbage",
        "signature-whitespace",
        "signature-unpadded",
        "signature-extra-padding",
        "signature-nonzero-pad-bits",
        "unsigned-attestation-id-tamper",
        "unsigned-independence-tamper",
    }:
        signature = str(attestation["signature"])
        if invalid_case == "unknown-key":
            attestation["key_id"] = "secret-unknown-key"
        elif invalid_case == "signer-key-mismatch":
            attestation["signer_id"] = "secret-impersonated-reviewer"
            _rewrite_gate_input(
                root,
                gate_id,
                lambda payload: payload.__setitem__(
                    "evaluator_id",
                    "secret-impersonated-reviewer",
                ),
            )
        elif invalid_case == "wrong-required-role":
            attestation["signer_role"] = _other_evaluator_role(gate_id)
            _rewrite_gate_input(
                root,
                gate_id,
                lambda payload: payload.__setitem__(
                    "evaluator_role",
                    _other_evaluator_role(gate_id),
                ),
            )
        elif invalid_case == "signed-after-evaluation":
            attestation["signed_at"] = "2026-07-25T03:00:01Z"
        elif invalid_case == "wrong-attestation-algorithm":
            attestation["algorithm"] = "RSA-PSS"
        elif invalid_case == "malformed-signature":
            attestation["signature"] = "secret-signature%%%"
        elif invalid_case == "wrong-signature":
            attestation["signature"] = base64.b64encode(b"\x00" * 64).decode(
                "ascii"
            )
        elif invalid_case == "short-signature":
            attestation["signature"] = base64.b64encode(b"\x00" * 63).decode(
                "ascii"
            )
        elif invalid_case == "long-signature":
            attestation["signature"] = base64.b64encode(b"\x00" * 65).decode(
                "ascii"
            )
        elif invalid_case == "signature-trailing-garbage":
            attestation["signature"] = signature + "!!"
        elif invalid_case == "signature-whitespace":
            attestation["signature"] = signature[:16] + "\n" + signature[16:]
        elif invalid_case == "signature-unpadded":
            attestation["signature"] = signature.rstrip("=")
        elif invalid_case == "signature-extra-padding":
            attestation["signature"] = signature + "="
        elif invalid_case == "signature-nonzero-pad-bits":
            alias = _noncanonical_base64_pad_bits(signature)
            if gate_id == "G7":
                assert alias == TEST_G7_SIGNATURE_BASE64[:-3] + "B=="
                assert alias.endswith("8XWVAB==")
            attestation["signature"] = alias
        elif invalid_case == "unsigned-attestation-id-tamper":
            attestation["attestation_id"] = f"ATT-{gate_id}-SECRET-TAMPER"
        else:
            attestation["independence_mode"] = "secret-unsigned-independence"
            _rewrite_gate_input(
                root,
                gate_id,
                lambda payload: payload.__setitem__(
                    "independence_mode",
                    "secret-unsigned-independence",
                ),
            )
        if invalid_case not in {
            "malformed-signature",
            "wrong-signature",
            "short-signature",
            "long-signature",
            "signature-trailing-garbage",
            "signature-whitespace",
            "signature-unpadded",
            "signature-extra-padding",
            "signature-nonzero-pad-bits",
            "unsigned-attestation-id-tamper",
            "unsigned-independence-tamper",
        }:
            _sign_attestation(tmp_path, attestation)
        _rewrite_attestation_payload(
            root,
            attestation_artifact,
            attestation,
        )
    else:
        keys = trust_payload["keys"]
        assert isinstance(keys, list)
        key = keys[0]
        assert isinstance(key, dict)
        public_key = str(key["public_key"])
        if invalid_case == "revoked-key":
            key["revoked"] = True
        elif invalid_case == "key-missing-role":
            key["roles"] = [_other_evaluator_role(gate_id)]
        elif invalid_case == "key-not-yet-valid":
            key["valid_from"] = "2026-07-25T02:00:01Z"
        elif invalid_case == "key-expired":
            key["valid_until"] = "2026-07-25T01:59:59Z"
        elif invalid_case == "key-set-issued-after-signing":
            trust_payload["issued_at"] = "2026-07-25T02:00:01Z"
        elif invalid_case == "wrong-key-algorithm":
            key["algorithm"] = "RSA-PSS"
        elif invalid_case == "malformed-public-key":
            key["public_key"] = "secret-public-key%%%"
        elif invalid_case == "malformed-spki-der":
            key["public_key"] = base64.b64encode(b"\x30\x01\x00").decode(
                "ascii"
            )
        elif invalid_case == "wrong-public-key-type":
            key["public_key"] = TEST_P256_SPKI_BASE64
        elif invalid_case == "public-key-trailing-garbage":
            key["public_key"] = public_key + "!!"
        elif invalid_case == "public-key-whitespace":
            key["public_key"] = public_key[:16] + "\n" + public_key[16:]
        elif invalid_case == "public-key-unpadded":
            key["public_key"] = public_key.rstrip("=")
        elif invalid_case == "public-key-extra-padding":
            key["public_key"] = public_key + "="
        elif invalid_case == "public-key-nonzero-pad-bits":
            alias = _noncanonical_base64_pad_bits(public_key)
            assert alias.endswith("wJB=")
            key["public_key"] = alias
        elif invalid_case == "public-key-trailing-der":
            key["public_key"] = base64.b64encode(
                base64.b64decode(public_key, validate=True) + b"\x00"
            ).decode("ascii")
        else:
            trust_payload["set_digest"] = SHA_A
        if invalid_case != "trust-self-digest-mismatch":
            _refresh_trust_set_digest(trust_payload)
        _rewrite_trust_payload(root, trust_artifact, trust_payload)

    expected_paths = {
        "input": f"manifests/gate-inputs/{gate_id}.json",
        "attestation": _expected_attestation_path(gate_id),
        "trust": "manifests/trusted-reviewer-keys.json",
        "wrong-attestation": wrong_attestation_path,
        "wrong-trust": wrong_trust_path,
    }
    with pytest.raises(GateEngineError) as captured:
        load_gate_input(root, gate_id)

    assert _error_code(captured.value) == expected_code
    assert captured.value.path == expected_paths[expected_location]
    diagnostic = repr(captured.value)
    assert str(root.resolve()) not in diagnostic
    assert "secret" not in diagnostic.lower()
    assert "secret-shadow" not in diagnostic
    assert "secret-wrapper" not in diagnostic
    assert "secret-impersonated" not in diagnostic
    assert "secret-signature" not in diagnostic
    assert "secret-public-key" not in diagnostic
    assert not (root / "manifests" / "gates" / f"{gate_id}.json").exists()


@pytest.mark.parametrize("gate_id", ["G7", "G8"])
@pytest.mark.parametrize(
    "invalid_case",
    ["same-path-wrong-digest", "ambiguous-registry-node"],
)
def test_attestation_subjects_resolve_one_exact_bound_path_and_digest(
    tmp_path: Path,
    gate_id: str,
    invalid_case: str,
) -> None:
    GateEngineError, _, load_gate_input, _ = _gate_api()
    root, context = _attested_gate_repo(tmp_path)
    attestation_artifact, attestation = _attestation_fixture(context, gate_id)
    source_artifact = _authority_artifact(
        context,
        "g7-audit" if gate_id == "G7" else "g8-subject",
    )

    if invalid_case == "ambiguous-registry-node":
        _add_ambiguous_subject_binding(
            root,
            gate_id=gate_id,
            source_artifact=source_artifact,
        )
    else:
        refs = attestation["subject_refs"]
        assert isinstance(refs, list)
        matching_refs = [
            ref
            for ref in refs
            if ref["path"] == str(source_artifact["path"])
        ]
        assert len(matching_refs) == 1
        matching_refs[0]["sha256"] = SHA_A
        if gate_id == "G7":
            attestation["subject_sha256"] = _g7_subject_digest(refs)
        else:
            attestation["subject_sha256"] = SHA_A
        _sign_attestation(tmp_path, attestation)
        _rewrite_attestation_payload(
            root,
            attestation_artifact,
            attestation,
        )

    with pytest.raises(GateEngineError) as captured:
        load_gate_input(root, gate_id)

    assert _error_code(captured.value) == "ATTESTATION_SUBJECT_INVALID"
    assert captured.value.path == _expected_attestation_path(gate_id)
    assert str(root.resolve()) not in repr(captured.value)
    assert not (root / "manifests" / "gates" / f"{gate_id}.json").exists()


@pytest.mark.parametrize(
    ("invalid_case", "expected_code", "expected_path"),
    [
        case
        for case in [
        (
            "attestation-ref-unbound",
            "EVALUATOR_ATTESTATION_UNBOUND",
            "manifests/gate-inputs/G7.json",
        ),
        (
            "attestation-ref-duplicate",
            "EVALUATOR_ATTESTATION_AMBIGUOUS",
            "manifests/gate-inputs/G7.json",
        ),
        (
            "trust-ref-unbound",
            "TRUSTED_KEY_SET_UNBOUND",
            "manifests/gate-inputs/G7.json",
        ),
        (
            "trust-ref-duplicate",
            "TRUSTED_KEY_SET_AMBIGUOUS",
            "manifests/gate-inputs/G7.json",
        ),
        (
            "attestation-wrong-path",
            "EVALUATOR_ATTESTATION_PATH_INVALID",
            "manifests/attestations/not-G7.json",
        ),
        (
            "trust-wrong-path",
            "TRUSTED_KEY_SET_PATH_INVALID",
            "manifests/not-trusted-keys.json",
        ),
        (
            "attestation-wrong-type",
            "EVALUATOR_ATTESTATION_TYPE_INVALID",
            "manifests/attestations/G7-auditor.json",
        ),
        (
            "malformed-attestation",
            "EVALUATOR_ATTESTATION_INVALID",
            "manifests/attestations/G7-auditor.json",
        ),
        (
            "noncanonical-attestation-json",
            "EVALUATOR_ATTESTATION_INVALID",
            "manifests/attestations/G7-auditor.json",
        ),
        (
            "duplicate-attestation-json",
            "EVALUATOR_ATTESTATION_INVALID",
            "manifests/attestations/G7-auditor.json",
        ),
        (
            "noncanonical-trust-json",
            "TRUSTED_KEY_SET_INVALID",
            "manifests/trusted-reviewer-keys.json",
        ),
        (
            "duplicate-trust-json",
            "TRUSTED_KEY_SET_INVALID",
            "manifests/trusted-reviewer-keys.json",
        ),
        (
            "wrapper-identity-mismatch",
            "ATTESTATION_IDENTITY_MISMATCH",
            "manifests/gate-inputs/G7.json",
        ),
        (
            "wrapper-independence-mismatch",
            "ATTESTATION_IDENTITY_MISMATCH",
            "manifests/gate-inputs/G7.json",
        ),
        (
            "subject-digest-mismatch",
            "ATTESTATION_SUBJECT_INVALID",
            "manifests/attestations/G7-auditor.json",
        ),
        (
            "subject-ref-unbound",
            "ATTESTATION_SUBJECT_INVALID",
            "manifests/attestations/G7-auditor.json",
        ),
        (
            "subject-self-reference",
            "ATTESTATION_SUBJECT_INVALID",
            "manifests/attestations/G7-auditor.json",
        ),
        (
            "revoked-key",
            "ATTESTATION_KEY_INVALID",
            "manifests/trusted-reviewer-keys.json",
        ),
        (
            "unknown-key",
            "ATTESTATION_KEY_INVALID",
            "manifests/attestations/G7-auditor.json",
        ),
        (
            "signer-key-mismatch",
            "ATTESTATION_KEY_INVALID",
            "manifests/trusted-reviewer-keys.json",
        ),
        (
            "wrong-required-role",
            "ATTESTATION_ROLE_INVALID",
            "manifests/attestations/G7-auditor.json",
        ),
        (
            "key-missing-role",
            "ATTESTATION_ROLE_INVALID",
            "manifests/trusted-reviewer-keys.json",
        ),
        (
            "key-not-yet-valid",
            "ATTESTATION_TIME_INVALID",
            "manifests/trusted-reviewer-keys.json",
        ),
        (
            "key-expired",
            "ATTESTATION_TIME_INVALID",
            "manifests/trusted-reviewer-keys.json",
        ),
        (
            "key-set-issued-after-signing",
            "ATTESTATION_TIME_INVALID",
            "manifests/trusted-reviewer-keys.json",
        ),
        (
            "signed-after-evaluation",
            "ATTESTATION_TIME_INVALID",
            "manifests/attestations/G7-auditor.json",
        ),
        (
            "wrong-algorithm",
            "ATTESTATION_KEY_INVALID",
            "manifests/attestations/G7-auditor.json",
        ),
        (
            "unsupported-key-algorithm",
            "ATTESTATION_KEY_INVALID",
            "manifests/trusted-reviewer-keys.json",
        ),
        (
            "malformed-public-key",
            "ATTESTATION_KEY_INVALID",
            "manifests/trusted-reviewer-keys.json",
        ),
        (
            "wrong-public-key-type",
            "ATTESTATION_KEY_INVALID",
            "manifests/trusted-reviewer-keys.json",
        ),
        (
            "malformed-signature",
            "ATTESTATION_SIGNATURE_INVALID",
            "manifests/attestations/G7-auditor.json",
        ),
        (
            "wrong-signature",
            "ATTESTATION_SIGNATURE_INVALID",
            "manifests/attestations/G7-auditor.json",
        ),
        (
            "short-signature",
            "ATTESTATION_SIGNATURE_INVALID",
            "manifests/attestations/G7-auditor.json",
        ),
        (
            "unsigned-payload-tamper",
            "ATTESTATION_SIGNATURE_INVALID",
            "manifests/attestations/G7-auditor.json",
        ),
        (
            "unsigned-roster-digest-tamper",
            "ATTESTATION_SIGNATURE_INVALID",
            "manifests/attestations/G7-auditor.json",
        ),
        (
            "trust-self-digest-mismatch",
            "TRUSTED_KEY_SET_INVALID",
            "manifests/trusted-reviewer-keys.json",
        ),
        ]
        if case[0]
        in {
            "subject-digest-mismatch",
            "subject-ref-unbound",
            "subject-self-reference",
            "unsigned-roster-digest-tamper",
        }
    ],
)
def test_g7_attestation_verification_fails_closed(
    tmp_path: Path,
    invalid_case: str,
    expected_code: str,
    expected_path: str,
) -> None:
    GateEngineError, _, load_gate_input, _ = _gate_api()
    root, context = _attested_gate_repo(tmp_path)
    artifacts = context["artifacts"]
    assert isinstance(artifacts, dict)
    g7_artifact = artifacts["g7-attestation"]
    trust_artifact = artifacts["trust"]
    assert isinstance(g7_artifact, dict)
    assert isinstance(trust_artifact, dict)

    if invalid_case in {
        "attestation-ref-unbound",
        "attestation-ref-duplicate",
        "trust-ref-unbound",
        "trust-ref-duplicate",
    }:
        removed_id = (
            str(g7_artifact["artifact_id"])
            if invalid_case.startswith("attestation")
            else str(trust_artifact["artifact_id"])
        )

        def remove_ref(payload: dict[str, object]) -> None:
            refs = payload["input_refs"]
            assert isinstance(refs, list)
            matches = [
                ref
                for ref in refs
                if isinstance(ref, dict)
                and ref.get("artifact_id") == removed_id
            ]
            assert len(matches) == 1
            if invalid_case.endswith("duplicate"):
                refs.append(deepcopy(matches[0]))
            else:
                payload["input_refs"] = [
                    ref
                    for ref in refs
                    if not (
                        isinstance(ref, dict)
                        and ref.get("artifact_id") == removed_id
                    )
                ]

        _rewrite_gate_input(root, "G7", remove_ref)
    elif invalid_case == "attestation-wrong-path":
        old_path = root / str(g7_artifact["path"])
        _rewrite_bound_artifact(
            root,
            artifact_id=str(g7_artifact["artifact_id"]),
            relative_path="manifests/attestations/not-G7.json",
            payload=old_path.read_bytes(),
        )
    elif invalid_case == "trust-wrong-path":
        old_path = root / str(trust_artifact["path"])
        _rewrite_bound_artifact(
            root,
            artifact_id=str(trust_artifact["artifact_id"]),
            relative_path="manifests/not-trusted-keys.json",
            payload=old_path.read_bytes(),
        )
    elif invalid_case == "attestation-wrong-type":
        _mutate_registry_node(
            root,
            str(g7_artifact["artifact_id"]),
            lambda node: node.__setitem__(
                "artifact_type",
                "gate-evaluation-attestation",
            ),
        )
    elif invalid_case == "malformed-attestation":
        _rewrite_bound_artifact(
            root,
            artifact_id=str(g7_artifact["artifact_id"]),
            relative_path=str(g7_artifact["path"]),
            payload=_canonical_json({"not": "an attestation"}),
        )
    elif invalid_case in {
        "noncanonical-attestation-json",
        "duplicate-attestation-json",
    }:
        attestation = deepcopy(context["g7_attestation"])
        assert isinstance(attestation, dict)
        if invalid_case == "noncanonical-attestation-json":
            encoded_attestation = (
                json.dumps(
                    attestation,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
        else:
            encoded_attestation = _canonical_json(attestation).replace(
                b'"signer_id":"reviewer-1"',
                b'"signer_id":"secret-shadow","signer_id":"reviewer-1"',
                1,
            )
        _rewrite_bound_artifact(
            root,
            artifact_id=str(g7_artifact["artifact_id"]),
            relative_path=str(g7_artifact["path"]),
            payload=encoded_attestation,
        )
    elif invalid_case in {
        "noncanonical-trust-json",
        "duplicate-trust-json",
    }:
        trust_payload = deepcopy(context["trust_payload"])
        assert isinstance(trust_payload, dict)
        if invalid_case == "noncanonical-trust-json":
            encoded_trust = (
                json.dumps(
                    trust_payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
        else:
            encoded_trust = _canonical_json(trust_payload).replace(
                b'"key_set_id":"stc-reviewer-keys"',
                (
                    b'"key_set_id":"secret-shadow",'
                    b'"key_set_id":"stc-reviewer-keys"'
                ),
                1,
            )
        _rewrite_bound_artifact(
            root,
            artifact_id=str(trust_artifact["artifact_id"]),
            relative_path=str(trust_artifact["path"]),
            payload=encoded_trust,
        )
    elif invalid_case == "wrapper-identity-mismatch":
        _rewrite_gate_input(
            root,
            "G7",
            lambda payload: payload.__setitem__(
                "evaluator_id",
                "different-reviewer",
            ),
        )
    elif invalid_case == "wrapper-independence-mismatch":
        _rewrite_gate_input(
            root,
            "G7",
            lambda payload: payload.__setitem__(
                "independence_mode",
                "not-the-signed-mode",
            ),
        )
    elif invalid_case in {
        "subject-digest-mismatch",
        "subject-ref-unbound",
        "subject-self-reference",
        "unknown-key",
        "signer-key-mismatch",
        "wrong-required-role",
        "signed-after-evaluation",
        "wrong-algorithm",
        "malformed-signature",
        "wrong-signature",
        "short-signature",
        "unsigned-payload-tamper",
        "unsigned-roster-digest-tamper",
    }:
        attestation = deepcopy(context["g7_attestation"])
        assert isinstance(attestation, dict)
        if invalid_case == "subject-digest-mismatch":
            attestation["subject_sha256"] = SHA_A
        elif invalid_case == "subject-ref-unbound":
            refs = attestation["subject_refs"]
            assert isinstance(refs, list)
            refs.append({"path": "reports/unbound.json", "sha256": SHA_A})
            refs.sort(key=lambda ref: str(ref["path"]))
            attestation["subject_sha256"] = _g7_subject_digest(refs)
        elif invalid_case == "subject-self-reference":
            refs = attestation["subject_refs"]
            assert isinstance(refs, list)
            refs.append(
                {
                    "path": "manifests/attestations/G7-auditor.json",
                    "sha256": str(g7_artifact["ref"]["sha256"]),
                }
            )
            refs.sort(key=lambda ref: str(ref["path"]))
            attestation["subject_sha256"] = _g7_subject_digest(refs)
        elif invalid_case == "unknown-key":
            attestation["key_id"] = "unknown-key"
        elif invalid_case == "signer-key-mismatch":
            attestation["signer_id"] = "impersonated-reviewer"
            _rewrite_gate_input(
                root,
                "G7",
                lambda payload: payload.__setitem__(
                    "evaluator_id",
                    "impersonated-reviewer",
                ),
            )
        elif invalid_case == "wrong-required-role":
            attestation["signer_role"] = "release_evaluator"
            _rewrite_gate_input(
                root,
                "G7",
                lambda payload: payload.__setitem__(
                    "evaluator_role",
                    "release_evaluator",
                ),
            )
        elif invalid_case == "signed-after-evaluation":
            attestation["signed_at"] = "2026-07-25T03:00:01Z"
        elif invalid_case == "wrong-algorithm":
            attestation["algorithm"] = "RSA-PSS"
        elif invalid_case == "malformed-signature":
            attestation["signature"] = "not-base64%%%"
        elif invalid_case == "wrong-signature":
            attestation["signature"] = base64.b64encode(b"\x00" * 64).decode(
                "ascii"
            )
        elif invalid_case == "short-signature":
            attestation["signature"] = base64.b64encode(b"\x00" * 63).decode(
                "ascii"
            )
        elif invalid_case == "unsigned-roster-digest-tamper":
            attestation["author_executor_roster_digest"] = SHA_C
        else:
            attestation["independence_mode"] = "tampered-after-signing"
            _rewrite_gate_input(
                root,
                "G7",
                lambda payload: payload.__setitem__(
                    "independence_mode",
                    "tampered-after-signing",
                ),
            )
        if invalid_case not in {
            "malformed-signature",
            "wrong-signature",
            "short-signature",
            "unsigned-payload-tamper",
            "unsigned-roster-digest-tamper",
        }:
            _sign_attestation(tmp_path, attestation)
        _rewrite_bound_artifact(
            root,
            artifact_id=str(g7_artifact["artifact_id"]),
            relative_path=str(g7_artifact["path"]),
            payload=_canonical_json(attestation),
        )
    else:
        trust_payload = deepcopy(context["trust_payload"])
        assert isinstance(trust_payload, dict)
        keys = trust_payload["keys"]
        assert isinstance(keys, list)
        key = keys[0]
        assert isinstance(key, dict)
        if invalid_case == "revoked-key":
            key["revoked"] = True
        elif invalid_case == "key-missing-role":
            key["roles"] = ["release_evaluator"]
        elif invalid_case == "key-not-yet-valid":
            key["valid_from"] = "2026-07-25T02:00:01Z"
        elif invalid_case == "key-expired":
            key["valid_until"] = "2026-07-25T01:59:59Z"
        elif invalid_case == "key-set-issued-after-signing":
            trust_payload["issued_at"] = "2026-07-25T02:00:01Z"
        elif invalid_case == "malformed-public-key":
            key["public_key"] = "not-base64%%%"
        elif invalid_case == "wrong-public-key-type":
            key["public_key"] = TEST_P256_SPKI_BASE64
        elif invalid_case == "unsupported-key-algorithm":
            key["algorithm"] = "RSA-PSS"
            attestation = deepcopy(context["g7_attestation"])
            assert isinstance(attestation, dict)
            attestation["algorithm"] = "RSA-PSS"
            _sign_attestation(tmp_path, attestation)
            _rewrite_bound_artifact(
                root,
                artifact_id=str(g7_artifact["artifact_id"]),
                relative_path=str(g7_artifact["path"]),
                payload=_canonical_json(attestation),
            )
        else:
            trust_payload["set_digest"] = SHA_A
        if invalid_case != "trust-self-digest-mismatch":
            _refresh_trust_set_digest(trust_payload)
        _rewrite_bound_artifact(
            root,
            artifact_id=str(trust_artifact["artifact_id"]),
            relative_path=str(trust_artifact["path"]),
            payload=_canonical_json(trust_payload),
        )

    with pytest.raises(GateEngineError) as captured:
        load_gate_input(root, "G7")

    assert _error_code(captured.value) == expected_code
    assert captured.value.path == expected_path
    assert str(root.resolve()) not in str(captured.value)
    assert not (root / "manifests" / "gates" / "G7.json").exists()


@pytest.mark.parametrize(
    ("invalid_case", "expected_code", "expected_path"),
    [
        (
            "human-approval-as-evaluator",
            "EVALUATOR_ATTESTATION_TYPE_INVALID",
            "manifests/attestations/G8-evaluator.json",
        ),
        (
            "audit-attestation-as-evaluator",
            "EVALUATOR_ATTESTATION_TYPE_INVALID",
            "manifests/attestations/G8-evaluator.json",
        ),
        (
            "wrong-gate-id",
            "ATTESTATION_IDENTITY_MISMATCH",
            "manifests/attestations/G8-evaluator.json",
        ),
        (
            "wrong-required-role",
            "ATTESTATION_ROLE_INVALID",
            "manifests/attestations/G8-evaluator.json",
        ),
        (
            "missing-distinguished-subject",
            "ATTESTATION_SUBJECT_INVALID",
            "manifests/attestations/G8-evaluator.json",
        ),
        (
            "subject-digest-mismatch",
            "ATTESTATION_SUBJECT_INVALID",
            "manifests/attestations/G8-evaluator.json",
        ),
        (
            "unbound-additional-subject",
            "ATTESTATION_SUBJECT_INVALID",
            "manifests/attestations/G8-evaluator.json",
        ),
        (
            "noncanonical-subject-json",
            "ATTESTATION_SUBJECT_INVALID",
            "manifests/gate-inputs/G8-evaluation-subject.json",
        ),
        (
            "duplicate-key-subject-json",
            "ATTESTATION_SUBJECT_INVALID",
            "manifests/gate-inputs/G8-evaluation-subject.json",
        ),
        (
            "invalid-utf8-subject-json",
            "ATTESTATION_SUBJECT_INVALID",
            "manifests/gate-inputs/G8-evaluation-subject.json",
        ),
        (
            "missing-lf-subject-json",
            "ATTESTATION_SUBJECT_INVALID",
            "manifests/gate-inputs/G8-evaluation-subject.json",
        ),
        (
            "distinguished-subject-wrong-path",
            "ATTESTATION_SUBJECT_INVALID",
            "manifests/attestations/G8-evaluator.json",
        ),
        (
            "distinguished-subject-wrong-type",
            "ATTESTATION_SUBJECT_INVALID",
            "manifests/gate-inputs/G8-evaluation-subject.json",
        ),
    ],
)
def test_g8_requires_distinct_typed_evaluator_and_canonical_subject(
    tmp_path: Path,
    invalid_case: str,
    expected_code: str,
    expected_path: str,
) -> None:
    GateEngineError, _, load_gate_input, _ = _gate_api()
    root, context = _attested_gate_repo(tmp_path)
    artifacts = context["artifacts"]
    assert isinstance(artifacts, dict)
    g8_attestation_artifact = artifacts["g8-attestation"]
    g8_subject_artifact = artifacts["g8-subject"]
    assert isinstance(g8_attestation_artifact, dict)
    assert isinstance(g8_subject_artifact, dict)

    if invalid_case == "human-approval-as-evaluator":
        replacement: dict[str, object] = {
            "approval_id": "APPROVAL-001",
            "subject_sha256": SHA_A,
            "pdf_digests": [SHA_B],
            "machine_visual_report_digest": SHA_C,
            "contact_sheet_digest": SHA_A,
            "all_pages_reviewed": True,
            "review_checks": {"layout": "PASS"},
            "disposition": "PASS",
            "open_issues": [],
            "signer_id": "reviewer-1",
            "signer_role": "release_evaluator",
            "algorithm": "Ed25519",
            "key_id": "reviewer-key-1",
            "signature": "not-an-evaluator-attestation",
            "signed_at": "2026-07-25T02:00:00Z",
        }
        _rewrite_bound_artifact(
            root,
            artifact_id=str(g8_attestation_artifact["artifact_id"]),
            relative_path=str(g8_attestation_artifact["path"]),
            payload=_canonical_json(replacement),
        )
    elif invalid_case == "audit-attestation-as-evaluator":
        replacement = deepcopy(context["g7_attestation"])
        assert isinstance(replacement, dict)
        _rewrite_bound_artifact(
            root,
            artifact_id=str(g8_attestation_artifact["artifact_id"]),
            relative_path=str(g8_attestation_artifact["path"]),
            payload=_canonical_json(replacement),
        )
    elif invalid_case in {
        "wrong-gate-id",
        "wrong-required-role",
        "missing-distinguished-subject",
        "subject-digest-mismatch",
        "unbound-additional-subject",
    }:
        attestation = deepcopy(context["g8_attestation"])
        assert isinstance(attestation, dict)
        if invalid_case == "wrong-gate-id":
            attestation["gate_id"] = "G7"
        elif invalid_case == "wrong-required-role":
            attestation["signer_role"] = "independent_auditor"
            _rewrite_gate_input(
                root,
                "G8",
                lambda payload: payload.__setitem__(
                    "evaluator_role",
                    "independent_auditor",
                ),
            )
        elif invalid_case == "missing-distinguished-subject":
            refs = attestation["subject_refs"]
            assert isinstance(refs, list)
            refs[:] = [
                ref
                for ref in refs
                if ref["path"]
                != "manifests/gate-inputs/G8-evaluation-subject.json"
            ]
            assert len(refs) == 1
            attestation["subject_sha256"] = refs[0]["sha256"]
        elif invalid_case == "subject-digest-mismatch":
            attestation["subject_sha256"] = SHA_A
        else:
            refs = attestation["subject_refs"]
            assert isinstance(refs, list)
            refs.append({"path": "reports/unbound-g8.json", "sha256": SHA_A})
            refs.sort(key=lambda ref: str(ref["path"]))
        _sign_attestation(tmp_path, attestation)
        _rewrite_bound_artifact(
            root,
            artifact_id=str(g8_attestation_artifact["artifact_id"]),
            relative_path=str(g8_attestation_artifact["path"]),
            payload=_canonical_json(attestation),
        )
    elif invalid_case in {
        "noncanonical-subject-json",
        "duplicate-key-subject-json",
        "invalid-utf8-subject-json",
        "missing-lf-subject-json",
    }:
        if invalid_case == "noncanonical-subject-json":
            subject_bytes = (
                b'{\n  "gate_id": "G8",\n'
                b'  "candidate": "stc-paper-v1.0.0-rc1",\n'
                b'  "review_scope": "complete release evaluation"\n}\n'
            )
        elif invalid_case == "duplicate-key-subject-json":
            subject_bytes = (
                b'{"candidate":"stc-paper-v1.0.0-rc1",'
                b'"gate_id":"G8","gate_id":"G8",'
                b'"review_scope":"complete release evaluation"}\n'
            )
        elif invalid_case == "invalid-utf8-subject-json":
            subject_bytes = (
                _canonical_json(
                    {
                        "gate_id": "G8",
                        "candidate": "stc-paper-v1.0.0-rc1",
                        "review_scope": "complete release evaluation",
                    }
                ).rstrip(b"\n")
                + b"\xff\n"
            )
        else:
            subject_bytes = _canonical_json(
                {
                    "gate_id": "G8",
                    "candidate": "stc-paper-v1.0.0-rc1",
                    "review_scope": "complete release evaluation",
                }
            ).rstrip(b"\n")
        _rewrite_bound_artifact(
            root,
            artifact_id=str(g8_subject_artifact["artifact_id"]),
            relative_path=str(g8_subject_artifact["path"]),
            payload=subject_bytes,
        )
        new_digest = _sha256(subject_bytes)
        attestation = deepcopy(context["g8_attestation"])
        assert isinstance(attestation, dict)
        refs = attestation["subject_refs"]
        assert isinstance(refs, list)
        distinguished = [
            ref
            for ref in refs
            if ref["path"]
            == "manifests/gate-inputs/G8-evaluation-subject.json"
        ]
        assert len(distinguished) == 1
        distinguished[0]["sha256"] = new_digest
        attestation["subject_sha256"] = new_digest
        _sign_attestation(tmp_path, attestation)
        _rewrite_bound_artifact(
            root,
            artifact_id=str(g8_attestation_artifact["artifact_id"]),
            relative_path=str(g8_attestation_artifact["path"]),
            payload=_canonical_json(attestation),
        )
    elif invalid_case == "distinguished-subject-wrong-path":
        old_path = root / str(g8_subject_artifact["path"])
        wrong_path = "manifests/gate-inputs/not-G8-evaluation-subject.json"
        _rewrite_bound_artifact(
            root,
            artifact_id=str(g8_subject_artifact["artifact_id"]),
            relative_path=wrong_path,
            payload=old_path.read_bytes(),
        )
        attestation = deepcopy(context["g8_attestation"])
        assert isinstance(attestation, dict)
        refs = attestation["subject_refs"]
        assert isinstance(refs, list)
        distinguished = [
            ref
            for ref in refs
            if ref["path"]
            == "manifests/gate-inputs/G8-evaluation-subject.json"
        ]
        assert len(distinguished) == 1
        distinguished[0]["path"] = wrong_path
        refs.sort(key=lambda ref: str(ref["path"]))
        _sign_attestation(tmp_path, attestation)
        _rewrite_attestation_payload(
            root,
            g8_attestation_artifact,
            attestation,
        )
    else:
        _mutate_registry_node(
            root,
            str(g8_subject_artifact["artifact_id"]),
            lambda node: node.__setitem__(
                "artifact_type",
                "release-smoke",
            ),
        )

    with pytest.raises(GateEngineError) as captured:
        load_gate_input(root, "G8")

    assert _error_code(captured.value) == expected_code
    assert captured.value.path == expected_path
    assert str(root.resolve()) not in str(captured.value)
    assert not (root / "manifests" / "gates" / "G8.json").exists()


@pytest.mark.parametrize(
    ("gate_id", "target_name"),
    [
        ("G7", "g7-attestation"),
        ("G8", "g8-attestation"),
        ("G7", "trust"),
        ("G8", "trust"),
        ("G8", "g8-subject"),
    ],
)
@pytest.mark.parametrize("phase", ["load", "evaluate", "verify-chain"])
def test_authority_artifact_symlinks_fail_closed_in_every_phase(
    tmp_path: Path,
    gate_id: str,
    target_name: str,
    phase: str,
) -> None:
    GateEngineError, evaluate_gate, load_gate_input, verify_gate_chain = _gate_api()
    root, context = _attested_gate_repo(tmp_path)
    artifact = _authority_artifact(context, target_name)
    target = root / str(artifact["path"])
    original = target.read_bytes()
    outside = tmp_path / f"outside-{gate_id}-{target.name}"
    outside.write_bytes(original)

    loaded = None
    if phase == "evaluate":
        _evaluate_through(root, int(gate_id[1:]) - 1)
        loaded = load_gate_input(root, gate_id)
    elif phase == "verify-chain":
        _evaluate_through(root, int(gate_id[1:]))

    target.unlink()
    target.symlink_to(outside)

    if phase == "load":
        with pytest.raises(GateEngineError) as captured:
            load_gate_input(root, gate_id)
        assert _error_code(captured.value) == "INPUT_ARTIFACT_UNSAFE"
        assert captured.value.path == str(artifact["path"])
    elif phase == "evaluate":
        assert loaded is not None
        with pytest.raises(GateEngineError) as captured:
            evaluate_gate(loaded)
        assert _error_code(captured.value) == "INPUT_ARTIFACT_DIGEST_MISMATCH"
        assert captured.value.path == str(artifact["path"])
    else:
        report = verify_gate_chain(root, through=gate_id)
        assert not report.success
        assert report.stale_gate_ids[0] == gate_id
        assert report.diagnostics[0].code == "GATE_DERIVATION_MISMATCH"
        assert report.diagnostics[0].path == f"manifests/gates/{gate_id}.json"
    assert outside.read_bytes() == original


@pytest.mark.parametrize(
    ("gate_id", "target_name", "expected_code"),
    [
        ("G7", "g7-attestation", "EVALUATOR_ATTESTATION_INVALID"),
        ("G8", "g8-attestation", "EVALUATOR_ATTESTATION_INVALID"),
        ("G7", "trust", "TRUSTED_KEY_SET_INVALID"),
        ("G8", "trust", "TRUSTED_KEY_SET_INVALID"),
        ("G8", "g8-subject", "ATTESTATION_SUBJECT_INVALID"),
    ],
)
@pytest.mark.parametrize("phase", ["load", "evaluate", "verify-chain"])
@pytest.mark.parametrize("race_target", ["leaf", "intermediate-directory"])
def test_authority_semantic_rereads_resist_same_byte_path_swap_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate_id: str,
    target_name: str,
    expected_code: str,
    phase: str,
    race_target: str,
) -> None:
    GateEngineError, evaluate_gate, load_gate_input, verify_gate_chain = _gate_api()
    gate_engine = importlib.import_module("stc_research.gate_engine")
    root, context = _attested_gate_repo(tmp_path)
    artifact = _authority_artifact(context, target_name)
    target = root / str(artifact["path"])
    target_bytes = target.read_bytes()
    outside_directory = tmp_path / f"outside-race-{gate_id}-{target_name}"
    outside_directory.mkdir()
    outside = outside_directory / target.name
    outside.write_bytes(target_bytes)

    loaded = None
    if phase == "evaluate":
        _evaluate_through(root, int(gate_id[1:]) - 1)
        loaded = load_gate_input(root, gate_id)
    elif phase == "verify-chain":
        _evaluate_through(root, int(gate_id[1:]))

    real_open = os.open
    real_read_bytes = Path.read_bytes
    target_parent_identity = (
        target.parent.stat().st_dev,
        target.parent.stat().st_ino,
    )
    outside_identity = (outside.stat().st_dev, outside.stat().st_ino)
    race_triggered = False
    external_opens: list[str] = []
    unsafe_path_reads: list[Path] = []

    def race_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal race_triggered
        path_text = os.fsdecode(os.fspath(path))
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) == outside_identity:
            external_opens.append(path_text)
        if (
            not race_triggered
            and dir_fd is not None
            and path_text == target.name
        ):
            parent = os.fstat(dir_fd)
            if (parent.st_dev, parent.st_ino) == target_parent_identity:
                target.unlink()
                target.symlink_to(outside)
                race_triggered = True
        return descriptor

    def observe_path_read(path: Path) -> bytes:
        if path == target and path.is_symlink():
            unsafe_path_reads.append(path)
        return real_read_bytes(path)

    monkeypatch.setattr(gate_engine.os, "open", race_open)
    monkeypatch.setattr(Path, "read_bytes", observe_path_read)

    if phase == "load":
        with pytest.raises(GateEngineError) as captured:
            load_gate_input(root, gate_id)
        assert _error_code(captured.value) == expected_code
        assert captured.value.path == str(artifact["path"])
    elif phase == "evaluate":
        assert loaded is not None
        with pytest.raises(GateEngineError) as captured:
            evaluate_gate(loaded)
        assert _error_code(captured.value) == expected_code
        assert captured.value.path == str(artifact["path"])
    else:
        report = verify_gate_chain(root, through=gate_id)
        assert not report.success
        assert report.stale_gate_ids[0] == gate_id
        assert report.diagnostics[0].code == "GATE_DERIVATION_MISMATCH"
        assert report.diagnostics[0].path == f"manifests/gates/{gate_id}.json"

    assert race_triggered
    assert external_opens == []
    assert unsafe_path_reads == []
    assert real_read_bytes(outside) == target_bytes


@pytest.mark.parametrize("gate_id", ["G7", "G8"])
def test_evaluate_rechecks_coherently_revoked_key_beyond_public_snapshot(
    tmp_path: Path,
    gate_id: str,
) -> None:
    GateEngineError, evaluate_gate, load_gate_input, _ = _gate_api()
    root, context = _attested_gate_repo(
        tmp_path,
        split_gate_keys=gate_id == "G8",
    )
    originally_valid = load_gate_input(root, gate_id)
    trust_artifact = _authority_artifact(context, "trust")
    trust_payload = deepcopy(context["trust_payload"])
    assert isinstance(trust_payload, dict)
    keys = trust_payload["keys"]
    assert isinstance(keys, list)
    revoked_key_id = "reviewer-key-1" if gate_id == "G7" else "reviewer-key-2"
    matching_keys = [
        key
        for key in keys
        if isinstance(key, dict) and key.get("key_id") == revoked_key_id
    ]
    assert len(matching_keys) == 1
    matching_keys[0]["revoked"] = True
    _refresh_trust_set_digest(trust_payload)
    _rewrite_trust_payload(root, trust_artifact, trust_payload)
    _evaluate_through(root, int(gate_id[1:]) - 1)

    hostile_but_canonical = _coherent_public_gate_input(
        root,
        gate_id,
        originally_valid,
    )

    with pytest.raises(GateEngineError) as captured:
        evaluate_gate(hostile_but_canonical)

    assert _error_code(captured.value) == "ATTESTATION_KEY_INVALID"
    assert captured.value.path == "manifests/trusted-reviewer-keys.json"
    assert not (root / "manifests" / "gates" / f"{gate_id}.json").exists()


@pytest.mark.parametrize("gate_id", ["G7", "G8"])
@pytest.mark.parametrize("invalid_authority", ["signature", "revoked-key"])
def test_verify_chain_rederives_coherent_invalid_authorization_at_target_gate(
    tmp_path: Path,
    gate_id: str,
    invalid_authority: str,
) -> None:
    _, _, _, verify_gate_chain = _gate_api()
    root, context = _attested_gate_repo(
        tmp_path,
        split_gate_keys=gate_id == "G8" and invalid_authority == "revoked-key",
    )

    if invalid_authority == "signature":
        attestation_artifact, attestation = _attestation_fixture(context, gate_id)
        attestation["signature"] = base64.b64encode(b"\x00" * 64).decode("ascii")
        _rewrite_attestation_payload(
            root,
            attestation_artifact,
            attestation,
        )
    else:
        trust_artifact = _authority_artifact(context, "trust")
        trust_payload = deepcopy(context["trust_payload"])
        assert isinstance(trust_payload, dict)
        keys = trust_payload["keys"]
        assert isinstance(keys, list)
        revoked_key_id = (
            "reviewer-key-1" if gate_id == "G7" else "reviewer-key-2"
        )
        matching_keys = [
            key
            for key in keys
            if isinstance(key, dict) and key.get("key_id") == revoked_key_id
        ]
        assert len(matching_keys) == 1
        matching_keys[0]["revoked"] = True
        _refresh_trust_set_digest(trust_payload)
        _rewrite_trust_payload(root, trust_artifact, trust_payload)

    _evaluate_through(root, int(gate_id[1:]) - 1)
    _write_exact_forged_gate_record(root, gate_id)

    report = verify_gate_chain(root, through=gate_id)

    assert not report.success
    assert report.stale_gate_ids == (gate_id,)
    assert _chain_contract(report) == (
        (
            "GATE_DERIVATION_MISMATCH",
            gate_id,
            f"manifests/gates/{gate_id}.json",
        ),
    )


@pytest.mark.parametrize(
    "inclusive_boundary",
    [
        "valid-from-and-issued-at",
        "valid-until",
        "evaluated-at",
        "unbounded-valid-until",
    ],
)
@pytest.mark.parametrize("gate_id", ["G7", "G8"])
def test_attestation_time_windows_are_inclusive_at_exact_boundaries(
    tmp_path: Path,
    inclusive_boundary: str,
    gate_id: str,
) -> None:
    _, _, load_gate_input, verify_gate_chain = _gate_api()
    root, context = _attested_gate_repo(tmp_path)
    trust_artifact = _authority_artifact(context, "trust")
    attestation_artifact, attestation = _attestation_fixture(context, gate_id)
    trust_payload = deepcopy(context["trust_payload"])
    assert isinstance(trust_payload, dict)
    keys = trust_payload["keys"]
    assert isinstance(keys, list)
    key = keys[0]
    assert isinstance(key, dict)

    if inclusive_boundary == "valid-from-and-issued-at":
        key["valid_from"] = attestation["signed_at"]
        trust_payload["issued_at"] = attestation["signed_at"]
    elif inclusive_boundary == "valid-until":
        key["valid_until"] = attestation["signed_at"]
    elif inclusive_boundary == "evaluated-at":
        attestation["signed_at"] = "2026-07-25T03:00:00Z"
        _sign_attestation(tmp_path, attestation)
    else:
        key["valid_until"] = None

    _refresh_trust_set_digest(trust_payload)
    _rewrite_trust_payload(root, trust_artifact, trust_payload)
    if inclusive_boundary == "evaluated-at":
        _rewrite_attestation_payload(
            root,
            attestation_artifact,
            attestation,
        )

    gate_input = load_gate_input(root, gate_id)

    assert gate_input.gate_id == gate_id
    assert gate_input.evaluator_role == _required_evaluator_role(gate_id)
    _evaluate_through(root, int(gate_id[1:]))
    report = verify_gate_chain(root, through=gate_id)
    assert report.success
    assert report.diagnostics == ()
