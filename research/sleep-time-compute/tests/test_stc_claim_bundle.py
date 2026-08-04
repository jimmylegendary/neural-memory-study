import hashlib
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]
BUILDER_PATH = REPO_ROOT / "research/sleep-time-compute/program/build_claim_bundle.py"


def _load_builder():
    assert BUILDER_PATH.exists(), "claim-bundle builder has not been implemented"
    spec = importlib.util.spec_from_file_location("stc_claim_bundle", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def test_every_load_bearing_claim_has_primary_source_and_locator():
    module = _load_builder()
    artifacts = module.build_artifacts(REPO_ROOT)
    bundle = artifacts["bundle"]
    claim_map = artifacts["claim_map"]
    sources = {
        source["source_id"]: source
        for source in json.loads(
            (
                REPO_ROOT
                / "research/sleep-time-compute/deep-research/SOURCE-REGISTRY.json"
            ).read_text(encoding="utf-8")
        )["sources"]
    }

    assert len(bundle["claims"]) >= 35
    for claim in claim_map["claims"]:
        if not claim["load_bearing"]:
            continue
        assert claim["support"], claim["claim_id"]
        for support in claim["support"]:
            assert support["locator"].strip(), claim["claim_id"]
            assert sources[support["source_id"]]["primary"] is True
            assert sources[support["source_id"]]["status"] == "frozen"


def test_veridraft_claim_contracts_are_type_safe():
    module = _load_builder()
    artifacts = module.build_artifacts(REPO_ROOT)
    bundle = artifacts["bundle"]
    source_ids = {
        source["source_id"]
        for source in json.loads(
            (
                REPO_ROOT
                / "research/sleep-time-compute/deep-research/SOURCE-REGISTRY.json"
            ).read_text(encoding="utf-8")
        )["sources"]
    }

    assert bundle["results"], "systems-paper profile requires a non-empty result set"
    result_ids = {result["result_id"] for result in bundle["results"]}
    for claim in bundle["claims"]:
        assert claim["evidence"], claim["claim_id"]
        assert all(
            evidence["source_id"] in source_ids for evidence in claim["evidence"]
        )
        assert all(evidence["locator"].strip() for evidence in claim["evidence"])
        if claim["type"] == "P1":
            assert claim["result_refs"], claim["claim_id"]
            assert set(claim["result_refs"]) <= result_ids
        if claim["type"] == "P2":
            assert all(
                evidence["kind"] == "source_artifact" for evidence in claim["evidence"]
            )


def test_device_projections_are_held_p3_claims():
    module = _load_builder()
    artifacts = module.build_artifacts(REPO_ROOT)
    claim_map = artifacts["claim_map"]
    bundle_by_id = {claim["claim_id"]: claim for claim in artifacts["bundle"]["claims"]}

    device = [claim for claim in claim_map["claims"] if "device" in claim["tags"]]
    assert device
    for mapped in device:
        bundled = bundle_by_id[mapped["claim_id"]]
        assert mapped["claim_type"] == "hypothesis"
        assert mapped["public"] is False
        assert bundled["type"] == "P3"
        assert bundled["boundary"] == "internal"
        assert bundled["visibility"] == "private"
        assert len(bundled["evidence"]) >= 2


def test_translation_selection_has_fixed_fidelity_metadata():
    module = _load_builder()
    selection = module.build_artifacts(REPO_ROOT)["translation_selection"]

    assert 12 <= len(selection["selected"]) <= 15
    assert len({item["paper_id"] for item in selection["selected"]}) == len(
        selection["selected"]
    )
    required = {
        "paper_id",
        "source_id",
        "selection_reason",
        "status",
        "exact_version",
        "source_url",
        "source_sha256",
        "license_state",
        "figure_count",
        "equation_count",
        "table_count",
        "existing_translation",
        "role",
    }
    for item in selection["selected"]:
        assert required <= set(item), item.get("paper_id")
        assert len(item["source_sha256"]) == 64
        int(item["source_sha256"], 16)
        assert item["status"] == "selected"


def test_generation_is_byte_deterministic(tmp_path):
    module = _load_builder()
    first = module.build_artifacts(REPO_ROOT)
    second = module.build_artifacts(REPO_ROOT)

    assert set(first) == set(second)
    for name in first:
        first_bytes = _canonical_bytes(first[name])
        second_bytes = _canonical_bytes(second[name])
        assert first_bytes == second_bytes
        assert (
            hashlib.sha256(first_bytes).hexdigest()
            == hashlib.sha256(second_bytes).hexdigest()
        )
