import importlib.util
import json
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[1]
VALIDATOR_PATH = PACKAGE_ROOT / "program" / "validate_program.py"
SHA_A = "a" * 64


def _load_validator():
    assert VALIDATOR_PATH.exists(), "program validator has not been implemented"
    spec = importlib.util.spec_from_file_location(
        "stc_program_validation", VALIDATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_program_fixture(
    root: Path,
    *,
    source_status: str = "frozen",
    locator: str = "p. 4, lines 112-118",
    translation_count: int = 12,
) -> Path:
    source = {
        "source_id": "SRC-STC-0001",
        "title": "Sleep-Time Compute as a New Scaling Axis",
        "source_type": "paper",
        "primary": True,
        "status": source_status,
        "canonical_url": "https://arxiv.org/abs/2504.13171v1",
        "immutable": True,
        "version": {
            "version_id": "arXiv:2504.13171v1",
            "content_sha256": SHA_A,
            "retrieved_at": "2026-08-05",
        },
        "published_at": "2025-04-17",
        "accessed_at": "2026-08-05",
        "rights": {
            "license_id": "arXiv-nonexclusive-distribute",
            "evidence_url": "https://info.arxiv.org/help/license/index.html",
            "redistribution_allowed": False,
            "review_status": "reviewed",
        },
    }
    claim = {
        "claim_id": "STC-C001",
        "text": "Offline consolidation can improve later task performance.",
        "claim_type": "direct_fact",
        "status": "supported",
        "public": True,
        "load_bearing": True,
        "support": [
            {
                "source_id": source["source_id"],
                "locator": locator,
                "relation": "supports",
            }
        ],
    }
    figure = {
        "figure_id": "STC-F001",
        "title": "Wake-to-sleep consolidation loop",
        "mode": "redrawn",
        "public": True,
        "source_ids": [source["source_id"]],
        "license": {
            "license_id": "original-redraw",
            "evidence_url": "https://example.org/figure-ledger/STC-F001",
            "evidence_locator": "figure-ledger entry STC-F001",
            "reuse_allowed": True,
            "review_status": "reviewed",
        },
    }
    selected = [
        {
            "paper_id": f"STC-T{i:02d}",
            "source_id": source["source_id"],
            "selection_reason": "Covers a load-bearing research cluster.",
            "status": "selected",
        }
        for i in range(1, translation_count + 1)
    ]

    _write_json(
        root / "SOURCE-REGISTRY.json", {"schema_version": "1.0.0", "sources": [source]}
    )
    _write_json(root / "claim-map.json", {"schema_version": "1.0.0", "claims": [claim]})
    _write_json(
        root / "figure-ledger.json", {"schema_version": "1.0.0", "figures": [figure]}
    )
    _write_json(
        root / "translation-selection.json",
        {"schema_version": "1.0.0", "selected": selected},
    )
    _write_json(
        root / "research-spine.json",
        {
            "schema_version": "1.0.0",
            "status": "frozen",
            "source_freeze": "2026-08-05",
            "registries": {
                "sources": "SOURCE-REGISTRY.json",
                "claims": "claim-map.json",
                "figures": "figure-ledger.json",
                "translations": "translation-selection.json",
            },
            "unresolved_load_bearing_claims": [],
        },
    )
    return root


def _codes(report: object) -> set[str]:
    return {diagnostic.code for diagnostic in report.diagnostics}


def test_research_spine_requires_frozen_sources_and_locators(tmp_path):
    module = _load_validator()
    root = write_program_fixture(tmp_path, source_status="candidate", locator="")

    report = module.validate_program(root)

    assert not report.success
    assert _codes(report) >= {"source-not-frozen", "claim-missing-locator"}


def test_translation_selection_is_between_twelve_and_fifteen(tmp_path):
    module = _load_validator()
    root = write_program_fixture(tmp_path, translation_count=11)

    report = module.validate_program(root)

    assert "translation-count-out-of-range" in _codes(report)


def test_duplicate_registry_ids_are_rejected(tmp_path):
    module = _load_validator()
    root = write_program_fixture(tmp_path)
    path = root / "SOURCE-REGISTRY.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["sources"].append(payload["sources"][0])
    _write_json(path, payload)

    report = module.validate_program(root)

    assert "duplicate-source-id" in _codes(report)


def test_non_iso_dates_are_rejected(tmp_path):
    module = _load_validator()
    root = write_program_fixture(tmp_path)
    path = root / "SOURCE-REGISTRY.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["sources"][0]["accessed_at"] = "2026/08/05"
    _write_json(path, payload)

    report = module.validate_program(root)

    assert "schema-validation-error" in _codes(report)


def test_mutable_source_requires_version_metadata(tmp_path):
    module = _load_validator()
    root = write_program_fixture(tmp_path)
    path = root / "SOURCE-REGISTRY.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload["sources"][0]
    source["canonical_url"] = "https://docs.example.org/latest/memory"
    source["immutable"] = False
    source.pop("version")
    _write_json(path, payload)

    report = module.validate_program(root)

    assert "source-mutable-without-version" in _codes(report)


def test_unsupported_claim_type_is_rejected(tmp_path):
    module = _load_validator()
    root = write_program_fixture(tmp_path)
    path = root / "claim-map.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["claims"][0]["claim_type"] = "certain_prediction"
    _write_json(path, payload)

    report = module.validate_program(root)

    assert "unsupported-claim-type" in _codes(report)


def test_public_direct_reuse_figure_requires_license_evidence(tmp_path):
    module = _load_validator()
    root = write_program_fixture(tmp_path)
    path = root / "figure-ledger.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    figure = payload["figures"][0]
    figure["mode"] = "direct_reuse"
    figure["license"] = {
        "license_id": "unknown",
        "evidence_url": "",
        "evidence_locator": "",
        "reuse_allowed": False,
        "review_status": "pending",
    }
    _write_json(path, payload)

    report = module.validate_program(root)

    assert "figure-license-evidence-missing" in _codes(report)


def test_frozen_spine_rejects_unresolved_load_bearing_claims(tmp_path):
    module = _load_validator()
    root = write_program_fixture(tmp_path)
    path = root / "research-spine.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unresolved_load_bearing_claims"] = ["STC-C099"]
    _write_json(path, payload)

    report = module.validate_program(root)

    assert "frozen-spine-has-unresolved-load-bearing-claims" in _codes(report)


def test_unknown_fields_are_rejected_by_strict_schemas(tmp_path):
    module = _load_validator()
    root = write_program_fixture(tmp_path)
    path = root / "claim-map.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["claims"][0]["untracked_annotation"] = "must not silently pass"
    _write_json(path, payload)

    report = module.validate_program(root)

    assert "schema-validation-error" in _codes(report)


def test_valid_frozen_program_succeeds(tmp_path):
    module = _load_validator()
    root = write_program_fixture(tmp_path)

    report = module.validate_program(root)

    assert report.success
    assert report.diagnostics == ()


def test_pre_research_phase_requires_all_alignment_artifacts(tmp_path):
    module = _load_validator()

    report = module.validate_phase(tmp_path, "pre-research")

    assert "pre-research-file-missing" in _codes(report)


def test_pre_research_seed_records_require_fixed_metadata(tmp_path):
    module = _load_validator()
    phase_root = tmp_path / "research" / "sleep-time-compute" / "pre-research"
    phase_root.mkdir(parents=True)
    for filename in module.PRE_RESEARCH_REQUIRED_FILES:
        path = phase_root / filename
        if filename == "SEED-CORPUS.json":
            _write_json(
                path,
                {
                    "schema_version": "1.0.0",
                    "frozen_at": "2026-08-05",
                    "purpose": "fixture",
                    "sources": [{"seed_id": "SEED-STC-001", "title": "Incomplete"}],
                },
            )
        else:
            path.write_text("fixture\n", encoding="utf-8")

    report = module.validate_phase(tmp_path, "pre-research")

    assert "seed-source-metadata-missing" in _codes(report)


def test_marker_and_latex_error_flags_are_independent(tmp_path):
    module = _load_validator()
    phase_root = tmp_path / "research" / "sleep-time-compute" / "pre-research"
    phase_root.mkdir(parents=True)
    (phase_root / "note.md").write_text("LaTeX Warning: undefined references\n")

    marker_only = module.validate_phase(
        tmp_path,
        "pre-research",
        reject_authoring_markers=True,
        reject_latex_reference_errors=False,
        enforce_contract=False,
    )
    latex_only = module.validate_phase(
        tmp_path,
        "pre-research",
        reject_authoring_markers=False,
        reject_latex_reference_errors=True,
        enforce_contract=False,
    )

    assert "latex-reference-error-found" not in _codes(marker_only)
    assert "latex-reference-error-found" in _codes(latex_only)


def test_repository_pre_research_package_satisfies_contract():
    module = _load_validator()
    repository_root = PACKAGE_ROOT.parents[1]

    report = module.validate_phase(
        repository_root,
        "pre-research",
        reject_authoring_markers=True,
    )

    assert report.success, report.diagnostics


def test_deep_research_phase_requires_all_evidence_artifacts(tmp_path):
    module = _load_validator()

    report = module.validate_phase(tmp_path, "deep-research")

    assert "deep-research-file-missing" in _codes(report)


def test_deep_research_rejects_empty_registries_and_missing_clusters(tmp_path):
    module = _load_validator()
    phase_root = tmp_path / "research" / "sleep-time-compute" / "deep-research"
    phase_root.mkdir(parents=True)
    for filename in module.DEEP_RESEARCH_REQUIRED_FILES:
        path = phase_root / filename
        if filename == "SOURCE-REGISTRY.json":
            _write_json(path, {"schema_version": "1.0.0", "sources": []})
        elif filename == "CITATION-POOL.json":
            _write_json(path, {"schema_version": "1.0.0", "citations": []})
        elif filename == "SATURATION.json":
            _write_json(path, {"schema_version": "1.0.0", "clusters": []})
        elif filename == "SOURCE-RIGHTS.json":
            _write_json(path, {"schema_version": "1.0.0", "rights": []})
        elif filename == "S2-STATUS.json":
            _write_json(path, {"schema_version": "1.0.0"})
        else:
            path.write_text("fixture\n", encoding="utf-8")

    report = module.validate_phase(tmp_path, "deep-research")

    assert _codes(report) >= {
        "deep-research-source-registry-too-small",
        "deep-research-citation-pool-too-small",
        "deep-research-saturation-cluster-missing",
        "deep-research-rights-registry-too-small",
        "deep-research-s2-status-incomplete",
        "deep-research-content-missing",
    }


def test_repository_deep_research_package_satisfies_contract():
    module = _load_validator()
    repository_root = PACKAGE_ROOT.parents[1]

    report = module.validate_phase(
        repository_root,
        "deep-research",
        reject_authoring_markers=True,
    )

    assert report.success, report.diagnostics


def test_deep_research_source_citation_and_rights_ids_are_one_to_one():
    phase_root = PACKAGE_ROOT / "deep-research"
    sources = json.loads(
        (phase_root / "SOURCE-REGISTRY.json").read_text(encoding="utf-8")
    )["sources"]
    citations = json.loads(
        (phase_root / "CITATION-POOL.json").read_text(encoding="utf-8")
    )["citations"]
    rights = json.loads(
        (phase_root / "SOURCE-RIGHTS.json").read_text(encoding="utf-8")
    )["rights"]

    source_ids = {item["source_id"] for item in sources}

    assert len(sources) == len(source_ids) >= 60
    assert {item["source_id"] for item in citations} == source_ids
    assert {item["source_id"] for item in rights} == source_ids
    assert all(item["status"] == "frozen" for item in sources)


def test_synthesis_phase_requires_all_strategic_artifacts(tmp_path):
    module = _load_validator()

    report = module.validate_phase(tmp_path, "synthesis")

    assert "synthesis-file-missing" in _codes(report)


def test_synthesis_contract_rejects_shallow_placeholder_files(tmp_path):
    module = _load_validator()
    phase_root = tmp_path / "research" / "sleep-time-compute" / "deep-research"
    phase_root.mkdir(parents=True)
    for filename in module.SYNTHESIS_REQUIRED_FILES:
        (phase_root / filename).write_text("placeholder\n", encoding="utf-8")

    report = module.validate_phase(tmp_path, "synthesis")

    assert "synthesis-content-missing" in _codes(report)


def test_repository_synthesis_package_satisfies_contract():
    module = _load_validator()
    repository_root = PACKAGE_ROOT.parents[1]

    report = module.validate_phase(
        repository_root,
        "synthesis",
        reject_authoring_markers=True,
    )

    assert report.success, report.diagnostics
