from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _modules():
    import importlib.util
    import sys

    qa_spec = importlib.util.spec_from_file_location("easy_qa", ROOT / "easy/sleep-time-compute/qa.py")
    qa = importlib.util.module_from_spec(qa_spec)
    assert qa_spec.loader
    sys.modules[qa_spec.name] = qa
    qa_spec.loader.exec_module(qa)
    build_spec = importlib.util.spec_from_file_location("easy_build", ROOT / "easy/sleep-time-compute/build.py")
    build = importlib.util.module_from_spec(build_spec)
    assert build_spec.loader
    sys.modules[build_spec.name] = build
    build_spec.loader.exec_module(build)
    return qa, build


def test_manifest_has_exactly_twelve_valid_booklets():
    qa, _ = _modules()
    report = qa.validate_manifest(ROOT)
    assert report["booklets"] == 12
    assert report["success"], report["errors"]


def test_every_background_link_exists():
    _, build = _modules()
    manifest, _, _, concepts = build.load_catalogs()
    for record in manifest["booklets"]:
        assert set(record["background_concepts"]) <= set(concepts)


def test_preprocessor_resolves_all_typed_placeholders(tmp_path: Path):
    _, build = _modules()
    record = build.record_for("E01")
    text = build.prepare_markdown(record, tmp_path / "sample.pdf")
    assert "{{BG:" not in text
    assert "{{FIG:" not in text
    assert "{{CLAIM:" not in text
    assert "TRAINING-BACKGROUND" in text
    assert "stc-f001.pdf" in text


@pytest.mark.parametrize("booklet_id", [f"E{i:02d}" for i in range(1, 13)])
def test_each_booklet_declares_a_figure_claim_and_study_section(booklet_id: str):
    _, build = _modules()
    record = build.record_for(booklet_id)
    assert record["figures"]
    assert record["claims"]
    assert record["study_sections"]
