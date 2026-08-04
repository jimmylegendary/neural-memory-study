from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

PROGRAM = Path(__file__).resolve().parents[1] / "program"
sys.path.insert(0, str(PROGRAM))

from build_release_manifest import build_manifest, write_manifest  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_release(root: Path, translation_scope: str = "internal-only") -> None:
    publication = root / "build/publications"
    translations = publication / "translations-kr"
    translations.mkdir(parents=True)
    core = {
        "study": "SLEEP-TIME-COMPUTE-STRATEGIC-STUDY-KR.pdf",
        "background": "TRAINING-BACKGROUND-FOR-SLEEP-TIME-COMPUTE-KR.pdf",
        "conference": "SLEEP-TIME-COMPUTE-CONFERENCE-KR.pdf",
        "appendix": "SLEEP-TIME-COMPUTE-CONFERENCE-APPENDIX-KR.pdf",
    }
    academic = []
    for index, (target, name) in enumerate(core.items(), start=1):
        path = publication / name
        path.write_bytes(f"pdf-{target}".encode())
        academic.append({"output": str(path.relative_to(root)), "pages": index, "sha256": _sha(path), "target": target})
    (publication / "build-manifest.json").write_text(json.dumps({"artifacts": academic, "success": True}))

    for name in ["SLEEP-TIME-COMPUTE-EASY-COMPANION-KR.pdf", "SLEEP-TIME-COMPUTE-DEEP-SEMINAR-112SLIDES-KR.pdf", "SLEEP-TIME-COMPUTE-DEEP-SEMINAR-112SLIDES-KR.pptx"]:
        (publication / name).write_bytes(name.encode())

    translated = translations / "STC-T99-FIXTURE-KR.pdf"
    translated.write_bytes(b"translation")
    (translations / "build-manifest.json").write_text(json.dumps({
        "success": True,
        "artifacts": [{
            "output": str(translated.relative_to(root)),
            "paper_id": "STC-T99",
            "pages": 3,
            "rights": translation_scope,
            "sha256": _sha(translated),
        }],
    }))

    easy_report = root / "easy/sleep-time-compute/reports/qa-report.json"
    easy_report.parent.mkdir(parents=True)
    easy_report.write_text(json.dumps({"status": "pass", "combined_pages": 62}))
    deck_report = root / "presentation/sleep-time-compute-deep/reports/release-qa.json"
    deck_report.parent.mkdir(parents=True)
    deck_report.write_text(json.dumps({"status": "pass", "slideXmlCount": 112, "notesXmlCount": 112}))
    (publication / "FINAL-QA.json").write_text(json.dumps({"status": "pass", "failed": 0}))


def test_release_fails_when_required_artifact_is_missing(tmp_path: Path) -> None:
    report = build_manifest(tmp_path, source_commit="fixture")
    assert report["release_status"] == "blocked"
    assert "missing-study-pdf" in report["blockers"]


def test_public_manifest_excludes_internal_translation(tmp_path: Path) -> None:
    _seed_release(tmp_path, translation_scope="internal-only")
    report = build_manifest(tmp_path, source_commit="fixture")
    assert report["release_status"] == "ready"
    assert report["public_artifacts"]["translations"] == []
    assert [item["id"] for item in report["internal_artifacts"]["translations"]] == ["STC-T99"]


def test_public_translation_is_routed_by_rights_state(tmp_path: Path) -> None:
    _seed_release(tmp_path, translation_scope="public")
    report = build_manifest(tmp_path, source_commit="fixture")
    assert [item["id"] for item in report["public_artifacts"]["translations"]] == ["STC-T99"]
    assert report["internal_artifacts"]["translations"] == []


def test_manifest_write_is_deterministic(tmp_path: Path) -> None:
    _seed_release(tmp_path)
    first = write_manifest(tmp_path, source_commit="fixture")
    first_bytes = first.read_bytes()
    second = write_manifest(tmp_path, source_commit="fixture")
    assert second.read_bytes() == first_bytes


def test_require_ready_raises_for_blocked_release(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="release blocked"):
        write_manifest(tmp_path, source_commit="fixture", require_ready=True)
