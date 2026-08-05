from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "presentation/sleep-time-compute-deep/tests/verify_deck_release.py"
PPTX = REPO / "build/publications/SLEEP-TIME-COMPUTE-DEEP-SEMINAR-112SLIDES-KR.pptx"


def test_artifact_only_mode_passes_without_ephemeral_preview_or_layout_files(tmp_path: Path) -> None:
    target = tmp_path / "build/publications" / PPTX.name
    target.parent.mkdir(parents=True)
    shutil.copy2(PPTX, target)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--artifact-only", "--no-write-report"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "pass"
    assert report["mode"] == "artifact-only"
    assert report["slideXmlCount"] == 112
    assert report["notesXmlCount"] == 112
    assert report["sourceMarkerCount"] == 112
    assert report["ephemeralChecksSkipped"] is True
    assert not (tmp_path / "presentation/sleep-time-compute-deep/reports/release-qa.json").exists()
