from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGER = REPO_ROOT / "scripts" / "package_deliverables.py"

EXPECTED_COUNTS = {
    "neural-memory": {
        "analysis": 3,
        "easy": 12,
        "seminar": 5,
        "study": 1,
        "translations": 10,
    },
    "sleep-time-compute": {
        "easy": 13,
        "manifests": 6,
        "seminar": 2,
        "study": 10,
        "translations": 15,
    },
}


def _run_packager(output: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(PACKAGER),
            "--root",
            str(REPO_ROOT),
            "--output",
            str(output),
            *extra_args,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_cli_creates_two_separate_project_packages(tmp_path: Path) -> None:
    output = tmp_path / "deliverables"
    result = _run_packager(output)

    assert result.returncode == 0, result.stderr
    assert {path.name for path in output.iterdir()} == {
        "README.md",
        "neural-memory",
        "sleep-time-compute",
    }
    for project, counts in EXPECTED_COUNTS.items():
        project_dir = output / project
        actual_categories = {
            path.name for path in project_dir.iterdir() if path.is_dir()
        }
        assert actual_categories == set(counts)
        assert (project_dir / "README.md").is_file()
        assert (project_dir / "manifest.json").is_file()
        assert (project_dir / "SHA256SUMS").is_file()


def test_manifests_preserve_classification_source_bytes_and_checksums(
    tmp_path: Path,
) -> None:
    output = tmp_path / "deliverables"
    result = _run_packager(output)

    assert result.returncode == 0, result.stderr
    source_paths_by_project: dict[str, set[str]] = {}
    for project, expected_counts in EXPECTED_COUNTS.items():
        project_dir = output / project
        manifest = json.loads((project_dir / "manifest.json").read_text())
        artifacts = manifest["artifacts"]

        assert manifest["schema_version"] == 1
        assert manifest["project"] == project
        assert manifest["artifact_count"] == sum(expected_counts.values())
        assert Counter(item["category"] for item in artifacts) == expected_counts

        checksum_lines = (project_dir / "SHA256SUMS").read_text().splitlines()
        assert len(checksum_lines) == manifest["artifact_count"]
        assert [item["path"] for item in artifacts] == sorted(
            item["path"] for item in artifacts
        )

        source_paths_by_project[project] = {item["source"] for item in artifacts}
        for item in artifacts:
            source = REPO_ROOT / item["source"]
            destination = project_dir / item["path"]
            assert source.is_file()
            assert destination.is_file()
            assert item["bytes"] == source.stat().st_size == destination.stat().st_size
            assert item["sha256"] == _sha256(source) == _sha256(destination)
            assert f"{item['sha256']}  {item['path']}" in checksum_lines

    assert source_paths_by_project["neural-memory"].isdisjoint(
        source_paths_by_project["sleep-time-compute"]
    )


def test_sleep_time_translation_rights_survive_packaging(tmp_path: Path) -> None:
    output = tmp_path / "deliverables"
    result = _run_packager(output)

    assert result.returncode == 0, result.stderr
    manifest = json.loads(
        (output / "sleep-time-compute" / "manifest.json").read_text()
    )
    translations = [
        item for item in manifest["artifacts"] if item["category"] == "translations"
    ]

    assert Counter(item["rights"] for item in translations) == {
        "internal-only": 13,
        "public": 2,
    }
    public_names = {Path(item["path"]).name for item in translations if item["rights"] == "public"}
    assert public_names == {
        "STC-T04-PLOS-SLEEP-SNN-KR.pdf",
        "STC-T13-PERTURBED-ADVERSARIAL-DREAMING-KR.pdf",
    }


def test_check_mode_detects_package_drift(tmp_path: Path) -> None:
    output = tmp_path / "deliverables"
    build_result = _run_packager(output)
    assert build_result.returncode == 0, build_result.stderr

    target = output / "neural-memory" / "study" / "BOOK.pdf"
    target.write_bytes(b"drift")
    check_result = _run_packager(output, "--check")

    assert check_result.returncode != 0
    assert "BOOK.pdf" in check_result.stderr
