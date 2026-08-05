"""Build the reader-facing Neural Memory and Sleep-Time Compute packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class PackagingError(RuntimeError):
    """Raised when the release package cannot be built or verified."""


@dataclass(frozen=True)
class ArtifactSpec:
    project: str
    category: str
    source: Path
    destination: Path
    metadata: dict[str, Any] = field(default_factory=dict)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
        if not matches:
            raise PackagingError(f"artifact pattern matched no files: {pattern}")
        files.extend(matches)
    return files


def _explicit_files(root: Path, paths: Iterable[str]) -> list[Path]:
    files = [root / path for path in paths]
    missing = [path.relative_to(root).as_posix() for path in files if not path.is_file()]
    if missing:
        raise PackagingError(f"missing source artifacts: {', '.join(missing)}")
    return files


def _simple_specs(
    *,
    root: Path,
    project: str,
    category: str,
    files: Iterable[Path],
    rights: str | None = None,
) -> list[ArtifactSpec]:
    specs = []
    for source in files:
        metadata = {"rights": rights} if rights else {}
        specs.append(
            ArtifactSpec(
                project=project,
                category=category,
                source=source,
                destination=Path(category) / source.name,
                metadata=metadata,
            )
        )
    return specs


def _translation_metadata(root: Path) -> dict[str, dict[str, Any]]:
    manifest_path = root / "build/publications/translations-kr/build-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PackagingError(f"cannot read translation manifest: {manifest_path}") from error

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise PackagingError("translation manifest has no artifacts array")

    result: dict[str, dict[str, Any]] = {}
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("output"), str):
            raise PackagingError("translation manifest contains a malformed artifact")
        name = Path(item["output"]).name
        rights = item.get("rights")
        paper_id = item.get("paper_id")
        if rights not in {"public", "internal-only"} or not isinstance(paper_id, str):
            raise PackagingError(f"translation metadata is incomplete: {name}")
        if name in result:
            raise PackagingError(f"duplicate translation metadata: {name}")
        result[name] = {"paper_id": paper_id, "rights": rights}
    return result


def build_specs(root: Path) -> dict[str, list[ArtifactSpec]]:
    neural: list[ArtifactSpec] = []
    neural.extend(
        _simple_specs(
            root=root,
            project="neural-memory",
            category="study",
            files=_explicit_files(root, ["build/BOOK.pdf"]),
        )
    )
    neural.extend(
        _simple_specs(
            root=root,
            project="neural-memory",
            category="easy",
            files=_existing_files(root, ["easy/pdf/*.pdf"]),
        )
    )
    neural.extend(
        _simple_specs(
            root=root,
            project="neural-memory",
            category="translations",
            files=_existing_files(root, ["translations-kr/pdf/*.pdf"]),
            rights="internal-only",
        )
    )
    neural.extend(
        _simple_specs(
            root=root,
            project="neural-memory",
            category="seminar",
            files=_existing_files(root, ["seminar/*.pdf", "seminar/*.pptx"]),
        )
    )
    neural.extend(
        _simple_specs(
            root=root,
            project="neural-memory",
            category="analysis",
            files=_explicit_files(
                root,
                [
                    "research/attn-vs-hope/Attn-vs-HOPE.xlsx",
                    "research/ttt-titans-efficiency-study.pdf",
                    "multiarch/multiarch-report.pdf",
                ],
            ),
        )
    )

    sleep: list[ArtifactSpec] = []
    sleep.extend(
        _simple_specs(
            root=root,
            project="sleep-time-compute",
            category="study",
            files=_explicit_files(
                root,
                [
                    "build/publications/TRAINING-BACKGROUND-FOR-SLEEP-TIME-COMPUTE-KR.pdf",
                    "build/publications/SLEEP-TIME-COMPUTE-CONFERENCE-KR.pdf",
                    "build/publications/SLEEP-TIME-COMPUTE-CONFERENCE-APPENDIX-KR.pdf",
                    "build/publications/SLEEP-TIME-COMPUTE-STRATEGIC-STUDY-KR.pdf",
                    "build/publications/SLEEP-TIME-COMPUTE-MONOGRAPH-KR.pdf",
                    "build/publications/SLEEP-TIME-COMPUTE-MONOGRAPH-KR.docx",
                    "build/publications/SLEEP-TIME-COMPUTE-MONOGRAPH-KR.html",
                    "build/publications/SLEEP-TIME-COMPUTE-POSITION-PAPER-EN.pdf",
                    "build/publications/SLEEP-TIME-COMPUTE-POSITION-PAPER-EN.docx",
                    "build/publications/SLEEP-TIME-COMPUTE-POSITION-PAPER-EN.html",
                ],
            ),
        )
    )
    sleep.extend(
        _simple_specs(
            root=root,
            project="sleep-time-compute",
            category="easy",
            files=_explicit_files(
                root,
                ["build/publications/SLEEP-TIME-COMPUTE-EASY-COMPANION-KR.pdf"],
            )
            + _existing_files(root, ["easy/sleep-time-compute/pdf/*.pdf"]),
        )
    )

    translation_meta = _translation_metadata(root)
    translation_files = _existing_files(
        root, ["build/publications/translations-kr/STC-T*.pdf"]
    )
    for source in translation_files:
        if source.name not in translation_meta:
            raise PackagingError(f"translation has no rights metadata: {source.name}")
        sleep.append(
            ArtifactSpec(
                project="sleep-time-compute",
                category="translations",
                source=source,
                destination=Path("translations") / source.name,
                metadata=translation_meta[source.name],
            )
        )
    extra_metadata = sorted(set(translation_meta) - {path.name for path in translation_files})
    if extra_metadata:
        raise PackagingError(
            f"translation metadata has no PDF: {', '.join(extra_metadata)}"
        )

    sleep.extend(
        _simple_specs(
            root=root,
            project="sleep-time-compute",
            category="seminar",
            files=_explicit_files(
                root,
                [
                    "build/publications/SLEEP-TIME-COMPUTE-DEEP-SEMINAR-112SLIDES-KR.pdf",
                    "build/publications/SLEEP-TIME-COMPUTE-DEEP-SEMINAR-112SLIDES-KR.pptx",
                ],
            ),
        )
    )

    publication_manifests = _explicit_files(
        root,
        [
            "build/publications/build-manifest.json",
            "build/publications/FINAL-QA.json",
            "build/publications/STC-STUDY-RELEASE-MANIFEST.json",
            "build/publications/SHA256SUMS",
        ],
    )
    translation_manifests = _explicit_files(
        root,
        [
            "build/publications/translations-kr/build-manifest.json",
            "build/publications/translations-kr/SHA256SUMS",
        ],
    )
    for source in publication_manifests:
        sleep.append(
            ArtifactSpec(
                project="sleep-time-compute",
                category="manifests",
                source=source,
                destination=Path("manifests/publications") / source.name,
            )
        )
    for source in translation_manifests:
        sleep.append(
            ArtifactSpec(
                project="sleep-time-compute",
                category="manifests",
                source=source,
                destination=Path("manifests/translations") / source.name,
            )
        )

    projects = {"neural-memory": neural, "sleep-time-compute": sleep}
    for project, specs in projects.items():
        destinations = [spec.destination.as_posix() for spec in specs]
        sources = [spec.source for spec in specs]
        if len(destinations) != len(set(destinations)):
            raise PackagingError(f"duplicate destination in {project}")
        if len(sources) != len(set(sources)):
            raise PackagingError(f"duplicate source in {project}")
        projects[project] = sorted(specs, key=lambda spec: spec.destination.as_posix())

    neural_sources = {spec.source for spec in projects["neural-memory"]}
    sleep_sources = {spec.source for spec in projects["sleep-time-compute"]}
    overlap = sorted(neural_sources & sleep_sources)
    if overlap:
        names = ", ".join(path.relative_to(root).as_posix() for path in overlap)
        raise PackagingError(f"artifacts assigned to both projects: {names}")
    return projects


def _artifact_record(root: Path, spec: ArtifactSpec) -> dict[str, Any]:
    record: dict[str, Any] = {
        "bytes": spec.source.stat().st_size,
        "category": spec.category,
        "path": spec.destination.as_posix(),
        "sha256": sha256(spec.source),
        "source": spec.source.relative_to(root).as_posix(),
    }
    record.update(spec.metadata)
    return record


def _manifest(root: Path, project: str, specs: list[ArtifactSpec]) -> dict[str, Any]:
    records = [_artifact_record(root, spec) for spec in specs]
    return {
        "artifact_count": len(records),
        "artifacts": records,
        "category_counts": dict(sorted(Counter(item["category"] for item in records).items())),
        "project": project,
        "schema_version": SCHEMA_VERSION,
    }


def _json_text(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _checksums_text(manifest: dict[str, Any]) -> str:
    return "".join(
        f"{item['sha256']}  {item['path']}\n" for item in manifest["artifacts"]
    )


def _project_readme(project: str, manifest: dict[str, Any]) -> str:
    if project == "neural-memory":
        title = "Neural Memory 산출물"
        description = (
            "기존 Neural Memory 연구의 학습서, 쉬운 설명, 논문 의역, seminar, "
            "분석 workbook을 한 위치에서 찾기 위한 배포 패키지다."
        )
        notes = (
            "`translations/`는 내부 연구용으로 취급한다. 원본 생성·편집 경로는 "
            "각 파일의 `manifest.json` 내 `source` 필드에 기록되어 있다."
        )
    else:
        title = "Sleep-Time Compute 산출물"
        description = (
            "Sleep-Time Compute의 study paper, training background, 쉬운 설명, "
            "15편 의역, 112-slide seminar 및 release 검증 자료를 모은 배포 패키지다."
        )
        notes = (
            "`translations/`의 외부 배포 가능 여부는 파일명이 아니라 "
            "`manifest.json`의 `rights` 필드를 따른다. `internal-only` 파일은 조직 외부로 "
            "배포하지 않는다."
        )

    rows = "\n".join(
        f"| `{category}/` | {count} |"
        for category, count in manifest["category_counts"].items()
    )
    return f"""# {title}

{description}

| 디렉터리 | 파일 수 |
|---|---:|
{rows}

총 {manifest['artifact_count']}개 artifact다. `manifest.json`은 source path, byte size,
SHA-256 및 권리 metadata를 제공하고 `SHA256SUMS`는 패키지 파일 무결성을 검증한다.

{notes}
"""


def _root_readme(manifests: dict[str, dict[str, Any]]) -> str:
    neural_count = manifests["neural-memory"]["artifact_count"]
    sleep_count = manifests["sleep-time-compute"]["artifact_count"]
    return f"""# 연구 산출물 배포 패키지

두 연구 프로그램의 reader-facing 파일을 동일한 계층에 분리했다.

- [`neural-memory/`](neural-memory/README.md): Neural Memory artifact {neural_count}개
- [`sleep-time-compute/`](sleep-time-compute/README.md): Sleep-Time Compute artifact {sleep_count}개

이 디렉터리는 배포·열람용 canonical index다. 원본 생성 경로는 유지되며 각 프로젝트의
`manifest.json`이 package file과 source file의 대응 관계를 기록한다.

재생성:

```bash
python3 scripts/package_deliverables.py --root . --output deliverables
```

검증:

```bash
python3 scripts/package_deliverables.py --root . --output deliverables --check
```
"""


def _remove_generated_project(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def build(root: Path, output: Path, projects: dict[str, list[ArtifactSpec]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifests = {
        project: _manifest(root, project, specs) for project, specs in projects.items()
    }

    for project, specs in projects.items():
        project_dir = output / project
        _remove_generated_project(project_dir)
        project_dir.mkdir(parents=True)
        for spec in specs:
            destination = project_dir / spec.destination
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(spec.source, destination)

        manifest = manifests[project]
        (project_dir / "manifest.json").write_text(
            _json_text(manifest), encoding="utf-8"
        )
        (project_dir / "SHA256SUMS").write_text(
            _checksums_text(manifest), encoding="utf-8"
        )
        (project_dir / "README.md").write_text(
            _project_readme(project, manifest), encoding="utf-8"
        )

    (output / "README.md").write_text(_root_readme(manifests), encoding="utf-8")


def check(root: Path, output: Path, projects: dict[str, list[ArtifactSpec]]) -> None:
    errors: list[str] = []
    expected_manifests = {
        project: _manifest(root, project, specs) for project, specs in projects.items()
    }

    expected_root_readme = _root_readme(expected_manifests)
    root_readme = output / "README.md"
    if not root_readme.is_file() or root_readme.read_text(encoding="utf-8") != expected_root_readme:
        errors.append("README.md is missing or stale")

    for project, manifest in expected_manifests.items():
        project_dir = output / project
        manifest_path = project_dir / "manifest.json"
        if not manifest_path.is_file():
            errors.append(f"{project}/manifest.json is missing")
            continue
        try:
            actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            errors.append(f"{project}/manifest.json is invalid JSON")
            continue
        if actual_manifest != manifest:
            errors.append(f"{project}/manifest.json is stale")

        checksum_path = project_dir / "SHA256SUMS"
        expected_checksums = _checksums_text(manifest)
        if not checksum_path.is_file() or checksum_path.read_text(encoding="utf-8") != expected_checksums:
            errors.append(f"{project}/SHA256SUMS is missing or stale")

        readme_path = project_dir / "README.md"
        expected_readme = _project_readme(project, manifest)
        if not readme_path.is_file() or readme_path.read_text(encoding="utf-8") != expected_readme:
            errors.append(f"{project}/README.md is missing or stale")

        for item in manifest["artifacts"]:
            destination = project_dir / item["path"]
            if not destination.is_file():
                errors.append(f"{project}/{item['path']} is missing")
                continue
            if destination.stat().st_size != item["bytes"] or sha256(destination) != item["sha256"]:
                errors.append(f"{project}/{item['path']} differs from source")

    if errors:
        raise PackagingError("package check failed:\n- " + "\n- ".join(errors))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--output", default="deliverables", help="package directory")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify an existing package without modifying it",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    try:
        if not root.is_dir():
            raise PackagingError(f"repository root does not exist: {root}")
        projects = build_specs(root)
        if args.check:
            check(root, output, projects)
            print(
                "PASS: verified "
                f"{sum(len(items) for items in projects.values())} artifacts in {output}"
            )
        else:
            build(root, output, projects)
            print(
                "PASS: packaged "
                f"{sum(len(items) for items in projects.values())} artifacts in {output}"
            )
    except PackagingError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
