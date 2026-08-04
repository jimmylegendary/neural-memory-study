#!/usr/bin/env python3
"""Reproducible LuaLaTeX builder for the STC publication family."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple


class BuildArtifact(NamedTuple):
    target: str
    source: str
    output: str
    pages: int
    sha256: str


class BuildManifest(NamedTuple):
    success: bool
    artifacts: tuple[BuildArtifact, ...]


TARGETS = {
    "background": (
        "paper-kr/sleep-time-compute-training-background",
        "main.tex",
        "TRAINING-BACKGROUND-FOR-SLEEP-TIME-COMPUTE-KR.pdf",
    ),
    "full": (
        "paper-kr/sleep-time-compute-study",
        "main.tex",
        "SLEEP-TIME-COMPUTE-STRATEGIC-STUDY-KR.pdf",
    ),
    "conference": (
        "paper-kr/sleep-time-compute-study",
        "conference.tex",
        "SLEEP-TIME-COMPUTE-CONFERENCE-KR.pdf",
    ),
    "appendix": (
        "paper-kr/sleep-time-compute-study",
        "appendix.tex",
        "SLEEP-TIME-COMPUTE-CONFERENCE-APPENDIX-KR.pdf",
    ),
}


def _page_count(path: Path) -> int:
    output = subprocess.check_output(["pdfinfo", str(path)], text=True)
    for line in output.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"pdfinfo did not report a page count for {path}")


def _latex_environment(root: Path) -> dict[str, str]:
    """Expose shared publication inputs to both LuaLaTeX and BibTeX.

    latexmk executes BibTeX from the out-of-tree build directory, so a
    bibliography path that is valid from the source directory is not valid
    for BibTeX.  Kpathsea's input variables keep the source portable while
    giving both programs the same absolute search root.  The trailing empty
    component preserves each tool's default search path.
    """

    environment = os.environ.copy()
    common = str((Path(root).resolve() / "paper-kr/common").resolve())
    for variable in ("TEXINPUTS", "BIBINPUTS"):
        prior = environment.get(variable, "")
        environment[variable] = os.pathsep.join(
            component for component in (common, prior, "") if component or component == ""
        )
    return environment


def _build_one(root: Path, target: str) -> BuildArtifact:
    source_rel, tex_name, output_name = TARGETS[target]
    source_dir = root / source_rel
    tex_path = source_dir / tex_name
    if not tex_path.exists():
        raise FileNotFoundError(f"missing publication source: {tex_path}")
    build_dir = root / "build/latex" / target
    output_dir = root / "build/publications"
    build_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "latexmk",
        "-lualatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-outdir={build_dir}",
        tex_name,
    ]
    subprocess.run(
        command,
        cwd=source_dir,
        env=_latex_environment(root),
        check=True,
        timeout=600,
    )
    built = build_dir / f"{Path(tex_name).stem}.pdf"
    destination = output_dir / output_name
    shutil.copy2(built, destination)
    return BuildArtifact(
        target=target,
        source=str(tex_path.relative_to(root)),
        output=str(destination.relative_to(root)),
        pages=_page_count(destination),
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
    )


def build_all(root: Path, targets: list[str] | None = None) -> BuildManifest:
    root = Path(root).resolve()
    requested = targets or ["background", "full", "conference", "appendix"]
    if any(target != "background" for target in requested) and "background" not in requested:
        requested = ["background", *requested]
    artifacts = tuple(_build_one(root, target) for target in requested)
    manifest = BuildManifest(True, artifacts)
    manifest_path = root / "build/publications/build-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "success": manifest.success,
                "artifacts": [artifact._asdict() for artifact in artifacts],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def clean(root: Path, targets: list[str] | None = None) -> None:
    root = Path(root).resolve()
    requested = targets or list(TARGETS)
    for target in requested:
        build_dir = root / "build/latex" / target
        if build_dir.exists():
            shutil.rmtree(build_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--target", choices=[*TARGETS, "all"], default="all")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    targets = list(TARGETS) if args.target == "all" else [args.target]
    if args.clean:
        clean(args.root, targets)
        return 0
    manifest = build_all(args.root, targets)
    for artifact in manifest.artifacts:
        print(f"{artifact.target}: {artifact.output} ({artifact.pages} pages, {artifact.sha256})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
