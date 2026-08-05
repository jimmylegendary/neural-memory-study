#!/usr/bin/env python3
"""Validate the final STC release as a cross-artifact package."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import zipfile
from pathlib import Path


EXPECTED_PAGES = {
    "build/publications/TRAINING-BACKGROUND-FOR-SLEEP-TIME-COMPUTE-KR.pdf": 57,
    "build/publications/SLEEP-TIME-COMPUTE-STRATEGIC-STUDY-KR.pdf": 70,
    "build/publications/SLEEP-TIME-COMPUTE-CONFERENCE-KR.pdf": 12,
    "build/publications/SLEEP-TIME-COMPUTE-CONFERENCE-APPENDIX-KR.pdf": 30,
    "build/publications/SLEEP-TIME-COMPUTE-EASY-COMPANION-KR.pdf": 62,
    "build/publications/SLEEP-TIME-COMPUTE-DEEP-SEMINAR-112SLIDES-KR.pdf": 112,
}


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _pages(path: Path) -> int:
    output = subprocess.run(
        ["pdfinfo", str(path)], check=True, capture_output=True, text=True
    ).stdout
    match = re.search(r"^Pages:\s+(\d+)$", output, re.MULTILINE)
    if not match:
        raise ValueError(f"{path}: pdfinfo did not report Pages")
    return int(match.group(1))


def validate(root: Path) -> list[str]:
    root = root.resolve()
    failures: list[str] = []
    for relative, expected in EXPECTED_PAGES.items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing artifact: {relative}")
            continue
        actual = _pages(path)
        if actual != expected:
            failures.append(f"{relative}: pages {actual}, expected {expected}")

    pptx = root / "build/publications/SLEEP-TIME-COMPUTE-DEEP-SEMINAR-112SLIDES-KR.pptx"
    if not pptx.is_file():
        failures.append("missing deck PPTX")
    else:
        with zipfile.ZipFile(pptx) as archive:
            names = archive.namelist()
            slides = [name for name in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)]
            notes = [name for name in names if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml", name)]
        if len(slides) != 112:
            failures.append(f"deck PPTX: {len(slides)} slides, expected 112")
        if len(notes) != 112:
            failures.append(f"deck PPTX: {len(notes)} notes, expected 112")

    translation = _load(root / "build/publications/translations-kr/build-manifest.json")
    items = translation.get("artifacts", [])
    if translation.get("success") is not True or len(items) != 15:
        failures.append("translation manifest: expected success and 15 artifacts")
    if sum(int(item.get("pages", 0)) for item in items) != 579:
        failures.append("translation manifest: expected 579 total pages")
    rights = [item.get("rights") for item in items]
    if rights.count("public") != 2 or rights.count("internal-only") != 13:
        failures.append("translation manifest: expected 2 public / 13 internal-only")

    deck_qa = _load(root / "presentation/sleep-time-compute-deep/reports/release-qa.json")
    for key, expected in {
        "status": "pass",
        "slideXmlCount": 112,
        "notesXmlCount": 112,
        "sourceMarkerCount": 112,
        "placeholderHits": 0,
        "geometryMismatchCount": 0,
        "elementInventoryMismatchCount": 0,
    }.items():
        if deck_qa.get(key) != expected:
            failures.append(f"deck QA: {key}={deck_qa.get(key)!r}, expected {expected!r}")

    release = _load(root / "build/publications/STC-STUDY-RELEASE-MANIFEST.json")
    if release.get("release_status") != "ready" or release.get("blockers") != []:
        failures.append("release manifest is not ready with zero blockers")
    public_t = release.get("public_artifacts", {}).get("translations", [])
    internal_t = release.get("internal_artifacts", {}).get("translations", [])
    if len(public_t) != 2 or len(internal_t) != 13:
        failures.append("release routing: expected 2 public / 13 internal translations")

    reports = root / "claims/stc-study/reports"
    gate = (reports / "final-gate.txt").read_text(encoding="utf-8")
    passed = re.findall(r"^\s*\[PASS\] (STC-C\d+)", gate, re.MULTILINE)
    blocked = re.findall(r"^\s*\[BLOCK\] (STC-C\d+)", gate, re.MULTILINE)
    if len(passed) != 47 or blocked != [f"STC-C{i:03d}" for i in range(48, 58)]:
        failures.append("Veridraft gate: expected 47 public claims and 10 held P3 claims")
    publish = (reports / "publish-public.txt").read_text(encoding="utf-8")
    events = (reports / "events-final.txt").read_text(encoding="utf-8")
    review = (reports / "review-public.txt").read_text(encoding="utf-8")
    if "PUBLISHED" not in publish or "egress: confidentiality decide() + redaction re-sweep passed" not in publish:
        failures.append("Veridraft public egress did not pass")
    if "hash chain intact: True" not in events:
        failures.append("Veridraft lifecycle hash chain is not intact")
    if "self-authored (no CAW-02 signature — digest N/A)" not in review:
        failures.append("Veridraft self-authored provenance state is not documented")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    failures = validate(args.root)
    if failures:
        print("\n".join(f"FAIL: {failure}" for failure in failures))
        return 1
    print("PASS: 6 core PDFs, 112-slide PPTX/notes, 15 translations/579 pages")
    print("PASS: rights routing 2 public / 13 internal-only")
    print("PASS: Veridraft 47 public claims / 10 patent-held P3, public egress, intact event chain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
