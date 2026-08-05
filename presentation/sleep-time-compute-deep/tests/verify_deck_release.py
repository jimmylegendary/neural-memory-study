#!/usr/bin/env python3
import argparse
import json
import math
import re
import zipfile
from pathlib import Path


REPO = Path.cwd()
PPTX = REPO / "build/publications/SLEEP-TIME-COMPUTE-DEEP-SEMINAR-112SLIDES-KR.pptx"
STARTER_LAYOUT = REPO / "build/deck-work/template-starter-layout"
FINAL_LAYOUT = REPO / "presentation/sleep-time-compute-deep/build/layouts"
PREVIEW_DIR = REPO / "presentation/sleep-time-compute-deep/build/slides"
REPORT = REPO / "presentation/sleep-time-compute-deep/reports/release-qa.json"


def numbered(entries: list[str], prefix: str) -> list[str]:
    rx = re.compile(rf"^{re.escape(prefix)}(\d+)\.xml$")
    return sorted((name for name in entries if rx.match(name)), key=lambda name: int(rx.match(name).group(1)))


def layout_elements(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    return {str(element.get("name") or element.get("id")): element for element in data["elements"]}


def close_bbox(left: list[float] | None, right: list[float] | None, tolerance: float = 0.02) -> bool:
    if left is None or right is None:
        return left == right
    return len(left) == len(right) and all(math.isclose(a, b, abs_tol=tolerance) for a, b in zip(left, right))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the committed deck and optional build-time visual evidence.")
    parser.add_argument(
        "--artifact-only",
        action="store_true",
        help="skip gitignored preview/layout checks for clean-checkout smoke validation",
    )
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args()
    failures: list[str] = []
    with zipfile.ZipFile(PPTX) as archive:
        names = archive.namelist()
        slides = numbered(names, "ppt/slides/slide")
        notes = numbered(names, "ppt/notesSlides/notesSlide")
        if len(slides) != 112:
            failures.append(f"slide XML count={len(slides)}")
        if len(notes) != 112:
            failures.append(f"notes XML count={len(notes)}")
        note_xml = "\n".join(archive.read(name).decode("utf-8", "replace") for name in notes)
        source_markers = note_xml.count("[Sources]")
        if source_markers != 112:
            failures.append(f"[Sources] markers={source_markers}")
        all_xml = "\n".join(
            archive.read(name).decode("utf-8", "replace")
            for name in names
            if name.endswith(".xml")
        )
        visible_text = "\n".join(re.findall(r"<a:t>(.*?)</a:t>", all_xml, re.I | re.S))
        placeholder_hits = re.findall(r"\b(?:TODO|TBD|FIXME|LOREM|PLACEHOLDER|undefined)\b", visible_text, re.I)
        if placeholder_hits:
            failures.append(f"placeholder hits={len(placeholder_hits)}")

    starter_files: list[Path] = []
    final_files: list[Path] = []
    geometry_mismatches: list[tuple[int, str]] = []
    element_count_mismatches: list[int] = []
    previews: list[Path] = []
    if not args.artifact_only:
        starter_files = sorted(
            STARTER_LAYOUT.glob("starter-slide-*.layout.json"),
            key=lambda path: int(path.name.split("-")[-1].split(".")[0]),
        )
        final_files = sorted(FINAL_LAYOUT.glob("slide-*.layout.json"), key=lambda path: int(path.stem.split("-")[-1].split(".")[0]))
        if len(starter_files) != 112 or len(final_files) != 112:
            failures.append(f"layout files starter/final={len(starter_files)}/{len(final_files)}")
        for index, (starter, final) in enumerate(zip(starter_files, final_files), start=1):
            before = layout_elements(starter)
            after = layout_elements(final)
            if set(before) != set(after):
                element_count_mismatches.append(index)
                continue
            for name in before:
                if not close_bbox(before[name].get("bbox"), after[name].get("bbox")):
                    geometry_mismatches.append((index, name))
        if element_count_mismatches:
            failures.append(f"element inventory mismatch slides={element_count_mismatches}")
        if geometry_mismatches:
            failures.append(f"geometry mismatches={geometry_mismatches[:12]}")

        previews = list(PREVIEW_DIR.glob("slide-*.png"))
        if len(previews) != 112:
            failures.append(f"preview count={len(previews)}")

    result = {
        "status": "pass" if not failures else "fail",
        "mode": "artifact-only" if args.artifact_only else "full",
        "ephemeralChecksSkipped": args.artifact_only,
        "pptx": str(PPTX),
        "slideXmlCount": len(slides),
        "notesXmlCount": len(notes),
        "sourceMarkerCount": source_markers,
        "placeholderHits": len(placeholder_hits),
        "previewCount": len(previews),
        "layoutCount": len(final_files),
        "geometryMismatchCount": len(geometry_mismatches),
        "elementInventoryMismatchCount": len(element_count_mismatches),
        "failures": failures,
    }
    if not args.no_write_report:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
