#!/usr/bin/env python3
"""Manifest, content, link, font, overflow, and render QA for STC easy companions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "easy/sleep-time-compute"
REQUIRED_IDS = [f"E{i:02d}" for i in range(1, 13)]
PLACEHOLDER_RE = re.compile(r"\{\{(BG|STUDY|CLAIM|FIG):([^}|]+)(?:\|[^}]+)?\}\}")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest(root: Path = ROOT) -> dict:
    core = root / "easy/sleep-time-compute"
    manifest = _json(core / "manifest.json")
    claims = {x["claim_id"] for x in _json(root / "claims/stc-study/claim-map.json")["claims"]}
    figures = {x["figure_id"]: x for x in _json(root / "claims/stc-study/figure-ledger.json")["figures"]}
    concepts = {x["concept_id"] for x in _json(root / "paper-kr/common/concept-registry.json")["concepts"]}
    errors: list[str] = []
    records = manifest.get("booklets", [])
    ids = [r.get("id") for r in records]
    if ids != REQUIRED_IDS:
        errors.append(f"booklet-order:{ids}")
    if len({r.get("slug") for r in records}) != len(records):
        errors.append("duplicate-slug")
    for record in records:
        booklet_id = record["id"]
        body_path = core / record["file"]
        if not body_path.exists():
            errors.append(f"missing-body:{booklet_id}")
            continue
        body = body_path.read_text(encoding="utf-8")
        for heading in manifest["required_blocks"]:
            if f"## {heading}" not in body:
                errors.append(f"missing-block:{booklet_id}:{heading}")
        if re.search(r"\b(?:TODO|TBD|PLACEHOLDER)\b|\?\?+", body, flags=re.IGNORECASE):
            errors.append(f"placeholder-language:{booklet_id}")
        if re.search(r"(?:PDF\s*)?(?:p\.|pp\.|page|페이지)\s*\d+", body, flags=re.IGNORECASE):
            errors.append(f"hard-coded-page-reference:{booklet_id}")
        declared_claims = set(record["claims"])
        declared_figures = set(record["figures"])
        declared_concepts = set(record["background_concepts"])
        for kind, value in PLACEHOLDER_RE.findall(body):
            if kind == "CLAIM" and value not in declared_claims:
                errors.append(f"undeclared-claim:{booklet_id}:{value}")
            elif kind == "FIG" and value not in declared_figures:
                errors.append(f"undeclared-figure:{booklet_id}:{value}")
            elif kind == "BG" and value not in declared_concepts:
                errors.append(f"undeclared-concept:{booklet_id}:{value}")
        errors.extend(f"unknown-claim:{booklet_id}:{x}" for x in sorted(declared_claims - claims))
        errors.extend(f"unknown-figure:{booklet_id}:{x}" for x in sorted(declared_figures - set(figures)))
        errors.extend(f"unknown-concept:{booklet_id}:{x}" for x in sorted(declared_concepts - concepts))
        for figure_id in declared_figures & set(figures):
            if not figures[figure_id]["license"]["reuse_allowed"]:
                errors.append(f"figure-reuse-denied:{booklet_id}:{figure_id}")
            image = root / "paper-kr/sleep-time-compute-study/figures" / f"{figure_id.lower()}.pdf"
            if not image.exists():
                errors.append(f"missing-figure:{booklet_id}:{figure_id}")
    return {"success": not errors, "errors": errors, "booklets": len(records)}


def _pages(path: Path) -> int:
    out = subprocess.check_output(["pdfinfo", str(path)], text=True, errors="replace")
    match = re.search(r"^Pages:\s+(\d+)", out, flags=re.MULTILINE)
    return int(match.group(1)) if match else 0


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pdf_report(paths: list[Path], strict: bool, render: bool) -> dict:
    errors: list[str] = []
    records: list[dict] = []
    render_root = ROOT / "build/easy-companion/render"
    if render:
        render_root.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if not path.exists():
            errors.append(f"missing-pdf:{path}")
            continue
        pages = _pages(path)
        text = subprocess.check_output(["pdftotext", str(path), "-"], text=True, errors="replace")
        if "�" in text:
            errors.append(f"replacement-glyph:{path.name}")
        if pages < 4:
            errors.append(f"too-short:{path.name}:{pages}")
        object_dump = subprocess.check_output(
            ["mutool", "show", str(path), "grep"], text=True, errors="replace", stderr=subprocess.DEVNULL
        )
        links = object_dump.count("/Subtype/Link")
        background_links = object_dump.count("TRAINING-BACKGROUND-FOR-SLEEP-TIME-COMPUTE-KR.pdf")
        if background_links == 0:
            errors.append(f"missing-background-link:{path.name}")
        if strict:
            gate = subprocess.run(
                ["python3", str(ROOT / "build/overflow_gate.py"), str(path)],
                text=True,
                capture_output=True,
            )
            if gate.returncode:
                errors.append(f"overflow:{path.name}:{gate.stdout.strip()}")
        if render:
            prefix = render_root / path.stem
            subprocess.run(["pdftoppm", "-png", "-r", "72", str(path), str(prefix)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        records.append({"path": str(path.relative_to(ROOT)), "pages": pages, "sha256": _sha(path), "links": links, "background_links": background_links})
    return {"success": not errors, "errors": errors, "artifacts": records}


def selected_records(spec: str | None) -> list[dict]:
    records = _json(CORE / "manifest.json")["booklets"]
    if not spec:
        return records
    wanted = {x.strip() for x in spec.split(",") if x.strip()}
    return [r for r in records if r["id"] in wanted]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--booklets")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--combined", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--render-all-pages", action="store_true")
    args = parser.parse_args()
    manifest_report = validate_manifest()
    records = selected_records(None if args.all else args.booklets)
    paths = [CORE / "pdf" / f"{r['id']}-{r['slug']}.pdf" for r in records]
    if args.combined:
        paths.append(ROOT / "build/publications/SLEEP-TIME-COMPUTE-EASY-COMPANION-KR.pdf")
    report = {
        "manifest": manifest_report,
        "pdf": pdf_report(paths, args.strict, args.render_all_pages),
    }
    report["success"] = report["manifest"]["success"] and report["pdf"]["success"]
    (CORE / "reports/qa-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
