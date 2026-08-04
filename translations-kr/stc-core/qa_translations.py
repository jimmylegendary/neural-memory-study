#!/usr/bin/env python3
"""Structural, rights, source-hash, render, and corpus QA for STC translations."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "translations-kr/stc-core"
OUT = ROOT / "build/publications/translations-kr"


@dataclass(frozen=True)
class QAReport:
    codes: tuple[str, ...]


def validate_record(record: dict) -> QAReport:
    codes = []
    if not record.get("source_sha256"):
        codes.append("missing-source-hash")
    if not record.get("version"):
        codes.append("missing-source-version")
    if record.get("rights") not in {"public", "internal-only", "metadata-only"}:
        codes.append("invalid-rights-state")
    if not record.get("source_path"):
        codes.append("missing-source-path")
    if not record.get("body_path"):
        codes.append("missing-translation-path")
    return QAReport(tuple(codes))


def compare_structure(expected: dict, actual: dict) -> QAReport:
    codes = []
    names = {"equations": "equation", "tables": "table", "figures": "figure"}
    for key, singular in names.items():
        if expected.get(key) is not None and expected.get(key) != actual.get(key):
            codes.append(f"{singular}-count-mismatch")
    return QAReport(tuple(codes))


def fixture_record(**overrides) -> dict:
    record = {
        "source_sha256": "a" * 64,
        "version": "v1",
        "rights": "internal-only",
        "source_path": "source.pdf",
        "body_path": "body.md",
    }
    record.update(overrides)
    return record


def _manifest() -> dict:
    return json.loads((CORE / "manifest.json").read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qa_record(record: dict, render: bool = False) -> dict:
    errors = list(validate_record(record).codes)
    warnings = []
    source = ROOT / record["source_path"]
    body = ROOT / record["body_path"]
    output = OUT / record["output"]
    if not source.exists():
        errors.append("source-missing")
    elif _sha(source) != record["source_sha256"]:
        errors.append("source-hash-mismatch")
    if not body.exists() or body.stat().st_size < 2500:
        errors.append("translation-body-missing-or-too-short")
    source_manifest = CORE / record["paper_id"] / "source-manifest.json"
    artifact_qa = CORE / record["paper_id"] / "translation-qa.json"
    if not source_manifest.exists():
        errors.append("source-manifest-missing")
    if not artifact_qa.exists():
        errors.append("translation-qa-missing")
    if not output.exists():
        errors.append("output-missing")
    else:
        if render:
            artifact = json.loads(artifact_qa.read_text(encoding="utf-8")) if artifact_qa.exists() else {}
            source_pages = artifact.get("source_pages")
            body_pages = artifact.get("pages") - source_pages if source_pages else artifact.get("pages")
            gate = subprocess.run(
                ["python3", str(ROOT / "build/overflow_gate.py"), str(output), str(body_pages)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=900,
            )
            if gate.returncode:
                errors.append("render-overflow")
        if artifact_qa.exists():
            artifact = json.loads(artifact_qa.read_text(encoding="utf-8"))
            if artifact.get("sha256") != _sha(output):
                errors.append("output-hash-mismatch")
            if record["source_type"] == "pdf" and artifact.get("source_plate_mode") != "complete-pdf-pages":
                errors.append("complete-source-plate-missing")
    if record["expected"].get("equations") is None:
        warnings.append("equation-count-deferred-to-complete-source-plate")
    if record.get("source_exception"):
        warnings.append("documented-source-equivalence-exception")
    return {
        "paper_id": record["paper_id"],
        "rights": record["rights"],
        "success": not errors,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--paper")
    group.add_argument("--existing-only", action="store_true")
    group.add_argument("--corpus-only", action="store_true")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--render-all-pages", action="store_true")
    args = parser.parse_args()
    records = _manifest()["papers"]
    if args.paper:
        records = [record for record in records if record["paper_id"] == args.paper]
    if args.existing_only:
        records = [record for record in records if record["body_kind"].startswith("existing")]
    if args.corpus_only:
        reports = [
            {
                "paper_id": record["paper_id"],
                "record_codes": validate_record(record).codes,
                "source_exists": (ROOT / record["source_path"]).exists(),
            }
            for record in records
        ]
    else:
        reports = [qa_record(record, render=args.render_all_pages) for record in records]
    success = 12 <= len(records) <= 15 and all(not report.get("record_codes") and report.get("source_exists", True) if args.corpus_only else report["success"] for report in reports)
    payload = {"success": success, "paper_count": len(records), "reports": reports}
    CORE.joinpath("reports/corpus-audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
