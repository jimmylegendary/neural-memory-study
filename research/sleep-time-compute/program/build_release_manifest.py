#!/usr/bin/env python3
"""Build the deterministic release manifest for the sleep-time-compute corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


SOURCE_FREEZE = "2026-08-05"
OUTPUT = Path("build/publications/STC-STUDY-RELEASE-MANIFEST.json")

CORE = (
    {
        "id": "training-background",
        "path": "build/publications/TRAINING-BACKGROUND-FOR-SLEEP-TIME-COMPUTE-KR.pdf",
        "kind": "background-paper",
        "builder": "research/sleep-time-compute/program/build_publications.py",
        "missing": "missing-background-pdf",
    },
    {
        "id": "strategic-study",
        "path": "build/publications/SLEEP-TIME-COMPUTE-STRATEGIC-STUDY-KR.pdf",
        "kind": "study-paper",
        "builder": "research/sleep-time-compute/program/build_publications.py",
        "missing": "missing-study-pdf",
    },
    {
        "id": "conference-paper",
        "path": "build/publications/SLEEP-TIME-COMPUTE-CONFERENCE-KR.pdf",
        "kind": "conference-paper",
        "builder": "research/sleep-time-compute/program/build_publications.py",
        "missing": "missing-conference-pdf",
    },
    {
        "id": "conference-appendix",
        "path": "build/publications/SLEEP-TIME-COMPUTE-CONFERENCE-APPENDIX-KR.pdf",
        "kind": "conference-appendix",
        "builder": "research/sleep-time-compute/program/build_publications.py",
        "missing": "missing-conference-appendix-pdf",
    },
    {
        "id": "easy-companion",
        "path": "build/publications/SLEEP-TIME-COMPUTE-EASY-COMPANION-KR.pdf",
        "kind": "easy-companion",
        "builder": "easy/sleep-time-compute/program/build_easy_companion.py",
        "missing": "missing-easy-companion-pdf",
    },
    {
        "id": "seminar-deck-pdf",
        "path": "build/publications/SLEEP-TIME-COMPUTE-DEEP-SEMINAR-112SLIDES-KR.pdf",
        "kind": "seminar-deck-pdf",
        "builder": "presentation/sleep-time-compute-deep/src/build-deck.mjs",
        "missing": "missing-deck-pdf",
    },
    {
        "id": "seminar-deck-pptx",
        "path": "build/publications/SLEEP-TIME-COMPUTE-DEEP-SEMINAR-112SLIDES-KR.pptx",
        "kind": "seminar-deck-source",
        "builder": "presentation/sleep-time-compute-deep/src/build-deck.mjs",
        "missing": "missing-deck-pptx",
    },
)


def _load(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _qa_pass(report: dict[str, Any] | None, kind: str) -> bool:
    if not report:
        return False
    if kind == "easy":
        if report.get("status") == "pass":
            return True
        return bool(
            report.get("success") is True
            or (
                isinstance(report.get("manifest"), dict)
                and report["manifest"].get("success") is True
                and isinstance(report.get("pdf"), dict)
                and report["pdf"].get("success") is True
            )
        )
    if kind == "final":
        return bool(
            report.get("status") == "pass"
            or report.get("overall_status") == "pass"
            or report.get("success") is True
        )
    return report.get("status") == "pass"


def _index_by_output(manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not manifest:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in manifest.get("artifacts", []):
        if isinstance(item, dict) and isinstance(item.get("output"), str):
            result[item["output"]] = item
    return result


def _artifact(
    root: Path,
    spec: dict[str, str],
    metadata: dict[str, Any] | None,
    source_commit: str,
    qa_report: str,
    qa_status: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    relative = spec["path"]
    path = root / relative
    if not path.is_file():
        return None, [spec["missing"]]
    actual = _sha256(path)
    blockers: list[str] = []
    expected = (metadata or {}).get("sha256")
    if expected and expected != actual:
        blockers.append(f"hash-mismatch-{spec['id']}")
    pages = (metadata or {}).get("pages")
    item: dict[str, Any] = {
        "id": spec["id"],
        "kind": spec["kind"],
        "path": relative,
        "sha256": actual,
        "bytes": path.stat().st_size,
        "audience": "internal-research",
        "rights_state": "authored-citation-backed",
        "builder": spec["builder"],
        "qa_report": qa_report,
        "qa_status": qa_status,
        "source_commit": source_commit,
    }
    if pages is not None:
        item["pages"] = pages
    return item, blockers


def build_manifest(root: Path, source_commit: str | None = None) -> dict[str, Any]:
    """Return a deterministic, rights-aware manifest without writing it."""
    root = root.resolve()
    commit = source_commit or _git_commit(root)
    academic_path = root / "build/publications/build-manifest.json"
    translation_path = root / "build/publications/translations-kr/build-manifest.json"
    easy_manifest_path = root / "easy/sleep-time-compute/reports/build-manifest.json"
    easy_qa_path = root / "easy/sleep-time-compute/reports/qa-report.json"
    deck_qa_path = root / "presentation/sleep-time-compute-deep/reports/release-qa.json"
    final_qa_path = root / "build/publications/FINAL-QA.json"

    academic = _load(academic_path)
    translations = _load(translation_path)
    easy_manifest = _load(easy_manifest_path)
    easy_qa = _load(easy_qa_path)
    deck_qa = _load(deck_qa_path)
    final_qa = _load(final_qa_path)
    academic_index = _index_by_output(academic)
    easy_index = _index_by_output(easy_manifest)

    blockers: list[str] = []
    if not academic or academic.get("success") is not True:
        blockers.append("academic-build-manifest-failed-or-missing")
    if not translations or translations.get("success") is not True:
        blockers.append("translation-build-manifest-failed-or-missing")
    easy_ok = _qa_pass(easy_qa, "easy")
    deck_ok = _qa_pass(deck_qa, "deck")
    final_ok = _qa_pass(final_qa, "final")
    if not easy_ok:
        blockers.append("easy-qa-failed-or-missing")
    if not deck_ok:
        blockers.append("deck-qa-failed-or-missing")
    if not final_ok:
        blockers.append("final-qa-failed-or-missing")

    core: list[dict[str, Any]] = []
    for spec in CORE:
        metadata = academic_index.get(spec["path"]) or easy_index.get(spec["path"])
        if spec["kind"].startswith("seminar"):
            metadata = {
                "pages": (deck_qa or {}).get("slideXmlCount"),
            }
            qa_report = str(deck_qa_path.relative_to(root))
            qa_status = "pass" if deck_ok else "fail"
        elif spec["kind"] == "easy-companion":
            qa_report = str(easy_qa_path.relative_to(root))
            qa_status = "pass" if easy_ok else "fail"
        else:
            qa_report = str(academic_path.relative_to(root))
            qa_status = "pass" if academic and academic.get("success") is True else "fail"
        item, item_blockers = _artifact(root, spec, metadata, commit, qa_report, qa_status)
        blockers.extend(item_blockers)
        if item:
            core.append(item)

    public_translations: list[dict[str, Any]] = []
    internal_translations: list[dict[str, Any]] = []
    for raw in (translations or {}).get("artifacts", []):
        if not isinstance(raw, dict) or not isinstance(raw.get("output"), str):
            blockers.append("invalid-translation-manifest-entry")
            continue
        path = root / raw["output"]
        paper_id = str(raw.get("paper_id") or Path(raw["output"]).stem)
        if not path.is_file():
            blockers.append(f"missing-translation-{paper_id}")
            continue
        actual = _sha256(path)
        if raw.get("sha256") and raw["sha256"] != actual:
            blockers.append(f"hash-mismatch-translation-{paper_id}")
        rights = str(raw.get("rights") or "internal-only")
        item = {
            "id": paper_id,
            "kind": "paper-translation",
            "path": raw["output"],
            "sha256": actual,
            "bytes": path.stat().st_size,
            "pages": raw.get("pages"),
            "audience": "public" if rights == "public" else "internal-research",
            "rights_state": rights,
            "builder": "translations-kr/program/build_translations.py",
            "qa_report": str(translation_path.relative_to(root)),
            "qa_status": "pass" if translations and translations.get("success") is True else "fail",
            "source_commit": commit,
        }
        (public_translations if rights == "public" else internal_translations).append(item)

    blockers = sorted(set(blockers))
    return {
        "schema_version": "1.0.0",
        "source_freeze": SOURCE_FREEZE,
        "source_commit": commit,
        "release_status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "qa": {
            "academic": {"status": "pass" if academic and academic.get("success") is True else "fail", "report": str(academic_path.relative_to(root))},
            "translations": {"status": "pass" if translations and translations.get("success") is True else "fail", "report": str(translation_path.relative_to(root))},
            "easy": {"status": "pass" if easy_ok else "fail", "report": str(easy_qa_path.relative_to(root))},
            "deck": {"status": "pass" if deck_ok else "fail", "report": str(deck_qa_path.relative_to(root))},
            "final": {"status": "pass" if final_ok else "fail", "report": str(final_qa_path.relative_to(root))},
        },
        "core_artifacts": core,
        "public_artifacts": {
            "authored": core,
            "translations": sorted(public_translations, key=lambda item: item["id"]),
        },
        "internal_artifacts": {
            "translations": sorted(internal_translations, key=lambda item: item["id"]),
        },
    }


def write_manifest(
    root: Path,
    source_commit: str | None = None,
    require_ready: bool = False,
) -> Path:
    report = build_manifest(root, source_commit=source_commit)
    if require_ready and report["release_status"] != "ready":
        raise RuntimeError("release blocked: " + ", ".join(report["blockers"]))
    target = root.resolve() / OUTPUT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    target = write_manifest(args.root, require_ready=args.require_ready)
    report = _load(target) or {}
    print(f"{target}: {report.get('release_status', 'unknown')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
