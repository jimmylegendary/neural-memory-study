#!/usr/bin/env python3
"""Run and record the complete STC study release QA suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


SLIDES_TEST = "/home/jimmy/.codex/plugins/cache/openai-primary-runtime/presentations/26.802.11031/skills/presentations/container_tools/slides_test.py"
OUTPUT = Path("build/publications/FINAL-QA.json")
LOG_DIR = Path("build/publications/qa-logs")


COMMANDS: tuple[dict[str, Any], ...] = (
    {"id": "research-pytest", "argv": ["uv", "run", "--project", "research/sleep-time-compute", "pytest", "-q", "research/sleep-time-compute/tests"]},
    {"id": "analytical-model-unittest", "argv": ["python3", "-m", "unittest", "-q", "experiments/E5-attn-vs-hope/test_model.py"]},
    {"id": "workbook-source-unittest", "argv": ["python3", "-m", "unittest", "-q", "experiments/E5-attn-vs-hope/sheet/test_workbook_source.py"]},
    {"id": "artifact-pytest", "argv": ["python3", "-m", "pytest", "-q", "paper-kr/common/tests", "translations-kr/stc-core/tests", "easy/sleep-time-compute/tests"]},
    {"id": "academic-build", "argv": ["python3", "paper-kr/common/build_publications.py", "--root", ".", "--target", "all", "--clean"]},
    {"id": "academic-background-qa", "argv": ["python3", "paper-kr/common/qa_publications.py", "--pdf", "build/publications/TRAINING-BACKGROUND-FOR-SLEEP-TIME-COMPUTE-KR.pdf", "--kind", "background", "--require-all-concepts"]},
    {"id": "academic-study-qa", "argv": ["python3", "paper-kr/common/qa_publications.py", "--pdf", "build/publications/SLEEP-TIME-COMPUTE-STRATEGIC-STUDY-KR.pdf", "--kind", "study", "--claim-map", "claims/stc-study/claim-map.json", "--figure-ledger", "claims/stc-study/figure-ledger.json"]},
    {"id": "academic-conference-qa", "argv": ["python3", "paper-kr/common/qa_publications.py", "--pdf", "build/publications/SLEEP-TIME-COMPUTE-CONFERENCE-KR.pdf", "--kind", "conference", "--min-pages", "12", "--max-pages", "18"]},
    {"id": "academic-appendix-qa", "argv": ["python3", "paper-kr/common/qa_publications.py", "--pdf", "build/publications/SLEEP-TIME-COMPUTE-CONFERENCE-APPENDIX-KR.pdf", "--kind", "appendix"]},
    {"id": "translation-build", "argv": ["python3", "translations-kr/stc-core/build_translations.py", "--all"]},
    {"id": "translation-qa", "argv": ["python3", "translations-kr/stc-core/qa_translations.py", "--all", "--strict", "--render-all-pages"]},
    {"id": "translation-audit", "argv": ["python3", "translations-kr/stc-core/audit_translations.py"]},
    {"id": "easy-build", "argv": ["python3", "easy/sleep-time-compute/build.py", "--all", "--combined"]},
    {"id": "easy-qa", "argv": ["python3", "easy/sleep-time-compute/qa.py", "--all", "--combined", "--strict", "--render-all-pages"]},
    {"id": "deck-content-test", "argv": ["node", "--test", "presentation/sleep-time-compute-deep/tests/content.test.mjs"]},
    {"id": "deck-release-qa", "argv": ["python3", "presentation/sleep-time-compute-deep/tests/verify_deck_release.py"]},
    {"id": "deck-overflow-qa", "argv": ["python3", SLIDES_TEST, "build/publications/SLEEP-TIME-COMPUTE-DEEP-SEMINAR-112SLIDES-KR.pptx", "--width", "1280", "--height", "720"], "env": {"PYTHONPATH": "build/deck-work/python-deps"}},
    {"id": "program-validation", "argv": ["python3", "research/sleep-time-compute/program/validate_program.py", "--all"]},
    {"id": "authoring-marker-reference-validation", "argv": ["python3", "research/sleep-time-compute/program/validate_program.py", "--all", "--reject-authoring-markers", "--reject-latex-reference-errors"]},
    {"id": "release-manifest-tests", "argv": ["python3", "-m", "pytest", "-q", "research/sleep-time-compute/tests/test_release_manifest.py"]},
    {"id": "git-whitespace", "argv": ["git", "diff", "--check"]},
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _metrics(output: str) -> dict[str, int]:
    metrics: dict[str, int] = {}
    patterns = {
        "pytest_passed": r"(\d+) passed",
        "pytest_failed": r"(\d+) failed",
        "unittest_ran": r"Ran (\d+) tests?",
        "node_passed": r"(?:#|ℹ) pass (\d+)",
        "node_failed": r"(?:#|ℹ) fail (\d+)",
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, output)
        if matches:
            metrics[key] = int(matches[-1])
    return metrics


def _tail(output: str, limit: int = 1200) -> str:
    compact = "\n".join(line.rstrip() for line in output.splitlines() if line.strip())
    return compact[-limit:]


def _run(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    argv = [str(item) for item in spec["argv"]]
    env = os.environ.copy()
    env.update(spec.get("env", {}))
    start = time.monotonic()
    proc = subprocess.run(argv, cwd=root, env=env, capture_output=True)
    duration = round(time.monotonic() - start, 3)
    output = proc.stdout + proc.stderr
    LOG_DIR_ABS = root / LOG_DIR
    LOG_DIR_ABS.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR_ABS / f"{spec['id']}.log"
    log.write_bytes(output)
    decoded = output.decode("utf-8", errors="replace")
    result = {
        "id": spec["id"],
        "command": subprocess.list2cmdline(argv),
        "exit_code": proc.returncode,
        "status": "pass" if proc.returncode == 0 else "fail",
        "duration_seconds": duration,
        "output_sha256": _sha(output),
        "output_bytes": len(output),
        "metrics": _metrics(decoded),
        "output_tail": _tail(decoded),
    }
    print(f"{result['status'].upper():4} {spec['id']:<40} {duration:8.3f}s exit={proc.returncode}", flush=True)
    return result


def _write(root: Path, commands: list[dict[str, Any]]) -> Path:
    failures = [item["id"] for item in commands if item["status"] != "pass"]
    totals = {
        "commands": len(commands),
        "passed_commands": len(commands) - len(failures),
        "failed_commands": len(failures),
        "pytest_passed": sum(item["metrics"].get("pytest_passed", 0) for item in commands),
        "pytest_failed": sum(item["metrics"].get("pytest_failed", 0) for item in commands),
        "unittest_ran": sum(item["metrics"].get("unittest_ran", 0) for item in commands),
        "node_passed": sum(item["metrics"].get("node_passed", 0) for item in commands),
        "node_failed": sum(item["metrics"].get("node_failed", 0) for item in commands),
        "duration_seconds": round(sum(item["duration_seconds"] for item in commands), 3),
    }
    report = {
        "schema_version": "1.0.0",
        "source_freeze": "2026-08-05",
        "status": "pass" if not failures else "fail",
        "failed": len(failures),
        "failures": failures,
        "totals": totals,
        "commands": commands,
        "notes": [
            "Raw command logs are local QA intermediates; each output hash and tail is recorded here.",
            "Veridraft digest_ok=False denotes an unsigned self-authored bundle; review-public.txt records digest N/A, not a mismatch.",
            "The 10 blocked P3 claims remain patent-first HELD and are excluded from the 47-claim public capsule.",
        ],
    }
    target = root / OUTPUT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--refresh-metrics-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.refresh_metrics_only:
        report = json.loads((root / OUTPUT).read_text(encoding="utf-8"))
        commands = report["commands"]
        for item in commands:
            log = root / LOG_DIR / f"{item['id']}.log"
            if log.is_file():
                item["metrics"] = _metrics(log.read_text(encoding="utf-8", errors="replace"))
        _write(root, commands)
        return 0
    commands: list[dict[str, Any]] = []
    for spec in COMMANDS:
        commands.append(_run(root, spec))
        if commands[-1]["status"] == "fail":
            _write(root, commands)
            return 1

    # Provisional green report lets the rights-aware manifest materialize; both are
    # then validated as ordinary release artifacts before the final report is written.
    _write(root, commands)
    post = (
        {"id": "release-manifest-ready", "argv": ["python3", "research/sleep-time-compute/program/build_release_manifest.py", "--root", ".", "--require-ready"]},
        {"id": "release-entrypoint-links", "argv": ["python3", "research/sleep-time-compute/program/check_release_links.py", "--root", "."]},
        {"id": "cross-artifact-release-validation", "argv": ["python3", "research/sleep-time-compute/program/validate_release_artifacts.py", "--root", "."]},
    )
    for spec in post:
        commands.append(_run(root, spec))
        if commands[-1]["status"] == "fail":
            _write(root, commands)
            return 1
    _write(root, commands)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
