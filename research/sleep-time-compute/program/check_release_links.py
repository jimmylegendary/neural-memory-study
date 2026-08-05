#!/usr/bin/env python3
"""Validate repository-local Markdown links in STC release entry points."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote


DEFAULT_FILES = (
    "README.md",
    "research/sleep-time-compute/README.md",
    "research/sleep-time-compute/DELIVERABLES.md",
    "translations-kr/README.md",
    "easy/README.md",
    "easy/sleep-time-compute/README.md",
    "presentation/sleep-time-compute-deep/README.md",
)
LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def broken_links(root: Path, files: tuple[str, ...] = DEFAULT_FILES) -> list[str]:
    root = root.resolve()
    failures: list[str] = []
    for name in files:
        document = root / name
        if not document.is_file():
            failures.append(f"{name}: document missing")
            continue
        for line_number, line in enumerate(document.read_text(encoding="utf-8").splitlines(), start=1):
            for match in LINK.finditer(line):
                raw = match.group(1).strip()
                target = raw.split(maxsplit=1)[0].strip("<>")
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                relative = unquote(target.split("#", 1)[0])
                if not relative:
                    continue
                resolved = (document.parent / relative).resolve()
                try:
                    resolved.relative_to(root)
                except ValueError:
                    failures.append(f"{name}:{line_number}: target escapes repository: {target}")
                    continue
                if not resolved.exists():
                    failures.append(f"{name}:{line_number}: missing target: {target}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()
    files = tuple(args.files) if args.files else DEFAULT_FILES
    failures = broken_links(args.root, files)
    if failures:
        print("\n".join(failures))
        print(f"FAIL: {len(failures)} broken local link(s)")
        return 1
    print(f"PASS: {len(files)} Markdown entry points, 0 broken local links")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
