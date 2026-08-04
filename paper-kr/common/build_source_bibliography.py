#!/usr/bin/env python3
"""Generate BibLaTeX records from the frozen 73-source STC registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _bib(value: str) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("$", r"\$")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _key(source_id: str) -> str:
    return source_id.lower().replace("-", "")


def _version_label(version: object) -> str:
    """Return a compact human-readable version while registry hashes stay canonical."""

    if isinstance(version, dict):
        return str(version.get("version_id") or version.get("retrieved_at") or "")
    return str(version or "")


def build_bibliography(registry_path: Path, output_path: Path) -> None:
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    records = registry["sources"]
    lines = [
        "% Generated from research/sleep-time-compute/deep-research/SOURCE-REGISTRY.json.",
        "% Do not edit by hand; source IDs are the stable citation keys.",
        "",
    ]
    for record in records:
        source_id = record["source_id"]
        authors = record.get("authors") or []
        if authors:
            author = " and ".join(_bib(name) for name in authors)
        else:
            organization = _bib(record.get("organization") or "Unknown organization")
            author = "{{" + organization + "}}"
        published = str(record.get("published_at") or "")
        year = published[:4] if len(published) >= 4 else "n.d."
        fields = [
            ("author", author),
            ("title", _bib(record["title"])),
            ("year", year),
            ("url", _bib(record["canonical_url"])),
            ("urldate", _bib(record.get("accessed_at") or "2026-08-05")),
        ]
        if record.get("doi"):
            fields.append(("doi", _bib(record["doi"])))
        if record.get("arxiv_id"):
            fields.extend(
                [
                    ("eprint", _bib(record["arxiv_id"])),
                    ("archivePrefix", "arXiv"),
                ]
            )
        version = _version_label(record.get("version"))
        if version:
            fields.append(("version", _bib(version)))
        fields.append(
            (
                "note",
                _bib(
                    f"Frozen source {source_id}; "
                    f"type={record['source_type'].replace('_', ' ')}; "
                    f"accessed={record.get('accessed_at', '2026-08-05')}"
                ),
            )
        )
        lines.append(f"@misc{{{_key(source_id)},")
        for index, (name, value) in enumerate(fields):
            comma = "," if index < len(fields) - 1 else ""
            lines.append(f"  {name} = {{{value}}}{comma}")
        lines.extend(["}", ""])
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_bibliography(args.registry, args.output)
    print(f"source bibliography: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
