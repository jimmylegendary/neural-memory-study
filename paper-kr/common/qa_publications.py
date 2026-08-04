#!/usr/bin/env python3
"""Structural and rendered-PDF QA for the STC publication family."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, NamedTuple


BGREF = re.compile(r"\\bgref\{([^}]+)\}")
CLAIM_REF = re.compile(r"\\claim\{(STC-C\d{3,4})\}")
FIGURE_REF = re.compile(r"\\stcfigure\{(STC-F\d{3})\}")
LABEL = re.compile(r"\\label\{([^}]+)\}")
AUTHORING_MARKER = re.compile(r"\b(?:TO" + r"DO|T" + r"BD|FIX" + r"ME)\b")
LATEX_UNRESOLVED = re.compile(
    r"(?:undefined references|Citation [`'][^`']+['`] on page .* undefined)",
    re.IGNORECASE,
)


class CrossRefReport(NamedTuple):
    referenced_concepts: tuple[str, ...]
    unresolved_concepts: tuple[str, ...]


class SourceTreeReport(NamedTuple):
    referenced_concepts: tuple[str, ...]
    unresolved_concepts: tuple[str, ...]
    referenced_claims: tuple[str, ...]
    unknown_claims: tuple[str, ...]
    referenced_figures: tuple[str, ...]
    unknown_figures: tuple[str, ...]
    duplicate_labels: tuple[str, ...]
    authoring_markers: tuple[str, ...]

    @property
    def success(self) -> bool:
        return not any(
            (
                self.unresolved_concepts,
                self.unknown_claims,
                self.unknown_figures,
                self.duplicate_labels,
                self.authoring_markers,
            )
        )


class PDFQAReport(NamedTuple):
    path: str
    kind: str
    pages: int
    diagnostics: tuple[str, ...]
    fonts: tuple[str, ...]

    @property
    def success(self) -> bool:
        return not self.diagnostics


def _ordered_unique(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _concept_map(registry: dict[str, Any]) -> dict[str, str]:
    concepts = registry.get("concepts") if isinstance(registry, dict) else None
    if isinstance(concepts, list):
        return {
            item["concept_id"]: item["label"]
            for item in concepts
            if isinstance(item, dict)
            and isinstance(item.get("concept_id"), str)
            and isinstance(item.get("label"), str)
        }
    return {
        key: value
        for key, value in registry.items()
        if isinstance(key, str) and isinstance(value, str)
    }


def validate_cross_document_refs(
    study_source: str, registry: dict[str, Any]
) -> CrossRefReport:
    referenced = _ordered_unique(BGREF.findall(study_source))
    known = _concept_map(registry)
    unresolved = tuple(concept for concept in referenced if concept not in known)
    return CrossRefReport(referenced, unresolved)


def validate_figure(record: dict[str, Any], *, audience: str) -> tuple[str, ...]:
    diagnostics: list[str] = []
    license_record = record.get("license", {})
    if not isinstance(license_record, dict):
        license_record = {}
    if audience == "public" and record.get("public") is False:
        diagnostics.append("figure-not-cleared-for-public")
    if audience == "public" and record.get("mode") == "direct_reuse":
        if not license_record.get("reuse_allowed"):
            diagnostics.append("public-reuse-not-authorized")
        if license_record.get("review_status") != "reviewed":
            diagnostics.append("public-reuse-license-not-reviewed")
    if audience == "public" and record.get("mode") == "redrawn":
        if not license_record.get("reuse_allowed"):
            diagnostics.append("public-redraw-not-authorized")
    return tuple(diagnostics)


def _tex_sources(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob("*.tex") if path.is_file())


def validate_source_tree(
    root: Path,
    *,
    concept_registry: dict[str, Any],
    claim_map: dict[str, Any],
    figure_ledger: dict[str, Any],
) -> SourceTreeReport:
    sources = _tex_sources(Path(root))
    text_by_path = {
        path: path.read_text(encoding="utf-8", errors="replace") for path in sources
    }
    combined = "\n".join(text_by_path.values())
    cross_refs = validate_cross_document_refs(combined, concept_registry)
    claims = _ordered_unique(CLAIM_REF.findall(combined))
    figures = _ordered_unique(FIGURE_REF.findall(combined))
    known_claims = {
        item.get("claim_id")
        for item in claim_map.get("claims", [])
        if isinstance(item, dict)
    }
    known_figures = {
        item.get("figure_id")
        for item in figure_ledger.get("figures", [])
        if isinstance(item, dict)
    }
    labels = LABEL.findall(combined)
    label_counts = {label: labels.count(label) for label in set(labels)}
    markers = tuple(
        str(path)
        for path, source in text_by_path.items()
        if AUTHORING_MARKER.search(source)
    )
    return SourceTreeReport(
        cross_refs.referenced_concepts,
        cross_refs.unresolved_concepts,
        claims,
        tuple(claim for claim in claims if claim not in known_claims),
        figures,
        tuple(figure for figure in figures if figure not in known_figures),
        tuple(sorted(label for label, count in label_counts.items() if count > 1)),
        markers,
    )


def validate_concept_registry(path: Path) -> tuple[str, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    diagnostics: list[str] = []
    if raw.get("schema_version") != "1.0.0":
        diagnostics.append("invalid-schema-version")
    concepts = raw.get("concepts")
    if not isinstance(concepts, list):
        return tuple(diagnostics + ["concepts-not-array"])
    required = {
        "concept_id",
        "label",
        "definition",
        "prerequisites",
        "first_study_section",
        "glossary_term",
    }
    ids = [item.get("concept_id") for item in concepts if isinstance(item, dict)]
    known = {item for item in ids if isinstance(item, str)}
    if len(known) != len(ids):
        diagnostics.append("duplicate-or-invalid-concept-id")
    for item in concepts:
        if not isinstance(item, dict):
            diagnostics.append("concept-not-object")
            continue
        concept_id = str(item.get("concept_id", "unknown"))
        for missing in sorted(required - set(item)):
            diagnostics.append(f"missing-field:{concept_id}:{missing}")
        prerequisites = item.get("prerequisites", [])
        if not isinstance(prerequisites, list):
            diagnostics.append(f"prerequisites-not-array:{concept_id}")
            continue
        for prerequisite in prerequisites:
            if prerequisite not in known:
                diagnostics.append(
                    f"missing-prerequisite:{concept_id}:{prerequisite}"
                )
    return tuple(diagnostics)


def _run_text(command: list[str]) -> str:
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.stdout


def qa_pdf(
    path: Path,
    *,
    kind: str,
    min_pages: int | None = None,
    max_pages: int | None = None,
) -> PDFQAReport:
    path = Path(path)
    diagnostics: list[str] = []
    if not path.exists():
        return PDFQAReport(str(path), kind, 0, ("pdf-missing",), ())
    info = _run_text(["pdfinfo", str(path)])
    match = re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)
    pages = int(match.group(1)) if match else 0
    if pages == 0:
        diagnostics.append("pdf-has-no-pages")
    if min_pages is not None and pages < min_pages:
        diagnostics.append(f"page-count-below-minimum:{pages}<{min_pages}")
    if max_pages is not None and pages > max_pages:
        diagnostics.append(f"page-count-above-maximum:{pages}>{max_pages}")
    text = _run_text(["pdftotext", str(path), "-"])
    if not text.strip():
        diagnostics.append("pdf-text-empty")
    if "�" in text:
        diagnostics.append("replacement-glyph-in-text")
    if AUTHORING_MARKER.search(text):
        diagnostics.append("authoring-marker-in-pdf")
    fonts: list[str] = []
    try:
        font_output = _run_text(["pdffonts", str(path)])
        for line in font_output.splitlines()[2:]:
            if line.strip():
                fonts.append(line.split()[0])
        if any("Noto" not in font for font in fonts if "CJK" in font):
            diagnostics.append("unexpected-cjk-font")
    except (FileNotFoundError, subprocess.SubprocessError):
        diagnostics.append("pdffonts-unavailable")
    return PDFQAReport(
        str(path), kind, pages, tuple(diagnostics), tuple(sorted(set(fonts)))
    )


def _load_optional(path: str | None, default: dict[str, Any]) -> dict[str, Any]:
    if not path:
        return default
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--min-pages", type=int)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--concept-registry")
    parser.add_argument("--claim-map")
    parser.add_argument("--figure-ledger")
    parser.add_argument("--require-all-concepts", action="store_true")
    parser.add_argument("--allow-incomplete-sections")
    args = parser.parse_args()

    diagnostics: list[str] = []
    source_report: SourceTreeReport | None = None
    concepts = _load_optional(args.concept_registry, {"concepts": []})
    claims = _load_optional(args.claim_map, {"claims": []})
    figures = _load_optional(args.figure_ledger, {"figures": []})
    if args.source_root:
        source_report = validate_source_tree(
            args.source_root,
            concept_registry=concepts,
            claim_map=claims,
            figure_ledger=figures,
        )
        if not source_report.success:
            diagnostics.extend(
                [f"unresolved-concept:{item}" for item in source_report.unresolved_concepts]
                + [f"unknown-claim:{item}" for item in source_report.unknown_claims]
                + [f"unknown-figure:{item}" for item in source_report.unknown_figures]
                + [f"duplicate-label:{item}" for item in source_report.duplicate_labels]
                + [f"authoring-marker:{item}" for item in source_report.authoring_markers]
            )
        if args.require_all_concepts:
            known = set(_concept_map(concepts))
            referenced = set(source_report.referenced_concepts)
            for missing in sorted(known - referenced):
                diagnostics.append(f"concept-not-linked:{missing}")
    pdf_report = qa_pdf(
        args.pdf,
        kind=args.kind,
        min_pages=args.min_pages,
        max_pages=args.max_pages,
    )
    diagnostics.extend(pdf_report.diagnostics)
    payload = {
        "success": not diagnostics,
        "pdf": pdf_report._asdict(),
        "source": source_report._asdict() if source_report else None,
        "diagnostics": diagnostics,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not diagnostics else 1


if __name__ == "__main__":
    raise SystemExit(main())
