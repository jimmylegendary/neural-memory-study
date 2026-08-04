"""Validate the canonical evidence spine for the sleep-time compute program."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, NamedTuple

from jsonschema import Draft202012Validator, FormatChecker

SUPPORTED_CLAIM_TYPES = frozenset(
    {"direct_fact", "author_claim", "synthesis", "hypothesis", "scenario"}
)
REGISTRY_KEYS = frozenset({"schema_version", "sources", "claims", "figures"})
AUTHORING_MARKERS = re.compile(r"\b(?:TO" + r"DO|T" + r"BD|FIX" + r"ME)\b")
LATEX_REFERENCE_ERRORS = re.compile(
    r"(?:undefined references|Citation[^\n]*undefined)", re.IGNORECASE
)


class Diagnostic(NamedTuple):
    code: str
    path: str
    message: str


class ProgramValidationReport(NamedTuple):
    success: bool
    diagnostics: tuple[Diagnostic, ...]


def _read_json(path: Path, diagnostics: list[Diagnostic]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        diagnostics.append(
            Diagnostic(
                "registry-file-missing", str(path), "Required registry file is missing."
            )
        )
    except json.JSONDecodeError as exc:
        diagnostics.append(
            Diagnostic(
                "invalid-json",
                f"{path}:{exc.lineno}:{exc.colno}",
                exc.msg,
            )
        )
    return None


def _load_schema(name: str) -> dict[str, Any]:
    schema_path = Path(__file__).with_name("schema") / name
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _validate_schema(
    payload: Any,
    schema_name: str,
    path: Path,
    diagnostics: list[Diagnostic],
) -> None:
    validator = Draft202012Validator(
        _load_schema(schema_name), format_checker=FormatChecker()
    )
    for error in sorted(
        validator.iter_errors(payload), key=lambda item: list(item.path)
    ):
        location = "/".join(str(part) for part in error.absolute_path)
        diagnostics.append(
            Diagnostic(
                "schema-validation-error",
                f"{path}#{location}" if location else str(path),
                error.message,
            )
        )


def _validate_wrapper(
    payload: Any,
    path: Path,
    collection_key: str,
    diagnostics: list[Diagnostic],
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        diagnostics.append(
            Diagnostic(
                "schema-validation-error", str(path), "Registry must be an object."
            )
        )
        return []
    expected = {"schema_version", collection_key}
    extras = sorted(set(payload) - expected)
    if extras:
        diagnostics.append(
            Diagnostic(
                "schema-validation-error",
                str(path),
                f"Unexpected registry properties: {', '.join(extras)}",
            )
        )
    if payload.get("schema_version") != "1.0.0":
        diagnostics.append(
            Diagnostic(
                "schema-validation-error",
                f"{path}#schema_version",
                "schema_version must equal 1.0.0.",
            )
        )
    collection = payload.get(collection_key)
    if not isinstance(collection, list):
        diagnostics.append(
            Diagnostic(
                "schema-validation-error",
                f"{path}#{collection_key}",
                f"{collection_key} must be an array.",
            )
        )
        return []
    return [item for item in collection if isinstance(item, dict)]


def _duplicate_diagnostics(
    records: list[dict[str, Any]],
    id_key: str,
    code: str,
    path: Path,
) -> list[Diagnostic]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for record in records:
        value = record.get(id_key)
        if isinstance(value, str):
            if value in seen:
                duplicates.add(value)
            seen.add(value)
    return [
        Diagnostic(code, f"{path}#{id_key}={value}", f"Duplicate ID: {value}")
        for value in sorted(duplicates)
    ]


def _resolve_manifest(root: Path) -> tuple[Path, Path]:
    direct = root / "research-spine.json"
    if direct.exists():
        return direct, root
    nested = (
        root / "research" / "sleep-time-compute" / "program" / "research-spine.json"
    )
    if nested.exists():
        return nested, root
    return direct, root


def _registry_path(base: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        return base / "__invalid_registry_path__"
    candidate = Path(value)
    return candidate if candidate.is_absolute() else base / candidate


def validate_program(root: Path) -> ProgramValidationReport:
    """Validate one research spine and every registry it references."""

    root = Path(root)
    diagnostics: list[Diagnostic] = []
    manifest_path, base = _resolve_manifest(root)
    manifest = _read_json(manifest_path, diagnostics)
    if not isinstance(manifest, dict):
        return ProgramValidationReport(False, tuple(diagnostics))

    _validate_schema(manifest, "research-spine.schema.json", manifest_path, diagnostics)
    registry_refs = manifest.get("registries", {})
    if not isinstance(registry_refs, dict):
        registry_refs = {}

    source_path = _registry_path(base, registry_refs.get("sources"))
    claim_path = _registry_path(base, registry_refs.get("claims"))
    figure_path = _registry_path(base, registry_refs.get("figures"))
    translation_path = _registry_path(base, registry_refs.get("translations"))

    source_payload = _read_json(source_path, diagnostics)
    claim_payload = _read_json(claim_path, diagnostics)
    figure_payload = _read_json(figure_path, diagnostics)
    translation_payload = _read_json(translation_path, diagnostics)

    sources = (
        _validate_wrapper(source_payload, source_path, "sources", diagnostics)
        if source_payload is not None
        else []
    )
    claims = (
        _validate_wrapper(claim_payload, claim_path, "claims", diagnostics)
        if claim_payload is not None
        else []
    )
    figures = (
        _validate_wrapper(figure_payload, figure_path, "figures", diagnostics)
        if figure_payload is not None
        else []
    )

    for index, source in enumerate(sources):
        _validate_schema(
            source,
            "source-record.schema.json",
            Path(f"{source_path}#sources/{index}"),
            diagnostics,
        )
    for index, claim in enumerate(claims):
        _validate_schema(
            claim,
            "claim-record.schema.json",
            Path(f"{claim_path}#claims/{index}"),
            diagnostics,
        )
    for index, figure in enumerate(figures):
        _validate_schema(
            figure,
            "figure-record.schema.json",
            Path(f"{figure_path}#figures/{index}"),
            diagnostics,
        )
    if translation_payload is not None:
        _validate_schema(
            translation_payload,
            "translation-selection.schema.json",
            translation_path,
            diagnostics,
        )

    diagnostics.extend(
        _duplicate_diagnostics(sources, "source_id", "duplicate-source-id", source_path)
    )
    diagnostics.extend(
        _duplicate_diagnostics(claims, "claim_id", "duplicate-claim-id", claim_path)
    )
    diagnostics.extend(
        _duplicate_diagnostics(figures, "figure_id", "duplicate-figure-id", figure_path)
    )

    selected = []
    if isinstance(translation_payload, dict):
        candidate = translation_payload.get("selected")
        if isinstance(candidate, list):
            selected = [item for item in candidate if isinstance(item, dict)]
    diagnostics.extend(
        _duplicate_diagnostics(
            selected, "paper_id", "duplicate-translation-paper-id", translation_path
        )
    )
    selected_count = sum(item.get("status") == "selected" for item in selected)
    if not 12 <= selected_count <= 15:
        diagnostics.append(
            Diagnostic(
                "translation-count-out-of-range",
                str(translation_path),
                f"Selected translation count is {selected_count}; expected 12 through 15.",
            )
        )

    source_ids = {
        source["source_id"]
        for source in sources
        if isinstance(source.get("source_id"), str)
    }
    for source in sources:
        source_id = str(source.get("source_id", "unknown"))
        if source.get("status") != "frozen":
            diagnostics.append(
                Diagnostic(
                    "source-not-frozen",
                    f"{source_path}#{source_id}",
                    "Every source in a frozen research spine must be frozen.",
                )
            )
        if source.get("immutable") is False:
            version = source.get("version")
            required = {"version_id", "content_sha256", "retrieved_at"}
            if not isinstance(version, dict) or not required <= set(version):
                diagnostics.append(
                    Diagnostic(
                        "source-mutable-without-version",
                        f"{source_path}#{source_id}",
                        "Mutable sources require pinned version metadata and a content hash.",
                    )
                )

    for claim in claims:
        claim_id = str(claim.get("claim_id", "unknown"))
        if claim.get("claim_type") not in SUPPORTED_CLAIM_TYPES:
            diagnostics.append(
                Diagnostic(
                    "unsupported-claim-type",
                    f"{claim_path}#{claim_id}",
                    f"Unsupported claim type: {claim.get('claim_type')!r}",
                )
            )
        support = claim.get("support")
        if not isinstance(support, list):
            support = []
        for index, reference in enumerate(support):
            if not isinstance(reference, dict):
                continue
            if not str(reference.get("locator", "")).strip():
                diagnostics.append(
                    Diagnostic(
                        "claim-missing-locator",
                        f"{claim_path}#{claim_id}/support/{index}",
                        "Every claim support edge requires a page, line, section, or anchored locator.",
                    )
                )
            referenced_source = reference.get("source_id")
            if referenced_source not in source_ids:
                diagnostics.append(
                    Diagnostic(
                        "claim-source-not-found",
                        f"{claim_path}#{claim_id}/support/{index}",
                        f"Unknown source ID: {referenced_source}",
                    )
                )

    for figure in figures:
        figure_id = str(figure.get("figure_id", "unknown"))
        for source_id in figure.get("source_ids", []):
            if source_id not in source_ids:
                diagnostics.append(
                    Diagnostic(
                        "figure-source-not-found",
                        f"{figure_path}#{figure_id}",
                        f"Unknown source ID: {source_id}",
                    )
                )
        if figure.get("public") is True and figure.get("mode") == "direct_reuse":
            license_record = figure.get("license")
            license_ok = (
                isinstance(license_record, dict)
                and license_record.get("reuse_allowed") is True
                and license_record.get("review_status") == "reviewed"
                and bool(str(license_record.get("license_id", "")).strip())
                and bool(str(license_record.get("evidence_url", "")).strip())
                and bool(str(license_record.get("evidence_locator", "")).strip())
            )
            if not license_ok:
                diagnostics.append(
                    Diagnostic(
                        "figure-license-evidence-missing",
                        f"{figure_path}#{figure_id}",
                        "Public direct-reuse figures require reviewed reuse permission and license evidence.",
                    )
                )

    for item in selected:
        source_id = item.get("source_id")
        if source_id not in source_ids:
            diagnostics.append(
                Diagnostic(
                    "translation-source-not-found",
                    f"{translation_path}#{item.get('paper_id', 'unknown')}",
                    f"Unknown source ID: {source_id}",
                )
            )

    unresolved = manifest.get("unresolved_load_bearing_claims")
    if (
        manifest.get("status") == "frozen"
        and isinstance(unresolved, list)
        and unresolved
    ):
        diagnostics.append(
            Diagnostic(
                "frozen-spine-has-unresolved-load-bearing-claims",
                str(manifest_path),
                "A frozen spine cannot contain unresolved load-bearing claim IDs.",
            )
        )

    diagnostics = sorted(
        set(diagnostics), key=lambda item: (item.code, item.path, item.message)
    )
    return ProgramValidationReport(not diagnostics, tuple(diagnostics))


def _scan_authoring_markers(paths: list[Path]) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for base in paths:
        if not base.exists():
            continue
        candidates = [base] if base.is_file() else sorted(base.rglob("*"))
        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in {
                ".md",
                ".tex",
                ".json",
            }:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if AUTHORING_MARKERS.search(line):
                    diagnostics.append(
                        Diagnostic(
                            "authoring-marker-found",
                            f"{path}:{line_number}",
                            "Unresolved authoring marker found.",
                        )
                    )
                if LATEX_REFERENCE_ERRORS.search(line):
                    diagnostics.append(
                        Diagnostic(
                            "latex-reference-error-found",
                            f"{path}:{line_number}",
                            "LaTeX reference error found.",
                        )
                    )
    return diagnostics


def _default_repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", nargs="?", type=Path, default=_default_repository_root()
    )
    parser.add_argument("--phase", choices=["pre-research", "deep-research"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--reject-authoring-markers", action="store_true")
    parser.add_argument("--reject-latex-reference-errors", action="store_true")
    args = parser.parse_args(argv)

    diagnostics: list[Diagnostic] = []
    manifest, _ = _resolve_manifest(args.root)
    if manifest.exists():
        diagnostics.extend(validate_program(args.root).diagnostics)
    elif not args.phase:
        diagnostics.append(
            Diagnostic(
                "registry-file-missing", str(manifest), "Research spine is missing."
            )
        )

    if args.reject_authoring_markers or args.reject_latex_reference_errors:
        scan_paths: list[Path] = []
        if args.phase:
            scan_paths.append(
                args.root / "research" / "sleep-time-compute" / args.phase
            )
        elif args.all:
            scan_paths.extend(
                [
                    args.root / "paper-kr",
                    args.root / "translations-kr" / "stc-core",
                    args.root / "easy" / "sleep-time-compute",
                    args.root / "presentation" / "sleep-time-compute-deep-study",
                    args.root / "research" / "sleep-time-compute" / "pre-research",
                    args.root / "research" / "sleep-time-compute" / "deep-research",
                ]
            )
        diagnostics.extend(_scan_authoring_markers(scan_paths))

    for diagnostic in sorted(set(diagnostics)):
        print(f"{diagnostic.code}\t{diagnostic.path}\t{diagnostic.message}")
    return 0 if not diagnostics else 1


if __name__ == "__main__":
    raise SystemExit(main())
