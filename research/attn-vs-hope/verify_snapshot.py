#!/usr/bin/env python3
"""Verify the archived native Attn-vs-HOPE Google Sheet export.

This standard-library-only verifier reads the XLSX as OOXML and never opens it
for writing.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import posixpath
import re
import sys
import unittest
import zipfile
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET


HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "EXPORT-MANIFEST.json"
WORKBOOK_PATH = HERE / "Attn-vs-HOPE.xlsx"

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
RID_ATTRIBUTE = f"{{{OFFICE_REL_NS}}}id"
M = f"{{{MAIN_NS}}}"

EXPECTED_SHA256 = "0c04b122a214b9e2543abf6925260f307ddbdda7bc9cedca7e3d1f92bb7b8267"
EXPECTED_H1_NOTE = (
    "Self-modifying Titans auxiliary memory의 prefill chunk 크기. 현재 2048은 "
    "분석 시나리오 값이며, 논문은 main memory와 기타 memory에 서로 다른 "
    "chunk를 사용할 수 있다고만 명시한다."
)
STALE_H1_TEXT = "현재 64"

EXPECTED_DEFINED_NAMES = {
    "B": "'시트1'!$B$2",
    "CMS_1_r": "'시트1'!$H$3",
    "CMS_C1": "'시트1'!$H$6",
    "CMS_C2": "'시트1'!$K$6",
    "CMS_C3": "'시트1'!$B$7",
    "CMS_E1": "'시트1'!$D$7",
    "CMS_E2": "'시트1'!$F$7",
    "CMS_E3": "'시트1'!$O$2",
    "CMS_n_r": "'시트1'!$H$4",
    "CONV_WIN": "'시트1'!$K$7",
    "C_mem": "'시트1'!$F$6",
    "Chunk": "'시트1'!$H$1",
    "ChunkP": "'시트1'!$F$1",
    "D": "'시트1'!$D$2",
    "DECODE_SPLITS": "'시트1'!$H$7",
    "ED": "'시트1'!$D$3",
    "ELEM_BYTES": "'시트1'!$B$5",
    "FA_KV_HBM_MULT": "'시트1'!$D$5",
    "FLOPS": "'시트1'!$K$3",
    "GIGA": "'시트1'!$K$1",
    "HBM_BW": "'시트1'!$K$2",
    "HOPE_CHUNKS": "'시트1'!$M$3",
    "ISL": "'시트1'!$B$3",
    "KV_SPLIT": "'시트1'!$F$5",
    "LRD": "'시트1'!$D$4",
    "MEM_CHUNKS": "'시트1'!$O$3",
    "MEM_HIDDEN": "'시트1'!$M$2",
    "OSL": "'시트1'!$B$4",
    "PARTIAL_BYTES": "'시트1'!$K$5",
    "PETA": "'시트1'!$O$1",
    "TERA": "'시트1'!$M$1",
    "bit": "'시트1'!$F$4",
    "head_dim": "'시트1'!$H$5",
    "nAux": "'시트1'!$B$6",
    "nCMS_Lev": "'시트1'!$H$2",
    "nE": "'시트1'!$F$2",
    "nMutable": "'시트1'!$D$6",
    "topK": "'시트1'!$F$3",
}


def _cms_bound(row: int, level: int) -> str:
    return (
        f'IF(nCMS_Lev>={level},IF(N{row}>=M{row},"compute",'
        '"memory_bw"),"inactive")'
    )


def _ungated_bound(row: int) -> str:
    return f'IF(N{row}>=M{row},"compute","memory_bw")'


EXPECTED_BOUND_FORMULAS = {
    **{f"P{row}": _cms_bound(row, 1) for row in range(24, 27)},
    **{f"P{row}": _cms_bound(row, 2) for row in range(27, 30)},
    **{f"P{row}": _cms_bound(row, 3) for row in range(30, 33)},
    "P33": _cms_bound(33, 1),
    **{f"P{row}": _ungated_bound(row) for row in range(41, 47)},
    **{f"P{row}": _cms_bound(row, 1) for row in range(65, 68)},
    **{f"P{row}": _cms_bound(row, 2) for row in range(68, 71)},
    **{f"P{row}": _cms_bound(row, 3) for row in range(71, 74)},
    "P74": _cms_bound(74, 1),
    **{f"P{row}": _ungated_bound(row) for row in range(82, 88)},
}

EXPECTED_KEY_OUTPUTS = {
    "full_attention_ttft": {"cell": "O18", "value_ms": 3204.916},
    "hope_forward_ttft": {"cell": "O46", "value_ms": 552.123},
    "hope_including_online_ttft": {"cell": "O48", "value_ms": 6757.768},
    "full_attention_itl": {"cell": "O59", "value_ms": 10.888},
    "hope_forward_itl": {"cell": "O87", "value_ms": 0.604},
    "hope_including_online_itl": {"cell": "O89", "value_ms": 18.890},
}

EXPECTED_MANIFEST = {
    "schema_version": 1,
    "source": {
        "kind": "Google Sheets",
        "id": "1BZLzsGgdE43GXAMtuhrg7btPRq94crQWjo5mb0VWNWk",
        "url": "https://docs.google.com/spreadsheets/d/1BZLzsGgdE43GXAMtuhrg7btPRq94crQWjo5mb0VWNWk",
        "updated_at": "2026-07-27T01:45:55.568Z",
    },
    "artifact": {
        "repository_path": "research/attn-vs-hope/Attn-vs-HOPE.xlsx",
        "exported_at": "2026-07-27T10:46:51.477084070+09:00",
        "size_bytes": 29321,
        "sha256": EXPECTED_SHA256,
    },
    "portable_xlsx_checks": {
        "sheet_count": 1,
        "visible_sheets": ["시트1"],
        "sheet": "시트1",
        "used_range": "A1:P89",
        "defined_name_count": 38,
        "formula_cell_count": 872,
        "note_comment_count": 117,
        "korean_note_comment_count": 117,
        "cached_formula_error_count": 0,
        "external_workbook_link_count": 0,
    },
    "live_sheet_checks": {
        "source_updated_at": "2026-07-27T01:45:55.568Z",
        "formula_count": 872,
        "effective_formula_error_count": 0,
    },
    "key_outputs_ms": EXPECTED_KEY_OUTPUTS,
    "corrected_cells": [
        {"range": "H1", "contract": "note says current 2048 and not stale current 64"},
        {"range": "P24:P33", "contract": "CMS-level bound gates"},
        {"range": "P41:P46", "contract": "Titans/update bound independent of CMS"},
        {"range": "P65:P74", "contract": "CMS-level bound gates"},
        {"range": "P82:P87", "contract": "Titans/update bound independent of CMS"},
    ],
    "artifact_roles": {
        "native_reference": {
            "path": "research/attn-vs-hope/Attn-vs-HOPE.xlsx",
            "purpose": "user-reviewed operational model",
        },
        "generated_companion": {
            "path": "experiments/E5-attn-vs-hope/sheet/Attn-vs-HOPE.xlsx",
            "purpose": "reproducible analytical companion",
        },
    },
}

CELL_RE = re.compile(r"^([A-Z]{1,3})([1-9][0-9]*)$")
FORMULA_CELL_RE = re.compile(
    r"(?<![A-Za-z0-9_.])(?P<col_abs>\$?)(?P<col>[A-Z]{1,3})"
    r"(?P<row_abs>\$?)(?P<row>[1-9][0-9]*)(?![A-Za-z0-9_])"
)
FORMULA_STRING_RE = re.compile(r'"(?:""|[^"])*"')
HANGUL_RE = re.compile(r"[가-힣]")
ERROR_VALUES = {
    "#NULL!", "#DIV/0!", "#VALUE!", "#REF!", "#NAME?", "#NUM!",
    "#N/A", "#GETTING_DATA", "#SPILL!", "#CALC!", "#FIELD!",
    "#BLOCKED!", "#UNKNOWN!",
}


class SnapshotFormatError(ValueError):
    """The OOXML package cannot be inspected safely."""


def _differences(expected: object, actual: object, path: str = "$") -> list[str]:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected object, found {type(actual).__name__}"]
        issues: list[str] = []
        expected_keys, actual_keys = set(expected), set(actual)
        issues.extend(f"{path}.{key}: missing" for key in sorted(expected_keys - actual_keys))
        issues.extend(f"{path}.{key}: unexpected" for key in sorted(actual_keys - expected_keys))
        for key in sorted(expected_keys & actual_keys):
            issues.extend(_differences(expected[key], actual[key], f"{path}.{key}"))
        return issues
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [f"{path}: expected array, found {type(actual).__name__}"]
        issues = []
        if len(expected) != len(actual):
            issues.append(f"{path}: expected {len(expected)} items, found {len(actual)}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            issues.extend(_differences(left, right, f"{path}[{index}]"))
        return issues
    return [] if expected == actual else [f"{path}: expected {expected!r}, found {actual!r}"]


def validate_manifest_data(manifest: dict[str, object]) -> list[str]:
    """Return all differences from the immutable manifest contract."""

    return _differences(EXPECTED_MANIFEST, manifest)


def _parse_xml(data: bytes, label: str) -> ET.Element:
    try:
        return ET.fromstring(data)
    except ET.ParseError as exc:
        raise SnapshotFormatError(f"{label}: invalid XML: {exc}") from exc


def external_link_issues(
    workbook_relationships: bytes,
    package_names: Iterable[str] = (),
) -> list[str]:
    """Return external-workbook relationship and OOXML-part violations."""

    try:
        root = _parse_xml(workbook_relationships, "workbook relationships")
    except SnapshotFormatError as exc:
        return [str(exc)]
    issues: list[str] = []
    for relationship in root.findall(f"{{{PACKAGE_REL_NS}}}Relationship"):
        rel_type = relationship.attrib.get("Type", "")
        target = relationship.attrib.get("Target", "").replace("\\", "/")
        if rel_type.rsplit("/", 1)[-1] == "externalLink" or "externalLinks/" in target:
            issues.append(
                f"external workbook relationship {relationship.attrib.get('Id', '<missing-id>')} -> {target}"
            )
    for name in package_names:
        normalized = name.lstrip("/").replace("\\", "/")
        if normalized.startswith("xl/externalLinks/"):
            issues.append(f"external workbook package part {normalized}")
    return issues


def _rels_path(source_part: str) -> str:
    return posixpath.join(
        posixpath.dirname(source_part), "_rels", posixpath.basename(source_part) + ".rels"
    )


def _resolve_target(source_part: str, target: str) -> str:
    target = target.replace("\\", "/")
    if target.startswith("/"):
        return posixpath.normpath(target.lstrip("/"))
    return posixpath.normpath(posixpath.join(posixpath.dirname(source_part), target))


def _relationship_map(root: ET.Element) -> dict[str, ET.Element]:
    return {
        rel.attrib["Id"]: rel
        for rel in root.findall(f"{{{PACKAGE_REL_NS}}}Relationship")
        if "Id" in rel.attrib
    }


def _column_number(column: str) -> int:
    number = 0
    for char in column:
        number = number * 26 + ord(char) - ord("A") + 1
    return number


def _column_name(number: int) -> str:
    if number < 1:
        raise SnapshotFormatError(f"invalid column number {number}")
    chars: list[str] = []
    while number:
        number, remainder = divmod(number - 1, 26)
        chars.append(chr(ord("A") + remainder))
    return "".join(reversed(chars))


def _coordinate_parts(coordinate: str) -> tuple[int, int]:
    match = CELL_RE.fullmatch(coordinate)
    if match is None:
        raise SnapshotFormatError(f"invalid cell coordinate {coordinate!r}")
    return _column_number(match.group(1)), int(match.group(2))


def _used_range(cells: Iterable[ET.Element]) -> str:
    coordinates = [
        _coordinate_parts(cell.attrib["r"]) for cell in cells if "r" in cell.attrib
    ]
    if not coordinates:
        raise SnapshotFormatError("worksheet has no serialized cells")
    columns = [column for column, _ in coordinates]
    rows = [row for _, row in coordinates]
    first = f"{_column_name(min(columns))}{min(rows)}"
    last = f"{_column_name(max(columns))}{max(rows)}"
    return first if first == last else f"{first}:{last}"


def _coordinate_in_range(coordinate: str, reference: str) -> bool:
    first, last = reference.split(":", 1) if ":" in reference else (reference, reference)
    column, row = _coordinate_parts(coordinate)
    first_column, first_row = _coordinate_parts(first)
    last_column, last_row = _coordinate_parts(last)
    return (
        min(first_column, last_column) <= column <= max(first_column, last_column)
        and min(first_row, last_row) <= row <= max(first_row, last_row)
    )


def _translate_formula_segment(segment: str, column_delta: int, row_delta: int) -> str:
    def replace(match: re.Match[str]) -> str:
        column = _column_number(match.group("col"))
        row = int(match.group("row"))
        if not match.group("col_abs"):
            column += column_delta
        if not match.group("row_abs"):
            row += row_delta
        if row < 1:
            raise SnapshotFormatError("shared formula translated below row 1")
        return (
            f"{match.group('col_abs')}{_column_name(column)}"
            f"{match.group('row_abs')}{row}"
        )

    return FORMULA_CELL_RE.sub(replace, segment)


def _translate_shared_formula(formula: str, source: str, target: str) -> str:
    source_column, source_row = _coordinate_parts(source)
    target_column, target_row = _coordinate_parts(target)
    column_delta, row_delta = target_column - source_column, target_row - source_row
    translated: list[str] = []
    cursor = 0
    for match in FORMULA_STRING_RE.finditer(formula):
        translated.append(
            _translate_formula_segment(formula[cursor:match.start()], column_delta, row_delta)
        )
        translated.append(match.group(0))
        cursor = match.end()
    translated.append(_translate_formula_segment(formula[cursor:], column_delta, row_delta))
    return "".join(translated)


def _effective_formulas(sheet_root: ET.Element) -> tuple[dict[str, str], list[str]]:
    cells = sheet_root.findall(f".//{M}c")
    anchors: dict[str, tuple[str, str, str | None]] = {}
    for cell in cells:
        formula = cell.find(f"{M}f")
        if formula is not None and formula.attrib.get("t") == "shared" and formula.text:
            shared_index = formula.attrib.get("si")
            if shared_index is not None:
                anchors[shared_index] = (
                    cell.attrib.get("r", ""), formula.text, formula.attrib.get("ref")
                )
    effective: dict[str, str] = {}
    issues: list[str] = []
    for cell in cells:
        coordinate = cell.attrib.get("r")
        formula = cell.find(f"{M}f")
        if coordinate is None or formula is None:
            continue
        if formula.text:
            effective[coordinate] = formula.text
            continue
        shared_index = formula.attrib.get("si")
        if formula.attrib.get("t") != "shared" or shared_index not in anchors:
            issues.append(f"{coordinate}: unresolved formula")
            continue
        anchor_coordinate, anchor_formula, shared_range = anchors[shared_index]
        if shared_range and not _coordinate_in_range(coordinate, shared_range):
            issues.append(f"{coordinate}: outside shared formula range {shared_range}")
            continue
        effective[coordinate] = _translate_shared_formula(
            anchor_formula, anchor_coordinate, coordinate
        )
    return effective, issues


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SnapshotFormatError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise SnapshotFormatError(f"{path.name}: root must be an object")
    return data


def _require_part(archive: zipfile.ZipFile, name: str) -> bytes:
    try:
        return archive.read(name)
    except KeyError as exc:
        raise SnapshotFormatError(f"missing OOXML part {name}") from exc


def _inspect_workbook(path: Path) -> tuple[dict[str, object], list[str]]:
    issues: list[str] = []
    summary: dict[str, object] = {}
    try:
        archive = zipfile.ZipFile(path, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise SnapshotFormatError(f"cannot open workbook ZIP: {exc}") from exc

    with archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            issues.append(f"ZIP CRC failure in {bad_member}")
        package_names = archive.namelist()
        workbook_part = "xl/workbook.xml"
        workbook_root = _parse_xml(_require_part(archive, workbook_part), workbook_part)
        workbook_rels_part = _rels_path(workbook_part)
        workbook_rels_bytes = _require_part(archive, workbook_rels_part)
        workbook_rels_root = _parse_xml(workbook_rels_bytes, workbook_rels_part)
        workbook_relationships = _relationship_map(workbook_rels_root)

        external: list[str] = []
        for relationship_part in sorted(name for name in package_names if name.endswith(".rels")):
            external.extend(
                f"{relationship_part}: {issue}"
                for issue in external_link_issues(archive.read(relationship_part))
            )
        external.extend(
            issue
            for issue in external_link_issues(workbook_rels_bytes, package_names)
            if issue.startswith("external workbook package part")
        )
        external = sorted(set(external))
        issues.extend(external)
        summary["external_workbook_link_count"] = len(external)

        sheet_nodes = workbook_root.findall(f"./{M}sheets/{M}sheet")
        visible_names = [
            node.attrib.get("name", "")
            for node in sheet_nodes
            if node.attrib.get("state", "visible") == "visible"
        ]
        summary["sheet_count"] = len(sheet_nodes)
        summary["visible_sheets"] = visible_names
        if len(sheet_nodes) != 1:
            issues.append(f"expected 1 sheet, found {len(sheet_nodes)}")
        if visible_names != ["시트1"]:
            issues.append(f"expected visible sheets ['시트1'], found {visible_names!r}")
        if not sheet_nodes:
            raise SnapshotFormatError("workbook has no worksheet")

        sheet_rid = sheet_nodes[0].attrib.get(RID_ATTRIBUTE)
        if sheet_rid not in workbook_relationships:
            raise SnapshotFormatError(f"worksheet relationship {sheet_rid!r} is missing")
        sheet_relationship = workbook_relationships[sheet_rid]
        if not sheet_relationship.attrib.get("Type", "").endswith("/worksheet"):
            raise SnapshotFormatError(f"{sheet_rid}: relationship is not a worksheet")
        sheet_part = _resolve_target(
            workbook_part, sheet_relationship.attrib.get("Target", "")
        )
        sheet_root = _parse_xml(_require_part(archive, sheet_part), sheet_part)
        cells = sheet_root.findall(f".//{M}c")
        cells_by_coordinate = {
            cell.attrib["r"]: cell for cell in cells if "r" in cell.attrib
        }

        used_range = _used_range(cells)
        summary["used_range"] = used_range
        if used_range != "A1:P89":
            issues.append(f"expected used range A1:P89, found {used_range}")

        defined_name_nodes = workbook_root.findall(f"./{M}definedNames/{M}definedName")
        defined_names = {
            node.attrib.get("name", ""): node.text or "" for node in defined_name_nodes
        }
        summary["defined_name_count"] = len(defined_name_nodes)
        if len(defined_name_nodes) != 38:
            issues.append(f"expected 38 defined names, found {len(defined_name_nodes)}")
        if len(defined_name_nodes) != len(defined_names):
            issues.append("defined names contain duplicate or missing names")
        missing = sorted(set(EXPECTED_DEFINED_NAMES) - set(defined_names))
        unexpected = sorted(set(defined_names) - set(EXPECTED_DEFINED_NAMES))
        if missing:
            issues.append(f"missing named ranges: {', '.join(missing)}")
        if unexpected:
            issues.append(f"unexpected named ranges: {', '.join(unexpected)}")
        for name in sorted(set(EXPECTED_DEFINED_NAMES) & set(defined_names)):
            if defined_names[name] != EXPECTED_DEFINED_NAMES[name]:
                issues.append(
                    f"defined name {name}: expected {EXPECTED_DEFINED_NAMES[name]!r}, "
                    f"found {defined_names[name]!r}"
                )

        formula_cells = [cell for cell in cells if cell.find(f"{M}f") is not None]
        summary["formula_cell_count"] = len(formula_cells)
        if len(formula_cells) != 872:
            issues.append(f"expected 872 formula cells, found {len(formula_cells)}")
        cached_errors: list[str] = []
        for cell in formula_cells:
            value = cell.find(f"{M}v")
            value_text = "" if value is None or value.text is None else value.text.strip()
            if cell.attrib.get("t") == "e" or value_text in ERROR_VALUES:
                cached_errors.append(
                    f"{cell.attrib.get('r', '<unknown>')}={value_text or '<empty>'}"
                )
        summary["cached_formula_error_count"] = len(cached_errors)
        if cached_errors:
            issues.append("cached formula errors: " + ", ".join(cached_errors[:20]))

        effective_formulas, formula_issues = _effective_formulas(sheet_root)
        issues.extend(formula_issues)
        for coordinate, expected_formula in EXPECTED_BOUND_FORMULAS.items():
            actual_formula = effective_formulas.get(coordinate)
            if actual_formula != expected_formula:
                issues.append(
                    f"{coordinate}: expected formula {expected_formula!r}, found {actual_formula!r}"
                )

        for output_name, output in EXPECTED_KEY_OUTPUTS.items():
            coordinate = str(output["cell"])
            cell = cells_by_coordinate.get(coordinate)
            value = None if cell is None else cell.find(f"{M}v")
            try:
                actual_value = float(value.text) if value is not None else math.nan
            except (TypeError, ValueError):
                actual_value = math.nan
            expected_value = float(output["value_ms"])
            if not math.isfinite(actual_value) or round(actual_value, 3) != expected_value:
                issues.append(
                    f"{coordinate} ({output_name}): expected {expected_value:.3f} ms, "
                    f"found {actual_value!r}"
                )

        sheet_rels_part = _rels_path(sheet_part)
        sheet_rels_root = _parse_xml(_require_part(archive, sheet_rels_part), sheet_rels_part)
        comment_parts = [
            _resolve_target(sheet_part, rel.attrib.get("Target", ""))
            for rel in sheet_rels_root.findall(f"{{{PACKAGE_REL_NS}}}Relationship")
            if rel.attrib.get("Type", "").endswith("/comments")
        ]
        if len(comment_parts) != 1:
            issues.append(f"expected 1 worksheet comments part, found {len(comment_parts)}")
        comments: list[ET.Element] = []
        for comment_part in comment_parts:
            comment_root = _parse_xml(_require_part(archive, comment_part), comment_part)
            comments.extend(comment_root.findall(f".//{M}comment"))
        comment_texts = {
            comment.attrib.get("ref", ""): "".join(comment.itertext())
            for comment in comments
        }
        korean_count = sum(bool(HANGUL_RE.search(text)) for text in comment_texts.values())
        summary["note_comment_count"] = len(comments)
        summary["korean_note_comment_count"] = korean_count
        if len(comments) != 117:
            issues.append(f"expected 117 notes/comments, found {len(comments)}")
        if korean_count != 117:
            issues.append(f"expected 117 Korean notes/comments, found {korean_count}")
        h1_note = comment_texts.get("H1")
        if h1_note != EXPECTED_H1_NOTE:
            issues.append(f"H1 note mismatch: expected {EXPECTED_H1_NOTE!r}, found {h1_note!r}")
        if h1_note is not None and STALE_H1_TEXT in h1_note:
            issues.append(f"H1 note contains stale text {STALE_H1_TEXT!r}")

    return summary, issues


def verify_snapshot() -> tuple[str | None, dict[str, object], list[str]]:
    issues: list[str] = []
    summary: dict[str, object] = {}
    digest: str | None = None
    try:
        manifest = _load_manifest(MANIFEST_PATH)
    except SnapshotFormatError as exc:
        issues.append(str(exc))
    else:
        issues.extend(validate_manifest_data(manifest))
    if not WORKBOOK_PATH.is_file():
        issues.append(f"missing workbook {WORKBOOK_PATH.name}")
        return digest, summary, issues
    digest = _hash_file(WORKBOOK_PATH)
    if digest != EXPECTED_SHA256:
        issues.append(f"checksum drift: expected {EXPECTED_SHA256}, found {digest}")
    size_bytes = WORKBOOK_PATH.stat().st_size
    if size_bytes != 29321:
        issues.append(f"size drift: expected 29321 bytes, found {size_bytes}")
    try:
        summary, workbook_issues = _inspect_workbook(WORKBOOK_PATH)
    except SnapshotFormatError as exc:
        issues.append(str(exc))
    else:
        issues.extend(workbook_issues)
    return digest, summary, issues


class CorruptionSelfTests(unittest.TestCase):
    def test_rejects_corrupted_manifest_value(self) -> None:
        manifest = copy.deepcopy(EXPECTED_MANIFEST)
        self.assertFalse(validate_manifest_data(manifest))
        manifest["artifact"]["sha256"] = "0" * 64
        self.assertTrue(validate_manifest_data(manifest))

    def test_rejects_corrupted_workbook_relationship_fixture(self) -> None:
        clean = b"""<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>
"""
        self.assertFalse(external_link_issues(clean))
        injected = b"""  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/externalLink" Target="externalLinks/externalLink1.xml"/>
"""
        corrupted = clean.replace(b"</Relationships>", injected + b"</Relationships>")
        self.assertTrue(external_link_issues(corrupted))


def run_self_tests() -> unittest.result.TestResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CorruptionSelfTests)
    return unittest.TextTestRunner(verbosity=2).run(suite)


def _print_verification(
    digest: str | None, summary: dict[str, object], issues: list[str]
) -> int:
    if digest is not None:
        print(f"SHA256 {digest}")
    if issues:
        for issue in issues:
            print(f"FAIL {issue}", file=sys.stderr)
        return 1
    print(
        "PASS "
        f"sheet=시트1 range={summary['used_range']} "
        f"names={summary['defined_name_count']} formulas={summary['formula_cell_count']} "
        f"korean_notes={summary['korean_note_comment_count']} "
        f"cached_errors={summary['cached_formula_error_count']} "
        f"external_links={summary['external_workbook_link_count']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args not in ([], ["--self-test"]):
        print("usage: verify_snapshot.py [--self-test]", file=sys.stderr)
        return 2
    if args == ["--self-test"]:
        result = run_self_tests()
        if not result.wasSuccessful():
            return 1
        print(f"SELF-TEST PASS {result.testsRun} corruption checks")
    digest, summary, issues = verify_snapshot()
    return _print_verification(digest, summary, issues)


if __name__ == "__main__":
    raise SystemExit(main())
