"""Structural contract tests for the generated Attn-vs-HOPE workbook.

These tests deliberately use only the Python standard library.  The workbook
is treated as an OOXML ZIP package so the checks remain independent of Excel,
LibreOffice, and the JavaScript builder runtime.
"""

from __future__ import annotations

import io
import json
import re
import unittest
import zipfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from xml.etree import ElementTree as ET


SHEET_DIR = Path(__file__).resolve().parent
BUILDER = SHEET_DIR / "build_workbook.mjs"
WORKBOOK = SHEET_DIR / "Attn-vs-HOPE.xlsx"
RESULTS = SHEET_DIR.parent / "results.json"

EXPECTED_SHEETS = [
    "00_Guide",
    "01_Inputs",
    "02_HW",
    "10_AttnMoE_Prefill",
    "11_AttnMoE_Decode",
    "20_HOPE_Prefill",
    "21_HOPE_Decode",
    "30_Compare",
    "40_Sweeps",
    "90_QA",
    "99_Sources",
]

CALCULATION_SHEETS = [
    "10_AttnMoE_Prefill",
    "11_AttnMoE_Decode",
    "20_HOPE_Prefill",
    "21_HOPE_Decode",
    "30_Compare",
    "40_Sweeps",
]

STAGE_HEADERS = [
    "Order",
    "Phase",
    "Component",
    "Stage",
    "Symbolic equation",
    "Executions/cadence",
    "FLOPs",
    "Required read B",
    "Mandatory write B",
    "Temporary/fusible write B",
    "Effective read B",
    "Effective write B",
    "Persistent state B",
    "AI",
    "Compute ms",
    "HBM ms",
    "Roofline ms",
    "Bound",
    "Evidence/assumption",
]

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

SOURCE_SHEETS = {"01_Inputs", "02_HW"}
ERROR_VALUES = {"#REF!", "#DIV/0!", "#VALUE!", "#NAME?", "#N/A", "#NUM!", "#NULL!"}
CELL_RANGE_RE = re.compile(
    r"(?:(?:'(?P<quoted>[^']+)'|(?P<bare>[A-Za-z_][A-Za-z0-9_]*))!)?"
    r"(?P<start>\$?[A-Z]{1,3}\$?\d+)"
    r"(?::(?P<end>\$?[A-Z]{1,3}\$?\d+))?"
)
NOOP_SOURCE_RE = re.compile(
    r"(?:[+-]\s*)?0(?:\.0+)?\s*\*\s*\(?\s*"
    r"'(?:01_Inputs|02_HW)'!\$?[A-Z]{1,3}\$?\d+\s*\)?",
    re.I,
)


def _xml(archive: zipfile.ZipFile, name: str) -> ET.Element:
    return ET.fromstring(archive.read(name))


def _column_number(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref).group(0)
    value = 0
    for letter in letters:
        value = value * 26 + ord(letter) - ord("A") + 1
    return value


def _column_name(index: int) -> str:
    value = index
    letters = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


def _cell_coordinates(cell_ref: str) -> tuple[int, int]:
    normalized = cell_ref.replace("$", "")
    match = re.fullmatch(r"([A-Z]+)(\d+)", normalized)
    if match is None:
        raise ValueError(f"invalid A1 cell reference: {cell_ref}")
    return _column_number(match.group(1)), int(match.group(2))


def _expand_range(start: str, end: str | None) -> list[str]:
    start_col, start_row = _cell_coordinates(start)
    end_col, end_row = _cell_coordinates(end or start)
    return [
        f"{_column_name(column)}{row}"
        for row in range(min(start_row, end_row), max(start_row, end_row) + 1)
        for column in range(min(start_col, end_col), max(start_col, end_col) + 1)
    ]


def _without_noop_source_terms(formula: str) -> str:
    """Remove zero-multiplied source refs before evaluating lineage."""

    return NOOP_SOURCE_RE.sub("0", formula)


def _formula_references(formula: str, current_sheet: str) -> list[tuple[str, str]]:
    cleaned = _without_noop_source_terms(formula)
    cleaned = re.sub(r'"(?:[^"]|"")*"', '""', cleaned)
    references: list[tuple[str, str]] = []
    for match in CELL_RANGE_RE.finditer(cleaned):
        target_sheet = match.group("quoted") or match.group("bare") or current_sheet
        for cell_ref in _expand_range(match.group("start"), match.group("end")):
            references.append((target_sheet, cell_ref))
    return references


def _walk_numbers(value: object) -> list[Decimal]:
    numbers: list[Decimal] = []
    if isinstance(value, bool) or value is None:
        return numbers
    if isinstance(value, (int, float)):
        numbers.append(Decimal(str(value)))
    elif isinstance(value, dict):
        for child in value.values():
            numbers.extend(_walk_numbers(child))
    elif isinstance(value, list):
        for child in value:
            numbers.extend(_walk_numbers(child))
    return numbers


class WorkbookPackage:
    def __init__(self, path: Path) -> None:
        self.archive = zipfile.ZipFile(path)
        workbook = _xml(self.archive, "xl/workbook.xml")
        rels = _xml(self.archive, "xl/_rels/workbook.xml.rels")
        targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall(f"{{{PKG_REL_NS}}}Relationship")
        }
        self.sheet_paths: dict[str, str] = {}
        for sheet in workbook.findall(f".//{{{MAIN_NS}}}sheet"):
            target = targets[sheet.attrib[f"{{{REL_NS}}}id"]].lstrip("/")
            if target.startswith("xl/"):
                path_name = target
            else:
                path_name = f"xl/{target}"
            self.sheet_paths[sheet.attrib["name"]] = path_name

        self.shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in self.archive.namelist():
            shared = _xml(self.archive, "xl/sharedStrings.xml")
            for item in shared.findall(f"{{{MAIN_NS}}}si"):
                self.shared_strings.append(
                    "".join(node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t"))
                )

    def close(self) -> None:
        self.archive.close()

    @property
    def sheet_names(self) -> list[str]:
        return list(self.sheet_paths)

    def sheet_xml(self, name: str) -> ET.Element:
        return _xml(self.archive, self.sheet_paths[name])

    def formulas(self, name: str) -> list[str]:
        return [
            node.text or ""
            for node in self.sheet_xml(name).findall(f".//{{{MAIN_NS}}}f")
        ]

    def cells(self, name: str) -> dict[str, ET.Element]:
        return {
            cell.attrib["r"].replace("$", ""): cell
            for cell in self.sheet_xml(name).findall(f".//{{{MAIN_NS}}}c")
        }

    def cell(self, name: str, cell_ref: str) -> ET.Element | None:
        return self.cells(name).get(cell_ref.replace("$", ""))

    def cell_formula(self, name: str, cell_ref: str) -> str | None:
        cell = self.cell(name, cell_ref)
        if cell is None:
            return None
        formula = cell.find(f"{{{MAIN_NS}}}f")
        return None if formula is None else (formula.text or "")

    def cell_value(self, name: str, cell_ref: str) -> str | float | None:
        cell = self.cell(name, cell_ref)
        if cell is None:
            return None
        cell_type = cell.attrib.get("t")
        if cell_type == "inlineStr":
            return "".join(node.text or "" for node in cell.iter(f"{{{MAIN_NS}}}t"))
        value = cell.find(f"{{{MAIN_NS}}}v")
        if value is None or value.text is None:
            return None
        if cell_type == "s":
            return self.shared_strings[int(value.text)]
        if cell_type in {"str", "e"}:
            return value.text
        try:
            return float(value.text)
        except ValueError:
            return value.text

    def find_row(self, name: str, column: str, value: str) -> int | None:
        for cell_ref in self.cells(name):
            if not cell_ref.startswith(column):
                continue
            if self.cell_value(name, cell_ref) == value:
                return int(re.search(r"\d+", cell_ref).group(0))
        return None

    def lineage_sources(
        self,
        sheet_name: str,
        cell_ref: str,
        seen: set[tuple[str, str]] | None = None,
    ) -> set[str]:
        target = (sheet_name, cell_ref.replace("$", ""))
        visited = set() if seen is None else seen
        if target in visited:
            return set()
        visited.add(target)
        formula = self.cell_formula(*target)
        if formula is None:
            return {sheet_name} if sheet_name in SOURCE_SHEETS else set()
        sources: set[str] = set()
        for dependency_sheet, dependency_cell in _formula_references(formula, sheet_name):
            if dependency_sheet in SOURCE_SHEETS:
                sources.add(dependency_sheet)
            elif dependency_sheet in self.sheet_paths:
                sources.update(
                    self.lineage_sources(dependency_sheet, dependency_cell, visited)
                )
        return sources

    def chart_specs(self) -> list[dict[str, object]]:
        specs: list[dict[str, object]] = []
        for name in self.archive.namelist():
            if not re.fullmatch(r"xl/(?:drawings/)?charts/chart\d+\.xml", name):
                continue
            root = _xml(self.archive, name)
            chart_types = {
                node.tag.rsplit("}", 1)[-1]
                for node in root.iter()
                if node.tag.rsplit("}", 1)[-1].endswith("Chart")
            }
            specs.append(
                {
                    "path": name,
                    "title": "".join(
                        node.text or ""
                        for node in root.iter()
                        if node.tag.rsplit("}", 1)[-1] == "t"
                    ),
                    "types": chart_types,
                    "formulas": [
                        node.text or ""
                        for node in root.iter()
                        if node.tag.rsplit("}", 1)[-1] == "f"
                    ],
                    "x_formulas": [
                        formula.text or ""
                        for axis in root.iter()
                        if axis.tag.rsplit("}", 1)[-1] == "xVal"
                        for formula in axis.iter()
                        if formula.tag.rsplit("}", 1)[-1] == "f"
                    ],
                    "y_formulas": [
                        formula.text or ""
                        for axis in root.iter()
                        if axis.tag.rsplit("}", 1)[-1] == "yVal"
                        for formula in axis.iter()
                        if formula.tag.rsplit("}", 1)[-1] == "f"
                    ],
                }
            )
        return specs

    def visible_text(self, name: str) -> list[str]:
        values: list[str] = []
        for cell in self.sheet_xml(name).findall(f".//{{{MAIN_NS}}}c"):
            cell_type = cell.attrib.get("t")
            if cell_type == "s":
                value = cell.find(f"{{{MAIN_NS}}}v")
                if value is not None and value.text is not None:
                    values.append(self.shared_strings[int(value.text)])
            elif cell_type == "inlineStr":
                values.append(
                    "".join(
                        node.text or ""
                        for node in cell.iter(f"{{{MAIN_NS}}}t")
                    )
                )
            elif cell_type == "str":
                value = cell.find(f"{{{MAIN_NS}}}v")
                if value is not None and value.text is not None:
                    values.append(value.text)
        return values


class TestWorkbookSource(unittest.TestCase):
    package: WorkbookPackage | None = None

    @classmethod
    def setUpClass(cls) -> None:
        if WORKBOOK.exists():
            cls.package = WorkbookPackage(WORKBOOK)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.package is not None:
            cls.package.close()

    def require_package(self) -> WorkbookPackage:
        if self.package is None:
            self.skipTest("generated workbook is missing; run build_workbook.mjs")
        return self.package

    def test_00_builder_and_generated_workbook_exist(self) -> None:
        missing = [str(path) for path in (BUILDER, WORKBOOK) if not path.exists()]
        self.assertFalse(missing, "missing Task 2 artifact(s): " + ", ".join(missing))

    def test_builder_uses_portable_results_interface(self) -> None:
        if not BUILDER.exists():
            self.skipTest("builder is missing")
        source = BUILDER.read_text(encoding="utf-8")
        self.assertIn('import("@oai/artifact-tool")', source)
        self.assertRegex(source, r'new URL\(["\']\.\./results\.json["\'],\s*import\.meta\.url\)')
        self.assertNotRegex(source, re.compile(r"openpyxl|xlsxwriter|exceljs|sheetjs", re.I))

    def test_position_inputs_are_sourced_or_explicitly_labeled_overrides(self) -> None:
        if not BUILDER.exists() or not RESULTS.exists():
            self.skipTest("builder or results.json is missing")
        source = BUILDER.read_text(encoding="utf-8")
        self.assertRegex(
            source,
            r'\["prefill_position",\s*"Prefill position p",\s*hopePrefill\.position,',
            "prefill position must be read from results.json, not hardcoded",
        )
        self.assertRegex(
            source,
            r'\["decode_position",\s*"Decode position p",\s*hope\.position,',
            "decode position must be read from results.json, not hardcoded",
        )

        package = self.require_package()
        results = json.loads(RESULTS.read_text(encoding="utf-8"))
        expected = results["hope_scenarios"]["shipped_credible_momentum"]["inputs"]
        for key, phase in (("prefill_position", "prefill"), ("decode_position", "decode")):
            with self.subTest(key=key):
                row = package.find_row("01_Inputs", "A", key)
                self.assertIsNotNone(row)
                self.assertEqual(float(expected[phase]["position"]), package.cell_value("01_Inputs", f"C{row}"))
                self.assertEqual(
                    f"hope_scenarios.shipped_credible_momentum.inputs.{phase}.position",
                    package.cell_value("01_Inputs", f"E{row}"),
                )

    def test_exact_sheet_names_and_order(self) -> None:
        package = self.require_package()
        self.assertEqual(EXPECTED_SHEETS, package.sheet_names)

    def test_no_external_workbook_links(self) -> None:
        package = self.require_package()
        names = package.archive.namelist()
        self.assertFalse(any(name.startswith("xl/externalLinks/") for name in names))
        for name in names:
            if not name.endswith(".rels"):
                continue
            root = _xml(package.archive, name)
            for rel in root.findall(f"{{{PKG_REL_NS}}}Relationship"):
                self.assertNotEqual("External", rel.attrib.get("TargetMode"))

    def test_stage_tables_use_all_design_section_8_headers(self) -> None:
        package = self.require_package()
        for sheet_name in CALCULATION_SHEETS[:4]:
            with self.subTest(sheet=sheet_name):
                text = set(package.visible_text(sheet_name))
                self.assertTrue(set(STAGE_HEADERS).issubset(text))

    def test_every_calculation_sheet_contains_formulas(self) -> None:
        package = self.require_package()
        for sheet_name in CALCULATION_SHEETS:
            with self.subTest(sheet=sheet_name):
                self.assertGreater(len(package.formulas(sheet_name)), 0)

    def test_cross_sheet_formula_references_are_quoted(self) -> None:
        package = self.require_package()
        quoted_ref = re.compile(r"'([^']+)'!\$?[A-Z]{1,3}\$?\d+")
        unquoted_ref = re.compile(r"(?<!')\b(?:[A-Za-z_][A-Za-z0-9_]*)!")
        for sheet_name in CALCULATION_SHEETS:
            for formula in package.formulas(sheet_name):
                with self.subTest(sheet=sheet_name, formula=formula[:80]):
                    refs = quoted_ref.findall(formula)
                    self.assertFalse(unquoted_ref.search(formula))
                    self.assertTrue(all(ref in EXPECTED_SHEETS for ref in refs))

    def test_zero_multiplier_source_refs_are_not_used_as_lineage_sentinels(self) -> None:
        package = self.require_package()
        self.assertNotIn(
            ("01_Inputs", "C5"),
            _formula_references("SUM(A1:A2)+0*('01_Inputs'!$C$5)", "30_Compare"),
            "the lineage parser itself must ignore a zero-multiplied source reference",
        )
        for sheet_name in CALCULATION_SHEETS:
            for formula in package.formulas(sheet_name):
                with self.subTest(sheet=sheet_name, formula=formula[:100]):
                    self.assertIsNone(
                        re.search(
                            r"0\s*\*\s*\(?\s*'(?:01_Inputs|02_HW)'!",
                            formula,
                            re.I,
                        ),
                        "calculation formula contains a no-op source-lineage sentinel",
                    )

    def test_stage_summary_row_3_has_substantive_transitive_lineage(self) -> None:
        package = self.require_package()
        for sheet_name in CALCULATION_SHEETS[:4]:
            for column in "ABCDEFGHIJKLMN":
                cell_ref = f"{column}3"
                with self.subTest(sheet=sheet_name, cell=cell_ref):
                    formula = package.cell_formula(sheet_name, cell_ref)
                    self.assertIsNotNone(formula, "stage summary metric must be formula-backed")
                    self.assertEqual(
                        formula,
                        _without_noop_source_terms(formula or ""),
                        "summary lineage may not depend on a zero-multiplied sentinel",
                    )
                    self.assertTrue(
                        package.lineage_sources(sheet_name, cell_ref),
                        "summary formula must transitively reach 01_Inputs or 02_HW",
                    )

    def test_calculation_ranges_do_not_embed_model_or_hardware_constants(self) -> None:
        package = self.require_package()
        results = json.loads(RESULTS.read_text(encoding="utf-8"))
        raw_roots = [
            results["model_inputs"],
            results["hardware"],
            results["attention_moe"]["inputs"]["prefill"],
            results["attention_moe"]["inputs"]["decode"],
            results["hope_scenarios"]["shipped_credible_momentum"]["inputs"]["prefill"],
            results["hope_scenarios"]["shipped_credible_momentum"]["inputs"]["decode"],
        ]
        forbidden = {number for root in raw_roots for number in _walk_numbers(root)}
        forbidden.update({Decimal("1073741824"), Decimal("1000000000000")})
        # 0/1/2/4 are universal algebraic coefficients in the published equations,
        # not proof that a model input was embedded. All other raw constants are banned.
        forbidden.difference_update({Decimal("0"), Decimal("1"), Decimal("2"), Decimal("4")})
        a1_reference = re.compile(
            r"(?:'[^']+'!|[A-Za-z_][A-Za-z0-9_]*!)?"
            r"\$?[A-Z]{1,3}\$?\d+(?::\$?[A-Z]{1,3}\$?\d+)?"
        )
        numeric_token = re.compile(
            r"(?<![A-Za-z_])(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?(?![A-Za-z_])"
        )
        for sheet_name in CALCULATION_SHEETS:
            for cell_ref, cell in package.cells(sheet_name).items():
                formula_node = cell.find(f"{{{MAIN_NS}}}f")
                if formula_node is None:
                    continue
                if sheet_name in CALCULATION_SHEETS[:4] and cell_ref.startswith("A"):
                    continue  # row ordinal formulas contain a layout offset, not a model constant
                formula = formula_node.text or ""
                without_strings = re.sub(r'"(?:[^"]|"")*"', '""', formula)
                without_refs = a1_reference.sub("", without_strings)
                embedded: set[Decimal] = set()
                for token in numeric_token.findall(without_refs):
                    try:
                        number = Decimal(token)
                    except InvalidOperation:
                        continue
                    if number in forbidden:
                        embedded.add(number)
                with self.subTest(sheet=sheet_name, cell=cell_ref, formula=formula[:100]):
                    self.assertFalse(
                        embedded,
                        f"formula embeds raw model/HW constant(s): {sorted(embedded)}",
                    )

    def test_numeric_calculation_cells_are_formula_backed(self) -> None:
        package = self.require_package()
        for sheet_name in CALCULATION_SHEETS:
            sheet = package.sheet_xml(sheet_name)
            for cell in sheet.findall(f".//{{{MAIN_NS}}}c"):
                ref = cell.attrib.get("r", "")
                row_match = re.search(r"\d+", ref)
                if not row_match or int(row_match.group(0)) < 3:
                    continue
                if cell.attrib.get("t") in {"s", "inlineStr", "str", "b", "e"}:
                    continue
                value = cell.find(f"{{{MAIN_NS}}}v")
                if value is None or value.text in {None, ""}:
                    continue
                with self.subTest(sheet=sheet_name, cell=ref):
                    self.assertIsNotNone(
                        cell.find(f"{{{MAIN_NS}}}f"),
                        f"numeric calculation cell {sheet_name}!{ref} is literal",
                    )

    def test_compare_contains_required_section_8_binding_metrics(self) -> None:
        package = self.require_package()
        required_headers = {
            "HBM read GiB",
            "HBM write GiB",
            "HBM total GiB",
            "AI FLOP/B",
            "Ridge AI",
            "Ridge class",
        }
        self.assertTrue(required_headers.issubset(set(package.visible_text("30_Compare"))))
        case_sheets = {
            6: "10_AttnMoE_Prefill",
            7: "20_HOPE_Prefill",
            8: "11_AttnMoE_Decode",
            9: "21_HOPE_Decode",
        }
        expected_summary_cells = {
            "E": "$B$3",
            "F": "$C$3",
            "G": "$D$3",
            "H": "$F$3",
            "I": "$M$3",
            "J": "$N$3",
        }
        for row, source_sheet in case_sheets.items():
            for column, source_cell in expected_summary_cells.items():
                with self.subTest(row=row, column=column):
                    formula = package.cell_formula("30_Compare", f"{column}{row}") or ""
                    self.assertIn(f"'{source_sheet}'!{source_cell}", formula)

    def test_compare_crossover_summaries_are_formula_linked_to_sweeps(self) -> None:
        package = self.require_package()
        self.assertEqual("Context crossover", package.cell_value("30_Compare", "A13"))
        self.assertEqual("Batch crossover", package.cell_value("30_Compare", "A14"))
        expectations = {
            "B13": (range(6, 12), "tokens"),
            "B14": (range(17, 23), "requests"),
        }
        for cell_ref, (rows, unit) in expectations.items():
            with self.subTest(cell=cell_ref):
                formula = package.cell_formula("30_Compare", cell_ref) or ""
                for row in rows:
                    self.assertIn(f"'40_Sweeps'!$G${row}", formula)
                    self.assertIn(f"'40_Sweeps'!$A${row}", formula)
                self.assertEqual(unit, package.cell_value("30_Compare", f"C{cell_ref[1:]}"))

    def test_three_approved_native_charts_have_expected_semantics(self) -> None:
        package = self.require_package()
        charts = package.chart_specs()
        self.assertEqual(3, len(charts))
        by_title = {str(chart["title"]): chart for chart in charts}
        stage = by_title.get("Stagewise vs aggregate latency (ms)")
        roofline = by_title.get("Roofline scatter: AI vs effective TFLOP/s")
        context = by_title.get("Context sweep: decode ms and persistent state GiB")
        self.assertIsNotNone(stage)
        self.assertIsNotNone(roofline)
        self.assertIsNotNone(context)
        self.assertIn("barChart", stage["types"])
        self.assertIn("'30_Compare'!$B$6:$B$9", stage["formulas"])
        self.assertIn("'30_Compare'!$C$6:$C$9", stage["formulas"])
        self.assertIn("scatterChart", roofline["types"])
        self.assertEqual(["'30_Compare'!$P$6:$P$9"], roofline["x_formulas"])
        self.assertEqual(["'30_Compare'!$Q$6:$Q$9"], roofline["y_formulas"])
        self.assertIn("lineChart", context["types"])
        for column in "ABCDE":
            self.assertIn(f"'40_Sweeps'!${column}$6:${column}$11", context["formulas"])

        swapped_xml = b"""<?xml version="1.0" encoding="utf-8"?>
<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart">
  <c:chart><c:plotArea><c:scatterChart><c:ser>
    <c:xVal><c:numRef><c:f>'30_Compare'!$Q$6:$Q$9</c:f></c:numRef></c:xVal>
    <c:yVal><c:numRef><c:f>'30_Compare'!$P$6:$P$9</c:f></c:numRef></c:yVal>
  </c:ser></c:scatterChart></c:plotArea></c:chart>
</c:chartSpace>"""
        fixture_bytes = io.BytesIO()
        with zipfile.ZipFile(fixture_bytes, "w") as fixture_archive:
            fixture_archive.writestr("xl/drawings/charts/chart1.xml", swapped_xml)
        fixture_bytes.seek(0)
        swapped_package = object.__new__(WorkbookPackage)
        swapped_package.archive = zipfile.ZipFile(fixture_bytes)
        try:
            swapped = swapped_package.chart_specs()[0]
            x_formulas = swapped.get("x_formulas", swapped["formulas"])
            y_formulas = swapped.get("y_formulas", swapped["formulas"])
            self.assertFalse(
                "'30_Compare'!$P$6:$P$9" in x_formulas
                and "'30_Compare'!$Q$6:$Q$9" in y_formulas,
                "flattened formula inspection incorrectly accepts swapped xVal/yVal roles",
            )
        finally:
            swapped_package.close()
            fixture_bytes.close()

    def test_qa_sheet_labels_imported_engine_references(self) -> None:
        package = self.require_package()
        text = "\n".join(package.visible_text("90_QA"))
        self.assertIn("Engine reference (imported)", text)
        self.assertIn("Workbook formula", text)
        self.assertIn("Delta", text)

    def test_qa_rows_are_formula_backed_and_all_pass(self) -> None:
        package = self.require_package()
        for row in range(6, 22):
            with self.subTest(row=row):
                for column in "BDEF":
                    self.assertIsNotNone(
                        package.cell_formula("90_QA", f"{column}{row}"),
                        f"90_QA!{column}{row} must remain formula-backed",
                    )
                self.assertEqual("PASS", package.cell_value("90_QA", f"F{row}"))
                delta = package.cell_value("90_QA", f"D{row}")
                tolerance = package.cell_value("90_QA", f"E{row}")
                self.assertIsInstance(delta, float)
                self.assertIsInstance(tolerance, float)
                self.assertLessEqual(abs(delta), tolerance)

    def test_bounded_ooxml_formula_error_scan(self) -> None:
        package = self.require_package()
        checked = 0
        failures: list[str] = []
        for sheet_name in EXPECTED_SHEETS:
            for cell_ref, cell in package.cells(sheet_name).items():
                formula = cell.find(f"{{{MAIN_NS}}}f")
                if formula is None:
                    continue
                checked += 1
                value = cell.find(f"{{{MAIN_NS}}}v")
                cached = "" if value is None or value.text is None else value.text
                if cell.attrib.get("t") == "e" or cached in ERROR_VALUES:
                    failures.append(f"{sheet_name}!{cell_ref}={cached}")
                if any(error in (formula.text or "") for error in ERROR_VALUES):
                    failures.append(f"{sheet_name}!{cell_ref} formula contains an error token")
        self.assertGreater(checked, 0)
        self.assertLessEqual(checked, 5000, "formula scan exceeded its explicit OOXML bound")
        self.assertFalse(failures, "OOXML formula errors: " + ", ".join(failures[:20]))


if __name__ == "__main__":
    unittest.main()
