"""Structural contract tests for the generated Attn-vs-HOPE workbook.

These tests deliberately use only the Python standard library.  The workbook
is treated as an OOXML ZIP package so the checks remain independent of Excel,
LibreOffice, and the JavaScript builder runtime.
"""

from __future__ import annotations

import re
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


SHEET_DIR = Path(__file__).resolve().parent
BUILDER = SHEET_DIR / "build_workbook.mjs"
WORKBOOK = SHEET_DIR / "Attn-vs-HOPE.xlsx"

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


def _xml(archive: zipfile.ZipFile, name: str) -> ET.Element:
    return ET.fromstring(archive.read(name))


def _column_number(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref).group(0)
    value = 0
    for letter in letters:
        value = value * 26 + ord(letter) - ord("A") + 1
    return value


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

    def test_calculation_formulas_have_quoted_cross_sheet_lineage(self) -> None:
        package = self.require_package()
        quoted_ref = re.compile(r"'([^']+)'!\$?[A-Z]{1,3}\$?\d+")
        unquoted_ref = re.compile(r"(?<!')\b(?:[A-Za-z_][A-Za-z0-9_]*)!")
        for sheet_name in CALCULATION_SHEETS:
            for formula in package.formulas(sheet_name):
                with self.subTest(sheet=sheet_name, formula=formula[:80]):
                    refs = quoted_ref.findall(formula)
                    self.assertTrue(refs, "formula has no quoted cross-sheet reference")
                    self.assertFalse(unquoted_ref.search(formula))
                    self.assertTrue(all(ref in EXPECTED_SHEETS for ref in refs))

    def test_calculation_ranges_do_not_embed_model_or_hardware_constants(self) -> None:
        package = self.require_package()
        forbidden = re.compile(
            r"(?<![A-Z0-9_])(?:989472000000000|3350000000000|85899345920|"
            r"131072|16384|8192)(?![A-Z0-9_])"
        )
        for sheet_name in CALCULATION_SHEETS:
            for formula in package.formulas(sheet_name):
                with self.subTest(sheet=sheet_name, formula=formula[:80]):
                    self.assertIsNone(forbidden.search(formula))

    def test_numeric_calculation_cells_are_formula_backed(self) -> None:
        package = self.require_package()
        for sheet_name in CALCULATION_SHEETS:
            sheet = package.sheet_xml(sheet_name)
            for cell in sheet.findall(f".//{{{MAIN_NS}}}c"):
                ref = cell.attrib.get("r", "")
                row_match = re.search(r"\d+", ref)
                if not row_match or int(row_match.group(0)) < 5:
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

    def test_three_approved_native_charts(self) -> None:
        package = self.require_package()
        charts = [
            name
            for name in package.archive.namelist()
            if re.fullmatch(r"xl/(?:drawings/)?charts/chart\d+\.xml", name)
        ]
        self.assertEqual(3, len(charts))

    def test_qa_sheet_labels_imported_engine_references(self) -> None:
        package = self.require_package()
        text = "\n".join(package.visible_text("90_QA"))
        self.assertIn("Engine reference (imported)", text)
        self.assertIn("Workbook formula", text)
        self.assertIn("Delta", text)


if __name__ == "__main__":
    unittest.main()
