from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa_translations import compare_structure, fixture_record, validate_record


def test_translation_requires_source_hash_and_version():
    record = fixture_record(source_sha256="", version="")
    report = validate_record(record)
    assert {"missing-source-hash", "missing-source-version"} <= set(report.codes)


def test_equation_table_figure_counts_must_match_manifest():
    report = compare_structure(
        expected={"equations": 7, "tables": 3, "figures": 4},
        actual={"equations": 6, "tables": 3, "figures": 4},
    )
    assert report.codes == ("equation-count-mismatch",)


def test_complete_structure_match_is_clean():
    report = compare_structure(
        expected={"equations": 7, "tables": 3, "figures": 4},
        actual={"equations": 7, "tables": 3, "figures": 4},
    )
    assert report.codes == ()
