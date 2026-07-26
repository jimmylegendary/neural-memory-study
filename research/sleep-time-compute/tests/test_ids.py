from dataclasses import FrozenInstanceError

import pytest

from stc_research.ids import parse_stable_id


@pytest.mark.parametrize(
    ("value", "kind", "number"),
    [
        ("SRC-STC-0001", "source", 1),
        ("EV-STC-00001", "evidence", 1),
        ("CL-STC-0042", "claim", 42),
        ("H-STC-005", "hypothesis", 5),
        ("H-STC-006", "hypothesis", 6),
        ("H-STC-007", "hypothesis", 7),
        ("PAPER-C4", "paper_claim", 4),
        ("RQ12", "question", 12),
        ("ART-STC-0042", "artifact", 42),
        ("EXP-STC-0007", "experiment", 7),
        ("RES-STC-0109", "result", 109),
        ("FIG-STC-042", "figure", 42),
        ("TAB-STC-007", "table", 7),
    ],
)
def test_parse_stable_id(value, kind, number):
    parsed = parse_stable_id(value)
    assert (parsed.kind, parsed.number, parsed.value) == (kind, number, value)


@pytest.mark.parametrize(
    "value",
    [
        "SRC-STC-1",
        "SRC-STC-0000",
        "EV-00001",
        "H-STC-000",
        "H-STC-1000",
        "PAPER-C5",
        "RQ0",
        "RQ13",
        "",
        " SRC-STC-0001",
    ],
)
def test_reject_noncanonical_ids(value):
    with pytest.raises(ValueError, match="noncanonical stable ID"):
        parse_stable_id(value)


@pytest.mark.parametrize(
    "value",
    [
        "SRC-STC-０００１",
        "EV-STC-٠٠٠٠١",
        "CL-STC-𝟘𝟘𝟘𝟙",
    ],
)
def test_reject_unicode_digit_confusables(value):
    with pytest.raises(ValueError, match="noncanonical stable ID"):
        parse_stable_id(value)


def test_stable_id_is_immutable():
    parsed = parse_stable_id("SRC-STC-0001")
    with pytest.raises(FrozenInstanceError):
        parsed.number = 2
