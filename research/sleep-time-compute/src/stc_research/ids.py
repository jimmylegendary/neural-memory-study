from __future__ import annotations

import re
from dataclasses import dataclass
from re import Pattern

PATTERNS: dict[str, Pattern[str]] = {
    "source": re.compile(r"^SRC-STC-([0-9]{4})$"),
    "evidence": re.compile(r"^EV-STC-([0-9]{5})$"),
    "claim": re.compile(r"^CL-STC-([0-9]{4})$"),
    "hypothesis": re.compile(r"^H-STC-([0-9]{3})$"),
    "paper_claim": re.compile(r"^PAPER-C([1-4])$"),
    "question": re.compile(r"^RQ([1-9]|1[0-2])$"),
    "artifact": re.compile(r"^ART-STC-([0-9]{4})$"),
    "experiment": re.compile(r"^EXP-STC-([0-9]{4})$"),
    "result": re.compile(r"^RES-STC-([0-9]{4})$"),
    "figure": re.compile(r"^FIG-STC-([0-9]{3})$"),
    "table": re.compile(r"^TAB-STC-([0-9]{3})$"),
}


@dataclass(frozen=True, slots=True)
class StableId:
    kind: str
    number: int
    value: str


def parse_stable_id(value: str) -> StableId:
    if not isinstance(value, str):
        raise TypeError(f"stable ID must be a string, got {type(value).__name__}")
    for kind, pattern in PATTERNS.items():
        match = pattern.fullmatch(value)
        if match is None:
            continue
        number = int(match.group(1))
        if number == 0:
            break
        return StableId(kind=kind, number=number, value=value)
    raise ValueError(f"noncanonical stable ID: {value!r}")
