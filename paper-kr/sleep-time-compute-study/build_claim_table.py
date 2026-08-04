#!/usr/bin/env python3
"""Generate the canonical manuscript claim ledger from the frozen claim map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _tex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def build_claim_table(claim_map_path: Path, output_path: Path) -> None:
    claims = json.loads(Path(claim_map_path).read_text(encoding="utf-8"))["claims"]
    lines = [
        "% Generated from claims/stc-study/claim-map.json. Do not edit by hand.",
        r"\chapter*{Canonical claim–evidence ledger}",
        r"\addcontentsline{toc}{chapter}{Canonical claim–evidence ledger}",
        r"\markboth{Canonical claim–evidence ledger}{Canonical claim–evidence ledger}",
        "본문의 파란 claim ID는 이 표의 canonical English statement와 source locator로 연결된다. ",
        "이 표는 번역문이 아니라 Veridraft gate를 통과한 frozen assertion surface다.",
        "",
        r"\scriptsize",
        r"\begin{longtable}{p{0.105\textwidth} p{0.105\textwidth} p{0.455\textwidth} p{0.245\textwidth}}",
        r"\caption{Veridraft-gated public claims (source freeze 2026-08-05)}\label{tab:claim-ledger}\\",
        r"\toprule",
        r"ID & Type / confidence & Canonical claim & Evidence locator \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"ID & Type / confidence & Canonical claim & Evidence locator \\",
        r"\midrule",
        r"\endhead",
        r"\midrule\multicolumn{4}{r}{다음 쪽에 계속}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for claim in claims:
        claim_id = claim["claim_id"]
        kind = _tex(claim["claim_type"].replace("_", " "))
        confidence = _tex(claim["confidence"])
        statement = _tex(claim["text"])
        support = []
        for item in claim.get("support", []):
            support.append(
                rf"\texttt{{{_tex(item['source_id'])}}}: {_tex(item.get('locator', ''))}"
            )
        evidence = r"\newline ".join(support) or "No public support listed"
        lines.append(
            rf"\hypertarget{{claim-ledger:{claim_id}}}{{\hyperlink{{claim:{claim_id}}}{{\textbf{{{claim_id}}}}}}} "
            rf"& {kind}\newline {confidence} & {statement} & {evidence} \\"
        )
    lines.extend([r"\end{longtable}", r"\normalsize", ""])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_claim_table(args.claim_map, args.output)
    print(f"claim table: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
