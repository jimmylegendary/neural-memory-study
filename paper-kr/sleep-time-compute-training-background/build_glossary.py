#!/usr/bin/env python3
"""Generate the Background Paper glossary from the canonical concept registry."""

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
    return "".join(replacements.get(char, char) for char in value)


def build_glossary(registry_path: Path, output_path: Path) -> None:
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    concepts = registry["concepts"]
    lines = [
        r"\chapter*{용어 사전과 개념 의존성}",
        r"\addcontentsline{toc}{chapter}{용어 사전과 개념 의존성}",
        r"\markboth{용어 사전과 개념 의존성}{용어 사전과 개념 의존성}",
        r"\setcounter{table}{0}",
        r"\renewcommand{\thetable}{G.\arabic{table}}",
        r"\label{bg:glossary}",
        "",
        "이 사전은 Study Paper가 참조하는 canonical concept registry에서 자동 생성한다.",
        "용어를 단독 정의로만 읽지 말고, 세 번째 열의 prerequisite를 먼저 확인한다.",
        "본문의 파란 링크는 각 개념이 처음 설명되는 절로 이동한다.",
        "",
        r"\small",
        r"\begin{longtable}{p{0.20\textwidth} p{0.52\textwidth} p{0.18\textwidth}}",
        r"\caption{Sleep-time compute를 읽기 위한 canonical glossary}\label{tab:bg-glossary}\\",
        r"\toprule",
        r"용어 (concept ID) & 한 문장 정의 & 선행 개념 \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"용어 (concept ID) & 한 문장 정의 & 선행 개념 \\",
        r"\midrule",
        r"\endhead",
        r"\midrule\multicolumn{3}{r}{다음 쪽에 계속}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for item in concepts:
        concept_id = item["concept_id"]
        label = _tex(item["label"])
        term = _tex(item["glossary_term"])
        definition = _tex(item["definition"])
        prerequisites = item.get("prerequisites", [])
        prerequisite_text = ", ".join(_tex(value) for value in prerequisites) or "없음"
        lines.append(
            rf"\hypertarget{{glossary:{concept_id}}}{{\hyperref[{label}]{{\textbf{{{term}}}}}}} "
            rf"\newline\texttt{{{_tex(concept_id)}}} & {definition} & "
            rf"\scriptsize {prerequisite_text} \\"
        )
    lines.extend(
        [
            r"\end{longtable}",
            r"\normalsize",
            "",
            r"\begin{keypoint}[의존성 표를 사용하는 법]",
            "어떤 Study claim이 낯설다면 해당 concept ID의 본문 label로 이동하고, 선행 개념을",
            "왼쪽에서 오른쪽 순서로 읽는다. 용어 정의는 결과를 보장하지 않는다. 예를 들어",
            r"\texttt{unlearning}의 정의를 안다는 것은 특정 방법이 삭제 요구를 완전히 충족한다는 뜻이 아니다.",
            r"\end{keypoint}",
            "",
        ]
    )
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_glossary(args.registry, args.output)
    print(f"glossary: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
