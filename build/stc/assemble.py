#!/usr/bin/env python3
"""Sleep-Time Compute v2 본문을 읽기 순서로 조립해 build/stc/BOOK-STC.md 를 만든다.

Neural Memory의 빌드 방법(`build/METHOD.md`)을 그대로 승계한다. 핵심 두 가지:

1. **LaTeX 로그는 overflow 게이트가 아니다.** luatexja는 줄바꿈 불가한 Latin/코드 런을
   `Overfull \\hbox` 없이 오른쪽 여백 밖으로 흘려보낸다. 게이트는 `build/overflow_gate.py`의
   전 페이지 픽셀 스캔이다.
2. **그림 경로는 절대경로로 바꾼다.** 장별 md는 상대경로로 쓰고, 조립 시 치환한다.

사용:
    python3 build/stc/assemble.py              # BOOK-STC.md 생성
    python3 build/stc/assemble.py --check      # 조립 없이 구조만 검사 (누락 장·순서·중복)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "study-kr-stc"
OUT = REPO / "build" / "stc" / "BOOK-STC.md"

# 읽기 순서. dossier/stc-v2/PRE-RESEARCH.md §6 ToC가 정본이다.
# 파일이 아직 없으면 --check 가 보고하고, 조립은 있는 것까지만 한다.
READING_ORDER = [
    ("front", ["00-preface", "00-toc", "01-part1-opener"]),
    ("part1", [
        "ch01-orientation-three-layers",
        "ch02-training-regimes",
        "ch03-finetuning-and-peft",
        "ch04-distillation",
        "ch05-rl-for-llm",
        "ch06-making-data",
        "ch07-forgetting-and-replay",
        "ch08-cls-and-biological-sleep",
        "ch09-memory-capacity",
        "ch10-external-memory",
    ]),
    ("front", ["02-part2-opener"]),
    ("part2", [
        "ch11-discriminant-and-census",
        "ch12-memorizing-transformers",
        "ch13-rag-to-memgpt",
        "ch14-letta-sleep-time-compute",
        "ch15-mem0-zep-reasoningbank",
        "ch16-generative-adapter-hinge",
        "ch17-do-lms-need-sleep",
        "ch18-nested-learning-memory-caching",
        "ch19-lora-and-model-editing",
        "ch20-seal",
        "ch21-lm-need-sleep",
        "ch22-falsifier-axis",
        "ch23-what-biology-showed",
    ]),
    ("front", ["03-part3-opener"]),
    ("part3", [
        "ch24-what-actually-counts",
        "ch25-unified-cost-accounting",
        "ch26-state-capacity-and-density",
        "ch27-is-it-promising",
        "ch28-scaling-law",
        "ch29-systems-and-device",
        "ch30-limits-and-falsifiers",
    ]),
    ("back", ["90-bibliography", "91-glossary", "92-appendix-census"]),
]

FIG_REL = re.compile(r"\]\((\.\./)+figures/")
FIG_ABS = f"]({REPO}/figures/"


def iter_expected():
    for d, names in READING_ORDER:
        for n in names:
            yield SRC / d / f"{n}.md"


def check() -> int:
    missing, present = [], []
    for p in iter_expected():
        (present if p.exists() else missing).append(p)

    # 예상 목록 밖의 파일이 있는가 (슬러그 오타 탐지)
    on_disk = {p.resolve() for p in SRC.rglob("*.md")} if SRC.exists() else set()
    expected = {p.resolve() for p in iter_expected()}
    stray = sorted(on_disk - expected)

    print(f"작성됨 {len(present)} / 예상 {len(present) + len(missing)}")
    if missing:
        print("\n미작성:")
        for p in missing:
            print(f"  - {p.relative_to(REPO)}")
    if stray:
        print("\n읽기 순서에 없는 파일 (슬러그 불일치 의심):")
        for p in stray:
            print(f"  ! {p.relative_to(REPO)}")
    return 1 if stray else 0


def assemble() -> int:
    parts, used, skipped = [], 0, 0
    for p in iter_expected():
        if not p.exists():
            skipped += 1
            continue
        text = p.read_text(encoding="utf-8")
        text = FIG_REL.sub(FIG_ABS, text)
        if not text.endswith("\n"):
            text += "\n"
        parts.append(text)
        used += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n\n".join(parts), encoding="utf-8")
    words = sum(len(t.split()) for t in parts)
    print(f"wrote {OUT.relative_to(REPO)}  ({used} files, {skipped} missing, ~{words:,} words)")
    print("\n다음 단계:")
    print("  cd build/stc && pandoc BOOK-STC.md -o BOOK-STC.pdf --pdf-engine=lualatex \\")
    print("      --template=../template.tex --lua-filter=../breakable.lua --toc \\")
    print("      --top-level-division=chapter \\")
    print(f"      --resource-path={REPO}:{REPO}/figures")
    print("  python3 ../overflow_gate.py BOOK-STC.pdf     # 반드시 PASS")
    print("  grep -c 'Missing character' build_err.txt    # 반드시 0")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="조립 없이 구조만 검사")
    a = ap.parse_args()
    return check() if a.check else assemble()


if __name__ == "__main__":
    sys.exit(main())
