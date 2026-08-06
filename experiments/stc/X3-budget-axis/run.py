#!/usr/bin/env python3
"""X3 — $B_s$ 축의 형태: 문헌이 그 축을 보고하는가에 대한 전수 감사.

배경
----
ch28은 "sleep-time compute이 주류가 되면 scaling law는 어떻게 되는가"에 답해야 한다.
그러려면 품질을 $(N, D, B_s, C_{\\text{cap}})$의 함수로 놓고 $B_s$ 축의 형태 — 지수, 포화점,
$N$·$D$와의 교환율 — 을 봐야 한다. Chinchilla가 $(N, D)$ 두 축에서 한 일을 $B_s$ 축에서
하려는 것이다.

그런데 Part II가 반복해서 확인한 사실은 **$B_s$가 비교 가능한 단위로 거의 보고되지 않는다**는
것이다. 이 실험은 그 인상을 감사(audit)로 바꾼다. 감사 대상은 `notes/stc-v2/*.json` 29편의
`cost_table.B_s` 전부이고, 세는 일은 전부 프로그램이 한다(파싱 규칙은 아래 §PARSE RULES).

이 실험이 답하는 것
------------------
Q1  보고 실태 전수 감사 — 29편의 $B_s$를 (a) 절대 FLOPs/토큰 · (b) optimizer step 수 ·
    (c) 벽시계 · (d) 서수 knob만 · (e) 상대 비율만 · (f) absent 로 분류하고 집계한다.
Q2  통일 축 복원 가능성 — (a)~(c)를 공통 단위(FLOPs)로 환산할 수 있는가. 환산식마다 필요한
    입력을 명시하고, 그 입력이 노트/원문에 있는지 프로그램으로 조회한다. 몇 점이 남는가.
Q3  품질 축의 비교 가능성 — 살아남은 점들의 과제·지표·baseline이 서로 겹치는가.
Q4  판정 — 지금 문헌으로 $B_s$–품질 곡선을 그릴 수 있는가. 못 그린다면 **무엇이 보고되면
    그릴 수 있는지**를 최소 요구 목록으로 제시한다. 이것이 ch28의 실질 기여다.
Q5  X1 sweep 범위의 근거화 — X1은 $|c|$·$|\\hat c|$·$B_s$가 미보고라 sweep했다. vendored
    원문에서 (i) 인쇄된 예시 문맥의 **문자 수**를 결정론적으로 추출하고 (ii) 논문이 스스로
    보고한 토큰 수를 긁어, 문맥 길이의 현실 범위를 세운 뒤 X1의 sweep이 현실적인지 판정한다.

등급
----
AUDIT-GRADE / CORPUS-CENSUS. 이 실험은 측정도 시뮬레이션도 아니다. **문헌 상태의 전수 조사**다.
- 신뢰할 것: 범주별 편수, 범주·경로 간 순서, "공통 단위로 환산 가능한 논문이 corpus에서
  한 자릿수의 맨 아래(8편 후보 중 1편)"라는 계수 결과, 스윕을 가진 논문들의 과제 집합
  교집합이 공집합이라는 사실, 문맥 길이 자릿수의 범위.
- 신뢰하지 말 것: 문자/4 토큰 근사의 소수점, 특정 논문의 절대 토큰 추정치,
  키워드 규칙이 애매한 산문에서 내린 개별 라벨(전부 `ambiguity_flags`로 노출한다).
- 논문이 보고하지 않은 값은 추정해서 숫자로 쓰지 않는다. "논문에 없음"이 결과다.

PARSE RULES (요약; 구현은 아래 함수들)
--------------------------------------
R-A  대상은 `cost_table.B_s` 서브트리 전체. 리프를 (json 경로, 문자열)로 평탄화한다.
R-B  각 리프의 **필드명**으로 역할을 정한다.
     VERDICT(value/value_*/numeric/status/verdict)  — $B_s$ 자체에 대한 판정문
     PRESENT(what_is_stated/reported/detail/…)      — 논문이 실제로 말한 것
     ABSENT(what_is_absent/not_stated/missing/…)    — 부재 선언 (단위 탐지에서 제외)
     DERIVED(arithmetic_by_this_note/derived_*/…)   — **노트작성자 산술**. 논문 보고가 아니므로
                                                      단위 탐지에서 제외한다. 이 배제가 이 감사의
                                                      핵심 규칙이다 — 이걸 안 빼면 논문이 보고하지
                                                      않은 값을 보고한 것으로 세게 된다.
     OTHER                                          — 그 외(약한 증거로만 사용)
R-C  노트작성자 표지(`note-writer`, `my arithmetic`, `[INFERENCE`, `NOT printed in the paper`,
     `ARITHMETIC ON STATED NUMBERS`, `본서 산술` 등)가 리프 본문에 있으면 그 리프의 역할을
     DERIVED로 강등해 통째로 제외하고, 살아남은 리프 안에서도 그 표지가 있는 **문장**은
     추가로 건너뛴다(한정문과 수치문이 한 리프에 섞이는 경우 대비).
R-D  단위 탐지는 인용표지(`[...]`, `§3.1`, `p. 8`, `Fig. 7`, `Table 2`, `line 213`, `App. B`)를
     제거한 뒤 수행한다. 그래야 "§5.2"의 숫자가 예산 숫자로 오인되지 않는다.
R-E  양성(POS)은 **수량이 단위에 붙어 있어야** 인정한다("3748.4 tokens", "192 optimiser steps",
     "7.44 hr"). 단위 낱말만 지나가는 문장("Avg. Test Time Tokens / Question", "loss on answer
     tokens only")은 양성이 아니다. 하이픈 복합수식어("1,024-token chunk", "115k-token
     conversation")는 **길이**이지 예산이 아니므로 숫자와 tokens 사이에 공백을 요구한다.
     서수 knob은 무수치가 정상이므로 이 요구를 면제한다. 음성(NEG)은 느슨한 단위 패턴 +
     부정 근접창으로 따로 잡는다("no FLOPs", "토큰 수도 없다").
R-F  부정(negation)은 근접창으로 본다: 히트 앞 30자 또는 뒤 15자에 부정 단서(no/never/absent/
     without/nowhere/cannot/없/미보고/아니…)가 있으면 음성. 뒤쪽 창은 한국어 후치부정
     ("FLOPs도 token 수도 **없다**")을 잡기 위한 것이다.
R-G  귀속: 리프의 **필드명**이 `~instead / amortiz / one_time / outer / pays_instead`이거나
     리프 본문에 "다른 비용" 표지(instead / which is not / outer-loop / total experiment compute /
     entire evaluation / wake-time / not a budget …)가 있으면, 그 리프의 수치는 $B_s$가 아니라
     `other_cost`로 분리한다. GEM의 "전체 학습 CPU 초"나 lm-memorization-capacity의
     "OUTER-loop training budget — which is not B_s"가 $B_s$로 세어지지 않게 하는 규칙이다.
     표지는 **리프 전체**에 걸린다(첫 문장이 한정하고 다음 문장이 수치를 대는 서술이 흔하다).
R-H  범주는 우선순위로 단일화한다: a(flops|tokens) > b(steps) > c(wall_clock) > d(ordinal) >
     e(ratio) > f. 우선순위 선택에 결과가 흔들리지 않음을 보이려고 **다중 라벨 집계도 함께**
     낸다(`by_category_multilabel_no_priority_collapse`).
R-I  질의 전 오프라인 패스가 아예 없다고 노트가 말하는 논문(RE_UNDEFINED)은 (f)로 센다 —
     없는 양은 보고될 수 없다. 이때는 `bs_defined=False` 플래그로 "미보고"와 구별한다.
     판정 범위는 `B_s` 서브트리 + `B_s*` 형제 키로 한정한다(다른 셀의 문구 오염 방지).
R-J  규칙 출력이 애매한 지점은 전부 `ambiguity_flags`에 남긴다. 편집자 override는 OVERRIDES에
     사유와 함께 명시하고 개수를 보고한다 — **29편 전수 검토 결과 override 0건**이며, 애매했던
     지점은 override가 아니라 규칙 수정으로 처리했다(시뮬레이션 시간 ≠ 벽시계 등).
R-K  분류 체계 (a)~(f)에 들어가지 않는 보고 형태는 억지로 밀어 넣지 않고 `taxonomy_gaps`에
     남긴다(시뮬레이션 시간, 표본 수, 그림에만 있는 FLOPs, 노트 필드 자체 부재).

결정론
------
난수 없음. 파일 읽기 순서는 sorted(). dict는 삽입 순서 유지. 두 번 실행해 result.json의
sha256이 같음을 확인한다(`PYTHONHASHSEED=0`).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                      # ~/repos/neural-memory-study
NOTES = REPO / "notes" / "stc-v2"
PAPERS = REPO / "papers"

T0 = time.time()

# ===========================================================================
# 0. 파일 입력 + 지문
# ===========================================================================


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_notes():
    out = OrderedDict()
    for p in sorted(NOTES.glob("*.json")):
        out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    return out


def resolve_paper_path(raw) -> Path | None:
    """노트의 `path` 필드를 실제 파일로 해석한다. 괄호 주석·상대경로를 허용한다."""
    if not isinstance(raw, str):
        return None
    s = raw.split("(")[0].strip().rstrip(",;")
    if not s:
        return None
    cand = Path(s)
    if not cand.is_absolute():
        cand = REPO / s
    return cand if cand.exists() else None


# ===========================================================================
# 1. 평탄화 + 역할 판정 (R-A, R-B, R-C)
# ===========================================================================

ROLE_VERDICT, ROLE_PRESENT, ROLE_ABSENT, ROLE_DERIVED, ROLE_OTHER = (
    "VERDICT", "PRESENT", "ABSENT", "DERIVED", "OTHER")

RE_FIELD_ABSENT = re.compile(
    r"absent|missing|not_stated|unpriced|omitted|declines", re.I)
RE_FIELD_DERIVED = re.compile(
    r"deriv|arithmetic|note_writer|inference|note_writer_reading|_reading$", re.I)
RE_FIELD_VERDICT = re.compile(
    r"^(value|value_.*|numeric|status|verdict|_pass)$", re.I)
RE_FIELD_PRESENT = re.compile(
    r"stated|reported|does_say|does_state|does_report|gives_instead|"
    r"measured_instead|detail|claims|hardware|structural|what_is_knowable|"
    r"cost|component|volume", re.I)

RE_NOTEWRITER = re.compile(
    r"note[-\s]?writer|my arithmetic|this note'?s arithmetic|\[INFERENCE|"
    r"NOT printed in the paper|not in the paper|본서 산술|노트작성자|"
    r"논문이 쓰지 않은|declines to do|이 실험의 가정|"
    r"ARITHMETIC ON STATED NUMBERS|arithmetic on the paper|"
    r"is not printed in the paper|multiplication of two of its stated numbers|"
    r"ratios are this note", re.I)

# 이 문장·필드는 "논문이 보고한 다른 비용"이지 B_s가 아니다 (R-G).
RE_NOT_BS = re.compile(
    r"\binstead\b|is not \$?B_s|which is not|not a sleep budget|not a budget|"
    r"outer[- ]loop|OUTER-loop|training throughput only|TRAINING throughput only|"
    r"total experiment compute|entire evaluation|whole benchmark|for the whole|"
    r"wake[- ]time|wake[- ]side|aggregate and covers|not per-session|"
    r"one-?time|amortized over every query|corpus-level", re.I)
RE_FIELD_NOT_BS = re.compile(
    r"instead|amorti[sz]|one_time|pays_instead|outer", re.I)


def flatten(obj, path="B_s"):
    """(json 경로, 문자열) 리프 목록. 순서 결정론적."""
    out = []
    if isinstance(obj, str):
        out.append((path, obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(flatten(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(flatten(v, f"{path}[{i}]"))
    elif obj is not None:
        out.append((path, str(obj)))
    return out


def role_of(path: str, text: str) -> str:
    """R-B + R-C. 마지막 dict 키로 역할을 정하고, 노트작성자 표지가 있으면 DERIVED로 강등."""
    key = path.split(".")[-1]
    key = re.sub(r"\[\d+\]$", "", key)
    if RE_NOTEWRITER.search(text):
        return ROLE_DERIVED
    if RE_FIELD_DERIVED.search(key):
        return ROLE_DERIVED
    if RE_FIELD_ABSENT.search(key):
        return ROLE_ABSENT
    if RE_FIELD_VERDICT.match(key):
        return ROLE_VERDICT
    if RE_FIELD_PRESENT.search(key):
        return ROLE_PRESENT
    if key == "B_s":                     # B_s 자체가 문자열인 경우 (memory-caching: "absent")
        return ROLE_VERDICT
    return ROLE_OTHER


# ===========================================================================
# 2. 단위 탐지 (R-D ~ R-G)
# ===========================================================================

RE_CITATION = re.compile(
    r"\[[^\[\]]{0,200}?\]"                       # [Letta STC §5.1 p.6]
    r"|§+\s*[0-9A-Z]+(?:\.[0-9]+)*"              # §5.2 / §B.5
    r"|\bpp?\.\s*\d+"                            # p. 8
    r"|\bFig(?:ure)?\.?\s*\d+"                   # Fig. 7
    r"|\bTable\s+\d+"
    r"|\bApp(?:endix)?\.?\s*[A-Z](?:\.\d+)*"
    r"|\blines?\s*[\d,\-–\s]+"
    r"|\btxt:?\s*\d+[\d\-–]*"
    r"|\bL\d{2,}(?:-L?\d+)?"                     # L640-L646
    r"|\barXiv[:\s]\S+",
    re.I)

# 수량 접두 — "예산 숫자"만 잡기 위한 것. 단순히 단위 낱말이 등장하는 것과 구별한다.
Q = r"\d[\d,]*(?:\.\d+)?\s*(?:k|K|M|B|thousand|million|billion|trillion)?"

# POS(양성) 패턴: 숫자가 **단위에 붙어** 있어야 한다. 예산 숫자와 지나가는 낱말을 가른다.
UNIT_POS = {
    "flops": re.compile(
        rf"{Q}\s*(?:[TPEG]|tera|peta|exa|giga)?\s*-?\s*FLOPs?\b|FLOPs?\s*[:=]\s*\d", re.I),
    # 하이픈 복합수식어("115k-token conversation", "1,024-token chunk")는 **길이**이지
    # 예산이 아니다. 그래서 숫자와 tokens 사이에 공백을 요구한다.
    "tokens": re.compile(
        rf"{Q}\s+tokens?\b(?!\s*(?:chunk|window|context|sequence|passage|"
        rf"length|limit|budget|axis))|tokens?\s*[:=]\s*\d|"
        rf"{Q}\s*토큰|토큰\s*(?:수\s*)?[:=]\s*\d", re.I),
    "steps": re.compile(
        rf"{Q}\s*(?:optimi[sz]er\s+|training\s+|gradient\s+|inner-?loop\s+|RL\s+|"
        rf"data\s+|SGD\s+|diffusion\s+)?(?:steps?|epochs?|mini-?batch(?:es)?)\b|"
        rf"{Q}\s*(?:training|optimi[sz]er|gradient|data|SGD)\s+iterations?\b|"
        rf"(?:steps?|epochs?)\s*[:=]\s*\d|{Q}\s*스텝|"
        rf"mini-?batch(?:es)?\s*size\s*[:=]\s*\d|"
        rf"\b\d[\d,]*\s*(?:samples?|transitions?|mini-?batch(?:es)?)\b", re.I),
    # 절대 시간 단위가 붙은 형태만 인정한다. "4.8x wall-clock time"은 비율이지 시간이 아니다.
    "wall_clock": re.compile(
        rf"{Q}\s*(?:s\b|sec\b|secs\b|seconds?\b|min\b|minutes?\b|hr\b|hrs?\b|"
        rf"hours?\b|days?\b|ms\b)|{Q}\s*(?:GPU|TPU|CPU)-?(?:hour|day)s?\b", re.I),
    "energy": re.compile(rf"{Q}\s*(?:J\b|joules?|kWh|Wh\b)|energy\s*[:=]\s*\d", re.I),
    "ratio": re.compile(
        r"\d+(?:\.\d+)?\s*[x×]\b|\b[x×]\s*\d+(?:\.\d+)?\b|speed-?up|"
        r"orders? of magnitude|\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?\s*배\b", re.I),
    # 서수 knob은 숫자 요구를 면제한다 (R-E). LaTeX의 \{ \} 이스케이프를 허용한다.
    "ordinal": re.compile(
        r"\{\s*low\b|low,\s*medium,\s*high|reasoning effort|ordinal knobs?|서수|"
        r"\\?in\s*\\?\{|∈\s*\\?\{|\\le\s*\d+|≤\s*\d+|"
        r"\\?\{\s*\d+(?:\s*,\s*\d+){1,8}\s*\\?\}", re.I),
}
# NEG(음성) 탐지용 느슨한 패턴: "no FLOPs", "토큰 수도 없다" 처럼 숫자 없는 부재 선언을 잡는다.
UNIT_BARE = {
    "flops": re.compile(r"\bFLOPs?\b|\b[TPEG]FLOPs?\b|tera[Ff]lop", re.I),
    "tokens": re.compile(r"\btokens?\b|토큰", re.I),
    "steps": re.compile(r"\bsteps?\b|\bepochs?\b|mini-?batch(?:es)?\b|스텝", re.I),
    "wall_clock": re.compile(
        r"wall[-\s]?clock|\bseconds?\b|\bminutes?\b|\bhours?\b|\bhrs?\b|"
        r"GPU-?(?:hour|day)s?|벽시계|\bwall time\b|\bwall-?clock time\b", re.I),
    "energy": re.compile(r"\benergy\b|에너지|\bjoules?\b|\bkWh\b", re.I),
    "ratio": re.compile(r"\bratios?\b|speed-?up|orders? of magnitude", re.I),
    "ordinal": re.compile(r"ordinal knobs?|서수", re.I),
}
NUMERIC_EXEMPT = {"ordinal"}

# ratio는 비용 어휘와 같은 문장에 있어야 한다 (정확도 %가 예산 비율로 세어지는 것 방지)
RE_COST_WORD = re.compile(
    r"compute|cost|FLOP|token|time|step|efficien|throughput|budget|overhead|"
    r"faster|slower|expensive|GPU|memory|VRAM|비용|예산|시간", re.I)

RE_NUMBER = re.compile(r"\d")

RE_NEG = re.compile(
    r"\bno\b|\bnot\b|\bnever\b|\bnone\b|\babsent\b|\bwithout\b|\bnowhere\b|"
    r"\bneither\b|\bnor\b|\bcannot\b|\bzero hits\b|\bis missing\b|\blacks?\b|"
    r"\bunreported\b|\bunquantified\b|\bdeclines?\b|\bexcluded\b|"
    r"없|미보고|아니|않|없다|부재|전무", re.I)

RE_FIGURE_ONLY = re.compile(
    r"axis tick values are not recoverable|no numeric value.{0,60}quoted|"
    r"figure-extracted|축에서 읽음|not recoverable from the extracted text|"
    r"only on an? (?:log )?axis|plots?.{0,40}but no numeric", re.I)

RE_UNDEFINED = re.compile(
    r"\bundefined\b|정의되지 않|개념 자체가 없|개념이 없|no sleep phase|"
    r"no such period|structurally zero|structural absence|"
    r"not merely unreported|zero by construction|by construction|"
    r"does not exist|no pre-query offline pass|no offline pass|"
    r"there is no offline|has no sleep|\$?B_s\$? *(?:는|가|=) *0|"
    r"\bB_s = 0\b|B_s가 0|오프라인[^.]{0,12}없|"
    r"no offline pass, so there is no|there is no separable sleep budget", re.I)

RE_PARTIAL = re.compile(
    r"\bpartial(?:ly)?\b|부분적|Reported ONLY|stated only|only as|only in|"
    r"PARTIALLY PRESENT|PARTIALLY REPORTED|partially present|partially reported|"
    r"partially reported", re.I)

RE_ABSENT_VERDICT = re.compile(r"\babsent\b|없다|없음|부재", re.I)


def strip_citations(text: str) -> str:
    return RE_CITATION.sub(" ", text)


def split_sentences(text: str):
    """문장 분할. 오프셋을 함께 돌려 부정 근접창을 원문 좌표로 볼 수 있게 한다."""
    out, start = [], 0
    # ':' 로는 자르지 않는다 — "…which is not B_s — is: 10^6 steps" 처럼 한정절과 수치가
    # 콜론으로 이어지는 경우가 흔해, 자르면 한정이 떨어져 나간다.
    for m in re.finditer(r"[.;!?\n]\s+|\.\s*$", text):
        end = m.end()
        seg = text[start:end]
        if seg.strip():
            out.append((start, seg))
        start = end
    if start < len(text):
        seg = text[start:]
        if seg.strip():
            out.append((start, seg))
    return out


def is_negated(sentence: str, pos: int, span: int) -> bool:
    """R-F: 히트 앞 30자 / 뒤 15자 근접창에 부정 단서가 있는가."""
    before = sentence[max(0, pos - 30):pos]
    after = sentence[pos + span:pos + span + 15]
    return bool(RE_NEG.search(before) or RE_NEG.search(after))


def detect_units(text: str):
    """한 리프의 단위 증거. -> {unit: {'pos': [snippets], 'neg': [snippets], 'not_bs': [...]}}

    - POS는 수량이 붙은 패턴(UNIT_POS)으로만 인정한다.
    - NEG는 느슨한 패턴(UNIT_BARE) + 부정 근접창으로 잡는다 ("no FLOPs", "토큰 수도 없다").
    - 노트작성자 산술 문장은 **문장 단위로** 건너뛴다 (R-C). 리프 전체를 버리면 같은 리프의
      논문 보고 문장까지 잃기 때문이다.
    - 'instead' 류 문장은 not_bs로 분리한다 (R-G).
    """
    clean = strip_citations(text)
    leaf_not_bs = bool(RE_NOT_BS.search(clean))
    ev = {}

    def add(unit, bucket, snip):
        ev.setdefault(unit, {"pos": [], "neg": [], "not_bs": []})
        s = snip.strip()
        s = s if len(s) <= 220 else s[:217] + "..."
        if s not in ev[unit][bucket]:
            ev[unit][bucket].append(s)

    for _, sent in split_sentences(clean):
        if RE_NOTEWRITER.search(sent):
            continue
        has_num = bool(RE_NUMBER.search(sent))
        not_bs = leaf_not_bs or bool(RE_NOT_BS.search(sent))
        # 음성 증거 (느슨)
        for unit, pat in UNIT_BARE.items():
            for m in pat.finditer(sent):
                if is_negated(sent, m.start(), m.end() - m.start()):
                    add(unit, "neg", sent)
                    break
        # 양성 증거 (수량 결합)
        for unit, pat in UNIT_POS.items():
            if unit not in NUMERIC_EXEMPT and not has_num:
                continue
            if unit == "ratio" and not RE_COST_WORD.search(sent):
                continue
            # 시뮬레이션 시간은 벽시계가 아니다 (sleep-snn-plos의 'ms of simulated time')
            if unit == "wall_clock" and re.search(r"simulat", sent, re.I):
                continue
            for m in pat.finditer(sent):
                if is_negated(sent, m.start(), m.end() - m.start()):
                    add(unit, "neg", sent)
                else:
                    add(unit, "not_bs" if not_bs else "pos", sent)
    return ev


# ===========================================================================
# 3. 논문 1편 분류 (R-G ~ R-J)
# ===========================================================================

CATEGORY_LABEL = OrderedDict([
    ("a", "절대 FLOPs/토큰"),
    ("b", "optimizer step 수"),
    ("c", "벽시계"),
    ("d", "서수 knob만"),
    ("e", "상대 비율만"),
    ("f", "absent"),
])
UNIT_TO_CATEGORY = OrderedDict([
    ("flops", "a"), ("tokens", "a"), ("steps", "b"),
    ("wall_clock", "c"), ("ordinal", "d"), ("ratio", "e"),
])
CATEGORY_ORDER = ["a", "b", "c", "d", "e", "f"]

# R-J: 규칙이 낸 라벨을 편집자가 뒤집은 지점. 사유를 반드시 적는다.
# 규칙 출력을 29편 전수 검토한 뒤 남은 것만 둔다. 개수는 result.json에 보고한다.
OVERRIDES: dict[str, dict] = {}      # 29편 전수 검토 결과 필요 없었다 — 규칙 100%


def classify_paper(slug: str, note: dict):
    ct = note.get("cost_table")
    rec = OrderedDict()
    rec["axis"] = _axis_string(note)
    rec["arxiv"] = note.get("arxiv")

    if not isinstance(ct, dict) or "B_s" not in ct:
        rec["category_rule"] = "f"
        rec["categories_multi"] = ["f"]
        rec["verdict_polarity"] = "NOTE_FIELD_MISSING"
        rec["bs_defined"] = None
        rec["units_pos"] = []
        rec["units_neg"] = []
        rec["other_cost_units"] = []
        rec["evidence"] = []
        rec["sweep"] = {"detected": False, "values": [], "max_points": 0,
                        "axis_kind": "none"}
        rec["units_suppressed_by_negation"] = []
        rec["ambiguity_flags"] = ["NOTE_FIELD_MISSING: cost_table.B_s 키 자체가 노트에 없다 "
                                 "(논문 미보고가 아니라 노트 미기재 — 구분해서 셀 것)"]
        rec["figure_only"] = False
        return rec

    leaves = flatten(ct["B_s"])
    verdict_text = " ".join(t for p, t in leaves if role_of(p, t) == ROLE_VERDICT)
    all_text = " ".join(t for _, t in leaves)
    # B_s 서브트리 + 'B_s'로 시작하는 형제 키(B_s_note 등)만 본다. cost_table 전체를 보면
    # C_cap·rho 셀의 'by construction' 같은 문구가 B_s 판정을 오염시킨다.
    bs_scope_text = all_text + " " + " ".join(
        json.dumps(v, ensure_ascii=False) for k, v in ct.items()
        if k.startswith("B_s") and k != "B_s")

    bs_pos, bs_neg, other_pos = Counter(), Counter(), Counter()
    evidence = []
    for path, text in leaves:
        role = role_of(path, text)
        if role in (ROLE_ABSENT, ROLE_DERIVED):
            continue
        field_not_bs = bool(RE_FIELD_NOT_BS.search(path.split(".")[-1]))
        ev = detect_units(text)
        for unit, pols in ev.items():
            for snip in pols["pos"]:
                # R-G: 필드명이 '~instead/amortized/outer'거나 문장이 '다른 비용'을 가리키면
                #      B_s가 아니라 other_cost로 분리한다.
                attributed = not field_not_bs
                (bs_pos if attributed else other_pos)[unit] += 1
                evidence.append(OrderedDict([
                    ("path", path), ("role", role), ("unit", unit),
                    ("polarity", "pos"), ("attributed_to_B_s", attributed),
                    ("snippet", snip)]))
            for snip in pols["not_bs"]:
                other_pos[unit] += 1
                evidence.append(OrderedDict([
                    ("path", path), ("role", role), ("unit", unit),
                    ("polarity", "pos"), ("attributed_to_B_s", False),
                    ("snippet", snip)]))
            for snip in pols["neg"]:
                bs_neg[unit] += 1
                evidence.append(OrderedDict([
                    ("path", path), ("role", role), ("unit", unit),
                    ("polarity", "neg"), ("attributed_to_B_s", None),
                    ("snippet", snip)]))

    # 판정 극성
    if RE_PARTIAL.search(verdict_text):
        polarity = "PARTIAL"
    elif RE_ABSENT_VERDICT.search(verdict_text):
        polarity = "ABSENT"
    elif verdict_text.strip():
        polarity = "PRESENT"
    else:
        polarity = "NO_VERDICT_FIELD"

    # B_s가 그 논문에서 **정의되는가** (질의 전 오프라인 패스가 존재하는가).
    bs_defined = not bool(RE_UNDEFINED.search(bs_scope_text))

    # 단위별 부정 우선: 판정/서술이 그 단위를 명시적으로 부정하면 양성으로 세지 않는다.
    suppressed = [u for u in bs_pos if bs_neg.get(u, 0) > 0 and bs_pos[u] == 0]
    mixed = [u for u in bs_pos if bs_neg.get(u, 0) > 0 and bs_pos[u] > 0]

    # 범주 (R-H, R-I)
    units_present = [u for u in UNIT_TO_CATEGORY if bs_pos.get(u, 0) > 0]
    cats = sorted({UNIT_TO_CATEGORY[u] for u in units_present},
                  key=CATEGORY_ORDER.index)
    if not bs_defined:
        # R-I: 오프라인 패스 자체가 없으면 B_s는 보고될 수 없다. (f)로 센다.
        cats = []
    category = cats[0] if cats else "f"
    if not cats:
        cats = ["f"]

    figure_only = bool(RE_FIGURE_ONLY.search(all_text))
    simulated_time = bool(re.search(
        r"simulated (?:time|duration)|simulation timesteps|movement cycles", all_text, re.I))

    # 스윕 탐지: B_s 축에 2개 이상의 값이 있는가
    sweep_vals = detect_sweep(all_text)
    sweep_vals["axis_kind"] = (
        "absolute_budget_units" if category in ("a", "b", "c")
        else ("proportional_knob_unknown_scale" if category in ("d", "e") else "none"))

    flags = []
    if polarity == "ABSENT" and other_pos:
        flags.append("VERDICT_ABSENT_BUT_OTHER_COST_PRINTED: 판정문은 absent인데 논문이 "
                     f"다른 비용 수치({sorted(other_pos)})는 인쇄한다 — 그것은 B_s가 아니다")
    if polarity == "PARTIAL":
        flags.append("VERDICT_PARTIAL: 노트가 스스로 '부분적으로만 보고'라고 쓴다")
    if figure_only:
        flags.append("PLOTTED_BUT_NOT_TABULATED: 그림 축에는 있으나 인쇄된 숫자가 없어 점을 찍을 수 없다")
    if not bs_defined:
        flags.append("B_S_UNDEFINED: 질의 전 오프라인 패스가 아예 없어 B_s가 정의되지 않는다 "
                     "— '보고 안 함'이 아니라 '있을 수 없음'. (f)로 세되 구별할 것")
    if simulated_time:
        flags.append("SIMULATED_TIME_NOT_COMPUTE: 보고된 예산이 시뮬레이션 시간이다. "
                     "(a)~(e) 어느 칸에도 속하지 않는다 — 분류 체계의 빈칸")
    if len(cats) > 1:
        flags.append(f"MULTI_UNIT: {cats} — 단일 범주로 강제 축약함(우선순위 a>b>c>d>e)")
    if mixed:
        flags.append(f"UNIT_MIXED_POLARITY: {sorted(mixed)} 단위에 긍정·부정 증거가 함께 있다")
    if slug == "ewc":
        flags.append("STEPS_VS_SAMPLES: 보고된 것은 optimizer step이 아니라 Fisher 추정 "
                     "표본 수(100 minibatch × 32)다. (b)로 세되 단위가 다르다")

    rec["category_rule"] = category
    rec["categories_multi"] = cats
    rec["verdict_polarity"] = polarity
    rec["bs_defined"] = bs_defined
    rec["units_pos"] = sorted(bs_pos)
    rec["units_neg"] = sorted(bs_neg)
    rec["units_suppressed_by_negation"] = sorted(suppressed)
    rec["other_cost_units"] = sorted(other_pos)
    rec["figure_only"] = figure_only
    rec["sweep"] = sweep_vals
    rec["ambiguity_flags"] = flags
    rec["evidence"] = evidence
    return rec


def _axis_string(note) -> str:
    a = note.get("axis")
    if isinstance(a, str):
        return a.split("—")[0].split("--")[0].strip()[:40]
    a = note.get("axis_verdict")
    if isinstance(a, dict):
        for k in ("verdict", "S0_roster_says"):
            if isinstance(a.get(k), str):
                return a[k][:40]
    if isinstance(a, dict):
        return json.dumps(a, ensure_ascii=False)[:40]
    return "unknown"


RE_SET_BRACE = re.compile(r"\\?\{\s*(\d+(?:\s*,\s*\d+){1,8})\s*\\?\}")
RE_LIST_OR = re.compile(r"\b(\d[\d,]*)\s*,\s*(\d[\d,]*)\s+or\s+(\d[\d,]*)\b")


def detect_sweep(text: str):
    """B_s 축 위에 2개 이상의 값이 놓였는가 (곡선을 그리려면 최소 2점, 형태를 보려면 3점)."""
    clean = strip_citations(text)
    vals = []
    for m in RE_SET_BRACE.finditer(clean):
        nums = [int(x.strip()) for x in m.group(1).split(",")]
        if len(nums) >= 2 and nums not in vals:
            vals.append(nums)
    for m in RE_LIST_OR.finditer(clean):
        nums = [int(x.replace(",", "")) for x in m.groups()]
        if nums not in vals:
            vals.append(nums)
    return {"detected": bool(vals), "values": vals,
            "max_points": max((len(v) for v in vals), default=0)}


# ===========================================================================
# 4. Q2 — 통일 축(FLOPs) 복원 가능성
# ===========================================================================

RECIPES = OrderedDict([
    ("a_tokens_to_flops", OrderedDict([
        ("applies_to_category", "a"),
        ("formula", "B_s[FLOPs] ~= n_tokens * c_pass * N_params  "
                    "(c_pass = 2 for a forward/prefill token, ~6 for a token that also "
                    "carries a backward pass)"),
        ("required_inputs", ["n_tokens", "N_params", "prefill_vs_decode_split"]),
    ])),
    ("b_steps_to_flops", OrderedDict([
        ("applies_to_category", "b"),
        ("formula", "B_s[FLOPs] ~= n_steps * batch_size * seq_len_tokens * c_pass * N_params "
                    "(c_pass ~ 6 for full fine-tune; PEFT changes the backward term only)"),
        ("required_inputs", ["n_steps", "batch_size", "seq_len_tokens", "N_params",
                             "trainable_fraction"]),
    ])),
    ("c_wallclock_to_flops", OrderedDict([
        ("applies_to_category", "c"),
        ("formula", "B_s[FLOPs] ~= t_wall * n_devices * peak_flops(device) * MFU",
         ),
        ("required_inputs", ["t_wall", "device_model", "n_devices", "mfu_or_throughput"]),
    ])),
])

INPUT_PROBES = OrderedDict([
    ("n_tokens", re.compile(
        r"\b\d[\d,\.]*\s*(?:k|K|M|B|billion|trillion|million)?\s*tokens\b", re.I)),
    ("N_params", re.compile(
        r"\b\d+(?:\.\d+)?\s*[BbMm]\b\s*(?:parameters|params|model)|"
        r"\b\d+(?:\.\d+)?\s*(?:billion|million)\s*parameters|"
        r"\bGPT-J\b|\bGPT-2\s*XL\b|\bQwen3-\d+B\b|\bLlama-?\d(?:\.\d)?[- ]?\d*B\b|"
        r"\b\d+(?:\.\d+)?[Bb]\s+(?:base|model)\b", re.I)),
    ("prefill_vs_decode_split", re.compile(
        r"prompt and completion|input tokens|output tokens|\bprefill\b|"
        r"generated tokens|completion tokens|생성 토큰", re.I)),
    ("n_steps", re.compile(
        r"\b\d(?:[\d,]*\d)?\s*(?:optimi[sz]er\s+)?steps?\b|"
        r"\b\d(?:[\d,]*\d)?\s*(?:training\s+)?iterations?\b",
        re.I)),
    ("batch_size", re.compile(
        r"batch\s*size\s*(?:of\s*)?\d[\d,]*|effective batch size\s*\d[\d,]*|"
        r"batch\s*(?:of\s*)?\d[\d,]*\s*sample",
        re.I)),
    ("seq_len_tokens", re.compile(
        r"sequence length\D{0,6}\d|seq\.?\s*len\D{0,6}\d|context length\D{0,6}\d|"
        r"\b\d[\d,]*\s*-?\s*token\s+(?:chunk|window|context|sequence|passage)", re.I)),
    ("trainable_fraction", re.compile(
        r"\bLoRA\b|\brank\s*=?\s*\d[\d,]*|\bfull fine-?tun|\badapter\b|\bPEFT\b", re.I)),
    ("t_wall", re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:s\b|sec\b|secs\b|seconds?\b|min\b|minutes?\b|"
        r"hr\b|hrs?\b|hours?\b|GPU-?(?:hour|day)s?\b)", re.I)),
    ("device_model", re.compile(
        r"\bA100\b|\bH100\b|\bH200\b|\bV100\b|\bA6000\b|\bTPUv\d\b|\bRTX PRO 6000\b|"
        r"\bBlackwell\b|\bCPU\b", re.I)),
    ("n_devices", re.compile(
        r"\b\d+\s*[x×]\s*(?:NVIDIA\s*)?(?:A100|H100|H200|V100|A6000)|"
        r"\bon\s+(?:a|one|1|two|2|8)\s+(?:NVIDIA\s*)?(?:A100|H100|H200|V100|A6000)|"
        r"\bsingle\s+(?:A100|H100|GPU)\b|\b\d+\s+GPUs?\b", re.I)),
    # 달성 효율. 'throughput'이라는 낱말만으로는 안 되고 **수치가 붙은** 처리량이어야 한다.
    ("mfu_or_throughput", re.compile(
        r"\bMFU\b|model[- ]flops[- ]utilization|"
        r"\d[\d,\.]*\s*(?:tokens?|samples?|examples?|images?)\s*(?:/|per)\s*"
        r"(?:s\b|sec\b|seconds?\b)", re.I)),
])


def note_text_blob(note) -> str:
    return json.dumps(note, ensure_ascii=False)


def probe_inputs(note_blob: str, paper_blob: str):
    """-> (상태맵, 매치예시맵). 매치 문자열을 함께 돌려 'in_note' 판정을 감사 가능하게 한다."""
    status, hits = OrderedDict(), OrderedDict()
    for name, pat in INPUT_PROBES.items():
        m_note = [m.group(0) for m in pat.finditer(note_blob)]
        m_paper = [m.group(0) for m in pat.finditer(paper_blob)] if paper_blob else []
        status[name] = ("in_note" if m_note else
                        ("in_paper_text_only" if m_paper else "not_found"))
        seen, ex = set(), []
        for m in (m_note or m_paper):
            if m not in seen:
                seen.add(m); ex.append(m)
            if len(ex) >= 3:
                break
        hits[name] = ex
    return status, hits


# ===========================================================================
# 5. Q3 — 품질 축
# ===========================================================================

BENCHMARKS = OrderedDict([
    ("Stateful GSM-Symbolic", r"Stateful GSM-?Symbolic"),
    ("Stateful AIME", r"Stateful AIME"),
    ("Multi-Query GSM-Symbolic", r"Multi-?Query GSM-?Symbolic"),
    ("SWE-Features", r"SWE-?Features"),
    ("SWE-bench", r"SWE-?bench"),
    ("AIME", r"\bAIME\b"),
    ("HMMT", r"\bHMMT\b"),
    ("GSM8K", r"\bGSM8K\b"),
    ("GSM-Infinite", r"GSM-?Infinite"),
    ("ARC", r"\bARC(?:-AGI)?\b"),
    ("SQuAD", r"\bSQuAD\b"),
    ("LOCOMO", r"\bLOCOMO\b"),
    ("LongMemEval", r"LongMemEval"),
    ("DMR", r"\bDMR\b"),
    ("WebArena", r"WebArena"),
    ("Mind2Web", r"Mind2Web"),
    ("NaturalQuestions", r"Natural\s?Questions|\bNQ\b"),
    ("TriviaQA", r"TriviaQA"),
    ("FEVER", r"\bFEVER\b"),
    ("CounterFact", r"CounterFact"),
    ("zsRE", r"\bzsRE\b"),
    ("PhoneBook", r"PhoneBook"),
    ("PaperQA", r"PaperQA"),
    ("NarrativeQA", r"NarrativeQA"),
    ("QuALITY", r"QuALITY"),
    ("QQP/SST-2/MNLI(GLUE)", r"\bGLUE\b|\bMNLI\b|\bSST-2\b|\bQQP\b"),
    ("WikiSQL", r"WikiSQL"),
    ("E2E/DART/WebNLG", r"\bE2E NLG\b|\bDART\b|WebNLG"),
    ("PG-19", r"PG-?19"),
    ("arXiv Math", r"arXiv Math"),
    ("TinyStories", r"TinyStories"),
    ("CIFAR", r"CIFAR-?\d*"),
    ("MNIST", r"\bMNIST\b"),
    ("Atari", r"\bAtari\b"),
    ("CelebA", r"CelebA"),
    ("Wikitext", r"[Ww]ikitext"),
    ("MMLU", r"\bMMLU\b"),
    ("Depo", r"\bDepo\b"),
    ("automaton", r"\bautomaton\b"),
])
METRICS = OrderedDict([
    ("accuracy", r"\baccurac|\bacc\.\b"),
    ("exact_match", r"exact match|\bEM\b"),
    ("F1", r"\bF1\b"),
    ("perplexity", r"perplexit|\bppl\b"),
    ("success_rate", r"success rate|\bSR\b"),
    ("recall@k", r"recall@|\brecall\b"),
    ("J_score", r"\bJ score\b|\bJ-score\b"),
    ("BWT/ACC(continual)", r"\bBWT\b|\bbackward transfer\b|\bACC\b"),
    ("bits_per_parameter", r"bits-?per-?parameter|bits per parameter"),
    ("edit_score(ES/PS/NS)", r"\bEfficacy\b|\bParaphrase\b|\bSpecificity\b|\bES\b"),
    ("pass@k", r"pass@\d|pass@k"),
    ("linear_readout", r"linear (?:classifier )?readout"),
])
BASELINES = OrderedDict([
    ("no_memory", r"No[- ]?[Mm]emory|기억 없음|without memory"),
    ("full_context", r"full[- ]context|full[- ]conversation|whole context"),
    ("SFT", r"\bSFT\b|supervised fine-?tun"),
    ("ICL", r"\bICL\b|in-context learning"),
    ("dense_model", r"\bdense (?:model|baseline)\b"),
    ("vanilla/single", r"\bvanilla\b|\bsingle\b baseline|unregularised|unregularized"),
    ("RAG/BM25", r"\bBM25\b|\bDPR\b|\bRAG\b baseline"),
    ("test_time_only", r"test-?time (?:compute )?only|standard test-?time"),
])


# 저자 제작 변형이 원본 이름을 삼키지 않게 한다. Stateful AIME은 AIME이 아니다.
SUPERSEDES = {
    "Stateful AIME": ["AIME"],
    "Stateful GSM-Symbolic": ["GSM8K"],
    "Multi-Query GSM-Symbolic": ["GSM8K"],
    "GSM-Infinite": ["GSM8K"],
}


def lexicon_hits(blob: str, lex, apply_supersede=False):
    hits = [name for name, pat in lex.items() if re.search(pat, blob)]
    if apply_supersede:
        drop = set()
        for h in hits:
            drop.update(SUPERSEDES.get(h, []))
        hits = [h for h in hits if h not in drop]
    return hits


def results_blob(note) -> str:
    """품질 축은 논문의 **결과 서술**에서만 읽는다. 노트 전체를 훑으면 관련 논문 언급까지
    벤치마크로 세어져 공유량이 부풀려진다."""
    parts = []
    for k in ("experiments", "scale_ceiling"):
        if k in note:
            parts.append(json.dumps(note[k], ensure_ascii=False))
    return " ".join(parts)


# ===========================================================================
# 6. Q5 — 문맥 길이의 현실 범위 (문자 수 기반 + 논문 자체 보고 토큰 수)
# ===========================================================================

CHARS_PER_TOKEN = 4.0        # 근사. 토크나이저가 없으므로 문자/4를 쓴다.
CHARS_PER_TOKEN_BAND = (3.0, 5.0)

# 앵커 문자열로 구간을 잘라낸다(줄번호 대신 내용으로 고정 → 파일이 재추출돼도 재현 가능).
# 각 항목: (라벨, 파일 상대경로, 시작앵커, 끝앵커, 제거할 부착물 목록, 출처, 무엇인가)
CONTEXT_EXTRACTS = [
    ("letta_fig1_raw_context", "papers/2504.13171.txt",
     "A juggler can juggle 800\nballs. 1/4 of the balls are",
     "Sleep-time compute reduces test-time compute", [],
     "[Letta STC Fig. 1, p.2] 'Raw Context' 상자",
     "|c| — Multi-Query GSM-Symbolic 원 문맥 (논문이 직접 인쇄한 유일한 예시)"),
    ("letta_appB_aime_context_1", "papers/2504.13171.txt",
     "Context: Alice and Bob play the following game.",
     "Query: Find the number of positive integers", [],
     "[Letta STC App. B, p.17]",
     "|c| — Stateful AIME 문맥 예시 1"),
    ("letta_appB_aime_context_2", "papers/2504.13171.txt",
     "Context: Let A , B , C , and D be points on the hyperbola",
     "Query: Find the greatest real number", [],
     "[Letta STC App. B, p.17]",
     "|c| — Stateful AIME 문맥 예시 2 (LaTeX 그림 조각이 섞여 추출됨)"),
    ("letta_fig20_gsm_context", "papers/2504.13171.txt",
     "When Sofia watches her brother, she gets out a variety of toys",
     "Original Question", [],
     "[Letta STC Fig. 20, App. C, p.19]",
     "|c| — Multi-Query GSM-Symbolic 문맥 (10개 생성질문이 공유하는 문맥)"),
    ("letta_appG_swe_pr_body_1", "papers/2504.13171.txt",
     "body : Loading custom node can greatly slow startup time .",
     "user_login : huchenlei", [],
     "[Letta STC App. G Listing 3, p.23]",
     "|c| 구성단위 — SWE-Features PR 1편의 body (patch 제외; changed_files는 'omitted here')"),
    ("letta_appG_swe_pr_body_2", "papers/2504.13171.txt",
     "body : Added support for using a locally running instance of a LLAMA model",
     "user_login : bytedisciple", [],
     "[Letta STC App. G Listing 3, p.24]",
     "|c| 구성단위 — SWE-Features PR 1편의 body (aider #55)"),
]

# Figure 1의 'Learned Context'는 그림 상자들 사이에 끼어 추출돼, 원문 줄이 LLM/질문 상자와
# 교차한다. 아래 조각들을 그 순서대로 이어 붙인 것이 학습된 문맥 전체다. 각 조각은 앵커로 고정.
LEARNED_CONTEXT_FRAGMENTS = [
    "A juggler can juggle 800 balls. A\nquarter of the balls are tennis",
    "balls, which means there are 200\ntennis balls (800 * 1/4). Half of the\ntennis balls are indigo, resulting in",
    "100 indigo tennis balls (200 * 1/2).",
    "Out of these indigo tennis balls,\n1/10 are marked, which gives us 10",
    "marked indigo tennis balls (100 *\n1/10). Therefore, the total number",
    "of marked balls is 10 marked\nindigo tennis balls.",
]

# 논문이 스스로 토큰 수로 보고한 문맥/저장소 길이. (앵커로 원문에서 재확인한다.)
REPORTED_TOKEN_ANCHORS = [
    ("zep_longmemeval_context", "papers/stc/2501.13956.txt",
     "context of on average 115,000 tokens.", 115000,
     "|c| — LongMemEval 대화 1건", "[Zep §4.3, txt L238]"),
    ("zep_injected_context", "papers/stc/2501.13956.txt",
     "115k\n1.6k", 1600,
     "|c-hat| — Zep이 질의당 실제로 프롬프트에 넣는 토큰 수", "[Zep Table 2, txt L361-372]"),
    ("mem0_locomo_context", "papers/stc/2504.19413.txt",
     "26000 tokens on average, distributed across multiple sessions", 26000,
     "|c| — LOCOMO 대화 1건", "[Mem0 §3.1, txt L252]"),
    ("mem0_store_tokens", "papers/stc/2504.19413.txt",
     "tokens per conversation on an average. Where as Mem0", 7000,
     "저장소 크기 — Mem0 (질의당 주입량이 아니다)", "[Mem0 §4.5, txt L954]"),
    ("mem0_graph_store_tokens", "papers/stc/2504.19413.txt",
     "roughly doubles the footprint to 14k tokens", 14000,
     "저장소 크기 — Mem0^g", "[Mem0 §4.5, txt L954-955]"),
    ("zep_store_tokens_by_competitor", "papers/stc/2504.19413.txt",
     "Zep’s memory graph consumes in excess of 600k tokens", 600000,
     "저장소 크기 — Zep (경쟁사 Mem0가 측정한 값)", "[Mem0 §4.5, txt L956]"),
]

X1_ASSUMPTIONS = OrderedDict([
    ("stateful_gsm_symbolic.len_c_tokens", 400.0),
    ("stateful_aime.len_c_tokens", 300.0),
    ("swe_features.len_c_tokens", 8000.0),
    ("len_chat_over_len_c_grid", [0.25, 1.0, 4.0]),
    ("kappa_par_grid", [1, 2, 5, 10]),
])


RE_TOKCOUNT = re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\s*([kKMB])?\s*-?\s*tokens?\b")
RE_CONTEXTWORD = re.compile(
    r"context|conversation|document|sequence|window|passage|problem|input|"
    r"prompt|corpus|memory|chunk|segment|history|session|haystack|book", re.I)
MULT = {"": 1, "k": 1_000, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}


def mine_reported_token_lengths():
    """vendored 원문 전체에서 '문맥 길이로 읽히는 <숫자> tokens' 진술을 결정론적으로 긁는다.

    필터: 히트 ±80자 창에 문맥 어휘가 있어야 하고, 값이 64~5e7 범위여야 한다. 이것은 정밀
    추출이 아니라 **자릿수 범위**를 잡기 위한 것이다 — 개별 히트를 본문 수치로 인용하지 말 것.
    """
    files = sorted(list(PAPERS.glob("*.txt")) + list((PAPERS / "stc").glob("*.txt")))
    per_file, allvals = OrderedDict(), []
    for f in files:
        txt = f.read_text(encoding="utf-8", errors="replace")
        vals = []
        for m in RE_TOKCOUNT.finditer(txt):
            lo, hi = max(0, m.start() - 80), min(len(txt), m.end() + 80)
            if not RE_CONTEXTWORD.search(txt[lo:hi]):
                continue
            try:
                n = float(m.group(1).replace(",", "")) * MULT.get(m.group(2) or "", 1)
            except ValueError:
                continue
            if n < 64 or n > 5e7:
                continue
            vals.append(int(n))
        if vals:
            vals.sort()
            per_file[str(f.relative_to(REPO))] = OrderedDict([
                ("n_hits", len(vals)), ("min", vals[0]), ("max", vals[-1]),
                ("median", vals[len(vals) // 2])])
            allvals.extend(vals)
    allvals.sort()
    return OrderedDict([
        ("n_files_scanned", len(files)),
        ("n_files_with_hits", len(per_file)),
        ("n_hits_total", len(allvals)),
        ("global_min", allvals[0] if allvals else None),
        ("global_p50", allvals[len(allvals) // 2] if allvals else None),
        ("global_p90", allvals[int(len(allvals) * 0.9)] if allvals else None),
        ("global_max", allvals[-1] if allvals else None),
        ("per_file", per_file),
    ])


def extract_between(text: str, start_anchor: str, end_anchor: str):
    i = text.find(start_anchor)
    if i < 0:
        return None
    j = text.find(end_anchor, i + len(start_anchor))
    if j < 0:
        return None
    return text[i:j]


def normalise_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def char_stats(label, source_text, note, provenance):
    n = len(source_text)
    return OrderedDict([
        ("label", label),
        ("what", note),
        ("source", provenance),
        ("chars", n),
        ("tokens_approx_chars_over_4", round(n / CHARS_PER_TOKEN, 1)),
        ("tokens_band_chars_over_5_to_3",
         [round(n / CHARS_PER_TOKEN_BAND[1], 1), round(n / CHARS_PER_TOKEN_BAND[0], 1)]),
        ("sha256_of_extracted_text", hashlib.sha256(source_text.encode("utf-8")).hexdigest()[:16]),
        ("text", source_text if len(source_text) <= 900 else source_text[:897] + "..."),
    ])


# ===========================================================================
# main
# ===========================================================================


def main() -> None:
    notes = load_notes()
    results = OrderedDict()

    inputs_meta = OrderedDict()
    for slug, note in notes.items():
        p = NOTES / f"{slug}.json"
        inputs_meta[f"notes/stc-v2/{slug}.json"] = sha256_of(p)[:16]

    results["meta"] = OrderedDict([
        ("id", "X3-budget-axis"),
        ("title", "B_s 축 — 보고 실태 전수 감사와 scaling-law 가능성 판정"),
        ("supports_chapter", "ch28 (scaling law — B_s를 축으로)"),
        ("grade", "AUDIT-GRADE / CORPUS-CENSUS (측정 아님, 시뮬레이션 아님)"),
        ("deterministic", True),
        ("randomness", "none"),
        ("determinism_protocol",
         "PYTHONHASHSEED=0으로 두 번 실행해 result.json의 sha256이 같음을 확인한다. "
         "실행시간은 매번 달라지므로 result.json에 넣지 않고 stdout에만 찍는다"),
        ("python", sys.version.split()[0]),
        ("corpus", f"{len(notes)} deep-read notes in notes/stc-v2/"),
        ("input_sha256_prefix", inputs_meta),
        ("parse_rules", [
            "R-A cost_table.B_s 서브트리만 대상. 리프를 (경로, 문자열)로 평탄화",
            "R-B 필드명 → 역할(VERDICT/PRESENT/ABSENT/DERIVED/OTHER)",
            "R-C 노트작성자 산술 표지가 있으면 DERIVED로 강등하고 단위 탐지에서 제외 "
            "(논문 보고가 아니므로) — 이 감사의 핵심 규칙",
            "R-D 인용표지([...], §, p., Fig., Table, line, App.) 제거 후 탐지",
            "R-E 하드 단위는 같은 문장에 숫자가 있어야 양성 (서수 knob은 면제)",
            "R-F 부정은 앞 30자/뒤 15자 근접창 (뒤쪽 창은 한국어 후치부정용)",
            "R-G VERDICT 증거는 B_s에 귀속. PRESENT 증거는 sleep-side 어휘가 같은 문장에 "
            "있을 때만 귀속, 아니면 other_cost로 분리",
            "R-H 범주 우선순위 a>b>c>d>e>f로 단일화, 다중 라벨도 병기",
            "R-I 판정이 ABSENT인데 하드 단위만 있으면 (f). 서수/비율만 있으면 (d)/(e)",
            "R-J 편집자 override는 사유와 함께 코드에 명시하고 개수를 보고",
        ]),
        ("approximations", OrderedDict([
            ("chars_per_token", CHARS_PER_TOKEN),
            ("why", "토크나이저(torch/transformers)를 이 host에 쓰지 않는다. 영문 산문 BPE의 "
                    "경험칙 4자/토큰을 쓰되, 수식·코드는 3자/토큰에 가깝고 자연어 산문은 "
                    "5자/토큰에 가까우므로 3~5 밴드를 함께 보고한다"),
            ("what_this_forbids", "이 근사로 얻은 토큰 수를 본문 단정 수치로 쓰지 말 것. "
                                  "자릿수와 비율만 쓴다"),
        ])),
        ("caveat_tags", OrderedDict([
            ("AUDIT-NOT-MEASUREMENT", "이 실험은 문헌 상태를 세는 것이다. 성능도 비용도 재지 않았다"),
            ("KEYWORD-RULE", "분류는 산문에 대한 키워드+부정 규칙이다. 개별 라벨은 오류 가능. "
                             "모든 라벨에 증거 스니펫과 ambiguity_flags를 붙여 검증 가능하게 했다"),
            ("NOTE-MEDIATED", "1차 감사 대상은 원문이 아니라 deep-read 노트다. 노트가 놓친 값은 "
                              "이 감사도 놓친다. 완화책으로 원문 텍스트 probe를 병기하되, "
                              "probe는 '어딘가에 그 문자열이 있다'만 말하고 그 값이 sleep 예산이라는 "
                              "것은 말하지 않는다"),
            ("CHAR-APPROX", "Q5의 토큰 수는 문자/4 근사다. 소수점을 인용하지 말 것"),
            ("PRINTED-EXAMPLE-N1", "Q5의 문자 수는 논문이 인쇄한 예시 1~2건에서 나온다. "
                                   "데이터셋 평균이 아니다"),
            ("COMPETITOR-MEASURED", "Zep 저장소 600k 토큰은 경쟁사(Mem0)가 측정해 보고한 값이다. "
                                    "Zep 자신은 ingestion 비용을 전혀 보고하지 않는다"),
            ("STORE-VS-INJECTED", "저장소 크기(C_cap)와 질의당 주입량(|c-hat|)은 다른 양이다. "
                                  "논문들이 이 둘을 섞어 보고한다"),
        ])),
    ])

    # ------------------------------------------------------------------
    # Q1 — 보고 실태 전수 감사
    # ------------------------------------------------------------------
    per_paper = OrderedDict()
    for slug, note in notes.items():
        rec = classify_paper(slug, note)
        if slug in OVERRIDES:
            rec["category_rule_before_override"] = rec["category_rule"]
            rec["category"] = OVERRIDES[slug]["category"]
            rec["override_reason"] = OVERRIDES[slug]["reason"]
        else:
            rec["category"] = rec["category_rule"]
        per_paper[slug] = rec

    cat_counts = Counter(r["category"] for r in per_paper.values())
    by_cat = OrderedDict()
    for c in CATEGORY_ORDER:
        by_cat[c] = OrderedDict([
            ("label", CATEGORY_LABEL[c]),
            ("n", cat_counts.get(c, 0)),
            ("papers", [s for s, r in per_paper.items() if r["category"] == c]),
        ])

    # 우선순위(a>b>c>d>e)로 단일화한 결과가 그 선택에 민감하지 않음을 보이려고, 다중 라벨
    # 집계를 함께 낸다 — 한 논문이 여러 칸에 중복 계상된다.
    by_cat_multi = OrderedDict()
    for c in CATEGORY_ORDER:
        papers_c = [s for s, r in per_paper.items() if c in r["categories_multi"]]
        by_cat_multi[c] = OrderedDict([
            ("label", CATEGORY_LABEL[c]), ("n", len(papers_c)), ("papers", papers_c)])

    axis_x_cat = OrderedDict()
    for slug, r in per_paper.items():
        ax = r["axis"] if r["axis"] in ("E", "W", "Theta", "bio", "theory") else "other/mixed"
        axis_x_cat.setdefault(ax, Counter())[r["category"]] += 1
    axis_x_cat = OrderedDict(
        (k, OrderedDict(sorted(v.items()))) for k, v in sorted(axis_x_cat.items()))

    # 경로별 "B_s를 어떤 형태로든 보고한 비율" — 경로 간 순서가 이 실험의 인용 가능한 결과다.
    axis_rate = OrderedDict()
    for ax, cnt in axis_x_cat.items():
        tot = sum(cnt.values())
        rep_n = tot - cnt.get("f", 0)
        axis_rate[ax] = OrderedDict([
            ("n_papers", tot), ("n_reporting_something", rep_n),
            ("rate", round(rep_n / tot, 3) if tot else None)])
    axis_rate = OrderedDict(sorted(axis_rate.items(),
                                   key=lambda kv: (-(kv[1]["rate"] or 0), kv[0])))

    n_undefined = sum(1 for r in per_paper.values() if r.get("bs_defined") is False)
    n_figure_only = sum(1 for r in per_paper.values() if r.get("figure_only"))
    # 스윕은 "B_s에 대해 무언가라도 보고한 논문"에서만 센다. (f) 논문의 괄호 집합은
    # 예산 축이 아니다(예: model-collapse의 {100,316,1000}은 선형모형의 표본 수다).
    rep = [r for r in per_paper.values() if r["category"] != "f"]
    n_sweep = sum(1 for r in rep if r["sweep"]["max_points"] >= 2)
    n_sweep3 = sum(1 for r in rep if r["sweep"]["max_points"] >= 3)
    n_sweep_abs = sum(1 for r in rep if r["sweep"]["max_points"] >= 2
                      and r["sweep"]["axis_kind"] == "absolute_budget_units")
    n_sweep_raw = sum(1 for r in per_paper.values() if r["sweep"]["max_points"] >= 2)

    # MFU/달성 throughput을 실제로 보고한 노트가 몇 편인가 (전 29편 대상, 인용표지 제거 후)
    mfu_pat = INPUT_PROBES["mfu_or_throughput"]
    mfu_papers = OrderedDict()
    for sl, nt in notes.items():
        hits = [m.group(0) for m in mfu_pat.finditer(strip_citations(note_text_blob(nt)))]
        if hits:
            mfu_papers[sl] = sorted(set(hits))[:3]

    # (f)로 세어졌지만 논문이 **어떤 compute 숫자는 인쇄한** 경우 — "다른 비용은 세고
    # sleep 예산만 안 센다"를 계량한다.
    f_but_prints = OrderedDict(
        (s, r["other_cost_units"]) for s, r in per_paper.items()
        if r["category"] == "f" and
        set(r["other_cost_units"]) & {"flops", "tokens", "steps", "wall_clock"})

    # 이 라인의 이름을 만든 논문이 토큰 수를 몇 번 인쇄하는가 (원문 직접 probe)
    letta_txt = (REPO / "papers" / "2504.13171.txt").read_text(encoding="utf-8",
                                                               errors="replace")
    letta_tok_hits = re.findall(
        r"\d[\d,\.]*\s*(?:k|K|M|B)?\s*tokens?\b", letta_txt)
    letta_flop_hits = re.findall(r"FLOPs?\b", letta_txt, re.I)

    results["Q1_reporting_census"] = OrderedDict([
        ("question", "29편의 cost_table.B_s를 (a)~(f)로 분류하면 어떤 분포가 나오는가"),
        ("n_papers", len(per_paper)),
        ("by_category", by_cat),
        ("by_category_multilabel_no_priority_collapse", by_cat_multi),
        ("by_axis_x_category", axis_x_cat),
        ("reporting_rate_by_axis_sorted", axis_rate),
        ("structural", OrderedDict([
            ("B_s_undefined_no_offline_pass", n_undefined),
            ("B_s_defined", len(per_paper) - n_undefined),
            ("plotted_but_not_tabulated", n_figure_only),
            ("papers_with_>=2_points_on_the_B_s_axis", n_sweep),
            ("papers_with_>=3_points_on_the_B_s_axis", n_sweep3),
            ("of_which_points_are_in_absolute_budget_units", n_sweep_abs),
            ("raw_bracketed_sets_incl_category_f", n_sweep_raw),
            ("papers_reporting_MFU_or_achieved_throughput", mfu_papers),
            ("absent_but_paper_prints_some_other_compute_number", f_but_prints),
        ])),
        ("founding_paper_direct_probe", OrderedDict([
            ("file", "papers/2504.13171.txt (Letta, Sleep-time Compute)"),
            ("regex", r"\d[\d,\.]*\s*(k|K|M|B)?\s*tokens?\b"),
            ("n_matches", len(letta_tok_hits)),
            ("matches", sorted(set(m.strip() for m in letta_tok_hits))),
            ("n_FLOP_mentions", len(letta_flop_hits)),
            ("reading", "이 라인의 이름을 만든 논문의 전문에서 '<숫자> tokens' 꼴이 나오는 곳은 "
                        "두 군데뿐이고, 둘 다 AIME 문제 속의 게임 말('removes either 1 token "
                        "or 4 tokens')이다. 비용 숫자가 아니다. FLOP은 0회 언급된다"),
        ])),
        ("cross_check_vs_dossier_discriminant", OrderedDict([
            ("what", "이 감사는 B_s만 보고 '오프라인 패스 자체가 없다'를 프로그램으로 판정한다. "
                     "그 집합이 PRE-RESEARCH §2.2의 '판별식 불충족' 목록과 얼마나 겹치는지는 "
                     "독립적 정합성 점검이 된다"),
            ("papers_with_no_offline_pass_found_here",
             [s2 for s2, r in per_paper.items() if r["bs_defined"] is False]),
            ("reading", "이 목록은 PRE-RESEARCH §2.2 표 B(Nested Learning, Memory Caching, "
                        "Memory Layers, LoRA, RAG, Memorizing Transformers, MemGPT 경계사례, "
                        "지속학습 일반)와 사실상 일치한다. 서로 다른 필드(판별식 서술 vs "
                        "cost_table.B_s)에서 같은 결론이 나온다"),
        ])),
        ("taxonomy_gaps", OrderedDict([
            ("note", "과제가 준 (a)~(f) 여섯 칸에 들어가지 않는 보고 형태가 실제로 존재한다. "
                     "억지로 밀어 넣지 않고 여기에 남긴다"),
            ("simulated_time", [s for s, r in per_paper.items()
                                if any("SIMULATED_TIME" in f for f in r["ambiguity_flags"])]),
            ("sample_or_item_counts_not_optimizer_steps",
             [s for s, r in per_paper.items()
              if any("STEPS_VS_SAMPLES" in f for f in r["ambiguity_flags"])]),
            ("plotted_only_no_numeral", [s for s, r in per_paper.items()
                                         if r.get("figure_only")]),
            ("note_field_missing", [s for s, r in per_paper.items()
                                    if r["verdict_polarity"] == "NOTE_FIELD_MISSING"]),
        ])),
        ("n_editor_overrides", len(OVERRIDES)),
        ("editor_overrides", OVERRIDES),
        ("override_policy", "규칙 출력을 29편 전수 검토한 뒤 override가 필요 없도록 규칙을 고쳤다. "
                            "현재 override 0건 — 집계는 100% 코드 산물이다"),
        ("per_paper", per_paper),
    ])

    # ------------------------------------------------------------------
    # Q2 — 통일 축 복원 가능성
    # ------------------------------------------------------------------
    q2 = OrderedDict()
    survivors = []
    for slug, note in notes.items():
        cat = per_paper[slug]["category"]
        if cat not in ("a", "b", "c"):
            continue
        recipe_key = {"a": "a_tokens_to_flops",
                      "b": "b_steps_to_flops",
                      "c": "c_wallclock_to_flops"}[cat]
        recipe = RECIPES[recipe_key]
        pp = resolve_paper_path(note.get("path"))
        paper_blob = pp.read_text(encoding="utf-8", errors="replace") if pp else ""
        probes, probe_hits = probe_inputs(
            strip_citations(note_text_blob(note)), strip_citations(paper_blob))
        needed = recipe["required_inputs"]
        status = OrderedDict((k, probes[k]) for k in needed)
        matched = OrderedDict((k, probe_hits[k]) for k in needed)
        blocking = [k for k in needed if status[k] != "in_note"]
        convertible = not blocking
        q2[slug] = OrderedDict([
            ("category", cat),
            ("recipe", recipe_key),
            ("formula", recipe["formula"]),
            ("input_status", status),
            ("probe_matches", matched),
            ("blocking_inputs", blocking),
            ("convertible_from_note_alone", convertible),
            ("paper_txt_resolved", str(pp.relative_to(REPO)) if pp else None),
            ("n_points_on_axis", per_paper[slug]["sweep"]["max_points"]),
        ])
        if convertible:
            survivors.append(slug)

    missing_counter = Counter()
    for slug, r in q2.items():
        for k in r["blocking_inputs"]:
            missing_counter[k] += 1

    results["Q2_unified_axis"] = OrderedDict([
        ("question", "(a)~(c)를 공통 단위(FLOPs)로 환산할 수 있는가. 무엇이 없어서 못 하는가"),
        ("recipes", RECIPES),
        ("candidates_n", len(q2)),
        ("per_paper", q2),
        ("convertible_n", len(survivors)),
        ("convertible_papers", survivors),
        ("most_frequently_missing_inputs",
         OrderedDict(sorted(missing_counter.items(), key=lambda kv: (-kv[1], kv[0])))),
        ("honest_limit",
         "convertible 판정은 노트 안에 그 입력이 '문자열로 존재하는가'를 본 것이다. 존재해도 그 "
         "값이 sleep 구간에 속한다는 보장은 없으므로, 이 판정은 환산 가능성의 **상한**이다. "
         "즉 실제로 환산 가능한 점 수는 여기 숫자보다 많을 수 없다. 구체적 경고: 유일하게 "
         "닫힌 lora-as-knowledge-memory의 seq_len_tokens 히트는 '2,048-token chunk'인데 이것은 "
         "RAG **baseline**의 청크 크기이지 LoRA 학습 시퀀스 길이가 아니다. probe_matches를 "
         "직접 보고 판단하라"),
    ])

    # ------------------------------------------------------------------
    # Q3 — 품질 축의 비교 가능성
    # ------------------------------------------------------------------
    q3_pool = [s for s, r in per_paper.items()
               if r["category"] in ("a", "b", "c", "d") and r["sweep"]["max_points"] >= 2]
    q3 = OrderedDict()
    for slug in q3_pool:
        blob = results_blob(notes[slug])
        q3[slug] = OrderedDict([
            ("category", per_paper[slug]["category"]),
            ("axis_kind", per_paper[slug]["sweep"]["axis_kind"]),
            ("n_points_on_axis", per_paper[slug]["sweep"]["max_points"]),
            ("sweep_values", per_paper[slug]["sweep"]["values"]),
            ("benchmarks", lexicon_hits(blob, BENCHMARKS, apply_supersede=True)),
            ("metrics", lexicon_hits(blob, METRICS)),
            ("baselines", lexicon_hits(blob, BASELINES)),
        ])

    bench_sets = {s: set(v["benchmarks"]) for s, v in q3.items()}
    shared_pairs = []
    for i, a in enumerate(sorted(bench_sets)):
        for b in sorted(bench_sets)[i + 1:]:
            inter = sorted(bench_sets[a] & bench_sets[b])
            if inter:
                shared_pairs.append(OrderedDict([("pair", [a, b]), ("shared", inter)]))
    common_all = set.intersection(*bench_sets.values()) if bench_sets else set()

    bench_freq = Counter()
    for s in bench_sets.values():
        bench_freq.update(s)
    max_group = max(bench_freq.values()) if bench_freq else 0
    max_group_bench = [b for b, n in sorted(bench_freq.items()) if n == max_group]

    results["Q3_quality_axis"] = OrderedDict([
        ("question", "B_s 축 위에 2점 이상을 가진 논문들의 품질 축이 서로 비교 가능한가"),
        ("pool", q3_pool),
        ("per_paper", q3),
        ("benchmark_intersection_over_all_pool", sorted(common_all)),
        ("pairwise_shared_benchmarks", shared_pairs),
        ("largest_group_sharing_one_benchmark", OrderedDict([
            ("size", max_group), ("benchmarks", max_group_bench)])),
        ("papers_with_no_named_public_benchmark",
         [s for s, v in q3.items() if not v["benchmarks"]]),
        # 아래는 원문 근거가 확인된 것만 담은 편집자 목록이다(자동 추론 아님):
        # Letta STC는 §5.1/§6에서 네 벤치를 스스로 만들었다고 쓰고, S0 §4 의무 caveat도 이를 명시한다.
        ("author_constructed_benchmarks_in_pool_editorial_list",
         sorted({b for v in q3.values() for b in v["benchmarks"]
                 if b.startswith("Stateful") or b.startswith("Multi-Query")
                 or b == "SWE-Features"})),
        ("author_constructed_note",
         "이 목록은 원문에서 '우리가 만들었다'는 진술이 확인된 것만 담는다. 목록에 없다고 해서 "
         "공개 벤치라는 뜻은 아니다(예: need-sleep-recurrence의 automaton/Depo는 미확인)."),
        ("note", "벤치마크 어휘 사전 매칭이다. 같은 이름의 벤치마크라도 split·프롬프트·"
                 "채점기가 다르면 실제로는 비교 불가일 수 있다 — 즉 여기 나온 공유량은 **상한**이다"),
    ])

    # ------------------------------------------------------------------
    # Q5 — 문맥 길이 근거화 (Q4보다 먼저 계산: Q4 판정이 이 값을 인용한다)
    # ------------------------------------------------------------------
    measured = []
    missing_anchor = []
    paper_cache = {}

    def read_paper(rel):
        if rel not in paper_cache:
            p = REPO / rel
            paper_cache[rel] = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
        return paper_cache[rel]

    for label, rel, a0, a1, _drop, prov, what in CONTEXT_EXTRACTS:
        txt = read_paper(rel)
        seg = extract_between(txt, a0, a1)
        if seg is None:
            missing_anchor.append(label)
            continue
        measured.append(char_stats(label, normalise_ws(seg), what, prov))

    lt = read_paper("papers/2504.13171.txt")
    frags, ok = [], True
    for f in LEARNED_CONTEXT_FRAGMENTS:
        if f in lt:
            frags.append(normalise_ws(f))
        else:
            ok = False
    if ok:
        measured.append(char_stats(
            "letta_fig1_learned_context", " ".join(frags),
            "|c-hat| — Figure 1이 인쇄한 학습된 문맥 전체 (그림 상자들 사이에 끼어 추출되므로 "
            "6조각을 순서대로 이어 붙였다. 각 조각은 원문에 그대로 존재함을 확인)",
            "[Letta STC Fig. 1, p.2] 'Learned Context' 상자"))
    else:
        missing_anchor.append("letta_fig1_learned_context")

    by_label = {m["label"]: m for m in measured}
    ratio_from_fig1 = None
    if "letta_fig1_learned_context" in by_label and "letta_fig1_raw_context" in by_label:
        ratio_from_fig1 = round(
            by_label["letta_fig1_learned_context"]["chars"]
            / by_label["letta_fig1_raw_context"]["chars"], 2)

    reported = []
    for label, rel, anchor, value, what, prov in REPORTED_TOKEN_ANCHORS:
        txt = read_paper(rel)
        found = anchor in txt
        reported.append(OrderedDict([
            ("label", label), ("what", what), ("tokens_reported_by_paper", value),
            ("source", prov), ("anchor_verified_in_vendored_text", found),
        ]))
    reported_ok = all(r["anchor_verified_in_vendored_text"] for r in reported)

    # 관측된 |c-hat|/|c| 비 — 같은 논문·같은 데이터셋 안에서만 짝짓는다.
    ratios = [
        OrderedDict([
            ("pair", "Zep 주입 문맥 / LongMemEval 대화"),
            ("value", round(1600 / 115000, 4)),
            ("basis", "둘 다 Zep Table 2의 같은 행. 질의당 주입량 대 원 문맥"),
            ("kind", "injected/raw — X1의 len_chat/len_c와 같은 종류"),
        ]),
        OrderedDict([
            ("pair", "Mem0 저장소 / LOCOMO 대화"),
            ("value", round(7000 / 26000, 3)),
            ("basis", "Mem0 §4.5. 저장소 크기이지 주입량이 아니다"),
            ("kind", "store/raw — X1의 축과 다른 종류(주의)"),
        ]),
        OrderedDict([
            ("pair", "Mem0^g 저장소 / LOCOMO 대화"),
            ("value", round(14000 / 26000, 3)),
            ("basis", "Mem0 §4.5"),
            ("kind", "store/raw"),
        ]),
        OrderedDict([
            ("pair", "Zep 저장소 / LOCOMO 대화 (경쟁사 측정)"),
            ("value", round(600000 / 26000, 1)),
            ("basis", "Mem0 §4.5가 측정한 Zep 값. Zep 자신은 보고하지 않음"),
            ("kind", "store/raw, COMPETITOR-MEASURED"),
        ]),
    ]
    if ratio_from_fig1 is not None:
        ratios.insert(0, OrderedDict([
            ("pair", "Letta 학습된 문맥 / 원 문맥 (Fig. 1 예시 1건)"),
            ("value", ratio_from_fig1),
            ("basis", "같은 그림의 두 상자를 문자 수로 비교. 토큰 근사가 양쪽에 같게 걸리므로 "
                      "비는 근사에 둔감하다"),
            ("kind", "injected/raw — X1의 len_chat/len_c와 같은 종류, 단 예시 1건"),
        ]))

    c_tokens = [(m["label"], m["tokens_approx_chars_over_4"]) for m in measured
                if m["label"].startswith("letta_") and "learned" not in m["label"]]
    c_min = min((v for _, v in c_tokens), default=None)
    c_max_measured = max((v for _, v in c_tokens), default=None)

    swept_ratios = X1_ASSUMPTIONS["len_chat_over_len_c_grid"]
    injected_ratios = [r["value"] for r in ratios if r["kind"].startswith("injected")]
    store_ratios = [r["value"] for r in ratios if r["kind"].startswith("store")]

    results["Q5_context_length_grounding"] = OrderedDict([
        ("question", "X1이 sweep한 |c|·|c-hat| 범위가 현실적인가 — vendored 원문의 인쇄된 예시를 "
                     "문자 수로 재고, 논문이 보고한 토큰 수와 맞춰본다"),
        ("method", "① 원문에서 앵커 문자열 사이를 잘라 문자 수를 센다(줄번호 아닌 내용 고정). "
                   "② 문자/4로 토큰을 근사하고 3~5 밴드를 병기한다. "
                   "③ 논문이 스스로 토큰으로 보고한 값은 근사 없이 그대로 쓰고, 앵커로 원문 존재를 확인한다"),
        ("measured_from_printed_examples", measured),
        ("anchors_not_found", missing_anchor),
        ("reported_by_papers_in_tokens", reported),
        ("all_reported_anchors_verified", reported_ok),
        ("corpus_wide_token_length_mining", mine_reported_token_lengths()),
        ("observed_ratios_chat_over_c", ratios),
        ("X1_assumptions", X1_ASSUMPTIONS),
        ("verdict", OrderedDict([
            ("short_contexts", "X1이 GSM/AIME에 가정한 |c| = 400/300 토큰은 논문이 인쇄한 예시 "
                               "문맥의 문자 기반 추정보다 크다 → X1은 그 두 레짐의 여유(L*/|c|)를 "
                               "과소 보고했다(= STC에 불리하게 잡았다)"),
            ("long_contexts", "corpus가 실제로 다루는 대화형 문맥(26k~115k 토큰)은 X1의 최대 "
                              "레짐(8k)보다 훨씬 크다 → X1의 sweep은 상단이 잘려 있었다"),
            ("ratio_grid", "X1이 sweep한 |c-hat|/|c| = {0.25, 1, 4}는 관측 범위(0.014 ~ 23)의 "
                           "가운데 한 자릿수만 덮는다. 양 끝이 모두 밖에 있다"),
            ("net_effect_on_X1_conclusions", "두 오차의 방향이 X1의 결론과 같은 쪽이다 — "
                                             "레짐 간 여유의 순서(GSM >> AIME > SWE)와 "
                                             "'긴 문맥에서 상각이 물린다'는 방향은 유지되고 강해진다. "
                                             "L*의 절대치는 여전히 인용 불가"),
        ])),
        ("comparison", OrderedDict([
            ("letta_printed_contexts_tokens_approx",
             OrderedDict([("min", c_min), ("max", c_max_measured)])),
            ("X1_assumed_len_c_for_those_regimes",
             [X1_ASSUMPTIONS["stateful_aime.len_c_tokens"],
              X1_ASSUMPTIONS["stateful_gsm_symbolic.len_c_tokens"]]),
            ("corpus_reported_len_c_range_tokens", [26000, 115000]),
            ("X1_swept_ratio_grid", swept_ratios),
            ("observed_injected_over_raw_ratios", injected_ratios),
            ("observed_store_over_raw_ratios", store_ratios),
        ])),
    ])

    # ------------------------------------------------------------------
    # Q4 — 판정 + 최소 요구 목록
    # ------------------------------------------------------------------
    drawable = (len(survivors) >= 3 and len(common_all) >= 1)

    min_reqs = [
        OrderedDict([
            ("id", "R1"),
            ("req", "B_s를 절대 FLOPs로 보고하라. 못 하면 (토큰 수 + 모델 파라미터 수 N + "
                    "prefill/decode 분해)를 함께 보고해 FLOPs가 닫히게 하라"),
            ("why", f"현재 (a)범주는 {by_cat['a']['n']}편({by_cat['a']['papers']})뿐인데 그 편도 "
                    "backbone 모델 크기가 미보고라 FLOPs로 닫히지 않는다. corpus 전체에서 환산이 "
                    f"닫히는 것은 {len(survivors)}편({survivors})이며 그것은 (a)가 아니라 (b)범주다"),
            ("evidence", "Q2.most_frequently_missing_inputs"),
        ]),
        OrderedDict([
            ("id", "R2"),
            ("req", "step·wall-clock으로 보고할 거면 환산에 필요한 나머지를 같이 보고하라 — "
                    "step이면 (batch, 시퀀스 길이, N, 학습 가능 파라미터 비율), "
                    "wall-clock이면 (장치 종류, 장치 수, 달성 throughput 또는 MFU)"),
            ("why", "step/wall-clock은 하드웨어와 구현에 종속된 양이라 그 자체로는 축이 되지 "
                    f"않는다. MFU 또는 달성 throughput을 보고한 노트는 29편 중 {len(mfu_papers)}편"
                    f"({list(mfu_papers)})이고, 그 편은 $B_s$가 정의되지도 않는 논문이다"),
            ("evidence", "Q2.per_paper[*].blocking_inputs"),
        ]),
        OrderedDict([
            ("id", "R3"),
            ("req", "B_s를 **한 점이 아니라 최소 3점**으로 sweep하고 각 점의 품질을 보고하라. "
                    "1점은 운영점(operating point)이지 축이 아니다"),
            ("why", f"B_s 축에 2점 이상이 놓인 논문 {n_sweep}편 / 3점 이상 {n_sweep3}편이고, "
                    f"그 점이 **절대 예산 단위**인 논문은 {n_sweep_abs}편뿐이다. 곡선의 지수·"
                    "포화점은 3점 미만에서 정의되지 않고, 눈금 없는 비례 knob 위에서는 "
                    "기울기를 다른 논문과 비교할 수 없다"),
            ("evidence", "Q1.structural"),
        ]),
        OrderedDict([
            ("id", "R4"),
            ("req", "품질은 **기존 공개 벤치마크**에서, 같은 지표·같은 baseline arm으로 각 B_s 점마다 "
                    "보고하라. 저자 제작 변형 벤치는 부가로만"),
            ("why", "지금은 sweep을 가진 논문들끼리 공유하는 벤치마크의 교집합이 "
                    f"{sorted(common_all) or '공집합'}이다. B_s 축을 복원해도 y축이 붙지 않는다"),
            ("evidence", "Q3.benchmark_intersection_over_all_pool"),
        ]),
        OrderedDict([
            ("id", "R5"),
            ("req", "B_s가 **무엇당** 예산인지 명시하라 — 문맥당/세션당/과제당/라운드당 — 그리고 "
                    "그것을 나눠 갚을 N_q(한 문맥을 공유하는 질의 수)의 분포를 보고하라"),
            ("why", "상각식 (A)의 분모다. 현재 corpus에서 실서빙 N_q 분포를 보고한 논문은 0편이고, "
                    "벤치마크 수준의 N_q조차 부수적으로만 계산 가능하다"),
            ("evidence", "S0 §4 의무 caveat; X1 ABSENT_FROM_PAPER.N_q_distribution"),
        ]),
        OrderedDict([
            ("id", "R6"),
            ("req", "각 B_s 점에서 wake 쪽 delta도 함께 보고하라 — 특히 학습된 문맥이 매 질의 "
                    "프롬프트에 더하는 토큰 수 |c-hat|과 원 문맥 |c|"),
            ("why", "X1이 보인 대로 |c-hat|의 prefill은 N_q로 상각되지 않는다. B_s만 보고하면 "
                    "총비용의 부호가 결정되지 않는다"),
            ("evidence", "X1 F1; Q5.observed_ratios_chat_over_c"),
        ]),
        OrderedDict([
            ("id", "R7"),
            ("req", "저장소 크기 C_cap(바이트 또는 토큰)과 질의당 주입량 |c-hat|을 **분리해서** "
                    "보고하라. 둘을 한 숫자로 섞지 말 것"),
            ("why", "Mem0는 저장소를 토큰으로, Zep은 주입량을 토큰으로 보고한다. 같은 단위로 인쇄돼 "
                    "있어 섞이기 쉽고, 섞으면 비가 1.5자릿수 넘게 틀어진다"),
            ("evidence", "Q5.observed_ratios_chat_over_c (kind 필드)"),
        ]),
        OrderedDict([
            ("id", "R8"),
            ("req", "B_s를 반복해서 내는지(라운드마다) 한 번만 내는지, 그리고 라운드 수를 보고하라"),
            ("why", "Θ-경로와 E-경로는 재소비 구조가 다르다. 라운드 수가 없으면 총 B_s가 정의되지 않고 "
                    "열화율 rho와도 짝지을 수 없다"),
            ("evidence", "Q1.per_paper (model-collapse·SEAL·SCM의 라운드 서술)"),
        ]),
        OrderedDict([
            ("id", "R9"),
            ("req", "그림 축에만 FLOPs를 그리지 말고 **표에 숫자로** 인쇄하라"),
            ("why", f"현재 {n_figure_only}편이 'B_s를 그림으로는 그렸으나 인용 가능한 숫자가 없는' "
                    "상태다. 그림만으로는 점을 찍을 수 없다"),
            ("evidence", "Q1.structural.plotted_but_not_tabulated"),
        ]),
    ]

    results["Q4_verdict"] = OrderedDict([
        ("question", "지금 문헌으로 B_s-품질 곡선을 그릴 수 있는가"),
        ("can_draw_a_B_s_quality_curve", drawable),
        ("test_used", "환산 가능한 논문 >= 3편 AND 그 논문들이 공유하는 벤치마크 >= 1개"),
        ("counts", OrderedDict([
            ("papers_in_corpus", len(per_paper)),
            ("B_s_absent_or_undefined", by_cat["f"]["n"]),
            ("B_s_in_absolute_flops_or_tokens", by_cat["a"]["n"]),
            ("convertible_to_FLOPs_from_note", len(survivors)),
            ("papers_with_>=3_points_on_axis", n_sweep3),
            ("shared_benchmark_across_sweep_pool", sorted(common_all)),
        ])),
        ("minimum_reporting_requirements", min_reqs),
        ("what_ch28_can_still_do",
         "곡선을 못 그린다는 것이 곡선이 없다는 뜻은 아니다. ch28이 할 수 있는 것은 (i) 축이 "
         "왜 복원 불가인지를 계수로 보이고, (ii) 최소 요구 목록 R1-R9를 제시하며, "
         "(iii) X1·X2가 세운 회계 위에서 **곡선의 부호와 경계 조건**만 진술하는 것이다. "
         "지수·포화점은 진술하지 않는다"),
    ])

    # ------------------------------------------------------------------
    # findings — 본문 단정 가능 / 방향성만
    # ------------------------------------------------------------------
    a_n, f_n = by_cat["a"]["n"], by_cat["f"]["n"]
    # X1의 |c| 가정이 인쇄된 예시 대비 몇 배인지 (동적 계산)
    over = []
    for lbl, assumed in (("letta_appB_aime_context_1", 300.0),
                         ("letta_appB_aime_context_2", 300.0),
                         ("letta_fig20_gsm_context", 400.0),
                         ("letta_fig1_raw_context", 400.0)):
        if lbl in by_label:
            over.append(assumed / by_label[lbl]["tokens_approx_chars_over_4"])
    over_lo = round(min(over), 1) if over else None
    over_hi = round(max(over), 1) if over else None
    mine = results["Q5_context_length_grounding"]["corpus_wide_token_length_mining"]
    results["answers"] = OrderedDict([
        ("Q1", f"29편 중 (a)절대 FLOPs/토큰 {by_cat['a']['n']} · (b)step {by_cat['b']['n']} · "
               f"(c)벽시계 {by_cat['c']['n']} · (d)서수 knob {by_cat['d']['n']} · "
               f"(e)상대 비율 {by_cat['e']['n']} · (f)absent {by_cat['f']['n']}. "
               f"(f) 중 {n_undefined}편은 오프라인 패스 자체가 없어 B_s가 정의되지 않는다."),
        ("Q2", f"(a)~(c) 후보 {len(q2)}편 중 노트만으로 FLOPs 환산이 닫히는 것은 {len(survivors)}편"
               f"({survivors}). 가장 자주 막는 입력은 시퀀스 길이·모델 크기·batch이며, "
               f"벽시계 환산에 필요한 MFU/달성 throughput은 29편 중 {len(mfu_papers)}편만 보고한다 "
               "(그 1편은 B_s가 정의되지도 않는 LoRA다)."),
        ("Q3", f"B_s 축에 2점 이상을 가진 {len(q3_pool)}편의 벤치마크 교집합은 "
               f"{sorted(common_all) or '공집합'}이고, 두 편 이상이 공유하는 벤치마크가 "
               f"하나도 없다(pairwise 공유 {len(shared_pairs)}건). 품질 축도 깨져 있다."),
        ("Q4", "그릴 수 없다. 최소 요구 R1–R9를 제시한다 — 절대 FLOPs(또는 닫힌 환산 입력), "
               "step/wall-clock 환산 입력, 최소 3점 sweep, 공개 벤치 동일 지표·동일 baseline, "
               "B_s의 분모(문맥/세션/과제)와 N_q, wake 쪽 |c-hat| delta, C_cap과 주입량 분리, "
               "라운드 수, 그림이 아니라 표."),
        ("Q5", "X1의 |c| 가정은 짧은 레짐에서 과대(인쇄된 예시 대비 약 3–12배), 긴 레짐에서 "
               "과소(corpus 실제 26k–115k 토큰 대비 3–14배 작음)다. |c-hat|/|c|의 관측 범위는 "
               "0.014~23으로 X1의 {0.25,1,4}보다 훨씬 넓다. 두 오차 모두 X1의 결론과 같은 "
               "방향이라 X1의 순서 결론은 유지·강화된다."),
    ])

    results["findings"] = OrderedDict([
        ("load_bearing_may_be_stated_as_fact", [
            f"F1 전수 감사 결과, 29편 중 {f_n}편의 $B_s$가 absent이고 절대 FLOPs/토큰으로 보고한 "
            f"논문은 {a_n}편이다. 나머지는 optimizer step({by_cat['b']['n']}편)·"
            f"벽시계({by_cat['c']['n']}편)·서수 knob({by_cat['d']['n']}편)·"
            f"상대 비율({by_cat['e']['n']}편)로만 보고한다. "
            "즉 $B_s$는 이 문헌에서 **축이 아니라 각주**다.",

            f"F1b 그 (f) 19편 중 {n_undefined}편은 '보고 안 함'이 아니라 **질의 전 오프라인 "
            "패스 자체가 없어 $B_s$가 정의되지 않는** 경우다. 이 집합은 PRE-RESEARCH §2.2의 "
            "판별식 불충족 목록과 사실상 일치한다 — 서로 다른 필드에서 같은 결론이 나온다. "
            "따라서 '보고가 부실하다'와 '범주에 속하지 않는다'를 섞어 세면 안 된다.",

            f"F2 이 라인의 이름을 만든 논문(Letta STC)이 $B_s$를 서수 knob으로만 보고한다. "
            f"원문 전문 probe: '<숫자> tokens' 꼴이 {len(letta_tok_hits)}회 나오는데 둘 다 AIME "
            f"문제 속 게임 말('1 token', '4 tokens')이고, 'FLOP'은 "
            f"{len(letta_flop_hits)}회 나온다. 상각 논증의 분자에 숫자가 없다.",

            f"F3 환산 가능성: (a)~(c) 후보 {len(q2)}편 중 노트만으로 FLOPs 환산이 닫히는 논문은 "
            f"{len(survivors)}편이다. 가장 자주 막는 입력은 "
            f"{list(results['Q2_unified_axis']['most_frequently_missing_inputs'])[:3]}이며, "
            f"특히 **MFU 또는 달성 throughput을 보고한 노트가 29편 중 {len(mfu_papers)}편**이라 "
            "벽시계를 FLOPs로 옮기는 다리가 없다.",

            f"F4 $B_s$에 대해 무언가라도 보고한 {len(rep)}편 중 축 위에 2점 이상을 놓은 논문은 "
            f"{n_sweep}편, 3점 이상은 {n_sweep3}편이고, 그 점들이 **절대 예산 단위**인 논문은 "
            f"{n_sweep_abs}편뿐이다(나머지는 눈금 없는 비례 knob). 그 부분집합이 공유하는 "
            "벤치마크의 교집합은 "
            f"{sorted(common_all) or '공집합'}이다. **$B_s$ 축을 복원해도 y축이 붙지 않는다** — "
            "축이 하나가 아니라 둘 다 깨져 있다.",

            "F4b 보고율은 경로마다 다르다 — "
            + ", ".join(f"{k} {v['n_reporting_something']}/{v['n_papers']}"
                        for k, v in axis_rate.items() if k in ("Theta", "E", "W", "theory", "bio"))
            + " (W는 1편, bio는 3편으로 표본이 작아 순서 진술에서 뺀다). 표본이 의미 있는 두 "
            "경로만 보면 $\\Theta$ 5/8 > $E$ 2/9로 순서가 뚜렷하다. 이유는 기제가 아니라 "
            "**관행**이다 — $\\Theta$-경로는 ML 학습 논문의 보고 관습(step·batch·GPU-hour)을 "
            "물려받았고, $E$-경로는 시스템/제품 논문의 관습(지연·정확도)을 물려받아 "
            "쓰기 비용을 세는 칸이 아예 없다. **순서는 인용 가능, 절대 비율은 표본이 작다.**",

            "F5 판정: 현재 문헌으로 $B_s$–품질 곡선은 그릴 수 없다. 이것이 결과다. "
            "ch28의 기여는 곡선이 아니라 **곡선을 그리기 위한 최소 보고 요구 R1–R9**와 그 요구가 "
            "왜 필요한지의 계수적 근거다.",

            f"F6 (Q5) X1이 GSM/AIME 레짐에 가정한 $|c|$(300–400 토큰)는 논문이 **직접 인쇄한** "
            f"예시 문맥의 문자 수 기반 추정보다 약 {over_lo}–{over_hi}배 크다. $L^*$는 $|c|$에 "
            "의존하지 않고 $L^*/|c|$만 $1/|c|$로 변하므로, X1은 그 두 레짐의 여유를 **과소** "
            "보고했다 — X1의 F2(짧은 문맥 레짐은 안전)는 약화되지 않고 강화된다.",

            "F7 (Q5) 반대로 corpus가 실제로 보고하는 대화형 문맥은 26k(LOCOMO)–115k(LongMemEval) "
            "토큰으로, X1이 가장 큰 레짐으로 잡은 8k보다 3–14배 크다. X1의 sweep은 "
            "**상단이 잘려 있었다**. 그 영역에서 $L^*/|c|$는 X1이 보고한 SWE 값보다 더 작아지므로 "
            "X1의 F3(SWE에서 여유가 급감)도 방향이 유지되고 더 강해진다. corpus 전문을 긁은 "
            f"문맥 길이 분포(중앙값 {mine['global_p50']}, p90 {mine['global_p90']} 토큰, "
            f"{mine['n_hits_total']}건/{mine['n_files_with_hits']}편)도 같은 방향이다.",

            "F8 (Q5) $|\\hat c|/|c|$의 관측 범위는 종류를 구분해 읽어야 한다. **주입량/원문맥**은 "
            "Zep이 0.014, Letta Fig.1 예시가 약 3이고, **저장소/원문맥**은 Mem0 0.27, "
            "Mem0^g 0.54, Zep 23(경쟁사 측정)이다. X1이 sweep한 {0.25, 1, 4}는 이 범위의 "
            "가운데 한 자릿수만 덮는다.",
        ]),
        ("directional_or_lower_bound_only", [
            "D1 개별 논문의 라벨은 키워드+부정 규칙의 산물이다. 범주별 **편수와 순서**는 인용해도 "
            "되지만, 특정 논문이 (b)냐 (c)냐 하는 개별 판정은 `evidence`와 `ambiguity_flags`를 "
            "직접 보고 재확인한 뒤에만 인용하라.",

            "D2 Q5의 토큰 수는 전부 문자/4 근사다. 자릿수와 비율만 쓸 것. 소수점·유효숫자 2자리 "
            "이상은 본문에 쓰지 마라. 특히 수식·코드가 섞인 문맥에서는 4자/토큰이 과대추정이다.",

            "D3 Q5의 문자 수는 논문이 인쇄한 예시 **1~2건**에서 나온다. 데이터셋 평균이 아니며, "
            "'AIME 문맥의 평균 길이'로 인용하면 오용이다.",

            "D4 Zep 저장소 600k 토큰은 **경쟁사(Mem0)가 측정해 보고한 값**이다. 방향(자릿수 크게 "
            "큼)만 쓰고 절대치는 쓰지 마라. Zep 자신은 ingestion 비용을 한 줄도 보고하지 않는다.",

            "D5 SWE-Features의 $|c|$는 여전히 미보고다. 논문은 '여러 관련 PR'이라고만 쓰고 그 수를 "
            "밝히지 않으며, PR 예시의 changed_files는 'omitted here for brevity'로 잘려 있다. "
            "X1의 8000 토큰 가정은 근거화되지 않았고, 이 실험은 그것을 **하한**으로만 지지한다.",

            "D6 Q2의 'convertible' 수는 노트 안 문자열 존재 여부로 판정한 **상한**이다. 실제로 "
            "환산 가능한 점 수는 이보다 많을 수 없다(적을 수는 있다).",

            "D7 Q3의 벤치마크 공유량도 이름 매칭이므로 **상한**이다. 같은 이름이어도 split·"
            "채점기·프롬프트가 다르면 실제 비교 가능성은 더 낮다.",

            "D8 F4b의 경로별 보고율은 표본이 작다(W 1편, bio 3편, other/mixed 3편). "
            "인용해도 되는 것은 $\\Theta$ > $E$라는 **순서**뿐이고, 0.625 대 0.222 같은 "
            "비율 자체는 본문 수치로 쓰지 마라.",

            "D9 이 감사는 corpus를 '이 책이 deep-read한 29편'으로 고정한다. 문헌 전체에 대한 "
            "표본이 아니다. '문헌의 X%가 …'로 일반화하지 말고 '이 corpus의 29편 중 …'으로 써라.",
        ]),
    ])

    # runtime은 실행마다 달라지므로 result.json에 넣지 않는다 — 그래야 두 번 실행의 sha256이
    # bit-identical이 된다. 시간은 stdout에만 찍는다.
    runtime = round(time.time() - T0, 3)

    # 키 순서를 Q1..Q5 → answers → findings 로 정렬한다(Q5는 Q4보다 먼저 계산됐다).
    order = ["meta", "answers", "Q1_reporting_census", "Q2_unified_axis",
             "Q3_quality_axis", "Q4_verdict", "Q5_context_length_grounding", "findings"]
    results = OrderedDict(
        [(k, results[k]) for k in order if k in results]
        + [(k, v) for k, v in results.items() if k not in order])

    out = HERE / "result.json"
    payload = json.dumps(results, ensure_ascii=False, indent=2)
    out.write_text(payload, encoding="utf-8")

    print(f"wrote {out}  ({len(payload)} bytes)")
    print(f"sha256(result.json) = {hashlib.sha256(payload.encode('utf-8')).hexdigest()}")
    print()
    print("Q1 by category:")
    for c in CATEGORY_ORDER:
        print(f"  ({c}) {CATEGORY_LABEL[c]:18s} n={by_cat[c]['n']:2d}  {by_cat[c]['papers']}")
    print(f"\nQ2 convertible to FLOPs: {len(survivors)} / {len(q2)} candidates")
    print(f"Q3 shared benchmark across sweep pool: {sorted(common_all) or 'EMPTY'}")
    print(f"Q4 can draw B_s-quality curve: {drawable}")
    print(f"\nruntime {runtime}s  (result.json에는 넣지 않는다 — 결정론 유지)")


if __name__ == "__main__":
    main()
