# 쉬운 버전 — 개념 booklet 12권

> Sleep-Time Compute Study의 최신 section별 쉬운 설명은 [`sleep-time-compute/README.md`](sleep-time-compute/README.md)와 [`SLEEP-TIME-COMPUTE-EASY-COMPANION-KR.pdf`](../build/publications/SLEEP-TIME-COMPUTE-EASY-COMPANION-KR.pdf)에 있다. 아래 G00–G11은 Neural Memory 계열의 선행 개념 booklet로 함께 유지한다.

수식·증명을 앞세우지 않고 **개념·물리적 직관 먼저**, 모든 기호를 풀어 쓰고, "이 계열을 시스템 모델링에 활용할 수준"을 목표로 한 쉬운 판본. 각 권은 개별 PDF (`pdf/`).

문체 규약: 💡직관(파랑) · 🔤기호 풀이(주황) · ⚙️시스템 모델링 관점(초록) · 비유(회색) · 주의/한계(빨강) · 핵심/요약(보라) 콜아웃 박스. 증명·유도는 제외; 수식은 "이 식은 ~라는 뜻"으로만 풀이.

| 권 | 제목 | 다루는 것 |
|---|---|---|
| G00 | 오리엔테이션과 기호 사전 | 한 개의 핵심 아이디어 + 전 권 공통 기호 대사전 |
| G01 | 5분 만에 이해하는 학습 | gradient·momentum·optimizer를 (state,update,cost) 객체로 |
| G02 | 빌딩블록 | associative memory·linear attention=fast-weight·SSM/Mamba·TTT·chunking |
| G03 | Titans | 시험 때 기억하는 법을 배우다 |
| G04 | Miras | 모든 것은 연결되어 있다 (설계공간 4축) |
| G05 | Atlas | 문맥을 더 잘 기억하기 (Omega·feature map·Muon) |
| G06 | TNT | 학습을 싸게 만들기 (chunk 경제학) |
| **G07** | **Nested Learning** | **전부 다른 속도의 메모리다 (핵심)** |
| **G08** | **Sleep** | **자고 일어나 정리하기 (핵심)** |
| G09 | 시스템 모델링 관점 | state 크기·decode 비용·메모리 계층·batching·scaling·HW |
| G10 | Memory Caching | 고정 메모리 벽을 체크포인트로 — 4 variants(Residual/GRM/Soup/SSC)·O(NL) 보간 |
| G11 | NSTM | 읽기/쓰기 빈도 분리·memory caching 실전 성패·NL/Sleep에 주는 힌트 |

빌드: `pandoc booklets/<id>.md --template=build/template-easy.tex --lua-filter=../build/breakable.lua --lua-filter=build/callouts.lua --lua-filter=build/mathfit.lua --pdf-engine=lualatex`.
검증: `../build/overflow_gate.py <pdf>` PASS + missing-char 0.

> **`breakable.lua`를 빼지 마라.** 이 줄은 원래 그 필터 없이 적혀 있었고, Sleep-Time Compute v2
> 판을 빌드할 때 `LoRA(arXiv:2106.09685)·ROME(arXiv:2202.05262)·MEMIT(...)`처럼 중점으로 이어진
> arXiv ID 런이 줄바꿈 지점을 못 찾아 오른쪽 여백을 넘었다. Neural Memory 판이 통과한 것은
> 본문에 그런 나열이 없었기 때문이지 안전해서가 아니다. 상세는 `../build/METHOD.md` §1.
>
> 이모지도 쓰지 마라 — 이 템플릿의 CJK 폰트에 없어 `Missing character`로 조용히 사라진다.
> 콜아웃은 `> **시스템 모델링 관점.**`처럼 **텍스트 라벨**로 연다.

Sleep-Time Compute v2 판은 `stc-v2/` (원고) → `pdf-stc-v2/` (PDF)이며,
`stc-v2/build-easy-v2.sh` 가 위 두 게이트를 전권에 걸어 한 번에 돌린다.
