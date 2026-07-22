# 8개 논문 한국어 의역본

각 원 논문을 **섹션 구조 그대로 따라가며** 자연스러운 한국어로 옮긴 스터디용 판본. 직역이 아닌 **의역**(sense-for-sense)이되, 주장·수식·수치·논리·그림은 **원문 그대로 보존**. 저자의 해석/비판은 넣지 않음(그건 스터디 페이퍼 `../study-kr/`의 몫). 각 논문은 개별 PDF (`pdf/`).

> 공식 번역이 아닌 스터디용 의역본입니다. 수식·수치·그림은 원문 기준이며, 그림은 원저자의 것입니다.

| 논문 | arXiv | 의역본 |
|---|---|---|
| Titans: Learning to Memorize at Test Time | 2501.00663 | `2501.00663-titans.md` |
| It's All Connected (Miras) | 2504.13173 | `2504.13173-miras.md` |
| Atlas: Learning to Optimally Memorize the Context at Test Time | 2505.23735 | `2505.23735-atlas.md` |
| TNT: Improving Chunkwise Training for Test-Time Memorization | 2511.07343 | `2511.07343-tnt.md` |
| Nested Learning: The Illusion of Deep Learning Architecture | 2512.24695 | `2512.24695-nested-learning.md` |
| Language Models Need Sleep | 2606.03979 | `2606.03979-sleep.md` |
| Memory Caching: RNNs with Growing Memory | 2602.24281 | `2602.24281-memory-caching.md` |
| Online Neural Space-Time Memory (NSTM) | 2607.15271 | `2607.15271-nstm.md` |

빌드: `pandoc <id>.md --template=build/template.tex --lua-filter=build/breakable.lua --lua-filter=build/mathfit.lua --pdf-engine=lualatex`. 검증: overflow gate PASS + missing-char 0.
