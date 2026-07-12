# 최종 QA 수선 리포트 (final-qa-fix-report)

**라운드**: 통합-수준 마무리 QA — 시각 결함 **수선 담당**
**날짜**: 2026-07-12
**입력 리포트**: `final-xref-report.md`, `final-figure-report.md`, `final-visual-report.md`, 인용 처리(=`study-kr/back/90-bibliography.md` §D 마감 라운드)
**기준**: STYLE-NOTATION v1.1, 손-번호 정본 체계(pandoc auto-number OFF), 정직성 계약·pair thesis 불변
**원칙**: load-bearing 내용(수식·API 이름·수치·논지) 불변. 이번 라운드 편집은 **줄바꿈 기회 추가(공백/환경 분할)** 뿐 — 문자 토큰·의미는 그대로.

---

## 1. 처리한 것 (Edit 완료 — 7건, 소스 5개 파일)

시각 리포트의 결함 D1–D5는 모두 **근원이 하나**다: 줄바꿈 불가능한 긴 inline run(디스플레이 수식 / inline 수식 리스트 / `·`로 이어붙인 `\texttt` 코드 스팬)이 텍스트 폭을 넘어 페이지 오른쪽 끝에서 잘림. 전부 마크다운 소스에서 텍스트로 교정 가능 → 수정함. (표 넘침·그림 잘림은 시각 리포트가 0건으로 확인했고, 템플릿이 `longtable`을 이미 로드해 큰 표는 자동 분할된다.)

| # | 결함 | 파일:행 | 수선 방법 |
|---|---|---|---|
| D1 | overfull display math — 5항 1줄 배치, 마지막 항 `α_t` 잘림 | `study-kr/part2/ch16-nested-learning.md:264` | `$$…$$` 를 `\begin{aligned}…\end{aligned}` 로 감싸 3항+2항 2줄 분할(`\\`). 5개 식·기호·부호 그대로. |
| D2 | overfull inline math list — `𝓜_mem` 잘림 | `study-kr/part2/ch16-nested-learning.md:383` | 한 덩어리 `$𝓜_k,…,𝓜_mem$` 를 심볼별 `$𝓜_k$, $𝓜_v$, …` 로 분리 → 콤마·공백에서 줄바꿈 허용. 5개 심볼 동일. |
| D3 | overfull `\texttt` run — `checkpoint` 이후 잘림 | `study-kr/part3/ch22-player-strategy.md:76` | `·` 앞뒤에 공백 삽입(`` `a`·`b` `` → `` `a` · `b` ``). 코드 스팬·구분자 동일, 공백만 추가. |
| D4 | 동상 재발(serving 계층 리스트) | `study-kr/part3/ch22-player-strategy.md:90` | 동상 — `·` 공백화. |
| D5 | 동상 재발(다섯 누락 API) | `study-kr/part3/ch24-proposals.md:146` | 동상 — `·` 공백화. |
| +1 | **같은 클래스 예방** — model-name 코드 리스트(5-span) | `study-kr/part3/ch21-hardware-lottery.md:67` | 시각 리포트엔 미표시(현재 페이지 중앙에 착지). 그러나 D1/D2 편집으로 reflow가 이 줄을 페이지 끝으로 밀 수 있는 동일 overflow-prone 패턴이라 선제 공백화. |
| +1 | **같은 클래스 예방** — `bind_to_sequence`·`mark_dirty` 2-span | `study-kr/part3/ch24-proposals.md:78` | 동상 — 규약 일관성 위해 공백화. |

**검증**: 변경된 세 구문(aligned 블록 / 심볼별 inline / 공백-`·` texttt)을 pandoc `-t latex` 로 실제 변환 → 각각 `\[\begin{aligned}…\end{aligned}\]`, 분리된 `\(…\)` 여러 개, 공백으로 끊기는 `\texttt{…} · \texttt{…}` 로 정상 산출. 템플릿(`build/template.tex`)은 `amsmath`(aligned 필수)·`longtable` 로드 확인. 컴파일 유효성 확보.

## 2. 결함 아님으로 확인(수정 안 함)

- **xref**: `final-xref-report.md` = genuine dangling 0, mismatch 0, 중복 0. `§1.7.x`(STYLE 문서 절 참조) house-style은 정본 관례로 유지. → 손댈 것 없음.
- **인용**: 참고문헌 §D "미해결 인용" = **(없음)**. 커밋 메시지의 "1 unresolved internal ref"(Kim et al. 2026)는 이미 §B.9로 편입 해소. → 손댈 것 없음.
- **그림**: `final-figure-report.md` 하드 결함 0(19-2 캡션 4-panel 불일치는 그림 QA가 이미 직접 수정). 15/15 장-번호·경로·캡션 정합.

---

## 3. deferred (템플릿/빌드/스크립트 레벨 — 사용자가 처리)

1. **[빌드] 전체 PDF 재빌드 필요.** 이번 수선은 `study-kr/` 소스 5개 파일에 반영됨. `build/BOOK.pdf`(및 `BOOK.md`/`BOOK.tex`)는 아직 이전 상태. 소스→PDF 조립 스크립트가 repo에 커밋돼 있지 않아(수동 pandoc+lualatex, template.tex 기반) 결정론적 재실행을 사용자 손에 남긴다. 스니펫 레벨 컴파일 유효성만 검증함, 309pp 아티팩트 재생성은 미실행.
2. **[템플릿/조립] D6 — Part 구성 비대칭.** 제1부만 한글 divider + 영문 "Part I …" 헤딩 이중(folio 13,14), 2·3부는 한글 divider 단일. front opener 조립/템플릿 레벨 선택 사항. 시각 결함 아님, 통일 원하면 template 측에서.
3. **[그림 재생성] D7 — matplotlib overplot.** 그림 19-2/25-1(exp-f 4-panel) 좌상단 라벨 겹침, 20-2(exp-b) 상단 라벨 겹침. PNG 원본 품질이라 텍스트로 불가 — figure 재생성 필요.
4. **[스크립트/정직성] exp-f 재현 경로 부재.** `figures/render_experiments.py` 가 exp-a..e만 생성; exp-f-scaling-fits.png의 render 경로가 어느 스크립트에도 없음(figure 리포트 §6.2). 정직성 계약(재현성) 갭 — 생성 코드를 SoT 스크립트에 편입 필요.
5. **[편집 정책] 그림 재사용 정책 비일관.** exp-* 이미지 2~4회 재임베드 vs SPEC "draw once, reuse everywhere" 텍스트 참조(figure 리포트 §6.1). 번호 충돌·정직성 리스크 없음 — part-분할 독자 편의 vs SPEC 준수의 human 편집 결정.

---

## 4. 판정

- 시각 리포트의 **수정 가능한 결함 D1–D5 전량 수선**(+동일 클래스 2건 선제) — load-bearing 불변, 컴파일 유효.
- dangling cross-ref / 미해결 인용: 잔여 0(입력 리포트가 이미 clean). 추가 정리 대상 없음.
- 나머지(빌드 재실행·part 비대칭·figure overplot·exp-f provenance·재사용 정책)는 텍스트로 해결 불가 → §3 deferred.
