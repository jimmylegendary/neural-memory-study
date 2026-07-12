# Final Visual Defect Scan — BOOK.pdf (309pp)

Round: 통합 수준 시각 결함 스캔 (per-chapter 리뷰가 못 잡는 것만)
Source: `/home/jimmy/repos/neural-memory-study/build/BOOK.pdf` (309 pages, A4, LuaTeX)
Method: (1) 대표 페이지 22장 100–200dpi 렌더 + 육안 검사; (2) 전체 309pp를 72dpi로 렌더해
오른쪽 여백 초과(page-edge 잉크) 자동 스캔 → 후보를 150–200dpi로 확대 재검. 텍스트 미수정, 위치·유형만 보고.

Tooling note: PDF page(렌더 index) vs folio(인쇄 쪽번호)는 6쪽 오프셋. 아래는 **folio(인쇄 쪽)** 기준, 괄호에 PDF page.

---

## 검사한 대표 페이지 (clean으로 확인)

- 제목/판권/차례 (folio i, 1–4): tofu 없음, 수식·로마자 혼용 정상.
- Part divider 3곳 — 제1부(folio 13), 제2부(folio 124), 제3부(folio 213): 제목만 있는 divider 페이지는 의도된 구성. 정상.
  - 참고: 제1부는 한글 divider(folio 13) 다음에 영문 "Part I …" 헤딩(folio 14)이 한 번 더 나오나, 2·3부에는 없음 — 결함은 아니고 구성 비대칭(참고용, 아래 D6).
- 수식 밀집 페이지 (folio 77, 95, 133, 188 등): SSD/duality 유도, (7-2)~(7-4), (9-1)(9-2), (M4), (12-7)(12-8) 등 대부분 여백 내 정상 렌더.
- 원문 재현 표: 표 7-1(folio 77), 표 18-1 매핑, 표 23-1(folio 270, memory-tier), glossary 표(folio 301), 예약기호 색인 표(folio 307) — 모두 텍스트 블록 내 수용, 넘침·잘림 없음.
- 그림 6장 전부 임베드 확인 — 잘림/왜곡/저해상/캡션분리 없음, 캡션이 그림 바로 아래 부착:
  - exp-a (그림 18-1, folio 217; 그림 19-1 등), exp-f (그림 19-2 folio 227, 그림 25-1 folio 290),
    exp-c (그림 19-3 folio 230), exp-b (그림 20-2 folio 241), exp-e, exp-d (그림 23-1 folio 270).
  - 그림 내부 라벨 겹침(예: 그림 19-2/25-1 좌상단 "Moneta"/"Gated DeltaNet" 라벨이 타이틀·데이터와 근접)은
    **원본 figure(matplotlib) 품질**이지 PDF 렌더 결함 아님 — 참고용(D7).
- 참고문헌(folio 295–), glossary index(folio 301–), 예약기호 색인(folio 307): tofu 없음, 정렬·정출 정상.

## 한글 빈칸(tofu)
- 전 페이지 텍스트 레이어 U+FFFD 0건, 렌더 육안 검사에서도 빈 글리프 상자 미발견. **tofu 없음.**

---

## 결함 (defects) — 모두 동일 근원: 줄바꿈 불가능한 긴 inline run이 텍스트 폭을 넘어 페이지 오른쪽 끝에서 **잘림**

per-chapter prose 리뷰가 못 잡는 통합 수준 결함. 자동 스캔에서 잉크가 페이지 가장자리(x≈595/596)에
닿는 쪽만 추려 확대 확인. (여백에 닿기만 하고 완결된 justified 텍스트·인용·수식 태그 — folio 104,158,189,297 등 — 은 결함 아님으로 제외.)

### D1 — 디스플레이 수식이 오른쪽 여백 밖으로 넘쳐 마지막 항이 잘림
- 위치: **folio 188 (PDF 189)**, §16.3.10 "1단계 — 완전 적응형 memory" 직후 디스플레이 수식.
- 증상: 5개 항 `k_t=…, v_t=…, q_t=…, η_t=…, α_t=𝓜_α(x_t; W_α…)` 가 한 줄에 배치되어 폭 초과.
  마지막 항이 `α_t = 𝓜_α(x_t; W_α` 에서 잘리고 `_{α,t-1})` 가 페이지 밖으로 사라짐(200dpi 확대 확인, 잉크 x=1644/1654).
- 소스: `study-kr/part2/ch16-nested-learning.md:264–270` (한 `$$…$$` 안에 `\quad`로 5항을 한 줄로 나열).
- 유형: overfull display math / 잘림.

### D2 — 인라인 수식 리스트가 여백을 넘어 마지막 심볼이 잘림
- 위치: **folio 192 (PDF 193)**, §16.4.2 첫 문단 "(1) fast memory들 (𝓜_k,𝓜_v,𝓜_η,𝓜_α,𝓜_mem …)".
- 증상: 인라인 수식 `$(\mathcal{M}_k,\mathcal{M}_v,\mathcal{M}_\eta,\mathcal{M}_\alpha,\mathcal{M}_{\mathrm{mem}}$`
  이 줄 끝에서 줄바꿈 불가로 폭을 넘어 `𝓜_mem`이 `𝓜_me`로 잘림(잉크 페이지 끝 도달).
- 소스: `study-kr/part2/ch16-nested-learning.md:383`.
- 유형: overfull inline math / 잘림.

### D3 — `\texttt`-코드 스팬을 `·`로 이어붙인 API 리스트가 잘림 (folio 259)
- 위치: **folio 259 (PDF 260)**, §22.4 마지막 문단.
- 증상: `TTT-state manager는 update_in_place·mark_dirty/writeback·checkpoint·…` 의 monospace 런이
  줄바꿈 불가로 페이지 끝에서 `…writeback·checkpoin` 처럼 `checkpoint` 이후가 잘림.
- 소스: `study-kr/part3/ch22-player-strategy.md:76`
  (`` `update_in_place`·`mark_dirty/writeback`·`checkpoint/rollback`·`bind_to_sequence` ``).
- 유형: overfull inline `\texttt` run / 잘림.

### D4 — 같은 API 코드 리스트가 인접 페이지에서 재차 잘림 (folio 260)
- 위치: **folio 260 (PDF 261)**, §22.5 "serving 계층 (update_in_place·mark_dirty/writeb…)".
- 증상: 동일 패턴 — `(update_in_place·mark_dirty/writeb` 에서 잘림.
- 소스: `study-kr/part3/ch22-player-strategy.md:90`.
- 유형: overfull inline `\texttt` run / 잘림. (D3와 동일 근원, 다른 인스턴스.)

### D5 — 다섯 누락 API 리스트가 잘림 (folio 285)
- 위치: **folio 285 (PDF 286)**, §24 요약 "소프트웨어 접착층은 제안 I … API(update_in_place·…·free_on_ope…)".
- 증상: `API(update_in_place·mark_dirty/writeback·checkpoint·rollback·bind_to_sequence·free_on_upd…)`
  monospace 런이 페이지 끝에서 `free_on_update` 근처가 잘림.
- 소스: `study-kr/part3/ch24-proposals.md:146`
  (`` `update_in_place`·`mark_dirty/writeback`·`checkpoint/rollback`·`bind_to_sequence`·`free_on_update` ``).
- 유형: overfull inline `\texttt` run / 잘림.

> **공통 근원 & Fix 방향(참고, 이 라운드에서는 미수정):**
> D1은 디스플레이 수식을 `aligned`/`split`로 2줄 분할. D2는 인라인 수식 항을 줄여 쓰거나 `\allowbreak`/문장 재배치.
> D3–D5는 `·`로 이어붙인 `\texttt{}` 스팬 사이에 줄바꿈 허용(`\allowbreak`/`\linebreak[0]`) 또는 세로 목록(bullet)로 전환.
> 이 세 클래스는 같은 API 문자열이 등장하는 곳마다 재발하므로, 소스에서 리스트 조판 규약을 한 번 정하는 것이 근본 교정.

---

## 참고 항목 (결함 아님)

- **D6** — 제1부에만 한글 divider + 영문 "Part I …" 헤딩이 이중(folio 13,14); 2·3부는 한글 divider 단일.
  구성 비대칭일 뿐 시각 결함 아님. 통일 원하면 확인 대상.
- **D7** — 그림 19-2 / 25-1(exp-f 4-panel) 좌상단 패널 라벨 겹침, 그림 20-2(exp-b) 상단 라벨 겹침은
  원본 PNG(matplotlib) 수준의 overplot이며 PDF 임베드/렌더 결함 아님. 가독성 개선을 원하면 figure 재생성 대상.

## 스캔 커버리지
- 육안 정밀 검사: folio i,1,13,77,95,104,124,133,158,188,192,213,217,227,230,241,259,260,270,285,290,295,301,307 (+ 확대 crop 다수).
- 자동 여백-초과 스캔: 309pp 전체 72dpi. 페이지 가장자리 도달 후보 전수 확대 재검 완료.
