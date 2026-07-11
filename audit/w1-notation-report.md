# W1 감사 보고서 — 표기·용어 일관성 축 (notation & terminology)

**감사자 축**: 통일 표기법 위반 / 용어 정책 위반 / 수식 번호·cross-ref 규약 위반
**기준 문서**: `style/STYLE-NOTATION.md` (v1.0, W1 동안 동결)
**대상**: `study-kr/part1/ch01–ch11`, `study-kr/part2/ch12–ch17` (17개 장 전수 정독)
**판정 기준**: critical = 다음 라운드에서 반드시 수정 / major = 병합(P2) 전 수정 / minor = 일괄 정리 대상

---

## 0. 총평

17개 장 전체가 통일 표기법의 **핵심 규약을 매우 높은 수준으로 준수**한다. 구체적으로 다음은 전 장에서 위반이 발견되지 않았다.

- $W$(fast) vs $\Theta$(slow) 구분, $\ell$(inner) vs $\mathcal{L}$(outer) 구분 — TNT의 원문 $\mathcal{L}$(inner)까지 정확히 $\ell$로 환원(ch15).
- 게이트 3종($\eta_t/\beta_t/\alpha_t$)의 의미·방향 — Titans의 $\theta_t{\to}\eta_t$, $\eta_t{\to}\beta_t$ 교차와 $\alpha_t$ 방향 반전(ch12), Atlas의 $\theta_t{\to}\beta_t$(ch14) 모두 대응표+본문 양쪽에서 일관 처리.
- 금지 기호 개명: TNT $V{\to}W^{\mathrm{g}}$, $S_{L_i}{\to}L_{\mathrm{s}}^{(i)}$, $\mathcal{M}_t{\to}\Pi_t$(ch15); NS 반복수 $k$ 금지 → $\mathrm{NS}_\kappa$(ch02/14/16); $b{\to}C$(ch12/13/14); $\mathcal{M}^*$ read-only 금지 / $\mathcal{M}^\star$ argmin 전용(ch12/13/14/16); Sleep $\gamma{\to}\rho$, $\alpha{\to}\lambda_{\mathrm{KD}}$, $\lambda{\to}\lambda_{\mathrm{on}}$(ch17).
- $C$(chunk) vs $c$(Omega window) 구분(ch09/14/15), 부호 관행의 (M2)/(M3) 고정과 "부호 관행 차이(알고리즘 동일)" 기록(ch12/13/14/16).
- Part II 6개 장 모두 §1.7 대응표를 원형 그대로 복사 + 장-국소 행 추가만 수행(행 삭제·의미 변경 없음). 장-국소 기호 충돌 회피(ch12의 $h_t{\to}r_\tau$, $S^{(t)}{\to}X^{(n)}$; ch13의 smoothing $\alpha{\to}\nu$; ch16의 M3 $\alpha{\to}\mu$; ch17의 $f{\to}\varepsilon$, $C{\to}x_{\mathrm{ctx}}$, $b{\to}n_{\mathrm{rand}}$)는 모범적.
- 수식 번호 체계: `\tag{N-k}` 형식·장내 연번·중복 없음, 존재하지 않는 태그를 참조하는 곳 0건. 원문 수식은 전부 `[X Eq. N]` 형식으로 책 번호와 혼용 없음. 산문 cross-ref는 전부 "→ N장" 텍스트 형식, Markdown 링크 0건.
- "Omega rule" 표기(Ω-rule 없음), 한글 음차 금지(모멘텀/청크/운동량/가중치 감쇠/놀라움/연상 기억 0건).

아래 발견 사항은 이 높은 기저 위의 잔여 결함이다. critical 3건은 전부 기계적으로 수정 가능하지만, 동결된 STYLE-NOTATION의 명시 규칙에 대한 반복적·교차 장 위반이므로 다음 라운드에서 반드시 정리해야 한다.

---

## 1. CRITICAL

### C-1. "forget gate" 용어 정책 위반 (ch02·ch12·ch13, 교차 장 반복)

§2.4 concept ledger: **retention gate가 공식 용어**이고 "forget gate"는 역사적 별칭으로 "**첫 등장 시 1회 병기만 허용**"이다(Titans 문맥에서 허용되는 것은 "forgetting mechanism" 인용뿐). 실태:

| 위치 | 내용 | 판정 |
|---|---|---|
| `part1/ch06-linear-attention-fwp.md:87` | "retention gate(→ 13장; 역사적 별칭 \"forget gate\")" | **적합** — 규칙이 의도한 유일한 사용 패턴 |
| `part1/ch02-training-as-a-system.md:6` (목표 박스), `:138` (**절 제목 §2.5.3 "…(1-ηλ)라는 forget gate"**), `:151`, `:273` (요약) | "forget gate"를 weight decay의 공식 별명처럼 4회 반복, 절 제목에까지 사용 | 위반 — "retention gate의 원형(→ 13장)" 계열 표현으로 교체 필요. 특히 절 제목 |
| `part2/ch12-titans.md:9` (2회), `:164` (2회), `:286` | Gated DeltaNet·TTT의 특징 서술에 "forget gate" 반복 (":179" 대응표 행은 허용) | 위반 — "retention gate(원문 표현 forget gate)" 1회 후 retention gate로 통일하거나, Titans 인용 문맥이면 "forgetting mechanism"으로 |
| `part2/ch13-miras.md:13` (2회), `:77` | **ch13:19이 스스로 "역사적 별칭 'forget gate'는 이 문장 한 번으로 병기를 마친다"고 선언**하고도, 그 선언 **앞**(:13)에서 2회, **뒤**(:77 "고전적 forget gate가 사는 곳")에서 1회 사용 — 장 내부 자기모순 | 위반 — :13은 Miras의 역사 서술이므로 "(당시 명칭) forget gate"류 인용 처리로, :77은 "고전적 gate"로 수정 |

**수정 지침**: 장마다 1회의 병기(ch06 패턴)만 남기고 전부 retention gate로 통일. ch02는 정의 소유 장(ch13)을 침범하지 않도록 "상수 retention(역사적 별칭 forget gate → 13장)" 형태 권장.

### C-2. 논문 첫 등장 시 풀네임+arXiv ID 병기 규칙(§2.5) 누락 — 5개 장

§2.5: 6편 축약 [Titans]…[Sleep]은 "**첫 등장 시(장마다 1회)** 풀네임+arXiv ID 병기". 다음 장은 해당 논문을 다수 인용하면서 장 내 어디에도 ID가 없다(기계 검색으로 확인).

| 장 | 누락 논문 (장 내 인용 횟수) |
|---|---|
| `part1/ch05-associative-memory.md` | [Titans]×6, [Miras]×3, **[Atlas]×21**, [NL]×8 — 전부 ID 없음 |
| `part1/ch11-continual-learning.md` | [Titans]×1 |
| `part2/ch13-miras.md` | [Titans]×2 (:5 첫 등장), [Atlas]×1 (:366) |
| `part2/ch14-atlas.md` | [NL] (:258, :320, :330), [Sleep] (:258, :330) |
| `part2/ch16-nested-learning.md` | [Titans]×6, [Miras]×3, [Atlas]×2, [TNT]×3, [Sleep]×6 — [NL]만 :9에서 완비 |

부수: `part2/ch15-tnt.md`는 [NL]을 :260·:288에서 먼저 인용하고 풀 인용은 :296에 온다(순서 위반). ch01·ch02·ch03·ch04·ch08·ch09·ch10·ch12·ch17은 모범적으로 완비.

**수정 지침**: 각 장의 해당 논문 첫 등장 지점에 "[Atlas] (*Atlas: …*, arXiv:2505.23735)" 형식 1회 삽입. "(→ 12장)" 참조가 붙어 있어도 §2.5는 장 단위 규칙이므로 면제되지 않는다.

### C-3. 조사 결합 규칙(§2.2) 위반 — "loop이/loop은/loop을" (ch13·ch14·ch15, 13건)

§2.2 표가 **명시적으로** "inner loop가 = 옳음 / inner loop이 = 틀림"을 규정한다(발음 "루프" 기준 → 가/는/를). 위반 목록(격조사만; `ch02:269` "loop이며", `ch16:359` "loop이되"는 계사 활용이라 위반 아님):

- `part2/ch13-miras.md:287` — "inner loop이"×2, "outer loop은"×1; `:289` — "inner loop을"
- `part2/ch14-atlas.md:19` — "inner loop이"; `:199` — "inner loop이"×2, "outer loop은"×2; `:213` — "outer loop은", "inner loop이"; `:215` — "inner loop을"; `:258` — "inner loop은"
- `part2/ch15-tnt.md:19` — "inner loop은"; `:87` — "loop이"; `:91` — "inner loop을"

같은 장들이 다른 자리에서는 올바른 "loop가/loop는/loop를"을 쓰고 있어(장 내부 혼용) 일괄 치환이 필요하다: 이→가, 은→는, 을→를.

---

## 2. MAJOR

### M-1. 파일명 슬러그가 §5.1 고정 목록과 불일치 — 4개 파일 (ch09는 미신고)

§5.1 "파일명 슬러그 (고정 — 임의 변경 금지)" 대비:

| §5.1 슬러그 | 실제 파일 | STYLE-ISSUE 신고 |
|---|---|---|
| ch01-orientation.md | ch01-orientation-**rosetta**.md | 있음 (`:3`) |
| ch03-online-learning.md | ch03-online-learning-**regret**.md | 있음 (`:3`) |
| ch04-meta-learning.md | ch04-meta-learning-**bilevel**.md | 있음 (`:3`) |
| ch09-chunkwise-parallel.md | ch09-chunkwise-parallel-**training**.md | **없음** |

P2 pandoc 병합·cross-ref 앵커 생성 시 전원이 이 목록을 참조하므로, (a) §5.1을 v1.1에서 실파일 기준으로 갱신하거나 (b) 파일명을 환원하는 단일 결정이 필요. 최소한 ch09에 STYLE-ISSUE 주석을 추가할 것.

### M-2. 예약 기호 $L$(sequence 길이)을 MLP 깊이로 재사용 — ch02·ch16, 장-국소 선언 없음

§1.2는 $L$ = sequence 길이로 예약하고, memory 깊이에는 $L_{\mathcal{M}}$을 배정했다.

- `part1/ch02-training-as-a-system.md:63` — "구체적으로 $L$층 MLP를 보자" ($W_\ell$, $\ell=1..L$). ch02는 $L$을 sequence 길이로 거의 쓰지 않아 실해는 작으나 선언이 없다.
- `part2/ch16-nested-learning.md:68` — "$L$층 MLP $\{W_\ell\cdot+b_\ell\}_{\ell=1}^{L}$" — **같은 장에서** $L$이 sequence 길이로도 상시 사용된다(예: `:208` "$\sum_{i=1}^{L}$", RULER 문맥). 한 장 안에서 두 의미가 공존하는 유일한 사례. ch16 §16.7은 layer 수에 $L_{\mathrm{layer}}$를 쓰고 있으므로, `:68`도 $L_{\mathrm{layer}}$(또는 국소 선언)로 통일 권장.

---

## 3. MINOR

1. **Kronecker $\otimes$** — `part2/ch14-atlas.md:31`의 Prop. 1 증명 스케치 "$(K^\top\otimes I_{d_v})\,\mathrm{vec}(W)$". §1.3의 $\otimes$ 금지에서 허용된 예외는 $x^{\otimes i}$뿐. gating 혼동 위험은 없으나 동결 규칙의 자구 위반 — vec 표기를 산문으로 풀거나 `<!-- STYLE-ISSUE -->` 주석으로 예외 등재 요청 권장. (ch14:46의 $x^{\otimes 2}$류는 허용 예외에 해당.)
2. **NL의 $o_t$** — `part2/ch16-nested-learning.md:305` Hope forward에서 $o_t$를 중간 출력으로 사용. §1.2는 "TNT/NL의 $o_t$를 $y_t$로 통일"로 예약. 두 출력(중간/최종)이라 구분 필요성은 인정되나 장-국소 선언도, 표 16-2의 대응 행도 없다. $z_t$류 장-국소 기호로 개명 + 대응표 행 추가 권장 (ch12는 같은 상황에서 $a_\tau$로 개명하고 표에 등재 — 그 패턴을 따를 것).
3. **재참조 없는 수식 번호 14건** — §5.3 "재참조되는 것만 번호를 붙인다" 위반: (3-6), (12-5), (12-6), (12-8), (13-2), (13-4), (13-5), (13-6), (13-9), (14-2), (14-5), (15-2), (15-5), (16-2). 번호 제거 또는 본문에서 실제 참조 추가 중 택일(대부분 요약 bullet에서 한 번 참조해 주면 자연 해소).
4. **weight decay 인자 표기 불일치** — `part1/ch01-orientation-rosetta.md:228` "$(1-\lambda)$" vs ch02 전체의 "$(1-\eta\lambda)$"(decoupled decay의 올바른 형태). ch01을 $(1-\eta\lambda)$로 수정.
5. **"GDN" 약어 미도입 + 혼용** — `part1/ch06:253` 표, `part2/ch13:120,334,348,350`에서 "GDN"/"GDN-H2"를 정의 없이 사용. ch12는 같은 대상을 "Gated DeltaNet-H2"로 풀네임 표기. 첫 사용 시 "(이하 GDN)" 병기 또는 풀네임 통일.
6. **표 캡션 서식 혼재** — ch04·ch05·ch16은 "**표 N-M — …**"(볼드), 나머지 장은 "표 N-M — …"(플레인). §5.4는 위치(표 위)만 규정하므로 규칙 위반은 아니나 P2 일괄 병합 전 한쪽으로 통일 권장. 캡션 위치 자체는 17개 장 전부 준수.
7. **한자 오식** — `part2/ch15-tnt.md:9` "大규모로" → "대규모로".
8. **"망각" 명사 사용** — `part1/ch02:136` "continual learning에서의 망각", `part1/ch05:224` 표 행 라벨 "망각". §2.1 금지 목록(~~망각 게이트~~)의 인접 사례로, 개념 명사는 forgetting(또는 catastrophic forgetting → 11장)으로 통일 권장. ch11은 "forgetting"을 일관 사용 중이라 장 간 불일치가 생긴 상태.
9. **"weights" 단독 복수형의 전면적 사용** — §2.1은 복수형 s 지양(예외: fast weights/slow weights)인데, 사실상 모든 장이 "그 weights 자체가", "memory weights" 등 단독 "weights"를 표준으로 쓴다(수백 건). 전수 수정은 비현실적·비생산적이므로 **v1.1에서 "weights" 단독 사용을 허용 예외로 명문화**하는 쪽을 권고 (P2 결정 사항).
10. **ch15 [NL] 인용 순서** — C-2에 포함되나 성격이 다름: 풀 인용(:296)이 존재하므로 첫 등장(:260)으로 이동만 하면 됨.

---

## 4. 규약 준수 확인 목록 (위반 없음 — 다음 라운드 회귀 검사용 기준선)

- [x] (M)/(M1)–(M5) 전역 번호 인용 규약: ch01의 `\tag{M}`, ch09의 `\tag{M4}` 재게시는 §5.3 허용 범위.
- [x] 존재하지 않는 수식 태그 참조: 0건. 장 간 수식 참조 "식 (9-2)(→ 9장)" 형식 준수.
- [x] $\gamma$ 사용처: window gate 전용 준수. ch13의 $\mathcal{S}_\gamma$는 장-국소 선언 완비(:198, 표 13-2).
- [x] $\delta$ 이중 용법: backprop $\delta_\ell$(층 첨자) vs Huber $\delta_t$(시간 첨자) 구분 유지 (ch02/13/16).
- [x] $m_t,h_t$ outer moment / $S_t$ inner momentum 분리; Adam $v_t{\to}h_t$ 개명 명시(ch02:155).
- [x] 첨자 규약(data-dependent는 $\eta_t$, 상수는 $\eta$) — ch02:53, ch12:249 등에서 명시적 재확인.
- [x] 정의 소유권: 재정의 침범 사례 발견 못함(retention gate 정의는 ch13, capacity formal은 ch14, chunk=semantic은 ch09가 소유하고 타 장은 참조만).
- [x] 그림 규약: 그림 없음(위반 불가). 표 캡션 위치 전부 표 위.
- [x] TODO-VERIFY 형식(주장+확인 방법) 준수, 수치의 출처 병기("ppl 13.78 [TNT Fig. 2]" 등) — 표본 검사 범위에서 위반 없음.

---

## 5. 수정 우선순위 요약

| # | 심각도 | 항목 | 파일 수 | 수정 난도 |
|---|---|---|---|---|
| C-1 | critical | forget gate → retention gate 정책 위반 | 3 (ch02, ch12, ch13) | 낮음 (자구 치환 + ch02 절 제목) |
| C-2 | critical | 장별 풀네임+arXiv ID 누락 | 5+1 (ch05, ch11, ch13, ch14, ch16 + ch15 순서) | 낮음 (삽입) |
| C-3 | critical | 조사 loop이/은/을 (13건) | 3 (ch13, ch14, ch15) | 낮음 (치환) |
| M-1 | major | 파일 슬러그 §5.1 불일치 (ch09 미신고) | 4 | P2 결정 필요 |
| M-2 | major | $L$ = MLP 깊이 재사용 | 2 (ch02, ch16) | 낮음 |
| m-1…m-10 | minor | §3 목록 | 다수 | 일괄 정리 |
