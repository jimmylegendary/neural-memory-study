# W1 재감사(폐쇄 검증) — 표기·용어 축 (notation & terminology)

**감사자 축**: 표기·용어 (원 리포트 `audit/w1-notation-report.md`)
**목적**: P2 보정(`audit/w1-fixplan.md` + `audit/w1-todo-resolutions.md`)이 17개 장에 실제로 적용됐는지 **폐쇄(closure) 검증**. 원 리포트 critical/major 각 항목의 닫힘/미해결 판정 + 수정으로 새로 생긴 위반 표본 검사.
**방법**: 전 장 grep 전수 재검(C-1/C-2/C-3 중심) + 대응 라인 정독. 기준 문서 `style/STYLE-NOTATION.md` **v1.1**.
**검증일**: 2026-07-12

---

## 0. 총평

원 리포트의 표기·용어 축 **critical 3건(C-1/C-2/C-3) 전부 닫힘**, major 2건(M-1/M-2) 닫힘, 확인한 minor(N-m2/m4/m5/m7/m8) 전부 닫힘. 표본 검사에서 **P2가 새로 만든 표기·용어 위반은 발견되지 않았다**. 잔여는 단 1건의 저심각도 순서 기술결함(ch15 [TNT] 첫 등장이 풀 인용보다 2줄 앞섬)뿐이며, 이는 원 리포트 비신고 항목(사실상 pre-existing 추정)이다.

전 17개 장 sweep 결과: **어떤 장이 [X] 태그를 인용하면 그 장 안에 X의 arXiv ID가 반드시 존재**한다(예외 0건).

---

## 1. CRITICAL 폐쇄 판정

### C-1. "forget gate" 용어 정책 — **닫힘 (CLOSED)**

전 장 `grep "forget gate"` 결과 = 6곳, 전부 규칙 부합:

| 위치 | 현재 상태 | 판정 |
|---|---|---|
| `ch06:87` | "retention gate(→ 13장; 역사적 별칭 \"forget gate\")" | 적합 — 모범 패턴(무변경) |
| `ch02:6` | "retention gate(역사적 별칭 forget gate → 13장)의 원형" — **장 내 유일 병기 1회** | 닫힘. 구 위반 :138(절 제목)·:151·:273 전부 제거 확인(ch02는 이제 :6만, 본문은 "forgetting mechanism"(:153) 사용) |
| `ch12:9` | "retention gate(원문 표현 forget gate)까지 더했다" — 병기 1회 | 닫힘. 구 위반 :164×2·:286 제거 확인(ch12 잔존 forget gate = :9 병기 + :177 대응표만) |
| `ch12:177` | 대응표 행 "$\alpha_t$ — forget gate …" | 허용(원 표기 대응표) |
| `ch13:13` | "(당시 명칭) forget gate"×2 — 인용 처리 | 닫힘(fixplan D 지시대로 Miras 역사 서술을 "(당시 명칭)"으로 한정) |
| `ch13:19` | 개명 선언 문단("역사적 별칭 …는 이 문장 한 번으로 병기를 마친다") | 닫힘(정의 소유 장의 선언) |
| `ch13:315` | 대응표 행 "forget gate → retention regularization" | 허용(대응표) |

구 위반 `ch13:77` "고전적 forget gate가 사는 곳"은 제거됨(현 grep 미검출 = "고전적 gate"로 교체 확인). ch02·ch12의 반복 위반 전건 소거.

### C-2. 논문 첫 등장 풀네임+arXiv ID — **닫힘 (CLOSED, minor 순서 잔여 1건)**

원 신고 5장 + ch15 순서 전건 확인:

| 장 | 원 누락 | 현재 | 판정 |
|---|---|---|---|
| `ch05` | [Titans]/[Miras]/[Atlas]/[NL] 전부 무ID | 목표·왜필요 박스(:5·:7·:9)에 4편 전부 풀네임+ID 병기 | 닫힘 |
| `ch11` | [Titans] 누락 | :214 첫 등장에 풀네임+ID(2501.00663) 병기; NL/Sleep은 :9 기존 완비 | 닫힘 |
| `ch13` | [Titans](:5)·[Atlas](:366) | :5 Titans 풀+ID, :366 bridge-out Atlas 풀+ID | 닫힘 |
| `ch14` | [NL]·[Sleep](:258) | :260에 NL/Sleep 풀+ID(첫 등장 co-located), TNT :307 풀+ID | 닫힘 |
| `ch16` | [Titans]/[Miras]/[Atlas]/[TNT]/[Sleep] | :5(TNT)·:7(Titans/Miras/Atlas)·:9(NL)·:15(Sleep) 전부 풀+ID | 닫힘 |
| `ch15` | [NL] 순서 위반(:296 → :260) | NL 풀 인용을 :262 첫 등장으로 이동, 구 :296 풀 인용 제거 확인(`grep "Nested Learning"` = :262 단일) | 닫힘(N-m10 포함) |

**17개 장 전수 sweep**: 태그↔ID 누락 0건.

**잔여(저심각도)**: `ch15:9`의 첫 [TNT] 언급("[TNT]가 LaCT…")이 풀 인용 `ch15:11`("[TNT] (*TNT: …*, arXiv:2511.07343)")보다 2줄 앞선다. §2.5 자구상 첫 등장 순서 기술결함이나 (i) TNT는 ch15 자체 논문, (ii) 원 리포트가 ch15에서 TNT를 신고하지 않음(NL만) → pre-existing 추정. 차기 라운드 이월 권고(:9의 hook 뒤에 풀 인용을 두거나 :11 풀 인용을 :9로 당김).

### C-3. 조사 결합 "loop이/은/을" — **닫힘 (CLOSED)**

원 신고 13건(ch13:287×3·289, ch14:19·199×4·213×2·215·258, ch15:19·87·91) 전건 소거. 전 장 `grep "loop이\|loop은\|loop을"` 잔존 = 2건뿐이며 둘 다 **계사(繫辭) 활용**으로 원 리포트가 명시 제외한 비위반:

- `ch02:271` "…의 loop이며" (= "loop이다"의 활용)
- `ch16:362` "inner loop이되 objective는…" (대응표 행, "loop이다"의 활용)

격조사 위반(이/은/을 → 가/는/를) 신규 발생 0건.

---

## 2. MAJOR 폐쇄 판정

### M-1. 파일 슬러그 §5.1 불일치 — **닫힘 (D10)**
STYLE v1.1 §5.1이 실파일명으로 정본화(코디네이터). ch01/03/04의 자기신고 STYLE-ISSUE 주석 제거는 D10 소관(cross-axis). 표기 축 관점에서 잔여 위반 없음.

### M-2. 예약 기호 $L$(sequence)을 MLP 깊이로 재사용 — **닫힘 (CLOSED)**
- `ch02:63` → `$L_{\mathrm{layer}}$층 MLP`($\ell=1..L_{\mathrm{layer}}$) 확인.
- `ch16:68` → `$L_{\mathrm{layer}}$층 MLP $\{W_\ell\cdot+b_\ell\}_{\ell=1}^{L_{\mathrm{layer}}}$` 확인. 같은 장의 sequence 길이 $L$(:208 등)과 분리됨.
- 잔존 "L층 MLP"(무첨자) 0건.

---

## 3. MINOR 폐쇄 판정 (표기 축)

| ID | 항목 | 판정 | 근거 |
|---|---|---|---|
| N-m2 | ch16:305 $o_t$ → $z_t$ | 닫힘 | :307 `z_t=\mathcal{M}(q_t;…)`, :346 대응표에 장-국소 행 "$o_t$ → $z_t$" 추가(ch12 $a_\tau$ 패턴 준수). 본문 잔존 $o_t$ = 대응표 설명행뿐 |
| N-m4 | ch01 $(1-\lambda)$ → $(1-\eta\lambda)$ | 닫힘 | ch01:142에 `$(1-\eta\lambda)$` 확인; 구 :228 bare $(1-\lambda)$는 분량 감축(D8)으로 제거, 잔존 0건 |
| N-m5 | GDN 약어 병기 | 닫힘 | ch06:117 "Gated DeltaNet(이하 GDN)", ch13:5 "Gated DeltaNet(이하 GDN)" — 둘 다 장 첫 GDN 지점. ch12는 bare GDN 0건(풀네임 유지) |
| N-m7 | ch15:9 "大규모" → "대규모" | 닫힘 | ch15 `grep "大"` = 0건 |
| N-m8 | "망각" 명사 → forgetting | 닫힘 | 전 장 `grep "망각"` = 0건(구 ch02:136·ch05:224 소거) |
| N-m1 | ch14:31 vec–Kronecker $\otimes$ | 닫힘(무변경) | v1.1 §1.3 예외 확장으로 소급 적합. 표기 잔존, 규칙상 합치 |
| N-m3 | 재참조 없는 수식 번호(14건) | 미검증 | fixplan상 "참조 추가/태그 제거 택일" 허용 minor. 이번 축 폐쇄 검증에서 전수 대조 안 함(차기 확인 권고) |
| N-m6 | 표 캡션 볼드→플레인(ch04/05/16) | 미검증(cross-axis) | D11 일괄 결정 소관 |

---

## 4. 신규 위반 표본 검사 (P2가 만든 것 없음)

- **금지 $\mathcal{M}^*$(read-only)**: 본문 산문 0건. 검출된 4곳(ch12:174·187, ch13:254, ch16:324)은 전부 "원 표기 → 통일 표기" **대응표 행**으로 허용.
- **$\theta_t$ = inner lr 금지**: 위반 0건. 검출 $\theta_t$는 대응표(ch12:175, ch14:182) 또는 CMS level parameter/일반 optimizer 유도(ch16:42·54–62, §1.2 `θ^{(ℓ)}` 정당)로 전부 합치.
- **한글 음차·번역**: `grep "모멘텀\|청크\|놀라움\|연상 기억\|가중치 감쇠\|Ω-rule"` = 0건.
- **loop 격조사 과교정**: 신규 "loop가/는/를" 오용 없음(계사 2건은 정상).
- **retrieval gap 수치**(K-F06, 인접 확인): ch16:446·ch17:317 "53.55 vs 43.70"으로 통일 확인, ch14:319 기준값과 일치("53.6/43.7" 잔존 0건).
- **ch01 대량 감축 후 C-2 회귀**: 없음(arXiv ID 3건 :9·:15·:97 생존, 표 :104 등 인용 형식 유지).

---

## 5. 폐쇄 요약표

| 원 항목 | 심각도 | 판정 | 비고 |
|---|---|---|---|
| C-1 forget gate | critical | **닫힘** | 6곳 전부 규칙 부합 |
| C-2 arXiv ID | critical | **닫힘** | 17장 sweep 태그↔ID 누락 0; ch15 TNT 순서 minor 잔여 1 |
| C-3 loop 조사 | critical | **닫힘** | 13건 소거, 잔존 2건은 계사(비위반) |
| M-1 파일 슬러그 | major | 닫힘(D10, cross-axis) | v1.1 정본화 |
| M-2 $L$ 재사용 | major | **닫힘** | ch02·ch16 → $L_{\mathrm{layer}}$ |
| N-m2 $o_t$→$z_t$ | minor | 닫힘 | 대응표 행 추가 |
| N-m4 $(1-\eta\lambda)$ | minor | 닫힘 | |
| N-m5 GDN | minor | 닫힘 | |
| N-m7 大규모 | minor | 닫힘 | |
| N-m8 망각 | minor | 닫힘 | |
| N-m1 ⊗ | minor | 닫힘(무변경) | v1.1 예외 |
| N-m3 수식 번호 | minor | 미검증 | 택일 허용, 차기 확인 |
| N-m6 캡션 서식 | minor | 미검증 | cross-axis |

**결론**: 표기·용어 축 critical·major 전건 폐쇄. 신규 위반 없음. 이월 = ch15 TNT 첫 등장 순서(저심각도), N-m3 수식 번호 전수 대조(차기).
