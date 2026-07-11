# W1 재감사 — 커버리지·분량·TODO 축 (폐쇄 검증)

- 재감사일: 2026-07-12 / 검증자 축: coverage·length·TODO (원 리포트 `audit/w1-coverage-report.md`)
- 대상: `study-kr/part1/ch01–ch11`, `study-kr/part2/ch12–ch17` (17장 전량, 현 working tree 기준)
- 방법: fixplan(`w1-fixplan.md`) exit 조건 §357 + TODO 판정(`w1-todo-resolutions.md`)이 본문에 실제 반영됐는지 grep 전수 + 표본 정독 대조
- **판정 요약: PARTIAL CLOSURE.** TODO 축은 닫혔다(TODO-VERIFY 0건, 판정 전건 반영). 분량 축은 critical 3건 중 ch01만 닫히고 **ch10·ch16 두 hard 기준이 미달**. 커버리지 축은 닫힘(감축으로 인한 내용 손실 미검출, FIG 세트 완비).

> **선행 사실(중요)**: git HEAD(`2850e44`)는 "p2 partial … 9 chapter fixers (rate-limited mid-run)"이나, working tree에는 8개 파일이 추가로 수정되어(uncommitted) 있어 **17장 전부에 P2가 실질 반영된 상태**다. 본 재감사는 working tree를 대상으로 한다. 커밋 미정리(uncommitted)는 형상관리상 잔여 작업이다.

---

## 1. TODO 축 — CLOSED ✅ (프로세스 잔결 2건)

### 1.1 TODO-VERIFY 잔존 grep 전수 = **0건** ✅
```
grep -r "TODO-VERIFY" study-kr/  → 0
grep -r "TODO"        study-kr/  → 0   (어떤 변형도 없음)
```
exit 조건 §360(의무분 0건 잔존)의 핵심 조건 충족. 37건 전량이 본문에서 제거됐다.

### 1.2 판정(w1-todo-resolutions.md) → 본문 반영 표본 대조 — 전건 착지 확인 ✅

CORRECTED 6건(가장 위험)은 전부 본문에 착지했다:

| 항목 | 판정 | 본문 착지 확인 |
|---|---|---|
| ch04:129 von Oswald 범위 | CORRECTED | ✅ ch04:126 "합성 linear regression 과제에 훈련된 self-attention-only … 단층 weight 수준 일치, 다층 GD++류" + `[von Oswald 2023 abstract, §3–4]` |
| ch06:138 Longhorn 채널별/대각근사 | CORRECTED | ✅ ch06:145 각주 — "출력 채널별 가중 노름 … $\epsilon_{t,i}$ … 대각 근사 $\mathbf{1}-\epsilon_{t,i}k_t^{\odot2}$" + `[Longhorn Eq.5, Thm 3.1; §3.2]` |
| ch06:167 RWKV-7 projection | CORRECTED | ✅ ch06:158 "같은 key의 서로 다른 채널별 변조 … lerp … 별도 projection 행렬을 더 두는 것이 아니라" + `[RWKV-7 Eq.6–7,15]` |
| ch08:65 동치 정리 전제 | CORRECTED | ✅ ch08:64 "`[Sun 2024 Theorem 1]` … 전제: $f=Wx$(LN/residual 제거), batch GD, $\eta=1/2$, $W_0=0$; 무정규화 linear attention) … `Theorem 2` — Nadaraya–Watson + exp kernel" |
| ch13:332 outer optimizer | UNRESOLVABLE→헤지 | ✅ 표 13-3(ch13:285/287) "optimizer 종류·batch·스케줄 원문 미명시 … peak LR만" + ch13:332 Atlas App.E 대조 문장 |
| ch17:9 / ch17:202 | CORRECTED | ✅ ch17:7 "offline consolidation … [NL] 스스로 범위 밖" / ch17:203 "Llama-3B/8B에 5×dim-64 MLP graft … regime 1 미실증" |

위험도 '높음' 잔여(ch16:297·394)도 착지: ch16:298 "(α_tI−η_tk_tk_t^⊤)가 어느 행렬에 적용되는지 원문 미명시", ch16:396 "§9.6 산문 'removes the momentum term' … 수식 위치는 원문에 없음". CONFIRMED 다수도 인라인 인용으로 강화됨(ch05:72/74/80/84 AGS·Krotov·Ramsauer, ch07:54/112 Mamba Thm1·8×, ch12:139 Eq.21 차원논증, ch14:39 Prop2). 부록 B의 추가 수정(ch13:291 `[Miras §5.4]`→`§5.3`)도 전 위치 반영됨.

### 1.3 프로세스 잔결(내용 결함 아님)
- **G1**: fixplan D9/§6.3가 지정한 검증 이력 SoT `audit/w1-verify-answers.md`가 **부재**. 이력은 `w1-todo-resolutions.md`(다른 파일명)에 존재하므로 정보 손실은 없으나, 지정 산출물명 불일치.
- **G2**: `<!-- VERIFIED(...) -->` 치환 규약이 **3건에만** 적용(ch16×2, ch17×1). 나머지 ~34건은 TODO 주석 삭제 + 인용을 본문 산문에 접합(§6.3의 대체 경로로 허용). 감사 추적이 고르지 않으나 §6.3 위반은 아님.

---

## 2. 분량 축 — PARTIAL ❌ (critical 2건 미달)

`wc -w` 재측정(원 리포트와 동일 방법). exit 조건 §361 기준 판정:

| 장 | 원 실측 | 재측정 | 목표(§361) | 판정 |
|---|---|---|---|---|
| **ch01** | 3,936 | **2,198** | ≤2,200 (hard) | **CLOSED** ✅ (경계 −2) |
| **ch10** | 4,734 | **3,867** | ≤3,160 (hard) | **미달** ❌ **+707 (+22%)** |
| **ch16** | 8,621 | **7,699** | 6,500–7,150 (hard) | **미달** ❌ **+549 (상한 +8%)** |
| ch03 | 4,485 | 3,794 | ≤3,795 (hard/major) | CLOSED ✅ (경계 −1) |
| ch06 | 4,253 | 5,079 | ≥5,000 (hard/major) | CLOSED ✅ |
| ch13 | 5,537 | 5,519 | ~5,060 (soft) | 미달(soft) — 사유기록 부재 |
| ch15 | 5,172 | 5,071 | ~4,430 (soft) | 미달(soft) — 사유기록 부재 |
| ch17 | 6,013 | **6,058** | ~5,690 (soft) | 미달(soft) — **오히려 +45 증가** |

**분석**
- **ch10 (critical, hard) 미달이 가장 크다.** 지시된 감축은 실행됐다 — 표 10-1이 차분표로 교체됨(ch10:17, 열 = [inference 어휘 | 이 라인의 어휘 | 확립한 장]; "대응의 성격" 산문열 삭제, "상세는 표 1-1(→1장)" 안내 포함), glossary 유지·MoE 행 추가(ch10:212). 그러나 4,734→3,867(−18%)은 목표 −33%에 크게 못 미친다. 707단어가 hard cap을 초과한 채 남아 있다.
- **ch16 (critical, hard) 상한 초과.** 8,621→7,699(−11%). 목표 상한 7,150을 549단어 넘는다. 감축 대상이었던 §16.3.5–16.3.6 유도 산문은 여전히 두껍다.
- **ch17 (soft) 역주행.** CORRECTED 2건(ch17:9, ch17:202)이 순(純) 산문을 더했고 상쇄 감축이 없어 6,013→6,058로 늘었다. soft 기준이라 exit 실패는 아니나, 감축 라운드의 의도와 반대 방향이며 fixplan §361이 요구한 "미달 시 사유 기록"에 해당하는 산출물이 없다(커밋 메시지에도 없음).
- ch01·ch03·ch06은 hard 기준을 정확히 만족(ch01·ch03은 1–2단어 경계로 아슬아슬 — 재측정 방법 ±10% 오차 감안 시 사실상 목표선상).

---

## 3. 커버리지 축 — CLOSED ✅ (감축이 내용 손실을 일으키지 않음)

### 3.1 감축 3장 표본 정독 — 보존 요소 전부 생존
- **ch01**: 표 1-1(ch01:48)·표 1-2(ch01:99)·식 (M)(ch01:93)·(M) 선행 제시·delta rule/(M1) worked micro(ch01:116)·systems bridge 모두 유지. worked example은 d=2 1개로 축약되고 수치 전개를 "2장의 worked example에서 완성"·"정밀 계산은 10장 budget 표"로 위임(ch01:116, 122) — **지시대로의 위임이지 삭제가 아님**.
- **ch10**: cheat sheet(표 10-2)·budget 표(표 10-3, 4K/64K/1M)·prefill/decode 비대칭(§10.4)·AI(C)·roofline(§10.5)·glossary(§10.7, 행 삭제 없이 압축; 파이프행 96) 전부 생존. 차분표 교체는 내용 이관이 아니라 중복 산문열 제거.
- **ch16**: 5-mechanism knowledge-transfer taxonomy(§16.3.4, 표 16-1)·의무 caveat(53.55 vs 43.70 retrieval gap, ch16:446; Thm 아닌 실증)·§16.4 outer/inner·bridge-in(→ch15)/bridge-out(→ch17)·transfer 5기제 전부 보존. 감축은 유도 산문에 국한.

### 3.2 FIG/FIG-REF·기타 커버리지 exit — CLOSED
- FIG 주석 **17개** + FIG-REF **3개**(ch10·ch15의 F-09 three-regimes 재사용, ch17의 F-16 cms-spectrum 재사용) = SPEC.md 세트와 정확히 일치(§362 충족).
- STYLE-ISSUE 슬러그 자기신고(D10, ch01/03/04): **0건 잔존** ✅.
- 참고문헌 누락(D13): ch02 Goodfellow(ch02:59), ch03 Orabona(ch03:93) 추가 확인 ✅.

### 3.3 감축으로 인한 신규 위반 — 표본 검사 결과 **미검출**
ch01/ch10/ch16의 감축 구간에서 표·식 번호 단선, 정의 삭제, 참조 깨짐은 발견되지 않았다. 유일한 역방향 신호는 ch17 분량 증가(§2)이나 이는 CORRECTED 반영의 부산물로 내용 위반이 아니다.

---

## 4. 종합 판정

| 축 | 판정 | 근거 |
|---|---|---|
| TODO (V-c2/V-M5) | **CLOSED** | TODO-VERIFY 0건, CORRECTED 6건+높음 잔여 전건 본문 착지, ch13:291 부록수정 반영 |
| 분량 (V-c1) | **PARTIAL** | ch01 닫힘 / **ch10(+707)·ch16(+549) hard 미달** / soft 3장 미달(ch17 역증가) |
| 커버리지 (V-M3 등) | **CLOSED** | 감축 내용손실 0, FIG 17+3 완비, 슬러그·refs 처리 |

**원 리포트 critical 2건에 대한 폐쇄 판정**
1. `[분량] ch01 2.4×·ch10 1.8× 초과` → **부분 폐쇄**. ch01 닫힘, **ch10 미폐쇄**(여전히 hard cap +22%). ch16(1.5×)도 목표 재배정 후에도 상한 +8% 미폐쇄.
2. `[TODO-VERIFY] 위험도 높음 12건 미해소` → **폐쇄**. 전건 판정·본문 반영 완료.

## 5. 차기 라운드 이월(open) 목록
1. **ch10 분량 707단어 추가 감축**(hard, 최우선) — 차분표는 이미 적용됨; 남은 부피는 §10.2–10.5 산문·glossary 설명행에서 뺄 것.
2. **ch16 분량 549단어 추가 감축**(hard) — §16.3.5–16.3.6 유도 산문 결과식화.
3. **ch17 soft 목표 초과 + 역증가** — CORRECTED 반영분을 상쇄할 실험군 8종 중복 산문 감축, 또는 목표 재배정 결정.
4. **ch13·ch15 soft 미달** — 사유 기록 산출물 부재; 감축 또는 목표 재배정 명문화.
5. **프로세스**: `audit/w1-verify-answers.md` 생성(또는 `w1-todo-resolutions.md`를 SoT로 공식화) + VERIFIED 치환 규약 전건 일괄 적용 여부 결정.
6. **형상관리**: working tree 미커밋 8파일 커밋(P2 완료 커밋으로 확정).
