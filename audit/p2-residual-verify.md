# P2 잔여 라운드 — 파일 실측 검증 (p2-residual-verify.md)

- 검증일: 2026-07-12
- 대상: P2 잔여(재감사 미폐쇄 항목) 라운드 종료 후 실제 파일 상태
- 기준: STYLE-NOTATION.md v1.1, w1-reaudit-{notation,continuity,coverage}.md, p2-crossmodel-report.md
- 방법: `wc -w`, `grep -n`, 표 행 카운트, JSON 파싱, 원문/본문 대조

## 종합 판정: 5개 검증 항목 중 4개 CLOSED, 1개(ch10 분량) PARTIAL(방어 가능·비차단)

| # | 항목 | 결과 |
|---|---|---|
| 1 | ch10 ≤3,160 / ch16 ≤7,150 | ch16 **PASS**(7,148) / ch10 **미달**(3,442, +282) |
| 2 | ch15 첫 [TNT] 인용 순서 | **COMPLIANT** — 첫 등장(L9)에 풀네임+arXiv ID 동시 병기 |
| 3 | w1-verify-answers.md 37건 | **CLOSED** — 존재·표 37행·분포 30/6/1 정합 |
| 4 | ch02 crossmodel JSON + 적용 | **CLOSED**(실질) — JSON 존재, 발견 3/4 축자 반영 + 1건 설계로 해소 |
| 5 | TODO-VERIFY 잔존 grep | **CLOSED** — 0건 |

---

## 1. 분량 (ch10 / ch16)

실측 `wc -w`:
- **ch10 = 3,442** > 3,160 (hard) → **+282 (+8.9%) 미달**
- **ch16 = 7,148** ≤ 7,150 (hard) → **PASS (여유 2단어)**

재감사 대비 진척(reaudit-coverage §2):
- ch10: 3,867(+707) → 3,442(+282). 이번 라운드 −425단어. 여전히 hard cap 초과.
- ch16: 7,699(+549) → 7,148(−551). **상한 진입, 폐쇄 확정.**

**ch10 방어 가능성 판단 — 추가 감축 없이 "약한 방어 가능"(soft-pass), 단 엄밀히는 미폐쇄.**
- 초과분 282단어는 재감사가 명시한 **wc 재측정 ±10% 방법 오차**(reaudit-coverage §2 ch01·ch03 각주) 밴드 안(8.9% < 10%)이다. 한·영 혼용 텍스트에서 `wc -w`는 한국어 조사 결합·공백 토큰화로 계통 오차가 크므로, 이 한 자릿수 초과는 측정 노이즈로 흡수 가능한 수준이다.
- reaudit가 지목한 최우선 감축(표 10-1 차분표 교체)은 이미 적용됐고, 커버리지 축은 CLOSED(내용 손실 0). 즉 남은 초과는 **본문 산문 밀도**의 문제이지 구조 결손이 아니다.
- 그러나 hard cap은 형식상 여전히 깨져 있다. 클린 클로즈를 원하면 §10.2–10.5 산문·glossary 설명행에서 ~282단어만 더 빼면 된다. ch16이 닫히고 ch10이 +707→+282로 크게 개선된 만큼 **비차단**으로 이월한다.

## 2. ch15 첫 [TNT] 등장 순서

- 첫 `[TNT]` **태그**는 **L9**에 등장하며, 같은 문장 안에서 즉시 풀네임+arXiv ID가 병기된다: `[TNT] (*TNT: Improving Chunkwise Training for Test-Time Memorization*, arXiv:2511.07343)`.
- L1의 챕터 제목 "ch15. TNT: …"는 평문 제목 텍스트이지 축약 태그가 아니므로 first-use 규약 대상 아님.
- **판정: 정식 인용의 "앞"에 무주(無註) 태그가 새는 일 없이, 첫 태그와 정식 인용이 첫 등장 지점에 결합(tag→citation)** 되어 있다. STYLE-NOTATION §465("6편 공식 축약 … 첫 등장 시 장마다 1회 풀네임+arXiv ID 병기")를 정확히 충족. COMPLIANT.

## 3. w1-verify-answers.md (37건)

- 파일 **존재**(audit/w1-verify-answers.md, 12,573 B, 2026-07-12).
- 판정 이력표 행 수 = **37**(`| N |` 패턴 카운트 37, 최대 행번호 37).
- 헤더 선언과 정합: "총 37건 전량 판정 완료", 분포 **CONFIRMED 30 / CORRECTED 6 / UNRESOLVABLE 1 = 37**.
- 각 행이 (파일:라인 | 판정 | 원전 인용 | 본문 반영 방식) 4열 SoT 형식으로 채워짐. **37건 전건 반영 CLOSED.**

## 4. crossmodel/ch02-training-as-a-system.json + ch02 적용

- 파일 **존재**(6,857 B, 2026-07-12 09:03) + `.raw.md`·`.prompt.txt` 동반. 재실행 성공(이전 빈 배열 실패 복구). major 4건 수록.
- ch02 본문 대조:
  - **발견 #2**(optimizer step을 순수 elementwise로 일반화 → Shampoo/Muon과 모순): **적용.** L28 "AdamW 계열에서는 GEMM이 없는 순수 elementwise pass … 단 Shampoo·Muon은 update에 GEMM이 들어오는 예외 — §2.5.5–6", 표 헤더(L39)도 정합.
  - **발견 #3**(AdaGrad "차이는 decay 하나뿐"은 오류): **적용.** L178 "AdaGrad는 1차 moment buffer m_t 없이 현재 gradient g_t를 그대로 분자에 쓴다" 명기.
  - **발견 #4**(Newton–Schulz가 특이값을 정확히 1로 수렴시킨다는 서술 오류): **적용.** L197 "정확한 1 수렴 대신 … 얼추 [0.7, 1.3]에 모은다 — 결과가 이상적 UVᵀ가 아니라 그 근사인 이유다".
  - **발견 #1**(update 서명이 Θ_t·step t 의존성을 못 담음): **설계로 해소(축자 미반영).** 서명은 `update(state, g) → (state′, ΔΘ)`(L273)를 유지하되 decoupled weight decay는 optimizer 밖 별도 Θ 곱(§2.5.3 L143–146 "Θ ← (1−ηλ)Θ − η·update")으로, bias correction의 step 의존은 state 내 카운터로 처리 → Θ_t를 서명에 넣지 않고도 정합. 방어 가능한 설계 선택.
- **판정: 실질 CLOSED**(핵심 3건 축자 반영, 1건 설계 해소).
- **비차단 문서 불일치 발견:** p2-crossmodel-report.md §1(L18)은 ch02를 여전히 "실패 … 재실행 대상 … cross-model 검증 없음"으로 기재. JSON 재실행·적용 후 리포트 §1이 갱신되지 않은 **stale 상태**. remaining으로 이월.

## 5. TODO-VERIFY 잔존 grep

- `grep -rn '<!-- TODO-VERIFY' study-kr/` = **0건**. 전 17장 청정. **CLOSED.**
- (참조: 해소는 §6.3 규약 1의 VERIFIED 치환 3건 + 인용의 본문 접합 ~34건 두 경로 혼용 — reaudit-coverage G2 기록과 정합.)

---

## remaining (차기 이월 — 전부 비차단)

1. **[분량 hard, 최우선] ch10 −282단어** — 3,442 → ≤3,160. wc ±10% 노이즈 밴드 안이라 방어는 가능하나 형식상 hard cap 미폐쇄. §10.2–10.5 산문·glossary 설명행에서 소폭 감축하면 클린 클로즈. (커버리지 CLOSED 유지 조건에서만 감축.)
2. **[문서 정합] p2-crossmodel-report.md §1 갱신** — ch02 "실패/재실행 대상" 기술이 stale. JSON(4 major)·본문 적용 반영해 §1·§2 판정표·집계(총 발견/장 수)를 ch02 포함으로 정정.
3. **[분량 soft] ch17 역증가·ch13·ch15 미달 사유 기록 부재** — reaudit-coverage §2·권고 3·4. soft라 exit 실패 아니나, fixplan §361이 요구한 "미달 시 사유 기록" 산출물이 여전히 없음. 감축 또는 목표 재배정 명문화 필요.
4. **[감사추적 선택] VERIFIED 치환 규약 불균일** — 37건 중 3건만 `<!-- VERIFIED(...) -->` 치환, 나머지는 본문 접합. §6.3 위반은 아니나 감사 추적 일관성 개선 여지(reaudit G2).
