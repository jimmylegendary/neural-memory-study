# W1 → W2 Fix Plan (P2 보정 라운드 실행 계획)

**작성**: fix coordinator, 2026-07-12
**입력**: `audit/w1-notation-report.md`(이하 **[N-*]**), `audit/w1-continuity-report.md`(**[K-F*]**), `audit/w1-coverage-report.md`(**[V-*]**)
**기준 문서**: `style/STYLE-NOTATION.md` **v1.1** (본 라운드에서 코디네이터가 개정 완료 — §8 이력 참조). 장 fixer는 v1.1을 다시 수정하지 않는다.
**그림 스펙**: `figures/SPEC.md` (코디네이터 작성 완료). 장 fixer는 본문에 `<!-- FIG: ... -->` / `<!-- FIG-REF: ... -->` 주석만 삽입한다. 그림 생성은 후속 figure pass.

**항목 ID 표기**: [N-C1]=notation critical 1, [N-M1]=major, [N-m1]=minor / [K-F01]…[K-F08]=continuity / [V-c1],[V-c2]=coverage critical, [V-M3]…[V-M5]=major, [V-m6]…[V-m8]=minor. 3종 리포트의 critical·major·minor 전 항목이 아래 §0(중앙 결정)·§1(선행 작업)·장별 섹션에 배정되었고, §2의 매핑표로 전수 확인 가능하다.

---

## §0. 중앙 결정 (cross-chapter — 장 fixer는 이 결정을 재심하지 않는다)

**D1. ch05 §5.4 ↔ ch14 약속 모순 [K-F02, critical] — 양측 수정으로 확정.**
(i) **ch14가 주 수정**: §14.3.1의 Prop 2 대목과 φ* 대목에 5장 §5.4 dense-Hopfield 사슬(Krotov→Ramsauer, energy 차수 = feature lift = capacity) 콜백 1–2문장을 삽입해 "corollary처럼 읽힌다"는 ch05의 약속을 이행시킨다. (ii) **ch05는 모순 문장만 수정**: "정밀한 비용 계산은 14장에서 한다"를 "비용의 **구조**는 14장 §14.7에서 확정하되, 구현 차수 $p$·sketch 차원 미공개로 절대값 계산은 그곳에서도 불가하다는 것까지가 결론"으로 완화. ch05의 65× 예시 계산은 ch05에 그대로 둔다(ch14로 이관하지 않음 — ch14는 이미 자체 crossover 계산 보유). 각 fixer는 자기 파일만 만진다.

**D2. ch16 → ch11 참조 단선 [K-F01, critical] — 참조 삽입 7곳 + 볼드 재정의 해제.**
리포트 지시 그대로 채택(아래 ch16 섹션에 위치 명시). ch11 쪽은 무수정 — 약속은 이미 ch11에 있고 수신 측만 고친다.

**D3. ch13의 elastic net 허위 역참조 [K-F03, major] — ch13 재서술로 확정 (ch03 추가 안 함).**
근거: ch03은 이미 1.4× 분량 초과로 감축 대상[V-M4]이고, 개념적 재료(ℓ1/ℓ2 벌점, soft/hard eviction)는 ch03 §3.6–3.7에 이미 있다 — 없는 것은 "elastic net"이라는 명칭·결합형뿐. 따라서 ch13:175에서 elastic net을 "3장의 도구함" 열거에서 빼고 장-국소로 볼드 도입한다. ch03 fixer는 이 건으로 아무것도 하지 않는다.

**D4. momentum-as-memory 소유권 [K-F05, minor] — STYLE §2.4 유지(ch02=객체, ch16=정리), ch12는 '도입(기원)'.**
ch12 표 12-3의 열 제목을 "개념 (이 논문이 도입)"으로 완화 + momentum-as-memory 행에 "(정의 소유: 2장·16장)" 병기. ch02·ch16은 이 건 무수정.

**D5. LTI 약어 충돌 [K-F07, minor] — 두 약어 공존 확정, 전역 기본 의미 = linear time-invariant.**
STYLE v1.1 §2.4에 판정 명문화 완료. 실행: ch17:78 첫 등장에 "7장의 LTI(linear time-invariant)와 무관한 약어다" 구분 문장 삽입(의무), ch12는 "LTI fast path" 첫 사용(:247)에 "(LTI = linear time-invariant → 7장)" 1회 병기(개명하지 않음 — ch07 정의와 동일 의미의 정당한 사용이므로).

**D6. retrieval gap 수치 [K-F06, minor] — 53.55 / 43.70 (원문 정밀도, ch14 표기)으로 전 책 통일.**
수정 대상: ch16:443, ch17:314. ch14는 이미 정확(기준). STYLE §6.4는 v1.1에서 정정 완료.

**D7. ch16 분량 [V-c1 일부] — 목표 11→13pp 재배정 + 약 16.4→13pp 감축 (둘 다).**
근거: NL은 6편 중 노트 밀도가 가장 높은 논문(Def 1–4, CMS 3변형, self-modifying 3단계, Hope, transfer 5기제, 의무 caveat 2건)으로 11pp에 수용 불가 — 상향 재배정이 정당. 그러나 실측 16.4pp에는 (M) 표준형에서 재유도 가능한 유도 스텝 산문(특히 §16.3.5–16.3.6)과 표 재구성의 중복 열이 있어 과잉 서술은 실재한다 — 목표를 실측에 맞추지 않고 13pp로 자른다. STYLE §4.3 v1.1 반영 완료.

**D8. ch01·ch10 분량 [V-c1] — ch01 목표 4pp(재배정 완료), ch10 표 10-1 차분화.**
ch01: 실측 ~7.5pp → 4pp(2,000–2,200단어). ch10: 표 10-1을 ch01 표 1-1의 전문 재수록에서 **차분표**(행 유지, 열은 [inference 어휘 | 이 라인의 어휘 | 확립한 장]만; "대응의 성격" 산문 열 삭제 + "상세는 표 1-1(→ 1장)" 안내문)로 교체. 세부는 각 장 섹션.

**D9. TODO-VERIFY [V-c2, V-M5] — 위험도 '높음' 전건 이번 라운드 해소 의무, 나머지는 배치 겸사 해소.**
해소 규약은 STYLE v1.1 §6.3(주석 삭제 + 원문 위치 병기/각주 대체, 이력은 `audit/w1-verify-answers.md`). 검증 배치(§1)는 **장 fixer 착수 전에** 코디네이터 측이 수행해 판정을 `audit/w1-verify-answers.md`에 적재하고, 각 장 fixer는 그 판정을 자기 장에만 반영한다. 의무 해소(높음): ch01:142, ch06:138·167, ch08:42·47·65·118, ch13:332, ch16:297·394, ch17:202 (+같은 배치로 자연 해소되는 ch08:93). 리포트 본문 집계는 12건, 표의 '높음' 표기는 11건 — 위 명단(11건)+ch08:93을 의무분으로 확정.

**D10. 파일 슬러그 [N-M1, V-m7] — 실파일명 승인 (§5.1 v1.1 개정 완료), 파일 rename 없음.**
실행: ch01:3, ch03:3, ch04:3의 STYLE-ISSUE 자기신고 주석 3건 제거. ch09는 추가 조치 불요(주석이 원래 없었고 슬러그가 이제 정본).

**D11. 기준 문서 일괄 결정 (v1.1 반영 완료 — 장 fixer 개별 판단 금지).**
"weights" 단독 복수형 허용[N-m9: 장 수정 없음], GDN 약어 규칙[N-m5], ⊗ Kronecker 예외 확장[N-m1: ch14:31 소급 적합 → 무수정], $L_{\mathrm{layer}}$ 신설[N-M2], 표 캡션 플레인 통일[N-m6: ch04·ch05·ch16 수정], TODO-VERIFY 해소 규약, LTI 판정, retrieval gap 수치.

**D12. 그림 [V-M3] — `figures/SPEC.md`가 SoT. W2에서는 FIG 주석만, 생성은 figure pass.**
재사용 그림(F-09 three-regimes, F-16 cms-spectrum)의 재사용 장에는 `<!-- FIG-REF: ... -->` 주석만 삽입(살아 있는 "그림 9-2" 텍스트 참조는 그림 생성 후 figure pass가 삽입 — 존재하지 않는 그림 번호를 본문에 만들지 않는다).

**D13. 참고문헌 누락 2건 [V-m6] — ch02에 Goodfellow, ch03에 Orabona 추가** (§2.5 형식의 본문 인용; 서지 목록은 P4 후공정).

**D14. 미교육 명칭 4건 [K-F04] — 배정: MoE/router = ch10 glossary 1행 + ch17 참조, GRPO = ch17 국소 글로스(ch11은 상한 초과라 추가 금지), Reptile/CAVIA = ch04 §4.3 1문장, KKT = ch13 괄호 글로스.**

---

## §1. 공통 선행 작업 (장 fixer 착수 전 완료)

**V-0. 검증 배치 실행 → `audit/w1-verify-answers.md` 생성** [V-c2, V-M5]. 각 TODO의 검색어는 해당 주석에 이미 있다.

| 배치 | 방법 | 해소 대상 (장:행) |
|---|---|---|
| B1 로컬 grep | `papers/*.txt` 검색 | ch01:142(2501.00663 "RWKV"), ch09:176, ch10:75, ch12:139, ch14:301(1차), ch15:258, ch17:9, ch17:202, ch11:147, ch11:148, ch16:394(§9.6 텍스트 확인분) |
| B2 외부 arXiv 대조 | 논문당 1회 fetch | **2407.04620 최우선(6건: ch08:42·47·65·93·118, ch09:120)**; 2503.14456(ch06:167·173), 2407.14207(ch06:138), 2406.06484+2102.11174(ch06:117, ch09:108), 2312.06635(ch06:44, ch09:73), 2312.00752(ch07:56), 2405.21060(ch07:114), 2212.07677(ch04:129), 2008.02217(ch05:90), 1606.01164(ch05:84), Amit 1985(ch05:74), 2504.05298(ch08:123), ch01:19(2407.04620 §1 wording) |
| B3 PDF 확인 | `papers/*.pdf` 열람 | ch12:141·297(Titans p.9·Table 5 캡션), ch13:332(Miras App. C), ch16:297(NL §8.1–8.2), ch14:301(Atlas Fig. 3 라벨 최종) |
| B4 웹 1회 | NVIDIA H100 datasheet | ch10:140 |

기록 형식(장 fixer가 소비): `장:행 | 판정(확인/수정 필요/판정 불가) | 근거 원문 위치 | 본문에 병기할 인용 문자열`. **연동 항목**: ch01:142의 판정은 ch12 §12.3.9에도 반영된다 — 두 장 fixer 모두 같은 판정을 읽는다.

**V-1. STYLE v1.1·figures/SPEC.md 적용 확인** — 코디네이터 완료분. 장 fixer는 pre-flight로 v1.1 §8(변경 이력)을 읽고 시작한다.

**공통 금지사항 (전 장 fixer)** — 각 장 섹션의 금지 목록에 추가로 항상 적용:
1. **자기 장 파일 1개만 수정한다.** 다른 장 파일, `style/`, `notes/`, `papers/`, `dossier/`, `figures/SPEC.md` 수정 금지. 기준 문서에 대한 추가 요구는 `<!-- STYLE-ISSUE: ... -->` 주석으로만.
2. 자기 장이 소유하지 않은 개념의 재정의 금지(STYLE §2.4) — 참조("(→ N장)")만.
3. (Part II) 표기 대응표의 행 삭제·의미 변경 금지. 행 추가만 허용.
4. §6.4 의무 caveat, ledger delta, bridge-in/out 문단의 삭제·완곡화 금지. 분량 감축은 이들을 건드리지 않는 범위에서.
5. `w1-verify-answers.md`에 기록된 판정 밖의 새 수치·새 사실 주장 추가 금지(추가하려면 원문 위치 병기 또는 신규 TODO-VERIFY).
6. FIG/FIG-REF 주석에 캡션·그림 번호 텍스트를 넣지 않는다(figure pass 담당).
7. 수정 후 STYLE v1.1 §7 자가 감사 체크리스트를 다시 통과시킨다.

---

## §2. 리포트 항목 → fixplan 매핑 (전수 확인용)

| 리포트 ID | 처리 위치 |
|---|---|
| N-C1 | ch02, ch12, ch13 |
| N-C2 | ch05, ch11, ch13, ch14, ch16 (+ch15 순서=N-m10) |
| N-C3 | ch13, ch14, ch15 |
| N-M1 | D10 → ch01, ch03, ch04 (주석 제거), STYLE v1.1 |
| N-M2 | STYLE v1.1 + ch02, ch16 |
| N-m1 | D11(v1.1 예외 확장) → ch14 무수정 기록 |
| N-m2 | ch16 |
| N-m3 | ch03, ch12, ch13, ch14, ch15, ch16 |
| N-m4 | ch01 |
| N-m5 | STYLE v1.1 + ch06, ch13 |
| N-m6 | STYLE v1.1 + ch04, ch05, ch16 |
| N-m7 | ch15 |
| N-m8 | ch02, ch05 |
| N-m9 | STYLE v1.1 (장 수정 없음) |
| N-m10 | ch15 |
| K-F01 | D2 → ch16 |
| K-F02 | D1 → ch05, ch14 |
| K-F03 | D3 → ch13 |
| K-F04 | D14 → ch04, ch10, ch13, ch17 |
| K-F05 | D4 → ch12 |
| K-F06 | D6 → ch16, ch17 (+STYLE v1.1) |
| K-F07 | D5 → ch12, ch17 (+STYLE v1.1) |
| K-F08 | ch09 |
| V-c1 | D7·D8 → ch01, ch10, ch16 |
| V-c2 | D9 → §1 배치 + ch01/06/08/13/16/17 |
| V-M3 | D12 → figures/SPEC.md + 전 장 FIG 주석 |
| V-M4 | ch03, ch06, ch13, ch15, ch17 |
| V-M5 | D9 → §1 배치 + STYLE v1.1 §6.3 |
| V-m6 | D13 → ch02, ch03 |
| V-m7 | D10 (=N-M1) |
| V-m8 | D8 → ch10 |

---

# 장별 수정 지시

각 항목: **[리포트 ID / 심각도]** 위치 — 지시. 행 번호는 W1 초안 기준(감사 시점) — fixer는 주변 문맥으로 재확인 후 적용.

---

## ch01 — `study-kr/part1/ch01-orientation-rosetta.md`

1. **[V-c1 / critical]** 분량 ~7.5pp(3,936단어) → **4pp(2,000–2,200단어)**. 지시:
   - §1.6 worked example: 예제 1개(d=2 수준)만 남기고 절반 이하로 축약. 수치 전개는 "2장의 worked example에서 완성한다" 1문장으로 위임.
   - §1.7 systems bridge: 절 자체는 유지(Part I 템플릿 의무)하되 back-of-envelope 계산을 1개로 줄이고, 나머지 계산은 "→ 10장 budget 표" 참조 1문장으로 대체.
   - 도입 산문·표 1-1/1-2 사이의 중복 해설 압축. **표 1-1, 표 1-2, (M) 선행 제시, "다음 장으로"는 보존.**
2. **[N-m4 / minor]** :228 — "$(1-\lambda)$" → "$(1-\eta\lambda)$" (ch02의 decoupled decay 표기와 일치).
3. **[N-M1 / major, D10]** :3 — 슬러그 STYLE-ISSUE 주석 제거(v1.1이 실파일명 확정).
4. **[V-c2 / critical]** :142 TODO-VERIFY(Titans App. C에 RWKV-7 포함 여부) — `w1-verify-answers.md`(배치 B1)의 판정을 반영: 확인이면 [Titans App. C] 병기 후 주석 삭제, 아니면 본문 단정 수정(§6.3 v1.1 규약). :19(Sun 2024 wording, 배치 B2)도 판정 반영.
5. **[V-M3 / major, D12]** 표 1-1 직전에 `<!-- FIG: ch01/fig-01-two-loops -->` 삽입.

**금지사항**: 공통 금지 1–7. 추가로 — 표 1-1의 행 구성·열 구조 변경 금지(ch10이 차분표로 참조 예정); Rosetta 사전 확장은 ch01 소유지만 이번 라운드는 감축 전용, 신규 행 추가 금지; ch02·ch10 파일 수정 금지(이관이 아니라 삭제+참조).

---

## ch02 — `study-kr/part1/ch02-training-as-a-system.md`

1. **[N-C1 / critical]** "forget gate" 4회 정리 — 장 내 병기는 1회만:
   - :138 절 제목 "§2.5.3 weight decay: $(1-\eta\lambda)$라는 forget gate" → "…$(1-\eta\lambda)$라는 retention의 원형" 류로 개제(절 제목에서 forget gate 제거).
   - :6(목표 박스)·:151·:273(요약) — "retention gate의 원형(역사적 별칭 forget gate → 13장)" 패턴으로 첫 1회만 병기, 나머지는 retention gate/retention으로. ch13의 정의 소유를 침범하지 않도록 정의 서술 없이 참조만.
2. **[N-M2 / major]** :63 — "$L$층 MLP" → "$L_{\mathrm{layer}}$층 MLP"($\ell=1..L_{\mathrm{layer}}$), v1.1 §1.2.
3. **[N-m8 / minor]** :136 — "continual learning에서의 망각" → "…에서의 forgetting(→ 11장)".
4. **[V-m6 / minor, D13]** backprop 도입 절에 Goodfellow, *Deep Learning* ch. 6–8 인용 추가(§2.5 외부 문헌 형식; arXiv 없음 — 저자-연도+서명).
5. **[V-M3 / major, D12]** loss surface 문단(§2.3 부근)에 `<!-- FIG: ch02/fig-01-loss-landscape -->` 삽입.

**금지사항**: 공통 금지 1–7. 추가로 — retention gate의 정의·방향 서술 추가 금지(소유 ch13); momentum-as-memory의 "정리로서의 지위" 서술 금지(소유 ch16, D4 — 현행 문장 유지); 분량 적정(9.8/10pp)이므로 순증 위주 수정 금지(인용 1건·기호 치환 외 본문 확장 금지).

---

## ch03 — `study-kr/part1/ch03-online-learning-regret.md`

1. **[V-M4 / major]** 분량 ~8.5pp → **상한 3,795단어(6pp+15%)**: §3.8 표 2개 중 1개 축약 + 중복 산문 정리. FTL 실패 손계산·FTRL→OGD worked example·mirror descent/Bregman(ch13이 수입하는 도구함)은 보존.
2. **[V-m6 / minor, D13]** FTRL 절에 Orabona 2019 (arXiv:1912.13213) 인용 추가.
3. **[N-M1 / major, D10]** :3 — 슬러그 STYLE-ISSUE 주석 제거.
4. **[N-m3 / minor]** 태그 (3-6) — 요약 bullet에서 실제 참조를 추가하거나 태그 제거(기본: 참조 추가).
5. **[K-F03 / 기록, D3]** elastic net은 **ch13 국소 도입으로 확정 — 이 장은 이 건으로 아무것도 하지 않는다** (ℓ1/ℓ2 eviction 구도 서술 현행 유지).
6. **[V-M3 / major, D12]** §3.7 loss geometry 표 부근에 `<!-- FIG: ch03/fig-01-loss-geometries -->` 삽입.

**금지사항**: 공통 금지 1–7. 추가로 — elastic net 명칭 도입 금지(D3, ch13 소유로 확정); 감축 시 ch13 §13.3.6이 역참조하는 mirror descent·Bregman 대목 삭제 금지.

---

## ch04 — `study-kr/part1/ch04-meta-learning-bilevel.md`

1. **[K-F04-3 / minor, D14]** §4.3 learning-to-learn 계보에 1문장 추가: "first-order 변형(Reptile), context-parameter 변형(CAVIA)" — ch12 §12.4가 이 명칭을 [Titans §3.1] 귀속으로 사용 중.
2. **[N-M1 / major, D10]** :3 — 슬러그 STYLE-ISSUE 주석 제거.
3. **[N-m6 / minor, D11]** 표 캡션 볼드 → 플레인("표 4-N — …"), v1.1 §5.4.
4. **[V-c2 관련 / 중간]** :129 TODO-VERIFY(von Oswald 주장 범위) — 배치 B2(2212.07677) 판정 반영, §6.3 v1.1 규약.

**금지사항**: 공통 금지 1–7. 추가로 — 분량 적정(6.4/6pp): Reptile/CAVIA 1문장 외 본문 확장 금지; MAML→$W_{\mathrm{init}}$ 서술의 소유 경계(개념 ch04, 역할 ch15) 변경 금지.

---

## ch05 — `study-kr/part1/ch05-associative-memory.md`

1. **[K-F02 / critical, D1(ii)]** §5.4 말미 — 모순 문장 수정: "이 거래의 정밀한 비용 계산은 14장에서 한다." → "이 거래의 비용 **구조**(capacity의 지불 통화가 state byte·matmul 폭이라는 것)는 14장 §14.7에서 확정한다 — 단, Atlas가 구현 차수 $p$와 sketch 차원을 공개하지 않아 절대값 계산은 그곳에서도 불가능하다는 것까지가 14장의 결론이다." 65× 예시 계산과 "corollary처럼 읽힌다" 문장은 유지(ch14가 콜백으로 이행).
2. **[N-C2 / critical]** [Titans]·[Miras]·[Atlas]·[NL] 각 장 내 첫 등장에 풀네임+arXiv ID 병기(§2.5 형식). Atlas는 21회 인용 중 첫 등장 1곳에만.
3. **[N-m8 / minor]** :224 — 표 행 라벨 "망각" → "forgetting".
4. **[N-m6 / minor, D11]** 표 캡션 볼드 → 플레인.
5. **[V-c2 관련 / 중간]** :74(Amit 0.138d)·:84(Krotov $d^{n-1}$)·:90(Ramsauer) TODO-VERIFY — 배치 B2 판정 반영.
6. **[V-M3 / major, D12]** §5.4 사슬 요약 문장("사슬을 한 줄로 요약한다") 직전에 `<!-- FIG: ch05/fig-01-capacity-chain -->` 삽입.

**금지사항**: 공통 금지 1–7. 추가로 — **ch14 파일 수정 금지**(콜백은 ch14 fixer 담당, D1); capacity formal 정의 서술 금지(소유 ch14 — 현행 "(…14장 소유다)" 경계 문장 유지); 분량 적정(+8%)이므로 순증 최소화.

---

## ch06 — `study-kr/part1/ch06-linear-attention-fwp.md`

1. **[V-M4 / major]** 분량 ~8.1pp → **10pp(5,000–5,500단어)로 보강** (유일한 미달 장, B5 = "audience 지렛대 최대"): RWKV-7 절·expressivity 사이드바를 TODO 해소 결과(아래 2)와 함께 확장. 신규 서술은 배치 판정·원문 위치 병기 필수.
2. **[V-c2 / critical]** TODO-VERIFY 5건 — :138(Longhorn 채널별 η — **높음**, 식 (6-5) 전제)·:167(RWKV-7 식 (6-6) — **높음**)·:44·:117·:173 — 배치 B2 판정 반영(§6.3 v1.1 규약). :138·:167 판정이 '수정 필요'면 해당 식·유도를 원문에 맞게 고친다.
3. **[N-m5 / minor, D11]** :253 표의 "GDN" — 장 첫 등장에 "Gated DeltaNet(이하 GDN)" 병기 후 장 내 통일(v1.1 §2.5).
4. **[V-M3 / major, D12]** §6.2 timeline 표 자리에 `<!-- FIG: ch06/fig-01-fwp-timeline -->` 삽입(표는 그림 생성 후 figure pass가 축약 여부 결정 — 지금은 유지).

**금지사항**: 공통 금지 1–7. 추가로 — §1.6 카탈로그의 기준형 변경 금지(카탈로그는 STYLE 소유); "retention gate(→ 13장; 역사적 별칭 …)" :87 문구는 모범 사례로 판정됨[N-C1] — 변경 금지; 보강 시 ch07(SSD)·ch08(TTT) 소유 영역 침범 금지.

---

## ch07 — `study-kr/part1/ch07-ssm-lineage.md`

1. **[V-c2 관련 / 낮음]** :56(Mamba Theorem 1 위치)·:114(Mamba-2 state 차원 N) TODO-VERIFY — 배치 B2 판정 반영.
2. **[V-M3 / major, D12]** §7.5 3경로 손계산 직전에 `<!-- FIG: ch07/fig-01-ssd-three-paths -->` 삽입. Mamba selectivity 절에 `<!-- FIG: ch07/fig-02-selectivity-gate -->` 삽입(P2 — figure pass에서 생성 보류 가능).
3. **[분량]** +19%는 허용 오차 밖(±15%)이나 major 지정 없음 — 중복 산문이 보이면 소폭 압축(선택), 신규 추가 금지.

**금지사항**: 공통 금지 1–7. 추가로 — LTI(linear time-invariant) 정의는 이 장 소유(:38) — D5에 따라 현행 유지, "Learning to Imitate" 언급 추가 금지; exact vs semantic 경계 서술은 ch09와 공유된 논점 — ch09 소유 명제("chunk 크기=semantic hyperparameter") 재정의 금지.

---

## ch08 — `study-kr/part1/ch08-ttt-lineage.md`

1. **[V-c2 / critical]** TODO-VERIFY 6건 — **높음 4건**: :42(learned inner lr 함수형)·:47(TTT-MLP 구조)·:65(두 동치 정리 번호·전제)·:118(비교 스케일·Mamba 16k 그림 번호) + :93(inner batch 16)·:123(Dalal 셋업) — 배치 B2(2407.04620 1편이 :42·47·65·93·118 전부, 2504.05298이 :123) 판정 반영. '정리로 제시'(:65) 단정이 틀렸으면 §6.2 귀속으로 전환.
2. **[V-M3 / major, D12]** §8.3 TTT layer 3요소 정리 직후에 `<!-- FIG: ch08/fig-01-ttt-layer -->` 삽입.

**금지사항**: 공통 금지 1–7. 추가로 — dual form의 일반화 서술 금지(일반화는 ch09 소유); 6편 departure 표(=Part II 목차)의 구조 변경 금지(Part II 6개 장 bridge-in이 이 표를 전제); 분량 적정(7.0/7pp) — 판정 반영 외 확장 금지.

---

## ch09 — `study-kr/part1/ch09-chunkwise-parallel-training.md`

1. **[K-F08 / minor]** :126 — 중의성 해소: "…[TNT Fig. 2]의 chunk-size mismatch이고(→ 15장), **'chunk 크기 = semantic hyperparameter' 명제**의 명명과 소유는 이 장에 있다(→ §2.4; mismatch 현상 자체의 소유는 15장)." 형태로 재작성 — "이 명제"가 가리키는 대상을 명시.
2. **[N-M1 / 기록, D10]** 슬러그는 v1.1로 정본화 — 이 장은 조치 불요(STYLE-ISSUE 주석 추가하지 말 것).
3. **[V-c2 관련 / 낮음–중간]** :73·:108·:120·:176 TODO-VERIFY — 배치 B1/B2 판정 반영.
4. **[V-M3 / major, D12]** (M4) 재게 직후에 `<!-- FIG: ch09/fig-01-chunkwise-dataflow -->`, 표 9-1 부근에 `<!-- FIG: ch09/fig-02-three-regimes -->` 삽입. F-09는 ch10·ch15가 재사용하는 "draw once" 그림 — 위치 선정에 특히 신중할 것(regime 정리 절의 도입).

**금지사항**: 공통 금지 1–7. 추가로 — chunk-size mismatch의 정의·명명 서술 금지(소유 ch15); 분량 소폭 미달(−12%)은 허용 — 보강 목적의 신규 서술 금지.

---

## ch10 — `study-kr/part1/ch10-systems-bridge.md`

1. **[V-c1 / critical, D8]** 분량 ~9pp(4,734단어) → **상한 3,160단어(5pp+15%)**:
   - **[V-m8]** 표 10-1을 차분표로 교체: ch01 표 1-1의 행 순서를 유지하되 열은 [inference 어휘 | 이 라인의 어휘 | 확립한 장]만. "대응의 성격" 산문 열 삭제. 표 캡션 아래(또는 위 안내문)에 "대응의 성격·상세 해설은 표 1-1(→ 1장)" 1문장.
   - glossary 55행 유지하되 행당 설명 1줄 상한으로 압축.
   - 4K/64K/1M budget 표·cheat sheet·prefill/decode 비대칭·AI(C) 제2식은 보존.
2. **[K-F04-1 / minor, D14]** glossary에 MoE/router 1행 추가(ch17 :54가 참조 예정).
3. **[V-c2 관련 / 중간·낮음]** :75(Atlas window c 실측)·:140(H100 수치) TODO-VERIFY — 배치 B1/B4 판정 반영.
4. **[V-M3 / major, D12]** budget 표 직후에 `<!-- FIG: ch10/fig-01-state-budget -->` (P2 — 생성은 여유 시). 비용 모델 절의 three-regime 언급 지점에 `<!-- FIG-REF: ch09/fig-02-three-regimes -->` 삽입.

**금지사항**: 공통 금지 1–7. 추가로 — ch01 표 1-1 원본 수정 금지; roofline/MFU 등 "독자 모국어" 개념에 정의 추가 금지(§2.4: 표기 통일만); glossary 행 삭제 금지(압축만).

---

## ch11 — `study-kr/part1/ch11-continual-learning.md`

1. **[N-C2 / critical]** [Titans] 첫 등장 1곳에 풀네임+arXiv ID 병기.
2. **[V-c2 관련 / 낮음]** :147(ReST^EM arXiv ID)·:148(SEAL arXiv ID) TODO-VERIFY — 배치 B1/B2 판정 반영.
3. **[K-F01 / 기록, D2]** ch16→ch11 단선의 수선은 **전부 ch16 쪽에서** 수행 — 이 장은 무수정. ch11의 "왜 필요한가" 박스·"16장에서 상술" 약속 문구 변경 금지.
4. **[K-F04-2 / 기록, D14]** GRPO 글로스는 ch17 국소로 확정 — 이 장에 추가하지 않는다(상한 초과 +11% 상태).

**금지사항**: 공통 금지 1–7. 추가로 — 분량 상한 초과 상태이므로 신규 내용 추가 전면 금지(ID 병기·TODO 판정 반영 제외); online/offline consolidation 정의(ch16·ch17이 참조하게 될 원본) 변경 금지.

---

## ch12 — `study-kr/part2/ch12-titans.md`

1. **[N-C1 / critical]** "forget gate" 5회(:9×2, :164×2, :286) — 첫 1회만 "retention gate(원문 표현 forget gate)" 병기, 나머지는 retention gate로. Titans 원문 인용 문맥이면 "forgetting mechanism"(§2.4 허용 표현)으로. 대응표 :179 행은 유지.
2. **[K-F05 / minor, D4]** 표 12-3(:263–268) — 열 제목 "개념 (이 장이 정의)" → "개념 (이 논문이 도입)"; momentum-as-memory 행에 "(정의 소유: 2장·16장)" 병기.
3. **[K-F07 / minor, D5]** :247 "LTI fast path" 첫 사용에 "(LTI = linear time-invariant → 7장)" 1회 병기. :276은 무수정.
4. **[N-m3 / minor]** 태그 (12-5)·(12-6)·(12-8) — 요약 bullet 참조 추가 또는 태그 제거(기본: 참조 추가).
5. **[V-c2 관련 / 중간·낮음]** :139(MAC $N_l=C$)·:141(Eq. 25 query 재적용)·:297(Table 5=400M) TODO-VERIFY — 배치 B1/B3 판정 반영. **연동**: ch01:142 판정(App. C의 RWKV-7 포함 여부)을 §12.3.9 서술에 반영.
6. **[V-M3 / major, D12]** §12.5(MAC/MAG/MAL) 도입부에 `<!-- FIG: ch12/fig-01-mac-dataflow -->` 삽입.

**금지사항**: 공통 금지 1–7. 추가로 — 표 12-1(Titans 대응표) 행 삭제·의미 변경 금지; retention gate 정의 서술 금지(소유 ch13 — "weight decay = per-token retention"의 Titans 문맥 서술[ch12 소유]과 구분 유지); 분량 적정(10.3/10pp) — 순증 최소화.

---

## ch13 — `study-kr/part2/ch13-miras.md`

1. **[N-C1 / critical]** "forget gate" — :13의 2회는 Miras 역사 서술이므로 "(당시 명칭) forget gate" 류 인용 처리, :77 "고전적 forget gate가 사는 곳" → "고전적 gate가 사는 곳". :19의 "병기를 마친다" 선언은 유지 — 결과적으로 장 내 병기 1회(:19 계열)만 남긴다.
2. **[N-C2 / critical]** [Titans](:5)·[Atlas](:366) 첫 등장에 풀네임+arXiv ID 병기.
3. **[N-C3 / critical]** :287 "inner loop이"×2·"outer loop은" / :289 "inner loop을" → 가/는/를(§2.2). 장 내 다른 위치의 올바른 용례와 통일.
4. **[K-F03 / major, D3]** :175 — 문장 재작성: "3장의 online optimization 도구함 — mirror descent, Bregman divergence — 이 통째로 sequence layer 설계로 수입되고, 여기에 **elastic net**(ℓ1+ℓ2 결합 벌점; 3장의 ℓ1/ℓ2 eviction 구도의 결합형으로, 이 장에서 도입한다)이 더해진다." — ch03 참조 목록에서 elastic net 제거 + 장-국소 볼드 도입.
5. **[K-F04-4 / minor, D14]** §13.3.6 (13-8) 유도의 KKT 첫 등장에 "(제약 최적화의 1차 최적성 조건)" 괄호 글로스.
6. **[N-m5 / minor, D11]** :120 GDN 첫 등장에 "Gated DeltaNet(이하 GDN)" 병기, 장 내(:334·:348·:350 포함) GDN으로 통일.
7. **[N-m3 / minor]** 태그 (13-2)·(13-4)·(13-5)·(13-6)·(13-9) — 참조 추가 또는 태그 제거.
8. **[V-c2 / critical]** :332 TODO-VERIFY(outer optimizer AdamW 여부 — **높음**, 표 13-3이 "표준 pre-training/peak 1.5e-3" 기재) — 배치 B3(Miras PDF App. C) 판정 반영; 틀렸으면 표 13-3 수정.
9. **[V-M4 / major]** 분량 ~10.5pp — soft 감축(목표 8pp, 상한 5,060단어 지향): 중복 산문 한정, 노트 반영 밀도(retention 동물원·surrogate 논의)는 보존. 상한 미달성 시 사유를 커밋 메시지/작업 로그에 기록.
10. **[V-M3 / major, D12]** 4축 도입 절 끝에 `<!-- FIG: ch13/fig-01-design-space -->` (P2).

**금지사항**: 공통 금지 1–7. 추가로 — **ch03 파일 수정 금지**(D3); 표 13-2(Miras 대응표) 행 삭제·의미 변경 금지; retention gate·attentional bias는 이 장 소유 — 정의 변경은 가능하나 dossier ledger와의 정합 유지(정합 깨는 변경 금지).

---

## ch14 — `study-kr/part2/ch14-atlas.md`

1. **[K-F02 / critical, D1(i)]** 콜백 삽입 2곳:
   - §14.3.1 Prop 2 대목(:31 이후 φ_p 논의): "이 정리 사슬은 5장 §5.4의 dense-Hopfield 사슬(energy 차수 = feature lift = capacity; Krotov→Ramsauer)의 정식화다 — 독자가 이미 가진 $\phi_p$·$\phi^*$ 직관을 그대로 재사용한다." 류의 1–2문장.
   - φ* / softmax attention 극한 대목: "(→ 5장 §5.4: exponential 극한 = softmax attention)" 참조.
   - §14.7(:299–301) "state 크기는 계산할 수 없다" 결론부에 1문장: "5장 §5.4가 예고한 '비용 계산'은 여기서 **구조**(지불 통화 = state byte·matmul 폭)로 확정된다 — 절대값은 구현 차수 미공개로 불가." (ch05 수정과 정합, D1(ii) 문구 참조.)
2. **[N-C2 / critical]** [NL]·[Sleep] 첫 등장(:258)에 풀네임+arXiv ID 병기.
3. **[N-C3 / critical]** 조사 치환: :19 "inner loop이" / :199 "inner loop이"×2·"outer loop은"×2 / :213 "outer loop은"·"inner loop이" / :215 "inner loop을" / :258 "inner loop은" → 가/는/를.
4. **[N-m1 / 기록, D11]** :31 vec–Kronecker $\otimes$ — v1.1 §1.3 예외 확장으로 **소급 적합, 무수정**. STYLE-ISSUE 주석 불요.
5. **[N-m3 / minor]** 태그 (14-2)·(14-5) — 참조 추가 또는 태그 제거.
6. **[V-c2 관련 / 중간]** :301 TODO-VERIFY(구현 p·sketch·head 분할 부재 최종 확인) — 배치 B1+B3 판정 반영(부재 확정 시 주석 삭제 + 현행 단정 유지에 원문 근거 병기).
7. **[K-F06 / 기록, D6]** 53.55/43.70 표기는 이 장이 기준 — 무수정.
8. **[V-M3 / major, D12]** §14.3 Omega rule 정식화 직후에 `<!-- FIG: ch14/fig-01-omega-window -->` 삽입.

**금지사항**: 공통 금지 1–7. 추가로 — **ch05 파일 수정 금지**(D1); 의무 caveat 2건(Muon 제거 시 ppl 개선, in-context recall 열세)의 완곡화 금지; 표 14-N(Atlas 대응표) 행 삭제·의미 변경 금지; 분량 적정(10.5/10pp) — 콜백 삽입분만큼 인접 중복 산문에서 상쇄 권장.

---

## ch15 — `study-kr/part2/ch15-tnt.md`

1. **[N-C2·N-m10 / critical]** [NL] 풀 인용을 :296에서 첫 등장(:260)으로 이동(순서 위반 해소; :296 자리는 축약 표기 [NL]로).
2. **[N-C3 / critical]** :19 "inner loop은" / :87 "loop이" / :91 "inner loop을" → 는/가/를.
3. **[N-m7 / minor]** :9 "大규모로" → "대규모로".
4. **[N-m3 / minor]** 태그 (15-2)·(15-5) — 참조 추가 또는 태그 제거.
5. **[V-c2 관련 / 낮음]** :258 TODO-VERIFY(Table 3 ppl corpus 기준) — 배치 B1 판정 반영.
6. **[V-M4 / major]** 분량 ~9.8pp — soft 감축(목표 7pp, 상한 4,430단어 지향): 파이프 재배관 해설 등 중복 산문 한정. 의무 caveat 2건(단순화-Titans 검증, mismatch 단일 설정 증거)·원문 인덱스 슬립 각주 3건은 보존. 미달성 시 사유 기록.
7. **[V-M3 / major, D12]** §15.4(계층 구조) 도입부에 `<!-- FIG: ch15/fig-01-tnt-hierarchy -->`, §15.2 mismatch 논의 지점에 `<!-- FIG-REF: ch09/fig-02-three-regimes -->` 삽입.

**금지사항**: 공통 금지 1–7. 추가로 — 표 15-N(TNT 대응표) 행 삭제·의미 변경 금지; chunk-size mismatch·hierarchical memory·two-stage training은 이 장 소유 — ch09 소유 명제(semantic hyperparameter)의 재정의 금지(인용만); Q-K projection "미회수 실마리" 문장(ch16 bridge의 근거) 삭제 금지.

---

## ch16 — `study-kr/part2/ch16-nested-learning.md`

1. **[K-F01 / critical, D2]** ch11 참조 계층 복원 — 삽입 7곳 + 볼드 해제:
   1. :15 — "**online (synaptic) consolidation**" 볼드 해제 → "online (synaptic) consolidation(→ 11장)"; 같은 문장의 offline (systems) consolidation에도 "(→ 11장)".
   2. §16.2의 catastrophic forgetting 첫 등장 — "catastrophic forgetting(→ 11장)".
   3. :418 CF 대목 — "(→ 11장)" 부가.
   4. :441 EWC — "EWC(→ 11장)".
   5. :486 CF 결산 — "(→ 11장)" 부가.
   6. §16.6 CLINC 대목 — "(CF 측정 프로토콜의 배경은 → 11장)" 류 1회.
   7. CTNL ledger 행 — "(CF 측정 축은 → 11장)" 병기.
2. **[K-F06 / minor, D6]** :443 — "53.6 vs 43.7" → "53.55 vs 43.70" ([Atlas Table 5], ch14 표기와 일치).
3. **[N-C2 / critical]** [Titans]·[Miras]·[Atlas]·[TNT]·[Sleep] 각 첫 등장에 풀네임+arXiv ID 병기(5건; [NL]은 :9 완비).
4. **[N-M2 / major]** :68 — "$L$층 MLP $\{W_\ell\cdot+b_\ell\}_{\ell=1}^{L}$" → $L_{\mathrm{layer}}$로 (§16.7의 기존 $L_{\mathrm{layer}}$ 용법과 통일; sequence 길이 $L$(:208 등)과 분리).
5. **[N-m2 / minor]** :305 — Hope forward의 중간 출력 $o_t$ → $z_t$(장-국소)로 개명 + 표 16-2에 대응 행 추가(ch12의 $a_\tau$ 패턴).
6. **[N-m3 / minor]** 태그 (16-2) — 참조 추가 또는 태그 제거.
7. **[N-m6 / minor, D11]** 표 캡션 볼드 → 플레인.
8. **[V-c1 / critical, D7]** 분량 ~16.4pp → **13pp(6,500–7,150단어)**: §16.3.5–16.3.6의 유도 스텝 산문을 결과식+(M) 표준형과의 차이 서술로 축약; Table 1–6 재구성 표의 중복 열 정리; 의무 caveat 2건(Thm 아닌 실증, retrieval gap)·ledger delta·§16.4(outer vs inner)·transfer 5기제는 보존.
9. **[V-c2 / critical]** :297(Eq. 88 decay 적용 대상 — **높음**, 재구현 관건)·:394(Hope momentum 포함 여부 — **높음**, ablation 해석 직결) TODO-VERIFY — 배치 B3/B1 판정 반영(§6.3 v1.1 규약).
10. **[V-M3 / major, D12]** CMS 절(§16.5 부근) 도입부에 `<!-- FIG: ch16/fig-01-cms-spectrum -->` 삽입.

**금지사항**: 공통 금지 1–7. 추가로 — **ch11 파일 수정 금지**(D2); consolidation·CF·EWC의 정의 서술 추가 금지(참조로만 — 이번 위반의 재발 방지); 표 16-2(NL 대응표) 행 삭제·의미 변경 금지(행 추가는 5번 항목대로); 감축이 bridge-in(:ch15 회수)·bridge-out(:ch17 인계) 문단을 건드리지 않을 것.

---

## ch17 — `study-kr/part2/ch17-sleep.md`

1. **[K-F07 / minor, D5]** :78 — LTI(Learning to Imitate) 볼드 정의 직후에 "7장의 LTI(linear time-invariant)와 무관한 약어다." 1문장 삽입(v1.1 §2.4 의무).
2. **[K-F06 / minor, D6]** :314 — "53.6 vs 43.7" → "53.55 vs 43.70".
3. **[K-F04-1·2 / minor, D14]** :54 MoE 가정 문장에 "(MoE/router → 10장 용어집)" 참조 부가; :183 GRPO 첫 등장에 한 줄 글로스(예: "GRPO(group-relative policy optimization — 그룹 내 상대 보상으로 baseline을 대신하는 RLVR 계열 방법)") — 원문 귀속 형식 유지.
4. **[V-c2 / critical]** :202 TODO-VERIFY(Sleep의 Hope가 NL 레시피 checkpoint인지 — **높음**, §17.4 regime 1 서술의 전제)·:9(NL 결산 귀속 위치, 중간) — 배치 B1 판정 반영; :202가 '판정 불가'면 §17.4 해당 서술을 §6.2 귀속/[평가]로 전환.
5. **[V-M4 / major]** 분량 ~11.5pp — soft 감축(목표 9pp, 상한 5,690단어 지향): 실험군 8종 서술의 중복 산문 한정. graft 의무 caveat·세 번째 regime 논의·표 17-3은 보존. 미달성 시 사유 기록.
6. **[V-M3 / major, D12]** §17.2(lifecycle) 도입부에 `<!-- FIG: ch17/fig-01-wake-sleep-lifecycle -->`, update-frequency 좌표 논의 지점에 `<!-- FIG-REF: ch16/fig-01-cms-spectrum -->` 삽입.

**금지사항**: 공통 금지 1–7. 추가로 — LTI 약어를 ch17 밖 의미(linear time-invariant)로 사용 금지(D5: 이 장 내부에서만 Learning to Imitate); 표 17-1(Sleep 대응표) 행 삭제·의미 변경 금지; online/offline consolidation 정의 서술 금지(소유 ch11 — 현행 참조 유지); KS/SKS/LTI/Dreaming/synaptic-pruning reset은 이 장 소유.

---

## 완료 판정 (P2 라운드 exit 조건)

1. §2 매핑표의 전 항목이 각 장에서 적용 완료(critical·major 전건 + minor 전건 — minor의 '태그' 항목만 참조 추가/태그 제거 선택 허용).
2. TODO-VERIFY: 의무분(높음 11건 + ch08:93) 0건 잔존, `audit/w1-verify-answers.md`에 전 판정 기록. 나머지 중·낮음은 배치가 커버한 만큼 해소, 미해소분은 주석 유지(차기 라운드 이월 목록화).
3. 분량: ch01 ≤2,200 / ch10 ≤3,160 / ch16 6,500–7,150 단어(critical 3건은 hard), ch03 ≤3,795·ch06 ≥5,000(major 2건은 hard), ch13/ch15/ch17은 soft(미달 시 사유 기록).
4. FIG/FIG-REF 주석: SPEC.md의 17개 그림 ID 전부가 소유 장에 1회씩 존재, 재사용 2건(ch10·ch15의 F-09, ch17의 F-16)은 FIG-REF로 존재.
5. 전 장이 STYLE v1.1 §7 체크리스트 재통과. 신규 STYLE-ISSUE는 수집해 차기 개정 후보로.
