# W1 재감사 — 연속성·개념 누적 (P2 폐쇄 검증)

**축**: 서사 연속성·개념 누적 (원 리포트 `audit/w1-continuity-report.md`).
**목적**: fixplan(`audit/w1-fixplan.md` D1–D6·D14)과 TODO 판정(`audit/w1-todo-resolutions.md`)이 실제 17개 장에 적용됐는지 폐쇄 검증. 원 리포트 F-01…F-08 각각의 닫힘/미해결 판정 + 수정으로 새로 생긴 위반 표본 검사.
**검증일**: 2026-07-12 (P2 보정 완료본 기준).
**방법**: 각 결함의 지시 위치를 본문에서 직접 확인(grep + 정독). F-01·F-02는 "형식적 참조 삽입"인지 "서사 실질 연결"인지까지 판정.

---

## 0. 총평 — 8/8 닫힘, 신규 위반 0

원 리포트의 critical 2건(F-01, F-02), major 1건(F-03), minor 5건(F-04…F-08)이 **전부 닫혔다**. 특히 두 critical은 형식적 참조 삽입에 그치지 않고 **서사가 실제로 이어진다**(§1, §2 상술). bridge 사슬(ch12→…→17)·concept ledger·의무 caveat은 수정 과정에서 훼손되지 않았고, 감축(D7 ch16, D8 등)이 bridge-in/out 문단이나 ledger 행을 건드리지 않았음을 표본 확인했다. 신규 전방참조·허위 역참조는 발견되지 않았다.

| ID | 심각도 | 판정 | 실질/형식 |
|---|---|---|---|
| F-01 | critical | **닫힘** | 실질 — ch16이 ch11로 7곳 참조 + 볼드 재정의 해제, 신경생리→consolidation→ch17 서사가 ch11 CLS 교육으로 관통 |
| F-02 | critical | **닫힘** | 실질 — ch14가 φ_p·φ* 직관을 명시 재사용해 "corollary" 약속 이행, ch05 비용-약속 모순 해소 |
| F-03 | major | **닫힘** | ch13이 elastic net을 "3장 도구함"에서 빼고 장-국소 도입 |
| F-04 | minor | **닫힘** | MoE/router·GRPO·Reptile/CAVIA·KKT 4건 전부 글로스/1문장 삽입 |
| F-05 | minor | **닫힘** | 표 12-3 열제목 "이 논문이 도입" + momentum 행에 "(정의 소유: 2장·16장)" |
| F-06 | minor | **닫힘** | ch16·ch17 retrieval gap 53.55/43.70으로 통일 |
| F-07 | minor | **닫힘** | ch17:79 LTI 구분 문장, ch12:245 "LTI=linear time-invariant → 7장" 병기 |
| F-08 | minor | **닫힘** | ch09:124 "'chunk 크기=semantic hyperparameter' 명제"로 지시대상 명시 |

---

## 1. F-01 (critical) — ch16 → ch11 단선 — **닫힘 (실질)**

**D2 지시 = 참조 7곳 + 볼드 재정의 해제. 전수 확인:**

1. `ch16:15` — "online (synaptic) consolidation**(→ 11장)**" + "offline (systems) consolidation**(→ 11장)**". 볼드 재정의 **해제됨**(STYLE §2.3 위반 해소). ✔ 2곳
2. `ch16:19` — "continual learning**(→ 11장)**" (§16.2 핵심 주장 (4)). ✔
3. `ch16:421` — CTNL ledger 행: "in-context catastrophic forgetting 측정 **(CF 측정 축은 → 11장)**". ✔
4. `ch16:444` — "EWC**(→ 11장)**" + CLINC 대목 "class-incremental 분류**(CF 측정 프로토콜의 배경은 → 11장)**". ✔ 2곳
5. `ch16:489` — CF 결산: "**catastrophic forgetting(→ 11장)은 해결되지 않았다.**" ✔

**"11장" 문자열 0회 → 7회.** ch16 전체에서 CF·consolidation·EWC의 모든 리터럴 등장이 이제 (→ 11장) 참조를 동반한다(재정의 0건).

**실질 연결 판정 (형식 아님)**: 단순 포인터 태그가 아니다. `ch16:15`의 신경생리학 동기(두 단계 consolidation)가 **ch11의 CLS·online/offline consolidation 교육으로 독자를 되돌려 보내고**, 같은 개념이 `ch16:501–503` bridge-out에서 offline consolidation → ch17로 이어진다. 즉 ch01·ch10·ch11이 설치한 의존 사슬의 수신 측이 복원되어, "ch11은 오직 NL·Sleep을 위해 존재한다"는 ch11의 자기 선언이 이제 ch16에서 실제로 회수된다. 서사가 끊긴 자리가 이어졌다.

---

## 2. F-02 (critical) — ch05 §5.4 ↔ ch14 약속 — **닫힘 (실질)**

**D1(i) ch14 콜백 2곳 + 비용결론 1문장:**
- `ch14:39` (Prop 2): "이 정리 사슬은 5장 §5.4의 dense-Hopfield 사슬(**Krotov→Ramsauer: energy 차수 = feature lift = capacity**)의 정식화다 — 그 사슬을 통과한 독자에게 Prop 2는 **corollary처럼 읽히며, 이미 가진 φ_p·φ* 직관을 그대로 재사용**하면 된다." ✔ (grep "Krotov|Ramsauer|dense Hopfield" 0회 → 등장)
- `ch14:49` (softmax 극한): "**(→ 5장 §5.4: exponential 극한 = softmax attention)**". ✔
- `ch14:301` (§14.7): "5장 §5.4가 예고한 '비용 계산'은 그래서 여기서 **구조** … 로 확정된다; 절대값 계산은 구현 차수 미공개로 여기서도 불가하다는 것까지가 이 장의 결론이다." ✔

**D1(ii) ch05 모순 문장 완화:**
- `ch05:92`: "이 거래의 비용 **구조**(…state byte·matmul 폭…)는 14장 §14.7에서 확정한다 — 단, Atlas가 구현 차수 p와 sketch 차원을 공개하지 않아 절대값 계산은 그곳에서도 불가능하다는 것까지가 14장의 결론이다." ✔ — "정밀한 비용 계산은 14장에서 한다"의 무조건 약속이 조건부로 교체됨. `ch05:90`의 "14장은 corollary처럼 읽힌다"·65× 예시는 유지(ch14가 콜백으로 이행).

**실질 연결 판정**: ch14가 재유도로 때우지 않고 ch05에서 배운 φ_p·φ* 직관을 명시적으로 호명·재사용하며, "corollary" 약속을 문자 그대로 이행한다. 비용-약속의 정면 모순(ch05 "정밀 계산" vs ch14 "계산 불가")은 양측이 **구조 확정 / 절대값 불가**로 정합화되어 사라졌다. 두 장이 이제 같은 문장을 공유한다.

---

## 3. F-03 (major) — ch13 elastic net 허위 역참조 — **닫힘**

`ch13:177`: "3장의 online optimization 도구함 — mirror descent, Bregman divergence — 이 통째로 … 수입되고, 여기에 **elastic net**(ℓ1+ℓ2 결합 벌점; **3장의 ℓ1/ℓ2 eviction 구도의 결합형으로, 이 장에서 도입한다**)이 더해진다." ✔ — elastic net이 "3장 도구함" 열거에서 빠지고 장-국소 볼드 도입으로 전환(D3). ch03은 무수정(ℓ1/ℓ2 eviction 구도는 실재하므로 역참조 유효). 신규 허위참조 없음.

---

## 4. 마이너 4건 — 전부 닫힘

- **F-04** (미교육 명칭): `ch04:89` Reptile/CAVIA 1문장 + "[Titans §3.1] … (→ 12장)"; `ch10:212` glossary "MoE / router" 행 신설(→ 17장); `ch17:55` "(MoE/router → 10장 용어집)"; `ch17:184` "GRPO = group-relative policy optimization — …"; `ch13:179` "KKT 조건(제약 최적화의 1차 최적성 조건)". ✔ 4/4
- **F-05** (momentum-as-memory 소유권): `ch12:263` 표 12-3 열제목 "개념 (**이 논문이 도입**)" + `ch12:266` 행 "momentum-as-memory **(정의 소유: 2장·16장)**". ✔
- **F-06** (retrieval gap): `ch16:446` "**53.55 vs 43.70**([Atlas Table 5], → 14장)"; `ch17:317` "attention **53.55 vs 43.70** [Atlas]". 옛 53.6/43.7 잔존 0건. ✔
- **F-07** (LTI 이중 사용): `ch17:79` "**7장의 LTI(linear time-invariant)와 무관한 약어다.**"; `ch12:245` "LTI(**linear time-invariant → 7장**) 시스템". ✔
- **F-08** (ch09 소유권 중의성): `ch09:124` "**'chunk 크기 = semantic hyperparameter' 명제**의 명명과 소유는 이 장에 있다(→ §2.4; **mismatch 현상 자체의 소유는 15장**)." — 지시대상 명시됨. ✔

---

## 5. 신규 위반 표본 검사 — 없음

- **bridge 사슬 무손상**: ch14 bridge-in(공백 3항)·bridge-out(→15, TNT ppl 13.78/36.45) 유지; ch16 bridge-in(§16.1 TNT 회수)·bridge-out(§16.5 → Sleep) 유지; concept ledger 행(ch16:413·415, ch12 표 12-3) 무삭제. ch16 감축(D7)이 이 문단들을 건드리지 않았음.
- **신규 전방참조 0**: F-02·F-03 콜백은 전부 후방참조(ch14→5장, ch13→3장). 새로 도입된 개념이 미교육 상태로 사용되는 곳 없음.
- **수치 에코 일관성 유지**: 17.37×(ch09:172), TNT Fig. 2(ch14:328) 등 장간 수치 그대로. F-06 통일로 오히려 개선.
- **false positive 1건 기록**: `ch17:259`의 "43.7"은 실험 비교표(SFT / AIME-25) 값으로 Atlas retrieval gap과 무관 — 수정 대상 아님.

---

## 6. 잔여 관찰 (비차단 — 차기 라운드 선택)

- `ch16:501` bridge-out의 볼드 **online consolidation** / **offline consolidation**은 §16.2에서 이미 (→ 11장)로 확립한 개념의 **대조 강조**이지 첫 정의가 아니므로 STYLE §2.3 위반이 아니다. 엄밀주의자가 이곳까지 볼드 해제를 원할 수 있으나, D2 지시 범위(‌:15만) 밖이고 정의-볼드가 아니므로 폐쇄 판정에 영향 없음. 필요 시 차기 라운드에서 강조→일반으로 전환 가능(cosmetic).

**결론: 연속성 축 P2 보정 폐쇄 검증 통과. critical 2 / major 1 / minor 5 전건 닫힘, 서사 실질 연결 확인, 신규 위반 0.**
