# W1 Continuity Audit — 서사 연속성·개념 누적

**감사 축**: (a) Part II bridge-out↔bridge-in 맞물림 + dossier Bridge 1–5 정합, (b) concept ledger delta의 누적 일관성, (c) Part II가 사용하는 배경 개념의 Part I 교육 여부(전방 참조 탐지).
**감사 대상**: `study-kr/part1/ch01–ch11`, `study-kr/part2/ch12–ch17` (17개 장 전부 검사).
**기준 문서**: `style/STYLE-NOTATION.md` (§2.3 정의 규칙, §2.4 정의 소유권, §4.2 장별 전작 사슬), `dossier/PRE-RESEARCH.md` (§1 Bridge 1–5, §2 concept ledger, §3 convergence).
**작성일**: 2026-07-12 (W1 1차 초안 기준).

---

## 0. 총평

Part II 6개 장의 bridge 사슬(ch08→12→13→14→15→16→17)은 **실제로 맞물린다**. 각 bridge-out이 지목한 한계를 차작 bridge-in이 같은 표현·같은 수치로 이어받고, dossier Bridge 1–5의 "carried / generalized / discarded" 목록 및 의무 정직성 항목(Muon ablation, TNT 단순화, Sleep graft)까지 본문에 반영되어 있다. concept ledger delta도 dossier §2와 거의 전 행에서 정합한다. 이 축의 골격은 건강하다.

발견된 결함은 골격이 아니라 **연결부의 국소 단선**이다: (1) ch16이 자신의 전용 배경 장인 ch11을 단 한 번도 참조하지 않고 ch11 소유 개념을 재정의하는 것(critical), (2) ch05→ch14 사이의 "약속-불이행" 모순(critical), (3) ch13의 ch03에 대한 허위 역참조(major), 그 외 minor 다수.

---

## 1. (a) Bridge 맞물림 검사 — 결과: 6/6 통과

| 사슬 | bridge-out (전작) | bridge-in (차작) | dossier 정합 | 판정 |
|---|---|---|---|---|
| ch08→ch12 | ch08 §8.7: TTT의 3결핍 — forgetting 부재 / momentum 부재 / deep memory 미검증 ([Titans App. C] 귀속) + 표 8-2 지도 | ch12 §12.1: 동일한 3결핍을 같은 순서로 복원, DeltaNet 계열과의 "빈 가운데" 구도 | (dossier는 Bridge 0을 명시 안 함 — 문제없음) | **통과** |
| ch12→ch13 | ch12 §12.8: App. C 회수 → "좌표축 미명명" / 왜 ℓ2·왜 momentum+decay / momentum 분해 예고 | ch13 §13.1: 같은 세 질문("왜 하필 그 선택인가") 복원 | Bridge 1 — carried(k→v loss, deep MLP, chunkwise, 게이트, bilevel)·generalized(surprise→임의 bias, forget→retention)·discarded(momentum 보류, algorithm 축 공백) 전부 ch13 본문·ledger에 존재 | **통과** |
| ch13→ch14 | ch13 §13.8: (i) optimizer 축 공백 (ii) per-token 단일쌍 objective (iii) capacity 이론 부재 | ch14 §14.1: 동일한 "세 개의 공백"을 항목별 복원 | Bridge 2 — 3항 일치; "test-time memorization" 개명(ch14 ledger 소유), Muon ablation 의무 caveat(§14.6 [평가]) 포함 | **통과** |
| ch14→ch15 | ch14 §14.8: wall-clock 0건, C의 semantic knob 방치, "다른 C로 serve하면?" 미질문 + TNT Fig. 2 수치 예고(13.78/36.45) | ch15 §15.1–15.2: wall-clock 부재, 5–10% utilization(LaCT 인용), Challenge 1/2/3; Fig. 2 수치 일치 | Bridge 3 — 전 요소 일치; App. D 단순화 caveat(§15.6, §15.8) 포함 | **통과** |
| ch15→ch16 | ch15 §15.8: "서로 다른 주기 + 느린 timescale에 주차된 지식이 reset을 생존" — 정당화 없는 트릭; Q-K projection은 미회수 실마리로 명시 | ch16 §16.1: 같은 문장 구조로 복원("심어 놓고도 정당화하지 않은 아이디어") | Bridge 4 — CMS=계층의 아키텍처화, reset→re-init, W_init→기제 4(MAML류), Miras Def. 1 재사용, Omega rule 수입 전부 ch16 ledger에 존재 | **통과** |
| ch16→ch17 | ch16 §16.8: CF 미해결 / offline consolidation 범위 밖 / capacity 고정 / 적응은 입력 중에만 — 4항 | ch17 §17.1: **동일 4항을 같은 순서로** 복원 | Bridge 5 — 4항 일치; graft-vs-cotrain 이음새(§17.1 "방법론적으로 결이 다른 물건", §17.8 (1)) 포함 | **통과** |

추가 확인: ch17 §17.8의 "라인의 완성형" 5개 항목과 "남은 open problems" 5개는 dossier §3.1/§3.2와 1:1 대응하며, 두 유보(개념적 완성 / 소급적 독법)도 dossier의 Qualification 1·2 그대로다. 수치 에코 일관성: 17.37× (ch09/ch10/ch15 동일), TNT Fig. 2 (ch14/ch15 동일). — 예외 1건은 §2.2 F-06 참조.

---

## 2. (b) Concept ledger 누적 일관성

### 2.1 통과 항목 (표본 검증 ~20행)

dossier §2 ledger의 다음 행들이 각 장 ledger delta와 정합함을 확인: surprise(12→13→14→16), momentum-as-memory(12→13 보류→14 부활→16 정리), retention 계보(12 weight decay→13 재이론화→14 γ admission→15 reset→17 성장), attentional bias(13→14→16), Omega rule(14→16 수입), deep memory(12→13 표준화→14 capacity), persistent memory(12→16 frequency-0), MAC/MAG/MAL(12→14·16), chunkwise(9→13 lag token→14 banded mask→15 경제학), chunk 크기=semantic knob(9 명제화·15 실증 — 양쪽 서술 일치), W_init(12 암묵→15 load-bearing→16 기제 4→17 teacher 계보), Q-K projection(15, 미회수 open thread — ch15 §15.8 명시), multi-timescale(12 삼분→15 계층→16 연속체), self-modification(16 rule→17 data), consolidation(16 online→17 offline), wake/sleep(17), TC⁰(12 증명 부재→16 실증-only). 명칭 계보(test-time training→memorization[ch14 소유]→NL의 이의[ch16 폐기행]→Sleep의 소거[ch17 폐기행])도 충돌 없이 누적적이다.

### 2.2 발견 사항

**F-05 (minor) — momentum-as-memory 정의 소유권 표기 충돌.**
`part2/ch12-titans.md:263–268` 표 12-3의 열 제목이 "개념 (이 장이 정의)"인데 momentum-as-memory 행이 그 아래 있다. STYLE §2.4는 이 개념의 정의 장을 **ch02(객체로서), ch16(정리로서)** 로 지정하고, 실제로 `part1/ch02-training-as-a-system.md:136`이 이 용어를 볼드로 도입하며 "이 장은 객체로서의 사실만 확정하고 정리로서의 지위는 16장이 다룬다"고 소유를 선언한다. ch12는 "기원(Titans)"이지 "정의 장"이 아니다. 수정: 표 12-3의 해당 행에 "(정의 소유: 2장·16장, 기원: 이 장)" 병기 또는 열 제목을 "개념 (이 논문이 도입)"으로 완화.

**F-06 (minor) — retrieval gap 수치의 장간 표기 불일치.**
ch14는 원문 그대로 **53.55 vs 43.70** [Atlas Table 5]를 두 번 인용(`part2/ch14-atlas.md:283, 319`). 그런데 ch16 `part2/ch16-nested-learning.md:443`과 ch17 `part2/ch17-sleep.md:314`는 "(→ 14장)"을 달고도 **53.6 vs 43.7**(dossier의 반올림값)로 쓴다. 참조 대상 장의 수치와 인용 수치가 다르다. 수정: ch16/ch17을 53.55/43.70으로 통일.

**F-07 (minor) — 약어 LTI의 이중 사용.**
ch07(`part1/ch07-ssm-lineage.md:38`)이 **LTI = linear time-invariant**를 볼드 정의하고 ch09:143, ch12:247·276("LTI fast path")이 그 의미로 재사용하는데, ch17(`part2/ch17-sleep.md:78`)은 **LTI = Learning to Imitate**를 볼드 정의한다(STYLE §2.4의 공식 명칭이므로 폐기 불가). 수정: ch17 첫 등장에 "(7장의 LTI(linear time-invariant)와 무관한 약어)" 한 줄 각주, 또는 ch12 ledger의 "LTI fast path"를 "chunk-당-상수 게이트 fast path"로 개명.

**F-08 (minor) — ch09의 소유권 문장 중의성.**
`part1/ch09-chunkwise-parallel-training.md:126`: "…[TNT Fig. 2]의 chunk-size mismatch이고(→ 9.7절), 이 명제의 명명과 소유는 이 장에 있다" — 문장 구조상 "이 명제"가 chunk-size mismatch(소유: ch15)를 가리키는 것으로 오독될 수 있다. 실제 의도는 semantic-hyperparameter 명제(소유: ch09). 수정: "…이고, 'chunk 크기=semantic hyperparameter' 명제의 명명과 소유는 이 장에 있다"로 명시.

---

## 3. (c) 가르쳐지지 않은 의존성 (전방 참조) — Part II → Part I 매핑

긍정 확인부터: Part I의 교육 커버리지는 전반적으로 촘촘하다. Part II가 기대는 배경 대부분이 실제 교육됨 — backprop·optimizer 객체·Muon/NS(ch02), OGD/regret/FTRL/mirror descent/Bregman/exponentiated gradient/simplex/f-divergence/Huber/ℓp(ch03), bilevel/MAML/hypergradient/mesa-optimization(ch04), Hopfield/crosstalk/delta rule/dense Hopfield/NW(ch05), FWP/DeltaNet 계열/TC⁰·state tracking(ch06), SSM/scan/SSD/LTI(ch07), TTT/dual form(ch08), chunkwise 4 인스턴스·(M4)(ch09), cost model·glossary(ch10), CF/EWC/replay/CLS/consolidation/GKD/LoRA/REINFORCE/ReST^EM/SEAL/Levenshtein(ch11). ch01 표 1-2의 "열쇠 장" 매핑과 각 장 "왜 필요한가" 박스도 실제 의존과 일치한다 — **단 하나의 예외가 아래 F-01이다.**

### 3.1 결함

**F-01 (critical) — ch16이 전용 배경 장 ch11을 0회 참조하고, ch11 소유 개념을 재정의한다.**
- 증거: `part2/ch16-nested-learning.md` 전체에서 "11장" 문자열 **0회** (ch17은 11회 참조).
- ch16이 참조 없이 사용하는 ch11 소유(STYLE §2.4) 개념: **catastrophic forgetting** (`:418, :486` 등), **online/offline consolidation** (`:15` — "**online (synaptic) consolidation**"을 볼드로 **첫 정의처럼** 재도입; §2.3 위반: "이전 장에서 이미 정의된 개념은 재정의하지 않고 참조한다"), **EWC** (`:441`), **replay**, anterograde amnesia·CLS 구도 (`:13–17` — ch11 §11.3–11.4가 정확히 이 구도를 위해 존재).
- 모순되는 자기 선언: ch11 "왜 필요한가" 박스(`part1/ch11-continual-learning.md:9`)는 "이 장은 오직 [NL]과 [Sleep] 두 편을 위해 존재한다… CLINC·CTNL, **16장에서 상술**"이라 하고, ch01 표 1-2도 NL의 열쇠 장에 **11장**을 명시하며, ch10 "다음 장으로"도 같은 약속을 한다. 약속의 수신자인 ch16만 그 사슬을 모른다.
- 수정(차기 라운드 필수): ch16 §16.2에 catastrophic forgetting(→ 11장), online/offline consolidation(→ 11장) 참조 삽입 + `:15`의 볼드를 참조형("online (synaptic) consolidation(→ 11장)")으로 강등; §16.6의 EWC·CLINC 대목과 §16.8의 CF 결산에도 (→ 11장) 부가.

**F-02 (critical) — ch05 §5.4의 약속을 ch14가 이행하지 않고, 비용 계산 약속은 모순으로 끝난다.**
- ch05 §5.4(`part1/ch05-associative-memory.md:80–96`)는 dense/modern Hopfield 사슬(Krotov 차수↑=feature lift=capacity↑, exponential 극한=softmax attention)을 절 제목부터 "**Atlas의 이론적 골격**"으로 세우고, "이 사슬을 지금 손에 쥐고 있으면 **14장은 corollary처럼 읽힌다**", "이 거래의 **정밀한 비용 계산은 14장에서 한다**"고 두 번 약속한다.
- ch14 §14.3.1(`part2/ch14-atlas.md:27–51`)은 capacity 이론 전체를 전개하면서 ch05를 **고전 Hopfield와의 대조로만** 두 번 언급하고(`:11, :29`), Krotov/Ramsauer·dense Hopfield 사슬로의 콜백이 전무하다(grep: "Krotov|Ramsauer|dense Hopfield" 0회). 독자가 배운 φ_p·φ* 직관이 재사용되지 않고 재유도된다.
- 비용 약속은 모순: ch14 §14.7(`:299–301`)은 "구현된 차수 p와 sketch 차원이 공개되지 않은 한 **state 크기는 계산할 수 없다**"고 결론 — ch05의 "정밀한 비용 계산은 14장에서" 약속과 정면 충돌.
- 수정(차기 라운드 필수): (i) ch14 §14.3.1의 Prop 2/φ* 대목에 "(→ 5장 §5.4: energy 차수 = feature lift = capacity 사슬)" 콜백 1–2문장 삽입, (ii) ch05의 약속 문구를 "비용의 **구조**는 14장에서 다루되, 구현 차수 미공개로 절대값은 계산 불가(14장 §14.7)"로 완화하거나 ch14 §14.7에 ch05의 65× 예시 계산을 조건부로 재수록.

**F-03 (major) — ch13의 허위 역참조: "3장의 도구함 — elastic net".**
`part2/ch13-miras.md:175` (§13.3.6 도입부): "3장의 online optimization 도구함 — mirror descent, Bregman divergence, **elastic net** — 이 통째로 수입된다." mirror descent·Bregman은 ch03 §3.5에 있으나 **"elastic net"·LASSO·soft-thresholding은 Part I 어디에도 없다** (ch03 §3.6–3.7은 ℓ1/ℓ2 벌점의 soft/hard eviction 구도까지만 교육 — 개념적 재료는 있으나 명칭·결합형은 미교육). 독자가 ch03을 뒤져도 찾지 못한다. 수정: 해당 문장에서 elastic net을 빼고 "(elastic net = ℓ1+ℓ2 결합, 여기서 정의)"로 장-국소 도입하거나, ch03 §3.6에 elastic net 한 문단 추가.

**F-04 (minor) — 미교육 명칭들 (명단).**
1. **MoE / router / sparse expert dispatch** — ch17 §17.3.3(`part2/ch17-sleep.md:54`)이 "일반성을 잃지 않고 각 MLP은 router를 가진 sparse MoE라고 가정한다"로 도입. Part I에 MoE 0회(ch11 §11.2는 parameter isolation·multi-LoRA로 인접 지반까지만; ch10 §10.7 glossary에도 MoE 행 없음). 독자군(inference 엔지니어)이 대개 알므로 minor — 권고: ch10 glossary에 MoE/router 1행 추가.
2. **GRPO** — ch17 `:183, :251, :257` 등에서 baseline으로 등장하나 어떤 장도 글로스하지 않음(ch11 §11.6은 REINFORCE·ReST^EM·SEAL까지). 권고: ch17 첫 등장에 한 줄 정의 또는 ch11 §11.6에 1문장.
3. **Reptile / CAVIA** — ch12 §12.4(`part2/ch12-titans.md:216`)가 [Titans §3.1] 귀속으로 명명. ch04 §4.3(learning-to-learn 계보)에 두 이름 부재. 권고: ch04 §4.3에 "first-order 변형(Reptile), context-parameter 변형(CAVIA)" 1문장 또는 ch12에서 괄호 글로스.
4. **KKT 조건** — ch13 §13.3.6 (13-8) 유도에 1회 사용, 미교육. 유도가 논문 귀속이므로 minor — 권고: "(제약 최적화의 1차 조건)" 괄호 글로스.

---

## 4. 발견 사항 총괄

| ID | 심각도 | 위치 | 요약 |
|---|---|---|---|
| F-01 | **critical** | part2/ch16-nested-learning.md:15,418,441,486 (+장 전체) | ch16이 ch11을 0회 참조; CF·online/offline consolidation·EWC 등 ch11 소유 개념을 참조 없이 사용, consolidation은 볼드 재정의(STYLE §2.3·§2.4 위반). ch01·ch10·ch11이 설치한 의존 사슬의 수신 측 단선 |
| F-02 | **critical** | part1/ch05-associative-memory.md:80–96 ↔ part2/ch14-atlas.md:27–51,299 | ch05 §5.4의 "Atlas의 이론적 골격/corollary/정밀 비용 계산은 14장" 약속을 ch14가 미이행(dense-Hopfield 콜백 0회) + 비용 계산은 "계산 불가"로 모순 |
| F-03 | major | part2/ch13-miras.md:175 ↔ part1/ch03 | "3장의 도구함"에 elastic net 포함 — ch03에 부재(허위 역참조) |
| F-04 | minor | part2/ch17-sleep.md:54,183 / part2/ch12-titans.md:216 / part2/ch13-miras.md(§13.3.6) | 미교육 명칭: MoE/router, GRPO, Reptile/CAVIA, KKT — 글로스 또는 Part I 1문장씩 |
| F-05 | minor | part2/ch12-titans.md:263–268 ↔ style §2.4, part1/ch02:136 | momentum-as-memory의 "이 장이 정의" 표기가 소유권(ch02/ch16)과 충돌 |
| F-06 | minor | part2/ch16:443, part2/ch17:314 ↔ part2/ch14:283 | retrieval gap 53.6/43.7 vs 53.55/43.70 — (→ 14장) 인용인데 수치 불일치 |
| F-07 | minor | part2/ch17-sleep.md:78 ↔ part1/ch07:38, part2/ch12:247,276 | 약어 LTI 이중 사용 (linear time-invariant vs Learning to Imitate) |
| F-08 | minor | part1/ch09-chunkwise-parallel-training.md:126 | "이 명제의 명명과 소유는 이 장" 문장이 chunk-size mismatch(ch15 소유)를 가리키는 것으로 오독 가능 |

**참고 (결함 아님)**: dossier §4의 장 번호 체계(Titans=ch11, interlude=ch14)는 최종 책 번호(Titans=ch12, continual=ch11)와 다르다. 책 내부(ch01 지도, ch10→ch11 bridge, ch11 박스, STYLE §4.2)는 전부 최종 번호로 자기일관적이므로 수정 불요 — 후속 감사자는 dossier 번호를 책 번호로 읽지 말 것.

---

## 5. 차기 라운드 수정 지시 (critical만)

1. **ch16에 ch11 참조 계층 복원** — §16.2: "catastrophic forgetting(→ 11장)", "online (synaptic) consolidation(→ 11장)"으로 볼드 재정의 해제; §16.6 EWC·CLINC 대목과 §16.8 CF 결산에 (→ 11장); CTNL ledger 행에 "(CF 측정 축은 → 11장)" 병기. 예상 규모: 5–7개 참조 삽입, 본문 재작성 불요.
2. **ch05↔ch14 약속 정합화** — ch14 §14.3.1에 dense-Hopfield 사슬 콜백(1–2문장, Prop 2와 φ* 대목), ch14 §14.7 또는 ch05 §5.4 말미의 비용-계산 약속 문구 조정(둘 중 한쪽 수정으로 모순 해소).
