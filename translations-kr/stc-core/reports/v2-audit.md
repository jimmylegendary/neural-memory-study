# STC 의역본 15편 — v2 기준 감사 보고

**감사일**: 2026-08-06
**대상**: `translations-kr/stc-core/STC-T01 … STC-T15`의 `body.md` 15편
**기준**: `notes/stc-v2/*.json`(사실), 각 논문 원문(판정), `style/STC-STYLE-NOTATION.md` §2.4(용어), `manifest.json`(권리)
**방법**: 노트의 `experiments`·`cost_table`·`core_mechanism`과 의역본을 항목 대조하고, 어긋나거나 노트에 없는 항목은 `sources/`의 고정 원문(pdftotext / BioC XML)을 직접 열어 판정했다. 외부 모델에 위임하지 않고 전량 직접 대조했다.

> **범위 밖**: 감사 진행 중(19:42) 다른 세션이 `STC-T16/`을 새로 만들고 `manifest.json`을 16편으로 갱신했다. T16은 이 감사의 대상이 아니며 확인하지 않았다. `manifest.json`은 충돌을 피하려고 건드리지 않았다.

---

## 0. 요약

| 구분 | 건수 | 대상 |
|---|---|---|
| **고침** — 원문 대조로 확정된 오류·왜곡 | 8편 17건 | T01, T02, T04, T05, T07, T09, T14, T15 |
| **보고만** — 용어 divergence | 4편 | T01, T02, T10, T16(범위 밖) |
| **보고만** — 누락·미수록 | 8편 | T01, T04, T06, T09, T11, T12, T13, T14 |
| **권리 metadata** | 이상 없음 | public 2편 CC BY 확인 완료 |
| **노트 오류** | **0건** | 대조한 모든 노트 수치가 원문과 일치 |

**가장 중대한 발견 2건**

1. **T04 §3.3이 논문의 대조실험 결론을 정반대로 서술하고 있었다.** 원문은 균일 noise sleep(Uniform-Noise Sleep)이 원래 sleep 구현과 **유사한 결과를 냈다**고 보고하고, 거기서 "sleep 중 발화를 구동하는 입력의 세부 성질은 replay에 필수적이지 않다"는 결론을 끌어낸다. 의역본은 "무작위 activity … 는 같은 효과를 내지 못한다"고 반대로 적고 있었다. 존재하지 않는 두 개의 대조군("plasticity를 끈 sleep", "wake update 수를 맞춘 control")도 함께 서술되어 있었다.
2. **T02는 폐기된 v1 metadata를 그대로 달고 있었고 표 2·3·4가 붕괴되어 있었다.** 제목이 v1 시기의 가제(`Sleep (offline consolidation for LLMs)`), 저자 목록에서 Adel Javanmard 누락, 버전이 `v1 … 2026년 6월 2일`. 실제 고정 원문은 v2(2026-07-10)다. 표 2는 Qwen3-8B 블록 전체가 표 밖으로 튀어나가 있고 HMMT-25 열이 비어 있었으며, 표 3·4는 셀이 본문에 낱줄로 흩어져 있었다.

---

## 1. 고친 것 (17건)

### STC-T01 — Sleep-time Compute (Letta)

수치는 전부 정확했다(5× / 13% / 18% / 2.5×, abstract 및 §5 대조 완료). 다만 **논문이 스스로 붙인 조건과 반례가 세 군데 빠져** 있어, 그대로 두면 v2 본문이 이 의역본을 인용할 때 주장 강도를 과대평가하게 된다.

| # | 위치 | 고친 내용 | 근거 |
|---|---|---|---|
| 1 | §5.1 | 5×가 **낮은 test-time budget 구간 한정**임을 명시. 높은 budget에서는 test-time-only 기준선이 근소하게 앞선다는 저자들의 보고와 그 미검증 가설, o1의 제한적 이득을 추가. pass@k 병렬 스케일링 비교(oracle verifier 이점을 가진 baseline을 넘어섰다는 저자들의 근거 포함)를 추가 — 이는 §5.1의 headline 주장 중 하나인데 의역본에 아예 없었다. | 원문 §5.1, Fig. 3–6 |
| 2 | §5.3 | "보고된 설정에서 2.5배" → **문맥당 질의 10개일 때의 최대치**, 생성 token만 센 비용 모형임을 명시. 질의 수가 적으면 총비용 관점에서 불리하다는 Figure 9 caption의 단서 추가. | 원문 §5.3, Fig. 9 |
| 3 | §6 | SWE-Features의 실제 수치(낮은 budget에서 test-time token 약 1.5배 감소)와, 높은 budget에서는 표준 조건의 precision이 더 높다는 반대 방향 결과, 평가 지표가 파일 집합 F1이고 테스트는 채점에 쓰이지 않는다는 사실 추가. | 원문 §6, Fig. 11 |

### STC-T02 — Language Models Need Sleep

| # | 위치 | 고친 내용 | 근거 |
|---|---|---|---|
| 4 | 머리말 1·3행 | 제목을 v1 가제에서 실제 제목으로 교체 | 원문 표지 |
| 5 | 9행 | 저자에 **Adel Javanmard** 추가 (4인) | 원문 표지 |
| 6 | 15행 | `arXiv:2606.03979v1 … 2026년 6월 2일` → **v2 … 2026년 7월 10일** | 원문 `arXiv:2606.03979v2 [cs.LG] 10 Jul 2026` |
| 7 | §4.1 표 2 | Qwen3-8B 블록 5행 복원, HMMT-25 열 복원(1.7B: 25.7/22.9/26.1/28.1/29.3, 8B: 42.4/43.7/44.9/45.1/46.1), Sleep 1.7B의 AIME-25 40.2 복원 | 원문 Table 2 |
| 8 | §4.1 표 3 | 8행 × 2열 표로 재구성(ablation 3행 포함) | 원문 Table 3 |
| 9 | §4.1 표 4 | 4행 표로 재구성(ICL 0 / TTT 10 / SEAL 72.5 / Sleep 80) | 원문 Table 4 |

표 1(Qwen3-8B ablation)은 원래 정상이었고 원문과 일치한다. 표 2와 표 1의 caption이 거의 같은 것은 **원문 자체가 그렇다**(둘 다 "Performance of different methods on mathematical reasoning benchmarks … average@16"). 의역본의 오류가 아니다.

### STC-T04 — Sleep prevents catastrophic forgetting in SNNs (PLOS)

| # | 위치 | 고친 내용 | 근거 |
|---|---|---|---|
| 10 | §3.1 | 수치 없음 → 실제 값 삽입(순차 학습 후 Task 1 0.52 ± 0.02 = 우연 수준, Task 2 0.69 ± 0.03; 단일 과제 기준 0.70 ± 0.02 / 0.69 ± 0.03). 10회 이상 시행 평균 ± 표준편차이고 **유의성 검정·신뢰구간이 논문에 없다**는 사실 명시 | 원문 Results |
| 11 | §3.3 **[정반대 서술]** | "무작위 activity 또는 plasticity를 끈 sleep은 같은 효과를 내지 못한다" 삭제. 실제 세 대조실험으로 교체: (i) 순진한 network + Task 2 발화율 noise → Task 2는 기준선(0.60 ± 0.03 / 0.49 ± 0.05), (ii) Task 1 후 Interleaved_{S,T1} → Task 1 유지 0.71 ± 0.02, Task 2 이득 없음 0.51 ± 0.02, (iii) **Uniform-Noise Sleep은 원래 구현과 유사(0.67 ± 0.05 / 0.69 ± 0.03)** → 입력의 세부 성질은 replay에 불필요하다는 저자들의 결론. Interleaved_{S,T2} 수치(0.70 ± 0.03 / 0.68 ± 0.05)와 강한 기준선 Interleaved_{T1,T2}(0.68 ± 0.03 / 0.65 ± 0.04)도 추가 | 원문 Results, S4 Fig |
| 12 | §3.3 | "wake update 수와 여러 control을 맞춘다" 삭제 — **그런 대조군은 논문에 없다**. 프로토콜은 100 movement cycle 대 100 movement cycle의 1:1 교대 단일 조건이다 | 원문 §Results, Methods |
| 13 | §4 | "여러 sleep duration, interleaving schedule, noise 조건을 비교해 이 주장을 뒷받침한다" 삭제 — **sweep이 존재하지 않는다**. 실제로 존재하는 대조(상위 1%/5%/10% synapse 동결: 0.54·0.65·0.70 대 0.68·0.61·0.53)로 교체하고, sleep 길이·비율·횟수 sweep 부재와 "한 번에 너무 오래 학습하면 회복 불가"라는 경계가 측정되지 않았다는 점을 명시. 근거 없던 homeostatic scaling 반박 문장 삭제 | 원문 §"Sleep replay protects…", S6 Fig |

### STC-T05 — Hippocampus–neocortex model (PNAS)

| # | 위치 | 고친 내용 | 근거 |
|---|---|---|---|
| 14 | §3.2 | "REM에서 coupling을 약하게 하고" → **내후각피질→해마 연결을 끊는(lesion) 완전 분리**. 두 stage 모두 C-HORSE projection은 비학습이고 sleep 중 갱신되는 것은 입출력 layer↔신피질 projection뿐이라는 조건 추가 | BioC Methods |
| 15 | §4 | 모호했던 "순서·비율 control"을 실제 조건으로 교체: NREM-only / NREM·REM 교대 / REM-only 3조건, 15블록 연속 NREM 대 15+15 교대(30블록) 비교, oscillation 완전 차단 및 stage별 차단 대조(SI Fig. S3–S4) | BioC Results, SI |

### STC-T07 — Deep Generative Replay

| # | 위치 | 고친 내용 | 근거 |
|---|---|---|---|
| 16 | §4.1 / §4.2 / §4.3 | §4.1의 실험 기구가 **permuted MNIST**임을 명시하고 4개 arm(GR/ER/Noise/None) 정의 추가. §4.1에 잘못 놓여 있던 LwF 비교를 **§4.2로 이동**(LwF는 SVHN→MNIST 실험에서만 비교된다). §4.3에 잘못 놓여 있던 "task identity 조건 구분"을 §4.2로 이동 — 원문에서 그것은 LwF가 task별 head를 쓰기 때문에 필요한 것이고, generative replay를 쓰는 scholar는 task context가 필요 없다는 대비다. GR이 ER을 능가한다고 주장하지 않고 "나쁘지 않다"고만 한다는 점, replay 없는 model이 새 task에서 오히려 약간 낫다는 점, §4.3의 분포 taxonomy(ER·GR은 입력+target, Noise는 target만, None은 둘 다 아님)와 disjoint 2-class × 5 분할 설정 추가 | 원문 §4.1–4.3, §5 |
| 17 | §5 | 원문에 없는 편집자 주장(generator의 privacy 미보장, 2017년 이후 난이도, "generative forgetting", sleep-time 위치 규정)이 **번역 본문처럼 섞여** 있었다. 이를 `[본서 주석]` 인용 블록으로 분리하고, 그 자리에 원문이 실제로 말하는 것(privacy는 DGR의 **동기**이며, SVHN class-incremental에서 성능 손실을 저자들이 스스로 보고하되 크기는 supplement로 넘겼고, EWC·LwF와 배타적이지 않다고 정리)을 넣음 | 원문 §1, §5 |

### STC-T09 — Zep

| # | 위치 | 고친 내용 | 근거 |
|---|---|---|---|
| 18 | §4.1 | Zep 94.8% / MemGPT 93.4%만 남기고 **가장 중요한 맥락이 통째로 빠져** 있었다. 추가: full-conversation baseline이 94.4%(gpt-4-turbo)·98.0%(gpt-4o-mini)로 MemGPT를 이미 넘어서므로 Zep의 우위는 0.4·0.2 point에 불과하다는 것, MemGPT 93.4%는 재측정이 아니라 인용값이고 저자들이 재현에 실패했다는 것, 신뢰구간·seed·유의성 검정이 없다는 것, 그리고 **저자들 스스로 이 benchmark를 부적절하다고 정리**한다는 것(60 메시지·단일 턴 사실 검색·모호한 문항·enterprise 대표성 부족) | 원문 §4.2, Table 1 |
| 19 | §4.2 | "18.5%"가 percentage point가 아니라 **상대 개선**임을 명시하고 절대치 병기(gpt-4o 60.2%→71.2%, +11.0 point; gpt-4o-mini 55.4%→63.8%, +8.4 point / 상대 15.2%), context 115K→1.6K, latency 28.9s→2.58s. **이 benchmark에는 MemGPT 수치가 아예 없다**는 사실(ingestion 우회 실패)과 질문 유형별 역전(single-session-assistant는 두 model 모두 하락) 추가 | 원문 §4.3, Tables 2–3 |

### STC-T14 — How much do language models memorize?

| # | 위치 | 고친 내용 | 근거 |
|---|---|---|---|
| 20 | §5.2 | 검증 대상이 GPT2-XL(약 1.56B)이고 예측이 실제 F1의 1.5 point 이내라는 사실, 그리고 논문의 실질적 배포 결론(**parameter당 token 10² 이상이면 membership inference 점수 0.5로 예측 = 이 정식화 안에서 loss 기반 공격 불성립**)을 추가. extraction rate가 dataset이 커지면 test extraction rate로 수렴해 성공한 추출이 전부 generalization으로 설명된다는 §4 결과도 추가 | 원문 §4, §5.2, Table 2 |

수치는 전부 정확했다(3.6 bits/parameter, grokking, 500K–1.5B, 수백 개 model — abstract 원문과 일치).

### STC-T15 — Understanding LoRA as Knowledge Memory

| # | 위치 | 고친 내용 | 근거 |
|---|---|---|---|
| 21 | §3 | **CounterFact 누락** 보완. Q1–Q3의 rank·용량·효율 sweep은 PhoneBook과 CounterFact 두 축에서 수행되는데 평가 설정 절에 CounterFact가 아예 없었다. PaperQA 구성(최근 학회 논문 15편, 450 QA pair, 오염 회피 목적)도 추가 | 원문 §3, §4 |
| 22 | §5 Q4–Q5 | "QA·summary·rewrite를 섞은 diverse curriculum이 **일관되게** 더 좋았다" → **model에 따라 갈렸다**. Llama에서는 전체 혼합이 최고지만 Qwen에서는 Summary+QA가 최고이고 전체 혼합은 근접 수준이며, Qwen에서 Original+QA는 QA 단독보다 낮다. format 순서(QA > Summary > Rewrite > Original)와 저자들이 QA 우위에 붙인 단서(평가가 QA 형식이라는 구조적 정렬) 추가 | 원문 §5 Q4–Q5, Table 1 |
| 23 | §6 Q9 | "ideal routing의 이득이 practical router에서 **크게 줄 수 있다**"는 과소 서술 → 실제로는 embedding router가 두 base model 모두에서 **단일 LoRA baseline보다도 낮았다**. 잘못 선택된 특화 module이 비특화 단일 module보다 해로울 수 있다는 저자들의 정리와, token 단위 router(Arrow·SpectR·LAG)도 embedding baseline을 일관되게 앞서지 못한다는 결과 추가 | 원문 §6 Q9, Table 6 |
| 24 | §6 Q10–Q11 | "merge 수를 늘릴수록 recall coverage와 interference가 **trade-off**를 이룬다"는 잘못된 형상 → **단조 감소**. Q11은 query가 실제로 유래한 ground-truth module만 골라 N=1→5로 늘린 실험이고, 성능은 N=1에서 최고이며 이후 계속 떨어진다(필요한 지식이 다 들어 있어도 희석·간섭이 일어난다). merge 연산자 6종 비교 결과(TIES 최고, Linear 의외로 강함, DARE 열위, CAT 붕괴의 원인은 rank 확장이 아니라 scale 불일치) 추가 | 원문 §6 Q10–Q11 |

Q14(top-3 > top-1)는 **원래 서술이 옳다** — Table 2의 closed-book 8개 셀 중 7개에서 top-3가 우세하다. Q11(단조 감소)과 Q14(top-3 우세)는 서로 다른 실험이며(전자는 정답 module만, 후자는 router가 고른 module) 모순이 아니다. Q15의 "30개 연속 질문"도 원문과 일치한다.

---

## 2. 용어 일관성 — 보고만 (고치지 않음)

의역본은 원문 용어를 따르는 것이 원칙이므로 v2 §2.4 확정 명칭을 강제로 밀어넣지 않았다. 아래는 **v2 본문이 이 의역본들을 인용할 때 용어를 갈아끼워야 하는 지점**의 목록이다.

| §2.4 확정 명칭 | 의역본의 표기 | 위치 | 판단 |
|---|---|---|---|
| consolidation | 기억 통합 / 기억 공고화 / 메모리 통합 (**세 표기가 T02 한 편 안에서 혼용**) | T02 다수(7·19·53·62·78·140·148·152·160·171·215·217·219·221·223·241·245·288·294·598·640·642행), T10 40·44·46·48행, T04 3행 | 원문 용어를 따른 것이지만 **한 편 안에서 세 갈래로 갈린 것은 의역본 자체의 결함**이다. v2 인용 시 반드시 통일 필요 |
| sleep-time compute | 수면 시(sleep-time) 공고화 | T02 223행 | 원문 `sleep-time consolidation`의 직역. 허용 범위 |
| learned context ($\hat c$) | 학습된 문맥(learned context) | T01 11행 | 첫 등장에 원어 병기 — §2.3 관행에 부합. 유지 가능 |
| external store ($E$) | 장기 기억 / 외부 기억 | T02 45·80·223행, T10 40·44·46·56·687행, T01 11행 | 원문이 `long-term memory`를 쓰는 대목이므로 의역본 쪽이 옳다. §2.4가 금지하는 것은 **v2 본문**에서의 사용이다 |
| slow/fast weights | 느린 가중치 / 빠른 가중치 | T10 106·168·201·288행 | 원문 `slow/fast weights`의 직역. v2 인용 시 원어로 되돌릴 것 |
| Θ-경로 / W-경로 / E-경로 | 모수적/비모수적, 파라메트릭/비파라미터 | T10 72·211·221·294·615·627·678·699·703·706·708·724·835·837·843행, T02 624·676행 | **여기서는 의역본이 옳다.** T10(Nested Learning)의 parametric/non-parametric은 회귀 해의 성질을 가리키는 논문 고유 개념이지 §2.4가 금지하는 경로 이분법이 아니다. v2 본문이 이를 경로 이분법으로 오독하지 않도록 주의 |

추가로 T10은 한 편 안에서 `비파라미터`와 `비모수적`을 혼용한다(211·221·294행 대 나머지). 원문은 단일 용어이므로 이것도 의역본 쪽 결함이다.

---

## 3. 누락 — 보고만 (전면 재번역 하지 않음)

### 3.1 절 단위 누락

**절 단위 누락은 없다.** 15편 모두 원문의 최상위 절 구조를 보존하고 있다. 확인 결과:

- T01: 원문 1–7 + 부록 A–M → 의역본 1–7 + 부록 안내 ✓
- T03: 원문 1–9(2.1–2.2, 3.1–3.4, 4.1–4.3, 5.1–5.3, 6.1–6.4, 7.1–7.3) → 의역본 1–9, 하위절은 산문으로 병합 ✓
- T05: 원문 Model Simulations / Discussion / Methods(5개 하위절) → 의역본 §6.1이 Wake Training+Testing을, §6.3이 Sleep Learning+Stage Protocol을 병합 ✓. 의역본의 §2·§4·§7은 원문에 없는 **합성 절**이다(paraphrase 모드에서는 허용되나, v2가 `[논문명 §N]` 형식으로 인용할 때 이 번호를 원문 절 번호로 쓰면 안 된다)
- T11: 의역본이 §5.5를 건너뛰고 5.4 → 5.6으로 가는 것은 **원문 자체가 그렇다**(원문도 5.4 다음이 5.6). 오류 아님

### 3.2 절 번호 신설 (인용 위험)

- **T11 §3.4** "체크포인트를 캐시할 것인가, 독립 압축기를 둘 것인가" — 내용은 원문에 실재하지만(단일 메모리의 checkpoint를 캐시할 것인가 세그먼트별 독립 메모리로 압축할 것인가), 원문에서는 **번호 없는 문단**이다. 원문 §3은 3.1–3.3에서 끝난다. v2가 `[Memory Caching §3.4]`로 인용하면 원문에 없는 번호를 가리키게 된다.
- **T05 §2·§4·§7**, **T12 §10·§16** 등도 같은 성격. §2.5("원문 수식 번호와 이 책의 번호를 절대 혼용하지 않는다")의 취지상, 의역본 절 번호는 원문 절 번호로 인용될 수 없다는 점을 v2 집필 규약에 못 박아야 한다.

### 3.3 절 안쪽 내용 누락 (수치·조건)

| 논문 | 빠진 것 | 왜 문제인가 |
|---|---|---|
| T04 | (고침 전) 수치 전무 | → **고침**. 이제 headline 수치가 들어 있다 |
| T06 | 수치 전무 — MNIST test error, interaction power별 결과, capacity 조건의 정량 형태가 하나도 없다 | v2가 "capacity가 $N$보다 빠르게 증가한다"를 인용할 때 근거 수치를 이 의역본에서 얻을 수 없다. **stc-v2에 T06 노트도 없다**(§4 참조) |
| T09 | (고침 전) 저자들의 benchmark 자기 부정 | → **고침** |
| T11 | LongBench 결과의 **과제별 역전** — 평균은 개선(Titans 19.53→19.81)이지만 TQA +23.5인 반면 TRC −22.3, MNs −8.7, GvR −2.1. 의역본은 "개선을 보인다"로만 요약 | 방향은 맞지만 분산이 숨겨진다. 원문 Table 4에 Avg 열이 없다는 사실도 함께 기록할 가치가 있다 |
| T12 | 수치 전무 (survey/formalism 논문이라 원문에도 표가 적음), reference experiment의 절대치 없음 | 원문 성격상 큰 문제 아님 |
| T13 | 수치 전무 — linear readout 정확도(CIFAR-10: PAD 58.25 대 w/o REM 46.00, w/o NREM 58.00, wake-only 42.25 / SVHN: 78.92 대 42.30, 73.25, 41.93)가 없다 | 의역본의 정성 서술("REM 제거 시 semantic 분리 약화", "NREM은 주로 robustness")은 이 표와 **일치한다**. 다만 v2가 "REM이 필수"의 크기를 인용하려면 원문 부록을 직접 봐야 한다 |
| T14 | (고침 전) 배포 결론과 extraction 수렴 | → **고침** |
| T01 | (고침 전) pass@k, 고budget 역전, o1, SWE 1.5× | → **고침** |

---

## 4. 노트(`notes/stc-v2/*.json`) 쪽 문제

**대조한 범위에서 노트의 사실 오류는 발견되지 않았다.** T01·T03·T04·T07·T09·T11·T14·T15의 `experiments` 필드를 원문과 교차 확인했고, 노트가 원문보다 정확한 경우가 여러 번 있었다(예: T04 노트는 원문 L305의 오식 `0.70 ± 002`를 잡아내 정정값을 병기하고 있다). **노트는 수정하지 않았다.**

다만 매핑 상 두 가지를 기록해 둔다.

1. **T05·T06에는 대응하는 stc-v2 노트가 없다.**
   - T05(PNAS, Singh–Norman–Schapiro, 해마–신피질 자율 상호작용)를 `scm-sleep-consolidated.json`과 짝지으면 안 된다. 그 노트는 arXiv 2604.20943 "SCM: Sleep-Consolidated Memory with Algorithmic Forgetting for LLMs"로 **완전히 다른 논문**이다.
   - T06(Dense Associative Memory, Krotov–Hopfield 1606.01164)에 대응하는 노트도 없다.
   - 따라서 이 두 편은 원문 직접 대조로만 감사했다(T05는 BioC XML, T06은 PDF).
2. 나머지 13편의 확정 매핑: T01→`letta-stc`, T02→`lm-need-sleep`, T03→`continual-facts-in-weights`, T04→`sleep-snn-plos`, T07→`deep-generative-replay`, T08→`memgpt`, T09→`zep`, T10→`nested-learning`, T11→`memory-caching`, T12→`rate-distortion-compaction`, T13→`pad-dreaming-elife`, T14→`lm-memorization-capacity`, T15→`lora-as-knowledge-memory`.
   - `need-sleep-recurrence.json`(2605.26099)은 T02와 **다른 논문**이다. 혼동 주의.
   - `pad-dreaming-elife.json`에는 `experiments` 필드가 없다(노트 자체가 3차 패스 보완본이며 실험 필드는 채워지지 않았다). T13 감사는 원문 직접 대조로 수행했다.

---

## 5. 권리 metadata

`manifest.json`이 `rights: public`으로 표시한 2편을 원문에서 직접 확인했다.

| 논문 | manifest | 원문 표기 | 판정 |
|---|---|---|---|
| **STC-T04** (PLOS Comput Biol 18(11):e1010628) | `public` / "CC BY 4.0; attribution retained" | 표지에 "This is an open access article distributed under the terms of the Creative Commons Attribution License, which permits unrestricted use, distribution, and reproduction in any medium, provided the original author and source are credited" | ✅ **CC BY 확인**. 단 PDF 본문에 **버전 번호 "4.0"이 인쇄되어 있지 않다**(PLOS 표준이 CC BY 4.0이라는 외부 지식에 의존). 의역본 §"데이터와 재현"이 "PLOS의 CC BY 4.0"이라고 단정하는 것도 같은 상태 |
| **STC-T13** (eLife 11:e76384) | `public` / "CC BY 4.0; attribution retained" | 표지에 "This article is distributed under the terms of the Creative Commons Attribution License, which permits unrestricted use and redistribution provided that the original author and source are credited" | ✅ **CC BY 확인**. 역시 **버전 번호 미인쇄** |

권고: `license_note`를 "CC BY (원문 표지 표기; 버전 번호는 PDF에 미인쇄, 발행처 표준은 4.0)"처럼 정확화하면 좋다. 다만 rights 값 `public` 자체는 두 편 모두 **타당**하다.

나머지 13편 검증:

- **STC-T05**: BioC XML에 `CC BY-NC-ND` / "Creative Commons Attribution-NonCommercial-NoDerivatives License 4.0 (CC BY-NC-ND)"가 명시되어 있다. manifest의 `internal-only` + "CC BY-NC-ND 4.0; Korean derivative is restricted to internal study"는 **정확하다**. 번역은 파생물(ND 금지 대상)이므로 internal-only가 맞는 판단이다.
- **arXiv 12편**(T01·T02·T03·T06·T07·T08·T09·T10·T11·T12·T14·T15): 12편 모두 PDF 본문에 CC 라이선스 표기가 **없다**(arXiv 라이선스는 abstract 페이지 metadata에 있고 PDF에는 찍히지 않는다). 따라서 보수적인 `internal-only`가 방어 가능하다. 다만 이 근거는 "확인 결과 non-CC"가 아니라 "**PDF만으로는 확인 불가**"라는 점을 분명히 해 둔다 — 공개가 필요해지면 arXiv abstract 페이지에서 편별 라이선스를 확인해야 한다.
- **STC-T15**의 venue는 원문 표지 각주 "Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026"과 일치한다. manifest의 "ICML 2026 / PMLR 306" ✅
- `reports/source-rights.json`의 15편 `public_release_allowed` 값은 manifest의 rights와 **모두 일치**한다(T04·T13만 true).

---

## 6. 남은 위험 (조치 필요, 이번 감사 범위 밖)

1. **재빌드 필요.** 고친 8편의 `body.md`가 `build/publications/translations-kr/*.pdf`에 반영되지 않았다. `reports/source-rights.json`과 각 편 `translation-qa.json`의 `output_sha256`·`sha256`은 **옛 PDF의 해시**이며 지금은 stale이다. `build_translations.py` 재실행 후 해시 갱신이 필요하다.
2. **`reports/reverse-check.md`가 무효다.** 15편 × 5 checkpoint 전부 PASS로 기록되어 있지만, T04는 primary-result checkpoint(`body.md:23`)가 가리키는 바로 그 절에서 논문 결론을 정반대로 서술하고 있었다. 이 파일은 v1 시기 산물로 보이며 재수행이 필요하다.
3. **편집자 주석의 미표시.** T07 §5은 이번에 `[본서 주석]`으로 분리했지만, 같은 성격의 미표시 편집 서술이 T01 §7, T06 §6, T08 §5, T09 §5, T12 §15, T13 §8, T15 §"한계와 향후 연구" 말미에도 있다(대체로 각 편 마지막 절에서 "이 논문은 sleep-time을 다루지 않는다", "system 문제로 남는다" 같은 v2 프레임 연결 문장). 코퍼스 차원의 규약이라면 `manifest.json`의 `translation_mode`에 명시하거나, 전 편에서 동일한 마커로 통일하는 편이 낫다. 현 상태로는 **독자가 원저자의 논의와 구분할 수 없다.**
4. **T02·T10은 "전문 무축약 번역"**을 표방하는데(다른 13편은 구조화 의역), 같은 `manifest.json`의 단일 `translation_mode` 아래 묶여 있다. `body_kind`가 구분하고는 있으나(`existing-full-v2-reconciled` / `existing-full`), 두 모드는 감사 기준이 달라야 한다.
5. **T16 동시 편집.** 다른 세션이 T16을 추가하며 `manifest.json`을 16편으로 갱신했다. 이 감사는 T16을 보지 않았다.
