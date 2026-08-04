# E07 · 논문·제품·negative evidence는 어디까지 왔는가

## 이 권이 답할 질문

Sleep-time compute가 학계·업계의 실제 방향인가, 몇 편의 논문을 큰 trend로 읽은 것인가? 답하려면 paper count가 아니라 evidence layer를 나눠야 한다: biological precursor, continual-learning algorithm, explicit parametric sleep, production external-memory synthesis, benchmark, negative evidence. 2026-08-05 freeze에서 가장 강한 production signal은 external-memory background synthesis이고, broad per-user foundation-weight sleep은 아직 연구 단계다.

이 권은 {{STUDY:STC-S09|evidence trajectory}}의 73개 고정 source audit를 설명한다. 중심 claim은 {{CLAIM:STC-C001}}, {{CLAIM:STC-C002}}, {{CLAIM:STC-C003}}, {{CLAIM:STC-C007}}, {{CLAIM:STC-C008}}, {{CLAIM:STC-C009}}, {{CLAIM:STC-C047}}이다.

## 먼저 세 문장

1. 논문은 “가능성”, 제품 문서는 “운영 의도”, 공개 code는 “구현”, 장기 telemetry는 “경제성”을 보여 준다. 이 네 evidence를 섞으면 성숙도를 과장한다.
2. OpenAI Dreaming, Mem0 Dream, Letta sleeptime은 응답 밖에서 외부 memory를 합성·정리하는 lifecycle의 production signal이다. 공개 자료는 이것을 per-user foundation-weight training이라고 말하지 않는다.
3. Parametric sleep은 Google 계열의 명시적 연구를 포함해 유망한 증거가 있지만, fleet-scale retention·rollback·delete·security가 동시에 검증된 공개 system은 확인되지 않았다.

## 직관

### Evidence ladder

1. **Analogy** — 뇌의 수면처럼 phase를 나누면 좋을 수 있다.
2. **Toy/mechanistic experiment** — 제한된 task에서 forgetting이 줄었다.
3. **Modern model experiment** — billion-scale checkpoint에서 improvement가 나왔다.
4. **Open implementation** — code와 config로 mechanism을 재현할 수 있다.
5. **Product behavior** — 실제 사용자 memory에 background job이 적용된다.
6. **Longitudinal operations** — 수개월 retention, cost, rollback, abuse가 측정된다.

현재 external-memory sleep은 4–5단계의 신호가 있고, parametric sleep은 주로 2–3단계다. 6단계의 공개 evidence는 부족하다.

### 날짜 freeze가 중요한 이유

이 분야는 paper와 product가 빠르게 바뀐다. “최신”이라는 표현은 반드시 cutoff와 version을 가져야 한다. 본 연구는 2026-08-05에 source URL, release, repository tag, paper version을 고정했다. {{BG:provenance|출처 추적성}}은 memory뿐 아니라 research claim에도 필요하다.

### Positive evidence와 negative evidence

Positive result만 모으면 trend가 언제나 강해 보인다. 반대로 capacity, model collapse, memory poisoning, serving reuse 같은 실패 근거를 함께 보면 조건부 verdict가 된다. DANN/SSRN record는 method와 result가 독립 검증에 충분하지 않아 load-bearing evidence에서 제외했다. 이것은 “틀렸다”는 판정이 아니라 evidence hygiene다.

> **주의.** 회사 이름이 paper에 등장한다는 사실과 회사의 production roadmap은 다르다. 공개 자료가 말한 medium과 scope를 넘어 추론하지 않는다.

## 예와 반례

### OpenAI Dreaming

공개 설명은 background synthesis가 user memory를 연결하고 더 유용한 derived memory를 만든다는 product signal이다. 중요한 한정은 external user memory라는 점이다. 공개 설명만으로 per-user foundation weight가 변경된다고 주장할 수 없다. 따라서 industry adoption evidence에는 포함하지만 parametric fleet evidence에는 포함하지 않는다.

### Mem0 Dream

Scheduled background synthesis와 provenance-preserving additive output이 문서·tagged code에서 확인된다. Merge와 supersede는 별도 ingest operation이다. 이 구분은 sleep output이 원 record를 파괴하는 in-place rewrite가 아니라 derived layer일 수 있음을 보여 준다.

### Letta sleeptime

Tagged group은 response 후 asynchronous agent가 durable memory를 수정하도록 schedule한다. 이 역시 “sleep”이라는 이름을 실제 orchestration에 연결한다. 하지만 동일한 label을 쓰는 모든 implementation이 같은 quality gate나 evidence semantics를 갖는 것은 아니다.

### 반례 — Benchmark 하나의 SOTA

특정 task에서 sleep method가 baseline을 이겼더라도 다른 retention stream, deletion, poisoning, cost에서 우월하다는 뜻은 아니다. {{BG:validation-set|검증 세트}}와 {{BG:test-set|테스트 세트}}를 반복 tuning에 혼용하면 성능도 과대평가된다.

### 반례 — 많은 citation = mainstream

Research attention은 adoption 가능성을 높이지만 infra deployment를 보장하지 않는다. Training job, user state version, compliance delete가 추가되는 system은 작은 accuracy gain만으로 도입되지 않는다.

## 그림 읽기

{{FIG:STC-F003}}

Timeline에서 점의 개수보다 layer 이동을 본다. 초기에는 memory theory와 forgetting algorithm이 중심이고, 2020년대에는 LLM external memory, neural memory, background product가 겹친다. “sleep”이라는 단어는 늦게 등장하지만 offline consolidation 기능은 오래전부터 있었다.

{{FIG:STC-F014}}

Landscape는 academia와 industry를 medium별로 나눈다. Industry의 밀도는 external memory·synthesis 쪽이 높고, academia는 parametric update·continual learning·capacity theory 쪽이 깊다. 두 집단의 교차점이 hybrid system과 benchmark다. 이것이 “lifecycle은 주류가 될 수 있지만 parametric sleep이 곧 주류라는 뜻은 아니다”라는 판단의 근거다.

## 대안과 비교

| evidence 종류 | 말할 수 있는 것 | 말할 수 없는 것 | 필요한 다음 증거 |
|---|---|---|---|
| biological model | phase separation mechanism | LLM fleet economics | modern-model ablation |
| continual-learning paper | retention under task stream | agent governance | long-lived agent benchmark |
| explicit sleep paper | method-level gain | production reliability | code, multi-cycle replication |
| product docs | deployed feature intent | internal weight update | architecture/telemetry disclosure |
| repository tag | implementation behavior | user-scale outcomes | longitudinal metrics |
| benchmark | common measurement | best architecture | multiple systems + cost |

{{BG:publication-gate|출판 게이트}}는 claim의 강도를 evidence ladder에 맞춘다. “보였다”, “제안했다”, “배포했다”, “장기적으로 검증했다”를 서로 바꾸지 않는 것이 핵심이다.

## 아직 모르는 것

- Product sleep이 실제 user utility, retention, cost에 미치는 longitudinal telemetry가 공개되지 않았다.
- Parametric sleep의 수백 cycle stability와 adapter generation 수가 보고되지 않았다.
- External synthesis의 잘못된 memory merge율과 human correction cost가 정량화되지 않았다.
- 회사별 “dreaming/synthesis/reflection”이 서로 다른 기능인데 공통 benchmark가 없다.
- Search saturation audit의 12 cluster 중 active frontier와 measurement gap은 future update가 필요하다.

> **핵심.** 2026년의 정직한 결론은 “background memory lifecycle은 이미 제품화 신호가 강하다. Broad parametric sleep은 promising research지만 mainstream production evidence는 아직 부족하다”다.

## 더 깊이 읽기

- 원 분석: {{STUDY:STC-S09|date-frozen evidence trajectory와 industry/academia 구분}}
- Training Background: {{BG:validation-set|검증 세트}}, {{BG:test-set|테스트 세트}}, {{BG:publication-gate|출판 게이트}}, {{BG:provenance|출처 추적성}}, {{BG:observability|관측 가능성}}
- Claim route: {{CLAIM:STC-C001}} · {{CLAIM:STC-C002}} · {{CLAIM:STC-C003}} · {{CLAIM:STC-C007}} · {{CLAIM:STC-C008}} · {{CLAIM:STC-C009}} · {{CLAIM:STC-C047}}
