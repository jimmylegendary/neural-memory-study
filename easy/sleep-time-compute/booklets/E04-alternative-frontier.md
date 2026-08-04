# E04 · Sleep 말고 가능한 최신 대안들

## 이 권이 답할 질문

장기 기억과 continual adaptation을 해결하려면 정말 sleep-time compute가 최선인가? 비교 대상은 “아무것도 안 함”이 아니라 2026년 현재 강한 대안들이다: long context와 retrieval, temporal graph, recurrent state, model editing, LoRA/adapter, latent compilation, self-adaptation. Sleep은 이들을 대체하는 새 layer라기보다, 여러 대안을 **언제 실행하고 어떤 state로 승격할지** 정하는 orchestration 축에 가깝다.

이 권은 {{STUDY:STC-S06|alternative frontier}}를 바탕으로 {{CLAIM:STC-C026}}, {{CLAIM:STC-C027}}, {{CLAIM:STC-C028}}, {{CLAIM:STC-C029}}, {{CLAIM:STC-C030}}, {{CLAIM:STC-C031}}을 비교한다.

## 먼저 세 문장

1. 사실의 최신성·출처·삭제가 중요하면 {{BG:external-memory|외부 기억}}과 retrieval이 기본값이다. 이 문제를 weight update로 풀 이유가 약하다.
2. 반복 사용되는 stable skill이나 style처럼 retrieval overhead가 계속 발생하면 {{BG:lora|LoRA}}·{{BG:adapter|adapter}}·latent compilation으로 선택적 parametric promotion을 고려할 수 있다.
3. Sleep의 경쟁력은 한 method의 peak accuracy보다, evidence를 accumulate해 **admission→compaction→evaluation→versioning**을 응답 밖에서 운영할 수 있다는 데 있다.

## 직관

### 대안 1 — Long context와 RAG

RAG는 mutable knowledge를 model 밖에 두고 필요한 passage만 context에 넣는다. 원문을 citation할 수 있고 update·delete가 쉽다. 약점은 retrieval miss, context token, index maintenance다. 하지만 factual memory에 대한 governance baseline으로 매우 강하다. Sleep이 RAG보다 낫다고 주장하려면 future query에서 반복 retrieval하는 비용보다 background synthesis 비용이 작고, 정확도·삭제 가능성도 유지되어야 한다.

### 대안 2 — Temporal graph

Entity와 relation에 validity interval을 붙이면 “A가 B였다가 C가 됨”을 표현할 수 있다. Vector similarity만으로 해결하기 어려운 contradiction과 temporal query에 강하다. Graph maintenance 자체가 background compute일 수 있지만 foundation weight를 바꾸지 않는다. 즉 external-memory STC의 대표 medium이다.

### 대안 3 — Recurrent/learned state

RNN, state-space model, Titans-like neural memory는 긴 prefix를 fixed-size 또는 learned state로 압축한다. 매 token의 context 재읽기를 줄일 수 있지만 state가 어떤 prefix에서 만들어졌는지 keying·sharing·cache invalidation이 필요하다. 이 문제는 model architecture와 serving infra를 함께 요구한다.

### 대안 4 — Model editing

{{BG:model-editing|모델 편집}}은 특정 factual association을 localization해 바꾼다. ROME은 targeted edit, MEMIT은 여러 memory의 batch edit 가능성을 보였다. 그러나 bounded edit 성공은 lifelong sequence에서 locality·retention·delete가 보장된다는 뜻이 아니다. “한 번 잘 고친다”와 “영원히 안전하게 축적한다”는 다른 문제다.

### 대안 5 — Adapter와 latent compilation

LoRA는 weight delta를 low-rank로 제한해 trainable parameter와 optimizer state를 줄인다. Generative Adapter처럼 context를 한 번의 forward로 parameter-like state로 compile하는 접근은 iterative gradient training을 피할 수 있다. 공통 관건은 compile cost가 later reuse로 상각되는지, version과 tenant별 delta를 어떻게 관리하는지다.

### 대안 6 — Self-adaptation

Self-Adapting Language Models 계열은 model이 adaptation data와 update rule을 스스로 생성한다. 이는 wake와 sleep 사이를 잇지만, synthetic data quality와 recursive error가 핵심 risk다. 좋은 data generation policy를 학습하는 것과 generated data가 truth를 보존하는 것은 별개다.

## 예와 반례

### 예 — 회사 규정 FAQ

규정은 출처·effective date·삭제가 중요하므로 temporal document store+RAG가 우선이다. 질문 빈도가 높은 stable procedure는 작은 domain adapter로 promotion할 수 있다. 매일 바뀌는 숫자까지 adapter에 넣으면 stale weight가 된다. Sleep job은 모든 정보를 weight로 옮기는 것이 아니라 promotion 후보를 평가한다.

### 예 — 개인 글쓰기 style

Style은 하나의 원문 passage보다 많은 episode의 공통 pattern이다. Retrieval로 매번 예시를 넣을 수 있지만 token cost가 반복된다. 충분한 consent와 검증이 있다면 user adapter가 효율적일 수 있다. 그러나 잘못 학습된 style은 모든 response에 확산되므로 versioned rollback이 필수다.

### 반례 — “Weight면 retrieval이 필요 없어 항상 빠르다”

Adapter routing, loading, batching fragmentation, cache invalidation까지 포함하면 per-user weight는 공짜가 아니다. 수백만 user의 sparse delta는 storage·metadata·activation 문제를 만든다. 또한 source citation이 필요하면 결국 external store를 유지해야 한다.

### 반례 — “External memory면 안전하다”

Poisoned record가 retrieval되거나, summary가 잘못 merge되거나, delete가 derived graph까지 전파되지 않으면 external memory도 위험하다. 외부라는 사실은 observability 가능성을 높일 뿐, 올바른 policy를 자동 제공하지 않는다.

## 그림 읽기

{{FIG:STC-F004}}

문제–대안 matrix에서 하나의 열이 모든 행을 지배하지 않는다. RAG는 provenance, adapter는 repeated access, graph는 temporal conflict, recurrent state는 sequence compression에서 강하다. 비교의 단위는 accuracy 하나가 아니라 update cost, read latency, interference, delete, rollback, reuse다.

{{FIG:STC-F006}}

Hybrid architecture는 external system-of-record를 아래에 두고, 검증된 high-reuse knowledge만 reversible adapter나 cache로 promotion한다. 이 그림에서 sleep cluster는 source-of-truth가 아니다. evidence를 읽고 derived state를 생성하는 compiler다. 따라서 derived artifact는 언제든 원 evidence에서 rebuild 가능해야 한다.

## 대안과 비교

| 접근 | 가장 잘 푸는 문제 | reusable state | sleep이 추가하는 가치 | 결정적 약점 |
|---|---|---|---|---|
| long context | 한 번의 깊은 이해 | 현재 context | 사전 selection·summary | 매 요청 재독해 |
| RAG/vector | factual lookup | document/index | dedup·index rebuild | retrieval miss |
| temporal KG | 변화·관계 | graph | merge·supersede | graph maintenance |
| recurrent state | prefix compression | hidden/neural state | snapshot·cache | state reuse/keying |
| editing | targeted correction | base weight | batch evaluation | lifelong locality |
| LoRA/adapter | stable repeated behavior | delta weight | promotion·version | fragmentation/delete |
| latent compiler | fast parameterization | generated delta | batch compilation | fidelity·amortization |

가장 합리적인 선택 순서는 “외부에 남겨도 되는가?”부터 묻는 것이다. 예라면 external memory가 governance 기준선이다. 이후 reuse가 높고 stable하며 latency 가치가 큰 subset만 parametric state로 올린다. 이 순서를 뒤집으면 모든 transient fact가 weight에 들어가 capacity와 deletion debt를 만든다.

## 아직 모르는 것

- Retrieval token을 adapter로 compile할 때 정확히 몇 회의 reuse에서 break-even인지 workload별 측정이 부족하다.
- Generative Adapter 같은 one-pass compilation이 iterative fine-tuning의 retention과 locality를 어디까지 따라가는지 대규모 evidence가 없다.
- Recurrent state cache의 cross-request reuse율과 invalidation cost가 KV prefix cache와 공정하게 비교되지 않았다.
- Editing·LoRA·external memory를 동일한 continual stream, delete request, poisoning attack에서 비교한 benchmark가 부족하다.
- Self-generated adaptation data의 quality를 external ground truth 없이 평가하는 방법이 미해결이다.

> **핵심.** Sleep은 “RAG 다음의 새 memory”가 아니다. RAG·graph·adapter·state를 어떤 cadence와 evidence gate로 유지할지 결정하는 lifecycle layer다.

## 더 깊이 읽기

- 원 분석: {{STUDY:STC-S06|최신 대안 frontier와 선택 조건}}
- Training Background: {{BG:external-memory|외부 기억}}, {{BG:lora|LoRA}}, {{BG:adapter|adapter}}, {{BG:model-editing|모델 편집}}, {{BG:test-time-training|test-time training}}, {{BG:synthetic-data|합성 데이터}}
- Claim route: {{CLAIM:STC-C026}} · {{CLAIM:STC-C027}} · {{CLAIM:STC-C028}} · {{CLAIM:STC-C029}} · {{CLAIM:STC-C030}} · {{CLAIM:STC-C031}}
