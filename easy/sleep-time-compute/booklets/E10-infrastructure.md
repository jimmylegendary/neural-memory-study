# E10 · Wake·sleep·long-term memory infra는 어떻게 나뉘는가

## 이 권이 답할 질문

배포 후 학습이 wake와 sleep으로 나뉜다면 cluster와 data plane도 나뉘어야 하는가? 그렇다. 다만 세 개의 독립 silo를 만드는 것이 아니라 wake SLA, background training, durable memory가 서로 다른 최적점을 가지므로 **control plane과 versioned state contract**로 연결해야 한다. Infra 목표는 state movement를 줄이고, sleep output을 안전하게 publish하며, 외부 memory와 model delta를 같은 lineage로 관리하는 것이다.

이 권은 {{STUDY:STC-S13|system infrastructure}}를 설명한다. 근거는 {{CLAIM:STC-C025}}, {{CLAIM:STC-C040}}, {{CLAIM:STC-C046}}, {{CLAIM:STC-C053}}, {{CLAIM:STC-C055}}다.

## 먼저 세 문장

1. Wake cluster는 TTFT·ITL·availability를, sleep cluster는 throughput·sample efficiency·batching을 최적화한다. 같은 GPU를 써도 scheduler objective가 다르다.
2. Long-term memory fabric은 episode, vector, graph, recurrent state, adapter, optimizer snapshot을 tiering하고 provenance·version·delete lineage를 유지한다.
3. Wake와 sleep 사이의 핵심 interface는 raw log dump가 아니라 **immutable evidence ID + state generation + compatibility/version metadata**다.

## 직관

### Wake cluster

Wake path는 사용자가 기다린다. Model weights, active adapter, KV/recurrent state, hot retrieval index가 low-latency tier에 있어야 한다. Online update를 허용하면 inference만이 아니라 read-modify-write가 생긴다. 하지만 긴 backward나 validation을 response path에 넣으면 tail latency가 무너진다.

### Sleep cluster

Sleep path는 여러 user/session의 job을 batch하고 cheap/off-peak accelerator를 쓸 수 있다. Teacher rollout, embedding, graph compaction, LoRA training, regression evaluation이 섞인다. High throughput이 중요하지만 한 job의 잘못된 output이 많은 future response에 영향을 주므로 isolation과 reproducibility가 필요하다.

### Long-term memory fabric

Memory medium마다 access pattern이 다르다.

- Episode log: append-heavy, immutable, cold-to-warm scan
- Vector index: embedding read, incremental insert, compaction
- Temporal graph: small random read/write, lineage traversal
- Recurrent/neural state: dense tensor snapshot, prefix keying
- Adapter: versioned blob, read-mostly serving, occasional write
- Optimizer/checkpoint: large sequential write/read, sleep job scoped

하나의 HBM에 모두 넣는 것은 비경제적이다. Hot state만 accelerator near-memory에 두고, warm metadata·index와 cold evidence를 tiering해야 한다.

### Control plane

{{BG:versioning|버전 관리}}은 “model v3” 하나로 끝나지 않는다. Base model, tokenizer, adapter, memory schema, index build, graph snapshot, policy, evaluator version의 compatible tuple이 필요하다. {{BG:rollback|롤백}}은 이 tuple의 active pointer를 이전 generation으로 원자적으로 돌리는 일이다.

### Recurrent state cache

Full attention serving은 token prefix가 같으면 KV prefix cache를 재사용한다. RNN/HOPE/Titans-like model은 prefix를 압축한 state를 재사용해야 한다. {{CLAIM:STC-C025}}가 지적하듯 state를 materialize·key·share하지 않으면 architecture의 linear-time 장점이 serving에서 사라질 수 있다. Cache key에는 model/adaptor version, exact prefix hash, state schema가 들어가야 한다.

> **시스템 모델링 관점.** State migration byte가 compute FLOP보다 큰 구간이 있다. Sleep job이 adapter를 만들기 위해 remote episode와 checkpoint를 모두 GPU로 옮기면 near-data preprocessing의 가치가 커진다.

## 예와 반례

### 한 sleep generation의 publish protocol

1. Control plane이 evidence snapshot ID와 policy/evaluator version을 고정한다.
2. Job scheduler가 graph/index job과 GPU training job을 DAG로 실행한다.
3. Output을 new generation namespace에 쓰고 old generation을 건드리지 않는다.
4. Regression, deletion, poisoning, cost gate를 실행한다.
5. 통과하면 active pointer를 atomic switch한다.
6. Canary metric이 나빠지면 이전 pointer로 즉시 rollback한다.
7. Retention period 뒤에 unreachable generation을 garbage collect한다.

이 protocol은 database compaction과 model deployment를 결합한 형태다.

### Data movement를 줄이는 방법

- Episode와 graph close-to-storage에서 dedup·filter해 accelerator로 보내는 token을 줄인다.
- Base model은 sleep GPU에 상주시키고 user/domain adapter job만 이동한다.
- Adapter snapshot은 copy-on-write로 만들고 unchanged block을 공유한다.
- Recurrent state는 wake region에서 reusable cache로 materialize하고 remote migration을 최소화한다.
- Evaluation dataset과 teacher logits를 content-addressed cache로 재사용한다.

### 반례 — Wake와 sleep이 같은 mutable weight를 동시에 씀

Inference가 읽는 중 training이 같은 weight를 갱신하면 reproducibility와 batching이 깨진다. Mutable in-place update 대신 generation을 분리하고 publish boundary에서 pointer를 바꿔야 한다.

### 반례 — 모든 user state를 HBM에 상주

Active user 분포는 long tail이다. HBM capacity를 늘려도 cold state가 hot state를 밀어내면 효율이 낮다. Access frequency, state size, regeneration cost로 tier를 정해야 한다.

## 그림 읽기

{{FIG:STC-F006}}

Architecture의 중심은 external source-of-record와 selective promotion이다. Wake cluster는 derived state를 읽고, sleep cluster는 evidence에서 새 derived state를 만든다. 화살표는 one-way copy가 아니라 provenance link와 invalidation path를 뜻한다.

{{FIG:STC-F011}}

State-migration roofline은 x축 byte movement와 y축 compute를 놓는다. Graph/index compaction은 bandwidth·random write bound, adapter training은 compute/HBM bound, checkpoint publish는 sequential bandwidth bound일 수 있다. Device solution은 workload를 한 숫자 “AI bandwidth”로 묶지 말고 phase별 operational intensity를 봐야 한다.

{{FIG:STC-F013}}

Failure control은 control plane의 요구사항이다. Poisoned evidence를 걸러도 derived adapter가 이미 publish됐다면 lineage를 따라 invalidation하고 rebuild해야 한다. Observability에는 answer metric뿐 아니라 state generation, evidence coverage, rollback latency가 포함된다.

## 대안과 비교

| plane | SLO | hot state | 주 병목 | 필수 primitive |
|---|---|---|---|---|
| wake inference | TTFT/ITL, availability | weights, KV/state, hot index | HBM/BW, batching | cache, routing |
| wake learning | bounded latency | fast state, accumulator | RMW, interference | isolated delta |
| sleep compute | throughput, utility | training activations, batch data | GPU compute/data feed | DAG, checkpoint |
| memory fabric | durability, lookup | tier별 working set | random write/migration | index, tiering |
| control plane | correctness | metadata/lineage | version explosion | atomic publish/rollback |

{{BG:sharding|샤딩}}은 user나 entity를 분산하지만 cross-user/domain consolidation이 필요할 때 shuffle이 생긴다. Reuse와 privacy boundary가 같은 shard key가 되도록 설계하면 이동을 줄일 수 있다.

## 아직 모르는 것

- Wake/sleep cluster를 물리 분리할지 elastic shared pool로 운영할지 workload evidence가 부족하다.
- Recurrent state cache의 real trace hit ratio와 eviction policy가 공개되지 않았다.
- Adapter·graph·vector를 하나의 transaction으로 publish하는 표준이 없다.
- Sleep job의 queue delay와 memory staleness가 user utility에 미치는 관계가 측정되지 않았다.
- Delete request가 derived weight까지 전파되는 rebuild SLA와 비용이 불명확하다.

> **핵심.** STC infra의 핵심 제품은 “밤에 쓰는 GPU”가 아니라 evidence에서 derived state를 만들고, 최소 이동으로 검증·publish·rollback하는 lifecycle fabric이다.

## 더 깊이 읽기

- 원 분석: {{STUDY:STC-S13|wake/sleep/memory/control plane architecture}}
- Training Background: {{BG:sharding|샤딩}}, {{BG:versioning|버전 관리}}, {{BG:rollback|롤백}}, {{BG:memory-routing|기억 라우팅}}, {{BG:observability|관측 가능성}}, {{BG:serving-artifact|서빙 산출물}}
- Claim route: {{CLAIM:STC-C025}} · {{CLAIM:STC-C040}} · {{CLAIM:STC-C046}} · {{CLAIM:STC-C053}} · {{CLAIM:STC-C055}}
