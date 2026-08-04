# E11 · Memory device 회사에 생기는 기회

## 이 권이 답할 질문

Sleep-time compute가 확산되면 memory device solution 회사는 단순히 더 큰 HBM·SSD를 파는 것 외에 무엇을 할 수 있는가? 기회는 새 FLOP 수요보다 **derived memory lifecycle의 byte movement, random rewrite, snapshot, provenance, delete/rebuild**에서 더 선명하다. Device는 저장 매체와 software primitive를 함께 제안해야 한다.

이 권은 {{STUDY:STC-S14|device opportunity map}}을 설명하며 {{CLAIM:STC-C048}}, {{CLAIM:STC-C049}}, {{CLAIM:STC-C050}}, {{CLAIM:STC-C051}}, {{CLAIM:STC-C052}}, {{CLAIM:STC-C053}}, {{CLAIM:STC-C054}}, {{CLAIM:STC-C055}}, {{CLAIM:STC-C056}}, {{CLAIM:STC-C057}}을 제품 가설로 변환한다.

## 먼저 세 문장

1. Sleep workload는 sequential checkpoint뿐 아니라 vector/graph metadata의 작은 random rewrite와 versioned adapter read가 크다. Endurance·index locality·copy-on-write가 중요하다.
2. 가장 차별화 가능한 solution은 provenance-first store, snapshot/rollback, near-data preprocessing, adapter fabric, reusable-state cache, secure delete/rebuild다.
3. 평가 지표는 raw bandwidth가 아니라 **retained useful evidence, migration byte, write amplification, rollback time, wake-SLA interference**여야 한다.

## 직관

### Opportunity 1 — Provenance-first append-only store

Episode와 derived memory를 content-addressed object로 저장하고 “어느 evidence와 transform에서 만들어졌는가”를 edge로 남긴다. Append-only generation은 in-place corruption을 줄이고 rollback을 쉽게 한다. 장치는 small metadata lookup과 sequential object write를 함께 최적화할 수 있다.

### Opportunity 2 — Snapshot과 copy-on-write

Sleep job마다 full model·index·graph를 복제하면 byte가 폭증한다. Unchanged block을 공유하고 changed delta만 쓰는 copy-on-write snapshot은 {{BG:versioning|버전 관리}}과 {{BG:rollback|롤백}}을 싸게 만든다. Atomic root pointer switch는 publish primitive가 된다.

### Opportunity 3 — High-endurance indexed tier

Vector insert, graph edge update, supersede metadata, decay score는 작은 random write를 반복한다. HBM peak bandwidth보다 endurance, write amplification, tail latency가 중요하다. Log-structured layout, compaction offload, metadata cache가 solution이 될 수 있다.

### Opportunity 4 — Near-data preprocessing

Raw episode를 모두 GPU로 보내기 전에 dedup, filtering, tokenization, embedding candidate search, graph neighborhood gather를 storage 근처에서 수행한다. 이 기능은 expensive accelerator token과 interconnect byte를 줄인다. {{CLAIM:STC-C051}}은 near-data를 training accelerator 대체가 아니라 data reduction stage로 본다.

### Opportunity 5 — Versioned adapter fabric

{{BG:adapter|Adapter}}는 full model보다 작지만 user/domain 수가 많아지면 active set 관리가 어렵다. Adapter store가 version, tenant, base-model compatibility, hotness를 index하고 GPU cache에 prefetch·merge·evict하면 serving fragmentation을 줄일 수 있다.

### Opportunity 6 — Reusable-state cache

RNN/neural-memory model은 prefix를 compressed state로 materialize한다. Cache가 model version+prefix hash로 state를 재사용하면 full prefill을 피할 수 있다. Device는 KV뿐 아니라 dense recurrent state, optimizer-like accumulator, memory block snapshot을 지원해야 한다.

### Opportunity 7 — Secure delete-and-rebuild

Source episode가 삭제되면 summary, vector, graph, adapter까지 lineage를 따라 invalidate해야 한다. Device/firmware가 cryptographic erase, generation tombstone, fast rebuild scan을 제공하면 compliance SLA가 제품 차별점이 된다.

> **핵심.** Memory device 기회는 capacity 판매가 아니라 “기억 generation의 lifecycle transaction”을 빠르고 검증 가능하게 만드는 데 있다.

## 예와 반례

### Device solution package 예

1. **Evidence Log Tier** — immutable append, encryption, content hash
2. **Derived Index Tier** — high-endurance vector/graph update, compaction engine
3. **Adapter Vault** — versioned small tensor object, compatibility metadata
4. **Hot State Cache** — KV/recurrent/neural state unified caching
5. **Snapshot Engine** — COW generation, atomic publish, rollback
6. **Lineage/Deletion Engine** — provenance traversal, tombstone, rebuild queue
7. **Telemetry** — byte moved, WA, cache hit, rollback, staleness counters

이는 단일 chip보다 controller, firmware, runtime, reference architecture의 결합이다.

### On-device sleep 예

Smartphone이 charging·Wi-Fi·thermal 여유 상태에서 private episode를 local summary로 만들고, small personalization adapter를 업데이트한다. Raw data는 device를 떠나지 않는다. NPU/DRAM/flash 사이 이동, flash endurance, encrypted snapshot, 실패 시 rollback이 중요하다. 모든 task가 아니라 privacy 가치가 큰 개인 assistant에서 유망하다.

### 반례 — HBM 용량만 2배

Working set이 adapter·index·graph·episode로 섞여 있고 access가 sparse하면 HBM에 모두 올리는 것은 비싸다. Capacity가 늘어도 cold object, random write, version metadata 문제는 남는다. Tiering policy와 object semantics가 없으면 usable utility는 선형 증가하지 않는다.

### 반례 — GPU 근처에 모든 preprocessing 배치

Dedup할 raw episode를 먼저 GPU까지 옮기면 interconnect와 HBM bandwidth를 이미 썼다. Storage-side filter가 90%를 제거할 수 있다면 near-data가 더 합리적이다. 반대로 deep semantic evaluator는 GPU가 필요하므로 function placement를 workload별로 나눠야 한다.

## 그림 읽기

{{FIG:STC-F011}}

Roofline의 각 점은 다른 device 병목을 가진다. Teacher/student training은 compute-dense, graph maintenance는 bandwidth/random-write, snapshot은 sequential capacity, adapter serve는 latency/cache hit가 지배한다. “Sleep accelerator” 하나보다 heterogeneous tier를 연결하는 것이 중요하다.

{{FIG:STC-F012}}

Opportunity map은 가까운 기회와 먼 가설을 나눈다. Snapshot, provenance store, indexed tier, adapter cache는 기존 storage primitive의 확장으로 시작할 수 있다. Learned in-memory consolidation이나 autonomous on-device dreaming은 연구 위험이 더 높다. SAIT/메모리 팀은 가까운 primitive부터 trace와 benchmark를 확보할 수 있다.

## 대안과 비교

| solution | 고객 문제 | device primitive | KPI | first demo |
|---|---|---|---|---|
| provenance store | audit/rebuild | append+lineage index | lineage query latency | memory delete trace |
| COW snapshot | safe publish | block sharing+atomic root | snapshot byte/time | adapter generations |
| indexed endurance tier | graph/vector rewrite | log structure+compaction | WA/endurance | temporal KG stream |
| near-data preprocessing | state migration | filter/dedup/gather | bytes avoided/J | episode synthesis |
| adapter fabric | version explosion | object index+prefetch | hit/batching gain | multi-tenant LoRA |
| reusable-state cache | RNN prefix reuse | state key+cache | state hit/TTFT | HOPE/NSTM serving |
| delete/rebuild engine | compliance | tombstone+scan | delete SLA/leakage | derived-memory DAG |

{{BG:optimizer-state|옵티마이저 상태}}도 sleep checkpoint에서 weight보다 2배 이상 클 수 있어 sharded snapshot과 cold tier가 필요하다. 다만 inference-only adapter는 optimizer state를 publish하지 않아도 되므로 train-time object와 serve-time object를 분리해야 한다.

## 아직 모르는 것

- 실제 production trace에서 graph/vector/adaptor/episode byte 비율과 hotness distribution이 공개되지 않았다.
- Near-data semantic filtering의 accuracy loss와 accelerator byte 절감 tradeoff가 측정되지 않았다.
- Recurrent state cache가 어떤 granularity에서 KV cache보다 경제적인지 workload가 부족하다.
- Adapter merge가 quality와 batch efficiency에 미치는 영향이 model별로 다르다.
- Secure delete가 parametric artifact에 대해 “완료”되었다고 증명하는 방법이 없다.

> **시스템 모델링 관점.** Device roadmap은 speculative STC FLOP forecast보다, 오늘 구현 가능한 lifecycle trace를 수집하고 state-migration roofline을 만드는 것에서 시작해야 한다.

## 더 깊이 읽기

- 원 분석: {{STUDY:STC-S14|10개 memory-device opportunity와 maturity}}
- Training Background: {{BG:adapter|adapter}}, {{BG:optimizer-state|옵티마이저 상태}}, {{BG:sharding|샤딩}}, {{BG:versioning|버전 관리}}, {{BG:rollback|롤백}}, {{BG:provenance|출처 추적성}}
- Claim route: {{CLAIM:STC-C048}} · {{CLAIM:STC-C049}} · {{CLAIM:STC-C050}} · {{CLAIM:STC-C051}} · {{CLAIM:STC-C052}} · {{CLAIM:STC-C053}} · {{CLAIM:STC-C054}} · {{CLAIM:STC-C055}} · {{CLAIM:STC-C056}} · {{CLAIM:STC-C057}}
