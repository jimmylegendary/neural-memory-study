# Sleep-Time Compute Question Tree

- 기준일: 2026-08-05
- root question: **어떤 workload와 제약에서 deferred consolidation이 가장 강한 대안보다 더 높은 lifetime utility를 만드는가?**
- 판정 단위: 논문 이름이 아니라 `problem × operator × memory medium × cadence × governance` 조합

## Q0. Root decision

```text
Q0 STC가 독립 연구·인프라 축인가?
├─ Q1 풀려는 문제가 실제로 중요한가?
├─ Q2 deferred phase가 필요한가?
├─ Q3 어떤 state를 어떻게 바꾸는가?
├─ Q4 가장 강한 대안을 이기는가?
├─ Q5 lifetime에 걸쳐 포화·망각·오염을 관리하는가?
├─ Q6 안전하게 운영·삭제·rollback할 수 있는가?
├─ Q7 규모가 커질 때 어떤 law와 system bottleneck이 나타나는가?
└─ Q8 device primitive로 환원되는가?
```

## Q1. 문제 정의

### Q1.1 Long-context comprehension

- 더 많은 token을 볼 수 있다는 것과 필요한 evidence를 이해·조합하는 것은 어떻게 다른가?
- full attention, recurrent/state-space, neural memory, retrieval가 실패하는 regime은 무엇인가?
- sleep이 summary나 latent skill을 만들 때 information loss와 query uncertainty를 어떻게 측정하는가?

### Q1.2 Test-time scaling and reuse

- 같은 사용자·agent·environment에서 query family가 반복되는가?
- sleep precomputation이 future test-time FLOPs를 얼마나 대체하는가?
- one-shot query라면 deferred compute는 낭비가 되는가?

### Q1.3 Continual adaptation

- 새로운 fact, preference, skill, policy 중 무엇을 장기 state에 기록해야 하는가?
- catastrophic forgetting, loss of plasticity, model drift를 분리해 측정하는가?
- feedback가 noisy·adversarial·nonstationary일 때 consolidation은 오염을 증폭하는가?

### Q1.4 Serving efficiency

- recurrent/neural-memory state를 prefix/KV cache처럼 공유할 수 있는가?
- session-specific mutable state가 batching과 replica consistency를 어떻게 깨뜨리는가?
- prefill, decode, background update의 자원 경쟁을 어떻게 격리하는가?

## Q2. 왜 deferred phase인가

### Q2.1 Wake path에서 불가능하거나 비싼 연산

- multi-episode deduplication, graph/community rebuild, global coreset selection
- generated replay, distillation, backward/optimizer, multi-seed evaluation
- delete fan-out, conflict resolution, provenance rebuild, canary/rollback preparation

### Q2.2 Wake에서 해야 하는 연산

- raw event admission, immediate safety filter, exact timestamp/source capture
- latency-bound retrieval, ephemeral fast state, user-visible correction
- snapshot boundary와 isolation metadata 생성

### Q2.3 Periodic refresh와의 경계

- consolidation이 user-local이면 global refresh보다 privacy와 freshness 이점이 있는가?
- common knowledge라면 tenant별 sleep이 중복 compute가 되는가?
- delta publish가 model residency와 fleet batching을 깨뜨리는가?

## Q3. Memory medium과 operator

### Q3.1 External exact memory

- text/event: provenance·delete가 쉽지만 token cost와 retrieval miss가 있다.
- vector: approximate similarity가 강하지만 exact temporal/causal structure를 잃는다.
- graph: relation·temporality·contradiction에 유리하지만 entity resolution과 compaction cost가 든다.

### Q3.2 Latent and recurrent memory

- fixed state에서 compression ratio, interference, reset boundary를 어떻게 측정하는가?
- learned write rule이 foreground TTT인지 background consolidation인지 구분되는가?
- latent state가 source item으로 역추적·삭제될 수 있는가?

### Q3.3 Parametric memory

- full fine-tuning, LoRA/adapter, sparse memory/expert, model editing 중 어느 granularity가 필요한가?
- exact fact를 weight에 쓰는 것이 composition·latency를 개선하는가?
- canonical copy, provenance, rollback 없이 weight를 system of record로 쓸 수 있는가?

### Q3.4 Policy and skill memory

- tool procedure, code patch, routing rule, reward policy는 fact memory와 다른 평가가 필요한가?
- executable memory의 verification·sandbox·rollback cost를 scaling law에 포함하는가?

## Q4. Strongest alternative gate

각 target problem은 다음 alternative 중 가장 강한 하나 이상과 비교해야 한다.

| problem | mandatory comparator |
|---|---|
| exact durable fact | external canonical store + retrieval |
| repeated low-latency fact | cache + external fallback |
| long-document repeated queries | offline index/summary + RAG, latent cartridge |
| user preference | profile store + prompt injection, adapter |
| continual skill | replay, regularization, modular expert, periodic refresh |
| current-distribution adaptation | TTT/online learning, calibration, retrieval |
| temporal relation | bi-temporal graph + invalidation |
| context overflow | compression, recurrent state, cache lifecycle |

비교 budget은 model, wake tokens, sleep FLOPs, storage bytes, accelerator time, network bytes, validation, delete/rollback을 함께 맞춘다.

## Q5. Capacity, forgetting, and compaction

### Q5.1 다섯 capacity

1. physical bytes/parameters/slots
2. functional retrievable information
3. interference-free plastic subspace
4. serving context and latency budget
5. governance capacity: lineage, deletion, versions, validation throughput

### Q5.2 Saturation questions

- monotonic append가 아니라 admission, merge, abstraction, eviction, decay를 누가 결정하는가?
- replay buffer가 줄면 generator state나 teacher corruption이 새로운 capacity가 되는가?
- LoRA/expert를 늘리면 router error와 model residency가 먼저 포화되는가?
- summary/graph compaction이 rare but important detail을 언제 파괴하는가?
- delete가 derived summary, embedding, graph, adapter, checkpoint에 전파되는가?

### Q5.3 Required lifetime curves

```text
retention(age, value, frequency)
plasticity(cumulative updates)
utility(memory budget)
interference(load factor)
delete completeness(derived-state fan-out)
recovery time(version count, state size)
```

## Q6. Training and data

### Q6.1 Sleep data construction

- raw replay, coreset, prioritized replay, synthetic/dream data, counterfactual, negative evidence
- teacher/student distillation target와 canonical anchor
- privacy filter, poison detection, temporal weighting, user consent

### Q6.2 Objective

- supervised SFT/continued pretraining
- forward/reverse KL distillation
- retention regularization, EWC/SI, gradient projection
- RL for admission/scheduling/routing
- bilevel objective: later wake utility를 sleep policy가 최적화

### Q6.3 Trainable state and publication

- shared weight, tenant adapter, sparse expert, latent memory, graph compactor 중 어느 것을 업데이트하는가?
- candidate 생성과 serving publish를 분리하는가?
- offline validation이 future query distribution을 대표하는가?

## Q7. Mainstream and promisingness

### Q7.1 Evidence signals

- independent reproduction, long-horizon benchmark, code/data, matched baseline
- product lifecycle에서 async transform과 later-wake consumption 확인
- training/serving cost와 governance failure 공개

### Q7.2 False signals

- 이름에 sleep/dream이 포함됨
- 단일 benchmark에서 vendor baseline을 이김
- context update를 durable lifelong learning으로 표현
- repository option이 production adoption을 뜻함

### Q7.3 Conditional scenarios

- broad STC axis: multiple foundation-model vendors expose durable deferred training APIs
- hybrid mainstream: external system of record + selective latent/parametric compilation
- external-memory dominant: search/graph/cache improve faster than safe weight writing
- niche STC: private/on-device/robotics/long-lived agent에 한정
- refresh dominant: shared post-training pipeline이 personalization보다 경제적

## Q8. Scaling and infrastructure

### Q8.1 Candidate independent variables

- accumulated wake evidence `D_w`
- sleep compute `C_s`
- durable capacity by tier `M_e, M_l, M_p`
- query reuse and predictability `R_q`
- state age/staleness `A`
- interference/plasticity load `I`
- verification/governance budget `V`
- bytes moved `B_move`

### Q8.2 Dependent variables

- later-wake loss/utility, exact recall, composition, latency
- retention, plasticity, calibration, hallucination, contamination
- amortized cost/query, energy, publish delay, recovery/delete SLA

### Q8.3 Device questions

- append-heavy raw evidence와 rewrite-heavy compaction이 어떤 endurance·QoS를 요구하는가?
- source, embedding, graph, optimizer/checkpoint state의 locality를 어떻게 보존하는가?
- HBM–CXL–DRAM–NVMe tier 사이 bytes moved가 sleep FLOPs보다 먼저 병목이 되는가?
- snapshot/COW, content-addressable version, delete proof, rollback을 device/runtime가 지원할 수 있는가?
- sparse adapter/expert와 cache state를 빠르게 activate/offload하는 최소 granularity는 무엇인가?

## Closure rule

각 leaf question은 `answered`, `bounded`, `experiment-required`, `production-trace-required`, `not-answerable-at-freeze` 중 하나로 닫는다. `not-answerable`은 실패가 아니라 과장된 결론을 막는 정식 결과다.
