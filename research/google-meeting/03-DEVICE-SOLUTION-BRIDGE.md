# Google 미팅용 Device/System Solution Bridge

- 작성 목적: HOPE, *Language Models Need Sleep*, *Memory Caching*, NSTM을 출발점으로 Google의 알고리즘 설명을 SAIT의 memory-device·serving-system 공동 연구 의제로 연결한다.
- 기준일: 2026-07-27
- 대상: Samsung SAIT 연구팀 ↔ Google 연구팀 기술 미팅
- 문서 성격: 미팅용 질문·가설·실험 설계안이다. 제품 로드맵이나 논문에 이미 존재하는 구현을 서술하는 문서가 아니다.

> **패키지 역할:** [① evidence pre-read / problem map](01-SIX-AXIS-PROBLEM-MAP.md) · [② canonical facilitator playbook](02-GOOGLE-KICK-QUESTION-PLAYBOOK.md) · ③ device/system appendix

## 0. 읽는 법과 증거 경계

이 문서는 다음 표식을 엄격히 구분한다.

- **[PAPER]** 원 논문이 직접 제안하거나 실험한 내용이다.
- **[INFERENCE]** 여러 논문의 결과로부터 도출한 시스템적 해석이며, 원 논문의 직접 주장이 아니다.
- **[SAIT-PROPOSAL]** 미팅에서 제안할 장치·시스템 공동 연구 가설이다.
- **[OPEN]** 공개 논문만으로는 답할 수 없어 Google에 확인해야 하는 항목이다.

특히 다음을 과장하지 않는다.

1. HOPE의 다중 update frequency는 production sleep scheduler의 구현 증거가 아니다.
2. *Language Models Need Sleep*의 실험은 parametric consolidation의 proof-of-concept이지, 무한 수명 용량·삭제·rollback·production cluster를 입증하지 않는다.
3. *Memory Caching*의 cache는 한 sequence 안에서 recurrent hidden-state checkpoint를 늘리는 모델 구조다. cross-request prefix cache나 lifetime memory store를 구현한 것이 아니다.
4. NSTM의 “Memory Caching”은 과거 fast-weight의 running average를 이용한 drift regularization이다. 위 *Memory Caching* 논문의 checkpoint retrieval cache와 다른 개념이다.
5. 아래의 HBM, near-memory, CXL/DDR, NVMe, vector/graph store 배치는 모두 공동 검증이 필요한 **SAIT proposal**이다.

## 1. 미팅에서 공유할 한 문장

> 장기 문맥을 더 잘 이해하는 모델은 결국 여러 형태의 state를 만들고 갱신한다. 다음 병목은 state를 만들 수 있느냐가 아니라, **각 state의 소유권·update cadence·재사용 단위·용량 상한·promotion/rollback 계약을 어떤 memory hierarchy와 serving system 위에 올릴 것인가**다.

Google 논문은 네 개의 중요한 알고리즘 단서를 제공한다.

| 논문 | [PAPER] 직접 보인 것 | 공개 논문에서 남은 시스템 공백 |
|---|---|---|
| [Nested Learning / HOPE](https://arxiv.org/abs/2512.24695) | optimizer를 associative memory로 재해석하고, self-modifying module과 서로 다른 update frequency의 CMS를 결합 | request ownership, batch serving, HBM traffic, lifetime cap, cross-session publication, deletion/rollback |
| [Language Models Need Sleep](https://arxiv.org/abs/2606.03979v2) | fast/high-frequency knowledge를 새 low-rank expert에 distill하는 Knowledge Seeding, RL 기반 synthetic curriculum을 만드는 Dreaming | expert pool exhaustion, merge/evict/recycle, external-memory 선택, production scheduler, provenance/rollback |
| [Memory Caching](https://arxiv.org/abs/2602.24281) | recurrent state checkpoint를 cache하고 aggregate해 fixed-state RNN과 growing-memory Transformer 사이의 capacity/complexity trade-off를 조절 | cross-request reuse, cache lifecycle, HBM residency, lifetime growth, serving admission/eviction |
| [Online Neural Space Time Memory, NSTM](https://arxiv.org/abs/2607.15271) | expensive memory update와 per-frame apply frequency를 분리하고, periodic update 및 historical-state regularization으로 online stability를 개선 | language serving으로의 일반화, multi-tenant ownership, durable consolidation, sleep publication, finite-capacity lifecycle |

비-Google 시스템 연구는 이 공백이 실제 serving 이슈임을 보강한다.

- [Marconi](https://arxiv.org/abs/2411.19379): recurrent state는 partial-prefix rollback이 어려워 exact-match 위주가 되고, reuse likelihood와 saved-FLOPs/byte를 고려한 admission/eviction이 필요하다. 보고된 수치는 workload 의존적이다.
- [RW-TTT](https://arxiv.org/abs/2605.28053): request-owned mutable state는 shared-static-weight batching을 깨뜨리며, owner/version/READ-WRITE contract가 batching 복원의 한 방법이다.
- [KVBuffer](https://arxiv.org/abs/2605.19049): 큰 recurrent state를 매 decode step read/write하는 대신 KV를 buffer하고 update를 chunkwise로 미루면 memory traffic을 줄일 수 있다.
- [HYPIC](https://arxiv.org/abs/2607.01299): segment transition operator와 zero-start state를 cache하면 independent segment state를 near-exact하게 합성할 수 있지만, 별도 metadata/state와 seam recomputation이 필요하다.

## 2. 여섯 개 연구축을 장치 문제로 번역하기

| 축 | 모델 측 핵심 질문 | device/system으로 번역한 질문 | 우선 계측값 |
|---|---|---|---|
| 1. long context와 test-time scaling | 전체 context를 실제로 이해하는가, 아니면 제한된 state에 과압축하는가? | exact KV, recurrent compressed state, checkpoint cache를 어떤 비율로 둘 것인가? | quality-vs-context, bytes/token, cache hit, retrieved-state utility |
| 2. recurrent state 재사용 | KV prefix cache만큼 state를 재사용할 수 있는가? | checkpoint/segment/state-transition을 어떤 key와 version으로 cache할 것인가? | exact/partial hit, saved FLOPs/byte, recompute suffix, state transfer bytes |
| 3. pretraining·prefill·decode 효율 | chunk와 update cadence를 어떻게 정할 것인가? | apply와 update를 분리해 어느 것은 HBM-resident, 어느 것은 batch writeback할 것인가? | HBM read/write, dirty bytes, update/apply ratio, utilization, TTFT/TPOT |
| 4. continual learning | fast state를 언제 durable knowledge로 바꿀 것인가? | external text/vector/graph와 adapter/expert/weight 중 destination을 어떻게 고를 것인가? | reuse, volatility, interference, build/serve/govern cost |
| 5. 유한 용량 | cache와 expert가 단조 증가할 때 무엇을 버리거나 합칠 것인가? | hot HBM, warm memory, cold store, recovery version을 하나의 cap 아래 어떻게 관리할 것인가? | retained/peak bytes, occupancy, eviction loss, rebuild cost, delete debt |
| 6. sleep-time scaling | 더 많은 sleep compute가 언제 더 좋은 장기기억으로 이어지는가? | wake latency와 sleep throughput을 분리하되 model/data 복제 비용을 언제 감수할 것인가? | marginal utility/GPU-s, moved bytes, queue age, promotion yield, rollback rate |

## 3. 제안하는 공통 아키텍처: state를 first-class memory object로 보기

**[SAIT-PROPOSAL]** 핵심은 “sleep accelerator” 하나가 아니라, 서로 다른 state를 동일한 control contract로 관리하는 **Memory Object Fabric**이다.

```text
       latency-sensitive wake plane                      throughput sleep plane
┌──────────────────────────────────┐              ┌─────────────────────────────┐
│ infer / retrieve / state apply   │  immutable   │ replay / dream / distill   │
│ optional session fast update     │──snapshot───▶│ compact / index / validate │
│ TTFT·TPOT SLO                    │              │ candidate build             │
└──────────────┬───────────────────┘              └────────────┬────────────────┘
               │ owner/versioned read-write                                  │
               ▼                                                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     Memory Object Fabric + Control Plane                    │
│ KV │ recurrent state │ checkpoint/transition │ adapter/expert │ text/vector/graph │
│ owner · scope · version · cadence · lineage · capacity · authorization      │
│ admit · place · cache · offload · promote · publish · rollback · delete     │
└──────────────┬─────────────────────────┬──────────────────────────┬────────────┘
               │                         │                          │
          HBM hot tier          near-memory / DRAM warm tier    NVMe/object cold tier
```

### 3.1 Memory object 최소 descriptor

모든 state가 다음 metadata를 가져야 cache, batching, tiering, rollback을 같은 scheduler가 다룰 수 있다.

| 필드 | 의미 | 필요한 이유 |
|---|---|---|
| `object_type` | KV, recurrent state, transition, latent, adapter, expert, text, vector, graph, raw evidence | 같은 “memory”라도 read/write 및 복원 규칙이 다름 |
| `owner_scope` | query/session/user/cohort/shared | batching 가능성과 격리·rollback blast radius 결정 |
| `parent_version` / `generation` | 어느 state에서 생성·갱신됐는지 | stale write 및 out-of-order sleep 완료 방지 |
| `read_write_effect` | immutable, append-only, RMW, copy-on-write | compatible batching과 coherence 결정 |
| `update_cadence` | token/chunk/session/day/pressure-trigger | NSTM·CMS식 multi-frequency를 scheduler 자원 요구로 변환 |
| `reuse_key` | exact prefix, segment digest, semantic key, adapter route | cache hit의 정확성 조건 정의 |
| `composability` | cuttable, associative compose, suffix-recompute, non-composable | KV와 recurrent state를 같은 prefix cache로 오인하지 않게 함 |
| `rebuildability` | canonical source에서 재생성 가능한지 | offload·delete·rollback 정책 결정 |
| `bytes` / `dirty_bytes` | retained, resident, writeback 변화량 | HBM placement와 transfer 비용 계산 |
| `lineage` / `auth_epoch` | source 및 삭제·권한 세대 | parametric promotion 후 correction/rollback 가능성 판단 |

### 3.2 State별 추천 물리 배치 가설

아래는 **[SAIT-PROPOSAL]**이며 제품 기능을 주장하지 않는다.

| State | access 특성 | 우선 tier 가설 | device/system 기회 |
|---|---|---|---|
| active KV | append-heavy, decode에서 반복 read | HBM; inactive prefix는 warm tier | page-level KV placement, quantized transfer, prefix-aware DMA |
| active recurrent/TTT state | request-owned RMW, state가 token KV보다 클 수 있음 | 적용 중 HBM, chunk 사이 warm-tier checkpoint | owner/version tag, dirty-range writeback, fused RMW, chunk buffering |
| Memory Caching checkpoint/transition | context와 함께 증가, 선택적 read | reuse 높은 subset만 HBM, 나머지 capacity tier | saved-FLOPs/byte 기반 admission, segment-aware indexing, near-memory aggregation 후보 |
| active adapter/expert | read-mostly apply, sleep 때 write/candidate 생성 | routed working set은 HBM, inactive catalog는 DRAM/CXL/NVMe | adapter paging, delta transfer, immutable candidate slot |
| optimizer/router/importance state | sleep/update에서만 hot, 본 weight보다 클 수 있음 | sleep HBM 또는 warm memory | update job affinity, sparse/low-rank state movement, checkpoint compression |
| text/vector/graph | random lookup + background append/compaction | DRAM/near-memory/NVMe/object store | ANN/traversal/compaction offload, embedding locality, tier-aware index |
| raw evidence/version/rollback | append, cold scan, rebuild source | encrypted durable cold/recovery tier | content-addressed snapshot, range transfer, provenance-preserving rebuild |

### 3.3 HBM과 near-memory의 역할을 분리하는 원칙

1. **매 token 읽는 state는 가능한 한 HBM에 둔다.** lower tier에 둔 채 매 token 이동시키면 capacity를 얻는 대신 bandwidth·latency 병목을 만든다.
2. **update를 chunkwise로 미룰 수 있으면 lower tier와 HBM 사이의 traffic을 amortize한다.** 이는 KVBuffer와 NSTM이 제시한 “apply와 update frequency 분리”를 일반화한 제안이다.
3. **checkpoint가 growing해도 active set은 bounded하게 둔다.** *Memory Caching*의 model capacity 성장과 serving HBM residency 성장은 동일할 필요가 없다.
4. **near-memory 연산은 bandwidth-dominant 작업부터 검증한다.** state aggregation, quantize/dequantize, vector distance, index scan, compaction이 후보이며, dense training GEMM을 무조건 near-memory로 보내는 제안은 하지 않는다.
5. **full histories 대신 delta와 immutable manifest를 이동한다.** 단, delta만으로 rebuild·delete가 불가능해지면 canonical evidence는 별도 보존한다.

### 3.4 State transition과 data-movement contract

**[SAIT-PROPOSAL]** state 이동은 단순한 tensor copy가 아니라 lifecycle transition으로 기록한다.

| Transition | 이동하는 것 | 이동하지 않는 것 | publish 조건 |
|---|---|---|---|
| wake HBM → warm checkpoint | changed state pages, owner/version, reuse key | 전체 prompt history의 반복 사본 | checkpoint checksum과 parent version 일치 |
| warm → wake HBM prefetch | selected checkpoint/transition, compatibility metadata | 선택되지 않은 archive | request generation과 position convention 일치 |
| external → parametric candidate | authorized training shard 또는 sufficient delta, source lineage | 무기한 raw-history GPU 복제 | held-out retention·interference·privacy gate |
| sleep candidate → serving | adapter/expert delta, router/index roots, manifest | optimizer workspace와 rejected candidates | exact-parent compare-and-swap와 canary 통과 |
| HBM expert → inactive offload | immutable expert version, route statistics | live dirty buffer | reader quiescence와 durable copy acknowledgement |
| rollback | compatible predecessor sub-roots + current authorization state | 과거 deletion/authorization epoch | 새 generation으로 재검증·재공개 |

데이터 이동 최소화의 우선순위는 다음과 같다.

1. source가 큰 extract/embed/graph job은 source shard 가까이에서 실행한다.
2. dense distillation은 model weight가 이미 resident하고 여러 job을 batch할 수 있는 sleep accelerator에서 실행한다.
3. wake↔sleep 사이에는 raw session 전체보다 immutable event range, object digest, low-rank delta를 우선 이동한다.
4. model weight를 sleep fleet에 복제하는 비용과 event shard를 이동하는 비용의 crossover를 실측한다. “compute-to-data”는 항상 옳은 규칙이 아니다.
5. movement 원장은 `storage read`, `host↔device`, `device↔device`, `network`, `replication`, `publication` bytes를 분리한다.

## 4. 핵심 kick questions와 device solution bridge

아래 질문은 Google이 자기 방법을 설명하게 만든 뒤, 자연스럽게 SAIT의 장치·시스템 가설로 이어지도록 설계했다. “예상 답변”은 공개 논문 기준이며 실제 답변으로 간주하지 않는다.

질문 ID의 기준은 [`02-GOOGLE-KICK-QUESTION-PLAYBOOK.md`](02-GOOGLE-KICK-QUESTION-PLAYBOOK.md)의 `KQ-01…06`과 `BQ-01…08`이다. 이 문서의 `DBQ`는 독립적인 주 질문이 아니라, 해당 답변을 device/system 요구사항으로 구체화할 때만 꺼내는 **Device Bridge Question**이다.

| Canonical playbook 질문 | 연결할 device bridge 질문 |
|---|---|
| KQ-01 reasoning-sufficient long context | DBQ-01 capacity/addressability |
| KQ-02 recurrent-state reuse | DBQ-02 composable cache, DBQ-04 ownership/batching |
| KQ-03 prefill/decode/update crossover | DBQ-03 cadence, DBQ-04 ownership/batching, DBQ-10 device primitive |
| KQ-04 bounded memory growth | DBQ-06 finite capacity |
| KQ-05 wake→sleep artifact | DBQ-05 promotion, DBQ-07 placement, DBQ-08 publication |
| KQ-06 sleep-time scaling | DBQ-09 scaling surface, DBQ-10 device primitive |
| BQ-01 event-triggered memorization | DBQ-03 cadence |
| BQ-02 safe merge geometry | DBQ-02 composition, DBQ-06 merge/recycle |
| BQ-03 memory-only proof | DBQ-01 sufficiency measurement, DBQ-05 promotion gate |
| BQ-04 learned CMS frequency | DBQ-03 cadence |
| BQ-05 pre-training economics | DBQ-03 update cost, DBQ-10 device primitive |
| BQ-06 provenance/delete/version | DBQ-04 ownership, DBQ-08 publication/rollback |
| BQ-07 dreaming failure | DBQ-08 validation/rollback, DBQ-09 negative scaling |
| BQ-08 lifecycle benchmark | DBQ-01 byte-quality surface, DBQ-09 lifecycle scaling |

### DBQ-01. Long-context 성능의 병목은 연산량인가, memory capacity인가, addressability인가?

**연결 축:** 1, 5

**우선순위:** 최상

**Kick question**

> *Memory Caching*에서 recurrent model의 recall gap을 fixed-size state capacity 문제로 보고 checkpoint를 늘렸는데, 실제로는 capacity 자체와 “필요한 checkpoint를 정확히 고르는 addressability” 중 어느 쪽이 먼저 saturation합니까? 같은 총 memory byte에서 checkpoint 수·checkpoint width·selection sparsity를 바꾼 scaling surface가 있습니까?

**Google 예상 기술 설명 — [PAPER]**

- MC는 recurrent hidden-state checkpoint를 cache해 effective capacity를 sequence length와 함께 늘리고, GRM/Soup/SSC 등 aggregate/select 방식을 제안한다.
- RNN의 선형 복잡도와 Transformer의 이차 복잡도 사이를 조절하며 recall-intensive task에서 recurrent baseline의 격차를 줄인다.
- 공개 결과에서는 Transformer가 일부 in-context recall에서 여전히 가장 강하지만, 그 결과만으로 capacity와 addressability 가운데 어느 요인이 지배적인지는 분리되지 않는다.

**반드시 물을 follow-up — [OPEN]**

- quality가 cache count에 대해 언제 포화되는가?
- selected checkpoint의 oracle recall과 실제 selector recall 차이는 얼마인가?
- checkpoint read bytes, aggregate FLOPs, HBM hit rate를 분리 측정했는가?

**우리 HW/system insight — [INFERENCE]**

checkpoint를 늘렸다는 사실만으로 addressability가 해결됐다고 볼 수 없다. long-context device 요구량은 raw sequence length보다 `활성 checkpoint 수 × checkpoint bytes × 재사용 빈도`에 더 직접적으로 좌우된다. capacity와 bandwidth를 같은 “memory 크기”로 묶어 비교하면 안 된다.

**Device solution 제안 — [SAIT-PROPOSAL]**

- HBM에는 selector가 고른 working set만 두고, 전체 checkpoint archive는 capacity tier에 둔다.
- cache admission score를 단순 recency가 아니라 `expected saved recompute FLOPs × hit probability / resident bytes`로 둔다.
- selector가 필요로 하는 compact key/summary는 HBM에, full state는 warm tier에 분리 배치한다.
- 공동 실험: 같은 총 byte cap에서 `KV-only`, `fixed recurrent`, `MC checkpoint`, `hybrid exact+compressed`를 비교하고 quality·TTFT·HBM bytes를 함께 그린다.

### DBQ-02. Recurrent state를 KV prefix cache처럼 재사용하려면 최소 대수적 primitive가 무엇인가?

**연결 축:** 2, 3

**우선순위:** 최상

**Kick question**

> KV는 prefix 중간 지점에서 잘라 재사용할 수 있지만 in-place recurrent state는 보통 exact-match밖에 안 됩니다. Google의 MC checkpoint 중 어떤 state는 prefix/segment 단위로 합성하거나 suffix만 재계산할 수 있습니까? cache entry에 state 외에 transition operator나 update metadata를 저장하면 재사용 영역을 넓힐 수 있을까요?

**Google 예상 기술 설명 — [PAPER]**

- MC는 sequence 내부의 과거 hidden-state checkpoint를 현재 online memory와 함께 사용하는 모델 구조를 설명한다.
- 공개 논문은 multi-request serving prefix cache, partial rollback, disaggregated P/D handoff의 complete protocol을 제시하지 않는다.

**산업 보강 근거 — [PAPER, non-Google]**

- Marconi는 recurrent in-place state가 partial overlap rollback을 어렵게 해 exact-match hit 위주가 된다고 분석한다.
- HYPIC은 linear-attention segment의 transition operator와 zero-start end-state를 함께 cache해 segment 합성을 시도한다. near-exact이며 seam recomputation과 추가 state 비용이 있다.

**우리 HW/system insight — [INFERENCE]**

cache primitive는 tensor shape가 아니라 algebraic semantics를 알아야 한다. `cuttable KV`, `exact checkpoint`, `composable transition`, `suffix-recompute state`, `non-composable mutable weight`를 구분하지 않으면 cache correctness와 hit rate를 동시에 잃는다.

**Device solution 제안 — [SAIT-PROPOSAL]**

- cache descriptor에 `prefix/segment digest`, `position convention`, `parent generation`, `composition operator`, `seam requirement`를 넣는다.
- HBM cache와 capacity tier가 동일 key space를 쓰되, entry마다 재계산 FLOPs와 transfer bytes를 기록한다.
- state-transfer DMA는 full-copy 외에 dirty-range, delta, transition+state pair를 지원하는 software-visible protocol부터 실험한다.
- 공동 실험: exact match, sparse checkpoint+suffix recompute, transition compose 세 모드의 accuracy/TTFT/bytes crossover 측정.

### DBQ-03. Apply frequency와 update frequency의 분리가 architecture 원리인가, workload-specific heuristic인가?

**연결 축:** 3, 6

**우선순위:** 최상

**Kick question**

> NSTM은 memory를 매 frame apply하되 update는 periodic하게 수행하고, HOPE/CMS는 여러 update frequency를 둡니다. 이 cadence를 novelty·drift·capacity pressure로 동적으로 정해도 안정성이 유지됩니까? update/apply ratio를 hardware scheduler에 노출할 수 있는 모델-side invariant가 있습니까?

**Google 예상 기술 설명 — [PAPER]**

- NSTM은 update가 apply보다 비싸고 video stream이 중복적이라는 관찰 아래 두 frequency를 분리한다. memory loss와 historical fast-weight running average를 이용해 drift를 완화한다.
- HOPE/CMS는 서로 다른 frequency로 갱신되는 memory spectrum을 모델링한다.
- 두 논문 모두 범용 production scheduler나 최적 cadence law를 확립하지는 않는다.

**반드시 물을 follow-up — [OPEN]**

- fixed cadence 대비 surprise/gradient norm/drift-trigger cadence의 stability curve가 있는가?
- skipped update가 만들어내는 quality debt를 runtime에서 예측할 수 있는가?
- update batch가 커질 때 optimizer state와 activation peak가 어떻게 증가하는가?

**우리 HW/system insight — [INFERENCE]**

cadence는 모델 hyperparameter인 동시에 memory-traffic control knob다. token마다 large state를 RMW하면 decode는 bandwidth-bound가 되지만, chunk마다 writeback하면 latency·staleness와 bandwidth 사이에 조절 가능한 surface가 생긴다.

**Device solution 제안 — [SAIT-PROPOSAL]**

- `APPLY`와 `UPDATE`를 별도 queue 및 kernel class로 노출한다.
- HBM에는 current read view와 bounded delta buffer를 두고, update 시 fused reduction/RMW 후 dirty range만 writeback한다.
- novelty가 낮으면 KVBuffer식 buffer를 늘리고, drift가 높으면 update window를 줄이는 pressure-aware controller를 검증한다.
- 계측: update/apply ratio, HBM read/write per output, dirty-byte ratio, staleness, quality debt, TPOT.

### DBQ-04. Self-modifying state를 multi-tenant serving에서 누구의 state로 볼 것인가?

**연결 축:** 2, 3, 4

**우선순위:** 높음

**Kick question**

> HOPE의 self-modifying memory를 serving에 적용할 때 state scope를 request, session, user, cohort 중 어디에 두는 것이 의도입니까? 서로 다른 state version을 가진 요청을 한 GEMM batch에 넣을 때 read phase와 update commit을 분리할 수 있습니까?

**Google 예상 기술 설명 — [PAPER]**

- HOPE는 self-modifying update와 CMS의 학습 표현을 제안하지만, multi-tenant serving ownership 및 batch commit protocol은 공개 논문의 주 대상이 아니다.

**예상 응답 방향 — [INFERENCE]**

- 따라서 Google이 “모델 구조 연구와 serving contract는 분리된 문제”라고 답할 가능성이 높다.

**산업 보강 근거 — [PAPER, non-Google]**

- RW-TTT는 owner/version/READ-WRITE effect를 표시해 compatible phase만 batch하고 update를 해당 owner에만 commit한다.

**우리 HW/system insight — [INFERENCE]**

mutable model state는 일반 KV보다 작더라도 coherence 비용이 더 크다. shared weight batching은 “같은 주소의 동일 version을 read-only로 읽는다”는 전제에 기대기 때문이다.

**Device solution 제안 — [SAIT-PROPOSAL]**

- state page에 `owner_id`, `version`, `read_epoch`, `dirty/commit bit`를 둔다.
- READ GEMM은 여러 owner를 batch하고, WRITE는 owner별 ordered commit으로 분리한다.
- request-local delta를 base weight와 materialize하지 않고 gather/fused-apply하는 경로를 검증한다.
- P/D 분리 시 transfer 대상은 KV뿐 아니라 `recurrent state + owner/version + outstanding delta`의 atomic bundle로 정의한다.

### DBQ-05. 언제 external memory에서 parametric memory로 승격해야 하는가?

**연결 축:** 4, 5, 6

**우선순위:** 최상

**Kick question**

> Sleep의 Knowledge Seeding은 fast/high-frequency knowledge를 low-frequency expert로 승격합니다. 하지만 correction 가능성이 높거나 한 번만 쓰일 knowledge는 text/vector/graph에 남기는 편이 더 싸고 reversible할 수 있습니다. Google이 생각하는 external-to-parametric promotion criterion은 reuse, uncertainty, interference, deletion risk 중 무엇입니까?

**Google 예상 기술 설명 — [PAPER]**

- Sleep은 sender update 적용 전 full-model state를 teacher로, prospective sender update·fast-expert reset·새 slow low-rank expert를 포함한 expanded state를 student로 두고 새 expanded parameter를 학습하는 Knowledge Seeding을 제안한다.
- Dreaming은 synthetic curriculum 생성·선택 및 RL 기반 개선을 수행한다.
- 공개 논문은 external text/vector/graph와 parametric expert 사이의 allocator를 비교하지 않는다.

**우리 HW/system insight — [INFERENCE]**

parametric promotion은 inference token을 줄이고 repeated retrieval을 없앨 수 있지만, training·optimizer·validation·version·rollback bytes를 만든다. external memory는 latency와 index 비용을 지불하지만 source-level correction과 selective deletion이 상대적으로 쉽다.

**Device solution 제안 — [SAIT-PROPOSAL]**

external-first policy를 baseline으로 두고 다음 조건을 모두 통과한 item만 adapter/expert 후보로 만든다.

```text
expected repeated reuse
× measured quality/latency gain
> build + move + serving-residency + validation + governance + failure cost
```

- volatile/personal/deletion-prone knowledge는 text/vector/graph 또는 user-isolated adapter에 유지한다.
- stable·high-reuse knowledge만 low-rank expert로 compile한다.
- shared base weight merge는 source lineage와 clean rebuild가 입증되기 전에는 제외한다.
- device 측은 active adapter cache, inactive catalog tier, delta-prefetch와 router telemetry를 제공한다.

### DBQ-06. Growing cache와 expert pool이 찼을 때 무엇이 보존되어야 하는가?

**연결 축:** 5

**우선순위:** 최상

**Kick question**

> MC는 context와 함께 checkpoint가 늘고, Sleep은 새 expert에 knowledge를 seed합니다. hard byte cap에 도달했을 때 delete, merge, distill, recycle 중 어떤 operator가 어떤 기억에 적합합니까? 특히 rare-but-critical memory와 frequent-but-reconstructable memory를 어떻게 구분합니까?

**Google 예상 기술 설명 — [PAPER]**

- MC의 selective variants는 active/load set을 줄이는 방향을 제공하지만 lifetime archive policy를 제시하지 않는다.
- Sleep 논문은 growing sparse module의 구현 대안으로 low-rank parameter를 미리 두고 activation 전까지 mask하는 방식을 제시한다. 실제 실험의 allocation 방식·pool 크기와 exhaustion 이후 동작은 공개 결과로 확인되지 않는다.
- HOPE의 느린 update frequency가 곧 bounded capacity나 semantic forgetting을 보장하지는 않는다.

**우리 HW/system insight — [INFERENCE]**

용량은 live tensor만 세면 안 된다. canonical evidence, external payload/index, latent/KV, adapter/expert, router/optimizer, replica, candidate, migration workspace, rollback version을 함께 세야 한다.

**Device solution 제안 — [SAIT-PROPOSAL]**

- hard cap을 두 개로 분리한다: `retained logical state`와 `peak physical residency`.
- 압력 순서: inactive HBM demotion → reconstructable cache eviction → duplicate merge → low-utility expert retire → canonical-source rebuild.
- rare critical memory는 frequency가 낮더라도 protected canary와 criticality tag로 보존한다.
- expert recycle 전 old/new task retention, routing collision, delete lineage, rollback test를 통과시킨다.
- admission/eviction objective는 hit rate 하나가 아니라 utility, saved compute, bytes, volatility, rebuild cost의 Pareto surface로 기록한다.

### DBQ-07. Wake와 sleep을 물리적으로 분리할 시점은 언제인가?

**연결 축:** 3, 6

**우선순위:** 높음

**Kick question**

> 논리적으로 wake apply와 sleep distill/dream은 분리되지만, 물리 cluster까지 분리하면 base weight 복제와 state 이동이 생깁니다. Google이 보는 crossover는 wake P99 interference, sleep batchability, model residency, data locality 중 무엇으로 결정됩니까?

**Google 예상 기술 설명 — [PAPER]**

- Sleep은 별도 학습 단계를 정의하지만 cluster topology와 data-movement accounting을 실험하지 않는다.
- NSTM은 live stream에서 periodic memory update와 per-frame apply를 분리한다.
- HOPE의 nested clock 역시 물리 cluster 수를 직접 정하지 않는다.

**우리 HW/system insight — [INFERENCE]**

NSTM은 offline sleep cluster의 증거라기보다 wake-plane frequency decoupling의 analogue로 해석하는 편이 정확하다. “wake cluster + sleep cluster”는 생물학적 비유가 아니라 queueing과 locality의 문제다. 별도 fleet는 SLO isolation을 주지만 base-model duplicate residency와 network transfer를 증가시킨다.

**Device solution 제안 — [SAIT-PROPOSAL]**

네 deployment mode를 trace-driven으로 비교한다.

1. shared fleet + wake priority/preemption
2. time-partitioned fleet
3. separate wake/sleep fleet
4. shard-local extract/index + pooled GPU distillation hybrid

판단 기준은 `wake P99`, `sleep queue age`, `base duplicate HBM`, `snapshot/delta network bytes`, `preemption waste`, `promotion yield/GPU-s`다. 초기에는 shared/time-partitioned를 baseline으로 하고, sustained sleep load와 SLO interference가 계측될 때만 separate fleet를 정당화한다.

### DBQ-08. Sleep 결과를 어떻게 안전하게 publish하고 되돌릴 것인가?

**연결 축:** 4, 5, 6

**우선순위:** 높음

**Kick question**

> Sleep job이 끝나는 동안 wake state, user correction, deletion이 더 진행될 수 있습니다. Google이 생각하는 publication unit은 expert tensor 하나입니까, 아니면 router·index·external memory를 포함한 generation manifest입니까? stale candidate와 selective rollback을 어떻게 처리합니까?

**Google 예상 기술 설명 — [PAPER]**

- 핵심 논문들은 model-side update, cache, consolidation 메커니즘을 제공하지만, heterogeneous stores를 묶는 atomic publication이나 source-level rollback을 평가하지 않는다.
- 이 질문은 논문 반박이 아니라 deployment gap 확인 질문이다.

**우리 HW/system insight — [INFERENCE]**

weight만 교체하고 router/index가 이전 generation이면 mixed state가 된다. adapter를 base에 merge한 뒤 source deletion이 들어오면 selective reversal이 매우 어려워진다.

**Device solution 제안 — [SAIT-PROPOSAL]**

- wake는 immutable snapshot과 exact parent generation을 sleep에 넘긴다.
- sleep은 live state를 in-place 수정하지 않고 copy-on-write candidate를 만든다.
- `adapter/expert + router + vector/graph roots + compatibility metadata`를 하나의 manifest로 검증한 뒤 pointer를 atomic swap한다.
- rollback은 old authorization epoch를 복원하지 않고, 현재 삭제/권한을 적용한 clean predecessor를 새 generation으로 publish한다.
- HBM에는 current와 bounded candidate working set만 두고, predecessor는 rollback window 동안 capacity/recovery tier에 둔다.

### DBQ-09. Sleep-time compute scaling law의 x축과 y축은 무엇이어야 하는가?

**연결 축:** 1, 4, 5, 6

**우선순위:** 높음

**Kick question**

> sleep GPU-hour를 늘렸을 때 성능이 오르는 것만으로는 scaling law가 되기 어렵습니다. Google은 sleep compute, retained memory byte, generated data quality, number of update levels 중 무엇을 독립 축으로 봅니까? 이전 knowledge retention과 future reuse까지 포함한 iso-lifecycle-cost curve를 만들 수 있을까요?

**Google 예상 기술 설명 — [PAPER]**

- Sleep은 Knowledge Seeding과 Dreaming이 continual learning·knowledge incorporation·few-shot task에서 이득을 보임을 보고한다.
- HOPE는 nested levels와 update frequencies라는 새로운 표현 축을 제안한다.
- 공개 결과는 보편적 lifetime scaling law, pool-exhaustion curve, total data movement/energy curve를 확립하지 않는다.

**우리 HW/system insight — [INFERENCE]**

같은 quality라도 external retrieval, recurrent state, expert consolidation은 비용을 다른 시점과 tier에 지불한다. query-time latency만 비교하면 sleep training과 recovery bytes를 숨기게 된다.

**Device solution 제안 — [SAIT-PROPOSAL]**

공동 scaling surface를 다음 독립 축으로 구성한다.

- wake inference compute/token
- wake mutable-state bytes 및 HBM traffic/token
- sleep GPU-s와 generated/replayed tokens
- retained logical bytes와 peak physical bytes
- network/storage movement bytes
- update levels/cadence

종속 변수는 fresh-task acquisition, old-memory retention, long-context recall, latency, energy, delete/rollback completeness다. 단일 headline score 대신 iso-quality 및 iso-lifecycle-cost frontier를 보고한다.

**정량 기준선:** [E5 deterministic results](../../experiments/E5-attn-vs-hope/results.json)와 [user-reviewed native live-Sheet archive](../attn-vs-hope/README.md)를 각각 reproducible companion과 operational reference로 사용한다.

### DBQ-10. Google 모델팀이 device에 요구하는 최소 primitive는 무엇인가?

**연결 축:** 2, 3, 5, 6

**우선순위:** 마무리 질문

**Kick question**

> 알고리즘을 특정 memory technology에 고정하지 않고도, 다음 세대 memory system이 제공하면 연구 속도를 올릴 최소 primitive 세 가지를 고른다면 무엇입니까: 더 큰 capacity, 더 높은 bandwidth, owner-versioned RMW, chunk writeback, state composition, adapter paging, near-memory aggregation, atomic snapshot 중 무엇이 우선입니까?

**Google 예상 기술 설명 — [OPEN]**

공개 논문만으로 우선순위를 확정할 수 없다. 이 질문의 목적은 공유 가능한 profiler 기반 bottleneck ranking이나 public/synthetic trace를 통해 실제 병목을 함께 좁히는 것이다.

**우리 HW/system insight — [INFERENCE]**

“HBM 용량을 늘려 달라”는 답만으로는 공동 설계가 되지 않는다. state의 access frequency, mutation granularity, reuse semantics, working-set curve가 있어야 bandwidth·capacity·near-memory 중 어디에 투자할지 결정할 수 있다.

**Device solution 제안 — [SAIT-PROPOSAL]**

첫 공동 prototype 후보를 다음 세 개로 압축해 제시한다.

1. **Owner-Versioned State Cache:** recurrent/TTT state의 READ batch와 ordered WRITE commit
2. **Cadence-Aware Hierarchical State Buffer:** token apply + chunk update, dirty-range writeback, HBM/warm-tier placement
3. **Transactional Memory Artifact Registry:** external memory와 adapter/expert의 snapshot, promotion, offload, rollback

Google에는 각 prototype에 필요한 tensor size, access trace, update cadence, acceptable staleness, correctness invariant를 요청한다.

**정량 기준선:** 현재 비교 수치와 formula lineage는 [E5 package](../../experiments/E5-attn-vs-hope/README.md)에서 재현하고, 미팅에서 검토한 native assumptions와 scenario cells는 [live-Sheet archive](../attn-vs-hope/README.md)에서 확인한다.

## 5. 질문별 빠른 bridge map

| 질문 | Google 설명을 듣고 잡아낼 신호 | 바로 연결할 SAIT 제안 | 공동 PoC 성공 기준 |
|---|---|---|---|
| DBQ-01 capacity vs addressability | cache count/selector saturation | HBM working-set + capacity tier | 동일 quality에서 HBM bytes 또는 TTFT 감소 |
| DBQ-02 state reuse primitive | checkpoint compose/rollback 가능 범위 | algebra-aware cache descriptor | exactness 조건을 지키며 saved FLOPs/byte 증가 |
| DBQ-03 apply/update cadence | update debt와 안정성 signal | chunk buffer + dirty writeback | quality floor 하에서 HBM write/token 감소 |
| DBQ-04 ownership/batching | state scope와 commit order | owner-versioned state cache | state contamination 0, batch throughput 증가 |
| DBQ-05 promotion | reuse/volatility/interference 판단 | external-first tiered promotion | iso-quality lifecycle cost 감소 |
| DBQ-06 finite capacity | pool/cache exhaustion 행동 | protected eviction/merge/recycle | hard cap 준수 + old/new retention gate 통과 |
| DBQ-07 cluster topology | model/data locality와 SLO | adaptive shared/separate fleet | wake P99를 지키며 sleep backlog 안정화 |
| DBQ-08 publication | candidate consistency unit | manifest-last atomic publish | stale publish 0, tested rollback/delete |
| DBQ-09 scaling law | 독립 자원축과 평가 horizon | lifecycle Pareto benchmark | 여러 substrate를 동일 원장으로 비교 |
| DBQ-10 device primitive | 실제 profiler bottleneck | 3개 prototype 우선순위 결정 | trace와 correctness contract 확보 |

## 6. SAIT가 제시할 구체적 공동 연구 package

아래 Package A–C를 이 패키지의 canonical 공동 연구 제안 단위로 사용한다. 앞 절의 device primitive와 prototype은 이 세 package를 구성하는 하위 building block이다.

### Package A — Recurrent/TTT State Trace Suite

**목표:** HOPE/MC/NSTM류 state의 실제 hardware 요구를 추측이 아니라 trace로 얻는다.

필수 trace:

```text
state tensor shape and dtype
read/write bytes per apply and per update
dirty-byte fraction
owner and version transitions
update/apply cadence
checkpoint count and selected working set
reuse-key type and hit/miss reason
recompute FLOPs saved by each hit
HBM/L2/host-memory traffic
prefill TTFT and decode TPOT
```

결과물은 특정 accelerator에 종속된 평균값이 아니라 sequence length, concurrency, chunk size, cache cap에 대한 curve여야 한다.

### Package B — Hierarchical Mutable-State Cache Prototype

**목표:** KV와 recurrent/parametric state를 동일 cache에 억지로 넣지 않고, semantics-aware controller로 함께 운영한다.

- HBM: active KV, selected recurrent state, current routed expert
- warm capacity tier: reusable checkpoint, transition, inactive adapter
- cold/recovery: canonical source, old generation, rebuild shard
- controller: owner/version, composition mode, saved-FLOPs/byte, dirty writeback

최소 비교군:

1. all-HBM
2. LRU tiering
3. reuse-only admission
4. saved-FLOPs/byte admission
5. cadence-aware dirty writeback

### Package C — Wake/Sleep Lifecycle Benchmark

**목표:** algorithm quality와 device cost를 동일 lifecycle 원장에서 비교한다.

시나리오:

- 반복 사용되는 stable fact/procedure
- 한 번만 쓰이는 event
- 나중에 correction/deletion되는 fact
- rare-but-critical instruction
- long-context exact retrieval
- domain drift와 adversarial/poisoned episode
- expert/cache hard-cap exhaustion
- sleep job이 stale generation에서 완료되는 race

평가:

```text
fresh acquisition
old retention
exact/paraphrase/compositional recall
TTFT/TPOT/throughput
HBM and movement bytes
sleep GPU-s and queue age
retained and peak state bytes
promotion yield and rollback rate
correction/delete completeness
```

## 7. Capacity pressure와 observability contract

### 7.1 반드시 분리할 byte 원장

**[SAIT-PROPOSAL]** 다음 항목을 중복 없이 분리한다.

```text
B_retained =
  canonical evidence
  + external text/vector/graph and index
  + latent/KV/recurrent state
  + adapter/expert/parametric delta
  + optimizer/router/catalog metadata
  + lineage/recovery versions

B_peak = B_retained
  + candidate/staging
  + pinned predecessor
  + migration/dual-index overlap
  + build workspace
  + temporary replicas
```

HBM placement에서는 frozen base weight와 execution workspace도 별도 더한다. “active expert 수가 고정” 또는 “hot buffer가 bounded”라는 사실만으로 total lifetime memory가 bounded하다고 간주하지 않는다.

### 7.2 Pressure signal

| pressure | 의미 | 허용 action | 금지 shortcut |
|---|---|---|---|
| `rho_hbm` | hot state residency 포화 | demote, quantize, smaller active set | owner/version 없는 arbitrary spill |
| `rho_bw` | state apply/update traffic 포화 | chunk update, buffer, fuse, recompute 선택 | quality debt를 기록하지 않은 update skip |
| `rho_cache` | checkpoint/catalog growth | admit/evict/merge/rebuild | rare critical item을 frequency만으로 삭제 |
| `rho_sleep` | background GPU queue 포화 | coalesce, cheaper operator, defer | partial candidate를 complete로 publish |
| `rho_route` | expert/vector 경쟁 증가 | hierarchy, prune, route canary | oracle route 수치로 production 주장 |
| `rho_gov` | delete/correction/rollback backlog | promotion 제한, external-first | tombstone만으로 parametric erase 주장 |

### 7.3 최소 dashboard

**Wake**

- TTFT/TPOT/P99, queueing, active batch width
- KV/recurrent/adapter별 HBM resident bytes
- physical HBM read/write bytes per output token
- exact/partial/segment cache hit 및 saved FLOPs/byte
- state owner/version conflict와 stale-read count
- apply/update ratio, dirty bytes, skipped-update debt

**Sleep**

- input snapshot, parent generation, candidate generation
- job type, queue/start/end/publish time
- GPU-s, HBM-byte-s, storage/network bytes
- replay/generated data mix, teacher/router version
- candidate validation pass/fail과 protected-slice regression
- stale completion, retry, preemption, promotion yield

**Memory lifecycle**

- state type·scope별 retained/active/archive bytes
- expert/checkpoint pool occupancy와 age/reuse distribution
- promotion/demotion/merge/recycle/rollback history
- source lineage, correction/delete dependency fan-out
- rebuildability 및 model/tokenizer/index compatibility
- utility per resident byte, per moved byte, per sleep GPU-s

## 8. 미팅 진행 순서

### 8.1 30분 압축안

| 시간 | 진행 | 목표 |
|---|---|---|
| 0–3분 | 공동 framing 제시 | “state lifecycle이 다음 병목”에 합의 |
| 3–9분 | KQ-01: reasoning-sufficient memory | long-context의 품질 기준 합의; 필요 시 DBQ-01 |
| 9–15분 | KQ-02: recurrent-state reuse | 안전한 reuse invariant 확인; 필요 시 DBQ-02/04 |
| 15–21분 | KQ-05: wake→sleep artifact | external↔parametric promotion 계약 확인; 필요 시 DBQ-05/08 |
| 21–25분 | 선택 branch 하나 | serving이면 KQ-03, capacity면 KQ-04, scaling이면 KQ-06 |
| 25–30분 | 공동 PoC 합의 | package 하나, shareable trace/summary 하나, 다음 체크포인트 결정 |

30분에서는 device 질문을 독립 순서로 모두 소화하지 않는다. 각 KQ 답변에서 실제 bottleneck이 드러날 때 crosswalk의 DBQ 하나만 꺼낸다. 답변이 “serving/lifecycle은 범위 밖”이면 DBQ-10으로 넘어가 공개 또는 synthetic trace 기반 공동 측정을 제안한다.

### 8.2 60분 심화안

| 시간 | 진행 | 목표 |
|---|---|---|
| 0–5분 | framing과 증거 경계 | paper result와 deployment proposal 분리 |
| 5–12분 | KQ-01 | reasoning sufficiency와 long-context quality 기준 |
| 12–19분 | KQ-02 | state identity, composition, prefix/segment reuse |
| 19–26분 | KQ-03 | prefill/decode/update의 phase별 crossover |
| 26–33분 | KQ-04 | hard cap, eviction/merge/recycle |
| 33–40분 | KQ-05 | wake artifact와 external/parametric promotion |
| 40–47분 | KQ-06 | sleep-time scaling surface와 failure regime |
| 47–54분 | 답변 기반 DBQ 1–2개 | ownership, cadence, publication, topology 중 실제 병목만 심화 |
| 54–60분 | 공동 package와 next step | prototype 하나, shareable evidence 하나, 실험 하나 확정 |

60분 미팅의 종료 조건은 “좋은 토론”이 아니라 다음 세 가지다.

1. Google이 공유 가능한 anonymized trace, aggregate profile, public/synthetic trace 중 하나의 필드 합의
2. SAIT가 모델링할 memory hierarchy와 baseline topology 합의
3. hard-cap 또는 cadence sweep 중 첫 공동 experiment 합의

## 9. Answer Capture Sheet schema

미팅 중 자유 서술만 남기면 논문 주장, Google 구두 의견, 우리 제안이 섞인다. 현장 기록과 사후 분석을 분리한다.

### 9.1 Live capture — 현장용 9개 column

| Column | 입력 규칙 |
|---|---|
| `Question_ID` | canonical `KQ-01…06` 또는 `BQ-01…08` |
| `Device_Bridge_ID` | 사용한 경우에만 `DBQ-01…10`; 없으면 `NONE` |
| `Answer_Verbatim` | 핵심 원문 또는 즉시 작성한 faithful paraphrase |
| `Answer_Type` | `PAPER_FACT`, `UNPUBLISHED_RESULT`, `OPINION`, `OPEN` |
| `State_Invariant_Bottleneck` | state 표현, correctness invariant, 지배 병목을 한 줄로 기록 |
| `Evidence_Shareability_Attribution` | paper/figure/code/profile/none + 공유 가능 범위 + 발언 attribution 상태 |
| `Allowed_Use` | `PUBLIC`, `JOINT_INTERNAL`, `SAIT_INTERNAL`, `DO_NOT_CITE` |
| `Open_Issue_SAIT_Bridge` | 상대가 인정한 공백과 연결할 SAIT 가설 |
| `Owner_Next_Action` | Google/SAIT/joint + 자료·계산·prototype·후속 일정 |

### 9.2 Post-meeting enrichment — 분석용 상세 schema

아래 column은 미팅 뒤 recording/minutes와 논문을 대조해 채운다. 현장에서 30개 이상을 동시에 입력하지 않는다.

| Column | 입력 규칙 |
|---|---|
| `Meeting_Date` | YYYY-MM-DD |
| `Question_ID` | canonical `KQ-01…06` 또는 `BQ-01…08` |
| `Device_Bridge_ID` | `DBQ-01…10` 또는 `NONE` |
| `Research_Axis` | 1–6, 복수 허용 |
| `Question_Verbatim` | 실제 던진 문장 |
| `Answer_Type` | `PAPER_FACT`, `UNPUBLISHED_RESULT`, `OPINION`, `OPEN` 중 하나 |
| `Google_Answer_Summary` | 해석 없이 3문장 이내 |
| `Evidence_Pointer` | paper/figure/table/code/profiler 여부; 없으면 `NONE` |
| `Can_Share_Artifact` | paper, code, trace, formula, none |
| `Attribution_Status` | named/role-only/chatham-house/not-attributable/unconfirmed |
| `Allowed_Use` | public/joint-internal/SAIT-internal/do-not-cite |
| `State_Object` | KV/recurrent/checkpoint/transition/adapter/expert/text/vector/graph/raw |
| `Owner_Scope` | request/session/user/cohort/shared |
| `Tensor_or_Record_Shape` | shape, dtype, bytes; 모르면 unknown |
| `Apply_Cadence` | token/frame/chunk/session/other |
| `Update_Cadence` | 고정값 또는 trigger |
| `Read_Bytes` | apply/update별 분리; logical인지 physical HBM인지 명시 |
| `Write_Bytes` | dirty/full-copy 구분 |
| `Reuse_Primitive` | exact prefix/segment/compose/suffix recompute/semantic route/none |
| `Working_Set_Curve` | context, concurrency, cap에 따른 active bytes |
| `Capacity_Bound` | hard/soft/unbounded/not measured |
| `Pressure_Action` | evict/merge/distill/recycle/offload/no-op |
| `Correctness_Invariant` | exactness, staleness, owner/version, accuracy tolerance |
| `Promotion_Criterion` | external→parametric 또는 local→shared 기준 |
| `Rollback_Delete` | supported/partial/not addressed + 단위 |
| `Primary_Bottleneck` | compute/HBM BW/HBM capacity/interconnect/storage/router/governance |
| `Google_Open_Question` | Google이 미해결이라고 인정한 항목 |
| `SAIT_Hypothesis` | 우리 제안; Google 답변과 분리 |
| `Proposed_Device_Primitive` | cache/tiering/RMW/DMA/near-memory/registry |
| `Validation_Experiment` | 독립변수, baseline, metric, pass/fail |
| `Confidence` | high/medium/low |
| `Owner` | Google/SAIT/joint |
| `Next_Action` | 자료 공유, 계산, prototype, 후속 미팅 |
| `Due_Date` | YYYY-MM-DD |

### 9.3 현장에서 가능하면 수치화할 다섯 항목

1. state shape/dtype 및 request당 bytes
2. physical HBM read/write per apply/update
3. update/apply cadence와 허용 staleness
4. cache cap 대비 working-set/hit/quality curve
5. candidate build·transfer·publish 비용과 실패/rollback 단위

숫자를 공유할 수 없다면 상대적 curve, order of magnitude, bottleneck ranking 중 하나를 요청한다. “memory-bound”라는 서술만 기록하지 않는다.

## 10. 미팅에서 피해야 할 과장과 표현

| 피할 표현 | 사용할 표현 |
|---|---|
| “HOPE가 장기기억을 해결했다” | “HOPE는 multi-frequency/self-modifying memory의 모델 원리를 제안했고 lifetime system은 open이다” |
| “Memory Caching이면 RNN이 Transformer를 대체한다” | “capacity/complexity trade-off를 열고 recall gap을 줄였지만 addressability와 serving growth는 남는다” |
| “NSTM이 sleep compute다” | “NSTM은 live wake-plane에서 apply/update frequency를 분리한 강한 analogue다” |
| “Sleep은 memory가 무한히 늘어난다” | “논문은 preallocate-and-mask를 구현 대안으로 제시하지만 실제 pool 크기와 exhaustion 동작은 검증하지 않았다” |
| “near-memory가 답이다” | “bandwidth-dominant operator와 movement crossover를 trace로 찾아야 한다” |
| “separate sleep cluster가 필수다” | “wake P99와 duplicate residency·movement의 crossover로 결정한다” |
| “checkpoint가 있으면 rollback된다” | “source lineage, router/index compatibility, authorization을 포함한 atomic publication이 있어야 semantic rollback이다” |

## 11. 권장 마무리 문장

> 저희가 보기에는 네 연구가 공통적으로 update frequency와 memory capacity를 새로운 scaling 축으로 열었습니다. 다음 공동 과제는 그 state를 HBM에 모두 올리는 것이 아니라, **어떤 state를 얼마나 자주 apply/update하고, 어느 단위로 재사용하며, 언제 external에서 parametric으로 promote하고, cap에 닿았을 때 어떻게 되돌릴지**를 hardware-visible contract로 만드는 것입니다. Google과 model-side correctness invariant 및 공유 가능한 access-profile summary를 합의하면, SAIT는 hierarchy·data movement·capacity pressure를 포함한 device/system Pareto surface를 함께 만들 수 있습니다.

## 참고한 1차 출처

- Behrouz et al., [*Nested Learning: The Illusion of Deep Learning Architectures*](https://arxiv.org/abs/2512.24695).
- Behrouz et al., [*Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories*, arXiv v2](https://arxiv.org/abs/2606.03979v2).
- Behrouz et al., [*Memory Caching: RNNs with Growing Memory*](https://arxiv.org/abs/2602.24281).
- Elmieh et al., [*Online Neural Space Time Memory for Dynamic Novel View Synthesis*](https://arxiv.org/abs/2607.15271).
- Pan et al., [*Marconi: Prefix Caching for the Era of Hybrid LLMs*](https://arxiv.org/abs/2411.19379).
- Yang et al., [*RW-TTT: Batched Serving for Request-Owned Test-Time Training State*](https://arxiv.org/abs/2605.28053).
- Zou and Zhong, [*KVBuffer: IO-aware Serving for Linear Attention*](https://arxiv.org/abs/2605.19049).
- Liu et al., [*HYPIC: Accelerating Hybrid-Attention LLM Serving with Position-Independent Caching*](https://arxiv.org/abs/2607.01299).
