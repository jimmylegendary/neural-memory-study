# Google 연구팀 미팅용 Kick Question Playbook

> 대상 연구축: **HOPE / Nested Learning**, **Language Models Need Sleep**, **Memory Caching**, **NSTM**
>
> 목적: 논문별 세부 구현을 묻는 자리가 아니라, 업계가 공유하는 문제를 먼저 고정한 뒤 Google의 설계 원리를 끌어내고 Samsung SAIT의 모델·메모리·디바이스 공동 설계 역량으로 연결한다.
>
> 작성 기준일: 2026-07-27

> **패키지 역할:** [① evidence pre-read / problem map](01-SIX-AXIS-PROBLEM-MAP.md) · ② canonical facilitator playbook · [③ device/system appendix](03-DEVICE-SOLUTION-BRIDGE.md)

---

## 0. 이 문서의 읽는 법

### 증거 강도 표기

- **[D: Direct]**: 논문 본문·수식·표·한계 절이 직접 주장하거나 정의한다.
- **[E: Empirical]**: 논문 실험 범위 안에서 관찰된 결과다. 일반 법칙으로 확대하지 않는다.
- **[I: Inference]**: 위 직접 근거에서 추론한 Google 측의 개연성 높은 답변이다.
- **[O: Open]**: 해당 논문만으로 답할 수 없거나 직접 검증되지 않은 문제다.
- **[H: Hypothesis]**: SAIT가 후속으로 제시할 독창적 가설·실험·시스템 제안이다.

### 수식 해석 정규화

이 플레이북은 self-modifying Titans/HOPE의 query를 다음처럼 취급한다.

$$q_t=x_tW_q,\qquad o_t=M_{\mathrm{memory},t-1}(q_t).$$

즉, 별도 $M_q$ 모듈을 가정하지 않는다. Nested Learning 원문의 일부 업데이트 집합에 $q$가 다시 등장하는 것은 앞뒤 정의와 충돌하는 표기상의 오류로 취급한다. 실제 질문과 시스템 비용 논의에서는 auxiliary memory를 $M_k,M_v,M_\eta,M_\alpha$ 네 개로 한정한다. 이 표기 문제 자체를 첫 질문으로 꺼내지는 않는다.

---

## 1. 우리가 먼저 고정할 공통 프레임

### 1.1 학습 생애주기

우리가 제안할 프레임은 배포 전후를 단순히 training/test로 나누지 않는다.

1. **배포 전 학습**: pre-training과 post-training. 전역적인 slow weight를 만든다.
2. **Wake-time 학습**: 실제 입력을 받는 동안 ICL, recurrent state, fast-weight/TTT update, 외부 memory write를 수행한다.
3. **Sleep-time 학습**: 입력 응답 latency path에서 분리된 시간에 replay, distillation, consolidation, pruning, dreaming을 수행한다.

### 1.2 기억 배치의 두 축

기억을 “모델 내부냐 외부냐” 한 축으로만 보면 중요한 설계 차이가 사라진다. 미팅에서는 다음 두 축을 동시에 사용한다.

| 축 | 한쪽 극단 | 중간 지대 | 반대 극단 |
|---|---|---|---|
| 표현 위치 | token/KV, text·vector·graph 등 외부 episodic memory | recurrent checkpoint, fast weight, adapter | shared model weight의 parametric semantic memory |
| 갱신 시간척도 | 매 token/매 request | chunk·session·day | release·장기 sleep cycle |

핵심 질문은 “어느 하나가 옳은가”가 아니라 **어떤 정보가, 어떤 근거로, 언제, 어느 표현으로 승격·압축·삭제되는가**다.

### 1.3 인프라 프레임

- **Wake inference/training cluster**: 낮은 tail latency, stateful batching, fast-state update.
- **Sleep training cluster**: replay·distillation·RL·data generation, 비동기적이고 throughput 중심.
- **Memory plane**: external store, recurrent checkpoint, fast weight, adapter, consolidated parameter의 version·provenance·검색·offload를 관리.
- **핵심 시스템 목적**: 세 plane 사이의 데이터 이동량과 write amplification을 최소화하면서 정확도·기억 유지·삭제 가능성을 보장한다.

### 1.4 여섯 연구축과 질문 매핑

| 사용자 연구축 | 주 질문 | 보조 질문 |
|---|---|---|
| 1. Full attention이 못 푸는 long-context 이해·test-time scaling | KQ-01 | BQ-03, BQ-08 |
| 2. RNN state 재사용과 serving 효율 | KQ-02, KQ-03 | BQ-02, BQ-06 |
| 3. RNN/TTT의 pre-training·prefill·decode 효율 | KQ-02, KQ-03 | BQ-01, BQ-05, BQ-08 |
| 4. 장기기억과 continual learning | KQ-01, KQ-04, KQ-05 | BQ-03, BQ-04, BQ-06, BQ-07, BQ-08 |
| 5. 고정 용량에서 단조 증가하는 기억 관리 | KQ-04, KQ-05 | BQ-01, BQ-02, BQ-04, BQ-06 |
| 6. Sleep-time compute라는 새 scaling 축 | KQ-05, KQ-06 | BQ-01, BQ-07, BQ-08 |

---

## 2. 미팅 운영 원칙

### 2.1 30초 오프닝

> “저희는 long-context를 단순히 더 많은 token을 보존하는 문제라기보다, **어떤 상태를 wake path에서 유지하고, 무엇을 sleep path에서 더 느린 기억으로 전환하며, 그 상태를 serving에서 얼마나 재사용할 수 있는가**의 공동 설계 문제로 보고 있습니다. 그래서 오늘은 개별 benchmark 숫자보다 retention–adaptation–serving의 세 계약과, 그것이 hardware memory hierarchy에 주는 요구를 중심으로 의견을 맞춰보고 싶습니다.”

### 2.2 질문 순서

60분 기준 권장 순서는 다음과 같다.

1. KQ-01로 “기억했는가”와 “이해했는가”를 구분한다.
2. KQ-02와 KQ-03으로 serving·prefill/decode의 실제 비용을 연다.
3. KQ-04로 bounded capacity를 정면에 둔다.
4. KQ-05로 parametric–external, wake–sleep의 전환 계약을 묻는다.
5. KQ-06으로 공동 scaling-law/시스템 연구 제안까지 확장한다.

질문을 한 뒤 바로 우리 해법을 길게 설명하지 않는다. 먼저 상대가 선택한 **state representation, invariant, metric, bottleneck**을 기록하고, 그 선택에 맞춰 후속 branch를 탄다.

---

# Part I. Top 6 Kick Questions

## KQ-01. 긴 문맥에서 “저장량”이 아니라 “이해 가능한 상태”의 충분조건은 무엇인가?

**커버 축:** 1, 4 / long-context understanding, recurrent compression, parametric vs non-parametric memory

### 문제 framing

Full attention은 모든 과거 token에 직접 접근하는 자라는 memory를 제공하지만 비용이 커진다. 고정 크기 recurrent state는 비용을 줄이지만 recall-intensive task에서 과적재된다. Memory Caching은 checkpoint 수 $N$을 통해 $O(L)$ recurrent와 $O(L^2)$ attention 사이를 $O(NL)$로 보간한다. 그러나 이것은 **접근 경로를 복구하는 것**이지, 그 상태가 다단계 추론·반사실·상태 추적에 필요한 관계를 충분히 보존했다는 보장은 아니다.

### 실제 질문 문장

> “We see the core long-context issue as preserving a state that is sufficient for downstream reasoning, not merely increasing the amount of retrievable history. Across full attention, recurrent compression, and cached recurrent checkpoints, what property would you use to decide that a memory state is sufficient for compositional reasoning rather than only for recall?”

한국어 의미:

> “저희는 long context의 핵심을 단순 보존량보다 downstream reasoning에 충분한 상태를 유지하는 문제로 봅니다. Full attention, recurrent compression, cached checkpoint 사이에서, 어떤 성질을 기준으로 그 상태가 단순 회상이 아니라 조합적 추론에 충분하다고 판단하시겠습니까?”

### 왜 kick인가

- “어느 모델 점수가 더 좋은가”가 아니라 **memory representation의 품질 기준**을 묻는다.
- Transformer의 retrieval 우위와 recurrence의 state-tracking 우위를 동시에 인정해 방어적인 비교를 피한다.
- 답변이 정보이론, task-conditioned sufficient statistics, learned retrieval, memory supervision 중 어디로 향하는지에 따라 공동 연구 지점이 선명해진다.

### Google 논문 기반 예상 답변

- **[D]** Memory Caching은 Transformer의 자라는 memory와 RNN의 고정 memory 사이를 checkpoint 수로 보간하며, SSC는 query가 관련 checkpoint 일부만 고른다(MC §3, Eq. 5·17·19).
- **[E]** MC 계열은 recurrent baseline보다 recall 격차를 줄이지만 짧은 in-context recall에서는 Transformer가 여전히 최고다(MC Table 3 및 결론).
- **[D]** HOPE는 self-modifying deep memory와 다중 시간척도 CMS를 결합한다(NL §8).
- **[E]** HOPE는 NIAH·BABILong·state-tracking에서 장점을 보고하지만, 짧은 in-context recall 표에서는 Transformer가 HOPE보다 높다(NL §9.2, §9.4, §9.5).
- **[I]** Google 측은 하나의 고정된 충분조건보다, deep memory의 표현력·checkpoint granularity·selective routing·multi-timescale update를 함께 학습해야 한다고 답할 가능성이 높다.
- **[O]** 네 논문은 “reasoning-sufficient memory”의 task-independent 정의나 보존 오차의 상한을 제시하지 않는다.

### 우리가 이어갈 기술 follow-up / 가설

- **[H] Reasoning Sufficiency Curve:** 동일한 원문맥에 대해 기억 형태별로 다음 세 곡선을 분리한다.

  $$R_{\mathrm{exact}}(B),\qquad R_{\mathrm{relational}}(B),\qquad R_{\mathrm{counterfactual}}(B),$$

  여기서 $B$는 저장 byte 또는 접근 byte다. 단순 needle recall, 관계 합성, 반사실 추론이 서로 다른 속도로 열화한다는 가설이다.
- **[H]** checkpoint/router가 token relevance만 예측하지 않고, “향후 어떤 연산 그래프에 필요할 관계인가”를 예측하도록 supervision을 둔다.
- **디바이스 연결:** memory tier마다 raw token, checkpoint, fast weight의 **quality-per-byte**를 측정해 HBM–CXL/host–SSD 배치를 결정한다.

### 답변에 따른 branch

- **상대가 task-conditioned sufficiency를 강조하면:** “그 criterion을 online admission signal로 계산할 때 비용과 label은 무엇인가?”로 이어간다.
- **정보 보존/압축률을 강조하면:** 동일 mutual information이라도 causal reasoning이 달라질 수 있음을 짚고 BQ-03의 memory-only supervision으로 이동한다.
- **routing을 강조하면:** router miss와 memory corruption을 분리 평가하는 BQ-02·BQ-08로 이동한다.
- **attention을 최종 safety net으로 남긴다고 하면:** 어느 layer·어느 token에서만 full attention을 켤지, heterogeneous accelerator path를 제안한다.

### 확인할 근거·식·그림

- Memory Caching: Eq. 5(online+cached memory aggregation), Eq. 17(SSC Top-k), Eq. 19–20(segment-size-1 special case가 gated global attention을 복원), Fig. 1·2·5.
- Nested Learning: CMS Eq. 71, HOPE Fig. 5, NIAH Table 1, short ICR Table 3, context usage Fig. 10.
- 미팅에서 요청할 추가 근거: 같은 모델·같은 byte budget에서 recall/reasoning/state-tracking을 동시에 그린 Pareto curve.

---

## KQ-02. Mutable recurrent state를 prefix KV처럼 안전하게 재사용할 수 있는가?

**커버 축:** 2, 3 / state reuse, prefix caching, multi-tenant serving

### 문제 framing

RNN·TTT·HOPE의 효율은 압축된 state에 기대지만, 실제 serving에서는 동일 prefix의 KV cache 재사용률이 중요한 경제성 변수다. Mutable state는 모델·prefix뿐 아니라 **update rule, chunk boundary, parameter version, session history**에 의존한다. 따라서 “state가 작다”와 “state를 재사용할 수 있다”는 다른 주장이다.

### 실제 질문 문장

> “For stateful recurrent or test-time-learning models, what is the right equivalence class for safely reusing a state across requests? Is matching the token prefix enough, or must the cache key include the update trajectory, chunk boundaries, model version, and user/session identity? We are especially interested in whether recurrent-state reuse can reach the practical reuse rate of prefix KV caching.”

### 왜 kick인가

- 논문의 asymptotic memory 절감이 실제 serving TCO로 이어지는 핵심 전제를 묻는다.
- 학술적 architecture 질문을 scheduler, versioning, copy-on-write, isolation 문제로 연결한다.
- Google이 상태를 “세션 전용”으로 보는지 “공유 가능한 artifact”로 보는지 확인할 수 있다.

### Google 논문 기반 예상 답변

- **[D]** MC는 segment 끝의 recurrent checkpoint를 저장하고 이후 token이 그것에 접근하게 한다. checkpoint continuation과 독립 segment compressor 두 선택 모두를 논의한다(MC §3.4).
- **[D]** SSC는 segment summary로 checkpoint를 route하고 선택된 memory만 accelerator에 올릴 수 있다고 설명한다(MC Eq. 16·17).
- **[D]** NSTM은 memory update와 apply 빈도를 분리해 한 번 갱신한 상태를 여러 synthesis frame이 공유한다. 1 update + 약 30 read라는 실증을 보인다(NSTM §3.1, Table 5).
- **[I]** 논문 계열의 자연스러운 답은 “read-only snapshot은 재사용할 수 있지만, update가 갈라지는 순간 versioned fork가 필요하다”일 가능성이 높다.
- **[O]** 다중 tenant LLM serving에서 hit rate, fork depth, state deduplication, prefix-aware batching을 측정한 결과는 없다.

### 우리가 이어갈 기술 follow-up / 가설

- **[H] State Cache Key:**

  $$K_{state}=H(\text{model-ver},\text{slow-weight-ver},\text{prefix},\text{chunking},\text{update-rule},\text{state-schema},\text{tenant-policy}).$$

- **[H]** mutable state를 immutable base snapshot + per-session delta log로 표현하면 prefix reuse와 personalization을 동시에 얻을 수 있다.
- **[H]** update frequency가 낮을수록 shared-state hit rate가 증가하므로, 정확도뿐 아니라 재사용률을 포함해 cadence를 학습해야 한다.
- **디바이스 연결:** state snapshot dedupe, hardware copy-on-write, version-tagged DMA, delta merge를 지원하는 memory appliance/near-memory engine을 제안한다.

### 답변에 따른 branch

- **세션 전용이라고 하면:** state 크기 절감만으로도 이득이 나는 crossover를 KQ-03에서 수치화한다.
- **공유 가능하다고 하면:** 정확한 invalidation 조건, deterministic update, RNG·precision 차이에 대한 equivalence test를 묻는다.
- **checkpoint를 외부 store에 둔다고 하면:** selection metadata 상주 위치와 HBM miss latency를 묻고 SSC device path로 연결한다.
- **delta adapter 형태를 선호하면:** consolidation 이후 base/delta 병합, rollback, privacy isolation을 BQ-06으로 확장한다.

### 확인할 근거·식·그림

- MC §3.4: 이어지는 checkpoint 대 독립 compressor.
- MC Eq. 16·17, Fig. 2: summary 기반 sparse selection.
- NSTM §3.1 및 Table 5: update 58.14 ms, apply 27.01 ms, amortized 28.1 ms/frame의 특정 실험.
- 미팅에서 요청할 추가 근거: prefix popularity 분포에 따른 state-cache hit rate, fork 수, copy byte, p99 latency.

---

## KQ-03. Prefill·decode·update를 분리하면 실제 crossover는 어디에 생기는가?

**커버 축:** 2, 3 / efficient pre-training, prefill, decode, update, roofline

### 문제 framing

Attention의 이차 FLOP와 KV 용량만 비교하면 recurrent/TTT가 유리해 보이지만 실제 장치에서는 다른 항이 지배할 수 있다. HOPE는 chunk 내부 병렬화가 가능해도 chunk 경계마다 mutable state read–modify–write가 발생한다. MC/SSC는 checkpoint를 offload할 수 있지만 router와 selected-state load가 필요하다. Decode에서는 batch·state locality·prefix reuse가 prefill과 전혀 다르다.

### 실제 질문 문장

> “The asymptotic advantage is clear, but for deployment we think the relevant comparison is a phase-separated roofline: prefill compute, decode bandwidth, state-update RMW, routing, and state migration. Where do you expect the real crossover among FlashAttention, fixed-state recurrence, Memory Caching/SSC, and self-modifying HOPE once those terms are measured separately?”

### 왜 kick인가

- “RNN은 $O(L)$이라 빠르다”는 수준을 넘어 kernel·HBM·state migration의 실제 경계를 묻는다.
- SAIT의 device solution을 자연스럽게 제안할 수 있는 가장 직접적인 질문이다.
- 상대가 공개하지 않은 kernel 가정과 병목을 설명하도록 유도한다.

### Google 논문 기반 예상 답변

- **[D]** HOPE는 sequence를 chunk로 나누고 현재 chunk의 key/value/rate/gradient를 병렬 계산하는 dual/chunk-wise form을 사용한다(NL §8.2).
- **[D]** CMS는 해당 cadence에 도달한 block만 갱신하며, chunk 내부 token은 병렬 처리될 수 있다고 주장한다(NL §7.1).
- **[D]** MC는 update는 $O(L)$로 유지하지만 모든 cache를 읽으면 retrieval이 $O(NL)$이다. SSC는 Top-k만 불러 memory 소비를 낮춘다(MC §3.1·3.3).
- **[E]** MC 논문의 throughput 결과는 Transformer와 base RNN 사이의 중간 지대를 보인다. NSTM은 update/apply의 비대칭을 실측하지만 비디오·단일 H100·특정 해상도 조건이다.
- **[O]** 네 논문 모두 LLM serving에서 prefill/decode/update를 통일된 kernel·동일 hardware·동일 quality로 비교한 roofline은 제공하지 않는다.

### 우리가 이어갈 기술 follow-up / 가설

- **[H] Phase-separated latency:**

  $$T_{total}=T_{prefill}+T_{decode}+n_uT_{update}+T_{route}+T_{migrate},$$

  $$T_i\ge \max\left(\frac{F_i}{\Pi_{eff}},\frac{R_i+W_i}{BW_{eff}}\right)+T_{launch,i}.$$

- **[H]** self-modifying memory의 핵심 병목은 arithmetic FLOP보다 chunk마다 발생하는 state RMW와 synchronization일 가능성이 높다.
- **[H]** SSC는 selected memory가 HBM에 이미 있을 때와 host/SSD에서 올라올 때가 완전히 다른 regime이므로 “Top-k” 외에 **bytes selected × miss latency**가 scaling 변수여야 한다.
- **디바이스 연결:** small-state GEMM/outer-product fusion, state-local SRAM, asynchronous checkpoint prefetch, update engine, state compression/decompression을 공동 제안한다.

### 답변에 따른 branch

- **compute-bound라고 하면:** 어떤 tensor-core shape와 fusion이 가능한지, state update가 main GEMM과 겹치는지 묻는다.
- **bandwidth/RMW-bound라고 하면:** state residency와 near-memory update가 주는 이득을 바로 제안한다.
- **router/offload-bound라고 하면:** prediction accuracy가 아니라 prefetch lead time과 miss penalty를 포함한 router loss를 제안한다.
- **training은 빠르지만 decode가 불확실하다고 하면:** BQ-05로 넘어가 phase별 benchmark 공동 정의를 제안한다.

### 확인할 근거·식·그림

- NL §7.1 Eq. 71, §8.2 chunk-wise Eq. 90 전후, M3 efficiency Fig. 12.
- MC Eq. 5·17, Fig. 4(training throughput).
- NSTM Table 5와 §4.4.
- 미팅에서 요청할 추가 근거: profiler의 physical DRAM bytes, L2 hit, kernel launch, state RMW bytes, prefill/decode tokens/s, p99.

**정량 기준선:** 저장소의 [E5 deterministic results](../../experiments/E5-attn-vs-hope/results.json)는 phase-separated analytical lower bound를, [user-reviewed native live-Sheet archive](../attn-vs-hope/README.md)는 미팅에서 검토한 운영 모델과 그 검증 계약을 보존한다. 두 artifact의 목적과 provenance를 섞지 않는다.

---

## KQ-04. 기억이 계속 늘어날 때, 성장은 필수인가 아니면 bounded policy로 대체 가능한가?

**커버 축:** 4, 5 / capacity, forgetting, cache growth, parameter growth, pruning

### 문제 framing

세 논문은 서로 다른 방식으로 capacity를 늘린다. MC는 checkpoint 수를 늘리고, Sleep은 low-rank expert를 추가·활성화하며, full attention은 KV를 늘린다. 반면 NSTM은 고정 크기 fast weight와 평균 cache로 안정화하지만 분 스케일에서만 평가했고 finite capacity를 한계로 적시한다. 결국 무한한 경험에서 memory가 단조 증가한다면 어느 방식도 그대로는 끝나지 않는다.

### 실제 질문 문장

> “If catastrophic forgetting is fundamentally a consequence of compression into finite capacity, do you view memory growth as necessary for lifelong learning, or can a bounded system achieve a controlled loss through learned consolidation, eviction, and abstraction? What invariant should decide that a memory can be merged or deleted?”

### 왜 kick인가

- 논문들이 직접 인정하는 미해결점을 정확히 찌르면서도 “성장 설계가 틀렸다”는 공격으로 들리지 않는다.
- parameter growth, checkpoint growth, pruning, external offload를 하나의 capacity budget 문제로 통합한다.
- storage-class memory, memory tiering, near-data consolidation이라는 device proposal로 곧바로 연결된다.

### Google 논문 기반 예상 답변

- **[D]** NL 결론은 파국적 망각을 제한된 capacity에 의한 compression의 자연스러운 결과로 규정하며 일반적으로 해결되지 않았다고 명시한다.
- **[D]** Sleep은 새 low-rank expert를 더 느린 memory block에 추가하고, consolidation 뒤 빠른 block의 과거 low-rank parameter를 reset한다. 구현상 처음부터 masked capacity를 둘 수도 있다고 제안한다(Sleep §3.2·3.3).
- **[D]** MC는 checkpoint 수와 segment scheme으로 capacity–cost를 조절하며 logarithmic segmentation은 cache 수를 $O(\log L)$로 줄이는 방향을 논한다(MC §4.2).
- **[D]** NSTM은 finite capacity, learned cache weighting, primacy bias를 후속 과제로 직접 적는다(NSTM §5).
- **[I]** Google 측은 bounded memory만으로 lossless lifelong learning은 어렵지만, task utility를 기준으로 한 lossy abstraction·growth·reset의 조합은 가능하다고 답할 가능성이 높다.
- **[O]** 보존해야 할 정보의 invariant, global capacity budget, expert/cache eviction policy, 장기간 성장 곡선은 확립되지 않았다.

### 우리가 이어갈 기술 follow-up / 가설

- **[H] Memory Utility Density:** 각 memory object $m$에 대해

  $$U(m)=\frac{\mathbb{E}[\Delta Q\mid m]\cdot p_{reuse}(m)-\lambda I(m)-\mu C_{read}(m)}{\mathrm{bytes}(m)}$$

  를 추정해 retain/merge/offload/delete를 결정한다. $I(m)$은 interference risk다.
- **[H]** episodic external memory는 provenance와 exact recall을, parametric memory는 반복적으로 재사용되는 semantic abstraction을 담당해야 한다. 승격 기준은 access frequency 하나가 아니라 compressibility·cross-task reuse·interference·deletion requirement의 함수여야 한다.
- **[H]** 완전 삭제 전에 raw token → checkpoint → adapter/expert → shared weight 순으로 abstraction ladder를 두고, 각 전환에서 reconstruction·behavioral invariance test를 수행한다.
- **디바이스 연결:** hot expert/checkpoint는 HBM, warm object는 CXL/host memory, cold episodic source는 SSD에 두고 consolidation engine이 tier 간 이동을 수행한다.

### 답변에 따른 branch

- **성장이 필수라고 하면:** 성장률 상한, router cost, checkpoint/expert fragmentation, inactive parameter의 physical storage를 묻는다.
- **bounded 가능하다고 하면:** 어떤 forgetting budget과 검증 set으로 삭제를 승인하는지 묻는다.
- **external memory로 넘긴다고 하면:** parametric abstraction과 external source의 양방향 consistency·version을 BQ-06으로 연결한다.
- **learned weighting을 강조하면:** NSTM의 primacy bias와 MC router miss를 함께 다루는 BQ-02로 이동한다.

### 확인할 근거·식·그림

- NL §10 “Is Catastrophic Forgetting Solved?”.
- Sleep Fig. 2, §3.2 parameter expansion, §3.3 reset/pruning 구현 주석.
- MC §4.2 logarithmic segmentation, Fig. 3.
- NSTM §5 limitations, Fig. 8.
- 요청할 추가 근거: wake token 수 대비 live parameter/checkpoint byte, retained task quality, merge/delete error, growth slope.

---

## KQ-05. Wake에서 Sleep으로 넘겨야 할 최소 충분 artifact는 무엇인가?

**커버 축:** 4, 5, 6 / wake–sleep contract, parametric vs external memory, consolidation

### 문제 framing

Sleep을 별도 cluster에서 수행하려면 wake cluster가 무엇을 넘길지 결정해야 한다. Raw transcript를 넘기면 privacy·bandwidth·저장 비용이 크다. Hidden state나 gradient만 넘기면 provenance와 재검증이 어려워진다. Model-generated replay만 사용하면 오류가 자기강화될 수 있다. 외부 text/vector/graph memory는 exact source를 남길 수 있지만 매 inference retrieval 비용이 든다.

### 실제 질문 문장

> “What is the minimal sufficient artifact that should cross the wake-to-sleep boundary: raw trajectories, selected text, hidden states, gradients, fast-weight checkpoints, or model-generated samples? And how would you decide which memories remain externally addressable versus being consolidated into parameters?”

### 왜 kick인가

- Sleep 논문의 학습 알고리즘을 data plane·memory plane의 실제 계약으로 바꾼다.
- parametric memory와 text/vector/graph memory를 경쟁재가 아니라 계층적 역할로 재정의한다.
- privacy, provenance, network bytes, re-training cost를 동시에 논의할 수 있다.

### Google 논문 기반 예상 답변

- **[D]** Sleep은 외부 데이터 접근이 제한된 상태를 가정하고, teacher/self가 생성한 data와 student on-policy sample을 혼합해 Knowledge Seeding을 수행한다(Sleep §3.3).
- **[D]** KS optimization 중에는 기존 student parameter를 freeze하고 새 expanded parameter만 학습한다. 그 뒤 미리 계산한 sender base-weight update를 적용하고, 더 빠른 block의 과거 low-rank parameter를 reset하며 새 slow expert를 activate한다.
- **[D]** Dreaming은 synthetic sample을 만들고 gradient-based importance로 Top-k와 일부 random sample을 선택하며, random expert를 통해 novelty를 주입한다(Sleep §3.4).
- **[D]** Sleep 논문은 NL/HOPE의 online consolidation과 자신이 제안한 offline consolidation을 서로 보완적인 과정으로 구분한다.
- **[I]** Google 측은 한 종류의 artifact보다 “teacher behavior를 재현할 수 있는 sampled curriculum + fast/slow parameter state”를 강조할 가능성이 높다.
- **[O]** raw evidence provenance, deletion, adversarial memory, privacy-preserving replay, external store와 consolidated weight의 consistency protocol은 다루지 않는다.

### 우리가 이어갈 기술 follow-up / 가설

- **[H] Wake-to-Sleep Capsule:** 각 경험을 다음 구조로 넘긴다.

  $$c=(\text{source-ref},\text{compressed episode},\text{state-delta},\text{uncertainty},\text{utility},\text{policy tags}).$$

  source-ref는 원문·권한·삭제 ID를 유지하고, state-delta는 학습 효율을 높인다.
- **[H]** sleep teacher는 parametric self만이 아니라 provenance가 있는 external episodic memory를 privileged context로 사용하고, student는 그 context 없이 행동을 재현한다. 이때 external memory는 ground truth와 rollback anchor 역할을 한다.
- **[H]** consolidation 승인 조건을 behavioral equivalence + source attribution + forgetting budget의 삼중 gate로 둔다.
- **디바이스 연결:** capsule 생성은 wake node에서 near-data 압축, sleep node에는 delta/selected sample만 전송해 inter-cluster byte를 줄인다.

### 답변에 따른 branch

- **synthetic replay만 충분하다고 하면:** epistemic drift와 source hallucination을 어떻게 감지하는지 BQ-07로 이어간다.
- **raw replay가 필요하다고 하면:** retention/privacy 비용과 on-device selection을 제안한다.
- **fast-weight checkpoint가 핵심이라고 하면:** checkpoint schema/version과 cross-device migration을 KQ-02로 연결한다.
- **external retrieval을 장기적으로 유지한다고 하면:** 어떤 access frequency에서 parametric consolidation이 경제적인지 KQ-06의 scaling objective로 묻는다.

### 확인할 근거·식·그림

- Sleep §3.3 on-policy distillation, Eq. 3·4 reward, Knowledge Seeding objective; §3.4 Dreaming Eq. 5.
- Sleep Fig. 2, Fig. 4, Table 3.
- NL §6 knowledge transfer, §7 CMS, Sleep §3.1 wake/offline consolidation.
- 요청할 추가 근거: artifact별 bytes, sleep convergence, source-faithfulness, delete/unlearn latency.

---

## KQ-06. Sleep-time compute를 독립 scaling 축으로 만들 수 있는가?

**커버 축:** 6 / scaling law, scheduler, marginal utility, infrastructure

### 문제 framing

기존 scaling은 주로 pre-training FLOP, parameter, data, inference-time compute를 다룬다. Sleep은 동일한 wake latency budget 아래에서도 유휴 시간에 consolidation·dreaming·parameter growth를 추가할 수 있는 새로운 축이다. 그러나 “sleep 단계가 많을수록 좋아졌다”는 실험과 “예측 가능한 scaling law가 있다”는 주장은 다르다. 필요한 것은 sleep FLOP, replay entropy, plastic capacity, frequency hierarchy, migration byte, forgetting을 함께 설명하는 법칙이다.

### 실제 질문 문장

> “Do you think sleep-time compute can become a predictable scaling axis, analogous to pre-training or test-time compute? If so, what variables should the law depend on—sleep FLOPs, newly activated capacity, replay diversity, consolidation depth, wake/sleep ratio, or state-migration bytes—and what signal should trigger the next sleep cycle?”

### 왜 kick인가

- 논문의 가장 미래지향적인 메시지를 정량 과학과 infra scheduling 문제로 끌어올린다.
- 답이 아직 없더라도 공동 연구 agenda를 우리가 주도할 수 있다.
- accelerator 수요를 “학습용/추론용” 두 종류가 아니라 wake–sleep–memory plane으로 재정의한다.

### Google 논문 기반 예상 답변

- **[E]** Sleep은 평가한 long-context task에서 consolidation level 수가 늘수록 성능이 좋아지고, 가장 느린 memory의 update frequency를 높이면 retention이 약화되는 경향을 보고한다(Sleep Fig. 4).
- **[E]** 새로운 언어 순차 학습에서도 HOPE-1→3에 따라 결과가 개선됐으며, 특정 reasoning 목표 성능까지의 wall-clock 비교에서는 SFT보다 Sleep이 유리했다고 보고한다.
- **[D]** Sleep은 consolidation과 dreaming을 분리하고, parameter growth와 reset을 주기적으로 수행한다.
- **[I]** Google 측은 현재 결과를 proof of concept로 보고, 보편 scaling law보다는 task·memory frequency·capacity에 조건된 empirical law가 먼저 필요하다고 답할 가능성이 높다.
- **[O]** 장기간 반복 sleep의 saturation, self-distillation collapse, energy/byte-normalized law, optimal trigger는 검증되지 않았다.

### 우리가 이어갈 기술 follow-up / 가설

- **[H] Sleep Utility Objective:**

  $$\max_{\pi_s}\;\Delta Q-\lambda C_{sleep}-\mu\Delta F-\nu B_{migrate}-\rho P_{growth},$$

  여기서 $\Delta Q$는 새 지식·추론 향상, $\Delta F$는 기존 능력 망각, $B_{migrate}$는 wake/sleep/memory plane 사이 이동 byte, $P_{growth}$는 활성 parameter 증가량이다.
- **[H] Conditional scaling candidate:**

  $$\Delta Q\approx A C_{sleep}^{\alpha}P_{plastic}^{\beta}H_{replay}^{\gamma}D_{cons}^{\delta}-B I^{\zeta},$$

  단, 모든 지수는 task·memory tier·wake distribution에 조건된다. $I$는 interference다.
- **[H] Sleep trigger:** fast-memory saturation, retrieval miss, gradient conflict, uncertainty, stale-knowledge score, idle-energy price를 결합한 event-driven scheduler가 고정 주기보다 낫다.
- **디바이스 연결:** foreground inference SLA와 background sleep throughput을 동시에 만족하는 heterogeneous cluster, low-priority preemptible kernels, state-locality-aware placement를 제안한다.

### 답변에 따른 branch

- **FLOP law가 가능하다고 하면:** quality를 어떤 단일 metric으로 정규화할지, growth와 replay를 어떻게 통제할지 묻는다.
- **task-dependent라 하면:** 공통 독립변수와 최소 benchmark suite를 BQ-08에서 공동 정의한다.
- **sleep trigger가 saturation 기반이라 하면:** online observable과 false-trigger cost를 묻고 BQ-01로 이동한다.
- **infra가 주 병목이라고 하면:** byte·energy·latency 포함 joint scaling 실험을 제안한다.

### 확인할 근거·식·그림

- Sleep Fig. 4(level 수와 lowest frequency), Fig. 5(순차 언어), Appendix B.5 efficiency.
- NL Fig. 10(context usage), Fig. 12(M3 training overhead), §10 open capacity statement.
- 공동 실험으로 요구할 plot: quality 대 sleep FLOP, active parameter, replay token, migration byte, energy를 각각 log-scale로 그린 iso-quality surface.

---

# Part II. Backup 8 Questions

## BQ-01. 고정 주기 update는 어떤 사건을 놓치며, event-triggered memorization이 가능한가?

**커버 축:** 3, 5, 6 / update cadence, saliency, missed events

### 문제 framing

NSTM은 update와 apply를 분리해 실시간성을 얻지만, update 사이에만 등장한 사건을 놓칠 수 있다고 직접 인정한다. 언어·agent에서도 모든 token을 update하면 비싸고, 고정 chunk만 update하면 중요한 rare event를 놓친다.

### 질문 문장

> “Once write frequency is decoupled from read frequency, how should the system detect a rare but high-value event that cannot wait for the next periodic update? Would you prefer learned event-triggered writes, adaptive chunking, or a small episodic buffer that is consolidated later?”

### 왜 kick인가

read/write decoupling의 성공 조건을 “평균 latency”에서 “정보 누락 위험”으로 확장한다.

### 예상 답변

- **[D]** NSTM은 missed events를 한계로 적고, multi-frame large-chunk update와 time embedding을 후속 방향으로 제안한다.
- **[D]** HOPE/CMS는 고정된 chunk·frequency schedule을 사용한다.
- **[I]** learned saliency 또는 short episodic buffer와 periodic update의 혼합이 자연스러운 답이다.
- **[O]** trigger의 false negative/positive 비용은 정량화되지 않았다.

### 우리 follow-up / 가설

- **[H]** update trigger를 surprise, uncertainty, novelty, expected future reuse의 조합으로 둔다.
- **[H]** rare-event는 작은 append-only external buffer에 먼저 기록하고, sleep에서 fast weight/parameter로 승격한다.
- **디바이스 연결:** always-on low-power saliency detector가 expensive update engine을 깨우는 구조.

### branch

- adaptive chunking이면 chunk-size 분산과 batching 손실을 묻는다.
- episodic buffer이면 overflow/priority 정책을 KQ-04로 연결한다.

### 확인 근거

- NSTM §5, Fig. 8.
- 추가 요청: event duration×update stride별 miss curve와 amortized cost.

---

## BQ-02. 어떤 parameter/state 공간에서 평균·soup·merge가 의미를 보존하는가?

**커버 축:** 2, 5 / caching geometry, invariant factorization, primacy bias

### 문제 framing

MC의 residual/soup, NSTM의 running arithmetic mean은 과거 state를 결합한다. 그러나 NSTM ablation은 pose와 identity가 한 fast weight에 섞이면 평균이 붕괴하고, invariant identity를 memory에 분리했을 때만 caching이 작동함을 보여준다.

### 질문 문장

> “Memory averaging works only when the cached states live in a compatible geometry. What representation-level condition tells us that two fast-weight states can be averaged or souped without destroying meaning, and how would you learn that condition rather than assume it?”

### 왜 kick인가

단순 cache 기법을 representation identifiability와 merge safety 문제로 격상한다.

### 예상 답변

- **[D]** MC에서 linear memory의 soup와 gated residual은 동치지만 deep/nonlinear memory에서는 다르다(MC §3.2).
- **[E]** NSTM에서는 decoupled invariant memory일 때 caching이 이득이고 coupled LaCT에서는 성능이 붕괴한다(Table 2, Fig. 5).
- **[D]** NSTM은 균일 평균의 primacy bias도 한계로 적는다.
- **[O]** merge 가능성의 일반 판정식은 없다.

### 우리 follow-up / 가설

- **[H]** 두 state의 functional interpolation loss와 local curvature를 merge gate로 사용한다.
- **[H]** raw parameter 평균 대신 query-conditioned function-space barycenter 또는 low-rank delta merge를 사용한다.
- **디바이스 연결:** merge 전 sample probe를 실행하는 near-memory validation engine.

### branch

- function-space 평가를 선호하면 probe set의 privacy/비용을 묻는다.
- learned router를 선호하면 router state의 장기 drift와 calibration을 묻는다.

### 확인 근거

- MC Eq. 14–15, §3.2.
- NSTM §3.3, Table 2, Fig. 5·8.

---

## BQ-03. 모델이 정보를 “사용했다”가 아니라 memory에 “실제로 넣었다”는 것을 어떻게 증명할까?

**커버 축:** 1, 4 / memory supervision, shortcut prevention

### 문제 framing

장문맥 모델은 현재 context·attention shortcut으로 답을 내고도 memory module이 학습된 것처럼 보일 수 있다. NSTM은 current-input attention을 차단한 memory-only loss로 이 문제를 다룬다. LLM에서도 비슷한 causal test가 필요하다.

### 질문 문장

> “How should we causally verify that information has actually been internalized into a neural memory rather than bypassed through the current context or another module? Do you see a language-model analogue of NSTM’s memory-only supervision as necessary?”

### 왜 kick인가

benchmark 향상을 memory causality 검증으로 바꾸며, module attribution에 대한 공동 실험을 연다.

### 예상 답변

- **[D]** NSTM의 $\mathcal L_{mem}$은 current input 경로를 mask하고 memory만으로 복원하게 한다.
- **[E]** 이를 제거하면 성능이 낮아진다(NSTM Table 2, Fig. 5).
- **[I]** 언어에서는 context ablation, delayed no-context query, module intervention이 대응 방법일 수 있다.
- **[O]** HOPE/Sleep에서 각 frequency block이 무엇을 실제 저장했는지 인과적으로 분해한 평가는 부족하다.

### 우리 follow-up / 가설

- **[H]** write 후 source context를 제거하고, target memory tier만 남기는 tier-isolation test를 표준화한다.
- **[H]** memory lesion, swap, stale-version injection으로 retrieval 성공과 internalization을 분리한다.
- **디바이스 연결:** tier별 enable/disable과 checksum을 지원하는 debug/telemetry path.

### branch

- behavior-only validation이면 provenance 문제를 BQ-06으로 연결한다.
- representation probe를 제안하면 probe가 실제 causality를 보장하는지 묻는다.

### 확인 근거

- NSTM Eq. 5, §3.2, Fig. 2, Table 2.

---

## BQ-04. CMS의 update frequency는 수동 hyperparameter인가, 학습해야 할 memory policy인가?

**커버 축:** 4, 5 / stability–plasticity, multi-timescale scheduling

### 문제 framing

CMS는 block마다 다른 chunk와 update frequency를 둔다. 느릴수록 persistent하지만 덜 adaptive하다. 실제 workload에서는 지식의 반감기와 중요도가 content마다 다르므로 고정 cadence가 최적이라는 보장은 없다.

### 질문 문장

> “Should memory frequency remain a block-level hyperparameter, or should the model learn a content- and workload-dependent consolidation clock? How would you prevent a learned scheduler from collapsing to either over-writing everything quickly or freezing everything permanently?”

### 왜 kick인가

CMS의 핵심 설계축을 adaptive memory OS 문제로 연결한다.

### 예상 답변

- **[D]** NL/Sleep은 명시적 frequency/chunk hierarchy를 사용하고 frequency가 persistence–adaptability를 제어한다고 본다.
- **[E]** Sleep 실험에서 more levels와 lower minimum frequency가 retention에 유리한 경향을 보인다.
- **[O]** content-conditioned learned clock과 안정성 제약은 직접 다루지 않는다.

### 우리 follow-up / 가설

- **[H]** scheduler objective에 forgetting, write cost, reuse, staleness를 함께 넣고 최소/최대 cadence constraint를 둔다.
- **[H]** block cadence가 아니라 memory object별 TTL과 consolidation priority를 학습한다.
- **디바이스 연결:** workload telemetry가 sleep scheduler와 memory tier를 공동 제어한다.

### branch

- learned frequency에 동의하면 label/reward와 differentiability를 묻는다.
- fixed hierarchy를 선호하면 workload shift 시 재튜닝 비용을 묻는다.

### 확인 근거

- NL Eq. 71, §7.1.
- Sleep Eq. 2, Fig. 4.

---

## BQ-05. “훈련 병렬화 가능”과 “대규모 pre-training에서 경제적” 사이의 빈틈은 무엇인가?

**커버 축:** 3 / efficient pre-training, optimizer state, synchronization

### 문제 framing

HOPE는 chunk dual form으로 sequence parallelism을 얻지만 auxiliary memory forward/backward, state update, outer-loop gradient가 추가된다. M3도 더 좋은 loss를 보이나 Muon보다 느리다고 보고한다. 병렬화 가능성만으로 wall-clock·energy 효율이 보장되지는 않는다.

### 질문 문장

> “Which part of HOPE’s nested update remains the scaling bottleneck in large pre-training: auxiliary-memory forward/backward, chunk-boundary synchronization, optimizer state, or outer-loop credit assignment? What result would convince you that the method is economically competitive, not only parallelizable?”

### 왜 kick인가

알고리즘적 parallel form과 시스템 경제성을 구분한다.

### 예상 답변

- **[D]** NL은 chunk 시작 전 필요한 값을 병렬 생성하고 previous-chunk state에 대한 gradient를 사용한다.
- **[D/E]** CMS는 예정 block만 update하지만 M3는 multi-momentum 때문에 Muon보다 느리고 AdaMuon과 비슷하다고 보고한다.
- **[O]** HOPE full pre-training의 energy/token, optimizer-memory byte, scaling efficiency는 충분히 제시되지 않았다.

### 우리 follow-up / 가설

- **[H]** nested update를 recompute, low-precision state, delayed update, fused local backward 네 축으로 분해한다.
- **[H]** quality-matched joule/token과 time-to-quality를 주 지표로 둔다.
- **디바이스 연결:** small-gradient engine, state RMW fusion, memory-local optimizer.

### branch

- synchronization이 병목이면 larger chunk의 quality trade-off를 묻는다.
- state memory가 병목이면 precision·compression 민감도를 묻는다.

### 확인 근거

- NL §8.2, §9.7, Fig. 12.
- Sleep Appendix B.5 time-to-quality.

---

## BQ-06. Parametric memory의 provenance·삭제·버전 관리는 어떻게 할 것인가?

**커버 축:** 2, 4, 5 / provenance, unlearning, security, multi-tenant isolation

### 문제 framing

External text/vector/graph memory는 출처와 삭제 대상을 추적하기 쉽다. 지식이 fast weight·adapter·slow weight로 들어가면 어느 source가 어떤 behavior에 기여했는지 흐려진다. 개인화 state가 shared consolidation에 섞이면 보안·규제 문제가 커진다.

### 질문 문장

> “As memories move from externally addressable records into fast weights and eventually shared parameters, what provenance and deletion contract should survive each consolidation step? Would you keep a reversible delta lineage, or treat parametric consolidation as intentionally lossy and non-addressable?”

### 왜 kick인가

학습 성능 논의에서 빠지기 쉬운 production 필수조건을 제기한다.

### 예상 답변

- **[D]** Sleep은 KS optimization 동안 기존 student parameter를 freeze하고 새 expanded expert만 학습한 뒤, sender update·fast-block expert reset·new slow-expert activation을 commit한다.
- **[I]** MC checkpoint와 NSTM snapshot은 구현상 versionable object로 다루기 자연스럽다.
- **[O]** source-level lineage, right-to-forget, poisoned memory rollback은 직접 다루지 않는다.

### 우리 follow-up / 가설

- **[H]** consolidation마다 teacher source ID, training capsule hash, parameter delta, validation suite를 묶은 memory commit을 만든다.
- **[H]** shared base에는 k-anonymous/organization-approved memory만 승격하고 개인 memory는 tenant delta로 유지한다.
- **디바이스 연결:** encrypted tenant state, hardware version tag, delta rollback, secure erase.

### branch

- reversible delta를 선호하면 storage amplification을 KQ-04로 연결한다.
- irreversible abstraction을 허용하면 deletion SLA와 risk boundary를 묻는다.

### 확인 근거

- Sleep §3.2·3.3 reset/freeze.
- MC checkpoint definition, NSTM snapshot recursion.
- 추가 요청: consolidation commit/rollback prototype과 poisoning test.

---

## BQ-07. Dreaming이 새 지식을 만드는가, 오류와 편향을 증폭하는가?

**커버 축:** 4, 6 / synthetic data, self-distillation collapse, safety

### 문제 framing

Dreaming은 external label 없이 synthetic curriculum을 만들지만 자기생성 data를 반복 학습하면 distribution collapse, epistemic overconfidence, hidden bias reinforcement가 생길 수 있다. Random expert를 통한 novelty는 다양성과 오류를 동시에 늘릴 수 있다.

### 질문 문장

> “What prevents repeated dreaming from turning consolidation into a self-reinforcing error loop? How would you separate useful novel synthesis from hallucinated novelty, especially when the teacher and the reward model share the same blind spots?”

### 왜 kick인가

Sleep scaling의 안전성과 장기 안정성을 검증하는 핵심 질문이다.

### 예상 답변

- **[D]** Sleep은 consolidation을 dreaming보다 먼저 수행하고, gradient importance Top-k + random sample, downstream-improvement reward를 사용한다.
- **[D]** 관련 연구 논의는 반복 self-distillation의 information leakage·collapse·OOD degradation 위험을 인정한다.
- **[E]** 논문 실험 범위에서는 two-stage design과 ablation이 이점을 보인다.
- **[O]** 수백·수천 cycle의 장기 collapse와 shared-bias reward hacking은 검증되지 않았다.

### 우리 follow-up / 가설

- **[H]** external episodic evidence와 독립 verifier를 dream acceptance gate로 사용한다.
- **[H]** replay entropy, source coverage, calibration, old-skill regression을 동시에 감시하고 하나라도 임계치를 넘으면 rollback한다.
- **디바이스 연결:** cheap speculative dream generation과 expensive verification을 서로 다른 accelerator tier에 배치한다.

### branch

- verifier를 강조하면 verifier independence와 cost를 묻는다.
- downstream reward를 강조하면 sparse reward의 gaming과 held-out leakage를 묻는다.

### 확인 근거

- Sleep §3.4 Eq. 5, ablation Table 3, related-work의 OPSD limitations.
- 요청할 추가 근거: sleep cycle 수에 따른 calibration·diversity·forgetting·collapse curve.

---

## BQ-08. Retrieval benchmark를 넘어 wake–sleep memory system 전체를 어떻게 평가할 것인가?

**커버 축:** 1, 3, 4, 6 / benchmark, phase separation, longitudinal evaluation

### 문제 framing

NIAH는 exact retrieval, perplexity는 평균 예측, short ICR은 local recall, NSTM은 분 스케일 visual recall을 본다. Lifelong memory system은 retention, update latency, old-skill interference, source fidelity, serving reuse, sleep cost를 동시에 평가해야 한다.

### 질문 문장

> “What would a convincing end-to-end benchmark for a wake–sleep memory system look like? We think it must jointly measure long-context reasoning, state reuse, write/read latency, forgetting, provenance, capacity growth, and sleep cost—not one metric at a time. Which failure mode should be the primary axis?”

### 왜 kick인가

논문별 score 비교를 넘어 공동 benchmark·공동 시스템 연구로 미팅을 닫을 수 있다.

### 예상 답변

- **[D/E]** 네 논문은 NIAH, BABILong, LongHealth, continual translation, knowledge incorporation, language modeling, minute-scale NVS 등 서로 보완적인 evidence를 제공한다.
- **[O]** 동일 workload에서 wake latency, sleep FLOP, capacity byte, reuse hit rate, forgetting을 함께 측정하는 benchmark는 없다.
- **[I]** Google 측도 단계별 microbenchmark와 longitudinal task suite의 결합 필요성에는 동의할 가능성이 높다.

### 우리 follow-up / 가설

- **[H] 3×3 benchmark grid:** task는 retrieval/reasoning/continual skill, phase는 wake-read/wake-write/sleep-consolidate로 나눈다.
- 각 cell에서 quality, p50/p99, joule, HBM/DRAM byte, live memory byte, old-task delta를 기록한다.
- **[H]** 동일 quality constraint 아래 system cost Pareto를 핵심 산출물로 둔다.
- **디바이스 연결:** Samsung hardware counter와 Google model trace를 결합한 공동 measurement protocol.

### branch

- quality-first라면 최소 SLA를 고정하고 cost를 비교한다.
- cost-first라면 byte/energy budget을 고정하고 quality degradation을 비교한다.

### 확인 근거

- NL Fig. 10, Tables 1·3·6.
- Sleep Figs. 3–6, Appendix B.5.
- MC Figs. 4·5.
- NSTM Tables 2·5, Fig. 8.

---

# Part III. 예상 답변을 Device Solution으로 전환하는 공통 브리지

Google의 답변이 어느 방향이든 다음 다섯 개 solution primitive로 번역한다.

아래 다섯 항목은 답변을 분류하는 **routing taxonomy**다. 최종 공동 연구 제안의 canonical 단위는 device appendix의 [Package A–C](03-DEVICE-SOLUTION-BRIDGE.md#6-sait가-제시할-구체적-공동-연구-package)다.

| Google이 강조한 병목 | SAIT가 이어갈 문장 | 가능한 device/system primitive |
|---|---|---|
| Mutable-state RMW | “연산량보다 state 이동과 version fork가 지배한다면…” | state-local SRAM, near-memory update, delta/COW engine |
| Sparse checkpoint/expert routing | “선택된 state만 적시에 HBM으로 올리는 것이 핵심이라면…” | async prefetch, tiered memory, router-aware DMA |
| Sleep distillation throughput | “foreground SLA와 독립된 background 학습 plane이 필요하다면…” | preemptible sleep kernels, heterogeneous accelerator, low-priority queue |
| Capacity growth·pruning | “논리적 growth와 물리적 상주량을 분리하면…” | masked expert store, hot/warm/cold placement, compaction |
| Provenance·multi-tenancy | “memory가 weight로 이동해도 lineage가 필요하다면…” | encrypted tenant delta, version tag, snapshot/rollback |

### 우리가 피해야 할 과장

- HOPE·Sleep이 **파국적 망각을 해결했다**고 말하지 않는다. 논문도 해결되지 않았다고 명시한다.
- MC가 Transformer의 recall을 일반적으로 이겼다고 말하지 않는다. 짧은 ICR에서는 Transformer가 우세하다.
- NSTM의 분 스케일 visual result를 LLM의 수일·수개월 lifelong memory 증거로 확대하지 않는다.
- “chunk-wise parallelizable”을 “production kernel에서 빠르다”와 동일시하지 않는다.
- Sleep의 단계 수 증가 실험을 보편 scaling law로 부르지 않는다.
- query는 fixed $W_q$ projection으로 만들며 별도 query-side mutable memory나 비용 항을 두지 않는다.

---

# Part IV. 답변 기록용 한 장 템플릿

각 질문 후 아래 아홉 항목만 빠르게 채운다. 상세 tensor·capacity·traffic 필드는 [device appendix의 post-meeting schema](03-DEVICE-SOLUTION-BRIDGE.md#9-answer-capture-sheet-schema)에서 사후 보강한다.

| Live column | 입력 규칙 |
|---|---|
| `Question_ID` | canonical `KQ-01…06` 또는 `BQ-01…08` |
| `Device_Bridge_ID` | 사용한 경우 `DBQ-01…10`; 없으면 `NONE` |
| `Answer_Verbatim` | 핵심 원문 또는 faithful paraphrase |
| `Answer_Type` | paper fact / unpublished result / opinion / open |
| `State_Invariant_Bottleneck` | state 표현, correctness invariant, 지배 비용 |
| `Evidence_Shareability_Attribution` | 근거, 공유 가능 범위, 발언 attribution 상태 |
| `Allowed_Use` | public / joint-internal / SAIT-internal / do-not-cite |
| `Open_Issue_SAIT_Bridge` | 상대가 인정한 공백과 연결할 SAIT 가설 |
| `Owner_Next_Action` | Google/SAIT/joint + 자료·계산·prototype·후속 일정 |

미팅 종료 전에는 다음 한 문장으로 공동 후속을 제안한다.

> “If we can agree on the state contract and the phase-separated metrics, we would like to build a joint crossover study that maps model quality to physical bytes, update traffic, reuse rate, and sleep-time compute across the same workload.”

---

# Primary Source Index

1. Ali Behrouz et al., **Nested Learning: The Illusion of Deep Learning Architectures**, arXiv:2512.24695 — <https://arxiv.org/abs/2512.24695>
2. Ali Behrouz, Farnoosh Hashemi, Adel Javanmard, Vahab Mirrokni, **Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories**, arXiv:2606.03979v2 (2026-07-10) — <https://arxiv.org/abs/2606.03979v2>
3. Ali Behrouz et al., **Memory Caching: RNNs with Growing Memory**, arXiv:2602.24281 — <https://arxiv.org/abs/2602.24281>
4. Baback Elmieh et al., **Online Neural Space Time Memory for Dynamic Novel View Synthesis**, arXiv:2607.15271 — <https://arxiv.org/abs/2607.15271>

로컬 검증본:

- [`papers/2606.03979v2.txt`](../../papers/2606.03979v2.txt) — **arXiv v2 기준 line evidence**
- [`translations-kr/2512.24695-nested-learning-FULL.md`](../../translations-kr/2512.24695-nested-learning-FULL.md)
- [`translations-kr/2606.03979-sleep-FULL.md`](../../translations-kr/2606.03979-sleep-FULL.md) — arXiv v1 한국어 번역·참고용이며, Sleep 주장 검증과 line anchor에는 위 v2 TXT를 사용한다.
- [`translations-kr/2602.24281-memory-caching.md`](../../translations-kr/2602.24281-memory-caching.md)
- [`translations-kr/2607.15271-nstm.md`](../../translations-kr/2607.15271-nstm.md)
