# Sleep-Time Compute 계보와 개념 경계의 변화

- source freeze: 2026-08-05
- 범위: biological analogy, continual learning, LLM memory, agent product lifecycle, systems implication
- 날짜 주의: source registry에서 `YYYY-01-01`은 원문이 연도만 제공한 경우의 catalog normalization이며 실제 1월 1일 publication 주장을 뜻하지 않는다.

## 1. 1989–2016: 왜 안정성과 가소성을 동시에 얻기 어려운가

| 연도 | 사건 | 당시 풀고자 한 문제 | 오늘의 STC framing에 남긴 것 | 근거 |
|---:|---|---|---|---|
| 1989 | McCloskey & Cohen, *Catastrophic Interference in Connectionist Networks* | 한 network가 sequentially 새 association을 배우면 이전 association이 급격히 붕괴하는 현상 | “계속 쓰면 언젠가 capacity가 넘는다”보다 먼저, **새 write가 old behavior를 망가뜨리는 interference**가 문제라는 출발점 | `SRC-STC-0068` |
| 1995 | McClelland, McNaughton & O'Reilly, Complementary Learning Systems | fast episodic hippocampal learning과 slow statistical cortical learning이 왜 분리되는가 | wake/sleep metaphor의 가장 강한 basis. 단, ML 구현이 biological stage와 동일하다는 증거는 아님 | `SRC-STC-0001` |
| 1995 | Wake–Sleep family가 latent generative model 학습 용어를 확립 | recognition/generative weights를 alternating phase로 학습 | “sleep”이라는 이름의 선례지만, deployment 후 accumulated experience를 durable later-wake state로 바꾸는 현대 STC와는 lifecycle이 다름 | canonical historical context; 직접 STC evidence로는 미사용 |
| 2016 | Deep Generative Dual Memory Network | fast episodic memory와 slow semantic model 사이 replay | later “sleep replay” neural architecture의 직접 전조 | `SRC-STC-0005` |

### 경계 변화 A

초기 질문은 “sleep이 유용한가?”가 아니었다. 더 정확한 질문은 다음이었다.

```text
새 정보를 빨리 받아들이는 update rule
        vs
이미 얻은 기능을 안정적으로 보존하는 representation
```

STC는 이 stability–plasticity conflict를 **시간과 medium의 분리**로 풀려는 한 계열이다.

## 2. 2017–2021: continual learning의 operator family가 분화

| 연도 | 사건 | Operator | 강점 | 남은 한계 | 근거 |
|---:|---|---|---|---|---|
| 2017 | Elastic Weight Consolidation | 중요 weight에 quadratic penalty | raw replay 없이 old task 보호 | importance 근사, task boundary, plasticity 감소 | `SRC-STC-0002` |
| 2017 | Deep Generative Replay | generator가 old-like samples 생성 | 원본 exemplar 없이 rehearsal 가능 | generator 자체 forgetting과 recursive error | `SRC-STC-0003` |
| 2017 | Gradient Episodic Memory | episodic buffer gradient constraints | old task loss 증가를 직접 제한 | buffer cost와 task labels | `SRC-STC-0004` |
| 2018 | FearNet | short-term memory를 long-term memory로 consolidation | explicit wake/sleep-like dual memory | small vision benchmarks와 scheduled consolidation | `SRC-STC-0006` |
| 2018 | Progress & Compress | active column이 배우고 knowledge base로 distill | acquisition과 consolidation 분리 | repeated compression cost, teacher drift | `SRC-STC-0007` |
| 2020 | Brain-Inspired Replay | generative replay와 representation replay 비교 | replay의 biological analogy와 practical operator 연결 | benchmark scale와 data realism | `SRC-STC-0008` |
| 2020 | Retrieval-Augmented Generation | facts를 weight 대신 external index에서 가져옴 | freshness·provenance·capacity를 model 밖으로 이동 | retrieval miss, context cost, index lifecycle | `SRC-STC-0031` |

### 경계 변화 B

이 시기에 장기기억 문제는 네 operator family로 갈라졌다.

1. **regularize:** weight를 덜 바꾼다.
2. **replay:** old evidence를 다시 보여 준다.
3. **expand/isolate:** task별 capacity를 추가한다.
4. **externalize:** 사실을 model 밖에 두고 query 때 읽는다.

현대 STC는 이 중 replay/distillation/expansion을 background phase로 묶을 수 있지만, externalization은 경쟁 방식이자 system-of-record로 함께 사용된다.

## 3. 2022: computational sleep evidence가 한꺼번에 등장

| 날짜/연도 | 사건 | 직접 결과 | 과대해석하면 안 되는 것 | 근거 |
|---|---|---|---|---|
| 2022 | *Learning Cortical Representations through Perturbed and Adversarial Dreaming* | perturbed/adversarial dream sample로 cortical representation 학습을 보완 | LLM deployment lifecycle 또는 user memory capacity 증거는 아님 | `SRC-STC-0009` |
| 2022 | PNAS autonomous hippocampus–neocortex model | autonomous interaction/replay로 sleep-dependent consolidation을 모델링 | architecture scale과 language knowledge로 바로 일반화 불가 | `SRC-STC-0010` |
| 2022 | PLOS Computational Biology sleep prevents catastrophic forgetting | 두 task의 spiking neural network에서 sleep phase가 joint synaptic representation 형성에 도움 | “모든 model에서 zero forgetting” 또는 unbounded lifelong capacity 증거가 아님 | `SRC-STC-0011` |
| 2022 | Nature Communications sleep replay consolidation | replay가 old knowledge 보존과 new learning 능력에 영향을 줌 | replay cost·quality·capacity 문제를 제거하지 않음 | `SRC-STC-0012` |
| 2022 | LoRA | low-rank trainable delta로 parameter-efficient adaptation | low-rank라고 interference·deletion·adapter accumulation이 자동 해결되지는 않음 | `SRC-STC-0030` |
| 2022 | Memorizing Transformers | kNN-style external memory로 long-range token retrieval | semantic consolidation보다 retrieval substrate | `SRC-STC-0025` |
| 2022 | ROME | localized factual association edit | repeated edit interaction과 global consistency는 별도 문제 | `SRC-STC-0032` |

### 2022 논문들의 정확한 위치

이 네 computational sleep 논문은 “sleep-like replay가 forgetting을 줄일 수 있다”는 existence proof를 제공한다. 하지만 대부분 small network, few tasks, controlled distribution이다. 따라서 2022는 STC의 **mechanism precedent**이지 2026 LLM system의 **scaling-law proof**가 아니다.

## 4. 2023–2024: LLM 외부기억과 안전 문제가 전면화

| 연도 | 사건 | 의미 | 제한/반증 | 근거 |
|---:|---|---|---|---|
| 2023 | MemGPT | finite context를 virtual memory hierarchy로 관리 | memory를 OS-style paging 대상으로 framing | background consolidation보다 explicit tier management가 중심 | `SRC-STC-0013` |
| 2023 | LongMem | frozen backbone에 decoupled long-term memory network | model update 없이 긴 history 활용 | retrieval/storage consistency와 serving cost | `SRC-STC-0026` |
| 2023 | MEMIT | transformer memory를 mass edit | many edits를 한 번에 적용하는 parametric alternative | locality·interference·reversal이 장기 운영 경계 | `SRC-STC-0033` |
| 2024 | Nature model collapse | recursively generated data로 세대를 반복하면 distribution tail과 quality가 손상될 수 있음 | synthetic dream을 무제한 자기재생하면 안전하다는 가정 반박 | `SRC-STC-0041` |
| 2024 | AgentPoison | agent long-term memory/RAG knowledge base를 poisoning하는 backdoor | external memory가 governance 없이 안전하다는 가정 반박 | `SRC-STC-0042` |

### 경계 변화 C

이 시기부터 “기억을 어디에 둘까?”는 성능 문제만이 아니게 됐다.

```text
write authority → provenance → validation → publish → retrieval
                         ↘ delete / rollback / rebase
```

외부 memory는 weight보다 감사가 쉽지만 공격 표면이 존재하고, parametric memory는 retrieval miss가 없을 수 있지만 삭제와 rollback이 어렵다. 따라서 두 medium 모두 governance plane이 필요하다.

## 5. 2025: sleep-time compute가 명시적인 scaling axis로 이름 붙음

| 시기 | 사건 | 핵심 제안 | STC와의 관계 | 근거 |
|---|---|---|---|---|
| Jan 2025 | Titans | surprise-driven neural memory를 test time에 update | wake-path fast state; sleep과 인접하지만 동일하지 않음 | `SRC-STC-0017` |
| Jan 2025 | Zep paper lineage | temporal knowledge graph로 evolving agent memory | external temporal system-of-record | `SRC-STC-0016` |
| Apr 2025 | Sleep-Time Compute | anticipated future queries에 offline compute를 써서 later query cost/accuracy 개선 | explicit new axis: shared future-query preparation | `SRC-STC-0014` |
| Apr 2025 | MIRAS | test-time memorization을 online optimization의 공통 framework로 정리 | neural memory operator taxonomy | `SRC-STC-0018` |
| Apr 2025 | Mem0 paper lineage | production-oriented agent memory extraction/update/retrieval | external product alternative | `SRC-STC-0015` |
| May 2025 | ATLAS | context memorization과 learning rule을 최적화 | wake fast-state architecture frontier | `SRC-STC-0019` |
| Jun 2025 | Self-Adapting Language Models | model이 input을 활용해 self-adapt | parametric bridge; deployment safety remains open | `SRC-STC-0029` |
| Sep 2025 | ReasoningBank | success/failure trajectory를 reusable strategy로 distill | external structured consolidation and test-time self-evolution | `SRC-STC-0043` |
| Nov 2025 | TNT | test-time memorization을 transferable하게 학습 | learned update rule | `SRC-STC-0024` |
| Nov 2025 | Google Nested Learning release | optimizer와 architecture를 nested update frequency continuum으로 해석 | wake/sleep을 frequency hierarchy로 확장 | `SRC-STC-0055` |
| 2025 | Memory Layers at Scale | large external memory layer capacity | parametric/external 중간 substrate | `SRC-STC-0028` |
| 2025 | DANN SSRN | synthetic sleep과 zero forgetting 주장 | abstract-only, non-peer-reviewed low evidence | `SRC-STC-0039` |

### Sleep-Time Compute 논문의 정확한 novelty

기존 test-time compute는 한 query가 도착한 뒤 더 많이 reasoning한다. Sleep-Time Compute는 query가 없을 때 future query distribution을 예상해 compute를 선투자하고, 여러 later query에 amortize한다.

```text
test-time scaling:       query q → compute C_q → answer
sleep-time scaling:  history H → background C_s → reusable state Z → q_1, …, q_R
```

따라서 핵심 변수는 sleep FLOPs 하나가 아니라 **future-query predictability**, **reuse count R**, **state validity horizon**, **write/validation cost**, **base-version churn**이다.

## 6. 2026 H1: parametric sleep, learned memory updater, capacity-aware alternatives

| 날짜 | 사건 | 무엇이 새로웠나 | 경계 | 근거 |
|---|---|---|---|---|
| Jan 26 | Memory-Statistics Tradeoff, ICLR 2026 | structural regularization memory complexity와 statistical excess risk의 upper/lower bound | two linear regression tasks | `SRC-STC-0050` |
| Jan 26 | MaRS, ICLR 2026 | statistical slot expansion + dual-stage contrastive/distillation | frozen LPM, controlled benchmarks | `SRC-STC-0049` |
| Feb 18/26 | Meta PAHF | explicit per-user memory를 live dual feedback으로 update | online external memory, strict sleep 아님 | `SRC-STC-0044`, `0057` |
| Feb | Memory Caching | recurrent/neural memory state 재사용과 serving 문제를 전면화 | cached state correctness와 reuse boundary | `SRC-STC-0022` |
| Mar | LoRA as Knowledge Memory | LoRA delta의 knowledge-memory behavior 분석 | adapter capacity/interference 경계 | `SRC-STC-0040` |
| Mar 19/24 | Meta Hyperagents | task/meta improvement procedure 자체를 editable program으로 진화 | program mutation, not per-user weight sleep | `SRC-STC-0045`, `0058` |
| Apr 22 | SCM | sleep-consolidated memory와 algorithmic forgetting 제안 | single-author research preview, 작은 demonstration | `SRC-STC-0047` |
| Apr 21 | Google ReasoningBank release | 실패와 성공을 strategy memory로 distill, MaTTS와 결합 | append simplification; advanced consolidation future work | `SRC-STC-0054` |
| Apr 30 | MemoPilot, ICML 2026 | frozen player를 돕는 memory updater를 multi-turn GRPO로 학습 | game domains; main model fixed | `SRC-STC-0051` |
| May 6 | Memini | external graph edge에 fast/slow coupled dynamics | external memory를 learning substrate로 해석 | `SRC-STC-0046` |
| May 19 | Microsoft STATE-Bench | memory가 enterprise task 성공·reliability·cost·UX를 개선하는지 측정 | mechanism-neutral; deployment comparator 필요 | `SRC-STC-0056`, `0067` |

## 7. 2026 H2: product STC와 강한 반증이 동시에 등장

| 날짜 | 사건 | 직접 관찰 | 결론에 미친 영향 | 근거 |
|---|---|---|---|---|
| Jun 4 | OpenAI Dreaming V3 | many conversations를 background에서 합성; multi-year freshness/scalability; broad rollout | **external-memory STC가 production scaling axis로 존재함**을 E4로 확인 | `SRC-STC-0052` |
| Jun | Language Models Need Sleep | Knowledge Seeding + RL imitation, Dreaming RL curriculum | parametric STC가 단순 fine-tune가 아니라 data/training program이 됨 | `SRC-STC-0021` |
| Jul 2 | Episodic-to-Semantic Consolidation | identity drift를 억제하는 consolidation proposal | external/hybrid semantic promotion frontier | `SRC-STC-0038` |
| Jul 9 | Proactive Memory Agent | SFT/GRPO로 memory policy 학습 | frozen main model + learned external memory updater 강화 | `SRC-STC-0048` |
| Jul 10–14 | Continual Facts in Weights v2 | broad study data가 composition을 개선하지만 sequential writes에서 old fact reachability는 붕괴 | per-fact parametric memory의 default choice를 약화 | `SRC-STC-0036` |
| Jul | Rate-Distortion Memory Compaction | 여러 memory medium을 distortion-budget 관점으로 통합 | memory capacity를 raw item count가 아닌 task utility loss로 framing | `SRC-STC-0035` |
| Jul | MemDefrag / NSTM | latent memory compaction과 online space-time state | wake fast-state/compaction frontier 확장 | `SRC-STC-0037`, `0023` |
| Aug 5 snapshot | Mem0 Dream docs | scheduled Synthesis, provenance links, 20-memory threshold, weekly/daily cadence, no foreground latency | OpenAI와 별개인 두 번째 E4 external sleep lifecycle | `SRC-STC-0063` |
| Aug 5 snapshot | Mem0 Memory Decay | bounded access-history ranking, no destructive delete | capacity를 physical delete 없이 active-set ranking으로 다루는 product pattern | `SRC-STC-0062` |

## 8. 세 학습 phase framing의 정제

사용자의 initial framing은 다음과 같이 유지하되, `wake`와 `sleep`을 clock time이 아니라 dependency/SLA로 정의해야 한다.

| Phase | Trigger | Critical path | Typical state | Training operator | Serving implication |
|---|---|---|---|---|---|
| pre-/post-training | deployment 이전 release gate | 없음 | global base weights, tokenizer, reward model | pretraining, SFT, preference/RL, distillation | versioned global model |
| wake test-time inference/learning | current request/trajectory | TTFT·ITL·task completion SLA | KV, recurrent state, fast weights, online memory | forward, retrieval, TTT, limited online update | low-latency compute + reusable state cache |
| sleep-time learning/consolidation | accumulated evidence, cadence, idle/resource window | foreground와 분리 | summary, graph, adapter, expert, snapshot | replay, distillation, SFT/LoRA, RL memory policy, compaction | background scheduler + validation + atomic publish |
| long-term memory plane | 모든 phase | lookup/update SLA | event log, text/vector/graph, adapter catalog, lineage | indexing, decay, promotion, delete/rollback | system-of-record + tiered storage |

`NREM`과 `DREAM`은 유용한 design vocabulary일 수 있으나 biological equivalence claim이 아니다.

- NREM-like: replay, dedup, compression, stability-oriented consolidation
- REM/dream-like: counterfactual/synthetic variation, curriculum generation, exploration
- homeostasis-like: decay, pruning, renormalization, capacity rebalancing

## 9. 2026-08-05의 최종 위치

### 이미 현실인 것

- background memory synthesis가 production product에 존재한다.
- external memory update policy가 hand prompt에서 learned/RL policy로 이동한다.
- provenance, supersede, merge, decay, snapshot, merge-on-success가 주요 primitive로 수렴한다.

### 아직 연구인 것

- per-user model-weight sleep training의 fleet-scale deployment
- repeated wake/sleep cycle의 stable scaling law
- exact deletion/unlearning과 cheap rollback
- state migration과 wake-SLA까지 포함한 end-to-end economics

### 현재 가장 강한 방향

```text
immutable episode log
    → external consolidation (deployed now)
    → retrieval/routing evaluation
    → high-reuse, stable, consented knowledge only
    → reversible parametric promotion (research frontier)
```

이 chronology는 “STC가 갑자기 2025년에 발명됐다”는 서사를 부정한다. 문제와 operator는 수십 년간 존재했으며, 2025–2026의 변화는 이를 **deployment afterlife의 별도 compute budget과 lifecycle product**로 명시하고 실제 제품에 넣기 시작했다는 데 있다.
