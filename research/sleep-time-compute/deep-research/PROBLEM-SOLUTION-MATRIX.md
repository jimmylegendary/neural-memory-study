# Problem–Solution Matrix: STC가 실제로 무엇을 이기려는가

- source freeze: 2026-08-05
- comparison rule: 각 문제에서 가장 약한 baseline이 아니라 **현재 가장 강한 non-STC default**와 비교한다.
- unit of judgment: quality, lifecycle cost, latency, capacity, governance를 분리하며 하나의 magic score로 평균하지 않는다.

## 1. Executive matrix

| Problem | Failure without adaptation | Strongest current default | Candidate STC contribution | Comparative verdict | Evidence anchor |
|---|---|---|---|---|---|
| static deployment | model knowledge/skill freezes while world and user drift | external retrieval + periodic global refresh | accumulated local evidence를 off-path에서 summary/strategy/adapter로 변환 | external STC는 already useful; broad per-user parametric STC는 conditional | `SRC-STC-0014`, `0021`, `0052`, `0063` |
| personalization | every session starts from zero or stale profile | editable per-user external memory | recurring preference·procedure를 background synthesis | personal facts는 external winner; stable behavior style는 reversible adapter 후보 | `SRC-STC-0044`, `0052`, `0057`, `0063` |
| agent experience | successful/failed trajectories are discarded | structured experience memory + reflection | many trajectories를 strategy로 distill하고 memory writer를 RL로 학습 | external/learned memory update가 strongest; main-weight sleep 불필요한 경우가 많음 | `SRC-STC-0043`, `0051`, `0054` |
| knowledge freshness | old facts remain parametric; retrieval index also stales | time-aware RAG/temporal graph + global refresh | background contradiction resolution, supersede, re-index | facts는 external temporal system-of-record가 우세 | `SRC-STC-0052`, `0063`, `0065`, `0066` |
| continual learning | new updates interfere with old behavior | replay + regularization + modular expansion | wake data를 replay/distill하고 low-rank/module에 publish | skill/domain adaptation에는 promising, open-ended factual writes에는 weak evidence | `SRC-STC-0002`–`0008`, `0036`, `0049`, `0050` |
| bounded capacity | memory/adapter/archive grows monotonically | admission, expiry, compaction, tiering, retrieval budget | idle time에 merge, distill, prune, promote/demote | necessary maintenance phase지만 fixed state cannot preserve an open entropy stream forever | `SRC-STC-0035`, `0050`, `0062`, `0063` |
| latency isolation | online training/retrieval harms TTFT/ITL | asynchronous queue + precomputed state/cache | expensive transform를 wake critical path 밖으로 이동 | STC의 가장 강한 system advantage; physical resource isolation은 별도 검증 필요 | `SRC-STC-0014`, `0052`, `0059`, `0063` |
| deletion/rollback | derived state survives source removal or bad update | provenance-bearing external log + versioned snapshots | candidate state를 isolated build 후 atomic publish | external state is default; parametric STC는 exact rollback/unlearning 전까지 제한 | `SRC-STC-0053`, `0060`, `0063` |
| long-context reasoning | context window can hold tokens but model cannot select/combine them reliably | retrieval, context compression, recurrent/neural memory, test-time reasoning | repeated corpus/trajectory를 concise reusable structure로 compile | repeated-query domain이면 useful; one-off query면 long context/RAG가 cheaper | `SRC-STC-0014`, `0022`, `0023`, `0035`, `0043` |
| on-device adaptation | cloud round trip, privacy, intermittent connectivity | local retrieval/profile + PEFT/federated update | idle/charging window에 device-local consolidation | high expected value but public STC device evidence is sparse; thermal/endurance gate 필수 | C12 measurement gap; `SATURATION.json` |

## 2. Decision rubric

각 row는 다음 질문을 순서대로 통과해야 한다.

1. **Is there a durable problem?** 단일 query를 더 오래 reasoning하면 사라지는 문제인가, 아니면 세션 간 state가 필요하나?
2. **Can retrieval solve it?** exact/provenance-sensitive 정보라면 external retrieval이 먼저다.
3. **Is there reuse?** compilation cost를 여러 later use가 상환하는가?
4. **Is the information stable?** volatility가 높으면 weight/adapter는 빨리 stale해진다.
5. **Can failure be reversed?** delete, correction, rollback blast radius가 acceptable한가?
6. **Can the queue remain stable?** admitted work가 sleep service보다 빠르게 쌓이지 않는가?
7. **Does quality improve at matched total lifecycle cost?** training FLOPs뿐 아니라 data prep, validation, bytes, replicas, recovery를 포함하는가?

## 3. Detailed problem analyses

### P1. Static deployment

**Problem.** pre-/post-training이 끝난 뒤 model은 고정되지만, enterprise policy·tool behavior·user environment는 계속 변한다.

**Non-STC alternatives.** 최신 문서는 RAG로 주입하고, globally important knowledge는 periodic refresh에서 다시 학습한다. 이는 source-of-truth와 model release를 분리해 운영하기 쉽다.

**STC hypothesis.** local history에서 recurring query/strategy distribution을 추정하고, idle time에 future-query index, summary, adapter를 만든다. Sleep-Time Compute는 query predictability와 repeated reuse가 있을 때 later inference compute를 amortize한다 (`SRC-STC-0014`).

**Verdict.** external background synthesis는 실사용 evidence가 있다 (`SRC-STC-0052`, `0063`). parametric variant는 global refresh보다 freshness가 빠르고 per-tenant specialization이 가능하지만, base-version churn과 deletion이 큰 domain에서는 default가 아니다.

**Required experiment.** same timestamped evidence와 same total compute에서 RAG, nightly summary, nightly LoRA, weekly global refresh를 비교하고 query reuse와 correction rate를 sweep한다.

### P2. Personalization

**Problem.** 개인 preference는 explicit/implicit, stable/transient, sensitive/non-sensitive가 섞여 있다. 하나의 profile summary에 모두 합치면 contradiction과 privacy risk가 생긴다.

**Strongest alternative.** per-user external records with provenance, temporal validity, consent, latest-only views. Meta PAHF는 explicit memory와 dual feedback으로 preference drift에 대응한다 (`SRC-STC-0044`).

**STC value.** OpenAI Dreaming과 Mem0 Dream은 many interactions에서 pattern을 background 합성한다. 이 기능은 실제 product evidence를 갖는다.

**Boundary.** “vegetarian”, current location, medical/legal constraint는 external record로 유지한다. repeated response style이나 tool procedure는 adapter promotion 후보가 될 수 있으나, user delete와 model upgrade를 함께 처리할 수 있어야 한다.

**Falsifier.** external profile + retrieval가 same quality와 lower cost/safer deletion을 제공하면 personal parametric STC는 불필요하다.

### P3. Agent experience

**Problem.** long-running agent가 매 task에서 같은 mistake를 반복하고, expensive exploration trajectory를 버린다.

**Strongest alternative.** structured external strategy memory. ReasoningBank는 success와 failure를 distill해 later tasks에 사용하고, MemoPilot은 memory updater를 multi-turn GRPO로 downstream reward에 맞춘다 (`SRC-STC-0043`, `0051`).

**STC value.** trajectory compression, counterfactual generation, skill validation을 background에서 실행하면 current task latency와 분리할 수 있다.

**Verdict.** 이 문제에서는 “model weights must sleep”보다 “memory writer and consolidation policy must learn”이 더 강한 현재 방향이다. frozen main model + learned external memory policy가 rollback과 model portability에서 유리하다.

### P4. Knowledge freshness

**Problem.** fact의 truth value와 relevance가 시간에 따라 바뀐다. parametric recall은 timestamp/provenance를 숨기기 쉽다.

**Strongest alternative.** external temporal graph, latest-only view, source-backed retrieval. Graphiti/Zep와 Mem0 Supersede는 old/new fact 관계를 명시한다.

**STC value.** background에서 contradiction을 찾고, derived summary를 재작성하고, stale index를 rebuild할 수 있다.

**Verdict.** STC는 **maintenance timing**으로 유용하지만 destination은 external이 우세하다. volatile fact를 weights에 재학습하는 것은 freshness 문제를 deletion/rollback 문제로 바꿀 수 있다.

### P5. Continual learning

**Problem.** sequential update는 interference, stability–plasticity tradeoff, replay growth를 일으킨다.

**Strongest alternatives.** replay, EWC-like regularization, gradient constraints, modular adapters/slots, periodic joint retraining.

**STC value.** acquisition과 consolidation을 분리하고, fresh wake model과 protected replay teacher를 함께 사용하며, low-rank/module에 bounded write를 수행한다.

**Negative evidence.** sequential factual weights는 study data를 사용해도 behavioral reachability가 무너질 수 있다 (`SRC-STC-0036`). Memory-Statistics Tradeoff는 stability state도 memory cost가 있음을 보인다 (`SRC-STC-0050`).

**Verdict.** stable skill/procedure, domain-specific behavior, repeated task family에는 promising. exact changing facts와 legal delete 대상에는 non-default.

### P6. Bounded capacity

**Problem.** naive append는 text/vector/graph, adapter catalog, optimizer/recovery state 모두를 증가시킨다.

**Strongest alternatives.** TTL/expiry, dedup, coreset, rate-distortion compaction, tiering, module expansion with routing, periodic full rebuild.

**STC value.** expensive dedup/merge/summarize/replay/defrag를 off-path에서 수행한다. Mem0 Dream은 additive synthesis와 non-destructive supersede/merge를 제공하고, Decay는 active retrieval ranking을 조절한다.

**Fundamental boundary.** fixed finite state는 positive-entropy novel stream을 constant distortion으로 영원히 보존할 수 없다. 시스템은 expand, forget/raise distortion, restrict admission/horizon, or move across a declared boundary 중 하나를 선택해야 한다.

**Verdict.** sleep maintenance is necessary for sustainable systems, but it cannot repeal capacity. “bounded hot set”과 “bounded lifetime storage”를 구분한다.

### P7. Latency isolation

**Problem.** online update가 TTFT/ITL, GPU residency, request batching을 방해한다.

**Strongest alternative.** no training on critical path; precompute retrieval index/cache; schedule background work with priority/preemption.

**STC value.** transform를 post-response 또는 scheduled batch로 옮겨 reuse할 state를 만든다. Letta, OpenAI, Mem0가 logical separation evidence를 제공한다.

**Hidden cost.** wake와 sleep이 같은 HBM/network/storage를 공유하면 logical background가 P99 interference를 일으킬 수 있다. `queue stable`과 `foreground safe`는 동일하지 않다.

**Verdict.** STC를 독립 축으로 부를 가장 강한 이유 중 하나. 단, separate cluster가 항상 답은 아니며 regional/shard-local hybrid와 state affinity가 data movement를 줄일 수 있다.

### P8. Deletion/rollback

**Problem.** 하나의 source episode가 summary, edge, embedding, cache, adapter, optimizer checkpoint로 파생된다. source만 지우면 derived state가 남을 수 있다.

**Strongest alternative.** immutable event lineage + tombstone/deny + asynchronous physical reclamation + versioned snapshot.

**STC value.** candidate state를 isolated snapshot에서 만들고 validation 후 atomic publish하며, 실패 시 merge하지 않는다. Letta Code reflection은 이 pattern을 구현한다 (`SRC-STC-0060`).

**Verdict.** external memory는 delete/rollback의 control plane이 될 수 있다. parametric state는 source-of-truth가 아니라 rebuildable cache로 취급해야 한다.

### P9. Long-context reasoning

**Problem.** full attention window가 커져도 relevance selection, distractor robustness, multi-session persistence는 자동 해결되지 않는다. 긴 context를 매 query 재읽는 비용도 증가한다.

**Strongest alternatives.** long context + FlashAttention, RAG, recurrent/SSM/neural memory, context compression, test-time compute.

**STC value.** repeated corpus를 reusable index/summary/strategy/adapter로 compile하면 same source의 repeated prefill과 reasoning을 줄인다.

**Break-even.** one-off document라면 compilation cost를 회수하지 못한다. future query distribution이 예측 가능하고 reuse가 충분할 때만 Sleep-Time Compute가 우세하다.

**Verdict.** long context를 대체하는 단일 architecture가 아니라, repeated workloads에서 long context를 **build path**로 쓰고 compact state를 **serve path**로 쓰는 complement다.

### P10. On-device adaptation

**Problem.** 개인 data를 cloud로 보내기 어렵고, network가 불안정하며, local sensors/robot behavior가 drift한다.

**Strongest alternatives.** device-local retrieval/profile, tiny adapters, federated learning, server-trained personalization, local TTT.

**STC value.** charging/idle/thermal headroom window에 local evidence를 consolidate하고, wake time에는 small resident state만 사용한다.

**Missing evidence.** C12 search는 dedicated STC device traces를 찾지 못했다. NAND write amplification, NPU training support, battery energy, thermal recovery, secure erase, cross-device sync가 측정되지 않았다.

**Verdict.** high expected value / low evidence maturity. memory-device company가 controlled trace와 reference design을 선점할 수 있는 영역이다.

## 4. Problem-to-medium routing rules

| If the memory is… | Prefer | Avoid by default |
|---|---|---|
| exact, volatile, attributable | external event/text/graph | opaque weight write |
| repeated procedural skill | reversible adapter/expert after validation | retrieval of full trajectory every time |
| one-off large document | RAG/long context | expensive compilation before reuse exists |
| stable corpus with many predictable queries | latent index/adapter compilation | re-prefill full corpus per query |
| sensitive personal data | encrypted external state with deletion lineage | shared global weights |
| rare safety-critical exception | protected canonical record + forced retrieval | pure recency decay/summary |
| high-volume duplicate observations | merge/dedup + representative archive | append-only hot index |
| uncertain model-generated insight | quarantined candidate with provenance | immediate durable publish |

## 5. What must be measured together

```text
quality:       exact, paraphrase, composition, procedure, calibration, rare tail
learning:      new acquisition, old retention, prompted recoverability, plasticity
service:       TTFT, ITL, task latency, sleep job age, queue backlog
compute:       wake FLOPs, sleep FLOPs, validation FLOPs, retries
state:         raw, external, latent, parametric, auxiliary, version/recovery bytes
movement:      HBM↔DRAM, host↔device, storage, network, replication bytes
governance:    poison success, stale rate, delete latency, rollback success
economics:     cost per useful later-wake reuse, compile break-even, base-version churn
```

## 6. Overall conclusion

STC가 가장 잘 푸는 단일 문제는 “모델이 잠을 자야 한다”가 아니라 다음 조합이다.

> 반복될 가능성이 높은 경험을 foreground latency 밖에서 더 작은 reusable state로 바꾸고, 다음 wake에서 그 state를 싸게 이용한다.

이 정의에서는 external synthesis가 이미 strongest deployed form이다. Parametric sleep은 skill/procedure compaction과 repeated-query domains에서 추가 upside가 있지만, fact freshness·capacity·deletion/rollback에서는 stronger default를 아직 이기지 못했다.

