# Approach Taxonomy: Sleep-Time Compute and Its Strongest Alternatives

- 기준일: 2026-08-05
- 분류 원칙: 논문·제품 이름보다 memory medium, update timing, trainable state, data, capacity behavior로 분류한다.
- strongest-result 열은 deep research에서 fixed source와 locator로 다시 동결할 seed 판정이다.

## 1. Family-level comparison

| Family | Memory medium | Update timing | Trainable / mutable state | Required data | Capacity behavior | Main cost | Governance | Strongest known result and boundary |
|---|---|---|---|---|---|---|---|---|
| Long context + full attention | live token sequence, KV cache | request prefill/decode | activation/KV; weight fixed | current prompt/documents | context length와 KV bytes에 직접 비례; old session은 기본 소실 | quadratic attention compute 또는 linear KV read; token price | prompt provenance 가능, derived state delete는 비교적 단순 | exact in-context evidence access가 강함; persistent learning이나 context comprehension을 보장하지 않음 |
| Recurrent/SSM/neural memory | fixed-size recurrent or fast-weight state | token/chunk마다 foreground | recurrent state, learned write rule | current stream | fixed state에서 compression/interference; reset boundary 중요 | state update FLOPs와 read/write bandwidth | latent source lineage·selective delete가 어려움 | Titans/MIRAS/ATLAS/HOPE가 long stream 성능을 확장; deployed cross-session sleep은 아님 |
| External text/event memory | database/object/file/log | wake append, async merge/summary | records, indexes, summaries | exact events and metadata | physical storage는 scale-out; retrieval/context budget이 기능적 한계 | storage, indexing, retrieval tokens | provenance·delete·version에 가장 유리 | canonical system of record로 강함; reasoning composition과 prompt pollution이 약점 |
| Vector retrieval / RAG | embedding + ANN index + source text | ingest 또는 async rebuild | embeddings, index, retriever | chunks, labels, queries | append는 쉬우나 redundancy·stale embedding·top-k bottleneck | embedding/index build, ANN query, retrieved tokens | source delete 가능하지만 derived cache/index fan-out 필요 | factual freshness와 exact source grounding에서 강함; retrieval miss와 multi-hop 조합이 제한 |
| Temporal/knowledge graph | node/edge, validity intervals, community summary | ingest-adjacent 및 periodic compaction | graph, entity resolution, summaries | episodes, entity/relation extraction | graph/summary가 단조 증가; merge·invalidation·rate budget 필요 | extraction LLM calls, graph traversal, rebuild | bitemporal provenance에 유리; shared summary delete가 어려움 | Graphiti/Zep류가 temporal contradiction과 cross-episode synthesis 제공; weight learning은 아님 |
| Deterministic external consolidation | summary, deduped records, compaction tree | scheduled/background | external records only | accumulated memory | explicit byte/token cap과 eviction 가능; semantic distortion 위험 | summarization, rewrite amplification | audit·rollback 가능하면 강함 | database compaction·agent reflection과 직접 연결; learned lifetime optimum 근거는 제한 |
| Test-time training / fast weights | activation, norm stats, adapter, fast weights | query/session foreground | selected parameters or state | current unlabeled/labeled stream | session scope면 bounded; persistent merge 시 interference | backward/optimizer가 P99와 batching에 영향 | tenant isolation, poison, rollback 필요 | distribution adaptation에 유효한 family; deferred cross-episode consolidation과 목적이 다름 |
| Replay-based continual learning | shared weight + replay buffer/generator | task boundary, periodic, sometimes offline | full weight/adapter/generator | old raw samples, coresets, synthetic replay | buffer/generator pressure로 이동; teacher corruption 가능 | training FLOPs, buffer, generation | sample lineage와 consent 필요 | joint-like retention에 가장 직접적인 baseline; endless capacity와 deletion은 해결하지 않음 |
| Regularization / gradient constraints | shared weight + importance/gradient state | task transition or every update | shared parameters, Fisher/importance/subspace | new data, importance or exemplars | protected directions 증가로 plasticity 감소 가능 | extra state, constrained optimizer | item-level provenance 없음 | EWC/GEM/MAS가 forgetting을 완화; lifetime capacity law나 background service는 아님 |
| Isolation / modular expansion | adapter, expert, task module, router | task/user arrival; optional sleep merge | new modules, router | task data and routing labels | free slot·router·residency가 포화; module 수 단조 증가 | parameter/storage, routing, loading | module-level rollback은 쉽고 item delete는 어려움 | interference를 줄이는 강한 대안; sharing과 global composition 비용이 증가 |
| Full fine-tuning / continued pretraining | dense shared weights | periodic batch or global refresh | all/shared weights, optimizer | curated corpus, replay mix | fixed parameters에서 interference·plasticity; checkpoint 수 증가 | high FLOPs, optimizer/checkpoint traffic | candidate/canary/rollback 가능, item provenance 약함 | common-domain refresh에는 경제적; user-local 잦은 update에는 비쌈 |
| PEFT / LoRA / adapter | low-rank or bottleneck delta | wake, task boundary, or sleep | adapter and optional merged weight | SFT/distillation/replay data | rank/module count와 router가 포화; merge는 interference를 다시 도입 | lower train FLOPs, adapter residency | version/rollback이 dense FT보다 쉬움 | knowledge/skill cache 후보; exact long-lived facts와 cumulative merge는 불안정 |
| Sparse memory/expert update | sparse table/expert slots | selected tokens/tasks; periodic | activated rows/experts | selected memory training data | slot occupancy·collision·router error | random access, sparse optimizer, residency | slot lineage가 설계에 따라 가능 | Meta memory layers/SMF가 capacity와 update locality를 분리; explicit sleep lifecycle은 미확인 |
| Model editing | localized weight changes | event-triggered foreground/batch | selected weights or editor state | edit tuple and paraphrase/context | edit accumulation이 locality·generalization을 훼손 | edit compute, validation matrix | edit log/rollback 필수; physical unlearning 미보장 | 소수 fact 수정에 강함; many sequential edits의 interference가 핵심 한계 |
| Knowledge unlearning | weights and derived memories | deletion request 후 batch | weights/adapters/index/checkpoints | forget/retain sets | repeated deletion이 utility·plasticity를 소모 | retraining/edit/evaluation | 법적·audit 중심; full fan-out proof 필요 | 별도 research family; STC가 기억을 늘릴수록 필수 counterpart가 됨 |
| Latent document compilation | cartridge/prompt embedding/adapter | ingest 후 offline | latent code or adapter | document and generated queries | document별 artifact 증가; compatibility/version pressure | compile FLOPs, artifact load | source linkage와 recompile/delete 필요 | 반복 query amortization 가능; one-shot 문서에는 비용 회수 실패 |
| Explicit sleep consolidation | external, latent, parametric 또는 혼합 | async snapshot/idle window | consolidator와 destination state | multi-episode trace, replay/dream/feedback | admission·promotion·eviction 없으면 모든 tier가 포화 | sleep compute, validation, publish, duplicated state | strongest requirement: lineage, isolation, atomic publish, rollback | Letta Sleep-time Compute와 일부 agent systems가 lifecycle을 구현; broad superiority 증거는 초기 |
| Periodic global refresh | global model and indexes | release cadence | dense/sparse global state | aggregated curated data | model version과 corpus가 scale; personalization은 희석 | large batch training, fleet rollout | mature release/canary/rollback | cross-user common knowledge에 강함; freshness와 private local memory에 약함 |
| Hybrid promotion lifecycle | canonical external + selective latent/parametric cache | wake capture, sleep promotion/eviction | router, external store, compiled state | exact source + reuse/value signals + validation set | tier별 cap을 독립 관리; duplication·consistency pressure | routing, compilation, multi-tier coherence | source-of-truth와 rollback을 유지할 수 있음 | 본 연구의 leading synthesis hypothesis; 아직 matched lifetime benchmark가 없음 |

## 2. Operator taxonomy

| Operator | Input | Output | Typical objective | Reversible? | Capacity effect | Required comparator |
|---|---|---|---|---|---|---|
| retain | event/source | same exact record | no loss | yes | monotonic growth | expiry/eviction |
| deduplicate | near-identical records | canonical + references | equivalence precision | usually | bytes decrease | exact retention |
| merge | multiple records | unified record | conflict resolution | with lineage | bytes decrease, coupling increases | no-merge retrieval |
| summarize | text/events | shorter semantic record | task/query utility | partly | token reduction, detail loss | extractive + raw fallback |
| graphize | episodes | entities/relations/time | relation fidelity | with source edges | metadata grows | vector/text retrieval |
| abstract | examples | rule/schema/skill | downstream generalization | difficult | compression, error amplification | exemplar replay |
| replay | old data/state | training batches | retention | source data yes | shifts pressure to buffer | no-replay, joint training |
| dream/augment | seeds/model | synthetic samples | robustness/coverage | generated data yes | generator/corruption pressure | matched extra real/synthetic data |
| distill | teacher + data | student/delta | KL/task loss | checkpoint-level | parametric compression | SFT/replay at same compute |
| fine-tune | curated data | weight/adapter delta | token/task/RL loss | checkpoint-level | interference or module growth | external memory, model edit |
| edit | fact/correction | localized delta | efficacy/locality/generalization | edit-log dependent | sequential edit pressure | retrieval + prompt correction |
| grow | task/value signal | slot/expert/adapter | reduce interference | module-level | physical growth | fixed-cap shared model |
| prune/evict | value/age/interference | freed state | utility under budget | only with archive | capacity restored, forgetting | decay/merge |
| verify | candidate + eval suite | promotion decision | risk-adjusted utility | yes | validation-state growth | unverified publish |
| publish/rollback | verified version | serving state | consistency/SLA | yes if versioned | duplicate residency | in-place mutation |

## 3. Memory-promotion state machine

```text
raw event
  ├─ reject/quarantine: unsafe, low-confidence, consent absent
  └─ canonical external record
       ├─ retain exact: rare/high-value/legal source
       ├─ merge/summary/graph: repeated semantic cluster
       ├─ latent compile: repeated document/query family
       ├─ adapter/expert promotion: high reuse + validated transfer
       └─ expire/evict: low value, stale, redundant

every derived state keeps source IDs, version, policy, evaluation and rollback edge
```

Promotion is not one-way. staleness, contradiction, delete request, model upgrade, measured interference can demote or rebuild derived state.

## 4. Crossover variables

| Variable | External memory favored when | Parametric/latent promotion favored when |
|---|---|---|
| query reuse `R_q` | low, unpredictable | high, stable family |
| exactness | exact quotation/source required | robust behavior/skill matters more than verbatim source |
| freshness | rapid correction and temporal validity | knowledge changes slowly |
| deletion | item-level delete SLA strict | canonical copy and retrain/rollback path exist |
| composition | retrieved context reliably supports reasoning | prompt/context pollution dominates and compiled skill transfers |
| latency | retrieval/token overhead acceptable | artifact is resident and saves repeated token/FLOPs |
| privacy/locality | trusted local store available | on-device/private adapter stays isolated |
| interference | updates frequent/noisy | sparse validated high-value promotions |
| model churn | base model changes often | adapter/cartridge compatibility stable |
| governance | auditability is primary | lineage-preserving compiler and rollback exist |

## 5. Taxonomy gap test

새 연구가 위 family에 들어가지 않으면 다음 질문으로 unknown을 만든다.

1. 실제로 새로운 memory medium인가, 기존 medium의 다른 이름인가?
2. update cadence가 다른가, 같은 operator를 background에서 실행했을 뿐인가?
3. capacity pressure를 제거했는가, 다른 state로 이동했는가?
4. retrieval/training/validation/publish 중 어느 비용을 숨겼는가?
5. source lineage, delete, rollback이 누락된 prototype인가?
6. future query reuse를 이용해 amortization하는가, 단순 extra compute인가?
