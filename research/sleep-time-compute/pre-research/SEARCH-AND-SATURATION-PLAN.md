# Deep-Research Search and Saturation Plan

- search window: earliest relevant prior through 2026-08-05
- evidence freeze: 2026-08-05 23:59 Asia/Seoul
- source priority: proceedings/journal/fixed preprint → official code/data → official product docs/release/tag
- discovery-only: blogs, news, third-party summaries, search snippets

## 1. Search clusters

| Cluster | Positive discovery families | Disconfirming families | Mandatory output |
|---|---|---|---|
| C01 biological/computational sleep | replay, complementary learning systems, wake–sleep, dreaming, homeostasis | stage non-equivalence, sleep model ablation failure, capacity limit | stage/operator boundary table |
| C02 continual learning | replay, regularization, gradient constraints, modularity, expansion, pruning | plasticity loss, task-boundary dependence, replay corruption | strongest-baseline matrix |
| C03 LLM parametric learning | continued pretraining, SFT, LoRA, sparse FT, model editing | sequential fact interference, locality failure, unlearning cost | weight-as-cache verdict |
| C04 TTT/fast state | test-time training, fast weights, online adaptation, neural memory | latency, stability, state reuse/serving difficulty | wake-vs-sleep boundary |
| C05 external memory | RAG, vector, temporal graph, agent memory, reflection | retrieval miss, context pollution, compaction loss, stale index | external system-of-record analysis |
| C06 latent compilation | document-to-adapter, cartridge, prompt/latent memory | compile amortization failure, base-model incompatibility | query-reuse crossover |
| C07 product lifecycle | Letta/MemGPT, Mem0, Zep, LangMem, Memento, Google, Meta, Microsoft, OpenAI | marketing-only, ingest-time-only, manual-only, missing durable destination | C1/C2/C3 lifecycle audit |
| C08 capacity/forgetting | bounded synapse, information capacity, rate–distortion, compaction | semantic capacity non-identifiability, governance saturation | five-capacity model |
| C09 training/data | replay, distillation, augmentation, synthetic data, RL scheduler, meta-learning | model collapse, poison, reward hacking, leakage | operator/data atlas |
| C10 systems/infra | background training, snapshot, COW, tiering, CXL, RDMA, offload | data movement, P99 interference, model residency | reference architecture + roofline |
| C11 governance/security | provenance, deletion, unlearning, rollback, memory poisoning | derived-state delete failure, audit gaps | publish/delete protocol |
| C12 device/workload | storage-near compute, computational storage, on-device personalization, robotics | endurance, thermal, write amplification, low reuse | device opportunity matrix |

## 2. Query protocol

각 cluster는 네 번의 독립 pass를 갖는다.

1. **terminology pass**: canonical term과 cited seminal work를 찾는다.
2. **functional-equivalence pass**: sleep이라는 말을 쓰지 않는 동일 lifecycle을 찾는다.
3. **negative pass**: failure, limit, collapse, interference, leakage, deletion, retraction을 결합한다.
4. **latest/status pass**: 2025–2026 venue, version, official repository, product release를 재검증한다.

### 2.1 Core query templates

```text
("sleep-time compute" OR "offline consolidation") model memory
("post-response" OR asynchronous OR background) agent memory reflection
(continual OR lifelong) learning (replay OR consolidation) capacity limit
(LLM OR language model) sequential facts weights interference
(RAG OR external memory) versus (fine-tuning OR model editing)
(memory compaction OR summary) rare fact loss provenance deletion
(test-time training OR fast weights) serving latency state reuse
(adapter OR LoRA OR expert) accumulation routing capacity
(synthetic replay OR generated replay) collapse corruption anchor
(CXL OR computational storage OR near-data) embedding training compaction
```

### 2.2 Company/status query templates

```text
site:research.google OR site:deepmind.google memory continual learning sleep
site:ai.meta.com/research memory layer continual fine-tuning
site:microsoft.com/en-us/research long memory agent continual
site:openai.com research memory learning agents official
site:docs.letta.com sleeptime reflection memory official
site:docs.mem0.ai memory graph consolidation delete official
site:help.getzep.com OR site:github.com/getzep graphiti observation memory
```

Product claim은 page의 현재 문구만 기록하지 않는다. version/tag/release date, archived copy 또는 content hash를 함께 기록한다.

## 3. Citation-graph snowballing

각 seed source에 대해 다음 edge를 기록한다.

- backward: theory, operator, dataset, baseline을 제공한 cited source
- forward: Semantic Scholar/OpenAlex/Google Scholar에서 후속 use, replication, critique
- sibling: 동일 저자·codebase·dataset lineage의 later version
- adversarial: title/abstract가 반대 결론을 갖는 work
- implementation: official repository, tag, issue, release note
- status: OpenReview/proceedings/DOI venue identity

Veridraft deep survey가 citation API에 접근하지 못하면 DOI/arXiv/OpenReview와 manual ledger를 사용한다. citation count가 아니라 신규 claim cluster coverage로 saturation을 판정한다.

## 4. Inclusion criteria

### 4.1 Technical source

- original method, theory, dataset, benchmark 또는 negative result를 직접 제시한다.
- fixed identifier와 version/status를 확인할 수 있다.
- load-bearing claim에 exact locator를 만들 수 있는 full text가 있다.
- preprint면 peer-review status를 과장하지 않는다.

### 4.2 Product/deployment source

- official documentation, repository, tag, release note 또는 API schema다.
- accumulated wake input, off-path transform, durable later-wake destination을 각각 확인한다.
- managed proprietary behavior와 OSS implementation을 분리한다.

### 4.3 Figure reuse

- public availability가 아니라 redistribution license를 확인한다.
- 허가가 불명확하면 value/label/relationship만 독립 redraw한다.
- screenshot이 필요한 product UI는 official brand/license와 publication fair-use boundary를 별도 검토한다.

## 5. Exclusion and downgrade criteria

| Condition | Treatment |
|---|---|
| search snippet/secondary summary only | discovery record, claim support 불가 |
| abstract-only inaccessible manuscript | E0–E1, 수치 인용 금지 |
| withdrawn/rejected/unaccepted submission | status 명시, strongest evidence로 사용 금지 |
| vendor benchmark without independent control | author claim, external validation 필요 |
| same lineage reuses dataset/code | independence count 1로 묶음 |
| mutable product page without version | current-state claim만, historical chronology 금지 |
| figure license unclear | redraw-only |
| missing locator | public load-bearing claim hold |

## 6. Saturation rule

### 6.1 Claim saturation

cluster별 연속 두 pass에서 다음이 모두 성립하면 `claim_saturated`로 표시한다.

1. 신규 approach family가 없다.
2. 기존 verdict를 바꾸는 독립 positive/negative result가 없다.
3. strongest comparator가 바뀌지 않는다.
4. load-bearing claim의 source/locator gap이 증가하지 않는다.

### 6.2 Lineage saturation

company/product cluster는 다음을 모두 확인해야 한다.

- paper/preprint chronology
- official code or absence
- current official docs/release
- durable destination and phase classification
- capacity, delete, rollback boundary

### 6.3 Quantitative saturation

최소 다음 curve를 제공하는 source가 없으면 해당 law cluster는 saturated로 부르지 않고 `measurement_gap`으로 닫는다.

```text
utility vs cumulative wake evidence
utility vs sleep compute
retention/plasticity vs update count
latency/cost vs query reuse
utility vs memory budget
bytes moved vs compute placement
```

### 6.4 Stop decision

| state | 의미 | 후속 행동 |
|---|---|---|
| saturated | 두 independent pass에서 신규 load-bearing structure 없음 | registry freeze |
| bounded | key lineage는 닫혔으나 workload 범위 제한 | 제한 문구와 함께 사용 |
| measurement_gap | 논문 수보다 필요한 curve가 없음 | experiment hypothesis로 이동 |
| access_gap | full text/code/product version 접근 불가 | unsupported claim hold |
| active_frontier | freeze 직전 신규 결과로 citation graph가 계속 변함 | date-bounded snapshot과 후속 watchlist |

## 7. Search log schema

각 query execution은 다음 필드를 가진다.

```json
{
  "search_id": "STC-SRCH-0001",
  "cluster": "C03",
  "pass": "negative",
  "query": "language model sequential facts weights interference",
  "executed_at": "2026-08-05",
  "provider": "web/arXiv/OpenReview/official-site",
  "results_screened": 0,
  "included_source_ids": [],
  "excluded_with_reason": [],
  "new_claim_families": [],
  "saturation_effect": "unchanged"
}
```

`results_screened`는 실제 count만 기록하며 검색 API가 반환하지 않은 경우 추정하지 않는다.

## 8. Planned evidence quotas

quota는 quality를 대체하지 않지만 coverage 누락을 탐지한다.

| Evidence class | 최소 목표 |
|---|---:|
| peer-reviewed/final technical papers | 45 |
| fixed preprints with full text | 30 |
| negative/limit papers | 15 |
| official product/code/release artifacts | 25 |
| systems/device primary sources | 20 |
| independent benchmark/data sources | 12 |
| load-bearing claims with exact locators | 100% |
| direct-reuse figures with reviewed rights | 100% |

동일 source가 여러 class에 기여해도 source count를 중복 부풀리지 않는다.

## 9. Deep-research handoff

검색 종료 시 다음을 동결한다.

- `SOURCE-REGISTRY.json`: version, hash, status, rights
- `CITATION-POOL.json`: canonical citation metadata
- `SATURATION.json`: cluster state와 search pass
- `NEGATIVE-EVIDENCE.md`: thesis를 약화하는 결과
- `SOURCE-RIGHTS.json`: full-text/figure redistribution boundary
- `CHRONOLOGY.md`: earliest public date와 final venue를 분리한 chronology

이 파일들이 없으면 Study Paper prose를 작성하지 않는다.
