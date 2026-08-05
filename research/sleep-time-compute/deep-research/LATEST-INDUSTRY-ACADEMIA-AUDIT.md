# Sleep-Time Compute 최신 산업·학계 감사

- source freeze: 2026-08-05 23:59 Asia/Seoul
- last verified date: 2026-08-05
- evidence rule: 논문은 원문·proceedings·DOI, 제품은 official release/documentation/tagged official repository만 근거로 사용한다.
- companion registries: `SOURCE-REGISTRY.json`, `CITATION-POOL.json`, `SOURCE-RIGHTS.json`, `SATURATION.json`

## 1. 무엇을 STC로 셀 것인가

이 감사는 이름에 `sleep`, `dream`, `reflection`이 들어가는지만 보지 않는다. 다음 세 조건을 모두 만족해야 **strict lifecycle STC**로 센다.

1. **C1 — accumulated wake input:** 여러 wake interaction·document·trajectory가 누적된다.
2. **C2 — off-path transform:** foreground 응답의 critical path와 분리된 시간에 압축·합성·replay·distillation·training이 수행된다.
3. **C3 — durable later-wake destination:** 변환 결과가 다음 세션에서 재사용되는 durable state에 기록된다.

이 세 조건을 만족해도 destination에 따라 다시 나눈다.

| Destination | 무엇이 바뀌는가 | 대표 장점 | 대표 위험 |
|---|---|---|---|
| parametric | base weight, adapter, expert, fast-weight state | 별도 retrieval 없이 행동·skill에 녹일 수 있음 | interference, rollback/delete, optimizer state, tenant isolation |
| external symbolic | text, event, summary, rule, file | provenance·편집·삭제·감사가 쉬움 | retrieval miss, context pollution, summary loss |
| external vector/graph | embedding, temporal graph, edge state | 확장성·관계 추론·시간 변화 표현 | index drift, stale edge, delete propagation |
| hybrid promotion | external system-of-record에서 검증된 항목만 adapter/expert로 승격 | governance와 amortization을 함께 노림 | promotion 기준·양방향 consistency가 복잡 |

### 1.1 증거 등급

| 등급 | 의미 |
|---|---|
| E0 | 이름·주장만 있고 검증 가능한 artifact가 없음 |
| E1 | abstract 또는 research preview만 있음 |
| E2 | fixed paper와 실험은 있으나 실제 lifecycle deployment 증거가 없음 |
| E3 | public code 또는 공식 문서에서 lifecycle이 확인됨 |
| E4 | 실제 product rollout과 durable behavior가 공식적으로 확인됨 |
| E5 | workload trace, independent reproduction, failure/rollback/delete curve까지 공개됨 |

현재 E5에 도달한 공개 시스템은 없다. 따라서 “mainstream인가”라는 질문은 연구 아이디어의 수가 아니라 E3–E4 시스템의 destination과 운영 경계를 기준으로 판단해야 한다.

## 2. Required company/product lineage audit

아래 표에서 **public implementation**, **deployment evidence**, **missing evidence**를 분리했다. 연구 논문이 있다고 해서 제품 배포로, 제품 기능이 있다고 해서 weight update로 해석하지 않는다.

| Entity | Research mechanism | public implementation | Product feature | deployment evidence | Destination / C1·C2·C3 | missing evidence | Evidence |
|---|---|---|---|---|---|---|---|
| Google/DeepMind | `Language Models Need Sleep`은 Knowledge Seeding에서 generalized distillation과 RL imitation을, Dreaming에서 RL 기반 synthetic curriculum을 사용해 fixed/pretrained LM에 offline adaptation을 시도한다. `Nested Learning`/Hope는 optimizer·memory를 다중 update frequency로 해석한다. `ReasoningBank`는 성공·실패 trajectory에서 transferable reasoning strategy를 추출한다. | Titans/MIRAS/ATLAS/Nested 계보는 논문이 고정되어 있다. ReasoningBank는 official code link가 공개됐다. 다만 Google production service의 background per-user weight write 구현은 확인되지 않았다. | 공식 Google Research 자료는 ReasoningBank를 continuous test-time self-evolution 연구로 설명한다. production Gemini memory의 strict sleep lifecycle로 공개한 것은 아니다. | 논문 실험 E2. ReasoningBank official research release E3에 가까우나 product rollout 주장은 없음. | Parametric research: C1/C2/C3를 논문 실험에서 구성. ReasoningBank: external structured memory, 주로 task 후 update라 C2의 background isolation은 미확인. | multi-tenant weight isolation, background cluster, rollback/delete propagation, repeated-cycle survival, production workload trace | `SRC-STC-0020`, `0021`, `0054`, `0055`; [ReasoningBank official](https://www.research.google/blog/reasoningbank-enabling-agents-to-learn-from-experience/) |
| Meta | PAHF는 live interaction의 pre-action clarification과 post-action feedback을 explicit per-user memory에 반영한다. Hyperagents는 task agent와 self-editable meta-agent를 한 프로그램으로 만들고 archive·persistent memory·performance tracking을 개선 대상으로 포함한다. | Hyperagents official repository lineage와 Meta research pages가 있다. PAHF는 paper가 있다. | 공개 research artifact이며 consumer product sleep-memory rollout으로 제시되지 않았다. | E2–E3 research/code evidence. | PAHF: external memory, C1/C3 yes, C2 no(online loop). Hyperagents: program/archive mutation, 여러 run 누적은 있으나 biological sleep 또는 background serving lifecycle 아님. | production adoption, tenant isolation, deletion/rollback, resource trace, model-weight consolidation 여부 | `SRC-STC-0044`, `0045`, `0057`, `0058`; [PAHF official](https://ai.meta.com/research/publications/learning-personalized-agents-from-human-feedback/), [HyperAgents official](https://ai.meta.com/research/publications/hyperagents/) |
| Microsoft | LongMem은 frozen backbone과 external memory side network를 결합한다. GenerativeAdapter는 data-to-adapter compilation에 가까운 parametric bridge다. STATE-Bench는 memory가 realistic enterprise task에서 reliability·cost·UX를 실제로 개선하는지 평가한다. | STATE-Bench repository가 MIT license로 공개되어 있고 memory-agnostic interface를 제공한다. | benchmark release이며 Microsoft agent product의 특정 background consolidation 구현을 뜻하지 않는다. | E3 benchmark implementation. GPT-5.1 no-memory baseline을 제공하지만 특정 memory mechanism의 production win은 아직 open challenge다. | Benchmark 자체는 destination neutral. LongMem은 external, GenerativeAdapter는 parametric artifact. | Microsoft product의 strict C1/C2/C3, delete/rollback, background scheduling, hardware trace | `SRC-STC-0026`, `0027`, `0056`, `0067`; [STATE-Bench official](https://opensource.microsoft.com/blog/2026/05/19/introducing-state-bench-a-benchmark-for-ai-agent-memory/) |
| OpenAI | ChatGPT Dreaming은 여러 conversation을 background에서 참조해 memory state를 합성하고 freshness·continuity·relevance를 개선한다. 2024 saved memory → 2025 Dreaming V0 → 2026 Dreaming V3로 진화했다. | 내부 implementation은 비공개다. 공식 release와 user controls가 공개되어 있다. | Plus/Pro US에서 시작해 추가 국가와 Free/Go로 확대되는 ChatGPT memory feature. | E4. 공식 자료가 hundreds of millions of users와 multi-year horizon의 staleness/correctness/scalability 문제, background process, 약 5× serving-compute reduction을 명시한다. | external synthesized memory로 해석하는 것이 가장 안전하다. C1 yes, C2 yes, C3 yes. model-weight update라는 evidence는 없음. | 저장 형식, model/version별 state compatibility, end-to-end deletion of derived summaries, rollback semantics, false-memory rate, independent evaluation, HBM/storage trace | `SRC-STC-0052`, `0053`; [Dreaming official](https://openai.com/index/chatgpt-memory-dreaming/)—lines 39–53, 241–284 in frozen capture |
| Letta/MemGPT | MemGPT는 virtual-context paging처럼 core/archival memory를 관리한다. Letta sleeptime agent는 foreground response 후 별도 agent가 conversation을 검토하고 memory를 수정한다. Letta Code reflection은 isolated worktree/snapshot에서 reflection 결과를 만들고 성공할 때만 merge한다. | tagged Letta `0.16.8` sleeptime code와 Letta Code `v0.28.18` reflection code가 고정되어 있다. | Letta agent platform의 memory/reflection 기능. | E3 public implementation. | external text/file memory. Sleeptime code: C1 yes, C2 yes, C3 yes. Reflection launcher: COW/versioned commit pattern. | managed-service fleet scale, quality curve vs sleep budget, automatic delete propagation, cross-version compatibility, independent benchmark | `SRC-STC-0013`, `0059`, `0060`; [sleeptime code](https://github.com/letta-ai/letta/blob/0.16.8/letta/groups/sleeptime_multi_agent_v4.py)—class 805, post-response 988, background task 1056, prompt 1169; [reflection code](https://github.com/letta-ai/letta-code/blob/v0.28.18/src/cli/helpers/reflection-launcher.ts)—2207–2234, 2281–2285 |
| Mem0 | V3 add는 async additive pipeline이다. Memory Decay는 access history로 search score를 0.3×–1.5× 조절한다. Dream은 Synthesis·Supersede·Merge를 분리한다. Synthesis는 recurring patterns를 background schedule에서 새 memory로 기록하고 source link를 유지한다. | OSS repository와 tagged OpenClaw dream gate가 있다. Product Dream의 내부 synthesizer는 proprietary일 수 있으나 API/lifecycle 문서는 공개다. | Pro: user별 7일 cadence, Enterprise: daily/configurable; 최소 20 memories; scheduled run 후 약 24시간 내 pattern 반영. Supersede/Merge는 ingest-time, Synthesis만 scheduled background. | E4 product documentation. | external memory. Synthesis: C1 yes, C2 yes, C3 yes. additive and idempotent; source memories retained. Weight update 증거 없음. | independent quality/false-pattern results, actual compute cost, delete propagation from source to pattern, tenant-scale failure data, schedule starvation, proprietary operator detail | `SRC-STC-0061`–`0064`; [Dream official](https://docs.mem0.ai/platform/features/dream)—lines 81–145, 190–226; [Decay](https://docs.mem0.ai/platform/features/memory-decay)—81–120, 243–271 |
| Zep | Graphiti는 temporally changing facts, episode provenance, incremental graph updates, hybrid semantic/BM25/graph retrieval을 제공한다. Zep episode는 raw input을 verbatim 보존하고 derived entity·edge·summary와 연결한다. | Graphiti pinned repository가 Apache-2.0으로 공개되어 있다. | temporal knowledge graph based agent memory. | E3 code/docs. | external graph/event memory. C1 yes, C3 yes. update가 ingestion path에 결합되어 있어 별도 scheduled C2는 공개 evidence로 확인되지 않음. | background cadence, bounded-capacity policy, delete cascade, stale-edge SLA, independent long-run reliability, parametric promotion | `SRC-STC-0016`, `0065`, `0066`; [Graphiti pinned README](https://github.com/getzep/graphiti/blob/aab852df94413fd0d55cbea2b7886173020281d5/README.md), [Zep Episodes](https://help.getzep.com/episodes) |

## 3. Product-level lifecycle result

### 3.1 Strict lifecycle 판정

| System | C1 accumulated input | C2 off-path transform | C3 durable destination | Strict STC | Medium | Evidence tier |
|---|---:|---:|---:|---:|---|---:|
| OpenAI Dreaming V3 | ✓ | ✓ | ✓ | **yes** | external synthesized user memory | E4 |
| Mem0 Dream Synthesis | ✓ | ✓ | ✓ | **yes** | external pattern memory with provenance | E4 |
| Letta sleeptime agent | ✓ | ✓ | ✓ | **yes** | external text/block/file memory | E3 |
| Letta Code reflection | ✓ | ✓ | ✓ | functional equivalent | versioned files/worktree | E3 |
| Google Language Models Need Sleep | ✓ | ✓ in experiment | ✓ adapted model | research-only strict STC | parametric | E2 |
| Google ReasoningBank | ✓ | task-end update; background unspecified | ✓ | partial | external reasoning strategies | E2–E3 |
| Meta PAHF | ✓ | ✗ online feedback loop | ✓ | no | external per-user memory | E2 |
| Meta Hyperagents | ✓ | evaluation/evolution loop | ✓ archive/program | adjacent | program + archive | E2–E3 |
| Microsoft STATE-Bench | n/a | n/a | n/a | benchmark, not mechanism | neutral | E3 |
| Zep Graphiti | ✓ | ingest-time incremental | ✓ | no strict C2 evidence | temporal graph | E3 |
| Mem0 Supersede/Merge | ✓ | ✗ ingest-time | ✓ | no; wake-path maintenance | external records | E4 docs |
| Mem0 Memory Decay | ✓ retrieval history | ✗ search-time | ✓ metadata | no; wake-path ranking | external retrieval metadata | E4 docs |

### 3.2 가장 중요한 산업 신호

2026-08-05 기준 공개 evidence는 “sleep-time compute가 등장했는가?”에는 **yes**라고 답하게 한다. OpenAI와 Mem0는 background synthesis를 실제 product lifecycle로 명시했고 Letta는 public code로 구현했다. 그러나 “deployment 이후 사용자마다 foundation-model weight를 계속 학습하는가?”에는 **public evidence 없음**이 답이다.

따라서 현재 주류가 되어 가는 축은 다음과 같다.

```text
wake interaction
    → append immutable/provenance-bearing episodes
    → cheap ingest-time contradiction/dedup
    → scheduled background synthesis
    → active summary/pattern/strategy state
    → retrieval into later wake context
```

이는 STC의 부정이 아니라 **external-memory STC가 먼저 제품화됐다**는 뜻이다. Weight-space consolidation은 skill compaction이나 높은 reuse가 있는 domain adapter에 더 적합하지만, per-user facts의 source-of-truth로는 삭제·감사·rollback 문제 때문에 아직 열세다.

## 4. Latest academic frontier

| Work | Update target | Training/data method | What it resolves | What remains open | Status / evidence |
|---|---|---|---|---|---|
| Sleep-Time Compute (2025) | adapter/parametric state | anticipated query generation + offline fine-tuning; reuse-amortized compute | future-query predictability를 이용한 inference compute amortization | distribution shift, delete/rollback, repeated cycles, base-model churn | arXiv, `SRC-STC-0014` |
| Language Models Need Sleep (2026) | model parameters | generalized distillation + RL imitation; Dreaming RL synthetic curriculum | post-deployment knowledge/skill assimilation | recursive synthetic drift, production cost, user isolation, long-run capacity | fixed preprint, `SRC-STC-0021` |
| Can a Language Model Learn Facts Continually in Its Weights? (2026) | weights | continued fact writes with study-data variants | recitation-to-use gap를 크게 줄이는 data treatment | 20 writes 뒤에도 old fact reachability가 무너짐; context remains reliable | central negative, `SRC-STC-0036` |
| ReasoningBank (ICLR 2026) | external structured strategies | success/failure trajectory distillation; memory-aware TTS | repeated strategic error와 exploration waste | append-only simplification; advanced consolidation left future | Google official + paper, `SRC-STC-0043`, `0054` |
| MemoPilot / From Player to Master (ICML 2026) | trainable memory updater, frozen player | multi-turn GRPO over later task reward | memory writing을 hand prompt가 아니라 downstream utility에 맞춤 | two game domains, policy/model transfer and safety | peer-reviewed, `SRC-STC-0051` |
| Memini multi-timescale memory (2026) | external directed graph edges | fast/slow coupled state inspired by synaptic consolidation | episodic sensitivity, consolidation, decay를 한 dynamics로 묶음 | large-scale agent benchmark and operations | arXiv, `SRC-STC-0046` |
| MaRS (ICLR 2026) | expandable slots/router | statistical slot expansion + contrastive/distillation adaptation | frozen backbone에서 controlled capacity expansion | foundation LLM stream, deletion and serving state reuse | peer-reviewed, `SRC-STC-0049` |
| Memory-Statistics Tradeoff (ICLR 2026) | structural regularization state | curvature-aware regularization | memory complexity와 excess risk 사이 upper/lower bound | two linear regression tasks; LLM extrapolation unproven | peer-reviewed theory, `SRC-STC-0050` |
| Rate-Distortion Memory Compaction (2026) | multiple memory media | query-agnostic distortion formulation | KV/prompt/recurrent/agent compaction을 공통 언어로 비교 | repeated compaction benchmark and semantic distortion estimator | arXiv, `SRC-STC-0035` |
| OpenAI Dreaming (2026) | external synthesized memory | proprietary background synthesis | multi-year staleness/relevance and scale | operator details and independent evaluation | production E4, `SRC-STC-0052` |
| Mem0 Dream (2026) | external pattern/status graph | scheduled synthesis + ingest-time supersede/merge | growth, duplicate, contradiction, higher-order pattern | external evaluation and deletion cascade | product docs E4, `SRC-STC-0063` |

## 5. Parametric vs external: 현재 evidence가 지지하는 역할 분담

| Memory content | Default destination | Why | Parametric promotion condition |
|---|---|---|---|
| user fact, consent, current location | external system-of-record | mutable, deletable, provenance required | 원칙적으로 승격하지 않거나 reversible adapter에만 저장 |
| transient project state | external event/graph | freshness와 exact timestamp가 중요 | 높은 반복 사용과 stable schema가 확인될 때 |
| repeated strategy/pitfall | structured external rule first | evidence trace와 editing이 필요 | many-query reuse와 validation gain이 compile cost보다 클 때 |
| domain skill/procedure | adapter/expert candidate | retrieval 없이 행동 prior로 작동할 가치 | canary·rollback·old-task regression gate 통과 |
| rare fact tail | external text/graph | recursive training과 compaction에서 tail loss 위험 | promotion하더라도 source copy 유지 |
| universal updated knowledge | periodic global refresh | tenant별 duplication보다 global training 효율적 | freshness SLA가 batch cadence를 허용할 때 |

이 분업은 영구 규칙이 아니라 현재 evidence의 비용·governance 최적점이다. 미래에 reversible weight snapshots, exact unlearning, stable lifelong optimization이 성숙하면 경계가 이동할 수 있다.

## 6. Missing evidence ledger

업계·학계 모두 다음 공개 자료가 부족하다.

1. **Repeated-cycle curve:** 10–1,000회 wake/sleep cycle에서 retention, plasticity, calibration이 어떻게 변하는가.
2. **Capacity curve:** cumulative evidence와 active memory budget을 함께 늘렸을 때 어떤 content가 먼저 unreachable해지는가.
3. **Recursive-depth curve:** synthetic dream이 자기 출력을 몇 세대 재사용할 때 tail과 diversity가 무너지는가.
4. **State migration bytes:** episode → summary → adapter/expert 이동에 실제로 몇 byte가 HBM·DRAM·SSD·network를 통과하는가.
5. **Wake-SLA interference:** background job이 TTFT·ITL·P99 latency에 주는 영향과 preemption cost.
6. **Delete propagation:** source event 삭제가 summary, graph edge, embedding, adapter, optimizer snapshot에 언제·어떻게 반영되는가.
7. **Rollback semantics:** bad consolidation을 어떤 atomic unit으로 되돌리고 later memories를 어떻게 rebase하는가.
8. **Independent product evaluation:** vendor가 아닌 evaluator가 long-horizon reliability와 false-memory rate를 측정한 결과.
9. **Cross-version compatibility:** base model 교체 시 user/agent memory를 re-embed, recompile, discard 중 무엇을 하는가.
10. **Energy/endurance traces:** accelerator·memory device·storage의 energy, endurance, write amplification 측정.

이 공백 때문에 현재 결론은 “STC가 무조건 새로운 universal scaling law”가 아니라 “background memory transformation이 이미 제품 축으로 등장했으며, parametric STC의 broad adoption은 아직 조건부”다.

## 7. Audit verdict

### 7.1 무엇이 확인됐는가

- **external-memory sleep is real and deployed.** OpenAI Dreaming과 Mem0 Dream이 E4, Letta가 E3다.
- **parametric sleep is scientifically promising but deployment-immature.** Google의 `Language Models Need Sleep`과 Sleep-Time Compute가 연구 근거를 만들었으나, long-run capacity·governance·fleet economics가 비어 있다.
- **the strongest current architecture is hybrid.** raw episodes를 provenance-bearing external system-of-record에 남기고, background에서 summary/strategy를 만들며, 반복 reuse가 검증된 skill만 reversible adapter/expert로 승격하는 구조다.
- **capacity is not one number.** parameter bits, behavioral reachability, retrievable index set, governed active set, hot working set을 분리해야 한다.

### 7.2 무엇을 주장하면 안 되는가

- “OpenAI/Mem0 Dream이 foundation-model weight를 학습한다.” — 공개 evidence 없음.
- “Google/Meta/Microsoft가 per-user sleep cluster를 production 배포했다.” — 공개 evidence 없음.
- “biological NREM/REM stage가 특정 optimizer와 대응한다.” — metaphor 이상으로 확정할 근거 없음.
- “모든 memory system이 sleep-time compute다.” — ingest-time update와 query-time retrieval은 C2를 만족하지 않는다.
- “citation graph 4라운드가 완료되어 literature saturation을 달성했다.” — Semantic Scholar failure로 거짓이다.

## 8. Source locator index

| Claim family | Primary source | Fixed locator |
|---|---|---|
| OpenAI background synthesis and scale | `SRC-STC-0052` | official page frozen lines 39–53; 241–284 |
| Mem0 scheduled synthesis | `SRC-STC-0063` | official page frozen lines 81–145; 190–226 |
| Letta async post-response sleep | `SRC-STC-0059` | tagged code class 805; 988; 1056; 1169; 1215 |
| Letta transactional reflection | `SRC-STC-0060` | tagged code 2207–2234; 2281–2285 |
| Zep raw episode provenance | `SRC-STC-0066` | official docs frozen lines 163–168 |
| ReasoningBank limitation | `SRC-STC-0054` | “Distilling insights” section: direct append and future consolidation statement |
| STATE-Bench benchmark gap | `SRC-STC-0056` | “Quantifying the memory gap” and “Bring your own memory” sections |
| sequential fact reachability failure | `SRC-STC-0036` | paper results after 20 writes and recovery interventions |
| recursive model collapse | `SRC-STC-0041` | Nature abstract, Methods, Figures 1–4 |
| memory poisoning | `SRC-STC-0042` | NeurIPS abstract and Sections 3–5 |
