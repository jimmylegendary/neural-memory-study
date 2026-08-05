# Strongest Alternatives to Sleep-Time Compute

- source freeze: 2026-08-05
- principle: STC와 비교할 때 “no memory”가 아니라 각 problem class의 best available architecture를 사용한다.
- conclusion preview: 대안들은 사라지지 않는다. STC가 주류가 되더라도 대부분 wake/sleep/long-term-memory stack의 구성요소로 남는다.

## 1. Comparison axes

| Axis | Question |
|---|---|
| timing | query-time, online wake, post-task, scheduled background, release-time 중 언제 update하는가? |
| medium | prompt/KV/recurrent state/text/vector/graph/adapter/base weight 중 어디에 쓰는가? |
| write cost | inference-only인가, gradient update인가, index/graph rebuild인가? |
| reuse | one query, one session, many sessions, cohort, global 중 누가 재사용하는가? |
| capacity | append, fixed state, expansion, compaction, expiration 중 무엇인가? |
| correctness | exact source, lossy summary, implicit parametric behavior 중 무엇인가? |
| governance | provenance, delete, rollback, tenant isolation이 가능한가? |
| serving | TTFT/ITL, cache reuse, model residency, state migration cost가 무엇인가? |

## 2. Family-level matrix

| Family | Representative frontier | Best use | Why it can beat STC | Primary limitation | Likely future role |
|---|---|---|---|---|---|
| long context | FlashAttention/FlashDecode serving, long-context models | one-off documents, exact current evidence | no training lag; source tokens remain explicit | quadratic/log-linear attention traffic, distractors, repeated prefill | wake exact-evidence path |
| context compression / recurrent state | Memory Caching, NSTM, recurrent/SSM | long sequences and reusable prefix/state | compact execution state, low per-token cost | hidden-state portability, cache compatibility, retrieval/addressability | wake working memory and hot cache |
| external retrieval | RAG, Memorizing Transformers, LongMem | volatile factual knowledge | freshness, provenance, edit/delete, huge storage | retrieval miss, index drift, context cost | default factual long-term memory |
| structured agent memory | MemGPT/Letta, Zep, Mem0, ReasoningBank | user/project/strategy memory | explicit lifecycle and learned writer possible with frozen model | summary/graph errors, poisoning, policy quality | dominant deployed STC medium |
| test-time training | Titans, MIRAS, ATLAS, TNT | within-session adaptation, long context memorization | immediately adapts to current evidence | wake latency, state reuse, drift, rollback | wake fast state; supplies sleep candidates |
| continual learning | replay, EWC, GEM, Progress & Compress, MaRS | stable multi-task/domain adaptation | established retention operators and expansion | replay growth, task assumptions, plasticity loss | sleep training toolbox |
| model editing | ROME, MEMIT | small targeted factual corrections | fast localized parametric change | many-edit interference, locality, reversal | surgical patch, not general lifelong store |
| latent compilation | GenerativeAdapter, document-to-adapter family | stable corpus with high query reuse | retrieval-free amortized serving | compile cost, base-version incompatibility, sparse evidence | niche/high-reuse bridge |
| periodic refresh | continued pretraining/SFT/global post-training | universal knowledge and common behavior | global reuse amortizes expensive validation | freshness cadence and personalization lag | foundation release plane |
| explicit STC | Sleep-Time Compute, Language Models Need Sleep | predictable repeated future tasks | off-path compute and multi-query amortization | capacity, recursive data, governance, queueing | external now; parametric conditional |
| hybrid promotion | external source-of-truth + selective adapter/expert | mixed facts and skills | combines governance with compiled reuse | two-medium consistency and scheduling | most plausible architecture |

## 3. Long context

### What it solves

Long context preserves raw evidence and allows a query to decide relevance after it arrives. It is therefore the strongest baseline when future queries are hard to predict or source wording matters.

### Why STC does not replace it

- Sleep compilation is lossy unless the source remains available.
- one-off documents have reuse count near one, so compile cost is hard to recover.
- correction and audit often require exact source spans.

### Where STC wins

If the same corpus is repeatedly prefetched and later queries are predictable, an index/summary/adapter can amortize repeated prompt tokens and inference. The original Sleep-Time Compute result is precisely a reuse/predictability argument (`SRC-STC-0014`).

### Strong combined design

```text
canonical document store
  ├─ query-time long-context/RAG for exact evidence
  └─ sleep-compiled summary/index/adapter for common queries
          └─ fallback to source when confidence is low
```

## 4. External retrieval

### Why it is currently the default factual memory

External retrieval decouples world state from model release. New facts can be indexed without gradient training, and provenance/delete can operate on concrete objects. Zep Graphiti adds temporal relations; Mem0 adds supersede/merge/decay/synthesis; OpenAI Dreaming demonstrates background personalized synthesis.

### Failure modes

- embedding/index version drift
- candidate miss before reranking
- top-k competition as store grows
- context pollution and prompt injection
- graph stale edges
- source/derived deletion mismatch

### Latest direction

The frontier is no longer plain vector lookup. It includes temporal graph updates, episode provenance, background pattern synthesis, learned memory writer/policy, and memory-aware test-time scaling (`SRC-STC-0043`, `0046`, `0048`, `0051`, `0063`, `0065`).

### Verdict vs parametric STC

For exact, volatile, private, or regulated facts external retrieval remains stronger. Parametric compilation must demonstrate sufficient reuse and reversible behavior to win.

## 5. Structured agent memory

### What changed in 2026

1. ReasoningBank moves from raw trajectory retention to success/failure strategy distillation.
2. MemoPilot trains the memory updater with multi-turn RL while keeping the player frozen.
3. Memini uses coupled fast/slow graph-edge dynamics for consolidation/forgetting.
4. OpenAI Dreaming and Mem0 Dream provide production background synthesis.

This means the alternative to “train the main model while asleep” is not a static vector database. It is a **learned, scheduled external-memory system**.

### Why this alternative is strong

- model-agnostic and portable across base upgrades
- memory writer can be trained separately
- source lineage can remain explicit
- candidate updates can be reviewed or rolled back
- per-user isolation does not require per-user model replicas

### What remains hard

Semantic summaries can hallucinate or suppress rare events; a learned writer can reward-hack; memory is poisonable; retrieval still consumes tokens. Therefore external STC needs the same validation and governance rigor as parametric STC.

## 6. Test-time training and fast state

### Frontier

Titans, MIRAS, ATLAS, TNT, Nested Learning/Hope, Memory Caching, and NSTM model memory as online optimization or recurrent state (`SRC-STC-0017`–`0024`).

### Why it beats sleep for some tasks

- immediately incorporates current query/context
- no wait for cadence or queue
- can forget at session boundary by design
- avoids persisting sensitive data if state is ephemeral

### Why it does not solve later sessions automatically

Fast state can be opaque, model-version-specific, and difficult to reuse across requests. Prefix/recurrent cache hit rate, state identity, batch scheduling, and migration cost become serving problems. If persisted, it acquires the same capacity/delete/rollback burden as other memory.

### Combined boundary

```text
wake TTT/fast state = immediate acquisition and current-session compression
sleep consolidation = select, validate, and publish only durable value
```

## 7. Continual learning

### Established families

- replay: canonical, exemplar, latent, generated
- regularization: EWC-like importance, distillation
- constraint: GEM-like gradient projection
- isolation/expansion: task modules, experts, slots
- compress: teacher-to-student or active-to-knowledge-base

### Why it is not “an old alternative that STC replaces”

These are the training operators STC would execute. Sleep changes the lifecycle and scheduler; it does not eliminate optimizer choice, data selection, plasticity control, or capacity accounting.

### Latest strongest points

- MaRS gives statistically grounded slot expansion and distillation for frozen large pretrained models (`SRC-STC-0049`).
- Memory-Statistics Tradeoff gives a controlled theoretical memory/risk tradeoff (`SRC-STC-0050`).
- Continual Facts shows study-data design matters but exposes reachability collapse in weights (`SRC-STC-0036`).

### Verdict

CL remains essential for parametric STC, but open-ended LLM deployment is harder than benchmark task sequences. A sleep paper that only reports one consolidation cycle is not a lifelong-learning result.

## 8. Model editing

### Strength

ROME/MEMIT attempt targeted factual association changes without full retraining. For urgent small corrections they can be cheaper than a complete sleep cycle.

### Weakness

- locality is evaluation-dependent
- edits interact as count grows
- old knowledge may remain reachable through other prompts
- exact reversal and base upgrade are nontrivial

### Place in stack

Model editing is a patch mechanism. A sleep system can batch and validate edits, but should keep external canonical evidence and treat edited weights as rebuildable derived state.

## 9. Latent compilation

### Idea

Compile a stable document set into adapter/latent state once and serve many queries cheaply. This lies between external retrieval and general weight training.

### Why it is promising

- base model can remain frozen
- artifact can be versioned per corpus/tenant
- serving can avoid long prompt and retrieval tokens

### Why it is not yet the default

C06 remained a measurement gap: public studies rarely sweep all of compilation cost, query reuse, correction rate, base-version churn, and lifecycle bytes. A good benchmark win at fixed corpus does not identify the economic crossover.

### Decision inequality


\[
R(H)\,\Delta U
>
C_{\mathrm{compile}}+C_{\mathrm{serve}}+C_{\mathrm{govern}}+C_{\mathrm{risk}}.
\]

Without a calibrated valid-reuse estimate, this remains a candidate policy.

## 10. Periodic refresh

### Why global refresh remains powerful

If knowledge is shared across millions of users, one high-quality global training run amortizes better than millions of personalized sleep jobs. Global refresh also centralizes safety evaluation and release management.

### Where it loses

- user/tenant-specific knowledge
- freshness below release cadence
- local tools/procedures
- private data that cannot join central training

### Likely equilibrium

```text
global common knowledge      → periodic refresh
volatile exact facts         → external retrieval
user/tenant patterns         → external background synthesis
stable high-reuse skill      → selective adapter/expert promotion
```

## 11. Hybrid promotion

### Architecture

1. append wake evidence to a provenance-bearing event log.
2. perform cheap ingest-time extraction/contradiction checks.
3. run background external synthesis and retrieval evaluation.
4. estimate future reuse, validity horizon, and correction/deletion risk.
5. compile only high-reuse stable items into reversible adapter/expert state.
6. preserve canonical evidence and rebuild recipe.
7. demote or rebuild when the base model, evidence, or consent changes.

### Why it is strongest today

It matches the observed industry direction—external background synthesis—while preserving a path to parametric efficiency. It also fits the strongest negative evidence: exact facts can remain externally reachable even if weight write accessibility degrades.

### New failure modes

- external and parametric versions disagree
- route chooses stale compiled state
- delete reaches source but not adapter
- low-reuse items thrash across promotion thresholds
- base upgrade invalidates many artifacts at once

Hybrid therefore needs promotion/demotion hysteresis, compatibility digests, and transactional publish.

## 12. Pairwise decision table

| Workload condition | Best first choice | Add STC when… |
|---|---|---|
| future query unknown, one-off source | long context / external retrieval | repeated use appears and stable summary/index helps |
| rapidly changing facts | temporal external memory | background contradiction/index maintenance is needed |
| repeated agent mistakes | structured strategy memory | consolidation can use many trajectories; memory writer can be learned |
| stable repeated domain procedure | adapter/expert or periodic fine-tune | foreground latency must be isolated and reuse pays compile cost |
| within-session nonstationarity | TTT/fast state | durable candidates emerge after session |
| small urgent factual correction | external override or model editing | corrections can be batched, validated, and rebuilt |
| global common update | periodic refresh | local exceptions or sub-release freshness matter |
| privacy-sensitive personalization | device/local external memory | secure reversible local adapter is proven cheaper/better |
| bounded hot memory, growing archive | compaction/tiering | background jobs can reclaim total physical bytes, not just hide them |

## 13. Required matched baselines

Every future STC study should include at least:

1. no-memory frozen model
2. long-context full source
3. external retrieval at matched canonical bytes
4. external background summary/graph synthesis
5. wake TTT/fast-state adaptation
6. periodic batch fine-tuning/global refresh
7. model editing for small-update workloads
8. hybrid external-to-adapter promotion
9. same total FLOPs scheduled without a sleep boundary

The comparison must match not only model parameter count but total answer-bearing state, validation effort, data movement, and retained recovery bytes.

## 14. Bottom line

다른 접근법은 가능성이 없는 것이 아니다. 오히려 최신 산업 방향은 external retrieval을 더 정교한 memory lifecycle로 진화시키고, 학계는 main model을 바꾸는 대신 memory writer/router를 학습하는 방향도 강하게 밀고 있다.

STC가 이들을 “대체”할 가능성보다 더 현실적인 미래는 다음이다.

> long context와 TTT가 wake acquisition을 담당하고, external memory가 canonical long-term state를 담당하며, sleep scheduler가 compaction·validation·selective parametric promotion을 담당한다.
