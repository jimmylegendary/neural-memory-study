# Agent-Memory Systems Primary-Source Audit

- Audit date: 2026-07-25
- Scope: MemGPT/Letta, Mem0, Zep/Graphiti, LangMem/LangGraph, Memento,
  Memento-Skills, Evolve, Mela, MIRROR, and HeLa-Mem
- Evidence boundary: official papers, documentation, repositories, tagged
  source, and release metadata
- Product boundary: managed-service behavior is not imputed to the open-source
  library, and current behavior is not imputed backward to an earlier paper
- Status: pre-ingestion audit. Stable evidence IDs are assigned after the
  evidence registry is live.

## Strict classification rule

An implementation counts as strict sleep-time compute only if all three
conditions hold:

1. accumulated wake experience is an input;
2. transformation runs outside the current user-response critical path or on
   an asynchronous snapshot;
3. the result persists and is visible to a later wake.

An `async def`, a “background memory” label, or a product feature named
“dream” is not sufficient. Scheduler ownership, response-path placement,
persistent destination, and next-wake visibility must each be verified.

| System/version | Wake experience | Off critical path | Persistent next-wake reuse | Result |
|---|---:|---:|---:|---|
| MemGPT 2023 | yes | no by default | yes | PARTIAL |
| Letta *Sleep-time Compute* paper | limited shared context | yes | context artifact | PARTIAL |
| Letta API Sleeptime Agent | yes | yes | yes | PASS |
| Letta Code Dream/Reflection | yes | yes | yes | PASS |
| Letta AI Memory SDK | yes | yes | yes | PASS for per-turn extraction |
| Mem0 paper architecture | yes | partly explicit | yes | PARTIAL |
| Mem0 Platform V3 | yes | yes | yes | PASS |
| Mem0 OSS `Memory.add` | yes | no by default | yes | PARTIAL |
| Mem0 OpenClaw auto-dream | yes | runs before next response | yes | PARTIAL |
| Zep Cloud | yes | yes | yes | PASS |
| Graphiti OSS | yes | no owned scheduler | yes | PARTIAL |
| LangMem `ReflectionExecutor` | yes | yes | yes | PASS |
| LangMem inline manager | yes | no | yes | PARTIAL |
| LangGraph `BaseStore` alone | no transformation | n/a | storage only | FAIL |
| Memento 2 paper loop | yes | no | yes | FAIL |
| Memento-Skills paper loop | yes | no | yes | FAIL |
| Memento-Skills DreamDaemon, post-paper code | yes | yes | yes | PASS |
| Evolve explicit consolidation cycle | yes | yes | yes | PASS |
| Mela functional neural memory | current sequence only | no | no cross-call state | FAIL |
| MIRROR Talker/Thinker/Controller | yes | yes | yes | PASS, with ordering caveat |
| HeLa-Mem paper architecture | yes | boundary not specified | yes | PARTIAL |
| HeLa-Mem released LongMemEval default | yes | no reflective sleep path | yes | FAIL for claimed sleep mechanism |

In this table, “wake experience” includes a deployed query that causes a
persistent artifact. If C1 is restricted to task outcome, user correction, or
reward, Evolve is only partial on that column because its teacher sees the
query/category rather than interaction feedback.

All audited PASS systems update external, non-parametric memory. This audit did
not find a production system that continually distils, reinforces, or otherwise
trains base-model weights during a deployed sleep phase.

The last-mile paper/code audit used the following exact snapshots. A digest
identifies reviewed bytes; it does not itself grant redistribution rights.

| Work/artifact | Reviewed identity | SHA-256 or Git commit |
|---|---|---|
| Mela paper | arXiv `2605.10537` v1 PDF | `bb13a4193727a50f5e32c0221cffeb3ded5d3f94402735606b209726ec3cdea4` |
| Mela code | GitHub `Musubi-ai/Mela` | `fa74b2b21c34915df016977ff6546b0a7f0bc545` |
| MIRROR lineage paper | arXiv `2506.00430` v3 PDF | `574f5d10d70060a11f9e6822e11ebc5c1681be55327f9d54e3df49543d4cad6f` |
| MIRROR workshop paper | OpenReview `IviO4bIZc7` | official workshop PDF reviewed separately |
| MIRROR code | GitHub `arcarae/MIRROR` | `78c6383a3d538250990a1edf1a3dbf7a1aef2ada` |
| HeLa-Mem paper | ACL Anthology `2026.acl-long.625` PDF | `228e8fd96362a0d6ec46651a6329dbfaad7652e0a4a8a959d94aa6387fe9dbf0` |
| HeLa-Mem checklist | ACL reproducibility checklist PDF | `29b5bc7c1e4dac27702b41ce3dcf68b5a8b3ba817a639d9e4a7e8dfd24016109` |
| HeLa-Mem code | GitHub `ReinerBRO/HeLa-Mem` | `19adb2d709aa74db759835dbc78bc6fa4f4349ef` |

## 1. MemGPT

**Identity and chronology.** The
[MemGPT paper](https://arxiv.org/abs/2310.08560) was first public on
2023-10-12. It predates the Letta company/framework split and the later
sleeptime-agent implementation.

### Actual memory lifecycle

- Every user and assistant message is written to recall storage.
- Prompt state contains working and core memory; archival memory is an external
  searchable store.
- At roughly 70% context use, a system warning asks the LLM to preserve
  important information.
- At context pressure, approximately half the context is evicted and a
  recursive summary is created.
- The LLM can call tools to edit core memory or move information to archival
  memory.
- Timed events are possible, but the paper does not evaluate an autonomous
  background consolidation service. Normal paging, summarization, and memory
  tools execute inside the foreground agent loop.

There is no weight training, RL, distillation, dataset augmentation, or
optimizer. The operator is inference-time tool use plus a context-pressure
policy.

### Evidence and boundaries

On the Deep Memory Retrieval benchmark, the paper reports:

| Model | Baseline | MemGPT |
|---|---:|---:|
| GPT-3.5 | 38.7 | 66.9 |
| GPT-4 | 32.1 | 92.5 |
| GPT-4 Turbo | 35.3 | 93.4 |

The prompt window is bounded through paging and recursive summarization, but
evicted messages are retained in recall storage “indefinitely.” External
capacity, retrieval competition, TTL, semantic deletion, and lifetime
consolidation compute are not bounded. Raw recall provides a partial recovery
path, but there is no claim-level lineage, transactional rollback, or
unlearning protocol.

**Strict result:** PARTIAL. Persistent long-term memory is real; the default
transformation is on the response path.

The code and benchmark data were released; the later Letta repository is
Apache-2.0. Results should be cited to the paper version, not to present Letta
behavior.

## 2. Letta Sleep-Time Compute Paper

**Identity.** Lin et al.,
[*Sleep-time Compute*](https://arxiv.org/abs/2504.13171), first public
2025-04-17.

The paper assumes a shared context \(C\) and unknown future query. Before those
queries arrive, an offline model transforms \(C\) into \(C'\), amortizing more
reasoning over later queries. It is an offline context-compilation study—not a
complete continual-memory lifecycle and not model training.

Reported outcomes include:

- approximately five-fold less test-time compute at matched accuracy;
- up to 13 percentage points on Stateful GSM-Symbolic;
- up to 18 points on AIME;
- approximately 2.5-fold lower average per-query cost when context is shared.

There is no deployed event stream, parametric update, distillation, RL,
optimizer, capacity policy, deletion, rollback, or source-lineage mechanism.
The durable object is the transformed context artifact.

**Strict result:** PARTIAL. It establishes the offline-compute allocation
principle, but not by itself a long-running agent-memory system.

## 3. Letta API Sleeptime Agent

### Implementation anchors

- [`sleeptime_multi_agent_v4.py`](https://github.com/letta-ai/letta/blob/0.16.8/letta/groups/sleeptime_multi_agent_v4.py)
- [`create_sleeptime_agent_async`](https://github.com/letta-ai/letta/blob/0.16.8/letta/server/server.py)
- [`enable_sleeptime` agent schema](https://github.com/letta-ai/letta/blob/0.16.8/letta/schemas/agent.py)
- [sleeptime v2 system prompt](https://github.com/letta-ai/letta/blob/0.16.8/letta/prompts/system_prompts/sleeptime_v2.py)

All four anchors are pinned to audited Letta tag `0.16.8` (commit
`1131535716e8a31c9a437f8695e25ac98f203a24`).

The foreground agent returns first. When a turn counter reaches
`sleeptime_agent_frequency`, the service collects unprocessed message pairs and
dispatches a separate run with `safe_create_task`; the default creation path
uses frequency five. The sleeper shares persistent memory block IDs with the
foreground agent and edits them through `memory_replace`, `memory_insert`,
`memory_rethink`, and `memory_finish_edits`. Run state is also persistent.

The prompt instructs removal or restructuring of outdated and redundant
information. Destination state is token-space core-memory blocks, not model
weights. Isolation exists at actor, project, agent, and group levels.

Capacity is bounded locally by block size, but there is no global learned
importance allocator. Run records provide operational history; exact
source-claim lineage and automatic rollback remain weak.

**Strict result:** PASS. Wake transcript, post-response background transform,
shared persistent destination, and next-wake use are all present in code.

Audited current API version: `0.16.8`, Apache-2.0. The initial background
sleeptime-agent work belongs to the 2025-04 lineage.

## 4. Letta Code Dreaming and Reflection

### Implementation anchors

- [trigger configuration](https://github.com/letta-ai/letta-code/blob/v0.28.18/src/cli/helpers/memory-reminder.ts)
- [post-turn trigger](https://github.com/letta-ai/letta-code/blob/v0.28.18/src/cli/helpers/post-turn-reflection.ts)
- [background launcher and worktree merge](https://github.com/letta-ai/letta-code/blob/v0.28.18/src/cli/helpers/reflection-launcher.ts)
- [reflection subagent prompt](https://github.com/letta-ai/letta-code/blob/v0.28.18/src/agent/subagents/builtin/reflection.md)
- [`letta dream`](https://github.com/letta-ai/letta-code/blob/v0.28.18/src/cli/subcommands/dream.ts)

Default triggering is `compaction-event`; alternatives are `step-count` with
default threshold 25 or off. After a foreground turn, the system captures the
transcript delta. Only one reflection per agent runs at a time; additional
requests queue.

A silent subagent works in a separate MemFS Git worktree. It commits memory and
skill changes, merges them into the parent MemFS, and recompiles later
conversation context. A manual `letta dream` path invokes the same class of
work.

The reflection policy can:

- synthesize patterns across transcripts;
- extract durable facts, mistakes, preferences, and reusable procedures;
- prefer newer evidence when resolving contradictions;
- keep active memory compact while moving detail to referenced external files;
- archive retired material in `ARCHIVE.md`;
- delete explicit forget requests, sensitive material, errors, and junk;
- promote reusable procedures into skills.

Destinations are system-memory Markdown, external files, and skills. Git
commits provide version provenance and rollback; isolated worktrees reduce
foreground/background conflicts; unresolved merge conflicts become pending
reminders. Raw recall remains separately available.

There is no parametric training, distillation, or RL. The subagent performs LLM
inference.

**Strict result:** PASS. Among audited open systems, this is the most explicit
combination of sleep consolidation, tier movement, deletion, conflict
isolation, and rollback.

Audited tag: `v0.28.18`, Apache-2.0. The repository became public on
2025-10-24; the current dreaming path primarily developed during 2026-02
through 2026-07.

## 5. Letta AI Memory SDK

The [official SDK](https://github.com/letta-ai/ai-memory-sdk) begins on
2025-08-30; audited version `v0.2.0`, Apache-2.0.

- Each subject receives a separate subconscious agent.
- `add_messages` launches an asynchronous Letta run that updates memory blocks.
- Callers can use `wait_for_run` to observe completion.
- Subject-agent deletion and optional archival passages are supported.
- Persistent state is external token memory, not weights.

**Strict result:** PASS for asynchronous turn-to-memory conversion.

The README separately lists “offline collective revisioning of all data” as a
roadmap item. That stronger whole-memory sleep should be recorded as
unimplemented, not inferred from the per-turn async path.

## 6. Mem0

### Chronology and paper architecture

Mem0 code first appeared on 2024-07-12 when the existing Embedchain repository
shifted to Mem0. The [Mem0 paper](https://arxiv.org/abs/2504.19413) was first
public on 2025-04-28 and was accepted to ECAI 2025. Audited current OSS package:
`mem0ai` 2.0.13, Apache-2.0.

The paper uses a conversation summary plus the ten most recent messages to
extract candidate facts. It searches the ten most similar vector memories, and
a second LLM chooses ADD, UPDATE, DELETE, or NOOP. The graph variant extracts
entities and triples and invalidates conflicting relations. The paper
explicitly makes only global-summary generation asynchronous; the complete
memory mutation pipeline is not consistently off-path.

There is no weight training, RL, distillation, or learned optimizer. LLM
inference and embedding models implement the operator.

| Paper method | LoCoMo overall | p95 total latency | Memory tokens |
|---|---:|---:|---:|
| Mem0 | 66.88 | 1.440 s | 1,764 |
| Mem0 Graph | 68.44 | 2.590 s | 3,616 |
| Full context | 72.90 | 17.117 s | 26,031 |

The paper reports 26% relative J-score improvement over OpenAI Memory, more
than 90% fewer tokens, and 91% lower p95 latency. These are vendor-run
comparisons and must retain their evaluation and accounting qualifiers.

**Paper strict result:** PARTIAL.

### Managed Platform V3

The current [V3 Add API](https://docs.mem0.ai/api-reference/memory/add-memories)
returns an `event_id` and `PENDING` state immediately, then processes a
background queue. The present pipeline is single-pass ADD-only: one LLM
extraction appends memories rather than automatically updating or deleting old
ones. User, agent, application, run, organization, and project scopes isolate
state. Later vector/entity searches expose results to wake.

**Strict result:** PASS.

### Open-source library

[`Memory.add`](https://github.com/mem0ai/mem0/blob/main/mem0/memory/main.py)
does not return until extraction, comparison, and storage complete.
`AsyncMemory` exposes nonblocking I/O syntax but does not own a scheduler. An
application can enqueue it, but that architecture should not be attributed to
the default library.

**Strict result:** PARTIAL.

### Capacity, decay, and deletion

Platform V3's ADD-only path grows by default. Explicit update, delete, batch
delete, user delete, export, and history APIs exist. `expiration_date` hides an
expired memory from search but does not establish physical erasure.

The [Memory Decay feature](https://docs.mem0.ai/platform/features/memory-decay)
is off by default. It scales search scores by 0.3--1.5 using up to the 20 most
recent accesses; retrieval reinforcement is fire-and-forget. This is ranking,
not semantic consolidation or deletion. History is present, but one-click
transactional rollback is not.

### OpenClaw auto-dream

The [dream gate](https://github.com/mem0ai/mem0/blob/v2.0.13/integrations/openclaw/dream-gate.ts)
defaults to 24 hours, five sessions, and 20 memories. Immediately before the
next user prompt it injects an `<auto-dream>` operation that can merge
duplicates, remove secrets/noise/TTL-expired content, rewrite unclear entries,
and call add/update/delete.

This source anchor is pinned to audited Mem0 tag `v2.0.13` (commit
`ca2abca2b884e038d3e525070e79d3057ef2012c`).

This is a real consolidation operator but executes before the next user
response rather than as an off-path background job.

**Strict result:** PARTIAL.

## 7. Zep Cloud and Graphiti

### Representation and chronology

Graphiti's first commit dates to 2024-08-13. The
[Zep paper](https://arxiv.org/abs/2501.13956) was first public on 2025-01-20.
Audited Graphiti tag: `v0.29.2`, Apache-2.0. Zep Cloud is a proprietary managed
service and must be analyzed separately.

The memory has three layers:

1. an episode subgraph that stores raw message, text, or JSON verbatim;
2. a semantic subgraph of entities and temporal fact edges;
3. a community subgraph with map-reduce summaries of entity clusters.

Semantic edges store source episode IDs. Transaction time
`created_at/expired_at` is distinct from event time `valid_at/invalid_at`.
Contradictory new facts normally set `invalid_at` on old edges rather than
erasing history.

Implementation anchors:

- [`Graphiti.add_episode`](https://github.com/getzep/graphiti/blob/v0.29.2/graphiti_core/graphiti.py#L980)
- [raw episode node](https://github.com/getzep/graphiti/blob/v0.29.2/graphiti_core/nodes.py#L318)
- [temporal entity edge](https://github.com/getzep/graphiti/blob/v0.29.2/graphiti_core/edges.py#L263)
- [`remove_episode`](https://github.com/getzep/graphiti/blob/v0.29.2/graphiti_core/graphiti.py#L1765)

### Graphiti OSS

`add_episode` performs extraction, deduplication, embedding, invalidation, and
database writes before returning. Its docstring recommends FastAPI background
tasks or Celery, but the library does not own that queue.

**Strict result:** PARTIAL. Persistence and transformation exist; autonomous
off-path scheduling does not.

### Zep Cloud

Official documentation establishes:

- [episodes](https://help.getzep.com/episodes) retain raw input verbatim;
- graph/thread add returns before asynchronous graph construction completes;
- [webhooks](https://help.getzep.com/v3/webhooks) expose
  `episode.processed` and `ingest.batch.completed`;
- [batch ingestion](https://help.getzep.com/adding-batch-data) accepts as many
  as 50,000 items, 350 per add call, and separates processing from live
  traffic;
- episodes in one graph are processed sequentially, generally under ten
  seconds but occasionally over minutes.

This creates a measurable eventual-consistency gap: the next wake can precede
memory publication.

**Strict result:** PASS.

### Evidence and lifecycle boundaries

- DMR: Zep 94.8%, MemGPT 93.4%, full context 94.4%.
- LongMemEval with GPT-4o-mini: full context 55.4% versus Zep 63.8%; latency
  31.3 s versus 3.20 s.
- LongMemEval with GPT-4o: full context 60.2% versus Zep 71.2%; latency 28.9 s
  versus 2.58 s.
- Average context falls from 115K to 1.6K tokens.
- The single-session-assistant category degrades, preventing a universal
  benefit claim.

Episode, node, edge, thread, and user-level right-to-be-forgotten deletion APIs
exist. However, removing an episode does not regenerate a shared entity
summary, and it does not automatically restore an edge that the removed episode
had invalidated. This is deletion from selected stores, not full causal erase
or semantic rollback.

The raw-to-derived source relation gives comparatively strong provenance.
Official documentation states no hard graph-count or graph-size limit, while
retrieval results and payloads are bounded. No automatic global capacity
compaction or forgetting policy was found. Cloud offers project/user isolation,
RBAC/ABAC, audit, and retention controls; OSS Graphiti mainly provides
`group_id` partitioning. The paper proposes a fine-tuned extractor as future
work; reported construction uses GPT-4o-mini inference.

## 8. LangMem and LangGraph

**Chronology.** LangGraph's repository begins on 2023-08-09. LangMem begins on
2025-01-21, with `ReflectionExecutor` added on 2025-02-05. Audited LangMem
version: 0.0.30, MIT.

Implementation anchors:

- [background quickstart](https://github.com/langchain-ai/langmem/blob/main/docs/docs/background_quickstart.md)
- [delayed processing](https://github.com/langchain-ai/langmem/blob/main/docs/docs/guides/delayed_processing.md)
- [`ReflectionExecutor`](https://github.com/langchain-ai/langmem/blob/main/src/langmem/reflection.py)
- [`MemoryStoreManager`](https://github.com/langchain-ai/langmem/blob/main/src/langmem/knowledge/extraction.py)
- [LangGraph `BaseStore` and TTL](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint/langgraph/store/base/__init__.py)

The quickstart's first “background” example awaits the manager inside the
response function and is therefore foreground. The actual off-path mechanism is
`ReflectionExecutor.submit`.

Local execution uses a worker thread and priority queue, supports
`after_seconds`, debounces by thread, and cancels/replaces a pending job when
new messages arrive. Remote execution creates a LangGraph run with
`after_seconds` and `multitask_strategy="rollback"`. A local worker can vanish
in a serverless deployment, so a durable remote executor is necessary.

The memory manager semantically searches related memories and asks an LLM to
emit insert, update, and delete tool calls. Its prompt asks for
deduplication/compression/consolidation. The public factory defaults
`enable_deletes=False`. Results persist as JSON documents and a vector index in
LangGraph `BaseStore`, with templated organization, user, or agent namespaces.

`query_limit=5` bounds the consolidation working set, not lifetime memory.
There is no automatic global eviction, importance budget, or transactional
rollback. Store adapters may offer TTL and a periodic best-effort sweep, but
expiration is not semantic consolidation. `InMemoryStore` is process-local;
production durability requires a persistent adapter such as Postgres.
Default provenance is limited to created/updated timestamps unless the
application stores source-message links.

There is no model-weight learning, RL, or distillation. A separate prompt
optimizer can reflect over trajectories and feedback and rewrite prompt text;
combined with the background executor, that becomes textual procedural-memory
sleep, not parametric training.

**Strict results:**

- `ReflectionExecutor`: PASS.
- Inline `MemoryStoreManager`: PARTIAL.
- `BaseStore` alone: FAIL—it stores state but performs no consolidation.

The official repository does not provide a quantitative memory benchmark.

## 9. Memento 2, Memento-Skills, and the later DreamDaemon

These three objects share a project name but not an evidence boundary. The
papers establish wake-time learning and theory; a later repository release
adds an actual background sleep implementation. They must not be merged into
one retrospective “paper system.”

### Memento 2

[*Memento 2: Learning by Doing for Agentic Language
Models*](https://arxiv.org/abs/2512.22716) first appeared as *Memento-II* on
2025-12-27. It keeps the language model frozen and updates external cases, a
Parzen policy \(\mu\), and a value function \(Q\) through a
read--act--feedback--write loop. The current task's feedback can change the
policy and memory before that task is finished, so its operational lifecycle
is wake-time learning: `C1=Y`, `C2=N`, and only the later-reuse condition is
theoretically satisfied.

Its strongest contribution here is a conditional convergence and value-error
story, not evidence of a sleep service. In particular, Corollary 15 gives a
bound of the form

\[
\lVert V^\star-V^{\pi_M}\rVert_\infty
\le
\frac{2R_{\max}}{(1-\gamma)^2}
\left[\epsilon_{\mathrm{LLM}}(r_M)+\delta_M\right],
\]

under local consistency and coverage assumptions. The surrounding theorems
also require bounded rewards, \(\gamma<1\), stationary memory dynamics during
evaluation, finite current memory and, for the two-timescale result, bounded
iterates and a compact attractor. This exposes a capacity tension rather than
solving one: dense coverage asks \(r_M\to0\), while a fixed finite memory needs
an explicit coreset, merge, eviction, or compression rule to make that limit
possible. The preprint supplies no experiment, released dataset, checkpoint,
or fixed-budget forgetting theorem.

**Strict result:** FAIL. It is an important wake-learning/theory precursor,
not a demonstrated deferred consolidation cycle.

### Memento-Skills paper

[*Memento-Skills: Let Agents Design
Agents*](https://arxiv.org/abs/2603.18743), first public 2026-03-19, stores
natural-language and code skills outside a frozen Gemini-3.1-Flash model.
After a failed attempt, the system uses the ground-truth answer in its judge,
reflects, edits or creates a skill, and retries the same question. That gives
`C1=Y,C2=N,C3=Y`: persistent procedural learning is real, but the update
remains part of the current task loop rather than sleep.

The separately trained router starts from an approximately 8K-skill catalog,
samples roughly 3K skills, generates synthetic goals, filters them with a
judge, and fits Qwen3-Embedding-0.6B with InfoNCE. The paper calls this
single-step offline RL; operationally it is supervised contrastive
contextual-bandit fitting because no multi-step return or deployed wake
trajectory enters the optimizer. Reported router metrics are:

| Router | R@1 | R@5 | R@10 | real-skill hit | judge success |
|---|---:|---:|---:|---:|---:|
| BM25 | .32 | .47 | .53 | .29 | .50 |
| Qwen embedding | .54 | .79 | .86 | .53 | .79 |
| Memento router | .60 | .82 | .90 | .58 | .80 |

On the paper's selected splits, GAIA rises from `65.1→91.6` on 100 learning
items and `52.3→66.0` on 65 held-out items; HLE rises from `30.8→54.5` on 788
learning items and `17.9→38.7` on 342 held-out items. The skill bank grows from
5 to 41 for GAIA and from 5 to 235 for HLE. These are single reported runs:
there are no seeds, confidence intervals, full cost accounting, or released
item-ID split for HLE.

The closest paper-time repository tree,
[`52c75ebb6eaa422168c8cd473cc9ed34c71165ba`](https://github.com/Memento-Teams/Memento-Skills/tree/52c75ebb6eaa422168c8cd473cc9ed34c71165ba),
contains the agent framework and public skill catalog, but not the paper's
router-training/evaluation pipeline, benchmark split artifacts, checkpoint,
failure-attribution evaluator, rewrite rollback, or unit-test gate. Its
runtime presents all local skill descriptions to the LLM for selection, so
wake prompt work grows with the catalog even when retrieval output is small.
There is no cap, deduplication law, eviction rule, source provenance, versioned
delete fan-out, or semantic rollback for the learned skill population.

**Strict result:** FAIL for the paper loop.

### Post-paper DreamDaemon

The repository's
[`v0.3.0` implementation](https://github.com/Memento-Teams/Memento-Skills/tree/71ac933ea1381d53389a2426f59634e0182071b8)
first added `DreamDaemon` on 2026-04-22, 34 days after the paper. A completed
response is summarized, appended to `_staging.md`, and eligible for a quick
asynchronous consolidation. A daemon also scans every 600 seconds; after the
24-hour/default staging gate, an LLM can create, update, or delete topic
Markdown and rebuild an index. The next wake injects that index and at most
three selected topics.

This code path satisfies `C1=C2=C3=Y` for external textual memory. It is not
evidence for the paper's reported benchmark gains and has no quantitative
evaluation of Dream. Defaults and implementation also diverge: the effective
gate is five staged sessions or 10 KB, while `dream.min_sessions=2` is not the
gate actually consulted. The quick post-session path will often consume
staging before the deeper periodic pass.

The implementation has per-file prompt/read limits and a lock, but no global
memory cap, source-level provenance, version lineage, tombstone propagation,
transactional publication journal, or rollback. Topic writes, deletion, index
rewrite, and staging clear are separate filesystem operations; interruption
can therefore publish a partial derived state or clear input without a
recoverable transaction. The prompt mentions skill candidates, while this
audited code path writes topic memory and its index, not paper-style skill
artifacts.

**Strict result:** PASS at the code level, with low empirical evidence and a
post-paper chronology qualifier.

## 10. Evolve

[*Evolve: A Persistent Knowledge Lifecycle for Small Language
Models*](https://arxiv.org/abs/2604.23424), first public 2026-04-25, combines a
local Qwen3.5 2B model with external section memory. The audited
[GitLab repository](https://gitlab.com/dikran.hovagimian/evolve) has one public
commit,
`edbf8adaec3506f2b2372f1913e887ef07b542f0`; its author timestamp is
2026-04-25 and committer timestamp is 2026-04-27, so an exact independently
recoverable arXiv-submission tree does not exist.

### Lifecycle and phase boundary

The query path classifies a question into a 42-category taxonomy, searches
staging and canonical sections, calls a GLM-4.7 teacher on a miss, and may
refresh expired hits inline. New sections are immediately searchable within
the same query. Acquisition and TTL refresh are therefore `Q/W`, not sleep.

| Operator | Input that causes the write | Current response waits/uses it | Later persistent reuse | Phase |
|---|---|---:|---:|---|
| acquire | query/category; no outcome feedback | yes | yes | `Q/W` |
| TTL refresh | retrieved expired sections | yes | yes | `Q/W` |
| consolidate | staged query-derived sections | no | yes | `S` under broad wake-experience C1 |

The separate consolidation pass is strict:

1. discard ephemeral or expired staging;
2. promote a staging section directly if no canonical overlap exists;
3. at cosine similarity at least `.85` within a category, ask a teacher to
   compile canonical plus staging into zero, one, or multiple replacements;
4. compare later same-cycle staging items against the newly promoted state;
5. clear staging after the whole loop succeeds.

The paper executes cold questions, warm questions, an explicit sleep cycle,
and post-consolidation questions. The repository also contains a nightly
`SleepScheduler`, disabled by default, with a 03:00 default and minimum staging
count of one; manual sleep is available. It is attached only in the persistent
interactive application, uses host/JVM local time, serializes scheduler-owned
jobs, and does not stop foreground queries or coordinate a separate manual or
cross-process sleep. It has no batch-size, teacher-call, wall-time, timeout,
checkpoint, or priority budget.

Under this audit's broad C1 definition, query-triggered persistent sections
are accumulated wake experience and the external transform satisfies
`C1=C2=C3=Y`. Under the narrower definition “interaction outcome or feedback
becomes learning data,” C1 is only partial: the teacher receives query and
category, not reward or user correction. That distinction must travel with
the PASS label.

### Results and the decisive evaluation qualifier

The local model uses `mxbai-embed-large-v1` embeddings, while GPT-OSS-120B
judges answers independently of the GLM teacher. The repository releases 250
questions and per-answer CSVs for each benchmark. Selected results are:

| Benchmark | no-memory baseline | cold section memory | post-sleep section memory | teacher calls/query, cold→post | canonical sections, pre→post |
|---|---:|---:|---:|---:|---:|
| custom | 33.0 | 84.0 | 85.0 | 1.25→.58 | 442→300 |
| Natural Questions | 20.4 | 60.2 | 61.4 | 1.15→.74 | 471→324 |
| TriviaQA | 30.4 | 82.4 | 84.0 | 1.26→.96 | 543→361 |

Scores in the table use the paper's semantic judge and its principal
non-augmentation path where applicable. The paper's section-versus-chunk claim
is not a standard source-document RAG comparison and is not a controlled
representation-only test: each mode separately generates teacher content.
For the released custom cold CSV, the score gap is roughly two points rather
than the paper table's larger reported gap, and several paper/CSV values
disagree. MMLU is the important negative control: a `70.4` baseline falls to
`66.8` with suppressive section memory and reaches only `68.4` with
augmentation.

The largest limitation is not visible in the aggregate table. Repository
inspection shows that cold, warm, and post-sleep phases use the **same 250
questions in the same order** for custom, Natural Questions, TriviaQA, and
MMLU. The falling teacher-call rate primarily demonstrates repeated-query
cache reuse. It is not held-out evidence that consolidated sections transfer
to new but related future queries. Wilson intervals over questions quantify
within-run binomial uncertainty; with one lifecycle run, they do not estimate
model-, teacher-, order-, or sleep-run variance, and overlapping intervals are
not a paired-difference test.

The sleep-specific accuracy deltas are small: custom `+1.0` score point,
Natural Questions `-0.6`, and TriviaQA `+1.2` from cold to post. There is no
parallel no-sleep lifecycle, so these runs do not identify a causal sleep
accuracy effect. Within the cold pass, later-question cache hit is only `14%`
for custom and `0%` for both Natural Questions and TriviaQA; the much larger
warm/post rates are dominated by exact-question replay. The reported
`teacher_calls` are logical pipeline calls and exclude malformed-output and
HTTP/429 retry requests, embeddings, local-model calls, judge calls, tokens,
and provider cost.

Latency is also not a matched baseline: Evolve's recorded timer covers the
suppress generation and a second augment generation, while the local baseline
generates once; judge time is excluded from both. These values characterize
the released pipeline but cannot support a clean end-to-end latency
comparison.

The audit also found paper-to-release inconsistencies that must remain visible
rather than be averaged away:

| Quantity | Paper/prose | Released CSV or summary |
|---|---:|---:|
| custom cold logical teacher calls | 313 | 312 |
| custom baseline latency | 2.0 s | .897 s |
| TriviaQA baseline latency | .9 s | 1.427 s |
| custom chunk-cold suppress/augment score | 79.4/78.2 | 82.0/81.8 |
| custom chunk-cold cache hit | 19.2% | 18.8% |
| MMLU warm logical teacher calls/query | 1.50 | 1.536 |

These do not by themselves invalidate the system, but they lower exact
reproducibility and prevent silently treating the manuscript table and public
artifacts as one version.

### Reproducibility, capacity, and safety

The released evaluator CSVs reproduce the paper's reported answer rates, but
the acquired section contents, vector/SQLite snapshots, consolidation
inputs/outputs, compile logs, and rerun seeds are absent. Store counts and
teacher compilation therefore require a new cloud-model run rather than
independent recomputation. No test source is present in the repository.

Capacity remains open. Retrieval caps the live prompt candidate pool at 15,
but a query's teacher pool is not equivalently capped; staging clears after a
successful cycle, while canonical memory grows with novel topics. Reported
31–34% compaction is one cycle over an exactly repeated question set, not a
lifetime equilibrium. Expired canonical sections persist if never retrieved,
because expiration cleanup applies to staging and canonical refresh is
hit-triggered. The claim that hit rate monotonically increases cannot hold
generally under TTL, distribution shift, replacement, or a finite budget.

The paper describes operations as atomic, but the implementation separately
updates the vector store and SQLite metadata. There is no cross-store
transaction, write-ahead journal, rebuild protocol, versioned manifest, or
global query lock; a crash or concurrent read can expose divergent stores or a
partial cycle. The vector root is persisted to disk only at normal application
shutdown, while SQLite changes are immediate, adding a crash-durability gap.
A sleep run also aborts on its first failed staging item after earlier item
mutations may already have committed. The section schema has no source URL,
evidence span, parent lineage, writer/model/prompt digest, tenant, or
authorization scope. Although
the custom evaluation rows contain a source URL and excerpt, the acquisition
pipeline passes only the question and category to the teacher. “Traceable to a
section ID” must not be upgraded into evidence provenance.

Additional source-level hazards are measurable benchmark targets, not merely
production hardening:

- overlap search takes a combined staging+canonical top-100 and filters to
  canonical afterward, so staging crowding can cause a false no-overlap move;
- staging iteration has no declared order, making same-cycle “bounce”
  consolidation order-dependent;
- multi-category expired hits are refreshed together through the first
  category's teacher;
- empty/malformed refresh or compile output can delete an incumbent or be
  interpreted as complete redundancy;
- persistent retrieved text enters the local generation prompt without a
  tenant boundary or prompt-injection defense.

**Strict result:** PASS for external consolidation. Evidence grade remains
preprint/single-run, and the released evaluation supports repeated-query
amortization rather than future-query semantic generalization.

## 11. Mela

[*Mela: Test-Time Memory Consolidation based on Transformation
Hypothesis*](https://arxiv.org/abs/2605.10537), first public 2026-05-11,
integrates a Hierarchical Memory Module into a decoder. The audited
[repository](https://github.com/Musubi-ai/Mela) was pinned at
`fa74b2b21c34915df016977ff6546b0a7f0bc545`.

### Mechanism and persistence boundary

Mela has high- and low-frequency neural-memory modules. Each applies functional
associative updates driven by surprise, with input-dependent momentum and
decay; the decoder reconstructs its memory feature from fine-grained and
gist-like states. MemStack distributes those features across early decoder
layers without adding memory tokens. The paper trains 400M, 800M, and 1.2B
models on 5B FineWeb-Edu tokens with 4K training context and evaluates language
model perplexity beyond that length.

The name “test-time memory consolidation” does not establish a deployed sleep
phase. In the released model path:

- the functional memory state is initialized from learned base tensors inside
  the current call;
- `MelaHMM.forward` accepts a `store_state` input but does not return a state;
- the model wrapper neither supplies a prior functional-memory state nor
  persists a new one;
- generation returns decoder KV cache, not the hierarchical neural-memory
  state.

Consequently, this implementation does not carry an HMM memory across model
calls or sessions. Its two frequencies are inner-sequence update rates, not
wake and sleep schedulers. There is no immutable wake snapshot, delayed
candidate, next-wake publication, deletion lineage, or rollback.

The paper's positive evidence is lower perplexity and long-context robustness
against trained transformer baselines. It is not evidence of cross-session
retention, continual factual acquisition, or lifetime capacity. The repository
contains no tests or released result checkpoint; reproducing the principal
claim requires training a model on the stated multi-billion-token corpus. The
audited repository carries Apache-2.0.

**Strict result:** FAIL. Mela is an important `Q`-phase parametric-state
architecture and wake/test-time comparator, not a sleep-time system precedent.

## 12. MIRROR

[*MIRROR: Complementary Encoding and Reconstructive Consolidation for
Persistent State in LLM Systems*](https://openreview.net/forum?id=IviO4bIZc7)
appeared at the ICLR 2026 MemAgents workshop. The audited
[repository](https://github.com/arcarae/MIRROR) was pinned at
`78c6383a3d538250990a1edf1a3dbf7a1aef2ada`.

### Actual asynchronous lifecycle

MIRROR separates a latency-sensitive Talker from an asynchronous Thinker. The
Thinker maintains Goals, Reasoning, and Memory threads; a Cognitive Controller
then fully regenerates a bounded first-person narrative, capped at roughly
3,000 tokens, from the prior narrative and the completed threads. A later
Talker reads that persistent narrative. Conversation and monologue histories
are separately capped at 20,000 and 10,000 tokens.

This is a direct external-state sleep analogue:

```text
turn t response
→ asynchronous thread update and reconstructive synthesis
→ bounded persistent narrative
→ turn t+1 read
```

The paper reports an average `84%` CuRaTe score versus `69%` for the compared
baseline across seven model architectures, described as a 21% relative
improvement. Controller-only ablations improve all seven models by 5–20
percentage points; the full system exceeds its best component in six of seven
models by 1–8 points, with a negative synergy exception on Claude. A
single-model comparison reports `+9.3` points for MIRROR versus `+2.4` for
extended reasoning. The latency appendix uses 80 conversations and 400 turns
with GPT-4o; it reports median 2.16 seconds and mean 2.52 seconds, with
background work active at response time on 0.8% of turns under simulated human
pauses.

### What bounded reconstruction does not prove

The `O(1)` statement is a bounded-state and bounded-per-turn claim, not
constant lifetime work: the narrative is regenerated after every turn, so
total consolidation work grows at least linearly with turns. More importantly,
the public concurrency paths expose a testable publication problem:

- the standard path may wait up to 100 seconds for the prior Thinker at the
  next user turn, so rapid turns can move deferred work back onto a later wake
  critical path;
- `ProductionMirror.max_background_threads` is declared but not enforced;
- concurrent workers share an unlocked Controller insight block;
- completion order is not tied to turn-generation IDs, so a stale older
  completion can overwrite a newer synthesis;
- publication has no compare-and-swap generation check, journal, or rollback.

CuRaTe is one synthetic benchmark scored by an LLM judge. It does not sweep
lifetime horizon, correction, deletion, adversarial state, or a fixed total
resource budget, and reconstructive errors can compound.
The audited repository carries CC BY 4.0; redistribution must preserve its
attribution terms.

**Strict result:** PASS for asynchronous external consolidation, qualified by
next-wake blocking and stale-completion hazards. MIRROR is the strongest
audited example of reconstructive bounded state, but not yet a transactionally
ordered sleep-memory service.

## 13. HeLa-Mem

[*HeLa-Mem: Hebbian Learning and Associative Memory for LLM
Agents*](https://aclanthology.org/2026.acl-long.625/) is an ACL 2026 main
paper. The audited
[repository](https://github.com/ReinerBRO/HeLa-Mem) was pinned at
`19adb2d709aa74db759835dbc78bc6fa4f4349ef`.

### Paper mechanism

The proposed external architecture contains an episodic graph, Hebbian
coactivation, spreading activation, and a reflective agent that distils dense
episodic hubs into semantic memory. The paper reports LoCoMo GPT-4o-mini
average F1 `34.74`, versus `29.87` without reflective consolidation, `32.19`
without spreading, and `34.28` without forgetting. It states
\(\eta=.02\), \(\lambda=.995\), \(\beta=.1\), a 60-day temporal scale, and
episodic/semantic retrieval depths of ten/five.

That is an important episodic-to-semantic design prior. It is not yet a
verified sleep lifecycle:

- the paper does not establish that hub reflection is scheduled outside a
  response critical path;
- the equation uses \((1-\lambda)w+\eta I\) while the code multiplies by
  `.995`; the written equation with \(\lambda=.995\) is incompatible with the
  reported accumulated-weight scale;
- released `global_decay()` and `adaptive_forgetting()` are not called in the
  evaluated path, forgetting is disabled by default, and `access_count` is not
  incremented;
- the repository states that LoCoMo code will be released later and currently
  exposes LongMemEval, so the principal ACL ablation cannot be independently
  rerun from the release;
- the released LongMemEval command does not enable consolidation by default.

The release still invokes an LLM knowledge extractor every ten turns and
populates semantic memory independently of hub-triggered reflection. Each
LongMemEval item constructs one graph for one question; retrieval reinforces
edges only after selecting the answer context, so no later query in that item
can benefit causally from the update. Fixed temporal edges and embedding
similarity can therefore explain the released spreading path without learned
Hebbian accumulation. The fact graph is unbounded; `max_capacity=100` caps
only an assistant-knowledge list. Main results have no reported multi-seed
interval. The audited repository root contains no license file or license
declaration, so public accessibility is not treated as redistribution
permission; a release may cite the canonical repository and short audited
anchors but may not bundle its source without separate rights evidence.

**Strict result:** PARTIAL for the paper architecture and FAIL for a verified
released reflective-sleep path. Treat the reported reflective and forgetting
effects as paper-level evidence awaiting an executable causal reproduction,
not as demonstrated production mechanisms.

## Cross-system findings

### A more useful design space

“Parametric versus external” is necessary but too coarse. At minimum compare:

- **destination:** shared weight, user adapter, prompt block, text file/skill,
  vector, temporal graph;
- **timing:** foreground, post-turn async, idle debounce, periodic batch,
  capacity/compaction event, or manual;
- **transformation depth:** extraction, deduplication, contradiction
  resolution, abstraction, proceduralization, tier movement, deletion;
- **governance:** source lineage, version rollback, freshness, isolation,
  capacity budget, and causal erase.

### Strongest currently public patterns

- Letta Code: Git-backed reflection, isolated conflict handling, rollback, and
  explicit tier movement.
- Zep: non-lossy raw episodes plus bi-temporal semantic graph and strong
  source-edge lineage.
- Mem0 Platform: unambiguous asynchronous ingestion, but current V3 is
  ADD-only.
- LangMem: composable scheduler and semantic manager, with durability and
  operations delegated to the application.
- Letta API: the most direct foreground-agent/sleeper role separation.
- Evolve: the clearest released cold--warm--sleep--post lifecycle with a
  concrete nightly scheduler, but only repeated-query evidence and no
  cross-store publication transaction.
- Memento-Skills DreamDaemon: a real post-paper daemon path whose prompt and
  file implementation is much stronger than its absent empirical evidence.
- MIRROR: the clearest bounded reconstructive inter-turn state, with a direct
  asynchronous fast/slow split but no generation-ordered publication.
- HeLa-Mem: the clearest recent episodic-graph-to-semantic proposal, but the
  published ablation and released default code path are not yet the same
  executable mechanism.

### Unsolved capacity problem

The deployed patterns remain combinations of:

- unbounded external storage plus top-k retrieval;
- search-time down-ranking of stale items;
- bounded prompt-block compaction;
- temporal invalidation of graph facts;
- manual archive/delete or TTL.

None establishes a joint law for utility, durable storage, retrieval
competition, consolidation compute, freshness, and governance workload.
Evolve further shows why a small prompt is not a capacity proof: staging can
clear and one cycle can shrink while unique canonical topics, expired
unqueried state, provenance debt, and cross-store repair work continue to
grow. Memento-Skills shows the analogous procedural version: retrieval output
can be bounded while catalog selection and lifecycle management scale with the
complete skill population. MIRROR adds a third distinction: constant-size
published state does not imply constant lifetime compute, because a full
bounded reconstruction can still run after every turn.

## Research consequences

The next study should ask:

1. When should one event remain a raw episode, become a semantic fact or
   procedure, enter prompt memory, compile to latent/KV, or enter parameters?
2. How do future wake loss and retrieval cost scale with sleep compute?
3. When does ADD-only accumulation dominate contradiction-aware rewrite, and
   when does it collapse under competition?
4. What Pareto frontier separates provenance-preserving consolidation from
   aggressive lossy compression?
5. How should staleness, write conflicts, failed sleep jobs, and publication
   delay enter end-to-end utility?
6. Which evidence, canary, deletion, and rollback contract must precede any
   parametric promotion?

The strongest nonclaim is equally important: system terminology is not
evidence. “Background,” “async,” “dream,” and “sleep” must be verified at the
response path, scheduler, persistent state, and later-read boundaries.
