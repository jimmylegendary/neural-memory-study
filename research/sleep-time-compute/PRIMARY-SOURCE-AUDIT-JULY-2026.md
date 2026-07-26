# July 2026 Primary-Source Audit

- Audit date: 2026-07-25
- Scope: four July papers that materially change the sleep-time-compute thesis
- Evidence boundary: current arXiv versions and their primary PDFs only
- Venue boundary: all four are preprints; no peer-reviewed acceptance was
  established by this audit
- Status: pre-ingestion audit. Canonical `SRC-STC-*`, `EV-STC-*`, and
  `CL-STC-*` identifiers are assigned only after the evidence registry is live.

## 1. Episodic-to-Semantic Consolidation Without Identity Drift

**Source.** Xue Qin, Simin Luan, Cong Yang, and Zhijun Li,
[arXiv:2607.01988](https://arxiv.org/abs/2607.01988), submitted 2026-07-02.
The PDF carries an *Information Sciences* running header, but the public arXiv
record does not establish acceptance.

### Actual operator and cadence

The agent state is split into identity-bearing state, an append-only episodic
event log, a separate SQLite semantic-fact table, and a policy. A deterministic
function groups new events by stable key, aggregates them, and upserts a
semantic row. It runs at startup and after an episodic write when an unspecified
threshold is crossed. A checkpoint limits the next scan, and each run emits a
new `consolidation_run` event.

The supported fact types are fixed:

- skill success rate;
- object property;
- zone risk;
- interaction pattern.

Each row stores a value, observation count, success rate, confidence, latest
supporting event, and rule version. This is a narrow, deterministic
episodic-to-semantic compiler—not a free-form LLM abstraction system.

### What “without identity drift” means

The semantic table is intentionally excluded from the SHA-256 identity
manifest. The resulting lemma says that consolidation does not change the
manifest hash because the updated state is not a hash input. It does **not**
show that behavior, knowledge, or long-run policy semantics cannot drift. The
paper itself leaves behavioral drift possible.

### Reported empirical scope

| Test | Scale | Reported result | Boundary |
|---|---:|---|---|
| V1 aggregation | 1,000 grasp events, 800 success/200 failure | exactly one semantic row; 6/6 field checks; idempotent and shuffle-invariant | deterministic fixture |
| V2 identity | 100 events | consolidation leaves the manifest byte-identical; changing one of seven manifest inputs changes the hash | structural test, not behavioral invariance |
| V3 planning | two objects; 1,000 decisions/control/seed, 10 seeds | calibrated memory reduces unproductive actions by 79.82%, 95% BCa CI [78.02%, 81.49%] | synthetic rule-based planner |
| Uniform-confidence ablation | same V3 fixture | −2.6%, CI [−4.7%, +0.4%] | row presence alone is insufficient |
| Throughput fixture | 100,000 input rows, one CPU | median 309 ms, p90 324.5 ms; output remains two stable keys | not a high-cardinality capacity test |

### Capacity, governance, and nonclaims

- Active output grows with the number of distinct fact keys; there is no
  fact-key cap or pruning policy.
- Upsert keeps one current semantic row per key, while raw evidence remains in
  the episodic log.
- There is no time decay. Equally supported old and recent evidence receive the
  same treatment.
- A phrase about capacity-driven episodic pruning is not accompanied by an
  algorithm or an accuracy analysis.
- Rebuilding the same snapshot with the same rule version is deterministic and
  crash replay is idempotent. This is not user-facing selective rollback.
- There is no right-to-be-forgotten API, causal erase, poisoning defense, or
  cross-identity transfer.
- Provenance is relatively strong: rule version, processed event interval,
  latest supporting event, and consulted fact IDs are retained.
- The evaluation uses synthetic fixtures and a rule-based planner, not a
  production fleet or a free-form LLM agent.

**Phase classification:** `B/S primitive`. It is a credible external
consolidation building block, but not a complete bounded or governed sleep
system.

## 2. MemDefrag: Latent Memory Defragmentation for Large Language Models

**Source.** Ruiyi Yan, Zhuoyuan Mao, and Yiwen Guo,
[arXiv:2607.05969](https://arxiv.org/abs/2607.05969), v1 2026-07-07; audited
current v2 2026-07-15.

### Representation and two triggers

Base weights stay frozen. Each knowledge fragment is prefilled through the
model and its per-layer hidden states are concatenated into a persistent latent
prefix. Recording a fragment uses no backpropagation.

Two operations must not be conflated:

1. **Every query:** a tracer middle layer ranks fragments by prompt-to-fragment
   attention density, relevant fragments are moved closer to the prompt, and
   only Top-K fragments form the temporary inference memory.
2. **Every write that would overflow:** when latent length would exceed
   `Nmax=12,800`, existing fragments receive length-proportional deletion
   quotas. Positions with the lowest token self-information are removed at the
   same locations across all layers.

The first operation is reversible query-time selection over the retained
prefix. The second permanently deletes latent positions unless an external
checkpoint exists.

### Reported empirical scope

- Main comparisons use Llama-3.1-8B-Instruct, MemoryLLM-8B, and M+-8B;
  additional tests use Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3, and
  Gemma-2-9B-it.
- NaturalQuestions and SQuAD use 500 groups each, fragments of at most 512
  tokens, and 50 writes. Six LongBench datasets reach 16,384 tokens.
- Hardware is four accelerators with 141 GB HBM3e each.
- On Llama layer 13, the target fragment has mean rank 1.66, Top-1 85.6%, and
  Top-3 95.7% after moving the target across all 20 positions in 500 groups.
- Vanilla latent-memory NaturalQuestions accuracy falls from 73.2% to below
  9.4% over 20 writes; SQuAD falls from 77.6% to 19.0%.
- At write 50, NaturalQuestions is 43.0% for MemDefrag Top-1 versus 17.4% for
  MemoryLLM and 17.6% for M+.
- At write 50, the best reported SQuAD variant is MemDefrag Top-2 at 27.2%
  versus 15.4% for MemoryLLM and 20.2% for M+.
- NaturalQuestions Top-2 at write 50 is 37.8% with informativeness-based
  deletion versus 20.8% with random deletion, a reported relative gain of
  81.73%.
- MemDefrag is best in 20 of 24 reported LongBench dataset-metric cells.
- NaturalQuestions runtime per group is 76.49 seconds for Top-2, 64.19 seconds
  for MemoryLLM, and 87.24 seconds for M+. It is not faster than MemoryLLM.

### Capacity, governance, and nonclaims

- The 512-token-fragment sequence first triggers deletion at write 26.
- The paper demonstrates a finite latent-token budget, but not a semantic-item
  lifecycle.
- Static K is selected empirically for each model/dataset. Gemma exhibits a
  long-run drift case in which Top-1 eventually fails.
- SQuAD still falls to 27.2%; the method slows degradation rather than solving
  continual forgetting.
- There is no conflict reconciliation, fact update semantics, TTL, provenance,
  privacy policy, causal delete, rollback, or migration to an external
  canonical store.

**Phase classification:** `W/Q` with a persistent bounded latent destination.
It is strong evidence for capacity-aware latent pruning, not deployed
cross-session sleep.

## 3. What to Keep, What to Forget

**Source.** Ashwin Gerard Colaco and Nada Lahjouji,
[What to Keep, What to Forget: A Rate–Distortion View of Memory Compaction in
LLMs and Agents](https://arxiv.org/abs/2607.08032), submitted 2026-07-09.

### Formal contribution

For compactor \(C_\theta:H\rightarrow Z\), usage operator \(U\), query \(Q\),
answer \(Y\), and budget \(B\), the paper writes:

\[
\min_\theta\;
\mathbb{E}_{H,Q,Y}
\left[\ell\!\left(U(C_\theta(H),Q),Y\right)\right]
\quad\text{subject to}\quad
\operatorname{rate}(Z)\le B.
\]

Its information-bottleneck form maximizes \(I(Z;Y\mid Q)\) subject to
\(I(Z;H)\le B\). For a query-agnostic compactor and finite answer space, the
DPI/Fano argument gives:

\[
P_e \ge
\frac{H(Y\mid Q)-B-1}{\log|\mathcal{Y}|}
\quad\text{when}\quad B<I(Y;H\mid Q).
\]

The discussion of an effective query-agnostic budget of roughly \(B-H(Q)\) is
an interpretation, not a second theorem with the same formal status.

The taxonomy spans granularity; lifecycle; lossiness, fidelity, and
reversibility; query/task adaptivity and importance surrogate; learnability;
mechanism; and storage substrate. Its three emphasized properties are
reversibility, query conditioning, and fidelity profile.

### What was actually evaluated

The proposed COMPACT-Bench normalizes systems using bytes per token of original
history and calls for complete accuracy-budget frontiers, loss attribution,
late-query reversibility, confidence calibration, latency/memory/dollar cost,
and repeated-compaction error curves. It was **not** implemented as a complete
benchmark.

The first reference experiment uses Qwen2.5-1.5B-Instruct on an RTX 4060 8 GB,
WikiText filler, 2K–8K contexts, five budgets, and three trials—1,395
generations total. It compares only KV eviction/scoring methods. Although the
normalization discussion covers quantization and prompt compression, no
cross-layer quantizer, prompt compressor, architectural state, and agent store
are jointly evaluated.

The second reference experiment stores twelve key-value facts and applies
5/10/15/20/25 compaction events. Archive-plus-retrieval remains near 0.95
recall; irreversible repeated summarization lies around 0.33–0.56, leaving an
approximately 0.5 gap at high cadence. This is a small operator simulation, not
a deployed agent-memory system.

### Exact novelty collision and remaining gap

This source precedes and directly overlaps:

- a cross-layer rate-distortion framing;
- query-aware distortion;
- reversible episodic versus lossy semantic tiers;
- explicit promotion/demotion as a design principle;
- asynchronous or sleep-time consolidation;
- a shared budget axis for heterogeneous compression operators.

The current project therefore claims none of those in isolation. The source
leaves the following unimplemented:

- an executable router and state machine across KV, working context, episodic
  external state, semantic state, latent state, adapters, and weights;
- item-level versions, lifetime, TTL, provenance, and access policy preserved
  through moves;
- learned promotion/demotion triggers and stop rules;
- cross-tier consistency and atomic publication;
- deletion fan-out, revocation, and rollback;
- migration bandwidth and data-motion-aware scheduling;
- recoverable routing into parametric destinations.

The paper itself names auditable, versionable compressed memory with forgetting
and revocation primitives as an open agenda. The defensible novelty boundary is
therefore a **governed versioned lifecycle router and its matched
cross-substrate benchmark/infra**, not rate-distortion theory.

**Phase classification:** `P`. This is a survey, formal proposal, and small
reference experiment rather than an implemented lifetime controller.

## 4. Can a Language Model Learn Facts Continually in Its Weights?

**Source.** Charles O’Neill,
[arXiv:2607.11020](https://arxiv.org/abs/2607.11020), v1 2026-07-13; audited
v2 2026-07-14. Single-author Baseten preprint.

### Task and training methods

The main model is Qwen3-4B with a rank-16 LoRA on all seven projection
matrices, merged after each fact. Some conditions use full fine-tuning; an 8B
model is used only for the entailment-gap replication.

The main corpus contains 247 invented one-sentence facts, with 236–247 facts
in individual contrasts. Each is tested through recall, paraphrase,
application, composition, and counterfactual questions. A fact-in-prompt
ceiling and original-model floor are measured for every fact.

Two write corpora are compared:

- **bare:** the statement in two trivial framings;
- **study:** 24 generated items per fact, including paraphrases, question-answer
  forms, a worked implication, and contrast with prior knowledge.

Training includes SFT, offline context distillation, and online forward- or
reverse-KL distillation. The primary sequential study uses 20 writes and three
seeds. The 100-write extension is a reduced factorial and its endpoint is
descriptive rather than a fully crossed confirmatory result.

### Main results

- The bare-statement entailment gap is 27.4 points. Diverse recall/paraphrase
  prompts reduce it to 5.4 points without directly presenting the derived
  conclusion.
- After 20 writes at 192 steps per fact, bare retention is 1%; study retention
  is 46%, a paired difference of 45.6 points with CI [38.8, 52.6].
- At 100 writes, study retention plateaus around 25–28%.
- In the 100-write comparison, SFT bare reaches 8.6–11.7% versus study
  33.0–37.7%; offline distillation bare reaches 25.0–29.4% versus study
  38.5–43.7%.
- A behaviorally forgotten fact retains 69%/79% of its statement
  log-probability lift, or 57%/67% after drift correction.
- Seventy percent of wrong answers to forgotten bare facts copy the newest
  fact. This supports address capture or access failure more than complete
  content erasure.
- Joint use succeeds 32% when both facts are in weights versus 91% when both are
  supplied in context.
- Re-presenting a forgotten study fact in the prompt restores 77–80%.

### Intervention audit

| Intervention | Result | Permitted conclusion |
|---|---|---|
| frozen original teacher | capability +2 points, KL 0.48, retention 54% after 20 writes | strongest tested sequential-distillation condition still loses nearly half |
| accumulated self-merged teacher | capability −31 points, KL 1.70, retention 21% (34% excluding capped loops) | recursive teacher drift is severe |
| KL penalty \(\lambda=0\) | capability −66, retention 1% | unconstrained bare writes destroy capability |
| KL penalty \(\lambda=.5\) | capability −19, retention 25% | capability/retention trade-off improves |
| KL penalty \(\lambda=1\) | capability −5, retention 36% | capability protection does not prove reachability |
| sleep-like full reconsolidation every 20 writes | at write 100, retention 25% versus 28% without it; capability protected by 12 points | periodic batch distillation is a capability safeguard, not a retention fix |
| bridge examples | no preregistered-success recipe | local address repair was insufficient |
| activation projection/patching/block swaps | maximum rescue 5.9 points, below the 10-point threshold | no strong local causal rescue |
| conflict-gradient projected Adam | +1.4 points, CI [−2.1, 5.0] | no advantage over norm-matched random direction |
| GRPO answer match | sparse/absent target generation signal | reward optimization did not reliably install the fact |

Linearized Adam predicts the next update with \(\rho=.795\), but not forgetting
after fifteen updates (\(\rho=-.258\)). A local gradient diagnostic is therefore
not a demonstrated lifetime controller.

### Capacity, governance, and nonclaims

- The paper measures interference over 20–100 writes; it does not implement a
  hard capacity or eviction budget.
- “Forgotten” means behavioral access failure, not targeted weight erasure.
- There is no selective delete, right-to-be-forgotten workflow, or production
  rollback/versioning mechanism.
- Saved adapters and checkpoints support experimental reconstruction, not
  governed item-level rollback.
- The study is one model family and one style of invented discrete fact. The 8B
  check covers only the entailment-gap result.
- Skills, real-world knowledge, and pretraining-scale consolidation remain
  outside the evidence.

The paper's strongest systems implication is conditional: weight writing may be
useful as a cache when a canonical copy exists elsewhere. If long survival,
composition, audit, or deletion matters, weights should not be the system of
record.

**Phase classification:** `S experiment`. It directly tests periodic
sleep-like parametric consolidation and produces a negative result.

## 5. Cross-paper consequences

These sources jointly narrow the research program:

1. **Canonical truth stays reversible.** External raw evidence is the system of
   record; semantic, latent, and parametric forms are versioned derived views.
2. **Parametric memory is a cache, not an archive.** Promotion into weights
   needs expected repeated reuse, addressability tests, capability preservation,
   low volatility, and a deletion-risk check.
3. **Capacity is multi-dimensional.** Fact-key cardinality, latent-token budget,
   retrieval competition, parametric interference, sleep service rate, and
   governance capacity need separate counters.
4. **Compaction requires a future-use distortion.** Transcript reconstruction
   is secondary, but serialized bytes are only an operational budget proxy and
   must not be presented as mutual information.
5. **Reversibility is operational.** A method is not reversible merely because
   an experimenter can rerun training. The system needs item lineage, an
   immutable prior version, atomic pointer swap, tombstone fan-out, and tested
   restoration.
6. **The central experiment is matched routing.** The same frozen events and
   future queries should be routed among raw external, semantic external,
   latent, user-parametric, shared-parametric, and hybrid destinations under
   matched sleep FLOPs, durable bytes, live tokens, latency, staleness, and
   deletion obligations.
