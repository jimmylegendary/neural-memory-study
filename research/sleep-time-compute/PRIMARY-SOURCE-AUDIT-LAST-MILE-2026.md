# 2026 Last-Mile Primary-Source Audit

- Audit date: 2026-07-25
- Scope: eight late-discovered works that materially sharpen the
  destination, cadence, capacity, evaluation, and infrastructure claims
- Evidence boundary: official arXiv/OpenReview/ACL records and the exact
  primary PDFs listed below
- Selection rule: a work is included only if it changes a planned comparison,
  metric, mechanism, or claim boundary; adjacent memory papers are not added
  merely to enlarge the bibliography
- Status: pre-ingestion audit. Canonical source, evidence, and claim IDs are
  assigned only after the control-plane registries are live.

This audit applies the same operational test as the rest of the program.
Strict sleep-time compute must be outside the current wake action's causal
critical path, transform accumulated lifetime experience, and publish durable
state reused by a later wake phase. A benchmark, an offline-trained component,
or a bounded live buffer is not automatically a sleep system.

## 1. Audited snapshots

The digest identifies the reviewed bytes; it does not assert redistribution
rights. ImprintBench's official OpenReview PDF endpoint was protected by a
Cloudflare challenge at the audit cutoff, so its two workshop records and
search-indexed workshop-PDF metadata were inspected but no local PDF digest is
claimed.

| Work | Audited primary snapshot | SHA-256 |
|---|---|---|
| ImprintBench, *Measuring the Limits of Continual Learning for LLMs* | official [OpenReview record `QIJgTW3Qd2`](https://openreview.net/forum?id=QIJgTW3Qd2), author-profile venue records, and indexed PDF metadata identifying the 2nd Workshop on Compositional Learning at ICML 2026 | unavailable at cutoff |
| *Understanding LoRA as Knowledge Memory: An Empirical Analysis* | [arXiv `2603.01097`](https://arxiv.org/abs/2603.01097) v4 PDF, 2026-07-14 | `03a18846a0e35a68437772ee9e142de9882f286b215e2dab3fcde9c4a4a575ab` |
| MemoryBench | [arXiv `2510.17281`](https://arxiv.org/abs/2510.17281) v7 PDF, 2026-06-03 | `00b2aee7c582f53dadf1c3b5bf6cd7fbf3da242d1a0328edb1338b9aeb318bf8` |
| *Are We Ready For An Agent-Native Memory System?* | [arXiv `2606.24775`](https://arxiv.org/abs/2606.24775) v1 PDF, 2026-06-23 | `6cad9bfcded2f1801c09049daa716a33e570cc407ddfbe40f32d3aa850361d85` |
| TiMem | [ACL Findings 2026 paper 1091](https://aclanthology.org/2026.findings-acl.1091/) PDF | `9a3cbefe0eaa9fa5acc40695402ad9e4ed7739b91049e8f5c6e9ba8d1182a800` |
| MEMORA, *Embodied Action Memory from Egocentric Videos for Reasoning and Planning* | [arXiv `2607.14252`](https://arxiv.org/abs/2607.14252) v1 PDF, 2026-07-15 | `b0a030d1608adecc3a45a863b014fd812d029466ce09fbad1b47c801bcb88d6f` |
| RecMem | [ACL Findings 2026 paper 1619](https://aclanthology.org/2026.findings-acl.1619/) PDF | `0e9ce858e3812378f18299899da53df2b16d18674e16862d7d1f7528cfd3339b` |
| GAM | [ACL 2026 long paper 1600](https://aclanthology.org/2026.acl-long.1600/) PDF | `7f39920a8a0d1aa71601a02d96ee7b37b2b04443f8bfadeb091437729ca81592` |

Venue identities require care:

- ImprintBench has separate CTB@ICML 2026 and CompLearn 2026 OpenReview
  records. The indexed PDF identifies acceptance to the 2nd Workshop on
  Compositional Learning at ICML 2026. No official ICML main-conference or
  PMLR proceedings identity was found, so the workshop records must not be
  promoted to one.
- The current LoRA and MemoryBench PDFs also identify ICML 2026, PMLR 306.
  Earlier anonymous or ICLR records establish chronology, not the final venue.
- TiMem and RecMem are ACL Findings 2026 papers; GAM is an ACL 2026 main paper.
- MEMORA reports an RSS 2026 FM4RoboPlan workshop oral. Its benchmark and code
  were future-release statements at the cutoff.

## 2. Chronology and phase classification

| Earliest public trace | Work | Durable destination | Operational classification |
|---:|---|---|---|
| 2025-09-05 ICLR submission; 2025-10-20 canonical arXiv | MemoryBench | whichever external memory system is under test | `P`: benchmark of wake feedback learning, not a sleep operator |
| 2025-09-20 anonymous record; 2026-03-01 arXiv | LoRA as Knowledge Memory | one or more LoRA modules | `P`: parametric destination study; no recurring deployed lifecycle |
| 2026-01-06 | TiMem | five-level temporal text hierarchy | `B/S partial`: higher levels are built on temporal closure and reused later |
| 2026-04-14 | GAM | topic graph plus archived event graph | `B partial`: consolidation fires at pause, session end, or overflow |
| 2026-05-15 | RecMem | subconscious raw store plus episodic and semantic memories | `W/B partial`: recurrence-gated durable consolidation, but not an autonomous idle service |
| 2026-05-25/27 official records | ImprintBench | context, auxiliary parameters, or model parameters under test | `P`: benchmark, strict sleep test not applicable |
| 2026-06-23 | Agent-Native Memory study | twelve heterogeneous external systems | `P`: characterization and systems benchmark |
| 2026-07-15 | MEMORA | entity, activity, environment, inferred-knowledge, habit, workflow, and preference stores | `S experimental PASS`: offline participant/video consolidation is reused in later evaluation |

The table deliberately separates operational phase from usefulness. A `P`
study can be more important to the central thesis than a low-evidence system
that happens to call its boundary operation “sleep.”

## 3. ImprintBench: destination comparisons need an internalization test

### What is evaluated

ImprintBench contains roughly 4,300 questions built from news, API changelogs,
and personalization streams. It separates six capabilities:

1. direct acquisition;
2. temporal update;
3. reference resolution;
4. composition;
5. implicit relevance;
6. boundary awareness.

It compares four broad method families: context management, native
long-context use, auxiliary parameters, and model-parameter updates. This
matters because direct factual recall can look successful while composition,
temporal replacement, or recognizing that a question is outside the learned
boundary still fails.

### Reported result and permitted interpretation

- At aggressive compression, context-management methods lose roughly 20–40
  points on the reported plots.
- Parametric methods are flatter as the corpus grows, but underperform
  in-context learning while the full corpus still fits.
- SFT and self-distillation have similar average results, but different error
  geometry: SFT shifts toward new facts while distillation better preserves
  older behavior.

These numerical/plot readings remain intake-local and cannot become
release-support evidence until the exact venue-specific PDF bytes are vendored
and anchored.

The defensible conclusion is not “weights win at scale.” It is that destination
choice must be evaluated across acquisition, update, composition, relevance,
and boundary detection under a matched resource envelope. The crossover point
remains model-, workload-, and operator-dependent.

### Lifecycle boundary

ImprintBench does not impose a lifetime hard cap, causal deletion, selective
rollback, provenance, or privacy contract. It is therefore a central
evaluation substrate and an external-validity track, not evidence for a
complete sleep lifecycle.

## 4. LoRA as Knowledge Memory: parametric storage has a routing layer

### Training data and operator

The paper evaluates PhoneBook, CounterFact, a 15-paper/450-question PaperQA
collection, NarrativeQA, QuALITY, InfinityBench, and LongBench v2. A document
may supervise the adapter as raw text, question-answer pairs, a summary,
rewrites, or a mixture. Ranks range from 2 to 1,024 and training spans roughly
1K–20K tokens in the controlled studies.

On PaperQA with the Llama judge, raw supervision scores `3.187`; mixed
supervision scores `6.822`. The result makes data transformation part of the
memory operator: the same source document produces materially different
usable memory depending on its sleep corpus.

### Capacity knee and module topology

- Very low ranks, around rank 4 in the reported setting, are highly
  parameter-efficient.
- Performance exhibits a finite rank-dependent knee; rank is a capacity axis,
  not a free compression knob.
- On the 64K-token PhoneBook corpus and under a matched trainable-parameter
  budget, an oracle-routed set of eight rank-4 modules outperforms one rank-32
  module.
- Real routing can lose that advantage and underperform the single module.
- Reported merge scores are TIES `5.41`, linear `4.92`, scaled concatenation
  `4.77`, and vanilla concatenation `0.19`.
- Beyond 100K context on InfinityBench, Multi-LoRA plus ICL scores `34.92`
  against `24.37` for the single-memory alternative in the reported table.

Project synthesis — not a source-authored law — adds a routing-and-serving
heuristic to parametric-memory accounting:

```text
normalized usable-parametric-memory score
  = isolated-module task utility
  × router hit rate
  × composition-retention ratio
  × serving hit rate
```

Each factor must be measured on the same held-out workload and normalized to
`[0, 1]`: isolated-module utility is the adapter's score relative to the
declared task-scale ceiling, composition retention is composed-module utility
divided by isolated-module utility, and serving hit rate is the fraction of
requests for which the selected module is resident before its latency
deadline. The product is a diagnostic score, not stored bits, independent
capacity, or a scaling law.

Adapter count, rank, catalog selection, cache residency, and merge
interference must therefore be measured separately. “Put it in LoRA” does not
remove retrieval; it changes retrieval from a vector/document problem into a
module-selection and composition problem.

### Lifecycle boundary

The experiments do not implement an accumulating daily stream, correction,
causal delete, unlearning, version rollback, or tenant lifecycle. They support
a parametric destination and its capacity/routing diagnostics, not a governed
sleep system.

## 5. MemoryBench: wake feedback must be a first-class baseline

MemoryBench supplies a user-feedback simulator, declarative and procedural
workloads, eleven datasets, a 4:1 split, on- and off-policy evaluation, and
explicit verbal/action as well as implicit feedback.

Its results show that no single memory implementation should be treated as the
default oracle. For Qwen3-8B off-policy evaluation, the reported open-domain
scores are Vanilla `0.6523`, A-Mem `0.6243`, and MemoryOS `0.5894`; on the
legal workload BM25-M scores `0.5011` and Vanilla `0.4637`. Mem0 is absent from
some workloads because its runtime was judged unreasonable by the study.

For this program, MemoryBench has two roles:

- a wake-learning control that prevents every gain from being attributed to
  deferred sleep;
- a feedback-stream generator for testing whether the same evidence should be
  acted on immediately, staged, or consolidated later.

It does not supply a sleep boundary, a total memory cap, deletion, rollback, or
lifetime privacy. Its scores must not be pooled with the confirmatory synthetic
lifetime benchmark without a declared external-validity boundary.

## 6. Agent-native systems: utility, exactness, and latency form a frontier

*Are We Ready For An Agent-Native Memory System?* compares twelve memory
systems and two baselines over five workloads and eleven datasets. Its
taxonomy covers representation/storage, extraction, retrieval/routing, and
maintenance.

### Reported frontier

No system dominates all workloads. Illustrative utility/latency pairs reported
by the paper include:

| System | Reported utility | Reported latency |
|---|---:|---:|
| LightMem | 48.3 | 3.67 s |
| MemTree | 63.5 | 15.9 s |
| MemoryOS | 82.0 | 28.6 s |

LongBench construction or service times also vary by orders of magnitude:
LightMem `17.3 s`, Mem0 `374.2 s`, MemoChat `460.2 s`, MemoryOS `490.0 s`,
and A-MEM `552.1 s` in the reported comparison. High-utility graph-oriented
systems such as Cognee and Zep also occupy expensive regions of the frontier.

The paper's evidence-gap plots show that compression and abstraction can
destroy exact evidence needed later. One plotted LongBench curve falls from
about `42.6` to `19.0`; a LoCoMo embedding-RAG comparison falls from `37.1` to
`7.4`. Exact LongMemEval variants differ across tables, so any downstream
numeric claim must preserve the table, workload, and method identity rather
than quote a detached before/after pair.

### Consequence

Global reorganization is not automatically best. The study argues that
localized maintenance is often more cost-effective. The sleep scheduler must
therefore choose both **whether** to maintain and **how much of the memory
graph** to touch.

At the cutoff, the arXiv abstract linked public code/data through
`OpenDataBox/MemoryData`, while conclusion wording still described a future
release. Reproducibility status must be tied to a pinned repository snapshot,
not inferred from either sentence alone.

## 7. TiMem: temporal hierarchy makes cadence explicit and costly

TiMem creates five external-memory levels:

1. segment;
2. session;
3. day;
4. week;
5. monthly profile.

Level 1 is updated online. Levels 2–5 are produced when their temporal windows
close, with a three-item history window in the reported hierarchy. This is a
concrete multi-timescale external consolidation mechanism, although session
closure can remain ingestion-adjacent rather than an autonomous idle job.

Reported scores are `75.30±0.16` on LoCoMo and `76.88±0.30` on LongMemEval-S.
LoCoMo recalled context falls by `52.20%`. The hierarchy is not free:

- LoCoMo calls increase from `2,871` for level 1 only to `3,717`, about
  `29.5%`;
- LongMemEval-S calls increase from `124,272` to `155,333`, about `25.0%`;
- P50/P95 latency is `2.35/4.91 s` on one evaluation and `1.76/4.48 s` on
  the other.

The hierarchy compresses and routes memory, but does not implement
storage-time forgetting or a total lifetime bound. It isolates users, while
secure storage, consent, and deletion remain future work. The system therefore
supports multi-timescale scheduling and its call-cost term, not a bounded or
fully governed memory claim.

## 8. MEMORA: sleep extends from text to embodied action histories

MEMORA extends EPIC-KITCHENS with about 45 hours of activity from eighteen
participants, divided into ten-second segments. During wake ingestion it emits
`Add`, `Update`, `Delete`, or `Noop` entity edits over environment, entity,
activity, and inferred-knowledge stores; the activity log itself is
append-only. Offline consolidation after a participant or video produces
habits, workflows, and preferences used for later assistance and robot-plan
evaluation.

The paper reports gains up to `20.5` accuracy points and `16.6%` relative
improvement on out-of-distribution robot planning. Its evaluation also uses
important anti-leakage controls:

- time-restricted retrieval;
- withholding consolidation products that occur after the query anchor;
- state-history rollback for the evaluation snapshot.

This is strict experimental external sleep and a needed multimodal extension
to the text-only benchmark. Its scope remains short—at most fifteen sessions
per participant—and it does not establish a lifetime hard cap, participant
consent workflow, causal erasure of physical-tool observations, or released
benchmark/code at the cutoff.

## 9. RecMem: recurrence is an admission rule, not a capacity bound

RecMem retains a verbatim “subconscious” raw store alongside episodic and
semantic text/vector memories. It promotes content only when semantic
similarity and recurrence-count thresholds are crossed.

The design produces major construction savings in the reported accounting:
LoCoMo construction uses `193.2K` tokens versus Mem0 `1,520.8K` and A-Mem
`1,459.93K`, about an 87% reduction. On LongMemEval-S, the reported reduction
relative to Mem0 is `77.5%`. Removing the subconscious store drops one score
from `81.10` to `51.88`.

The raw store is therefore not expendable residue; it is a recovery and
rare-event substrate. But it is also unbounded in the implementation. The
similarity/count threshold can suppress valuable one-off events, and the paper
does not implement deletion, version rollback, or privacy controls.

Recurrence should be treated as one learned or tuned feature in admission:

```text
admit(x)
  = policy(utility, recurrence, surprise, volatility,
           sensitivity, evidence quality, recovery value)
```

It cannot stand in for a lifetime capacity mechanism.

## 10. GAM: a bounded progression buffer can feed an unbounded archive

GAM maintains a fixed 2,048-token event-progression buffer. Session end, pause,
or overflow triggers a semantic-boundary operation that writes a summary/raw
dual node into a topic graph and archives the raw event graph.

On LoCoMo, the reported average F1 is `40.00`, versus `35.38` for Mem0, with
`1,370.18` versus `1,533.94` tokens per query. Longer histories expose a
different boundary: from session 3 to 27, event nodes grow `81→657`, topic
nodes `8→84`, and edges `371→3,307`; latency rises `3.49→4.71`, while one
reported F1 trajectory falls `.65→.48`.

The 2,048-token bound applies only to the live progression buffer. It does not
bound the topic graph, event graph, archive, index, provenance, or maintenance
work. Although graph-local pruning/deletion is described as an affordance, the
paper does not implement a causal deletion or lifetime guarantee.

GAM is therefore a direct counterexample to the inference:

```text
bounded hot state  ⇒  bounded total memory
```

The implication is false unless all downstream durable views and recovery
copies are included.

## 11. Cross-paper consequences for the research program

### 11.1 Destination choice is a two-stage routing problem

External memory requires evidence/item retrieval. Multi-LoRA requires adapter
retrieval and possibly composition. Shared weights remove an explicit catalog
at inference but introduce interference, update, rollback, and deletion costs.
The benchmark must separate:

1. admission and destination routing;
2. retrieval or module routing inside the chosen destination;
3. answer generation given the delivered state.

### 11.2 Internalization is wider than direct recall

The minimum parametric/external comparison now includes direct acquisition,
temporal update, reference resolution, composition, implicit relevance, and
boundary awareness. A method that stores an answer but cannot compose it or
recognize its domain has not fully internalized the memory.

### 11.3 Bounded live state and bounded lifetime state are different claims

GAM's live buffer, RecMem's construction budget, TiMem's recalled context, and
LoRA's per-module rank can all be bounded while raw archives, graphs, catalogs,
adapter count, lineage, or rewrite work continue to grow. Every capacity claim
must report:

```text
live bytes
+ durable active bytes
+ archive/recovery bytes
+ index and graph bytes
+ parametric and adapter bytes
+ optimizer/importance bytes
+ cumulative construction and rewrite work
```

### 11.4 Multiple sleep clocks add service demand

TiMem shows that session/day/week/month clocks increase calls by roughly
25–30% in its reported settings. More clocks can improve abstraction and
freshness while destabilizing a sleep queue. Cadence experiments must measure
both downstream utility and incremental construction calls, latency,
data movement, and queue age.

### 11.5 Local maintenance is a scheduler action

The agent-native study's utility–latency frontier and RecMem's selective
promotion both support subgraph-, tenant-, or item-local jobs. `run sleep` is
too coarse an action. The scheduler needs a scope variable:

```text
action = (when, tenant, region, evidence set, operator, destination, budget)
```

### 11.6 Embodied memory changes governance

MEMORA extends the lifecycle from text supplied by a user to observations of
people, places, tools, and routines. Consent, access, correction, and deletion
must attach to evidence and derived memories across modalities. A text-only
right-to-be-forgotten test is insufficient for an embodied agent.

## 12. New hypotheses and falsifiers

These are project hypotheses, not claims made by the audited papers.

### X-STC-Q — Parametric routing knee

At a fixed total adapter-parameter budget, modular low-rank memories beat a
single adapter only while router error and multi-module composition cost stay
below the interference saved by isolation.

**Falsifier:** freeze route-recall LCB, out-of-catalog FPR UCB, and serving-cost
qualification thresholds on CALIBRATION. If no router qualifies, the deployable
test is `INCONCLUSIVE`. Among all qualified routers, simultaneous
modular-minus-single UCBs at or below the frozen materiality margin in every
reuse/composition cell falsify the modular-region hypothesis; “competent” may
not be redefined after TEST.

### X-STC-R — Clock-count queue crossover

Adding consolidation timescales first improves utility per active byte, then
lowers net lifecycle utility once construction-call arrival exceeds available
sleep service or repeated abstraction distortion accumulates.

**Falsifier:** additional clocks remain Pareto-improving across queue load,
latency, cost, and distortion holdouts.

### X-STC-S — Internalization completeness gap

Direct-recall parity overstates parametric internalization. The largest
parametric/external gap will appear in temporal replacement, boundary
awareness, or cross-memory composition rather than direct acquisition.

**Falsifier:** destination rankings are stable across all six ImprintBench-style
capabilities under matched evidence and compute.

### X-STC-T — Recovery-substrate floor

For volatile or low-frequency evidence, removing raw/reversible state creates a
non-zero correction and rare-event utility floor that lossy semantic memory
cannot cross, even if active memory becomes much smaller.

**Falsifier:** a bounded lossy view matches archive-backed systems on late
queries, correction, one-off events, and rollback without retaining an
equivalent information-bearing state elsewhere.

## 13. Big-three bounded-negative update

A final bounded search of public Google/DeepMind, Meta/FAIR, and Microsoft
research corpora found no additional thesis-changing direct sleep-time work
beyond the company-lineage audit. This means only that no further qualifying
public artifact was found under the documented search scope and cutoff. It
does not assert that unpublished internal work is absent.

## 14. Claim boundary after this audit

The strongest defensible framing is now:

> Sleep-time compute is a governed, asynchronous lifecycle for selecting,
> transforming, routing, validating, publishing, and retiring accumulated
> experience across external, latent, modular-parametric, and shared-parametric
> destinations under finite service, state, and recovery budgets.

Existing work establishes many pieces, including temporal hierarchies,
recurrence-gated promotion, bounded hot buffers, offline multimodal
consolidation, parametric adapter storage, and broad memory-system frontiers.
No audited work jointly solves destination routing, internalization quality,
lifetime capacity, queue stability, provenance, correction, causal deletion,
rollback, and matched cross-substrate evaluation. That joint problem—not the
sleep metaphor itself—is the remaining research target.
