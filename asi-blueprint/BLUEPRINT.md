# ASI blueprint — worker / harness / supervisor / right-brain / neuro-symbolic memory

Integrating (a) the Behrouz neural-memory line (Titans→Miras→Atlas→TNT→Nested Learning→Sleep, + the giants:
Hopfield, Fast-Weight, ICL-as-GD, Mamba/GLA/RetNet/TTT), (b) TaskOps (verification-first work-truth harness),
and (c) the user's cognitive architecture, toward a concrete, buildable ASI direction.

Status: BLUEPRINT + TaskOps reposition = authored (this file). Fit-paper research, concept-KG, whitespace, and the
two product idea-notes = filled by the `asi-blueprint-research` workflow (background) + author synthesis.

---

## 0. The one-paragraph thesis

Every sequence model is already a memory: an associative memory defined by five axes — **architecture ×
attentional-bias(inner objective) × retention(forget) × inner-optimizer × update-frequency** (Miras + Nested
Learning). The neural-memory line's endgame is "**state = weights at every timescale**": the KV cache dissolves into
a spectrum of parameter blocks indexed by how often they update (per-token fast weights → chunk-cadenced CMS levels →
sleep-cadenced grown experts → frozen weights), with **surprise** (gradient of an inner loss) as the write signal and
**wake/sleep consolidation** as the lifecycle. The user's ASI is that spectrum arranged into a **cognitive
architecture**: a frozen AR **worker**, a continually-learning **supervisor** (the fast/slow memory spectrum), a
non-AR associative **right-brain** (System-1 intuition), a **neuro-symbolic external memory** (symbolic graph +
latent NM state), all under a verification-first **harness** (TaskOps) whose job is to keep continual learning
**honest** — to gate what gets written to durable memory so self-modification can't poison the mind. The single
most important structural realization (below): **the supervisor and the symbolic+latent graph-memory product are the
same build — TaskOps's work-graph is the symbolic scaffold; attaching NM latent state + verification-gated
consolidation to it produces both at once.**

---

## 1. Component ↔ substrate map (rigorous)

| User component | What it *is* in the substrate | Load-bearing papers |
|---|---|---|
| **Worker** (frozen AR LLM) | The "weights frozen" world; System-2 sequential executor. Attention = frequency-∞, unbounded-capacity associative memory (Atlas φ*, Miras NW row). | Transformer; BOOK ch1 |
| **Harness** (TaskOps) | An **outer nested-learning level**: a slow associative memory over the worker's attempts — compresses (task graph), retains (saturation=retention gate), writes on **surprise** (friction). Currently hand-coded / not learned. | Nested Learning §outer-optimizer; Miras retention |
| **Supervisor** (NM-based LLM) | A **test-time-memorizing model** (Titans/Atlas) organized as a **Continuum Memory System** (NL): per-token fast + chunk-cadenced + sleep-cadenced memory. Accumulates cross-task knowledge the frozen worker cannot. Write signal = surprise; lifecycle = wake/sleep. | Titans, Atlas, NL (CMS), Sleep |
| **Right-brain (@)** (non-AR intuition) | A **System-1 associative / energy-based / diffusion** engine that *proposes holistic hypotheses* (not golden answers). Substrate: modern Hopfield (retrieval = energy minimization = pattern completion), fast-weight association, ICL-as-implicit-optimization; non-AR generation (diffusion / energy-based). | Hopfield 2008.02217, Fast-Weight 2102.11174, ICL-GD 2212.07677 (+ diffusion LLMs — fit-paper research) |
| **External memory ① text library** | **CMS frequency-hierarchy applied to retrieval**: hot-recent (fast) → consolidated summaries (slow) → frozen canon (freq-0). Sleep = compress logs into durable library entries. | NL (CMS), Sleep (consolidation) |
| **External memory ② symbolic+latent graph** | **Neuro-symbolic continual memory**: symbolic KG scaffold (explicit = 명시지) where each node/edge carries an NM **latent state** (tacit = 암묵지) updated by surprise + consolidated by sleep. "Copy a brain" = init latent from a person's data. "Merge geniuses" = symbolic graph-union + latent state-merging. | NL (self-modifying memory, CMS), Sleep (Knowledge Seeding, Dreaming), Miras (retention), Hopfield (associative slots) |

**The dual axes of the whole system:**
- **Hemisphere axis**: AR-sequential (left: worker + verifier) vs non-AR-holistic (right: associative/diffusion intuition). = System-2 / System-1.
- **Memory-time axis**: frozen (worker) → fast test-time (supervisor) → slow consolidated (external memory) → symbolic-durable (graph). = the update-frequency continuum.

ASI = **hemisphere axis × memory-time axis, gated by verification-first TaskOps.**

---

## 2. The convergence (the key structural insight)

The user listed the supervisor and the symbolic+latent graph-memory as *separate* products. They are one build:

```
TaskOps work-graph            =  the SYMBOLIC scaffold (nodes = tasks/knowns/unknowns/frictions; edges = decomposition/dependency/resolution)
  + NM latent state per node  =  the TACIT/latent memory (surprise-updated associative state; "what I learned that I can't name")
  + verification-gated write   =  only self_verified→low-trust, externally_verified→durable  (honest consolidation)
  + wake/sleep consolidation   =  offline compaction of a session's frictions into cross-task knowledge (Knowledge Seeding upward)
  ─────────────────────────────
  =  BOTH the continually-learning SUPERVISOR and the neuro-symbolic GRAPH-MEMORY product, simultaneously.
```

This is why TaskOps is the linchpin, not a side tool: it already produces the **symbolic truth-graph** and already has the
**surprise (friction) / retention (saturation) / assurance-tier** primitives. Making its memory *neural + consolidating*
is the same act as building the supervisor and the graph-memory. The two products fall out of one architecture.

---

## 3. TaskOps re-read through the neural-memory lens — position, direction, upgrades

This extends the previous strategic analysis (verification-first; oracle-separation; assurance-tier→claimSafe; hard
test set) with the NM formalism. The two analyses **converge**: GPT-5.6 said "make the outer policy learned from
state-action-outcome"; Nested Learning says "the outer optimizer is the same associative-memory object as the inner
one." Same conclusion from two directions.

**3.1 Re-identify TaskOps in NM vocabulary.** TaskOps *is* an outer nested-learning level: a slow-frequency
associative memory over the worker's attempts. Its `surpriseHistory` = NM **surprise**; its `saturation`/fixpoint =
NM **retention gate** ("decline to retain more attempts on this resource"); its `inheritedContext` = a crude
**knowledge-transfer / meta-learned-init** channel; its retry/escalation ladder = a hand-coded **inner optimizer**.

**3.2 Concrete upgrades the NM line prescribes (each fixes a known TaskOps weakness):**

| TaskOps weakness (prior analysis) | NM prescription | Upgrade |
|---|---|---|
| novelty = string-signature change → timeout/noise/regression counted as "novel" | surprise = **magnitude of the update** an observation forces (gradient/info-gain), not signature difference | Replace signature-novelty with a **friction magnitude = expected-update size**; extend retry only on *reducible, relevant* surprise. Directly fixes the flawed retry-extension. |
| saturation = fixpoint heuristic (too narrow / eyeballed) | retention gate as a Bregman/f-divergence decision; capacity theory (Atlas) says forgetting is inevitable under finite compression | Make saturation a **retention/capacity decision** with a certificate (attempt/model/resource-pool/harness levels) — matches the "SaturationCertificate" idea and grounds the "model ceiling" claim. |
| one flat task graph; cross-task memory is just markdown inheritance | **Continuum Memory System**: a spectrum by update frequency | Give TaskOps a **CMS-structured memory**: per-attempt (fast) → per-task (chunk) → per-project (slow) → **sleep-consolidated** (durable). inheritedContext becomes principled. This *is* the supervisor. |
| no offline learning; each session starts cold-ish | **Sleep**: offline Knowledge-Seeding (distill lessons upward) + pruning + Dreaming (self-generated practice) | Add a **TaskOps "sleep" job**: after a work session, consolidate frictions→resolutions into cross-task knowledge, prune per-attempt noise, optionally "dream" synthetic practice tasks. Continual improvement between sessions. |
| hand-coded escalation policy | the outer optimizer is a *learnable* memory over (state, surprise) | Collect **state→action→outcome** transitions (already proposed) and let the escalation policy become a learned associative memory — the outer loop trained like the inner one. |

**3.3 The niche this earns (defensible, and it is OURS).** The neural-memory line's biggest *unanswered* risk
(PRE-RESEARCH §3.2.5; and the reward-hacking-as-budget-grows point): **task-free, safe self-modification has no
safety analysis; consolidation/Dreaming can write hallucinated lessons into durable memory.** TaskOps's
verification-first + assurance-tiering is exactly the missing control plane: **consolidation must be
verification-gated** — a `self_verified` lesson enters low-trust latent state; only `externally_verified` lessons are
promoted to durable/symbolic memory; Dreaming's self-generated data is quarantined until grounded. Positioning:

> **TaskOps = the verification-first control plane that makes continual-learning / self-modifying memory HONEST.**
> Not "a better retry loop" — the safety substrate without which a continually-learning mind poisons itself.

This is the same "scaling under imperfect verification / anti-reward-hacking" niche from the prior analysis, now
applied to the memory line — and it is the one thing none of the six papers (or the crowded test-time-scaling field)
provides.

---

## 4. Build order (evidence-disciplined, integrates the prior P0/P1)

- **P0 (legitimacy + honesty, from prior analysis, still first):** verifier 3-tier separation (self / independent /
  final-judge-once-no-feedback); `assuranceTier → claimSafe`; budget vector; hard/held-out test set. *These are also
  the write-gates for the memory below — do them first or the supervisor learns lies.*
- **P1 (the convergence build — smallest step that yields supervisor + graph-memory):** attach a **latent NM state +
  friction-as-surprise magnitude** to TaskOps work-graph nodes; a **verification-gated write** (tiered by assurance);
  a **per-session sleep consolidation** job (Knowledge Seeding upward). This is the neuro-symbolic memory MVP *and*
  the supervisor MVP.
- **P2 (CMS structure):** generalize the session memory to a frequency continuum (attempt/task/project/durable).
- **P3 (right-brain):** wire a non-AR intuition proposer (diffusion/energy/associative — see fit-paper research) as a
  hypothesis source feeding TaskOps's verify loop (System-1 proposes / System-2 verifies).
- **P4 (merge):** graph-union + latent-state merging across memories ("merging geniuses") — riskiest, last.

---

## 5. Honest guardrails (so this doesn't become an over-claim)

- The neural-memory line's empirical ceiling is **1.3B / 100B tokens**; the retrieval gap to attention is measured and
  unclosed; **zero serving-economics numbers** exist. The "completed form" is *conceptual*, not demonstrated at scale.
  Our blueprint inherits that risk — treat the supervisor/graph-memory as **research bets**, not shipped capability.
- Crowded neighbors we must NOT reinvent (confirm via fit-paper research): GraphRAG/HippoRAG (graph memory for RAG),
  model-merging/task-arithmetic (merging), diffusion LLMs (non-AR). Our whitespace is the *combination under
  verification-gated continual learning*, not any single piece.
- Prior-analysis lesson still binds: **measure base rates before building catchers; a well-argued feature can yield
  zero measured benefit (P2/P3 quiz null).** Every layer above must be gated by "can we measure it works?"

---

## 6. Deliverables (DONE — from the `asi-blueprint-research` workflow + synthesis)
- **`RESEARCH-MAP.md`** — §A fit-paper landscape (6 components), §B concept-KG pointer, §C the ranked whitespace, §D
  the through-line. Most-defensible whitespace = **TaskOps governs the memory-write/consolidation path** (novelty×utility
  admission, CF-as-pre-write-regression, consolidated=reproducible, versioned-auditable memory).
- **`CONCEPT-KG.json`** — 60 nodes / 125 edges / 8 clusters connecting the neural-memory line to the architecture.
- **`product-graph-memory.md`** — idea-note #2 (Mnemo-Graph: symbolic id + writable latent slot; overcapacity→split).
- **`product-right-brain.md`** — idea-note #1 (Intuit: experience-conditioned non-AR proposer + intrinsic-Δ routing; riskiest, defer).

**Triple convergence (the headline):** my synthesis, the neural-memory line's #1 open problem (safe self-modification),
and the whitespace critic all reduce to one sentence — *TaskOps's moat is governing **what a continually-learning
agent is allowed to learn** (the verification-gated write path), not what it outputs.* The supervisor, the
graph-memory, and the right-brain all hang off that one governed write path, which is why they are one build, not three.
