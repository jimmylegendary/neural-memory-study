# Research map — fit-papers, concept KG, and the genuine whitespace

Output of the `asi-blueprint-research` workflow (11 agents: 3 mechanism-extractors over the local corpus + 6
fit-paper searches + KG + whitespace critic). Concept KG (60 nodes / 125 edges / 8 clusters) → `CONCEPT-KG.json`.

> **Citation caveat (honesty):** the fit search surfaced both well-established papers and several *2026-dated*
> preprints (IDs like 26xx.xxxxx). The established refs below are reliable; the very-recent ones are marked
> **[verify]** — treat as leads to confirm, not settled citations, before any of them enters a paper.

---

## A. Fit-paper landscape (stand on these shoulders)

### A1 — Right-brain: non-AR intuition proposer (System-1)
- **Diffuse Thinking** (2510.27469) **[verify]** — dLLM as a *thought proposer* (not verifier); closest published role-match.
- **LLaDA** (2502.09992) — real open 8B non-AR diffusion LM; the deployable "whole-answer-at-once" substrate.
- **SEDD** (2310.16834) — score-entropy discrete diffusion; the theory under LLaDA/Mercury.
- **Mercury** (2506.17298) **[verify]** — production-scale diffusion LM; existence proof a fast parallel branch works.
- **Energy-Based Transformers / EBT** (2507.02092) — holistic candidate refined against a global energy; "think harder = more optimization steps." *(Caution: EBT frames energy as a VERIFIER objective — we want the opposite polarity, a proposer.)*
- **IRED** (2406.11179, ICML'24) + **Diffusion-of-Thoughts** (2402.07754) — non-AR multi-step reasoning that emerges holistically from energy/diffusion descent.
- Landscape verdict: two clusters — (1) deployable non-AR generation (SEDD→LLaDA→Mercury), (2) energy/iterative reasoning (EBT/IRED). **Nobody conditions the proposer's attractor landscape on the agent's own accumulated memory** — that's our opening (§C).

### A2 — Neuro-symbolic graph memory (symbolic AND latent, continual)
- **HippoRAG 2 / "From RAG to Memory"** (2502.14802) — mature hybrid: explicit triple-KG + embedding linking + PPR. Best-benchmarked baseline.
- **Zep / Graphiti** (2501.13956) — temporal KG for agent memory; fact invalidation + retroactive edits = the continual-update reference.
- **A-MEM** (2502.12110, NeurIPS'25) — self-organizing agentic memory that grows link structure + keeps latent embeddings.
- **LatentGraphMem** (2601.03417) **[verify]** / **HYMEM** (2603.10291) **[verify]** — the two closest claimed "latent+symbolic in one continually-updated memory" instances.
- **Neuro-Symbolic Reasoning over KGs survey** (2412.10390) **[verify]** + **GraphRAG survey** (2408.08921) — the design-space maps.
- Verdict: symbolic-first agent memory (GraphRAG/Zep/A-MEM) and latent KG-embedding are both mature but **not joined into a per-node writable latent state holding tacit knowledge** — our opening (§C).

### A3 — Continual graph learning + consolidation
- **Continual Learning on Graphs survey** (2402.11565) — vocabulary + baselines (EWC-style, replay).
- **Bayesian-Guided Continual KGE** (2508.02426) **[verify]** + **FastKGE / incremental-LoRA** (IJCAI'24) — add entities/relations over time without full replay / catastrophic forgetting.
- **MemGPT / Letta** (2310.08560) + **Generative Agents** (2304.03442, reflection-as-consolidation) — the self-editing-memory baselines.
- Verdict: continual-KG learning and agent-memory consolidation are two mature literatures **not yet joined**; and none gate consolidation on verification (§C).

### A4 — Merging ("combine geniuses")
- **Task Arithmetic** (2212.04089) → **TIES** (2306.01708) → **DARE** (2311.03099) → **Model Soups** (2203.05482): the mature weight-space merge stack.
- **FuseLLM/FuseChat** (2401.10491 / 2402.16107) — representation-space knowledge fusion across *incompatible* models (closest to "merge geniuses").
- **Sparse Upcycling** (2212.05055) — architecture-space alternative (preserve each expert, route via MoE).
- **Model Merging survey** (2408.07666) — design space + documented failure modes (interference, forgetting).
- Verdict: mature in **weight space**; the decision is *where* the merge happens (weight / representation / symbolic). For a symbolic+latent graph, merge = **graph-union (symbolic) + per-slot latent fusion** — a combination nobody ships.

### A5 — Dual-process (System-1/2) agent architectures
- **Talker-Reasoner** (2410.08328) — canonical two-box fast/slow agent (shared memory).
- **SwiftSage** (2305.17390, NeurIPS'23) — cheap fast executor by default, slow LLM on failure/uncertainty.
- **SOFAI + metacognition** (2508.17959) **[verify]** — metacognitive arbiter = slow controller over a fast executor.
- **Modular Agentic Planner / MAP** (Nature Comms 2025) — brain-inspired Monitor/Orchestrator gating a fast executor.
- Verdict: two-box split is well-supported; **our specific topology — a slow *verification-first* controller that governs the fast executor's *memory writes* — is only partially matched.**

### A6 — TTT-memory as an agent supervisor
- **Titans** (2501.00663) — the primitive to repurpose: surprise-gating maps onto a verifier's accept/reject.
- **Evo-Memory** (2511.20857) **[verify]** / **TAME** (2602.03224) **[verify]** — the eval regimes a "Titans-supervisor" claim must beat; TAME foregrounds *trustworthy* evolving memory (our exact frame).
- Verdict: "test-time learning for agents" has converged on **memory-content evolution** (RAG stores) or **meta-RL adaptation policy** — **a Titans-style neural memory as the SUPERVISOR over a verification-first harness is essentially unclaimed.**

---

## B. Concept KG (`CONCEPT-KG.json`)
60 nodes / 125 edges, clustered: Right-Brain (13), Graph-Memory (11), Supervisor (8), Text-Library (7), Harness&Worker (6),
5-Axis Memory Frame (5), Source Theory (8), Wake/Sleep (2). Edge relations: generalizes / is-special-case-of /
maps-to-component / enables / consolidates-into / provides-substrate-for.

---

## C. The genuine whitespace (ranked; the critic's honest verdict)

**MOST DEFENSIBLE — the TaskOps "govern the memory-write path" cluster.** Every self-improving-agent line (SEAL,
Reflexion, Voyager) leaves the write/consolidation path **completely ungoverned**; the neural-memory mechanisms hand
the exact hooks (surprise gate, Learning-to-Imitate reward, chunk-boundary snapshots) that only need an
agentic-governance wrapper nobody in ML-memory had reason to build. It sits on TaskOps's proven brand (deterministic
gates + Veridraft "no claim ships without a warrant"), is near-term buildable, and retires documented TaskOps
weaknesses. Four concrete claims:
1. **Two-gate write admission**: a memory write commits only if `surprise > θ` (novel) **AND** a utility gate passes
   (verified-useful, not a worker error / hallucination / flaky output). Warrant discipline extended from *outputs*
   to *what the agent is allowed to learn*.
2. **CF as a hard pre-write regression gate**: keep a ledger of consolidations that damaged prior tasks; before any
   write commits, re-run a held probe-set over prior knowledge — degrade → **block** the write + add region to the ledger.
3. **"Consolidated = reproducible, not stored"**: a lesson is complete only when the durable store *reproduces the
   source's answers* (probe-agreement above threshold) — a **semantic** completion gate that directly retires
   TaskOps's own "structural-not-semantic" finding.
4. **Versioned, auditable continual memory**: snapshot + version at every consolidation boundary (a natural checkpoint
   in the chunkwise mechanism); every belief traceable to the snapshot+write that produced it.

**STRONG — graph-memory as associative substrate:**
5. **Per-node writable latent slot** (fast-weight / Hopfield bank) updated by an **error-correcting delta rule** with
   a self-generated write-strength (read-then-convex-mix): writes *edit* the association, not clobber/accumulate.
6. **Associative overcapacity as the graph-restructuring trigger**: when a node's slot exceeds ~key-dim
   near-orthogonal pairs (silent wrong blends begin), **split the node / orthogonalize keys / spawn children** — a
   first-principles alternative to similarity-threshold community detection.
7. **Graph aging via "never-erase, decline-to-retain"**: content-derived per-edge retention (novel overwrites hard,
   redundant barely nudges); accessibility fades while state stays latently recoverable; symbolic tombstone for hard deletes.

**RISKIEST (defer) — experience-conditioned attractor proposer (right-brain item 1):** long dependency chain (needs
the latent graph substrate first), repurposes diffusion/Hopfield for an "insight" objective they weren't trained for,
**silent failure mode** (wrong-but-fluent intuitions). Adjacent thin idea worth keeping but not as a moat: expose the
proposer's confidence (retrieval separation Δ, metastable-set size k) as an intrinsic uncertainty the controller
routes on (low Δ → force verification).

**Stay OFF (crowded, not ours to claim):** non-AR-proposer+AR-refiner (Diffuse Thinking owns it), diffusion-LM-at-scale
(LLaDA/Mercury), graph+community-summary retrieval (GraphRAG), test-time memory architecture (Titans/Atlas/Miras),
abstract wake/sleep-for-LLMs (the Sleep paper).

---

## D. The through-line
All three of {my synthesis, the neural-memory line's #1 open problem (safe self-modification), the whitespace critic}
converge on one sentence: **TaskOps's moat is governing what a continually-learning agent is allowed to LEARN — the
verification-gated write/consolidation path — not what it outputs.** Everything else (supervisor, graph-memory,
right-brain) hangs off that governed write path.
