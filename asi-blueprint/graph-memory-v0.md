# Mnemo-Graph v0 — the minimal, honest starting design (Sleep-grounded)

Purpose: the *first thing to build* for the symbolic+latent graph, scoped so it (a) confronts the two porting cautions
head-on, (b) is falsifiable before any scale-up, (c) sits on the TaskOps work-graph so building it advances the
3-core ("worker + governed-continual-memory + TaskOps governance"). Grounded in the workflow's NL+Sleep extraction.

## The two cautions that kill a naive port (from the extraction — must be answered, not ignored)
1. **No end-to-end gradient.** NL/Sleep's "memory/surprise/frequency" are defined over differentiable weights trained
   by backprop. A symbolic graph has no such global gradient. → **A KG node does not become a Titans memory for free.**
2. **Knowledge Seeding needs shared logits.** Sleep's upward distillation is on-policy KD (teacher/student share an
   output distribution). A text/graph store has no shared logit distribution. → **distillation must be replaced.**

## v0's answers (this is the design)
- **Answer to caution 1 — the write signal is LOCAL, not global.** Each node's latent slot is a *linear associative
  memory* `W_n` (start with a matrix — Atlas: O(d_k) capacity, simplest; NOT a deep MLP yet). A frozen pretrained
  encoder gives `(k,v)` from an observation. The slot trains on its OWN local associative loss `||W_n·k − v||²` — a
  per-slot online regression. No backprop through the graph. This is the entire crux, resolved by staying local.
- **Answer to caution 2 — consolidation-completeness is REPRODUCTION, not logit-KD.** When knowledge moves upward, the
  child is "done" only when it *reproduces the parent's answers on a probe set* (query→read agreement above
  threshold). This is Sleep's Learning-to-Imitate reward stripped to answer-agreement — and it is *exactly* TaskOps's
  "consolidated = reproducible, not stored" governance gate. The two projects share this mechanism.

## The v0 mechanism (five operations)
```
node n = { symbolic:{id,label,provenance,assuranceTier},  latent: W_n (matrix),  health:{Δ separation, load} }

WRITE(n, obs)                      # surprise-gated, LOCAL, the only mutation
  k,v = encode_frozen(obs)
  surprise = ||W_n·k − v||^2       # Titans/NL: zero surprise ⇒ no write (store stays small & salient)
  if surprise < θ: return
  η = f(surprise)                  # content-derived write-strength (self-modifying Titans Eq.86)
  W_n = W_n(αI) − η·(W_n·k − v)·kᵀ # delta rule: EDIT the association (not clobber, not blind-accumulate)

READ(n, q) = W_n·k_q               # forward, side-effect-free, safe to call constantly

SLEEP()  (offline job; wake never touches it)  — order is STABILIZE then (optionally) DREAM:
  for each saturated node (Δ < τ  ⇐ Atlas overcapacity):
     child = GROW()                # Sleep: grow-don't-regularize; freeze parent, train only child
     REPLAY parent's (k,v) → child # port of Knowledge Seeding WITHOUT logits (replay, not KD)
     if child REPRODUCES parent on probe-set (answer-agreement > ρ):   # = TaskOps consolidated-gate
        PRUNE parent slot          # Sleep: synaptic pruning; parent gist now owned upward
     else: keep parent (promotion not done)

GOVERN (TaskOps): every WRITE + every SLEEP promotion passes two-gate admission
  = novelty(surprise>θ) × utility(verified-useful, not a worker error/hallucination);
  self_verified → low-trust latent only;  externally_verified → durable/symbolic promotion.
```
Forgetting is safe by the **CMS recoverable-loop** (§7.1): a pruned fast slot's gist survives in the grown child, so a
restore path can regenerate detail — no single point of failure.

## Operational definition of "tacit knowledge" (so it is testable, not mystical)
**Tacit = context-dependent associations `k→v` conditioned on the node/context, not expressible as one symbolic edge
or one static embedding.** A symbolic edge states *that* A relates to B; the slot stores *how* a key-pattern completes
to a value *in this node's context* — knowable only by probing (query k → read v), which is precisely what "tacit"
means. Concrete: node `python-tracebacks` whose slot maps {a specific traceback pattern → the fix approach}, learned
from many observations, never written as a rule. This makes the philosophical claim a falsifiable one (below).

## The falsifiable v0 experiment (earn-your-place, per the session ethos)
- **Baseline A**: static-embedding graph node (GraphRAG/HippoRAG-style; retrieval only, no writable slot).
- **Treatment**: surprise-gated associative slot (above).
- **Task**: a stream with a **context-dependent** regularity non-linear in the embedding — e.g. `in context A: K1→V1`
  but `in context B: K1→V2` (same key, different value by node/context) — which a single static embedding *cannot*
  represent but an associative slot can.
- **Measure**: (1) recall of the context-dependent association; (2) interpretability (probe the slot); (3)
  boundedness under a long bursty stream (does grow-prune keep it bounded?); (4) graceful forgetting + restore.
- **Falsifier (STOP condition)**: if Baseline A matches the slot → the latent layer does **not** earn its place; the
  whole product is demoted. If the slot wins on (1)+(2) → tacit capture is real; proceed to v1 (deep-MLP slot, CMS tiers).

## Idea-extraction brief for the neural-memory seminar session (what to lift, and its porting status)
- **LIFT DIRECTLY (mechanism transfers):** surprise-gated write (Titans/NL §3.1); content-derived learning-rate &
  retention (self-modifying Titans Eq.86); grow-don't-regularize + freeze-old/train-new + synaptic pruning (Sleep
  §3.2/3.3); consolidate-then-dream ordering (Sleep §3.4 — CF-robustness); wake/sleep phase separation;
  forgetting-as-recoverable-loop → restore path (CMS §7.1); overcapacity→split trigger (Atlas capacity bound).
- **PORT WITH A SURROGATE (does NOT transfer as-is):** Knowledge Seeding logit-KD → **answer-agreement reproduction**;
  end-to-end gradient write signal → **local per-slot associative loss**.
- **DEFER / RISKY:** Dreaming (self-generated data + RL) — CF-prone when iterated; only after stabilize-first + under
  TaskOps governance. Self-modifying encoder (Eq.83-88) — elegant but heavy. M3 multi-timescale momentum — compute overhead.
- **DO NOT PORT:** geometrically-clean divisibility chunk scheduling — real ingestion is **bursty and irregular**;
  keep consolidation event-triggered (on saturation / idle), not on a clean `C^ℓ` cadence.

## Where this sits in the plan
This v0 is simultaneously the **graph-memory MVP** and the **memory subsystem of the future independent TaskOps
harness tool**. Sequencing: TaskOps-as-skill → bench validation (the P0 hard test set) → extract to independent
harness agent tool, *with this governed latent-memory as its memory subsystem*. Build order stays: P0 (governance =
the write-gates) first, then this v0 latent slot, then CMS tiers, then (only if earned) the non-AR proposer.
