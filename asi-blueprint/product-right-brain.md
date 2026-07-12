# Product idea-note — Non-AR "right-brain" intuition engine ("직관 제안자")

*Working name:* **Intuit** (System-1 proposer). A **non-autoregressive** model that emits *holistic hypotheses /
intuitions* — not golden answers, not a verifier — for the AR worker + verification-first harness (System-2) to check.
Its distinctive bet: the proposer's landscape is **conditioned on the agent's own accumulated memory graph**, so an
"intuition" is one-shot pattern completion over *lived experience*, not a sample from frozen pretraining priors.

## The concept in one line
A holistic proposer (`completion, Δ, k`) that settles — in parallel, non-left-to-right — into an **attractor formed
from this agent's Mnemo-Graph latent slots**, and ships an **intrinsic confidence** (retrieval separation Δ,
metastable-set size k) that the controller routes on: high Δ → trust & proceed; low Δ → confabulation risk → force verification.

## Why non-AR / why "right-brain"
- AR generation commits left-to-right (System-2, sequential, verifiable). Intuition is **global and simultaneous** —
  it "sees the whole" before committing. The substrate offers three papered routes to that: **diffusion LMs** (LLaDA,
  SEDD, Mercury — parallel whole-answer generation), **energy-based** models (EBT, IRED — a candidate refined against
  a global energy; "think harder = more steps"), and **modern Hopfield / associative** retrieval (energy minimization
  = pattern completion). All are non-AR and holistic.
- Role, not architecture: **propose, don't verify.** This slots perfectly into TaskOps's verification-first design —
  the right-brain is the non-golden hypothesis source; TaskOps is the grounding. System-1 proposes / System-2 verifies.

## The whitespace it claims (from RESEARCH-MAP §C — and its honest risk rank: RISKIEST)
- **Experience-conditioned attractors**: bind the Mnemo-Graph's per-node latent slots in as the Hopfield stored
  patterns (or diffusion conditioning), so the proposer settles into attractors formed from *this agent's history* —
  vs Diffuse-Thinking/LLaDA which propose from a frozen pretrained dLLM that never sees the accumulated store.
- **Intrinsic calibration-ready confidence** `(Δ, k)` with every completion — a model-intrinsic uncertainty (no extra
  model call) that becomes the routing key between the intuition branch and the verifier. (Inverts EBT's energy: EBT
  uses energy as a verifier objective; we expose separation as a *proposer* self-signal.)

## Architecture sketch
```
PROPOSE(context, memoryGraph):
  patterns = latent_slots(memoryGraph relevant to context)   # experience conditioning
  x* , trace = non_AR_settle(context, patterns)              # diffusion denoise OR Hopfield energy descent
  Δ = retrieval_separation(trace); k = metastable_set_size(trace)
  return (x*, Δ, k)                                           # a hypothesis + its intrinsic confidence

CONTROLLER (TaskOps):
  (h, Δ, k) = PROPOSE(...)
  if Δ high:  hand h to worker as a strong lead
  else:       treat h as a hunch → require verify-grounding before any use   # low-Δ = confabulation risk
```

## Hardest subproblems (why it's the riskiest, defer to P3)
1. **Long dependency chain**: needs the Mnemo-Graph latent substrate to exist first (product #2). Don't start here.
2. **Objective mismatch**: diffusion/Hopfield cores were never trained for an "insight/usefulness" objective — a
   symbolic node "does not become a Hopfield net for free"; an energy proposer will **confidently fabricate plausible
   blends**. Silent, fluent-but-wrong failure mode.
3. **Δ is separation in embedding space, not truth**: a low-Δ blend can be correct, a crisp high-Δ attractor can be
   wrong. Calibrating Δ → actual-error-rate per domain is the real research work.

## MVP (once Mnemo-Graph MVP exists)
- Wrap an existing open non-AR model (LLaDA) as a **proposer** conditioned on retrieved Mnemo-Graph slots; ship
  `(completion, Δ, k)`; route low-Δ proposals into TaskOps's verify loop. **Measure**: do experience-conditioned
  proposals beat frozen-dLLM proposals at surfacing *useful, verifiable* leads? Does Δ predict verify pass/fail?

## Stand on (fit-papers)
LLaDA (2502.09992), SEDD (2310.16834), Diffuse Thinking (2510.27469 [verify]), EBT (2507.02092), IRED (2406.11179),
DoT (2402.07754); Hopfield (2008.02217) for the confidence geometry.

## Honest positioning
- **Do NOT claim** non-AR-proposer+AR-refiner (Diffuse Thinking owns it) or diffusion-LM-at-scale (LLaDA/Mercury).
  Our only novel claims are **experience-conditioned attractors** and **intrinsic-Δ routing** — both dependent on the
  graph-memory. Ship it as a *feature of the dual-brain system*, not a standalone foundation model.
- β (creativity dial: how far to let the proposer roam from stored attractors) is a real knob but "a feature, not a
  moat" — don't position it as core novelty.
