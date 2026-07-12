# Product idea-note — Neuro-symbolic continual graph memory ("암묵지 그래프")

*Working name:* **Mnemo-Graph** (symbolic id + latent slot). A memory that is simultaneously **symbolic** (an
inspectable, addressable knowledge graph = 명시지) and **latent** (a small writable associative state per node/edge =
암묵지), continually updated by surprise and consolidated by a verification-gated sleep. This is also — per the
blueprint's convergence — the substrate of the **supervisor** and the natural extension of the **TaskOps work-graph**.

## The concept in one line
Give every symbolic node/edge a tiny **fast-weight / Hopfield associative slot** as its tacit latent state, written by
an **error-correcting delta rule** (read current value → write a convex mix, so updates *edit* the association rather
than clobber or blindly accumulate), and let **associative overcapacity** be the graph-restructuring signal.

## Why now / why us
- The neural-memory line just formalized every piece we need: surprise = write signal (Titans), retention =
  never-erase-decline-to-retain (Miras), capacity = O(d_k) pairs before silent crosstalk (Atlas), consolidation =
  Knowledge-Seeding + pruning (Sleep). None of them attach these to a *symbolic, auditable* graph.
- The agent-memory field (GraphRAG, HippoRAG 2, Zep/Graphiti, A-MEM) built the symbolic/temporal graph but kept nodes
  as **static text + a frozen embedding** — retrieval-only, no writable latent state, no in-place association edit.
- TaskOps already produces a **symbolic truth-graph with provenance + assurance tiers** — we own the governed-write
  substrate the whole thing needs.

## The whitespace it claims (from RESEARCH-MAP §C)
1. **Per-node writable latent slot** via error-correcting delta rule + self-generated write-strength — genuinely
   novel vs static-embedding graph memory.
2. **Overcapacity → node split**: use the associative capacity limit (~key-dim near-orthogonal pairs, then silent
   wrong blends) as the clustering/splitting criterion — a first-principles alternative to similarity-threshold
   community detection.
3. **Retention-premetric aging**: content sets per-edge write-strength + decay; accessibility fades while state stays
   latently recoverable (human-like); symbolic **tombstone** for correctness-critical hard deletes.
Combined they are a memory that *learns tacit regularities in-place, restructures itself at its capacity limit, and
forgets gracefully* — none of which retrieval-only graph memory does.

## Architecture sketch
```
Node n = { symbolic: {id, label, text, provenance, assuranceTier},
           latent:  W_n  (small fast-weight matrix / Hopfield bank),
           health:  {separation Δ, load, retention} }
Edge e = { symbolic: {type, endpoints, tombstone?}, retention scalar (content-derived) }

WRITE(observation x)               # the ONLY mutation; where the gate lives
  k,v   = encode(x)
  surprise = ||W_n(k) - v||^2      # Titans: prediction error = novelty
  if surprise < θ: return          # routine → near-zero writes (self-throttling)
  η = write_strength(surprise)     # data-dependent (Titans)
  W_n = (1-η)·W_n + η·delta(k,v)   # error-correcting edit, not clobber
  if separation(W_n) < τ: SPLIT(n) # Atlas overcapacity → restructure
READ(query q)  = W_n(k_q)          # forward pass, side-effect-free, safe to call constantly
CONSOLIDATE()  = verification-gated sleep job (see TaskOps note): promote self_verified→durable,
                 run CF probe-regression, snapshot+version.
```

## Where the write signal comes from (the hardest subproblem)
An *isolated* slot has no end-to-end backprop over the graph. Substitute the associative loss surrogate:
`||W_n(k) - v||^2` on locally encoded (k,v) from the node's own retrievals/observations — a **per-slot online
regression**, not a global objective. Open: how (k,v) are encoded (shared frozen encoder vs learned), and calibrating
the split threshold τ to real crosstalk, not embedding artifacts. **This is the make-or-break research question.**

## MVP (buildable on TaskOps now)
- Attach a matrix-valued slot to each TaskOps work-graph node; write frictions/resolutions via the delta rule with a
  surprise gate; read = content-addressed "have we seen this / what worked."
- Instrument per-slot separation Δ; when Δ crosses τ, split the node (proves the overcapacity-as-restructuring claim
  on real data). **Measure**: does the split criterion beat a similarity-threshold baseline at recall + interpretability?
- Everything write-gated by the TaskOps two-gate admission (novelty × utility) → honest by construction.

## Stand on (fit-papers)
Titans (surprise/delta), Atlas (capacity/crosstalk), Miras (retention), Fast-Weight Programmers (2102.11174, the slot
mechanics), Hopfield (2008.02217, energy/capacity); baselines to beat: HippoRAG 2 (2502.14802), Zep (2501.13956),
A-MEM (2502.12110). Continual side: Continual-Graph survey (2402.11565), Bayesian-CKGE (2508.02426 [verify]).

## Honest risks
- "A KG node does not become a Hopfield net for free" — the latent-slot infra (encoders, per-slot training, storage)
  is real engineering, not an analogy.
- Silent failure: overcapacity returns wrong blends with no error raised — the Δ monitor MUST fire before recall
  corrupts, or the memory quietly rots.
- Storage/versioning cost of a growing latent+symbolic store is unsolved (meaningful diff over neural slots).
- Crowded adjacency: don't reframe as "GraphRAG++"; the claim is the **writable, self-restructuring, forgetting**
  latent layer, not graph retrieval.
