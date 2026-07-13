# Graph-NM — extending the neural-memory framework from sequences to graphs

The neural-memory line generalized memory to an *associative memory* but only applies it along a **sequence** (the
update recurrence runs over position t). Our memory is a **graph model**. This is the research map for porting the NM
framework to graphs — from a large search of the GNN + graph-memory + graph-associative-memory literature, framed
axis-by-axis. (Citation caveat: refs flagged **[verify]** are recent/workshop and need confirmation before citing.)

## The core object (the one thing to build)
> **graph-NM = Titans' surprise-gated, retention-regularized, test-time-updated associative memory, with the sequence
> recurrence replaced by topology-propagated message passing.**

Precisely: every node (and optionally every supernode in a coarsening hierarchy) carries a small **associative memory
module `M_i`** (a parameterized k→v map, *not* a fixed RNN hidden state). On each graph event at node i:
1. compute **per-node surprise** `s_i = ‖M_i(k) − v‖` (Titans-sense: how much the event violates what the node's memory predicts),
2. take an **inner-loop gradient step** on that associative loss whose **learning rate is gated by `s_i`** (violations written hard, predicted events barely),
3. apply **capacity-aware decay** (forgetting rises with memory load + predictability),
4. **propagate the write through topology**: neighbors receive a surprise-weighted message and take *damped* associative steps — **consolidation flows along edges (message passing), not along a sequence**, with cross-node interference bounded by a topology-indexed crosstalk limit.
Read = content/structure-addressed associative recall (energy-descent pattern completion over the node + neighborhood).
A coarsening hierarchy gives **update-frequency tiers** (fine=every event, coarse=error-gated fast→slow handoff = wake/sleep).

## The core open problem (verbatim from the synthesis)
Every building block exists in a **separate** literature; **no published system assembles them** — there is no graph
memory whose **structure-addressed associative write is surprise-gated and topology-propagated at inference.** That
assembly is the one open problem.

## NM 5-axis → graph mapping (giants' shoulders, per axis)
| NM axis (sequence original) | Best existing graph mechanism | What's still missing |
|---|---|---|
| **Architecture** (memory IS a small neural module) | TGN per-node memory `s_i` (GRU-written, attention-read) `2006.10637`; **gLSTM** puts an outer-product associative store *in* a node `2510.08450` | in dyn-GNNs the memory is a recurrent hidden STATE, not a module optimized at inference; gLSTM is closest but not test-time-written |
| **Attentional bias** (inner k→v associative loss) | Graph Hopfield energy `2603.03464`[verify]; CDAM `2404.07123`; Energy Transformer `2302.07253` | patterns fixed at train time on a STATIC graph; no continual per-node associative loss whose gradient fires the write |
| **Retention** (adaptive forget/decay) | JODIE learned projection = between-event decay `1908.01207`; GSSC `2406.05815` | all graph decay is smooth recency/time drift, NOT capacity-aware surprise-scaled forgetting |
| **Inner-optimizer** (learn-at-test-time) | Graph TTT: GT3 (params) `2210.08813`, GTrans (data) `2210.03561`; GraphSSM (memory-as-online-objective) `2406.00943` | TTT updates ALL shared params via a uniform global SSL loss and RESETS per graph — episodic, ungated, not per-node accumulating |
| **Update-frequency** (CMS tiers) | HMNet multi-rate persistent tiers `2305.17852`; DiffPool learnable coarsening `1806.08804`; Loukas spectral guarantees `1808.10650` | coarsening is train-time single-sweep, discarded after forward pass — no persistence, no per-tier cadence, no gating |
| **Surprise-gated write** (lifecycle) | Persistent Message Passing persist-vs-overwrite gate `2103.01043`; DyRep intensity gate | every gate is INPUT- or event-rate-conditioned, **never derived from the memory's own prediction error** |
| **Capacity / crosstalk** (lifecycle) | **gLSTM**: over-squashing = sensitivity vs **capacity**, O(mem-dim) crosstalk cliff `2510.08450`; effective-resistance bound `2302.02941`; Löwe-Vermet Hopfield-on-random-graph capacity `1303.4542` | no capacity theorem for a *learned* message-passing memory indexed by graph topology (spectral gap / degree / community overlap) |
| **Wake/sleep consolidation** (lifecycle) | SGNN-GR generative "dreaming" replay (KDD'22); PI-GNN grow-don't-regularize `2305.13825`; HPN prototype growth `2111.15422` | all OFFLINE + task-boundary-driven, not online surprise-gated fast→slow handoff |

## Verdict on my three pre-guessed parallels (honest — all NUANCED, which favors us)
1. **TGN-memory = per-node associative memory** → **nuanced-toward-REFUTED.** TGN gives the *skeleton* (per-node
   persistent state + attention read) but it's a train-frozen GRU hidden state with no associative KV loss, no
   surprise gate, no test-time learning. **gLSTM is the only work with a real associative store in a node.** So TGN is
   the substrate to *retrofit*, not the memory itself.
2. **Over-squashing = capacity/crosstalk** → **CONFIRMED, but only for HALF.** gLSTM decomposes over-squashing into
   *sensitivity* (reachability — effective resistance/commute time) vs *capacity* (interference — O(d) cliff). Only
   the **capacity** flavor is the true associative-crosstalk analog (mirrors Atlas's O(d_k)); sensitivity is about
   reachability, not storage.
3. **Coarsening = CMS tiers** → **nuanced-toward-ASPIRATIONAL.** The structural analogy is exact, but no coarsening
   method is a *memory* (pooling = train-time single-sweep; HMNet = fixed spatial pyramid, clocked not gated). **The
   parallel names the right unbuilt object — it is our whitespace, not prior art.**

The favorable read: my intuitions pointed at the right *objects*, and the literature shows they are **unbuilt**, not
already-realized. The three "parallels" are three pieces of open whitespace.

## Whitespace — what is genuinely ours (ranked)
**MOST DEFENSIBLE (start here): the surprise-gated graph write.** Write strength = the gradient magnitude of each
node's OWN associative KV loss `∝ ‖∇_{M_i} L_assoc(k_i, v_i)‖` — replacing TGN's unconditional GRU write. Every
facet's "what-doesn't-exist" converged on this. It is a **single, falsifiable, isolatable** contribution: swap it into
TGN's memory updater, ablate against exact TGN/JODIE baselines, and show it matters precisely where **memoryless
models (DyGFormer/GraphMixer) fail — long-horizon, low-redundancy event streams.**

Then, in dependency order:
- **Topology-indexed capacity theorem** — how many associations a node stores before message-passing interference
  corrupts recall, indexed by effective resistance / spectral gap / neighbor overlap (extends gLSTM's O(d) cliff +
  Löwe-Vermet to a *learned* memory). The hard part: the interaction of the two over-squashing modes (hard-to-REACH vs FULL).
- **Test-time (online) fast→slow consolidation** — a learnable coarsening operator defines slow-tier supernodes;
  consolidate a fast node's accumulated surprise into its supernode *during* inference (not offline task-boundary replay).
- **Subgraph-valued associative memory** — stored patterns ARE subgraphs; retrieval = subgraph pattern completion.
  Hardest: a differentiable energy whose attractors are permutation-invariant, variable-size subgraphs.
- **Structure-addressed associative write** — address slots by relational/topological key (anchor + hop-signature) so
  recall generalizes across isomorphic neighborhoods (PMP write-gate × associative recall × learned forgetting).

**STAY OFF (crowded, not ours):** per-node memory for dyn-GNNs (TGN/JODIE own it), graph pooling/coarsening
(DiffPool/SAGPool own it), continual-GNN anti-forgetting (mature: ER-GNN/PI-GNN/HPN/SGNN-GR), over-squashing
diagnosis/rewiring (Alon-Yahav/Topping/Di Giovanni own it), graph-Mamba/SSM long-range (saturated 2024), graph TTT
under distribution shift (GT3/GTrans own it), memoryless windowed CTDG on their benchmarks (their trap — arguing
against our own thesis), Hopfield-read bolted onto a GNN (Energy Transformer owns it).

## Where to START (the falsifiable v0 of graph-NM — one clean paper/build)
**"Surprise-gated associative write for temporal graph memory."** Take TGN as the substrate. Replace its GRU write
with: per-node associative module `M_i`; on event, `s_i = ‖M_i(k_i)−v_i‖²`; write `M_i ← M_i − η(s_i)·∇_{M_i}s_i` with
`η ∝ ‖∇s_i‖` + capacity-aware decay; propagate a surprise-weighted damped step to 1-hop neighbors. **Ablate** vs
vanilla TGN / JODIE / memoryless DyGFormer on **long-horizon, low-redundancy** temporal-graph benchmarks (the regime
where memory should matter). **Falsifier:** if surprise-gating does not beat the unconditional GRU write there, the
whole graph-NM thesis is weak — stop. This is small, isolatable, and directly extends `gLSTM`'s in-node associative
store with the one missing thing (a surprise gate).

## Connection to the rest of the blueprint
- This is the **graph** version of `graph-memory-v0.md`: the earlier v0 attached a per-node slot; the *graph-specific*
  novel axis is **topology-propagated surprise + a topology-indexed capacity bound** — that is the part sequences don't have.
- **TaskOps governance still wraps it**: the surprise gate decides *whether* to write; the two-gate admission
  (novelty × utility) and reproduction-completeness decide *whether to promote/consolidate* — the write-path governance
  is the same across sequence-NM and graph-NM.
- gLSTM (`2510.08450`) is the anchor prior work; our contribution over it = surprise-gating + test-time write +
  topology-propagated consolidation + CMS tiers + a governed write path.
