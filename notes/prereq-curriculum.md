# Part I: Background — Prerequisite Curriculum

**Target paper:** 150–200 pp. study of the Google Research TTT / neural-memory line
(Behrouz et al.): Titans (2501.00663), Miras / "It's All Connected" (2504.13173),
Atlas (2505.23735), TNT (2511.07343), Nested Learning (2512.24695),
Language Models Need Sleep (2606.03979).

**Audience:** transformer-inference / efficient-transformer systems experts with
**zero training background**. They are fluent in: KV cache, paged attention,
FlashAttention tiling, GEMM shapes, MFU, roofline, prefill/decode asymmetry,
batching, quantization, scan/prefix-sum kernels. They have never had to think
about a backward pass, an optimizer state, or a loss surface.

**Design principle:** the six papers' central move is *"an optimizer applied to a
small network IS the sequence-mixing layer, and conversely every optimizer IS an
associative memory."* Therefore the curriculum must teach training concepts **as
first-class objects with state, update rules, and cost models** — exactly the way
this audience already thinks about attention kernels — not as folklore for people
who "just train models." Every module ends with a *systems bridge* that maps the
new concept onto something the reader already operates in production.

**Part I page budget:** 66–78 pp. full path; 48 pp. minimum viable path (trim
column below). This leaves ~110–130 pp. for Parts II+ (the six papers, synthesis,
systems analysis).

---

## 0. What each paper actually assumes (grounding from the texts)

Skim evidence from `papers/*.txt`, used to size the modules:

| Paper | Background it explicitly leans on (from its §2 / preliminaries / method) |
|---|---|
| **Titans** 2501.00663 | softmax attention; kernel/linear attention and its recurrent form `M_t = M_{t-1} + k_tᵀv_t`; RNN-as-memory read/write; **gradient descent + momentum + weight decay reinterpreted as the memory update** ("surprise" = gradient of an associative-memory ℓ2 loss); decay/forget gates generalizing Mamba-2/Gated-DeltaNet gating; **chunkwise/mini-batch GD tensorization** citing the TTT dual form (Yu Sun et al. 2024); parallel associative scan (S5); sliding-window attention; hybrid architectures (MAC/MAG/MAL) |
| **Miras** 2504.13173 | attention; general linear-RNN form `M_t = A_t * M_{t-1} + v_t k_tᵀ`; **Hebbian vs. Delta learning rules**; forget gates as **retention regularization**; **FTRL vs. "Learning–Retaining" online-learning viewpoints** (§3, 16 mentions of FTRL); loss geometry beyond ℓ2 — ℓp, Huber, KL/f-divergence, Bregman divergence, elastic net; taxonomy table classifying RetNet/GLA/Mamba/DeltaNet/Longhorn/TTT/RWKV-7 by (memory arch, attentional bias, retention gate, learning algorithm); chunkwise parallel training of the three new models |
| **Atlas** 2505.23735 | everything in Miras, plus: memory-capacity theory of associative memories (Hopfield line, 16 mentions); **polynomial/exponential feature maps** as softmax approximators; the **Omega rule** (sliding-window inner objective, 56 mentions); **Muon as the inner optimizer** (Newton–Schulz orthogonalization, second-order flavor, 11 mentions); locally-optimal vs. online-GD memory (Mesa-layer contrast); DeepTransformers/Dot as strict generalizations of attention; parallelization of the Omega rule |
| **TNT** 2511.07343 | assumes TTT/Titans as given; **chunkwise parallel training is the paper's entire subject** (81 mentions of "chunk"): chunk-size speed-vs-quality tension, hardware utilization/throughput, non-linear inter-chunk recurrence (LayerNorm between chunks) as the parallelization blocker, hierarchical global/local memory, **periodic state resets → context parallelism**, two-stage train-large-chunk / finetune-small-chunk |
| **Nested Learning** 2512.24695 | the most theory-dense: formal associative-memory definition; **backprop itself as associative memory** (maps inputs → local error signals); **momentum as a 2-level memory; Adam as the *optimal* element-wise ℓ2 associative memory**; AdaGrad/Shampoo/Muon as preconditioned memories (110 mentions of momentum, 23 of Muon); nested/multi-level optimization with **update frequencies** per level; in-context learning as an emergent inner level; **Continuum Memory System** (spectrum of update frequencies replacing short/long-term dichotomy); self-referential learning (Schmidhuber); M3 optimizer; Hope module; continual learning framing (brain-wave / consolidation motivation) |
| **Sleep** 2606.03979 | builds directly on Nested Learning (cites Behrouz et al. 2025 for ICL-as-meta-learning and the amnesia analogy); **continual learning + catastrophic forgetting**; wake/sleep phases; **knowledge distillation** ("Knowledge Seeding" = generalized on-policy distillation), **RL-based imitation learning and RL for synthetic-data "Dreaming"**; replay; memory consolidation (96 mentions of "consolidat-", 109 of "sleep") |

Consequence: modules B1 (optimizers-as-objects) and B7 (chunkwise parallelism)
carry the most weight; B2 (online learning) is non-optional because of Miras;
B9 (continual learning + distillation + RL-lite) exists solely for NL + Sleep and
can be scheduled late.

---

## 1. Dependency graph

```
B0 Orientation & Rosetta Stone
 └─► B1 Training as a System: backprop + optimizers as objects
      ├─► B2 Online learning, OGD, regret, FTRL
      │     └─► B5 (Longhorn derivation), Miras §3
      ├─► B3 Meta-learning / bilevel / learning-to-learn
      │     └─► B6 TTT lineage
      └─► B4 Associative memory (Hopfield → delta rule)
            └─► B5 Linear attention & fast-weight programming
                  ├─► B5b SSM lineage S4 → Mamba → Mamba-2 (duality)
                  ├─► B6 TTT lineage proper (also needs B3)
                  └─► B7 Chunkwise-parallel training  ◄── B5b, B6
                        └─► B8 Systems bridge: cost models on your roofline
B9 Continual learning, distillation, RL-lite   ◄── B1, B3 (only NL & Sleep need it;
                                                    can be deferred to just before
                                                    Part II covers those two papers)
```

Linear reading order: **B0 → B1 → B2 → B3 → B4 → B5 → B5b → B6 → B7 → B8 → B9.**
Every module below states: why, who needs it, pages, canonical refs, bridge.

---

## 2. Modules

### B0. Orientation: the Rosetta Stone (inference world ↔ learning world)

- **Why.** The audience's entire mental model is *forward-only*. The six papers
  collapse the training/inference wall: a gradient step happens **per token at
  decode time**. Before any math, establish the dictionary and the two-loop
  picture so nothing later reads as category error.
- **Content.**
  - The two worlds: parameters frozen vs. parameters-that-move; "test time" no
    longer means "no learning."
  - The dictionary (to be maintained as a running margin feature through Part I):
    KV cache ↔ non-parametric, lossless, growing memory; linear-RNN state ↔
    fixed-size lossy compressed memory; a TTT layer's fast weights ↔ a "writable
    KV cache with a compression codec"; optimizer state (momentum buffer) ↔
    persistent per-layer accumulator; chunk ↔ tile; associative scan ↔ the
    prefix-sum kernels they already know; inner learning rate ↔ a data-dependent
    gate.
  - Inner loop vs. outer loop, stated once, informally (formalized in B3).
  - Map of the six papers and which module unlocks which.
- **Required by:** all six papers.
- **Depth:** 3–4 pp. (min path: 3)
- **Refs.** No external canon needed; forward-reference Sun et al. 2024
  (arXiv:2407.04620) §1 for the "hidden state is a model" slogan and Titans §2's
  memory-perspective taxonomy (read/write operations, Eq. 6–7).

---

### B1. Training as a system: backprop, SGD, momentum, AdamW, Muon — as OBJECTS

- **Why.** This is the load-bearing module. Titans' memory update **is literally
  GD + momentum + weight decay** (Titans §3.1: surprise = gradient, past-surprise
  = momentum, forgetting = weight decay). Atlas swaps the inner optimizer for
  **Muon**. Nested Learning's headline result is *"Adam is the optimal
  element-wise ℓ2 associative memory"* and reconstructs SGD/momentum/AdaGrad/
  Shampoo/Muon as memory modules. A reader who knows optimizers only as "the
  thing you import from PyTorch" cannot parse a single equation of these papers.
- **Content.**
  1. **Loss, gradient, backprop.** Scalar loss over a batch; reverse-mode AD as
     repeated VJPs; backward pass ≈ 2× forward FLOPs (systems framing first:
     GEMM shapes of forward vs. backward — `dW = xᵀδ` is an outer product /
     rank-b GEMM; this exact shape *is* the memory-write later). Activation
     memory and recomputation, briefly (needed to reason about TTT decode cost).
  2. **Loss surfaces**, only what's needed: non-convexity, curvature intuition,
     why step size matters, sharp vs. flat minima in two paragraphs. No
     convergence proofs.
  3. **The optimizer zoo, each presented as a stateful object with an
     `update(state, grad) → (state, Δw)` signature:**
     - SGD: stateless; `Δw = −η g`.
     - Momentum / EMA: state = one buffer; `m ← β m + g` — **flag explicitly
       that this is a linear recurrence in the same algebra as a linear-RNN
       state; the papers exploit this identity** (Titans Eq. 10; NL §4.4).
     - Weight decay: ℓ2 regularization vs. decoupled decay (AdamW); `w ← (1−λ)w
       − …` — **flag that `(1−λ)` is a forget gate**; this identity is Titans'
       forgetting mechanism and Miras' "retention regularization."
     - Adam/AdamW: two buffers, element-wise second-moment normalization;
       "sign-ish descent with per-coordinate learning rates."
     - AdaGrad → Shampoo (one paragraph each): preconditioning as choosing a
       metric; needed only for NL §4.
     - **Muon**: momentum + Newton–Schulz orthogonalization of the update matrix
       (msign); why orthogonalizing ≈ second-order-ish spectral normalization;
       cost = a few extra GEMMs — the audience will immediately grasp "5
       Newton–Schulz iterations = 5 small matmuls per step," use that. Required
       verbatim by Atlas (Eq. for Muon-updated memory) and NL (M3).
  4. Mini-batching vs. per-sample updates; the batch axis vs. the sequence axis
     (in TTT-line papers the *sequence* axis plays the role the *batch* axis
     plays in training — say this in bold, it defuses the single biggest
     confusion in the whole subject).
- **Required by:** all six. (Titans: GD+momentum+WD; Miras: "memory learning
  algorithm" axis; Atlas: Muon, Newton–Schulz; TNT: inherits; NL: entire §4;
  Sleep: consolidation = slow-frequency parameter updates.)
- **Depth:** 10–12 pp. (min path: 8 — cut Shampoo/AdaGrad detail and loss-surface
  visualization, keep Muon)
- **Canonical refs.**
  - Rumelhart, Hinton & Williams, *Learning representations by
    back-propagating errors*, Nature 1986.
  - Goodfellow, Bengio & Courville, *Deep Learning*, ch. 6–8 (reference text).
  - Robbins & Monro 1951 (SGD, historical anchor only); Polyak 1964 +
    Sutskever et al., ICML 2013 (momentum).
  - Kingma & Ba, *Adam*, arXiv:1412.6980; Loshchilov & Hutter, *Decoupled
    Weight Decay Regularization (AdamW)*, arXiv:1711.05101, ICLR 2019.
  - Duchi et al., *AdaGrad*, JMLR 2011; Gupta et al., *Shampoo*, ICML 2018
    (arXiv:1802.09568).
  - Jordan et al. 2024, *Muon: An optimizer for the hidden layers of neural
    networks*, kellerjordan.github.io/posts/muon (this blog IS the citation the
    papers use); Liu et al., *Muon is Scalable for LLM Training*,
    arXiv:2502.16982 (systems-flavored follow-up, good for this audience).
  - Li et al., *Visualizing the Loss Landscape of Neural Nets*, NeurIPS 2018
    (one figure's worth).
- **Systems bridge.** FLOP and memory-traffic budget of one optimizer step;
  optimizer state as the "register file" of training; why `dW = xᵀδ` has the
  same GEMM shape as a KV-cache append turned into an outer-product write.

---

### B2. Online learning, online gradient descent, regret, FTRL

- **Why.** Miras §3 frames every sequence model as an **online learner over the
  token stream** and proves Learning–Retaining generalizes **FTRL**; Longhorn
  (needed in B5) *derives* its SSM as the closed-form solution of an online
  learning problem; Titans' per-token memory update is online GD in disguise;
  Atlas' pitch ("locally optimal") is precisely online-vs-batch optimality over
  a sliding window. Regret is the paper-native language for "how good is this
  compression of the stream."
- **Content.**
  - Online protocol: at step t receive input, predict, suffer loss, update.
    Note this *is* autoregressive decoding's shape — the audience already runs
    this loop, they just never updated state by a gradient.
  - Online gradient descent (Zinkevich); regret definition; sublinear regret =
    "as good as the best fixed memory in hindsight" — translate into: the
    fixed-size state provably doesn't fall too far behind an oracle compressor.
  - Follow-the-Leader instability → **FTRL**: minimize (past losses +
    regularizer). One worked example: FTRL with ℓ2 regularizer recovers OGD.
    This is the exact scaffold Miras Eq.-level material stands on
    ("attentional bias" = the loss term, "retention gate" = the regularizer).
  - Mirror descent / Bregman divergence in one page (Miras' KL-retention
    'Memora' variant and f-divergence gates; skip proofs).
  - Loss-geometry sidebar: ℓp, Huber as robust alternatives to ℓ2 (Miras'
    Moneta/Yaad variants); one figure of loss shapes suffices.
- **Required by:** Miras (hard requirement), Atlas (Omega rule as sliding-window
  online objective), Longhorn/RWKV-7 background inside B5, Titans (implicitly),
  NL (level-wise objectives).
- **Depth:** 6–7 pp. (min path: 5 — compress mirror descent to a box)
- **Canonical refs.**
  - Zinkevich, *Online convex programming and generalized infinitesimal
    gradient ascent*, ICML 2003.
  - Shalev-Shwartz, *Online Learning and Online Convex Optimization*,
    Foundations & Trends in ML, 2011 (primary teaching source).
  - Hazan, *Introduction to Online Convex Optimization*, arXiv:1909.05207.
  - Orabona, *A Modern Introduction to Online Learning*, arXiv:1912.13213
    (FTRL chapters; freshest treatment).
  - McMahan, *Follow-the-Regularized-Leader and Mirror Descent: Equivalence
    Theorems and L1 Regularization*, AISTATS 2011.
- **Systems bridge.** Regret vs. recall@position curves; "the state is a cache
  with an eviction policy chosen by an optimization problem" — forget gate =
  learned eviction.

---

### B3. Meta-learning, bilevel optimization, learning-to-learn

- **Why.** The TTT-line's formal definition (quoted verbatim in Atlas §2:
  "the sequence model is a **meta in-context learner with two optimization
  levels**") is a bilevel program: inner loop = memory weights via GD on the
  attentional bias; outer loop = everything else via ordinary pretraining
  **through** the inner loop. Nested Learning generalizes two levels to K
  levels with per-level update frequencies; Sleep casts ICL itself as
  meta-learning. Without this module the reader cannot answer "wait, who trains
  the learning rate of the test-time learner?" — the single most common
  audience question.
- **Content.**
  - Bilevel optimization: inner problem, outer problem, hypergradient;
    differentiate-through-the-inner-update (unrolling) vs. implicit
    differentiation (one paragraph); truncated unrolling.
  - Learning-to-learn lineage: Schmidhuber 1987/1992 fast weights & 1993
    self-referential nets (NL cites these directly); Andrychowicz 2016
    (learned optimizers — the conceptual ancestor of "optimizer as module");
    MAML (initialization as the meta-variable — maps onto "learned initial
    memory state W₀", which Titans/NL both use).
  - What the outer loop learns in the TTT line: projections (k,v,q), the inner
    learning rates ηₜ, gates αₜ, momentum coefficients, memory init — i.e., the
    *hyperparameters of the inner optimizer become data-dependent, outer-learned
    functions*. This sentence is the skeleton key to all six papers.
  - ICL-as-implicit-GD results (von Oswald; Akyürek; mesa-optimization) as the
    empirical/theory bridge: attention already secretly does one GD step —
    these papers make it explicit and deep.
- **Required by:** Titans ("meta in-context model", learned gates), Miras
  (framework axis 4), Atlas (locally-optimal inner solve; Mesa-layer contrast),
  TNT (two-stage training = altering the inner problem between stages), NL
  (K-level nesting, self-modifying nets), Sleep (ICL-as-meta-learning premise).
- **Depth:** 6–8 pp. (min path: 5 — cite rather than derive hypergradients)
- **Canonical refs.**
  - Schmidhuber, *Evolutionary principles in self-referential learning*
    (diploma thesis, 1987); Schmidhuber, *Learning to control fast-weight
    memories: An alternative to dynamic recurrent networks*, Neural
    Computation 1992; Schmidhuber, *A 'self-referential' weight matrix*,
    ICANN 1993 (NL's declared ancestry).
  - Andrychowicz et al., *Learning to learn by gradient descent by gradient
    descent*, NeurIPS 2016, arXiv:1606.04474.
  - Finn, Abbeel & Levine, *MAML*, ICML 2017, arXiv:1703.03400.
  - Hospedales et al., *Meta-Learning in Neural Networks: A Survey*, TPAMI
    2021, arXiv:2004.05439.
  - Franceschi et al., *Bilevel Programming for Hyperparameter Optimization
    and Meta-Learning*, ICML 2018 (formal bilevel framing).
  - von Oswald et al., *Transformers learn in-context by gradient descent*,
    ICML 2023, arXiv:2212.07677; von Oswald et al., *Uncovering
    mesa-optimization algorithms in Transformers*, arXiv:2309.05858 (the
    Mesa-layer Atlas positions against); Akyürek et al., *What learning
    algorithm is in-context learning?*, ICLR 2023, arXiv:2211.15661.
- **Systems bridge.** Unrolled inner loop = a static dataflow graph you can
  compile/fuse (this is *why* chunkwise kernels in B7 exist at all); truncation
  length ↔ chunk size.

---

### B4. Associative memory: Hopfield → outer products → the delta rule

- **Why.** "Associative memory" is the *declared foundational abstraction* of
  the entire line: Miras/Atlas define models by Def. 1 ("mapping M: K → V
  learned under an attentional-bias objective"); NL escalates to "every
  component — including backprop and momentum — is associative memory"; Atlas'
  capacity theorems (why deep/polynomial memories beat matrix ones) only mean
  something against classical Hopfield-capacity results (TNT cites Hopfield 13
  times). The audience has never met any of this.
- **Content.**
  - Key→value association as the primitive; distributed storage in a matrix.
  - Correlation/outer-product (Hebbian) memories: `M = Σ vᵢkᵢᵀ`; retrieval
    `Mq`; crosstalk/interference when keys are non-orthogonal; capacity
    O(d) flavor. **Punchline: Hebbian write = the vanilla linear-attention
    update — memory overflow is crosstalk** (Titans §2's critique verbatim).
  - Hopfield 1982: energy view, attractors, ~0.14d capacity (one page,
    qualitative). Dense/modern Hopfield: Krotov–Hopfield polynomial energies →
    exponential capacity; Ramsauer et al.: modern-Hopfield update = softmax
    attention. **This chain (polynomial energy → capacity ↑; exp → softmax) is
    the exact theoretical scaffold of Atlas' feature maps φ_p and φ\*** — set it
    up here so Atlas §3.1–3.2 reads as a corollary.
  - **The delta rule (Widrow–Hoff LMS 1960):** `M ← M − η(Mk − v)kᵀ` = one GD
    step on ‖Mk − v‖²; "overwrite the old value for this key, don't just add."
    Present as *the* upgrade from Hebbian; note it is literally B1's SGD applied
    to a linear memory — the two curricula converge here, on purpose.
  - Attention as (non-parametric) associative memory: Bietti et al.; ties back
    to what the audience knows as the KV cache.
- **Required by:** all six (definitionally); deepest for Atlas + NL + TNT.
- **Depth:** 8–10 pp. (min path: 6 — compress classical Hopfield to 1 pp.)
- **Canonical refs.**
  - Kohonen, *Correlation Matrix Memories*, IEEE Trans. Computers 1972;
    Anderson 1972 (linear associative memory).
  - Hopfield, *Neural networks and physical systems with emergent collective
    computational abilities*, PNAS 1982.
  - Krotov & Hopfield, *Dense Associative Memory for Pattern Recognition*,
    NeurIPS 2016, arXiv:1606.01164.
  - Ramsauer et al., *Hopfield Networks is All You Need*, ICLR 2021,
    arXiv:2008.02217.
  - Widrow & Hoff, *Adaptive switching circuits*, IRE WESCON 1960 (delta
    rule/LMS; the papers cite the 1988 reprint).
  - Bietti et al., *Birth of a Transformer: A Memory Viewpoint*, NeurIPS 2023,
    arXiv:2306.00802 (cited by Titans/Miras/Atlas as "attention = associative
    memory").
- **Systems bridge.** KV cache = associative memory with exact (non-parametric)
  storage and O(N) lookup; matrix memory = lossy O(1)-state alternative;
  capacity results = "how big is the effective cache of a d×d state."

---

### B5. Linear attention as fast-weight programming; DeltaNet → Gated DeltaNet, Longhorn, RWKV-7

- **Why.** Every one of the six papers positions itself against this model
  family, and Miras' Table 1 literally classifies RetNet/GLA/Mamba/DeltaNet/
  Longhorn/TTT/Gated-DeltaNet/RWKV-7 inside one framework. This is also where
  the audience's efficient-attention knowledge gives maximum leverage — teach it
  as "the KV cache compressed into a d×d state," then re-derive each variant as
  an optimizer choice.
- **Content.**
  - Kernel trick on softmax: Katharopoulos et al.; associativity → recurrent
    form `Mₜ = Mₜ₋₁ + vₜkₜᵀ`, `yₜ = Mₜqₜ` (Titans Eq. 4–5). O(1) state, O(d²)
    per token.
  - **Fast-weight programming equivalence** (Schlag–Irie–Schmidhuber): linear
    attention = a slow net writing a fast weight matrix. This 1992→2021 arc is
    the historical spine of the whole study; give it a proper timeline figure.
  - Decay/gating family: RetNet (scalar decay), GLA (data-dependent
    channel-wise gate), general form `Mₜ = Aₜ ∗ Mₜ₋₁ + vₜkₜᵀ` (Miras Eq. 3).
  - **DeltaNet**: delta rule as the update (from B4); `Mₜ = Mₜ₋₁(I − ηₜkₜkₜᵀ) +
    ηₜvₜkₜᵀ` — a rank-1 Householder-like transition; why this fixes Hebbian
    crosstalk; Yang et al. 2024's parallelization made it practical (details in
    B7). **Gated DeltaNet** = decay + delta (Mamba-2 gate + targeted writes).
  - **Longhorn**: SSM update derived as *closed-form solution of an online
    associative-recall objective* — the cleanest concrete instance of B2's
    program; half a page of derivation.
  - **RWKV-7 "Goose"**: generalized delta rule — vector-valued gating,
    in-context learning rates, relaxed value replacement; state-tracking
    expressivity beyond TC⁰ (one paragraph + pointer).
  - Expressivity sidebar (½ pp.): negative eigenvalues (Grazzi et al.),
    DeltaProduct/multi-step, Merrill's illusion-of-state critique — needed
    because Titans/Atlas argue "deep memory" partly on these grounds.
- **Required by:** Titans §2 (direct), Miras (its Table 1 is this module),
  Atlas (baseline + Omega generalizes delta), TNT (contrast class "linear
  memory modules"), NL (linear attention is its running 2-level example),
  Sleep (indirect).
- **Depth:** 10–12 pp. (min path: 8 — RWKV-7 and expressivity sidebar to boxes)
- **Canonical refs.**
  - Katharopoulos et al., *Transformers are RNNs*, ICML 2020, arXiv:2006.16236.
  - Schlag, Irie & Schmidhuber, *Linear Transformers Are Secretly Fast Weight
    Programmers*, ICML 2021, arXiv:2102.11174; Irie et al., *Going Beyond
    Linear Transformers with Recurrent Fast Weight Programmers*, NeurIPS 2021,
    arXiv:2106.06295.
  - Sun et al., *Retentive Network*, arXiv:2307.08621.
  - Yang et al., *Gated Linear Attention Transformers with Hardware-Efficient
    Training*, ICML 2024, arXiv:2312.06635.
  - Yang, Wang, Zhang, Shen & Kim, *Parallelizing Linear Transformers with the
    Delta Rule over Sequence Length*, NeurIPS 2024, arXiv:2406.06484 (DeltaNet
    at scale).
  - Yang, Kautz & Hatamizadeh, *Gated Delta Networks: Improving Mamba2 with
    Delta Rule*, ICLR 2025, arXiv:2412.06464.
  - Liu, Wang, Wu, Feng, Stone & Liu, *Longhorn: State Space Models are
    Amortized Online Learners*, ICLR 2025, arXiv:2407.14207.
  - Peng et al., *RWKV-7 "Goose" with Expressive Dynamic State Evolution*,
    arXiv:2503.14456.
  - Grazzi et al., *Unlocking State-Tracking in Linear RNNs Through Negative
    Eigenvalues*, ICLR 2025, arXiv:2411.12537; Siems et al., *DeltaProduct*,
    arXiv:2502.10297; Merrill et al., *The Illusion of State in State-Space
    Models*, ICML 2024, arXiv:2404.08819 (sidebar only).
- **Systems bridge.** Per-token decode cost table: softmax attention (KV cache
  grows, O(N·d) reads) vs. linear state (O(d²) FLOPs, O(d²) resident bytes,
  bandwidth-bound elementwise+GEMV mix); state size vs. KV-cache size crossover
  math the audience can verify on their own serving stack.

---

### B5b. SSM lineage: S4 → Mamba → Mamba-2 and the linear-attention duality

- **Why.** The papers treat "modern linear RNNs" and SSMs as one family (Titans
  cites Mamba 27×; the general form `Mₜ = Aₜ∗Mₜ₋₁ + vₜkₜᵀ` covers both). The
  audience has likely *served* Mamba blocks without the derivation. Mamba-2's
  State-Space Duality is the formal glue that lets Miras put Mamba in the same
  table as linear attention, and its chunked algorithm previews B7.
- **Content.** Continuous SSM → discretization (ZOH in two lines, no ODE
  theory); S4/HiPPO in one paragraph (structured A for long memory); S5 /
  parallel associative scan (Blelloch) — the audience knows scans, exploit it;
  Mamba: selectivity = input-dependent (Δ, B, C) ⇒ gates, scan-based training;
  Mamba-2: scalar-times-identity Aₜ ⇒ **SSD: the SSM computes the same function
  as masked linear attention; semiseparable-matrix view; chunked
  block-decomposition algorithm** (1-2 figures). Position: Mamba-2's gate =
  the (1−αₜ) forget factor Titans generalizes to weight decay.
- **Required by:** Titans (gating lineage + baselines), Miras (table),
  Atlas/TNT (baselines), NL (frequency view of SSMs); Sleep (none — skippable
  for that paper alone).
- **Depth:** 6–8 pp. (min path: 5 — S4/HiPPO compressed to 1 pp.)
- **Canonical refs.**
  - Gu et al., *HiPPO*, NeurIPS 2020, arXiv:2008.07669 (one-paragraph cite).
  - Gu, Goel & Ré, *S4*, ICLR 2022, arXiv:2111.00396.
  - Smith, Warrington & Linderman, *S5*, ICLR 2023, arXiv:2208.04933 (the
    associative-scan formulation Titans reuses for momentum).
  - Gu & Dao, *Mamba*, arXiv:2312.00752 / COLM 2024.
  - Dao & Gu, *Transformers are SSMs (Mamba-2 / SSD)*, ICML 2024,
    arXiv:2405.21060.
  - Blelloch, *Prefix Sums and Their Applications*, CMU TR 1990; Martin &
    Cundy, *Parallelizing Linear Recurrent Neural Nets Over Sequence Length*,
    ICLR 2018, arXiv:1709.04057.
- **Systems bridge.** Scan kernels vs. chunked matmul forms: why Mamba-1 is a
  bandwidth-bound scan while Mamba-2/GLA turn the same math into GEMMs
  (tensor-core utilization) — direct rehearsal for B7.

---

### B6. The TTT lineage proper: test-time adaptation → TTT-Linear/TTT-MLP

- **Why.** Titans is explicitly "TTT + momentum + weight decay + deep memory at
  scale" (it adopts TTT's mini-batch tensorization); TNT's title is "TTT inside
  TTT"; Atlas/Miras cite Sun et al. 2024 as the founding deep-memory instance.
  The reader needs the original TTT layer cold before reading any of the six.
- **Content.**
  - Prehistory (1–2 pp.): test-time training for distribution shift (Sun et al.
    2020: auxiliary self-supervised task at test time); TENT (entropy
    minimization, adapting only norms) — establishes "updating weights at
    inference is a known, respectable trick," which disarms this audience's
    instinctive "you can't do that in prod."
  - **TTT layers** (Sun et al. 2024): hidden state = weights W of a small model
    (Linear or 2-layer MLP); update rule = one step of self-supervised
    reconstruction `ℓ(W; xₜ) = ‖f(θ_K xₜ; W) − θ_V xₜ‖²`; output = f(θ_Q xₜ; W).
    Learned inner learning rate; the **dual form** (mini-batch the inner GD over
    a chunk of tokens and tensorize into matmuls — forward-reference B7 where it
    is fully generalized); outer-loop training through the layer.
  - TTT-Linear ≡ (un-gated) delta-rule memory — make the B4/B5 identification
    explicit (Miras' table row "TTT-Linear: ℓ2 bias, no retention, GD").
  - Scaling evidence and video extension (Dalal et al. 2025) in half a page.
  - Where the six papers depart from TTT: momentum+decay (Titans), objective
    zoo (Miras), window objective + Muon (Atlas), training efficiency (TNT),
    K-level generalization (NL), consolidation of fast weights into slow
    weights (Sleep). One table; this table is the outline of Part II.
- **Required by:** all six directly.
- **Depth:** 6–8 pp. (min path: 6)
- **Canonical refs.**
  - Sun et al., *Test-Time Training with Self-Supervision for Generalization
    under Distribution Shifts*, ICML 2020, arXiv:1909.13231.
  - Wang et al., *Tent: Fully Test-Time Adaptation by Entropy Minimization*,
    ICLR 2021, arXiv:2006.10726.
  - Sun, Li, Dalal, Xu et al., *Learning to (Learn at Test Time): RNNs with
    Expressive Hidden States*, arXiv:2407.04620 (THE reference; teach from it).
  - Dalal et al., *One-Minute Video Generation with Test-Time Training*,
    arXiv:2504.05298 (cited by Atlas as Dalal et al. 2025).
- **Systems bridge.** Decode-time cost of a TTT layer = forward + backward +
  update of a tiny MLP per token (or per mini-batch of tokens): count the
  GEMMs, note backward-in-decode is new territory for inference engines
  (no autograd in most serving stacks → hand-derived gradients in kernels).

---

### B7. Chunkwise-parallel training of recurrent/fast-weight models — THE systems enabler

- **Why.** Without this module nothing in the line ships: Titans' §3.2 ("How to
  Parallelize the Long-term Memory Training") is its practicality claim; Atlas
  §3.4 parallelizes the Omega rule; **TNT is entirely about this module's
  limits** (chunk-size tension, LayerNorm-between-chunks blocking
  parallelization, resets for context parallelism). This is also where the
  audience's FlashAttention/tiling expertise becomes an unfair advantage —
  write it *for* them.
- **Content.** Teach the general scheme carefully, then instantiate four times.
  1. **The general scheme.** Split length N into chunks of size C. Within a
     chunk: compute all contributions *in parallel against the chunk-initial
     state* (matmul-rich, no recurrence). Across chunks: propagate state
     sequentially (or by associative scan when the transition is linear/
     associative). Formally: recurrences of shape `Sₜ = aₜSₜ₋₁ + bₜ` admit
     intra-chunk closed forms; the trick is choosing what to freeze at the
     chunk boundary so intra-chunk work becomes GEMMs. Complexity:
     O(N/C · [C²d + Cd² + poly(d)]) — chunk size interpolates between pure
     recurrence (C=1) and pure parallel attention (C=N). Draw the
     three-regime diagram (recurrent / chunkwise / parallel) once, reuse
     everywhere.
  2. **Instance 1 — vanilla linear attention / RetNet / GLA:** intra-chunk =
     local attention-like term + cross-chunk = state term; decays folded into
     per-chunk scalars/diagonals (Hua et al.; RetNet; GLA's
     hardware-efficient two-level tiling).
  3. **Instance 2 — DeltaNet:** non-diagonal (Householder) transitions ⇒ naive
     chunking fails; **WY representation / UT transform** turns products of
     rank-1 updates into two GEMMs per chunk (Yang et al. 2024). One careful
     page — this is the intellectual template for parallelizing *any*
     "optimizer-step" recurrence.
  4. **Instance 3 — TTT's dual form:** inner mini-batch GD over a chunk:
     gradients for all C tokens taken **w.r.t. the chunk-initial weights W₀**
     (that's the approximation!), summed/weighted ⇒ `W_C = W₀ − Σηᵢ∇ℓᵢ(W₀)` and
     outputs computed with an intra-chunk correction term — all GEMMs.
     Emphasize: **chunk size C is now a *semantic* hyperparameter** (it changes
     the function computed — staler inner gradients — not just the schedule);
     contrast with FlashAttention tiling, which is bit-exact. This distinction
     is the pivot of TNT.
  5. **Instance 4 — Titans:** adds momentum and decay: decay term ⇒ within-chunk
     LTI factors β; momentum ⇒ a *linear* recurrence over per-token gradient
     terms, solvable by associative scan (S5) inside the chunk;
     chunkwise-constant (η, θ, α) variant as an LTI simplification (Titans
     §3.2). Atlas' Omega-rule and Muon inner steps follow the same pattern
     (Newton–Schulz = a few extra GEMMs per chunk).
  6. **The systems frontier (TNT):** why deep memories still get poor MFU —
     nonlinear inter-chunk dependencies (LayerNorm/normalization between chunk
     states) serialize; small C ⇒ skinny GEMMs; TNT's answer: hierarchical
     global (large-chunk) + local (small-chunk, periodically reset) memories ⇒
     **state resets break the sequential chain ⇒ context parallelism across
     devices**; two-stage pretrain-large-C then finetune-small-C. Present
     TNT-the-technique here in skeleton form; TNT-the-paper's evaluation stays
     in Part II.
- **Required by:** Titans, Miras, Atlas, TNT (all hard); NL (HOPE/CMS training),
  Sleep (indirect via NL).
- **Depth:** 10–12 pp. (min path: 8 — merge instances 1–2)
- **Canonical refs.**
  - Hua et al., *Transformer Quality in Linear Time* (FLASH), ICML 2022,
    arXiv:2202.10447 (origin of the mixed chunk trick).
  - Sun et al., RetNet, arXiv:2307.08621 (explicit three-regime chunkwise
    formulation).
  - Yang et al., GLA, arXiv:2312.06635 (hardware-efficient chunkwise, two-level
    tiling, secondary chunking).
  - Yang et al., arXiv:2406.06484 (WY/UT-transform chunkwise delta rule).
  - Sun et al., arXiv:2407.04620, §2.5 + appendix (TTT dual form).
  - Titans arXiv:2501.00663 §3.2; Atlas arXiv:2505.23735 §3.4 (Omega
    parallelization); TNT arXiv:2511.07343 §2–4.
  - Blelloch 1990 (scan); Smith et al. S5 arXiv:2208.04933 (associative scan
    for momentum).
  - Software anchor: `flash-linear-attention` library (Yang & Zhang, GitHub,
    2024) — the de-facto kernel zoo for this whole family; worth citing for
    reproducibility.
- **Systems bridge.** This module *is* the bridge; end with a worked roofline:
  arithmetic intensity as a function of C for (GLA, DeltaNet, TTT-MLP, Titans),
  GEMM shape tables (C×d times d×d etc.), where tensor cores idle, why the
  quality-optimal C (small) and the MFU-optimal C (large) diverge — TNT's
  raison d'être, stated as a graph the reader could re-derive.

---

### B8. Consolidated systems bridge: reading this literature on your own roofline

- **Why.** Requirement (8): individual bridges appear per-module; this chapter
  consolidates them into reusable analysis tools for Part II and quietly
  teaches the reader to audit the papers' efficiency claims — the stance the
  whole study paper wants its audience to adopt.
- **Content.**
  - The full Rosetta table (B0's dictionary, now complete, two pages).
  - Cost-model cheat sheet per architecture class: state bytes, decode
    FLOPs/token, decode bytes/token, prefill parallelism type (attention-like /
    scan / chunkwise-GEMM / serialized), train-time MFU determinants. One row
    per: softmax attn + KV cache, SWA, linear attn, GLA/Mamba-2, DeltaNet,
    TTT-Linear/MLP, Titans-LMM, Atlas.
  - KV-cache-vs-fast-weights memory-budget arithmetic at 4K/64K/1M context.
  - Prefill/decode asymmetry for memory-as-weights models; what "the backward
    pass at inference" does to a serving engine (no autograd: fused
    hand-derived gradient kernels; state checkpointing across requests;
    per-session weight state = a new kind of session cache).
  - FlashAttention tiling vs. chunkwise training: exact vs. semantic tiling
    (recap of B7's pivot); roofline placement of each regime.
  - Glossary (2 pp.): every training-side term used in Part II with a
    one-line systems gloss.
- **Required by:** the reading experience of all of Part II; TNT especially.
- **Depth:** 5–6 pp. (min path: 4)
- **Canonical refs.** Williams, Waterman & Patterson, *Roofline*, CACM 2009;
  Dao et al., *FlashAttention*, NeurIPS 2022, arXiv:2205.14135 and Dao,
  *FlashAttention-2*, arXiv:2307.08691; Kwon et al., *vLLM/PagedAttention*,
  SOSP 2023, arXiv:2309.06180 (all as *anchors the audience knows*, cited to
  align vocabulary, not to teach).

---

### B9. Continual learning, consolidation, distillation, and RL-lite (for Nested Learning + Sleep)

- **Why.** NL's motivation is continual learning and catastrophic forgetting
  (anterograde-amnesia framing; consolidation; brain-wave frequencies →
  Continuum Memory System); Sleep's mechanism is **knowledge distillation
  (on-policy / generalized) + RL-driven synthetic-data "Dreaming" + replay**.
  None of this is inference-stack knowledge. Scheduled last because only these
  two papers need it — in the book it can sit immediately before their Part II
  chapters instead of inside Part I proper.
- **Content.**
  - Catastrophic forgetting: definition, why SGD on new data destroys old
    mappings (interference in shared weights — ties back to B4 crosstalk);
    classic mitigations in one page each: replay, regularization (EWC),
    parameter isolation.
  - Complementary Learning Systems (hippocampus fast / cortex slow) in one
    page — the papers' neuro analogies (consolidation, sleep, brain waves,
    multi-frequency updates) all decode against CLS; without it the NL/Sleep
    rhetoric reads as decoration.
  - Update-frequency framing: fast weights (per token) → slow weights
    (per pretraining) as a *spectrum*, preparing NL's CMS ("higher-frequency
    neurons adapt fast, lower-frequency store persistently") and Sleep's
    wake/sleep phase split.
  - Knowledge distillation: Hinton KD; sequence-level/on-policy distillation
    (GKD) — Sleep's "Knowledge Seeding" is generalized on-policy distillation
    upward (small fast memory → larger slow network), so teach the on-policy
    variant, not just logit matching.
  - RL in 3 pages, engineer's cut: policy gradient/REINFORCE, imitation
    learning vs. RL, RLHF/RLVR shape of modern post-training — just enough to
    parse Sleep's "Dreaming" (RL-generated curriculum, self-improvement) and
    its RL-based imitation component.
- **Required by:** NL (hard: CMS, continual-learning evals), Sleep (hard: both
  stages); others: none.
- **Depth:** 5–7 pp. (min path: 4 — cut EWC math, keep replay + CLS + KD + PG)
- **Canonical refs.**
  - McCloskey & Cohen, *Catastrophic interference in connectionist networks*,
    Psych. of Learning & Motivation 1989; French, *Catastrophic forgetting in
    connectionist networks*, TiCS 1999.
  - Kirkpatrick et al., *Overcoming catastrophic forgetting (EWC)*, PNAS 2017,
    arXiv:1612.00796.
  - McClelland, McNaughton & O'Reilly, *Why there are complementary learning
    systems in the hippocampus and neocortex*, Psych. Review 1995.
  - Hinton, Vinyals & Dean, *Distilling the Knowledge in a Neural Network*,
    arXiv:1503.02531; Agarwal et al., *On-Policy Distillation of Language
    Models (GKD)*, ICLR 2024, arXiv:2306.13649.
  - Williams, *REINFORCE*, Machine Learning 1992; Ouyang et al.,
    *InstructGPT/RLHF*, NeurIPS 2022, arXiv:2203.02155 (shape only).
- **Systems bridge.** Wake/sleep as a serving-fleet lifecycle: online per-session
  fast-weight state vs. periodic offline consolidation jobs (think: "background
  compaction for weights"); replay buffer as a data-plane artifact.

---

## 3. Page-budget summary

| # | Module | Full (pp.) | Min path (pp.) | Hard requirement of |
|---|--------|-----------|----------------|---------------------|
| B0 | Orientation & Rosetta Stone | 3–4 | 3 | all |
| B1 | Training as a system; optimizers as objects (incl. Muon) | 10–12 | 8 | all (NL, Atlas deepest) |
| B2 | Online learning, OGD, regret, FTRL | 6–7 | 5 | Miras, Atlas, Longhorn/B5 |
| B3 | Meta-learning / bilevel / learning-to-learn | 6–8 | 5 | all (NL, Sleep deepest) |
| B4 | Associative memory: Hopfield → delta rule | 8–10 | 6 | all (Atlas, NL deepest) |
| B5 | Linear attention & FWP; DeltaNet family, Longhorn, RWKV-7 | 10–12 | 8 | Titans, Miras, Atlas, TNT, NL |
| B5b | SSM lineage S4 → Mamba → Mamba-2 duality | 6–8 | 5 | Titans, Miras, Atlas, TNT |
| B6 | TTT lineage proper | 6–8 | 6 | all |
| B7 | Chunkwise-parallel training (the enabler) | 10–12 | 8 | Titans, Miras, Atlas, TNT |
| B8 | Consolidated systems bridge & glossary | 5–6 | 4 | all of Part II |
| B9 | Continual learning, consolidation, KD, RL-lite | 5–7 | 4 | NL, Sleep only |
| | **Total Part I** | **66–78** | **48** | |

Placement option: B9 can be moved out of Part I to a "Part II.5 interlude"
directly before the NL and Sleep chapters, keeping core Part I at 61–71 pp.

## 4. Paper-to-module requirement matrix

| Module | Titans | Miras | Atlas | TNT | Nested L. | Sleep |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| B0 | ● | ● | ● | ● | ● | ● |
| B1 backprop/optimizers | ●● | ● | ●● (Muon) | ● | ●●● | ● |
| B2 online/regret/FTRL | ○ | ●●● | ●● | ○ | ● | ○ |
| B3 meta/bilevel | ●● | ● | ●● | ● | ●●● | ●● |
| B4 associative memory | ●● | ●● | ●●● | ●● | ●●● | ● |
| B5 linear attn/FWP/Delta | ●● | ●●● | ●● | ●● | ●● | ○ |
| B5b SSM lineage | ●● | ●● | ● | ● | ● | ○ |
| B6 TTT lineage | ●●● | ●● | ●● | ●●● | ●● | ● |
| B7 chunkwise parallel | ●●● | ●● | ●●● | ●●●● | ● | ○ |
| B8 systems bridge | ● | ● | ● | ●●● | ● | ● |
| B9 continual/KD/RL | ○ | ○ | ○ | ○ | ●●● | ●●●● |

(●-count = depth of reliance; ○ = incidental.)

## 5. Sequencing rationale (why this order and not another)

1. **B1 before everything**: every later module treats an optimizer as a
   component; the reinterpretations (Titans' surprise/momentum/decay, NL's
   Adam-as-memory) are only "aha" if SGD/momentum/AdamW/Muon are already
   concrete `(state, update, cost)` objects.
2. **B2 before B5**: Longhorn and Miras' FTRL section are unreadable without
   regret/FTRL; conversely B2 is cheap (5–7 pp.) because the online protocol is
   structurally identical to autoregressive decoding the audience knows.
3. **B4 before B5**: delta rule must be met as a 1960s associative-memory
   update *before* DeltaNet, so DeltaNet lands as "LMS on a fast weight
   matrix," not as a novel gadget. Hebbian-crosstalk → delta-overwrite is also
   the narrative engine for why Titans/Atlas want richer inner objectives.
4. **B5 before B5b**: teaching Mamba-2 *after* linear attention lets the SSD
   duality do unification work instead of being trivia; Miras' table then
   closes the loop.
5. **B6 after B3+B5**: TTT is exactly the intersection (meta-learned online
   learner whose special case is linear attention); presenting it last among
   the model modules makes the six papers' departures enumerable in one table.
6. **B7 after all model modules**: chunkwise parallelism generalizes across
   GLA/DeltaNet/TTT/Titans; presenting the four instances against one general
   scheme (freeze-at-boundary + intra-chunk GEMM + inter-chunk scan) is both
   more compressive and precisely the abstraction TNT assumes.
7. **B8 as consolidation**: converts per-module bridges into the analysis
   toolkit Part II uses to audit each paper's efficiency claims.
8. **B9 last / relocatable**: exclusively serves NL + Sleep; front-loading it
   would tax readers before it pays off.

## 6. Verified external citations (web-checked 2026-07-11)

- Longhorn — arXiv:2407.14207, ICLR 2025 ([arXiv](https://arxiv.org/abs/2407.14207))
- RWKV-7 "Goose" — arXiv:2503.14456 ([arXiv](https://arxiv.org/abs/2503.14456))
- TTT-Linear/TTT-MLP — arXiv:2407.04620 ([arXiv](https://arxiv.org/abs/2407.04620))
- Gated DeltaNet — arXiv:2412.06464, ICLR 2025 ([arXiv](https://arxiv.org/abs/2412.06464))

Remaining citations are standard canon cross-checked against the six papers'
own bibliographies (e.g., Titans' reference list confirms FlashAttention-2,
S5, and the DeltaNet-parallelization NeurIPS 2024 entry verbatim).
