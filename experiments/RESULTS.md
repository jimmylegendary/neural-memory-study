# Part III experimental results — narrative for authors

> **Status**: all planned local experiments executed (2026-07-12). This file is the citable
> narrative; the machine-readable warrant source is [`experiments/results.json`](results.json)
> (consolidated, per-number provenance + caveat tags) and each `experiments/<exp>/results.json`.
> Figures: [`figures/exp-*.png`](../figures/) (repro: `figures/render_experiments.py`).
>
> **Honesty contract (applies to every number below).** These are **exploration-grade** results
> from pure analytic cost models (hatir + hat-schema twins) and CPU micro-benchmarks. The
> load-bearing claims are **ratios, crossover positions, tier orderings, and bound
> classifications** — never silicon-accurate absolutes. The 6 source papers publish **no H100
> wall-clock decode**, so absolute µs/token and mJ/token are roofline **lower bounds** and are
> **carried forward to the in-house A100 runbook** (Part III-a). Novel device twins
> (scratchpad, PIM) are `simulation_ready=False` and cited as **directional DSE only**.

## What we set out to measure

The spine of Part III is the **D4 workload-split pair thesis**:

> **decode/serving-state management is a memory-centric opportunity** — a per-token
> read-modify-write (RMW) of the *entire* fast-weight state, **unshared** across sequences,
> **write-heavy**, and **not content-addressable**, which is qualitatively different from the
> append-once/read-many KV cache — while **training/prefill is the accelerator regime**, where
> the chunk size C is literally the x-axis of the roofline.

Eight experiments quantify each half. Three independent methods — a closed-form model (E3), the
hatir tool (E1.x), and a plan hand-calc — **agree to <1%** on the anchor config
`neural-mem-1.3B (d=2048, m=16, L=24, GQA-8, bf16)`, which is the cross-validation backbone
(see `cross_validation` in the consolidated JSON).

---

## Decode half of the thesis

### 1. Decode is a memory-bound whole-state RMW (E1.1, E3)
The anchor decode step moves **6.44 GB/token** (read `m·d²` + write `m·d²` over L=24 layers), at an
arithmetic intensity of **0.59 FLOP/byte** — two-plus orders of magnitude below the H100 twin ridge
of 295 FLOP/B, so the step is **deterministically memory-bound**. State traffic dominates the
decode-step GEMV compute by **≈394×**: *the step cost simply IS the state traffic*. The roofline
lower bound is **1.92 ms/token** and **46.5 mJ/token** (deferred as absolutes; the *structure* —
memory-bound, RMW-dominated — is the claim). RMW traffic scales with model width
(`∝ m·d²·L`): 0.45 → 6.44 → 343 GB/token from Titans-170M to a hypothetical 70B.
→ **Part III ch19 scaling, ch12/ch14 decode-cost boxes.**

### 2. KV vs TTT are qualitatively different memory workloads (E1.3, E3) — Figure (a)
The traffic-composition contrast is the cleanest single result:

| workload | write as % of read | shareable? | grows with context S? |
|---|---|---|---|
| **KV cache** | **0.003%** (append-once) | yes (prefix reuse) | **yes**, read ∝ S |
| **TTT state** | **100%** (symmetric RMW) | **no** (per-seq, always dirty) | **no**, constant in S |

Because KV read grows with context while TTT traffic is fixed, there is a **crossover context S\***
below which the KV cache is the cheaper memory system and above which TTT is:
**S\*_read ≈ 65k tokens** (KV read = TTT read half) and **S\*_rmw ≈ 131k** (KV read = full TTT RMW)
at the anchor. S\* scales `∝ m·d²`: **16k → 1.05M tokens** across 340M → 70B; fp8 KV doubles it,
MHA collapses it. Corollary (serving sanity check): dense-attention KV per-token latency **rises**
with context (0.79 → 2.58 ms from 2k → 32k), while TTT per-token latency is **context-independent**.
→ **Part III introduction (why decode-state is a NEW opportunity), ch01 Rosetta-Stone follow-up.**

### 3. Batch scaling: BW-bound long before capacity-bound (E1.3, E3)
For latency-sensitive decode, aggregate state RMW **saturates HBM bandwidth far before HBM
capacity** — the opposite of the KV cache, which is capacity-bound. At a 10 ms/token target,
B_max is **20.8 / 5.2 / 0.97 sequences** for 340M / 1.3B / 7B — a bandwidth wall, not a memory wall.
Critically, **on-die L2 (50 MB) cannot hold even one sequence's whole-model state at any scale**, so
TTT state **cannot be pinned on-die whole-model**; residency must be per-layer / streamed. This
directly motivates E1.2. → **Part III ch23 serving projection.**

### 4. State placement is a *designable* choice (E1.2) — Figure (b)
Because TTT state size is **fixed by (d, m, L) and independent of context**, on-chip residency is a
real design knob (unlike context-growing KV). For a resident 134 MB per-layer RMW, a 268 MB
scratchpad beats HBM3 by **6.8× energy / 14.9× time** — *when the state fits*. The **residency
crossover**: a 268 MB on-chip buffer holds one whole layer's state up to **d=2048 (1.3B, 134 MB)**
and **spills by d=4096 (7B, 537 MB)**; crossover width **d\*≈2896**. Whole-model residency is even
tighter — only **Titans-170M (226 MB total)** fits, so at scale placement is **per-layer / streamed,
not pinned**. For the write-heavy elementwise epilogue, PIM is a **1.6× energy win** at TTT's ~3
MAC/elem (and *both* an energy and latency win at rank-1-ish 1 MAC/elem); its latency cost only bites
past ~3 MAC/elem. → **Part III ch20 HW bottleneck, ch24 proposals, ch10 cost-model sheet.**

### 5. Update-frequency continuum = memory-hierarchy spec (E1.4) — Figure (d)
Nested-Learning (HOPE) / Sleep give each memory level an **update cadence**; that cadence translates
**directly** into a tier assignment. The gating rule: *a resident copy is pinned by the **faster** of
its read/write cadence.* So a per-token fast-weight level lands on **HBM3**; a genuinely down-sampled
CMS-mid level (read+write every 4096 tokens) legally drops to **CXL**; Sleep-consolidated experts
(offline cadence) also to CXL — because sub-per-token access **amortises** the traffic below the
critical path. A CMS level that is *queried every token but written per chunk* stays pinned to HBM by
its read cadence. Per-**layer** fast-weight blocks (67 MB) fit the scratchpad for a **7.4× energy
win**; all-layer does not. → **Part III ch19/ch24, the "cadence → hierarchy" section, ch16/ch17.**

### 6. KV-manager semantics are a category mismatch for RMW state (E1.5)
The software-stack argument, grounded in `hat-kv-manager/manager.py`:
- **G1 — reuse = 0.** Content-addressed prefix reuse (the manager's #1 value, 0.5 hit-rate for KV) is
  **structurally 0** for RMW state: content mutates every token, no block hash recurs.
- **G2 — stale accretion 256×.** Append-only `place()` with no in-place update accretes T stale
  versions → **256× blow-up** of the true 1-state working set over 256 tokens.
- **G3 — unpriced dirty writeback.** `_evict_from()` `del`-drops dirty state for free (clean/
  recomputable assumption); a real TTT eviction **owes a full writeback** the manager prices as **0**
  (≈3.05 mJ / 130 µs owed at the anchor).

A TTT-state manager needs new event types absent from KVCacheManager: `update_in_place`,
`mark_dirty/writeback`, `checkpoint/rollback`, `bind_to_sequence`, `free_on_update`.
→ **Part III pair-thesis software paragraph, ch10 systems bridge; roadmap memo for hat-kv-manager.**

---

## Training/prefill half of the thesis

### 7. Chunk C is the roofline x-axis (E2.1) — Figure (c)
On a commodity CPU roofline, the DeltaNet-style chunkwise scan reproduces the **shape** of the
corrected `AI(C)` curve: **C=1 (per-token rank-1 RMW = the TTT decode regime)** sits at AI≈1 FLOP/B,
deep in the memory-bound plateau (~12% of host roofline); growing C climbs the roofline and crosses
into compute-bound at a measured **C\*≈32** on the host. The same knob that makes decode
memory-bound makes prefill/training compute-bound — **one algorithm, two regimes**. The *value* of
C\* is ridge-dependent (H100 closed-form C\* ≈ 306–430, E3); only the **curve shape** transports
(host ridge 34 FLOP/B ≈ 9× below H100's 295). → **Part III prefill/training half, ch09 §chunk-size,
ch15 TNT.**

### 8. On-die/off-die RMW bandwidth cliff (E2.2) — Figure (e)
The local, *measured* analogue of the E1.2 fit→spill: an in-cache RMW runs at **~265 GB/s**;
overrun the L3 capacity boundary and effective bandwidth **collapses ~3.9×** to the **~68 GB/s**
DRAM plateau. And RMW sustains only **~0.5×** the element throughput of a read at saturated DRAM —
it moves 2× the bytes per element. That factor-of-two is **the write-back tax that append-once KV
avoids**, measured directly. Shape only; do not transport GB/s to H100. → **Part III ch20 / state-
placement section, ch10 (hierarchy is discontinuous at capacity boundaries).**

---

## Cross-validation and carry-forward

**Three-way agreement (<1%)** on the anchor across E3 (closed form), E1.1/E1.3 (hatir), and the plan
hand-calc: `rmw_GB_token` 6.442/6.4425/6.4, `rmw_ms_token` 1.923/1.9231/1.9, `S*_read`
65536/65471/65000, `state_MB_layer` 134.2/134.218/134. This is what lets us promote the ratios to
text with confidence while withholding the absolutes.

**Carried forward to the in-house A100 runbook** (explicitly *not* claimed here):
1. Absolute per-token decode wall-clock (ms/token, tok/s) at every scale — papers publish none.
2. Achieved-vs-roofline efficiency of the RMW step (kernel-launch / scheduling / tail).
3. Real r/w-asymmetric DRAM energy/token (twin uses a symmetric 7 pJ/B).
4. S\* validation with a *real* KV kernel vs a *real* TTT-state kernel head-to-head.
5. Novel-twin (scratchpad/PIM) numbers stay directional until silicon.
6. B_max under a full serving stack (weights + activations + KV co-resident) — E3 B_max is an upper bound.

---

## Figure index (repro: `figures/render_experiments.py`)

| file | supports | source |
|---|---|---|
| `exp-a-kv-ttt-crossover.png` | KV vs TTT traffic crossover S\* + S\* scaling | E1.3 / E3 |
| `exp-b-state-placement.png` | device energy/latency tradeoff + residency crossover | E1.2 |
| `exp-c-chunk-roofline.png` | AI(C) on the host roofline; memory→compute at C\* | E2.1 |
| `exp-d-frequency-tiers.png` | update-cadence → memory-tier admissibility table | E1.4 |
| `exp-e-rmw-cliff.png` | on-die/off-die RMW bandwidth cliff (E1.2 analogue) | E2.2 |
