#!/usr/bin/env python3
"""Consolidate all Part-III local experiments (E3, E1.1-E1.5, E2.1-E2.2) into a
single machine-readable warrant source for veridraft P1 quantitative claims.

Every headline number carries: value, unit, source experiment id, and a caveat tag.
Run: /home/jimmy/repos/neural-memory-study/.venv/bin/python consolidate.py
Output: /home/jimmy/repos/neural-memory-study/experiments/results.json
"""
import json, os, datetime

ROOT = "/home/jimmy/repos/neural-memory-study/experiments"

def load(exp):
    with open(os.path.join(ROOT, exp, "results.json")) as f:
        return json.load(f)

E3   = load("E3-analytical")
E11  = load("E1.1-decode-baseline")
E12  = load("E1.2-state-placement")
E13  = load("E1.3-kv-vs-ttt")
E14  = load("E1.4-frequency-tiers")
E15  = load("E1.5-kvmgr-gap")
E21  = load("E2.1-chunk-intensity")
E22  = load("E2.2-rmw-cliff")

# ---- shared caveat vocabulary (tags referenced by every number) --------------
CAVEAT_TAGS = {
    "REL":  "exploration-grade / RELATIVE: hatir + hat-schema twins are pre-silicon "
            "analytic cost models (repo self-declares ranking+crossover, not "
            "silicon-accurate absolutes); only ratios and crossover POSITIONS are promoted to text.",
    "NO-WALLCLOCK": "the 6 source papers do NOT publish H100 wall-clock decode; absolute "
            "us/token & mJ/token are roofline lower bounds, externally unverifiable, "
            "deferred to the in-house A100 runbook (Part III-a).",
    "PER-KERNEL": "per-kernel roofline only: no kernel-launch / scheduling / decode "
            "tail-effect overheads; per-token latency is a lower bound.",
    "NOVEL-SIM-FALSE": "novel twins (novel-swscratchpad-npu, novel-pim-cim) are provenance="
            "analytical -> verify committee marks simulation_ready=False / PPA-abstain; "
            "PIM & scratchpad results are DIRECTIONAL DSE only, never a shipping-device claim.",
    "CPU-SHAPE": "CPU micro-bench validates the SHAPE (how the quantity scales) and the "
            "memory<->compute crossover ONLY; it does NOT transport to H100 absolutes "
            "(host ridge ~34 FLOP/B sits ~9x below the H100 twin ridge 295 FLOP/B; "
            "host cache cliff ~100x off the H100 on/off-die ridge).",
    "M-ASSUMED": "state multipliers m (2-layer MLP=8d^2, +momentum=16d^2) and (d,L) per "
            "param scale are canonical transformer shapes from published architecture "
            "descriptions, not released decode traces.",
    "RESIDENCY-3RD-CLASS": "residency.py excludes PERSISTENT tensors from the transient set; "
            "TTT state is a 3rd class (persistent BUT mutated every token) so it is tagged "
            "activation-class to force the spill sim; eviction-policy precision not claimed.",
}

def num(value, unit, src, tags, note=None):
    d = {"value": value, "unit": unit, "source": src, "caveat_tags": tags}
    if note:
        d["note"] = note
    return d

anchor = "neural-mem-1.3B (d=2048, m=16, L=24, GQA-8 kv_dim=1024, bf16)"

consolidated = {
  "meta": {
    "title": "Part III consolidated experimental warrant source (TTT/neural-memory decode = memory-centric)",
    "thesis": "D4 workload-split pair thesis: decode/serving state management is a memory-centric "
              "opportunity (per-token RMW of the whole fast-weight state; unshared, write-heavy, "
              "not content-addressable -> qualitatively unlike append-once/read-many KV cache); "
              "training/prefill is the accelerator regime (chunk C is the roofline x-axis).",
    "generated": datetime.date.today().isoformat(),
    "grade": "EXPLORATION-GRADE. Load-bearing claims = ratios, crossover positions, tier orderings, "
             "bound classifications. Absolute wall-clock deferred to in-house A100 runbook (Part III-a).",
    "anchor_config": anchor,
    "hw_twin": {
      "provenance": "dgx_h100_x4.json HBM level (hat-schema SoT)",
      "bw_hbm_bps": E3["hw"]["bw_hbm_bps"],
      "epb_hbm_pj_per_byte": E3["hw"]["epb_hbm"],
      "peak_flops_s": E3["hw"]["peak_flops_s"],
      "ridge_flop_per_byte": E3["hw"]["ridge_flop_per_byte"],
      "hbm_cap_bytes": E3["hw"]["hbm_cap_bytes"],
      "ondie_l2_bytes": E3["hw"]["ondie_bytes"],
    },
    "caveat_tag_glossary": CAVEAT_TAGS,
    "source_experiments": {
      "E3-analytical":       "closed-form cost model + crossover calculator (S*, C*, B_max) over 6 papers x 5 model scales",
      "E1.1-decode-baseline":"hatir decode-step baseline on DGX-H100 twin: traffic/time/energy/bound",
      "E1.2-state-placement":"hatir state-placement DSE: HBM vs SRAM vs scratchpad vs PIM, residency crossover, spill",
      "E1.3-kv-vs-ttt":      "hatir KV-vs-TTT contrast: S* crossover, B_max, read/write traffic decomposition",
      "E1.4-frequency-tiers":"hatir frequency-tiered placement: update cadence -> memory-hierarchy tier admissibility",
      "E1.5-kvmgr-gap":      "hat-kv-manager semantic-gap analysis: prefix-reuse=0, stale accretion, unpriced dirty writeback",
      "E2.1-chunk-intensity":"CPU numpy micro-bench: AI(C) curve shape + memory->compute crossover on host roofline",
      "E2.2-rmw-cliff":      "CPU numpy micro-bench: RMW bandwidth cliff at the on-die/off-die (L3->DRAM) capacity boundary",
    },
  },

  # ========================================================================
  # CLAIM 1: TTT decode is a memory-bound whole-state RMW (baseline structure)
  # ========================================================================
  "claim1_decode_is_memorybound_rmw": {
    "supports": "pair thesis, decode half: 'decode = per-token RMW of the entire fast-weight state'",
    "anchor": anchor,
    "state_MB_per_layer":     num(134.2, "MB", "E1.1/E3", ["REL","M-ASSUMED"]),
    "rmw_GB_per_token":       num(6.44, "GB/token", "E1.1/E3/E1.3", ["REL","M-ASSUMED"],
                                  "read m*d^2 + write m*d^2 over L layers; 3-way agree (E1.1=6.4425, E3=6.442, E1.3=6.4425)"),
    "decode_ms_per_token":    num(1.92, "ms/token", "E1.1/E3", ["REL","NO-WALLCLOCK","PER-KERNEL"],
                                  "state-movement lower bound at HBM3 3.35 TB/s"),
    "energy_mJ_per_token":    num(46.5, "mJ/token", "E1.1", ["REL","NO-WALLCLOCK"],
                                  "HBM 7 pJ/B symmetric + epilogue MACs"),
    "arith_intensity_flopB":  num(0.59, "FLOP/byte", "E3", ["REL"],
                                  "<< ridge 295 FLOP/B => deterministically memory-bound (E1.1 gemv-inclusive AI=2.25 also << ridge)"),
    "bound":                  num("memory", "class", "E1.1/E3/E1.3", ["REL"]),
    "memory_over_gemv_compute_ratio": num(393.8, "x", "E1.1", ["REL"],
                                  "state RMW dominates the GEMV compute of a decode step by ~394x => step cost IS the state traffic"),
    "scaling_rmw_GB_per_token": {
      "note": "RMW traffic grows with model width d^2*m*L; independent of context S (the KV contrast).",
      "unit": "GB/token", "source": "E3", "caveat_tags": ["REL","M-ASSUMED"],
      "Titans-170M": 0.453, "Titans-340M": 1.611, "Titans-760M": 3.624,
      "neural-mem-1.3B": 6.442, "hypo-7B": 34.36, "hypo-70B": 343.6,
    },
  },

  # ========================================================================
  # CLAIM 2: KV vs TTT are qualitatively different memory workloads (S* crossover)
  # ========================================================================
  "claim2_kv_vs_ttt_crossover": {
    "supports": "pair thesis core: why decode-state management is a NEW memory opportunity, not KV cache",
    "anchor": anchor,
    "traffic_decomposition_write_pct_of_read": {
      "kv":  num(0.003, "%", "E1.3", ["REL"], "KV write (append) is 0.003% of read => append-once/read-many"),
      "ttt": num(100.0, "%", "E1.3", ["REL"], "TTT write == read (equal halves) => symmetric RMW, always dirty"),
    },
    "S_star_read_tokens": num(65471, "tokens", "E1.3", ["REL","M-ASSUMED"],
        "context at which KV read traffic == TTT read half; below S* KV is cheaper, above S* TTT is cheaper. "
        "E3 closed form 65536, hatir 65471 (agree)."),
    "S_star_rmw_tokens":  num(130942, "tokens", "E1.3", ["REL","M-ASSUMED"],
        "context at which KV read == full TTT RMW (read+write); the honest upper crossover."),
    "S_star_scaling": {
      "unit": "tokens (read-crossover, GQA-8 bf16)", "source": "E3/E1.3", "caveat_tags": ["REL","M-ASSUMED"],
      "formula": "S* = m*d^2*(ttt_dtype/kv_dtype)/kv_dim",
      "Titans-340M": 16384, "Titans-760M": 36864, "neural-mem-1.3B": 65536,
      "hypo-7B": 262144, "hypo-70B": 1048576,
      "fp8_KV_doubles_S_star": True,
      "MHA_collapses_S_star": "kv_dim=d shrinks S* linearly (see E3 sweeps.S_star.kv_dim_d_MHA)",
    },
    "kv_latency_grows_with_context": {
      "source": "E1.3", "caveat_tags": ["REL","PER-KERNEL"],
      "note": "dense-attention KV decode per-token latency RISES with context (read-many wall); "
              "TTT per-token latency is context-INDEPENDENT (constant RMW).",
      "kv_ms_per_token": {"S_2k": 0.785, "S_32k": 2.581},  # serving_sanity
      "ttt_ms_per_token_const": 1.9231,
    },
  },

  # ========================================================================
  # CLAIM 3: batch scaling -- decode-state is BW-bound long before capacity-bound
  # ========================================================================
  "claim3_batch_scaling_Bmax": {
    "supports": "pair thesis: serving economics of decode-state (BW binds, opposite of capacity-bound KV cache)",
    "binding_resource": num("BW", "class", "E1.3/E3", ["REL"],
        "at every model scale B_max is BW-bound at 10ms, far below the HBM-capacity batch => "
        "opposite of KV cache which is capacity-bound"),
    "ondie_holds_one_seq": num(False, "bool", "E1.3/E3", ["REL"],
        "on-die L2 (50MB) cannot hold even ONE sequence's whole-model state at any scale => "
        "TTT state cannot be pinned on-die whole-model; residency must be per-layer/streamed (motivates E1.2)"),
    "B_max_bw_10ms": {
      "unit": "sequences", "source": "E3/E1.3", "caveat_tags": ["REL","NO-WALLCLOCK"],
      "Titans-340M": 20.8, "Titans-760M": 9.24, "neural-mem-1.3B": 5.2, "hypo-7B": 0.97, "hypo-70B": 0.1,
    },
    "B_max_bw_50ms": {
      "unit": "sequences", "source": "E3/E1.3", "caveat_tags": ["REL","NO-WALLCLOCK"],
      "Titans-340M": 104.0, "Titans-760M": 46.2, "neural-mem-1.3B": 26.0, "hypo-7B": 4.87, "hypo-70B": 0.49,
    },
  },

  # ========================================================================
  # CLAIM 4: state placement -- on-chip residency is a DESIGNABLE choice (fixed size)
  # ========================================================================
  "claim4_state_placement_dse": {
    "supports": "memory-centric opportunity: TTT state size is context-INDEPENDENT (fixed by d,m,L) so "
                "on-chip residency is a designable knob, unlike context-growing KV",
    "device_ranking_134MB_layer_RMW": {
      "note": "flat single-tier device twins so traffic (2*S) is byte-comparable; state=134MB/layer (d=2048,m=16)",
      "source": "E1.2", "caveat_tags": ["REL","NOVEL-SIM-FALSE"],
      "HBM3_3.35TBps_7pJ":      {"energy_uJ": 1919.3, "time_us": 80.13, "bound": "memory"},
      "ondie_SRAM_10TBps_0.4pJ":{"energy_uJ": 147.6,  "time_us": 26.84, "bound": "memory", "fits_one_layer": False},
      "sw_scratchpad_60TBps_0.9pJ":{"energy_uJ": 281.9, "time_us": 5.37, "bound": "compute", "fits_one_layer": True},
      "PIM_inbank_8.2TBps_3.9pJ":{"energy_uJ": 1167.7, "time_us": 167.8, "bound": "compute"},
    },
    "scratchpad_vs_hbm_when_resident": {
      "energy_win_x": num(6.8, "x", "E1.2", ["REL","NOVEL-SIM-FALSE"], "268MB scratchpad vs HBM3 for a resident 134MB RMW"),
      "time_win_x":   num(14.9, "x", "E1.2", ["REL","NOVEL-SIM-FALSE"]),
    },
    "residency_crossover": {
      "last_config_fits_one_layer_in_268MB": num("neural-mem-1.3B (134MB/layer, d=2048)", "config", "E1.2", ["REL","NOVEL-SIM-FALSE"]),
      "first_config_spills":                 num("hypo-7B (537MB/layer, d=4096)", "config", "E1.2", ["REL","NOVEL-SIM-FALSE"]),
      "crossover_width_d_star":              num(2896, "hidden dim d", "E1.2", ["REL","NOVEL-SIM-FALSE"]),
      "whole_model_fits_268MB_scratchpad":   num("only Titans-170M (L=12, 226MB total)", "config", "E1.2", ["REL","NOVEL-SIM-FALSE","RESIDENCY-3RD-CLASS"],
          "every config >=340M overruns and spills per token => at scale placement is per-layer/streamed, not pinned whole-model"),
    },
    "pim_for_write_heavy_epilogue": {
      "note": "PIM wins energy on the write-heavy elementwise epilogue while its intensity stays low",
      "source": "E1.2", "caveat_tags": ["REL","NOVEL-SIM-FALSE"],
      "at_ttt_3_macs_per_elem": {"energy_win_x": 1.64, "time_cost_x": 2.09},
      "pim_time_crossover_macs_per_elem": 3,
      "note2": "PIM latency only bites past ~3 MAC/elem (compute-bound epilogue); at rank-1-ish 1 MAC/elem it is 1.74x energy AND 0.7x time",
    },
  },

  # ========================================================================
  # CLAIM 5: update-frequency continuum == memory-hierarchy spec (E1.4)
  # ========================================================================
  "claim5_frequency_tiered_placement": {
    "supports": "Nested-Learning(HOPE)/Sleep update-cadence continuum translates DIRECTLY to a tier assignment; "
                "sub-per-token access cadence is what lets a write-heavy RMW state legally drop to DDR/CXL",
    "anchor": "neural-mem-1.3B, per-token critical-path budget t_tok = 961.6 us (all-layer RMW on HBM3)",
    "gating_rule": num("resident copy gated by max(read-cadence, write-cadence) i.e. the FASTER access",
                       "rule", "E1.4", ["REL","NOVEL-SIM-FALSE"]),
    "placement_recommendation": {
      "source": "E1.4", "caveat_tags": ["REL","NOVEL-SIM-FALSE","M-ASSUMED"],
      "fast-weights (read+write every token)":       {"tier": "HBM3", "reason": "read-gated to fastest admissible resident tier"},
      "CMS-chunk (read/token, write/64 tokens)":      {"tier": "HBM3", "reason": "read cadence pins it hot despite chunked write"},
      "CMS-mid (read+write /4096 tokens)":            {"tier": "CXL",  "reason": "down-sampled cadence => cheapest tier still admissible"},
      "sleep-experts (read+write /262144 tokens)":    {"tier": "CXL",  "reason": "offline/idle-window cadence"},
      "frozen-weights (read/token, never written)":   {"tier": "HBM3", "reason": "read-gated, read-heavy side (accelerator-friendly)"},
    },
    "latency_tolerance_widens_with_cadence": num("P_read * t_tok (linear in update period)", "rule", "E1.4", ["REL"]),
    "per_layer_scratchpad_energy_win_x": num(7.42, "x", "E1.4", ["REL","NOVEL-SIM-FALSE"],
        "per-LAYER fast-weight block (67MB) fits the 268MB scratchpad: 7.4x lower energy than HBM3; all-layer does NOT fit"),
  },

  # ========================================================================
  # CLAIM 6: KV-manager semantics are a category mismatch for RMW state (E1.5)
  # ========================================================================
  "claim6_kvmanager_semantic_gap": {
    "supports": "software-stack half of the pair thesis: existing KV-cache manager cannot manage TTT state",
    "G1_prefix_reuse_hit_rate": {
      "kv": num(0.5, "hit-rate", "E1.5", ["REL"]),
      "ttt": num(0.0, "hit-rate", "E1.5", ["REL"],
          "content-addressed reuse (the manager's #1 value) is structurally ZERO for RMW state: content changes every token, no block hash recurs"),
    },
    "G2_stale_accretion_blowup_x": num(256.0, "x (== T)", "E1.5", ["REL"],
        "append-only place() with no in-place update accretes T=256 stale versions => 256x the true 1-state working set"),
    "G3_unpriced_dirty_writeback": {
      "mgr_charged_writeback": num(0.0, "uJ", "E1.5", ["REL"], "_evict_from() del-drops dirty state for free (clean/recomputable assumption)"),
      "owed_writeback_energy_uJ": num(3053.5, "uJ", "E1.5", ["REL","NO-WALLCLOCK"], "dirty TTT state owes a full writeback the manager prices as ZERO"),
      "owed_writeback_time_us":   num(130.2, "us", "E1.5", ["REL","NO-WALLCLOCK"]),
    },
    "missing_api_for_ttt_state": num(
        ["update_in_place(slot)", "mark_dirty/writeback(slot)", "checkpoint/rollback(seq)",
         "bind_to_sequence(seq)", "free_on_update(slot)"],
        "api list", "E1.5", ["REL"],
        "event types a TTT-state manager needs that KVCacheManager lacks"),
  },

  # ========================================================================
  # CLAIM 7: chunk C is the roofline x-axis (prefill/training half) -- CPU shape (E2.1)
  # ========================================================================
  "claim7_chunk_is_roofline_axis": {
    "supports": "pair thesis, training/prefill half: growing chunk C moves the SAME algorithm from "
                "memory-bound (C=1 sequential decode regime) to compute-bound (accelerator regime)",
    "host_roofline": {
      "source": "E2.1", "caveat_tags": ["CPU-SHAPE"],
      "peak_gflops": 919.3, "peak_gbps": 26.69, "ridge_flop_per_byte": 34.45,
      "h100_ridge_flop_per_byte": 295.4, "host_ridge_gap_factor": 8.57,
    },
    "AI_at_C1_sequential_flopB": num(1.0, "FLOP/byte", "E2.1", ["CPU-SHAPE"],
        "C=1 per-token rank-1 RMW (== TTT decode regime) sits at AI~1 => deep memory-bound plateau, "
        "measured only ~12% of host roofline"),
    "C_star_measured_discrete": num(32, "tokens", "E2.1", ["CPU-SHAPE"],
        "measured memory->compute crossover on host (all d in {512,1024,2048}); continuous ~23-24"),
    "C_star_h100_closedform": {
      "unit": "tokens", "source": "E3", "caveat_tags": ["REL"],
      "note": "on the H100 twin ridge (295 FLOP/B) C* is ~306-430 depending on d; the CPU confirms the SHAPE, not this value",
      "d512": 430.2, "d1024": 374.5, "d2048": 337.1, "d4096": 316.6, "d8192": 306.0,
    },
    "measured_gflops_vs_C_d2048": {
      "unit": "GFLOP/s measured wall-clock", "source": "E2.1", "caveat_tags": ["CPU-SHAPE"],
      "C1": 4.56, "C8": 51.1, "C32": 149.9, "C128": 391.3, "C512": 689.5, "C1024": 675.4,
    },
  },

  # ========================================================================
  # CLAIM 8: on-die/off-die RMW bandwidth cliff -- CPU shape analogue of E1.2 (E2.2)
  # ========================================================================
  "claim8_rmw_bandwidth_cliff": {
    "supports": "state placement: an in-cache (on-die) RMW is cheap; overrun the capacity boundary and "
                "effective BW collapses -- the local analogue of the E1.2 scratchpad fit->spill cliff",
    "host_cache": {"L2_per_core_MB": 1.0, "L3_shared_MB": 16.0},
    "incache_peak_gbps": num(264.8, "GB/s", "E2.2", ["CPU-SHAPE"], "1-thread RMW, in-cache"),
    "dram_gbps":         num(67.9, "GB/s", "E2.2", ["CPU-SHAPE"], "1-thread RMW, spilled to DRAM"),
    "l3_to_dram_cliff_x": num(3.90, "x", "E2.2", ["CPU-SHAPE"],
        "on-die vs off-die RMW bandwidth ratio at the L3 capacity boundary (the robust, overhead-free headline)"),
    "readonly_over_rmw_at_dram": num(0.50, "x", "E2.2", ["CPU-SHAPE"],
        "RMW moves 2x the bytes/element of a read => ~half the effective element throughput at saturated DRAM; "
        "quantifies the write-back tax that append-once KV avoids"),
  },

  # ========================================================================
  # Cross-validation ledger (3-way agreement E1.1 x E3 x E1.3)
  # ========================================================================
  "cross_validation": {
    "note": "closed-form (E3), hatir tool (E1.1/E1.3), and plan hand-calc agree on the anchor to <1%",
    "anchor": anchor,
    "rmw_GB_token":   {"E3": 6.442, "E1.1": 6.4425, "E1.3": 6.4425, "plan": 6.4, "agree": True},
    "rmw_ms_token":   {"E3": 1.923, "E1.1": 1.9231, "E1.3": 1.9231, "plan": 1.9, "agree": True},
    "S_star_read":    {"E3": 65536, "E1.3": 65471, "plan": 65000, "agree": True},
    "state_MB_layer": {"E3": 134.2, "E1.1": 134.218, "E1.5": 134.2, "plan": 134, "agree": True},
    "B_max_bw_10ms":  {"E3": 5.2, "E1.3": 5.2, "agree": True},
    "C_star_shape":   {"E2.1_cpu_discrete": 32, "E3_h100_d2048": 337.1,
                       "note": "same curve shape, ridge-dependent value; CPU ridge 9x below H100"},
  },

  # ========================================================================
  # Carry-forward to the in-house A100 runbook (what stays a ratio here)
  # ========================================================================
  "carry_forward_to_A100_runbook": [
    "Absolute per-token decode wall-clock (ms/token, tokens/s) at every model scale: the papers publish "
    "none, so E1.1/E3 give only a roofline lower bound. Measure real decode latency + achieved HBM BW on A100.",
    "Achieved-vs-roofline efficiency of the RMW step (kernel-launch, scheduling, tail): E1.1 is per-kernel only.",
    "Real energy/token: HBM twin uses a symmetric 7 pJ/B; measure actual r/w-asymmetric DRAM energy on A100.",
    "S* validation with a REAL KV read kernel + a REAL TTT state kernel head-to-head (E1.3 uses roofline halves).",
    "Novel-twin (scratchpad/PIM) numbers stay DIRECTIONAL until silicon: E1.2/E1.4 scratchpad wins are not device claims.",
    "B_max under a full serving stack (weights + activations + KV co-resident): E3 B_max_cap is an upper bound.",
  ],

  "global_caveats": [
    CAVEAT_TAGS["REL"],
    CAVEAT_TAGS["NO-WALLCLOCK"],
    CAVEAT_TAGS["PER-KERNEL"],
    CAVEAT_TAGS["NOVEL-SIM-FALSE"],
    CAVEAT_TAGS["CPU-SHAPE"],
    CAVEAT_TAGS["M-ASSUMED"],
    CAVEAT_TAGS["RESIDENCY-3RD-CLASS"],
  ],
}

out = os.path.join(ROOT, "results.json")
with open(out, "w") as f:
    json.dump(consolidated, f, indent=2)
print("wrote", out)
print("claims:", [k for k in consolidated if k.startswith("claim")])
print("bytes:", os.path.getsize(out))
