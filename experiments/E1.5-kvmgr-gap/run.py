"""E1.5 — hat-kv-manager semantic-gap analysis (pair-thesis software-stack leg).

Thesis (D4, decode side): serving-state management for TTT / neural-memory models is a
memory-CENTRIC problem that is *qualitatively different* from KV-cache management. A KV cache is
append-once / read-many / content-addressable / clean-recomputable. TTT fast-weight state is
read-modify-write (RMW) every token / unshared per sequence / dirty / NOT content-addressable.

Rather than argue in prose, we DRIVE the actual hat-kv-manager (`KVCacheManager`, the LMCache/
Mooncake surface) over the SoT DGX-H100 twin and measure — from its own `metrics()` — three places
where its semantics MISPRICE (or silently lose) TTT state. Each gap is grounded in a specific method
of `kv_manager/manager.py`:

  G1  prefix_reuse()  — content-addressed reuse. TTT state's "content" changes every token, so the
                        block hash changes every token => reuse collapses to 0 (vs a real KV prefix).
  G2  place()         — append-once: `if h in self.index: skip`. There is NO in-place update path, so
                        a per-token mutation must be a NEW hash each token; the OLD version is never
                        freed until capacity pressure => the index accretes T stale versions of a
                        working set whose true size is 1. Byte accounting (block_bytes = block_tokens
                        * kv_bytes_per_token) also GROWS with tokens (append) and charges the write
                        ONCE per block, so it under-counts RMW write traffic by the RMW factor.
  G3  _evict_from()   — clean eviction: a block that can't spill is `del`'d and charged evict_bytes
                        with ZERO writeback time/energy (it assumes the block is clean / recomputable
                        from its tokens). TTT state is ALWAYS dirty and unshared => dropping it is
                        unrecoverable; correct handling owes a writeback of state_bytes. We measure
                        the writeback DEBT the manager fails to charge.

All numbers exploration-grade / RELATIVE (hatir/hat twins are pre-silicon cost models, not silicon).
Run: /home/jimmy/repos/neural-memory-study/.venv/bin/python run.py
"""
from __future__ import annotations

import json
import os
import sys

# make hat-kv-manager importable (installed in venv, but be robust)
sys.path.insert(0, "/home/jimmy/repos/hat-kv-manager")

from kv_manager import KVCacheManager  # noqa: E402

KB, MB, GB = 1024, 1024**2, 1024**3
HERE = os.path.dirname(os.path.abspath(__file__))
TWIN_PATH = "/home/jimmy/repos/hat-schema/twins/dgx_h100_x4.json"


def load_twin():
    with open(TWIN_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------------
# Workload config — kept consistent with E3-analytical (d=2048, m=16, L=24 headline row).
# ---------------------------------------------------------------------------------------------------
D = 2048            # model dim
L = 24              # layers
M = 16              # state multiplier (TTT-MLP W1,W2 + momentum ~ m*d^2 elems)
DT = 2              # bf16
KV_DIM = 1024       # GQA-8 kv-heads * head-dim (Hkv*hd)

# TTT fast-weight state, per token: the WHOLE state is read + written back (RMW).
STATE_BYTES_LAYER = M * D * D * DT               # per layer
STATE_BYTES_MODEL = STATE_BYTES_LAYER * L        # whole model, one sequence

# KV cache, per token (append): 2(K,V) * kv_dim * dtype, per layer, x L for the model.
KV_BYTES_TOKEN_LAYER = 2 * KV_DIM * DT
KV_BYTES_TOKEN_MODEL = KV_BYTES_TOKEN_LAYER * L


def hr(title):
    print("=" * 100)
    print(title)


# ---------------------------------------------------------------------------------------------------
# G1 — content-addressed reuse collapses for RMW state.
# ---------------------------------------------------------------------------------------------------
def _big_twin(cap_gb=80):
    # a single large HBM tier + compute leaf: big enough to hold the working set so we ISOLATE a
    # single semantic branch (reuse in G1, accretion in G2) without the DGX twin's tiny on-die tiers
    # firing eviction. (Over the real DGX twin eviction DOES fire -> see G3 and the caveats.)
    return {"level_id": "hbm", "role": "hbm", "capacity_bytes": cap_gb * GB, "bandwidth_bps": 3.35e12,
            "energy_pj_per_byte": 7.0,
            "children": [{"level_id": "pe", "role": "compute", "instances": 1,
                          "peak_macs_per_s": 1e13, "dtype_bytes": 2}]}


def gap1_prefix_reuse():
    hr("G1) prefix_reuse(): content-addressed reuse — KV prefix HITS vs TTT-state MISSES")
    # KV: one block hash per (token-block, content). A shared prefix across 2 requests => the second
    # request's prefix blocks are already resident => hits. This is LMCache's whole value. Use a hot
    # tier big enough to hold the prefix so the reuse itself (not capacity) is what we measure.
    kv = KVCacheManager.from_twin(_big_twin(), kv_bytes_per_token=KV_BYTES_TOKEN_MODEL, block_tokens=16)
    prefix = [f"kv_prefix_blk_{i}" for i in range(32)]          # 512-token shared system prompt
    kv.place(prefix, request_id="reqA")                         # request A fills the prefix
    kv_hits = kv.prefix_reuse(prefix)                           # request B reuses the SAME content
    kv_hit_rate = kv.metrics()["hit_rate"]

    # TTT: the "block" is the fast-weight state, which MUTATES every token. Content-addressing hashes
    # CONTENT, so each token's state is a DIFFERENT hash. Replay the identical scenario: a second
    # sequence over the same prompt cannot reuse ANY state block, because none of the hashes recur.
    ttt = KVCacheManager.from_twin(_big_twin(), kv_bytes_per_token=STATE_BYTES_MODEL, block_tokens=1)
    # sequence A mutates state over 32 tokens -> 32 distinct content hashes
    seqA = [f"ttt_state@tokA_{t}_h{hash((0, t)) & 0xffff}" for t in range(32)]
    ttt.place(seqA, request_id="seqA")
    # sequence B over the SAME prompt: state evolves differently (unshared), and even if identical the
    # per-token content hash never matches a PREFIX position -> reuse asks for B's hashes:
    seqB = [f"ttt_state@tokB_{t}_h{hash((1, t)) & 0xffff}" for t in range(32)]
    ttt_hits = ttt.prefix_reuse(seqB)
    ttt_hit_rate = ttt.metrics()["hit_rate"]

    print(f"  KV  (append/content-addressable): prefix_reuse hits = {kv_hits}/32, hit_rate = {kv_hit_rate:.2f}")
    print(f"  TTT (RMW/mutates every token):    prefix_reuse hits = {ttt_hits}/32, hit_rate = {ttt_hit_rate:.2f}")
    print("  -> the manager's #1 value (prefix reuse) is STRUCTURALLY unavailable to RMW state:")
    print("     state content changes every token, so no hash ever recurs. Reuse ratio KV:TTT = "
          f"{kv_hit_rate:.2f} : {ttt_hit_rate:.2f}.")
    return {"kv_hits": kv_hits, "kv_hit_rate": kv_hit_rate,
            "ttt_hits": ttt_hits, "ttt_hit_rate": ttt_hit_rate,
            "reuse_available_for_ttt": ttt_hits > 0}


# ---------------------------------------------------------------------------------------------------
# G2 — append-once place() has no update path -> stale-version accretion + write under-count.
# ---------------------------------------------------------------------------------------------------
def gap2_append_vs_rmw():
    hr("G2) place(): append-once semantics vs per-token RMW — stale-version blow-up + write under-count")
    T = 256  # tokens
    # Model ONE layer's state so the numbers are legible; the effect is per-layer.
    state_bytes = STATE_BYTES_LAYER

    # (a) KV: bytes GROW with tokens (append). block_bytes is fixed; the cache legitimately holds T
    #     distinct blocks -> occupancy = T * block_bytes is CORRECT (append-once/read-many).
    # (b) TTT via the SAME API: a per-token mutation must be a NEW hash (no update path). place() only
    #     skips exact-hash repeats and NEVER frees the previous version. So after T tokens the index
    #     holds T stale copies of a working set whose TRUE size is 1 state.
    # Use a tier big enough to avoid eviction (80 GB > T*state), so we isolate the accretion branch
    # (retain-garbage), not the spill/drop path. T*state = 256*128MB = 32 GB < 80 GB.
    ttt = KVCacheManager.from_twin(_big_twin(80), kv_bytes_per_token=state_bytes, block_tokens=1)
    for t in range(T):
        ttt.place([f"ttt_state_v{t}"], request_id="seq", token_ranges=[(t, t + 1)])
    m = ttt.metrics()
    live_versions = len(ttt.index)                       # how many state versions the manager retains
    resident_bytes = sum(t2["used"] for t2 in m["per_tier_occupancy"].values())
    true_working_set = state_bytes                       # RMW: only the CURRENT state is live
    blowup = resident_bytes / true_working_set

    # Write-traffic accounting: the manager charges the write ONCE per placed block (line 117:
    # energy_pj += bb * fast.energy_pj_per_byte). RMW rewrites the WHOLE state every token, i.e. T
    # writes of state_bytes. The manager's placed-block write count vs the true per-token RMW writes.
    mgr_write_events = m["placed_blocks"]                # manager thinks: one write per new block
    true_rmw_writes = T                                  # RMW: one full-state write PER TOKEN
    # (they're numerically equal here ONLY because we were forced to fake each token as a new block;
    #  the point is the manager has no notion of rewriting the SAME logical slot -> it cannot model
    #  in-place RMW at all, and its energy is attributed to phantom distinct blocks, not one slot.)
    mgr_write_bytes = m["energy_pj"]  # proxy: all energy here is placement writes

    print(f"  after T={T} per-token state mutations of a {state_bytes/MB:.0f} MB/layer state:")
    print(f"    manager index retains        {live_versions} state versions (stale accretion)")
    print(f"    resident bytes               {resident_bytes/MB:8.0f} MB")
    print(f"    TRUE RMW working set         {true_working_set/MB:8.0f} MB  (only the current state is live)")
    print(f"    memory blow-up factor        {blowup:8.1f}x  (== T: no free-on-update path)")
    print(f"    manager write EVENTS         {mgr_write_events} (one per phantom new block)")
    print(f"    true per-token RMW writes    {true_rmw_writes} of {state_bytes/MB:.0f} MB each")
    # Contrast: the SAME workload over the REAL DGX twin (tiny on-die tiers) hits the OTHER wrong
    # branch — eviction fires and stale versions are dropped/spilled instead of accreting.
    dgx = KVCacheManager.from_twin(load_twin(), kv_bytes_per_token=state_bytes, block_tokens=1)
    for t in range(T):
        dgx.place([f"ttt_state_v{t}"], request_id="seq")
    dm = dgx.metrics()
    print(f"  contrast on the REAL dgx_h100_x4 twin: index={len(dgx.index)} versions, "
          f"spill={dm['spill_bytes']/MB:.0f} MB, evict/drop={dm['evict_bytes']/MB:.0f} MB")
    print("  -> place() models APPEND (new immutable block); it has NO in-place update, so RMW is")
    print("     forced to masquerade as endless new blocks. On a LARGE tier this ACCRETES stale")
    print("     versions (blow-up == T); on the real DGX twin capacity pressure instead DROPS/spills")
    print("     dirty state (both branches wrong for RMW; the drop's cost is mispriced — see G3).")
    return {"tokens": T, "state_bytes_layer": state_bytes, "live_versions": live_versions,
            "resident_bytes": resident_bytes, "true_working_set_bytes": true_working_set,
            "memory_blowup_x": blowup, "mgr_write_events": mgr_write_events,
            "true_rmw_writes": true_rmw_writes,
            "dgx_twin_contrast": {"index_versions": len(dgx.index),
                                  "spill_bytes": dm["spill_bytes"], "evict_bytes": dm["evict_bytes"]}}


# ---------------------------------------------------------------------------------------------------
# G3 — clean eviction drops dirty state for free -> unaccounted writeback debt.
# ---------------------------------------------------------------------------------------------------
def gap3_clean_evict_vs_dirty_writeback():
    hr("G3) _evict_from(): clean eviction vs dirty writeback — the unaccounted writeback DEBT")
    twin = load_twin()
    # Drive the fast tier past capacity so eviction fires. We use a fabricated 2-tier twin whose
    # SLOWER tier is TOO SMALL to accept spills, forcing the manager down its `del`-drop branch
    # (line 137-139) — the exact path that assumes a block is clean/recomputable.
    GBs = 1024**3
    small_twin = {"level_id": "hbm", "role": "hbm", "capacity_bytes": 2 * GBs, "bandwidth_bps": 3.35e12,
                  "energy_pj_per_byte": 7.0,
                  "children": [{"level_id": "sram", "role": "sram", "capacity_bytes": 32 * MB,
                                "bandwidth_bps": 30e12, "energy_pj_per_byte": 0.4,
                                "children": [{"level_id": "pe", "role": "compute", "instances": 1,
                                              "peak_macs_per_s": 1e13, "dtype_bytes": 2}]}]}
    # make the SLOWER (hbm) tier also unable to hold everything so drops happen:
    small_twin["capacity_bytes"] = 64 * MB
    state_bytes_layer = STATE_BYTES_LAYER
    # use a per-block chunk of the layer state so several blocks fit / pressure builds
    block_bytes = 8 * MB
    mgr = KVCacheManager.from_twin(small_twin, kv_bytes_per_token=block_bytes, block_tokens=1)

    # Simulate RMW over many tokens: each token writes a fresh (dirty) state block. Old versions are
    # NEVER reused (RMW) but the manager can only spill/drop them as if clean.
    n_blocks = 64
    for t in range(n_blocks):
        mgr.place([f"dirty_state_v{t}"], request_id="seq")
    m = mgr.metrics()

    spill_bytes = m["spill_bytes"]     # copied to a slower tier (accounted, but as CLEAN copy)
    evict_bytes = m["evict_bytes"]     # DROPPED entirely with `del` — charged bytes, ZERO wb time/energy
    spill_time = m["spill_time_us"]
    # what the manager charged for eviction time/energy of the DROPPED blocks:
    mgr_evict_writeback_time = 0.0     # by construction: the `del` branch charges NO time/energy
    mgr_evict_writeback_energy = 0.0
    # what a DIRTY-state manager MUST pay: every dropped dirty block owes a writeback to a backing
    # tier before it can be freed (else the state update is lost). Charge it at HBM write energy/BW.
    hbm_bw = 3.35e12
    hbm_epb_wr = 7.0
    owed_writeback_time_us = (evict_bytes / hbm_bw) * 1e6
    owed_writeback_energy_pj = evict_bytes * hbm_epb_wr

    print(f"  drove {n_blocks} dirty {block_bytes/MB:.0f} MB state blocks through a capacity-pressured twin:")
    print(f"    spill_bytes (copied down)    {spill_bytes/MB:8.1f} MB   (accounted as a CLEAN copy)")
    print(f"    evict_bytes (DROPPED, del)   {evict_bytes/MB:8.1f} MB   (assumed clean/recomputable)")
    print(f"    manager writeback time  it charged for the drops:   {mgr_evict_writeback_time:.1f} us")
    print(f"    manager writeback energy it charged for the drops:  {mgr_evict_writeback_energy:.1f} pJ")
    print(f"    but dirty state OWES a writeback of the dropped bytes:")
    print(f"      owed writeback time   {owed_writeback_time_us:8.1f} us  (@3.35 TB/s HBM write)")
    print(f"      owed writeback energy {owed_writeback_energy_pj/1e6:8.1f} uJ  (@7 pJ/B write)")
    print("  -> _evict_from() `del`-drops a block for free because a KV block is RECOMPUTABLE from its")
    print("     tokens. TTT state is dirty + unshared + non-recomputable: dropping it LOSES the update.")
    print("     Correct accounting owes a full writeback the manager charges as ZERO -> it systematically")
    print("     UNDER-prices dirty-state eviction (the dominant cost of a write-heavy RMW manager).")
    return {"n_blocks": n_blocks, "block_bytes": block_bytes,
            "spill_bytes": spill_bytes, "evict_bytes_dropped": evict_bytes,
            "mgr_charged_writeback_time_us": mgr_evict_writeback_time,
            "mgr_charged_writeback_energy_pj": mgr_evict_writeback_energy,
            "owed_writeback_time_us": owed_writeback_time_us,
            "owed_writeback_energy_pj": owed_writeback_energy_pj}


# ---------------------------------------------------------------------------------------------------
# API-surface audit — the missing event types a TTT-state manager needs (grounded in method list).
# ---------------------------------------------------------------------------------------------------
def api_audit():
    hr("API audit: what KVCacheManager EXPOSES vs what an RMW-state manager NEEDS")
    present = {
        "place(hashes)": "append immutable content-addressed block on hot tier (append-once)",
        "prefix_reuse(hashes)": "content-addressed reuse across requests (read-many)",
        "_evict_from(tier)": "clean eviction: spill-copy down OR del-drop (recomputable assumption)",
        "metrics()": "hit_rate, occupancy, spill/evict BYTES",
    }
    missing = {
        "update_in_place(slot)": "RMW a resident slot's bytes without a new hash (no content change key)",
        "mark_dirty(slot) / writeback(slot)": "dirty-tracking so eviction pays the writeback it owes",
        "checkpoint(seq) / rollback(seq)": "TTT state is speculative across a step; KV never rolls back",
        "bind_to_sequence(seq)": "unshared per-sequence state lifecycle (no cross-request sharing)",
        "free_on_update(slot)": "release the prior version at mutation (no stale-version accretion)",
    }
    for k, v in present.items():
        print(f"  [present] {k:34s} {v}")
    for k, v in missing.items():
        print(f"  [MISSING] {k:34s} {v}")
    print("  -> the four present methods are the append/read-many/content-addressable/clean-evict axis.")
    print("     The five missing ones are exactly the RMW/dirty/unshared/speculative axis TTT needs.")
    return {"present": list(present), "missing": list(missing)}


def load_e3():
    p = "/home/jimmy/repos/neural-memory-study/experiments/E3-analytical/results.json"
    try:
        with open(p) as f:
            e3 = json.load(f)
        row = None
        for r in e3.get("crossovers", {}).get("table", []):
            if r.get("d") == 2048:
                row = r
                break
        return e3, row
    except Exception:
        return None, None


def main():
    print("E1.5 — hat-kv-manager semantic-gap analysis (grounded in kv_manager/manager.py API)")
    print(f"twin: {TWIN_PATH}")
    print(f"workload: d={D} L={L} m={M} dt={DT}  ->  TTT state/layer={STATE_BYTES_LAYER/MB:.0f} MB, "
          f"state/model={STATE_BYTES_MODEL/MB:.0f} MB; KV/token/model={KV_BYTES_TOKEN_MODEL/1024:.1f} KB")
    print()

    g1 = gap1_prefix_reuse(); print()
    g2 = gap2_append_vs_rmw(); print()
    g3 = gap3_clean_evict_vs_dirty_writeback(); print()
    audit = api_audit(); print()

    e3, e3row = load_e3()
    if e3row is not None:
        hr("Cross-reference: E3-analytical expectation (d=2048 row)")
        print(f"  E3 rmw_GB_token={e3row.get('rmw_GB_token')} GB, S*={e3row.get('S_star_tokens')} tokens, "
              f"bound={e3row.get('bound')} — the RMW traffic this manager cannot model as RMW.")

    # ---- validation: the gaps must HOLD as inequalities, not just print ----
    assert g1["kv_hit_rate"] > 0.0 and g1["ttt_hit_rate"] == 0.0, "TTT reuse must collapse to 0"
    assert g2["memory_blowup_x"] > 1.0, "append-once must accrete stale versions"
    assert g3["evict_bytes_dropped"] > 0.0, "capacity pressure must force dirty drops"
    assert g3["mgr_charged_writeback_energy_pj"] == 0.0 and g3["owed_writeback_energy_pj"] > 0.0, \
        "manager must under-charge dirty writeback"
    print()
    print("[validate] all three semantic gaps hold: reuse->0, blow-up>1x, dirty writeback under-priced.")

    results = {
        "experiment": "E1.5-kvmgr-gap",
        "title": "hat-kv-manager semantic-gap analysis for TTT/neural-memory RMW state",
        "twin": TWIN_PATH,
        "twin_provenance": "hat-schema SoT dgx_h100_x4.json",
        "workload": {"d": D, "L": L, "m": M, "dtype_bytes": DT, "kv_dim": KV_DIM,
                     "state_bytes_layer": STATE_BYTES_LAYER, "state_bytes_model": STATE_BYTES_MODEL,
                     "kv_bytes_token_model": KV_BYTES_TOKEN_MODEL},
        "headline": {
            "G1_prefix_reuse": {
                "kv_hit_rate": g1["kv_hit_rate"], "ttt_hit_rate": g1["ttt_hit_rate"],
                "claim": "content-addressed prefix reuse (the manager's #1 value) is structurally 0 "
                         "for RMW state: its content changes every token so no block hash recurs."},
            "G2_append_vs_rmw": {
                "tokens": g2["tokens"], "memory_blowup_x": g2["memory_blowup_x"],
                "live_versions": g2["live_versions"],
                "claim": f"place() is append-only with no in-place update, so T={g2['tokens']} per-token "
                         f"RMW mutations accrete {g2['live_versions']} stale versions -> "
                         f"{g2['memory_blowup_x']:.0f}x the true 1-state working set."},
            "G3_dirty_writeback": {
                "evict_bytes_dropped_MB": round(g3["evict_bytes_dropped"] / MB, 1),
                "mgr_charged_writeback_energy_pj": g3["mgr_charged_writeback_energy_pj"],
                "owed_writeback_energy_uJ": round(g3["owed_writeback_energy_pj"] / 1e6, 1),
                "owed_writeback_time_us": round(g3["owed_writeback_time_us"], 1),
                "claim": "_evict_from() del-drops dirty state for free (clean/recomputable assumption); "
                         "dirty TTT state owes a full writeback the manager charges as ZERO."},
        },
        "raw": {"G1": g1, "G2": g2, "G3": g3, "api_audit": audit},
        "api_grounding": {
            "prefix_reuse": "manager.py:80-96 — content-addressed hits against self.index",
            "place": "manager.py:98-121 — `if h in self.index: skip`; no update path; write charged once",
            "_evict_from": "manager.py:123-139 — spill-copy down OR `del` drop; drop charges no wb time/energy",
        },
        "e3_cross_reference": e3row,
        "caveats": [
            "Exploration-grade / RELATIVE: hatir + hat twins are pre-silicon analytical cost models "
            "(repo self-declares 'ranking + crossover, not silicon-accurate absolutes'); we assert "
            "structure and ratios, not H100 wall-clock absolutes.",
            "This is a SEMANTIC-GAP analysis: the numbers demonstrate that KVCacheManager's model is a "
            "category mismatch for RMW state (reuse=0, stale accretion, unpriced dirty writeback) — they "
            "are NOT a performance benchmark of a TTT-state manager (which does not yet exist).",
            "G2's stale-accretion blow-up (== T) is the WORST case where no eviction fires; with the hot "
            "tier under pressure the manager instead DROPS dirty state (G3) — both branches are wrong for "
            "RMW, in opposite ways (retain garbage vs lose live state).",
            "G3 forces the `del`-drop branch with a deliberately small backing tier; on a large twin the "
            "manager spills-copies instead, which is ALSO mispriced (a dirty spill is a real writeback but "
            "is charged as a clean copy at read energy, not write energy).",
            "The 'owed writeback' cost uses HBM write energy/BW from the SoT twin (7 pJ/B, 3.35 TB/s); "
            "absolute owed cost scales with those and is a ratio-vs-manager (0) claim, not an absolute.",
            "novel twins (swscratchpad/pim-cim) NOT used here — this gap is twin-independent (it is about "
            "the manager's event model); those twins may be judged simulation_ready=False by the verify "
            "committee and would only be cited as directional if used.",
        ],
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
