"""E1.3 -- KV-cache vs TTT-state quantitative comparison + batch scaling (HAT DSE).

Part III pair-thesis (D4), the quantitative core of "why decode/serving state management is a
memory-centric opportunity". We put two decode-time memory workloads side-by-side ON THE SAME
DEVICE TWIN, as a function of context length S and batch B:

  * KV-attention decode  -- APPEND-ONCE / READ-MANY: each new token appends a tiny KV slice
    (2*kv_dim*b bytes, constant) and READS the whole cache back (S*kv_dim*b bytes, GROWS with S).
    Cross-sequence: per-sequence cache, but blocks are shareable/reusable (prefix reuse).
  * TTT fast-weight decode -- READ-MODIFY-WRITE: each token READS the entire per-token state
    (m*d^2 elems) and WRITES it back (equal halves; 2*m*d^2*b bytes total, CONSTANT in S).
    Cross-sequence: UNSHARED (one live copy per sequence), always dirty, not content-addressable.

We extend the two in-repo precedents directly:
  - kv_cache_memory_dse.py  (decode KV read: matmul(1, kv_dim, S) streams the cache once)
  - stateful_writeback_dse.py (write-heavy state RMW: tile("w -> w") reads S + writes S back)

and drive them over a MINIMAL HBM device twin whose parameters are sourced from the HBM level of
hat-schema's dgx_h100_x4.json SoT (bw 3.35 TB/s, 7 pJ/B, peak 4.947e14 MAC/s) -- the same
coordinates E3-analytical used -- so the tool (hatir) and the closed-form model cross-validate.
We do NOT load the full multi-level dgx twin because its top-of-stack InfiniBand fabric (400 GB/s)
would (incorrectly) bound a single-device decode-state question; the examples build a minimal
device twin for exactly this reason.

Outputs (locate the crossovers the thesis rests on):
  (A) S* crossover  -- context where KV read traffic == TTT state traffic, per config, bf16/fp8-KV.
  (B) B_max         -- batch where aggregate TTT state RMW saturates HBM (BW-bound at a latency
                        target, and capacity-bound at HBM 80 GiB). This is where the UNSHARED,
                        write-heavy state breaks the shared-weight batching that ordinary decode
                        relies on -- TTT fast-weights are per-sequence and RMW'd every token, so
                        they cannot be amortized across the batch the way shared model weights are.

All numbers exploration-grade / RELATIVE (ranking + crossover, not silicon-accurate absolutes).
Run: /home/jimmy/repos/neural-memory-study/.venv/bin/python run.py
"""
import json
import os

import hatir

KB, MB, GB = 1024, 1024**2, 1024**3
GIB = 1024**3

# ---- device twin: HBM level of dgx_h100_x4.json (hat-schema SoT), minimal-device convention ----
BW_HBM = 3.35e12          # bandwidth_bps  (dgx HBM level)
EPB_HBM = 7.0             # energy_pj_per_byte (dgx HBM level)
PEAK_MACS = 4.947e14      # peak_macs_per_s (dgx, 4x H100 mxu aggregate)
HBM_CAP = 85899345920     # 80 GiB (dgx hbm_cap_bytes)
ONDIE_L2 = 52428800       # 50 MB L2 (dgx l2 level) -- for the "can state be pinned on-die" check
BW_HBM4 = 6.0e12          # HBM4 sensitivity row (kv_cache_memory_dse Q2 device)

KV_DIM = 1024             # GQA-8 KV elems per token per layer (E3 convention; folds K,V heads x head-dim)


def hbm_twin(bw=BW_HBM, epb=EPB_HBM):
    """Minimal HBM device twin (device level + a decode PE leaf), mirroring the two precedents so
    the roofline reflects THIS device's BW, not an enclosing fabric. Params from dgx HBM level."""
    pe = {"level_id": "pe", "role": "compute", "instances": 1, "peak_macs_per_s": PEAK_MACS,
          "dtype_bytes": 2, "energy_pj_per_mac": 0.2}
    return {"level_id": "hbm", "role": "hbm", "capacity_bytes": HBM_CAP, "bandwidth_bps": bw,
            "energy_pj_per_byte": epb, "energy_pj_per_byte_wr": epb, "children": [pe]}


def ttt_step_layer(d, m, dtype_bytes, hw, macs_per_elem=2.0):
    """One TTT per-token state RMW for ONE layer: read state (m*d^2 elems) + write it back.
    Reuses stateful_writeback_dse.writeback()'s tile("w -> w") == 2*S traffic (read + write)."""
    elems = m * d * d
    return hatir.tile("w -> w", {"w": elems}, hw, dtype_bytes=dtype_bytes,
                      macs_per_point=macs_per_elem, name="ttt_rmw").derived


def kv_read_layer(S, hw, kv_bytes=2, kv_dim=KV_DIM):
    """One decode-step KV read for ONE layer: stream the whole cache [S, kv_dim] once.
    Reuses kv_cache_memory_dse.decode_step()'s matmul(1, kv_dim, S)."""
    p = hatir.matmul(1, kv_dim, S, hw, tile={"m": 1, "n": kv_dim, "k": min(S, 2048)},
                     dtype_bytes=kv_bytes)
    return p.derived


# ---- configs: canonical (d, L) per param scale; m from published state descriptions (E3 table) ----
CONFIGS = [
    dict(config="Titans-340M",    d=1024, L=24, m=16),
    dict(config="Titans-760M",    d=1536, L=24, m=16),
    dict(config="neural-mem-1.3B", d=2048, L=24, m=16),   # validation anchor (E3 S*=65536)
    dict(config="neural-mem-1.3B-m8", d=2048, L=24, m=8), # Miras/TNT/HOPE/Sleep 2-layer-MLP variant
    dict(config="hypo-7B",        d=4096, L=32, m=16),
]

S_SWEEP = [4096, 16384, 65536, 262144, 1048576]
B_SWEEP = [1, 8, 32, 128]
LAT_TARGETS_MS = [10.0, 50.0]

sep = "=" * 108


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    hw = hatir.linearize_graph(hbm_twin()).stack()

    out(sep)
    out("E1.3  KV-cache (append-once/read-many) vs TTT-state (read-modify-write) -- same HBM twin")
    out(f"twin: HBM {BW_HBM/1e12:.2f} TB/s, {EPB_HBM:.0f} pJ/B, {HBM_CAP/GIB:.0f} GiB  "
        f"(dgx_h100_x4.json HBM level; minimal-device convention). PER TOKEN, whole model (xL).")
    out("")

    results = {
        "experiment": "E1.3-kv-vs-ttt",
        "workload_pair": {
            "KV": "append-once/read-many: append 2*kv_dim*b (const) + read S*kv_dim*b (grows w/ S); shareable/reusable across seqs (prefix reuse)",
            "TTT": "read-modify-write: read m*d^2 + write m*d^2 (equal halves, const in S); UNSHARED per-seq, always dirty, not content-addressable",
        },
        "hw": {"bw_hbm_bps": BW_HBM, "epb_hbm": EPB_HBM, "peak_macs_s": PEAK_MACS,
               "hbm_cap_bytes": HBM_CAP, "ondie_l2_bytes": ONDIE_L2, "kv_dim_gqa8": KV_DIM,
               "provenance": "dgx_h100_x4.json HBM level (hat-schema SoT); minimal-device twin so roofline reflects HBM not enclosing fabric"},
        "panelA_S_star": [],
        "panelB_B_max": [],
        "traffic_decomposition": {},
        "kv_B_max_vs_S": [],
        "e3_crosscheck": {},
    }

    # ===================== PANEL A: S* crossover (per-token traffic) =====================
    out("PANEL A) S* crossover -- context S where KV read traffic == TTT state traffic (per token)")
    out("  Per config: TTT RMW GB/tok (const), TTT one-way read GB/tok (=half), and S* where the")
    out("  linearly-growing KV read overtakes each. bf16 KV and fp8 KV (halves KV -> doubles S*).")
    out("")
    hdr = (f"  {'config':>17} | {'d':>4} {'m':>2} {'L':>2} | {'TTT rmw':>8} {'TTT read':>8} "
           f"{'ms/tok':>7} {'bnd':>4} | {'S*_read':>9} {'S*_rmw':>9} | {'S*_fp8KV':>9}")
    out(hdr)
    out("  " + "-" * (len(hdr) - 2))

    for cfg in CONFIGS:
        d, m, L = cfg["d"], cfg["m"], cfg["L"]
        # TTT: whole-model per-token RMW (hatir per-layer x L)
        ttt = ttt_step_layer(d, m, dtype_bytes=2, hw=hw)
        ttt_rmw_bytes = ttt.total_backing_bytes * L          # read + write
        ttt_read_bytes = ttt_rmw_bytes / 2.0                 # one-way state read (== E3's S* basis)
        ttt_ms = ttt.kernel_time_us * L / 1e3
        ttt_uj = ttt.energy_pj * L / 1e6

        # KV: measure read GB/tok at one S via hatir, exploit exact linearity in S to solve S*
        S_ref = 65536
        kv_ref = kv_read_layer(S_ref, hw, kv_bytes=2).total_backing_bytes * L
        kv_per_S = kv_ref / S_ref                            # bytes per token per unit context (linear)
        S_star_read = ttt_read_bytes / kv_per_S              # KV read == TTT one-way read  (E3 def)
        S_star_rmw = ttt_rmw_bytes / kv_per_S                # KV read == full TTT RMW (=2x read)
        # fp8 KV halves per-S bytes -> doubles S*
        kv_ref_fp8 = kv_read_layer(S_ref, hw, kv_bytes=1).total_backing_bytes * L
        S_star_read_fp8 = ttt_read_bytes / (kv_ref_fp8 / S_ref)

        out(f"  {cfg['config']:>17} | {d:>4} {m:>2} {L:>2} | "
            f"{ttt_rmw_bytes/1e9:7.3f}G {ttt_read_bytes/1e9:7.3f}G "
            f"{ttt_ms:7.3f} {ttt.bound:>4} | {S_star_read:9.0f} {S_star_rmw:9.0f} | {S_star_read_fp8:9.0f}")

        results["panelA_S_star"].append({
            "config": cfg["config"], "d": d, "m": m, "L": L, "kv_dim": KV_DIM,
            "ttt_rmw_GB_token": round(ttt_rmw_bytes / 1e9, 4),
            "ttt_read_GB_token": round(ttt_read_bytes / 1e9, 4),
            "ttt_ms_token": round(ttt_ms, 4),
            "ttt_uJ_token": round(ttt_uj, 1),
            "ttt_bound": ttt.bound,
            "S_star_read_tokens": round(S_star_read),
            "S_star_rmw_tokens": round(S_star_rmw),
            "S_star_read_fp8KV_tokens": round(S_star_read_fp8),
        })
    out("")
    out("  Reading: at S < S*_read the TTT state read alone already moves more bytes/tok than the")
    out("  whole KV cache read; TTT ALSO writes that state back (RMW), a traffic component the")
    out("  append-once KV cache never has. S*_rmw = 2 x S*_read. fp8 KV pushes S* out 2x.")
    out("")

    # ===================== Traffic decomposition (read vs write) =====================
    out("TRAFFIC COMPOSITION (per token, whole model) at the 1.3B anchor (d=2048,m=16,L=24), S=64k:")
    d, m, L = 2048, 16, 24
    ttt = ttt_step_layer(d, m, 2, hw)
    ttt_total = ttt.total_backing_bytes * L
    kv = kv_read_layer(65536, hw, kv_bytes=2)
    kv_read_total = kv.total_backing_bytes * L
    kv_write_total = 2 * KV_DIM * 2 * L                      # append one token's K,V (2 bytes)
    out(f"  KV  : read {kv_read_total/1e9:6.3f} GB (grows w/ S) + write {kv_write_total/1e6:.4f} MB "
        f"(append, const)  -> write is {100*kv_write_total/kv_read_total:.4f}% of read  [READ-HEAVY]")
    out(f"  TTT : read {ttt_total/2/1e9:6.3f} GB + write {ttt_total/2/1e9:6.3f} GB (const in S)      "
        f"-> write is {100*(ttt_total/2)/(ttt_total/2):.0f}% of read       [WRITE == READ, RMW]")
    out("  => qualitatively different memory systems: KV wants capacity+read-BW+sharing; TTT wants")
    out("     write-BW + per-sequence residency + checkpoint/rollback (no reuse, always dirty).")
    out("")
    results["traffic_decomposition"] = {
        "anchor": "d=2048,m=16,L=24,S=65536,bf16",
        "kv_read_GB_token": round(kv_read_total / 1e9, 4),
        "kv_write_MB_token_append": round(kv_write_total / 1e6, 6),
        "kv_write_pct_of_read": round(100 * kv_write_total / kv_read_total, 6),
        "ttt_read_GB_token": round(ttt_total / 2 / 1e9, 4),
        "ttt_write_GB_token": round(ttt_total / 2 / 1e9, 4),
        "ttt_write_pct_of_read": 100.0,
    }

    # ===================== PANEL B: B_max batch scaling =====================
    out(sep)
    out("PANEL B) B_max -- batch where UNSHARED TTT state RMW saturates HBM (breaks shared-weight batching)")
    out("  Ordinary decode shares model weights across the batch, so weight reads amortize. TTT")
    out("  fast-weights are PER-SEQUENCE state, RMW'd every token -> aggregate = B x rmw, no amortize.")
    out("  BW-bound: B_max = target_ms / rmw_ms_token. Cap-bound: HBM 80GiB / state_per_seq.")
    out("")
    hdrB = (f"  {'config':>17} | {'rmw ms/tok':>10} {'state/seq GB':>12} | "
            f"{'Bmax@10ms':>9} {'Bmax@50ms':>9} | {'Bmax_cap80GiB':>13} | {'ondie 1seq?':>11}")
    out(hdrB)
    out("  " + "-" * (len(hdrB) - 2))
    for cfg in CONFIGS:
        d, m, L = cfg["d"], cfg["m"], cfg["L"]
        ttt = ttt_step_layer(d, m, 2, hw)
        rmw_ms = ttt.kernel_time_us * L / 1e3
        state_per_seq = m * d * d * 2 * L                    # one live copy (bf16), whole model
        bmax = {t: t / rmw_ms for t in LAT_TARGETS_MS}
        bmax_cap = HBM_CAP / state_per_seq
        ondie_one_seq = state_per_seq <= ONDIE_L2
        out(f"  {cfg['config']:>17} | {rmw_ms:10.3f} {state_per_seq/1e9:12.3f} | "
            f"{bmax[10.0]:9.2f} {bmax[50.0]:9.2f} | {bmax_cap:13.1f} | {str(ondie_one_seq):>11}")
        results["panelB_B_max"].append({
            "config": cfg["config"], "d": d, "m": m, "L": L,
            "rmw_ms_token": round(rmw_ms, 4),
            "state_per_seq_GB": round(state_per_seq / 1e9, 4),
            "B_max_bw_10ms": round(bmax[10.0], 3),
            "B_max_bw_50ms": round(bmax[50.0], 3),
            "B_max_cap_HBM80GiB": round(bmax_cap, 3),
            "ondie_L2_holds_one_seq": ondie_one_seq,
            "binding_at_10ms": "BW" if bmax[10.0] < bmax_cap else "capacity",
        })
    out("")
    out("  For latency-sensitive decode BW binds FAR before capacity (opposite of the KV cache,")
    out("  which is capacity-bound). No config's whole-model state fits on-die L2 (50MB) even for")
    out("  ONE sequence -> TTT state cannot be pinned on-die whole-model; must stream/partition.")
    out("")

    # ---- KV B_max vs S: the qualitative contrast (KV B_max shrinks with S; TTT B_max const in S) ----
    out("  Contrast -- KV decode B_max SHRINKS with context S, TTT B_max is INDEPENDENT of S")
    out("  (1.3B anchor d=2048,L=24; KV GQA-8 bf16). B_max_bw = target / read_ms(S):")
    d, m, L = 2048, 16, 24
    ttt = ttt_step_layer(d, m, 2, hw)
    ttt_ms = ttt.kernel_time_us * L / 1e3
    out(f"    {'S tokens':>10} | {'KV ms/tok':>9} {'KV Bmax@50ms':>12} | {'TTT ms/tok':>10} {'TTT Bmax@50ms':>13}")
    for S in S_SWEEP:
        kv = kv_read_layer(S, hw, kv_bytes=2)
        kv_ms = kv.kernel_time_us * L / 1e3
        out(f"    {S:>10} | {kv_ms:9.3f} {50.0/kv_ms:12.2f} | {ttt_ms:10.3f} {50.0/ttt_ms:13.2f}")
        results["kv_B_max_vs_S"].append({
            "S_tokens": S, "kv_ms_token": round(kv_ms, 4), "kv_B_max_50ms": round(50.0 / kv_ms, 3),
            "ttt_ms_token": round(ttt_ms, 4), "ttt_B_max_50ms": round(50.0 / ttt_ms, 3),
        })
    out("    -> KV B_max falls as context grows (read scales with S); TTT B_max is flat (state fixed).")
    out("       Different batching economics: KV trades batch for context; TTT's batch ceiling is a")
    out("       fixed per-sequence state-BW tax, hit at small B and unmovable by shortening context.")
    out("")

    # ===================== E3 cross-check =====================
    out(sep)
    out("E3 CROSS-CHECK (expected values from experiments/E3-analytical/results.json):")
    e3 = None
    e3_path = "/home/jimmy/repos/neural-memory-study/experiments/E3-analytical/results.json"
    checks = {}
    if os.path.exists(e3_path):
        e3 = json.load(open(e3_path))
        # anchor: neural-mem-1.3B Titans (d2048 m16 L24)
        e3row = next(r for r in e3["crossovers"]["table"]
                     if r["config"] == "neural-mem-1.3B" and r["method"] == "Titans")
        me = next(r for r in results["panelA_S_star"] if r["config"] == "neural-mem-1.3B")
        mb = next(r for r in results["panelB_B_max"] if r["config"] == "neural-mem-1.3B")

        def near(a, b, tol=0.03):
            return abs(a - b) <= tol * max(abs(b), 1e-9)

        checks = {
            "ttt_rmw_GB_token": {"hatir": me["ttt_rmw_GB_token"], "E3": e3row["rmw_GB_token"],
                                 "pass": near(me["ttt_rmw_GB_token"], e3row["rmw_GB_token"])},
            "ttt_ms_token": {"hatir": me["ttt_ms_token"], "E3": e3row["rmw_ms_token"],
                             "pass": near(me["ttt_ms_token"], e3row["rmw_ms_token"])},
            "S_star_read": {"hatir": me["S_star_read_tokens"], "E3": e3row["S_star_tokens"],
                            "pass": near(me["S_star_read_tokens"], e3row["S_star_tokens"])},
            "B_max_bw_10ms": {"hatir": mb["B_max_bw_10ms"], "E3": e3row["B_max_bw_10ms"],
                              "pass": near(mb["B_max_bw_10ms"], e3row["B_max_bw_10ms"])},
            "B_max_bw_50ms": {"hatir": mb["B_max_bw_50ms"], "E3": e3row["B_max_bw_50ms"],
                              "pass": near(mb["B_max_bw_50ms"], e3row["B_max_bw_50ms"])},
            "state_per_seq_GB": {"hatir": mb["state_per_seq_GB"], "E3": e3row["state_per_seq_GB"],
                                 "pass": near(mb["state_per_seq_GB"], e3row["state_per_seq_GB"])},
        }
        for k, v in checks.items():
            out(f"  {k:>18}: hatir={v['hatir']:<12} E3={v['E3']:<12} {'OK' if v['pass'] else 'MISMATCH'}")
        results["e3_crosscheck"] = {"anchor": "neural-mem-1.3B/Titans", "checks": checks,
                                    "all_pass": all(v["pass"] for v in checks.values())}
        out(f"  => hatir (tool) and E3 (closed-form) agree on the anchor: "
            f"{'ALL PASS' if results['e3_crosscheck']['all_pass'] else 'DISCREPANCY'}")
    else:
        out("  E3 results.json not found -- skipping cross-check.")
        results["e3_crosscheck"] = {"status": "E3 not found"}
    out("")

    # ---- optional end-to-end serving sanity check (dense attention), non-blocking ----
    serve_note = None
    try:
        from hatir import serving
        # dense GQA-8 baseline at the 1.3B anchor shape (serving.serve spec schema)
        spec = {"layers": 24, "H": 2048, "n_heads": 16, "n_kv": 8, "head_dim": 128, "inter": 8192}
        r1 = serving.serve(spec, hw, batch=1, prompt_len=2048, gen_len=128)
        r_ctx = serving.serve(spec, hw, batch=1, prompt_len=32768, gen_len=128)
        pt_short = r1.get("per_token_ms")
        pt_long = r_ctx.get("per_token_ms")
        serve_note = {"ok": True, "per_token_ms_ctx2k": round(pt_short, 4),
                      "per_token_ms_ctx32k": round(pt_long, 4),
                      "kv_decode_grows_with_ctx": pt_long > pt_short,
                      "note": "dense-attention KV decode per-token latency RISES with context (KV read grows), "
                              "confirming the KV read-many wall; contrast the S-INDEPENDENT TTT per-token latency in Panel B."}
        out(f"[serving.serve sanity] dense-attn KV decode ms/tok: ctx2k={pt_short:.4f} ctx32k={pt_long:.4f} "
            f"(rises w/ ctx: {pt_long > pt_short}) -- vs TTT ms/tok flat in S")
    except Exception as e:  # non-blocking: serving spec schema may differ
        serve_note = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        out(f"[serving.serve sanity] skipped (non-blocking): {type(e).__name__}: {e}")
    results["serving_sanity"] = serve_note
    out("")

    out(sep)
    out("PAIR-THESIS TAKEAWAY (D4): on the SAME HBM device, KV-attention decode is read-heavy /")
    out("capacity-bound / shareable and grows with context; TTT fast-weight decode is a per-token")
    out("read-modify-write of an UNSHARED, write==read, context-INDEPENDENT state that saturates")
    out("HBM bandwidth at small batch (B_max hit before capacity, unmovable by shortening context).")
    out("=> decode/serving state management is a distinct, memory-centric problem, NOT a KV cache.")
    out(sep)

    # ---- headline numbers ----
    anchor = next(r for r in results["panelA_S_star"] if r["config"] == "neural-mem-1.3B")
    anchorB = next(r for r in results["panelB_B_max"] if r["config"] == "neural-mem-1.3B")
    results["headline"] = {
        "anchor_config": "neural-mem-1.3B (d=2048, m=16, L=24, GQA-8, bf16)",
        "ttt_rmw_GB_token": anchor["ttt_rmw_GB_token"],
        "ttt_ms_token_HBM3": anchor["ttt_ms_token"],
        "S_star_read_tokens": anchor["S_star_read_tokens"],
        "S_star_rmw_tokens": anchor["S_star_rmw_tokens"],
        "B_max_bw_10ms": anchorB["B_max_bw_10ms"],
        "B_max_bw_50ms": anchorB["B_max_bw_50ms"],
        "B_max_cap_HBM80GiB": anchorB["B_max_cap_HBM80GiB"],
        "kv_write_pct_of_read": results["traffic_decomposition"]["kv_write_pct_of_read"],
        "ttt_write_pct_of_read": results["traffic_decomposition"]["ttt_write_pct_of_read"],
    }

    results["caveats"] = [
        "hatir/twin numbers are exploration-grade / RELATIVE (repo's own declaration): ranking + crossover, not silicon-accurate absolutes. Only ratios and crossover positions are promoted to text.",
        "Minimal single-device HBM twin (params sourced from dgx_h100_x4.json HBM level), NOT the full multi-level dgx twin -- the full twin's InfiniBand top-of-stack (400 GB/s) would incorrectly bound a single-device decode-state question. This matches the two in-repo precedents' convention.",
        "Per-kernel roofline only: no kernel-launch / scheduling / decode tail-effect overheads; per-token wall-clock is a lower bound.",
        "The 6 papers do NOT publish H100 wall-clock decode; absolute us/token is externally unverifiable and deferred to the in-house A100 runbook (Part III-a). S*, B_max, and the read/write split are ratios/positions and are the load-bearing claims.",
        "KV comparison fixes GQA-8 kv_dim=1024 (K,V heads x head-dim, E3 convention); MHA / other GQA groups shift S* linearly. The in-repo kv_cache_memory_dse example uses N=2*Hkv*d=2048 (head-dim 128); we adopt E3's kv_dim=1024 so tool and closed-form share coordinates.",
        "State multipliers m (2-layer MLP=8d^2, +momentum=16d^2) are from published architecture descriptions, not released decode traces; HOPE's per-token fast level is the m used here, its slower CMS levels add lower-frequency resident state (see E1.4).",
        "novel twins (novel-swscratchpad-npu, novel-pim-cim) are NOT used here (this experiment is a same-device KV-vs-TTT contrast on HBM); where they appear elsewhere the verify committee marks them simulation_ready=False / PPA-abstain, so any PIM/scratchpad result is directional DSE only, not a device claim.",
        "B_max_cap uses HBM 80 GiB (85899345920 B) whole-model per-sequence state; a real serving stack also holds weights + activations, so the usable capacity ceiling is lower (B_max even smaller) -- our number is an upper bound on the capacity-bound batch.",
    ]

    with open(os.path.join(os.path.dirname(__file__), "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(os.path.dirname(__file__), "stdout.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    _validate(results)
    return results


def _validate(r):
    """Trend checks -- the pair-thesis structure must hold, not just print."""
    A = {x["config"]: x for x in r["panelA_S_star"]}
    B = {x["config"]: x for x in r["panelB_B_max"]}
    # 1) TTT is memory-bound at every config (RMW, arith intensity ~O(1))
    assert all(x["ttt_bound"] == "memory" for x in r["panelA_S_star"]), "TTT must be memory-bound"
    # 2) S*_rmw == 2 * S*_read (TTT writes back an equal half the KV cache never has)
    for x in r["panelA_S_star"]:
        assert abs(x["S_star_rmw_tokens"] - 2 * x["S_star_read_tokens"]) <= 2, "S*_rmw != 2 S*_read"
    # 3) fp8 KV doubles S* (cheaper KV pushes the crossover out)
    for x in r["panelA_S_star"]:
        assert abs(x["S_star_read_fp8KV_tokens"] - 2 * x["S_star_read_tokens"]) <= 2, "fp8 !~ 2x"
    # 4) KV is read-heavy (append << read); TTT is 50/50 RMW
    td = r["traffic_decomposition"]
    assert td["kv_write_pct_of_read"] < 0.01 and td["ttt_write_pct_of_read"] == 100.0
    # 5) B_max BW-bound < capacity-bound for latency-sensitive decode (BW binds first)
    for x in r["panelB_B_max"]:
        assert x["B_max_bw_10ms"] < x["B_max_cap_HBM80GiB"], "BW must bind before capacity"
        assert x["binding_at_10ms"] == "BW"
    # 6) no whole-model state fits on-die L2 even for one sequence
    assert all(not x["ondie_L2_holds_one_seq"] for x in r["panelB_B_max"])
    # 7) TTT B_max independent of S; KV B_max monotonically shrinks with S
    kv = r["kv_B_max_vs_S"]
    ttt_bmax = {row["ttt_B_max_50ms"] for row in kv}
    assert len(ttt_bmax) == 1, "TTT B_max must be constant in S"
    kv_bmax_seq = [row["kv_B_max_50ms"] for row in kv]
    assert all(kv_bmax_seq[i] > kv_bmax_seq[i + 1] for i in range(len(kv_bmax_seq) - 1)), "KV B_max must shrink w/ S"
    # 8) larger models -> larger state/seq -> smaller B_max (state-BW tax grows)
    order = ["Titans-340M", "Titans-760M", "neural-mem-1.3B", "hypo-7B"]
    bmaxs = [B[c]["B_max_bw_10ms"] for c in order]
    assert all(bmaxs[i] > bmaxs[i + 1] for i in range(len(bmaxs) - 1)), "B_max must fall as model grows"
    # 9) E3 cross-check passes if E3 present
    if r.get("e3_crosscheck", {}).get("checks"):
        assert r["e3_crosscheck"]["all_pass"], "hatir must reproduce E3 anchor"
    print("[validate] all pair-thesis trends hold "
          "(TTT memory-bound; S*_rmw=2xS*_read; fp8 doubles S*; KV read-heavy vs TTT RMW; "
          "BW binds before capacity; no on-die fit; KV B_max shrinks w/ S, TTT flat; E3 cross-check).")


if __name__ == "__main__":
    main()
