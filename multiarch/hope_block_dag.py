#!/usr/bin/env python3
"""HOPE block — per-op DAG cost + roofline on the H100 twin, and an HBM->HBM-PIM what-if.

ONE HOPE block, batch B=1, chunk of C tokens, model dim d. Cold/rigorous: every tensor op listed with
FLOPs and HBM bytes; per-op roofline time = max(FLOP/peak, bytes/BW); block time = serial sum (no
assumed magic overlap). Then: replace HBM with HBM-PIM and run the Titans/CMS UPDATE-APPLY (the write-
heavy, low-AI state RMW) in-bank on PIM -> that traffic leaves the HBM bus; recompute the bottleneck.

Architecture (stated assumptions):
  Self-modifying Titans: 5 adaptive memories {k,v,mem,eta,alpha} + static W_q.
    - M_k, M_v, M_mem : 2-layer MLP d->h_t->d  (h_t=d, compact Titans memory)  -> 2 d^2 params each
    - M_eta, M_alpha  : 2-layer MLP d->h_t->1  (scalar hyperparam output)       -> ~d^2 params each
    - W_q (static)    : d->d                                                    -> d^2 params
  CMS: 3 MLP levels, FFN shape rule d->4d->d (h_cms=4d) -> 8 d^2 params each; fast level updates per chunk.
  Forward: o=M_mem(q); y=CMS3(CMS2(CMS1(o))). Update: self-target v_hat=M(v), pred M(k), backward, DGD-apply.
UPDATE-APPLY cost model (transparent): update_compute = 4x memory forward (target+pred+backward);
  DGD-apply = low-AI RMW of the memory weights (read+write params, ~4 MAC/elem).
Roofline: activations (C x d, C x 4d) counted as HBM traffic only when they exceed on-die L2 (50 MB); else on-chip.
"""
import json, math, sys, argparse

# ---- H100 twin (hat-schema dgx_h100_x4.json single-device) ----
BW   = 3.35e12          # HBM3 B/s
PEAK = 9.894e14         # dense bf16 FLOP/s (2*4.947e14 MAC/s)
RIDGE= PEAK/BW          # ~295 FLOP/B
L2   = 52428800         # 50 MB on-die
BYTES= 2                # bf16
# ---- HBM-PIM what-if (in-bank compute for the write-heavy update-apply) ----
PIM_BW   = 8.2e12       # in-bank aggregate bandwidth (novel-pim-cim; ~2.4x external HBM bus)
PIM_MACS = 1.2e12       # in-bank MAC throughput (novel-pim-cim); RMW apply ~1 MAC/element
APPLY_MAC_PER_ELEM = 1.0
HBM_ELEM_RATE = BW/BYTES  # HBM's effective element streaming rate (1.67e12 elem/s) — PIM must beat this


def mlp(C, d_in, h, d_out):
    """2-layer MLP forward: FLOP + weight params."""
    flop = 2*C*d_in*h + 2*C*h*d_out
    params = d_in*h + h*d_out
    act = C*h                      # hidden activation elements (sigma)
    return flop, params, act


def build_ops(C, d):
    ops = []   # each: dict(name, flop, wbytes(weight read), abytes(activation r/w), state_rmw_bytes, kind)
    def add(name, flop, params_read=0, act=0, state_rmw_params=0, kind="gemm"):
        wb = params_read*BYTES
        # activation HBM traffic only if a single activation tensor exceeds L2 (else stays on-die)
        ab = (act*BYTES) if act*BYTES > L2 else 0
        srmw = state_rmw_params*BYTES*2   # read + write the state
        ops.append(dict(name=name, flop=flop, wbytes=wb, abytes=ab, state_rmw=srmw, kind=kind))

    # ---------- Self-mod Titans FORWARD ----------
    f,p,a = 2*C*d*d, d*d, 0;               add("q=X·Wq (static)", f, p, a)
    for nm,din,h,dout in [("M_k(X)",d,d,d),("M_v(X)",d,d,d)]:
        f,p,a = mlp(C,din,h,dout);          add(f"k/v: {nm}", f, p, a)
    for nm in ["M_eta(X)","M_alpha(X)"]:
        f,p,a = mlp(C,d,d,1);               add(f"gate: {nm}", f, p, a)
    f,p,a = mlp(C,d,d,d);                    add("o=M_mem(q) [pre-update read]", f, p, a)

    # ---------- Self-mod Titans UPDATE (5 memories) ----------
    mem_fwd = {"k":mlp(C,d,d,d),"v":mlp(C,d,d,d),"mem":mlp(C,d,d,d),"eta":mlp(C,d,d,1),"alpha":mlp(C,d,d,1)}
    for sq,(mf,mp,ma) in mem_fwd.items():
        add(f"update-compute M_{sq} (tgt+pred+bwd ~4x fwd)", 4*mf, mp, ma, kind="gemm")   # reads weights for grad
        add(f"DGD-apply M_{sq} (state RMW, low-AI)",         4*mp, 0,  0, state_rmw_params=mp, kind="update")

    # ---------- CMS FORWARD (3 FFN-shape levels, series) ----------
    for lvl in [1,2,3]:
        f,p,a = mlp(C,d,4*d,d);             add(f"CMS L{lvl} fwd (d->4d->d)", f, p, a)

    # ---------- CMS UPDATE (fast level L1 per chunk; mid/slow amortized ~0) ----------
    f,p,a = mlp(C,d,4*d,d)
    add("CMS L1 update-compute (~4x fwd)", 4*f, p, a, kind="gemm")
    add("CMS L1 DGD-apply (state RMW)",    4*p, 0, 0, state_rmw_params=p, kind="update")
    return ops


def roofline(ops, pim=False):
    """Per-op roofline. pim=True -> 'update' ops run the state-RMW in-bank (bytes leave HBM bus) at PIM FLOP rate."""
    rows=[]; t_total=0
    for o in ops:
        if pim and o["kind"]=="update":
            # in-bank RMW: state does NOT cross HBM bus. cost = max(in-bank move, in-bank apply-compute).
            # gradient/forward GEMMs are NOT offloaded (need tensor cores) — only the low-AI apply is.
            hbm_bytes = o["wbytes"] + o["abytes"]          # state_rmw removed from bus
            elems = o["state_rmw"]/BYTES
            comp_t = (elems*APPLY_MAC_PER_ELEM)/PIM_MACS   # in-bank apply throughput
            mem_t  = max(o["state_rmw"]/PIM_BW, hbm_bytes/BW)  # in-bank move (fast) or residual bus traffic
        else:
            hbm_bytes = o["wbytes"] + o["abytes"] + o["state_rmw"]
            comp_t = o["flop"]/PEAK
            mem_t  = hbm_bytes/BW
        t = max(comp_t, mem_t)
        rows.append(dict(name=o["name"], kind=o["kind"], flop=o["flop"], bytes=hbm_bytes,
                         ai=(o["flop"]/hbm_bytes if hbm_bytes else float('inf')),
                         comp_t=comp_t, mem_t=mem_t, t=t, bound=("compute" if comp_t>mem_t else "memory")))
        t_total+=t
    return rows, t_total


def summarize(C, d):
    ops = build_ops(C, d)
    for tag, pim in [("BASELINE (HBM)", False), ("HBM-PIM (update in-bank)", True)]:
        rows, T = roofline(ops, pim=pim)
        F = sum(r["flop"] for r in rows); Bt = sum(r["bytes"] for r in rows)
        mem_T = sum(r["mem_t"] for r in rows); comp_T = sum(r["comp_t"] for r in rows)
        state_bytes = sum(o["state_rmw"] for o in ops)
        print(f"\n===== C={C}, d={d} — {tag} =====")
        print(f"  block: {F/1e9:.2f} GFLOP, {Bt/1e6:.2f} MB HBM, AI={F/Bt:.2f} FLOP/B, "
              f"ridge={RIDGE:.0f} -> {'COMPUTE' if F/Bt>RIDGE else 'MEMORY'}-bound (block avg)")
        print(f"  time: {T*1e6:.1f} us (serial max-sum) | if-all-compute {comp_T*1e6:.1f} us | if-all-mem {mem_T*1e6:.1f} us")
        print(f"  state-RMW (update write-back) = {state_bytes/1e6:.2f} MB "
              f"({100*state_bytes/max(Bt+ (state_bytes if pim else 0),1):.0f}% of baseline HBM)")
        # top-3 time-dominating ops
        top = sorted(rows, key=lambda r:-r["t"])[:4]
        print("  bottleneck ops (by time):")
        for r in top:
            print(f"    {r['name']:42} {r['t']*1e6:7.2f} us  [{r['bound']}, AI={r['ai']:.1f}]")
    return ops


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--d",type=int,default=2048)
    ap.add_argument("--C",type=int,nargs="*",default=[1,64,512,2048]); a=ap.parse_args()
    print(f"H100 twin: BW={BW:.2e} B/s, peak={PEAK:.3e} FLOP/s, ridge={RIDGE:.1f} FLOP/B, L2={L2/1e6:.0f}MB")
    print(f"HBM-PIM: in-bank BW={PIM_BW:.2e} B/s, MAC-rate={PIM_MACS:.2e} (HBM elem-rate={HBM_ELEM_RATE:.2e}); state-RMW leaves HBM bus")
    out={}
    for C in a.C:
        summarize(C, a.d)
    # machine-readable dump for figures
    dump={"d":a.d,"twin":{"BW":BW,"PEAK":PEAK,"RIDGE":RIDGE},"C":{}}
    for C in a.C:
        ops=build_ops(C,a.d)
        for tag,pim in [("baseline",False),("pim",True)]:
            rows,T=roofline(ops,pim=pim)
            dump["C"].setdefault(str(C),{})[tag]={"t_us":T*1e6,
                "flop":sum(r["flop"] for r in rows),"bytes":sum(r["bytes"] for r in rows),
                "ops":[{"name":r["name"],"kind":r["kind"],"t_us":r["t"]*1e6,"bound":r["bound"],
                        "ai":r["ai"],"flop":r["flop"],"bytes":r["bytes"]} for r in rows]}
    import os
    json.dump(dump,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"results_full","_hope_block_dag.json"),"w"),
              ensure_ascii=False,indent=1)
    print("\n-> multiarch/results_full/_hope_block_dag.json")


if __name__=="__main__":
    main()
