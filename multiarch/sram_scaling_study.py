#!/usr/bin/env python3
"""H100 on-chip SRAM 1x/2x/5x/10x — HOPE-block decode when the fast-weight state becomes resident.

E1.2 결론("decode-state 병목은 on-chip 용량 문제")의 직접 검증. on-chip SRAM을 키우면 block의
파라미터(가중치+state)가 상주해 RMW/read가 HBM(3.35 TB/s)이 아니라 SRAM(~10 TB/s, epb 1/17)에서 돈다.

가정(명시): on-chip SRAM BW = 1.0e13 B/s(L2급; 전용 scratchpad면 더 높을 수 있음), SRAM epb 0.4 pJ/B.
residency 규칙(block 단위): cap>=전체 파라미터 -> 전부 on-chip; cap>=state -> RMW state만 on-chip(가중치 read는 HBM);
else 전부 HBM. baseline H100 L2 = 50 MB. 1x/2x/5x/10x = 50/100/250/500 MB.
결과는 ideal roofline 하한(냉정). fig-sram-scaling.png + _sram_scaling.json.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hope_block_dag import build_ops, PEAK, BYTES

HERE=os.path.dirname(os.path.abspath(__file__))
HBM_BW=3.35e12
SRAM_BW=1.0e13
CAPS_MB=[50,100,250,500]     # 1x,2x,5x,10x of H100 L2 (50MB)


def block_state_and_params(C,d):
    """UNIQUE on-chip footprint (NOT summed traffic — weights are re-read, but resident once).
    Titans updatable state = 5 memories (Mk,Mv,Mmem=2d² each, Mη,Mα=d² each = 8d²) + CMS L1 (8d²) = 16d².
    Read-only extra = Wq (d²) + CMS L2,L3 (16d²) = 17d². Total unique params = 33d²."""
    state_params = 16*d*d            # updatable+resident state (RMW'd)
    total_params = 33*d*d            # all block weights (state ⊂ this)
    return state_params*BYTES, total_params*BYTES   # (state bytes, total unique param bytes)


def roofline(ops, cap_bytes, state_bytes, param_bytes):
    total=param_bytes            # param_bytes = total unique (state ⊂ it)
    if cap_bytes>=total:        mode="all-on-chip"
    elif cap_bytes>=state_bytes:mode="state-on-chip"
    else:                        mode="all-HBM"
    t=0.0
    for o in ops:
        if o["kind"]=="update":                      # state RMW
            bw=SRAM_BW if mode in ("all-on-chip","state-on-chip") else HBM_BW
            hbm=o["wbytes"]+o["abytes"]+o["state_rmw"]
        else:                                        # forward/gradient (reads weights)
            bw=SRAM_BW if mode=="all-on-chip" else HBM_BW
            hbm=o["wbytes"]+o["abytes"]+o["state_rmw"]
        t+=max(o["flop"]/PEAK, hbm/bw)
    return {"mode":mode,"t_us":t*1e6}


def main():
    d=2048
    out={"d":d,"assumptions":{"SRAM_BW":SRAM_BW,"HBM_BW":HBM_BW,"baseline_L2_MB":50},"rows":[]}
    print(f"H100 on-chip SRAM scaling (d={d}); on-chip {SRAM_BW/1e12:.0f}TB/s vs HBM {HBM_BW/1e12:.2f}TB/s")
    for C in (1,64):
        st,pa=block_state_and_params(C,d)
        print(f"\n--- C={C}: block state={st/1e6:.0f}MB, weights={pa/1e6:.0f}MB, total={ (st+pa)/1e6:.0f}MB ---")
        print(f"  {'SRAM(x)':10}{'cap':>8}{'mode':>16}{'decode µs':>12}{'vs 50MB':>10}")
        base=None
        for i,mb in enumerate(CAPS_MB):
            r=roofline(build_ops(C,d), mb*1e6, st, pa)
            if base is None: base=r["t_us"]
            print(f"  {['1x','2x','5x','10x'][i]:10}{mb:>6}MB{r['mode']:>16}{r['t_us']:>11.0f}{base/r['t_us']:>9.2f}x")
            out["rows"].append({"C":C,"cap_MB":mb,"mult":['1x','2x','5x','10x'][i],
                                "state_MB":round(st/1e6,1),"weights_MB":round(pa/1e6,1),
                                "mode":r["mode"],"decode_us":round(r["t_us"],1)})
    json.dump(out,open(os.path.join(HERE,"results_full","_sram_scaling.json"),"w"),ensure_ascii=False,indent=1)

    # figure
    import matplotlib; matplotlib.use("Agg")
    from matplotlib import font_manager as fm
    _C="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if os.path.exists(_C): fm.fontManager.addfont(_C); matplotlib.rcParams["font.family"]="Noto Sans CJK KR"
    matplotlib.rcParams["axes.unicode_minus"]=False
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.6,4.2))
    r1=[x for x in out["rows"] if x["C"]==1]
    caps=[x["cap_MB"] for x in r1]; ts=[x["decode_us"] for x in r1]; modes=[x["mode"] for x in r1]
    cols={"all-HBM":"#c1121f","state-on-chip":"#c9a227","all-on-chip":"#2a9d8f"}
    ax.bar([f"{m} ({c}MB)" for m,c in zip(['1x','2x','5x','10x'],caps)],ts,color=[cols[m] for m in modes])
    for i,(t,m) in enumerate(zip(ts,modes)): ax.text(i,t+3,m,ha="center",fontsize=7)
    st=r1[0]["state_MB"]; pa=r1[0]["weights_MB"]
    ax.set_ylabel("decode C=1 block time (µs, ideal 하한)")
    ax.set_title(f"H100 on-chip SRAM 확대 → HOPE block decode (d=2048)\nstate {st:.0f}MB + weights {pa:.0f}MB; 상주하면 HBM 대신 SRAM({SRAM_BW/1e12:.0f}TB/s)",fontsize=9.5)
    ax.grid(axis="y",ls=":",alpha=.4)
    fig.tight_layout(); fig.savefig(os.path.join(HERE,"figures","fig-sram-scaling.png"),dpi=150)
    print("\n-> _sram_scaling.json + fig-sram-scaling.png")


if __name__=="__main__":
    main()
