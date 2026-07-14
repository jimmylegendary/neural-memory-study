#!/usr/bin/env python3
"""HOPE block figures: (fig_dag) tensor-op DAG with shapes; (fig_roof) roofline + bottleneck + PIM."""
import json, os
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
_CJK="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_CJK):
    fm.fontManager.addfont(_CJK); matplotlib.rcParams["font.family"]="Noto Sans CJK KR"
matplotlib.rcParams["axes.unicode_minus"]=False
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"figures")
D=json.load(open(os.path.join(HERE,"results_full","_hope_block_dag.json")))
MEM="#c1121f"; CMP="#1a4e8a"; UPD="#c9a227"

# ---------------- fig_dag ----------------
def box(ax,x,y,w,h,txt,fc,ec="#333",fs=7):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.02",fc=fc,ec=ec,lw=1))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs)
def arr(ax,x1,y1,x2,y2,lbl=""):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=9,color="#555",lw=1))
    if lbl: ax.text((x1+x2)/2,(y1+y2)/2+0.012,lbl,ha="center",fontsize=6,color="#444")

def fig_dag():
    fig,ax=plt.subplots(figsize=(12.5,6.8)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.text(0.5,0.975,"HOPE block DAG (B=1, chunk C, dim d) — self-mod Titans → CMS",ha="center",fontsize=10,weight="bold")
    ax.text(0.5,0.94,"파랑=GEMM(큰 C면 compute-bound / 작은 C면 memory-bound) · 금색=state RMW(C 무관, decode 지배, PIM 후보)",ha="center",fontsize=7.5,color="#555")
    # input
    box(ax,0.01,0.55,0.075,0.08,"X\n[C,d]","#eee",fs=8)
    # projections column (parallel, x=0.11)
    proj=[("q=X·Wq [C,d]\n(static)",0.79),("k=M_k(X) [C,d]",0.685),("v=M_v(X) [C,d]",0.58),
          ("η=M_η(X) [C,1]",0.485),("α=M_α(X) [C,1]",0.40)]
    for txt,y in proj:
        box(ax,0.11,y,0.135,0.075,txt,"#dbe7f3",fs=6.5); arr(ax,0.085,0.59,0.11,y+0.037)
    # ---- FORWARD READ path (top, y=0.79) ----
    box(ax,0.30,0.79,0.15,0.08,"o = M_mem(q)\n[C,d] ← pre-update","#dbe7f3",fs=6.5)
    arr(ax,0.245,0.827,0.30,0.83,"q")
    for i,x in enumerate([0.49,0.645,0.80]):
        box(ax,x,0.79,0.135,0.08,f"CMS L{i+1}\n[C,d]→[C,4d]→[C,d]","#dbe7f3",fs=6.2)
    arr(ax,0.45,0.83,0.49,0.83); arr(ax,0.625,0.83,0.645,0.83); arr(ax,0.78,0.83,0.80,0.83)
    box(ax,0.945,0.795,0.045,0.07,"y\n[C,d]","#eee",fs=7); arr(ax,0.935,0.83,0.945,0.83)
    ax.text(0.49,0.885,"CMS = FFN 자리 (3 frequency level; L1만 매 chunk update)",fontsize=6.3,color="#666")
    # ---- UPDATE path (bottom, y=0.15) ----
    ax.text(0.01,0.29,"UPDATE (5개 메모리 {k,v,η,α,mem} 각각; q는 static이라 제외):",fontsize=8,weight="bold",color="#333")
    for txt,x in [("self-target\nv_hat = M(v) [C,d]",0.11),("pred = M(k)\n[C,d]",0.27),("grad ∇L (backward)\n~2× fwd (autograd)",0.43)]:
        box(ax,x,0.14,0.135,0.10,txt,"#dbe7f3",fs=6.5)
    box(ax,0.60,0.12,0.34,0.14,"DGD-apply:  M ← M(αI − η·k·k^T) − η·∇L\n= state read-modify-write  (5×[d,d] weights)\n금색: C 무관, decode 지배 → PIM 후보","#f6ecc7",fs=7)
    arr(ax,0.245,0.19,0.27,0.19); arr(ax,0.405,0.19,0.43,0.19); arr(ax,0.565,0.19,0.60,0.19)
    arr(ax,0.18,0.685,0.18,0.24); arr(ax,0.18,0.58,0.245,0.24)   # k,v feed update
    ax.text(0.60,0.085,"→ 새 state M_t (다음 토큰/청크용). 이번 토큰 출력 o_t 와 의존성 없음 → 겹칠 수 있음(§overlap)",fontsize=6.3,color="#666")
    # cost legend
    ax.text(0.01,0.035,"op 비용: projection 4Cd² · CMS level 16Cd² · update/memory ~16Cd² · DGD-apply = 16d² params RMW(C 무관).  "
                       "forward/gradient GEMM은 FLOP∝C, state RMW는 고정 → C*가 memory↔compute 경계.",fontsize=6.5,color="#444")
    fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig-hope-dag.png"),dpi=150); plt.close(fig)

# ---------------- fig_roof ----------------
def fig_roof():
    Cs=sorted(D["C"].keys(), key=int); ridge=D["twin"]["RIDGE"]
    fig,axs=plt.subplots(1,3,figsize=(12.5,3.9))
    # (a) block AI vs C
    ai=[D["C"][c]["baseline"]["flop"]/D["C"][c]["baseline"]["bytes"] for c in Cs]
    axs[0].plot([int(c) for c in Cs],ai,"o-",color="#2a6f97")
    axs[0].axhline(ridge,color=MEM,ls="--",label=f"ridge={ridge:.0f}")
    axs[0].set_xscale("log");axs[0].set_yscale("log");axs[0].set_xlabel("chunk C");axs[0].set_ylabel("block AI (FLOP/B)")
    axs[0].set_title("(a) roofline 위치 vs C\nAI<ridge=memory / >ridge=compute",fontsize=9);axs[0].legend(fontsize=7);axs[0].grid(ls=":",alpha=.4)
    # (b) C=1 bottleneck breakdown
    ops=D["C"]["1"]["baseline"]["ops"]; ops=sorted(ops,key=lambda o:-o["t_us"])[:7]
    names=[o["name"].split("(")[0][:22] for o in ops]; ts=[o["t_us"] for o in ops]
    cols=[MEM if o["bound"]=="memory" else CMP for o in ops]
    for i,o in enumerate(ops):
        if o["kind"]=="update": cols[i]=UPD
    axs[1].barh(range(len(ops))[::-1],ts,color=cols); axs[1].set_yticks(range(len(ops))[::-1]); axs[1].set_yticklabels(names,fontsize=6.5)
    axs[1].set_xlabel("time (µs)");axs[1].set_title("(b) C=1(decode) 병목 op\n금색=state RMW",fontsize=9);axs[1].grid(axis="x",ls=":",alpha=.4)
    # (c) baseline vs PIM per C
    xb=[int(c) for c in Cs]
    axs[2].plot(xb,[D["C"][c]["baseline"]["t_us"] for c in Cs],"o-",color="#333",label="baseline (HBM)")
    axs[2].plot(xb,[D["C"][c]["pim"]["t_us"] for c in Cs],"s--",color=UPD,label="HBM-PIM (serial)")
    axs[2].set_xscale("log");axs[2].set_xlabel("chunk C");axs[2].set_ylabel("block time (µs)")
    axs[2].set_title("(c) baseline vs HBM-PIM\n(직렬; PIM 이득은 concurrency 필요)",fontsize=9);axs[2].legend(fontsize=7);axs[2].grid(ls=":",alpha=.4)
    fig.suptitle("HOPE block 1개 roofline (H100 twin: BW 3.35TB/s, peak 0.99PF, ridge 295) — d=2048",fontsize=10)
    fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(os.path.join(FIG,"fig-hope-roofline.png"),dpi=150); plt.close(fig)

fig_dag(); fig_roof()
print("-> figures/fig-hope-dag.png, fig-hope-roofline.png")
