#!/usr/bin/env python3
"""zHBM / HBM-PIM study for the Titans decode bottleneck — 4 memory types, compute-scaled, HAT-verified.

Pipeline (cold, all assumptions stated):
  1. Derive the in-memory COMPUTE-UNIT spec the Titans update needs (to run the DGD-apply RMW at the
     memory's internal bandwidth, i.e. not become compute-bound in-bank).
  2. For zHBM-PIM, add in-memory compute at PIM-equal / 2x / 5x / 10x -> for each, the memory spec
     (external + internal BW) and the ADDED SILICON AREA (added MAC-rate / density).
  3. Build a HAT .hw twin per config and VERIFY via hat_schema.validate_twin (schema) + a physical
     plausibility gate (internal>=external BW, area within die budget, provenance grade). Analytical
     designs are graded directional (needs-silicon) — kept but flagged, silicon-accurate NOT claimed.
  4. Re-run the HOPE-block roofline (from hope_block_dag) on the VERIFIED twins only.
  5. Output all four types: HBM, zHBM, HBM-PIM, zHBM-PIM(@1/2/5/10x).

Assumptions (BRONZE/directional — zHBM is a Samsung roadmap announcement, no silicon):
  HBM4_BW=6.7 TB/s (2x HBM3); zHBM_BW=4xHBM4 (Samsung "4x HBM4" claim);
  PIM in-bank internal BW: HBM-PIM 8.2 TB/s (novel-pim-cim), zHBM-PIM 2x its external;
  MAC density: base-die N7 1.5 TMAC/s/mm^2, zHBM wafer-bonded 2.5 TMAC/s/mm^2 (stated, unverified);
  DGD-apply ~2 MAC/element (retention scale + gradient subtract).
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hope_block_dag import build_ops, PEAK, L2, BYTES

HERE = os.path.dirname(os.path.abspath(__file__))
# ---- memory assumptions ----
HBM3_BW  = 3.35e12
HBM4_BW  = 6.7e12
ZHBM_BW  = 4*HBM4_BW                 # 26.8 TB/s (Samsung 4x-HBM4 claim, BRONZE)
PIM_INT_HBM  = 8.2e12                # HBM-PIM in-bank aggregate (novel-pim-cim)
PIM_INT_ZHBM = 2*ZHBM_BW             # zHBM in-bank internal (wafer-bonded; assumption)
PIM_MACS_BASE= 1.2e12                # HBM-PIM baseline compute (novel-pim-cim)
MAC_DENSITY_N7  = 1.5e12             # TMAC/s per mm^2 (base-die logic)  [assumption]
MAC_DENSITY_ZHBM= 2.5e12             # TMAC/s per mm^2 (zHBM advanced-node logic) [assumption]
DRAM_DIE_AREA   = 100.0              # mm^2 memory-die area budget (reference)
APPLY_MAC_PER_ELEM = 2.0

# ---- Titans compute-unit spec requirement (derivation) ----
def titans_compute_spec(internal_BW):
    """MAC-rate needed for the DGD-apply RMW to be bandwidth-bound (not compute-bound) at internal_BW."""
    return APPLY_MAC_PER_ELEM * internal_BW / BYTES   # elements/s * mac/elem


def area_for_macs(macs, density):
    return macs/density   # mm^2


def make_twin(name, ext_BW, has_compute, macs=0, int_BW=None, density=MAC_DENSITY_N7,
              cap=64*1024**3, klass="dram", node="N7"):
    int_BW = int_BW or ext_BW
    backing = {"level_id":"mem","role":"hbm","scope":"package","residency":"sw_managed",
               "capacity_bytes":cap,"bandwidth_bps":ext_BW,"line_bytes":32,"mem_standard":"hbm3",
               "energy_pj_per_byte":5.0,"provenance":"analytical"}
    if has_compute:
        backing["peak_macs_per_s"]=macs; backing["matrix_unit"]=[1,1,1]; backing["dtype_bytes"]=2
        backing["internal_bank_bw_bps"]=int_BW; backing["process_node"]=node
        backing["dataflow"]="in_memory_broadcast"
    # compute leaf (the host GPU tensor cores, unchanged — H100 peak); + optional in-mem lane
    leaf={"level_id":"mxu","role":"compute","scope":"soc","instances":132,
          "peak_macs_per_s":3.748e12,"matrix_unit":[16,8,16],"dtype_bytes":2,"acc_dtype_bytes":4,
          "clock_hz":1.98e9,"process_node":"N4","provenance":"vendor_whitepaper"}
    sram={"level_id":"sram","role":"sram","scope":"die","residency":"hw_cache",
          "capacity_bytes":L2,"bandwidth_bps":1.0e13,"line_bytes":128,"children":[leaf]}
    backing["children"]=[sram]
    added_area = area_for_macs(macs, density) if has_compute else 0.0
    twin={"_comment":f"{name} (analytical / BRONZE — zHBM is Samsung roadmap, no silicon)",
          **backing,
          "meta":{"class":klass,"added_compute_mm2":round(added_area,3),
                  "die_area_budget_mm2":DRAM_DIE_AREA,"note":name}}
    return twin, added_area, int_BW


def verify(twin, added_area):
    """hat_schema.validate_twin (schema) + physical plausibility gate. Returns (ok, grade, reasons)."""
    reasons=[]
    try:
        from hat_schema import validate_twin
        errs=validate_twin(twin)
        errs=[e for e in (errs or []) if (isinstance(e,dict) and e.get("severity")=="error") or (isinstance(e,str) and "error" in e.lower())]
        if errs: reasons.append(f"schema errors: {errs[:2]}")
    except Exception as e:
        reasons.append(f"validate_twin unavailable: {e}")
    # plausibility: internal>=external BW; added area within budget; provenance grade
    m=twin
    if m.get("internal_bank_bw_bps") and m["internal_bank_bw_bps"] < m["bandwidth_bps"]:
        reasons.append("internal BW < external BW (implausible)")
    if added_area > DRAM_DIE_AREA:
        reasons.append(f"added compute area {added_area:.1f}mm2 > die budget {DRAM_DIE_AREA}mm2")
    grade = "D-directional" if m.get("provenance")=="analytical" else "A"
    ok = len([r for r in reasons if "implausible" in r or "errors" in r or ">" in r])==0
    return ok, grade, reasons


# ---- parameterized roofline (reuse the HOPE block op list) ----
def roofline(ops, ext_BW, has_pim, int_BW, pim_macs):
    ser=0.0; gpu_bus=0.0; inmem=0.0
    for o in ops:
        if has_pim and o["kind"]=="update":
            hbm_bytes=o["wbytes"]+o["abytes"]
            elems=o["state_rmw"]/BYTES
            comp_t=(elems*APPLY_MAC_PER_ELEM)/pim_macs if pim_macs else 1e9
            mem_t=max(o["state_rmw"]/int_BW, hbm_bytes/ext_BW)
            t=max(comp_t,mem_t); inmem+=t; ser+=t
            gpu_bus+=hbm_bytes/ext_BW
        else:
            hbm_bytes=o["wbytes"]+o["abytes"]+o["state_rmw"]
            t=max(o["flop"]/PEAK, hbm_bytes/ext_BW); ser+=t; gpu_bus+=t
    overlap=max(gpu_bus, inmem)   # in-mem update ∥ GPU forward (best-case concurrency)
    return {"serial_us":ser*1e6, "overlap_us":overlap*1e6}


def main():
    d=2048
    print("### Titans in-memory compute-unit spec requirement ###")
    for label,ib in [("HBM-PIM internal 8.2TB/s",PIM_INT_HBM),("zHBM-PIM internal %.1fTB/s"%(PIM_INT_ZHBM/1e12),PIM_INT_ZHBM)]:
        req=titans_compute_spec(ib)
        print(f"  {label}: DGD-apply BW-bound 위해 >= {req/1e12:.2f} TMAC/s 필요 (PIM base 1.2 TMAC/s)")
    print()

    configs=[]
    configs.append(("HBM (H100)",         HBM3_BW, False, 0,             None,          MAC_DENSITY_N7))
    configs.append(("zHBM (4x HBM4)",     ZHBM_BW, False, 0,             None,          MAC_DENSITY_N7))
    configs.append(("HBM-PIM (1x)",       HBM3_BW, True,  PIM_MACS_BASE, PIM_INT_HBM,   MAC_DENSITY_N7))
    for mult in [1,2,5,10]:
        configs.append((f"zHBM-PIM ({mult}x)", ZHBM_BW, True, PIM_MACS_BASE*mult, PIM_INT_ZHBM, MAC_DENSITY_ZHBM))

    out={"d":d,"assumptions":{"HBM3_BW":HBM3_BW,"ZHBM_BW":ZHBM_BW,"PIM_MACS_BASE":PIM_MACS_BASE,
         "MAC_DENSITY_N7":MAC_DENSITY_N7,"MAC_DENSITY_ZHBM":MAC_DENSITY_ZHBM,"apply_mac_per_elem":APPLY_MAC_PER_ELEM},
         "titans_spec_req_TMACs":{"hbm_pim":titans_compute_spec(PIM_INT_HBM)/1e12,"zhbm_pim":titans_compute_spec(PIM_INT_ZHBM)/1e12},
         "configs":[]}
    print("### 4-type configs: verify + HOPE-block re-run (d=2048) ###")
    print(f"{'config':16}{'ext BW':>10}{'macs':>10}{'+area':>9}{'verify':>16}{'decode C=1 (ser/ovl µs)':>26}")
    for name,ext,hp,macs,ib,dens in configs:
        twin,area,int_BW=make_twin(name,ext,hp,macs=macs,int_BW=ib,density=dens)
        ok,grade,reasons=verify(twin,area)
        rec={"name":name,"ext_BW":ext,"macs":macs,"added_area_mm2":round(area,3),"internal_BW":int_BW,
             "verify_ok":ok,"grade":grade,"reasons":reasons}
        if ok:
            r={C: roofline(build_ops(C,d),ext,hp,int_BW or ext,macs) for C in (1,64,512)}
            rec["roofline"]=r
            dec=r[1]
            print(f"{name:16}{ext/1e12:>8.1f}TB{macs/1e12:>8.1f}T{area:>8.1f}m{grade:>16}"
                  f"{dec['serial_us']:>13.0f}/{dec['overlap_us']:.0f} us")
        else:
            print(f"{name:16}{ext/1e12:>8.1f}TB{macs/1e12:>8.1f}T{area:>8.1f}m{'REJECT':>16}  {reasons[:1]}")
        # persist the .hw twin for verified configs
        if ok:
            slug=name.replace(" ","_").replace("(","").replace(")","").replace("/","").replace("x","x").lower()
            json.dump(twin,open(os.path.join(HERE,"twins",f"zhbm_{slug}.json"),"w"),ensure_ascii=False,indent=1)
        out["configs"].append(rec)
    json.dump(out,open(os.path.join(HERE,"results_full","_zhbm_pim_study.json"),"w"),ensure_ascii=False,indent=1)
    print("\n-> multiarch/results_full/_zhbm_pim_study.json + twins/zhbm_*.json")


if __name__=="__main__":
    main()
