# -*- coding: utf-8 -*-
import json, re, os
REPO = "/home/jimmy/repos/neural-memory-study"
OUT = "/tmp/claude-1000/-home-jimmy--claude-workspace/d13e378b-5c3b-41e3-9ece-e09e962f2c96/tasks/wc5wbh5oj.output"
secs = {s["key"]: s for s in json.load(open(OUT))["result"]["sections"]}
corpus = {x["arxiv"]: x["name"] for x in json.load(open(f"{REPO}/research/ttt-efficient-corpus.json")) if x.get("arxiv")}

SHORT = {
 "2407.04620":"TTT","2501.12352":"Test-time Regression","2505.23884":"LaCT","2501.00663":"Titans",
 "2505.23735":"Atlas","2511.07343":"TNT","2512.24695":"HOPE","2606.03979":"Sleep","2504.13173":"Miras",
 "2605.28053":"RW-TTT","2411.19379":"Marconi","2604.06169":"In-Place TTT","2506.05233":"MesaNet",
 "2406.06484":"DeltaNet","2405.21060":"Mamba-2","2312.00752":"Mamba","2503.14456":"RWKV-7",
 "2504.03624":"Nemotron-H","2602.21204":"TTT=Linear Attn","2102.11174":"Fast Weight Programmers",
 "2006.16236":"Linear Transformers","2307.08621":"RetNet","2312.06635":"GLA",
}
MANUAL = {
 "2008.02217":"Hopfield Networks is All You Need","2309.12252":"DEER: Parallelizing non-linear sequential models",
 "2309.05858":"Uncovering Mesa-Optimization Algorithms in Transformers","2212.07677":"Transformers Learn In-Context by Gradient Descent",
 "2410.01201":"Were RNNs All We Needed? (minGRU/minLSTM)","2408.10189":"Transformers to SSMs (MOHAWK/Phi-Mamba)",
}
def fullname(i): return corpus.get(i) or MANUAL.get(i) or SHORT.get(i) or "(제목 확인 필요)"

def clean(md):
    md = re.sub(r'<!--\s*검증노트.*?-->', '', md, flags=re.S)
    md = re.sub(r'<!--.*?-->', '', md, flags=re.S)
    i = md.find('### ')
    if i > 0: md = md[i:]
    return md.strip()

# ---- framing ----
HEADER = r"""% assembled
"""
INTRO = """# 0. 들어가며 — 목적 · 범위 · 읽는 법

이 글은 **TTT(test-time training)와 Titans 계열의 '효율과 scaling'을 논문의 그림·표·수치를 근거로 깊이 파고드는 집중 조명**이다. 세 가지 scaling 병목 — ① model-size, ② training의 parallel/batch, ③ serving의 batch inference(Prefill/Decode 포함) — 각각에 대해 (1) 병목이 왜 생기는지, (2) 그것을 푸는 논문들이 무엇을·어떻게·왜 working하는지, (3) 이 모든 것이 만나는 **memory device**의 한계와 개선 해법을 다룬다.

**세 가지 caveat.**

1. 이 글은 pre-training 레시피나 아키텍처 세부가 아니라 **'효율·scaling 병목과 해법'의 시스템 관점**이다. 같은 논문도 여기서는 효율 축만 조명한다.
2. 다루는 논문 상당수가 2025~2026년 것이라 **수치·재현성이 미확정**인 경우가 많다. 문장마다 근거등급을 달았고 모든 수치는 best-effort 검증이다(다중 에이전트 심층 집필 → 원문·그림 대조 사실검증).
3. **단일 논문으로는 그림이 불완전하다.** 병목과 해법, 이론과 시스템을 함께 봐야 이해된다. 서로 다른 갈래(distillation·병렬화·서빙)는 억지로 하나로 봉합하지 않았다.

**읽는 법.** 근거등급은 **[a]** 논문 명시 · **[b]** 메커니즘상 추론 · **[c]** 미명시·불확실. 인용은 대괄호 번호 **[N]**(끝의 References). 그림은 각 논문의 원문 Figure를 그대로 인용했으며 캡션에 출처·Figure 번호·[N]을 밝혔다(제3자 그림 — 저작권은 원저자, 재사용 시 확인 필요). LaCT의 "~70% GPU utilization"은 원문 보고 수치이나 실측 하드웨어(A100/H100) 표기가 소스마다 엇갈려 하드웨어 세부는 [b]로 둔다.

**구조.** Part I(1–2장) TTT·Titans 집중 조명 → Part II(3–5장) 세 병목 → Part III(6–8장) 세 해법 → Part IV(9–10장) memory device 한계·개선 → 11장 종합 지도 → 12장 열린 질문 → References.
"""

WRAP = {
 "I":"> **Part I 정리.** TTT/Titans의 표현력은 '추론 중 self-supervised gradient로 fast weight를 갱신'하는 데서 오고, 바로 그 **순차 갱신**이 이후 세 병목의 공통 뿌리다. LaCT가 정량화한 FLOPs util<5%가 그 증거다.",
 "II":"> **Part II 정리.** 세 병목은 한 뿌리의 세 그림자다 — **파라미터로 안 커지고**(유한 메모리 용량), **학습이 병렬화가 안 되고**(순차 inner-loop·낮은 util), **서빙이 배치가 안 된다**(request-owned mutable state).",
 "III":"> **Part III 정리.** 해법도 세 방향 — (규모) 큰 Transformer를 **distill로 물려받기**, (학습) 순차성을 **큰 chunk·병렬화로 접기**, (서빙) **배치·캐시 프리미티브 자체를 재설계**. 공통 조건은 '표현력을 지키며 순차성·용량 비용을 줄인다'.",
 "IV":"> **Part IV 정리.** memory device는 세 병목이 만나는 물리적 결절 — **용량·간섭·대역폭**. 개선은 '더 크게(sparse/expandable)·더 정확히(exact)·더 싸게(quantize/in-place)'의 세 축이며 각 축이 특정 병목을 직접 겨눈다.",
}
PARTS = [
 ("Part I — 집중 조명", [("1","ttt"),("2","titans")], "I"),
 ("Part II — Scaling 병목 3종", [("3","bn-model"),("4","bn-train"),("5","bn-serve")], "II"),
 ("Part III — 해법 3종 (무엇을 · 어떻게 · 왜 working)", [("6","sol-model"),("7","sol-train"),("8","sol-serve")], "III"),
 ("Part IV — memory device 집중 조명", [("9","mem-lim"),("10","mem-sol")], "IV"),
]
SYNTH = """# 11. 종합 — 세 병목 × 해법 × memory device 지도

| 병목 | 근본 원인 | 대표 해법(무엇을) | 왜 working | memory device 연결 |
|---|---|---|---|---|
| ① model-size | 문맥을 파라미터가 아니라 유한 메모리 용량으로 저장(d²에 ~d) | distill/linearize로 규모 상속, 용량 확장, 파라미터 성장 | 규모는 물려받고 용량은 feature/state로 키움 | **용량 상한**이 진원 |
| ② training | inner-loop 순차 의존 → 작은 chunk면 FLOPs util<5% | 큰 chunk(LaCT), chunkwise-parallel, 비선형 recurrence 병렬화 | 순차성을 chunk 경계로 밀거나 병렬 형태로 접음 | **state footprint**가 chunk·batch 제약 |
| ③ serving | request마다 mutable state 소유 → batching·prefix-cache 붕괴 | RW-TTT, hybrid prefix caching, In-Place, IO-aware | 재사용·배치 프리미티브를 recurrent에 맞게 재설계 | **대역폭·footprint**가 decode bound |

관통하는 한 줄: **모든 효율 이득은 결국 memory device의 유한한 용량·대역폭과 거래한다.** 표현력을 위해 state를 키우면 training FLOPs·serving footprint가 오르고, 이를 줄이려 압축·양자화하면 용량·정확도가 깎인다. 프론티어는 세 꼭짓점을 '동시에' 미는 것 — LaCT(큰 chunk+큰 state), In-Place TTT(state 없이 기존 가중치 재활용), RW-TTT(요청별 state 배치 서빙) — 이나 이들은 아직 서로 다른 갈래다.
"""
OPENQ = """# 12. 열린 질문

1. **큰 chunk vs 세밀한 적응.** LaCT의 2K~1M chunk는 util을 살리지만 chunk 내부의 순차 의존을 지연시킨다. 정확도 손실 없이 둘을 동시에 얻는 chunk 스케줄의 이론적 상한은? [c]
2. **TTT ≈ linear attention 등가의 함의.** 복잡한 다층-MLP·momentum TTT조차 학습된 linear attention operator로 등가 재작성된다면, "test-time 학습"의 표현력 우위는 실제로 어디서 오는가? [c]
3. **용량 벽.** d²에 ~d라는 associative-memory 용량 한계를 sparse/expandable state가 상수배 개선인지, 스케일링 지수 자체를 바꾸는지. [c]
4. **request-owned state 서빙의 스케일.** RW-TTT가 수천 동시 요청에서도 성립하나 — 요청별 fast weight의 메모리 폭발과 phase 호환 배치의 한계. [b]
5. **hybrid recurrent state의 표준 캐시 프리미티브.** Marconi·HYPIC·Sparse Prefix가 제각각인데 KV cache에 준하는 표준이 나올 수 있나? [c]
6. **scratch 대규모 학습의 길.** model-size 병목이 distillation으로만 실용적으로 풀린다면, TTT/Titans를 처음부터 초대형으로 pre-train하는 경로는 닫힌 것인가? [c]
7. **재현성.** Titans Revisited의 지적처럼, neural memory의 실이득이 어느 규모·태스크에서 baseline을 확실히 넘는지 아직 합의가 없다. [a]
8. **물리-알고리즘 co-design.** memory device의 물리(HBM 용량·대역폭)와 알고리즘(state 용량·chunk)을 함께 모델링하는 비용 프레임으로 무엇을 예측할 수 있나? [c]
"""

# ---- figure + citation processing ----
cite_order = []
def note_cite(i):
    if i not in cite_order: cite_order.append(i)

figc = re.compile(r'\[FIG:([^\]|/]+)/([^\]|]+)(?:\|([^\]]*))?\]')
def fig_repl_factory(chnum):
    counter = {"k": 0}
    def repl(m):
        lab, fig, cap = m.group(1), m.group(2), (m.group(3) or "").strip()
        if lab == "..." or "/" in lab:
            return ""
        path = f"{REPO}/reffigs/{lab}/figures/{fig}.png"
        if not os.path.exists(path):
            return ""
        counter["k"] += 1
        aid = lab.replace("ext-", "")
        note_cite(aid)
        n = re.sub(r'[^0-9].*$', '', fig.replace("fig", ""))
        short = SHORT.get(aid, aid)
        capfull = f"그림 {chnum}.{counter['k']}. {cap} (출처: {short}, Figure {n}, {{{{{aid}}}}})"
        return f"\n\n![{capfull}]({path}){{width=78%}}\n\n"
    return repl

def process_chapter(md, chnum):
    md = clean(md)
    md = figc.sub(fig_repl_factory(chnum), md)
    return md

# assemble
doc = [INTRO]
for ptitle, items, wk in PARTS:
    doc.append(f"\n# {ptitle}\n")
    for num, key in items:
        doc.append(f"\n## {num}. {secs[key]['title']}\n")
        doc.append(process_chapter(secs[key]["markdown"], num))
    doc.append("\n" + WRAP[wk] + "\n")
doc.append("\n" + SYNTH)
doc.append("\n" + OPENQ)
body = "\n".join(doc)

# citation pass: {{id}} -> [N] in order of appearance (figures already inserted {{id}})
seen = []
def cnum(m):
    i = m.group(1)
    if i not in seen: seen.append(i)
    return f"[{seen.index(i)+1}]"
body = re.sub(r'\{\{(\d{4}\.\d{4,5})\}\}', cnum, body)

# references
refs = ["\n\n# References\n"]
for k, i in enumerate(seen, 1):
    refs.append(f"{k}. {fullname(i)}. arXiv:{i}. <https://arxiv.org/abs/{i}>")
body += "\n".join(refs)

# sanitize unicode sub/superscripts for lualatex mono font
for a, b in {'ₜ':'t','ₖ':'k','ᵥ':'v','ₘ':'m','ₐ':'a','ₙ':'n','ᵀ':'T','ᵗ':'t','₋':'-','₀':'0','₁':'1','₂':'2','₃':'3','⁻':'-','⟂':'⊥','▪':'-'}.items():
    body = body.replace(a, b)

open(f"{REPO}/research/ttt-titans-efficiency-study.md", "w").write(body)
open("/tmp/claude-1000/-home-jimmy--claude-workspace/d13e378b-5c3b-41e3-9ece-e09e962f2c96/scratchpad/book.md", "w").write(body)
print("assembled", len(body), "chars;", len(seen), "references;", body.count("!["), "figures")
