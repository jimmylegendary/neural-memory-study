#!/usr/bin/env python3
"""Seminar deck for the TTT+Titans efficiency study paper (research/ttt-titans-efficiency-study.md).
~21 slides: TTT/Titans spotlight -> 3 bottlenecks -> 3 solutions -> memory device -> synthesis/open Q."""
import os
from pptx_lib import (deck, title_slide, section_divider, slide, box, text,
                      INK, MUTE, BLUE, BLUEB, RED, REDB, GOLD, GOLDB, GREEN, GREENB,
                      GREY, GREYB, WHITE, SW)
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
try:
    from mathpng import eq
    HAVE_EQ = True
except Exception:
    HAVE_EQ = False

L, C = PP_ALIGN.LEFT, PP_ALIGN.CENTER
HERE = os.path.dirname(os.path.abspath(__file__))
p = deck()


def bullets(s, items, x=0.65, y=1.35, w=12.1, size=15, gap=6, color=INK):
    """items: list of (text, opts?) ; opts may set color/bold/size/sub(bool)."""
    lines = []
    for it in items:
        opts = {}
        if isinstance(it, tuple):
            it, opts = (it[0], it[1]) if len(it) == 2 else (it[0], {})
        sub = opts.get("sub", False)
        bullet = "   – " if sub else "•  "
        lines.append((bullet + it, {"size": opts.get("size", size - (1.5 if sub else 0)),
                                     "color": opts.get("color", MUTE if sub else color),
                                     "bold": opts.get("bold", False),
                                     "line_spacing": 1.28, "space_after": opts.get("sa", gap)}))
    text(s, x, y, w, 5.6, lines, align=L, anchor=MSO_ANCHOR.TOP)


def panel(s, x, y, w, h, head, body, fc, ec):
    box(s, x, y, w, h,
        [(head, {"size": 14.5, "bold": True, "color": ec, "align": L, "space_after": 5})]
        + [(b, {"size": 12.5, "color": INK, "align": L, "space_after": 3}) for b in body],
        fc=fc, ec=ec, align=L, anchor=MSO_ANCHOR.TOP)


# ---------- S1 title ----------
title_slide(p, "TTT와 Titans의 효율",
            "Scaling 병목과 해법, 그리고 memory device 집중 조명   ·   근거등급  a=명시 · b=추론 · c=미명시",
            "neural-memory-study · 집중 조명 study paper · 2026-07-16")

# ---------- S2 들어가며 ----------
s = slide(p, "0 · 들어가며", "목적 · 범위 · 읽는 법", GREY)
panel(s, 0.55, 1.35, 7.7, 4.9, "목적",
      ["TTT / Titans 계열을 '효율 · scaling' 시스템 관점으로 집중 조명",
       "세 scaling 병목(model-size · training batch · serving P/D)과",
       "각 해법(무엇을 · 어떻게 · 왜 working), 그리고 memory device 한계 · 개선"], GREYB, GREY)
panel(s, 8.45, 1.35, 4.33, 2.3, "3가지 caveat",
      ["① 아키텍처 세부가 아니라 효율·scaling 관점",
       "② 2026 최신 多 → 수치·재현성 미확정 → 근거등급",
       "③ 단일 논문은 불완전 — 갈래는 갈래대로"], REDB, RED)
panel(s, 8.45, 3.8, 4.33, 2.45, "흐름",
      ["Ⅰ TTT·Titans 집중 조명",
       "Ⅱ 병목 3종  →  Ⅲ 해법 3종",
       "Ⅳ memory device 한계·개선",
       "→ 종합 지도 · 열린 질문"], BLUEB, BLUE)

# ================= PART I =================
section_divider(p, "Part Ⅰ", "TTT · Titans 집중 조명", BLUE)

# S4 TTT 정의
s = slide(p, "1 · TTT 집중 조명", "은닉상태가 곧 '학습 중인 모델'", BLUE)
bullets(s, [
    ("은닉상태를 작은 모델의 weights W_t 로 보고, 매 토큰 self-supervised gradient 1스텝으로 갱신  [a]",),
    ("출력은 그 갱신된 모델의 forward:  z_t = f_{W_t}(q_t).  f 가 선형=TTT-Linear, 2-layer MLP=TTT-MLP  [a]", {"sub": True}),
    ("메모리 용량이 '벡터 크기'가 아니라 '모델 가중치 수'로 스케일  [a]",),
    ("→ 문맥이 길어질수록 손실 계속 감소 (Mamba는 ~16k에서 정체)  [a]", {"sub": True}),
    ("뿌리: fast-weight programmer(Schlag) — linear attention ≡ fast weight, delta rule = gradient step  [a]",),
], y=1.35)
if HAVE_EQ:
    try:
        path, w_in, h_in = eq(r"W_t = W_{t-1}-\eta\,\nabla_W\,\ell(W_{t-1};x_t),\quad \ell=\lVert f_W(k_t)-v_t\rVert^2", pt=18)
        from pptx_lib import fit_image
        box(s, 0.55, 5.55, 12.23, 1.2, "", fc=WHITE, ec=GREY)
        fit_image(s, path, 0.7, 5.62, 11.9, 1.06)
    except Exception:
        pass

# S5 순차성=병목 뿌리
s = slide(p, "1 · TTT 집중 조명", "순차성 = 모든 병목의 뿌리", BLUE)
panel(s, 0.55, 1.35, 6.05, 4.9, "이중성(duality)이 효율의 열쇠",
      ["같은 연산의 세 형태:",
       "  parallel(학습) · recurrent(O(1) 추론) · chunkwise",
       "TTT의 dual form: 청크 내 gradient들을 하나의",
       "큰 행렬곱으로 묶음 (primal 대비 >5×, JAX 기준) [a]",
       "청크 내부는 병렬, W의 이월은 청크 경계에서만 [b]"], BLUEB, BLUE)
panel(s, 6.75, 1.35, 6.03, 4.9, "그러나 순차성은 남는다",
      ["W_t 는 W_{t-1}에 의존 → 본질적으로 순차 [b]",
       "chunkwise는 순차성을 '청크 경계로 밀어낼' 뿐",
       "작은 청크(16–64토큰) → FLOPs util < 5% (LaCT) [a]",
       "큰 청크 → util↑ 이나 세밀한 적응력↓ (trade-off) [b]",
       "→ 이 trade-off가 이후 모든 병목을 낳는다"], REDB, RED)

# S6 Titans 메커니즘
s = slide(p, "2 · Titans 집중 조명", "deep neural memory + surprise·momentum·forget", GOLD)
bullets(s, [
    ("long-term memory M = deep MLP; 각 토큰에서 M의 가중치를 test time에 갱신  [a]",),
    ("갱신 신호 = associative loss ‖M(k_t)−v_t‖² 의 gradient = 'surprise'  [a]", {"sub": True}),
    ("여기에 momentum(과거 surprise 누적) + weight-decay(adaptive forget) 결합  [a]", {"sub": True}),
    ("3가지 통합 방식: MAC(memory as context) · MAG(as gate) · MAL(as layer)  [a]",),
    ("attention(단기 정확) + neural memory(장기 압축)의 하이브리드  [a]",),
    ("chunk-parallel 학습으로 순차 갱신을 가속  [a]", {"sub": True}),
], y=1.35)

# S7 Titans 효율 라인
s = slide(p, "2 · Titans 집중 조명", "효율 라인 Atlas·TNT·HOPE·Sleep — 그리고 비판", GOLD)
panel(s, 0.55, 1.35, 6.05, 4.9, "효율·표현력 확장 라인",
      ["Atlas: Omega rule(문맥 단위)+Muon(2차)+다항 feature 용량 [a]",
       "TNT: chunkwise training 가속 — global chunk + local 병렬 [a]",
       "HOPE / Nested Learning: optimizer=memory · CMS ·",
       "         self-modifying Titans [a]",
       "Sleep: offline consolidation — parameter 성장 +",
       "         Knowledge Seeding + Dreaming(RL) [a]"], GOLDB, GOLD)
panel(s, 6.75, 1.35, 6.03, 4.9, "⚠ skeptic — Titans Revisited",
      ["경량 재구현으로 재현성·실기여를 비판적 검증 [a]",
       "neural memory는 유효하나,",
       "chunking 때문에 baseline을 '항상' 넘지는 못함 [a]",
       "",
       "→ neural memory의 실이득이 어느 규모·태스크에서",
       "   확실한지는 아직 합의 없음 [a]"], REDB, RED)

# ================= PART II =================
section_divider(p, "Part Ⅱ", "Scaling 병목 3종", RED)

# S9 병목1
s = slide(p, "3 · 병목 ①", "model-size scaling — 파라미터가 아니라 '용량'", RED)
bullets(s, [
    ("TTT/Titans는 문맥을 '파라미터'가 아니라 '메모리 용량'으로 저장한다",),
    ("용량 이론: d² 파라미터를 가진 메모리에 ~d 규모의 정보만 안정 저장 (associative/Hopfield)  [b]", {"sub": True}),
    ("Chunk mismatch: 청크 단위 갱신이 규모에서 성능을 열화 (Titans Revisited)  [a]",),
    ("고정 용량 recurrent state의 긴 문맥 정보이론적 벽 — 길수록 간섭 누적  [b]", {"sub": True}),
    ("hybrid attention 비율 의존 + scaling-law 상 표현력↔효율 tradeoff  [a]",),
    ("→ 단순히 파라미터를 키운다고 문맥 기억이 비례해 늘지 않는다  [b]", {"color": RED, "bold": True}),
], y=1.4)

# S10 병목2
s = slide(p, "4 · 병목 ②", "training parallel/batch scaling", RED)
bullets(s, [
    ("inner-loop 갱신의 순차 의존 → 시퀀스 방향 병렬화(sequence-parallel)가 어렵다  [b]",),
    ("작은 online minibatch(16–64토큰) → GPU FLOPs 이용률이 흔히 < 5%  [a]", {"sub": True}),
    ("backprop-through-the-inner-loop: 갱신 궤적을 미분하는 학습의 메모리·계산 비용  [b]",),
    ("batch × state footprint: 표현력 위해 state를 키우면 학습 배치가 눌린다  [b]", {"sub": True}),
    ("비선형 recurrence의 병렬화는 가능하나 조건부(수렴·안정성)  [b]",),
    ("→ 알고리즘이 아니라 '하드웨어 활용률'이 학습 스케일의 병목  [b]", {"color": RED, "bold": True}),
], y=1.4)

# S11 병목3
s = slide(p, "5 · 병목 ③", "serving batch inference + Prefill/Decode(P/D)", RED)
bullets(s, [
    ("batching은 '요청들이 공유하는 static weight'를 전제 — TTT는 이를 깬다  [a]",),
    ("request마다 mutable fast-weight state(low-rank delta/learner state)를 소유  [a]", {"sub": True}),
    ("recurrent state는 KV cache처럼 prefix-cache(재사용)되지 않는다  [a]",),
    ("prefill(대량 병렬 흡수) vs decode(순차 갱신, state 대역폭 bound)의 비대칭  [b]", {"sub": True}),
    ("hybrid(attention+recurrent)에서 P/D disaggregation이 더 어렵다  [b]",),
    ("→ 모델은 빨라도 '동시 요청 배치'가 안 되면 서빙 처리량이 안 는다  [b]", {"color": RED, "bold": True}),
], y=1.4)

# ================= PART III =================
section_divider(p, "Part Ⅲ", "해법 3종 — 무엇을 · 어떻게 · 왜 working", GREEN)

# S13 해법1
s = slide(p, "6 · 해법 ①", "model-size — 규모를 '물려받고' 용량을 키운다", GREEN)
panel(s, 0.55, 1.35, 6.05, 4.9, "무엇을 · 어떻게",
      ["규모 상속: 큰 Transformer를 distill/linearize",
       "  MOHAWK·Llamba·LoLCATs(405B)·Liger [a]",
       "용량 확장: Atlas 다항 feature, sparse/expandable state",
       "성장: Sleep(param expansion+Knowledge Seeding",
       "        +Dreaming), HOPE nested depth",
       "hybrid: attention 일부 남겨 규모 병목 우회"], GREENB, GREEN)
panel(s, 6.75, 1.35, 6.03, 4.9, "왜 working",
      ["규모는 사전학습된 Transformer의 것을 물려받아",
       "  scratch 대규모 학습 없이 405B급 도달 [a]",
       "용량은 feature/state 차원을 키워 d²→d 벽을 완화 [b]",
       "성장은 필요할 때 파라미터를 늘려 고정 용량을",
       "  넘어섬 (offline consolidation) [a]",
       "한계: distill은 '남의 규모'에 의존 [b]"], BLUEB, BLUE)

# S14 해법2
s = slide(p, "7 · 해법 ②", "training — 순차성을 chunk·병렬화로 접는다", GREEN)
panel(s, 0.55, 1.35, 6.05, 4.9, "무엇을 · 어떻게",
      ["LaCT: 극단적 large chunk(2K–1M토큰) [a]",
       "  → util을 orders-of-mag 개선,",
       "     nonlinear state를 파라미터의 40%까지 [a]",
       "chunkwise-parallel: DeltaNet WY, TNT 계층",
       "비선형 recurrence 병렬화: DEER→ParaRNN,",
       "  Predictability (Newton/fixed-point/scan)",
       "커널: Comba(Triton, +40%), Tiled FLA"], GREENB, GREEN)
panel(s, 6.75, 1.35, 6.03, 4.9, "왜 working",
      ["큰 chunk는 갱신을 compute-bound로 만들어",
       "  TensorCore를 채운다(util↑) [a]",
       "WY/householder는 delta 재귀를 하나의",
       "  행렬 연산으로 재작성 → chunk 병렬 [a]",
       "비선형 병렬화는 순차 재귀를 병렬 solve로 대체 [b]",
       "trade-off: 큰 chunk는 청크 내 세밀 적응을 지연 [b]"], BLUEB, BLUE)

# S15 해법3
s = slide(p, "8 · 해법 ③", "serving — 배치·캐시 프리미티브를 재설계", GREEN)
panel(s, 0.55, 1.35, 6.05, 4.9, "무엇을 · 어떻게",
      ["RW-TTT: owner/version/READ-WRITE 태깅으로",
       "  호환 phase만 batch, write는 owner에만 commit [a]",
       "hybrid prefix caching: Marconi·HYPIC·Sparse Prefix",
       "IO-aware 서빙: KVBuffer",
       "최소 state: In-Place TTT(별도 state 제거)",
       "hybrid stack: Nemotron·Kimi Linear (KV↓)"], GREENB, GREEN)
panel(s, 6.75, 1.35, 6.03, 4.9, "왜 working",
      ["RW-TTT: In-Place-TTT 8스트림에서 274.61 tok/s",
       "  = 순차 9.31× · 동일메모리 복제 3.44× [a]",
       "재사용 프리미티브를 recurrent state에 맞게 재정의",
       "  → Marconi 최대 34.4× 캐시 이득 [a]",
       "In-Place는 요청별 state 부담 자체를 줄임 [b]",
       "한계: hybrid prefix-cache 표준은 아직 없음 [c]"], BLUEB, BLUE)

# ================= PART IV =================
section_divider(p, "Part Ⅳ", "memory device 집중 조명", GOLD)

# S17 memory device 한계
s = slide(p, "9 · memory device", "한계 — 하나의 장치, 두 상한", GOLD)
panel(s, 0.55, 1.35, 6.05, 4.9, "(i) 신경 메모리 (저장 장치로서)",
      ["유한 용량: d² 파라미터에 ~d 정보 (Hopfield/assoc) [b]",
       "interference: key 충돌 시 덮어쓰기·간섭 [b]",
       "retrieval error가 저장량 증가에 따라 성장 [b]",
       "고정 용량 → 긴 문맥에서 정보이론적 열화 [b]"], GOLDB, GOLD)
panel(s, 6.75, 1.35, 6.03, 4.9, "(ii) physical memory (HBM)",
      ["state footprint의 HBM 점유 [b]",
       "decode가 state 읽기의 memory-bandwidth에 bound [b]",
       "batch × state → 동시성의 메모리 상한 [b]",
       "",
       "→ 세 병목(model-size·training·serving)이",
       "   만나는 물리적 결절점 [b]"], REDB, RED)

# S18 memory device 개선
s = slide(p, "10 · memory device", "개선 — 더 크게 · 더 정확히 · 더 싸게", GOLD)
bullets(s, [
    ("더 크게(용량↑, FLOPs는 묶음): Sparse State Expansion · Sparse Delta Memory [a]",),
    ("더 정확히(압축 대신 정확 저장): exact memory — Hippocampus for Linear Attention · EFLA [a]",),
    ("자라는 메모리: KV-Means · Memory Caching(growing) [a]",),
    ("간섭↓: erase-write 주소 분리(Gated DeltaNet-2, Erase-then-Delta) [a]", {"sub": True}),
    ("용량 자체↑: 고용량 feature map(Atlas 다항 · sympow); 압축(Lattice · Trellis · LoLA) [a]", {"sub": True}),
    ("물리 개선: TTQ(양자화로 footprint↓) · KVBuffer(대역폭) · In-Place(별도 state 제거) [a]",),
    ("→ 각 개선 축이 특정 병목을 직접 겨냥한다  [b]", {"color": GOLD, "bold": True}),
], y=1.35)

# ================= 종합 / 열린질문 / 마무리 =================
s = slide(p, "11 · 종합", "병목 × 해법 × memory device 지도", GREY)
rows = [
    ("병목", "근본 원인", "대표 해법", "memory device", GREYB, GREY, True),
    ("① model-size", "용량 d²→d", "distill · 용량확장", "용량 상한", WHITE, GREY, False),
    ("② training", "순차 → util<5%", "큰 chunk · 병렬화", "state footprint", WHITE, GREY, False),
    ("③ serving", "request-owned state", "RW-TTT · prefix cache", "대역폭 · footprint", WHITE, GREY, False),
]
xs = [0.55, 3.0, 6.1, 9.5]
ws = [2.35, 3.0, 3.3, 3.28]
y0 = 1.5
for ri, (a, b, c, d, fc, ec, hd) in enumerate(rows):
    yy = y0 + ri * 0.86
    for ci, val in enumerate((a, b, c, d)):
        box(s, xs[ci], yy, ws[ci], 0.78, val, fc=(GREYB if hd else WHITE), ec=GREY,
            size=12.5 if not hd else 13, bold=hd, tcolor=(ec if (hd or ci == 0) else INK), align=L)
text(s, 0.55, 5.35, 12.2, 1.4,
     [("관통하는 한 줄:  모든 효율 이득은 결국 memory device의 유한한 용량·대역폭과 거래한다.",
       {"size": 15, "bold": True, "color": INK}),
      ("프론티어 = 큰 chunk(LaCT) · 최소 state(In-Place) · 요청별 서빙(RW-TTT)을 '동시에' — 아직 갈래는 분리돼 있다.",
       {"size": 13, "color": MUTE})], align=L)

s = slide(p, "12 · 열린 질문", "무엇이 아직 안 풀렸나", GREY)
bullets(s, [
    ("큰 chunk(util) vs 세밀한 순차 적응(정확도)을 동시에? chunk 내부 의존 손실의 이론 상한  [c]",),
    ("TTT ≈ linear attention 등가라면, 'test-time 학습'의 표현력 우위는 실제로 어디서 오나?  [c]",),
    ("d²→d 용량 벽: sparse/expandable이 상수 개선인가, 스케일 지수 자체를 바꾸나?  [c]",),
    ("request-owned state 서빙이 수천 동시 요청에서 스케일하나? (메모리 폭발)  [b]",),
    ("hybrid recurrent state의 표준 prefix-cache 프리미티브는?  [c]",),
    ("model-size가 distill로만 풀린다면, scratch 대규모 학습 경로는 닫힌 것인가?  [c]",),
    ("재현성(Titans Revisited) — 실이득이 어느 규모·태스크에서 확실한가?  [a]",),
    ("물리(HBM)×알고리즘(state) co-design(HATIR류 비용모델)로 무엇을 예측하나?  [c]",),
], y=1.35, gap=5)

s = slide(p, "마무리", "한 장 요약", GREEN)
bullets(s, [
    ("TTT/Titans의 표현력 = '순차적 test-time 갱신' — 바로 그 순차성이 모든 병목의 뿌리",),
    ("세 병목은 한 뿌리의 세 그림자: 파라미터로 안 커지고 · 학습이 병렬화 안 되고 · 서빙이 배치 안 됨",),
    ("해법 세 방향: 규모 상속(distill) · 순차성 접기(큰 chunk·병렬화) · 프리미티브 재설계(서빙)",),
    ("memory device = 세 병목이 만나는 물리적 결절; 개선 3축 = 더 크게 · 더 정확히 · 더 싸게",),
    ("근거: study paper(research/ttt-titans-efficiency-study.md) + efficient-TTT 93편 코퍼스",
     {"color": MUTE, "size": 13}),
], y=1.5, gap=10)

out = os.path.join(HERE, "TTT-efficiency-seminar.pptx")
p.save(out)
print("saved", out, "with", len(p.slides._sldIdLst), "slides")
