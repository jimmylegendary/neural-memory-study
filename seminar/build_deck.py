#!/usr/bin/env python3
"""Seminar deck builder — draws diagrams directly with pptx_lib. Built section by section.
Run: ../.venv/bin/python build_deck.py  -> seminar/TTT-seminar.pptx   (python-pptx lives in repo .venv)
Render check: soffice --headless --convert-to pdf ... ; pdftoppm -png (글/그림 겹침 육안검사).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pptx_lib import (deck, slide, title_slide, section_divider, box, arrow, text, picture,
                      SW, SH, PP_ALIGN, INK, MUTE, BLUE, BLUEB, RED, REDB, GOLD, GOLDB,
                      GREEN, GREENB, GREY, GREYB, WHITE)

HERE = os.path.dirname(os.path.abspath(__file__))
p = deck()

# ============================== TITLE ==============================
title_slide(p,
    "테스트-타임 메모리 계열 — Titans에서 Sleep까지, 그리고 서빙 비용",
    "학습을 처음 보는 inference·systems 엔지니어를 위한 6편(Titans·Miras·Atlas·TNT·Nested Learning·Sleep) 해부 + 배포 비용 모델링",
    "이승호 (SAIT) · 2시간 발표 + 30분 Q&A")

# ============================== 목차 (TOC) ==============================
s = slide(p, "목차", "오늘 다룰 것")
def toc_col(x, w, head, hc, items):
    box(s, x, 1.5, w, 0.62, head, fc=WHITE, ec=hc, size=15.5, tcolor=hc, bold=True)
    text(s, x + 0.1, 2.4, w - 0.15, 4.7, [(it, {"size": 12, "color": INK}) for it in items],
         size=12, line_spacing=1.3, space_after=2)
toc_col(0.55, 4.0, "A. Background (학습 입문)", BLUE, [
    "A1. 학습 한 바퀴 — 2-layer MLP:", "     forward→loss→backward→optimizer→update",
    "A2. Optimizer의 역사 (수식과 함께)", "     계보 지도(제공 이미지) →",
    "     GD · SGD · Momentum", "     Adagrad · RMSProp · Adam · AdamW",
    "A3. 대안 계열", "     linear attention", "     DeltaNet → Gated DeltaNet", "     SSM · Mamba-2"])
toc_col(4.7, 4.0, "B. 논문 6편 (발명→구현→증명)", GREEN, [
    "B1. Titans — deep neural memory", "     surprise·momentum·forget, MAC/MAG/MAL",
    "B2. Miras — 4축 설계공간", "     메모리 = online 최적화",
    "B3. Atlas — Omega rule·용량·Muon", "B4. TNT — chunk·2-stage·global/local·Q-K",
    "B5. Nested Learning / HOPE", "     optimizer=memory · self-mod · CMS",
    "B6. Sleep — offline consolidation", "     Knowledge Seeding · Dreaming"])
toc_col(8.85, 3.9, "C. System modeling (우리 기여)", GOLD, [
    "C1. pair thesis — decode=memory-bound", "C2. 8개 가속기 크로스 측정",
    "C3. HOPE-block roofline", "     + zHBM · HBM-PIM · SRAM scaling",
    "C4. scaling 청사진", "", "부록. QA 모음 (교차·논문별)"])

# ============================== 들어가기 (scope) ==============================
s = slide(p, "들어가기", "이 세미나의 목적 · 범위 · 주의")
box(s, 0.55, 1.45, 12.23, 1.25,
    [("목적·방향", {"size": 15, "bold": True, "color": BLUE, "align": PP_ALIGN.LEFT, "space_after": 5}),
     ("inference·systems 엔지니어가 test-time memory 계열(Titans~Sleep)을 shape·비용 수준으로 이해한다. 각 논문의 핵심 발명을 먼저 보고 → 어떻게 구현·증명되는지 → 논문 간 framing이 이어지는 흐름 → 마지막에 이 배포 형태가 서빙에서 얼마나 드는지(우리 기여).",
      {"size": 14, "color": INK, "align": PP_ALIGN.LEFT})],
    fc=BLUEB, ec=BLUE, align=PP_ALIGN.LEFT)
text(s, 0.55, 2.95, 8, 0.4, "미리 알아둘 3가지 (읽는 규칙)", size=15, color=RED, bold=True, space_after=0)
cav = [
    ("① 학습(pre-training)은 다루지 않는다", "Titans 기반 모델을 처음부터 훈련하는 과정은 범위 밖. 이 계열의 test-time 동작과 서빙에 집중한다. (Background의 학습 설명은 '무엇을 배우는가'를 읽기 위한 최소한.)"),
    ("② 논문에 구체 수치가 빠진 곳이 많다", "모델 파라미터 shape·차원·학습 하이퍼 등 구체값이 명시 안 된 부분이 흔하다. 그림·설명에서 추론한 것은 '추론'으로, 본문에 있는 것은 '명시'로 구분해 표시한다."),
    ("③ 논문 1편만으론 불완전하다", "한 논문의 설명이 스스로 완결되지 않아, 다른 논문(특히 HOPE)을 같이 봐야 이해되는 부분이 많다. 예: Titans의 투영 분리는 HOPE에서야 명시된다. 그래서 6편을 하나의 흐름으로 읽는다."),
]
y = 3.5
for h, d in cav:
    box(s, 0.55, y, 12.23, 1.15,
        [(h, {"size": 14.5, "bold": True, "color": RED, "align": PP_ALIGN.LEFT, "space_after": 4}),
         (d, {"size": 13, "color": INK, "align": PP_ALIGN.LEFT})],
        fc=REDB, ec=RED, align=PP_ALIGN.LEFT)
    y += 1.26

# ============================== SECTION A DIVIDER ==============================
section_divider(p, "PART A", "Background — 학습이 처음이라면", BLUE)

# ============================== A1. TRAINING DAG ==============================
s = slide(p, "A1 · 학습 한 바퀴", "2-layer MLP + head: forward → loss → backward → optimizer → update")

# --- forward chain (top) ---
fy, fh = 1.55, 0.85
fx = [0.55, 2.68, 4.81, 6.94, 9.07, 11.2]; fw = 1.5
fwd = [
    ("x", "[B, d_in]", BLUEB, BLUE),
    ("h = W₁·x", "[B, d_h]", BLUEB, BLUE),
    ("a = σ(h)", "[B, d_h]", BLUEB, BLUE),
    ("o = W₂·a", "[B, d_out]", BLUEB, BLUE),
    ("z = head(o)", "logits [B, V]", BLUEB, BLUE),
    ("L", "loss (scalar)", REDB, RED),
]
ops = ["matmul W₁", "σ (GeLU)", "matmul W₂", "head W_h", "L2 loss"]
for i, (t1, t2, fc, ec) in enumerate(fwd):
    box(s, fx[i], fy, fw, fh, [(t1, {"size": 13, "bold": True}), (t2, {"size": 10, "color": MUTE})], fc=fc, ec=ec)
for i in range(5):
    ax1 = fx[i] + fw; ax2 = fx[i + 1]
    arrow(s, ax1, fy + fh / 2, ax2, fy + fh / 2, color=BLUE, lw=1.6)
    # op just above the arrow (short — box already shows the formula)
    text(s, ax1 - 0.25, fy - 0.34, (ax2 - ax1) + 0.5, 0.3, ops[i], size=9, color=BLUE, align=PP_ALIGN.CENTER)

text(s, 0.55, 1.0, 1.9, 0.3, "① forward", size=12, color=BLUE, bold=True)
text(s, 7.0, 2.55, 6.0, 0.4, "batch B개 샘플 동시 처리 · loss는 B개 평균(또는 합)으로 누적", size=10.5, color=MUTE, italic=True)

# --- backward (middle) ---
by = 3.55
box(s, 0.55, by, 12.2, 0.72,
    [("② loss.backward() — autograd 역전파", {"size": 12.5, "bold": True, "color": RED, "align": PP_ALIGN.LEFT}),
     ("∂L/∂z → ∂L/∂W₂, ∂L/∂a → ∂L/∂h → ∂L/∂W₁, ∂L/∂head.  각 파라미터 W의 gradient g = ∂L/∂W 를 구해 W.grad 에 채운다(순전파의 역순, 체인 룰).",
      {"size": 11, "color": INK, "align": PP_ALIGN.LEFT})],
    fc=REDB, ec=RED, align=PP_ALIGN.LEFT)
arrow(s, 11.9, fy + fh + 0.15, 0.9, by - 0.02, color=RED, lw=1.4, dashed=True)  # loss -> back

# --- optimizer + update (bottom) ---
oy = 4.62
box(s, 0.55, oy, 7.7, 1.55,
    [("③ optimizer.step()  — AdamW (gradient g를 받아 '얼마나·어느 방향' 결정)", {"size": 12.5, "bold": True, "color": GOLD, "align": PP_ALIGN.LEFT}),
     ("m ← β₁·m + (1−β₁)·g          (1차 모멘텀: 방향의 관성)", {"size": 11, "mono": True, "align": PP_ALIGN.LEFT}),
     ("v ← β₂·v + (1−β₂)·g²         (2차 모멘텀: 좌표별 스케일)", {"size": 11, "mono": True, "align": PP_ALIGN.LEFT}),
     ("m̂,v̂ = 편향보정 ;  ΔW = −η·m̂/(√v̂+ε)   −  η·λ·W  ← decoupled decay", {"size": 11, "mono": True, "align": PP_ALIGN.LEFT})],
    fc=GOLDB, ec=GOLD, align=PP_ALIGN.LEFT)
box(s, 8.5, oy, 4.25, 1.55,
    [("④ W ← W + ΔW", {"size": 13, "bold": True, "color": GOLD}),
     ("W₁, W₂, head 를 갱신.", {"size": 11, "color": INK}),
     ("optimizer.zero_grad()로 g 비우고", {"size": 10.5, "color": MUTE}),
     ("다음 batch로 → 반복.", {"size": 10.5, "color": MUTE})],
    fc=GOLDB, ec=GOLD)

box(s, 0.55, 6.45, 12.2, 0.62,
    [("핵심 한 문장.", {"size": 11.5, "bold": True, "color": GREEN, "align": PP_ALIGN.LEFT}),
     ("학습 = ①forward로 예측 → ②backward로 'gradient(어디가 틀렸나)' → ③optimizer로 'gradient를 기억·가공(m,v)해 갱신량' → ④weight 이동. 이 계열은 ②③④를 추론 중에도 돌린다.",
      {"size": 11, "color": INK, "align": PP_ALIGN.LEFT})],
    fc=GREENB, ec=GREEN, align=PP_ALIGN.LEFT)

# ============================== A2.0  OPTIMIZER 계보도 (제공 이미지 그대로) ==============================
s = slide(p, "A2 · Optimizer의 역사", "한 장으로 보는 직관 지도", BLUE)
# 제공받은 intuition map을 원본 그대로 삽입 (640x310, ratio 2.065)
IMG_W = 10.9; IMG_H = IMG_W / 2.065        # 5.28
picture(s, os.path.join(HERE, "assets", "optimizer-map.png"), (SW - IMG_W) / 2, 1.35, w=IMG_W)
box(s, 0.55, 6.78, 12.2, 0.55,
    [("파란 줄기 = '어느 방향으로'(관성)를 고치는 흐름 · 초록 줄기 = '얼마나(보폭)'를 고치는 흐름 → Adam이 둘을 합침. 다음 장부터 이 중 GD·SGD·Momentum·Adagrad·RMSProp·Adam·AdamW를 1페이지씩 수식과 함께.",
      {"size": 12.5, "color": INK, "align": PP_ALIGN.LEFT})], fc=GREYB, ec=GREY, align=PP_ALIGN.LEFT)

# ============================== A2.x  개별 OPTIMIZER (1p 1개, 수식 포함) ==============================
def opt_slide(tag, name, full, branch, intuition, eqn, solves, remains, ec=BLUE):
    s = slide(p, tag, f"{name}  ·  {full}", ec)
    bgb = BLUEB if ec == BLUE else (GREENB if ec == GREEN else GOLDB)
    text(s, 0.55, 1.18, 12.2, 0.34, branch, size=13.5, color=ec, bold=True, space_after=0)
    # 직관 (제공 이미지의 문구)
    box(s, 0.55, 1.6, 12.23, 1.12,
        [("직관 (왜 이렇게?)", {"size": 14, "bold": True, "color": ec, "align": PP_ALIGN.LEFT, "space_after": 4}),
         (intuition, {"size": 16.5, "color": INK, "align": PP_ALIGN.LEFT})],
        fc=bgb, ec=ec, align=PP_ALIGN.LEFT)
    # 수식
    eqlines = [("갱신 규칙 (수식)", {"size": 14, "bold": True, "color": INK, "align": PP_ALIGN.LEFT, "space_after": 6})]
    for e in eqn:
        eqlines.append((e, {"size": 15.5, "mono": True, "color": INK, "align": PP_ALIGN.LEFT, "space_after": 7}))
    box(s, 0.55, 2.86, 12.23, 2.12, eqlines, fc=WHITE, ec=GREY, align=PP_ALIGN.LEFT)
    # 해결 / 남은 문제
    box(s, 0.55, 5.14, 6.0, 1.96,
        [("이 optimizer가 해결한 것", {"size": 14, "bold": True, "color": GREEN, "align": PP_ALIGN.LEFT, "space_after": 5}),
         (solves, {"size": 14.5, "color": INK, "align": PP_ALIGN.LEFT})],
        fc=GREENB, ec=GREEN, align=PP_ALIGN.LEFT)
    box(s, 6.78, 5.14, 6.0, 1.96,
        [("남은 문제 → 다음", {"size": 14, "bold": True, "color": RED, "align": PP_ALIGN.LEFT, "space_after": 5}),
         (remains, {"size": 14.5, "color": INK, "align": PP_ALIGN.LEFT})],
        fc=REDB, ec=RED, align=PP_ALIGN.LEFT)

opt_slide("A2 · GD", "GD", "Gradient Descent (batch)", "뿌리 — 가장 정확한 한 걸음",
    "\"모든 자료를 다 검토해서, 내 위치의 산 기울기를 계산해 갈 방향을 찾았다.\"",
    ["g_t = (1/N) · Σ_{i=1..N} ∇ℓ_i(θ_t)      # 전체 N개 데이터의 평균 gradient",
     "θ_{t+1} = θ_t − η · g_t                  # η = learning rate(보폭)"],
    "손실을 가장 빨리 줄이는 방향(가장 가파른 내리막)을 데이터 전체로 정확히 계산한다.",
    "한 스텝마다 데이터 N개를 전부 봐야 함 → 너무 느리다.  → SGD", ec=BLUE)

opt_slide("A2 · SGD", "SGD", "Stochastic Gradient Descent", "파란 줄기 — 속도",
    "\"전부 다 봐야 한 걸음은 너무 오래 걸리니까, 조금만 보고 빨리 판단한다. 같은 시간에 더 많이 간다.\"",
    ["g_t = ∇ℓ_B(θ_t)          # 미니배치 B개만 사용 (|B| ≪ N)",
     "θ_{t+1} = θ_t − η · g_t"],
    "스텝당 비용이 급감 → 같은 시간에 훨씬 많은 스텝. gradient noise가 얕은 지역최소 탈출에도 도움.",
    "gradient가 매 스텝 출렁여 지그재그. 방향도 보폭도 여전히 그대로.  → Momentum · Adagrad", ec=BLUE)

opt_slide("A2 · Momentum", "Momentum", "SGD with Momentum", "파란 줄기 — '스텝 방향' 개선",
    "\"스텝을 계산해서 움직인 후, 아까 내려오던 관성 방향으로 또 가자.\"",
    ["v_t = μ · v_{t-1} + g_t        # μ≈0.9, 과거 방향의 관성(누적)",
     "θ_{t+1} = θ_t − η · v_t"],
    "일관된 방향은 누적 가속하고 출렁이는 성분은 상쇄 → 지그재그 완화, 좁은 골짜기를 빠르게 통과.",
    "보폭(스케일)은 아직 모든 좌표가 동일. '좌표별 보폭'은 아래 초록 줄기(Adagrad→RMSProp)가 맡고, 뒤에서 Adam이 방향과 합친다.", ec=BLUE)

opt_slide("A2 · Adagrad", "Adagrad", "Adaptive Gradient", "초록 줄기 — '스텝 사이즈' 개선",
    "\"안 가 본 곳은 성큼 빠르게 훑고, 많이 가 본 곳은 잘 가니까 갈수록 보폭을 줄여 세밀히 탐색.\"",
    ["G_t = G_{t-1} + g_t²                  # 좌표별 gradient² 누적",
     "θ_{t+1} = θ_t − η · g_t / (√G_t + ε)   # 좌표마다 다른 보폭"],
    "좌표마다 다른 learning rate — 드물게 큰 gradient(희소 feature)는 크게, 자주 큰 좌표는 작게.",
    "G_t가 단조 증가 → 분모가 계속 커져 보폭이 0으로 죽는다(학습 정지).  → RMSProp", ec=GREEN)

opt_slide("A2 · RMSProp", "RMSProp", "Root Mean Square Propagation", "초록 줄기 — EMA로 보완",
    "\"보폭을 줄이는 건 좋은데, (전부 누적하지 말고) 최근 맥락을 봐 가며 하자.\"",
    ["E[g²]_t = ρ · E[g²]_{t-1} + (1−ρ) · g_t²   # ρ≈0.9, 최근값 지수이동평균",
     "θ_{t+1} = θ_t − η · g_t / (√E[g²]_t + ε)"],
    "합 대신 EMA → 분모가 무한정 커지지 않아 보폭이 죽지 않는다. 비정상(non-stationary) 목표에도 안정.",
    "방향(momentum)은 아직 안 씀. 초기(0에서 시작) 편향 보정도 없다.  → Adam", ec=GREEN)

opt_slide("A2 · Adam", "Adam", "Adaptive Moment Estimation", "두 줄기의 합류 ★",
    "\"RMSProp(보폭) + Momentum(방향)을 합치자 — 방향도 스텝 사이즈도 적절하게!\"",
    ["m_t = β₁·m_{t-1} + (1−β₁)·g_t          # 1차 모멘텀: 방향",
     "v_t = β₂·v_{t-1} + (1−β₂)·g_t²         # 2차 모멘텀: 좌표별 보폭",
     "m̂ = m_t/(1−β₁^t),  v̂ = v_t/(1−β₂^t)     # 초기 0-편향 보정",
     "θ_{t+1} = θ_t − η · m̂ / (√v̂ + ε)"],
    "방향(m) + 좌표별 보폭(v) + 편향 보정을 한 번에. 튜닝 거의 없이 잘 동작 → 사실상 표준.",
    "L2 weight decay를 gradient에 더하면 1/√v̂로 나뉘어 좌표별로 왜곡된다.  → AdamW", ec=GOLD)

opt_slide("A2 · AdamW", "AdamW ★", "Adam with decoupled Weight decay", "종착점 — 오늘의 표준",
    "\"weight decay(정규화)를 gradient에 섞지 말고, weight에서 '직접' 빼자 — 그래야 좌표마다 똑같이 적용된다.\"",
    ["m_t, v_t, m̂, v̂ : Adam과 동일 (decay를 g_t에 넣지 않음)",
     "θ_{t+1} = θ_t − η · ( m̂/(√v̂+ε)  +  λ·θ_t )      # λ·θ 항을 분리(decoupled)"],
    "decay가 adaptive 스케일(1/√v̂)에 섞이지 않아 정규화가 일관 → 일반화↑. 오늘날 LLM 학습의 표준.",
    "여기까지가 '표준 optimizer'. 이 계열 논문은 이 optimizer 자체를 memory로 재해석한다(HOPE).  → Part B", ec=GOLD)

# save
out = os.path.join(HERE, "TTT-seminar.pptx")
p.save(out)
print(f"saved {out} ({len(p.slides.__iter__.__self__._sldIdLst)} slides)")
