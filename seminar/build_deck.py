#!/usr/bin/env python3
"""Seminar deck builder — draws diagrams directly with pptx_lib. Built section by section.
Run: python build_deck.py  -> seminar/TTT-seminar.pptx
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pptx_lib import (deck, slide, title_slide, section_divider, box, arrow, text,
                      SW, SH, PP_ALIGN, INK, MUTE, BLUE, BLUEB, RED, REDB, GOLD, GOLDB,
                      GREEN, GREENB, GREY, GREYB, WHITE)

HERE = os.path.dirname(os.path.abspath(__file__))
p = deck()

# ============================== TITLE ==============================
title_slide(p,
    "테스트-타임 메모리 계열 — Titans에서 Sleep까지, 그리고 서빙 비용",
    "학습을 처음 보는 inference·systems 엔지니어를 위한 6편(Titans·Miras·Atlas·TNT·Nested Learning·Sleep) 해부 + 배포 비용 모델링",
    "이승호 (SAIT) · 2시간 발표 + 30분 Q&A")

# ============================== AGENDA ==============================
s = slide(p, "AGENDA", "오늘의 지도")
rows = [
    ("A. Background", "학습이 처음인 분을 위해 — 2-layer MLP 학습 한 바퀴, optimizer의 역사(GD→AdamW), 그리고 대안 계열(linear attention·DeltaNet·Mamba2)", BLUE),
    ("B. 논문 6편", "핵심 발명을 먼저 → 어떻게 구현 → 어떻게 증명. Titans가 연 프레임을 Miras·Atlas·TNT가 확장하고, HOPE가 전부 통합, Sleep이 offline으로 확장", GREEN),
    ("C. System modeling (우리 기여)", "이 배포 형태가 서빙에서 얼마나 드는가 — decode는 memory-bound. 8개 가속기 크로스 측정 + scaling 청사진", GOLD),
]
y = 1.7
for lab, desc, col in rows:
    box(s, 0.55, y, 3.3, 1.15, lab, fc=WHITE, ec=col, size=15, tcolor=col, bold=True)
    text(s, 4.1, y + 0.12, 8.6, 1.0, desc, size=12.5, color=INK)
    y += 1.45
text(s, 0.55, 6.9, 12, 0.4, "핵심 원리 하나: 학습(training)은 '가중치가 움직이는 것'이고, 이 계열은 그 움직임을 추론(inference) 중에도 계속한다.",
     size=12, color=MUTE, italic=True)

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

# save
out = os.path.join(HERE, "TTT-seminar.pptx")
p.save(out)
print(f"saved {out} ({len(p.slides.__iter__.__self__._sldIdLst)} slides)")
