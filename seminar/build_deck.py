#!/usr/bin/env python3
"""Seminar deck builder — draws diagrams directly with pptx_lib. Built section by section.
Run: ../.venv/bin/python build_deck.py  -> seminar/TTT-seminar.pptx   (python-pptx lives in repo .venv)
Render check: soffice --headless --convert-to pdf ... ; pdftoppm -png (글/그림 겹침 육안검사).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pptx_lib import (deck, slide, title_slide, section_divider, box, arrow, text, picture, fit_image,
                      SW, SH, PP_ALIGN, MSO_ANCHOR, INK, MUTE, BLUE, BLUEB, RED, REDB, GOLD, GOLDB,
                      GREEN, GREENB, GREY, GREYB, WHITE)
import mathpng as mp

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(os.path.dirname(HERE), "reffigs")   # repo-root reffigs/<id>/figures/figN.png
MFIG = os.path.join(os.path.dirname(HERE), "multiarch", "figures")   # our own system-modeling figures
def rf(pid, fig):
    return os.path.join(REF, pid, "figures", f"fig{fig}.png")

def our_fig(s, fname, x, y, w, h, caption, ec=GOLD):
    """Embed one of OUR system-modeling figures (multiarch/figures) with a caption (no external source)."""
    fit_image(s, os.path.join(MFIG, fname), x, y, w, h - 0.42, frame=True)
    text(s, x, y + h - 0.4, w, 0.4, caption, size=10.5, color=INK, align=PP_ALIGN.CENTER, space_after=0)

def claim_row(s, x, y, w, n, title, body, ec=GREEN):
    """A numbered core-claim card (used on each paper's '핵심 발명' slide)."""
    box(s, x, y, 0.5, 0.5, str(n), fc=ec, ec=ec, size=16, tcolor=WHITE, bold=True)
    box(s, x + 0.62, y, w - 0.62, 0.5, "", fc=WHITE, ec=GREY, lw=0)
    text(s, x + 0.72, y - 0.02, w - 0.9, 0.55,
         [(title + "  ", {"size": 13.5, "bold": True, "color": ec}), (body, {"size": 12, "color": INK})],
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE, space_after=0)

def fig_with_caption(s, pid, fig, x, y, w, h, caption, ec=GREEN):
    """Embed a paper figure fit to (x,y,w,h) with a caption + source line below."""
    fit_image(s, rf(pid, fig), x, y, w, h - 0.5, frame=True)
    text(s, x, y + h - 0.46, w, 0.46,
         [(caption, {"size": 10.5, "color": INK, "align": PP_ALIGN.CENTER}),
          (f"출처: {pid} Fig.{fig}", {"size": 8.5, "color": MUTE, "align": PP_ALIGN.CENTER, "italic": True})],
         align=PP_ALIGN.CENTER, space_after=1)

p = deck()

# ============================== TITLE ==============================
title_slide(p,
    "테스트-타임 메모리 계열 — Titans에서 Sleep까지, 그리고 서빙 비용",
    "학습을 처음 보는 inference·systems 엔지니어를 위한 6편(Titans·Miras·Atlas·TNT·Nested Learning·Sleep) 해부 + 배포 비용 모델링",
    "이승호 (SAIT) · 2시간 발표 + 30분 Q&A")

# ============================== 목차 (TOC) ==============================
s = slide(p, "목차", "오늘 다룰 것 — 세 파트")
def toc_card(x, w, head, sub, hc, items):
    box(s, x, 1.42, w, 5.5, "", fc=WHITE, ec=hc, lw=1.7)                 # 컨테이너(객체)
    box(s, x, 1.42, w, 0.82, [(head, {"size": 14.5, "bold": True, "color": WHITE, "space_after": 2}),
        (sub, {"size": 10, "color": WHITE})], fc=hc, ec=hc)              # 헤더 밴드
    lines = []
    for txt, lvl in items:
        if lvl == 0:
            lines.append((txt, {"size": 12.5, "bold": True, "color": hc, "align": PP_ALIGN.LEFT, "space_after": 2}))
        else:
            lines.append(("    · " + txt, {"size": 10.5, "color": MUTE, "align": PP_ALIGN.LEFT, "space_after": 1}))
    text(s, x + 0.28, 2.48, w - 0.52, 4.3, lines, line_spacing=1.22, space_after=2, align=PP_ALIGN.LEFT)
toc_card(0.55, 4.0, "A. Background", "학습이 처음이라면", BLUE, [
    ("A1. 학습 한 바퀴 (2-layer MLP)", 0), ("forward→loss→backward→optimizer→update", 1),
    ("A2. Optimizer의 역사 (수식)", 0), ("계보 지도 → GD·SGD·Momentum", 1),
    ("Adagrad·RMSProp·Adam·AdamW", 1),
    ("A3. 대안 계열 (개념별 1장)", 0), ("linear attention · DeltaNet · Gated DeltaNet", 1),
    ("SSM · Mamba · Mamba-2", 1)])
toc_card(4.7, 4.0, "B. 논문 6편", "핵심 발명 → 논거 → 증명", GREEN, [
    ("B1. Titans", 0), ("test-time 신경망 메모리 · MAC/MAG/MAL", 1),
    ("B2. Miras", 0), ("4축 설계공간 (메모리=online 최적화)", 1),
    ("B3. Atlas", 0), ("Omega rule · 용량 확장 · Muon", 1),
    ("B4. TNT", 0), ("chunk 정렬 · global/local · serving", 1),
    ("B5. Nested Learning / HOPE", 0), ("optimizer=memory · self-mod · CMS", 1),
    ("B6. Sleep", 0), ("offline consolidation · Dreaming", 1)])
toc_card(8.85, 3.9, "C. System modeling", "우리 기여 — 서빙 비용", GOLD, [
    ("C0. 방법론 (HATIR · HAT schema)", 0), ("무엇으로 뽑았고 왜 신뢰할 수 있나", 1),
    ("C1. pair thesis — decode=memory-bound", 0),
    ("C2. 8개 가속기 크로스 측정", 0),
    ("C3. HOPE-block roofline (H100 트윈)", 0),
    ("C4. zHBM · HBM-PIM · SRAM scaling", 0),
    ("C5. scaling 청사진", 0)])

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
by = 3.5
box(s, 0.55, by, 12.23, 0.9,
    [("② loss.backward() — autograd 역전파", {"size": 12.5, "bold": True, "color": RED, "align": PP_ALIGN.LEFT, "space_after": 3}),
     ("출력 쪽 ∂L/∂z 부터 체인 룰로 거슬러 ∂L/∂W₂ → ∂L/∂a → ∂L/∂h → ∂L/∂W₁ 까지. 각 파라미터 W의 gradient g=∂L/∂W 를 구해 W.grad 에 채운다(순전파의 역순).",
      {"size": 11, "color": INK, "align": PP_ALIGN.LEFT})],
    fc=REDB, ec=RED, align=PP_ALIGN.LEFT)
arrow(s, 11.9, fy + fh + 0.12, 0.9, by - 0.02, color=RED, lw=1.4, dashed=True)  # loss -> back

# --- optimizer + update (bottom) ---
oy = 4.58
box(s, 0.55, oy, 7.7, 1.72, "", fc=GOLDB, ec=GOLD)
text(s, 0.72, oy + 0.08, 7.4, 0.3, "③ optimizer.step() — AdamW (g를 받아 '얼마나·어느 방향' 결정)",
     size=12, color=GOLD, bold=True, space_after=0, align=PP_ALIGN.LEFT)
_ap, _aw, _ah = mp.eq(r"\begin{aligned}"
                      r"m &\leftarrow \beta_1 m + (1-\beta_1)\,g\\[2pt]"
                      r"v &\leftarrow \beta_2 v + (1-\beta_2)\,g^2\\[2pt]"
                      r"\Delta W &= -\,\eta\,\frac{\hat m}{\sqrt{\hat v}+\epsilon}\;-\;\eta\lambda W"
                      r"\end{aligned}", pt=13)
fit_image(s, _ap, 0.8, oy + 0.45, 4.9, 1.15)
for i, nt in enumerate(["1차 모멘텀 = 방향", "2차 모멘텀 = 보폭", "decoupled weight decay"]):
    text(s, 5.75, oy + 0.5 + i * 0.38, 2.4, 0.3, "← " + nt, size=9.5, color=MUTE, space_after=0, align=PP_ALIGN.LEFT)
box(s, 8.5, oy, 4.25, 1.72,
    [("④ W ← W + ΔW", {"size": 13, "bold": True, "color": GOLD, "space_after": 4}),
     ("W₁, W₂, head 를 갱신.", {"size": 11, "color": INK, "space_after": 2}),
     ("optimizer.zero_grad()로 g 비우고", {"size": 10.5, "color": MUTE, "space_after": 1}),
     ("다음 batch로 → 반복.", {"size": 10.5, "color": MUTE})],
    fc=GOLDB, ec=GOLD)

box(s, 0.55, 6.5, 12.23, 0.82,
    [("핵심 한 문장", {"size": 11.5, "bold": True, "color": GREEN, "align": PP_ALIGN.LEFT, "space_after": 3}),
     ("학습 = ①forward 예측 → ②backward로 gradient(어디가 틀렸나) → ③optimizer로 가공(m,v)해 갱신량 → ④weight 이동. 이 계열은 ②③④를 '추론 중에도' 돌린다.",
      {"size": 10.5, "color": INK, "align": PP_ALIGN.LEFT})],
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
    # 수식 (진짜 LaTeX → 이미지, 옆에 한글 주석)
    ex, ey, ew, eh = 0.55, 2.84, 12.23, 2.42
    box(s, ex, ey, ew, eh, "", fc=WHITE, ec=GREY)
    text(s, ex + 0.22, ey + 0.09, 4, 0.3, "갱신 규칙 (수식)", size=13.5, color=INK, bold=True, space_after=0)
    imgs = [(mp.eq(ltx, pt=18), note) for ltx, note in eqn]   # render big, then auto-fit to the card
    gap, avail = 0.16, eh - 0.62
    raw = sum(h for (_p, _w, h), _n in imgs) + gap * max(0, len(imgs) - 1)
    scale = min(1.0, avail / raw) if raw > 0 else 1.0         # shrink only if it would overflow
    cur = ey + 0.56 + max(0.0, (avail - raw * scale) / 2)     # vertically center the block
    for (path, w, h), note in imgs:
        pw, ph = w * scale, h * scale
        picture(s, path, ex + 0.45, cur, h=ph)
        if note:
            text(s, ex + 0.45 + pw + 0.42, cur - 0.03, ew - (pw + 1.45), ph + 0.06, note,
                 size=13, color=MUTE, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
        cur += ph + gap
    # 해결 / 남은 문제
    box(s, 0.55, 5.4, 6.0, 1.83,
        [("이 optimizer가 해결한 것", {"size": 14, "bold": True, "color": GREEN, "align": PP_ALIGN.LEFT, "space_after": 5}),
         (solves, {"size": 14, "color": INK, "align": PP_ALIGN.LEFT})],
        fc=GREENB, ec=GREEN, align=PP_ALIGN.LEFT)
    box(s, 6.78, 5.4, 6.0, 1.83,
        [("남은 문제 → 다음", {"size": 14, "bold": True, "color": RED, "align": PP_ALIGN.LEFT, "space_after": 5}),
         (remains, {"size": 14, "color": INK, "align": PP_ALIGN.LEFT})],
        fc=REDB, ec=RED, align=PP_ALIGN.LEFT)

opt_slide("A2 · GD", "GD", "Gradient Descent (batch)", "뿌리 — 가장 정확한 한 걸음",
    "\"모든 자료를 다 검토해서, 내 위치의 산 기울기를 계산해 갈 방향을 찾았다.\"",
    [(r"g_t = \frac{1}{N}\sum_{i=1}^{N}\nabla \ell_i(\theta_t)", "전체 N개 데이터의 평균 gradient"),
     (r"\theta_{t+1} = \theta_t - \eta\, g_t", "η = learning rate (보폭)")],
    "손실을 가장 빨리 줄이는 방향(가장 가파른 내리막)을 데이터 전체로 정확히 계산한다.",
    "한 스텝마다 데이터 N개를 전부 봐야 함 → 너무 느리다.  → SGD", ec=BLUE)

opt_slide("A2 · SGD", "SGD", "Stochastic Gradient Descent", "파란 줄기 — 속도",
    "\"전부 다 봐야 한 걸음은 너무 오래 걸리니까, 조금만 보고 빨리 판단한다. 같은 시간에 더 많이 간다.\"",
    [(r"g_t = \nabla \ell_{\mathcal{B}}(\theta_t)", "미니배치 B개만 사용 (|B| ≪ N)"),
     (r"\theta_{t+1} = \theta_t - \eta\, g_t", "")],
    "스텝당 비용이 급감 → 같은 시간에 훨씬 많은 스텝. gradient noise가 얕은 지역최소 탈출에도 도움.",
    "gradient가 매 스텝 출렁여 지그재그. 방향도 보폭도 여전히 그대로.  → Momentum · Adagrad", ec=BLUE)

opt_slide("A2 · Momentum", "Momentum", "SGD with Momentum", "파란 줄기 — '스텝 방향' 개선",
    "\"스텝을 계산해서 움직인 후, 아까 내려오던 관성 방향으로 또 가자.\"",
    [(r"v_t = \mu\, v_{t-1} + g_t", "μ≈0.9, 과거 방향의 관성(누적)"),
     (r"\theta_{t+1} = \theta_t - \eta\, v_t", "")],
    "일관된 방향은 누적 가속하고 출렁이는 성분은 상쇄 → 지그재그 완화, 좁은 골짜기를 빠르게 통과.",
    "보폭(스케일)은 아직 모든 좌표가 동일. '좌표별 보폭'은 아래 초록 줄기(Adagrad→RMSProp)가 맡고, 뒤에서 Adam이 방향과 합친다.", ec=BLUE)

opt_slide("A2 · Adagrad", "Adagrad", "Adaptive Gradient", "초록 줄기 — '스텝 사이즈' 개선",
    "\"안 가 본 곳은 성큼 빠르게 훑고, 많이 가 본 곳은 잘 가니까 갈수록 보폭을 줄여 세밀히 탐색.\"",
    [(r"G_t = G_{t-1} + g_t^{2}", "좌표별 gradient² 누적"),
     (r"\theta_{t+1} = \theta_t - \eta\,\frac{g_t}{\sqrt{G_t}+\epsilon}", "좌표마다 다른 보폭")],
    "좌표마다 다른 learning rate — 드물게 큰 gradient(희소 feature)는 크게, 자주 큰 좌표는 작게.",
    "G_t가 단조 증가 → 분모가 계속 커져 보폭이 0으로 죽는다(학습 정지).  → RMSProp", ec=GREEN)

opt_slide("A2 · RMSProp", "RMSProp", "Root Mean Square Propagation", "초록 줄기 — EMA로 보완",
    "\"보폭을 줄이는 건 좋은데, (전부 누적하지 말고) 최근 맥락을 봐 가며 하자.\"",
    [(r"\mathbb{E}[g^2]_t = \rho\,\mathbb{E}[g^2]_{t-1} + (1-\rho)\,g_t^{2}", "ρ≈0.9, 최근값 지수이동평균(EMA)"),
     (r"\theta_{t+1} = \theta_t - \eta\,\frac{g_t}{\sqrt{\mathbb{E}[g^2]_t}+\epsilon}", "분모가 무한정 커지지 않음")],
    "합 대신 EMA → 분모가 무한정 커지지 않아 보폭이 죽지 않는다. 비정상(non-stationary) 목표에도 안정.",
    "방향(momentum)은 아직 안 씀. 초기(0에서 시작) 편향 보정도 없다.  → Adam", ec=GREEN)

opt_slide("A2 · Adam", "Adam", "Adaptive Moment Estimation", "두 줄기의 합류 ★",
    "\"RMSProp(보폭) + Momentum(방향)을 합치자 — 방향도 스텝 사이즈도 적절하게!\"",
    [(r"g_t = \nabla L(\theta_t) + \lambda\,\theta_t", "weight decay = L2 정규화를 gradient에 더함"),
     (r"m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\ \ v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^{2}", "방향 m · 보폭 v"),
     (r"\hat{m}=\frac{m_t}{1-\beta_1^{t}},\ \ \hat{v}=\frac{v_t}{1-\beta_2^{t}}", "초기 0-편향 보정"),
     (r"\theta_{t+1} = \theta_t - \eta\,\frac{\hat{m}}{\sqrt{\hat{v}}+\epsilon}", "λθ가 m̂ 안에 섞여 1/√v̂로 스케일됨")],
    "방향(m) + 좌표별 보폭(v) + 편향 보정을 한 번에. 튜닝 거의 없이 잘 동작 → 사실상 표준. weight decay는 보통 L2로 적용(위 g_t).",
    "그런데 λθ가 √v̂로 나뉘어 좌표마다 다르게 적용 = 왜곡. 이 항을 밖으로 빼면? → AdamW", ec=GOLD)

opt_slide("A2 · AdamW", "AdamW ★", "Adam with decoupled Weight decay", "종착점 — 오늘의 표준",
    "\"weight decay(정규화)를 gradient에 섞지 말고, weight에서 '직접' 빼자 — 그래야 좌표마다 똑같이 적용된다.\"",
    [(r"\text{(Adam+L2)}\quad \theta_{t+1} = \theta_t - \eta\,\frac{\hat{m}}{\sqrt{\hat{v}}+\epsilon}", "g=∇L+λθ → λθ가 √v̂에 얽힘 (왜곡)"),
     (r"\text{(AdamW)}\quad \theta_{t+1} = \theta_t - \eta\,\frac{\hat{m}}{\sqrt{\hat{v}}+\epsilon}\ -\ \eta\lambda\theta_t", "−ηλθ를 분리 → √v̂와 무관하게 일정")],
    "위 두 식의 차이 = 맨 끝 −ηλθ 항. AdamW는 decay를 adaptive 스케일(1/√v̂) 밖으로 빼 정규화가 좌표마다 일관 → 일반화↑. 오늘 LLM 표준.",
    "여기까지가 '표준 optimizer'. 이 계열 논문은 이 optimizer 자체를 memory로 재해석한다(HOPE).  → Part B", ec=GOLD)

# ============================== A3.  대안 계열 ==============================
s = slide(p, "A3 · 다른 접근들", "왜 softmax attention을 벗어나나 — sequence를 '상태'로 압축하는 계열", BLUE)
box(s, 0.55, 1.5, 5.85, 1.75,
    [("softmax attention", {"size": 14, "bold": True, "color": RED, "align": PP_ALIGN.LEFT, "space_after": 4}),
     ("• 문맥이 길수록 KV 캐시가 선형으로 증가(메모리 폭발)", {"size": 12.5, "color": INK, "align": PP_ALIGN.LEFT}),
     ("• 토큰마다 전체를 다시 봄 → 계산 O(n²)", {"size": 12.5, "color": INK, "align": PP_ALIGN.LEFT}),
     ("• 무엇을 '기억'할지 압축이 없다 — 전부 보관", {"size": 12.5, "color": INK, "align": PP_ALIGN.LEFT})],
    fc=REDB, ec=RED, align=PP_ALIGN.LEFT)
arrow(s, 6.55, 2.37, 7.35, 2.37, color=INK, lw=2.0)
box(s, 7.45, 1.5, 5.3, 1.75,
    [("고정 크기 '상태' S로 압축하는 RNN류", {"size": 14, "bold": True, "color": BLUE, "align": PP_ALIGN.LEFT, "space_after": 4}),
     ("토큰마다 상태 S를 '어떤 규칙으로' 갱신하느냐가", {"size": 12.5, "color": INK, "align": PP_ALIGN.LEFT}),
     ("계열을 가른다. 이 관점이 곧 Titans의", {"size": 12.5, "color": INK, "align": PP_ALIGN.LEFT}),
     ("memory update(=test-time 학습)로 이어진다.", {"size": 12.5, "bold": True, "color": BLUE, "align": PP_ALIGN.LEFT})],
    fc=BLUEB, ec=BLUE, align=PP_ALIGN.LEFT)
fams = [
    ("Linear Attention", GREEN, "S를 더하기만 (additive)", "누적 간섭이 문제"),
    ("DeltaNet → Gated", GOLD, "delta rule로 덮어쓰기·수정", "+ forget gate"),
    ("SSM · Mamba-2", BLUE, "구조화된 선형 recurrence", "선택적 상태 갱신"),
]
fx = 0.55
for name, c, l1, l2 in fams:
    box(s, fx, 3.55, 3.95, 1.5,
        [(name, {"size": 14.5, "bold": True, "color": c, "space_after": 5}),
         (l1, {"size": 12.5, "color": INK}), (l2, {"size": 12, "color": MUTE})],
        fc=WHITE, ec=c, lw=1.5)
    fx += 4.15
box(s, 0.55, 5.3, 12.23, 1.18,
    [("공통 렌즈 — 다음 6개를 관통하는 한 문장", {"size": 13, "bold": True, "color": INK, "align": PP_ALIGN.LEFT, "space_after": 4}),
     ("모두 \"상태 S를 토큰마다 갱신하는 선형 recurrence\"다. 규칙의 차이일 뿐 — 더하기(linear), 덮어쓰기(delta), 잊기(gated), 구조화(SSM). "
      "→ Titans는 이 갱신을 '작은 신경망 메모리를 gradient로 학습'하는 것으로 승격시킨다(Part B).",
      {"size": 12, "color": INK, "align": PP_ALIGN.LEFT})],
    fc=GREYB, ec=GREY, align=PP_ALIGN.LEFT)
text(s, 0.55, 6.62, 12.23, 0.5,
     "* delta rule: 현재 상태가 key로 조회했을 때 내놓는 값과 목표 value의 '예측오차'만큼만 상태를 고치는 규칙. 손실 ½‖S·k − v‖²에 대한 gradient step 한 번과 같다.",
     size=10, color=MUTE, italic=True, align=PP_ALIGN.LEFT, space_after=0)

# --- A3 개념별 상세 (1개념 = 1페이지) ---
def concept_slide(tag, name, full, ec, problem, latex, symbols, meaning, benefit, limit):
    s = slide(p, tag, f"{name}  ·  {full}", ec)
    bgb = {id(BLUE): BLUEB, id(GREEN): GREENB, id(GOLD): GOLDB}.get(id(ec), BLUEB)
    box(s, 0.55, 1.38, 12.23, 0.9,
        [("이게 푸는 문제", {"size": 13, "bold": True, "color": ec, "align": PP_ALIGN.LEFT, "space_after": 3}),
         (problem, {"size": 12.5, "color": INK, "align": PP_ALIGN.LEFT})], fc=bgb, ec=ec, align=PP_ALIGN.LEFT)
    box(s, 0.55, 2.4, 12.23, 1.74, "", fc=WHITE, ec=GREY)
    text(s, 0.75, 2.5, 5, 0.3, "갱신식", size=12.5, color=INK, bold=True, space_after=0, align=PP_ALIGN.LEFT)
    _p, _w, _h = mp.eq(latex, pt=18)
    fit_image(s, _p, 0.8, 2.92, 6.0, 1.05)
    sym_lines = [("기호의 의미", {"size": 12, "bold": True, "color": INK, "align": PP_ALIGN.LEFT, "space_after": 3})]
    for sy in symbols:
        sym_lines.append((sy, {"size": 11, "color": INK, "align": PP_ALIGN.LEFT, "space_after": 2}))
    text(s, 7.05, 2.52, 5.6, 1.6, sym_lines, align=PP_ALIGN.LEFT, line_spacing=1.12, space_after=2)
    box(s, 0.55, 4.26, 12.23, 0.92,
        [("물리적 의미", {"size": 13, "bold": True, "color": INK, "align": PP_ALIGN.LEFT, "space_after": 3}),
         (meaning, {"size": 12, "color": INK, "align": PP_ALIGN.LEFT})], fc=GREYB, ec=GREY, align=PP_ALIGN.LEFT)
    box(s, 0.55, 5.3, 6.0, 1.8,
        [("이득", {"size": 13, "bold": True, "color": GREEN, "align": PP_ALIGN.LEFT, "space_after": 3}),
         (benefit, {"size": 12, "color": INK, "align": PP_ALIGN.LEFT})], fc=GREENB, ec=GREEN, align=PP_ALIGN.LEFT)
    box(s, 6.78, 5.3, 6.0, 1.8,
        [("한계 → 뒤에서 HOPE가 해결", {"size": 13, "bold": True, "color": RED, "align": PP_ALIGN.LEFT, "space_after": 3}),
         (limit, {"size": 12, "color": INK, "align": PP_ALIGN.LEFT})], fc=REDB, ec=RED, align=PP_ALIGN.LEFT)

concept_slide("A3 · 대안 (1/6)", "Linear Attention", "선형 어텐션 (Katharopoulos 2020)", GREEN,
    "softmax attention은 계산 O(n²)이고 KV 캐시가 문맥 길이에 비례해 커진다. 이를 고정 크기 상태의 O(n) 순환식으로 바꾸고 싶다.",
    r"S_t = S_{t-1} + \phi(k_t)\, v_t^{\top}, \qquad o_t = \phi(q_t)^{\top} S_t",
    ["S_t : 상태 행렬 (d×d) = fast weights", "φ(·) : 양수 feature map (커널)",
     "k_t, v_t : key·value 벡터", "q_t : query,  o_t : 출력"],
    "softmax의 '유사도 합'을 외적 φ(k)vᵀ의 누적으로 대체한다. 상태 S에 (key→value) 연상을 계속 더해, 고정 크기 상태로 무한 문맥을 요약.",
    "시간 O(n)·메모리 O(1)(문맥 무관). 순환 형태라 디코드가 토큰당 상수 비용.",
    "그냥 '더하기'만 한다 → 오래된·충돌하는 연상이 누적 간섭(crosstalk)하고 용량이 포화. (덮어쓰기가 없음)")

concept_slide("A3 · 대안 (2/6)", "DeltaNet", "델타넷 (Schlag/Yang, delta rule)", BLUE,
    "linear attention의 '무한 누적 간섭'을 줄이자. 이미 저장된 연상을 덮어쓰거나 오차만큼 고치자.",
    r"S_t = S_{t-1} - \beta_t\,\big(S_{t-1}k_t - v_t\big)\,k_t^{\top}",
    ["S_{t-1}k_t : 지금 상태가 k_t에 내놓는 값", "(S_{t-1}k_t − v_t) : 예측오차",
     "β_t : 쓰기 강도(0~1) = learning rate", "k_t kᵀ : 그 key 방향에만 갱신"],
    "key k_t로 조회한 값을 목표 v_t에 가깝게 '오차만큼' 수정. 이는 손실 ½‖S·k_t − v_t‖²에 대한 gradient step 한 번(β=학습률)과 정확히 같다.",
    "기존 연상을 덮어써 간섭↓ → 같은 용량으로 더 정확한 회상.",
    "오래된 기억을 '잊는' 장치가 없어 용량이 결국 포화한다. (forget이 없음)")

concept_slide("A3 · 대안 (3/6)", "Gated DeltaNet", "게이트 델타넷 (+forget)", GOLD,
    "DeltaNet엔 망각이 없어 용량이 포화한다. 오래된 기억을 서서히 감쇠(잊기)하자.",
    r"S_t = \alpha_t\, S_{t-1} - \beta_t\,\big(S_{t-1}k_t - v_t\big)\,k_t^{\top}",
    ["α_t : forget gate (0~1, 데이터 의존)", "α_t·S_{t-1} : 상태 전체를 감쇠",
     "β_t, k_t : DeltaNet과 동일", "(S k − v) : 예측오차"],
    "매 스텝 상태를 α_t배로 감쇠(오래된 기억 소거)한 뒤 delta rule로 새 정보를 기입. = decay(정규화)가 붙은 online gradient step.",
    "유한 용량을 재활용 → 긴 문맥에서도 안정, 최신 정보 반영.",
    "갱신 규칙(무엇을·얼마나·잊을지)이 여전히 '고정된 손실·고정 학습률'. 규칙 자체를 학습하지는 못한다.")

concept_slide("A3 · 대안 (4/6)", "SSM (S4)", "상태공간 모델 (Structured State Space)", GREEN,
    "RNN은 순차적·불안정하고 attention은 O(n²). 긴 의존성을 '구조화된 선형 recurrence'로 안정적·병렬적으로 다루자.",
    r"h_t = A\, h_{t-1} + B\, x_t, \qquad y_t = C\, h_t",
    ["h_t : 은닉 상태 (N차원)", "A : 상태전이 (HiPPO 등으로 구조화)",
     "B : 입력→상태,  C : 상태→출력", "x_t 입력,  y_t 출력"],
    "입력을 구조화된 선형 시스템의 상태에 누적. A의 구조(HiPPO)가 과거를 잘 요약하도록 설계돼, convolution으로 병렬 학습·recurrence로 O(1) 디코드.",
    "긴 의존성을 안정적으로 포착, 선형 시간, 병렬 학습 가능.",
    "A,B,C가 입력과 무관(시간 불변) → 내용에 따라 '무엇을 기억할지' 선택하지 못한다.")

concept_slide("A3 · 대안 (5/6)", "Mamba (S6)", "선택적 SSM (Selective State Space)", BLUE,
    "SSM의 A,B,C가 고정이라 내용 선택이 안 된다. 상태 갱신을 입력에 따라 '선택적'으로 만들자.",
    r"(B_t, C_t, \Delta_t) = f(x_t), \qquad \bar{A}_t = \exp(\Delta_t A)",
    ["Δ_t : 입력 의존 시간간격 = 게이트", "B_t, C_t : 입력 의존 투영",
     "Ā_t : 이산화된 상태전이", "f(·) : 작은 투영망"],
    "토큰마다 '얼마나 상태를 갱신·유지할지'를 입력이 정한다(selective). 중요한 토큰은 크게 반영, 무의미하면 건너뜀. hardware-aware 스캔으로 병렬.",
    "내용 기반 선택 → Transformer급 품질 + 선형 시간, 초장문 처리.",
    "여전히 상태 갱신이 '고정된 선형식 + 선택 게이트'. 메모리를 gradient로 '학습'하지는 않는다.")

concept_slide("A3 · 대안 (6/6)", "Mamba-2 (SSD)", "State-Space Duality", GOLD,
    "attention과 SSM이 따로 논다. 둘을 이론으로 잇고(duality), 텐서코어로 더 빠르게 만들자.",
    r"Y = \big(L \circ (C B^{\top})\big)\, X \;\;\Longleftrightarrow\;\; \text{linear SSM recurrence}",
    ["L : 하삼각 마스크 (decay 구조)", "C Bᵀ : attention 유사 행렬",
     "∘ : 원소곱(Hadamard)", "X 입력,  Y 출력"],
    "SSM의 순환식이 '마스킹된 attention'과 수학적으로 등가(duality)임을 보인다 → 훨씬 큰 상태를 텐서코어 친화적으로 빠르게 구현.",
    "attention의 표현력 + SSM의 선형성. 큰 상태를 고속 처리.",
    "여전히 '고정 규칙'의 선형 recurrence. \"sequence layer = 상태 갱신 규칙\" 관점만 확립 — 규칙을 학습으로 승격하는 건 Titans~HOPE의 몫.")

# ==================================================================================
# ============================== PART B — 논문 6편 ==============================
# ==================================================================================
section_divider(p, "PART B", "논문 6편 — 핵심 발명 먼저, 그다음 어떻게 증명했나", GREEN)

def claim_slide(tag, title, ec, claims, flow, proof):
    """핵심 발명/주장 슬라이드: 번호 카드 N개 + 논거 흐름 + 증명 요약."""
    s = slide(p, tag, title, ec)
    text(s, 0.55, 1.12, 12, 0.32, "핵심 발명 / 주장", size=14, color=ec, bold=True, space_after=0)
    y = 1.6
    for i, (t, b) in enumerate(claims, 1):
        box(s, 0.55, y, 0.6, 0.72, str(i), fc=ec, ec=ec, size=17, tcolor=WHITE, bold=True)
        box(s, 1.28, y, 11.47, 0.72,
            [(t, {"size": 13.5, "bold": True, "color": ec, "align": PP_ALIGN.LEFT, "space_after": 3}),
             (b, {"size": 12, "color": INK, "align": PP_ALIGN.LEFT})],
            fc=WHITE, ec=GREY, align=PP_ALIGN.LEFT, lw=1.0)
        y += 0.84
    box(s, 0.55, y + 0.05, 12.2, 0.82,
        [("논거 흐름", {"size": 12.5, "bold": True, "color": INK, "align": PP_ALIGN.LEFT, "space_after": 3}),
         (flow, {"size": 12, "color": INK, "align": PP_ALIGN.LEFT})], fc=BLUEB, ec=BLUE, align=PP_ALIGN.LEFT)
    box(s, 0.55, y + 0.97, 12.2, 0.72,
        [("증명 (뒤에서 논문 그림·표로)", {"size": 12.5, "bold": True, "color": GREEN, "align": PP_ALIGN.LEFT, "space_after": 3}),
         (proof, {"size": 12, "color": INK, "align": PP_ALIGN.LEFT})], fc=GREENB, ec=GREEN, align=PP_ALIGN.LEFT)
    return s

# ------------------------------ B1. Titans ------------------------------
claim_slide("B1 · Titans (2501.00663)", "Titans — test-time에 학습되는 신경망 메모리", GREEN,
    [("문맥을 KV에 쌓지 말고, test-time에 '학습되는' deep neural memory(작은 MLP)에 압축한다",
      "attention의 O(n²)·KV 폭발 대신, 고정 크기 신경망이 무엇을 기억할지 스스로 학습"),
     ("무엇을·얼마나 기억할지 = surprise(gradient) + momentum(과거 surprise) + forgetting(decay)",
      "예측이 놀라운(=gradient 큰) 토큰을 더 기억, 관성으로 이어가고, 오래된 건 잊는다"),
     ("메모리를 아키텍처에 꽂는 3가지 방식: MAC · MAG · MAL",
      "persistent(작업기억) + core(attention) + contextual(장기 메모리)의 결합 방식 차이")],
    "linear/KV의 용량·간섭 한계 → 메모리를 '학습되는 MLP'로 → 무엇을 배우나=k→v 연상(‖M(k)−v‖²) → "
    "얼마나=surprise+momentum+forget → 병렬화=momentum을 associative scan으로 → 어디에 꽂나=MAC/MAG/MAL",
    "언어모델링 perplexity · NIAH(needle) · BABILong(초장문 추론)에서 baseline 상회, 메모리 깊을수록 개선")

def pbox(s, x, y, w, h, header, body, ec=GREEN, fc=None, hsize=12.5, bsize=11.5):
    """라벨 붙은 설명 박스 (Part B 논거 흐름용)."""
    if fc is None:
        fc = {id(BLUE): BLUEB, id(GREEN): GREENB, id(GOLD): GOLDB, id(RED): REDB, id(GREY): GREYB}.get(id(ec), GREENB)
    body_lines = body if isinstance(body, list) else [(body, {})]
    lines = [(header, {"size": hsize, "bold": True, "color": ec, "align": PP_ALIGN.LEFT, "space_after": 3})]
    for it in body_lines:
        t, o = it if isinstance(it, tuple) else (it, {})
        d = {"size": bsize, "color": INK, "align": PP_ALIGN.LEFT, "space_after": 2}; d.update(o)
        lines.append((t, d))
    box(s, x, y, w, h, lines, fc=fc, ec=ec, align=PP_ALIGN.LEFT)

# B1.1 문제 — 비어 있던 가운데
s = slide(p, "B1 · Titans — 문제", "효율이 절실한 바로 그 자리(긴 문맥)에서 품질이 무너진다", GREEN)
pbox(s, 0.55, 1.4, 6.0, 1.15, "attention (정확하지만 비쌈)",
     "토큰마다 과거 전체를 다시 본다 → 계산 O(n²), KV 캐시가 길이에 비례해 증가. 무한정 길게는 불가능.", ec=RED)
pbox(s, 6.78, 1.4, 6.0, 1.15, "linear recurrent (싸지만 압축)",
     "Mamba·DeltaNet: 길이에 비례하는 싼 비용. 그러나 과거 전체를 '고정 크기 state 하나'에 눌러 담는다.", ec=BLUE)
pbox(s, 0.55, 2.68, 12.23, 0.72, "근본 모순",
     "싼 비용이 가장 절실한 구간 = 아주 긴 문맥. 그런데 긴 문맥일수록 작은 state에 제대로 안 담긴다 → 효율이 필요한 자리에서 품질이 무너지도록 설계돼 있다.", ec=RED)
pbox(s, 0.55, 3.55, 6.0, 2.25, "직전 시도 TTT의 3가지 결핍",
     [("state를 작은 모델 weights로 보고 gradient로 갱신 — LM 스케일에서 처음 작동. 그러나:", {"space_after": 4}),
      ("① 지우기가 없다 (순수 SGD → write만 무한 누적 → 포화)", {}),
      ("② optimizer가 기억을 못 한다 (토큰마다 자기 gradient 하나만 반영)", {}),
      ("③ deep memory의 이득을 실험으로 답하지 않았다", {})], ec=GOLD)
pbox(s, 6.78, 3.55, 6.0, 2.25, "다른 한쪽 — DeltaNet 계열",
     [("memory를 matrix로 묶어 '정확한 병렬 학습 공식'을 얻음. Gated DeltaNet은 forget gate까지 추가.", {"space_after": 4}),
      ("그러나 딜레마: '표현력 있는 갱신 규칙 + 깊은 비선형 memory'와 '병렬 학습 가능성'이 양립 불가처럼 보였다.", {}),
      ("→ 그 가운데가 비어 있었다.", {"bold": True, "color": RED})], ec=BLUE)
pbox(s, 0.55, 5.95, 12.23, 0.85, "Titans의 자리",
     "표현력(깊은 비선형 memory + 똑똑한 갱신 규칙)과 병렬 학습 가능성을 동시에 — 그 '빈 가운데'를 채운다.", ec=GREEN)

# B1.2 아이디어 — memory는 weights다
s = slide(p, "B1 · Titans — 아이디어", "memory = 작은 신경망의 weights, 그 weights가 recurrent state", GREEN)
pbox(s, 0.55, 1.4, 12.23, 0.82, "관점: 모든 sequence 모델을 세 요소로 분해",
     "memory 구조 · write(갱신) 연산 · read(검색) 연산. 질문: test-time에 memory를 gradient로 갱신하고 그 규칙을 잘 설계하면 고정 state의 한계를 넘을 수 있지 않을까?", ec=GREEN)
pbox(s, 0.55, 2.34, 6.0, 1.5, "memory가 푸는 문제 = associative memory regression",
     [("작은 MLP(long-term memory module, LMM)의 weights = state.", {}),
      ("토큰을 key/value로 투영하고, memory가 key를 넣으면 value를 재생하도록 학습:", {})], ec=GREEN)
fit_image(s, mp.eq(r"\ell(M_{t-1};x_t) = \big\| M_{t-1}(k_t) - v_t \big\|^2", pt=17)[0], 7.0, 2.55, 5.6, 0.9)
pbox(s, 0.55, 3.96, 6.0, 1.75, "결정적 구분 — 두 층위",
     [("key/value/query 투영 weights = inner 문제의 hyperparameter.", {}),
      ("→ inference 중엔 절대 안 움직이고 pre-training 값으로 고정.", {"bold": True, "color": BLUE}),
      ("실제로 매 토큰 움직이는 건 memory의 weights뿐이다.", {"bold": True, "color": GREEN})], ec=BLUE)
pbox(s, 6.78, 3.96, 6.0, 1.75, "무엇을 얼마나 세게 외울까 = surprise",
     [("지금 토큰이 이미 저장된 내용을 얼마나 '위반'하는가.", {}),
      ("= 손실의 gradient 크기.", {"bold": True}),
      ("gradient가 클수록 새롭고 예상 밖 → 더 세게 외운다.", {})], ec=GOLD)
pbox(s, 0.55, 5.83, 12.23, 0.95, "왜 이게 새로운가",
     "고정 state(벡터/행렬)를 '학습되는 신경망'으로 바꾸고, 무엇을 기억할지를 '손실의 gradient'가 정하게 했다. 뒤 논문들이 이 손실·규칙·구조를 각각 일반화한다.", ec=GREY)

# B1.3 방법 — 세 성분 update
s = slide(p, "B1 · Titans — 방법", "write 연산 = 'momentum + weight decay가 붙은 gradient descent 한 스텝'", GREEN)
tt_eqs = [
    (r"u_t = \nabla_M\, \ell(M_{t-1}; x_t)", "① surprise = 지금 토큰의 gradient (놀라움)"),
    (r"S_t = \eta_t\, S_{t-1} - \theta_t\, u_t", "② momentum = 과거 surprise 누적(놀란 직후도 기억)"),
    (r"M_t = (1-\alpha_t)\, M_{t-1} + S_t", "③ weight decay = α로 오래된 기억 감쇠(잊기)"),
    (r"y_t = M_t(q_t)", "read = query q_t를 갱신된 memory에 통과"),
]
box(s, 0.55, 1.42, 7.35, 3.05, "", fc=WHITE, ec=GREY)
text(s, 0.75, 1.5, 6.8, 0.3, "master update (η, θ, α는 상수가 아니라 토큰마다 학습된 gate)", size=11.5, color=INK, bold=True, space_after=0, align=PP_ALIGN.LEFT)
cy = 1.9
for ltx, note in tt_eqs:
    _pp, ww, hh = mp.eq(ltx, pt=14)
    fit_image(s, _pp, 0.9, cy, min(ww, 6.4), 0.34)
    text(s, 0.9, cy + 0.36, 6.9, 0.26, note, size=10, color=MUTE, align=PP_ALIGN.LEFT, space_after=0)
    cy += 0.62
pbox(s, 8.1, 1.42, 4.68, 1.45, "왜 momentum인가",
     "큰 surprise 직후엔 손실이 이미 낮아 gradient가 급감 → 이어지는 중요한 토큰이 약하게 저장됨. 과거 surprise를 buffer에 누적해 '놀란 시간대 전체'를 기억.", ec=GOLD, bsize=11)
pbox(s, 8.1, 2.97, 4.68, 1.5, "왜 weight decay인가",
     "수백만 토큰을 다루면 깊은 memory도 포화. α_t배만 남겨 잊는다. 이는 Mamba-2·GLA·Gated DeltaNet의 gate를 '임의의 깊은 memory'로 일반화한 것.", ec=GOLD, bsize=11)
pbox(s, 0.55, 4.6, 12.23, 0.72, "첫 번째 발명",
     "optimizer의 내부 state(momentum buffer S_t)를 sequence 레이어의 recurrent state로 '승격'시켰다. ablation 기여도: weight decay > momentum.", ec=GREEN)
pbox(s, 0.55, 5.45, 12.23, 1.33, "A2와의 정확한 대응",
     [("이 세 식은 A2의 'momentum + weight decay가 붙은 gradient descent'와 글자 그대로 같은 모양이다.", {"space_after": 3}),
      ("다만 대상이 '모델 파라미터'가 아니라 '추론 중 memory M'이다 — 학습(training)의 도구를 추론(inference) 안으로 가져왔다.", {"bold": True, "color": BLUE})], ec=BLUE)

# B1.4 outer vs inner loop (이 섹션의 심장)
s = slide(p, "B1 · Titans — outer vs inner loop", "그 gate들은 대체 누가 학습하나? — 두 개의 loop", GREEN)
pbox(s, 0.55, 1.4, 6.0, 2.35, "바깥 loop = 평범한 pre-training (딱 한 번, 배포 후 동결)",
     [("학습 대상: key/value/query 투영, gate를 뱉는 head, persistent token, attention 블록, 그리고 memory의 '초기 상태'.", {}),
      ("도구: AdamW · 대량 데이터.", {}),
      ("→ 이 전부가 배포 뒤엔 고정(frozen).", {"bold": True, "color": BLUE})], ec=BLUE)
pbox(s, 6.78, 1.4, 6.0, 2.35, "안쪽 loop = memory weights 갱신 (매 토큰, inference 중에도)",
     [("sequence마다 진화하는 텐서는 딱 둘: memory weights W_t 와 momentum buffer S_t.", {}),
      ("나머지는 전부 동결.", {}),
      ("→ 매 토큰 memory를 고쳐 쓴다.", {"bold": True, "color": GREEN})], ec=GREEN)
pbox(s, 0.55, 3.9, 12.23, 1.05, "가장 중요한 사실 — outer gradient가 unroll된 inner update를 '관통'한다",
     "key 투영을 조금 바꾸면 → 매 토큰 inner gradient가 바뀌고 → memory 궤적 전체가 바뀌어 → 최종 손실이 바뀐다. 그 연쇄 전체의 gradient가 투영을 학습시킨다. (= 'optimizer를 미분한다')", ec=GOLD)
pbox(s, 0.55, 5.05, 12.23, 0.82, "gate에 대한 답",
     "η·β·α의 '값'은 inference 중 매 토큰 새로 계산되지만, 그 값을 만드는 '함수'는 pre-training에서만 학습되고 이후 동결. 학습은 함수를 만들고, inference는 그 함수로 memory를 고쳐 쓴다.", ec=GREEN)
pbox(s, 0.55, 5.95, 12.23, 0.85, "그 결과 — decode의 정의가 바뀐다",
     "더는 'read-only forward + KV append'가 아니라 forward + backward(gradient) + optimizer step + read. backward가 decode 안으로 들어온다. (→ Part C에서 이 비용을 잰다.)", ec=RED)

# B1.5 chunkwise 병렬화
s = slide(p, "B1 · Titans — 병렬화", "매 토큰 순차 gradient는 가속기 최악 → chunkwise로 GEMM화", GREEN)
fig_with_caption(s, "2501.00663", 1, 0.55, 1.5, 6.05, 3.6,
                 "Figure 1 — chunk 안은 병렬, chunk 경계만 비선형 재진입")
pbox(s, 6.78, 1.5, 6.0, 1.5, "핵심 트릭",
     "sequence를 크기 C의 chunk로 자르고, chunk 안 모든 gradient를 'chunk 시작 시점의 동결된 weights'에서 한꺼번에 평가 → C개를 동시에 = GEMM 2개. C가 write GEMM의 내적 차원 = arithmetic intensity의 손잡이.", ec=GREEN, bsize=11)
pbox(s, 6.78, 3.1, 6.0, 2.0, "세 조각",
     [("• chunk 안 갱신은 linear → cumsum, 경계 재진입은 nonlinear → gradient", {}),
      ("• momentum buffer = 1차 선형 recurrence → Mamba의 selective scan과 같은 모양 → parallel scan(log-depth)", {}),
      ("• weight decay = 누적 곱을 대각으로 접은 matmul로 흡수", {})], ec=BLUE, bsize=10.5)
pbox(s, 0.55, 5.25, 12.23, 1.0, "핵심 협상",
     "chunk 안은 linear라서 병렬, chunk 사이는 nonlinear라서 표현력. 순차 임계경로가 L이 아니라 L/C번의 MLP 적용으로 줄고, 그 비선형 경계가 표현력의 근거. (DeltaNet이 closed-form 위해 전부 linear로 남은 것과 의도적으로 다른 선택.)", ec=GREY)

# B1.6 합성 — MAC/MAG/MAL
s = slide(p, "B1 · Titans — 합성", "attention(정밀 단기) + LMM(서서히 잊는 장기)을 합치는 3가지", GREEN)
fig_with_caption(s, "2501.00663", 2, 0.55, 1.5, 6.05, 3.5,
                 "Figure 2 — MAC: 검색한 과거를 현재 앞에 붙여 attention이 함께 봄")
pbox(s, 6.78, 1.5, 6.0, 1.35, "MAC (Memory as Context) — 가장 강함",
     "contextual 갈래가 과거를 검색해 현재 입력 앞에 붙이고, attention이 '검색된 역사 + 현재'를 함께 본다 → 장기 정보 필요 여부를 토큰 단위로 판단. attention이 write filter 역할.", ec=GREEN, bsize=10.5)
pbox(s, 6.78, 2.95, 6.0, 1.1, "MAG · MAL",
     "MAG: sliding-window attention과 LMM을 병렬로 돌려 gate로 섞음. MAL: LMM을 attention 앞 레이어로 직렬 — 논문이 '가장 약함'이라 명시(파이프라인 힘이 각 단으로 상한).", ec=BLUE, bsize=10.5)
pbox(s, 6.78, 4.15, 6.0, 0.95, "숨은 한 방",
     "H3 이래 대부분의 기존 hybrid(Samba·Griffin)가 사실상 MAL 모양 → 'MAC/MAG > MAL' 결과는 hybrid 설계 관행 자체에 대한 기소장.", ec=GOLD, bsize=10.5)
pbox(s, 0.55, 5.2, 6.0, 1.05, "세 갈래 (공통)",
     "persistent(입력 독립·동결 작업기억) + core(단기 attention·ICL) + long-term memory(위 식으로 test-time 학습).", ec=GREY, bsize=11)
pbox(s, 6.78, 5.2, 6.0, 1.05, "test-time 역할 분담",
     "memory = 여전히 학습 중 · attention = in-context learning · persistent = 고정. (그림의 눈송이 = 동결.)", ec=GREY, bsize=11)

# B1.7 결과
s = slide(p, "B1 · Titans — 결과", "state가 길이에 무관하게 일정 → 초장문에서 훨씬 큰 모델을 이긴다", GREEN)
fig_with_caption(s, "2501.00663", 6, 0.55, 1.5, 6.05, 3.5,
                 "Figure 6 — BABILong: MAC가 GPT-4·Mamba 2.8B·RWKV 7B를 앞섬(적은 파라미터로)")
fig_with_caption(s, "2501.00663", 7, 6.85, 1.5, 5.9, 3.5,
                 "Figure 7 — 메모리 깊이(1→4층)↑ → 모든 길이에서 perplexity↓")
pbox(s, 0.55, 5.15, 12.23, 1.1, "읽는 법 (정직성: 방향·비율 위주)",
     [("• S-NIAH: Titans는 2K~16K 전 구간 높은 정확도. Mamba-2는 긴 구간 붕괴(얕은 state), DeltaNet은 과제별 붕괴(forget 없음), TTT는 처짐(retention gate 없음) — 세 성분이 각 경쟁자를 이긴 이유와 일치.", {}),
      ("• fine-tuning BABILong: 작은 MAC가 파라미터 ~70배인 Llama3.1-8B/70B·Qwen2.5-72B를 넘어 2M 토큰 너머까지 유지. (비통제 비교라 인과는 ablation 범위에서만 주장.)", {})], ec=GREY, bsize=10.5)

# B1.8 한계 & 다음
s = slide(p, "B1 · Titans — 한계 & 다음", "point design이 분해되어 좌표계가 된다 → Miras", GREEN)
pbox(s, 0.55, 1.4, 6.0, 3.55, "정직한 한계 5가지",
     [("① point design — 왜 하필 제곱오차 손실? 왜 SGD+momentum+decay? 논거 없음.", {}),
      ("② 미명시 세부 — gate 함수형·memory 폭·chunk 크기(v1 부재).", {}),
      ("③ 근사 비용 미측정 — chunk 안 gradient는 stale, chunk sweep 없음. train(chunk)/serve(토큰) 불일치 제기조차 안 됨 (→ TNT의 출발점).", {}),
      ("④ 표현력 정리에 증명 없음.", {}),
      ("⑤ 증거가 전부 ≤760M·4K 훈련길이. decode wall-clock 수치는 전무(→ Part C).", {})], ec=RED, bsize=10.5)
pbox(s, 6.78, 1.4, 6.0, 1.7, "시스템 관점 핵심 교환",
     [("state가 길이에 무관하게 일정 (memory weights + momentum buffer = 고정 크기).", {"bold": True, "color": GREEN}),
      ("KV cache는 길이에 비례 → 2M 토큰 능력이 정확히 이 교환 위에 섬. 대가: momentum이 state 2배, decode가 read-modify-write.", {})], ec=GREEN, bsize=11)
pbox(s, 6.78, 3.2, 6.0, 1.75, "다음으로의 연결 — Miras",
     [("Appendix C가 Gated DeltaNet·Longhorn·RWKV-7·TTT를 전부 자기 update의 특수 사례로 '회수'.", {}),
      ("momentum 끄면 Gated DeltaNet, forget+momentum 끄면 TTT …", {}),
      ("→ 계보 전체가 '하나의 설계 공간의 점들'. 그 좌표축을 명명한 것이 Miras.", {"bold": True, "color": GOLD})], ec=GOLD, bsize=10.5)
pbox(s, 0.55, 5.1, 12.23, 1.15, "한 줄 요약",
     "Titans = memory를 weights로 보고, surprise를 gradient로 정의, momentum으로 시간 흐름, weight decay로 잊기, 깊은 MLP에 넣고도 chunkwise로 병렬 학습. 남은 질문(왜 이 손실? 왜 이 규칙?)이 Miras·Atlas·HOPE로 이어진다.", ec=GREEN)

# ------------------------------ B2. Miras ------------------------------
claim_slide("B2 · Miras (2504.13173)", "Miras — 메모리는 하나의 online 최적화; 4축 설계공간", GREEN,
    [("sequence layer = 하나의 online 최적화 문제로 통일해 볼 수 있다",
      "Titans·DeltaNet·linear attention 등은 모두 '어떤 목적을 어떤 규칙으로 최소화하나'의 특수해"),
     ("설계는 4개의 축으로 분해된다: attentional bias · retention · architecture · algorithm",
      "① 무엇을 기억(손실) ② 무엇을 잊기(retention) ③ 메모리 구조 ④ 갱신 알고리즘"),
     ("이 축에서 새 인스턴스를 만든다: Moneta · Yaad · Memora (FTRL·Learning-Retaining)",
      "발명이라기보다 '설계 공간의 존재 증명 + 각 축의 효과를 실험으로 확인'")],
    "Titans 재해석(surprise=attentional bias, forget=retention) → 축을 일반화 → 손실을 L2에서 Lp/robust로, "
    "retention을 다양하게 → 새 조합 Moneta/Yaad/Memora → 축별 ablation",
    "모델 크기·문맥 길이 스케일링(Fig.3), 손실차수 p·retention q의 효과(Fig.4)로 각 축이 실제로 성능을 가른다")

# B2.1 문제 + 두 관찰
s = slide(p, "B2 · Miras — 문제와 두 관찰", "Titans는 한 점만 찍고 끝났다 — 왜 L2? 왜 momentum? forgetting이 맞나?", GREEN)
pbox(s, 0.55, 1.4, 12.23, 1.15, "문제 — 특수 사례는 보였는데 '일반형의 좌표축'이 공백",
     [("Titans는 존재 증명이었지만 3가지에 답하지 않았다: ① 왜 L2 loss(내부 목적함수)?  ② 왜 momentum·weight decay(딥러닝 관행을 그냥 이식)?  ③ forgetting이 옳은 개념인가?", {}),
      ("Titans 스스로 'L2 너머의 목적함수'와 '더 나은 optimizer'를 open problem으로 남겼다.", {"bold": True, "color": RED})], ec=RED)
pbox(s, 0.55, 2.7, 6.0, 2.15, "관찰 1 — 전부 associative memory",
     [("거의 모든 시퀀스 모델 = key를 넣으면 value를 돌려주는 연상 메모리.", {}),
      ("내부 목적함수는 단 2종뿐이었다: dot-product 유사도 / L2 regression.", {}),
      ("이 내부 loss를 attentional bias로 명명 = '이 메모리가 무엇을 우선해서 기억하는가'.", {"bold": True, "color": GREEN})], ec=GREEN)
pbox(s, 6.78, 2.7, 6.0, 2.15, "관찰 2 — forgetting은 없다, retention만 있다",
     [("기존 forget gate = 전부 L2 정규화의 특수형.", {}),
      ("하는 일은 '지우기'가 아니라 '새것 배우기 vs 이전 상태 머무르기'의 저울질 — 모델은 유지하지 않기로 결정할 뿐.", {}),
      ("→ forget gate를 retention gate로 개명.", {"bold": True, "color": GOLD})], ec=GOLD)
pbox(s, 0.55, 5.0, 12.23, 0.85, "→ 이어지는 결론",
     "이 두 관찰을 밀고 나가면 시퀀스 모델 설계가 4개의 독립적 축으로 분해된다. '시퀀스 모델 = 무언가를 최소화하는 연상 메모리'.", ec=BLUE)

# B2.2 4축
s = slide(p, "B2 · Miras — 4축 설계공간", "메모리를 4개의 독립적 '선택'으로 분해 — retention이 결정적 레버", GREEN)
fig_with_caption(s, "2504.13173", 1, 0.55, 1.5, 5.6, 4.2,
                 "Figure 1 — 매 token 최소화되는 안쪽 목적 = attentional bias + retention, GD로 푼다")
axes = [
    ("① Memory architecture", "메모리 구조 — vector · matrix · deep MLP."),
    ("② Attentional bias", "무엇을 우선해 기억? = 내부 목적함수 L. dot-product·L2·Lp·robust."),
    ("③ Retention gate", "새것 배우기 vs 옛것 유지의 균형. long-context를 좌우하는 결정적 레버."),
    ("④ Memory learning algorithm", "그 목적을 어떻게 푸나 — GD·momentum·Newton·closed-form."),
]
ay = 1.55
for t, b in axes:
    box(s, 6.5, ay, 6.25, 0.98, [(t, {"size": 12.5, "bold": True, "color": GREEN, "align": PP_ALIGN.LEFT, "space_after": 3}),
        (b, {"size": 11, "color": INK, "align": PP_ALIGN.LEFT})], fc=GREENB, ec=GREEN, align=PP_ALIGN.LEFT)
    ay += 1.08
pbox(s, 0.55, 5.95, 12.23, 0.85, "의의",
     "Titans를 특수점으로 품는 지도. '무엇을 바꾸면 무엇이 좋아지나'를 조직적으로 탐색 가능하게 했다. 이후 모든 논문이 이 어휘로 자기 위치를 설명한다.", ec=BLUE)

# B2.3 전부가 이 공간의 점
s = slide(p, "B2 · Miras — 전부가 이 공간의 점", "attentional bias 하나만 바꿔도 기존 모델들이 재유도된다", GREEN)
pbox(s, 0.55, 1.4, 6.0, 1.7, "bias = dot-product  → '덧쓰기'",
     [("'비슷한 key에 비례해 크게 써라' → 순수하게 더하기만 하는 write.", {}),
      ("= linear attention · RetNet · Mamba-2 (옛 Hebbian 덧쓰기).", {"bold": True, "color": GREEN})], ec=GREEN)
pbox(s, 6.78, 1.4, 6.0, 1.7, "bias = L2 regression  → '고쳐쓰기'",
     [("'이미 있던 걸 고쳐 써라' → 예측오차만큼 수정.", {}),
      ("= DeltaNet · Gated DeltaNet · RWKV-7. (A3의 '덧쓰기→고쳐쓰기'가 '목적함수만 바꿨을 뿐'으로 압축.)", {"bold": True, "color": BLUE})], ec=BLUE)
pbox(s, 0.55, 3.25, 12.23, 1.15, "softmax attention도 이 공간의 한 점",
     [("L2 regression을 '압축 없이 비모수적'으로 푼 극한 → retention이 없다 → state(KV cache)가 계속 커진다.", {}),
      ("attention의 메모리가 길이에 선형으로 느는 이유 = '압축하지 않는 연상 메모리이기 때문'으로 설명된다.", {"bold": True, "color": RED})], ec=RED)
pbox(s, 0.55, 4.55, 12.23, 1.7, "한 문장 정리 + 정직성",
     [("덧쓰기(linear)·고쳐쓰기(delta)·안 압축(softmax)이 전부 'attentional bias 축의 세 좌표'로 통일된다.", {"bold": True, "color": GREEN, "space_after": 3}),
      ("다만 이 표의 등호는 '원 설계 방정식' 수준의 동일시다. retention 열의 L2는 세부가 다른 gate들을 뭉뚱그린 것 — 분류학으론 정확하되 구현 완전등가로 읽으면 과독(논문 각주도 이 해상도 한계를 인정).", {})], ec=GREY)

# B2.4 Moneta / Yaad / Memora
s = slide(p, "B2 · Miras — 빈 칸을 채운 3모델", "bias·retention 축에 새 선택지를 넣다 (optimizer는 일부러 plain GD)", GREEN)
fig_with_caption(s, "2504.13173", 2, 0.55, 1.5, 5.7, 3.9,
                 "Figure 2 — Moneta/Yaad/Memora: recurrent · SWA hybrid · layer 설계")
pbox(s, 6.4, 1.5, 6.35, 1.15, "Moneta — Lp loss",
     "Lp(p=3)로 잘 안 떠오르는 '놀라운 token'에 L2보다 날카로운 기억 압력 + Lq(q=4) 정규화로 메모리 norm을 통제된 껍질에 붙잡음.", ec=GREEN, bsize=10.5)
pbox(s, 6.4, 2.72, 6.35, 1.15, "Yaad — Huber loss (robust)",
     "오차가 학습된 threshold보다 크면(=outlier token) 그 크기를 깎아 기억 → 노이즈·적대 구간이 자기 크기만큼 메모리를 흔들지 못한다.", ec=BLUE, bsize=10.5)
pbox(s, 6.4, 3.94, 6.35, 1.45, "Memora — KL retention (가장 우아)",
     [("retention을 KL divergence로 → 메모리를 확률 simplex에 가둠. 갱신이 softmax 형태라 state가 항상 양수·합=1.", {}),
      ("→ 문맥이 아무리 길어도 state가 발산 불가 (retention-as-renormalization).", {"bold": True, "color": GOLD})], ec=GOLD, bsize=10.5)
pbox(s, 0.55, 5.55, 12.23, 0.72, "실험 설계 의도",
     "셋 다 optimizer는 plain GD(momentum 일부러 제거) — expressivity를 optimizer가 아니라 '목적함수와 retention'에 싣고 그 효과만 보려는 통제.", ec=GREY)

# B2.5 결과
s = slide(p, "B2 · Miras — 결과", "attention 한 layer도 안 쓰고 attention-hybrid를 이긴다", GREEN)
fig_with_caption(s, "2504.13173", 3, 0.55, 1.5, 6.05, 3.5,
                 "Figure 3 — scaling: baseline은 16K 넘으면 perplexity 도로 상승, 세 변형은 완만 유지")
pbox(s, 6.85, 1.5, 5.9, 1.5, "헤드라인",
     "더 나은 bias+retention이면 attention 없이도 hybrid를 이긴다. 1.3B/100B token에서 순수 recurrent Yaad가 Gated DeltaNet은 물론 hybrid Samba·GDN-H2까지 perplexity에서 이겼다.", ec=GREEN, bsize=11)
pbox(s, 6.85, 3.1, 5.9, 1.4, "ablation의 교훈",
     "p(bias)를 바꾸면 성능은 오르내리나 scaling 모양은 불변. q(retention)를 바꾸면 scaling 패턴 자체가 바뀐다 → 기여순위 retention > deep memory > bias.", ec=GOLD, bsize=11)
pbox(s, 0.55, 5.15, 12.23, 1.1, "'retention이 결정적 레버'의 직접 증거 + 정직한 한계",
     "위 ablation이 관찰 2(retention이 핵심)의 가장 직접적 증거. 한계: 실증이 1.3B·100B token에서 멈춘다. 7B+·RLHF 후·exact-copy recall 실무에서도 순서가 유지되는지는 미답, 효율 증거도 FLOPs 곡선 하나뿐(wall-clock 없음).", ec=GREY)

# B2.6 요약 + 다음
s = slide(p, "B2 · Miras — 요약 & 다음", "진짜 기여는 세 모델이 아니라 좌표계 → Atlas가 빈 축을 채우러 간다", GREEN)
pbox(s, 0.55, 1.4, 12.23, 0.9, "진짜 기여 = 좌표계",
     "Moneta/Yaad/Memora 수치는 1.3B에서 멈추지만, attentional bias·retention gate·4축이라는 '어휘'는 이후 모든 논문이 무상으로 가져다 쓴다. Titans=존재 증명, Miras=그 move가 연 공간의 좌표계.", ec=GREEN)
pbox(s, 0.55, 2.45, 12.23, 2.4, "Miras가 다음 논문에 넘긴 숙제 3가지 = Atlas의 목차",
     [("① 비어 있는 optimizer 축 — 세 모델 다 plain GD(momentum 제거). 더 나은 optimizer(Titans momentum·Newton법)가 새 bias·gate와 합쳐지면? → Atlas는 inner loop에 Muon을 심는다.", {}),
      ("② token 단위 목적함수 — 목적함수가 여전히 'token 하나'만 본다. 최근 여러 token이 함께 잘 저장됐는지 아무도 안 물음. → Atlas는 Omega rule(최근 여러 token 함께 최적화)을 도입.", {}),
      ("③ capacity 이론 부재 — online 최적화 도구를 통째로 수입했지만 그 보증이 요구하는 convexity를 2-layer MLP가 깬다. → Atlas는 정식 capacity 이론을 붙인다.", {})], ec=GOLD)
pbox(s, 0.55, 5.0, 12.23, 0.82, "한 줄",
     "Miras가 지도를 그렸다면, Atlas는 그 지도의 '빈 축'을 최적화 이론으로 채우러 간다.", ec=BLUE)

# ------------------------------ B3. Atlas ------------------------------
claim_slide("B3 · Atlas (2505.23735)", "Atlas — 토큰이 아니라 '문맥'을 기억하고, 용량을 키운다", GREEN,
    [("Omega rule: 개별 토큰이 아니라 sliding-window '문맥 전체'를 한 번에 기억",
      "γ_{t,i} 가중으로 최근 c개 토큰을 함께 최적화 — 'learning to memorize the context'"),
     ("feature map φ_p로 메모리 용량을 matrix 한계(√params) 너머 d^p까지 확장",
      "polynomial feature map으로 표현력↑ (softmax는 사실상 무한 용량의 극단)"),
     ("메모리 학습을 GD가 아니라 Muon(근사 2차) optimizer로",
      "곡률 정보를 써 더 잘·빠르게 메모리를 적합 — A2의 optimizer 발전을 메모리 학습에 적용")],
    "메모리 용량이 병목(Titans Fig.7의 연장) → Hopfield 용량 이론 → feature map 투영으로 확장 → "
    "문맥 단위 목적(Omega) → 최적화도 Muon으로 → DeepTransformers/Dot 변형",
    "BABILong에서 Titans 초과(Fig.4), associative recall 용량 실험(Fig.7), 문맥·FLOPs 스케일링(Fig.8)")

# B3.1 문제 세 가지
s = slide(p, "B3 · Atlas — 문제 세 가지", "목표 자체를 바꾼다: '이 token 하나' → '최근 문맥 전체를 함께'", GREEN)
pbox(s, 0.55, 1.42, 12.23, 0.72, "관점 — Miras가 비워둔 optimizer 축 포함, 세 구멍을 한 논문에서 메운다",
     "무엇을 목표로 삼는가(objective) · 얼마나 담을 수 있는가(capacity) · 얼마나 잘 채우는가(optimizer).", ec=GREEN)
pbox(s, 0.55, 2.28, 4.0, 2.5, "① online 갱신 (근시안)",
     [("memory가 매 step 현재 token 하나만 보고 최적화, 과거는 retention gate로 흐릿하게 유지될 뿐.", {}),
      ("→ token 단위 greedy 암기. '문맥 전체가 잘 저장됐나'는 아무도 안 묻는다.", {"bold": True, "color": RED})], ec=RED, bsize=11)
pbox(s, 4.68, 2.28, 4.0, 2.5, "② 용량 한계",
     [("단순 행렬 memory는 파라미터를 아무리 많이 줘도, 저장 가능한 연상(k→v) 수가 key 차원 d_k 수준에서 막힌다.", {}),
      ("→ state를 키워 파라미터를 더 쓰는 것 ≠ 용량을 늘리는 것.", {"bold": True, "color": RED})], ec=RED, bsize=11)
pbox(s, 8.78, 2.28, 4.0, 2.5, "③ 관리 빈약",
     [("inner optimizer가 거의 전부 1차 gradient descent.", {}),
      ("→ loss 표면의 나쁜 local minimum에 앉으면 질 낮은 k→v 매핑을 그대로 저장.", {"bold": True, "color": RED})], ec=RED, bsize=11)
pbox(s, 0.55, 4.95, 12.23, 0.9, "세 처방 = 부제의 세 단어",
     "Omega rule(목표를 window로 넓힘) · feature map(용량 상한을 옮김) · Muon(그 상한 안에서 도달 품질을 올림). 이 셋의 조합이 이 섹션 전체.", ec=BLUE)

# B3.2 아이디어 + 용량 이론
s = slide(p, "B3 · Atlas — 목표 전환 + 용량 이론", "context를 기억한다, 그리고 용량을 정리로 정의한다", GREEN)
fig_with_caption(s, "2505.23735", 1, 0.55, 1.5, 5.7, 3.5,
                 "Figure 1 — token 하나(왼) vs 최근 c개 함께(오, Omega). 오른쪽 끝 = attention도 windowed regression")
pbox(s, 6.4, 1.5, 6.35, 1.35, "핵심 전환",
     "여러 token에 걸친 사건은 어느 하나도 개별로는 놀랍지 않지만 묶음으로는 중요할 수 있다('어제 김대리가 예산 승인'). token 단위 loss는 이를 구조적으로 놓친다 → 최근 c개를 함께 최적화.", ec=GREEN, bsize=10.5)
pbox(s, 6.4, 2.95, 6.35, 1.55, "용량 이론 (이 계보 최초)",
     [("Prop 1: 행렬 memory W는 GD로 최적화해도 최대 O(d_k)개만 정확 저장(순수 rank 논증).", {}),
      ("Prop 2: polynomial feature map φ_p로 차원을 Θ(d_kᵖ)로 → 용량 O(d_kᵖ), 어떤 optimizer로도 이 상한을 못 넘는다.", {"bold": True, "color": GOLD})], ec=GOLD, bsize=10)
pbox(s, 0.55, 5.15, 12.23, 1.05, "함의 — attention이 이기는 이유가 정리로 설명된다",
     "무한 차원 feature(Kronecker self-tensoring)를 만들면 exp(q·k)=φ*(q)·φ*(k) → softmax attention = 무한 차원 위 associative memory = 용량 무한. attention이 긴 문맥 recall에서 고정 state를 이기는 건 신비가 아니라 '용량 상한의 차이'. (선형독립 key 정확 보간이라는 이상화 기준 → 근사 대리지표로 읽기.)", ec=BLUE)

# B3.3 Omega rule
s = slide(p, "B3 · Atlas — Omega rule", "delta rule을 window 크기만큼 rank로 일반화 (c=1이면 Titans)", GREEN)
fig_with_caption(s, "2505.23735", 2, 0.55, 1.5, 6.05, 3.5,
                 "Figure 2 — SWA(비모수) vs Atlas: window c=1,4,7로 키우면 의존이 하삼각 전체로 번짐")
fit_image(s, mp.eq(r"\min_{M}\ \sum_{i=t-c+1}^{t} \gamma_{t,i}\,\big\|M(k_i)-v_i\big\|^2", pt=15)[0], 6.9, 1.55, 5.85, 0.7)
pbox(s, 6.78, 2.35, 6.0, 1.35, "기호",
     [("γ_{t,i} = window gate. 0이면 그 token을 최적화에서 hard pruning(admission control), 1이면 온전히 포함.", {}),
      ("c = window 길이. sliding window라 step당 gate 수가 c개로 상수 → recurrent 장점 유지.", {})], ec=GREEN, bsize=10.5)
pbox(s, 6.78, 3.78, 6.0, 1.22, "대수적 정체",
     [("DeltaNet 전이 = key 1개짜리 rank-1 수정. Omega rule = c개 key 합 = rank-c 수정.", {}),
      ("c=1로 두면 정확히 Titans (Titans = Omega의 window-1 특수 사례).", {"bold": True, "color": BLUE})], ec=BLUE, bsize=10.5)
pbox(s, 0.55, 5.15, 12.23, 1.05, "왜 global이 아니라 window인가",
     "문맥 전체 global 최적화는 (i) 매 step 모든 과거 key/value를 들고 있어야 해 고정 state 존재 이유가 사라지고 (ii) 무관 구간을 잘라낼 gate가 없다. Omega는 sliding window로 둘 다 해결(γ가 pruning gate).", ec=GREY)

# B3.4 Muon + Transformer 가족
s = slide(p, "B3 · Atlas — Muon optimizer + Transformer 가족", "근사 2차 관리, 그리고 같은 기계로 Transformer 재유도", GREEN)
fig_with_caption(s, "2505.23735", 3, 0.55, 1.5, 5.7, 3.6,
                 "Figure 3 — Atlas Layer 배선 + hybrid(MAG/MAL). Atlas/Atlas++/OmegaNet, DeepTransformers·Dot")
pbox(s, 6.4, 1.5, 6.35, 1.85, "Muon inner optimizer (③ 관리)",
     [("momentum buffer S에 windowed gradient를 쌓고 Newton-Schulz 반복을 κ번 → semi-orthogonal(UVᵀ)로 수렴 → update singular value 균등화 = 2차 정보 근사.", {}),
      ("κ = NS 반복수 = 'internal test-time compute' dial. 더 돌리면 품질↑·inference FLOPs↑ (state 안 건드리고 연산↔품질 교환). 실전 κ=5.", {"bold": True, "color": GOLD})], ec=GOLD, bsize=10)
pbox(s, 6.4, 3.5, 6.35, 1.6, "부산물 — Transformer 두 일반화 가족",
     [("attention = Nadaraya-Watson kernel regression의 비모수해로 재서술.", {}),
      ("feature map을 정확한 exp φ*로 → DeepTransformers(unnormalized softmax의 strict 일반화). 거기에 Omega rule → Dot = 'error-correcting attention'.", {})], ec=BLUE, bsize=10)
pbox(s, 0.55, 5.25, 12.23, 0.95, "병렬화 유지",
     "window는 banded mask 하나로 처리(각 token gradient 1회 계산 + mask 합산) → window는 훈련 비용 거의 안 바꾸는 품질 knob. Muon도 chunk 경계 상태에서 gradient를 평가하면 momentum이 linear scan으로 분리되고 NS-5는 batched matmul.", ec=GREY)

# B3.5 결과 + 한계
s = slide(p, "B3 · Atlas — 결과 & 한계", "초장문 외삽은 정점 — 그러나 Muon의 실증 가치는 논쟁적", GREEN)
fig_with_caption(s, "2505.23735", 4, 0.55, 1.5, 6.05, 3.5,
                 "Figure 4 — BABILong: 4K 훈련으로 10M까지 외삽, Titans 붕괴 지점에서도 유지")
pbox(s, 6.85, 1.5, 5.9, 1.35, "결과 (방향)",
     "BABILong: 1M까지 Titans 동급, 10M(Titans 붕괴)에서도 높은 정확도 — 훈련 문맥의 수천 배 외삽. LM·commonsense에서 Atlas/Atlas++가 Titans·GDN·Transformer++ 상회(순수 recurrent 최고 그룹).", ec=GREEN, bsize=10.5)
pbox(s, 6.85, 2.95, 5.9, 2.15, "불편한 진실 (한계)",
     [("① in-context retrieval은 여전히 Transformer 우위 — gap 좁혔으나 못 닫음(용량 이론이 예측한 방향).", {}),
      ("② ablation: window·feature map·deep memory는 명확히 기여하나 Muon 제거 시 perplexity가 오히려 개선. Muon이 산 건 reasoning 0.21점뿐 → 'optimally'는 대부분 window·capacity의 공.", {}),
      ("③ κ 품질 곡선 측정 없음. ④ wall-clock 수치 전무(전부 구조 논증).", {"bold": True, "color": RED})], ec=RED, bsize=9.5)

# B3.6 정리 + 다음
s = slide(p, "B3 · Atlas — 정리 & 다음", "'무엇을 얼마나 잘 기억하는가'의 정점 → 이제 '얼마가 드는가'(TNT)", GREEN)
pbox(s, 0.55, 1.42, 12.23, 1.1, "정리",
     "세 구멍을 세 손잡이로: Omega(목표)·feature map(용량)·Muon(관리). 부산물로 softmax를 무한용량 associative memory로 정식화하고 Transformer 두 일반화 가족(DeepTransformers·Dot)을 파생. objective·capacity·optimizer·이론까지 다 갖춘 정점.", ec=GREEN)
pbox(s, 0.55, 2.65, 12.23, 2.2, "다음으로의 연결 — TNT",
     [("Atlas의 모든 것은 chunkwise 트릭 위에 서 있다 — gradient를 'chunk 시작 상태'라는 stale snapshot에서 평가하는 근사.", {}),
      ("그 근사의 오차는 정량화된 적이 없고, chunk 크기 C는 그냥 throughput knob으로만 취급됐다 (사실 C는 '계산되는 함수 자체'를 바꾸는 semantic knob).", {}),
      ("특히 치명적 mismatch가 방치: 훈련은 큰 C에서, decode는 C=1의 세계. → TNT가 정확히 여기서 시작해 chunk 경제학 전체를 재설계한다.", {"bold": True, "color": GOLD})], ec=GOLD)
pbox(s, 0.55, 5.05, 12.23, 0.85, "조용한 복선",
     "Atlas가 'test-time training'을 'test-time memorization'으로 개명하며 'in-context 적응은 학습이 아니다'라고 선을 그은 순간, '그럼 진짜 continual learning은 어디서?'가 미결로 남는다 → Nested Learning·Sleep.", ec=BLUE)

# ------------------------------ B4. TNT ------------------------------
claim_slide("B4 · TNT (2511.07343)", "TNT — 학습 레시피와 serving 구조를 함께 설계", GREEN,
    [("train chunk와 serve chunk가 다르면 품질이 붕괴한다 (핵심 관찰)",
      "C=64로 학습한 550M Titans를 chunk=1로 서빙하면 성능 급락 — 학습/추론 불일치"),
     ("해법: train chunk(큰)와 serve chunk(작은)를 decouple하고 2-stage로 정렬",
      "큰 chunk로 싸게 병렬 학습 → 2단계로 작은 serve chunk에 맞춰 재정렬"),
     ("hierarchical memory: global(prefill, 병렬) / local(decode, 세밀)",
      "긴 문맥은 global로 한 번에, 생성은 local로 토큰별 — 병렬성과 정밀도를 모두"),
     ("Q-K projection 추가로 메모리 읽기/쓰기 표현력 보강",
      "chunk 정렬과 함께 serving 품질을 끌어올리는 보조 장치")],
    "deep memory를 큰 chunk로 싸게 훈련 → 그런데 serve chunk가 다르면 붕괴(2.6배 악화) → 2-stage로 정렬 → "
    "계층 메모리로 병렬+세밀 → serving 구조 정의",
    "chunk sweep의 V자 곡선(Fig.2), 최대 17배 속도(Fig.4/5), 품질 표로 train/serve 정렬 효과 입증")

# B4.1 문제 — chunk tradeoff + 3 challenges
s = slide(p, "B4 · TNT — 문제", "chunk 크기 하나가 품질과 속도를 동시에 결정한다 (= 훈련 경제학)", GREEN)
pbox(s, 0.55, 1.4, 12.23, 1.15, "chunkwise-parallel training의 트레이드오프",
     [("C개 token을 묶어 'chunk 시작 상태'에서 모든 gradient를 한꺼번에 계산(GPU가 놀지 않게).", {}),
      ("C↑ → matmul 커져 하드웨어 잘 돎, 그러나 gradient가 stale(낡음) → 근사 품질↓.   C↓ → gradient 신선하나 연산이 잘게 쪼개져 하드웨어가 논다 (작은 chunk = peak 대비 5~10% 미만 = MFU 한 자릿수).", {"bold": True, "color": RED})], ec=GREEN)
pbox(s, 0.55, 2.68, 6.0, 1.55, "Challenge 1 — 비선형 recurrence는 병렬화가 안 된다",
     [("deep memory는 fine-grained 위해 작은 chunk를 원하는데 그럼 memory-bound.", {}),
      ("linear attn는 SRAM 상주 chunk kernel로 회피하나, 그 kernel은 '선형 상태 전이'에 의존. MLP+LayerNorm 비선형 recurrence엔 이식 불가(parallel scan 불가).", {})], ec=GOLD, bsize=10.5)
pbox(s, 6.78, 2.68, 6.0, 1.55, "Challenge 2 — write/read 도메인 shift",
     [("write는 k→v로 학습되는데 읽을 땐 query q로 읽는다 → 학습 함수의 입력 domain 밖에서 평가 → retrieval 품질↓.", {}),
      ("attention엔 이 문제가 구조적으로 없다(q·모든 k 내적을 명시적으로). 압축 memory로 넘어올 때만 생기는 세금.", {})], ec=GOLD, bsize=10.5)
pbox(s, 0.55, 4.35, 12.23, 0.6, "Challenge 3 — chunk-size mismatch (다음 슬라이드에서 그림으로)",
     "훈련 chunk와 서빙 chunk가 달라지면 품질이 무너진다.", ec=RED, bsize=11.5)
pbox(s, 0.55, 5.05, 12.23, 0.82, "핵심 가설 (한 문장)",
     "훈련 효율과 inference 성능을 '하나의 chunk 크기'가 동시에 결정하게 놔두지 말고, 두 단계로 분리(decouple)하라. TNT = 'Titans iNside Titans' — memory 안에 memory를 중첩.", ec=BLUE)

# B4.2 Challenge 3 — mismatch V자
s = slide(p, "B4 · TNT — chunk-size mismatch", "새 실증 발견: 비대칭 V자 절벽 (훈련 해상도에 over-specialize)", GREEN)
fig_with_caption(s, "2511.07343", 2, 0.55, 1.5, 6.05, 3.6,
                 "Figure 2 — 550M Titans(C=64로 pre-train), inference chunk만 바꿔 측정")
pbox(s, 6.85, 1.5, 5.9, 1.6, "V자 곡선",
     [("바닥이 정확히 훈련 때 쓴 C=64에 걸림 (ppl 13.78, 최적). 양쪽으로 벗어나면 급격히 악화.", {}),
      ("충격: 더 작은 chunk = 더 신선한 gradient = 더 좋아야 상식적. 그런데 C=8에서 ppl 36.45로 2.6배 폭발.", {"bold": True, "color": RED})], ec=RED, bsize=10.5)
pbox(s, 6.85, 3.2, 5.9, 1.5, "왜 치명적인가",
     [("이상적 서빙 = decode에서 chunk=1(매 token 온라인 갱신).", {}),
      ("큰 chunk로 싸게 훈련한 모델은 chunk=1 근처에서 품질이 무너진다 → 그 이상적 서빙으로 직행할 수가 없다.", {"bold": True, "color": GOLD})], ec=GOLD, bsize=10.5)
pbox(s, 0.55, 5.25, 12.23, 0.95, "읽는 법 (정직성)",
     "이 그림은 TNT 원저자 Figure 2이고 우리 스터디 실측이 아니다. 절대 수치보다 '비대칭 V자 절벽'이라는 방향성을 기억. (Titans가 제기조차 안 했던 train/serve 불일치가 여기서 정면으로 드러난다.)", ec=GREY)

# B4.3 아키텍처 global/local + reset
s = slide(p, "B4 · TNT — global/local 계층 + reset", "reset이 병렬화 불가능한 비선형 사슬을 끊는다", GREEN)
fig_with_caption(s, "2511.07343", 3, 0.55, 1.5, 5.7, 3.55,
                 "Figure 3 — Stage 1: global(위, 큰 chunk 순차) + N개 local(아래, reset·병렬). Q-K는 local에만")
pbox(s, 6.4, 1.5, 6.35, 1.35, "역할 분담",
     [("global 1개: 큰 chunk(2048)로 sequence 전체 순차 관통 → long-range 보존, 드물고 큰 dense matmul = compute-bound (16K에 8번 handoff).", {}),
      ("local N개: 학습된 초기상태에서 주기적 reset → 대량 병렬, fine-grained 담당.", {})], ec=GREEN, bsize=10)
pbox(s, 6.4, 2.95, 6.35, 1.35, "reset이 핵심인 이유",
     [("비선형 recurrence는 parallel scan 불가(결합법칙 없음). reset이 그 사슬을 끊어 각 shard가 독립 → 여러 장치·batch 축으로 병렬.", {}),
      ("대가: local은 shard 경계에서 다 잊음 → global이 보전 (ablation: global 빼면 21→25.6 붕괴).", {"bold": True, "color": RED})], ec=BLUE, bsize=10)
pbox(s, 6.4, 4.4, 6.35, 0.72, "W_init의 승격",
     "reset 지점이 0이 아니라 '학습된' 초기상태 W_init(meta-learn된 prior). Titans에서 암묵적이던 초기상태가 reset을 살아남게 하는 핵심 부품으로.", ec=GOLD, bsize=9.5)
pbox(s, 0.55, 5.25, 12.23, 0.9, "한 줄",
     "reset = sequence 축을 잘라 '진짜 batch 축'으로 되돌리는 연산. 병렬성을 사기 위해 local의 기억을 주기적으로 태우고, 그 보험을 저해상도 global에 든다.", ec=GREY)

# B4.4 Q-K projection + Stage 2
s = slide(p, "B4 · TNT — Q-K projection + 2-stage", "train-big / serve-small: chunk를 두 독립 knob으로 분리", GREEN)
pbox(s, 0.55, 1.4, 12.23, 1.35, "Q-K projection (Challenge 2 처방)",
     [("query를 그대로 memory에 넣지 말고, 지금까지 관측된 key들이 스팬하는 부분공간으로 사영한 뒤 넣는다.", {}),
      ("사영 행렬 Π_t = Σ k·kᵀ 를 매 token rank-1로 누적(과거 key 저장 불필요, 상수 크기). = KV cache append의 rank-1 GEMM 대응물, Π·q는 mat-vec 한 번. domain shift에 민감한 local에만 붙고 global은 raw query.", {})], ec=GREEN)
pbox(s, 0.55, 2.9, 6.0, 1.75, "Stage 2 (Challenge 3 처방)",
     [("Stage1(큰 chunk)로 싸게 pre-train 끝낸 뒤, 작은 local chunk로 짧게 fine-tune.", {}),
      ("놀라운 관찰: 이 짧은 fine-tuning이 mismatch를 교정할 뿐 아니라 원래 성능을 넘어선다. 비용은 pre-training의 5~8%뿐.", {"bold": True, "color": GOLD})], ec=GOLD, bsize=11)
pbox(s, 6.78, 2.9, 6.0, 1.75, "train-big / serve-small 레시피",
     [("이상 목표 chunk=1 → autoregressive 서빙과 정확히 맞물림.", {}),
      ("global이 큰 chunk dense 연산으로 prompt 흡수(prefill), Stage2로 적응된 local이 생성 중 token마다 갱신(decode).", {})], ec=BLUE, bsize=11)
pbox(s, 0.55, 4.8, 12.23, 1.05, "핵심 성취",
     "chunk 크기가 더 이상 '하나의 타협값'이 아니라, Stage1의 throughput knob과 Stage2의 해상도 knob이라는 서로 독립인 두 knob이 된다. '싸게 학습 + 정확한 서빙'이 동시에 가능.", ec=GREEN)

# B4.5 결과 + 한계
s = slide(p, "B4 · TNT — 결과 & 한계", "같은 chunk끼리도 7.7배 → 이득의 원천은 구조 자체", GREEN)
fig_with_caption(s, "2511.07343", 4, 0.55, 1.5, 6.05, 3.5,
                 "Figure 4 — runtime: 순수 JAX인데 32K에서 FlashAttention을 step당 이김")
pbox(s, 6.85, 1.5, 5.9, 1.55, "속도 · 품질",
     [("목표 loss 3.20 도달: Titans(C=8) ~19.5h vs TNT ~1.1h = 17.37배. 같은 chunk 8끼리도 7.7배 → 원천은 chunk 키움이 아니라 구조.", {}),
      ("150M/10B: Stage1만으로 모든 RNN baseline+vanilla Transformer ppl 이김, Stage2가 더 내림. reasoning은 Gated Transformer까지.", {})], ec=GREEN, bsize=10)
pbox(s, 6.85, 3.15, 5.9, 1.05, "ablation",
     "local 1→4개 ppl 단조 개선(20.15). global 빼면 25.6 붕괴. Q-K projection 빼면 ~1 ppl 손해. 역할 분담이 실증됨.", ec=BLUE, bsize=10.5)
pbox(s, 6.85, 4.28, 5.9, 0.85, "정직한 한계",
     "모든 검증이 momentum·gating·Muon을 제거한 '단순화 Titans' 위, 150M/10B(이 라인 최소 규모). 본류 모델과의 합성은 미측정.", ec=RED, bsize=10)
pbox(s, 0.55, 5.15, 6.05, 1.0, "kernel 이전에 구조가 승부를 갈랐다",
     "custom kernel 최적화된 Gated Transformer는 아직 못 이기지만, TNT는 custom kernel 없는 순수 JAX인데도 32K에서 FlashAttention을 step당 이겼다.", ec=GREY, bsize=10.5)

# B4.6 요약 + NL 연결
s = slide(p, "B4 · TNT — 요약 & 다음", "이미 세 개의 update frequency로 도는 시스템 → Nested Learning의 씨앗", GREEN)
pbox(s, 0.55, 1.4, 12.23, 0.85, "TNT가 base 4부작에 남긴 것",
     "이 계열이 대규모로 갈 수 있도록 '훈련 경제학의 바닥'을 깔았다. TNT는 표현력이 아니라 훈련 경제학을 푼 systems 편.", ec=GREEN)
pbox(s, 0.55, 2.4, 12.23, 1.35, "결정적 관념 — 이미 세 개의 update frequency",
     [("global은 2048 token마다, local은 매 token마다, W_init·slow weights는 훈련에서만 갱신된다.", {}),
      ("그리고 느린 시간 스케일에 주차된 지식(global·W_init)이 빠른 스케일의 reset에서 살아남는다.", {"bold": True, "color": GOLD})], ec=GOLD)
pbox(s, 0.55, 3.9, 12.23, 1.75, "다음으로의 연결 — Nested Learning",
     [("다만 TNT에게 이건 그냥 공학적 방편이었다 — '왜 이게 모델 전체의 조직 원리여야 하는지'는 말하지 않았다.", {}),
      ("다음 논문 Nested Learning이 그 선언을 한다: 모델과 훈련 절차 전체가 '각자의 update frequency로 자기 context를 압축하는 중첩 optimization 문제들의 시스템'.", {"bold": True, "color": BLUE}),
      ("TNT의 global/local 이분 → CMS의 주파수 연속체로, TNT의 reset → NL의 re-initialization으로, 하중을 받게 된 W_init → 다섯 knowledge-transfer 기제 중 하나로 분류된다.", {})], ec=BLUE)
pbox(s, 0.55, 5.85, 12.23, 0.5, "여기까지가 base 4부작", "이제 집중해서 볼 두 논문(HOPE·Sleep)으로 들어간다.", ec=GREY, bsize=11)

# ------------------------------ B5. Nested Learning / HOPE ------------------------------
claim_slide("B5 · Nested Learning / HOPE (2512.24695)", "HOPE — 모든 부품이 '어떤 주파수의 메모리'인 아키텍처", GOLD,
    [("Expressive Optimizers: optimizer는 gradient를 압축하는 associative memory다",
      "momentum·Adam·Muon = 메모리의 특수형 → 더 표현력 큰 optimizer(DGD·M3)로 승격"),
     ("Self-Modifying Titans: 투영·게이트까지 메모리로 만들어 자기 갱신 규칙을 스스로 수정",
      "6개 메모리 {k,v,q,η,α,mem} 중 q만 static, 나머지는 test-time에 갱신"),
     ("CMS(Continuum Memory System): long/short를 '연속 주파수' 스펙트럼으로 일반화",
      "여러 주파수의 MLP 블록을 이어 붙임 — TNT의 global/local을 연속체로 확장. 합 = HOPE")],
    "backprop도 self-referential memory → optimizer도 memory → 그럼 architecture와 optimizer는 같은 것의 "
    "다른 레벨 → 레벨을 더 쌓자(higher-order ICL) → self-mod + CMS = HOPE. ICL은 창발이 아니라 ≥2레벨의 structural 결과",
    "760M/1.3B 벤치(Transformer++·RWKV7·DeltaNet·Titans 대비), BABILong, 메모리 레벨 ablation, M3 optimizer(ViT)")

# B5.1 Nested Learning paradigm + expressive optimizers
s = slide(p, "B5 · Nested Learning — 관점", "학습 자체를 '중첩된 최적화(=메모리)의 층'으로 본다", GOLD)
fig_with_caption(s, "2512.24695", 2, 0.55, 1.5, 6.0, 3.5,
                 "Nested Learning: 모델과 그 학습절차를 중첩된 최적화 문제들의 집합으로", ec=GOLD)
box(s, 6.75, 1.5, 6.0, 1.6,
    [("핵심 재해석", {"size": 13, "bold": True, "color": GOLD, "align": PP_ALIGN.LEFT, "space_after": 4}),
     ("optimizer가 하는 일(과거 gradient를 모아 다음 스텝을 정함) = '연상 메모리'다. "
      "그럼 momentum은 선형 메모리, Adam·Muon도 메모리의 특수형.", {"size": 12, "color": INK, "align": PP_ALIGN.LEFT})],
    fc=GOLDB, ec=GOLD, align=PP_ALIGN.LEFT)
fit_image(s, mp.eq(r"\text{optimizer}\ \equiv\ \text{associative memory over gradients}", pt=14)[0], 6.95, 3.2, 5.6, 0.5)
box(s, 6.75, 3.85, 6.0, 1.15,
    [("그래서 무엇이 열리나", {"size": 13, "bold": True, "color": GOLD, "align": PP_ALIGN.LEFT, "space_after": 4}),
     ("optimizer를 '더 표현력 있는 메모리'(Deep GD·delta-momentum·M3)로 바꿀 수 있고, 이 최적화 층을 "
      "여러 개 쌓으면 higher-order in-context learning이 된다.", {"size": 11.5, "color": INK, "align": PP_ALIGN.LEFT})],
    fc=GOLDB, ec=GOLD, align=PP_ALIGN.LEFT)
box(s, 0.55, 5.2, 12.2, 1.0,
    [("ICL에 대한 주장", {"size": 12.5, "bold": True, "color": BLUE, "align": PP_ALIGN.LEFT, "space_after": 2}),
     ("in-context learning은 큰 모델에서 '창발'하는 마법이 아니라, 최소 2개 레벨의 중첩 최적화를 가지면 "
      "구조적으로(structural) 따라 나오는 성질이다 — Fig.7의 레벨 ablation이 이를 뒷받침.",
      {"size": 12, "color": INK, "align": PP_ALIGN.LEFT})], fc=BLUEB, ec=BLUE, align=PP_ALIGN.LEFT)

# B5.2 Self-Modifying Titans
s = slide(p, "B5 · Self-Modifying Titans", "투영·게이트까지 메모리 — 자기 갱신 규칙을 스스로 수정", GOLD)
mems = [("M_k", "key 투영", GOLDB), ("M_v", "value 투영", GOLDB), ("M_q", "query 투영 (static)", GREYB),
        ("M_η", "momentum 게이트", GOLDB), ("M_α", "forget 게이트", GOLDB), ("M_mem", "본체 메모리", GOLDB)]
mx = 0.55
for name, role, fc in mems:
    ec = GREY if fc is GREYB else GOLD
    box(s, mx, 1.6, 1.95, 1.05,
        [(name, {"size": 15, "bold": True, "color": ec, "space_after": 3}), (role, {"size": 10.5, "color": INK})],
        fc=fc, ec=ec, lw=1.4)
    mx += 2.05
text(s, 0.55, 2.75, 12.2, 0.3, "↑ 6개 메모리 중 q만 static(고정), 나머지 5개는 추론 중 갱신됨", size=12, color=MUTE, italic=True, space_after=0)
box(s, 0.55, 3.25, 6.0, 1.7,
    [("무엇이 새로운가", {"size": 13, "bold": True, "color": GOLD, "align": PP_ALIGN.LEFT, "space_after": 4}),
     ("Titans는 본체 메모리 M_mem만 갱신했다. HOPE는 '무엇을 key/value로 삼을지(투영)'와 "
      "'얼마나 기억·잊을지(게이트)'까지 메모리로 만들어 test-time에 함께 갱신한다.",
      {"size": 12, "color": INK, "align": PP_ALIGN.LEFT})], fc=GOLDB, ec=GOLD, align=PP_ALIGN.LEFT)
box(s, 6.75, 3.25, 6.0, 1.7,
    [("어떻게 갱신하나", {"size": 13, "bold": True, "color": GOLD, "align": PP_ALIGN.LEFT, "space_after": 4}),
     ("target을 스스로 만든다(self-target: v̂_□ = M_□(v_t)). 갱신은 DGD-with-weight-decay, "
      "gradient는 full backprop. 출력은 갱신 전(chunk 시작) 메모리로 읽는다.",
      {"size": 12, "color": INK, "align": PP_ALIGN.LEFT})], fc=GOLDB, ec=GOLD, align=PP_ALIGN.LEFT)
box(s, 0.55, 5.1, 12.2, 1.1,
    [("의미", {"size": 12.5, "bold": True, "color": BLUE, "align": PP_ALIGN.LEFT, "space_after": 2}),
     ("Titans의 '고정된 투영'을 풀어, 모델이 자신의 학습 규칙(무엇을 어떻게 기억할지)을 데이터에 맞춰 "
      "바꾸게 만든 것. q를 고정으로 남긴 건 안정적 읽기 기준을 유지하기 위함(추론).",
      {"size": 12, "color": INK, "align": PP_ALIGN.LEFT})], fc=BLUEB, ec=BLUE, align=PP_ALIGN.LEFT)

# B5.3 CMS
s = slide(p, "B5 · CMS — Continuum Memory System", "long/short를 '연속 주파수'의 메모리 블록들로", GOLD)
freqs = [("f₁ (느림·저주파)", "큰 chunk마다 1스텝", GREEN), ("f₂ (중간)", "중간 chunk마다", GOLD), ("f₃ (빠름·고주파)", "작은 chunk마다", RED)]
fx = 0.9
for i, (name, role, c) in enumerate(freqs):
    box(s, fx, 1.7, 3.0, 1.0, [(name, {"size": 13, "bold": True, "color": c, "space_after": 3}),
        (role, {"size": 11, "color": INK})], fc=WHITE, ec=c, lw=1.5)
    if i < 2:
        arrow(s, fx + 3.0, 2.2, fx + 3.9, 2.2, color=INK, lw=1.6)
    fx += 3.9
text(s, 0.9, 2.8, 11.5, 0.3, "forward: y = MLP^(f₃)( … MLP^(f₁)(x) )  · 각 블록은 자기 주파수 경계에서 gradient를 모아 1스텝 갱신",
     size=11.5, color=MUTE, italic=True, space_after=0)
box(s, 0.55, 3.35, 6.0, 1.6,
    [("TNT와의 관계", {"size": 13, "bold": True, "color": GOLD, "align": PP_ALIGN.LEFT, "space_after": 4}),
     ("TNT의 global/local(2단계)을 임의 개수의 주파수로 일반화. 느린 블록=장기 문맥, 빠른 블록=국소 세부. "
      "블록 간 전달은 '가중치 복사'가 아니라 forward 활성으로.", {"size": 12, "color": INK, "align": PP_ALIGN.LEFT})],
    fc=GOLDB, ec=GOLD, align=PP_ALIGN.LEFT)
box(s, 6.75, 3.35, 6.0, 1.6,
    [("meta-learned 초기상태", {"size": 13, "bold": True, "color": GOLD, "align": PP_ALIGN.LEFT, "space_after": 4}),
     ("각 블록의 '리셋 지점'(초기 상태)은 사전학습으로 meta-learn. 느린 블록이 빠른 블록의 리셋 초기값을 제공. "
      "이건 런타임 diff 전달이 아니라 시작점 설정(→ B6 Sleep가 그 diff 전달을 맡음).",
      {"size": 11.5, "color": INK, "align": PP_ALIGN.LEFT})], fc=GOLDB, ec=GOLD, align=PP_ALIGN.LEFT)
box(s, 0.55, 5.1, 12.2, 1.1,
    [("HOPE = self-modifying Titans + CMS", {"size": 12.5, "bold": True, "color": BLUE, "align": PP_ALIGN.LEFT, "space_after": 2}),
     ("자기수정(어떻게 기억할지) + 연속 주파수(언제 기억할지)를 합치면, '모든 부품이 어떤 주파수의 메모리'인 "
      "하나의 중첩 시스템 = HOPE가 된다.", {"size": 12, "color": INK, "align": PP_ALIGN.LEFT})],
    fc=BLUEB, ec=BLUE, align=PP_ALIGN.LEFT)

# B5.4 WRAP-UP — 앞 4편을 하나로
s = slide(p, "B5 · wrap-up", "HOPE의 렌즈로 앞 4편을 한 문장씩 다시 읽기", GOLD)
wraps = [
    ("Titans", "메모리 = gradient로 학습되는 신경망(surprise+momentum+forget).",
     "HOPE: 그 optimizer(momentum 등)마저 메모리 → 아키텍처와 optimizer가 같은 것의 두 레벨.", GREEN),
    ("Miras", "4축(attentional bias·retention·architecture·algorithm)으로 설계공간을 정의.",
     "HOPE: 'algorithm' 축(momentum/Adam/Muon)이 사실 메모리 → architecture·algorithm 축을 하나로 통합.", GREEN),
    ("Atlas", "feature map으로 용량 확장 + Muon(2차) optimizer로 메모리 학습.",
     "HOPE: Muon류 = gradient의 '표현력 큰 메모리' → M3(더 표현력 있는 optimizer)로 일반화.", GREEN),
    ("TNT", "global/local(장기/단기) 2단계 메모리 + train/serve chunk 정렬.",
     "HOPE: long/short를 CMS의 '연속 주파수' 스펙트럼으로 확장 — 2단계가 아니라 연속체.", GREEN),
]
wy = 1.55
for name, before, after, c in wraps:
    box(s, 0.55, wy, 1.7, 1.15, name, fc=c, ec=c, size=14, tcolor=WHITE, bold=True)
    box(s, 2.35, wy, 10.4, 1.15,
        [(f"원래: {before}", {"size": 12, "color": INK, "align": PP_ALIGN.LEFT, "space_after": 3}),
         (f"HOPE: {after}", {"size": 12, "bold": True, "color": GOLD, "align": PP_ALIGN.LEFT})],
        fc=WHITE, ec=GREY, align=PP_ALIGN.LEFT, lw=1.0)
    wy += 1.25
box(s, 0.55, 6.6, 12.2, 0.55,
    [("한 문장: HOPE는 '모든 부품(투영·게이트·optimizer·장단기 메모리)이 어떤 주파수의 메모리'라는 하나의 중첩 시스템으로 앞 4편을 통합한다.",
      {"size": 12.5, "bold": True, "color": BLUE, "align": PP_ALIGN.LEFT})], fc=BLUEB, ec=BLUE, align=PP_ALIGN.LEFT)

# B5.5 proof
s = slide(p, "B5 · HOPE — 증명", "레벨이 ICL을, 표현력 큰 optimizer가 학습을 개선한다", GOLD)
fig_with_caption(s, "2512.24695", 7, 0.4, 1.5, 4.15, 3.5,
                 "메모리 레벨↑ → ICL(NIAH) 성능↑ (structural)", ec=GOLD)
fig_with_caption(s, "2512.24695", 9, 4.62, 1.5, 4.15, 3.5,
                 "BABILong: HOPE가 대형 모델과 견줌", ec=GOLD)
fig_with_caption(s, "2512.24695", 11, 8.84, 1.5, 4.15, 3.5,
                 "M3 optimizer: ViT에서 AdamW·Muon 대비 개선", ec=GOLD)
box(s, 0.55, 5.2, 12.2, 1.0,
    [("읽는 법", {"size": 12.5, "bold": True, "color": INK, "align": PP_ALIGN.LEFT, "space_after": 2}),
     ("왼쪽=중첩 레벨을 늘릴수록 in-context 성능이 오른다(ICL=structural 주장의 증거). 가운데=초장문 추론에서 경쟁력. "
      "오른쪽=optimizer를 메모리로 보는 관점이 실제 학습(ViT)에서도 이득.", {"size": 12, "color": INK, "align": PP_ALIGN.LEFT})],
    fc=GREYB, ec=GREY, align=PP_ALIGN.LEFT)

# ------------------------------ B6. Sleep ------------------------------
claim_slide("B6 · Sleep (2606.03979)", "Sleep — 자는 동안 지식을 '위로' 옮겨 파라미터를 키운다", GREEN,
    [("wake/sleep lifecycle: 추론(wake)과 오프라인 통합(sleep)을 분리",
      "sleep은 별도 오프라인 단계 — 이때는 서빙하지 않고 그동안 배운 것을 정리·통합한다"),
     ("Memory Consolidation = Knowledge Seeding: 배운 지식을 상위로 distill해 low-rank expert를 '성장'",
      "NL의 online consolidation은 초기상태(reset)만 meta-learn — '학습된 diff의 위쪽 전달'은 못 했다. Sleep가 그것을 함"),
     ("Dreaming: 스스로 만든 데이터로 자기개선(SEAL식 RL)",
      "teacher가 '꿈'(자기 생성 시퀀스)을 만들고 RL로 강화 — 외부 데이터 없이 능력 향상")],
    "NL online consolidation(초기상태 meta-learn + circle-back)은 고정 용량 한계 → 오프라인으로 파라미터 성장 → "
    "GKD(on-policy)+LTI(RL) → 새 low-rank expert만 학습(나머지 freeze) → synaptic pruning → Dreaming 자기개선",
    "class-incremental(CLINC/Banking/DBpedia) 지속학습, 메모리 레벨 효과, BABILong 등으로 통합 효과 입증")

# B6.1 lifecycle + consolidation
s = slide(p, "B6 · Sleep — 통합 메커니즘", "느린 블록에 새 저차원 expert를 키워 지식을 위로 옮긴다", GREEN)
fig_with_caption(s, "2606.03979", 2, 0.55, 1.5, 6.0, 3.5,
                 "Memory Consolidation: 모델이 스스로 파라미터 수를 늘려 용량을 확장")
fig_with_caption(s, "2606.03979", 8, 6.85, 1.5, 5.9, 3.5,
                 "Sleep cycle마다 router가 새 expert를 선택·갱신(왼→오른쪽)")
box(s, 0.55, 5.15, 12.2, 1.05,
    [("B5(HOPE)와의 연결 — 빠진 조각", {"size": 12.5, "bold": True, "color": BLUE, "align": PP_ALIGN.LEFT, "space_after": 2}),
     ("HOPE의 CMS는 '느린 블록이 빠른 블록의 리셋 초기값'을 줄 뿐(런타임 diff 전달 아님). Sleep는 오프라인에서 "
      "빠른(고주파) 메모리가 배운 것을 새 low-rank expert로 만들어 느린 블록에 '실제로' 얹는다 = 위쪽 전달.",
      {"size": 12, "color": INK, "align": PP_ALIGN.LEFT})], fc=BLUEB, ec=BLUE, align=PP_ALIGN.LEFT)

# B6.2 offline data generation (self-generated)
s = slide(p, "B6 · Sleep — 데이터는 어떻게 만드나", "전부 오프라인·자기생성: teacher→student, GKD + RL", GREEN)
# flow: teacher -> (samples) -> student(new expert) ; GKD + LTI
box(s, 0.55, 1.6, 2.7, 1.2, [("Teacher", {"size": 14, "bold": True, "color": GREEN, "space_after": 3}),
    ("현재 모델 LM_θ", {"size": 11, "color": INK}), ("샘플·'꿈' 생성", {"size": 11, "color": MUTE})], fc=GREENB, ec=GREEN)
arrow(s, 3.3, 2.2, 4.25, 2.2, color=INK, lw=1.8)
box(s, 4.35, 1.6, 2.7, 1.2, [("Student", {"size": 14, "bold": True, "color": GOLD, "space_after": 3}),
    ("새 low-rank expert", {"size": 11, "color": INK}), ("{A: d×r, B: r×d}", {"size": 11, "mono": True, "color": MUTE})], fc=GOLDB, ec=GOLD)
arrow(s, 7.1, 2.2, 8.05, 2.2, color=INK, lw=1.8)
box(s, 8.15, 1.6, 4.6, 1.2, [("학습 신호", {"size": 14, "bold": True, "color": RED, "space_after": 3}),
    ("GKD(on-policy rollout) + RL(LTI)", {"size": 11, "color": INK}), ("보상=Levenshtein·의미 유사도", {"size": 11, "color": MUTE})], fc=REDB, ec=RED)
rows = [
    ("Knowledge Seeding (통합)", "teacher가 만든 on-policy 데이터로 student를 GKD. 새 expert만 backward(나머지 freeze). router가 새 expert를 활성화."),
    ("LTI — RL 파트", "정답 시퀀스와의 Levenshtein/의미 보상으로 policy-gradient. 미분 불가한 목표를 RL로."),
    ("Dreaming (자기개선)", "teacher가 자기 생성한 '꿈'으로 SEAL식 편집·강화 — 외부 라벨 없이 능력을 끌어올림."),
    ("reset = (de)activation", "새 주기 시작 시 빠른 블록의 이전 expert들을 비활성화(pruning)해 간섭을 막는다."),
]
ry = 3.15
for t, b in rows:
    box(s, 0.55, ry, 12.2, 0.82, [(t + "  ", {"size": 12.5, "bold": True, "color": GREEN, "align": PP_ALIGN.LEFT}),
        (b, {"size": 11.5, "color": INK, "align": PP_ALIGN.LEFT})], fc=WHITE, ec=GREY, align=PP_ALIGN.LEFT, lw=1.0, anchor=MSO_ANCHOR.MIDDLE)
    ry += 0.9
text(s, 0.55, 6.85, 12.2, 0.3, "핵심: 학습 데이터가 전부 모델 자신에게서 나온다(self-generated) — 외부 코퍼스 추가 없이 지속 학습.",
     size=11.5, color=MUTE, italic=True, space_after=0)

# B6.3 proof
s = slide(p, "B6 · Sleep — 증명", "지속 학습에서 망각을 줄이고 성능을 유지한다", GREEN)
fig_with_caption(s, "2606.03979", 3, 0.55, 1.5, 6.05, 3.6,
                 "class-incremental(CLINC 등): sleep 통합이 이전 클래스 망각을 완화")
fig_with_caption(s, "2606.03979", 6, 6.85, 1.5, 5.9, 3.6,
                 "BABILong: 오프라인 통합 후에도 장문 추론 능력 유지·향상")
box(s, 0.55, 5.25, 12.2, 0.95,
    [("읽는 법 · 6편의 도착점", {"size": 12.5, "bold": True, "color": INK, "align": PP_ALIGN.LEFT, "space_after": 2}),
     ("Titans가 연 'test-time 학습'은 → Sleep에서 'wake=빠른 학습 / sleep=느린 통합'의 완결된 생애주기가 된다. "
      "고정 용량의 한계를 파라미터 성장으로 넘은 것이 이 논문의 마지막 퍼즐.", {"size": 12, "color": INK, "align": PP_ALIGN.LEFT})],
    fc=GREYB, ec=GREY, align=PP_ALIGN.LEFT)

# ==================================================================================
# ============================== PART C — System modeling ==============================
# ==================================================================================
section_divider(p, "PART C", "System modeling (우리 기여) — 이 배포는 서빙에서 얼마나 드나", GOLD)

# C1 pair thesis
s = slide(p, "C1 · pair thesis", "학습은 compute-bound, 디코드는 memory-bound", GOLD)
box(s, 0.55, 1.5, 6.0, 2.0,
    [("학습(training) — compute-bound", {"size": 14, "bold": True, "color": BLUE, "align": PP_ALIGN.LEFT, "space_after": 5}),
     ("chunk C를 크게 잡아 병렬 → 연산/바이트(arithmetic intensity)가 높다.", {"size": 12.5, "color": INK, "align": PP_ALIGN.LEFT}),
     ("roofline의 오른쪽(연산 한계) → 더 빠른 연산기가 이득.", {"size": 12.5, "color": INK, "align": PP_ALIGN.LEFT})],
    fc=BLUEB, ec=BLUE, align=PP_ALIGN.LEFT)
box(s, 6.75, 1.5, 6.0, 2.0,
    [("디코드(decode) — memory-bound", {"size": 14, "bold": True, "color": RED, "align": PP_ALIGN.LEFT, "space_after": 5}),
     ("chunk=1, 토큰마다 '전체 상태(state+weights)'를 read-modify-write.", {"size": 12.5, "color": INK, "align": PP_ALIGN.LEFT}),
     ("arithmetic intensity가 낮다 → roofline의 왼쪽(대역폭 한계).", {"size": 12.5, "color": INK, "align": PP_ALIGN.LEFT})],
    fc=REDB, ec=RED, align=PP_ALIGN.LEFT)
our_fig(s, "fig4-sstar-invariance.png", 0.55, 3.75, 6.05, 2.55,
        "S*(KV↔TTT 교차점)=65536 tokens — 8개 가속기 전부 동일 = 하드웨어가 아니라 workload의 성질")
box(s, 6.75, 3.75, 6.0, 2.3,
    [("왜 중요한가", {"size": 13, "bold": True, "color": GOLD, "align": PP_ALIGN.LEFT, "space_after": 5}),
     ("메모리 상태를 갱신하는 이 계열은 디코드에서 구조적으로 memory-bound다. 그리고 그 경계(S*)는 "
      "하드웨어를 바꿔도 변하지 않는다(invariant). 즉 '어느 칩을 써도' 디코드 비용은 메모리 대역폭이 지배한다.",
      {"size": 12.5, "color": INK, "align": PP_ALIGN.LEFT}),
     ("→ 이후 슬라이드: 실제 8칩 측정 · HOPE block roofline · 메모리 기술(zHBM/PIM/SRAM) · 청사진.",
      {"size": 12, "bold": True, "color": BLUE, "align": PP_ALIGN.LEFT})],
    fc=GOLDB, ec=GOLD, align=PP_ALIGN.LEFT)

# C2 cross-arch
s = slide(p, "C2 · 8개 가속기 크로스 측정", "5자릿수 스프레드, 그러나 대부분 memory-bound", GOLD)
our_fig(s, "fig1-decode-ms.png", 0.55, 1.5, 6.05, 3.6,
        "토큰당 decode 시간(ideal 하한, log) — MTIA v2 ~31ms에서 WSE-3 ~0.0003ms까지")
our_fig(s, "fig4-sstar-invariance.png", 6.85, 1.5, 5.9, 3.6,
        "S* 불변: 8개 아키텍처가 같은 교차점 — workload 고유 성질")
box(s, 0.55, 5.25, 12.2, 1.0,
    [("읽는 법", {"size": 12.5, "bold": True, "color": INK, "align": PP_ALIGN.LEFT, "space_after": 2}),
     ("8칩 중 7개가 memory-bound(대역폭이 결정). 예외는 WSE-3(Cerebras) — 거대한 on-chip SRAM으로 상태를 상주시켜 "
      "compute 쪽으로 넘어가는 'knee'가 생긴다. 각 칩 spec은 HAT hw twin으로 검증(die/PE 구조까지).",
      {"size": 12, "color": INK, "align": PP_ALIGN.LEFT})], fc=GREYB, ec=GREY, align=PP_ALIGN.LEFT)

# C3 HOPE-block roofline
s = slide(p, "C3 · HOPE block roofline (H100 트윈)", "1개 HOPE block을 실제 수치로 — 어디가 병목인가", GOLD)
our_fig(s, "fig-hope-dag.png", 0.55, 1.5, 6.05, 3.5,
        "1 HOPE block(self-mod Titans→CMS)의 전 연산 DAG (B=1, chunk C, d=2048)")
our_fig(s, "fig-hope-roofline.png", 6.85, 1.5, 5.9, 3.5,
        "roofline·op별 시간·PIM 비교 (BW 3.35TB/s, peak 0.99PF, ridge 295)")
box(s, 0.55, 5.15, 12.2, 1.05,
    [("병목", {"size": 12.5, "bold": True, "color": RED, "align": PP_ALIGN.LEFT, "space_after": 2}),
     ("C=1 디코드는 roofline 왼쪽(memory-bound). op별로 보면 CMS/DGD-apply의 state read-modify-write가 지배적. "
      "state(≈134MB) + weights를 매 토큰 HBM(3.35TB/s)에서 읽어야 해 block당 ~203µs(ideal 하한).",
      {"size": 12, "color": INK, "align": PP_ALIGN.LEFT})], fc=REDB, ec=RED, align=PP_ALIGN.LEFT)

# C4 memory tech
s = slide(p, "C4 · 메모리 기술 — zHBM · HBM-PIM · SRAM", "memory-bound라면 답은 '대역폭'과 '상주'", GOLD)
our_fig(s, "fig-zhbm-pim.png", 0.55, 1.5, 6.05, 3.6,
        "HBM/zHBM/HBM-PIM/zHBM-PIM decode 비용 + zHBM 내부 연산 추가 면적")
our_fig(s, "fig-sram-scaling.png", 6.85, 1.5, 5.9, 3.6,
        "on-chip SRAM 1x→10x: state가 상주하는 순간(5x~) HBM 대신 SRAM")
box(s, 0.55, 5.25, 12.2, 1.0,
    [("결론", {"size": 12.5, "bold": True, "color": GOLD, "align": PP_ALIGN.LEFT, "space_after": 2}),
     ("zHBM(대역폭↑)이 HBM-PIM(in-bank 연산이 너무 느림)보다 유리. SRAM은 '문턱' 구조 — state(≈134MB)가 "
      "on-chip에 들어가는 5x(250MB)부터 이득(203→150µs), 10x(500MB)면 weights까지 전부 상주.",
      {"size": 12, "color": INK, "align": PP_ALIGN.LEFT})], fc=GOLDB, ec=GOLD, align=PP_ALIGN.LEFT)

# C5 blueprint
s = slide(p, "C5 · scaling 청사진", "논문의 방향 × 모델링 결과 = 어디로 가야 하나", GOLD)
our_fig(s, "fig8-hope-sleep-scaling.png", 0.55, 1.5, 6.05, 3.6,
        "HOPE(메모리 확장)+Sleep(파라미터 성장) 스케일링 — 상태·파라미터가 함께 자란다")
bl = [
    ("① 논문 방향이 memory 압력을 키운다", "HOPE(self-mod+CMS)로 state↑, Sleep(expert 성장)로 파라미터↑ → 디코드 memory-bound가 구조적으로 심해진다."),
    ("② 대역폭 우선", "zHBM류 고대역폭 메모리가 1순위 지렛대(PIM보다 유리)."),
    ("③ state 상주 설계", "on-chip 용량을 state가 들어갈 문턱 이상으로 — Cerebras의 knee가 그 증거."),
    ("④ 소프트웨어 효율이 프론티어", "cross-vendor 소프트웨어 효율(HATIR)이 남은 최대 변수."),
]
by = 1.55
for t, b in bl:
    box(s, 6.75, by, 6.0, 1.15, [(t, {"size": 12.5, "bold": True, "color": GOLD, "align": PP_ALIGN.LEFT, "space_after": 3}),
        (b, {"size": 11.5, "color": INK, "align": PP_ALIGN.LEFT})], fc=GOLDB, ec=GOLD, align=PP_ALIGN.LEFT)
    by += 1.22
box(s, 0.55, 5.25, 6.05, 1.05,
    [("한 줄", {"size": 12.5, "bold": True, "color": BLUE, "align": PP_ALIGN.LEFT, "space_after": 2}),
     ("이 계열을 키울수록 서빙은 '메모리 문제'가 된다 — 대역폭·상주·소프트웨어가 스케일의 열쇠.",
      {"size": 12, "color": INK, "align": PP_ALIGN.LEFT})], fc=BLUEB, ec=BLUE, align=PP_ALIGN.LEFT)

# ============================== 마무리 ==============================
s = slide(p, "마무리", "6편을 한 흐름으로, 그리고 서빙 비용까지", GREEN)
closing = [
    ("Titans", "test-time에 학습되는 신경망 메모리(surprise+momentum+forget)를 연다.", GREEN),
    ("Miras", "그 설계를 4축 공간으로 조직화 — 무엇을·무엇을 잊고·어떤 구조로·어떤 알고리즘으로.", GREEN),
    ("Atlas", "문맥 단위 기억(Omega)과 용량 확장(feature map)+Muon으로 표현력을 키운다.", GREEN),
    ("TNT", "train/serve chunk 정렬과 global/local 계층으로 '어떻게 배포하나'를 답한다.", GREEN),
    ("HOPE", "optimizer=memory·self-mod·CMS로 앞 4편을 하나의 중첩 시스템으로 통합.", GOLD),
    ("Sleep", "wake/sleep 생애주기로 지식을 오프라인에서 위로 옮겨 파라미터를 키운다.", GREEN),
    ("우리 기여", "이 배포는 디코드에서 memory-bound — 8칩 측정+HOPE roofline+zHBM/SRAM로 비용과 청사진을 냈다.", GOLD),
]
cy = 1.5
for name, desc, c in closing:
    box(s, 0.55, cy, 1.7, 0.68, name, fc=c, ec=c, size=12.5, tcolor=WHITE, bold=True)
    box(s, 2.35, cy, 10.4, 0.68, desc, fc=WHITE, ec=GREY, size=12, tcolor=INK, align=PP_ALIGN.LEFT, lw=1.0)
    cy += 0.76
text(s, 0.55, 6.95, 12, 0.3, "감사합니다 · Q&A", size=13, color=INK, bold=True, align=PP_ALIGN.CENTER, space_after=0)

# save
out = os.path.join(HERE, "TTT-seminar.pptx")
p.save(out)
print(f"saved {out} ({len(p.slides.__iter__.__self__._sldIdLst)} slides)")
