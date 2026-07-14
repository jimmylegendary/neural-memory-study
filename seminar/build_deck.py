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

s = slide(p, "B2 · Miras — 4축 설계공간", "메모리를 4개의 독립적 '선택'으로 분해", GREEN)
fig_with_caption(s, "2504.13173", 1, 0.55, 1.5, 5.6, 4.2,
                 "Miras 프레임워크: 4가지 핵심 선택으로 sequence layer를 구성")
axes = [
    ("① Attentional bias", "무엇을 기억? = 손실 함수. Titans의 ‖M(k)−v‖²는 한 예. Lp·robust로 바꿀 수 있다."),
    ("② Retention gate", "무엇을 잊나? = 정규화/감쇠. forget gate가 여기 속한다."),
    ("③ Memory architecture", "메모리의 구조: vector · matrix · deep MLP."),
    ("④ Learning algorithm", "어떻게 갱신? GD · momentum · FTRL · Learning-Retaining."),
]
ay = 1.55
for t, b in axes:
    box(s, 6.5, ay, 6.25, 0.98, [(t, {"size": 13, "bold": True, "color": GREEN, "align": PP_ALIGN.LEFT, "space_after": 3}),
        (b, {"size": 11.5, "color": INK, "align": PP_ALIGN.LEFT})], fc=GREENB, ec=GREEN, align=PP_ALIGN.LEFT)
    ay += 1.08
box(s, 0.55, 5.95, 12.2, 0.85,
    [("의의", {"size": 12.5, "bold": True, "color": BLUE, "align": PP_ALIGN.LEFT, "space_after": 2}),
     ("Titans를 특수점으로 품는 지도를 그려, '무엇을 바꾸면 무엇이 좋아지나'를 조직적으로 탐색 가능하게 했다. 이후 Atlas·HOPE는 이 축들을 각자 밀어붙인 결과다.",
      {"size": 12, "color": INK, "align": PP_ALIGN.LEFT})], fc=BLUEB, ec=BLUE, align=PP_ALIGN.LEFT)

s = slide(p, "B2 · Miras — 증명", "축을 움직이면 성능이 예측대로 변한다", GREEN)
fig_with_caption(s, "2504.13173", 3, 0.55, 1.5, 6.05, 3.6,
                 "모델 크기·문맥 길이에 따른 스케일링 — Miras 변형들이 안정적으로 개선")
fig_with_caption(s, "2504.13173", 4, 6.85, 1.5, 5.9, 3.6,
                 "손실 차수 p·retention q의 효과 — 축 선택이 문맥 길이별 성능을 가른다")
box(s, 0.55, 5.25, 12.2, 0.95,
    [("읽는 법", {"size": 12.5, "bold": True, "color": INK, "align": PP_ALIGN.LEFT, "space_after": 2}),
     ("각 축을 독립적으로 흔들었을 때 성능이 단조·예측 가능하게 반응 → '설계 공간'이 실재함을 뒷받침. "
      "이 관점이 다음 Atlas(용량 축)와 HOPE(알고리즘 축)의 출발점.", {"size": 12, "color": INK, "align": PP_ALIGN.LEFT})],
    fc=GREYB, ec=GREY, align=PP_ALIGN.LEFT)

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

s = slide(p, "B3 · Atlas — 메커니즘", "문맥 단위 기억(Omega) + 용량 확장(feature map)", GREEN)
fig_with_caption(s, "2505.23735", 2, 0.55, 1.5, 6.0, 3.5,
                 "SWA vs Atlas/Omega: 문맥 내 토큰 의존성을 더 넓게 연결")
box(s, 6.75, 1.5, 6.0, 1.35,
    [("Omega rule (문맥 단위 목적)", {"size": 12.5, "bold": True, "color": GREEN, "align": PP_ALIGN.LEFT, "space_after": 4}),
     ("최근 c개 토큰을 γ 가중으로 한 번에 적합:", {"size": 11.5, "color": INK, "align": PP_ALIGN.LEFT})],
    fc=GREENB, ec=GREEN, align=PP_ALIGN.LEFT)
fit_image(s, mp.eq(r"\min_{M}\ \sum_{i=t-c+1}^{t} \gamma_{t,i}\,\big\|M(k_i)-v_i\big\|^2", pt=15)[0], 6.95, 2.85, 5.6, 0.75)
box(s, 6.75, 3.75, 6.0, 1.25,
    [("용량 확장 + Muon", {"size": 12.5, "bold": True, "color": GREEN, "align": PP_ALIGN.LEFT, "space_after": 4}),
     ("φ_p(k)로 차원을 d→d^p로 올려 저장 용량↑. 학습은 Muon(근사 2차)로 곡률을 활용.",
      {"size": 11.5, "color": INK, "align": PP_ALIGN.LEFT})], fc=GREENB, ec=GREEN, align=PP_ALIGN.LEFT)
box(s, 0.55, 5.15, 12.2, 1.05,
    [("한 줄 요지", {"size": 12.5, "bold": True, "color": BLUE, "align": PP_ALIGN.LEFT, "space_after": 2}),
     ("Titans가 '무엇을 기억하나(연상)'를 열었다면, Atlas는 '얼마나 많이·문맥 단위로 기억하나(용량)'를 키운다. "
      "Miras의 architecture·algorithm 축을 밀어붙인 결과.", {"size": 12, "color": INK, "align": PP_ALIGN.LEFT})],
    fc=BLUEB, ec=BLUE, align=PP_ALIGN.LEFT)

s = slide(p, "B3 · Atlas — 증명", "Titans를 넘어서고, 용량이 실제로 커진다", GREEN)
fig_with_caption(s, "2505.23735", 4, 0.55, 1.5, 6.05, 3.6,
                 "BABILong: Atlas가 Titans 성능을 초과, 초장문에서 효과적")
fig_with_caption(s, "2505.23735", 7, 6.85, 1.5, 5.9, 3.6,
                 "associative memory recall: feature map으로 저장 용량이 늘어남을 확인")
box(s, 0.55, 5.25, 12.2, 0.95,
    [("읽는 법", {"size": 12.5, "bold": True, "color": INK, "align": PP_ALIGN.LEFT, "space_after": 2}),
     ("왼쪽=같은 벤치에서 Titans 대비 향상(용량·Omega의 효과). 오른쪽=저장할 연상 쌍이 많아져도 회상 유지 → "
      "'용량 확장'이 말뿐이 아님을 직접 보여줌.", {"size": 12, "color": INK, "align": PP_ALIGN.LEFT})],
    fc=GREYB, ec=GREY, align=PP_ALIGN.LEFT)

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

s = slide(p, "B4 · TNT — 메커니즘", "메모리 계층(global/local) + 2-stage 정렬", GREEN)
fig_with_caption(s, "2511.07343", 1, 0.55, 1.5, 6.05, 3.55,
                 "메모리 계층 다이어그램: 같은 t의 갱신을 계층적으로(global↔local)")
fig_with_caption(s, "2511.07343", 3, 6.85, 1.5, 5.9, 3.55,
                 "Stage 1 아키텍처 개요 — 큰 chunk 병렬 학습")
box(s, 0.55, 5.2, 12.2, 1.0,
    [("왜 2-stage인가", {"size": 12.5, "bold": True, "color": GREEN, "align": PP_ALIGN.LEFT, "space_after": 2}),
     ("Stage1: 큰 chunk로 저비용 병렬 학습(throughput 확보). Stage2: serve chunk(작게)에 맞춰 재정렬해 "
      "train/serve 불일치를 제거. 이렇게 해야 '싸게 학습 + 정확한 서빙'이 동시에 된다.",
      {"size": 12, "color": INK, "align": PP_ALIGN.LEFT})], fc=GREENB, ec=GREEN, align=PP_ALIGN.LEFT)

s = slide(p, "B4 · TNT — 증명", "chunk 정렬이 품질을, 계층이 속도를 만든다", GREEN)
fig_with_caption(s, "2511.07343", 2, 0.55, 1.5, 6.05, 3.6,
                 "serve chunk 민감도(V자): train chunk와 맞을 때 최적, 어긋나면 급락")
fig_with_caption(s, "2511.07343", 4, 6.85, 1.5, 5.9, 3.6,
                 "runtime 비교: 시퀀스가 길어질수록 최대 17배 빠른 서빙")
box(s, 0.55, 5.25, 12.2, 0.95,
    [("읽는 법", {"size": 12.5, "bold": True, "color": INK, "align": PP_ALIGN.LEFT, "space_after": 2}),
     ("왼쪽 V자 = 'train/serve chunk mismatch가 진짜 품질을 죽인다'의 직접 증거. 오른쪽 = 계층 메모리로 "
      "긴 문맥 서빙이 실제로 빨라짐. 이 논문은 '어떻게 배포하나'를 처음으로 정면으로 다룬다.",
      {"size": 12, "color": INK, "align": PP_ALIGN.LEFT})], fc=GREYB, ec=GREY, align=PP_ALIGN.LEFT)

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
