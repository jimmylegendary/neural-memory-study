#!/usr/bin/env python3
"""X4' — 저장을 읽는 연산자가 자기 자신일 때: 교사 되먹임 채널과 그 선행 경보.

배경 — 왜 X4를 다시 설계하는가
------------------------------
원래 X4의 질문은 **"반복하면 얼마나 빨리 무너지는가"**였다. Tier 1 열 편을 읽은 뒤
그 질문 자체가 바뀌었다(TIER1-ADJUDICATION-2026-08-11 §5).

  [2606.04703 continual-internalization] $K=3$. $\\mathcal{R}_k$가 **자기생성**이다 —
      라운드 $k$의 정책이 만든 궤적을 요약한 $E^{(k)}$로 $\\Theta_k$ 자신이 교사가 되어
      $\\Theta_{k+1}$을 만든다. 3라운드에 progressive capability collapse.
      그리고 이 논문의 가장 무거운 관측: **진단 지표가 배포 지표보다 한 라운드 먼저 죽는다**
      (구성 C — 내재화 $25.9\\to31.0$으로 **오르는** 라운드에 문맥 성적은 $28.1\\to14.1$로
      절반이 되고, 한 라운드 뒤 내재화가 12.8로 18.2점을 잃는다) [§5.2.3, Table 4].
  [2605.11836 lifelong-normalization] $K=200/5{,}000/20{,}000$. $\\mathcal{R}_k$가 **외생**이다 —
      고정 분포에서 i.i.d. 추출된 라벨 배치, $\\mathrm{gen}(\\cdot;B_s)$ = 항등.
      LN(centering+whitening)+ridge라는 **수축**이 있고, 그것을 빼면 같은 스트림 20K에서 붕괴한다.
      early edits promote future edits — 부호가 반대다. **이전 라운드 항목의 유지는 재지 않는다.**
  [2601.11042 editing-collapse-spectral] 지배적 특이방향을 보존(REVIVE)하면 10,000 편집에서
      86.34%를 유지한다. 보호가 없으면 MEMIT/RECT는 3,000에, AlphaEdit는 8,000에 붕괴.
      수축의 두 번째 형태 = **부분공간 투영**. 그리고 순차 편집에서 가중치 노름의 이상 증가 [§4.2].

두 논문은 반대 방향이고 **둘 다 옳다.** 병합하지 않고 조건절로 만든다:
  $\\rho$는 (U-$\\Theta$)의 반복 그 자체가 아니라 **(갱신 규칙, 스트림의 출처, 재는 축)**
  삼중항의 성질이다.

이 실험이 묻는 것 (§5.5)
------------------------
  Q1 (G-B, 최우선) **저장을 읽는 연산자가 자기 자신일 때** 그 되먹임이 붕괴를 앞당기는가.
     $\\Theta_k$가 (i) 정책이자 (ii) $E^{(k)}$를 읽어 학습 타깃을 만드는 연산자일 때,
     읽기 능력의 손상이 다음 라운드의 감독을 약하게 하고 복리로 누적되는가.
     이것은 catastrophic forgetting(간섭)도 model collapse(분포 모멘트)도 아닌 **제3의 채널**이며
     corpus 어느 논문에도 없고 원 X4의 최소 모형에도 없었다.
  Q2 그 앞당김을 **무엇이 먼저 알려 주는가.** 후보 셋을 라운드 함수로 같이 인쇄한다 —
     (a) 「문맥을 준 성적 − 문맥 없는 성적」[2606.04703이 제공한 배포 가능한 선행 지표],
     (b) running covariance의 **조건수**[2605.11836 App. B.3.3],
     (c) **가중치 노름 증가비**[2601.11042 §4.2].
     선행인지 후행인지는 **측정 결과**이지 가정이 아니다. 후행이면 후행이라고 적는다.
  Q3 (G-C) 「원본 앵커」와 「정오 필터」 중 무엇이 일하는가. 지금까지 ch29 판정 4의 근거가
     둘을 뭉개고 있다. 2×2×2가 값싸게 가른다.
  Q4 (G-A, 주공백) **외생 스트림 + 수축 있는 갱신 규칙**에서 「이전 라운드에 쓴 것」의 유지는
     어떻게 되는가. 2605.11836은 그 축을 재지 않았고 2606.04703은 그 구성을 돌리지 않았다.
  Q5 초기 라운드에 **조건 간 순서 반전**이 있는가. 두 논문 모두 보고했으므로 이제 검사할 가설이다.

설계 — 최소 모형이 무엇을 보존하고 무엇을 못 하는가
--------------------------------------------------
LLM을 쓸 수 없다(이 host에 NVIDIA GPU 없음, torch 금지). 기제가 보존되는 최소 모형을 쓰되
**재현하지 못한 것을 재현했다고 말하지 않는다.**

(a) **읽기 채널을 파라미터 안에 넣는다.** 이것이 원 X4에 없던 것이다.
    저장 $E$의 항목은 $e_f = S v_f$($S$ = 고정 직교행렬 = store codec, $v_f$ = 사실 $f$의 값).
    기저 모형 $\\Theta_0 = \\Theta_{\\mathrm{bg}} + \\gamma S^{\\top}$ 은 **일반적 읽기 연산자**를
    내장한다 — $\\Theta_0 e_f = \\Theta_{\\mathrm{bg}} S v_f + \\gamma v_f$. 사실별 암기가 아니라
    한 장의 행렬이므로 **한 번도 쓰지 않은 사실에도 작동한다.** 이것이 in-context 능력의 대응물이고,
    이 능력의 열화가 정확히 2606.04703의 'experience-use ability' 손상이다.
    쓰기 갱신 $\\Delta\\Theta = -\\eta\\,(\\Theta k - v)k^{\\top}$ 의 행공간은 키 방향이고,
    키와 저장 방향은 직교가 아니라 $\\cos\\approx d^{-1/2}$ 로 겹치므로 읽기 채널이
    **간섭으로 서서히 죽는다** — 손으로 넣은 감쇠 상수가 아니라 기하의 귀결이다.
(b) **교사 되먹임.** replay/consolidation 타깃을 현재 모형이 저장을 읽어 만든다:
    $v^{\\mathrm{tgt}} = \\mathrm{unit}\\big(\\Theta_k\\,\\mathrm{unit}(k_{\\mathrm{item}} + \\kappa e_f)\\big)$.
    이것이 2606.04703의 context distillation이다(교사 = 문맥 조건화된 자기 자신, 학생 = 문맥 없는 자기 자신).
    읽기가 죽으면 타깃이 죽고, 타깃이 죽으면 읽기가 더 죽는다. 복리.
(c) **2×2×2** {타깃: teacher(off-policy, 문맥 조건화) / student(on-policy, 문맥 없음)}
    × {정오 필터: filter(rejection sampling, 정답 코드북 대조) / none}
    × {스트림 출처: exo(외생 라벨 배치, gen=항등) / self(모형이 저장을 읽어 만든 내용)}
    + 참조 팔 {target=gold = **보존된 원본 앵커**} × {exo, self}.
    앞의 둘이 G-C를 가르고 셋째가 G-A를 연다.
(d) **수축(contraction) 팔** — G-A 전용. gold·exo/self 위에
    ridge(라운드 시작점으로의 근접항 = 2605.11836의 ridge 대응물)와
    subspace($\\Theta_0$의 상위 특이부분공간에서 갱신을 제거 = 2601.11042 REVIVE의 대응물)를 얹는다.
(e) **예산 정합.** 필터가 항목을 버려도 라운드당 항목 수 $N_{\\mathrm{total}}$과 경사 스텝 $S$를 고정한다
    (수락된 것에서 복원추출로 채운다). [2606.04703 App. A.3] 'steps are matched, not tokens'와 같은 이유.
(f) **천장에서 돌지 않는다.** 원 X4의 access는 $k\\le25$에서 1.0에 붙어 초기 순서가 정의되지 않았다.
    paraphrase spread를 키워 작동점을 access$\\approx$0.6으로 내렸다. 그 결과 **초기 순서 반전이
    관측 가능한 양**이 되었다.

라운드의 조작적 정의 (→ B1 지적)
--------------------------------
$K^\\ast$가 200인지 20,000인지 비교하려면 라운드의 크기를 인쇄해야 한다. 이 실험은
라운드당 항목 수·경사 스텝 수·**측정된 $\\lVert\\Delta\\Theta\\rVert_F$**·누적 경로/순이동 비를 전부 출력한다.
2605.11836의 한 라운드 = $\\eta_\\Theta=10^{-6}$ 닫힌 형태 ridge 1스텝(100편집),
2606.04703의 한 라운드 = batch 128 · 5 epoch full SFT. 두 눈금은 자릿수가 다르다.

등급
----
EXPLORATION-GRADE / MECHANISM-TRANSFER. **이것은 LLM 실험이 아니다.**
- 이송 가능: 부호, 함수형, 조건 간 **순서**, crossover의 존재와 이동 방향, 정성적 dissociation,
  선행/후행의 **순서**.
- 이송 불가: 절대 $\\rho$, 유지율 %, $K^\\ast$의 라운드 수, 반감기, 필요 $\\omega$·$\\lambda$ 값,
  선행 시간의 라운드 수.
- 논문이 보고하지 않은 값을 추정해 본문 수치로 쓰지 않는다. "논문에 없음"이 결과다.
- **공백이 남으면 남는다고 적는다.** 억지로 채우지 않는다.

결정론
------
np.random.default_rng 고정 시드 5개, 시드 간 표준편차 동반 보고(단일 시드 결론 금지).
BLAS 스레드 1로 고정(멀티스레드 reduction 순서는 부동소수 결과를 흔든다 — 이 저장소에서 실측된
유일한 비결정성 유형, REPRODUCE.md §3). 보고값 6자리 반올림. wall-clock은 result.json에 넣지 않는다.
PYTHONHASHSEED=0으로 2회 실행해 result.json의 sha256 동일을 확인한다.
"""

from __future__ import annotations

import os

# BLAS 스레드 고정: 반드시 numpy import 이전에.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RND = 6


def r(x):
    """중첩 구조 재귀 반올림 — float 잔털이 sha256을 흔들지 않게."""
    if isinstance(x, dict):
        return {k: r(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        if not math.isfinite(v):
            return None
        return round(v, RND)
    return x


# =============================================================================
# 설정
# =============================================================================

CFG = dict(
    d=128,               # 파라미터 행렬 차원 (Theta in R^{d x d})
    n_cohort=8,          # 라운드 0에 기입하는 cohort 사실 수 (유지 측정 대상)
    n_stream=150,        # 이후 라운드 수 K
    n_heldout=24,        # 한 번도 쓰지 않는 사실 — 순수 읽기 능력 프로브
    n_distract=40,       # 코드북 방해 항목
    J_max=8,             # 사실당 학습용 paraphrase 키 수
    n_probe=6,           # 사실당 평가 전용 프로브 키 수 (학습에 절대 안 씀)
    spread=2.0,          # paraphrase 확산 — 작동점을 천장에서 내리는 손잡이
    gamma=2.5,           # Theta_0에 심는 읽기 연산자의 세기
    kappa=1.0,           # 문맥(저장 항목) 주입 세기
    N_total=16,          # 라운드당 항목 수 (예산 정합, 전 조건 고정)
    steps=20,            # 라운드당 경사 스텝 수 (예산 정합, 전 조건 고정)
    lr=0.4,              # 경사 보폭
    n_cap=200,           # capability 프로브 수
    n_genprobe=64,       # 생성 분포 2차 모멘트용 고정 프로브 수
    ridge_eps=1e-3,      # 조건수 계산의 ridge (rank 결손 라운드에서 유한하게)
    prox_mu=0.10,        # 수축 팔(ridge)의 근접항 계수
    revive_rank=8,       # 수축 팔(subspace)이 보호하는 상위 특이방향 수
)

SEEDS = [101, 102, 103, 104, 105]
OMEGA = 0.5              # 요인설계의 replay 혼합비 (고정)
OMEGA_SWEEP = [0.25, 0.75]

# 체크포인트 격자 — 초기 구간을 촘촘히 (순서 반전 검사가 거기서 일어난다)
CKPT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50,
        60, 75, 90, 105, 120, 135, 150]

SMOOTH_W = 10            # 습득 지표(라운드당 사실 1개)의 이동평균 폭


# =============================================================================
# 세계
# =============================================================================

def unit(a):
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.where(n > 0, n, 1.0)


class World:
    """사실 = (앵커 키, 코드북 값). 저장 항목 e_f = S v_f. Theta_0는 일반 읽기 연산자를 내장한다."""

    def __init__(self, cfg, seed):
        rng = np.random.default_rng(seed)
        d = cfg["d"]
        C, K, H = cfg["n_cohort"], cfg["n_stream"], cfg["n_heldout"]
        M = C + K + H
        self.cfg, self.d, self.M = cfg, d, M
        self.C, self.K, self.H = C, K, H

        self.theta_bg = rng.standard_normal((d, d)) / math.sqrt(d)
        Q, R_ = np.linalg.qr(rng.standard_normal((d, d)))
        Q = Q * np.sign(np.diag(R_))            # QR 부호 규약 고정 -> 결정론
        self.S = Q
        self.theta0 = self.theta_bg + cfg["gamma"] * Q.T
        self.theta0_fro = float(np.linalg.norm(self.theta0))

        self.codebook = unit(rng.standard_normal((M + cfg["n_distract"], d)))
        self.values = self.codebook[:M]
        self.chance = 1.0 / self.codebook.shape[0]
        self.store = self.values @ Q.T          # e_f = S v_f (S 직교 -> 단위)

        self.anchors = unit(rng.standard_normal((M, d)))
        npool = cfg["J_max"] + cfg["n_probe"]
        noise = rng.standard_normal((M, npool, d)) / math.sqrt(d)
        pool = unit(self.anchors[:, None, :] + cfg["spread"] * noise)
        self.train_keys = pool[:, :cfg["J_max"], :]
        self.probe_keys = pool[:, cfg["J_max"]:, :]

        self.cap_q = unit(rng.standard_normal((cfg["n_cap"], d)))
        self.cap_t = unit(self.cap_q @ self.theta0.T)
        self.gen_q = unit(rng.standard_normal((cfg["n_genprobe"], d)))

        # 수축(subspace) 팔이 보호할 Theta_0의 상위 특이부분공간
        U, sv, Vt = np.linalg.svd(self.theta0)
        rk = cfg["revive_rank"]
        self.U_prot = U[:, :rk]
        self.V_prot = Vt[:rk, :].T
        self.theta0_top_sv = sv[:rk].tolist()


# =============================================================================
# 측정
# =============================================================================

def dec_acc(theta, keys, labels, cb):
    """문맥 없는 디코딩 정확도 = 내재화 성적."""
    return float(np.mean(np.argmax((keys @ theta.T) @ cb.T, axis=1) == labels))


def dec_acc_ctx(theta, keys, labels, cb, store, kappa):
    """문맥(저장 항목)을 준 디코딩 정확도 = 문맥 성적."""
    x = unit(keys + kappa * store[labels])
    return float(np.mean(np.argmax((x @ theta.T) @ cb.T, axis=1) == labels))


def cos_to_value(theta, keys, vals):
    y = keys @ theta.T
    ny = np.linalg.norm(y, axis=1)
    return np.sum(y * vals, axis=1) / np.where(ny > 0, ny, 1.0)


def part_ratio(X):
    """참여비 (Σλ)^2/Σλ^2 — 2차 모멘트의 유효 차원. 값이 1로 가면 한 방향으로 붕괴."""
    if X.shape[0] < 2:
        return None
    G = X.T @ X
    tr = float(np.trace(G))
    fr = float(np.sum(G * G))
    return (tr * tr / fr) if fr > 0 else None


def mean_pairwise_cos(X):
    if X.shape[0] < 2:
        return None
    Z = unit(X)
    C = Z @ Z.T
    n = C.shape[0]
    off = (float(C.sum()) - float(np.trace(C))) / (n * (n - 1))
    return off


def top_mode_share(X, cb):
    if X.shape[0] < 1:
        return None
    lab = np.argmax(X @ cb.T, axis=1)
    return float(np.max(np.bincount(lab)) / len(lab))


# =============================================================================
# 쓰기 연산자 (+ 수축 두 형태)
# =============================================================================

def write_round(theta, K_items, V_items, cfg, contraction):
    """한 라운드의 쓰기. steps·항목 수는 예산 정합으로 고정."""
    start = theta
    m = K_items.shape[0]
    lr, S = cfg["lr"], cfg["steps"]
    mu = cfg["prox_mu"] if contraction == "ridge" else 0.0
    for _ in range(S):
        g = (K_items @ theta.T - V_items).T @ K_items / m
        if mu > 0.0:
            g = g + mu * (theta - start)
        theta = theta - lr * g
    return theta, start


def apply_subspace_guard(theta, start, W):
    """REVIVE 대응물: 라운드 순갱신에서 Theta_0의 상위 특이부분공간 성분을 제거."""
    dT = theta - start
    U, V = W.U_prot, W.V_prot
    dT = dT - U @ (U.T @ dT)
    dT = dT - (dT @ V) @ V.T
    return start + dT


# =============================================================================
# 한 조건 1회 실행
# =============================================================================

def run_cond(cfg, seed, target, filt, stream, omega=OMEGA, contraction="none"):
    """target: gold(보존된 원본) | teacher(문맥 조건화 자기 출력) | student(문맥 없는 자기 출력)
       filt:   filter(정답 대조 rejection sampling) | none
       stream: exo(외생 라벨 배치) | self(모형이 저장을 읽어 만든 내용)
       contraction: none | ridge | subspace
    """
    W = World(cfg, seed)
    rng = np.random.default_rng(seed * 7919 + 13)
    d, C, K = cfg["d"], cfg["n_cohort"], cfg["n_stream"]
    N, kap = cfg["N_total"], cfg["kappa"]
    cb, th = W.codebook, W.theta0.copy()

    coh_pk = W.probe_keys[:C].reshape(-1, d)
    coh_lab = np.repeat(np.arange(C), cfg["n_probe"])
    coh_anchor, coh_vals = W.anchors[:C], W.values[:C]
    hid = np.arange(C + K, W.M)
    ho_pk = W.probe_keys[hid].reshape(-1, d)
    ho_lab = np.repeat(hid, cfg["n_probe"])
    ho_e = W.store[hid]

    # --- 라운드 0: cohort 기입 (전 조건 동일, gold, 수축 없음)
    cos_before = cos_to_value(th, coh_anchor, coh_vals)
    gram = np.zeros((d, d))
    gram_n = 0
    buf_f, buf_k = [], []
    for fid in range(C):
        ks = W.train_keys[fid][np.arange(N) % cfg["J_max"]]
        vs = np.repeat(W.values[fid][None, :], N, axis=0)
        th, _ = write_round(th, ks, vs, cfg, "none")
        gram += ks.T @ ks; gram_n += ks.shape[0]
        buf_f.append(np.full(cfg["J_max"], fid)); buf_k.append(W.train_keys[fid])
    bf = np.concatenate(buf_f); bk = np.concatenate(buf_k)
    cos_after = cos_to_value(th, coh_anchor, coh_vals)
    denom = np.where(np.abs(cos_after - cos_before) > 1e-9, cos_after - cos_before, np.nan)

    tr = {k: [] for k in (
        "access", "storage", "lift", "ctx_cohort", "capability", "read_op_cos",
        "ctx_heldout", "free_heldout", "read_lift_heldout", "drift", "norm_ratio",
        "cond_cov", "gen_pr", "gen_cos", "gen_mode_share")}
    per_round = {k: [] for k in (
        "dtheta", "free_new", "ctx_new", "reject_new", "reject_replay",
        "written_pr", "written_cos", "written_mode_share", "n_written")}

    def snapshot():
        tr["access"].append(dec_acc(th, coh_pk, coh_lab, cb))
        tr["storage"].append(dec_acc(th, coh_anchor, np.arange(C), cb))
        cn = cos_to_value(th, coh_anchor, coh_vals)
        tr["lift"].append(float(np.nanmean((cn - cos_before) / denom)))
        tr["ctx_cohort"].append(dec_acc_ctx(th, coh_pk, coh_lab, cb, W.store, kap))
        tr["capability"].append(
            float(np.mean(np.sum(unit(W.cap_q @ th.T) * W.cap_t, axis=1))))
        tr["read_op_cos"].append(
            float(np.mean(np.sum(unit(ho_e @ th.T) * unit(ho_e @ W.theta0.T), axis=1))))
        ch = dec_acc_ctx(th, ho_pk, ho_lab, cb, W.store, kap)
        fh = dec_acc(th, ho_pk, ho_lab, cb)
        tr["ctx_heldout"].append(ch); tr["free_heldout"].append(fh)
        tr["read_lift_heldout"].append(ch - fh)
        tr["drift"].append(float(np.linalg.norm(th - W.theta0) / W.theta0_fro))
        tr["norm_ratio"].append(float(np.linalg.norm(th) / W.theta0_fro))
        ev = np.linalg.eigvalsh(gram / max(gram_n, 1))
        eps = cfg["ridge_eps"]
        tr["cond_cov"].append(float((ev[-1] + eps) / (ev[0] + eps)))
        G = unit(W.gen_q @ th.T)
        tr["gen_pr"].append(part_ratio(G))
        tr["gen_cos"].append(mean_pairwise_cos(G))
        tr["gen_mode_share"].append(top_mode_share(G, cb))

    snapshot()   # k = 0

    n_new = max(1, int(round((1.0 - omega) * N)))
    n_rep = N - n_new
    skipped_new = 0
    starved = 0

    for t in range(K):
        fid = C + t
        # ---------- 새 내용 (스트림 출처 축)
        if stream == "exo":
            v_new, ok_new, rej_new = W.values[fid], True, 0.0
        else:
            x = unit(W.anchors[fid] + kap * W.store[fid])
            y = unit(x @ th.T)
            v_new = y
            ok_new = bool(int(np.argmax(y @ cb.T)) == fid)
            rej_new = 0.0 if ok_new else 1.0
        ks_l, vs_l, gen_l = [], [], []
        if ok_new or filt == "none":
            kk = W.train_keys[fid][np.arange(n_new) % cfg["J_max"]]
            ks_l.append(kk); vs_l.append(np.repeat(v_new[None, :], n_new, axis=0))
            gen_l.append(v_new[None, :])
        else:
            skipped_new += 1

        # ---------- replay/consolidation (타깃 축 × 정오 필터 축)
        rej_rep = 0.0
        if n_rep > 0:
            sel = rng.integers(0, bk.shape[0], size=n_rep * 2)
            kr, fr = bk[sel], bf[sel]
            if target == "gold":
                vr = W.values[fr]
            elif target == "teacher":
                vr = unit(unit(kr + kap * W.store[fr]) @ th.T)
            else:
                vr = unit(kr @ th.T)
            if filt == "filter" and target != "gold":
                ok = np.argmax(vr @ cb.T, axis=1) == fr
                rej_rep = float(1.0 - ok.mean())
                kr, vr = kr[ok], vr[ok]
            if kr.shape[0] > 0:
                gen_l.append(vr)
                idx = np.arange(n_rep) % kr.shape[0]     # 예산 정합 복원 채움
                ks_l.append(kr[idx]); vs_l.append(vr[idx])

        # ---------- 쓰기
        if ks_l:
            Kb = np.concatenate(ks_l); Vb = np.concatenate(vs_l)
            th_new, start = write_round(th, Kb, Vb, cfg, contraction)
            if contraction == "subspace":
                th_new = apply_subspace_guard(th_new, start, W)
            per_round["dtheta"].append(float(np.linalg.norm(th_new - th)))
            per_round["n_written"].append(int(Kb.shape[0]))
            gram += Kb.T @ Kb; gram_n += Kb.shape[0]
            th = th_new
        else:
            starved += 1
            per_round["dtheta"].append(0.0)
            per_round["n_written"].append(0)

        # ---------- 라운드 지표
        lab = np.full(cfg["n_probe"], fid)
        per_round["free_new"].append(dec_acc(th, W.probe_keys[fid], lab, cb))
        per_round["ctx_new"].append(
            dec_acc_ctx(th, W.probe_keys[fid], lab, cb, W.store, kap))
        per_round["reject_new"].append(rej_new)
        per_round["reject_replay"].append(rej_rep)
        Gg = np.concatenate(gen_l) if gen_l else np.zeros((0, d))
        per_round["written_pr"].append(part_ratio(Gg))
        per_round["written_cos"].append(mean_pairwise_cos(Gg))
        per_round["written_mode_share"].append(top_mode_share(Gg, cb) if len(Gg) else None)

        bf = np.concatenate([bf, np.full(cfg["J_max"], fid)])
        bk = np.concatenate([bk, W.train_keys[fid]])
        snapshot()

    out = dict(tr)
    out.update(per_round)
    out["skipped_new"] = skipped_new
    out["starved_rounds"] = starved
    out["chance"] = W.chance
    out["theta0_top_sv"] = W.theta0_top_sv
    return out


# =============================================================================
# 집계 · 모양 통계 (X4에서 이월 — 비교 가능성을 위해 정의를 바꾸지 않는다)
# =============================================================================

def stack(runs, key):
    rows = []
    for x in runs:
        v = [np.nan if y is None else float(y) for y in x[key]]
        rows.append(v)
    return np.array(rows, dtype=float)


def mstd(runs, key):
    A = stack(runs, key)
    return np.nanmean(A, axis=0), np.nanstd(A, axis=0, ddof=0)


def at_ckpt(vec, ck=CKPT):
    return [float(vec[i]) for i in ck if i < len(vec)]


def smooth(v, w=SMOOTH_W):
    """뒤쪽 이동평균 — 라운드당 사실 1개짜리 습득 지표의 잡음을 줄인다(정의 인쇄)."""
    v = np.asarray(v, dtype=float)
    out = np.empty_like(v)
    for i in range(len(v)):
        lo = max(0, i - w + 1)
        out[i] = np.nanmean(v[lo:i + 1])
    return out


def first_below(traj, thr, skip0=True):
    a = np.asarray(traj, dtype=float)
    a = a[1:] if skip0 else a
    idx = np.where(a < thr)[0]
    return (int(idx.min()) + (1 if skip0 else 0)) if len(idx) else None


def kstar(traj, thr):
    """K* = 지표가 thr 이상을 유지한 마지막 라운드. 끝까지면 censored."""
    a = np.asarray(traj, dtype=float)[1:]
    idx = np.where(a >= thr)[0]
    if len(idx) == 0:
        return 0, False
    kk = int(idx.max()) + 1
    return kk, kk >= len(a)


_LAM = np.concatenate([np.linspace(0.002, 0.2, 120), np.linspace(0.205, 2.0, 180)])


def _lsq(D, y):
    coef, *_ = np.linalg.lstsq(D, y, rcond=None)
    res = y - D @ coef
    return coef, float(res @ res)


def fit_shapes(y):
    """linear / exp / plateau / logistic 네 후보. X4와 동일한 격자·동일한 정의."""
    y = np.asarray(y, dtype=float)
    k = np.arange(1, len(y) + 1, dtype=float)
    tss = float(((y - y.mean()) ** 2).sum())
    out = {}
    coef, sse = _lsq(np.stack([np.ones_like(k), k], axis=1), y)
    out["linear"] = {"sse": sse, "a": coef[0], "b": coef[1]}
    best = None
    for lam in _LAM:
        c, s = _lsq(np.exp(-lam * k)[:, None], y)
        if best is None or s < best[1]:
            best = (lam, s, c)
    out["exp"] = {"sse": best[1], "lam": best[0], "A": best[2][0]}
    best = None
    for lam in _LAM:
        c, s = _lsq(np.stack([np.ones_like(k), np.exp(-lam * k)], axis=1), y)
        if best is None or s < best[1]:
            best = (lam, s, c)
    out["plateau"] = {"sse": best[1], "lam": best[0], "c": best[2][0], "A": best[2][1]}
    best = None
    for k0 in np.linspace(1.0, float(len(y)) * 1.2, 90):
        for tau in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0):
            c, s = _lsq((1.0 / (1.0 + np.exp((k - k0) / tau)))[:, None], y)
            if best is None or s < best[2]:
                best = (k0, tau, s, c)
    out["logistic"] = {"sse": best[2], "k0": best[0], "tau": best[1], "a": best[3][0]}
    for m in ("linear", "exp", "plateau", "logistic"):
        out[m]["r2"] = (1.0 - out[m]["sse"] / tss) if tss > 0 else None
    out["best"] = min(("linear", "exp", "plateau", "logistic"), key=lambda m: out[m]["sse"])
    sses = sorted(out[m]["sse"] for m in ("linear", "exp", "plateau", "logistic"))
    out["sse_ratio_2nd_over_best"] = (sses[1] / sses[0]) if sses[0] > 0 else None
    return out


def shape_stats(traj):
    """모형 선택에 의존하지 않는 모양 통계 (X4 정의 그대로 + 상승-후-절벽 검사)."""
    a = np.asarray(traj, dtype=float)
    rho = -np.diff(a) * 100.0
    a0 = a[0]
    rel = a / a0 if a0 > 0 else a
    peak = int(np.argmax(a))
    return {
        "value_at_k0": float(a0),
        "grace_round_first_below_95pct_of_k0": first_below(rel, 0.95),
        "first_below_75pct_of_k0": first_below(rel, 0.75),
        "first_below_50pct_of_k0": first_below(rel, 0.50),
        "first_below_abs_0.5": first_below(a, 0.5),
        "rho_mean_first10_pp": float(np.nanmean(rho[:10])),
        "rho_max_pp": float(np.nanmax(rho)),
        "rho_argmax_round": int(np.nanargmax(rho) + 1),
        "rho_max_over_first10": (None if abs(np.nanmean(rho[:10])) < 1e-9
                                 else float(np.nanmax(rho) / np.nanmean(rho[:10]))),
        "peak_round": peak,
        "peak_value": float(a[peak]),
        "rise_before_fall": bool(peak >= 1 and a[peak] > a0 + 1e-9),
    }


def _avg_rank(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = _avg_rank(a), _avg_rank(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = math.sqrt(float(ra @ ra) * float(rb @ rb))
    return float(ra @ rb / den) if den > 0 else None


def alarm_round(traj, frac=0.75):
    """자기 자신의 k=0 값 대비 frac 아래로 처음 내려간 라운드. 선행/후행 판정의 공통 자."""
    a = np.asarray(traj, dtype=float)
    if not np.isfinite(a[0]) or a[0] <= 0:
        return None
    return first_below(a / a[0], frac)


def alarm_round_absmove(traj, frac=0.25):
    """**방향 무관** 경보 — 첫 값 대비 비가 [1-frac, 1/(1-frac)] 밖으로 처음 나간 라운드.
    내려가는 지표(읽기 lift)와 올라가는 지표(조건수·노름)를 같은 자로 재기 위한 규칙이다.
    내려가는 쪽에서는 alarm_round(·, 1-frac)과 정확히 같다."""
    a = np.asarray(traj, dtype=float)
    if len(a) == 0 or not np.isfinite(a[0]) or a[0] <= 0:
        return None
    lo, hi = 1.0 - frac, 1.0 / (1.0 - frac)
    ratio = a / a[0]
    for i in range(1, len(ratio)):
        if not np.isfinite(ratio[i]):
            continue
        if ratio[i] < lo or ratio[i] > hi:
            return i
    return None


def observed_direction(traj):
    """이 실행에서 실제로 관측된 방향. 앞 1/3 평균과 뒤 1/3 평균의 차로 판정하고
    (평평한 구간의 잔털에 흔들리지 않게), 단조 비율을 함께 돌려준다.
    원 논문이 보고한 방향과 다르면 그대로 인쇄한다."""
    a = np.asarray(traj, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 6:
        return "undetermined", None
    t = len(a) // 3
    head, tail = float(np.mean(a[:t])), float(np.mean(a[-t:]))
    frac_up = float(np.mean(np.diff(a) > 0))
    if abs(tail - head) <= 1e-9 * max(abs(head), 1.0):
        return "flat", frac_up
    return ("rising" if tail > head else "falling"), frac_up


# =============================================================================
# main
# =============================================================================

def cond_name(t, f, s, omega=OMEGA, contr="none"):
    n = f"tgt={t}|filt={f}|stream={s}"
    if omega != OMEGA:
        n += f"|omega={omega}"
    if contr != "none":
        n += f"|contr={contr}"
    return n


def main():
    cfg = CFG
    K = cfg["n_stream"]

    # ---------------- 조건 목록
    factorial = []
    for t in ("teacher", "student"):
        for f in ("filter", "none"):
            for s in ("exo", "self"):
                factorial.append((t, f, s, OMEGA, "none"))
    reference = [("gold", "none", "exo", OMEGA, "none"),
                 ("gold", "none", "self", OMEGA, "none")]
    contraction_arms = [("gold", "none", s, OMEGA, c)
                        for s in ("exo", "self") for c in ("ridge", "subspace")]
    omega_arms = [("gold", "none", "exo", w, "none") for w in OMEGA_SWEEP] + \
                 [("teacher", "filter", "self", w, "none") for w in OMEGA_SWEEP]
    all_conds = reference + factorial + contraction_arms + omega_arms

    results = {}
    for (t, f, s, w, c) in all_conds:
        runs = [run_cond(cfg, sd, t, f, s, omega=w, contraction=c) for sd in SEEDS]
        results[cond_name(t, f, s, w, c)] = dict(spec=(t, f, s, w, c), runs=runs)

    chance = float(results[cond_name("gold", "none", "exo")]["runs"][0]["chance"])

    # ---------------- 조건별 집계
    agg = {}
    for name, blob in results.items():
        runs = blob["runs"]
        a = {}
        for key in ("access", "storage", "lift", "ctx_cohort", "capability",
                    "read_op_cos", "ctx_heldout", "free_heldout", "read_lift_heldout",
                    "drift", "norm_ratio", "cond_cov", "gen_pr", "gen_cos",
                    "gen_mode_share"):
            m, sd = mstd(runs, key)
            a[key] = {"ckpt_k": [k for k in CKPT if k <= K],
                      "mean": at_ckpt(m), "std": at_ckpt(sd)}
            a["_full_" + key] = m
        for key in ("free_new", "ctx_new", "dtheta", "reject_new", "reject_replay",
                    "written_pr", "written_cos", "written_mode_share", "n_written"):
            m, sd = mstd(runs, key)
            ms = smooth(m)
            a[key + "_smooth"] = {"ckpt_k": [k for k in CKPT if 1 <= k <= K],
                                  "mean": [float(ms[k - 1]) for k in CKPT if 1 <= k <= K],
                                  "std_raw": [float(sd[k - 1]) for k in CKPT if 1 <= k <= K]}
            a["_full_" + key] = m
            a["_smooth_" + key] = ms
        a["ctx_minus_free_new_smooth"] = {
            "ckpt_k": [k for k in CKPT if 1 <= k <= K],
            "mean": [float(a["_smooth_ctx_new"][k - 1] - a["_smooth_free_new"][k - 1])
                     for k in CKPT if 1 <= k <= K]}
        a["_full_ctx_minus_free_new"] = a["_smooth_ctx_new"] - a["_smooth_free_new"]
        a["skipped_new_mean"] = float(np.mean([x["skipped_new"] for x in runs]))
        a["starved_rounds_mean"] = float(np.mean([x["starved_rounds"] for x in runs]))
        agg[name] = a

    # ---------------- B0 작동점 (천장에서 돌지 않는다는 증명)
    ref = agg[cond_name("gold", "none", "exo")]
    B0 = {
        "why": "원 X4의 access는 k<=25에서 1.0에 붙어 초기 구간의 조건 간 순서가 정의되지 않았다"
               "(설계 변경 3). paraphrase spread를 0.6 -> 2.0으로 올려 작동점을 내렸다.",
        "cohort_access_at_k0_mean": float(ref["_full_access"][0]),
        "cohort_access_at_k0_std_across_seeds": float(np.nanstd(
            stack(results[cond_name("gold", "none", "exo")]["runs"], "access")[:, 0], ddof=0)),
        "n_conditions_at_ceiling_k0": int(sum(
            1 for n in agg if agg[n]["_full_access"][0] > 0.99)),
        "chance_level": chance,
        "ceiling_free_window_rounds": "전 조건 k=0부터 천장(1.0) 아래. 초기 순서가 정의된다.",
        "ctx_heldout_at_k0_mean": float(ref["_full_ctx_heldout"][0]),
        "free_heldout_at_k0_mean": float(ref["_full_free_heldout"][0]),
        "reading_operator_is_general": "free_heldout은 전 라운드 chance 근방이고 ctx_heldout은 k=0에서 "
                                       "높다 — 읽기 연산자가 사실별 암기가 아니라 한 장의 행렬임을 뜻한다.",
    }

    # ---------------- B1 2x2x2 요인설계
    def cell(name):
        a = agg[name]
        ks_rel, cens_rel = kstar(a["_full_access"] / max(a["_full_access"][0], 1e-9), 0.5)
        ks_abs, cens_abs = kstar(a["_full_access"], 0.5)
        per_seed_rel = []
        for x in results[name]["runs"]:
            v = np.asarray(x["access"], dtype=float)
            kk, _ = kstar(v / max(v[0], 1e-9), 0.5)
            per_seed_rel.append(kk)
        return {
            "access_k0": float(a["_full_access"][0]),
            "access_k10": float(a["_full_access"][10]),
            "access_k50": float(a["_full_access"][50]),
            "access_k150": float(a["_full_access"][K]),
            "access_std_k150": float(a["access"]["std"][-1]),
            "Kstar_rel50": ks_rel, "Kstar_rel50_censored": cens_rel,
            "Kstar_rel50_per_seed": per_seed_rel,
            "Kstar_rel50_std_across_seeds": float(np.std(per_seed_rel, ddof=0)),
            "Kstar_abs50": ks_abs, "Kstar_abs50_censored": cens_abs,
            "capability_k150": float(a["_full_capability"][K]),
            "read_lift_heldout_k0": float(a["_full_read_lift_heldout"][0]),
            "read_lift_heldout_k50": float(a["_full_read_lift_heldout"][50]),
            "read_lift_heldout_k150": float(a["_full_read_lift_heldout"][K]),
            "free_new_smooth_last10": float(np.nanmean(a["_full_free_new"][-10:])),
            "drift_k150": float(a["_full_drift"][K]),
            "skipped_new_mean": a["skipped_new_mean"],
            "starved_rounds_mean": a["starved_rounds_mean"],
        }

    B1 = {
        "design": "{타깃: teacher/student} x {정오 필터: filter/none} x {스트림: exo/self}, "
                  f"omega={OMEGA}, 수축 없음. 참조 팔 = 보존된 원본 앵커(gold) x {{exo,self}}.",
        "Kstar_definition": "K*_rel50 = cohort access가 자기 자신의 k=0 값의 50% 이상을 유지한 "
                            "마지막 라운드. K*_abs50 = 절대 0.5 기준(X4 정의와의 연결용). "
                            f"censored=True는 K={K}에서 임계를 넘지 않았다는 뜻(하한).",
        "cells": {n: cell(n) for n in
                  [cond_name(*c) for c in reference + factorial]},
    }
    # 주효과 (K*_rel50 위)
    def ks_of(t, f, s):
        return B1["cells"][cond_name(t, f, s)]["Kstar_rel50"]
    main_eff = {
        "target_teacher_minus_student_rounds": float(np.mean(
            [ks_of("teacher", f, s) - ks_of("student", f, s)
             for f in ("filter", "none") for s in ("exo", "self")])),
        "filter_on_minus_off_rounds": float(np.mean(
            [ks_of(t, "filter", s) - ks_of(t, "none", s)
             for t in ("teacher", "student") for s in ("exo", "self")])),
        "stream_self_minus_exo_rounds": float(np.mean(
            [ks_of(t, f, "self") - ks_of(t, f, "exo")
             for t in ("teacher", "student") for f in ("filter", "none")])),
    }
    inter = {
        "filter_effect_within_exo": float(np.mean(
            [ks_of(t, "filter", "exo") - ks_of(t, "none", "exo") for t in ("teacher", "student")])),
        "filter_effect_within_self": float(np.mean(
            [ks_of(t, "filter", "self") - ks_of(t, "none", "self") for t in ("teacher", "student")])),
        "target_effect_within_exo": float(np.mean(
            [ks_of("teacher", f, "exo") - ks_of("student", f, "exo") for f in ("filter", "none")])),
        "target_effect_within_self": float(np.mean(
            [ks_of("teacher", f, "self") - ks_of("student", f, "self") for f in ("filter", "none")])),
    }
    B1["main_effects_on_Kstar_rel50"] = main_eff
    B1["interactions_on_Kstar_rel50"] = inter

    # ---------------- B2 교사 되먹임 채널 (G-B)
    def feedback_block(name):
        a = agg[name]
        rl = a["_full_read_lift_heldout"]
        fn = a["_smooth_free_new"]
        return {
            "read_lift_heldout_ckpt": a["read_lift_heldout"],
            "read_op_cos_ckpt": a["read_op_cos"],
            "reject_new_smooth_ckpt": a["reject_new_smooth"],
            "reject_replay_smooth_ckpt": a["reject_replay_smooth"],
            "alarm_round_read_lift_75pct": alarm_round(rl, 0.75),
            "alarm_round_free_new_75pct": alarm_round(fn, 0.75),
            "acquisition_last10_mean": float(np.nanmean(a["_full_free_new"][-10:])),
        }

    fb_names = [cond_name("teacher", "none", "self"), cond_name("teacher", "none", "exo"),
                cond_name("student", "none", "self"), cond_name("student", "none", "exo"),
                cond_name("gold", "none", "self"), cond_name("gold", "none", "exo")]
    # 되먹임 강도: 읽기 채널이 루프 안에 있는 팔(self stream, 자기생성 타깃) 대 없는 팔
    in_loop = [cond_name(t, f, "self") for t in ("teacher", "student") for f in ("filter", "none")]
    out_loop = [cond_name(t, f, "exo") for t in ("teacher", "student") for f in ("filter", "none")]
    B2 = {
        "what_is_new": "라운드 k의 파라미터가 (i) 정책이자 (ii) E^(k)를 읽어 학습 타깃을 만드는 "
                       "연산자다. 읽기 능력의 손상이 다음 라운드의 감독을 약하게 하는 채널 — "
                       "corpus 어느 논문에도 없고 원 X4에도 없었다(공백 G-B).",
        "how_it_is_measured": "read_lift_heldout = (문맥 준 성적 - 문맥 없는 성적), 한 번도 쓰지 "
                              "않은 24개 사실 위. 사실별 암기가 아니라 읽기 연산자 자체의 건강도다.",
        "by_condition": {n: feedback_block(n) for n in fb_names},
        "read_channel_decay_is_not_imposed":
            "감쇠 상수를 손으로 넣지 않았다. 쓰기 갱신의 행공간(키 방향)과 저장 방향의 "
            f"겹침이 cos ~ d^-1/2 = {1.0 / math.sqrt(cfg['d']):.4f} 이고, 그 누적이 유일한 원인이다.",
        "compounding_test": {
            "definition": "self 스트림에서 새 내용은 읽기로 만들어진다. 읽기가 죽으면 쓰는 내용이 "
                          "틀리고, 틀린 내용을 쓰면 읽기가 더 죽는다. 정오 필터가 이 고리를 끊는지 본다.",
            "self_stream_new_content_reject_rate_first10": {
                n: float(np.nanmean(agg[n]["_full_reject_new"][:10])) for n in in_loop},
            "self_stream_new_content_reject_rate_last10": {
                n: float(np.nanmean(agg[n]["_full_reject_new"][-10:])) for n in in_loop},
            "exo_stream_new_content_reject_rate_last10": {
                n: float(np.nanmean(agg[n]["_full_reject_new"][-10:])) for n in out_loop},
        },
        "read_lift_at_k150_in_loop_mean": float(np.mean(
            [agg[n]["_full_read_lift_heldout"][K] for n in in_loop])),
        "read_lift_at_k150_out_loop_mean": float(np.mean(
            [agg[n]["_full_read_lift_heldout"][K] for n in out_loop])),
        "capability_at_k150_in_loop_mean": float(np.mean(
            [agg[n]["_full_capability"][K] for n in in_loop])),
        "capability_at_k150_out_loop_mean": float(np.mean(
            [agg[n]["_full_capability"][K] for n in out_loop])),
    }
    # 이 실험에서 나온 되먹임 채널의 실제 서명 — 방향이 사전 예상과 다르다. 그대로 인쇄한다.
    def stop_block(n):
        a = agg[n]
        dth = np.asarray(a["_full_dtheta"], dtype=float)
        return {
            "acquisition_first10": float(np.nanmean(a["_full_free_new"][:10])),
            "acquisition_last10": float(np.nanmean(a["_full_free_new"][-10:])),
            "dtheta_first10": float(np.nanmean(dth[:10])),
            "dtheta_last10": float(np.nanmean(dth[-10:])),
            "dtheta_last10_over_first10": float(np.nanmean(dth[-10:]) / max(np.nanmean(dth[:10]), 1e-12)),
            "items_written_per_round_last10": float(np.nanmean(a["_full_n_written"][-10:])),
            "retention_Kstar_rel50": B1["cells"][n]["Kstar_rel50"] if n in B1["cells"] else None,
            "read_lift_heldout_k150": float(a["_full_read_lift_heldout"][K]),
            "reject_new_last10": float(np.nanmean(a["_full_reject_new"][-10:])),
        }
    B2["the_loop_stops_rather_than_the_forgetting_accelerates"] = {
        "what_was_expected": "되먹임이 붕괴를 **앞당긴다** — 즉 자기생성 팔에서 유지가 더 빨리 무너지고 "
                             "읽기 채널도 더 빨리 죽는다.",
        "what_happened": "반대 방향이다. 자기생성 팔은 읽기가 죽으면 **쓸 내용을 만들지 못해 루프가 "
                         "멈춘다** — 습득이 0으로 가고, 라운드당 ||Delta Theta||가 줄고, 그래서 간섭이 "
                         "줄어 cohort 유지와 잔여 읽기 능력이 **좋아 보인다.** 붕괴의 서명은 '더 빨리 "
                         "잊는다'가 아니라 '조용히 멈춘다'이며, 습득 축 말고는 모든 지표가 반대 방향을 "
                         "가리킨다.",
        "by_condition": {n: stop_block(n) for n in
                         [cond_name(*c) for c in reference + factorial]},
        "link_to_X4_PART_A": "X4 F3과 같은 구조다 — PART A에서 MLE 분산이면 생성 분포가 죽어 스텝이 "
                             "작아지고 위치 위험이 발산 대신 **포화**했다. 여기서는 생성 능력이 죽어 "
                             "쓰기가 작아지고 유지가 **좋아진다.** 두 경우 다 '곡선이 좋아 보이는 것'이 "
                             "루프가 멈춘 결과다.",
        "consequence_for_reading_the_table": "**스트림 출처 축의 K* 비교를 유지 이득으로 읽으면 안 된다.** "
                                             "self 팔의 긴 K*는 습득이 멈춘 대가다. 두 축을 함께 보지 "
                                             "않으면 정확히 거꾸로 결론이 난다.",
    }

    # ---------------- B3 선행 지표 세 후보
    def lead_block(name):
        a = agg[name]
        dep_ret = a["_full_access"]                      # 배포축 1: 이전 라운드 유지
        dep_acq = a["_smooth_free_new"]                  # 배포축 2: 내재화 성적(습득)
        cand = {
            # 후보 1: [2606.04703]이 제공한 배포 가능한 선행 지표 — 같은 항목 위의 문맥/무문맥 차
            "ctx_minus_free_new": (a["_full_ctx_minus_free_new"], "falling", "down"),
            # 후보 1b: 같은 양을 한 번도 쓰지 않은 사실 위에서 — 읽기 연산자 자체의 건강도
            "read_lift_heldout": (a["_full_read_lift_heldout"][1:], "falling", "down"),
            # 후보 2: running covariance의 조건수 [2605.11836 App. B.3.3] — 원 논문에서는 상승
            "cond_cov": (a["_full_cond_cov"][1:], "rising", "abs"),
            # 후보 3: 가중치 노름 증가비 [2601.11042 §4.2] — 원 논문에서는 상승
            "norm_ratio": (a["_full_norm_ratio"][1:], "rising", "abs"),
        }
        # 배포 지표는 둘 다 '높을수록 좋은' 양이므로 **하강** 규칙으로만 경보한다.
        a_ret = alarm_round(dep_ret, 0.75)
        a_acq = alarm_round(dep_acq, 0.75)
        out = {"alarm_deployment_retention_access": a_ret,
               "alarm_deployment_acquisition_free_new": a_acq,
               "acquisition_never_alarmed": bool(a_acq is None),
               "acquisition_never_alarmed_reason":
                   ("외생 스트림에서는 습득 성적이 K 끝까지 자기 첫 값의 75% 아래로 내려가지 않는다 "
                    "— 그 자체가 [2605.11836]의 결과(획득 축은 버틴다)와 같은 방향이다."
                    if a_acq is None else None)}
        for cn, (cv, src_dir, rule) in cand.items():
            v = np.asarray(cv, dtype=float)
            ak = alarm_round(v, 0.75) if rule == "down" else alarm_round_absmove(v, 0.25)
            obs, frac_up = observed_direction(v)
            sign_ok = (src_dir == obs)
            out[cn] = {
                "direction_in_source": src_dir,
                "direction_observed_here": obs,
                "frac_increasing_steps": frac_up,
                "sign_agrees_with_source": bool(sign_ok),
                "alarm_rule": ("첫 값의 75% 아래로 처음" if rule == "down"
                               else "첫 값 대비 비가 [0.75, 1.333] 밖으로 처음(방향 무관)"),
                "alarm_round": ak,
                "lead_vs_retention_rounds": (None if (ak is None or a_ret is None) else a_ret - ak),
                "lead_vs_acquisition_rounds": (None if (ak is None or a_acq is None) else a_acq - ak),
                "value_k1": float(v[0]), "value_k150": float(v[-1]),
                "ratio_k150_over_k1": (float(v[-1] / v[0]) if v[0] > 0 else None),
                "ckpt_k": [k for k in CKPT if 1 <= k <= K],
                "ckpt_mean": [float(v[k - 1]) for k in CKPT if 1 <= k <= K],
            }
        return out

    lead_names = [cond_name(*c) for c in reference + factorial]
    B3 = {
        "why": "설계 변경 2 — X4의 lift는 이 책 안의 양이라 외부와 대조되지 않는다. "
               "2606.04703이 제공한 배포 가능한 선행 지표는 「문맥을 준 성적 − 문맥 없는 성적」이다.",
        "alarm_rule": "배포 지표(유지·습득)는 높을수록 좋은 양이므로 **하강** 규칙 — 자기 첫 값의 "
                      "75% 아래로 처음 내려간 라운드. 원 논문에서 하강하는 후보(문맥차·읽기 lift)도 "
                      "같은 규칙. 원 논문에서 상승하는 후보(조건수·노름비)는 방향 무관 규칙 — 첫 값 "
                      "대비 비가 [0.75, 1.333] 밖으로 처음 나간 라운드. "
                      "lead > 0 이면 그 지표가 배포 지표보다 **먼저** 울렸다는 뜻.",
        "two_deployment_axes": "유지(retention) = cohort access, 습득(acquisition) = 그 라운드에 "
                               "내재화한 사실의 문맥 없는 성적. [2606.04703]의 배포 지표는 습득형이고, "
                               "유지 축은 두 논문 다 재지 않은 축(G-A)이다. **어느 축을 쓰느냐로 "
                               "선행/후행이 갈리므로 둘 다 인쇄한다.**",
        "sign_check": "각 후보에 대해 원 논문이 보고한 방향과 이 실행에서 관측된 방향을 함께 적는다. "
                      "부호가 다르면 그 후보는 이 최소 모형에서 **같은 양이 아니다** — 경보 라운드를 "
                      "비교하더라도 원 논문의 지표를 검증한 것이 아니다.",
        "smoothing": f"습득 지표는 라운드당 사실 1개라 잡음이 크다 — 뒤쪽 이동평균 폭 {SMOOTH_W}.",
        "by_condition": {n: lead_block(n) for n in lead_names},
    }
    # 요약: 각 후보가 몇 개 조건에서 실제로 선행했는가
    summ = {}
    for cn in ("ctx_minus_free_new", "read_lift_heldout", "cond_cov", "norm_ratio"):
        lr_ret = [B3["by_condition"][n][cn]["lead_vs_retention_rounds"] for n in lead_names]
        lr_acq = [B3["by_condition"][n][cn]["lead_vs_acquisition_rounds"] for n in lead_names]
        lr_ret = [v for v in lr_ret if v is not None]
        lr_acq = [v for v in lr_acq if v is not None]
        n_sign = int(sum(1 for n in lead_names if B3["by_condition"][n][cn]["sign_agrees_with_source"]))
        summ[cn] = {
            "direction_in_source": B3["by_condition"][lead_names[0]][cn]["direction_in_source"],
            "direction_observed_here": B3["by_condition"][lead_names[0]][cn]["direction_observed_here"],
            "n_conditions_with_matching_sign": n_sign,
            "n_conditions_total": len(lead_names),
            "n_conditions_scored_retention": len(lr_ret),
            "n_leading_retention": int(sum(1 for v in lr_ret if v > 0)),
            "median_lead_retention_rounds": (float(np.median(lr_ret)) if lr_ret else None),
            "n_conditions_scored_acquisition": len(lr_acq),
            "n_leading_acquisition": int(sum(1 for v in lr_acq if v > 0)),
            "median_lead_acquisition_rounds": (float(np.median(lr_acq)) if lr_acq else None),
        }
    B3["summary_across_conditions"] = summ
    B3["verdict"] = {
        "usable_here": [cn for cn in summ if summ[cn]["n_conditions_with_matching_sign"] > 0],
        "not_usable_here": [cn for cn in summ if summ[cn]["n_conditions_with_matching_sign"] == 0],
        "why_not_usable": "이 최소 모형에서 running covariance의 조건수는 rank가 차면서 **내려가고**, "
                          "가중치 노름비도 **내려간다**(쓰기가 Theta_0의 읽기 항을 깎기 때문). "
                          "원 논문들이 보고한 방향은 둘 다 상승이다. 부호가 반대이므로 이 두 후보는 "
                          "**이 모형에서 검증할 수 없다** — 재현 실패이지 원 논문에 대한 반증이 아니다.",
    }

    # ---------------- B4 초기 라운드 순서 반전 (설계 변경 3)
    early_w, late_w = list(range(1, 6)), list(range(K - 9, K + 1))
    names_fac = [cond_name(*c) for c in reference + factorial]
    e_score = {n: float(np.nanmean([agg[n]["_full_access"][i] for i in early_w])) for n in names_fac}
    l_score = {n: float(np.nanmean([agg[n]["_full_access"][i] for i in late_w])) for n in names_fac}
    order_e = sorted(names_fac, key=lambda n: -e_score[n])
    order_l = sorted(names_fac, key=lambda n: -l_score[n])
    n_inv = 0; pairs = 0
    for i in range(len(names_fac)):
        for j in range(i + 1, len(names_fac)):
            x, y = names_fac[i], names_fac[j]
            de, dl = e_score[x] - e_score[y], l_score[x] - l_score[y]
            if abs(de) > 1e-9 and abs(dl) > 1e-9:
                pairs += 1
                if de * dl < 0:
                    n_inv += 1
    B4 = {
        "why": "설계 변경 3 — 두 논문 모두 초기 라운드의 순서 반전을 보고했으므로 검사할 가설이다. "
               "원 X4에서는 access가 천장에 붙어 이 질문 자체가 정의되지 않았다.",
        "early_window_rounds": [early_w[0], early_w[-1]],
        "late_window_rounds": [late_w[0], late_w[-1]],
        "early_ranking_best_first": order_e,
        "late_ranking_best_first": order_l,
        "spearman_early_vs_late": spearman([e_score[n] for n in names_fac],
                                           [l_score[n] for n in names_fac]),
        "n_pairs_scored": pairs,
        "n_pairs_inverted": n_inv,
        "frac_pairs_inverted": (n_inv / pairs) if pairs else None,
        "rise_before_fall_by_condition": {
            n: shape_stats(agg[n]["_full_access"])["rise_before_fall"] for n in names_fac},
        "peak_round_by_condition": {
            n: shape_stats(agg[n]["_full_access"])["peak_round"] for n in names_fac},
        "note": "상승-후-절벽은 [2606.04703] 구성 C(WWQA +5.1 다음 -18.2)의 모양이다. "
                "여기서 재현되면 모양의 존재가 이송되고, 진폭은 이송되지 않는다.",
    }

    # ---------------- B5 G-C: 원본 앵커 대 정오 필터
    def acc150(n):
        return float(agg[n]["_full_access"][K])

    def ks(n):
        return B1["cells"][n]["Kstar_rel50"] if n in B1["cells"] else None

    B5 = {
        "gap": "G-C — ch29 판정 4의 근거가 「원본 앵커」와 「정오 필터」를 뭉개고 있다. 여기서 가른다.",
        "exo_stream": {
            "anchor_gold": {"Kstar_rel50": ks(cond_name("gold", "none", "exo")),
                            "access_k150": acc150(cond_name("gold", "none", "exo"))},
            "generated_with_filter_teacher": {
                "Kstar_rel50": ks(cond_name("teacher", "filter", "exo")),
                "access_k150": acc150(cond_name("teacher", "filter", "exo"))},
            "generated_no_filter_teacher": {
                "Kstar_rel50": ks(cond_name("teacher", "none", "exo")),
                "access_k150": acc150(cond_name("teacher", "none", "exo"))},
            "filter_recovers_frac_of_anchor_gap": None,
        },
        "self_stream": {
            "anchor_gold": {"Kstar_rel50": ks(cond_name("gold", "none", "self")),
                            "access_k150": acc150(cond_name("gold", "none", "self"))},
            "generated_with_filter_teacher": {
                "Kstar_rel50": ks(cond_name("teacher", "filter", "self")),
                "access_k150": acc150(cond_name("teacher", "filter", "self"))},
            "generated_no_filter_teacher": {
                "Kstar_rel50": ks(cond_name("teacher", "none", "self")),
                "access_k150": acc150(cond_name("teacher", "none", "self"))},
            "filter_recovers_frac_of_anchor_gap": None,
        },
        "replay_filter_bite_rate": {
            n: {"first10": float(np.nanmean(agg[n]["_full_reject_replay"][:10])),
                "last10": float(np.nanmean(agg[n]["_full_reject_replay"][-10:]))}
            for n in [cond_name(t, "filter", s) for t in ("teacher", "student")
                      for s in ("exo", "self")]},
        "new_content_filter_bite_rate": {
            n: {"first10": float(np.nanmean(agg[n]["_full_reject_new"][:10])),
                "last10": float(np.nanmean(agg[n]["_full_reject_new"][-10:]))}
            for n in [cond_name(t, "filter", s) for t in ("teacher", "student")
                      for s in ("exo", "self")]},
    }
    for st in ("exo", "self"):
        blk = B5[st + "_stream"]
        anc = blk["anchor_gold"]["Kstar_rel50"]
        nof = blk["generated_no_filter_teacher"]["Kstar_rel50"]
        wf = blk["generated_with_filter_teacher"]["Kstar_rel50"]
        gap = anc - nof
        blk["filter_recovers_frac_of_anchor_gap"] = (
            float((wf - nof) / gap) if gap != 0 else None)

    # ---------------- B6 G-A: 외생 스트림 + 수축
    B6 = {
        "gap": "G-A(주공백) — 외생 스트림 + 수축 있는 갱신 규칙에서 「이전 라운드에 쓴 것」의 유지. "
               "2605.11836은 그 축을 재지 않았고(F19) 2606.04703은 그 구성을 돌리지 않았다.",
        "contraction_forms": {
            "ridge": f"라운드 시작점으로의 근접항 mu={cfg['prox_mu']} — [2605.11836]의 ridge 대응물",
            "subspace": f"Theta_0의 상위 {cfg['revive_rank']}개 특이방향에서 라운드 순갱신을 제거 "
                        "— [2601.11042] REVIVE의 대응물",
        },
        "cells": {},
    }
    for s in ("exo", "self"):
        for c in ("none", "ridge", "subspace"):
            n = cond_name("gold", "none", s, OMEGA, c)
            a = agg[n]
            ksr, cen = kstar(a["_full_access"] / max(a["_full_access"][0], 1e-9), 0.5)
            per_seed = []
            for x in results[n]["runs"]:
                v = np.asarray(x["access"], dtype=float)
                kk, _ = kstar(v / max(v[0], 1e-9), 0.5)
                per_seed.append(kk)
            B6["cells"][n] = {
                "Kstar_rel50": ksr, "censored": cen,
                "Kstar_rel50_per_seed": per_seed,
                "Kstar_rel50_std_across_seeds": float(np.std(per_seed, ddof=0)),
                "access_k50": float(a["_full_access"][50]),
                "access_k150": float(a["_full_access"][K]),
                "capability_k150": float(a["_full_capability"][K]),
                "read_lift_heldout_k150": float(a["_full_read_lift_heldout"][K]),
                "free_new_last10": float(np.nanmean(a["_full_free_new"][-10:])),
                "drift_k150": float(a["_full_drift"][K]),
                "dtheta_mean_per_round": float(np.nanmean(a["_full_dtheta"])),
            }
    B6["retention_gain_rounds_exo"] = {
        c: B6["cells"][cond_name("gold", "none", "exo", OMEGA, c)]["Kstar_rel50"]
           - B6["cells"][cond_name("gold", "none", "exo", OMEGA, "none")]["Kstar_rel50"]
        for c in ("ridge", "subspace")}
    B6["retention_gain_rounds_self"] = {
        c: B6["cells"][cond_name("gold", "none", "self", OMEGA, c)]["Kstar_rel50"]
           - B6["cells"][cond_name("gold", "none", "self", OMEGA, "none")]["Kstar_rel50"]
        for c in ("ridge", "subspace")}
    B6["acquisition_cost_of_contraction_exo"] = {
        c: (B6["cells"][cond_name("gold", "none", "exo", OMEGA, c)]["free_new_last10"]
            - B6["cells"][cond_name("gold", "none", "exo", OMEGA, "none")]["free_new_last10"])
        for c in ("ridge", "subspace")}

    # ---------------- B7 2차 모멘트와 모드 붕괴 (설계 변경 5)
    B7 = {
        "why": "설계 변경 5 — [2606.04703]의 붕괴 형태가 행동 모드 붕괴(premature answer 63.82%)이므로 "
               "사실 주입 시뮬에서도 생성 분포의 퍼짐을 라운드 함수로 찍어야 F3이 한 언어로 말해진다.",
        "two_measurements": {
            "model_level_gen_*": f"고정 프로브 {cfg['n_genprobe']}개에 대한 모형 출력의 분포 — "
                                 "조건 간 비교 가능한 모형 내재 양. 참여비(PR)·평균 쌍코사인·최빈 코드북 점유율.",
            "batch_level_written_*": "그 라운드에 실제로 쓴 서로 다른 타깃들의 분포 — 학습 신호의 퍼짐.",
        },
        "by_condition": {},
    }
    for n in names_fac:
        a = agg[n]
        B7["by_condition"][n] = {
            "gen_pr_ckpt": a["gen_pr"], "gen_cos_ckpt": a["gen_cos"],
            "gen_mode_share_ckpt": a["gen_mode_share"],
            "written_cos_smooth_ckpt": a["written_cos_smooth"],
            "written_mode_share_smooth_ckpt": a["written_mode_share_smooth"],
            "gen_pr_k0": float(a["_full_gen_pr"][0]),
            "gen_pr_k150": float(a["_full_gen_pr"][K]),
            "gen_pr_ratio_k150_over_k0": float(a["_full_gen_pr"][K] / a["_full_gen_pr"][0]),
            "gen_mode_share_k0": float(a["_full_gen_mode_share"][0]),
            "gen_mode_share_k150": float(a["_full_gen_mode_share"][K]),
        }
    B7["second_moment_collapse_ordering"] = {
        "definition": "gen_pr(참여비)가 자기 k=0 값의 50% 아래로 처음 내려간 라운드 대 "
                      "cohort access가 자기 k=0의 50% 아래로 처음 내려간 라운드.",
        "by_condition": {
            n: {"pr_half_round": first_below(
                    agg[n]["_full_gen_pr"] / max(agg[n]["_full_gen_pr"][0], 1e-12), 0.5),
                "access_half_round": first_below(
                    agg[n]["_full_access"] / max(agg[n]["_full_access"][0], 1e-12), 0.5)}
            for n in names_fac},
    }

    # ---------------- B8 라운드의 조작적 정의 (B1 지적)
    B8 = {
        "why": "라운드당 크기를 인쇄하지 않으면 K*가 200인지 20,000인지 비교되지 않는다 "
               "(TIER1-ADJUDICATION B1).",
        "items_per_round": cfg["N_total"],
        "grad_steps_per_round": cfg["steps"],
        "lr": cfg["lr"],
        "mac_proxy_per_round": int(cfg["steps"] * cfg["N_total"] * 2 * cfg["d"] * cfg["d"]),
        "mac_proxy_note": "이 장난감 안의 단위다. 실제 B_s(토큰·FLOPs·wall-clock)와 같지 않다.",
        "theta0_fro_note": "아래 dtheta는 전부 ||Theta_0||_F로 나눈 값이 아니라 절대 Frobenius다. "
                           "drift는 비율이다.",
        "by_condition": {},
        "external_yardsticks_for_comparison": {
            "2605.11836_one_round": "n_t=100 편집 위 닫힌 형태 ridge 1스텝, eta_Theta=1e-6, "
                                    "편집 대상 MLP 행렬 4-17개. K=200/5,000/20,000.",
            "2606.04703_one_round": "batch 128 · 5 epoch full SFT (8xA800). K=3.",
            "why_it_matters": "두 눈금은 라운드당 ||Delta Theta||가 자릿수로 다르다. "
                              "K* 200 대 20,000을 같은 축에 놓으려면 이 나눗셈이 선행해야 한다. "
                              "두 논문 다 ||Delta Theta||를 인쇄하지 않았다 — 그래서 그 나눗셈은 "
                              "현재 corpus로 **불가능하다**.",
        },
    }
    th0 = float(np.mean([World(cfg, sd).theta0_fro for sd in SEEDS]))
    B8["theta0_fro_mean_across_seeds"] = th0
    for n in [cond_name(*c) for c in reference + factorial + contraction_arms]:
        a = agg[n]
        dth = np.asarray(a["_full_dtheta"], dtype=float)
        B8["by_condition"][n] = {
            "dtheta_fro_mean_per_round": float(np.nanmean(dth)),
            "dtheta_fro_first_round": float(dth[0]),
            "dtheta_fro_last_round": float(dth[-1]),
            "dtheta_over_theta0_fro_mean": float(np.nanmean(dth) / th0),
            "cumulative_path_len": float(np.nansum(dth)),
            "net_drift_fro": float(a["_full_drift"][K] * th0),
            "path_over_net_ratio": float(np.nansum(dth) / max(a["_full_drift"][K] * th0, 1e-12)),
            "n_written_items_per_round_mean": float(np.nanmean(a["_full_n_written"])),
            "n_written_items_per_round_last10": float(np.nanmean(a["_full_n_written"][-10:])),
        }

    # ---------------- B9 omega 부수쓸이
    B9 = {
        "why": "X4의 omega 축(F6)이 새 설계에서도 같은 방향인지 확인. 요인설계는 omega=0.5 고정이므로 "
               "두 대표 팔에서만 쓴다.",
        "cells": {},
    }
    for base in (("gold", "none", "exo"), ("teacher", "filter", "self")):
        for w in [OMEGA_SWEEP[0], OMEGA, OMEGA_SWEEP[1]]:
            n = cond_name(*base, w, "none")
            if n not in agg:
                continue
            a = agg[n]
            ksr, cen = kstar(a["_full_access"] / max(a["_full_access"][0], 1e-9), 0.5)
            B9["cells"][n] = {
                "omega": w, "Kstar_rel50": ksr, "censored": cen,
                "access_k150": float(a["_full_access"][K]),
                "free_new_last10": float(np.nanmean(a["_full_free_new"][-10:])),
                "capability_k150": float(a["_full_capability"][K]),
            }
    B9["monotone_in_omega"] = {}
    for base in (("gold", "none", "exo"), ("teacher", "filter", "self")):
        seq = [B9["cells"][cond_name(*base, w, "none")]["Kstar_rel50"]
               for w in [OMEGA_SWEEP[0], OMEGA, OMEGA_SWEEP[1]]]
        B9["monotone_in_omega"]["|".join(base)] = {
            "Kstar_by_omega": seq,
            "nondecreasing": bool(all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))),
        }

    # ---------------- C 판정
    C_ = {
        "C0_model_free_shape_statistics": {
            "why": "네 후보의 SSE 차이가 작을 수 있으므로 판정은 모형 선택이 아니라 모형 무관 통계로 한다.",
            "access": {n: shape_stats(agg[n]["_full_access"]) for n in names_fac},
            "read_lift_heldout": {n: shape_stats(np.maximum(agg[n]["_full_read_lift_heldout"], 0.0))
                                  for n in names_fac},
        },
        "C1_shape_fits_access": {n: fit_shapes(agg[n]["_full_access"][1:]) for n in names_fac},
        "C2_cross_seed_stability": {},
        "C3_rho_is_not_a_rate": {},
    }
    # 시드 간 순서 안정성 — 이송 가능한 양은 순서다
    pair_list = [(cond_name("gold", "none", "exo"), cond_name("teacher", "none", "exo")),
                 (cond_name("gold", "none", "self"), cond_name("teacher", "none", "self")),
                 (cond_name("teacher", "none", "exo"), cond_name("student", "none", "exo")),
                 (cond_name("teacher", "none", "self"), cond_name("student", "none", "self")),
                 (cond_name("teacher", "filter", "self"), cond_name("teacher", "none", "self")),
                 (cond_name("teacher", "filter", "exo"), cond_name("teacher", "none", "exo")),
                 (cond_name("gold", "none", "exo", OMEGA, "ridge"), cond_name("gold", "none", "exo")),
                 (cond_name("gold", "none", "exo", OMEGA, "subspace"), cond_name("gold", "none", "exo"))]
    for (x, y) in pair_list:
        sx = [np.asarray(rr["access"], dtype=float) for rr in results[x]["runs"]]
        sy = [np.asarray(rr["access"], dtype=float) for rr in results[y]["runs"]]
        def _ks(v):
            kk, _ = kstar(v / max(v[0], 1e-9), 0.5)
            return kk
        dif = [_ks(a) - _ks(b) for a, b in zip(sx, sy)]
        C_["C2_cross_seed_stability"][f"{x}  VS  {y}"] = {
            "Kstar_diff_per_seed": dif,
            "mean": float(np.mean(dif)), "std": float(np.std(dif, ddof=0)),
            "sign_consistent_across_all_seeds": bool(
                all(d > 0 for d in dif) or all(d < 0 for d in dif) or all(d == 0 for d in dif)),
        }
    for n in [cond_name("gold", "none", "exo"), cond_name("teacher", "none", "self"),
              cond_name("teacher", "filter", "self")]:
        a = np.asarray(agg[n]["_full_access"], dtype=float)
        rho = -np.diff(a) * 100.0
        C_["C3_rho_is_not_a_rate"][n] = {
            "rho_pp_windows": {"k1_10": float(np.nanmean(rho[:10])),
                               "k11_30": float(np.nanmean(rho[10:30])),
                               "k31_60": float(np.nanmean(rho[30:60])),
                               "k61_100": float(np.nanmean(rho[60:100])),
                               "k101_150": float(np.nanmean(rho[100:150]))},
            "rho_max_over_first10": shape_stats(a)["rho_max_over_first10"],
        }

    # =========================================================================
    # 조립
    # =========================================================================
    doc = {
        "meta": {
            "id": "X4prime-feedback-decay",
            "title": "저장을 읽는 연산자가 자기 자신일 때 — 교사 되먹임 채널과 그 선행 경보",
            "grade": "EXPLORATION-GRADE / MECHANISM-TRANSFER (LLM 실험 아님)",
            "supersedes": "experiments/stc/X4-consolidation-decay (PART B만). X4를 지우지 않는다.",
            "relation_to_X4": {
                "what_survives_untouched": [
                    "X4 PART A 전체(가우시안 위치-척도 해석 + 몬테카를로) — 이번 열 편의 어느 것도 "
                    "건드리지 않는다.",
                    "F1 — 1/i 가중은 선형회귀의 성질이 아니라 데이터 스케줄의 성질이다.",
                    "F2 — [model-collapse]의 정리가 볼 수 없는 좌표는 2차 모멘트다.",
                    "F3 — 평평한 손실 곡선은 안정의 증거가 아니다.",
                    "F9의 마지막 문장 — 'K>100에서 원본 보존 replay와 함께 K*를 측정한 실험이 "
                    "corpus에 없다'는 여전히 참이다(2605.11836은 K=20,000이나 replay가 없고 유지를 "
                    "재지 않으며, 2606.04703은 K=3이고 E-경로다).",
                ],
                "what_is_replaced": [
                    "X4 PART B의 설계 전체. omega x replay-source 2축 쓸이가 "
                    "2x2x2{타깃, 정오 필터, 스트림 출처} + 수축 팔로 대체된다.",
                    "X4 B5의 breadth 2x2(저장 방식 x 이후 쓰기 방식)는 이번 설계에 없다 — "
                    "X4의 X1(방향 불일치)은 그대로 X4에 남는다.",
                    "작동점. X4의 access는 k<=25에서 1.0이었다. 여기서는 k=0부터 천장 아래다.",
                ],
                "what_is_new": [
                    "교사 되먹임 채널(G-B) — 파라미터가 저장을 읽어 학습 타깃을 만드는 연산자다.",
                    "외부와 같은 단위의 선행 지표 — 「문맥 준 성적 − 문맥 없는 성적」.",
                    "수축 두 형태(ridge / subspace)와 외생 스트림에서의 유지(G-A).",
                    "시뮬 파트의 2차 모멘트·모드 점유율.",
                    "라운드의 조작적 정의(라운드당 ||Delta Theta||, 누적 경로/순이동).",
                ],
                "X4_findings_that_this_run_re_tests": {
                    "F4 (과제 지표가 늦게 움직인다)": "천장 밖 작동점에서 다시 검사 — B3·C0",
                    "F5 (비율이 아니라 crossover를 보고하라)": "K*_rel50/K*_abs50 — B1·C3",
                    "F6 (replay는 보존된 원본에 앵커되어야 한다)": "G-C 분해로 재검사 — B5",
                },
            },
            "deterministic": True,
            "randomness": f"seeds={SEEDS}, 전부 np.random.default_rng. World와 라운드 표집 모두 "
                          "시드에서 파생. BLAS 스레드 1. 보고값 6자리. wall-clock 미포함.",
            "no_gpu": "이 host에 NVIDIA GPU 없음. numpy만 사용(torch/transformers 미사용).",
            "reproduction": {
                "command": "cd experiments/stc/X4prime-feedback-decay && "
                           "PYTHONHASHSEED=0 ../../../.venv/bin/python run.py",
                "determinism_check": "같은 명령을 2회 실행하고 result.json의 sha256이 같은지 확인한다. "
                                     "다르면 실험이 아니다(REPRODUCE.md §3). 이 파일은 wall-clock을 "
                                     "담지 않으므로 bit-identical이 기대값이다.",
                "runtime_order": "단일 CPU 스레드에서 1분 미만.",
                "deps": "numpy만. scipy·matplotlib·torch 불필요.",
            },
            "why_this_model": {
                "reading_operator_in_the_parameters":
                    "Theta_0 = Theta_bg + gamma S^T 는 저장 항목 e_f = S v_f 에 대해 일반적으로 "
                    "작동하는 한 장의 읽기 행렬이다 — 사실별 암기가 아니므로 한 번도 쓰지 않은 "
                    "사실에도 작동하고, 그 능력의 열화가 'experience-use ability' 손상의 대응물이다.",
                "why_the_read_channel_dies":
                    "쓰기 갱신의 행공간은 키 방향이고 키와 저장 방향의 겹침은 cos ~ d^-1/2 이다. "
                    "감쇠 상수를 손으로 넣지 않았다 — 간섭의 누적이 유일한 원인이다.",
                "what_it_cannot_model":
                    "합성·함의, experience granularity(추상화 수준), injection alignment"
                    "(상태 조건부 선택). 선형 연상기억에 대응물이 없다.",
            },
            "budget_matched": f"omega·필터·수축을 바꿔도 라운드당 항목 수 {cfg['N_total']}와 "
                              f"경사 스텝 {cfg['steps']}를 고정한다(필터가 버리면 수락분에서 복원추출로 "
                              "채운다). [2606.04703 App. A.3] 'steps are matched, not tokens'.",
            "round_definition": {
                "items_per_round": cfg["N_total"],
                "grad_steps_per_round": cfg["steps"],
                "measured_dtheta_fro_per_round": "→ B8_round_yardstick",
                "why": "이것 없이는 K*가 200인지 20,000인지 비교되지 않는다(→ B1 지적).",
            },
            "sim_config": cfg,
            "seeds": SEEDS,
            "omega_factorial": OMEGA,
            "chance_level": chance,
            "design_changes_applied": {
                "1_teacher_feedback_channel": "B2 — 최우선. 라운드 k의 파라미터가 정책이자 "
                                              "E^(k)를 읽어 타깃을 만드는 연산자.",
                "2_leading_indicator_matched_to_outside": "B3 — 「문맥 준 성적 − 문맥 없는 성적」 "
                                                          "+ 조건수 + 노름 증가비.",
                "3_off_the_ceiling": "B0·B4 — access k=0에서 천장 아래, 초기 순서 반전 검사.",
                "4_2x2x2": "B1·B5·B6 — {타깃}x{정오 필터}x{스트림 출처}.",
                "5_second_moment_in_the_sim": "B7 — 생성 분포의 퍼짐과 최빈 모드 점유율.",
                "6_out_of_scope_declared": "meta.honest_limits_out_of_scope.",
            },
            "sourced_forms": {
                "2606_04703_collapse_in_3_rounds":
                    "K=3에 progressive capability collapse; 구성 C는 상승 후 절벽 "
                    "(WWQA +5.1 다음 -18.2) [Table 4].",
                "2606_04703_leading_indicator":
                    "문맥 성적이 내재화 성적보다 한 라운드 먼저 무너진다 — 구성 C: 문맥 28.1→14.1인 "
                    "라운드에 내재화는 25.9→31.0으로 오르고, 다음 라운드 12.8 [Table 4; §5.2.3].",
                "2606_04703_teacher_feedback":
                    "'the model may provide weaker experience-conditioned supervision and "
                    "destabilize the model-experience loop' [§5.2.3, L669-671].",
                "2606_04703_mode_collapse":
                    "3라운드 모델의 premature-answer 비율 global 63.82% / step-wise 0% [Table 2].",
                "2606_04703_injection_effect":
                    "1라운드 global→step-wise: WebWalkerQA +8.0, GAIA +5.9, BrowseComp-ZH +0.7 "
                    "[Table 1]. **이 축은 이 최소 모형의 범위 밖이다.**",
                "2605_11836_positive_cumulative":
                    "'a counter-intuitive positive cumulative effect where early edits can promote "
                    "the success of future edits' [Abstract]. 외생 스트림 + LN/ridge 수축.",
                "2605_11836_condition_number":
                    "'the condition number rises temporarily around 400K edits, yet STABLEEDIT "
                    "remains stable while ULTRAEDIT degrades more noticeably at this point' "
                    "[App. B.3.3].",
                "2605_11836_no_retention_axis":
                    "이전 라운드 항목의 유지를 한 번도 재지 않는다(F19). GLUE는 30K에서 끊긴다.",
                "2601_11042_norm_growth":
                    "'parameter matrix norms tend to exhibit abnormal growth during sequential "
                    "editing' [§4.2, p.6].",
                "2601_11042_subspace_protection":
                    "REVIVE: Delta W_safe = (I - U_k U_k^T) Delta W (I - V_k V_k^T); 10,000 편집에서 "
                    "평균 86.34% 유지. 무보호 MEMIT/RECT는 3,000, AlphaEdit는 8,000에 붕괴 [§4.2].",
            },
            "absent_from_corpus": {
                "teacher_feedback_channel_as_a_measured_axis":
                    "저장을 읽는 능력의 열화를 라운드 함수로 정량 보고한 논문 0편. 2606.04703이 "
                    "기제로 서술하고 문맥/내재화 두 열을 인쇄하지만 그것을 경보 지표로 다루지 않는다.",
                "retention_under_exogenous_stream_with_contraction":
                    "G-A. 2605.11836은 축을 안 재고 2606.04703은 구성을 안 돌린다. **여전히 공백이다** "
                    "— 이 실험은 장난감 안에서만 답한다.",
                "delta_theta_per_round":
                    "라운드당 ||Delta Theta||를 인쇄한 논문 0편. 그래서 K*=200과 K*=20,000을 "
                    "같은 축에 놓는 나눗셈이 현재 corpus로 불가능하다.",
                "second_moment_of_generator_by_round":
                    "생성 분포의 퍼짐을 라운드 함수로 정량 보고한 논문 여전히 0편.",
                "half_life_or_rho_as_a_rate":
                    "라운드당 열화율을 비율로 보고한 논문 여전히 0편.",
            },
            "caveat_tags": {
                "NOT-AN-LLM-EXPERIMENT": "선형 연상기억 + 심어 넣은 읽기 연산자다. 절대치 이송 불가. "
                                         "부호·함수형·조건 간 순서·선행/후행의 순서만 이송한다.",
                "READ-OPERATOR-IS-PLANTED": "실제 LLM의 in-context 능력은 학습으로 생겨난 회로다. "
                                            "여기서는 Theta_0에 한 장의 행렬로 심었다. 그 능력이 "
                                            "**어떻게** 죽는지의 기하는 다를 수 있다.",
                "DECAY-IS-GEOMETRIC-INTERFERENCE": "읽기 채널의 열화율은 d^-1/2 겹침의 누적이다. "
                                                   "실제 모형의 열화율과 무관하다 — 존재와 부호만 이송.",
                "FILTER-POWER-DEPENDS-ON-THE-QUERY": "정오 필터의 위력은 생성기를 **어떤 입력에서** "
                                                     "평가하느냐에 달렸다. 여기서 replay 타깃은 이미 "
                                                     "쓴 키에서 생성되므로 필터가 거의 물지 않는다. "
                                                     "이 구조적 사실 자체가 결과다(B5).",
                "CONTRACTION-IS-A-TOY-ANALOGUE": "ridge 근접항과 상위 특이부분공간 제거는 LN "
                                                 "(centering+whitening)과 REVIVE의 **형태**만 옮긴 것이다. "
                                                 "mu와 rank 값은 이송 불가.",
                "HORIZON-150-NOT-20000": "K=150이다. 2605.11836의 20,000 라운드 결론을 이 실험이 "
                                         "확인하거나 반증하지 않는다.",
                "KSTAR-CENSORED": f"일부 조건은 K={K} 안에서 임계를 넘지 않는다. 그 K*는 하한이다.",
                "SMOOTHING-IS-A-CHOICE": f"습득 지표의 이동평균 폭 {SMOOTH_W}는 선택이다. 경보 라운드는 "
                                         "이 폭에 의존한다 — 선행/후행의 **부호**만 읽어야 한다.",
                "ALARM-THRESHOLD-IS-A-CHOICE": "하강 75% 규칙과 방향 무관 25% 규칙은 둘 다 선택이다. "
                                               "조건 간 비교는 같은 규칙 아래에서만 유효하고, 절대 "
                                               "라운드 수는 이송 불가.",
                "TWO-OF-THREE-INDICATORS-HAVE-THE-WRONG-SIGN-HERE":
                    "조건수와 노름비는 원 논문에서 상승하는 양인데 이 모형에서는 하강한다. 그 둘의 "
                    "경보 라운드는 인쇄하되 **원 논문의 지표를 검증한 것으로 읽으면 안 된다**(→ X6).",
                "SURVIVAL-HALF-ONLY": "합성·함의는 여전히 없다. survival 절반만 다룬다.",
                "NO-CAPACITY-CEILING": "선형 사상에는 bits/param 상한이 없다. 여기서 관측된 붕괴의 "
                                       "원인이 LLM에서와 같다고 말할 수 없다.",
            },
            "honest_limits_out_of_scope": {
                "experience_granularity": {
                    "what": "E^(k)의 항목이 궤적 국소 세부인가 재사용 가능한 전략 진술인가 "
                            "[2606.04703 §5.1].",
                    "why_out_of_scope": "선형 연상기억에 '추상화 수준'의 대응물이 없다. 흉내 내면 "
                                        "자유 변수를 하나 더 만들 뿐이다.",
                    "but_note": "instance-level 풀은 74.4%가 구체적 URL·도메인, 93.9%가 개체 특정 "
                                "문자열이고 principle-level 풀은 84.0%가 재사용 가능한 전략 진술이다 "
                                "(instance-level 3.7%). **이 축이 실제 LLM에서 가장 큰 효과를 낸 둘 중 "
                                "하나다.**",
                },
                "injection_alignment": {
                    "what": "ret(E^(k), h_j)가 항등사상(global)인가 상태 조건부 선택(step-wise)인가.",
                    "why_out_of_scope": "이 모형의 read는 항상 해당 사실의 저장 항목을 정확히 준다 — "
                                        "검색 오정렬이라는 자유도가 없다.",
                    "but_note": "1라운드 효과가 WebWalkerQA +8.0 / GAIA +5.9이고, 3라운드에서 global은 "
                                "premature-answer 63.82%로 퇴화하는데 step-wise는 0%다 [Table 1, Table 2]. "
                                "**실제 LLM에서 가장 큰 효과를 낸 축이며 이 실험은 그것을 다루지 않는다.**",
                },
                "composition_and_entailment": "선형 사상에 기제가 없다. X4와 동일하게 범위 밖.",
                "capacity_ceiling": "bits/param 상한이 없다. X4와 동일하게 범위 밖.",
            },
        },
        "A_link_to_X4_partA": {
            "status": "재계산하지 않는다. X4의 PART A가 정본이다.",
            "path": "experiments/stc/X4-consolidation-decay/result.json :: A_analytic_gaussian",
            "why_it_still_stands": "해석 파트는 데이터 스케줄(replace / replace-multiple / accumulate)의 "
                                   "성질을 정확히(exact) 유도한 것이고, 이번 열 편은 그 유도를 건드리지 "
                                   "않는다. 다만 §5.2의 조건절이 그 위에 얹힌다 — 그 세 스케줄은 "
                                   "'스트림의 출처' 축의 세 점이며, 나머지 두 축(갱신 규칙, 재는 축)은 "
                                   "PART A가 고정해 둔 것이다.",
            "what_this_run_adds_to_it": "PART A는 2차 모멘트를 해석적으로만 열었다. 설계 변경 5에 따라 "
                                        "시뮬 파트에서도 같은 좌표를 인쇄한다(→ B7).",
        },
        "B0_operating_point": B0,
        "B1_factorial_2x2x2": B1,
        "B2_teacher_feedback_channel_G_B": B2,
        "B3_leading_indicators": B3,
        "B4_early_round_order_reversal": B4,
        "B5_anchor_vs_filter_G_C": B5,
        "B6_exogenous_plus_contraction_G_A": B6,
        "B7_second_moment_and_mode_collapse": B7,
        "B8_round_yardstick": B8,
        "B9_omega_subsweep": B9,
        "C_verdict": C_,
    }

    # ---------------- findings (본문 단정문 대 하한/방향성)
    ks_gold_exo = B1["cells"][cond_name("gold", "none", "exo")]["Kstar_rel50"]
    ks_t_self = B1["cells"][cond_name("teacher", "none", "self")]["Kstar_rel50"]
    ks_tf_self = B1["cells"][cond_name("teacher", "filter", "self")]["Kstar_rel50"]
    ks_s_self = B1["cells"][cond_name("student", "none", "self")]["Kstar_rel50"]
    ks_t_exo = B1["cells"][cond_name("teacher", "none", "exo")]["Kstar_rel50"]
    ks_s_exo = B1["cells"][cond_name("student", "none", "exo")]["Kstar_rel50"]

    stop = B2["the_loop_stops_rather_than_the_forgetting_accelerates"]["by_condition"]
    n_gs = cond_name("gold", "none", "self")
    n_ts = cond_name("teacher", "none", "self")
    n_ge = cond_name("gold", "none", "exo")
    lead_ho = summ["read_lift_heldout"]
    exo_never = [n for n in lead_names
                 if B3["by_condition"][n]["acquisition_never_alarmed"]]
    pr_ord = B7["second_moment_collapse_ordering"]["by_condition"]
    pr_lead = [n for n in names_fac
               if pr_ord[n]["pr_half_round"] is not None
               and pr_ord[n]["access_half_round"] is not None
               and pr_ord[n]["pr_half_round"] < pr_ord[n]["access_half_round"]]

    def seedc(x, y):
        """C2에서 시드 부호 일치 여부를 문장으로 — 단일 시드 결론 금지 규율의 집행."""
        blk = C_["C2_cross_seed_stability"][f"{x}  VS  {y}"]
        n_pos = sum(1 for v in blk["Kstar_diff_per_seed"] if v > 0)
        return (f"시드 5개 중 {n_pos}개에서 부호 유지"
                f"{', 전 시드 일치' if blk['sign_consistent_across_all_seeds'] else ', **전 시드 일치 아님**'}"
                f" (per-seed {blk['Kstar_diff_per_seed']}, sd {blk['std']:.1f})")

    doc["findings"] = {
        "LOAD_BEARING_본문_단정문으로_쓸_것": {
            "G1_a_third_channel_exists_and_it_dies_on_its_own":
                "저장을 읽어 감독을 만드는 연산자 자체가 라운드와 함께 죽는다. 감쇠 상수를 넣지 "
                "않았다 — 쓰기 갱신의 행공간(키 방향)과 저장 방향의 겹침 cos ~ d^-1/2 이 누적된 "
                "결과다. 모양은 **오래 평평하다 절벽**이다(전 조건에서 k=20까지 lift >= 0.99, 이후 "
                "급락). 즉 **간섭도 분포 모멘트도 아닌 제3의 채널이 최소 모형에서 자동으로 나타난다.**",
            "G2_on_the_exogenous_stream_this_channel_dies_completely_invisibly":
                "외생 스트림에서는 배포 지표(습득)가 K 끝까지 자기 첫 값의 75% 아래로 **한 번도 "
                f"내려가지 않는다**({len(exo_never)}개 조건에서 경보 없음; 마지막 10라운드 습득 "
                f"{stop[n_ge]['acquisition_last10']:.3f}). 같은 구간에 읽기 채널은 "
                f"경보(라운드 {B3['by_condition'][n_ge]['read_lift_heldout']['alarm_round']})를 지나 "
                f"k={K}에서 {stop[n_ge]['read_lift_heldout_k150']:.4f}로 사실상 소멸한다. "
                "**한 능력이 통째로 죽는 동안 배포 지표는 한 번도 움직이지 않는다.** "
                "[2605.11836]이 '20,000 라운드에도 안정'이라고 읽은 축이 바로 이 배포 지표 쪽이고, "
                "그 논문은 읽기·유지 축을 재지 않았다(F19). → 이것이 X4 F4의 강화판이며, "
                "천장 아티팩트가 아닌 작동점에서 얻어졌다.",
            "G3_the_collapse_signature_is_the_loop_stopping_not_faster_forgetting":
                "자기생성 스트림에서 읽기가 죽으면 **쓸 내용을 만들지 못해 루프가 멈춘다.** "
                f"새 내용 기각률이 0 -> {stop[n_ts]['reject_new_last10']:.2f}로 오르고, 습득이 "
                f"{stop[n_ts]['acquisition_first10']:.3f} -> {stop[n_ts]['acquisition_last10']:.3f}로 "
                f"떨어지며, 라운드당 ||Delta Theta||가 첫 10라운드 대비 "
                f"{stop[n_ts]['dtheta_last10_over_first10']:.2f}배로 줄어든다. 그 결과 cohort **유지가 "
                f"좋아지고**(self 팔 K* 주효과 {main_eff['stream_self_minus_exo_rounds']:+.1f} 라운드) "
                "잔여 읽기 능력도 exo 팔보다 커 보인다"
                f"(k={K} 읽기 lift 루프 안 {B2['read_lift_at_k150_in_loop_mean']:.4f} 대 "
                f"루프 밖 {B2['read_lift_at_k150_out_loop_mean']:.4f}). "
                "**습득 축을 빼면 모든 지표가 반대 방향을 가리킨다.** 이것은 X4 F3과 같은 구조다 — "
                "PART A에서 생성 분산이 죽어 스텝이 작아지고 위험이 발산 대신 포화했던 것과 같은 "
                "일이 여기서는 생성 **능력**에서 일어난다. → **스트림 출처 축의 K* 비교를 유지 "
                "이득으로 읽으면 정확히 거꾸로 결론이 난다.**",
            "G4_off_policy_beats_on_policy_in_the_same_direction_as_the_paper":
                "문맥 조건화 타깃(teacher, off-policy)이 문맥 없는 자기 출력(student, on-policy)보다 "
                f"네 개 (필터 x 스트림) 칸 전부에서 오래 버틴다. K*_rel50 주효과 "
                f"= {main_eff['target_teacher_minus_student_rounds']:+.1f} 라운드. "
                "[2606.04703]의 'off-policy context-distillation provides a substantially more "
                "stable training signal than on-policy'와 **부호가 같다.** 의미론이 전혀 없는 선형 "
                "모형에서도 나오므로, 그 이득의 일부는 과제 의미가 아니라 **타깃을 만드는 입력에 "
                "저장이 들어 있느냐**에서 온다. "
                f"단 시드 안정성이 축마다 다르다 — 외생 스트림: {seedc(cond_name('teacher','none','exo'), cond_name('student','none','exo'))}; "
                f"자기생성 스트림: {seedc(cond_name('teacher','none','self'), cond_name('student','none','self'))}. "
                "**자기생성 스트림에서는 전 시드 일치가 아니므로 그 칸의 주장은 평균 수준으로만 쓴다.**",
            "G5_the_anchor_dominates_the_filter_in_both_regimes":
                "G-C 판정. 외생 스트림에서 원본 앵커 K*="
                f"{B5['exo_stream']['anchor_gold']['Kstar_rel50']} 대 생성 타깃 "
                f"{B5['exo_stream']['generated_no_filter_teacher']['Kstar_rel50']}이고 정오 필터가 "
                f"메우는 몫은 {B5['exo_stream']['filter_recovers_frac_of_anchor_gap']:.3f}다. "
                f"자기생성 스트림에서도 앵커 {B5['self_stream']['anchor_gold']['Kstar_rel50']} 대 "
                f"{B5['self_stream']['generated_no_filter_teacher']['Kstar_rel50']}이고 필터가 메우는 "
                f"몫은 {B5['self_stream']['filter_recovers_frac_of_anchor_gap']:.3f}에 그친다. "
                "→ **ch29 판정 4의 근거는 「원본 앵커」쪽이다.** 필터는 앵커의 대체물이 아니다. "
                "다만 필터의 효과는 자기생성 스트림에서만 0이 아니므로, 판정 4를 쓸 때 스트림 "
                "출처를 명시해야 한다. "
                f"앵커 효과의 시드 안정성 — 외생: {seedc(cond_name('gold','none','exo'), cond_name('teacher','none','exo'))}; "
                f"자기생성: {seedc(cond_name('gold','none','self'), cond_name('teacher','none','self'))}. "
                "**이 결론은 이 실험에서 시드 안정성이 가장 높은 결론이다.**",
            "G6_rejection_sampling_is_only_as_good_as_the_query_it_samples":
                "필터가 replay 타깃에서 **늦게** 문다 — 첫 10라운드 기각률 ~0, 마지막 10라운드 0.55-0.79. "
                "이유는 구조적이다: 타깃을 **이미 쓴 키**에서 생성하므로 저장-접근 해리(학습에 쓴 키는 "
                "오래 맞고 프로브는 먼저 틀린다)가 그대로 필터의 사각지대가 된다. 필터는 배포 지표가 "
                "이미 무너진 뒤에 작동을 시작한다. **rejection sampling의 위력은 필터의 성질이 아니라 "
                "생성기를 평가하는 입력 분포의 성질이다.** 이것은 부호가 아니라 구조 명제이고 "
                "실제 LLM에도 그대로 옮겨진다.",
            "G7_contraction_changes_the_answer_more_than_the_repetition_does":
                "같은 스트림·같은 예산에서 수축(ridge 근접항 / 상위 특이부분공간 제거)을 얹는 것만으로 "
                "유지 지평이 외생 스트림에서 "
                f"ridge {B6['retention_gain_rounds_exo']['ridge']:+d} · "
                f"subspace {B6['retention_gain_rounds_exo']['subspace']:+d} 라운드 움직이고, 그 대가로 "
                f"습득이 ridge {B6['acquisition_cost_of_contraction_exo']['ridge']:+.3f} · "
                f"subspace {B6['acquisition_cost_of_contraction_exo']['subspace']:+.3f} 만큼 깎인다. "
                "→ **열화는 「반복한다」의 성질이 아니라 갱신 규칙의 성질이다**라는 §5.2의 조건절이 "
                "최소 모형에서 독립적으로 재현되고, 동시에 **수축은 공짜가 아니다**(유지-습득 교환). "
                f"시드 안정성 — ridge: {seedc(cond_name('gold','none','exo',OMEGA,'ridge'), cond_name('gold','none','exo'))}; "
                f"subspace: {seedc(cond_name('gold','none','exo',OMEGA,'subspace'), cond_name('gold','none','exo'))}. "
                "**두 수축 형태 중 ridge만 전 시드에서 일치하므로, subspace 쪽은 평균 수준으로만 쓴다.**",
            "G8_report_the_round_size_or_Kstar_is_meaningless":
                "라운드당 ||Delta Theta||_F가 조건마다 다르고(B8) 같은 조건 안에서도 라운드에 따라 "
                "변한다. 라운드 수만으로 조건을 비교하면 라운드 크기 차이를 열화율로 오독한다. "
                "그리고 corpus의 두 논문 다 이 값을 인쇄하지 않으므로 **K=20,000(2605.11836)과 "
                "K=3(2606.04703)을 같은 축에 놓는 것은 현재 불가능하다.** 이것은 이 실험의 결과가 "
                "아니라 이 실험이 corpus에 대해 확인한 결핍이다.",
        },
        "DIRECTIONAL_하한_방향성으로만_쓸_것": {
            "D1_all_absolute_numbers":
                f"K*의 라운드 수(예: gold|exo {ks_gold_exo}, teacher|self {ks_t_self}, "
                f"teacher+filter|self {ks_tf_self}), 읽기 lift의 값, 경보 라운드, mu·rank·omega의 값은 "
                "전부 이 장난감의 산물이다. 본문에 숫자로 옮기지 말 것.",
            "D2_lead_time_is_directional_only":
                "선행/후행 판정은 (a) 하강 75% 경보 규칙, (b) 습득 지표의 이동평균 폭, "
                f"(c) 상승 후보에 쓴 방향 무관 25% 규칙 세 선택에 의존한다. 이송 가능한 것은 "
                "**어느 지표가 먼저 울렸는가의 순서**이고 몇 라운드 먼저인지는 아니다.",
            "D3_which_indicator_leads_depends_on_which_deployment_axis":
                "읽기 지표는 **습득 축**에 대해서는 앞서고(미기입 사실 위 read lift가 "
                f"{lead_ho['n_conditions_scored_acquisition']}개 중 {lead_ho['n_leading_acquisition']}개 "
                f"조건에서 선행, 중앙값 {lead_ho['median_lead_acquisition_rounds']} 라운드) "
                "**유지 축**에 대해서는 대체로 뒤선다"
                f"({lead_ho['n_conditions_scored_retention']}개 중 {lead_ho['n_leading_retention']}개만 "
                f"선행, 중앙값 {lead_ho['median_lead_retention_rounds']} 라운드). "
                "[2606.04703]의 배포 지표는 습득형이므로 **그 논문과 같은 축에서는 부호가 같다.** "
                "그러나 '선행 지표'라는 말을 축 없이 쓰면 안 된다.",
            "D4_contraction_magnitudes":
                "ridge의 mu와 subspace의 rank는 조율되지 않은 단일 값이다. 수축의 **방향**(유지 지평을 "
                "늘리고 습득을 깎는다)만 읽어야 하고, 두 형태 중 어느 쪽이 더 나은지는 이 실험이 "
                "결정하지 않는다.",
            "D5_second_moment_amplitude":
                "생성 분포의 참여비·최빈 점유율의 **값**은 코드북 크기와 프로브 수에 직접 의존한다. "
                "라운드에 대한 **단조 방향**만 이송한다. 참여비가 과제 지표보다 먼저 반감한 조건은 "
                f"{len(pr_lead)}개이고 전부 자기생성 스트림이다 — 방향은 X4 F3과 같으나 "
                "조건 의존적이다.",
            "D6_horizon":
                f"K={K}이다. 2605.11836의 20,000 라운드도 2601.11042의 10,000 편집도 이 실험이 "
                "확인하거나 반증하지 않는다. 여기서 censored로 나온 K*는 전부 하한이다.",
        },
        "CONFIRMED_외부와_같은_방향으로_재현된_것": {},
        "MECHANISM_EXCLUSION_최소모형이_재현하지_못한_것": {},
        "GAPS_STILL_OPEN_억지로_채우지_않는다": {
            "G_A_partially_addressed_not_closed":
                "외생 스트림 + 수축에서의 유지를 이 실험은 **장난감 안에서** 답한다. corpus의 공백은 "
                "그대로다 — 실제 LLM에서 그 구성을 돌리고 이전 라운드 항목의 유지를 잰 논문은 여전히 "
                "0편이고, 이 실험이 그 자리를 채우지 않는다.",
            "G_B_mechanism_shown_magnitude_unknown":
                "제3의 채널이 최소 모형에서 자동으로 생긴다는 것은 보였다. 실제 LLM에서 그 채널이 "
                "간섭·분포붕괴 대비 **얼마나** 무거운지는 이 실험이 말하지 않는다. 2606.04703의 "
                "문맥/내재화 두 열이 유일한 실측이고 그것은 K=3이다.",
            "G_C_answered_conditionally":
                "앵커 대 필터는 갈렸으나 **레짐 조건부로** 갈렸다. 어느 레짐이 실제 배포에 해당하는지는 "
                "이 실험이 결정하지 않는다.",
            "still_absent_everywhere":
                "라운드당 ||Delta Theta|| · 반감기 · 라운드당 비율로서의 rho · 생성 분포의 2차 모멘트 — "
                "네 가지 다 corpus에서 여전히 0편이다. 이 실험은 그것을 **장난감 안에서** 인쇄할 뿐이다.",
            "out_of_scope_and_it_matters":
                "experience granularity와 injection alignment는 이 모형의 범위 밖인데, 실제 LLM에서 "
                "가장 큰 효과를 낸 둘이다(injection 1라운드 효과 WebWalkerQA +8.0; 3라운드 global "
                "premature-answer 63.82% 대 step-wise 0%). **이 실험의 결론은 그 두 축이 고정되었을 "
                "때의 결론이다.**",
        },
        "honest_limits":
            "이것은 LLM 실험이 아니다. 선형 연상기억에 읽기 연산자를 한 장의 행렬로 심고 그 행렬이 "
            "간섭으로 죽게 둔 최소 모형이다. 읽기 능력이 실제 모형에서 어떤 기하로 죽는지는 다를 수 "
            "있고, 열화 속도는 전적으로 이 장난감의 것이다. 이송 가능한 것은 여섯이다 — "
            "(1) 저장을 읽는 연산자가 자기 자신일 때 제3의 채널이 생기고 그것이 스스로 죽는다는 구조 명제, "
            "(2) 그 채널이 배포 지표를 한 번도 움직이지 않고 소멸할 수 있다는 것, "
            "(3) 자기생성 루프의 붕괴 서명이 '더 빨리 잊는다'가 아니라 '조용히 멈춘다'이며 그래서 "
            "유지 지표가 거꾸로 좋아 보인다는 것, (4) off-policy > on-policy의 부호, "
            "(5) 앵커가 필터를 지배하고 필터는 늦게 문다는 순서, (6) 수축이 반복보다 답을 더 크게 "
            "바꾸되 습득을 대가로 받는다는 교환. 그 밖의 모든 수는 이 파일 안에서만 참이다.",
    }

    # 재현/재현 실패 둘 다 결과로 적는다 — 실행 결과에서 자동 판정
    cf = doc["findings"]["CONFIRMED_외부와_같은_방향으로_재현된_것"]
    mx = doc["findings"]["MECHANISM_EXCLUSION_최소모형이_재현하지_못한_것"]
    lead_ctx = summ["ctx_minus_free_new"]
    if lead_ctx["n_leading_acquisition"] <= lead_ctx["n_conditions_scored_acquisition"] / 2:
        mx["X1_the_matched_leading_indicator_did_not_lead_here"] = (
            "「문맥 준 성적 − 문맥 없는 성적」은 이 최소 모형에서 습득 지표를 "
            f"{lead_ctx['n_conditions_scored_acquisition']}개 조건 중 "
            f"{lead_ctx['n_leading_acquisition']}개에서만 앞섰다"
            f"(중앙값 리드 {lead_ctx['median_lead_acquisition_rounds']} 라운드). "
            "[2606.04703]이 실제 LLM에서 본 한 라운드 선행이 여기서는 일반적으로 재현되지 않는다. "
            "원 논문을 반박하지 않는다 — 그 논문의 선행은 검색 오정렬(global injection)이 만든 "
            "것이고 그 축은 여기 없다.")
    else:
        cf["C1_the_matched_leading_indicator_leads_on_the_acquisition_axis"] = (
            f"「문맥 준 성적 − 문맥 없는 성적」이 점수화 가능한 "
            f"{lead_ctx['n_conditions_scored_acquisition']}개 조건 중 "
            f"{lead_ctx['n_leading_acquisition']}개에서 습득 지표보다 먼저 울렸다"
            f"(중앙값 {lead_ctx['median_lead_acquisition_rounds']} 라운드). 부호가 [2606.04703] "
            "구성 A·C(문맥 성적이 내재화 성적보다 한 라운드 먼저 무너진다)와 같다. "
            "**나머지 5개 조건은 외생 스트림이라 습득 지표가 아예 경보하지 않아 점수화 불가다** — "
            "그 자체가 G2의 근거다. 라운드 수는 이송 불가.")
    inv = B4["frac_pairs_inverted"]
    if inv is not None and inv > 0:
        cf["C2_early_late_order_inversion_exists"] = (
            f"초기(k=1-5)와 후기(k={K-9}-{K}) 순위 사이에 {B4['n_pairs_inverted']}/{B4['n_pairs_scored']} "
            f"쌍이 뒤집힌다(Spearman {B4['spearman_early_vs_late']:.4f}). 원 X4에서는 천장 때문에 이 "
            "질문 자체가 정의되지 않았다. **초기 라운드의 조건 순위를 최종 순위로 읽으면 안 된다**는 "
            "것이 두 논문에 이어 최소 모형에서도 확인된다.")
    else:
        mx["X2_no_early_late_order_inversion"] = (
            "초기와 후기의 조건 순위가 뒤집히지 않았다. 두 논문이 보고한 초기 반전이 이 모형에서는 "
            "나타나지 않는다 — 그 반전이 일반적 선형 간섭의 성질이 아니라는 뜻일 뿐, 원 논문을 "
            "반박하지 않는다.")
    ratio = B2["capability_at_k150_in_loop_mean"] / max(B2["capability_at_k150_out_loop_mean"], 1e-12)
    mx["X3_capability_and_read_channel_are_coupled_here"] = (
        f"일반 capability와 읽기 채널이 같은 방향으로 함께 움직인다(루프 안 capability "
        f"{B2['capability_at_k150_in_loop_mean']:.4f} 대 밖 {B2['capability_at_k150_out_loop_mean']:.4f}, "
        f"비 {ratio:.3f}). [2606.04703]은 둘을 해리 가능한 것으로 다룬다(step-wise는 experience-use "
        "ability를 지키면서 과제 성적도 지킨다). **이 모형은 그 해리를 만들지 못한다** — 해리를 "
        "만드는 축(injection alignment)이 범위 밖이기 때문이다.")
    ms0 = float(np.mean([agg[n]["_full_gen_mode_share"][0] for n in names_fac]))
    ms1 = float(np.mean([agg[n]["_full_gen_mode_share"][K] for n in names_fac]))
    pr_rat = float(np.mean([agg[n]["_full_gen_pr"][K] / agg[n]["_full_gen_pr"][0] for n in names_fac]))
    mx["X4_behavioral_mode_collapse_is_not_reproduced_only_the_spread_shrinks"] = (
        f"생성 분포의 **퍼짐**은 확실히 줄어든다(참여비 비 k={K}/k=0 평균 {pr_rat:.3f}). 그러나 "
        f"**한 모드로의 쏠림은 거의 없다** — 최빈 코드북 점유율이 {ms0:.3f} -> {ms1:.3f}로만 오른다. "
        "[2606.04703]의 붕괴는 premature answer 63.82%라는 **하나의 행동 모드로의 쏠림**이다. "
        "이 모형은 그 형태를 만들지 못한다. 선형 사상에는 '종료'라는 행동이 없고, 쏠림을 만드는 "
        "축(injection alignment)이 범위 밖이기 때문이다. **F3의 2차 모멘트 주장은 퍼짐 축에서만 "
        "한 언어로 말해지고, 모드 쏠림 축에서는 아직 아니다.**")
    bad_sign = B3["verdict"]["not_usable_here"]
    if bad_sign:
        mx["X6_two_externally_attested_indicators_have_the_opposite_sign_here"] = (
            f"{bad_sign} 는 원 논문에서 **상승**하는 양이나 이 최소 모형에서는 **하강**한다 "
            f"(조건수 {B3['by_condition'][n_ge]['cond_cov']['ratio_k150_over_k1']:.4f}배, "
            f"노름비 {B3['by_condition'][n_ge]['norm_ratio']['ratio_k150_over_k1']:.4f}배). "
            "원인은 구조적이다 — 여기서 running covariance는 서로 다른 키가 쌓이며 rank가 차서 "
            "조건수가 내려가고, ||Theta||는 Theta_0에 심어 둔 읽기 항이 쓰기에 깎여 내려간다. "
            "**그러므로 이 실험은 그 두 지표를 검증하지도 반증하지도 않는다.** 선행 지표 후보 셋 중 "
            "**하나(문맥 준 성적 − 문맥 없는 성적)만 이 모형에서 같은 양으로 성립한다.**")
    rbf = [n for n, v in B4["rise_before_fall_by_condition"].items() if v]
    if rbf:
        cf["C3_rise_before_fall_reproduces_but_only_with_a_retained_anchor"] = (
            f"[2606.04703] 구성 C의 '상승 후 절벽' 모양이 {len(rbf)}개 조건에서 재현되고 "
            f"**전부 원본 앵커(gold) 팔이다**(정점 라운드 "
            f"{[B4['peak_round_by_condition'][n] for n in rbf]}). 생성 타깃 팔은 첫 라운드가 최고점이다. "
            "→ 상승 구간의 존재는 이송되나, 그 논문의 구성 C는 global injection 팔이었으므로 "
            "**기제가 같다고 말할 수 없다.** 모양의 일치일 뿐이다.")
    else:
        mx["X5_no_rise_before_fall"] = (
            "어느 조건에서도 상승 후 절벽이 나오지 않았다. [2606.04703] 구성 C의 모양은 이 모형의 "
            "기제로 설명되지 않는다.")

    out = HERE / "result.json"
    out.write_text(json.dumps(r(doc), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"conditions={len(all_conds)} seeds={len(SEEDS)} K={K}")
    print(f"access@k0 (gold|exo) = {B0['cohort_access_at_k0_mean']:.4f}  chance={chance:.5f}")
    print(f"K*_rel50: gold|exo={ks_gold_exo} teacher|self={ks_t_self} "
          f"teacher+filter|self={ks_tf_self} student|self={ks_s_self} "
          f"teacher|exo={ks_t_exo} student|exo={ks_s_exo}")
    print(f"main effects (rounds): target={main_eff['target_teacher_minus_student_rounds']:+.1f} "
          f"filter={main_eff['filter_on_minus_off_rounds']:+.1f} "
          f"stream={main_eff['stream_self_minus_exo_rounds']:+.1f}")
    print(f"filter within exo={inter['filter_effect_within_exo']:+.1f} "
          f"within self={inter['filter_effect_within_self']:+.1f}")
    print(f"contraction gain exo={B6['retention_gain_rounds_exo']} self={B6['retention_gain_rounds_self']}")
    print(f"order inversion={B4['n_pairs_inverted']}/{B4['n_pairs_scored']} "
          f"spearman={B4['spearman_early_vs_late']}")
    print("lead summary:", json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
