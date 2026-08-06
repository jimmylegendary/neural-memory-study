#!/usr/bin/env python3
"""X4 — 반복 consolidation의 열화율 rho: (U-Theta)를 K라운드 돌리면 무엇이 어떻게 무너지는가.

배경
----
S0 §2.4는 각 논문에서 비용 4종(B_s, L_w, C, rho)을 뽑으라고 요구한다. 넷 중
**rho(라운드당 망각·열화율)가 corpus에서 가장 심하게 미보고된 값**이다. 이유는 구조적이다:
대부분의 논문이 sleep 라운드를 한 번만 돌고, 반복했을 때 무엇이 무너지는지 재지 않는다.
corpus에서 rho를 실제로 말하는 논문은 둘뿐이고 둘 다 반쪽이다.

  [model-collapse 2404.01413] rho를 **닫힌 형태**로 준 유일한 논문.
      Replace          누적 오차 ~ k         [Eq. 4]   (발산)
      Replace-Multiple 누적 오차 ~ log k     [Eq. 11]  (발산)
      Accumulate       누적 오차 <= pi^2/6   [Eq. 3]   (유계)
      (1차 차분이 rho_k = 1 / (1/k) / (1/k^2) — model-collapse 노트 M-10, note-writer 산술)
      단 이 정리는 선형회귀·공변량 고정·**라운드마다 분산 sigma^2를 상수로 재주입**하는 모형이다.
      즉 생성 분포의 2차 모멘트는 가정으로 못 박혀 있어 원리적으로 표현될 수 없다.
      정작 이 논문의 VAE 실험은 바로 그 좌표에서 실패한다 — 손실은 (거의) 유계인데 다양성이
      사라진다(안경·액세서리) [§2.3 L532-536]. 캡션은 "Avoids Model Collapse"라고 쓰지만
      본문은 accumulate에서도 test error가 증가한다고 적는다 [§2.3 L537-538].

  [continual-facts-in-weights 2607.11020] rho를 **측정**한 유일한 논문(K=20, 100).
      단 라운드당 비율이 아니라 유지 곡선·종점 수준으로만 보고한다.
      그 논문의 not_reported: "감쇠 상수 없음, 반감기 없음, k=100 너머 외삽 없음".

이 실험이 묻는 것
----------------
Theta-경로가 배포에서 성립하려면 K가 커져도 견뎌야 한다. 그러므로 물어야 할 것은
"rho가 얼마인가"(절대치, 이송 불가)가 아니라 **"rho가 K에 대해 어떤 모양인가"**이다.
모양·부호·순서·crossover는 이송 가능하다(REPRODUCE.md §0의 정직성 계약).

  Q1 [model-collapse]의 세 스케줄이 **다른 추정 문제**에서도 같은 함수형·부호로 나오는가.
     나온다면 1/i 가중은 선형회귀의 성질이 아니라 데이터 스케줄의 성질이다.
  Q2 그 정리가 가정으로 고정한 좌표 — 생성 분포의 2차 모멘트 — 를 풀면 무엇이 나오는가.
  Q3 사실을 순차 주입하며 K라운드 돌릴 때 (i) 새 사실 습득, (ii) 이전 사실 유지, (iii) rho가
     어떻게 움직이는가. rho는 선형인가 지수인가 포화인가 — 아니면 그 셋 다 아닌가.
  Q4 replay 혼합비 omega가 그 모양을 어떻게 바꾸는가. replay의 **출처**(보존된 원본 vs
     모델 자기생성)가 Q1의 accumulate/replace 대비를 재현하는가.
  Q5 Theta-경로가 K라운드를 견디려면 무엇이 필요한가.

설계 선택과 근거
---------------
LLM을 쓸 수 없다(호스트에 NVIDIA GPU 없음, torch 금지). 그러므로 기제가 보존되는 최소 모형을
쓰되, 최소 모형이 재현하지 못하는 것을 재현했다고 말하지 않는다. **재현 실패도 결과로 적는다.**

(a) 해석 파트 — 가우시안 위치-척도 추정.
    왜 이것인가: [model-collapse]의 정리는 선형회귀에서 나왔다. 설계행렬이 **아예 없는** 문제에서도
    같은 함수형이 나오면, 그 결과는 회귀의 성질이 아니라 스케줄의 성질이다. 이것이 부호 검증의
    요점이다. 그리고 가우시안은 2차 모멘트가 모형의 자유 변수라서, 그 정리가 가정으로 고정해
    버린 좌표를 열어 볼 수 있다 — 선형회귀 정리에서는 원리적으로 불가능하다.
    유도는 전부 정확(exact)하다. 근사 없음. 몬테카를로로 교차검증한다.

(b) 시뮬 파트 — 선형 연상기억(linear associative memory)에 사실 순차 주입.
    왜 이것인가: [continual-facts-in-weights]의 루프는 Theta_{k+1} = Theta_k + Delta Theta_k
    (adapter를 merge)이고 각 라운드는 생성 집합 R_k 위의 완전 최적화다. 그 구조를 보존하는
    최소 모형이 "키->값 사상의 순차 덮어쓰기"이며, 그 논문이 실제로 측정한 네 축이 전부 정의된다.
      access     = 학습에 안 쓴 paraphrase 프로브로 디코딩          -> strict accuracy
      storage    = 학습에 쓴 정본 키로 디코딩 + 정규화 lift          -> 그 논문 Eq. 5.1의 대응물
      capability = Theta_0가 원래 갖고 있던 사상의 코사인 보존       -> held-out capability
      drift      = ||Theta_k - Theta_0||_F / ||Theta_0||_F          -> KL(Theta_k||Theta_0)의 대응물
    없는 것은 합성(composition)·함의(entailment)다. 선형 사상에 그 기제가 없다. 그래서 이 실험은
    survival(생존) 절반만 다루고 creation(entailment gap) 절반은 다루지 않는다.

(c) 예산 정합. omega를 쓸어도 라운드당 항목 수 N_total과 경사 스텝 수 S를 고정한다.
    [continual-facts-in-weights App. A.3]의 "steps are matched, not tokens" 규율과 같은 이유다.

(d) rho의 모양을 세 후보(linear/exp/plateau)로만 재지 않는다. 네 번째 후보 **logistic 임계**를
    같이 적합한다. 사전에 정한 것이 아니라 1차 실행에서 access 곡선이 매끄러운 감쇠가 아니라
    "오래 평평하다 절벽"이었기 때문이다. 다만 네 후보의 SSE 차이는 작아 **모형 선택은 결정적이지
    않다** — 그래서 판정은 모형 선택이 아니라 모형에 무관한 두 통계로 한다:
      grace 구간(지표가 0.95 이상으로 버티는 라운드 수)과 rho의 앞구간/최댓값 비.
    이 저장소의 규칙상 crossover 위치는 이송 가능한 양이므로, 보고해야 할 것은 K*(omega)다.

등급
----
EXPLORATION-GRADE / MECHANISM-TRANSFER. 이것은 LLM 실험이 아니다.
- 이송 가능: 부호, 함수형, 조건 간 **순서**, crossover의 존재와 그 이동 방향, 정성적 dissociation.
- 이송 불가: 절대 rho, 유지율 %, K*의 라운드 수, 필요 omega 값, 반감기.
- 해석 파트의 유도는 exact이고 몬테카를로로 확인한다. 그러나 그 exactness는 가우시안 모형
  안에서 참이며, LLM에 대해 참이라는 뜻이 아니다.
- 논문이 보고하지 않은 값을 추정해 본문 수치로 쓰지 않는다. "논문에 없음"이 결과다.

결정론
------
난수는 (b) 파트와 (a)의 몬테카를로 검증에만 쓴다. 전부 np.random.default_rng 고정 시드,
시드 5개, 시드 간 표준편차를 함께 보고한다(단일 시드 결론 금지). BLAS 스레드 수를 1로 못 박는다 —
멀티스레드 reduction 순서는 부동소수 결과를 흔들 수 있고, 그것이 이 저장소에서 실측된 유일한
비결정성의 원인 유형이었다(REPRODUCE.md §3). 보고값은 6자리 반올림. wall-clock은 result.json에
넣지 않는다(넣으면 bit-identity가 깨진다). PYTHONHASHSEED=0으로 2회 실행해 sha256 동일을 확인한다.
"""

from __future__ import annotations

import os

# BLAS 스레드 고정: 반드시 numpy import 이전에.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
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
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if not math.isfinite(v) else round(v, RND)
    if isinstance(x, np.integer):
        return int(x)
    return x


# =============================================================================
# PART A — 해석: 가우시안 위치-척도 모형의 자기학습 루프
# =============================================================================
#
# 문제. 진짜 분포 N(mu, sigma^2). 라운드 1은 실데이터 n개로 적합. 라운드 k>=2는 **직전 라운드가
#       적합한 분포**에서 n개를 뽑아 다시 적합. 추정량은 표본평균 mu_hat, 표본분산 sigma2_hat.
#       위험은 [model-collapse]의 E_test와 같은 자리에 둔다: L(k) = E[(mu_hat_k - mu)^2].
#
# 스케줄 3종은 그 논문의 세 regime과 1:1 대응한다.
#   Replace          R_k = gen(Theta_{k-1};B_s)                 |R_k| = n
#   Replace-Multiple R_k = gen(Theta_{k-1};k B_s)               |R_k| = k n
#   Accumulate       R_k = R_{k-1} U gen(Theta_{k-1};B_s)       |R_k| = k n, R_1은 실데이터
#
# ---------------------------------------------------------------------------
# A-1. 분산 고정(variance-pinned) — 그 논문의 가정을 그대로 켠 경우
# ---------------------------------------------------------------------------
#   mu_hat_k = mu_hat_{k-1} + eps_k / w_k,  eps_k = (블록 k 표본평균) - mu_hat_{k-1}
#   Replace          w_k = 1,  Var(eps_k) = sigma^2/n        -> L(k) = (sigma^2/n) k
#   Replace-Multiple w_k = 1,  Var(eps_k) = sigma^2/(k n)    -> L(k) = (sigma^2/n) H_k ~ ln k
#   Accumulate       w_k = k,  Var(eps_k) = sigma^2/n        -> L(k) = (sigma^2/n) sum 1/i^2
#                                                                    <= (sigma^2/n) pi^2/6
#   eps는 마팅게일 차분이라 서로 무상관 -> 분산이 그대로 더해진다. 유도 끝.
# Accumulate의 1/k 가중이 곧 그 논문 §3.2 L818-822의 직관("iteration i는 코퍼스의 1/i")이다.
# 여기엔 설계행렬도 inverse-Wishart 인자도 없다. 그런데 함수형이 같다.


def analytic_pinned(K: int, n: int, sigma2: float) -> dict:
    ks = np.arange(1, K + 1, dtype=float)
    base = sigma2 / n
    return {
        "k": ks.astype(int).tolist(),
        "replace": (base * ks).tolist(),
        "replace_multiple": (base * np.cumsum(1.0 / ks)).tolist(),
        "accumulate": (base * np.cumsum(1.0 / ks ** 2)).tolist(),
        "accumulate_bound_pi2_over_6": base * (math.pi ** 2) / 6.0,
    }


# ---------------------------------------------------------------------------
# A-2. 분산 자유(variance-free) — 그 가정을 푼 경우
# ---------------------------------------------------------------------------
# 자기 표본으로 재적합하면 생성 분산 자체가 움직인다. MLE 분산(1/n)에서
#   E[sigma2_hat_k | sigma2_hat_{k-1}] = ((n-1)/n) sigma2_hat_{k-1}
# 이므로 REPLACE에서 E[sigma2_hat_k] = sigma^2 c^k,  c = (n-1)/n  — 기하 붕괴.
#   반감기 = ln2 / ln(n/(n-1)) ~ 0.693 n 라운드.
# 이것이 [model-collapse]의 정리가 원리적으로 볼 수 없는 좌표다(sigma^2를 상수로 재주입하므로).
# 그리고 위치 위험은 L(k) = sigma^2 (1 - c^k) -> sigma^2 로 **포화**한다.
# 즉 분산이 죽으면 위치 오차는 발산하지 않고 saturate 한다. 손실 곡선이 평평해지는 것이
# 안정의 증거가 아닐 수 있다는 뜻이다.
# 불편(unbiased) 분산이면 E[sigma2_hat_k]=sigma^2 (마팅게일)이고 L(k) = (sigma^2/n) k —
# [Eq. 4]와 정확히 같은 선형 발산. "선형 발산이냐 포화냐"는 추정량의 편향 보정이 가른다.
#
# ACCUMULATE의 정확 재귀. 등블록 크기 n에서 항등식
#   sigma2_hat_k = (1/k) sum_i s_i^2 + (1/k) sum_i (m_i - mu_hat_k)^2
# 에 다음을 넣으면 기댓값에 대해 **정확한** 재귀가 나온다(근사 없음).
#   E[s_i^2] = ((n-1)/n) u_{i-1},  u_j := E[sigma2_hat_j],  E[s_1^2] = ((n-1)/n) sigma^2
#   m_i - mu_hat_k = eps_i (1 - 1/i) - sum_{l=i+1}^{k} eps_l/l   (i>=2)
#   m_1 - mu_hat_k = - sum_{l=2}^{k} eps_l/l
#   E[eps_l^2] = u_{l-1}/n,  E[eps_i eps_l] = 0 (i != l)


def analytic_free_replace(K: int, n: int, sigma2: float) -> dict:
    c = (n - 1.0) / n
    ks = np.arange(1, K + 1)
    return {
        "k": ks.tolist(),
        "E_var_mle": (sigma2 * c ** ks).tolist(),
        "risk_mle": (sigma2 * (1.0 - c ** ks)).tolist(),
        "risk_unbiased_var": ((sigma2 / n) * ks).tolist(),
        "var_half_life_rounds": math.log(2.0) / math.log(n / (n - 1.0)),
        "risk_saturation_limit": sigma2,
    }


def analytic_free_accumulate(K: int, n: int, sigma2: float) -> dict:
    cw = (n - 1.0) / n
    u = np.zeros(K + 1); risk = np.zeros(K + 1)
    T = np.zeros(K + 1); A = np.zeros(K + 1); Q = np.zeros(K + 1); P = np.zeros(K + 1)
    u[1] = cw * sigma2
    risk[1] = sigma2 / n
    for k in range(2, K + 1):
        um = u[k - 1]                       # = E[sigma2_hat_{k-1}]
        T[k] = T[k - 1] + um / (n * k * k)
        A[k] = A[k - 1] + um
        Q[k] = Q[k - 1] + (1.0 - 1.0 / k) ** 2 * um / n
        P[k] = P[k - 1] + T[k]
        within = cw * (sigma2 + A[k]) / k
        between = (T[k] + Q[k] + (k - 1) * T[k] - P[k]) / k
        u[k] = within + between
        risk[k] = sigma2 / n + T[k]
    return {"k": list(range(1, K + 1)),
            "E_var_mle": u[1:].tolist(), "risk_mle": risk[1:].tolist()}


def mc_gaussian(K: int, n: int, sigma2: float, reps: int, seed: int) -> dict:
    """몬테카를로: MLE 분산으로 replace / accumulate를 실제로 돌려 재귀식을 검증."""
    rng = np.random.default_rng(seed)
    sd = math.sqrt(sigma2)
    out = {}

    cur_mu, cur_sd = np.zeros(reps), np.full(reps, sd)
    vh, rh = np.zeros(K), np.zeros(K)
    for k in range(K):
        x = cur_mu[:, None] + cur_sd[:, None] * rng.standard_normal((reps, n))
        m, v = x.mean(axis=1), x.var(axis=1)          # MLE, ddof=0
        cur_mu, cur_sd = m, np.sqrt(np.maximum(v, 0.0))
        vh[k], rh[k] = v.mean(), (m ** 2).mean()      # mu = 0
    out["replace"] = {"E_var": vh.tolist(), "risk": rh.tolist()}

    s1 = np.zeros(reps); s2 = np.zeros(reps); cnt = 0
    cur_mu, cur_sd = np.zeros(reps), np.full(reps, sd)
    vh, rh = np.zeros(K), np.zeros(K)
    for k in range(K):
        x = cur_mu[:, None] + cur_sd[:, None] * rng.standard_normal((reps, n))
        s1 += x.sum(axis=1); s2 += (x ** 2).sum(axis=1); cnt += n
        m = s1 / cnt
        v = np.maximum(s2 / cnt - m ** 2, 0.0)
        cur_mu, cur_sd = m, np.sqrt(v)
        vh[k], rh[k] = v.mean(), (m ** 2).mean()
    out["accumulate"] = {"E_var": vh.tolist(), "risk": rh.tolist()}
    return out


# =============================================================================
# PART B — 시뮬: 선형 연상기억에 사실 M개 순차 주입, K라운드 반복
# =============================================================================

def _unit(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


class World:
    def __init__(self, cfg, seed):
        rng = np.random.default_rng(seed)
        d = cfg["d"]
        M = cfg["n_cohort"] + cfg["n_later"]
        self.cfg, self.d, self.M = cfg, d, M
        self.theta0 = rng.standard_normal((d, d)) / math.sqrt(d)
        self.theta0_fro = float(np.linalg.norm(self.theta0))

        self.anchors = _unit(rng.standard_normal((M, d)))
        npool = cfg["J_max"] + cfg["n_probe"]
        noise = rng.standard_normal((M, npool, d)) / math.sqrt(d)
        pool = _unit(self.anchors[:, None, :] + cfg["spread"] * noise)
        self.train_keys = pool[:, : cfg["J_max"], :]   # 학습용
        self.probe_keys = pool[:, cfg["J_max"]:, :]    # 평가 전용 — 학습에 절대 안 씀

        self.codebook = _unit(rng.standard_normal((M + cfg["n_distract"], d)))
        self.values = self.codebook[:M]
        self.chance = 1.0 / self.codebook.shape[0]

        self.cap_q = _unit(rng.standard_normal((cfg["n_cap"], d)))
        self.cap_target_n = _unit(self.cap_q @ self.theta0.T)


def _decode_acc(theta, keys, labels, codebook):
    return float(np.mean(np.argmax((keys @ theta.T) @ codebook.T, axis=1) == labels))


def _decode_margin(theta, keys, labels, codebook):
    """정답 코드북 항목의 cos - 최대 오답 cos. 임계 지표가 못 보는 연속 여유."""
    y = keys @ theta.T
    ny = np.linalg.norm(y, axis=1, keepdims=True)
    s = (y / np.where(ny > 0, ny, 1.0)) @ codebook.T
    own = s[np.arange(len(labels)), labels].copy()
    s[np.arange(len(labels)), labels] = -np.inf
    return float(np.mean(own - s.max(axis=1)))


def _cos_to_value(theta, keys, vals):
    y = keys @ theta.T
    ny = np.linalg.norm(y, axis=1)
    return np.sum(y * vals, axis=1) / np.where(ny > 0, ny, 1.0)


def _write(theta, K_items, V_items, steps, lr):
    m = K_items.shape[0]
    for _ in range(steps):
        g = (K_items @ theta.T - V_items).T @ K_items / m
        theta = theta - lr * g
    return theta


def run_sim(cfg, seed, omega, replay_source, breadth_store, breadth_later,
            normalize_self=False):
    """한 조건 1회 실행.

    replay_source: 'none' | 'gold' (보존된 원본 값)  | 'self' (현재 모델의 출력을 타깃으로)
                   'self'가 곧 [continual-facts-in-weights §6.2]의 own-merges,
                   PART A의 replace regime에 해당한다.
    """
    W = World(cfg, seed)
    rng = np.random.default_rng(seed * 7919 + 13)
    d, C, K = cfg["d"], cfg["n_cohort"], cfg["n_later"]
    N_total, S, lr = cfg["N_total"], cfg["steps"], cfg["lr"]

    theta = W.theta0.copy()
    cohort_probe = W.probe_keys[:C].reshape(-1, d)
    cohort_probe_lab = np.repeat(np.arange(C), cfg["n_probe"])
    cohort_anchor = W.anchors[:C]
    cohort_vals = W.values[:C]
    cos_before = _cos_to_value(theta, cohort_anchor, cohort_vals)

    def items_for(fact_id, breadth, count):
        if breadth == "narrow":
            ks = np.repeat(W.anchors[fact_id][None, :], count, axis=0)
        else:
            ks = W.train_keys[fact_id][np.arange(count) % cfg["J_max"]]
        return ks, np.repeat(W.values[fact_id][None, :], count, axis=0)

    buf_k, buf_v = [], []
    for fid in range(C):
        ks, vs = items_for(fid, breadth_store, N_total)
        theta = _write(theta, ks, vs, S, lr)
        jk, jv = items_for(fid, breadth_store, cfg["J_max"] if breadth_store == "broad" else 1)
        buf_k.append(jk); buf_v.append(jv)

    cos_after = _cos_to_value(theta, cohort_anchor, cohort_vals)
    denom = np.where(np.abs(cos_after - cos_before) > 1e-9, cos_after - cos_before, np.nan)

    acc, sto, lift, cap, dri = [], [], [], [], []

    def snapshot():
        acc.append(_decode_acc(theta, cohort_probe, cohort_probe_lab, W.codebook))
        sto.append(_decode_acc(theta, cohort_anchor, np.arange(C), W.codebook))
        cn = _cos_to_value(theta, cohort_anchor, cohort_vals)
        lift.append(float(np.nanmean((cn - cos_before) / denom)))
        cap.append(float(np.mean(np.sum(_unit(W.cap_q @ theta.T) * W.cap_target_n, axis=1))))
        dri.append(float(np.linalg.norm(theta - W.theta0) / W.theta0_fro))

    snapshot()  # k = 0

    all_k = np.concatenate(buf_k, axis=0)
    all_v = np.concatenate(buf_v, axis=0)
    n_new = max(1, int(round((1.0 - omega) * N_total)))
    n_rep = N_total - n_new

    new_acc, new_margin = [], []
    for t in range(K):
        fid = C + t
        ks, vs = items_for(fid, breadth_later, n_new)
        if n_rep > 0 and replay_source != "none":
            sel = rng.integers(0, all_k.shape[0], size=n_rep)
            kr = all_k[sel]
            if replay_source == "gold":
                vr = all_v[sel]
            else:
                y = kr @ theta.T
                vr = _unit(y) if normalize_self else y
            ks = np.concatenate([ks, kr], axis=0)
            vs = np.concatenate([vs, vr], axis=0)
        theta = _write(theta, ks, vs, S, lr)

        lab = np.full(cfg["n_probe"], fid)
        new_acc.append(_decode_acc(theta, W.probe_keys[fid], lab, W.codebook))
        new_margin.append(_decode_margin(theta, W.probe_keys[fid], lab, W.codebook))

        jk, jv = items_for(fid, breadth_later, cfg["J_max"] if breadth_later == "broad" else 1)
        all_k = np.concatenate([all_k, jk], axis=0)
        all_v = np.concatenate([all_v, jv], axis=0)
        snapshot()

    return {"access": acc, "storage": sto, "lift": lift, "capability": cap,
            "drift": dri, "new_acc": new_acc, "new_margin": new_margin,
            "chance": W.chance}


def agg(runs, key):
    A = np.array([x[key] for x in runs], dtype=float)
    return A.mean(axis=0), A.std(axis=0, ddof=0)


# =============================================================================
# PART C — rho의 모양 판정
# =============================================================================
# 후보 4종을 같은 데이터에 SSE로 비교한다. 전부 결정론(격자 + 닫힌 형태 최소제곱).
#   linear    R(k) = a + b k                        (rho 상수)
#   exp       R(k) = A exp(-lam k)                  (rho 지수 감쇠, 0으로)
#   plateau   R(k) = c + A exp(-lam k)              (바닥값으로 포화)
#   logistic  R(k) = a / (1 + exp((k - k0)/tau))    (임계 — 오래 버티다 절벽)
# [continual-facts-in-weights §4.3]은 k=100에서 0이 아니라 25-28%로 포화(plateau)한다고 보고한다.
# 그러므로 "어느 형태가 이기는가"는 그 논문과의 모양 수준 대조이기도 하다.

_LAM = np.concatenate([np.linspace(0.002, 0.2, 120), np.linspace(0.205, 2.0, 180)])


def _lsq(D, y):
    coef, *_ = np.linalg.lstsq(D, y, rcond=None)
    res = y - D @ coef
    return coef, float(res @ res)


def fit_shapes(y):
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
            if best is None or s < best[1]:
                best = (k0, tau, s, c)
    out["logistic"] = {"sse": best[2], "k0": best[0], "tau": best[1], "a": best[3][0]}

    for m in ("linear", "exp", "plateau", "logistic"):
        out[m]["r2"] = (1.0 - out[m]["sse"] / tss) if tss > 0 else None
    out["best"] = min(("linear", "exp", "plateau", "logistic"),
                      key=lambda m: out[m]["sse"])
    return out


def kstar(traj, thr):
    """K* = 지표가 thr 이상을 유지한 마지막 라운드. 끝까지 유지되면 censored."""
    a = np.asarray(traj, dtype=float)[1:]     # k=1..K
    idx = np.where(a >= thr)[0]
    if len(idx) == 0:
        return 0, False
    kk = int(idx.max()) + 1
    return kk, kk >= len(a)


def first_below(traj, thr):
    """지표가 thr 아래로 처음 떨어진 라운드. 안 떨어지면 None."""
    a = np.asarray(traj, dtype=float)[1:]
    idx = np.where(a < thr)[0]
    return (int(idx.min()) + 1) if len(idx) else None


def shape_stats(traj):
    """모형 선택에 의존하지 않는 모양 통계.

    grace   = 지표가 0.95 아래로 처음 내려간 라운드 (그때까지는 사실상 손실 0)
    fall    = 0.9 아래로 내려간 라운드부터 0.1 아래로 내려간 라운드까지의 폭
    rho 앞구간이 0에 가깝고 최댓값이 뒤쪽 라운드에 있으면 '상수 비율'은 기각된다.
    """
    a = np.asarray(traj, dtype=float)
    rho = -np.diff(a) * 100.0
    g = first_below(a, 0.95)
    lo9, lo1 = first_below(a, 0.9), first_below(a, 0.1)
    return {
        "grace_round_first_below_0.95": g,
        "first_below_0.9": lo9,
        "first_below_0.1": lo1,
        "fall_width_0.9_to_0.1": (None if (lo9 is None or lo1 is None) else lo1 - lo9),
        "rho_mean_first10_pp": float(rho[:10].mean()),
        "rho_max_pp": float(rho.max()),
        "rho_argmax_round": int(np.argmax(rho) + 1),
        "rho_max_over_first10": (None if abs(rho[:10].mean()) < 1e-9
                                 else float(rho.max() / rho[:10].mean())),
    }


def _avg_rank(x):
    """동순위를 평균 순위로 처리. 묶임이 있을 때 argsort 순위는 임의 순서를 만들어 가짜 상관을 낸다."""
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
    """동순위 보정 Spearman. 어느 한쪽이 상수면 정의되지 않으므로 None."""
    ra, rb = _avg_rank(a), _avg_rank(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = math.sqrt(float(ra @ ra) * float(rb @ rb))
    return float(ra @ rb / den) if den > 0 else None


# =============================================================================
# main
# =============================================================================

def main():
    t0 = time.time()

    # ------------------------------------------------------------------ PART A
    K_A, n_A, sigma2 = 200, 50, 1.0
    pinned = analytic_pinned(K_A, n_A, sigma2)
    free_rep = analytic_free_replace(K_A, n_A, sigma2)
    free_acc = analytic_free_accumulate(K_A, n_A, sigma2)

    # MC 검증 지평은 60으로 짧게 잡는다: replace의 분산은 곱셈 과정이라 꼬리가 무겁고,
    # k가 커지면 유한 표본으로 **평균**을 재는 것 자체가 어려워진다(유도 오차가 아니라 MC 오차).
    K_MC, MC_REPS, MC_SEEDS = 60, 15000, [11, 12, 13, 14, 15]
    mc_runs = [mc_gaussian(K_MC, n_A, sigma2, MC_REPS, s) for s in MC_SEEDS]

    # 판정 기준: k마다 z를 재므로 다중비교다. 개별 3sigma가 아니라 family-wise p로 판정한다.
    #   p_fw = 1 - (1 - erfc(zmax/sqrt(2)))^N,  N = 비교 횟수(= K_MC)
    # p_fw > 0.01 이면 "재귀식과 MC가 일치"로 본다.
    mc_check = {}
    for reg, ana in (("replace", free_rep), ("accumulate", free_acc)):
        for key, aname in (("E_var", "E_var_mle"), ("risk", "risk_mle")):
            A = np.array([m[reg][key] for m in mc_runs], dtype=float)
            mu = A.mean(axis=0)
            se = A.std(axis=0, ddof=1) / math.sqrt(len(MC_SEEDS))
            ref = np.array(ana[aname], dtype=float)[:K_MC]
            z = np.abs(mu - ref) / np.maximum(se, 1e-15)
            p_one = math.erfc(float(z.max()) / math.sqrt(2.0))
            p_fw = 1.0 - (1.0 - p_one) ** K_MC
            mc_check[f"{reg}.{key}"] = {
                "max_abs_z_over_k": float(z.max()),
                "median_abs_z": float(np.median(z)),
                "n_comparisons": K_MC,
                "familywise_p_of_max_z": p_fw,
                "consistent": bool(p_fw > 0.01),
                "mc_at_kmc": float(mu[-1]), "exact_at_kmc": float(ref[-1]),
                "mc_se_at_kmc": float(se[-1]),
            }

    idx = [0, 4, 9, 49, 99, 199]
    A_table = {
        "checkpoints_k": [pinned["k"][i] for i in idx],
        "pinned_replace": [pinned["replace"][i] for i in idx],
        "pinned_replace_multiple": [pinned["replace_multiple"][i] for i in idx],
        "pinned_accumulate": [pinned["accumulate"][i] for i in idx],
        "pinned_accumulate_bound_pi2_over_6": pinned["accumulate_bound_pi2_over_6"],
        "ratio_replace_over_accumulate": [pinned["replace"][i] / pinned["accumulate"][i]
                                          for i in idx],
        "ratio_replace_over_replace_multiple": [pinned["replace"][i] / pinned["replace_multiple"][i]
                                                for i in idx],
        "free_replace_E_var": [free_rep["E_var_mle"][i] for i in idx],
        "free_replace_risk": [free_rep["risk_mle"][i] for i in idx],
        "free_accumulate_E_var": [free_acc["E_var_mle"][i] for i in idx],
        "free_accumulate_risk": [free_acc["risk_mle"][i] for i in idx],
    }

    # ------------------------------------------------------------------ PART B
    cfg = {"d": 128, "n_cohort": 5, "n_later": 150, "n_distract": 30, "n_cap": 200,
           "J_max": 8, "n_probe": 3, "spread": 0.6, "N_total": 16, "steps": 40, "lr": 0.5}
    seeds = [101, 102, 103, 104, 105]
    omegas = [0.0, 0.125, 0.25, 0.375, 0.5, 0.75]
    CHK = [10, 25, 50, 100, 150]

    def summarise(runs, tag, omega, src):
        m_acc, s_acc = agg(runs, "access")
        m_sto, _ = agg(runs, "storage")
        m_lft, s_lft = agg(runs, "lift")
        m_cap, s_cap = agg(runs, "capability")
        m_dri, _ = agg(runs, "drift")
        m_na, s_na = agg(runs, "new_acc")
        m_nm, s_nm = agg(runs, "new_margin")
        ks50, cen50 = kstar(m_acc, 0.5)
        ks25, cen25 = kstar(m_acc, 0.25)
        # 시드별 K*의 산포 (단일 시드 결론 금지)
        per_seed_ks = [kstar(x["access"], 0.5)[0] for x in runs]
        rho_acc = -np.diff(m_acc) * 100
        rho_lft = -np.diff(m_lft) * 100
        return {
            "omega": omega, "replay_source": src,
            "Kstar_access_at_0.5": ks50, "Kstar_censored": cen50,
            "Kstar_access_at_0.25": ks25, "Kstar_censored_25": cen25,
            "Kstar_per_seed": per_seed_ks,
            "Kstar_sd_across_seeds": float(np.std(per_seed_ks, ddof=0)),
            "access_at_checkpoints": {str(c): float(m_acc[c]) for c in CHK},
            "access_sd_at_checkpoints": {str(c): float(s_acc[c]) for c in CHK},
            "storage_at_checkpoints": {str(c): float(m_sto[c]) for c in CHK},
            "lift_at_checkpoints": {str(c): float(m_lft[c]) for c in CHK},
            "lift_sd_at_checkpoints": {str(c): float(s_lft[c]) for c in CHK},
            "capability_at_checkpoints": {str(c): float(m_cap[c]) for c in CHK},
            "capability_sd_at_checkpoints": {str(c): float(s_cap[c]) for c in CHK},
            "drift_at_checkpoints": {str(c): float(m_dri[c]) for c in CHK},
            "new_fact_acc_mean": float(m_na.mean()),
            "new_fact_acc_sd_across_seeds": float(s_na.mean()),
            "new_fact_margin_mean": float(m_nm.mean()),
            "new_fact_margin_sd_across_seeds": float(s_nm.mean()),
            "rho_access_pp": {"round1": float(rho_acc[0]),
                              "mean_1_10": float(rho_acc[:10].mean()),
                              "mean_last10": float(rho_acc[-10:].mean()),
                              "max": float(rho_acc.max()),
                              "argmax_round": int(np.argmax(rho_acc) + 1)},
            "rho_lift_pp": {"round1": float(rho_lft[0]),
                            "mean_1_10": float(rho_lft[:10].mean()),
                            "mean_last10": float(rho_lft[-10:].mean()),
                            "max": float(rho_lft.max()),
                            "argmax_round": int(np.argmax(rho_lft) + 1)},
            "shape_stats_access": shape_stats(m_acc),
            "shape_stats_lift": shape_stats(m_lft),
            "access_floor_round": first_below(m_acc, 0.05),
            "lift_when_access_at_floor": (
                None if first_below(m_acc, 0.05) is None
                else float(m_lft[first_below(m_acc, 0.05)])),
            "shape_fit_access": fit_shapes(m_acc[1:]),
            "shape_fit_lift": fit_shapes(m_lft[1:]),
            "access_trajectory": m_acc.tolist(),
            "lift_trajectory": m_lft.tolist(),
            "capability_trajectory": m_cap.tolist(),
        }

    sweep = {}
    for src in ("gold", "self"):
        for om in omegas:
            if om == 0.0:
                if src == "self":
                    continue
                tag, s2 = "omega=0.0|replay=none", "none"
            else:
                tag, s2 = f"omega={om}|replay={src}", src
            runs = [run_sim(cfg, s, om, s2, "broad", "broad") for s in seeds]
            sweep[tag] = summarise(runs, tag, om, s2)
    names = list(sweep.keys())
    chance = 1.0 / (cfg["n_cohort"] + cfg["n_later"] + cfg["n_distract"])

    # self-replay 타깃의 스케일 채널 제거 ablation (PART A의 2차 모멘트 발견과 연결)
    self_norm_ablation = {}
    for om in (0.25, 0.5):
        runs = [run_sim(cfg, s, om, "self", "broad", "broad", normalize_self=True) for s in seeds]
        m_acc, _ = agg(runs, "access")
        m_lft, _ = agg(runs, "lift")
        base = sweep[f"omega={om}|replay=self"]
        self_norm_ablation[str(om)] = {
            "raw_self_Kstar": base["Kstar_access_at_0.5"],
            "unitnorm_self_Kstar": kstar(m_acc, 0.5)[0],
            "raw_self_lift_at_150": base["lift_at_checkpoints"]["150"],
            "unitnorm_self_lift_at_150": float(m_lft[150]),
            "gold_Kstar_same_omega": sweep[f"omega={om}|replay=gold"]["Kstar_access_at_0.5"],
        }

    # 2x2 — 저장 방식 vs 이후 쓰기 방식 ([continual-facts-in-weights §7.2]의 설계)
    grid22 = {}
    for bs in ("narrow", "broad"):
        for bl in ("narrow", "broad"):
            runs = [run_sim(cfg, s, 0.0, "none", bs, bl) for s in seeds]
            m_acc, s_acc = agg(runs, "access")
            m_sto, _ = agg(runs, "storage")
            m_cap, _ = agg(runs, "capability")
            m_dri, _ = agg(runs, "drift")
            grid22[f"store={bs}|later={bl}"] = {
                "access_at_50": float(m_acc[50]), "access_sd_at_50": float(s_acc[50]),
                "access_at_150": float(m_acc[150]),
                "storage_at_50": float(m_sto[50]),
                "capability_at_50": float(m_cap[50]),
                "drift_at_50": float(m_dri[50]),
                "Kstar_access_at_0.5": kstar(m_acc, 0.5)[0],
            }
    g = grid22
    later_eff = 0.5 * ((g["store=narrow|later=broad"]["access_at_50"] - g["store=narrow|later=narrow"]["access_at_50"])
                       + (g["store=broad|later=broad"]["access_at_50"] - g["store=broad|later=narrow"]["access_at_50"])) * 100
    store_eff = 0.5 * ((g["store=broad|later=narrow"]["access_at_50"] - g["store=narrow|later=narrow"]["access_at_50"])
                       + (g["store=broad|later=broad"]["access_at_50"] - g["store=narrow|later=broad"]["access_at_50"])) * 100
    cap_later_eff = 0.5 * ((g["store=narrow|later=broad"]["capability_at_50"] - g["store=narrow|later=narrow"]["capability_at_50"])
                           + (g["store=broad|later=broad"]["capability_at_50"] - g["store=broad|later=narrow"]["capability_at_50"])) * 100

    # drift <-> capability 상관 ([continual-facts-in-weights §6.1]의 KL<->damage 대응)
    dcorr = {}
    for c in ("25", "50", "100"):
        dr = [sweep[n]["drift_at_checkpoints"][c] for n in names]
        cl = [sweep[n]["capability_at_checkpoints"]["10"] - sweep[n]["capability_at_checkpoints"][c]
              for n in names]
        dcorr[f"k={c}"] = {"spearman_drift_vs_capability_loss": spearman(dr, cl),
                           "n_conditions": len(names),
                           "capability_range": [min(sweep[n]["capability_at_checkpoints"][c] for n in names),
                                                max(sweep[n]["capability_at_checkpoints"][c] for n in names)]}

    # 저장-접근 해리 (그 논문 §5.1). 임계 지표가 바닥에 닿은 시점에 연속 신호가 얼마나 남았는가.
    dissoc = {}
    for n in names:
        fr = sweep[n]["access_floor_round"]
        dissoc[n] = {
            "access_floor_round": fr,
            "lift_when_access_at_floor": sweep[n]["lift_when_access_at_floor"],
            "access_150": sweep[n]["access_at_checkpoints"]["150"],
            "storage_150": sweep[n]["storage_at_checkpoints"]["150"],
            "lift_150": sweep[n]["lift_at_checkpoints"]["150"],
        }
    _lf = [v["lift_when_access_at_floor"] for v in dissoc.values()
           if v["lift_when_access_at_floor"] is not None]
    dissoc_summary = {
        "n_conditions_reaching_access_floor": len(_lf),
        "lift_at_access_floor_min": (min(_lf) if _lf else None),
        "lift_at_access_floor_max": (max(_lf) if _lf else None),
        "paper_value_for_the_same_quantity": "median 69% (bare) / 79% (study), 보정 후 57% / 67%",
        "paper_source": "[continual-facts-in-weights §5.1, p.11]",
    }

    # 단조성 통계 (censoring 때문에 K* 하나로는 판정 불가 -> 두 지표를 함께 본다)
    def _mono(src, field):
        oms = [o for o in omegas if o > 0]
        vals = [sweep[f"omega={o}|replay={src}"][field] if field != "acc150"
                else sweep[f"omega={o}|replay={src}"]["access_at_checkpoints"]["150"]
                for o in oms]
        return {"omegas": oms, "values": vals, "spearman_vs_omega": spearman(oms, vals)}

    monotone = {
        "gold_Kstar": _mono("gold", "Kstar_access_at_0.5"),
        "gold_access_at_150": _mono("gold", "acc150"),
        "self_Kstar": _mono("self", "Kstar_access_at_0.5"),
        "self_access_at_150": _mono("self", "acc150"),
        "note": ("gold의 K*는 omega>=0.25에서 censored(=150)라 단조성 판정에 쓸 수 없다. "
                 "gold는 access@150으로, self는 K*로 판정한다."),
    }
    # 새 사실 습득: 0/1 정확도는 포화. 연속 여유(margin)가 예산 거래를 드러내는가.
    all_om = [sweep[n]["omega"] for n in names]
    acq_mono = {
        "omega_values": all_om,
        "new_fact_acc": [sweep[n]["new_fact_acc_mean"] for n in names],
        "new_fact_margin": [sweep[n]["new_fact_margin_mean"] for n in names],
        "spearman_omega_vs_acc": spearman(all_om, [sweep[n]["new_fact_acc_mean"] for n in names]),
        "spearman_omega_vs_margin": spearman(all_om, [sweep[n]["new_fact_margin_mean"] for n in names]),
    }

    gold_vs_self = {}
    for om in omegas:
        if om == 0.0:
            continue
        gk, sk = f"omega={om}|replay=gold", f"omega={om}|replay=self"
        gold_vs_self[str(om)] = {
            "gold_Kstar": sweep[gk]["Kstar_access_at_0.5"],
            "self_Kstar": sweep[sk]["Kstar_access_at_0.5"],
            "gold_over_self_Kstar_ratio": (sweep[gk]["Kstar_access_at_0.5"]
                                           / max(sweep[sk]["Kstar_access_at_0.5"], 1)),
            "gold_access_150": sweep[gk]["access_at_checkpoints"]["150"],
            "self_access_150": sweep[sk]["access_at_checkpoints"]["150"],
            "gold_minus_self_access_150_pp": (sweep[gk]["access_at_checkpoints"]["150"]
                                              - sweep[sk]["access_at_checkpoints"]["150"]) * 100,
            "gold_capability_100": sweep[gk]["capability_at_checkpoints"]["100"],
            "self_capability_100": sweep[sk]["capability_at_checkpoints"]["100"],
        }

    nr = sweep["omega=0.0|replay=none"]
    ks_by_omega_gold = {str(o): sweep[f"omega={o}|replay=gold"]["Kstar_access_at_0.5"]
                        for o in omegas if o > 0}
    ks_by_omega_self = {str(o): sweep[f"omega={o}|replay=self"]["Kstar_access_at_0.5"]
                        for o in omegas if o > 0}

    # ------------------------------------------------------------------ 조립
    R = {
        "meta": {
            "id": "X4-consolidation-decay",
            "title": "반복 consolidation의 열화율 rho — (U-Theta)를 K라운드 돌리면 무엇이 어떻게 무너지는가",
            "grade": "EXPLORATION-GRADE / MECHANISM-TRANSFER (LLM 실험 아님)",
            "supports_chapters": ["ch29 system·infra와 memory-device 기회", "ch30 한계와 반증 조건"],
            "deterministic": True,
            "randomness": (f"PART A 몬테카를로 seeds={MC_SEEDS} (reps={MC_REPS}/seed); "
                           f"PART B seeds={seeds}. 전부 np.random.default_rng 고정 시드. "
                           "BLAS 스레드 1로 고정(numpy import 이전). 보고값 6자리 반올림. "
                           "wall-clock은 result.json에 넣지 않음. "
                           "PYTHONHASHSEED=0으로 2회 실행해 sha256 동일 확인."),
            "no_gpu": "이 host에 NVIDIA GPU 없음. numpy만 사용(torch/transformers 미사용).",
            "why_this_model": {
                "gaussian_location_scale": (
                    "[model-collapse]의 정리는 선형회귀에서 나왔다. 설계행렬이 아예 없는 문제에서도 같은 "
                    "함수형이 나오면 그 결과는 회귀의 성질이 아니라 데이터 스케줄의 성질이다. "
                    "또 가우시안은 2차 모멘트가 자유 변수라, 그 정리가 가정으로 고정해 버린 좌표를 열어 볼 수 있다."),
                "linear_associative_memory": (
                    "[continual-facts-in-weights]의 루프는 adapter를 merge하는 순차 (U-Theta)다. "
                    "그 구조를 보존하는 최소 모형이 키->값 사상의 순차 덮어쓰기이며, "
                    "그 논문이 실제로 측정한 네 축(access/storage/capability/drift)이 전부 정의된다."),
                "what_it_cannot_model": (
                    "합성·함의. 선형 사상에는 그 기제가 없다. 그래서 creation(entailment gap) 절반은 "
                    "다루지 않고 survival 절반만 다룬다."),
            },
            "budget_matched": ("omega를 쓸어도 라운드당 항목 수 N_total=16과 경사 스텝 S=40을 고정한다. "
                               "[continual-facts-in-weights App. A.3]의 'steps are matched, not tokens' 규율."),
            "sim_config": cfg,
            "sim_chance_level": chance,
            "sourced_forms": {
                "model_collapse_replace": {
                    "form": "L_test(k) = (sigma^2 d/(T-d-1)) * k",
                    "source": "[model-collapse Eq. 4; App. E Eq. 9] — Dohmatob et al. 2024a 결과의 재서술"},
                "model_collapse_replace_multiple": {
                    "form": "L_test(k) = (sigma^2 d/(T-d-1)) * sum_{i<=k} 1/i ~ log k",
                    "source": "[model-collapse Eq. 11, App. E]"},
                "model_collapse_accumulate": {
                    "form": "L_test(k) = (sigma^2 d/(T-d-1)) * sum_{i<=k} 1/i^2 <= (..) * pi^2/6 ~ 1.6449",
                    "source": "[model-collapse Eq. 3, Theorem 2]",
                    "conditions": "T >= d+2, Sigma = I, ridgeless, well-specified, 공변량 고정, 라운드마다 분산 sigma^2 재주입"},
                "model_collapse_rho": {
                    "form": "rho_k ∝ 1 / (1/k) / (1/k^2) for replace / replace-multiple / accumulate",
                    "source": "위 셋의 1차 차분 — model-collapse 노트 M-10 (note-writer 산술)"},
                "model_collapse_vae_failure": {
                    "value": "accumulate에서도 VAE test error는 증가하고 생성 다양성이 감소(안경·액세서리 소실). 캡션은 'Avoids Model Collapse'",
                    "source": "[model-collapse §2.3, L532-538; Fig. 5 caption L505-508]"},
                "cfw_retention_plateau": {
                    "value": "k=100에서 study 유지율이 0이 아니라 25-28%로 포화",
                    "source": "[continual-facts-in-weights §4.3, p.9-10]"},
                "cfw_later_write_effect": {
                    "value": "later-write effect +37.6 pp / storage-method effect -2.4 pp / 상호작용 +2.4 pp",
                    "source": "[continual-facts-in-weights §7.2, p.17-18]"},
                "cfw_frozen_vs_own_merges": {
                    "value": "frozen-teacher +2 pp capability, KL 0.48, 54% 유지 / own-merges -31 pp, KL 1.70, 21% 유지; 쌍대차 33.0 pp",
                    "source": "[continual-facts-in-weights §6.2, p.14-15]"},
                "cfw_kl_damage_corr": {
                    "value": "capability damage vs KL from Theta_0: rho_S = 0.83 (12조건) / 0.946 (400문항 세트)",
                    "source": "[continual-facts-in-weights §6.1, p.14]"},
                "cfw_storage_vs_access": {
                    "value": "k=20에서 strict 전부 실패한 사실도 log-prob lift의 median 69%(bare)/79%(study) 유지. 소거 바닥에 닿은 사실 없음",
                    "source": "[continual-facts-in-weights §5.1, p.11]"},
                "cfw_consolidation_dissociation": {
                    "value": "20라운드마다 frozen-anchor consolidation: capability는 회복(+12 pp @k=100), 유지율은 회복 실패(25% vs 28%)",
                    "source": "[continual-facts-in-weights §4.3, p.10]"},
                "cfw_kl_penalty": {
                    "value": "lambda_KL=1.0에서 capability -66 -> -5 pp, 유지율 1% -> 36%. 단 Theta_0로부터의 KL은 1.8-2.4로 그대로",
                    "source": "[continual-facts-in-weights §6.3, p.15]"},
            },
            "absent_from_corpus": {
                "rho_as_a_rate": "라운드당 열화율을 비율로 보고한 논문 0편. 하나는 닫힌 형태만, 하나는 곡선·종점만.",
                "half_life": "[continual-facts-in-weights]가 명시: 감쇠 상수 없음, 반감기 없음, k=100 너머 외삽 없음.",
                "K_beyond_100": "corpus에서 sleep 라운드를 100회 넘게 돌린 논문 0편.",
                "replay_ratio_axis": "Theta-경로 논문 중 replay 혼합비를 축으로 쓸어 본 논문 0편.",
                "second_moment_of_generator": "생성 분포의 퍼짐(다양성)을 라운드 함수로 정량 보고한 논문 0편. [model-collapse]는 그림과 문장으로만.",
            },
            "caveat_tags": {
                "NOT-AN-LLM-EXPERIMENT": "가우시안 추정 + 선형 연상기억이다. 절대치 이송 불가. 부호·함수형·순서·crossover만 이송한다.",
                "EXACT-WITHIN-THE-TOY": "PART A의 유도는 근사가 아니라 정확하다. 단 그 정확성은 가우시안 모형 안에서만 참이다.",
                "SURVIVAL-HALF-ONLY": "선형 사상에 합성·함의가 없다. [continual-facts-in-weights]의 survival 절반만 다루고 creation 절반은 다루지 않는다.",
                "MECHANISM-EXCLUSION-NOT-CONFIRMATION": "이 모형이 재현하지 '못한' 효과는 그 효과가 일반적 선형 간섭이 아님을 뜻할 뿐, 원 논문을 반박하지 않는다.",
                "MC-HORIZON-SHORTER-THAN-ANALYTIC": f"몬테카를로 검증은 k<= {K_MC}까지다. replace의 분산은 곱셈 과정이라 꼬리가 무거워 큰 k에서 평균 추정 자체가 어렵다(MC 오차이지 유도 오차가 아니다). 해석식은 k={K_A}까지 보고한다.",
                "SHAPE-FIT-IS-MODEL-SELECTION": "SSE 비교는 네 후보 안에서의 선택이다. 다른 함수형이 더 맞을 수 있다.",
                "KSTAR-CENSORED": "일부 조건은 K=150 안에서 임계를 넘지 않는다. 그 K*는 하한이며 '>=150'으로 읽어야 한다.",
                "OMEGA-IS-NOT-A-TOKEN-BUDGET": "omega는 항목 수 비율이다. 실제 시스템의 토큰·FLOPs 예산과 같지 않다.",
                "NO-CAPACITY-CEILING": "선형 사상에는 [lm-memorization-capacity]식 bits/param 상한이 없다. 여기서 관측된 포화·절벽의 원인이 LLM에서와 같다고 말할 수 없다.",
                "CAPABILITY-COUPLED-TO-RETENTION": "이 모형에서 capability와 사실 유지는 같은 스케일(~sqrt(M/d))로 함께 무너진다. 원 논문은 둘을 해리 가능한 것으로 측정한다.",
            },
        },

        "A_analytic_gaussian": {
            "setup": {"K": K_A, "n_samples_per_round": n_A, "sigma2": sigma2,
                      "risk": "E[(mu_hat_k - mu)^2] — 위치 좌표 초과 제곱오차",
                      "estimator": "표본평균 + MLE 표본분산(ddof=0). 불편 분산은 별도 행으로 대비."},
            "A1_variance_pinned": A_table,
            "A1_form_match_vs_model_collapse": {
                "replace": "L(k) = (sigma^2/n) k         <-> [Eq. 4]  (sigma^2 d/(T-d-1)) k        : 함수형 동일, 부호 동일",
                "replace_multiple": "L(k) = (sigma^2/n) H_k    <-> [Eq. 11] (..) sum 1/i         : 함수형 동일, 부호 동일",
                "accumulate": "L(k) = (sigma^2/n) sum 1/i^2  <-> [Eq. 3]  (..) sum 1/i^2         : 함수형 동일, 부호 동일, 상수 pi^2/6 동일",
                "sign_disagreement": "없음. 세 스케줄 모두 부호·함수형 일치.",
                "constant_differs_and_why": (
                    "배율이 sigma^2/n 대 sigma^2 d/(T-d-1)로 다르다. 후자의 (T-d-1)은 "
                    "inverse-Wishart 기댓값 [model-collapse Lemma 3, Eq. 5]에서 오는 설계행렬 인자이고, "
                    "위치 모형에는 설계행렬이 없어 그 인자가 없다. 배율이 다른 것은 결함이 아니라 "
                    "'함수형은 스케줄이 정하고 배율은 문제가 정한다'는 결과다."),
            },
            "A2_variance_free": {
                "replace_var_geometric_collapse": (
                    "E[sigma2_hat_k] = sigma^2 ((n-1)/n)^k — 기하 붕괴. "
                    f"반감기 {free_rep['var_half_life_rounds']:.4f} 라운드 (= ln2/ln(n/(n-1)) ~ 0.693 n)."),
                "replace_risk_saturates": (
                    "MLE 분산이면 위치 위험이 sigma^2 (1 - ((n-1)/n)^k) 로 **포화**한다. "
                    "발산이 아니라 포화가 나오는 이유는 생성 분포가 죽어 스텝이 작아지기 때문이다."),
                "unbiased_variance_restores_linear_divergence": (
                    "불편 분산이면 E[sigma2_hat_k] = sigma^2 (마팅게일)이고 위험은 정확히 (sigma^2/n) k — "
                    "[Eq. 4]와 같다. 즉 '선형 발산이냐 포화냐'는 추정량의 편향 보정이 가른다."),
                "accumulate_protects_the_second_moment": (
                    f"정확 재귀: k={K_A}에서 E[sigma2_hat] = {free_acc['E_var_mle'][-1]:.6f} "
                    f"(replace는 {free_rep['E_var_mle'][-1]:.3e}). "
                    f"비 = {float(free_acc['E_var_mle'][-1]) / float(free_rep['E_var_mle'][-1]):.4g}배. "
                    "누적은 1차 모멘트뿐 아니라 2차 모멘트도 붕괴에서 구해 낸다."),
                "accumulate_var_over_replace_var_at_K": (float(free_acc["E_var_mle"][-1])
                                                          / float(free_rep["E_var_mle"][-1])),
                "note": ("[model-collapse]의 정리는 이 절 전체를 표현할 수 없다. 그 모형은 매 라운드 "
                         "N(0, sigma^2 I_T)를 새로 주입하므로 2차 모멘트가 상수로 못 박혀 있다."),
            },
            "A3_monte_carlo_check": {
                "horizon_K": K_MC, "reps_per_seed": MC_REPS, "seeds": MC_SEEDS,
                "criterion": ("|MC평균 - 정확값| / (시드 간 표준오차)를 k마다 계산하고, "
                              "다중비교를 보정한 family-wise p = 1-(1-erfc(zmax/sqrt2))^K 로 판정한다. "
                              "p > 0.01 이면 일치. 개별 3sigma를 쓰면 60회 비교에서 우연히 걸린다."),
                "per_quantity": mc_check,
                "all_consistent": bool(all(v["consistent"] for v in mc_check.values())),
            },
        },

        "B_simulation_fact_injection": {
            "protocol": (f"cohort {cfg['n_cohort']}개를 먼저 쓰고, 이후 {cfg['n_later']}라운드 동안 "
                         "매 라운드 새 사실 1개를 순차로 쓴다(각 쓰기 후 merge). "
                         "cohort의 access/storage/lift/capability/drift를 라운드 함수로 추적. "
                         "[continual-facts-in-weights §7.2]의 'C stored facts + K later writes' 설계와 같은 형태."),
            "B1_omega_sweep": sweep,
            "B2_Kstar_by_omega": {
                "definition": "K* = cohort access가 0.5 이상을 유지한 마지막 라운드(임계 crossover). 150은 censored(하한).",
                "no_replay": nr["Kstar_access_at_0.5"],
                "gold_replay": ks_by_omega_gold,
                "self_replay": ks_by_omega_self,
                "monotonicity": monotone,
            },
            "B3_gold_vs_self_replay": gold_vs_self,
            "B3b_new_fact_acquisition_cost": {
                "what": ("예산 정합(N_total, steps 고정) 하에서 omega를 올리면 새 사실 습득이 손해를 보는가. "
                         "0/1 정확도와 연속 여유(정답 cos - 최대 오답 cos)를 함께 본다."),
                "cells": acq_mono,
            },
            "B4_self_replay_scale_channel_ablation": {
                "what": ("self replay의 타깃을 모델의 원 출력(raw) 대신 단위 정규화(unit-norm)로 바꾼다. "
                         "이는 출력 **스케일(2차 모멘트) 채널만** 제거하는 조작이다. "
                         "PART A의 '분산이 죽는다'가 이 시뮬에서도 손해의 일부인지 확인한다."),
                "cells": self_norm_ablation,
            },
            "B5_breadth_2x2": {
                "design": "[continual-facts-in-weights §7.2]의 2x2 (저장 방식 x 이후 쓰기 방식), omega=0",
                "cells": grid22,
                "later_write_effect_pp_at_k50": later_eff,
                "storage_method_effect_pp_at_k50": store_eff,
                "capability_later_write_effect_pp_at_k50": cap_later_eff,
                "paper_reference": "[continual-facts-in-weights §7.2]: later +37.6 pp, storage -2.4 pp (broad=study가 유리한 방향)",
            },
            "B6_storage_vs_access_dissociation": {
                "definition": ("access = 학습에 안 쓴 프로브로 디코딩(임계 지표), "
                               "storage = 학습에 쓴 정본 키로 디코딩(역시 임계), "
                               "lift = 쓰기가 만든 신호 중 남은 비율(연속). "
                               "핵심 수치는 access가 바닥(<=0.05)에 닿은 라운드에서의 lift다 — "
                               "그 논문 §5.1이 재는 것과 같은 양."),
                "cells": dissoc,
                "summary": dissoc_summary,
            },
            "B7_drift_capability": {
                "by_checkpoint": dcorr,
                "paper_reference": "[continual-facts-in-weights §6.1]: KL vs damage rho_S = 0.83 / 0.946",
            },
        },

        "C_verdict": {
            "C0_model_free_shape_statistics": {
                "why": ("네 후보의 SSE 차이가 작아 모형 선택은 결정적이지 않다. "
                        "그래서 판정은 모형에 무관한 통계로 한다: grace 구간과 rho의 앞구간/최댓값."),
                "access": {n: sweep[n]["shape_stats_access"] for n in names},
                "lift": {n: sweep[n]["shape_stats_lift"] for n in names},
            },
            "C1_rho_shape_access": {
                n: {"best": sweep[n]["shape_fit_access"]["best"],
                    "r2_linear": sweep[n]["shape_fit_access"]["linear"]["r2"],
                    "r2_exp": sweep[n]["shape_fit_access"]["exp"]["r2"],
                    "r2_plateau": sweep[n]["shape_fit_access"]["plateau"]["r2"],
                    "r2_logistic": sweep[n]["shape_fit_access"]["logistic"]["r2"],
                    "logistic_k0": sweep[n]["shape_fit_access"]["logistic"]["k0"],
                    "logistic_tau": sweep[n]["shape_fit_access"]["logistic"]["tau"]}
                for n in names},
            "C2_rho_shape_lift_continuous": {
                n: {"best": sweep[n]["shape_fit_lift"]["best"],
                    "r2_linear": sweep[n]["shape_fit_lift"]["linear"]["r2"],
                    "r2_exp": sweep[n]["shape_fit_lift"]["exp"]["r2"],
                    "r2_plateau": sweep[n]["shape_fit_lift"]["plateau"]["r2"],
                    "r2_logistic": sweep[n]["shape_fit_lift"]["logistic"]["r2"],
                    "plateau_c": sweep[n]["shape_fit_lift"]["plateau"]["c"],
                    "plateau_lam": sweep[n]["shape_fit_lift"]["plateau"]["lam"]}
                for n in names},
            "C3_rho_is_not_a_rate_for_the_task_metric": {
                "no_replay_rho_access_pp": nr["rho_access_pp"],
                "no_replay_rho_lift_pp": nr["rho_lift_pp"],
                "reading": ("과제 지표(access)의 rho는 라운드에 대해 상수가 아니고 특정 라운드에 몰린다. "
                            "연속 신호(lift)의 rho는 매끄럽다. 같은 루프의 rho가 **무엇을 재느냐에 따라 모양이 다르다.**"),
            },
        },
    }

    # ------------------------------------------------------------------ findings
    nr_sa = nr["shape_stats_access"]
    nr_sl = nr["shape_stats_lift"]
    ratio_txt = ", ".join(f"{o}: {v['gold_over_self_Kstar_ratio']:.2f}x"
                          for o, v in gold_vs_self.items())
    var_ratio = float(free_acc["E_var_mle"][-1]) / float(free_rep["E_var_mle"][-1])

    R["findings"] = {
        "LOAD_BEARING_본문_단정문으로_쓸_것": {
            "F1_the_1_over_i_weighting_is_a_property_of_the_schedule": (
                "[model-collapse]의 세 함수형(replace ~ k, replace-multiple ~ log k, "
                "accumulate ~ sum 1/i^2 <= pi^2/6)은 **설계행렬이 아예 없는** 가우시안 위치 모형에서 "
                "그대로 재유도된다. 부호 불일치 없음. 배율만 다르고(sigma^2/n 대 sigma^2 d/(T-d-1)) "
                "그 차이는 inverse-Wishart 인자 [Lemma 3]가 위치 모형에 없기 때문이다. "
                "따라서 1/i 가중은 선형회귀의 성질이 아니라 **데이터 스케줄의 성질**이고, "
                "그 결론을 Theta-경로 일반으로 옮기는 것은 그 논문 자체가 허용하는 것보다 넓게 정당화된다. "
                f"라운드 {A_table['checkpoints_k'][-1]}에서 replace/accumulate 위험비는 "
                f"{A_table['ratio_replace_over_accumulate'][-1]:.1f}배다(비율이므로 이송 가능)."),
            "F2_the_theorem_cannot_see_the_coordinate_that_actually_fails": (
                "그 정리는 라운드마다 분산 sigma^2를 상수로 재주입하므로 **2차 모멘트를 원리적으로 볼 수 없다.** "
                "그 가정을 풀면 replace에서 E[sigma2_hat_k] = sigma^2 ((n-1)/n)^k 로 기하 붕괴하고 "
                "반감기는 ~0.693 n 라운드다. accumulate는 2차 모멘트도 지켜 낸다"
                f"(k={K_A}에서 비 {var_ratio:.3g}배). "
                "그리고 바로 그 좌표가 [model-collapse] 자신의 VAE 실험이 실패한 좌표다 — "
                "손실은 (거의) 유계인데 다양성이 사라진다 [§2.3 L532-536]. "
                "즉 '유계'는 1차 모멘트의 성질이고, 실제로 죽는 것은 2차 모멘트다. "
                "이 진술은 그 논문의 그림·문장과 부호가 같고, 그 논문의 정리로는 표현할 수 없다. "
                "**그러므로 accumulate의 우월성은 그 논문이 증명한 것보다 넓다** — 1차 모멘트만이 아니라 "
                "생성 분포의 퍼짐까지 지킨다. 이것은 그 논문이 주장하지 않은 방향의 강화다."),
            "F3_a_flat_loss_curve_is_not_evidence_of_stability": (
                "MLE 분산에서는 위치 위험이 sigma^2로 **포화**한다 — 생성 분포가 죽어 스텝이 작아지기 때문이다. "
                "같은 루프를 불편 추정량으로 돌리면 (sigma^2/n)k 로 선형 발산한다. "
                "**손실 곡선이 평평해지는 것은 안정의 증거가 아닐 수 있다.** "
                "held-out perplexity만 보는 프로덕션 모니터는 이 붕괴에 발화하지 않는다. "
                "ch29의 운영 함의: 반복 sleep 루프의 모니터링 지표는 1차 모멘트가 아니라 "
                "생성 분포의 퍼짐이어야 한다. corpus에서 그것을 라운드 함수로 보고한 논문은 0편이다."),
            "F4_the_task_metric_does_not_move_while_the_store_is_already_dying": (
                f"같은 루프에서 과제 지표(access)와 연속 신호(lift)의 rho는 모양이 다르다. "
                f"replay 없음 조건에서 access는 라운드 {nr_sa['grace_round_first_below_0.95']}까지 "
                f"0.95 아래로 내려가지 않는다(앞 10라운드 rho 평균 {nr_sa['rho_mean_first10_pp']:.2f} pp) — "
                f"그런데 같은 구간에서 lift는 이미 매 라운드 {nr_sl['rho_mean_first10_pp']:.2f} pp씩 깎이고 있다. "
                f"그 뒤 access는 라운드 {nr_sa['first_below_0.9']}에서 {nr_sa['first_below_0.1']}까지 "
                f"폭 {nr_sa['fall_width_0.9_to_0.1']}라운드 만에 무너지고, rho의 최댓값 "
                f"{nr_sa['rho_max_pp']:.2f} pp는 라운드 {nr_sa['rho_argmax_round']}에 나온다. "
                "**저장은 1라운드부터 꾸준히 깎이는데 과제 지표는 한동안 전혀 움직이지 않다가 절벽으로 떨어진다.** "
                "그러므로 '라운드당 몇 %'라는 단일 비율로 rho를 보고하는 관행 자체가 오도다 — "
                "무엇을 재느냐가 모양을 바꾼다. corpus의 두 논문이 rho를 서로 다른 형태로 보고하는 것"
                "(닫힌 형태 대 유지 곡선)은 게으름이 아니라 이 구조적 사실의 반영이다. "
                "(네 후보 함수형 적합도 함께 보고하지만 SSE 차이가 작아 모형 선택은 결정적이지 않다. "
                "위 진술은 모형 선택에 의존하지 않는다.)"),
            "F5_what_must_be_reported_is_the_crossover_not_the_rate": (
                f"과제 지표가 임계로 무너지므로 이송해야 할 양은 rho가 아니라 **K*(omega)** — "
                f"성능이 기준선을 유지하는 마지막 라운드다. replay 없음에서 K*={nr['Kstar_access_at_0.5']}"
                f"(시드별 {nr['Kstar_per_seed']}, sd {nr['Kstar_sd_across_seeds']:.1f}). "
                f"self replay는 omega에 대해 K*가 단조 증가하고"
                f"(Spearman {monotone['self_Kstar']['spearman_vs_omega']:.2f}, 값 {monotone['self_Kstar']['values']}), "
                f"gold replay는 omega>=0.25에서 K*가 K=150에 censored라 access@150으로 본다"
                f"(Spearman {monotone['gold_access_at_150']['spearman_vs_omega']:.2f}). "
                "이 저장소의 규율상 crossover 위치는 이송 가능한 양이므로, ch29는 K*를 쓰고 rho를 쓰지 않는다."),
            "F6_replay_must_be_anchored_to_retained_originals": (
                "같은 omega·같은 예산에서 replay 타깃을 **보존된 원본**(gold)으로 하면 "
                "**모델 자기 출력**(self)으로 할 때보다 모든 omega에서 오래 버틴다. "
                f"K* 비(gold/self): {ratio_txt}. self는 어느 omega에서도 K=150을 넘기지 못하고 "
                "gold는 omega>=0.25에서 전부 censored다. "
                "이는 PART A의 accumulate 대 replace 대비를 사실 주입 설정에서 재현한 것이고, "
                "[continual-facts-in-weights §6.2]의 frozen-teacher(+2 pp capability, 54% 유지) 대 "
                "own-merges(-31 pp, 21% 유지, 쌍대차 33.0 pp) 부호와도 일치한다. "
                "**설계 규칙: replay는 보존된 원본에 앵커되어야 한다. 현재 모델로 replay 타깃을 다시 만들면 "
                "replace regime의 발산을 다시 불러온다.** "
                "그리고 스케일 채널만 제거하는 ablation(self 타깃을 단위 정규화)이 손해의 일부를 되돌린다 "
                f"(omega=0.5에서 K* {self_norm_ablation['0.5']['raw_self_Kstar']} -> "
                f"{self_norm_ablation['0.5']['unitnorm_self_Kstar']}) — "
                "PART A의 2차 모멘트 발견이 이 시뮬 안에서도 작동한다는 뜻이다."),
            "F7_gold_and_self_replay_buy_different_things": (
                "gold replay는 사실 유지를 사고 capability는 사지 못한다. self replay는 그 반대다 — "
                "현재 동작에 앵커하므로 capability를 지키지만 이미 잃은 내용을 되살리지 못한다. "
                f"예: omega=0.75에서 self의 k=100 capability가 "
                f"{gold_vs_self['0.75']['self_capability_100']:.3f}인 반면 gold는 "
                f"{gold_vs_self['0.75']['gold_capability_100']:.3f}이고, access는 반대로 gold가 우세하다. "
                "이 이중 해리는 [continual-facts-in-weights]가 독립적으로 관측한 두 사실과 부호가 같다 — "
                "KL 페널티(현재 base 앵커)는 capability를 -66에서 -5 pp로 되살리지만 [§6.3], "
                "frozen-anchor consolidation은 capability만 회복시키고 유지율은 회복시키지 못한다 [§4.3]. "
                "**따라서 'replay를 넣는다'는 하나의 처방이 아니라 서로 다른 것을 사는 두 처방이다.**"),
            "F8_damage_tracks_distance_from_the_original": (
                f"조건 간에서 capability 손실은 Theta_0로부터의 거리와 함께 간다 — "
                f"drift 대 capability-loss Spearman = "
                f"{dcorr['k=25']['spearman_drift_vs_capability_loss']:+.3f} (k=25), "
                f"{dcorr['k=50']['spearman_drift_vs_capability_loss']:+.3f} (k=50). "
                "부호와 크기 모두 [continual-facts-in-weights §6.1]의 rho_S=0.83(12조건)/0.946(400문항)과 같은 자리다. "
                "여기에는 언어도 KL도 없고 Frobenius 거리뿐이므로, "
                "**'원본에서 멀어진 만큼 능력을 잃는다'는 법칙은 LLM 고유 현상이 아니라 "
                "순차 파라미터 덮어쓰기의 일반 성질이다.** 그 논문의 관측을 반박하지 않고, "
                "그 관측이 특별한 설명을 요구하지 않음을 보인다. "
                f"단 k=100에서는 {dcorr['k=100']['spearman_drift_vs_capability_loss']:+.3f}로 무너지는데, "
                "그 지점에서는 조건 대부분의 capability가 바닥에 붙어 순위 자체가 무의미해지기 때문이다."),
        },

        "DIRECTIONAL_하한_방향성으로만_쓸_것": {
            "D1_absolute_numbers": ("rho의 절대 pp, 유지율 %, K*의 라운드 수, 필요 omega 값, 반감기 라운드 수는 "
                                    "전부 이 장난감 모형의 산물이다. 본문에 숫자로 옮기지 말 것. "
                                    "옮길 수 있는 것은 부호·모양·조건 간 순서·crossover의 이동 방향뿐이다."),
            "D2_no_plateau_level_correspondence": (
                "적합된 plateau 바닥값 c는 이 시뮬의 전 조건에서 **음수**로 나온다(외삽된 점근선이 0 아래). "
                "즉 이 모형에는 [continual-facts-in-weights §4.3]의 '25-28%로 포화'에 대응시킬 바닥값이 없다. "
                "**두 바닥값을 비교하지 말 것.** 대응시킬 수 있는 것은 '연속 신호는 매끄럽게, 과제 지표는 임계로 "
                "무너진다'는 모양의 차이뿐이며, 그 논문의 plateau는 이 모형이 갖지 않은 용량 상한에서 올 가능성이 크다."),
            "D3_censoring": ("gold replay의 상위 omega 조건은 K=150 안에서 임계를 넘지 않아 K*가 censored다. "
                             "'>=150'은 하한이지 '무한히 버틴다'가 아니다."),
            "D4_drift_capability_magnitude": (
                "F8의 상관은 **부호**가 load-bearing이고 크기는 아니다. Frobenius 거리는 KL이 아니며, "
                "조건 수가 11개뿐이라 Spearman의 신뢰구간이 넓다. "
                f"체크포인트 의존성도 크다(k=25/50/100에서 "
                f"{dcorr['k=25']['spearman_drift_vs_capability_loss']:+.2f}/"
                f"{dcorr['k=50']['spearman_drift_vs_capability_loss']:+.2f}/"
                f"{dcorr['k=100']['spearman_drift_vs_capability_loss']:+.2f})."),
            "D5_acquisition_cost_is_real_but_sub_threshold": (
                f"예산 정합 하에서 omega를 올려도 새 사실 습득 **정확도**는 전 조건 1.000으로 꿈쩍하지 않는다"
                f"(정확도가 전부 동일해 Spearman은 정의되지 않는다: {acq_mono['spearman_omega_vs_acc']}). "
                f"그러나 같은 예측의 **연속 여유**는 단조로 줄어든다"
                f"({max(acq_mono['new_fact_margin']):.3f} -> {min(acq_mono['new_fact_margin']):.3f}, "
                f"Spearman(omega, margin) = {acq_mono['spearman_omega_vs_margin']:.2f}). "
                "즉 '유지율을 사려면 습득을 지불한다'는 거래는 **존재하지만 임계 지표로는 보이지 않는다** — "
                "이것은 F4와 같은 현상이 습득 축에서 되풀이된 것이다. "
                "본문에서 이 거래를 정량으로 주장하지 말 것. 방향만 쓸 수 있다."),
        },

        "MECHANISM_EXCLUSION_최소모형이_재현하지_못한_것": {
            "X1_later_write_breadth_effect_has_the_opposite_sign": (
                f"2x2에서 later-write 효과는 {later_eff:+.2f} pp, storage-method 효과는 {store_eff:+.2f} pp였다. "
                "비대칭의 **방향**(이후 쓰기가 지배하고 저장 방식은 거의 무관)은 "
                "[continual-facts-in-weights §7.2]의 +37.6 / -2.4 pp와 같지만, "
                "later-write 효과의 **부호**는 반대다 — 이 모형에서는 broad한 이후 쓰기가 더 크게 파괴한다"
                f"(capability도 같은 방향: {cap_later_eff:+.2f} pp). "
                "**해석: 그 논문이 보고한 'narrow한 이후 쓰기가 더 파괴적'이라는 결과는 일반적 선형 덮어쓰기로 "
                "설명되지 않는다.** 좁은 집합 위의 과최적화가 recitation 같은 퇴화 해로 모델을 끌고 가는 "
                "LLM 고유 기제가 필요하다. 반면 '**간섭은 들어오는 쓰기가 만들고 저장 방식은 거의 무관하다**'는 "
                "비대칭 자체는 선형 간섭만으로도 나온다 — 이 부분은 그 논문의 결과를 설명하는 데 "
                "특별한 기제가 필요 없다는 뜻이다."),
            "X2_capability_and_retention_are_coupled_here": (
                "이 모형에서 capability 손상과 사실 망각은 같은 스케일(~sqrt(M/d))로 함께 무너져 해리되지 않는다. "
                "[continual-facts-in-weights §4.3]은 frozen-anchor consolidation이 capability만 회복시키고 "
                "유지율은 회복시키지 못하는 해리를 측정한다. **그 해리는 순차 덮어쓰기의 일반 성질이 아니다.**"),
            "X3_storage_access_dissociation_reproduces_in_direction_but_not_in_size": (
                "저장-접근 해리는 **의미론이 전혀 없는 이 모형에서도 방향으로는 재현된다**: "
                "과제 지표가 바닥(<=0.05)에 닿은 라운드에서도 쓰기가 만든 신호(lift)는 0이 아니라 "
                f"{dissoc_summary['lift_at_access_floor_min']:.3f}~{dissoc_summary['lift_at_access_floor_max']:.3f} 남아 있다. "
                "[continual-facts-in-weights §5.1]은 같은 양을 median 69%(bare)/79%(study), "
                "보정 후 57%/67%로 보고한다 — **부호는 같고 크기는 이 모형이 훨씬 작다.** "
                "그러므로 이 실험이 뒷받침하는 것은 좁은 진술뿐이다: "
                "'저장은 남았는데 접근이 실패한다'는 패턴 자체는 매끄럽게 감쇠하는 저장 위에 "
                "임계 판독을 얹으면 자동으로 생기므로, 그 패턴의 **존재**는 특별한 접근 기제의 증거가 아니다. "
                "그러나 그 논문이 보고한 **크기**(거의 소거되지 않음)는 이 기제로 설명되지 않으며, "
                "그 논문의 'nothing to search' 해석을 반박하지도 지지하지도 않는다."),
        },

        "F9_what_theta_path_needs_to_survive_K_rounds": (
            "다섯 조건이다. "
            "(1) **원본을 버리지 말 것.** 유계성은 실데이터가 코퍼스에 남아 있는 데서 나온다. "
            "합성만 늘리는 것(replace-multiple)은 발산을 선형에서 로그로 낮출 뿐 유계로 만들지 못한다 [Eq. 11]. "
            "(2) **replay 타깃을 현재 모델로 다시 만들지 말 것.** self replay는 replace regime이고, "
            "같은 예산에서 gold replay보다 항상 먼저 무너진다. "
            "(3) **1차 모멘트만 모니터링하지 말 것.** 손실이 평평해도 생성 분포의 2차 모멘트가 죽고 있을 수 있으며, "
            "그 좌표는 [model-collapse]의 정리로는 보이지 않고 그 논문의 VAE 실험이 바로 거기서 실패했다. "
            "(4) **과제 지표만 모니터링하지 말 것.** 과제 지표는 저장이 절반 넘게 깎일 때까지 움직이지 않다가 "
            "절벽으로 떨어진다. 경보가 울릴 때는 이미 늦다. 연속 신호(쓰기가 만든 lift의 잔존률)를 함께 봐야 한다. "
            "(5) **rho가 아니라 K*를 SLA로 쓸 것.** '라운드당 몇 %'는 운영 지표가 될 수 없다. "
            "그리고 이 다섯을 다 지켜도 K*는 유한하며, 서비스가 요구하는 K를 넘는지는 corpus의 어느 논문도 답하지 않는다 "
            "— sleep 라운드를 100회 넘게 돌린 논문이 0편이기 때문이다. "
            "**이것이 ch30의 반증 조건이다: Theta-경로를 지지하려면 K>100에서 원본 보존 replay와 함께 "
            "K*를 측정한 실험이 필요하고, 현재 corpus에는 없다.**"),

        "honest_limits": (
            "이것은 LLM 실험이 아니다. 가우시안 추정 문제와 선형 연상기억이다. "
            "PART A의 유도는 정확하지만 그 정확성은 모형 안에서만 참이다. "
            "PART B는 [continual-facts-in-weights]의 survival 절반만 재현하고 creation(entailment gap) 절반은 "
            "재현하지 않는다 — 선형 사상에 합성이 없기 때문이다. 그리고 재현하지 못한 두 효과"
            "(later-write breadth의 부호, capability-retention 해리)는 원 논문의 반박이 아니라 "
            "'그 효과는 일반적 선형 간섭이 아니다'라는 기제 배제로만 읽어야 한다. "
            "본문 단정문으로 쓸 것은 함수형·부호·조건 간 순서·crossover의 존재와 이동 방향이며, "
            "절대 rho, 유지율 %, K*의 라운드 수, 필요 omega는 아니다."),
    }

    out = HERE / "result.json"
    payload = json.dumps(r(R), ensure_ascii=False, indent=2)
    out.write_text(payload, encoding="utf-8")
    sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    print(f"wrote {out}")
    print(f"sha256(result.json) = {sha}")
    print(f"runtime = {time.time() - t0:.2f} s  (result.json에 기록하지 않음 — bit-identity 보존)\n")

    print("== PART A-1: 분산 고정 (model-collapse의 가정) ==")
    for j, kk in enumerate(A_table["checkpoints_k"]):
        print(f"  k={kk:4d}  replace={A_table['pinned_replace'][j]:10.5f}  "
              f"rep-mult={A_table['pinned_replace_multiple'][j]:8.5f}  "
              f"accum={A_table['pinned_accumulate'][j]:8.5f}  "
              f"(bound {A_table['pinned_accumulate_bound_pi2_over_6']:.5f}; "
              f"rep/acc={A_table['ratio_replace_over_accumulate'][j]:7.1f}x)")
    print("== PART A-2: 분산 자유 ==")
    print(f"  replace    E[var] k={K_A} = {free_rep['E_var_mle'][-1]:.6e}   반감기 {free_rep['var_half_life_rounds']:.2f} 라운드")
    print(f"  accumulate E[var] k={K_A} = {free_acc['E_var_mle'][-1]:.6f}")
    print(f"  replace risk k={K_A} = {free_rep['risk_mle'][-1]:.6f}  ->  포화 한계 {sigma2}")
    print(f"== PART A-3: 몬테카를로 대조 (k<={K_MC}, reps={MC_REPS}x{len(MC_SEEDS)}seeds, "
          f"family-wise p 판정) ==")
    for k2, v in mc_check.items():
        print(f"  {k2:20s} max|z|={v['max_abs_z_over_k']:5.2f}  med|z|={v['median_abs_z']:4.2f}  "
              f"p_fw={v['familywise_p_of_max_z']:.3f}  "
              f"mc={v['mc_at_kmc']:.6g} +- {v['mc_se_at_kmc']:.2g}  exact={v['exact_at_kmc']:.6g}  "
              f"{'OK' if v['consistent'] else 'MISMATCH'}")
    print(f"\n== PART B: omega sweep (5 seeds, chance={chance:.4f}) ==")
    for n in names:
        s = sweep[n]
        sa = s["shape_stats_access"]
        print(f"  {n:24s} K*={s['Kstar_access_at_0.5']:3d}{'(cens)' if s['Kstar_censored'] else '      '}"
              f" sd={s['Kstar_sd_across_seeds']:4.1f} grace={str(sa['grace_round_first_below_0.95']):>4s}"
              f" rho10={sa['rho_mean_first10_pp']:5.2f} rhomax={sa['rho_max_pp']:5.2f}@{sa['rho_argmax_round']:3d}"
              f" | acc@150={s['access_at_checkpoints']['150']:.2f}"
              f" lift@150={s['lift_at_checkpoints']['150']:.2f}"
              f" cap@100={s['capability_at_checkpoints']['100']:+.3f}"
              f" margin={s['new_fact_margin_mean']:.3f}")
    print(f"  lift가 1~10라운드에 이미 깎이는 양(replay 없음): "
          f"{nr['shape_stats_lift']['rho_mean_first10_pp']:.2f} pp/round  "
          f"vs access {nr['shape_stats_access']['rho_mean_first10_pp']:.2f} pp/round")
    print(f"  access 바닥 도달 시 남은 lift: "
          f"{dissoc_summary['lift_at_access_floor_min']:.3f}~{dissoc_summary['lift_at_access_floor_max']:.3f} "
          f"(논문 대응값 69-79%)")
    print(f"  습득: acc Spearman(omega)={acq_mono['spearman_omega_vs_acc']} (전부 동순위), "
          f"margin Spearman(omega)={acq_mono['spearman_omega_vs_margin']:.2f} "
          f"({max(acq_mono['new_fact_margin']):.3f}->{min(acq_mono['new_fact_margin']):.3f})")
    print("== PART B: 2x2 breadth (k=50) ==")
    for n, v in grid22.items():
        print(f"  {n:28s} acc={v['access_at_50']:.2f}(sd {v['access_sd_at_50']:.2f}) "
              f"sto={v['storage_at_50']:.2f} cap={v['capability_at_50']:+.3f} K*={v['Kstar_access_at_0.5']}")
    print(f"  later-write effect = {later_eff:+.2f} pp | storage-method effect = {store_eff:+.2f} pp "
          f"| capability later-effect = {cap_later_eff:+.2f} pp")
    print(f"  drift<->cap-loss Spearman: " + ", ".join(
        f"{c}={v['spearman_drift_vs_capability_loss']:+.3f}" for c, v in dcorr.items()))
    print("== PART B: self-replay 스케일 채널 ablation ==")
    for om, v in self_norm_ablation.items():
        print(f"  omega={om}: raw K*={v['raw_self_Kstar']:3d}  unit-norm K*={v['unitnorm_self_Kstar']:3d}  "
              f"gold K*={v['gold_Kstar_same_omega']:3d}")


if __name__ == "__main__":
    main()
