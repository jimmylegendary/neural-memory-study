#!/usr/bin/env python3
"""X1 — 통일 비용 회계: sleep-time compute 상각 논증의 유효 범위를 계산한다.

배경
----
E-경로의 창시 논문(Letta, arXiv:2504.13171)은 질의당 평균 비용을

    C_avg = kappa_cost * B_t + B_s / N_q                                   (paper)

로 놓고, sleep-time 예산 B_s를 N_q개 질의에 상각하면 이득이라고 논증한다.
이 식에는 **생성 토큰만** 들어간다. 논문의 모든 그림 x축이 "Avg. Test Time
Tokens / Question"이고, 입력(prefill) 토큰은 어떤 항에도 나타나지 않는다.

그런데 이 방법의 기제 자체가 **입력을 늘리는 것**이다. learned context c-hat은
매 질의의 프롬프트에 붙고, 원 문맥은 버려지지 않는다. 병렬 스케일링에서는 최대
10벌이 concatenate된다 [Letta STC §5.2].

즉 논문의 회계는 한 번 내는 비용(B_s)은 세고, **매 질의마다 내는 비용**(c-hat의
prefill)은 세지 않는다. 전자는 N_q로 나뉘고 후자는 나뉘지 않으므로, 후자가 있으면
"N_q를 키우면 언제나 이긴다"가 성립하지 않는 영역이 생긴다.

이 실험이 묻는 것은 "5x가 거짓인가"가 아니다. 5x는 생성 토큰 축에서 관측된 사실이다.
묻는 것은 **그 축이 비용 축과 같다는 암묵 등식이 어느 범위에서 유효한가**이다.

무엇을 계산하는가
----------------
Q1  임계 길이 L*  — c-hat이 이보다 길면 N_q를 아무리 키워도 baseline을 이길 수 없다.
                    논문의 회계(w_in=0)에서는 L* = 무한대라 이 경계가 표현되지 않는다.
Q2  레짐별 여유   — 논문이 보고한 세 실험(GSM-Symbolic / AIME / SWE-Features)에서
                    L*가 각각 얼마이고, 그 값이 현실적인 c-hat 길이에 비해 큰가 작은가.
Q3  손익분기 N_q* — 임계 안쪽에서 몇 개 질의부터 이득으로 돌아서는가.
Q4  민감도        — kappa_par(병렬 벌수), kappa_cost(논문이 벤더 문서로 고정한 상수),
                    w_in(입력 토큰 가중치)에 대한 L*의 민감도.

등급
----
EXPLORATION-GRADE / ACCOUNTING. 이 실험은 측정이 아니라 회계다.
- 신뢰할 것: 임계의 존재, 레짐 간 여유의 **순서**, 민감도의 방향.
- 신뢰하지 말 것: 절대 비용, 특정 벤더 청구액, L*의 소수점.
- 원 논문이 |c|, |c-hat|, B_s를 보고하지 않으므로 이 셋은 sweep한다. 단일 정답이 없다.

이 실험은 STC에 **유리한 쪽으로** 가정을 몰아 놓았다(§ASSUMPTIONS). 그럼에도 임계가
존재한다는 것이 결과다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 회계 모델
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Accounting:
    """비용 단위 = 'sleep-time 생성 토큰 1개'.

    w_in       : 입력(prefill) 토큰 1개의 상대 비용.
                 FLOPs 관점에서는 dense 모델 기준 prefill·decode 모두 토큰당 약 2N FLOPs이므로
                 w_in = 1. 과금 관점에서는 공급자가 입력을 싸게 매기므로 0.2~0.3이 흔하다.
                 논문은 w_in = 0 을 암묵 가정한 셈이다.
    kappa_cost : wake 생성 토큰이 sleep 생성 토큰보다 비싼 배수.
                 논문은 10으로 고정했고 근거는 벤더 문서 링크 하나다 [Letta STC §5.3 fn.4].
    """

    w_in: float
    kappa_cost: float = 10.0
    label: str = ""


def wake_per_query(acc, len_c, len_q, len_chat, gen, kappa_par, keep_raw=True):
    """매 질의마다 내는 비용. N_q로 상각되지 않는다."""
    prefill = (len_c if keep_raw else 0.0) + kappa_par * len_chat + len_q
    return acc.kappa_cost * gen + acc.w_in * prefill


def sleep_once(acc, len_c, B_s_gen, n_rewrites):
    """한 번만 내는 비용. N_q로 상각된다."""
    return B_s_gen + acc.w_in * (n_rewrites * len_c)


def critical_len_chat(acc, len_c, len_q, gen_base, gen_stc, kappa_par, keep_raw=True):
    """c-hat이 이보다 길면 상각이 원리적으로 불가능해지는 임계 길이 L*.

    조건: wake_per_query(STC) < wake_per_query(baseline)
      kappa_cost*gen_stc + w_in*(len_c + kappa_par*L + len_q)
        < kappa_cost*gen_base + w_in*(len_c + len_q)
      => w_in * kappa_par * L < kappa_cost * (gen_base - gen_stc)
    w_in = 0 이면 임계가 없다(무한대) — 이것이 논문의 회계다.
    """
    if acc.w_in <= 0:
        return float("inf")
    return acc.kappa_cost * (gen_base - gen_stc) / (acc.w_in * kappa_par)


def breakeven_Nq(acc, len_c, len_q, len_chat, gen_base, gen_stc,
                 B_s_gen, n_rewrites, kappa_par, keep_raw=True):
    """STC가 baseline과 같아지는 N_q. inf = 어떤 N_q로도 이길 수 없음."""
    base = wake_per_query(acc, len_c, len_q, 0.0, gen_base, 0.0, keep_raw=True)
    stc = wake_per_query(acc, len_c, len_q, len_chat, gen_stc, kappa_par, keep_raw)
    margin = base - stc
    if margin <= 0:
        return float("inf")
    return sleep_once(acc, len_c, B_s_gen, n_rewrites) / margin


# ---------------------------------------------------------------------------
# 논문에서 읽어온 값 / 논문에 없어 sweep하는 값
# ---------------------------------------------------------------------------

REGIMES = {
    "stateful_gsm_symbolic": {
        "gen_base": 600.0,
        "reduction": 5.0,
        "gen_source": "[Letta STC Fig.3 p.6] test-time 토큰 축 상단 근처. 수치표 없음 — 축에서 읽음",
        "reduction_source": "[Letta STC Abstract; §5.1 p.6] 'roughly 5x'",
        "context_character": "GSM8K에 1~2절 추가한 문장 몇 개. 짧다",
        "len_c_plausible": 400.0,
    },
    "stateful_aime": {
        "gen_base": 22500.0,
        "reduction": 5.0,
        "gen_source": "[Letta STC Fig.4 p.7] 축 상단 (모델별 1000~22500)",
        "reduction_source": "[Letta STC Abstract; §5.1 p.6]",
        "context_character": "AIME 문제의 진술문들. 짧다. 일부 인스턴스는 문맥이 비어 있음 [App. J]",
        "len_c_plausible": 300.0,
    },
    "swe_features": {
        "gen_base": 10000.0,
        "reduction": 1.5,
        "gen_source": "[Letta STC Fig.11 p.12] 3000~10000",
        "reduction_source": "[Letta STC §6 p.12] 'roughly 1.5x'",
        "context_character": "형제 PR 묶음 = 실제 코드베이스. 논문은 크기를 밝히지 않는다",
        # 2026-08-11 정정. 초판은 8000 토큰을 가정했다. 그 가정이 방법에게 유리한
        # 쪽으로 틀려 있었다 — 프로덕션 실측이 한 자릿수 이상 크다. 아래 값은
        # 세 읽기 중 STC에 가장 유리한 것(호출당 프롬프트의 기존분)을 앵커로 쓴다.
        "len_c_plausible": 49000.0,
        "len_c_status": "MEASURED-PROXY",
        "len_c_source": (
            "[2608.00101 Fig.10 p.8; Fig.11 p.8; Table 4 p.4] 공유 저장소 위 코딩 "
            "에이전트의 프로덕션 추적 13.5M 세션·95.1M 턴. 호출당 프롬프트 중앙값 "
            "68K에서 턴 안에서 새로 쌓이는 function-call 28%를 빼면 기존분 72% ≈ 49K"),
        "len_c_measured_variants": {
            "per_call_preexisting_72pct": {
                "tokens": 49000.0,
                "source": "[2608.00101 Fig.10 p.8 × Fig.11 p.8] 0.72 × 68K. 본서 산술",
                "reading": "질의 도착 전에 이미 존재하던 프롬프트만 센다 — 이 회계의 |c|에 가장 가깝다",
                "direction": "셋 중 STC에 가장 유리",
            },
            "per_call_median_68k": {
                "tokens": 68000.0,
                "source": "[2608.00101 Fig.10, p.8] LLM 호출당 프롬프트 토큰 중앙값",
                "reading": "한 호출이 실제로 읽는 전량",
                "direction": "중간",
            },
            "per_turn_median_227_6k": {
                "tokens": 227600.0,
                "source": "[2608.00101 Table 4, p.4] 사용자 턴당 프롬프트 토큰 중앙값. P90 1.50M",
                "reading": "한 사용자 턴이 소비하는 전량(턴당 중앙 4.5 호출)",
                "direction": "셋 중 STC에 가장 불리",
            },
        },
        "len_c_superseded": {
            "value": 8000.0,
            "why": ("초판 가정. 실측의 6분의 1에서 28분의 1이었고 방향이 방법에게 "
                    "유리한 쪽이었다. X3 Q5가 이미 '하한으로만 지지된다'고 판정했고, "
                    "2608.00101이 그 하한이 얼마나 낮았는지를 확정했다"),
            "printed_ratio_that_this_supersedes": "L*/|c| = 4.17 (FLOPs 회계, kappa_par=1)",
        },
    },
}

FROM_PAPER = {
    "kappa_cost": {"value": 10.0,
                   "source": "[Letta STC §5.3 p.8 + fn.4] Databricks provisioned-throughput 문서 링크",
                   "note": "민감도 분석 없음. 측정 아님"},
    "n_rewrites_max": {"value": 10.0, "source": "[Letta STC App. K p.27] J <= 10 rethink_memory 호출"},
    "kappa_par_grid": {"value": [1, 2, 5, 10],
                       "source": "[Letta STC §5.2 p.8; Fig.7 legend p.10]",
                       "note": "논문은 5벌이 10벌보다 대체로 낫다고 보고하며 기제를 제시하지 않음"},
}

ABSENT_FROM_PAPER = {
    "len_c": ("원 문맥 길이 — 토큰/문자 수 미보고. GSM-Symbolic·AIME의 len_c_plausible은 "
              "여전히 이 실험의 가정이다. SWE-Features만 2026-08-11에 프로덕션 실측 앵커로 "
              "교체했다(len_c_status=MEASURED-PROXY). 교체해도 논문이 자기 |c|를 인쇄한 "
              "것은 아니므로 이 칸은 absent로 남는다"),
    "len_chat": "learned context 길이 — 미보고. 상태 용량 C가 absent인 이유",
    "B_s_gen": "sleep 생성 토큰 수 — 미보고. 상각 논증의 분자가 숫자로 없음",
    "N_q_distribution": "실서빙에서 한 문맥을 공유하는 질의 수의 분포 — 미보고. {1,2,5,10}으로 sweep만",
}

ASSUMPTIONS = {
    "A1": "sleep 생성 토큰에는 kappa_cost 배수를 적용하지 않는다(throughput-optimized). STC에 유리",
    "A2": "sleep의 문맥 읽기는 n_rewrites회로만 계산한다. 실제 rethink 루프는 누적 프롬프트를 다시 읽으므로 과소계상. STC에 유리",
    "A3": "baseline은 c-hat 없이 원 문맥만 읽는다. 공정",
    "A4": "정확도는 동일하다고 놓는다(논문의 '동일 정확도 도달' 프레이밍). 정확도 차이는 이 회계에 없다",
    "A5": "원 문맥을 유지한다(keep_raw=True). 논문이 원시 문맥과 learned context를 함께 쓴다고 명시",
}


def main() -> None:
    accs = [
        Accounting(w_in=0.00, kappa_cost=10.0, label="paper_implicit(w_in=0)"),
        Accounting(w_in=0.25, kappa_cost=10.0, label="pricing(w_in=0.25)"),
        Accounting(w_in=1.00, kappa_cost=10.0, label="flops(w_in=1.0)"),
    ]

    results = {
        "meta": {
            "id": "X1-unified-cost-accounting",
            "title": "Sleep-time compute 상각 논증의 유효 범위",
            "grade": "EXPLORATION-GRADE / ACCOUNTING (측정 아님)",
            "deterministic": True,
            "randomness": "none",
            "unit": "cost normalised so that one sleep-time generated token = 1",
            "regimes": REGIMES,
            "from_paper": FROM_PAPER,
            "absent_from_paper": ABSENT_FROM_PAPER,
            "assumptions_favouring_stc": ASSUMPTIONS,
            "caveat_tags": {
                "ACCOUNTING-NOT-MEASURED": "닫힌 형태 회계다. wall-clock도 청구액도 아니다",
                "PAPER-ABSENT-SWEPT": "len_c, len_chat, B_s_gen은 원 논문에 없어 가정/sweep한다",
                "AXIS-READ": "gen_base는 수치표가 없어 그림 축 범위에서 읽은 값",
                "MEASURED-PROXY-LEN-C": (
                    "SWE-Features의 len_c는 Letta의 벤치마크에서 잰 값이 아니라 같은 종류의 "
                    "워크로드(공유 저장소 위 코딩 에이전트)의 프로덕션 실측을 대입한 것이다. "
                    "자릿수 비교로만 읽는다"),
                "SELF-BUILT-BENCH": "5x·1.5x 자체가 저자 제작 벤치의 값",
                "PRO-STC-ASSUMPTIONS": "가정 A1·A2는 STC에 유리한 쪽으로 잡혀 있다",
            },
        }
    }

    # -- Q1/Q2: 레짐별 임계 길이 L* ------------------------------------------------
    q12 = {}
    for rname, R in REGIMES.items():
        gen_base = R["gen_base"]
        gen_stc = gen_base / R["reduction"]
        len_c = R["len_c_plausible"]
        rows = {}
        for acc in accs:
            for kp in FROM_PAPER["kappa_par_grid"]["value"]:
                L = critical_len_chat(acc, len_c, 30.0, gen_base, gen_stc, float(kp))
                rows[f"{acc.label},kappa_par={kp}"] = {
                    "L_star_tokens": None if L == float("inf") else round(L, 1),
                    "L_star_over_len_c": None if L == float("inf") else round(L / len_c, 2),
                }
        q12[rname] = {
            "gen_base": gen_base, "reduction": R["reduction"], "len_c_assumed": len_c,
            "len_c_status": R.get("len_c_status", "ASSUMED"),
            "context_character": R["context_character"],
            "critical_len_chat": rows,
        }
        # L*는 |c|에 의존하지 않는다. 비 L*/|c| 만 의존한다.
        # 실측이 여러 읽기를 주면 그 전부에 대해 비를 인쇄한다 — 하나만 고르면
        # 고른 사람이 판정하는 것이 된다.
        variants = R.get("len_c_measured_variants")
        if variants:
            vrows = {}
            for vname, V in variants.items():
                lc = V["tokens"]
                per_v = {}
                for acc in accs:
                    for kp in FROM_PAPER["kappa_par_grid"]["value"]:
                        L = critical_len_chat(acc, lc, 30.0, gen_base, gen_stc, float(kp))
                        per_v[f"{acc.label},kappa_par={kp}"] = (
                            None if L == float("inf") else round(L / lc, 3))
                vrows[vname] = {
                    "len_c_tokens": lc, "source": V["source"], "reading": V["reading"],
                    "L_star_over_len_c": per_v,
                }
            q12[rname]["L_star_over_len_c_by_measured_reading"] = vrows
            q12[rname]["superseded"] = R.get("len_c_superseded")
    results["Q1_Q2_critical_length_by_regime"] = {
        "description": ("c-hat이 L*를 넘으면 N_q를 아무리 키워도 baseline을 이길 수 없다. "
                        "L*/|c| 가 1보다 훨씬 크면 여유가 크고, 1 근처면 실제로 물린다"),
        "values": q12,
    }

    # -- Q3: 임계 안쪽에서의 손익분기 N_q ------------------------------------------
    q3 = {}
    for rname, R in REGIMES.items():
        gen_base = R["gen_base"]
        gen_stc = gen_base / R["reduction"]
        len_c = R["len_c_plausible"]
        B_s = 2.5 * gen_base          # sleep이 baseline 생성의 2.5배를 쓴다고 가정(sweep 대상)
        rows = {}
        for acc in accs:
            for ratio in (0.25, 1.0, 4.0):
                nq = breakeven_Nq(acc, len_c, 30.0, len_c * ratio, gen_base, gen_stc,
                                  B_s, 3.0, 1.0)
                rows[f"{acc.label},|c_hat|/|c|={ratio}"] = None if nq == float("inf") else round(nq, 2)
        q3[rname] = {"B_s_gen_assumed": B_s, "n_rewrites_assumed": 3.0, "kappa_par": 1, "values": rows}
    results["Q3_breakeven_Nq"] = {
        "description": "STC가 baseline과 같아지는 질의 수. null = 상각 불가",
        "values": q3,
    }

    # -- Q4: kappa_cost 민감도 (SWE 레짐, FLOPs 회계) --------------------------------
    R = REGIMES["swe_features"]
    gen_base, gen_stc, len_c = R["gen_base"], R["gen_base"] / R["reduction"], R["len_c_plausible"]
    q4 = {}
    for kc in (1.0, 2.0, 5.0, 10.0, 20.0):
        acc = Accounting(w_in=1.0, kappa_cost=kc, label=f"kc={kc}")
        for kp in (1, 10):
            L = critical_len_chat(acc, len_c, 30.0, gen_base, gen_stc, float(kp))
            q4[f"kappa_cost={kc},kappa_par={kp}"] = {
                "L_star_tokens": round(L, 1),
                "L_star_over_len_c": round(L / len_c, 2),
            }
    results["Q4_kappa_cost_sensitivity"] = {
        "description": ("SWE-Features 레짐, FLOPs 회계에서 kappa_cost가 임계에 미치는 영향. "
                        "kappa_cost는 논문이 측정하지 않고 벤더 문서에서 인용한 상수다"),
        "values": q4,
    }

    # -- 판정 -----------------------------------------------------------------------
    gsm = q12["stateful_gsm_symbolic"]["critical_len_chat"]["flops(w_in=1.0),kappa_par=1"]
    gsm10 = q12["stateful_gsm_symbolic"]["critical_len_chat"]["flops(w_in=1.0),kappa_par=10"]
    swe = q12["swe_features"]["critical_len_chat"]["flops(w_in=1.0),kappa_par=1"]
    swe10 = q12["swe_features"]["critical_len_chat"]["flops(w_in=1.0),kappa_par=10"]

    results["findings"] = {
        "F1_the_missing_term_is_structural": (
            "논문의 식은 한 번 내는 비용(B_s)만 N_q로 나눈다. learned context의 prefill은 "
            "매 질의마다 내는 비용이므로 상각되지 않는다. 이 항이 0이 아니면 'N_q를 키우면 "
            "언제나 이긴다'가 성립하지 않는 영역이 원리적으로 존재한다. 논문의 회계(w_in=0)는 "
            "그 영역의 경계를 무한대로 보내 표현 자체를 못 한다."),
        "F2_gsm_regime_is_safe": (
            f"Stateful GSM-Symbolic 레짐에서는 여유가 크다. FLOPs 회계·kappa_par=1에서 "
            f"L* = {gsm['L_star_tokens']} 토큰 (문맥의 {gsm['L_star_over_len_c']}배), "
            f"kappa_par=10에서도 {gsm10['L_star_tokens']} 토큰 ({gsm10['L_star_over_len_c']}배). "
            "즉 짧은 문맥·큰 절감률 영역에서 논문의 5x는 prefill 항을 넣어도 뒤집히지 않는다. "
            "이 실험은 그것을 반박하지 않는다."),
        "F3_swe_regime_is_where_it_bites": (
            f"SWE-Features 레짐에서는 여유가 사라진다. 절감률이 1.5x로 작고 문맥이 프로덕션 "
            f"코드베이스라 L* = {swe['L_star_tokens']} 토큰이고 이것은 문맥의 "
            f"{swe['L_star_over_len_c']}배 — 즉 **원 문맥보다 짧다.** kappa_par=10이면 "
            f"{swe10['L_star_tokens']} 토큰 ({swe10['L_star_over_len_c']}배)까지 좁아진다. "
            "learned context가 원 문맥의 일부여야만 매 질의에서 값을 치를 수 있다는 뜻인데, "
            "Letta의 c-hat은 c보다 길고 원 c도 유지된다(가정 A5). 논문이 SWE-Features에서 "
            "'이득이 작다'고만 보고한 자리에, 회계는 '이득이 사라지는 경계를 이미 넘었다'는 "
            "더 강한 진술을 허용한다."),
        "F6_the_len_c_assumption_was_wrong_in_the_methods_favour": (
            "2026-08-11 정정. 초판은 SWE-Features의 |c|를 8,000 토큰으로 가정하고 "
            "L*/|c| = 4.17배를 인쇄했다. 같은 종류의 워크로드에 대한 프로덕션 실측은 "
            "호출당 프롬프트 중앙값 68K, 턴당 중앙값 227.6K, 턴당 P90 1.50M이다 "
            "[2608.00101 Fig.10 p.8; Table 4 p.4]. 세 읽기를 대입하면 FLOPs 회계·kappa_par=1에서 "
            "L*/|c| 가 각각 0.68 / 0.49 / 0.15로, **셋 다 1보다 작다.** 인쇄된 4.17배와 부등호의 "
            "방향이 반대다. 초판의 가정은 실측의 6분의 1에서 28분의 1이었고, 그 방향이 방법에게 "
            "유리한 쪽이었다. 정정하면 레짐 순서(SWE-Features 최하위)는 그대로이고 간격은 벌어진다. "
            "정직성 한계 둘: (i) L* 안의 B_t는 여전히 그림 축에서 읽은 값(AXIS-READ)이므로 이번에 "
            "측정된 것은 계산의 |c| 절반뿐이다. (ii) 실측은 Letta의 벤치마크가 아니라 다른 "
            "프로덕션 배포에서 나왔다(MEASURED-PROXY-LEN-C) — 자릿수 비교로만 읽는다."),
        "F4_ordering_is_the_load_bearing_result": (
            "레짐 간 여유의 순서 — GSM-Symbolic >> AIME > SWE-Features — 는 세 파라미터"
            "(절감률, 문맥 길이, 병렬 벌수)의 방향만으로 결정되며, 미보고 값의 선택에 둔감하다. "
            "본문 단정문으로 쓸 수 있는 것은 이 순서다. L*의 절대치는 아니다."),
        "F5_kappa_cost_is_load_bearing_and_unmeasured": (
            "kappa_cost는 임계에 선형으로 들어가는데, 논문은 이 값을 측정하지 않고 벤더 문서 "
            "한 줄에서 인용했으며 민감도 분석을 하지 않았다. kappa_cost가 1에 가까우면(배치 "
            "처리로 wake도 throughput-optimized로 돌리는 경우) 임계는 10분의 1로 줄어든다."),
        "honest_limit": (
            "이 실험은 Letta의 5x를 반박하지 않는다. 5x는 생성 토큰 축에서 관측된 사실이다. "
            "이 실험이 반박하는 것은 '생성 토큰 축이 비용 축과 같다'는 암묵 등식이며, "
            "그 등식이 깨지는 지점이 논문이 이미 약하다고 보고한 레짐과 일치한다는 것이다."),
    }

    out = HERE / "result.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    for k, v in results["findings"].items():
        print(f"\n[{k}]\n{v}")


if __name__ == "__main__":
    main()
