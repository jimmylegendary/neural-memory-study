#!/usr/bin/env python3
"""X2 — 세 경로의 상태 회계: 무엇이 어디에 몇 바이트로 쌓이고, 그 바이트에 정보가 얼마나 담기는가.

왜 필요한가
----------
"Memory device 회사에 어떤 기회가 있는가"는 세 경로가 만드는 상태의 **양·위치·수명·
트래픽**이 다르다는 사실 위에서만 답할 수 있다. 그런데 corpus의 어느 논문도 상태 용량을
바이트로 보고하지 않는다(cost_table의 C_cap이 거의 전부 absent). 이 실험은 논문들이
보고한 **구성 파라미터**로부터 바이트를 복원하고, 별도 논문이 측정한 **기억 용량 하한**과
결합해 세 경로를 같은 자로 잰다.

세 경로의 상태
-------------
E : learned context / 그래프 / 벡터.  세션·사용자 단위. 모델 밖. 콜드~웜.
W : fast weights.                    활성 세션 단위. HBM 상주. 토큰마다 read-modify-write.
Θ : 가중치 delta.                     사용자 단위. 영속. 요청 시 로드 필요.

핵심 질문
--------
Q1  같은 앵커에서 세 경로가 만드는 바이트는 각각 얼마인가.
Q2  저장 dtype이 바이트당 정보 밀도의 하한을 어떻게 정하는가.
Q3  Θ-경로가 주류가 되면 사용자 수에 따라 웜 스토리지가 어떻게 늘어나는가.
Q4  직접 편집(MEMIT류)과 delta(LoRA류)는 서빙 구조상 무엇이 다른가.

등급
----
EXPLORATION-GRADE / ACCOUNTING. 측정이 아니라 회계다.
- 신뢰할 것: 경로 간 **자릿수 차이**, dtype 축의 **기울기**, 사용자 수 스케일링의 **기울기**.
- 신뢰하지 말 것: 특정 배포의 절대 바이트. 경로 간 정보 밀도 비교는 **철회했다**(findings F3_withdrawn).
- 3.64 bits/parameter는 원 논문이 명시한 **하한**이다([lm-memorization-capacity §3.2, txt L664]).
  상한이 아니다. 무작위 문자열로 from-scratch 학습한 소형 GPT에서 측정된 값이므로
  LoRA delta 적용은 별도로 **이 실험의 외삽**이다.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

BYTES_BF16 = 2

# ---------------------------------------------------------------------------
# 출처가 있는 상수
# ---------------------------------------------------------------------------

SOURCED = {
    "bits_per_parameter": {
        "value": 3.64,
        "range": [2.86, 4.23],
        "source": "[lm-memorization-capacity Fig.6 caption; Table 1] GPT half precision 3.64 bpp; "
                  "Table 1 평균 bf16 3.51 / fp32 3.83, 설정별 범위 2.86–4.23",
        "bound_direction": "LOWER BOUND",
        "bound_source": "[lm-memorization-capacity §3.2, txt L664] "
                        "'we are only ever measuring a lower bound on model capacity'",
        "caveat": "무작위 uniform 문자열을 from-scratch 학습한 소형 GPT에서 측정한 **하한**이다. "
                  "따라서 파라미터 P개의 delta가 담는 정보량은 3.64·P bits **이상**이며, "
                  "이 값을 상한으로 쓰면 Θ층의 용량을 과소평가한다. "
                  "LoRA delta 적용은 이 실험의 외삽이라는 점은 별개로 유지된다.",
        "correction_note": "이 실험의 초판(2026-08-06)은 이 값을 상한으로 놓고 'Θ가 정보 희소하다'는 "
                           "비교를 했다. 방향이 틀렸으므로 철회했다. 자세한 것은 findings의 "
                           "F3_withdrawn 항목.",
    },
    "W_state_MB_per_layer_1p3B": {
        "value": 134.2,
        "source": "[NM experiments/results.json claim1] neural-mem-1.3B (d=2048, m=16, L=24, bf16)",
    },
    "W_rmw_GB_per_token_1p3B": {
        "value": 6.44,
        "source": "[NM experiments/results.json claim1] 전 층 상태의 토큰당 read-modify-write",
    },
    "memit_edits": {
        "value": 10000, "hours": 7.44, "model": "GPT-J 6B",
        "source": "[memit §5.2.2; App. B.4] 10,000 편집 총 7.44 hr, A6000",
        "derived": "편집당 2.6773 sec (note-writer 산술, 논문에 없음)",
    },
    "mem0_store_tokens": {
        "value_range": [7000, 14000],
        "source": "[mem0 §4] LOCOMO 대화당 저장 토큰 (바이트 미보고)",
    },
    # --- 2026-08 추가: E-경로 rate 앵커 (R-1 발화의 근거) --------------------
    # 이 실험의 초판은 E-경로를 14,000 토큰 대화 하나에 앵커한 **고정 바이트**로 적었다.
    # 제3자 계측이 사용자당 발자국을 이력 길이의 함수로 인쇄하면서 그 형태가 깨졌다.
    # 회계(= bytes/token)는 맞았고 앵커(= 곱하는 이력 길이)가 약 70배 작았다.
    "E_path_measured_bytes_per_token": {
        "values": {"embedRAG": 7.0, "Mem0": 12.0, "HippoRAG_v2": 62.0},
        "spread": "약 9배 (62/7 = 8.9)",
        "measured_at": "단일 사용자 이력 약 64K → 약 1M 토큰 스윕, 1M 토큰 지점",
        "source": "[agent-memory-syscharacterization 2606.06448 §4.7, p.10; Fig. 9c] "
                  "'On-disk footprint grows roughly proportionally to content volume for most "
                  "systems, with a ~9x spread at 1M tokens. Multi-view designs inflate the "
                  "proportionality constant (HippoRAG v2 reaches ~62 MB at 1M tokens) ... "
                  "(Mem0 at ~12 MB)'",
        "fleet_projection_printed_by_the_paper": "1M 토큰 발자국을 100K 사용자로 투영하면 "
                                                 "약 0.7 TB(embedRAG) ~ 약 6.2 TB(HippoRAG v2)",
        "caveat": "이질적 store(역색인·밀집 벡터·그래프 엣지·다중 뷰·메타데이터)의 on-disk "
                  "발자국이다. dtype 표기 없음, 내용/인덱스 미분리. 따라서 dtype이 정확히 정하는 "
                  "Θ-delta 바이트와 **같은 열에 놓지 않는다**. 쓸 수 있는 것은 자릿수와 성장 모양뿐.",
    },
    "E_path_bounded_store_saturated_chars": {
        "value": 48506,
        "detail": "ALFWorld 통합 단계 168에서 '50 items totalling 48,506 characters "
                  "(i.e., the cap is saturated, average ~970 characters per item)'",
        "source": "[faulty-memories-update 2605.12978 App. G.2, p.53]",
        "reading": "상한을 건 store는 포화하면 이력 길이와 무관한 상수가 된다. "
                   "상한 있음/없음이 E-경로 성장 모양을 가르는 1차 변수다.",
    },
    "E_path_no_default_forgetting": {
        "source": "[2606.06448 §4.7, p.10] 'None of the evaluated systems prune or forget by "
                  "default, so footprint grows monotonically under default behavior; bounding "
                  "fleet storage requires an independent forgetting policy'",
        "reading": "상한 없는 store의 단조 성장은 설계 실패가 아니라 정책 부재다.",
    },
    "E_path_retrieval_latency_flat": {
        "source": "[2606.06448 Fig. 9d, p.10] store가 커져도 검색 지연이 거의 평평하다",
        "reading": "E의 콜드 계층 배정은 크기가 아니라 접근 패턴이 정한다. "
                   "크기 진술이 무너져도 배정은 선다.",
    },
    "memory_layers_flops_invariance": {
        "source": "[memory-layers §5.1] 'Models with the same base model configuration have "
                  "negligible differences in FLOPs' — memory 파라미터를 10^2배 늘려도 FLOPs 사실상 불변",
    },
}

ABSENT = {
    "C_cap_in_bytes": "corpus 전체에서 상태 용량을 바이트로 보고한 논문 0편. "
                      "letta-stc·zep·memory-layers·mem0 모두 absent. 이 실험은 구성 파라미터에서 복원한다",
}


# ---------------------------------------------------------------------------
# 앵커 모델
# ---------------------------------------------------------------------------

ANCHORS = {
    "neural-mem-1.3B": {"d": 2048, "L": 24, "note": "NM 앵커. W-경로 연속성", "params_B": 1.3,
                        "has_fast_weights": True},
    "dense-8B (d=4096, L=32)": {
        "d": 4096, "L": 32, "params_B": 8.0,
        "note": "SEAL / LM-Need-Sleep / MEMIT가 다루는 스케일. Llama-3-8B급 dense transformer",
        "has_fast_weights": False,
    },
}

# Θ delta의 저장 dtype. 정보 밀도는 dtype에 직접 나뉘므로 이 축이 결과를 지배한다.
DTYPES = {"bf16": 2.0, "int8": 1.0, "int4": 0.5}

# E-경로는 **고정 바이트가 아니라 rate**다. 사용자당 바이트 = rate × 이력 길이.
# 초판은 이 곱의 오른쪽 인자를 14,000 토큰(대화 하나)으로 고정했고, 그것이 이 실험의
# 유일한 앵커였다. 실측 레짐은 사용자당 수십만~수백만 토큰이므로 약 70배 차이가 난다.
E_RATES_BYTES_PER_TOKEN = {
    "text_assumed": 4.0,            # 이 실험의 가정: 토큰당 4바이트 UTF-8
    "embed_fp32_assumed": 12.0,     # 이 실험의 가정: 512-토큰 청크당 1536-d fp32 = 12 B/tok
    "measured_embedRAG": 7.0,       # [2606.06448 §4.7, p.10]
    "measured_Mem0": 12.0,          # [2606.06448 §4.7, p.10] — 가정 embed 행과 자릿수까지 일치
    "measured_HippoRAG_v2": 62.0,   # [2606.06448 §4.7, p.10] — 다중 뷰 + 뷰별 임베딩 + 다중 인덱스
}

# 두 이력 앵커를 함께 인쇄한다. 하나만 인쇄하면 R-1이 잡아낸 오류가 되살아난다.
HISTORY_ANCHORS = {
    "session_14k_tokens": 14_000,       # 대화 하나 (초판의 유일한 앵커)
    "long_horizon_1M_tokens": 1_000_000,  # 장기 에이전트 사용자 하나 (실측 스윕의 끝점)
}


def lora_delta_params(d: int, L: int, r: int, matrices_per_layer: int = 2) -> int:
    """LoRA delta 파라미터 수.

    한 행렬(d x d)에 대한 LoRA는 A(r x d) + B(d x r) = 2rd 파라미터.
    matrices_per_layer=2 는 {W_q, W_v} — LoRA 논문이 GPT-3에서 기본으로 쓴 조합
    [lora §7.1]. 4로 두면 {W_q,W_k,W_v,W_o}.
    """
    return matrices_per_layer * 2 * r * d * L


def main() -> None:
    R: dict = {
        "meta": {
            "id": "X2-state-capacity-accounting",
            "title": "세 경로의 상태 용량·정보 밀도·스케일링 회계",
            "grade": "EXPLORATION-GRADE / ACCOUNTING (측정 아님)",
            "deterministic": True,
            "randomness": "none",
            "sourced_constants": SOURCED,
            "absent_from_corpus": ABSENT,
            "anchors": ANCHORS,
            "caveat_tags": {
                "ACCOUNTING-NOT-MEASURED": "구성 파라미터에서 복원한 바이트다. 실측 아님",
                "BPP-IS-A-LOWER-BOUND": "3.64 bits/param은 원 논문이 하한이라고 밝힌 값이다. 상한으로 쓰면 용량을 과소평가한다. LoRA delta 적용은 별개로 외삽",
                "CORPUS-SILENT-ON-BYTES": "상태 용량을 바이트로 보고한 논문이 corpus에 없음",
                "E-PATH-TOKENS-ONLY": "E-경로 용량은 토큰 수만 보고됨. 바이트 환산은 이 실험의 가정",
                "E-PATH-IS-A-RATE-NOT-A-CONSTANT": (
                    "E-경로 사용자당 바이트는 상수가 아니라 rate × 이력 길이다. 고정 바이트로 "
                    "인용하면 이력 앵커가 숨는다. 이 실험의 초판이 그 오류를 냈고 R-1이 발화했다"),
                "E-BYTES-NOT-DTYPE-ANNOTATED": (
                    "실측된 E 바이트는 이질적 store의 on-disk 발자국이며 dtype 표기도 "
                    "내용/인덱스 분리도 없다. Θ의 dtype 회계와 같은 열에 놓지 않는다"),
            },
        }
    }

    # -- Q1: 경로별 상태 바이트 --------------------------------------------------
    q1 = {}
    for name, A in ANCHORS.items():
        d, L = A["d"], A["L"]
        row = {"anchor": A}

        # W-경로: fast weights를 가진 아키텍처에만 존재한다.
        # dense transformer에는 W층이 없으므로, 8B 항은 "같은 폭의 TTT류 모델이라면"이라는
        # 반사실 가정 위에서만 읽어야 한다. Llama-3-8B 자체의 값이 아니다.
        if A["has_fast_weights"]:
            w_bytes = SOURCED["W_state_MB_per_layer_1p3B"]["value"] * 1e6 * L
            w_src = "measured-in-NM (neural-mem-1.3B)"
            w_caveat = None
        else:
            base = SOURCED["W_state_MB_per_layer_1p3B"]["value"] * 1e6
            w_bytes = base * (d / 2048) ** 2 * L
            w_src = "counterfactual: d^2 * L extrapolation from the NM anchor"
            w_caveat = ("이 앵커는 dense transformer이므로 W층이 존재하지 않는다. "
                        "이 값은 '같은 폭의 TTT류 모델이라면'이라는 반사실 수치이며, "
                        "Llama-3-8B의 상태량이 아니다. 경로 간 자릿수 비교 용도로만 쓴다.")
        row["W_path"] = {
            "bytes_per_active_session": round(w_bytes),
            "MB": round(w_bytes / 1e6, 1),
            "residence": "HBM 상주 (토큰마다 RMW)",
            "lifetime": "세션 (transient)",
            "traffic_GB_per_token": (SOURCED["W_rmw_GB_per_token_1p3B"]["value"]
                                     if A["has_fast_weights"] else None),
            "provenance": w_src,
            "counterfactual_caveat": w_caveat,
        }

        # Θ-경로: LoRA delta
        theta = {}
        for r in (4, 16, 64):
            for mpl, mname in ((2, "q,v"), (4, "q,k,v,o")):
                p = lora_delta_params(d, L, r, mpl)
                theta[f"r={r},{mname}"] = {
                    "params": p,
                    "bytes_bf16": p * BYTES_BF16,
                    "MB": round(p * BYTES_BF16 / 1e6, 2),
                    "info_capacity_bits_at_least": round(p * SOURCED["bits_per_parameter"]["value"]),
                    "info_capacity_MB_at_least": round(p * SOURCED["bits_per_parameter"]["value"] / 8 / 1e6, 2),
                }
        row["Theta_path_lora"] = {
            "variants": theta,
            "residence": "웜 스토리지 → 요청 시 로드",
            "lifetime": "사용자 (영속)",
        }

        # Θ-경로: 직접 편집(MEMIT류)은 delta를 만들지 않는다.
        full_bytes = A["params_B"] * 1e9 * BYTES_BF16
        row["Theta_path_direct_edit"] = {
            "delta_bytes": 0,
            "per_user_bytes_if_personalised": round(full_bytes),
            "per_user_GB_if_personalised": round(full_bytes / 1e9, 2),
            "why": ("MEMIT/ROME류는 공유 가중치를 제자리에서 고친다. delta 아티팩트가 없으므로 "
                    "사용자별 개인화를 하려면 모델 전체 사본이 필요하다"),
            "edit_cost": SOURCED["memit_edits"],
        }

        # E-경로
        e = {}
        for tok in SOURCED["mem0_store_tokens"]["value_range"]:
            e[f"{tok}_tokens"] = {
                "as_utf8_text_bytes_assumed_4B_per_token": tok * 4,
                "as_utf8_text_KB": round(tok * 4 / 1e3, 1),
                "info_bits_face_value": tok * 4 * 8,
                "as_embeddings_d1536_fp32_bytes_512tok_chunks":
                    round(max(1, tok / 512) * 1536 * 4),
            }
        # E-경로를 rate × 이력 길이로 다시 적는다. 표에 인쇄되는 것은 곱이 아니라
        # **두 인자**여야 한다 — 그래야 독자가 자기 레짐의 이력 길이로 재계산할 수 있다.
        by_hist = {}
        for hname, htok in HISTORY_ANCHORS.items():
            by_hist[hname] = {
                "history_tokens": htok,
                "bytes_by_rate": {rk: round(rv * htok) for rk, rv in E_RATES_BYTES_PER_TOKEN.items()},
                "MB_by_rate": {rk: round(rv * htok / 1e6, 3) for rk, rv in E_RATES_BYTES_PER_TOKEN.items()},
            }
        theta_ref = lora_delta_params(4096, 32, 16, 2) * BYTES_BF16  # dense-8B, r=16, {q,v}, bf16
        row["E_path"] = {
            "variants": e,
            "rates_bytes_per_token": E_RATES_BYTES_PER_TOKEN,
            "by_history_anchor": by_hist,
            "bounded_store_saturated_bytes": SOURCED["E_path_bounded_store_saturated_chars"]["value"],
            "residence": "콜드/벡터 스토어 → 질의마다 ret() + prefill",
            "lifetime": "세션~사용자",
            "note": ("E는 고정 바이트가 아니라 rate다. 사용자당 바이트 = rate × 이력 길이. "
                     "토큰→바이트 환산율은 이 실험의 가정이었고 실측과 자릿수까지 맞았다"
                     "(가정 embed 12 B/tok = 실측 Mem0 12 B/tok). 틀린 것은 이력 길이 앵커다."),
            "crossover_with_theta_delta": {
                "theta_delta_bytes_dense8B_r16_qv_bf16": round(theta_ref),
                "E_exceeds_theta_at_1M_tokens": {
                    rk: bool(rv * HISTORY_ANCHORS["long_horizon_1M_tokens"] > theta_ref)
                    for rk, rv in E_RATES_BYTES_PER_TOKEN.items()
                },
                "reading": ("이력을 1M 토큰으로 놓으면 상한 없는 다중 뷰 store(62 B/tok)의 "
                            "사용자당 바이트가 dense-8B Θ delta를 넘는다. 두 양이 교차하므로 "
                            "경로 간 크기 순위는 존재하지 않는다 — 존재하는 것은 교차점이다."),
            },
        }
        q1[name] = row
    R["Q1_state_bytes_by_path"] = q1

    # -- Q2: 바이트당 정보 밀도 --------------------------------------------------
    # 정보 밀도 = (파라미터당 담기는 bits) / (파라미터당 저장 bytes).
    # 분자는 논문이 측정한 3.64 bits/param 상한, 분모는 순전히 저장 dtype이 정한다.
    # 따라서 이 비교는 dtype 축을 빼놓고는 성립하지 않는다.
    A8 = ANCHORS["dense-8B (d=4096, L=32)"]
    p16 = lora_delta_params(A8["d"], A8["L"], 16, 2)
    bpp = SOURCED["bits_per_parameter"]["value"]
    theta_info_bits = p16 * bpp

    theta_by_dtype = {}
    for dt, bytes_per_param in DTYPES.items():
        store = p16 * bytes_per_param
        theta_by_dtype[dt] = {
            "store_bytes": round(store),
            "store_MB": round(store / 1e6, 2),
            "bits_per_stored_byte": round(theta_info_bits / store, 3),
        }

    e_tokens = 14000
    e_store_bytes = e_tokens * 4
    e_density_face = 8.0
    e_density_compressed = 8.0 / 3.0  # 자연어 텍스트의 보수적 압축비 3:1 가정

    R["Q2_information_density"] = {
        "description": ("저장 바이트 1개에 담기는 정보 bits의 **하한**. 분자는 "
                        "[lm-memorization-capacity]의 3.64 bits/param이며 그 논문이 "
                        "명시적으로 하한이라고 밝힌 값이다(§3.2, txt L664). 분모는 저장 dtype."),
        "Theta_lora_r16_qv_8B": {
            "params": p16,
            "info_bits_at_least": round(theta_info_bits),
            "info_MB_at_least": round(theta_info_bits / 8 / 1e6, 2),
            "by_dtype": theta_by_dtype,
            "reading": "각 dtype 행의 bits_per_stored_byte는 '적어도 이만큼'이다. 위가 아니라 아래가 막혀 있다.",
        },
        "E_text_14k_tokens": {
            "store_bytes": e_store_bytes,
            "store_KB": round(e_store_bytes / 1e3, 1),
            "bits_per_stored_byte_face": e_density_face,
            "bits_per_stored_byte_compressed_3to1": round(e_density_compressed, 3),
        },
        "cross_path_density_comparison": {
            "verdict": "WITHDRAWN — 이 회계로는 판정할 수 없다",
            "why": ("Θ 쪽 값은 하한이고 E 쪽 값은 액면가(또는 압축 가정)다. "
                    "하한과 액면가를 나눈 비는 어느 방향으로도 결론을 주지 않는다. "
                    "Θ의 실제 밀도가 하한보다 얼마나 높은지를 재려면 delta에 직접 정보를 "
                    "주입해 회수율을 재는 실험이 필요하고, 그것은 X2의 범위 밖이다."),
            "what_survives": ("dtype 축의 결론은 하한/상한과 무관하게 성립한다. 분자가 무엇이든 "
                              "분모(저장 bytes/param)는 dtype이 정하므로, bf16→int8→int4로 내리면 "
                              "저장 바이트당 정보 밀도가 그대로 2배씩 오른다."),
        },
        "note": ("bf16은 파라미터 하나에 최소 3.64 bits가 담기는데 16 bits를 쓴다. "
                 "즉 최소 4.4배의 저장 여유가 표현 자체에서 나온다. "
                 "이것은 알고리즘 한계가 아니라 dtype 선택이다."),
    }

    # -- Q3: 사용자 수 스케일링 --------------------------------------------------
    q3 = {}
    for users in (1e3, 1e5, 1e6, 1e8):
        row = {}
        for r in (4, 16, 64):
            p = lora_delta_params(A8["d"], A8["L"], r, 2)
            row[f"lora_r={r}_bf16"] = {"TB": round(p * BYTES_BF16 * users / 1e12, 3)}
            row[f"lora_r={r}_int4"] = {"TB": round(p * DTYPES["int4"] * users / 1e12, 3)}
        row["direct_edit_full_copy_bf16"] = {
            "TB": round(A8["params_B"] * 1e9 * BYTES_BF16 * users / 1e12, 1)}
        row["E_path_14k_tokens_text"] = {"TB": round(14000 * 4 * users / 1e12, 6)}
        # E-경로는 이력 앵커 없이 한 열로 인쇄하면 안 된다. 두 끝을 함께 적는다.
        row["E_path_1M_tokens_multiview_measured"] = {
            "TB": round(E_RATES_BYTES_PER_TOKEN["measured_HippoRAG_v2"]
                        * HISTORY_ANCHORS["long_horizon_1M_tokens"] * users / 1e12, 3),
            "source": "[2606.06448 §4.7, p.10] 62 B/tok",
        }
        row["E_path_1M_tokens_single_index_measured"] = {
            "TB": round(E_RATES_BYTES_PER_TOKEN["measured_embedRAG"]
                        * HISTORY_ANCHORS["long_horizon_1M_tokens"] * users / 1e12, 3),
            "source": "[2606.06448 §4.7, p.10] 7 B/tok",
        }
        q3[f"{users:.0e}_users"] = row
    R["Q3_scaling_with_users"] = {
        "description": "Θ-경로가 주류가 되면 웜 스토리지가 사용자 수에 선형으로 늘어난다",
        "anchor": "dense-8B (d=4096, L=32), LoRA on {q,v}",
        "values": q3,
    }

    # -- Q4: 서빙 구조 대비 --------------------------------------------------------
    R["Q4_serving_structure"] = {
        "shared_weight_batching": {
            "E_path": "유지된다. 가중치는 모두 공유. 상태는 프롬프트로 들어온다",
            "W_path": "부분적으로 깨진다. 세션마다 별도 fast-weight 상태가 HBM에 상주 "
                      "(NM pair thesis의 decode 절반)",
            "Theta_path_lora": "delta 단위로 깨진다. 요청마다 사용자 delta를 로드/머지해야 하며, "
                               "배치 안의 사용자가 다르면 가중치가 다르다",
            "Theta_path_direct_edit": "완전히 깨진다. 개인화하려면 모델 전체 사본",
        },
        "write_amplification": {
            "Theta_edit_cost": SOURCED["memit_edits"],
            "note": "편집당 2.68 sec (note-writer 산술)는 쓰기 비용이 읽기 비용과 자릿수가 다름을 뜻한다. "
                    "E-경로의 쓰기는 LLM 호출 몇 번, W-경로의 쓰기는 토큰당 RMW",
        },
        "traffic_character": {
            "E": "질의당 1회 검색 + prefill 증가. 읽기 편향",
            "W": f"토큰당 {SOURCED['W_rmw_GB_per_token_1p3B']['value']} GB RMW (1.3B 앵커). 쓰기 편향, 공유 불가",
            "Theta": "요청당 delta 로드(수 MB) + 갱신은 sleep 라운드에만. 읽기 편향, 사용자별",
        },
    }

    # -- 판정 ----------------------------------------------------------------------
    R["findings"] = {
        "F1_corpus_reports_no_bytes": (
            "corpus 전체에서 상태 용량을 바이트로 보고한 논문이 0편이다. 세 경로를 같은 자로 재려면 "
            "구성 파라미터에서 복원하는 수밖에 없고, 그 사실 자체가 이 분야의 비교 불가능성을 보여준다."),
        "F2_three_paths_differ_by_orders_of_magnitude": (
            f"8B 앵커에서 사용자당 Θ delta(LoRA r=16, q·v, bf16)는 "
            f"{R['Q2_information_density']['Theta_lora_r16_qv_8B']['by_dtype']['bf16']['store_MB']} MB, "
            f"같은 폭의 TTT류 모델이라면 W-경로 상태는 활성 세션당 "
            f"{q1['dense-8B (d=4096, L=32)']['W_path']['MB']} MB다(반사실 수치 — dense 8B에는 W층이 없다). "
            f"E-경로는 **이력 앵커와 함께만** 적을 수 있다 — 14K 토큰 세션에서 텍스트 "
            f"{R['Q2_information_density']['E_text_14k_tokens']['store_KB']} KB이고, "
            f"1M 토큰 이력에서는 실측 rate로 "
            f"{q1['dense-8B (d=4096, L=32)']['E_path']['by_history_anchor']['long_horizon_1M_tokens']['MB_by_rate']['measured_embedRAG']}"
            f"–"
            f"{q1['dense-8B (d=4096, L=32)']['E_path']['by_history_anchor']['long_horizon_1M_tokens']['MB_by_rate']['measured_HippoRAG_v2']}"
            " MB다. 후자의 위쪽 끝은 Θ delta를 넘는다. "
            "수명과 상주 위치가 전부 다르고 크기 순위조차 이력 길이에 달렸으므로 "
            "'용량'을 한 숫자로 말할 수 없다."),
        "F3_quantisation_is_a_precondition_not_an_optimisation": (
            f"저장 바이트당 정보량의 **하한**은 Θ가 bf16에서 "
            f"{R['Q2_information_density']['Theta_lora_r16_qv_8B']['by_dtype']['bf16']['bits_per_stored_byte']} "
            f"bits/byte, int8에서 "
            f"{R['Q2_information_density']['Theta_lora_r16_qv_8B']['by_dtype']['int8']['bits_per_stored_byte']}, "
            f"int4에서 "
            f"{R['Q2_information_density']['Theta_lora_r16_qv_8B']['by_dtype']['int4']['bits_per_stored_byte']}다. "
            "파라미터 하나에 **최소** 3.64 bits가 담기는데 bf16은 16 bits를 쓴다 — 표현 자체에서 "
            "최소 4.4배의 저장 여유가 나온다. 이 여유는 알고리즘 한계가 아니라 dtype 선택이므로, "
            "**delta 양자화는 Θ-경로 배포의 최적화가 아니라 전제다.** "
            "그리고 corpus의 어느 논문도 delta 양자화를 논의하지 않는다. "
            "이 결론은 3.64가 하한이든 상한이든 성립한다 — 분모만으로 정해지기 때문이다."),
        "F3_withdrawn_cross_path_density_claim": (
            "**철회.** 이 실험의 초판은 3.64 bits/param을 상한으로 놓고 'E가 Θ보다 바이트 효율이 "
            "높다'고 비교했다. 원 논문은 이 값을 명시적으로 **하한**이라고 밝힌다"
            "([lm-memorization-capacity §3.2, txt L664] 'we are only ever measuring a lower bound "
            "on model capacity'). 하한을 상한으로 쓰면 Θ층 용량을 과소평가하므로 그 비교는 "
            "방향이 보장되지 않는다. 경로 간 밀도 비교는 이 회계로 판정 불가로 남긴다 — "
            "판정하려면 delta에 직접 정보를 주입해 회수율을 재는 실험이 필요하다."),
        "F4_memory_device_opportunity_is_warm_not_hot": (
            f"Θ-경로가 주류가 되면 사용자 10^6명 기준 웜 스토리지가 LoRA r=16 bf16에서 "
            f"{q3['1e+06_users']['lora_r=16_bf16']['TB']} TB, int4로 내리면 "
            f"{q3['1e+06_users']['lora_r=16_int4']['TB']} TB, r=64 bf16이면 "
            f"{q3['1e+06_users']['lora_r=64_bf16']['TB']} TB다. "
            f"직접 편집 방식으로 개인화하면 {q3['1e+06_users']['direct_edit_full_copy_bf16']['TB']} TB로 "
            "세 자릿수가 뛴다. 기회의 성격은 HBM 대역폭이 아니라 **요청 지연 안에 delta를 끌어올 수 있는 "
            "웜 계층과 그 계층의 쓰기 내구성**이다. 이는 NM이 W-경로에서 짚은 HBM RMW 기회와 다른 자리다."),
        "F5_edit_vs_delta_is_a_serving_fork": (
            "MEMIT/ROME류 직접 편집은 delta 아티팩트를 만들지 않는다. 공유 가중치를 고치므로 "
            "사용자별 개인화가 구조적으로 불가능하고, 하려면 모델 전체 사본이 필요하다. "
            "Θ-경로가 서빙에 도달하려면 delta 형태(LoRA류)여야 한다는 제약이 여기서 나온다. "
            "논문들은 이 갈림길을 논의하지 않는다."),
        "F6_R1_falsifier_fired_E_is_a_rate": (
            "**이 실험이 사전등록한 반증 R-1이 발화했다(2026-08-11).** ch26 §26.5 R-1은 "
            "'E-경로가 임베딩·그래프·다중 인덱스를 누적해 사용자당 저장이 수십 MB를 넘어 "
            "Θ delta 자릿수에 들어오면 표 26-2의 E 행이 무너진다. 필요한 관측은 실험이 아니라 "
            "인쇄 한 줄이다'라고 적었다. 그 인쇄는 2026-06-04에 이미 나와 있었다 — 1M 토큰 "
            "단일 사용자 이력에서 HippoRAG v2 약 62 MB, Mem0 약 12 MB, embedRAG 약 7 MB, "
            "시스템 간 산포 약 9배[2606.06448 §4.7, p.10; Fig. 9c]. 62 MB는 이 실험의 dense-8B "
            "Θ-delta 앵커 16.78 MB를 넘는다. "
            "죽은 것: E 행의 **형태**(고정 바이트). 이제 rate × 이력 길이로 적는다. "
            "살아남은 것 셋: (i) F1/판정 1의 범주 오류 논증은 오히려 증명되었다 — 두 양이 "
            "교차하므로 순위가 없다. (ii) 콜드 계층 배정은 유지된다 — store가 커져도 검색 "
            "지연이 거의 평평하다[Fig. 9d]. (iii) **이 실험의 바이트/토큰 비율이 자릿수까지 "
            "맞는다** — 가정한 fp32 임베딩 12 B/tok과 실측 Mem0 12 B/tok이 일치하고, "
            "상한 건 store의 포화 크기 48,506자 ≈ 48.5 KB[2605.12978 App. G.2, p.53]가 "
            "이 실험의 텍스트 행 56.0 KB와 13.4% 안에서 만난다. "
            "**틀린 것은 회계가 아니라 앵커다** — 14,000 토큰 대화 하나는 실측 레짐보다 약 70배 작다."),
        "honest_limits": (
            "3.64 bits/param은 원 논문이 밝힌 하한이고, 무작위 문자열 암기 실험에서 나온 값이므로 "
            "LoRA delta 적용은 별도의 외삽이다. E-경로 바이트 환산율은 이 실험의 가정이고(실측과 "
            "자릿수까지 맞았으나 여전히 가정이다), 실측된 E 바이트는 dtype 표기가 없는 이질적 "
            "store의 on-disk 발자국이므로 Θ의 dtype 회계와 같은 열에 놓을 수 없다 — 62 MB 대 "
            "16.78 MB는 **자릿수 비교**이지 같은 자로 잰 두 값의 대응이 아니다. "
            "본문 단정문으로 쓸 것은 경로 간 자릿수 차이, dtype 축의 기울기, 사용자 수 스케일링의 "
            "기울기, 그리고 E의 rate와 그 산포이며, 절대 바이트도 경로 간 밀도 비교도 아니다."),
    }

    out = HERE / "result.json"
    out.write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}\n")
    for k, v in R["findings"].items():
        print(f"[{k}]\n{v}\n")


if __name__ == "__main__":
    main()
