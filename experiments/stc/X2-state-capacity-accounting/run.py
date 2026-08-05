#!/usr/bin/env python3
"""X2 — 세 경로의 상태 회계: 무엇이 어디에 몇 바이트로 쌓이고, 그 바이트에 정보가 얼마나 담기는가.

왜 필요한가
----------
"Memory device 회사에 어떤 기회가 있는가"는 세 경로가 만드는 상태의 **양·위치·수명·
트래픽**이 다르다는 사실 위에서만 답할 수 있다. 그런데 corpus의 어느 논문도 상태 용량을
바이트로 보고하지 않는다(cost_table의 C_cap이 거의 전부 absent). 이 실험은 논문들이
보고한 **구성 파라미터**로부터 바이트를 복원하고, 별도 논문이 측정한 **기억 용량 상한**과
결합해 세 경로를 같은 자로 잰다.

세 경로의 상태
-------------
E : learned context / 그래프 / 벡터.  세션·사용자 단위. 모델 밖. 콜드~웜.
W : fast weights.                    활성 세션 단위. HBM 상주. 토큰마다 read-modify-write.
Θ : 가중치 delta.                     사용자 단위. 영속. 요청 시 로드 필요.

핵심 질문
--------
Q1  같은 앵커에서 세 경로가 만드는 바이트는 각각 얼마인가.
Q2  그 바이트에 담기는 정보량(bits)은 얼마인가 — 바이트당 정보 밀도가 경로마다 다른가.
Q3  Θ-경로가 주류가 되면 사용자 수에 따라 웜 스토리지가 어떻게 늘어나는가.
Q4  직접 편집(MEMIT류)과 delta(LoRA류)는 서빙 구조상 무엇이 다른가.

등급
----
EXPLORATION-GRADE / ACCOUNTING. 측정이 아니라 회계다.
- 신뢰할 것: 경로 간 **자릿수 차이**, 밀도의 **순서**, 사용자 수 스케일링의 **기울기**.
- 신뢰하지 말 것: 특정 배포의 절대 바이트, 3.64 bpp를 LoRA에 적용한 값의 정확도.
- 3.64 bits/parameter는 무작위 문자열로 from-scratch 학습한 소형 GPT에서 측정된 값이며,
  이를 LoRA delta에 적용하는 것은 **이 노트의 외삽**이다. 상한으로만 쓴다.
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
        "caveat": "무작위 uniform 문자열을 from-scratch 학습한 소형 GPT에서 측정. "
                  "LoRA delta 적용은 이 실험의 외삽이며 상한으로만 사용",
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
                "BPP-EXTRAPOLATED": "3.64 bits/param을 LoRA delta에 적용한 것은 외삽. 상한으로만",
                "CORPUS-SILENT-ON-BYTES": "상태 용량을 바이트로 보고한 논문이 corpus에 없음",
                "E-PATH-TOKENS-ONLY": "E-경로 용량은 토큰 수만 보고됨. 바이트 환산은 이 실험의 가정",
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
                    "info_capacity_bits_upper": round(p * SOURCED["bits_per_parameter"]["value"]),
                    "info_capacity_MB_upper": round(p * SOURCED["bits_per_parameter"]["value"] / 8 / 1e6, 2),
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
        row["E_path"] = {
            "variants": e,
            "residence": "콜드/벡터 스토어 → 질의마다 ret() + prefill",
            "lifetime": "세션~사용자",
            "note": "논문이 바이트를 보고하지 않아 토큰→바이트 환산은 가정",
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
        "description": ("저장 바이트 1개에 담기는 정보 bits. 분자는 [lm-memorization-capacity]의 "
                        "3.64 bits/param 상한, 분모는 저장 dtype. E는 텍스트 액면가와 3:1 압축 가정"),
        "Theta_lora_r16_qv_8B": {
            "params": p16,
            "info_bits_upper": round(theta_info_bits),
            "info_MB_upper": round(theta_info_bits / 8 / 1e6, 2),
            "by_dtype": theta_by_dtype,
        },
        "E_text_14k_tokens": {
            "store_bytes": e_store_bytes,
            "store_KB": round(e_store_bytes / 1e3, 1),
            "bits_per_stored_byte_face": e_density_face,
            "bits_per_stored_byte_compressed_3to1": round(e_density_compressed, 3),
        },
        "ratio_E_over_Theta_compressed_by_dtype": {
            dt: round(e_density_compressed / v["bits_per_stored_byte"], 2)
            for dt, v in theta_by_dtype.items()
        },
        "note": ("bf16에서 Θ가 E보다 바이트 비효율인 것은 표현이 정보를 못 담아서가 아니라 "
                 "파라미터 하나에 담기는 3.64 bits를 16 bits에 저장하기 때문이다. "
                 "int4로 내리면 순서가 뒤집힌다 — 즉 이것은 알고리즘 한계가 아니라 dtype 선택이다."),
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
            f"E-경로 저장은 {R['Q2_information_density']['E_text_14k_tokens']['store_KB']} KB, "
            f"같은 폭의 TTT류 모델이라면 W-경로 상태는 활성 세션당 "
            f"{q1['dense-8B (d=4096, L=32)']['W_path']['MB']} MB다(반사실 수치 — dense 8B에는 W층이 없다). "
            "수명과 상주 위치가 전부 다르므로 '용량'을 한 숫자로 말할 수 없다."),
        "F3_theta_density_is_a_dtype_choice_not_a_limit": (
            f"저장 바이트당 정보량은 Θ가 bf16에서 "
            f"{R['Q2_information_density']['Theta_lora_r16_qv_8B']['by_dtype']['bf16']['bits_per_stored_byte']} "
            f"bits/byte, int8에서 "
            f"{R['Q2_information_density']['Theta_lora_r16_qv_8B']['by_dtype']['int8']['bits_per_stored_byte']}, "
            f"int4에서 "
            f"{R['Q2_information_density']['Theta_lora_r16_qv_8B']['by_dtype']['int4']['bits_per_stored_byte']}다. "
            f"3:1 압축을 가정한 텍스트(2.67 bits/byte) 대비 비율은 dtype에 따라 "
            f"{R['Q2_information_density']['ratio_E_over_Theta_compressed_by_dtype']}로 뒤집힌다. "
            "즉 Θ-경로가 바이트 비효율로 보이는 것은 파라미터당 3.64 bits를 16 bits에 담기 때문이지 "
            "표현의 한계가 아니다. **delta 양자화는 Θ-경로 배포의 선택 사항이 아니라 전제다** — "
            "그리고 corpus의 어느 논문도 delta 양자화를 논의하지 않는다."),
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
        "honest_limits": (
            "3.64 bits/param은 무작위 문자열 암기 실험에서 나온 값이고 LoRA delta 적용은 외삽이다. "
            "E-경로 바이트는 토큰 수에서 환산한 가정치다. 본문 단정문으로 쓸 것은 자릿수 차이와 "
            "밀도의 순서이며, 절대 바이트가 아니다."),
    }

    out = HERE / "result.json"
    out.write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}\n")
    for k, v in R["findings"].items():
        print(f"[{k}]\n{v}\n")


if __name__ == "__main__":
    main()
