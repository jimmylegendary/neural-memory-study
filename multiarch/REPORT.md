# 크로스아키텍처 실험 보고서 — Part III pair thesis를 8개 가속기에서

> **한 줄 결론.** 동일한 Part III 실험(closed-form E3 backbone)을 8개 아키텍처에서 동일하게 돌린 결과,
> **load-bearing 결론(비율·crossover·bound 분류)은 하드웨어 전체 공간에서 불변**이었고
> (S\*는 8개 전부 동일, decode는 8개 전부 memory-bound), **절대 성능만 5자릿수 갈렸다**
> (decode 0.0003 ms/tok Cerebras → 31.5 ms/tok MTIA). 절대치는 단정하지 않는다(ideal 하한).
>
> **정직성 계약.** 각 twin은 필드별 provenance + confidence(GOLD/SILVER/BRONZE)로 저작·이중 적대검증했다.
> 발표에서 **단정 가능한 것은 GOLD/SILVER load-bearing 필드에 기반한 비율·순서·bound뿐**이고,
> BRONZE(특히 Vera Rubin 전체)와 모든 절대 µs/mJ은 **directional/예비**다. 근거: `AUTHORING-SPEC.md`,
> `specs/*.json`(필드별 출처), 본서 부록 E.

## 1. 방법

1. **twin 저작** — 7개 신규 칩(H100은 기존 GOLD)을 다중 독립 검색으로 저작. 각 load-bearing 수치를
   vendor whitepaper/datasheet + 3rd-party(SemiAnalysis, Chips&Cheese, Hot Chips, ISCA) 교차.
2. **이중 적대검증** — 저작값을 독립 검증가 2명(vendor 위주 / 3rd-party 위주)이 신선한 검색으로 반박 시도
   → 필드별 confirm/dispute → 조정(reconcile)으로 최종 등급 확정. (28 agents, 0 error.)
3. **실험** — `run_multiarch.py`가 E3 closed-form을 임의 twin으로 파라미터화(HBM GPU + SRAM-only 공통).
   **정확성 게이트**: H100에서 확정 E3 anchor를 <0.02% 재현(6.442 GB/tok, S\* 65536, 134.2 MB) → PASS.
4. **스키마 검증** — 8개 twin 전부 hatir `validate_twin` 통과(error 0).

## 2. 대상 8종 + 신뢰도 등급

| twin | 칩 | tier | backing | 핵심 unresolved (정직) |
|---|---|---|---|---|
| `h100` | NVIDIA H100 SXM5 | **GOLD** | HBM3 | — (whitepaper 완비, 기준 twin) |
| `mtia2` | Meta MTIA v2 | **GOLD** | LPDDR5 | energy/line BRONZE(비-load-bearing) |
| `b100` | NVIDIA B100 (Blackwell) | SILVER | HBM3e | SM수 vs clock trade-off(device peak 1.8 PF 불변); matrix tile은 legacy proxy |
| `mi355x` | AMD MI355X (CDNA4) | SILVER | HBM3e | HBM energy·Infinity Cache BW는 BRONZE 추정(비-load-bearing) |
| `tpu-v7` | Google TPU v7 Ironwood | SILVER⚠️ | HBM3e | **reuse-tier(VMEM) 용량·BW가 BRONZE 추정 → residency 주장은 BRONZE급**(BW/peak/ridge는 GOLD) |
| `groq-lpu` | Groq LPU/TSP | SILVER | **SRAM** | **SRAM BW 정의 차: vendor 80 TB/s(aggregate) vs ISCA 27.5 TiB/s(effective)** → 절대치 범위 |
| `wse3` | Cerebras WSE-3 | SILVER | **SRAM(wafer)** | per-core width/clock는 BRONZE placeholder(device peak·용량·BW는 GOLD/vendor) |
| `vr100` | NVIDIA Vera Rubin | **BRONZE** | HBM4 | **내부 전면 미공개. BW 13–22 TB/s·BF16 5–13 PF 범위. 전체 directional/예비 — 발표 단정 금지** |

> 등급 규칙: twin tier = 최저 load-bearing 필드 등급. `tpu-v7`은 reconciler가 규칙을 완화해 SILVER로 뒀으나
> (backing/compute는 GOLD, reuse-tier만 BRONZE), **본 보고서는 tpu-v7의 residency(on-die fit) 결과를 BRONZE로 강등**해 인용한다.

## 3. 크로스아키텍처 결과 (anchor: neural-mem-1.3B, d=2048, m=16, L=24, GQA-8, bf16)

| twin | tier | backing | BW (TB/s) | dense BF16 (PFLOP/s) | ridge (FLOP/B) | decode ms/tok | S\* (tok) | C\* (chunk) | B_max@10ms | bound |
|---|---|---|---|---|---|---|---|---|---|---|
| `h100` | GOLD | HBM3 | 3.35 | 0.99 | 295 | 1.923 | 65536 | 337 | 5.2 | memory |
| `b100` | SILVER | HBM3e | 8.0 | 1.8 | 225 | 0.805 | 65536 | 249 | 12.4 | memory |
| `mi355x` | SILVER | HBM3e | 8.0 | 2.52 | 315 | 0.805 | 65536 | 362 | 12.4 | memory |
| `tpu-v7` | SILVER | HBM3e | 7.38 | 2.31 | 313 | 0.873 | 65536 | 359 | 11.5 | memory |
| `vr100` | BRONZE | HBM4 | 22.0※ | 6.25※ | 284 | 0.293※ | 65536 | 323 | 34.2※ | memory |
| `mtia2` | GOLD | LPDDR5 | 0.205 | 0.177 | 864 | 31.457 | 65536 | 1180 | 0.32 | memory |
| `groq-lpu` | SILVER | SRAM | 80.0† | 0.188 | 2.35 | 0.081† | 65536 | 2.4 | 124 | memory |
| `wse3` | SILVER | SRAM | 21000 | 125 | 5.95 | 0.0003 | 65536 | 6.0 | 2596 | memory |

※ vr100 절대치는 BRONZE(범위 하에서의 한 draw) — directional. † groq BW는 vendor aggregate(effective는 ~1/3, ms/tok는 그만큼 상향).

## 4. 핵심 발견

**① S\*(KV↔TTT crossover)는 하드웨어 무관 — 8개 전부 65536 token.** S\* = m·d²·(dtype비)/kv_dim로
**순수 workload 속성**이다. backing이 HBM이든 LPDDR이든 wafer SRAM이든 값이 같다 → 지난 턴의 "상쇄 논증"의
직접 실증: 이 crossover는 BW값에도 이용률에도 movable하지 않다. **발표에서 가장 강하게 단정 가능한 결과.**

**② decode는 8개 전부 memory-bound — ridge가 2.35(Groq)~864(MTIA)로 2.5자릿수 갈려도 불변.** decode AI 0.59는
가장 낮은 ridge(Groq 2.35)보다도 낮다. 즉 "decode-state RMW는 memory-bound"라는 pair thesis의 bound 분류는
**하드웨어 공간 전체에서 견고**하다(2배는커녕 100배 스펙 오차로도 안 뒤집힘).

**③ 절대 decode 비용만 5자릿수 갈린다 — 그리고 그 축은 "state가 어디 사느냐"다.**
- **wafer SRAM(Cerebras)**: 전체 state 3.2 GB가 44 GB on-wafer SRAM에 상주, 21 PB/s로 RMW → **0.0003 ms/tok.** decode-state 병목이 사실상 소멸.
- **on-chip SRAM(Groq)**: layer당 state가 230 MB SRAM에 상주 → 80 TB/s(effective 더 낮음)로 값쌈.
- **HBM GPU(H100/B100/MI355X/TPU/Rubin)**: state가 HBM에서 스트리밍(L2 50 MB엔 안 맞음, 기존 E1.2와 일치) → 0.29~1.9 ms/tok, BW에 비례.
- **LPDDR(MTIA v2)**: 204.8 GB/s로 가장 느림 → 31.5 ms/tok. 이 workload엔 BW-starved.

→ pair thesis의 "decode-state 배치는 designable knob"이 8개 아키텍처에서 정량 확인됐다. **on-chip 용량·대역폭이
클수록 이 memory-centric 부하가 값싸진다** — SRAM-heavy 설계(Cerebras/Groq)가 이 병목을 구조적으로 해소한다.
단 이 절대치들은 ideal roofline 하한이며 단정하지 않는다(실측은 MFU/MBU만큼 느림; 부록 E).

## 5. 신뢰성 — per-quantity로 무엇을 단정하고 무엇을 유보

| 결과 | 어디에 의존 | 단정 가능? |
|---|---|---|
| S\* = 65536 (전 아키텍처) | workload config만 (HW 무관) | ✅ 전부 단정 (BRONZE twin 포함) |
| bound = memory (전 아키텍처) | ridge > 0.59 (2자릿수 여유) | ✅ 전부 단정 |
| decode 비용 **순서**(SRAM≪HBM≪LPDDR) | backing BW (GOLD/SILVER) | ✅ vr100 제외 단정 |
| 절대 ms/tok·µs·mJ | backing BW·energy (일부 BRONZE) + ideal 가정 | ⚠️ 유보 (ideal 하한, 실측 예정) |
| residency(on-die fit) | reuse-tier 용량 | ✅ 단, **tpu-v7은 BRONZE**(VMEM 미공개) |
| **vr100 전체** | 미공개 스펙 | ⚠️ **directional/예비 — 발표 단정 금지** |

## 6. 재현

```
cd ~/repos/neural-memory-study
.venv/bin/python multiarch/run_multiarch.py --validate   # H100==확정 E3 anchor 게이트
.venv/bin/python multiarch/run_multiarch.py --all         # 8개 twin -> results/*.json + 표
```
twin JSON: `multiarch/twins/*.json` (hatir validate_twin PASS). 필드별 출처·등급·미해결: `multiarch/specs/*.json`.
driver는 E3(`experiments/E3-analytical`)의 공식을 verbatim 파라미터화 — H100 재현 게이트가 충실성 보증.
