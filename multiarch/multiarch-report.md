# 크로스아키텍처 실험 보고서

## Part III pair thesis를 8개 가속기에서 동일하게 측정하다

> **한 줄 결론.** 동일한 Part III 실험(closed-form E3 backbone)을 8개 아키텍처에서 동일하게 돌린 결과, **load-bearing 결론(비율·crossover·bound)은 하드웨어 전체 공간에서 견고**했다: **S\* 는 8개 전부 동일**(HW 무관), **decode 는 7/8 memory-bound**(여유 380–1455×) — 유일 예외는 wafer-scale Cerebras로 roofline **knee** 에 걸린다. 반면 **절대 성능은 5자릿수 갈렸다**(decode 0.0003 ms/tok Cerebras → 31.5 ms/tok MTIA). 절대치는 단정하지 않는다(ideal 하한).

> **핵심.** 이 문서의 최우선 가치는 **성능 수치 신뢰성**이다. 각 twin 은 필드별 provenance + confidence(GOLD/SILVER/BRONZE)로 저작하고 **2라운드 적대 검증**(spec 저작 + 결과 재검증)을 통과시켰다. 발표에서 **단정 가능한 것은 GOLD/SILVER load-bearing 필드에 기반한 비율·순서·bound 뿐**이고, BRONZE(특히 Vera Rubin 전체)와 모든 절대 µs/mJ 은 directional/예비다.

---

## 1. 방법

1. **twin 저작 (라운드 1).** 7개 신규 칩을 다중 독립 검색으로 저작. 각 load-bearing 수치를 vendor whitepaper/datasheet + 3rd-party(SemiAnalysis, Chips&Cheese, Hot Chips, ISCA)로 교차. 저작값을 **독립 검증가 2명**(vendor 위주 / 3rd-party 위주)이 신선한 검색으로 반박 시도 → 조정으로 최종 등급 확정.
2. **결과 재검증 (라운드 2).** 8개 twin headline 을 다시 독립 재확인 + 결과표 산술·물리 감사 + 보고서 과대주장 감사.
3. **실험.** `run_multiarch.py` 가 E3 closed-form 을 임의 twin 으로 파라미터화(HBM GPU + SRAM-only 공통). **정확성 게이트**: H100 에서 확정 E3 anchor 를 <0.02% 재현(6.442 GB/tok, S\* 65536, 134.2 MB).
4. **스키마 검증.** 8개 twin 전부 hatir `validate_twin` 통과(error 0).

anchor 설정: neural-mem-1.3B (d=2048, m=16, L=24, GQA-8, bf16).

---

## 2. 대상 8종 + 신뢰도 등급

| twin | 칩 | tier | backing |
|---|---|---|---|
| h100 | NVIDIA H100 SXM5 | GOLD | HBM3 |
| mtia2 | Meta MTIA v2 | GOLD | LPDDR5 |
| b100 | NVIDIA B100 (Blackwell) | SILVER | HBM3e |
| mi355x | AMD MI355X (CDNA4) | SILVER | HBM3e |
| tpu-v7 | Google TPU v7 Ironwood | SILVER※ | HBM3e |
| groq-lpu | Groq LPU/TSP | SILVER | SRAM |
| wse3 | Cerebras WSE-3 | SILVER | SRAM(wafer) |
| vr100 | NVIDIA Vera Rubin | BRONZE | HBM4 |

> **주의.** vr100 은 내부 전면 미공개(BW 13–22 TB/s·BF16 5–13 PF 범위) → 전체 directional/예비, **발표 단정 금지**. tpu-v7(※)은 backing/compute 는 GOLD 이나 on-chip VMEM(reuse-tier)이 미공개 추정이라 **residency 주장만 BRONZE 로 강등**해 인용한다. groq/wse3 는 HBM 이 없는 SRAM 기반이라 backing tier 가 on-chip SRAM 이다.

---

## 3. 결과 — 하드웨어 스펙 (anchor 단일 가속기)

| twin | tier | backing | BW (TB/s) | dense BF16/FP16 (PF) | 용량 |
|---|---|---|---|---|---|
| h100 | GOLD | HBM3 | 3.35 | 0.99 | 80 GB |
| b100 | SILVER | HBM3e | 8.0 | 1.8 | 192 GB |
| mi355x | SILVER | HBM3e | 8.0 | 2.52 | 288 GB |
| tpu-v7 | SILVER | HBM3e | 7.38 | 2.31 | 192 GB |
| vr100 | BRONZE | HBM4 | 22.0※ | 6.25※ | 288 GB |
| mtia2 | GOLD | LPDDR5 | 0.205 | 0.177 | 128 GB |
| groq-lpu | SILVER | SRAM | 80.0† | 0.188 | 220 MB |
| wse3 | SILVER | SRAM | 21000 | 12.5‡ | 44 GB |

## 4. 결과 — pair thesis 지표

| twin | decode ms/tok | ridge (F/B) | S\* (tok) | C\* | Bmax(10ms) | bound (여유) |
|---|---|---|---|---|---|---|
| h100 | 1.923 | 295 | 65536 | 337 | 5.2 | memory (497×) |
| b100 | 0.805 | 225 | 65536 | 249 | 12.4 | memory (379×) |
| mi355x | 0.805 | 315 | 65536 | 362 | 12.4 | memory (530×) |
| tpu-v7 | 0.873 | 313 | 65536 | 359 | 11.5 | memory (527×) |
| vr100 | 0.293※ | 284 | 65536 | 323 | 34.2※ | memory (478×) |
| mtia2 | 31.457 | 864 | 65536 | 1180 | 0.32 | memory (1455×) |
| groq-lpu | 0.081† | 2.35 | 65536 | 2.4 | 124 | memory (4×) |
| wse3 | 0.0003 | 0.60 | 65536 | 0.6 | 32596 | **knee (1.01×)** |

주: ※ vr100 절대치는 BRONZE(범위 하 한 draw) — directional. † groq BW 는 vendor aggregate 80 TB/s; ISCA effective ~27.5 TiB/s 면 ridge ~6·ms/tok ~3배 — 절대치 유보. ‡ wse3 12.5 PF = **dense** FP16(라운드2가 잡은 교정: 125 PF 는 Cerebras **sparse** 마케팅 수치, 10× 낙관; 용량·BW 는 GOLD 불변).

---

## 5. 핵심 발견

**① S\* (KV↔TTT crossover)는 하드웨어 무관 — 8개 전부 65536 token.** S\* = m·d²·(dtype비)/kv_dim 로 **순수 workload 속성**이다. backing 이 HBM 이든 LPDDR 이든 wafer SRAM 이든 값이 같다 → "상쇄 논증"의 직접 실증: 이 crossover 는 BW 값에도 이용률에도 movable 하지 않다. **발표에서 가장 강하게 단정 가능한 결과.**

**② decode 는 7/8 memory-bound (여유 380–1455×), 유일 예외 Cerebras 는 roofline knee.** decode AI 0.59 는 HBM/LPDDR 6종의 ridge(225–864)보다 2–3자릿수 낮아 memory-bound 가 압도적으로 견고하다(스펙 100배 오차로도 불변). Groq 는 여유 ~4× 로 여전히 memory-bound. **단 Cerebras WSE-3 만은 ridge 0.60 ≈ AI 0.59**(여유 ~1%) — wafer SRAM 대역폭(21 PB/s)이 dense FLOPS(12.5 PF) 대비 워낙 커서 decode RMW 가 memory-bound 가 아니라 **balanced(knee)** 가 된다. 즉 "decode-state 는 memory-bound"는 HBM/LPDDR/on-chip-SRAM(Groq)까지는 견고하되 극단적 BW-rich 한 wafer-scale 에서는 knee 로 이동한다 — **이 경계 자체가 결과다.**

**③ 절대 decode 비용만 5자릿수 갈린다 — 그 축은 "state 가 어디 사느냐"다.**

- **wafer SRAM(Cerebras)**: 전체 state 3.2 GB 가 44 GB on-wafer SRAM 에 상주, 21 PB/s 로 RMW → 0.0003 ms/tok. decode-state 병목이 사실상 소멸.
- **on-chip SRAM(Groq)**: layer 당 state(134 MB)가 220 MB SRAM 에 상주 → 값쌈.
- **HBM GPU(H100/B100/MI355X/TPU)**: state 가 HBM 에서 스트리밍(L2 50 MB 엔 안 맞음) → 0.8–1.9 ms/tok(GOLD/SILVER), BW 에 비례. (Rubin 0.29 ms/tok 는 BRONZE 이라 범위 제외.)
- **LPDDR(MTIA v2)**: 204.8 GB/s 로 가장 느림 → 31.5 ms/tok. 이 workload 엔 BW-starved.

→ pair thesis 의 "decode-state 배치는 designable knob" 이 8개 아키텍처에서 정량 확인됐다. **on-chip 용량·대역폭이 클수록 이 memory-centric 부하가 값싸진다** — SRAM-heavy 설계(Cerebras/Groq)가 이 병목을 구조적으로 해소한다. 단 이 절대치들은 ideal roofline 하한이며 단정하지 않는다(실측은 MFU/MBU 만큼 느림).

---

## 6. 신뢰성 — per-quantity 로 무엇을 단정하고 무엇을 유보

| 결과 | 어디에 의존 | 단정 가능? |
|---|---|---|
| S\* = 65536 (전 아키텍처) | workload config 만 (HW 무관) | 예 — 전부 (BRONZE twin 포함) |
| bound = memory (7/8) | ridge ≫ 0.59 (380–1455×) | 예 — HBM/LPDDR/Groq |
| Cerebras = knee (예외) | ridge 0.60 ≈ AI 0.59 | 예 — 단 "경계(knee)"로 |
| decode 비용 순서(SRAM≪HBM≪LPDDR) | backing BW (GOLD/SILVER) | 예 — vr100 제외 |
| 절대 ms/tok·µs·mJ | backing BW·energy + ideal 가정 | 유보 — ideal 하한, 실측 예정 |
| residency (on-die fit) | reuse-tier 용량 | 예 — 단 tpu-v7 은 BRONZE |
| vr100 전체 | 미공개 스펙 | 유보 — directional/예비 |

---

## 부록 A. 필드별 출처 (load-bearing headline)

각 twin 의 3대 load-bearing 필드(backing 대역폭·device peak·용량)의 등급과 출처. 전체 필드·미해결은 `specs/*.json`.

| twin | BW | peak | 용량 | 대표 출처 |
|---|---|---|---|---|
| h100 | GOLD | GOLD | GOLD | NVIDIA H100 whitepaper |
| b100 | GOLD | GOLD | GOLD | cudocompute Blackwell, wccftech, exxact |
| mi355x | GOLD | GOLD | GOLD | AMD MI355X datasheet, Chips&Cheese, CDNA4 whitepaper |
| tpu-v7 | GOLD | GOLD | GOLD | Google Cloud tpu7x docs, SemiAnalysis |
| mtia2 | GOLD | GOLD | GOLD | ai.meta.com MTIA blog, MTIA-ISCA25, ServeTheHome |
| groq-lpu | SILVER | SILVER | GOLD | Groq Spec Sheet v1.5, ISCA 2020 TSP paper |
| wse3 | GOLD | (교정)‡ | GOLD | cerebras.ai architecture |
| vr100 | BRONZE | BRONZE | BRONZE | GTC 발표(288GB HBM4), Spheron/ThunderCompute (datasheet 없음) |

‡ wse3 peak 는 라운드2에서 sparse(125 PF)→dense(12.5 PF)로 교정(부록 B). groq BW 는 vendor aggregate 80 TB/s vs ISCA effective ~27.5 TiB/s 정의차 → 절대치 유보. vr100 은 vendor datasheet PDF 부재로 전 필드 BRONZE.

---

## 부록 B. 리뷰 로그 — 다중 라운드 검증 감사추적

### 라운드 1 — twin 저작 + 이중 적대검증 (workflow, 28 agents, 0 error)
칩당: 다중 독립 검색 → 신선 검색 반박 2명 → 조정. 잡은 것(예): B100 SM수 vs clock trade-off, TPU VMEM 미공개, Groq SRAM BW 정의차, Vera Rubin 전면 미공개(BRONZE). 전부 validate_twin PASS.

### 라운드 2 — 결과 재검증 + 감사 (workflow, 10 agents, 0 error)

**확정(confirm) — 신선 검색으로 재확인, 수정 불요:** b100/mi355x/tpu-v7/mtia2/groq-lpu/h100 의 4개 headline 전부 vendor+3rd-party 재확인. vr100 은 "진짜 미공개"임을 재확인 → BRONZE 정당.

**수정 반영(applied fixes):**

> **주의.** 리뷰가 실제로 잡아 고친 것들 — 다중 검증이 작동한 증거다.

1. **[MAJOR] wse3 compute peak 교정.** 125 PFLOPS 는 Cerebras **sparse** FP16 마케팅 수치(10× sparsity). **dense = 12.5 PFLOPS** 로 교정. 파급: ridge 5.95→0.60 → **Cerebras 만 roofline knee**(나머지 7개는 memory-bound). 오류 교정이 오히려 더 정확한 발견을 드러냄.
2. **[MINOR] 표 B_max(wse3) 2596→32596** (전사 오류, results/wse3.json 과 일치).
3. **[MINOR] robustness 주장 범위 한정.** "100배 오차로도 memory-bound 불변"은 HBM/LPDDR(여유 380–1455×)에만 참. Groq ~4×, Cerebras ~1%(knee)로 명시.
4. **[NIT] HBM-GPU decode 범위** 하단 0.29(vr100 BRONZE) → GOLD/SILVER 범위 0.8–1.9 로 교체.
5. **[NIT] 표기.** 헤더 "dense BF16"→"BF16/FP16", tpu-v7 tier ⚠️ 복원.

**감사 결론.** 두 감사관 모두 physical_ok=true. 남은 인정 한계: 절대 ms/tok·µs·mJ 은 ideal roofline 하한(단정 안 함); AI 정의는 E3 검증본 관례(Cerebras knee 는 ~1% 여유라 "경계"로만 진술); groq 절대치는 aggregate BW 기반이라 유보.

---

## 부록 C. 재현

```
cd ~/repos/neural-memory-study
.venv/bin/python multiarch/run_multiarch.py --validate   # H100==확정 E3 anchor 게이트
.venv/bin/python multiarch/run_multiarch.py --all         # 8개 twin -> results/*.json + 표
```

twin JSON: `multiarch/twins/*.json` (validate_twin PASS). 필드별 출처·등급·미해결: `multiarch/specs/*.json`. driver 는 E3(`experiments/E3-analytical`)의 공식을 verbatim 파라미터화 — H100 재현 게이트가 충실성 보증. 성능 수치 신뢰성의 상위 근거(HATIR·HAT spec fidelity)는 본서 부록 E 참조.
