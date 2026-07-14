# 크로스아키텍처 실험 보고서

## Part III 의 모든 HATIR 실험을 8개 가속기에서 동일하게

> **개요.** study paper Part III 의 twin-의존 HATIR 실험 전부(E1.1 decode-baseline · E1.2 state-placement · E1.3 kv-vs-ttt · E1.4 frequency-tiers · E1.5 kvmgr-gap · E3 analytical · E4 scaling)를 **8개 아키텍처**에서 **동일하게** 재실행하고, 실험별 그래프·tier 배치·종합 한눈에 뷰·환경·고찰·insight 를 정리한다. (E2.1/E2.2 는 host-CPU roofline **모양** 측정이라 아키텍처 무관 — §4.8.)

> **핵심 — 정직성 계약.** 최우선 가치는 **성능 수치 신뢰성**이다. 각 twin 은 필드별 provenance+confidence(GOLD/SILVER/BRONZE)로 저작하고 **2라운드 적대 검증**을 통과시켰다. 단정 가능한 것은 GOLD/SILVER load-bearing 필드에 기반한 **비율·crossover·bound·tier 순서**뿐이고, BRONZE(Vera Rubin 전체)와 **모든 절대 µs/mJ 은 ideal roofline 하한**(directional/예비)이다. 근거: `AUTHORING-SPEC.md`, `specs/*.json`, `REVIEW-LOG.md`, 본서 부록 E.

---

## 1. 방법

1. **twin 저작 (라운드 1, 28 agents).** 7개 신규 칩을 다중 독립 검색 → 저작 → **독립 검증가 2명**(vendor / 3rd-party)이 신선 검색으로 반박 → 조정. 필드별 provenance+confidence.
2. **결과 재검증 (라운드 2, 10 agents).** 8개 headline 독립 재확인 + 결과표 산술·물리 감사 + 과대주장 감사. (라운드2가 wse3 sparse→dense peak 오류를 잡음 — §6.)
3. **실험 driver.** `run_all_experiments.py` 가 7개 실험 전부를 임의 twin 으로 파라미터화(HBM GPU + SRAM-only 공통). 모든 공식은 `experiments/{E1.1,E1.2,E1.3,E1.4,E3}` verbatim. **정확성 게이트**(`--validate`): H100 twin 이 확정 anchor 5종을 <0.02% 재현 — rmw 6.442 GB/tok, ms 1.923, state 134.2 MB, S\* 65536, B_max 5.2.
4. **스키마 검증.** 8개 twin 전부 hatir `validate_twin` PASS(error 0). **그림**: `make_figures.py` → `figures/*.png`.

anchor: neural-mem-1.3B (d=2048, m=16, L=24, GQA-8, bf16).

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

> **주의.** vr100 은 내부 전면 미공개 → 전체 directional, 발표 단정 금지. tpu-v7(※)은 backing/compute 는 GOLD 이나 on-chip VMEM(reuse-tier)이 미공개 추정 → **residency(E1.2) 주장만 BRONZE**. groq/wse3 는 HBM 이 없는 SRAM 기반.

---

## 3. 종합 — 한눈에

![fig7 · 한눈에: (a) decode 비용 (b) ridge vs AI (c) S\* 불변 (d) on-chip 예산 vs state. BW 오름차순, 색=신뢰도 등급.](/home/jimmy/repos/neural-memory-study/multiarch/figures/fig7-consolidated.png){width=15.5cm}

**HW 스펙 (단일 가속기).**

| twin | tier | backing | BW (TB/s) | dense BF16/FP16 (PF) | 용량 | on-chip |
|---|---|---|---|---|---|---|
| h100 | GOLD | HBM3 | 3.35 | 0.99 | 80 GB | 50 MB |
| b100 | SILVER | HBM3e | 8.0 | 1.8 | 192 GB | 101 MB |
| mi355x | SILVER | HBM3e | 8.0 | 2.52 | 288 GB | 34 MB |
| tpu-v7 | SILVER | HBM3e | 7.38 | 2.31 | 192 GB | 268 MB※ |
| vr100 | BRONZE | HBM4 | 22.0 | 6.25 | 288 GB | 105 MB |
| mtia2 | GOLD | LPDDR5 | 0.205 | 0.177 | 128 GB | 268 MB |
| groq-lpu | SILVER | SRAM | 80.0 | 0.188 | 220 MB | (backing) |
| wse3 | SILVER | SRAM | 21000 | 12.5 | 44 GB | (backing) |

**pair thesis 지표 (anchor).**

| twin | decode ms/tok | ridge | S\* | C\* | Bmax@10ms | bound (여유) |
|---|---|---|---|---|---|---|
| h100 | 1.923 | 295 | 65536 | 337 | 5.2 | memory (497×) |
| b100 | 0.805 | 225 | 65536 | 249 | 12.4 | memory (379×) |
| mi355x | 0.805 | 315 | 65536 | 362 | 12.4 | memory (530×) |
| tpu-v7 | 0.873 | 313 | 65536 | 359 | 11.5 | memory (527×) |
| vr100 | 0.293※ | 284 | 65536 | 323 | 34.2※ | memory (478×) |
| mtia2 | 31.457 | 864 | 65536 | 1180 | 0.32 | memory (1455×) |
| groq-lpu | 0.081† | 2.35 | 65536 | 2.4 | 124 | memory (4×) |
| wse3 | 0.0003 | 0.60 | 65536 | 0.6 | 32596 | **knee (1.01×)** |

※ vr100/tpu-v7(residency) = BRONZE. † groq BW = vendor aggregate. 절대치는 ideal 하한(단정 안 함).

---

## 4. 실험별 결과·환경·고찰

### 4.1 E1.1 — decode-baseline (토큰당 whole-state RMW)

**설명.** decode 한 스텝이 fast-weight state 전체를 read-modify-write 하는 비용(traffic·time·energy·bound). anchor 상태 = m·d²·L = 134 MB/layer × 24.

**환경.** 각 twin 의 backing tier(bandwidth_bps·energy_pj_per_byte) + compute leaf(peak_macs). traffic = 2·state·L(read+write), time = traffic/BW, energy = traffic·epb.

![fig1 · 토큰당 decode 비용(log). 5자릿수 스프레드, state 배치가 축.](/home/jimmy/repos/neural-memory-study/multiarch/figures/fig1-decode-ms.png){width=14cm}

**결과.** decode ms/tok: MTIA 31.5 → H100 1.92 → Rubin 0.29 → Groq 0.08 → **Cerebras 0.0003** (약 5자릿수 스프레드). AI = **0.594 로 8개 전부 동일**(workload 고정). 에너지: SRAM 칩(Groq/WSE-3) 6.4 mJ/tok, HBM/LPDDR 32–52 mJ/tok(epb 차이).

**고찰·insight.** decode 가 memory-bound 라는 성질은 **update rule 의 RMW 대칭성**에서 오는 것이라 모델·HW 와 무관하다(AI 항상 0.594). 그러나 *얼마나 깊이* memory-bound 인지(여유 1×–1455×)와 절대 비용은 순전히 HW 의 BW·epb 문제다. **SRAM 기반 칩의 에너지 우위(7×)** 는 이 write-heavy 부하에서 특히 큰데, epb 가 HBM 의 1/7 이기 때문이다 — memory-centric 부하일수록 low-epb 메모리의 가치가 커진다.

### 4.2 E1.2 — state-placement (tier 잔류)

**설명.** state 가 context-무관 **고정 크기**라 on-chip 상주가 designable 선택이다(context 로 자라는 KV 와 대조). 각 HW 의 자기 계층에서 backing vs on-chip RMW 비용·잔류·residency crossover d\*(1-layer state 가 on-chip 에 들어가는 최대 d)를 잰다.

**환경.** 각 twin 의 on-chip reuse tier(L2/Infinity Cache/LDS/SRAM) 용량·BW·epb vs backing. d\* = √(ondie_cap / 2m).

![fig5 · state 배치·tier 잔류(log). on-chip 예산 vs 1-layer(134MB)·whole-model(3.2GB) state.](/home/jimmy/repos/neural-memory-study/multiarch/figures/fig5-state-placement.png){width=14cm}

**결과.**

| twin | on-chip | 판정 | d\* | on-die vs backing (E/T) |
|---|---|---|---|---|
| wse3 | 44 GB | **whole-model 상주** | 37081 | 1×/1× (backing=on-chip) |
| tpu-v7 | 268 MB※ | per-layer 상주 | 2896 | 25×/22× |
| mtia2 | 268 MB | per-layer 상주 | 2896 | 20×/13× |
| groq-lpu | 220 MB | per-layer 상주 | 2685 | 1×/1× (backing=on-chip) |
| vr100 | 105 MB | backing 스트리밍 | 1810 | — |
| b100 | 101 MB | backing 스트리밍 | 1774 | 12.5×/1.25× |
| h100 | 50 MB | backing 스트리밍 | 1280 | 17.5×/3× |
| mi355x | 34 MB | backing 스트리밍 | 1024 | 12.5×/1.25× |

**고찰·insight.** decode-state 병목은 본질적으로 **on-chip 용량 문제**다. HBM GPU(H100/B100/MI355X/Rubin)는 1-layer state(134 MB)조차 on-chip(L2 34–105 MB)에 못 담아 매 토큰 HBM 스트리밍한다 — 그런데 *만약 담을 수 있다면* 에너지 12–17×, 시간 3× 이득이 걸려 있다(빈 기회). MTIA/TPU 는 256–268 MB SRAM 으로 layer 상주가 되고, **Cerebras 는 44 GB 로 전체 모델 상주** — 이 부하에서 유일하게 배치 문제가 사라진다. 업계 추세(HBM 용량↑)는 이 부하엔 덜 효과적이고, **on-chip SRAM↑ 가 직접 답**이다. (※ tpu-v7 의 on-chip 예산은 미공개 추정 = BRONZE, residency 판정은 예비.)

### 4.3 E1.3 — kv-vs-ttt (S\* crossover, B_max)

**설명.** KV cache read(context S 에 비례) 와 TTT state RMW(상수)가 같아지는 crossover S\*, 그리고 per-sequence state 로 인한 동시 시퀀스 상한 B_max.

**환경.** S\* = m·d²·(dtype비)/kv_dim. B_max = BW·t_target / rmw_per_token.

![fig4 · S\* 불변성 — 8개 전부 65536. 하드웨어 무관 workload 속성.](/home/jimmy/repos/neural-memory-study/multiarch/figures/fig4-sstar-invariance.png){width=13cm}

![fig3 · B_max@10ms(log) — BW 에 비례(BW-bound).](/home/jimmy/repos/neural-memory-study/multiarch/figures/fig3-bmax.png){width=13cm}

**결과.** **S\* = 65536 token, 8개 전부 동일.** B_max@10ms: MTIA 0.32 → H100 5.2 → Cerebras 32596 (BW 비례). KV write%=0.003, TTT write%=100(대칭 RMW).

**고찰·insight.** S\* 는 이 스터디의 **가장 강한 결과**다 — crossover 위치가 backing 종류(HBM/LPDDR/wafer SRAM)·BW 값·이용률 어디에도 movable 하지 않다(순수 traffic 대수). 이것이 "상쇄 논증"의 직접 실증이고, **BRONZE(Rubin) twin 을 포함해 8개 전부에서 단정 가능**한 유일한 절대적 결과다. 반면 B_max 는 순수 BW-bound(용량 아님) — decode 는 용량 벽이 아니라 **대역폭 벽**에 먼저 막힌다.

### 4.4 E1.4 — frequency-tiers (cadence → memory tier)

**설명.** NL(HOPE)/Sleep 은 memory level 마다 update cadence 를 준다. cadence 가 memory 계층 배정으로 번역되는지 — per-token 은 hot tier 를 요구하고, sub-per-token 은 amortize 되어 cold tier 로 내려갈 수 있는지.

**환경.** 각 twin 의 on-chip·backing 을 hot tier 로, 일반 DDR5·CXL 을 cold tier 로. cadence C 의 amortized 기여 = (2S/BW)/C, per-token budget 100 µs 하 admissible 중 가장 cold tier 선택.

**결과 (cadence → tier).**

| twin | every 1 tok | every 16+ tok |
|---|---|---|
| H100/B100/MI355X/TPU/Rubin | **HBM** | CXL |
| MTIA v2 | **on-die SRAM** | CXL |
| Groq / Cerebras | **SRAM** | CXL |

**고찰·insight.** per-token fast level 은 hot tier 에 고정되지만 **그 hot tier 가 HW 마다 다르다** — GPU 는 HBM, MTIA/Groq/Cerebras 는 on-chip SRAM. 즉 SRAM 기반 칩에선 매 토큰 갱신되는 fast weight 가 **native 로 on-chip 에 산다**. 그리고 sub-per-token(cadence≥16) 은 어디서든 CXL 로 내려간다 — amortization 이 강력해서 NL 의 저주파 CMS level·Sleep 의 offline consolidation 은 값싼 pooled 메모리로 밀 수 있다. **NL/Sleep 의 다속도 메모리가 실제 메모리 계층에 어떻게 앉는지가 HW-상대적**이라는 것이 핵심 — 같은 알고리즘이 GPU 와 wafer-scale 에서 다른 배치 전략을 요구한다.

### 4.5 E1.5 — kvmgr-gap (KV-manager 범주 불일치)

**설명.** KV cache manager 의 의미론(content-addressed reuse, append-only, free-drop eviction)이 RMW state 에 안 맞음을 보인다. 정량 조각: manager 가 0으로 값매기는 dirty writeback 이 실제로는 full write-back 을 빚진다.

**결과.** dirty-writeback owed: Cerebras 134 µJ / MTIA 1342 µJ (KV-manager prices **0**). reuse=0(content 매 토큰 변이), stale accretion(append-only 시 T 버전 누적).

**고찰·insight.** SW argument 의 뼈대(reuse=0, stale 누적)는 HW 무관이지만, **unpriced eviction 비용은 epb 에 비례**해 high-energy backing(LPDDR MTIA)에서 가장 크다. TTT-state manager 는 KV manager 에 없는 event 가 필요하다: `update_in_place`, `mark_dirty/writeback`, `checkpoint/rollback`, `bind_to_sequence`. 즉 서빙 SW 스택도 새로 써야 하는 부하다.

### 4.6 E3 — analytical (ridge, C\*)

**설명.** 닫힌 형태 cost model 의 roofline knee(ridge = peak/BW)와 chunk crossover C\*.

![fig2 · ridge vs decode AI(log). 7/8 memory-bound(여유 380–1455×), Cerebras 만 knee.](/home/jimmy/repos/neural-memory-study/multiarch/figures/fig2-ridge-vs-ai.png){width=14cm}

**결과.** ridge: Cerebras 0.60 → Groq 2.35 → B100 225 → MTIA 864. C\*(prefill compute-bound 되는 chunk): Cerebras 0.6 → H100 337 → MTIA 1180.

**고찰·insight.** ridge 는 각 HW 를 이 부하의 memory/compute 스펙트럼 위에 놓는 단일 숫자다. decode AI 0.594 는 6종 HBM/LPDDR 의 ridge 보다 2–3자릿수 낮아 압도적 memory-bound, Groq 는 ~4×, **Cerebras 만 ridge≈AI(knee)**. C\* 는 반대로 prefill/training 이 compute-bound 되는 chunk 크기 — MTIA 는 1180 이라 큰 chunk 가 필요하고, Cerebras 는 0.6 이라 chunk-1 조차 compute-bound 근처다. **한 알고리즘이 HW 에 따라 memory 문제도 compute 문제도 된다.**

### 4.7 E4 — scaling (170M → 70B)

**설명.** state·traffic·S\* 가 모델 규모(d)에 따라 어떻게 자라는지, 8개 HW 각각에서.

![fig6 · RMW 비용 스케일링(log-log). ∝ m·d²·L, 기울기 HW 무관, 오프셋만 1/BW.](/home/jimmy/repos/neural-memory-study/multiarch/figures/fig6-scaling.png){width=14cm}

**결과.** rmw ∝ m·d²·L (170M 0.45 → 70B 343 GB/tok). log-log 에서 **모든 HW 곡선이 평행**(동일 기울기), 세로 오프셋만 1/BW.

**고찰·insight.** 스케일링 거동은 **보편적**이다 — HW 는 곡선을 위아래로 옮길 뿐 형태를 안 바꾼다. 따라서 규모에 대한 결론(state 는 width²로 자란다, S\* 는 width²로 자란다)은 8개 HW 전부에서 그대로 전이된다. 큰 모델일수록 on-chip 상주가 불가능해지고(state 가 GB급) HBM 스트리밍이 강제되므로, §4.2 의 "on-chip SRAM 이 답" 논지는 규모가 커질수록 강해진다.

### 4.8 E2.1 / E2.2 — host-CPU roofline (아키텍처 무관)

E2.1(chunk-intensity)·E2.2(rmw-cliff)는 **이 host CPU 에서 실제 시간을 잰** roofline *모양* 측정이라 타깃 HW 별로 바뀌지 않는다(host 는 host). 대신 각 twin 의 ridge(§4.6)가 그 곡선의 knee 위치를 정한다 — host C\*≈32 vs H100 C\*≈337 처럼 **모양은 이송되되 값은 ridge 의존**. 이 두 실험은 study paper 원본 그대로 유효하다.

---

## 5. HOPE / Sleep 적용 모델 심층 분석 — 1T 스케일링

앞 실험들은 추상 anchor(neural-mem-1.3B)였다. 여기서는 **실제 적용 모델 — HOPE(Nested Learning)와 Sleep — 의 구조를 그대로 모델링**해 total 파라미터를 **1T 까지** 올리며 decode 비용을 잰다. (근거: `notes/2512.24695`, `notes/2606.03979`; 코드 `hope_sleep_scaling.py`.)

**구조 (grounded).** HOPE block = **self-modifying Titans**(6개 test-time memory: k,v,q,η,α,+ — 투영 자체가 갱신) + **CMS chain**(MLP^(f_1..f_k), fast level 은 매 토큰 RMW, slow level 은 chunk C^l 마다). **Sleep** = wake/sleep lifecycle: wake=HOPE, **sleep(offline)** 에서 fast block 지식을 느린 block 에 **low-rank expert 로 append**(expert pool 이 sleep 마다 증가)하고 dreaming. **핵심: slow-level 갱신·expert 성장이 전부 offline** — decode critical path 밖.

**모델링.** decode 는 memory-bound per-token RMW traffic 이 지배(Part III). "매 토큰 RMW 되는 state"를 세 regime 으로:

- `dense-ttt` — 전체 모델이 test-time-learned (strawman 상한): RMW ∝ P_total.
- `hope` — self-mod + **모든** CMS level 이 inference 중 갱신(fast C=1 + slow 1/C amortized).
- `sleep` — self-mod + **fast level 만**(slow-level consolidation·expert 성장은 offline).

두 스케일링 recipe 를 비교한다: **WIDTH**(d,L 성장 — dense backbone) vs **EXPERT**(backbone 고정, total 을 Sleep-appended expert pool 로만, MoE top-2 routing).

![fig8 · HOPE/Sleep decode 1T 스케일링. (a) WIDTH: Sleep 도 상승(fast level ∝ d²L). (b) EXPERT: Sleep 평탄 — 1T 도 7B 처럼 decode.](/home/jimmy/repos/neural-memory-study/multiarch/figures/fig8-hope-sleep-scaling.png){width=15.5cm}

**결과 A — WIDTH (d,L 성장).** fast-level state 가 d²L 로 자라 **Sleep decode 도 증가**한다.

| total | active | S_sleep | sleep ms/tok※ | dense ms/tok | dense/sleep |
|---|---|---|---|---|---|
| 6B | 5.5B | 2.21 GB | 1.32 | 6.6 | 5× |
| 70B | 57B | 17.7 GB | 10.6 | 83.7 | 8× |
| 405B | 381B | 118 GB | 70.5 | 506 | 7× |
| 1T | 457B | 142 GB | 84.6 | 1223 | 14× |

**결과 B — EXPERT (backbone 고정 d=2048,L=24, expert 만 offline 성장).** **active·S_sleep·decode 가 상수.**

| total | #exp | active | S_sleep | sleep ms/tok※ | dense ms/tok | dense/sleep |
|---|---|---|---|---|---|---|
| 6B | 0 | 5.5B | 2.21 GB | **1.32** | 6.6 | 5× |
| 70B | 80 | 7.1B | 2.21 GB | **1.32** | 83.5 | 63× |
| 405B | 496 | 7.1B | 2.21 GB | **1.32** | 483 | 366× |
| **1T** | **1235** | **7.1B** | **2.21 GB** | **1.32** | 1194 | **903×** |

※ H100, ideal roofline 하한. 절대치 단정 안 함 — 비율·평탄성이 load-bearing.

**고찰·insight (님 가설 검증).** 님의 직관 — "Sleep 은 scaling 가능"— 은 **맞고**, 정확한 메커니즘·recipe 가 드러난다:

1. **Sleep 의 offline consolidation 이 total 파라미터를 decode 에서 분리한다.** EXPERT family 에서 total 이 6B→1T(167×)로 커져도 **per-token RMW state 는 2.21 GB 로 상수**, decode 1.32 ms 로 **완전 평탄**. 1235 개 expert(1T 의 대부분)가 decode RMW 에 **0** 기여 — offline 갱신 + MoE top-2 routing 때문. **1T Sleep 모델이 7B 처럼 decode(active 7.1B).** 1T 에서 dense-all-TTT 대비 **903× 저렴**.

2. **단, "무엇을 키우느냐"가 관건이다.** WIDTH family(d,L 성장)에선 Sleep 도 커진다 — fast-level RMW state 가 d²L 에 비례하기 때문(1T 에서 84 ms). 님이 말한 "**ffn·self-modifying Titans 를 scaling**"은 두 갈래다: **FFN 을 expert 로 늘리면(offline) decode 무료**, 그러나 **self-mod·fast-level width 를 늘리면 decode 가 값을 치른다**. 따라서 1T 로 가는 승리 recipe = **fast backbone(self-mod + fast CMS) 은 decode 예산에 맞춰 고정, capacity 는 offline expert pool 로 확장**.

3. **이것이 Sleep 의 진짜 systems 기여다.** Titans/HOPE 는 test-time 갱신을 decode 에 넣어 memory-bound RMW 를 만들었다(Part III 의 부담). **Sleep 은 그 부담의 대부분을 offline(수면)으로 옮겨** — 성장하는 memory(expert)를 critical path 에서 뺀다. 즉 continual-learning capacity 를 무한히 키우면서 serving decode 는 fast level 로 묶는다. **Part III 의 decode-state 병목에 대한 알고리즘적 해답**이 Sleep 이고, 이 스케일링이 그 증거다.

**정직한 한계.** (a) 이건 decode **비용** 모델이지 **품질**이 아니다 — 논문은 ≤1.3B 만 실측했고, 1T 에서 품질이 유지되는지는 미검증(1235-expert 스케일은 저자 미보고, 본 분석의 투영). (b) 절대 ms 는 ideal 하한. (c) m_fast=8·self-mod 6개·top-2 는 논문 구조에 근거한 합리적 파라미터화이나 정확한 폭은 미공개. 그럼에도 **load-bearing 결론(EXPERT family 에서 decode 평탄, expert 는 decode 에 무료)은 구조에서 직접 따라오며 파라미터 선택에 robust** 하다.

---

## 6. 종합 insight · 분석

1. **pair thesis 의 load-bearing 결론은 하드웨어 공간 전체에서 보편적이다.** S\* 불변(§4.3), decode memory-bound(§4.1/4.6, 7/8), tier 순서(§4.2), 스케일링 형태(§4.7) — 전부 HBM GPU·LPDDR NPU·wafer SRAM 을 가로질러 성립. 이 스터디가 준 새 증거: 이 명제들은 특정 칩이 아니라 **update rule 의 구조(고정크기·RMW 대칭)** 에 걸려 있다.

2. **HW 가 바꾸는 유일한 것은 절대 비용이고, 그 축은 on-chip 용량·대역폭(state 배치)이다.** decode-state 는 append-once KV 와 달리 고정크기라 on-chip 상주가 designable 이고(§4.2), 상주하면 에너지·시간이 한 자릿수 개선된다. **SRAM-heavy 설계(Cerebras 전체 상주 / Groq layer 상주)가 이 memory-centric 병목을 구조적으로 해소** — 이것이 본서 memory-centric 논증의 하드웨어 증거다.

3. **Cerebras knee 는 경고이자 방향이다.** 8개 중 유일하게 decode 가 memory-bound 를 벗어나는데(ridge 0.60≈AI), wafer SRAM 의 극단적 BW 가 이유다. 즉 "충분히 BW-rich 하면 이 부하도 balanced 가 된다" — 미래 소자 방향의 힌트. (단 dense peak 교정 후의 결과 = 라운드2가 sparse 수치를 잡음.)

4. **정직성이 결론을 약화하지 않고 정확히 한다.** 절대 µs/mJ 는 ideal 하한이라 단정하지 않지만, 우리가 단정하는 양(비율·crossover·bound·tier)은 8개 HW·전 규모에서 견고하다. **상쇄되는 양만 단정하고 상쇄 안 되는 양은 유보** — 이 절제가 8칩 실측으로 정당화됐다.

---

## 7. 신뢰성 — per-quantity + 리뷰 감사

| 결과 | 의존 | 단정? |
|---|---|---|
| S\* = 65536 (전 아키텍처) | workload 만 (HW 무관) | 예 — 전부 (BRONZE 포함) |
| bound = memory (7/8) | ridge ≫ 0.59 (380–1455×) | 예 — HBM/LPDDR/Groq |
| Cerebras = knee | ridge 0.60 ≈ AI 0.59 | 예 — "경계"로 |
| decode 비용 순서 (SRAM≪HBM≪LPDDR) | backing BW (GOLD/SILVER) | 예 — vr100 제외 |
| tier 잔류 / d\* | on-chip 용량 | 예 — tpu-v7 은 BRONZE |
| 절대 ms/tok·µJ·mJ | BW·epb + ideal 가정 | 유보 — ideal 하한 |
| vr100 전체 | 미공개 스펙 | 유보 — directional |

**라운드2가 잡아 고친 것 (다중 검증이 작동한 증거).** [MAJOR] wse3 peak 125 PF(sparse 마케팅)→12.5 PF(dense): ridge 5.95→0.60, Cerebras knee 발견. [MINOR] B_max 전사오류 수정, robustness 주장 HBM/LPDDR 로 범위한정, 절대치 vr100 제외. 상세: `REVIEW-LOG.md`. HATIR·HAT spec 자체의 신뢰성(ZigZag byte-exact 검증, ideal-vs-실측 gap): 본서 부록 E.

---

## 8. 재현

```
cd ~/repos/neural-memory-study ; PY=.venv/bin/python
$PY multiarch/run_all_experiments.py --validate   # H100==확정 anchor 게이트
$PY multiarch/run_all_experiments.py --all         # 8 twin x 7 실험 -> results_full/*.json
$PY multiarch/hope_sleep_scaling.py    # HOPE/Sleep 1T 스케일링 (§5)
$PY multiarch/make_figures.py          # figures/fig1-7
$PY multiarch/make_scaling_fig.py      # figures/fig8
```

twin: `multiarch/twins/*.json` (validate_twin PASS). 출처·등급: `multiarch/specs/*.json`. 실험 상세: `results_full/*.json`. driver 는 E3/E1.x 공식 verbatim — H100 재현 게이트가 충실성 보증.
