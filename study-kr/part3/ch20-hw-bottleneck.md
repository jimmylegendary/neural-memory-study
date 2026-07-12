# ch20. 하드웨어 병목 분석: decode roofline과 새 serving primitive

> **이 장의 목표** — 독자가 이 장을 마치면 (1) TTT 계열 decode step을 아키텍처 class별 roofline 위에 올려, 그것이 왜 KV cache와 **다른 종류의 memory-bound**인지 (state read+**write** vs append-read) 수치로 진술할 수 있고, (2) decode 안으로 들어온 **backward pass**를 하나의 새 serving primitive로 이름 붙이고, 그것이 오늘의 forward-only 서빙 kernel에 무엇을 요구하는지 설명할 수 있으며, (3) NS-5·deep-memory의 FLOP density, chunk kernel의 $C^*$, 그리고 MFU가 실제로 어디로 새는지를 pair thesis의 두 절반 — decode(memory-centric) vs prefill/training(accelerator) — 으로 나누어 배치할 수 있어야 한다.
> **왜 필요한가** — 18장은 완성형을 pair thesis로 내리고 8개 실측 claim을 장에 매핑했다. 이 장은 그중 decode 절반의 뼈대(claim 1·8)와 state placement(claim 4), 그리고 training 절반의 진입점(claim 7)을 roofline 하나 위에서 만나게 한다. 여섯 편의 원 논문은 decode wall-clock을 **하나도** 공개하지 않았고(§6.4 라인 공통 caveat), fused deep-memory kernel은 존재하지 않는다(dossier §3.2). 이 장은 그 빈 자리를 roofline·GEMM shape·memory 계층이라는 독자의 모국어로 처음 측정한다.

## 20.1 Bridge-in: crossover에서 roofline으로

18장은 KV cache와 TTT state의 트래픽이 한 문맥 길이에서 교차한다는 것을 보였다(그림 18-1). 그 교차는 "어느 memory 시스템이 더 싼가"를 문맥 길이의 함수로 답했지만, "왜 decode가 이토록 느린가, 그리고 그 느림을 하드웨어의 어느 축이 결정하는가"는 아직 답하지 않았다. 그 답은 roofline 위에 있다.

decode step을 하나의 객체로 본다 — (traffic, arithmetic intensity, bound class)를 갖는 객체다(→ 10장의 roofline 도구). 이 장의 첫 주장은 단정적이다: **이 라인의 decode step은 아키텍처 class와 무관하게 결정적으로 memory-bound이며, 그 bound를 만드는 것은 fast-weight state 전체의 per-token read-modify-write(RMW)다.** KV cache가 "context를 읽는" memory 부하라면, TTT state는 "state를 고쳐 쓰는" memory 부하다. 두 부하는 roofline의 같은 축(대역폭)에 걸리지만, 걸리는 방식이 질적으로 다르다. 이 절부터 그 차이를 class별로 편다.

먼저 표기를 환기한다(→ §0 요약 카드). fast-weight state는 $W_t$, momentum buffer는 $S_t$, state multiplier(state가 $d^2$의 몇 배인가)는 $m$, hidden 차원 $d$, layer 수 $L_{\mathrm{layer}}$(sequence 길이 $L$이 **아님**), chunk 크기 $C$, inner learning rate $\eta_t$, Newton–Schulz $\kappa$회 반복은 $\mathrm{NS}_\kappa$다. 이 장의 모든 절대 수치는 §18.6 정직성 계약 아래에서만 읽는다: **비율·crossover·bound class만 load-bearing이고, 절대 µs/token·mJ/token은 roofline 하한(A100 runbook으로 이월)이며, novel twin(scratchpad·PIM)은 directional DSE다.**

## 20.2 아키텍처 class별 decode roofline

decode의 비용은 그 step이 옮기는 byte로 결정된다. TTT 계열의 decode는 token마다 fast-weight state $W_t$(그리고 momentum을 쓰면 $S_t$까지)를 **읽고, 갱신하고, 되쓴다**. 이 RMW 트래픽은 layer마다 state 크기 $m\,d^2$의 두 배(read $m\,d^2$ + write $m\,d^2$)이며, $L_{\mathrm{layer}}$개 layer에 걸쳐 누적된다(원소당 $s$ byte). anchor 설정 `neural-mem-1.3B (d=2048, m=16, L=24, GQA-8, bf16)`에서 이 값은 **6.44 GB/token**이고, arithmetic intensity는 **0.59 FLOP/byte**로 H100 twin ridge 295 FLOP/byte보다 두 자릿수 아래다(실험 E1.1/E3, 세 방법 1% 이내 합치). 그 결과 state 트래픽이 decode step의 GEMV 연산을 **약 394× 압도한다** — 즉 **step 비용이 곧 state 트래픽이다**. bound class는 memory로 결정적이다.

state multiplier $m$은 아키텍처 class가 정한다. §18.6의 canonical shape(M-ASSUMED)를 따르면: matrix/linear memory는 $m\approx 2$($d\times d$ 상태 하나 수준), 표준 deep memory(2-layer residual MLP, → 12장)는 $m\approx 8$($8d^2$), 여기에 momentum을 더한 Titans-LMM은 $m\approx 16$($16d^2$, anchor), Hope의 6-memory 블록(→ 16장)은 더 크다. 표 20-1은 이 $m$ 사다리를 anchor 폭($d=2048$, $L=24$, bf16)에서 RMW/token으로 옮긴 것이다.

표 20-1 — 아키텍처 class별 per-token decode RMW 트래픽(anchor 폭 $d=2048$, $L_{\mathrm{layer}}=24$, bf16; state=$m\,d^2$ 원소, RMW=$2\,m\,d^2\,L_{\mathrm{layer}}\,s$ byte — $s$는 dtype byte, bf16이면 $s{=}2$). 절대 GB는 M-ASSUMED canonical shape에서 유도한 하한이며, load-bearing한 것은 class 간 **순서**와 전 class가 memory-bound라는 **bound 판정**이다.

| 아키텍처 class | state multiplier $m$ | state/layer | RMW/token | bound |
|---|---|---|---|---|
| matrix / linear memory | ≈ 2 | ≈ 17 MB | ≈ 0.8 GB | memory |
| deep memory (2-layer MLP) | ≈ 8 | ≈ 67 MB | ≈ 3.2 GB | memory |
| + momentum (Titans-LMM, anchor) | ≈ 16 | 134 MB | 6.44 GB | memory |
| Hope (6-memory 블록) | 최상단 | 최상단 | 최상단 | memory |

핵심은 표의 마지막 열이 **전부 같다**는 것이다. deep-memory MLP의 forward/backward는 연산(GEMV)과 트래픽(state RMW)이 함께 $m\,d^2$로 자라 arithmetic intensity가 낮은 채 머문다. 단 이 "함께 $m\,d^2$" 기제는 GEMV 계열에 한정된다 — Atlas의 $\mathrm{NS}_5$처럼 $d\times d$ 행렬을 반복 orthogonalize하는 연산은 그 kernel만 떼어 보면 연산 $O(\kappa d^3)$·트래픽 $O(d^2)$로 AI가 $O(d)$까지 올라 **고립 kernel로는 compute-bound**다(§20.4). 그럼에도 $C{=}1$ decode step 전체는 memory-bound로 남는데, 그 NS 연산 시간이 whole-state RMW 트래픽 시간에 가리기 때문이다(§20.4 마지막 문단). 요컨대 class를 올려도 decode step의 bound 판정은 memory로 유지되지만 그 이유는 class마다 다르고 — GEMV 계열은 AI가 낮아서, NS 계열은 트래픽이 연산을 가려서 — bound class는 그대로인 채 총 RMW 트래픽과 latency만 함께 커진다. 이것이 decode 절반이 "새 memory 기회"인 첫 번째 구조적 이유다: capacity를 늘리려 memory를 깊게·무겁게 만들수록 decode는 대역폭 벽에 더 세게 부딪힌다.

이 벽은 폭과 함께 자란다. RMW 트래픽은 문맥 길이 $S$에 **무관**하고 모델 폭에 따라 $m\,d^2\,L_{\mathrm{layer}}$로 자란다: Titans-170M의 **0.45 GB/token**에서 anchor의 **6.44 GB/token**을 거쳐 가상의 70B에서 **343 GB/token**까지다(실험 E3). KV cache가 문맥에 비례해 read 트래픽이 자라는 것과 정반대로, TTT는 폭에 비례해 RMW 트래픽이 자란다. 두 스케일링의 교차가 18장의 crossover 문맥 길이($S^\star$)이며, 그 상세는 19장이 scaling 축으로 편다.

> **[해설]** inference 관점에서 이 표를 한 문장으로 옮기면 이렇다. **decode step 하나가 "1.3B 모델의 weights 절반쯤을 매 token마다 다시 읽고 다시 쓰는" 트래픽을 낸다.** KV cache append가 token마다 몇 KB를 덧붙이는 것과 비교하면 — KV write는 read의 0.003%인 append-once인데(실험 E1.3), TTT는 write가 read와 같은 대칭 RMW(100%)다 — 이것은 같은 "memory-bound"라는 단어로 부르기 어려운 다른 부하다. 앞으로 이 장이 "decode가 memory-bound"라고 말할 때, 그것은 KV cache의 memory-bound(read-many, shareable)가 아니라 **whole-state RMW의 memory-bound(write-heavy, unshared)** 를 뜻한다.

절대값에 대한 정직성. anchor의 roofline 하한은 **1.92 ms/token**, **46.5 mJ/token**이다(실험 E1.1). 이 두 수는 이 장의 주장이 딛는 근거가 **아니다**. 6편의 원 논문이 H100 decode wall-clock을 하나도 공개하지 않았으므로(§6.4), 이 절대값은 외부 검증 불가한 roofline 하한이며 사내 A100 runbook(Part III-a)으로 이월되는 검증 대상이다. 본문이 딛는 것은 그 하한이 드러내는 **구조** — memory-bound, RMW-지배, class-무관 — 뿐이다.

## 20.3 decode에 들어온 backward pass: 새 serving primitive

RMW의 byte 수는 무엇이 옮겨지는지를 말하지만, 무엇이 **계산되는지**는 말하지 않는다. 그리고 계산되는 것 안에 이 라인이 서빙 시스템에 던진 진짜 새로움이 있다.

독자는 decode의 forward pass를 안다: $y_t = \mathcal{M}(q_t; W_t)$, memory를 읽는 한 번의 전방 통과다. 그런데 TTT 계열의 decode는 그 다음에 **state를 갱신**해야 하고, 갱신의 핵심은 inner loss의 gradient다(→ 8장·12장):

$$
g_t^{\mathrm{in}} \;=\; \nabla_W\,\ell\big(W_{t-1};\,k_t,v_t\big),
\qquad
S_t = \beta_t S_{t-1} - \eta_t\, g_t^{\mathrm{in}},
\qquad
W_t = \alpha_t W_{t-1} + S_t
\tag{20-1}
$$

식 (20-1)의 첫 항 $g_t^{\mathrm{in}}$을 계산하려면 memory MLP를 통과하는 **backward pass**가 필요하다. deep memory $\mathcal{M}(z;W)=z+W_1\,\sigma(W_2 z)$의 gradient는 forward activation을 거슬러 오차를 전파해 얻는다 — 이것은 훈련의 backward pass와 **같은 연산**이다. 요컨대:

> **[평가]** 이 라인이 서빙 엔진에 요구하는 진짜 새 primitive는 whole-state RMW의 byte 수가 **아니라**, **backward pass가 decode의 critical path 위로 올라왔다는 사실 자체**다. 오늘의 inference kernel(FlashAttention decode, paged attention)은 정의상 forward-only다 — 추론 중 weights는 변하지 않는다는 불변식(→ 1장 Rosetta) 위에 세워졌기 때문이다. 그 불변식이 이 라인에서 폐기되면서, "훈련에만 존재하던" forward+backward의 합성이 token마다 한 번씩 서빙 loop 안에서 돌아간다. 이것이 pair thesis의 decode 절반을 "새 기회"로 만드는 것의 핵심이며, byte 회계는 그 결과일 뿐이다.

이 primitive가 kernel에 요구하는 것은 세 가지다. 첫째, **forward+backward fused decode kernel**이 필요하다. read($y_t$), backward($g_t^{\mathrm{in}}$), momentum·retention·write(식 (20-1)의 나머지)를 한 번의 state streaming 안에서 융합해야 memory-bound step에서 state를 두 번 이상 왕복하지 않는다. 그런 fused deep-memory kernel은 현재 존재하지 않는다(dossier §3.2; A1 각도의 근거). 둘째, forward activation을 backward에서 재사용하려면 그 activation의 **residency**가 문제가 된다(§20.6). 셋째, Atlas의 Muon(→ 14장)을 쓰면 backward 뒤에 momentum 행렬을 orthogonalize하는 $\mathrm{NS}_\kappa$ 반복이 $\kappa$회 더 붙는다(backward inner pass가 $\kappa$번 반복되는 것이 아니라, gradient 이후의 별도 행렬 반복이다; §20.4).

> **[해설]** GEMM shape로 읽으면 (20-1)의 backward는 낯설지 않다. linear memory의 극한에서 $g_t^{\mathrm{in}} = (\text{오차})\,k_t^\top$는 rank-1 outer-product write이고(→ 6장의 delta rule), deep memory에서는 그것이 2-layer MLP를 거슬러 오르는 두 번의 GEMV가 된다. 문제는 shape가 낯설다는 게 아니라 **위치**가 낯설다는 것이다 — 이 GEMV들이 batch 축이 없는(per-request, unshared) 상태에서 token마다 순차로 돈다. shared-weight batching이 깨지는 지점(→ 23장)이 바로 여기서 시작된다.

## 20.4 NS-5와 deep-memory의 FLOP density: PIM 경계선을 긋는다

decode가 memory-bound라는 것은 FLOP이 적다는 뜻이 아니라 FLOP이 트래픽에 비해 적다는 뜻이다. 그 FLOP이 **어떤 모양**인지가 memory-centric 논증의 정직한 경계를 긋는다.

식 (20-1) 한 step의 FLOP은 세 덩어리로 나뉜다. (a) memory MLP를 통과하는 forward+backward — deep memory에서 layer당 몇 개의 $d\times d$ GEMV, class를 올릴수록 커지는 대부분의 FLOP. (b) Atlas의 경우 momentum $S_t$를 semi-orthogonalize하는 $\mathrm{NS}_\kappa$ — $\kappa=5$가 표준이고 각 반복이 행렬 곱 몇 번이므로, 이것도 GEMM-shaped FLOP을 $\kappa$배로 더한다(→ 14장). (c) elementwise epilogue — momentum decay($\beta_t S_{t-1}$), retention($\alpha_t W_{t-1}$), write의 AXPY, 그리고 Miras 계열의 renorm(→ 13장). 이 (c)만이 memory-side, elementwise, 1–3 MAC/elem의 PIM-shaped 연산이다.

이 분해가 정직성 계약의 한 경계를 만든다. dossier의 판정(§5, "Forced" 4번)을 이 장의 수치로 재확인하면:

> **[평가]** (a)와 (b) — memory MLP의 forward/backward와 NS-5 — 은 **GEMV/GEMM-shaped**이고, 이 라인이 여섯 편에 걸쳐 알고리즘을 dense matmul로 다시 빚어 온 이유가 바로 이 FLOP을 tensor-core에 태우기 위해서다(chunk anchoring, banded mask, batched NS-5 — → 9장·14장). 따라서 "이 라인의 연산이 일반 PIM을 요구한다"는 주장은 이 FLOP 분포에 의해 **반증된다** — 대부분의 FLOP은 PIM이 못 태우는 GEMM이다. PIM이 정당한 지점은 오직 (c) elementwise epilogue뿐이고, 그것은 전체 FLOP의 소수다. E1.2는 이 epilogue에 대해 PIM이 TTT의 ~3 MAC/elem에서 **1.6× energy 이득**(단 latency는 2.1× 손해), rank-1에 가까운 1 MAC/elem에서는 energy·latency **둘 다** 이득임을 보인다 — 그러나 `simulation_ready=False`인 directional DSE로만 인용한다(§18.6).

FLOP density의 실천적 결론은 decode 절반에서 일관된다. class를 올려 FLOP을 늘려도(deep memory, NS-5) decode **step 전체**의 유효 arithmetic intensity는 ridge 아래에 머문다(§20.2) — NS-5 kernel 자체는 고립하면 compute-bound($O(d)$ AI)일 수 있으나, $C=1$ 영역에서는 그 연산조차 whole-state RMW 트래픽에 가려 step은 memory-bound를 벗어나지 못한다. FLOP density가 문제가 되는 것은 decode가 아니라 chunk를 키우는 순간, 즉 prefill/training 영역에서다. 그 전환을 다음 절이 roofline 위에서 본다.

## 20.5 chunk kernel: 같은 알고리즘이 compute-bound로 넘어가는 곳

pair thesis의 두 절반은 서로 다른 두 알고리즘이 아니라 **같은 알고리즘의 두 chunk 영역**이다. 이 절은 그 전환을 roofline 위에서 측정한다.

chunk 크기 $C$는 arithmetic intensity를 결정하는 knob이다(→ 9장). $C=1$은 per-token rank-1 RMW — 곧 §20.2의 decode 영역 — 이고 AI≈1 FLOP/byte로 memory-bound 평원에 앉는다. $C$를 키우면 chunk 안의 여러 token이 같은 chunk-start state $W_{\xi(t,C)}$에서 gradient를 평가하므로(stale-snapshot 근사, 식 (M4), → 9장) state 재사용이 늘고 AI가 roofline을 타고 오른다. 어느 $C^*$에서 memory↔compute 교차가 일어난다.

<!-- FIG: exp-c -->
![그림 20-1 — chunk 크기 $C$에 따른 arithmetic intensity의 host roofline 상승: $C{=}1$(decode 영역)의 memory-bound 평원에서 $C^*$의 compute-bound 영역으로](../../figures/exp-c-chunk-roofline.png)

그림 20-1 — DeltaNet-style chunkwise scan의 measured AI(C) 곡선. $C{=}1$(per-token RMW = TTT decode 영역)은 host roofline의 약 12%에 머무는 memory-bound 평원이고, $C$를 키우면 host ridge(≈34 FLOP/byte)를 넘어 measured $C^*{\approx}32$에서 compute-bound로 오른다. $d{=}2048$ measured throughput은 $C{=}1$의 4.56 GFLOP/s에서 $C{=}512$의 689.5 GFLOP/s까지 오른다. **곡선의 모양과 교차의 존재만 이전되고 $C^*$의 절대값은 이전되지 않는다**: host ridge는 H100 twin ridge(295 FLOP/byte)의 약 1/9이며, H100 closed-form $C^*$는 $d$에 따라 306–430이다($d{=}2048$에서 337.1). 실험 E2.1(CPU micro-bench, CPU-SHAPE) / E3 재구성.

이 그림이 pair thesis를 한 장면으로 압축한다: **같은 knob $C$가 decode를 memory-bound로, prefill/training을 compute-bound로 만든다 — 한 알고리즘, 두 영역.** $C^*$의 위치는 ridge에 의존하므로($C^*$는 ridge에 비례) H100에서는 300 근처, host CPU에서는 32 근처지만, "낮은 $C$는 memory-bound, 높은 $C$는 compute-bound"라는 **곡선의 모양**은 두 하드웨어에서 같다. 이것이 CPU 실측에서 이전되는 유일한 것이다(CPU-SHAPE tag, §18.6).

이제 MFU가 어디로 새는지가 이 곡선 위에서 보인다. Atlas는 품질 최적 $C$(작은 chunk)에서 deep memory의 FLOPs utilization이 **5–10% 미만**임을 보고한다([Atlas], dossier bridge 3). 그 누수의 정체는 그림 20-1의 왼쪽 절벽이다:

> **[평가]** MFU 누수를 roofline 위에서 읽으면 그 큰 몫은 kernel 비효율이 아니라 **작동점의 선택**으로 드러난다(단 Atlas의 kernel-level profiling 분해가 없으므로, 작동점의 구조적 몫과 구현 overhead의 몫을 정량으로 가르지는 못한다 — measured host $C{=}1$이 roofline attainable의 약 12%라는 것은 memory-bound 상한 아래에 구현 손실도 남아 있음을 함의한다). 품질이 최적인 $C$는 작고(→ 15장 TNT의 chunk-1 decode 논점), 작은 $C$는 그림 20-1의 memory-bound 평원 — roofline의 12% — 위에 있다. 즉 MFU는 "새는" 것이 아니라 **품질을 위해 지불되는** 것이다. 둘째 원인은 stale-snapshot 근사가 chunk마다 chunk-start state를 다시 읽어 재계산하는 트래픽이고, 셋째는 per-request·unshared 상태에서 batch 축이 얇아 GEMM이 grouped-GEMM으로 파편화되는 것이다(→ 23장). TNT가 two-stage 훈련으로 chunk-1을 **품질 최적점으로** 옮긴 것(→ 15장)은 이 셋 중 첫째를 정면으로 공략한 것이다 — 작동점을 옮겨 MFU 누수의 원인 자체를 없앤다. 단 chunkwise staleness의 오차 한계는 6편 어디에도 없으므로(§6.4), 이 작동점 이동의 품질 대가는 정량화되지 않은 채 남아 있다.

이 절반 — chunk를 키워 compute-bound로 넘어가는 prefill/training — 이 accelerator 영역이다. 승부는 fused chunk kernel(banded-mask windowed loss, batched NS-5)과 grouped-GEMM에서 나며, TNT가 이미 plain JAX로 FlashAttention을 32K에서 step당 이긴 지점이다(→ 15장). 이 영역에서 memory-centric 논증을 펴는 것은 pair thesis가 금지한다: 여섯 편이 알고리즘을 dense matmul로 빚어 기존 accelerator에 맞췄기 때문이다.

## 20.6 state placement: 상주는 설계 knob이고, 경계에서 대역폭이 꺾인다

decode가 memory-bound이고 그 bound가 whole-state RMW라면, 남은 설계 자유도는 **그 state를 어디에 두는가**다. KV cache와 달리 TTT state 크기는 문맥에 무관하게 $(d,m,L)$로 고정되므로(§20.2), on-chip 상주 여부가 **설계 가능한 knob**이 된다 — 문맥에 따라 자라 상주 계획을 세울 수 없는 KV cache와의 결정적 차이다.

먼저 상주의 물리적 한계. on-die L2(50 MB급)는 anchor에서 **한 sequence의 whole-model state조차** 담지 못한다(어느 스케일에서도, 실험 E1.3/E3). 따라서 whole-model을 on-die에 pin하는 선택지는 없고, 상주는 **per-layer/streamed**여야 한다. 이 제약이 state placement를 layer 단위 DSE로 만든다.

<!-- FIG: exp-b -->
![그림 20-2 — per-layer RMW state의 device별 energy/latency와 residency crossover: state가 fit하는 폭에서는 scratchpad가 HBM을 이기고, 폭이 커지면 spill한다](../../figures/exp-b-state-placement.png)

그림 20-2 — 134 MB/layer RMW를 네 device twin에 올린 결과. state가 268 MB scratchpad에 fit하는 동안 scratchpad는 HBM3 대비 **6.8× energy / 14.9× time** 이득을 낸다. 그러나 residency crossover가 있다: 268 MB 버퍼는 anchor($d{=}2048$, 134 MB/layer)까지 한 layer의 state를 담고 $d{=}4096$(7B, 537 MB/layer)에서 **spill**하며, crossover 폭은 $d^\star{\approx}2896$이다. whole-model이 268 MB에 드는 것은 Titans-170M(226 MB total)뿐이므로, 340M 이상에서는 상주가 per-layer/streamed다. scratchpad·PIM twin은 `simulation_ready=False`인 **directional DSE**이며, 이 이득 배율은 shipping-device 주장이 아니다(NOVEL-SIM-FALSE, §18.6). 실험 E1.2.

그림 20-2의 메시지는 두 겹이다. 겉으로는 "state가 on-chip에 fit하면 scratchpad가 크게 이긴다"(analytic twin이 예측한 fit→spill 이득, directional)이고, 그 아래에는 "fit 여부 자체가 폭의 함수인 crossover"가 있다. 이 fit→spill 불연속은 추상적 경고가 아니라 host silicon에서 **직접 측정되는** 물리적 절벽으로도 확인된다. 그 절벽을 host CPU에서 직접 재면 다음이 나온다.

<!-- FIG: exp-e -->
![그림 20-3 — on-die/off-die 경계에서의 RMW 대역폭 cliff: in-cache RMW가 capacity 경계를 넘으면 유효 대역폭이 불연속으로 꺾인다](../../figures/exp-e-rmw-cliff.png)

그림 20-3 — L3 capacity 경계에서 RMW 유효 대역폭이 꺾이는 cliff. in-cache RMW는 **~265 GB/s**로 돌지만 capacity를 넘겨 DRAM으로 spill하면 **~68 GB/s**로 **약 3.9× 붕괴**한다(1-thread). 또한 saturated DRAM에서 RMW는 read의 **~0.5× element throughput**만 낸다 — RMW가 element당 2× byte를 옮기기 때문이며, 이것이 **append-once KV cache가 피하는 write-back 세(稅)** 를 직접 측정한 값이다. **GB/s 절대값은 H100으로 이전하지 않는다**(host cache cliff는 H100 on/off-die ridge와 100× 어긋난다); 이전되는 것은 cliff의 **존재**와 3.9× 붕괴의 **모양**뿐이다(CPU-SHAPE, §18.6). 실험 E2.2.

그림 20-2와 20-3은 같은 현상의 두 얼굴이다 — 20-2는 device twin에서 fit→spill을, 20-3은 host silicon에서 그 spill의 대역폭 대가를 잰다. 둘을 합치면 state placement의 설계 규칙이 나온다: **RMW state는 상주 경계(capacity)를 넘는 순간 대역폭이 불연속으로 붕괴하므로, memory 계층의 경계는 이 부하에서 부드러운 grade가 아니라 절벽이다.** 이것이 memory-centric 기회의 두 번째 load-bearing 지점(§18.3의 update-frequency↔tier 배치)과 만난다: state를 어느 tier에 두는가가 곧 어느 대역폭 plateau에 앉는가를 결정한다.

그 tier 배치를 update cadence가 정한다(claim 5, 실험 E1.4). 규칙은 단정적이다 — **상주 사본은 read/write 중 빠른 쪽 cadence로 tier가 정해진다.** per-token fast-weight는 read가 매 token이므로 HBM3에 pin되고, per-token으로 **쓰이는** state를 CXL 같은 느린 tier에 두면 매 token 그 대역폭 cliff의 아래쪽에 앉는다. 반대로 genuinely down-sampled된 CMS-mid level(4096 token마다 read+write)이나 sleep-consolidated expert(offline cadence)는 sub-per-token 접근이 트래픽을 critical path 아래로 amortize하므로 CXL로 **합법적으로** 내려갈 수 있다(→ 16장·17장). per-layer fast-weight 블록(67 MB)이 scratchpad에 fit할 때의 7.4× energy 이득(E1.4)은 이 배치의 방향(per-layer streamed면 on-chip 이득)만 load-bearing이고, novel twin이라 `simulation_ready=False`인 directional DSE로만 인용한다.

> **[해설]** inference 엔지니어의 감각으로 옮기면 이렇다. paged KV cache에서 "page를 어느 tier에 둘까"는 접근 빈도(hot/cold)로 정한다 — 그것과 정확히 같은 규칙이 여기서도 성립하되, 축이 "얼마나 자주 읽히나"에서 "얼마나 자주 **쓰이나**"로 바뀐다. write-heavy·unshared라는 성격 때문에, cold page를 evict하듯 dirty state를 버릴 수 없다 — evict는 full writeback을 빚진다(claim 6, → 23장). placement가 잘못되면 그 writeback이 매번 대역폭 cliff의 밑바닥에서 일어난다.

## 20.7 종합: kernel 수준에서 본 pair

이 장의 측정을 pair thesis의 두 절반으로 되접는다. 어느 쪽도 소홀히 하지 않는 것이 이 책의 입장이다(dossier §5의 comprehensive stance).

**decode 절반 = memory-centric.** decode step은 class와 무관하게 whole-state RMW의 memory-bound(§20.2, 394× state-지배)이고, 그 안에는 backward pass라는 새 serving primitive가 있다(§20.3). 이 절반의 하드웨어 기회는 memory-centric이다 — high-RMW-bandwidth state residency, forward+backward fused decode kernel, cadence-tiered placement, write-heavy elementwise epilogue를 위한 near-memory 엔진(§20.4·20.6). 이 논증은 논문들 자신의 cost 회계(whole-state RMW, unshared, 문맥-무관 state)에 근거하며 부과된 것이 아니다.

**training/prefill 절반 = accelerator.** 같은 알고리즘도 chunk $C$를 키우면 compute-bound로 넘어가고(§20.5, 그림 20-1), 그 FLOP은 GEMV/GEMM-shaped이며 NS-5조차 matmul이다(§20.4). 이 절반의 승부는 fused chunk kernel과 grouped-GEMM, 즉 tensor-core 위에서 난다. 이 영역에 memory-centric 소자를 옹호하는 것은 여섯 편의 중심 엔지니어링 패턴(알고리즘을 dense matmul로 빚기)과 모순되므로 이 책은 하지 않는다.

> **[평가]** 흥미로운 하드웨어 제안은 둘 중 하나가 아니라 **쌍**이다. decode를 위한 RMW-bandwidth·backward-capable serving 소자와, prefill/training을 위한 fused chunk·grouped-GEMM accelerator는 같은 배포의 두 부하를 나눠 맡는다. 어느 한쪽만 옹호하는 제안은 이 workload의 절반을 무시하는 것이다. 그리고 이 쌍을 위한 전용 artifact는 아직 어느 것도 확인되지 않는다 — deep-memory decode를 한 번의 state stream으로 융합하는 fused kernel도, 그 backward를 decode 경로에 품는 serving runtime·소자도(기존 GPU가 backward·grouped-GEMM 자체를 실행할 수 있다는 것과는 별개의, 전용 서빙 artifact의 부재다; dossier §3.2). 이 라인이 기다리는 "FlashAttention-moment"이 무엇을 예고하는지는 21장의 hardware-lottery 독법이 잇는다.

이 장의 모든 절대값은 다시 한번 하한이다. 1.92 ms/token, 46.5 mJ/token, scratchpad의 6.8×/14.9×, cliff의 265→68 GB/s — 이 중 load-bearing은 **394× state-지배(bound 판정), $C^*$의 존재와 곡선 모양, 3.9× cliff 비율, class 간 순서**뿐이고, 나머지 절대값은 A100 runbook으로 이월되거나 directional DSE로 남는다.

## 요약

- 이 라인의 decode step은 아키텍처 class(matrix→deep→+momentum→Hope)와 무관하게 **결정적으로 memory-bound**이며, state RMW가 GEMV 연산을 약 394× 압도한다 — step 비용이 곧 whole-state 트래픽이다(anchor 6.44 GB/token, AI 0.59 FLOP/byte). class를 무겁게 할수록 더 memory-bound가 될 뿐이다.
- decode 안으로 들어온 **backward pass**가 이 라인이 서빙 엔진에 던진 진짜 새 primitive다. forward-only인 오늘의 decode kernel과 달리, token마다 memory MLP를 거슬러 gradient를 계산해야 하며, 이를 융합할 fused deep-memory kernel은 존재하지 않는다.
- FLOP density가 memory-centric 논증의 경계를 긋는다: memory MLP forward/backward와 NS-5는 GEMM-shaped(→ accelerator)이고, PIM이 정당한 곳은 elementwise epilogue(decay·renorm·AXPY)의 소수 FLOP뿐이며 그것도 directional(1.6× energy)이다.
- 같은 chunk knob $C$가 decode를 memory-bound($C{=}1$, roofline 12%)로, prefill/training을 compute-bound($C^*$ 위)로 만든다 — 한 알고리즘, 두 영역. MFU가 5–10%로 새는 것은 kernel 결함이 아니라 **품질 최적 작동점(작은 $C$)이 memory-bound 평원 위에 있기 때문**이다.
- TTT state 크기는 문맥에 무관하므로 on-chip 상주가 설계 knob이 되며, state placement의 memory 계층은 capacity 경계에서 대역폭이 **불연속으로 3.9× 꺾이는 절벽**이다(RMW는 read의 0.5× throughput만 내는 write-back 세를 문다). 상주 tier는 read/write 중 빠른 cadence로 정해진다.
- 흥미로운 HW 제안은 **쌍**이다: decode용 RMW-bandwidth·backward-capable serving 소자 + prefill/training용 fused chunk·grouped-GEMM accelerator. 모든 절대값은 roofline 하한(A100 runbook 이월)이거나 directional이고, load-bearing은 비율·crossover·bound·순서뿐이다.

## 자가 점검 체크리스트

- [ ] 아키텍처 class별 decode RMW 트래픽을 $2\,m\,d^2\,L_{\mathrm{layer}}\,s$로 계산하고, 전 class가 왜 memory-bound인지 설명할 수 있다.
- [ ] decode의 backward pass가 왜 "새 serving primitive"인지, forward-only KV decode와 무엇이 다른지 진술할 수 있다.
- [ ] 한 decode step의 FLOP을 (memory MLP f/b, NS-5, elementwise epilogue)로 나누고, PIM이 정당한 지점을 그중 어디로 한정해야 하는지 말할 수 있다.
- [ ] chunk $C$가 왜 roofline의 x축인지, $C^*$의 절대값은 이전 안 되지만 곡선 모양은 이전되는 이유를 설명할 수 있다.
- [ ] MFU가 5–10%로 "새는" 진짜 원인(작동점 선택)과 TNT의 chunk-1 이동이 그것을 어떻게 공략하는지 말할 수 있다.
- [ ] state placement의 fit→spill 절벽과 3.9× 대역폭 cliff가 왜 부드러운 grade가 아닌지, cadence가 tier를 어떻게 정하는지 설명할 수 있다.
- [ ] 어느 수치가 load-bearing(비율·crossover·bound·순서)이고 어느 것이 이월되는 하한·directional인지 판별할 수 있다.
- [ ] 이 장의 병목을 inference 어휘(paged KV placement ↔ RMW-state placement; forward-only decode ↔ backward-in-decode)로 옮길 수 있다.

## 다음 장으로

이 장은 pair thesis의 두 절반을 roofline·kernel·placement의 언어로 측정하고, 어느 것도 아직 존재하지 않는 두 소자 — backward-capable decode 소자와 fused chunk accelerator — 를 pair로 남겨 두었다. 그 소자들이 왜 아직 없는지, 그리고 이 라인이 왜 하필 matmul 안으로 스스로를 설계해 넣었는지는 하드웨어의 역사가 답한다. 21장은 hardware-lottery의 독법으로 그 질문을 잇는다 — attention은 GEMM density로 이겼고, 이 가족은 chunkwise·NS·reset으로 그 승리를 모방하도록 설계되어 있다. 그것이 채택과 kernel 생태계에 무엇을 예고하는지가 다음 장의 주제다.
