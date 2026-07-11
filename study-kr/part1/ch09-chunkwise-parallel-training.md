# ch09. Chunkwise-parallel training: 하나의 일반 scheme, 네 개의 인스턴스

> **이 장의 목표** — 독자가 이 장을 마치면 다음을 할 수 있어야 한다.
> (1) 임의의 fast-weight recurrence를 받아 "이것은 exact하게 chunk 병렬화되는가, anchored 근사가 필요한가"를 transition의 구조(스칼라/대각 linear, 비대각 linear, nonlinear)만 보고 판정할 수 있다.
> (2) GLA·DeltaNet·TTT dual form·Titans의 chunkwise 알고리즘을 하나의 일반 scheme("경계에서 얼리고, 내부는 GEMM, 경계 사이는 handoff 또는 scan")의 네 인스턴스로 유도할 수 있다.
> (3) chunk 크기 $C$가 왜 어떤 family에서는 순수 성능 knob이고 어떤 family에서는 **계산되는 함수 자체를 바꾸는 semantic hyperparameter**인지 손계산으로 보일 수 있다.
> (4) arithmetic intensity를 $C$의 함수로 유도하고, 품질 최적 $C$와 MFU 최적 $C$가 왜 갈라지는지 roofline 위에서 설명할 수 있다.
>
> **왜 필요한가** — 이 장이 없으면 이 라인의 어떤 것도 실물이 되지 않는다. [Titans] (*Titans: Learning to Memorize at Test Time*, arXiv:2501.00663)의 §3.2 "How to Parallelize the Long-term Memory Training"은 그 논문의 실용성 주장 전체를 담당하고, [Atlas] (arXiv:2505.23735)는 §3.4에서 Omega rule을, 부록(Eq. 36–41)에서 Muon 내장 momentum을 같은 기법으로 병렬화한다. [Miras] (arXiv:2504.13173)의 세 신모델(Moneta/Yaad/Memora)도 같은 chunkwise 기법으로 훈련된다 [Miras §5.3]. 그리고 [TNT] (arXiv:2511.07343)는 **이 장이 다루는 기법의 한계 자체가 논문 한 편의 주제다** — chunk 크기의 품질↔throughput 긴장, 비선형 inter-chunk recurrence라는 병렬화 blocker, reset을 통한 context parallelism. [NL] (arXiv:2512.24695)의 CMS는 이 장의 scheme을 level마다 반복 적용한 것이며(식 (M5), → 16장), [Sleep] (arXiv:2606.03979)은 NL을 경유해 간접적으로 이 장에 의존한다. 독자의 FlashAttention/tiling 감각이 가장 큰 지렛대가 되는 장이자, 그 감각이 가장 위험한 오도가 되는 장이다.

## 9.1 문제 설정: training 안에 들어온 decode loop

8장이 남긴 계산 문제를 정면으로 다시 세운다. 이 라인의 layer는 정의상 per-token recurrence다: token $t$의 상태 $W_t$는 $W_{t-1}$ 없이는 계산될 수 없다. inference의 decode라면 이 순차성은 원래 세계의 조건이므로 놀랍지 않다. 문제는 **training**이다. outer loop(→ 2장, 4장)는 sequence 길이 $L$ 전체에 대한 forward를 요구하는데, 이것을 per-token으로 순차 실행하면 training이 통째로 decode loop가 된다 — step당 연산은 GEMV 한두 번과 rank-1 update, 즉 arithmetic intensity가 $O(1)$ 수준인 shape이 $L$번 직렬로 이어진다. 독자의 어휘로 정확히 옮기면: **batch 1짜리 decode를 32K번 돌려서 한 개의 training step을 만드는 것**과 같다. tensor core는 놀고, wall-clock은 무너진다. [TNT §1]은 이 문제를 정량으로 못 박는다 — 작은 chunk의 deep-memory 훈련은 peak FLOPs의 5–10% 미만밖에 쓰지 못한다는 보고(LaCT, Zhang, Bi, et al. 2025)를 인용하면서다.

이 문제의 해법 계보가 **chunkwise-parallel training**이다(축약: chunkwise training). 정의: sequence를 크기 $C$의 chunk로 자르고, **chunk 내부의 계산은 chunk 시작 상태에 대해 병렬(GEMM-rich)로, chunk 사이의 상태 전달은 순차(또는 linear recurrence면 associative scan)로** 재조직하는 훈련 기법의 총칭이다. 기원은 linear-attention 계열의 mixed-chunk 기법(Hua et al. 2022, arXiv:2202.10447)이고, RetNet(arXiv:2307.08621)이 recurrent/parallel/chunkwise의 세 계산 모드를 명시적 정식화로 굳혔으며, GLA(Yang et al. 2024, arXiv:2312.06635)가 hardware-efficient kernel로, DeltaNet 병렬화(Yang et al. 2024, arXiv:2406.06484)와 TTT dual form(Sun et al. 2024, arXiv:2407.04620)이 각각 비대각 transition과 gradient-step recurrence로 확장했다. 이 계열의 kernel 구현은 `flash-linear-attention` 라이브러리(Yang & Zhang, GitHub, 2024)에 집대성되어 있다.

세 모드의 전체 지형은 7장에서 exact한 세계(Mamba-2/GLA)에 대해 이미 그렸다(표 7-3). 이 장의 임무는 그 표를 **일반화**하는 것이다. 일반화의 대가로 표에 새 열이 하나 생긴다 — "exact인가?" — 그리고 이 열이 갈라지는 지점이 여섯 논문 전체의 pivot이다.

<!-- FIG: ch09/fig-02-three-regimes -->

표 9-1 — 세 계산 regime의 일반형 (표 7-3의 일반화)

| 모드 | token당 비용 차수 | 주 연산 | 병렬성 | exact? |
|---|---|---|---|---|
| recurrent ($C=1$) | $O(d_kd_v)$ 또는 $O(P_f)$ | GEMV + rank-1/param RMW | 순차 | 항상 exact (기준 정의) |
| parallel ($C=L$) | $O(L\,d)$ | GEMM (attention-like) | 완전 병렬 | family에 따라 다름 |
| chunkwise ($1<C<L$) | $O(C\,d + d_kd_v/\text{token})$ | GEMM + 경계 handoff/scan | chunk 내 병렬 | **family에 따라 다름 — 이 장의 주제** |

($P_f$는 deep memory 한 개의 parameter 수. "exact"의 기준은 $C=1$의 per-token recurrence가 정의하는 함수다.)

## 9.2 일반 scheme: 경계에서 무엇을 얼리는가

네 인스턴스에 들어가기 전에, 모든 인스턴스가 공유하는 뼈대를 한 번만 정확히 세운다. 이 절이 이 장의 심장이다.

**분해.** sequence를 chunk $n = 0, 1, \ldots$ (token $nC{+}1 \ldots (n{+}1)C$)로 자른다. 임의의 fast-weight update를 다음 두 부분으로 분해한다:

1. **state에 linear한 부분** — retention gate의 곱($\alpha_t W_{t-1}$), momentum의 EMA($\beta_t S_{t-1}$), Hebbian/delta write의 state 의존항. 이들은 $S_t = a_tS_{t-1} + b_t$ 꼴의 **linear recurrence**이고, linear recurrence는 정확하게 병렬화된다: 계수의 누적곱과 입력의 가중합으로 닫힌 형태를 쓰거나(→ 아래), Blelloch(1990)의 associative scan으로 log-깊이에 계산한다. 독자가 이미 아는 prefix-sum kernel이 바로 이것이다(Rosetta: scan/prefix-sum kernel ↔ momentum의 associative scan — **동일**).
2. **state에 nonlinear한 부분** — deep memory의 gradient $\nabla_W\ell(W_{t-1};k_t,v_t)$처럼 state가 MLP의 forward/backward를 **통과하는** 항. 이 recurrence에는 닫힌 형태도 scan도 없다 — 비선형 recurrence의 sequence 방향 병렬화는 미해결 문제로 남아 있고, parallel scan은 적용되지 않는다 [TNT §1]. 유일하게 알려진 수는 **얼리는 것**이다: chunk 안의 모든 gradient를 chunk 시작 상태 $W_{\xi(t,C)}$에서 평가한다 ($\xi(t,C)=C\lfloor(t-1)/C\rfloor$, → §1.2). 이것이 이 책이 **stale-snapshot 근사**(chunk-start anchor)라고 부르는 조작이며, 표준형 (M4)가 그 일반형이다:

$$
W_t \;=\; W_{\xi(t,C)} \;-\; \sum_{\tau=\xi(t,C)+1}^{t} \eta_\tau\,\nabla_W\,\ell\big(W_{\xi(t,C)};\,k_\tau,v_\tau\big)
\tag{M4}
$$

<!-- FIG: ch09/fig-01-chunkwise-dataflow -->

핵심 문장을 반복한다: **chunk 안의 모든 gradient는 chunk 시작 상태 $W_{\xi(t,C)}$에서 평가된다. 따라서 $C$는 스케줄이 아니라 계산되는 함수 자체를 바꾸는 semantic hyperparameter다** — FlashAttention tiling(bit-exact)과의 결정적 차이. anchor가 공유되므로 chunk 내부의 $C$개 gradient는 상호 독립이고, "weight를 공유하는 batch $C$짜리 forward+backward" 한 번으로 계산된다. 2장에서 굵게 강조한 문장이 여기서 kernel 수준의 실체를 얻는다: **이 라인에서 훈련의 batch 축 역할을 하는 것은 sequence 축이다** — chunk가 곧 inner loop의 mini-batch다.

**intra-chunk 계산의 정체.** 얼리고 나면 chunk 내부의 linear recurrence는 전부 $C\times C$의 **삼각 구조**로 물화(materialize)된다. 두 가지 꼴이 있다. (i) 계수가 미리 알려진 explicit recurrence(decay 누적곱, momentum 가중합)는 삼각 가중 행렬과의 GEMM 한 번이 된다 — Atlas가 momentum을 $\bar\beta$-가중 행렬 곱으로 물화하는 것이 이 꼴이다 [Atlas Eq. 39]. (ii) 입력이 과거 출력에 의존하는 implicit recurrence(delta rule의 pseudo-value, → 9.4절)는 단위 삼각 행렬의 **triangular solve**가 된다. 두 경우 모두 독자가 FlashAttention tile 안에서 매일 보는 $C\times C$ score 행렬과 같은 shape의 구조화된 연산이다. "chunk 내부는 attention처럼, chunk 경계는 RNN처럼"(→ 8장)이라는 슬로건의 일반형이 이것이다.

**inter-chunk 계산의 정체.** chunk 경계에서는 상태 하나($W_{nC}$, 필요시 $S_{nC}$)만 다음 chunk로 넘긴다. 이 handoff가 linear면(인스턴스 1, 2) chunk 사이도 scan으로 묶을 수 있고, nonlinear면(인스턴스 3, 4) $L/C$개의 순차 사슬이 남는다 — 이 사슬의 길이가 deep memory 훈련의 최종 병목이며, 이를 끊는 것이 TNT의 reset이다(→ 9.8절).

**비용.** chunk당 intra 항 $O(C^2d)$ + state 상호작용 $O(Cd^2)$ + 경계 연산 $O(\mathrm{poly}(d))$, 총

$$
O\!\Big(\tfrac{L}{C}\,\big[\,C^2d + C\,d^2 + \mathrm{poly}(d)\,\big]\Big)
\tag{9-1}
$$

$C=1$이면 순수 recurrence, $C=L$이면 순수 parallel attention — chunk 크기는 두 극한 사이를 보간한다. 여기까지는 7장의 exact한 세계와 같은 산수다. 다른 것은 단 하나, 인스턴스 3부터 이 보간이 **품질 축도 함께 움직인다**는 점이다.

## 9.3 인스턴스 1 — GLA / RetNet / Mamba-2: 대각 transition, exact한 folding

가장 쉬운 사례부터. 식 (6-2)와 §1.6 카탈로그의 gate형 update $W_t = \alpha_t W_{t-1} + v_tk_t^\top$ ($\alpha_t$: 스칼라, GLA는 채널별 대각 — folding은 elementwise로 동일하게 작동한다)는 state에 완전히 linear하다. 장-국소 기호로 **누적 retention** $\bar\alpha_{\tau\to t} := \prod_{j=\tau+1}^{t}\alpha_j$ ($\bar\alpha_{t\to t}=1$)를 정의하면, chunk 시작 $\xi := \xi(t,C)$에서 출발한 닫힌 형태가 즉시 나온다:

$$
W_t \;=\; \bar\alpha_{\xi\to t}\,W_{\xi} \;+\; \sum_{\tau=\xi+1}^{t} \bar\alpha_{\tau\to t}\;v_\tau k_\tau^\top,
\qquad
y_t \;=\; \bar\alpha_{\xi\to t}\,(W_\xi q_t) \;+\; \sum_{\tau=\xi+1}^{t} \bar\alpha_{\tau\to t}\,(k_\tau^\top q_t)\,v_\tau
\tag{9-2}
$$

읽는 법: 출력의 첫 항은 이전 chunk까지의 압축 상태를 decay시켜 읽는 **inter-chunk 항**(GEMV, chunk당 GEMM $Q_nW_\xi^\top$로 묶임), 둘째 항은 decay 가중치가 곱해진 **causal-masked attention**(intra-chunk 항)이다. chunk의 $K_n, V_n, Q_n \in \mathbb{R}^{C\times d}$를 행으로 쌓으면 score $Q_nK_n^\top$, decay-mask 적용, value 곱 — 전부 7장의 식 (7-4)에서 이미 본 GEMM들이고, 실제로 Mamba-2의 SSD chunked algorithm이 정확히 이 인스턴스다(→ 7장). 경계 handoff도 linear이므로 chunk 사이마저 scan으로 처리할 수 있다.

결정적 사실: (9-2)는 **항등 변형**이다. 어떤 근사도 없다. $C$를 1로 하든 64로 하든 4096으로 하든 같은 $W_t$, 같은 $y_t$가 나온다(부동소수점 오차 제외). 따라서 이 family에서 $C$는 FlashAttention의 tile 크기와 정확히 같은 지위 — SRAM 크기와 GEMM 살찌우기에 맞춰 고르는 순수 성능 knob — 를 가진다. 구현 디테일로는 decay 누적곱의 dynamic range 때문에 GLA가 log-공간 계산과 chunk 내부의 2차 tile 분할(secondary-level chunking)을 쓴다는 점만 언급해 둔다 [GLA §4.3].

## 9.4 인스턴스 2 — DeltaNet: 비대각 transition과 WY representation

6장이 미뤄 둔 빚을 갚는다. DeltaNet의 update (식 (6-3), → 6장)는

$$
W_t = W_{t-1}\big(I - \eta_t k_tk_t^\top\big) + \eta_t v_tk_t^\top
$$

로, transition이 generalized Householder $I-\eta_tk_tk_t^\top$ — **비대각 행렬**이다. state에 여전히 linear하므로 원리상 exact 병렬화가 가능하지만, 9.3절의 folding은 실패한다: 대각 계수의 누적곱은 elementwise로 접히지만, Householder의 누적곱 $\prod_\tau(I-\eta_\tau k_\tau k_\tau^\top)$은 일반 $d_k\times d_k$ 행렬이고, 이를 순진하게 물화하면 token당 $O(d^3)$ 행렬곱 — 병렬화로 얻은 것을 도로 태운다. Yang et al. 2024 (arXiv:2406.06484)가 DeltaNet을 실용화한 열쇠가 수치선형대수의 고전인 **WY representation**이다: Householder 곱을 물화하는 대신 **compact한 합의 형태로 유지**한다.

유도는 세 줄이다. 첫째, delta update를 다시 쓴다. **pseudo-value** $\tilde v_t := \eta_t\,(v_t - W_{t-1}k_t)$ (장-국소 기호: 교정된 value)를 정의하면

$$
W_t = W_{t-1} + \tilde v_t\,k_t^\top
\tag{9-3}
$$

— delta rule은 "raw value $v_t$ 대신 교정된 value $\tilde v_t$를 쓰는 Hebbian write"다(6장의 self-limiting write 논의의 대수적 재진술). 둘째, (9-3)을 chunk 시작부터 전개하면 $W_t = W_\xi + \sum_{\tau=\xi+1}^{t} \tilde v_\tau k_\tau^\top$. 셋째, 이 전개를 $\tilde v_t$의 정의에 도로 대입하면 pseudo-value 사이의 **implicit recurrence**가 나온다:

$$
\tilde v_t \;=\; \eta_t\Big(v_t - W_\xi k_t - \sum_{\tau=\xi+1}^{t-1} (k_\tau^\top k_t)\,\tilde v_\tau\Big)
\tag{9-4}
$$

$\tilde v_t$가 자기보다 앞선 $\tilde v_\tau$들에 의존한다 — 그러나 의존 계수가 스칼라 $k_\tau^\top k_t$뿐이다. chunk의 pseudo-value를 행으로 쌓아 $\tilde V \in \mathbb{R}^{C\times d_v}$, $D_\eta := \mathrm{Diag}(\eta_{\xi+1},\ldots,\eta_{\xi+C})$라 하면 (9-4)는

$$
\big(I + D_\eta\,\mathrm{tril}(K_nK_n^\top, -1)\big)\,\tilde V \;=\; D_\eta\big(V_n - K_nW_\xi^\top\big)
\tag{9-5}
$$

라는 **단위 하삼각 $C\times C$ 선형계**다. 좌변 행렬은 대각이 1인 하삼각이므로 항상 가역이고, forward substitution(TRSM — 독자가 쓰는 BLAS의 표준 연산)으로 $O(C^2)$ 스칼라곱 × $d_v$열에 풀린다. 이 삼각계를 만들어 푸는 절차가 Householder 곱의 compact 누적(수치선형대수 문헌의 UT transform, Joffrain et al. 2006)에 해당한다 [Yang et al. 2024 Eq. 10–11]. 나머지는 전부 GEMM이다: score $K_nK_n^\top$ ($C\times d_k$ 대 $d_k\times C$), RHS의 $K_nW_\xi^\top$ ($C\times d_k$ 대 $d_k\times d_v$), 출력 $Y_n = Q_nW_\xi^\top + \mathrm{tril}(Q_nK_n^\top,0)\,\tilde V$, 경계 갱신 $W_{\xi+C} = W_\xi + \tilde V^\top K_n$ ($d_v\times C$ 대 $C\times d_k$).

두 가지를 명시한다. 첫째, **이것도 항등 변형이다.** (9-3)→(9-5)의 어느 단계에도 근사가 없다 — $\tilde v_t$는 진짜 순차 delta rule이 만들었을 바로 그 값이고(triangular solve가 token 순서의 인과를 정확히 계산한다), 따라서 DeltaNet의 $C$ 역시 순수 성능 knob이다. Gated DeltaNet(식 (6-4))은 9.3절의 decay folding과 이 절의 WY를 한 kernel에 합성하면 된다. 둘째, 이 절이 **"optimizer-step recurrence를 병렬화하는 지적 템플릿"**인 이유다: delta rule은 곧 linear memory 위의 1-step GD(→ 5장, 6장)이므로, 방금 우리는 "gradient step의 열(列)을 exact하게 GEMM으로 재조직"하는 데 성공한 것이다. 그것이 가능했던 조건 — loss가 $\ell_2$이고 memory가 linear라서 **gradient가 state에 linear** — 를 기억해 두라. 다음 절에서 이 조건이 깨지는 순간 무엇을 지불하게 되는지 본다.

## 9.5 인스턴스 3 — TTT dual form: anchor라는 선택, 함수라는 대가

8장 §8.4가 TTT-Linear에 대해 완결한 dual form — anchored 전개 (8-3)과 닫힌 출력 (8-4) — 을 이 장의 언어로 재정위한다. dual form은 (M4) 그 자체다: chunk 내 gradient를 전부 $W_\xi$에서 평가하고, intra-chunk 출력은 residual value의 causal attention 꼴로, 경계 상태는 rank-$C$ GEMM으로 계산한다. TNT는 이 형태를 라인의 표준 정식화로 승계했고 [TNT Eq. 3], 이 책의 $\xi(t,C)$ 표기도 TNT의 것이다(→ §1.2).

여기서 이 장의 관점이 주는 새 정보는 다음의 대조다. **TTT-Linear는 ungated DeltaNet과 같은 update rule이다**(→ 8장 §8.3). 그렇다면 9.4절의 WY가 그대로 적용되어 exact 병렬화가 가능하다. 그런데 Sun et al. 2024는 그 길 대신 anchored mini-batch GD를 택했다 [Sun et al. 2024 §2.4–2.5]. 왜인가. 두 이유가 이 라인 전체의 설계 논리를 드러낸다.

첫째, **deep memory 때문이다.** $\mathcal{M}$이 2-layer MLP가 되는 순간 $\nabla_W\ell$은 $W$에 nonlinear해지고(backward가 MLP를 통과한다), (9-4)류의 implicit recurrence는 스칼라 계수로 닫히지 않는다. WY의 전제 조건이 사라진 자리에서, 알려진 유일한 병렬화 수단이 anchor다. 즉 stale-snapshot 근사는 취향이 아니라 **비선형 recurrence의 병렬화라는 미해결 문제에 대한 현존 유일 응답**이다 [TNT §1]. MLP memory의 intra-chunk 계산도 같은 원리로 tensorize된다는 것을 Sun et al. 2024가 App. A에서 보였다 — 형태는 linear 경우보다 복잡하지만 골격(anchored gradient의 batched 계산 + intra-chunk correction)은 동일하다.

둘째, 더 미묘하고 더 중요하다: **근사가 kernel에서 model 정의로 이사했다.** TTT layer는 "inner mini-batch 크기 $C$의 mini-batch GD로 update하는 layer"로 **정의**된다. dual form은 그 정의를 exact하게 계산하는 알고리즘이다. 비교하라 — FlashAttention은 softmax attention이라는 고정된 함수의 exact한 재배열이고, DeltaNet의 WY는 순차 delta rule이라는 고정된 함수의 exact한 재배열이다. 반면 TTT에서 $C$를 바꾸는 것은 **다른 layer를 정의하는 것**이다: $C=1$이면 순차 delta rule(DeltaNet과 일치), $C=L$이면 1-step batch GD — 곧 linear attention의 additive write로 퇴화하고(8장 §8.8의 손계산: $C{=}1$은 4, $C{=}2$는 6), 그 사이의 모든 $C$는 서로 다른 함수다. 함수의 family가 $C$로 매개변수화된 것이다. 이것이 다음 명제의 정확한 의미다.

**명제 (chunk 크기 = semantic hyperparameter).** anchored chunkwise family(TTT, Titans, Atlas의 deep memory)에서 chunk 크기 $C$는 계산 스케줄이 아니라 계산되는 함수의 매개변수다. 같은 slow weights $\Theta$라도 $C$가 다르면 다른 sequence-to-sequence 함수가 실행된다. 따라서 (i) 훈련은 특정 $C$의 함수에 대해 이루어지고, (ii) serving에서 다른 $C$를 쓰는 것은 훈련되지 않은 함수를 실행하는 것이다.

(ii)의 결과가 실제로 관측된 것이 [TNT Fig. 2]의 chunk-size mismatch이고(→ 15장; 수치는 9.7절), **'chunk 크기 = semantic hyperparameter' 명제**의 명명과 소유는 이 장에 있다(→ §2.4; mismatch 현상 자체의 소유는 15장). 반면 exact family(GLA·DeltaNet·Mamba-2)에서는 train과 serve의 $C$가 달라도 함수가 같으므로 mismatch가 **정의상 존재하지 않는다** — "우리 모델은 chunk 병렬화된다"는 문장을 논문에서 만나면, 독자는 이제 이 두 지위 중 어느 쪽인지부터 물어야 한다.

## 9.6 인스턴스 4 — Titans: retention folding + momentum scan

Titans의 update (M2)는 세 성분의 합성이다: anchored gradient(비선형, 인스턴스 3의 수법), retention gate $\alpha_t$(스칼라 linear, 인스턴스 1의 수법), momentum $S_t$(linear recurrence, scan). Titans §3.2의 병렬화는 정확히 이 세 수법의 조립이고, 그래서 이 장의 마지막 인스턴스로 배치했다 — 새 아이디어가 아니라 일반 scheme의 총동원이다.

**1단계 — retention folding** [Titans Eq. 16]. momentum을 잠시 끄고 $W_t = \alpha_t W_{t-1} - \eta_t\nabla_W\ell(W_{\xi};k_t,v_t)$를 chunk 시작부터 전개하면(식 (9-2)와 같은 folding, 9.3절의 $\bar\alpha$ 재사용):

$$
W_t \;=\; \bar\alpha_{\xi\to t}\,W_{\xi} \;-\; \sum_{\tau=\xi+1}^{t} \bar\alpha_{\tau\to t}\;\eta_\tau\,\nabla_W\ell\big(W_{\xi};k_\tau,v_\tau\big)
\tag{9-6}
$$

(원문 Eq. 16의 $\beta_i = \prod_j(1-\alpha_j)$는 통일 표기의 $\bar\alpha_{0\to i}$에 해당한다 — Titans의 gate 기호는 이 책과 방향·배치가 다르므로 주의: 원문 $\theta_t$→통일 $\eta_t$, 원문 $\eta_t$→통일 $\beta_t$, 원문 $\alpha_t$→통일 $1-\alpha_t$. 전체 대응표는 12장 §1.7.1.) linear memory라면 gradient가 $(W_\xi k_\tau - v_\tau)k_\tau^\top$이므로 (9-6)의 합 전체가 $\big(W_\xi K_n^\top - V_n^\top\big)\,\mathrm{Diag}(\bar\alpha\eta)\,K_n$ — rank-$C$ GEMM 하나로 물화된다 [Titans Eq. 17]. deep memory라면 같은 합이 "anchor $W_\xi$에 대한 batch $C$짜리 forward+backward 한 번"으로 계산된다. 어느 쪽이든 2장의 $dW = \delta\,x^\top$ shape 그대로다.

**2단계 — momentum scan** [Titans Eq. 18]. momentum을 켜면 $S_t = \beta_t S_{t-1} - \eta_t g_t$, $g_t := \nabla_W\ell(W_\xi;k_t,v_t)$ (anchored gradient). 관건은 anchor 덕분에 **$g_t$ 전부를 미리 계산할 수 있다**는 것 — momentum recurrence가 memory 상태와 분리된, 입력이 알려진 순수 linear recurrence가 된다. 그러면 이것은 S5(arXiv:2208.04933)식 associative scan의 교과서 사례다: 쌍 $(\beta_t,\, -\eta_t g_t)$에 결합 법칙을 만족하는 연산 $(a,b)\bullet(a',b') := (a'a,\ a'b + b')$을 주면 prefix 곱의 둘째 성분이 곧 $S_t$다. 독자의 prefix-sum kernel이 그대로 돌아간다. Atlas는 같은 recurrence를 scan 대신 삼각 가중 행렬로 물화해 푼다 [Atlas Eq. 36–39] — 9.2절에서 말한 explicit recurrence의 두 등가 구현이다.

**3단계 — 조립.** chunk 하나의 훈련 kernel은 결국: (i) anchored gradient의 batched forward+backward(GEMM 덩어리), (ii) momentum의 intra-chunk scan 혹은 삼각 GEMM, (iii) 식 (9-6)의 누적 retention 적용, (iv) 출력 계산, (v) 경계 handoff — deep memory에서는 이 handoff가 비선형이므로 chunk 사이는 $L/C$ 길이의 순차 사슬로 남는다. Titans는 여기에 변형 하나를 더 제시한다: 게이트 $\eta,\beta,\alpha$를 token의 함수가 아니라 **chunk의 함수**(chunk 내 상수)로 두면 intra-chunk가 LTI 시스템이 되어 global convolution으로도 계산할 수 있다는 것이다 — 표현력을 잃는 대신 더 빨라지는 선택지로, 실험은 token-dependent 쪽을 썼다 [Titans §3.2].

**Atlas의 확장.** Omega rule은 gradient의 평가 대상을 token 하나에서 window $c$개로 바꾸지만(식 (M3), → 14장), 병렬화 구조는 동일하다: chunk를 자르고 gradient를 직전 chunk의 마지막 상태에 anchor한다 [Atlas §3.4]. sliding-window masking으로 window 항을 intra-chunk GEMM에 흡수하고, Muon을 쓸 때는 미리 계산된 momentum 항들에 $\mathrm{NS}_\kappa$를 적용한 뒤 update한다 [Atlas Eq. 40–41] — $S_t$가 전부 병렬로 나와 있으므로 Newton–Schulz도 위치별로 병렬인 batched GEMM $\kappa$회다. 2장에서 "NS-5 = 작은 GEMM 다섯 번"으로 배운 비용 감각이 그대로 이식된다.

**[해설]** 이 조립에는 systems 독자가 즉시 봐야 할 비용이 하나 숨어 있다. momentum scan의 원소 $g_t, S_t$는 **weight와 같은 shape**이다. linear memory에서는 $g_t$가 rank-1이라 (8-4)류의 닫힌 형태로 물화를 피할 수 있지만, deep memory에서는 $g_t$가 진짜 parameter-shape($P_f$개 원소)이고, chunk당 $C$개를 만들었다 스캔한다 — traffic이 $O(C\,P_f)$다. 작은 $C$의 deep-memory 훈련이 compute-bound가 아니라 memory-bound라는 [TNT §1]의 진단은 이 구조에서 나온다.

## 9.7 Semantic hyperparameter로서의 $C$: 두 지위의 총정리

네 인스턴스를 한 표로 접는다.

표 9-2 — 네 인스턴스와 chunk 크기 $C$의 지위

| 인스턴스 | transition의 구조 | intra-chunk 핵심 연산 | inter-chunk | exact? | $C$의 지위 |
|---|---|---|---|---|---|
| GLA / RetNet / Mamba-2 | 스칼라·대각 linear | decay-masked attention GEMM | 스칼라 folding·scan 가능 | **exact** | 성능 knob (tile과 동급) |
| DeltaNet / Gated DeltaNet | 비대각 linear (Householder 곱) | $C{\times}C$ triangular solve + GEMM | 행렬 handoff | **exact** | 성능 knob |
| TTT dual form | nonlinear (또는 linear라도 anchored 정의) | residual-value causal attention (8-4) | 순차 handoff | anchored 근사 | **semantic** |
| Titans (+Atlas/Muon) | nonlinear + linear momentum + 스칼라 retention | batched fwd/bwd + scan + folding (+$\mathrm{NS}_\kappa$) | 순차 handoff (비선형) | anchored 근사 | **semantic** |

위 두 행과 아래 두 행을 가르는 판정 기준은 단 하나다: **update가 state에 linear한가.** linear면 삼각 구조로 exact하게 물화되고, nonlinear면 얼려야 하며, 얼린 결과는 함수의 정의에 들어간다.

semantic이라는 말의 실증적 무게는 [TNT Fig. 2]가 보여 준다. $C=64$로 pretraining한 550M Titans를 서로 다른 inference chunk로 돌리면 validation perplexity가 $C{=}8$에서 36.45, $16$에서 34.15, $32$에서 24.23, **훈련값인 $64$에서 13.78로 최적**, 이후 $128$에서 15.5, $256$에서 17.88, $512$에서 22.4로 다시 나빠진다 [TNT Fig. 2]. 두 가지가 주목할 만하다. 첫째, "작은 chunk = 신선한 gradient = 항상 더 좋음"이라는 직관이 틀렸다 — 모델은 훈련된 $C$라는 **해상도에 과적응**하고, 더 신선한 update조차 훈련 분포 밖이면 해가 된다. 둘째, 이것은 serving의 이상적 동작점인 $C=1$ decode(per-token online write, → 1장 Rosetta)를 정면으로 막는다: 큰 $C$로 훈련된 모델을 $C=1$로 돌리면 무너진다. 이 관측과 그 처방(두 단계 훈련)의 본격 분석은 15장의 몫이며, 그 증거가 단일 설정(550M, gating/momentum 없는 단순화 Titans)의 그림 하나라는 정직성 caveat도 15장이 진다.

이 장이 짊어질 정직성 caveat는 따로 있다: **stale-snapshot 근사의 오차에 대한 형식적 bound는 여섯 논문 어디에도 없다.** anchored update가 순차 update에서 얼마나 벗어나는지($C$, gate 값, curvature의 함수로서)는 정리도 lemma도 없이 전적으로 실험(위의 Fig. 2가 사실상 전부)에 맡겨져 있다. staleness를 다루는 이론 어휘 자체는 3장의 online learning(지연된 feedback 하의 regret)이 제공할 후보지만, 이 라인의 누구도 그 연결을 수행하지 않았다. 독자가 이 라인을 production에 들일 때 감수해야 할 미정량 리스크의 목록 첫 줄에 이것을 적어 두라.

마지막으로 층위를 하나 올린다. [NL]은 이 장의 그림을 뒤집어 읽는다: "chunk 경계마다만 update한다"를 근사의 부산물이 아니라 **update frequency라는 설계 축**으로 승격시키고, 서로 다른 $C^{(\ell)}$로 도는 level의 스펙트럼(CMS)으로 아키텍처를 재구성한다(식 (M5), → 16장). 그 관점에서 이 장의 $C$는 가장 안쪽 level의 update 주기이고, outer loop의 optimizer step은 가장 바깥 level의 그것이다 — 같은 scheme이 재귀적으로 반복된다.

## 9.8 Systems frontier: 그래도 deep memory는 느리다 — TNT의 처방 (골격)

인스턴스 3, 4까지 조립해도 deep memory 훈련의 MFU는 낮다. 원인은 이 장의 언어로 정확히 셋이다. (1) **순차 사슬**: inter-chunk handoff가 비선형이라(상태가 MLP weights 그 자체이고, 구현에 따라 chunk 끝 normalization까지 낀다 [Atlas App. E]) chunk 사이를 scan으로 묶을 수 없고, 사슬 길이 $L/C$가 그대로 critical path다. (2) **skinny GEMM**: 품질이 요구하는 작은 $C$(8–64)에서 batched fwd/bwd의 batch 축이 $C$이므로 GEMM이 말라서 tensor core가 차지 않는다 — decode에서 batch를 못 채울 때의 그 현상이다. (3) **parameter-shape traffic**: 9.6절의 [해설]대로 scan 원소가 $P_f$-shape라 memory-bound다. 세 원인의 합산 결과가 앞서 인용한 peak 대비 5–10% 미만의 FLOPs utilization이다 [TNT §1].

[TNT]의 처방은 이 장의 scheme 안에서 정확히 두 수를 둔다(기법의 골격만 여기서 세우고, 평가와 전모는 15장). 첫 수는 **계층화**다: 큰 chunk $C_{\mathrm{g}}$(실험은 2048)로 도는 global memory $W^{\mathrm{g}}$가 장거리 문맥을 맡아 순차 사슬을 $L/C_{\mathrm{g}}$개의 크고 dense한 handoff로 줄이고, 작은 chunk의 local memory $W^{\mathrm{l}(i)}$들이 세밀한 해상도를 맡는다. 둘째 수가 결정타다: local memory의 상태를 **주기 $L_{\mathrm{s}}^{(i)}$마다 학습된 초기 상태 $W_{\mathrm{init}}$로 reset**한다. reset은 shard 경계를 가로지르는 상태 의존을 (gradient 경로까지 포함해) 완전히 절단하므로, $L/L_{\mathrm{s}}$개의 shard가 **서로 독립**이 된다 — 비선형 recurrence에는 scan이 없다는 벽을, recurrence를 병렬화하는 대신 **recurrence 자체를 주기적으로 끝내 버리는** 방식으로 우회한 것이다. 독립 shard는 device들에 분산하거나(context parallelism — 독자에게는 data parallelism과의 유비로: 자를 수 없던 sequence 축이 reset 덕분에 batch 축처럼 잘리게 됐다) 한 device의 batch 축에 쌓아 kernel을 살찌운다. reset이 버리는 장거리 문맥은 global memory가 줍고, $W_{\mathrm{init}}$이 outer loop에서 학습되므로(4장의 MAML 유비, 8장 표 8-1의 그 $W_{\mathrm{init}}$) 매 shard는 0이 아니라 meta-learn된 prior에서 출발한다. 여기에 훈련은 큰 $C_{\mathrm{l}}$로, 마지막 ~5%의 compute로 작은 $C_{\mathrm{l}}'$(이상적으로 1)에 fine-tune하는 **two-stage 전략**이 얹혀 9.7절의 mismatch를 치유한다 — 훈련의 $C$와 serving의 $C$를 분리해, semantic hyperparameter가 강요하던 단일 절충값을 두 개의 knob으로 쪼갠 것이다. 이 조합으로 TNT는 가장 정확한 Titans baseline 대비 최대 17.37× 빠르게 같은 loss에 도달했다고 보고한다 [TNT Table 1]. 나머지 — Q-K projection $\Pi_t$, 다중 해상도 $\{C_{\mathrm{l}}^{(i)}\}$, 수치와 한계 — 는 15장에서.

## 9.9 Worked micro-example: 같은 두 token, 두 개의 병렬화

8장 §8.8의 예제를 그대로 이어받아 — $d_k=d_v=2$, $k_1=k_2=(1,0)^\top$, $v_1=(2,0)^\top$, $v_2=(4,0)^\top$, $W_0=0$, $\eta_t=1$, $q_2=(1,0)^\top$ — 이 장의 두 주장(WY의 exactness, momentum scan의 exactness)을 손으로 확인한다. 8장에서 이미 안다: 순차 delta rule($C{=}1$)의 답은 $y_2=(4,0)^\top$, anchored dual form($C{=}2$)의 답은 $(6,0)^\top$이었다.

**Part A — DeltaNet을 $C=2$로, WY/UT로.** (9-5)를 조립한다. $K_2 = \begin{pmatrix}1&0\\1&0\end{pmatrix}$, $V_2=\begin{pmatrix}2&0\\4&0\end{pmatrix}$, $D_\eta=I$, $W_0=0$이므로 RHS $=V_2$. score는 $K_2K_2^\top = \begin{pmatrix}1&1\\1&1\end{pmatrix}$, 그 strict lower 부분은 $\begin{pmatrix}0&0\\1&0\end{pmatrix}$. 삼각계는

$$
\begin{pmatrix}1&0\\1&1\end{pmatrix} \tilde V = \begin{pmatrix}2&0\\4&0\end{pmatrix}
\;\Rightarrow\;
\tilde V = \begin{pmatrix}2&0\\2&0\end{pmatrix}
$$

forward substitution 두 줄이다: 1행 $\tilde v_1^\top=(2,0)$; 2행 $\tilde v_2^\top = (4,0)-(2,0) = (2,0)$ — solve가 "token 2는 token 1이 이미 2를 써 두었음을 안다"는 인과를 정확히 재현했다(8장의 anchored 계산은 이 빼기를 못 해서 $(4,0)$을 통째로 썼다). 상태와 출력은

$$
W_2 = \tilde V^\top K_2 = \begin{pmatrix}2&2\\0&0\end{pmatrix}\begin{pmatrix}1&0\\1&0\end{pmatrix} = \begin{pmatrix}4&0\\0&0\end{pmatrix},
\qquad y_2 = W_2q_2 = (4,0)^\top
$$

순차 계산과 **정확히 일치**한다. chunk를 2로 잘랐지만 함수는 그대로다 — exact family의 실감이다.

**Part B — Titans의 momentum scan ($C=2$, $\beta_t\equiv\tfrac12$, $\alpha_t\equiv 1$).** anchored gradient부터: $g_t = (W_0k_t - v_t)k_t^\top = -v_tk_t^\top$이므로 $g_1 = \begin{pmatrix}-2&0\\0&0\end{pmatrix}$, $g_2 = \begin{pmatrix}-4&0\\0&0\end{pmatrix}$ — 두 개가 **동시에**(batch 2의 GEMM으로) 나온다. 순차로 momentum을 돌리면: $S_1 = \tfrac12 S_0 - g_1 = \begin{pmatrix}2&0\\0&0\end{pmatrix}$, $S_2 = \tfrac12 S_1 - g_2 = \begin{pmatrix}5&0\\0&0\end{pmatrix}$. 이제 scan으로: 쌍 $(\tfrac12, -g_1)$과 $(\tfrac12, -g_2)$에 9.6절의 결합 연산 $(a,b)\bullet(a',b')=(a'a,\ a'b+b')$을 적용하면 $\big(\tfrac14,\ \tfrac12\begin{pmatrix}2&0\\0&0\end{pmatrix} + \begin{pmatrix}4&0\\0&0\end{pmatrix}\big)$이고 둘째 성분이 $\begin{pmatrix}5&0\\0&0\end{pmatrix}$ — 순차 계산과 일치한다. momentum recurrence 자체는 linear이므로 scan은 **아무 근사도 더하지 않는다**. 상태는 $W_1 = W_0+S_1$, $W_2 = W_1+S_2 = \begin{pmatrix}7&0\\0&0\end{pmatrix}$, 읽으면 $y_2=(7,0)^\top$.

세 숫자를 나란히 놓자: 순차 delta rule 4, anchored TTT($C{=}2$) 6, anchored Titans($C{=}2$, momentum) 7. **4→4(Part A)는 exact 병렬화** — $C$가 함수를 건드리지 않았다. **4→6은 anchor의 staleness** — 근사가 함수를 바꿨다. **6→7은 momentum** — token 1의 surprise 절반($\tfrac12\times 2$)이 token 2의 write에 실려 들어온 것이고, 이는 근사가 아니라 (M2)가 정의한 함수의 정당한 출력이다. 어느 차이가 알고리즘이고 어느 차이가 근사인지 — 이 장에서 가져가야 할 감각이 정확히 이 구분이다.

## 9.10 Systems bridge: roofline 위의 $C$

이 장 전체를 독자의 cost model로 접는다. 식 (9-1)의 chunk당 비용을 linear-memory chunkwise kernel(인스턴스 1–2, head당, $d_k=d_v=d$)에 대해 구체화하자. FLOPs(MAC당 2 FLOP): score $Q_nK_n^\top$에 $2C^2d$, masked score와 value/pseudo-value의 곱에 $2C^2d$, 상태 읽기 $Q_nW_\xi^\top$에 $2Cd^2$, 상태 갱신($U^\top K_n$ 또는 $K_n^\top V_n$류)에 $2Cd^2$ — 합계 $F(C) \approx 4C^2d + 4Cd^2$. bytes(bf16, 2 B/원소): $K,Q,V$ 읽기와 $Y$ 쓰기 $4Cd$ 원소, 상태 read-modify-write $2d^2$ 원소 — 합계 $B(C) \approx 8Cd + 4d^2$ bytes. 나누면 놀랄 만큼 깨끗한 식이 나온다:

$$
\mathrm{AI}(C) \;=\; \frac{F(C)}{B(C)} \;=\; \frac{4Cd\,(C+d)}{4d\,(2C+d)} \;=\; \frac{C\,(C+d)}{2C+d}
\;\;\xrightarrow{\,C\ll d\,}\;\; \approx C
\tag{9-7}
$$

**작은 chunk regime에서 arithmetic intensity는 곧 chunk 크기다.** roofline의 x축 좌표를 $C$라는 단일 knob이 직접 쥐고 있는 셈이다. 숫자를 넣어 보자($d=1024$).

표 9-3 — $\mathrm{AI}(C)$와 GEMM shape ($d=1024$, bf16, head당)

| $C$ | 주 GEMM shape ($C\times d \cdot d\times d$ 등) | $\mathrm{AI}(C)$ [FLOP/B] | ridge 대비 |
|---|---|---|---|
| 8 | $8\times1024$ 짜리 — 극단적 skinny | 7.9 | ~3% |
| 16 | $16\times1024$ | 15.8 | ~5% |
| 64 | $64\times1024$ | 60.4 | ~20% |
| 256 | $256\times1024$ | 213 | ~72% |
| 1024 | $1024\times1024$ — 정방 | 683 | ridge 초과 |

ridge는 대표적 최신 accelerator의 공개 spec(NVIDIA H100 SXM datasheet: bf16 dense 약 989 TFLOP/s, HBM3 약 3.35 TB/s)으로 약 295 FLOP/B다. 표가 말하는 것: 품질이 원하는 $C$(8–64 — TTT의 inner mini-batch, TNT 실험에서 품질 최선이던 Titans $C{=}8$)에서 kernel은 ridge의 3–20% 지점, 즉 깊은 bandwidth-bound 영역에 앉아 있고, MFU가 원하는 $C$는 300 이상이다. **품질 최적 $C$와 MFU 최적 $C$는 같은 축의 반대편 끝에 있다.** [TNT §1]이 인용한 "peak의 5–10% 미만"은 이 표의 위쪽 행들을 그대로 읽은 것이며, exact family라면 이 긴장은 "훈련이 좀 느리다"로 끝나지만 semantic family에서는 $C$가 품질 축이기도 하므로 **절충 불가능한 이율배반**이 된다 — 이것이 TNT라는 논문이 존재하는 이유의 전부이고, 독자는 이제 그것을 (9-7) 한 줄에서 재유도할 수 있다.

deep memory(인스턴스 3–4)는 여기에 두 겹을 얹는다. batched fwd/bwd의 GEMM들은 batch 축이 $C$인 $(C\times d)\cdot(d\times 4d)$ shape이라 — decode에서 batch를 못 채운 GEMM과 동일한 병리로 — 작은 $C$에서 tensor core 효율이 더 깎이고, momentum·gradient가 parameter-shape($P_f\approx 8d^2$)라 scan traffic이 $O(C\,P_f)$로 커진다(9.6절 [해설]). 같은 $C$에서 deep memory의 실효 MFU가 linear memory보다 더 나쁜 이유이며, TNT의 reset이 shard들을 batch 축으로 쌓아 공격하는 지점이 정확히 첫 번째 겹이다. 이 회계의 아키텍처 전반 cheat sheet — softmax attention부터 Titans까지 한 표 — 는 10장이 완성한다.

## 요약

- chunkwise-parallel training의 일반 scheme은 하나다: update를 state-linear 성분과 state-nonlinear 성분으로 분해해, linear 성분(retention 누적곱, momentum EMA)은 folding·scan·삼각 GEMM으로 **정확히**, nonlinear 성분(deep memory의 gradient)은 chunk-start anchor로 **얼려서** 계산한다. chunk 내부는 $C\times C$ 삼각 구조의 GEMM, chunk 사이는 handoff다.
- 판정 기준은 transition의 구조다: 스칼라·대각 linear(GLA/RetNet/Mamba-2)는 decay folding으로, 비대각 linear(DeltaNet)는 WY representation의 pseudo-value 삼각계 (9-5)로 — 둘 다 **exact**이고 $C$는 순수 성능 knob이다.
- nonlinear transition에는 exact 재조직이 없다; stale-snapshot 근사 (M4)가 현존 유일 응답이고, 그 결과 근사가 model 정의로 들어가 **chunk 크기 $C$는 semantic hyperparameter가 된다** — 같은 $\Theta$라도 $C$가 다르면 다른 함수다(손계산: 순차 4 vs anchored 6). FlashAttention tiling(bit-exact)과 범주가 다르다.
- Titans의 병렬화는 세 수법의 조립이다: anchored gradient의 batched fwd/bwd [Titans Eq. 16–17] + momentum의 associative scan [Titans Eq. 18] + retention folding; Atlas는 같은 골격에 window masking과 $\mathrm{NS}_\kappa$를 얹는다 [Atlas §3.4, Eq. 36–41].
- semantic의 실증: $C{=}64$로 훈련된 550M Titans는 inference chunk 64에서 ppl 13.78로 최적이고 8에서 36.45, 512에서 22.4로 무너진다 [TNT Fig. 2] — 모델은 훈련 해상도에 과적응하며, $C{=}1$ decode라는 이상적 serving 모드를 막는다.
- stale-snapshot 근사의 오차에 대한 형식적 bound는 여섯 논문 어디에도 없다 — 이 라인의 최대 미정량 리스크다.
- arithmetic intensity는 $\mathrm{AI}(C) = C(C+d)/(2C+d) \approx C$ (작은 $C$): 품질 최적 $C$(8–64)는 ridge의 3–20%, MFU 최적 $C$는 300+ — 이 이율배반이 TNT의 존재 이유이고, TNT는 계층화(global $C_{\mathrm{g}}$) + 주기적 reset(context parallelism) + two-stage 훈련으로 두 knob을 분리한다.

## 자가 점검 체크리스트

- [ ] 임의의 update rule을 받아 state-linear/nonlinear 성분으로 분해하고, exact 병렬화 가능 여부를 판정할 수 있다.
- [ ] DeltaNet의 pseudo-value 재작성 (9-3)–(9-5)를 백지에서 유도하고, 삼각계의 forward substitution이 왜 순차 delta rule과 bit-동등한지(부동소수점 제외) 설명할 수 있다.
- [ ] $d=2$ 예제에서 순차(4)/anchored(6)/momentum(7)의 세 답을 재현하고, 각 차이가 근사인지 알고리즘인지 구분할 수 있다.
- [ ] "chunk 크기 = semantic hyperparameter" 명제를 정확히 진술하고, exact family에는 train/serve chunk mismatch가 정의상 없는 이유를 말할 수 있다.
- [ ] Titans chunk kernel의 5단계 조립(batched fwd/bwd → scan → folding → 출력 → handoff)과 각 단계의 GEMM/scan 지위를 나열할 수 있다.
- [ ] $\mathrm{AI}(C)$ 식 (9-7)을 유도하고, 자기 조직의 accelerator spec으로 ridge 대비 utilization을 추정할 수 있다.
- [ ] 이 장을 inference 어휘로 옮길 수 있다: chunk = semantic tile(bit-exact tile이 아님), momentum scan = prefix-sum kernel, 작은 $C$의 skinny GEMM = batch 못 채운 decode GEMM, TNT의 reset = sequence 축을 batch 축처럼 자르는 허가증.

## 다음 장으로

이 장으로 Part I의 모델·알고리즘 재료는 전부 갖춰졌다: memory의 어휘(5장), update rule의 계보(6–8장), 그리고 그것을 하드웨어에 올리는 유일한 방법(이 장). 남은 질문은 회계다 — softmax attention부터 Titans까지, state는 몇 byte이고 decode token당 FLOPs와 traffic은 얼마이며, prefill의 병렬성은 어느 유형이고, 논문들의 효율 주장은 어디까지 믿을 수 있는가. 10장은 이 장의 (9-7)식 분석을 아키텍처 전 계열의 cost-model cheat sheet로 확장하고, Part II에서 여섯 논문을 감사(audit)할 때 쓸 도구 상자와 용어집을 완성한다.
