# ch02. Training as a system: backprop과 optimizer를 (state, update, cost) 객체로 읽는다

> **이 장의 목표** — 이 장을 마치면 독자는 다음을 할 수 있어야 한다.
> 1. backward pass가 계산하는 두 GEMM의 shape를 쓰고, forward 대비 FLOPs 비율(약 2×)을 유도할 수 있다.
> 2. SGD / momentum / AdamW / AdaGrad / Shampoo / Muon 각각을 `update(state, gradient) → (state′, ΔΘ)` 서명을 갖는 **stateful 객체**로 기술하고, param당 state 크기와 step당 비용을 산정할 수 있다.
> 3. "momentum은 linear recurrence다", "weight decay의 감쇠 인자는 retention gate(역사적 별칭 forget gate → 13장)의 원형이다"라는 두 identity를 수식으로 진술할 수 있다 — 이 둘을 합치면 그대로 [Titans]의 memory update가 된다.
> 4. 훈련의 batch 축과 이 라인의 sequence 축이 어떻게 역할을 교대하는지 설명할 수 있다.
>
> **왜 필요한가** — 6편 전부가 이 장을 전제하며, 특히 다음 지점들이 이 장 없이는 읽히지 않는다.
> - [Titans] (*Titans: Learning to Memorize at Test Time*, arXiv:2501.00663) §3.1: long-term memory의 write 연산이 **문자 그대로 gradient descent + momentum + weight decay 한 step**이다(원문 Eq. 8, 10, 13–14). 세 optimizer 스칼라(learning rate, momentum decay, weight decay)가 token의 함수인 gate로 승격된 것이 전부다.
> - [Atlas] (*Atlas: Learning to Optimally Memorize the Context at Test Time*, arXiv:2505.23735) Eq. 32–33: inner optimizer를 GD에서 **Muon**(momentum + Newton–Schulz 직교화)으로 교체한다. Muon이 객체로 잡혀 있지 않으면 이 교체의 의미와 비용을 판정할 수 없다.
> - [NL] (*Nested Learning: The Illusion of Deep Learning Architecture*, arXiv:2512.24695) §4.2–4.3, App. B: momentum을 gradient stream을 압축하는 associative memory로, **Adam을 element-wise $\ell_2$ objective의 최적 associative memory**로 재구성하고, AdaGrad/Shampoo/Muon을 preconditioned memory로 배열한다. 이 장의 객체 모델이 그 재구성의 원자재다.
> - [Miras] (*It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization*, arXiv:2504.13173; 이 책은 framework 이름 Miras로 통칭한다): sequence model을 (memory architecture, attentional bias, retention gate, memory learning algorithm)의 조합으로 분해한다. 넷째 축 "learning algorithm"의 값들이 바로 이 장의 optimizer 목록이다.
> - [TNT] (*TNT: Improving Chunkwise Training for Test-Time Memorization*, arXiv:2511.07343): chunkwise mini-batch GD를 상속하며, "chunk = inner loop의 mini-batch"라는 이 장 §2.6의 대응 없이는 문제 설정 자체가 성립하지 않는다.
> - [Sleep] (*Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories*, arXiv:2606.03979): wake/sleep lifecycle의 consolidation은 낮은 update frequency로 실행되는 parameter update다 — "update가 일어난다는 것"의 비용 구조를 이 장이 확정한다.

독자의 세계에서 weight는 read-only다. serving 엔진은 checkpoint를 메모리에 올리고, 이후 모든 연산은 그 위의 순수 함수다. 이 장은 그 checkpoint를 만들어낸 공정, 즉 **training**을 하나의 시스템으로 분해한다. 분해의 단위는 이 책 전체가 쓸 단위와 같다: **state가 무엇이고, update 식이 무엇이며, cost가 얼마인가.** 이 세 질문에 답하는 형태로 훈련을 배우면, 이후 장들에서 "optimizer가 곧 sequence layer다"라는 이 라인의 중심 명제가 범주 착오가 아니라 자연스러운 동형사상으로 읽힌다.

## 2.1 파라미터가 움직이는 세계: training loop의 해부

훈련의 목적은 단순하다. parameter 집합 $\Theta$ — 이 책의 표기로 **slow weights** — 를 조금씩 움직여, 데이터에 대한 **loss** $\mathcal{L}(\Theta)$라는 스칼라 하나를 줄이는 것이다. language model이라면 $\mathcal{L}$은 next-token prediction의 cross-entropy를 batch 전체에 대해 평균한 값이다. 핵심은 이것이 **스칼라**라는 사실이다: 수십억 개의 parameter가 만들어내는 모든 행동이 숫자 하나로 요약되고, 훈련은 그 숫자 하나를 내리는 방향으로 전체 parameter를 갱신하는 반복이다.

한 번의 훈련 step은 네 단계로 이루어진다.

1. **forward pass**: batch를 모델에 통과시켜 loss를 계산한다. 독자가 아는 prefill과 커널 구성이 같다 — GEMM과 attention의 연속이다.
2. **loss 계산**: logits에서 스칼라 $\mathcal{L}$로의 reduction. 비용은 무시 가능하다.
3. **backward pass**: $\mathcal{L}$의 모든 parameter에 대한 미분, 즉 gradient를 계산한다(§2.3). FLOPs는 forward의 약 2×다.
4. **optimizer step**: gradient와 optimizer의 내부 state를 읽어 $\Theta$를 갱신한다(§2.5). GEMM이 없는 순수 elementwise pass이며, 철저히 bandwidth-bound다.

이 loop를 독자가 매일 운용하는 serving loop 옆에 놓으면 대응이 즉시 보인다. 표 2-1은 1장의 Rosetta-Stone 사전을 training loop에 특화한 것이다.

표 2-1 — training loop와 serving loop의 대응

| 항목 | serving loop (독자의 세계) | training loop (이 장) |
|---|---|---|
| weights | read-only (mmap된 checkpoint) | read-modify-write (매 step 갱신) |
| 반복 단위 | request / token | step / batch |
| 지속 상태 | KV cache (request 수명) | optimizer state $m_t, h_t$ (훈련 전체 수명) |
| 주 커널 | GEMM + attention (forward만) | forward GEMM + backward GEMM 2종 + elementwise step |
| roofline 위치 | prefill compute-bound, decode bandwidth-bound | fwd/bwd compute-bound, optimizer step bandwidth-bound |
| 성능 지표 | tokens/s, TTFT | tokens/s, MFU |

이 표에서 독자에게 낯선 칸은 두 개뿐이다: backward pass와 optimizer state. 이 장의 나머지는 그 두 칸을 채운다. 그리고 미리 말해 두면, 이 라인의 논문들이 하는 일은 이 표의 왼쪽 열과 오른쪽 열을 **한 프로세스 안에 합치는 것**이다 — decode loop 안에서 작은 네트워크의 weights가 매 token 갱신된다. "weight update 없음"이라는 serving의 불변식은 이 라인에서 폐기되는 불변식이다(→ 1장).

## 2.2 gradient: 스칼라 하나가 모든 parameter에 내리는 지시

**gradient** $\nabla_\Theta \mathcal{L}$는 $\Theta$와 같은 shape의 텐서로, 각 원소는 "그 parameter를 아주 조금 올리면 loss가 얼마나 변하는가"라는 민감도다. 1차 Taylor 근사로 쓰면

$$
\mathcal{L}(\Theta + \Delta\Theta) \;\approx\; \mathcal{L}(\Theta) + \langle \nabla_\Theta \mathcal{L},\; \Delta\Theta \rangle
$$

이고, 내적을 가장 빠르게 음수로 만드는 선택은 $\Delta\Theta = -\eta\, \nabla_\Theta\mathcal{L}$이다. 여기서 $\eta$는 **learning rate** — 한 step의 보폭이다(이 책에서 무첨자 $\eta$는 항상 outer loop의 상수 learning rate이고, 시간 첨자가 붙은 $\eta_t$는 token의 함수인 inner loop의 gate다. 첨자의 유무 자체가 정보다). 이것이 **gradient descent**의 전부다: 민감도 방향의 반대로, 보폭 $\eta$만큼, 반복해서 이동한다.

시스템 관점에서 gradient의 첫 번째 성질은 shape다. gradient는 parameter와 1:1 대응하는 같은 크기의 텐서이므로, gradient를 만들고 소비하는 모든 단계의 메모리 트래픽은 최소 "모델 크기 × 상수 배"다. 7B 모델이면 gradient 텐서 하나가 이미 수십 GB의 이동을 뜻한다. 두 번째 성질은 계산 경로다: 원소가 $10^9$개를 넘는 텐서의 편미분 전부를, 어떻게 forward 한 번 남짓의 비용으로 얻는가. 그것이 backward pass다.

## 2.3 backward pass: 반복되는 VJP, 그리고 $dW = \delta\, x^\top$라는 shape

가장 순진한 방법부터 기각하자. 편미분을 수치적으로 구하려면 parameter 하나를 $\epsilon$만큼 흔들고 forward를 다시 돌려야 한다 — parameter가 $N$개면 forward $N+1$번, $N \sim 10^9$이므로 논외다. **backpropagation**(reverse-mode automatic differentiation)은 forward 한 번 + backward 한 번으로 $N$개 편미분 전부를 얻는다(Rumelhart, Hinton & Williams 1986; 교과서 서술은 Goodfellow, Bengio & Courville 2016, *Deep Learning*, ch. 6–8). 원리는 chain rule을 출력에서 입력 방향으로, 즉 스칼라 loss에서 시작해 거꾸로 적용하는 것이다.

backward의 원자 연산은 **VJP(vector-Jacobian product)**다. 어떤 layer가 $y = f(x)$를 계산했다면, backward는 상류에서 내려온 $\partial\mathcal{L}/\partial y$를 받아 $\partial\mathcal{L}/\partial x = (\partial y/\partial x)^\top\, \partial\mathcal{L}/\partial y$를 하류로 넘긴다. Jacobian $\partial y/\partial x$를 실체화하지 않고 "Jacobian을 벡터에 곱한 결과"만 계산한다는 점이 요체다 — 독자에게 익숙한 어휘로는, 큰 중간 행렬을 materialize하지 않고 fused kernel 한 번으로 통과시키는 것과 같은 감각이다.

구체적으로 $L_{\mathrm{layer}}$층 MLP를 보자($\ell = 1,\dots,L_{\mathrm{layer}}$). layer $\ell$은 pre-activation $z_\ell = W_\ell\, x_{\ell-1}$과 출력 $x_\ell = \sigma(z_\ell)$을 계산한다(이 장에서 $W_\ell$의 첨자는 layer 인덱스다; 시간 첨자를 단 fast weights $W_t$와 혼동하지 말 것 — 여기의 $W_\ell$들은 전부 $\Theta$의 성분이다). **backprop 오차** $\delta_\ell := \partial\mathcal{L}/\partial z_\ell$을 정의하면, chain rule은 두 줄짜리 recursion이 된다:

$$
\delta_\ell = \big(W_{\ell+1}^\top\, \delta_{\ell+1}\big) \odot \sigma'(z_\ell),
\qquad
\frac{\partial \mathcal{L}}{\partial W_\ell} = \delta_\ell\, x_{\ell-1}^\top
\tag{2-1}
$$

첫 식은 오차를 한 층 아래로 전파하고(오차 전파), 둘째 식은 그 층의 weight gradient를 만든다(gradient 생성). 식 (2-1)의 둘째 식을 뚫어지게 볼 필요가 있다. **weight gradient는 "그 층의 입력"과 "그 층의 오차"의 outer product다.** 훈련이 layer의 weight에 가하는 모든 변화는 rank-1 outer product들의 합이라는 뜻이다.

batch로 쌓으면 GEMM shape가 드러난다. 입력을 행으로 쌓은 $X_{\ell-1} \in \mathbb{R}^{B\times d}$, 오차를 행으로 쌓은 $\Delta_\ell \in \mathbb{R}^{B\times d'}$에 대해:

- forward: $Z_\ell = X_{\ell-1} W_\ell^\top$ — $(B\times d)(d\times d')$ GEMM 1개.
- backward 오차 전파: $\partial\mathcal{L}/\partial X_{\ell-1} = \Delta_\ell W_\ell$ — $(B\times d')(d'\times d)$ GEMM 1개.
- backward gradient 생성: $\partial\mathcal{L}/\partial W_\ell = \Delta_\ell^\top X_{\ell-1}$ — $(d'\times B)(B\times d)$ GEMM 1개.

forward가 GEMM 1개일 때 backward는 같은 크기의 GEMM 2개다. 그래서 **backward pass ≈ 2× forward FLOPs**라는 훈련 세계의 상수가 나온다(모델 전체로는 attention 등도 같은 비율을 따르므로 total step ≈ 3× forward). 독자는 이 세 GEMM의 shape를 이미 안다 — 셋째 GEMM $(d'\times B)(B\times d)$는 정확히 rank-$B$ 누적, 즉 "outer product $B$개를 쌓는" 연산이다.

여기서 이 책 전체를 관통할 다리를 하나 놓는다. **$dW = \delta\, x^\top$라는 shape는, KV cache append를 outer-product write $v_t k_t^\top$로 바꾼 것과 같은 GEMM family다.** KV cache는 $(k_t, v_t)$ 쌍을 무손실로 이어 붙이는 memory이고(→ 1장 Rosetta), 뒤에서 만날 matrix memory는 같은 쌍을 $d\times d$ 상태에 rank-1로 눌러 쓰는 memory인데(→ 5장, 6장), 그 write 연산의 GEMM shape가 gradient 생성의 GEMM shape와 동일하다. 우연이 아니다 — 5장에서 보겠지만 delta rule의 write는 문자 그대로 $\ell_2$ regression loss의 gradient이고, 그 loss의 gradient가 식 (2-1)의 특수한 경우이기 때문이다. 훈련 커널과 memory-write 커널이 같은 GEMM이라는 사실은 이 라인 전체의 하드웨어적 행운이며, 9장의 chunkwise 병렬화가 성립하는 물질적 기반이다.

backward에는 FLOPs 외의 비용이 하나 더 있다. 식 (2-1)을 계산하려면 forward 때의 $x_{\ell-1}$과 $z_\ell$이 필요하다 — backward가 끝날 때까지 모든 층의 중간 결과를 붙들고 있어야 한다는 뜻이다. 이 **activation memory**는 batch 크기 × sequence 길이 × hidden 차원 × 깊이에 비례하며, 큰 모델 훈련에서 weight보다 먼저 메모리를 바닥낸다. 표준 대응은 **recomputation**(activation/gradient checkpointing)이다: 일부 층의 activation을 버리고 backward 중에 forward를 다시 계산한다 — FLOPs를 내고 bytes를 사는 거래로, FlashAttention이 softmax 중간값을 버리고 재계산하는 것과 정확히 같은 종류의 거래다. 이 개념이 필요한 이유는 후반부에 있다: 이 라인의 layer는 decode 중에 작은 네트워크의 forward + backward + update를 수행하므로(→ 8장), "backward가 무엇을 저장해야 하는가"가 serving 커널 설계 문제로 넘어온다. 그리고 serving 스택에는 autograd가 없으므로, 그 gradient는 식 (2-1)을 손으로 전개한 fused kernel로 구현된다(→ 10장).

## 2.4 loss surface: 왜 $\eta$ 하나로는 부족한가

$\mathcal{L}(\Theta)$를 $N$차원 지형으로 상상하자. gradient descent는 이 지형을 국소 경사만 보고 내려간다. 지형은 non-convex지만, optimizer 설계를 지배하는 것은 전역 구조가 아니라 국소 **curvature** — gradient가 방향에 따라 얼마나 빨리 변하는가 — 다. 좁고 가파른 골짜기를 생각하면 된다: 골짜기를 가로지르는 방향은 curvature가 커서 조금만 움직여도 gradient가 뒤집히고, 골짜기를 따라가는 방향은 curvature가 작아 한참을 가도 경사가 그대로다. 최적 보폭이 방향마다 다른데 $\eta$는 하나뿐이므로, $\eta$를 가파른 방향에 맞추면 완만한 방향에서 기어가고, 완만한 방향에 맞추면 가파른 방향에서 진동하거나 발산한다. 이것이 ill-conditioning이며, 이 장 후반의 optimizer 동물원은 전부 이 문제에 대한 서로 다른 응답이다: 방향별 진동을 평균으로 상쇄하고(momentum), 좌표별로 보폭을 다시 재고(AdaGrad, Adam), 아예 좌표계를 바꾸고(Shampoo), update의 방향 성분만 남긴다(Muon).

<!-- FIG: ch02/fig-01-loss-landscape -->

minima의 모양도 한 단락만큼은 알아야 한다. 훈련이 도달하는 minimum 주변의 지형은 뾰족할 수도(sharp) 평평할 수도(flat) 있는데, sharp minimum은 parameter의 작은 요동에도 loss가 크게 변하므로 훈련 데이터와 테스트 데이터의 미세한 분포 차이에 취약하다는 실증 보고가 있다(Li et al. 2018, arXiv:1712.09913의 loss landscape 시각화가 표준 참조다). 이 책에서 loss surface 이론은 여기까지만 필요하다 — 수렴 증명은 다루지 않는다. 시스템 독자에게 필요한 요약은 하나다: **step의 품질은 gradient만으로 결정되지 않고, gradient를 어떤 상태(state)와 어떤 기하(metric)로 가공하느냐에 달려 있다.** 그 가공기가 optimizer다.

## 2.5 optimizer = (state, update, cost) 객체

이제 이 장의 중심 정의다. **optimizer는 다음 서명을 갖는 stateful 객체다:**

$$
\mathrm{update}: (\text{state}_t,\; g_t) \;\longmapsto\; (\text{state}_{t+1},\; \Delta\Theta_t)
$$

여기서 $g_t = \nabla_\Theta \mathcal{L}$은 step $t$의 gradient다. 객체마다 세 속성을 기록한다: **state**(step 사이에 살아남는 텐서가 무엇이고 param당 몇 byte인가), **update**(state와 gradient에서 $\Delta\Theta$를 만드는 식), **cost**(step당 FLOPs와 메모리 트래픽). 독자는 attention kernel을 정확히 이 방식으로 사고한다 — KV cache가 state, attention 식이 update, roofline 위치가 cost. 같은 틀을 optimizer에 적용하는 것뿐이다. 표 2-2가 이 절 전체의 요약이고, 각 소절이 한 객체씩 채운다.

표 2-2 — optimizer 객체 요약 (state 크기는 fp32 기준 param당 byte; $\Theta$ 자체는 제외)

| 객체 | state | update 식 (요지) | state 크기 | step 비용 특성 | 이 라인에서의 쓰임 |
|---|---|---|---|---|---|
| SGD | 없음 | $\Delta\Theta=-\eta\, g_t$ | 0 B | elementwise, bandwidth-bound | delta rule = 1-step SGD (→ 5장), TTT (→ 8장) |
| + momentum | $m_t$ | $m_t=\beta m_{t-1}+g_t$; $\Delta\Theta=-\eta\, m_t$ | 4 B | elementwise | [Titans]의 $S_t$ (→ 12장) |
| + weight decay | (없음) | $\Theta\leftarrow(1-\eta\lambda)\Theta+\cdots$ | 0 B | elementwise | retention gate의 원형 (→ 12·13장) |
| AdamW | $m_t, h_t$ | $\Delta\Theta=-\eta\,\hat m_t/(\sqrt{\hat h_t}+\epsilon)$ | 8 B | elementwise | outer loop의 기본값; [NL]의 최적성 정리 (→ 16장) |
| AdaGrad | $h_t$ (누적) | $\Delta\Theta=-\eta\, g_t/(\sqrt{h_t}+\epsilon)$ | 4 B | elementwise | FTRL과의 연결 (→ 3장), [NL §4] |
| Shampoo | $L_t, R_t$ | $\Delta\Theta=-\eta\, L_t^{-1/4} G_t R_t^{-1/4}$ | 행렬당 $m^2{+}n^2$ | GEMM + 주기적 행렬 root | preconditioning의 극점, [NL §4] |
| Muon | $m_t$ | $\Delta\Theta=-\eta\,\mathrm{NS}_\kappa(m_t)$ | 4 B | **GEMM ~10–15개**/행렬 | [Atlas]의 inner optimizer (→ 14장), [NL]의 M3 |

### 2.5.1 SGD: stateless한 기준점

- **state**: 없다.
- **update**: $\Delta\Theta = -\eta\, g_t$ (Robbins & Monro 1951이 확률적 근사로 정식화한 역사적 기준점).
- **cost**: $\Theta$ 읽기 + $g$ 읽기 + $\Theta$ 쓰기. param당 FLOP 2개 남짓의 순수 elementwise pass.

SGD의 "S(stochastic)"는 gradient를 데이터 전체가 아니라 무작위 mini-batch에서 추정한다는 뜻이다(§2.6). stateless라는 성질은 사소해 보이지만 이 라인에서는 기준선 역할을 한다: TTT-Linear(→ 8장)와 delta rule(→ 5장)의 inner loop는 정확히 이 stateless SGD이고, [Titans]의 기여는 inner loop에 state를 **추가**한 것으로 요약된다.

### 2.5.2 momentum: 첫 번째 state, 그리고 linear recurrence라는 정체

- **state**: buffer $m_t$ 하나 ($\Theta$와 같은 shape).
- **update**:

$$
m_t = \beta\, m_{t-1} + g_t, \qquad \Delta\Theta_t = -\eta\, m_t
\tag{2-2}
$$

- **cost**: SGD 대비 read-modify-write 대상 텐서가 하나 늘어난다(트래픽 약 2×). FLOPs는 여전히 param당 상수 개.

$\beta \in [0,1)$은 momentum decay 상수다(보통 0.9). 식 (2-2)를 풀면 $m_t = \sum_{i\le t} \beta^{\,t-i} g_i$ — **momentum buffer는 과거 gradient들의 지수가중합**이다. 고전적 해석은 두 가지다: 물리적으로는 공이 관성을 갖고 골짜기를 구르는 것이고(Polyak 1964; 심층학습 문맥의 재조명은 Sutskever et al. 2013), 통계적으로는 잡음 낀 gradient의 저역 통과 필터다 — §2.4의 가파른 방향 진동은 부호가 번갈아 나타나므로 합산에서 상쇄되고, 완만한 방향의 일관된 성분은 최대 $1/(1-\beta)$배까지 증폭된다.

이 책이 강조하는 세 번째 해석이 있다. **식 (2-2)는 linear recurrence다 — 독자가 아는 linear-RNN state 갱신, RetNet의 상수 decay, scan kernel이 처리하는 점화식과 정확히 같은 대수다.** 이 identity는 장식이 아니라 이 라인의 두 기둥을 떠받친다. 첫째, [Titans]는 momentum buffer를 그대로 inner loop에 이식해 "past surprise" $S_t$로 삼는데(원문 Eq. 10; → 12장), recurrence가 linear이기 때문에 chunk 안에서 associative scan으로 병렬 계산된다(→ 9장) — 독자의 prefix-sum kernel이 여기서 재취업한다. 둘째, [NL §4.2]는 식 (2-2)를 "gradient stream을 key 없이 압축하는 associative memory"로 읽고, GD + momentum 전체를 2-level 구조(안쪽 level이 gradient를 momentum memory로 압축하고, 바깥 level이 그 state를 weights에 적용)로 재구성한다 — 이것이 **momentum-as-memory**이며, 이 장은 객체로서의 사실만 확정하고 정리로서의 지위는 16장이 다룬다. memory로 읽는 순간 capacity 질문이 성립하는데, [NL §4.3]은 $\beta=0.9$일 때 누적 기여의 50% 이상이 최근 gradient 6개에, 99% 이상이 최근 43개에 집중됨을 지적한다 — momentum은 반감기가 짧은 memory이고, 그 짧음이 continual learning에서의 forgetting과 연결된다(→ 11장, 16장).

### 2.5.3 weight decay: $(1-\eta\lambda)$라는 retention의 원형

- **state**: 없다 (기존 객체에 붙는 modifier다).
- **update**: 두 형태를 구분해야 한다. (i) **L2 regularization**: loss에 $\frac{\lambda}{2}\|\Theta\|_2^2$를 더한다 — gradient에 $\lambda\Theta$가 섞여 들어가므로, Adam처럼 gradient를 재척도하는 optimizer에서는 decay 강도까지 함께 왜곡된다. (ii) **decoupled weight decay**: gradient 경로와 무관하게 $\Theta \leftarrow (1-\eta\lambda)\,\Theta - \eta\,(\text{optimizer의 update})$로 직접 곱해 감쇠시킨다. AdamW의 W가 이것이며, 두 형태가 다르다는 지적 자체가 논문 하나다(Loshchilov & Hutter 2019, arXiv:1711.05101).
- **cost**: elementwise 곱 하나. 무시 가능.

decoupled 형태에서 update의 구조가 투명해진다. 매 step $\Theta$에 $(1-\eta\lambda)$가 곱해지므로, update 기여를 $u_i$라 하면

$$
\Theta_t \;=\; \sum_{i \le t} (1-\eta\lambda)^{\,t-i}\; u_i
\tag{2-3}
$$

— **weights 자체가 과거 update들의 지수가중 memory이고, $(1-\eta\lambda)$는 그 memory의 유지 비율**이다. 식 (2-2)와 식 (2-3)을 나란히 놓으면 buffer와 weights가 같은 대수의 두 인스턴스임이 보인다. 독자의 어휘로 번역하면 $(1-\eta\lambda)$는 retention gate의 원형 — 정확히는, cache에서 오래된 항목을 지수적으로 밀어내는 학습된 eviction의 가장 원시적 형태다. 이 인자를 상수에서 token의 함수 $\alpha_t$로 승격시킨 것이 식 (M2)의 retention 항 $\alpha_t W_{t-1}$이고, [Titans]는 이를 memory의 forgetting mechanism으로(→ 12장), [Miras]는 retention gate로 정식화한다(→ 13장; 정의 소유권은 그 장들에 있다). 이 장에서 확정할 사실은 하나다: **weight decay는 이미 언제나 memory 관리 연산이었다.**

### 2.5.4 Adam과 AdamW: 좌표별 보폭의 표준

- **state**: buffer 2개. 1차 moment $m_t$, 2차 moment $h_t$ (관행상 $v_t$로 쓰지만 이 책에서 $v$는 value 전용이다).
- **update** (Kingma & Ba 2015, arXiv:1412.6980):

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t,\qquad
h_t = \beta_2 h_{t-1} + (1-\beta_2)\, g_t \odot g_t,\qquad
\Delta\Theta_t = -\eta\; \frac{\hat m_t}{\sqrt{\hat h_t} + \epsilon}
\tag{2-4}
$$

  ($\hat m_t = m_t/(1-\beta_1^t)$, $\hat h_t = h_t/(1-\beta_2^t)$는 bias correction — EMA가 0으로 초기화된 탓에 초기 step에서 과소평가되는 것을 보정하는 산수이며, 이 책에서 더 깊이 다루지 않는다.)
- **cost**: state 텐서 2개의 read-modify-write. param당 fp32 8 byte의 state, step 트래픽은 SGD의 약 3×. 여전히 GEMM 없는 elementwise pass다.

$m_t$는 식 (2-2)의 EMA 형태이고, 새 성분은 $h_t$다: **좌표별 gradient 제곱의 EMA**, 즉 각 좌표의 최근 gradient 크기 추정치다. update가 $m_t/\sqrt{h_t}$이므로 각 좌표의 보폭은 자기 gradient의 전형적 크기로 나눠 정규화된다 — gradient가 상시 큰 좌표는 억제되고 상시 작은 좌표는 증폭되어, 모든 좌표가 대략 $\pm\eta$ 스케일로 움직인다. 그래서 Adam은 "부호에 가까운(sign-ish) descent + 좌표별 learning rate"로 요약된다. §2.4의 ill-conditioning에 대한 대각 근사 응답인 셈이다. transformer 훈련에서 Adam이 사실상 강제인 이유도 여기 있다: embedding row는 token 빈도에 따라, layer는 깊이에 따라 gradient 스케일이 수십 배씩 다른데, 단일 $\eta$의 SGD는 이 이질성을 감당하지 못한다. AdamW = 식 (2-4) + §2.5.3의 decoupled decay이며, 이 조합이 LLM pretraining의 기본값이다.

이 객체가 이 라인에서 갖는 특별한 지위는 [NL]이 부여한다: element-wise $\ell_2$ regression objective를 놓으면 그 **최적** associative memory의 닫힌 형태가 정확히 Adam의 $m/\sqrt{h}$ 구조로 떨어진다는 재구성이다 [NL App. B]. 즉 Adam은 "gradient stream을 좌표별로 요약하는 memory"의 한 최적점이다. 유도는 16장이 소유한다 — 여기서는 state 2개짜리 객체라는 사실과, 그 두 buffer가 뒤에서 각각 memory의 지위를 얻게 된다는 예고만 접수하면 된다.

### 2.5.5 AdaGrad와 Shampoo: preconditioning이라는 일반화

Adam의 $1/\sqrt{h_t}$를 일반화하면 optimizer 설계의 남은 절반이 보인다. **preconditioning**은 update를 $\Delta\Theta = -\eta\, P^{-1} g_t$로 바꾸는 것 — gradient를 그대로 쓰지 않고 행렬 $P$가 정의하는 좌표계에서 다시 재는 것이다. $P$가 loss의 curvature를 닮을수록 step은 2차(Newton) 방법에 가까워진다. Adam은 $P$를 대각으로 제한한 경우다.

**AdaGrad**(Duchi et al. 2011, JMLR)는 대각 preconditioner의 원형으로, Adam과의 차이는 decay가 없다는 것 하나다: $h_t = h_{t-1} + g_t \odot g_t$로 전 이력을 **누적**하고 $\Delta\Theta = -\eta\, g_t/(\sqrt{h_t}+\epsilon)$로 갱신한다. 누적이므로 보폭은 단조 감소한다 — 유한한 스트림을 한 번 지나가는 online learning의 이론(regret 보장)에서 자연스러운 선택이고, 실제로 AdaGrad는 FTRL 계열 online 알고리즘과 정확히 접속된다(→ 3장). "state를 decay 없이 누적하는가, EMA로 잊는가"라는 이 대비를 기억해 두면 3장의 FTRL vs OGD, 13장의 retention 논의가 같은 축의 반복임이 보인다.

**Shampoo**(Gupta et al. 2018, arXiv:1802.09568)는 대각 제한을 푼다. 행렬 parameter $W \in \mathbb{R}^{m\times n}$의 gradient $G_t$에 대해 좌우 두 통계 $L_t = L_{t-1} + G_t G_t^\top$ ($m\times m$), $R_t = R_{t-1} + G_t^\top G_t$ ($n\times n$)를 유지하고 $\Delta W = -\eta\, L_t^{-1/4}\, G_t\, R_t^{-1/4}$로 갱신한다 — 전체 $mn \times mn$ preconditioner를 Kronecker 곱 구조로 근사한 것이다. cost 프로파일이 질적으로 다르다: state가 param 개수가 아니라 행렬 차원의 제곱($m^2 + n^2$)으로 붙고, update에 GEMM과 행렬 거듭제곱근(주기적으로만 재계산하는 것이 관행)이 들어온다. 이 장에서 Shampoo가 필요한 이유는 실무가 아니라 계보다: [NL §4]는 SGD → Adam → AdaGrad → Shampoo → Muon을 "gradient를 어떤 memory로 압축해 어떤 좌표계를 학습하는가"의 한 스펙트럼으로 배열하며, 그 서열의 비대각 지점이 Shampoo다.

### 2.5.6 Muon: update의 직교화, 그리고 test-time compute로의 예고

- **state**: momentum buffer $m_t$ 하나. Adam의 절반이다.
- **update** (Jordan et al. 2024, blog: kellerjordan.github.io/posts/muon):

$$
m_t = \beta\, m_{t-1} + g_t, \qquad
\Delta W = -\eta\; \mathrm{NS}_\kappa\!\big(m_t / \|m_t\|_F\big)
\tag{2-5}
$$

- **cost**: elementwise가 아니다. $\mathrm{NS}_\kappa$가 parameter shape의 GEMM을 반복당 2–3개, $\kappa=5$ 기준 행렬당 총 10–15개 추가한다.

**Muon**은 hidden layer의 2차원 weight 행렬 전용 optimizer다(embedding·output head 등은 관행상 AdamW로 남긴다 — 같은 모델 안에 두 객체가 공존한다). 아이디어는 한 문장이다: momentum 행렬 $m_t$를 그대로 쓰지 않고, **가장 가까운 semi-orthogonal 행렬로 사영해서 쓴다.** SVD로 $m_t = U\Sigma V^\top$라 쓰면 그 사영은 $UV^\top$ — 특이값을 전부 1로 갈아 끼운 행렬로, matrix sign 계열 연산이라 **msign**으로도 불린다. 왜 이것이 좋은가: gradient/momentum 행렬은 소수의 지배적 방향(큰 특이값)에 에너지가 몰려 있어, 그대로 적용하면 update가 사실상 저rank가 된다. 직교화는 지배적 방향을 누르고 희귀하지만 유효한 방향을 살려 **모든 방향에 고른 크기로** 쓰게 한다 — Adam이 좌표별로 하던 스케일 평준화를 특이값 스펙트럼에 대해 하는 것이며, spectral norm 기하에서의 steepest descent이자 2차 정보의 근사로 읽힌다.

SVD는 비싸므로 실제 구현은 **Newton–Schulz iteration** $\mathrm{NS}_\kappa$를 쓴다: 홀수 행렬 다항식 $X \leftarrow aX + bX(X^\top X) + cX(X^\top X)^2$을 $\kappa$회 반복하면 $X$의 특이벡터는 보존되고 특이값만 1로 수렴한다. Muon은 $\kappa=5$와 튜닝된 계수 $(a,b,c)=(3.4445,\,-4.7750,\,2.0315)$를 쓴다(Jordan et al. 2024). 시스템 독자에게 이 구현 선택이 핵심이다: **"직교화 = parameter shape의 GEMM 몇 개"**이므로, Muon은 optimizer step을 bandwidth-bound elementwise pass에서 tensor core가 도는 연산으로 바꾸면서도 state는 buffer 하나로 유지한다. 대규모 LLM pretraining에서의 실증은 Liu et al. 2025 (arXiv:2502.16982)가 보고한다.

이 객체가 이 책에 등장하는 진짜 이유는 outer loop가 아니다. [Atlas]는 Muon을 **inner loop의 optimizer로** 이식한다: memory를 GD 대신 식 (2-5)로 갱신한다(원문 Eq. 32–33; 통일 표기로는 $S_t$에 $\mathrm{NS}_\kappa$를 적용해 $W_t = \alpha_t W_{t-1} + \eta_t\,\mathrm{NS}_\kappa(S_t)$ 꼴 — → 14장). 그 순간 $\kappa$는 decode 중 매 chunk마다 지불하는 추가 GEMM 개수가 되고, [Atlas §5]는 이 반복 횟수를 명시적으로 "internal test-time compute parameter"라고 부른다 — 반복을 늘리면 더 나은 memorization을 살 수 있다는 것이다.

[NL]은 한 걸음 더 나가 $\mathrm{NS}_\kappa$ 반복 자체를 "momentum update 안의 내부 최적화 level"로 읽고(→ 16장), Adam + Muon + 다중 주기 momentum을 결합한 M3 optimizer를 제안한다 [NL Alg. 1]. 이 장의 어휘로 요약하면: Muon은 (state 1개, GEMM 위주 update, 낮은 state cost / 높은 compute cost)의 객체이고, 이 라인은 그 객체를 layer 안으로 옮겨 심는다.

## 2.6 batch 축 vs sequence 축 — 이 주제 최대의 혼동 지점

지금까지 gradient는 "batch에서 계산된 것"이었다. **mini-batch**는 데이터셋에서 독립적으로 뽑은 $B$개 샘플이고, batch gradient는 per-sample gradient의 평균이다. $B$를 키우는 이유는 두 가지다: 통계적으로는 독립 샘플 평균이므로 gradient 추정의 분산이 $1/B$로 줄고, 시스템적으로는 §2.3의 세 GEMM에서 $B$가 클수록 GEMM이 두꺼워져 MFU가 오른다 — batch는 처음부터 품질과 하드웨어 효율을 동시에 사는 knob이었다. 반대 극단인 $B=1$의 per-sample(online) update는 같은 대수에 잡음과 skinny GEMM을 얹은 것이다.

이제 이 책에서 가장 중요한 경고를 볼드로 박는다. **이 라인의 논문들에서, 훈련의 batch 축이 하던 역할을 sequence 축이 대신한다. inner loop의 "mini-batch"는 독립 샘플 $B$개가 아니라 연속한 token $C$개 — 즉 chunk다.** [Titans]가 "mini-batch gradient descent"라고 쓸 때 그 batch는 서로 이웃한 token들이고, [TNT]의 chunk 크기 실험은 전부 이 의미의 batch 크기 실험이다. 같은 단어가 두 세계에서 다른 축을 가리키므로, 이 대응을 놓치면 논문의 모든 문장이 한 축씩 어긋나게 읽힌다.

역할은 같되 성질이 다르고, 그 차이가 뒤 장들의 논점을 만든다. 첫째, batch 샘플은 독립이지만 sequence token은 상관되어 있고 인과 순서를 갖는다 — 순서가 있으므로 "chunk 안의 gradient를 어느 시점의 state에서 평가하는가"라는, batch 세계에는 없던 질문이 생긴다. 그 질문의 답(chunk 시작 state에 고정하는 stale-snapshot 근사)이 chunk 크기 $C$를 단순한 tiling 파라미터가 아니라 **계산되는 함수 자체를 바꾸는 semantic hyperparameter**로 만든다(식 (M4); 명제의 소유는 9장). FlashAttention의 tile 크기는 bit-exact한 성능 knob이지만 $C$는 아니다 — 독자의 tiling 직관을 그대로 이식하면 안 되는 지점이다. 둘째, 그래도 시스템적 역할은 동일하게 작동한다: $C$가 클수록 inner loop의 GEMM이 두꺼워지고 MFU가 오른다. 품질이 원하는 $C$(작게)와 하드웨어가 원하는 $C$(크게)의 긴장이 [TNT]의 존재 이유다(→ 9장, 15장). 독자의 어휘로 최종 번역하면: **prefill은 큰 chunk의 병렬 write, decode는 $C=1$의 online write다** — 이 문장이 성립하도록 만드는 장치 일체가 9장의 내용이다.

## 2.7 Worked micro-example: $2\times2$ memory에 손으로 돌리는 네 개의 optimizer

이 절에서는 위의 객체들을 전부 한 자리에서, 암산 가능한 숫자로 돌려 본다. 대상은 이 라인의 최소 세팅이다: 상태 $W \in \mathbb{R}^{2\times 2}$ (0으로 초기화), 저장할 association은 key $k = (1, 0)^\top$, value $v = (1, 2)^\top$ 하나, inner loss는 이 책의 기본형 $\ell(W; k, v) = \|Wk - v\|_2^2$이다. 식 (2-1)이 곧바로 gradient를 준다: 오차 $e = Wk - v$에 대해 $\delta = 2e$이고 gradient는 outer product

$$
\nabla_W \ell = 2\,(Wk - v)\,k^\top .
$$

**(a) SGD, $\eta = 0.25$.**
step 1: $W_0 k = (0,0)^\top$이므로 $e_1 = (-1,-2)^\top$, $g_1 = 2 e_1 k^\top = \begin{bmatrix} -2 & 0 \\ -4 & 0 \end{bmatrix}$. 갱신: $W_1 = W_0 - 0.25\, g_1 = \begin{bmatrix} 0.5 & 0 \\ 1 & 0 \end{bmatrix}$. 읽기: $W_1 k = (0.5, 1)^\top$ — 목표의 절반이다.
step 2: $e_2 = (-0.5, -1)^\top$, $g_2 = \begin{bmatrix} -1 & 0 \\ -2 & 0 \end{bmatrix}$, $W_2 = \begin{bmatrix} 0.75 & 0 \\ 1.5 & 0 \end{bmatrix}$, 읽기 $(0.75, 1.5)^\top$. 매 step 오차가 정확히 반감한다($\eta \cdot 2\,k^\top k = 0.5$이므로). SGD는 목표에 기하급수적으로 다가가되 유한 step에서는 도달하지 못한다. 그리고 둘째 열이 끝까지 0임을 눈여겨보라 — gradient가 $k^\top$ 방향으로만 뻗는 outer product이기 때문이며, "write는 key가 가리키는 subspace만 건드린다"는 5장 crosstalk 논의의 씨앗이다.

**(b) momentum, $\beta = 0.5$, $\eta = 0.25$.**
step 1: $m_1 = g_1$이므로 SGD와 동일하게 $W_1 = \begin{bmatrix} 0.5 & 0 \\ 1 & 0 \end{bmatrix}$.
step 2: $m_2 = 0.5\, m_1 + g_2 = \begin{bmatrix} -1 & 0 \\ -2 & 0 \end{bmatrix} + \begin{bmatrix} -1 & 0 \\ -2 & 0 \end{bmatrix} = \begin{bmatrix} -2 & 0 \\ -4 & 0 \end{bmatrix}$, $W_2 = W_1 - 0.25\, m_2 = \begin{bmatrix} 1 & 0 \\ 2 & 0 \end{bmatrix}$. 읽기: $W_2 k = (1, 2)^\top = v$, **정확히 도달**했다. 과거 gradient의 잔향($0.5\,m_1$)이 현재 step에 실려 보폭을 늘린 결과다. 이 세팅에서는 운 좋게 정확히 착지했지만 일반적으로는 같은 관성이 과속(overshoot)도 만든다 — $\beta$와 $\eta$의 균형 문제다. [Titans]의 $S_t$가 하는 일이 대수적으로 정확히 이것이다(부호 관행만 식 (M2)를 따른다).

**(c) weight decay, $(1-\eta\lambda) = 0.9$.**
(b)의 $W_2$에서 새 token이 없다고 하자(gradient 0). decoupled decay는 그래도 매 step 곱해진다: $W_3 = 0.9\, W_2 = \begin{bmatrix} 0.9 & 0 \\ 1.8 & 0 \end{bmatrix}$, 읽기 $(0.9, 1.8)^\top$. 저장된 association이 접근 없이도 step마다 10%씩 증발한다 — TTL이 있는 cache entry처럼. 이 감쇠 인자를 token마다 계산되는 $\alpha_t$로 바꾸면 식 (M2)의 retention 항이 된다.

**(d) Adam의 좌표별 평준화.**
(a)의 $g_1$에서 0이 아닌 두 좌표를 보자: $(1,1)$ 원소는 $-2$, $(2,1)$ 원소는 $-4$로 크기가 2배 다르다. $t=1$에서 식 (2-4)를 bias correction까지 적용하면 $\hat m_1 = g_1$, $\hat h_1 = g_1 \odot g_1$이므로 update는 $-\eta\, g_1/\sqrt{g_1 \odot g_1} = -\eta\,\mathrm{sign}(g_1)$ — 두 좌표 모두 크기가 정확히 $\eta$다. gradient 크기가 2배 차이 나도 첫 step의 보폭은 동일하다: "sign-ish descent"를 이보다 짧게 보여 주는 계산은 없다. ($t\ge2$부터는 EMA가 이력을 반영해 순수 sign에서 벗어난다.)

**(e) Newton–Schulz의 특이값 평준화.**
Muon의 $\mathrm{NS}_\kappa$가 하는 일을 보기 위해, 특이값이 $(1.2,\ 0.4)$인 momentum 행렬을 생각하자(예: $m = \mathrm{diag}(1.2, 0.4)$; 대각 행렬이면 NS 반복이 특이값 각각에 대한 스칼라 다항식이 된다). 손계산을 위해 고전적 3차 반복 $x \leftarrow 1.5x - 0.5x^3$을 쓴다(실전의 Muon은 §2.5.6의 튜닝된 5차 다항식을 쓰지만 원리는 같다):

| 반복 | $\sigma_1$ | $\sigma_2$ | 비율 |
|---|---|---|---|
| 0 | 1.2 | 0.4 | 3.0× |
| 1 | $1.8 - 0.864 = 0.936$ | $0.6 - 0.032 = 0.568$ | 1.65× |
| 2 | $1.404 - 0.410 = 0.994$ | $0.852 - 0.092 = 0.760$ | 1.31× |

두 번의 반복만에 3배 차이가 1.3배로 줄었고, 반복을 계속하면 둘 다 1로 수렴한다. 특이벡터(방향)는 건드리지 않고 특이값(방향별 크기)만 1로 밀어붙인다 — (d)의 Adam이 좌표축에서 하던 평준화를 임의의 직교 방향에 대해 하는 것이다. 반복 한 번의 비용이 $2\times2$에서는 암산이지만 $d\times d$에서는 GEMM 2–3개라는 것, 그것이 Muon cost 모델의 전부다.

다섯 계산의 교훈을 한 줄로 모으면: **(a)가 delta rule이고, (a)+(b)+(c)가 [Titans]의 memory update이고, (e)를 (b)에 끼우면 [Atlas]의 memory update다.** Part II에서 만날 update rule들은 이 절의 손계산에 gate와 chunk를 입힌 것 이상이 아니다.

## 2.8 Systems bridge: optimizer 한 step의 비용 회계

이 절은 이 장의 객체들을 독자의 자원 회계 감각에 정착시킨다. 기준 모델로 7B-parameter dense transformer를 놓자.

**state 크기 — optimizer state는 모델보다 크다.** serving에서 7B 모델은 bf16으로 약 14 GB다. mixed-precision 훈련의 표준 레시피에서는 param당 bf16 weight 2 B + fp32 master weight 4 B + Adam $m_t$ 4 B + $h_t$ 4 B = **14 B/param**, 약 98 GB가 gradient와 activation을 세기도 전에 상주한다. "훈련은 왜 서빙보다 몇 배의 HBM을 먹는가"의 답의 절반이 optimizer state다. 표 2-3이 객체별 회계다.

표 2-3 — optimizer 객체별 비용 회계 (param당; state는 fp32 관행 기준)

| 객체 | state bytes | step 추가 트래픽 (state read+write) | step 연산 성격 | roofline 위치 |
|---|---|---|---|---|
| SGD | 0 | 0 | elementwise | bandwidth-bound |
| momentum | 4 B | 8 B | elementwise | bandwidth-bound |
| AdamW | 8 B | 16 B | elementwise | bandwidth-bound |
| AdaGrad | 4 B | 8 B | elementwise | bandwidth-bound |
| Shampoo | 행렬당 $m^2{+}n^2$ | 통계 갱신 GEMM | GEMM + 주기적 root | 혼합 |
| Muon | 4 B | 8 B | **GEMM 10–15개/행렬** | compute 쪽으로 이동 |

**step의 roofline 위치.** AdamW step은 param당 read가 weight·gradient·$m$·$h$, write가 weight·$m$·$h$ — 대략 24–28 B를 움직이며 FLOP은 십수 개다. arithmetic intensity가 1 FLOP/byte 언저리인, 독자의 분류로는 decode와 같은 극단적 bandwidth-bound 커널이다. 훈련 step 전체의 시간 구조가 이제 읽힌다: forward/backward는 prefill을 닮았고(두꺼운 GEMM, compute-bound), optimizer step은 decode를 닮았다(거대한 state의 순회, bandwidth-bound). Muon은 이 그림에서 유일하게 step을 GEMM으로 바꾸는 객체다 — state는 Adam의 절반이면서 연산은 tensor core로 옮긴다. "state를 덜 쓰고 compute를 더 쓴다"는 이 거래는 14장에서 inner loop의 test-time compute 논의로 반복된다.

**optimizer state의 정체 — 훈련의 register file.** momentum buffer와 second moment는 매 step 읽고 쓰는, 훈련 job 수명 동안 상주하는 accumulator다. Rosetta 사전의 대응(→ 1장)을 이 장의 결과로 뒷받침하면: KV cache가 request 수명의 state이듯 optimizer state는 training-run 수명의 state이고, 차이는 수명과 소유자뿐이다. 이 라인이 하는 일은 이 state의 소유자를 한 번 더 바꾸는 것이다 — momentum buffer $S_t$가 **per-session state**가 되어 decode loop 안으로 들어온다. 그 순간 "optimizer state 회계"는 훈련 클러스터의 문제가 아니라 serving의 session cache sizing 문제가 된다(→ 10장, Part III).

**두 GEMM의 동일성 — 이 장에서 가져갈 단 하나의 shape.** 식 (2-1)의 $dW = \delta\, x^\top$ (batched: $(d'\times B)(B\times d)$)와, 뒤 장들의 memory write $v_t k_t^\top$ (chunked: $(d_v\times C)(C\times d_k)$)는 같은 GEMM이다. 훈련의 gradient 생성 커널과 fast-weight memory의 write 커널이 shape 수준에서 동일하므로, 훈련용으로 존재하는 모든 커널 기술 — tiling, tensor-core 활용, rank-누적 — 이 이 라인의 inference에 이식 가능하다. 9장의 chunkwise 알고리즘들은 이 동일성의 체계적 착취다.

## 요약

- 훈련은 forward(≈ prefill), backward(≈ forward의 2×, GEMM 2종), optimizer step(elementwise, bandwidth-bound)의 loop이며, loss라는 스칼라 하나가 전체 parameter의 update를 지휘한다.
- backward pass는 VJP의 연쇄이고, weight gradient는 항상 "층 입력 × 층 오차"의 outer product $dW = \delta\, x^\top$다 — 이 shape는 memory write $v k^\top$와 동일한 GEMM이다.
- optimizer는 `update(state, g) → (state′, ΔΘ)` 서명의 객체다: SGD는 stateless, momentum은 buffer 1개, AdamW는 2개, Shampoo는 행렬 통계, Muon은 buffer 1개 + GEMM 연산.
- momentum 식 $m_t = \beta m_{t-1} + g_t$는 linear recurrence — linear-RNN state와 같은 대수이고, [Titans]의 $S_t$와 [NL]의 momentum-as-memory가 이 identity 위에 선다.
- decoupled weight decay의 $(1-\eta\lambda)$는 상수 retention 인자이며, 이를 token의 함수로 승격한 것이 식 (M2)의 retention 항 $\alpha_t W_{t-1}$이다.
- Adam은 좌표별 gradient 스케일로 보폭을 정규화하는 sign-ish descent이고, [NL App. B]는 이를 element-wise $\ell_2$ objective의 최적 associative memory로 재구성한다.
- Muon = momentum + Newton–Schulz 직교화($\mathrm{NS}_\kappa$, $\kappa=5$): update의 특이값을 평준화하며, 비용은 행렬당 GEMM 10–15개다. [Atlas]가 이를 inner optimizer로 이식한다.
- 이 라인에서 sequence 축이 batch 축의 역할을 맡는다: chunk = inner loop의 mini-batch. 단 token은 독립이 아니므로 $C$는 semantic hyperparameter가 된다(→ 9장).

## 자가 점검 체크리스트

- [ ] backward pass의 GEMM 2개(오차 전파, gradient 생성)의 shape를 쓰고, 왜 backward가 forward의 약 2× FLOPs인지 설명할 수 있다.
- [ ] SGD / momentum / AdamW / Muon의 state 구성과 update 식을 암기가 아니라 재구성으로 쓸 수 있고, param당 state byte를 말할 수 있다.
- [ ] momentum recurrence가 linear-RNN state 갱신과 같은 대수임을 보이고, 그것이 [Titans]의 $S_t$와 어떻게 연결되는지 말할 수 있다.
- [ ] L2 regularization과 decoupled weight decay의 차이를 설명하고, $(1-\eta\lambda)$ 인자가 어떤 gate의 원형인지 말할 수 있다.
- [ ] §2.7의 (d)와 (e)를 재현해, Adam이 좌표별로·$\mathrm{NS}_\kappa$가 특이값별로 각각 무엇을 평준화하는지(그리고 무엇을 보존하는지) 손계산으로 보일 수 있다.
- [ ] optimizer state를 inference 어휘로 옮길 수 있다: 어떤 의미에서 "훈련의 register file"이고, optimizer step이 roofline의 어디에 앉으며, $dW = \delta x^\top$가 어떤 serving 커널과 같은 shape인지 말할 수 있다.

## 다음 장으로

이 장의 optimizer들은 "고정된 데이터셋 위를 여러 epoch 도는" 세계에서 태어났다. 그러나 이 라인의 optimizer는 다른 세계에서 산다: 데이터가 한 번만, 순서대로, 되돌릴 수 없이 흘러가는 token stream 위에서 매 step 갱신해야 한다. 그 세계에서 "잘 배우고 있다"를 재는 자는 loss 수렴이 아니라 **regret** — 사후 최적의 고정 상태 대비 얼마나 뒤처졌는가 — 이고, update rule을 설계하는 문법은 FTRL이다. 3장은 이 online learning의 언어를 장착한다. [Miras]가 sequence model 전체를 그 언어로 다시 쓴 논문이므로, 3장 없이는 13장이 읽히지 않는다.
