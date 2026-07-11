# ch13. Miras: It's All Connected — sequence model 설계 공간의 통일 이론

## 13.1 Bridge-in: Titans가 남긴 질문 — "왜 하필 그 선택인가"

12장의 [Titans] (*Titans: Learning to Memorize at Test Time*, arXiv:2501.00663)는 하나의 point design을 출하했다. deep MLP memory의 weights를 token마다 gradient descent로 갱신하되, data-dependent momentum("past surprise")과 weight decay("forgetting")를 붙이고, inner objective는 $\ell_2$ associative regression으로 고정하며, attention과는 MAC/MAG/MAL 세 방식으로 합성한다(→ 12장). 그리고 Titans 원문 스스로가 자기 설계의 일반성을 암시했다: [Titans App. C]는 Gated DeltaNet(이하 GDN), Longhorn, RWKV-7, TTT가 모두 Titans update 식의 특수 사례임을 보인다. 그런데 바로 그 지점에서 논문이 끝난다 — 특수 사례가 있다면 일반형이 있을 텐데, 그 설계 공간이 무엇인지, 각 축에서 Titans의 선택이 왜 좋은지는 답하지 않았다.

구체적으로 세 가지 질문이 열려 있었다. 첫째, **왜 $\ell_2$인가?** inner loss $\ell(W;k_t,v_t)=\|\mathcal{M}(k_t;W)-v_t\|_2^2$는 관행이지 논증이 아니다. 둘째, **왜 momentum + weight decay인가?** inner optimizer로 SGD with momentum을 쓴 것 역시 outer 세계의 관행 이식이며, 다른 optimizer가 더 나은지는 묻지 않았다. 셋째, **"forgetting"이 옳은 추상인가?** Titans는 $\alpha_t$를 forgetting mechanism이라 불렀지만, gate가 실제로 하는 일이 무엇인지에 대한 이론은 없었다. Titans는 "$\ell_2$ regression 너머의 inner objective"와 "더 나은 inner optimizer"를 명시적인 open problem으로 남겼다.

이 장의 논문 [Miras] (*It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization*, arXiv:2504.13173, Behrouz, Razaviyayn, Zhong, Mirrokni, Google Research, 2025-04)는 이 질문에 새 아키텍처가 아니라 **분류학**으로 답한다. 논문의 실제 제목은 *It's All Connected*지만, 이 책은 논문이 제안한 framework의 이름인 **Miras**로 통칭한다. Miras는 "유산(Legacy)"을 뜻하는 단어로, 후속 연구가 밟을 설계 단계를 물려준다는 의도의 명명이다 [Miras 각주 1].

## 13.2 문제의식과 논문의 핵심 주장

Miras가 정리한 지형은 이렇다. Transformer는 in-context learning과 스케일링 능력으로 SOTA지만 quadratic 시간과 선형 증가하는 KV cache가 long-context 적용을 막는다. efficient recurrent 대안들은 context를 고정 크기 state로 압축하며, 이 계열은 세 개의 경험적 전선에서 개선되어 왔다 [Miras §1]: (1) learning rule — Hebbian rule에서 delta rule로(→ 5장, 6장); (2) (당시 명칭) forget gate — LSTM에서 Mamba-2를 거쳐 Titans의 gate로; (3) memory 구조 — RetNet류의 vector memory에서 Titans/TTT의 deep memory로. 그러나 이 세 전선을 관통하는 설계 프레임은 없었다. 기존의 통일 시도들 — Mamba-2의 SSD framework(→ 7장), Longhorn의 online-learning 관점, Sun et al. 2024의 TTT regression 관점(→ 8장), 그리고 동시기 작업인 Wang et al. 2025의 test-time regression framework — 은 각각 일부만 포괄하거나, forget gate(당시 명칭)를 표현하지 못하거나, 모든 모델을 단일 regression objective에 강제로 끼워 넣어 Hebbian과 delta의 실질적 차이를 지워버린다 [Miras §2]. 특히 Wang et al.에 대해 Miras는 부록에서, RetNet/Mamba를 regression solver로 근사해야만 하고 HGRN2·Moneta·Yaad·Memora는 아예 표현할 수 없다고 반박한다.

이 지형 위에서 Miras의 주장은 두 개의 관찰과 하나의 프레임으로 요약된다.

**관찰 1 — 모두가 같은 objective를 쓰고 있었다.** 거의 모든 기존 sequence model은 associative memory이며, 그 내부 objective는 단 두 종류 — dot-product similarity 아니면 $\ell_2$ regression — 뿐이다 [Miras Table 1]. 이 내부 objective를 논문은 **attentional bias**라 명명한다: 인지과학에서 특정 자극을 우선시하는 성향을 가리키는 용어를 빌려, "이 memory가 무엇을 우선해 기억하는가"를 정의하는 loss로 정식화한 것이다.

**관찰 2 — forgetting은 존재하지 않는다. retention이 있을 뿐이다.** 기존 forget gate들은 전부 $\ell_2$류 regularization의 특수형이며, gate가 하는 일은 "지우기"가 아니라 "새 association을 배우는 것과 이전 state에 머무르는 것 사이의 trade-off"다. 모델은 memory를 소거하는 것이 아니라 유지하지 않기로 결정할 뿐이며, 이는 뇌가 기억을 지우는 것이 아니라 retrieval failure로 접근 불가능해진다는 신경과학의 관점과 일치한다 [Miras Remark 3]. 그래서 논문은 forget gate를 **retention gate**로 개명한다 — 이 책이 이 용어를 공식 명칭으로 채택한 근거가 이 절이다(역사적 별칭 "forget gate"는 이 문장 한 번으로 병기를 마친다).

**프레임 — Miras framework.** 위 관찰은 sequence model 설계를 네 개의 독립적인 축으로 분해한다: (i) memory architecture, (ii) attentional bias, (iii) retention gate, (iv) memory learning algorithm. 기존 모델 전부가 이 4-tuple의 한 점이고, 지금까지의 발전은 이 네 축을 설계 축으로 인식하지 못한 채 주로 축 (i)(architecture)와 축 (iii)(retention)를 움직였다 — 축 (ii)는 dot-product/$\ell_2$ 두 선택지 사이에서만, 축 (iv)는 Longhorn·DeltaProduct 정도로만 제한적으로 탐사했다. 논문은 축 (ii)와 (iii)에 새 선택지를 채워 넣은 세 모델 — **Moneta**, **Yaad**, **Memora** — 를 출하해 프레임의 생산성을 실증하고, 1.3B/100B tokens 스케일에서 attention-free 순수 recurrent 모델이 attention hybrid까지 이기는 결과를 headline으로 내세운다 [Miras §6.1].

이 장의 재구성 관점에서 말하면: Titans가 "optimizer를 sequence layer로 만들 수 있다"는 존재 증명이었다면, Miras는 그 move가 열어놓은 공간의 좌표계이며, 이후 장들의 논문이 전부 이 좌표계 위에서 자신의 위치를 서술한다.

## 13.3 Core mechanism (통일 표기)

### 13.3.1 Associative memory의 재정식화와 attentional bias

Miras의 출발점은 5장의 고전적 associative memory를 optimization 문제로 다시 정의하는 것이다 — Hopfield energy와 capacity의 관점 대신, **무엇을 최소화하는 객체인가**가 정의의 전부다.

**Definition (associative memory와 attentional bias)** [Miras Def. 3.1]. key 집합 $\mathcal{K}\subseteq\mathbb{R}^{d_k}$와 value 집합 $\mathcal{V}\subseteq\mathbb{R}^{d_v}$가 주어질 때, associative memory는 operator $\mathcal{M}:\mathcal{K}\to\mathcal{V}$이고, 그 mapping의 학습은 objective $L$ — **attentional bias** — 의 최소화다:

$$
\mathcal{M}^\star \;=\; \arg\min_{\mathcal{M}}\; L\big(\mathcal{M}(K);\,V\big).
\tag{13-1}
$$

여기서 $k_t,v_t$는 attention에서와 정확히 같은 방식으로 입력 token의 linear projection으로 만들어진다($k_t=W_Kx_t$, $v_t=W_Vx_t$). $\mathcal{M}^\star$는 argmin의 최적해를 가리키는 전용 기호이며(§1.2 예약), Titans의 read-only pass 표기와 무관하다. memory를 parameter $W$로 매개화하면 최적화는 $W$ 위에서 수행되고, 과거 데이터의 retention을 제어하는 regularizer $R(W)$를 추가할 수 있다 [Miras Remark 1]. 이후 per-token 축약형으로 $\ell(W_{t-1};k_t,v_t):=L(\mathcal{M}(k_t;W_{t-1}),v_t)$를 쓴다.

결정적인 것은 [Miras Remark 2]다: 식 (13-1)의 학습은 **meta-learning / bilevel 문제**다. attentional bias는 **inner loop에서** — 즉 test time에, token마다 — 최적화되고, 나머지 모든 parameter(projection, convolution, gate 생성층)는 **outer loop에서** 통상의 pre-training으로 최적화된다(→ 4장). 이 한 문단이 이 라인 전체의 두-loop 구조를 논문 안에서 처음으로 명시적 정의로 못박은 지점이며, 이 장 §13.4에서 전면 전개한다.

식 (13-1)을 online으로 푸는 가장 단순한 방법은 새 $(k_t,v_t)$ 쌍이 도착할 때마다 gradient descent 한 걸음을 딛는 것이다:

$$
W_t \;=\; W_{t-1} \;-\; \eta_t\,\nabla_W\,\ell(W_{t-1};k_t,v_t),
$$

즉 표준형 (M1) 그대로다 [Miras Eq. 5]. Titans가 momentary surprise라 불렀던 양(→ 12장)은 이 관점에서는 그냥 임의의 attentional bias에 대한 online GD의 gradient다 — surprise의 일반화가 여기서 완료된다.

### 13.3.2 두 개의 optimization 관점: FTRL과 Learning–Retaining

Miras의 이론적 척추는 (M1)을 해석하는 두 관점이다. 이 두 관점이 있어야 "retention gate가 정확히 무엇인가"가 정의되고, 새 gate의 유도가 기계적으로 가능해진다.

**관점 1 — FTRL viewpoint.** (M1)은 loss 열 $\ell(W;k_1,v_1),\ell(W;k_2,v_2),\dots$ 위의 online gradient descent 한 걸음이고, OGD는 FTRL(→ 3장)의 특수 사례다. $W_0=0$일 때 (M1)은 다음과 동치다 [Miras Eq. 7]:

$$
W_t \;=\; \arg\min_{W\in\mathcal{W}} \underbrace{\sum_{i=1}^{t}\hat\ell_i(W;k_i,v_i)}_{\text{attentional bias}} \;+\; \underbrace{\frac{1}{\eta_t}\,R_t(W)}_{\text{memory stability}},
\tag{13-2}
$$

고전형에서는 $\hat\ell_i(W)=\langle W-W_{i-1},\,\nabla_W\ell(W_{i-1};k_i,v_i)\rangle$ (각 시점 loss의 국소 선형화)이고 $R_t(W)=\tfrac12\|W\|_2^2$다. 첫 항은 **모든 과거 token을** 잘 기억하는가를 재고, 둘째 항은 memory의 크기를 벌한다. 주의: (M1)과 FTRL의 **정확한** 동치는 $\eta_t=\eta$ 상수일 때다([Miras Eq. 7]도 상수 $\eta$와 $\tfrac{1}{2\eta}\|W\|^2$를 쓴다). data-dependent $\eta_t$에서는 $\frac{1}{\eta_t}R_t$를 단일 계수로 앞에 둔 (13-2)는 variable-step OGD($W_t=-\sum_i\eta_i g_i$)를 정확히 재현하지 못하고($-\eta_t\sum_i g_i$가 된다) 일반화된 FTRL 관점으로만 성립한다 — 정확히 맞추려면 $\eta_i$ 가중을 선형화 loss 항 안에 넣어야 한다. $\hat\ell_i$와 $R_t$를 일반화하면 mirror descent류 알고리즘이 나온다 — 이 자리가 뒤에서 dual-accumulator형 update(Moneta)가 태어나는 자리다.

**관점 2 — Learning–Retaining viewpoint.** 같은 (M1)을 "최신 쌍을 배우되 이전 state 근처에 머무르기"로 읽을 수도 있다:

$$
W_t \;=\; \arg\min_{W\in\mathcal{W}} \underbrace{\tilde\ell_t(W;k_t,v_t)}_{\text{learning}} \;+\; \underbrace{\mathrm{Ret}_t(W,\,W_{t-1})}_{\text{retention}},
\tag{13-3}
$$

여기서 $\tilde\ell_t$는 $\ell$의 근사(선형화 등)이고, retention 항은 **local과 global로 분해**된다 [Miras §3.3]:

$$
\mathrm{Ret}_t(W,W_{t-1}) \;=\; \frac{1}{\eta_t}\,D_t(W,\,W_{t-1}) \;+\; \frac{1}{\alpha_t}\,G_t(W).
\tag{13-4}
$$

**local retention** $D_t$는 $W_{t-1}$로부터의 이탈을 벌하는 premetric으로, 이미 배운 것을 유지하는 힘이다. 그 계수 $\eta_t$를 논문은 **meta in-context learning rate**라 부른다: $\eta_t$가 크면 새것을 많이 배우고 옛것을 많이 놓아준다. **global retention** $G_t$는 state 자체의 크기를 제한한다 — 고전적 gate가 사는 곳이 바로 이 항이다. 두 관점의 관계는 다음이 정리한다.

**Proposition** [Miras Prop. 3.2, 증명은 App. B]. $\eta_t=\eta$ 상수, $\mathcal{W}=\mathbb{R}^d$이고, $h_t(W):=\sum_{i=1}^{t-1}\hat\ell_i(W)+\tfrac1\eta R(W)$가 strictly convex라 하자. $\mathrm{Ret}_t$를 $h_t$의 Bregman divergence로 놓으면 Learning–Retaining update는 FTRL update와 정확히 일치한다.

즉 Learning–Retaining이 더 일반적인 렌즈이고, 논문의 모든 유도가 이 관점에서 나온다. 단, 정직하게 짚어야 한다:

> **[평가]** Prop. 3.2의 가정 — 상수 $\eta$, 무제약 $\mathcal{W}$, 누적 objective의 strict convexity — 는 실제 출하된 모델에서 전부 깨진다. gate는 data-dependent($\eta_t$)이고, Memora의 $\mathcal{W}$는 simplex 제약이 있으며, 2-layer MLP memory는 convexity를 깨뜨린다. 따라서 이 동치는 설계의 동기이지 보증이 아니다. 3장에서 배운 regret 보증이 여기 자동으로 이식되지 않는 이유가 정확히 이것이다.

마지막으로 [Miras Remark 4]: 두 관점은 GD 기반 유도를 위한 것일 뿐, 식 (13-1)의 정의 자체는 optimizer-agnostic이다. Newton법으로 풀면 Mesa-layer가 되고, non-parametric하게 풀면 softmax attention이 된다. 이 개방성이 다음 절의 통일표를 가능하게 한다.

### 13.3.3 Miras의 네 축

이제 framework를 조립할 수 있다. **Miras framework**는 sequence model을 다음 4-tuple로 정의한다 [Miras §4]:

1. **Memory architecture** — memory의 구조: vector, matrix($\mathcal{M}(k;W)=Wk$), MLP, 또는 그 이상. 선택적으로 $W$에 제약을 건다($\ell_2$ ball, scaled probability simplex 등 — 제약 자체가 안정성 장치다).
2. **Attentional bias** — objective $L$ (또는 그 근사 $\hat\ell/\tilde\ell$). memory가 입력을 어떻게 mapping하고 어떤 사건을 우선하는지 결정한다.
3. **Retention gate** — $R_t$ (FTRL) 또는 $\mathrm{Ret}_t$ (Learning–Retaining; 식 (13-4)의 local+global 분해). plasticity와 stability의 균형. 논문은 long-context 성능의 결정적 레버가 이 축이라고 주장한다.
4. **Memory learning algorithm** — GD, GD+momentum, implicit GD, Newton, multi-step GD, closed-form/non-parametric 해.

부수 축으로 data-dependent 여부, scalar vs channel-wise parameter가 있다 — 같은 4-tuple 좌표의 모델들이 부수 축에서만 갈리는 경우가 많다.

<!-- FIG: ch13/fig-01-design-space -->

### 13.3.4 기존 모델의 재유도: 전부가 이 공간의 점이다

Miras의 통일 결과 [Miras Table 1, §4]를 통일 표기로 재구성한다. 유도의 뼈대만 한 번 보이면 나머지는 기계적이다.

**Hebbian 계열.** attentional bias를 dot-product similarity $\tilde\ell_t=-2\langle Wk_t,\,v_t\rangle$로, local retention을 $\|W-\alpha W_{t-1}\|_F^2$로 놓고 (13-3)을 GD로 풀면, 일차 조건 $-2v_tk_t^\top + 2(W-\alpha W_{t-1})=0$에서 곧바로

$$
W_t \;=\; \alpha\,W_{t-1} \;+\; v_t k_t^\top
\tag{13-5}
$$

가 나온다 [Miras Eq. 8]. $\alpha=1$이면 linear attention, $\alpha$가 학습 상수면 RetNet($n{=}1$ vector state)/Lightning Attention($n{>}1$), $\alpha_t$가 data-dependent scalar면 Mamba-2, diagonal이면 GLA다. dot-product bias는 "비슷한 key에 비례해 크게 쓰라"는 objective이므로 최소화의 해가 순수 가산 write가 된다 — Hebbian write에 replacement가 없는 이유가 objective 수준에서 설명된다.

**delta 계열.** bias를 $\ell_2$ regression $\|Wk_t-v_t\|_2^2$로 바꾸고 같은 retention, (stochastic) GD를 쓰면 6장 카탈로그의 기준형

$$
W_t \;=\; \alpha_t\,W_{t-1}\big(I-\eta_t k_t k_t^\top\big) \;+\; \eta_t\,v_t k_t^\top
\tag{13-6}
$$

이 된다 [Miras Eq. 9; 행렬 곱 순서와 write 항 계수는 표기 관행 차이, 알고리즘 동일]. $\alpha=1$이면 DeltaNet, data-dependent scalar $\alpha_t$면 GDN, channel-wise vector면 RWKV-7이다. Longhorn은 같은 objective를 **implicit GD**(closed-form proximal step)로 푼 것 — algorithm 축만 다른 점이고, DeltaProduct는 token당 여러 GD step을 딛는 multi-step 변형이다. Hebbian → delta의 개선이 "덧쓰기 → 고쳐쓰기"였다는 6장의 서사가, 여기서는 "objective의 교체"라는 한 문장으로 압축된다.

**delta 너머.** Titans-LMM은 nonlinear $\ell_2$ bias(deep MLP memory) + local $D_t=\|W-W_{t-1}\|_F^2$와 global $G_t=\|W\|_2^2$ 둘 다 + **GD with momentum**, 즉 표준형 (M2) 그대로다. Miras는 각주에서 Titans gate와 Mamba-2/GDN gate의 차이를 짚는다: 완전 소거($\alpha_t\to 0$)의 극한에서 Mamba-2류는 다음 token을 "생애 첫 데이터"로 취급하지만, Titans는 소거 직전의 memory로 새 token의 surprise를 먼저 측정하는 cold-start 전략을 쓴다 [Miras Table 1 각주 2]. Mesa-layer는 전체 이력 objective $\sum_{i\le t}\|\mathcal{M}(k_i;W)-v_i\|_2^2+\|W\|_2^2$를 Newton법으로 정확히 푼 극한이다. 그리고 **softmax attention**: $\ell_2$ regression loss의 non-parametric Nadaraya–Watson 해(→ 8장)로, retention이 없고 state가 곧 커지는 집합 $\{(k_t,v_t)\}$ — 즉 KV cache 그 자체다. attention이 선형 state 증가를 갖는 이유는 과거 KV를 **압축 없이 보존하는 associative memory**이기 때문이다 — 단 retrieval 정확도는 softmax kernel weighting에 달려 있어(key 충돌·가중 평균) '완벽한 recall'이 보장되는 것은 아니다.

표 13-1 — 기존 모델의 Miras 좌표 ([Miras Table 1] 재구성; update 식은 식 (13-5)·(13-6) 계열의 §1.6 카탈로그 기준형)

| 모델 | architecture | attentional bias | retention | algorithm |
|---|---|---|---|---|
| linear attention | matrix | dot-product | 없음 ($\alpha=1$) | GD |
| RetNet | vector | dot-product | $\ell_2$ (상수 $\alpha$) | GD |
| GLA / Mamba-2 | matrix | dot-product | $\ell_2$ (data-dep. $\alpha_t$) | GD |
| HGRN2 | matrix | $\ell_1$류 (tied write $v_t(1-\alpha_t)^\top$) | $\ell_2$ | GD |
| DeltaNet | matrix | $\ell_2$ regression | 없음 ($\alpha=1$) | GD |
| Longhorn | matrix | $\ell_2$ regression | — | **implicit GD** |
| GDN | matrix | $\ell_2$ regression | $\ell_2$ (scalar $\alpha_t$) | GD |
| RWKV-7 | matrix | $\ell_2$ regression | $\ell_2$ (channel-wise) | GD |
| DeltaProduct | matrix | $\ell_2$ regression | $\ell_2$ | multi-step GD |
| TTT-Linear / TTT-MLP | matrix / 2-layer MLP | $\ell_2$ regression | 없음 | GD |
| Titans-LMM | deep MLP | nonlinear $\ell_2$ | local+global $\ell_2$ | GD + momentum — (M2) |
| Mesa-layer | matrix | 전체 이력 $\ell_2$ | $\ell_2$ | Newton |
| softmax attention | non-parametric (KV cache) | $\ell_2$ regression | 없음 | Nadaraya–Watson 해 |
| **Moneta** | 2-layer MLP | $\ell_p$ ($p{=}3$) | $\ell_q$ ($q{=}4$) + $\ell_2$ | GD |
| **Yaad** | 2-layer MLP | Huber (mixture) | local+global $\ell_2$ | GD |
| **Memora** | 2-layer MLP | $\ell_2$ | KL / simplex | GD |

단, 논문 자신의 각주가 이 표의 해상도 한계를 인정한다: retention 열의 "$\ell_2$"는 세부가 다른 $\ell_2$류 gate들을 뭉뚱그린 것이고, 유도된 그대로의 $\ell_2$ retention을 쓰는 것은 엄밀히는 Titans와 RWKV-7뿐이다 [Miras Table 1 각주 †]. 또한 4-tuple 좌표는 backbone을 유일하게 식별하지 않는다 — conv, hybrid attention, channel-wise화 같은 부수 축이 남는다.

### 13.3.5 새로운 attentional bias: $\ell_p$, Huber, value-shift robustness

축 (ii)를 여는 세 변형이다 [Miras §5.1]. 공통 동기는 $\ell_2$의 노이즈 민감성이다.

**변형 1 — $\ell_p$ bias.** $L=\|\mathcal{M}(k_t;W)-v_t\|_p^p$ ($p\ge1$) [Miras Eq. 10]. matrix memory에서 GD step의 gradient는

$$
\nabla_W\,\ell_p \;=\; p\,\big[\mathrm{Sign}(Wk_t-v_t)\odot|Wk_t-v_t|^{\,p-1}\big]\,k_t^\top
\tag{13-7}
$$

이다 [Miras Eq. 11]. 여기서 $\mathrm{Sign}(\cdot)$과 $|\cdot|$는 element-wise 부호·절댓값 연산자다(장-국소 정의). $p$는 오차 민감도의 다이얼이다: $p<2$는 큰 residual의 영향을 눌러 노이즈에 강건해지고, $p>2$는 큰 오차 — 즉 심하게 놀란 token — 를 증폭해 기억한다. 극한 $p=1$에서는 update가 $W_t=W_{t-1}-\eta_t\,\mathrm{Sign}(W_{t-1}k_t-v_t)\,k_t^\top$로 단순화되어 [Miras Eq. 12], residual의 크기를 버리고 부호만 write 신호로 쓴다(residual이 $\pm1$로 양자화될 뿐, $\eta_t$와 $k_t$가 곱해져 누적되므로 $W$의 entry와 memory 출력 자체는 일반 실수다). 논문은 이를 **value-less associative memory**라 부른다: 어떤 key가 왔었다는 사실은 저장하되 value의 크기는 저장하지 않는, 인간이 극단적 사건의 세부를 기억에서 눌러버리는 coping mechanism의 유비다.

여기에 시스템적으로 중요한 조건 하나가 붙는다 [Miras Remark 5]: $\mathrm{Sign}$과 $|\cdot|$는 미분 불가능하므로 그대로 두면 outer-loop backprop이 죽는다. 그래서 $\mathrm{Sign}(x)\approx\tanh(\nu x)$, $|x|\approx\sqrt{x^2+\epsilon}$ ($\epsilon=10^{-6}$)로 매끈하게 바꾼다(smoothing 계수는 원문의 $\alpha$를 retention gate와의 충돌 때문에 $\nu$로 개명, 표 13-2). 왜 이것이 사활적인지는 §13.4에서 명확해진다.

**변형 2 — Huber bias: coping mechanism의 정식화.** robust regression의 Huber loss를 attentional bias로 쓴다. 세 가지 적용법이 있다 [Miras §5.1 Variant 2]: (i) 좌표별 Huber 합 — threshold $\delta_t$ 이내 좌표에는 $\ell_2$ gradient, 밖의 좌표에는 sign(value-less) gradient를 쓰는 좌표별 혼합 [Miras Eq. 14]; (ii) residual norm의 Huber $\ell=\mathcal{H}(\|\mathcal{M}(k_t;W)-v_t\|_2)$ — norm이 $\delta_t$를 넘으면 residual을 정규화된 방향벡터로 바꿔 $\delta_t$ 배율로 쓰는, 사실상 **per-token gradient clipping** [Miras Eq. 15]; (iii) 매끈한 혼합 — token 단위로 $\ell_2$와 $\ell_1$ bias 중 하나를 선택:

$$
W_t = W_{t-1} - \begin{cases}
\eta_t\,\nabla_W\ell_2(W_{t-1};k_t,v_t) & \|\mathcal{M}(k_t;W_{t-1})-v_t\|_2\le\delta_t,\\[2pt]
\eta_t\,\delta_t\,\nabla_W\ell_1(W_{t-1};k_t,v_t) & \text{otherwise}.
\end{cases}
$$

[Miras Eq. 16] 형태 (iii)이 Yaad에 실린다. 핵심 설계는 $\delta_t$가 **입력 의존적으로 학습된다**는 것: 어떤 사건이 outlier여서 크기를 깎아 기억해야 하는지를 memory 스스로 token마다 판단한다.

**변형 3 — value shift에 강건한 memory.** worst-case 정식화 $L=\max_{\|\delta_v\|_2\le\Delta}\tfrac12\|\mathcal{M}(k_t;W)-(v_t+\delta_v)\|_2^2$ [Miras Eq. 17]. 내부 max는 닫힌 해가 있어 $L=\tfrac12\|r\|_2^2+\Delta\|r\|_2+\tfrac12\Delta^2$ ($r$은 residual)로 정리되고, update는 통상 $\ell_2$ 항에 정규화된 residual 방향 항이 $\Delta$ 배율로 더해진 형태가 된다. $\Delta$는 학습 가능하다. 결과적으로 이 변형은 $\ell_2$와 $\ell_2$-norm 항의 혼합 — Huber (ii)의 매끈한 친척이다.

### 13.3.6 새로운 retention gate: 정규화의 동물원

축 (iii)이 이 논문의 가장 독창적인 절이다 [Miras §5.2]. 3장의 online optimization 도구함 — mirror descent, Bregman divergence — 이 통째로 sequence layer 설계로 수입되고, 여기에 **elastic net**($\ell_1{+}\ell_2$ 결합 벌점; 3장의 $\ell_1$/$\ell_2$ eviction 구도의 결합형으로, 이 장에서 도입한다)이 더해진다.

**변형 1 — $f$-divergence retention과 scaled probability simplex.** state를 유계 영역에 가두는 것은 수치 안정성의 고전적 처방이다. $\mathcal{W}=\{W:\|W\|_1=c,\ W_{jl}\ge0\}$ — scaled probability simplex — 로 제약하면 $W$를 measure로 볼 수 있고, local retention $D_t$를 $f$-divergence $\sum_{jl}W'_{jl}\,f(W_{jl}/W'_{jl})$로 정의할 수 있다. 선형화된 loss와 결합하면 곱셈형 update $W_t=W_{t-1}\odot g(-\zeta_t-\eta_t\nabla_W\ell(W_{t-1};k_t,v_t))$가 나온다 [Miras Eq. 18]. $g=(f')^{-1}$이고 $\zeta_t$는 $\|W_t\|_1=c$를 강제하는 정규화 상수다. KL 특수화($f(\tau)=\tau\ln\tau$)에 Shannon entropy를 global retention $G_t(W)=\sum_{jl}W_{jl}\log W_{jl}$로 더하면, KKT 조건(제약 최적화의 1차 최적성 조건)에서

$$
W_t \;\leftarrow\; c\,\mathrm{softmax}\big((1-\lambda_t)\log W_{t-1} \;-\; \eta_t'\,\nabla_W\ell(W_{t-1};k_t,v_t)\big),
\qquad
\lambda_t=\frac{1/\alpha_t}{1/\alpha_t+1/\eta_t},\quad
\eta_t'=\frac{1}{1/\alpha_t+1/\eta_t}
\tag{13-8}
$$

가 유도된다 [Miras Eq. 21]. 이것은 online learning의 exponentiated gradient / multiplicative weights를 memory rule로 만든 것이다. softmax가 매 step state를 양수·합-정규화 상태로 유지하므로 **state는 context 길이와 무관하게 절대 발산할 수 없다**. 잊기는 decay가 아니라 확률 질량의 경쟁에서 나온다 — 이 책은 이 질적으로 다른 안정화 기제를 **retention-as-renormalization**이라 부른다(Miras에 암묵적인 개념의 명시화).

**변형 2 — elastic net retention: hard와 soft forgetting.** LASSO($\ell_1$)와 Ridge($\ell_2$)의 혼합을 global retention $G_t(W)=\tfrac{1}{2c_2}\|W\|_2^2+\tfrac{1}{c_1}\|W\|_1$로, local은 $D_t=\tfrac12\|W-W_{t-1}\|_2^2$로 놓으면 (Learning–Retaining에서)

$$
W_t \;=\; \mathcal{S}_\gamma\big(\lambda\,W_{t-1} \;-\; \zeta\,\nabla_W\ell(W_{t-1};k_t,v_t)\big),
\qquad
\gamma=\frac{\eta\,c_2}{c_1(\eta+c_2)},\quad \lambda=\frac{c_2}{c_2+\eta},\quad \zeta=\eta\lambda
\tag{13-9}
$$

가 나온다 [Miras Eq. 22; elastic-net 계수는 원문 $\beta,\alpha$를 $c_2,c_1$로 장-국소 개명]. $\mathcal{S}_\gamma(z)=\mathrm{sign}(z)\max\{0,|z|-\gamma\}$는 element-wise soft-thresholding 연산자다(장-국소 선언; Atlas의 window gate $\gamma_{t,i}$와 무관). 해석이 아름답다: $\lambda\in(0,1)$의 곱셈 감쇠는 **soft forgetting** — 고전적 gate 그 자체 — 이고, thresholding은 **hard forgetting** — $\gamma$ 이하로 작아진 entry를 정확히 0으로 스냅해 용량을 해방한다. 미분 불가능성은 여기서도 smooth surrogate $\mathcal{S}_\gamma(z)\approx|z|\arctan(z/\gamma)/(\pi/2)$로 처리한다.

**변형 3 — elastic net의 FTRL형.** 같은 regularizer를 FTRL viewpoint(식 (13-2))에 넣으면 이중 변수 구조가 나온다: $A_t=A_{t-1}-\eta\nabla_W\ell(W_{t-1};k_t,v_t)$, $W_t=\mathcal{S}_{\eta/c_1}(A_t)$ [Miras Eq. 23]. memory가 날 것의 gradient accumulator $A_t$와 노출용 뷰 $W_t$의 두 층으로 갈라진다.

**변형 4 — 일반 $\ell_q$ stability.** $1<q\le2$에 대해 $R(W)=\tfrac{1}{2\eta(q-1)}\|W\|_q^2$로 놓으면(FTRL) 고전적 결과(Shalev-Shwartz 2012, §2.6)에 의해 $A_t=A_{t-1}-\eta\nabla_W\ell$, $W_t=A_t/\|A_t\|_p^{\,p-2}$ ($p=q/(q-1)$, dual norm)가 된다. accumulator는 자유롭게 크지만 노출되는 memory는 norm이 통제된 껍질 위에 산다.

**변형 5 — Bregman divergence retention = mirror descent.** strictly convex $f$로 $F(W)=\sum_{jl}f(W_{jl})$를 만들고 $D_t$를 $F$의 Bregman divergence로 놓으면 $W_t=g(-\eta\nabla_W\ell(W_{t-1};k_t,v_t)+F'(W_{t-1}))$, $g=(F')^{-1}$ element-wise. $f(\tau)=\tau^2/2$면 plain GD로 퇴화하고, $f'$를 inverse sigmoid로 고르면 $W_t=\sigma(\ln\frac{W_{t-1}}{1-W_{t-1}}-\eta\nabla_W\ell)$ — 모든 state entry가 $(0,1)$에 갇힌다. retention gate의 선택이 곧 **recurrence에 주입되는 nonlinearity의 선택**이 된다는 것, 즉 gate 축과 expressivity 축이 붙어 있다는 것이 이 변형의 교훈이다.

### 13.3.7 출하된 세 모델: Moneta, Yaad, Memora

세 모델의 공통 사양 [Miras §5.3]: memory architecture는 12장의 표준 deep memory에 LayerNorm을 더한 2-layer residual MLP, expansion 4, GELU — $\mathcal{M}(z;W)=z+\mathrm{LN}(W_1\,\sigma(W_2 z))$. memory algorithm은 셋 다 **plain GD** — momentum 없음(Titans에서 후퇴한 유일한 축이며, 의도된 후퇴다: expressivity를 optimizer가 아니라 objective와 retention에 싣는 실험 설계다). inner learning rate $\eta_t$와 retention $\alpha_t$는 분리(decoupled)되고, 모든 gate($\eta_t,\alpha_t,\delta_t\in\mathbb{R}^d$)는 channel-wise이며, parameter 비용을 묶기 위해 RWKV-7을 따라 rank 32–64의 low-rank projection이 emit한다.

**Moneta$(p,q)$** — $\ell_p$ bias + $\ell_q$-stability(+$\ell_2$) retention의 조합:

$$
A_t \;=\; \alpha_t\,A_{t-1} \;-\; \eta_t\,\nabla_W\,\ell_p(W_{t-1};k_t,v_t),
\qquad
W_t \;=\; \frac{A_t}{\|A_t\|_q^{\,q-2}},
\tag{13-10}
$$

gradient는 식 (13-7) [Miras Eq. 24–25]. 출하 값은 $(p,q)=(3,4)$다. 단 $q=4$는 §13.3.6 변형 4의 FTRL 유도 범위($1<q\le2$)를 벗어난 경험적 확장이며, $q=4$ update에 대한 별도 유도나 보증은 원문에 없다. 설계 논리: $p=3$은 잘 회상되지 않는(놀라운) token에 대한 기억 압력을 $\ell_2$보다 날카롭게 하고, $\ell_q$ 정규화는 노출 memory를 norm-통제 껍질에 유지해 곱셈 감쇠보다 강한 안정화를 제공한다.

**Yaad** — Huber mixture bias + Titans식 local+global $\ell_2$ retention:

$$
W_t \;=\; \alpha_t\,W_{t-1} \;-\;
\begin{cases}
\eta_t\,\nabla_W\ell_2(W_{t-1};k_t,v_t) & \|\mathcal{M}(k_t;W_{t-1})-v_t\|_2\le\delta_t,\\[2pt]
\eta_t\,\delta_t\,\nabla_W\ell_1(W_{t-1};k_t,v_t) & \text{otherwise},
\end{cases}
\tag{13-11}
$$

$\delta_t$는 학습된 입력 의존 threshold다 [Miras Eq. 26]. 논리: 노이즈·적대적 스팬 같은 outlier token이 자기 크기만큼 memory를 흔들게 두어서는 안 된다.

**Memora** — plain $\ell_2$ bias + KL retention, 즉 (13-8)의 기계 그대로:

$$
W_t \;=\; \mathrm{softmax}\big(\alpha_t\,\log W_{t-1} \;-\; \eta_t\,\nabla_W\ell_2(W_{t-1};k_t,v_t)\big).
\tag{13-12}
$$

[Miras Eq. 27; (13-8)의 $1-\lambda_t$ 역할을 $\alpha_t$가 맡는다.] 논리: simplex-제약(softmax-재정규화) state는 증명 가능하게 유계이며, context가 아무리 길어도 state가 폭발할 수 없다. $W$가 MLP weights일 때는 같은 rule이 slice별로 적용된다.

**블록 구조와 hybrid** [Miras §5.3]. Miras layer는 Llama macro 구조에서 attention 자리에 들어간다: SwiGLU 채널 MLP, RoPE, RMSNorm. token-mixing 블록 내부는 q/k/v projection 각각 뒤에 depthwise-separable 1D conv(kernel 4), 훈련 안정성을 위한 q·k의 $\ell_2$ normalization, 그리고 memory 읽기 출력의 normalization + linear output gate. hybrid 변형(Moneta-H/Yaad-H/Memora-H)은 Samba를 따라 Miras layer와 Sliding Window Attention layer를 순차 교차한다.

### 13.3.8 표기 대응표

표 13-2 — [Miras] 원 표기 ↔ 통일 표기 (§1.7.2 사본 + 장-국소 행)

| Miras 원 표기 | 통일 표기 | 의미 / 주의 |
|---|---|---|
| $\mathcal{M}$, $W$ (혼용) | $W_t$ / $\mathcal{M}(\cdot;W_t)$ | |
| $\mathcal{M}^*=\arg\min_\mathcal{M} L(\cdot)$ | $\mathcal{M}^\star$ | Titans의 read-star와 무관 |
| $L$ — attentional bias | $L$ | 동일 (공식 용어) |
| $\ell(W_{t-1};\mathbf{k}_t,\mathbf{v}_t)$ | $\ell(W_{t-1};k_t,v_t)$ | 동일 |
| $\hat\ell_i$ (선형화), $\tilde\ell_t$ (surrogate) | 동일 | |
| $\eta_t$ — inner lr ("meta in-context lr") | $\eta_t$ | 동일 |
| $\alpha_t$ — global retention 계수 ($\alpha\mathcal{M}_{t-1}$) | $\alpha_t$ | 동일 방향 (통일안의 기준) |
| $\beta_t\in[0,1]^d$ — decoupled retention gate (§5.3) | $\alpha_t$ 계열로 흡수; 병기 필요시 $\alpha^{\mathrm{g}}_t$ | ⚠ 통일 $\beta_t$(momentum decay)와 무관 |
| $R_t$ (FTRL), $\mathrm{Ret}_t$, $D_t$ (local), $G_t$ (global) | 동일 | |
| $A_t$ — dual accumulator (Moneta, FTRL형) | $A_t$ | 장-국소 |
| $\mathcal{S}_\gamma$ — soft-thresholding | $\mathcal{S}_\gamma$ (장-국소 선언 후 사용) | ⚠ window gate $\gamma_{t,i}$와 구분 |
| $p,q$ — $\ell_p$ bias / $\ell_q$ stability 차수 | 동일 | query $q_t$와 혼동 금지 |
| $\delta_t$ — Huber threshold (learned) | $\delta_t$ | backprop $\delta_\ell$과 첨자로 구분 |
| $\lambda_t,\eta_t',\zeta$ — KKT 유도 파생 계수 | 장-국소 유지 | |
| GD with momentum 행 (Titans 행) | (M2) | |
| Remark 5의 smoothing 계수 $\alpha$ ($\tanh(\alpha x)$) | $\nu$ (장-국소) | ⚠ retention gate $\alpha_t$와 충돌 방지 |
| Eq. 22의 elastic-net 계수 $\beta,\alpha$ | $c_2, c_1$ (장-국소) | ⚠ momentum $\beta_t$·retention $\alpha_t$와 충돌 방지 |
| Eq. 28의 누적 곱 $\beta_i=\prod_{j\le i}\alpha_j$ | $\bar\alpha_i$ (장-국소) | ⚠ momentum $\beta_t$와 충돌 방지 |
| $b$ — chunk 크기 | $C$ | §1.2 |
| $E_b,\ B_b$ — chunk 내 broadcast 행렬 | $E_C,\ B_C$ (장-국소) | |
| lag token 인덱스 $i=kb+1$ | $t=nC+1$ | chunk 인덱스 $n$은 0-based |

## 13.4 Outer-loop training vs inner-loop test-time learning

이 절이 training 무경험 독자에게 가장 중요한 절이다. Miras에는 loss가 두 개, learning rate가 두 개, 학습이 두 개 있다. 표 하나로 먼저 가른다.

표 13-3 — Miras의 두 loop

| | outer loop (pre-training) | inner loop (test time) |
|---|---|---|
| 최적화 대상 | $\Theta$: $W_Q,W_K,W_V$, depthwise conv, output gate, RMSNorm/LN scale, SwiGLU 채널 MLP, gate hypernetwork(low-rank), 초기 memory $W_{\mathrm{init}}$ | $W_t$ (2-layer MLP fast weights); Moneta는 accumulator $A_t$ 추가 |
| objective | $\mathcal{L}$: next-token prediction | $\ell$: 각 layer의 attentional bias |
| update rule | 표준 pre-training (unrolled inner loop 관통 backprop; optimizer 종류·batch·스케줄은 원문 미명시 [Miras App. C — Table 5 외 훈련 레시피 부재]) | (13-10)/(13-11)/(13-12) — smoothed GD 1 step/token |
| 갱신 시점 | training step마다; inference에서는 **동결** | 매 token (훈련·prefill에서는 chunk 단위 근사) |
| learning rate | outer $\eta$ — 원문은 peak LR만 명시(3e-3/1.5e-3/1.25e-3 [Miras Table 5]); 스케줄·optimizer 미명시 | inner $\eta_t$ (channel-wise, token마다 $\Theta$가 생성) |

**outer loop에서 meta-learn되는 것.** key/value/query를 제조하는 projection, conv, output gate, 채널 MLP — 여기까지는 Transformer 훈련과 다르지 않다. Miras 고유의 것은 두 가지다. 첫째, **inner-loop hyperparameter를 emit하는 hypernetwork가 학습된다**: $\eta_t,\alpha_t,\delta_t$는 사람이 정하는 상수가 아니라 현재 token의 함수 $\eta_t=\eta(x_t;\Theta)$로서, rank 32–64 low-rank projection이 채널별로 뽑는다. 즉 "이 gate는 누가 학습하는가?"의 답은: **gate의 값은 inner loop에서 소비되지만, gate를 만드는 정책은 outer loop가 학습한다.** outer loop는 "inner loop가 얼마나 공격적으로 쓰고 얼마나 유지할지"의 per-token·per-channel 정책을 배우는 것이다. 둘째, memory의 초기 상태 $W_{\mathrm{init}}$($W_0=W_{\mathrm{init}}$)도 $\Theta$의 일부로 학습된다.

**gradient는 어떻게 inner loop를 관통하는가.** inference 어휘로 말하면, inner update의 궤적 전체가 forward graph의 일부다. 매 inner step $W_t=\alpha_tW_{t-1}-\eta_t\nabla_W\ell(\cdots)$은 (a) 그 token의 $k_t,v_t$, (b) emit된 gate들, (c) 직전 state의 미분 가능한 함수이므로, $\partial\mathcal{L}/\partial\Theta$는 penultimate token의 memory 읽기에서 출발해 unrolled recurrence를 거슬러 첫 token까지 흐른다. 이것이 4장에서 본 MAML류 "backprop through an optimizer"이며, 훈련 비용과 메모리에 unrolled inner step이 포함되는 이유다. 그리고 이제 §13.3.5의 smooth surrogate가 왜 사활적인지 명확해진다: $\mathrm{Sign}$은 거의 모든 곳에서 도함수가 0이라 outer gradient가 죽고, $|\cdot|$와 $\mathcal{S}_\gamma$는 원점(꺾인 점)에서 미분 불가능하다 — 폭주가 아니라 이 비미분 가능성이 문제다($|\cdot|$의 도함수는 $\pm1$로 유계여서 소실·폭주를 일으키지 않는다). $\tanh(\nu x)$, $\sqrt{x^2+\epsilon}$, $\arctan$-thresholding은 **inner의 수학을 outer가 미분할 수 있게 만드는 접착제**다. inference-only 세계에는 대응물이 없는, 이 라인 고유의 설계 제약이다.

**chunkwise-parallel training.** 이 unrolled 훈련을 가속기에서 실행 가능하게 만드는 것이 9장의 chunkwise 기법이고, Miras는 Titans/TTT의 레시피를 계승한다 [Miras §5.3]. sequence를 크기 $C$(논문 설정 16 또는 64)의 chunk로 나누고, chunk 안 모든 token의 gradient를 직전 chunk의 마지막 state에서 평가한다 — 즉 $\nabla_W\ell(W_{t-1};k_t,v_t)$ 대신 $\nabla_W\ell(W_{\xi(t,C)};k_t,v_t)$, 표준형 (M4)의 stale-snapshot 근사 그대로다. retention까지 포함해 recurrence를 전개하면 ($\bar\alpha_i:=\prod_{j\le i}\alpha_j$)

$$
W_t \;=\; \bar\alpha_t\,W_0 \;-\; \sum_{i=1}^{t}\frac{\bar\alpha_t}{\bar\alpha_i}\,\eta_i\,\nabla_W\,\ell\big(W_{\xi(i,C)};\,k_i,v_i\big)
\tag{13-13}
$$

[Miras Eq. 28; 첫 chunk 기준, $W_0$는 chunk 시작 state]. chunk 안에서 anchor가 고정되므로 $C$개의 gradient는 서로 독립이고, $\ell_p$ loss의 경우 chunk 전체의 gradient 합이 $p\,E_C\odot B_C\odot\mathrm{Sign}(W_0K_{(n)}-V_{(n)})\odot|W_0K_{(n)}-V_{(n)}|^{p-1}\,K_{(n)}^\top$ — column-stack된 chunk 행렬 $K_{(n)},V_{(n)}$ 위의 **batched GEMM 하나 + element-wise epilogue**로 붕괴한다 [Miras Eq. 29]. $E_C$는 $\eta_i$의, $B_C$는 감쇠비 $\bar\alpha_C/\bar\alpha_i$의 broadcast 행렬이다.

선형성을 깨는 비선형은 chunk 경계로 밀어낸다. Moneta의 $\ell_q$ 정규화 $W_t=A_t/\|A_t\|_q^{q-2}$는 token마다가 아니라 chunk 끝에서 한 번 적용한다. Yaad는 $\ell_1$·$\ell_2$ 두 branch의 chunk gradient를 모두 계산한 뒤 $\|\mathcal{M}(k_t;\cdot)-v_t\|_2\le\delta_t$ mask로 혼합한다. Memora가 어려운 경우다: 경계에서 log와 softmax를 연달아 적용하면 $\mathrm{softmax}(\log(x)-g)$ 꼴이 되어 log가 상쇄된다. 그래서 **lag token**을 도입한다 — 각 chunk의 첫 token($t=nC+1$)에서만 진짜 비선형 step (13-12)를 한 번 밟고, chunk의 나머지는 그 결과에서 선형으로 달린다 [Miras Eq. 30].

훈련된(그리고 chunkwise로 서빙되는) 모델은 per-token 정의 (13-10)–(13-12)의 **근사물**이다: gradient는 최대 $C-1$ token 묵은 state에서 평가되고, state 비선형은 chunk당 한 번만 발화한다. chunk 크기가 스케줄이 아니라 계산되는 함수 자체를 바꾸는 semantic hyperparameter라는 9장의 명제가 여기서도 성립하며, Miras는 이 근사의 오차를 정량화하지 않고 Titans/TTT의 경험적 정당화를 상속한다 — 6편 어디에도 이 staleness의 오차 한계는 없다(→ 9장, 15장).

**inference에서 움직이는 것.** layer당·head당 fast weights $W=\{W_1,W_2\}$ (그리고 Moneta의 $A_t$)만 갱신되고, 나머지 전부 — projection, conv, gate hypernetwork, 채널 MLP — 는 동결이다. gate들은 동결된 projection이 현재 token에서 계산해 내놓는 값이므로, inference 시에도 token마다 다르지만 **학습되고 있지는 않다**. per-token 갱신은 forward(예측 $\mathcal{M}(k_t;W_{t-1})$ 형성) → backward(2-layer MLP 관통 $\nabla_{W_1},\nabla_{W_2}$) → element-wise retention/정규화 → query로 읽기 $y_t=\mathcal{M}(q_t;W_t)$의 네 단계이고, 비용은 context 길이와 무관한 상수다. 정량은 §13.7에서 다룬다. 12장과의 대비: Titans decode는 momentum buffer $S_t$까지 끌고 다녔지만, Miras 세 모델은 plain GD라 $S_t$가 없다 — 대신 Moneta는 FTRL 이중 구조의 $A_t$라는 두 번째 state를 끌고 다닌다.

## 13.5 Concept ledger delta

표 13-4 — Miras가 ledger에 가한 변경

| 개념 | 변경 유형 | 내용 |
|---|---|---|
| attentional bias | **신규 (공식 용어)** | inner objective를 설계 축으로 명명·정식화. Titans/TTT의 암묵적 $\ell_2$는 한 선택지로 강등 |
| retention gate | **재이론화 + 개명** | forget gate → retention regularization; local $D_t$ + global $G_t$ 분해; "모델은 지우지 않는다, 유지하지 않을 뿐" |
| Miras framework (4축) | 신규 | architecture × bias × retention × algorithm; 이후 논문 전부가 이 좌표계를 전제 |
| Learning–Retaining / FTRL viewpoint | 신규 | 두 유도 렌즈 + Prop. 3.2 포함 관계; FTRL 자체는 3장 소유 |
| meta in-context learning rate | 신규 (해석) | $\eta_t$ = 새것 학습량과 옛것 방출량을 동시에 정하는 다이얼 |
| value-less associative memory | 신규 | $p{=}1$ 극한: key의 발생만 $\pm1$로 저장 |
| hard / soft forgetting | 신규 | elastic net에서: 곱셈 감쇠(soft) vs soft-thresholding의 0-스냅(hard) |
| retention-as-renormalization | 신규 (암묵 → 이 책이 명명) | simplex/norm 구속에 의한 안정성: decay 없는 forgetting |
| lag token | 신규 (훈련 기법) | Memora의 chunk 경계 1회 비선형 step |
| Moneta / Yaad / Memora | 신규 (모델) | 축 (ii)·(iii)의 세 실증점 |
| surprise (Titans, → 12장) | **흡수·일반화** | momentary surprise = 임의 attentional bias의 gradient |
| Titans-LMM | 지위 변화 | 독립 아키텍처 → Table 1의 한 행 ((M2) 좌표) |
| momentum $S_t$ (inner) | **보류 (사실상 일시 폐기)** | 세 모델 모두 plain GD; algorithm 축은 의도적으로 미탐사 — Atlas가 상속 |

누적 관점에서 가장 큰 변화는 어휘의 세대 교체다. 12장까지 서사의 주인공이던 "surprise"는 이 장 이후 특정 bias($\ell_2$)의 gradient에 대한 별명이 된다. 반대로 Titans의 inner momentum은 ledger에서 일시 퇴장한다 — 바로 그 빈 축이 14장의 Atlas가 여는 문이다.

## 13.6 실험과 스케일

**설정** [Miras §6, App. C]. 훈련 context window 4096. 언어모델링·상식추론은 FineWeb-Edu, scaling 곡선은 C4. 모델 크기는 본문 기준 120M/340M/760M/1.3B이고, token 수는 소형(120M·340M) 15B, 760M 30B, 1.3B 100B다. 아키텍처 세부는 [Miras Table 5]: 12 block/dim 768/16 head(peak LR 3e-3), 24/1024/16(1.5e-3), 24/1536/16(1.25e-3). 단 App. C 표는 크기를 170M/340M/780M으로 적어 본문의 120M/760M과 표기가 어긋난다 — 원문 자체의 불일치이므로 이 책은 본문 수치(760M 등)로 인용하되 여기 한 번 기록해 둔다. baseline은 Transformer++, RetNet, GLA, Mamba, Mamba2, DeltaNet, TTT, GDN과 hybrid인 Samba, GDN-H2이며, **baseline 수치는 Titans 논문이 보고한 값을 그대로 가져온 것이다** [Miras §6 Setup] — 같은 저자 라인의 재사용이므로 훈련 설정은 정합하지만, 독립 재현이 아니라는 점은 기억할 것. 같은 저자 라인의 Atlas가 AdamW·cosine·batch 0.5M tokens를 명시하는 것과 달리 [Atlas App. E], Miras에는 outer optimizer의 종류·batch·스케줄이 없다 [Miras App. C — Table 5 외 훈련 레시피 부재] — 재현 시도자는 이 부재를 알고 시작해야 한다.

**언어모델링과 상식추론** [Miras Table 2]. 340M/15B에서 Moneta가 WikiText ppl 26.19로 GDN 27.01, TTT 27.44, Transformer++ 31.52를 앞선다. 760M/30B에서는 Yaad 20.99 / Moneta 21.18 / Memora 22.28로, 순수 recurrent끼리는 GDN(21.18)과 대등하거나 앞서지만 hybrid Samba(20.63)·GDN-H2(19.88)에는 밀린다 — 이 스케일에서는 hybrid가 여전히 우세하다. 그런데 Miras 쪽도 hybrid를 만들면(Samba식 SWA 교차) Memora-H 18.24 / Yaad-H 18.59 / Moneta-H 18.72로 GDN-H2를 다시 앞선다. 그리고 flagship인 1.3B/100B: **순수 recurrent인 Yaad가 ppl 15.18, Moneta 15.52로, GDN(16.42)만이 아니라 hybrid인 Samba(16.13)와 GDN-H2(15.91)까지 이긴다** [Miras Table 2]. LAMBADA ppl도 같은 그림이다(Moneta 11.47 vs GDN 12.17). "더 나은 attentional bias + retention이면 attention 없이 attention hybrid를 이긴다"가 이 논문의 헤드라인 주장이고, 적어도 이 스케일·이 벤치마크에서는 수치가 그것을 지지한다.

**S-NIAH (RULER)** [Miras Table 3]. 1K–8K 길이의 single needle-in-haystack 세 변형에서 평균 Moneta 93.5 / Yaad 92.9 / Memora 92.1 — GDN 75.8, TTT 66.1, DeltaNet 57.9, Mamba2 52.0과의 격차가 크다. 특히 haystack이 합성 노이즈인 S-NIAH-PK에서 Moneta는 8K에서도 98.8을 유지하는데, 논문은 이를 노이즈에 강건한 $p$-norm objective의 효과로 귀속한다 [Miras §6.3].

**scaling 곡선** [Miras Fig. 3]. FLOPs-매칭 ppl(모델 크기 축)과 context 길이 축(2K–32K, 340M·760M) 모두에서 세 변형이 baseline보다 좋은 기울기를 보인다. 논문의 효율 주장 전체가 이 FLOPs 기준 그림 하나에 얹혀 있다는 점은 §13.7에서 다시 짚는다.

**ablation** [Miras §6.4]. 세 개가 실렸고 각각이 설계 축 하나씩을 겨냥한다. (1) $p\in\{1,1.5,2,2.8,3,3.2,4\}$ sweep: 성능은 $p$에 대해 **비단조**이고 최적은 $p=3$, 최악은 $p=4$다. 흥미롭게도 $p$는 context-길이 scaling의 모양은 바꾸지 않는다. (2) $q\in\{2,3,4,5\}$ sweep: 반대로 $q$는 **scaling 패턴 자체를 바꾼다** — retention gate의 품질이 long-context 거동을 지배한다는 논문 주장의 가장 직접적인 증거다. (3) Yaad 구성요소 제거 [Miras Table 4]: 평균 LM 점수 53.98에서 retention gate 제거 시 50.63(−3.35), $\delta$ 입력-독립화 52.19(−1.79), $\ell_2$ branch 제거 52.86(−1.12), $\ell_1$ branch 제거 53.04(−0.94), MLP를 linear memory로 교체 51.57(−2.41). 기여 순위가 retention > deep memory > threshold의 입력 의존성 > bias 세부라는 것 — 논문이 "retention이 결정적 레버"라 주장한 순서 그대로다.

**스케일의 정직한 상한.** 이 라인 전체가 그렇듯 실증은 1.3B params / 100B tokens에서 끝난다. 7B+에서, SFT/RLHF 이후에, exact-copy recall이 지배하는 실무 부하에서 순수 recurrent가 hybrid를 이기는 순서가 유지되는지는 이 논문이 답하지 않는 질문이며(§13.8), long-context 검증도 single-needle S-NIAH 하나뿐이다. multi-needle, multi-hop, 32K 초과 외삽은 미검증이다.

## 13.7 Systems/serving 함의

이 절은 논문 내용을 독자의 세계 — decode 비용, state 크기, kernel — 로 번역한다. 논문 자신은 이 산수를 하지 않으므로, 별도 표기 없는 정량은 전부 이 책의 계산이다.

> **[해설] state 크기.** Miras layer의 recurrent state는 head당 2-layer MLP 전체다: $W_1\in\mathbb{R}^{d\times 4d}$, $W_2\in\mathbb{R}^{4d\times d}$ ($d$ = head 차원), 합계 $8d^2$ 개의 fp 스칼라. matrix-state 모델(DeltaNet/GDN)의 $d^2$의 **8×**이고, Moneta는 accumulator $A_t$가 같은 모양이므로 **16×**($16d^2$)다. $d=64$면 head당 32K entry(Moneta 64K) vs matrix state 4K. KV cache와의 손익분기: cache는 head당 $2Nd$ entry이므로 $8d^2=2Nd$에서 $N=4d$ — $d=64$ 기준 **약 256 token**(Moneta 512)만 넘으면 state가 cache보다 작고, 이후는 context가 아무리 길어도 상수다. linear-RNN 계열의 표준 serving 이점(O(1) decode 메모리·FLOPs, cache paging 불필요)은 그대로 성립하되, state가 선배들보다 8–16× 뚱뚱하다는 것이 Miras의 세금이다. per-sequence state 상주 비용, prefix caching용 state checkpoint, speculative branch당 state 복제가 전부 그 배율로 커진다.

**decode: RMW 트래픽이 지배한다.** per-token 갱신은 memory MLP의 forward($W_2z$와 $W_1h$ 두 층 합쳐 $8d^2$ MAC) + backward($\approx$ forward의 2배, $16d^2$ MAC) + query 읽기 forward($8d^2$ MAC)에 element-wise retention/정규화가 더해져, head-layer당 대략 $28$–$32\,d^2$ MAC(FLOP로는 약 $56$–$64d^2$) — 같은 범위로 센 GDN류 update(약 $3$–$4d^2$ MAC)의 대략 8–10× 수준이다(이 책의 추산; 1 MAC = 2 FLOP 단위를 섞지 않도록 주의). 그러나 결정 변수는 FLOPs가 아니다. **매 token마다 $8d^2$($16d^2$) state 전체를 read-modify-write** 해야 하므로 decode는 memory-bandwidth-bound이고, 이 라인의 decode state RMW 트래픽 논증(→ 10장)이 Miras에서 8–16× 배율로 적용된다. attention decode가 KV cache를 read-only로 스트리밍하는 것과 달리, 여기는 write가 절반이다.

**prefill/훈련: chunkwise = GEMM 골격.** (13-13)의 chunkwise 형태 덕에 prefill과 훈련은 GLA/Titans/TTT식 chunkwise kernel과 같은 골격이다: chunk당 $|W_0K_{(n)}-V_{(n)}|^{p-1}K_{(n)}^\top$류 batched GEMM + element-wise epilogue, 순차 의존은 chunk 경계에만 남는다($L/C$ step). Miras 고유의 kernel 고려사항 네 가지: (1) Moneta의 $\|A_t\|_q$는 경계마다 $8d^2$ entry 전체에 대한 global norm reduction — GEMM phase 사이에 끼는 reduction kernel이다. (2) Memora의 softmax/log-sum-exp는 token 축이 아니라 지정된 parameter slice마다 수행된다([Miras Eq. 21]과 §13.3.7: $W$가 MLP weights일 때 slice별 적용) — reduction 범위와 비용은 구현 선택에 달렸지 단일 $8d^2$ 전역 reduction이 강제되는 것은 아니다; 대신 state가 유계·정규화되어 저정밀 저장에 원리적으로 우호적이다(단 log가 소값의 양자화 오차를 증폭한다). (3) Yaad는 $\ell_1$·$\ell_2$ 두 branch gradient + mask로 element-wise 작업이 약 2×, token당 residual-norm reduction 하나가 추가된다. (4) smooth surrogate들($\tanh$, $\sqrt{x^2+\epsilon}$, $\arctan$)은 값싼 element-wise epilogue다. gate hypernetwork(rank 32–64)의 비용은 무시 가능하다.

**batching과 hybrid.** per-request fast-weight state는 shared-weight batching을 깨뜨린다는 1장의 Rosetta 항목이 그대로 적용되고, state가 8–16× 커진 만큼 grouped-GEMM decode의 per-request working set도 그 배율로 커진다. hybrid 변형(-H)은 SWA를 교차하므로 window 크기의 KV cache($O(w\,d)$)가 recurrent state 위에 다시 얹힌다 — "cache 없는 serving"은 순수 변형에만 해당한다.

**증거의 한계.** 논문에는 wall-clock 수치, kernel 구현, decode throughput 비교가 전혀 없다 — 효율 증거는 FLOPs-매칭 perplexity [Miras Fig. 3] 하나다. "fast parallelizable training"은 측정이 아니라 Titans/TTT chunkwise 레시피에서 상속된 구조적·점근적 주장으로 읽어야 한다. 이 부재는 Miras만의 흠이 아니라 6편 전체의 공백이며(decode wall-clock은 라인 어디에도 없다), 15장의 TNT가 처음으로 훈련 쪽 wall-clock을 내놓는다.

> **[평가]** 설계 관점에서 아까운 지점 하나: elastic net(식 (13-9))의 hard forgetting(0-스냅)은 state sparsity를 만들어 내는 gate — 즉 state 압축·양자화의 자연스러운 훅 — 인데, 출하된 세 모델 어디에도 실리지 않았다. serving 경제학에 가장 직접적으로 닿는 변형이 실증 없이 카탈로그에만 남은 셈이다.

## 13.8 한계와 bridge-out

**논문이 남긴 것.** 첫째, **memory algorithm 축이 비어 있다.** 세 모델 모두 plain GD다. momentum(Titans), implicit GD(Longhorn), multi-step(DeltaProduct), Newton(Mesa)이 새 bias·gate와 합성되는지는 열린 문제로 명시된다 — 논문 스스로 Newton과 다른 optimizer를 future work로 지목한다. 둘째, **inner objective가 여전히 per-token single-pair다.** 모든 bias가 "지금 이 $(k_t,v_t)$ 하나"를 얼마나 잘 저장하는가만 묻고, 최근 $c$개 token이 **함께** 잘 저장되어 있는가는 아무도 묻지 않는다. 셋째, **이론이 없다.** 새 bias·gate 어느 것에도 capacity, regret, state-tracking 급의 결과가 없다 — online convex optimization의 도구함을 통째로 수입했지만 그 보증이 요구하는 convexity를 2-layer MLP memory가 깨기 때문이다(§13.3.2의 [평가] 참조). 넷째, chunkwise 근사(stale gradient, 경계 비선형, lag token)의 오차는 미정량이고 $C$-ablation도 없다. 다섯째, $(p,q)=(3,4)$는 sweep의 승자일 뿐 데이터 통계와 잇는 이론이 없으며, 과제별·적응형 $p$는 미탐사다. 여섯째, 8–16× state와 3–4× update 비용이 production batch에서 품질 이득을 정당화하는지 — serving 경제학 — 는 수치가 전무하다.

> **[평가]** 이 논문의 실질 기여는 세 모델보다 좌표계다. Moneta/Yaad/Memora의 수치는 1.3B에서 인상적이지만 그 스케일에서 멈추고, 반면 attentional bias / retention gate / 4축이라는 어휘는 이후 모든 논문이 무상으로 사용한다. 통일 주장 자체에도 결이 있다: Table 1은 각 모델의 "원 설계 방정식" 수준의 동일시이며, 논문 각주가 인정하듯 retention 열은 $\ell_2$류를 뭉뚱그린 것이다. 분류학으로서는 정확하되, 각 칸의 등호를 구현 수준의 등가로 읽으면 과독이다.

**Bridge-out → 14장 (Atlas).** Miras가 열어 둔 세 문제 — (i) 비어 있는 optimizer 축, (ii) per-token 단일 쌍 objective, (iii) 부재하는 capacity 이론 — 는 그대로 다음 논문의 목차가 된다. [Atlas] (*Atlas: Learning to Optimally Memorize the Context at Test Time*, arXiv:2505.23735)는 셋을 한꺼번에 공격한다: (i) inner loop에 Muon을 이식해 — momentum을 Newton–Schulz로 semi-orthogonalize — 병렬화 가능한 근사 2차 memory update를 만들고, (ii) 최근 $c$ token의 window 전체를 함께 최적화하는 Omega rule로 "token의 surprise"를 "context의 surprise"로 바꾸며($c{=}1$이 delta rule/Titans, $c{=}L$이 Mesa-layer로 퇴화하는 스펙트럼), (iii) matrix memory $O(d_k)$ → polynomial feature $O(d_k^p)$ → $\phi^*$ 무한 capacity(= softmax attention)의 정식 capacity 이론으로 §13.3.4의 attention 행 — "attention은 압축하지 않는 memory" — 에 정리를 붙인다. Miras가 지도를 그렸다면, Atlas는 그 지도의 빈 축들을 최적화 이론으로 채우러 간다. 명칭도 이때 바뀐다: test-time *training*이 아니라 test-time *memorization*이라는 Atlas의 고집이 왜 단순한 수사가 아닌지 — 14장에서 본다.
