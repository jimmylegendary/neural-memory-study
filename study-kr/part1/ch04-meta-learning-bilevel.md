# ch04. Meta-learning과 bilevel optimization: inner loop vs outer loop

<!-- STYLE-ISSUE: 이 장의 파일명이 STYLE-NOTATION §5.1의 고정 슬러그(ch04-meta-learning.md)와 달리 ch04-meta-learning-bilevel.md로 지정되어 작성되었다. P2 병합 시 슬러그 통일 필요. -->

> **이 장의 목표** — 이 장을 마치면 다음을 할 수 있어야 한다.
> 1. sequence model을 bilevel optimization 문제로 형식화하고, 임의의 파라미터가 inner loop 소속인지 outer loop 소속인지를 기호($W$ vs $\Theta$)와 역할만으로 판별할 수 있다.
> 2. "outer loop가 inner loop를 **통과해** 학습한다"는 문장을 computational graph 수준에서 설명하고, 1-step 예제의 hypergradient를 손으로 계산할 수 있다.
> 3. "이 gate는 누가 학습하는가?"라는 질문에 함수와 값을 구분해 정확히 답할 수 있다.
> 4. MAML의 initialization-as-meta-variable 아이디어를 이 라인의 $W_{\mathrm{init}}$과 연결할 수 있다.
>
> **왜 필요한가** — 6편 전부가 이 장의 어휘 위에 서 있다.
> - [Titans] (*Titans: Learning to Memorize at Test Time*, arXiv:2501.00663)는 memory module을 "meta in-context model"로 정의하고, inner loss 안의 projection $W_K, W_V$를 "hyperparameter"라고 부른다 [Titans §3.1]. 이 문장은 bilevel 어휘 없이는 파싱되지 않는다.
> - [Miras] (*It's All Connected*, arXiv:2504.13173 — 실제 제목은 *It's All Connected*이지만 이 책은 framework 이름 Miras로 통칭한다)의 네 번째 설계 축 "memory learning algorithm(= optimizer)" [Miras §1]은 "inner loop의 optimizer를 무엇으로 고르는가"라는 질문 그 자체다.
> - [Atlas] (*Atlas: Learning to Optimally Memorize the Context at Test Time*, arXiv:2505.23735)는 라인의 공식 정의를 명문화한다: "the sequence model is a meta in-context learner with two optimization levels" [Atlas §2]. Atlas의 "locally optimal" 주장과 Mesa-layer 대조는 inner 문제를 어디까지 푸는가의 문제다.
> - [TNT] (*TNT: Improving Chunkwise Training for Test-Time Memorization*, arXiv:2511.07343)의 local memory는 학습되는 공유 초기 상태 $W_{\mathrm{init}}$으로 주기적으로 reset된다 [TNT §4.1.1, Eq. 6] — MAML의 initialization-as-meta-variable가 그대로 load-bearing 부품이 된 사례다.
> - [NL] (*Nested Learning: The Illusion of Deep Learning Architecture*, arXiv:2512.24695)은 두 level을 K개 level의 nested optimization으로 일반화한다 [NL Abstract, §1].
> - [Sleep] (*Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories*, arXiv:2606.03979)은 ICL을 meta-learning process로 보는 정식화를 출발점으로 삼는다 [Sleep §1].

2장에서 optimizer를 (state, update, cost)를 갖는 객체로 만들었고, 3장에서 그 객체를 token stream 위에서 도는 online learner로 바꿨다. 이 장은 마지막 남은 구조적 질문에 답한다: **그 online learner 자체는 누가 만들었는가?** 답은 "또 하나의 optimizer가, 더 느린 시간축에서"이며, 이 이중 구조를 정확히 말하는 언어가 bilevel optimization이다. 1장에서 비형식적으로 도입한 inner loop / outer loop 구분을 여기서 형식화한다.

## 4.1 두 개의 최적화, 두 개의 시간축

독자의 세계에는 견고한 불변식이 하나 있다: **serving 중에 weight는 변하지 않는다.** 엔진 설계 전체 — weight를 한 번 로드하고, 여러 요청이 shared weights 위에서 batching되고, KV cache만 요청마다 자란다 — 가 이 불변식 위에 서 있다. 이 라인은 이 불변식을 정확히 절반만 폐기한다(→ 1장 Rosetta 사전: "폐기되는 불변식"). weight의 **일부**($W$, fast weights → 6장)는 decode 중 매 token 갱신되고, 나머지($\Theta$, slow weights)는 여전히 동결된다. 어느 쪽에 속하는지가 이 책의 기호 규약 제1조($W$ vs $\Theta$)이고, 그 구분의 수학적 실체가 이 절의 내용이다.

**bilevel optimization**은 한 최적화 문제의 제약 조건 안에 또 다른 최적화 문제가 들어 있는 구조를 말한다. 일반형은 다음과 같다. 바깥 문제는 $\Theta$를 움직여 outer loss $\mathcal{L}$을 줄이려 하는데, $\mathcal{L}$이 의존하는 $W^\star$가 그 자체로 안쪽 최적화의 해다:

$$
\min_{\Theta}\;\mathcal{L}\big(\Theta,\,W^\star(\Theta)\big)
\qquad\text{s.t.}\qquad
W^\star(\Theta)\;\in\;\arg\min_{W}\;\ell(W;\Theta)
\tag{4-1}
$$

식 (4-1)이 말하는 것은 의존성의 방향이다: $\Theta$가 바뀌면 안쪽 문제의 정의 자체가 바뀌고, 따라서 그 해 $W^\star(\Theta)$가 바뀌고, 그 결과 바깥 loss가 바뀐다. 안쪽 문제를 **inner loop**(최적화 대상 $W$, objective $\ell$), 바깥 문제를 **outer loop**(최적화 대상 $\Theta$, objective $\mathcal{L}$)라고 부른다. 소문자 $\ell$과 대문자 $\mathcal{L}$의 구분은 이 두 loop의 구분을 시각화한 것이다(표기 규약 §0).

그런데 이 라인의 실제 모델은 (4-1)의 argmin 형태가 아니다. inner 문제는 **끝까지 풀리지 않는다**. token 하나가 도착할 때마다 GD 한 step만 밟고, 그 중간 상태들의 궤적 자체가 출력을 만든다. 즉 우리가 다루는 것은 **trajectory 형태의 bilevel 문제**다:

$$
\min_{\Theta}\;\mathcal{L}(\Theta)\;=\;\sum_{t=1}^{L}\mathcal{L}_t\big(y_t\big),
\qquad
y_t=\mathcal{M}\big(q_t;\,W_t(\Theta)\big)
\tag{4-2}
$$

$$
W_t(\Theta)\;=\;W_{t-1}(\Theta)\;-\;\eta_t\,\nabla_W\,\ell\big(W_{t-1}(\Theta);\,k_t,v_t\big),
\qquad W_0=W_{\mathrm{init}}
\tag{4-3}
$$

식 (4-3)은 식 (M1) 그대로이되, 모든 재료의 $\Theta$ 의존성을 명시한 것이다: $k_t=W_Kx_t$, $v_t=W_Vx_t$, $q_t=W_Qx_t$의 projection 행렬이 $\Theta$의 일부이고, inner learning rate는 $\eta_t=\eta(x_t;\Theta)$처럼 slow weights가 만드는 token의 함수이며, 초기 상태 $W_{\mathrm{init}}$도 $\Theta$에 속한다. momentum과 retention이 붙은 (M2)도 구조는 동일하다 — $\beta_t,\alpha_t$의 생산 함수가 $\Theta$에 추가될 뿐이다. 식 (4-2)의 $\mathcal{L}_t$는 next-token prediction loss다. 표기를 단순화했지만 실제로는 $y_t$가 backbone의 상위 layer들을 거쳐 token 분포가 되며, 그 경로의 파라미터 전부가 $\Theta$에 속하므로 $\mathcal{L}_t(y_t)$로 축약해도 구조는 같다.

두 loop는 시간축이 다르다. inner loop는 **한 시퀀스 안에서** token마다 돈다 — 3장에서 본 대로, 이것은 독자가 이미 운영하는 autoregressive decoding loop와 같은 모양이고, 다만 state 갱신이 gradient step이라는 점만 다르다. outer loop는 **pretraining 동안** mini-batch마다 돌고, serving이 시작되면 멈춘다. 독자의 어휘로 옮기면: $\Theta$는 컴파일된 커널에 박힌 상수처럼 배포 시점에 고정되는 것이고, $W_t$는 KV cache처럼 요청과 함께 태어나 요청과 함께 죽는 runtime state다. "training이 두 번 있다"가 아니라, **training(outer)이 학습해서 내놓는 산출물이 또 하나의 작은 learner(inner)**라는 것 — 이것이 meta-learning이라는 이름의 이유다. **meta-learning**(learning-to-learn)은 학습 절차 자체의 구성 요소(초기값, learning rate, update rule, objective의 파라미터)를 더 바깥 최적화의 변수로 삼는 기법의 총칭이다.

## 4.2 hypergradient: gradient를 통과하는 gradient

outer loop도 결국 2장의 optimizer 객체다: state는 $(\Theta, m_t, h_t)$(AdamW의 moment buffer → 2장), update는 AdamW step, 필요한 입력은 $\nabla_\Theta\mathcal{L}$이다. 문제는 이 gradient의 경로다. $\Theta$는 식 (4-3)의 inner update **안에** 들어 있으므로, $\nabla_\Theta\mathcal{L}$을 얻으려면 inner loop 전체를 미분해야 한다. 이렇게 inner 최적화 절차를 관통해 계산되는 outer 변수의 gradient를 **hypergradient**라고 부른다.

가장 정직한 계산법은 **unrolling**이다: 식 (4-3)의 update를 $t=1$부터 $L$까지 펼쳐 놓으면 하나의 거대한 computational graph가 된다. inner update도 미분 가능한 연산의 합성일 뿐이므로 — $\nabla_W\ell$ 자체가 graph의 한 노드다 — 이 graph 전체에 2장의 backward pass를 그대로 적용하면 $\nabla_\Theta\mathcal{L}$이 나온다. 이것이 "outer loop가 inner loop를 통과해 학습한다"의 문자 그대로의 의미다. 독자에게 익숙한 그림으로 옮기면 unrolled inner loop는 **같은 연산이 $L$번 반복되는, data-dependent 분기가 없는 정적 dataflow graph**다. 정적이라는 사실이 뒤에서 결정적으로 중요해진다(§4.7).

1 step만 손으로 미분해 보면 hypergradient의 구조가 드러난다. inner lr를 장-국소 기호 $\eta_{\mathrm{in}}$(학습되는 스칼라 상수; 무첨자 $\eta$는 outer lr로 예약되어 있으므로 구분한다)으로 두고, $g(W):=\nabla_W\ell(W;k,v)$라 하자. 한 step $W_1=W_0-\eta_{\mathrm{in}}\,g(W_0)$ 뒤에 outer loss $\mathcal{L}(W_1)$을 평가하면, chain rule로

$$
\frac{\partial \mathcal{L}}{\partial \eta_{\mathrm{in}}}
=-\,g(W_0)^\top\,\nabla_{W_1}\mathcal{L},
\tag{4-4}
$$

$$
\frac{\partial \mathcal{L}}{\partial W_0}
=\Big(I-\eta_{\mathrm{in}}\,\nabla^2_W\ell(W_0)\Big)\,\nabla_{W_1}\mathcal{L}.
\tag{4-5}
$$

식 (4-4)는 "inner gradient의 방향이 outer loss를 줄이는 방향과 얼마나 정렬되어 있는가"를 재고, 식 (4-5)에는 inner loss의 **Hessian** $\nabla^2_W\ell$이 나타난다. inner update가 이미 gradient를 포함하므로 그것을 다시 미분하면 gradient의 gradient, 즉 2차 미분이 튀어나오는 것이다. $L$ step을 펼치면 식 (4-5)의 Jacobian $(I-\eta_{\mathrm{in}}\nabla^2_W\ell_\tau)$들이 곱으로 연쇄된다 — RNN의 BPTT에서 transition Jacobian이 연쇄되는 것과 정확히 같은 구조이고, 같은 병(긴 연쇄의 소실·폭발, 그리고 궤적 전체를 저장해야 하는 activation memory)을 앓는다. 행렬 $W$의 경우 Jacobian이 고차 tensor가 되지만 구조는 동일하다.

이 병의 표준 처방이 **truncated unrolling**이다: graph를 $T$ step마다 잘라, 자른 지점 이전으로는 gradient를 흘리지 않는다. 계산과 메모리를 아끼는 대신 hypergradient가 편향된다 — 잘린 구간 너머로 전파됐어야 할 신호가 0으로 처리되기 때문이다. 이 트레이드오프를 기억해 두면 9장이 쉬워진다: chunkwise training의 stale-snapshot 근사(식 (M4))는 chunk 시작 상태에 gradient anchor를 동결하는, 정확히 이 truncation의 사촌이며, 거기서 truncation 길이의 역할을 chunk 크기 $C$가 맡는다.

unrolling의 대안으로 **implicit differentiation**이 있다: inner 문제가 argmin까지 풀린다고 가정하면(식 (4-1)의 형태), 최적점의 1차 조건 $\nabla_W\ell(W^\star;\Theta)=0$에 implicit function theorem을 적용해 궤적을 저장하지 않고도 hypergradient를 얻는다. 대가는 inner Hessian이 낀 선형계를 푸는 비용이며, hyperparameter optimization과 meta-learning의 형식적 통합은 Franceschi et al. 2018 (arXiv:1806.04910)이 정리했다. 이 라인이 implicit 노선을 쓰지 않는 이유는 이제 자명하다: inner 문제가 애초에 argmin까지 풀리지 않고, 중간 궤적 $W_1,\dots,W_L$ 하나하나가 $y_t$를 만들기 때문이다. 예외가 궤적 대신 매 step 정확한 해를 쓰는 Mesa-layer이고(§4.5), 그래서 Atlas가 이를 대조군으로 세운다.

마지막으로 근사의 계보 하나: 식 (4-5)의 Hessian 항을 통째로 버리고 $\partial\mathcal{L}/\partial W_0\approx\nabla_{W_1}\mathcal{L}$로 쓰는 것이 first-order MAML(FOMAML)류의 근사다. 정확도와 비용의 이 저울질은 학습되는 것이 무엇이냐에 따라 달라지는데, 그 "무엇"의 목록이 다음 절의 주제다.

## 4.3 learning-to-learn 계보: 무엇을 meta-변수로 삼는가

meta-learning의 역사는 "outer loop에 무엇을 넘길 것인가"의 역사다. 이 라인이 직접 인용하는 조상만 추리면 세 갈래다.

첫째, **Schmidhuber의 자기 수정 계보**. Schmidhuber 1987(diploma thesis)은 학습 절차 자체를 학습 대상으로 삼는 self-referential learning을 제안했고, Schmidhuber 1992(*Learning to control fast-weight memories*, Neural Computation)는 느린 network가 빠른 network의 weight를 써넣는 구조 — fast weights(→ 6장)의 원형 — 를, Schmidhuber 1993(ICANN)은 자기 자신의 weight를 읽고 수정하는 self-referential weight matrix를 제시했다. 이것은 골동품 인용이 아니다. [NL]은 backprop 자체가 self-referential process라는 논증과 self-modifying Titans의 설계에서 Schmidhuber 1992/1993을 직접 인용한다 [NL §4, §7].

둘째, **learned optimizer**. Andrychowicz et al. 2016 (arXiv:1606.04474)은 update rule 자체를 학습했다: 작은 recurrent network가 gradient를 입력받아 $\Delta w$를 출력하고, 그 network의 파라미터를 outer loop가 학습한다. "optimizer는 import하는 고정 부품이 아니라 학습 가능한 모듈이다"라는 관점의 원조이며, 2장에서 optimizer를 (state, update, cost) 객체로 세운 것은 정확히 이 관점을 미리 깔아 둔 것이다. [Miras]가 네 번째 설계 축으로 "memory learning algorithm(= optimizer)"을 놓을 때 [Miras §1], 이 축의 사상적 기원이 여기다.

셋째, **MAML**. Finn, Abbeel & Levine 2017 (arXiv:1703.03400)의 **MAML**(Model-Agnostic Meta-Learning)은 meta-변수를 단 하나, **초기값**으로 고른다: 새 task가 오면 초기값 $W_{\mathrm{init}}$에서 GD 몇 step으로 적응하고, "적응 후 성능"을 outer loss로 삼아 $W_{\mathrm{init}}$ 자체를 학습한다. hypergradient는 §4.2에서 유도한 그대로이며 — 식 (4-5)의 Hessian 항을 버린 변형이 FOMAML이다 — 학습이 끝난 $W_{\mathrm{init}}$은 "어느 task로든 몇 step 만에 갈 수 있는 출발점"이 된다. 이 아이디어의 이 라인 버전이 **meta-learned initial state $W_{\mathrm{init}}$**이다: memory의 초기 상태를 난수가 아니라 outer loop가 학습한 값으로 두는 것. [Titans]에서는 암묵적 세부였던 이것이 [TNT]에서는 구조의 기둥이 된다 — local memory가 shard 경계마다 "shared, learnable initial state $W_{\mathrm{init}}$"으로 reset되고 [TNT §4.1.1, Eq. 6], reset이 정보 폐기가 아니라 **좋은 출발점으로의 복귀**가 되는 것은 $W_{\mathrm{init}}$이 meta-learn되어 있기 때문이다. 독자의 세계로 옮기면 $W_{\mathrm{init}}$은 세션 시작마다 복원되는 golden snapshot — 모든 요청이 공유하는 초기 상태 이미지 — 이고, MAML은 그 이미지를 굽는 절차다.

**표 4-1 — learning-to-learn 계보: meta-변수의 선택**

| 계보 | outer loop가 학습하는 것($\Theta$ 쪽) | inner loop | 이 라인에서의 대응 |
|---|---|---|---|
| Schmidhuber 1987/1992/1993 | 자기 수정 규칙, fast-weight를 쓰는 slow net | fast weights의 갱신 | fast weight programming(→ 6장), self-modifying Titans(→ 16장) |
| Andrychowicz et al. 2016 | update rule 자체(작은 net) | 대상 모델의 학습 | learned $\eta_t,\beta_t,\alpha_t$ 생산 함수; "optimizer = 모듈" 관점(→ 2장) |
| MAML (Finn et al. 2017) | 초기값 $W_{\mathrm{init}}$ | task 적응 GD 몇 step | $W_{\mathrm{init}}$ (Titans 암묵 [Titans §3.1] → TNT load-bearing [TNT §4.1.1]) |
| 이 라인 (TTT 계열, → 8장) | 위 전부 + projection + inner objective의 파라미터 | token stream 위 online GD | 식 (4-2)–(4-3) |

전체 조망은 Hospedales et al. 2021 (arXiv:2004.05439)의 survey가 표준 참고 문헌이다. 표의 마지막 행이 말하듯, 이 라인은 계보의 어느 한 갈래가 아니라 **세 갈래 전부를 한 layer 안에 합류**시킨 것이다.

## 4.4 이 라인의 bilevel 구조: outer loop는 정확히 무엇을 배우는가

이제 6편 전체를 여는 열쇠 문장을 말할 수 있다: **inner optimizer의 hyperparameter들이, outer loop가 학습하는 data-dependent 함수가 된다.** 고전 훈련에서 learning rate·momentum 계수·weight decay·초기값은 사람이 고르는 hyperparameter였다. 이 라인에서는 그 각각이 $\eta_t=\eta(x_t;\Theta)$, $\beta_t=\beta(x_t;\Theta)$, $\alpha_t=\alpha(x_t;\Theta)$, $W_{\mathrm{init}}\in\Theta$로 바뀐다 — 사람이 아니라 outer loop가 고르고, 상수가 아니라 token마다 값이 바뀐다. [Titans]는 gate들이 $x_t$의 함수로 계산되는 data-dependent 계수임을 명시하고 [Titans §3.1], inner loss $\ell(W_{t-1};k_t,v_t)=\|\mathcal{M}(k_t;W_{t-1})-v_t\|_2^2$의 projection에 대해 "parameters $W_K$ and $W_V$ are hyperparameters"라고 못 박는다 [Titans §3.1]. [Atlas]는 같은 구조를 정의로 승격시킨다: inner loop에서는 memory module의 파라미터만 최적화되며 그때 나머지 전부는 고정된 hyperparameter이고, outer loop에서 그 나머지 — projection, MLP 등 — 가 최적화된다 [Atlas §2].

training 무경험 독자의 1번 질문 — "test-time learner의 learning rate는 누가 학습하는가?" — 에 이제 답한다. 핵심은 **함수와 값의 분리**다. 함수 $\eta(\cdot\,;\Theta)$의 파라미터는 $\Theta$의 일부로서 **outer loop가 pretraining 중에** 학습한다. 값 $\eta_t=\eta(x_t;\Theta)$는 **inner loop가 serving 중에** token마다 평가한다. serving에서 함수는 동결되어 있지만 값은 매 token 다르다 — "학습된 learning rate"라는 말은 언제나 함수에 대한 말이다. gate 생산 함수는 실제로는 작은 head(예: low-rank projection + activation)이므로, decode 경로에 GEMV 몇 개가 추가되는 비용으로 읽으면 된다.

**표 4-2 — 누가 무엇을 학습하는가 (이 라인의 표준 배치)**

| 구성 요소 | 기호 | 소속 | 움직이는 시점 |
|---|---|---|---|
| projection 행렬 | $W_K,W_V,W_Q$ | $\Theta$ (outer) | pretraining만; serving에선 동결 |
| gate 생산 함수 (inner lr / momentum / retention) | $\eta(\cdot;\Theta),\beta(\cdot;\Theta),\alpha(\cdot;\Theta)$ | $\Theta$ (outer) | 함수는 pretraining만; 값 $\eta_t,\beta_t,\alpha_t$는 serving 중 매 token 평가 |
| memory 초기 상태 | $W_{\mathrm{init}}$ | $\Theta$ (outer) | pretraining만; serving에선 reset 목적지 |
| backbone (attention, MLP, LN, embedding) | $\Theta$ | $\Theta$ (outer) | pretraining만 |
| memory 상태 | $W_t$ | fast (inner) | serving 중 매 token, 식 (M1)/(M2) |
| inner momentum buffer | $S_t$ | fast (inner) | serving 중 매 token (M2) |
| outer optimizer state | $m_t,h_t$ | outer의 부속 (→ 2장) | pretraining만; 배포물에 포함되지 않음 |

이 표를 기준으로 6편의 변주를 미리 읽어 두자. [Miras]는 표의 "inner objective $\ell$"과 retention 항을 설계 축으로 열고(attentional bias → 13장), [Atlas]는 inner 문제의 범위를 token 하나에서 sliding window로 넓히고 inner optimizer를 Muon으로 바꾼다(Omega rule → 14장). [TNT]는 outer 훈련의 경제학 — 어떤 chunk 크기로 unrolling을 자를 것인가 — 을 다루고(→ 15장), [NL]은 표의 행들을 아예 재해석한다: momentum과 AdaGrad조차 two-level nested optimization으로 분해되고 [NL §1], 모델과 훈련 절차 전체가 각자의 context flow를 가진 nested, multi-level optimization 문제들의 집합이 되며 [NL Abstract], pre-training 자체가 "context가 전체 pre-training data인 in-context learning"으로 재서술된다 [NL §1]. 두 loop는 K개 level의 스펙트럼으로 일반화되고 level마다 update frequency(→ 16장)가 붙는다. [Sleep]은 ICL을 meta-learning process로 보는 정식화 [Sleep §1]를 전제로, serving 중 동결이라는 $\Theta$의 지위 자체를 wake/sleep lifecycle(→ 17장)로 허문다 — "$\Theta$는 serving 중 불변"이라는 이 절의 규칙이 성립하는 마지막 논문이 TNT이고, 그 규칙의 해체가 라인의 종착점이라는 것까지 보이면 Part II의 지도는 완성이다.

> **[해설]** 표 4-2는 inference 엔지니어에게 배포 체크리스트로 읽힌다: 배포물에 들어가는 것은 $\Theta$ 전부(gate 생산 함수 포함)와 $W_{\mathrm{init}}$이고, 요청마다 새로 할당해야 하는 것은 $W_t$와 $S_t$(그리고 (M2)라면 그것이 전부)다. outer optimizer state $m_t,h_t$는 KV cache가 아니라 훈련 클러스터에 남는다. per-request 상태가 shared-weight batching을 깨뜨린다는 함의는 10장과 Part III의 주제다.

## 4.5 Attention은 이미 GD를 하고 있었다: ICL과 mesa-optimization

bilevel 구조가 이 라인의 발명품이 아니라는 정황 증거가 있다: **평범하게 훈련된 transformer 안에서 inner loop가 저절로 생긴다**는 연구들이다. Akyürek et al. 2023 (arXiv:2211.15661)은 in-context로 linear regression을 푸는 transformer가 GD나 ridge regression 같은 표준 학습 알고리즘을 내부적으로 구현할 수 있음을 구성적으로 보였고, von Oswald et al. 2023 (arXiv:2212.07677)은 linear self-attention layer의 weight를 적절히 두면 그 layer의 forward pass가 in-context 예제들에 대한 regression loss의 GD 한 step과 일치함을 보이고, 학습된 transformer가 실제로 그런 해에 도달한다고 보고한다.
<!-- TODO-VERIFY: von Oswald 2212.07677의 경험적 주장 범위(어떤 과제·어떤 아키텍처에서 GD와의 일치를 보였는지)를 원문 §4-5에서 확인 후 문장 조정. 확인 방법: arXiv:2212.07677 본문 실험 절. -->
후속작 von Oswald et al. 2023 (arXiv:2309.05858)은 이를 **mesa-optimization** — forward pass 안에서 내부 목적함수를 세우고 최적화하는, 훈련이 만들어낸 절차 — 로 명명하고, in-context regression 문제를 매 step 정확히 최적해까지 푸는 **Mesa-layer**를 제안했다. Atlas는 Mesa-layer를 "모든 과거 token에 대해 memory를 최적화하지만 훈련이 느린" 극한으로 규정하고 자신의 대조군으로 세운다 [Atlas §3, 각주 1].

이 결과들이 이 장의 문법에서 하는 말은 정확히 이것이다: softmax attention은 이미 일종의 inner 문제의 **non-parametric 해**다(KV cache가 곧 그 "풀이의 상태"라는 대응은 1장 Rosetta 사전의 첫 행이다). 그렇다면 6편의 라인은 무에서 유를 만드는 것이 아니라, attention이 **암묵적·비모수적·얕게** 하던 일을 **명시적·모수적($W_t$로 압축)·깊게**(deep memory, momentum, 학습된 gate) 만드는 프로젝트다. inner loop를 명시적으로 쓰는 순간 무엇을 얻는가 — objective를 갈아끼우고(13장), 범위를 넓히고(14장), optimizer를 바꾸고(14장), 훈련 경제학을 설계할(15장) 자유 — 가 Part II의 줄거리다.

## 4.6 Worked micro-example: 손으로 도는 두 개의 loop

스칼라 하나로 bilevel 전체를 돌려 보자. memory는 $1\times1$ 행렬(스칼라) $W$, 읽기는 $\mathcal{M}(k;W)=Wk$, inner loss는 이 예제에서만 계수 $\tfrac12$을 붙여 $\ell(W;k,v)=\tfrac12(Wk-v)^2$로 둔다(미분을 깔끔하게 하기 위한 장-국소 선택이다). outer 데이터는 다음 시나리오다: pair $(k,v)=(1,\,1)$을 memory에 쓴 뒤, query $q=1$로 읽어서 목표값 $v^{\mathrm{out}}=1$을 재현해야 한다. outer loss는 $\mathcal{L}=\tfrac12(W_1q-v^{\mathrm{out}})^2$. meta-변수는 두 개다: inner lr $\eta_{\mathrm{in}}$(초기값 $0.5$)와 초기 상태 $W_0=W_{\mathrm{init}}$(초기값 $0$).

**inner loop (forward).** momentary gradient는 $g_0=(W_0k-v)\,k=(0\cdot1-1)\cdot1=-1$. 한 step:

$$
W_1=W_0-\eta_{\mathrm{in}}\,g_0=0-0.5\cdot(-1)=0.5 .
$$

읽기: $y=W_1q=0.5$. outer loss: $\mathcal{L}=\tfrac12(0.5-1)^2=0.125$. 절반만 기억된 상태다 — $\eta_{\mathrm{in}}=0.5$는 이 key를 반만 overwrite한다.

**outer loop (hypergradient).** 식 (4-4)와 (4-5)에 숫자를 넣는다. 먼저 $\nabla_{W_1}\mathcal{L}=(y-v^{\mathrm{out}})\,q=-0.5$.

- $\dfrac{\partial\mathcal{L}}{\partial\eta_{\mathrm{in}}}=-\,g_0\cdot\nabla_{W_1}\mathcal{L}=-(-1)(-0.5)=-0.5.$
- Hessian은 $\nabla^2_W\ell=k^2=1$이므로 $\dfrac{\partial\mathcal{L}}{\partial W_0}=\big(1-\eta_{\mathrm{in}}k^2\big)\,\nabla_{W_1}\mathcal{L}=(1-0.5)\cdot(-0.5)=-0.25.$

두 번째 식에서 "gradient를 다시 미분하면 2차 미분이 나온다"가 숫자로 보인다: 괄호 안의 $1-\eta_{\mathrm{in}}k^2$가 바로 식 (4-5)의 Jacobian이다.

**outer step.** outer lr $\eta=1$의 GD로 $\eta_{\mathrm{in}}$만 갱신하면 $\eta_{\mathrm{in}}\leftarrow0.5-1\cdot(-0.5)=1.0$. inner loop를 다시 돌리면 $W_1=0-1\cdot(-1)=1$, $y=1$, $\mathcal{L}=0$. outer loop가 단 한 step 만에 "이 과제에는 완전 overwrite가 정답"임을 학습했다. 실제로 $\eta_{\mathrm{in}}=1/k^2$는 이 key에 대한 정확한 overwrite — 5장에서 delta rule이라 부르게 될 바로 그 지점 — 이며, 학습된 inner lr가 delta rule을 재발견한 셈이다.

**관찰 두 가지.** (1) $\eta_{\mathrm{in}}=1$이 된 뒤에는 $\partial W_1/\partial W_0=1-\eta_{\mathrm{in}}k^2=0$: 한 step에 완전히 overwrite되는 key에 대해서는 초기값으로 gradient가 아예 흐르지 않는다. $W_{\mathrm{init}}$이 학습할 거리가 있으려면 inner가 다 지우지 못하는 무언가 — 본 적 없는 key, 부분 갱신, 부족한 step 수 — 가 남아 있어야 한다는 것을 이 $0$이 말해 준다. (2) 차원을 $d$로 올리면 inner update는 $\Delta W=-\eta_{\mathrm{in}}(Wk-v)k^\top$, 즉 (오차)$\,k^\top$ 모양의 rank-1 outer product다 — 1장 Rosetta 사전의 "KV cache append를 rank-1 GEMM write로 읽기" 행이 여기서 그대로 재사용된다. 이 예제의 모든 산수는 그 GEMM의 $1\times1$ 특수 경우다.

## 4.7 Systems bridge: 정적 graph, truncation, 그리고 serving의 분리

이 장의 내용이 독자의 roofline 감각과 만나는 지점을 정리한다.

**첫째, unrolled inner loop는 컴파일 가능한 정적 graph다.** 식 (4-3)을 $L$번 펼친 graph는 같은 연산 블록의 반복이고 data-dependent 제어 흐름이 없다. 따라서 kernel fusion·재배치·tiling의 대상이 된다 — FlashAttention이 attention의 정적 graph를 tile 단위로 재조직했듯, 이 graph를 chunk 단위로 재조직한 것이 9장의 chunkwise-parallel training이다. 단 결정적 차이가 있다: FlashAttention tiling은 bit-exact지만, chunk는 gradient anchor를 chunk 시작 상태에 동결하는 **semantic 근사**로서 계산되는 함수 자체를 바꾼다(식 (M4), → 9장). §4.2의 truncated unrolling에서 본 "비용 대 편향" 트레이드오프가, truncation 길이 $\to$ chunk 크기 $C$로 이름만 바꿔 재등장하는 것이다.

**둘째, outer 훈련의 비용은 궤적의 저장이다.** inner loop를 관통하는 backward pass는 원칙적으로 $W_1,\dots,W_L$ 전 궤적을 activation memory에 요구한다(2장의 activation·recomputation 논의의 직계). naive하게는 상태 크기 $|W|$의 $L$배 — matrix memory $d\times d$에 $L{=}32\mathrm{K}$면 이미 감당 불가 — 이고, 그래서 실전은 chunk 경계 상태만 저장하고 chunk 내부를 재계산하는 쪽으로 간다(→ 9장). "훈련이 비싼 이유"의 절반은 FLOP이 아니라 이 궤적 저장이라는 감각을 여기서 얻어 두면 TNT의 훈련 경제학(→ 15장)이 바로 읽힌다.

**셋째, serving은 두 loop 중 하나만 가져간다.** 배포 후에는 outer loop가 존재하지 않는다 — $\nabla_\Theta$도, $m_t,h_t$도, 궤적 저장도 없다. 대신 inner loop가 decode 안으로 들어온다: 매 token, 작은 memory net의 forward + backward + update(1장 Rosetta 사전의 "decode = $C{=}1$의 per-token online write + read" 행). 이때의 backward는 autograd가 아니라 손으로 유도된 gradient의 고정 kernel이라는 점, 그 구체적 GEMM 수와 비용 모델은 8장과 10장이 담당한다.

**표 4-3 — 두 loop를 (state, update, cost) 객체로**

| | inner loop | outer loop |
|---|---|---|
| state | $W_t$ (+ $S_t$) — 요청마다 생성·소멸 | $\Theta$ + $m_t,h_t$ — fleet 전체에 하나 |
| update | 식 (M1)/(M2), token마다 | AdamW step(→ 2장), mini-batch마다, hypergradient 필요 |
| cost | decode 경로의 추가 FLOP·state RMW 트래픽 | pretraining 클러스터의 궤적 저장 + backward |
| 수명 | 한 시퀀스/세션 | pretraining 기간; serving에선 동결(해체는 → 17장) |

## 요약

- bilevel optimization은 outer 문제의 제약 안에 inner 최적화가 들어 있는 구조이며, 이 라인의 sequence model은 inner 문제를 argmin까지 풀지 않고 GD 궤적 자체를 출력으로 쓰는 trajectory 형태의 bilevel 문제다 (식 (4-2)–(4-3)).
- inner loop는 fast weights $W$를 inner loss $\ell$로, outer loop는 slow weights $\Theta$를 outer loss $\mathcal{L}$로 최적화한다. 두 loop는 시간축이 다르다: token마다 vs mini-batch마다, serving 중 vs pretraining 중.
- outer loop는 inner update의 computational graph를 unrolling해 backprop함으로써 hypergradient $\nabla_\Theta\mathcal{L}$을 얻고, 그 과정에서 inner loss의 Hessian이 나타난다 (식 (4-5)). truncation은 비용을 깎는 대신 hypergradient를 편향시키며, 9장의 chunk 크기 $C$가 같은 역할을 한다.
- meta-learning 계보의 세 갈래 — Schmidhuber의 자기 수정, learned optimizer, MAML의 학습된 초기값 — 가 이 라인에서 각각 self-modifying 구조(16장), 학습된 gate 생산 함수, $W_{\mathrm{init}}$(TNT의 reset 목적지)으로 합류한다.
- 열쇠 문장: inner optimizer의 hyperparameter들($\eta_t,\beta_t,\alpha_t$, projection, $W_{\mathrm{init}}$)은 outer loop가 학습하는 data-dependent 함수가 된다. "누가 학습하는가"는 항상 함수(outer, pretraining)와 값(inner, serving)을 나눠 답한다.
- 평범한 transformer의 ICL이 이미 암묵적 GD라는 결과들(Akyürek; von Oswald; mesa-optimization)이 이 라인의 정당화 근거이며, 6편은 그 암묵적 inner loop를 명시적·모수적·깊게 만드는 프로젝트다.

## 자가 점검 체크리스트

- [ ] 식 (4-2)–(4-3)을 백지에 쓰고, 각 기호가 $W$/$\Theta$ 어느 쪽 소속인지 표시할 수 있다.
- [ ] 1-step inner update에 대해 $\partial\mathcal{L}/\partial\eta_{\mathrm{in}}$과 $\partial\mathcal{L}/\partial W_0$를 유도하고, 후자에 Hessian이 나타나는 이유를 설명할 수 있다.
- [ ] "이 모델의 learning rate는 학습된다"는 문장을 함수/값 구분으로 풀어 말하고, serving 중에 무엇이 동결되고 무엇이 token마다 변하는지 답할 수 있다.
- [ ] MAML의 meta-변수 선택과 TNT의 $W_{\mathrm{init}}$ reset이 왜 같은 아이디어인지 설명할 수 있다.
- [ ] truncated unrolling의 트레이드오프를 말하고, chunk 크기 $C$가 왜 그 사촌인지 한 문장으로 연결할 수 있다.
- [ ] inner/outer loop를 inference 어휘로 옮길 수 있다: $\Theta$ = 컴파일된 커널의 상수(배포 시 고정), $W_t$ = KV cache형 runtime state, inner loop = backward가 들어온 decode loop, outer loop = 그 decode loop 전체를 미분하는 pretraining.

## 다음 장으로

이 장은 "누가, 언제 최적화하는가"를 답했지만, inner loss가 왜 하필 $\|\mathcal{M}(k;W)-v\|_2^2$ 꼴인지 — 즉 **무엇을** 최적화하는가 — 는 건드리지 않았다. 그 답은 60년 된 associative memory 이론에 있다: key에서 value를 연상해 내는 장치로서의 행렬, outer-product로 쓰고 곱셈으로 읽는 Hopfield–Hebbian 계보, 그리고 그 한계(crosstalk)를 고치는 delta rule. 5장은 이 계보를 복원해 inner loss의 기본형과 "memory가 넘친다"의 수학적 의미(capacity)를 확정하고, 그로써 6장에서 linear attention을 "결함 있는 associative memory"로 읽을 준비를 마친다.
