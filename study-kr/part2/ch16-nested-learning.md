# ch16. Nested Learning: deep learning 아키텍처라는 착시, 그리고 Hope

## 16.1 Bridge-in: TNT가 남긴 문제

[TNT](→ 15장)는 이 라인의 훈련 경제학 회차였다. deep memory의 chunkwise training(→ 9장)이 안고 있던 세 가지 문제 — 작은 chunk 훈련의 비효율, write/read 도메인 불일치, train/serve chunk-size mismatch — 를 hierarchical memory와 two-stage training으로 풀었고, 그 과정에서 하나의 구조를 남겼다. 큰 chunk로 천천히 갱신되는 global memory $W^{\mathrm{g}}$와, meta-learn된 초기 상태 $W_{\mathrm{init}}$로 주기적으로 reset되는 local memory들의 공존이 그것이다. 서로 다른 주기로 갱신되는 구성요소들, 그리고 느린 timescale에 저장된 지식이 빠른 timescale의 reset을 살아남는 구조 — 이것이 TNT가 심어 놓고도 정당화하지 않은 아이디어다. TNT에서 이 계층은 어디까지나 throughput을 위한 트릭이었다. 왜 update 주기의 계층이 모델 전체의 조직 원리여야 하는지, 그 계층 위에서 아키텍처와 optimizer는 서로 어떤 관계인지에 대한 대답은 없었다.

Part II의 사슬을 이 책의 기준 수식 (M2)로 요약하면 공백이 더 선명해진다. [Titans](→ 12장)는 (M2) 자체 — GD + momentum + weight decay가 sequence layer라는 등식 — 를 세웠고, [Miras](→ 13장)는 inner objective $\ell$과 retention을 설계 축으로 만들었으며, [Atlas](→ 14장)는 $\ell$의 범위(window)와 inner optimizer(Muon)를 바꿨고, [TNT](→ 15장)는 훈련 경제학을 바꿨다. 다섯 논문 모두 "sequence layer 하나"의 성분을 바꿨다. 그런데 그 layer를 훈련시키는 바깥의 AdamW, 그 AdamW의 momentum, gradient를 흘려보내는 backpropagation은 여전히 설계 공간 바깥의 고정된 배경이었다. 이 장의 논문은 질문의 단위를 바꾼다: layer가 아니라 **모델과 그 훈련 절차 전체**가 하나의 설계 대상이고, 그 전체는 서로 다른 주기로 자기 문맥을 압축하는 최적화 문제들의 중첩 시스템이라는 것이다.

이 장의 논문은 [NL] (*Nested Learning: The Illusion of Deep Learning Architecture*, arXiv:2512.24695; Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni, Google Research, v1 2025-12-31; NeurIPS 2025 발표본 존재)이다. [NL]은 이 라인의 이론 종합 회차다. TNT가 실용으로 발견한 "다중 timescale + reset 생존"을 존재론으로 선언하고, 그 위에서 세 가지를 생성한다: 더 표현력 있는 optimizer들(Delta Momentum, DMGD, DGD, M3), 주파수 스펙트럼 memory(**Continuum Memory System**), 그리고 자기 자신의 update 알고리즘을 학습하는 sequence block(**self-modifying Titans**). 세 산물의 결합이 **Hope**다.

## 16.2 문제의식과 논문의 핵심 주장

논문이 스스로 정의한 문제는 LLM의 정적임(static)이다 [NL §1]. 배포 후 LLM의 적응 가능한 부분은 in-context learning뿐이므로, 지식은 두 곳 중 하나에만 존재한다: (i) 지금 context window 안(attention의 KV cache), (ii) "end of pre-training" 시점에 동결된 MLP weights 안. 논문은 이를 anterograde amnesia — 발병 이후 새 장기 기억을 형성하지 못해 영원한 현재만을 사는 신경학적 조건 — 에 비유한다 [NL §1]. context에 들어온 정보는 결코 장기 저장소(feedforward layer)에 도달하지 못하고, window가 밀려나면 사라진다.

신경생리학이 두 개의 처방을 시사한다 [NL §1.1]. 첫째, 장기 기억 형성은 최소 두 단계의 consolidation을 거친다: 각성 중 즉시 일어나는 **online (synaptic) consolidation**과, 수면 중 replay로 일어나는 offline (systems) consolidation이다. [NL]은 명시적으로 전자만 다루고 후자를 범위 밖으로 선언한다 [NL §1.1] — 이 선언이 17장 [Sleep]으로 가는 인계선이다. 둘째, 뇌는 다중 timescale로 계산을 조직한다: 감각 처리의 빠른 gamma파(30–150 Hz)에서 memory consolidation의 느린 delta/theta파(0.5–8 Hz)까지, 주파수마다 다른 인지 기능이 결부된다. 반면 현대 deep learning 모델의 update frequency는 정확히 두 극단뿐이다: attention은 매 token 상태를 다시 계산하므로 $\infty$, MLP는 test time에 동결이므로 $0$ [NL §1.1, §6].

세 번째 관찰은 구조의 균일성이다 [NL §1.1]. 뇌의 memory는 특정 부위에 격리된 시스템이 아니라 영역들에 분산된 회로이고(반구절제술 후에도 핵심 네트워크가 재조직되어 기능한다), 신경 요소들은 재사용 가능하다. 반면 현대 스택은 attention + SSM + convolution + MLP의 이질적 조합으로 보인다. 논문의 진단: 이 이질성은 착시다. 우리가 최적화 문제의 **해**(예: attention이라는 닫힌형)만 보고 **문제**(그 해가 최적화하는 objective와 문맥)를 보지 않기 때문에 서로 달라 보일 뿐, 모든 구성요소는 gradient descent로 최적화되는 linear 또는 MLP associative memory이며 차이는 level·objective·update frequency뿐이라는 것이다 [NL §5.1]. 논문 제목의 "illusion"이 이것이다.

핵심 주장은 넷으로 정리된다. (1) 어떤 machine learning 모델이든 훈련 절차와 함께 하나의 **nested, multi-level, 병렬 최적화 문제 시스템**으로 표현되며, 각 문제는 자기 **context flow**(token, gradient, 상위 신호)를 자기 주기로 압축하는 associative memory다 [NL §3]. (2) 이 렌즈로 보면 backpropagation·momentum·Adam·AdaGrad·Muon은 전부 gradient들을 압축하는 associative memory이고, 특히 Adam은 특정 element-wise $\ell_2$ objective의 **최적** memory다 [NL §4, App B]. (3) 이 프레임은 기술(記述)용이 아니라 생성용이다: 새 optimizer들, CMS, self-modifying Titans가 프레임에서 도출된다 [NL §4.4–§4.5, §7, §8]. (4) 더 많은 layer가 아니라 더 많은 **level**이 computational depth와 continual learning을 사준다 — stacking의 새 차원 [NL §1.2, §3.2].

이 프레임의 부산물 하나가 이 책 독자에게 특히 중요하다. in-context learning은 emergent한 현상이 아니라 구조적 성질 — NL 표현에서 level이 2개 이상이면 반드시 생기는 성질 — 이라는 주장이다 [NL §6]. Transformer의 ICL은 token들에 대한 어떤 regression objective의 non-parametric 해라는 지위에서 나오고, recurrent 모델의 ICL은 하위 level의 parametric 학습에서 나온다. pre-training조차 "corpus 전체를 context로 하는 ICL"의 한 사례로 재분류된다 [NL §6].

> **[해설]** 논문 제목은 Merrill et al.의 state-tracking 논문 *The Illusion of State in State-Space Models*를 겨냥한 표현이다. 그 논문이 "SSM의 state는 보기보다 얕다"고 했다면, [NL]은 "아키텍처의 다양성이야말로 착시이고, 진짜 설계 변수는 level이다"라고 되받는다. 실제로 computational depth 한계(Merrill et al.)는 [NL]이 §1에서 layer stacking으로 풀리지 않는 문제 목록의 첫 항목으로 인용하는 근거다.

## 16.3 Core mechanism (통일 표기)

이 절은 [NL]의 재해석 사슬을 원문 순서대로 따라간다: 대수적 엔진(§16.3.1) → 훈련 자체의 memory화(§16.3.2) → level과 nested system의 형식화(§16.3.3) → level 간 knowledge transfer(§16.3.4) → optimizer 재해석(§16.3.5)과 새 optimizer(§16.3.6) → 아키텍처 재해석(§16.3.7) → 생성물 셋: CMS(§16.3.8), M3(§16.3.9), self-modifying Titans(§16.3.10) → Hope(§16.3.11). 모든 수식은 통일 표기이며, 원 표기와의 대응은 §16.3.12의 표 16-2에 있다.

### 16.3.1 대수적 엔진: associative memory 정의와 proximal/FTRL 등가

출발점은 [Miras]의 associative memory 정의를 그대로 수입한 [NL Def. 1]이다(정의 소유는 → 13장): key 집합 $\mathcal{K}\subseteq\mathbb{R}^{d_k}$와 value 집합 $\mathcal{V}\subseteq\mathbb{R}^{d_v}$가 주어질 때, associative memory는 사상 품질을 재는 objective $\tilde\ell$에 대해

$$
\mathcal{M}^\star \;=\; \arg\min_{\mathcal{M}}\ \tilde\ell\big(\mathcal{M}(K);\,V\big)
$$

로 얻어지는 연산자다. [NL]이 이 정의에 더하는 것은 두 가지다. 첫째, 용어의 신경심리학적 고정: **memory는 입력이 일으킨 신경 갱신이고, learning은 유효한 memory를 획득하는 과정**이다 [NL §3.1]. 이 정의에서는 gradient descent에 의한 어떤 level의 어떤 갱신도 memory다. 둘째, key와 value가 token일 필요가 없다는 명시다 — gradient, sub-sequence, 상위 신호 모두 가능하다 [NL §3.1]. 이때 어떤 최적화 문제(level)가 압축하는 데이터 스트림을 그 문제의 **context flow**라고 부른다. sequence block의 context flow는 token이고, optimizer의 context flow는 gradient다.

재해석의 대수적 엔진은 gradient descent의 두 가지 등가 형식이다. 한 스텝의 GD는 linearized objective + proximal 항의 argmin과 같다 [NL Eq. 2]:

$$
\theta_{t+1} \;=\; \arg\min_{\Phi}\ \Big\{\big\langle \nabla_\theta\,\tilde\ell(\theta_t; x_t),\ \Phi\big\rangle \;+\; \tfrac{1}{2\eta_t}\,\|\Phi-\theta_t\|_2^2\Big\},
$$

즉 "1차 Taylor 근사를 $\ell_2$ proximal로 규제하며 최소화"가 GD 한 스텝의 정체다. 스텝들을 누적하면(상수 $\eta$) follow-the-regularized-leader 형식이 되고 [NL Eq. 3], 그 해는 $\theta_{t+1}=\theta_1-\eta\sum_{s\le t}\nabla\tilde\ell(\theta_s;x_s)$다 — 3장에서 다룬 FTRL(→ 3장) 그대로다. 이후 [NL]의 모든 재해석은 이 치환을 반복한다: 어떤 update 식이 보이면 그것을 "어떤 objective의 proximal argmin"으로 되읽고, 그 objective와 context flow가 무엇인지 묻는 것이다.

### 16.3.2 훈련은 surprise에 대한 memory다: LSS와 backprop의 자기참조성

먼저 부호 관행을 국소 선언한다: 이 책의 전역 기호(→ §1.2 예약표)를 따라 **Local Surprise Signal (LSS)** 을 $u_t=\nabla_{y_t}\mathcal{L}$로 고정한다. [NL] 원문은 §3.1에서 같은 부호로 정의하되 GGD 문맥([NL Eq. 58–60])에서는 $-\nabla_{y_t}\mathcal{L}$를 $u_t$로 쓰기도 하므로, 이 장은 self-generated value가 필요한 자리에서 $-u_t$를 명시해 쓴다.

$y=\theta x$인 1-layer linear layer를 task loss $\mathcal{L}$(예: next-token prediction)로 훈련하는 SGD 한 스텝은 다음처럼 인수분해된다 [NL Eq. 8]:

$$
\theta_{t+1} \;=\; \theta_t \;-\; \eta_{t+1}\,\nabla_\theta\,\mathcal{L}(\theta_t;x_{t+1})
\;=\; \theta_t \;-\; \eta_{t+1}\, u_{t+1}\, x_{t+1}^\top,
\qquad u_{t+1}=\nabla_{y_{t+1}}\mathcal{L}(\theta_t;x_{t+1}).
$$

여기서 $\nabla_\theta\mathcal{L}$은 12장의 surprise — 지금 입력이 기존 저장 내용과 얼마나 다른가 — 그대로이고, $u_{t+1}$은 그 surprise의 출력 공간 버전, 즉 "이 입력에 대한 모델의 예측이 얼마나 놀라운가"를 재는 국소 예측 오차다 [NL §3.1]. gradient가 rank-1 outer product $u\,x^\top$로 쪼개진다는 사실이 재해석의 문을 연다: 이 update는 proximal 형식으로

$$
\theta_{t+1} \;=\; \arg\min_{\Phi}\ \big\langle \Phi\,x_{t+1},\ u_{t+1}\big\rangle \;+\; \tfrac{1}{2\eta_{t+1}}\|\Phi-\theta_t\|_2^2
\tag{16-1}
$$

과 동치이며 [NL Eq. 9], 이것은 Def. 1의 associative memory — 데이터 포인트 $x_{t+1}$을 key로, 그 LSS $u_{t+1}$을 value로 하는 dot-product objective의 memory — 다. 요약하면: **backpropagation으로 훈련되는 layer는 "각 입력 → 그 입력에 대한 예측 오차"의 사상을 weights에 압축하는 memory다** [NL §3.1, §4.1].

깊은 모델로 가도 형태는 같다. $L$층 MLP $\{W_\ell\cdot+b_\ell\}_{\ell=1}^{L}$에서 backprop은 layer별 gradient를

$$
\frac{\partial \mathcal{L}}{\partial W_\ell} \;=\; \delta_\ell\, \hat x_{\ell-1}^\top,
\qquad
\delta_\ell \;=\; J_{\phi_\ell}(z_\ell)^\top\, W_{\ell+1}^\top\, \delta_{\ell+1}
$$

로 계산한다 [NL Eq. 29]. $z_\ell$은 pre-activation, $\hat x_\ell=\phi_\ell(z_\ell)$은 layer 출력, $J$는 activation의 Jacobian, $\delta_\ell$은 layer $\ell$의 backprop 오차(국소 오차 신호)다(→ 2장). 따라서 layer $\ell$의 GD update는 식 (16-1)과 같은 꼴의 proximal argmin — "layer 입력 $\hat x_{\ell-1}$을 국소 오차 $\delta_\ell$에 사상하는 memory" — 이 된다 [NL Eq. 31]. 훈련이란 각 layer가 (입력, 국소 오차) 쌍들을 자기 parameter에 압축하는 과정이다.

여기서 [NL]이 가장 힘주어 말하는 미묘함이 나온다. 식 (16-1)을 보고 "backprop은 gradient들에 대한 linear attention이다"라고 결론지으면 틀린다 [NL §4.1]. linear attention(→ 6장)에서는 key와 value가 memory 상태와 무관하게 주어지므로 recurrence가 병렬화된다. 그러나 backprop의 value는 $\delta_\ell$이고, 이것은 memory 자신의 현재 상태가 만들어 낸다:

$$
v_t \;=\; f_{W_t}(x_t) \;=\; -\,u_t \;=\; -\nabla_{y_t}\mathcal{L}(W_t;x_t)
\qquad [\text{NL Eq. 58}]
$$

— 즉 backprop은 자기 훈련 target을 스스로 생성하는 **self-referential** memory(Schmidhuber 1993)이며, 바로 그 성질 때문에 linear recurrence처럼 scan으로 병렬화되지 않는다 [NL §4.5]. 이 문장은 이 장의 systems 논의 전체를 지배한다: §16.4에서 보겠지만, Hope의 chunkwise 훈련은 정확히 이 자기참조성을 chunk 경계 snapshot으로 절단하는 타협이다.

> **[해설]** inference 독자의 어휘로 옮기면: pre-training은 "corpus 전체를 context로 하는 prefill"이고, 각 layer는 KV cache 없이 (입력 → 오차) 쌍을 weights에 압축하는 memory다(→ 1장 Rosetta). 2장이 optimizer를 (state, update, cost) 객체로 만들었을 때 이 관점의 절반에 이미 도달해 있었다 — [NL]은 나머지 절반, 즉 "그 객체의 state가 곧 한 level 위의 memory"라는 재명명을 수행한다.

### 16.3.3 update frequency, level, 그리고 NSAM

여러 최적화 문제로 분해된 모델에 질서를 주는 것이 **update frequency**다. [NL Def. 2]: 어떤 구성요소 $A$ — weights나 momentum 같은 parametric 요소든, attention block 같은 non-parametric 요소든 — 의 frequency $f_A$는 단위 시간당 갱신 횟수다(단위 시간 = 데이터 포인트 하나에 대한 갱신 한 번). attention은 매 token 상태를 다시 계산하므로 $f=\infty$, 동결된 MLP는 $f=0$이다 — 이것이 §16.2에서 말한 "두 극단"의 형식화다. 순서는 연산자 $A\succ B$로 준다: $f_A>f_B$이거나, $f_A=f_B$이되 $B$의 시점 $t$ 상태 계산이 $A$의 시점 $t$ 상태를 요구하면 $A$가 빠르다. 어느 쪽도 성립하지 않으면 같은 **level** — 같은 주기, 상호 독립 계산 — 이다. Adam의 1차/2차 moment가 대표적 동률 사례다: 둘 다 매 스텝 갱신되지만 서로의 계산을 요구하지 않으므로 한 level에 병렬로 놓인다 [NL Def. 2, App B]. level은 이 순서로 정렬되며, **높은 level일수록 낮은 frequency**다.

이제 시스템 전체가 정의된다. **Nested System** [NL Def. 3]: $K$개의 정렬된 level이 있고, level $k$는 최적화 문제들의 집합 $\{(L^{(k)}_i,\ \mathcal{C}^{(k)}_i,\ \Theta^{(k)}_i)\}_{i=1}^{N_k}$ — objective, context flow, feasible parameter 집합의 삼중항 — 로 구성되며, 각 문제는 식 (16-1) 꼴의 proximal GD로 최적화된다 [NL Eq. 19]. 단 일부 "box"는 non-parametric 해(예: attention)로 풀리는 것이 허용된다 [NL §3.2]. **NSAM (Nested System of Associative Memories)** [NL Def. 4]는 모든 문제의 context가 key-value 쌍 집합인 특수형이고 [NL Eq. 20], Appendix A의 일반형 [NL Def. 6, 7]은 proximal 스텝 안의 선형화를 full loss로 되돌린다. 그리고 **Neural Learning Module**: 아키텍처와 그 훈련/최적화 과정을 하나의 NSAM으로 함께 표현한 것이 설계의 단위이며, 모델의 행동은 시스템 전체의 성질이다 — 같은 Transformer라도 SGD로 훈련한 것과 Adam으로 훈련한 것은 **다른 module**이다 [NL §3.2].

정의가 실제로 어떻게 작동하는지, 논문의 계단식 예제 네 개가 보여 준다 [NL §3.1–§3.2].

- **(a) 1-layer MLP + GD** = 1-level: 데이터를 LSS에 사상하는 memory 하나 (§16.3.2).
- **(b) 1-layer MLP + momentum GD** = 2-level [NL Eq. 10–13]: momentum buffer가 gradient들을 압축하는 inner memory이고(자세한 전개는 §16.3.5), outer는 그 상태를 weights에 적용한다. 갱신 주기는 같지만 weights의 계산이 momentum 상태를 요구하므로 momentum이 더 빠른(낮은) level이다.
- **(c) linear attention + GD** = 2-level [NL Eq. 14–18]: outer는 projection $W_K,W_V,W_Q$를 GD로 최적화하고, inner는 recurrence $W_t=W_{t-1}+v_tk_t^\top$를 돈다. 이 recurrence 자체가 $\tilde\ell(W;k_t,v_t)=-\langle Wk_t,v_t\rangle$에 대한 GD 1스텝($\eta_t=1$)이다(→ 6장). 두 gradient flow는 서로를 관통하지 않는다: projection의 최적화에서 $\mathcal{M}$에 대한 backprop은 없고, inner에서 projection은 동결 hyperparameter다 — FWP(→ 6장)의 slow/fast 구분과 정확히 겹친다 [NL §3.1].
- **(d) linear attention + momentum GD** = 3-level: (b)와 (c)의 합성 [NL §3.1].

이 사다리의 정점에 이 논문에서 가장 교육적인 구성물이 있다. **AdaTransformer**(논문의 별칭 Linear Attention++) [NL §3.2, Fig. 3, Eq. 69]: Transformer block과, MLP block을 linear attention으로 치환한(key/value는 앞단 attention 것을 공유, 초기 memory 상태는 meta-learn) block을 비교하면, 출력 식은 대수적으로 동일하다:

$$
y_{\mathrm{block}} = y_{\mathrm{attn}}\,W_{\mathrm{MLP}}
\qquad\text{vs}\qquad
y_{\mathrm{block}} = y_{\mathrm{attn}}\,W_{\mathrm{LinAttn}}.
$$

유일한 차이는 **두 weight가 사는 level**이다. $W_{\mathrm{MLP}}$은 level 1(f=0)에 있어 context에 대해 불변이고, $W_{\mathrm{LinAttn}}$은 level 2에서 $W_t=W_{t-1}+v_tk_t^\top$로 in-context 갱신되며 그 초기값 $W_{\mathrm{init}}$이 level 1에서 NTP objective로 meta-learn된다. 따라서 **recurrent memory module이란 level 하나가 추가된 MLP block이고, hybrid 아키텍처란 일부 MLP block에 in-context 적응이 생긴 Transformer다** [NL §3.2, §6]. "attention + SSM의 이종 결합"이라는 통념이 이 프레임에서 해소되는 방식이 이것이다.

여기서 따름정리 하나가 12장의 개념을 회수한다. 초기 상태가 meta-learn되지 **않는** 경우($W_0=0$인 초기 linear Transformer들), 그 block에는 pre-training 지식을 담는 persistent memory(→ 12장)가 없다. 이때 관행적으로 붙는 출력 gating의 linear layer가 바로 그 빠진 persistent memory 역할을 대신한다는 것이 [NL §5]의 gating 해석이다 — "gating이 왜 도움이 되는가"라는 경험적 수수께끼에 대한 구조적 답변이다.

### 16.3.4 level 간 knowledge transfer: 다섯 가지 메커니즘

문제와 주기를 정했으면 남는 설계 축은 level들이 서로 어떻게 지식을 주고받는가다. [NL §3.3]은 두 block $B^{(0)}$(저주파·상위)과 $B^{(1)}$(고주파·하위) 사이의 **knowledge-transfer taxonomy** 다섯 가지를 제시한다.

**표 16-1 — 다섯 가지 knowledge transfer 메커니즘 [NL §3.3]**

| # | 메커니즘 | 형식 | 대표 사례 |
|---|---|---|---|
| 1 | direct parametric conditioning | $\mathcal{M}^{(0)}(\cdot):=\mathcal{M}^{(0)}(\cdot;\Theta^{(1)})$ 또는 $\mathcal{M}^{(0)}(\cdot;\mathcal{M}^{(1)}(\cdot))$ [NL Eq. 24–25] | FWP: fast weight의 출력이 slow net의 출력을 조건화(→ 6장) |
| 2 | direct non-parametric conditioning | $\mathcal{M}^{(0)}(\cdot;\mathcal{C}^{(1)})$ [NL Eq. 27] | softmax attention이 raw context에 조건화 |
| 3 | backpropagation across levels | 같은 gradient flow, 다른 갱신 주기 | CMS의 level 연결(§16.3.8) |
| 4 | initialization (MAML형) | $\Theta^{(1)}_0=\arg\min_\Phi \mathbb{E}_{\mathcal{C}\sim\mathcal{C}^{(0)}}\big[\ell(\mathcal{M}^{(1)}(\cdot;\Phi),\mathcal{C})\big]$ [NL Eq. 28] | meta-learn된 $W_{\mathrm{init}}$ (Titans/TTT/Atlas) |
| 5 | generation | weight 생성(hypernetwork) 또는 context 생성 | **아키텍처가 optimizer의 context(gradient)를 생성** |

1·2번의 공통 특징은 **backprop이 level을 건너지 않는다**는 것이다: 각 block은 상대의 상태를 자기 문제의 hyperparameter로 취급한다 [NL §3.3]. 3번은 반대로 두 상태가 같은 gradient flow에 있되 갱신 주기만 다르다 — §16.3.8의 CMS가 이 배선을 쓴다. 4번은 4장의 MAML(→ 4장)이 그대로 분류되는 자리이며, 이 라인이 [Titans]부터 지고 온 meta-learned initial state $W_{\mathrm{init}}$(→ 12장, 15장)의 공식 좌석이다. [NL §6]은 이 지점에서 라인 내부의 계보 차이를 짚는다: Titans·Atlas·Miras·TTT 같은 deep memory 모델들은 4번 transfer(고주파 level의 지식이 초기값 학습을 통해 저주파 level로)를 갖지만, 대부분의 linear RNN에는 level 간 transfer가 아예 없다.

5번이 이 논문 고유의 한 수다. hypernetwork는 익숙한 사례지만, 결정적 예는 이것이다: **아키텍처는 optimizer의 context — gradient들 — 를 생성한다** [NL §3.3, §6]. 이것은 learned optimizer에 국한된 말이 아니라 vanilla GD/Adam에도 성립하는 사실 서술이다. 아키텍처가 다르면 생성되는 gradient의 분포·패턴이 다르므로, gradient를 압축하는 memory(= optimizer)의 최적 설계도 달라져야 한다 — **architecture-specific optimizer**라는, 논문이 명명만 하고 납품하지 않는 미래 과제의 근거가 여기서 나온다.

정리하면 neural learning module의 설계는 두 결정으로 환원된다 [NL §3.3]: (1) 최적화 문제들과 그 주기의 설계(NSAM의 구성), (2) level 간 knowledge transfer의 설계. meta-learning, MAML, hypernetwork, learned optimizer는 전부 이 두 축 위의 특수 선택지로 분류된다.

### 16.3.5 optimizer 재해석: momentum, Adam, AdaGrad, Muon

**momentum은 gradient들에 대한 Hebbian memory다.** layer $\ell$의 momentum GD를 통일 표기로 쓰면 [NL Eq. 33]

$$
W_{\ell,t+1} \;=\; W_{\ell,t} \;+\; m_{t+1},
\qquad
m_{t+1} \;=\; \beta\, m_t \;-\; \eta_{t+1}\, \delta_\ell\, \hat x_{\ell-1}^\top
\tag{16-2}
$$

이다(원문은 decay에 $\alpha$를 쓴다 — 통일 retention gate $\alpha_t$와 무관하므로 이 책은 $\beta$로 개명, 표 16-2; 부호는 (M2) 관행으로 고정, 알고리즘 동일). $\beta=1$이면 $m$의 갱신은 $\min_m\ \langle m\,\hat x_{\ell-1},\ \delta_\ell\rangle$의 GD 1스텝이고, $\beta\neq 1$은 여기에 $m$에 대한 $\ell_2$ regularization을 더한 것과 같다 [NL §4.2, Eq. 34]. 즉 momentum은 과거 gradient들을 자기 parameter에 압축하는 **value-less(Hebbian) associative memory** — 하나의 low-pass filter — 다. 2장에서 (state, update, cost) 객체였던 $m_t$가 여기서 "weights보다 한 level 아래(빠른 쪽)의 memory"로 재명명된다. 12장에서 관찰이었던 **momentum-as-memory**가 이 장에서 정리(theorem적 지위)로 승격되는 지점이다. 전체 구조는 2-level이다: inner가 gradient들을 momentum memory에 압축하고, outer가 그 상태를 weights에 적용한다 [NL §3.1].

memory라면 capacity를 물을 수 있다. decay $\beta=0.9$일 때 $i$스텝 전 gradient의 기여는 $\beta^i(1-\beta)$이고, 누적 기여를 계산하면 **마지막 6개 gradient가 전체 기여의 50% 이상, 마지막 43개가 99% 이상**을 차지한다 [NL §4.3]. 43스텝보다 먼 과거는 1% 미만 — momentum은 loss landscape에 대한 장기 기억이 아니라 최근 기억이다. 이것이 continual learning에서 실패로 나타난다: task들의 gradient가 직교 방향 $\{u_i\}$에 사는 설정 [NL Eq. 45]에서 task $t$를 오래 최적화하면 momentum이 $u_t$ 방향으로 완전히 이동해 옛 gradient subspace의 기억을 잃고, optimizer는 이전 task를 훼손하는 방향을 피할 정보가 없어진다. 논문의 진단이 정확하다: **이것은 모델 capacity의 실패가 아니라 optimizer의 memory management 실패다** [NL §4.3]. 그리고 따름정리 하나 [NL §4.5]: "end of pre-training"에서 우리는 관행적으로 optimizer state를 버리는데, 그 순간 loss landscape에 대해 momentum이 저장한 지식이 통째로 사라진다 — continual learning module이라면 이 저주파 level들도 보존 대상이라는 것이 NL의 답이다.

**Adam은 특정 objective의 최적 memory다** [NL App B]. element-wise memory $m$에 대해 objective

$$
\tilde\ell_t \;=\; \sum_{i=1}^{t}\ \big\|\, m \odot g_{i+1} \;-\; P_t \,\big\|_2^2 \;+\; \lambda\,\|m\|_F^2,
\qquad g_{t+1} = -\nabla_{W}\mathcal{L}(W_t;x_{t+1})
\qquad [\text{NL Eq. 101}]
$$

을 상정하면 — gradient들을 어떤 전역 통계 $P_t$에 사상하라는 $\ell_2$ regression — 닫힌형 최적해는 $m^* = (h+\lambda I)^{-1}\odot \tilde m\odot P_t$이고, 여기서 $\tilde m_{i+1}=\tilde m_i+\beta_1 g_{i+1}$(1차 moment), $h_{i+1}=h_i+\beta_2\, g_{i+1}^2$(2차 moment)다 [NL Eq. 102]. $P_t$를 gradient의 "분산" $\sqrt{\sum_i g_i^2}$로 고르면 update가 $W_{i+1}\approx W_i-\tfrac{\eta}{\sqrt{\beta_2}}\,\tilde m_{i+1}\big/\big(h_i^{1/2}+\varepsilon\big)$로 정리되어 **Adam이 그대로 나온다** [NL Eq. 105]. 같은 objective의 outer-product 버전($\|m\,g^\top - P\|^2$, $h_{i+1}=h_i+\beta_2\, g\,g^\top$)은 AdaGrad-with-momentum을 회수하고 [NL Eq. 106–111], RMSProp·SignSGD·NAdam·AMSGrad·RAdam·Lion은 Adam과의 알려진 관계로, Shampoo·SOAP은 AdaGrad의 preconditioning 근사라는 관계로 따라온다 [NL §4.2]. 1차/2차 moment가 같은 주기·상호 독립이라 한 level의 병렬 memory 두 개가 된다는 것은 §16.3.3에서 본 대로다.

> **[평가]** 이 역공학들은 존재 논증이지 유일성 결과가 아니다. Eq. 101의 objective와 $P_t$의 선택은 Adam이 나오도록 상정된 것이고, 논문도 이를 정리로 주장하지 않는다. "Adam은 최적 associative memory다"는 항상 "그 특정 element-wise $\ell_2$ objective에 대해"라는 한정과 함께 읽어야 한다. 프레임의 가치는 Adam의 새 도출이 아니라, 다음 소절에서 보듯 **변형을 체계적으로 생성하는 문법**을 준다는 데 있다.

**preconditioning과 Muon.** preconditioned GD $W_{t+1}=W_t-\eta_{t+1}P_{t+1}^{-1}g_{t+1}$ [NL Eq. 38]의 $P$는 "gradient를 선택된 좌표계로 사상하는 법"을 내부 objective $\min_P \tilde\ell(P(\hat g);g)$로 배우는 nested memory로 읽힌다 [NL Eq. 39–41]. Muon(→ 2장에서 객체로, → 14장에서 inner optimizer로) $W_{t+1}=W_t+\mathrm{NS}_\kappa(m_{t+1})$ [NL Eq. 42]의 경우 좌표계는 직교 공간이다: orthogonalization objective

$$
\tilde\ell\big(P(g);g\big) \;=\; \big\|P(g)^\top P(g)-I\big\|_F^2
\qquad [\text{NL Eq. 43}]
$$

를 $O_0=g$에서 GD 1스텝(내부 step size $\zeta$)으로 풀면 $O_{i+1}=O_i-\zeta\big(O_i-g+2\,O_i(O_i^\top O_i-I)\big)$가 되어 Newton–Schulz의 3차 다항 반복이 그대로 나온다 [NL Eq. 44]. 결론: **Muon의 NS 반복은 momentum 갱신 한 번당 $\kappa$스텝을 도는 내부 최적화 level이다.** [NL §6]은 이를 "neuron당 더 많은 계산"이라 부른다 — level 추가가 CMS 같은 memory 계층만이 아니라 계산 심도의 증폭기이기도 하다는 예시다.

### 16.3.6 프레임의 생성적 사용: 새 learning rule들 — Delta Momentum, DMGD, DGD, GGD

재해석이 옳다면 optimizer 설계는 memory 설계와 같은 문법을 따라야 한다: association을 바꾸고, objective를 바꾸고, memory 구조를 바꾸고, feature map을 바꾸고, 출력 비선형성을 바꾸고, 마지막으로 learning rule 자체를 바꾼다 [NL §4.4–§4.5]. 6장과 13장에서 sequence layer에 대해 했던 조작들이 optimizer에 하나씩 이식된다.

- **preconditioned momentum** [NL Eq. 47]: momentum의 value를 $P_i$로 두면 $m_{i+1}=m_i-\eta_i\,P_i\,\nabla_W\mathcal{L}(W_i;x_i)$ — momentum이 (preconditioner ↔ gradient)의 사상을 배우는 memory가 된다.
- **Delta Momentum** [NL Eq. 48–49]: momentum의 내부 objective를 dot-product에서 $\ell_2$ regression $\|m\,g_i^\top - P_i\|_2^2$로 바꾸면($g_i=\nabla_W\mathcal{L}(W_i;x_i)$) update는 delta rule(→ 5장) 꼴이 된다:
  $$
  m_{i+1} \;=\; \beta_{i+1}\, m_i \;-\; \eta_i\,\big(m_i\, g_i - P_i\big)\, g_i^\top .
  $$
  Hebbian momentum과 달리 update가 자기 현재 상태에 의존하므로 **gradient-dependent decay**가 생기고, 제한된 capacity($O(N)$)의 관리가 좋아진다 — linear attention→DeltaNet 전이(→ 6장)의 optimizer 버전이다. 논문은 진동 곡률 toy $\psi(r,\theta)=r^2+k\,(r-\theta+\alpha\sin(\omega r))^2$ [NL Eq. 53]에서 표준 momentum보다 빠른 수렴을 보인다 [NL Fig. 4]: landscape가 고주파로 바뀌는 구간에서 표준 momentum(가중 평균)은 무관한 과거 gradient에 끌려다니지만, delta momentum은 필요할 때 감쇠·정지한다.
- **Deep Momentum Gradient Descent (DMGD)** [NL Eq. 50]: momentum을 행렬에서 MLP로 승격한다 — $W_{i+1}=W_i+m_{i+1}(u_i)$, $m_{i+1}=\beta_{i+1}m_i-\eta_i\,\nabla\tilde\ell^{(2)}(m_i;u_i,1)$, $u_i=\nabla_W\mathcal{L}(W_i;x_i)$. deep memory(→ 12장)가 token에 대해 했던 것을 gradient에 대해 하는 것이다: 과거 gradient의 비선형 사상까지 저장할 capacity.
- **higher-order feature map** [NL Eq. 51]과 **비선형 출력** [NL Eq. 52]: 각각 $\phi(\nabla\mathcal{L})$로 key를 들어올리는 것(→ 14장의 $\phi_p$와 같은 문법), 그리고 $W_{i+1}=W_i+\sigma(m_{i+1}(u_i))$. 후자에서 $\sigma=\mathrm{NS}$, $m$을 linear로 두면 **Muon이 특수 사례로 회수된다** — Muon은 "비선형 출력을 단 Hebbian momentum"이었던 것이다.

이 소절의 정점은 learning rule 자체의 교체다. **Delta Gradient Descent (DGD)** [NL §4.5, Eq. 56–57]: 식 (16-1)의 dot-product objective는 각 데이터 포인트(gradient)를 상태와 무관하게 취급한다 — i.i.d. 표본이라면 합리적이지만, 고도로 상관된 token 공간에서는 낭비다. objective를 $\ell_2$ regression으로 바꾸면 ($u_t=-\nabla_{y_t}\mathcal{L}(W_t;x_t)$)

$$
W_{t+1} \;=\; \arg\min_{\Phi}\ \tfrac{1}{2}\big\|\Phi\, x_t - u_t\big\|_2^2 \;+\; \tfrac{1}{2\eta_t}\|\Phi-W_t\|_2^2
\qquad [\text{NL Eq. 56}]
$$

이고, 입력이 정규화되어 있으면($\|x_t\|_2=\lambda$ — normalization layer가 있는 신경망이면 성립) Sherman–Morrison lemma $(x_tx_t^\top+\eta_t I)^{-1}=\tfrac{1}{\eta_t}\big(I-\tfrac{1}{\lambda^2+\eta_t}x_tx_t^\top\big)$ [NL Eq. 117, App C]로 닫힌형이 나온다:

$$
W_{t+1} \;=\; W_t\big(I \;-\; \eta_t'\, x_t x_t^\top\big) \;-\; \eta_t'\,\nabla_{y_t}\mathcal{L}(W_t;x_t)\, x_t^\top,
\qquad \eta_t'=\frac{\eta_t}{1+\eta_t}.
\tag{16-3}
$$

GD에 **data-dependent decay** $(I-\eta_t'x_tx_t^\top)$가 자동으로 붙는다 — 지금 입력과 상관된 방향의 기존 저장 내용을 정확히 그만큼 지우고 쓰는, 표본 간 의존을 반영하는 rule이다. 13장의 어휘로는 attentional bias를 dot-product에서 $\ell_2$로 바꾼 것이고, 5장의 어휘로는 delta rule의 재발명이되 이번에는 **learning rule의 층위**에서다. 그리고 일반화: **Generalized Gradient Descent (GGD)** [NL Def. 5, Eq. 59–60]는 self-generated value $u_t=f_{W_t}(x_t)$를 갖는 임의의 self-referential memory

$$
W_{t+1} \;=\; \arg\min_W\ \tilde\ell(x_t, u_t) \;+\; \mathrm{Ret}\big(W,\ \{W_i\}_{i=t-c+1}^{t}\big)
$$

의 family다($\mathrm{Ret}$은 최근 상태들 근방에 해를 묶는 retention 항 — 표기와 이론은 → 13장). backprop(GD·DGD 포함)이 이 family의 원소이고, 같은 형식을 momentum에 적용한 것이 **Generalized Momentum (GM)**이다 — 단, momentum은 key/value가 하위 level에서 주어지므로 self-referential이 아닌 conventional memory라는 구분이 붙는다 [NL §4.5].

### 16.3.7 아키텍처 재해석: 카탈로그의 재확인

[NL §5]는 [Miras]의 결과(→ 13장)를 NSAM 어휘로 재수록하는 절이므로 여기서는 자리만 확인한다. softmax attention은 $\mathcal{M}^\star=\arg\min_{\mathcal{M}}\sum_{i=1}^{L}s(k_i,q)\|v_i-\mathcal{M}\|_2^2$의 **non-parametric** 해(Nadaraya–Watson, → 8장·13장)이고 [NL Eq. 62], window를 최근 $c$ token으로 제한하면 sliding-window attention이다 [NL Eq. 63]. Hebbian rule의 RNN들(linear attention, RetNet, RWKV)은 dot-product objective + weight decay의 GD [NL Eq. 64], delta rule의 RNN들(DeltaNet, Longhorn, RWKV-7)은 $\ell_2$ objective + local retention의 SGD [NL Eq. 65], OjaNet은 $-2\langle Wk_t,v_t\rangle+\|W^\top v_t\|_2^2$의 GD [NL Eq. 66–67], Omega rule(→ 14장)은 캐시된 window에 대한 비-online 갱신 [NL Eq. 68], $L_p$ attentional bias는 → 13장 — 전부 이 책 §1.6 카탈로그와 1:1이다. [NL]의 추가 기여는 목록이 아니라 **자리 지정**이다: 이들 전부가 pre-training보다 한 level 위(빠른 쪽)의 문제이며, key/value/query를 공급하는 projection들은 그 아래 level에서 최적화된다 [NL §5]. 여기서 §16.2의 균일성 명제가 완결된다: 모든 구성요소는 linear 또는 deep feedforward memory이고, 차이는 level·objective·learning rule뿐이다 — "아키텍처의 이질성은 최적화 문제의 해만 보는 데서 오는 착시다" [NL §5.1].

### 16.3.8 Continuum Memory System (CMS)

기존 backbone의 memory는 이분법이다: sequence 축을 능동적으로 섞는 working memory(attention)와, pre-training 지식의 persistent 저장소인 feedforward layer(f=0) [NL §7]. **Continuum Memory System (CMS)** 은 이 이분법을 update frequency의 스펙트럼으로 일반화한다 [NL §7.1]. CMS는 MLP block의 사슬 $\mathrm{MLP}^{(f_1)},\dots,\mathrm{MLP}^{(f_k)}$이고, level $\ell$의 chunk 크기는

$$
C^{(\ell)} \;:=\; \frac{\max_i f_i}{f_\ell}\; C
$$

— 즉 level $\ell$은 $C^{(\ell)}$ token마다 한 번 갱신된다. forward는 단순 합성 $y_t=\mathrm{MLP}^{(f_k)}\big(\mathrm{MLP}^{(f_{k-1})}(\cdots \mathrm{MLP}^{(f_1)}(x_t))\big)$ [NL Eq. 70]이고, 갱신은 이 책의 표준형 (M5) 그대로다 [NL Eq. 71]:

$$
\theta^{(\ell)}_{i+1} = \theta^{(\ell)}_{i} - e_{i,\ell},
\qquad
e_{i,\ell} = \begin{cases}
\displaystyle\sum_{t=i-C^{(\ell)}+1}^{i} \eta^{(\ell)}_t\, \varepsilon\big(\theta^{(\ell)}_t; x_t\big) & i \equiv 0 \pmod{C^{(\ell)}}\\[2pt]
0 & \text{otherwise},
\end{cases}
$$

여기서 $\varepsilon(\cdot)$은 임의 optimizer의 error 항이다(GD면 $\nabla_\theta\mathcal{L}$; 실험에서는 단순 gradient 누적). 두 가지를 힘주어 표시한다. 첫째, **여기서 누적되는 loss는 inner associative loss $\ell$이 아니라 task loss $\mathcal{L}$(next-token prediction)이다** [NL §7.1] — CMS는 inference 중에도 자기 문맥에 대한 NTP objective의 truncated gradient step을 계속 밟는, 문자 그대로의 "느린 훈련"이다. 둘째, 표준 Transformer의 MLP block은 $k=1$, frequency 0의 퇴화 사례다 — CMS는 대체물이 아니라 일반화다.

level들의 배선은 §16.3.4의 transfer 메커니즘 선택에 따라 세 가지 변형이 있다 [NL §7.1]. **Nested CMS**: level $s$가 level $s{+}1$의 초기 상태를 meta-learn하고($\theta^{(s+1)}_0=\arg\min_\Phi\mathbb{E}_{T\sim\mathcal{C}^{(s)}}[\ell(\Theta,T;\Phi)]$ [NL Eq. 72]), 각 block은 자기 context가 끝나면 — $\lceil C^{(s)}/C^{(s+1)}\rceil$ 스텝마다 — 그 meta-learn된 초기값으로 **re-initialize**된다. higher-order ICL을 주는 배선이며, TNT의 periodic reset(→ 15장)이 "throughput 트릭"에서 "level 구조의 원리"로 일반화되는 자리다. **Sequential CMS**: block들을 직렬로 놓고 모든 초기 상태를 최저 주파수 level에서 backprop으로 일괄 meta-learn한다 [NL Eq. 73]. **Independent (head-wise) CMS**: 서로 다른 context 길이의 block들을 병렬로 두고 $y_t=\mathrm{Agg}\big(\mathrm{MLP}^{(f_k)}(x_t),\dots,\mathrm{MLP}^{(f_1)}(x_t)\big)$로 결합한다 [NL Eq. 74] — $\mathrm{Agg}$는 예컨대 학습된 가중합이다.

CMS가 continual learning을 돕는 논리는 명시적이다 [NL §7.1]. 빠른 block $\mathrm{MLP}^{(f_s)}$의 갱신이 옛 지식을 덮어써도, 그 지식은 보통 더 느린 block들($s'<s$)에 살아 있다. 그리고 초기 상태의 backprop transfer가 그 지식을 다시 빠른 block으로 **순환**시킬 수 있다 — 시간 차원의 loop가 생겨 중요한 지식의 완전한 소실이 어려워진다. 12장의 persistent memory, 15장의 "느린 곳에 주차된 지식"이 여기서 하나의 스펙트럼 원리로 합쳐진다.

효율 논거도 논문 자신이 준다 [NL §7.1]. 갱신은 예정 시각이 된 block에만 일어나므로, $L_{\mathrm{layer}}$개 layer, 4-level CMS, 최고 주파수 $\hat f$, 은닉 차원 $d$ 기준으로 스텝당 건드리는 parameter는 평균 $O\big(\tfrac{1}{\hat f}\cdot\tfrac{L_{\mathrm{layer}}}{5}\cdot d^2\big)$ — forward 전체에 비해 작은 조각이다. 그리고 (M5)의 구조상 $i\not\equiv 0 \pmod{C^{(\ell)}}$인 위치들에는 chunk 내부의 순차 의존이 없으므로 전부 병렬 처리된다 — 9장의 chunkwise 스킴(→ 9장)이 그대로 적용되는 이유다.

마지막 조각은 배포 경로다. **ad-hoc level stacking** [NL §7.3]: CMS block들의 초기값을 pre-trained Transformer의 MLP weights로 놓는다 — $\mathrm{MLP}^{(f_i)}_0 = \mathrm{MLP}_{\mathrm{pre\text{-}trained}_i}$. inner learning rate $\eta^{(\ell)}$이 보간 다이얼이 된다: $\eta^{(\ell)}\to 0$이면 동결된 pre-trained 동작과 동일하고, 키울수록 in-context 적응이 커진다. §16.6의 continual learning 실험에서 Llama-3 계열을 개조하는 데 쓰이는 레시피가 정확히 이것이다.

### 16.3.9 CMS를 optimizer에 적용하면: M3

CMS가 frequency 스펙트럼이라는 **원리**라면, 그 원리는 token이 아닌 context flow에도 적용되어야 한다. **M3 (Multi-scale Momentum Muon)** [NL §7.2, Alg. 1]는 gradient context flow에 CMS(independent 변형)를 적용한 proof of concept이다. §16.3.5의 진단 — momentum은 43스텝 너머를 기억하지 못한다 — 에 대한 처방으로 momentum을 두 주기로 나눈다:

$$
m^{(1)}_t = m^{(1)}_{t-1} + \beta_1\, g_t \quad(\text{매 스텝}),
\qquad
m^{(2)}_t = m^{(2)}_{t-1} + \beta_3 \sum_{i=t-\hat C+1}^{t} g_i \quad(\hat C\ \text{스텝마다}),
$$

여기에 Adam형 2차 moment $h_t=h_{t-1}+\beta_2\,g_t^2$를 유지하고, 두 momentum을 각각 $\mathrm{NS}_\kappa$로 직교화한 뒤 가중합으로 결합한다:

$$
\Theta_t \;=\; \Theta_{t-1} \;-\; \eta\;\frac{\mathrm{NS}_\kappa\big(m^{(1)}_t\big) + \mu\,\mathrm{NS}_\kappa\big(m^{(2)}_t\big)}{\sqrt{h_t}+\epsilon}
\qquad [\text{NL Alg. 1, Eq. 75}]
$$

($\mu$는 aggregation 계수 — 원문 기호 $\alpha$는 retention gate와 충돌하므로 개명, 표 16-2). 한 줄 요약은 논문 자신의 것이다: **M3 = Adam + Muon + CMS**, independent CMS의 $\mathrm{Agg}$가 가중합인 사례다. 설계의 요점은 느린 momentum이 decay를 조절하는 방식(AdEMAMix, Multiscale Muon 등 동시대 다중 momentum 접근)이 아니라 **갱신을 늦추는 방식**으로 장기 문맥을 얻는다는 것 — CMS의 chunked accumulation을 gradient에 그대로 옮긴 것이다 [NL §7.2]. 논문은 M3가 계산 오버헤드를 가지며 대규모 네트워크로의 스케일에 도전이 있음을 자인한다 [NL §7.2] — 수치는 §16.6.

### 16.3.10 self-modifying Titans

Hope의 sequence block은 [Titans]의 마지막 동결 지점을 푼다. Transformer — 그리고 12장의 Titans까지도 — 의 projection $W_K,W_V,W_Q$와 gate 산출 head들은 in-context에서 동결이다. 따라서 token을 contextualize하는 방식 자체가 pre-training 지식으로 상계된다: 1-layer Transformer의 projection은 token과 위치만의 함수라서 문맥에 따라 의미가 바뀌는 단어의 다양한 encoding을 원리적으로 놓치고, 깊이가 이를 뒤 layer에서 보상하더라도 앞 layer의 병목은 남는다 [NL §8]. 최근 모델들의 short convolution·canon layer는 국소 token mixing만 제공할 뿐 문맥 전역에의 적응이 아니다 [NL §8]. [NL]의 처방은 근본적이다: **모든 projection과 gate를 test-time에 갱신되는 memory로 승격시킨다.**

**1단계 — 완전 적응형 memory** [NL Eq. 79–82]. 다섯 산출 $k_t, v_t, q_t, \eta_t, \alpha_t$ 전부를 각자의 memory가 만든다:

$$
k_t = \mathcal{M}_k(x_t; W_{k,t-1}),\quad
v_t = \mathcal{M}_v(x_t; W_{v,t-1}),\quad
q_t = \mathcal{M}_q(x_t; W_{q,t-1}),\quad
\eta_t = \mathcal{M}_\eta(x_t; W_{\eta,t-1}),\quad
\alpha_t = \mathcal{M}_\alpha(x_t; W_{\alpha,t-1}),
$$

각 $\mathcal{M}_\square$와 본체 $\mathcal{M}_{\mathrm{mem}}$은 $\min\ \ell(\mathcal{M}_\square; k_t, v_t)$를 어떤 optimizer로 최적화하고, 출력은 $y_t=\mathcal{M}(q_t;W_{\mathrm{mem}})$이다. 강조할 지점: **inner learning rate $\eta_t$와 retention gate $\alpha_t$(→ 13장) 자체가 memory의 출력이다** — 12장에서 slow head가 산출하던 optimizer hyperparameter가 이제 in-context로 학습되는 객체가 된다. 그리고 모든 초기 상태 $\mathcal{M}_{\square,0}$은 sequence 전반에 걸쳐 meta-learn된다 — 논문은 이것이 fast adaptation·훈련 안정성·노이즈 강건성에 필수라고 명시한다 [NL §8.1].

**2단계 — self-modification** [NL Eq. 83–85]. 1단계는 두 가지가 아쉽다: 모든 memory가 같은 $v_t$를 target으로 공유하는 준최적 설계이고, 여전히 자기 학습 과정을 바꾸는 self-modification이 아니다 [NL §8.1]. 그래서 각 memory가 **자기 target을 스스로 생성한다**:

$$
\hat v_{\square,t} \;=\; \mathcal{M}_\square\big(v_t;\ W_{\square,t-1}\big)
\qquad [\text{NL Eq. 84}]
$$

— 모델이 자기 훈련 데이터를 만들며, 그럼으로써 자기 update 알고리즘을 학습한다. Schmidhuber의 self-referential weight matrix(1993)와 SRWM(Irie et al. 2022)의 계보를 chunk 병렬화가 가능한 형태로 계승한 것이다(**self-modifying Titans**라는 이 책의 공식 명칭; "self-referential"은 이 계보를 부를 때 쓴다). 이 단계에서 $q_t = W_Q x_t$만은 비적응 projection으로 남는다 [NL §8.1].

**3단계 — learning rule** [NL Eq. 86–88]. token은 고도로 상관되어 있으므로, §16.3.6의 논리에 따라 inner optimizer로 dot-product GD가 아니라 **DGD + weight decay**를 쓴다. inner objective는 표준 $\ell(W;k,v)=\|\mathcal{M}(k;W)-v\|_2^2$이고, update는

$$
W_{\square,t} \;=\; W_{\square,t-1}\big(\alpha_t I \;-\; \eta_t\, k_t k_t^\top\big)
\;-\; \eta_t\, \nabla_W\, \ell\big(W_{\square,t-1};\ k_t,\ \hat v_{\square,t}\big),
\qquad \square\in\{k,v,q,\eta,\alpha,\mathrm{mem}\}
\tag{16-4}
$$

이며 read는 $y_t=\mathcal{M}(q_t;W_{\mathrm{mem},t-1})$이다. 식 (16-4)를 (M2)·(16-3)과 겹쳐 읽으면 구조가 보인다: $\alpha_t I$는 retention gate(남기는 비율, (M2)의 $\alpha_tW_{t-1}$과 동일 방향), $-\eta_t k_tk_t^\top$는 DGD가 주는 data-dependent decay — 지금 key와 상관된 방향을 지우고 쓰는 항 — 이고, 마지막 항은 self-generated target에 대한 gradient다. memory 구조는 여섯 개 모두 표준 deep memory, 즉 2-layer residual MLP $\mathcal{M}_\square(z)=z+W_{\square,1}\,\sigma(W_{\square,2}\,z)$다 [NL Eq. 89] — 구조가 같아야 할 필요는 없다고 논문은 명시하지만 실험은 이 구성이다.

memory가 linear(행렬)인 특수 사례에서는 닫힌 recurrence가 나온다 [NL Eq. 92–93]. $\ell_2$ objective면 $W_{\square,t}=W_{\square,t-1}(\alpha_t I-\eta_t k_tk_t^\top)-\eta_t\big(W_{\square,\xi}\,k_t-\hat v_{\square,t}\big)k_t^\top$($\xi$는 chunk anchor, §16.4) — Gated DeltaNet(→ 6장)의 일반화된 delta rule 꼴이다. dot-product objective면 gradient 항이 $\hat v_{\square,t}k_t^\top$로 줄어든다 [NL Eq. 92] (원문 인쇄 부호는 (M1) 관행과 반대 — 부호 관행 차이로 기록, 표 16-2).

세 가지 정직한 각주를 남긴다. 첫째, 식 (16-4)의 $\square$ 집합에는 원문 그대로 $q$가 포함되어 있으나 [NL Eq. 88, 96], 같은 절의 본문은 $q_t=x_tW_q$가 유일한 비적응 projection이라고 명시하고 [NL §8.1], ablation에서 "w/o inner-projection $q$"가 perplexity를 오히려 **개선**한다(12.19 vs 12.24 [NL Table 6]). 최종 설계는 $q$ 정적으로 읽는 것이 정합적이며, 이 데이터는 "level 추가가 항상 이득은 아니다"라는, 논문 자신의 프레임에 대한 내부 반례이기도 하다(§16.8). 둘째, deep memory(2-layer MLP)에 대해 $(\alpha_t I-\eta_t k_tk_t^\top)$ 항이 정확히 어느 행렬에 어떻게 적용되는지 — layer별인지, 입력 측 행렬에만인지 — 는 원문이 명시하지 않는다.
<!-- TODO-VERIFY: deep memory에서 (α_t I − η_t k_t k_t^⊤) 우측 곱의 적용 대상(W_{□,1}? W_{□,2}? 층별 전부?)이 [NL Eq. 88]에 미명시. 확인 방법: papers/2512.24695.pdf의 §8.1–8.2와 공개 구현(있다면) 대조. 행렬 memory 사례(Eq. 92–93)만 타입이 명확함. -->
셋째, read 시점의 표기가 원문 안에서 흔들린다: [NL Eq. 82]는 $\mathcal{M}_{\mathrm{mem},t}$에서, [NL Eq. 83, 94]는 $\mathcal{M}_{\mathrm{mem},t-1}$에서 읽는다. 배포형(chunkwise, §16.4)에서는 둘 다 chunk 경계 snapshot으로 수렴하므로 이 구분은 실질적으로 소멸한다.

### 16.3.11 Hope block

**Hope** [NL §8.3]는 지금까지의 두 산물을 한 block에 직렬로 잇는다: **self-modifying Titans**(작은 state, 표현력 있는 DGD rule) 뒤에 **CMS**(큰 capacity, 단순한 rule)를 붙인다. 논문의 설계 논리가 명시적이다 — CMS는 지속 저장을 위해 capacity가 커야 하므로 단순한 learning rule을 쓰고, self-modifying Titans는 state가 작은 대신 표현력 있는 rule을 쓴다; 두 시스템은 상보적이다 [NL §8.3]. forward는 (normalization과 convolution 생략)

$$
o_t = \mathcal{M}\big(q_t;\ W_{\mathrm{mem},t-1}\big)
\quad\text{(식 (16-4)의 갱신과 함께)},
\qquad
y_t = \mathrm{MLP}^{(f_k)}\big(\cdots\, \mathrm{MLP}^{(f_1)}(o_t)\big)
\qquad [\text{NL Eq. 94–97}]
$$

이다. 실무 세부 두 가지 [NL §8.3]: $q,k$는 $\ell_2$ normalize하고(식 (16-3)의 DGD가 요구하는 $\|x\|=\lambda$ 전제와 정합), window 4의 local convolution을 쓴다. 변형 **Hope-Attention**은 self-modifying Titans 부분을 softmax global attention으로 치환한 것 — CMS의 기여를 분리 측정하기 위한 통제 변형이다 [NL §8.3].

12장의 어휘로 자리를 지정하면: Hope는 Titans-MAC/MAG처럼 attention과 memory를 병렬 합성한 것이 아니라, Titans block의 **모든 구성요소를 한 level씩 위로 올리고**(projection·gate의 memory화), 뒤따르는 MLP를 주파수 스펙트럼(CMS)으로 펼친 것이다. Titans의 persistent/long-term/short-term 삼분류(→ 12장)는 여기서 연속체의 세 점으로 흡수된다.

### 16.3.12 표기 대응표

**표 16-2 — [NL] 원 표기 ↔ 통일 표기 대응표** (STYLE-NOTATION §1.7.5의 표를 그대로 복사; "(장-국소 추가)"로 표시된 행은 이 장에서 더한 것)

| NL 원 표기 | 통일 표기 | 의미 / 주의 |
|---|---|---|
| Def. 1의 $\mathcal{M}^*$ | $\mathcal{M}^\star$ | |
| $W_t$ (inner), $\theta^{(f_\ell)}$ (level params) | $W_t$, $\theta^{(\ell)}$ | frequency 병기 필요시 $f_\ell$ 사용 |
| $u_{t}=\nabla_{y}L$ — Local Surprise Signal | $u_t$ | 동일 (공식 용어 LSS) |
| $\delta_\ell$ — backprop 오차 | $\delta_\ell$ | 동일 |
| $m_{\ell,t}$ — momentum (outer) | $m_t$ | |
| $\tilde M$, $H$ — Adam 1차/2차 moment | $m_t$, $h_t$ | |
| $f_A$, levels, $C^{(\ell)}$ | $f_\ell$ (또는 $f_A$), $C^{(\ell)}$ | 동일 |
| $\mathcal{M}_{\square,t}$, $\square\in\{k,v,q,\eta,\alpha,\mathrm{mem}\}$ — self-modifying memories | 동일 | |
| $\hat v_{\square,t}$ — self-generated target | 동일 | |
| DGD의 $\eta_t'=\eta_t/(1+\eta_t)$ | 장-국소 유지 | |
| M3의 $M^{(1)},M^{(2)},V_t$; $\beta_1,\beta_2,\beta_3$ | $m^{(1)},m^{(2)},h_t$; $\beta$들은 장-국소 상수 | ⚠ $V_t\to h_t$ (value와 충돌 방지) |
| $\mathrm{NewtonSchulz}_T$ | $\mathrm{NS}_\kappa$ | |
| $\otimes$ — outer product | $vk^\top$ 표기 | $\otimes$ 금지 |
| 고유명사: NSAM, Neural Learning Module, CMS, Hope, DGD/GGD/GM, Delta Momentum, DMGD, M3, AdaTransformer | 유지 | 첫 등장 시 정의 |
| **(장-국소 추가)** $\alpha_{\ell,t+1}$ — optimizer 문맥의 momentum decay ([NL Eq. 33, 47–52]) | $\beta$, $\beta_{i+1}$ | ⚠ 통일 $\alpha_t$(retention)와 무관 — NL은 optimizer 문맥에서 $\alpha$를 decay로 쓴다 |
| **(장-국소 추가)** $\alpha_t$ — self-modifying Titans의 retention gate ([NL Eq. 88]) | $\alpha_t$ | 동일 방향 (남기는 비율) |
| **(장-국소 추가)** M3의 aggregation 계수 $\alpha$ ([NL Alg. 1]) | $\mu$ | retention gate와 충돌 방지 |
| **(장-국소 추가)** $\zeta$ — NS 유도의 내부 step size ([NL Eq. 44]) | $\zeta$ | 장-국소 유지 |
| **(장-국소 추가)** $\mathcal{M}_{\square,C\lceil t/C\rceil}$ — chunk 경계 snapshot | $W_{\square,\xi(t,C_\square)}$ | 원문 인덱스는 문면상 현재 chunk 끝이나 문맥상 직전 chunk 끝 상태 (§16.4 각주) |
| **(장-국소 추가)** $\tilde L$ — 임의 level의 내부 objective | $\tilde\ell$ | 통일 체계에서 $\mathcal{L}$은 outer(task) 전용 |
| **(장-국소 추가)** $P_t$ — preconditioner / momentum의 value | $P_t$ | 장-국소 |
| **(장-국소 추가)** Eq. 92의 dot-product 사례 부호 | (M1) 관행으로 고정 | 부호 관행 차이(알고리즘 동일) |

## 16.4 Outer-loop training vs inner-loop test-time learning

이 논문에서는 이 절의 제목 자체가 심문 대상이다. [NL]의 핵심 주장이 바로 "outer/inner의 이분법은 $K$-level 스펙트럼의 $K=2$ 특수 사례"이고, 극단적으로는 "pre-training도 corpus를 context로 하는 ICL이며, neural learning module에는 training time과 test time의 경계가 없다"이기 때문이다 [NL §6]. 그러나 **실제로 실험된 Hope**는 정확히 이 책의 이분법대로 만들어지고 서빙된다: pre-training이라는 한 번의 outer 단계가 있고, 그 뒤 inference에서 정해진 rule로 움직이는 상태들이 있다. 이 절은 그 구분을 Hope에 대해 완전히 전개한 뒤, 어디까지가 이분법이고 어디부터가 스펙트럼인지 표시한다.

**표 16-3 — Hope의 level 지도: 무엇이 어디서 누구에 의해 갱신되는가**

| 구성요소 | 층위 | 갱신 주기 | 갱신 rule | 학습/산출 주체 |
|---|---|---|---|---|
| embedding, $W_Q$, convolution, normalization | slow $\Theta$ | pre-training에서만 | AdamW | outer loop |
| 여섯 memory의 초기 상태 $\mathcal{M}_{\square,0}$ (각각 2-layer residual MLP의 weights) | slow $\Theta$ (meta-learn) | pre-training에서만 | AdamW — gradient가 unrolled inner 갱신을 관통 | outer loop (transfer 기제 4: initialization) |
| CMS 각 level의 초기값 $\theta^{(\ell)}_0$ | slow $\Theta$ (meta-learn) | pre-training에서만 | Sequential이면 최저 주파수 level에서 일괄 backprop [NL Eq. 73] | outer loop (transfer 기제 3+4) |
| gate $\eta_t,\ \alpha_t$ | inner (파생값) | 매 token (chunk 경계 snapshot에서 생성) | $\mathcal{M}_\eta,\mathcal{M}_\alpha$의 forward | 값은 inner가, 산출 함수(그 memory의 초기 상태)는 outer가 |
| self-target $\hat v_{\square,t}$ | inner (파생값) | 매 token | $\mathcal{M}_\square(v_t;W_{\square,\xi})$ | inner (self-generated) |
| fast weights $W_{\square,t}$, $\square\in\{k,v,\eta,\alpha,\mathrm{mem}\}$ | fast $W$ | $C_\square$ token마다 일괄 적용 (훈련·서빙 동일 스케줄) | DGD + weight decay, 식 (16-4) (+ momentum — 아래 각주) | inner loop |
| CMS parameters $\theta^{(\ell)}_t$ | 중간 주파수 | $C^{(\ell)}$ token마다 | (M5): task-loss error 누적 [NL Eq. 71] | inner loop이되 objective는 outer의 $\mathcal{L}$ (NTP) |
| AdamW의 $m_t, h_t$ (훈련에만 존재) | optimizer 자신의 level | 훈련 스텝마다 | EMA (→ 2장) | "end of pre-training"에서 폐기 — [NL §4.5]가 문제 삼는 지점 |

### 16.4.1 outer loop: 무엇이 meta-learn되고, gradient는 어떻게 inner를 관통하는가

pre-training이 학습하는 것은 표의 위 세 행 — 정적 projection·embedding·conv·norm, 여섯 memory의 초기 상태, CMS의 per-level 초기값 — 전부다. outer objective는 context들에 대한 기대 NTP loss이고, 이것은 4장의 MAML형 bilevel 구조(→ 4장) 그대로다: **outer gradient가 inner-loop 갱신 경로를 관통해 흐른다.** 구체적으로 autograd graph에는 (i) 각 memory의 per-chunk DGD recurrence (16-4) 전체와 (ii) CMS의 chunked 갱신 (M5)이 포함되며, $\partial \mathcal{L}/\partial \mathcal{M}_{\square,0}$은 이 unrolled 경로를 거슬러 계산된다. training 무경험 독자를 위해 다시 말하면: 이것은 "optimizer를 통과하는 backprop"이며, 훈련을 비싸게 만드는 정체가 이것이고, 서빙에서는 완전히 보이지 않는다. [NL]은 이 pre-training 자체를 level 1의 ICL — context flow가 corpus 전체인 — 로 재분류하고, 그 최적화를 담당하는 AdamW조차 App B에 의해 그 자신이 2-level nested memory임을 상기시킨다 [NL §6, App B].

훈련을 성립시키는 것은 chunkwise 병렬화다 [NL §8.2]. 길이 $L$의 시퀀스를 $\lceil L/C\rceil$개 chunk로 자르고(각 memory의 update frequency는 $f_\square = L/C_\square$), 두 가지를 chunk 경계 snapshot에 고정한다. chunk anchor를 $\xi:=\xi(t,C_\square)=C_\square\lfloor (t-1)/C_\square\rfloor$로 쓰면:

$$
k_t=\mathcal{M}_k(x_t;W_{k,\xi}),\quad
v_t=\mathcal{M}_v(x_t;W_{v,\xi}),\quad
\eta_t=\mathcal{M}_\eta(x_t;W_{\eta,\xi}),\quad
\alpha_t=\mathcal{M}_\alpha(x_t;W_{\alpha,\xi}),\quad
\hat v_{\square,t}=\mathcal{M}_\square(v_t;W_{\square,\xi}),
$$

$$
W_{\square,t} \;=\; W_{\square,t-1}\big(\alpha_t I-\eta_t k_tk_t^\top\big)
\;-\;\eta_t\,\nabla_W\,\ell\big(W_{\square,\xi};\ k_t,\ \hat v_{\square,t}\big)
\qquad [\text{NL Eq. 90}]
\tag{16-5}
$$

— (i) 다음 chunk 전체의 key/value/self-target/gate를 직전 chunk 끝 상태에서 **한 번의 배치 pass**로 생성하고, (ii) inner gradient도 그 snapshot 기준으로 평가한다. 그러면 chunk 안의 모든 gradient가 chunk 처리 시작 전에 병렬 계산 가능해지고, TTT의 dual form(→ 8장)과 Titans의 병렬 훈련(→ 12장)이 그대로 적용된다 [NL §8.2]. chunk 크기는 실무상 두 개를 쓴다: $\mathcal{M}_{\mathrm{mem}}$용 하나, 나머지 다섯 memory 공용 하나 [NL §8.2]. CMS 쪽은 (M5)의 구조 자체가 병렬성을 준다: 경계가 아닌 위치들엔 순차 의존이 없다 [NL §7.1].

각주(표기 슬립): 원문 [NL Eq. 90]의 snapshot 인덱스는 $C\times\lceil t/C\rceil$로 인쇄되어 있어 문면상으로는 **현재** chunk의 끝 — 아직 존재하지 않는 상태 — 을 가리킨다. 같은 절의 산문("직전 chunk의 마지막 상태에 대해 gradient를 취한다" [NL §8.2])이 의도를 확정하므로, 이 책은 [TNT Eq. 5–6]의 인덱스 슬립(→ 15장)과 같은 유형으로 처리하고 $\xi(t,C)$로 통일해 쓴다.

이 지점에서 이 장의 가장 중요한 systems 문장이 나온다. §16.3.2에서 backprop이 병렬화되지 않는 이유는 value의 자기참조성이었다. self-modifying Titans는 정의상 그 자기참조성을 **아키텍처로** 가져온 모델이다 — 그렇다면 왜 훈련이 병렬화되는가? 답: **chunk 안에서 value 생성기를 stale snapshot으로 동결했기 때문이다.** 정확한 self-reference(현재 상태가 target을 생성)는 여전히 순차적이고 [NL §4.5], chunkwise stale-snapshot 근사(→ 9장의 (M4), "semantic hyperparameter" 명제)가 병렬성을 사는 공학적 타협이다. 배포·훈련된 Hope는 수학적 대상(정확한 self-modifying 모델)의 근사물이며, chunk 크기 $C_\square$는 그 근사의 충실도와 throughput을 맞바꾸는 다이얼이다. SRWM이 정확한 self-reference를 갖고도 병렬 훈련이 없어 가지 못한 스케일을 Hope가 가는 이유가 정확히 이 근사이고 [NL §9.5], 그 대가(충실도 손실의 정량)는 논문에 없다(§16.8).

### 16.4.2 inner loop: inference에서 무엇이 어떤 rule로 움직이는가

decode 중 token 하나가 들어오면 Hope block에서 일어나는 일은 세 겹이다. (1) **여섯 개의 fast memory** — $\mathcal{M}_k,\mathcal{M}_v,\mathcal{M}_q,\mathcal{M}_\eta,\mathcal{M}_\alpha,\mathcal{M}_{\mathrm{mem}}$, 각각 2-layer residual MLP — 가 식 (16-5)의 스케줄로 갱신된다. 서빙도 훈련과 같은 chunkwise 스케줄을 쓰므로, per-token 상환 비용은 "여섯 소형 MLP의 forward + read" 더하기 "$1/C_\square\times$(chunk당 병렬 inner-gradient 계산 = memory당 소수의 GEMM)"이다. gate $\eta_t,\alpha_t$는 별도 학습기가 아니라 이 memory들의 forward 산출이다 — "이 gate는 누가 학습하는가?"의 답은 두 겹이다: **gate의 값은 inner에서 fast memory가 만들고, 그 memory의 초기 상태(= gate를 만드는 법)는 outer에서 meta-learn되었다.** (2) **CMS의 각 level**이 $C^{(\ell)}$ token마다 (M5)로 갱신된다 — inference 중에도 모델은 자기 문맥에 대한 NTP objective의 truncated gradient step을 기하급수 간격의 주파수들로 계속 밟는다. (3) 나머지 — $W_Q$, embedding, conv, norm — 는 움직이지 않는다.

state의 정체가 이 라인의 어떤 전작과도 다른 규모다: KV cache처럼 $O(L)$로 자라는 것이 아니라 **시퀀스 길이에 상수**이되, 그 상수가 "여섯 개 2-layer MLP의 parameter 텐서 전부 + inference 중 표류하는 CMS weights"다. state가 곧 weights다. fast memory들에는 별도의 test-time optimizer가 필요 없다 — DGD rule과 gate가 곧 optimizer다. 단, ablation 표의 "w/o Momentum" 행(13.58 vs 12.24 [NL Table 6])은 배포된 inner rule이 식 (16-4)에 명시되지 않은 Titans식 momentum 항 $S_t$(→ 12장)도 지니고 있음을 시사한다 — 본문 수식과 구현 사이의 이 간극은 원문이 명시적으로 해소하지 않는다.
<!-- TODO-VERIFY: Hope의 inner rule에 momentum 항이 실제 포함되는지(식 96에는 없음, Table 6 ablation에는 'w/o Momentum' 행 존재). 확인 방법: papers/2512.24695.pdf §9.6 서술 및 공개 구현 대조. -->

마지막으로 이분법과 스펙트럼의 경계를 정확히 긋는다. Hope에서 outer/inner 이분법이 완전히 성립하는 것은 **한 번의 pre-training 뒤 동결 서빙**이라는 표준 배포를 가정할 때다. [NL]의 제안대로 CMS를 계속 살아 있게 두는 continual 배포에서는 중간 주파수 level들이 "훈련도 추론도 아닌" 제3의 지위 — 이 책 1장의 어휘로는 폐기된 불변식("추론은 weights를 바꾸지 않는다")의 영구적 위반 — 를 차지한다. 이것이 §16.7의 서빙 논의와 17장의 lifecycle 논의로 이어지는 문이다.

## 16.5 Concept ledger delta

**표 16-4 — [NL]이 개념 원장에 가한 변경** (기원 표기: 관련 장)

| 구분 | 개념 | 내용 / 전작과의 관계 |
|---|---|---|
| 신규 | **Nested Learning / Nested System / NSAM** [NL Def. 3, 4] | 모델+훈련 절차 = 중첩 최적화 문제 시스템. 이 책 §1.6 카탈로그 전체를 포섭하는 상위 프레임 |
| 신규 | **context flow** | 각 level이 압축하는 데이터 스트림 (token / gradient / 상위 신호) |
| 신규 | **update frequency와 level** [NL Def. 2] | attention $f=\infty$, frozen MLP $f=0$; 높은 level = 낮은 주파수. 17장 [Sleep]이 동일 정의를 재사용 |
| 신규 | **Neural Learning Module** | 아키텍처×optimizer의 결합이 설계 단위; Transformer+SGD ≠ Transformer+Adam |
| 신규 | **Local Surprise Signal (LSS)** $u_t$ | surprise(→ 12장)의 출력 공간 대응물; backprop 재해석의 축 |
| 신규 | backprop = **self-referential** associative memory | 훈련 병렬화 불가의 구조적 이유; Schmidhuber 1993 계보의 복권 |
| 신규 | **knowledge-transfer taxonomy** (5 기제) | meta-learned $W_{\mathrm{init}}$(→ 12·15장)이 기제 4(initialization)로 공식 분류됨 |
| 신규 | **DGD / GGD / GM / Delta Momentum / DMGD** | delta rule(→ 5장)·Miras의 objective 축(→ 13장)의 optimizer 층위 이식 |
| 신규 | **CMS** (+ Nested / Sequential / Independent 변형) | TNT의 global/local hierarchy(→ 15장)의 아키텍처화; long/short-term 이분법의 스펙트럼화 |
| 신규 | **self-modifying Titans** | Titans(→ 12장)의 projection·gate 전부의 memory화 + self-generated target |
| 신규 | **Hope / Hope-Attention** | 라인의 새 기준 모델; CMS 분리 측정용 통제 변형 |
| 신규 | **M3** | CMS를 gradient context flow에 적용; momentum-as-memory의 생성적 산물 |
| 신규 | **ad-hoc level stacking** | pre-trained MLP → CMS 초기값; retrofit 배포 경로 |
| 신규 | **parametric vs non-parametric ICL** | Transformer ICL = non-parametric 해, recurrent/TTT = parametric; ICL은 emergent가 아니라 구조적 |
| 신규 | **CTNL** benchmark | MTOB(Kalamang)+Manchu 순차 학습으로 in-context catastrophic forgetting 측정 |
| 확장 | momentum-as-memory | 12장의 관찰($S_t$=past surprise) → 정리: 모든 gradient 기반 optimizer가 gradient에 대한 memory; capacity 산수(6/43 스텝)까지 정량화 |
| 확장 | surprise | momentary/past(→ 12장) → LSS로 출력 공간 확장; backprop 전체가 surprise 기반 memory |
| 확장 | retention gate (→ 13장) | 개념 자체는 13장 소유 그대로; DGD가 같은 원리(data-dependent decay)를 **learning rule 층위**에서 재생산 |
| 확장 | meta-learned $W_{\mathrm{init}}$ (→ 4·15장) | TNT의 load-bearing 장치 → 5대 transfer 중 기제 4로 일반화 |
| 확장 | periodic reset (→ 15장) | TNT의 throughput 트릭 → Nested CMS의 context-end re-initialization |
| 확장 | persistent memory (→ 12장) | prefix token → 주파수 0의 극점; init이 meta-learn 안 되면 gating이 대역을 맡는다는 corollary |
| 확장 | Omega rule (→ 14장) | 그대로 수입; GGD의 $\mathrm{Ret}$ 자리에 위치 지정 |
| 폐기 | 아키텍처 vs optimizer의 구분 | 같은 객체(associative memory)의 다른 level — 논문 제목의 한 수 |
| 폐기 | long-term/short-term memory 이분법 | 주파수 연속체로 대체 |
| 폐기 | "test-time" learning/memorization이라는 명칭 | continual 설정에서 오도적 — parametric ICL로 대체 제안 [NL §6] (이 책은 라인 관례상 TTT 계열 명칭을 유지하되 이 이의를 기록한다) |
| 폐기 | ICL의 emergent 지위 | 2개 이상 level의 구조적 귀결 (단, 좋은 성능에는 잘 훈련된 저주파 level이 필요 [NL §6]) |

## 16.6 실험과 스케일

**공통 setup.** 언어 모델 실험은 FineWeb-Edu와 장문 문서의 혼합 corpus, 32K vocabulary, AdamW(모델별 learning rate 튜닝, 나머지는 [Titans]의 기본 설정)로 from scratch 훈련이다 [NL §9.2, §9.3]. 두 스케일: **760M parameters / 30B tokens**와 **1.3B / 100B tokens** [NL §9.3]. 장문 이해(RULER) 실험은 약 50B tokens 훈련 [NL §9.2]. baseline은 Transformer++, Samba(attention+linear RNN hybrid), RetNet, DeltaNet, RWKV-7, Comba, TTT, Miras(Memora), DLA, Titans이다.

**언어 모델링과 상식 추론** [NL Table 2]. 760M/30B에서 Hope는 Wikitext ppl 18.68, LAMBADA ppl 20.07, 8개 상식 추론 평균 acc 52.28로 전 baseline 최고다(Titans 51.68, RWKV-7 50.55, Samba 51.46, Transformer++ 50.11). 1.3B/100B에서 Wikitext ppl 14.39, LAMBADA ppl 10.08, 평균 58.04(Titans 56.82, Samba 54.46, Transformer++ 53.38). 논문은 attention-free 모델 대비 격차가 스케일과 함께 커진다고 보고한다 [NL §9.3].

**RULER needle-in-a-haystack** [NL Table 1]. Hope는 attention-free 모델 중 전 과제·전 난이도 최고다. 어려운 축에서의 대비가 요점이다: S-NIAH-3(UUID needle) 16K에서 Hope 24.8 vs Titans 21.2, RWKV-7 5.8; MK-NIAH 16K에서 Hope 14.8 vs Titans 8.2. linear memory(RWKV-7, Comba)가 길이에 따라 붕괴하는 곳에서 deep memory(Titans, Hope)의 capacity가 버티고, 그 위에서 self-modification+CMS의 증분이 Titans와의 차이로 나타난다는 것이 논문의 독해다 [NL §9.2]. 통제 변형도 유의미하다: Hope-Attention은 S-NIAH-1에서 4K/8K/16K 모두 100으로, 같은 attention을 쓰는 Transformer(88.6/76.4/79.8)를 상회한다 — attention이 동일하므로 이 증분은 CMS의 몫이다.

**BABILong** [NL Fig. 9]. 벤치마크 원 설정대로 fine-tune한 소형 모델 비교에서, Titans·ARMT·Hope가 1M token까지 경합하다가 그 뒤 Titans와 ARMT는 급락하고 **Hope는 10M token까지 성능을 유지한다** — 논문은 CMS 설계를 원인으로 지목한다 [NL §9.2]. GPT-4급 대형 모델은 128K–256K에서 무너지고, RAG-Llama-8B는 하락 후 낮은 수준에서 유지된다. 정직성 항목: 이 10M 결과는 **fine-tuning을 전제**한다 — zero-shot으로는 Hope를 포함한 모든 소형 모델이 크게 떨어지며, 논문은 저주파 level들이 과제에 맞게 조정되어야 고주파 memory 관리가 작동한다고 설명한다 [NL §9.2].

**continual learning** [NL Fig. 6, Fig. 8]. class-incremental 분류 세 벤치마크 — CLINC(150 intent), Banking(77 intent), DBpedia(70 클래스) — 에서, Llama3-8B/Llama-3B를 ad-hoc level stacking(§16.3.8)으로 개조하고 15B tokens continual pre-training한 Hope 변형이 ICL, EWC, 그리고 외부 학습기를 쓰는 SOTA인 InCA를 전부 상회한다 [NL §9.1]. 신설 benchmark **CTNL**(MTOB의 Kalamang + Manchu; ChRF 측정)에서는: 언어 하나씩 배울 때는 Hope 변형들이 ICL과 동급 이상이고, **두 언어를 순차로** 배우게 하면 ICL은 붕괴(사실상 pre-training 지식으로 회귀)하는 반면 Hope-1/2/3(추가 memory level 1/2/3개)은 level 수에 따라 단조 개선되어 Hope-3는 단일 언어 성능을 거의 회복한다 [NL §9.1]. level ablation [NL Fig. 7] (MK-NIAH, LongHealth, QASPER; ICL 및 DuoAttention baseline): level이 많을수록 좋고, 최저 주파수는 512가 최고이되 2K가 근접 성능에 훨씬 저렴해 논문이 효율 sweet spot으로 지목한다 [NL §9.1]. Cartridges와의 비교는 계산 프로파일이 달라(자가 학습 비용 등) 통제 실험이 필요하다며 명시적으로 유보한다 [NL §9.1].

**in-context recall과 MAD** [NL Table 3, 4]. 의무 서술 caveat이 여기 있다: 짧은 in-context recall에서는 **여전히 Transformer가 최고이고 격차가 크다** — FDA에서 Transformer 67.3 vs Hope 41.9, SWDE 71.4 vs 65.9. Hope는 attention-free 중 최고(Titans·RWKV-7·Comba 전부 상회)로 격차를 좁혔을 뿐이다 [NL §9.4]. [Atlas]의 53.6 vs 43.7(→ 14장)에서 확인된 in-context retrieval gap이 이 라인의 종합판에서도 **미해소**라는 뜻이다. 반면 합성 벤치마크 MAD에서는 Hope가 Transformer를 포함한 전부를 이긴다(compression 51.2, fuzzy ICR 52.1, memory 85.2) [NL Table 4].

**formal language 인식** [NL Table 5]. Irie et al. 2023의 구성을 따르는 parity, $(aa)^*$, $(abab)^*$, $a^nb^n$, $a^nb^nc^n$, Shuffle-2에서 Hope는 훈련 분포 밖 길이 구간(Bin1)을 포함해 **전부 100.0**이다. Transformer는 parity Bin0 46.4/Bin1 0.0으로 실패하고, linear attention과 DeltaNet도 Bin1 대부분이 0이다. LSTM과 SRWM도 전 항목 100이지만 병렬 훈련이 불가하다 — Hope의 차별점은 "비선형 recurrence급 state tracking + 병렬 훈련"의 양립이다 [NL §9.5]. 정직성 항목: 이것은 12장의 [Titans Thm 4.1]($TC^0$ 초과, 증명 없음)과 같은 계열의 주장에 대한 **실증이지 정리가 아니다** — [NL]에는 $K$-level NSAM의 표현력 정리가 없고, state-tracking 실험이 그 자리를 대신한다.

**component ablation** [NL Table 6]. 760M급 설정에서 구성요소를 하나씩 제거한 결과다.

**표 16-5 — Hope 구성요소 ablation ([NL Table 6] 재구성; 언어 모델링 평균 ppl ↓ / 상식 추론 평균 acc ↑)**

| 변형 | ppl | acc |
|---|---|---|
| Hope (전체) | 12.24 | 58.1 |
| DGD → 단순 GD | 13.41 | 56.5 |
| momentum 제거 | 13.58 | 56.9 |
| weight decay 제거 | 13.71 | 57.2 |
| CMS 제거 | 13.04 | 57.3 |
| inner projection $k$ 동결 | 13.77 | 56.9 |
| inner projection $v$ 동결 | 13.90 | 55.1 |
| inner projection $q$ 동결 | 12.19 | 57.4 |

읽는 법: DGD·momentum·weight decay·CMS·inner $k$·inner $v$는 전부 제거 시 악화 — 특히 inner $v$(self-target의 원료)의 동결이 acc를 가장 크게 깎는다(55.1). 단 하나의 예외가 inner $q$ 동결로, ppl이 오히려 개선된다(12.19) — §16.3.10의 각주와 §16.8의 "level 배치는 경험적"이라는 한계로 직결되는 행이다.

**M3** [NL §9.7]. ViT-24M/86M을 ImageNet-21K(이미지 11M장, 10,450 클래스, patch 16, MLP 차원 1536/3072)에서 optimizer만 바꿔 훈련하면 M3가 AdamW·Muon 대비 최저 train/test loss를 준다 [NL Fig. 11]. 효율은 별도 측정이다: 140M·1.3B Transformer LM 훈련에서 M3는 momentum이 여럿이고 NS pipeline이 추가라 **Muon보다 느리고 AdaMuon과 동급**이다 [NL Fig. 12] — 품질 이득이 비용과 함께 온다는 정직한 보고다.

**스케일의 정직한 상한.** from-scratch의 실증 상한은 이 라인 공통의 **1.3B / 100B tokens**이고(retrofit은 Llama3-8B까지), Hope 자체의 decode throughput·latency·메모리 사용량 수치는 논문에 없다 — wall-clock이 측정된 것은 M3(훈련 시간)뿐이다. 6편 전체에 decode wall-clock이 부재하다는 라인 공통 한계가 종합판에서도 이어진다. 7B+에서, 그리고 잘 튜닝된 production Transformer 대비 Hope의 우위가 유지되는가는 열린 문제다(§16.8).

## 16.7 Systems/serving 함의

**state의 산수: KV cache가 아니라 per-request weights.** Hope block 하나의 inference state는 여섯 개의 2-layer residual MLP memory — memory당 행렬 2개, 도합 **약 12개의 $d\times d_h$급 행렬** — 에 inference 중 표류하는 CMS level별 MLP weights를 더한 것이다. KV cache의 $O(L\cdot d)$ 성장과 달리 시퀀스 길이에 상수이지만, 그 상수가 weights 규모다. 비교 프레임은 통상의 linear-model 산수다: context가 길어질수록 Hope가 이기고 짧을수록 attention이 이긴다 — 단 crossover 지점이 단일 matrix memory(→ 6장)보다 훨씬 오른쪽에 있다. 그리고 질적 차이가 하나 있다: 이 state는 read-only cache가 아니라 **mutable weights**다. per-request로 fast weights가 다르므로 shared-weight batching이 깨지고(→ 1장 Rosetta: grouped-GEMM decode), 대화 checkpoint/restore는 이 텐서들의 영속화를 뜻한다. Titans/TTT/Atlas 서빙과 같은 문제 클래스이되, 단일 memory Titans block 대비 fast-weight 텐서가 약 6배라는 점이 다르고, chunk 주기 갱신이 이를 부분 상쇄한다.

**decode state의 RMW 트래픽.** memory 관점의 첫 논증 지점은 read-modify-write 트래픽이다. 각 fast memory는 $C_\square$ token마다 자기 parameter 텐서 전체를 읽고-갱신하고-쓴다. token당 상환 write 트래픽은 (state bytes)/$C_\square$ — chunk 크기가 대역폭 다이얼이다. CMS는 같은 구조를 level별로 반복한다: level $\ell$은 $C^{(\ell)}$ token마다 자기 MLP 전체를 RMW하고, 논문 자신의 상환 추정이 4-level CMS·최고 주파수 $\hat f$ 기준 스텝당 $O\big(\tfrac{1}{\hat f}\cdot\tfrac{L_{\mathrm{layer}}}{5}\cdot d^2\big)$ parameters다 [NL §7.1]. "최저 주파수 2K가 sweet spot"이라는 실험 결론 [NL §9.1]은 이 트래픽 상환이 품질을 크게 잃지 않고 가능하다는 보고로 읽힌다.

**update frequency ↔ memory 계층 배치.** 두 번째 논증 지점: [NL]의 주파수 스펙트럼은 그대로 배치 스펙트럼이다. 매 chunk 갱신되는 여섯 fast memory는 decode 중 상시 접근되는 hot set으로 HBM 상주가 강제되는 반면, $C^{(\ell)}=2\mathrm{K}$급 CMS level은 수천 token에 한 번만 RMW되므로 더 차가운 계층·더 느린 경로를 허용한다. update frequency가 곧 forward 비용이자 배치 제약이라는 논문의 효율 논거 [NL §7.1]는, 독자의 어휘로 "level 번호가 곧 cache tier 힌트"라는 뜻이다.

**병렬화와 kernel 형태.** 훈련과 prefill은 §16.4의 chunkwise dual 스킴이다: chunk의 모든 파생값 생성이 배치 pass 하나, snapshot 기준 inner gradient가 배치 GEMM 묶음, 경계에서 일괄 적용. memory가 MLP이므로 kernel은 단일 scan이 아니라 TTT-MLP형 **batched small-GEMM 시퀀스**이고(→ 8장, 9장), linear 특수 사례(식 (16-5)의 행렬 버전)는 일반화된 delta rule이므로 DeltaNet 계열의 WY/UT-transform chunk kernel(→ 9장)이 자연스러운 템플릿이다. $\mathcal{M}_{\mathrm{mem}}$용과 나머지용의 **두 chunk 크기**는 kernel 설계의 자유도다 — 품질에 민감한 본체와 덜 민감한 보조 memory들의 staleness를 따로 튜닝할 수 있다. decode는 token당 여섯 소형 MLP forward + 상환 update GEMM으로 arithmetic intensity가 GEMM 지배적이고, latency는 context 길이에 평탄하다 — 긴 문맥에서 attention 대비 통상의 linear-model 우위이며, 짧은 recall에서는 Transformer가 이긴다는 crossover가 [NL Table 3]의 수치 그대로다.

**retrofit이라는 배포 경로.** ad-hoc level stacking(§16.3.8)은 이 라인에서 처음 나온 현실적 이식 스토리다: 기존 pre-trained Transformer의 MLP들을 CMS level 초기값으로 삼고 15B tokens continual pre-training으로 개조한다 [NL §7.3, §9.1]. from-scratch pre-training 없이 continual learning 능력을 얹는 경로라는 점에서, 이 라인의 채택 장벽(스케일 재훈련 비용)을 우회하는 제안이다.

> **[평가]** 가장 파괴적인 함의는 수식이 아니라 배포 모델에 있다. [NL]이 의도하는 continual 배포에서 모델은 **서빙 중에 사용자 데이터로 자신의 NTP truncated gradient step을 계속 밟는다** — train/serve 경계의 소거는 곧 "추론은 weights를 바꾸지 않는다"는 서빙 불변식의 영구 폐기다. per-tenant weight 발산, 잘못 학습된 상태의 rollback, self-update된 weights의 provenance와 안전성, speculative decoding과의 상호작용(→ 1장 Rosetta의 snapshot/rollback 항목) — 어느 것도 논문이 다루지 않으며, Hope의 서빙 경제학(throughput/latency/메모리)은 정량 수치가 전무하다. 이 라인의 서빙 문제는 종합판에 와서 풀리기는커녕 6배로 늘었다.

## 16.8 한계와 bridge-out

논문이 스스로 그은 한계선부터 정리한다.

**catastrophic forgetting은 해결되지 않았다.** 결론부의 자문자답이 명시적이다: forgetting은 압축의 자연스러운 귀결 — 유한 capacity의 네트워크는 새 정보를 위해 잊도록 강제된다 — 이며, Hope와 CMS는 실험된 과제들에서 이를 **줄였을** 뿐 일반적으로 "풀지" 않았다 [NL §10]. CMS가 하는 일은 어디서, 얼마나 빨리 잊는지를 주파수 축 위에 재배치하는 것이다. level 간 capacity의 원리적 배분은 열려 있다.

**level 설계는 경험적이다.** 몇 개의 level, 어떤 주파수, 어떤 구성요소가 자기 level을 받을 자격이 있는지에 대한 이론이 없다. inner $q$ ablation(12.19 vs 12.24 [NL Table 6])은 잘못 놓인 level이 해가 될 수 있음을 보이는 내부 반례이고, chunk 크기·level 수·주파수는 전부 손으로 고른 hyperparameter다(2K sweet spot도 실험적 발견이다). 주파수 스케줄을 학습하는 문제는 미착수다.

**이론은 재해석까지만이다.** §16.3.5의 [평가]에서 본 대로 optimizer 역공학들은 존재 논증이며, $K$-level NSAM이 level 수에 따라 어떤 표현력 계층을 오르는지에 대한 정리 — 예컨대 computational depth가 level에 어떻게 의존하는지 — 는 없다. formal language 100.0은 실증이다(§16.6). [Titans Thm 4.1]의 증명 부재(→ 12장)라는 라인의 이론 공백이 종합판에서도 이어진다.

**self-reference의 충실도 분석이 없다.** chunkwise stale-snapshot 근사가 정확한 self-modifying 모델을 얼마나 훼손하는지 — chunk 크기에 대한 오차 한계 — 는 어디에도 없다(→ 9장의 라인 공통 caveat). 정확한 self-reference는 여전히 순차적이며, 더 나은 병렬화가 존재하는지는 열린 문제다.

**서빙 경제학과 스케일.** §16.6–16.7에서 본 대로: Hope의 wall-clock 부재, per-request mutable weights의 미해결 인프라 문제, from-scratch 1.3B/100B 상한, M3의 자인된 오버헤드와 대규모 미검증, 그리고 "architecture-specific optimizer" — 아키텍처가 만드는 gradient 분포에 맞춘 optimizer — 는 §16.3.4에서 근거까지 마련해 놓고 설계는 전부 미래 과제로 남겼다 [NL §6]. in-context retrieval gap(FDA 67.3 vs 41.9)도 미해소다. Cartridges류 self-study 접근과의 비교는 유보되었다 [NL §9.1].

> **[평가]** 이 장의 프레임으로 라인을 되돌아보면 [NL]의 실제 기여는 두 겹이다. 재해석의 층("모든 것이 associative memory다")은 검증 불가능한 관점 선언에 가깝지만, 그 관점이 **생성한 물건들** — DGD, CMS, self-modifying Titans, M3 — 은 ablation으로 각각의 기여가 측정된 구체물이다(표 16-5). 관점의 가치는 관점 자체의 참·거짓이 아니라 생성물의 성능으로 정산되었다고 읽는 것이 공정하며, 그 정산서에는 inner $q$ 행이라는 반례와 미측정된 서빙 비용이라는 미지급 항목이 함께 적혀 있다.

**Bridge-out: [Sleep]으로.** [NL]이 다음 논문에 넘기는 것은 스스로 범위 밖으로 선언한 반쪽이다. §16.2에서 신경생리학은 두 단계의 consolidation을 말했다: [NL]이 구현한 것은 **online consolidation** — 입력이 흐르는 동안, CMS의 주파수 스펙트럼으로 — 뿐이고, 수면 중 replay로 기억을 재조직하는 **offline consolidation**은 명시적으로 다루지 않았다 [NL §1.1]. 따라서 [NL]의 세계에서 모든 적응은 입력이 흐르는 동안 일어나고, capacity는 고정이며, 그래서 forgetting은 필연으로 남는다. 빠른 level의 지식이 예정된 갱신으로 덮어써지기 전에, 그것을 **어디로 옮길 것인가?**

[Sleep](→ 17장)의 답은 train/test 구분의 잔재를 마저 지우고 **wake/sleep lifecycle**을 세우는 것이다. wake는 [NL]의 CMS가 도는 online consolidation 그대로다 — update frequency의 정의(Def. 2)와 CMS 갱신식 (M5)를 [Sleep]은 문자 그대로 수입한다. sleep은 두 개의 offline 프로세스다: (1) Memory Consolidation — 빠른 block의 예정된 갱신이 지식을 덮어쓰기 직전, 그 지식을 다음 느린 block에 **새로 활성화한 low-rank expert로 위쪽 distillation**(Knowledge Seeding)하고, 빈 빠른 block은 reset한다(synaptic pruning) — forgetting을 regularization이 아니라 **capacity 성장**의 문제로 재정의하는 수다; (2) Dreaming — 자기 생성 데이터로 자기를 강화하는 RL loop다. sleep이 발화하는 시점은 정확히 CMS의 chunk 경계다 — TNT의 reset이 [NL]에서 Nested CMS의 re-initialization이 되었듯, [Sleep]에서는 "consolidation 후 reset"이라는 memory 위생 원리가 된다. [NL]이 "잊히기 전에 옮길 곳이 없다"는 문제를 만들었다면, 17장은 옮기는 절차 — 그리고 이 라인 전체의 완성형 — 를 다룬다.


