# ch16. Nested Learning: deep learning 아키텍처라는 착시, 그리고 Hope

## 16.1 Bridge-in: TNT가 남긴 문제

[TNT] (*TNT: Improving Chunkwise Training for Test-Time Memorization*, arXiv:2511.07343; → 15장)는 이 라인의 훈련 경제학 회차로, deep memory의 chunkwise training(→ 9장)이 안고 있던 세 문제(작은 chunk 훈련의 비효율, write/read 도메인 불일치, train/serve chunk-size mismatch)를 hierarchical memory와 two-stage training으로 풀며 하나의 구조를 남겼다: 큰 chunk로 천천히 갱신되는 global memory $W^{\mathrm{g}}$와, meta-learn된 초기 상태 $W_{\mathrm{init}}$로 주기적으로 reset되는 local memory들의 공존이다. 서로 다른 주기로 갱신되는 구성요소들, 그리고 느린 timescale의 지식이 빠른 timescale의 reset을 살아남는 구조 — 이것이 TNT가 심어 놓고도 정당화하지 않은 아이디어다. TNT에서 이 계층은 throughput 트릭이었을 뿐, 왜 그것이 모델 전체의 조직 원리여야 하는지, 그 위에서 아키텍처와 optimizer가 어떤 관계인지엔 대답이 없었다.

Part II의 사슬을 이 책의 기준 수식 (M2)로 요약하면 공백이 더 선명해진다. [Titans] (*Titans: Learning to Memorize at Test Time*, arXiv:2501.00663; → 12장)는 (M2) 자체 — GD + momentum + weight decay가 sequence layer라는 등식 — 를 세웠고, [Miras] (*It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization*, arXiv:2504.13173; → 13장)는 inner objective $\ell$과 retention을 설계 축으로 만들었으며, [Atlas] (*Atlas: Learning to Optimally Memorize the Context at Test Time*, arXiv:2505.23735; → 14장)는 $\ell$의 범위(window)와 inner optimizer(Muon)를 바꿨고, [TNT](→ 15장)는 훈련 경제학을 바꿨다. 다섯 논문 모두 "sequence layer 하나"의 성분을 바꿨을 뿐, 그 layer를 훈련시키는 바깥의 AdamW·momentum·backpropagation은 여전히 설계 공간 바깥의 고정된 배경이었다. 이 장의 논문은 질문의 단위를 바꾼다: layer가 아니라 **모델과 훈련 절차 전체**가 하나의 설계 대상이고, 그 전체는 서로 다른 주기로 자기 문맥을 압축하는 최적화 문제들의 중첩 시스템이라는 것이다.

이 장의 논문은 [NL] (*Nested Learning: The Illusion of Deep Learning Architecture*, arXiv:2512.24695; Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni, Google Research, v1 2025-12-31; NeurIPS 2025 발표본 존재)이다. [NL]은 이 라인의 이론 종합 회차다. TNT가 실용으로 발견한 "다중 timescale + reset 생존"을 존재론으로 선언하고, 그 위에서 세 가지를 생성한다: 더 표현력 있는 optimizer들(Delta Momentum, DMGD, DGD, M3), 주파수 스펙트럼 memory(**Continuum Memory System**), 그리고 자기 자신의 update 알고리즘을 학습하는 sequence block(**self-modifying Titans**). 세 산물의 결합이 **Hope**다.

## 16.2 문제의식과 논문의 핵심 주장

논문이 정의한 문제는 LLM의 정적임(static)이다 [NL §1]. 배포 후 적응 가능한 부분은 in-context learning뿐이라 지식은 (i) 지금 context window(attention의 KV cache), (ii) "end of pre-training"에 동결된 MLP weights 두 곳에만 존재한다. 논문은 이를 anterograde amnesia에 비유한다 [NL §1]: context 정보는 장기 저장소(feedforward layer)에 도달하지 못하고 window가 밀려나면 사라진다.

신경생리학이 두 처방을 시사한다 [NL §1.1]. 첫째, 장기 기억 형성은 최소 두 단계의 consolidation을 거친다: 각성 중 online (synaptic) consolidation과 수면 replay의 offline (systems) consolidation(→ 11장). [NL]은 명시적으로 전자만 다루고 후자를 범위 밖으로 선언하며 [NL §1.1] — 이 선언이 17장 [Sleep] (*Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories*, arXiv:2606.03979)으로 가는 인계선이다. 둘째, 뇌는 다중 timescale로 계산을 조직한다. 반면 현대 모델의 update frequency는 두 극단뿐이다: attention은 매 token 재계산이라 $\infty$, MLP는 test time 동결이라 $0$ [NL §1.1, §6].

세 번째 관찰은 구조의 균일성이다 [NL §1.1]. 뇌 memory는 분산·재사용 요소인 반면 현대 스택은 attention+SSM+convolution+MLP의 이질적 조합으로 보인다. 진단: 이 이질성은 착시다 — 최적화 문제의 **해**(예: attention 닫힌형)만 보고 **문제**(objective·문맥)를 안 봐서 달라 보일 뿐, 모든 구성요소는 GD로 최적화되는 linear 또는 MLP associative memory이고 차이는 level·objective·update frequency뿐이다 [NL §5.1]. 제목의 "illusion"이 이것이다.

핵심 주장은 넷이다. (1) 어떤 ML 모델이든 훈련 절차와 함께 하나의 **nested, multi-level, 병렬 최적화 문제 시스템**으로 표현되며, 각 문제는 자기 **context flow**(token, gradient, 상위 신호)를 자기 주기로 압축하는 associative memory다 [NL §3]. (2) 이 렌즈에서 backprop·momentum·Adam·AdaGrad·Muon은 전부 gradient를 압축하는 associative memory이고 특히 Adam은 특정 element-wise $\ell_2$ objective의 **최적** memory다 [NL §4, App B]. (3) 이 프레임은 기술용이 아니라 생성용이다: 새 optimizer·CMS·self-modifying Titans가 도출된다 [NL §4.4–§4.5, §7, §8]. (4) 더 많은 layer가 아니라 더 많은 **level**이 computational depth와 continual learning(→ 11장)을 사준다 — stacking의 새 차원 [NL §1.2, §3.2].

부산물 하나가 특히 중요하다: in-context learning은 emergent가 아니라 level이 2개 이상이면 반드시 생기는 구조적 성질이다 — Transformer는 non-parametric 해, recurrent 모델은 하위 level의 parametric 학습, pre-training조차 "corpus 전체를 context로 하는 ICL"로 재분류된다 [NL §6].

> **[해설]** 논문 제목은 Merrill et al. *The Illusion of State in State-Space Models*를 겨냥한다: 그쪽이 "SSM의 state는 얕다"면 [NL]은 "아키텍처 다양성이야말로 착시, 진짜 변수는 level"이라 되받는다(computational depth 한계는 [NL] §1의 layer-stacking-불가 목록 첫 항목).

## 16.3 Core mechanism (통일 표기)

이 절은 [NL]의 재해석 사슬을 원문 순서로 따라간다: 대수적 엔진 → 훈련 자체의 memory화 → level/nested system 형식화 → level 간 knowledge transfer → optimizer 재해석·새 optimizer → 아키텍처 재해석 → 생성물 셋(CMS·M3·self-modifying Titans) → Hope. 모든 수식은 통일 표기, 원 표기 대응은 §16.3.12 표 16-2에 있다.

### 16.3.1 대수적 엔진: associative memory 정의와 proximal/FTRL 등가

출발점은 [Miras]의 associative memory 정의를 그대로 수입한 [NL Def. 1]이다(정의 소유는 → 13장): key 집합 $\mathcal{K}\subseteq\mathbb{R}^{d_k}$와 value 집합 $\mathcal{V}\subseteq\mathbb{R}^{d_v}$가 주어질 때, associative memory는 사상 품질을 재는 objective $\tilde\ell$에 대해

$$
\mathcal{M}^\star \;=\; \arg\min_{\mathcal{M}}\ \tilde\ell\big(\mathcal{M}(K);\,V\big)
$$

로 얻어지는 연산자다. [NL]이 더하는 것은 둘이다. 첫째, 용어의 신경심리학적 고정: **memory는 입력이 일으킨 신경 갱신, learning은 유효한 memory를 획득하는 과정**이라 [NL §3.1] 어떤 level의 어떤 GD 갱신도 memory다. 둘째, key/value가 token일 필요가 없다 — gradient, sub-sequence, 상위 신호 모두 가능하다 [NL §3.1]. 한 level이 압축하는 데이터 스트림을 그 문제의 **context flow**라 부른다(sequence block=token, optimizer=gradient).

재해석의 대수적 엔진은 GD의 두 등가 형식이다. 한 스텝의 GD는 linearized objective + proximal 항의 argmin과 같다 [NL Eq. 2]:

$$
\theta_{t+1} \;=\; \arg\min_{\Phi}\ \Big\{\big\langle \nabla_\theta\,\tilde\ell(\theta_t; x_t),\ \Phi\big\rangle \;+\; \tfrac{1}{2\eta_t}\,\|\Phi-\theta_t\|_2^2\Big\},
$$

즉 GD 한 스텝은 "1차 Taylor 근사의 $\ell_2$-proximal 최소화"다. 누적하면(상수 $\eta$) FTRL 형식이 되고 [NL Eq. 3], 해는 $\theta_{t+1}=\theta_1-\eta\sum_{s\le t}\nabla\tilde\ell(\theta_s;x_s)$ — 3장의 FTRL(→ 3장) 그대로다. 이후 [NL]의 모든 재해석은 update 식을 "어떤 objective의 proximal argmin"으로 되읽고 그 objective와 context flow를 묻는 이 치환을 반복한다.

### 16.3.2 훈련은 surprise에 대한 memory다: LSS와 backprop의 자기참조성

먼저 부호 관행: 전역 기호(→ §1.2)를 따라 **Local Surprise Signal (LSS)** 을 $u_t=\nabla_{y_t}\mathcal{L}$로 고정한다. [NL]은 §3.1에서 같은 부호로 정의하되 GGD 문맥([NL Eq. 58–60])에서는 $-\nabla_{y_t}\mathcal{L}$를 $u_t$로도 쓰므로, 이 장은 self-generated value 자리에서 $-u_t$를 명시한다.

$y=\theta x$인 1-layer linear layer를 task loss $\mathcal{L}$(NTP)로 훈련하는 SGD 한 스텝은 다음처럼 인수분해된다 [NL Eq. 8]:

$$
\theta_{t+1} \;=\; \theta_t \;-\; \eta_{t+1}\,\nabla_\theta\,\mathcal{L}(\theta_t;x_{t+1})
\;=\; \theta_t \;-\; \eta_{t+1}\, u_{t+1}\, x_{t+1}^\top,
\qquad u_{t+1}=\nabla_{y_{t+1}}\mathcal{L}(\theta_t;x_{t+1}).
$$

여기서 $\nabla_\theta\mathcal{L}$은 12장의 surprise 그대로이고, $u_{t+1}$은 그 출력 공간 버전(국소 예측 오차)이다 [NL §3.1]. gradient가 rank-1 outer product $u\,x^\top$로 쪼개진다는 사실이 재해석의 문을 연다: 이 update는 proximal 형식으로

$$
\theta_{t+1} \;=\; \arg\min_{\Phi}\ \big\langle \Phi\,x_{t+1},\ u_{t+1}\big\rangle \;+\; \tfrac{1}{2\eta_{t+1}}\|\Phi-\theta_t\|_2^2
\tag{16-1}
$$

과 동치이며 [NL Eq. 9], Def. 1의 associative memory — $x_{t+1}$을 key, 그 LSS $u_{t+1}$을 value로 하는 dot-product objective의 memory — 다. 요약: **backprop으로 훈련되는 layer는 "각 입력 → 그 예측 오차"의 사상을 weights에 압축하는 memory다** [NL §3.1, §4.1].

깊은 모델로 가도 형태는 같다. $L_{\mathrm{layer}}$층 MLP $\{W_\ell\cdot+b_\ell\}_{\ell=1}^{L_{\mathrm{layer}}}$에서 backprop은 layer별 gradient를

$$
\frac{\partial \mathcal{L}}{\partial W_\ell} \;=\; \delta_\ell\, \hat x_{\ell-1}^\top,
\qquad
\delta_\ell \;=\; J_{\phi_\ell}(z_\ell)^\top\, W_{\ell+1}^\top\, \delta_{\ell+1}
$$

로 계산한다 [NL Eq. 29]($z_\ell$=pre-activation, $\hat x_\ell=\phi_\ell(z_\ell)$=layer 출력, $J$=activation Jacobian, $\delta_\ell$=layer $\ell$의 backprop 오차; → 2장). 따라서 layer $\ell$의 GD update도 식 (16-1) 꼴의 proximal argmin — "입력 $\hat x_{\ell-1}$을 국소 오차 $\delta_\ell$에 사상하는 memory" — 이다 [NL Eq. 31].

가장 미묘한 지점: 식 (16-1)을 보고 "backprop은 gradient에 대한 linear attention"이라 결론지으면 틀린다 [NL §4.1] — linear attention(→ 6장)은 key/value가 memory 상태와 무관해 병렬화되지만, backprop의 value $\delta_\ell$은 memory 자신의 현재 상태가 만든다:

$$
v_t \;=\; f_{W_t}(x_t) \;=\; -\,u_t \;=\; -\nabla_{y_t}\mathcal{L}(W_t;x_t)
\qquad [\text{NL Eq. 58}]
$$

— 즉 backprop은 자기 훈련 target을 스스로 생성하는 **self-referential** memory(Schmidhuber 1993)이며, 그 성질 때문에 scan으로 병렬화되지 않는다 [NL §4.5] — Hope의 chunkwise 훈련(§16.4)은 정확히 이 자기참조성을 chunk 경계 snapshot으로 절단하는 타협이다. inference 어휘로 각 layer는 KV cache 없이 (입력 → 오차)를 weights에 압축하는 memory이고(→ 1장 Rosetta), [NL]은 2장이 만든 optimizer-as-object 관점의 나머지 절반 — 그 state가 곧 한 level 위 memory — 을 더한다.

### 16.3.3 update frequency, level, 그리고 NSAM

여러 최적화 문제로 분해된 모델에 질서를 주는 것이 **update frequency**다. [NL Def. 2]: 구성요소 $A$(parametric이든 attention 같은 non-parametric이든)의 frequency $f_A$는 단위 시간(데이터 포인트 하나당 갱신 한 번)당 갱신 횟수다. attention은 $f=\infty$, 동결 MLP는 $f=0$ — §16.2 "두 극단"의 형식화다. 순서는 $A\succ B$: $f_A>f_B$이거나, $f_A=f_B$이되 $B$의 시점 $t$ 계산이 $A$의 시점 $t$ 상태를 요구하면 $A$가 빠르다. 어느 쪽도 아니면 같은 **level**(같은 주기, 상호 독립)이다 — Adam의 1차/2차 moment가 대표적 동률 사례로 한 level에 병렬로 놓인다 [NL Def. 2, App B]. 정렬에서 **높은 level일수록 낮은 frequency**다.

이제 시스템 전체가 정의된다. **Nested System** [NL Def. 3]: $K$개의 정렬된 level, level $k$는 (objective, context flow, feasible parameter)의 삼중항 집합 $\{(L^{(k)}_i,\ \mathcal{C}^{(k)}_i,\ \Theta^{(k)}_i)\}_{i=1}^{N_k}$이고 각 문제는 식 (16-1) 꼴 proximal GD로 최적화되되 일부 "box"는 non-parametric 해(예: attention)가 허용된다 [NL Eq. 19, §3.2]. **NSAM (Nested System of Associative Memories)** [NL Def. 4]는 모든 context가 key-value 집합인 특수형이고 [NL Eq. 20], Appendix A 일반형 [NL Def. 6, 7]은 선형화를 full loss로 되돌린다. **Neural Learning Module**은 아키텍처와 그 훈련 과정을 하나의 NSAM으로 함께 표현한 설계 단위이며, 같은 Transformer라도 SGD판과 Adam판은 **다른 module**이다 [NL §3.2].

정의가 실제로 어떻게 작동하는지, 논문의 계단식 예제 네 개가 보여 준다 [NL §3.1–§3.2].

- **(a) 1-layer MLP + GD** = 1-level: 데이터를 LSS에 사상하는 memory 하나 (§16.3.2).
- **(b) 1-layer MLP + momentum GD** = 2-level [NL Eq. 10–13]: momentum buffer가 gradient를 압축하는 inner memory이고(전개는 §16.3.5) outer가 그 상태를 weights에 적용한다 — 주기는 같지만 weights 계산이 momentum 상태를 요구하므로 momentum이 더 빠른 level이다.
- **(c) linear attention + GD** = 2-level [NL Eq. 14–18]: outer가 projection $W_K,W_V,W_Q$를 GD로, inner가 recurrence $W_t=W_{t-1}+v_tk_t^\top$($\tilde\ell(W;k_t,v_t)=-\langle Wk_t,v_t\rangle$의 GD 1스텝)를 돈다(→ 6장). inner는 projection을 동결 hyperparameter로 두지만, outer gradient $\partial\mathcal{L}/\partial W_K$는 stop-gradient가 아닌 한 recurrence를 관통한다(NL 산문의 "관통하지 않는다"는 inner 갱신이 projection을 안 바꾼다는 뜻일 뿐). FWP(→ 6장)의 slow/fast 구분과 겹친다 [NL §3.1].
- **(d) linear attention + momentum GD** = 3-level: (b)와 (c)의 합성 [NL §3.1].

정점은 **AdaTransformer**(별칭 Linear Attention++) [NL §3.2, Fig. 3, Eq. 69]: Transformer block과, MLP block을 linear attention으로 치환한(key/value는 앞단 attention 공유, 초기 상태 meta-learn) block의 출력이 대수적으로 동일하다:

$$
y_{\mathrm{block}} = y_{\mathrm{attn}}\,W_{\mathrm{MLP}}
\qquad\text{vs}\qquad
y_{\mathrm{block}} = y_{\mathrm{attn}}\,W_{\mathrm{LinAttn}}.
$$

유일한 차이는 **두 weight가 사는 level**이다: $W_{\mathrm{MLP}}$은 level 1(f=0)이라 context 불변, $W_{\mathrm{LinAttn}}$은 level 2에서 $W_t=W_{t-1}+v_tk_t^\top$로 in-context 갱신(초기값 $W_{\mathrm{init}}$은 level 1 NTP로 meta-learn)된다. 즉 **recurrent memory module=level 하나 추가된 MLP block, hybrid 아키텍처=일부 MLP block에 in-context 적응이 생긴 Transformer** [NL §3.2, §6] — "attention + SSM 이종 결합" 통념이 해소된다.

따름정리(12장 개념 회수): 초기 상태가 meta-learn되지 **않으면**($W_0=0$ 초기 linear Transformer) 그 block엔 persistent memory(→ 12장)가 없고, 출력 gating의 linear layer가 이를 대신한다는 것이 [NL §5]의 gating 해석 — "gating이 왜 도움이 되는가"의 구조적 답이다.

### 16.3.4 level 간 knowledge transfer: 다섯 가지 메커니즘

남는 설계 축은 level들이 서로 어떻게 지식을 주고받는가다. [NL §3.3]은 두 block $B^{(0)}$(저주파·상위)·$B^{(1)}$(고주파·하위) 사이의 **knowledge-transfer taxonomy** 다섯 가지를 제시한다.

표 16-1 — 다섯 가지 knowledge transfer 메커니즘 [NL §3.3]

| # | 메커니즘 | 형식 | 대표 사례 |
|---|---|---|---|
| 1 | direct parametric conditioning | $\mathcal{M}^{(0)}(\cdot):=\mathcal{M}^{(0)}(\cdot;\Theta^{(1)})$ 또는 $\mathcal{M}^{(0)}(\cdot;\mathcal{M}^{(1)}(\cdot))$ [NL Eq. 24–25] | FWP: fast weight의 출력이 slow net의 출력을 조건화(→ 6장) |
| 2 | direct non-parametric conditioning | $\mathcal{M}^{(0)}(\cdot;\mathcal{C}^{(1)})$ [NL Eq. 27] | softmax attention이 raw context에 조건화 |
| 3 | backpropagation across levels | 같은 gradient flow, 다른 갱신 주기 | CMS의 level 연결(§16.3.8) |
| 4 | initialization (MAML형) | $\Theta^{(1)}_0=\arg\min_\Phi \mathbb{E}_{\mathcal{C}\sim\mathcal{C}^{(0)}}\big[\ell(\mathcal{M}^{(1)}(\cdot;\Phi),\mathcal{C})\big]$ [NL Eq. 28] | meta-learn된 $W_{\mathrm{init}}$ (Titans/TTT/Atlas) |
| 5 | generation | weight 생성(hypernetwork) 또는 context 생성 | **아키텍처가 optimizer의 context(gradient)를 생성** |

1·2번의 공통 특징은 **backprop이 level을 건너지 않는다**는 것 — 각 block은 상대 상태를 자기 hyperparameter로 취급한다 [NL §3.3]. 3번은 두 상태가 같은 gradient flow에 있되 주기만 다르다(§16.3.8 CMS가 이 배선). 4번은 4장 MAML(→ 4장)이 분류되는 자리이자, 라인이 [Titans]부터 지고 온 meta-learned initial state $W_{\mathrm{init}}$(→ 12·15장)의 공식 좌석이다 — deep memory 모델(Titans·Atlas·Miras·TTT)은 4번을 갖지만 대부분의 linear RNN엔 level 간 transfer가 없다 [NL §6].

5번이 이 논문 고유의 한 수다: **아키텍처는 optimizer의 context(gradient)를 생성한다** [NL §3.3, §6] — vanilla GD/Adam에도 성립한다. 아키텍처가 다르면 gradient 분포가 달라 그것을 압축하는 memory(= optimizer)의 최적 설계도 달라져야 한다 — 논문이 명명만 하고 납품하지 않는 **architecture-specific optimizer**의 근거다.

정리: module 설계 = (1) 최적화 문제와 주기(NSAM 구성) + (2) level 간 knowledge transfer, 두 결정이다 [NL §3.3]. meta-learning·MAML·hypernetwork·learned optimizer는 전부 이 두 축의 특수 선택지다.

### 16.3.5 optimizer 재해석: momentum, Adam, AdaGrad, Muon

**momentum은 gradient들에 대한 Hebbian memory다.** layer $\ell$의 momentum GD를 통일 표기로 쓰면 [NL Eq. 33]

$$
W_{\ell,t+1} \;=\; W_{\ell,t} \;+\; m_{t+1},
\qquad
m_{t+1} \;=\; \beta\, m_t \;-\; \eta_{t+1}\, \delta_\ell\, \hat x_{\ell-1}^\top
\tag{16-2}
$$

이다(원문 decay $\alpha$는 통일 retention $\alpha_t$와 무관해 $\beta$로 개명, 표 16-2; 부호는 (M2) 관행). $\beta=1$이면 식 (16-2)의 $m$ 갱신은 $\min_m\ \langle m\,\hat x_{\ell-1},\ \delta_\ell\rangle$의 GD 1스텝이고, $\beta\neq 1$은 여기에 $m$의 $\ell_2$ regularization을 더한 것이다 [NL §4.2, Eq. 34]. 즉 momentum은 과거 gradient를 자기 parameter에 압축하는 **value-less(Hebbian) associative memory**이며, 2장의 (state, update, cost) 객체 $m_t$가 "weights보다 한 level 아래의 memory"로 재명명된다 — 12장의 **momentum-as-memory**가 정리적 지위로 승격된다(구조는 §16.3.3 (b)의 2-level 그대로).

memory라면 capacity를 물을 수 있다. decay $\beta=0.9$에서 최근 $n$개의 정상상태 기여는 $1-\beta^n$이므로 **50%를 넘으려면 7개·99%를 넘으려면 44개**가 필요하다(원문은 6/43으로 적었으나 $1-0.9^6=46.9\%$, $1-0.9^{43}=98.9\%$로 임계값에 못 미친다) [NL §4.3] — momentum은 장기가 아니라 최근 기억이다. 이것이 continual learning 실패로 나타난다(직교 task gradient 설정 [NL Eq. 45]에서 momentum이 새 task로 이동해 옛 subspace를 잃음). 진단은 **모델 capacity가 아니라 optimizer의 memory management 실패다** [NL §4.3]. 따름정리 [NL §4.5]: "end of pre-training"에서 optimizer state를 버리면 momentum이 저장한 landscape 지식이 통째로 사라진다 — continual learning module이라면 이 저주파 level도 보존 대상이다.

**Adam은 특정 objective의 최적 memory다** [NL App B]. element-wise memory $m$에 대해 objective

$$
\tilde\ell_t \;=\; \sum_{i=1}^{t}\ \big\|\, m \odot g_{i+1} \;-\; P_t \,\big\|_2^2 \;+\; \lambda\,\|m\|_F^2,
\qquad g_{t+1} = -\nabla_{W}\mathcal{L}(W_t;x_{t+1})
\qquad [\text{NL Eq. 101}]
$$

을 상정하면 — gradient를 전역 통계 $P_t$에 사상하는 $\ell_2$ regression — 닫힌형 최적해의 recurrence는 $\tilde m_{i+1}=\tilde m_i+\beta_1 g_{i+1}$(1차), $h_{i+1}=h_i+\beta_2\, g_{i+1}^2$(2차)이고 [NL Eq. 102], $P_t=\sqrt{\sum_i g_i^2}$로 고르면 **Adam형 정규화 update를 근사적으로 회수한다** [NL Eq. 105]. 단 위 두 recurrence는 EMA가 아니라 누적 합(AdaGrad 계열)이고 bias correction도 없으므로 표준 Adam의 EMA와 정확히 같지는 않다. outer-product 버전은 AdaGrad-with-momentum을 [NL Eq. 106–111], 나머지(RMSProp·SignSGD·NAdam·AMSGrad·RAdam·Lion; Shampoo·SOAP)는 Adam·AdaGrad와의 알려진 관계로 따라온다 [NL §4.2].

> **[평가]** 이 역공학들은 존재 논증이지 유일성 결과가 아니다: Eq. 101의 objective·$P_t$는 Adam이 나오도록 상정된 것이라 "Adam은 최적 memory"는 항상 "그 특정 element-wise $\ell_2$ objective에 대해"로 읽어야 한다. 프레임의 가치는 새 도출이 아니라 **변형을 체계적으로 생성하는 문법**에 있다.

**preconditioning과 Muon.** preconditioned GD $W_{t+1}=W_t-\eta_{t+1}P_{t+1}^{-1}g_{t+1}$ [NL Eq. 38]의 $P$는 "gradient를 선택된 좌표계로 사상하는 법"을 내부 objective $\min_P \tilde\ell(P(\hat g);g)$로 배우는 nested memory다 [NL Eq. 39–41]. Muon(→ 2·14장) $W_{t+1}=W_t+\mathrm{NS}_\kappa(m_{t+1})$ [NL Eq. 42]에서 좌표계는 직교 공간이다: orthogonalization objective

$$
\tilde\ell\big(P(g);g\big) \;=\; \tfrac12\big\|P(g)-g\big\|_F^2 \;+\; \tfrac12\big\|P(g)^\top P(g)-I\big\|_F^2
\qquad [\text{cf. NL Eq. 43}]
$$

를 $O_0=g$에서 GD 1스텝(내부 step size $\zeta$)으로 풀면 Newton–Schulz의 3차 다항 반복 [NL Eq. 44]이 그대로 나온다. (주의: [NL Eq. 43]은 직교화 항만 인쇄하나 그 gradient엔 $O-g$ 항이 없어, Eq. 44의 $O_i-g$가 나오려면 proximity 항 $\tfrac12\|P(g)-g\|^2$이 함께 있어야 한다 — 위 objective에 포함했다.) 결론: **Muon의 NS 반복은 momentum 갱신 한 번당 $\kappa$스텝을 도는 내부 최적화 level이다.** [NL §6]은 이를 "neuron당 더 많은 계산"이라 부른다 — level 추가가 memory 계층만이 아니라 계산 심도의 증폭기이기도 하다.

### 16.3.6 프레임의 생성적 사용: 새 learning rule들 — Delta Momentum, DMGD, DGD, GGD

재해석이 옳다면 optimizer 설계는 memory 설계와 같은 문법을 따른다: 6장·13장에서 sequence layer에 했던 조작(association·objective·memory 구조·feature map·출력 비선형성·learning rule)이 optimizer에 이식된다 [NL §4.4–§4.5].

- **preconditioned momentum** [NL Eq. 47]: value를 preconditioner $P_i$로 두면 momentum이 (preconditioner ↔ gradient) 사상을 배운다.
- **Delta Momentum** [NL Eq. 48–49]: 내부 objective를 dot-product에서 $\ell_2$ regression $\|m\,g_i^\top - P_i\|_2^2$로 바꾸면($g_i=\nabla_W\mathcal{L}(W_i;x_i)$) update가 delta rule(→ 5장) 꼴이 된다:
  $$
  m_{i+1} \;=\; \beta_{i+1}\, m_i \;-\; \eta_i\,\big(m_i\, g_i - P_i\big)\, g_i^\top .
  $$
  update가 자기 상태에 의존해 **gradient-dependent decay**가 생긴다(제한된 capacity $O(N)$ 관리 개선) — linear attention→DeltaNet 전이(→ 6장)의 optimizer 버전으로, 진동 곡률 toy에서 표준 momentum보다 빠르게 수렴한다 [NL Eq. 53, Fig. 4].
- **Deep Momentum GD (DMGD)** [NL Eq. 50]: momentum을 행렬에서 MLP로 승격한다 — $W_{i+1}=W_i+m_{i+1}(u_i)$, $m_{i+1}=\beta_{i+1}m_i-\eta_i\,\nabla\tilde\ell^{(2)}(m_i;u_i,1)$($u_i=\nabla_W\mathcal{L}$). deep memory(→ 12장)가 token에 했던 것을 gradient에 하는 것 — 과거 gradient의 비선형 사상까지 저장.
- **higher-order feature map** [NL Eq. 51]·**비선형 출력** [NL Eq. 52]: $\phi(\nabla\mathcal{L})$로 key 들어올리기(→ 14장 $\phi_p$ 문법)와 $W_{i+1}=W_i+\sigma(m_{i+1}(u_i))$. $\sigma=\mathrm{NS}$·$m$이 linear면 **Muon이 특수 사례로 회수된다** — Muon은 "비선형 출력을 단 Hebbian momentum"이었다.

정점은 learning rule 자체의 교체다. **Delta Gradient Descent (DGD)** [NL §4.5, Eq. 56–57]: 식 (16-1)의 dot-product objective는 각 gradient를 상태와 무관하게 취급한다 — i.i.d.면 합리적이나 상관된 token 공간에서는 낭비다. objective를 $\ell_2$ regression으로 바꾸면 ($u_t=-\nabla_{y_t}\mathcal{L}(W_t;x_t)$)

$$
W_{t+1} \;=\; \arg\min_{\Phi}\ \tfrac{1}{2}\big\|\Phi\, x_t - u_t\big\|_2^2 \;+\; \tfrac{1}{2\eta_t}\|\Phi-W_t\|_2^2
\qquad [\text{NL Eq. 56}]
$$

이고, 입력이 정규화되어 있으면($\|x_t\|_2=\lambda$ — normalization layer가 있으면 성립) Sherman–Morrison lemma [NL Eq. 117, App C]로 닫힌형이 나온다:

$$
W_{t+1} \;=\; W_t\big(I \;-\; \eta_t'\, x_t x_t^\top\big) \;-\; \eta_t'\,\nabla_{y_t}\mathcal{L}(W_t;x_t)\, x_t^\top,
\qquad \eta_t'=\frac{\eta_t}{1+\eta_t\lambda^2}\;(\lambda{=}\|x_t\|;\ \lambda{=}1\text{이면 }\tfrac{\eta_t}{1+\eta_t}).
\tag{16-3}
$$

GD에 **data-dependent decay** $(I-\eta_t'x_tx_t^\top)$가 자동으로 붙는다 — 지금 입력과 상관된 방향을 지우고 쓰는 rule이다. 13장 어휘로는 attentional bias를 dot-product→$\ell_2$로 바꾼 것, 5장 어휘로는 delta rule의 재발명이되 **learning rule 층위**에서다. 일반화: **Generalized Gradient Descent (GGD)** [NL Def. 5, Eq. 59–60]는 self-generated value $u_t=f_{W_t}(x_t)$를 갖는 임의의 self-referential memory

$$
W_{t+1} \;=\; \arg\min_W\ \tilde\ell\big(\mathcal{M}(x_t;W),\ u_t\big) \;+\; \mathrm{Ret}\big(W,\ \{W_i\}_{i=t-c+1}^{t}\big)
$$

의 family다($\mathrm{Ret}$은 최근 상태 근방에 해를 묶는 retention 항 — → 13장). [NL Eq. 59]는 $\tilde\ell(x_t,u_t)$로 인쇄하지만 argmin이 의미를 가지려면 $\tilde\ell$이 mapping $\mathcal{M}(x_t;W)$를 통해 $W$에 의존해야 한다. backprop(GD·DGD 포함)이 이 family의 원소이고, 같은 형식을 momentum에 적용한 것이 **Generalized Momentum (GM)**이다 — 단 momentum은 key/value가 하위 level에서 주어져 self-referential이 아닌 conventional memory다 [NL §4.5].

### 16.3.7 아키텍처 재해석: 카탈로그의 재확인

[NL §5]는 [Miras]의 결과(→ 13장)를 NSAM 어휘로 재수록하므로 자리만 확인한다. softmax attention은 $\sum_i s(k_i,q)\|v_i-\mathcal{M}\|_2^2$의 **non-parametric** 해(Nadaraya–Watson, → 8·13장) [NL Eq. 62·63], Hebbian RNN(linear attention·RetNet·RWKV) [NL Eq. 64], delta rule RNN(DeltaNet·Longhorn·RWKV-7) [NL Eq. 65], OjaNet [NL Eq. 66–67], Omega rule(→ 14장) [NL Eq. 68], $L_p$ bias(→ 13장) — 전부 dot-product·$\ell_2$·Oja·window objective의 GD 변형으로 §1.6 카탈로그와 1:1이다. [NL]의 추가 기여는 목록이 아니라 **자리 지정**이다: 이들 전부가 pre-training보다 한 level 위이고 projection들은 그 아래 level에서 최적화된다 [NL §5]. 여기서 §16.2 균일성 명제가 완결된다(차이는 level·objective·learning rule뿐) [NL §5.1].

### 16.3.8 Continuum Memory System (CMS)

<!-- FIG: ch16/fig-01-cms-spectrum -->

기존 backbone의 memory는 이분법이다: sequence 축을 섞는 working memory(attention)와 pre-training 지식의 persistent 저장소 feedforward layer(f=0) [NL §7]. **Continuum Memory System (CMS)** 은 이를 update frequency 스펙트럼으로 일반화한다 [NL §7.1]. CMS는 MLP block 사슬 $\mathrm{MLP}^{(f_1)},\dots,\mathrm{MLP}^{(f_k)}$이고 level $\ell$의 chunk 크기는

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

여기서 $\varepsilon(\cdot)$은 임의 optimizer의 error 항이다(GD면 $\nabla_\theta\mathcal{L}$; 실험은 단순 gradient 누적). 두 강조: 첫째, **누적되는 loss는 inner $\ell$이 아니라 task loss $\mathcal{L}$(NTP)다** [NL §7.1] — CMS는 inference 중에도 NTP truncated gradient step을 밟는 문자 그대로의 "느린 훈련"이다. 둘째, 표준 Transformer MLP block은 $k=1$·frequency 0의 퇴화 사례로, CMS는 대체물이 아니라 일반화다.

level 배선은 §16.3.4 transfer 선택에 따라 세 변형이다 [NL §7.1]. **Nested CMS**: level $s$가 level $s{+}1$의 초기 상태를 meta-learn하고($\theta^{(s+1)}_0=\arg\min_\Phi\mathbb{E}_{T\sim\mathcal{C}^{(s)}}[\ell(\Theta,T;\Phi)]$ [NL Eq. 72]) 각 block은 자기 context가 끝날 때마다 그 초기값으로 **re-initialize**된다 — higher-order ICL을 주며, TNT의 periodic reset(→ 15장)이 "throughput 트릭"에서 "level 구조의 원리"로 일반화되는 자리다. **Sequential CMS**: block을 직렬로 놓고 모든 초기 상태를 최저 주파수 level에서 일괄 meta-learn한다 [NL Eq. 73]. **Independent (head-wise) CMS**: 서로 다른 context 길이의 block을 병렬로 두고 $y_t=\mathrm{Agg}\big(\mathrm{MLP}^{(f_k)}(x_t),\dots,\mathrm{MLP}^{(f_1)}(x_t)\big)$(예: 학습된 가중합)로 결합한다 [NL Eq. 74].

CMS가 continual learning을 돕는 논리는 명시적이다 [NL §7.1]: 빠른 block $\mathrm{MLP}^{(f_s)}$가 옛 지식을 덮어써도 그 지식은 더 느린 block($s'<s$)에 살아 있고, 초기 상태의 backprop transfer가 이를 다시 빠른 block으로 **순환**시켜 완전한 소실을 어렵게 한다(12장 persistent memory·15장 "느린 곳에 주차된 지식"이 한 스펙트럼 원리로 합쳐진다).

효율 논거도 논문이 준다 [NL §7.1]. 갱신은 예정 시각의 block에만 일어나 스텝당 건드리는 parameter는 평균 $O\big(\tfrac{1}{\hat f}\cdot\tfrac{L_{\mathrm{layer}}}{5}\cdot d^2\big)$($L_{\mathrm{layer}}$ layer, 4-level, 최고 주파수 $\hat f$, 차원 $d$) — forward 전체의 작은 조각이다. (M5) 구조상 $i\not\equiv 0 \pmod{C^{(\ell)}}$ 위치엔 순차 의존이 없어 전부 병렬 처리된다 — 9장 chunkwise 스킴(→ 9장)이 그대로 적용된다.

배포 경로: **ad-hoc level stacking** [NL §7.3]은 CMS block의 초기값을 pre-trained Transformer의 MLP weights로 놓는다($\mathrm{MLP}^{(f_i)}_0 = \mathrm{MLP}_{\mathrm{pre\text{-}trained}_i}$). inner learning rate $\eta^{(\ell)}$이 보간 다이얼이다: $\eta^{(\ell)}\to 0$이면 동결 pre-trained와 동일, 키울수록 in-context 적응이 커진다. §16.6의 Llama-3 개조 레시피가 이것이다.

### 16.3.9 CMS를 optimizer에 적용하면: M3

CMS 원리는 token이 아닌 gradient context flow에도 적용된다. **M3 (Multi-scale Momentum Muon)** [NL §7.2, Alg. 1]는 gradient context flow에 CMS(independent 변형)를 적용한 proof of concept으로, §16.3.5의 진단(momentum은 43스텝 너머를 못 기억한다)에 대한 처방으로 momentum을 두 주기로 나눈다:

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

($\mu$는 aggregation 계수 — 원문 $\alpha$는 retention gate와 충돌해 개명, 표 16-2). 요약: **M3 = Adam + Muon + CMS**(independent CMS의 $\mathrm{Agg}$가 가중합). 요점: 느린 momentum이 decay 조절(AdEMAMix 등)이 아니라 **갱신을 늦추는 방식**으로 장기 문맥을 얻는다 — CMS chunked accumulation을 gradient에 옮긴 것 [NL §7.2]. 논문은 M3의 계산 오버헤드·대규모 도전을 자인한다(수치 §16.6) [NL §7.2].

### 16.3.10 self-modifying Titans

Hope의 sequence block은 [Titans]의 마지막 동결 지점을 푼다. Transformer·Titans의 projection $W_K,W_V,W_Q$·gate head는 in-context 동결이라 token contextualize가 pre-training 지식으로 상계된다: 1-layer projection은 token·위치만의 함수라 문맥에 따라 뜻이 바뀌는 단어를 원리적으로 놓치고, short convolution·canon layer도 국소 mixing일 뿐이다 [NL §8]. 처방: **모든 projection과 gate를 test-time에 갱신되는 memory로 승격시킨다.**

**1단계 — 완전 적응형 memory** [NL Eq. 79–82]. 다섯 산출 $k_t, v_t, q_t, \eta_t, \alpha_t$ 전부를 각자의 memory가 만든다:

$$
\begin{aligned}
&k_t = \mathcal{M}_k(x_t; W_{k,t-1}),\quad
v_t = \mathcal{M}_v(x_t; W_{v,t-1}),\quad
q_t = \mathcal{M}_q(x_t; W_{q,t-1}),\\
&\eta_t = \mathcal{M}_\eta(x_t; W_{\eta,t-1}),\quad
\alpha_t = \mathcal{M}_\alpha(x_t; W_{\alpha,t-1}),
\end{aligned}
$$

각 $\mathcal{M}_\square$와 본체 $\mathcal{M}_{\mathrm{mem}}$은 $\min\ \ell(\mathcal{M}_\square; k_t, v_t)$를 최적화하고 출력은 $y_t=\mathcal{M}(q_t;W_{\mathrm{mem}})$이다. 요점: **inner learning rate $\eta_t$와 retention gate $\alpha_t$(→ 13장) 자체가 memory의 출력이다** — 12장에서 slow head가 산출하던 hyperparameter가 in-context 학습 객체가 된다. 모든 초기 상태 $\mathcal{M}_{\square,0}$은 sequence 전반에 meta-learn된다 [NL §8.1].

**2단계 — self-modification** [NL Eq. 83–85]. 1단계는 모든 memory가 같은 $v_t$를 공유하는 준최적 설계라 아직 self-modification이 아니다 [NL §8.1] — 그래서 각 memory가 **자기 target을 스스로 생성한다**:

$$
\hat v_{\square,t} \;=\; \mathcal{M}_\square\big(v_t;\ W_{\square,t-1}\big)
\qquad [\text{NL Eq. 84}]
$$

— 모델이 자기 훈련 데이터를 만들어 자기 update 알고리즘을 학습한다. Schmidhuber의 self-referential weight matrix(1993)·SRWM(Irie et al. 2022) 계보를 chunk 병렬화 가능한 형태로 계승한 것이다(명칭 **self-modifying Titans**). 이 단계에서 $q_t = W_Q x_t$만은 비적응 projection으로 남는다 [NL §8.1].

**3단계 — learning rule** [NL Eq. 86–88]. token이 고도로 상관되므로 §16.3.6의 논리에 따라 inner optimizer로 dot-product GD 대신 **DGD + weight decay**를 쓴다. inner objective는 표준 $\ell(W;k,v)=\|\mathcal{M}(k;W)-v\|_2^2$, update는

$$
W_{\square,t} \;=\; W_{\square,t-1}\big(\alpha_t I \;-\; \eta_t\, k_t k_t^\top\big)
\;-\; \eta_t\, \nabla_W\, \ell\big(W_{\square,t-1};\ k_t,\ \hat v_{\square,t}\big),
\qquad \square\in\{k,v,q,\eta,\alpha,\mathrm{mem}\}
\tag{16-4}
$$

이며 read는 $y_t=\mathcal{M}(q_t;W_{\mathrm{mem},t-1})$이다. 식 (16-4)를 (M2)·(16-3)과 겹쳐 읽으면: $\alpha_t I$는 retention gate((M2)의 $\alpha_tW_{t-1}$과 동일 방향), $-\eta_t k_tk_t^\top$는 DGD의 data-dependent decay, 마지막 항은 self-generated target에 대한 gradient다. (단 우측 곱은 memory가 행렬일 때만 type이 맞으므로 linear matrix memory 특수 사례로 읽는 것이 정확하다; deep-memory 공백은 각주 2.) 여섯 memory 모두 표준 deep memory(2-layer residual MLP $\mathcal{M}_\square(z)=z+W_{\square,1}\,\sigma(W_{\square,2}\,z)$) [NL Eq. 89](구조가 같을 필요는 없으나 실험은 이 구성).

memory가 linear(행렬)인 특수 사례엔 닫힌 recurrence가 나온다 [NL Eq. 92–93]. $\ell_2$ objective면 $W_{\square,t}=W_{\square,t-1}(\alpha_t I-\eta_t k_tk_t^\top)-\eta_t\big(W_{\square,\xi}\,k_t-\hat v_{\square,t}\big)k_t^\top$($\xi$=chunk anchor, §16.4) — Gated DeltaNet(→ 6장)의 일반화된 delta rule 꼴이다. dot-product objective면 gradient 항이 $\hat v_{\square,t}k_t^\top$로 준다 [NL Eq. 92](원문 인쇄 부호는 (M1)과 반대 — 부호 관행 차이, 표 16-2).

두 정직한 각주를 남긴다. 첫째, 식 (16-4)의 $\square$ 집합엔 원문 그대로 $q$가 포함되나 [NL Eq. 88, 96], 같은 절 본문은 $q_t=x_tW_q$를 유일한 비적응 projection으로 명시하고 [NL §8.1] ablation에서 "w/o inner-projection $q$"가 perplexity를 오히려 **개선**한다(12.19 vs 12.24 [NL Table 6]). 최종 설계는 $q$를 정적으로 읽는 것이 정합적이며, 이는 "level 추가가 항상 이득은 아니다"라는 논문 자신의 프레임에 대한 내부 반례다(§16.8). 둘째, deep memory(2-layer MLP)에서 $(\alpha_t I-\eta_t k_tk_t^\top)$ 항이 어느 행렬에 어떻게 적용되는지는 원문이 명시하지 않는다(read 시점 표기 $\mathcal{M}_{\mathrm{mem},t}$ vs $\mathcal{M}_{\mathrm{mem},t-1}$의 흔들림도 배포형 chunkwise에서는 chunk 경계 snapshot으로 수렴해 소멸한다).
<!-- VERIFIED(2026-07-12): §8.1–8.2 정독 — Eq.88/90의 우측 곱은 M이 2층 residual MLP(Eq.89/91)일 때 적용 대상 미정의(원문 그대로의 공백). Eq.92–93(행렬 memory)만 타입 명확. 공식 구현 부재로 코드 대조 불가. -->

### 16.3.11 Hope block

**Hope** [NL §8.3]는 두 산물을 한 block에 직렬로 잇는다: **self-modifying Titans**(작은 state, 표현력 있는 DGD rule) 뒤에 **CMS**(큰 capacity, 단순한 rule)를 붙인다 — CMS는 지속 저장을 위해 capacity가 커 단순 rule, self-modifying Titans는 state가 작아 표현력 rule로 상보적이다 [NL §8.3]. forward는 (normalization·convolution 생략)

$$
z_t = \mathcal{M}\big(q_t;\ W_{\mathrm{mem},t-1}\big)
\quad\text{(식 (16-4)의 갱신과 함께)},
\qquad
y_t = \mathrm{MLP}^{(f_k)}\big(\cdots\, \mathrm{MLP}^{(f_1)}(z_t)\big)
\qquad [\text{NL Eq. 94–97}]
$$

이다. 실무 세부 [NL §8.3]: $q,k$는 $\ell_2$ normalize(식 (16-3) DGD의 $\|x\|=\lambda$ 전제와 정합), window 4 local convolution. 변형 **Hope-Attention**은 self-modifying Titans를 softmax global attention으로 치환한 통제 변형으로 CMS 기여를 분리 측정한다 [NL §8.3].

12장 어휘로: Hope는 Titans-MAC/MAG처럼 attention과 memory를 병렬 합성한 것이 아니라, Titans block의 **모든 구성요소를 한 level씩 위로 올리고**(projection·gate의 memory화) 뒤따르는 MLP를 CMS 주파수 스펙트럼으로 펼친 것이다. Titans의 persistent/long-term/short-term 삼분류(→ 12장)는 연속체의 세 점으로 흡수된다.

### 16.3.12 표기 대응표

표 16-2 — [NL] 원 표기 ↔ 통일 표기 대응표 (STYLE-NOTATION §1.7.5의 표를 그대로 복사; "(장-국소 추가)"로 표시된 행은 이 장에서 더한 것)

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
| **(장-국소 추가)** $o_t$ — Hope 내부(self-modifying Titans)의 중간 출력 ([NL Eq. 94–97]) | $z_t$ | 최종 출력 $y_t$·pre-activation $z_\ell$과 첨자로 구분 (ch12의 중간항 개명 패턴) |

## 16.4 Outer-loop training vs inner-loop test-time learning

이 절의 제목 자체가 심문 대상이다. [NL]의 핵심 주장이 "outer/inner 이분법은 $K$-level 스펙트럼의 $K=2$ 특수 사례"이고 극단적으로 "pre-training도 corpus를 context로 하는 ICL이라 training/test 경계가 없다"이기 때문이다 [NL §6]. 그러나 **실제로 실험된 Hope**는 이 책의 이분법대로 만들어지고 서빙된다: 한 번의 outer(pre-training) 뒤 inference에서 정해진 rule로 움직이는 상태들이 있다. 이 절은 그 구분을 Hope에 전개한 뒤 어디까지가 이분법이고 어디부터 스펙트럼인지 표시한다.

표 16-3 — Hope의 level 지도: 무엇이 어디서 누구에 의해 갱신되는가

| 구성요소 | 층위·주체 | 갱신 주기 | 갱신 rule |
|---|---|---|---|
| embedding, $W_Q$, convolution, normalization | slow $\Theta$ (outer) | pre-training에서만 | AdamW |
| 여섯 memory의 초기 상태 $\mathcal{M}_{\square,0}$ (각 2-layer residual MLP) | slow $\Theta$ meta-learn (outer, transfer 4) | pre-training에서만 | AdamW — gradient가 unrolled inner를 관통 |
| CMS 각 level의 초기값 $\theta^{(\ell)}_0$ | slow $\Theta$ meta-learn (outer, transfer 3+4) | pre-training에서만 | Sequential이면 최저 주파수 level에서 일괄 backprop [NL Eq. 73] |
| gate $\eta_t,\ \alpha_t$ | inner 파생값 (값=inner, 산출 함수=outer) | 매 token (chunk 경계 snapshot) | $\mathcal{M}_\eta,\mathcal{M}_\alpha$의 forward |
| self-target $\hat v_{\square,t}$ | inner 파생값 (self-generated) | 매 token | $\mathcal{M}_\square(v_t;W_{\square,\xi})$ |
| fast weights $W_{\square,t}$, $\square\in\{k,v,\eta,\alpha,\mathrm{mem}\}$ | fast $W$ (inner) | $C_\square$ token마다 일괄 (훈련·서빙 동일) | DGD + weight decay, 식 (16-4) (+ momentum — 각주) |
| CMS parameters $\theta^{(\ell)}_t$ | 중간 주파수 (inner, objective=outer $\mathcal{L}$) | $C^{(\ell)}$ token마다 | (M5): task-loss error 누적 [NL Eq. 71] |
| AdamW의 $m_t, h_t$ (훈련에만) | optimizer 자신의 level | 훈련 스텝마다 | EMA (→ 2장); "end of pre-training"에서 폐기 [NL §4.5] |

### 16.4.1 outer loop: 무엇이 meta-learn되고, gradient는 어떻게 inner를 관통하는가

pre-training이 학습하는 것은 표 16-3의 위 세 행 전부다. outer objective는 4장 MAML형 bilevel(→ 4장) 그대로로 **outer gradient가 inner-loop 갱신 경로(각 memory의 per-chunk DGD recurrence (16-4)·CMS 갱신 (M5))를 관통**해 $\partial \mathcal{L}/\partial \mathcal{M}_{\square,0}$이 unrolled 경로를 거슬러 계산된다. training 무경험 독자에게: "optimizer를 통과하는 backprop"이라 훈련이 비싸고 서빙에서는 안 보인다.

훈련을 성립시키는 것은 chunkwise 병렬화다 [NL §8.2]. 길이 $L$을 $\lceil L/C\rceil$개 chunk로 자르고(각 memory의 update frequency $f_\square = L/C_\square$) 두 가지를 chunk 경계 snapshot에 고정한다: chunk anchor $\xi:=C_\square\lfloor (t-1)/C_\square\rfloor$에 대해 파생값 $k_t,v_t,\eta_t,\alpha_t$와 self-target $\hat v_{\square,t}=\mathcal{M}_\square(v_t;W_{\square,\xi})$는 전부 snapshot 상태 $W_{\square,\xi}$에서 생성되고, fast weight는

$$
W_{\square,t} \;=\; W_{\square,t-1}\big(\alpha_t I-\eta_t k_tk_t^\top\big)
\;-\;\eta_t\,\nabla_W\,\ell\big(W_{\square,\xi};\ k_t,\ \hat v_{\square,t}\big)
\qquad [\text{NL Eq. 90}]
\tag{16-5}
$$

— (i) 다음 chunk 전체의 key/value/self-target/gate를 직전 chunk 끝 상태에서 **한 번의 배치 pass**로 생성하고 (ii) inner gradient도 그 snapshot 기준으로 평가한다. 그러면 chunk 내 모든 gradient가 chunk 처리 전에 병렬 계산 가능해져 TTT dual form(→ 8장)·Titans 병렬 훈련(→ 12장)이 그대로 적용된다(chunk 크기는 실무상 둘: $\mathcal{M}_{\mathrm{mem}}$용과 나머지 다섯 공용) [NL §8.2].

각주(표기 슬립): 원문 [NL Eq. 90]의 snapshot 인덱스 $C\times\lceil t/C\rceil$는 문면상 **현재** chunk 끝(아직 없는 상태)을 가리키나, 같은 절 산문("직전 chunk의 마지막 상태" [NL §8.2])이 의도를 확정하므로 [TNT Eq. 5–6] 슬립(→ 15장)과 같은 유형으로 $\xi(t,C)$로 통일한다.

§16.3.2에서 backprop이 병렬화되지 않는 이유는 value의 자기참조성이었다. self-modifying Titans는 그 자기참조성을 **아키텍처로** 가져온 모델인데, 왜 훈련이 병렬화되는가? 답: **chunk 안에서 value 생성기를 stale snapshot으로 동결했기 때문이다.** 정확한 self-reference(현재 상태가 target 생성)는 여전히 순차적이고 [NL §4.5], chunkwise stale-snapshot 근사(→ 9장 (M4), "semantic hyperparameter" 명제)가 병렬성을 사는 공학적 타협이다. 배포된 Hope는 정확한 self-modifying 모델의 근사물이며 chunk 크기 $C_\square$가 충실도–throughput 다이얼이다. SRWM이 정확한 self-reference를 갖고도 병렬 훈련이 없어 못 간 스케일을 Hope가 가는 이유가 이 근사이고 [NL §9.5], 그 대가(충실도 손실의 정량)는 논문에 없다(§16.8).

### 16.4.2 inner loop: inference에서 무엇이 어떤 rule로 움직이는가

decode 중 token 하나가 들어오면 Hope block의 일은 세 겹이다. (1) **fast memory들**($\mathcal{M}_k$, $\mathcal{M}_v$, $\mathcal{M}_\eta$, $\mathcal{M}_\alpha$, $\mathcal{M}_{\mathrm{mem}}$; $\mathcal{M}_q$ 적응 여부는 원문 상충이라 — §16.3.10 각주 1 — 다섯(정적)~여섯(적응형), 각 2-layer residual MLP)가 식 (16-5) 스케줄로 갱신된다. chunkwise 스케줄의 $1/C_\square$ 상환은 **훈련·prefill**에서만 성립한다(chunk 전체 token이 이미 있어 한 batch로 계산). autoregressive **decode**는 미래 token이 아직 없어 다음 chunk를 미리 batch할 수 없다: 이상적 $C=1$ decode는 매 token 순차 갱신(상환 없음, → 15장 TNT의 목표), $C>1$ decode는 $C$개를 버퍼링해 경계에서 지연 batch-update(chunk-size mismatch 위험, → 9·15장)하는 형태다 — 어느 쪽이든 정확한 decode 비용·latency는 공개 구현·측정이 없어 미확정이다. gate $\eta_t,\alpha_t$는 별도 학습기가 아니라 이 memory들의 forward 산출이다(표 16-3). (2) **CMS의 각 level**이 $C^{(\ell)}$ token마다 (M5)로 갱신된다 — inference 중에도 NTP truncated gradient step을 기하급수 간격으로 계속 밟는다. (3) 나머지($W_Q$·embedding·conv·norm)는 움직이지 않는다.

state는 KV cache처럼 $O(L)$로 자라지 않고 시퀀스 길이에 상수다(산수는 §16.7). fast memory엔 별도 test-time optimizer가 필요 없다(DGD rule·gate가 곧 optimizer). 단, ablation 표의 "w/o Momentum" 행(13.58 vs 12.24 [NL Table 6])과 §9.6 산문("removes the momentum term in the self-modifying Titans")은 배포된 inner rule이 식 (16-4)에 없는 Titans식 momentum 항 $S_t$(→ 12장)를 실제로 지님을 말해 준다 — 그 정확한 수식 위치는 원문 어디에도 없다.
<!-- VERIFIED(2026-07-12): §9.6이 momentum 항 존재를 명시(ablation 행 설명), 수식(Eq.88/90)에는 부재 — 간극 자체가 원문 사실. 공식 구현 부재. -->

마지막으로 이분법/스펙트럼의 경계: outer/inner 이분법이 완전히 성립하는 것은 **한 번의 pre-training 뒤 동결 서빙**뿐이고, CMS를 계속 살려 두는 continual 배포에서는 중간 주파수 level이 "훈련도 추론도 아닌" 제3의 지위(→ 1장 불변식의 영구 위반)를 차지한다 — §16.7·17장 lifecycle으로 이어지는 문이다.

## 16.5 Concept ledger delta

표 16-4 — [NL]이 개념 원장에 가한 변경 (기원 표기: 관련 장)

| 구분 | 개념 | 내용 / 전작과의 관계 |
|---|---|---|
| 신규 | **Nested Learning / Nested System / NSAM** [NL Def. 3, 4] | 모델+훈련 절차 = 중첩 최적화 시스템; §1.6 카탈로그 전체를 포섭하는 상위 프레임 |
| 신규 | **context flow** | 각 level이 압축하는 데이터 스트림 (token/gradient/상위 신호) |
| 신규 | **update frequency와 level** [NL Def. 2] | attention $f=\infty$, frozen MLP $f=0$; 높은 level=낮은 주파수. 17장 [Sleep]이 재사용 |
| 신규 | **Neural Learning Module** | 아키텍처×optimizer 결합이 설계 단위; Transformer+SGD ≠ +Adam |
| 신규 | **Local Surprise Signal (LSS)** $u_t$ | surprise(→ 12장)의 출력 공간 대응물; backprop 재해석의 축 |
| 신규 | backprop = **self-referential** associative memory | 훈련 병렬화 불가의 구조적 이유; Schmidhuber 1993 계보 복권 |
| 신규 | **knowledge-transfer taxonomy** (5 기제) | meta-learned $W_{\mathrm{init}}$(→ 12·15장)이 기제 4로 공식 분류 |
| 신규 | **DGD / GGD / GM / Delta Momentum / DMGD** | delta rule(→ 5장)·Miras objective 축(→ 13장)의 optimizer 층위 이식 |
| 신규 | **CMS** (+ Nested / Sequential / Independent) | TNT global/local hierarchy(→ 15장)의 아키텍처화; long/short-term 이분법의 스펙트럼화 |
| 신규 | **self-modifying Titans** | Titans(→ 12장) projection·gate 전부의 memory화 + self-generated target |
| 신규 | **Hope / Hope-Attention** | 라인의 새 기준 모델; CMS 분리 측정용 통제 변형 |
| 신규 | **M3** | CMS를 gradient context flow에 적용; momentum-as-memory의 생성적 산물 |
| 신규 | **ad-hoc level stacking** | pre-trained MLP → CMS 초기값; retrofit 배포 경로 |
| 신규 | **parametric vs non-parametric ICL** | Transformer=non-parametric 해, recurrent/TTT=parametric; emergent 아닌 구조적 |
| 신규 | **CTNL** benchmark | MTOB(Kalamang)+Manchu 순차 학습으로 in-context CF 측정 (CF 축 → 11장) |
| 확장 | momentum-as-memory | 12장 관찰($S_t$=past surprise) → 정리: 모든 gradient optimizer가 gradient의 memory; capacity 산수(6/43)까지 정량화 |
| 확장 | surprise | momentary/past(→ 12장) → LSS로 출력 공간 확장; backprop 전체가 surprise 기반 memory |
| 확장 | retention gate (→ 13장) | 13장 소유 그대로; DGD가 같은 원리(data-dependent decay)를 **learning rule 층위**에서 재생산 |
| 확장 | meta-learned $W_{\mathrm{init}}$ (→ 4·15장) | TNT load-bearing 장치 → 5대 transfer 중 기제 4로 일반화 |
| 확장 | periodic reset (→ 15장) | TNT throughput 트릭 → Nested CMS의 context-end re-initialization |
| 확장 | persistent memory (→ 12장) | prefix token → 주파수 0의 극점; init 미-meta-learn 시 gating이 대역을 맡는 corollary |
| 확장 | Omega rule (→ 14장) | 그대로 수입; GGD의 $\mathrm{Ret}$ 자리에 위치 지정 |
| 폐기 | 아키텍처 vs optimizer 구분 | 같은 객체(associative memory)의 다른 level — 논문 제목의 한 수 |
| 폐기 | long-term/short-term 이분법 | 주파수 연속체로 대체 |
| 폐기 | "test-time" learning/memorization 명칭 | continual 설정에서 오도적 — parametric ICL로 대체 제안 [NL §6] (이 책은 명칭 유지, 이의만 기록) |
| 폐기 | ICL의 emergent 지위 | 2개 이상 level의 구조적 귀결 (단, 좋은 성능엔 잘 훈련된 저주파 level 필요 [NL §6]) |

## 16.6 실험과 스케일

**공통 setup.** 언어 모델 실험은 FineWeb-Edu+장문 혼합 corpus, 32K vocab, AdamW(모델별 lr 튜닝, 나머지 [Titans] 기본)로 from scratch 훈련이다 [NL §9.2, §9.3]. 두 스케일: **760M / 30B tokens**와 **1.3B / 100B tokens** [NL §9.3]; RULER는 약 50B tokens 훈련 [NL §9.2]. baseline은 Transformer++, Samba, RetNet, DeltaNet, RWKV-7, Comba, TTT, Miras(Memora), DLA, Titans이다.

**언어 모델링과 상식 추론** [NL Table 2]. 760M/30B에서 Hope는 Wikitext ppl 18.68, LAMBADA ppl 20.07, 8개 상식 추론 평균 acc 52.28로 전 baseline 최고다(차점 Titans 51.68). 1.3B/100B에서 Wikitext 14.39, LAMBADA 10.08, 평균 58.04(Titans 56.82)로, attention-free 대비 격차가 스케일과 함께 커진다 [NL §9.3].

**RULER needle-in-a-haystack** [NL Table 1]. Hope는 attention-free 중 전 과제·난이도 최고다: S-NIAH-3(UUID) 16K에서 24.8 vs Titans 21.2·RWKV-7 5.8, MK-NIAH 16K에서 14.8 vs Titans 8.2 — linear memory가 붕괴하는 곳에서 deep memory capacity가 버틴다 [NL §9.2]. 통제 변형 Hope-Attention은 S-NIAH-1 4K/8K/16K 모두 100으로 같은 attention의 Transformer(88.6/76.4/79.8)를 상회 — 이 증분은 CMS의 몫이다.

**BABILong** [NL Fig. 9]. 원 설정대로 fine-tune한 소형 모델 비교에서 Titans·ARMT·Hope가 1M token까지 경합하다 Titans·ARMT는 급락하고 **Hope는 10M token까지 유지된다**(논문은 CMS 설계를 원인으로 지목) [NL §9.2]. 정직성 항목: 이 10M은 **fine-tuning 전제**다 — zero-shot에서는 Hope 포함 모든 소형 모델이 크게 떨어지며, 저주파 level이 과제에 맞게 조정돼야 고주파 memory 관리가 작동한다 [NL §9.2].

**continual learning** [NL Fig. 6, Fig. 8]. class-incremental 분류(CF 측정 프로토콜의 배경은 → 11장) 세 벤치마크 — CLINC(150 intent), Banking(77 intent), DBpedia(70 클래스) — 에서, Llama3-8B/Llama-3B를 ad-hoc level stacking(§16.3.8)으로 개조하고 15B tokens continual pre-training한 Hope 변형이 ICL, EWC(→ 11장), 외부 학습기 SOTA인 InCA를 전부 상회한다 [NL §9.1]. 신설 benchmark **CTNL**(MTOB Kalamang+Manchu; ChRF)에서는: 언어 하나씩은 Hope 변형이 ICL과 동급 이상, **두 언어를 순차로** 배우면 ICL은 붕괴(pre-training으로 회귀)하는 반면 Hope-1/2/3(추가 level 1/2/3개)은 level 수에 단조 개선되어 Hope-3는 단일 언어 성능을 거의 회복한다 [NL §9.1]. level ablation [NL Fig. 7]: level이 많을수록 좋고, 최저 주파수는 512가 최고이되 2K가 근접 성능에 훨씬 저렴해 효율 sweet spot이다 [NL §9.1].

**in-context recall과 MAD** [NL Table 3, 4]. 의무 서술 caveat이 여기 있다: 짧은 in-context recall에서는 **여전히 Transformer가 최고이고 격차가 크다** — FDA에서 Transformer 67.3 vs Hope 41.9, SWDE 71.4 vs 65.9. Hope는 attention-free 중 최고(Titans·RWKV-7·Comba 전부 상회)로 격차를 좁혔을 뿐이다 [NL §9.4]. [Atlas]의 53.55 vs 43.70([Atlas Table 5], → 14장)에서 확인된 in-context retrieval gap이 이 라인의 종합판에서도 **미해소**라는 뜻이다. 반면 합성 벤치마크 MAD에서는 Hope가 Transformer를 포함한 전부를 이긴다(compression 51.2, fuzzy ICR 52.1, memory 85.2) [NL Table 4].

**formal language 인식** [NL Table 5]. Irie et al. 2023 구성의 parity, $(aa)^*$, $(abab)^*$, $a^nb^n$, $a^nb^nc^n$, Shuffle-2에서 Hope는 훈련 분포 밖 길이 구간(Bin1)을 포함해 **전부 100.0**이다(Transformer는 parity Bin1 0.0으로 실패, linear attention·DeltaNet도 Bin1 대부분 0). LSTM·SRWM도 전 항목 100이나 병렬 훈련이 불가하다 — Hope의 차별점은 "비선형 recurrence급 state tracking + 병렬 훈련"의 양립이다 [NL §9.5]. 정직성 항목: 이것은 **실증이지 정리가 아니다** — [NL]엔 $K$-level NSAM의 표현력 정리가 없고 state-tracking 실험이 그 자리를 대신한다(→ §16.8; 12장 [Titans Thm 4.1]과 같은 계열).

**component ablation** [NL Table 6] (760M급, 구성요소를 하나씩 제거).

표 16-5 — Hope 구성요소 ablation ([NL Table 6] 재구성; 언어 모델링 평균 ppl ↓ / 상식 추론 평균 acc ↑)

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

읽는 법: DGD·momentum·weight decay·CMS·inner $k$·inner $v$는 전부 제거 시 악화(특히 inner $v$(self-target 원료) 동결이 acc를 가장 크게 깎음, 55.1). 유일한 예외가 inner $q$ 동결로 ppl이 오히려 개선된다(12.19) — §16.3.10 각주·§16.8 "level 배치는 경험적" 한계로 직결된다.

**M3** [NL §9.7]. ViT-24M/86M을 ImageNet-21K에서 optimizer만 바꿔 훈련하면 M3가 AdamW·Muon 대비 최저 train/test loss를 준다 [NL Fig. 11]. 효율은 별도: Transformer LM(140M·1.3B) 훈련에서 momentum이 여럿·NS pipeline 추가라 **Muon보다 느리고 AdaMuon과 동급**이다 [NL Fig. 12] — 품질 이득이 비용과 함께 온다.

**스케일의 정직한 상한.** from-scratch 실증 상한은 라인 공통 **1.3B / 100B tokens**(retrofit은 Llama3-8B까지)이고, Hope의 decode throughput·latency·메모리 수치는 논문에 없다 — wall-clock 측정은 M3(훈련 시간)뿐으로, 6편 공통의 decode wall-clock 부재가 종합판에서도 이어진다. 7B+ 및 production Transformer 대비 우위는 열린 문제다(§16.8).

## 16.7 Systems/serving 함의

**state의 산수: KV cache가 아니라 per-request weights.** Hope block 하나의 inference state는 여섯 개의 2-layer residual MLP memory(memory당 행렬 2개, 도합 **약 12개의 $d\times d_h$급 행렬**)에 inference 중 표류하는 CMS level별 MLP weights를 더한 것 — KV cache의 $O(L\cdot d)$ 성장과 달리 시퀀스 길이에 상수이되 그 상수가 weights 규모다(crossover는 단일 matrix memory(→ 6장)보다 훨씬 오른쪽). 질적 차이: 이 state는 read-only cache가 아니라 **mutable weights**라 per-request로 달라 shared-weight batching이 깨지고(→ 1장 Rosetta: grouped-GEMM decode), checkpoint/restore는 이 텐서들의 영속화다. Titans/TTT/Atlas와 같은 문제 클래스이되 fast-weight 텐서가 약 6배이고 chunk 주기 갱신이 이를 부분 상쇄한다.

**decode state의 RMW 트래픽.** 각 fast memory는 $C_\square$ token마다 자기 parameter 텐서 전체를 read-modify-write하므로 token당 상환 write 트래픽은 (state bytes)/$C_\square$ — chunk 크기가 대역폭 다이얼이다. CMS도 level별로 같은 구조를 반복하며 상환 추정은 §16.3.8의 $O\big(\tfrac{1}{\hat f}\cdot\tfrac{L_{\mathrm{layer}}}{5}\cdot d^2\big)$이다 [NL §7.1]. "최저 주파수 2K가 sweet spot" [NL §9.1]은 이 트래픽 상환이 품질을 크게 잃지 않는다는 보고다.

**update frequency ↔ memory 계층 배치.** 주파수 스펙트럼 = 배치 스펙트럼: 매 chunk 갱신되는 여섯 fast memory는 decode 중 상시 접근되는 hot set(HBM 상주)인 반면, $C^{(\ell)}=2\mathrm{K}$급 CMS level은 수천 token에 한 번만 RMW돼 더 차가운 계층을 허용한다 — "level 번호가 곧 cache tier 힌트" [NL §7.1].

**병렬화와 kernel 형태.** 훈련·prefill은 §16.4의 chunkwise dual 스킴이다. memory가 MLP이므로 kernel은 TTT-MLP형 **batched small-GEMM 시퀀스**이고(→ 8·9장), linear 특수 사례(식 (16-5) 행렬 버전)는 일반화된 delta rule이라 DeltaNet 계열 WY/UT-transform chunk kernel(→ 9장)이 자연 템플릿이다. 두 chunk 크기($\mathcal{M}_{\mathrm{mem}}$용·나머지용)는 staleness를 따로 튜닝하는 kernel 자유도다. decode의 정확한 update 비용($C=1$ 순차 vs $C>1$ 경계-지연 batch)은 미측정이고(§16.4.2), 짧은 recall에서는 Transformer가 이긴다는 crossover가 [NL Table 3] 그대로다.

**retrofit 배포 경로.** ad-hoc level stacking(§16.3.8)은 이 라인 최초의 현실적 이식 스토리다: 기존 Transformer MLP를 CMS 초기값으로 삼고 15B tokens continual pre-training으로 개조해 [NL §7.3, §9.1] from-scratch 없이 채택 장벽을 우회한다.

> **[평가]** 가장 파괴적인 함의는 수식이 아니라 배포 모델에 있다. continual 배포에서 모델은 **서빙 중 사용자 데이터로 자신의 NTP truncated gradient step을 계속 밟는다** — train/serve 경계의 소거는 서빙 불변식의 영구 폐기다. per-tenant weight 발산·rollback·provenance·안전성, speculative decoding 상호작용(→ 1장) — 어느 것도 논문이 다루지 않고 Hope의 서빙 경제학(throughput/latency/메모리)은 수치가 전무하다. 이 라인의 서빙 문제는 종합판에서 풀리기는커녕 6배로 늘었다.

## 16.8 한계와 bridge-out

논문이 스스로 그은 한계선부터 정리한다.

**catastrophic forgetting(→ 11장)은 해결되지 않았다.** forgetting은 압축의 귀결(유한 capacity는 새 정보를 위해 잊도록 강제)이며, Hope·CMS는 실험된 과제에서 이를 **줄였을** 뿐 일반적으로 "풀지" 않았다 [NL §10]. CMS가 하는 일은 어디서·얼마나 빨리 잊는지를 주파수 축에 재배치하는 것 — level 간 capacity의 원리적 배분은 열려 있다.

**level 설계는 경험적이다.** level 수·주파수·구성요소 배치에 대한 이론이 없다. inner $q$ ablation(12.19 vs 12.24 [NL Table 6])은 잘못 놓인 level이 해가 됨을 보이는 내부 반례이고, chunk 크기·level 수·주파수는 전부 손으로 고른 hyperparameter다(2K sweet spot도 실험적 발견). 주파수 스케줄 학습은 미착수다.

**이론은 재해석까지만이다.** optimizer 역공학은 존재 논증이고(§16.3.5 [평가]), $K$-level NSAM이 level 수에 따라 오르는 표현력 계층에 대한 정리는 없다 — formal language 100.0은 실증이다(§16.6). [Titans Thm 4.1]의 증명 부재(→ 12장)라는 이론 공백이 종합판에서도 이어진다.

**self-reference의 충실도 분석이 없다.** chunkwise stale-snapshot 근사가 정확한 self-modifying 모델을 얼마나 훼손하는지(chunk 크기에 대한 오차 한계)는 어디에도 없다(→ 9장 라인 공통 caveat). 정확한 self-reference는 여전히 순차적이고, 더 나은 병렬화 존재 여부는 열린 문제다.

**서빙 경제학과 스케일.** §16.6–16.7대로: Hope의 wall-clock 부재, per-request mutable weights의 미해결 인프라, from-scratch 1.3B/100B 상한, M3의 자인된 오버헤드·대규모 미검증, 그리고 "architecture-specific optimizer"는 §16.3.4에서 근거까지 두고도 설계는 미래 과제로 남았다 [NL §6]. in-context retrieval gap(FDA 67.3 vs 41.9)도 미해소, Cartridges류 비교도 유보다 [NL §9.1].

> **[평가]** 이 장의 프레임으로 라인을 되돌아보면 [NL]의 기여는 두 겹이다. 재해석의 층("모든 것이 associative memory다")은 검증 불가능한 관점 선언에 가깝지만, 그 관점이 **생성한 물건** — DGD, CMS, self-modifying Titans, M3 — 은 ablation으로 기여가 측정된 구체물이다(표 16-5). 관점의 가치는 참·거짓이 아니라 생성물 성능으로 정산되며, 그 정산서엔 inner $q$ 반례와 미측정 서빙 비용이 함께 적힌다.

**Bridge-out: [Sleep]으로.** [NL]이 다음 논문에 넘기는 것은 스스로 범위 밖으로 선언한 반쪽이다. §16.2에서 신경생리학은 두 단계의 consolidation을 말했다: [NL]이 구현한 것은 **online consolidation** — 입력이 흐르는 동안, CMS의 주파수 스펙트럼으로 — 뿐이고, 수면 중 replay로 기억을 재조직하는 **offline consolidation**은 명시적으로 다루지 않았다 [NL §1.1]. 따라서 [NL]의 세계에서 모든 적응은 입력이 흐르는 동안 일어나고, capacity는 고정이며, 그래서 forgetting은 필연으로 남는다. 빠른 level의 지식이 예정된 갱신으로 덮어써지기 전에, 그것을 **어디로 옮길 것인가?**

[Sleep](→ 17장)의 답은 train/test 구분의 잔재를 마저 지우고 **wake/sleep lifecycle**을 세우는 것이다. wake는 [NL]의 CMS online consolidation 그대로다(update frequency 정의 Def. 2·CMS 갱신식 (M5)를 [Sleep]이 문자 그대로 수입). sleep은 두 offline 프로세스다: (1) Memory Consolidation — 빠른 block의 예정된 갱신이 지식을 덮어쓰기 직전 그 지식을 다음 느린 block에 **새 low-rank expert로 위쪽 distillation**(Knowledge Seeding)하고 빈 block은 reset(synaptic pruning) — forgetting을 regularization이 아니라 **capacity 성장**으로 재정의한다; (2) Dreaming — 자기 생성 데이터로 자기를 강화하는 RL loop. sleep이 발화하는 시점은 정확히 CMS의 chunk 경계다 — TNT의 reset → Nested CMS re-initialization → [Sleep]의 "consolidation 후 reset" memory 위생 원리로 이어진다. [NL]이 "잊히기 전에 옮길 곳이 없다"는 문제를 만들었다면, 17장은 옮기는 절차 — 그리고 이 라인 전체의 완성형 — 를 다룬다.


