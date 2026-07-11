# ch14. Atlas: Learning to Optimally Memorize the Context at Test Time

## 14.1 Bridge-in: Miras가 남긴 문제

이 장의 논문은 [Atlas] (*Atlas: Learning to Optimally Memorize the Context at Test Time*, arXiv:2505.23735; Ali Behrouz, Zeman Li, Praneeth Kacham, Majid Daliri, Yuan Deng, Peilin Zhong, Meisam Razaviyayn, Vahab Mirrokni, Google, 2025-05-29 v1)다. 출발점은 13장이 닫으며 남겨 둔 세 개의 공백이다.

[Miras] (*It's All Connected*, arXiv:2504.13173; 이 책은 framework 이름 Miras로 통칭한다)는 sequence model 전체를 (memory 구조 × attentional bias × retention gate × learning algorithm)의 4축 설계 공간으로 재편했다. 그러나 그 4축 중 네 번째 축 — inner loop의 **learning algorithm** — 은 선언만 되고 사실상 비어 있었다. Miras가 실제로 출하한 세 모델 Moneta·Yaad·Memora는 전부 plain gradient descent로 inner objective를 최적화하며, Newton법류의 고차 optimizer는 명시적으로 future work로 미뤄졌다(→ 13장). 첫 번째 공백이다.

두 번째 공백은 objective의 시야다. Titans도 Miras의 세 모델도, 그리고 DeltaNet 계열 전부가, inner loss를 **현재 token 하나의 (k_t, v_t) 쌍**에 대해서만 세운다. 식 (M1)/(M2)의 $\ell(W_{t-1};k_t,v_t)$가 그것이다. 어떤 모델도 "직전 $c$개 token이 **함께** 잘 저장되어 있는가"를 objective 수준에서 묻지 않았다. [Titans] (*Titans: Learning to Memorize at Test Time*, arXiv:2501.00663)의 momentum이 부분적 보완이기는 하다 — [Atlas §4.1]의 각주는 Titans를 "모든 과거 token에 대해 momentum이 만들어 내는 implicit decay 가중치로 memory를 최적화하는, 병렬화를 유지한 예외"로 자리매김한다. 그러나 momentum은 **per-token gradient들의 지수 감쇠 합**일 뿐, loss 자체는 여전히 token 하나짜리다. surprise(→ 12장)의 언어로 말하면, 여러 token에 걸친 사건은 어느 한 token도 개별적으로는 놀랍지 않으면서 전체로는 기억할 가치가 있을 수 있는데, per-token loss는 이 경우를 구조적으로 놓친다.

세 번째 공백은 이론이다. 이 라인은 "고정 크기 state에 문맥을 압축한다"를 반복해 왔지만, 그 state가 **몇 개의 연상을 저장할 수 있는지**에 대한 정량적 정리는 Titans에도 Miras에도 없다. 5장의 고전 Hopfield capacity는 binary pattern과 outer-product write에 대한 결과라서, deep MLP memory와 gradient 기반 write에는 그대로 이식되지 않는다.

[Atlas]는 이 세 공백을 한 논문에서 전부 메우겠다는 논문이다. optimizer 축에는 Muon(→ 2장)을 inner loop 안으로 이식하고, objective 축에는 sliding-window inner loss(**Omega rule**)를 도입하며, 이론 축에는 memory capacity의 형식적 정의와 세 개의 정리를 제공한다. 그리고 부산물로 — 사실은 부산물 이상인데 — softmax attention 자체를 "capacity가 무한한 associative memory"로 회수하고, 그 자리에서 Transformer의 strict generalization 두 가족(DeepTransformers, Dot)을 파생시킨다.

## 14.2 문제의식과 논문의 핵심 주장

논문이 스스로 정의한 문제는 현대 recurrent model의 세 가지 설계 결함이다 [Atlas §1]. (1) **online 갱신**: memory가 현재 token만 보고 최적화되고 이전 state는 retention으로만 유지된다 — 개별 token의 greedy memorization이며, 문맥 전체가 잘 저장되었는지는 아무도 묻지 않는다. (2) **제한된 memory capacity**: 구조와 key-value feature mapping이 "완벽하게 매핑 가능한 쌍의 수"를 제한한다. (3) **표현력 없는 memory 관리**: inner optimizer가 거의 전부 1차 gradient descent라서, token 동역학의 1차 정보에만 의존해 나쁜 local minima에 수렴하고 질 낮은 key→value 매핑을 배울 수 있다. 이 셋이 각각 Omega rule, feature map, Muon으로 대응된다는 것이 논문의 구도다.

이 논문에는 라인 전체의 어휘를 바꾸는 용어 주장이 하나 있다. [Atlas §1]은 "**test-time memorization**"이라는 표현을 "test-time training" 대신 쓰겠다고 명시한다. 근거는: inner loop가 하는 일은 현재 global context 안에서의 저장과 인출뿐이고, pre-training으로 학습된 core parameter(outer loop)와 초기 state는 전혀 갱신되지 않으며, memory를 비우고 나면 새로운 독립 context로 이월되는 persistent learning이나 skill 습득이 없다는 것이다. 이 책도 Atlas 이후 문맥에서 이 구분을 존중한다 — inference 시점에 weights가 변한다고 해서 checkpoint가 학습되는 것이 아니다. 이 구분 자체가 이 장의 논점 중 하나이며, §14.4에서 두 loop의 경계로 다시 정확히 그린다.

핵심 주장을 논문 자신의 분류표로 요약하면 이렇다. [Atlas Table 1]은 현대 recurrent model들을 다섯 성질로 비교한다: (1) dynamic decay(data-dependent retention), (2) deep neural memory, (3) non-linear capacity(feature map에 의한 초선형 capacity), (4) **locally optimal**(token에 대한 근사 2차 정보로 memory를 관리), (5) **flexible context**(문맥의 어느 부분을 기억할지 유연하게 선택). attention과 SWA는 (4)(5)를 non-parametric하게 갖지만 state가 자라고, 기존 recurrent 계열은 (1)–(3)의 부분집합만 갖는다. Atlas는 다섯을 전부 체크하는 유일한 parallelizable recurrent model이라는 것이 논문의 자기 위치 규정이다.

## 14.3 Core mechanism (통일 표기)

이 절은 세 층으로 전개된다: capacity 이론(왜 feature map인가) → Omega rule(왜 window인가) → Muon(왜 2차 근사인가). 모든 수식은 통일 표기이며, 표준형 (M1)–(M4)와의 차이로 서술한다. Atlas의 통일형은 식 (M3)다.

### 14.3.1 Capacity 이론: 고정 크기 memory는 몇 쌍을 저장하는가

[Atlas §3.1]은 이 라인 최초로 **memory capacity**를 형식적으로 정의한다: memory가 **정확히**(inner loss 0으로) 매핑할 수 있는, 선형독립 key를 가진 $(k_i,v_i)$ 쌍의 최대 개수 $m$. 5장의 고전 capacity(Hopfield의 확률적 저장 한계)와 달리, 이것은 정확 보간(exact interpolation) 기준의 결정론적 정의다.

**Proposition 1** [Atlas Prop. 1]: matrix memory $W\in\mathbb{R}^{d_v\times d_k}$가 $\ell(W;k_t,v_t)=\|Wk_t-v_t\|_2^2$를 gradient descent로 최적화하면, 저장 가능한 쌍은 최대 $O(d_k)$개다. 증명의 뼈대는 순수한 rank 논증이다 [Atlas App. C]: 정확 저장은 $WK=V$ ($K=[k_1\cdots k_m]$)를 요구하고, vectorize하면 $(K^\top\otimes I_{d_v})\,\mathrm{vec}(W)=\mathrm{vec}(V)$ — 미지수 $d_kd_v$개에 방정식 $md_v$개 — 이므로 $m\le d_k$. 이 한계는 tight하다: $m\le d_k$이고 $K$가 full column rank이면 Moore–Penrose pseudoinverse로 정확 보간해를 지을 수 있고, full-batch GD는 (loss $\|Wk-v\|_2^2$의 계수 2를 반영한) step size $0<\eta<1/\lambda_{\max}(KK^\top)$에서, 그리고 $W_0=0$ 초기화 아래에서 minimum-norm 보간해로 수렴한다(GD의 implicit bias; $\eta<2/\lambda_{\max}$는 $\tfrac12$-loss 관행의 값이다). 같은 rank 제약이 multi-head attention의 "low-rank bottleneck"이기도 하다는 지적이 붙는다.

이 명제의 함의를 독자의 언어로 옮기면: $d_k\times d_v$개의 파라미터를 가진 memory가 $d_k$개의 연상밖에 저장하지 못한다 — $d_k,d_v$가 함께 커진다고 가정할 때 capacity는 파라미터 수 $P=d_kd_v$에 대해 **sub-linear**다(예: $d_v=\Theta(d_k)$이면 $O(\sqrt{P})$). $d_v$를 고정하면 $O(d_k)=O(P/d_v)$로 $P$에 선형임에 주의. state 크기를 늘리는 것(파라미터를 더 쓰는 것)과 capacity를 늘리는 것은 같은 일이 아니다.

**Theorem 1** [Atlas Thm. 1]: $L_{\mathcal{M}}\ge 2$층 MLP memory(입력 차원 $d_k$, hidden 차원 $d_h^{(j)}$)는 최소 $O(d_kd_v)$, 최대 $O\big(d_kd_v\sum_{i=1}^{L_{\mathcal{M}}}\min_{j\ge i}d_h^{(j)}\,d_h^{(i+1)}\big)$쌍을 저장한다. 증명은 ReLU MLP의 piecewise-affine 구조를 쓴다: 고정된 activation pattern 위에서 MLP는 하나의 affine 사상 $A(\cdot)+B$이고, $m\le\mathrm{rank}(A)$는 합성 경로의 최소 폭들로 위에서 눌린다 [Atlas App. C]. 논문은 두 방향으로 결론짓는다: depth는 표현력만이 아니라 **capacity 자체**를 올리며(Titans의 deep memory ablation이 경험적으로 보였던 것의 이론적 대응), 그러나 상한은 여전히 $(d_k,d_v)$에 대해 subquadratic이라 deep memory만으로는 초선형 capacity에 도달할 수 없다고.

> **[평가]** Theorem 1을 검증된 정리로 읽어서는 안 된다. Prop 1의 capacity 정의는 "$\mathbb{R}^{d_k}$의 선형독립 key"를 요구하므로 입력 key에 대해서는 어떤 memory든 $m\le d_k$인데, Theorem 1의 하한 $O(d_kd_v)$는 $d_v>1$이면 이를 초과한다 — 즉 Theorem 1은 입력 key가 아니라 MLP가 내부에서 lift한 표현의 선형독립성을 세는, 원문이 명시하지 않은 다른 capacity 개념을 암묵적으로 쓴다. 또한 App. C 증명의 $m\le\mathrm{rank}(A)$는 value의 선형독립을 가정하지 않아 하한을 깔끔히 세우지 못한다(AK=V에서 $\mathrm{rank}(V)\le\mathrm{rank}(A)$는 나와도 $m\le\mathrm{rank}(A)$는 따라오지 않는다). 따라서 "depth가 capacity 자체를 올린다"는 검증된 정리가 아니라 원문의 주장이며, 이 책은 그 방향성만(Titans의 deep-memory ablation과 정합하는 선에서) 인용한다.

남은 손잡이가 key의 차원이다. key·value 차원을 직접 키우면 projection 파라미터가 차원당 $O(d)$씩 늘고 긴 문맥에서 메모리 사용량이 커진다. 대신 [Atlas §3.1]은 separable kernel $\sigma(x,y)=\phi(x)^\top\phi(y)$를 key와 query에 적용한다. **polynomial feature map** $\phi_p(x)=[x^\beta]_{|\beta|\le p}$ — 차수 $p$ 이하의 모든 monomial을 쌓은 벡터 — 를 쓰면 lifted 차원은 $D=\binom{d_k+p}{p}=\Theta(d_k^p)$가 된다.

**Proposition 2** [Atlas Prop. 2]: lifted key 위의 matrix memory가 $\ell(W;\phi_p(k_t),v_t)=\|W\phi_p(k_t)-v_t\|_2^2$를 최적화하면 capacity는 최대 $O(d_k^p)$. 그리고 증명 [Atlas App. C]은 더 강한 사실을 준다: $\mathrm{rank}(W\Phi)\le\mathrm{rank}(\Phi)\le D$이므로, **어떤 최적화 방법을 쓰든** $D$쌍을 넘길 수 없다. 즉 feature map은 optimizer와 독립인 capacity 상한 그 자체를 옮기는 손잡이고, optimizer(다음의 Muon)는 그 상한 안에서 실제 도달 품질을 올리는 손잡이다 — 두 축이 직교한다는 것이 이 논문 설계의 논리적 골격이다. 그리고 이 정리 사슬은 5장 §5.4의 dense-Hopfield 사슬(Krotov→Ramsauer: energy 차수 = feature lift = capacity)의 정식화다 — 그 사슬을 통과한 독자에게 Prop 2는 corollary처럼 읽히며, 이미 가진 $\phi_p$·$\phi^*$ 직관을 그대로 재사용하면 된다.

polynomial map에는 두 가지 추가 해석이 붙는다 [Atlas §3.1]. 첫째, **Taylor 근사로서의 softmax**: $\exp(q^\top k)\approx a_0+a_1\,q^\top k+a_2\,(q^\top k)^2+\cdots+a_p\,(q^\top k)^p$ [Atlas Eq. 5]. 계수 $a_i$를 $1/i!$로 초기화하되 **학습 가능**하게 두면, polynomial kernel은 "절단된, 학습 가능한 softmax kernel"이 된다. 둘째, **input feature gating**: $a_i\to 0$은 차수 $i$의 feature block 전체를 잘라내고, $a_1\to 1$에 나머지 0이면 $\phi(x)=x$로 돌아간다 — RNN의 gate를 memory가 아니라 **입력 표현**에 적용한 것이다.

극한이 이 절의 결론이다. Kronecker self-tensoring으로

$$
\phi^*(x)=\Big(1,\;x,\;\tfrac{x^{\otimes 2}}{\sqrt{2!}},\;\tfrac{x^{\otimes 3}}{\sqrt{3!}},\;\dots\Big)^{\!\top}
$$

를 정의하면 $\exp(q^\top k)=\phi^*(q)^\top\phi^*(k)$가 **정확히** 성립한다 [Atlas Eq. 22–23]. 따라서 softmax attention은 무한 차원 feature 공간 위의 associative memory이고, **capacity가 unbounded다** [Atlas §4.2] (→ 5장 §5.4: exponential 극한 = softmax attention). 1장의 Rosetta 사전에서 "KV cache = 압축하지 않는 memory"라고 썼던 대응이 여기서 정리의 형태를 얻는다: attention이 긴 문맥 recall에서 고정 state 모델을 이기는 이유는 신비가 아니라 capacity 상한의 차이다.

한 가지 표기 주의: 원문은 $\phi_p$를 §3.1에서는 "차수 $\le p$의 모든 monomial"로, [Atlas Eq. 22]에서는 self-tensoring $x^{\otimes p}$로 두 번 다르게 정의한다. 두 정의는 lifted 차원의 스케일($\Theta(d_k^p)$)에서는 같은 급이며, 이 책은 §3.1의 정의를 기본으로 쓴다.

### 14.3.2 Omega rule: token이 아니라 context를 memorize한다

기존 online 모델들의 inner 문제는 retention gate(→ 13장)를 붙인 per-token 최적화다 [Atlas Eq. 6]:

$$
\min_W\;\ell(W;k_t,v_t)+\mathrm{Ret}_t(W,W_{t-1}).
$$

반대쪽 극단은 전체 문맥에 대한 global 최적화 $\min_W\sum_{i=1}^{t}\ell(W;k_i,v_i)$다 [Atlas Eq. 7]. global 형태의 문제는 두 가지다 [Atlas §3.2]: (1) **효율** — 매 step 최적화 제약이 늘고, 일반적인 비선형 memory나 충분통계가 없는 objective에서는 test time에 모든 과거 key·value를 cache해야 하며(고정 state의 존재 이유 소멸; 단 linear least-squares는 Sherman–Morrison RLS로 충분통계+inverse state만 유지하면 되는 예외 — 그 실제 결격 사유는 순차 inverse update·추가 $O(d_k^2)$ state·linear memory 한정이다, 아래 Mesa-layer), 정확해를 원하면 병렬화 불가능한 solver가 필요하다. (2) **context pruning 불가** — 문맥이 중간에 바뀌거나 무관한 구간이 끼면 "전부에 대한 최적"이 오히려 해가 되는데, global objective에는 특정 token을 잘라낼 직접적 gate가 없다.

Atlas의 중간해가 **Omega rule**이다. window 길이 $c\ge 1$에 대해 inner objective를 마지막 $c$개 token의 gated 합으로 세운다 [Atlas Eq. 8–9]:

$$
\min_W\;\sum_{i=t-c+1}^{t}\gamma_{t,i}\,\big\|\mathcal{M}(k_i;W)-v_i\big\|_2^2
\tag{14-1}
$$

<!-- FIG: ch14/fig-01-omega-window -->

여기서 $\gamma_{t,i}\in[0,1]$이 **window gate**다: step $t$의 window 안에서 $i$번째 token이 최적화에 참여하는 정도를 정하는 input-dependent gate로, $\gamma_{t,i}\to 0$이면 그 token을 최적화에서 **직접(hard) 잘라내고**, $\gamma_{t,i}\to 1$이면 온전히 포함한다 — 논문의 표현으로 **in-context pruning**이다 [Atlas §3.2]. 이 gate가 감당 가능한 이유가 바로 sliding window 구조다: step당 필요한 gate 수가 $c$개로 **상수**다. global 최적화(Eq. 7)에 input-dependent gate를 달려면 prefix 길이만큼의 gate 값이 필요해 — 공유 gate-producer의 파라미터는 고정이지만 gate 값의 수와 이를 계산·저장하는 비용이 문맥 길이에 따라 자라 — recurrent model의 장점이 사라진다.

Omega rule은 계보 전체를 극한으로 회수한다 [Atlas §3.2].

- $c=1$: online delta rule(→ 5장, 6장), 즉 (M1)이다. 여기에 momentum을 더하면 **Titans의 LMM update가 정확히 나온다** [Atlas Eq. 12–13] — 통일 표기로 $S_t=\beta_tS_{t-1}-\eta_t\nabla_W\ell(W_{t-1};k_t,v_t)$, $W_t=\alpha_tW_{t-1}+S_t$, 곧 (M2)다. Titans는 Omega의 window-1 특수 사례다.
- $c=$ 문맥 전체, linear memory, $\gamma\equiv1$: (regularized) least-squares $\min_W\sum_{i=1}^{t}\|Wk_i-v_i\|_2^2$가 되고 [Atlas Eq. 14], 이를 Sherman–Morrison 재귀로 정확히 푸는 것이 Mesa-layer다(→ 이 책 §1.6 카탈로그의 대조군). 정확하지만 병렬화 불가, linear memory 한정, 그리고 hard gate가 없어 pruning 불가 — Atlas가 자신을 차별화하는 세 가지 결격 사유다.

memory의 은유로 요약하면 [Atlas §3.2]: window loss의 gradient는 개별 token의 surprise가 아니라 **surprise of the context** — 마지막 $c$개 token의 context-aware 결합 — 이다. 12장의 momentary/past surprise 2분법이 여기서 세 번째 항을 얻는 셈이다.

> **[해설]** inference 어휘로 옮기면 이렇다. per-token delta rule이 "cache의 마지막 항목 하나에 대해서만 정합성을 유지하는 write"라면, Omega rule은 "최근 $c$개 항목에 대한 batch write-repair"다. 그리고 $\gamma_{t,i}$는 학습된 admission control이다: retention gate $\alpha_t$가 이미 저장된 것 중 무엇을 남길지(eviction)를 정한다면, $\gamma_{t,i}$는 애초에 무엇을 저장 대상에 넣을지(admission)를 정한다. 두 gate는 같은 축이 아니다.

### 14.3.3 OmegaNet: rank-c 전이로의 일반화

Omega rule + polynomial feature + GD + weight decay가 **OmegaNet**이다 [Atlas Eq. 10]:

$$
W_t \;=\; \alpha_t W_{t-1} \;-\; \nabla_W\sum_{i=t-c+1}^{t}\gamma_{t,i}\,\big\|\mathcal{M}(\phi(k_i);W_{t-1})-v_i\big\|_2^2
\tag{14-2}
$$

식 (14-2)는 정확히 표준형 (M3)다. 기호를 전부 확정하면: $W_t$는 fast weights(기본형은 2-layer residual MLP의 weights, §14.3.6), $\alpha_t\in[0,1]$은 retention gate(남기는 비율; Atlas 원문도 같은 방향), gradient는 $W_{t-1}$에서 평가되며, per-token step size는 $\gamma_{t,i}$에 흡수된다(관행은 (M3) 아래 §1.4 참조). linear memory($\mathcal{M}(z;W)=Wz$, $W\in\mathbb{R}^{d_v\times D}$)로 특수화하면 closed form이 나온다 [Atlas Eq. 11] — 이 책의 열벡터 관행으로:

$$
W_t=W_{t-1}\Big(\alpha_t I-\sum_{i=t-c+1}^{t}\gamma_{t,i}\,\phi(k_i)\phi(k_i)^\top\Big)+\sum_{i=t-c+1}^{t}\gamma_{t,i}\,v_i\,\phi(k_i)^\top .
\tag{14-3}
$$

(원문은 행벡터 관행이라 전이 항을 좌곱 $(\mathrm{diag}(\alpha_t)-\sum\gamma\,\phi\phi^\top)M_{t-1}$로 쓰며, $\mathrm{diag}(\alpha_t)$ 표기는 retention gate가 채널별 벡터일 수 있음을 시사한다 — 스칼라 gate이면 두 표기는 동치다.)

식 (14-3)을 6장의 카탈로그 옆에 놓으면 이 rule의 정체가 선명해진다. DeltaNet의 전이는 $W_{t-1}(I-\eta_tk_tk_t^\top)$ — rank-1 수정이다. Omega rule의 전이는 $c$개 key의 gated 합 $\sum_i\gamma_{t,i}\phi(k_i)\phi(k_i)^\top$ — **rank-$c$ 수정**이다. (gated) delta rule의 rank-1 구조를 window 크기만큼의 rank로 일반화한 것이 Omega rule의 대수적 내용이다.

원문 대조 주의가 하나 있다. [Atlas Eq. 11]은 value 항 $\sum\gamma\,v_i\phi(k_i)^\top$ 앞에 음(−) 부호를 인쇄하는데, [Atlas Eq. 10]의 $\ell_2$ loss를 실제로 전개하면 value 항의 부호는 양(+)이다(delta rule 전개와 동일; → 5장). 이 책은 전개가 맞는 (14-3)의 부호로 쓰고, 원문의 부호는 [Atlas App. D.4]와 Table 1에도 걸쳐 있는 표기 관행 흔들림의 일부로 처리한다(§14.3.4 말미).

> **[해설]** 식 (14-3)은 GEMM으로 읽힌다. 전이 항은 $[\phi(K)]_{c\times D}$ 꼴 행렬의 gated Gram matrix이고, write 항은 $(d_v\times c)\times(c\times D)$ GEMM — rank-$c$ write다. 1장 Rosetta의 "훈련 수식을 GEMM shape로 읽는다"의 전형으로, window를 키우는 것은 write GEMM의 내적 차원 $c$를 키우는 것이다.

### 14.3.4 Atlas: inner loop의 Muon — 근사 2차 memory 관리

Omega rule과 feature map까지 갖춰도 optimizer가 GD면 1차다. windowed loss의 표면에서 나쁜 local minimum에 앉으면 낮은 품질의 key→value 매핑이 저장된다는 것이 세 번째 진단이었다 [Atlas §5]. Atlas는 같은 objective (14-1)을 **Muon**(→ 2장; Jordan et al. 2024)으로 최적화한다. 통일 표기로:

$$
S_t \;=\; \beta_t\,S_{t-1}\;-\;\nabla_W\sum_{i=t-c+1}^{t}\gamma_{t,i}\,\big\|\mathcal{M}(\phi(k_i);W_{t-1})-v_i\big\|_2^2,
\qquad
W_t \;=\; \alpha_t\,W_{t-1}\;+\;\eta_t\,\mathrm{NS}_\kappa(S_t)
\tag{14-4}
$$

[Atlas Eq. 32–33]. 기호: $S_t$는 momentum buffer로 memory의 각 weight 행렬과 같은 shape의 텐서(12장의 past surprise와 같은 객체이되, 이제 **windowed** gradient를 누적한다), $\beta_t$는 momentum decay gate(원문 기호 $\theta_t$), $\eta_t$는 learning-rate gate, $\mathrm{NS}_\kappa(\cdot)$는 Newton–Schulz 행렬 반복 $\kappa$회다. $\mathrm{NS}_\kappa$는 $\kappa\to\infty$에서 $S$의 SVD $S=U\Sigma V^\top$에 대한 **nearest semi-orthogonal matrix** $UV^\top$로 수렴한다 — raw momentum 대신 그 orthogonalization을 update로 쓰면 update의 singular value가 균등화되고, 논문은 이를 "token에 대한 2차 정보의 근사"로 읽는다 [Atlas §5]. 이것이 Table 1의 **locally optimal memory** 성질이다: attention/SWA는 이 성질을 non-parametric하게(regression을 그냥 풀어서) 갖고, GD 기반 RNN 전부는 갖지 못하며, Atlas는 병렬화를 유지한 채 parametric하게 달성한 최초의 사례라는 것이 논문의 주장이다 [Atlas Table 1, §1].

$\kappa$는 이 라인에서 처음 등장하는 종류의 손잡이다. [Atlas §5]는 $\kappa$를 명시적으로 "internal **test-time compute** parameter"로 규정한다: 반복을 더 돌리면 semi-orthogonalization이 정확해지고 잠재적으로 memorization이 좋아지되, inference FLOPs가 선형으로 는다. state 크기를 전혀 바꾸지 않고 decode 연산량과 품질을 교환하는 dial이다(§14.7에서 serving 관점으로 재론). 실전값은 $\kappa=5$("NS-5")다.

재구현자를 위한 정직한 경고를 여기 두어야 한다. Atlas의 recurrence는 원문 안에서 세 곳의 표기가 서로 다르다. [Atlas Table 1]의 행은 $S_t=\theta_tS_{t-1}-\nabla\ell$, $M_t=\alpha_tM_{t-1}-\eta_t\,\mathrm{NS}\text{-}5(S_t)$로 쓰고(심지어 Table 1의 Titans 행은 $\eta$와 $\theta$의 역할을 [Atlas Eq. 12–13]과 맞바꿔 인쇄한다), [Atlas Eq. 33]은 momentum에 gradient를 **더하며**($+\nabla$), [Atlas App. D.4 Eq. 57–58]은 per-token 계수 $\eta_i^{(t)}$를 합 안에 넣고 $M_t=\alpha_tM_{t-1}+\mathrm{NS}\text{-}5(S_t)$로 $\eta_t$를 밖에서 제거한다. 이 중 **Table 1의 행은 단순 표기 관행이 아니라 부호 오탈자**다: $S_t$에 $-\nabla\ell$(descent 방향)를 쌓아 놓고 $M_t$에서 그 $\mathrm{NS}(S_t)$를 다시 **빼면** 두 음부호가 겹쳐 loss ascent가 된다. 서로 동등한 descent 관행은 [Atlas Eq. 33]($+\nabla/-\mathrm{NS}$)과 [Atlas App. D.4]($-\nabla/+\mathrm{NS}$) 둘뿐이며, 알고리즘의 의미(windowed gradient의 decayed momentum을 쌓고 orthogonalize하고 retention을 곱한 뒤 step)는 이 둘에서만 동일하다. 이 책은 App. D.4 관행, 즉 (M2)/(M3)의 부호로 식 (14-4)를 고정한다.

### 14.3.5 Transformer 일반화 가족: DLA/SWLA → DeepTransformers → Dot

Atlas의 두 번째 기둥은 같은 기계로 Transformer를 다시 유도하는 절이다. 출발점은 softmax attention의 재서술이다: attention은 Nadaraya–Watson kernel regression

$$
\mathcal{M}^\star(q)=\arg\min_{z}\sum_{i=1}^{L}s(k_i,q)\,\|v_i-z\|_2^2=\sum_{i=1}^{L}\frac{s(k_i,q)}{\sum_{j}s(k_j,q)}\,v_i
$$

의 **non-parametric 해**다 [Atlas Eq. 17] ($z$는 최적화 변수, $s(\cdot,\cdot)$는 exp kernel 유사도, $\mathcal{M}^\star$는 argmin 최적해 — §1.2의 예약 의미 그대로다). 합을 마지막 $c$개 token으로 제한하면 그대로 sliding window attention(SWA)이 나온다 [Atlas Eq. 18]. 이 대응이 주는 통찰이 이 절의 논지다: attention은 자기 attentional bias를 **global하게, 비모수적으로** 최적화하고, 현대 recurrent model은 **parametric online learner**다. Omega rule은 attention의 window 수준 최적화를 parametric 세계로 수입한 것이고, SWA는 Atlas가 parametric하게 푸는 것과 같은 windowed regression의 비모수 해다 — 같은 문제, 다른 해 공간.

이 대응 위에 논문은 controlled baseline 두 개와 본 모델 두 개를 세운다.

**DLA(Deep Linear Attention)** [Atlas Eq. 19]: dot-product attentional bias $\ell=-\langle\mathcal{M}(\phi(k_t);W),v_t\rangle$(최소화) + deep MLP memory + GD/decay. 부호 주의 — 원문 Eq. 19는 양의 내적 $\langle\cdot\rangle$을 쓰고도 양의 Hebbian write를 적어 부호 모순을 보인다: $+vk^\top$ write가 나오려면 loss가 음의 내적이거나 gradient ascent여야 한다. $\phi=$ identity로 특수화하면 gated linear attention $W_t=\alpha_tW_{t-1}+v_tk_t^\top$로 붕괴하므로(→ 6장), DLA는 Hebbian/linear-attention 설정에서 **deep memory의 기여만** 분리해 재는 baseline이다. **SWLA** [Atlas Eq. 20]는 같은 bias를 window 합으로 바꾼 windowed gated Hebbian rule — linear memory closed form은 $W_t=\alpha_tW_{t-1}+\sum_{i=t-c+1}^{t}\gamma_{t,i}v_i\phi(k_i)^\top$ — 로, **windowed objective의 기여만** 분리한다.

**DeepTransformers**: DLA의 $\phi$를 정확한 exponential map $\phi^*$로 바꾼 것이다 [Atlas Eq. 25]. linear memory이면 $W_t=\sum_{i\le t}v_i\phi^*(k_i)^\top$이고 읽기가

$$
y_t=W_t\,\phi^*(q_t)=\sum_{i\le t}v_i\,\exp(q_t^\top k_i)
$$

— 정확히 **unnormalized softmax attention**이다 [Atlas Eq. 26]. 따라서 DeepTransformers(deep memory + $\phi^*$)는 **unnormalized** exponential attention의 strict generalization이고, unnormalized Transformer는 그 특수 사례(linear memory + Hebbian rule + exp kernel)다 — 정규화된 표준 Transformer까지 포함하려면 softmax 분모를 계산하는 별도 state·정규화 연산이 필요하다(아래 한계). sliding-window 판이 SWDT다.

**Dot(Deep Omega Transformer)**: Hebbian rule 자리에 Omega rule을 넣는다 [Atlas Eq. 27]:

$$
W_t=W_{t-1}-\nabla_W\sum_{i=t-c+1}^{t}\gamma_{t,i}\,\big\|\mathcal{M}(\phi^*(k_i);W)-v_i\big\|_2^2 .
$$

linear memory closed form에서 $c=1$로 두면 [Atlas Eq. 30–31]이 나오는데, 논문은 이를 "**Transformers with Delta rule**"이라 부른다: unbounded memory가 새 $(k,v)$를 attention처럼 append하는 **동시에**, 그 key에 대해 이전 state가 예측하던 value를 빼서 교정한다 — error-correcting attention이다. 단, DeepTransformers와 Dot 모두 softmax의 분모(normalizer)가 빠진 **unnormalized** 형태로만 분석·정의된다 [Atlas Table 1 각주]. 정규화가 훈련된 모델에서 어떻게 복원되는지는 논문에 명시가 없다 — §14.8의 한계 목록에 다시 올린다.

이 가족이 개념적으로 사주는 것은 Table 1의 다섯 번째 성질 **flexible context**의 계보다: window + 학습된 $\gamma$ gate로 "무엇을 기억할지 고르는" 능력은 SWA, SWDT, OmegaNet, Dot, Atlas가 공유하며, 이 성질을 기준으로 보면 Atlas 계열은 "SWA의 parametric 쌍둥이"다.

### 14.3.6 Backbone, 표준 memory, Atlas++

architecture backbone은 현대 recurrent LM 관행을 따른다 [Atlas §5.1]: 층마다 Q/K/V linear projection + 크기 4의 short convolution, 그리고 학습 안정화를 위한 key·query normalization. memory module의 기본형은 이 라인의 표준 deep memory다:

$$
\mathcal{M}(z;W)=z+W_1\,\sigma(W_2 z)
$$

— 2층, expansion 4, GELU, residual [Atlas Eq. 42, App. E]. App. E는 chunk 끝마다 layer norm을 더한다. **Atlas++**는 memory를 gated MLP로 올린 변형이다 [Atlas Eq. 43]:

$$
\mathcal{M}(z;W)=z+W_1\big(\sigma(W_2z)\odot W_3z\big),
$$

$W_1,W_2,W_3$ 전부가 inner loop에서 갱신되는 fast weights다. 합성은 Titans의 문법을 그대로 쓴다(→ 12장): MAG(SWA 브랜치와 gate 결합), MAL(memory block 다음 SWA block), 그리고 BABILong 실험에서는 MAC을 persistent memory tokens 없이 쓴다 [Atlas §6.3].

설계 선택과 담당 결함의 대응을 한 줄씩 정리하면 — window loss(14-1)는 online 결함(1)을, $\phi_p/\phi^*$와 deep/gated memory는 capacity 결함(2)을, $\mathrm{NS}_\kappa(S_t)$는 관리 결함(3)을 맡고, $\alpha_t$(retention)와 $\beta_t$(momentum)는 Titans에서 상속된 상태 유지 장치다.

### 14.3.7 표기 대응표

표 14-1 — [Atlas] 원 표기 ↔ 통일 표기 대응 (§1.7.3 기준; 아래쪽 6행은 이 장의 장-국소 추가)

| Atlas 원 표기 | 통일 표기 | 의미 / 주의 |
|---|---|---|
| $M_t$ — memory 상태 | $W_t$ | |
| $\eta_t$ — learning-rate gate | $\eta_t$ | 동일 |
| $\theta_t$ — momentum decay | $\beta_t$ | ⚠ Titans와 정반대 배치였음 — 통일로 해소 |
| $\alpha_t$ — weight-decay(forget) gate, $\alpha_tM_{t-1}$ | $\alpha_t$ | 동일 방향 |
| $S_t$ — momentum (windowed gradient 누적) | $S_t$ | 동일 |
| $\gamma_i^{(t)}$ — window gates | $\gamma_{t,i}$ | 동일 (첨자 표기만 정리) |
| $c$ — sliding window 길이 | $c$ | 동일. chunk $C$와 구분 |
| $\phi_p$, $\phi^*$ | 동일 | |
| NewtonSchulz-$k$ / NS5 | $\mathrm{NS}_\kappa$, $\kappa=5$ | 반복 횟수 기호 $k$ 금지 |
| Eq. 32–33 vs App. D.4의 부호·step-size 차이 | (M3)의 부호로 고정 | "부호 관행 차이(알고리즘 동일)"로 처리 |
| capacity의 $m$ (저장 쌍 수), $D=\binom{d_k+p}{p}$ | 동일 | 장-국소 |
| DLA/SWLA/OmegaNet/DeepTransformers/Dot/SWDT | 고유명사 유지 | §1.6 카탈로그 형태로 소개 |
| $b$ — 병렬화 chunk 크기 [Atlas §3.3] | $C$ | Titans의 $b$와 같은 knob (§1.5: $b$ 금지) |
| $t'=t-\mathrm{mod}(t,b)$ — chunk 시작 | $\xi(t,C)$ | (M4)의 gradient anchor |
| $u_t=\nabla\ell(M_{t'};k_t,v_t)$ — 사전 계산 gradient [Atlas §5.1] | $\tilde g_t$ (장-국소) | ⚠ $u_t$는 NL의 Local Surprise Signal로 예약 |
| $M_s$ — sliding-window mask [Atlas §3.3] | $M_{\mathrm{s}}$ (장-국소) | banded 0/1 mask |
| $a_i$ — 학습 Taylor 계수 [Atlas Eq. 5] | 동일 | init $1/i!$, outer-loop 학습 대상 |
| $\Theta,E$ — gate 대각 행렬 [Atlas Eq. 39] | 산문으로 서술 | broadcast scan의 구현 세부 |

## 14.4 Outer-loop training vs inner-loop test-time learning

Atlas에서 "이 gate는 누가 학습하는가?"라는 질문의 답은 예외 없이 하나다: **전부 outer loop가 학습한다**. inner loop가 학습하는 것은 memory의 내용물뿐이다. 논문 자신이 [Miras]의 Definition 1을 그대로 이어받아 두 loop를 정의한다 [Atlas Def. 1]: inner loop는 $\theta_{\mathcal{M}}=\{W_1,W_2,\dots\}$ — memory module의 파라미터 — 만을 최적화하고, 그 동안 모델의 다른 모든 파라미터는 고정된 hyperparameter다; outer loop는 그 나머지 전부를 최적화한다.

표 14-2 — Atlas의 두 loop: 무엇이 어디서 움직이는가

| 대상 | loop | 갱신 rule / 시점 | 비고 |
|---|---|---|---|
| $W_Q,W_K,W_V$ (projection), 크기-4 conv, backbone MLP·norm | outer ($\Theta$) | AdamW, pre-training 중 | inner loss의 "hyperparameter" |
| gate-producer: $\alpha_t,\eta_t,\beta_t$와 $c$개의 $\gamma_{t,i}$를 산출하는 outer-학습 메커니즘 (입력·구조 — 특히 $\gamma_{t,i}$가 $x_t$만의 함수인지 window token·pairwise feature에도 의존하는지 — 는 원문 미공개) | outer ($\Theta$) | AdamW | gate **값**은 매 token 바뀌지만 gate를 **만드는 함수**는 frozen |
| Taylor 계수 $a_i$ (feature map gating) | outer ($\Theta$) | AdamW, init $1/i!$ | [Atlas Eq. 5] |
| memory 초기 상태 $W_{\mathrm{init}}$ (memory MLP의 초기화) | outer 상태 | pre-training이 결정 | 매 context 시작 시 $W_0=W_{\mathrm{init}}$로 재시작 |
| memory weights $W_t$ ($W_1,W_2$; Atlas++는 $W_3$까지) | **inner** | 식 (14-4), 매 token/chunk | context가 끝나면 폐기 |
| momentum buffer $S_t$ | **inner** | 식 (14-4), 매 token/chunk | weight 행렬당 하나, 같은 shape |
| window buffer: 최근 $c-1$개의 $\phi(k),v$와 gate | **inner** (rolling) | ring-buffer append | 크기 $O(c(D+d_v))$, $D=\binom{d_k+p}{p}$ (lifted $\phi(k)$ 차원), 길이 무관 |

training 무경험 독자가 가장 헷갈리는 지점을 짚는다. "memory가 test time에 훈련된다"는 문장은 **pre-trained checkpoint가 변한다는 뜻이 아니다**. outer loop는 pre-training에서 단 한 번, "inner loop가 어떻게 갱신해야 하는가"를 — gate-producer, projection, feature 계수, 초기 상태의 형태로 — 학습한다. inference에서는 그 배운 update 절차가 매 context마다 $W_{\mathrm{init}}$에서 다시 실행될 뿐이다. 이것이 [Atlas §1]의 test-time memorization 명명이 가리키는 정확한 사실이고, 12장의 bilevel 구조(→ 4장)가 Atlas에서도 그대로 유지된다는 뜻이다.

**outer gradient는 inner loop를 관통한다.** update 식 (14-4)는 그 자체로 미분 가능한 계산 그래프다 — gradient 평가($\nabla_W\ell$), gate 곱, momentum 누적, 그리고 Newton–Schulz 행렬 다항식까지 전부. outer loss $\mathcal{L}$(next-token prediction)의 backward는 이 unrolled recurrence 전체를 거꾸로 타고 내려가므로, 예컨대 $W_K$의 outer gradient에는 $\nabla_W\|\mathcal{M}(\phi(k_i);W)-v_i\|^2$를 다시 $k_i$로 미분하는 2차 미분 성격의 항이 들어간다(TTT·Titans와 같은 기제; → 8장, 12장). Atlas 특유의 추가분은 backward가 $\mathrm{NS}_\kappa$의 행렬 다항식까지 통과해야 한다는 점이다. 따라서 아래의 chunkwise 형태는 단순한 추론 최적화가 아니라 **실제로 미분되는 그래프의 정의**다.

**chunkwise 병렬화 1 — window는 banded mask 하나 값이다** [Atlas §3.3]. 소박한 구현은 위치마다 $c$개의 gradient 행렬 $\nabla_W\ell\in\mathbb{R}^{d\times d}$를 실체화해야 해서 메모리·IO가 폭발한다. 대신 시퀀스를 크기 $C$(원문 $b$)의 chunk로 나누고, chunk 안 모든 gradient를 chunk 시작 상태 $W_{\xi(t,C)}$에서 평가한다 — 9장의 stale-snapshot 근사 그대로다. 설명 편의상 $\gamma_{t,i}=\eta_t$로 두면, chunk 내부의 $t$ ($\xi(t,C)<t\le\xi(t,C)+C$)에 대해

$$
W_t=\Big(\prod_{\tau=\xi+1}^{t}\alpha_\tau\Big)W_{\xi}
\;-\;\underbrace{\sum_{n=\xi+1}^{t}\Big(\prod_{\tau=n+1}^{t}\alpha_\tau\Big)\eta_n\!\!\sum_{i=n-c+1}^{n}\!\!\nabla_W\,\ell\big(W_{\xi};k_i,v_i\big)}_{G_t},
\qquad \xi:=\xi(t,C)
\tag{14-5}
$$

[Atlas Eq. 16]. $G_t$의 계산은 Titans의 병렬 gradient 계산과 동일하되, einsum broadcast 단계에 **sliding-window mask** $M_{\mathrm{s}}$를 하나 더 곱한다: $c=1$이면 $M_{\mathrm{s}}$는 항등 행렬이고, $c>1$이면 각 대각 성분 바로 앞 $c-1$개 위치를 추가로 1로 둔 banded 하삼각 0/1 행렬이다. 각 token의 gradient는 한 번만 계산되고, 자신이 속한 $c$개 window로의 기여는 mask가 합산한다 — per-position gradient 텐서를 실체화하지 않고. 논문은 이 때문에 window가 online($c=1$) 버전 대비 유의미한 계산 overhead를 더하지 않는다고 주장한다 [Atlas §3.3]. FlashAttention tiling과의 차이는 9장의 명제 그대로다: tiling은 bit-exact지만, 여기의 $C$는 gradient anchor를 바꾸므로 **계산되는 함수 자체를 바꾸는 semantic hyperparameter**다(식 (M4); → 9장). window $c$는 objective의 파라미터, chunk $C$는 병렬화의 파라미터 — 둘은 독립 knob이며 반드시 구분해야 한다.

**chunkwise 병렬화 2 — momentum이 memory에서 분리된다** [Atlas §5.1]. Muon까지 병렬화하는 열쇠는 구조적 관찰 하나다: gradient를 chunk 경계 상태에서 평가하기로 한 순간, $\tilde g_t:=\nabla_W\ell(W_{\xi(t,C)};k_t,v_t)$는 chunk 전체에 대해 **사전 계산 가능**하고, 그러면 momentum recurrence $S_t=\beta_tS_{t-1}-\eta_t\tilde g_t$는 진화하는 memory $W$와 완전히 독립인 linear scan이 된다. unroll하면

$$
S_t=\Big(\prod_{j\le t}\beta_j\Big)S_0-\sum_{i\le t}\Big(\prod_{j=i+1}^{t}\beta_j\Big)\,\eta_i\,\tilde g_i
\tag{14-6}
$$

[Atlas Eq. 39] — gate들의 대각 행렬과 쌓아 둔 gradient의 broadcast-and-matmul로 chunk 내 모든 $S_t$를 한 번에 얻는다(1장 Rosetta의 scan/prefix-sum 대응 그대로). 그다음 $\mathrm{NS}_5$는 per-matrix 다항식 — 반복마다 $X\leftarrow aX+b(XX^\top)X+c'(XX^\top)^2X$ 꼴의 서너 개 matmul(계수는 → 2장) — 이므로 chunk 내 전 위치의 $S_t$에 **batched matmul로 동시에** 적용된다 [Atlas Eq. 40]. 마지막 남는 순차 부분은 $W_t=\alpha_tW_{t-1}+\eta_t\,\mathrm{NS}_5$-항의 gated 누적 scan 하나다 [Atlas Eq. 41]. 이 세 단계 — gradient 병렬, momentum scan, batched NS — 가 "근사 2차 inner optimizer를 가진 최초의 parallelizable recurrent architecture"라는 주장 [Atlas §1]의 실체다.

**outer 훈련 recipe** [Atlas App. E]: FineWeb, T5 tokenizer(32K vocab), 훈련 문맥 4K(SWA 성분은 2K), outer optimizer AdamW(learning rate 4e-4, cosine schedule, batch 0.5M tokens, weight decay 0.1), 스케일별 peak LR은 표 14-4 참조.

## 14.5 Concept ledger delta

표 14-3 — [Atlas]가 ledger에 더하고 바꾼 것

| 구분 | 개념 | 내용 |
|---|---|---|
| 신규 | **Omega rule** (이 장 소유) | windowed inner objective (14-1); delta rule($c=1$)의 strict generalization, Mesa-layer($c=L$)로의 보간 |
| 신규 | **window gate $\gamma_{t,i}$** (이 장 소유) | in-context pruning; step당 상수($c$)개라서 학습 가능한 hard gate가 성립 |
| 신규 | **memory capacity (formal)** (이 장 소유) | 선형독립 key의 정확 저장 쌍 수; matrix $O(d_k)$ → deep MLP subquadratic → $\phi_p$로 $O(d_k^p)$ → $\phi^*$는 unbounded |
| 신규 | $\phi_p$ + 학습 Taylor 계수 $a_i$, $\phi^*$ | capacity 손잡이로서의 feature map; softmax attention = unbounded-capacity associative memory의 정식화 |
| 신규 | **locally optimal memory** | 근사 2차 정보에 의한 memory 관리 (Muon inner); Table 1의 새 열 |
| 신규 | **flexible context** | 무엇을 기억할지 고르는 능력 (window + $\gamma$); Table 1의 새 열 |
| 신규 | $\mathrm{NS}_\kappa$ 반복수 = test-time compute dial | state 불변으로 decode FLOPs↔품질 교환하는 최초의 layer 내부 손잡이 |
| 신규 | sliding-window mask $M_{\mathrm{s}}$ | window 합을 banded 0/1 mask 하나로 처리하는 chunkwise 구현 장치 |
| 신규(모델) | OmegaNet, Atlas, Atlas++, DLA, SWLA, DeepTransformers, SWDT, Dot | §1.6 카탈로그에 등재된 여덟 이름 |
| 확장 | surprise (→ 12장) | momentary/past에 이어 세 번째 형태: **surprise of the context** = windowed loss의 gradient |
| 확장 | momentum-as-memory (→ 12장, 2장) | raw momentum → $\mathrm{NS}_\kappa$-orthogonalized momentum; optimizer-as-architecture 축의 두 번째 사례 |
| 확장 | attentional bias (→ 13장) | per-token family에 window 축이 추가됨; Miras Table의 두 열 확장판이 [Atlas Table 1] |
| 확장 | retention (→ 13장) | $\alpha_t$(저장분의 eviction)와 별개로 $\gamma_{t,i}$(저장 전 admission)라는 직교 gate가 생김 |
| 개명 | test-time training → **test-time memorization** (이 장 소유) | inner loop는 저장·인출일 뿐 학습이 아니라는 용어 교정; "진짜 continual learning은 따로 필요하다"는 후속작 — [NL] (*Nested Learning: The Illusion of Deep Learning Architecture*, arXiv:2512.24695)과 [Sleep] (*Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories*, arXiv:2606.03979) — 의 복선 |

## 14.6 실험과 스케일

**스케일과 구성** [Atlas Table 7, App. E]:

표 14-4 — 훈련 구성 ([Atlas Table 7])

| 모델 | blocks | dim | heads | peak LR | tokens |
|---|---|---|---|---|---|
| 170M | 12 | 768 | 16 | 3e-3 | 15B |
| 340M | 24 | 1024 | 16 | 1.5e-3 | 15B |
| 760M | 24 | 1536 | 16 | 1.25e-3 | 30B |
| 1.3B | 18 | 2048 | 8 | 7e-4 | 100B |

훈련 문맥은 4K, 데이터는 FineWeb이다. 원문 §6의 Setup 문단은 스케일을 "340M, 400M, 790M, 1.3B"로 적는데 [Atlas §6], App. E의 Table 7과 결과 표들의 열 제목(760M/1.3B)은 위 표와 같다 — 원문 내부 불일치이며, 이 책은 Table 7을 따른다. baseline 수치의 대부분은 재실행이 아니라 Titans·Miras·Gated DeltaNet 논문들에서 상속되었다는 점도 원문이 명시한다 [Atlas §6, App. E].

**Language modeling + commonsense reasoning** [Atlas Table 2]. 1.3B / 100B tokens에서: Atlas는 Wikitext ppl 14.97 / LAMBADA ppl 10.98 / 평균 acc 57.62, Atlas++는 14.40 / 10.72 / 58.03, OmegaNet은 14.91 / 11.26 / 57.23. 비교군은 Titans (LMM) 15.60 / 11.41 / 56.82, Gated DeltaNet 16.42 / 12.17 / 55.32, Samba(hybrid) 16.13 / 13.29 / 54.00, Transformer++ 18.53 / 18.32 / 52.25. Transformer 일반화 가족도 자기 비교군을 이긴다: DeepTransformers 평균 56.19, Dot 57.35 vs Transformer++ 52.25. 같은 순서가 760M에서도 유지된다: OmegaNet 52.56 / Atlas 52.77 / Atlas++ 53.09 vs Titans 51.56, Transformer++ 48.69; hybrid는 Atlas(MAG)가 Wikitext ppl 18.62, 평균 53.08로 MAL(19.07, 52.63)보다 낫다 [Atlas Table 2].

**S-NIAH (RULER)** [Atlas Table 3]. 4K로 훈련된 모델을 2K–16K needle-in-haystack에서 평가한다. 순수 recurrent 비교에서 Atlas는 S-NIAH-N 16K에서 84.0으로 Titans 80.2를 앞서고, DeltaNet(5.4)·TTT(4.4)와는 자릿수가 다르다. hybrid와 Transformer-like 가족은 더 강하다: Dot은 전 설정에서 93.2–100(S-NIAH-W 16K의 93.2가 최솟값), Atlas(MAG)는 S-NIAH-PK 16K에서 98.6 — 훈련 문맥의 4× 외삽이다.

**BABILong** [Atlas §6.3, Fig. 4]. MAC backbone(persistent memory tokens 없이)으로 benchmark protocol에 따라 fine-tune한 설정이다. Atlas는 1M token까지 Titans와 동급이다가, 10M에서 Titans가 무너지는 지점에서 **+80% accuracy를 유지한다** — 이 논문의 헤드라인 long-context 주장이다. 논문은 이를 Muon(관리), polynomial kernel(capacity), context memorization(objective)의 합작으로 귀속시킨다 [Atlas §6.3].

**MAD synthetic suite** [Atlas Table 4]. 평균 Atlas 79.50 / OmegaNet 78.98 vs Titans 76.44, Transformers 75.46, Gated DeltaNet 71.04. 최대 격차는 memorization(91.4)과 fuzzy recall 축이다.

**In-context recall — 의무 caveat** [Atlas Table 5]. SWDE/NQ/DROP/FDA/SQuAD/TQA 평균에서 **Transformer가 여전히 이긴다**: 53.55 vs Atlas 43.70, OmegaNet 43.13, Titans 42.31, Gated DeltaNet 40.28. 특히 FDA에서 72.5 vs 40.7로 격차가 크다. Atlas는 recurrent 중 최고이고 gap을 좁혔지만 닫지 못했다 — 이것은 §14.3.1의 capacity 이론이 예측하는 방향 그대로다: $\phi^*$의 unbounded capacity를 가진 비모수 memory(attention)와 유한 state의 parametric memory 사이의 격차는, Atlas의 모든 장치를 넣고도 남는다.

**Ablation — 의무 caveat** [Atlas Table 6, 760M]. 완전한 Atlas: language modeling ppl 19.97 / reasoning acc 52.77. 성분 제거의 효과는: linear memory로 강등 21.03 / 49.74, $c=1$(window 제거) 21.98 / 49.26, polynomial mapping 제거 22.14 / 50.57, **Muon 제거 19.65 / 52.56**, gated-MLP memory 추가 19.53 / 53.09, +Attn(MAG) 19.90 / 53.08, +Attn(MAL) 20.26 / 52.63. 숫자를 정직하게 읽어야 한다: window($c=1$ 대비 −2.01 ppl)와 feature map(−2.17 ppl)과 deep memory(−1.06 ppl)는 명확히 기여하지만, **Muon을 제거하면 perplexity는 오히려 개선된다**(19.65 < 19.97). Muon이 산 것은 reasoning accuracy 0.21pt뿐이다.

> **[평가]** 이 ablation은 Atlas의 세 축 중 optimizer 축의 가치가 760M 스케일에서 미결임을 뜻한다. 논문 제목이 약속하는 "optimally memorize"의 optimality는 window와 capacity 축이 대부분 벌어다 준 것이고, "locally optimal"을 담당하는 Muon의 기여는 과제 의존적이며 perplexity 기준으로는 음수다. $\kappa$를 test-time compute dial이라 부르려면 $\kappa$에 대한 품질 곡선이 단조임을 보여야 하는데, 그 측정은 논문에 없다.

**window 크기와 scaling** [Atlas Fig. 5, Fig. 8]. 고정된 global context에서 window $c$를 키우면 성능이 단조 개선된다 — $\gamma$ gate가 필요할 때 잘라내 주기 때문이라는 것이 논문의 설명이다 [Atlas §6.6]. 파라미터 수와 훈련 문맥 길이에 대한 scaling도 baseline보다 유리하다 [Atlas Fig. 8]. MQAR에서는 memory 크기당 정확도 기준으로 DeltaNet류 대비 최고를 보고한다 [Atlas §6.5, Fig. 7].

**Learnability micro-study** [Atlas §6.4]. $d=256$의 온라인 regression으로 memory 후보(작은 MLP + Adam/RMSprop/SGD)의 학습 능력 자체를 잰 부속 실험이다. 다섯 종류의 목표 사상 중 low-rank·MLP·attention-출력 사상은 잘 배우지만, 과거 입력의 기억을 요구하는 두 설정(attention+MLP, SWA+MLP)에서 최악이고, 특히 **SWA 목표가 full-attention 목표보다 더 어렵다**. 저자들의 가설: online 학습기는 오래된 입력을 '잊지' 못해서, 잊기가 필수인 sliding-window 사상에서 더 크게 실패한다 — gated windowed objective의 필요성을 뒷받침하는 정황 증거다. 단, 이것은 $\gamma$ gate가 그 문제를 실제로 푼다는 분리 실험이 아니라 동기 부여 실험이다.

**스케일의 정직한 상한.** 이 논문의 모든 품질 주장은 1.3B params / 100B tokens에서 끝난다. 라인 전체의 실증 상한이기도 하다(→ 12장, 15–17장 공통). 그리고 훈련이든 decode든 **wall-clock 수치는 한 건도 없다** — "significant overhead가 없다", "병렬화된다"는 전부 구조 논증이지 측정이 아니다.

## 14.7 Systems/serving 함의

**state 크기의 회계.** Atlas의 per-layer state는 세 덩어리다: (1) memory weights $W$ — 표준형이면 $W_1\in\mathbb{R}^{d_m\times 4d_m}$, $W_2\in\mathbb{R}^{4d_m\times d_m}$로 $8d_m^2$ 원소($d_m$ = memory 폭), (2) momentum $S$ — 같은 shape로 또 $8d_m^2$, (3) window buffer — $O(c(d_k+d_v))$로 무시 가능. 합계 $\approx 16d_m^2$ 원소이며 문맥 길이와 무관하다. Transformer KV cache는 layer당 $2L_{\mathrm{ctx}}d$ 원소로 문맥에 비례한다.

> **[해설]** 같은 정밀도를 가정하고 등호를 놓으면 crossover 문맥 길이는 $L^*=16d_m^2/(2d)$다. memory 폭을 model 폭과 같게 잡으면($d_m=d$) $L^*=8d$ — 1.3B 구성($d=2048$)에서 약 16K tokens다. 즉 16K보다 짧은 문맥에서는 KV cache가 더 작은 상태이고, BABILong의 10M-token 구간에서는 Atlas의 state가 KV cache의 수백분의 일이다. 단 이 계산에는 두 개의 미지수가 있다. 첫째, 실제 구현의 memory가 head별로 쪼개지는지, $d_m$이 얼마인지 논문에 없다. 둘째, $\phi_p$는 memory 첫 층의 입력 폭을 $\Theta(d_k^p)$로 불리므로(차수 2만 해도 sketch 전 $\sim d_k^2$), **구현된 차수 $p$와 sketch 차원이 공개되지 않은 한 state 크기는 계산할 수 없다** — 구현 $p$·sketch 차원·memory head 분할은 원문 본문에도 [Atlas App. C], [Atlas App. E], [Atlas Fig. 3]의 라벨에도 없다. capacity의 이득은 cache 성장이 아니라 state 크기와 matmul 폭으로 지불된다 — 그 청구서의 액수가 논문에 없다. 5장 §5.4가 예고한 "비용 계산"은 그래서 여기서 **구조** — capacity의 지불 통화가 state byte와 matmul 폭이라는 것 — 로 확정된다; 절대값 계산은 구현 차수 미공개로 여기서도 불가하다는 것까지가 이 장의 결론이다.

**decode의 state 트래픽.** 매 token, 각 layer의 memory는 read-modify-write다: $W$와 $S$를 읽고, gradient·momentum·NS를 계산하고, 둘 다 다시 쓴다. 트래픽은 token당 $\approx 2\times 16d_m^2$ 원소의 읽기+쓰기로, DeltaNet류의 $d\times d$ matrix state 대비 (expansion 4의 2층 + momentum 때문에) 원소 수로 16배 급이다 — 단 이는 표준 memory·명시적 $\phi_p$ lift 없을 때의 값이고, lift가 있으면 첫 층 입력이 $D=\binom{d_k+p}{p}$로 커져 더 크다(§14.7 [해설]의 미공개 차원 caveat). 이 RMW 스트림이 decode의 실질 대역폭 예산을 정하며, KV cache처럼 append-only가 아니므로 캐시 계층에 상주시키는 전략이 달라진다 — 상세한 배치 논의는 10장과 Part III의 몫이다.

**연산의 성격은 전부 dense matmul이다.** windowed gradient는 banded einsum($M_{\mathrm{s}}$; 식 (14-5)), momentum은 broadcast scan(14-6), $\mathrm{NS}_5$는 행렬 다항식 — 논문이 명시적으로 "tensorize computations and maximize matmuls"를 설계 목표로 선언한다 [Atlas §3.3]. Titans 대비 추가 상수는 $\mathrm{NS}_5$다: update당·weight 행렬당 반복 5회 × 서너 개의 정방 matmul ≈ 15–20개의 추가 matmul. chunk 안에서 전 위치에 batch되므로 GEMM shape는 좋다. window 자체는 mask 하나라서 훈련 시 거의 공짜다 — **$c$는 훈련 비용을 거의 바꾸지 않는 품질 knob**이라는 것이 Atlas의 병렬화가 만든 특이한 경제학이다.

**병렬화 구조.** intra-chunk는 (gradient, momentum, NS) 3단 전부 병렬이고 inter-chunk만 gated scan이다. momentum recurrence가 memory state에서 분리된다는 §14.4의 관찰이 구조적 핵심으로, sequence-parallel 훈련이 자연스럽게 얹힌다. 단 chunk 사슬 자체는 여전히 순차다 — 이 사슬을 끊는 것은 [TNT] (*TNT: Improving Chunkwise Training for Test-Time Memorization*, arXiv:2511.07343)의 reset이 처음이다(→ 15장).

**serving 관점의 새 dial: $\kappa$.** NS 반복수는 state 크기·checkpoint 형식을 전혀 건드리지 않고 decode FLOPs와 (주장되는) memorization 품질을 교환한다. speculative decoding이 요청 단위의 품질/지연 트레이드라면 $\kappa$는 layer 내부의 트레이드다 — 다만 §14.6의 [평가]대로 품질 곡선의 단조성이 미측정이므로, 현재로서는 "존재하는 knob"이지 "검증된 knob"이 아니다.

**batching과 hybrid.** per-request로 변하는 fast weights는 shared-weight batching을 깨뜨린다 — 요청마다 다른 $W_t,S_t$로 같은 layer를 실행해야 하므로 decode는 grouped-GEMM 형태가 된다(→ 1장 Rosetta, 10장). MAG/MAL hybrid는 2K SWA 브랜치를 달고 있어 serving이 rolling KV cache와 recurrent state를 **동시에** 관리해야 한다. 반대로 DeepTransformers/Dot은 정확한 $\phi^*$ 아래에서 유한한 recurrent state가 없으므로 attention과 같은 비용 구조다 — 이들은 serving 효율이 아니라 Transformer 대비 품질로 경쟁하는 갈래다.

**요약하면**: kernel 표면의 신규 워크로드는 (i) chunk 전체에 걸친 batched NS-5 fusion, (ii) banded gradient mask, (iii) outer 훈련의 backprop-through-NS 세 가지이며, 어느 것도 공개 구현이 없다. 그리고 다시 — throughput/latency 수치가 논문에 전무하므로, serving 시점의 quality-per-FLOP 이야기는 전부 미정량이다.

## 14.8 한계와 bridge-out

논문 안팎의 한계를 정리한다.

1. **retrieval gap 미해소** [Atlas Table 5]: 53.55 vs 43.70. capacity 이론은 격차의 방향을 설명하지만, 남은 격차가 capacity 한계인지, inner 최적화의 한계인지, lossy 고정 state의 본질인지는 이 논문으로 판별되지 않는다.
2. **Muon의 기여가 equivocal** [Atlas Table 6]: perplexity는 제거 시 개선. $\kappa$-품질 곡선 미측정. optimizer 축의 실증은 후속(NL의 레벨 관점; → 16장)으로 넘어간다.
3. **feature map의 구현 미공개**: 실제 차수 $p$, sketch, 결과 차원이 없어 state·FLOPs 회계가 불가능하다(§14.7). $\phi^*$와 deep memory의 결합도 linear closed form 밖에서는 정의되지 않는다 — 무한 차원 feature는 실체화할 수 없으므로, "w/o Polynomial Mapping" ablation의 존재는 실제 Atlas가 $\phi_p$를 쓴다는 정황이다.
4. **unnormalized 일반화**: DeepTransformers/Dot은 softmax 분모를 버린 채 분석된다. softmax의 안정성·품질 중 분모의 몫이 얼마인지는 열린 문제다.
5. **capacity 이론의 이상화**: 정리들은 "선형독립 key의 정확 보간" 기준이고, Theorem 1은 단일 activation region 논증이다. 실전 retrieval 품질의 근사적 대리 지표이지 그 자체가 아니다.
6. **표기 삼중 불일치**(§14.3.4): 재구현자는 Table 1 / Eq. 32–33 / App. D.4 중 하나의 관행을 골라야 한다. 의미는 하나다.
7. **chunkwise staleness 무정량**: gradient anchor $W_{\xi(t,C)}$ 근사의 오차 한계가 없고(라인 공통; → 9장), chunk $C$와 window $c$의 상호작용도 미탐구다.
8. **window 스케줄**: Fig. 5는 "클수록 좋다"까지만 말한다. $c$의 비용 최적, 적응적 선택, 학습된 스케줄은 없다.

**Bridge-out — [TNT]로.** Atlas까지 와서 이 라인은 objective(Omega), capacity(feature map), optimizer(Muon), 이론(capacity 정리)을 다 갖췄다. 그런데 여섯 편을 통틀어 이 시점까지 **wall-clock 수치가 0건**이다. 모든 것이 chunkwise 트릭 — stale snapshot에서의 gradient — 위에 서 있는데 그 근사는 무정량이고, chunk 크기 $C$는 throughput knob으로만 취급될 뿐 **계산되는 함수를 바꾸는 semantic knob**이라는 사실(→ 9장)은 아무도 논점으로 삼지 않았다. 특히 Atlas는 "훈련한 $C$와 다른 $C$로 serve하면 무슨 일이 일어나는가"를 묻지 않는다 — decode는 $C=1$의 세계인데 훈련은 큰 $C$에서 이뤄지는 구조적 mismatch가 방치되어 있다. [TNT]는 정확히 여기서 시작한다: 훈련·추론 chunk-size mismatch를 **발견**하고(550M Titans가 $C=64$ 훈련 후 $C=64$ 평가에서 ppl 13.78, $C=8$ 평가에서 36.45 [TNT Fig. 2]), 계층적 memory와 학습된 $W_{\mathrm{init}}$로의 주기적 reset, 그리고 two-stage 훈련으로 chunk 경제학 전체를 재설계한다(→ 15장). Atlas가 "무엇을 얼마나 잘 기억하는가"의 정점이라면, TNT는 "그걸 만들고 돌리는 데 얼마가 드는가"라는, 이 책의 독자가 처음부터 묻고 있던 질문의 장이다.

그리고 하나 더, 조용한 복선이 있다. Atlas가 "test-time **memorization**"이라는 개명을 고집한 순간 — in-context 적응은 학습이 아니라고 선을 그은 순간 — "그러면 진짜 continual learning은 어디서 오는가"라는 질문이 라인의 장부에 미결로 올라갔다. 그 답이 [NL]의 nested level들과 [Sleep]의 wake/sleep lifecycle이다(→ 16장, 17장).
