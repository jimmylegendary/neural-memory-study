# ch18. Nested Learning과 Memory Caching — 연속체는 다리인가

## 18.1 Bridge-in: 전작이 남긴 문제

ch17은 오프라인 재귀를 $W$에 접었다. KV cache를 비우기 전에 같은 문맥 위로 (U-W)를 $N$회 더 돌리는 것이 곧 sleep이라는 구성이었고, 그 결과 $W$-경로에는 "질의 전에 실행되는 pass"라는 자리가 처음으로 생겼다. 그 자리가 생기자마자 따라오는 질문이 하나 있다. **$W$에 접는 것과 $\Theta$에 쓰는 것은 정말 다른 종류의 행위인가, 아니면 같은 축 위의 다른 눈금인가.**

Nested Learning(*Nested Learning: The Illusion of Deep Learning Architecture*, arXiv:2512.24695, 이하 [NL])은 후자를 주장한다. 갱신 주파수라는 축 하나를 세우고 그 위에 attention부터 pre-training까지를 순서지은 뒤, 학습과 추론의 구분 자체가 "지식 전달 과정을 최고 주파수 level에서 최저 주파수 level(즉 pre-training)로부터 끊어낸 결과"라고 쓴다 [NL §6]. 같은 절에 "Neural Learning Module에는 training time도 test time도 없다"는 박스가 붙는다 [NL §6]. 이 주장이 성립하면 이 책의 3층 프레임은 축 하나로 접힌다.

**이 장은 그 주장을 검증한다.** 검증의 결론을 먼저 적는다. **연속체는 표기의 다리이지 기제의 다리가 아니다.** [NL]이 구현한 모든 갱신은 (U-W) 모양이고, $\Theta$는 그 축 위에서 test-time에 아무것도 움직이지 않는 $f=0$ 끝점으로만 등장한다. 축이 두 끝점을 이름으로 잇는 것과 상태가 두 끝점 사이를 실제로 이동하는 것은 다른 사건이며, [NL]에는 후자가 없다.

같은 장에 Memory Caching(*Memory Caching: RNNs with Growing Memory*, arXiv:2602.24281, 이하 [MC])을 함께 놓는다. 두 논문은 같은 저자 그룹에서 나왔고, [MC]는 [NL]과 Miras의 "memory 상태를 forward pass 중 최적화되는 동적 객체로 보는" 관점을 명시적으로 승계한다고 §2에서 밝힌다 [MC §2]. 그러나 [MC]가 실제로 한 일은 갱신식이 아니라 **읽기식**을 바꾼 것이고, $B_s=0$이다. 세그먼트 경계에서 일어나는 사건은 갱신이 아니라 체크포인트 저장이다. 그래서 이 장은 [MC]를 sleep-time compute 사례로 읽지 않고 **$W$층 용량 천장 논증**으로 읽는다 — 고정 크기 $W$ 하나로 recall이 무너진다면 $W$를 몇 개나 들고 있어야 하는가에 대한 정량적 답이다.

ch11이 이 장에 넘긴 것은 두 편의 탈락 사유였다: [NL]은 조건 (1)과 (4)에서, [MC]는 조건 (2)와 (3)에서 걸린다(→ ch11 §11.4.2, §11.4.3). 이 장은 그 판정을 반복하지 않고 **증거로 보인다** — 어느 객체가 어느 시계로 움직이는지의 표로.

## 18.2 문제의식

두 논문이 스스로 정의한 문제는 다르다. 순서대로 옮긴다.

[NL]의 출발점은 Transformer가 두 개의 극단 주파수만 쓴다는 관찰이다.

> "Transformer architectures are based on two extreme frequencies of update: i.e., infinity and 0 for attention and MLP blocks, respectively" [NL §1]

이 문장을 이 책의 기호로 옮기면 $f=\infty$는 attention의 KV 읽기이고 $f=0$은 $\Theta$다. [NL]의 모든 기여는 그 사이에 놓인다. 두 번째 출발점은 Transformer에서 사영 $W_k, W_v, W_q$가 최저 주파수 level에서 최적화되므로 문맥 안에서 얼어 있고, 그래서 토큰이 문맥화되는 방식에 상한이 걸린다는 관찰이다 [NL §8, §8.1]. 해법은 그 사영들 자체를 갱신되는 memory로 승격시키는 것이다.

여기에 신경과학 유비가 붙는다. §1은 gamma(30–150 Hz), beta(13–30 Hz), delta/theta(0.5–8 Hz) 대역을 나열하고, 빠른 온라인 consolidation과 느린 오프라인 consolidation의 2단계 그림을 세운 뒤 곧바로 범위를 좁힌다.

> "in this work, we focus on the first stage: memory consolidation as an online process" [NL §1]

**논문이 오프라인 단계를 이름 붙여 부르고 명시적으로 사양한다.** 이 문장이 이 장의 판정에서 가장 무거운 증거이며, 동시에 corpus 전체에서 가장 쓸모 있는 bridge-out이다(→ §18.8).

[MC]의 출발점은 recall이다. recurrent 모델이 recall 집약 과제에서 뒤처지는 원인을 "자라는 시퀀스를 압축해야 하는 고정 용량이 과거 정보를 잊게 만들며, 이것이 결정적 병목"이라고 진단하고 Zoology 계열의 recall–throughput tradeoff를 인용한다 [MC §1]. 해법은 놀랄 만큼 단순하다 — 과거 memory 상태를 버리지 않는다. 시퀀스를 $N$개 세그먼트로 자르고 각 세그먼트 끝의 iterate를 보존한 뒤, 읽기를 그 캐시 집합 위의 게이트된 혼합으로 바꾼다.

> **[해설]** 두 문제의식은 같은 자리를 반대편에서 누른다. [NL]은 "$W$의 갱신 **주기**가 하나뿐이라서 문제다"라고 하고, [MC]는 "$W$의 **개수**가 하나뿐이라서 문제다"라고 한다. 전자는 시간축을 쪼개고 후자는 상태축을 쪼갠다. 그리고 어느 쪽도 문맥 경계를 넘지 않는다 — 쪼갠 것이 시간이든 개수든, 시퀀스가 끝나면 전부 사라진다. 이 공통점이 두 논문을 한 장에 묶는 이유이고, 동시에 둘 다 sleep-time compute이 아닌 이유다.

이 장이 소유하는 구분 하나를 여기서 못 박는다.

> **정의.** 어떤 논문이 **갱신식을 바꾼다**는 것은 (U-W)·(U-$\Theta$)·(U-E) 중 한 규칙의 우변을 바꾸는 것이다. **읽기식을 바꾼다**는 것은 식 (R)의 세 인자 $\Theta, W, \mathrm{ret}(E,q)$ 중 하나를 **읽는 방식**을 바꾸는 것이며, 상태를 만드는 규칙은 그대로 둔다. 두 변경은 비용 회계에서 서로 다른 항으로 들어간다 — 갱신식 변경은 $B_s$ 또는 쓰기 시점의 지연에 들어가 식 (A)로 상각될 수 있고, **읽기식 변경은 질의마다 다시 지불되므로 상각되지 않는다.**

## 18.3 Core mechanism (통일 표기)

### 18.3.1 Nested Learning — 두 갱신 규칙

[NL]의 축은 Definition 2가 정의한다. 한 데이터 포인트에 대한 갱신 한 걸음을 단위 시간으로 삼아, 임의의 성분 $A$에 대해 단위 시간당 갱신 횟수를 $f_A$로 둔다 [NL Def. 2, §3.2]. 성분들은 $f$로 순서지어 **level**이 되고, 높은 level일수록 낮은 주파수다. 이 장에서 $f$는 장-국소 기호이며 단위는 언제나 토큰이다 — level $\ell$의 chunk 크기를 $C^{(\ell)}$이라 하면 $f_\ell = 1/C^{(\ell)}$이다.

구현된 갱신은 둘이다. 첫째가 Continuum Memory System(CMS)이다.

$$
W^{(\ell)}_{i} \;=\; W^{(\ell)}_{i-1} \;-\; \eta^{(\ell)} \!\!\sum_{t=i-C^{(\ell)}}^{i}\!\! \nabla_{W}\, \mathcal{L}\big(W^{(\ell)}_{t};\, x_t\big)
\quad \text{if } i \equiv 0 \ (\mathrm{mod}\ C^{(\ell)}),
\qquad \text{else } W^{(\ell)}_{i}=W^{(\ell)}_{i-1}
\tag{18-1}
$$

식 (18-1)은 [NL Eq. 71]의 통일 표기 환원이다. 표준형 (U-W)와의 차이는 셋이다. retention gate가 없고($\alpha_t \equiv 1$), chunk 안의 합 자체가 momentum 누적기 역할을 하며 경계에서 초기화되고, **objective가 inner loss $\ell$이 아니라 outer loss $\mathcal{L}$이다.** 세 번째가 결정적이다. 논문은 이렇게 쓴다.

> "the objective of $\mathcal{L}(\cdot)$ is the objective of choice for the task at hand, e.g., for language modeling it is next token prediction objective" [NL §7.1]

즉 (18-1)은 **(U-$\Theta$)의 목적함수를 (U-W)의 시계와 소멸성 위에서 계산한다.** 이 한 줄이 논문 전체의 모호성이다. 게다가 CMS가 움직이는 대상은 보조 상태가 아니라 모델 자신의 MLP 블록이다 — vanilla Transformer 블록이 $k=1$, 주파수 0인 퇴화 사례로 나오고 [NL §7.1], retrofit 경로에서는 문자 그대로 pre-trained Llama-3의 MLP 텐서에서 시작한다 [NL §7.3]. 겉모습만 보면 $\Theta$-경로다.

둘째가 self-modifying Titans다.

$$
W^{\square}_t \;=\; W^{\square}_{t-1}\big(\alpha_t I - \eta_t k_t k_t^{\top}\big) \;+\; S^{\square}_t,
\qquad
S^{\square}_t \;=\; -\,\eta_t\, \nabla_W\, \ell\big(W^{\square}_{t-1};\, k_t,\, \hat v_{\square,t}\big)
\tag{18-2}
$$

$\square \in \{k, v, q, \eta, \alpha, \mathrm{memory}\}$이고 각각이 2층 residual MLP다 [NL Eq. 88–89]. (U-W)와의 차이 셋: retention gate가 스칼라가 아니라 데이터 의존 행렬 $\alpha_t I - \eta_t k_t k_t^{\top}$이고, $\eta_t$와 $\alpha_t$가 스케줄이 아니라 **자기 자신도 $W$층 memory의 출력**이며, 회귀 목표 $\hat v_{\square,t} = \mathcal{M}_{\square}(v_t; W^{\square}_{t-1})$가 갱신 대상 memory 자신에게서 생성되므로 외생 목표가 없다.

두 규칙 모두 병렬화를 위해 stale snapshot으로 실행된다. chunk $\lceil t/C \rceil$의 $k_t, v_t, \hat v_{\square,t}, \eta_t, \alpha_t$와 모든 inner gradient가 **직전 chunk 끝 상태**에서 생성된다 [NL Eq. 90, §8.2]. Hope는 (18-2)의 작은 상태·표현력 높은 규칙과 (18-1)의 큰 용량·단순한 규칙을 결합한 블록이고, Hope-Attention은 CMS의 기여만 분리하려고 softmax attention에 CMS를 붙인 변형이다 [NL §7.1, §8.3].

### 18.3.2 Memory Caching — 읽기식만 바뀐다

[MC]는 시퀀스를 $N$개 세그먼트 $S^{(1)},\dots,S^{(N)}$으로 자른다. 각 세그먼트 안에서 갱신은 baseline에서 **한 글자도 바뀌지 않는다**.

$$
W^{(s)}_t \;=\; f\big(W^{(s)}_{t-1};\ k_t, v_t\big), \qquad 1 \le t \le L^{(s)}
\tag{18-3}
$$

$f$는 baseline 아키텍처의 갱신 규칙 그대로다 — linear attention이면 $W_{t-1} + v_t k_t^{\top}$, Titans이면 (U-W)의 $\alpha_t, \beta_t, \eta_t$ 세 항이 그대로 [MC Eq. 4, 34–35]. 세그먼트 끝에서 일어나는 일은 갱신이 아니라 보존이다: $W^{(i)} := W^{(i)}_{L^{(i)}}$.

바뀌는 것은 식 (R)이다. 일반형은 캐시 집합 위의 집계다 [MC Eq. 5]. 논문이 제시하는 네 구체형 중 둘을 적는다.

$$
\hat y_t \;=\; \gamma^{(s)}_t\, \mathcal{M}\big(q_t; W^{(s)}_t\big) \;+\; \sum_{i=1}^{s-1} \gamma^{(i)}_t\, \mathcal{M}\big(q_t; W^{(i)}\big),
\qquad
\gamma^{(i)}_t = \mathrm{softmax}_i \big\langle u_t,\ \mathrm{MeanPool}(S^{(i)}) \big\rangle
\tag{18-4}
$$

$$
r^{(i)}_t = \big\langle u_t,\ \mathrm{MeanPool}(S^{(i)}) \big\rangle,
\quad
R_t = \arg\mathrm{Top}\text{-}k\big(\{r^{(i)}_t\}_{i=1}^{s-1}\big),
\quad
\hat y_t = \gamma^{(s)}_t \mathcal{M}\big(q_t; W^{(s)}_t\big) + \sum_{i \in R_t} \gamma^{(i)}_t \mathcal{M}\big(q_t; W^{(i)}\big)
\tag{18-5}
$$

(18-4)가 Gated Retrieval Memory(GRM) [MC Eq. 9–10], (18-5)가 Sparse State Caching(SSC) [MC Eq. 16–17]이다. 나머지 둘은 게이트를 1로 둔 Residual [MC Eq. 7]과 파라미터 공간에서 섞어 토큰마다 자기 회수용 $W^*_t = \sum_i \gamma^{(i)}_t W^{(i)}$를 짓는 Memory Soup [MC Eq. 14–15]이다. $u_t = x_t W_u$는 $q_t$와 별개의 사영이다.

두 극단이 이 설계의 좌표를 준다. $N=1$이면 보통의 RNN이고, $N=L$이면 모든 과거 토큰의 상태가 캐시되어 attention과 거의 일치한다 [MC §3.1]. 세그먼트 크기 1에 value 없는 벡터 memory를 두면 게이트된 global softmax attention 블록이 문자 그대로 복원된다 [MC Eq. 18–20]. **즉 이 논문은 RNN과 Transformer 사이의 보간을 읽기 쪽에서 구성한 것이다.**

이 장이 소유하는 두 번째 정의가 여기서 나온다.

> **정의.** **$W$층 용량 성장**이란, 한 시퀀스가 진행되는 동안 그 시퀀스가 보유하는 fast-weight 상태의 총량이 늘어나는 것을 말한다. 자라는 것은 $W$ 하나의 크기가 아니라 **보존된 (U-W) iterate의 개수 $N$**이고, 유효 용량은 $N$에 선형이며 $1 \le N \le L$이다 [MC §3.1]. **$W$층 용량 천장**은 그 $N$의 상계를 정하는 것이 아키텍처가 아니라 **세그먼트 분할 스케줄**이라는 사실을 말한다 — 균일 분할이면 $N = L/C$로 문맥 길이에 선형이어서 상계가 없고, 로그 분할이면 $N \le \log_2 L$로 상계가 생기지만 먼 과거 토큰의 해상도가 현저히 떨어져 recall 집약 과제의 능력이 제한된다 [MC §4.2].

로그 분할의 구성은 이진 표현을 그대로 쓴다. $L=37=(100101)_2$이면 세그먼트 길이가 32, 4, 1로 잡히고, 그래서 총 비용이 $O(pL\log L)$로 내려간다 [MC §4.2]. 균일 분할이면 $O(pL^2/C)$다. **상계를 얻는 대가가 해상도이며, 이 교환이 $W$층 용량 논의의 핵심이다.**

한 가지 함정이 있다. $W$가 행렬이고 게이트가 없으면 residual 변형은 사전 합산되어 고정 크기로 붕괴한다: $\hat y_t = (W^{(s)}_t + \sum_{i<s} W^{(i)}) q_t$ [MC Eq. 13]. 성장이 실재하려면 게이트가 문맥 의존이어야 하고, 그 결과 게이트는 토큰마다 다시 계산되어야 하며 사전 계산도 다음 토큰 재사용도 불가능하다 [MC §3.1]. **용량 성장의 대가가 정확히 이 지점에서 발생한다.**

### 18.3.3 표기 대응표

표 18-1 — [NL] 원 표기 → 이 책의 표기

| 원 표기 | 이 책의 표기 | 주의 |
|---|---|---|
| $\Theta_i^{(k)}$ / $\theta^{(f_\ell)}$ — 임의 level의 파라미터 | pre-training level 위의 모든 level은 $W$. $\Theta$는 level 1에만 | 하드 충돌. 논문의 $\theta$를 그대로 $\Theta$로 옮기면 논문이 조용히 $\Theta$-경로로 재분류된다. §1 예약 규칙상 금지 |
| $\mathcal{L}(\cdot)$ — Eq. 71에서 CMS 시계로 평가되는 task objective | $\mathcal{L}$ (outer/task loss) | 기호도 뜻도 같다. 단 **outer loss를 inner 시계로 평가한다**는 사실을 본문에 적어야 한다 |
| $\tilde L$ — inner memory objective | $\ell$ | 소문자 $\ell$이 inner loss 예약 기호다 |
| $\mathcal{M}$, $\mathcal{M}_\square$ — 가변 객체 | 상태는 $W$, 읽기는 $\mathcal{M}(\cdot; W)$ | 논문은 $\mathcal{M}$으로 객체 자체를 가리킨다. Eq. 88 인용 시 분리 |
| $C^{(\ell)}$ — level별 chunk 크기(토큰) | $C$ (chunk) | 용량 $C_{\text{cap}}$ 자리로 흘러들어가지 않게 한다 |
| $f_A$ — 갱신 주파수 | $f$ (장-국소) | 비용 기호 아님. 단위는 항상 토큰, $f_\ell = 1/C^{(\ell)}$ |
| $\hat v_{\square,t}$ — 자기 생성 회귀 목표 | $\hat v_{\square,t}$ 유지. **$\hat c$가 아니다** | $\hat c$는 $E$-경로의 learned context 예약 기호다. 혼동하면 없는 $E$-경로 독법이 생긴다 |
| $\eta^{(\ell)}$, $\eta_t$ | 전부 $\eta_t$ 계열(inner) | $\eta_\Theta$는 §9.1의 15B 토큰 continual pre-training에만 해당 |
| $S_t$ — 누적 momentum 기여 | $S_t$ | 첨자를 유지해 $S(c;B_s)$와 분리 |
| "level" | "level" 그대로 | 이 책의 "층"으로 번역하지 않는다. 세 층은 $\Theta/W/E$이고, [NL]의 level은 전부 $W$ 안에 있다 |

표 18-2 — [MC] 원 표기 → 이 책의 표기

| 원 표기 | 이 책의 표기 | 주의 |
|---|---|---|
| $M$, $M_t$, $M^{(s)}_t$ — 모듈 겸 상태 | 상태는 $W$, 캐시된 체크포인트는 $W^{(i)}$ | 읽기 함수 $\mathcal{M}(q;W)$와 상태를 분리 |
| $\theta_{M^{(i)}} = \{W^{(i)}_1,\dots,W^{(i)}_c\}$ — memory MLP 층 가중치 | $W^{(i)}$ 하나로 접는다 | 논문의 아래첨자가 이 책의 층 기호와 충돌하므로 본문에서 원 표기를 쓰지 않는다 |
| $L$ = 시퀀스 길이, $L_M$ = memory MLP 깊이 | $L$ = 시퀀스 길이, 깊이는 $L_{\mathcal{M}}$, 블록 수는 $L_{\text{layer}}$ | 충돌 없음. NM 승계 규칙상 $L$은 sequence 길이다 |
| $\mathcal{L}(M(k_t); v_t)$ — attentional bias | $\ell$ | 논문의 대문자를 그대로 옮기면 층이 뒤집힌다 |
| $S^{(i)}$ — 세그먼트 | $S^{(i)}$ 유지 | 같은 논문 안에서 위첨자 $S^{(i)}$=세그먼트, 아래첨자 $S_t$=momentum이 공존한다 [MC Eq. 35] |
| $N$ — 세그먼트 수 | $N$ (장-국소: 캐시 개수) | 예약 기호 $N_q$(문맥 공유 질의 수)와 혼동 금지 |
| $C = L/N$ — 세그먼트 크기 | $C$ (chunk) | 우연히 일치. 용량은 반드시 $C_{\text{cap}}$ |
| $\gamma^{(i)}_t$ — 캐시 $i$의 기여 게이트 | $\gamma^{(i)}_t$ 유지. **$\alpha_t$로 매핑 금지** | $\alpha_t$는 시간축 retention이고 $\gamma$는 읽기측 mixture weight다. 시간축 감쇠가 아니다 |
| $\alpha_t, \beta_t, \eta_t$ (Titans 대입) | 동일 | (U-W)와 글자 그대로 대응 |
| $u_t = x_t W_u$ — connector | $u_t$ | $q_t$와 별개 사영. $u_t = q_t$ 대안은 §3.1에 언급되나 그 행의 수치는 비어 있다(→ §18.6) |

## 18.4 어느 층을 언제 쓰는가

이 절이 이 장의 판정을 증거로 만든다. 방법은 하나다 — 두 논문에 등장하는 **모든 움직이는 객체**를 나열하고, 각각에 대해 (a) 누가 움직이는가, (b) 어느 시계인가, (c) 문맥이 끝난 뒤에도 남는가, (d) 배포 아티팩트가 바뀌는가를 채운다. (c)와 (d)가 판별식의 조건 (4)와 $\Theta$-경로 판별식에 각각 대응한다(→ ch11 §11.2).

표 18-3 — [NL]: 이 값은 누가, 어느 시계로 움직이는가

| 객체 | 움직이는 주체 | 시계 | 문맥 종료 후 존속 | 배포 아티팩트 변화 | 출처 |
|---|---|---|---|---|---|
| 임베딩, norm, window 4 short convolution, 정적 사영 $W_q$ | outer loop (AdamW) | 학습 라운드 | — (test-time 불변) | 아니오 | [NL §8.1, §8.3] |
| 여섯 self-modifying memory의 초기 상태 $W^{\square}_0$ | outer loop (meta-learn) | 학습 라운드 | 매 문맥 초기값으로 복귀 | 아니오 | [NL §8.1] |
| 여섯 memory의 running weights | inner, 식 (18-2) | chunk $C$ (직전 chunk 스냅숏 기준) | 소멸 | 아니오 | [NL Eq. 88, 90] |
| $\eta_t$, $\alpha_t$ | 학습되지 않는다 — $W^{\eta}, W^{\alpha}$가 토큰마다 생성 | 토큰 | 소멸 | 아니오 | [NL Eq. 86–88] |
| 회귀 목표 $\hat v_{\square,t}$ | 학습되지 않는다 — 갱신 대상 memory 자신이 생성 | 토큰(실행은 chunk 스냅숏) | 소멸 | 아니오 | [NL Eq. 87–88] |
| CMS level 파라미터의 running state | inner, 식 (18-1) | $C^{(\ell)}$ 토큰. 실측된 최저 주파수는 **2K 토큰** | Nested 변형은 $\lceil C^{(s)}/C^{(s+1)}\rceil$ step 뒤 초기값으로 재초기화 | 아니오 | [NL Eq. 71–72, §9.1] |
| CMS level 초기값 | outer loop (meta-learn; Nested에서는 level $s$가 $s{+}1$의 초기값을 학습) | 학습 라운드 | — | 아니오 | [NL Eq. 72–73] |
| retrofit의 15B 토큰 continual pre-training | outer loop, 진짜 (U-$\Theta$) | 평가 전 1회, 저자측 | 영구 | **예** | [NL §9.1] |
| M3 optimizer state | optimizer 상태 | 학습 step | 서빙에 존재하지 않음 | 아니오 | [NL §7.2, Alg. 1] |

표 18-3의 마지막에서 두 번째 행이 이 표의 전부다. **배포 아티팩트가 바뀌는 행이 정확히 하나이고, 그 하나는 기제가 아니라 통제군이다.** 15B 토큰 continual pre-training은 Hope에도 ICL baseline에도 동일하게 적용된다 — "MLP 블록에 어떤 변경도 없이 같은 15B 토큰 continual pre-training 과정을 거친" baseline과 비교한다고 논문이 직접 쓴다 [NL §9.1]. 실험 설계상 arm 사이에 고정된 값은 기여로 셀 수 없다.

나머지 모든 행의 "문맥 종료 후 존속" 칸이 소멸이다. 논문 자신이 자기 범주를 그렇게 정의한다 — parametric in-context learning에 대해 "획득된 in-context 지식은 현재 문맥이 제거되면 존속하지 않는다"고 박스로 진술한다 [NL §6]. 그리고 시계 칸에 세션도 sleep 라운드도 없다. 전부 토큰이고, 가장 느린 실측 눈금이 2K 토큰이다 [NL §9.1].

> **정의.** **갱신 주파수 연속체**란, 서로 다른 갱신 주기를 갖는 성분들을 단위 시간당 갱신 횟수 $f$라는 하나의 축 위에 순서지어 놓고 $f=\infty$(attention)와 $f=0$(pre-trained MLP)을 그 축의 두 끝점으로 삼는 서술을 말한다 [NL Def. 2, §1].

> **정의.** 연속체 주장이 **기제의 다리**이려면 두 조건이 필요하다. (i) 축 위의 어떤 갱신이 문맥 경계를 넘어 존속하고, (ii) 그 산출물이 배포 아티팩트에 되돌아 쓰인다. 둘 중 하나라도 없으면 그 연속체는 **표기의 다리**다 — 축 하나가 두 끝점을 이름으로 이을 뿐, 어떤 상태도 끝점 사이를 실제로 이동하지 않는다.

> **[평가]** 표 18-3에 (i)을 만족하는 행이 없고 (ii)를 만족하는 행은 통제군 하나뿐이다. 따라서 [NL]의 연속체는 표기의 다리다. 이 판정은 논문의 기여를 깎지 않는다 — [NL]은 $W$층을 주파수로 구조화하는 어휘(level, update frequency, persistent 대 adaptive 지식 저장)를 만들었고, $\Theta$-경로는 뒤에 그 어휘를 빌려 쓴다(→ ch21). 판정이 말하는 것은 어휘가 건너간 자리에 기제는 건너가지 않았다는 것이다. 물리적으로 그 자리를 건너려면 **문맥이 끝난 뒤에도 살아남는 쓰기**가 있어야 하고, 그것을 하려면 오프라인 pass가 있어야 하며, 그 pass를 [NL]은 §1에서 명시적으로 범위 밖에 두었다.

한 가지 유혹을 미리 차단한다. §7.3의 ad-hoc level stacking은 $\eta^{(\ell)} \to 0$으로 두면 "갱신된 memory 블록이 초기 상태에 가깝게 유지되어, 적응 없이 pre-trained 블록을 그대로 쓰는 결과"가 된다고 쓴다 [NL §7.3]. 스칼라 하나가 동결된 $\Theta$와 적응하는 $W$ 사이를 연속적으로 보간하는 것이며, 진짜 다리라면 이렇게 생겼을 것이다. 그러나 보간의 한쪽 끝은 "$\Theta$처럼 **행동하는** $W$"이지 $\Theta$가 아니다. 체크포인트에 되돌아 쓰이는 경로가 식 어디에도 없다.

표 18-4 — [MC]: 이 값은 누가, 어느 시계로 움직이는가

| 객체 | 움직이는 주체 | 시계 | 문맥 종료 후 존속 | 출처 |
|---|---|---|---|---|
| $\Theta$ — $W_k, W_v, W_q$, connector $W_u$, backbone | outer loop (AdamW, lr 4e-4, batch 0.5M tokens) | 학습 라운드 | 불변 | [MC App. B] |
| $W^{(s)}_t$ — 온라인 fast weight (2층 MLP, expansion 4, GELU) | inner, 식 (18-3) = 표준 (U-W) | 토큰 | 소멸 | [MC Eq. 4] |
| $\{W^{(i)}\}_{i<s}$ — 캐시된 체크포인트 | **학습되지 않는다.** 과거 inner iterate를 보존한 것 | 세그먼트 경계에서 동결, 이후 읽기 전용 | 소멸 | [MC §3] |
| $\gamma^{(i)}_t$, $r^{(i)}_t$ — 게이트·router 점수 | $W_u$를 outer가 학습, 값은 입력 의존으로 매 토큰 재계산 | 토큰 | 소멸 | [MC §3.1, §3.3] |
| $\mathrm{MeanPool}(S^{(i)})$ — 세그먼트 요약 | 학습 없음(평균 또는 합) | 세그먼트 확정 시 1회, 사전 계산 가능 | 소멸 | [MC §3.3] |
| $N$, 세그먼트 분할 스케줄 | 학습되지 않는 하이퍼파라미터 | 설계 시 고정 | — | [MC §5 Setup] |

표 18-4에는 "배포 아티팩트 변화" 열이 없다. **$\Theta$가 한 번도 갱신되지 않기 때문이다.** 그리고 세 번째 행이 이 논문의 정체를 드러낸다 — 캐시는 학습되지도, 갱신되지도 않는다. 이미 계산된 iterate를 버리지 않았을 뿐이다. 세그먼트 경계에서 소비되는 추가 추론은 0이다.

> **[해설]** Rosetta 표의 어느 항목에 붙는지가 두 논문의 차이를 한 줄로 준다(→ ch01). [NL]의 상태는 "batch로 weight 공유"가 깨지는 자리에 붙는다 — 가변 객체가 MLP 블록 자체이므로 요청마다 사본이 필요하다. [MC]의 상태는 "paged KV cache / session cache"에 붙는다 — 시퀀스 시작에 할당되고 시퀀스 끝에 해제되는, 페이지 단위만 훨씬 큰 cache class다. 어느 쪽도 "모델 재배포"에 붙지 않는다. 그것이 두 논문이 $\Theta$-경로가 아니라는 말의 서빙 쪽 번역이다.

## 18.5 비용 4종

표 18-5 — 비용 4종. 두 논문 모두 네 칸이 전부 비어 있다

| 기호 | [NL] | [MC] |
|---|---|---|
| $B_s$ | 논문에 없음 — 정의되지 않는다. 질의 전에 실행되는 pass가 없다. 가장 가까운 보고치는 wake 시점의 상각 갱신 비용 $O\!\big(\frac{1}{\hat f}\times\frac{L_{\text{layer}}}{5}\times d_{\text{in}}^2\big)$ **파라미터**이며 FLOPs도 토큰도 아니다 [NL §7.1] | 논문에 없음 — $B_s=0$. 오프라인 성분은 §4.3의 post-training 변형 하나뿐이고 그 변형의 수치가 논문에 없다 [MC §4.3] |
| $L_w$ | 논문에 없음 — Hope에 대한 지연·TTFT·ITL·처리량·wall-clock 측정이 전무하다. 논문의 유일한 효율 측정은 M3 **optimizer**의 학습 시간이다 [NL §9.7] | 논문에 없음 — 보고된 것은 차수 $O(p \times N)$/token과 **training throughput** 곡선뿐이다. decode 지연은 측정되지 않았다 [MC §3.1, §5.7] |
| $C_{\text{cap}}$ | 논문에 없음 — 바이트도, 파라미터 수도, 평가된 Hope 모델의 $d_{\text{in}}$·$L_{\text{layer}}$·level 개수도 적혀 있지 않다. 그래서 논문 자신의 $O(\cdot)$ 식조차 수치로 평가할 수 없다 [NL §7.1, §9.1] | 논문에 없음 — 차수만 있다. 캐시 개수 $1 \le N \le L$, 로그 분할이면 $N \le \log_2 L$ [MC §3.1, §4.2] |
| $\rho$ | 논문에 없음 — 라운드당 열화가 보고되지 않는다. 유일하게 정량화된 망각 지평은 모델이 아니라 optimizer의 것이다($\beta=0.9$에서 최근 6개 gradient가 누적 기여의 50% 이상, 43개가 99% 이상) [NL §4.3] | 논문에 없음 — 라운드 반복이 없으므로 개념 자체가 정의되지 않는다. 논문이 말하는 forgetting은 시퀀스 내 memory overflow다 [MC §1, §3] |

식 (A)를 두 논문 어디에도 적용할 수 없다. [NL]은 $C_{\text{sleep}}=0$이고, 모든 문맥이 meta-learn된 초기 상태에서 다시 시작하므로 질의 사이에 공유되는 아티팩트가 없어 $N_q$가 정의되지 않는다. [MC]도 마찬가지로 $C_{\text{sleep}}=0$이고, 캐시는 하나의 시퀀스 안에서만 살아 질의 간에 재사용되지 않는다. **두 논문 모두 손익분기 $N_q^*$가 존재하지 않는다.**

> **[평가]** 여기서 §18.2가 도입한 구분이 회계로 값을 한다. [MC]가 바꾼 것은 읽기식이므로, 그 비용은 상각 대상이 아니라 **토큰마다 반복되는 고정 부담**이다. 식 (A)의 분자에 들어갈 수 있는 항은 세그먼트 요약 $\mathrm{MeanPool}(S^{(i)})$ 하나인데 그것은 평균 한 번이라 사실상 0이고, 실제로 무거운 항 — 캐시 $N$개에 대한 per-token forward — 은 분모 $N_q$가 커져도 줄지 않는다. 상각 논증이 이 논문에 닿지 않는 이유는 비용이 커서가 아니라 **비용이 놓인 자리가 다르기 때문**이다. 읽기식을 바꾼 논문을 sleep-time compute으로 분류하면 이 구조적 사실이 통째로 감춰진다.

STC 기준 §6.4의 규칙을 그대로 적용한다. 네 칸이 전부 비어 있으므로 이 장은 두 논문에 대해 "효율적이다"라는 서술을 쓰지 않는다. [NL]의 §9.1이 최저 주파수 2K를 "significantly more efficient forward pass"라는 이유로 권장하고, [MC]의 §5.7이 SSC의 오버헤드가 최소라고 서술하지만, 두 진술 모두 단위가 붙은 값을 동반하지 않는다.

## 18.6 실험과 스케일

### 18.6.1 [NL]이 보고하는 것

from-scratch 학습은 FineWeb-Edu + 장문맥 혼합, vocab 32K, AdamW로 760M/30B tokens와 1.3B/100B tokens 두 규모다. 760M에서 Hope의 Wiki perplexity 18.68, LAMBADA perplexity 20.07, 평균 정확도 52.28(Titans 51.68, RWKV-7 50.55). 1.3B에서 Wiki 14.39, LAMBADA 10.08, 평균 58.04(Titans 56.82, Samba 54.46, Transformer++ 53.38) [NL Table 2, §9.3]. RULER 계열은 약 50B tokens 학습에서 Hope가 attention-free 중 최고이고, S-NIAH-3 16K에서 Hope 24.8 대 Titans 21.2 대 RWKV-7 5.8, MK-NIAH 16K에서 Hope 14.8 대 Titans 8.2다. Hope-Attention은 S-NIAH-1에서 4K/8K/16K 전부 100/100/100으로 Transformer의 88.6/76.4/79.8을 넘는다 [NL Table 1, §9.2]. 형식 언어 6종(parity, $(aa)^*$, $(abab)^*$, $a^n b^n$, $a^n b^n c^n$, Shuffle-2)에서는 전부 100.0으로 LSTM·SRWM과 동률이다 [NL Table 5, §9.5].

불리한 결과도 같은 표들 안에 있다. 짧은 in-context recall에서는 Transformer가 결정적으로 앞선다 — FDA 67.3 대 Hope 41.9이고, Hope는 attention-free 중 최고일 뿐이다 [NL Table 3, §9.4]. 10M 토큰까지 버틴다는 BABILong 결과는 **fine-tuned 모델의 결과**이며, 그림이 fine-tuned 군과 zero-shot 대형 모델 군을 분리해 그리고 Hope는 전자에 있다 [NL Figure 9, §9.2]. 그리고 논문이 스스로 가장 가까운 비교 대상으로 지목한 Cartridges는 비교에서 빠지는데, 그 이유가 수치 없는 비용 시인이다 — "Hope는 memory 사용량이 더 높고 계산 비용에 근본적인 차이가 있다" [NL §9.1]. **자원 열세를 인정하는 유일한 자리가 곧 그것을 측정하지 않는 유일한 자리다.**

### 18.6.2 캡션이 자기 표에 반증되는 자리 — [NL] Table 6

표 18-6 — [NL Table 6, §9.6] Hope ablation. perplexity는 낮을수록 좋고 reasoning 정확도는 높을수록 좋다

| 구성 | ppl | reasoning acc |
|---|---|---|
| Hope | 12.24 | 58.1 |
| w/o DGD | 13.41 | 56.5 |
| w/o momentum | 13.58 | 56.9 |
| w/o weight decay | 13.71 | 57.2 |
| w/o CMS | 13.04 | 57.3 |
| w/o inner-projection $k$ | 13.77 | 56.9 |
| w/o inner-projection $v$ | 13.90 | 55.1 |
| **w/o inner-projection $q$** | **12.19** | 57.4 |

Table 6의 캡션은 "Hope의 모든 구성요소가 성능에 긍정적으로 기여한다"고 적는다. 마지막 행이 그것을 반증한다 — inner-projection $q$를 **제거하면** perplexity가 12.24에서 12.19로 내려간다 [NL Table 6, §9.6]. level을 하나 더한 것이 언어모델링을 악화시킨 것이며, 이 사실은 §18.8이 인용할 "level 설계는 경험적이다"라는 미해결 항목과 정확히 같은 자리를 가리킨다.

지속학습 헤드라인에는 다른 종류의 문제가 있다. class-incremental 설정(CLINC 150 intent, Banking 77, DBpedia 70개 2차 클래스)에서 Hope가 ICL·EWC·InCA를 셋 다 이긴다 [NL Figure 6, §9.1]. 그런데 **restructuring을 분리하는 통제군이 없다.** Hope와 ICL baseline이 똑같이 15B 토큰 continual pre-training을 받지만, MLP 블록을 여러 level로 재구성하는 처치는 Hope 쪽에만 들어간다. 재구성이 함께 들여오는 추가 적응 파라미터와 추가 갱신 계산을 등파라미터·등계산 arm으로 분리한 실험이 없는 상태에서, 논문은 격차를 "Hope의 다중 level in-context learning" 덕으로 귀속시킨다 [NL §9.1].

주파수 축 자체의 실측은 한 자리뿐이다. LongHealth(환자 문서 20편, 5.1K–6.8K words, 문항 200개), QASPER, MK-NIAH에서 level 수와 최저 주파수를 교차한 결과이며, level이 많을수록 좋고 "최저 주파수가 높을수록 성능이 낮다"고 보고한다 [NL Figure 7, §9.1]. 그런데 권장 동작점은 더 나쁜 쪽이다 — 최저 주파수 2K가 512에 "가까운 성능"을 내면서 forward pass가 훨씬 효율적이라는 이유로 최적으로 제시된다 [NL §9.1]. **주파수 연속체를 품질 축으로 팔고 비용으로 튜닝하며, 양쪽 다 수치가 없다.**

효율 측정은 논문에 정확히 하나 있고 자기 산출물에 불리하다. M3 optimizer는 140M과 1.3B에서 Muon보다 느리고 AdaMuon과 비슷하다 [NL Figure 12, §9.7]. Hope 자체의 wall-clock·지연·처리량·메모리 측정은 없다. 한 가지 더 적는다 — "Adam은 최적 associative memory다"라는 결과는 상정된 element-wise 목적함수와 gradient 분산을 목표로 택했을 때의 **존재 구성**이며 유일성 결과가 아니다 [NL §4.2, App. B].

스케일 상한은 이렇다. from-scratch는 1.3B/100B tokens가 최대이고 7B 이상 Hope는 없다. retrofit 경로는 Llama3-8B와 Llama-3B 위에 15B 토큰 continual pre-training이며, optimizer 연구는 ViT-86M/ImageNet-21K다 [NL §9.1, §9.3, §9.7]. **그리고 실측된 가장 느린 CMS 눈금이 2K 토큰이다.** 연속체 자체의 실증 상한이 2K 토큰이라는 뜻이고, 세션은 물론이고 sleep 라운드와는 자릿수가 세 자리 이상 떨어져 있다.

> **[평가]** 주파수 논증의 동기가 된 생물학은 뇌파 대역이다 — gamma 30–150 Hz에서 delta/theta 0.5–8 Hz까지 약 300배 범위이며 전부 지각 시간 규모 안에 있다 [NL §1]. 논문은 이 유비를 구현 범위가 1 토큰에서 2K 토큰인 계산 스펙트럼으로 옮긴 뒤, 같은 유비로 축이 pre-training까지 닿는다고 말한다. 유비는 끝점의 증거가 아니다. 논문은 이 양보를 하지 않으므로 이것은 이 책의 판단이다.

### 18.6.3 [MC]가 보고하는 것

설정은 760M(24 block, dim 1536, 16 head, peak lr 1.25e-3, 30B tokens)과 1.3B(18 block, dim 2048, 8 head, 7e-4, 100B tokens)이고, FineWeb + Long-Data-Collections, vocab 32K, AdamW, batch 0.5M tokens다. 학습 문맥은 {2K, 4K, 8K, 16K, 32K}, 세그먼트는 {16, 32, 64, 128, 256, 512}에서 고르며 Table 1의 기본은 문맥 4K·세그먼트 256이다. NIAH·in-context retrieval·LongBench는 16K 문맥으로 학습했다 [MC §5, App. B].

1.3B에서 Titans(LMM) 평균 56.82가 +Log-Linear++ 57.19, +SSC 57.58, +Memory Soup 57.91, +GRM 58.33으로 오르고, DLA 53.72 → +GRM 55.96, SWLA 52.55 → +GRM 54.60이다. 같은 표의 baseline은 Transformer++ 53.19, Samba* 54.46, Miras(Memora) 55.76이다. 760M에서는 Titans 51.56 → +GRM 52.55, DLA 50.48 → +GRM 51.41이고 Transformer++ 49.64다 [MC Table 1]. NIAH 16K에서 S-NIAH-3(uuid)은 Titans 21.2 → +SSC 27.0 → +Soup 28.6 → +GRM 32.2, DLA 4.0 → +GRM 18.2이며 Transformer는 40.8이다 [MC Table 2]. in-context retrieval 평균은 Titans(LMM) 31.75 → +GRM 40.50, DLA 30.51 → +GRM 38.03이고 Transformer 41.00이다 [MC Table 3].

**즉 격차는 좁혀졌고 닫히지 않았다.** 가장 어려운 recall 설정에서 Transformer 40.8 대 최고 MC 32.2이고, retrieval 평균에서도 41.00 대 40.50이다. 초록의 "close the gap with Transformers"는 방향으로는 맞고 정도로는 표에 앞선다.

### 18.6.4 캡션이 자기 표에 반증되는 자리 — [MC] Table 5와 그 이웃들

표 18-7 — [MC Table 5, p.13] ablation. 세 열은 ppl / commonsense / retrieval

| 구성 | Titans(GRM) | Titans(SSC) |
|---|---|---|
| 전체 | 13.3 / 58.3 / 40.5 | 13.4 / 57.6 / 36.3 |
| − context-dependent | 13.4 / 57.4 / 33.0 | 13.4 / 57.1 / 32.6 |
| − gating | 13.5 / 56.9 / 32.4 | 13.5 / 56.8 / 31.9 |
| − linear memory | 13.7 / 56.3 / 34.5 | 13.8 / 56.8 / 33.4 |
| **− shared $u$ and $q$** | **00.0 / 00.0 / 00.0** | **00.0 / 00.0 / 00.0** |

마지막 행이 양쪽 모두 000/000/000으로 비어 있다. §5.6 본문은 앞의 세 설계 선택만 논하고 이 행을 언급하지 않는다. 그런데 Table 5의 캡션은 "MC의 모든 설계 선택이 그 효과에 긍정적으로 기여한다"고 단정한다 [MC Table 5, §5.6]. **빈 행 위에 세운 단정이다.** 같은 표에서 SSC의 context-dependent $\gamma$ 제거는 perplexity를 전혀 움직이지 않는다(13.4 → 13.4) — 그 설계 요소의 이득은 retrieval(36.3 → 32.6)에 국한되며 언어모델링 축에서는 "평균적으로 유의한 개선"이라는 서술이 지지되지 않는다. 그리고 Table 5의 Titans(GRM) ppl 13.3은 Table 1의 어떤 Titans+GRM perplexity와도 맞지 않는다(760M Wiki 19.14 / LMB 20.21, 1.3B 15.37 / 11.29). ablation의 모델 크기·문맥·세그먼트가 명시되지 않아 Table 1과 대조할 수 없다.

표 18-8 — [MC] 본문 진술과 자기 표의 대조

| 본문 진술 | 표가 말하는 것 |
|---|---|
| "Titans + MC and DLA + MC achieves +0.8% performance gain over the Titans" [§5.1] | 대응하는 쌍이 없다. 760M Titans 51.56 → +GRM 52.55(+0.99), 1.3B 56.82 → 58.33(+1.51), 1.3B DLA+GRM 55.96은 Titans 56.82보다 0.86 **낮다** [Table 1] |
| "GRM and then SSC achieves better results among our provided methods" [§5.1(3)] | Avg에서 Memory Soup이 모든 비퇴화 블록에서 SSC보다 위다 — 760M DLA 51.33 vs 50.85, 1.3B DLA 55.08 vs 54.64, 760M Titans 52.33 vs 52.29, 1.3B Titans 57.91 vs 57.58 [Table 1] |
| SSC가 "performs on par or better compared to other variants" [§5.7] | Table 3 Avg에서 SSC는 GRM보다 DLA 33.09 vs 38.03, Titans 36.27 vs 40.50으로 뒤지고 S-NIAH-3@16K도 27.0 vs 32.2다. SSC는 일관되게 열등하며 그 대가로 효율을 산다 |
| "All MC-enhanced variants provide performance gains compared to their base RNNs" [§5.4] | Table 4에 Avg 열이 없다. 14개 과제 단순평균은 Titans 19.53 → +GRM 19.81로 **+0.28**이다[본서 계산]. TRC 37.1 → 14.8, MNs 11.8 → 3.1, GvR 10.5 → 8.4로 세 과제가 역전되고, 이득 전부가 TQA 26.2 → 49.7 하나에서 나온다 |

세 가지를 더 적는다. 첫째, 효율 근거가 training throughput 곡선 하나뿐이다 [MC Figure 4, §5.7] — decode 지연, TTFT/ITL, 상태 바이트 중 아무것도 측정되지 않았고 비용 주장은 $O(pNL)$이라는 차수 서술에 머문다. 상수 $p$의 실측이 없는데 $p$가 이 설계의 전부다. 둘째, §3.4가 두 갈래(체크포인트 연쇄 대 독립 압축기)를 제시하고 §5.6을 보라고 넘기지만 §5.6과 Table 5 어디에도 그 축의 ablation이 없다. **논문이 스스로 지정한 검증이 누락되었다.** 셋째, §4.3의 post-training MC는 길이 외삽을 "크게" 개선한다고 주장하지만 이를 뒷받침하는 표도 그림도 없다 [MC §4.3] — v2 기준으로 이 논문에서 유일하게 오프라인에 해당하는 변형이 정확히 미측정 상태다.

스케일과 통계의 한계도 정직하게 적는다. 1.3B/100B tokens, 학습 문맥 최대 32K(평가는 16K)이고 instruction tuning이나 RLHF 단계가 없다. MQAR(5 seed 평균)을 제외한 Table 1–5 전부가 단일 시드이며 오차막대가 없어, Avg 차이의 상당수 — 예컨대 760M Titans+Soup 52.33 대 +SSC 52.29의 0.04 — 는 시드 잡음과 구분되지 않는다. 760M Transformer++ 평균 49.64는 RetNet(48.19)을 제외한 모든 recurrent baseline보다 낮고 S-NIAH-1@4K에서도 88.6으로 DLA 96.4에 지므로, 이 규모에서 Transformer 대비 비교의 기준선이 약하다 [MC Table 1, Table 2]. 마지막으로 §4.1은 attention–RNN 하이브리드가 세그먼트 크기 1의 MC와 동치라고 논증하지만 MC를 실제로 하이브리드에 얹은 실험이 없다 — Samba*와 Titans(MAL)은 baseline으로만 등장한다.

## 18.7 Systems/serving 함의

독자의 질문은 언제나 하나다 — 그래서 decode에서 뭐가 바뀌는가. 두 논문의 답이 다르다.

[NL]에서 바뀌는 것은 **가변 객체의 정체**다. Hope 블록의 서빙 상태는 여섯 개의 2층 residual MLP memory와 흘러가는 CMS tier들이며, 전부 가중치 모양이다 [NL Eq. 71, 89]. 여기서 Rosetta 표의 "batch로 weight 공유" 행이 깨진다 — 동시 요청이 이 텐서들을 공유할 수 없으므로 in-flight 시퀀스마다 사본이 필요하다. $W$-경로 논문인데도 그렇게 되는 이유는 가변 객체가 작은 행렬 상태가 아니라 MLP 블록이기 때문이다.

> **[해설]** 그 대신 얻는 것이 있다. 상태가 문맥 길이에 대해 $O(1)$이므로 읽기 트래픽이 문맥에 평평하다 — KV cache의 $O(L)$과 대비된다. decode 지연은 문맥에 평평해지고, 대가는 블록마다 토큰마다 추가되는 여섯 번의 작은 MLP forward다. 그 교환의 손익분기가 짧은 recall에서의 열세로 나타난다(FDA 67.3 대 41.9). 그리고 용량은 연속체를 따라 **평평하다** — CMS tier 하나의 상태는 $C^{(\ell)}$과 무관하게 MLP 블록 하나 전체이며, 주파수가 사는 것은 바이트가 아니라 쓰기 대역폭이다. 논문 자신의 $O\!\big(\frac{1}{\hat f}\frac{L_{\text{layer}}}{5}d_{\text{in}}^2\big)$ 파라미터 식이 정확히 이 진술의 쓰기 쪽이고, 바이트로도 $d_{\text{in}}$·$L_{\text{layer}}$의 실제 값으로도 주어지지 않는다 [NL §7.1].

[MC]에서 바뀌는 것은 **상태 성장의 모양**이다. 균일 세그먼트 $C$면 캐시 개수가 $N = L/C$이므로 상태 바이트가 문맥 길이에 선형으로 자란다 [MC §3.1, §4.2]. $W$-경로의 간판인 "고정 크기 상태"라는 장점이 여기서 사라진다. 이 논문의 진짜 주장은 고정 상태로 recall을 하려면 대가를 치러야 한다는 것이고, 그 대가가 정확히 KV cache와 같은 **선형 성장**이며, 다른 것은 성장의 상수뿐이다.

> **[해설]** 상수를 잡아 본다. **아래 산술은 이 책의 계산이며 논문의 보고값이 아니다.** 논문이 바이트를 적지 않으므로 bf16(2 bytes/elem)을 가정하고 나머지는 논문 자신의 제원을 쓴다(1.3B: 18 block, dim 2048, 8 head; memory는 2층 MLP·expansion 4). memory 모듈이 head별로 놓인다고 가정하면 $d_{\text{head}} = 2048/8 = 256$이고 head당 $2 \times 256 \times 1024 = 524{,}288$ 파라미터, block당 8 head = 4.19M, 18 block = 75.5M 파라미터/체크포인트, bf16으로 **약 151 MB/체크포인트**다. 문맥 16K·세그먼트 256이면 $N=64$이므로 시퀀스당 약 9.7 GB다. 같은 제원의 KV cache는 토큰·layer당 $2 \times 2048 \times 2 = 8$ KB, 18 layer면 147.5 KB/token이고 16K 문맥에서 약 2.4 GB다. 즉 이 가정 아래 MC는 KV cache의 **약 4배** 바이트를 쓰고, 손익분기 세그먼트 길이는 $C \approx 1024$이다. memory가 head별이 아니라 model-dim 전체에 놓이면 체크포인트가 604M 파라미터로 8배가 되어 배수는 더 커진다. **단정하는 것은 방향과 자릿수다 — 두 배치 가정 어느 쪽에서도 MC의 시퀀스당 상태는 같은 모델의 KV cache보다 무겁다.** 논문은 이 비교를 하지 않는다.

읽기 쪽 결과가 더 날카롭다. 회수는 토큰마다 모든 캐시에 대한 forward이므로 $O(p \times N)$이다 [MC §3.1]. 위 가정에서 GRM과 Soup은 decode 토큰마다 9.7 GB를 읽고, FLOPs로는 $2 \times 75.5\text{M} \times 64 \approx 9.7$ GFLOP/token으로 1.3B dense forward(2.6 GFLOP/token)의 약 3.7배다[본서 계산]. **decode가 완전히 대역폭 지배가 된다.**

SSC가 서빙 가능한 유일한 변형인 이유가 여기서 나온다. top-$k$만 로드하므로 읽기가 $O(p \times k)$로 고정되고, 논문이 명시적으로 시스템 논거를 단 유일한 지점이다 — 이런 계산은 "캐시된 memory의 상태를 accelerator에 상주시킬 필요가 없으며 토큰마다 '선택된' memory만 로드하면 된다" [MC §3.3]. $\mathrm{MeanPool}$이 사전 계산 가능하므로 router는 캐시 상태를 읽지 않고 돌고, HBM 밖(호스트 DRAM·NVMe)에 캐시를 두고 선택된 것만 끌어오는 계층 구조가 성립한다. **다만 $k$의 값이 논문에 없다.** 그리고 §18.6이 보인 대로 SSC는 품질에서 일관되게 열등하다 — 서빙 가능한 변형과 성능이 좋은 변형이 갈린다.

배치 쪽도 적는다. 캐시는 시퀀스별로 다르지만 $\Theta$는 공유되므로 shared-weight batching이 깨지지는 않는다. 대신 per-sequence 상태가 KV cache보다 훨씬 크므로 동시 시퀀스 수가 상태 바이트로 제한되고, paged KV의 페이지 단위가 토큰이 아니라 수십~수백 MB짜리 체크포인트가 되어 단편화와 스케줄링 난도가 오른다[본서 추론]. SSC의 top-$k$는 이 문제를 읽기 측에서만 완화하고 저장 측은 그대로 둔다.

## 18.8 한계와 bridge-out

### 18.8.1 논문 자신이 남긴 문제

[NL]이 남긴 것 중 이 책에 가장 중요한 항목은 §1의 범위 진술이다. 오프라인 consolidation을 이름 붙여 소개하고 "본 연구에서는 첫 단계, 즉 온라인 과정으로서의 memory consolidation에 집중한다"고 선언한다 [NL §1]. **corpus 전체에서 가장 쓸모 있는 bridge-out이며, 이것이 뒤 장에서 상속으로 실현된다**(→ ch21). 그 밖에 catastrophic forgetting이 일반적으로 해결되지 않았고 "압축의 자연스러운 귀결"이며 level 사이 용량 배분이 열려 있다는 진술 [NL §10], level 설계가 경험적이라는 사실(512 대 2K는 원리가 아니라 효율 판단이고, Table 6은 level 하나를 더한 것이 perplexity를 악화시킴을 보인다), Cartridges 비교의 유보, M3의 대형 확장 시 계산 오버헤드 우려 [NL §7.2], chunkwise stale snapshot이 진짜 self-reference를 얼마나 근사하는지에 대한 충실도 분석의 부재 [NL §8.2]가 남는다. 자기 MLP 블록을 gradient로 갱신하는 모델의 서빙 경제 — 테넌트별 분기, 표류한 상태의 체크포인트, 롤백 — 는 제기조차 되지 않는다.

[MC]가 명시적으로 남긴 future work는 하나다: 더 표현력 있는 pooling·routing 기제 [MC §6]. 현재 설계는 MeanPooling + 내적 + top-$k$라는 최소형이다. 나머지 미결은 §18.6이 적은 세 공백 — 체크포인트 연쇄 대 독립 압축기의 비교 부재, 세그먼트 분할 스케줄의 최적 형태(균일이 로그보다 나았다는 관찰만 있고 이유가 없다), post-training MC의 미측정 — 과, $q_t$를 이용해 모델이 자기 입력 시퀀스를 짓는다는 §4.1의 관찰을 아키텍처로 만들지 않은 것이다.

### 18.8.2 이 책의 판정

> **[평가]** [NL]의 연속체는 표기의 다리다(§18.4). 그러나 이 판정에는 정확한 부호가 붙는다 — **다리가 아니라는 사실이 Part II의 구조를 정당화한다.** 만약 갱신 주파수 하나로 $\Theta$와 $W$가 이어졌다면 이 책의 세 경로 구분은 눈금의 구분에 불과했을 것이고, $E$-경로만 따로 놓으면 그만이었을 것이다. 그런데 표 18-3이 보인 대로, 축 위에서 아무리 주파수를 낮춰도 얻는 것은 "느리게 움직이는 $W$"이지 $\Theta$가 아니다. 문맥 경계를 넘는 존속과 배포 아티팩트로의 되쓰기는 주파수를 낮춰서 도달할 수 있는 상태가 아니라 **다른 종류의 연산**이다. 세 경로를 따로 세우는 이유가 여기 있다.

> **[평가]** [MC]는 sleep-time compute이 아니다. 그러나 이 장에서 가장 값비싼 수치를 공급한다 — $W$층에서 recall을 사려면 상태가 문맥에 선형으로 자라야 하고, 상계를 걸려면 해상도를 내놓아야 하며, 그 성장의 상수는 같은 모델의 KV cache보다 크다(§18.7). $W$-경로가 "고정 크기 상태"라는 이름으로 파는 이점은 recall 요구가 높아지는 순간 사라진다. Part III의 용량 회계는 이 사실 위에서 세워진다.

### 18.8.3 경로 간 인용 여부

두 논문 모두 다른 두 경로에 침묵한다. 종류가 다르다.

[NL]의 $E$-경로 인용은 사실상 전무하다 — MemGPT도, 8개월 앞서 이 라인의 이름을 만든 Letta STC도, Mem0도 Zep도 ReasoningBank도 없고 RAG의 방법 인용도 없다. "RAG"는 BABILong에서 Llama-8B의 인용 없는 baseline 변형으로만 등장한다 [NL §9.2]. $\Theta$-경로는 얇고 오래되었다 — EWC 한 건이 지속학습 baseline으로 쓰일 뿐 LoRA도 ROME도 MEMIT도 SEAL도 모델 편집 문헌 전체가 없다. $W$-경로는 조밀하고(Titans, Miras, Atlas, TTT, DeltaNet, RWKV-7 등) 생물학은 대량으로 인용되되 온라인 단계에만 쓰인다. **모든 시간척도에서 갱신되는 모든 것을 통일하겠다는 논문이 외부 기억 시스템을 한 편도, 가중치 경로를 2017년 정규화 기법 하나로만 읽는다.** 통일된 것은 $W$-경로와 optimizer이며, 어휘만 셋을 덮을 만큼 넓다.

[MC]의 침묵은 더 완전하다. 전문에서 "sleep"이 0회 등장하고, $E$-경로 인용 0건, $\Theta$-경로 인용 0건이다. 인용되는 것은 전부 $W$-경로와 효율 아키텍처 계열이다(Titans, Atlas, Nested Learning/Miras, log-linear attention, MoBA, model soups, MoE, fast weight programmers, Hopfield, linear attention, Zoology/Based). $\Theta$-경로 문헌과 공유되는 조상은 model souping과 MoE 둘뿐인데, [MC]는 이들을 학습이 아니라 **읽기측 혼합**으로만 쓴다.

역방향 관찰을 하나 덧붙인다. $E$·$\Theta$ 경로 논문들은 용량을 아키텍처 문제로 다루지 않고, $W$-경로만 용량을 복잡도 차수로 정량화한다. **비용을 차수로라도 쓰는 경로가 정확히, sleep-time compute 담론과 어휘 수준에서도 접점이 없는 경로다.**

### 18.8.4 다음 장이 받아가는 것

ch19는 LoRA와 모델 편집을 읽는다. 이 장이 넘기는 것은 정확히 이 장이 없다고 판정한 것 — **$\Theta$에 실제로 쓰는 연산**이다. 주파수를 낮춰서는 $\Theta$에 닿지 못하므로, $\Theta$-경로는 낮은 주파수의 $W$가 아니라 별도의 쓰기 primitive에서 출발해야 한다. ch19가 그 primitive를 세우고 그 상한을 잰다. 그리고 이 장이 [MC]에서 얻은 용량 천장 — 상태를 늘리면 recall을 사지만 성장이 선형이라는 사실 — 은 ch19에서 rank $r$로 다시 나타난다. 두 경로가 같은 질문을 다른 자원으로 묻는다.

## 요약

- [NL]이 구현한 모든 갱신은 (U-W) 모양이고, 시계가 전부 살아 있는 시퀀스의 토큰이며, 실측된 가장 느린 눈금이 2K 토큰이다 [NL §9.1].
- 표 18-3에서 배포 아티팩트를 바꾸는 행은 15B 토큰 continual pre-training 하나뿐이고, 그것은 ICL baseline에도 동일하게 적용되므로 기제가 아니라 통제군이다 [NL §9.1].
- 따라서 갱신 주파수 연속체는 **표기의 다리이지 기제의 다리가 아니다.** 주파수를 낮춰 얻는 것은 느리게 움직이는 $W$이며, 문맥 경계를 넘는 존속과 체크포인트로의 되쓰기는 주파수로 도달할 수 없는 다른 연산이다.
- [NL] Table 6의 캡션은 모든 구성요소가 기여한다고 적지만, inner-projection $q$를 제거하면 perplexity가 12.24에서 12.19로 내려간다 [NL Table 6]. 지속학습 헤드라인에는 restructuring을 분리하는 통제군이 없다 [NL §9.1].
- [MC]는 갱신식을 한 글자도 바꾸지 않는다. 바꾼 것은 읽기식이고 $B_s=0$이며, 세그먼트 경계의 사건은 갱신이 아니라 체크포인트 저장이다 [MC Eq. 4, 7].
- [MC] Table 5의 한 행이 000/000/000으로 비어 있는데 캡션은 모든 설계 선택이 긍정적으로 기여한다고 단정한다 [MC Table 5].
- $W$층 용량의 상계를 정하는 것은 아키텍처가 아니라 세그먼트 분할 스케줄이다 — 균일 분할이면 $N=L/C$로 상계가 없고, 로그 분할이면 $N \le \log_2 L$이되 먼 과거의 해상도를 잃는다 [MC §4.2].
- 두 논문 모두 비용 4종의 네 칸이 전부 비어 있고, 식 (A)의 손익분기 $N_q^*$가 존재하지 않는다. 읽기식 변경은 질의마다 다시 지불되므로 원리적으로 상각되지 않는다.

## 자가 점검 체크리스트

- [ ] 연속체 주장이 기제의 다리인지 표기의 다리인지 판정하는 두 조건을 말할 수 있고, [NL]에 대해 그 판정을 표로 재구성할 수 있다.
- [ ] "outer loss를 inner 시계로 평가한다"가 왜 [NL]의 모호성 전부인지, 식 (18-1)을 (U-W)·(U-$\Theta$)와 대조해 설명할 수 있다.
- [ ] 갱신식을 바꾼 논문과 읽기식을 바꾼 논문을 구분하고, 그 구분이 비용 회계의 어느 항으로 나타나는지 말할 수 있다.
- [ ] $W$층 용량 천장을 세그먼트 분할 스케줄의 함수로 진술하고, 균일 분할과 로그 분할의 교환을 계산할 수 있다.
- [ ] 두 논문의 캡션이 자기 표에 반증되는 자리를 각각 수치로 지목할 수 있다.
- [ ] [MC]의 체크포인트 하나를 바이트로 환산하고, 같은 모델의 KV cache와 비교해 결론이 배치 가정에 어떻게 의존하는지 말할 수 있다.
- [ ] [NL]의 서빙 상태와 [MC]의 서빙 상태를 각각 Rosetta 사전의 어느 항목("batch로 weight 공유", "paged KV cache / session cache", "모델 재배포")에 대응시키고, 어느 것도 마지막 항목이 아닌 이유를 설명할 수 있다.

