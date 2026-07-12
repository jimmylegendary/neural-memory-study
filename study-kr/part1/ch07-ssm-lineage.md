# ch07. SSM 계보: S4 → Mamba → Mamba-2, 그리고 SSD duality

> **이 장의 목표** — 이 장을 마치면 다음을 할 수 있어야 한다. (1) continuous SSM의 ZOH discretization을 두 줄로 재현하고, step size $\Delta_t$가 왜 이 라인 전체의 gate의 기원인지 설명한다. (2) Mamba의 selectivity가 왜 convolution 훈련 모드를 제거하고 scan kernel을 강제하는지 설명한다. (3) SSD(state-space duality)에 따라 같은 recurrence를 recurrent form, masked-attention form, chunkwise form 세 경로로 손계산하고 결과가 일치함을 보인다. (4) Mamba-1의 scan과 Mamba-2의 chunkwise GEMM을 roofline 어휘로 비교한다.
>
> **왜 필요한가** — [Titans] (*Titans: Learning to Memorize at Test Time*, arXiv:2501.00663)는 이 장의 gate 계보를 자기 update의 weight decay로 일반화하는 것으로 스스로를 자리매김하고 [Titans §3.1, App. A.1], momentum recurrence의 병렬화에 S5의 parallel associative scan을 그대로 재사용한다 [Titans §3.2, Eq. 18]. [Miras] (*It's All Connected*, arXiv:2504.13173)의 general form $W_t = A_t * W_{t-1} + v_t k_t^\top$ [Miras Eq. 3]과 분류표 [Miras Table 1]가 Mamba-2를 linear attention(→ 6장)과 한 표에 넣을 수 있는 형식적 근거가 바로 이 장의 SSD다. [Atlas](arXiv:2505.23735)와 [TNT](arXiv:2511.07343)를 포함해 Part II의 실험 절에는 이 계열이 baseline으로 반복 등장한다. 마지막으로 Mamba-2의 chunked block decomposition은 9장에서 배울 chunkwise-parallel training의 첫 리허설이다 — 단, 여기서는 chunk가 아직 "정확한(exact)" 성능 knob이라는 점이 결정적 차이다.

독자는 이 계열을 이미 서빙해 봤을 가능성이 높다. Mamba block의 decode가 왜 빠른지, state가 왜 고정 크기인지는 운영 감각으로 알고 있을 것이다. 이 장의 목적은 그 운영 감각 밑에 깔린 유도를 채우고, 6장에서 만든 linear attention 어휘와 이 계열을 **하나의 update 식**으로 접합하는 것이다. 접합이 끝나면 "SSM이냐 linear attention이냐"는 질문 자체가 소멸하고, Part II의 논문들이 그러듯 "retention gate가 무엇이고 write rule이 무엇인가"만 남는다.

## 7.1 continuous SSM과 discretization: 모든 gate의 기원

**state space model(SSM)**은 연속 시간 선형 시스템을 sequence layer로 이식한 것이다. 이 소절에서만 $t$를 연속 시간 변수로 쓰고, discretization 후에는 책의 표준대로 token 인덱스로 되돌린다. 채널 하나(스칼라 입력 $x(t)\in\mathbb{R}$)에 대해 hidden state $h(t)\in\mathbb{R}^{N}$을 두고

$$
h'(t) = A\,h(t) + B\,x(t), \qquad y(t) = C^\top h(t)
$$

로 정의한다. $A\in\mathbb{R}^{N\times N}$, $B, C\in\mathbb{R}^{N}$은 이 소절에서는 상수다. 독자의 어휘로 먼저 번역하면: $h$는 길이와 무관한 **고정 크기 buffer**이고, 이 buffer가 지금까지의 입력 스트림 전체를 lossy하게 압축해 들고 있다. Rosetta 사전(→ 1장)의 "linear-RNN state = 고정 크기 lossy 압축 memory" 행이 정확히 이 대상이다.

token 단위로 계산하려면 discretization이 필요하다. step size $\Delta$를 두고 **zero-order hold(ZOH)** — 각 step 동안 입력이 상수라고 가정 — 를 적용하면 두 줄로 끝난다:

$$
h_t = \bar{A}\,h_{t-1} + \bar{B}\,x_t, \qquad
\bar{A} = \exp(\Delta A), \quad \bar{B} = (\Delta A)^{-1}\big(\exp(\Delta A) - I\big)\,\Delta B \approx \Delta B
\tag{7-1}
$$

ODE 이론은 여기까지만 필요하다. 식 (7-1)이 말하는 것: discretization은 연속 시스템을 **linear recurrence** — 즉 6장의 state 갱신과 같은 대수 구조 — 로 바꾸고, 그 계수 $\bar A$를 "지수 함수를 통과한 $\Delta$"로 만든다.

굳이 연속 시간에서 출발하는 이유를 시스템 엔지니어의 언어로 답해 두면: 이것은 물리가 아니라 **parameterization 선택**이다. decay를 $\exp(\Delta a)$ 꼴로 묶어 두면 $\Delta>0$인 한 유지 비율이 자동으로 $(0,1)$ 구간에 갇혀 recurrence가 폭주하지 않고, 초기화도 "시간 상수"라는 해석 가능한 스케일로 잡을 수 있다. 또 하나, 모델 폭 $d$의 각 채널이 독립적인 $N$차원 state bank를 갖는 SISO 구조라는 점을 기억해 두라 — layer 전체 state가 $d\times N$ 행렬이 되는 이 배치는 §7.3에서 head 단위의 $d_v\times d_k$ matrix memory로 재조직된다.

$\Delta$의 역할을 뜯어보는 것이 이 장에서 가장 중요하다. $A$가 음의 실수부를 갖는 diagonal이라고 하자(실전 SSM의 표준 설정). 채널 성분 $a<0$에 대해 $\bar a = \exp(\Delta a)\in(0,1)$이다. 그러면:

- $\Delta \to 0$: $\bar a \to 1$, $\bar B \to 0$. state를 **전부 유지**하고 현재 입력을 거의 쓰지 않는다.
- $\Delta$ 큼: $\bar a \to 0$이고, 안정한 $a<0$에서 $\bar B = \big((\exp(\Delta a)-1)/a\big)B \to -B/a$로 **포화**한다($\bar B\approx\Delta B$는 $\Delta$가 작을 때의 1차 근사일 뿐, 큰 $\Delta$에서는 무한히 커지지 않는다). 옛 state는 사라지므로 현재 입력이 상대적으로 지배한다.

즉 $\Delta$는 "시간 해상도"라는 물리적 해석을 갖지만, 계산적으로는 **유지 비율과 write 강도를 한 knob으로 묶은 gate**다. 이 라인의 모든 gate — Mamba의 selective $\Delta_t$, Mamba-2·GLA의 decay, [Titans]의 weight decay, [Miras]의 retention gate(→ 13장) — 는 전부 이 한 줄의 후손이다.

**S4와 HiPPO — 한 문단.** $A$를 아무렇게나 두면 $\bar A$의 거듭제곱이 신호를 지수적으로 죽이거나 폭주시켜 긴 문맥을 기억하지 못한다. HiPPO(Gu et al. 2020, arXiv:2008.07669)는 state가 "지금까지 입력의 polynomial 근사 계수"를 유지하도록 $A$를 구조적으로 설계했고, S4(Gu, Goel & Ré 2022, arXiv:2111.00396)는 그 구조를 diagonal-plus-low-rank로 안정화·고속화해 긴 sequence 벤치마크를 처음 뚫었다. 세부는 이 책에 필요 없다. 필요한 것은 하나: S4는 **LTI(linear time-invariant)** — $A,B,C,\Delta$가 token에 무관 — 라는 사실이다.

LTI의 계산적 귀결은 독자에게 익숙한 그림이다. 계수가 시불변이면 recurrence 전체가 convolution으로 접힌다: $y = \bar K * x$, $\bar K = (C^\top\bar B,\; C^\top\bar A\bar B,\; C^\top\bar A^2\bar B,\ldots)$. 그래서 S4는 훈련·prefill에서는 FFT convolution으로 완전 병렬, decode에서는 식 (7-1)의 recurrence로 token당 $O(N)$이다. **같은 모델이 phase에 따라 두 개의 계산 모드를 갖는다** — 독자가 매일 다루는 prefill/decode asymmetry가, 커널 구현이 아니라 모델 정의 수준에서 나타난 첫 사례다.

**S5와 associative scan — 한 문단.** S5(Smith, Warrington & Linderman 2023, arXiv:2208.04933)는 $A$를 diagonal로 두고, convolution 대신 **parallel associative scan**으로 recurrence를 병렬화했다(선행: Martin & Cundy 2018, arXiv:1709.04057; scan 알고리즘 자체는 Blelloch 1990). 원리는 독자가 아는 prefix-sum과 동일하고 monoid만 다르다. $s_t = a_t s_{t-1} + b_t$ 꼴의 recurrence에서 원소를 $(a_t, b_t)$ 쌍으로 두면, 연속 구간의 합성이

$$
(a_2, b_2)\circ(a_1, b_1) = (a_2 a_1,\; a_2 b_1 + b_2)
$$

로 정의되는 associative 연산이 되어 Blelloch tree로 $O(\log L)$ 깊이에 계산된다. 이 kernel을 기억해 두라 — [Titans §3.2, Eq. 18]는 momentum recurrence $S_t = \beta_t S_{t-1} - \eta_t u_t$ ($u_t$는 chunk 안에서 병렬로 구한 per-token gradient)가 정확히 이 꼴의 linear recurrence임을 지적하고 같은 scan으로 푼다. 9장에서 재회한다.

## 7.2 Mamba: selectivity — gate가 input의 함수가 되다

LTI의 대가는 표현력이다. 계수가 token에 무관하므로 S4는 모든 token을 **같은 비율로** 감쇠시키고 같은 강도로 쓴다. "이 token은 기억하고 저 token은 무시한다"는 content 기반 선택이 원리적으로 불가능하다. Gu & Dao 2023 (arXiv:2312.00752)은 selective copying과 induction head 류의 합성 과제로 이 한계를 시연하고(그림 7-1), 해법으로 **selectivity**를 제안했다: $\Delta_t, B_t, C_t$를 입력 $x_t$의 함수로 만든다($A$ 자체는 고정하되 $\Delta_t$를 통해 시변이 된다). 이것이 Mamba다.

![그림 7-1 — Mamba의 selectivity를 요구하는 두 합성 과제. 입력과 출력 간격이 일정한 표준 copying(왼쪽)은 실제 입력을 볼 필요가 없어 시불변(LTI) convolution 모델이 완벽히 푼다. 그러나 간격이 무작위인 selective copying(오른쪽 위)과, 문맥에 따라 답을 회수해야 하는 induction heads(오른쪽 아래)는 관련 token(색칠)을 무관 token(흰색)과 content 기준으로 구별해 유지·무시를 결정하는 시변 모델을 요구한다. 이것이 LTI를 깨고 $\Delta_t,B_t,C_t$를 입력의 함수로 만든 이유다. 출처: Gu & Dao, *Mamba* (arXiv:2312.00752), 원문 Fig.2 — **원저자의 그림(third-party), 본서의 결과가 아님**.](/home/jimmy/repos/neural-memory-study/figures/ref/ext-2312.00752-fig2.png)

selectivity의 의미는 §7.1의 $\Delta$ 해석에서 바로 나온다. $\bar a_t = \exp(\Delta_t a)$가 token마다 달라지므로, 모델은 token을 보고 "state를 유지할까, 밀어낼까"를 결정한다. Gu & Dao 2023은 특정 파라미터화($N=1$, $A=-1$, $B=1$, $s_\Delta=\mathrm{Linear}$, $\tau_\Delta=\mathrm{softplus}$)에서 selective SSM이 고전 RNN의 gate 식 $h_t = (1-g_t)h_{t-1} + g_t x_t$, $g_t = \sigma(\mathrm{Linear}(x_t))$로 환원됨을 정리로 보여 [Gu & Dao 2023 Theorem 1, §3.5.1; 증명 App. C], 이것이 LSTM 이래의 gating과 같은 계보임을 명시한다.

<!-- FIG: ch07/fig-02-selectivity-gate -->

계산 구조의 귀결이 독자에게 더 중요하다. 계수가 시변이 되는 순간 convolution kernel $\bar K$가 존재하지 않는다. **LTI를 깨면 FFT 모드가 소멸하고, recurrence(또는 scan)만 남는다.** Mamba가 "hardware-aware selective scan"이라는 커널 엔지니어링 — scan을 SRAM 안에서 수행하는 kernel fusion, backward를 위한 state recomputation — 에 논문 한 절을 쓰는 이유가 이것이다. 독자의 세계로 옮기면 Mamba의 커널은 FlashAttention과 같은 부류의 IO-aware 최적화다: 중간 텐서의 DRAM 왕복을 크게 줄인다. 그러나 fusion은 scan의 낮은 arithmetic intensity 자체를 없애지 못해 연산은 여전히 memory-bandwidth-bound이고([Mamba §3.3.2]는 selective scan을 memory-bandwidth-bound로 규정한다), 게다가 **op mix 문제**(tensor core를 쓰지 못하는 elementwise 연산 위주)까지 겹친다. 두 제약이 공존한다는 것이 Mamba-2의 출발점이다.

recomputation 항목은 2장에서 배운 개념이 실물로 등장하는 첫 장면이므로 짚고 간다. 훈련의 backward pass는 forward의 중간값 — 여기서는 매 token의 state $h_t$ — 을 다시 필요로 한다. 길이 $L$의 sequence에서 $d\times N$ state를 전부 저장하면 activation memory가 $L$에 비례해 커지므로, Mamba의 커널은 forward에서 state를 버리고 backward에서 입력으로부터 다시 계산한다. 2장의 "activation memory vs recomputation" trade-off가 커널 설계 결정으로 나타난 정확한 사례이며, FlashAttention이 attention 행렬을 저장하지 않고 backward에서 재계산하는 것과 같은 수다.

> **[해설]** 이 라인의 어휘로 미리 번역해 두면: Mamba의 $\Delta_t$는 retention($\bar a_t$)과 write 강도($\bar B_t$의 스케일)를 **한 knob에 묶은** gate다. 이후 논문들은 이 결합을 풀어낸다 — [Miras]는 유지 비율을 retention gate $\alpha_t$로 독립시키고(→ 13장), [Titans]는 write 강도를 inner learning rate $\eta_t$로 독립시킨다(→ 12장). 반면 write rule 자체는 Mamba에서도 여전히 Hebbian 덧셈이다. 즉 Mamba의 혁신은 "무엇을 지울까"에 있지, "어떻게 쓸까"에는 없다. 후자를 바꾸는 것이 delta rule 계열(→ 6장)과 TTT(→ 8장)다.

serving 관점의 접점 하나. Mamba layer의 decode state는 채널당 $N$개 float, 모델 폭 $d$ 전체로는 $d\times N$ 행렬이다(Mamba-1의 기본 설정은 $N=16$, Gu & Dao 2023). 문맥이 1M token이어도 state 크기는 불변 — KV cache처럼 자라지 않는다. 독자가 아는 "Mamba는 decode가 싸다"의 실체가 이 고정 크기 RMW(read-modify-write)다.

운영 측면 각주 하나를 더. Mamba block의 decode 상태는 SSM state만이 아니다. block 앞단의 짧은 causal depthwise convolution이 최근 입력 몇 token의 window를 상태로 요구하므로, 서빙 엔진이 실제로 관리하는 per-sequence 상태는 "SSM state + conv state"의 묶음이다. 그래도 총량이 문맥 길이와 무관한 상수라는 성질은 변하지 않는다 — paged KV cache처럼 블록을 할당·해제·조각모음하는 관리 문제가 사라지는 대신, 고정 크기 상태를 요청마다 하나씩 들고 다니는 문제로 바뀐다. 이 상태 묶음이 곧 1장의 Rosetta 사전이 "per-session weight state — 새로운 cache class"라고 예고한 대상의 가장 온건한 형태다.

## 7.3 Mamba-2와 SSD: 한 recurrence, 두 얼굴, 세 계산 경로

Mamba-2(Dao & Gu 2024, arXiv:2405.21060)의 첫 수는 **제약**이다: transition을 per-head 스칼라 곱으로 줄인다 — $\bar A_t = \alpha_t I$, $\alpha_t\in(0,1)$은 $x_t$의 함수. 표현력을 일부 포기하는 대신, 이 제약이 두 가지를 산다. 첫째, head의 state들이 하나의 행렬로 묶인다. 둘째, 그 행렬 갱신이 6장에서 이미 본 식이 된다:

$$
W_t = \alpha_t W_{t-1} + v_t k_t^\top, \qquad y_t = \mathcal{M}(q_t; W_t) = W_t\,q_t
\tag{7-2}
$$

여기서 SSM 원 표기와의 대응은 $B_t \leftrightarrow k_t$, $C_t \leftrightarrow q_t$, $x_t \leftrightarrow v_t$이며, $W_t\in\mathbb{R}^{d_v\times d_k}$다. 6장 카탈로그의 GLA/Mamba-2 행 그대로다: $\alpha_t$가 상수면 RetNet, $\alpha_t\equiv 1$이면 vanilla linear attention, diagonal이면 GLA, data-dependent 스칼라면 Mamba-2. Mamba-1의 채널별 elementwise 감쇠도 [Miras Eq. 3]의 general form $W_t = A_t * W_{t-1} + v_t k_t^\top$($A_t$: diagonal 또는 스칼라)에 diagonal 사례로 포섭된다. [Miras Eq. 8]의 논의는 이 세 특수화($\alpha=1$ / 학습되는 상수 / data-dependent 스칼라)를 명시적으로 열거하며 Mamba-2를 마지막 사례로 지목한다.

표 7-1 — SSM 원 표기 ↔ 통일 표기 ↔ inference 어휘 (장-국소 대응표)

| SSM 원 표기 (Mamba/Mamba-2) | 통일 표기 | inference 어휘 |
|---|---|---|
| $B_t$ (입력 투영) | $k_t$ | write 주소 (key) |
| $C_t$ (출력 투영) | $q_t$ | read 주소 (query) |
| $x_t$ (채널 값) | $v_t$ | 저장되는 값 (value) |
| $\bar A_t = \exp(\Delta_t A) = \alpha_t I$ | $\alpha_t$ | 남기는 비율 = 학습된 eviction |
| state $h_t$의 head 묶음 | $W_t\in\mathbb{R}^{d_v\times d_k}$ | 고정 크기 lossy 압축 KV cache |
| $\Delta_t$ | retention과 write 강도의 결합 knob | cache 정책과 write 크기를 함께 정하는 스위치 |

이제 이 장의 핵심 개념이다. **SSD(state-space duality)**는 식 (7-2)의 recurrence가 계산하는 함수가 **masked linear attention과 동일하다**는 동치 관계다 [Dao & Gu 2024]. 유도는 unroll 두 줄이다. $W_0 = 0$에서 식 (7-2)를 풀면 $W_t = \sum_{j\le t}\big(\prod_{s=j+1}^{t}\alpha_s\big)\,v_j k_j^\top$이고, 읽기를 대입하면

$$
y_t = \sum_{j=1}^{t}\Big(\prod_{s=j+1}^{t}\alpha_s\Big)\,(q_t^\top k_j)\,v_j
\quad\Longleftrightarrow\quad
Y = \big(\Gamma \odot (QK^\top)\big)V, \qquad \Gamma_{tj} = \prod_{s=j+1}^{t}\alpha_s \;\;(t\ge j)
\tag{7-3}
$$

(decay mask의 원문 기호는 $L$이지만 sequence 길이 $L$과 충돌하므로 이 장에서는 장-국소 기호 $\Gamma$로 쓴다. 하삼각 바깥, 즉 $t<j$인 성분은 0 — causal mask다.)

이 식이 말하는 것: **decay 누적곱을 mask로 갖는 attention**과, 고정 크기 state의 recurrence는 같은 함수의 두 표현이다. 왼쪽(recurrent form)은 token당 $O(d_k d_v)$에 순차 계산하고, 오른쪽(attention form)은 $O(L^2)$에 완전 병렬 계산한다. 어느 쪽으로 계산할지는 **정확도가 아니라 하드웨어 사정으로 고르는 알고리즘 선택**이다.

Dao & Gu 2024는 이것을 행렬 구조론으로 일반화한다. sequence mixing 전체를 하삼각 행렬 $\Gamma\odot(QK^\top)$의 곱으로 보면 이 행렬은 **semiseparable** 구조 — 하삼각 영역에 완전히 포함된 부분행렬(특히 대각선 아래 off-diagonal block)이 낮은 rank를 갖는 구조화 행렬; 대각을 가로지르는 block은 full rank일 수 있다 — 를 가지며, 이를 dense로 실체화해 곱하면 attention 모드, 인수분해된 구조를 이용해 곱하면 recurrent 모드가 된다 [Dao & Gu 2024, Def. 3.1]. "duality"라는 이름은 같은 구조화 행렬에 대한 이 두 곱셈 알고리즘의 쌍대성을 가리킨다. 이 대응과 두 집합의 교집합을 한 장으로 요약한 것이 그림 7-2다.

![그림 7-2 — state-space duality의 전체 지도. 왼쪽은 SSM 표기와 masked-attention 표기의 성분별 대응($C\leftrightarrow Q$ read 주소, $B\leftrightarrow K$ write 주소, $X\leftrightarrow V$ 값, 그리고 state 행렬 $A\leftrightarrow$ decay mask $L$)과 각각의 linear form·quadratic form을 나열한다. 오른쪽은 SSM 집합과 structured masked attention 집합이 겹치는 영역이 곧 SSD임을 보인다 — $A$에 scalar-identity 구조를 준 SSM(= 이 장 식 (7-2)의 $\bar A_t=\alpha_t I$)이 1-semiseparable SMA와 정확히 만나며, RetNet·linear attention이 그 교집합 안에 놓인다(원문의 decay mask $L$은 이 책 표기에서 $\Gamma$). 출처: Dao & Gu, *Mamba-2* (arXiv:2405.21060), 원문 Fig.4 — **원저자의 그림(third-party), 본서의 결과가 아님**.](/home/jimmy/repos/neural-memory-study/figures/ref/ext-2405.21060-fig4.png)

실전 답은 둘의 절충이다. sequence를 크기 $C$의 chunk로 자르고(§1.2의 chunk 시작 offset $\xi(t,C) = C\lfloor(t-1)/C\rfloor$ 사용), chunk 경계에서만 state를 전달하면:

$$
y_t = \Big(\prod_{s=\xi(t,C)+1}^{t}\alpha_s\Big)\, W_{\xi(t,C)}\,q_t \;+\; \sum_{j=\xi(t,C)+1}^{t}\Big(\prod_{s=j+1}^{t}\alpha_s\Big)(q_t^\top k_j)\,v_j
\tag{7-4}
$$

첫 항은 chunk 경계 state의 기여(cross-chunk: GEMV를 chunk 단위로 모으면 $C\times d_k$ 대 $d_k\times d_v$ GEMM), 둘째 항은 chunk 내부의 masked attention($C\times C$ GEMM)이다(그림 7-3). 경계 state의 갱신 역시 $d_v\times C$ 대 $C\times d_k$ GEMM 하나다. 결과: **모든 무거운 연산이 GEMM이 된다.**

![그림 7-3 — SSD의 chunk 분해 알고리즘. sequence mixing 행렬(위)을 chunk 단위 block으로 나누면 대각 block(파랑)은 chunk 내부의 masked-attention 계산 — 식 (7-4)의 둘째 항, intra-chunk — 이 되고, off-diagonal block(주황)은 chunk 경계의 hidden state를 통해 전달되는 chunk 간 기여 — 첫 항, inter-chunk — 가 된다. 아래 그림은 같은 분해를 입력/state/출력의 chunk 흐름으로 다시 그린 것으로, chunk 안은 병렬(수직 화살표)로, 경계 state만 순차(수평 화살표)로 전달됨을 보인다. 이 분해가 무거운 연산을 전부 GEMM으로 바꿔 tensor core를 회복하는 것이 Mamba-2 속도 향상의 핵심이다. 출처: Dao & Gu, *Mamba-2* (arXiv:2405.21060), 원문 Fig.5 — **원저자의 그림(third-party), 본서의 결과가 아님**.](/home/jimmy/repos/neural-memory-study/figures/ref/ext-2405.21060-fig5.png)

이것이 Mamba-2가 tensor core를 되찾은 방법이고, [Dao & Gu 2024]는 이 SSD 알고리즘이 Mamba-1의 fused selective scan 대비 2–8× 빠르다고 보고한다. 스칼라 gate 덕에 커널이 단순해져 state 차원도 Mamba-1보다 크게 키울 수 있게 되었다. 원문은 이를 Mamba-1의 8× 이상이라고 표현하며 [Dao & Gu 2024 §1], 실험은 과제별로 $N \in \{16, 64, 256\}$을 쓴다 — 단일 default $N$은 논문에 명시가 없다 [Dao & Gu 2024 Fig. 8, Table 5].

한 가지를 지금 박아 두어야 한다. 식 (7-4)의 chunk 분해는 **항등 변형**이다. transition이 linear이기 때문에 결합법칙으로 항을 재배열했을 뿐, 계산되는 함수는 $C$와 무관하게 식 (7-3)과 bit-exact(부동소수점 재배열 오차 제외)로 같다. FlashAttention tiling과 정확히 같은 지위다. 9장에서 만나는 chunkwise training은 다르다 — state 갱신이 gradient(state에 비선형)가 되는 순간 이 재배열이 불가능해지고, chunk 시작 상태에 gradient를 고정하는 **근사**(식 (M4)의 stale-snapshot)가 들어오며, 그때부터 $C$는 함수 자체를 바꾸는 semantic hyperparameter(→ 9장)가 된다. "Mamba-2의 chunk는 tiling이고, TTT의 chunk는 근사다" — 이 한 문장이 이 장과 9장을 가르는 경계선이다.

decode 쪽 접점 하나. SSD는 훈련·prefill의 이야기고, decode 비용은 Mamba-1이든 Mamba-2든 per-token state RMW로 같은 차수다(상세는 §7.6). "Mamba-2가 빠르다"는 주장을 서빙 엔지니어가 들을 때는 **어느 phase의 throughput인지**를 물어야 한다 — 이 질문 습관은 Part II에서 각 논문의 효율 주장을 감사할 때 그대로 재사용된다.

## 7.4 계보 정리: 세 세대의 gate, 그리고 이 라인의 접속점

[Titans App. A.1]은 linear recurrent model의 역사를 gate의 성격으로 3세대로 나눈다. 이 구분은 Part II 전체의 지도이므로 표로 고정해 둔다.

표 7-2 — linear recurrent model의 세대 구분 ([Titans App. A.1]의 세대 구분 재구성; 모델 배치 일부는 6장 카탈로그 기준)

| 세대 | transition/gate | write rule | 대표 모델 | 통일 표기 update |
|---|---|---|---|---|
| 1세대 | data-independent decay | Hebbian 덧셈 | S4, S5, RetNet, LRU, RWKV | $W_t = \alpha W_{t-1} + v_t k_t^\top$ |
| 2세대 | data-dependent gate | Hebbian 덧셈 | Mamba, Mamba-2, Griffin, GLA, RWKV-6 | $W_t = \alpha_t W_{t-1} + v_t k_t^\top$ |
| 3세대 | (gate 유무 다양) | delta rule / online learning | DeltaNet, Gated DeltaNet, Longhorn, TTT, RWKV-7 | $W_t = W_{t-1}(I - \eta_t k_t k_t^\top) + \eta_t v_t k_t^\top$ 등 |
| 다음 세대 | gate + momentum (token flow) | GD + momentum + weight decay | Titans-LMM | 식 (M2) |

이 표의 세로축이 곧 이 책의 서사다. 1→2세대는 "무엇을 지울까"를 학습 가능하게 만들었고(이 장), 2→3세대는 "어떻게 쓸까"를 Hebbian 덧셈에서 optimization step으로 승격시켰으며(6장, 8장), [Titans]는 거기에 momentum — 논문의 표현으로는 token flow — 을 더해 자신을 다음 세대로 규정한다 [Titans App. A.1].

master update와의 접속은 이렇게 읽으면 된다. 식 (M2)는 $W_t = \alpha_t W_{t-1} + S_t$였다. Mamba-2는 여기서 momentum을 끄고($\beta_t = 0$) surprise 항을 Hebbian write $v_t k_t^\top$로 바꾼 특수 사례다. 그리고 이 Hebbian write조차 gradient의 언어로 다시 읽힌다. [Miras]는 dot-product attentional bias $\tilde\ell_t = -2\langle W k_t,\, v_t\rangle$에 GD 한 걸음을 적용하면(step size는 흡수) 정확히 $+\,v_t k_t^\top$의 write가 나옴을 보이고, 이에 따라 Mamba-2를 "dot-product bias + $\ell_2$ retention + GD" 조합으로 분류한다 [Miras Eq. 8, Table 1]. 이 bias는 $W$에 대해 선형이라 gradient가 현재 state를 전혀 참조하지 않는다 — 옛 값을 읽고 고쳐 쓰는 correction이 원리적으로 없고, 그래서 crosstalk(→ 5장)가 남는다. 이 자리 배치가 이 라인 전체의 설계 공간(어떤 bias를 최소화할까 × 무엇을 남길까 × 어떤 optimizer로)을 여는 첫 수가 된다.

gate의 다음 진화도 여기서 예고된다. [Titans]는 자신의 forgetting mechanism — update 앞에 곱해지는 data-dependent retention — 가 Mamba-2류 gating mechanism의 일반화라고 명시한다 [Titans §1, §3.1]. 무엇이 일반화인가? Mamba-2의 gate는 행렬 state 하나를 스칼라로 감쇠시키지만, Titans의 retention은 memory가 "작은 모델의 weights"로 승격된 뒤에도 같은 자리에서 작동하는 weight decay다. 2장에서 optimizer 객체의 성분으로 배운 weight decay — $(1-\eta\lambda)w$의 그 감쇠 — 가, 여기서는 sequence 축 위에서 token마다 값이 달라지는 학습된 eviction으로 재해석된다. 같은 수식, 다른 축이다. 이 재해석의 정식 전개는 12장의 몫이고, 이 장은 그 재료인 gate 계보를 공급했다.

의미론적 차이 하나는 기록해 둘 가치가 있다. [Miras footnote 2]는 Mamba-2류의 gating과 [Titans]의 retention이 **완전 소거의 의미**에서 다르다고 지적한다: Mamba-2의 gate가 0이 되면 memory 전체가 지워지고 다음 token은 "처음 보는 데이터"가 되는 반면, Titans는 소거하기 **전의** 상태 $M_{t-1}$에서 surprise를 측정한 뒤 그 update를 남긴다(원문 표현 "cold start"). 즉 $\alpha_t\to 1$이어도 새 상태는 $W_{\mathrm{init}}$이 아니라 surprise 항 $S_t$이며($M_t=(1-\alpha_t)M_{t-1}+S_t$이고 $S_t$의 gradient는 소거 전 $M_{t-1}$에서 평가된다 [Titans Eq. 13–14]), 직전 memory에서 얻은 정보가 살아남는다. (학습된 초기 상태 $W_{\mathrm{init}}$로 실제로 되돌리는 reset은 Titans가 아니라 [TNT]의 local memory에서 나타나며, 15장에서 시스템 설계 축으로 커진다.)

마지막으로 두 개의 예고. 첫째, [NL](arXiv:2512.24695)은 이 장의 전 계보를 update frequency의 스펙트럼 위에 재배열한다 — SSM state는 token마다 갱신되는 가장 빠른 level일 뿐이다(→ 16장). 둘째, 독자의 어휘로 이 절 전체를 한 줄로 압축하면: **gate는 학습된 cache eviction policy이고, 이 장의 역사는 eviction policy가 고정 상수에서 content-aware 함수로 진화해 온 역사다.** write policy의 진화는 다음 장의 몫이다.

## 7.5 Worked micro-example: 한 recurrence를 세 경로로 계산하기

$d_k = d_v = 2$, $L = 3$으로 식 (7-2)–(7-4)를 전부 손으로 확인한다. 입력은 다음과 같다 ($W_0 = 0$이므로 $\alpha_1$은 결과에 안 쓰인다):

| $t$ | $k_t^\top$ | $v_t^\top$ | $\alpha_t$ | $q_t^\top$ |
|---|---|---|---|---|
| 1 | $(1,\,0)$ | $(2,\,0)$ | — | — |
| 2 | $(0,\,1)$ | $(0,\,4)$ | $1/2$ | — |
| 3 | $(1,\,0)$ | $(1,\,1)$ | $1/2$ | $(1,\,0)$ |

<!-- FIG: ch07/fig-01-ssd-three-paths -->

**경로 1 — recurrent form (식 7-2).** decode가 하는 계산이다.

$$
W_1 = v_1 k_1^\top = \begin{pmatrix}2&0\\0&0\end{pmatrix},\qquad
W_2 = \tfrac12 W_1 + v_2 k_2^\top = \begin{pmatrix}1&0\\0&4\end{pmatrix},\qquad
W_3 = \tfrac12 W_2 + v_3 k_3^\top = \begin{pmatrix}1.5&0\\1&2\end{pmatrix}
$$

읽기: $y_3 = W_3 q_3 = (1.5,\;1)^\top$.

**경로 2 — attention form (식 7-3).** mask는 $\Gamma_{tj} = \prod_{s=j+1}^{t}\alpha_s$이므로 3행은 $(\alpha_2\alpha_3,\;\alpha_3,\;1) = (\tfrac14,\;\tfrac12,\;1)$이다. score는 $q_3^\top k_j = (1,\;0,\;1)$. 곱하면 가중치 $(\tfrac14,\;0,\;1)$:

$$
y_3 = \tfrac14\,v_1 + 0\cdot v_2 + 1\cdot v_3 = (0.5,\,0)^\top + (1,\,1)^\top = (1.5,\;1)^\top
$$

**경로 3 — chunkwise form (식 7-4), $C=2$.** chunk 0 = token 1–2를 처리해 경계 state $W_2$를 만들고, token 3은 chunk 1에 속한다($\xi(3,2)=2$). cross-chunk 항과 intra-chunk 항:

$$
y_3 = \alpha_3\,W_2\,q_3 + (q_3^\top k_3)\,v_3 = \tfrac12(1,\,0)^\top + (1,\,1)^\top = (1.5,\;1)^\top
$$

세 경로가 같은 값을 준다 — duality가 근사가 아니라 항등임을 숫자로 확인했다. 두 가지 관찰을 덧붙인다. 첫째, $W_2$를 memory로 읽어 보면 $k_1=(1,0)\to(1,0)$ (원래 값 $(2,0)$이 gate에 반 감쇠됨), $k_2=(0,1)\to(0,4)$ (방금 써서 온전함)이 저장돼 있다. gate는 eviction의 연속 버전 — 지우는 대신 흐리게 만든다. 둘째, $k_3 = k_1$로 key가 충돌하는데 write는 Hebbian 덧셈이므로 $y_3$에는 감쇠된 옛 값 $\tfrac14 v_1$이 새 값 $v_3$에 **섞여** 나온다. 5장의 crosstalk가 gate로 완화될 뿐 제거되지 않는 현장이다. 같은 key를 덮어쓰고 싶다면 write rule 자체를 바꿔야 한다 — 6장의 delta rule이 하는 일이고, 8장의 TTT가 일반화하는 일이다.

## 7.6 Systems bridge: scan에서 GEMM으로 — 9장 리허설

이 장의 계산 이야기를 독자의 roofline 위에 정리한다.

**decode.** 세 모드 중 recurrent form만 남는다. head당 비용은 state $W\in\mathbb{R}^{d_v\times d_k}$의 read-modify-write: FLOPs는 갱신(outer product 누적)과 읽기(GEMV)로 $O(d_k d_v)$, 메모리 트래픽도 state 왕복 $O(d_k d_v)$ bytes다. FLOP/byte 비가 $O(1)$이므로 **decode는 여전히 bandwidth-bound다** — KV cache 스캔이 state RMW로 바뀌었을 뿐, "decode는 메모리가 지배한다"는 독자의 세계관은 그대로 유효하다. 달라진 것은 그 트래픽이 문맥 길이와 무관하게 상수라는 점이다.

batching과의 접점도 짚어 둔다. decode에서 state는 요청마다 다르지만 모양은 전부 같으므로($d_v\times d_k$ 행렬들), batch 전체의 읽기·갱신이 균일한 batched GEMV와 rank-1 update로 묶인다 — 길이가 제각각인 KV cache 스캔을 paged layout으로 달래던 세계와 비교하면, 스케줄러가 다뤄야 할 자유도가 하나 줄어든 셈이다. 단, 여기서 batching이 평화로운 이유는 memory가 행렬 하나라서 per-request 연산이 batched GEMM 한 번으로 정규화되기 때문임을 기억해 두라. 8장에서 memory가 per-request로 훈련되는 MLP가 되는 순간, 같은 질문이 grouped GEMM의 문제로 재등장한다(→ 10장).

**prefill/훈련 — Mamba-1의 병목.** selective scan의 op mix는 elementwise 곱-합이다. kernel fusion으로 중간값의 DRAM 왕복을 크게 줄여도 scan의 arithmetic intensity는 낮아 연산이 여전히 memory-bandwidth-bound이고([Mamba §3.3.2]), 동시에 연산이 GEMM이 아니므로 tensor core가 놀아 상한이 기기의 vector-ALU throughput — 최신 GPU에서 tensor core FLOPS의 작은 분수 — 에도 걸린다([Mamba-2 §2.1]). 즉 Mamba-1은 **bandwidth 제약과 op mix 제약을 동시에** 안는다 — fusion은 전자를 완화할 뿐 제거하지 못하고 후자는 건드리지 못한다. roofline 그림으로 말하면: 지붕의 낮은 쪽 처마 밑에 앉아 있는 것이다.

**prefill/훈련 — Mamba-2의 답.** 식 (7-4)는 chunk당 세 종류의 GEMM을 만든다: intra-chunk score $QK^\top$($C\times d_k$ 대 $d_k\times C$), mask 적용 후 value 곱($C\times C$ 대 $C\times d_v$), 그리고 state 항($C\times d_k$ 대 $d_k\times d_v$와 경계 갱신). token당 비용은 $O(C\,d + d_k d_v)$, sequence 전체로는 $O(LCd + L\,d_kd_v)$ — $C=1$이면 순수 recurrence, $C=L$이면 순수 attention으로 퇴화하는 보간식이다. $C$를 키울수록 GEMM이 두꺼워져 arithmetic intensity가 올라가고 tensor core 활용이 회복된다. 이것이 "같은 수학을 GEMM으로 다시 쓰기"의 전부이며, FlashAttention에서 tile 크기를 SRAM에 맞추던 감각과 같은 종류의 튜닝이다.

표 7-3 — 식 (7-2)/(7-3)/(7-4)의 세 가지 계산 모드

| 모드 | 총 FLOPs 차수 | 주 연산 (하드웨어 유닛) | 병렬성 | exact? | 주 용도 |
|---|---|---|---|---|---|
| recurrent ($C=1$) | $O(L\,d_kd_v)$ | GEMV + rank-1 RMW (vector ALU) | 순차 | exact | decode |
| quadratic ($C=L$) | $O(L^2 d)$ | GEMM (tensor core) | 완전 병렬 | exact | 짧은 prefill |
| chunkwise ($1<C<L$) | $O(LCd + L\,d_kd_v)$ | GEMM + 경계 state 전달 (tensor core + scan) | chunk 내 병렬, 경계 순차/scan | exact | 훈련, 긴 prefill |

이 표에서 가장 중요한 열은 "exact?"다. 이 장의 세 모드는 전부 exact — $C$는 품질에 영향 없는 순수 성능 knob이다. 9장에서 write rule이 gradient step으로 바뀌면 같은 표의 chunkwise 행이 근사로 강등되고, $C$가 품질 축으로 승격되며, "품질 최적 $C$(작음) vs MFU 최적 $C$(큼)"의 긴장이 생긴다 — 그 긴장이 [TNT] 한 편의 주제다. 이 장은 그 대비의 기준선(모든 것이 exact였던 세계)을 제공한다.

## 요약

- SSM은 continuous 선형 시스템의 ZOH discretization이고, step size $\Delta$는 유지 비율($\bar a = \exp(\Delta a)$)과 write 강도를 한 knob으로 묶은 gate다 — 이 라인의 모든 gate의 기원이다.
- S4/HiPPO는 LTI라서 훈련·prefill을 convolution으로 병렬화하지만, 같은 이유로 content 기반 선택이 불가능하다.
- Mamba의 selectivity는 $\Delta_t, B_t, C_t$를 입력의 함수로 만들어 표현력을 얻는 대신 convolution 모드를 잃고, IO-aware scan kernel에 의존한다 — bandwidth 문제는 fusion으로 풀리지만 op mix 문제는 남는다.
- Mamba-2는 transition을 per-head 스칼라 $\alpha_t$로 제약해 update를 $W_t = \alpha_t W_{t-1} + v_t k_t^\top$ (식 7-2)로 만들고, SSD에 의해 이 recurrence는 decay-mask attention $Y = \big(\Gamma\odot(QK^\top)\big)V$ (식 7-3)와 동일한 함수다.
- 같은 함수를 recurrent($C=1$, decode) / quadratic($C=L$, 짧은 prefill) / chunkwise($1<C<L$, 훈련) 세 모드로 계산할 수 있고, 셋 모두 exact다 — 이 장의 chunk는 tiling이지 근사가 아니다.
- Mamba-2의 chunkwise 재구성은 무거운 연산을 전부 GEMM으로 바꿔 tensor core를 되찾았고, [Dao & Gu 2024]는 fused scan 대비 2–8×의 속도 향상을 보고한다. decode 비용은 두 세대가 같은 차수다.
- 계보는 gate의 3세대(상수 decay → data-dependent gate → delta/online learning)로 정리되며 [Titans App. A.1], Mamba-2는 (M2)에서 surprise 항을 Hebbian write로 바꾼 특수 사례다 — write rule의 승격이 다음 두 장의 주제다.

## 자가 점검 체크리스트

- [ ] ZOH discretization 식 (7-1)을 재현하고, $\Delta\to 0$과 $\Delta$가 큰 극한에서 $\bar a$와 $\bar B$가 각각 어떻게 되는지로 gate 동작을 설명할 수 있다.
- [ ] selectivity가 왜 convolution 훈련 모드를 제거하고 scan을 강제하는지, 그리고 kernel fusion이 풀지 못하는 문제가 무엇인지 설명할 수 있다.
- [ ] §7.5의 예제를 recurrent/attention/chunkwise 세 경로로 손계산해 같은 $y_3$를 얻을 수 있다.
- [ ] "Mamba-2의 chunk는 exact한 tiling이고 9장의 chunk는 semantic한 근사다"를 transition의 선형성을 근거로 한 문장으로 정당화할 수 있다.
- [ ] Mamba-1 scan과 Mamba-2 SSD의 차이를 op mix와 tensor core 활용의 언어로, decode와 prefill을 구분해서 설명할 수 있다.
- [ ] Mamba의 state와 gate를 inference 어휘로 옮길 수 있다: state = 고정 크기 lossy 압축 KV cache, gate = 학습된 eviction policy, prefill = chunk 단위 병렬 write, decode = 상수 크기 state RMW.

## 다음 장으로

이 장의 끝에서 write rule은 여전히 Hebbian 덧셈이었고, 그래서 key 충돌의 crosstalk는 gate로 흐려질 뿐 지워지지 않았다. 6장의 delta rule은 이를 "한 걸음의 gradient descent"로 바꾸는 첫 승격이었다. 8장은 이 승격을 끝까지 밀어붙인다: state를 아예 작은 모델의 weights로 선언하고, token마다 그 모델을 self-supervised loss로 한 걸음씩 훈련하는 TTT-Linear/TTT-MLP를 다룬다. 그 순간 이 장의 duality가 누리던 exactness는 깨진다 — state 갱신이 비선형이 되므로 chunk 분해는 항등이 아니라 근사가 되고, 그 근사를 GEMM으로 조직하는 dual form이 8장과 9장의 중심 문제가 된다.
