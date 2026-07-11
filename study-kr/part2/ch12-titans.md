# ch12. Titans: Learning to Memorize at Test Time

## 12.1 Bridge-in: TTT가 남긴 문제

Part II의 첫 논문은 [Titans] (*Titans: Learning to Memorize at Test Time*, arXiv:2501.00663; Ali Behrouz, Peilin Zhong, Vahab Mirrokni, Google Research, 2024-12-31 v1)이다. 이 장의 출발점은 8장이 멈춘 자리다. TTT-Linear/TTT-MLP(→ 8장)는 "hidden state는 작은 모델의 weights이고, state update는 그 모델에 대한 gradient descent step이다"라는 등식을 language modeling 스케일에서 처음 작동시켰고, chunk 안의 gradient를 chunk 시작 상태에서 평가하는 mini-batch 근사(→ 8장의 dual form, → 9장의 일반화)로 훈련을 GEMM 위에 올려놓았다. 그러나 TTT가 남긴 결핍은 명확했다.

첫째, **지우는 방법이 없다.** TTT의 inner optimizer는 순수 SGD다. state는 고정 크기인데 write는 무한히 쌓이므로, 문맥이 길어지면 언젠가 포화한다. 5장의 어휘로 말하면 crosstalk이 누적되는데 eviction이 없다. 둘째, **optimizer가 기억을 갖지 않는다.** 매 token의 update는 그 token 하나의 gradient만 반영한다. token 사이의 흐름 — 어떤 사건이 중요했다면 그 직후 token도 함께 저장해야 한다는 시간적 구조 — 이 update rule에 존재하지 않는다. 셋째, **deep memory의 가치가 미검증이다.** TTT는 MLP state를 허용했지만 깊이가 실제로 무엇을 사주는지 실험으로 답하지 않았다.

한편 6장의 DeltaNet 계열은 정반대의 트레이드를 택했다. DeltaNet과 Gated DeltaNet은 memory를 matrix로 묶어 두는 대가로 정확한 closed-form chunkwise recurrence를 얻었고, Gated DeltaNet은 여기에 retention gate(원문 표현 forget gate)까지 더했다. 즉 pre-Titans 지형에서는 "표현력 있는 inner optimizer + 깊은 nonlinear memory"와 "병렬화 가능한 훈련"이 양립 불가능한 것처럼 보였다. 한쪽 끝에 retention gate를 가진 linear memory(Gated DeltaNet), 다른 쪽 끝에 forgetting도 momentum도 없는 gradient 기반 deep memory(TTT)가 있었고, 가운데가 비어 있었다.

[Titans]는 그 가운데를 채우겠다는 논문이다: momentum과 weight decay를 모두 갖춘 inner optimizer로 깊은 MLP memory를 test time에 훈련하되, chunkwise 병렬화와 associative scan으로 훈련 throughput을 지킨다. 그리고 Appendix C에서 Gated DeltaNet, Longhorn, RWKV-7, TTT 전부를 자신의 특수 사례로 회수한다 — 이 회수가 다음 장 [Miras] (*It's All Connected*, arXiv:2504.13173; 이 책은 framework 이름 Miras로 통칭한다)의 출발점이 된다.

## 12.2 문제의식과 논문의 핵심 주장

논문이 스스로 정의한 문제는 linear model의 자기모순이다 [Titans §1]. attention은 정확하지만 $O(L^2)$이고, linear recurrent model은 $O(L)$이지만 고정 크기 state에 역사를 압축한다. 그런데 linear cost가 가장 절실한 구간이 바로 very long context이고, very long context야말로 작은 vector·matrix state에 제대로 압축될 수 없는 구간이다. 효율이 필요한 곳에서 품질이 무너지도록 설계되어 있는 셈이다.

[Titans]는 이 모순을 memory의 언어로 재서술한다. 모든 sequence model은 memory 구조, write(update) 연산, read(retrieval) 연산의 3요소로 분해된다 [Titans §2]. Transformer는 압축 없이 KV 쌍을 append하는 growing memory이고(read는 유사도 검색), linear attention은 $v_tk_t^\top$를 하나의 matrix에 더해 쓰는 additive write라 overflow가 예정되어 있으며, GLA·Mamba-2 계열은 data-dependent erase를, DeltaNet 계열은 replace-then-write를 더한 것이다(→ 6장, 7장). 이 관점에서 논문은 다섯 질문을 세운다: (Q1) 좋은 memory 구조는 무엇인가, (Q2) 좋은 update rule은 무엇인가, (Q3) 좋은 retrieval은 무엇인가, (Q4) 서로 다른 memory 시스템 여러 개를 어떻게 합성하는가, (Q5) memory는 deep해야 하는가 [Titans §1, §2].

또 하나의 문제의식이 이 라인 전체의 성격을 결정한다. training data를 외우는 것은 일반화·privacy·OOD 관점에서 바람직하지 않으므로, pre-training에서는 **외우는 방법**을 meta-learn하고 실제 memorization은 test context 위에서만 일어나야 한다는 것이다 [Titans §3.1]. 즉 "learning to memorize at test time"이라는 제목 자체가 4장의 bilevel 구조 선언이다.

핵심 주장은 넷이다. (1) 무엇을 외울지는 **surprise**가 결정하며, surprise는 inner loss의 gradient다. (2) surprise에는 관성이 필요하며, 그것이 곧 momentum이다. (3) 잊기는 per-token weight decay이며, 이는 현대 linear RNN의 gate 전부의 일반화다. (4) memory는 matrix가 아니라 깊은 MLP여야 한다. 그리고 이 네 가지를 다 넣고도 훈련은 chunkwise로 병렬화된다는 것이 시스템 측 주장이다.

## 12.3 Core mechanism (통일 표기)

이 절은 책 전체의 기준 수식 (M2)를 유도하는 절이다. 이후 모든 Part II 장은 여기서 확정되는 형태와의 차이로 자기 논문을 서술한다.

### 12.3.1 memory는 weights다: 상태와 함수의 분리

**neural long-term memory module (LMM)** 은 작은 신경망이며, 그 weights 자체가 sequence model의 recurrent state다. 이 책의 표기로 상태는 fast weights $W_t$, 읽기는 함수 $\mathcal{M}(\cdot;W_t)$이다. [Titans]는 $L_{\mathcal{M}} \ge 1$층 MLP를 쓴다(입출력 폭 $d$) [Titans §3.1]. 원문은 $\mathcal{M}(x)$(write를 동반하는 forward)와 $\mathcal{M}^*(x)$(read 전용 forward)를 한 기호에 겹쳐 쓰지만, 이 책은 상태와 함수를 분리한다(§12.3.10의 대응표 참조) — write는 $W_{t-1}\to W_t$의 명시적 update 식으로, read는 $y_t=\mathcal{M}(q_t;W_t)$로 쓴다.

memory가 풀 문제는 associative memory(→ 5장) regression이다. token $x_t\in\mathbb{R}^d$를 slow weights의 projection으로 key와 value로 바꾸고 —

$$
k_t = W_K x_t,\qquad v_t = W_V x_t,\qquad W_K, W_V \in \mathbb{R}^{d\times d}
$$

— memory가 key에서 value를 재생하도록 하는 inner objective를 세운다 [Titans Eq. 11, Eq. 12]:

$$
\ell(W;\,k_t,v_t) \;=\; \big\|\mathcal{M}(k_t;W) - v_t\big\|_2^2 .
\tag{12-1}
$$

attention의 KV lookup과 같은 추상 — key와 비슷한 query가 오면 연관된 value를 돌려준다 — 을 weights 안에 저장하겠다는 것이다. 여기서 결정적인 문장이 나온다: "training of the memory is in the inner-loop, and so parameters $W_K$ and $W_V$ are hyperparameters in the above loss function" [Titans §3.1]. 즉 $W_K, W_V$는 inner 문제의 **hyperparameter**로, inference 중에는 절대 움직이지 않고 outer loop에서만 학습된다. 이 구분은 §12.4에서 전면 전개한다.

### 12.3.2 momentary surprise와 기본 update

무엇을 얼마나 세게 쓸 것인가? [Titans]의 답: 지금 들어온 token이 이미 저장된 내용을 얼마나 위반하는가, 즉 inner loss의 gradient 크기다. **momentary surprise**는 $g_t^{\mathrm{in}} = \nabla_W\,\ell(W_{t-1};k_t,v_t)$로 정의되며, gradient가 클수록 새롭고 예상 밖인 입력이므로 더 세게 외운다 [Titans §3.1]. 기본 update는 1-step gradient descent, 즉 표준형 (M1) 그대로다 [Titans Eq. 8]:

$$
W_t \;=\; W_{t-1} \;-\; \eta_t\, g_t^{\mathrm{in}},
\qquad g_t^{\mathrm{in}} = \nabla_W\,\ell(W_{t-1};k_t,v_t),
\tag{12-2}
$$

여기서 $\eta_t$는 data-dependent inner learning rate — token $x_t$의 함수로 slow head가 산출하는 게이트다. 주의: 원문에서 이 학습률의 기호는 $\theta_t$이고, 원문의 $\eta_t$는 아래에서 momentum decay로 쓰인다. 이 책은 기호를 교차 정리했다(표 12-1).

> **[해설]** $\mathcal{M}$이 linear($W\in\mathbb{R}^{d\times d}$)이면 식 (12-2)는 정확히 delta rule(→ 5장)이고, 따라서 (12-2)는 DeltaNet(→ 6장)과 TTT(→ 8장)가 이미 서 있던 자리다. inference 독자의 어휘로: KV cache append가 rank-1 GEMM write로 바뀐 것이며, "쓰기 전에 현재 저장값을 읽어 오차만 쓴다"는 점에서 append가 아니라 read-modify-write다(→ 1장 Rosetta).

### 12.3.3 past surprise: momentum이 sequence layer 안으로 들어오다

momentary surprise만으로는 실패하는 상황이 있다. 큰 surprise 직후에는 loss가 이미 낮아져 gradient가 급감하므로, 놀라운 사건 **뒤에 이어지는** 중요한 token들이 약하게 저장된다. 논문의 비유로, 놀라운 순간은 한동안 주의를 붙들어 그 시간 구간 전체를 기억하게 만든다 [Titans §3.1]. 그래서 surprise를 둘로 분해한다: **past surprise** $S_t$(최근 과거의 surprise 누적)와 momentary surprise $g_t^{\mathrm{in}}$ [Titans Eq. 9, Eq. 10]:

$$
S_t \;=\; \beta_t\, S_{t-1} \;-\; \eta_t\, g_t^{\mathrm{in}},
\qquad
W_t \;=\; W_{t-1} + S_t .
\tag{12-3}
$$

$S_t$는 weights와 같은 shape의 버퍼이며, 이것은 문자 그대로 SGD-with-momentum의 momentum buffer다(→ 2장의 optimizer-as-object: state $S_t$, update 식 (12-3), cost는 elementwise 연산). 즉 [Titans]의 첫 번째 발명은 "optimizer state를 sequence layer의 recurrent state로 승격"시킨 것이고, 논문은 이를 momentum이 "시간축 위의 surprise에 대한 memory"로 작동한다고 읽는다. **data-dependent surprise decay** $\beta_t\in[0,1]$은 token의 함수다: $\beta_t\to 0$이면 과거 surprise의 전파를 끊고(문맥 전환), $\beta_t\to 1$이면 온전히 전파한다(현재 token이 직전 문맥과 강하게 결속) [Titans §3.1]. $\beta_t,\eta_t$를 상수가 아니라 token의 함수로 두는 것이 설계의 요점이다 — 과거 surprise가 지금의 write에 영향을 줘야 하는지는 두 token이 같은 문맥에 있는지에 달려 있기 때문이다.

### 12.3.4 forgetting: weight decay가 per-token retention이 되다

수백만 token을 다루려면 deep memory라도 언젠가 가득 찬다. 그래서 마지막 성분으로 **forgetting mechanism**을 넣는다 — 그리고 그것의 정체는 per-token **weight decay**다 [Titans Eq. 13, Eq. 14]. 통일 표기로 쓰면 이것이 이 책의 master update (M2)다:

$$
S_t \;=\; \beta_t\, S_{t-1} \;-\; \eta_t\, \nabla_W\,\ell(W_{t-1};k_t,v_t),
\qquad
W_t \;=\; \alpha_t\, W_{t-1} \;+\; S_t .
\tag{12-4}
$$

읽는 법: $\alpha_t\in[0,1]$은 **남기는 비율**이다. $\alpha_t\to 1$이면 과거의 추상을 전부 보존한 채 덧쓰고, $\alpha_t\to 0$이면 memory를 통째로 소거한다. **방향 주의**: Titans 원문의 $\alpha_t$는 "잊는 비율"이라 $(1-\alpha_t)\mathcal{M}_{t-1}$로 등장하며, 이 책의 $\alpha_t^{\text{(통일)}} = 1-\alpha_t^{\text{(Titans)}}$이다(표 12-1). 이 게이트가 뒤에 [Miras]가 retention gate(→ 13장)로 재이론화하는 대상이고, Titans 문맥에서는 "forgetting mechanism"이라는 원문 표현을 그대로 인용할 수 있다. 논문은 이 weight decay가 Mamba-2, GLA, Gated DeltaNet의 gating을 임의의(deep) memory로 일반화한 것임을 보인다 [Titans §3.1, App. C].

정리하면, [Titans]의 write 연산 전체는 **"mini-batch gradient descent + momentum + weight decay" 한 step이고, optimizer의 세 스칼라($\eta_t,\beta_t,\alpha_t$)가 모두 token마다 게이트로 산출된다.** 훈련을 한 번도 본 적 없는 독자를 위해 다시 말하면: 2장에서 optimizer를 (state, update, cost)를 가진 객체로 배웠는데, [Titans]는 그 객체를 통째로 모델 내부에 넣고 그 hyperparameter 세 개를 학습된 함수로 바꾼 것이다. 뒤의 ablation(§12.6)은 세 성분의 기여를 weight decay > momentum > convolution > persistent memory 순으로 매긴다 [Titans Table 5].

### 12.3.5 read: 갱신 없는 forward

읽기는 weight 갱신이 없는 순수 forward pass다 [Titans Eq. 15]:

$$
q_t = W_Q x_t,
\qquad
y_t \;=\; \mathcal{M}(q_t;\,W_t).
\tag{12-5}
$$

$W_Q$ 역시 outer에서 학습되는 projection이다. read가 상태를 바꾸지 않는다는 것은 KV cache lookup과 동일한 성질이며, write와 read가 명시적으로 분리된 두 연산이라는 것이 이 라인의 두-연산(compression/retrieval) 추상의 원형이다.

### 12.3.6 왜 deep memory인가

$\mathcal{M}$을 matrix $W\in\mathbb{R}^{d\times d}$로 두면 식 (12-1)은 online **linear** regression이고, 그 최적해는 역사가 선형 사상으로 압축 가능하다는 가정을 깔고 있다 [Titans §3.1]. **deep memory** — $L_{\mathcal{M}}\ge 2$층 MLP — 는 key–value 구조가 비선형이어도 저장할 수 있다는 것이 논문의 논거이며(2층 이상 MLP의 표현력 우위를 인용), §12.6의 깊이 실험과 ablation(linear memory로 바꾸면 long-context 점수가 92.68→85.34로 하락 [Titans Table 5])이 실증을 맡는다. 이 라인의 후속작들이 표준형으로 굳히는 2-layer residual MLP(§1.3의 "표준 deep memory")의 출발점이 여기다. 단, [Titans] v1은 memory MLP의 폭·activation 등 세부를 명시하지 않는다.

### 12.3.7 persistent memory

LMM은 **contextual memory** — 내용이 전적으로 입력 문맥에 의존하는 memory — 다(attention의 KV도 마찬가지다). [Titans]는 여기에 입력과 무관한 세 번째 memory를 더한다. **persistent memory**는 $N_p\ge 1$개의 학습되는 input-independent 벡터 $P=[p_1\ \ldots\ p_{N_p}]$를 sequence 앞에 붙이는 것이다 [Titans Eq. 19]:

$$
x_{\mathrm{new}} = P \,\Vert\, x .
$$

세 가지 정당화가 제시된다 [Titans §3.3]. (1) memory 관점: task를 수행하는 방법에 대한 지식은 입력에 따라 변하면 안 되므로 input-independent parameter에 살아야 한다. (2) FFN 관점: $\mathrm{FFN}(x)=W_V\,\mathrm{Softmax}(W_K x)$ [Titans Eq. 20] — FFN은 K/V가 data-independent한 attention이며(Sukhbaatar et al. 2019, arXiv:1907.01470 인용), persistent token은 attention 내부에서 같은 역할을 한다. (3) 기술적 관점: causal attention은 초반 token에 과도한 weight를 주는 attention-sink 편향이 있는데, 학습 가능한 prefix token이 그 편향을 흡수·재분배한다.

### 12.3.8 세 가지 합성: MAC / MAG / MAL, 그리고 LMM 단독

Q4의 답으로 [Titans]는 attention을 정밀한 short-term memory, LMM을 서서히 잊는 long-term memory로 두고 세 가지 합성을 제시한다 [Titans §4]. 세 변형 모두 persistent token을 포함한다. 풀네임은 **Memory as Context (MAC)**, **Memory as Gate (MAG)**, **Memory as Layer (MAL)** 이며 이하 약어로 쓴다.

<!-- FIG: ch12/fig-01-mac-dataflow -->

**MAC** [Titans §4.1, Eqs. 21–25]. sequence를 크기 $C$의 segment로 자른다(원문의 segment 크기 $C$ — 이 책은 chunk 크기와 동일시한다, 표 12-1). $n$번째 segment의 token 행렬을 $X^{(n)}\in\mathbb{R}^{C\times d}$(token $nC{+}1\ldots(n{+}1)C$)라 하자. segment가 들어오면, 직전 segment까지 갱신된 memory 상태 $W_{nC}$에서 **읽고**, attention을 돌린 뒤, 그 출력을 memory에 **쓴다**:

$$
r_\tau = \mathcal{M}(q_\tau;\,W_{nC}) \quad (\text{segment의 각 token } \tau,\ q_\tau = W_Q x_\tau),
$$

$$
\tilde{X}^{(n)} = P \,\Vert\, r \,\Vert\, X^{(n)},
\qquad
a = \mathrm{Attn}\big(\tilde{X}^{(n)}\big),
$$

$$
W_{(n+1)C} \leftarrow \text{식 (12-4)를 } a \text{의 token들에 대해 실행},
\qquad
y_\tau = \mathrm{gate}\big(a_\tau,\ \mathcal{M}(a_\tau;\,W_{(n+1)C})\big).
$$

즉 attention window 안에는 persistent token, memory에서 검색해 온 역사 $r$(원문 표기 $h_t$ — outer 2차 moment와의 충돌을 피해 개명, 표 12-1; 원문 mask 표기로는 $N_l$개의 long-term memory token인데, [Titans Eq. 21]의 query가 segment 전체 — $C$개 token — 이므로 차원상 $N_l = C$가 따라 나온다; 단 [Titans Fig. 3a] 캡션은 별도 상수 $N_l$ 표기를 유지한다), 현재 segment가 나란히 놓이고, 그 안에서 full causal attention이 돈다(전체 sequence 관점의 attention mask는 segment별 block-diagonal이다 [Titans Fig. 3a]). 설계 이유가 중요하다: (i) attention이 검색된 역사와 현재 데이터를 동시에 보므로 지금 long-term 정보가 필요한지를 token 단위로 판단할 수 있고, (ii) memory에는 attention이 처리한 표현만 쓰이므로 attention이 **write filter** 역할을 해 쓸모없는 token으로 인한 memory overflow를 줄인다 [Titans §4.1]. 원문의 write 식 $\mathcal{M}_t = \mathcal{M}_{t-1}(y_t)$ [Titans Eq. 24]는 "forward pass를 통해 weight를 갱신한다"는 관행 표기로, segment 내부의 write 세분(granularity)은 명시되어 있지 않다. 최종 read도 attention 출력 $a_\tau$를 $W_Q$ 재적용 없이 그대로 query로 쓴다 [Titans Eq. 25]. 또한 retrieval이 갱신 **전** 상태 $W_{nC}$를 읽으므로 read는 항상 한 segment만큼 stale하다.

**MAG** [Titans §4.2, Eqs. 26–28]. segmentation 없이 두 branch를 병렬로 돌린다. branch 1은 persistent prefix가 항상 보이는 sliding window attention(SWA), branch 2는 prefix가 붙은 전체 sequence 위를 식 (12-4)로 도는 LMM이다. 두 branch의 출력을 학습된 elementwise gate로 결합한다 — 실험에서는 각 branch 출력을 학습 가능한 벡터 weight로 normalize한 뒤 비선형 $\sigma(\cdot)$를 거쳐 곱한다 [Titans §4.2]. 원문은 이 결합을 $\otimes$로 쓰지만 이 책은 $\mathrm{gate}(\cdot,\cdot)$로 쓴다($\otimes$ 금지, §1.3). SWA가 선명한 short-term, LMM이 fading long-term이며, 구조가 다른 head들을 가진 multi-head layer로도 읽을 수 있다(Hymba 인용).

**MAL** [Titans §4.3, Eqs. 29–31]. LMM을 하나의 layer로 SWA 앞에 직렬로 쌓는다: $\tilde{x}=P\,\Vert\,x$, $z=\mathrm{LMM}(\tilde{x})$, $y=\text{SW-Attn}(z)$ ($z$는 LMM branch의 per-token read를 쌓은 중간 출력, 장-국소 기호). 이는 H3 이래 사실상 모든 hybrid(Samba, Griffin 등)가 쓰는 관행 패턴인데, 논문은 이 설계가 **가장 약하다**고 명시한다 — 파이프라인의 힘이 각 단의 힘으로 상한되어 attention과 memory의 상보성을 살리지 못한다는 것이다 [Titans §4.3]. 기존 hybrid 전부가 MAL 모양이라는 점에서, MAC/MAG > MAL이라는 §12.6의 결과는 개별 모델 비교를 넘어 hybrid 설계 관행 자체에 대한 기소장이다.

**LMM 단독**: attention 없이 memory module만 sequence model로 쓰는 변형. memory 시스템의 각 부분은 독립적으로도 작동해야 한다는 §12.2의 Q4 철학에 따른 대조군이자, long-term memory가 홀로도 강하다는 주장의 시험대다 [Titans §4.3].

**블록 세부** [Titans §4.4]: 모든 블록에 residual connection; q/k/v 계산에 SiLU activation; query와 key의 $\ell_2$-normalization; q/k/v projection 뒤 1D depthwise-separable convolution(효과는 작지만 양수, 비용 저렴); 최종 출력 projection 앞에 normalization + linear gating.

**표현력 주장**: [Titans]는 Transformer·diagonal linear RNN·DeltaNet이 $TC^0$에 갇히는 반면(Merrill et al. 인용) Titans는 $TC^0$ 너머의 문제를 풀 수 있어 state tracking에서 이론적으로 더 표현력 있다고 주장한다(Thm 4.1). **v1에는 이 정리의 증명이 없다** — 라인 끝(NL)까지도 이 주장은 정리가 아닌 실증으로만 뒷받침된다는 점을 이 책은 반복해 기록한다. 추가 표현력의 출처로 지목되는 것은 §12.4에서 보게 될 inter-chunk nonlinear recurrence다.

### 12.3.9 특수 사례 회수: 이 update는 기존 계보 전부를 포함한다

[Titans App. C]는 라인 전체에서 가장 많이 인용되는 부록이다. memory를 linear($W_t\in\mathbb{R}^{d\times d}$)로, loss를 $\ell=\frac{1}{2}\|W k_t - v_t\|_2^2$로 두면 식 (12-4)는 다음이 된다 [Titans Eq. 32, Eq. 33]:

$$
W_t = \mathrm{diag}(\alpha_t)\, W_{t-1} + S_t,
\qquad
S_t = \mathrm{diag}(\beta_t)\, S_{t-1} - \mathrm{diag}(\eta_t)\big(W_{t-1}k_t k_t^\top - v_t k_t^\top\big).
\tag{12-6}
$$

식 (12-6)에서 $\beta_t = 0$(momentum 차단)으로 두면 Gated DeltaNet의 rule $W_t = W_{t-1}(I-\eta_t k_t k_t^\top) + \eta_t v_t k_t^\top$ (+ decay) [Titans Eq. 34]와 일치한다. Longhorn은 같은 loss를 implicit online learning으로 풀어 step size가 $\eta_t/(1+\eta_t k_t^\top k_t)$로 바뀐 delta 형태이되 retention gate가 없고 [Titans Eq. 35], RWKV-7은 같은 loss·형식을 쓴다는 이유로 명시적으로 포함되며("similar approaches such as RWKV-7 … LMM is generalizing all such models" [Titans App. C]), TTT는 forgetting도 momentum도 없는 gradient 기반 특수 사례다 [Titans App. C]. 결국 LMM은 이 계보 전체를 네 축에서 일반화한다: (1) momentum 기반(token 흐름 인지) update — 논문은 자신이 linear-recurrent 계열 최초의 momentum rule이라고 주장한다, (2) deep memory, (3) inter-chunk nonlinear recurrence, (4) retention gate. 6장의 카탈로그(§1.6) 한 줄 요약으로: Titans-LMM = (M2) + 표준 deep memory.

> **[해설]** 이 부록의 함의는 양방향이다. 앞으로는 "기존 모델 전부가 (M2)의 성분을 끈 것"이라는 통일이고, 뒤로는 "그럼 왜 하필 $\ell_2$ loss, 왜 하필 momentum+decay인가"라는 질문이다. [Titans]는 전자만 답하고 후자를 열어 둔다 — 그 열린 질문이 13장의 [Miras]다.

### 12.3.10 표기 대응표

표 12-1 — [Titans] 원 표기 ↔ 이 책의 통일 표기 (STYLE-NOTATION §1.7.1 사본 + 장-국소 행)

| Titans 원 표기 | 통일 표기 | 의미 / 주의 |
|---|---|---|
| $\mathcal{M}_t$ (상태 겸 함수) | $W_t$ (상태), $\mathcal{M}(\cdot;W_t)$ (함수) | 상태/함수 분리 (P1) |
| $\mathcal{M}(x)$ — write 동반 forward | update 식 + read로 분해해 표기 | |
| $\mathcal{M}^*(x)$ — read-only forward | $\mathcal{M}(x;W_t)$ | ★는 통일 체계에서 argmin 전용 |
| $\theta_t$ — inner learning rate | $\eta_t$ | ⚠ 기호 교차 주의 |
| $\eta_t$ — surprise(momentum) decay | $\beta_t$ | ⚠ Titans의 $\eta$는 lr가 **아니다** |
| $\alpha_t$ — forget gate, $(1-\alpha_t)\mathcal{M}_{t-1}$ | $\alpha_t^{\text{(통일)}} = 1-\alpha_t^{\text{(Titans)}}$ | **방향 반전**: 통일 $\alpha_t$는 남기는 비율 |
| $S_t$ — past surprise (momentum buffer) | $S_t$ | 동일 |
| $\nabla\ell(\mathcal{M}_{t-1};x_t)$ — momentary surprise | $g_t^{\mathrm{in}}=\nabla_W\ell(W_{t-1};k_t,v_t)$ | |
| $\mathbf{k}_t=x_tW_K$ (행벡터 관행) | $k_t=W_Kx_t$ (열벡터) | |
| $b$ — inner mini-batch(chunk) 크기 | $C$ | |
| segment $S^{(i)}$, 크기 $C$ (MAC) | "segment" (산문), 크기 $C$ | segment 크기 = chunk 크기로 동일시함을 명시 |
| $N_p,\ P,\ p_i$ — persistent memory | 동일 | |
| $d_{in}$ | $d$ | |
| $L_{\mathcal{M}}$ — memory 깊이 | 동일 | |
| $M_0$ — 초기 상태 | $W_{\mathrm{init}}$ | meta-learn됨 (암묵적) |
| $y_t=\mathcal{M}^*(q_t)$ | $y_t=\mathcal{M}(q_t;W_t)$ | |
| $\otimes$ — MAG의 gating 결합 | $y\odot g$ 또는 $\mathrm{gate}(\cdot,\cdot)$ | $\otimes$ 금지 |
| $\beta_i=\prod_{j\le i}(1-\alpha_j)$ — 누적 decay (Eq. 16) | $\bar\alpha_i=\prod_{j\le i}\alpha_j$ (장-국소) | ⚠ 통일 $\beta_t$(momentum decay)와 충돌 방지 |
| $u_t=\nabla\ell(M_{t'};x_t)$ — chunk-anchored gradient (Eq. 18) | $\hat g_\tau$ (장-국소) | ⚠ NL의 $u_t$(LSS)와 충돌 방지 |
| $\Theta_b,\ \mathbf{B}_b$ — chunk별 diagonal (Eq. 17) | $\mathrm{diag}(\tilde\eta_1,\ldots,\tilde\eta_C)$, $\tilde\eta_i=\eta_i\,\bar\alpha_C/\bar\alpha_i$ (장-국소) | 두 diagonal을 하나로 합쳐 표기; ⚠ 예약 기호 $c$(Omega window 길이)를 피해 개명 |
| $t'=t-\mathrm{mod}(t,b)$ — chunk 시작 (Eq. 16) | $\xi(t,C)=C\lfloor(t-1)/C\rfloor$ | 원문 정의는 chunk 마지막 token($t=b$)에서 $t'=b$가 되는 경계 슬립 — §1.2 정의로 교정 |
| $h_t$ — MAC의 retrieved history (Eq. 21) | $r_\tau$ (장-국소) | ⚠ outer 2차 moment $h_t$와 충돌 방지 |
| $S^{(t)},\ \tilde{S}^{(t)}$ — segment, 증강 segment (Eq. 22) | $X^{(n)},\ \tilde{X}^{(n)}$ (장-국소) | ⚠ momentum $S_t$와 충돌 방지 |
| $y_t$ — MAC의 attention 출력 (Eq. 23) | $a_\tau$ (장-국소) | layer 최종 출력 $y$와 구분 |
| $o_t$ — 최종 출력 (Eq. 25) | $y_\tau$ | 통일 규약: layer 출력은 $y$ |

## 12.4 Outer-loop training vs inner-loop test-time learning

이 절이 이 장의 심장이다. training 경험이 없는 독자의 첫 질문 — "그 게이트는 누가 학습하는가?" — 에 성분 단위로 답한다.

표 12-2 — [Titans]에서 무엇이 어느 loop에 사는가

| 구성 요소 | 소속 | 갱신 시점 · rule | 크기/비용 감각 |
|---|---|---|---|
| projection $W_K, W_V, W_Q$ | $\Theta$ (slow) | pre-training에서만, AdamW로 | inner loss의 hyperparameter |
| 게이트 산출 head ($\eta_t,\beta_t,\alpha_t$를 emit) | $\Theta$ (slow) | pre-training에서만 | inference에서는 token→스칼라 3개 산출 |
| persistent tokens $P$ | $\Theta$ (slow) | pre-training에서만; test time에 동결 | $N_p\times d$ |
| attention 블록, conv, normalization, LM head | $\Theta$ (slow) | pre-training에서만 | 통상적 backbone |
| 초기 memory 상태 $W_{\mathrm{init}}$ | $\Theta$ (slow) | pre-training에서 암묵적으로 | v1은 설정을 명시하지 않음 |
| memory weights $W_t$ | inner (fast) | **매 token, 식 (12-4)로, inference 중에도** | $P_{\mathcal{M}}\approx L_{\mathcal{M}}d^2$ per layer per sequence |
| momentum buffer $S_t$ | inner (fast) | 매 token, 식 (12-4)로 | $W_t$와 같은 shape — state 2배의 주범 |

**outer loop: 무엇이 meta-learn되는가.** memory의 일시적 상태를 제외한 전부다. outer objective는 평범한 next-token cross-entropy $\mathcal{L}$이고, 논문은 이 구조를 Reptile/CAVIA 계열 meta-learning으로 명시적으로 자리매김한다 [Titans §3.1]: inner에서는 $\mathcal{M}$의 weights를 최적화하고, outer에서는 아키텍처의 나머지를 최적화한다. 기술적으로 결정적인 사실은 이것이다 — **outer gradient는 unroll된 inner update를 관통한다.** 아래 chunkwise closed form이 곧 forward graph이며, autodiff가 그 그래프를 그대로 backprop한다. 4장의 어휘로 "optimizer를 미분한다": $W_K$를 바꾸면 매 token의 inner gradient가 바뀌고, 그것이 $W_t$의 전 궤적을 바꿔 최종 $\mathcal{L}$을 바꾼다. 그 연쇄 전체에 대한 gradient가 $W_K$를 학습시킨다. 그래서 게이트에 대한 답은: $\eta_t,\beta_t,\alpha_t$의 **값**은 inference 중 매 token 새로 계산되지만, 그 값을 만드는 **함수**(head의 weights)는 pre-training에서만 학습되고 이후 동결된다. 단, v1은 이 head의 함수형·초기화를 명시하지 않는다 — 재구현자가 채워야 하는 공백이다.

**chunkwise 병렬화: inner loop를 GEMM으로.** 순수 online rule (12-4)는 $O(L)$ FLOPs이지만 엄격히 순차적이고 matmul이 없다 — 가속기에서 최악의 형태다. [Titans §3.2]는 TTT(→ 8장)를 따라 sequence를 크기 $C$의 chunk로 자르고, chunk 안의 모든 gradient를 chunk 시작 상태에서 평가한다(9장의 stale-snapshot 근사, 표준형 (M4)). momentum을 잠시 끄고 첫 chunk($W_0=W_{\mathrm{init}}$, $t\le C$)에 집중하면 recurrence가 닫힌 꼴로 풀린다 [Titans Eq. 16]:

$$
W_t \;=\; \bar\alpha_t\, W_0 \;-\; \sum_{i=1}^{t} \eta_i\, \frac{\bar\alpha_t}{\bar\alpha_i}\, \nabla_W\,\ell\big(W_0;\,k_i,v_i\big),
\qquad
\bar\alpha_i = \prod_{j=1}^{i}\alpha_j .
\tag{12-7}
$$

retention은 누적 곱 $\bar\alpha$로 접히고, 모든 gradient가 같은 동결 상태 $W_0$에서 평가되므로 $C$개를 동시에 계산할 수 있다. 일반 위치에서는 anchor가 $W_{\xi(t,C)}$가 된다 — 이 근사가 계산되는 함수 자체를 바꾸는 semantic한 선택이라는 것, 즉 FlashAttention tiling(bit-exact)과 결정적으로 다르다는 것은 9장의 명제다. linear memory로 구체화하면 chunk 하나의 gradient 합이 GEMM 두 개로 tensorize된다 [Titans Eq. 17]:

$$
\sum_{i=1}^{C} \eta_i\, \frac{\bar\alpha_C}{\bar\alpha_i}\, \nabla_W\,\ell(W_0;k_i,v_i)
\;=\; \big(W_0 K^\top - V^\top\big)\,\mathrm{diag}\big(\tilde\eta_1,\ldots,\tilde\eta_C\big)\, K,
\qquad \tilde\eta_i = \eta_i\, \bar\alpha_C/\bar\alpha_i,
\tag{12-8}
$$

여기서 $K\in\mathbb{R}^{C\times d_k}$, $V\in\mathbb{R}^{C\times d_v}$는 chunk의 key/value 행-쌓기다. 원문 표기 $\Theta_b\mathbf{B}_b(W_0X-X)X^\top$는 행벡터 관행에 더해 $(k,v)$ 자리에 $x$를 쓰는 축약이라 재구현 시 되돌려야 한다. chunk마다 diagonal 계수만 materialize하면 되고 $L/C$개를 모두 들고 있을 필요가 없다 [Titans §3.2]. 식 (12-8)을 GEMM shape로 읽으면(→ 1장 Rosetta) 이것은 $(d_v\times C)\cdot(C\times d_k)$의 rank-$C$ write다: **chunk 크기 $C$가 곧 write GEMM의 내적 차원이고, 따라서 arithmetic intensity의 손잡이다.** MLP memory($L_{\mathcal{M}}\ge 2$)의 경우도 층별 gradient를 chunk-start weights에서 batch로 평가하는 같은 구조라고만 서술된다 [Titans §3.2].

momentum은 어떻게 되는가? chunk 안에서 $\hat g_\tau = \nabla_W\ell(W_{\xi(\tau,C)};k_\tau,v_\tau)$를 전부 미리 계산해 두면 [Titans Eq. 18]

$$
S_\tau \;=\; \beta_\tau\, S_{\tau-1} \;-\; \eta_\tau\, \hat g_\tau
\tag{12-9}
$$

는 input-dependent 전이 계수 $\beta_\tau$, 입력 $\hat g_\tau$, hidden state $S_\tau$의 1차 **선형** recurrence다 — 정확히 S5/Mamba의 selective scan과 같은 모양이므로 parallel associative scan(→ 7장, 9장)으로 chunk 내 모든 $S_\tau$를 log-depth에 얻는다. 독자가 이미 아는 scan/prefix-sum kernel이 그대로 훈련 그래프의 한가운데에 등장하는 것이다.

결과 구조가 [Titans]의 표현력-병렬성 협상이다: **intra-chunk는 linear**(matmul + scan으로 병렬), **inter-chunk는 nonlinear**(chunk 경계에서 갱신된 MLP weights가 다음 chunk의 loss에 비선형으로 재진입). 순차 임계 경로가 $L$이 아니라 $L/C$번의 MLP-weight 적용으로 줄고, 동시에 이 비선형 경계가 Thm 4.1의 표현력 주장의 근거로 지목된다 — DeltaNet 계열이 정확한 closed form을 위해 전부 linear로 남은 것과 의도적으로 다른 트레이드다. 추가 가속 옵션으로 $\alpha,\eta,\beta$를 token이 아니라 chunk의 함수로 두면 식 (12-9)가 LTI(linear time-invariant → 7장) 시스템이 되어 S4식 global convolution으로 계산 가능하지만 [Titans §3.2], 실험은 표현력을 위해 per-token 게이트를 유지했고 이 fast path는 끝내 평가되지 않았다.

**outer 레시피** [Titans §5.1]: 훈련 길이 4K, Llama-2 tokenizer(32K vocab), AdamW lr 4e-4 cosine annealing, batch 0.5M tokens, weight decay 0.1 — Gated DeltaNet의 프로토콜을 따른다. 여기서 두 층위가 한 문장에 겹친다: outer optimizer(AdamW, weight decay $\lambda=0.1$)는 $\Theta$를 학습하는 통상의 훈련이고, inner optimizer(식 (12-4))는 모델 그 자체다. 기호로도 구분된다 — outer는 무첨자 상수($\eta$, $\lambda$), inner는 시간 첨자 게이트($\eta_t,\beta_t,\alpha_t$).

**inner loop: inference에서 실제로 움직이는 것.** sequence마다 진화하는 tensor는 단 둘, $W_t$와 $S_t$다. 나머지 전부는 동결이다. per-token 비용을 세어 보자.

> **[해설]** $P_{\mathcal{M}}$을 memory parameter 수라 하면(폭 $d$ MLP에서 $P_{\mathcal{M}}\approx L_{\mathcal{M}}d^2$), write 한 번 = k/v projection $2d^2$ MAC + $\mathcal{M}$ forward $\approx P_{\mathcal{M}}$ + weight gradient를 위한 backward $\approx 2P_{\mathcal{M}}$(backprop의 표준 2× forward 비용; linear memory에서는 rank-1 outer product $(Wk_t-v_t)k_t^\top$로 퇴화) + $S,W$의 elementwise 갱신 $\approx 2P_{\mathcal{M}}$ + 게이트 head 3개(미미). read 한 번 = projection + forward $\approx d^2 + P_{\mathcal{M}}$. 합계 token당 $\approx 5$–$6\times P_{\mathcal{M}}$ FLOP 상당 — **context 길이와 무관한 상수**이며, 같은 state 크기의 linear-attention/DeltaNet layer 대비 작은 상수배, 그 상수는 $L_{\mathcal{M}}$에 선형이다. 이 산정은 이 책의 계산이고, 논문의 실측은 "LMM이 Mamba-2/Gated DeltaNet보다 약간 느리다" [Titans Fig. 9]까지다.

여기서 독자의 세계가 실제로 뒤집히는 지점을 명시한다: **backward pass가 decode 안으로 들어온다.** decode = "read-only forward + KV append"라는 불변식은 이 라인에서 폐기되고, decode 한 step은 forward + backward + optimizer step + read(식 (12-5))가 된다. 다만 update rule 자체에는 train/inference 불일치가 없다 — prefill은 훈련 때의 chunkwise 공식 그대로 긴 prompt를 chunk 병렬로 흡수하고, decode는 $C=1$의 token-by-token online write로 진행하면 되며, 차이는 outer gradient를 함께 계산하느냐뿐이다. 훈련의 batch 축 역할을 sequence 축이 대신한다는 것(inner loop의 mini-batch = chunk)도 여기서 처음 실물로 확인된다 — 2장과 9장이 예고한 최대 혼동 지점이다.

변형별 흐름: LMM/MAG/MAL은 (prefix가 붙은) 스트림의 모든 token에서 memory를 갱신하고, MAC은 segment 단위로 "읽기 → attention → attention 출력만 쓰기"를 반복한다. test time에 persistent parameter는 동결(task 지식), attention weights는 window 안의 in-context learner, LMM만이 "여전히 학습 중"이다 [Titans §4.1].

## 12.5 Concept ledger delta

[Titans]는 이 라인의 시조이므로 폐기하는 개념은 없고, ledger의 첫 행들을 만든다. 아래 표는 누적 관점 — 각 개념이 이후 어느 장에서 어떻게 진화하는지 — 을 함께 적는다.

표 12-3 — [Titans]가 ledger에 추가한 개념

| 개념 (이 논문이 도입) | 내용 | 이후의 운명 |
|---|---|---|
| surprise (momentary / past) | momentary = $g_t^{\mathrm{in}}$, past = $S_t$ | Miras: 임의 attentional bias의 gradient로 일반화(→ 13장); Atlas: 문맥의 surprise(window)로(→ 14장); NL: Local Surprise Signal(→ 16장) |
| momentum-as-memory (정의 소유: 2장·16장) | optimizer state $S_t$의 recurrent state 승격; linear-recurrent 계열 최초의 momentum rule 주장 [Titans App. C] | Miras의 출시 모델들은 제거; Atlas가 Muon으로 부활(→ 14장); NL이 정리로 격상(→ 16장) |
| weight decay = per-token retention | $\alpha_t W_{t-1}$; Mamba-2/GLA/Gated DeltaNet gate의 deep-memory 일반화 | Miras가 retention gate로 공식 재이론화(→ 13장) |
| deep memory | $L_{\mathcal{M}}\ge 2$ MLP; linear memory = 선형 압축 가정의 거부 | 2-layer residual MLP가 라인 표준으로(→ 13장); capacity 이론은 Atlas(→ 14장) |
| persistent memory | $N_p$개 학습 prefix token; task 지식의 input-independent 저장 | NL/CMS에서 frequency-0 극점으로 재해석(→ 16장) |
| contextual memory | 내용이 입력 문맥에 의존하는 memory의 총칭(LMM + attention KV) | persistent와의 이분법이 NL의 frequency 연속체로 확장(→ 16장) |
| MAC / MAG / MAL | 합성 3형; MAL(관행 hybrid)의 열등 판정 | Atlas·NL의 hybrid 실험이 재사용(→ 14, 16장) |
| intra-chunk linear / inter-chunk nonlinear recurrence | 병렬성과 표현력의 협상 지점; Thm 4.1의 근거 | TNT가 reset으로 이 비선형 사슬을 끊어 context parallelism 획득(→ 15장) |
| meta-learned $W_{\mathrm{init}}$ (암묵) | $W_0$이 outer 학습 대상이라는 함의; v1은 명시하지 않음 | TNT에서 load-bearing으로 승격(→ 15장), 개념 소유는 4장·15장 |
| chunk-당-상수 게이트 (LTI fast path) | 제안만 되고 미평가 | 라인 어디에서도 재평가되지 않은 열린 갈래 |

표기 차원의 유산도 있다: Titans의 forget-방향 $\alpha_t$는 [Miras]가 retention 방향으로 뒤집어 재정의하며, 이 책은 처음부터 Miras 방향(남기는 비율)으로 통일했다(표 12-1).

## 12.6 실험과 스케일

**설정** [Titans §5.1]: 170M/340M/400M 모델은 FineWeb-Edu 15B tokens, 760M은 30B tokens로 훈련. 훈련 길이 4K. baseline은 Transformer++, RetNet, GLA, Mamba, Mamba-2, DeltaNet, TTT, Gated DeltaNet과 hybrid(Samba, Gated DeltaNet-H2). 400M baseline 수치는 재실행이 아니라 Gated DeltaNet 논문의 보고치를 재사용했다 [Titans App. B].

**Language modeling + commonsense reasoning** [Titans Table 1]. 모든 스케일에서 LMM 단독이 비-hybrid baseline 전부를 이긴다. 340M: LMM 평균 46.17 vs Gated DeltaNet 45.42, TTT 44.51 — TTT와의 격차가 곧 momentum+forgetting의 값이고, Gated DeltaNet과의 격차가 deep nonlinear memory의 값이라는 것이 논문의 독법이다 [Titans §5.2]. 760M: LMM Wiki ppl 20.04 / 평균 51.56 vs Gated DeltaNet 21.18 / 49.69. hybrid에서는 MAC/MAG/MAL 셋 모두 Samba와 Gated DeltaNet-H2를 이긴다 — 760M에서 MAG Wiki ppl 18.61, MAC 평균 52.51 vs Gated DeltaNet-H2 19.88 / 51.49. 일관된 순서는 MAC ≈ MAG > MAL이며, 모듈이 같고 배치만 다르므로 이 격차는 순수하게 합성 설계의 몫이다 [Titans §5.2].

**S-NIAH (RULER, 2K–16K)** [Titans Table 2]. 기제별 귀속이 가장 선명한 실험이다. Titans 계열은 전 구간 80–99%를 유지한다(MAC PK-16K 98.4, MAG N-16K 98.6). 대조: Mamba-2는 PK-16K 5.4, W-8K/16K 0.0으로 붕괴 — erase는 있으나 얕은 state로는 부족하다; DeltaNet은 PK-16K 71.4까지 버티지만 N/W에서 무너진다 — replace는 해도 진짜 erase(forgetting)가 없다; TTT는 16K에서 처진다(PK-16K 88.4) — retention gate 부재 [Titans §5.3]. 즉 momentum+forgetting이 TTT를, deep nonlinear memory + erasure가 Mamba-2를, forgetting이 DeltaNet을 각각 이기게 만든 성분이라는 주장과 표의 패턴이 맞물린다.

**BABILong** [Titans Fig. 6]. few-shot 설정에서 Titans (MAC)는 Mamba-2.8B, RWKV-6-7B, RecurrentGemma-9B, Gemma-9B, Llama3.1-8B, GPT-4, GPT-4o-mini를 모두 이긴다 — 훨씬 적은 parameter로. fine-tuning 설정에서는 작은 MAC가 fine-tune된 RMT·Mamba, RAG를 단 Llama3.1-8B(약 70× 더 많은 parameter [Titans §5.4]), 그리고 GPT-4, Qwen2.5-72B, Llama3.1-70B를 넘어서며 2M tokens 너머까지 정확도를 유지한다. 이 라인의 ">2M context" 헤드라인은 전부 이 실험(MAC, fine-tuned, baseline 수치는 벤치마크 저자 보고)에 얹혀 있다.

**memory 깊이** [Titans Fig. 7, Fig. 8]. Pile 부분집합, 170M/360M/760M에서 $L_{\mathcal{M}}=1,2,3,4$ 비교: 깊을수록 전 길이에서 perplexity가 좋고 길이에 강건하며(효과는 작은 스케일에서 최대), 대가는 훈련 throughput의 선형 하락이다. 모든 깊이에서 tokens/sec은 sequence 길이에 대해 일정 — 즉 훈련 비용은 길이에 선형이다.

**효율** [Titans Fig. 9]. LMM은 Mamba-2/Gated DeltaNet보다 약간 느리다 — deep memory의 본질 비용에 더해 fused kernel 부재가 원인으로 지목된다. MAL이 전체에서 가장 빠른데, 이는 FlashAttention의 성숙도 덕이다 [Titans §5.8].

**ablation** [Titans Table 5]. base LMM: ppl 27.01 / reasoning 47.83 / long-context 92.68. 성분 제거 시 ppl: weight decay 제거 29.04(최악), momentum 제거 28.98, convolution 제거 28.73, deep→linear memory 28.49(long-context는 85.34로 급락), persistent memory 제거 27.63. attention을 붙이면 전부 개선: MAC 26.67/48.65/97.95, MAG 25.70/48.60/96.70, MAL 25.91/47.87/96.91. 기여 순위: weight decay > momentum > convolution > persistent memory [Titans §5.9].

> **[해설]** 원문은 §5.9에도 Table 5 캡션에도 이 ablation의 모델 규모를 명시하지 않는다. 다만 Table 5의 ppl 열은 Table 1의 Wiki·LMB perplexity 평균과 자릿수까지 일치하므로(예: LMM 400M — $(25.03+28.99)/2=27.01$; MAC $(25.61+27.73)/2=26.67$; 4행 모두 확인), ablation은 400M 스케일 수치로 읽는 것이 정합적이다.

**보조 도메인**: time series forecasting에서는 Simba 프레임워크의 Mamba 자리에 LMM을 넣어 ETT/ECL/Traffic/Weather에서 최고 MSE/MAE를 보고하고(예: ETTm1 0.358/0.387 vs PatchTST 0.387/0.400) [Titans Table 3], DNA modeling(GenomicsBenchmarks)에서는 HyenaDNA·Based·Mamba와 경쟁 수준이다(Enhancer Cohn 75.2 최고) [Titans Table 4].

**스케일의 정직한 자리매김.** 이 논문의 증거는 전부 ≤760M parameters / ≤30B tokens / 훈련 길이 4K에서 나왔다. v1 각주는 "더 큰 모델의 결과를 다음 버전에 싣겠다"고 약속하지만 [Titans §5], 이 라인 여섯 편 전체의 실증 상한은 끝내 1.3B params / 100B tokens에 머문다. 또한 위의 모든 효율 수치는 **훈련** throughput이다 — decode wall-clock 수치는 이 논문에도, 라인 여섯 편 어디에도 없다.

## 12.7 Systems/serving 함의

**state 수학: KV cache와의 교환.** sequence당 LMM layer 하나의 recurrent state = memory weights + momentum buffer = $2P_{\mathcal{M}}$개의 수다. 아래 비교가 이 논문의 존재 이유를 요약한다.

표 12-4 — state 크기: LMM vs KV cache (bf16, layer당·sequence당) **[해설]** — 수치는 이 책의 계산

| 항목 | 크기 식 | $d=1024$, 2-layer memory, $L=$ 2M tokens |
|---|---|---|
| LMM state ($W_t + S_t$) | $2P_{\mathcal{M}} \approx 2L_{\mathcal{M}}d^2$ | 약 4M 값 ≈ 8MB — **길이 불변** |
| attention KV cache | $2Ld$ | 약 4G 값 ≈ 8GB — 길이 비례 |

2M-token 문맥에서 attention의 KV cache는 layer당 GB 단위로 자라 사정권 밖이지만 LMM state는 그대로다 — BABILong의 >2M 능력은 정확히 이 교환 위에 서 있다. 대가도 명확하다: momentum buffer 때문에 state가 Gated DeltaNet식 matrix state의 2배이고, 그 buffer는 decode step 사이에 반드시 이월되어야 한다. MAC에서는 attention이 한 segment($C$ tokens) + $N_p$ persistent + 검색된 history token만 보므로 attention 쪽 KV cache가 segment 크기로 유계이고, MAG/MAL에서는 SWA window가 그 역할을 한다.

**decode의 재정의와 state 트래픽.** decode 한 step은 이제 read-only lookup이 아니라 $2P_{\mathcal{M}}$ 전체에 대한 read-modify-write다: $W_t$와 $S_t$를 읽고, elementwise로 갱신해, 도로 쓴다. token당 FLOPs가 $O(P_{\mathcal{M}})$ 상수(§12.4)인 동시에 token당 state 트래픽도 $O(P_{\mathcal{M}})$이므로, decode에서 이 layer의 arithmetic intensity는 낮은 상수에 고정된다 — KV cache 스트리밍이 지배하던 자리에 weight-state RMW 스트리밍이 들어선 것뿐이라는 점에서 독자에게 익숙한 memory-bound 그림이지만, 이제 그 트래픽이 문맥 길이와 무관하게 일정하다는 점이 다르다. 정량적 roofline 분석은 10장이 맡는다.

**병렬화와 kernel.** prefill은 훈련 시의 chunkwise 공식(식 (12-7)–(12-9))을 그대로 써서 prompt를 chunk 병렬로 흡수한다 — prefill = 큰 chunk의 병렬 write라는 Rosetta 대응이 문자 그대로 성립한다. MAC의 attention은 segment별 block-diagonal mask이므로 [Titans Fig. 3a] $C+N_p+N_l$ 크기 window들에 대한 FlashAttention이고 총비용 $O(L\cdot C)$, segment 간 의존은 memory의 순차 사슬뿐이다. kernel 현실: 발표 시점의 LMM에는 fused kernel이 없고(batched MLP forward + backward + scan + AXPY를 묶은 chunk kernel이 필요), 그런데도 Mamba-2 급 throughput의 사정권에 있다 [Titans Fig. 9]. MAL이 가장 빠른 것은 순전히 FlashAttention의 성숙도다 — 즉 "MAL vs MAC/MAG"는 오늘 측정되는 throughput과 품질의 교환이며, LMM kernel이 성숙하면 순위가 뒤집힐 수 있는 sociotechnical한 순위다.

**batching의 붕괴와 serving 신질문.** shared weights 전제가 깨진다: request마다 $W_t, S_t$가 다르므로 naive multi-tenant batching은 per-request weight 사본을 요구하고, batched decode는 shared-weight GEMM이 아니라 per-sample weights의 grouped GEMM/bmm workload가 된다 — TTT가 이미 강제한 패턴이지만 state가 MLP 전체 + momentum이라 더 무겁다. retention gate $\alpha_t$는 **학습된 eviction policy**다: paged-KV 같은 runtime 관리 기계는 사라지지만, 무엇이 언제 지워지는지가 runtime에 불투명해진다. state가 optimizer 궤적의 산물이므로 accumulation 순서·scan 정밀도 같은 numerics가 "cache" 자체를 표류시킬 수 있고 — KV cache에는 없던 결정성 문제 — speculative decoding의 rollback은 $(W_t,S_t)$ 궤적의 snapshot/복원이라는 비자명한 연산이 된다. test-time weight 변이는 multi-tenant 격리와 prompt privacy(문맥이 weights에 흡수된다) 질문도 연다. 논문은 이 중 무엇도 다루지 않는다.

**update frequency 관점의 예고.** 이 아키텍처에는 이미 세 개의 갱신 주기가 공존한다: 매 token 갱신되는 $W_t$(가장 빠름), window 안에서만 유효한 attention KV, 그리고 절대 갱신되지 않는 $P$와 $\Theta$(frequency 0). 어떤 상태를 얼마나 자주 갱신하며 memory 계층 어디에 둘 것인가라는 질문의 원형이 여기 있고, 이 삼분법을 연속체로 펴는 것이 16장(NL)의 일이다.

마지막으로 의무 caveat을 반복한다: **이 논문을 포함해 라인 전체에 decode wall-clock 수치가 없다.** 위 문단의 serving 논의는 구조에서 연역한 것이지 측정이 아니다.

## 12.8 한계와 bridge-out

논문 스스로 인정하거나 본문 검토에서 드러나는 한계를 모은다.

1. **point design이다.** 왜 $\ell_2$ regression인가, 왜 SGD+momentum+decay인가에 대한 논거가 없다. Appendix C가 기존 모델 전부를 특수 사례로 회수하는 순간, "그 축들 위의 다른 점은?"이라는 질문이 자동으로 생긴다. 논문도 더 나은 inner objective와 inner optimizer, memorization 특화 아키텍처(Memory Mosaics 등)를 명시적 future work로 남긴다 [Titans §3.1, App. C].
2. **미명시 세부가 많다.** 게이트 $\eta_t,\beta_t,\alpha_t$의 함수형·초기화, memory MLP 폭·activation, chunk 크기 $C$와 MAC segment 크기의 실험값, $W_{\mathrm{init}}$의 설정, chunk 경계에서 momentum buffer의 훈련/decode 시 처리 — v1에 없다. 학습된 optimizer 궤적이 수백만 step 동안 안정적이라는 보장도 없다($\|W_t\|$를 묶는 것은 학습된 decay뿐이다).
3. **근사의 비용이 미측정이다.** chunk 안 gradient는 stale하다. $C$에 대한 ablation이 없고, 훈련 시 chunk 단위 갱신과 decode 시 token 단위 갱신의 정합(train/serve 일치) 문제도 제기되지 않는다. staleness의 오차 한계는 이 논문은 물론 라인 여섯 편 어디에도 없다.
4. **Thm 4.1은 증명이 없다** (§12.3.8). 표현력 주장은 v1에서 단언이다.
5. **스케일과 kernel.** ≤760M/30B의 증거로는 momentum과 deep memory의 우위가 7B+·최적화된 DeltaNet 계열 kernel 앞에서도 유지되는지 알 수 없다. 약속된 "다음 버전"의 대형 모델과 코드 공개(PyTorch/JAX)는 v1 시점에 없다.
6. **outer 훈련의 memory 비용.** 4K 문맥의 unrolled inner loop를 backprop하려면 chunk별 inner state/activation을 저장해야 하는데, activation memory 회계나 gradient-checkpointing 레시피가 없다 — 문맥이나 $L_{\mathcal{M}}$을 키울 때 실질 병목이 될 수 있는 항목이다.

**Bridge-out → 13장 [Miras].** [Titans]가 남긴 가장 생산적인 유산은 결핍이 아니라 회수다. Appendix C가 "Gated DeltaNet = $\beta_t{=}0$, Longhorn = implicit step + no-forget, TTT = no-forget no-momentum, RWKV-7 = 같은 loss"를 보인 순간, 이 계보 전체가 하나의 설계 공간의 점들임이 드러났다 — 그런데 그 공간의 좌표축이 무엇인지는 [Titans]가 명명하지 않았다. 모두가 $\ell_2$(또는 dot-product) objective와 $\ell_2$ retention만 쓰고 있었다는 관찰, 그리고 "objective / retention / 아키텍처 / 알고리즘"이라는 4축의 명명이 [Miras]의 출발점이다. 그 과정에서 [Miras]의 출시 모델들은 [Titans]의 자랑인 momentum을 오히려 떼어내고(단순 GD로 회귀), optimizer 축은 비워 둔 채 후속작(Atlas)에 넘긴다 — 시조의 point design이 분해되어 좌표계가 되는 과정을 다음 장에서 본다.
