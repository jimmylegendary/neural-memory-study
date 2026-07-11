# ch03. Online learning: OGD, regret, FTRL

> **이 장의 목표** — 독자가 이 장을 마치면 다음을 할 수 있어야 한다.
> (1) token stream 위의 학습을 online protocol로 정식화하고, 각 단계를 autoregressive decode loop의 단계에 대응시킬 수 있다.
> (2) online gradient descent(OGD)의 update를 쓰고, 그 품질을 regret로 정량화하며, "sublinear regret"가 무엇을 보장하고 무엇을 보장하지 않는지 말할 수 있다.
> (3) Follow-the-Leader의 실패를 손계산으로 재현하고, FTRL의 regularizer가 왜 memory를 안정화하는지 설명할 수 있다.
> (4) FTRL의 두 항(loss 합 + regularizer)이 [Miras]의 attentional bias와 retention gate 자리임을 지적하고, mirror descent가 Memora형 softmax update의 원형임을 알아볼 수 있다.
>
> **왜 필요한가** — [Miras] (*It's All Connected*, arXiv:2504.13173; 이 책은 framework 이름 Miras로 통칭한다)의 §3 전체가 이 장의 어휘로 쓰여 있다: fixed-state sequence model의 gradient-based per-token write를 online gradient descent 한 step으로 정식화하고 [Miras Eq. 5], 이를 (FTRL Viewpoint)와 (Learning-Retaining Viewpoint)라는 두 online-optimization 렌즈로 해석한다 [Miras §3.2–3.3]. [Atlas] (*Atlas: Learning to Optimally Memorize the Context at Test Time*, arXiv:2505.23735)의 문제의식 자체가 기존 recurrent model의 "online nature" 비판이고, Omega rule(→ 14장)은 online objective를 window objective로 바꾸는 제안이다 [Atlas §3.2]. 6장에서 만날 Longhorn은 SSM update를 online learning 문제의 닫힌 해로 유도하고(Liu et al. 2025, arXiv:2407.14207), [Titans] (*Titans: Learning to Memorize at Test Time*, arXiv:2501.00663)의 per-token memory update도 online GD의 재해석이다 [Miras §3.1]. [NL] (*Nested Learning: The Illusion of Deep Learning Architecture*, arXiv:2512.24695)의 level별 objective 역시 각 level이 자기 주기의 stream 위에서 푸는 online 문제다. 요컨대 이 장은 Part II 전체가 딛고 설 "stream 위의 최적화" 언어를 공급한다.

## 3.1 Online protocol: 독자가 이미 돌리고 있는 loop

2장은 훈련을 (state, update, cost)를 갖는 객체로 보는 법을 가르쳤다. 거기서 데이터 소비는 batch training(dataset 전체를 미리 두고 매 step i.i.d. mini-batch를 뽑아 여러 epoch 재방문)이었다. 이 장은 그 전제를 버린다.

**online learning**은 데이터가 스트림으로 한 번씩만 도착하는 설정에서의 학습이다. 분포 가정도, epoch도, 재방문도 없다. 형식적으로 **online protocol**은 매 step $t=1,\dots,L$에서 다음 네 단계를 반복한다:

1. 입력을 받는다 (이 라인에서는 key/value 쌍 $(k_t, v_t)$).
2. 현재 state $W_{t-1}$로 예측한다 (memory 읽기 $\mathcal{M}(k_t; W_{t-1})$).
3. loss $\ell_t(W_{t-1})$를 관측한다 ("이 예측이 얼마나 틀렸나").
4. state를 갱신한다: $W_{t-1} \to W_t$.

이 loop를 독자는 이미 매일 돌리고 있다 — autoregressive decode가 정확히 이 구조다. 차이는 단 하나, 4단계의 내용물이다: 지금까지 inference에서 "state 갱신"은 KV cache append(학습 없는, 계산 없는 write)였고, 이 라인에서는 그것이 gradient step이 된다. protocol의 골격은 바뀌지 않는다.

표 3-1 — online protocol과 decode loop의 단계별 대응

| online protocol | autoregressive decode (독자의 세계) | TTT-line layer (이 라인) |
|---|---|---|
| 입력 수신 | 다음 token 도착 | $(k_t, v_t)$ 투영 |
| 예측 | forward pass, logits | memory 읽기 $y_t = \mathcal{M}(q_t; W_t)$ |
| loss 관측 | (없음 — 정답을 모름) | inner loss $\ell(W_{t-1}; k_t, v_t)$ — 정답은 $v_t$ 자신 |
| state 갱신 | KV cache append | gradient step $W_{t-1} \to W_t$ |

표의 3행이 가장 낯설 것이다 — inference에는 "정답"이 없는데 무슨 loss를 관측하는가? 이 라인의 inner loss는 외부 정답이 필요 없는 self-supervised 회귀다: "key $k_t$를 넣으면 value $v_t$가 나와야 한다"는 조건 자체가 loss가 된다 — $\ell(W; k_t, v_t) = \|\mathcal{M}(k_t; W) - v_t\|_2^2$. stream의 각 항목이 자기 label을 들고 도착하는 셈이다(기원은 5장 associative memory).

분포 가정이 없다는 점은 장식이 아니다. online learning 이론은 stream이 **적대적**(adversarial)이어도 깨지지 않는 성능 하한 — worst-case 누적 손해 — 을 추구한다(서빙을 평균 부하가 아니라 tail latency로 설계하는 감각이다). 실제 token stream은 i.i.d.가 아니므로(주제 전환, 코드 블록, 두 번 다시 안 나오는 needle), 이 보장은 비정상성(non-stationarity)에 대한 보험이 된다.

2장의 경고를 다시 새긴다: **이 라인에서 훈련의 batch 축 역할은 sequence 축이 대신한다.** protocol의 $t$는 mini-batch가 아니라 token 인덱스이고, inner loop는 문맥의 token들을 두 번 다시 지나가지 않는 single-pass 학습이다 — online learning이 그 자연 언어다.

한 가지 예고. 이 장의 모든 식은 "한 token에 한 step"의 순차 이상형이다. state 갱신이 gradient step이 되는 순간 $W_t$가 $W_{t-1}$에 의존하는 sequential chain이 생겨 prefill식 병렬화가 공짜가 아니게 되고, 실전 구현은 전부 이 이상형의 chunk 단위 근사다 — 그 chain을 GEMM으로 펴는 문제는 9장이 담당한다.

## 3.2 OGD와 regret: stream 압축의 품질 언어

online protocol의 4단계를 채우는 가장 단순한 방법은 2장의 SGD를 그대로 이식하는 것이다. **online gradient descent(OGD)**는 매 step 현재 항목의 loss에 대한 gradient로 한 걸음 내려간다(Zinkevich 2003, ICML):

$$
W_t = W_{t-1} - \eta_t \, \nabla_W \ell_t(W_{t-1})
\tag{3-1}
$$

대수적으로는 2장의 SGD와 같은 식이고, 다른 것은 데이터 출처다: mini-batch를 i.i.d.로 뽑는 대신 stream이 주는 순서대로 각 항목을 정확히 한 번 소비한다. $\ell_t(W) := \ell(W; k_t, v_t)$로 두면 식 (3-1)은 표준형 (M1)(delta rule이자 TTT-Linear의 write)과 글자까지 같아진다 — [Miras]의 출발점이다: [Miras Eq. 5]는 fixed-state sequence model의 기본 memory update를 이 식으로 놓고, [Titans]의 momentary surprise(→ 12장)가 이 online GD gradient의 재해석임을 명시한다 [Miras §3.1]. 즉 **OGD는 (M1)의 최적화-이론 쪽 이름이다.**

이 알고리즘이 "잘한다"는 것을 어떻게 정량화하는가? single-pass stream에는 held-out set이 없다. online learning의 답이 **regret**다: horizon $L$까지의 누적 loss를, 사후에(in hindsight) 고를 수 있는 최선의 고정 state와 비교한 상대 손해로 정의한다.

$$
\mathrm{Reg}_L \;=\; \sum_{t=1}^{L} \ell_t(W_t) \;-\; \min_{W^\star} \sum_{t=1}^{L} \ell_t(W^\star)
\tag{3-2}
$$

비교 대상 $W^\star$를 **comparator**라 부른다: stream 전체를 다 보고 단 하나 고를 수 있는 최적의 고정 memory 상태다. 식 (3-2)는 세 겹으로 읽는다. 첫째, regret는 절대 성능이 아니라 상대 손해다 — stream 자체가 압축 불가능하면 comparator도 손해를 보고 regret는 그 차이만 잰다. 둘째, comparator는 고정이다 — "그때그때 바뀌는 oracle"이 아니라 "최선의 단일 압축 상태"와 비교한다. 셋째, 목표는 **sublinear regret**($\mathrm{Reg}_L / L \to 0$), 즉 step당 평균 손해가 comparator 대비 0으로 수렴하는 것 — 이때 "online learner가 최선의 고정 state를 따라잡는다"고 말한다.

핵심 정리: $\ell_t$가 convex이고 gradient norm이 $G$, 탐색 영역 지름이 $D$로 유계이면, iterate를 지름 $D$의 볼록 영역 $\mathcal{W}$로 projection하는 $\eta_t \propto 1/\sqrt{t}$ OGD는 $\mathrm{Reg}_L = O(GD\sqrt{L})$를 달성한다(Zinkevich 2003; 교과서적 정리는 Shalev-Shwartz 2011, *Online Learning and Online Convex Optimization*; Hazan 2019, arXiv:1909.05207). 지름 $D$는 이 projection을 통해서만 bound에 들어온다 — 무제약 memory-layer 식 (3-1)은 그 projection이 없는 형태이므로, 보장이 아니라 knob의 정체를 빌려 오는 것이다. $\sqrt{L}$은 sublinear이라 평균 regret는 $O(1/\sqrt{L})$로 사라진다. 직관만 취하면: $1/\sqrt{t}$ 감쇠는 "초반에 크게 적응, 후반에 안정화"라는 스케줄이고, 적응(plasticity)과 안정(stability)의 교환이 $\sqrt{L}$에서 균형을 이룬다.

step size 스케줄에서 고전 이론과 이 라인의 온도차가 보인다. 고전의 $\eta_t \propto 1/\sqrt{t}$는 worst-case regret를 겨냥한 감쇠 스케줄이라, memory layer에 그대로 이식하면 "문맥이 길수록 새 정보를 덜 쓰는" 층 — long context에서 원하는 행동의 정반대 — 이 된다. 이 라인은 스케줄을 폐기한다: $\eta_t = \eta(x_t; \Theta)$로 step size를 token의 함수로 만들고, 그 함수를 slow weights가 outer loop에서 학습한다(→ 4장). 그러면 $\eta_t$는 수렴용 감쇠 계수가 아니라 per-token write 강도 신호("중요한 token은 세게, 뻔한 token은 흘려보내라")가 된다. regret 이론이 주는 것은 보장이 아니라 knob의 정체(step size = write 강도)이고, 이것이 13장이 $\eta_t$를 "meta in-context learning rate"라 부르는 이유다.

> **[해설]** inference 어휘로: $d_v \times d_k$ matrix memory는 KV cache의 고정 크기 손실 압축이고(→ 1장 Rosetta), regret bound는 그 압축의 품질 보증서다 — 이 $O(d^2)$ state는 stream을 다 보고 고른 최선의 $O(d^2)$ 압축 상태보다 평균적으로 뒤지지 않는다. 단, 보장의 단위가 개별 조회가 아니라 stream 전체 평균이라 특정 needle 하나의 회수는 보장하지 않는다(§3.7에서 다시).

정직하게 한계도 적는다. 첫째, comparator가 고정이므로 분포가 도중에 바뀌는 stream(문서 경계, 주제 전환)에서는 "최선의 고정 state" 자체가 약한 기준이다(comparator 이동을 허용하는 dynamic regret 확장은 존재만 언급한다). 둘째, 보장은 convexity를 요구하는데, 이 라인의 실제 memory는 2-layer MLP(표준 deep memory)라 inner loss가 비볼록이므로 $O(\sqrt{L})$ 보장은 성립하지 않는다. 실제로 [Miras]는 FTRL·mirror descent라는 online convex optimization의 기계를 설계 언어로 수입하되, 자신의 변형(Moneta/Yaad/Memora)에 대한 regret bound는 제시하지 않는다 — 정당화는 전적으로 실험이다. 이 장이 가르치는 것은 보장이 아니라 **설계 어휘**다: 이 라인의 모든 update rule은 "어떤 online 문제를 어떤 online 알고리즘으로 푸는가"의 답으로 읽을 수 있다.

## 3.3 FTL: 왜 "지금까지의 최적해"로 점프하면 안 되는가

OGD는 gradient 한 걸음이라는 최소한의 갱신이다. 반대편 극단은 매 step, 지금까지 본 모든 loss의 최적해로 점프하는 것이다. 이를 **Follow-the-Leader(FTL)**라 부른다:

$$
W_t \;=\; \arg\min_{W} \sum_{i=1}^{t-1} \ell_i(W)
\tag{3-3}
$$

겉보기에 FTL은 이상적이다 — 매 순간 이력 전체에 최선이다. 문제는 두 겹이다. 첫째는 비용: 매 token마다 과거 전체를 다시 최적화하는 것은 decode마다 KV cache 전체를 재스캔하는 것과 같은 낭비다. 둘째가 치명적이다 — FTL은 **불안정**하다. loss가 평평한(linear) 모양이면 마지막 항목 하나가 argmin을 탐색 영역의 반대편 끝으로 던지고, 적대적 stream은 이를 이용해 learner를 매 step 진동시켜 regret를 $\Theta(L)$(선형, 즉 학습 실패)로 만든다 — §3.8에서 여섯 step 손계산으로 재현한다. 한 문장으로: FTL의 state는 마지막 token에 과민하다.

비용도 GEMM shape로 읽어 둘 가치가 있다: 식 (3-3)의 argmin은 이력 전체의 least-squares 해라 매 step Gram 역행렬 적용이 드는, OGD의 rank-1 write와 자릿수가 다른 per-token 비용이다(Gram 비정칙 가능성이 다음 절 regularizer의 복선이고, 그 실제 입주자가 6장의 Mesa-layer다).

> **[해설]** 이 라인의 족보에는 FTL의 "전체 이력 최적해"를 유지하는 극한들이 있다: softmax attention은 전체 이력의 $\ell_2$ regression을 non-parametric하게 정확히 푸는 해(상태 = KV cache 그 자체, §1.6 카탈로그)이고, Mesa-layer(→ 6장)는 같은 objective를 Newton법으로 푼다 — 대가는 자라는 cache와 step당 최적화 비용이다. recurrent model들은 반대편 끝(OGD)에서 출발해 두 극단 사이를 설계한다 — [Miras]가 "memory learning algorithm"을 독립 설계 축으로 선언한 그 스펙트럼이다.

## 3.4 FTRL: regularizer가 memory를 안정화한다

FTL의 진동을 고치는 고전적 처방은 argmin 안에 **regularizer**를 더하는 것이다. **Follow-the-Regularized-Leader(FTRL)**는 다음을 푼다:

$$
W_t \;=\; \arg\min_{W} \; \sum_{i=1}^{t-1} \ell_i(W) \;+\; \frac{1}{\eta}\, R(W)
\tag{3-4}
$$

$R(W)$(전형적으로 $\frac{1}{2}\|W\|_2^2$)는 두 가지 일을 한다: 연속된 해 $W_{t-1}, W_t$ 사이 거리를 $\eta$ 스케일로 묶어 마지막 항목의 지배력을 없애고(안정화), state 크기를 벌점화해 memory가 무한정 자라는 것을 막는다. 적절한 $\eta$ 아래 FTRL은 convex 설정에서 다시 $O(\sqrt{L})$ regret를 회복한다(Shalev-Shwartz 2011; McMahan 2011, AISTATS; 현대적 통합 정리는 Orabona 2019, *A Modern Introduction to Online Learning*, arXiv:1912.13213).

FTRL이 이 책에서 각별한 이유는 다음 계산에 있다. loss를 각 step의 gradient로 선형화하고 — $\hat\ell_i(W) = \langle g_i, W \rangle$, $g_i = \nabla_W \ell_i(W_{i-1})$ — $R = \frac{1}{2}\|W\|_2^2$를 넣으면 argmin이 닫힌 형태로 풀린다:

$$
W_t \;=\; \arg\min_W \Big\langle \sum_{i=1}^{t-1} g_i,\, W \Big\rangle + \frac{1}{2\eta}\|W\|_2^2
\;=\; -\eta \sum_{i=1}^{t-1} g_i
\;\;\Longrightarrow\;\;
W_t = W_{t-1} - \eta\, g_{t-1}
\tag{3-5}
$$

즉 **linearized loss + $\ell_2$ regularizer의 FTRL은 정확히 OGD다**(제약이 없고 $W_0 = 0$일 때). [Miras Eq. 7]이 명시하는 이 동치는 두 방향으로 읽는다. OGD는 FTRL의 특수경우이므로 두 슬롯(loss 모양, regularizer 모양)을 갈아 끼우면 OGD를 일반화하는 update rule의 공장이 생기고, 역으로 FTRL의 state는 본질적으로 **gradient 누적기**다 — 식 (3-5)의 중간 형태 $W_t = -\eta\sum g_i$가 이를 노출한다. 이 둘째 독법은 예고편이다: 13장 Moneta는 raw 누적기 $A_t$와 정규화된 노출 상태 $W_t$를 분리하는 FTRL형(dual accumulator) update를 쓴다.

이제 [Miras §3.2]의 (FTRL Viewpoint)를 통일 표기로 쓰면 이 장과의 관계가 즉시 보인다:

$$
W_t \;=\; \arg\min_{W \in \mathcal{W}} \; \sum_{i=1}^{t} \hat\ell_i(W; k_i, v_i) \;+\; \frac{1}{\eta_t}\, R_t(W)
$$

첫 항 — 과거 모든 (key, value) 쌍을 얼마나 잘 기억하는가 — 이 [Miras]가 attentional bias(→ 13장)라 이름 붙이는 자리이고, 둘째 항 — memory의 크기와 갱신을 다스리는 벌점 — 이 retention gate(→ 13장)의 자리다. [Miras §3.3]은 같은 update를 다른 렌즈 — 최신 쌍 하나만 fit하되 이전 state $W_{t-1}$ 근처에 머무는 (Learning-Retaining Viewpoint) — 로도 쓰고, 상수 $\eta$·제약 없는 $\mathcal{W} = \mathbb{R}^d$·누적 objective의 strict convexity 아래 두 렌즈의 동치를 증명한다 [Miras Prop. 3.2, App. B]. 주의하라 — 출하되는 구성(deep memory, data-dependent gate, 제약된 state)에서는 세 조건 모두 깨지므로, 이 동치는 설계 동기이지 배포물의 보장이 아니다. 두 viewpoint의 본격적 사용법은 13장이 소유하고, 이 장은 그 둘이 online learning의 표준 기계(FTRL, proximal step)임을 못박는다.

Learning-Retaining 렌즈의 최소형 — proximal step — 도 봐 둘 가치가 있다:

$$
W_t \;=\; \arg\min_W \; \ell_t(W) + \frac{1}{2\eta_t}\|W - W_{t-1}\|_2^2
$$

여기서 $\ell_t$를 선형화하면 explicit gradient step, 즉 식 (3-1)이 그대로 나온다. 선형화하지 않고 정확히 풀면 **implicit GD**(proximal step)가 되는데, $\ell_2$ regression loss에서는 이것도 닫힌 해 — 유효 step size가 $\eta_t/(1 + \eta_t k_t^\top k_t)$로 자동 감쇠하는 delta rule 형태 — 를 갖는다. Longhorn(Liu et al. 2025, arXiv:2407.14207, *State Space Models are Amortized Online Learners*)이 정확히 이 자리의 모델이고 유도는 6장에서 한다 — "SSM 하나가 online learning 문제의 닫힌 해"라는 문장이 이 장 뒤에는 마케팅이 아니라 명세로 들려야 한다.

마지막으로 반대 방향의 확장. [Atlas §3.2, Eq. 6]은 기존 recurrent model 전부를 "현재 token 하나의 loss + retention"을 푸는 online 문제로 요약한 뒤, 이 **online nature**(매 step 현재 항목만 greedy하게 최적화)를 sub-optimal memorization의 원인으로 지목한다. Omega rule(→ 14장)은 loss 항을 최근 $c$개 token의 window 합으로 바꾼다: $c=1$이면 online(delta rule), $c$가 문맥 전체면 global 최적화다 [Atlas §3.2]. FTRL의 "모든 과거 항"과 OGD의 "현재 항 하나" 사이를 window $c$가 매개한다 — online learning 스펙트럼 위에서 여섯 논문이 서로 다른 점을 고르는 그림이 이렇게 완성된다.

## 3.5 Mirror descent와 Bregman divergence

FTRL의 regularizer 슬롯에 $\|W\|_2^2$를 넣는 것은 state 공간이 유클리드라는 암묵적 가정이다. state에 구조가 있으면(성분이 전부 양수여야 한다든가, 합이 고정이라든가) "가깝다"의 척도부터 바꾸는 것이 자연스럽다 — 그 일반화의 도구가 **Bregman divergence**다. strictly convex한 함수 $F$에 대해

$$
D_F(W, W') \;=\; F(W) - F(W') - \langle \nabla F(W'),\, W - W' \rangle
\tag{3-6}
$$

즉 "$W'$에서의 1차 근사가 $W$에서 실제 $F$ 값을 얼마나 밑도는가"이다. $F = \frac{1}{2}\|\cdot\|_2^2$이면 $D_F$는 squared Euclidean distance로 환원되고, $F(W) = \sum_j W_j \log W_j$ (negative entropy, 확률 simplex 위)이면 $D_F$는 KL divergence가 된다. **mirror descent**는 §3.4의 proximal step에서 Euclidean 벌점을 $D_F$로 교체한 것이다:

$$
W_t \;=\; \arg\min_W \; \langle g_t, W \rangle + \frac{1}{\eta}\, D_F(W, W_{t-1})
$$

$F$가 entropy이고 state가 simplex 위이면 이 argmin은 곱셈형 닫힌 해를 낳는다:

$$
W_{t,j} \;=\; \frac{W_{t-1,j}\, e^{-\eta g_{t,j}}}{\sum_{j'} W_{t-1,j'}\, e^{-\eta g_{t,j'}}}
$$

— exponentiated gradient(= multiplicative weights) update다(FTRL과 mirror descent의 관계는 McMahan 2011). 성질을 눈여겨보라: state 성분은 항상 양수이고 매 step 재정규화되어 총합이 보존되므로, forgetting이 곱셈 감쇠(decay)가 아니라 **성분들 사이의 질량 경쟁**으로 일어난다 — 어떤 성분이 커지려면 다른 성분이 줄어야 하고, state는 문맥이 아무리 길어져도 원리적으로 발산할 수 없다.

이 기하가 장식이 아니라 실전 선택지인 이유는 제약에 있다. [Miras §5.2]는 수치 불안정(state 값 폭주)을 막으려 state를 스케일된 probability simplex(성분 비음수, 합 고정) 안에 가두는 retention 변형을 제안한다. Euclidean 세계에서 이 제약은 매 step 별도 projection을 요구하지만, entropy 기하에서는 simplex가 update의 자연 서식지라 곱셈형 update와 재정규화가 제약을 부수 비용 없이 유지한다. 어느 기하에서 update하느냐가 어느 제약을 공짜로 얻느냐를 정한다 — regularizer 선택이 곧 기능(안정성 보장) 선택이라는 FTRL 슬롯 관점의 사례다.

이 기계의 행선지는 13장이다. Memora의 KL-retention update $W_t = \mathrm{Softmax}(\alpha_t \log W_{t-1} - \eta_t \nabla_W \ell)$는 위 exponentiated-gradient 식에 retention gate를 결합한 것이고, [Miras §5.2]의 f-divergence retention 일반화도 같은 틀($D_F$ 자리에 다른 divergence)이다. mirror descent를 여기서 만나 두면 13장의 "memory update에 웬 softmax?"라는 당혹이 "entropy regularizer구나"로 바뀐다. systems 접점: 유계·정규화된 state는 발산 걱정 없는 memory 설계의 원리적 근거다(수치 표현 함의는 13장 systems 절).

## 3.6 Loss geometry: $\ell_2$, $\ell_1$, $\ell_p$, Huber

FTRL의 두 슬롯 중 regularizer는 §3.4–3.5가 다뤘다. 남은 슬롯은 loss의 모양이다. residual $r = \mathcal{M}(k_t; W) - v_t$가 어떤 함수로 들어가는가에 따라 같은 OGD라도 "어떤 token이 state를 세게 흔드는가"가 달라진다 — 1차원 residual로 축소해 gradient 크기를 보는 것이 빠르다.

<!-- FIG: ch03/fig-01-loss-geometries -->

표 3-2 — loss 모양별 gradient 크기 $|d\,\mathrm{loss}/dr|$ (스칼라 residual $r$, Huber threshold $\delta = 1$)

| loss | gradient 크기 식 | $r=0.1$ | $r=1$ | $r=3$ |
|---|---|---|---|---|
| $\ell_2$: $r^2$ | $2\lvert r\rvert$ | 0.2 | 2 | 6 |
| $\ell_1$: $\lvert r\rvert$ | $1$ | 1 | 1 | 1 |
| $\ell_3$: $\lvert r\rvert^3$ | $3r^2$ | 0.03 | 3 | 27 |
| Huber($\delta{=}1$) | $\lvert r\rvert$ (내부), $\delta$ (외부) | 0.1 | 1 | 1 |

읽는 법: $\ell_2$는 오차에 비례해 쓴다 — 크게 벗어난 token 하나(오타, 노이즈, 적대적 스팬)가 state를 그 크기만큼 흔든다. $\ell_1$은 방향만 쓰고 크기를 버린다 — "이 key가 있었다"만 기록하는 극단으로, [Miras]는 $p=1$을 value-less associative memory라 부른다(→ 13장). $p>2$는 반대로 큰 오차를 증폭한다 — surprising token일수록 더 세게 기록한다. Huber는 threshold $\delta$ 안에서 $\ell_2$, 밖에서 $\ell_1$이라 gradient 크기가 $\delta$에서 포화한다 — 정확히 **per-token gradient clipping**(rate limiter·saturating counter의 직관)이다.

matrix memory 일반형도 적어 둔다. residual $r_t = W_{t-1}k_t - v_t$에 대해 $\ell_p$ bias의 gradient step은 $W_t = W_{t-1} - p\,\eta_t \big(\mathrm{sign}(r_t) \odot |r_t|^{p-1}\big)\, k_t^\top$ 꼴이다 [Miras Eq. 11] — residual에 elementwise 비선형(sign, 성분별 거듭제곱)을 먹인 뒤에도 여전히 rank-1 outer product로 쓴다(loss 모양을 바꿔도 GEMM 골격은 보존된다 — §3.7). 실전 주의 하나: sign·절댓값의 미분 불가능점 때문에 이 update를 **통과하는** outer loop backprop(→ 4장)이 불안정해질 수 있어, [Miras Remark 5]는 $\tanh$와 $\sqrt{x^2+\epsilon}$ 근사로 이를 매끄럽게 만든다 — inner update의 모양이 outer 훈련의 안정성 제약을 받는, 두 loop가 서로를 구속하는 첫 사례다.

이 축의 제품이 13장의 Moneta($\ell_p$ bias, 매끄럽게 근사된 sign·절댓값)와 Yaad(Huber bias, token마다 학습된 threshold $\delta_t$)다. 이 장은 모양의 어휘만 공급한다 — 어떤 모양이 이기는가는 [Miras] 실험이 답하고, 수치·ablation은 13장이 다룬다.

## 3.7 Systems bridge: 학습된 eviction policy로서의 retention

이 장의 개념들을 독자의 production 어휘로 되감는다.

**state는 cache이고, retention은 eviction이다.** 독자가 아는 KV cache eviction은 명시적 정책이다 — sliding window로 자르고 quota로 총량을 막고 score로 골라 버린다. fast-weight memory에는 그런 정책 코드가 없다: §3.4의 regularizer $R$(retention 항)이 그 역할을 하되, 무엇이 얼마나 오래 살아남는가가 discrete한 규칙이 아니라 **벌점 항 선택으로 결정된다**(Rosetta의 "retention gate = 학습된 eviction"이 이 뜻이다). 벌점의 모양이 eviction의 성격을 정한다: $\ell_2$는 모든 성분을 조금씩 깎는 곱셈 감쇠(soft), $\ell_1$은 작은 항목을 정확히 0으로 자르는 절삭(hard), KL은 질량 경쟁(§3.5) — 세 형태 모두 13장에서 실제 모델로 만나고, 정책을 사람이 튜닝하는 대신 gate를 만드는 slow weights가 outer loop에서 학습한다(→ 4장).

**regret와 recall@position은 같은 것의 두 투영이다.** needle-in-a-haystack recall 곡선은 pointwise 측정(특정 (key, value) 쌍이 $N$ token 뒤에도 회수되는가)이고, regret는 aggregate 보장(stream 전체 누적 loss가 comparator 대비 유계인가)이다. 압축 state가 needle을 못 꺼낸다는 것은 그 쌍의 절대 $\ell_t$가 크다는 뜻이고, comparator가 그 쌍을 회수하는 한 이는 regret의 양의 국소 성분이 된다(comparator도 같은 needle을 놓치면 국소 기여는 0 이하일 수 있다). 그러나 sublinear regret는 평균의 보장이지 개별 needle의 보장이 아니다 — "regret가 낮은데 recall이 나쁜" 모델은 aggregate 최적화가 pointwise 회수를 함의하지 않는다는 사실의 전시일 뿐 모순이 아니다. retention 품질이 long-context 성능을 지배한다는 [Miras]의 주장(→ 13장)은 이 구분 위에서 읽어야 한다.

**고정 comparator의 사각지대는 non-stationarity다.** 실서비스 문맥은 system prompt, 검색 결과, 대화 이력이 한 stream에 이어 붙는 "서로 다른 문서들의 연결"이라, 최선의 단일 압축 상태가 모든 구간의 평균이라는 애매한 대상이 되고 고정 comparator 기준의 낮은 regret조차 위안이 못 된다. 실전 응답은 두 갈래다: 경계에서 state를 통째로 비우는 data-dependent retention gate $\alpha_t$(→ 12–13장), 그리고 주기적으로 state를 리셋해 sequential chain을 끊는 설계(→ 15장).

**비용의 GEMM shape.** matrix memory $W \in \mathbb{R}^{d_v \times d_k}$와 $\ell_2$ loss에서 OGD 한 step은 두 조각이다: 읽기 $Wk_t$ (GEMV), 쓰기 $\nabla_W \ell = (Wk_t - v_t)\,k_t^\top$ (rank-1 outer product) — 2장의 $dW = (\text{오차})\,k^\top$ 그 shape다. per-token decode는 state 전체의 read-modify-write이라 bandwidth-bound이고, token $C$개를 chunk로 묶으면 rank-$C$ GEMM이 되어 tensor core로 이동한다 — 그 변환이 9장의 전부다. 변형들의 추가 비용도 shape로 읽힌다: FTRL형 dual accumulator는 상주 state를 두 배로 만들고, loss·retention 교체는 GEMM 본체를 건드리지 않는 값싼 epilogue다 — sign·절댓값·threshold 마스크는 elementwise이고, softmax/simplex 재정규화는 축 방향 reduction을 하나 더 붙일 뿐 GEMM 골격은 그대로다 — 13–14장의 변형들이 전부 같은 chunkwise 기계 위에서 훈련되는 이유다.

## 3.8 Worked micro-example: FTL의 진동, FTRL의 안정화, OGD와의 동치

수치로 확인하자. 무대는 가장 작은 memory다: 스칼라 state $w \in [-1, 1]$ (1×1 matrix memory), loss는 linear $\ell_t(w) = z_t\, w$. linear loss는 장난감이 아니다 — [Miras] 분류에서 Hebbian family(linear attention, RetNet, Mamba-2, GLA; → 6장, 13장)의 attentional bias가 정확히 linear($\tilde\ell_t = -2\langle Wk_t, v_t\rangle$)라, $k_t = 1$로 두면 이 예제는 그 family의 1×1 버전이다($z_t = -2v_t$).

adversarial stream을 $z_1 = 0.5$, 이후 $z_t = -1, +1, -1, +1, -1$ (교대)로 잡고 $L = 6$까지 돌린다. FTL은 식 (3-3)에 따라 $w_t = \arg\min_{w \in [-1,1]} \big(\sum_{i<t} z_i\big) w$, 즉 누적합의 반대쪽 끝점 $-\mathrm{sign}(\sum_{i<t} z_i)$을 고른다(이력이 없는 $t=1$은 $w_1 = 0$).

표 3-3 — FTL의 여섯 step (누적합은 step 시작 시점 기준)

| $t$ | $\sum_{i<t} z_i$ | $w_t$ (FTL) | $z_t$ | loss $z_t w_t$ |
|---|---|---|---|---|
| 1 | 0 | 0 | 0.5 | 0 |
| 2 | 0.5 | $-1$ | $-1$ | $+1$ |
| 3 | $-0.5$ | $+1$ | $+1$ | $+1$ |
| 4 | 0.5 | $-1$ | $-1$ | $+1$ |
| 5 | $-0.5$ | $+1$ | $+1$ | $+1$ |
| 6 | 0.5 | $-1$ | $-1$ | $+1$ |

누적 loss는 $5$다. comparator는 $\sum_{t=1}^{6} z_t = -0.5$이므로 최선의 고정 $w^\star = +1$, 그 누적 loss는 $-0.5$다. 따라서 $\mathrm{Reg}_6 = 5 - (-0.5) = 5.5$ — step당 거의 1씩 horizon에 비례해 자란다. 메커니즘이 표에 그대로 보인다: 마지막 항목이 누적합의 부호를 뒤집을 때마다 FTL은 도메인의 반대편 끝으로 점프하고, 적대적 stream은 그 점프 직후의 방향을 정확히 벌준다 — state가 마지막 token에 과민하다는 §3.3의 진단이 이 진동이다.

이제 FTRL이다. $R(w) = \frac{1}{2}w^2$, $\eta = 0.5$로 식 (3-4)를 풀면 $w_t = -\eta \sum_{i<t} z_i$ (모두 $[-1,1]$ 안이므로 제약은 발동하지 않는다). 같은 stream에서 궤적은 표 없이도 한 줄로 적힌다: $w_1 = 0$, 이후 누적합이 $+0.5$와 $-0.5$를 오가므로 $w_t$는 $\mp 0.25$로만 진동한다. loss는 $t=1$에 $0$, 이후 매 step $+0.25$ — 누적 loss $1.25$, $\mathrm{Reg}_6 = 1.25 - (-0.5) = 1.75$. 진동 진폭이 $1$에서 $0.25$로 줄었고 그것을 정하는 것이 $\eta$다 — $1/\eta$가 클수록 state는 덜 움직이고 덜 배운다. plasticity와 stability의 교환이 숫자 하나로 손에 잡힌다: [Miras]가 $\eta_t$를 "meta in-context learning rate"라 부르며 "크면 더 배우고 더 잊는다"고 한 것 [Miras §3.3]이 이 knob의 이름이다.

끝으로 식 (3-5)의 동치를 이 숫자로 검산한다. linear loss의 gradient는 $\nabla \ell_t(w) = z_t$이므로 OGD는 $w_t = w_{t-1} - \eta z_{t-1}$이다: $w_2 = 0 - 0.5(0.5) = -0.25$, $w_3 = -0.25 - 0.5(-1) = +0.25$, $w_4 = 0.25 - 0.5(1) = -0.25$ — 위 FTRL 궤적과 자리마다 일치한다. "과거 전체의 regularized argmin"과 "gradient 한 걸음"이 같은 궤적이라는 것, 이것이 이 장의 중심 동치다(제약이 발동하는 설정에서는 FTRL과 OGD가 갈라질 수 있고, 그 지도는 McMahan 2011이다).

Hebbian 독법으로 마무리한다: $z_t = -2v_t$로 두면 위 OGD는 $w_t = w_{t-1} + 2\eta\, v_{t-1}$, 즉 순수 가산 write — 방금 돌린 여섯 step이 linear attention의 1×1 decode 궤적이었던 셈이다.

## 요약

- online learning은 stream 위에서 "수신 → 예측 → loss 관측 → state 갱신"을 반복하는 protocol로, autoregressive decode와 loop 모양이 같다 — 새 요소는 state 갱신이 gradient step이 된다는 것 하나다.
- OGD $W_t = W_{t-1} - \eta_t \nabla_W \ell_t(W_{t-1})$는 표준형 (M1)의 최적화-이론 이름이고, fixed-state sequence model의 gradient-based per-token write는 online GD 한 step으로 읽힌다 [Miras Eq. 5] — Newton(Mesa)·non-parametric(attention)·implicit(Longhorn) 해는 이 memory-learning-algorithm 축의 다른 점들이다(§3.3).
- regret는 stream 압축 품질의 언어다: 사후 최선의 고정 comparator $W^\star$ 대비 누적 손해. convex·유계 설정에서 OGD는 $O(GD\sqrt{L})$ sublinear regret를 달성하되(Zinkevich 2003), 이는 평균 보장이라 개별 needle 회수를 함의하지 않고 비볼록 deep memory에는 적용되지 않는다.
- FTL(과거 전체의 argmin으로 점프)은 linear loss에서 마지막 token에 과민해 진동하며 regret가 $\Theta(L)$이다. FTRL은 regularizer $\frac{1}{\eta}R(W)$로 이를 안정화한다.
- linearized loss + $\ell_2$ regularizer의 FTRL은 정확히 OGD이고(식 (3-5)) FTRL의 state는 gradient 누적기다. 이 두 슬롯(loss 모양, regularizer 모양)이 [Miras]에서 attentional bias와 retention gate(→ 13장)가 된다 [Miras §3.2–3.3].
- mirror descent는 "이전 state에 가깝게"의 척도를 Bregman divergence(식 (3-6))로 일반화하며, entropy 선택은 곱셈형(softmax형) update를 낳는다 — 13장 Memora의 KL-retention이 이 기계다.
- loss의 모양은 outlier token이 state를 흔드는 정도를 정한다: $\ell_2$는 비례, $\ell_1$은 방향만, $p>2$는 증폭, Huber는 per-token gradient clipping — Moneta와 Yaad의 축이다(→ 13장).
- retention gate는 학습된 cache eviction policy이고, 이 장의 알고리즘 변형은 전부 GEMM 골격을 보존하는 elementwise epilogue 교체다.

## 자가 점검 체크리스트

- [ ] online protocol의 4단계를 쓰고, 각 단계를 decode loop의 단계와 대응시켜 설명할 수 있다.
- [ ] regret의 정의(식 (3-2))를 쓰고, comparator의 역할과 "sublinear regret"의 의미·한계(평균 보장, convexity 전제)를 설명할 수 있다.
- [ ] 표 3-3의 FTL 진동을 스스로 재현하고, 왜 linear loss에서 FTL이 실패하는지 한 문장으로 말할 수 있다.
- [ ] linearized FTRL + $\ell_2$ regularizer에서 OGD를 유도하고(식 (3-5)), FTRL의 loss 항과 regularizer 항이 [Miras]의 attentional bias / retention gate 자리에 각각 대응함을 설명할 수 있다.
- [ ] mirror descent에서 entropy regularizer가 곱셈형 update를 주는 이유와, 그것이 왜 발산 불가능한 state를 만드는지 설명할 수 있다.
- [ ] "retention gate = 학습된 eviction policy", "regret = aggregate 보장 vs recall@position = pointwise 측정"을 inference 어휘로 옮길 수 있다.

## 다음 장으로

이 장은 세 knob — step size $\eta$, regularizer $R$, loss의 모양 — 을 도입했지만 그것을 **누가 정하는가**는 열어 두었다. 고전 online learning에서는 사람이 정한다. 이 라인에서는 $\eta_t$·$\alpha_t$가 token의 함수로서 slow weights에서 나오고, 초기 상태 $W_{\mathrm{init}}$과 투영 $W_K, W_V, W_Q$까지 전부 pretraining이 학습한다 — online learner의 hyperparameter 전체가 또 하나의 학습 문제의 변수다. 이 구조(inner loop와 outer loop)를 형식화하는 것이 bilevel optimization이고, "이 gate는 누가 학습하는가?"에 체계적으로 답하는 것이 4장의 일이다.
