# ch03. Online learning: OGD, regret, FTRL

<!-- STYLE-ISSUE: STYLE-NOTATION §5.1의 고정 슬러그는 ch03-online-learning.md이나, 오케스트레이터 지시 경로가 ch03-online-learning-regret.md로 주어져 후자로 작성함. P2에서 슬러그 통일 필요. -->

> **이 장의 목표** — 독자가 이 장을 마치면 다음을 할 수 있어야 한다.
> (1) token stream 위의 학습을 online protocol로 정식화하고, 각 단계를 autoregressive decode loop의 단계에 대응시킬 수 있다.
> (2) online gradient descent(OGD)의 update를 쓰고, 그 품질을 regret로 정량화하며, "sublinear regret"가 무엇을 보장하고 무엇을 보장하지 않는지 말할 수 있다.
> (3) Follow-the-Leader의 실패를 손계산으로 재현하고, FTRL의 regularizer가 왜 memory를 안정화하는지 설명할 수 있다.
> (4) FTRL의 두 항(loss 합 + regularizer)이 [Miras]의 attentional bias와 retention gate 자리임을 지적하고, mirror descent가 Memora형 softmax update의 원형임을 알아볼 수 있다.
>
> **왜 필요한가** — [Miras] (*It's All Connected*, arXiv:2504.13173; 이 책은 framework 이름 Miras로 통칭한다)의 §3 전체가 이 장의 어휘로 쓰여 있다: 모든 fixed-state sequence model의 per-token write를 online gradient descent 한 step으로 정식화하고 [Miras Eq. 5], 이를 (FTRL Viewpoint)와 (Learning-Retaining Viewpoint)라는 두 online-optimization 렌즈로 해석한 뒤 후자의 일반성을 증명한다 [Miras §3.2–3.3, Prop. 3.2]. [Atlas] (*Atlas: Learning to Optimally Memorize the Context at Test Time*, arXiv:2505.23735)의 문제의식 자체가 "기존 recurrent model의 online nature" 비판이며, Omega rule(→ 14장)은 online objective를 sliding-window objective로 바꾸는 제안이다 [Atlas §3.2]. 6장에서 만날 Longhorn은 SSM update를 online learning 문제의 닫힌 해로 유도하고(Liu et al. 2025, arXiv:2407.14207), [Titans] (*Titans: Learning to Memorize at Test Time*, arXiv:2501.00663)의 per-token memory update도 online GD의 재해석이다 [Miras §3.1]. [NL] (*Nested Learning: The Illusion of Deep Learning Architecture*, arXiv:2512.24695)의 level별 objective 역시 각 level이 자기 주기의 stream 위에서 푸는 online 문제다. 요컨대 이 장은 Part II 전체가 딛고 설 "stream 위의 최적화" 언어를 공급한다.

## 3.1 Online protocol: 독자가 이미 돌리고 있는 loop

2장은 훈련을 (state, update, cost)를 갖는 객체로 보는 법을 가르쳤다. 거기서의 데이터 소비 방식은 batch training이었다: dataset 전체가 미리 주어져 있고, 매 step i.i.d.로 mini-batch를 뽑아 여러 epoch에 걸쳐 반복 방문한다. 이 장은 그 전제를 버린다.

**online learning**은 데이터가 스트림으로 한 번씩만 도착하는 설정에서의 학습이다. 어떤 분포 가정도 없고, epoch도 없고, 재방문도 없다. 형식적으로 **online protocol**은 매 step $t=1,\dots,L$에서 다음 네 단계를 반복한다:

1. 입력을 받는다 (이 라인에서는 key/value 쌍 $(k_t, v_t)$).
2. 현재 state $W_{t-1}$로 예측한다 (memory 읽기 $\mathcal{M}(k_t; W_{t-1})$).
3. loss $\ell_t(W_{t-1})$를 관측한다 ("이 예측이 얼마나 틀렸나").
4. state를 갱신한다: $W_{t-1} \to W_t$.

이 loop의 모양을 독자는 이미 매일 돌리고 있다. autoregressive decode가 정확히 이 구조다: token을 받고, 현재 state로 forward를 돌려 다음 token을 예측하고, state를 갱신한 뒤 다음 token으로 넘어간다. 차이는 단 하나, 4단계의 내용물이다. 지금까지의 inference에서 "state 갱신"은 KV cache append — 학습이 없는, 계산 없는 write — 였다. 이 라인에서는 4단계가 gradient step이 된다. 그것이 전부다. protocol의 골격은 바뀌지 않는다.

표 3-1 — online protocol과 decode loop의 단계별 대응

| online protocol | autoregressive decode (독자의 세계) | TTT-line layer (이 라인) |
|---|---|---|
| 입력 수신 | 다음 token 도착 | $(k_t, v_t)$ 투영 |
| 예측 | forward pass, logits | memory 읽기 $y_t = \mathcal{M}(q_t; W_t)$ |
| loss 관측 | (없음 — 정답을 모름) | inner loss $\ell(W_{t-1}; k_t, v_t)$ — 정답은 $v_t$ 자신 |
| state 갱신 | KV cache append | gradient step $W_{t-1} \to W_t$ |

표에서 3행이 독자에게 가장 낯설 것이다. inference에는 "정답"이 없는데 무슨 loss를 관측하는가? 답: 이 라인의 inner loss는 외부 정답이 필요 없는 self-supervised 회귀다. "key $k_t$를 넣으면 value $v_t$가 나와야 한다"는 조건 자체가 loss가 된다 — $\ell(W; k_t, v_t) = \|\mathcal{M}(k_t; W) - v_t\|_2^2$. 즉 stream의 각 항목이 자기 자신의 label을 들고 도착한다. 이 형태는 5장(associative memory)에서 그 기원을 만나고, 여기서는 "loss가 매 token 관측 가능하다"는 사실만 필요하다.

분포 가정이 없다는 점은 장식이 아니다. online learning의 이론은 stream이 **적대적**(adversarial)이어도 성립하는 보장을 추구한다 — 최악의 입력 순서에서도 깨지지 않는 성능 하한. 이것은 독자에게 익숙한 사고방식이다: 서빙 시스템을 평균 부하가 아니라 tail latency와 worst-case 트래픽 패턴으로 설계하듯, online learning은 평균 성능이 아니라 worst-case 누적 손해로 알고리즘을 평가한다. 실제 token stream은 i.i.d.가 아니다 — 주제가 바뀌고, 코드 블록이 시작되고, 같은 needle이 두 번 다시 안 나온다. adversarial 보장은 이런 비정상성(non-stationarity)에 대한 보험이다.

마지막으로 2장의 경고를 반복한다: **이 라인에서 훈련의 batch 축 역할은 sequence 축이 대신한다.** 위 protocol의 $t$는 mini-batch 인덱스가 아니라 token 인덱스다. inner loop의 "학습 데이터"는 지금 처리 중인 그 문맥의 token들이며, 같은 문맥을 두 번 지나가지 않는다. online learning이 이 라인의 자연 언어인 이유가 바로 이것이다 — inner loop는 정의상 single-pass 학습이다.

## 3.2 OGD와 regret: stream 압축의 품질 언어

online protocol의 4단계를 채우는 가장 단순한 방법은 2장의 SGD를 그대로 이식하는 것이다. **online gradient descent(OGD)**는 매 step 현재 항목의 loss에 대한 gradient로 한 걸음 내려간다(Zinkevich 2003, ICML):

$$
W_t = W_{t-1} - \eta_t \, \nabla_W \ell_t(W_{t-1})
\tag{3-1}
$$

대수적으로는 2장의 SGD와 같은 식이다. 다른 것은 데이터의 출처다: mini-batch를 i.i.d.로 뽑는 대신 stream이 주는 순서 그대로, 각 항목을 정확히 한 번 소비한다. 그리고 $\ell_t(W) := \ell(W; k_t, v_t)$로 두는 순간, 식 (3-1)은 이 책의 표준형 (M1) — delta rule이자 TTT-Linear의 write — 과 글자까지 같아진다. 이것은 우연이 아니라 [Miras]의 출발점이다: [Miras Eq. 5]는 fixed-state sequence model의 기본 memory update를 정확히 이 식으로 놓고, [Titans]가 momentary surprise(→ 12장)라 불렀던 양이 이 online GD의 gradient에 대한 재해석임을 명시한다 [Miras §3.1]. 즉 **OGD는 (M1)의 최적화-이론 쪽 이름이다.**

그렇다면 이 알고리즘이 "잘한다"는 것을 어떻게 정량화하는가? batch training에는 test loss가 있지만, single-pass stream에는 held-out set이 없다. online learning의 답이 **regret**다: horizon $L$까지의 누적 loss를, 사후에(in hindsight) 고를 수 있는 최선의 고정 state와 비교한 상대 손해로 정의한다.

$$
\mathrm{Reg}_L \;=\; \sum_{t=1}^{L} \ell_t(W_t) \;-\; \min_{W^\star} \sum_{t=1}^{L} \ell_t(W^\star)
\tag{3-2}
$$

여기서 비교 대상 $W^\star$를 **comparator**라 부른다: stream 전체를 다 보고 나서 단 하나 고를 수 있는 최적의 고정 memory 상태다. 식 (3-2)를 세 겹으로 읽는다. 첫째, regret는 절대 성능이 아니라 상대 손해다 — stream 자체가 어려우면(압축 불가능한 랜덤 데이터면) comparator도 손해를 보고, regret는 그 차이만 잰다. 둘째, comparator는 고정이다 — "그때그때 바뀌는 oracle"이 아니라 "최선의 단일 압축 상태"와 비교한다. 셋째, 목표는 **sublinear regret**다: $\mathrm{Reg}_L / L \to 0$, 즉 step당 평균 손해가 comparator 대비 0으로 수렴. 이때 "online learner가 최선의 고정 state를 따라잡는다"고 말한다.

핵심 정리는 다음과 같다: $\ell_t$가 convex이고, gradient norm이 $G$로, 탐색 영역의 지름이 $D$로 유계이면, step size $\eta_t \propto 1/\sqrt{t}$의 OGD는 $\mathrm{Reg}_L = O(GD\sqrt{L})$를 달성한다(Zinkevich 2003; 교과서적 정리는 Shalev-Shwartz 2011, *Online Learning and Online Convex Optimization*, Foundations and Trends in ML; Hazan 2019, arXiv:1909.05207). $\sqrt{L}$은 sublinear이므로 평균 regret는 $O(1/\sqrt{L})$로 사라진다. 증명은 이 장의 범위 밖이고, 직관만 취한다: step size가 $1/\sqrt{t}$로 줄어드는 것이 "초반에는 크게 적응하고 후반에는 안정화한다"는 스케줄이며, 적응(plasticity)과 안정(stability)의 교환이 $\sqrt{L}$이라는 값에서 균형을 이룬다.

> **[해설]** inference 어휘로 번역하면 이렇다. $d_v \times d_k$ matrix memory는 KV cache의 고정 크기 손실 압축이다(→ 1장 Rosetta). regret bound는 이 압축의 품질 보증서다: "이 $O(d^2)$ state는, stream을 다 보고 고른 최선의 $O(d^2)$ 압축 상태보다 평균적으로 뒤지지 않게 된다." 보장의 단위가 개별 조회가 아니라 stream 전체 평균이라는 점이 중요하다 — 특정 needle 하나의 회수는 보장하지 않는다(§3.7에서 다시).

정직하게 한계도 적는다. 첫째, comparator가 고정이므로 분포가 도중에 바뀌는 stream — 문서 경계, 주제 전환 — 에서는 "최선의 고정 state" 자체가 약한 기준이다. comparator가 시간에 따라 움직이는 것을 허용하는 dynamic regret 계열의 확장이 있으나 이 책은 개념의 존재만 언급한다. 둘째, 위 보장은 convexity를 요구한다. 이 라인의 실제 memory는 2-layer MLP(표준 deep memory)이고 그 inner loss는 비볼록이므로, $O(\sqrt{L})$ 보장은 성립하지 않는다. 실제로 [Miras]는 FTRL과 mirror descent라는 online convex optimization의 기계를 설계 언어로 수입하지만, 자신이 제안한 새 변형(Moneta/Yaad/Memora)에 대한 regret bound는 제시하지 않는다 — 정당화는 전적으로 실험이다. 이 장이 가르치는 것은 보장이 아니라 **설계 어휘**다: 이 라인의 모든 update rule은 "어떤 online 문제를 어떤 online 알고리즘으로 푸는가"의 답으로 읽을 수 있고, 그 독법이 Part II를 관통한다.

## 3.3 FTL: 왜 "지금까지의 최적해"로 점프하면 안 되는가

OGD는 gradient 한 걸음이라는 최소한의 갱신이다. 반대편 극단도 생각할 수 있다: 매 step, 지금까지 본 모든 loss의 최적해로 점프하는 것이다. 이를 **Follow-the-Leader(FTL)**라 부른다:

$$
W_t \;=\; \arg\min_{W} \sum_{i=1}^{t-1} \ell_i(W)
\tag{3-3}
$$

겉보기에 FTL은 이상적이다 — 매 순간 이력 전체에 대해 최선이다. 문제는 두 겹이다. 첫째는 비용이다: 매 token마다 과거 전체에 대한 최적화를 다시 푸는 것은, decode마다 KV cache 전체를 재스캔하는 것과 같은 종류의 낭비다. 둘째가 치명적이다: FTL은 **불안정**하다. loss가 평평한(linear) 모양이면 마지막 항목 하나가 argmin을 탐색 영역의 반대편 끝으로 던져 버릴 수 있고, 적대적 stream은 이를 이용해 learner를 매 step 진동시켜 regret를 $\Theta(L)$ — 선형, 즉 학습 실패 — 로 만든다. §3.8의 worked example에서 이 진동을 여섯 step 손계산으로 재현한다. 원인을 한 문장으로 요약하면: FTL의 state는 stream의 마지막 token에 과민하다. cache 어휘로는, 최신 항목 하나가 cache 전체의 배치를 뒤집는 정책이다.

> **[해설]** 그런데 이 라인의 족보에는 FTL의 "전체 이력 최적해"를 실제로 유지하는 극한들이 있다. softmax attention은 매 $t$마다 전체 이력에 대한 $\ell_2$ regression을 non-parametric하게 정확히 푸는 해이고(상태 = KV cache 그 자체, §1.6 카탈로그), Mesa-layer(→ 6장)는 전체 이력 objective를 Newton법으로 정확히 푼다. 이들이 FTL의 불안정을 피하는 방식은 두 가지다: 압축하지 않거나(attention — 상태가 자라므로 점프랄 것이 없다), regularizer를 함께 풀거나(Mesa-layer — 다음 절의 FTRL을 정확히 푸는 셈이다). 그 대가는 독자가 이미 아는 것들이다: 자라는 cache, 그리고 step당 최적화 비용. 이 라인의 recurrent model들은 반대편 끝 — gradient 한 걸음의 OGD — 에서 출발해, 두 극단 사이 어딘가를 설계한다. [Miras]가 "memory learning algorithm"을 독립된 설계 축으로 선언한 것은 정확히 이 스펙트럼을 두고 하는 말이다.

## 3.4 FTRL: regularizer가 memory를 안정화한다

FTL의 진동을 고치는 고전적 처방은 argmin 안에 **regularizer**를 더하는 것이다. **Follow-the-Regularized-Leader(FTRL)**는 다음을 푼다:

$$
W_t \;=\; \arg\min_{W} \; \sum_{i=1}^{t-1} \ell_i(W) \;+\; \frac{1}{\eta}\, R(W)
\tag{3-4}
$$

$R(W)$ — 전형적으로 $\frac{1}{2}\|W\|_2^2$ — 는 두 가지 일을 한다. 첫째, 연속된 해 $W_{t-1}, W_t$ 사이의 거리를 $\eta$ 스케일로 묶어 마지막 항목의 지배력을 없앤다(안정화). 둘째, state의 크기 자체를 벌점화한다 — memory가 무한정 자라는 것을 막는다. 적절한 $\eta$ 선택 하에 FTRL은 convex 설정에서 다시 $O(\sqrt{L})$ regret를 회복한다(Shalev-Shwartz 2011; McMahan 2011, AISTATS).

FTRL이 이 책에서 각별한 이유는 다음 계산에 있다. loss를 각 step의 gradient로 선형화하고 — $\hat\ell_i(W) = \langle g_i, W \rangle$, $g_i = \nabla_W \ell_i(W_{i-1})$ — $R = \frac{1}{2}\|W\|_2^2$를 넣으면 argmin이 닫힌 형태로 풀린다:

$$
W_t \;=\; \arg\min_W \Big\langle \sum_{i=1}^{t-1} g_i,\, W \Big\rangle + \frac{1}{2\eta}\|W\|_2^2
\;=\; -\eta \sum_{i=1}^{t-1} g_i
\;\;\Longrightarrow\;\;
W_t = W_{t-1} - \eta\, g_{t-1}
\tag{3-5}
$$

즉 **linearized loss + $\ell_2$ regularizer의 FTRL은 정확히 OGD다**(제약이 없을 때; $W_0 = 0$). 이 동치는 [Miras Eq. 7]이 명시적으로 쓰는 사실이고, 두 방향으로 읽어야 한다. 한 방향: OGD는 FTRL의 특수경우다 — 그러므로 FTRL의 두 슬롯(loss의 모양, regularizer의 모양)을 갈아 끼우면 OGD를 일반화하는 update rule의 공장이 생긴다. 반대 방향: FTRL의 state는 본질적으로 **gradient 누적기**다 — 식 (3-5)의 중간 형태 $W_t = -\eta\sum g_i$는 state가 "지금까지의 gradient 합"임을 노출한다. 이 둘째 독법은 예고편이다: 13장의 Moneta는 raw 누적기 $A_t$와 정규화된 노출 상태 $W_t$를 분리하는 FTRL형(dual accumulator) update를 쓴다.

이제 [Miras §3.2]의 (FTRL Viewpoint)를 통일 표기로 쓰면 이 장과의 관계가 즉시 보인다:

$$
W_t \;=\; \arg\min_{W \in \mathcal{W}} \; \sum_{i=1}^{t} \hat\ell_i(W; k_i, v_i) \;+\; \frac{1}{\eta_t}\, R_t(W)
$$

첫 항 — 과거 모든 (key, value) 쌍을 얼마나 잘 기억하는가 — 이 [Miras]가 attentional bias(→ 13장)라 이름 붙이는 자리이고, 둘째 항 — memory의 크기와 갱신을 다스리는 벌점 — 이 retention gate(→ 13장)의 자리다. 이 장에서 배운 FTRL의 뼈대에 [Miras]는 인지과학적 이름과 설계 공간을 입힌 것이다. [Miras §3.3]은 같은 update를 다른 렌즈 — 최신 쌍 하나만 fit하되 이전 state $W_{t-1}$ 근처에 머무는 (Learning-Retaining Viewpoint) — 로도 쓰고, 상수 $\eta$, 제약 없는 $\mathcal{W} = \mathbb{R}^d$, 누적 objective의 strict convexity라는 조건 아래 두 렌즈가 동치임을 증명한다 [Miras Prop. 3.2, App. B]. 조건에 주의하라 — 실제로 출하되는 구성(deep memory, data-dependent gate, 제약된 state)에서는 세 조건 모두 성립하지 않으므로, 이 동치는 설계의 동기이지 배포물의 보장이 아니다. 두 viewpoint의 본격적 사용법은 13장이 소유한다; 이 장의 임무는 그 둘이 online learning의 표준 기계(FTRL, proximal step)라는 사실을 못박는 것이다.

Learning-Retaining 렌즈의 최소형 — proximal step — 은 한 번 직접 봐 둘 가치가 있다:

$$
W_t \;=\; \arg\min_W \; \ell_t(W) + \frac{1}{2\eta_t}\|W - W_{t-1}\|_2^2
$$

여기서 $\ell_t$를 선형화하면 explicit gradient step, 즉 식 (3-1)이 그대로 나온다. 선형화하지 않고 정확히 풀면 **implicit GD**(proximal step)가 되는데, $\ell_2$ regression loss에서는 이것도 닫힌 해를 갖는다 — 유효 step size가 $\eta_t/(1 + \eta_t k_t^\top k_t)$로 자동 감쇠하는 delta rule 형태다. Longhorn(Liu et al. 2025, arXiv:2407.14207, *State Space Models are Amortized Online Learners*)이 정확히 이 자리를 차지하는 모델이며, 유도는 6장에서 한다. "SSM 하나가 online learning 문제의 닫힌 해"라는 문장이 이 장을 읽은 뒤에는 마케팅이 아니라 명세로 들려야 한다.

마지막으로 반대 방향의 확장. [Atlas §3.2, Eq. 6]은 기존 recurrent model 전부를 "현재 token 하나의 loss + retention"을 푸는 online 문제로 요약한 뒤, 바로 이 **online nature** — 매 step 현재 항목만 greedy하게 최적화하는 것 — 를 sub-optimal memorization의 원인으로 지목한다. Omega rule(→ 14장)은 loss 항을 최근 $c$개 token의 window 합으로 바꾸며, $c=1$이면 online(delta rule)으로, $c$가 문맥 전체면 global 최적화로 환원된다 [Atlas §3.2]. FTRL의 언어로 말하면: FTRL의 "모든 과거 항의 합"과 OGD의 "현재 항 하나" 사이를 window 길이 $c$가 매개한다. online learning의 스펙트럼 위에서 여섯 논문이 서로 다른 점을 고르는 그림이 이렇게 완성된다.

## 3.5 Mirror descent와 Bregman divergence

FTRL의 regularizer 슬롯에 $\|W\|_2^2$를 넣는 것은 state 공간이 유클리드 공간이라는 암묵적 가정이다. state에 구조가 있으면 — 모든 성분이 양수여야 한다든가, 합이 고정이라든가, 유계여야 한다든가 — "가깝다"의 척도부터 바꾸는 것이 자연스럽다. 그 일반화의 도구가 **Bregman divergence**다. strictly convex한 함수 $F$에 대해

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

— exponentiated gradient, 또는 multiplicative weights라 불리는 update다(FTRL과 mirror descent의 정확한 관계는 McMahan 2011). 성질을 눈여겨보라: state 성분은 정의상 항상 양수이고, 매 step 재정규화되므로 총합이 보존된다. forgetting이 곱셈 감쇠(decay)로 일어나는 것이 아니라 **성분들 사이의 질량 경쟁**으로 일어난다 — 어떤 성분이 커지려면 다른 성분이 줄어야 한다. 따라서 state는 문맥이 아무리 길어져도 원리적으로 발산할 수 없다.

이 기계의 행선지는 13장이다. Memora의 KL-retention update $W_t = \mathrm{Softmax}(\alpha_t \log W_{t-1} - \eta_t \nabla_W \ell)$는 위 exponentiated-gradient 식에 retention gate를 결합한 것이고, [Miras §5.2]의 f-divergence retention 일반화도 같은 틀 — $D_F$ 자리에 다른 divergence — 이다. mirror descent를 여기서 만나 두면, 13장에서 "memory update에 웬 softmax?"라는 당혹 대신 "아, entropy regularizer구나"라는 인식이 온다. systems 접점 한 줄: 유계·정규화된 state는 발산 걱정이 없는 memory 설계의 원리적 근거이며, 수치 표현 관점의 함의(양수 유계 상태의 저장)는 13장의 systems 절이 다룬다.

## 3.6 Loss geometry: $\ell_2$, $\ell_1$, $\ell_p$, Huber

FTRL의 두 슬롯 중 regularizer 쪽은 §3.4–3.5가 다뤘다. 남은 슬롯은 loss의 모양이다. residual을 $r = \mathcal{M}(k_t; W) - v_t$라 하자. loss가 $r$의 어떤 함수인가에 따라, 같은 OGD라도 "어떤 token이 state를 세게 흔드는가"가 달라진다. 1차원 residual로 축소해 gradient 크기를 직접 보는 것이 가장 빠르다.

표 3-2 — loss 모양별 gradient 크기 $|d\,\mathrm{loss}/dr|$ (스칼라 residual $r$, Huber threshold $\delta = 1$)

| loss | gradient 크기 식 | $r=0.1$ | $r=1$ | $r=3$ |
|---|---|---|---|---|
| $\ell_2$: $r^2$ | $2\lvert r\rvert$ | 0.2 | 2 | 6 |
| $\ell_1$: $\lvert r\rvert$ | $1$ | 1 | 1 | 1 |
| $\ell_3$: $\lvert r\rvert^3$ | $3r^2$ | 0.03 | 3 | 27 |
| Huber($\delta{=}1$) | $\lvert r\rvert$ (내부), $\delta$ (외부) | 0.1 | 1 | 1 |

읽는 법: $\ell_2$는 오차에 비례해 쓴다 — 예측을 크게 벗어난 token 하나(오타, 노이즈, 적대적 스팬)가 state를 그 크기만큼 크게 흔든다. $\ell_1$은 방향만 쓰고 크기를 버린다 — "이 key가 있었다"는 사실만 기록하는 극단으로, [Miras]는 $p=1$을 value-less associative memory라 부른다(→ 13장). $p>2$는 반대로 큰 오차를 증폭한다 — surprising token일수록 더 세게 기록하는 설계다. Huber는 threshold $\delta$ 안에서 $\ell_2$, 밖에서 $\ell_1$로 행동한다: gradient 크기가 $\delta$에서 포화하므로, 이것은 정확히 **per-token gradient clipping**이다. 독자가 rate limiter나 saturating counter에 대해 가진 직관이 그대로 적용된다 — 어떤 단일 이벤트도 정해진 한도 이상으로 상태를 밀 수 없다.

이 축의 제품이 13장의 Moneta($\ell_p$ bias, 매끄럽게 근사된 sign·절댓값)와 Yaad(Huber bias, token마다 학습된 threshold $\delta_t$)다. 이 장은 모양의 어휘만 공급한다 — 어떤 모양이 언어 모델링에서 실제로 이기는가는 이론이 아니라 [Miras]의 실험 절이 답하는 질문이고, 그 수치와 ablation은 13장에서 다룬다.

## 3.7 Systems bridge: 학습된 eviction policy로서의 retention

이 장의 개념들을 독자의 production 어휘로 되감는다.

**state는 cache이고, retention은 eviction이다.** 독자가 아는 KV cache eviction은 명시적 정책이다: sliding window로 오래된 항목을 자르고, quota로 총량을 막고, score 기반으로 항목을 골라 버린다. fast-weight memory에는 그런 정책 코드가 없다. 대신 §3.4의 regularizer $R$(또는 retention 항)이 그 역할을 한다 — 무엇이 얼마나 오래 살아남는가가 discrete한 규칙이 아니라 **최적화 문제의 벌점 항 선택으로 결정된다.** Rosetta 사전의 해당 행이 정확히 이 뜻이다: retention gate = 학습된 eviction. 벌점의 모양이 eviction의 성격을 정한다: $\ell_2$ 벌점은 모든 성분을 조금씩 깎는 곱셈 감쇠(soft)로, $\ell_1$ 성분은 작은 항목을 정확히 0으로 자르는 절삭(hard)으로, KL은 질량 경쟁(§3.5)으로 나타난다 — 세 형태 모두 13장에서 실제 모델로 만난다. eviction 정책을 사람이 튜닝하는 대신, 이 라인에서는 gate를 만드는 slow weights가 outer loop에서 그것을 학습한다(→ 4장).

**regret와 recall@position은 같은 것의 두 투영이다.** 독자가 long-context 평가에서 보는 needle-in-a-haystack recall 곡선은 pointwise 측정이다: 특정 (key, value) 쌍이 $N$ token 뒤에도 회수되는가. regret는 aggregate 보장이다: stream 전체에 대한 누적 loss가 comparator 대비 유계인가. 압축 state가 needle을 못 꺼낸다는 것은 그 쌍의 $\ell_t$가 comparator 대비 크다는 것이므로, recall 실패는 regret의 국소 성분이다. 방향에 주의: sublinear regret는 평균의 보장이지 개별 needle의 보장이 아니다 — state는 평균적으로 잘하면서 특정 needle을 얼마든지 버릴 수 있다. "regret가 낮은데 recall이 나쁜" 모델은 모순이 아니라, aggregate 최적화가 pointwise 회수를 함의하지 않는다는 사실의 전시다. retention의 품질이 long-context 성능을 지배한다는 [Miras]의 실험적 주장(→ 13장)을 읽을 때 이 구분을 갖고 있어야 한다.

**비용의 GEMM shape.** matrix memory $W \in \mathbb{R}^{d_v \times d_k}$와 $\ell_2$ loss에서 OGD 한 step의 계산은 정확히 두 조각이다: 읽기 $Wk_t$ (GEMV), 쓰기 $\nabla_W \ell = (Wk_t - v_t)\,k_t^\top$ (rank-1 outer product). 2장에서 본 $dW = (\text{오차})\,k^\top$ 그 shape다. per-token decode에서는 state 전체를 읽고 다시 쓰는 read-modify-write이므로 이 연산은 bandwidth-bound다; token $C$개를 chunk로 묶으면 rank-$C$ GEMM이 되어 tensor core 쪽으로 이동한다 — 그 변환이 9장의 전부다. 이 장에서 배운 변형들의 추가 비용도 shape로 읽힌다: FTRL형 dual accumulator는 상주 state를 두 배로 만들고(serving에서 state residency와 checkpoint 비용 2×), loss와 retention의 교체는 GEMM 골격을 건드리지 않는 elementwise epilogue 교체다 — sign, 절댓값, threshold 마스크, softmax 재정규화 전부가 그렇다. "알고리즘 축의 설계 변경이 kernel 골격을 보존한다"는 이 관찰이, 13–14장의 변형들이 전부 같은 chunkwise 기계 위에서 훈련될 수 있는 이유다.

## 3.8 Worked micro-example: FTL의 진동, FTRL의 안정화, OGD와의 동치

수치로 확인하자. 무대는 가장 작은 memory다: 스칼라 state $w \in [-1, 1]$ (1×1 matrix memory), loss는 linear $\ell_t(w) = z_t\, w$. linear loss는 장난감이 아니다 — [Miras]의 분류에서 Hebbian family(linear attention, RetNet, Mamba-2, GLA; → 6장, 13장)의 attentional bias가 정확히 linear($\tilde\ell_t = -2\langle Wk_t, v_t\rangle$)이므로, $k_t = 1$로 두면 이 예제는 linear-attention family의 1×1 버전이다($z_t = -2v_t$로 읽으면 된다).

adversarial stream을 $z_1 = 0.5$, 이후 $z_t = -1, +1, -1, +1, -1$ (교대)로 잡고 $L = 6$까지 돌린다. FTL은 식 (3-3)에 따라 $w_t = \arg\min_{w \in [-1,1]} \big(\sum_{i<t} z_i\big) w$, 즉 누적합의 반대쪽 끝점 $-\mathrm{sign}(\sum_{i<t} z_i)$을 고른다(이력이 없는 $t=1$은 관례상 $w_1 = 0$).

표 3-3 — FTL의 여섯 step (누적합은 step 시작 시점 기준)

| $t$ | $\sum_{i<t} z_i$ | $w_t$ (FTL) | $z_t$ | loss $z_t w_t$ |
|---|---|---|---|---|
| 1 | 0 | 0 | 0.5 | 0 |
| 2 | 0.5 | $-1$ | $-1$ | $+1$ |
| 3 | $-0.5$ | $+1$ | $+1$ | $+1$ |
| 4 | 0.5 | $-1$ | $-1$ | $+1$ |
| 5 | $-0.5$ | $+1$ | $+1$ | $+1$ |
| 6 | 0.5 | $-1$ | $-1$ | $+1$ |

누적 loss는 $5$다. comparator를 계산하면: $\sum_{t=1}^{6} z_t = -0.5$이므로 최선의 고정 $w^\star = +1$이고 그 누적 loss는 $-0.5$다. 따라서 $\mathrm{Reg}_6 = 5 - (-0.5) = 5.5$ — step당 거의 1씩, horizon에 비례해 자란다. 표에서 메커니즘이 그대로 보인다: 마지막 항목이 누적합의 부호를 뒤집을 때마다 FTL은 도메인의 반대편 끝으로 점프하고, 적대적 stream은 그 점프 직후의 방향을 정확히 벌준다. state가 마지막 token에 과민하다는 §3.3의 진단이 이 진동이다.

이제 FTRL이다. $R(w) = \frac{1}{2}w^2$, $\eta = 0.5$로 식 (3-4)를 풀면 $w_t = -\eta \sum_{i<t} z_i$ (모두 $[-1,1]$ 안이므로 제약은 발동하지 않는다).

표 3-4 — FTRL($\eta = 0.5$)의 여섯 step

| $t$ | $\sum_{i<t} z_i$ | $w_t = -0.5 \sum_{i<t} z_i$ | $z_t$ | loss $z_t w_t$ |
|---|---|---|---|---|
| 1 | 0 | 0 | 0.5 | 0 |
| 2 | 0.5 | $-0.25$ | $-1$ | $+0.25$ |
| 3 | $-0.5$ | $+0.25$ | $+1$ | $+0.25$ |
| 4 | 0.5 | $-0.25$ | $-1$ | $+0.25$ |
| 5 | $-0.5$ | $+0.25$ | $+1$ | $+0.25$ |
| 6 | 0.5 | $-0.25$ | $-1$ | $+0.25$ |

누적 loss는 $1.25$, $\mathrm{Reg}_6 = 1.25 - (-0.5) = 1.75$. 같은 적대적 stream에서 진동의 진폭이 1에서 $0.25$로 줄었고, 진폭을 정하는 것이 정확히 $\eta$다 — regularizer의 무게 $1/\eta$가 클수록 state는 덜 움직이고, 그만큼 덜 배운다. plasticity와 stability의 교환이 숫자 하나로 손에 잡힌다. [Miras]가 $\eta_t$를 "meta in-context learning rate"라 부르며 "크면 더 배우고 더 잊는다"고 서술하는 것 [Miras §3.3]이 이 knob의 이름이다.

끝으로 식 (3-5)의 동치를 이 숫자로 검산한다. linear loss의 gradient는 $\nabla \ell_t(w) = z_t$이므로 OGD는 $w_t = w_{t-1} - \eta z_{t-1}$이다: $w_2 = 0 - 0.5(0.5) = -0.25$, $w_3 = -0.25 - 0.5(-1) = +0.25$, $w_4 = 0.25 - 0.5(1) = -0.25$ — 표 3-4의 FTRL 열과 자리마다 일치한다. "과거 전체의 regularized argmin"과 "gradient 한 걸음"이 같은 궤적이라는 것, 이것이 이 장의 중심 동치다. (제약이 실제로 발동하는 설정에서는 FTRL(lazy projection)과 OGD(greedy projection)가 갈라질 수 있다 — 그 관계의 정밀한 지도는 McMahan 2011이다.)

Hebbian 독법으로 마무리한다: $z_t = -2v_t$로 두면 위 OGD는 $w_t = w_{t-1} + 2\eta\, v_{t-1}$, 즉 순수 가산 write다. 방금 손으로 돌린 여섯 step이 linear attention의 1×1 decode 궤적이었던 셈이다.

## 요약

- online learning은 stream 위에서 "수신 → 예측 → loss 관측 → state 갱신"을 반복하는 protocol이며, autoregressive decode와 loop 모양이 같다. 새 요소는 state 갱신이 gradient step이 된다는 것 하나다.
- OGD $W_t = W_{t-1} - \eta_t \nabla_W \ell_t(W_{t-1})$는 이 책 표준형 (M1)의 최적화-이론 이름이다. fixed-state sequence model의 per-token write는 online GD 한 step이다 [Miras Eq. 5].
- regret는 stream 압축 품질의 언어다: 사후 최선의 고정 comparator $W^\star$ 대비 누적 손해. convex·유계 설정에서 OGD는 $O(GD\sqrt{L})$의 sublinear regret를 달성한다(Zinkevich 2003). 이 보장은 평균에 대한 것이며 개별 needle의 회수를 함의하지 않고, 비볼록 deep memory에는 적용되지 않는다.
- FTL(과거 전체의 argmin으로 점프)은 linear loss에서 마지막 token에 과민해 진동하며 regret가 $\Theta(L)$이다. FTRL은 regularizer $\frac{1}{\eta}R(W)$로 이를 안정화한다.
- linearized loss + $\ell_2$ regularizer의 FTRL은 정확히 OGD이고(식 (3-5)), FTRL의 state는 gradient 누적기다. 이 두 슬롯(loss 모양, regularizer 모양)이 [Miras]에서 attentional bias와 retention gate(→ 13장)가 된다 [Miras §3.2–3.3].
- mirror descent는 "이전 state에 가깝게"의 척도를 Bregman divergence로 일반화하며, entropy 선택은 곱셈형(softmax형) update를 낳는다 — 13장 Memora의 KL-retention이 정확히 이 기계다.
- loss의 모양은 outlier token이 state를 흔드는 정도를 정한다: $\ell_2$는 비례, $\ell_1$은 방향만, $p>2$는 증폭, Huber는 per-token gradient clipping이다 — Moneta와 Yaad의 축이다(→ 13장).
- retention gate는 학습된 cache eviction policy이고, 이 장의 알고리즘 변형은 전부 GEMM 골격을 보존하는 elementwise epilogue 교체다.

## 자가 점검 체크리스트

- [ ] online protocol의 4단계를 쓰고, 각 단계를 decode loop의 단계와 대응시켜 설명할 수 있다.
- [ ] regret의 정의(식 (3-2))를 쓰고, comparator의 역할과 "sublinear regret"의 의미·한계(평균 보장, convexity 전제)를 설명할 수 있다.
- [ ] 표 3-3의 FTL 진동을 스스로 재현하고, 왜 linear loss에서 FTL이 실패하는지 한 문장으로 말할 수 있다.
- [ ] linearized FTRL + $\ell_2$ regularizer에서 OGD를 유도하고(식 (3-5)), FTRL state가 gradient 누적기임을 지적할 수 있다.
- [ ] FTRL의 loss 항과 regularizer 항이 [Miras]의 attentional bias / retention gate 자리에 각각 대응함을 설명할 수 있다.
- [ ] mirror descent에서 entropy regularizer가 곱셈형 update를 주는 이유와, 그것이 왜 발산 불가능한 state를 만드는지 설명할 수 있다.
- [ ] "retention gate = 학습된 eviction policy", "regret = aggregate 보장 vs recall@position = pointwise 측정"을 inference 어휘로 옮길 수 있다.

## 다음 장으로

이 장은 세 개의 knob — step size $\eta$, regularizer $R$, loss의 모양 — 을 도입했지만, 그것을 **누가 정하는가**는 열어 두었다. 고전 online learning에서는 사람이 정한다. 이 라인에서는 다르다: $\eta_t$와 $\alpha_t$는 token의 함수로서 slow weights가 만들어 내고, 초기 상태 $W_{\mathrm{init}}$과 투영 $W_K, W_V, W_Q$까지 전부 보통의 pretraining이 학습한다. 즉 online learner의 hyperparameter 전체가 또 하나의 학습 문제의 변수다. 학습 문제 안에 학습 문제가 들어 있는 이 구조 — inner loop와 outer loop — 를 형식화하는 것이 bilevel optimization이고, "이 gate는 누가 학습하는가?"라는 질문에 체계적으로 답하는 것이 4장의 일이다.
