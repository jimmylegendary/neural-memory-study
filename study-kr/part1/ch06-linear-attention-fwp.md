# ch06. Linear attention과 fast-weight programming: DeltaNet, Gated DeltaNet, Longhorn, RWKV-7

> **이 장의 목표** — 독자가 이 장을 마치면 다음을 할 수 있어야 한다.
> 1. softmax attention에서 kernel trick으로 linear attention의 재귀형을 유도하고, 그것을 "KV cache를 고정 크기 행렬로 압축하는 write 연산"으로 읽을 수 있다.
> 2. RetNet → GLA → DeltaNet → Gated DeltaNet → Longhorn → RWKV-7을 별개의 발명품이 아니라 **하나의 (state, update, cost) 객체에서 inner objective·optimizer·retention을 바꿔 낀 변형들**로 배치할 수 있다.
> 3. Longhorn의 implicit gradient descent를 직접 유도하고, 왜 step size에 대해 무조건 안정인지 논증할 수 있다.
> 4. $d=2$ 손계산으로 Hebbian write의 crosstalk와 delta write의 overwrite 동작을 재현하고, 각 모델의 per-token decode 비용을 자기 서빙 스택의 roofline 위에 올려놓을 수 있다.
>
> **왜 필요한가** — 6편 전부가 이 장의 모델 family를 직접 전제한다.
> - [Titans] (*Titans: Learning to Memorize at Test Time*, arXiv:2501.00663)의 §2는 linear attention의 재귀형 [Titans Eq. 4–5]를 출발점으로 놓고, 그 "additive write의 memory overflow" 비판과 두 가지 개선 방향 — forget 기제(GLA·Mamba-2 계열)와 write 개선(delta rule 계열) — 으로 자기 위치를 정의한다 [Titans §2]. 12장의 bridge-in은 이 서사를 그대로 잇는다.
> - [Miras] (*It's All Connected*, arXiv:2504.13173; 이 책은 framework 이름 Miras로 통칭)의 Table 1은 이 장의 모델 전부를 4축(memory 구조, attentional bias, retention, learning algorithm)으로 분류한다. [Miras Eq. 8]과 [Miras Eq. 9]가 이 장의 식 (6-2), (6-3)이다. 13장은 이 장의 모델들이 이미 손에 익었다고 가정한다.
> - [Atlas] (*Atlas: Learning to Optimally Memorize the Context at Test Time*, arXiv:2505.23735)의 Omega rule은 delta rule의 sliding-window 일반화다. DeltaNet을 모르면 14장의 "token 하나 최적화 vs window 최적화" 대비가 공허해진다.
> - [TNT] (*TNT: Improving Chunkwise Training for Test-Time Memorization*, arXiv:2511.07343)는 이 family를 "linear memory modules" 대조군으로 쓰고, [NL] (*Nested Learning*, arXiv:2512.24695)은 linear attention을 nested optimization의 상시 2-level 예제로 사용한다.

이 장은 독자의 홈그라운드에서 시작한다. KV cache, GEMV, roofline — 전부 이미 아는 어휘다. 새로 배우는 것은 단 하나의 관점 전환이다: **linear attention의 state 갱신은 "작은 모델의 weight를 한 걸음 학습시키는 것"과 같은 수식이며, 따라서 이 family의 모든 변형은 optimizer의 선택 문제로 환원된다.** 이 관점이 서면 Titans 라인 6편은 "inner optimizer를 점점 좋은 것으로 갈아 끼우는 연대기"로 읽힌다.

## 6.1 Kernel trick: KV cache를 $d_v \times d_k$ 행렬로 압축한다

softmax attention의 decode 비용부터 복기한다. token $t$에서 attention은 query $q_t$로 지금까지 쌓인 모든 key를 조회한다:

$$
y_t = \sum_{j=1}^{t} \frac{\exp(q_t^\top k_j)}{\sum_{l=1}^{t} \exp(q_t^\top k_l)}\, v_j .
$$

독자가 매일 보는 그 비용 구조다: KV cache는 token당 $(d_k + d_v)$개 원소씩 자라고, decode 한 step은 cache 전체를 한 번 읽는다. 길이 $L$에서 read traffic이 $O(L)$ — decode가 bandwidth-bound가 되는 바로 그 이유다.

**linear attention**은 이 합에서 $\exp$를 제거하는 데서 출발한다. Katharopoulos et al. 2020 (arXiv:2006.16236)은 similarity를 **feature map** $\phi$의 내적 $\phi(q)^\top\phi(k)$로 바꾸면(원문은 $\phi(x)^\top\phi(y) \ge 0$을 위해 $\mathrm{elu}+1$을 사용) 합의 결합 순서를 재배열할 수 있음을 지적했다 [Titans Eq. 3]:

$$
y_t = \frac{\phi(q_t)^\top \sum_{j\le t} \phi(k_j)\, v_j^\top}{\phi(q_t)^\top \sum_{l\le t} \phi(k_l)}
\;\;\Longrightarrow\;\;
\text{누적량 } \sum_{j\le t} v_j\,\phi(k_j)^\top \text{만 유지하면 된다.}
$$

$\phi$를 항등으로 두면(이하 이 장의 기본 설정; RetNet 이후의 관행) 누적량은 $d_v\times d_k$ 행렬 하나가 되고, 전체가 재귀형으로 떨어진다 [Titans Eq. 4–5]:

$$
W_t = W_{t-1} + v_t k_t^\top, \qquad y_t = W_t\, q_t
\tag{6-1}
$$

(원문들은 행벡터 관행으로 $M_t = M_{t-1} + K_t^\top V_t$로 쓴다. 이 책은 열벡터·$W\in\mathbb{R}^{d_v\times d_k}$ 관행으로 통일한다 — §표기, 1장.) 분모의 누적 벡터 $z_t = z_{t-1} + \phi(k_t)$도 원리상 함께 유지해야 하지만, RetNet 이후의 모델들은 분모 누적을 아예 버리고 출력 쪽 normalization으로 대체한다 — RetNet은 head 출력에 GroupNorm을 두고 그 scale-invariance를 수치 안정화에 사용하며 (Sun et al. 2023, arXiv:2307.08621, §2.2·§3.1), GLA는 "normalizer 없는" linear attention이 실전에서 잘 동작함을 명시하고 head별 LayerNorm을 출력에 둔다 (Yang et al. 2024, arXiv:2312.06635, §2).

식 (6-1)을 Rosetta-Stone 사전(→ 1장)으로 읽자. 이것은 **KV cache의 손실 압축**이다. softmax attention은 모든 $(k_j, v_j)$ 쌍을 그대로 보관하는 non-parametric memory이고 — [Miras §4]는 이를 "$\ell_2$ regression의 Nadaraya–Watson 해, retention 없음"으로 정식화한다 — linear attention은 같은 스트림을 고정 크기 행렬 $W_t$에 겹쳐 쌓는다. write는 rank-1 outer product $v_tk_t^\top$의 accumulate(BLAS로 말하면 GER), read는 GEMV 한 번이다. cache가 자라지 않으므로 decode의 FLOPs/token과 bytes/token이 문맥 길이와 무관한 상수가 된다. 이득의 정체는 산술 강도가 아니라 **traffic 총량의 상한**이다 — 이 구분은 §6.9에서 숫자로 확인한다.

단, append와 write의 차이에 밑줄을 긋는다. KV cache append는 append-once다: 한 번 쓴 항목은 불변이고, 간섭도 없다. 식 (6-1)의 write는 **read-modify-write**다: 모든 쌍이 같은 $d_v\times d_k$ 저장소에 겹쳐 써지고, key들이 직교하지 않는 한 서로 간섭한다. 5장에서 본 그대로 — 식 (6-1)은 correlation matrix memory의 Hebbian write이고, 겹쳐 쓰기의 대가는 crosstalk(→ 5장)이다. [Titans §2]는 이것을 "additive write의 memory overflow"라고 부르며, 이 장의 나머지 전부는 이 한 문제에 대한 답들의 계보다.

## 6.2 Fast-weight programming: 1992년의 발명, 2021년의 재발견

식 (6-1)의 $W_t$에 이름을 붙일 차례다. 이 책 전체에서 가장 중요한 용어 구분이 여기서 도입된다.

**fast weights**는 sequence가 흐르는 동안 token마다 갱신되는 weight — 식 (6-1)의 $W_t$ — 를 말한다. **slow weights**는 pretraining이 끝나면 얼어붙는 보통의 parameter — projection $W_K, W_V, W_Q$를 포함한 $\Theta$ 전부 — 를 말한다. 표기 규약도 이 구분을 따른다: fast는 $W$, slow는 $\Theta$. 독자의 세계에서 slow weights는 "모델"이고 fast weights는 "요청마다 존재하는 상태"다. KV cache가 그랬듯이, $W_t$는 session state이지 model parameter가 아니다.

**fast weight programming (FWP)**은 Schmidhuber 1992 (*Learning to control fast-weight memories*, Neural Computation 4(1))가 제안한 구도다: 한 network(slow net)가 다른 network의 weight(fast weights)를 입력에 따라 **써 넣는다**. slow net의 출력이 데이터가 아니라 "다른 모델의 parameter"라는 점에서, slow net은 fast net을 *프로그래밍*한다. Schlag, Irie & Schmidhuber 2021 (arXiv:2102.11174)은 kernel trick으로 얻은 linear attention이 정확히 이 1992년 구도임을 보였다: projection들이 slow net이고, outer-product write가 프로그래밍 명령이며, read $W_tq_t$가 프로그램 실행이다. 같은 그룹의 후속작 Irie et al. 2021 (arXiv:2106.06295)은 fast net 쪽을 재귀적으로 확장했다. 그리고 같은 2021년 논문이 Hebbian write 대신 **delta rule(→ 5장)을 fast-weight update로 쓰는 모델** — 오늘날 DeltaNet이라 불리는 것의 원형 — 을 제안했다 [Miras §4]. 30년 묵은 아이디어가 attention의 어휘로 번역되는 순간이 이 라인의 역사적 기점이다.

<!-- FIG: ch06/fig-01-fwp-timeline -->

표 6-1 — FWP 계보 timeline. 이 장이 다루는 구간은 굵게.

| 연도 | 사건 | 의미 |
|---|---|---|
| 1960 | delta rule (Widrow–Hoff) | error-driven write의 원형 (→ 5장) |
| 1972 | correlation matrix memory (Kohonen) | outer-product associative memory (→ 5장) |
| 1992 | fast-weight controller (Schmidhuber) | slow net이 fast weights를 프로그램 |
| **2020** | **linear attention (arXiv:2006.16236)** | **kernel trick, 재귀형 (6-1)** |
| **2021** | **FWP 동치 + delta-rule 변형 (arXiv:2102.11174)** | **linear attention = FWP; DeltaNet 원형** |
| **2023** | **RetNet (arXiv:2307.08621)** | **상수 decay gate** |
| **2024** | **GLA (arXiv:2312.06635)** | **data-dependent diagonal gate + chunkwise kernel** |
| **2024** | **DeltaNet 병렬화 (arXiv:2406.06484)** | **delta rule의 scale-up (WY, → 9장)** |
| 2024 | TTT (arXiv:2407.04620) | state = 작은 MLP의 weights (→ 8장) |
| **2024** | **Longhorn (arXiv:2407.14207)** | **online 문제의 closed-form 해 = implicit GD** |
| **2025** | **Gated DeltaNet (arXiv:2412.06464)** | **gate + delta 결합** |
| **2025** | **RWKV-7 (arXiv:2503.14456)** | **generalized delta rule (채널별 gate)** |
| 2025 | Titans (arXiv:2501.00663) | momentum + retention + deep memory (→ 12장) |

FWP 관점이 주는 실질적 이득은 두 가지다. 첫째, **두 시간 척도의 분리**가 명시된다. fast weights는 inner loop(→ 1장, 4장)에서 token마다 움직이고, slow weights는 outer loop에서 gradient로 학습된다. "이 write rule의 $\eta_t$는 누가 정하는가?"라는 질문의 답이 항상 준비된다: write rule 자체는 inner loop의 사건이고, write rule의 계수를 만들어내는 함수($\eta_t = \eta(x_t;\Theta)$ 같은 gate head)는 slow weights의 일부로 outer loop가 학습한다. 둘째, 이 장의 모델들을 (state, update, cost) 객체로 다룰 어휘가 생긴다. state는 $W_t$($d_v\times d_k$ 행렬, per head), update는 이하 각 절의 한 줄 수식, cost는 GEMV + rank-1 갱신이다. systems 독자에게 fast weights란 결국 "kernel이 매 step 갱신하는 레지스터 파일 같은 on-chip 상주 후보 상태"이며, 실제로 이 family의 고성능 kernel들은 $W_t$를 SRAM에 상주시키는 형태로 짜인다(→ 9장).

## 6.3 Gate의 도입: RetNet과 GLA — 학습된 eviction

Hebbian write의 첫 번째 문제는 지우는 수단이 없다는 것이다. 식 (6-1)은 명시적 감쇠 기제 없이 더하기만 하므로(부호가 맞는 항끼리 상쇄되지 않는 한 state의 norm이 자라고), 오래된 쌍이 영원히 남아 crosstalk를 누적시킨다. cache 어휘로 말하면 **eviction policy가 없는 cache**다. 첫 번째 교정은 자명한 방향이다: 매 step 이전 state를 조금 깎는다.

$$
W_t = \alpha_t\, W_{t-1} + v_t k_t^\top
\tag{6-2}
$$

여기서 $\alpha_t\in[0,1]$은 retention gate(→ 13장; 역사적 별칭 "forget gate")이고, 이 책의 방향 규약대로 **남기는 비율**이다($\alpha_t = 1$이면 전부 유지). 식 (6-2)의 스펙트럼 위에 세 모델이 놓인다 [Miras Eq. 8, §4]:

- **RetNet** (Sun et al. 2023, arXiv:2307.08621): $\alpha$가 데이터와 무관한 고정 상수 — head 인덱스로 미리 정한 감쇠율($\gamma = 1 - 2^{-5-\mathrm{arange}(h)}$ 꼴)이고 학습되지 않는다. head마다 다른 이 고정 감쇠로 다중 시간 척도를 만든다 [RetNet §2.2 Eq. 8].
- **GLA** (Yang et al. 2024, arXiv:2312.06635): $\alpha_t$가 입력의 함수인 **data-dependent diagonal gate**. key 채널별로 남기는 비율이 달라진다 — 통일 표기로 $W_t = W_{t-1}\,\mathrm{Diag}(\alpha_t) + v_tk_t^\top$.
- **Mamba-2**: $\alpha_t$가 data-dependent **스칼라**. SSM 계보에서 도달한 같은 지점이며, 유도와 duality는 7장이 담당한다.

[Miras §4]의 재해석이 이 지점에서 처음 빛을 발한다: 식 (6-1)의 Hebbian write조차 "1-step gradient descent"다 — 다만 inner objective가 $\ell_2$ regression이 아니라 dot-product similarity $\tilde\ell_t = -2\langle W k_t, v_t\rangle$일 뿐이다 [Miras Eq. 8]. 이 objective는 아래로 unbounded이므로 GD가 멈출 이유가 없고, 그래서 norm이 자란다. gate는 이 발산을 억제하는 regularization — Miras의 언어로 retention — 이며, "왜 gate가 필요한가"는 "왜 이 inner objective에는 자기 제한이 없는가"의 다른 표현이다.

systems 접점: (6-2)의 추가 비용은 state에 대한 elementwise 곱 하나, 즉 $d_kd_v$ FLOPs와 (fusion이 없다면) state 한 벌의 추가 read-write traffic이다. decode에서는 어차피 state RMW가 지배 항이므로 gate의 한계 비용은 사실상 0이고, prefill/training에서는 chunk 경계마다 감쇠 계수를 접어 넣는 형태로 GEMM화된다(→ 9장). 학습된 eviction을 공짜로 얻는 셈이다 — 단, eviction은 **채널 단위**다. 특정 "항목"을 지목해 지울 수는 없다. 그 능력이 다음 절의 주제다.

## 6.4 DeltaNet: append에서 overwrite로 — write 연산의 교정

gate는 "전체를 잊는" 수단이지 "이 key의 옛 값을 고쳐 쓰는" 수단이 아니다. key $k$가 새 값 $v^{\text{new}}$로 다시 나타났을 때 Hebbian write는 $v^{\text{old}}k^\top$ 위에 $v^{\text{new}}k^\top$를 그냥 더한다 — 읽으면 두 값의 합이 나온다. 올바른 동작은 **먼저 옛 값을 읽어서 빼고, 새 값을 쓰는 것**이다. 이것이 delta rule(→ 5장)이고, 그것을 fast-weight update로 채택한 모델이 **DeltaNet**이다(원형은 Schlag et al. 2021, scale-up은 Yang et al. 2024, arXiv:2406.06484 [Miras §4; Titans §2]).

유도는 식 (M1)의 linear memory 전개, 그 이상도 이하도 아니다. inner objective를 $\ell(W;k_t,v_t) = \|W k_t - v_t\|_2^2$로 두면 $\nabla_W \ell = 2(Wk_t - v_t)k_t^\top$이고(계수 2는 $\eta_t$에 흡수한다), 1-step GD는

$$
W_t = W_{t-1} - \eta_t (W_{t-1}k_t - v_t)k_t^\top
= W_{t-1}\big(I - \eta_t k_t k_t^\top\big) + \eta_t v_t k_t^\top .
\tag{6-3}
$$

이 한 줄에 세 개의 얼굴이 있다.

1. **optimizer의 얼굴**: (6-3)은 online regression의 SGD step이다. 예측 $W_{t-1}k_t$를 만들고(read, GEMV), 오차 $W_{t-1}k_t - v_t$를 재고, 오차의 outer product를 빼는(write, GER) — 2장에서 배운 $dW = (\text{오차})\,x^\top$ 그 GEMM shape다. 오차가 0이면 update도 0이다. 즉 delta write는 **self-limiting**이다: 같은 쌍이 반복 제시되면 저장이 완성되는 순간 write가 멈춘다. Hebbian write가 같은 쌍을 무한히 덧쌓는 것과 대조된다.
2. **memory의 얼굴**: $\eta_t = 1$, $\|k_t\| = 1$이면 (6-3)은 "key $k_t$ 방향의 옛 내용을 정확히 지우고 $v_t$를 기록"하는 exact overwrite다. append-only cache가 진짜 read-modify-write 저장소로 바뀐다.
3. **선형대수의 얼굴**: transition $I - \eta_t k_tk_t^\top$는 **generalized Householder 변환**이다 — $k_t$ 방향의 고유값이 $1-\eta_t\|k_t\|^2$, 나머지 방향은 1. $\eta_t\|k_t\|^2\in(0,1)$이면 수축, $=2$면 반사다. gate 계열의 transition(스칼라·대각)과 달리 **비대각**이라는 사실이 chunkwise 병렬화의 난이도를 결정적으로 바꾸는데(누적 곱이 elementwise로 접히지 않는다), Yang et al. 2024 (arXiv:2406.06484)의 WY representation이 이를 chunk당 GEMM 두어 번으로 해결했다. 유도는 9장의 몫이다.

실무 구현은 key를 SiLU 통과 후 $\ell_2$ normalize하고, $\eta_t$를 sigmoid head의 출력 $\eta_t = \sigma(\cdot)\in(0,1)$ — 원문의 표현으로 "writing strength" — 인 data-dependent 값으로 만든다 (Yang et al. 2024, arXiv:2406.06484, §3.1·§3.3). $\ell_2$ 정규화는 Yang et al. 2024의 선택이다: $\|k_t\|_2=1$이면 $\eta_t=1$일 때 $I - k_tk_t^\top$가 정확한 projection이 되어 위의 exact-overwrite 해석이 성립한다(원형인 Schlag et al. 2021, arXiv:2102.11174, §4.2는 $\ell_1$ 계열 sum normalization을 썼다). 결과적으로 $\eta_t$가 "이 token을 얼마나 세게 쓸 것인가"의 학습된 per-token write intensity가 된다.

**Gated DeltaNet**(이하 GDN; Yang, Kautz & Hatamizadeh 2025, arXiv:2412.06464)은 두 계보의 합류다: 식 (6-2)의 retention과 식 (6-3)의 targeted overwrite를 한 update에 싣는다 [Titans §2; Miras Eq. 9]:

$$
W_t = \alpha_t\, W_{t-1}\big(I - \eta_t k_t k_t^\top\big) + \eta_t v_t k_t^\top .
\tag{6-4}
$$

([Miras Eq. 9]는 write 항의 $\eta_t$를 흡수한 표기이고 좌·우곱은 행벡터 관행의 차이다 — 알고리즘은 동일하다.) 의미는 정확히 "전역 eviction($\alpha_t$) + 항목별 overwrite($\eta_t$)"이며, Miras의 4축으로는 $\ell_2$ bias + $\ell_2$ retention + 1-step GD의 자리에 놓인다 [Miras Table 1]. 이 형태가 이 장 family의 사실상의 완성형이고, Titans는 여기서 optimizer 축(momentum)과 memory 구조 축(deep MLP)을 더 밀고 나간 것이다(→ 12장).

## 6.5 Longhorn: update rule을 발명하지 말고, 문제를 풀어라

지금까지의 모델들은 update rule을 **설계**했다. **Longhorn** (Liu et al. 2025, arXiv:2407.14207)의 제안은 방법론 자체의 전환이다: per-token으로 풀고 싶은 online 최적화 문제를 먼저 적고, 그 문제의 **closed-form 해**를 update rule로 삼는다. 논문의 표현으로 SSM은 "amortized online learner"다. 3장에서 배운 online learning의 프레임이 여기서 처음으로 모델 유도에 통째로 쓰인다.

문제 설정은 proximal 형태다: 새 쌍은 잘 맞추되, 이전 state에서 너무 멀어지지 말 것.

$$
W_t = \arg\min_{W}\; \|W - W_{t-1}\|_F^2 + \eta_t\, \|W k_t - v_t\|_2^2 .
$$

이차식이므로 미분해서 0으로 놓으면 $W(I + \eta_t k_tk_t^\top) = W_{t-1} + \eta_t v_tk_t^\top$, Sherman–Morrison으로 역행렬을 풀면

$$
W_t = W_{t-1}\big(I - \epsilon_t\, k_t k_t^\top\big) + \epsilon_t\, v_t k_t^\top,
\qquad
\epsilon_t = \frac{\eta_t}{1 + \eta_t\, k_t^\top k_t}.
\tag{6-5}
$$

표기 주의 두 가지. 첫째, 원문의 objective는 두 번째 항이 **출력 채널별 가중 노름**이다 — 스칼라 $\eta_t$ 자리에 채널별 벡터 $\eta_t\in\mathbb{R}^{d_v}$(sigmoid 출력; 원문 기호 $\beta_t$)가 앉고, $\eta_{t,i}=0$인 채널의 state 행은 그대로 보존된다. 닫힌 해도 그에 맞춰 출력 행별로 $\epsilon_{t,i} = \eta_{t,i}/(1+\eta_{t,i}\,k_t^\top k_t)$를 갖는다 [Longhorn Eq. 5, Thm 3.1]. 위 (6-5)는 이 행별 식에서 $\eta_t$를 스칼라로 둔 단순화이며, 유도의 본질(implicit GD, 아래 안정성 논증)은 그대로 보존된다. 둘째, 원문 구현은 parallel scan을 위해 transition $I - \epsilon_{t,i}\,k_tk_t^\top$를 대각 근사 $\mathbf{1} - \epsilon_{t,i}\,k_t^{\odot 2}$로 치환한다 [Longhorn §3.2] — 즉 출하된 Longhorn kernel은 delta 계열의 Householder형 transition이 아니라 그 **대각 근사판**이다. 근사의 대가로 얻는 것은 병렬화 형태다: full transition은 비대각이라 WY 계열 chunkwise 기법을 요구하지만, 대각 transition은 GLA류 scan으로 접힌다(→ 9장). 이하의 분석은 근사 전의 full transition에 대한 것이다.

형태는 DeltaNet (6-3) 그대로이고, 바뀐 것은 learning rate의 재정의 하나다. 그러나 이 하나가 optimizer의 **종류**를 바꾼다. (6-3)은 gradient를 $W_{t-1}$에서 평가하는 explicit GD(forward Euler)이고, (6-5)는 gradient를 도착점 $W_t$에서 평가하는 방정식 $W_t = W_{t-1} - \eta_t\nabla_W\ell(W_t;k_t,v_t)$을 푼 **implicit gradient descent**(backward Euler, proximal step)다. Miras Table 1이 Longhorn에만 "Implicit GD"라는 별도의 행을 준 이유다 [Miras Table 1].

implicit의 값어치는 무조건 안정성이다. $k_t$ 방향의 transition 계수를 비교하면:

- explicit (6-3): $1 - \eta_t\|k_t\|^2$. $\eta_t\|k_t\|^2 > 2$면 절댓값이 1을 넘어 state가 폭주한다. 안정성이 step size 제약이라는 형태로 사용자에게 전가된다.
- implicit (6-5): $1 - \epsilon_t\|k_t\|^2 = \dfrac{1}{1+\eta_t\|k_t\|^2} \in (0,1)$ — **임의의 $\eta_t > 0$에서** 안정하다. $\eta_t \to \infty$ 극한에서도 발산하지 않고 "이 key에 대해 $v_t$를 정확히 저장하라"는 hard write로 수렴할 뿐이다.

training 무경험 독자를 위해 옮기면: explicit GD의 step size는 발산이라는 절벽이 있는 tuning 대상이고, implicit GD는 그 절벽을 수식 안에서 제거한 것이다. "임의의 $\eta_t>0$에서 안 터진다"는 것은 closed-form의 수학적 성질이라, gate head가 어떤 값을 내놓아도 안정성이 보장되어 step-size 안정성이 outer-loop 학습에서 분리된다 — 실제 Longhorn은 여기에 더해 $\eta_t$(원문 $\beta_t$)를 sigmoid로 $(0,1)$에 가두지만, 안정성 자체는 그 제한과 무관한 closed-form의 성질이다 [Longhorn §3.2]. retention gate는 없다($\alpha \equiv 1$): proximal 항 $\|W - W_{t-1}\|_F^2$ 자체가 유일한 (local) retention이며, 전역 감쇠 없이도 위의 수축 계수가 오래된 내용을 서서히 밀어낸다. systems 관점의 비용은 정직하게 0에 가깝다 — $\epsilon_t$ 계산은 내적 하나와 나눗셈 하나이고, kernel 구조는 DeltaNet과 동일하다. "공짜 안정성"이라는 점이 이 모델의 systems 요약이다.

## 6.6 RWKV-7: generalized delta rule — gate를 채널별 벡터로

**RWKV-7 "Goose"** (Peng et al. 2025, arXiv:2503.14456)는 이 장 family의 현재 시점 최종 일반화다. GDN (6-4)의 스칼라 손잡이들을 전부 벡터로 승격한다: 스칼라 retention $\alpha_t$는 채널별 decay 벡터 $w_t\in(0,1)^{d_k}$로, 스칼라 write intensity $\eta_t$는 채널별 **in-context learning rate** 벡터 $a_t\in[0,1]^{d_k}$로 승격되고, 지우는 key와 쓰는 key가 **같은 key의 서로 다른 채널별 변조**로 분리된다: 제거용 $\hat k_t$는 $k_t$에 학습된 채널별 배율(원문의 removal-key multiplier)을 $\odot$로 곱한 뒤 head별 $\ell_2$ 정규화한 것이고, 기록용 $\tilde k_t$는 $k_t$의 각 채널을 $a_t$ 방향으로 보간한 것이다 — $\tilde k_t = k_t \odot \mathrm{lerp}(\mathbf{1}, a_t, \nu)$, $\nu$는 학습된 보간 계수(원문 표현 "replacement rate booster") [RWKV-7 Eq. 6–7, Eq. 15]. 별도의 projection 행렬을 더 두는 것이 아니라 같은 key precursor를 채널 단위로 두 갈래 성형하는 것이므로, projection 비용은 그대로다. 구조적으로 쓰면

$$
W_t = W_{t-1}\Big(\mathrm{Diag}(w_t) - \hat k_t\,\big(a_t \odot \hat k_t\big)^\top\Big) + v_t\, \tilde k_t^\top .
\tag{6-6}
$$

(6-6)은 원문의 state evolution [RWKV-7 Eq. 17]을 행벡터 관행에서 이 책의 열벡터 관행으로 전치한 것으로, $a_t$가 제거 항 내부에 $\odot$로 곱해지는 위치까지 원문과 배치가 같다. 제거 key를 $\ell_2$ 정규화해 두는 이유도 원문이 명시한다: 제거량을 단위 norm으로 고정해 두면 in-context learning rate $a_t$가 "state에서 얼마나 지우고 얼마나 다시 써 넣는가"를 다른 항에 오염되지 않고 단독으로 조절하는 손잡이가 된다 [RWKV-7 Eq. 7 부근]. GDN에서는 지우기 강도와 쓰기 강도가 $\eta_t$ 하나에 묶여 있었다 — (6-6)은 그 묶음을 채널 단위로 풀어낸 것이다.

$w_t = \alpha_t\mathbf{1}$, $a_t = \alpha_t\eta_t\mathbf{1}$, $\hat k_t = k_t$로 두면 transition이 $\alpha_t(I-\eta_t k_tk_t^\top)$로 (6-4)의 첫 항과 정확히 일치한다. 단 write 항까지 맞추려면 $\tilde k_t = \eta_t k_t$여야 하는데($v_t\tilde k_t^\top = \eta_t v_tk_t^\top$가 되도록), RWKV-7의 $\tilde k_t = k_t\odot\mathrm{lerp}(\mathbf 1,a_t,\nu)$가 임의의 GDN을 정확히 재현한다는 보장은 원문에 없다. 따라서 (6-6)은 GDN의 transition 구조를 **일반화**하는 것으로 읽되, 모든 GDN을 부분 경우로 정확히 포함한다는 단정은 유보한다. Miras의 분류로는 delta 계열에서 gate가 채널별 벡터($m=d$)인 경우가 정확히 RWKV-7이다 [Miras Eq. 9, §4]. transition이 "대각 - rank-1"이라는 사실에 주목하라. gate 계열(순수 대각)과 delta 계열(항등 - rank-1)의 합집합이며, 이 구조 덕에 chunkwise 병렬화는 DeltaNet과 같은 WY 계열 기법으로 처리된다(→ 9장).

"이 gate는 누가 학습하는가"라는 §6.2의 질문을 (6-6)에 적용하면 답은 모두 slow weights로 귀결되나, 두 부류를 구분해야 한다: $w_t$·$a_t$는 작은 head가 token마다 산출하는 data-dependent 값이고($a_t$도 sigmoid 계열 산출의 $[0,1]^{d_k}$ 벡터다 [RWKV-7 Eq. 4]), 두 key 변조의 채널 배율 $\xi$·보간 계수 $\nu$는 outer loop가 학습하는 token-independent 고정 파라미터다. 어느 쪽이든 slow weights이고, inner loop에서 움직이는 것은 여전히 $W_t$ 하나뿐이다. 손잡이 수가 늘었을 뿐 두 시간 척도의 구도는 (6-1)에서 한 치도 달라지지 않았다.

정리하면 RWKV-7은 이 장의 gate·delta 계열을 두 축의 격자로 읽게 한다. 한 축은 retention의 해상도(상수 → data-dependent 스칼라 → 채널별 벡터)이고, 다른 축은 write의 정밀도(Hebbian additive → delta overwrite → 지우기·쓰기를 분리한 채널별 overwrite)다. RetNet은 retention 축으로만 한 칸, DeltaNet은 write 축으로만 한 칸 움직였고, GDN은 두 축에서 스칼라 한 칸씩 오른 지점이며, RWKV-7은 두 축을 모두 벡터 해상도까지 민 오른쪽 위 모서리다. Longhorn만은 이 격자 밖에 선다 — 좌표를 옮긴 것이 아니라 explicit GD를 implicit GD로 갈아 끼워 step size의 안정성 자체를 다시 정의했기 때문이다(§6.5). Miras가 이 장의 모델을 한 판에 담을 수 있는 것도 격자 위 좌표와 격자 밖 한 점이라는 이 구조 덕이다 [Miras Table 1, §4]. 이 평면은 여기서 닫힌다 — Titans 이후의 확장(→ 12장)은 격자를 더 촘촘히 하는 대신 optimizer 축(momentum)과 memory 구조 축(deep MLP)이라는 두 개의 새 차원을 여는 일이다.

> **[해설]** (6-6)을 inference 어휘로 옮기면 이렇다. 채널별 decay $w_t$는 cache 항목의 TTL이 **채널마다 다르게** 설정되는 eviction이고, 제거용 $\hat k_t$와 기록용 $\tilde k_t$의 분리는 같은 cache line에 대한 invalidate mask와 write mask를 따로 가진다는 뜻이며, $a_t$는 채널별 write intensity다. GDN이 "line 단위 eviction + line 단위 overwrite"였다면 RWKV-7은 그 두 연산 모두에 **byte-enable 신호**를 단 것이다 — 제어 자유도는 벡터로 늘었지만, 저장소($d_v\times d_k$ 행렬 하나)와 지배 비용(state RMW)은 그대로다.

표현력에 관한 원 논문의 주장도 이 transition 구조에서 나온다: [RWKV-7] 논문은 generalized delta rule이 (표준 복잡도 가정 하에) transformer의 $TC^0$ 한계를 넘는 state tracking을 가능하게 한다고 주장한다 — 정확한 정리 진술과 그 단서는 다음 절에서, 이 주장이 어느 이론 지형 위에 놓여 있는지와 함께 본다.

systems 접점: state는 여전히 head당 $d_v\times d_k$ 하나이고 decode의 지배 비용도 그대로다. 추가되는 것은 gate·learning-rate·key 변조를 만들어내는 여러 개의 작은 head — RWKV-7 계보의 관행대로 low-rank projection — 이며, 이는 decode당 skinny GEMM 몇 개, 즉 지배 항 대비 소액이다. **비용은 GDN급 그대로 두고 update rule의 자유도만 올린 설계**라는 것이 systems 한 줄 요약이다.

이 장의 모델들을 한 표로 모은다. 각 행이 "무엇을 바꿨는가"에 답하는지 확인하라 — 전부 식 (M)의 성분 선택이다.

표 6-2 — 이 장의 모델 카탈로그 (통일 표기; read는 모두 $y_t = W_t q_t$).

| 모델 | update | inner objective | retention | inner optimizer |
|---|---|---|---|---|
| linear attention | $W_t = W_{t-1} + v_tk_t^\top$ (6-1) | dot-product | 없음 | 1-step GD (Hebbian) |
| RetNet | $W_t = \alpha W_{t-1} + v_tk_t^\top$ | dot-product | 상수 decay | 1-step GD |
| GLA | $W_t = W_{t-1}\mathrm{Diag}(\alpha_t) + v_tk_t^\top$ | dot-product | 학습된 diagonal | 1-step GD |
| Mamba-2 (→ 7장) | $W_t = \alpha_t W_{t-1} + v_tk_t^\top$ | dot-product | 학습된 scalar | 1-step GD |
| DeltaNet | (6-3) | $\ell_2$ regression | 없음 | 1-step GD |
| GDN | (6-4) | $\ell_2$ regression | 학습된 scalar | 1-step GD |
| Longhorn | (6-5) | $\ell_2$ regression | 없음 ($\alpha\equiv 1$) | implicit GD |
| RWKV-7 | (6-6) | $\ell_2$ regression | 채널별 vector | 1-step GD (generalized delta) |
| TTT-Linear (→ 8장) | (M1) | $\ell_2$ regression | 없음 | 1-step GD |
| Titans-LMM (→ 12장) | (M2) | $\ell_2$ (deep memory) | $\alpha_t$ + momentum | GD + momentum |

## 6.7 Expressivity 사이드바: state의 착각과 음의 고유값

이 절은 반 페이지짜리 지도다 — Titans·Atlas가 "deep memory"를 주장할 때 딛고 서는 이론 지형이 여기 있다.

Merrill et al. 2024 (arXiv:2404.08819)는 대각 transition의 SSM이 (log-precision 가정 하에) transformer와 같은 회로 복잡도 계열 $TC^0$에 머문다고 주장한다 — 겉보기에 재귀적 state가 있어도 순차 계산 고유의 문제(예: $S_5$ 치환 합성 같은 $NC^1$-complete state tracking)를 풀 수 없다는, 논문 제목 그대로 "illusion of state"다. 반면 Grazzi et al. 2025 (arXiv:2411.12537)는 DeltaNet류의 비대각 transition에서 $\eta_t$의 허용 범위를 $(0,2)$로 넓혀 transition 고유값이 $[-1,1]$ 전체를 덮게 하면 — 즉 **음의 고유값**을 허용하면 — parity 같은 state-tracking 문제가 풀리게 됨을 보였다. DeltaProduct (Siems et al. 2025, arXiv:2502.10297)는 token당 GD를 여러 step 밟아(Householder 곱; [Miras Table 1]의 multi-step GD 행) transition의 rank 자체를 올린다.

§6.6이 예고한 RWKV-7의 표현력 주장은 정확히 이 지형 위에 놓인다. 원 논문의 정리는 둘이다. 첫째, **단일 layer** RWKV-7이 $S_5$ 원소 5개의 swap tracking — $AC^0$ 환원 하에서 $NC^1$-complete인 문제 — 을 푼다 [RWKV-7 Thm 2, App. D.1]. 둘째, 임의의 정규 언어에 대해 그것을 인식하는 **4-layer** RWKV-7 모델이 존재한다 [RWKV-7 Thm 3, App. D.2]. 두 정리 모두 $TC^0 \ne NC^1$ conjecture를 전제로 "transformer가 못 하는 것을 한다"로 읽힌다. 그리고 단서 하나가 결정적이다: 증명은 (6-6)의 제거 항에 계수 $2$를 둔 변형 — $W_{t-1}\big(\mathrm{Diag}(w_t) - 2\,\hat k_t(a_t\odot\hat k_t)^\top\big)$ 꼴, 즉 transition 고유값 $-1$을 허용하는 판 — 에 대한 것이다 [RWKV-7 App. D.1]. 출하 아키텍처(계수 $1$)가 아니라 음의 고유값을 허용한 변형에 대한 결과이며, 이는 Grazzi et al.의 $\eta_t\in(0,2)$ 확장과 같은 기제다. "RWKV-7이 $TC^0$를 넘는다"를 옮길 때는 이 단서 — layer 수, complexity conjecture, 그리고 계수 $2$ 변형 — 를 함께 옮겨야 정직한 문장이 된다.

> **[해설]** 이 지형이 6편에 주는 함의는 다음과 같다. transition의 구조(대각 < 대각+rank-1 < 그 곱)가 곧 모델이 표현할 수 있는 state 동역학의 계급이고, 이 장의 계보는 그 사다리를 한 칸씩 오르는 과정이기도 했다. Titans 라인은 여기서 한 축을 더 꺾는다 — transition을 더 꾸미는 대신 memory 자체를 nonlinear(deep MLP)로 만들고 read를 $\mathcal{M}(q;W)$로 비선형화하는 방향이다(→ 8장, 12장). matrix memory의 read가 $q$에 대해 선형이라는 제약은 어떤 gate로도 벗겨지지 않기 때문이다.

systems 접점 하나: 표현력 사다리는 공짜가 아니라 **병렬화 예산**과 교환된다. 대각 transition은 scan으로, rank-1은 WY로 병렬화되지만, rank가 오르고 비선형이 끼어들수록 chunk 경계의 순차 의존이 두꺼워진다. 사다리의 두 칸이 이 교환의 양극단을 보여준다. Grazzi et al.의 $\eta_t\in(0,2)$ 확장은 transition 고유값 범위만 $[-1,1]$로 넓힐 뿐 여전히 rank-1이므로 WY 기법이 그대로 통해 표현력을 거의 공짜로 산다. 반대로 DeltaProduct는 token당 GD를 여러 step 밟아 transition rank를 올리는 대신, chunk 안에서 Householder 곱이 그만큼 순차로 쌓여 병렬화 GEMM 폭이 두꺼워지는 비용을 치른다 [Siems et al. 2025]. state 동역학의 계급을 한 칸 올릴 때마다 tensor core를 채우는 형태가 조금씩 나빠진다는 이 교환이 이 라인의 kernel 설계 전체를 관통하며, 그 정식 무대가 9장이고 극한이 TNT다(→ 15장).

## 6.8 Worked micro-example: $d=2$에서 세 가지 write를 손으로 돌린다

$d_k = d_v = 2$. 저장할 두 쌍은

$$
k_1 = \begin{pmatrix}1\\0\end{pmatrix},\; v_1 = \begin{pmatrix}2\\0\end{pmatrix},
\qquad
k_2 = \begin{pmatrix}0.6\\0.8\end{pmatrix},\; v_2 = \begin{pmatrix}0\\1\end{pmatrix}.
$$

두 key 모두 단위 norm이고 $k_1^\top k_2 = 0.6$ — 일부러 직교시키지 않았다. $W_0 = 0$에서 시작한다.

**(a) Hebbian (6-1).** $W_1 = v_1k_1^\top = \begin{pmatrix}2&0\\0&0\end{pmatrix}$, $W_2 = W_1 + v_2k_2^\top = \begin{pmatrix}2&0\\0.6&0.8\end{pmatrix}$.

읽기: $W_2k_1 = (2,\,0.6)^\top$ — 참값 $v_1=(2,0)^\top$ 대비 둘째 채널에 $0.6$의 crosstalk. 정확히 $v_2(k_2^\top k_1)$, 즉 5장의 crosstalk 공식 그대로다. $W_2k_2 = (1.2,\,1.0)^\top$ — 이번엔 옛 쌍이 새 key의 읽기를 오염시킨다($v_1(k_1^\top k_2) = (1.2,0)^\top$). 어느 key도 정확히 회수되지 않는다.

**(b) DeltaNet (6-3), $\eta_t = 1$.** step 1은 $W_0=0$이라 Hebbian과 동일: $W_1 = \begin{pmatrix}2&0\\0&0\end{pmatrix}$. step 2에서 먼저 **읽는다**: 예측 $W_1k_2 = (1.2,\,0)^\top$, 오차 $e = W_1k_2 - v_2 = (1.2,\,-1)^\top$. 오차의 outer product를 뺀다:

$$
W_2 = W_1 - e\,k_2^\top
= \begin{pmatrix}2&0\\0&0\end{pmatrix} - \begin{pmatrix}0.72&0.96\\-0.6&-0.8\end{pmatrix}
= \begin{pmatrix}1.28&-0.96\\0.6&0.8\end{pmatrix}.
$$

읽기: $W_2k_2 = (1.28\cdot 0.6 - 0.96\cdot 0.8,\; 0.36+0.64)^\top = (0,\,1)^\top = v_2$ — **exact**. 최신 key는 완벽히 저장된다. 대신 $W_2k_1 = (1.28,\,0.6)^\top$: 옛 쌍은 새 key와 겹치는 성분만큼 수정됐고, 오차 크기는 $\approx 0.94$로 Hebbian의 $0.6$보다 오히려 크다. 이것은 버그가 아니라 semantics다 — delta rule이 보장하는 것은 **최신 binding의 정확성**(마지막으로 쓴 값이 이긴다)이지 과거의 보존이 아니다. 과거 보존은 retention 축의 몫이고($\alpha_t$, 13장), 같은 쌍들이 반복 제시되면(여기처럼 선형독립·consistent한 경우) LMS로서 그 해에 수렴한다(→ 5장; inconsistent 계에서는 감소 step size가 필요하다). 두 key가 직교했다면 (a)와 (b) 모두 오차가 0이었음을 직접 확인해 보라 — 간섭의 원천은 오직 $k_1^\top k_2 \ne 0$이다.

**(c) Longhorn (6-5).** $\eta_2 = 1$이면 $\epsilon_2 = 1/(1+1) = 0.5$: 읽기 $W_2k_2 = 0.5\,(1.2,0)^\top + 0.5\,(0,1)^\top = (0.6,\,0.5)^\top$ — 명목 $\eta$가 같아도 절반만 쓴다. step size를 키우면: $\eta_2 = 4$일 때 explicit (6-3)은 $k_2$ 방향 계수가 $1-4 = -3$이 되어 읽기가 $(-3.6,\,4)^\top$으로 폭주하지만, implicit은 $\epsilon_2 = 4/5 = 0.8$로 $(0.24,\,0.8)^\top$ — $v_2$에 안정적으로 접근한다. $\eta_2\to\infty$ 극한에서 $\epsilon_2 \to 1$, 즉 (b)의 exact overwrite로 수렴한다.

**(d) gate의 시간 상수 감각.** RetNet처럼 $\alpha = 0.9$ 상수면 100 token 뒤 첫 write의 잔존 계수는 $0.9^{99} \approx 3\times 10^{-5}$ — eviction의 반감기가 $\log 2 / \log(1/0.9) \approx 6.6$ token이다. gate 값이 곧 cache 항목의 TTL이라는 대응을 숫자로 확인할 수 있다. 채널별 gate의 값어치도 같은 산수로 보인다: RWKV-7류 decay 벡터가 $w = (0.9,\; 0.999)$라면 첫째 채널의 반감기는 6.6 token, 둘째 채널은 $\log 2/\log(1/0.999) \approx 693$ token — 같은 state 행렬 안에 **두 자릿수 차이의 시간 척도**가 채널 단위로 공존한다. RetNet이 head 단위로만 만들 수 있던 다중 시간 척도를 (6-6)은 채널 단위로, 그것도 token마다 다시 정해서 만든다.

표 6-3 — micro-example 결과 요약 (읽기 오차의 $\ell_2$ norm).

| write 방식 | $q=k_1$ 오차 | $q=k_2$ 오차 | 성질 |
|---|---|---|---|
| Hebbian (6-1) | 0.60 | 1.20 | 둘 다 부정확, 감쇠 기제 없이 norm 누적 |
| DeltaNet (6-3), $\eta=1$ | 0.94 | **0 (exact)** | 최신 binding 우선, self-limiting |
| Longhorn (6-5), $\eta=1$ | — | 0.78 | 절반 write, 무조건 안정 |

## 6.9 Systems bridge: 당신의 roofline 위에 올려놓기

이제 이 family를 독자의 모국어 — bytes, FLOPs, crossover — 로 결산한다. head당, bf16(2 bytes), $d_k = d_v = d_h$ 기준이다.

표 6-4 — per-token decode 비용 (per head; $L$ = 현재 문맥 길이, $w$ = window 크기).

| 구조 | 상주 state (원소 수) | decode FLOPs/token | decode traffic/token | prefill/train 병렬 형태 |
|---|---|---|---|---|
| softmax attention | $L(d_k{+}d_v)$ — 증가 | $\approx 4Ld_h$ | KV 전체 read $\approx 2L(d_k{+}d_v)$ B | attention GEMM (exact tiling) |
| sliding-window attn | $w(d_k{+}d_v)$ | $\approx 4wd_h$ | $\approx 2w(d_k{+}d_v)$ B | 동일, window 마스크 |
| linear attn / RetNet / GLA | $d_kd_v$ — 고정 | $\approx 4$–$6\,d_kd_v$ | state RMW $\approx 4\,d_kd_v$ B | chunkwise GEMM + (scan) (→ 9장) |
| DeltaNet / GDN / Longhorn | $d_kd_v$ — 고정 | $\approx 6$–$8\,d_kd_v$ | 동일 | chunkwise GEMM + WY (→ 9장; 출하 Longhorn kernel은 대각 근사 scan — §6.5) |
| RWKV-7 | $d_kd_v$ + gate head들 | 상동 + low-rank head | 동일 | 동일 계열 |

세 가지 계산을 직접 해보면 이 표가 몸에 붙는다.

**첫째, crossover.** state가 KV cache보다 작아지는 지점은 $L^* = \dfrac{d_kd_v}{d_k+d_v}$이다. $d_k=d_v=128$이면 $L^* = 64$ token — 고작 64 token만 넘으면 고정 state가 이긴다. 단, 독자의 스택이 GQA로 KV head를 8:1로 줄이고 있다면 KV의 token당 발자국이 8× 작아져 crossover는 512 token으로 밀린다. "linear attention은 언제나 메모리 이득"이 아니라 **자기 서빙 구성에 대입해야 하는 산수**라는 뜻이다. 모델 전체 감각: 32 layer × 32 head × $d=128$이면 state는 $32\cdot 32\cdot 128\cdot 128\cdot 2\,\mathrm{B} = 32\,\mathrm{MiB}$로 문맥 길이와 무관한 상수이고, 같은 구성의 MHA KV cache는 token당 512 KiB — 4K 문맥에서 이미 2 GiB다.

**둘째, roofline 위치.** DeltaNet decode 한 step은 head당 FLOPs $\approx 6\,d_kd_v$, state RMW traffic $\approx 4\,d_kd_v$ B로 산술 강도가 $\sim 1.5$ FLOP/B다. softmax attention decode의 강도도 $\sim 1$ FLOP/B 수준 — **둘 다 깊은 bandwidth-bound이고, linear 계열의 승리는 강도 개선이 아니라 traffic 절대량의 상한**이다. tensor core를 놀리지 않는 형태(GEMM-rich)로 바꾸는 것은 decode가 아니라 prefill/training의 chunkwise 재구성이 하는 일이며(→ 9장), 이 구분 — decode는 bandwidth 문제, training은 utilization 문제 — 이 이 라인의 systems 논의 전체를 가로지른다.

**셋째, batching.** 식 (6-1)~(6-6)의 read/write는 요청마다 다른 $W_t$에 대한 GEMV/GER이므로, weight가 공유되는 보통의 batched GEMM과 달리 **state 축이 batch에 들어온다**. batch $B$의 decode는 $B$개의 독립 GEMV — 사실상 batched/grouped GEMV — 가 되고, per-request state $B\cdot 32\,\mathrm{MiB}$가 HBM 상주분으로 잡힌다. KV cache 관리자가 하던 일(할당, 상주, 축출, 요청 간 격리)을 fast-weight state 관리자가 그대로 물려받는 그림이며, 이 문제의식은 TTT 계열에서 backward pass까지 decode에 들어오면서 본격화된다(→ 8장, 10장).

## 요약

- linear attention은 kernel trick으로 softmax attention의 KV cache를 head당 $d_v\times d_k$ 고정 행렬 $W_t$로 손실 압축한 것이다. write는 rank-1 outer product, read는 GEMV다 [Titans Eq. 3–5].
- $W_t$는 fast weights — inner loop에서 token마다 움직이는 상태 — 이고, projection·gate head 등 $\Theta$는 slow weights다. linear attention = fast-weight programming이라는 동치(Schlag et al. 2021)가 이 라인의 역사적 기점이다.
- Hebbian write는 dot-product bias의 1-step GD라서 자기 제한이 없고 crosstalk가 누적된다. 교정 축은 둘이다: retention(RetNet의 상수 → GLA·Mamba-2의 data-dependent gate)과 write 교정(delta rule) [Titans §2; Miras §4].
- DeltaNet의 update (6-3)은 $\ell_2$ regression의 1-step GD이며 transition은 generalized Householder다. GDN (6-4)은 retention과 overwrite를 결합한 이 family의 완성형이다.
- Longhorn (6-5)은 같은 문제의 closed-form proximal 해 = implicit GD로, 임의의 $\eta_t>0$에서 무조건 안정이다. update rule 설계가 "online 문제 선택 + optimizer 선택"으로 대체될 수 있음을 보인 사례다.
- RWKV-7 (6-6)은 스칼라 gate들을 채널별 벡터로 일반화한 generalized delta rule이다 [RWKV-7 Eq. 17]. 지우는 key와 쓰는 key는 별도 projection이 아니라 같은 key의 채널별 변조이며 [RWKV-7 Eq. 6–7, Eq. 15], 표 6-2의 전 모델이 식 (M)의 성분 선택 하나씩으로 구분된다 [Miras Table 1].
- 원문 이론·구현의 단서 두 가지: 출하된 Longhorn kernel은 delta transition의 대각 근사판이고 [Longhorn §3.2], RWKV-7의 $TC^0$ 초과 정리(1-layer $S_5$ tracking, 4-layer 정규 언어)는 제거 항 계수 $2$ — 음의 고유값 허용 — 변형에 대한 것이다 [RWKV-7 Thm 2–3, App. D].
- decode에서 이 family의 이득은 산술 강도가 아니라 traffic 상한이다. crossover($L^* = d_kd_v/(d_k{+}d_v)$)와 per-request state 상주는 자기 서빙 구성으로 계산해봐야 하는 산수다.

## 자가 점검 체크리스트

- [ ] softmax attention에서 출발해 식 (6-1)의 재귀형을 유도하고, 어디서 근사가 들어갔는지(kernel 교체) 지목할 수 있다.
- [ ] 표 6-2의 각 행에 대해 "inner objective가 무엇이고, optimizer가 무엇이고, retention이 무엇인가"에 즉답할 수 있다.
- [ ] $d=2$ 예제를 백지에서 재계산해 Hebbian의 crosstalk와 delta의 exact overwrite(그리고 옛 key의 손상)를 재현할 수 있다.
- [ ] Longhorn의 $\epsilon_t = \eta_t/(1+\eta_t k_t^\top k_t)$를 Sherman–Morrison으로 유도하고, explicit GD와의 안정성 차이를 transition 고유값으로 설명할 수 있다.
- [ ] "이 gate($\alpha_t, \eta_t, a_t$)는 누가 학습하는가?"에 fast/slow weights 구분으로 답할 수 있다.
- [ ] 이 장의 내용을 inference 어휘로 옮길 수 있다: state = 압축된 KV cache, retention gate = 학습된 eviction, delta write = read-modify-write, decode 비용 = 고정 크기 state의 RMW traffic.

## 다음 장으로

이 장은 linear attention 쪽 계보만 따라왔지만, 같은 시기에 전혀 다른 출발점 — 연속 시간 state space model — 에서 정확히 같은 지점에 도착한 계보가 있다. S4에서 Mamba로, 그리고 Mamba-2에 이르러 두 계보가 수학적으로 동일함이 증명된다(state-space duality). 7장은 그 합류를 다룬다: 왜 Mamba-2의 gate가 표 6-2의 $\alpha_t$ 행에 앉아 있는지, 그리고 scan 기반 계보가 어떻게 GEMM 기반 chunkwise 형태로 수렴하는지. 그 다음 8장에서 비로소 질문이 뒤집힌다 — state를 행렬이 아니라 **작은 모델의 weights 전체**로 만들면 어떻게 되는가?
