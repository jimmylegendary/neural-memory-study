# ch05. Associative memory: Hopfield에서 delta rule까지

> **이 장의 목표** — 이 장을 마친 독자는 다음을 할 수 있다.
> 1. KV cache와 matrix memory를 하나의 associative memory 추상화의 양 극단(무손실 non-parametric vs 고정 크기 parametric)으로 배치하고, 그 trade-off를 byte와 FLOP 수치로 말할 수 있다.
> 2. outer-product write의 crosstalk을 $d=2$ 예제에서 손으로 계산하고, 이것이 [Titans §2] (*Titans: Learning to Memorize at Test Time*, arXiv:2501.00663)가 말하는 "memory overflow"의 미시적 실체임을 설명할 수 있다.
> 3. delta rule을 $\ell_2$ associative loss의 1-step gradient descent로 유도하고, 그것이 식 (M1)의 linear memory 전개형임을 보일 수 있다.
> 4. Hopfield → dense Hopfield → softmax attention으로 이어지는 capacity 개선의 사슬을 서술하고, 그 사슬 위에서 [Atlas §3.1] (*Atlas: Learning to Optimally Memorize the Context at Test Time*, arXiv:2505.23735)의 capacity 정리와 feature map이 왜 그런 모양인지 예측할 수 있다.
>
> **왜 필요한가** — associative memory는 이 라인이 스스로 선언한 기초 추상화다. [Miras Def. 3.1] (*It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization*, arXiv:2504.13173 — 이 책은 framework 이름 Miras로 통칭한다)과 [NL Def. 1] (*Nested Learning: The Illusion of Deep Learning Architecture*, arXiv:2512.24695)은 sequence model 전체를 "objective 아래에서 학습되는 mapping $\mathcal{M}: K \to V$"로 정의하고, [Titans §2]는 linear attention의 additive write가 "memory overflow"를 일으킨다는 비판 — 곧 이 장의 crosstalk 논의 — 위에 momentum과 gate를 쌓는다. [Atlas §3.1]의 capacity 정리(Prop 1, Thm 1, Prop 2)와 polynomial/exponential feature map은 고전 Hopfield capacity와 dense Hopfield의 energy 사슬을 그대로 재사용하며, Atlas와 TNT는 related work에서 자신들의 정식화가 Hopfield 1982의 associative memory 개념에 "architecturally founded"되어 있다고 같은 문장으로 명시한다 [Atlas App. A; TNT App. A] ([TNT] = *TNT: Improving Chunkwise Training for Test-Time Memorization*, arXiv:2511.07343). [NL §1]은 "backprop과 momentum까지 전부 associative memory"라는 확장으로 이 추상화를 라인의 끝까지 밀어붙인다. 이 장이 없으면 Part II의 어떤 정의도 출발점을 갖지 못한다.

## 5.1 key→value 연상: 독자가 이미 매일 서빙하는 연산

독자는 associative memory를 이 장에서 처음 배우는 것이 아니라 이미 운영하고 있다. decode 한 step에서 attention이 하는 일 — query $q_t$를 cache에 쌓인 key들과 대조해서 연관된 value를 꺼내 오는 것 — 이 정확히 연상(association)이다. **associative memory**는 key 집합 $K \subseteq \mathbb{R}^{d_k}$와 value 집합 $V \subseteq \mathbb{R}^{d_v}$ 사이의 mapping을 저장했다가, query가 주어지면 연관된 value를 복원하는 시스템이다. 이 라인은 이 고전적 개념을 그대로 정식화해서 출발점으로 삼는다: [Miras Def. 3.1]은 associative memory를 operator $\mathcal{M}: K \to V$로 정의하고 그 mapping을 학습시키는 objective를 attentional bias라고 부르며(정식화의 소유는 13장), [NL Def. 1]도 같은 정의를 채택한다. 이 장은 그 정의가 딛고 선 고전 — 1960–80년대의 correlation matrix memory, Hopfield network, delta rule — 을 다룬다.

associative memory는 두 축으로 분류된다. 첫째 축은 **무엇을 연상하는가**다. key와 value가 서로 다른 대상이면 **hetero-associative**(예: 이름 → 얼굴, key 벡터 → value 벡터), 손상된 pattern으로부터 그 pattern 자신을 복원하면 **auto-associative**다. KV cache도, 이 라인의 memory도 전부 hetero-associative이고, §5.3의 Hopfield network는 auto-associative다. 둘째 축은 **어떻게 저장하는가**다. 모든 쌍을 그대로 분리 보관하면 non-parametric(KV cache가 정확히 이것이다), 고정 크기 parameter 덩어리 하나에 겹쳐 쓰면 parametric이다. 이 장의 주인공은 후자다. 표기는 책 표준을 따른다: memory 상태는 $W$, 읽기는 $y = \mathcal{M}(q; W)$이며, 이 장의 memory는 대부분 linear여서 $\mathcal{M}(q; W) = Wq$다.

이 장의 위치도 분명히 하자. 2–4장은 훈련의 커리큘럼이었다: gradient와 backward pass, optimizer라는 stateful 객체, online 프로토콜과 regret, 그리고 두 loop. 이 장은 기억의 커리큘럼의 출발점이다: 무엇을 저장하고, 어디에, 몇 개나 담기고, 넘치면 어떻게 되는가. 두 커리큘럼은 §5.5에서 하나로 합류한다 — 좋은 write rule이 정확히 1-step gradient descent이기 때문이다. 그 합류 지점이 식 (M1)이고, 이 라인의 여섯 논문은 전부 그 지점 위에 서 있다.

systems 관점에서 이 분류를 다시 읽으면, long context 문제는 저장 방식 선택의 문제가 된다. non-parametric 저장은 정확하지만 state가 $O(L)$로 자란다 — KV cache의 byte 수를 매일 계산하는 독자에게 새로운 이야기가 아니다. parametric 저장은 state가 $O(1)$로 고정되는 대신 어딘가에서 반드시 정보를 잃는다. 그래서 이 장의 질문은 이것이다: **고정 크기 $W$에 (k, v) 쌍을 몇 개나, 얼마나 정확히 넣을 수 있고, 넘치면 어떤 방식으로 망가지는가.** 이 질문의 두 이름이 capacity와 crosstalk이다.

한 가지 주의를 미리 두자. Rosetta-Stone 사전(→ 1장)의 대응에는 성격이 있고, "동일"인 것과 "유비"인 것을 섞으면 잘못된 직관이 이식된다. attention lookup ↔ memory read는 같은 추상화의 재서술이므로 안심하고 기대도 된다. 그러나 KV cache append ↔ memory write는 유비이고, 그 차이가 이 장의 논점 그 자체다: cache는 append-once라서 서로 다른 entry가 간섭하지 않지만, 이 장의 write는 같은 좌표 공간 위에 겹쳐 쓴다 — 처음에는 blind하게(§5.2), 나중에는 read-modify-write로(§5.5). 간섭이 없는 세계에서 온 독자가 이 장에서 배워야 하는 것이 바로 간섭의 물리다.

## 5.2 outer-product memory: 행렬 하나에 여러 쌍을 겹쳐 쓰기

parametric 저장의 가장 오래된 답은 각 쌍을 outer product로 만들어 전부 더하는 것이다(**correlation matrix memory**, Kohonen 1972, IEEE Trans. Computers; Anderson 1972). $m$개의 쌍 $(k_i, v_i)$를 저장하면

$$
W \;=\; \sum_{i=1}^{m} v_i k_i^\top \;\in\; \mathbb{R}^{d_v \times d_k}
\tag{5-1}
$$

이고, 읽기는 GEMV 한 번이다. key들이 unit norm이라고 하고 $q = k_j$로 읽으면

$$
W k_j \;=\; v_j \;+\; \sum_{i \neq j} \big(k_i^\top k_j\big)\, v_i .
\tag{5-2}
$$

첫 항이 원하는 value이고, 둘째 항이 **crosstalk**이다: 다른 key와의 내적 크기만큼 다른 value들이 섞여 들어오는 간섭. key들이 서로 orthogonal하면 crosstalk은 정확히 0이고 복원은 완벽하다. 그러나 $\mathbb{R}^{d_k}$에는 서로 orthogonal한 벡터가 $d_k$개뿐이므로, 이 방식의 exact 저장은 $m \le d_k$ 쌍까지다 — capacity가 $O(d)$라는 첫 감각이 여기서 나온다. key가 orthogonal하지 않은 일반적인 경우(random unit key)에는 쌍별 내적이 대략 $1/\sqrt{d_k}$ 크기이므로, $m$개를 겹쳐 쓰면 신호 대 잡음 비가 $\sqrt{d_k/m}$ 꼴로 떨어진다. 즉 $m$이 $d_k$에 접근하기 한참 전부터 모든 read가 조금씩 오염된다. 숫자로 옮기면: $d_k = 128$에서 $m = 32$쌍이면 SNR $\approx 2$, $m = 128$이면 SNR $\approx 1$ — 신호와 잡음이 같은 크기가 된다. retrieval 품질이 요구 수준 아래로 떨어지기 시작하는 $m$이 곧 그 memory의 유효 slot 수다.

> **[해설]** 이 실패 모드는 독자가 아는 cache의 실패 모드와 다르다. cache miss나 hash collision은 특정 entry가 없거나 틀리는 국소 사건이지만, crosstalk은 **모든 read가 동시에 점진적으로 오염되는** 전역 사건이다. "cache가 가득 찼다"에 해당하는 경계가 명시적으로 없고, 품질이 연속적으로 저하된다. 뒤에서 볼 retention gate(→ 13장)는 이 암묵적 저하를 명시적·학습된 eviction으로 바꾸려는 시도다.

식 (5-1)을 online으로, token이 하나 올 때마다 갱신하는 형태로 쓰면

$$
W_t \;=\; W_{t-1} \;+\; v_t k_t^\top
\tag{5-3}
$$

이고, 이것을 **Hebbian write**라고 부른다(Hebb의 "함께 발화하는 뉴런은 함께 배선된다"는 학습 원리에서 온 이름이다). 두 가지 성질이 결정적이다. 첫째, **blind write다**: 쓰기 전에 읽지 않는다. 이 key가 이미 저장되어 있는지 확인하지 않고 무조건 더한다. 같은 key가 두 번 오면 value가 이중으로 쌓인다. 둘째, local·incremental이다: 현재 token의 $k_t, v_t$만 필요하고 과거를 다시 볼 필요가 없다 — 그래서 하드웨어 친화적이다.

그리고 이 장의 첫 번째 punchline: **식 (5-3)은 6장에서 만날 linear attention의 state update와 문자 그대로 동일하다.** [Titans §2]가 linear Transformer를 비판하는 문장이 정확히 이 지점을 겨눈다: 재귀식이 key·value를 matrix memory에 "additively compress"하는 구조라서, long context에서는 이 additive 본성이 memory overflow를 일으켜 성능을 크게 훼손한다는 것이다. Titans가 말하는 overflow의 미시적 실체가 식 (5-2)의 crosstalk 누적이다. 60년 된 correlation memory의 결함이 2020년대 efficient attention의 결함으로 그대로 재등장한다.

Rosetta-Stone 대응도 여기서 정확해진다. KV cache append도 write지만, 각 쌍이 분리된 slot에 저장되므로 간섭이 없다(무손실 append-once). Hebbian write는 같은 $d_v \times d_k$ 좌표 공간 위에 겹쳐 쓴다(간섭 = crosstalk). 그리고 shape를 보라: $v_t k_t^\top$는 rank-1 outer product로, 2장의 backward pass에서 본 $dW = \delta x^\top$와 같은 GEMM shape다. "훈련의 gradient write와 memory write는 같은 연산"이라는 이 책의 반복 주제가 여기서 처음 물리적으로 드러난다.

crosstalk의 크기가 key들의 기하에 전적으로 달려 있다는 사실은 설계 여지를 하나 연다. 고전 문헌에서 key는 주어진 데이터였다. 그러나 이 라인의 모델에서 key는 학습된 projection이 만든다: $k_t = W_K x_t$이고, $W_K$는 slow weights $\Theta$의 일부다(→ 4장). 즉 outer loop는 pretraining 동안 crosstalk이 덜 생기도록 key를 벌려 놓는 방향으로 $W_K$를 학습할 수 있다 — inner 문제의 "데이터"처럼 보이던 것이 사실은 outer loop가 학습하는 함수의 출력이며, 4장의 "inner 문제의 hyperparameter는 outer에서 학습되는 함수가 된다"는 명제가 여기서 첫 구체적 사례를 얻는다. 다만 outer loop가 고칠 수 있는 것은 key의 평균적 기하까지다. 고정 크기 행렬에 선형독립 방향이 $d_k$개뿐이라는 상한(§5.5)은 어떤 projection도 우회하지 못한다.

## 5.3 Hopfield network: energy로 본 memory와 최초의 capacity 상수

Hopfield 1982 (PNAS)는 auto-associative 문제 — 손상된 pattern에서 원본을 복원하기 — 를 energy 최소화로 정식화했다. state는 이진 벡터 $s \in \{-1, +1\}^d$, 저장할 pattern은 $x_1, \dots, x_m \in \{-1,+1\}^d$, weight는 또다시 Hebbian이다: $W = \sum_i x_i x_i^\top$ (대각 성분은 0으로 둔다). 여기에 **energy function**

$$
E(s) \;=\; -\tfrac{1}{2}\, s^\top W s
$$

를 정의하면, 뉴런을 하나씩 $s_j \leftarrow \mathrm{sign}\big((W s)_j\big)$로 갱신하는 dynamics가 $E$를 단조 감소시킨다. 저장된 pattern들은 (memory가 제대로 작동하는 한) $E$의 local minimum, 즉 **attractor**가 된다. 손상된 입력에서 시작해 energy의 내리막을 따라가면 가장 가까운 저장 pattern으로 수렴한다 — content-addressable memory의 원형이다.

주소로 찾는 memory(load/store)와 내용으로 찾는 memory의 구분은 독자에게 이미 익숙하다 — TLB나 cache의 tag match가 후자, 즉 content-addressable이다. Hopfield network는 content-addressable memory를 학습 가능한 신경망으로 구현한 최초의 사례군에 속하고, 이 라인이 "memory"라는 단어를 쓸 때의 의미는 언제나 이쪽이다: 주소가 아니라 query의 내용이 무엇을 꺼낼지 정한다.

systems 독자를 위한 재서술: Hopfield의 read는 lookup 한 번이 아니라 **GEMV + sign을 수렴할 때까지 반복하는 fixed-point iteration**이다. 읽기가 반복 계산이라는 것, 그리고 "복원 품질"이 energy 지형의 기하에 달려 있다는 것이 이후 이야기의 씨앗이다.

capacity는 어떤가. Hopfield 1982는 pattern 수가 약 $0.15\,d$를 넘으면 회복 오류가 심각해진다고 실험적으로 보고했고, 이후 statistical mechanics 분석은 임계값을 약 $0.138\,d$로 확정했다(Amit, Gutfreund & Sompolinsky; Phys. Rev. Lett. 1985가 임계비 약 0.14를 보고, 0.138은 Ann. Phys. 1987의 정밀화). 흔히 "capacity ≈ $0.14\,d$"로 요약된다. 더 넣으면 우아하게 저하되는 것이 아니라 임계점을 지나며 회복 자체가 붕괴한다.

임계 아래라고 깨끗한 것도 아니다. energy 지형에는 저장한 적 없는 가짜 minimum — 저장 pattern들의 홀수 개 혼합 같은 **spurious attractor** — 가 함께 생기고, 초기 상태가 나쁘면 read는 거기로 수렴한다. 그리고 임계를 넘으면 저장 pattern들이 attractor 자격 자체를 잃는다(혼합 상태 분석과 포화 붕괴 모두 같은 AGS 분석 계열의 표준 결과다; Amit, Gutfreund & Sompolinsky, Phys. Rev. A, 1985). §5.2의 crosstalk이 모든 read의 점진적 오염이었다면 Hopfield의 과적재는 절벽이다 — 같은 $O(d)$ 병목이라도, 망가지는 모양은 write rule과 read dynamics에 따라 다르다.

여기서 교훈은 상수 0.14가 아니라 스케일이다. **parameter는 $d^2$개인데 저장 능력은 $O(d)$다.** 저장 능력이 parameter 수가 아니라 key 공간의 차원(rank)에 묶여 있다는 것 — 이 병목의 정체는 §5.5 말미에서 [Atlas §3.1 Prop 1]로 정확해지고, Atlas 원문 스스로 자신의 결과를 "Willshaw model과 Hopfield network의 고전적 capacity 결과와 일치하며, capacity가 입력 embedding의 rank에 묶인다"고 자리매김한다 [Atlas App. C, Prop 1 증명].

## 5.4 dense/modern Hopfield: energy를 바꾸면 capacity가 바뀐다 — Atlas의 이론적 골격

고전 Hopfield energy는 더 일반적인 형태 $E(s) = -\sum_{i=1}^{m} F(x_i^\top s)$에서 $F(z) = z^2/2$인 특수 경우로 볼 수 있다(전개하면 식 (5-1)의 이차형식이 나온다). Krotov & Hopfield 2016 (arXiv:1606.01164)의 **dense associative memory**는 이 $F$를 고차 다항식 $F(z) = z^n$으로 바꾸는 한 수로 capacity를 폭증시킨다: 차수 $n$의 energy에서 capacity는 $d^{n-1}$ 스케일로 커진다 — 고정 오류율 기준으로 저장 가능 pattern 수가 상수 × $d^{\,n-1}$(상수는 허용 오류율에 따른다)이고, 무오류 기준에서는 로그 인자가 붙는다 [Krotov & Hopfield 2016 Eq. 5–6]. $n = 2$에서 고전 Hopfield의 $0.14\,d$가 복원된다.

왜 그런지의 직관은 kernel이다. $(x^\top s)^n = \phi_n(x)^\top \phi_n(s)$ — 차수 $n$ monomial feature map의 내적이다. 즉 energy의 차수를 올리는 것은 key들을 더 높은 차원의 공간으로 lift해서 "서로 orthogonal할 자리"를 늘리는 것과 같다. §5.2에서 capacity를 제한한 것이 "orthogonal한 방향의 개수 = $d_k$"였으므로, 공간을 $d^n$ 차원으로 키우면 한계도 따라 올라간다.

이 사슬의 극한이 현대적 결말이다. $F$를 exponential로 보내면(연속 state + log-sum-exp energy), Ramsauer et al. 2021 (arXiv:2008.02217)이 보였듯 1-step retrieval update가 **정확히 softmax attention의 형태**가 되고(pattern을 선형 사상으로 key/value화하고 softmax의 역온도를 $1/\sqrt{d_k}$로 두는 조건 [Ramsauer et al. 2021 Eq. 10, App. A.4]), capacity는 $d$에 exponential로 커진다(저장 가능 pattern 수의 하한이 1보다 큰 상수의 $(d-1)/4$ 거듭제곱 꼴 [Ramsauer et al. 2021 Thm 3]). 즉 Transformer의 attention은 exponential energy를 갖는 modern Hopfield network의 retrieval로 읽을 수 있다.

이 대응은 read의 비용 구조까지 설명한다. 고전 Hopfield의 read는 수렴까지 도는 fixed-point iteration이었다(§5.3). Ramsauer et al. 2021은 exponential energy 아래에서는 한 번의 update만으로 저장 pattern 근방으로 retrieval이 사실상 완료된다고 보고한다(잘 분리된 pattern에 대해 one update 후 오차가 separation에 지수적으로 작다 [Ramsauer et al. 2021 Thm 4]) — attention이 반복 없이 softmax 한 번으로 read를 끝내는 관행은 이 성질의 번역이다. energy를 날카롭게 만들수록 attractor의 basin이 가팔라져서, 반복 read가 1-step read로 접힌다.

<!-- FIG: ch05/fig-01-capacity-chain -->

사슬을 한 줄로 요약한다: **energy의 차수를 올린다 = key를 feature space로 lift한다 = capacity가 올라간다; exponential 극한에서 softmax attention에 도달한다.** 이 사슬이 [Atlas §3.1]의 이론적 골격 그 자체다. Atlas는 (a) matrix memory + $\ell_2$ loss의 capacity가 $O(d_k)$임을 증명하고(Prop 1), (b) 차수 $p$ polynomial feature map $\phi_p$로 key를 lift하면 $O(d_k^p)$로 올라감을 보이고(Prop 2), (c) exponential feature map $\phi^*$의 극한에서 softmax attention이 unbounded memory를 갖는 associative memory로 나타남을 이용한다 [Atlas §4.2]. Atlas 원문은 이 kernel 장치가 Krotov & Hopfield 2016의 방법을 계승한 것임을 명시한다 [Atlas §3.1]. 이 사슬을 지금 손에 쥐고 있으면 14장은 corollary처럼 읽힌다. (capacity의 formal 정의와 정리의 상세는 14장 소유다; 이 장은 고전 쪽만 정식으로 다룬다.)

systems 접점 하나: "capacity를 올린다 = key를 lift한다 = key 차원과 그에 붙는 연산이 커진다"이므로 이것은 공짜가 아니라 **state 크기·FLOP과의 거래**다. $\phi_p$의 lifted 차원은 $D = \Theta(d_k^p)$이고, memory state는 $W \in \mathbb{R}^{d_v \times D}$로 함께 커진다. 감각을 위한 숫자 하나: $d_k = 128$, 차수 $p = 2$의 전체 monomial map이면 $D = \binom{128+2}{2} = 8385$, state는 32 KiB에서 약 2 MiB로 65× 커진다(bf16, $d_v = 128$ 기준). capacity를 사는 통화가 state byte라는 것 — 이 거래의 비용 **구조**(지불 통화가 state byte·matmul 폭이라는 것)는 14장 §14.7에서 확정한다. 단, Atlas가 구현 차수 $p$와 sketch 차원을 공개하지 않아 절대값 계산은 그곳에서도 불가능하다는 것까지가 14장의 결론이다.

## 5.5 delta rule: append에서 error-correcting overwrite로

Hebbian write의 결함은 blind라는 것이었다. 같은 key가 다시 오면 확인 없이 위에 더해 이중 저장이 되고, 비슷한 key들은 서로를 오염시킨다. 개선의 방향은 write를 최적화 문제로 바꾸는 것이다. "이 쌍이 지금 얼마나 잘 저장되어 있는가"를 측정하는 손실

$$
\ell(W; k_t, v_t) \;=\; \big\| W k_t - v_t \big\|_2^2
\tag{5-4}
$$

을 정의하고, 새 쌍이 올 때마다 이 손실을 한 걸음 내려간다. gradient는 2장의 chain rule을 한 번 쓰면 $\nabla_W \ell = 2\,(W k_t - v_t)\,k_t^\top$ — 오차 벡터와 key의 outer product, 또 rank-1이다. 계수 2를 $\eta_t$에 흡수하고 한 step 내려가면

$$
W_t \;=\; W_{t-1} - \eta_t \big(W_{t-1} k_t - v_t\big) k_t^\top
\;=\; W_{t-1}\big(I - \eta_t\, k_t k_t^\top\big) + \eta_t\, v_t k_t^\top .
\tag{5-5}
$$

이것이 **delta rule**이다(Widrow & Hoff 1960의 LMS 알고리즘; 이 라인의 논문들은 1988년 reprint를 인용한다). 식 (M1)에 linear memory $\mathcal{M}(q;W)=Wq$를 대입한 전개형이며, 6장에서 만날 DeltaNet의 update 그 자체다. 이름에 주의: delta rule은 1960년의 학습 규칙이고, DeltaNet은 그것을 sequence layer로 쓰는 2020년대의 모델이다.

식 (5-5)는 세 가지로 읽을 수 있고, 셋 다 이후 장들에서 계속 쓰인다.

1. **read-modify-write.** $W_{t-1} k_t$는 이 key에 지금 저장된 value의 read다. delta rule은 그 read와 목표 $v_t$의 차이, 즉 오차만큼만 고쳐 쓴다. Hebbian write가 append-only log라면 delta rule은 key로 주소를 찾아가는 in-place update다. 같은 key가 두 번 오면 두 번째 write는 첫 번째를 덮어쓴다 — 이중 저장이 사라진다.
2. **1-step SGD.** 식 (5-5)는 2장의 SGD 객체를 손실 (5-4)에 적용한 것, 그 이상도 이하도 아니다. 독자가 2장에서 만난 가장 단순한 optimizer가 사실은 1960년의 memory write 규칙이었던 것이다. "optimizer는 associative memory다"라는 이 라인의 중심 명제([NL])가 여기서 처음 실체를 얻는다. 훈련의 커리큘럼(2장)과 기억의 커리큘럼(이 장)은 의도적으로 이 지점에서 합류한다.
3. **부분 소거 후 재기록.** $W_{t-1}(I - \eta_t k_t k_t^\top)$ 항을 보라. $\eta_t = 1$, $\|k_t\| = 1$이면 $I - k_t k_t^\top$는 $k_t$ 방향 성분을 정확히 지우는 projection이다: "이 key에 대한 옛 기억을 지우고 새 value를 쓴다." $\eta_t < 1$이면 부분만 지운다. $\eta_t$가 write 강도를 조절하는 gate라는 것 — 이것이 13장에서 inner learning rate를 data-dependent gate로 학습시키는 관점의 예고편이다.

비용의 대조도 명확히 해 두자. Hebbian write는 outer product 하나 — $d_k d_v$ MAC — 면 끝난다. delta rule은 쓰기 전에 읽어야 하므로 prediction read $W_{t-1} k_t$(GEMV)가 write 경로에 추가된다: FLOP은 대략 2배, 그리고 state 전체를 읽는 traffic이 매 write마다 발생한다. 더 나은 write rule은 공짜가 아니라 read를 지불하고 산다 — 이 패턴은 라인 내내 반복된다. momentum은 buffer $S_t$의 저장과 갱신을(12장), Omega rule은 window 안 $c$개 token의 재최적화를(14장) 같은 방식으로 지불한다.

delta rule의 보증은 이렇다. $\eta_t = 1$이고 key가 unit norm이면 update 직후 $W_t k_t = v_t$가 정확히 성립한다 — 방금 쓴 쌍은 crosstalk 없이 복원된다. 같은 쌍들을 반복 제시하며 돌리면(cycling) LMS는 least-squares 해로 수렴하고, key들이 선형독립이고 $m \le d_k$이면 exact interpolation 해 $W^\star = V K^{+}$ (pseudoinverse)에 도달한다. [Atlas §3.1 Prop 1]의 증명이 정확히 이 논리다: exact 저장 조건 $WK = V$는 $m d_v$개의 방정식과 $d_k d_v$개의 미지수를 갖는 선형계이므로, 선형독립 key에 대한 가해 조건은 $m \le d_k$이고, 이때 full-batch gradient descent는 minimum-norm 해로 수렴한다 [Atlas App. C, Prop 1 증명].

unit norm이 아닌 일반 key에서는 exact write의 조건이 하나 붙는다. update 직후 $W_t k_t = v_t$를 원하면, 식 (5-5)에 대입해 보면 $\eta_t = 1/\|k_t\|_2^2$이어야 한다 — norm이 큰 key일수록 살살 써야 한다(adaptive filtering 문헌이 normalized LMS라는 이름으로 표준화한 선택이다). 한 걸음 더 가서 step size를 $\eta_t/(1+\eta_t k_t^\top k_t)$로 바꾸면 어떤 $\eta_t > 0$에서도 안정한 implicit GD가 되는데, 이것이 6장 카탈로그의 Longhorn이 채택한 update다. write 강도의 스칼라 하나를 바꾸는 것만으로 모델 하나가 갈라져 나온다 — inner optimizer의 선택이 아키텍처의 축이라는 이 라인의 문법을 미리 보여 주는 사례다.

그리고 냉정한 사실 하나. delta rule은 Hebbian의 crosstalk을 교정하지만, **capacity 상한 자체는 올리지 못한다.** matrix memory + $\ell_2$ loss인 한 상한은 여전히 $O(d_k)$다 [Atlas §3.1 Prop 1]. write rule의 개선은 상한에 *도달*하게 해 주는 것이고, 상한을 *올리는* 것은 다른 축이다: memory를 deep하게 만들거나(2-layer 이상 MLP memory의 capacity 하한은 $O(d_k d_v)$, [Atlas §3.1 Thm 1]), key를 feature map으로 lift해야 한다($\phi_p$로 $O(d_k^p)$, [Atlas §3.1 Prop 2]). **write rule의 축과 capacity의 축은 독립이다** — 이 구분이 Part II의 설계 공간을 가른다. Titans와 Miras는 주로 write rule 축을 움직이고(momentum 추가, retention 추가, loss 교체), Atlas는 두 축을 동시에 움직인다.

## 5.6 attention은 압축하지 않는 associative memory다

이제 사다리의 꼭대기를 명시하자. softmax attention의 read

$$
y_t \;=\; \sum_{i \le t} \mathrm{softmax}_i\!\big(q_t^\top k_i / \sqrt{d_k}\big)\, v_i
$$

는 저장된 모든 쌍을 그대로 보관한 채(KV cache), read 시점에 kernel 가중 평균으로 연상하는 연산이다. parameter에 겹쳐 쓰지 않으므로 crosstalk이 없고, cache가 자라는 한 capacity 제한도 없다. 통계학의 언어로는 $\ell_2$ regression의 non-parametric Nadaraya–Watson 해이며(→ 6장, 10장), 이 책의 아키텍처 카탈로그에서 attention이 "압축하지 않는 극한"으로 분류되는 근거다. attention을 associative memory로 읽는 이 관점의 대표 인용은 Bietti et al. 2023 (arXiv:2306.00802)이고, Titans·Miras·Atlas 세 편 모두 이 관점을 명시적으로 채택한다 [Titans §1; Miras §2; Atlas §2].

이로써 이 장의 위계가 완성된다:

표 5-1 — 이 장의 세 write 방식과 그 대가.

| write 방식 | 저장 | read 오염 | exact capacity | read 비용 |
|---|---|---|---|---|
| Hebbian (5-3) | parametric, blind append | crosstalk $\propto k_i^\top k_j$ | $\le d_k$ (orthogonal일 때만) | $O(d_k d_v)$ GEMV |
| delta rule (5-5) | parametric, read-modify-write | 교정됨 (단일 pass에서는 recency 편향) | $\le d_k$ (선형독립이면 도달 가능) | $O(d_k d_v)$ GEMV |
| softmax attention | non-parametric append | 없음 | 무제한 (cache가 자라는 한) | $O(L\,d)$ 전체 스캔 |

이 라인의 존재 이유는 이 표의 가운데 행을 위로 밀어 올리는 것이다: attention의 연상 품질에 다가가되, state는 고정 크기로 유지하기. 그 수단들이 Part II의 목차 그 자체다 — write에 momentum과 retention을 더하고(12장), inner objective와 gate를 다시 고르고(13장), memory를 deep하게 만들고 key를 lift하며 window 단위로 함께 최적화한다(14장).

확장의 예고 하나. [NL §1]은 이 추상화를 아키텍처 바깥으로 밀어붙인다: 훈련의 backprop 자체가 "각 data sample을 그 예측의 오차에 mapping하는 associative memory"이고 [NL §3.1], momentum은 gradient들을 압축하는 memory이며, Adam은 element-wise $\ell_2$ objective에 대한 optimal associative memory라고 주장한다 [NL §1]. 그 상세는 16장의 일이다. 여기서 기억할 것은 하나다: 그 모든 주장의 원형이 이 장의 식 (5-4)–(5-5)라는 것.

## 5.7 Worked micro-example: crosstalk을 손으로 계산하고 delta rule로 고치기

$d_k = d_v = 2$, 쌍 두 개를 저장한다. key는 둘 다 unit norm이고 겹침(내적)은 $\mu = k_1^\top k_2 = 0.6$이다:

$$
k_1 = \begin{pmatrix}1\\0\end{pmatrix},\;
v_1 = \begin{pmatrix}1\\0\end{pmatrix};\qquad
k_2 = \begin{pmatrix}0.6\\0.8\end{pmatrix},\;
v_2 = \begin{pmatrix}0\\1\end{pmatrix}.
$$

**(1) Hebbian.** 식 (5-1)로 두 쌍을 겹쳐 쓰면

$$
W_{\mathrm{heb}} = v_1 k_1^\top + v_2 k_2^\top
= \begin{pmatrix}1&0\\0&0\end{pmatrix} + \begin{pmatrix}0&0\\0.6&0.8\end{pmatrix}
= \begin{pmatrix}1&0\\0.6&0.8\end{pmatrix}.
$$

읽어 보면 $W_{\mathrm{heb}} k_1 = (1,\;0.6)^\top$인데 정답은 $v_1 = (1,\;0)^\top$이다. 오차 $(0,\;0.6)^\top$는 식 (5-2)가 예언한 crosstalk $\mu\, v_2 = 0.6\, v_2$ 그대로다. 대칭적으로 $W_{\mathrm{heb}} k_2 = (0.6,\;1)^\top$이고 오차는 $\mu\, v_1$이다. 두 read 모두 오차 norm이 정확히 $\mu = 0.6$ — 60%의 오염이다.

**(2) delta rule, 첫 pass.** $\eta_t = 1$, $W_0 = 0$에서 시작해 식 (5-5)로 순서대로 쓴다.

- $(k_1, v_1)$ write: read는 $W_0 k_1 = 0$, 오차는 $-v_1$. 따라서 $W_1 = v_1 k_1^\top$. 빈 memory에의 첫 write는 Hebbian과 동일하다 — 읽을 것이 없기 때문이다.
- $(k_2, v_2)$ write: read는 $W_1 k_2 = (0.6,\;0)^\top$ — $k_2$의 자리에 이미 $k_1$ 기억의 그림자가 있다. 오차는 $(0.6,\;-1)^\top$ (norm $\approx 1.17$)이고,

$$
W_2 = W_1 - (0.6,\;-1)^\top k_2^\top = \begin{pmatrix}0.64 & -0.48\\ 0.6 & 0.8\end{pmatrix}.
$$

검산: $W_2 k_2 = (0.64\cdot 0.6 - 0.48\cdot 0.8,\;\; 0.6\cdot 0.6 + 0.8\cdot 0.8)^\top = (0,\;1)^\top = v_2$. **방금 쓴 쌍은 정확히 복원된다** — delta rule의 약속이다. 그런데 $W_2 k_1 = (0.64,\;0.6)^\top$이고 오차 norm은 $\approx 0.70$이다. Hebbian의 0.60보다 **오히려 나쁘다.** overwrite는 최신 쌍의 정확성을 위해 옛 쌍의 자리를 침범한다.

**(2b) 같은 쌍을 두 번 쓰면.** blind write의 의미를 숫자로 못박자. $W_{\mathrm{heb}}$에 $(k_2, v_2)$를 Hebbian으로 한 번 더 쓰면 read가 $(W_{\mathrm{heb}} + v_2 k_2^\top)\,k_2 = (0.6,\;2.0)^\top$이 된다 — value가 이중으로 쌓였다. 반면 $W_2$에 $(k_2, v_2)$를 delta rule로 다시 쓰면 오차 $W_2 k_2 - v_2 = 0$이라 update가 정확히 0이다. 이미 잘 저장된 쌍에 대한 delta write는 no-op이다 — read-modify-write의 가치가 이 한 줄에 있다.

**(3) cycling.** 두 쌍을 번갈아 다시 쓰면 오차가 기하급수로 줄어든다. 각 write 직전, 그 key의 read 오차 norm을 추적하면: $1.17 \to 0.70 \to 0.42 \to 0.25 \to \cdots$ — 매 write마다 $\times\,\mu = 0.6$, 한 cycle(두 쌍을 한 번씩 재기록)마다 $\times\,\mu^2 = 0.36$이다. 수렴 극한은 exact 해

$$
W^\star = V K^{-1} = \begin{pmatrix}1 & -0.75\\ 0 & 1.25\end{pmatrix},
\qquad W^\star k_1 = v_1,\;\; W^\star k_2 = v_2
$$

이고, 이것이 가능한 이유는 $m = 2 \le d_k = 2$에 key가 선형독립이기 때문이다 — [Atlas §3.1 Prop 1]의 조건 그대로다. read 오차의 전 과정을 표로 정리하면:

표 5-2 — 각 시점에서의 read 오차 norm ($\|\mathcal{M}(k_i; W) - v_i\|_2$).

| 시점 | $k_1$ 오차 | $k_2$ 오차 |
|---|---|---|
| $W_{\mathrm{heb}}$ (Hebbian 동시 저장) | 0.60 | 0.60 |
| $W_2$ (delta 1st pass) | 0.70 | 0.00 |
| $W_3$ ($k_1$ 재기록) | 0.00 | 0.42 |
| $W_4$ ($k_2$ 재기록) | 0.25 | 0.00 |
| $W^\star$ (극한) | 0.00 | 0.00 |

세 가지 교훈. 첫째, crosstalk은 추상적 개념이 아니라 **key 내적 그 자체**다(0.6이라는 숫자가 두 번 그대로 나타났다). 둘째, delta rule의 단일 online pass는 최신 쌍을 정확히 쓰는 대신 옛 기억을 침범한다 — **recency 편향은 overwrite 방식에 내장된 성질**이다. 셋째, cycling이 가능하면 least-squares 해로 수렴하지만, sequence modeling에서는 지나간 token을 다시 쓸 수 없다(단 한 번의 online pass). 그래서 이 라인은 다른 보완 장치를 쌓는다: 과거의 gradient 방향을 유지하는 momentum([Titans], → 12장), 무엇을 남길지 학습하는 retention(→ 13장), 그리고 최근 window 전체를 공동으로 재최적화하는 Omega rule([Atlas], → 14장). Atlas가 문제 설정에서 지적하는 비판 — 현재 token에 대해서만 greedy하게 최적화하는 online update는 개별 token을 memorize할 뿐 context 전체가 *함께* 잘 저장되었는지 묻지 않는다 [Atlas §1] — 이 정확히 이 표의 $W_2$ 행을 겨눈다.

참고로 이 예제의 모든 연산은 $2\times2$ GEMV 두 번과 rank-1 outer product 한 번씩이었다. 차원을 $d$로 올려도 연산의 종류는 같다 — 그것이 다음 절의 비용 계산이다.

## 5.8 Systems bridge: $d \times d$ state의 "유효 cache 크기"

이 장의 결과를 독자의 단위계 — byte, FLOP, bandwidth — 로 번역한다. head 하나, $d_k = d_v = 128$, bf16(2 bytes) 기준이다.

표 5-3 — non-parametric vs parametric associative memory, head 하나 기준 ($d_k = d_v = 128$, bf16).

| 항목 | KV cache (softmax attention) | matrix memory $W \in \mathbb{R}^{128\times128}$ |
|---|---|---|
| state 크기 | 512 B/token, $L$에 비례 (64K token이면 32 MiB) | 32 KiB 고정 |
| write | append: 512 B 복사, 간섭 없음 | rank-1 read-modify-write: $O(d^2)$ FLOPs + state 전체 왕복 |
| read | 전체 스캔: $O(L \cdot d)$ FLOPs·bytes | GEMV: $2d^2 \approx 33$ KFLOPs, 32 KiB read |
| exact 저장 한계 | cache가 자라는 한 무제한 | $\le d_k = 128$쌍 [Atlas §3.1 Prop 1] |
| forgetting | 명시적 eviction policy (paged cache 등) | crosstalk에 의한 암묵적 저하, 또는 학습된 gate (→ 13장) |

이 표에서 세 가지 계산을 뽑아 두면 Part II 내내 쓴다.

**첫째, byte 등가와 capacity 등가는 다르다.** 32 KiB짜리 matrix memory는 byte로는 KV cache 64 token 분량이다. 그러나 exact 저장 한계는 $d_k = 128$쌍 — byte 등가의 2배다. 반대로 말하면 겨우 2배다: $d\times d$ state는 "$L$을 $d$ 수준으로 압축해 주는 마법"이 아니라, **잘 관리해야 $O(d)$쌍을 담는 유한 cache**다. context가 $d_k$를 넘는 순간부터는 반드시 lossy하고, 무엇을 잃을지는 write rule과 gate가 정한다. capacity 이론은 "언제부터 압축이 불가피한가"를 알려 주는 도구다.

스케일을 실전 크기로 올려 보자. head 8개 × layer 48개면 matrix memory의 상주 state는 $32\,\mathrm{KiB} \times 8 \times 48 = 12\,\mathrm{MiB}$/request다. 같은 구성의 KV cache는 token당 $512\,\mathrm{B} \times 8 \times 48 = 192\,\mathrm{KiB}$이므로 64K context에서 12 GiB — 1024× 차이다. 이 1000×가 라인 전체의 유혹이고, 이 장의 capacity 결과는 그 가격표다: state를 1000× 줄이는 대가로, exact 저장 한계는 context 길이와 무관한 head당 $d_k$개 쌍으로 고정된다.

**둘째, decode의 연산 강도.** matrix memory의 decode 한 step은 read GEMV($2d^2$ FLOPs) + write의 prediction GEMV와 rank-1 교정(합쳐서 $O(d^2)$)이고, 그동안 state 32 KiB를 읽고 다시 쓴다(RMW). FLOP과 byte가 같은 $O(d^2)$ 차수이므로 arithmetic intensity는 몇 FLOP/byte 수준 — 최신 GPU의 ridge point보다 두 자릿수 아래로, 철저히 **bandwidth-bound**다. per-token으로는 KV 스캔의 $O(L\,d)$보다 싸지만, tensor core를 놀리는 연산이라는 점이 9장(chunkwise 병렬화)의 출발 동기가 된다.

**셋째, batching이 깨진다.** 지금까지의 GEMV들에서 $W$는 request마다 다른 per-sequence state다. shared weights를 전제로 여러 request를 한 GEMM으로 묶던 독자의 batching 감각은 여기서 무너진다 — batch 축이 사라진 GEMV들의 모음이 되고, grouped-GEMM류의 kernel이 필요해진다. 이 함의는 10장에서 cost model로 정리한다.

마지막으로 shape 관찰 하나. 이 장의 모든 write — Hebbian의 $v_t k_t^\top$, delta의 $(W k_t - v_t)k_t^\top$ — 는 rank-1 outer product였고, 2장의 backward pass가 만들던 $dW = \delta x^\top$와 동일한 GEMM shape였다. token을 $C$개 모아 쓰면 rank-$C$ GEMM이 된다. "memory write를 GEMM으로 묶는다"는 이 한 문장이 9장 chunkwise training의 전부다.

## 요약

- associative memory는 key→value mapping을 저장·복원하는 시스템이며, KV cache(무손실 non-parametric)와 matrix memory(고정 크기 parametric)는 같은 추상화의 양 극단이다. 이 라인의 공식 정의는 [Miras Def. 3.1]/[NL Def. 1]이 이 고전 개념을 재정식화한 것이다.
- Hebbian write $W_t = W_{t-1} + v_t k_t^\top$는 blind append이고, read 오염(crosstalk)의 크기는 key 간 내적이다. 이것이 [Titans §2]가 비판하는 linear attention의 "memory overflow"의 실체다.
- Hopfield 1982는 auto-associative memory를 energy 최소화로 정식화했고 capacity는 약 $0.14\,d$ — parameter $d^2$개에 저장 능력은 $O(d)$로, 병목은 parameter 수가 아니라 key 차원의 rank다.
- dense Hopfield의 사슬 — polynomial energy는 capacity를 $d^{n-1}$로, exponential energy는 softmax attention으로 — 이 [Atlas §3.1]의 $\phi_p$/$\phi^*$ 장치의 원형이다.
- delta rule은 $\ell_2$ associative loss의 1-step gradient descent이고(식 (M1)의 linear 전개), read-modify-write로 crosstalk을 교정하지만 matrix memory의 capacity 상한 $O(d_k)$ 자체는 올리지 못한다 [Atlas §3.1 Prop 1]. write rule의 축과 capacity의 축은 독립이다.
- 단일 online pass의 delta rule은 최신 쌍을 정확히 쓰는 대신 옛 기억을 침범한다(recency 편향). momentum(12장)·retention(13장)·window 재최적화(14장)는 이 구조적 성질에 대한 서로 다른 보완이다.
- delta rule의 exact write 조건은 $\eta_t = 1/\|k_t\|_2^2$(normalized LMS)이고, step size를 $\eta_t/(1+\eta_t k_t^\top k_t)$로 바꾸면 Longhorn의 implicit GD가 된다(→ 6장) — inner optimizer의 선택이 모델을 가른다.
- 이 장의 모든 write는 rank-1 outer product로, 2장의 $dW = \delta x^\top$와 같은 GEMM shape다. decode 한 step은 GEMV+RMW로 bandwidth-bound이며, per-sequence state는 shared-weight batching을 깨뜨린다.

## 자가 점검 체크리스트

- [ ] Hebbian write와 delta rule의 update 식을 쓰고, 각각을 blind append와 read-modify-write로 설명할 수 있다.
- [ ] 주어진 2–3개의 (k, v) 쌍에 대해 crosstalk을 손으로 계산하고, 그 크기가 key 내적임을 보일 수 있다.
- [ ] 식 (5-4)의 gradient를 계산해 delta rule (5-5)를 유도하고, 이것이 식 (M1)과 같음을 확인할 수 있다.
- [ ] Hopfield → dense Hopfield → softmax attention의 capacity 사슬을 설명하고, [Atlas]의 $\phi_p$와 $\phi^*$가 이 사슬의 어느 고리에 해당하는지 말할 수 있다.
- [ ] matrix memory의 capacity가 parameter 수 $d_k d_v$가 아니라 $O(d_k)$인 이유를 [Atlas §3.1 Prop 1]의 선형계 논리로 설명할 수 있다.
- [ ] Hebbian write, delta rule, crosstalk, capacity를 각각 KV cache append, in-place cache update, 검색 오염, 유효 cache 크기라는 inference 어휘로 옮길 수 있다.

## 다음 장으로

이 장은 write rule과 capacity를 "그 자리에 놓인" memory의 문제로 다뤘다: 쌍이 주어지면 어떻게 쓰고, 몇 개까지 담기는가. 6장은 이 memory를 sequence layer로 조립한다 — Hebbian write (5-3)을 그대로 layer로 만들면 linear attention이고(이 대응을 fast weight programming이라 부른다), delta rule (5-5)를 넣으면 DeltaNet이다. 그 순간 새 질문들이 열린다: key·value를 만드는 projection과 $\eta_t$ 같은 gate는 누가 학습하는가(outer loop — 4장의 답을 재사용한다), 그리고 token마다 순차적인 이 recurrence를 GPU에서 어떻게 GEMM으로 묶는가(9장의 주제다). 이 장이 남긴 미해결 질문 — 단 한 번의 online pass에서 무엇을 남기고 무엇을 덮어쓸 것인가 — 는 3장의 online learning 언어로 정식화된 뒤, 12장의 momentum과 13장의 retention 이론이 서로 다른 답을 내놓는다.
