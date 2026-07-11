# ch01. Orientation: inference 세계와 learning 세계의 Rosetta Stone

<!-- STYLE-ISSUE: STYLE-NOTATION §5.1의 ch01 슬러그는 ch01-orientation.md이나, 집필 지시서가 ch01-orientation-rosetta.md 경로를 명시적으로 지정하여 본 파일은 지시서 경로를 따랐다. P2 audit에서 슬러그 통일 필요. -->

> **이 장의 목표** — 이 장을 마친 독자는 다음을 할 수 있다.
> 1. KV cache와 linear-RNN state를 memory의 write/read 어휘로 재서술하고, softmax attention을 "압축하지 않는 memory"라고 정확한 의미로 부를 수 있다.
> 2. fast weights와 slow weights, inner loop와 outer loop를 비형식적으로 구분하고, "이 learning rate는 누가 학습하는가?"라는 질문에 loop의 이름으로 답할 수 있다.
> 3. Rosetta-Stone 사전(표 1-1)의 각 대응이 **동일**인지 **유비**인지 **차이가 논점**인지 판별하고, 잘못된 직관 이식을 스스로 차단할 수 있다.
> 4. 여섯 편의 논문이 이 책의 기준 수식 (M)의 어느 성분을 바꾸는지를 한 장의 지도로 그릴 수 있다.
>
> **왜 필요한가** — 여섯 편 전부가 이 장을 전제한다. 구체적으로: [Titans] (*Titans: Learning to Memorize at Test Time*, arXiv:2501.00663) §2는 attention과 현대 linear recurrent 모델 전부를 memory unit에 대한 write/read 연산으로 재서술하고([Titans Eq. 6–7]) 그 위에 자기 설계를 세운다 — 이 장의 사전은 바로 그 재서술을 독자의 KV cache 어휘에 접속하는 장치다. [TNT] (*TNT: Improving Chunkwise Training for Test-Time Memorization*, arXiv:2511.07343)의 중심 논증은 "chunk는 FlashAttention의 tile과 달리 계산되는 함수 자체를 바꾼다"는 구분 위에 서 있다 — 이 구분을 §1.3에서 처음 세운다. [NL] (*Nested Learning: The Illusion of Deep Learning Architecture*, arXiv:2512.24695)과 [Sleep] (*Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories*, arXiv:2606.03979)의 "momentum도 Adam도 backprop도 전부 associative memory다"라는 주장은 두 loop의 그림(§1.4) 없이는 범주 오류로 읽힌다. 이 장은 그 오독들을 본문이 시작되기 전에 제거한다.

## 1.1 두 세계: weights가 얼어 있는 세계와 움직이는 세계

독자의 세계는 하나의 불변식 위에 서 있다: **forward는 weights를 읽기만 한다.** serving 엔진의 모든 설계가 이 가정에서 나온다. weights는 로드 시점에 배치되고, 양자화되고, 여러 request가 공유한다. request 사이에 남는 유일한 가변 상태는 KV cache이고, 그마저 append-only다. prefill과 decode의 비대칭, continuous batching, paged attention — 전부 "parameter는 상수"라는 전제의 파생물이다.

이 책이 다루는 라인은 그 불변식을 폐기한다. 폐기가 정의적 특징이다. 이 라인의 sequence layer는 decode 한 step마다 자기 내부의 작은 신경망 weights를 **gradient step으로 갱신한다**. token 하나를 처리한다는 것은 그 작은 네트워크에 대해 forward를 돌리고, loss를 계산하고, backward를 돌리고, weights를 update하는 것이다. "test time"이라는 말은 더 이상 "학습이 없는 시간"을 뜻하지 않는다. 이 발상의 직계 원형인 Sun et al. 2024 (arXiv:2407.04620)는 §1에서 이를 "hidden state가 모델 그 자체이고, update rule이 self-supervised learning의 한 step"이라는 구도로 요약한다.

<!-- TODO-VERIFY: Sun et al. 2024 (arXiv:2407.04620) §1의 슬로건 원문 표현("the hidden state is a model, the update rule is a step of self-supervised learning") 정확한 wording 확인 필요. 확인 방법: arXiv:2407.04620 abstract/§1 대조. 현재 본문은 직접 인용 아닌 의역으로 처리함. -->

여기서 즉시 구분해야 할 것이 있다. 움직이는 것은 모델의 weights 전부가 아니다. 이 라인의 모델에는 두 종류의 parameter가 공존한다.

- **fast weights** — 하나의 sequence를 처리하는 동안 token마다 갱신되는 작은 네트워크의 weights. 이 책 전체에서 기호 $W$로 쓴다. request가 시작될 때 초기값 $W_{\mathrm{init}}$에서 출발하고, request가 끝나면 버려지거나 세션 상태로 관리된다. (정식 정의와 1992년까지 거슬러 가는 계보 → 6장.)
- **slow weights** — 독자가 아는 통상적 의미의 parameter: projection 행렬, backbone, 그리고 뒤에 볼 gate 생성 함수들. 기호 $\Theta$로 쓰며, pretraining이 끝나면 얼어붙는다. serving 관점에서 $\Theta$는 여전히 읽기 전용이다.

즉 폐기되는 것은 "모든 weights가 얼어 있다"이고, 유지되는 것은 "$\Theta$는 얼어 있다"이다. 대문자 구분($W$ vs $\Theta$)은 이 책에서 가장 중요한 표기 결정이며, 어떤 수식을 보든 기호만으로 "지금 어느 세계에 있는가"를 판별하게 해 준다.

이 두 종류의 parameter는 두 개의 학습 과정에 각각 속한다. **inner loop** — sequence의 시간축을 따라 token마다 $W_t$를 갱신하는 과정으로, decode와 같은 시간축에서 돈다. **outer loop** — 통상적 pretraining으로, inner loop 전체를 관통해서 $\Theta$를 학습한다. 지금은 이 비형식적 구분이면 충분하며 §1.4에서 조금 더 채우고, 형식화는 4장이 담당한다.

한 가지 오해를 미리 끊는다. 이것은 fine-tuning이 아니다. label도, 별도의 데이터 파이프라인도, 사람이 개입하는 훈련 job도 없다. "학습"은 layer의 forward 안에서, request마다, token마다 일어나는 내장 연산이다. 독자의 직관으로 가장 가까운 상은 이렇다: **KV cache에 압축 codec이 달렸고, 그 codec의 인코딩 연산이 gradient step이다.** 이 상이 얼마나 정확한지가 이 장의 나머지 주제다.

## 1.2 모든 sequence 모델은 이미 memory였다

[Titans] §2는 이 라인의 출발점이 되는 재서술을 제시한다: 임의의 recurrent 계열 sequence 모델은 memory unit에 대한 write 연산과 read 연산의 쌍이다 [Titans Eq. 6–7]. 통일 표기로 쓰면,

$$
W_t = f(W_{t-1}, x_t) \qquad \text{(write)}
\tag{1-1}
$$

$$
y_t = g(W_t, x_t) \qquad \text{(read)}
\tag{1-2}
$$

이다. 상태 $W_t$가 memory이고, $f$가 write rule, $g$가 read rule이다. 이 추상화가 실제로 얼마나 넓은지를 독자가 이미 아는 두 극단으로 확인한다.

**사례 1: softmax attention + KV cache.** 상태는 cache 그 자체, 즉 지금까지의 쌍 $\{(k_\tau, v_\tau)\}_{\tau\le t}$의 목록이다. write는 append다 — 기존 항목을 건드리지 않고 새 쌍을 뒤에 붙인다. read는 attention lookup이다 — query $q_t$와 모든 key의 유사도로 value들을 가중 평균한다. 이 memory는 **압축하지 않는다**. 저장은 무손실이고, capacity는 무한하며, 대가는 read 비용으로 지불한다: state 크기가 $O(L)$로 자라고 token당 read가 $O(L)$이다. 이 책은 softmax attention을 "capacity가 무한한 non-parametric **associative memory**"라고 부른다 — associative memory는 key를 주면 연관된 value를 돌려주는 저장 구조를 말한다(고전 정식화 → 5장, 이 라인의 공식 Definition → 13장).

**사례 2: linear attention / linear-RNN state.** 상태는 고정 크기 행렬 $W\in\mathbb{R}^{d_v\times d_k}$이다. write는 outer-product 누적 $W_t = W_{t-1} + v_t k_t^\top$, read는 $y_t = W_t q_t$다. 독자가 "efficient attention의 $d\times d$ state"로 아는 그 물건이다. 이 memory는 **압축한다**. state 크기와 token당 비용이 $O(d_v d_k)$로 고정되는 대신, 대가를 정보 손실로 지불한다 — 서로 직교하지 않는 key들이 겹쳐 쓰이며 회수가 오염된다(이 현상의 이름은 crosstalk이고 정량화는 5장이 담당한다; §1.6에서 손으로 먼저 체험한다).

두 사례는 하나의 trade-off의 양 끝이다: lossless-growing이냐 lossy-fixed냐. 독자는 이 trade를 이미 운영 감각으로 안다 — 긴 context에서 KV cache가 GPU memory를 삼키는 문제와, 그것을 피하려는 온갖 압축·eviction 기법 말이다. 이 라인 전체의 출발 질문은 다음과 같다: **write rule $f$를 더 똑똑하게 만들면 — 구체적으로, write를 optimizer의 한 step으로 만들면 — 고정 크기 state의 손실을 얼마나 줄일 수 있는가?**

이 관점에서 모든 sequence layer 설계는 네 가지 질문에 대한 답이다: 무엇을 저장하는가(state), 어떻게 쓰는가(write/update rule), 어떻게 읽는가(read), 무엇을 버리는가(retention). Part I의 나머지와 Part II의 여섯 논문은 전부 이 네 칸을 채우는 서로 다른 방식이다.

## 1.3 Rosetta-Stone 사전

아래 표가 이 책의 사전이다. 이후 모든 장은 새 개념을 도입할 때 이 대응을 재사용하며, 각 행의 셋째 열이 가장 중요하다: 그 대응이 **동일**(같은 대상의 재서술)인지, **유비**(정확하지만 새 요소가 있음)인지, **차이가 논점**(직관을 그대로 이식하면 틀림)인지를 구분하지 않으면 독자는 잘못된 직관을 이식받는다.

표 1-1 — Rosetta-Stone 사전: inference 어휘 ↔ 이 라인의 어휘

| inference 세계 (독자의 어휘) | 이 라인의 어휘 | 대응의 성격 |
|---|---|---|
| KV cache | non-parametric memory state; softmax attention = capacity 무한($\phi^*$)의 associative memory | **동일 대상의 재서술** — attention은 압축하지 않는 memory |
| KV cache append | memory **write** (outer-product Hebbian write가 최근접 대응; delta rule은 append가 아니라 **overwrite**) | 유비 + 차이: cache는 append-once, fast weights는 read-modify-**write** |
| attention lookup ($q\cdot K$) | memory **read** $y_t=\mathcal{M}(q_t;W_t)$ | 동일 추상화 (k→v 연상) |
| linear-RNN state ($d\times d$) | matrix memory = 고정 크기 lossy 압축 memory | 동일 |
| prefill | 큰 chunk의 병렬 write (compression) — TNT에선 global memory가 담당 | 유비(정확) |
| decode | $C=1$의 per-token online write + read | 유비(정확) — 단 **backward pass가 decode에 들어온다**는 점이 신세계 |
| FlashAttention tiling | chunkwise-parallel training의 chunk | ⚠ **차이가 논점**: tiling은 bit-exact, chunk는 함수 자체를 바꾸는 semantic 근사 (M4) |
| GEMM shape 감각 | $\nabla_W\ell$의 outer-product 구조: $dW = (\text{오차})\,k^\top$는 rank-$C$ GEMM | 동일 — 훈련 수식을 GEMM shape로 읽는 것이 이 책의 교수법 |
| scan/prefix-sum kernel | momentum의 associative scan (S5식), $\Pi_t$의 누적 합 | 동일 |
| roofline / arithmetic intensity | chunk 크기 $C$가 arithmetic intensity를 결정; 품질 최적 $C$(작음) vs MFU 최적 $C$(큼)의 긴장 | 동일 도구, 새 독립변수 |
| batching (shared weights 전제) | per-request fast-weight state는 shared-weight batching을 **깨뜨린다** → grouped-GEMM decode | ⚠ 차이가 논점 |
| paged KV cache / session cache | per-session weight state — 새로운 cache class (sizing, checkpoint/restore, eviction) | 유비 → Part III의 주제 |
| cache eviction policy | retention gate = **학습된** eviction | 유비(정확) |
| speculative decoding의 rollback | memory state snapshot/rollback (optimizer-trajectory 상태 포함) | 유비 + 미해결 문제 |
| optimizer state (경험 없음 → 2장에서 도입) | 훈련의 "register file / accumulator": $m_t,h_t$ | 도입용 유비 |
| weight update 없음(추론 불변식) | **폐기되는 불변식** — 이 라인의 정의적 특징 | 차이 그 자체 |
| sequence 축 | 훈련의 batch 축 역할을 **sequence 축이 대신한다** (inner loop의 mini-batch = chunk) | ⚠ 최대 혼동 지점 — 2장·9장에서 반복 강조 |
| distributed training (경험 없음) | context parallelism: reset이 sequential chain을 끊어 shard 병렬화 (TNT) | 도입용 유비 (data parallelism과의 대비) |
| 서빙 fleet의 백그라운드 job | sleep phase = 주기적 offline consolidation/dreaming job ("weights의 background compaction") | 유비 → 17장, Part III |

세 등급을 차례로 짚는다.

**동일 — 재서술만 하면 되는 행들.** KV cache가 memory라는 것, attention lookup이 read라는 것, $d\times d$ state가 고정 크기 압축 memory라는 것은 §1.2에서 이미 확인했다. 여기에 두 행을 추가로 강조한다. 첫째, GEMM shape 행: 이 책에서 만나게 될 gradient의 핵심 형태 $\nabla_W \ell = (\text{오차 벡터})\,k^\top$는 rank-1 outer product이고, token $C$개를 모으면 $(d_v\times C)\cdot(C\times d_k)$의 rank-$C$ GEMM이다. 즉 **훈련 수식은 독자가 매일 보는 GEMM shape로 읽을 수 있고, 이 책은 일관되게 그렇게 읽는다**. 둘째, scan 행: 뒤에서 momentum이 등장하면 그것은 $S_t = \beta_t S_{t-1} + (\cdot)$ 꼴의 선형 recurrence이고, 이는 독자가 아는 prefix-sum/associative scan kernel의 대수 구조 그대로다(→ 9장).

**유비 — 정확하되 새 요소가 하나씩 있는 행들.** prefill은 "큰 덩어리를 한 번에 병렬로 memory에 써 넣는 compression"으로, decode는 "chunk 크기 $C=1$의 per-token online write + read"로 옮겨진다. 이 대응은 놀랄 만큼 정확하며 15장에서 [TNT]의 global/local memory 구조가 그대로 이 구도를 탄다. 새 요소는 하나다: **decode 안에 backward pass가 들어온다.** 독자의 세계에서 decode는 forward-only였다. cache eviction 행도 정확한 유비다: 무엇을 잊을지 정하는 **retention gate** — 남길 비율 $\alpha_t\in[0,1]$을 token마다 정하는 학습된 계수(공식 정의 → 13장) — 는 정확히 "학습된 eviction policy"다. 손으로 짠 LRU 대신, 무엇을 버릴지 자체를 outer loop가 데이터로부터 배운다.

**차이가 논점 — 직관 이식을 차단해야 하는 ⚠ 행들.** 세 개가 있고, 셋 다 이 책 후반부의 논점을 미리 결정하므로 하나씩 정확히 세운다.

첫째, **FlashAttention tiling ↔ chunk.** 둘 다 "sequence를 블록으로 잘라 GEMM 친화적으로 만든다"는 점에서 같아 보인다. 그러나 FlashAttention의 tiling은 online-softmax 재배열로 **수학적으로 동일한 함수를 계산한다** — tile 크기는 성능 knob일 뿐 결과를 바꾸지 않는다. 반면 **chunkwise-parallel training** — recurrent한 write를 chunk 단위로 병렬화하는 이 라인의 표준 기법(정식 전개 → 9장) — 에서 chunk 안의 모든 gradient는 chunk 시작 시점의 state에서 평가된다(식 (M4)의 stale-snapshot 근사). 따라서 chunk 크기 $C$를 바꾸면 **계산되는 함수 자체가 바뀐다**. $C$는 스케줄이 아니라 semantic hyperparameter이며(명제화 → 9장), 이 사실이 [TNT] 한 편의 존재 이유다. tiling 직관을 그대로 이식하면 "C는 크게 잡을수록 좋다"는 결론이 나오는데, 그것이 정확히 이 라인이 부딪힌 함정이다.

둘째, **batching.** 독자의 continuous batching은 "모든 request가 같은 weights를 곱한다"는 전제로 여러 request의 GEMV를 하나의 GEMM으로 합친다. fast weights는 이 전제를 깨뜨린다: 같은 layer라도 request마다 $W_t$가 다르므로, decode에서 같은 연산을 묶으면 shared-GEMM이 아니라 grouped-GEMM이 된다. 이는 serving 쪽 비용 구조를 바꾸는 실질적 차이다(→ 10장, Part III).

셋째, **sequence 축 ↔ batch 축.** 훈련 문헌에서 "mini-batch로 gradient를 평균한다"고 할 때의 batch 축 역할을, 이 라인의 inner loop에서는 **sequence 축이 대신한다** — inner loop의 mini-batch가 곧 chunk다. 독자가 batch라는 단어를 만날 때마다 "지금 이것이 outer 훈련의 example 축인지, inner loop의 token 축인지"를 물어야 한다. 이것이 이 주제 전체에서 가장 흔한 혼동이며, 2장과 9장에서 반복해서 경고한다.

마지막으로 append 행의 미세하지만 중요한 차이: KV cache의 write는 append-once다 — 쓴 것을 다시 건드리지 않는다. fast-weight write는 read-modify-write다 — 기존 state를 읽고, 수정하고, 다시 쓴다. Hebbian 누적($+v_tk_t^\top$)이 append의 최근접 대응이고, **delta rule**(같은 key의 옛 value를 지우고 새로 쓰는 write rule; 정식 정의 → 5장)은 append가 아니라 overwrite다. 이 차이가 memory 품질을 가른다는 것을 §1.6에서 숫자로 확인한다.

## 1.4 두 개의 loop: 누가 무엇을 학습하는가

training 무경험 독자가 이 라인의 논문을 처음 열면 반드시 부딪히는 질문이 있다: "test time에 학습을 한다면, 그 학습의 learning rate는 누가 정하는가? 그것도 학습해야 하지 않나?" 답은 두 loop의 분업이며, 이 절에서 비형식적으로 확정한다.

이 책 전체의 기준 수식(master update)을 미리 본다. 유도는 하지 않는다 — 각 부품은 2장(optimizer), 5장(associative memory), 12장(Titans)이 조립한다. 지금은 읽는 법만 익힌다.

$$
S_t \;=\; \beta_t\, S_{t-1} \;-\; \eta_t\, \nabla_W\, \ell\big(W_{t-1};\, k_t, v_t\big),
\qquad
W_t \;=\; \alpha_t\, W_{t-1} \;+\; S_t
\tag{M}
$$

읽기는 $y_t=\mathcal{M}(q_t;W_t)$이다. 기호를 하나씩 독자의 어휘로 옮긴다.

- $\ell(W_{t-1};k_t,v_t)$ — **inner loss**: 현재 memory가 key $k_t$에서 value $v_t$를 얼마나 재구성하지 못하는지 재는 per-token 소형 loss. 기본형은 $\|\mathcal{M}(k_t;W_{t-1})-v_t\|_2^2$이다. "이 쌍이 이미 잘 저장되어 있는가"의 측정기다.
- $\nabla_W \ell$ — 그 실패를 줄이는 방향. 이번 token이 memory에 남기는 write 신호이며, [Titans]는 이것을 **surprise**라고 부른다(공식 정의 → 12장): 이미 저장된 내용과 다를수록(=놀라울수록) gradient가 크고, 크게 쓰인다.
- $S_t$ — gradient의 지수이동평균 누적 buffer, 즉 **momentum**(객체로서의 정의 → 2장). 형태에 주목하라: $S_t=\beta_t S_{t-1}+(\cdot)$은 선형 recurrence이고, 독자의 scan kernel이 바로 처리하는 그 구조다.
- $\eta_t,\ \beta_t,\ \alpha_t$ — 순서대로 inner learning rate, momentum decay, retention gate. 셋 다 시간 첨자가 붙어 있다는 것 자체가 정보다: 상수 hyperparameter가 아니라 **token마다 계산되는 gate**라는 뜻이다. $\alpha_t$는 남기는 비율이다 — $\alpha_t=1$이면 전부 유지, $0$이면 완전 소거. 학습된 eviction이라는 §1.3의 유비가 여기 계수 하나로 구현되어 있다.
- $\mathcal{M}(\cdot;W)$ — read 함수. 상태 $W$와 함수 $\mathcal{M}$을 분리해 쓰는 것이 이 책의 규약이다.

식 (M)이 말하는 것을 한 문장으로 줄이면 이렇다: **gradient descent + momentum + weight decay를 token 스트림 위에서 돌리면, 그것이 곧 sequence layer다.** 여섯 논문은 전부 이 한 줄의 성분 교체로 읽을 수 있으며, 그 지도가 §1.5다.

이제 처음의 질문에 답한다. $\eta_t$는 누가 정하는가? gate들은 slow weights가 만드는 token의 함수다: $\eta_t=\eta(x_t;\Theta)$, $\alpha_t=\alpha(x_t;\Theta)$ 같은 형태로, gate를 계산하는 작은 head의 parameter가 $\Theta$의 일부다. 그리고 $\Theta$는 outer loop — 통상적 pretraining — 가 학습한다. outer loop는 inner loop 전체(수천 step의 memory 갱신)를 관통해 backpropagate하면서, "어떤 token에서 세게 쓰고, 무엇을 잊고, 얼마나 관성을 줄지"의 정책 자체를 배운다. 요약하면: **inner loop는 $W$를 움직이고, outer loop는 inner loop의 hyperparameter를 포함한 $\Theta$를 학습한다.** projection $W_K,W_V,W_Q$, gate 생성 head, 초기 상태 $W_{\mathrm{init}}$ — 전부 outer의 소유물이다. 이 문장이 여섯 편 전부를 여는 열쇠이며, 형식화(bilevel optimization)는 4장이, 각 논문에서의 구체적 분업표는 Part II 각 장의 "outer vs inner" 절이 담당한다.

## 1.5 여섯 편의 지도

여섯 편은 한 연구 라인(Google Research, Behrouz 계열)의 연작이며, 각 편이 식 (M)의 어느 성분을 바꾸는지로 정리하면 전체 구도가 한 눈에 들어온다. [Miras] (arXiv:2504.13173; 논문 제목은 *It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization*이지만 이 책은 framework 이름 Miras로 통칭한다)와 [Atlas] (*Atlas: Learning to Optimally Memorize the Context at Test Time*, arXiv:2505.23735)가 여기서 처음 등장한다.

표 1-2 — 여섯 편의 논문: 식 (M) 기준의 위치와 Part I 열쇠 장

| 논문 | (M)에서 바꾸는 것 | 한 줄 요약 | 열쇠가 되는 Part I 장 |
|---|---|---|---|
| [Titans] (2501.00663) | (M) 그 자체의 수립 | GD + momentum + weight decay가 test time의 deep memory 갱신 규칙이 된다; surprise가 write 강도를 정한다 | 2·5·6·8·9장 |
| [Miras] (2504.13173) | $\ell$ (attentional bias)과 retention 항 | 기존 모델 전부(RetNet·Mamba-2·DeltaNet·TTT·Titans 등)를 "online optimizer가 곧 sequence layer"라는 하나의 framework로 재유도 | 3·5·6장 |
| [Atlas] (2505.23735) | $\ell$의 범위(window $c$)와 inner optimizer(Muon/$\mathrm{NS}_\kappa$), capacity | per-token write를 sliding-window 목적함수(Omega rule)로 바꾸고, memory capacity를 feature map으로 끌어올린다 — test-time memorization의 정식화 | 2·3·5·9장 |
| [TNT] (2511.07343) | 훈련 경제학: chunk 크기 $C$의 speed–quality 충돌 | hierarchical global/local memory + 주기적 state reset + 2-stage training으로 큰 $C$ 훈련과 작은 $C$ serving을 양립 | 8·9·10장 |
| [NL] (2512.24695) | 층위: 2-level → K-level, update frequency | 모델과 훈련 절차 전체를 "각자 자기 주파수로 갱신되는 nested associative memory"의 스펙트럼으로 재구성 (Hope, CMS) | 2·4·5·11장 |
| [Sleep] (2606.03979) | lifecycle: train/test 경계 자체 | wake에 빠른 memory로 흡수하고, sleep에 consolidation·dreaming으로 느린 weights에 이관하는 주기적 생애 주기 | 4·11장 |

세로로 읽으면 계보다: [Titans]가 기준 수식을 세우고, [Miras]가 그것을 일반 이론으로 넓히고, [Atlas]가 목적함수와 optimizer를 강화하고, [TNT]가 훈련 비용 문제를 풀고, [NL]이 전체를 K-level로 일반화하고, [Sleep]이 train/test 구분 자체를 없앤다. 참고로 [Titans]는 자신의 update가 Gated DeltaNet, Longhorn, TTT layer 등 기존 recurrent 모델들을 특수 사례로 포함한다고 논한다 [Titans App. C] — 즉 식 (M)은 이 라인 바깥의 efficient-attention 계보까지 덮는 일반형이다.

<!-- TODO-VERIFY: Titans App. C가 특수 사례로 복원한다고 명시하는 모델 목록(Gated DeltaNet/Longhorn/TTT 외 RWKV-7 포함 여부 — RWKV-7은 Titans v1보다 늦게 공개되어 원문에 없을 가능성 높음). 확인 방법: papers/2501.00663.txt Appendix C 절 검색 "special case", "Longhorn", "DeltaNet". -->

가로로 읽으면 이 책의 사용법이다: Part I의 각 장은 특정 논문의 특정 메커니즘을 여는 열쇠로 설계되어 있다. 2장은 optimizer를 (state, update, cost)를 갖는 객체로 세운다 — [Titans]의 momentum과 [Atlas]의 Muon, [NL]의 "Adam도 memory다"가 전부 여기 걸린다. 3장(online learning, regret, FTRL)은 [Miras]의 이론 골격이다. 4장(meta-learning, bilevel)은 §1.4의 두 loop를 형식화한다. 5장(associative memory)과 6장(linear attention, fast weight programming)은 §1.2의 두 사례 사이 공간을 채우고, 7장(SSM 계보)이 Mamba 계열을 같은 표에 합류시킨다. 8장이 TTT 원형을, 9장이 chunkwise-parallel training이라는 공용 병렬화 스킴을, 10장이 비용 모델 cheat sheet를, 11장이 [NL]·[Sleep] 전용 배경(continual learning, distillation)을 담당한다. 급한 독자는 표 1-2의 자기 목표 논문 행에서 열쇠 장만 뽑아 읽어도 되지만, Part I의 장 순서는 의존성 순서이므로 순서대로 읽는 것이 가장 싸다.

## 1.6 Worked micro-example: $2\times 2$ memory를 손으로 굴려 보기

말로 세운 사전을 숫자로 확인한다. $d_k=d_v=2$의 matrix memory $W\in\mathbb{R}^{2\times 2}$를 놓고, 세 가지 write rule — append, Hebbian, delta — 로 같은 데이터를 저장해 본다. 모든 계산은 암산 가능한 크기다.

저장할 쌍은 셋이다(열벡터):

$$
k_A=\begin{pmatrix}1\\0\end{pmatrix},\ v_A=\begin{pmatrix}4\\0\end{pmatrix};\qquad
k_B=\begin{pmatrix}0\\1\end{pmatrix},\ v_B=\begin{pmatrix}0\\2\end{pmatrix};\qquad
k_C=\begin{pmatrix}1\\1\end{pmatrix},\ v_C=\begin{pmatrix}1\\1\end{pmatrix}.
$$

**(a) Append (KV cache).** 세 쌍을 목록에 그대로 쌓는다. read는 lookup이므로 $k_A$를 물으면 $v_A$가 정확히 나온다. 무손실 — 대신 state가 쌍의 개수만큼 자란다. 기준선이다.

**(b) Hebbian write (linear attention의 누적).** $W \leftarrow W + v k^\top$로 차례로 쓴다. A와 B를 쓰면

$$
W = v_A k_A^\top + v_B k_B^\top
= \begin{pmatrix}4&0\\0&0\end{pmatrix} + \begin{pmatrix}0&0\\0&2\end{pmatrix}
= \begin{pmatrix}4&0\\0&2\end{pmatrix}.
$$

read를 확인하면 $W k_A = (4,0)^\top = v_A$ — 정확하다. $k_A \perp k_B$이므로 두 저장이 서로를 건드리지 않았다. 이제 C를 마저 쓴다: $v_C k_C^\top = \begin{pmatrix}1&1\\1&1\end{pmatrix}$이므로

$$
W' = \begin{pmatrix}5&1\\1&3\end{pmatrix}.
$$

read가 무너진다: $W' k_A = (5,1)^\top \ne (4,0)^\top$, 그리고 방금 쓴 C조차 $W' k_C = (6,4)^\top \ne (1,1)^\top$이다. $k_C$가 $k_A, k_B$와 직교하지 않아 저장들이 겹쳐 오염된 것 — §1.2에서 예고한 crosstalk이다. append-once 직관("쓰면 그대로 남는다")이 압축 memory에서 깨지는 지점이 정확히 여기다.

**(c) Delta write (gradient step으로서의 write).** C를 "그냥 더하는" 대신, memory가 $k_C$에 대해 지금 무엇을 답하는지 먼저 읽고 그 오차만큼만 고쳐 쓴다. inner loss를 $\ell(W;k_C,v_C)=\|Wk_C-v_C\|_2^2$로 놓으면 gradient는

$$
\nabla_W \ell = 2\,(W k_C - v_C)\,k_C^\top
$$

이고 — 오차 벡터와 key의 outer product, 즉 rank-1 행렬이다 — (b)의 A·B 저장 상태 $W=\begin{pmatrix}4&0\\0&2\end{pmatrix}$에서 예측은 $Wk_C=(4,2)^\top$, 오차는 $Wk_C-v_C=(3,1)^\top$이다. gradient를 성분으로 쓰면 $\nabla_W\ell = 2\begin{pmatrix}3\\1\end{pmatrix}\begin{pmatrix}1&1\end{pmatrix} = \begin{pmatrix}6&6\\2&2\end{pmatrix}$이고, 한 step을 $W'' = W - \eta_t\nabla_W\ell$, $\eta_t=\tfrac14$로 밟으면

$$
W'' = \begin{pmatrix}4&0\\0&2\end{pmatrix} - \tfrac14\begin{pmatrix}6&6\\2&2\end{pmatrix}
= \begin{pmatrix}2.5&-1.5\\-0.5&1.5\end{pmatrix}.
$$

read를 확인한다: $W'' k_C = (2.5-1.5,\ -0.5+1.5)^\top = (1,1)^\top = v_C$. **방금 쓴 쌍이 정확히 회수된다.** 같은 고정 크기 state인데, write rule을 "누적"에서 "gradient step"으로 바꾸는 것만으로 새 항목의 저장 품질이 달라졌다. 이것이 delta rule이고(1960년대 기원과 정식 전개 → 5장), (M)의 최소형 — 식 (M1), momentum도 retention도 없는 1-step GD write — 그 자체다. 대가도 정직하게 확인하자: $W'' k_A = (2.5,-0.5)^\top$로 A의 회수는 손상됐다. $k_C$가 $k_A$와 직교하지 않는 한 공짜는 없다 — 몇 쌍까지 버틸 수 있는가가 capacity 질문이며 5장과 14장의 주제다.

learning rate가 실제 knob라는 것도 이 크기에서 바로 보인다. 위에서 $\eta_t=\tfrac14$ 대신 $\eta_t=\tfrac12$를 쓰면 $W''k_C = (-2,0)^\top$ — 목표를 지나쳐 반대편으로 넘어간다($\|k_C\|^2=2$이므로 유효 step이 2배로 뻥튀기된다). step 크기가 회수 정확도를 직접 좌우하므로, 이 라인이 $\eta_t$를 고정 상수가 아니라 **학습된 data-dependent gate**로 만든 것은 사치가 아니라 필수였던 셈이다.

이 예제에서 가져갈 것은 세 가지다. 첫째, **write는 계산이다** — 그리고 그 계산의 모양은 (오차)$\,k^\top$의 rank-1 outer product, token을 $C$개 모으면 rank-$C$ GEMM이다. 둘째, 같은 state라도 write rule(append / Hebbian / delta)에 따라 memory 품질이 갈린다 — optimizer의 선택이 곧 memory의 선택이다. 셋째, 방금 우리는 $\nabla_W\ell$을 autograd 없이 손으로 유도해서 썼다 — serving kernel에 backward를 넣을 때 실제로 벌어지는 일이 정확히 이것이다(→ 8장, 10장).

## 1.7 Systems bridge: 첫 back-of-envelope

사전의 마지막 페이지는 숫자다. 독자의 로그와 대시보드에 잡히는 양들로 이 라인의 물체 크기를 가늠해 본다. 아래 산수는 이 책의 예시 계산이며 특정 논문의 수치 주장이 아니다.

**state 크기.** 전형적 GQA 구성 한 layer를 잡자: KV head 8개, head 차원 128, bf16(2 byte). KV cache는 token당 layer당 $2\times 8\times 128\times 2\,\mathrm{B} = 4\,\mathrm{KiB}$를 쓴다. 128K context면 layer당 $4\,\mathrm{KiB}\times 131{,}072 = 512\,\mathrm{MiB}$ — 32-layer 모델이면 request 하나가 16 GiB다. 같은 head 구성의 matrix memory(TTT-Linear급, head당 $128\times128$)는 head당 32 KiB, layer당 256 KiB, 32 layer에 8 MiB로 **context 길이와 무관하게 고정**이다. 128K 시점에서 layer당 2048×의 차이이고, context가 길수록 벌어진다. 이것이 이 라인이 long-context에서 갖는 구조적 지렛대다.

**대신 지불하는 것.** 첫째, decode step의 연산이 바뀐다: 이제 매 token마다 memory 네트워크의 forward에 더해 backward와 update가 돈다. backward의 FLOP은 forward의 약 2배라는 규칙(유도 → 2장)을 받아들이면, memory 모듈만 보면 token당 연산이 대략 3× 이상으로 불어난다. 둘째, state가 read-modify-write 대상이 되므로 token당 state 크기의 최소 2배(읽기+쓰기)의 memory 트래픽이 붙는다 — KV cache의 append(쓰기 4 KiB)와 달리, 8 MiB짜리 state 전체가 매 token 왕복하는 그림이다. decode가 원래도 bandwidth-bound라는 독자의 감각은 여기서도 유효하며, 정밀한 roofline 배치는 10장이 담당한다. 셋째, §1.3의 ⚠ 행 그대로, per-request state는 shared-weight batching을 깨뜨려 decode를 grouped-GEMM으로 만든다.

**훈련 쪽은 더 심각하다.** 이 recurrence를 그대로 훈련하면 sequence 방향으로 직렬이라 가속기가 논다. 그래서 라인 전체가 chunkwise-parallel training(→ 9장)에 의존하는데, chunk를 키우면 GEMM은 통통해지지만 §1.3에서 세운 대로 **함수 자체가 바뀌어 품질이 떨어진다**. 이 긴장이 어느 정도인가 하면, [TNT]는 deep memory 모듈의 훈련이 작은 chunk에서 peak 대비 5–10% 미만의 FLOPs utilization로 떨어지는 경우가 흔하다고 보고한다 [TNT §1]. 품질 최적의 $C$(작다)와 MFU 최적의 $C$(크다)가 갈라지는 이 구도는 roofline이라는 익숙한 도구 위에 $C$라는 새 독립변수가 올라온 것이며, [TNT] 한 편(15장)이 통째로 이 문제를 다룬다.

정리하면: 이 라인은 KV cache의 memory 폭발을 고정 크기 state로 바꾸는 대신, (i) decode에 backward를, (ii) state에 RMW 트래픽을, (iii) serving에 새로운 cache class(per-session weight state)를, (iv) 훈련에 chunk 경제학을 들여온다. 이 네 가지가 Part I 나머지 장들의 systems bridge에서 반복적으로 계량된다.

## 요약

- softmax attention + KV cache는 압축하지 않는(capacity 무한, non-parametric) associative memory이고, linear-RNN의 $d\times d$ state는 고정 크기 lossy memory다. 모든 sequence 모델은 write (1-1)과 read (1-2)의 쌍이다 [Titans Eq. 6–7].
- 이 라인의 정의적 특징은 decode 안의 gradient step이다. "weights는 읽기 전용"이라는 inference 불변식은 fast weights $W$에 대해 폐기되고, slow weights $\Theta$에 대해서는 유지된다.
- inner loop는 $W_t$를 token마다 갱신하고, outer loop는 inner loop의 hyperparameter(gate, projection, $W_{\mathrm{init}}$)를 포함한 $\Theta$를 학습한다. "이 learning rate는 누가 학습하는가"의 답은 항상 outer loop다.
- 기준 수식 (M)은 "GD + momentum + weight decay가 곧 sequence layer"라는 라인 전체의 요약이며, 여섯 논문은 각각 (M)의 성분 — $\ell$과 retention(Miras), window와 optimizer(Atlas), 훈련 경제학(TNT), 층위(NL), lifecycle(Sleep) — 을 바꾼 것이다.
- Rosetta 대응은 동일/유비/차이의 세 등급이며, ⚠ 3대 함정은 (i) FlashAttention tiling ↔ chunk(bit-exact vs semantic), (ii) shared-weight batching의 붕괴, (iii) sequence 축이 batch 축을 대신한다는 것이다.
- write는 (오차)$\,k^\top$ 꼴의 rank-1/rank-$C$ GEMM이며, write rule의 선택(append/Hebbian/delta)이 곧 memory 품질의 선택이다 — $2\times2$ 손계산으로 확인했다.
- 고정 크기 state의 대가는 decode 내 backward, state RMW 트래픽, grouped-GEMM decode, 그리고 chunk 크기의 품질–MFU 긴장이다.

## 자가 점검 체크리스트

- [ ] KV cache와 $d\times d$ matrix state를 각각 네 가지 설계 질문(state / write / read / retention)으로 서술할 수 있다.
- [ ] "이 layer의 inner learning rate는 누가 학습하는가?"에 두 loop의 어휘로 답하고, $\eta_t = \eta(x_t;\Theta)$라는 형태의 의미를 설명할 수 있다.
- [ ] 식 (M)의 각 기호($\ell$, $\nabla_W\ell$, $S_t$, $\eta_t/\beta_t/\alpha_t$, $\mathcal{M}$)를 inference 어휘로 한 문장씩 옮길 수 있다.
- [ ] §1.6의 Hebbian write와 delta write를 손으로 재계산하고, delta write가 $k_C$의 회수를 정확하게 만든 이유와 $\eta_t=\tfrac12$에서 overshoot가 나는 이유를 말할 수 있다.
- [ ] FlashAttention tiling과 chunkwise training의 chunk가 왜 다른지(bit-exact 재배열 vs stale-snapshot 근사) 설명할 수 있다.
- [ ] 표 1-1의 임의 행에 대해 대응의 성격(동일/유비/차이)을 판별하고, 그 행의 개념을 inference 어휘로 옮길 수 있다.

## 다음 장으로

이 장은 "write가 gradient step이다"라고 선언만 했다 — gradient가 무엇이고 어떻게 계산되는지는 아직 블랙박스다. 2장은 그 블랙박스를 연다: backward pass를 밑바닥부터(독자는 한 번도 짜 본 적이 없다는 전제로) 세우고, SGD·momentum·AdamW·Muon을 각각 (state, update, cost)를 갖는 객체로 정의한다. 그 과정에서 이 장의 복선 두 개가 회수된다: momentum buffer가 linear-RNN state와 같은 대수를 탄다는 것, 그리고 weight decay의 $(1-\lambda)$가 retention gate와 같은 물건이라는 것. 식 (M)의 부품이 전부 손에 잡히고 나면, Titans의 memory update는 "처음 보는 수식"이 아니라 "아는 부품의 재배선"이 된다.
