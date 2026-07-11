# ch01. Orientation: inference 세계와 learning 세계의 Rosetta Stone

> **이 장의 목표** — 이 장을 마친 독자는 다음을 할 수 있다.
> 1. KV cache와 linear-RNN state를 memory의 write/read 어휘로 재서술하고, softmax attention을 "압축하지 않는 memory"라 부를 수 있다.
> 2. fast/slow weights와 inner/outer loop를 구분하고, "이 learning rate는 누가 학습하는가?"에 loop의 이름으로 답할 수 있다.
> 3. Rosetta-Stone 사전(표 1-1)의 각 대응이 **동일**/**유비**/**차이가 논점** 중 무엇인지 판별할 수 있다.
> 4. 여섯 논문이 기준 수식 (M)의 어느 성분을 바꾸는지 한 장의 지도로 그릴 수 있다.
>
> **왜 필요한가** — 여섯 편 전부가 이 장을 전제한다. [Titans] (*Titans: Learning to Memorize at Test Time*, arXiv:2501.00663) §2는 attention과 현대 linear recurrent 모델 전부를 memory unit의 write/read 연산으로 재서술하고([Titans Eq. 6–7]), 이 장의 사전은 그것을 독자의 KV cache 어휘에 접속한다. [TNT] (*TNT: Improving Chunkwise Training for Test-Time Memorization*, arXiv:2511.07343)의 중심 논증은 "chunk는 FlashAttention tile과 달리 계산되는 함수 자체를 바꾼다"는 구분 위에 선다. [NL] (*Nested Learning: The Illusion of Deep Learning Architecture*, arXiv:2512.24695)과 [Sleep] (*Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories*, arXiv:2606.03979)의 "momentum도 Adam도 backprop도 전부 associative memory다"라는 주장은 두 loop의 그림(§1.4) 없이는 범주 오류로 읽힌다.

## 1.1 두 세계: weights가 얼어 있는 세계와 움직이는 세계

독자의 세계는 하나의 불변식 위에 서 있다: **forward는 weights를 읽기만 한다.** weights는 여러 request가 공유하고, 유일한 가변 상태는 append-only KV cache다 — prefill/decode 비대칭, continuous batching, paged attention이 전부 이 전제의 파생물이다.

이 라인은 그 불변식의 폐기가 정의적 특징이다: sequence layer가 decode 한 step마다 자기 내부의 작은 신경망 weights를 **gradient step으로 갱신한다** — token 하나의 처리가 곧 그 네트워크의 forward·loss·backward·update다. 직계 원형 Sun et al. 2024 (arXiv:2407.04620)는 이를 "hidden state가 모델 그 자체이고 update rule이 self-supervised learning의 한 step"이라 요약한다 [2407.04620 abstract·§1].

움직이는 것은 weights 전부가 아니다. 두 종류의 parameter가 공존한다.

- **fast weights** ($W$) — token마다 갱신되는 작은 네트워크의 weights. $W_{\mathrm{init}}$에서 출발해 request가 끝나면 버려지거나 세션 상태로 관리된다(정식 정의·계보 → 6장).
- **slow weights** ($\Theta$) — 독자가 아는 통상적 parameter(projection, backbone, gate 생성 head). pretraining 후 얼어붙고 serving 관점에서 읽기 전용이다.

즉 폐기되는 것은 "모든 weights가 얼어 있다"이고 wake/serving 동안 유지되는 것은 "$\Theta$는 얼어 있다"이다(예외: [Sleep]의 sleep phase는 이 불변식을 주기적으로 깨고 slow weights를 offline으로 갱신·확장한다 → 17장) — 이 대문자 구분($W$ vs $\Theta$)이 이 책의 가장 중요한 표기 결정이다. 두 parameter는 두 학습 과정, 곧 token마다 $W$를 갱신하는 **inner loop**와 pretraining으로 $\Theta$를 학습하는 **outer loop**에 속한다(형식화 → 4장). 미리 끊어 둘 오해: 이것은 fine-tuning이 아니다 — label도 훈련 job도 없이 layer forward 안의 내장 연산이며, 가장 가까운 상은 **"KV cache에 압축 codec이 달렸고 그 인코딩 연산이 gradient step"**이다.

## 1.2 모든 sequence 모델은 이미 memory였다

[Titans] §2의 재서술이 이 라인의 출발점이다: 임의의 recurrent sequence 모델은 memory unit의 write/read 연산 쌍이다 [Titans Eq. 6–7].

$$
W_t = f(W_{t-1}, x_t) \qquad \text{(write)}
\tag{1-1}
$$

$$
y_t = g(W_t, x_t) \qquad \text{(read)}
\tag{1-2}
$$

독자가 아는 두 극단이 양 끝이다. **softmax attention + KV cache**는 append-only 목록을 attention으로 읽는 **압축하지 않는** memory다 — 무손실·capacity 무한 대신 state와 token당 read가 $O(L)$로 자란다. 이 책은 이를 "capacity 무한의 non-parametric **associative memory**"(key로 연관 value를 회수하는 저장 구조; 고전 정식화 → 5장, 공식 Definition → 13장)라 부른다. **linear attention / linear-RNN state**는 고정 크기 행렬 $W\in\mathbb{R}^{d_v\times d_k}$에 outer product $v_t k_t^\top$를 누적하는 **압축하는** memory다 — 비용이 고정되는 대신 직교하지 않는 key들이 겹쳐 쓰여 회수가 오염된다(crosstalk; → 5장, §1.6에서 체험).

lossless-growing이냐 lossy-fixed냐 — 독자가 이미 운영 감각으로 아는 trade다. 이 라인의 출발 질문은 **write rule $f$를 optimizer의 한 step으로 만들면 고정 크기 state의 손실을 얼마나 줄일 수 있는가**이고, 이 관점에서 모든 sequence layer는 네 질문(state / write / read / retention)에 대한 답이다 — 이후 모든 장이 이 네 칸을 채운다.

## 1.3 Rosetta-Stone 사전

아래 표가 이 책의 사전이다 — 이후 모든 장이 새 개념 도입 시 재사용한다. 셋째 열이 가장 중요하다: **동일**(같은 대상의 재서술)·**유비**(정확하나 새 요소 있음)·**차이가 논점**(직관을 그대로 이식하면 틀림)을 구분하지 않으면 잘못된 직관을 이식받는다.

<!-- FIG: ch01/fig-01-two-loops -->

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

표의 **동일·유비** 행은 해당 장이 숫자로 회수한다. 지금 붙일 것은 **⚠ 차이가 논점**인 셋뿐이다. (i) FlashAttention tiling은 bit-exact 재배열이지만 chunk 안 gradient는 모두 chunk 시작 state에서 평가되므로(식 (M4)의 stale-snapshot 근사) **$C$가 계산되는 함수 자체를 바꾼다** — $C$가 semantic hyperparameter라는 사실(명제화 → 9장)이 [TNT] 한 편의 존재 이유다. (ii) per-request fast weights는 shared-weight batching의 전제를 깨서 decode를 grouped-GEMM으로 만든다(→ 10장). (iii) inner loop에서는 훈련의 batch 축을 **sequence 축이 대신한다**(inner mini-batch = chunk) — outer의 example 축인지 inner의 token 축인지 매번 물어야 한다(2장·9장).

## 1.4 두 개의 loop: 누가 무엇을 학습하는가

training 무경험 독자의 첫 질문은 정해져 있다: "test time에 학습한다면 그 learning rate는 누가 정하는가?" 답은 두 loop의 분업이다. 먼저 기준 수식(master update)을 미리 본다 — 유도는 2장·5장·12장이 조립하고 지금은 읽는 법만 익힌다.

$$
S_t \;=\; \beta_t\, S_{t-1} \;-\; \eta_t\, \nabla_W\, \ell\big(W_{t-1};\, k_t, v_t\big),
\qquad
W_t \;=\; \alpha_t\, W_{t-1} \;+\; S_t
\tag{M}
$$

읽기는 $y_t=\mathcal{M}(q_t;W_t)$. 기호를 독자의 어휘로 옮긴다.

- $\ell(W_{t-1};k_t,v_t)$ — **inner loss**: 기본형 $\|\mathcal{M}(k_t;W_{t-1})-v_t\|_2^2$, "이 쌍이 이미 잘 저장되어 있는가"의 per-token 측정기.
- $\nabla_W \ell$ — 그 실패가 가장 커지는 방향; 실제 write 신호는 반대 부호인 $-\eta_t\nabla_W\ell$이다(식 (M)의 $-\eta_t$ 항). [Titans]는 이 gradient를 **surprise**라 부른다(정의 → 12장).
- $S_t$ — gradient 누적 buffer, 즉 **momentum**(정의 → 2장). $S_t=\beta_t S_{t-1}+(\cdot)$은 scan kernel이 처리하는 선형 recurrence다.
- $\eta_t,\ \beta_t,\ \alpha_t$ — inner learning rate, momentum decay, retention gate(→ 13장). 시간 첨자가 곧 정보다(상수 아닌 **token마다 계산되는 gate**); $\alpha_t$는 남기는 비율($1$=유지, $0$=소거) = §1.3의 학습된 eviction.
- $\mathcal{M}(\cdot;W)$ — read 함수. 상태와 함수의 분리가 이 책의 규약이다.

식 (M) 한 줄 요약: **GD + momentum + weight decay를 token 스트림에서 돌리면 그것이 sequence layer다.** 처음 질문의 답: gate들은 slow weights가 만드는 token의 함수($\eta_t=\eta(x_t;\Theta)$)이고, $\Theta$(gate·projection·$W_{\mathrm{init}}$ 포함)는 outer loop가 inner loop 전체를 관통해 backpropagate하며 학습한다 — "무엇을 세게 쓰고 무엇을 잊을지"의 정책 자체를 배운다. 형식화(bilevel optimization)는 4장, 논문별 분업표는 Part II 각 장의 "outer vs inner" 절이 담당한다.

## 1.5 여섯 편의 지도

여섯 편은 한 연구 라인(Google, Behrouz 계열)의 연작이고, 각 편이 (M)의 어느 성분을 바꾸는지가 전체 지도다. [Miras] (arXiv:2504.13173; 제목은 *It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization*이지만 이 책은 framework 이름 Miras로 통칭)와 [Atlas] (*Atlas: Learning to Optimally Memorize the Context at Test Time*, arXiv:2505.23735)가 여기서 처음 등장한다.

표 1-2 — 여섯 편의 논문: 식 (M) 기준의 위치와 Part I 열쇠 장

| 논문 | (M)에서 바꾸는 것 | 한 줄 요약 | 열쇠가 되는 Part I 장 |
|---|---|---|---|
| [Titans] (2501.00663) | (M) 그 자체의 수립 | GD + momentum + weight decay가 test time의 deep memory 갱신 규칙이 된다; surprise가 write 강도를 정한다 | 2·5·6·8·9장 |
| [Miras] (2504.13173) | $\ell$ (attentional bias)과 retention 항 | 기존 모델 전부(RetNet·Mamba-2·DeltaNet·TTT·Titans 등)를 "online optimizer가 곧 sequence layer"라는 하나의 framework로 재유도 | 3·5·6장 |
| [Atlas] (2505.23735) | $\ell$의 범위(window $c$)와 inner optimizer(Muon/$\mathrm{NS}_\kappa$), capacity | per-token write를 sliding-window 목적함수(Omega rule)로 바꾸고, memory capacity를 feature map으로 끌어올린다 — test-time memorization의 정식화 | 2·3·5·9장 |
| [TNT] (2511.07343) | 훈련 경제학: chunk 크기 $C$의 speed–quality 충돌 | hierarchical global/local memory + 주기적 state reset + 2-stage training으로 큰 $C$ 훈련과 작은 $C$ serving을 양립 | 8·9·10장 |
| [NL] (2512.24695) | 층위: 2-level → K-level, update frequency | 모델과 훈련 절차 전체를 "각자 자기 주파수로 갱신되는 nested associative memory"의 스펙트럼으로 재구성 (Hope, CMS) | 2·4·5·11장 |
| [Sleep] (2606.03979) | lifecycle: train/test 경계 자체 | wake에 빠른 memory로 흡수하고, sleep에 consolidation·dreaming으로 느린 weights에 이관하는 주기적 생애 주기 | 4·11장 |

표 1-2를 세로로 읽으면 계보, 가로로 읽으면 사용법(각 행의 "열쇠 장" = 그 메커니즘을 여는 Part I 배경)이다. [Titans]는 자신의 update가 Gated DeltaNet, Longhorn, TTT layer, RWKV-7 등을 특수 사례로 포함한다고 논하므로 [Titans App. C], 식 (M)은 이 라인 바깥의 efficient-attention 계보까지 덮는 일반형이다. Part I는 의존성 순서로 읽는 것이 가장 싸다.

## 1.6 Worked micro-example: $2\times 2$ memory를 손으로 굴려 보기

숫자로 확인한다. $d_k=d_v=2$ memory에 직교하는 두 쌍 $k_A=(1,0)^\top\!\to(4,0)^\top$, $k_B=(0,1)^\top\!\to(0,2)^\top$를 Hebbian write($W\leftarrow W+vk^\top$)로 쌓으면 $Wk_A=v_A$는 정확하다. 그러나 직교하지 않는 셋째 쌍 $k_C=(1,1)^\top\!\to v_C=(1,1)^\top$를 누적한 $W'=\begin{pmatrix}5&1\\1&3\end{pmatrix}$에서는 $W'k_C=(6,4)^\top\ne v_C$로 방금 쓴 것조차 오염된다 — append-once 직관이 압축 memory에서 깨지는 crosstalk이다.

write를 gradient step으로 바꾸면 달라진다. inner loss $\ell(W;k_C,v_C)=\|Wk_C-v_C\|_2^2$의 gradient $\nabla_W\ell = 2\,(Wk_C-v_C)\,k_C^\top$는 오차와 key의 rank-1 outer product다. A·B 저장 상태(오차 $(3,1)^\top$)에서 $\eta_t=\tfrac14$로 한 step 밟으면 $W''k_C=(1,1)^\top=v_C$로 **방금 쓴 쌍이 정확히 회수된다** — write rule을 누적에서 gradient step으로 바꾼 것만으로 같은 크기 state의 품질이 달라졌다. 이것이 delta rule(→ 5장)이자 (M)의 최소형 (M1)이다. 공짜는 아니어서 A의 회수는 손상되고(capacity 질문 → 5장·14장), gate 3종의 수치 전개는 2장의 worked example에서 완성한다.

## 1.7 Systems bridge: 첫 back-of-envelope

마지막 페이지는 숫자다 — 아래 산수는 예시이며 특정 논문의 수치 주장이 아니다. 전형적 GQA 한 layer(KV head 8, head 차원 128, bf16)에서 KV cache는 token당 $2\times 8\times 128\times 2\,\mathrm{B} = 4\,\mathrm{KiB}$ — 128K context면 layer당 512 MiB, 32-layer면 request 하나가 16 GiB다. 반면 같은 구성의 matrix memory(TTT-Linear급, head당 $128\times128$)는 32 layer 전체에 8 MiB로 **context 길이와 무관하게 고정**이다 — 이 라인이 long-context에서 갖는 구조적 지렛대다.

대신 네 가지를 지불한다: decode에 들어온 backward·update(token당 연산 ≈3×; backward ≈ forward의 2배 → 2장), state 전체의 read-modify-write 트래픽, shared-weight batching을 깨는 grouped-GEMM decode, chunk를 키우면 함수가 바뀌어 떨어지는 훈련 품질 — [TNT]는 작은 chunk의 deep memory 훈련이 peak 대비 5–10% 미만 FLOPs utilization로 흔히 떨어진다고 보고한다 [TNT §1]. 정밀 계산과 roofline 배치는 10장 budget 표가 담당한다.

## 요약

- softmax attention + KV cache는 압축하지 않는(capacity 무한, non-parametric) associative memory이고, linear-RNN의 $d\times d$ state는 고정 크기 lossy memory다 — 모든 sequence 모델은 write (1-1)과 read (1-2)의 쌍이다 [Titans Eq. 6–7].
- 정의적 특징은 decode 안의 gradient step이다: "weights는 읽기 전용" 불변식이 fast weights $W$에는 폐기되고 slow weights $\Theta$에는 유지된다. inner loop는 $W_t$를 갱신하고, outer loop는 gate·projection·$W_{\mathrm{init}}$을 포함한 $\Theta$를 학습한다 — "누가 학습하는가"의 답은 항상 outer loop다.
- 여섯 논문은 기준 수식 (M)의 성분 — $\ell$·retention(Miras), window·optimizer(Atlas), 훈련 경제학(TNT), 층위(NL), lifecycle(Sleep) — 을 바꾼 것이다.
- write는 (오차)$\,k^\top$ 꼴의 rank-1/rank-$C$ GEMM이고 write rule의 선택(append/Hebbian/delta)이 곧 memory 품질의 선택이다($2\times2$ 손계산으로 확인). ⚠ 3대 함정: FlashAttention tiling ↔ chunk(bit-exact vs semantic), shared-weight batching의 붕괴, sequence 축이 batch 축을 대신한다는 것.
- 고정 크기 state의 대가는 decode 내 backward, state RMW 트래픽, grouped-GEMM decode, chunk 크기의 품질–MFU 긴장이다.

## 자가 점검 체크리스트

- [ ] KV cache와 $d\times d$ matrix state를 네 설계 질문(state / write / read / retention)으로 서술할 수 있다.
- [ ] "inner learning rate는 누가 학습하는가?"에 두 loop의 어휘로 답하고 $\eta_t = \eta(x_t;\Theta)$의 의미를 설명할 수 있다.
- [ ] 식 (M)의 각 기호($\ell$, $\nabla_W\ell$, $S_t$, $\eta_t/\beta_t/\alpha_t$, $\mathcal{M}$)를 inference 어휘로 옮길 수 있다.
- [ ] §1.6의 Hebbian/delta write를 손으로 재계산하고 delta write가 $k_C$ 회수를 정확하게 만든 이유를 말할 수 있다.
- [ ] 표 1-1의 임의 행의 대응 성격(동일/유비/차이 — 예: bit-exact tiling vs semantic chunk)을 판별하고 그 개념을 inference 어휘로 옮길 수 있다.

## 다음 장으로

이 장은 "write가 gradient step이다"라고 선언만 했다 — gradient가 무엇이고 어떻게 계산되는지는 아직 블랙박스다. 2장은 그 블랙박스를 연다: backward pass를 밑바닥부터 세우고, SGD·momentum·AdamW·Muon을 각각 (state, update, cost)를 갖는 객체로 정의한다. 그 과정에서 이 장의 복선 두 개가 회수된다: momentum buffer가 linear-RNN state와 같은 대수를 탄다는 것, 그리고 weight decay의 $(1-\eta\lambda)$가 retention gate와 같은 물건이라는 것. 식 (M)의 부품이 손에 잡히고 나면, Titans의 memory update는 "처음 보는 수식"이 아니라 "아는 부품의 재배선"이 된다.
