# ch01. Orientation: inference 세계와 learning 세계의 Rosetta Stone

> **이 장의 목표** — 이 장을 마친 독자는 다음을 할 수 있다.
> 1. KV cache와 linear-RNN state를 memory의 write/read 어휘로 재서술하고, softmax attention을 정확한 의미로 "압축하지 않는 memory"라고 부를 수 있다.
> 2. fast/slow weights와 inner/outer loop를 구분하고, "이 learning rate는 누가 학습하는가?"에 loop의 이름으로 답할 수 있다.
> 3. Rosetta-Stone 사전(표 1-1)의 각 대응이 **동일**인지 **유비**인지 **차이가 논점**인지 판별할 수 있다.
> 4. 여섯 논문이 기준 수식 (M)의 어느 성분을 바꾸는지를 한 장의 지도로 그릴 수 있다.
>
> **왜 필요한가** — 여섯 편 전부가 이 장을 전제한다. [Titans] (*Titans: Learning to Memorize at Test Time*, arXiv:2501.00663) §2는 attention과 현대 linear recurrent 모델 전부를 memory unit의 write/read 연산으로 재서술한다([Titans Eq. 6–7]) — 이 장의 사전은 그 재서술을 독자의 KV cache 어휘에 접속한다. [TNT] (*TNT: Improving Chunkwise Training for Test-Time Memorization*, arXiv:2511.07343)의 중심 논증은 §1.3에서 세우는 "chunk는 FlashAttention tile과 달리 계산되는 함수 자체를 바꾼다"는 구분 위에 서 있다. [NL] (*Nested Learning: The Illusion of Deep Learning Architecture*, arXiv:2512.24695)과 [Sleep] (*Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories*, arXiv:2606.03979)의 "momentum도 Adam도 backprop도 전부 associative memory다"라는 주장은 두 loop의 그림(§1.4) 없이는 범주 오류로 읽힌다.

## 1.1 두 세계: weights가 얼어 있는 세계와 움직이는 세계

독자의 세계는 하나의 불변식 위에 서 있다: **forward는 weights를 읽기만 한다.** weights는 로드 시점에 배치되어 여러 request가 공유하고, request 사이의 유일한 가변 상태는 append-only KV cache다. prefill/decode 비대칭, continuous batching, paged attention — 전부 이 전제의 파생물이다.

이 라인은 그 불변식의 폐기가 정의적 특징이다. sequence layer가 decode 한 step마다 자기 내부의 작은 신경망 weights를 **gradient step으로 갱신한다** — token 하나의 처리가 곧 그 네트워크의 forward, loss, backward, update다. 직계 원형인 Sun et al. 2024 (arXiv:2407.04620)는 이를 "hidden state가 모델 그 자체이고, update rule이 self-supervised learning의 한 step"이라는 구도로 요약한다 [2407.04620 abstract·§1].

움직이는 것은 weights 전부가 아니다. 두 종류의 parameter가 공존한다.

- **fast weights** — sequence 처리 동안 token마다 갱신되는 작은 네트워크의 weights, 기호 $W$. $W_{\mathrm{init}}$에서 출발해 request가 끝나면 버려지거나 세션 상태로 관리된다(정식 정의와 계보 → 6장).
- **slow weights** — 독자가 아는 통상적 parameter(projection 행렬, backbone, gate 생성 head), 기호 $\Theta$. pretraining 후 얼어붙고, serving 관점에서 여전히 읽기 전용이다.

즉 폐기되는 것은 "모든 weights가 얼어 있다"이고 유지되는 것은 "$\Theta$는 얼어 있다"이다 — 이 대문자 구분($W$ vs $\Theta$)이 이 책의 가장 중요한 표기 결정이다. 두 parameter는 두 학습 과정에 속한다: **inner loop**는 decode와 같은 시간축에서 token마다 $W_t$를 갱신하고, **outer loop**는 통상적 pretraining으로서 inner loop 전체를 관통해 $\Theta$를 학습한다(형식화 → 4장). 미리 끊어 둘 오해: 이것은 fine-tuning이 아니다 — label도 훈련 job도 없이 layer의 forward 안에서 일어나는 내장 연산이며, 독자의 직관으로 가장 가까운 상은 **"KV cache에 압축 codec이 달렸고, 그 인코딩 연산이 gradient step"**이다.

## 1.2 모든 sequence 모델은 이미 memory였다

[Titans] §2의 재서술이 이 라인의 출발점이다: 임의의 recurrent 계열 sequence 모델은 memory unit에 대한 write 연산과 read 연산의 쌍이다 [Titans Eq. 6–7].

$$
W_t = f(W_{t-1}, x_t) \qquad \text{(write)}
\tag{1-1}
$$

$$
y_t = g(W_t, x_t) \qquad \text{(read)}
\tag{1-2}
$$

독자가 아는 두 극단이 이 추상화의 양 끝이다. **softmax attention + KV cache**: 상태는 쌍 $\{(k_\tau, v_\tau)\}_{\tau\le t}$의 목록, write는 append, read는 attention lookup이다. 이 memory는 **압축하지 않는다** — 무손실·capacity 무한 대신 state가 $O(L)$로 자라고 token당 read가 $O(L)$이다. 이 책은 이를 "capacity 무한의 non-parametric **associative memory**"(key로 연관 value를 회수하는 저장 구조; 고전 정식화 → 5장, 공식 Definition → 13장)라 부른다. **linear attention / linear-RNN state**: 상태는 고정 크기 행렬 $W\in\mathbb{R}^{d_v\times d_k}$, write는 outer-product 누적 $W_t = W_{t-1} + v_t k_t^\top$, read는 $y_t = W_t q_t$다. 이 memory는 **압축한다** — state와 token당 비용이 고정되는 대신, 직교하지 않는 key들이 겹쳐 쓰이며 회수가 오염된다(crosstalk; 정량화 → 5장, §1.6에서 체험).

lossless-growing이냐 lossy-fixed냐 — 독자가 KV cache 폭발과 압축·eviction 기법으로 이미 운영 감각으로 아는 trade다. 이 라인의 출발 질문: **write rule $f$를 optimizer의 한 step으로 만들면, 고정 크기 state의 손실을 얼마나 줄일 수 있는가?** 이 관점에서 모든 sequence layer는 네 질문 — state / write / read / retention — 에 대한 답이고, 이후 모든 장이 이 네 칸을 채운다.

## 1.3 Rosetta-Stone 사전

아래 표가 이 책의 사전이다. 이후 모든 장은 새 개념 도입 시 이 대응을 재사용하며, 셋째 열이 가장 중요하다: 대응이 **동일**(같은 대상의 재서술)인지 **유비**(정확하지만 새 요소가 있음)인지 **차이가 논점**(직관을 그대로 이식하면 틀림)인지 구분하지 않으면 독자는 잘못된 직관을 이식받는다.

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

표의 해설은 세 묶음만 덧붙인다 — 나머지 행은 해당 장이 회수한다.

**동일 — GEMM과 scan.** gradient는 $\nabla_W\ell = (\text{오차})\,k^\top$의 rank-1 outer product이고 token $C$개를 모으면 rank-$C$ GEMM이다 — 훈련 수식을 GEMM shape로 읽는 것이 이 책의 교수법이다. momentum은 $S_t=\beta_tS_{t-1}+(\cdot)$ 꼴의 선형 recurrence로, 독자의 prefix-sum/associative scan kernel의 대수 그대로다(→ 9장).

**유비 — 새 요소가 하나씩.** prefill은 큰 chunk의 병렬 write로, decode는 $C=1$의 per-token online write + read로 옮겨진다 — 단 **decode 안에 backward pass가 들어온다**는 것이 신세계다. retention gate(남길 비율 $\alpha_t\in[0,1]$; 공식 정의 → 13장)는 정확히 "학습된 eviction policy"다. 그리고 KV cache write는 append-once지만 fast-weight write는 read-modify-write이며, delta rule(→ 5장)은 append가 아니라 overwrite다 — 이 차이를 §1.6에서 숫자로 확인한다.

**차이가 논점 — ⚠ 셋.** (i) FlashAttention tiling은 수학적으로 동일한 함수의 재배열이지만, chunkwise-parallel training(→ 9장)의 chunk 안에서는 모든 gradient가 chunk 시작 시점의 state에서 평가된다(식 (M4)의 stale-snapshot 근사) — 따라서 **$C$를 바꾸면 계산되는 함수 자체가 바뀐다.** $C$는 semantic hyperparameter이며(명제화 → 9장), 이 사실이 [TNT] 한 편의 존재 이유다. (ii) per-request fast weights는 "모든 request가 같은 weights를 곱한다"는 continuous batching의 전제를 깨서 decode를 grouped-GEMM으로 만든다(→ 10장). (iii) 훈련의 batch 축 역할을 inner loop에서는 **sequence 축이 대신한다** — inner mini-batch가 곧 chunk다. batch를 만날 때마다 outer의 example 축인지 inner의 token 축인지 물어야 한다(2장·9장에서 반복 경고).

## 1.4 두 개의 loop: 누가 무엇을 학습하는가

training 무경험 독자의 첫 질문은 정해져 있다: "test time에 학습한다면, 그 learning rate는 누가 정하는가?" 답은 두 loop의 분업이다. 먼저 이 책의 기준 수식(master update)을 미리 본다 — 유도는 2장·5장·12장이 조립하고, 지금은 읽는 법만 익힌다.

$$
S_t \;=\; \beta_t\, S_{t-1} \;-\; \eta_t\, \nabla_W\, \ell\big(W_{t-1};\, k_t, v_t\big),
\qquad
W_t \;=\; \alpha_t\, W_{t-1} \;+\; S_t
\tag{M}
$$

읽기는 $y_t=\mathcal{M}(q_t;W_t)$이다. 기호를 독자의 어휘로 옮긴다.

- $\ell(W_{t-1};k_t,v_t)$ — **inner loss**: memory가 key $k_t$에서 value $v_t$를 얼마나 재구성하지 못하는지 재는 per-token loss. 기본형은 $\|\mathcal{M}(k_t;W_{t-1})-v_t\|_2^2$ — "이 쌍이 이미 잘 저장되어 있는가"의 측정기다.
- $\nabla_W \ell$ — 그 실패를 줄이는 방향, 즉 이번 token의 write 신호. [Titans]는 이것을 **surprise**라고 부른다(공식 정의 → 12장).
- $S_t$ — gradient의 누적 buffer, 즉 **momentum**(객체로서의 정의 → 2장). $S_t=\beta_t S_{t-1}+(\cdot)$은 scan kernel이 바로 처리하는 선형 recurrence다.
- $\eta_t,\ \beta_t,\ \alpha_t$ — 순서대로 inner learning rate, momentum decay, retention gate. 시간 첨자 자체가 정보다: 상수가 아니라 **token마다 계산되는 gate**다. $\alpha_t$는 남기는 비율($1$=전부 유지, $0$=완전 소거) — §1.3의 학습된 eviction이 계수 하나로 구현된 것이다.
- $\mathcal{M}(\cdot;W)$ — read 함수. 상태와 함수의 분리가 이 책의 규약이다.

식 (M)을 한 문장으로: **GD + momentum + weight decay를 token 스트림 위에서 돌리면, 그것이 곧 sequence layer다.**

처음 질문의 답: gate들은 slow weights가 만드는 token의 함수다 — $\eta_t=\eta(x_t;\Theta)$ 같은 형태로, gate를 계산하는 head의 parameter가 $\Theta$의 일부다. 그리고 $\Theta$는 outer loop가 inner loop 전체를 관통해 backpropagate하며 학습한다 — "어떤 token에서 세게 쓰고, 무엇을 잊을지"의 정책 자체를 데이터로부터 배운다. 요약하면: **inner loop는 $W$를 움직이고, outer loop는 inner loop의 hyperparameter를 포함한 $\Theta$를 학습한다.** projection $W_K,W_V,W_Q$, gate 생성 head, 초기 상태 $W_{\mathrm{init}}$ — 전부 outer의 소유물이다. 형식화(bilevel optimization)는 4장이, 논문별 분업표는 Part II 각 장의 "outer vs inner" 절이 담당한다.

## 1.5 여섯 편의 지도

여섯 편은 한 연구 라인(Google Research, Behrouz 계열)의 연작이며, 각 편이 식 (M)의 어느 성분을 바꾸는지가 전체 지도다. [Miras] (arXiv:2504.13173; 논문 제목은 *It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization*이지만 이 책은 framework 이름 Miras로 통칭한다)와 [Atlas] (*Atlas: Learning to Optimally Memorize the Context at Test Time*, arXiv:2505.23735)가 여기서 처음 등장한다.

표 1-2 — 여섯 편의 논문: 식 (M) 기준의 위치와 Part I 열쇠 장

| 논문 | (M)에서 바꾸는 것 | 한 줄 요약 | 열쇠가 되는 Part I 장 |
|---|---|---|---|
| [Titans] (2501.00663) | (M) 그 자체의 수립 | GD + momentum + weight decay가 test time의 deep memory 갱신 규칙이 된다; surprise가 write 강도를 정한다 | 2·5·6·8·9장 |
| [Miras] (2504.13173) | $\ell$ (attentional bias)과 retention 항 | 기존 모델 전부(RetNet·Mamba-2·DeltaNet·TTT·Titans 등)를 "online optimizer가 곧 sequence layer"라는 하나의 framework로 재유도 | 3·5·6장 |
| [Atlas] (2505.23735) | $\ell$의 범위(window $c$)와 inner optimizer(Muon/$\mathrm{NS}_\kappa$), capacity | per-token write를 sliding-window 목적함수(Omega rule)로 바꾸고, memory capacity를 feature map으로 끌어올린다 — test-time memorization의 정식화 | 2·3·5·9장 |
| [TNT] (2511.07343) | 훈련 경제학: chunk 크기 $C$의 speed–quality 충돌 | hierarchical global/local memory + 주기적 state reset + 2-stage training으로 큰 $C$ 훈련과 작은 $C$ serving을 양립 | 8·9·10장 |
| [NL] (2512.24695) | 층위: 2-level → K-level, update frequency | 모델과 훈련 절차 전체를 "각자 자기 주파수로 갱신되는 nested associative memory"의 스펙트럼으로 재구성 (Hope, CMS) | 2·4·5·11장 |
| [Sleep] (2606.03979) | lifecycle: train/test 경계 자체 | wake에 빠른 memory로 흡수하고, sleep에 consolidation·dreaming으로 느린 weights에 이관하는 주기적 생애 주기 | 4·11장 |

세로로 읽으면 계보다: [Titans]가 기준 수식을 세우고, [Miras]가 일반 이론으로 넓히고, [Atlas]가 목적함수와 optimizer를 강화하고, [TNT]가 훈련 비용 문제를 풀고, [NL]이 전체를 K-level로 일반화하고, [Sleep]이 train/test 구분 자체를 없앤다. 참고로 [Titans]는 자신의 update가 Gated DeltaNet, Longhorn, TTT layer, RWKV-7 등 기존 recurrent 모델들을 특수 사례로 포함한다고 논한다 [Titans App. C] — 식 (M)은 이 라인 바깥의 efficient-attention 계보까지 덮는 일반형이다. 가로로 읽으면 사용법이다: 각 논문 행의 "열쇠 장"이 그 메커니즘을 여는 Part I 배경이다. Part I의 장 순서는 의존성 순서이므로 순서대로 읽는 것이 가장 싸다.

## 1.6 Worked micro-example: $2\times 2$ memory를 손으로 굴려 보기

말로 세운 사전을 숫자로 확인한다. $d_k=d_v=2$의 matrix memory $W\in\mathbb{R}^{2\times 2}$에 두 쌍 $k_A=(1,0)^\top, v_A=(4,0)^\top$과 $k_B=(0,1)^\top, v_B=(0,2)^\top$를 Hebbian write($W\leftarrow W+vk^\top$)로 쓰면 $W=\begin{pmatrix}4&0\\0&2\end{pmatrix}$이고, read는 $Wk_A=v_A$로 정확하다 — key가 서로 직교하기 때문이다. 여기에 직교하지 않는 셋째 쌍 $k_C=(1,1)^\top, v_C=(1,1)^\top$를 마저 누적하면 $W'=\begin{pmatrix}5&1\\1&3\end{pmatrix}$이 되어 $W'k_A=(5,1)^\top\ne v_A$이고, 방금 쓴 것조차 $W'k_C=(6,4)^\top\ne v_C$다 — 저장들이 겹쳐 오염되는 crosstalk이며, append-once 직관이 압축 memory에서 깨지는 지점이 정확히 여기다.

write를 gradient step으로 바꾸면 결과가 달라진다. inner loss $\ell(W;k_C,v_C)=\|Wk_C-v_C\|_2^2$의 gradient는 $\nabla_W\ell = 2\,(Wk_C-v_C)\,k_C^\top$ — 오차 벡터와 key의 rank-1 outer product다. A·B 저장 상태에서 오차는 $(3,1)^\top$이고, $\eta_t=\tfrac14$로 한 step 밟으면 $W''=\begin{pmatrix}2.5&-1.5\\-0.5&1.5\end{pmatrix}$, 이제 $W''k_C=(1,1)^\top=v_C$ — **방금 쓴 쌍이 정확히 회수된다.** 같은 고정 크기 state에서 write rule을 누적에서 gradient step으로 바꾼 것만으로 저장 품질이 달라졌다. 이것이 delta rule(정식 전개 → 5장)이자 (M)의 최소형 (M1)이다. 공짜는 아니다: $W''k_A=(2.5,-0.5)^\top$로 A의 회수는 손상됐고(몇 쌍까지 버티는가가 capacity 질문 → 5장·14장), $\eta_t=\tfrac12$로 키우면 목표를 지나쳐 반대편으로 넘어간다. gate 3종이 이 그림에 각각 무엇을 더하는지의 수치 전개는 2장의 worked example에서 완성한다.

가져갈 것은 두 가지다: **write는 계산이고** 그 모양은 (오차)$\,k^\top$의 rank-1(모으면 rank-$C$) GEMM이라는 것, 그리고 write rule의 선택(append/Hebbian/delta)이 곧 memory 품질의 선택이라는 것.

## 1.7 Systems bridge: 첫 back-of-envelope

사전의 마지막 페이지는 숫자다. 아래 산수는 이 책의 예시 계산이며 특정 논문의 수치 주장이 아니다. **state 크기**: 전형적 GQA 구성 한 layer(KV head 8개, head 차원 128, bf16)에서 KV cache는 token당 layer당 $2\times 8\times 128\times 2\,\mathrm{B} = 4\,\mathrm{KiB}$ — 128K context면 layer당 512 MiB, 32-layer 모델이면 request 하나가 16 GiB다. 같은 head 구성의 matrix memory(TTT-Linear급, head당 $128\times128$)는 32 layer 전체에 8 MiB로 **context 길이와 무관하게 고정**이다. 이것이 이 라인이 long-context에서 갖는 구조적 지렛대다.

대신 지불하는 것이 네 가지다. (i) decode 안에 backward와 update가 들어와 memory 모듈의 token당 연산이 대략 3× 이상으로 분다(backward ≈ forward의 2배 규칙, 유도 → 2장). (ii) state가 read-modify-write 대상이 되어, append 4 KiB가 아니라 state 전체가 매 token 왕복하는 트래픽이 붙는다. (iii) per-request state가 shared-weight batching을 깨뜨려 decode를 grouped-GEMM으로 만든다. (iv) 훈련은 chunkwise 병렬화에 의존하는데, chunk를 키우면 함수 자체가 바뀌어 품질이 떨어진다 — [TNT]는 deep memory 훈련이 작은 chunk에서 peak 대비 5–10% 미만의 FLOPs utilization로 떨어지는 경우가 흔하다고 보고한다 [TNT §1]. 이 네 항목의 back-of-envelope 계산과 정밀한 roofline 배치는 10장 budget 표가 담당한다.

## 요약

- softmax attention + KV cache는 압축하지 않는(capacity 무한, non-parametric) associative memory이고, linear-RNN의 $d\times d$ state는 고정 크기 lossy memory다. 모든 sequence 모델은 write (1-1)과 read (1-2)의 쌍이다 [Titans Eq. 6–7].
- 정의적 특징은 decode 안의 gradient step이다: "weights는 읽기 전용" 불변식은 fast weights $W$에 대해 폐기되고, slow weights $\Theta$에 대해 유지된다.
- inner loop는 $W_t$를 token마다 갱신하고, outer loop는 gate·projection·$W_{\mathrm{init}}$을 포함한 $\Theta$를 학습한다 — "누가 학습하는가"의 답은 항상 outer loop다.
- 기준 수식 (M)은 "GD + momentum + weight decay가 곧 sequence layer"라는 라인 전체의 요약이며, 여섯 논문은 (M)의 성분 — $\ell$과 retention(Miras), window와 optimizer(Atlas), 훈련 경제학(TNT), 층위(NL), lifecycle(Sleep) — 을 바꾼 것이다.
- ⚠ 3대 함정: FlashAttention tiling ↔ chunk(bit-exact vs semantic), shared-weight batching의 붕괴, sequence 축이 batch 축을 대신한다는 것.
- write는 (오차)$\,k^\top$ 꼴의 rank-1/rank-$C$ GEMM이며, write rule의 선택(append/Hebbian/delta)이 곧 memory 품질의 선택이다 — $2\times2$ 손계산으로 확인했다.
- 고정 크기 state의 대가는 decode 내 backward, state RMW 트래픽, grouped-GEMM decode, chunk 크기의 품질–MFU 긴장이다.

## 자가 점검 체크리스트

- [ ] KV cache와 $d\times d$ matrix state를 네 가지 설계 질문(state / write / read / retention)으로 서술할 수 있다.
- [ ] "inner learning rate는 누가 학습하는가?"에 두 loop의 어휘로 답하고, $\eta_t = \eta(x_t;\Theta)$의 의미를 설명할 수 있다.
- [ ] 식 (M)의 각 기호($\ell$, $\nabla_W\ell$, $S_t$, $\eta_t/\beta_t/\alpha_t$, $\mathcal{M}$)를 inference 어휘로 한 문장씩 옮길 수 있다.
- [ ] §1.6의 Hebbian/delta write를 손으로 재계산하고, delta write가 $k_C$의 회수를 정확하게 만든 이유를 말할 수 있다.
- [ ] FlashAttention tiling과 chunk가 왜 다른지(bit-exact 재배열 vs stale-snapshot 근사) 설명할 수 있다.
- [ ] 표 1-1의 임의 행의 대응 성격(동일/유비/차이)을 판별하고, 그 개념을 inference 어휘로 옮길 수 있다.

## 다음 장으로

이 장은 "write가 gradient step이다"라고 선언만 했다 — gradient가 무엇이고 어떻게 계산되는지는 아직 블랙박스다. 2장은 그 블랙박스를 연다: backward pass를 밑바닥부터 세우고, SGD·momentum·AdamW·Muon을 각각 (state, update, cost)를 갖는 객체로 정의한다. 그 과정에서 이 장의 복선 두 개가 회수된다: momentum buffer가 linear-RNN state와 같은 대수를 탄다는 것, 그리고 weight decay의 $(1-\eta\lambda)$가 retention gate와 같은 물건이라는 것. 식 (M)의 부품이 손에 잡히고 나면, Titans의 memory update는 "처음 보는 수식"이 아니라 "아는 부품의 재배선"이 된다.
