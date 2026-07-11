# ch08. TTT lineage: test-time adaptation에서 TTT-Linear/TTT-MLP와 dual form까지

> **이 장의 목표** — 이 장을 마치면 다음을 할 수 있어야 한다.
>
> 1. **test-time training (TTT)**을 "hidden state = 작은 model의 weights, update rule = 그 model의 learning step"이라는 한 문장으로 정의하고, 이를 식 (M1)의 원형으로 쓸 수 있다.
> 2. TTT-Linear의 update를 손으로 전개해 delta rule(→ 5장)·DeltaNet(→ 6장)과의 동일성을 유도할 수 있다.
> 3. **dual form**을 유도하고, chunk 크기 $C$가 kernel tuning 파라미터가 아니라 계산되는 함수 자체를 바꾸는 근사임을 $d=2$ 손계산으로 확인할 수 있다.
> 4. TTT layer 하나의 decode 비용을 GEMM/GEMV 단위로 세고, serving engine에 무엇이 새로 필요한지 말할 수 있다.
>
> **왜 필요한가** — 6편 전부가 이 장을 직접 전제한다. [Titans] (*Titans: Learning to Memorize at Test Time*, arXiv:2501.00663)는 자신의 memory module을 "generalized TTT layer"로 규정하고 [Titans App. C], TTT의 mini-batch tensorization 위에 자기 병렬화를 쌓는다 [Titans §3.2]. [Miras] (*It's All Connected*, arXiv:2504.13173; 이 책은 framework 이름 Miras로 통칭)는 TTT-Linear/TTT-MLP를 taxonomy의 기준 행으로 분류한다 [Miras Table 1]. [Atlas] (*Atlas: Learning to Optimally Memorize the Context at Test Time*, arXiv:2505.23735)의 Omega rule(→ 14장)은 TTT의 per-token inner objective를 window로 넓힌 것이고, [TNT] (*TNT: Improving Chunkwise Training for Test-Time Memorization*, arXiv:2511.07343)는 자기 약어를 "TTT iNside TTT"로도 읽을 수 있다고 밝힌다 [TNT footnote 1]. [NL] (*Nested Learning*, arXiv:2512.24695)은 TTT의 2-level 구조를 K-level로, [Sleep] (*Language Models Need Sleep*, arXiv:2606.03979)은 TTT가 fast weights에 쓴 것을 slow weights로 옮기는 lifecycle로 확장한다. 요컨대 Part II의 여섯 장은 전부 "TTT에서 무엇을 바꿨는가"라는 질문으로 조직되며, 그 비교의 기준선이 이 장이다.

## 8.1 전사: inference 중의 weight update는 이미 검증된 기법이었다

독자의 세계에서 가장 단단한 불변식부터 짚는다. serving engine의 전제는 "checkpoint를 로드한 뒤 weights는 read-only"다. KV cache는 자라고 페이지가 스왑되지만, weight tensor에 store 연산이 들어가는 일은 없다. 이 라인의 논문들은 바로 그 불변식을 폐기하는데, 이 폐기는 2024년의 발명이 아니라 vision의 distribution-shift 문헌에서 이미 십수 년 가까이 다듬어진 기법의 연장이다. 이 절의 목적은 하나다: "production에서 그런 걸 한다고?"라는 독자의 본능적 반응을, 선행 사례로 미리 무장해제하는 것.

**test-time adaptation**은 deployment 중 만나는 입력에 맞춰 model의 일부 parameter를 그 자리에서 갱신하는 기법 family를 가리킨다. 이름의 직계 조상은 Sun et al. 2020 (arXiv:1909.13231)의 test-time training이다. 구조는 Y자형이다: 공유 feature extractor 위에 (a) 본 task head와 (b) self-supervised task head(입력 이미지의 회전 각도 예측)를 함께 올려 훈련한다. test 시점에는 label이 없으므로 본 task loss는 계산할 수 없지만, 회전 예측 loss는 입력만으로 계산된다. 그래서 test 입력 하나가 들어올 때마다 그 loss의 gradient로 공유 extractor를 몇 step 갱신한 뒤 예측한다. 핵심 통찰은 **label 없이도 학습 신호를 만들 수 있다**는 것이다 — self-supervision이 test time의 학습을 가능하게 하는 열쇠다.

Wang et al. 2021 (arXiv:2006.10726)의 TENT는 같은 아이디어의 더 가벼운 판본이다. 별도 auxiliary head조차 없이, 예측 분포의 entropy를 loss로 삼아 normalization layer의 scale/shift parameter만 갱신한다. 원 훈련 데이터에도, 훈련 절차에도 접근하지 않는 "fully test-time" 설정이다. 두 기법을 한 줄로 요약하면 이렇다: test 입력 $x$에 대해 어떤 보조 loss $\ell_{\mathrm{aux}}$를 정의할 수 있는 한, $W \leftarrow W - \eta\,\nabla_W \ell_{\mathrm{aux}}(W; x)$는 언제나 실행 가능한 연산이다. "training"과 "inference"의 구분은 시스템 설계의 관행이지 수학의 제약이 아니다.

systems 관점의 접점: 이 vision 기법들의 갱신 단위는 image 하나였다. 2024년의 TTT layer는 그 단위를 token 하나 — 독자의 어휘로는 decode step 하나 — 로 좁히고, 갱신 대상을 backbone이 아니라 layer 안에 내장된 작은 memory network로 한정한 것이다. 즉 이 라인은 "inference 중 weight update"라는 이미 존재하던 기법을, sequence modeling의 state update 자리에 이식했다.

## 8.2 TTT layer: hidden state가 곧 model이다

Sun et al. 2024 (arXiv:2407.04620, *Learning to (Learn at Test Time): RNNs with Expressive Hidden States*)의 출발점은 RNN state의 표현력 문제다. 7장까지의 여정을 state의 자료구조로 요약하면: Mamba류의 vector state($d$차원), linear attention류의 matrix state($d\times d$, 6장)가 있었다. 다음 단계는 무엇인가? Sun et al. 2024의 답: **state를 아예 작은 neural network의 weights 전체로 승격시킨다.** 이 논문의 슬로건 — hidden state is a model, and the update rule is a step of self-supervised learning — 이 이 라인 전체의 설계 원리다. 이 책의 표기로, 그 작은 network가 $\mathcal{M}(\cdot\,;W)$이고 그 weights $W$가 fast weights(→ 6장)다.

TTT layer는 세 요소로 정의된다. 첫째, **inner objective**. 원문은 이를 self-supervised reconstruction으로 서술한다: token $x_t$의 한 view(training view) $k_t = W_K x_t$를 입력으로 받아 다른 view(label view) $v_t = W_V x_t$를 복원하도록 memory를 학습시킨다.

$$
\ell(W; k_t, v_t) \;=\; \big\|\,\mathcal{M}(k_t;W) - v_t\,\big\|_2^2
\tag{8-1}
$$

> **[해설]** 식 (8-1)은 5장의 associative-memory regression과 문자 그대로 같은 식이다. "self-supervised reconstruction"(TTT의 언어)과 "key→value 연상 저장"(associative memory의 언어)은 같은 objective의 두 이름이며, 이 동일성이 Miras가 여섯 모델 family를 한 표에 넣을 수 있는 이유다(→ 13장). 독자의 어휘로는: KV cache가 무손실로 보관하던 $(k,v)$ 쌍을, 여기서는 regression으로 $W$에 눌러 담는다.

둘째, **update rule**. 매 token마다 (8-1)에 대한 1-step gradient descent를 실행한다. 이것이 표준형 (M1) 그대로다:

$$
W_t = W_{t-1} - \eta_t\,\nabla_W\,\ell(W_{t-1};k_t,v_t)
$$

여기서 $\eta_t$는 상수가 아니라 **학습된 data-dependent inner learning rate**, 즉 $\eta_t = \eta(x_t;\Theta)$ 형태로 slow weights가 token마다 산출하는 게이트다. 독자에게 이 게이트의 정확한 대응물은 Mamba의 input-dependent gate(→ 7장)다 — 같은 역할("이 token을 얼마나 강하게 state에 반영할 것인가")을 optimizer의 언어로 다시 말한 것뿐이다.
<!-- TODO-VERIFY: Sun et al. 2024의 learned inner lr의 정확한 함수형(η_t = η_base·σ(θ_lr·x_t)로 기억됨)과 위치. 확인 방법: arXiv:2407.04620 §2.4 부근 "learnable W" / "inner-loop learning rate" 검색 -->

셋째, **read**. 세 번째 view(test view) $q_t = W_Q x_t$로 갱신된 memory를 읽는다: $y_t = \mathcal{M}(q_t; W_t)$. 원문 표기 $\theta_K,\theta_V,\theta_Q$는 이 책의 $W_K,W_V,W_Q$에 대응한다.

$\mathcal{M}$의 구조에 따라 두 instantiation이 있다. **TTT-Linear**는 $\mathcal{M}(k;W)=Wk$, 즉 state가 linear attention과 같은 $d_v\times d_k$ 행렬이다. **TTT-MLP**는 $\mathcal{M}$이 2-layer MLP다 — 이후 라인이 표준화하는 deep memory(→ 12장)의 원형이다. 6장 카탈로그(표 6-2)의 TTT-Linear 행이 말하듯, 두 모델 다 (M1)이 전부다: momentum도 retention gate도 없다. 이 "없음"이 Part II를 여는 열쇠 구멍이다.
<!-- TODO-VERIFY: 원문 f의 정확한 구조(residual + LayerNorm 포함 여부, TTT-MLP의 hidden 배수·activation). 확인 방법: arXiv:2407.04620 §2.6 또는 App. 아키텍처 절 -->

Rosetta-Stone으로 옮기면 TTT layer의 fast weights는 **"compression codec이 달린 writable KV cache"**다. KV cache는 append-only 무손실 저장에 $O(L)$ lookup이고, $W_t$는 고정 크기 lossy 저장에 $O(1)$ lookup(GEMV 한 번)이다. 결정적 차이는 write 경로다: cache append는 memcpy지만, TTT의 write는 read-modify-write이며 그 "modify"가 gradient 계산이다. 1장에서 예고한 문장을 여기서 처음 실감하게 된다 — **backward pass가 decode 안으로 들어온다.**

## 8.3 TTT-Linear는 delta rule이다

TTT-Linear의 update를 전개하면 이 layer가 새 발명품이 아니라 5장·6장의 재발견임이 드러난다. $\mathcal{M}(k;W)=Wk$를 (8-1)에 넣고 $W$로 미분하면 $\nabla_W \ell = 2\,(Wk_t - v_t)\,k_t^\top$ — 예측 오차 벡터와 key의 outer product, 즉 rank-1 행렬이다. 이후 식에서는 상수 2를 학습되는 게이트 $\eta_t$에 흡수한다(게이트가 어차피 outer loop가 정하는 함수이므로 무해하다). 그러면 (M1)은

$$
W_t \;=\; W_{t-1}\big(I - \eta_t\,k_tk_t^\top\big) \;+\; \eta_t\,v_tk_t^\top
\tag{8-2}
$$

가 된다. 이는 6장의 DeltaNet 행과 문자 그대로 동일하다 — **TTT-Linear는 gate 없는 DeltaNet이고, 둘 다 Widrow–Hoff delta rule(→ 5장)의 sequence-layer 판본이다.** [Miras Table 1]도 정확히 이렇게 분류한다: TTT-Linear = (matrix memory, $\ell_2$ attentional bias, retention 없음, GD), TTT-MLP = (2-layer MLP, $\ell_2$, retention 없음, GD). 같은 표에서 DeltaNet과의 차이는 memory 구조와 유래뿐이다.

의미론도 5장 그대로다. Hebbian write($W \mathrel{+}= v_tk_t^\top$)는 값을 무조건 더해 crosstalk를 쌓지만, delta write는 **residual** $v_t - W_{t-1}k_t$만 쓴다. 이미 알고 있는 내용이면 gradient가 작아 거의 쓰지 않고, 어긋난 만큼만 고쳐 쓴다 — 같은 key가 다시 오면 add가 아니라 overwrite다. 이 "오차 기반 write"가 Titans가 surprise라고 부르게 될 것의 원형이다(→ 12장).

두 극한이 이 그림을 완성한다. 첫째, chunk 하나로 sequence 전체를 잡고 모든 gradient를 초기 상태 $W_0$에서 평가하면(다음 절의 언어로 $C=L$의 stale 극한), $W_0=0$일 때 TTT-Linear는 vanilla linear attention과 같은 함수가 된다 — [Sun et al. 2024]가 정리로 제시하는 동치다. 둘째, parametric model 대신 non-parametric learner(kernel regression)를 inner learner로 넣으면 softmax attention이 나온다 — 5장이 확립한 "softmax attention = $\ell_2$ regression의 non-parametric Nadaraya–Watson 해"의 TTT 판본이다(→ 5장). 즉 attention도 linear attention도 TTT framework의 특수 사례이며, 원문의 프레임은 "새 대안"이 아니라 "기존 layer들을 포함하는 일반화"다.
<!-- TODO-VERIFY: 두 동치의 정확한 정리 번호와 전제(W_0=0, LN/residual 제거 등). 확인 방법: arXiv:2407.04620 Theorem 1/2 및 해당 증명 절 -->

systems 접점: state의 shape과 write의 GEMM 구조가 DeltaNet과 동일하므로, kernel 수준에서 TTT-Linear는 새로운 비용을 만들지 않는다. 새로운 것은 관점이다 — update를 "explicit한 학습 문제의 1 step"으로 명명하는 순간, inner objective를 갈아 끼우고($\to$ Miras), objective의 범위를 넓히고($\to$ Atlas), optimizer를 갈아 끼우는($\to$ Atlas의 Muon) 설계 공간이 열린다.

## 8.4 Dual form: per-token GD를 chunk 단위 GEMM으로

(M1)의 시스템적 결함은 순차성이다. $W_t$는 $W_{t-1}$을 필요로 하므로 token 방향으로 병렬화되지 않고, step 하나의 연산은 GEMV 한두 번과 rank-1 update — 독자의 언어로 arithmetic intensity가 낮아 tensor core가 노는 shape이다. inference의 decode라면 이 순차성은 어차피 자명하지만, **training에서는 치명적이다**: sequence 길이 $L$ 전체를 이 속도로 훑으면 wall-clock이 무너진다. 여기서 Sun et al. 2024의 두 번째 기여가 나온다. per-token 순차 계산을 **primal form**, 이를 chunk 단위로 tensorize한 등가(부분적으로는 근사) 계산을 **dual form**이라 부른다 [Sun et al. 2024 §2.5]. Titans는 이 재정식화를 자기 병렬화의 토대로 명시한다 [Titans §3.2].

아이디어는 mini-batch gradient descent다. 2장에서 굵게 강조했던 문장을 소환한다: **이 라인에서 훈련의 batch 축 역할을 하는 것은 sequence 축이다.** 연속한 $C$개 token을 inner 문제의 mini-batch로 묶고, chunk 안의 모든 gradient를 chunk 시작 상태 $W_{\xi(t,C)}$ ($\xi(t,C)=C\lfloor(t-1)/C\rfloor$)에서 평가한다 — 표준형 (M4) 그대로다. token $\tau$의 gradient가 직전 token들의 write를 반영하지 못하고 chunk 시작 시점의 낡은 snapshot을 보므로, 9장은 이를 stale-snapshot 근사(chunk-start anchor)라고 부르고 일반형으로 다룬다. 이 장에서는 TTT-Linear 인스턴스만 완결하자. (8-2)에 anchor를 적용하면 chunk 내부는 닫힌 형태가 된다:

$$
W_t \;=\; W_{\xi} \;-\; \sum_{\tau=\xi+1}^{t} \eta_\tau\,\big(W_{\xi}\,k_\tau - v_\tau\big)k_\tau^\top,
\qquad \xi := \xi(t,C)
\tag{8-3}
$$

read $y_t = W_t\,q_t$에 (8-3)을 대입하면 출력도 닫힌 형태다:

$$
y_t \;=\; W_{\xi}\,q_t \;+\; \sum_{\tau=\xi+1}^{t} \eta_\tau\,\big(v_\tau - W_{\xi}\,k_\tau\big)\,\big(k_\tau^\top q_t\big)
\tag{8-4}
$$

식 (8-4)를 소리 내어 읽으면 정체가 드러난다. 둘째 항은 **chunk 내부의 causal-masked attention**이다: score는 $k_\tau^\top q_t$, "value"는 residual $v_\tau - W_\xi k_\tau$, mask는 $\tau \le t$. 행렬로 구현하면 chunk의 $K,V,Q \in \mathbb{R}^{C\times d}$를 쌓아 (i) residual 행렬 $U := V^\top - W_\xi K^\top$ — shape $(d_v\times d_k)(d_k\times C)$의 GEMM, (ii) score 행렬 $KQ^\top$ — $(C\times d_k)(d_k\times C)$, (iii) causal mask 후 $U$와의 곱 — $(d_v\times C)(C\times C)$, (iv) chunk 끝 상태 갱신 $(d_v\times C)(C\times d_k)$의 GEMM 네 개로 끝난다. $C\times C$ score 행렬은 독자가 FlashAttention의 tile 내부에서 매일 보는 $S=QK^\top$와 같은 shape이다. dual form의 실체는 한 줄로: **chunk 내부는 attention처럼, chunk 경계는 RNN처럼.**

그러나 FlashAttention과의 유사성은 여기서 끝나고, 결정적 차이가 시작된다. FlashAttention의 tiling은 같은 수식의 bit-exact한 재배열이라 tile 크기는 성능에만 영향을 준다. dual form의 $C$는 **계산되는 함수 자체를 바꾼다**: $C=1$이면 순수 online GD(primal과 동일), $C=L$이면 사실상 1-step batch GD(8.3절의 linear-attention 극한), 그 사이의 모든 $C$는 서로 다른 layer다. 그래서 이 책은 $C$를 **semantic hyperparameter**라고 부른다 — 이 명제의 일반화와 명명은 9장이 맡고, 이 staleness가 train/serve 사이에서 일으키는 사고는 TNT의 주제다(→ 15장). 미리 정직하게 적어 두면, 이 stale 근사의 오차에 대한 형식적 bound는 여섯 논문 어디에도 없다(→ 9장, 15장).

원문은 quality(작은 $C$가 유리)와 throughput(큰 $C$가 유리) 사이의 실험적 절충으로 중간 크기의 chunk를 default로 채택했다.
<!-- TODO-VERIFY: Sun et al. 2024의 default inner mini-batch 크기(b=16으로 기억됨)와 그 sweep 실험 위치. 확인 방법: arXiv:2407.04620 §2.5와 ablation 절에서 "batch size b" 검색 -->

## 8.5 Outer loop: 이 layer 자체는 누가 훈련하는가

training 무경험 독자의 1번 질문에 정면으로 답한다: "test time에 학습한다는 이 layer는, 그 자체로는 누가 학습시키나?" 답은 outer loop — 보통의 pretraining이다. TTT layer는 4장의 bilevel 구조의 교과서적 인스턴스다. inner loop는 sequence 하나 안에서 매 token마다 $W_t$를 움직이고, outer loop는 corpus 전체에 대한 next-token prediction loss $\mathcal{L}$로 slow weights $\Theta$를 움직인다. 표 8-1이 분업의 전모다.

표 8-1 — TTT layer에서 inner와 outer의 분업

| 대상 | 소속 | 언제 움직이는가 | serving 관점의 위치 |
|---|---|---|---|
| $W_t$ (memory 상태) | inner (fast weights) | 매 token, 매 request | per-session state — KV cache가 있던 자리 |
| $W_K, W_V, W_Q$ | outer ($\Theta$) | pretraining에서만 | checkpoint, read-only |
| $\eta$-head ($\eta_t=\eta(x_t;\Theta)$의 산출기) | outer ($\Theta$) | pretraining에서만 (출력값 $\eta_t$는 매 token 계산) | checkpoint, read-only |
| $W_{\mathrm{init}}$ (초기 memory 상태) | outer ($\Theta$) | pretraining에서만 | checkpoint; sequence 시작 시 $W_0$로 복사 |
| backbone 나머지 | outer ($\Theta$) | pretraining에서만 | checkpoint, read-only |

outer 학습이 가능한 이유는 inner update 식 전체가 미분 가능한 연산의 합성이기 때문이다. $W_t$는 $\Theta$의 함수다 — $k_\tau, v_\tau, \eta_\tau$가 전부 $\Theta$로부터 나오므로, $\mathcal{L}$의 gradient는 unrolled inner loop를 **통과해** $W_K, W_V, W_Q, \eta$-head까지 흘러간다(4장의 unrolling). 4장의 skeleton-key 문장이 여기서 실체를 얻는다: **inner optimizer의 hyperparameter들이, outer loop가 학습하는 data-dependent 함수가 된다.** learning rate를 사람이 튜닝하지 않는다. gradient가 튜닝한다.

$W_{\mathrm{init}}$도 짚어 둘 가치가 있다. sequence가 시작될 때 memory는 0이 아니라 학습된 초기 상태 $W_0 = W_{\mathrm{init}}$에서 출발하며, 이것 역시 $\Theta$의 일부다 — MAML의 "initialization이 meta-variable"(→ 4장)의 직계 대응이고, TNT에서는 local memory reset의 착지점으로 load-bearing해진다(→ 15장).

serving 관점에서 이 표는 안심 포인트이기도 하다. 폐기되는 불변식은 정확히 한 칸이다: $W_t$만 움직이고, $\Theta$는 여전히 read-only checkpoint다. "weights가 움직인다"는 per-session fast-weight state의 이야기지, 서빙 중 checkpoint가 변한다는 이야기가 아니다 — 그 경계까지 허무는 것은 [Sleep]에 가서다(→ 17장). systems 접점 하나 더: unrolled inner loop는 정적 dataflow graph이므로 컴파일·fusion이 가능하고(4장 systems bridge), dual form은 그 graph를 GEMM 위주로 재배열한 것에 지나지 않는다.

## 8.6 스케일 증거와 파생

[Sun et al. 2024]는 TTT-Linear/TTT-MLP를 같은 규모의 Transformer 및 Mamba와 비교해, 문맥이 길어질수록 뒤쪽 token의 perplexity가 계속 내려가는 반면 Mamba는 일정 길이 이후 개선이 정체한다고 보고한다 — 고정 크기 vector/matrix state의 capacity 한계(→ 5장)와 정합적인 결과다.
<!-- TODO-VERIFY: 비교 스케일(125M–1.3B로 기억됨), 데이터셋(Pile/Books), 그리고 "Mamba는 16k 이후 정체" 주장의 정확한 그림 번호. 확인 방법: arXiv:2407.04620 §3 실험 절과 Figure 2 부근 -->

다만 이 결과의 규모 감각은 정직하게 유지해야 한다. 이 라인 전체(TTT부터 Sleep까지)의 실증 상한은 1.3B parameters / 100B tokens 수준이며, 독자가 운영하는 frontier-scale serving의 증거는 아직 없다. 이 장이 확립하는 것은 "동작한다"이지 "그 규모에서 이긴다"가 아니다.

파생 하나가 이 라인 바깥에서 TTT layer의 실용성을 보였다. Dalal et al. 2025 (arXiv:2504.05298)는 pre-trained Diffusion Transformer에 TTT-MLP layer를 삽입·finetune해 1분 길이의 video 생성을 시연했다 — TTT layer가 처음부터 함께 pretraining되지 않아도 기존 backbone에 graft될 수 있음을 보인 사례로, [Sleep]의 graft 전략(→ 17장)을 예고한다.
<!-- TODO-VERIFY: Dalal et al. 2025의 backbone 종류·규모(CogVideo-X 5B로 기억됨)와 "storyboard 조건부 Tom and Jerry 1분 생성" 셋업. 확인 방법: arXiv:2504.05298 abstract·§1 -->

systems 접점: "pre-trained backbone에 나중에 끼워 넣을 수 있는가"는 독자에게 배포 경로의 문제다. video 결과는 TTT layer가 아키텍처 전면 재훈련 없이 retrofit 가능한 부품임을 시사한다.

## 8.7 여섯 논문이 TTT에서 바꾸는 것 — Part II의 지도

Titans가 스스로 정리한 TTT와의 차이가 좋은 출발점이다. [Titans App. C]는 세 가지를 꼽는다: (1) TTT layer에는 forgetting mechanism이 없어 긴 sequence에서 고정 크기 memory가 넘친다, (2) update가 momentary surprise(현재 token의 gradient)에만 의존하고 token 흐름을 반영하는 momentum이 없다, (3) deep memory를 허용은 하되 그 득실을 실험적으로 검증하지 않았다. 이 세 결핍을 (M2)의 $\alpha_t, S_t$와 deep-memory 실증으로 메운 것이 Titans다. 같은 방식으로 여섯 편 전부를 "(M1)의 어느 성분을 바꿨는가"로 요약할 수 있다 — 표 8-2가 그 지도이고, 이 표가 사실상 Part II의 목차다.

표 8-2 — 여섯 논문이 TTT((M1))에서 바꾸는 성분

| 논문 (장) | 바꾸는 성분 | 무엇으로 |
|---|---|---|
| [Titans] (12장) | inner update rule | momentum $S_t$ + retention gate $\alpha_t$: (M1)→(M2); deep memory의 실증 |
| [Miras] (13장) | inner objective와 retention | $\ell$을 attentional bias family($\ell_p$, Huber, …)로, retention을 정규화항으로 일반화 |
| [Atlas] (14장) | objective의 범위와 inner optimizer | per-token → window 길이 $c$의 Omega rule((M3)); GD → Muon($\mathrm{NS}_\kappa$); feature map $\phi_p,\phi^*$ |
| [TNT] (15장) | 훈련 경제학 | hierarchical global/local memory, periodic state reset → context parallelism, train/serve의 $C$ 분리 |
| [NL] (16장) | loop의 층위 수 | 2-level → K-level(update frequency spectrum, CMS); optimizer 자체를 associative memory로 재해석 |
| [Sleep] (17장) | lifecycle | wake에서 fast weights에 쌓인 것을 sleep phase에 slow weights로 consolidation |

systems 독자를 위한 열쇠: 각 행은 비용 구조의 서로 다른 축을 건드린다. Titans/Miras/Atlas는 decode step당 FLOPs와 state 크기를(inner 연산이 무거워진다), TNT는 training MFU와 병렬화 토폴로지를, NL은 update가 일어나는 빈도의 계층 — 곧 memory 계층 배치를, Sleep은 serving fleet의 background job을 바꾼다. Part II의 각 장 §"systems 함의"가 이 축을 따라간다.

## 8.8 Worked micro-example: $d=2$ 손계산 — chunk가 함수를 바꾼다

TTT-Linear 하나를 $d_k=d_v=2$로 놓고, 같은 두 token을 $C=1$(primal, 순수 online)과 $C=2$(dual form, chunk-start anchor)로 처리해 출력이 달라짐을 확인하자. update는 식 (8-2), 게이트는 상수 $\eta_t=1$로 고정한다(상수 2는 이미 흡수됨). key의 norm이 1이므로 $\eta_t=1$은 "한 step에 완전 overwrite"를 뜻한다.

표 8-3 — 예제 입력

| $t$ | $k_t$ | $v_t$ | 비고 |
|---|---|---|---|
| 1 | $(1,0)^\top$ | $(2,0)^\top$ | |
| 2 | $(1,0)^\top$ | $(4,0)^\top$ | **같은 key**, 다른 value — overwrite 시험 |

초기 상태는 $W_0 = 0$ (2×2), 질의는 $q_2=(1,0)^\top$이다.

**Case A — $C=1$ (per-token online GD).** token 1: residual은 $W_0k_1 - v_1 = -v_1$이므로 $W_1 = v_1k_1^\top = \begin{pmatrix}2&0\\0&0\end{pmatrix}$. token 2: 이번에는 memory가 이미 안다 — $W_1k_2=(2,0)^\top$, residual은 $(2,0)^\top-(4,0)^\top=(-2,0)^\top$. update는 그 오차만큼만: $W_2 = W_1 + (2,0)^\top(1,0) = \begin{pmatrix}4&0\\0&0\end{pmatrix}$. 읽으면 $y_2 = W_2q_2 = (4,0)^\top = v_2$. delta rule이 옛 값 2를 새 값 4로 정확히 **overwrite**했다.

**Case B — $C=2$ (dual form, anchor $W_0$).** 두 token의 gradient가 모두 $W_0=0$에서 평가된다. token 1의 기여는 $v_1k_1^\top$, token 2의 기여는 residual $W_0k_2 - v_2 = -v_2$에서 나온 $v_2k_2^\top$ — token 1이 이미 key $(1,0)$에 2를 써 두었다는 사실을 **모른 채** value 전체를 쓴다. 따라서 $W_2 = v_1k_1^\top + v_2k_2^\top = \begin{pmatrix}6&0\\0&0\end{pmatrix}$이고, $y_2 = (6,0)^\top$. 식 (8-4)로 직접 계산해도 같다: $y_2 = W_0q_2 + (v_1)(k_1^\top q_2) + (v_2)(k_2^\top q_2) = 0 + (2,0)^\top\cdot 1 + (4,0)^\top\cdot 1 = (6,0)^\top$.

결과 정리: 같은 layer, 같은 입력, 같은 산술인데 $C=1$은 $4$를, $C=2$는 $6$을 돌려준다. 읽는 법이 세 겹이다. 첫째, staleness의 정체 — delta rule의 미덕은 residual을 쓰는 것인데, anchor가 낡으면 residual 계산이 틀려서 value를 통째로 쓴다. 둘째, $6 = 2+4$는 Hebbian superposition, 곧 5장의 crosstalk다: anchor가 0인 chunk 안에서 delta rule은 linear attention의 additive write로 퇴화한다 — 8.3절의 "batch GD 극한 = linear attention" 정리를 $2\times 2$에서 재현한 셈이다. 셋째, 이것이 **$C$가 semantic hyperparameter라는 명제의 최소 증명**이다: FlashAttention의 tile 크기를 바꿔서 출력이 4에서 6으로 변하는 일은 절대 없다. GEMM shape도 확인해 두자 — Case B의 연산은 $U=V^\top - W_0K^\top$ (2×2), score $KQ^\top$ (2×2, causal mask), correction $(2\times 2)(2\times 2)$로, 전부 (아주 작은) GEMM이다. $C$를 키우면 이 행렬들이 커지며 tensor core가 차오른다. 품질은? 방금 계산했듯, 그 대가로 지불한다.

## 8.9 Systems bridge: decode 안의 backward pass

이 장의 새 개념을 독자의 cost model 위에 정착시키며 마친다. TTT layer의 decode step 하나는 세 phase다: read(forward), write의 gradient 계산(forward + backward), write의 적용(update). TTT-Linear($d_k=d_v=d$)부터 세면 — read $y=W q$는 $d^2$ MAC; write는 $Wk$ ($d^2$), residual ($d$), rank-1 outer-product 적용 ($d^2$)으로 도합 약 $2d^2$ MAC. 즉 **write가 read와 같은 오더의 FLOPs다.** KV cache append(memcpy, FLOPs 0)에 익숙한 독자에게 이것이 첫 번째 구조 변화다. 두 번째는 traffic이다: 매 token마다 $W$ 전체($d^2$개 원소)를 read-modify-write하므로 per-token state traffic이 $2d^2$ floats — arithmetic intensity는 여전히 $O(1)$ MAC/byte 수준이고, decode는 bandwidth-bound로 남되 그 대상이 KV cache에서 fast-weight state로 바뀐다.

TTT-MLP(2-layer, hidden $4d$ 기준)는 backward가 명시적으로 등장한다. forward(read 또는 $\mathcal{M}(k_t)$) ≈ $8d^2$ MAC. write는 forward($k_t$) $8d^2$ + backward(VJP 연쇄, 2장의 rule of thumb대로 forward의 약 2배) $\approx 16d^2$ + parameter update(원소별 AXPY) $\approx 8d^2$ — 합계 write ≈ forward의 4배. 2장에서 훈련 비용 산식으로 배운 "backward ≈ 2× forward"가 이제 **decode latency 산식에 직접 들어온다.** 이것이 1장 Rosetta의 "폐기되는 불변식" 행의 정량적 의미다.

표 8-4 — per-token decode 비용의 자릿수 비교 (MAC 단위, 상수·norm류 생략; 정식 cheat sheet는 → 10장)

| layer | resident state | read FLOPs | write FLOPs | state traffic/token |
|---|---|---|---|---|
| softmax attention | $O(L\,d)$ (KV cache) | $O(L\,d)$ | ≈ 0 (append) | $O(L\,d)$ read |
| TTT-Linear | $d^2$ | $d^2$ | $\approx 2d^2$ | $2d^2$ RMW |
| TTT-MLP | $\approx 8d^2$ | $\approx 8d^2$ | $\approx 32d^2$ | $\approx 16d^2$ RMW |

serving engine에의 함의를 세 가지로 못 박는다. 첫째, **autograd는 없다.** inference stack에는 backward graph가 없으므로 gradient는 손으로 유도해 kernel에 하드코딩해야 한다 — TTT-Linear라면 식 (8-2) 자체가 kernel이고, TTT-MLP라면 2-layer VJP 연쇄를 read·update와 함께 하나의 fused kernel로 만든다(실제로 이 계열의 공개 구현들이 취하는 형태다). 둘째, **shared-weight batching이 깨진다.** $W_t$가 request마다 다르므로 batch 전체가 하나의 weight로 GEMM을 치는 전제가 무너지고, decode는 grouped-GEMM(request별 작은 GEMM 묶음) 형태가 된다 — 1장 Rosetta의 경고 행 그대로다. 셋째, **새로운 session-cache class가 생긴다.** per-session $W_t$는 sizing·checkpoint/restore·eviction을 요구하는 상태이며, KV cache처럼 prefix를 공유할 수도 없다(write가 비가역 압축이므로). 이 관리 문제는 Part III의 주제다. 마지막으로 prefill/decode 비대칭은 그대로 이식된다: prefill은 큰 $C$의 dual form(GEMM 잔치), decode는 $C=1$의 primal(GEMV + rank-1) — 단, 8.8절이 보였듯 이 둘은 **같은 함수가 아니다**. train과 serve의 $C$가 다를 때 무슨 일이 나는가가 [TNT]의 출발점이다(→ 15장).

## 요약

- test-time adaptation(Sun et al. 2020의 TTT, TENT)은 "inference 중 weight update"가 이 라인 이전부터 존재하던 검증된 기법임을 보여 준다; 2024년의 TTT layer는 그 갱신 단위를 image에서 token으로, 갱신 대상을 backbone에서 내장 memory로 좁힌 것이다.
- TTT layer의 정의는 식 (M1) 그대로다: state = 작은 model의 weights $W_t$, update = $\ell(W;k_t,v_t)=\|\mathcal{M}(k_t;W)-v_t\|_2^2$에 대한 1-step GD, read = $y_t=\mathcal{M}(q_t;W_t)$. momentum도 retention도 없다 — 이 결핍이 Part II 전체의 출발점이다.
- inner 문제의 모든 hyperparameter($W_K,W_V,W_Q$, $\eta_t$의 산출기, $W_{\mathrm{init}}$)는 outer loop가 next-token prediction으로 학습하는 slow weights $\Theta$다; serve 시점에 $\Theta$는 여전히 read-only이고 움직이는 것은 $W_t$뿐이다.
- TTT-Linear는 gate 없는 DeltaNet, 곧 delta rule의 sequence-layer 판본이다 [Miras Table 1]; batch-GD 극한에서 linear attention, non-parametric 극한에서 softmax attention을 특수 사례로 포함한다.
- dual form은 chunk-start anchor의 stale gradient 근사로 chunk 내부 계산을 GEMM 네 개로 재배열하며, 그 내부 구조는 causal-masked attention과 동형이다("chunk 내부는 attention처럼, chunk 경계는 RNN처럼").
- chunk 크기 $C$는 semantic hyperparameter다: FlashAttention tiling과 달리 $C$가 바뀌면 출력이 바뀐다 — $d=2$ 손계산에서 $C{=}1$은 4, $C{=}2$는 6을 반환했고, staleness의 오차 bound는 여섯 논문 어디에도 없다.
- decode에 backward pass가 들어온다: write 비용은 read의 2–4배 FLOPs이고 state 전체의 RMW traffic을 동반하며, serving engine에는 hand-derived gradient kernel, grouped-GEMM decode, per-session weight-state 관리가 새로 필요하다.

## 자가 점검 체크리스트

- [ ] TTT layer의 세 요소(state, update, read)를 통일 표기의 (M1) 형태로 적을 수 있다.
- [ ] 식 (8-2)를 (8-1)에서 직접 유도하고, DeltaNet·delta rule과의 동일성 및 Hebbian write와의 차이(residual write)를 설명할 수 있다.
- [ ] "이 layer의 learning rate는 누가 학습하는가"라는 질문에 inner/outer 분업표(표 8-1)로 답할 수 있다.
- [ ] $d=2$ 예제를 재현해 $C=1$과 $C=2$의 출력이 왜 다른지(stale residual → value write → crosstalk) 계산으로 보일 수 있다.
- [ ] dual form의 chunk 내부 항이 causal-masked attention과 같은 GEMM shape임을 (8-4)에서 읽어낼 수 있다.
- [ ] TTT layer의 decode를 inference 어휘로 옮길 수 있다: fast weights = codec 달린 writable KV cache, write = state 전체의 RMW + rank-1/rank-$C$ GEMM, dual form의 chunk = bit-exact tile이 아닌 semantic tile.

## 다음 장으로

이 장은 dual form을 TTT-Linear 한 경우에 대해서만 완결했다. 그러나 8.4절의 요령 — chunk 경계에서 무언가를 얼려 chunk 내부를 GEMM으로 만든다 — 은 TTT만의 것이 아니다. GLA는 decay를 접어 넣고, DeltaNet은 WY representation으로 rank-1 곱들을 두 개의 GEMM으로 바꾸며, Titans는 momentum을 associative scan으로 풀어낸다. 9장은 이 네 가지를 하나의 일반 scheme의 instantiation으로 통일하고, 이 장이 손계산으로만 보인 명제 — chunk 크기는 semantic hyperparameter다 — 를 정식으로 세운다. 그 명제가 하드웨어 경제학과 충돌하는 지점(품질 최적의 작은 $C$ vs MFU 최적의 큰 $C$)이 곧 TNT의 존재 이유다.
