# ch17. Do Language Models Need Sleep? — 오프라인 재귀

## 17.1 Bridge-in: 전작이 남긴 문제

ch16은 문맥을 파라미터 모양으로 접는 데 성공했다. Generative Adapter는 chunk 하나를 forward pass 한 번으로 흡수해 LoRA 모양의 산출물을 만들고, base 체크포인트는 바이트 단위로 그대로 둔다. 그런데 시계가 wake였다 — 문맥이 주어지면 그 자리에서 접고, 그 접은 결과로 같은 문맥에 답한다. 판별식 조건 (1)에서 탈락한 이유가 그것이다(→ ch11 §11.4.5). **접는 기제는 있었고 접는 시각이 없었다.**

이 장의 논문(*Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference*, arXiv:2605.26099, 이하 [Do LMs Need Sleep?])은 접는 시각을 만든다. 접기를 답 생성 경로에서 떼어 내 **KV cache를 비우는 순간**에 붙이고, 그 자리에서 같은 chunk를 여러 번 다시 통과시킨다. 그러면 $W$-경로에서 (U-W)가 wake의 토큰 시계 위가 아니라 별도의 시계 위에서 추가로 실행되는 구성이 처음으로 성립한다. ch11이 이 논문을 $W$-경로의 유일한 정면 사례로 분류한 근거다(→ ch11 표 11-2).

제목부터 정리하고 시작한다. 이 논문의 제목은 ch21이 다루는 `Language Models Need Sleep`(2606.03979)과 한 글자 차이로 겹치고, 저자들이 그 충돌을 인지해 자기 제목을 고쳤다고 각주에 적는다 [Do LMs Need Sleep? 각주 2]. 이 책은 이 논문을 [Do LMs Need Sleep?], ch21의 논문을 [LM Need Sleep]으로 부른다. 두 논문은 경로도 기제도 다르다 — 이쪽은 $W$, 저쪽은 $\Theta$다.

그런데 이 논문에는 **상속받은 open question이 없다.** "X가 남긴 문제를 우리가 받는다"에 해당하는 문장이 논문 어디에도 없고, 인용부호로 복원할 선행 질문도 없다. ch14가 그랬듯 이 장도 없는 계보를 지어내지 않는다(→ ch11 §11.7). 대신 이 논문이 실제로 물려받은 것은 질문이 아니라 **진단**이고, 논문의 첫 동작은 그 진단을 뒤집는 것이다.

물려받은 진단은 이렇다. attention-SSM 하이브리드가 긴 문맥에서 지는 이유는 고정 크기 상태의 **용량**이 모자라서다 — Jelassi et al.의 복사 과제 결과와 Arora et al.의 recall–throughput 트레이드오프가 그 자리를 잡았다. 이 논문은 같은 관찰을 받아 원인 귀속을 바꾼다.

> "This suggests that the bottleneck is not merely memory capacity as suggested by prior work [32, 2], but the amount of computation available for transforming evicted context into a useful internal state." [Do LMs Need Sleep? §1]

그리고 §2에서 같은 말을 명시적 반박의 형태로 다시 쓴다 — "Contrary to these works, we show that such models can fail as the required reasoning depth to solve a task increases, even when the amount of information to store is held fixed." [Do LMs Need Sleep? §2]

세 갈래의 암묵적 답이 더 붙는다. 첫째, Allen-Zhu의 $k$-hop 지식 회수 과제에서 SSM이 transformer에 크게 지는데 저장할 정보량은 충분하다는 관찰 — 이 논문은 거기에 기제(consolidation 계산 부족)와 처방(오프라인 재귀)을 공급하고 그 과제와 학습 프로토콜을 그대로 가져다 쓴다 [Do LMs Need Sleep? §6.2]. 둘째, chunk마다 gradient step을 **한 번** 도는 test-time training 계열에 대해 "한 번이면 충분한가"를 아니오로 답한다 [Do LMs Need Sleep? §2]. 셋째, depth-recurrence 계열이 예측 시점의 재귀가 도움이 된다는 것을 확립했는데, 이 논문은 그 계열이 묻지 않은 질문에 답한다 — 그 재귀를 예측 경로에서 **통째로 걷어낼 수 있는가**. "Unlike previous looped models, our model does not need to loop at prediction time." [Do LMs Need Sleep? §1]

> **[해설]** 그래서 이 장의 bridge-in은 상속 진술이 아니라 **상속된 오진의 교정**이다. 독자의 어휘로 옮기면 이렇다. KV cache가 버려질 때 남는 것이 고정 크기 상태라는 사실은 모두가 안다. 지금까지 그 상태의 문제는 **얼마나 담기는가**로 이야기되었다. 이 논문은 문제를 **담기 전에 얼마나 계산했는가**로 옮긴다. 담는 양이 아니라 담기까지의 계산량이 축이 되면, 그 축은 예산 $B_s$이고 그 예산은 wake 밖에 둘 수 있다.

## 17.2 문제의식

논문이 자기 문제를 세우는 문장은 하나다.

> "However, scalable memory is not the same as scalable reasoning. A fast weight memory may support long-range recall [48], but it is unclear whether it can support deep computation over tokens that are no longer present in the KV cache." [Do LMs Need Sleep? §1]

이 문장을 실험으로 옮기려면 두 가지를 동시에 고정해야 한다. 저장량을 고정한 채 요구되는 추론 깊이만 올려야 하고, 답을 만드는 쪽의 예산이 늘어나지 못하게 막아야 한다. 논문은 두 개의 제약으로 그것을 만든다.

첫째, **hard-eviction 제약**이다. $C$ 토큰마다 KV cache를 완전히 비운다 — "we impose a strict context window size as well as a hard-eviction constraint: we clear the context window every 24 tokens" [Do LMs Need Sleep? §4]. 경계를 넘어 살아남는 것은 SSM 블록의 fast weight뿐이다. 이 제약의 역할은 성능 향상이 아니라 **채널의 유일화**다. 경계 이후의 정확도는 전적으로 $W$에 무엇이 접혔는지의 함수가 된다.

둘째, **prediction-phase latency 제약**이다.

> "We impose a prediction-phase latency constraint: during the prediction phase, each answer token is predicted with a single standard forward pass. Extra loops or chain-of-thought tokens are disallowed because they increase prediction latency." [Do LMs Need Sleep? §4]

이 두 제약 아래에서 wake는 **토큰당 표준 forward pass 정확히 한 번**으로 못 박히고, chain-of-thought가 데이터에서 제거되어 있어 우회로가 없다. 비교군과 처리군의 wake 경로가 커널 수준에서 동일하다.

> **[평가]** 이 책이 보기에 이 두 제약이 논문의 기제보다 오래 남을 기여다. 지금까지 이 책이 다룬 논문 중 "wake 비용을 고정한다"를 조작적으로 정의한 것이 없다. Letta STC는 wake 토큰 수를 줄였다고 보고했지만 prefill 항을 세지 않았고(→ ch14 §14.5.3), $E$-경로 전반은 회수된 레코드가 프롬프트를 늘리는 항을 회계에 넣지 않는다(→ ch10). 논문 자신은 이 두 제약을 실험 통제로만 소개하고 기여로 내세우지 않는다.

논문의 핵심 주장은 이 무대 위에서 한 문장으로 진술된다 — "Our key insight is that recurrence can be used not only for prediction but also for memory consolidation." [Do LMs Need Sleep? §1] 재귀를 예측에 쓰면 test-time compute이고, consolidation에 쓰면 sleep-time compute이다. 같은 연산자의 시계만 바꾼 것이다.

여기에 논문이 스스로 붙이는 방법론적 단서가 하나 있다. 개선의 축이 학습 토큰 예산 고정 하의 정확도라는 것이다 — 논문은 과제 실패를 "고정된 학습 토큰 예산 안에서 풀리지 않음"으로 정의하고 그 위에서 곡선을 그린다 [Do LMs Need Sleep? §4]. 이 정의가 §17.6의 해석 전부를 지배한다.

## 17.3 Core mechanism (통일 표기)

### 17.3.1 표기 대응표

이 논문은 예약 기호와 **정면으로** 충돌한다. 충돌이 세 건이고 그중 두 건은 뜻이 정반대로 뒤집히는 종류라, 수식을 하나라도 보이기 전에 대응표를 먼저 놓는다.

표 17-1 — `Do LMs Need Sleep?` 원 표기 → 이 책의 표기

| 원 표기 | 이 책의 표기 | 심각도 | 주의 |
|---|---|---|---|
| $S_t$ — fast weight 상태 그 자체 [Eq. 3] | $W_t$ | 정면 충돌 | 예약된 $S_t$는 inner momentum 버퍼다. 이 논문에는 momentum이 아예 없으므로 예약된 뜻의 $S_t$는 이 장에 **한 번도 등장하지 않는다** |
| $\beta_t\in(0,1)$ — 쓰기 크기를 정하는 입력 게이트 [Eq. 3] | $\eta_t$ | 정면 충돌 | 예약된 $\beta_t$는 momentum decay다. 이 논문의 $\beta_t$는 토큰당 쓰기 보폭이므로 $\eta_t$의 역할이다 |
| $\alpha_t\in(0,1)$ — forget 게이트, $\alpha_t S_{t-1}$ [Eq. 3] | $\alpha_t$ (변환 불요) | 없음 | 방향이 이미 통일 규약과 같다(**남기는 비율**). Titans와 달리 부호 뒤집기가 필요 없다(→ NM ch12) |
| $L$ — context window 크기 = hard-eviction 주기 ($24$, $75$, $2000$, $512$) | $C$ (chunk) | 정면 충돌 | 예약된 맨 $L$은 sequence 길이다. 이 아키텍처에서 eviction 창이 곧 chunk다 |
| $T$ — 전체 sequence 길이 ($100$, $360$, $12{,}000$) | $L$ | 교환 | 위 행의 귀결로 원문의 $T$와 $L$이 **이름을 맞바꾼다**. 이 장에서 가장 위험한 대응이다 |
| $D$ — 블록 수 [Eq. 4–5] | $L_{\mathrm{layer}}$ | 충돌 | $D$는 Atlas의 lifted 차원에 예약되어 있다(→ NM ch14) |
| $t$ — Rule-110 rollout 단계 수 ($4,8,\dots,32$) [§4, §6.1] | $n_{\mathrm{roll}}$ (장-국소) | 정면 충돌 | 원문이 **같은 문서 안에서** $t$를 토큰 첨자와 rollout 깊이 양쪽으로 쓴다. 예약된 $t$는 토큰 첨자다 |
| $k$ — Depo hop 수 ($1,2,4,8,16$) [§3.2, §6.2] | $n_{\mathrm{hop}}$ (장-국소) | 정면 충돌 | 예약된 $k$는 sleep 라운드 첨자, $k_t$는 key 벡터다 |
| $h$ — Algorithm 1의 은닉 활성값 | $H^{(n,j)}\in\mathbb{R}^{C\times d}$ (장-국소) | 충돌 | 예약된 $h_t$는 outer 2차 moment다. 겸사겸사 pass 첨자 $j$를 명시하게 된다 |
| $m,\ m_c$ — loss mask [Alg. 1] | $\mu,\ \mu_n$ (장-국소) | 충돌 | 예약된 $m_t$는 outer 1차 moment다 |
| $c$ — 토큰 chunk [Alg. 1] | $c_n$ (첨자 필수, 장-국소) | 약한 충돌 | 예약된 맨 $c$는 Atlas Omega-rule의 window 길이다 |
| $o_t$ — layer 출력 [Eq. 2–3] | $y_t$ | 없음 | 통상의 통일 개명 |
| $\mathcal{L}$ — MaskedCE [Alg. 1] | $\mathcal{L}$ (그대로) | 없음 | 원문이 outer loss에 $\mathcal{L}$을 써서 통일 규약과 우연히 일치한다. TNT와는 반대다(→ NM ch15) |
| $N$ — sleep pass 수 | $N$ — **이 장이 신설하는 예약 기호** | 신설 | Titans의 $N_p$(persistent memory 토큰)와도, $N_q$(문맥을 공유하는 질의 수)와도 다르다 |
| $q_t,k_t,v_t$; $W_Q,W_K,W_V$; $K_t,V_t$ | 동일 | 없음 | 원문이 이미 열벡터로 선언한다 [§3.1]. 전치 변환이 필요 없는 드문 경우다 |
| $\mathcal{M}(\cdot;W)$ — 읽기 함수 | 해당 없음 | 부재 | 이 논문에는 memory 모듈 추상이 없다. 읽기가 $y_t=W_t q_t$로 직접 쓰인다 [Eq. 3]. 없는 $\mathcal{M}$을 소급해 씌우지 않는다 |

$T\leftrightarrow L$ 교환이 특히 위험하다. 아래 본문에서 $C=2000$이고 $L$이 2,000–3,300이라고 쓰면 원문에서는 $L=2000$, $T\in[2000,3300]$이다. 원문과 대조할 독자는 이 표를 손에 들고 읽어야 한다.

### 17.3.2 한 pass 안의 쓰기

논문이 설명용으로 내거는 갱신 규칙은 Mamba-2 형태의 게이트된 Hebbian 쓰기다.

$$
W_t \;=\; \alpha_t\, W_{t-1} \;+\; \eta_t\, v_t k_t^{\top},
\qquad
y_t \;=\; W_t\, q_t
\tag{17-1}
$$

식 (17-1)은 [Do LMs Need Sleep? Eq. 3]이며 개명 세 건($S_t\to W_t$, $\beta_t\to\eta_t$, $o_t\to y_t$)만 적용한 것이다. 표준형 (U-W)와 두 군데가 다르다. 첫째, momentum 항이 **없다** — $\beta_t S_{t-1}$에 해당하는 것이 아예 존재하지 않으므로 통일 표기로 옮기면 $S_t \equiv \eta_t v_t k_t^\top$, $\beta_t\equiv 0$이다. 둘째, gradient 자리에 raw outer product가 들어간다. $-\eta_t\nabla_W\ell(W_{t-1};k_t,v_t)$가 아니라 $+\eta_t v_t k_t^\top$이다. 게이트 $\alpha_t,\eta_t$는 둘 다 $x_t$에서 계산되는 데이터 의존 게이트다 [Do LMs Need Sleep? §3.1].

논문은 이 차이를 실수가 아니라 입장으로 방어한다.

> "our method uses a learned recurrent forward pass as the memory-update rule, allowing more flexible forms of consolidation that need not correspond to a one-step gradient descent on a fixed scalar objective." [Do LMs Need Sleep? §2]

**그런데 논문이 보여 주는 식은 논문이 돌린 식이 아니다.** 바로 다음 문장이 그렇게 말한다 — "In our experiments we use Gated Delta Networks (GDNs), which add a delta-rule correction to this update; however, the specific update rule does not matter for our discussion." [Do LMs Need Sleep? §3.1] 실제로 돌아간 규칙은 delta 보정이 붙은 형태이고 논문은 그것을 한 번도 전개하지 않는다. §6.3에서는 세 번째 규칙이 등장한다 — Jet layer는 "dynamic convolution instead of the fixed convolution in GDN"으로만 서술된다 [Do LMs Need Sleep? §6.2]. 세 규칙 중 어느 것이 일을 하는지 가르는 ablation은 없다.

### 17.3.3 이 장이 소유하는 두 정의

> **정의.** **KV cache를 비우기 전 offline recurrence로 fast weight에 접는 기제**란, eviction 경계에서 KV cache를 버리기 **직전**, 같은 chunk의 토큰에 대해 블록 스택을 $N$번 더 통과시켜 각 SSM 블록의 fast weight $W^{(\ell)}$만 갱신하고, 그 과정에서 정제된 은닉 특징 $H$와 재구축된 KV cache는 경계에서 폐기하여 **오직 $W$만 경계를 넘게 하는** 절차를 말한다. 세 요소가 모두 필요하다 — 경계 직전이라는 시각, 같은 토큰의 반복 흡수, 그리고 $W$ 이외 전부의 폐기. 하나라도 빠지면 다른 기제다. 특징을 함께 넘기면 depth-recurrent 예측 모델이고, 경계 이후에 돌리면 문맥 재적재이며, 한 번만 돌리면 통상의 SSM이다.

index를 명시해 쓰면 이렇다. chunk 첨자를 $n$, pass 첨자를 $j$로 둔다.

$$
\begin{aligned}
W^{(0,N)} &= 0,\\
H^{(n,0)} &= \mathrm{Embed}(c_n),
\qquad W^{(n,0)} = W^{(n-1,N)},\\
\big(H^{(n,j)},\,W^{(n,j)}\big) &= \mathrm{Blocks}\big(H^{(n,j-1)},\,W^{(n,j-1)};\,\Theta\big),
\qquad j=1,\dots,N,\\
&\text{그 뒤 } H^{(n,N)}\text{과 KV cache를 폐기. } W^{(n,N)}\text{만 경계를 넘는다.}
\end{aligned}
\tag{17-2}
$$

식 (17-2)는 [Do LMs Need Sleep? Alg. 1, §5]의 의사코드를 첨자를 드러내 옮긴 것이다. 원문은 `for n = 1,…,N` 안에서 `h, S ← Blocks(h, S)`로만 쓰고 첨자를 적지 않는다. 그 의사코드가 지고 있고 산문이 말하지 않는 사실이 셋이다.

1. **$W$는 pass 사이에 초기화되지 않는다.** pass $j$는 pass $j-1$이 끝낸 상태에서 시작하므로 같은 $C$개 토큰이 **누적으로** $N$번 접힌다.
2. **$H$는 chunk 안에서 pass를 건너 운반된다.** 그래서 pass마다 $(k_t,v_t)$가 달라진다. 그리고 경계에서 파괴된다.
3. 따라서 chunk를 넘는 유일한 채널이 $W$다 — "Unlike prior depth-recurrent models where gradient flows through recursively refined feature vectors, the gradient flows through the refined fast weights because we discard the refined features after sleep." [Do LMs Need Sleep? §5]

토큰 수준으로 내리면 $N$이 무엇을 하는지가 드러난다. $\xi(t,C)$를 chunk 시작 offset이라 하면,

$$
\begin{aligned}
k_t^{(j)} &= W_K\,\big[H^{(n,j-1)}\big]_{t,:},
\qquad
v_t^{(j)} = W_V\,\big[H^{(n,j-1)}\big]_{t,:},\\
W_t^{(n,j)} &= \alpha_t^{(j)}\,W_{t-1}^{(n,j)} \;+\; \eta_t^{(j)}\,v_t^{(j)}\big(k_t^{(j)}\big)^{\top},
\qquad
W_{\xi(t,C)}^{(n,j)} = W^{(n,j-1)}.
\end{aligned}
\tag{17-3}
$$

식 (17-3)이 이 장의 중심 문장을 담는다. pass $j$의 게이트 $\alpha_t^{(j)},\eta_t^{(j)}$는 $W^{(n,j-1)}$이 만들어 낸 특징의 함수다. 즉 상태 → 특징 → 게이트 → 상태의 닫힌 고리를 **입력 토큰을 얼려 둔 채** $N$번 도는 것이다.

> **정의.** **sleep 깊이 $N$을 늘린다**는 것은, 같은 chunk를 $W$에 $N$번 누적해 쓰되 매 pass의 $(k_t,v_t)$를 직전 pass의 $W$가 만든 특징에서 다시 뽑는다는 뜻이다. 그러므로 이것은 "(U-W) step을 더 밟는 것"이 아니라 **(U-W)와 특징 정제 고정점 반복의 합성**이다. 늘어나는 것은 저장량이 아니라 evicted 문맥 위에서 수행 가능한 **계산의 깊이**이며, 고정된 스칼라 목적함수에 대한 $N$번의 하강이 아니다.

이 정의가 왜 필요한지는 대조로 분명해진다. chunkwise-parallel 학습에서 한 chunk 안의 gradient는 전부 stale snapshot $W_{\xi(t,C)}$에서 평가된다(→ NM ch09). 여기서는 pass 내부의 재귀가 통상의 online SSM 재귀 그대로이고, snapshot 의미론이 **pass 수준으로 올라간다** — pass $j$는 chunk 내용이 이미 $j-1$번 쓰인 상태에서 출발한다.

마지막으로 두 가지를 못 박는다. $W_{\mathrm{init}}$은 meta-learn되지 않는다. 매 sequence에서 0으로 초기화된다 [Do LMs Need Sleep? Alg. 1]. Titans·TNT 계열이 지고 있던 학습된 초기 상태가 여기에는 없다(→ NM ch12, NM ch15). 그리고 sleep 중에 $\Theta$는 움직이지 않는다 — sleep은 forward pass다. "learned local rule"의 learned는 게이트 생산자와 투영 행렬이 **outer loop에서** 학습되었다는 뜻이지, test-time에 규칙이 학습된다는 뜻이 아니다.

## 17.4 어느 층을 언제 쓰는가

**이 논문이 test-time에 움직이는 층은 $W$ 하나다.** $\Theta$는 배포 전 outer loop에서만 움직이고, $E$는 존재하지 않는다 — 텍스트로도 벡터로도 그래프로도 아무것도 쓰지 않는다. 그러므로 층 사이의 순서 문제가 성립하지 않는다. corpus에서 갱신 대상이 단일한 몇 안 되는 논문이다.

표 17-2 — 이 값은 누가 학습하는가

| 객체 | 사는 곳 | outer loop 학습 | sleep에서 이동 | wake에서 이동 | 출처 |
|---|---|---|---|---|---|
| $\Theta$ — $W_Q,W_K,W_V$, MLP, normalization, 출력 투영, embedding | slow weights | 예. MaskedCE에 대해 Muon(+AdamW lr 5e-5)으로 전체 그래프 관통 | 아니오 — sleep은 forward pass다 | 아니오 | Alg. 1; §6 |
| $\alpha_t,\eta_t$의 게이트 생산자 ("learned local rule") | slow weights ($\subset\Theta$) | 예. 규칙이 "학습된다"는 말의 유일한 의미 | 아니오. 생산자는 고정이고 **출력** $\alpha_t^{(j)},\eta_t^{(j)}$만 입력 특징을 따라 pass마다 달라진다 | 아니오 | §3.1 |
| $W^{(\ell)}$ — SSM/GDN/Jet 블록당 fast weight 행렬 | fast weights | 저장량으로는 아니오. 학습 중 gradient가 **통과**할 뿐이며 초기값은 meta-learn이 아니라 0이다 | 예 — **sleep이 움직이는 유일한 객체.** chunk에 대한 $N$회 누적 sweep | 예 — 답 토큰을 처리하는 통상의 단일 pass 재귀도 $W$에 쓴다 | Alg. 1; §5 |
| $H^{(n,j)}$ — sleep 내부의 정제된 은닉 특징 | 활성값(일시적) | 아니오(파라미터가 아니다) | 예 — $N$개 pass를 건너 운반 | 경계에서 **파괴** | Alg. 1; §5 |
| KV cache $K_t,V_t$ | 비모수 상태 | 아니오 | $N$개 pass마다 재구축된 뒤 비워짐 | hard eviction에서는 예측 chunk 시작 시 비어 있음. sliding window에서는 최근 $C-1$ 토큰 유지 | §5; §6.4 |
| $N$ — sleep 깊이 | 하이퍼파라미터 | 아니오 — run마다 고정. $N$ 값마다 **별개로 학습된 모델**이다 | 아니오 — 적응적이지 않고 halting 기제가 없다 | 아니오 | §6.1; §6.3 |
| $C$ — eviction 창 | 하이퍼파라미터 | 아니오 — 과제마다 고정($24$ / $75$ / $2000$ / $512$) | 아니오 | 아니오 | §4; §6.2; §6.3; §6.4 |
| Ouro 1.4B에 삽입한 Jet layer 6개 | slow weights(신규 추가, 파라미터 10% 미만 증가) | 예. sliding-window 설정에서는 **2단계** — Jet layer만 1 epoch warm-up 후 전체 모델 2 epoch | 아니오 | 아니오 | §6.3; §6.4 |
| $E$ — external store | — | 해당 없음 | 해당 없음 | 해당 없음 | 이 논문에 external store가 없다 |

표에서 읽어야 할 행은 $W^{(\ell)}$ 행이다. **같은 상태가 두 시계에서 갱신된다.** sleep에서는 eviction 경계마다 $N$번 누적으로, wake에서는 답 토큰을 처리하며 토큰마다 한 번씩이다. 두 시계가 같은 갱신 규칙 (17-1)을 쓰고 같은 $\Theta$가 만든 게이트를 쓴다. 이 논문의 sleep이 별도의 학습 절차가 아니라 **같은 연산자의 반복 적용**인 이유가 여기 있다.

언제 자는지는 누가 정하는가. 학습 시점에는 **loss mask가 정한다.** Algorithm 1은 chunk의 mask $\mu_n$이 전부 0이면 $N$번 loop를 돌고, 그렇지 않으면 한 pass만 돌고 손실을 잰다 [Do LMs Need Sleep? Alg. 1]. 즉 sleep/wake 분할이 런타임 정책이 아니라 **라벨 배치**에서 나온다. 서빙에서 그 분할을 무엇이 정할지는 논문이 다루지 않는다.

### 17.4.1 판별식을 정면으로 통과한다는 것

> **정의.** $W$-경로가 **판별식을 정면으로 통과한다**는 것은, (U-W)를 wake의 토큰 시계 위에서가 아니라 **eviction 경계라는 별도의 시계 위에서 추가로** 실행하고($\Theta$·$E$는 건드리지 않는다), 그 추가 실행이 $B_s>0$을 소비하며, 그 결과인 $W$가 그 실행을 유발하지 않은 질의에 소비되는 구성을 말한다. $E$-경로는 모델 밖 레코드를, $\Theta$-경로는 배포 아티팩트를 바꾸는 데 반해, 여기서는 요청에 붙는 가중치 형태의 상태만 바뀐다.

네 조건을 이 논문에 하나씩 댄다(→ ch11 §11.2.1). 조건 (2)와 (3)은 이론의 여지가 없다 — $N$개의 추가 forward pass가 $B_s>0$이고, 바뀌는 것은 $W$다. 조건 (1)과 (4)는 **실험 설정마다 답이 다르다.**

Depo 설정에서는 둘 다 만족한다. 한 cycle이 최대 75개 노드·최대 300 토큰으로 서술된 뒤 그 뒤에 질의–답 쌍 10개가 최대 60 토큰에 걸쳐 붙고 전체 $L=360$이며, 창은 $C=75$이라 cycle 하나가 네 개 cache 창에 걸쳐 조각난다 [Do LMs Need Sleep? §6.2]. consolidation이 질의보다 앞서고, 만들어진 $W$를 **열 개의 질의가 공유한다.** 그리고 저자들이 그 설정의 성격을 명시한다 — "the model must form a query-agnostic representation because both $k$ and the start node are randomly sampled for each example" [Do LMs Need Sleep? §6.2].

헤드라인인 GSM-Infinite 설정에서는 다르다. 논문이 질문을 문맥 **앞**에 놓고 chain-of-thought를 데이터에서 제거한다.

> "We place the question before the context and exclude Chain-of-Thought traces from the data... This order gives the model the query before it reads the long problem context, allowing it to selectively consolidate information relevant to the question while ignoring filler tokens." [Do LMs Need Sleep? §6.3]

이 배치에서 consolidation은 질의 조건부다. 조건 (1)이 문자 그대로는 성립하지 않는다 — 상태를 바꾸는 연산이 질의 도착 **뒤**에 실행된다. 조건 (4)도 약해진다. 요청당 질의가 하나이므로 바뀐 상태를 읽는 질의가 그 상태를 만든 질의와 같다. 남는 것은 "답 토큰이 나오기 전에"라는 약화된 시계뿐이다.

> **[평가]** ch11이 이 논문을 통과로 분류한 것은 옳고, 그 통과는 **Depo 설정이 지고 있다.** 헤드라인 실험은 판별식의 조건 (1)과 (4)를 동시에 약화한 구성이며, 그 대가는 정확히 상각의 소멸이다 — 질의 조건부 consolidation에는 나눌 $N_q$가 없다. 이 책은 두 사실을 함께 적는다. 이 논문은 $W$-경로가 sleep-time compute이 될 수 있음을 보인 유일한 정면 사례이면서, 자기 최대 성과를 그 성질을 포기한 설정에서 얻었다.

## 17.5 비용 4종

**네 칸이 전부 비어 있다.** 하나도 수치로 보고되지 않는다. 대신 이 논문에는 다른 논문에 없는 것이 있다 — 세 칸이 왜 비었는지가 구조적으로 설명된다.

표 17-3 — 비용 4종

| 기호 | 값 | 논문이 말하는 것 | 출처 |
|---|---|---|---|
| $B_s$ | **논문에 없음**(수치). 구조로만 진술된다 | eviction 경계마다 loop 범위에 대한 $N$회 추가 forward pass. loop 범위는 from-scratch 모델과 Ouro는 전 블록, Jet-Nemotron은 28개 중 가운데 14개. 실행된 $N$은 $\{1,2,3,4\}$·$\{1,2,4\}$·$\{1,2,4,6\}$. "training cost grows roughly linearly with the number of recurrent steps $N$" | §6.3; §6.5 |
| $L_w$ | **논문에 없음**(측정). 아키텍처 보장으로만 존재 | 답 토큰마다 표준 forward pass 정확히 한 번, 추가 loop·CoT 금지. "this shifts extra computation to the sleep while preserving the latency of wake-time prediction" | Abstract; §4 |
| $C_{\text{cap}}$ | **바이트는 논문에 없음.** 고정 크기라는 사실과 shape 재료만 있다 | "Unlike the KV cache $K_t$ and $V_t$, the fast-weight $S_t$ does not grow in size with $t$." $d=256$(4층 하이브리드), $d=512$(10층 Depo), 28블록(Jet, 14개 loop), Jet layer 6개 삽입(Ouro) | §3.1; §6; §6.3 |
| $\rho$ | **논문에 없음** | 이름이 붙은 유일한 retention 기제는 토큰당 forget 게이트 $\alpha_t\in(0,1)$이고, 이는 sequence 안의 감쇠이지 측정된 열화율이 아니다 | §3.1 |

빈칸의 성격이 칸마다 다르므로 하나씩 적는다.

$B_s$는 FLOP도 경계당 토큰 수도 없다. 유일한 계산량 수치는 실험 전체의 훈련 비용이다 — "The automaton experiments require less than one A6000 GPU-day. The Depo and GSM-Infinite experiments require roughly 1–2 H100 GPU-days per run." [Do LMs Need Sleep? §6] 이 값은 세션당 sleep 예산이 아니라 run당 학습 예산이다. **추론 시점의 sleep 비용은 어떤 형태로도 보고되지 않는다.**

$C_{\text{cap}}$은 유도조차 불가능하다. 블록당 상태 차원($d_k\times d_v$), head 수, dtype이 전부 없다. 세션당 바이트를 쓰려면 GDN과 Jet의 기본값을 논문 밖에서 들여와야 하므로, 이 책은 그 칸을 채우지 않고 공백으로 기록한다.

$\rho$는 관측될 레짐에 논문이 **들어가지 않는다.** 실제로 지나가는 consolidation 경계 수가 automaton에서 4개($L=100$, $C=24$), Depo에서 4개, GSM-Infinite hard eviction에서 1–2개($L\in[2000,3300]$, $C=2000$), sliding window에서 대략 4–6개다 [Do LMs Need Sleep? §4, §6.2, §6.3, §6.4]. 게다가 $W$가 sequence마다 0으로 초기화되어 요청을 넘어 존속하지 않는다. 반복 consolidation의 열화를 잴 대상이 애초에 만들어지지 않는다.

$L_w$가 가장 중요한 칸이다.

> **[평가]** **헤드라인 지연 주장은 측정된 적이 없다.** "wake-time 예측의 지연을 보존하면서 추가 계산을 sleep으로 옮긴다"는 초록의 문장은 프로토콜 구성상 참이다 — 답 토큰마다 표준 forward pass 한 번으로 못 박아 두었으니 wake 경로가 baseline과 같을 수밖에 없다. 그러나 그것은 **정의에 의한 참**이지 측정이 아니다. ms/token도, TTFT도, wake 시점 tokens/s도 논문에 없다. 그리고 같은 침묵이 반대편에도 있다 — sleep 자체의 서빙 시점 벽시계 비용도 측정되지 않는다. 논문의 유일한 throughput 그림은 명시적으로 훈련 측정이다("Training throughput comparison on 1 NVIDIA H200 GPU" [Fig. 6 캡션]). systems 독자를 향해 지연 성질로 팔면서 systems 측정을 하나도 제시하지 않는다. 이 책은 표 17-3의 $L_w$ 칸에 "논문에 없음"이라고 쓰고, "효율적"이라는 서술을 본문에 쓰지 않는다(→ ch01의 회계 규약).

상각식 (A)는 설정마다 다르게 거동한다. Depo에서는 $N_q=10$이 구조적으로 정해져 있다 — 한 번 접은 $W$를 질의 열 개가 쓴다. corpus에서 $N_q$가 설계로 확정된 드문 자리이지만, 정작 이 논문은 상각을 논증하지 않고 $C_{\text{sleep}}$을 재지 않아 손익분기 $N_q^{*}$를 계산할 수 없다. GSM-Infinite 헤드라인에서는 $N_q=1$이므로 (A)가 $C_{\text{avg}}=C_{\text{wake}}+C_{\text{sleep}}$으로 붕괴한다. **상각이 아니라 재배치다.** wake에서 뺀 계산이 같은 요청 안의 다른 국면으로 옮겨 갈 뿐이고, 요청 전체의 계산량은 $N$에 대략 비례해 늘어난다.

## 17.6 실험과 스케일

### 17.6.1 프로토콜 — 읽기 전에 알아야 할 것 셋

첫째, **세 설정이 모두 합성이고 전부 저자가 만들었거나 재구성했다.** Rule-110 무대는 온전히 저자들의 구성이고, Depo는 Allen-Zhu의 과제이되 저자들이 다시 파라미터를 잡았으며, GSM-Infinite는 절차적 생성기라 저자들이 학습·평가 집합을 스스로 뽑았다 [Do LMs Need Sleep? §4, §6.2, §6.3]. 확립된 긴 문맥 벤치마크는 기각의 대상으로만 등장한다 — "Unlike retrieval-focused long-context tasks such as RULER [31]..." [Do LMs Need Sleep? §6.3]. 자연 텍스트 위의 결과가 없다.

둘째, **learning rate 프로토콜이 양방향으로 비대칭인데 한 방향만 진술된다.** "We tune the Muon learning rate on the $N=1$ model, giving the no-loop baseline an advantage, and use the selected value, 2e-3, for all looped models." [Do LMs Need Sleep? §6] baseline에 이점을 준다는 서술은 정당하지만, 같은 문장이 **처리군은 한 번도 튜닝되지 않았다**는 뜻이기도 하다. AdamW lr은 탐색 없이 5e-5로 고정했고, §6.3의 Muon lr 1e-3은 선행 연구를 따랐을 뿐 튜닝하지 않았다 [Do LMs Need Sleep? §6]. loop 모델은 유효 깊이가 다르므로 $N=1$에서 고른 lr을 이식하는 것은 방향을 알 수 없는 교란이다.

셋째, **시드가 하나다.** "For fair comparison, we fix random seeds ensuring that all runs use exactly the same data ordering." [Do LMs Need Sleep? §6] Fig. 2–6 어디에도 신뢰구간도 음영대도 반복 실행도 없다. batch 크기는 512(automaton)·128(Depo)·256(GSM-Infinite)이다.

그리고 이 절 전체를 지배하는 사실이 하나 더 있다. **계산량을 맞춘 baseline이 없다.** loop는 비용을 $N$에 대략 비례해 늘리는데 [Do LMs Need Sleep? §6.5], Fig. 2b·3·4·5의 모든 비교는 학습 **토큰** 또는 **step**을 맞춘 것이지 FLOP을 맞춘 것이 아니다. 같은 x좌표에서 $N=4$ 실행은 $N=1$ 실행의 대략 4배 계산을 이미 썼다. "재귀가 통상의 용량이나 통상의 계산으로는 살 수 없는 것을 산다"는 중심 주장이 **더 넓거나 더 깊거나 더 오래 학습한 비-loop 모델**과 겨뤄진 적이 없다.

### 17.6.2 Rule-110 — 계산이 병목이라는 논증

무대는 이렇다. 길이 24의 독립 이진 문자열 넷, 문자 단위 토크나이저, 상태 토큰 96개 + 라벨 토큰 4개로 $L=100$, hard eviction 주기 $C=24$. 라벨 $i$는 상태 $i$를 Rule-110으로 $n_{\mathrm{roll}}$번 전이시킨 뒤의 첫 비트다 [Do LMs Need Sleep? §4]. 모델은 4층 GDN-attention 하이브리드, $d=256$, 배치는 attention → GDN → attention → GDN이다.

Fig. 2a가 논문의 진단을 보인다. $n_{\mathrm{roll}}$을 4에서 16까지 올리면 **sequence 길이를 고정한 채** 비-loop 하이브리드의 정확도가 무너진다. 저장할 정보량이 그대로인데 요구되는 계산 깊이만 올라간 것이므로, 실패의 원인이 용량이 아니라는 논증이 성립한다. 곡선의 수치 끝점은 인쇄되어 있지 않다.

Fig. 2b가 처방이다. $n_{\mathrm{roll}}=32$에서 $N=1$은 "remains close to random guessing, reaching only about 10% exact accuracy after nearly 5B training tokens", $N=2$는 "approximately 20% accuracy", $N=3$과 $N=4$는 "above 30%"다 [Do LMs Need Sleep? §6.1]. 저자들의 귀속은 정확하다 — "Because the context length, eviction rule, and prediction-phase computation are fixed across these runs, the improvement comes from additional consolidation-time computation during sleep." [Do LMs Need Sleep? §6.1]

두 가지를 함께 적어야 한다. 첫째, **과제는 여전히 풀리지 않았다.** 거의 5B 학습 토큰 뒤의 최고치가 30%대이고 baseline은 무작위 추측 근처다. sleep이 돕는다는 논증은 성립하고 sleep이 과제를 다룰 만하게 만든다는 논증은 성립하지 않는다. 둘째, 초록의 "regular transformer as well as SSM-attention hybrid models fail"은 결과가 아니라 **동어반복**이다. 논문 자신이 이유를 적는다 — "Under this hard eviction constraint, a standard transformer cannot do better than random guessing as the KV cache has been destroyed before prediction is made." [Do LMs Need Sleep? §4] 경계를 넘어 살아남는 상태가 정의상 없는 것이다. 캐시를 유지하도록 허용한 full-attention baseline은 **어느 과제, 어느 스케일에서도 실행되지 않았다.**

### 17.6.3 Depo — 수치가 없는 절

Depo는 Allen-Zhu의 $n_{\mathrm{hop}}$-hop 지식 회수 과제다. cycle 하나가 최대 75노드·최대 300토큰이고 양쪽 시점에서 300으로 left-padding되며, 질의–답 쌍 10개가 최대 60토큰, $L=360$, $C=75$다. hop 수는 학습 중 $[1,16]$에서 균등 추출하고 시험은 $n_{\mathrm{hop}}\in\{1,2,4,8,16\}$에서 잰다 [Do LMs Need Sleep? §6.2].

**이 절은 정량치를 하나도 보고하지 않는다.** 정확도도, 최종 손실값도, 표도 없이 0–100k step의 시험 손실 곡선만 있다. 확보되는 결과는 계단 모양의 정성 진술이다 — "increasing the number of offline loops improves learning speed for queries that require 4 or more hops. The 1-loop model makes little progress on 4-hop and harder queries, and the 2-loop model similarly stalls on 8-hop and harder queries. Within our training budget, only the 4-loop model begins to improve on the hardest 16-hop task." [Do LMs Need Sleep? §6.2] 마지막 문장이 말하는 것은 16-hop이 풀렸다가 아니라 **움직이기 시작했다**이다.

그리고 이 절에는 논문 내부 모순이 있다. §6의 실험 세부는 §6.2용으로 "10-layer 모델을 $d=512$로 처음부터 학습했다"고 적는데, Fig. 3의 캡션은 같은 결과를 "4-layer GDN-attention hybrid"의 것이라고 적는다 [Do LMs Need Sleep? §6, Fig. 3 캡션]. 어느 모델이 Fig. 3을 냈는지 독자가 판정할 수 없고, 확인할 부록이 논문에 없다.

### 17.6.4 GSM-Infinite — 실모델·실수치

여기가 유일하게 사전학습 모델·실제 수치·포화/이득 패턴이 함께 나오는 자리다. 문제당 2,000–3,300 토큰, 연산 수는 $[1,8]$에서 균등 추출, 평가는 저자들이 뽑은 held-out 1,600 예제다 [Do LMs Need Sleep? §6.3].

표 17-4 — GSM-Infinite 정확도 (원문 정밀도 그대로)

| 설정 | 모델 · loop 범위 | 연산 수 | $N=1$ | 최대 $N$ | 논문의 상대 개선 라벨 |
|---|---|---|---|---|---|
| hard eviction $C=2000$ | Jet-Nemotron 2B · 28블록 중 14 | 6 | 0.742 | 0.812 ($N{=}6$) | 9% |
| 〃 | 〃 | 8 | 0.351 | 0.388 ($N{=}6$) | 11% |
| 〃 | Ouro 1.4B · 전 블록 | 6 | 0.419 | 0.615 ($N{=}4$) | 47% |
| 〃 | 〃 | 8 | 0.210 | 0.272 ($N{=}4$) | 30% |
| sliding window $C=512$ | Ouro 1.4B · 전 블록 | 2 | 0.596 | 0.905 ($N{=}4$) | 52% |
| 〃 | 〃 | 4 | 0.839 | 0.926 ($N{=}4$) | 10% |
| 〃 | 〃 | 6 | 0.251 | 0.320 ($N{=}4$) | 27% |
| 〃 | 〃 | 8 | 0.116 | 0.137 ($N{=}4$) | 18% |

*라벨은 전부 **상대** 개선이다. 절대 점수 차는 각각 7.0·3.7·19.6·6.2·30.9·8.7·6.9·2.1 백분점이다. 출처: [Do LMs Need Sleep? §6.3, §6.4, Fig. 4, Fig. 5].*

모델 두 개의 성격이 다르다. Jet-Nemotron 2B는 Qwen 2.5 1.5B에서 일부 attention layer를 Jet layer로 바꿔 만든 하이브리드이고, Ouro 1.4B는 원래 attention 전용 looped 모델이라 fast weight 기억을 주기 위해 MLP 없는 Jet layer 6개를 삽입했다(파라미터 10% 미만 증가) [Do LMs Need Sleep? §6.3]. Ouro의 이득이 큰 것에 대해 저자들은 "The gap is wider for Ouro, which may reflect its depth-recurrent pretraining"이라고만 적는다.

표에 넣지 않은 네 줄이 더 중요하다.

**쉬운 문제에서 $N$은 아무것도 사지 않는다.** Jet의 2연산 네 값은 0.985 / 0.985 / 0.980 / 0.979, 4연산은 0.995 / 0.994 / 0.993 / 0.992로 $N\in\{1,2,4,6\}$ 전체의 폭이 각각 0.6과 0.3 백분점이다. Ouro의 2연산은 0.868 / 0.863 / 0.857(폭 1.1), 4연산은 0.932 / 0.923 / 0.903(폭 2.9)이다. **어느 끝이 어느 $N$인지 논문은 배정하지 않는다.** 배정과 무관하게 확정되는 것은 폭 자체이고, 6·8연산의 7.0·19.6 백분점과 나란히 놓으면 이득이 어려운 꼬리에만 있다는 사실이 남는다. 저자들의 서술은 포화만 인정한다 — "For easier two- and four-operation problems, accuracy often approaches saturation regardless of the number of loops, especially for Jet, which has more fast weight memory capacity than Ouro." [Do LMs Need Sleep? §6.3]

**$N$에 대한 비단조가 있다.** Ouro 8연산 패널에 인쇄된 세 값은 0.272 / 0.210 / 0.209이고, 산문이 no-loop를 0.210, 4-loop를 0.272로 지목하므로 [Do LMs Need Sleep? §6.3] 남는 0.209가 2-loop다. 즉 가장 어려운 연산 수에서 $N=2$가 $N=1$보다 근소하게 **낮다**. 이 배정은 논문이 명시하지 않고, 반전 자체를 논의하지도 않는다. 초록의 주장은 "sleep 깊이를 늘리면 성능이 오른다"이다.

**가장 큰 헤드라인은 결손의 회복이다.** sliding-window 2연산의 52%는 0.596에서 0.905로 가는 값인데, **같은 모델**이 hard eviction $C=2000$에서 같은 2연산을 loop 없이 0.857–0.868로 푼다(표 17-4 위쪽과 §17.6.4 앞 문단). 논문 스스로 "this baseline performs poorly even on two-operation problems"라고 적는다 [Do LMs Need Sleep? §6.4]. 새 능력이 아니라 설정이 만든 구멍을 메운 값이다.

**배포에 가장 가까운 설정이 레시피 없이는 학습되지 않는다.** sliding window에서는 sleep 뒤 최근 $C-1$ 토큰을 남기고 그보다 오래된 것만 축출하므로 $N=1$이면 통상의 SWA-SSM 하이브리드가 된다. 그런데 학습에 2단계가 필요하다 — Jet layer만 1 epoch warm-up한 뒤 전체 모델을 2 epoch 학습하며, "We find that for $N>1$, using hard eviction for the warm-up stage is crucial for the model to learn to refine the fast weights." [Do LMs Need Sleep? §6.4] 왜 그런지는 설명되지 않고 레시피로만 보고된다.

### 17.6.5 throughput과 스케일 상한

논문의 유일한 systems 측정은 Fig. 6이고 **훈련 측정이다.** Ouro 1.4B, H200 한 장, $L=12{,}000$, FlashAttention 2, 창 크기 1K·2K·4K. 결과는 두 문장이다 — 창이 충분히 크면 문맥 창을 가로지르는 직렬성이 완전 병렬 baseline 대비 throughput을 의미 있게 바꾸지 않는다, 그리고 "Throughput is roughly inversely proportional to $N$" [Do LMs Need Sleep? Fig. 6 캡션]. Fig. 6b는 OOM을 피하려고 문맥 chunk 축에 대한 activation checkpointing을 켜야 했다. 곡선 값은 인쇄되어 있지 않아 인용할 수 있는 숫자가 없다.

실증 상한을 정직하게 적으면 이렇다. 가장 큰 모델이 2B와 1.4B, 가장 긴 과제 sequence가 3,300 토큰(12,000 토큰은 아무것도 학습하지 않고 어떤 과제도 평가하지 않는 throughput 프로브에서만 등장한다), 최대 sleep 깊이가 6이고 그것도 블록 절반만 도는 Jet에서다. run당 계산량은 1–2 H100 GPU-day다. 그리고 Ouro의 $N$은 성능이 아니라 **학습 메모리** 때문에 4에서 잘렸다 — "To keep memory cost during training manageable while using a reasonable batch size, we use $N=\{1,2,4\}$ for Ouro." [Do LMs Need Sleep? §6.3] 파라미터에서도 문맥에서도 배포 규모보다 서너 자릿수 아래이며, 이미 1.4B·3.3k 토큰에서 메모리 제약에 걸려 있다.

## 17.7 Systems/serving 함의

독자의 1번 질문에 대한 이 논문의 답은 **decode에서는 아무것도 바뀌지 않는다**이다. 그리고 그 답이 참인 방식이 곧 이 논문의 systems 공백이기도 하다.

**wake 경로가 baseline과 커널 단위로 같다.** 답 토큰마다 표준 forward pass 한 번, 추가 loop 없음, chain-of-thought 토큰 없음이다 [Do LMs Need Sleep? §1, §4]. 따라서 토큰당 decode FLOPs도, decode가 끌어오는 가중치 트래픽도, 산술 강도도 $N=1$ 실행과 같다. 달라지는 것은 $W$의 **내용**뿐이다. $E$-경로가 회수 레코드로 prefill을 늘리고(→ ch14 §14.7) $\Theta$-경로가 배포 아티팩트를 바꾸는 것과 대비된다(→ ch19).

**대신 eviction 경계마다 chunked prefill의 버스트가 붙는다.** 그 버스트는 안에서 성질이 아주 다른 두 부분으로 갈라진다.

첫째, SSM 블록의 fast weight다. 고정 크기이고 $t$에 따라 자라지 않으므로 [Do LMs Need Sleep? §3.1], sleep 동안 이 상태는 $C$번이 아니라 $N\times C$번 read-modify-write된다. 작고 상주하는 상태에 대한 **대역폭 가벼운·계산 무거운** 버스트이며, 독자가 아는 KV cache 바운드 decode 레짐의 정반대 모양이다.

둘째, 같은 loop 안의 attention 블록이다. pass마다 특징이 달라지므로 KV cache를 매 pass 다시 쌓아야 하고, 그러면 chunk에 대한 $O(C^2)$ attention 일이 $N$번 반복된다. $C=2000$에서 이것은 작은 재계산이 아니다.

> **[해설]** 논문이 준 값으로 이 책이 산술만 해 두면 이렇다. Jet-Nemotron 설정은 28개 블록 중 가운데 14개만 loop하고 $N=6$까지 돈다 [Do LMs Need Sleep? §6.3]. 그러면 한 chunk의 consolidation 비용은 그 chunk를 평범하게 prefill하는 비용의 $1+(N-1)\cdot\frac{14}{28}=3.5$배다. 요청당 경계 수는 hard eviction GSM-Infinite에서 1–2개, sliding window에서 4–6개다(§17.5). 이 곱셈을 논문은 하지 않는다. 그리고 이 값은 이 책의 산술이므로 표 17-3의 $B_s$ 칸은 여전히 "논문에 없음"으로 남는다.

**서빙에서 배치가 깨지는 자리가 두 겹이다.** 첫째는 낯익다 — 요청마다 다른 $W$를 들면 shared-weight batching이 성립하지 않는다(→ ch01의 Rosetta 표). 이 논문이 그 대가를 새로 만들지는 않는다. 둘째가 새롭다. **eviction 경계는 요청별 토큰 수에서 발생하므로 배치 안에서 정렬되지 않는다.** 한 요청이 자고 나머지가 decode하는 배치에는 커널 모양이 둘 살아 있다 — $N$-pass chunked prefill과 batch-1에 가까운 decode step이다. 스케줄러는 자는 요청의 이웃을 세우든지, consolidation을 선점 가능한 별도 잡으로 빼든지 해야 한다. 논문은 경계 skew를 다루지 않으며, batching에 관한 유일한 언급은 GPU 활용률을 위해 설정마다 batch 크기를 조정했다는 것이다 [Do LMs Need Sleep? Fig. 6 캡션].

**상태가 요청보다 오래 살지 않는다.** $W$는 sequence마다 0으로 초기화되고 [Do LMs Need Sleep? Alg. 1] 요청이 끝나면 사라진다. 그래서 사용자당·세션당 지속 저장소가 **생기지 않는다.** checkpoint/restore도, 세션 캐시 축출 정책도, 요청 간 상각도 없다. 이것이 "요청 내부 consolidation"이라는 말의 서빙 쪽 의미다.

> **[평가]** 이 한 줄이 $W$-경로와 $E$-경로의 구조적 차이를 가장 짧게 드러낸다. $E$-경로의 sleep 산출물은 요청보다 오래 살고, 그래서 식 (A)의 $N_q$가 1보다 커질 수 있다. 이 논문의 sleep 산출물은 요청과 함께 죽는다. 같은 이름 아래 있지만 회계의 종류가 다르다 — 하나는 상각이고 하나는 요청 내부의 재배치다(→ ch24, ch25).

**독자가 원하는 비교 팔이 데이터에서 제거되어 있다.** 서빙 관점에서 이 방법의 자연스러운 경쟁자는 wake에서 추론 깊이를 사는 표준 수단, 즉 chain-of-thought 토큰이다. "경계에서 $N$번의 sleep pass"와 "decode에서 $\kappa$개의 CoT 토큰" 중 어느 쪽이 같은 품질을 싸게 사는가가 배치·SLA 결정을 가르는 질문인데, 논문은 CoT를 데이터에서 빼는 것으로 그 팔을 없앴다 [Do LMs Need Sleep? §6.3]. 두 번째 팔은 어느 과제에서도 실행되지 않는다.

**memory-centric 논증은 여기서 멈춘다.** 작고 상주하는 상태에 대한 $N\times C$회 read-modify-write는 그 논증의 근거가 실제로 있는 드문 패턴이다. 그러나 논문에 바이트가 없으므로(§17.5) 이 장은 패턴만 기록하고 정량화는 이 책의 실험이 소유하는 자리로 넘긴다(→ ch26, ch29).

## 17.8 한계와 bridge-out

### 17.8.1 논문 자신이 남긴 문제

논문의 §7이 세 가지를 스스로 적는다.

1. **학습 비용과 불안정.** "this gain is not free: during training, we need to perform $N$ deeper forward and backward passes, which can make training slow and unstable." [Do LMs Need Sleep? §7] 처방으로 implicit gradient, truncated BPTT, 안정화 기법을 이름으로 열거하지만 **하나도 시도하지 않는다.** 불안정이 얼마나 자주 발생하는지에 대한 증거도 없고, 모든 곡선이 단일 시드라 불안정은 그림에 나타나지 않는다(§17.6.1).
2. **sequence 축 병렬성의 상실.** 학습이 문맥 창을 가로질러 재귀적이므로 sequence 축을 완전히 병렬화할 수 없다. 논문은 창이 충분히 크면 wall-clock을 해치지 않는다고 하지만, 근거는 H200 한 장·$L=12{,}000$·$C\in\{1\text{K},2\text{K},4\text{K}\}$뿐이다 [Do LMs Need Sleep? §6.5]. 작은 $C$나 다중 GPU에서의 거동은 열려 있다.
3. **이득이 순차적이지 않은 과제에서도 남는가.** "Sleep makes training sequential across context and depth dimension, but this sequentiality is also why our method shows gains on the tasks we consider, whose solutions are themselves sequential." [Do LMs Need Sleep? §7] 저자들이 스스로 적는 범위 제한이다 — **세 과제는 이 방법이 통하는 성질을 가졌기 때문에 골라졌다.**

논문이 질문으로 세우지 않았으나 이 책이 미해결로 기록하는 것이 넷 더 있다.

**$N$을 어떻게 고르는가.** $N$은 run마다 고정된 하이퍼파라미터이고 값마다 별개로 학습된 모델이다(표 17-2). 적응 깊이 연구를 관련연구에서 인용하면서도 halting 기준도, 예제별 배분도 없다 [Do LMs Need Sleep? §2]. §17.6.4가 보인 대로 이득은 어려운 꼬리에만 있는데 선택적으로 지출할 기제가 없으므로, 서빙에서는 모든 요청이 같은 $N$배의 consolidation을 낸다.

**어디서 포화하는가.** 최대 시험값이 $N=6$이고 그것도 블록 절반만 도는 Jet에서다. Ouro의 $N$은 학습 메모리 때문에 4에서 잘렸다 [Do LMs Need Sleep? §6.3]. 포화점은 측정되지 않았다.

**어느 국소 규칙이 일을 하는가.** 논문이 명시적으로 범위 밖으로 선언한다 — "the specific update rule does not matter for our discussion" [Do LMs Need Sleep? §3.1]. 초록이 "a learned local rule"을 기여 성분으로 내거는데, GDN·Mamba-2·Jet·delta 보정을 가르는 ablation은 없다(§17.3.2).

**인용한 대안과의 경쟁·합성이 전부 미시험이다.** context distillation, Cartridges, in-context autoencoding, 실제 gradient step을 밟는 test-time training, chunk당 LoRA adapter, 그리고 $\Theta$-경로의 sleep — 여섯 묶음이 §2에서 서술되고 **어느 것도 baseline으로 돌지 않으며 어느 것과도 결합되지 않는다.** 논문의 전체 비교 집합은 같은 아키텍처의 $N=1$ 판본이다. 그래서 §2가 말로 반박한 "chunk당 gradient 한 번" 계열보다 이 방법이 낫다는 증거가 논문에 없다.

### 17.8.2 경로 간 인용 여부

참고문헌 65개를 전수 확인하면 이 논문의 인용 지도는 corpus의 다른 논문들과 모양이 다르다.

**$E$-경로: 교전한다.** Lin et al. = Letta의 `Sleep-time Compute`(2504.13171, → ch14)를 **두 번** 인용한다 — 오프라인 계산으로 예상 질문을 미리 푸는 방법으로 한 번, "offline planning phase를 sleep이라 부른다"로 한 번 [Do LMs Need Sleep? §2]. Cartridges는 경로 구분을 가장 또렷하게 쓰는 자리에서 인용된다 — "These methods shorten what remains in the attention context, whereas our method transfers evicted context into weight-based memory." [Do LMs Need Sleep? §2] 그러나 MemGPT(2310.08560)·Mem0(2504.19413)·Zep(2501.13956)·ReasoningBank(2509.25140)은 **참고문헌에 없다**(→ ch13, ch15). 프로덕션 외부기억 문헌이 통째로 빠져 있다.

**$\Theta$-경로: 교전한다.** Behrouz et al.의 `Language models need sleep`(→ ch21)이 인용되어 있는데, 그 항목에는 arXiv ID도 venue도 연도도 없이 제목만 있다 — OpenReview 투고본에서 인용했다는 뜻이다 [Do LMs Need Sleep? §2, References]. Generative Adapter(→ ch16)도 context distillation 목록 안에 들어 있다. 반면 SEAL(2506.10943)·Nested Learning(2512.24695)·SCM(2604.20943)·LoRA(2106.09685)·ROME(2202.05262)·MEMIT(2210.07229)·Memory Layers(2412.09764)는 전부 없다(→ ch18, ch19, ch20).

**자기 경로: 침묵한다.** fast weight programming 계보(Schlag 등, Hebb)와 현대 gated/delta 선형 attention 계열(Dao·Gu, Samba, Griffin, Hymba)은 두텁게 인용한다. 그런데 Titans(2501.00663)·Miras(2504.13173)·Atlas(2505.23735)·TNT(2511.07343)·Nested Learning(2512.24695)은 65개 항목 중 **하나도 없다**(→ NM ch12, NM ch14, NM ch15).

침묵은 두 종류로 갈라야 한다. 연대가 강제한 침묵은 계보 사실이고, 선택된 침묵이 발견이다. 이 논문의 v3 스탬프는 2026년 6월 5일이고 위에 열거한 부재 항목은 **전부 그보다 앞선다.** 인용 가능한 상태로 전부 존재했다. 연대가 실제로 개입한 자리는 반대 방향에 하나 있다 — 제목이 겹치는 Behrouz et al.은 당시 arXiv에 오르기 전 OpenReview 투고본이었고, 그럼에도 인용되어 있다.

> **[평가]** 그러므로 이 논문에 대해 "경로가 서로를 인용하지 않는다"고 쓰면 틀린다. 이 논문은 $E$-경로에 이름을 준 논문도, $\Theta$-경로의 제목 쌍둥이도 인용하고 산문으로 자기와 구별한다. 정확한 발견은 더 좁고 날카롭다. 첫째, **교전의 깊이가 균일하게 한 문장이고 실험적 접촉이 0이다** — 모든 그림의 비교 집합이 자기 자신의 $N=1$ 판본이다. 둘째, 더 큰 단절이 경로 사이가 아니라 **경로 안**에 있다. 이 논문의 "learned local rule"은 gradient step이 **아님**을 입장으로 내세우는데, 그것은 Titans 계열이 정확히 반대편을 잡은 자리다. $W$-경로의 두 절반이 서로 말하지 않는다. ch11의 침묵 지도에서 이 항은 경로 사이가 아니라 경로 안에 기록된다.

### 17.8.3 다음 장이 받아가는 것

이 장은 $W$-경로에 "질의 전에 실행되는 pass"라는 자리를 만들었다. 그 자리가 생기자마자 두 질문이 따라온다.

첫째, **$W$에 접는 것과 $\Theta$에 쓰는 것이 정말 다른 종류의 행위인가.** 이 장의 sleep은 갱신 규칙을 바꾸지 않고 같은 규칙의 적용 횟수만 늘렸다. 갱신 주파수를 축으로 놓으면 그 조작은 축 위의 이동으로 읽힌다. ch18이 그 축을 주장하는 논문과 그 축 위에서 상태를 키우는 논문을 함께 받아 검증한다 — 연속체가 표기의 다리인지 기제의 다리인지가 거기서 판정된다.

둘째, **$C_{\text{cap}}$의 빈칸.** 이 장은 바이트를 쓸 수 없다고 기록했고, 그 칸을 채우는 것은 이 책의 실험이며 판정은 ch26이 소유한다. 계산량을 맞춘 비교가 없다는 §17.6.1의 공백은 $B_s$를 축으로 세우는 ch28로 가고, forward 쓰기가 gradient 쓰기에 무엇을 내주는가라는 §17.3.2의 미결은 ch16에서 이어져 ch19로 간다.

그리고 이 장이 남기는 가장 무거운 한 줄을 ch24·ch25가 받는다. **이 논문은 $W$-경로가 sleep-time compute이 될 수 있음을 보인 유일한 정면 사례이면서, 그 sleep의 산출물이 요청보다 오래 살지 않는 사례이기도 하다.** 판별식을 통과하는 것과 상각되는 것은 같은 일이 아니다.

## 요약

- `Do LMs Need Sleep?`(2605.26099)은 상속받은 open question 없이 시작한다. 이 장의 bridge-in은 상속 진술이 아니라 **상속된 오진의 교정**이다 — attention-SSM 하이브리드의 실패 원인을 용량에서 계산으로 옮기고, 선행 연구를 명시적 반박의 형태로 인용한다 [§1, §2].
- 기제는 (U-W) 하나이며 두 군데가 표준형과 다르다. gradient 자리에 게이트된 outer product가 들어가고 momentum이 없으며, (U-W)에 없는 pass 첨자 $N$이 추가된다. 같은 chunk를 $W$에 $N$번 누적해 쓰되 매 pass의 $(k_t,v_t)$를 직전 pass의 $W$가 만든 특징에서 다시 뽑는다 — (U-W)와 특징 정제 고정점 반복의 합성이다. 표기 충돌은 corpus에서 가장 심하다: 원문의 $S_t$가 이 책의 $W_t$, $\beta_t$가 $\eta_t$이고, 원문의 $L$과 $T$가 **이름을 맞바꾼다**(표 17-1).
- **논문이 보여 주는 식은 논문이 돌린 식이 아니다.** Eq. 3은 Mamba-2 형태이고 실험은 GDN이며 §6.3에는 Jet layer가 또 다르게 등장한다. 국소 규칙 ablation은 없고, 논문 자신이 "the specific update rule does not matter"라고 범위 밖으로 선언한다 [§3.1].
- 층은 $W$ 하나다. $\Theta$는 outer loop에서만 움직이고 $E$는 존재하지 않으며, 학습 시점의 sleep/wake 분할은 런타임 정책이 아니라 **loss mask**가 정한다 [Alg. 1]. 판별식 통과는 **Depo 설정이 진다** — 거기서만 consolidation이 질의보다 앞서고 $N_q=10$이 구조로 정해진다. 헤드라인인 GSM-Infinite는 질문을 문맥 앞에 놓아 consolidation이 질의 조건부가 되고, 그 대가로 $N_q=1$에서 식 (A)가 $C_{\text{avg}}=C_{\text{wake}}+C_{\text{sleep}}$으로 붕괴한다.
- 비용 4종이 **전부 논문에 없다.** $B_s$는 구조로만, $L_w$는 아키텍처 보장으로만 있고, $C_{\text{cap}}$은 상태 차원·head 수·dtype이 없어 유도조차 불가능하며, $\rho$는 관측될 레짐(경계 1–6개, sequence마다 0 초기화)에 논문이 들어가지 않는다. 헤드라인 지연 주장은 프로토콜 구성상 참이지 측정된 값이 아니다.
- 실험은 세 설정 모두 합성이고 저자 제작·재구성이며, 시드 하나에 오차막대가 없고, learning rate는 $N=1$에서만 튜닝되었고, **계산량을 맞춘 baseline이 없다.** 캐시를 유지하도록 허용한 full-attention baseline은 어느 과제·어느 스케일에서도 실행되지 않았다.
- 수치가 나오는 곳은 GSM-Infinite 하나다. 이득은 6·8연산 꼬리에만 있고(Ouro 6연산 0.419 → 0.615), 쉬운 문제에서는 폭이 0.3–2.9 백분점이며, Ouro 8연산에는 $N=2$가 $N=1$보다 낮은 반전이 있다. 가장 큰 상대 이득 52%는 같은 모델이 다른 설정에서 loop 없이 이미 도달하는 수준의 회복이다.
- 서빙에서 decode는 바뀌지 않고, 경계마다 chunked prefill 버스트가 붙으며, 그 산출물은 요청과 함께 죽는다. 사용자당 지속 저장소가 생기지 않으므로 요청 간 상각도 없다.

## 자가 점검 체크리스트

- [ ] 이 장의 bridge-in이 왜 상속 진술이 아니라 오진의 교정인지 말하고, 노트가 "none-stated"인 논문에 계보를 지어내지 않는 것이 왜 이 책의 규율인지 설명할 수 있는가.
- [ ] 식 (17-3)을 표준형 (U-W)와 항별로 대조해 다른 곳 둘(momentum 부재, gradient 자리의 outer product)과 **추가된 것 하나**(pass 첨자 $j$)를 지목하고, 그것이 왜 "(U-W) step을 더 밟는 것"과 다른지 말할 수 있는가.
- [ ] 표 17-1에서 $L$과 $T$가 맞바뀌는 행을 짚고, 본문의 "$C=2000$, $L\in[2000,3300]$"이 원문에서는 무엇으로 쓰여 있는지 말할 수 있는가.
- [ ] 판별식 네 조건을 Depo 설정과 GSM-Infinite 설정에 각각 대고, 어느 조건이 어느 설정에서 왜 약해지는지, 그 약화가 식 (A)에서 무엇을 없애는지 말할 수 있는가.
- [ ] 표 17-3의 네 칸이 각각 **어떤 종류로** 비어 있는지 구분할 수 있는가(수치 부재 / 측정 부재 / 유도 불가 / 레짐 미진입). 그리고 이 논문에 대해 "효율적"이라고 쓸 수 없는 이유를 한 문장으로 말할 수 있는가.
- [ ] 계산량을 맞춘 baseline의 부재가 Fig. 2b·3·4·5의 모든 비교에서 무엇을 뜻하는지 말하고, 같은 x좌표에서 $N=4$ 실행이 이미 쓴 계산량을 어림할 수 있는가.
- [ ] 이 논문이 $E$-경로와 $\Theta$-경로를 각각 몇 문장 인용하는지, 그리고 자기 경로의 어느 계보에 침묵하는지 말할 수 있는가. 그 침묵이 연대가 강제한 것이 아니라 선택된 것임을 어떻게 판정하는가.
- [ ] (Rosetta) 이 논문의 sleep을 "warm-up / 사전 컴파일" 행으로 옮겨 설명하고, 그럼에도 "prefix cache" 행과 달리 산출물이 요청을 넘지 못하는 이유를 $W$의 초기화 규칙으로 말할 수 있는가.
