# ch16. Generative Adapter — 경첩

## 16.1 Bridge-in: 전작이 남긴 문제

ch12–ch15는 $E$-경로를 끝까지 따라갔다. 그 경로의 구조적 약속은 하나였다 — 문맥의 산출물을 모델 **밖**에 레코드로 두고, 질의 시점에 $\mathrm{ret}(E,q)$로 다시 끌어온다. 그 약속의 대가는 ch10이 항 ②로 분해한 prefill 증가이고, ch15가 마지막 체크리스트 항목에서 확인한 캐시 재사용의 상실이다. 레코드는 질의마다 다르게 뽑히므로 공유 프리픽스를 위치 0에서 깨뜨린다.

이 장은 반대편 선택지를 연다. 문맥을 밖에 두는 대신 **파라미터의 모양으로 접어 넣는다.** 접어 넣은 뒤에는 원 문맥 토큰을 입력에서 지우고 closed-book으로 답한다. 그러면 항 ②가 사라지고 공유 프리픽스 문제도 사라진다. 결정적인 것은 그 접기가 **학습이 아니라 forward**라는 점이다 — backward pass도, optimizer 상태도, 학습률 스케줄도 없다.

**이 논문은 어떤 선행 논문의 open question도 인용해 받지 않는다.** 오히려 반대를 명시한다 — "As far as we know, we are the first to explore this direction" [Generative Adapter §1], 그리고 기여 목록의 첫 항에서 "To our knowledge, we are the first to explore retaining the relevant temporary knowledge through generated parameter-efficient model updates for state-of-the-art pretrained LMs" [Generative Adapter §1]. Related Work(§6)는 세 개의 대조 문단으로 짜여 있고("Different from previous work…", "Unlike those methods…", "Instead, GenerativeAdapter is…"), 각 문단은 받은 질문이 아니라 차이를 주장한다.

ch14가 $E$-경로에서 상속 없는 시작이었다면 이쪽은 종류가 다르다. ch14는 같은 공동체 안에서 계보 진술을 생략했지만, 이 논문은 **다른 공동체에서 입장한다** — fast-weight와 PEFT 엔지니어링 쪽이며, 지속학습으로도 에이전트 기억으로도 실이 닿지 않는다.

없는 계보를 지어내지 않는 대신, 이 논문이 **암묵적으로 답한** 질문 하나를 복원해 둔다. Schmidhuber(1992·1993)의 fast-weight programmer 질문 — slow network가 fast network의 가중치를 프로그램할 수 있는가 — 이 여기서는 "동결된 7B pretrained LM과의 접촉에서 살아남는가"라는 형태로 던져진다. 논문 자신이 고전적 구성과의 차이를 이름 붙인다.

> "Instead of using a slow network to program a separate fast model, our method can be viewed as a self-programming model, i.e., context encoded by the base LM is used to update the base LM itself." [Generative Adapter §6, Fast Weights]

그리고 이 논문이 **질문이 아니라 기정사실로** 수입한 것이 하나 더 있다. "acquiring knowledge through continual pretraining has shown to be data-inefficient (Yang et al., 2024; Allen-Zhu & Li, 2024)" [Generative Adapter §1]. 이 문장이 논문 전체의 동기를 떠받치는데, 질문이 붙어 있지 않다.

ch11이 이 장에 넘긴 일은 하나다 — **산출물의 모양과 시계가 갈리는 사례를 끝까지 따라가는 것**(→ ch11 §11.4.5). 이 장은 그 판정을 결론으로 옮겨 적지 않고 판별식으로 다시 밟는다(§16.4).

## 16.2 문제의식

논문이 세우는 대립은 두 항이다.

> "However, there is an accuracy compute tradeoff—finetuning incurs significant training cost and prompting increases inference overhead." [Generative Adapter Abstract]

§1이 그 두 항을 각각 풀어 쓴다. prompting 쪽은 저장·추론 부담이다 — "to maintain additional memory across sessions, some extra prompts must be added to the input, which incur an inference-time or storage overhead (Chevalier et al., 2023)". finetuning 쪽은 학습 부담이다 — "it requires a training phase that is more computationally expensive than a single forward pass" [Generative Adapter §1].

목표는 §2.1이 한 문장으로 못 박는다. "To contextualize a base model, Θbase, to a given context C, our goal is to obtain an updated model, ΘC, that can respond to user instructions using the information provided in the context C." 그리고 그 갱신이 **스트리밍**임을 곧바로 덧붙인다.

> "We specifically focus on test-time contextualization, where context arrives incrementally as a stream of data… In this online adaptation scenario, the model must be efficiently adapted to each new context chunk as it becomes available." [Generative Adapter §2.1]

적용 대상 시나리오는 셋이다 — 문서에서의 지식 획득, 시연에서의 학습, 사용자 개인화 [Generative Adapter Abstract, §1, §4].

> **[해설]** 이 책의 어휘로 옮기면 논문이 그린 2×2가 사실은 1×2다. prompting은 문맥을 **토큰**으로 두고, finetuning은 문맥을 **가중치**로 옮기되 gradient로 옮긴다. 논문이 원하는 것은 가중치이되 gradient가 없는 것이다. 그것은 $\Theta$-경로의 어휘로 진술된 $W$-경로의 제안이다. 이 문장이 이 장 전체의 요약이며, §16.4가 그것을 판별식으로 증명한다.

한 가지를 먼저 못 박는다. **이 논문에는 "sleep"도 "offline"도 "idle"도 "latency"도 한 번 나오지 않는다**(전문 검색으로 확인). 이 논문은 ch01의 조작적 정의에서만 이 범주에 접근하고, 자기 규정으로는 한 번도 sleep-time compute을 자처하지 않으며, 실제로 판별식 조건 (1)에서 탈락한다(→ ch11 §11.3). 그럼에도 이 장이 Part II의 한가운데 놓이는 이유는 이 논문이 **경첩**이기 때문이다 — $\Theta$-경로가 gradient로 쓰는 바로 그 아티팩트를, $W$-경로의 시계와 쓰기 규칙으로 만들어 낸다.

## 16.3 Core mechanism (통일 표기)

이 절이 이 장의 두 정의를 소유한다.

> **정의.** **한 번의 forward로 문맥을 파라미터화하는 기제**는, 문맥을 소비하는 계산이 base 모델의 forward pass 하나와 파라미터 없는 후처리만으로 끝나고, 그 산출물이 가중치와 같은 모양의 텐서이며, 읽기 시점에 원 문맥 토큰이 입력에 존재하지 않는 기제를 말한다. 셋 중 하나라도 어긋나면 다른 기제다 — backward pass가 있으면 학습이고, 산출물이 토큰이면 압축이며, 읽기에 원 문맥이 남아 있으면 prompting이다.

배치부터 적는다. base LM은 Mistral-7B-Instruct v0.2 또는 Llama2-7B-Chat이고 전 과정에서 동결된다. adapter가 실리는 자리는 **본 실험 전체에서 블록당 하나** — multi-head attention의 출력 투영 행렬이다 [Generative Adapter §3]. 문맥은 1,024 토큰의 chunk로 잘려 들어온다($C = 1{,}024$) [Generative Adapter §3 Training].

읽기를 표준형 (R)로 환원하면 이렇다.

$$
\hat y \;=\; f\big(q;\ \Theta,\ W_t,\ \varnothing\big),
\qquad
\Theta \;=\; \Theta_{\text{base}} \cup \Theta_G
\tag{16-1}
$$

$E = \varnothing$이다. **Part II에서 외부 저장소가 아예 없는 첫 장이다.** 읽기는 $\Theta$와 $W$만 통과하고 $\mathrm{ret}(\cdot,\cdot)$이 등장하지 않는다. 층별로는 유효 행렬이 덧셈으로 만들어진다 — 원문 $W^{(l)} = W^{(l)}_{\text{base}} + W^{(l)}_\Delta$ [Generative Adapter §2.2]가 이 책의 기호로 $\Theta^{(l)} + W^{(l)}_t$이며, $l = 1,\dots,L_{\text{layer}}$ 중 adapter가 실린 자리에서만 두 번째 항이 0이 아니다.

쓰기는 두 식이다. 상태 갱신이 먼저다.

$$
S_t \;=\; \beta_t\, S_{t-1} \;+\; A_2\, H_t^{\top} H_t\, B_1,
\qquad
\beta_t \equiv 1, \quad S_0 = 0
\tag{16-2}
$$

$H_t \in \mathbb{R}^{M_{\text{ctx}} \times d}$는 동결된 base LM이 $t$번째 chunk에 대해 이미 만든 hidden state 행렬이고, $A_2, B_1$은 학습된 사영 행렬이며, $S_t \in \mathbb{R}^{d_r \times d_r}$은 $d_r = 1{,}024$의 정사각 상태다 [Generative Adapter Eq. 3; §3 Hyperparameters]. 읽기용 가중치는 그 상태에서 매 chunk **다시 계산된다**.

$$
W_t \;=\; A_1\, \mathrm{norm}\big(S_t\big)\, B_2
\tag{16-3}
$$

식 (16-3)이 표준형 (U-W)와 갈라지는 첫 지점이다. $W_t$는 $W_{t-1}$을 이어받지 않는다 — retention이 통째로 $S_t$ 쪽으로 옮겨가 있다 [Generative Adapter Eq. 4, Eq. 6]. 그리고 (16-2)의 재귀는 배치 형태와 **정확히 같다**.

$$
S_t \;=\; A_2\Big(\sum_{i=1}^{t} H_i^{\top} H_i\Big) B_1
\;=\; A_2\Big(\sum_{m=1}^{M_{\text{ctx}}} h_m \otimes h_m\Big) B_1
\tag{16-4}
$$

식 (16-4)는 원문 Eq. 1–2의 형태다. 오른쪽 끝이 말하는 것은 쓰기가 **토큰들의 합**이라는 것이다. 이 성질이 chunk 병렬화를 가능하게 하고 — "GenerativeAdapter is able to generate the adaptors for prefixes of chunks simultaneously by processing the context chunks in parallel" [Generative Adapter App. B] — 동시에 recency도 순서도 축출도 금지한다.

정규화는 안정화 장치다.

$$
\mathrm{norm}(M) \;=\; U V^{\top}, \qquad M = U \Sigma V^{\top}
\tag{16-5}
$$

양의 특이값을 전부 1로 되돌린다 [Generative Adapter Eq. 7]. 이것이 없으면 학습이 수렴하지 않는다는 것이 도입 이유이고 [Generative Adapter §2.4], 부수 효과로 rank $r$까지 절단하면 곱이 두 개의 얇은 행렬이 된다 — **LoRA 형태다**(→ ch03 §03.2의 정의). 논문이 그 유도를 직접 적는다 [Generative Adapter Eq. 8, Eq. 9]. $r = 128$이다 [Generative Adapter §3 Hyperparameters].

학습되는 것은 사영 행렬뿐이고, 그 학습은 배포 전 outer loop 한 번이다.

$$
\Theta_{G} \;\leftarrow\; \Theta_{G} \;-\; \eta_\Theta \nabla_{\Theta_G} \mathcal{L},
\qquad
\mathcal{L} \;=\; -\Big[\log P\big(x_{1:m} \mid \Theta_{\text{base}} + W(x_{1:m})\big) \;+\; \log P\big(x_{m+1:n} \mid \Theta_{\text{base}} + W(x_{1:m})\big)\Big]
\tag{16-6}
$$

앞 항이 reconstruction, 뒤 항이 completion이다 [Generative Adapter Eq. 5, §2.3]. 원문은 두 log-likelihood를 **최대화**하므로 이 책의 최소화 관행에 맞춰 부호를 뒤집었다. gradient는 동결된 base LM을 **통과해** $\Theta_G$에 도달한다 [Generative Adapter §2.3]. 여기서 $\mathcal{R}$에 해당하는 것은 replay도 합성 데이터도 아니다 — SlimPajama 원시 웹 토큰 1B와 뒤이은 instruction-tuning 혼합이다 [Generative Adapter §3 Training].

표준형 (U-W)와 항별로 대조하면 넷이 비어 있고 하나가 바뀐다.

- $\alpha_t$: **없다.** $W_t$는 $W_{t-1}$의 게이트된 이월이 아니라 $S_t$의 무상태 함수다. $\alpha_t = 0$에 retention을 통째로 $S_t$로 옮긴 것과 같다.
- $\beta_t$: **항등적으로 1이다.** 감쇠도 게이팅도 축출도 없다. 아무것도 잊히지 않는다.
- $\eta_t$: **없다.** 쓰기에 step size가 없고, 가장 가까운 것은 손으로 정한 고정 adapter 배율뿐이다 — Mistral 1/16, Llama2 1/8 [Generative Adapter App. B]. 스케줄이 아니라 상수다.
- $\nabla_W\, \ell(W_{t-1}; k_t, v_t)$: **chunk 자신의 Gram 행렬의 학습된 쌍선형 사영으로 대체된다.** 이 논문에는 inner loss $\ell$이 없다. $H_t^\top H_t = \sum_m h_m \otimes h_m$이므로 연상쌍 $(k_t, v_t)$가 $(h_m, h_m)$으로 붕괴한다 — key와 value가 분리되어 있지 않고, 모델이 각 hidden state를 자기 자신에 대해 쓴다. Hebbian 쓰기다.
- 비선형이 놓인 자리가 다르다. (U-W) 계열은 통상 쓰기에 비선형을 두고 읽기를 선형으로 두는데, 여기서는 읽기 쪽 (16-3)에 SVD가 매 chunk 들어간다. 그리고 상태 $S_t$($d_r \times d_r$)가 산출물 $W_t$($d_{\text{out}} \times d_{\text{in}}$)보다 작다.

두 번째 정의가 이 장의 방법론적 소유물이다.

> **정의.** **산출물의 모양과 갱신의 시계를 분리해 보는 관점**은, 어떤 절차를 분류할 때 (i) 그 절차가 남기는 상태의 **물리적 모양**(텍스트 레코드인가, fast weight 텐서인가, 가중치 delta인가)과 (ii) 그 상태가 움직이는 **시간척도**(토큰·chunk인가, 세션인가, 학습 라운드인가)를 독립된 두 축으로 두고, **경로 판정을 시계와 갱신식 쪽에만 맡기는** 관점이다. 모양은 저장·전송·롤백의 단위를 정하고, 시계는 갱신식과 비용 구조를 정한다. 두 축은 함께 움직이지 않으며, 이 논문이 그 사실의 가장 깨끗한 사례다.

### 16.3.1 표기 대응표

표 16-1 — Generative Adapter 원 표기 → 이 책의 표기

| 원 표기 | 이 책의 표기 | 주의 |
|---|---|---|
| $\Delta,\ \Delta_t,\ W_\Delta,\ W_{\Delta t}$ — 문맥 의존 가산 adapter | $W_t$ | **이 장에서 가장 중요한 치환이다.** 논문의 단어 "adapter"는 LoRA에서 온 $\Theta$-경로 어휘인데, 여기서 그것이 가리키는 대상은 $W$-경로 객체다 |
| $S_t$ — $d_r \times d_r$ 누적 부분합, "the memory of history context chunks" [§2.2] | $S_t$ | 슬롯은 정확히 (U-W)의 $S_t$ 자리다. **내용이 다르다** — 이 책의 $S_t$는 gradient momentum, 논문의 것은 Hebbian 누적기다 |
| $\Theta_{\text{base}},\ \Theta_C,\ \Theta_{\Sigma(t)}$ — 동결 base 가중치, 문맥화된 모델 | $\Theta = \Theta_{\text{base}} \cup \Theta_G$; $\Theta_{\Sigma(t)} \to \Theta + W_t$ | outer loop에서 움직이는 것은 $\Theta_G$뿐이고 $\Theta_{\text{base}}$는 이 논문에서 한 번도 움직이지 않는다. $\Theta_{\Sigma(t)}$라는 표기가 $\Theta$가 갱신된다는 인상을 주지 않게 한다 |
| $C,\ C_t$ — 문맥, $t$ 시점 도착 chunk [§2.1] | 내용은 $c_t$, 크기는 $C = 1{,}024$ 토큰 | 용량은 언제나 $C_{\text{cap}}$. 논문의 $C_t$를 본문에 그대로 쓰지 않는다 |
| $\Sigma,\ \Sigma(t)$ — 스트리밍 문맥 $(C_1,\dots,C_t)$ [§2.1] | $c_{1:t}$ | 논문 내부 충돌: 같은 $\Sigma$를 Eq. 7의 특이값 행렬로 재사용한다. 특이값 쪽 $\Sigma$는 식 (16-5) 안에서만 쓴다 |
| $G,\ G^{(l)}$ — layer별 adapter generator | $G$ 유지 | 파라미터 $\{A_1, A_2, B_1, B_2\}^{(l)}$는 $\Theta$의 부분집합이다. (U-$\Theta$)의 $\mathrm{gen}(\cdot; B_s)$와 혼동 금지 — 그쪽은 학습 집합을 만드는 절차다 |
| $H,\ H^{(l)},\ H_t$ — base LM hidden state 행렬, $M \times d_h$ [§2.2] | $H_t$ (로만체, hidden states) | 예약된 $\mathcal{H}_k$(누적 경험)는 이 논문에 인스턴스가 없다 — sleep 라운드가 없다 |
| $M$ — 문맥 토큰 수 [§2.2]이자 $\mathrm{norm}(M)$의 일반 행렬 [§2.4] | 토큰 수는 산문에서 $M_{\text{ctx}}$, 일반 행렬은 식 (16-5) 안에서만 | 맨 $M$ 금지 — $\mathcal{M}(\cdot; W)$(읽기 함수)로 읽힌다 |
| $L$ — Transformer 블록 수 [§2.2]; $L_{\text{reconstruction}},\ L_{\text{completion}}$ [§2.3] | 블록 수는 $L_{\text{layer}}$ | $L$은 sequence 길이로 예약되어 있다. 원문의 두 $L$은 **최대화** 대상 log-likelihood이므로 식 (16-6)으로 옮길 때 부호를 뒤집었다 |
| $r$ — SVD 절단 rank, 128 [§3] | $r = 128$ | 충돌 없음. 논문 자신이 Eq. 8–9에서 LoRA rank와 동일시한다 |
| $d_r,\ d_h,\ d_{\text{in}},\ d_{\text{out}}$ — generator 중간 차원(1,024), hidden 차원, layer 입출력 차원 | $d_h = d$; $d_r$은 장-국소 기호로 유지 | $d_r$은 $r$과 다른 값이다(1,024 대 128) |
| $A_1, A_2, B_1, B_2$ — 학습 가능한 generator 행렬 [Eq. 1] | 그대로 유지 | $\alpha_t, \beta_t$는 게이트 전용 기호이므로 이 넷과 무관하다. 이 $A$·$B$는 **outer loop에서 학습되며** test-time에 적합되는 LoRA의 $A$·$B$가 아니다 |
| $x$ — 선형층 입력 벡터 [§2.4]이자 토큰 $x_1 \dots x_n$ [§2.3] | 인용할 때마다 구분 | 논문 내부 중복이다 |

## 16.4 어느 층을 언제 쓰는가

먼저 "이 값은 누가 학습하는가"에 전부 답한다.

표 16-2 — 이 값은 누가 학습하는가

| 객체 | 학습 주체 | 언제 | test-time 이동 | 출처 |
|---|---|---|---|---|
| $\Theta_{\text{base}}$ — 동결 base LM (Mistral-7B-Instruct v0.2 또는 Llama2-7B-Chat, 7B) | 이 논문에서는 없음 | 없음. 다른 곳의 pre-training에서 물려받음 | 아니오 | §1, §3 |
| $\Theta_G = \{A_1^{(l)}, A_2^{(l)}, B_1^{(l)}, B_2^{(l)}\}$ — adapter generator, 약 500M 파라미터 | outer loop gradient descent(Adam), **동결된 base LM을 통과해** 역전파 | 배포 전 한 번, 2단계: (i) SlimPajama 1B 토큰 자기지도 사전학습, (ii) QA·ICL·instruction 혼합 2 epoch | 아니오 | §2.3, §3, App. B |
| $S_t^{(l)} \in \mathbb{R}^{1024 \times 1024}$ — 누적 상태 | 학습되지 않음. forward-only 재귀로 **움직인다** | 도착하는 1,024 토큰 chunk마다 한 번. 문맥 스트림 시작마다 $S_0 = 0$ | **예** | Eq. 3, §2.2, §3 |
| $W_t$ — 생성된 adapter, 32M 파라미터 | 학습되지 않음. $S_t$에서 식 (16-3)으로 결정론적 재계산(gradient도 optimizer 상태도 없음) | $S_t$ 갱신 직후 chunk마다. 매 chunk 덮어씀 | **예** | Eq. 6, §3, Table 1 |
| $\mathrm{norm}(\cdot)$ — SVD 정규화 | 파라미터 없음 | 문맥화 시점, 매 chunk, adapter가 실린 매 layer. `torch.svd_lowrank()` 반복 1회로 구현 | 해당 없음 | Eq. 7, App. B |
| adapter 배율 (Mistral 1/16, Llama2 1/8) | 손으로 정한 하이퍼파라미터 | outer loop 학습 전 고정, base 모델마다 | 아니오 | App. B |
| $d_r = 1{,}024$, $r = 128$ | 손으로 정한 하이퍼파라미터 | outer loop 학습 전 고정 | 아니오 | §3 |
| adapter를 어느 모듈이 지는가 (본 실험은 attention 출력 투영만; feedforward down-projection은 ablation에만) | 설계 결정, 학습되지 않음 | outer loop 학습 전 고정 | 아니오 | §3, Table 2 |

시간척도는 셋이고, 셋 중 **sleep 라운드는 없다**. (i) outer loop 학습 — 배포 전 한 번, gradient 기반, $\Theta_G$ 대상. (ii) chunk 단위 문맥화 — forward-only, $S_t$와 $W_t$ 대상. (iii) 토큰 단위 decode — 아무것도 움직이지 않는다. sleep 라운드 첨자 $k$와 세션 첨자 $\tau$는 이 논문에서 **한 번도 인스턴스를 갖지 않는다**. wake 시점에 움직이는 것은 $S_t$와 $W_t$뿐이고 나머지는 전부 정적이다.

이제 경로를 판정한다. 결론을 옮겨 적지 않고 ch01 §01.4의 세 판별 조건을 하나씩 댄다.

**$\Theta$-경로 조건 — 배포 아티팩트가 바뀌는가.** 바뀌지 않는다. 논문이 정면으로 진술한다 — "the pretrained base LM remains frozen while we train the LM-specific adapter generator" [Generative Adapter §1], 그리고 §2 서두에서 "Unlike continual pretraining and supervised fine-tuning which update the pretrained LM via gradient descent, our method achieves adaptation using forward passes only". $\Theta_{\text{base}}$는 미분되지도 수정되지도 않으므로 문맥화 이후의 체크포인트는 이전과 바이트 단위로 같다. **탈락.**

**$E$-경로 조건 — (U-E)만 실행하고 $\Theta, W$가 불변인가.** 텍스트도 벡터도 그래프도 기록되지 않고, 읽기 시점에 $\mathrm{ret}(E,q)$가 없다. 논문의 요점 자체가 문맥을 다시 제시하지 **않는** 것이다 — 문맥화된 모델은 "in a closed-book fashion"으로 답한다 [Generative Adapter §4.1]. $E$ 자리가 비어 있으므로 (U-E)의 인스턴스가 없다. **탈락.**

**$W$-경로 조건 — 문맥을 fast weight로 접어 넣고 원 문맥을 버리는가.** 그렇다. 식 (16-2)–(16-3)이 (U-W)의 자리마다 대응물을 갖는다(gradient-free 변종이지만 형태는 (U-W)다), 상태는 고정 크기이고, 원 문맥 토큰은 읽기 입력에서 사라진다. **통과.**

$W$-경로 논문이다. 여기까지가 경로 판정이고, 소속 판정은 별개다. ch11 §11.2의 네 조건을 대면 이렇다 — (1) 질의 전인가: **아니다.** 접기는 문맥이 도착한 그 자리에서 wake pass 안에 일어나고, 모델은 방금 접은 그 문맥에 대해 답한다. (2) $B_s > 0$인가: 구조적으로는 그렇다(forward pass 하나 + SVD). (3) 어느 한 층의 상태가 실제로 바뀌는가: 그렇다, $W$가 움직인다. (4) 이후 질의에 쓰이는가: 같은 스트림 안에서는 그렇다. 조건 (1)에서 탈락하므로 **sleep-time compute이 아니다.** 이 논문은 $W$-경로의 **wake 쪽 끝점**이다.

> **[평가]** 이 판정이 왜 중요한가. 산출물만 보면 이 논문은 $\Theta$-경로처럼 보인다 — 만들어 내는 물건이 LoRA이고, 논문이 그 형태를 직접 유도하며 [Generative Adapter Eq. 8–9], 크기도 ch03 §03.6의 delta 아티팩트 공식으로 그대로 계산된다. 그런데 시계는 chunk이고, 쓰기 규칙에는 loss도 학습률도 backward pass도 없다. **산출물의 모양으로 경로를 나누면 틀린다** — 이것이 이 책의 방법론적 주장이고, 이 논문이 그 주장의 가장 깨끗한 반례 제조기다. 모양으로 나누면 이 논문이 $\Theta$-경로로 들어오고, 같은 오류가 SCM을 $\Theta$-경로로, Memory Caching을 sleep-time compute으로 들여보낸다(→ ch11 §11.4). 시계와 갱신식으로 나누면 세 판정이 동시에 바로잡히고, 바로잡힌 판정이 곧 비용·용량·롤백 단위의 판정이 된다. 이 장에서 그 실익이 구체적으로 나타나는 자리는 §16.7이다 — 롤백 단위가 "이전 체크포인트로 되돌리기"가 아니라 "$S_0 = 0$으로 초기화"이고, 그 차이가 서빙 설계를 통째로 바꾼다.

갱신 규칙의 **순서** 문제는 생기지 않는다. 이 논문은 (U-W) 하나만, 한 단계로 실행한다. 앞서는 (U-E)도 없고 뒤따르는 (U-$\Theta$)도 없다.

## 16.5 비용 4종

표 16-3 — 비용 4종

| 기호 | 값 | 근거 |
|---|---|---|
| $B_s$ | **수치로는 논문에 없음.** 구조로만 진술된다 — 문맥에 대한 동결 base LM의 forward pass 1회, adapter가 실린 layer마다·1,024 토큰 chunk마다 rank-128 randomized SVD 1회(`torch.svd_lowrank`, 반복 1회), 그리고 식 (16-2)–(16-3)의 사영 GEMM 넷. Figure 3의 왼쪽 패널이 문맥화 FLOPs를 teraFLOPs 단위 로그축으로 512–32K 구간에 그리지만 **본문이 그 패널의 어떤 수치도 인용하지 않는다** | §3, §4.1, App. B, Figure 3 |
| $L_w$ | **논문에 없음.** "latency", "ms", "TTFT", "time to first token"이 한 번도 나오지 않는다(전문 검색으로 확인). 어떤 종류의 벽시계 decode 측정도 없다 | 전문 검색 |
| $C_{\text{cap}}$ | **개수로만 보고. 바이트는 논문에 없음.** 문맥당 32M floats — "the generated adapter of 32 million parameters" [§3], Table 1의 "Extra Storage (M floats)" 열이 32. dtype이 어디에도 없고, 저장 대상이 $S_t$인지 Eq. 9의 저랭크 인수쌍인지도 적혀 있지 않다 | §3, Table 1 |
| $\rho$ | **논문에 없음.** 라운드당 열화량이 정의되지도 측정되지도 않았다. "forget", "forgetting", "catastrophic"이 한 번도 나오지 않는다(전문 검색으로 확인). 같은 모델이 문맥화를 반복한 뒤 앞선 자료로 재평가되는 실험이 없다 | 전문 검색, Eq. 3 |

$B_s$에 붙은 유일한 정량 수식어는 비교급 하나다 — "GenerativeAdapter requires preprocessing time (forward passes only) that is orders of magnitude smaller than CPT (which involves multiple forward and backward passes), as demonstrated in Figure 3" [Generative Adapter §4.1]. "orders of magnitude"가 전부다. 일회성 상각 성분은 따로 있다: generator의 outer loop 학습이 SlimPajama 1B 토큰에 대해 "approximately 20 hours using 8 NVIDIA H100 GPUs"이고 [Generative Adapter App. B], 뒤이은 instruction tuning 2 epoch의 벽시계는 보고되지 않았다.

$L_w$ 칸에 가장 가까운 대용물은 MSC에서의 질의당 추론 **계산량**이다(TFLOPS): Generative Adapter 0.505, closed-book 0.505, full-conversation prompting 2.059, UltraGist는 64/128/256/512/1K/2K 압축 토큰에서 각각 0.514 / 0.552 / 0.627 / 0.772 / 1.067 / 1.658 [Generative Adapter Table 1]. 문맥화 이후의 질의당 비용이 소수 셋째 자리까지 base 모델과 **같게** 보고된 것이 이 논문의 핵심 수치다. Figure 3의 가운데 패널이 StreamingQA에서 같은 평탄함을 문맥 길이에 대해 보인다.

$\rho$ 자리에 있는 것은 다른 양이다. 하나는 **한 스트림 안에서 문맥 길이에 따른 품질 저하**다 — Mistral/StreamingQA에서 F1이 512부터 32K까지 51.5 → 49.3 → 44.7 → 40.9 → 36.7 → 32.7 → 32.0으로 떨어진다 [Generative Adapter Table 5]. 이것은 상태가 차면서 생기는 압축 손실이지 라운드당 망각률이 아니다. $\beta_t = 1$이 스트림 전체를 하나의 누적으로 만들기 때문에 라운드를 분리할 수조차 없다. 다른 하나는 쓰기 후에도 일반 언어모델링 능력이 남는지를 보는 completion perplexity 7.40이다 [Generative Adapter Table 2]. 그런데 **문맥화되지 않은 base 모델의 기준 perplexity가 논문에 없어** 이 값이 함의하는 열화량을 정량화할 수 없다.

**상각식 (A)는 닫히지 않는다.** 분자 쪽은 알려져 있다 — MSC에서 질의당 $2.059 - 0.505 = 1.554$ TFLOPS가 절약된다 [Generative Adapter Table 1]. 분모 쪽 $C_{\text{sleep}}$은 Figure 3의 눈금 없는 로그축 곡선으로만 존재하고 본문에 수치로 인용된 적이 없다. 따라서 손익분기 $N_q^\ast$를 계산할 수 없다. 그럼에도 상각 논증은 두 번 등장한다 — "In practical scenarios with many queries from the same user on edge computing devices, the benefits of our method are even more evident" [Generative Adapter §1], "In real world scenarios with many queries from the same user, the benefits of our method are even more pronounced" [Generative Adapter §4.3]. $N_q$ 값도 없다.

> **[평가]** 이 논문은 ch14의 상시 caveat — 상각식이 $N_q$를 안다고 가정하는데 실서빙 $N_q$ 분포를 보고한 논문이 없다 — 에 한 겹을 더한다. 여기서는 $N_q$ 분포가 없을 뿐 아니라 **sleep 쪽 비용 자체가 수치로 없다.** 비율의 두 항이 모두 비어 있는 상태에서 비율이 주장된다. ch01이 정한 규칙에 따라 이 장은 이 논문에 대해 "효율적"이라고 쓰지 않는다. 쓸 수 있는 문장은 하나뿐이다 — **질의당 추론 FLOPs가 base 모델과 같다고 보고되었다.**

## 16.6 실험과 스케일

스케일 상한부터 적는다. base 모델은 7B 둘뿐이고 그 아래도 위도 없다 [Generative Adapter §3]. generator는 약 500M 파라미터, $d_r = 1{,}024$, $r = 128$, chunk 1,024 토큰, adapter가 실린 자리는 블록당 attention 출력 투영 하나다. 사전학습 코퍼스는 SlimPajama에서 무작위 추출한 1B 토큰(8,192 토큰 세그먼트로 분할)이고, instruction tuning은 COQA·DROP·NarrativeQA·PubMedQA·Quail·MS MARCO·MetaICL·BookSum·PwC 혼합이다 [Generative Adapter §3, App. A, Table 3]. 문맥은 32K까지 평가된다.

표 16-4 — 문서 QA, F1, Mistral-7B-Instruct v0.2 [Generative Adapter Table 5]

| 데이터셋 | 방법 | 512 | 1K | 2K | 4K | 8K | 16K | 32K |
|---|---|---|---|---|---|---|---|---|
| SQuAD | zero-shot prompting | 10.8 | | | | | | |
| SQuAD | supervised fine-tuning | 20.7 | | | | | | |
| SQuAD | continuous pretraining | 30.0 | | | | | | |
| SQuAD | in-context prompting | 45.4 | 44.9 | 43.6 | 42.6 | 42.5 | 38.6 | 35.1 |
| SQuAD | Generative Adapter | 48.8 | 43.0 | 39.9 | 35.9 | 33.8 | 30.3 | 28.0 |
| StreamingQA | zero-shot prompting | 13.6 | | | | | | |
| StreamingQA | supervised fine-tuning | 19.5 | | | | | | |
| StreamingQA | continuous pretraining | 22.2 | | | | | | |
| StreamingQA | in-context prompting | 47.2 | 48.7 | 48.1 | 48.7 | 48.0 | 46.0 | 39.3 |
| StreamingQA | Generative Adapter | 51.5 | 49.3 | 44.7 | 40.9 | 36.7 | 32.7 | 32.0 |

*closed-book 행(zero-shot·SFT·CPT)은 문맥 길이와 무관한 단일 값이라 첫 열에만 적었다. 원문 표의 (모델, 데이터셋) 블록 배정은 초록이 인용한 SFT 19.5(Mistral/StreamingQA)와 §4.1의 Llama2 4K 절단 규칙으로 고정했다.*

**초록의 헤드라인 수치가 본문 표와 화해하지 않는다.** 초록은 "achieving a 63.5% improvement in F1 score over the model with supervised fine-tuning (from 19.5 to 31.5) for contexts as long as 32K tokens"라고 쓴다 [Generative Adapter Abstract]. 기준값 19.5는 표 16-4의 Mistral/StreamingQA SFT와 일치한다. 그런데 **31.5는 논문 어디에도 없다** — 표 16-4의 32K 값은 32.0이다. 게다가 19.5 → 31.5는 63.5%가 아니라 61.5% 증가이고, 19.5 → 32.0이라면 64.1%다. 목표값도 백분율도 맞지 않는다. 이 책은 두 값을 나란히 적고, 어느 쪽도 초록의 진술을 지지하지 않는다고 기록한다.

가장 센 baseline과의 대조는 다르게 읽힌다. Mistral에서 Generative Adapter가 in-context prompting을 이기는 칸은 열넷 중 **셋**뿐이다 — SQuAD 512(48.8 대 45.4), StreamingQA 512(51.5 대 47.2), StreamingQA 1K(49.3 대 48.7). 32K에서는 SQuAD 28.0 대 35.1, StreamingQA 32.0 대 39.3으로 진다. 초록의 표현("effective in injecting knowledge into the LM's parameters")이 재는 상대는 prompting이 아니라 가장 약한 baseline인 SFT다. Llama2에서는 8K 이상에서 이기지만(SQuAD 28.2/24.9/23.6 대 25.2/9.6/6.4; StreamingQA 28.7/26.0/25.7 대 27.8/17.5/11.6), 그 구간의 baseline은 **설계상 잘려 있다** — "for Llama2-7B-Chat, if the context length exceeds the maximum limit of 4K tokens, we truncate the prompt to include only the last 4K tokens" [Generative Adapter §4.1]. 둘 다 문서 전체를 보는 512에서는 in-context prompting 64.8 대 Generative Adapter 36.2로 28.6점 차이다. 장문맥 우위 증거의 절반이 능력 비교가 아니라 절단 아티팩트다.

CPT와의 대조에는 교차가 있다. SQuAD/Mistral에서 512–16K는 전부 CPT의 30.0을 넘지만 32K에서 28.0으로 진다. SQuAD/Llama2도 32K에서 23.6 대 23.9로 진다. StreamingQA 두 블록에서는 모든 길이에서 이긴다.

나머지 두 시나리오는 증거의 성격이 다르다. MetaICL은 26개 테스트 과제, $K \in \{1,2,4,8,16\}$ 시연, 설정마다 5회 반복 표집이다 [Generative Adapter §4.2]. 그런데 헤드라인 "our method achieves an average accuracy of 44.9 across 26 tasks" [Generative Adapter Abstract]는 **초록에만 있다** — §4.2에도, 어떤 표에도 없고, 어느 base 모델인지도 어느 $K$인지도 붙어 있지 않다. §4.2가 내놓는 것은 Figure 4의 범주별 곡선과 Figures 5–6의 과제별 곡선뿐이고 본문에 수치 집계가 없다. 세 헤드라인 중 하나를 논문 자신의 본문으로 검증할 수 없다.

표 16-5 — 개인화, Multi-Session Conversation, Mistral-7B-Instruct v0.2 [Generative Adapter Table 1]

| 방법 | F1 | 추론 계산 (TFLOPS) | 추가 저장 (M floats) |
|---|---|---|---|
| Closed-book | 8.1 | 0.505 | 0 |
| Full-conversation Prompting | 66.0 | 2.059 | 128+ |
| UltraGist (512 토큰) | 40.8 | 0.772 | 32 |
| UltraGist (1K 토큰) | 44.4 | 1.067 | 64 |
| Generative Adapter | 40.2 | 0.505 | 32 |

*UltraGist의 나머지 네 설정(64/128/256/2K 토큰)은 F1 26.5/32.2/38.3/42.4, 추론 0.514/0.552/0.627/1.658, 저장 4/8/16/128이다.*

"4× 절감" 주장은 산술이 맞는다 — $2.059 / 0.505 = 4.08$배 계산, $128 / 32 = 4$배 저장 [Generative Adapter Abstract, §4.3]. 그 대가도 같은 표에 있다: full-conversation prompting 66.0 대 Generative Adapter 40.2로, 전체 문맥 품질의 61%를 유지한다. 25.8점 F1 결손이 4× 절감의 가격이며, 초록의 형용사("highly competitive")는 비율에 대한 것이지 정확도에 대한 것이 아니다. 그리고 같은 절에 표와 어긋나는 문장이 하나 있다 — "Comparing to UltraGist at the same level of storage cost (compressed into 512 tokens), GenerativeAdapter further reduces inference cost without performance drop" [Generative Adapter §4.3]. 표 16-5는 UltraGist(512) 40.8, Generative Adapter 40.2다. 0.6점 하락이 하락 없음으로 서술되어 있다. 계산 절감(0.772 → 0.505)은 실재하지만 문장은 옆의 표가 지지하지 않는다.

표 16-6 — ablation, Mistral-7B-Instruct v0.2, 검증 perplexity [Generative Adapter Table 2]

| 설정 | reconstruction ppl | completion ppl |
|---|---|---|
| 기본값 (두 과제 + SVD norm + attention 출력 투영) | 1.75 | 7.40 |
| 사전학습 과제: reconstruction만 | 1.75 | 34.34 |
| 사전학습 과제: completion만 | 6.38 | **6.71** |
| 정규화: Frobenius | 7.72 | 7.32 |
| 모듈: feedforward (down projection; 기본값의 3배 갱신 파라미터) | **1.68** | **7.26** |

표 16-6에 두 가지 역전이 있다. 첫째, "relying solely on one task does not yield good perplexity on the validation set for both metrics" [Generative Adapter §5.1]는 reconstruction-only 행(1.75 / 34.34)에만 참이다. completion-only 행은 completion perplexity에서 기본값보다 **낫다**(6.71 대 7.40). 두 과제 혼합은 Pareto 개선이 아니라 교환이고 논문이 그렇게 쓰지 않는다. 둘째, feedforward 배치가 두 지표 모두에서 기본값보다 좋은데(1.68 / 7.26 대 1.75 / 7.40) **본 실험 전체가 기본값 구성을 쓴다** — "For efficiency, our main experiments train adapter generators to only update the output projection layers" [Generative Adapter §3]. 모든 헤드라인 수치가 저자 자신의 ablation이 열등하다고 판정한 구성에서 나왔고, 더 나은 구성은 어떤 downstream 벤치마크에서도 돌려지지 않았다.

그 ablation 전체가 두 개의 검증 perplexity 위에 서 있다는 점도 함께 적어야 한다. perplexity와 downstream 품질의 연결은 보고되지 않은 예비 실험에서 주장된다 — "As we observe in our preliminary study, the quality of the resulting adapter generator is highly correlated with these metrics" [Generative Adapter §5.1]. 어떤 ablation 행도 SQuAD·StreamingQA·MetaICL·MSC에서 평가되지 않았다. 설계 결정 셋(정규화·과제 혼합·모듈 배치)의 근거가 전부 대용 지표다.

실증 상한을 정직하게 정리하면 이렇다. 모델은 7B 둘, 문맥은 32K까지, 그러나 **우위**의 상한은 훨씬 낮다 — Mistral에서 in-context prompting 대비 우위는 1K 토큰까지다. MSC 대화는 평균 2.5K 토큰이고, 단일 연속 스트림을 넘는 실험이 없다 — 다중 세션도, 다일(多日)도, 반복 라운드도 없다. MSC(Table 1)와 ablation(Table 2)은 Mistral 하나로만 돌았다. 분산은 어디에도 없다: MetaICL은 5회 반복 표집을 하면서 표준편차를 본문에도 그림에도 적지 않고, QA 결과에는 오차 막대가 없다. 그리고 가장 아픈 공백 하나 — **문맥마다 gradient로 적합한 LoRA가 baseline으로 없다.** 그것이 비용을 맞춘 $\Theta$-경로 비교군이자 이 방법이 배출하는 바로 그 물건인데, 가중치 쪽 baseline은 full-parameter SFT와 CPT 둘뿐이다.

## 16.7 Systems/serving 함의

독자의 1번 질문에 논문이 직접 답한다. **문맥화 이후 decode 비용은 base 모델의 decode 비용이다** — 0.505 대 0.505 TFLOPS, 소수 셋째 자리까지 같다 [Generative Adapter Table 1]. 문맥 길이에 대해서도 평탄하다 [Generative Adapter Figure 3 중앙 패널]. 이 등식이 성립하는 조건은 하나뿐인데 논문이 그 조건을 명시하지 않는다 — delta가 **merge된 형태**로 서빙되어야 한다(→ ch03 §03.5).

> **[해설]** **아래는 이 책의 산술이며 논문의 보고값이 아니다.** 논문이 블록 수를 적지 않으므로 7B 관행대로 $L_{\text{layer}} = 32$, $d = 4{,}096$을 가정한다. **분리된 delta 형태**로 서빙하면 adapter가 실린 행렬마다 토큰당 $2rd$ MAC이 더 붙는다 — $r = 128$에서 행렬당 약 1.05M MAC, 블록당 하나씩 32개면 토큰당 약 33.5M MAC(약 67 MFLOPs)이고, 7B forward의 토큰당 약 14 GFLOPs에 대해 약 0.5%다. 어느 형태든 덧셈 항은 작다. 문제는 크기가 아니라 형태 선택이 서빙 구조를 정한다는 것이다.
>
> 상태량도 같은 산술로 확인된다. ch03 §03.6의 delta 아티팩트 공식 $|\Delta\Theta| = 2 N_{\text{mod}} d r$에 $N_{\text{mod}} = L_{\text{layer}} \times |\mathcal{S}| = 32 \times 1$을 넣으면 $2 \times 32 \times 4096 \times 128 = 33{,}554{,}432$이고, 상태 $S_t$ 쪽도 $32 \times 1024 \times 1024 = 33{,}554{,}432$로 같은 수가 나온다. 논문의 "32 million floats"와 자릿수·값이 모두 맞는다 [Generative Adapter §3, Table 1]. 바이트로 옮기면 bf16에서 약 64 MB, fp32에서 약 128 MB인데 **dtype이 논문에 없으므로 이 두 값은 가정 위의 값이다.** 단정할 수 있는 것은 자릿수와 비율이다.

이 상태량이 KV cache와 갈리는 지점은 성장 방식이다. 표 16-5의 full-conversation prompting은 2.5K 토큰 대화에서 128+ M floats인데 Generative Adapter는 문맥 길이와 무관하게 32 M floats로 고정이다. 교차점은 이르다 — 위 두 점을 선형으로 이으면 약 600 토큰 대화 아래에서는 KV cache 쪽이 더 작다(이 추정은 이 책의 산술이다). 그 아래에서 이 기제는 **저장을 늘린다**.

문맥화 자체는 학습 단계가 아니라 prefill이다. hidden state를 얻으려면 동결 base LM의 full forward가 문맥 전체에 대해 필요하므로, FLOPs는 그 문맥을 한 번 prefill하는 것과 같은 자릿수다 — prompting-with-KV-cache가 어차피 지불하는 그 값이다. **절감은 첫 pass에 있지 않고 그 뒤의 모든 pass에 있다.** prefill 위에 얹히는 추가 작업은 chunk마다·layer마다 $1024 \times 1024$ 행렬의 rank-128 randomized SVD인데, $O(d_r^2 r)$로 layer당 chunk당 약 $1.3 \times 10^8$ MAC, 32 layer면 chunk당 약 $4 \times 10^9$이다. 1,024 토큰 chunk를 7B로 prefill하는 약 $1.4 \times 10^{13}$ FLOPs에 비하면 작다(이 대조도 이 책의 산술이다). 작지만 **GEMM이 아니고 배치가 잘 되지 않는 커널**이 깨끗한 prefill 파이프라인 한가운데에 들어온다.

**배칭이 이 설계의 다루어지지 않은 결과다.** merge된 형태면 동시 서빙되는 사용자마다 7B 가중치 사본이 따로 필요해지고 shared-weight batching이 죽는다 — ch01 Rosetta 표의 "batch로 weight 공유 → $\Theta$-경로에서 깨짐" 행이 그대로 발동한다. 분리된 delta 형태면 요청별 LoRA이므로 multi-LoRA 서빙 방식으로 배치되지만, 손으로 적합한 $r{=}16$ LoRA의 수 M floats이 아니라 요청당 32 M floats다. 논문은 동시 다중 사용자 서빙을 한 문장도 다루지 않는다 — "batch"는 학습 하이퍼파라미터에만 나온다.

여기서 §16.4가 예고한 실익이 드러난다. **롤백 단위가 다르다.** $\Theta$-경로였다면 잘못 쓴 내용을 되돌리는 방법은 이전 체크포인트로의 롤백뿐이다. 이 기제에서는 $S_0 = 0$으로 초기화하는 것이 전부이고, 비용은 문맥을 다시 접는 forward 한 번이다. 배포 아티팩트가 불변이므로 사용자별 delta가 오염되어도 base 모델은 감사 대상이 아니다. 이것이 산출물 모양이 아니라 시계로 분류했을 때 실제로 달라지는 운영 결론이다.

> **[평가]** 논문이 두 번 기대는 edge 논증은 edge 측정 없이 서 있다 — "most computations occur on edge devices without power GPUs" [Generative Adapter §4.3]. 그런 장치의 구속 조건은 보통 메모리 대역폭과 용량이고, 자라는 KV cache를 고정 64–128 MB adapter로 바꾸는 것은 실제로 좋은 교환이다. 그러나 논문이 재는 것은 FLOPs뿐이고, **대역폭 구속 장치에서 4× FLOPs 절감은 4× 지연 절감이 아니다.** 표 16-3의 $L_w$ 칸이 비어 있는 한 이 논증은 검증되지 않은 채로 남는다.

## 16.8 한계와 bridge-out

논문이 명시한 문제는 셋이고 하나로 모인다. "For future work, it would be interesting to further explore scaling up the adapter generator, such as by integrating adapters into additional layers, and to investigate more selective update rules (Schlag et al., 2021)" [Generative Adapter §7]. 뒤쪽 절반이 **논문이 자기 결손을 이름 부르는 자리**다 — Schlag et al. (2021)은 fast-weight programmer의 delta-rule 방식 쓰기·지우기로 인용되며, 그것이 정확히 식 (16-2)에 없는 $\beta_t$와 축출이다. 나머지 둘은 같은 공백의 재진술이다 — §3의 "For efficiency… defer the full exploration of other modules for future work"와 §5.1의 "Due to computational constraints, a more thorough exploration was not feasible".

논문이 남기지 않았지만 남는 문제는 여섯이다. **용량 포화** — $\beta_t = 1$과 고정 $d_r \times d_r$ 상태에서 품질은 스트림 길이에 대해 단조 감소할 수밖에 없고 표 16-4가 그것을 보이는데(51.5 → 32.0), 어디서 포화하는지도 $d_r$이나 $r$을 키우면 어떻게 되는지도 특성화되지 않았다. 둘은 1,024와 128로 한 번 정해지고 끝까지 변하지 않는다. **덮어쓰기와 낡음** — 이미 누적된 정보를 제거하거나 정정할 방법이 기제 안에 없고, 모순되거나 갱신된 사실을 제시하는 실험이 없다. **합성** — 같은 $S_t$에 들어간 두 문맥은 식 (16-4)에 의해 무차별 합산되고, 따로 생성된 두 adapter를 결합하는 실험은 없다. **순서** — 식 (16-4)가 토큰들의 합이므로 쓰기는 순열 불변이고, 문맥을 섞어도 같은 adapter가 나온다. 평가된 세 시나리오 중 둘(MSC의 다중 세션 대화, MetaICL의 입출력 시연)이 순서를 지니는데 순서 민감도를 재는 실험이 없다. **이 순열 불변성은 식 (16-4)로부터의 이 책의 추론이며 논문이 진술하지 않는다.** **generator의 이식성** — generator는 base LM마다 하나씩 학습되는데("the LM-specific adapter generator" [Generative Adapter §1]) base 모델이 갱신되면 살아남는지에 대한 언급이 없다. 배포하는 쪽의 실제 질문이다. **대용 지표의 타당성** — 설계 결정 전부가 §5.1의 보고되지 않은 상관 주장 위에 서 있다.

**경로 간 인용.** $W$-경로는 이름으로 광범위하게 인용된다 — §6이 "Fast Weights" 문단으로 열리고 Hinton & Plaut (1987), Ba et al. (2016), Schmidhuber (1992·1993), Schlag et al. (2021)(§6과 §7에서 두 번), Clark et al. (2022), 그리고 선형 attention 계열이 들어온다. 다만 계보가 **2022년에서 끊긴다** — 2024–2025년의 test-time training 라인은 없고, 이쪽은 연대가 강제한 침묵이다(2024년 11월 8일). $\Theta$-경로는 PEFT와 continual pretraining으로 인용된다(LoRA, Houlsby et al. 2019, prefix tuning, AdaLoRA, DoRA, Yang et al. 2024, Allen-Zhu & Li 2024, Hu et al. 2023, Tack et al. 2024). 그러나 **consolidation으로는 한 번도 인용되지 않는다** — EWC도 replay도 generative replay도 CLS도 생물학적 sleep도 참고문헌에 없다. $\Theta$-경로의 망각·통합 절반이 통째로 빠져 있다. $E$-경로는 **정확히 한 번**, 데이터셋 프로토콜로만 등장한다 — MSC 설정을 Packer et al. (2024)를 따라 잡는 문장 하나이며 [Generative Adapter §4.3], MemGPT는 방법으로 서술되지도 baseline으로 돌려지지도 않는다. RAG는 §1에서 한 번 언급되고 버려진다. sleep-time compute 문헌은 전무하고, 이 논문은 Letta STC보다 다섯 달 앞서므로 그 침묵은 **한 방향으로만** 연대가 강제한 것이다. Part III가 물어야 할 것은 역방향이다 — $E$-경로가 이 논문을 되받아 인용했는가.

> **[평가]** 인접하되 구별해야 할 계열이 하나 더 있다. Chevalier et al. (2023)·ICAE·UltraGist·Compressed Context Memory의 압축 라인은 이 논문에 조밀하게 인용되고 UltraGist가 유일한 강한 효율 baseline이다(표 16-5). 이 책의 프레임에서 이들은 $E$-경로가 **아니다** — 산출물이 읽기 시점에 소비되는 토큰 임베딩이므로 저장소에 쓰는 것이 아니라 프롬프트를 줄인다. 같은 표 안에서 경쟁하지만 층이 다르다. 그리고 가장 아쉬운 공백은 §16.6이 지목한 그것이다 — 같은 모양·같은 rank의 delta를 gradient로 적합한 baseline이 없으므로, **forward 쓰기가 gradient 쓰기 대비 품질을 얼마나 포기하는지 이 논문은 말할 수 없다.**

ch17이 받아 가는 것은 둘이다. 첫째, **접기를 wake 경로 밖으로 옮기는 이동.** 같은 물건, 다른 시계다 — 그리고 그 이동이 정확히 $W$-경로 논문을 sleep-time compute으로 바꾸는 조작이다. 이 장이 판별식 조건 (1)에서 탈락한 자리가 그대로 ch17의 출발점이 된다. 둘째, **$\beta_t = 1$과 축출 부재라는 결손**, 그리고 논문이 §7에서 이름으로 가리킨 그 수선책. 용량 천장은 ch18이, forward 쓰기가 gradient 쓰기에 내주는 것은 ch19가 받는다.

## 요약

- Generative Adapter는 문맥을 한 번의 forward로 파라미터화한다. 산출물은 LoRA 모양이고 논문이 그 형태를 직접 유도하지만 [Eq. 8–9], 갱신식은 (U-W)의 gradient-free 변종이고 시계는 chunk다.
- 경로 판정은 셋 다 밟아 나온다 — $\Theta$-경로 탈락(배포 체크포인트가 바이트 단위로 불변), $E$-경로 탈락($\mathrm{ret}$도 레코드도 없음), $W$-경로 통과. 소속 판정은 별개이며 조건 (1)에서 탈락한다: 접기가 wake 안에서 일어난다.
- **산출물의 모양으로 경로를 나누면 틀린다.** 이 장의 판정이 이 책의 방법론적 주장을 실증한다 — 모양은 저장·전송·롤백 단위를 정하고, 시계와 갱신식이 경로를 정한다.
- (U-W)의 항 중 넷이 비어 있다: $\alpha_t$ 없음, $\beta_t \equiv 1$, $\eta_t$ 없음, inner loss $\ell$ 없음. gradient 자리에 chunk Gram 행렬의 학습된 쌍선형 사영이 들어가고 $(k_t, v_t)$가 $(h_m, h_m)$으로 붕괴한다.
- 비용 4종 중 세 칸이 비어 있다. $L_w$·$\rho$는 논문에 없고 $B_s$는 그림에만 있으며, $C_{\text{cap}}$은 32M floats로 보고되되 dtype이 없어 바이트로 환산되지 않는다. 상각식 (A)는 닫히지 않는다 — 절감 1.554 TFLOPS/질의는 알지만 $C_{\text{sleep}}$이 수치로 없다.
- 초록의 헤드라인 수치가 본문 표와 화해하지 않는다. "from 19.5 to 31.5, 63.5% improvement"에서 31.5는 논문 어디에도 없고(표 5는 32.0), 19.5 → 31.5는 61.5%다.
- 품질 우위는 좁다. Mistral에서 in-context prompting을 이기는 칸이 열넷 중 셋이고, Llama2의 장문맥 우위는 baseline이 마지막 4K로 잘린 결과이며, MSC에서 전체 문맥 품질의 61%를 유지한다(40.2 대 66.0).
- 서빙 순이득은 문맥 길이와 무관한 고정 상태량과 base와 동일한 질의당 추론 FLOPs이고, 순비용은 요청당 32M floats·prefill 중간의 비-GEMM SVD·merge 형태에서의 shared-weight batching 상실이다.

## 자가 점검 체크리스트

- [ ] $\Theta$·$E$·$W$ 세 경로 조건을 이 논문에 각각 대고, 어느 조건이 어느 문장으로 탈락·통과하는지 원문 위치와 함께 말할 수 있다.
- [ ] 식 (16-2)–(16-3)을 표준형 (U-W)와 항별로 대조해 비어 있는 넷을 짚고, $\beta_t \equiv 1$이 $\rho$를 왜 측정 대상이 아니라 정의 대상 밖으로 밀어내는지 설명할 수 있다.
- [ ] 표 16-3의 세 빈칸이 각각 왜 비어 있는지 말하고, 이 논문에 대해 "효율적"이라고 쓸 수 없는 이유를 상각식 (A)의 두 항으로 설명할 수 있다.
- [ ] 초록의 31.5·63.5%와 표 16-4의 32.0·19.5의 관계를 재계산하고, 이 책이 어느 값을 본문에 쓰는지 말할 수 있다.
- [ ] 32M floats가 ch03 §03.6의 $|\Delta\Theta| = 2N_{\text{mod}}dr$과 $S_t$ 양쪽에서 같은 수로 나오는 것을 재계산하고, 그럼에도 바이트를 단정할 수 없는 이유를 말할 수 있다.
- [ ] 식 (16-4)의 순열 불변성을 유도하고, 그것이 MSC와 MetaICL에서 왜 위험한지, 그리고 그 진술이 논문의 주장이 아니라 이 책의 추론인 이유를 말할 수 있다.
- [ ] (Rosetta) 이 논문이 왜 "모델 재배포" 행의 물건을 만들면서 "KV cache" 행의 시계로 움직이는지 설명하고, 그 어긋남이 "batch로 weight 공유" 행에서 어떤 결과로 나타나는지 두 서빙 형태로 나누어 말할 수 있다.
