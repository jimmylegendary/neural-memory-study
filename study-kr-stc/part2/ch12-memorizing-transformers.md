# ch12. Memorizing Transformers — 외부 기억의 초기형

## 12.1 Bridge-in: 전작이 남긴 문제

ch10은 $E$층을 세웠다. 구현 형태를 text·vector·graph 셋으로 갈랐고, 읽기 비용을 세 항 — 검색 자체·prefill 증가·쓰기 — 으로 분해했으며(→ ch10 식 (10-5)), 상각이 지우는 항이 쓰기 하나뿐임을 확정했다. ch11은 그 문헌 전체에 판별식을 댔고 Memorizing Transformers(arXiv:2203.08913)를 조건 (1)과 (2)에서 탈락시켰다(→ ch11 §11.3). 그러므로 이 장은 sleep-time compute의 사례를 읽는 장이 아니라 **$B_s$ 축의 원점**을 읽는 장이다.

원점을 먼저 읽는 이유는 회계 때문이다. ch10의 RAG는 저장 전 추론 없이도 쓰기 비용을 corpus당 한 번 지불했다. 이 논문은 한 걸음 더 내려간다 — 쓰기의 산출물이 wake pass가 이미 만든 텐서 그 자체여서 한계 예산이 0이다. $E$-경로가 $B_s$를 한 푼도 쓰지 않고 얻을 수 있는 것의 상한이 여기서 정해지고, 이후 논문들이 지불하는 $B_s$는 이 상한 **위에서 얼마를 더 샀는가**로 평가되어야 한다.

**이 논문은 어떤 선행 논문의 open question도 인용해 받지 않는다.** Related Work([Memorizing Transformers §2])는 3인칭 현재시제의 위치 선정 서베이이고, 인용부호도 "X가 남긴 문제"도 없다. 모든 비교는 저자들이 스스로 세운 대조다 — "In contrast, we use a very large cache without compression" [Memorizing Transformers §2]. 이 책은 없는 계보를 지어내지 않는다(→ ch11 §11.7).

그래서 이 장의 bridge-in은 상속 진술이 아니라 **암묵적으로 답해진 세 질문**의 복원이다. 첫째, Transformer-XL과 Compressive Transformer가 단계 사이로 나르는 비미분 캐시는 어디까지 키울 수 있고, 나르는 상태를 반드시 압축해야 하는가. 답은 262,144쌍까지이고 압축은 불필요하다 — dense attention을 근사 top-$k_{nn}$ 회수로 바꾸면 된다 [Memorizing Transformers §2, Table 5]. 둘째, 긴 문맥 모델이 주어진 문맥을 실제로 쓰는가(Sun et al. 2021의 부정적 결과). 논문은 그 질문을 인용하지 않고 [Memorizing Transformers §4.6]을 그 답으로 지은 뒤 같은 결과를 방증으로 인용한다 [Memorizing Transformers §4.6]. 셋째, kNN-LM(Khandelwal et al. 2020)이 출력 softmax에 둔 비미분 저장소를 stack **안쪽**에 둘 수 있는가. 답은 그렇다이고, 자리는 중상위 attention layer 한 곳이며 학습된 per-head 게이트가 딸린다 [Memorizing Transformers §2, §3.1].

ch11이 이 장에 넘긴 질문은 하나였다 — 저장되는 것이 요약이 아니라 원본 텐서일 때 무엇이 얻어지고 무엇이 잃어지는가. 미리 적으면 이렇다. **얻는 것은 정확한 회수와 0에 가까운 쓰기 비용, 잃는 것은 어떤 체크포인트에나 붙일 수 있는 이식성이다.**

## 12.2 문제의식

논문의 첫 두 문장이 이 라인 전체의 대립 구도를 세운다.

> "Language models typically need to be trained or finetuned in order to acquire new knowledge, which involves updating their weights. We instead envision language models that can simply read and memorize new data at inference time, thus acquiring new knowledge immediately." [Memorizing Transformers Abstract]

이 책의 어휘로 옮기면 $\Theta$-경로와 $E$-경로의 대립이다. 그런데 2022년 3월의 이 논문에는 sleep도 consolidation도 replay도 없다. 대신 저자들이 쓰는 대비는 **저장 속도**다.

> "Facts and information which are stored in the form of weight matrices must be slowly trained over hundreds of thousands of training steps. By using attention, however, a model can simply memorize facts ... In this case, attention acts as a form of information retrieval." [Memorizing Transformers §1]

수사적 핵심은 attention과 information retrieval의 **동일시**다. 그것이 성립하면 문맥을 키우는 대신 저장소를 키우는 선택지가 열린다. 저장소를 키우려면 미분 가능성을 포기해야 한다는 것이 두 번째 주장이다.

> "gradients are not backpropagated into the external memory, which is critical to the scalability of our technique. ... if the external memory is not differentiable, then we can instead reuse keys and values that were previously computed on prior training steps, which drastically reduces the amount of computation for large memories." [Memorizing Transformers §1]

세 번째 주장은 압축의 불필요다. Compressive Transformer 계열이 나르는 상태를 요약해 줄이는 것과 정반대로, 이 논문은 "kNN lookup does not do averaging or summarization of tokens at long distances, but retrieves exact values"라고 쓰고 설계를 "a very large cache without compression"으로 요약한다 [Memorizing Transformers §1, §2].

> **[해설]** 세 주장을 이 책의 기호로 옮기면 ch11이 이 장에 넘긴 자리가 그대로 나온다. $S(c;B_s)$를 표현 공간의 **항등사상**으로 두고, 대신 $\mathrm{ret}(E,q)$를 모델 내부의 attention 연산 자체로 만든다. RAG도 $S(\cdot;B_s)$를 항등으로 두지만 $\mathrm{ret}$을 별도 encoder와 프롬프트 접합에 맡겼다(→ ch10 §10.2). 두 논문은 같은 자리에서 반대로 갈라진다 — RAG는 읽기를 모델 **밖**으로 밀어내 학습 가능한 일급 객체로 만들었고, 이 논문은 읽기를 모델 **안**으로 끌어들여 attention의 한 종류로 만들었다.

한 가지를 미리 못 박아 둔다. 논문은 첫 문장에서 $\Theta$-경로를 대립항으로 지목하면서 인용을 하나도 달지 않고, 145항의 참고문헌에 PEFT도 LoRA도 모델 편집도 catastrophic forgetting도 없다. $E$-경로는 **자기가 읽지 않은 문헌을 상대로 자기를 정의하며** 시작했다(§12.8).

## 12.3 Core mechanism (통일 표기)

이 절이 이 장의 두 정의를 소유한다 — 구조가 먼저이고 접합 방식이 다음이다.

> **정의.** **비미분 외부 기억을 attention이 읽는 구조**는, 저장소 $E$에 어떤 gradient도 흐르지 않고($\nabla_E \equiv 0$), $E$의 내용이 모델 자신이 이미 계산한 내부 표현이며, 읽기가 프롬프트 접합이 아니라 stack 내부의 특정 layer에서 residual stream으로 합류하는 구조를 말한다. 이 셋 중 하나라도 어긋나면 다른 구조다 — gradient가 흐르면 $\Theta$의 일부이고, 내용이 텍스트면 프롬프트 접합이며, 읽기가 입력단에서 일어나면 문맥 확장이다.

배치는 이렇다. 12층 decoder-only transformer($d{=}1024$, 8 heads × $d_{\text{head}}{=}128$, $d_{\text{ff}}{=}4096$)에서 layer 9 하나만 kNN-augmented layer이고 나머지 11개 층은 $E$를 보지 못한다 [Memorizing Transformers §3.1, §4.2]. 쓰기는 chunk 경계마다 일어난다.

$$
E_t \;=\; \mathrm{wr}\big(E_{t-1},\ \hat c_t\big),
\qquad
\hat c_t \;=\; S\big(c_t;\ B_s\big) \;=\; \big\{(k_i, v_i)\big\}_{i \in c_t},
\qquad
\mathrm{wr} = \text{append} + \text{FIFO evict to } n_E
\tag{12-1}
$$

식 (12-1)이 말하는 것은 (U-E)에서 $S(\cdot;B_s)$가 항등이라는 것이다. $c_t$는 forward pass가 방금 소비한 $C = 512$(또는 2048) 토큰의 chunk이고, $(k_i, v_i)$는 그 layer의 attention이 이미 물질화한 텐서 그대로다 — "after each training step, the (key, value) pairs in the local context are appended to the end of the external memory. If the document is very long, old (key, value) pairs will be dropped from the memory to make room for new ones" [Memorizing Transformers §3.1]. 여기서 시간 첨자가 $t$(chunk)이고 $k$(sleep 라운드)가 아니라는 점이 이 장 전체의 회계를 결정한다.

두 번째 정의가 읽기 쪽이다.

> **정의.** **kNN 검색을 attention에 접합하는 방식**은, attention의 질의 벡터 $q_t$를 그대로 검색 질의로 사용해 $E$에서 근사 top-$k_{nn}$쌍을 뽑고, 뽑힌 쌍에 대해 다시 softmax attention을 돌린 뒤, 그 출력과 local attention 출력을 학습된 per-head 스칼라 게이트로 섞어 residual stream에 되돌리는 접합을 말한다. 별도의 encoder도, 유사도 임계값도, 프롬프트 토큰도 개입하지 않는다.

$$
\mathrm{ret}(E, q_t) \;=\; \mathrm{Top}\text{-}k_{nn}\Big(\big\{\langle q_t, k_i\rangle : (k_i,v_i) \in E\big\}\Big),
\qquad k_{nn} = 32
\tag{12-2}
$$

$$
V_m \;=\; \mathrm{softmax}\big(q_t K_{nn}^{\top}\big)\, V_{nn},
\qquad (K_{nn}, V_{nn}) = \mathrm{ret}(E, q_t) \in \mathbb{R}^{k_{nn} \times d_{\text{head}}}
\tag{12-3}
$$

$$
g \;=\; \sigma(b_g), \quad b_g \in \Theta \ \text{(head당 스칼라 하나)},
\qquad
V_a \;=\; V_m \odot g \;+\; V_c \odot (1-g)
\tag{12-4}
$$

식 (12-2)–(12-4)는 원문 [Memorizing Transformers §3.1]의 산문 서술과 Eq. (1)–(2)에 대응한다. (12-2)의 요지는 검색의 키 공간이 모델 자신의 attention 키 공간이라는 것이다 — "The same queries are used for both the local context, and for the external memory. The keys and values also belong to the same distribution" [Memorizing Transformers §3.1]. 즉 $\mathrm{ret}$은 attention이 어차피 계산하는 내적의 argmax일 뿐이다. (12-3)의 $K_{nn}, V_{nn}$은 회수된 $k_{nn}$개 쌍을 쌓은 행렬이고 여기에는 position bias를 붙이지 않으며, (12-4)의 $V_c$는 local chunk와 Transformer-XL 캐시에 대한 통상의 dense attention 출력이며 $V_a$가 residual stream으로 올라간다.

읽기 게이트 $g$는 **내용에 의존하지 않는다** — "the value of the gate $g$ does not depend on the content of the token at each position, although that would be a trivial extension to implement" [Memorizing Transformers §3.1]. test-time의 라우팅 결정이 아니라 outer loop에서 한 번 학습되는 head별 구조적 선호다. 그리고 학습이 끝나면 "most heads learned to attend almost exclusively to external memory" [Memorizing Transformers §3.1].

전체를 표준형 (R)로 환원하면 이렇다.

$$
\hat y \;=\; f\big(q;\ \Theta,\ \varnothing,\ \mathrm{ret}(E, q)\big),
\qquad \mathrm{ret}\text{는 정확히 한 layer에서만 진입}
\tag{12-5}
$$

$W = \varnothing$이다. inner loss도 test-time gradient도 고정 크기 압축 상태도 없다. Transformer-XL 캐시는 무손실 KV 이월이지 $W$가 아니다 — $\alpha_t$도 $S_t$도 $\eta_t$도 없다.

논문 자신이 (12-5)의 결과에 놀란다. "This is surprising, because the external memory is not differentiable, and is added only to one layer of the Transformer ... We conclude that the lower layers of a Transformer don't necessarily need long-range context, and having a differentiable memory is not as important as one might suspect." [Memorizing Transformers §4.3]

훈련 중에만 나타나는 항이 하나 있다. $E$에 담긴 $(k_i,v_i)$는 **더 오래된 $\Theta$**가 만든 것이라 질의와 분포가 어긋나고, 논문은 이를 staleness라 부르며 key/query 정규화로 대응한다 — "Normalization does not eliminate staleness, but it at least ensures that older keys and newer keys do not differ in magnitude" [Memorizing Transformers §3.2]. $\Theta$가 동결된 순수 추론에서는 이 항이 사라진다. 훈련 시점과 배포 시점의 거동이 갈리는 유일한 자리다.

### 12.3.1 표기 대응표

표 12-1 — Memorizing Transformers 원 표기 → 이 책의 표기

| 원 표기 | 이 책의 표기 | 주의 |
|---|---|---|
| $M$ — memory size, head당 $(k,v)$ 쌍 수 | $n_E$ (개수), $C_{\text{cap}}$ (바이트) | 맨 $M$ 금지 — $\mathcal{M}(\cdot;W)$(읽기 함수)로 읽힌다 |
| $k$ — 회수 이웃 수, $k=32$ [§4.2] | $k_{nn}$ | 이중 충돌: $k$는 sleep 라운드 첨자, $k_t$는 key 벡터 |
| $g$ — per-head 게이트, $g=\sigma(b_g)$ [Eq. (1)] | $g$ ("읽기 게이트 $g$") | $\alpha_t$(쓰기 시점 retention)와 혼동 금지. $b_g$는 $\Theta$에 속한다 |
| "context" / "context size" = 512 또는 2048 | $C$ (chunk) | 용량은 언제나 $C_{\text{cap}}$. Table 4의 Context 열이 $C$, Memory 열이 $n_E$ |
| "external memory" | $E$ (표현 수준 $E$) | MemGPT·Mem0의 $E$와 같은 단어, 다른 물질 — "활성값을 담는 $E$" 대 "텍스트를 담는 $E$" |
| $V_m,\ V_c,\ V_a$ — memory·local·결합 출력 | 그대로 유지 | 저장된 값 벡터는 $v_i$. $V_i$로 쓰지 않는다 |
| $\Delta_i$ — $\text{CE}_{8192}(x_i) - \text{CE}_{32\mathrm{K}}(x_i)$ [§4.6] | $\Delta_i$ | 부호 관행: 양수 = 큰 $n_E$가 개선 |
| "training step" — $E$를 쓰는 단위 | chunk 경계, 첨자 $t$ | 학습 사건이 아니다. 추론에서도 동일하게 일어나며, 원문이 그렇게 쓴 이유는 모든 측정이 훈련 측정이기 때문이다 |
| (장-국소) ch10의 $K_{\text{ret}}$ | $k_{nn}$ | 같은 역할, 다른 단위 — ch10은 텍스트 레코드, 여기서는 $(k,v)$ 쌍 |
| (장-국소) 회수된 키·값 행렬 | $K_{nn},\ V_{nn}$ | $K_{\text{ret}}$을 재사용하지 않는다 — ch10에서 그 기호는 회수 **개수**(스칼라)다 |

## 12.4 어느 층을 언제 쓰는가

**test-time에 학습되는 것은 아무것도 없다.** 움직이는 것은 $E$뿐이고 그것도 gradient가 아니라 memcpy로 움직인다. 게이트를 포함한 모든 학습 객체가 배포 전 pre-training에서 고정된다 — corpus 전체에서 가장 깨끗한 분리다.

표 12-2 — 이 값은 누가 학습하는가

| 객체 | 학습 주체 | 언제 | test-time 이동 | 출처 |
|---|---|---|---|---|
| transformer weights (embedding, 12층 attention+FFN, 출력 투영) | outer loop | 500K steps, Adafactor, lr 1.0, 32 TPU cores | 아니오 | §4.2 |
| T5 relative position bias | outer loop | $\Theta$와 함께 | 아니오. local dense attention에만 적용, $E$에서 회수한 $K_{nn}$에는 미적용 | §3.1 |
| 게이트 바이어스 $b_g$ (head당 스칼라 1개) | outer loop | $\Theta$와 함께. 대부분의 head에서 1 근처로 수렴 | 아니오. 내용 비의존이고, 내용 의존화는 "trivial extension"이라 부르며 실험하지 않는다 | §3.1 Eq. (1) |
| $E_{knn}$의 내용 — $(k,v)$ 쌍 | 없음. gradient가 흐른 적이 없다 | chunk 경계마다 축자 append, 용량 초과 시 FIFO 축출, 문서 경계마다 소거 | **예 — 유일하게 움직이는 상태** | §1, §3.1 |
| $E_{xl}$의 내용 — Transformer-XL 캐시 | 없음 | 매 step 직전 chunk의 $(k,v)$로 교체, **모든** layer에서 | 예 | §3 |
| 근사 kNN 색인 | 없음. recall 약 90%의 고정 근사 | 해당 없음 | $E$ 위의 자료구조로서 예 | §3.3 |
| kNN layer의 위치 (12층 중 9층) | 없음 — 하이퍼파라미터 | 훈련 전 고정 | 아니오 | §4.2, App. A.1 |
| $n_E$ 자체 | 없음. 그러나 **훈련 후에 바꿀 수 있는 유일한 값**이다 | 추론에서 올리거나 20K step 추가 finetune | 배포 시점 knob — 이 논문의 실질 기여 | §1, §4.5, Table 5 |

실행되는 갱신 규칙은 (U-E) 하나뿐이다. $\Theta$는 배포 전 outer loop에서만 움직이고 $W$는 존재하지 않으므로 순서 문제가 생기지 않는다. 시계는 chunk 경계 $t$이며 sleep 라운드 $k$가 아니다 — 쓰기는 유휴 시간이 아니라 wake pass 안에서 동기적으로 일어나고, 별도의 오프라인 job도 쓰기 배칭도 없다. 이것이 ch11의 조건 (1)·(2)가 걸린 자리다.

$\mathrm{wr}$은 append와 FIFO 축출뿐이다. merge도 rewrite도 중복 제거도 중요도 점수도 없다 — MemGPT의 working-context 치환(→ ch13)이나 Mem0의 ADD/UPDATE/DELETE(→ ch15)에 해당하는 것이 없다. 그래서 $E$는 창 안에서 단조 증가하고 창 밖에서 완전 망각이며, 어떤 실험도 "축출되었다"와 "top-$k_{nn}$ 경쟁에서 졌다"를 구분하지 않는다. 범위는 문서당·batch 원소당이고 문서 경계에서 소거되므로 [Memorizing Transformers §3.1] 상각은 **문서 내부에서만** 성립한다. 식 (A)의 $N_q$는 여기서 같은 문서에 남은 토큰 수다.

> **[해설]** ch01의 Rosetta 표는 KV cache를 $W$의 유비로 놓았다(→ ch01). 이 논문은 그 유비가 **문자 그대로 성립하는** 유일한 자리이되 층이 다르다 — 저장되는 것이 실제 $(k,v)$이고 창 안에서 append-only 무손실이지만, 놓이는 층은 $W$가 아니라 $E$다. 뒤집어 말하면 이 $E$는 뒤의 $E$-경로 논문들과 달리 **읽기 함수를 바꾸지 않는다.** 읽는 연산은 여전히 softmax attention이고, 바뀐 것은 그 대상뿐이다.

## 12.5 비용 4종

이 논문은 서빙 수치를 하나도 보고하지 않는다. 유일한 벽시계 측정은 가장 작은 모델의 **훈련** step 시간이며 한 문장으로 한 번 나온다.

표 12-3 — 비용 4종

| 기호 | 값 | 근거 |
|---|---|---|
| $B_s$ | **논문에 없음**. 구조적으로 0이다 — 오프라인 pass가 없으므로 분리 가능한 sleep 예산이 성립하지 않는다. 쓰기의 한계 비용은 memcpy와 색인 삽입뿐이고 논문은 측정하지 않는다 | §3.1, §1 |
| $L_w$ | **논문에 없음**. 추론 지연도 TTFT도 ITL도 decode 측정도 없고 "latency"라는 단어가 나오지 않는다. 131K·262K가 "maintaining a reasonable step time"이라는 진술은 정성적이며 "reasonable"에 숫자가 없다 | §1, §4.2 |
| $C_{\text{cap}}$ | **개수로만 보고. 바이트는 논문에 없음.** $n_E \in \{1536,\ 8192,\ 65\mathrm{K},\ 131\mathrm{K},\ 262\mathrm{K}\}$, 범위는 head당(8 heads)·한 layer·문서당·batch 원소당, $d_{\text{head}}{=}128$. 논문은 이 값들을 곱한 적이 없고 dtype도 적지 않는다 | §3.1, §4.2, Table 4·5 |
| $\rho$ | **수치로는 논문에 없음.** 열화 기제는 staleness 하나이고 그것은 **훈련 시점** 아티팩트다. 정량화되지 않았다 | §3.2, §4.5 |

논문에 있는 유일한 계산량 수치는 다음이다. "the step time increased from 0.2s to 0.25s when we added a memory of size 8K, and to 0.6s when we added a memory of size 65K (measured on TPUv3)" [Memorizing Transformers §4.2]. $n_E{=}8192$에서 +25%, $n_E{=}65536$에서 3배다[본서 산술]. 이 값은 kNN 검색·추가 attention·쓰기를 **한 덩어리로** 묶고 있고, 추론 step이 아니라 훈련 step이며, 약 200M 파라미터에서만 보고되었다. 셋 중 하나만 어겨도 서빙 감각으로 옮길 수 없다[본서 판단].

$\rho$ 칸은 hedge가 다섯 겹이다. "In some of our experiments", "sometimes resulted in worse performance", "could be due to staleness" [Memorizing Transformers §3.2], 다시 "In some cases, training was unstable ... possibly due to distributional shift early in the training" [Memorizing Transformers §4.5]. 불안정했다는 실행은 논문 어디에도 그려지거나 표로 실리지 않았다. $\Theta$가 배포 후 움직이지 않으므로 (U-$\Theta$) 의미의 라운드 간 망각은 애초에 적용되지 않고, FIFO 축출로 인한 손실은 한 번도 분리되지 않았다.

> **[평가]** **상각식은 여기서 붕괴한다.** 식 (A)의 $C_{\text{sleep}}$이 0이므로 $C_{\text{avg}} = C_{\text{wake}}$이고 손익분기 $N_q^\ast$를 계산할 대상이 없다 — $E$ 쓰기는 $N_q = 1$에서 이미 자기 값을 한다. corpus에서 상각식이 정당하게 붕괴하는 유일한 논문이다.

> **[평가]** 그 대신 이 논문은 corpus의 어떤 $E$-경로 논문도 갖지 못한 것을 갖고 있다 — $N_q$가 뽑혀 나올 **분포**다. 문서 길이 히스토그램이 다섯 corpus 전부에 실려 있고 arXiv 논문은 최대 거의 1.6M 토큰, Github 저장소는 최대 900만 토큰을 조금 넘는다 [Memorizing Transformers App. A]. 이 논문에는 상각할 것이 없는데 분포는 있고, 상각을 주장하는 뒤의 논문들에는 분포가 없다.

읽기 쪽 비용은 측정이 아니라 **인용으로 주장된다**. 확장 가능한 구현으로 ScaNN과 Faiss가 지목되고 근사 kNN이 "can scale into the billions"라고 적히지만 [Memorizing Transformers §1, §3.3], FLOP 수도 색인 빌드 시간도 gather 대역폭도 없다. 중심 효율 논증이 측정되지 않은 항 위에 서 있다[본서 판단].

## 12.6 실험과 스케일

**이 논문의 유일한 정량 지표는 token 단위 평균 perplexity다.** downstream 과제가 하나도 없다 — QA도 code completion pass@k도 요약도 긴 문맥 벤치마크도 없다. 초록의 능력 주장("capable of making use of newly defined functions and theorems during test time")을 떠받치는 것은 Isabelle에서 손으로 확인한 10개 예시이고 그중 8개를 저자들이 성공으로 판정했다 [Memorizing Transformers §4.6].

표 12-4 — 500K step 훈련 후 평균 token perplexity [Memorizing Transformers Table 4]

| $C$ | $n_E$ | XL 캐시 | arXiv | PG19 | C4(4K+) | GitHub | Isabelle |
|---|---|---|---|---|---|---|---|
| 512 | 없음 | 없음 | 3.29 | 13.71 | 17.20 | 3.05 | 3.09 |
| 2048 | 없음 | 없음 | 2.69 | 12.37 | 14.81 | 2.22 | 2.39 |
| 512 | 없음 | 512 | 2.67 | 12.34 | 15.38 | 2.26 | 2.46 |
| 2048 | 없음 | 2048 | 2.42 | 11.88 | 14.03 | 2.10 | 2.16 |
| 512 | 1536 | 없음 | 2.61 | 12.50 | 14.97 | 2.20 | 2.33 |
| 512 | 8192 | 없음 | 2.49 | 12.29 | 14.42 | 2.09 | 2.19 |
| 512 | 8192 | 512 | 2.37 | 11.93 | 14.04 | 2.03 | 2.08 |
| 512 | 65K | 512 | 2.31 | 11.62 | 14.04 | 1.87 | 2.06 |
| 2048 | 8192 | 2048 | 2.33 | 11.84 | 13.80 | 1.98 | 2.06 |
| 2048 | 65K | 2048 | 2.26 | 11.37 | 13.64 | 1.80 | 1.99 |

논문이 표 12-4에서 뽑아 쓰는 주장은 셋이다. C4(4K+)에서 $n_E{=}8192$을 붙이면 vanilla가 17.20 → 14.42, Transformer-XL이 15.38 → 14.04가 된다. 다섯 열 전부에서 최소값이 마지막 행이다[본서 관찰]. 그리고 "using even a small external memory of size 1536 provides a gain in perplexity which is almost as good as using a local context of size 2048 but no memory" [Memorizing Transformers §4.3]. **셋째 주장은 다섯 열 중 둘에서 뒤집힌다** — $n_E{=}1536$ 행과 $C{=}2048$ 행을 대조하면 arXiv(2.61 대 2.69)·GitHub(2.20 대 2.22)·Isabelle(2.33 대 2.39)에서는 성립하지만 PG19(12.50 대 12.37)와 C4(4K+)(14.97 대 14.81)에서는 진다[본서 관찰]. 논문은 이 열별 분해 없이 주장만 적는다.

용량 사다리는 Table 5가 잇는다. 500K step 사전학습 뒤 20K step만 더 돌려 $n_E$를 올리면 arXiv perplexity가 $C{=}2048$·사전학습 65K 기준 65K에서 2.26, 131K에서 2.23, 262K에서 2.21까지 내려가고, 논문은 상한을 문서 길이로 귀속시킨다 [Memorizing Transformers §4.5, Table 5]. 같은 표에 [Memorizing Transformers §3.2]의 불안정성 서사와 반대 방향의 행이 있다: $C{=}512$에서 65K로 직접 훈련하면 2.31, 8192로 사전학습 뒤 65K로 finetune하면 2.32다[본서 관찰]. 두 프로토콜을 같은 용량에서 비교한 유일한 자리에서 직접 훈련이 낫다[본서 관찰].

스케일 주장 — $n_E{=}8192$을 붙인 작은 모델이 파라미터 5배인 vanilla와 맞먹는다 — 은 Figure 1 하나에만 근거한다 [Memorizing Transformers §4.4, §5]. 1B와 8B의 perplexity 값이 논문 어디에도 없고, 데이터셋 arXiv Math 하나, $n_E{=}8192$ 하나, $C{=}2048$, XL 캐시 없음, 오차 막대 없음이다. 소급 적용 주장도 Figure 6 하나에 근거한다: $E$ 없이 사전학습된 1B 모델을 20K step(사전학습의 4%) finetune하면 격차의 85%가, 100K step이면 전부 닫힌다 [Memorizing Transformers §4.5].

부록의 ablation은 전부 한 점에서 돌았다 — $C{=}512$, XL 캐시 512, $n_E{=}8192$, 약 200M 파라미터, 데이터셋 1개[본서 관찰]. layer 위치는 3층 2.40, 6층 2.36, 9층 2.37, 12층 2.43이고 [Memorizing Transformers Table 14], 논문 전체가 쓰는 기본값 9층은 자기 스윕의 최고값이 아니다[본서 관찰]. 이웃 수는 32에서 2.38, 128·256에서 2.37이다 [Table 15]. 시드 표준편차는 Transformer-XL 2.67 ± 0.01, Memorizing Transformer 2.37 ± 0.005이며 [Table 16], 이것이 유일한 분산 추정이고 Figure 1·Figure 6에도 오차 막대가 없다[본서 관찰]. 즉 layer 스윕은 전체 폭 0.07을 점당 1회 실행으로 가르고, Table 15의 2.38은 Table 4가 2.37로 보고한 바로 그 구성이다 — 시드 폭 기준 약 2σ이며 논문은 언급하지 않는다[본서 산술]. 두 정성 연구도 **다른 아키텍처**에서 돌았다: $\Delta_i$ 분석은 게이트 없는 구버전을, Isabelle 사례 연구는 부록이 이득 없다고 보고한 2-layer 32K 구성을 썼다 [Memorizing Transformers §4.6, App. A.1].

> **[평가]** 다섯 corpus 중 셋을 저자들이 만들었고, 그중 둘은 측정 대상인 장거리 의존성을 **제조하는** 방식으로 만들어졌다. Github은 저장소 하나를 무작위 디렉터리 순회로 이어 붙이고, Isabelle은 theory 파일을 "ordered according to their import dependencies, so that later files use sub-theorems that are proved in earlier files"로 이어 붙인다 — 정의가 참조보다 반드시 앞선다 [Memorizing Transformers §4.1]. 표 12-4에서 가장 큰 이득이 나오는 열이 정확히 그 둘이다(GitHub 3.05 → 1.80, Isabelle 3.09 → 1.99). Table 4의 캡션은 모든 모델이 500K step 훈련되었다고 적지만 Isabelle은 과적합 때문에 100K step에서 멈췄다 [Memorizing Transformers §4.2]. 회수 baseline은 한 번도 돌지 않았다 — kNN-LM, Compressive Transformer, Big Bird가 [Memorizing Transformers §2]에서 논의되고 아무것도 재현되지 않으며, 유일한 대조군은 저자 자신의 vanilla와 Transformer-XL이다. 근사 회수는 recall 약 90%에서 돌았고, 더 나쁜 근사에도 견고하다는 주장에 수치도 구현체 이름도 없다 [Memorizing Transformers §3.3, §4.2]. 이 장은 위 어느 것도 개별 결함으로 세지 않는다. 세는 것은 **결론의 폭**이다 — 실증된 것은 "저자들이 만든 장거리 의존 corpus에서, 저자들의 baseline 대비, 200M 규모에서 perplexity가 내려간다"이며 초록이 주장하는 능력은 그보다 넓다.

$\Delta_i$ 분석이 이득의 실제 모양을 보여 준다. 22K 토큰짜리 arXiv 논문에서 $n_E{=}8192$과 $n_E{=}32\mathrm{K}$를 비교하면 앞의 8192 토큰 구간에서 차이가 정확히 0이고, 그 뒤로도 "the benefit of external memory is somewhat sparse ... mainly driven by a small percentage of tokens"이며 일부 토큰은 오히려 **나빠진다** [Memorizing Transformers §4.6]. 논문은 원인을 top-$k_{nn}$ 탈락으로 추정하지만 축출과 혼잡을 가르는 실험은 없다. 이득이 몰리는 자리는 희귀 토큰이다 — 고유명사, 참조, 인용, 함수 이름.

## 12.7 Systems/serving 함의

서빙 감각으로 직접 옮겨지는 문장은 하나다. "Unlike standard dense attention, the retrieved memories contain a different set of (key, value) pairs for each query." [Memorizing Transformers §3.1] $K$와 $V$가 질의들 사이에서 공유되지 않는다는 뜻이고, 그러면 그 layer는 GEMM이기를 그만두고 **gather**가 된다(나머지 11개 층은 그대로 GEMM이다). 프롬프트 접합형 $E$-경로와의 결정적 차이다 — ch10의 항 ②(prefill 증가)가 여기서는 0이고, 대신 토큰당 gather 트래픽 항이 새로 생긴다.

> **[해설]** 바이트로 환산해 본다. **아래 계산은 이 책의 산술이며 논문의 보고값이 아니다.** 논문이 dtype을 적지 않으므로 bf16(2 bytes/elem)을 가정하고 나머지는 논문 자신의 shape다(8 heads × $d_{\text{head}}{=}128$, key·value 양쪽 저장, kNN layer 1개). $n_E{=}65{,}536$이면 $65536 \times 8 \times 128 \times 2 \times 2 = 268{,}435{,}456$ bytes = 256 MiB, 같은 방식으로 1536 → 6 MiB, 8192 → 32 MiB, 262,144 → 1 GiB이고 전부 **문서 하나당**이다. 검산: 같은 토큰 수의 12층 KV cache 전체가 3 GiB이므로 한 층에만 사는 저장소가 그 12분의 1인 것이 맞다. 절대치는 dtype 가정에 걸려 있으니 하한·방향성으로 읽고, 단정하는 것은 자릿수와 비율이다.

집계하면 문제가 보인다. $C{=}512$에서 batch가 문서 256개이고 [Memorizing Transformers §4.2] batch 원소마다 별도의 $E$가 필요하므로, 곱하면 $n_E{=}8192$에서 8 GiB, $n_E{=}65{,}536$에서 64 GiB다[본서 산술]. 논문은 두 값을 나란히 적어 놓고 한 번도 곱하지 않는다.

> **[평가]** 용량이 batch와 곱해진다는 것이 이 설계의 서빙 질문 전부다. 논문은 그것을 아키텍처 사실로만 진술하고 회계하지 않는다 — batch 원소마다 별도의 $E$가 필요하다는 문장과 batch 크기를 적은 문장 사이에 곱셈이 없다 [Memorizing Transformers §3.1, §4.2].

> **[평가]** gather 트래픽에는 정확한 손익분기가 있다. 회수 후 gather는 kNN layer 하나에서 토큰당 $k_{nn} \times d_{\text{head}} \times 2 \times 2 \times 8 = 131{,}072$ bytes = 128 KiB를 옮기고 이 값은 $n_E$와 무관한 반면, 저장소 전체를 dense scan하면 $n_E \times 8 \times 128 \times 2 \times 2$ bytes를 옮긴다. 둘이 같아지는 지점은 $n_E = C \cdot k_{nn} = 512 \times 32 = 16{,}384$쌍이다. 그보다 작으면 **회수가 전부 훑기보다 더 많은 바이트를 읽고**, 크면 회수가 이기며 이득이 선형으로 커진다. 논문의 1536과 8192는 둘 다 손익분기 아래다. 이 항등식은 shape에서 정확히 따라 나오지만, §12.5의 step-time 삼중값은 검색·attention·쓰기를 묶은 훈련 측정이므로 그 증거가 되지 못한다.

> **[해설]** 독자의 세계로 옮기면 둘이 남는다. 첫째, **shared-weight batching이 온전하다.** 사용자별로 $\Theta$에 쓰이는 것이 없으므로 요청들은 가중치 사본 하나를 공유하고, 요청별 상태는 KV cache처럼 거동한다 — 문서 시작에 할당, 상한까지 성장, 문서 끝에 해제. Rosetta 표에서 이 논문에 대응하는 항목은 "모델 재배포"가 아니라 "KV cache"다. 논문 스스로 그 대비를 윤리 절에 적으며 사용자 지식의 $O(1)$ 삭제를 아키텍처 속성으로 내세운다 [Memorizing Transformers Ethics]. 둘째, 접근 패턴이 직관과 반대다. 문서당 256 MiB는 SRAM이 아니라 HBM에 내려앉고, 논문이 가리키는 10억 규모에서는 host DRAM이나 디스크의 ANN 색인 뒤로 간다 [Memorizing Transformers §3.3]. 대부분이 차가운 큰 영역에 대한 무작위 gather이고, decode에서 익숙한 streaming KV cache의 정반대다.

decode 제약도 하나 명시해 둔다. 쓰기가 chunk 경계에서만 일어나므로 생성 중인 모델은 **현재 chunk 안에서 자기가 만든 토큰을 회수할 수 없다** — 최대 512(또는 2048) 토큰이 chunk가 닫힐 때까지 $E$에 보이지 않는다[본서 추론]. 논문은 생성을 돌리지 않아 표면화되지 않지만, 이 기제를 decode 경로에 놓는 어떤 설계에도 걸린다[본서 추론].

## 12.8 한계와 bridge-out

논문이 남긴 문제는 다섯이다. 거대한 $E$를 무엇으로 채울 것인가 — "How to make the best use of this capability is a topic for future work" [Memorizing Transformers §5]. 읽기 게이트를 내용 의존으로 만들 것인가(명명만 하고 실행하지 않음). staleness가 실제로 불안정성의 원인인가. $n_E$의 수익 체감점을 문서 길이가 정하는가 top-$k_{nn}$ 혼잡이 정하는가 — [Memorizing Transformers §4.5]는 전자로 [Memorizing Transformers §4.6]은 후자로 귀속시키고 화해시키지 않는다. 왜 하위 층은 장거리 문맥이 필요 없는가 — 관찰 하나에서 결론으로 곧장 간다.

여섯 번째가 이 책에 가장 중요하다. 윤리 절이 삭제 가능성을 아키텍처 요구사항으로 제기한다 — "The same is not true of differentiable model parameters" [Memorizing Transformers Ethics]. 2022년에 미리 적힌 **$\Theta$-경로에 대한 반론**이며 ch19–ch21이 통과해야 할 관문이다. 반대로 한 번도 묻지 않은 것도 있다: $E$와 $\Theta$를 **함께** 쓸 수 있는가. 논문은 둘을 대안으로만 세운다.

**경로 간 인용.** $E$-경로의 전(前)-에이전트 회수 계보는 읽는다 — kNN-LM, Yogatama et al. 2021, REALM, MARGE, RAG, Faiss, ScaNN [Memorizing Transformers §2]. $W$-경로는 전무하다: fast weight도 Hebbian도 meta-learning도 test-time training도 없다. 이 corpus의 $W$-경로 논문은 전부 2022년 3월 이후이므로 그쪽은 연대가 강제한 침묵이지만, 2022년 이전 fast-weight 문헌의 부재는 선택이다(감사의 글은 Imanol Schlag의 교정을 사례하면서 참고문헌에는 그 계보를 넣지 않는다 — 이 병기는 이 책의 주석이다). $\Theta$-경로도 전무하고 이쪽은 명백히 선택이다: LoRA는 2021년 6월부터, ROME은 2022년 2월부터 있었다. 생물학·CLS도 전무하다 — 이중 시간척도 기억 시스템을 지으면서 CLS 어휘를 한 번도 쓰지 않는다.

> **[평가]** 침묵은 양방향이다. 뒤의 $E$-경로도 이 논문을 인용하지 않는다 — MemGPT·Mem0·Zep·ReasoningBank·Letta STC 어느 쪽도 참조하지 않으며 corpus에서 이 논문을 인용하는 파일은 둘뿐이다. 그중 하나가 압축 주장을 정면으로 기각한다: 이런 방법들은 "sparsify retrieval but not storage. The full KV still lives somewhere, so the true rate(Z) is unchanged and the compression is of compute, not memory" [Rate–Distortion Memory Compaction, arXiv:2607.08032 — 이 논문 자신의 진술이 아니다. → ch22]. 이 책의 판단은 그 기각이 옳다는 것이다. §12.7의 문서당 256 MiB가 표 12-3의 빈 $C_{\text{cap}}$ 칸을 채운 값이며, **표현 수준 $E$는 계산을 압축하고 저장을 압축하지 않는다.**

ch13이 받아 가는 것은 둘이다. $B_s$ 축의 원점 — 쓰기에 아무것도 쓰지 않고 얻어지는 것의 상한 — 과, 프롬프트 접합이 아닌 읽기의 손익 회계다. ch13은 같은 $E$를 텍스트로 되돌린다. 그때 항 ②(prefill 증가)가 되살아나고, 대신 어떤 체크포인트에나 붙는 이식성이 생기며, $S(c;B_s)$가 항등에서 떨어져 나가기 시작한다.

## 요약

- Memorizing Transformers는 sleep-time compute이 아니다. 쓰기가 wake pass 안 chunk 경계에서 동기적으로 일어나고 한계 예산이 0이므로 조건 (1)·(2)에 걸린다(→ ch11). 이 장은 그것을 $B_s$ 축의 **원점**으로 읽는다.
- 갱신 규칙은 (U-E) 하나뿐이고 $S(\cdot;B_s)$는 표현 공간의 항등사상이다. $\hat c_t$는 forward pass가 이미 만든 $(k_i,v_i)$ 그대로이며, $W$는 없고 $\Theta$는 배포 후 움직이지 않는다.
- 읽기는 프롬프트 접합이 아니라 12층 중 layer 9에서의 top-$k_{nn}$($k_{nn}{=}32$, recall 약 90%) 회수 후 attention이고, local 출력과 내용 비의존 per-head 게이트로 섞인다.
- 비용 4종 중 값이 든 칸이 없다. $B_s$·$L_w$·$\rho$는 논문에 없고($B_s$는 구조적으로 0, $\rho$는 이름만), $C_{\text{cap}}$은 개수로만 보고되며 바이트가 없다. 유일한 벽시계 값은 200M 모델의 훈련 step 시간 0.2/0.25/0.6초다.
- 실증 범위는 좁다. 지표는 perplexity 하나, 다섯 corpus 중 셋이 저자 제작이고 그중 둘은 장거리 의존성을 제조하는 방식으로 이어 붙였으며, 회수 baseline은 없고 헤드라인 스케일 주장에는 표도 오차 막대도 없다.
- 서빙 순이득은 shared-weight batching 유지와 $O(1)$ 삭제이고, 순비용은 한 층의 GEMM→gather 전환과 batch와 곱해지는 문서당 상태다(bf16 가정 시 $n_E{=}65{,}536$에서 문서당 256 MiB).
- 상속도 계승도 없다. 뒤의 $E$-경로는 이 논문을 인용하지 않고, 인용하는 한 편은 압축 주장을 기각한다 — 이 기제는 저장이 아니라 계산을 압축한다.

## 자가 점검 체크리스트

- [ ] 이 논문이 조건 (1)·(2)에 걸리는 이유를 갱신식과 시계로 설명하고, 그럼에도 $E$-경로의 조상으로 읽는 이유를 말할 수 있다.
- [ ] 식 (12-1)–(12-5)를 (R)·(U-E)와의 차이로 진술하고, $S(\cdot;B_s)$가 항등일 때 상각식 (A)가 왜 붕괴하는지 설명할 수 있다.
- [ ] 표 12-3의 네 칸이 각각 왜 비어 있는지 근거와 함께 말하고, 유일한 벽시계 수치를 서빙 지연으로 옮기면 안 되는 이유를 셋 댈 수 있다.
- [ ] $n_E = C \cdot k_{nn}$ 손익분기를 유도하고, 그 아래에서 회수가 전부 훑기보다 더 많은 바이트를 읽는다는 결론을 자기 shape로 재계산할 수 있다.
- [ ] Rosetta: 이 $E$가 왜 "모델 재배포"가 아니라 "KV cache" 행에 대응하는지, 그 대응이 ch01의 "KV cache ↔ $W$" 행과 어디서 갈라지는지 설명할 수 있다.
