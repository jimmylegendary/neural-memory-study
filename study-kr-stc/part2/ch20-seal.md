# ch20. SEAL — 스스로 학습 데이터를 만든다

## 20.1 Bridge-in: 전작이 남긴 문제

ch19는 $\Theta$에 쓰는 도구를 손에 쥐여 주었다. LoRA는 갱신을 저랭크 부분공간에 가두어 delta 아티팩트를 만들고, ROME·MEMIT은 특정 사실을 지목해 직접 덮어쓴다. 세 도구 모두 **어떻게 쓰는가**에 답한다. 어느 것도 **무엇을 쓸 것인가**에는 답하지 않는다. LoRA는 학습 집합을 받아서 쓰고, MEMIT은 삼항 관계를 받아서 쓴다. 그 입력을 누가 만드는지는 도구 밖에 남는다.

ch16이 같은 공백을 다른 쪽에서 건드렸다. Generative Adapter는 delta의 내용을 forward pass 하나로 결정하되, 그 결정 규칙은 배포 전에 한 번 학습되고 그 뒤로 고정된다. 문맥마다 무엇을 접을지는 학습된 사영 행렬이 정하고, 그 정책은 문맥을 보고 다시 조정되지 않는다.

SEAL(*Self-Adapting Language Models*, arXiv:2506.10943, 이하 [SEAL])은 그 빈자리에 모델 자신을 놓는다. 새 문맥이 도착하면 모델이 먼저 **그 문맥으로 자기를 어떻게 학습시킬지를 글로 쓴다.** 그 글이 학습 집합이 되고, 그것으로 (U-$\Theta$)를 돌려 얻은 모델의 downstream 점수가 다시 그 글을 쓴 정책의 보상이 된다.

정직하게 적어야 할 것이 하나 있다. **이 논문은 어떤 선행 논문의 open question도 인용해 받지 않는다.** 대신 두 갈래의 **실무 한계**를 명시적으로 이어받고, 두 갈래 모두 저자가 겹친다. 하나는 test-time training 프로토콜이다 — "손으로 조율한 heuristic에 의존해 augmentation과 optimization 설정을 고르는 대신, 우리는 SEAL이 그 결정을 학습하도록 훈련한다" [SEAL §3.2, p.6]. 다른 하나는 합성 데이터 생성이다 — "SEAL은 정적이거나 heuristic한, 손으로 조율된 생성 전략에 의존하는 대신, gradient 기반 자기 갱신에 적용되었을 때 합성 데이터의 downstream 효용을 직접 최대화하는 생성 policy를 RL로 학습함으로써 이 라인 위에 선다" [SEAL §2, p.2]. 두 인용의 공통 구조가 이 장 전체의 형태다 — **손으로 고정된 규칙 하나를 학습되는 정책으로 바꾼다.**

ch14가 $E$-경로에서 상속 없는 시작이었고 ch16이 다른 공동체에서의 입장이었다면, 이쪽은 세 번째 종류다. 계보는 조밀한데 **질문의 형태로 진술되지 않는다.** 그래서 이 책은 논문이 인용한 두 한계 진술을 그대로 옮기고, 그것이 ch19가 남긴 공백과 같은 공백임을 이 책의 판단으로 잇는다.

## 20.2 문제의식

논문이 세우는 대립은 한 줄이다.

> "Large language models (LLMs) are powerful but static; they lack mechanisms to adapt their weights in response to new tasks, knowledge, or examples." [SEAL Abstract, p.1]

§1이 그 정적임의 내용을 좁힌다. 현재의 모델은 과제 데이터를 fine-tuning이나 in-context learning을 통해 **"있는 그대로(as-is)" 소비하고 학습하는데**, "그런 데이터는 학습에 최적인 형식(또는 분량)이 아닐 수 있고, 현재의 접근법은 모델이 자기 학습 데이터를 어떻게 변환하고 그로부터 어떻게 가장 잘 배울지에 대한 맞춤 전략을 개발하도록 하지 못한다" [SEAL §1, pp.1–2].

그래서 가설이 던져진다 — "흥미로운 가설 하나를 탐구한다: LLM이 자기 자신의 학습 데이터와 학습 절차를 변환하거나 생성함으로써 self-adapt할 수 있는가?" [SEAL §1, p.1].

목표 능력은 둘로 갈린다 [SEAL §3.2 서두, p.5]. (1) 새 정보를 가중치에 통합해서 **문맥 없이** 회상되게 하는 것 — no-context SQuAD. (2) 소수 예제로부터 새 과제로 일반화하는 것 — ARC.

> **[해설]** 이 두 목표는 이 책의 프레임에서 같은 층에 대한 두 질문이다. (1)은 $E$-경로가 $\mathrm{ret}(E,q)$로 푸는 문제를 $\Theta$에 밀어 넣은 것이고, (2)는 ch16이 forward로 만든 delta를 gradient로 만들되 그 gradient의 설정까지 생성하게 한 것이다. 논문은 두 목표를 "적용 사례 둘"로 나열하지만, 통일 표기로 보면 하나의 식 (U-$\Theta$)에서 $\mathrm{gen}(\cdot;B_s)$가 무엇을 뱉느냐만 다르다 — (1)에서는 데이터, (2)에서는 데이터와 갱신 규칙 둘 다.

신규성의 자리도 못 박아 둔다. 가중치 갱신 자체는 새롭지 않고(LoRA SFT), RL 알고리즘도 새롭지 않다(ReST$^{EM}$). 새로운 것은 **루프를 닫은 것**이다 — 갱신된 모델의 점수가 그 갱신의 데이터를 만든 생성기의 학습 신호가 된다. 논문의 표현으로는 "별도의 adaptation 모듈이나 보조 network에 의존하는 선행 접근과 달리, SEAL은 모델의 생성을 자기 adaptation 과정을 파라미터화하고 제어하는 데 직접 쓴다" [SEAL Abstract, p.1]이다.

## 20.3 Core mechanism (통일 표기)

먼저 이름 충돌 하나를 막는다. **SEAL이 말하는 "inner loop"는 NM이 말하는 inner loop가 아니다.** NM과 Titans/TTT 계열에서 inner loop는 토큰마다 $W$를 미는 per-token 갱신이다. SEAL에서 inner loop는 **SFT 실행 한 번 전체**다 [SEAL Alg. 1 line 5, p.4]. 걸리는 시간은 초 단위에서 분 단위다 — self-edit 평가 1회가 30–45초이고 [SEAL §5, p.9], few-shot의 과제당 TTT는 30초에서 수 분이다 [SEAL §A.4, p.19]. 이 논문에는 per-token inner loop가 어디에도 없다. 이 책의 독자가 오독할 확률이 가장 높은 지점이 여기다.

배치를 적어 둔다. 지식 통합의 base 모델은 Qwen2.5-7B [SEAL §B.1, p.20], few-shot은 Llama-3.2-1B-Instruct다 [SEAL §4.1, p.6]. 아키텍처는 전 과정에서 수정되지 않는다.

sleep 시점의 첫 동작은 **decode 한 번**이다.

$$
\mathcal{R}_k \;\sim\; \mathrm{LM}\big(\,\cdot\;\big|\;c\,;\ \Theta_k\big)
\tag{20-1}
$$

$c$는 문맥이다 — SQuAD passage 하나이거나 few-shot 시연 묶음 하나 [SEAL §3.1, p.3]. (이 장에서 소문자 $c$는 문맥을 뜻한다. NM이 $c$에 준 Atlas의 Omega-rule 창 길이는 이 장에 등장하지 않는다.) 식 (20-1)이 표준형 (U-$\Theta$)의 $\mathcal{R}_k = \mathrm{gen}(\mathcal{H}_k; B_s)$와 대응하는 자리이며, 대응은 두 군데에서 어긋난다. $\mathrm{gen}$이 외생 절차가 아니라 $\mathrm{LM}(\cdot;\Theta_k)$ 자신이고, $\mathcal{H}_k$가 누적된 경험이 아니라 **문맥 하나**다.

두 번째 동작이 실제 쓰기다.

$$
\Theta' \;\leftarrow\; \mathrm{SFT}\big(\Theta_k,\ \mathcal{R}_k\big),
\qquad
\Theta' = \Theta_k + \Delta,
\quad
\Delta = \tfrac{\alpha_{\mathrm{LoRA}}}{r}\,BA,
\quad
B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times d}
\tag{20-2}
$$

**이 식은 논문에 없다.** SEAL에는 inner 갱신에 대한 번호 붙은 식이 한 개도 없고, §3.1–§3.2가 산문으로만 진술한다 — "supervised finetuning을 통해 파라미터를 갱신한다: $\theta' \leftarrow \mathrm{SFT}(\theta, SE)$" [SEAL §3.1, p.4], "각 sequence에 대해 표준 causal language-modeling loss를 계산하고 파라미터를 갱신해 $\theta'$을 얻는다" [SEAL §3.2, p.5]. LoRA 재파라미터화는 ch19의 원문에서 상속되며 재진술되지 않는다. 식 (20-2)의 우변은 그 산문과 하이퍼파라미터 표를 합친 **이 책의 복원**이다. 갱신이 저랭크인 이유도 산문에 있다 — "갱신당 데이터 양이 적고 전체 갱신 횟수가 많으므로, 효율적이고 가벼운 튜닝을 위해 low-rank adapter를 쓴다" [SEAL §3.2, p.5].

세 번째 동작이 이 논문의 본체다. $\Theta'$을 downstream 과제 $\mathcal{T}$에서 채점하고, 그 점수를 $\mathcal{R}_k$를 쓴 정책의 보상으로 되먹인다.

$$
\mathcal{L}_{\mathrm{RL}}(\Theta_k) \;:=\; -\,\mathbb{E}_{(c,\mathcal{T})\sim\mathcal{D}}
\Big[\ \mathbb{E}_{\mathcal{R}\sim \mathrm{LM}(\cdot|c;\Theta_k)}\big[\,R(\mathcal{R},\mathcal{T},\Theta_k)\,\big]\Big]
\tag{20-3}
$$

$$
R(\mathcal{R},\mathcal{T},\Theta_k) \;=\;
\begin{cases}
1 & \mathcal{R}\text{로 adaptation한 결과가 }\mathcal{T}\text{에서 }\mathrm{LM}_{\Theta_k}\text{의 성능을 개선하면}\\
0 & \text{그 외}
\end{cases}
\tag{20-4}
$$

식 (20-3)은 [SEAL Eq. 1, §3.1, p.4], 식 (20-4)는 [SEAL Eq. 2, §3.1, p.4]다. 보상이 행동만이 아니라 **파라미터 자신에 의존한다**는 것을 논문이 명시한다 — "표준 RL 설정과 달리, 우리 설정에서 주어진 행동에 배정되는 보상은 그 행동이 취해진 시점의 모델 파라미터 $\theta$에 의존한다" [SEAL §3.1, p.4]. 따라서 RL의 state는 $(c, \Theta)$인데 관측은 $c$뿐이다.

gradient 추정량과 그 성질 — 왜 filtered behavior cloning이 되는지, 왜 확률을 내리는 방향의 gradient가 없는지, stop-gradient 근사가 무엇을 바꾸는지 — 은 **ch05 §05.7.1이 소유한다.** 이 장은 그 결론만 받아 쓴다: 식 (20-3)의 gradient는 $R=1$을 받은 self-edit들에 대한 SFT로 축약되고, 실제 outer step은

$$
\Theta_{k+1} \;=\; \Theta_k \;-\; \eta_\Theta\,\nabla_{\Theta_k}\mathcal{L}_{\mathrm{RL}}
\tag{20-5}
$$

이며 구현은 LoRA SFT다 — batch 10, 2 epoch, 학습률 3e-4, rank 64, alpha 128, "모든 MLP과 attention projection layer에 적용" [SEAL §B.2, p.20].

읽기는 표준형 (R)에서 두 항이 비어 있다.

$$
\hat y \;=\; f\big(q;\ \Theta',\ W=\varnothing,\ \mathrm{ret}(E,q)=\varnothing\big)
\tag{20-6}
$$

문맥은 질의 시점에 사라진다 — "갱신된 모델은 원문에 접근하지 않은 채 그 passage에 관한 질문으로 평가된다" [SEAL Fig. 2 캡션, p.5]. 답변 프롬프트는 두 줄이다: "Let's answer a question directly and concisely. Question: {question} Answer:" [SEAL §B.3, p.20]. **corpus 전체에서 가장 순수한 $\Theta$-경로 읽기다.**

### 20.3.1 표기 대응표

표 20-1 — SEAL 원 표기 → 이 책의 표기

| 원 논문 기호 | 이 책 | 근거 |
|---|---|---|
| $\theta$ — $\mathrm{LM}_\theta$의 파라미터 [§3.1] | $\Theta$ | slow weights. 시간 첨자 붙은 소문자 금지 |
| $\theta'$ — inner 갱신 후 [§3.1] | $\Theta'$ | outer의 $\Theta_{k+1}$과 **다른 이동**임을 눈에 보이게 |
| $\theta_t$ — outer RL 반복 $t$ [Eq. 1, Alg. 1] | $\Theta_k$ | $t$는 토큰 첨자로 예약. **SEAL의 $t$는 토큰이 아니다** |
| $C$ — 문맥(passage 또는 시연) [§3.1] | $c$ | 맨 $C$는 chunk 크기, 용량은 $C_{\text{cap}}$ |
| $\tau$ — downstream 평가 [§3.1] | $\mathcal{T}$ | $\tau$는 세션 첨자로 예약 |
| $SE$ — self-edit | $\mathcal{R}$, $\mathcal{R}_k$ | (U-$\Theta$)의 학습 집합 자리 |
| $L_{RL}$ [Eq. 1] | $\mathcal{L}_{\mathrm{RL}}$ | outer loss는 calligraphic. 맨 $L$은 sequence 길이 |
| $y_s^{(i,j)}$ — self-edit의 $s$번째 토큰, 길이 $T$ [Eq. 4] | $z_t^{(i,j)}$, $t=1,\dots,L_{\mathcal{R}}$ | $y$는 layer 출력, $s$는 SGD step |
| $\alpha$ — LoRA alpha [Tables 3–4] | $\alpha_{\mathrm{LoRA}}$ | $\alpha_t\in[0,1]$은 retention gate |
| $r$ — **보상**[Eq. 2–4] **이자 LoRA rank**[Table 3]; 같은 rank를 Table 4는 $R$로 인쇄 | 보상 $\to R$, rank $\to r$. **두 뜻이 맞바뀐다** | ch01의 예약 규칙 |
| "training steps" — "375 step 초과 설정은 폐기" [§A.2] | $s$; ARC 필터는 $s \le 375$ | $k$(라운드)와 절대 혼동 금지 |
| layer 수 — 기호 없음. "모든 MLP과 attention projection layer" [§B.2] | 필요하면 $L_{\text{layer}}$, 대상 모듈 집합은 $\mathcal{S}$ | 논문은 개수를 주지 않는다 |
| learning rate — inner와 outer 양쪽 | inner $\eta_{\mathrm{in}}$, outer $\eta_\Theta$ | few-shot에서 $\eta_{\mathrm{in}}$은 **모델이 생성**한다 |
| $D$ — $(C,\tau)$ 쌍의 데이터셋 [Alg. 1] | $\mathcal{D}$ | 맨 $D$는 예약됨 |
| $N$, $M$ — minibatch 문맥 수, 문맥당 self-edit 표본 수 [Eq. 3] | 그대로 | 충돌 없음 |

경고 한 줄을 크게 붙인다. **이 corpus에서 원문과 이 책의 $r$·$R$이 정확히 반대 뜻인 논문은 SEAL 하나다.** 본문에서 [SEAL Eq. 2]와 [SEAL Table 3–4]를 나란히 인용할 때마다 이 맞바꿈이 작동한다 — 논문 §3에서 $r$은 보상이고, 부록에서는 $r$과 $R$이 **둘 다** rank다.

### 20.3.2 이 장이 소유하는 두 정의

> **정의.** **self-edit**은 모델 자신이 생성한 자연어 산출물로서, 자기 가중치 갱신의 **내용을 지정**한다. 최소 형식은 갱신에 쓰일 학습 집합 그 자체이고, 최대 형식은 학습 집합에 더해 optimizer 설정(학습률·epoch 수·loss mask)과 데이터 변환 도구의 호출까지를 포함한다. 통일 표기로 self-edit은 $\mathcal{R}_k$이고 식 (20-1)로 만들어진다 — 즉 (U-$\Theta$)의 $\mathrm{gen}(\cdot;B_s)$가 **decode pass 한 번**이다.

정의만으로는 부족하다. 이 정의가 무엇을 가리키는지는 실제로 생성된 두 형식이 정한다.

**형식 A — 데이터.** 지식 통합 실험의 self-edit은 passage의 "함의(implications)" 목록이다. 프롬프트가 전문 그대로다: "Let's read the following passage and produce a list of implications derived directly or indirectly from the content. / Passage: {passage} / Implications:" [SEAL §B.3, p.20]. 생성된 sequence는 single-passage 레짐에서 **줄바꿈으로 쪼개져 여러 개의 학습 문서**가 되고, multi-passage 레짐에서는 통째로 하나의 학습 문서가 된다 [SEAL §B.3, p.20]. [SEAL Fig. 2]가 그 예를 보인다 — Apollo 계획 passage에서 "1. The Apollo program faced opposition from Kennedy's science advisor, Jerome Wiesner, who had…"라는 문장이 나오고, 그 문장으로 LoRA SFT를 돌린 모델이 원문 없이 "Jerome Wiesner"를 답한다 [SEAL Fig. 2, p.5].

**형식 B — 갱신 규칙.** few-shot 실험의 self-edit은 데이터가 아니라 **설정**이다. [SEAL Fig. 3]이 생성된 내용을 그대로 인쇄한다: `basic_augmentations: true`, `size_augmentations: false`, `chain_augmentations: false`, `repeat_augmentations: false`, `strategy: loss on all tokens`, `learning rate: 1e-05`, `epochs: 3` [SEAL Fig. 3, p.5]. 선택 가능한 도구는 회전·뒤집기·반사·전치·해상도 변경과 그 연쇄·반복, 그리고 학습률·epoch 수·loss를 전체 토큰에 걸지 출력 토큰에만 걸지다 [SEAL §3.2, p.6].

> **[해설]** 형식 B가 이 장에서 가장 저평가된 사실이다. 표준형 (U-$\Theta$)에서 $\mathrm{gen}(\cdot;B_s)$은 **데이터 생성기**로만 정의되어 있고, $\eta_\Theta$·epoch·loss mask는 식 밖의 하이퍼파라미터다. 형식 B는 그 하이퍼파라미터를 $\mathcal{R}_k$ 안으로 끌어들인다. 프레임에 대한 SEAL의 가장 날카로운 확장이 이것이며, 이 책은 이후 $\mathrm{gen}$이 **갱신 규칙까지 뱉을 수 있는 채널**을 가진 것으로 다룬다.

두 번째 정의가 루프 자체다.

> **정의.** **자기 생성 데이터로 $\Theta$를 움직이는 루프**는, (U-$\Theta$)의 생성기 $\mathrm{gen}(\cdot;B_s)$와 갱신 대상 $\Theta$가 **같은 파라미터 벡터**인 루프다. 라운드 $k$에서 (i) $\Theta_k$가 $\mathcal{R}_k$를 쓰고, (ii) $\mathcal{R}_k$가 $\Theta_k$를 $\Theta'$으로 밀고, (iii) $\Theta'$의 점수가 (i)을 수행한 정책을 $\Theta_{k+1}$로 민다. 갱신 대상과 갱신 저자가 분리되지 않는다는 것이 이 루프의 정의적 성질이고, 논문의 표현으로는 "SEAL은 모델의 생성을 자기 adaptation 과정을 파라미터화하고 제어하는 데 직접 쓴다" [SEAL Abstract, p.1]다.

이 루프가 표준형 (U-$\Theta$)와 갈리는 지점은 여섯이다.

1. **$\mathrm{gen}$이 $\Theta$다.** 표준형은 $\mathrm{gen}$을 외생 절차로 둔다.
2. **outer 목적이 $\mathcal{R}_k$ 위의 loss가 아니다.** 표준형은 $\nabla_\Theta \mathcal{L}(\mathcal{R}_k;\Theta)$를 취하는데, SEAL의 outer gradient는 $\log p(\mathcal{R}_k \mid c)$에 대한 것이다 — gradient는 **$\mathcal{R}$을 썼을 확률**을 통해 흐르고, $\mathcal{R}$을 소비한 SFT를 통해서는 흐르지 않는다.
3. **$\mathcal{H}_k$가 문맥 하나다.** 누적 경험이 아니다 — "우리 방법이 단일 예제 TTT 에피소드로 학습되기는 하지만…" [SEAL §2 Test-Time Training, p.3].
4. **갱신이 부분공간에 갇힌다 — 한 레짐만 빼고.** 상세는 §20.4.
5. **$\mathrm{gen}$이 데이터만이 아니라 갱신 규칙을 뱉는다** (형식 B).
6. **보상이 문맥마다 라벨된 downstream 과제를 요구한다.** 표준형에 없는 oracle이며, 논문 자신이 스케일의 걸림돌로 지목한다 [SEAL §5, p.9].

## 20.4 어느 층을 언제 쓰는가

"이 값은 누가 학습하는가"에 전부 답한다.

표 20-2 — 이 값은 누가 학습하는가

| 객체 | 학습 주체 | 언제 움직이는가 | 질의 후 남는가 | 출처 |
|---|---|---|---|---|
| $\Theta$ — base backbone (Qwen2.5-7B / Llama-3.2-1B-Instruct) | 여기서 사전학습하지 않음. **동결도 아님 — 레짐에 따라 다름**(아래 세 경우) | (a) few-shot ARC: 안 움직임 (b) 지식 통합 $n{=}1$: 안 움직임 (c) 지식 통합 CPT $n{=}200, 2067$: **움직인다**(full finetuning) | (a),(b) base는 불변, adapter가 남음 (c) 움직인 backbone이 곧 산출물 | §4.1 p.6; §4.2 p.7; Table 2 p.8; §A.2 p.18; §B.2 p.20 |
| self-edit 생성 policy (= 생성기로 본 $\Theta_k$) | **outer RL 학습**(배포 전, 오프라인) | ReST$^{EM}$ 라운드마다 1회. 지식 통합 2라운드, few-shot 1라운드 | 예 — 출하되는 모델이 이것 | §3.1 pp.3–4; §A.2 p.18; §B.2 p.20 |
| $\mathcal{R}_k$ — self-edit 자체 | 파라미터가 아님. sleep 시점 decode로 **생성됨** | 새 문맥 $c$마다 1회, 질의 도착 전 | single-passage에서는 갱신 후 **폐기**. CPT에서는 전체를 코퍼스로 **보존** | §B.2 p.20; §4.2 p.7 |
| $\Delta$ — inner SFT가 만든 LoRA adapter (실제 새 지식) | **sleep 시점 갱신**(문맥마다, gradient 기반) | 문맥마다 1회. decode 중에는 아님 | 배포 시 **예**. RL 학습 중에는 **아니오** — 라운드당 750회 inner 실행이 전부 평가 후 버려짐 | Abstract p.1; §3.2 p.5; §B.5 p.21 |
| inner 하이퍼파라미터 ($\eta_{\mathrm{in}}$, epoch 수, loss mask) | **few-shot에서만 모델이 생성**하고 그 생성기가 RL로 학습됨. 지식 통합에서는 **저자가 grid search** | few-shot: 과제마다 sleep 시점. 지식 통합: 오프라인 고정 | 정책은 남고 값은 과제별 | §3.2 p.6; Fig. 3 p.5; §A.2 p.18; Tables 3–4 |
| augmentation 도구 선택 | few-shot에서만 모델이 생성. 도구 자체는 고정 함수 | 과제마다 sleep 시점 | 정책이 남음 | §3.2 p.6 |
| LoRA rank·alpha·대상 모듈 $\mathcal{S}$ | 저자가 고정. 학습되지도 생성되지도 않음 | 없음 | 해당 없음 | §A.2 p.18; §B.2 p.20; Tables 3–4 |
| $W$ — fast weights | **이 논문에 존재하지 않음** | — | — | 전편 부재. per-token 상태 갱신이 정의되지 않음 |
| $E$ — external store | **이 논문에 존재하지 않음** | — | — | 전편 부재. wake 프롬프트가 비어 있음 §B.3 p.20 |
| wake 시점의 이동 | **아무것도 움직이지 않음** | — | — | §3.2 p.5, Fig. 2 |

순서는 라운드 $k$마다 다섯 단계다. (i) $\mathcal{R}_k \sim \mathrm{LM}(\cdot|c;\Theta_k)$, (ii) inner (U-$\Theta$)로 $\Theta_k \to \Theta'$, (iii) $\mathcal{T}$에서 $\Theta'$을 채점해 $R$을 얻음, (iv) **$\Theta'$을 버림**, (v) outer (U-$\Theta$)로 $\Theta_k \to \Theta_{k+1}$. 배포 시에는 (i)–(ii)만 돌고 $\Theta'$을 남긴다.

**시간척도가 셋이고 첨자가 셋이다.** $s$는 하나의 self-edit을 소비하는 SFT 안의 optimizer step이고(few-shot의 폐기 필터가 $s \le 375$다 [SEAL §A.2, p.18]), 그 위에 모델이 고르는 epoch 수가 있으며, 그 위에 sleep 라운드 $k$가 있다. **SEAL이 쓰는 "step"이라는 단어를 $k$로 옮기면 틀린다.** 이 셋을 겹쳐 세는 논문이 corpus에서 SEAL 하나다.

판별식 네 조건을 댄다(→ ch11 §11.3). (1) 질의 전인가: **그렇다** — 문맥이 도착하면 self-edit 생성과 SFT가 먼저 끝나고 질문은 그 뒤에 온다. (2) $B_s > 0$인가: **그렇다** — decode 한 번 + SFT 한 번. (3) 어느 한 층의 상태가 바뀌는가: **그렇다** — $\Theta$가 움직이고 배포 아티팩트가 바뀐다. (4) 이후 질의에 쓰이는가: **그렇다** — "이 self-edit들은 지속적인 가중치 갱신을 낳아 지속적인 adaptation을 가능하게 한다" [SEAL Abstract, p.1]. 네 조건 전부 통과이고, 실행되는 식은 **(U-$\Theta$) 하나뿐**이므로 순서 문제도 생기지 않는다.

여기서 한 가지를 정정해 둔다. **"backbone은 절대 안 움직인다"는 서술은 이 논문에 대해 참이 아니다.** 논문이 두 번 명시한다 — "표는 두 레짐에서의 평균 no-context SQuAD 정확도를 보고한다: single-passage 갱신(LoRA)과 소규모 continued pretraining(full finetuning)" [SEAL §4.2, p.7], 그리고 Table 2의 열 머리말이 "Single Passage (n = 1; LoRA)", "Continued Pretraining (n = 200; full-FT)", "(n = 2067; full-FT)"다 [SEAL Table 2, p.8]. 네 칸(도메인 × 루프) 중 **정확히 한 칸**에서만 full finetuning이 일어난다 — 지식 통합 CPT 레짐의 inner 갱신이다. 나머지 셋은 전부 LoRA이고, 특히 **outer 루프는 두 도메인 모두에서 LoRA다.** 다시 말해 SEAL이 RL로 배우는 것은 어느 경우에도 base tensor를 움직이지 않는다. 움직이는 것은 CPT의 inner 갱신뿐이다.

> **[평가]** 이 구분은 회계 문제가 아니라 서빙 문제다. 세 칸에서 지속 상태는 adapter이고 한 칸에서는 **7B 체크포인트 전체**다. 그 한 칸에서 $C_{\text{cap}}$이 자릿수 단위로 뛰고, ch01 Rosetta의 "shared-weight batching → $\Theta$-경로에서 깨짐" 행의 발동 조건도 정반대로 바뀐다(§20.7). 그리고 §20.6이 보일 Table 2의 열 사이 비교가 **스케일 계열이 아닌** 이유가 바로 이것이다 — 47.0 → 58.2는 $n$의 변화와 갱신 파라미터화의 변화를 함께 담고 있다.

<!-- TODO-VERIFY: CPT 레짐이 정말 full-FT인지. §4.2 p.7 본문과 Table 2 p.8 열 머리말은 full-FT라고 두 번 말하는데, 부록 Table 4 p.21("Multi-Passage Knowledge Incorporation Hyperparameters")은 여전히 LoRA rank·alpha 탐색 공간 [32,64]를 싣는다. 확인 방법: 공개된 n=200/n=2067 학습 설정에서 LoraConfig/get_peft_model 사용 여부. 본문은 두 번 진술된 쪽을 따랐다. -->

## 20.5 비용 4종

표 20-3 — 비용 4종

| 기호 | 값 | 근거 |
|---|---|---|
| $B_s$ | **일부만 보고. 벽시계로만이고, RL 학습 루프에 대해서만이며, 배포 시점 sleep 단계에 대해서는 없다.** self-edit 평가 1회 "대략 30–45초"; 지식 통합 1라운드 = 50 passage × 5 completion × 3 seed = 750회 inner 실행, 2×H100에서 "약 6시간"; 헤드라인은 2라운드; few-shot은 과제당 TTT "30초에서 수 분", ReST$^{EM}$ 전체 "약 2–3시간"(A100/H100/H200 한 장); proxy-reward 변형은 RL 학습 전체가 약 5분. **FLOPs·self-edit 토큰 수는 논문에 없음. 배포 시점 1문맥 sleep 비용(decode 1회 + LoRA SFT 1회)을 보상 평가와 분리해 잰 값도 논문에 없음. CPT 실행 비용도 논문에 없음** | §5 p.9; §A.4 p.19; §B.5 p.21; Table 9 §B.10 p.23 |
| $L_w$ | **논문에 없음.** 지연도 TTFT도 tokens/s도 질의당 비용도 어떤 형태로도 보고되지 않는다. wake에 대한 진술은 구조적인 것 하나뿐이다 — 원문 없이 평가하고 프롬프트가 두 줄이다. 그것이 함의하는 prefill 절감에 수치가 붙어 있지 않다 | §B.3 p.20; Fig. 2 캡션 p.5 |
| $C_{\text{cap}}$ | **rank로만 보고. 바이트도 파라미터 수도 없음.** few-shot TTT adapter는 rank 128·alpha 16을 `q_proj, v_proj, gate_proj, down_proj, up_proj`에; few-shot RL SFT adapter는 rank 16·alpha 16; 지식 통합 RL SFT adapter는 rank 64·alpha 128을 "모든 MLP과 attention projection layer"에; inner adapter 탐색 공간은 rank·alpha 모두 [32,64]. **adapter당 바이트, 학습 파라미터 수, 사용자·세션당 상태 예산, 질의 후 adapter의 처분(유지·병합·축출)은 전부 논문에 없음.** CPT 레짐의 상태는 adapter가 아니라 7B 모델 전체인데 그 비용도 회계되지 않는다 | §A.2 p.18; §B.2 p.20; Tables 3–4 pp.20–21; §4.2 p.7 |
| $\rho$ | **정성적으로만. 열화를 보이되 rate를 인쇄하지 않는다.** "Figure 6에서 보듯 편집 횟수가 늘수록 이전 과제의 성능이 점진적으로 하락하며, 이는 SEAL이 여전히 catastrophic forgetting에 취약함을 시사한다. 그래도 완전한 붕괴 없이 여러 번의 갱신을 수행할 수 있다" [§5, pp.8–9]. [SEAL Fig. 6]은 8회까지의 히트맵이고, Table 5는 그 히트맵의 **항목별 표준오차**(비영 항목에서 0.0211–0.0406)를 싣는다. **정확도 값 자체는 어디에도 표로 없다.** 따라서 편집 1회당 손실 %p, 실패 임계까지의 편집 횟수, replay·EWC 통제군은 전부 논문에 없음 — "현재 학습 설정에서 retention을 명시적으로 최적화하지 않는다" [§5, p.8] | §5 pp.8–9; Fig. 6 p.8; Table 5 §B.6 p.22 |

**상각식 (A)는 닫히지 않는다.** $C_{\text{wake}}$ 쪽은 구조적으로 0에 가깝다 — 질의 시점에 추가 상태 읽기도 검색 hop도 없다. 그런데 $C_{\text{sleep}}$이 FLOPs로도 토큰으로도 없고, 배포 시점 sleep 단계가 격리되어 측정된 적도 없다. 분모 $N_q$도 마찬가지다. 논문에서 가장 가까운 수는 평가 설계에서 나온다 — 200 passage에 974개 질문, "passage당 대략 5개" [SEAL §B.4, p.20]. **$N_q \approx 5$로 읽는 것은 이 책의 대용 해석이며, 논문은 이 수를 상각 인자로 제시하지도 상각 분석을 수행하지도 않는다.**

> **[평가]** 이 논문은 계산을 질의 이전으로 **옮기자고 제안하면서, 옮기는 계산도 사는 절감도 측정하지 않는다.** 보고된 두 수(self-edit 평가 30–45초, 라운드당 6시간)는 전부 학습 루프 비용이다. ch01 §01.5가 정한 실천 규칙에 따라 이 장은 SEAL에 대해 "효율적"이라고 쓰지 않는다. 쓸 수 있는 문장은 둘뿐이다 — **wake 시점 프롬프트에서 passage가 사라진다**, 그리고 **RL 학습 라운드 하나가 2×H100 6시간이다.** ch01 §01.5의 비용 4종 표를 이 논문에서 채울 수 없다는 사실 자체가 ch25가 받아야 할 자료다.

## 20.6 실험과 스케일

스케일 상한부터 적는다. 최대 모델은 Qwen2.5-7B이고, 스케일 점은 3B와 7B **두 개뿐**이다 [SEAL Table 6, §B.7]. 이 논문의 가장 새로운 기제 — 모델이 자기 optimizer 설정을 쓰는 형식 B — 는 **Llama-3.2-1B-Instruct에서만** 돌았다. RL 학습이 닿은 데이터는 지식 통합에서 2라운드 × 50 문맥 = 100 문맥, few-shot에서 "11개 학습 문제에 대한 단 한 번의 outer loop"다 [SEAL §A.2, p.18; §B.2, p.20]. 본문과 [SEAL Fig. 1]이 outer loop를 반복되는 루프로 제시하는 데 반해 few-shot 도메인에서 실제로 돈 라운드는 하나이며, 두 도메인의 학습량이 이렇게 비대칭이라는 사실은 본문에 정리되어 있지 않다. 하드웨어 상한은 2×H100 또는 2×H200이다. 토큰 예산은 어디에도 보고되지 않는다.

**지식 통합.** no-passage-in-context SQuAD v1.1이고, 채점은 gpt-4.1-2025-04-14가 greedy decoding으로 하는 자동 채점이다 [SEAL §B.4, pp.20–21]. RL 학습 집합과 평가 집합이 겹치지 않음을 논문이 명시한다 [SEAL §B.1, p.20].

표 20-4 — no-context SQuAD 정확도 [SEAL Table 2, p.8]

| 방법 | Single Passage ($n{=}1$; LoRA) | CPT ($n{=}200$; full-FT) | CPT ($n{=}2067$; full-FT) |
|---|---|---|---|
| Base model | 32.7 | 32.7 | 29.0 |
| Train on Passage | 33.5 | 36.0 | 31.2 |
| Train on Passage + Synthetic (RL 없는 base 모델 self-edit) | 39.7 | 50.6 | 43.4 |
| Train on Passage + GPT-4.1 Synthetic | 46.3 | **59.4** | **49.2** |
| SEAL | **47.0** | 58.2 | 46.4 |

**초록의 헤드라인은 세 열 중 한 열에서만 성립한다.** 초록과 §1이 모두 "SEAL의 자기 생성 데이터가 GPT-4.1이 생성한 합성 데이터를 능가한다"고 쓰는데 [SEAL Abstract, §1, p.1], 그것이 참인 칸은 $n{=}1$의 47.0 대 46.3 — **마진 0.7점** — 하나뿐이다. 두 continued-pretraining 열에서는 GPT-4.1이 이긴다: 59.4 대 58.2, 49.2 대 46.4. 논문 본문은 이 사실을 진술한다 — "single-passage의 경우와 유사한 방법 순위를 관찰하되, GPT-4.1의 합성 데이터가 SEAL을 근소하게 앞선다" [SEAL §4.2, p.7]. 진술은 §4.2에 있고 범위 조건은 초록에 없다.

열 사이를 스케일 계열로 읽어서는 안 된다. §20.4가 정정한 대로 $n{=}1$은 LoRA이고 $n{=}200$·$n{=}2067$은 full finetuning이므로, 47.0 → 58.2에는 문서 수의 변화와 갱신 파라미터화의 변화가 함께 들어 있다. $n{=}200 \to 2067$의 하락도 방법의 열화로만 읽을 수 없다 — 평가 집합이 함께 바뀌어 base 모델도 32.7 → 29.0으로 떨어진다. 읽을 수 있는 것은 **base 대비 마진**이다(이 뺄셈은 이 책의 산술이다): SEAL은 25.5 → 17.4, GPT-4.1은 26.7 → 20.2, 그리고 heuristic 기준선인 Entigraph(10 생성)는 23.3 → 19.6이다. **세 마진이 같은 방향으로 줄되 SEAL이 가장 많이 준다.**

RL 라운드의 수익 형태도 여기서 읽힌다. 'Implications' 프롬프트의 궤적은 39.7 → 43.7 → 47.0이다 [SEAL Table 10, §B.11, p.24]. 논문의 읽기는 "두 번의 반복이면 SEAL이 GPT-4.1 데이터를 넘어서기에 충분하고, 이후 반복은 수익이 체감한다" [SEAL §4.2, p.7]이다. **이것이 $B_s$–품질 곡선의 형태에 대한 이 논문의 유일한 증거이며, 두 라운드에서 포화한다.**

**few-shot ARC.** ICL 0%, RL 없는 TTT + Self-Edit 20%, SEAL 72.5%, Oracle TTT 100%다 [SEAL Table 1, p.7]. 이 네 수에는 즉시 붙어야 할 단서가 있다. 학습 11개·평가 8개의 ARC 과제가 "base Llama-3.2-1B-Instruct의 최적 TTT 설정 아래에서 풀릴 수 있도록 필터링"되었다 [SEAL §4.1, p.6; §A.1, p.18]. **따라서 Oracle TTT의 100%는 실증 결과가 아니라 필터의 정의적 귀결이고, ICL의 0%도 ICL이 못 하고 TTT가 하는 과제만 고른 결과다.** 헤드라인 전체가 손으로 고른 19개 과제 위에 서 있다. 그리고 그 위에서도 학습된 정책은 손으로 정한 설정에 **27.5점 뒤진다** — 논문이 인정한다 [SEAL §4.1, p.6].

**형식이 RL보다 세다.** 논문은 7개 프롬프트에 대해 ReST$^{EM}$ 2라운드를 돌린 표를 부록에 싣는다.

표 20-5 — self-edit 프롬프트 민감도, no-context SQuAD [SEAL Table 10, §B.11, p.24]

| 프롬프트 | RL 없음 | 라운드 1 | 라운드 2 | GPT-4.1 합성 |
|---|---|---|---|---|
| Implications (본문의 주 설정) | 39.7 | 43.7 | **47.0** | 46.3 |
| Implications-long | **49.3** | 52.4 | 54.4 | 54.1 |
| Implications-very-long | 45.0 | 51.5 | 52.1 | 40.9 |
| Rewrite | **49.4** | 55.3 | 55.6 | 54.4 |
| Self-QA | 37.3 | 42.8 | 48.7 | 39.2 |
| No-Prompt | 13.8 | 12.7 | 18.9 | 28.6 |

세 가지가 한 표에서 나온다. 첫째, **RL을 한 번도 돌리지 않은 두 기준선(49.3, 49.4)이 논문의 RL 헤드라인(47.0)을 넘는다.** 프롬프트 한 단어를 바꾸는 것이 RL 루프 전체보다 값이 크다. 둘째, 그럼에도 **RL의 기여는 실재한다** — 논문의 진술대로 "모든 경우에서 ReST$^{EM}$은 성능을 대략 6에서 11 퍼센트포인트 향상시켰다" [SEAL §B.11, p.23]. 좋은 프롬프트를 출발점으로 삼으면 55.6·54.4까지 간다. 문제는 기제가 아니라 **본문이 보고한 기준선의 선택**이다. 셋째, **"self-directed adaptation"이 이 표에서 반증된다.** 모델이 자기 self-edit 형식을 스스로 정하게 두면(No-Prompt) RL 없이 13.8, 2라운드 뒤에도 18.9이고, 같은 설정의 GPT-4.1(28.6)에도 진다. 자기 편집은 **저자가 고른 형식 안에서만** 작동한다.

**비싼 부품을 지워도 거의 잃지 않는다.** TTT 기반 보상 — finetune-and-evaluate 루프, 즉 이 논문이 선행 합성 데이터 연구 대비 내세우는 바로 그 기여 — 를 GPT-4.1 루브릭 채점(4범주 각 1–5점 합)으로 바꾸면 47.0 대 **45.6**이고, 시간은 약 6시간 대 **약 5분**이다 [SEAL Table 9, §B.10, p.23]. 약 72배의 학습 비용 절감에 1.4점이다. 논문 자신이 "정제하면 post-finetuning 성능을 직접 최적화하는 '참' 보상을 능가할 수도 있다"고 쓴다 [SEAL §B.10, p.23].

**비교 대상들.** ch16의 Generative Adapter를 같은 벤치에 세우면 single-passage에서 66.8 대 47.0으로 SEAL이 **19.8점 진다** — 헤드라인 수치가 나온 바로 그 설정이다. CPT에서는 58.2 대 28.0으로 SEAL이 이긴다 [SEAL Table 7, §B.8, p.22]. 논문이 교란을 밝힌다: Generative Adapter는 Mistral-7B 기반이고 SEAL은 Qwen2.5-7B이며, SEAL을 Mistral에서 돌리지 않았다. heuristic 기준선 Entigraph는 $n{=}200$에서 56.0, $n{=}2067$에서 **48.6**으로 SEAL의 46.4를 넘는다(10 생성 기준; SEAL은 5 생성이므로 계산량이 맞춰져 있지 않고 논문도 그렇게 밝힌다) [SEAL Table 8, §B.9, pp.22–23]. 논문의 읽기는 "RL로 학습된 self-edit과 구조화된 heuristic 방법 모두 합성 데이터 생성의 강한 접근"이다.

모델 크기 논증은 출발 정확도가 다른 두 점 위에 있다. 3B에서 25.1 → 31.9 → 37.0, 7B에서 32.7 → 39.7 → 47.0이고 개선 비율이 1.75배 대 2.04배다 [SEAL Table 6, §B.7, p.22]. 논문이 곧바로 물러선다 — "실제로 더 키워 보지 않고는 결론을 내리기 어렵다".

**보고된 수치 자체의 균열 셋.** (i) base 정확도가 표마다 다르다 — Table 2·6은 32.7, Table 7·9는 32.0이고 논문은 화해시키지 않는다. (ii) 지식 통합의 모든 정확도가 gpt-4.1-2025-04-14 채점이며 **같은 벤더 모델이 동시에 가장 강한 기준선이다.** 사람과의 일치도도 채점기 분산도 없고, 그 겹침을 논문 어디서도 논의하지 않는다. (iii) 식에 쓰인 보상과 실제로 쓴 보상이 다르다 — 각주 2가 지식 통합에서는 임계값이 아니라 후보 중 argmax에 보상을 준다고 밝히고, few-shot에는 문서화되지 않은 필터가 더 붙는다($s > 375$인 설정 폐기) [SEAL 각주 2 p.4; §A.2 p.18]. 어느 쪽도 Eq. 2와 Alg. 1에 없다. RL 알고리즘 선택의 근거도 곡선 없는 부정 결과다 — "GRPO와 PPO 같은 다양한 on-policy 방법을 실험했으나 학습이 불안정했다" [SEAL §3.1, p.4].

**망각.** 순차 self-edit 8회에 걸쳐 이전 passage의 성능이 단조 하락한다 [SEAL §5, pp.8–9; Fig. 6, p.8]. 이 결과의 정확도 값은 표로 인쇄되지 않고 **표준오차만** 인쇄된다 [SEAL Table 5, §B.6, p.22]. 통제군도 없다. 따라서 이 장은 $\rho$를 수치로 옮길 수 없다.

## 20.7 Systems/serving 함의

독자의 1번 질문에 한 문장으로 답한다. **decode에서는 아무것도 바뀌지 않는다** — 그리고 그것이 이 설계의 전부이자 위험 전부다. 적응된 모델은 평범한 transformer이고, 붙은 adapter 하나(또는 CPT에서는 그냥 가중치)를 제외하면 추가 상태 읽기도, 검색 hop도, 토큰당 재귀도 없다. 이 진술은 wake 시점 기제가 아예 정의되지 않았다는 사실로부터의 이 책의 추론이다 — §3은 데이터와 optimizer만 바꾼다.

wake 프롬프트는 **비어 있는 쪽으로 줄어든다.** passage도 검색 레코드도 없이 두 줄이다 [SEAL §B.3, p.20]. $E$-경로와의 대비가 여기서 가장 깨끗해진다 — 검색 시스템은 질의마다 prefill로 값을 치르고, SEAL은 질의마다 0을 치르고 한 번에 전부를 치른다. 식 (A)에서 $C_{\text{wake}} \approx 0$이면 평균 비용이 $C_{\text{sleep}}/N_q$로 축약되므로, **이 설계는 전부가 상각 베팅이다.** 그런데 질의당 절감의 크기는 원래 문맥에 있었을 passage 길이인데, 논문이 passage 토큰 수를 적지 않으므로 **이 장은 절감 수치를 인쇄하지 않는다.**

상태량도 인쇄할 수 없다. $C_{\text{cap}}$을 ch03 §03.6의 식 (3-6)으로 계산하려면 $(r, L_{\text{layer}}, \mathcal{S}, d)$ 넷이 필요한데, 지식 통합 실험의 $\mathcal{S}$가 "모든 MLP과 attention projection layer"라는 산문으로만 있어 재구성되지 않는다. 단정할 수 있는 것은 **레짐 사이의 대비**뿐이다 — 세 칸의 지속 상태는 adapter이고 CPT 한 칸에서는 7B 체크포인트 전체이며, ch03의 기준점(GPT-3 175B에서 $r{=}4$ delta가 체크포인트를 약 10,000배 줄인다)에 비추면 그 대비는 자릿수 몇 개짜리다.

**배칭이 두 갈래로 갈린다.** single-passage 레짐은 "문서마다 adapter를 따로 학습하고 평가한다" [SEAL §B.8, p.22]. 문서마다 가중치 상태가 생기므로 ch01 Rosetta의 "shared-weight batching → $\Theta$-경로에서 깨짐" 행이 그대로 발동하고, 요청별 adapter 라우팅(grouped GEMM, multi-adapter 서빙)이 전제가 된다. ch06 §06.9가 이 대비에 이미 이름을 붙여 두었다 — single-passage는 **사용자·문서 수준 $\Theta$-경로**이고 CPT는 **인구 수준 $\Theta$-경로**이며, Rosetta의 그 행이 발동하는 것은 전자뿐이다. CPT 레짐은 반대다 — 모두가 한 벌의 가중치를 쓰므로 배칭이 살아나지만 문서·사용자 단위 격리가 사라지고, 그 레짐이 바로 $n{=}2067$에서 heuristic에 지는 자리다(46.4 대 48.6). **$\Theta$-경로 서빙 문제의 두 뿔이며, 논문은 둘을 모두 보고하면서 그 교환을 이름 부르지 않는다.** 서빙·배칭·adapter 라우팅에 대한 논의가 논문에 한 줄도 없다.

sleep 단계의 워크로드 모양도 여기서 결정된다. 논문의 실행 구성은 생성·평가에 vLLM, SFT에 DeepSpeed ZeRO-3다 [SEAL §B.5, p.21]. 즉 **SEAL을 돌리는 fleet은 문맥마다 학습 job을 하나씩 돌린다** — 서빙 스택과 학습 스택이 같은 자리에 상주해야 한다. 그리고 한 라운드는 750개의 순차적 finetune-and-evaluate 단위이므로 통상적 의미의 배칭도 되지 않는다.

> **[해설]** 그래서 §20.6의 proxy-reward 결과가 systems 독자에게 이 논문에서 가장 정보량이 크다. 보상을 "학습 실행"에서 "채점 호출"로 바꾸면 sleep 라운드가 먹는 하드웨어의 **종류**가 바뀐다 — 학습 GPU 예약이 API 호출 예산으로 내려온다. 47.0 대 45.6, 약 6시간 대 약 5분이다 [SEAL Table 9, §B.10, p.23]. 그리고 sleep 시점에 전선을 건너가는 것은 가중치가 아니라 **텍스트**다. 가중치 delta는 그 텍스트를 소비하는 SFT가 로컬에서 만든다 — 논문이 이 성질을 이점으로 주장한다("생성된 데이터는 CPT에 재사용되거나 임의의 base 모델에 적용될 수 있다" [SEAL §B.8, p.22]).

배포의 실제 걸림돌은 계산이 아니다. **RL 루프가 문맥마다 라벨된 downstream 과제를 요구한다.** 서빙 fleet에는 그런 라벨이 없다. 논문이 스스로 이름 붙인다 — 이 결합이 "SEAL의 RL 학습이 라벨 없는 코퍼스로 확장되는 것을 막는다" [SEAL §5, p.9].

## 20.8 한계와 bridge-out

논문이 명시한 문제는 셋이다 [SEAL §5, pp.8–9]. **망각** — 순차 self-edit이 누적되면 이전 과제가 무너지는데 완화책이 구현되어 있지 않다. 제안된 방향은 회귀에 벌점을 주는 reward shaping, null-space 제약 편집, 표현 중첩, inner loop를 SFT 대신 RL로 두는 것이고 **전부 미실행**이다. **계산 부담** — self-edit 평가 하나가 모델 전체를 finetune하고 평가하므로 30–45초다. **문맥 의존 평가** — 위에서 본 라벨 oracle 문제이며, 논문의 제안은 문맥이 아직 창에 있을 때 모델이 자기 평가 문항까지 생성하게 하는 것이다.

남긴 문제 넷을 더 적는다. 3B/7B 두 점으로는 이득이 모델 크기와 함께 커지는지 판정할 수 없다 [SEAL §B.7]. teacher–student 분리 — 학생을 별도 교사의 편집으로 갱신하고 교사를 RL로 학습시키는 구성 — 는 전문이 제시되되 **한 번도 실행되지 않았다** [SEAL §3.1, p.4]. self-edit 형식이 지배적인데 7개 프롬프트로도 해결되지 않았다 [SEAL §B.11]. 그리고 §6이 제안하는 전망 — 사전학습 코퍼스 생성기, chain-of-thought 중간의 가중치 갱신, 상호작용마다 self-edit을 합성하는 에이전트 — **셋 모두 실험이 없다.**

> **[평가]** 이 장이 논문보다 강하게 읽는 것 둘. 첫째, 식 (20-3)이 실제로 최적화되지 않는다. 보상은 $\Theta_k$에 의존하지만 미분 불가이므로 논문은 그것을 $\Theta_k$에 대해 고정된 것으로 취급한다 [SEAL §3.1, p.4]. 증명되는 것은 stop-gradient를 적용한 대리 목적이지 식 (20-3) 자체가 아니고, 두 목적이 같은 최적점을 갖는다는 논증은 없다(→ ch05 §05.7.1). 둘째, **상태의 수명이 논의되지 않는다.** 질의가 끝난 뒤 adapter를 사용자별로 유지하는지, 병합하는지, 축출하는지, 버전을 매기는지 — $\Theta$-경로 서빙의 첫 번째 질문인데 논문에 진술이 없다. 가장 가까운 문장이 초록의 "지속적인 가중치 갱신"이다.

**ch06이 세운 제약이 이 루프에 걸리는 방식.** ch06 §06.8이 이미 못 박은 대로, accumulate의 유계성 결과 (6-2)를 떠받치는 네 전제가 SEAL에서 전부 깨진다 — 라운드마다 재초기화하지 않고 살아 있는 checkpoint를 증분 편집하며, 공변량이 동결되어 있지 않고, 보상으로 후보를 거른다. 따라서 $\pi^2/6$은 이 루프의 안전 증서가 아니다. SEAL 쪽에서 확인할 수 있는 것은 셋이다. (i) 관측된 열화는 **라운드 축이 아니라 문맥 축**에 있다 — [SEAL Fig. 6]은 $k$가 아니라 순차 self-edit 8회에 대한 것이다. (ii) 라운드 축은 $k=2$에서 멈추므로 자기생성 반복의 지속가능성이 관측된 적이 없다. (iii) 스케줄로 보면 single-passage는 $\Theta'$을 매번 버리므로 ch06의 accumulate도 replace도 아니고, **CPT 레짐만이 생성물을 코퍼스로 누적한다.** 판정은 ch22가 한다.

**경로 간 인용.** $W$-경로는 조밀하게 인용되되 **한 갈래만**이다 — TTT 계열(Sun 2020·2024, Gandelsman, Akyürek 2025)과 자기참조 network 계열(Schmidhuber 1987·1992, Irie 2022)이며, ch16의 Generative Adapter는 인용될 뿐 아니라 baseline으로 돌아간다. 반면 **fast-weight sequence layer 계열은 전무하다** — Titans(2501.00663)도 Atlas(2505.23735)도 DeltaNet도 Mamba도 선형 attention도 없고, "fast weights"·"linear attention"·"state space"라는 어구 자체가 등장하지 않는다. 이 목록에서 **Nested Learning은 빼야 한다** — 2512.24695는 이 장이 읽는 판본(v2, 2025-09-18)보다 뒤에 나왔으므로 연대가 강제한 침묵이고, 인용 부재를 이 논문의 선택으로 셀 수 없다(→ ch18). 나머지 다섯은 전부 v2보다 앞서 나왔고, 따라서 선택된 침묵이다. $E$-경로는 **완전한 침묵**이다: MemGPT·RAG·vector store·Letta·Mem0·Zep 어느 것도 85개 참고문헌과 본문에 없다. 생물학과 CLS도 **0회**이고, catastrophic forgetting은 1989년·2014년의 고전적 틀로만 인용된다. 용량 이론도 걸리지 않는다.

> **[평가]** **corpus 전체에서 한 경로가 다른 경로들의 존재를 모르는 가장 강한 사례가 이 논문이다.** 결정적인 것은 침묵의 위치다. SEAL §6은 "확장된 상호작용에 걸쳐 작동하며 변화하는 목표에 동적으로 적응하는 에이전트 시스템"을 목표로 제시하고, "상호작용 뒤 에이전트가 가중치 갱신을 촉발하는 self-edit을 합성"하자고 쓴다 [SEAL §6, p.9]. **그것은 $E$-경로의 문제 진술 그 자체이며, $E$-경로는 그 문제를 가중치가 아니라 레코드에 써서 푼다.** SEAL은 그 대안을 인용하지도 비교하지도 않고, 왜 레코드 대신 가중치여야 하는지 논증하지 않는다. 연대도 변명이 되지 않는다 — Letta STC는 동시대(2025-04)이고 MemGPT는 1년 반 넘게 앞선다. 같은 자리에서 생물학 쪽 침묵도 기록해 둔다: 이 논문은 sleep-time의 문제(오프라인 통합, 망각, 지속 적응)에 전부 도달하면서 **"sleep"이라는 단어를 한 번도 쓰지 않는다.** CLS(1995)도 EWC(2016)도 Deep Generative Replay(2017)도 이 논문보다 8년 이상 앞서므로 이 침묵 역시 연대가 아니라 선택이다. 이 장에서 연대가 강제한 침묵으로 분류되는 것은 Nested Learning 한 건뿐이다.

ch21이 받아 가는 것은 둘이다. 첫째, **완화책 없는 망각.** [SEAL Fig. 6]이 보인 하락은 SEAL이 문제로 등록만 하고 남긴 것이고, `Language Models Need Sleep`의 consolidation·dreaming 수명주기가 정확히 그 자리를 메우겠다고 주장한다 — 자기 생성에 replay와 증류를 더하면 무엇이 달라지는지가 다음 장의 질문이다. 둘째, **$\mathcal{H}_k$의 확장.** SEAL의 $\mathcal{H}_k$는 문맥 하나이고 라운드는 둘이다. 경험이 누적되고 라운드가 반복될 때 (U-$\Theta$)가 어떤 모양이 되는지는 이 논문이 답하지 않는다. 반증 축 — $n{=}2067$에서 heuristic에 지는 첫 신호, 그리고 가중치에 얼마나 계속 쓸 수 있는가 — 는 ch22가 받는다.

## 요약

- SEAL은 (U-$\Theta$)를 **두 시계로 두 번** 실행한다. inner는 문맥마다 self-edit으로 SFT하고, outer는 그 결과 점수를 보상으로 self-edit **생성 정책**을 민다. (U-W)도 (U-E)도 인스턴스를 갖지 않는다.
- **self-edit**은 모델이 자기 가중치 갱신의 내용을 지정해 쓴 자연어 산출물이다. 실증된 형식은 둘 — 함의 목록(데이터)과 augmentation·학습률·epoch·loss mask 명세(갱신 규칙). 후자가 표준형 $\mathrm{gen}(\cdot;B_s)$에 대한 이 논문의 가장 날카로운 확장이다.
- 판별식 네 조건을 전부 통과하며, 읽기는 $W=\varnothing$·$\mathrm{ret}(E,q)=\varnothing$인 **corpus에서 가장 순수한 $\Theta$-경로 읽기**다. wake 프롬프트에 passage가 없다.
- **초록의 "GPT-4.1 능가"는 세 열 중 한 열에서만 성립한다** — $n{=}1$의 47.0 대 46.3뿐이고, 두 CPT 열에서는 GPT-4.1이 이긴다(59.4 대 58.2, 49.2 대 46.4). 논문 §4.2는 이를 진술하고 초록은 범위 조건을 달지 않는다.
- 표의 열은 스케일 계열이 아니다. $n{=}1$은 LoRA, $n{=}200$·$2067$은 full finetuning이고, 후자에서만 base tensor가 움직인다. outer 루프는 두 도메인 모두 LoRA이므로 **RL이 배우는 것은 어느 경우에도 backbone을 움직이지 않는다.**
- 자기 표가 논지를 세 번 깎는다 — RL 없는 프롬프트 둘(49.3, 49.4)이 RL 헤드라인(47.0)을 넘고, 정의적 기여인 TTT 보상을 지워도 45.6에 약 72배 싸며, 모델이 형식을 스스로 정하면 18.9로 무너진다. few-shot 벤치는 TTT 가해성으로 필터된 19개 과제여서 Oracle의 100%가 구성상 참이다.
- 비용 4종 중 셋이 사실상 비어 있다. **계산을 질의 이전으로 옮기자는 논문이 옮기는 계산도 사는 절감도 재지 않았고**, 상각식 (A)는 닫히지 않는다.
- 서빙 함의는 두 뿔이다 — per-document adapter는 shared-weight batching을 깨고, CPT는 배칭을 되살리되 격리를 잃고 그 레짐에서 heuristic에 진다. 실제 걸림돌은 계산이 아니라 문맥마다 라벨을 요구하는 보상 oracle이다.

## 자가 점검 체크리스트

- [ ] 식 (20-1)–(20-2)를 표준형 (U-$\Theta$)와 항별로 대조해 $\mathrm{gen}$이 $\Theta$ 자신이라는 점과 outer gradient가 $\log p(\mathcal{R}_k|c)$를 통해 흐른다는 점을 말할 수 있고, 식 (20-2)가 논문에 없는 이 책의 복원임을 밝힐 수 있다.
- [ ] SEAL의 "inner loop"가 NM·Titans 계열의 inner loop와 무엇이 다른지 한 문장으로 말하고, $s$·epoch·$k$ 세 첨자를 각각 어디에 붙이는지 구별할 수 있다.
- [ ] 표 20-4의 세 열에서 "GPT-4.1 능가"가 어느 칸에만 성립하는지 수치로 말하고, 열 사이 비교가 스케일 계열이 아닌 이유를 갱신 파라미터화로 설명할 수 있다.
- [ ] 표 20-5로부터 RL의 기여와 프롬프트의 기여를 분리해서 진술하고, No-Prompt 행이 "self-directed adaptation"에 대해 무엇을 반증하는지 말할 수 있다.
- [ ] 표 20-3의 빈칸 넷이 각각 왜 비어 있는지 말하고, 이 논문에 대해 "효율적"이라고 쓸 수 없는 이유를 식 (A)의 두 항으로 설명할 수 있다.
- [ ] ch06의 accumulate 유계성 결과가 이 루프에 적용되지 않는 이유를 네 전제로 대고, SEAL에서 관측된 열화가 라운드 축이 아니라 문맥 축에 있다는 사실을 [SEAL Fig. 6]으로 설명할 수 있다.
- [ ] (Rosetta) "모델 재배포" 행과 "shared-weight batching" 행이 이 논문의 두 레짐에서 각각 어떻게 작동하는지 말하고, 그 차이가 $C_{\text{cap}}$과 롤백 단위에 무엇을 하는지 설명할 수 있다.
