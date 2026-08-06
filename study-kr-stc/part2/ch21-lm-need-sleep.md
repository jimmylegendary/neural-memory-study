# ch21. Language Models Need Sleep — 증류와 dreaming

## 21.1 Bridge-in: 전작이 남긴 문제

이 장의 bridge-in은 이 책에서 유일하게 교과서적으로 성립한다. 독자는 앞의 다섯 장에서 그 반대를 봤다. ch14의 Letta `Sleep-time Compute`는 상속할 open question이 아예 없어 **상속 없는 시작**으로 써야 했고, ch15의 Mem0·ReasoningBank은 같은 경로에서 이 라인의 이름을 만든 논문을 인용하지 않았으며, ch17의 `Do LMs Need Sleep?`은 다른 두 경로를 한 문장씩 언급하면서 정작 자기 경로의 주류 계보에 침묵했고, ch18의 Nested Learning은 상속 진술 자체가 없었다. 이 장은 다르다. [LM Need Sleep] (*Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories*, arXiv:2606.03979)은 전작이 남긴 공백을 **이름과 함께 지목하고 그 자리에서 시작한다.**

전작이 적은 문장은 ch18이 이미 인용했다 — [NL]은 오프라인 consolidation을 개념으로 소개한 뒤 "본 연구에서는 첫 단계, 즉 온라인 과정으로서의 memory consolidation에 집중한다"고 범위를 스스로 닫았다. 이 논문은 그 문장을 받아 이렇게 연다.

> "Behrouz et al. (2025), recently, presented the Nested Learning (NL) paradigm and aimed at the first form of memory consolidation (i.e., online consolidation). ... In this paper, we focus on the second type of consolidation: i.e., offline consolidation via sleep." [LM Need Sleep §1, p.3]

양쪽이 다 닫혀 있다. 전작이 "여기까지"라고 적었고 후속이 "그다음"이라고 적었다. 상속은 여기서 끝나지 않는다. 이 논문은 전작이 **자기 실패 조건까지** 적어 둔 것을 그대로 물려받는다.

> "While actively updating this memory system can enhance the resistance to CF, the CF can happen when the update period of all models matched at some point (Behrouz et al. 2025). Therefore, it is crucial that before each update of a memory block, a mechanism consolidates the abstracted knowledge of that block to more stable parameters." [LM Need Sleep §2.2, p.6]

이 문장이 §21.3의 스케줄을 통째로 결정한다. sleep은 임의의 시각이 아니라 **어떤 블록이 자기 가중치를 덮어쓰기 직전**에 발화한다. 이유가 상속된 진단이기 때문이다.

논문은 온라인 전용 consolidation의 결함을 셋으로 정리한다. (i) 같은 추상화 수준에서 지식을 옮기므로 "it still uses the same amount of model's capacity", (ii) "Is selective and retrieval-dependent" — 활발히 회수되는 기억만 강화된다, (iii) "Depends on the context: the update caused by the online consolidation is based only on the context of the model, and so misses the higher-level understanding of new knowledge with existing one" [LM Need Sleep §1, p.3]. 세 항목 각각이 뒤에서 기제 하나씩에 대응한다 — (i)에 parameter expansion, (ii)에 teacher 분포로부터의 재표집, (iii)에 Dreaming이 대응한다.

이차 상속도 명시적이다. Dreaming 절은 SEAL(Zweiger et al. 2025, arXiv:2506.10943, → ch20)이 남긴 장애물 셋을 축자적으로 옮겨 적는다: "(1) Due to the cost of supervised fine-tuning (SFT) in SEAL's inner-loop, it is limited to small number of self-edits (dreams in our terminology). (2) Potential catastrophic forgetting as the cause of iterative self-improvement in sleep periods. (3) The sampling process only samples from the existing knowledge space of the model" [LM Need Sleep §3.4, p.9]. 두 번째 항목은 SEAL이 스스로 보고한 한계이며 그렇게 귀속되어 있다.

> **[평가]** 상속 진술이 성립하려면 두 가지가 다 있어야 한다 — 전작이 공백을 자기 언어로 적었을 것, 그리고 후속이 그 공백을 지목했을 것. 이 corpus에서 둘이 함께 성립하는 사례는 이 쌍이 거의 유일하다. 그런데 이 예외가 왜 성립하는지를 적지 않으면 서술이 정직하지 않다. **[NL]과 [LM Need Sleep]의 제1저자가 같은 사람이다.** 사슬이 이어진 것이 아니라 한 연구 프로그램 안에서 이어졌다. 이 책이 §21.8에서 보일 침묵의 지도 — 이 논문이 W-경로의 sleep 문헌을 한 편도 인용하지 않고 E-경로를 두 문장으로 닫는다는 사실 — 와 나란히 놓으면, 계보가 작동하는 단위는 경로가 아니라 저자군이다. 상속의 예외성 자체가 단절의 증거다.

## 21.2 문제의식

논문이 스스로 세운 문제는 배포 이후의 정지다. LLM은 "are largely static after their initial deployment ... their knowledge and skills become progressively stale, operating with a fixed \"knowledge cutoff\" date" [LM Need Sleep §1, p.1]. 기존 처방 둘을 이름과 함께 기각한다. 확장된 데이터로 재-pre-training하는 것은 "computationally expensive and impractical for frequent updates"이고, 지속적 파라미터 갱신 또는 low-rank adaptation은 "with iterative updates often results in Catastrophic Forgetting"이다 [LM Need Sleep §1, p.1].

그 위에서 논문이 이탤릭으로 세운 질문이 이 장 전체의 축이다.

> "How the model can effectively transfer the fragile short-term memories into more stable long-term knowledge?" [LM Need Sleep §1, p.2]

> **[해설]** 이 책의 어휘로 옮기면 문장 하나가 남는다. 이번 세션의 지식을 담은 $W$가 있고 pre-training을 담은 $\Theta$가 있을 때, **$W \to \Theta$ 연산자는 무엇이고 언제 도는가.** ch01이 세운 세 층 프레임에서 이 질문은 자명하게 읽히지만, 원 논문에는 그 층 구분이 없다 — 논문은 처음부터 끝까지 모든 것을 "parameters"라고 부른다. 이 장이 하는 일의 절반은 그 한 단어를 두 층으로 갈라 읽는 것이다.

논문의 재구획은 더 넓다. train/test 분할 자체를 폐기한다. [LM Need Sleep §3.1]의 박스 진술이 그것이다 — "For a continual learner, there is no boarder and clear distinction between training and test time"(원문 오기 그대로) [LM Need Sleep §3.1, p.3]. 그 자리에 wake/sleep lifecycle을 놓고, 선행 문헌 전체가 그 구분에 묶여 있다고 주장한다: "To the best of our knowledge, the existing literature (including sleep-inspired studies) remains firmly anchored in the conventional distinction of training and testing phases" [LM Need Sleep §2.3, p.6].

마지막으로 catastrophic forgetting의 원인 자체를 바꿔 놓는다. CF는 "an inherent cause of model's limited capacity (e.g., number of parameters), where parameters need to be overridden to incorporate the new knowledge"이며 [LM Need Sleep §3.2, p.7], 부록이 이를 정식으로 선언한다: "This reframes catastrophic forgetting as a problem of insufficient capacity rather than of sampling distribution, and addresses it by gradual parameter growth between consolidation steps rather than by re-conditioning a fixed teacher" [LM Need Sleep App. A.4 (i), p.24].

이 재정의가 기제를 강제한다. CF가 표집 분포의 문제라면 replay가 답이고(→ ch07), 용량의 문제라면 답은 용량을 늘리는 것이다. 논문은 후자를 택했고, 그래서 이 장의 sleep은 **가중치를 고쳐 쓰는 절차가 아니라 가중치를 늘린 뒤 새로 생긴 자리에만 쓰는 절차**가 된다.

> **[평가]** CF를 용량 문제로 재정의하면서 용량을 측정한 문헌을 한 편도 인용하지 않는다. `How much do LMs memorize`(arXiv:2505.24832, 2025-05)도 `Understanding LoRA as Knowledge Memory`(arXiv:2603.01097, 2026-03)도 이 논문의 v2 시점(2026-07-10)보다 앞서 나왔고 둘 다 참고문헌에 없다(→ ch09). 용량 주장을 용량 문헌과 접촉 없이 세운 것이며, 얼마의 rank가 얼마의 망각을 막는지에 대한 형식적 진술도 논문 안에 없다. 이 공백의 판정은 ch22가 받는다.

## 21.3 Core mechanism (통일 표기)

이 논문의 기제는 NM ch17이 단계별로 이미 전개했다. 이 절은 그것을 반복하지 않고, 세 층 프레임이 강제하는 세 가지만 한다 — wake가 어느 층을 움직이는가, sleep의 쓰기가 표준형 (U-$\Theta$)와 어디서 갈라지는가, 그리고 그 둘의 **순서**.

먼저 읽기를 고정한다. 식 (R)은 이 논문에서 다음으로 붕괴한다.

$$
\hat y \;=\; f\big(q;\ \Theta,\ W,\ \varnothing\big)
\tag{21-1}
$$

$E$가 비어 있고, 논문이 그렇게 적는다 [LM Need Sleep App. A.3, p.23]. 이 사실 자체가 §21.8의 재료다.

### 21.3.1 wake — 주파수 사다리와 (U-W)

$\Theta$는 이 논문에서 한 덩어리가 아니다. 갱신 주파수로 분할된 블록 사슬 $\theta^{(1)},\dots,\theta^{(K_{\mathrm{lv}})}$이며 $f_1 \ge \dots \ge f_{K_{\mathrm{lv}}}$이다 [LM Need Sleep §2.1, Def. 1]. sequence layer(attention의 KV cache 또는 Titans/TTT류의 고정 크기 memory)가 $f \to \infty$ 끝점에 있고, 동결된 pre-trained MLP가 $f = 0$ 끝점에 있다 [LM Need Sleep §2.2, p.5-6]. 정의와 그 계보는 ch18이 소유하므로 여기서 다시 세우지 않는다.

유한 주파수 블록의 wake 갱신은 NM 식 (M5)의 형태다.

$$
\theta^{(\ell)}_{i+1} = \theta^{(\ell)}_{i} - e_{i,\ell},
\qquad
e_{i,\ell} = \begin{cases}
\displaystyle\sum_{t=i-C^{(\ell)}+1}^{i} \eta^{(\ell)}_t\, \varepsilon\big(\theta^{(\ell)}_t; x_t\big) & i \equiv 0 \pmod{C^{(\ell)}}\\[2pt]
0 & \text{otherwise}
\end{cases}
\tag{21-2}
$$

*원문 [LM Need Sleep Eq. 2]의 합은 하한이 $t = i - C^{(\ell)}$(포함)이어서 크기 $C^{(\ell)}$인 chunk에 대해 $C^{(\ell)}+1$개 항을 더한다. 이 책은 NM 식 (M5)의 하한 $t = i - C^{(\ell)} + 1$을 쓴다. 원문의 off-by-one이며, 조용히 고치지 않고 여기 적어 둔다.*

$\varepsilon(\cdot)$은 임의 optimizer의 오차 항이고 chunk 크기는 $C^{(\ell)} := \max_{\ell'} C^{(\ell')} / f_\ell$로 정의된다 [LM Need Sleep §2.2]. 원문은 이 오차 항을 $f(\cdot)$로 쓰는데 같은 식의 위첨자에서 $f_\ell$을 주파수로 쓰고 있어 기호가 겹친다 — 대응표에 기록한다.

핵심은 판정이다. **식 (21-2)의 갱신은 (U-W)다.** 시계가 살아 있는 시퀀스의 토큰이고, 산출물이 문맥 경계를 넘어 존속하지 않으며, 배포 아티팩트를 바꾸지 않는다. ch18이 [NL]에 대해 내린 판정 — 주파수를 낮춰 얻는 것은 느리게 움직이는 $W$이지 $\Theta$가 아니다 — 이 그대로 적용된다. 논문이 이 블록들을 "parameters"라 부른다는 사실은 판정을 바꾸지 않는다.

### 21.3.2 sleep — 확장, seeding, 리셋

sleep은 스케줄로 발화한다. 모든 $b \in \mathbb{N}$에 대해 step 인덱스 $\{C^{(1)}b,\ C^{(2)}b,\ \dots,\ C^{(K_{\mathrm{lv}})}b\}$에서 돈다 [LM Need Sleep §3.2, p.7]. **학습되지 않는다.** 주파수가 중첩되므로 빠른 블록은 느린 이웃이 한 번 갱신되기 전에 여러 번 통합되는데, 논문 자신의 예시가 $f=1\text{K}$가 $f=10\text{K}$에 먹이는 경우로 느린 갱신 1회당 통합 10회다 [LM Need Sleep §3.2, p.7].

한 번의 sleep은 세 조각으로 이루어진다. 순서가 그림으로도 명시되어 있다 — Figure 2는 모델이 **먼저** 자기 파라미터 수를 늘리고 **그다음** 높은 주파수 memory에서 낮은 주파수 memory로 추상화를 seeding하는 두 단계로 통합을 그린다 [LM Need Sleep Fig. 2, p.7]. 첫째, 받는 쪽 블록 $\ell^*$가 **커진다**.

$$
\theta^{(\ell^*)} \;\leftarrow\; \theta^{(\ell^*)} \cup \big\{A^{(\ell^*),\,n_{\ell^*}+1},\ B^{(\ell^*),\,n_{\ell^*}+1}\big\},
\qquad
A \in \mathbb{R}^{d\times d_{\mathrm{low}}},\ B \in \mathbb{R}^{d_{\mathrm{low}}\times d},\ d_{\mathrm{low}} \ll d
\tag{21-3}
$$

$n_{\ell}$은 블록 $\ell$의 expert 수이고 매 sleep마다 1 증가한다. 각 블록은 "without loss of generality" sparse MoE로 가정되며 router $\mathcal{R}^{(\ell)}$이 expert 집합 위에서 dispatch한다 [LM Need Sleep §3.2, p.7]. **$\dim\Theta$가 매 sleep마다 엄격히 커진다** — 표준형 (U-$\Theta$)로부터의 가장 큰 이탈이 여기다[본서 판단].

둘째, 새로 생긴 자리에 쓴다. 이것이 이 장이 소유하는 첫 번째 정의다.

> **Knowledge Seeding**(upward distillation)은 확장 **전**의 자기 자신을 teacher로, 확장 **후**의 자기 자신을 student로 삼아, 새로 활성화된 저랭크 expert에만 gradient를 흘려 teacher의 출력 분포와 표집 거동을 재현하도록 학습시키는 절차다. teacher가 student보다 **작다**는 점에서 고전적 distillation의 방향이 뒤집혀 있다.

논문은 이를 세 단계 (a)/(b)/(c)로 적는다 [LM Need Sleep §3.3, p.7-8]. (a) 보내는 쪽 블록 $\ell^*-1$의 갱신 $e_{i,\ell^*-1}$을 **계산만 하고 적용하지 않는다**. (b) teacher는 갱신 전 $\Theta$, student는 예정 갱신과 새 expert를 얹은 $\Theta_{\exp}$로 두고 목적함수를 돈다. (c) 갱신을 실제로 적용하고, 새 expert를 활성화하고, **보내는 쪽 블록에 그동안 쌓인 저랭크 expert들을 리셋한다** — 논문의 표현으로 "the low-rank experts previously added to $\mathrm{MLP}^{(f_{\ell^*-1})}$ are reset (synaptic pruning)" [LM Need Sleep §3.3(c), p.8].

목적함수는 두 항의 결합이다. distillation 항은 GKD의 on-policy 혼합이다.

$$
\adjustbox{max width=\linewidth}{$\displaystyle
\mathcal{L}(\Theta,\Theta_{\exp}) = (1-\lambda_{\mathrm{on}})\,\mathbb{E}_{(x,y)\sim\mathcal{D}}\Big[\mathcal{F}\big(\mathrm{LM}_{\Theta}\,\|\,\mathrm{LM}_{\Theta_{\exp}}\big)(y|x)\Big] + \lambda_{\mathrm{on}}\,\mathbb{E}_{x\sim\mathcal{D}}\,\mathbb{E}_{y\sim\mathrm{LM}_{\Theta_{\exp}}(\cdot|x)}\Big[\mathcal{F}\big(\mathrm{LM}_{\Theta}\,\|\,\mathrm{LM}_{\Theta_{\exp}}\big)(y|x)\Big]
$}
\tag{21-4}
$$

$\mathcal{D}$는 "by sampling from the teacher model"로 만들어진다. 같은 문단이 설계 결정 둘을 명시한다 — student의 표집 분포를 관통해 backprop하지 않으며, "we freeze all the parameters in the student model and only updates the expanded parameters" [LM Need Sleep §3.3, p.8]. 뒤의 것이 이 쓰기를 rank-bounded로 만든다(GKD 자체는 → ch04)[본서 추론].

보상 항은 **Learning to Imitate**(LTI)다. 약어 LTI는 NM ch07의 linear time-invariant와 무관하며 이 장 안에서만 이 뜻으로 쓴다. teacher가 생성한 $d^{(i)}$의 무작위 prefix를 student에게 완성시키고 두 성분을 섞는다.

$$
\adjustbox{max width=\linewidth}{$\displaystyle
R\big(\hat d^{(i)};d^{(i)};\mathrm{LM}_{\Theta_{\exp}}\big) = \rho_{r}\, R_{\mathrm{sem}}\big(\cdot\big) + (1-\rho_{r})\, R_{\mathrm{abs}}\big(\cdot\big),
\qquad
R_{\mathrm{abs}} = \begin{cases}1 - \dfrac{z(\hat d^{(i)},d^{(i)})}{\max\{|\hat d^{(i)}|,|d^{(i)}|\}} & z \le z_0\\[6pt] 0 & \text{otherwise}\end{cases}
$}
\tag{21-5}
$$

$z(\cdot,\cdot)$은 Levenshtein 거리, $z_0$는 유사도 문턱이다 [LM Need Sleep Eq. 3, Eq. 4]. $R_{\mathrm{sem}}$은 이진값이며 "we use a reward model that is frozen"으로만 서술된다 — **그 reward model은 논문 어디에서도 이름도 크기도 밝혀지지 않는다** [LM Need Sleep §3.3, p.8]. 두 항을 합친 최종 목적함수는 다음이다.

$$
\mathcal{L}_{\mathrm{KS}}(\Theta,\Theta_{\exp}) = \mathbb{E}_{x\sim\mathcal{D}}\Big[(1-\lambda_{\mathrm{KD}})\,\mathbb{E}_{y\sim\mathrm{LM}_{\Theta_{\exp}}}\big[R(y)\big] - \lambda_{\mathrm{KD}}\,\mathbb{E}_{y\sim\mathrm{LM}_{\Theta_{\exp}}}\,\mathcal{F}\big(\mathrm{LM}_{\Theta}\,\|\,\mathrm{LM}_{\Theta_{\exp}}\big)(y|x)\Big]
\tag{21-6}
$$

식 (21-4)와 (21-6)이 이 논문의 두 중심 목적함수인데 **둘 다 원문에서 번호가 없다** — 번호는 보상의 세부인 Eq. 3–5에만 붙어 있다. 그리고 원문은 식 (21-6)의 divergence를 $\mathcal{D}$로 쓰면서 같은 줄에서 $\mathcal{D}$를 표집 데이터셋으로도 쓰고, 한 문단 앞에서는 같은 divergence를 $\mathcal{F}$로 썼다. 이 책은 divergence를 $\mathcal{F}$로 통일한다. teacher를 정의하는 문장 자체에도 층 첨자가 어긋나 있다 — [LM Need Sleep §3.3(b)]는 받는 쪽 블록을 두고 "$\mathrm{MLP}^{(f_{\ell^*})}(\cdot)$ still uses its original parameters $\theta^{(f_{\ell^*-1})}$"라고 적어, 블록 $\ell^*$의 파라미터를 $\ell^*-1$의 것으로 지시한다 [LM Need Sleep §3.3(b), p.8]. 이 절이 세운 **보내는 쪽 $\ell^*-1$ / 받는 쪽 $\ell^*$** 구획으로만 읽어야 문장이 성립하며, 이 책은 그렇게 읽었다. 더 무거운 결함이 하나 남는다: **식 (21-6)은 보상에서 divergence를 뺀 형태이므로 최대화 대상인데, 같은 페이지의 산문은 저랭크 파라미터가 "optimized via the Knowledge Seeding objective to minimize the distillation loss"라고 적는다** [LM Need Sleep §3.3(b), p.8]. 논문은 최적화 방향을 끝내 명시하지 않는다.

### 21.3.3 Dreaming

sleep의 두 번째 단계는 이 장이 소유하는 두 번째 정의다.

> **Dreaming**은 sleep 라운드의 두 번째 단계로, 모델이 과제 문맥 $c_{\mathrm{task}}$를 조건으로 자기 샘플 $\mathrm{DREAM}^{(i)}$을 생성하되 MoE router가 의도적으로 무작위 expert를 하나 더 고르게 하고, 생성된 후보를 gradient norm으로 거른 뒤 살아남은 것을 각각 **격리된** 모델 사본에 LoRA SFT로 적용하는 절차다.

$$
\adjustbox{max width=\linewidth}{$\displaystyle
\{\mathrm{DREAM}^{(i)}\}_{i=1}^{n_{\mathrm{dr}}} \sim \mathrm{LM}_{\Theta}(\cdot\,|\,c_{\mathrm{task}}),
\qquad
g^{(i)}_{\mathrm{DR}} = \nabla_{\Theta}\mathcal{L}_{\mathrm{SFT}}\big(\mathrm{DREAM}^{(i)},\Theta\big),
\qquad
\theta'^{(i)} \leftarrow \mathrm{SFT}\big(\theta^{(i)},\mathrm{DREAM}^{(i)}\big)
$}
\tag{21-7}
$$

선택은 $\|g^{(i)}_{\mathrm{DR}}\|$ 상위 $n_{\mathrm{top}}$개에 다양성 유지를 위한 무작위 $n_{\mathrm{rand}}$개를 더한 것이다. 무작위 expert 선택의 목적은 논문 표현으로 "incorporates random irrelevant knowledge to the dreaming"이다 [LM Need Sleep §3.4, p.10]. 생성 정책은 SEAL의 이진 개선 보상을 그대로 물려받아 ReST$^{\mathrm{EM}}$으로 학습된다.

$$
R\big(\mathrm{DREAM}^{(i)},\tau(\cdot),\mathrm{LM}_{\theta^{(i)}}\big) = \begin{cases}1 & \mathrm{LM}_{\theta'^{(i)}}\text{가 }\tau\text{ 아래에서 }\mathrm{LM}_{\theta^{(i)}}\text{을 개선하면}\\ 0 & \text{그 외}\end{cases}
\tag{21-8}
$$

$\tau(\cdot)$는 "a measure to asses the performance in the downstream evaluation"이다 [LM Need Sleep Eq. 5, §3.4, p.9].

> **[평가]** 식 (21-8)이 이 논문의 lifecycle 주장과 정면으로 부딪힌다. [LM Need Sleep §3.1]의 박스 정의는 자고 있는 모델이 "receives minimal (or none) input data"라고 적는데, 식 (21-8)은 sleep 중에 과제 쌍 $(c_{\mathrm{task}}, \tau)$와 **평가 가능한 downstream 측정**이 주어져 있기를 요구한다. ARC 프로토콜에서는 dream이 아예 "from the few-shot demos"로부터 생성된다 [LM Need Sleep App. B.3, p.25]. 즉 이 sleep은 task-aware이고, 새벽 3시에 유휴 상태로 도는 배포 모델에는 $\tau$가 없다. 논문은 그때 무엇이 이진 개선 보상을 대체하는지 말하지 않는다.

### 21.3.4 표준형 (U-$\Theta$)와의 차이

한 번의 sleep 라운드 $k$를 표준형으로 환원하면 다음이 된다.

$$
\Theta_{k+1} \;=\; \mathrm{Reset}^{(\ell^*-1)}\Big(\ \mathrm{Grow}\big(\Theta_k\big) \;-\; \eta_\Theta\,\mathcal{P}_{\{A,B\}}\,\nabla_{\Theta}\,\mathcal{L}_{\mathrm{KS}}\big(\mathcal{R}_k;\Theta_k\big)\ \Big),
\qquad
\mathcal{R}_k = \mathcal{D} \sim \mathrm{LM}_{\Theta_k}
\tag{21-9}
$$

$\mathrm{Grow}(\cdot)$은 식 (21-3)의 차원 증가 매장, $\mathcal{P}_{\{A,B\}}$는 새로 활성화된 저랭크 expert로의 사영, $\mathrm{Reset}^{(\ell^*-1)}(\cdot)$은 보내는 쪽 블록의 누적 expert 소거다. 표준형 (U-$\Theta$)와의 차이는 여섯 개이고 전부 회계에 걸린다.

1. **차원이 상수가 아니다.** $\Theta_{k+1}$은 $\Theta_k$보다 엄격히 큰 공간에 산다. 표준형은 고정 크기 $\Theta$에서 gradient를 뺀다.
2. **gradient가 직전까지 존재하지 않던 부분공간으로 사영된다.** 나머지가 전부 동결되므로 sleep 뒤에도 옛 $\Theta$는 비트 단위로 동일하다. EWC류 정규화보다 훨씬 강한 비간섭 보장이며, 대가는 옛 가중치를 아예 건드리지 않는 것이다(→ ch07의 정규화 계열과 대조).
3. **학습 집합이 자기 생성이다.** $\mathcal{R}_k$가 $\mathrm{LM}_{\Theta_k}$에서 표집되므로 표준형의 $\mathcal{H}_k$는 저장된 경험이 아니라 **모델이 자기 자신에게서 다시 뽑는 분포**다. 실데이터 replay buffer가 없다(반복 시 무엇이 무너지는지는 → ch06).
4. **목적함수가 task loss가 아니다.** 표적이 과거 자기 출력 분포이며, 토큰 수준(GKD)과 표집 수준(LTI) 양쪽에서 그렇다.
5. **파괴 연산자가 있다.** 표준형에 삭제는 없다. 매 sleep이 리셋으로 끝나며, 이것이 $C_{\text{cap}}$을 유계로 만드는 유일한 기제이자 $\rho$를 잴 수 있었던 유일한 자리다.
6. **고정 시계로 발화하고, 한 주기에 여러 번 돈다.** 연속한 블록 쌍마다 통합이 일어나므로 주기당 $K_{\mathrm{lv}}-1$회다 [LM Need Sleep App. A.4 (ii), p.24]. 그리고 Dreaming이 같은 주기 안에서 **다른 $\mathcal{R}_k$·다른 loss·다른 학습 대상**으로 (U-$\Theta$)를 한 번 더 인스턴스화한다.

> **[평가]** 이 여섯 중 1·5번이 회계에서 가장 비싸다. 차원이 변하고 상태가 지워지는 절차는 "checkpoint 크기 = $C_{\text{cap}}$"이라는 Rosetta 대응(→ ch01)을 라운드마다 다시 계산하게 만든다.

### 21.3.5 표기 대응표

원 논문 기호 → 이 책 기호. 방향은 항상 이쪽이며, 본문 수식에는 왼쪽 열이 등장하지 않는다.

표 21-1. [LM Need Sleep]의 표기 대응

| 원 논문 기호 | 이 책 기호 | 주의 |
|---|---|---|
| $\gamma$ — 보상 혼합 계수 [Eq. 3] | $\rho_r$ | NM은 window gate 충돌 때문에 $\rho$로 개명했는데, 이 책은 $\rho$를 망각·열화율로 예약했다. 두 개명이 겹치므로 $\rho_r$로 확정한다 |
| $\alpha$ — distill 대 reward 계수 | $\lambda_{\mathrm{KD}}$ | retention gate $\alpha_t$와 충돌 |
| $\lambda$ — GKD on-policy 비율 | $\lambda_{\mathrm{on}}$ | outer weight decay $\lambda$와 충돌 |
| $r(\cdot),\ r_{\mathrm{sem}},\ r_{\mathrm{abs}}$ — 보상 | $R(\cdot),\ R_{\mathrm{sem}},\ R_{\mathrm{abs}}$ | $r$은 LoRA rank 전용. 같은 논문이 rank를 $r=64$로 쓰므로 충돌이 원문 내부에 있다 [Table 5] |
| $C$ — dreaming 과제 문맥 [§3.4] | $c_{\mathrm{task}}$ | 삼중 충돌: chunk 크기 $C$, 용량 $C_{\text{cap}}$, 그리고 같은 논문이 두 절 앞에서 쓰는 $C^{(\ell)}$ |
| $\mathcal{D}$ — 데이터셋 겸 divergence | $\mathcal{D}$(데이터셋), $\mathcal{F}$(divergence) | 한 줄 안에서 같은 기호를 두 뜻으로 쓴 원문 결함 |
| $f(\cdot)$ — optimizer 오차 항 [Eq. 2] | $\varepsilon(\cdot)$ | 같은 식의 위첨자 $f_\ell$(주파수)과 충돌 |
| $\boldsymbol\theta^{(f_\ell)}$, $C^{(\ell)}$, $e_{i,\ell}$ | $\theta^{(\ell)}$, $C^{(\ell)}$, $e_{i,\ell}$ | NM 식 (M5) 표기 |
| $L$ / $T$ — 시퀀스 길이 혼용 | $L$ | $L_w$(wake 지연)·$\mathcal{L}$(outer loss)과 구분 |
| $s_\ell$ — 블록 $\ell$의 expert 수 | $n_\ell$ | $s$는 SGD step 첨자 전용 |
| $k$ — CMS level 수; Top-$k$ 선택 크기 | $K_{\mathrm{lv}}$; $n_{\mathrm{top}}$ | $k$는 sleep 라운드 전용 |
| $m$ — 생성 dream 수; $b$ — 무작위 유지 수 | $n_{\mathrm{dr}}$; $n_{\mathrm{rand}}$ | $m_t$는 outer 1차 moment |
| $\mathbf{A},\mathbf{B}$, $d_{low}$ | $A^{(\ell),j}, B^{(\ell),j}$, $d_{\mathrm{low}}$ | ch03의 $(A,B,r)$ 위에 얹힌다 |
| $\mathcal{R}^{(f_\ell)}$ — MoE router | $\mathcal{R}^{(\ell)}$ | $\mathcal{R}_k$(sleep 라운드 학습 집합)와 첨자로 구분 |
| (원 논문 기호 아님) 환원용 확장·리셋·사영 연산자 | $\mathrm{Grow}(\cdot)$, $\mathrm{Reset}^{(\ell^*-1)}(\cdot)$, $\mathcal{P}_{\{A,B\}}$ | 이 환원을 $\Pi$로 쓰면 지속학습 성능 행렬 $\Pi$(ch07 소유, ch22가 재사용)와 충돌한다. $\Pi$와 맨 $P$를 둘 다 비워 둔다 |

## 21.4 어느 층을 언제 쓰는가

이 절이 STC 프레임에서 새로 보이는 것의 전부다. NM ch17은 이 논문을 아키텍처 라인의 완성형으로 읽었고, 그 독법에서 식 (21-2)는 상속된 CMS 기계였다. 세 층으로 갈라 읽으면 그림이 달라진다 — **식 (21-2)가 $W$층 운동이고, §21.3.2 전체가 $W \to \Theta$ 이송 연산자이며, 리셋이 그것을 복사가 아닌 이송으로 만든다.** 이 논문은 corpus에서 그 연산자를 명시적으로 가진 유일한 사례다.

이 장이 소유하는 세 번째 정의가 그것이다.

> **wake $W$ → sleep $\Theta$ 순서**란, 한 주기 안에서 (U-W)가 먼저 돌아 세션의 내용을 $W$에 적재하고, 그다음 sleep이 그 $W$의 내용을 읽어 새로 확장한 $\Theta$의 자리에 쓰고, 마지막에 $W$ 쪽의 누적분을 지우는 순서를 말한다. 지우기가 있으므로 이것은 복사가 아니라 **이송**이다.

순서가 기제다. 세 단계 (a)/(b)/(c)의 배치가 그 증거다 — 보내는 쪽의 갱신을 **계산만 하고 미룬 채** teacher/student를 세우고, 쓰기가 끝난 뒤에야 적용과 리셋을 함께 한다. 갱신을 먼저 적용했다면 teacher는 이미 사라진 상태이고 distillation의 표적이 없다. 즉 이 논문의 teacher는 다른 모델도 더 큰 모델도 아니라 **아직 덮어쓰이지 않은 몇 밀리초 전의 자기 자신**이다[본서 추론].

표 21-2. 이 값은 누가 학습하는가

| 대상 | 학습 주체 | 시계 | wake에 움직이나 | sleep에 움직이나 | 층 |
|---|---|---|---|---|---|
| backbone weights (Llama-3B/8B, Llama-3.2-1B, Qwen3-1.7B/8B) | 이 논문 이전의 누군가 | outer-loop 1회 | 아니오 | 아니오(동결) | $\Theta$ |
| CMS 사다리 자체($f_\ell$, $C^{(\ell)}$) | 손으로 설정 | 하이퍼파라미터 | 아니오 | 아니오 | — |
| 유한 주파수 블록의 base weights $\theta^{(\ell)}$ | 모델 자신, 서빙 중인 토큰으로부터 | $C^{(\ell)}$ 토큰마다 | **예** | 계산은 wake, 적용은 sleep 경계 (c) | $W$ |
| sequence layer 상태(KV cache 또는 고정 크기 memory) | 모델, 토큰마다 | 매 토큰 | **예** | 논의 없음(블랙박스) | $W$ |
| 새 저랭크 expert $\{A,B\}$ | 확장 전 자기 자신으로부터의 distillation | sleep 1회 | 아니오 | **예 — 이 논문의 쓰기 연산** | $\Theta$ |
| 보내는 블록의 기존 expert들 | 아무도 — **파괴된다** | sleep (c) | 아니오 | 리셋(목표값 미명시) | $W\!\to\!\varnothing$ |
| MoE router $\mathcal{R}^{(\ell)}$ | **미명세** | 미명세 | 미명세 | 미명세(dream 표집 시 교란만 명시) | — |
| dream별 LoRA adapter $\theta'^{(i)}$ | 격리된 모델 사본 | sleep, 그리고 ARC에서는 **wake** | ARC 프로토콜에서 예 | 예 | $\Theta$ 모양 |
| dream 생성 정책 | ReST$^{\mathrm{EM}}$, 식 (21-8) | sleep | 아니오 | 예 | $\Theta$ |
| frozen semantic reward model | 아무도 — 외부·미식별 | — | 아니오 | 아니오 | 외부 |
| $\lambda_{\mathrm{on}},\lambda_{\mathrm{KD}},\rho_r,z_0,n_{\mathrm{dr}},n_{\mathrm{top}},n_{\mathrm{rand}}$ | 아무도 — 손으로 설정, **값이 논문에 없음** | — | 아니오 | 아니오 | — |
| external store $E$ | 존재하지 않음 | — | — | — | $E$ |

표에서 세 줄이 특히 무겁다. 첫째, router 줄이 전부 "미명세"다. 새로 활성화된 expert로 dispatch를 배우는 과정도, 리셋된 expert로의 dispatch를 멈추는 과정도 논문에 없다 — 기제의 나머지가 전부 이 줄에 의존하는데도 그렇다. 둘째, 하이퍼파라미터 줄의 값이 하나도 없다. Table 5가 싣는 것은 LR·batch·steps·LoRA rank·LoRA alpha뿐이다 [LM Need Sleep Table 5, p.26]. 셋째, dream별 LoRA 줄만이 **wake 경로에 gradient를 들여놓는다** — ARC에서 "for each unseen task, the model generates 5 dreams and applies them independently before predicting the held-out output" [LM Need Sleep App. B.3, p.25].

> **[해설]** 마지막 줄이 이 논문의 유일한 요청별 상태다. 나머지 전부는 **전역**이다. 논문은 사용자별·세션별 가중치를 만들지 않고 모델 하나를 유지한다. 이 사실은 §21.7에서 다시 쓴다 — $\Theta$-경로가 곧 per-user delta라는 도식은 이 논문에 근거가 없다.

## 21.5 비용 4종

표 21-3. [LM Need Sleep]의 비용 4종

| 값 | 논문이 보고하는 것 | 판정 |
|---|---|---|
| $B_s$ | FLOPs·토큰 수·절대 wall-clock 전부 없음. 비율만 있다 — "With the same number of steps, SFT is 4x more efficient than our method" [App. B.5, p.26]. 동일 성능 기준으로는 SFT가 AIME-24 4.3×, AIME-25 3.6×, HMMT-25 4.8×의 wall-clock을 쓴다 [App. B.5, p.26]. 절차량은 부분적으로 있다 — sleep 100 step, effective batch 32, LR 5e-6 [Table 5]; ARC는 과제당 dream 60개 생성·45개 기각 [App. B.3]; SQuAD 지속 pre-training은 passage당 dream 5개 × 200 passage [§4.1, p.12] | **절대값 논문에 없음.** teacher 표본 수, on-policy rollout 수, sleep 시 시퀀스 길이, 실험당 sleep 주기 수, GPU 수·GPU-hour가 전부 없다 |
| $L_w$ | 지연·TTFT·ms/token·throughput 숫자가 논문 전체에 없다. 가장 가까운 진술은 파라미터 수 주장이다 — "5 MLP blocks with dimension 64 as the additional parameters and keep the active parameter count unchanged" [App. B, p.25] | **논문에 없음.** active parameter 불변은 계수(計數) 주장이지 측정이 아니다. 식 (21-2)가 요구하는 wake 시 optimizer 오차 누적에도, ARC의 wake 내부 5회 LoRA 적합에도 비용이 붙어 있지 않다 |
| $C_{\text{cap}}$ | 파라미터로는 부분적으로 있다 — 통합 1회당 저랭크 expert 하나, 즉 $2\,d\,d_{\mathrm{low}}$개 [§3.2, p.7]. 실험 전체 추가분은 "5 MLP blocks with dimension 64" [App. B, p.25]. dreaming SFT는 rank $r=64$, alpha 128 [Table 5] | **바이트로는 논문에 없음.** $d$와 $d_{\mathrm{low}}$의 값이 없고, expert pool은 사전 할당 후 마스킹되므로 [§3.3 구현 주석, p.9] 상한이 $t=0$에 정해지는데 **그 pool 크기가 없다.** 세션별·사용자별 값이 아니라 전역 값이다 |
| $\rho$ | 라운드당 열화 수치가 없다. 정성 서술과 그림만 있다 — 지속학습에서 ICL은 급락하고 Hope는 이득을 더 지킨다, "Performance improves monotonically with additional consolidation stages" [§4.1, p.11-12, Figure 5]. Cartridges와 SFT는 "outside of the plot" [§4.1, p.12]. "increasing the lowest frequency reduces performance" [§4.1, p.11, Figure 4] | **논문에 없음. 그리고 측정 불가능하다** — 이 논문에는 sleep 주기 수를 독립변수로 놓은 실험이 하나도 없다. Hope-1/2/3은 통합 **단계 수**를 바꾼 것이지 **라운드 수**가 아니다 [§4.1, p.11] |

식 (A)의 손익분기 $N_q^*$는 이 논문에 대해 세울 수 없다. 이유가 구조적이다. 쓰기가 문맥별이 아니라 **스케줄별**이고, 수혜자가 이후의 모든 질의이므로 상각의 분모 $N_q$가 "앞으로의 전 트래픽"이 된다. 분모가 무한하면 상각된 sleep 비용은 0으로 간다.

> **[평가]** 이것은 논문의 승리가 아니라 **프레임에 대한 발견**이다. 식 (A)는 한 문맥을 $N_q$개 질의가 공유한다는 $E$-경로의 모양을 전제한다. $\Theta$-경로는 그 전제를 받지 않으므로 (A)가 퇴화하고, 그 순간 구속력 있는 질문이 두 개로 옮겨간다 — **쓰기가 살아남는가($\rho$)와 용량이 버티는가($C_{\text{cap}}$)**. 그리고 이 논문은 그 둘을 다 보고하지 않는다. 세 층 프레임에서 $\Theta$-경로의 증거 공백은 정확히 이 모양이며, Part III의 비용 회계가 이 자리에서 시작한다.

> **[평가]** 그러므로 이 장은 이 논문을 "효율적"이라고 쓰지 않는다. 비용 4종의 네 칸 중 절대값이 있는 칸이 0개다. 논문이 가진 유일한 효율 근거는 SFT 대비 상대비이고, §21.6이 보이듯 그 SFT는 자기 표에서 base model보다 낮은 baseline이다. 비용 침묵을 성능 주장으로 메우지 않는 것이 이 책의 규칙이다.

## 21.6 실험과 스케일

실증 상한부터 적는다. 가장 큰 모델은 8B다 — 지속학습에 Llama3-8B, 수학 추론에 Qwen3-8B [LM Need Sleep §4.1, p.11; Table 2, p.12]. 가장 작은 것은 Llama-3.2-1B [App. B.3]. from-scratch pre-training은 없고 8B 위의 모델도 없다. 가장 큰 데이터 스케일은 BABILong의 10M 토큰 문맥이고, 지식 주입은 200 passage / 974 문항이다 [§4.1, p.12]. **실증된 sleep 예산은 effective batch 32에서 100 optimization step이며, 실증된 wake/sleep 주기 수는 어느 실험에서도 진술되지 않았다[본서 관찰].**

데이터 쪽 규모도 적어 둔다. 지속학습 텍스트 분류는 CLINC150(150 in-scope intent, 10 도메인, 총 23.7K 질의 중 in-scope 22.5K·OOS 1.2K), Banking77(77 intent, 13,083 예제, 불균형), DBpedia(level-2 70 클래스, 훈련 10K·테스트 1K 부표집)이고, 장문맥 쪽은 LongHealth(각 5.1K–6.8K 단어의 사례 문서 20편, 문항 200개)와 QASPER(약 1.6K 편의 논문에 대한 약 5K 문항, 전문을 문맥으로)다 [LM Need Sleep App. B.1, p.25]. MK-NIAH는 길이 격자가 진술되지 않았다.

수학 추론 표가 숫자를 가진 헤드라인이다. Qwen3-8B, AIME-24 / AIME-25 / HMMT-25, avg@16 [LM Need Sleep Table 2, p.12]: Base(Instruct) 73.8 / 68.1 / 42.4, SFT 75.5 / 66.4 / 43.7, GRPO 76.4 / 68.1 / 44.9, OPSD 76.6 / 67.4 / 45.1, Sleep 79.2 / 69.0 / 46.1. Qwen3-1.7B에서는 Base 49.8 / 34.5 / 25.7, SFT 47.3 / 36.1 / 22.9, GRPO 51.0 / 38.6 / 26.1, OPSD 51.6 / 40.0 / 28.1, Sleep 53.2 / 40.2 / 29.3이다. 최강 baseline OPSD 대비 마진은 8B에서 +2.6 / +1.6 / +1.0, 1.7B에서 +1.6 / +0.2 / +1.2이다[본서 산술]. 같은 표에서 SFT는 1.7B의 AIME-24(47.3 대 49.8)와 HMMT-25(22.9 대 25.7), 8B의 AIME-25(66.4 대 68.1)에서 **base model보다 낮다.**

지식 주입은 SQuAD 무문맥 정확도 평균이다(단일 passage $n=1$ / 지속 pre-training $n=200$) [LM Need Sleep Table 3, p.12]: Base 31.9 / 31.9, dreaming 없는 fine-tuned 모델 33.4 / 32.0, SEAL 46.7 / 43.2, Sleep(Transformer) 48.1 / 44.3, Sleep(Transformer + four-level) 48.9 / 46.2. dreaming 제거는 35.7 / 36.2로 떨어뜨려 −13.2 / −10.0이며, **Dreaming이 이 결과를 지고 있다**[본서 산술]. 반면 gradient 기반 선택 제거는 47.1 / 45.2, 무작위 expert 제거는 48.0 / 44.7로 각각 −1.8 / −1.0, −0.9 / −1.5에 그친다[본서 산술]. ARC는 성공률로 ICL 0, TTT 10, SEAL 72.5, Sleep 80이다 [LM Need Sleep Table 4, p.12].

### 21.6.1 캡션이 자기 표에 반증되는 자리

[LM Need Sleep §4.2]는 "All the components contribute positively to the performance of our method"라고 적는다 [LM Need Sleep §4.2, p.13]. 그 문장이 인용하는 표가 그것을 반증한다.

표 21-4. 구성요소 ablation, Qwen3-8B, AIME-24 / AIME-25 / HMMT-25, avg@16 [LM Need Sleep Table 1, p.11]

| 구성 | AIME-24 | AIME-25 | HMMT-25 |
|---|---|---|---|
| Sleep (전체) | 79.2 | **69.0** | 46.1 |
| − Imitation Learning | 76.8 | 67.9 | 45.0 |
| − Semantic Reward | 78.9 | **69.2** | 44.5 |
| − w/o Expansion | 78.2 | 67.9 | 44.9 |
| OPSD | 76.6 | 67.4 | 45.1 |
| OPSD + Expansion | 77.9 | 68.2 | 45.9 |

**Semantic Reward를 빼면 AIME-25가 69.0에서 69.2로 오른다.** 구성요소를 제거했는데 성능이 올라간 칸이 표 안에 있고, 그 표를 인용하는 문장이 모든 구성요소가 긍정적으로 기여한다고 적는다. 하필 그 구성요소가 §21.3.2에서 확인한 대로 **이름도 크기도 밝혀지지 않은 외부 frozen reward model**에 의존하는 항이다. 논문의 효율 비교(App. B.5)에도 그 모델의 비용은 계상되어 있지 않다.

두 번째로, 헤드라인 구조적 기여의 크기가 작다. 확장을 뺀 Sleep이 78.2 / 67.9 / 44.9이므로 확장의 기여는 1.0 / 1.1 / 1.2다. 반대 방향이 더 날카롭다 — 확장만 baseline OPSD에 얹으면 76.6 / 67.4 / 45.1 → 77.9 / 68.2 / 45.9로 +1.3 / +0.8 / +0.8 오른다. Sleep이 OPSD를 이기는 총 마진이 +2.6 / +1.6 / +1.0이므로 **그 마진의 절반가량 — HMMT-25에서는 1.0 중 0.8 — 이 Knowledge Seeding 없이 구조적 확장만으로 얻어진다.** ch04가 이 계산을 먼저 적었고 판정을 이 장에 넘겼다. 판정은 이렇다.

> **[평가]** 이 논문이 새로 도입한 것은 Knowledge Seeding과 Dreaming 둘인데, 수학 추론에서 마진의 절반은 그 둘이 아니라 파라미터를 더 준 것에서 나오고, 지식 주입에서 마진의 대부분은 Dreaming에서 나온다. **Knowledge Seeding — 이 논문의 이름이 걸린 기여 — 이 수치로 자기 몫을 보인 표가 없다.** 게다가 표 21-4 전체에 시드도 신뢰구간도 오차막대도 없고, 여기서 논의되는 차이의 다수가 0.2–1.5점이다. avg@16이라는 표기만 있고 분산은 어디에도 없다 [Table 1/2 캡션].

### 21.6.2 숫자가 없는 실험, 그리고 이름의 충돌

일곱 개 실험 중 **넷이 숫자 없이 그림으로만** 보고된다 — 지속학습 텍스트 분류(Figure 3), 통합 단계 수 sweep(Figure 4), 신규 언어 지속 번역(Figure 5), BABILong(Figure 6) [LM Need Sleep §4.1, pp.11-12]. 논문에도 부록에도 이들의 값을 싣는 표가 없다. 이 논문의 가장 강한 스케일 주장인 "Hope achieve almost perfect score in scaling to 10M of tokens" [§4.1, p.11]에도 붙은 숫자가 없다.

그림뿐인 넷 가운데 Figure 4는 이 논문이 자기 설계 손잡이를 쓸어 본 유일한 자리라 특히 아깝다. 진술된 추세는 둘인데 두 번째가 반직관적이다 — 통합 단계 수를 늘리면 in-context learning과 장문맥 이해가 좋아지고, **"increasing the lowest frequency reduces performance, suggesting that making the most persistent memory more adaptive weakens retention"**이다 [LM Need Sleep §4.1, p.11]. 가장 존속하는 memory를 더 적응적으로 만들수록 retention이 나빠진다는 이 관찰은 $\Theta$-경로 설계에 직접 걸리는 진술인데 수치가 없다. 같은 그림의 캡션은 세 패널 중 QASPER만 축이 뒤집혀 있다고 경고한다("Lower values indicate better performance for QASPER") — 재작도할 때 반드시 옮겨야 할 단서다.

그 10M 결과에는 저자 자신의 유보가 붙어 있다: "Finally, we observe that all small models, including Hope, suffer substantial performance drops without fine-tuning" [LM Need Sleep App. B.2, p.26]. 소형 baseline은 전부 공식 BABILong 훈련 프로토콜로 fine-tune되었고 비교 대상인 GPT-4·GPT-4o-mini는 zero-shot이다 — 그림 캡션이 그 사실을 스스로 적는다("Red points correspond to fine-tuned models, whereas blue points correspond to zero-shot evaluations of large-scale models") [Fig. 6 캡션, p.11]. **동등 비교가 아니다.**

이름의 충돌도 있다. [LM Need Sleep §4.1]은 plain Hope를 baseline으로 도입해 놓고("We also include Hope ... as a multi-level in-context updating baseline without explicit distillation process") 바로 뒤에서 "Results in Figure 3 show that Hope performs best across datasets"라고 쓴다 [LM Need Sleep §4.1, p.11]. 승리를 선언하는 문장이 baseline의 이름을 부르며, 결과가 그림뿐이라 본문만으로는 어느 쪽인지 판별되지 않는다.

내부 참조도 깨져 있다. [LM Need Sleep §4.2]는 "The results are reported in Figure 1"이라고 적지만 Figure 1은 관행적 ML과 지속학습을 대비한 도식이고 실제 결과는 Table 1에 있다. 같은 문장이 자기 개선 ablation을 "Table 3"으로 넘기는데 그것은 지식 주입 표다. 그리고 Table 1과 Table 2의 캡션이 바이트 단위로 동일하다 [LM Need Sleep §4.2, p.13; Tables 1–2].

### 21.6.3 비교 조건과 기반 구조

효율 주장은 SFT 하나에 대해서만 측정되었다. step당 SFT가 4배 효율적이고, 동일 성능 기준으로 SFT가 4.3× / 3.6× / 4.8×의 wall-clock을 쓴다 [LM Need Sleep App. B.5, p.26]. GRPO와 OPSD에 대한 등-계산 비교는 없다. 그런데 Table 5는 GRPO에 500 step을, SFT와 Sleep에 각각 100 step을 주고 LR과 batch는 동일하게 둔다 [LM Need Sleep Table 5, p.26] — **예산이 맞춰져 있지 않고**, 어느 arm이 수렴했는지 보이는 학습 곡선도 없다.

ARC 헤드라인은 40개의 이진 결과 위에 서 있다. 지표가 "the fraction of dreams that yield a correct answer"이고 held-out 8과제 × dream 5개이므로, 80%는 32/40이고 SEAL의 72.5%는 29/40이다 — **격차가 결과 3개**다 [LM Need Sleep App. B.3, p.25]. 훈련 풀은 11과제이며 과제 집합은 "to avoid tasks that remain unsolvable under standard configurations"로 걸러졌다. 그리고 SEAL 비교를 싣는 Table 3의 **backbone이 논문 어디에도 명시되지 않는다** — "We follow the experimental setup of Zweiger et al. (2025), including the choice of models and parameters, for the sake of fair comparison"이 모델에 대한 유일한 진술이다 [LM Need Sleep §4.1, p.12].

가장 무거운 것은 마지막이다. **이 논문의 두 기여가 자기 아키텍처 위에서 함께 돈 적이 없다.** [LM Need Sleep §3]은 모든 것을 CMS/MoE backbone 위에서 정의하고 [LM Need Sleep §4.1]의 통합 실험은 Hope를 쓰는데, Dreaming 결과는 "Sleep (Transformer)"와 "Sleep (Transformer + four-level)"이며 [Table 3, p.12] ARC는 Llama-3.2-1B다 [App. B.3, p.25]. 통합과 dreaming을 Hope/CMS 위에서 함께 돌린 실험이 논문에 없다. 두 절반이 서로 다른 기반에서 검증되었고 그 합성은 미검증이다.

여기서 NM ch17이 확정한 사실이 STC 프레임에서 다시 판정된다. 이 논문의 sleep 기계 — $\lambda_{\mathrm{on}}, \lambda_{\mathrm{KD}}, \rho_r, z_0$, reward model, ReST$^{\mathrm{EM}}$ 루프 — 는 어느 것도 pre-training에서 end-to-end로 meta-learn되지 않으며, 실험의 대부분이 pre-trained Llama/Qwen 위에 sleep 기계를 얹은 **graft**다(→ NM ch17). NM은 이것을 라인의 이음새로 읽었다 — 지금까지 update rule 자체를 outer loop가 미분해 학습하던 계보가 여기서 알고리즘적 wrapper로 바뀌었다는 것이다.

> **[평가]** STC 프레임에서는 다른 값을 갖는다. **graft라는 사실은 이 논문의 $W \to \Theta$ 이송 연산자가 학습된 연산자가 아니라 손으로 배치한 절차라는 뜻이다.** 이송의 시점(스케줄), 이송의 폭($d_{\mathrm{low}}$), 이송의 표적(teacher 분포), 이송 뒤의 삭제(리셋)가 전부 하이퍼파라미터이고 그 값들이 논문에 없다. 세 층 사이에서 무엇을 언제 옮길지가 이 책의 중심 질문인데, corpus에서 그 연산자를 가진 유일한 논문이 그것을 학습하지 않는다. 이것은 결함이 아니라 **이 경로의 현재 위치**다 — $\Theta$-경로는 이송을 할 줄 알지만 아직 언제 얼마나 이송할지를 데이터로 정하지 못한다. 반대로 이 사실은 graft가 갖는 한 가지 이점도 정직하게 만든다: 기성 checkpoint 위에 얹을 수 있으므로 8B 실증이 가능했다. 라인의 from-scratch 상한(1.3B/100B tokens, → ch18)이 깨진 것은 아니다.

마지막으로 우선권 주장 하나를 기록한다. App. A.2는 동시대 OPSD 계열과의 차별점 셋 중 **첫 번째로 발표 순서**를 든다 — "In addition to the fact that this study is older than such methods, Sleep fundamentally delivers a different messages that..." [LM Need Sleep App. A.2, p.23], 근거는 2025년 9월부터 OpenReview에 공개되어 있었다는 p.1 각주다. 통제된 비교가 아니라 순서가 첫 논거로 놓였다[본서 판단].

## 21.7 Systems/serving 함의

독자의 질문은 하나다 — 그래서 decode에서 뭐가 바뀌는가. 이 논문의 답은 세 겹이다.

**첫째, inference가 forward 전용이 아니게 되고, 그 변화가 sleep이 아니라 wake에 착지한다.** 식 (21-2)는 유한 주파수 블록마다 토큰당 optimizer 오차 항을 누적하고 $C^{(\ell)}$ step마다 가중치에 적용할 것을 요구한다. Figure 7이 이 책에서 가장 값진 그림인 이유가 여기 있다 — 논문에서 갱신 주기가 토큰 수로 나타나는 **유일한** 자리이고, 사다리가 1K → 5K → 10K로 적혀 있다 [LM Need Sleep Fig. 7, p.22].

> **[해설]** 서빙 엔진 쪽으로 옮기면 이렇다. decode 경로 안에 backward가 가능한 kernel이 상주해야 하고, 활성 블록마다 파라미터 크기의 accumulator가 하나씩 필요하며, 1K 토큰마다 가중치 쓰기가 발생한다. Rosetta 표의 "decode = $C=1$의 per-token write"(→ ch01)가 여기서 $C^{(\ell)}=1000$으로 커진 형태다. **논문은 이 wake 비용에 어떤 숫자도 붙이지 않는다.**

**둘째, 정적 텐서 모양이 의도적 서빙 결정이고 그 이유가 적혀 있다.** 구현 주석이 말한다 — "Implementing the growing sparse modules can be extremely challenging if it requires a direct change in the dimensionality of tensors ... Alternatively, we can initially have those parameters in the model, but masked them in the forward and backward pass, before their initial activation in a sleep stage" [LM Need Sleep §3.3, p.9]. 컴파일된 그래프·kernel autotuning·checkpoint 레이아웃이 sleep을 넘어 안정적으로 유지되므로, 이 corpus에서 가장 서빙 친화적인 설계 결정이다[본서 판단]. 대가는 §21.5가 적은 대로 $t=0$에 정해지는 **크기 미상의 하드 천장**이다.

**셋째, 존속하는 상태가 KV cache에서 가중치로 옮겨간다 — 크기가 아니라 종류가 바뀐다.** BABILong 설정은 10M 토큰 문맥에 닿는데 [§4.1, App. B.2] 그 영역에서 $O(L)$ KV cache는 선택지가 아니다. 대신 통합 1회당 존속 delta는 $A \in \mathbb{R}^{d\times d_{\mathrm{low}}}$, $B \in \mathbb{R}^{d_{\mathrm{low}}\times d}$ 하나, 즉 LoRA 모양의 객체다. 서빙 문제가 paged-KV 관리에서 **adapter 버전 관리와 축출**로 옮겨간다[본서 추론].

> **[평가]** 여기서 반드시 끊어 둘 것이 있다. 이 상태는 **전역**이다. 논문은 모델 하나를 유지하며 세션별·사용자별 상태가 없다. 따라서 "$\Theta$-경로는 사용자별 가중치 delta를 만들고, 그 delta 트래픽이 memory-centric 기회를 연다"는 서사를 이 논문에 귀속시킬 수 없다. 이 책은 근거가 있는 자리에만 그 논증을 쓴다(NM D4 승계). 이 논문이 실제로 공급하는 것은 per-user 상태가 아니라 **버전 관리 문제**다 — sleep이 $\{C^{(\ell)}b\}$마다 발화하고 매 발화가 base 갱신·expert 활성화·다른 블록 expert 리셋을 수행하므로, checkpoint 관리·cache 무효화·회귀 테스트·롤백이 전부 주기 단위 운영이 된다. 그리고 리셋의 목표값이 명시되지 않았으므로, 외부 스냅샷 없이는 순진한 구현이 sleep 이전 모델로 되돌아갈 수 없다.

**sleep job의 모양도 통상과 다르다.** Knowledge Seeding은 teacher 표집 데이터셋, on-policy student rollout, LTI prefix 완성을 필요로 하고, Dreaming은 $n_{\mathrm{dr}}$회 생성 + 후보마다 backward 1회 + 살아남은 dream마다 격리된 LoRA SFT를 필요로 한다. ARC 수치로는 과제당 60 생성·60 backward·15 LoRA 적합이다 [LM Need Sleep App. B.3, p.25]. 지배적 비용 모양이 training FLOPs가 아니라 **autoregressive decode**라는 뜻이며, 그러면 sleep job은 훈련 클러스터가 아니라 wake 트래픽과 같은 하드웨어·같은 대역폭 지배 영역을 놓고 경쟁한다. 논문이 절대 비용을 보고하지 않으므로 이 추론은 검증할 수 없다. 덧붙여 후보 dream마다 도는 gradient 계산은 App. B.5의 4배 수치에 **포함되어 있지 않다** — 그 비교는 Sleep step과 SFT step을 견줄 뿐 선택 오버헤드를 세지 않는다.

마지막으로 배칭이다. ARC 프로토콜은 요청 하나 안에서 dream 5개를 생성하고 각각을 독립 적용한 뒤 예측한다. LoRA로 갈라진 모델 인스턴스 5개와 그 안의 fine-tuning 5회가 사용자 응답 경로에 들어간다는 뜻이고, Rosetta 표의 "batch로 weight 공유" 행이 정확히 이 지점에서 깨진다[본서 추론]. **이 프로토콜에 붙은 지연 수치는 없다.**

## 21.8 한계와 bridge-out

### 21.8.1 논문 자신이 남긴 문제

먼저 형식을 적는다. **이 논문에는 limitations 절이 없다.** [LM Need Sleep §5] Conclusion은 다섯 문장이고 유보도 실패 모드도 부정 결과도 담고 있지 않다 [LM Need Sleep §5, p.13]. §21.6이 적은 불리 사실은 전부 표·부록·내부 불일치에서 복원한 것이다. 참고문헌에도 기계적 정리의 흔적이 남아 있다 — Pang et al., Singh et al., Shenfeld et al., Zhang et al.(OPSDL), Zhao et al.의 항목이 중복 등재되어 있고 [pp.17-21], 그 다수가 동시대 OPSD 흐름을 두 페이지로 훑는 App. A.4에 걸린다.

그럼에도 논문이 간접적으로 남긴 문제는 뚜렷하고, 대부분 이 책의 비용 4종에 직결된다.

- **반복 안정성.** [LM Need Sleep §3.2]는 고정 크기 느린 블록에 반복 통합하는 것을 "a critical bottleneck"이라 진단하고 확장을 처방한 뒤, sleep 주기 수를 독립변수로 놓은 실험을 한 번도 보고하지 않는다. lifecycle을 주장하는 논문에서 라운드 수가 측정되지 않았다.
- **pool 고갈.** 구현 주석의 사전 할당·마스킹은 무한 성장을 하드 천장으로 바꾼다 [§3.3, p.9]. 천장에 닿으면 무슨 일이 일어나는지, pool이 얼마나 큰지 둘 다 없다.
- **언제 잘 것인가.** 스케줄 $\{C^{(\ell)}b\}$는 주파수 사다리에서 상속된다. 트리거를 학습하는 것이 없고, 실제 서빙 트래픽이 균일한 토큰 스트림이 아니라는 사실은 다뤄지지 않는다.
- **리셋의 수치적 정의.** "reset (synaptic pruning)"이 0인지 재초기화인지 저장된 초기값 복원인지 명시되지 않는다 [§3.3(c), p.8]. 망각 거동과 checkpoint 서사가 이 선택에 함께 걸린다.
- **무엇이 보존되는가.** consolidation이 원자료 replay가 아니라 "추상화"를 추출한다고 주장하면서, 무엇이 남고 무엇이 버려지는지에 대한 측정도 rate–distortion 형태의 진술도 없다. 통합 1회당 얼마의 rank가 얼마의 망각을 막는지도 정식화되어 있지 않다.
- **reward model의 편향과 drift.** 고정 심판이 매 sleep마다 바뀌는 student를 채점하는데 그 편향·drift·reward hacking이 검토되지 않는다. 그리고 그 항의 제거가 벤치마크 하나를 개선한다(표 21-4).
- **재귀적 자기 수정의 안전.** 스케줄에 따라 자기 가중치를 고쳐 쓰고 무작위 expert로 "random irrelevant knowledge"를 주입하는 모델의 안전·거버넌스 논의가 논문 어디에도 없다.
- **서빙 회계.** 종단 wake/sleep 주기의 wall-clock 분해가 없고, replay buffer라는 고전적 대안과의 등-계산 비교도 없다 — 생물학 프레임이 불러들이는 바로 그 대안인데도 그렇다.

### 21.8.2 이 책의 판정

> **[평가]** 이 논문은 corpus에서 $W \to \Theta$ 이송을 **명시적 연산자로 가진 유일한 사례**이며, 그 점에서 세 경로 사이의 실제 다리다. ch18은 갱신 주파수를 낮추는 것만으로는 $\Theta$에 닿지 못한다고 판정했다. 이 논문은 그 판정을 우회하지 않고 인정한 뒤, 닿는 방법을 별도의 오프라인 절차로 만들었다 — 확장하고, 위로 증류하고, 아래를 지운다. **이 책이 $\Theta$-경로를 별도 축으로 세우는 근거가 여기서 실증된다.**
>
> 그러나 다리가 놓였다는 것과 다리가 하중을 견딘다는 것은 다르다. 하중 시험에 해당하는 값이 $\rho$인데, 그 값은 없을 뿐 아니라 **주어진 증거로부터 계산될 수도 없다**(§21.5). $\Theta$-경로 논문에서 이보다 결정적인 결측은 없다. 이 논문의 논지 전체가 반복되는 wake/sleep 주기 위에 서 있고, 반복이 측정되지 않았기 때문이다.

> **[평가]** 두 번째 판정은 기여의 배분에 대한 것이다. 표 21-4가 말하는 바를 그대로 읽으면, 수학 추론에서 마진의 절반은 파라미터를 더 준 것에서 나오고, 지식 주입에서 마진의 대부분은 Dreaming에서 나오며, Semantic Reward는 한 벤치마크에서 음의 기여를 한다. **이 논문의 이름이 걸린 기여인 Knowledge Seeding이 자기 몫을 수치로 보인 자리가 없다.** 그럼에도 [LM Need Sleep §4.2]는 모든 구성요소가 긍정적으로 기여한다고 적는다. 이 corpus에서 캡션이 자기 표에 반증되는 사례는 이것이 처음이 아니며(→ ch15, ch18), 반복된다는 사실 자체가 Part III의 재료다.

### 21.8.3 경로 간 인용 여부

연대를 먼저 갈라 둔다. 이 논문의 v2는 2026년 7월 10일 자다. 따라서 `Rate–Distortion Memory Compaction`(2607.08032)과 `Can a LM Learn Facts Continually in Its Weights?`(2607.11020)의 부재는 **연대가 강제한 것**이며 이 장은 그것을 흠으로 세지 않는다(→ ch22). 아래는 전부 선택된 침묵이다.

**$E$-경로 — 이름으로 부르고 두 문장으로 닫는다.** Letta의 `Sleep-time Compute`(2504.13171)과 MemGPT(2310.08560)가 둘 다 인용되어 있다. 교전 방식이 특징적이다. [LM Need Sleep §2.3]은 Letta의 절차를 "mainly is designed as a proposal for spending compute on the context rather than training the model itself"라 요약하고, App. A.3이 이렇게 닫는다 — "Despite similarity in the name, their method is fundamentally different from our work. While this method aim to find a good summary of the interactions of the model with users in the text space, when the model is idle, our proposal is on using distillation to transfer text data knowledge into a form of parametric weight update" [LM Need Sleep App. A.3, p.23]. MemGPT은 "orchestration systems like MemGPT manage context via virtual memory paging rather than architectural modification" 한 문장이다. Mem0·Zep·ReasoningBank은 전부 없다.

> **[평가]** 기각이 **정의로 이루어지고 실험으로 이루어지지 않았다.** $E$-경로 시스템은 일곱 실험 어디에도 baseline으로 등장하지 않는다 — SQuAD 지식 주입처럼 검색이나 요약 baseline이 자명한 통제군인 과제에서도 그렇다. 이 논문은 자기 이름의 소유권을 두고 다투면서, 상대를 한 번도 돌려 보지 않는다. 그리고 두 경로를 **함께** 쓰는 구성(외부 요약과 파라미터 통합의 합성)은 질문으로도 제기되지 않는다 — 정의로 닫혔기 때문이다. ch12가 2022년에 이미 이 질문이 한 번도 제기된 적 없다고 기록했고, 2026년에도 여전히 그렇다.

**$W$-경로 — 아키텍처는 전수, sleep은 전무.** 인용은 두텁다. Titans(2501.00663)는 BABILong baseline으로 실제로 돌기까지 하고, TTT(2407.04620)는 ARC baseline이며, Nested Learning(2512.24695)은 직계 부모다. linear attention, Mamba, xLSTM, Schmidhuber의 Fast Weight Programs(1992), Cartridges, DuoAttention이 모두 있다. 그런데 **`Do Language Models Need Sleep?`(2605.26099)이 참고문헌에 없다.** 제목이 한 글자 차이로 겹치는 W-경로의 sleep 논문이고, 두 논문의 간격은 한 달이다. `Memory Caching`(2602.24281)도 없다.

침묵은 **일방향**이다. ch17이 확인한 대로 `Do LMs Need Sleep?` 쪽은 이 논문을 인용하고 있으며, 다만 그 항목에 arXiv ID도 venue도 연도도 없이 제목만 있다 — OpenReview 투고본에서 인용했다는 뜻이다. 즉 W-경로 쪽은 $\Theta$-경로의 이 논문을 보았고, $\Theta$-경로 쪽은 W-경로의 sleep 논문을 보지 않았다. **W-경로 문헌을 가장 두껍게 인용하는 논문이 정확히 W-경로의 sleep 문헌만 비워 두었다.** 아키텍처는 전수 조사하고 같은 경로의 같은 주제는 0건이다.

**$\Theta$-경로 형제와 이론 축.** SEAL은 인용되고 Table 3·4의 baseline으로도 돈다. LoRA와 EWC도 있다. 반면 SCM(2604.20943)은 없고 — 같은 경로의 다른 sleep 논문이다 — ROME(2202.05262)·MEMIT(2210.07229)도 없어 **가중치에 직접 쓰는 다른 방식 전체가 교전되지 않는다**(→ ch19). Generative Adapter(2411.05877)와 Memory Layers(2412.09764)도 없고, 생물학 프레임이 기대는 바로 그 기제인 Deep Generative Replay(1705.08690)도 없다. 용량 축은 §21.2가 적은 대로 비어 있다.

**생물학 축 — 가장 두껍게 인용되고 가장 얕게 쓰인다.** [LM Need Sleep §1]이 두 페이지 가까이를 인간 sleep 신경과학에 쓰고, CLS(McClelland 외 1995), Tononi–Cirelli의 synaptic homeostasis, sharp-wave ripple 연구가 줄지어 인용된다. 그런데 **어느 생물학 결과도 정량적 예측으로 바뀌지 않는다.** 가장 직접적으로 관련된 두 계산신경과학 결과 — Gonzalez 외 2020과 Tadros 외 2022, 둘 다 sleep 유사 replay가 인공신경망의 CF를 줄인다는 것 — 은 [LM Need Sleep §2.3]의 목록 한 줄에 담긴 뒤 한 번도 비교 대상이 되지 않는다(→ ch08, ch23). 은유는 이송되었고 측정은 이송되지 않았다.

### 21.8.4 다음 장이 받아가는 것

이 장은 $\Theta$-경로가 실제로 작동하는 형태를 보였다 — 확장하고, 위로 증류하고, 아래를 지우면 배포 가중치가 바뀐다. ch22가 받아가는 것은 그 절차가 **계속** 돌 수 있는가라는 질문이다. 이 장이 남긴 두 결측이 정확히 그 질문의 두 변수다. $\rho$는 반복이 열화를 낳는지 묻고, $C_{\text{cap}}$은 사전 할당된 pool이 언제 바닥나는지 묻는다. 이 논문은 둘 다 보고하지 않으며, 논지 전체가 그 둘에 걸려 있다. ch22는 가중치에 사실을 계속 넣을 수 있는가를 정면으로 묻는 반증 축을 세우고, ch09가 세운 용량 천장과 ch06이 세운 자기 생성 데이터의 경계를 그 위에 얹는다. 이 장의 자기 생성 $\mathcal{R}_k$와 리셋 연산자는 그때 다시 호출된다.

## 요약

- [LM Need Sleep]의 bridge-in은 이 corpus에서 거의 유일하게 양쪽이 닫혀 있다 — [NL]이 오프라인 consolidation을 범위 밖으로 선언했고, 이 논문이 "In this paper, we focus on the second type of consolidation: i.e., offline consolidation via sleep"으로 그 자리를 받는다 [LM Need Sleep §1, p.3]. 두 논문의 제1저자가 같다.
- 세 층 프레임에서 새로 보이는 것은 순서다. 식 (21-2)의 wake 갱신은 (U-W)이고, [LM Need Sleep §3.3]의 세 단계가 $W \to \Theta$ 이송 연산자이며, 마지막의 리셋이 그것을 복사가 아니라 **이송**으로 만든다. corpus에서 이 연산자를 명시적으로 가진 유일한 논문이다.
- 표준형 (U-$\Theta$)와의 차이 여섯 중 둘이 회계에 치명적이다 — $\dim\Theta$가 라운드마다 커지고, 매 라운드가 파괴 연산자로 끝난다.
- 비용 4종의 절대값이 네 칸 모두 없다. $B_s$는 SFT 대비 상대비만 있고(step당 4배, 동일 성능 4.3× / 3.6× / 4.8× [App. B.5, p.26]), $L_w$는 어떤 시간 수치도 없으며, $C_{\text{cap}}$은 바이트가 없고, $\rho$는 **주어진 증거로부터 계산될 수도 없다** — sleep 주기 수를 독립변수로 놓은 실험이 없기 때문이다.
- 상각식 (A)가 이 논문에서 퇴화한다. 쓰기가 문맥별이 아니라 스케줄별이라 $N_q$가 "앞으로의 전 트래픽"이 된다. (A)는 $E$·$W$-경로의 도구이고 $\Theta$-경로는 그것을 받지 않는다.
- Semantic Reward를 제거하면 AIME-25가 69.0에서 69.2로 **오른다** [LM Need Sleep Table 1, p.11]. 그럼에도 [LM Need Sleep §4.2]는 "All the components contribute positively to the performance of our method"라고 적는다 [p.13]. 그 구성요소는 이름도 크기도 밝혀지지 않은 외부 frozen reward model에 의존한다.
- OPSD 대비 마진의 절반가량이 Knowledge Seeding 없이 구조적 확장만으로 얻어진다(OPSD + Expansion이 76.6 / 67.4 / 45.1 → 77.9 / 68.2 / 45.9). 일곱 실험 중 넷은 숫자가 없고, 시드·신뢰구간·오차막대가 논문 전체에 없다.
- 기제는 pre-trained Llama/Qwen 위의 **graft**이며 라인 최초로 end-to-end meta-learn되지 않았다(→ NM ch17). STC 프레임에서 이 사실의 뜻은 하나다 — 이송의 시점·폭·표적·삭제가 전부 손으로 정한 하이퍼파라미터이고, 그 값들이 논문에 없다.
- 경로 간 침묵이 특정적이다. $E$-경로는 이름으로 부르고 두 문장으로 기각하되 baseline으로 한 번도 돌리지 않는다. $W$-경로는 아키텍처 계보를 전수 인용하면서 W-경로의 sleep 문헌(2605.26099, 2602.24281)만 0건이며, 그 침묵은 일방향이다.

## 자가 점검 체크리스트

- [ ] 이 장의 bridge-in이 왜 예외인지 두 조건으로 진술하고, 그 예외가 성립하는 이유가 저자 동일성이라는 사실까지 말할 수 있다.
- [ ] wake $W$ → sleep $\Theta$ 순서를 세 단계 (a)/(b)/(c)로 재구성하고, 왜 보내는 쪽 갱신의 적용을 미뤄야 하는지 설명할 수 있다.
- [ ] 식 (21-9)가 표준형 (U-$\Theta$)와 갈라지는 여섯 지점을 열거하고, 그중 어느 둘이 $C_{\text{cap}}$과 $\rho$에 직접 걸리는지 지목할 수 있다.
- [ ] 이 논문에 대해 상각식 (A)를 세울 수 없는 이유를 $N_q$의 정의로 설명하고, 그 결과 구속력 있는 질문이 어느 두 값으로 옮겨가는지 말할 수 있다.
- [ ] 표 21-4에서 캡션 주장이 반증되는 칸을 수치로 지목하고, Knowledge Seeding과 parameter expansion의 기여를 분리해 계산할 수 있다.
- [ ] "graft이지 meta-learn이 아니다"라는 사실을 NM의 독법과 이 책의 독법에서 각각 어떤 뜻으로 읽는지 구분해 말할 수 있다.
- [ ] 이 논문의 존속 상태를 Rosetta 사전의 어느 항목("모델 재배포", "checkpoint 크기 = $C_{\text{cap}}$", "paged KV cache / session cache")에 대응시킬지 정하고, 마지막 항목이 **아닌** 이유 — 상태가 세션별이 아니라 전역이라는 것 — 을 설명할 수 있다.
