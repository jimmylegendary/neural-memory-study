# ch19. LoRA와 모델 편집 — 가중치에 쓰는 두 방식

## 19.1 Bridge-in: 전작이 남긴 문제

ch18은 갱신 주파수 연속체가 $\Theta$에 닿지 않음을 보였다. 축 위에서 주파수를 아무리 낮춰도 얻는 것은 느리게 움직이는 $W$이고, 문맥 경계를 넘는 존속과 배포 아티팩트로의 되쓰기는 눈금을 옮겨 도달할 수 있는 상태가 아니라 다른 종류의 연산이다(→ ch18 §18.4). 그렇다면 질문은 하나만 남는다. **$\Theta$에는 무엇으로 쓰는가.**

이 장이 그 도구를 다룬다. 도구는 두 계열이다. 하나는 갱신을 저랭크 곱으로 제약해 **분리 가능한 delta 아티팩트**를 만드는 계열이고(LoRA, *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685, 이하 [LoRA]), 다른 하나는 배포 가중치의 지정된 행렬을 **제자리에서 고쳐 쓰는** 계열이다(ROME, *Locating and Editing Factual Associations in GPT*, arXiv:2202.05262, 이하 [ROME]; MEMIT, *Mass-Editing Memory in a Transformer*, arXiv:2210.07229, 이하 [MEMIT]). 두 계열의 대비가 이 장의 축이다.

bridge-in의 상태를 정직하게 적는다. **세 논문 중 어느 것도 sleep-time compute 계보의 open question을 물려받지 않았다.** 상속 진술이 있는 것은 [LoRA] 하나이고, 그 상대는 이 라인이 아니라 intrinsic dimension 문헌이다 — "과매개화된 학습 모델이 실제로는 낮은 intrinsic dimension 위에 놓인다는 것을 보인 Li et al. (2018a), Aghajanyan et al. (2020)에서 영감을 얻는다. 우리는 **모델 적응 중 가중치의 변화량 역시** 낮은 intrinsic rank를 갖는다고 가설한다" [LoRA §1, 동일 취지 §4.1]. [ROME]은 어느 선행 논문의 미해결 질문도 인용문으로 복원하지 않으며, 가장 가까운 문장은 저자 자신의 공백 진단이다 — "그것들이 지식을 저장하는 방식은 아직 충분히 탐구되지 않았다" [ROME §1]. [MEMIT]도 상속 진술이 없다. 이 논문이 물려받은 것은 문장이 아니라 **규모의 수열**이다: 편집 1개(단일 편집 패러다임) → 10개(De Cao et al.의 개선 목적함수) → 75개(Mitchell et al.) → 그리고 이 논문의 10,000개 [MEMIT §1, §2].

> **[평가]** 이 비대칭 자체가 관측이다. **도구는 위로 흘렀고 질문은 아래로 흐르지 않았다.** [ROME]이 $\Theta$-경로의 primitive가 된 것은 후행 논문들이 이 도구를 가져다 쓴 결과이며, 이 논문 자신은 sleep·consolidation·replay·forgetting·external memory 중 어떤 어휘도 한 번도 쓰지 않는다. 그러므로 이 장의 bridge-in은 인용의 복원이 아니라 **재구성**이다. 재구성이라는 사실은 §19.2에서 논문 자신의 문장으로 못 박는다.

## 19.2 문제의식

세 논문이 스스로 정의한 문제는 서로 다르고, 셋 중 어느 것도 기억 문제가 아니다.

[LoRA]의 문제는 배포 경제학이다. "fine-tuning의 주된 단점은 새 모델이 원 모델과 같은 수의 파라미터를 갖는다는 것이다" [LoRA §1]. 175B 규모에서 이것은 불편이 아니라 배포 불가능성이 된다 — "사전학습 모델이 크면(GPT-3의 경우 $|\Phi_0| \approx 1750$억) 독립적으로 fine-tune된 인스턴스를 여러 벌 저장하고 배포하는 것은, 가능하다 해도, 어렵다" [LoRA §2]. 여기에 두 번째 제약이 붙는다. 기존 parameter-efficient 대안은 추론 지연을 늘리거나 쓸 수 있는 시퀀스 길이를 잡아먹고, 품질에서 fine-tuning을 따라가지 못한다 [LoRA §1].

[ROME]의 문제는 해석가능성이다. "대규모 언어모델은 사실을 어디에 저장하는가. 이 논문에서 우리는 GPT의 factual association이 직접 편집할 수 있는 국소화된 계산에 대응한다는 증거를 보고한다" [ROME §1]. 절의 제목 자체가 "사실 연관 저장을 이해하기 위한 가중치 개입"이며, 편집은 가설 검정의 도구로 도입된다. 그리고 논문은 자기 범위를 명시적으로 제한한다.

> "ROME의 목적은 지식 저장 기제를 이해하기 위한 도구로 쓰이는 것이다. 이것은 한 번에 사실 하나만 편집하며, **대규모 모델 학습을 위한 실용적 방법으로 의도되지 않았다.**" [ROME §3.7]

> **[평가]** 이 문장을 먼저 실어야 한다. 이 책이 [ROME]을 $\Theta$-경로의 쓰기 primitive로 세우는 것은 **이 책의 재구성이며 저자의 주장 범위를 넘는다.** 이후 이 장이 [ROME]에서 끌어내는 모든 함의 — 상태량, 서빙, 상각 — 는 논문이 하지 않은 논의이고, 그 사실은 각 자리에서 다시 표기한다. 재구성이 부당하다는 뜻은 아니다. 후행 문헌이 실제로 이 도구를 그렇게 썼다. 다만 논문에 없는 주장을 논문에 귀속시키지 않는다는 것이다.

[MEMIT]의 문제는 용량이다. 첫 문장이 곧 문제다 — "가중치를 직접 편집해서 딥네트워크에 몇 개의 memory를 넣을 수 있는가" [MEMIT §1]. 재학습이 비용상 금지적이므로 지식을 직접 갱신하는 방법을 찾되 [MEMIT §1], 성공 기준은 "효과 있음" 하나가 아니라 generalization(재구문 프롬프트에서도 새 사실을 회상)·specificity(편집하지 않은 유사 주어가 오염되지 않음)·fluency(반복 붕괴 없음)를 함께 만족하는 것이라고 [MEMIT §3]이 정의한다. 그리고 선행 기법의 순차 반복이라는 우회로가 실패한다는 것을 자기 근거로 세운다 — "현재 최고 성능 지식편집 기법을 순진하게 순차 적용하면 규모가 나지 않는다" [MEMIT §1].

이 장이 소유하는 첫 구분을 여기서 못 박는다.

> **정의.** $\Theta$에 쓰는 방식은 산출물의 형태로 갈린다. **delta 방식**은 갱신을 base 가중치와 **분리 저장 가능한 객체** $\Delta\Theta$로 만들고, 배포 시점에 그것을 더할지 말지를 선택할 수 있게 남긴다. **직접 편집 방식**은 지정된 가중치 행렬을 제자리에서 고쳐 갱신을 base에 **흡수시키며**, 분리 저장 가능한 객체를 남기지 않는다. 두 방식은 wake-time 연산 그래프에 대해서는 같은 결과를 내고($L_w=0$), **배포 아티팩트의 단위와 되돌리기 연산에 대해서는 정반대다.**

> **[해설]** 이 구분은 rank의 크기와 무관하다. [MEMIT]의 갱신도 구조상 rank가 편집 개수 이하인 저랭크 갱신이지만[본서 관찰], 그것을 $\Theta^{(l)}_{\mathrm{out}}$에 흡수시키는 순간 분리 가능한 아티팩트가 사라진다. 반대로 [LoRA]는 merge를 실행하면 같은 자리에 도달하지만, $(A,B)$를 그대로 보관하면 언제든 되돌릴 수 있다(→ ch03 §03.5의 task switch). **갈림길은 저랭크냐 아니냐가 아니라 아티팩트를 남기느냐다.**

---

## 19.3 Core mechanism (통일 표기)

세 논문 모두 (U-$\Theta$) 하나만 실행한다. (U-W)도 (U-E)도 없고, 식 (R)은 $\hat y = f(q;\ \Theta)$로 붕괴한다 — 편집 후에도 forward pass는 원래 아키텍처 그대로이고 바뀐 것은 $\Theta$의 일부 값뿐이다.

### 19.3.1 delta 방식 — 저랭크 부분공간으로의 사영

기제는 ch03이 소유한다. 여기서는 대비에 필요한 두 줄만 다시 놓는다. 갱신 대상은 적용 행렬 $m \in \mathcal{S}$마다 붙는 저랭크 곱이고,

$$
\Theta^{(m)} \;=\; \Theta_0^{(m)} \;+\; \tfrac{\alpha_{\mathrm{LoRA}}}{r}\, B^{(m)} A^{(m)},
\qquad
B^{(m)} \in \mathbb{R}^{d_{\mathrm{out}} \times r},\quad A^{(m)} \in \mathbb{R}^{r \times d_{\mathrm{in}}}
\tag{19-1}
$$

학습은 $(A,B)$에 대해서만 돈다 [LoRA §4.1, Eq. 3]. $B^{(m)}=0$ 초기화 때문에 쓰기는 반드시 항등원에서 출발한다 [LoRA §4.1].

(U-$\Theta$)와의 차이는 셋이다. (i) gradient를 $\Theta$ 전체가 아니라 $(A,B)$에 대해 취하므로 $\Theta_k - \Theta_0$가 항상 rank $\le r$인 다양체 위에 머문다. (ii) $\mathcal{R}_k$는 사람이 만든 고정 labeled dataset이며 $\mathrm{gen}(\cdot;B_s)$가 항등이다 — **$B_s$라는 knob 자체가 없다.** (iii) 첨자 $k$가 sleep 라운드가 아니라 한 번의 adaptation run 안의 평범한 optimizer step $s$다. 라운드가 반복되지 않으므로 누적 열화 문제가 제기되지 않는다 [LoRA App. D: GPT-3은 2 epoch, GPT-2는 5 epoch].

### 19.3.2 직접 편집 — 학습률이 없는 쓰기

[ROME]은 (U-$\Theta$)의 극단적 특수형이다. gradient step이 아니라 **등식제약 최소제곱의 닫힌 해**로 단 하나의 행렬에 rank-one delta를 더한다. 쓰기 문제의 정식화가 먼저다.

$$
\min_{\widehat\Theta^{(l^*)}_{\mathrm{proj}}} \big\| \widehat\Theta^{(l^*)}_{\mathrm{proj}} K - V \big\|
\quad \text{s.t.} \quad
\widehat\Theta^{(l^*)}_{\mathrm{proj}}\, k_* \;=\; v_*
\tag{19-2}
$$

기존 key–value 대응은 최소제곱으로 유지하면서 새 쌍만 등식제약으로 강제한다 [ROME Eq. 2, Eq. 7]. "층에 저장된 다른 기억과의 간섭을 최소화하면서"라는 표현이 이 제약형에서 나온다 [ROME Fig. 4 캡션]. 해는 닫혀 있다.

$$
\Delta\Theta_k \;=\; \frac{\big(v_* - \Theta^{(l^*)}_{\mathrm{proj}} k_*\big)\,\big(C_K^{-1} k_*\big)^{\!\top}}{\big(C_K^{-1} k_*\big)^{\!\top} k_*},
\qquad
\operatorname{rank}(\Delta\Theta_k) = 1
\tag{19-3}
$$

[ROME Eq. 2, 유도는 App. A Eq. 5–17]. $C_K = KK^{\!\top}$는 기존 키 분포의 uncentered second moment이며, Wikipedia 2020-05-01 스냅샷에서 뽑은 hidden state 100,000개로 한 번 추정해 캐시하고 이후 모든 편집이 같은 것을 재사용한다 [ROME App. E.5].

$k_*$는 학습되지 않는다 — 무작위 prefix를 붙인 텍스트들에 대해 subject 마지막 토큰의 활성을 평균한 값이다 [ROME Eq. 3]. $v_*$만 최적화되는데, **최적화 변수가 가중치가 아니라 활성 벡터 하나**다.

$$
v_* \;=\; \arg\min_{z}\; \ell_{\mathrm{ROME}}(z),
\qquad
\ell_{\mathrm{ROME}}(z) \;=\; -\tfrac{1}{N}\!\sum_j \log P\big[o_*\,\big|\,x_j + p\big] \;+\; \lambda_{\mathrm{KL}}\, D_{\mathrm{KL}}\big(\cdot\,\|\,\cdot\big)
\tag{19-4}
$$

첫 항이 새 객체의 확률을 올리고, 둘째 항이 "{subject}는 …"류 프롬프트의 예측 분포를 원 모델에 묶어 subject의 정체가 표류하는 것을 막는다 [ROME Eq. 4]. 논문이 명시한다 — "이 최적화는 모델 가중치를 직접 바꾸지 않는다. 벡터 표현 $v_*$를 찾을 뿐이다" [ROME §3.1].

**구조적 결론 하나를 여기서 못 박는다. (19-3)에는 $\eta_\Theta$가 존재하지 않는다.** 반복도 없다. $\Theta$-경로의 쓰기가 반드시 SGD여야 하는 것은 아니라는 존재증명이며, 이것이 (U-$\Theta$) 표준형과의 가장 큰 차이다. 부속 최적화(19-4)의 learning rate 0.5는 활성 벡터 $z$에 대한 것이므로 $\eta_\Theta$로 번역해서는 안 된다 [ROME App. E.5].

[MEMIT]은 이 구조에서 두 가지를 바꾼다. 첫째, 등식제약을 목적함수로 흡수한다.

$$
\Theta_1^{\mathrm{out}} \;=\; \arg\min_{\hat\Theta}\Big(\underbrace{\textstyle\sum_{i \le n}\big\lVert\hat\Theta k_i - m_i\big\rVert^2}_{\text{옛 연관 보존}} \;+\; \underbrace{\textstyle\sum_{i > n}\big\lVert\hat\Theta k_i - m_i\big\rVert^2}_{\text{새 연관 삽입}}\Big)
\tag{19-5}
$$

"Meng et al. (2022)과 달리 우리는 새 연관 하나만 더하는 제약으로는 문제를 풀 수 없으므로 확장된 목적함수를 정의한다" [MEMIT §4.2, Eq. 9]. 해는 다시 닫혀 있고,

$$
\Delta^{(l)} \;=\; \mathrm{Res}\, K_1^{\!\top}\big(\Gamma_0 + K_1 K_1^{\!\top}\big)^{-1},
\qquad
\mathrm{Res} \;=\; M_1 - \Theta_0^{\mathrm{out}} K_1,
\qquad
\Gamma_0 \;=\; \lambda\,\mathbb{E}_k\big[k k^{\!\top}\big]
\tag{19-6}
$$

[MEMIT Eq. 14, Eq. 15]. 접근할 수 없는 사전학습 (key, value) 쌍이 유도 과정에서 소거되고 **$K_0$의 2차 통계만 남는다**는 것이 이 식의 관건이다. 즉 replay 데이터 없이 replay 효과를 통계로 근사한다 — $\Theta$-경로에서 $\mathcal{R}_k$를 명시적 데이터 없이 대체한 드문 사례다. $\lambda$는 새것 대 옛것의 가중이며 전형값이 $1.5\times10^4$라고 논문이 쓴다 [MEMIT §4.2]. **구조적으로 $\lambda$는 (U-W)의 $\alpha_t$가 하던 retention gate 역할을 한다** — $\lambda\to\infty$면 $\Delta^{(l)}\to0$이고, 이것이 $\Theta$-경로에서 유일하게 명시적인 retention 손잡이다.

둘째, 쓰기를 한 층이 아니라 임계 MLP 층 집합 $\Lambda$에 분산한다. 근거는 residual stream 전개다 — 각 MLP가 최종 은닉 상태에 **덧셈으로** 기여하므로 원하는 변화량을 여러 층에 나눠 실을 수 있다 [MEMIT Eq. 6, §4.1]. 분배 규칙은 균등이다.

$$
r^{(l)}_i \;=\; \frac{z_i - h^{l^*}_i}{\,l^* - l + 1\,}
\tag{19-7}
$$

목표 잔차를 남은 층 수로 나눈다 [MEMIT Eq. 20]. 동기는 "파라미터 변화량을 작게 유지하면 robust해진다"는 선행 관찰이며, [MEMIT §4.3]이 풀겠다고 세운 최적화 문제를 실제로 풀지 않고 이 heuristic으로 근사한다.

그리고 **순서가 본질이다.** 한 층을 고치면 아래층 활성이 전부 바뀌므로, $\Lambda$를 오름차순으로 돌며 층마다 활성을 다시 수집한다 [MEMIT §4.3, Algorithm 1]. 이것이 (U-$\Theta$) 1회를 $|\Lambda|$회의 직렬 sub-pass로 만들고, 배치화가 불가능한 유일한 축이 된다.

$z_i$는 활성 공간의 목표이며 이 논문에서 gradient descent가 도는 유일한 곳이다 [MEMIT Eq. 16]. 즉 두 단계다 — **활성 공간에서 목표를 gradient로 정하고, 가중치 공간으로는 대수로 사영한다.** 표준형에 없는 분리이며, 세 논문 중 직접 편집 두 편이 공유하는 구조다.

### 19.3.3 쓸 자리를 정하는 계측기

두 편집 논문 모두 "어디에 쓸 것인가"를 학습하지 않는다. causal mediation analysis로 재고 사람이 읽어 고정한다. [ROME]은 clean run / corrupted run / corrupted-with-restoration run 세 번을 돌려 각 은닉 상태의 indirect effect를 재고 사실 1000개에 대해 평균한다 [ROME §2.1]. GPT-2 XL에서 average total effect는 18.6%, 개별 은닉 상태의 최대 average indirect effect는 15번째 층·subject 마지막 토큰에서 8.7%, MLP 기여 최대는 6.6%, 같은 토큰에서 attention은 1.6%다 [ROME §2.2]. 편집 층은 이 지도를 보고 18로 고정된다 [ROME App. E.5]. [MEMIT]은 같은 계측을 GPT-J에 다시 수행해 층 하나가 아니라 **범위**를 고른다 — "Meng et al. (2022)이 이 검정으로 단일 편집 층을 식별한 것과 달리, 우리는 임계 MLP 층의 전 범위를 고른다" [MEMIT §4.1]. 결과는 GPT-J $\Lambda=\{3,4,5,6,7,8\}$, GPT-NeoX $\Lambda=\{6,7,8,9,10\}$이다 [MEMIT App. B.4]. **모델이 바뀌면 사람이 다시 정해야 한다[본서 추론].**

### 19.3.4 표기 대응표

표 19-1. 원 논문 기호 → 이 책 기호. [LoRA]의 대응표는 ch03이 소유하므로 여기서는 편집 두 편만 옮기고, 두 논문에서 같은 자리를 차지하는 행은 논문 열로 합쳤다(의미 변경·삭제 없음). 마지막 두 행은 이 장에서 새로 생긴 장-국소 행이다.

| 원 논문 기호 | 논문 | 원 의미 | 이 책 기호 | 심각도 |
|---|---|---|---|---|
| $W$, $W^{(l)}_{\mathrm{proj}}$, $W^{(l)}_{\mathrm{fc}}$ | ROME §3.1 | 편집 대상 MLP 행렬 | $\Theta^{(l^*)}_{\mathrm{proj}}$, $\Theta^{(l^*)}_{\mathrm{fc}}$ | 치명 — 놓치면 $W$-경로로 오분류된다 |
| $W_y$, $W^l_{\mathrm{out}}$, $W^l_{\mathrm{in}}$, $W_0$, $W_1$, $\hat W$ | MEMIT Eq. 1·4·7·9 | 가중치 행렬 일반 | $\Theta_y$, $\Theta^{(l)}_{\mathrm{out}}$, $\Theta^{(l)}_{\mathrm{in}}$, $\Theta^{\mathrm{out}}_0$, $\Theta^{\mathrm{out}}_1$, $\hat\Theta$ | 치명 — 전부 slow weights의 부분집합 |
| $C$ / $C_0$, $C^l$ | ROME Eq. 2 / MEMIT Eq. 15 | 옛 key의 uncentered covariance | $C_K$ / $\Gamma_0$, $\Gamma^{(l)}$ | 치명 — $C$(chunk)·$C_{\mathrm{cap}}$(용량)과 충돌 |
| $E$ | ROME App. D | 이웃 subject 집합 | 기호 없이 산문으로 | 치명 — 3층 프레임 기호를 덮는다 |
| $E$ | MEMIT Eq. 5 / Eq. 1 | 편집 요청 집합 / 마지막 토큰 인덱스 | $\mathcal{E}_{\mathrm{edit}}$ / $T_{\mathrm{end}}$ | 치명 + 논문 내부 이중사용 |
| $S$ | ROME §3.3 | ES·PS·NS의 조화평균 | "종합점수"로 풀어 씀 | 치명 — 맨 $S$ 금지 |
| $S$ | MEMIT §4 / §5.2.2 / Algorithm 1 | 마지막 subject token / Editing Score / 편집 층 집합 | $t_S$ / $\mathrm{Score}$ / $\Lambda$ | 치명 + 논문 내부 삼중사용. Algorithm 1 헤더와 본문(§4.1의 $R$)이 서로 다른 기호를 쓴다 |
| $R$ | MEMIT §4.1 / §4.2 | 임계 MLP 층 집합 / 잔차 행렬 | $\Lambda$ / $\mathrm{Res}$ | 높음 + 논문 내부 이중사용. $\mathcal{R}_k$(sleep 학습집합)와도 충돌 |
| $D$ | MEMIT Eq. 1 | transformer 층 수 | $L_{\mathrm{layer}}$ | 중 — 이 책에서 $L$은 sequence 길이다 |
| $L \triangleq \max(R)$ | MEMIT Eq. 16 | 목표 층 | $l^*$ | 치명 — 개명 없이는 식 (19-7)이 읽히지 않는다 |
| $H$, $D$ | ROME App. A | 은닉 폭, MLP 확장 폭 | $d$, MLP 확장 폭 | 중요 — **논문이 두 값의 수치를 명시하지 않아 $C_{\mathrm{cap}}$ 계산이 막힌다** |
| $r$ (in $(s,r,o)$) | ROME §2 / MEMIT §3 | relation | $\varrho$ | 높음 — $r$은 rank 예약 |
| $r^l_i$ | MEMIT Eq. 20 | 층 $l$에 실을 잔차 벡터 | $r^{(l)}_i$ (첨자 필수) | 중 + 논문 내부 삼중사용 |
| $k_*$, $v_*$, $K$, $V$ | ROME §3.1 | linear associative memory의 키·값 | 별표 유지 $k_*$, $v_*$ | 높음 — $k_t, v_t$(U-W)와 시간척도가 다르다 |
| $k_i$, $k^l_i$ / $m_i$, $m^l_i$ | MEMIT Eq. 19·20 | subject key / 기억된 값 | $k^{(l)}_i$ / $m^{(l)}_i$ | 중 — $k$(sleep 라운드 첨자)와 구분해 항상 첨자를 단다 |
| $L(z)$ | ROME Eq. 4 | $v_*$ 최적화 목적함수 | $\ell_{\mathrm{ROME}}(z)$ | 중 — $\mathcal{L}$도 $\ell$도 아니다 |
| $\lambda$ | ROME App. E.5 / MEMIT Eq. 15 | KL 항 계수 $1\times10^2$ / covariance 가중 | $\lambda_{\mathrm{KL}}$ / $\lambda$ | 낮음. 후자는 $\alpha_t$의 $\Theta$-경로 대응물 |
| $\eta$ | ROME / MEMIT | 부속 최적화의 학습률(0.5) | $\eta_\Theta$로 **번역 금지** | 치명 — $\Theta$ 갱신에는 학습률이 존재하지 않는다 |
| $u$ / $n$ | MEMIT Eq. 9 / Eq. 7·§5.2.2 | 새 연관 개수 / 기존 연관 수·편집 수 | $n_{\mathrm{edit}}$ / 문맥별 명시 | 중 + 논문 내부 이중사용 |
| $P$ / $\sigma$ | MEMIT Eq. 16 / Eq. 4·App. A | prefix 개수 / 비선형성·임베딩 표준편차 | $N_{\mathrm{pref}}$ / $\sigma$·$\sigma_{\mathrm{emb}}$ | 중 + 논문 내부 충돌 |
| $\Lambda$ | ROME App. A | rank-one 갱신의 좌벡터 | **본문에서 쓰지 않고 (19-3)에 전개**했다 | 장-국소 — MEMIT의 $\Lambda$(임계 층 집합)와 충돌한다 |
| $G$, $G'$ | ROME §2·§3.2 | 편집 전·후 모델 | $f(\cdot;\Theta_k)$, $f(\cdot;\Theta_{k+1})$ | 장-국소 — 편집이 배포 아티팩트를 바꾼다는 사실을 이 대응이 그대로 드러낸다 |

---

## 19.4 어느 층을 언제 쓰는가

세 논문 전부 (U-$\Theta$) 단독이다. 표 19-2가 "이 값은 누가 학습하는가"에 전부 답한다.

표 19-2. 세 논문의 객체별 (누가·언제·무슨 규칙). $W$와 $E$ 행이 셋 다 비어 있다는 것이 이 표의 첫 결론이다.

| 객체 | 논문 | 누가 정하는가 | 언제 | 규칙 |
|---|---|---|---|---|
| $\Theta_0$ (backbone 전체) | 셋 다 | 이 논문들 밖의 pre-training | 범위 밖. 이후 동결 | — |
| $A^{(m)}, B^{(m)}$ | LoRA | outer-loop 학습(AdamW) | adaptation 구간. test-time 불변 | (U-$\Theta$)를 rank-$r$ 부분공간에 사영 |
| merge된 $\Theta_0^{(m)}+\tfrac{\alpha_{\mathrm{LoRA}}}{r}B^{(m)}A^{(m)}$ | LoRA | 학습되지 않음. 배포 시 1회 **계산** | deploy 시점 | 산술 |
| $r$, $\alpha_{\mathrm{LoRA}}$, $\mathcal{S}$ | LoRA | 사람. heuristic | adaptation 이전 | — |
| $C_K$ / $\Gamma^{(l)}$ | ROME / MEMIT | 학습이 아니라 **표본 추정**. Wikipedia 100,000 / Wikitext 100,000(GPT-J)·50,000(GPT-NeoX) | 모든 편집 이전, 모델당 1회 | 2차 통계 |
| $l^*$ / $\Lambda$ | ROME / MEMIT | 학습되지 않음. **causal tracing 지도를 사람이 읽어 고정** | 모델당 1회 | — |
| $\lambda$, $\lambda_{\mathrm{KL}}$ | MEMIT / ROME | 사람이 고르는 hyperparameter | 편집 이전 | retention 강도 |
| $k_*$ / $k^{(l)}_i$ | ROME / MEMIT | 학습 아님. forward pass로 읽어 prefix 평균 | 편집 시점, 사실마다 | (19-3)·(19-6)의 입력 |
| $v_*$ / $z_i$ | ROME / MEMIT | **gradient descent** — 단, 변수는 가중치가 아니라 활성 벡터 | 편집 시점, 사실마다. MEMIT은 편집 간 독립이라 병렬 가능 | $\ell_{\mathrm{ROME}}(z)$ / [MEMIT Eq. 16] |
| $\Theta^{(l^*)}_{\mathrm{proj}}$ / $\Theta^{(l)}_{\mathrm{out}},\,l\in\Lambda$ | ROME / MEMIT | **아무도 학습하지 않는다.** 닫힌 해를 계산해 더한다 | sleep 라운드 1회. MEMIT은 층 오름차순 순차 | (19-3) / (19-6) |
| $\mathcal{R}_k$ (무엇을 쓸 것인가) | 셋 다 | **생성되지 않는다.** 사람이 만든 dataset·데이터셋 레코드에서 온다 | 편집·학습 입력 | $\mathrm{gen}(\cdot;B_s)$가 항등 |
| $W$ (fast weights) | 셋 다 | 존재하지 않는다 | — | — |
| $E$ (external store) | 셋 다 | 존재하지 않는다 | — | — |

시계와 순서를 산문으로 정리한다.

**[LoRA]의 순서는 단일하다.** 고정 dataset을 받아 $(A,B)$에 대해 optimizer step $s$를 돌고, 배포 직전 merge한다. 층 간 순서 문제가 없다 — 적용 행렬들이 서로 독립이다.

**[ROME]의 순서는 넷이다.** (a) 모델당 1회 $C_K$ 추정, (b) 모델당 1회 causal tracing으로 $l^*$ 고정, (c) 편집마다 $k_*$ 읽기, (d) 편집마다 $v_*$ 최적화 후 (19-3)으로 쓰기. (a)와 (b)가 고정비이고 (c)·(d)가 한계비다.

**[MEMIT]의 순서는 안쪽에 직렬 구간을 갖는다.** $\Gamma^{(l)}$ 사전 추정 → 모든 $i$에 대해 $z_i$ 산출(편집 간 완전 독립, 논문 표현으로 "embarrassingly parallel" [MEMIT §5.2.2]) → $l=\min\Lambda$부터 $\max\Lambda$까지 오름차순으로 $K^{(l)}$·$\mathrm{Res}^{(l)}$을 **재수집**한 뒤 $\Delta^{(l)}$ 적용. 마지막 단계의 순서를 바꾸면 활성 재수집 전제가 깨진다.

두 가지를 명시한다. 첫째, **세 논문 모두 sleep 라운드가 $k=1$에서 멈춘다.** [ROME]은 정의상 한 번에 사실 하나만 편집하고 [ROME §3.7], [MEMIT]은 배치 편집을 한 번 실행하고 끝나며 반복 적용 실험이 없다. [LoRA]는 delta 하나를 학습하고 두 delta를 누적하거나 합성하는 실험을 하지 않는다. 그러므로 **$\rho$를 잴 실험이 이 장의 어느 논문에도 없다**(→ §19.5).

둘째, **$\mathrm{gen}(\cdot;B_s)$가 셋 다 항등이다.** [LoRA]의 $\mathcal{R}$은 GLUE·WikiSQL·E2E 같은 기성 dataset이고, [ROME]·[MEMIT]의 $\mathcal{R}$은 zsRE·CounterFact 레코드다. 무엇을 쓸지는 언제나 외부에서 주어진다.

> **[평가]** 이것이 이 장의 위치를 정한다. **이 장은 $\Theta$에 쓰는 도구를 완성하지만 무엇을 쓸지는 아무것도 제공하지 않는다.** 세 논문 중 어느 것도 예산 $B_s$를 써서 학습 집합을 만드는 단계를 갖지 않으므로, ch11의 판별식 조건 (2)를 만족시킬 방법이 이 장 안에는 없다(→ ch11 §11.3). 그 빈칸을 채우는 것이 다음 장의 일이다.

---

## 19.5 비용 4종

표 19-3. 세 논문의 비용 4종. 논문이 침묵한 칸은 "논문에 없음"이며 추정치를 넣지 않았다.

| | [LoRA] | [ROME] | [MEMIT] |
|---|---|---|---|
| $B_s$ | 논문에 없음(FLOPs·토큰·wall-clock 전부). 보고된 것은 학습 throughput 32.5 → 43.1 tokens/s per V100과 VRAM 1.2TB → 350GB뿐 [LoRA §4.2, 각주 5] | GPT-2 XL 편집 1회 **약 2초**(NVIDIA A6000) [ROME App. E.5]. FLOPs·토큰 없음, GPT-J 편집 시간 없음, $C_K$ 추정 비용 없음 | GPT-J 6B, 10,000 편집 **총 7.44 hr** [MEMIT §5.2.2]. 분해: $z_i$ 산출 23,546.65초 + 나머지 갱신 3,226.35초 [App. B.4]. GPT-NeoX 실행 시간 없음 |
| $L_w$ | merge된 형태에서 **0** — "구성상 fine-tune된 모델 대비 추가 지연을 전혀 도입하지 않는다" [LoRA §4.1] | 논문에 없음. 편집 후 추론 지연을 한 번도 재지 않는다. 유일한 시간 수치(2초, 100ms)는 **편집 비용**이다 | 논문에 없음. latency·throughput·TTFT가 전문에 없다 |
| $C_{\mathrm{cap}}$ | $\lvert\Delta\Theta\rvert = 2N_{\mathrm{mod}}\,d\,r$. GPT-3 175B·$r{=}4$·$\mathcal{S}=\{\Theta^{(q)},\Theta^{(v)}\}$에서 350GB → **35MB** [LoRA §4.2, §5.1] (정본은 → ch03 §03.6) | 논문에 없음. 바이트 수치가 0건이다. 갱신이 rank 1이므로 상태량은 두 벡터(좌벡터는 은닉 폭 차원, 우벡터는 MLP 확장 폭 차원)인데 **논문이 두 차원의 수치를 어디에서도 밝히지 않는다** [ROME App. A] | 바이트는 논문에 없음. 논문이 보고하는 용량 단위는 **편집 개수**이고 그 값이 10,000이다 [MEMIT Table 1, Table 2] |
| $\rho$ | 논문에 없음. 순차 적응·delta 누적·반복 write 실험이 0건 | 논문에 없음. 모든 표가 편집 1회 후의 값이다 | 논문에 없음. 반복 라운드 실험이 없다. 유일한 반복 열화 증거는 **경쟁 기법**의 것 — 순차 ROME은 zsRE Specificity 0.9 (±0.1), Score 2.6으로 붕괴한다 [MEMIT Table 1] |

식 (A)를 이 장에 적용하면 세 논문 모두에서 **손익분기 $N_q^*$가 정의되지 않는다.** 이유가 E-경로와 정반대다. 직접 편집은 $C_{\text{wake}}$를 바꾸지 않으므로 — 갱신이 $\Theta^{(l)}_{\mathrm{out}} \mathrel{+}= \Delta^{(l)}$ 라는 **같은 shape의 제자리 덧셈**이다 [MEMIT Algorithm 1] — 상각은 "손익분기"가 아니라 "한 번 지불하고 무한히 상각"의 형태가 된다. 즉 $N_q \to \infty$ 극한이며, 분모에 들어갈 것은 질의 수가 아니다.

> **[해설]** $\Theta$-경로에서 상각의 실제 분모는 **한 번의 sleep 라운드에 묶어 넣은 편집 수**다. [MEMIT]의 진짜 기여를 이 언어로 옮기면 그 분모를 1에서 10,000으로 키운 것이다. 편집당 시간은 26,773.00초 ÷ 10,000 = **2.6773초**이고, $z_i$ 단계가 총 시간의 **87.95%**를 차지한다. **이 두 값은 이 책의 나눗셈이다** — 논문은 자기 방법에 대해서는 총 시간과 두 단계의 초만 보고하고 편집당 값도 비율도 계산하지 않는다. 대조군에는 같은 나눗셈을 한다 — "ROME은 GPT-J 10,000 편집에 44,248.26초 ≈ 12.29시간이 걸리며, 이는 편집당 약 4초에 해당한다" [MEMIT App. B.3]. 그 4초가 이 장에서 유일하게 논문이 인쇄한 편집당 비용이고, MEMIT의 2.6773초는 그보다 낮다.

[ROME]에서 실제로 상각되는 항은 하나뿐이다. $C_K$(hidden state 100,000개 표본 추정)는 편집 횟수에 상각되고 편집당 한계비는 약 2초다. 반면 hypernetwork 계열은 반대 구조다 — "KE와 MEND 같은 hypernetwork는 추론 시점에는 훨씬 빠르지만(100ms 수준) **수 시간에서 수 일**의 추가 학습 오버헤드를 요구한다" [ROME App. E.5]. 즉 두 계열의 비교는 고정비 대 한계비의 문제이고, **손익분기 편집 수를 논문이 계산하지 않으며 "수 시간에서 수 일"이라는 범위가 너무 넓어 원문 수치만으로는 계산할 수도 없다.**

비용 회계에서 빠진 항이 어느 논문에나 있다. [ROME]은 $C_K$ 추정 비용과 causal tracing 비용을 값으로 주지 않는다. [MEMIT]은 covariance 추정(Wikitext 100,000 표본)과 causal tracing(진술 501개 × 노이즈 표본 10개)이 7.44 hr에 포함되는지 명시하지 않고, "가장 계산이 비싼 단계"로 지목한 대형 정사각 행렬의 역행렬에 개별 시간을 붙이지 않는다 [MEMIT App. B.4]. **더 큰 빈칸은 쓰기 검증이다.** [ROME]은 사람 15명이 반사실 시나리오 50개를 평가했고 편집 후 모델이 근거 없는 새 사실을 지어낸다고 인정하지만 [ROME §3.6, §3.7], 그 검증 비용은 어떤 비용 수치에도 들어 있지 않다.

> **[평가]** "2초"는 쓰기 연산의 비용이지 **올바르게 썼는지 확인하는 비용**이 아니다. $\Theta$-경로에서 쓰기가 싸다는 인상은 검증을 회계 밖에 둔 결과이며, 이 장의 세 논문 중 어느 것도 그 항을 세지 않는다. Part III(ch25·ch27)가 이 항을 회계 안으로 들여온다.

같은 누락이 후속 문헌에서 반복된다는 것을 기록해 둔다. `Understanding LoRA as Knowledge Memory`(arXiv:2603.01097, 이하 [LoRA-KM])의 유일한 비용 비교는 문서 하나에 대한 질문 30개의 벽시계 시간인데, 구성요소 목록이 모델 로딩·adapter 로딩·RAG 설정·질의 임베딩·검색·merge·활성화·토큰화·추론으로 되어 있고 **학습 항이 없다** [LoRA-KM App. U]. 즉 그 비교는 식 (A)에서 $C_{\text{sleep}}=0$, $N_q=30$으로 놓은 것이다. 그리고 그 비교 안에서 RAG가 논문이 제안한 multi-LoRA preloaded보다 빠르다 — FlashAttention 적용 시 43.5초 대 65.9초, 미적용 시 86.1초 대 109.3초인데 논문은 ICL과만 비교한다 [LoRA-KM Fig. 22, §7 Q15].

---

## 19.6 실험과 스케일

### 19.6.1 [ROME] — 사실 하나를 넣는 데 드는 것

실증 상한부터 적는다. **편집 실증의 상한은 GPT-J 6B다** [ROME Table 4]. causal tracing은 GPT-NeoX 20B까지 올라가지만 그 모델의 편집 결과는 없다 [ROME App. B.3]. 편집 규모의 상한은 회당 사실 1개이고, 지식 종류의 상한은 factual association이며 "논리·공간·수치 지식 같은 다른 종류의 학습된 믿음은 조사하지 않았다"고 논문이 쓴다 [ROME §3.7].

헤드라인은 저자 제작 벤치마크 위의 값이다. CounterFact는 21,919 레코드로 저자가 만들었고 종합점수의 가중(ES·PS·NS의 조화평균)도 저자가 정했다 [ROME §3.3, App. D]. 그 위에서 ROME은 GPT-2 XL 종합점수 89.2, GPT-J 91.5로 모든 기준선을 앞선다 [ROME Table 4]. **표준 벤치마크에서는 결론이 다르다.** zsRE paraphrase에서 ROME 88.1 (±0.5)은 MEND-zsRE 99.3 (±0.1)과 KE-zsRE 90.0 (±0.3)에 진다 [ROME Table 1]. 저자는 이들이 zsRE 분포에 맞춰 학습되었기 때문이라고 설명한다 [ROME §3.2]. 스케일 하단에서도 역전이 있다 — GPT-2 Medium의 zsRE efficacy에서 ROME 96.6 (±0.2)이 FT+L 97.2 (±0.2)보다 낮다 [ROME Table 6].

사실 **하나**를 넣는 대가가 이미 측정되어 있다. GPT-2 XL에서 편집 후 specificity는 75.4 (0.7)로 편집 전 78.1 (0.6)보다 낮고, fluency는 621.9 (0.5)로 편집 전 626.6 (0.3)보다 낮다 [ROME Table 4]. 그리고 사람 평가가 자동 지표를 뒤집는다 — 평가자는 ROME을 FT+L보다 삽입된 사실에 일관적이라고 볼 확률이 1.8배 높지만, **더 유창하다고 볼 확률은 1.3배 낮고**, 저자는 이 손실이 "우리의 다른 지표에 포착되지 않는다"고 인정한다 [ROME §3.6].

쓰기 주소 규칙이 보편적이지 않다는 것도 논문 안에 있다. Windows Media Player는 마지막 토큰이 아니라 'Windows'가 결정적이고, Mitsubishi Electric은 'Mitsubishi'가, Madame de Montesson은 'Madame'이 결정적이며, subject 안에 결정적 토큰이 아예 없는 사례와 마지막 subject 토큰의 효과가 **음수**인 사례가 있다 [ROME Fig. 11, Fig. 14]. **논문은 그 비율을 보고하지 않는다.** 자동화된 $\Theta$ 쓰기 파이프라인에서 이것은 주소 실패율이 된다[본서 판단].

### 19.6.2 [MEMIT] — 편집 용량의 실측 상한

이 장이 소유하는 두 번째 정의를 여기서 못 박는다.

> **정의.** **편집 용량의 실측 상한**은 한 번의 (U-$\Theta$) 라운드에 써넣었을 때 efficacy·generalization·specificity 세 지표가 동시에 무너지지 않는 최대 편집 수다. 이 값은 아키텍처의 성질이 아니라 **쓰기 알고리즘의 성질**이며, 붕괴가 관측되지 않은 sweep의 끝점은 상한이 아니라 **관측된 하한**이다.

[MEMIT]의 sweep은 log-scale로 $n \in \{1,\allowbreak 2,\allowbreak 3,\allowbreak 6,\allowbreak 10,\allowbreak 18,\allowbreak 32,\allowbreak 56,\allowbreak 100,\allowbreak 178,\allowbreak 316,\allowbreak 562,\allowbreak 1000,\allowbreak 1778,\allowbreak 3162,\allowbreak 5623,\allowbreak 10000\}$이며 [MEMIT §5.2.2 각주 1], 10,000에서 끝난다. 그 지점에서 MEMIT는 붕괴하지 않는다. 반면 기준선의 붕괴점은 명시된다 — "ROME은 $n=10$까지는 잘 작동하지만 $n=32$부터 열화한다", "MEND는 $n=1$에서 잘 작동하다가 $n=6$에서 급락하고 $n=1{,}000$ 전에 efficacy를 전부 잃는다" [MEMIT §5.2.2]. 따라서 **10,000은 MEMIT의 한계가 아니라 이 corpus가 인용할 수 있는 관측된 하한이다[본서 추론].**

10,000 편집 지점의 수치를 그대로 옮긴다 [MEMIT Table 2, CounterFact]. GPT-J 편집 없음: 종합점수 22.4 / ES 15.2 (0.7) / PS 17.7 (0.6) / NS 83.5 (0.5) / GE 622.4 (0.3) / RS 29.4 (0.2). MEMIT: 85.8 / 98.9 (0.2) / 88.6 (0.5) / **73.7 (0.5)** / 619.9 (0.3) / 40.1 (0.2). GPT-NeoX 20B에서는 82.0 / 97.2 (0.8) / 82.2 (1.6) / **70.8 (1.4)** / 606.4 (1.0) / 36.9 (0.6)이며 편집 없음 대비 NS가 81.6에서 70.8로 내려간다. **즉 편집하지 않은 이웃 주어의 정답률이 10pt 가까이 떨어진다[본서 산술].** 이것이 $\Theta$-경로 대량 쓰기의 실측 부작용이다.

그리고 다른 벤치마크에서는 그 값이 훨씬 나쁘다.

> **zsRE에서 MEMIT의 종합점수는 50.7이다** [MEMIT Table 1]. 분해하면 Efficacy 96.7 (±0.3), Paraphrase 89.7 (±0.5), **Specificity 26.6 (±0.5)**이고, 조화평균이 이 낮은 항에 지배된다. 그런데 그 26.6은 **편집하지 않은 GPT-J의 27.0 (±0.5)보다 낮다.** 논문은 이 격차를 편집 없는 모델의 zsRE specificity 자체가 낮다는 사실과 연결해 설명하지 않고 "bleedover가 최소"라고만 쓴다 [MEMIT §5.2.1].

기준선과의 비교에도 유보가 붙는다. naive fine-tuning(FT-W)이 efficacy에서 MEMIT를 앞서고(99.4 대 98.9), 시간도 1,716.21초로 26,773.00초보다 약 15.6배 빠르다(배수는 이 책의 나눗셈이다) [MEMIT Table 2, App. B.1, App. B.4]. MEMIT의 우위는 efficacy가 아니라 specificity(73.7 대 46.9)와 fluency(619.9 대 293.9)에 있다. 작은 $n$에서는 ROME이 generalization으로 MEMIT를 이기며, 저자는 이를 ROME의 hard equality 제약 덕분이라고 설명한다 [MEMIT §5.2.2] — **soft least-squares 완화는 공짜가 아니다.** 선행 최대 규모 주장의 출처인 SERAC과는 공개 코드가 없어 비교하지 않았고 [MEMIT §5.1], $\lambda$ ablation은 역전 관계다 — "$\lambda$가 커지면 specificity와 fluency가 단조 증가하지만 동시에 efficacy와 generalization이 떨어진다" [MEMIT App. F.3]. **모든 축에서 좋은 $\lambda$는 없으며, 보고된 운용점은 조화평균이라는 저자의 집계 선택이 최대가 되는 지점이다.**

마지막으로 평가 범위의 상한이다. **일반 능력 평가가 전혀 없다** — perplexity도 downstream benchmark도 0건이고, 모델 손상의 유일한 대리 지표가 bi/tri-gram 엔트로피다 [MEMIT App. C.2]. GPT-NeoX 20B는 데이터 점이 하나뿐이며 scaling curve·카테고리 실험·ablation·실행 시간이 전부 GPT-J에서만 수행됐다. 그리고 저자가 지식 범위를 명시한다 — "우리가 다루는 지식 표현은 방향성 (s, r, o) 관계로 범위가 제한된다. 공간·시간 추론, 수학 지식, 언어 지식, 절차 지식, **심지어 대칭 관계도** 다루지 못한다" [MEMIT §6]. 즉 "10,000개 memory"는 10,000개 삼중항이지 일반적 의미의 기억 10,000개가 아니다[본서 추론].

### 19.6.3 delta 방식의 실측 — 후속 문헌이 보고한 것

[LoRA] 자신의 표에도 유보가 있다. GLUE에서 QQP는 두 RoBERTa 크기 모두에서 full fine-tuning보다 낮고(base 90.8 대 91.9, large 91.6 대 92.2), **Adapter가 LoRA를 이기는 행이 존재한다** — RoBERTa large의 QQP에서 Adapter$_H$(6.0M) 92.1이 LoRA†(0.8M) 91.6보다 높고, SST-2에서 Adapter$_P$(0.8M) 96.6이 LoRA† 96.2보다 높다 [LoRA Table 2]. (†는 Houlsby et al. (2019)에 맞춘 제한 setup이다 — 전 과제 공통 batch size, sequence 길이 128 고정, 그리고 MRPC·RTE·STS-B에서 MNLI checkpoint warm-start 제거 [LoRA §5.1, App. D.1].)

[LoRA-KM]은 저랭크 delta를 지식 저장소로 쓸 때의 경계를 측정한다. 결과가 이 장의 논지에 불리한 쪽으로 선명하다.

**첫째, RAG에 LoRA를 붙이면 RAG가 무너진다.** QuALITY에서 Llama-3.1-8B는 RAG 63.79 → Single LoRA+RAG **42.42**로 21.4점 떨어지고, Qwen3-8B는 64.53 → 44.10, Qwen3-1.7B는 43.43 → 33.15다 [LoRA-KM Table 2]. 세 큰 모델에서 방향이 같다. 남은 Llama-3.2-1B만 26.12 → 29.86으로 반대인데, 그 29.86은 아래 둘째 항의 복사 정황에 걸린 값이다 [LoRA-KM Table 2]. 그리고 **논문 본문은 이 패턴을 한 번도 언급하지 않는다.** 오히려 구현 함의 상자에는 "LoRA는 외부 문맥과 결합할 때 더 강한 성능을 내며 standalone LoRA·RAG·ICL을 능가한다"고 적혀 있다 [LoRA-KM §7 Q13].

**둘째, 그 42.42는 복사 오류의 정황을 띤다.** 같은 값이 Llama-3.1-8B QuALITY의 Single LoRA closed-book, Multi Top3 closed-book, Single LoRA+RAG open-book 세 칸과 [Table 6]의 QuALITY Embedding Top-3까지 네 번 등장한다. 나아가 QuALITY 열에서는 **네 모델 전부** "Single LoRA + RAG" 행이 같은 모델의 "Single LoRA" closed-book 행과 **정확히 같다**(29.86 / 42.42 / 33.15 / 44.10) [LoRA-KM Table 2]. 문맥을 붙였는데 네 모델의 소수점 둘째 자리까지 하나도 움직이지 않는 것은 측정이 아니라 복사의 서명이다. 이 장은 이 사실을 매끄럽게 넘기지 않고 그대로 적는다.

**셋째, 선행 연구의 기준선이 절반의 모델에서 저자의 single-LoRA를 이긴다.** closed-book NarrativeQA에서 KMSDCD(Caccia et al. 2025)는 Llama-3.1-8B 29.07 대 27.05, Qwen3-8B 28.68 대 25.78로 앞서고, 네 모델 전부에서 모든 multi-LoRA 구성을 이긴다. **논문은 이 행에 논평하지 않는다** [LoRA-KM Table 2].

**넷째, 100K 토큰 이상 구간의 증거가 통계적으로 비어 있다.** LongBench v2는 문서 **30개**로 평가되고 보고된 정확도가 전부 3.33의 배수다(30.00, 20.00, 33.33, 16.67, 23.33, 36.67, 26.67). 무작위 추측 바닥이 약 25%인 객관식 벤치에서 ICL이 두 모델 모두 20.00, RAG가 20.00/16.67로 **바닥 아래**다. ∞Bench는 문서 **20개**를 쓴다 [LoRA-KM App. Q, Table 7]. "LoRA 기반 방법이 100K+ 토큰 레짐에서 일률적으로 붕괴하지는 않는다"는 결론은 이 표본 크기로는 지지되지 않는다.

### 19.6.4 $\Theta$층 용량의 실측 공급원 — Memory Layers

`Memory Layers at Scale`(arXiv:2412.09764, 이하 [MemLayers])은 이 장의 도구가 아니다. **배포 후 어떤 층도 갱신되지 않으므로 sleep-time compute의 사례가 아니며**(→ ch11 §11.4.4), 이 책이 인용하는 지위는 하나다 — $\Theta$층의 용량이 FLOPs와 분리 가능하다는 **실측 공급원**이다. 용량 논의의 정본은 ch09가 소유한다(→ ch09 §09.5). 이 장이 흡수해야 할 것은 그 실측에 붙는 유보다.

**채택된 변형이 자기 ablation 표에서 최고가 아니다.** [MemLayers Table 3]에서 `+random values`가 (2.06, 11.32, 7.20)로 세 지표 전부 최고인데 채택된 `+swilu`는 (2.07, 11.79, 7.64)이고, `+gated`(2.08, 11.60, 7.54)조차 뒤의 두 지표에서 `+swilu`를 앞선다. 저자는 학습 속도 비용과 "**더 큰 모델 크기에서는 이득이 일관되지 않았다**"를 이유로 배제하지만 [MemLayers §5.4], **그 대형 모델 수치를 논문에 싣지 않는다.** 배제 근거가 보고되지 않은 실험에 있다.

**vanilla Memory는 PEER에 진다.** 1.3b에서 Memory의 NQ 9.83이 PEER 12.33에, TQA 39.47이 42.46에 뒤진다 [MemLayers Table 1]. [MemLayers §5.1]의 "PEER 아키텍처는 Memory와 유사한 성능을 보인다"는 NQ에서 25%가 넘는 상대 격차(9.83 대 12.33, 상대화는 이 책의 나눗셈이다)를 완곡화한 서술이다. 즉 이 논문의 우위는 memory layer 자체가 아니라 **Memory+의 추가 장치**에서 온다.

**MOE 기준선은 학습과 평가의 설정이 다르다.** "MOE 모델은 expert choice로 학습하고 top-1 routing으로 평가한다" [MemLayers §4.1]. 이 불일치가 MOE 성능을 얼마나 떨어뜨리는지 논문이 측정하지 않으므로, MOE가 "큰 차이로" 뒤진다는 결론이 이 선택에 얼마나 의존하는지 알 수 없다.

**8B에서 Memory+는 Llama3.1 8B에 9개 중 7개가 뒤진다.** 앞서는 것은 HellaSwag(60.29 대 60.05)와 PIQA(79.82 대 79.16) 둘뿐이고, HotpotQA(26.06 대 27.85)·HumanEval(31.71 대 37.81)·MBPP(42.20 대 48.20)·MMLU(63.04 대 66.00)·NQ(27.06 대 29.45)·OBQA(34.40 대 34.60)·TQA(68.15 대 70.36)에서 뒤진다 [MemLayers Table 2]. 비교 대상이 15배의 토큰을 받았다는 점을 논문이 밝히지만, [MemLayers §5.3]의 "Llama3.1 8B의 성능에 근접한다"는 표현은 방향을 흐린다.

---

## 19.7 Systems/serving 함의

이 장이 소유하는 세 번째 정의를 여기서 못 박는다.

> **정의.** **$\Theta$-경로 쓰기 도구의 서빙 함의**는 세 항으로 갈린다. (a) wake-time 연산 그래프가 바뀌는가, (b) 배포 아티팩트의 **단위**가 무엇인가, (c) shared-weight batching이 살아남는가. 세 항은 독립이 아니며, **(a)를 0으로 만드는 방식이 (b)와 (c)를 정한다.**

(a)부터 답한다. **세 도구 모두 wake-time 연산 그래프를 바꾸지 않는다.** [LoRA]는 merge 후 GEMM shape가 base와 글자 그대로 같으므로 이것을 초록 수준의 셀링 포인트로 삼는다 — "adapter와 달리 추가 추론 지연이 없다" [LoRA Abstract].

> **[해설]** 직접 편집도 같은 자리에 도달하지만 논문이 그 사실을 **주장조차 하지 않는다**[본서 추론]. 갱신이 $\Theta^{(l)}_{\mathrm{out}} \mathrel{+}= \Delta^{(l)}$ 라는 같은 shape의 제자리 덧셈이므로 [MEMIT Algorithm 1] 편집 후 모델의 파라미터 수·연산 그래프·토큰당 메모리 트래픽이 편집 전과 동일하다. 그러나 두 논문 어느 쪽도 추론 지연을 측정하지 않았고 [ROME]의 유일한 시간 수치는 편집 비용이므로, 표 19-3의 $L_w$ 칸은 "논문에 없음"으로 남는다. **구조가 뻔한 것과 논문이 잰 것은 다르다.**

(b)와 (c)에서 두 방식이 갈린다. Rosetta 사전으로 옮기면 세 도구 전부 "모델 재배포 = (U-$\Theta$) 1회" 행에 놓이지만, 재배포되는 **단위**가 다르다. delta 방식의 단위는 base 350GB와 별도의 35MB 파일이다 — base는 HBM에 상주하고 갈아끼우는 것은 35MB뿐이며, 되돌리기는 뺄셈 한 번이다(→ ch03 §03.5). **직접 편집의 단위는 체크포인트 전체다.** 편집이 원 가중치에 흡수되므로 별도 저장 아티팩트가 없고, 그래서 "checkpoint 크기 = $C_{\mathrm{cap}}$"이라는 Rosetta 행이 이 방식에서는 무의미해진다 — 편집을 100개 하든 10,000개 하든 체크포인트 크기가 같다[본서 추론].

> **[평가]** 이 "무료"가 함정이다. 상태량이 0으로 보이는 대가로 세 가지가 사라진다. 첫째, **되돌릴 수 없다** — [ROME]과 [MEMIT] 어디에도 편집 취소 연산이 없고, 롤백은 이전 체크포인트로 교체하는 것뿐이다. 둘째, 편집 집합이 다른 두 배포본은 완전히 별개의 체크포인트가 된다. 셋째, **사용자별 개인화가 구조적으로 불가능하다.** 직접 편집은 공유 가중치를 제자리에서 고치므로 애초에 분리 가능한 아티팩트가 생기지 않고, 사용자마다 다른 사실을 넣으려면 사용자마다 모델 전체 사본이 필요하다. 이 책의 실험 [X2]가 네 서빙 구조를 비교해 이 갈림길을 정량화했으며, 그 판정은 ch26과 ch29가 내린다. 이 장은 갈림길이 어디에 있는지만 지목한다.

(c)를 마저 적는다. 편집이 **전역적**이면 — 10,000개 사실이 모든 사용자에게 같은 사실이면 — 가중치 1벌을 계속 공유할 수 있으므로 shared-weight batching이 살아남는다. 사용자별 편집을 시작하는 순간 per-user $\Theta$가 생기고 대전제가 깨진다. **논문은 이 구분을 제기조차 하지 않는다**[본서 판단]. 상각 논증도 같은 조건에 걸린다 — §19.5의 "한 번 지불하고 무한히 상각"은 편집이 전역일 때만 성립한다.

운영 쪽 항 셋을 더 적는다. 어느 것도 논문에 없고 전부 이 책의 추론이다.

**KV·prefix cache 무효화.** [MEMIT]은 GPT-J의 층 3–8 MLP를 고치므로 그 층 이후의 캐시된 KV가 전부 stale해진다. 논문은 KV cache를 한 번도 언급하지 않는다(검색어 'KV' 0건). prefix cache를 대량으로 운용하는 서빙에서 $\Theta$ 갱신은 캐시 전면 무효화 이벤트이며, 이것이 $\Theta$-경로 배포의 숨은 운영 비용이다.

**sleep 라운드의 자원 격리.** [MEMIT]은 GPT-NeoX 편집에 "float16으로 모델을 돌릴 48GB GPU 하나와, 편집 방법을 실행할 조금 작은 GPU 하나"가 필요하다고 쓴다 [MEMIT §9]. 그리고 covariance 통계는 fp32로 수집·저장하는데 모델은 float16으로 돈다 [MEMIT App. B.3, B.4]. **sleep 라운드는 서빙과 같은 커널·같은 dtype으로 돌지 않으며, 편집기의 peak memory 발자국이 모델과 같은 급이다.** 유휴 서빙 GPU에 sleep을 얹는 그림은 이 수치와 맞지 않는다.

**갱신 주기는 일 단위다.** 10,000 사실에 7.44시간(GPT-J 6B, A6000)이고, 논문의 뉴스 신선도 데모도 11월 8일 선거 결과를 11월 14일 이전에 확보하는 며칠 단위 서술이다 [MEMIT §5.2.2, App. E]. 하루 1회 배치가 현실적 상한이며 실시간 갱신은 이 방법의 범위 밖이다.

마지막으로 $E$-경로와의 비용 위치를 대조한다. 편집 이후의 질의는 검색 대역폭을 0 지불한다 — 지식이 이미 가중치 안에 있다. $E$-경로는 질의마다 $\mathrm{ret}(E,q)$의 대역폭과 지연을 지불한다. 반대로 [MEMIT]은 지식 하나를 바꾸는 데 편집당 2.6773초의 오프라인 비용을 지불하고, $E$-경로는 레코드 하나를 쓰는 데 거의 0을 지불한다. **쓰기가 비싸고 읽기가 공짜인 층 대 쓰기가 싸고 읽기가 유료인 층** — 이 대비가 Part III의 3경로 비용 분해의 뼈대다.

---

## 19.8 한계와 bridge-out

### 19.8.1 논문 자신이 남긴 문제

[ROME]이 남긴 것 중 이 책에 가장 무거운 항목은 쓰기 검증이다 — "모델에 저장된 사실 연관이 성공적으로 바뀐 뒤에도, 모델은 증거에 근거가 없고 거짓일 가능성이 높은 그럴듯한 새 사실을 추측한다. 이것은 언어모델을 사실의 출처로 쓰는 유용성을 제한할 수 있다" [ROME §3.7]. 두 번째는 $C_{\mathrm{cap}}$ 회계에 직접 들어온다 — "ROME이 편집하는 연관은 방향성을 가진다. 예컨대 '시애틀의 상징적 랜드마크는 스페이스 니들이다'는 '스페이스 니들은 시애틀의 상징적 랜드마크다'와 따로 저장되므로 둘 다 바꾸려면 두 번 편집해야 한다" [ROME §3.7]. **per-fact 상태량이 사실 수가 아니라 연관 방향 수로 스케일한다[본서 추론].** 나머지는 다중 동시 편집(후속 논문으로 넘김), 사실 이외의 학습된 믿음, 학습된 속성을 표현하는 벡터 공간의 구조다. 그리고 "중간층 MLP 어디에나 같은 사실을 동등하게 저장할 수 있다고 **추측한다**"는 문장이 실험이 아니라 conjecture로 남는다 [ROME §2.3] — $\Theta$-경로 쓰기 자유도의 전제가 미검증 상태로 후속 논의에 상속된다.

[MEMIT]이 물음표로 남긴 문장은 하나다 — "해석가능성 기반 방법이 전통적인 불투명 fine-tuning의 흔한 대안이 될 수 있을까" [MEMIT §6]. 그 밖에 대칭 관계를 포함한 지식 범위, 어떤 관계는 왜 robust한 specificity로 편집하기 어려운가, 그리고 $z_i$ 산출이 "embarrassingly parallel이므로 배치화할 수 있다"는 주장만 있고 실행되지 않은 것이 남는다 [MEMIT §5.2.2].

[LoRA]가 남긴 것 중 이 장에 직접 걸리는 항목은 셋이다. 어느 행렬에 걸지를 heuristic으로 정한다는 자인("더 원리적인 방법이 있는가" [LoRA §8]), "모델 크기와 적응에 최적인 rank 사이의 관계는 여전히 열린 질문이다" [LoRA App. H.2], 그리고 merge와 배칭이 양립하지 않는다는 한계다 [LoRA §4.2].

### 19.8.2 이 책의 판정

> **[평가]** 이 장은 $\Theta$-경로의 primitive를 세우지만, 그 primitive를 제공한 두 논문 중 하나는 스스로 실용 방법이 아니라고 선언했고([ROME §3.7], → §19.2) 다른 하나는 기억 문제가 아니라 배포 경제학을 풀었다([LoRA §1]). **재구성은 이 책의 것이고, 재구성이 성립하는 근거는 후행 문헌이 실제로 이 도구를 그렇게 썼다는 사실 하나다.** 그러므로 이 장에서 확립된 것은 "$\Theta$에 쓸 수 있다"이지 "$\Theta$에 써야 한다"가 아니다.

> **[평가]** 두 방식의 갈림길은 rank도 정확도도 아니라 **아티팩트**다. 같은 $L_w=0$을 두 방식이 정반대 대가로 산다 — delta 방식은 아티팩트를 남겨 되돌리기·개인화·다중 배포를 얻는 대신 요청마다 다른 가중치를 감수하고, 직접 편집은 아티팩트를 없애 shared-weight batching을 지키는 대신 되돌리기와 개인화를 통째로 포기한다. **이 갈림길은 알고리즘의 세부가 아니라 서빙 구조의 분기점이며**, 정량 판정은 ch26·ch29가 [X2] 위에서 내린다.

> **[평가]** 그리고 이 장의 도구는 절반만 완성되어 있다. 세 논문 전부에서 $\mathrm{gen}(\cdot;B_s)$가 항등이므로(→ §19.4) **무엇을 쓸지를 만드는 단계가 없다.** 쓰기 검증도 마찬가지다 — [ROME]이 그 필요를 명시했지만 어떤 비용 수치에도 그 항이 없다. $\Theta$-경로가 sleep-time compute이 되려면 두 빈칸이 채워져야 하고, 그중 첫 번째를 다음 장이 채운다.

### 19.8.3 경로 간 인용 여부

세 논문 모두 다른 두 경로에 완전히 침묵한다. 먼저 연대를 갈라 둔다. $E$-경로(MemGPT 2023-10, Letta STC 2025-04)와 $W$-경로의 sleep 논문들은 **전부 이 장의 세 논문보다 뒤에 나왔으므로**, 그 부재는 연대가 강제한 것이고 흠이 아니다. 진짜 관측은 **동시대의 침묵**이다.

[LoRA]는 참고문헌 51항목 안에 retrieval·외부 기억 문헌 0건, fast weights·test-time training·linear attention 0건, 지속학습·catastrophic forgetting·replay·EWC·CLS 0건이다. [ROME]은 전문 검색에서 sleep 0건, consolidat 0건, continual 0건, forget/catastroph 0건, replay 0건, fast weight 0건, external memory 0건, adapter 0건이다. [MEMIT]도 sleep·consolidat·replay·catastrophic·continual·forgetting이 전부 0건이며('forgetfulness'가 FT-W 기준선 설명에 1회), LoRA 0건·adapter 0건·low-rank 0건이다.

> **[평가]** 두 방향의 침묵이 이 장의 발견이다. 첫째, **같은 $\Theta$층 안에서 두 공동체가 서로를 모른다.** [LoRA](2021-06)는 [ROME](2022-02)보다 8개월 앞서 나왔고 둘 다 같은 종류의 저랭크 가중치 쓰기인데, [ROME]은 [LoRA]를 인용하지 않고 자기 갱신을 low-rank adaptation 문헌이 아니라 associative memory 문헌(Kohonen 1972, Anderson 1972, Bau et al. 2020)에 연결한다. [MEMIT]도 PEFT 계보를 한 번도 언급하지 않으며, fine-tuning은 오직 naive 기준선으로만 등장한다. **두 논문은 서로를 모른 채 같은 저랭크 쓰기 공간을 반대 방향에서 채운다** — [LoRA]는 rank를 설계 변수로 고정하고, [MEMIT]의 rank는 편집 개수의 부산물로 결정된다[본서 관찰]. 둘째, **$\Theta$-경로의 쓰기 기법이 지속학습 공동체의 어휘를 하나도 쓰지 않고 만들어졌다.** [MEMIT]의 $\Gamma_0$ 보존 항은 기능적으로 EWC의 Fisher 정칙화와 같은 자리에 있지만(옛 지식을 2차 통계로 보호한다) EWC를 인용하지 않는다.

연결점이 하나 있다. [MEMIT]은 Kohonen(1972)과 Anderson(1972)을 [MEMIT §2]와 Eq. 7에서 명시적으로 인용하며 "선형 층을 associative memory로 보는 고전적 관점을 채택한다"고 선언한다. 이 두 문헌은 Neural Memory 모노그래프가 정리한 fast-weight·linear-attention 계보의 공통 조상이기도 하다(→ NM ch03). **$W$-경로와 $\Theta$-경로는 서로를 인용하지 않지만 같은 1972년 조상을 공유한다** — 이 장에서 쓸 수 있는 유일한 실제 연결점이다.

마지막으로 [MEMIT]이 실제로 인용하는 "외부 기억"의 정체를 밝혀 둔다. [MEMIT §2] 첫 문단이 통째로 상징 knowledge base 계보(Cyc, WordNet, DBpedia, Freebase, YAGO, ConceptNet, NELL, Wikidata)이며, retrieval·RAG·vector database 언급은 0건이다. **같은 external memory라는 말을 쓰지만 대상이 세계 지식과 사용자 문맥으로 갈린다** — 이 corpus의 $E$-경로와는 문제 자체가 다르다. Part III가 이 구분을 유지한다.

### 19.8.4 다음 장이 받아가는 것

ch20은 SEAL을 읽는다. 이 장이 넘기는 것은 정확히 이 장이 비었다고 판정한 칸이다 — **도구는 있는데 무엇을 쓸 것인가.** [MEMIT]이 "무엇을 쓸지 이미 안다"는 전제 위에서 "어떻게 많이 쓸지"만 푼 자리에서, 다음 장의 논문은 $\mathcal{R}_k$를 모델이 스스로 만들게 한다. 그 순간 $\mathrm{gen}(\cdot;B_s)$가 항등이 아니게 되고 $B_s$가 처음으로 조절 가능한 knob이 되며, ch11의 판별식 조건 (2)가 비로소 만족된다. 이 장이 잰 편집 용량의 실측 상한과 쓰기 부작용은 그때 그대로 제약으로 따라간다.

## 요약

- $\Theta$에 쓰는 방식은 산출물의 형태로 갈린다. **delta 방식은 분리 가능한 아티팩트를 남기고, 직접 편집은 base에 흡수시켜 아무것도 남기지 않는다.** 갈림길은 저랭크냐 아니냐가 아니다.
- [ROME]의 갱신 (19-3)에는 **$\eta_\Theta$가 존재하지 않는다.** 등식제약 최소제곱의 닫힌 해이며, $\Theta$-경로의 쓰기가 반드시 SGD여야 하는 것은 아니라는 존재증명이다.
- [ROME]은 스스로 "지식 저장 기제를 이해하기 위한 도구이며 대규모 모델 학습을 위한 실용적 방법으로 의도되지 않았다"고 쓴다 [ROME §3.7]. 이 논문을 $\Theta$-경로 primitive로 세우는 것은 **이 책의 재구성이며 저자의 주장 범위를 넘는다.**
- 편집 용량의 실증치는 GPT-J 6B에 10,000개 (s, $\varrho$, o) 연관이고, 그 지점에서 specificity가 83.5에서 73.7로 떨어진다 [MEMIT Table 2]. **10,000은 붕괴점이 아니라 관측된 하한이다.**
- 같은 방법의 zsRE 종합점수는 50.7이며 Specificity 26.6에 지배되는데, **그 값은 편집하지 않은 GPT-J의 27.0보다 낮다** [MEMIT Table 1].
- 10,000 편집의 sleep 비용은 7.44시간(GPT-J 6B, A6000)이고 그중 87.95%가 $z_i$ 산출이다 [MEMIT §5.2.2, App. B.4]. **편집당 2.6773초는 이 책의 나눗셈이고 논문에 없다.**
- 세 논문 모두 $\mathrm{gen}(\cdot;B_s)$가 항등이고 sleep 라운드가 $k=1$에서 멈춘다. 그러므로 $B_s$의 knob도 $\rho$를 잴 실험도 이 장 안에 없다.
- [MemLayers]는 배포 후 갱신이 없으므로 sleep-time compute의 사례가 아니라 **$\Theta$층 용량의 실측 공급원**이며(→ ch11 §11.4.4), 채택된 `+swilu`가 자기 ablation에서 최고가 아니고 8B에서 Llama3.1 8B에 9개 중 7개가 뒤진다 [MemLayers Table 2, Table 3].

## 자가 점검 체크리스트

- [ ] delta 방식과 직접 편집을 산출물의 형태로 정의하고, 두 방식이 같은 $L_w=0$을 서로 다른 대가로 산다는 것을 (b)·(c) 항으로 설명할 수 있다.
- [ ] (19-3)과 (19-6)을 (U-$\Theta$) 표준형과 대조해 **학습률이 없는 쓰기**의 구조를 진술하고, $\lambda$가 $\alpha_t$의 어느 역할을 대신하는지 말할 수 있다.
- [ ] 편집 용량의 실측 상한을 정의하고, 왜 10,000이 상한이 아니라 하한인지, 그리고 그 값이 어느 지표를 얼마나 희생한 값인지 수치로 말할 수 있다.
- [ ] 표 19-3의 12칸 중 "논문에 없음"인 칸을 짚고, $C_{\mathrm{cap}}$이 [ROME]에서 계산 불가능한 정확한 이유($H$·$D$ 미명시)를 댈 수 있다.
- [ ] 이 장의 세 논문이 어느 경로를 왜 인용하지 않는지 설명하면서, **연대가 강제한 침묵과 동시대의 선택된 침묵**을 구분할 수 있다.
- [ ] Rosetta 사전의 "모델 재배포 = (U-$\Theta$) 1회", "checkpoint 크기 = $C_{\mathrm{cap}}$", "batch로 weight 공유 → $\Theta$-경로에서 깨짐" 세 행을 delta 방식과 직접 편집에 각각 적용했을 때 어느 행이 어느 방식에서 무의미해지는지 말할 수 있다.
