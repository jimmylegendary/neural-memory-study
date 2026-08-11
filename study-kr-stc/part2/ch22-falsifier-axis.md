# ch22. 반증 축 — 가중치에 사실을 계속 넣을 수 있는가

## 22.1 Bridge-in: 전작이 남긴 문제

ch19는 $\Theta$에 쓰는 도구를 주었고(LoRA와 직접 편집), ch20은 무엇을 쓸지를 모델이 스스로 정하게 했으며(SEAL의 self-edit), ch21은 $W$에서 $\Theta$로 옮기는 이송 연산자를 세웠다. 세 장이 답한 질문은 전부 같은 형태다 — **한 번의 쓰기를 어떻게 하는가.** 이 장은 방향을 뒤집는다. 같은 루프를 스무 번, 백 번 돌린 뒤에 무엇이 남는가.

이 질문을 실제로 돌린 논문이 corpus에 한 편 있다. [Continual Facts] (*Can a Language Model Learn Facts Continually in Its Weights?*, arXiv:2607.11020)는 발명된 사실 하나씩을 rank-$r$ adapter로 학습해 $\Theta$에 merge하는 절차를 20회, 이어서 100회 반복하고, **같은 내용을 프롬프트에 넣은 대조군**을 나란히 세운 뒤 잔여물을 잰다. $\Theta$-경로의 다른 논문은 전부 통합 라운드 1회 또는 짧은 수열만 보인다.

그런데 이 장의 bridge-in은 ch21처럼 닫혀 있지 않다. **이 논문은 sleep-time compute 문헌의 어떤 open question도 상속하지 않는다.** 이 라인의 논문을 한 편도 인용하지 않고, "sleep-time compute"이라는 말도 쓰지 않는다(§22.8.5). 대신 상속을 두 곳에서 명시적으로 적는다.

> "This extends the finding of Shenfeld et al. (2025), that KL from the base policy predicts forgetting, to knowledge writing." [Continual Facts §6.1, p.14]

> "These nulls accord with the warning of Hase et al. (2023) that localisation need not identify the best site to edit." [Continual Facts §7.3, p.18]

그리고 공백은 스스로 만든다 — "These observations lack a common account of why some writes produce usable knowledge, why some survive, and what a forgotten fact leaves behind" [Continual Facts §1, p.1]. 여기서 "these observations"가 가리키는 다섯 편(Gekhman 외 2024, Berglund 외 2024, Lampinen 외 2025, Allen-Zhu & Li 2024, Shenfeld 외 2025) 중 어느 것도 이 책의 세 경로에 속하지 않는다.

연대는 먼저 갈라 둔다. 이 논문의 v2는 2026년 7월 14일 자이고 ch21이 다룬 논문의 v2는 7월 10일 자다. 따라서 ch21이 이 논문에 침묵한 것은 **연대가 강제한 것**이다. 반대 방향은 아니다 — SEAL(2506.10943), Nested Learning(2512.24695), `Language Models Need Sleep`(2606.03979), SCM(2604.20943)은 모두 선행하며 이 논문의 참고문헌에 없다(§22.8.5).

> **[해설]** 이 배치가 반증 축의 성격을 정한다. **가장 강한 반증 후보가 반증 대상을 읽지 않고 쓰였다.** 이것은 약점이면서 동시에 강점이다 — 약점은 이 논문의 판정이 $\Theta$-경로 논문들이 실제로 제안한 절차를 대상으로 삼지 않았다는 것이고, 강점은 그 판정이 이 라인의 주장에 맞춰 설계되지 않았다는 것이다. 수렴이 독립적일 때 증거는 더 강해진다.

## 22.2 문제의식

논문은 지속 쓰기를 세 요구로 분해하고 각각에 판정을 내린다.

> "Each write must create usable knowledge, general abilities must survive accumulating updates, and earlier facts must remain reachable. Data breadth makes the first tractable, while a frozen teacher or explicit penalty makes the second tractable. The third remains unresolved." [Continual Facts §9.3, p.20]

이 세 요구가 이 장이 소유하는 첫 정의를 낳는다.

> **정의.** **$\Theta$-경로의 반증 조건**은 다음 셋 중 하나라도 성립하면 (U-$\Theta$)를 반복하는 설계가 목적을 달성하지 못한다는 진술이다. (i) **생성 실패** — 한 번의 (U-$\Theta$)가 회수는 되지만 사용은 되지 않는 객체를 만든다. (ii) **역량 실패** — 라운드가 누적될수록 $\Theta$가 원래 갖고 있던 일반 능력이 무너진다. (iii) **도달 실패** — 앞서 쓴 내용이 $\Theta$ 안에 남아 있는데도 질의로 닿지 않는다. 세 조건은 독립이며, 하나를 고치는 처방이 다른 하나를 고치지 않는다.

논문이 겨냥하는 것은 알고리즘이 아니라 쓰기가 만드는 **객체**다 — "the object that a write creates: what kind of knowledge it contains, whether later writes preserve it, and what remains after questions about it fail. This turns catastrophic forgetting into a property of the written object, measured against the same content placed in context" [Continual Facts §1, p.1].

판정 문장은 두 개다. "When facts must be composed or survive later writes, the reliable channel is context rather than the weights" [Abstract, p.1], 그리고 "weights are therefore the wrong system of record: they store content without creating an address for it" [§9.3, p.20].

> **[평가]** 두 번째 문장의 "weights"는 이 책의 어휘로 $\Theta$만을 가리킨다. 논문에는 $W$가 존재하지 않고(§22.4) $W$-경로 문헌도 한 편 없다(§22.8.5). 그러므로 이 판정을 fast weights로 옮기는 것은 논문이 뒷받침하지 않는 확장이다. 본서는 이 문장을 인용할 때마다 층을 명시한다.

## 22.3 Core mechanism (통일 표기)

기제는 순수 (U-$\Theta$) 루프이며, 표준형과 세 곳에서 갈라진다. 라운드 $k$의 학습 집합은 한 문장짜리 발명 사실 $s_k$에서 생성되고, 갱신은 rank-$r$ 부분공간 $\mathcal{S}_r$ 안에서 수렴까지 최적화된 뒤 $\Theta$에 **merge**된다.

$$
\Theta_{k+1} \;=\; \Theta_k + \Delta\Theta_k,
\qquad
\Delta\Theta_k \;=\; B_kA_k \;=\; \arg\min_{\Delta\Theta\in\mathcal{S}_r}\ \mathcal{L}\big(\mathcal{R}_k;\ \Theta_k+\Delta\Theta\big)
\tag{22-1}
$$

merge가 루프를 지속적으로 만든다 — "Sequential conditions merge each fact's adapter into the running model before the next fact trains" [Continual Facts App. A.3, p.25]. $\mathcal{S}_r$는 $r\in\{4,16\}$, $\alpha_{\mathrm{LoRA}}=32$, dropout 0으로 attention q/k/v/o와 MLP gate/up/down 일곱 투영 행렬 전부에 걸린다 [App. A.3, p.25]. full fine-tuning 조건에서는 $\mathcal{S}_r$가 전 파라미터 공간이다 [§3.4, p.6].

이 논문의 실제 연구 대상은 optimizer가 아니라 $\mathrm{gen}(\cdot;B_s)$다.

$$
\mathcal{R}_k \;=\; \mathrm{gen}\big(s_k;\ B_s\big),
\qquad
|\mathcal{R}_k| \;=\;
\begin{cases}
2 & \text{bare-statement}\\
24 & \text{study}
\end{cases}
\tag{22-2}
$$

bare-statement는 사실 문장 자체를 두 가지 사소한 틀에 넣은 것이고, study는 사실 하나당 24개 항목 — paraphrase, 질문-답 쌍, 함의를 전개한 예, 위반된 기본값과의 대조 — 이다 [§3.1, p.4; App. A.1, p.25]. **corpus의 다른 $\Theta$-경로 논문에서 $\mathcal{R}_k$는 방법이 만들어 내는 부산물이고, 여기서는 독립변수다.**

대조군은 (U-E)의 퇴화한 최선 사례다.

$$
\hat y \;=\; f\big(q;\ \Theta_0,\ W=\varnothing,\ \mathrm{ret}(E,q)=s_k\big),
\qquad
\hat c_k = c_k,\quad B_s = 0
\tag{22-3}
$$

즉 $S(\cdot;B_s)$가 항등사상이고 검색이 완벽하다. **이 대조군은 $E$-경로의 구현이 아니라 상한이다[본서 추론].** Letta식 요약도, 검색 오차도 모델링되지 않는다 [§2, p.3; Fig. 1, p.2].

계기는 두 개다. 첫째, 쓰기가 만든 객체의 **종류**를 재는 entailment gap.

$$
\mathrm{gap} \;=\; \mathrm{acc}_{\mathrm{lenient}} - \mathrm{acc}_{\mathrm{strict}}
\tag{22-4}
$$

strict는 요구된 결론만 인정하고 lenient는 그 결론을 함의하는 사실도 인정한다. 두 정책은 중첩이 아니라 독립 판정이라 작은 음수 gap이 나올 수 있다 [Eq. 2.1, §2.2, p.4]. 둘째, 내용이 **남아 있는지**를 재는 저장 탐침.

$$
R_{\log p}(k) \;=\; \frac{\log p_{\Theta_{j+k}}(s) - \log p_{\Theta_0}(s)}{\log p_{\Theta_j}(s) - \log p_{\Theta_0}(s)}
\tag{22-5}
$$

$R_{\log p}=1$이면 완전 보존, $0$이면 쓰기 전 사전분포로의 소거다 [Eq. 5.1, §5.1, p.11]. 세 번째로 역량 손상의 좌표는 고정 프롬프트 풀 $P$ 위의 토큰당 KL이다 — $\mathrm{drift}(\Theta)=\mathbb{E}_{x\sim P}[\frac{1}{|x|}\sum_t \mathrm{KL}(\pi_\Theta \,\|\, \pi_{\Theta_0})]$ [Eq. 6.1, §6, p.13].

표준형 (U-$\Theta$)와의 차이는 다섯이다. (1) 한 라운드가 gradient step 1회가 아니라 **수렴까지의 완전한 최적화**(batch size 1에서 24·96·192 step)이고 그 뒤 merge가 붙는다. (2) 갱신이 rank로 제약되는데 이 제약이 원인은 아니다(§22.6.1). (3) $\mathcal{R}_k$의 폭이 조작된다. (4) 근접항 $\lambda_{\mathrm{KL}}\,\mathrm{KL}(\pi_{\Theta}\|\pi_{\Theta_k})$의 **anchor가 $\Theta_0$가 아니라 현재의 $\Theta_k$**여서 $\Theta_0$로부터 잰 KL은 내려가지 않는다 [§6.3, p.15]. (5) 20라운드마다 $\Theta$를 $\Theta_0$로 **되돌리고** 지금까지의 모든 사실을 batch distillation으로 다시 쓰는 둘째 시간척도가 있다 [§4.3, p.10]. [NL]/[Sleep]의 어휘로는 $f_1=1$, $f_2=1/20$의 2-level 시스템이되 위 층이 아래 층 위에 쌓이지 않고 리셋한다. 본서는 이것을 **frozen-anchor consolidation**이라 부르고, ch11·ch21이 소유한 consolidation(느린 층으로의 상향 증류)과 매번 구분해 적는다.

### 22.3.1 표기 대응표

표 22-1 [Continual Facts]의 원 표기와 본서 표기

| 원 논문 표기 | 본서 표기 | 이유 |
|---|---|---|
| $\theta,\ \theta_0,\ \theta_j$ | $\Theta,\ \Theta_0,\ \Theta_j$ | 전부 slow weights다. 소문자 $\theta^{(\ell)}$은 CMS level 파라미터로 예약 |
| $\Delta\theta$ | $\Delta\Theta$ | 동일. $W$의 이동과 구분 |
| $\lambda$ [§6.3, Fig. 14] | $\lambda_{\mathrm{KL}}$ | 맨 $\lambda$는 outer weight decay, $\lambda_{\mathrm{KD}}$는 ch21이 씀 |
| $\alpha = 32$ [App. A.3] | $\alpha_{\mathrm{LoRA}}$ | $\alpha_t\in[0,1]$은 retention gate. 정면 충돌 |
| $\rho$ (Spearman 상관) | $\rho_S$ | **$\rho$는 이 장의 주제인 망각률이다.** 바꾸지 않으면 비용 표가 읽히지 않는다 |
| $R(k)$ [Eq. 5.1] | $R_{\log p}(k)$ | $\mathcal{R}_k$(라운드 학습 집합)·$R$(보상)과의 근접 오독 방지 |
| $D_s$ [Eq. 3.1] | $\mathcal{R}_k$ | 문자 그대로 (U-$\Theta$)의 학습 집합이다 |
| $s$ (사실 문장) | $s_k$ (항상 첨자) | $S_t$는 momentum, $s$는 SGD step 첨자로 예약 |
| $k$ (수행한 쓰기 수) | $k$ (유지) | sleep 라운드 첨자와 일치. 단 이 장에서 $k$를 key로 쓰지 않는다 |
| $\pi_T,\ \pi_\theta,\ \pi_0$ | $\pi_T,\ \pi_\Theta,\ \pi_{\Theta_0}$ | 대문자 $\Pi$는 ch07이 도입한 지속학습 성능 행렬로 예약되어 있다. 정책은 어떤 경우에도 대문자화하지 않는다 |
| $L_{\mathrm{SFT}},\ L_{\mathrm{offline}}$ | $\mathcal{L}_{\mathrm{SFT}},\ \mathcal{L}_{\mathrm{offline}}$ | outer loss는 calligraphic. 맨 $L$은 sequence 길이 |
| $g_{\mathrm{use}}$ [Eq. 7.1] | $g_{\mathrm{use}}$ (outer 전용 선언 후) | 이 논문에는 inner loop가 없다 |
| "consolidation" [§4.3] | "frozen-anchor consolidation" | 기호가 아니라 용어 충돌. 리셋 후 재작성이지 상향 증류가 아니다 |

## 22.4 어느 층을 언제 쓰는가

표 22-2 층별 갱신 배정 — 무엇이 언제 움직이는가

| 대상 | 누가 학습하는가 | 언제 움직이는가 | 규모·설정 | 출처 |
|---|---|---|---|---|
| $\Theta_0$ (Qwen3-4B, bf16, thinking 비활성) | 이 논문 밖의 pre-training | LoRA 조건에서는 never(base frozen), full-FT 조건에서는 쓰기 중 | 4B. 복제 1회만 8B | [§2, p.3; App. A.3, p.25] |
| adapter $A_k,B_k$ | **sleep 라운드의 쓰기 그 자체** | 사실 $k$의 오프라인 쓰기 중. 이후 merge되고 별도 객체로는 폐기 | $r=16$(또는 4), $\alpha_{\mathrm{LoRA}}=32$, 일곱 투영 행렬 | [App. A.3, p.25] |
| AdamW optimizer state | 쓰기가 만들고 쓰기가 버린다 | 쓰기 중에만. wake로 넘어가지 않는다 | lr $2\times10^{-4}$, 답 토큰에만 loss, batch size 1 | [App. A.3, p.25] |
| teacher $\pi_T$ | 아무도 학습하지 않는다 | frozen-teacher 조건에서 never. own-merges 조건에서는 현재 merge된 모델이라 움직인다 | adapter를 끄고 사실을 문맥에 넣은 forward pass | [§6.2, p.14] |
| $\mathcal{R}_k$ (24항목 study set) | 별도 모델이 한 번 생성 | 전 실행 전에 고정. 생성기는 평가 문항을 보지 않는다 | 사실당 24항목 / bare는 2 | [App. A.3, p.26] |
| $W$ (fast weights) | **아무도 학습하지 않는다. $W=\varnothing$** | never. 토큰당·청크당 움직이는 상태가 없다 | 해당 없음 | 논문 전체(부재로 확인) |
| $E$ (문맥 채널) | 아무도 학습하지 않는다 | **wake 시각에, 매 질문마다 프롬프트 토큰으로** | 사실당 한 문장. 동시 투입 최대 2개 | [Fig. 1, p.2; §5.2, p.12] |

읽는 법은 한 줄이다. **이 논문에는 wake-time 이동이 전혀 없다.** 서빙은 평범한 frozen forward pass이고 흥미로운 일은 전부 요청과 요청 사이에서 일어난다. 판별식(→ ch11)의 네 조건은 그대로 충족된다 — 질의 전에, $B_s>0$을 써서, $\Theta$를 실제로 바꾸고, 그 변화가 이후 질의에 쓰인다. 다만 층 배정이 비대칭이다. **(U-$\Theta$)만 처치군이고, (U-E)는 통제군으로 병렬 실행되며, (U-W)는 아예 등장하지 않는다.** 두 층이 순서로 결합하는 자리는 딱 하나, offline context distillation이다.

$$
\mathcal{L}_{\mathrm{offline}}(\Theta) \;=\; \mathbb{E}_{x,y\sim\pi_T}\Big[\textstyle\sum_t \mathrm{KL}\big(\pi_T \,\|\, \pi_\Theta\big)\Big],
\qquad
\pi_T(\cdot\mid x,y_{<t}) = \pi_{\Theta_0}\big(\cdot \mid s_k\oplus x,\ y_{<t}\big)
\tag{22-6}
$$

[Eq. 3.2, §3.2, p.5]. 교사가 정확히 식 (R)의 $E$-채널 읽기이고 학생이 $\Theta$ 쓰기다. 즉 **(U-E)의 산출을 표적으로 삼아 (U-$\Theta$)를 도는 구성**이며, 이것이 논문에서 가중치 기반 composition의 최고 성적을 낸다(70% 대 study SFT 60%, 프롬프트 상한 83%) [§3.1, p.5; App. B.1, p.27]. on-policy 변형은 $y\sim\pi_\Theta$로 표집하고 forward 또는 reverse KL을 쓴다 [§3.2, p.5].

> **[평가]** $W$의 부재는 결함이 아니라 **구조적 침묵**이며, 그 자체가 이 장의 판정 범위를 정한다. $\Theta$-경로의 대표 반증 논문이 $W$-경로를 인지하지 않은 채 쓰였으므로, "가중치는 잘못된 기록 매체다"라는 결론은 $\Theta$에 대한 결론이고 fast weights로 이월되지 않는다. ch17이 다룬 $W$-경로의 오프라인 recurrence는 이 논문의 어떤 arm과도 대응되지 않는다.

## 22.5 비용 4종

표 22-3 [Continual Facts]의 비용 4종

| 기호 | 논문이 보고한 값 | 판정 |
|---|---|---|
| $B_s$ | optimizer step 수와 항목 수로만. 사실당 24·96·192 step(batch size 1, AdamW lr $2\times10^{-4}$) [§3.1, p.4; App. A.3, p.25]. batch 조건은 $N$개 사실에 $24\times N$ step, 즉 $N=100$에 2,400 step, $N=200$에 4,800 step [App. A.3, p.25] | FLOPs·tokens·wall-clock **논문에 없음** |
| $L_w$ | **논문에 없음.** 지연·TTFT·처리량·서빙 비용 수치가 한 개도 없다. wake 시각의 유일한 수치는 512-token 답변 상한과 그에 대한 truncation 비율(96 step에서 bare 6.1% 대 study 0.1%) [App. D.3, p.31] — 이것은 실패율이지 지연이 아니다 | — |
| $C_{\text{cap}}$ | **논문에 없음.** 바이트도 파라미터 수도 없다. 명시된 것은 $r=16$(또는 4), $\alpha_{\mathrm{LoRA}}=32$, 일곱 투영 행렬, 그리고 매 adapter가 merge된다는 사실뿐 [App. A.3, p.25] | — |
| $\rho$ | 이 논문에서 가장 풍부한 값이지만 **라운드당 비율이 아니라 곡선과 종점 수준으로만** 보고된다. $k=20$·192 step에서 bare 1% 대 study 46%, 짝지은 차 45.6pp(95% CI [38.8, 52.6], $n=57$) [§4.1, p.8]. $k=100$에서 study는 25–28% 고원 [§4.3, p.9]. 역량 쪽 $\rho$: $k=20$에서 SFT $r$=4 $-11$, SFT $r$=16 $-16$, CD $r$=4 $-6$, CD $r$=16 $-28$pp [§6.2·§6.4, p.14–15] | 라운드당 감쇠 상수·적합 곡선·반감기 **없음** |

$B_s$의 step 수조차 계산 예산이 아니다. 논문이 직접 적는다 — "The steps are matched, but generation and token throughput differ, so this remains an operating-point comparison" [§3.2, p.6], "The two data conditions are matched on steps, not tokens or examples" [App. A.3, p.26]. study 쓰기는 step당 12배 많은 항목을 통과시킨다.

> **[평가]** 그러므로 **상각식 (A)를 이 논문에 대해 세울 수 없다.** $B_s$가 FLOPs로 없고 $N_q$도 없다. 손익분기 $N_q^*$를 만들어 내는 서술은 이 논문이 뒷받침하지 않는다. 그런데 결측의 방향이 한쪽이다 — $\Theta$-경로는 merge 이후 사실당 decode 비용이 0인데(§22.7) 그 이점이 회계되지 않았고, $E$-경로 대조군은 매 질문마다 프롬프트 토큰을 내는데 그 비용도 회계되지 않았다. **두 채널을 비교한 논문이 두 채널의 값을 다 매기지 않았다.** ch25가 이 결측을 통일 회계로 받는다.

$\rho$를 인용할 때는 어느 $\rho$인지 반드시 밝혀야 한다. 저장에서 잰 $\rho$와 접근에서 잰 $\rho$가 정반대이기 때문이다(§22.6.3). $k=20$에서 strict 문항을 전부 틀린 사실도 자기 쓰기가 만든 log-probability 상승분의 중앙값 69%(bare) / 79%(study)를 — drift 보정 후에도 57% / 67%를 — 그대로 갖고 있고, "No fact in either condition approaches the erased floor" [§5.1, p.11].

## 22.6 실험과 스케일

**계기부터 밝힌다. 이 논문의 벤치마크는 전부 자체 제작이다.** 사실 247개는 발명된 개체에 대한 한 문장 진술이고 대부분 익숙한 기본값을 위반한다. 학습 데이터는 모델이 생성했고, 문항은 다른 계열의 모델이 작성했으며(정답을 노출하지 않고, 원 모델이 틀리는 것만 남긴다), 채점은 GPT-5.4-mini(temperature 0, reasoning off)가 하되 통과는 다른 계열의 둘째 판정기를 통과해야 한다 [§2, p.3; §2.2, p.4]. 문항은 recall·paraphrase·application·composition·counterfactual 다섯 유형이다 [Table 1, p.3]. 사전 인증 게이트(floor <20%, ceiling >80%, truncation <5%, leak ≤3/30, pass audit ≥24/30)를 floor 3%, ceiling 84%, leak 1/30, pass-audit 29/30으로 통과했다 [§2.1, p.3]. 단, truncation 기준(<5%)은 두 조건에서 통과하지 못했고(원 모델 floor 23%, bare-statement 24 step 11%) 논문은 게이팅 대신 공개를 택했다 [§2.1, p.3; App. A.4, p.26].

### 22.6.1 한 번의 쓰기는 무엇을 만드는가

96 step, 다섯 유형 평균, strict 정책에서 study 데이터가 10.1pp 이긴다(95% CI [+7.0, +13.3], 243개 사실 짝지음). 그런데 lenient 정책에서는 **bare-statement가 13.6pp 이긴다** [§3.1, p.4]. 이 부호 반전이 entailment gap의 정의를 정당화한다[본서 판단].

유형별로 보면 갈라지는 자리가 분명하다. bare-statement는 24 step 안에 recall 97%, paraphrase 95%에 도달하고 96 step에서 포화한다. study는 96 step에서 88%와 83%로 각각 11pp, 13pp 뒤진다. counterfactual은 반대다 — bare는 24 step에서 192 step까지 21–23%에 머물고, study는 45–50%에 도달해 96 step에서 29pp 앞선다 [§3.1, p.4]. composition은 study 60%, bare 40%, 프롬프트 상한 83%, context distillation 70%다 [§3.1, p.5; App. B.1, p.27]. 96 step에서 넓은 데이터는 application을 +21pp, composition을 +18pp, counterfactual을 +29pp 올리는데 "extra bare-statement optimisation improves none of them" [§3.1, p.5].

entailment gap은 bare-statement 사용 문항에서 22–42pp, study 학습 후 1–5pp, 사실을 프롬프트에 넣으면 1–6pp다 [§3.1, p.4].

이 gap이 rank 제약의 산물이 아니라는 것을 논문이 직접 확인한다. 쓰기 강도를 맞춘 full fine-tuning(lr $3\times10^{-5}$, 보류 사실 5개로 선택)에서 bare 28pp, study 4pp, 대비 +24.0pp(95% CI [+20.9, +27.2])로 재현된다 — "Narrow supervision, not the number of trainable parameters, creates the gap" [§3.4, p.7]. Qwen3-8B에서도 bare gap 26pp, 헤드라인 대비 +22.9pp(4B에서 +22.8)로 재현된다 [§3.4, p.6].

**여기서 논문이 자기 기제 설명을 스스로 반증한다.** 통제된 생성 요인분해(같은 32개 사실, rank-16, lr $2\times10^{-4}$, 192 step, 시드 3개, 9개 조건)에서 bare-statement는 27.4pp gap(95% CI [20.2, 35.1])을 남기고, **파생 결론이 데이터 어디에도 없는** 다양한 recall·paraphrase 프롬프트 24개만으로 gap이 5.4pp([1.8, 8.8])로 줄어든다. 그리고 study set이 그 주위에 설계된 바로 그 성분 — 함의와 대조를 담은 use-bearing 예시 — 은 단순한 다양 반복보다 **8.7pp 낮은 점수를 받는다**(95% CI [−15.7, −1.9]) [§7.1, p.17].

> "The supported causal variable is therefore prompt breadth, not the presence of an explicit reasoning step." [Continual Facts §7.1, p.17]

> **[평가]** 이 한 줄이 §22.8의 처방을 바꾼다. 널리 퍼진 직관 — sleep 라운드에서 추론 흔적(reasoning trace)을 만들어 학습시키면 지식이 "사용 가능하게" 들어간다 — 이 이 논문의 자기 요인분해 안에서 부정되었다. 작동하는 것은 **프롬프트의 폭**이다. ch20의 SEAL이 self-edit으로 만드는 것도, ch21의 Dreaming이 만드는 것도 use-bearing 예시이며, 두 논문 어느 쪽도 폭과 use-bearing을 분리한 통제군을 갖고 있지 않다.

### 22.6.2 쓴 것은 살아남는가

20회 순차 쓰기(사실당 192 step, 시드 3개, 매번 merge) 후 study는 46%, bare-statement는 1%를 유지한다. 짝지은 차 45.6pp(95% CI [38.8, 52.6], $n=57$ fact-seed cell) [§4.1, p.8]. 쓰기 실패가 아니다 — bare-statement는 후속 쓰기가 오기 전에는 문항의 65%를 맞힌다. 예산 인공물도 아니다 — 사실당 24 step으로 낮춰도(역량 54%) bare는 6%만 유지한다 [§4.1, p.8]. $k=20$의 방법별 순위는 bare 1, offline distillation 13, online forward 27, online reverse 31, study 46이다 [Fig. 5b, p.8].

이 대비를 인용할 때 두 조건이 따라붙는다. 첫째, 헤드라인의 운영점을 논문 자신이 "The heaviest bare-statement budget, 192 steps on a single sentence, is deliberately severe"라고 적는다 [§3.1, p.4] — 질적 주장은 24 step 조건에서도 서지만 1%라는 수는 온건한 설정을 서술하지 않는다. 둘째, **순위표의 증류 계열은 이미 망가진 레짐에서 잰 값이다** — 20번째 쓰기에 이르면 출력의 35–46%가 반복 루프에 빠지고 역량이 42–49%로 내려가며, 논문은 "the on-policy advantage is real in direction but modest in a degraded regime"이라고 적는다 [§4.1, p.8]. 같은 자리에서 offline distillation의 유지율이 본문 14%와 Fig. 5b의 13으로 어긋난다 [§4.1, p.8].

한 번의 쓰기가 만든 **종류**가 생존을 예측한다: 45개 조건×문항유형 쌍에서 entailment gap과 생존율의 Spearman 상관이 $\rho_S=-0.526$(95% CI [−0.657, −0.197])이고, 조건별 평균으로 접으면 $-0.675$다 [§4.1, p.8].

100회로 늘리면 study 유지율은 0으로 가지 않고 **25–28%의 고원**에 앉는다 [§4.3, p.9]. 20회마다 도는 frozen-anchor consolidation은 $k=100$에서 유지율 25%로 무통합 28%와 잡음 안에서 같고, 첫 통합 직후 5회 시점에서도 41% 대 38%로 "fails to restore reachability even immediately"다. 반면 일반 역량은 매 통합에서 회복하고 $k=100$에서 무통합보다 12pp 높다 [§4.3, p.10].

### 22.6.3 잊힌 사실은 어디에 있는가

$k=20$에서 strict 문항을 전부 틀린 사실은 bare 54개, study 3개이고 [Fig. 9, p.11], 그 사실들의 $R_{\log p}$는 §22.5에 적은 대로 소거 바닥에 근접하지 않는다. 틀린 답의 내용도 특징적이다 — bare-statement 모델은 실패의 **70%**에서 질의된 개체에 **가장 최근에 쓴 사실**의 내용을 붙이고, study 모델은 1%다 [§5.1, p.11]. 고전적 savings 효과는 역전된다: 잊힌 study 사실 하나를 재학습시키는 데 step 예산 전부가 들었고, 한 번도 쓰지 않은 통제 사실은 8 step이 들었다 [§5.1, p.12].

합성은 망각보다 먼저 무너진다. 두 사실을 함께 써야만 답할 수 있는 문항에서 둘 다 가중치에 있으면 32%, 둘 다 프롬프트에 있으면 91%다(짝지은 차 −58.1pp, 95% CI [−76.8, −35.4], 14쌍) [§5.2, p.12]. 개별 사용 가능성 심사는 배치의 87.5%가 통과하므로 개별 실패로는 설명되지 않는다. 병목은 자기 검색이다 — 학습했고 문항에는 답할 수 있는 사실을 그냥 진술하라고 하면 34%만 올바른 내용을 낸다(8%가 축자적). 논문의 문장은 "There is nothing to search"다 [§5.2, p.12].

반대로 문맥에 문장을 넣어 주면 잊힌 사실이 study 조건에서 77–80%, bare 조건에서 14%로 돌아오고, 문맥 사용 능력 자체는 일반 역량보다 빨리 닳지 않는다(20회에 걸쳐 83%→74%, 이중차분 +9.2pp, 95% CI [+3.1, +15.3]) [§5.3, p.13].

### 22.6.4 쓰기의 대가는 무엇으로 측정되는가

12개 조건에서 역량 손실은 $\Theta_0$로부터의 KL을 따라간다($\rho_S=0.83$) [§6.1, p.14]. 요인분해의 400항목 역량 집합에서는 종점 KL과 $\rho_S=0.946$(95% CI [0.787, 0.991]), 누적 local KL과 $0.909$다 [§6.1, p.14].

두 처방이 작동한다. 첫째, **frozen teacher**: 순차 증류에서 교사를 $\Theta_0$로 고정하면 역량 +2pp, KL 0.48, 유지율 54%이고, 교사를 자기 누적 merge로 두면 −31pp, KL 1.70, 유지율 21%(상한에 걸린 출력 제외 시 34%)다 — 짝지은 유지율 차 33.0pp(95% CI [24.6, 41.0]) [§6.2, p.14–15]. batch 판본은 200개 사실을 −3 ~ +1pp 역량으로 쓴다. 둘째, **근접항**: $\lambda_{\mathrm{KL}}=0$에서 −66pp·1%이던 것이 $0.5$에서 −19pp·25%, $1.0$에서 −5pp·36%가 된다(쓰기 정확도는 53%, 49%로 떨어진다) [§6.3, p.15].

같은 12개 조건이 논문 자신의 틀에 저항하는 해리를 둘 보인다. rank 4의 context distillation은 순차 조건 중 역량을 가장 덜 깎으면서(−6pp) 유지율은 같은 rank의 SFT보다 나쁘고(27% 대 42%), rank를 올리면 유지율이 오르고 역량이 내려간다 — 논문 표현으로 "the reverse of its usual reputation"이며, 논문이 다른 곳에서 승인하며 인용하는 Biderman 외 2024와 반대 방향이다 [§6.4, p.15; §8, p.19].

> **[평가]** 그런데 같은 sweep에서 $\Theta_0$로부터 잰 KL은 1.8–2.4 사이에 머문다 — "even as capability recovers by more than sixty points" [§6.3, p.15]. 논문 자신의 표현대로 KL은 "a cost coordinate that holds across methods, not a sufficient causal mechanism"이다 [§6.1, p.14]. **KL은 조종간이 아니라 계기판이고, 이것이 §22.7의 운영 함의를 정한다.**
>
> **이 문장의 범위를 KL로 좁혀 둔다.** 초판은 이것을 "가장 깨끗한 정량 법칙이 조종간이 아니라 계기판"이라고 썼고, 그 문면대로 읽으면 역량 실패에 조종 가능한 손잡이가 없다는 뜻이 된다. 그 확장은 이제 거짓이다 — 손잡이가 하나 나왔고, 그것은 목적함수가 아니라 **갱신 자체의 기하**에 붙는다. [Spectral Collapse](*Spectral Characterization and Mitigation of Sequential Knowledge Editing Collapse*, arXiv:2601.11042)는 매 라운드 현재 가중치 행렬을 SVD해 상위 특이방향의 부분공간을 구하고, 갱신을 그 여공간으로 사영해 지배적 특이 삼중항을 보존한다. 결과는 max singular-subspace similarity가 모든 라운드에서 1로 유지되고 [Spectral Collapse App. G.6, pp.15-16], 무보호 편집기가 3,000–8,000 편집에서 붕괴하는 스트림에서 10,000 편집 뒤에도 GLUE 평균 86.34%가 남는다 [Spectral Collapse §4.2, p.6]. **KL이 계기판이라는 판정은 KL에 대해 옳고, 역량 실패 전반에 대해서는 옳지 않다.**

간섭의 소재도 특정된다. 2×2 교차 설계(저장 방식 × 이후 쓰기 방식, 24개 저장 사실, 시드 3개, 동일한 10회 후속 쓰기)에서 최종 유지율은 9.0 / 47.8 / 7.8 / 44.2%이고, 효과 분해는 **이후 쓰기 효과 +37.6pp**(95% CI [28.9, 46.3]), 저장 방식 효과 −2.4pp([−9.2, 4.1]), 상호작용 +2.4pp([−11.5, 14.1])다 — "Interference is caused by the incoming write, not by how the earlier fact was stored" [§7.2, p.17–18].

통제 시도는 전부 실패한다. 사전 등록한 step-16 조기 예측자는 최종 이득과 순위상관 0.083, 선형화 예측자(식 7.1)는 다음 갱신과 $\rho_S=0.795$이지만 전체 수열 후 망각과는 $-0.258$이다. 측정된 충돌 방향을 잘라내는 protected optimizer는 +1.4pp(95% CI [−2.1, 5.0], 목표 15pp)로 norm을 맞춘 무작위 사영(+1.8)과 구별되지 않는다. bridging 데이터 8개 조합 중 네 조건 결합을 통과하는 것이 없고, activation 기반 개입(사영·patching·block swap)의 최대 구제폭은 10pp 문턱에 대해 5.9pp다 [§7.3, p.18].

### 22.6.5 이 결과가 서 있는 바닥

**둘째 평가의 주 대비는 실패했다.** 목적 제작한 prior-conflict 계기(240개 사실, 4개 레시피, 시드 3개)에서 strict 정책의 모든 신뢰구간이 0을 가로지르고 사전 설정한 +10pp 지지 기준에 닿지 않는다. truncation되지 않은 행만으로 다시 계산하면 gap이 +0.1pp(invert)와 +0.2pp(neutral)로 붕괴한다 — 잔여 신호가 truncation 인공물이었다 [App. D.3, p.31]. 다양성 효과는 application 문항만으로 사후 제한한 뒤에야 되살아난다 [App. D.4, p.33].

**헤드라인 대비는 문항 양식에 의존한다.** 기본값을 진술하고 대조를 묻는 cue를 붙이면 bare-statement의 strict 정확도가 23%에서 99%로 오르고 entailment gap이 31.8pp에서 0.9pp로 무너진다 [App. D.4, p.32]. [Continual Facts §3]을 지탱하는 22–42pp gap은 cue 없는 문항에서만 존재하며, 이 논문의 주 평가는 구성상 cue가 없다.

**문맥 채널은 가중치 채널이 도달한 규모에서 시험되지 않았다.** 문맥에 동시에 들어간 사실은 최대 2개인데([Continual Facts §5.2]의 쌍 실험) 가중치 채널은 100개까지 밀린다 [§5.2, p.12]. 게다가 그 대조군 자체가 완벽하지 않다 — 프롬프트 상한은 composition에서 83%이고 tier별로 77–95%이며 [App. B.1, p.27; App. D.4, p.33], cue 유무로 76%와 95% 사이를 오가고 [App. D.4, p.32], bare-statement 쓰기 스트림 아래에서는 프롬프트 기준선 자체가 45.5pp 떨어진다 [§7.2, p.18].

나머지 바닥은 이렇다. **100회 확장 — 이 책이 가장 원하는 결과인 25–28% 고원 — 이 논문에서 가장 약한 증거다.** 범위가 12개 셀에서 8개로 축소되었고("after partial records existed"), 종점 대비가 사후에 선택되었으며, 조건별로 학습률이 조정되었고, 불확실성 구간이 없고, 문맥 상한이 77.3%로 인증 기준 80%에 미달한 채 수용되었다 [§4.3, p.9; App. B.3, p.27]. 목적함수 한 계열은 아예 평가되지 못했다 — GRPO 조건은 "received almost no signal because the base model never produced the target counterfactuals" [§3.3, p.6]. 요인분해 16셀 중 1셀(reverse-KL·bare·LoRA)은 생성 상한 반복으로 검열되어 생존 상관은 15개 조건의 complete-case 분석이다 [§3.3, §4.1]. 확증 결과는 전부 4B이고 8B는 gap 한 측정의 복제 1회뿐이며 [App. C.4, p.29], 같은 조건의 유지율이 [Continual Facts §6.2]에서 11%, [Continual Facts §6.4]에서 12%로 인쇄된다 [p.14, p.15].

넷이 더 붙는다. **수치의 비교 가능 범위가 절마다 다르다** — [Continual Facts §6.4]의 12개 조건은 "their own optimisation budget and question set, so retention levels here are not comparable to those of §4"이고 [§6.4, p.15], 요인분해는 네 쌍이 조건별로 조정된 학습률을 써서 "an association across operating points rather than a one-variable causal effect"다 [§3.3, p.6]. **하중을 지는 주장 여럿이 작은 $n$ 위에 있다** — 역savings는 잊힌 study 사실 3개, protected optimizer의 null은 고유 저장 사실 5개(60개 짝 단위로 재표집), 합성 붕괴는 14쌍 47문항, 사전 tier 역전은 19개 삼중쌍(부호검정 $p=0.064$)이고, 요인분해의 864 단위는 32개 사실의 반복측정이다 [§5.1, p.12; §7.3, p.18; App. A.6, p.26; §5.2, p.12; §4.2, p.9; App. C.1, p.28]. **인쇄 오류가 더 있다** — App. D.1이 프롬프트 상한이라고 적은 "(neutral 96%, invert 80%, confirm 77%)"은 실제로는 App. A.5의 문항 생존율(95.9 / 79.5 / 76.3)이고 같은 부록의 Fig. 20은 상한을 97 / 87 / 89로 인쇄하며 [p.30; p.26], "13,728 records (240 facts × 4 recipes × 3 seeds)"는 곱이 2,880이다 [App. D.3, p.31]. 재현 쪽에서는 의존성 lockfile을 고정해 두고도 "the exact base-model revision was not recorded in the run rows"이다 [App. A.3, p.26]. 마지막으로 [Continual Facts §7.2]의 간섭 대비에는 논문이 배제하지 못한 교란이 남는다 — bare-statement 스트림에서 반복 생성 실패가 4.2%·3.2%(study 스트림 0%)이고, 논문은 "broad generation pathology remains a possible explanation"이라고 적는다 [§7.2, p.18].

## 22.7 Systems/serving 함의

**decode에서 바뀌는 것은 없다[본서 추론].** fast weight도, 토큰당 상태 갱신도, KV cache를 넘어 자라는 상태도 없다. $k$회 쓰기 후의 서빙은 Qwen3-4B 모양 $\Theta$ 하나 위의 평범한 frozen forward pass다 [App. A.3, p.25].

> **[해설]** adapter가 보존되지 않고 merge되므로 **저장된 사실당 한계 decode 비용이 정확히 0이다.** 사실 100개를 넣은 모델의 토큰당 비용은 0개를 넣은 모델과 같다. 이것이 이 논문에서 $\Theta$-경로가 $E$-경로에 대해 갖는 유일한 명백한 우위이고, **논문은 이 사실을 한 번도 적지 않는다.** 반대편에서 문맥 채널은 질문마다 문장 토큰을 prefill하고 KV를 잡는다. 그 비용도 논문에 없다. 그래서 "context wins"라는 결론은 서빙 산술이 뒷받침하는 것보다 강하게 읽힌다.

트래픽은 읽기가 아니라 **쓰기 쪽**에 있다. 사실 하나당 4B 모델 위의 forward+backward 192 step(batch size 1), adapter용 AdamW moment, 그리고 merge — merge는 전 층 일곱 투영 텐서에 걸친 full read-modify-write다. 즉 100회 쓰기는 전체 모델 RMW 100회를 뜻하고, 논문은 이 중 어느 것도 측정하지 않는다. frozen-anchor consolidation의 교환비는 그 위에서 더 나쁘다 — batch 조건이 $N$개 사실에 $24\times N$ step이므로 $k=100$의 통합 1회가 2,400 step, 단일 쓰기의 열 배가 넘고 [App. A.3, p.25], $m$회마다 통합하는 정책은 쓰기 $k$까지 누적 $O(k^2/m)$ step을 쓴다[본서 산술]. **그 비용이 사는 것은 역량이지 도달성이 아니다** [§4.3, p.10].

batching은 온전하다. 모든 사실이 전역 $\Theta$ 하나에 merge되고 사용자별·세션별 가중치 상태가 없기 때문이다. Rosetta 사전의 "batch로 weight 공유 ↔ $\Theta$-경로에서 깨짐" 행은 **이 논문의 설정에서는 발동하지 않는다.** 사용자별 사실로 옮기는 순간 per-user $\Delta\Theta$가 생기고 grouped-GEMM 문제가 상속되는데, 이 논문은 사용자별 쓰기를 고려하지 않는다 — 이 책이 가장 궁금해하는 배포 형태에 대해서는 아무 말도 하지 않는다.

운영 계기 하나는 즉시 쓸 수 있다. drift(식 6.1)는 고정 프롬프트 풀에 대해 고정된 참조 모델과 비교하는 forward pass이므로 서빙 fleet에서 싸게 돌릴 수 있고, 12개 조건에서 $\rho_S=0.83$, 16개 조건에서 $0.946$으로 역량 손상을 정렬한다. **단 KL은 계기판이지 조종간이 아니다** — [Continual Facts §6.3]이 KL을 움직이지 않고 역량을 60pp 넘게 회복시켰다. KL을 카나리아로 쓰고 제어 신호로 쓰지 않는 것이 이 증거가 허락하는 사용법이다[본서 판단].

**조종간은 다른 좌표에 있다.** [Spectral Collapse]가 그 좌표를 제시한다 — 갱신의 스펙트럼 지지집합을 제약하는 것은 손잡이이고, 목적함수가 아니라 **갱신에** 적용되며, 벽시계 대가가 100 편집당 +4.05%(GPT-J)·+1.91%(LLaMA3)다 [Spectral Collapse Table 9, p.19]. 서빙 관점에서 이 비용의 형태가 중요하다 — 라운드마다 가중치 슬라이스 하나의 SVD와 조밀 행렬곱 둘이며, decode 경로에는 아무것도 남기지 않는다. 즉 **쓰기 경로에 몇 %를 더 내고 쓰기 지평을 두 자릿수 늘리는 거래**다. 그리고 같은 논문이 KL보다 싼 카나리아도 준다 — 가중치 노름 증가비다. GPT-J 3층에서 10,000 편집에 걸쳐 $\ell_2$ 노름이 무보호 198.5배로 부푸는 동안 보호를 걸면 1.55배에 머문다 [Spectral Collapse Table 2, p.7] [본서 산술]. 노름은 forward pass도 필요 없이 가중치에서 바로 읽히므로, 세 계기 — KL drift, 노름 증가비, 특이방향 유사도 — 중 가장 싼 것이 가장 이른 것일 수 있다. 어느 것이 먼저 발화하는지를 한 축에 놓고 잰 논문은 아직 없다.

## 22.8 한계와 bridge-out

### 22.8.1 지속 주입의 실패 양상

증거를 모으면 실패가 하나가 아니다. 이 장이 소유하는 둘째 정의다.

> **정의.** **지속 주입의 실패 양상**은 (U-$\Theta$)를 반복할 때 관측되는 네 가지 서로 다른 붕괴다. **(a) 회수-사용 분리** — 쓰기가 축자 회수는 만들되 사용 가능한 지식은 만들지 않는다(entailment gap 22–42pp). **(b) 역량 침식** — 누적 갱신이 $\Theta_0$의 일반 능력을 깎으며, 그 크기는 $\Theta_0$로부터의 KL로 정렬된다. **(c) 주소 상실** — 내용은 $\Theta$에 남아 있으나 질의가 닿지 못하고, 실패한 질의는 최근에 쓴 사실로 흘러간다. **(d) 합성 불능** — 개별적으로 사용 가능한 두 사실이 함께 쓰이지 못한다. 네 양상은 서로 다른 처방을 요구한다 — (a)는 $\mathcal{R}_k$의 폭이, (b)는 frozen teacher 또는 근접항이 다루고, (c)와 (d)에는 이 논문이 시도한 어떤 처방도 듣지 않았다.

이 분해가 중요한 이유는 처방이 섞이지 않기 때문이다. (b)를 고치는 두 처방은 (c)를 고치지 못한다 — frozen-anchor consolidation이 역량을 12pp 회복시키면서 유지율을 잡음 범위 안에 두는 것이 그 증거다. 반대로 (a)를 고치는 데이터 폭은 (c)를 크게 완화한다(1% → 46%). 즉 **네 양상은 하나의 "catastrophic forgetting"이 아니다.**

### 22.8.2 이송된 불리 사실 — 결함의 유형학

Part I의 ch06·ch07·ch09가 개념을 실었고 논문 평가에 속하는 불리 사실을 이 장으로 넘겼다. 개별 나열 대신 유형으로 묶는다. 각 행이 반증 축에 거는 제약이 다르다.

표 22-4 이송된 불리 사실의 유형학

| 유형 | 대표 사례와 출처 |
|---|---|
| 결과가 표 없이 곡선으로만 존재 | EWC 논문에는 결과표가 없고 EWC의 정확도 수치가 본문 어디에도 없다(두 표 모두 hyperparameter 표) [EWC Table 1, Table 2]. DGR의 §4.1–§4.3 네 실험이 전부 곡선이며, 유일한 수치 표는 단일 과제 예비 실험이다 [DGR Table 1] |
| 인쇄값이 자기 데이터와 어긋남 | GEM Table 2가 App. B.3와 불일치(GEM 0.654 vs 0.6783, iCaRL 0.508 vs 0.5462) [GEM F2]; Table 3의 1-epoch BWT가 App. B.2와 10배 차이(+0.05 vs +0.0048) [F3]. EWC의 Atari replay buffer 크기가 본문 $5\times10^5$과 Table 2의 50,000으로 열 배 어긋난다 [EWC App. §4.2 vs Table 2]. 기억 용량 논문은 같은 설정을 두 부록 표에서 다르게 인쇄하고(2.29×10⁶/2.97% vs 2.39×10⁶/1.11%) [U13], 수렴 문단에서 같은 4M 데이터셋을 3.56–3.65×10⁶과 2.95×10⁶ 두 값으로 적으며 같은 6.86M 모델의 Table 1 용량(2.51×10⁷ bits)과 약 7배 어긋난다 [U4], 모델 크기 범위를 세 가지로 적으며 [U9], Table 2의 'GPT2-Medium' 행이 실제로는 GPT-2 small이고 [U8], §4의 그림 참조 두 곳이 틀렸으며 [U17], 부록 데이터셋 크기가 캡션·패널 제목·산문에서 서로 다르다 [U14]. BWT 분모 문제는 ch07이 이미 정정했다 |
| 교정되지 않은 원고 상태 | EWC의 vendored v2는 최종본이 아니다 — Table 2의 캡션은 "MNIST 그림들의 hyperparameter"라고 적혀 있으나 내용은 Atari의 것이고, Table 1의 열 머리는 이 원고에 없는 "Figure 3"을 참조하며, 미해결 LaTeX 참조 `Appendix app:atari`가 남아 있고, 에이전트 차이 목록의 두 항목이 모두 '(e)'로 인쇄되어 있다. §2.1의 SGD 실패 서술은 과제 A와 B가 뒤바뀐 채 인쇄되어 있어 축자 인용하면 오류가 그대로 전파된다 [EWC §2.1, §2.2, Table 1, Table 2] |
| 통제군과 구별되지 않음 | **rate–distortion Experiment 1: 여섯 KV 압축법 중 어느 것도 random-eviction 대조군을 유의하게 이기지 못한다.** 저자가 직접 적는다 — "The gaps are modest at this scale, and the ordering of methods is beside the point; what matters is the comparability the BPT axis delivers" [Rate–Distortion §14.1]. GEM이 '오라클 상한'을 3열 중 2열에서 넘어선다(iid 0.83/0.87 vs GEM 0.86/0.88) [F4]. DGR은 자기 상한 대조군(ER)을 이긴 적이 없고 비길 뿐이며 [DGR §4.2], EWC arm은 아예 없다 |
| 기제가 자기 실험에 반증됨 | EWC: Fisher의 nullspace 방향 섭동이 역Fisher 방향만큼 성능을 떨어뜨린다 — "we are over-confident about certain parameters being unimportant" [EWC §2.2]. [Continual Facts]: use-bearing 예시가 단순 다양 반복보다 8.7pp 낮다(§22.6.1) |
| 저장을 기각한 뒤 저장을 되살림 | EWC의 Atari 시스템은 "separate short-term memory buffers for each inferred task"를 유지한다 — 도입부가 replay를 기각한 바로 그 자원이다. GEM은 $M=5120$으로 MNIST 20,000의 25.6%, CIFAR 50,000의 10.24%를 원문 그대로 저장한다 [F5] |
| 조건이 붙었으나 검증되지 않음 | EWC의 task-recognition 보증은 과제 수가 $o(n/\log n)$으로 자랄 것을 요구하는데 실험이 이를 확인하지 않는다. Atari의 Fisher는 replay buffer에서 뽑은 100개 미니배치로 추정되어 stale·off-policy 분포 위에 서 있고, 논문은 이를 언급하지도 표본 수 민감도를 시험하지도 않는다 [EWC §2.2, Table 2]. GEM의 보증은 국소 선형성과 대표성 있는 기억을 요구하는 선형화이며 결론 자체가 "이전 과제의 손실을 높이지 **않을 가능성이 크다**"인데, 투영된 갱신이 실제로 과거 손실을 높인 횟수를 세는 실험이 없다 — $\Pi$를 만든 코드로 직접 잴 수 있는데도 그렇다 [GEM F11]. Atari의 EWC penalty는 게임당 2천만 frame 전에는 꺼져 있고 그 게이트를 바꾼 실험이 없다. MNIST의 $\lambda$ 값이 본문에 없고, multi-task 형태(과제별 penalty $O(k\lvert\Theta\rvert)$ vs 병합 $O(\lvert\Theta\rvert)$)가 결정되지 않아 논문이 자기 용량 거동을 확정하지 못한다. DGR의 보증은 "생성기가 입력 분포를 복원하는 한" 성립하는데 복원 지표(FID·coverage)가 없다 |
| 권고와 은유가 실증 없이 남음 | DGR이 권고하는 세 프레임워크 결합은 "there is no straightforward mixture of any two frameworks"라며 약한 결합 하나(LwF-GR)만 만들어졌고 그것은 테스트 시각 과제 문맥을 요구한다 [DGR §5]. 설계의 하중을 지는 생물학적 근거 — "해마는 replay buffer보다 생성 모델에 가깝다" — 는 인용된 신경과학 세 편에만 기대며, 논문 자신은 생성적 해마와 buffer를 구별할 어떤 시험도 하지 않는다 [DGR §1] |
| 예측이 세워졌으나 확립되지 않음 | **rate–distortion 예측 (4)의 super-linearity가 확립되지 않았다.** 압축 이벤트 5–25회에서 irreversible recall이 0.33–0.56 구간이라고 보고할 뿐 적합 곡선도 지수도 선형 귀무 대조도 없고, 그림 캡션의 표현은 "loses roughly half its facts at every frequency"로 가속이 아니라 평평하고 나쁨을 뜻한다 [Rate–Distortion §14.2] |
| 단일 시드·단일 설정·좁은 baseline | GEM의 15개 20-과제 실행이 전부 단일 실행이고 오차막대가 없으며 [F12], 헤드라인을 만든 $\gamma$의 $0$ ablation이 없고 [F10], memory ablation은 CIFAR-100에서만이며 [F6], 비용 표는 MNIST·CPU만이고 [F7], CIFAR는 과제별 최종 선형 분류기를 둬 테스트 시각 과제 식별자를 요구하며 [F8], FWT는 방법 품질이 아니라 데이터셋 유사도를 잰다 [F9]. EWC baseline은 재구현이고 정규화 강도가 데이터셋 간 세 자릿수 차이이며 [F13], 용어가 두 쪽 안에서 불일치한다 [F14]. EWC의 MNIST는 시드·분산 미보고에 arm이 둘뿐이고 Fig. 3C 일반화 주장이 단일 게임 위에 있다. DGR은 아키텍처·생성기·$\omega$·시드가 전부 단일이고 SVHN 실패가 크기 없이 supplement로 유예된다 |
| 강한 결론에 자기 부록의 반례 | 기억 용량 논문의 membership-inference 법칙은 인쇄된 상수로 계산하면 $\lvert D\rvert\to\infty$에서 $F_1=1.006$, 최대 $1.17>1$이 나와 한 문단 뒤의 "0.5로 수렴" 진술과 반대다 [U6]. "예측이 참 $F_1$의 1.5점 이내"라는 주장이 자기 Table 2에서 최대 9.31점 어긋난다 [U7]. 합성 데이터에서 성능이 0.54 아래로 내려가지 않는다는 부록 그림이 그 수렴 주장의 반례다 [U20]. 초록의 "grokking"은 본문의 double descent이고 인용도 틀렸다 [U16] |

> **[평가]** 이 표에서 반증 축이 실제로 가져가는 것은 두 줄이다. 첫째, **망각 대응의 두 원형(EWC·DGR·GEM)은 재현 가능한 수치를 거의 남기지 않았고, 남긴 곳에서는 통제군이 방법과 갈리지 않았다.** 그러므로 "이미 해결된 문제"라는 전제로 (U-$\Theta$)를 반복하는 설계는 근거가 없다. 둘째, **압축 쪽 이론도 같은 상태다** — 무엇을 버릴지 정하는 법이 무작위 폐기를 이긴다는 증거가 그 문헌의 유일한 자체 실험에 없다. $\Theta$-경로가 기대는 두 축(망각을 막는 법, 버릴 것을 고르는 법)이 둘 다 실증으로 서 있지 않다.

### 22.8.3 용량·망각·붕괴가 함께 거는 제약

세 축은 따로 인용되어 왔지만 함께 걸린다. 이 장의 셋째 정의다.

> **정의.** **용량·망각·붕괴의 결합 제약**은 (U-$\Theta$)를 $k$회 반복하는 설계가 동시에 만족해야 하는 세 조건이다. **(i) 용량** — 누적 저장량이 $H(\Theta)$를 넘을 수 없으며(→ ch09), 실측된 도달점은 파라미터당 3.64 bits이되(→ ch09 §09.2; [기억 용량 Fig. 6 캡션], 반정밀도) 이는 **하한**이라 $\Theta$-경로를 반증하지 못한다. **(ii) 망각** — 라운드마다의 상대 손실 $\rho$가 0으로 수렴하지 않으면 유효 저장량은 용량 천장보다 훨씬 앞에서 포화한다(→ ch07의 $\Pi$와 식 (7-11)). **(iii) 붕괴** — $\mathcal{R}_k$가 자기 생성일 때 누적 스케줄에서만 오차가 유계이고(→ ch06), 교체 스케줄에서는 발산한다.

세 축이 [Continual Facts]에서 만나는 지점이 100회 실험이다. 유지율이 0으로 가지 않고 25–28% 고원에 앉는 것은 (i)의 천장에 닿아서가 아니다 — 4B 모델에 사실 100개는 어떤 용량 회계로도 근처에 가지 않는다. 원인은 (iii)도 아니다 — $\mathcal{R}_k$는 별도 모델이 한 번 생성해 고정했고 자기 생성 루프가 없다. 남는 것은 (ii)뿐이며, 그것도 저장의 (ii)가 아니라 **접근의 (ii)**다.

**그리고 2026-05·06에 세 논문이 (iii)의 축을 세 점으로 나누어 놓았다. 이 셋이 정렬되면 열화의 원인이 「반복」에서 「자기생성」으로 좁혀진다.**

- **자기생성이면 무너진다.** [Continual Internalization](*Rethinking Continual Experience Internalization for Self-Evolving LLM Agents*, arXiv:2606.04703)은 라운드 $k$의 정책이 만든 궤적을 요약해 라운드 $k+1$의 학습 집합으로 삼는 닫힌 루프를 돌린다. $K=3$에서 여덟 구성 중 일곱이 무너지고, 붕괴의 형태가 행동 모드 붕괴다(premature answer 63.82%). 논문 자신의 가장 강한 처방은 **루프를 여는 것**이다 — "off-policy context-distillation on high-quality teacher trajectories provides a substantially more stable training signal than on-policy".
- **수축이 있으면 무너지지 않는다.** [Lifelong Normalization](*More Edits, More Stable*, arXiv:2605.11836)은 스트림이 밖에서 오고(고정 분포에서 i.i.d. 추출된 라벨 배치, $\mathrm{gen}(\cdot;B_s)$가 항등) 갱신이 running 통계로 정규화된 gradient 위의 닫힌 형태 ridge 1스텝인 구성에서 **20,000 라운드(2M 편집)**를 버틴다 [Lifelong Normalization §5.2, p.7; App. B.3.1-B.3.2, p.22]. 정규화 모듈을 빼면 같은 스트림에서 무너진다 [Lifelong Normalization Fig. 1(a), p.1]. **replay는 0이고 원본 아카이브도 0이다** — 어떤 편집도 두 번 쓰이지 않는다.
- **수축의 형태는 하나가 아니다.** [Spectral Collapse]는 정규화가 아니라 부분공간 투영으로 같은 일을 하고 10,000–20,000 편집을 버틴다. 즉 라운드당 이동을 유계로 만드는 장치가 정규화(LN)든 사영(REVIVE)이든 결과가 같다.

**두 caveat를 함께 인쇄한다.** 첫째, [Lifelong Normalization]은 **이전 라운드에 쓴 항목의 유지를 한 번도 재지 않는다** — 재는 것은 현재 배치의 Efficacy·Generalization과 무관 입력의 Specificity다. 그러므로 "20,000 라운드를 버텼다"는 (ii)에 대한 진술이 아니라 (iii)에 대한 진술이다. 둘째, 두 라운드의 눈금이 다르다 — [Continual Facts]의 한 라운드는 수렴까지의 LoRA 재학습이고 [Lifelong Normalization]의 한 라운드는 편집 100건·약 10초의 ridge 해다. **라운드당 예산 $B_s$나 라운드당 $\lVert\Delta\Theta\rVert$를 붙이지 않고 100과 20,000을 한 축에 놓는 것은 틀린다.**

> **[평가]** 그러므로 $\Theta$-경로의 실제 병목은 지금까지 이 라인이 인용해 온 용량 상한이 아니다. **가중치는 넣을 자리가 없어서 지는 것이 아니라 넣은 것을 부를 이름이 없어서 진다.** 이 문장이 ch09의 3.64 bpp 논의와 ch07의 $\rho$ 논의를 재배치한다 — 용량은 아직 구속력 있는 제약이 아니고, 구속하는 것은 주소다.
>
> **그리고 이 판정은 2026-05 이후의 세 논문으로 강해진다.** 이 장의 반증 조건 셋 중 (ii) 역량 실패는 이제 **처방이 있는 실패**다 — 갱신의 기하를 통제하면 10,000–20,000 편집까지 간다. (i) 생성 실패도 처방이 있다(데이터의 폭). **(iii) 도달 실패만 어떤 처방으로도 움직이지 않았고, 그러므로 이 장이 남기는 유일한 살아 있는 반증자다.** 세 축 중 둘이 고쳐졌는데 남은 하나가 그대로라는 것은 이 장의 결론을 약화시키지 않고 좁힌다.
>
> **정정된 형태로 다시 쓴다. 열화는 「반복한다」의 성질이 아니라 갱신 규칙의 성질이다.** $\rho$는 (갱신 규칙, 스트림의 출처, 재는 축)이라는 삼중항의 함수이며, 이 셋을 지정하지 않은 $\rho$ 진술은 이 corpus에서 검증되지 않는다. 우선순위도 증거가 정한다 — **자기생성 여부가 1순위**이고(붕괴한 쪽에는 닫힌 루프가 있었고 버틴 쪽에는 루프가 아예 없다), 무엇을 재는가가 2순위, 라운드당 $\Theta$ 이동량이 3순위, 안정화 장치가 4순위다. 그러므로 이 장의 판정은 "가중치에 사실을 계속 쌓을 수 없다"가 아니라 **"기하를 통제하지 않으면 쌓을 수 없고, 통제해도 주소는 돌아오지 않는다"**이다. 자기 생성 $\mathcal{R}_k$를 쓰는 설계(→ ch20의 SEAL, ch21의 Dreaming)는 정확히 1순위 축의 나쁜 쪽에 있고, 그 조합에 기하 통제를 얹은 실험은 corpus에 0건이다.
>
> **가드레일 한 줄.** [Spectral Collapse]가 인쇄하는 Efficacy·Paraphrase는 편집한 사실을 **단서와 함께** 물어보는 문항이다. 이 장의 §22.6.5가 보인 대로 단서를 붙이면 bare-statement의 strict 정확도가 23%에서 99%로 오른다. **그러므로 그 논문의 높은 Efficacy를 도달성의 증거로 읽으면 안 된다** — 그것은 쓰기가 내용을 만들었다는 증거와 일반 역량이 살아남았다는 증거, 즉 (i)과 (ii)에 대한 증거이지 (iii)의 반증이 아니다. 단서 없는 문항에서 같은 사실이 불려 나오는지는 그 논문이 재지 않는다.

### 22.8.4 논문의 open questions와 살아남기 위한 조건

논문이 남긴 것 중 이 책이 받는 것은 넷이다. **도달성은 SFT와 offline·online 증류 전반에서 미해결이다** — "The third remains unresolved" [§9.3, p.20]. **왜 모든 사실을 다시 증류해도 다시 주소가 붙지 않는지** 실험이 답하지 않는다 [§4.3, p.10]. **fine-tune된 skill이 discrete fact와 같은 방식으로 무너지는지** 시험되지 않았다 [§9.2, p.20]. **지식을 쓸 수 있는 RL 목적함수**는 GRPO가 신호를 받지 못한 채 미래 과제로 남았다 [§3.3, p.6].

> **[평가]** 이 증거가 허락하는 것은 판정이 아니라 **조건**이다. $\Theta$-경로가 살아남으려면 다음 여섯을 동시에 만족해야 한다. 초판은 다섯을 적었고, 2026-05 이후의 세 논문이 여섯째를 강제했다.
>
> **(C1) 쓰기의 표적은 폭이지 추론이 아니다.** $\mathcal{R}_k$는 넓어야 하고, 넓다는 것은 use-bearing 예시를 넣는 것이 아니라 같은 내용에 닿는 표면형을 늘리는 것이다(−8.7pp, §22.6.1). SEAL·Dreaming식 self-edit이 이 조건을 만족하는지는 두 논문 어디에도 통제군이 없어 판정되지 않는다.
> **(C2) 교사는 고정되어야 한다.** 자기 누적 merge를 교사로 쓰는 순차 증류가 12개 조건 중 최악이다(−31pp, KL 1.70, 유지율 21%). 반대로 $\Theta_0$ 고정 교사는 20회 쓰기를 +2pp·KL 0.48·유지율 54%로 통과시킨다. **이송 연산자를 자기 자신에 겨누는 설계는 이 증거에 정면으로 걸린다.**
> **(C3) 근접항이 있어야 하되 KL을 지표로 삼아서는 안 된다.** $\lambda_{\mathrm{KL}}$은 역량을 −66pp에서 −5pp로 되살리면서 $\Theta_0$ KL을 움직이지 않는다. KL은 배포 카나리아이고 목적함수의 조종간이 아니다.
> **(C4) 주소를 $\Theta$ 밖에 둬야 한다.** 이 논문이 실증으로 지지하는 유일한 합성은 $\Theta$를 기록 매체가 아니라 **정본이 다른 곳에 있는 write-through cache**로 쓰는 것이다 — "Weight updates remain useful as a cache when a canonical copy exists elsewhere, because frozen-teacher batch distillation preserves general capability while consolidating many facts" [§9.2, p.20]. 이 구성에서 $E$-경로와 $\Theta$-경로는 경쟁자가 아니라 보완재다.
> **(C5) 회계를 세워야 한다.** 이 논문은 두 채널을 비교하면서 FLOPs·지연·바이트를 하나도 재지 않았고, 문맥 채널을 $|E|\le 2$에서만 돌린 뒤 100개 사실 규모로 결론을 옮겼다. **$|E|=100$에서 문맥 채널이 내는 prefill과 KV 비용은 merge된 $\Theta$가 내지 않는 비용이다.** 이 결측이 메워지기 전에는 "context wins"가 서빙 결론이 아니다.
> **(C6) 업데이트의 기하를 통제해야 한다.** C2·C3이 무엇을 학습 신호로 삼을지에 거는 조건이라면, C6은 그 신호가 $\Theta$를 **어느 방향으로 얼마나** 움직이게 둘지에 거는 조건이다. 두 형태가 실증으로 서 있다 — running 통계로 gradient를 정규화하는 것(20,000 라운드까지 유지, 빼면 같은 스트림에서 붕괴)과 지배적 특이방향의 부분공간을 보존하고 갱신을 그 여공간으로 사영하는 것(10,000 편집 뒤 GLUE 평균 86.34% 유지, max SS = 1)이다. 둘 다 목적함수가 아니라 **갱신에** 붙고, 둘 다 싸다 — 100 편집당 벽시계 +4.05%/+1.91% [Spectral Collapse Table 9, p.19]. 이 조건이 여섯 중 유일하게 **구현이 이미 존재하고 비용이 인쇄된** 조건이며, 그러므로 가장 먼저 검사해야 할 조건이다. 검사 형태도 정해져 있다 — 자기 생성 $\mathcal{R}_k$ 위에서 기하 통제를 켜고 끄는 대조이고, corpus에 0건이다.
>
> 최종 판정은 하지 않는다. 경로별 판정은 ch27이, 반증 조건의 정식화는 ch30이 소유한다. 이 장이 넘기는 것은 위 여섯 조건과, 그 조건이 어떤 관측으로 깨지는지다. **그리고 여섯 중 하나가 이 장의 결론을 바꿨다는 사실을 함께 넘긴다** — 이 장의 초판은 반복 자체를 피고로 세웠고, C6은 피고를 갱신 규칙으로 바꾼다.

### 22.8.5 경로 간 인용 여부

비대칭이 뚜렷하다. **$E$-경로는 사상으로 깊이 교전하되 이름은 하나도 인용하지 않는다.** Snell 외 2022(context distillation), Cartridges(2506.06266), gist token, Memory Layers(2412.09764), Ovadia 외 2024가 인용되고, MemGPT·Letta STC·Mem0·Zep·ReasoningBank은 전부 없다. 그러면서 $E$-경로의 논지에 독립적으로 도달한다 — "Retrieval-augmented systems preserve this handle outside the weights. The write stores content, while the context supplies its address at use time" [§9.1, p.20]. 수렴이 독립적이므로 증거로는 더 강하고, **$E$-경로의 방법에 대해서는 아무 증거도 주지 않는다**(검색기가 오라클이고 $B_s=0$이므로).

**$W$-경로는 완전한 침묵이다.** 참고문헌에 Titans·Miras·Atlas·TNT·Nested Learning·`Language Models Need Sleep`·TTT·fast-weight programming·Mamba·DeltaNet·RWKV가 한 편도 없고, "fast weights"·"test-time training"이라는 말이 본문에 나오지 않는다.

**$\Theta$-경로는 옛 계보만 인용한다.** LoRA·ROME·MEMIT·EWC·Biderman 외 2024·`How much do LMs memorize`(2505.24832)는 있고, SEAL·Nested Learning·`LM Need Sleep`·SCM·Generative Adapter·rate–distortion은 전부 없다. **[Continual Facts §4.3]에서 주기적 오프라인 통합을 직접 구현하면서 CLS도, McClelland도, 생물 sleep 문헌도 인용하지 않는다.** 인용된 심리학은 Tulving & Pearlstone 1966과 McGeoch 1932이고, 논문은 "We use this vocabulary descriptively, without claiming a shared mechanism"이라고 못 박는다 [§8, p.19].

> **[평가]** 이 침묵 지도가 ch11의 발견을 반복한다. 그런데 방향이 반대라 더 무겁다 — 앞의 장들에서는 기제 논문이 서로를 안 봤고, 여기서는 **반증 논문이 반증 대상을 안 봤다.** 그러므로 이 논문의 결론은 $\Theta$-경로 논문들이 제안한 절차에 대한 판정이 아니라, 그 절차들이 공유하는 **연산자**에 대한 판정이다. 그 구분을 ch27이 유지해야 한다.
>
> **그리고 C6을 공급한 두 논문은 또 하나의 서로 모르는 공동체다.** [Spectral Collapse]에서 sleep·offline·consolidat·replay·continual·test-time·forgetting이 전부 0회이고, LoRA 원 논문조차 참고문헌에 없으며, EWC·CLS·RAG·MemGPT·SEAL도 없다. 즉 **반복 가중치 쓰기에 대해 쓰면서 이 장의 두 계보(반증 축과 $\Theta$-경로 옛 계보) 어느 쪽과도 참고문헌이 서로소인 세 번째 공동체다.** [Lifelong Normalization]은 침묵하지 않는다는 점에서 다르되 더 나쁘다 — $E$-경로를 명시적으로 인용하고 "growing knowledge implies an ever-larger memory footprint and roughly linear growth in inference latency"라며 명시적으로 기각하는데, **그 두 축(메모리 발자국, 추론 지연) 어느 쪽도 그 논문에서 측정되지 않는다** [Lifelong Normalization §6, p.9]. 침묵 지도는 부재만 기록하므로 이 행을 담지 못한다. 이 책은 그것을 **측정 없는 대립**이라는 별도 범주로 세고, 그 계수를 ch24가 받는다.

다음 장은 방향을 다시 돌린다. 이 논문이 오프라인 통합을 구현하면서 생물학 문헌을 한 편도 보지 않았다는 사실은, 이 라인 전체가 은유는 빌려 쓰고 원본은 읽지 않았다는 ch08의 관찰과 같다. ch23은 은유의 원본으로 돌아가 **생물학이 실제로 무엇을 보였는지**를 확인한다 — sleep이 망각을 막는다는 실증이 어떤 baseline 위에서 성립하고, dreaming의 표현학습 역할이 어느 통제 대비 유의했는지.

## 요약

- [Continual Facts]는 sleep-time compute 문헌의 어떤 open question도 상속하지 않는다. 명시된 상속은 Shenfeld 외 2025(KL이 망각을 예측)와 Hase 외 2023(국소화가 편집 지점을 지정하지 않는다) 둘뿐이고, 공백 진술은 스스로 만들었다. **가장 강한 반증 후보가 반증 대상을 읽지 않고 쓰였다.**
- 기제는 순수 (U-$\Theta$) 루프이며 표준형과 다섯 곳에서 갈라진다 — 라운드마다 수렴까지 최적화 후 merge, rank 제약, $\mathrm{gen}(\cdot;B_s)$의 조작, 현재 $\Theta_k$를 anchor로 하는 근접항, 20라운드마다 $\Theta_0$로 리셋하는 둘째 시간척도. 그리고 $W=\varnothing$이다 — 따라서 "weights are the wrong system of record"는 $\Theta$에 대한 판정이며 fast weights로 이월되지 않는다.
- 비용 4종에서 $L_w$와 $C_{\text{cap}}$이 **논문에 없음**이고, $B_s$는 optimizer step 수로만 있으며 논문 스스로 "matched on steps, not tokens or examples"라고 적는다. 상각식 (A)를 세울 수 없다.
- 데이터의 폭이 생성과 생존을 모두 정한다(96 step strict에서 +10.1pp; $k=20$에서 46% 대 1%). 그러나 study set이 설계된 근거였던 use-bearing 예시는 단순 다양 반복보다 **8.7pp 낮다**(95% CI [−15.7, −1.9]) — 인과 변수는 프롬프트의 폭이지 명시적 추론 단계가 아니다.
- 망각은 소거가 아니라 접근 상실이다. 모든 strict 문항을 틀린 사실도 log-probability 상승분의 중앙값 69% / 79%를 유지하며, bare 조건 실패의 70%가 가장 최근에 쓴 사실의 내용으로 흘러간다. 두 사실의 합성은 가중치 32% 대 프롬프트 91%로 무너진다.
- 간섭의 소재는 저장 방식이 아니라 **들어오는 쓰기**다(이후 쓰기 효과 +37.6pp 대 저장 방식 −2.4pp). $\Theta_0$ 고정 교사와 근접항이 역량을 되살리지만 도달성은 어느 처방으로도 회복되지 않았다.
- **열화는 「반복한다」의 성질이 아니라 갱신 규칙의 성질이다.** 2026-05·06의 세 논문이 같은 축에 정렬한다 — 자기생성 루프면 $K=3$에서 여덟 구성 중 일곱이 무너지고, 스트림이 밖에서 오고 갱신에 수축이 걸리면 20,000 라운드까지 버티며, 수축의 형태는 정규화(LN)와 부분공간 사영(REVIVE) 둘 다 가능하다. 그러므로 이 장의 판정은 "쌓을 수 없다"가 아니라 **"기하를 통제하지 않으면 쌓을 수 없다"**이고, 살아남기 위한 조건에 **C6 — 업데이트의 기하를 통제해야 한다**가 추가된다. 단 20,000 라운드를 버틴 논문은 **이전 라운드에 쓴 항목의 유지를 재지 않으므로**, 그 결과는 역량 실패에 대한 진술이지 도달 실패에 대한 진술이 아니다.
- 반증 조건 셋 중 (i) 생성 실패와 (ii) 역량 실패는 이제 처방이 있는 실패다 — 전자는 데이터의 폭, 후자는 기하 통제이며 후자의 벽시계 대가는 100 편집당 +4.05%/+1.91%다. **(iii) 도달 실패만 어떤 처방으로도 움직이지 않았고, 그것이 이 장이 남기는 유일한 살아 있는 반증자다.** KL이 조종간이 아니라는 판정은 KL에 대해서만 유지되고, 역량 실패 전반으로 확장되지 않는다.
- 이송된 불리 사실을 유형으로 묶으면 두 줄이 남는다 — 망각 대응의 원형들이 재현 가능한 수치를 거의 남기지 않았고, 압축 이론의 유일한 자체 실험에서 어느 방법도 random-eviction을 유의하게 이기지 못했다. $\Theta$-경로가 기대는 두 축이 둘 다 실증으로 서 있지 않다.
- 실증 상한은 확증 결과 전부 Qwen3-4B이고 8B는 한 측정의 복제 1회다. 문맥 채널은 $|E|\le 2$에서만 돌았고 가중치 채널은 100개까지 밀렸으므로, "context wins"는 실행된 적 없는 레짐으로의 외삽이다.

## 자가 점검 체크리스트

- [ ] $\Theta$-경로의 반증 조건 셋(생성·역량·도달)을 진술하고, 각각에 대해 이 논문이 내린 판정과 그 근거 수치를 댈 수 있다.
- [ ] 지속 주입의 실패 양상 네 가지를 구분하고, 어느 처방이 어느 양상만 고치는지를 frozen-anchor consolidation의 결과(역량 +12pp, 유지율 25% 대 28%)로 설명할 수 있다.
- [ ] use-bearing 예시가 −8.7pp였다는 사실이 SEAL과 Dreaming의 설계 전제에 무엇을 요구하는지 말할 수 있다.
- [ ] 이 논문의 비용 4종 중 어느 칸이 왜 비어 있는지, 그리고 그 결측이 $\Theta$·$E$ 어느 쪽에 유리하게 작동하는지 양방향으로 설명할 수 있다.
- [ ] 용량·망각·붕괴 결합 제약에서 100회 실험의 고원(25–28%)을 만든 것이 어느 축인지 나머지 둘을 배제하는 논증으로 지목할 수 있다.
- [ ] 「반복하면 무너진다」와 「자기생성이면 무너진다」를 가르는 관측을 세 논문의 배치로 재현하고, 20,000 라운드를 버틴 결과가 왜 도달 실패에 대한 증거가 아닌지 설명할 수 있다.
- [ ] 조건 C6이 C2·C3과 다른 좌표에 붙는 이유를 말하고, 그 조건을 검사하는 대조(자기 생성 $\mathcal{R}_k$ 위에서 기하 통제 on/off)를 설계할 수 있다.
- [ ] 이송 표의 열한 가지 결함 유형 중 최소 넷을 대표 사례와 함께 재현하고, 그중 어느 둘이 $\Theta$-경로의 전제를 직접 무너뜨리는지 말할 수 있다.
- [ ] merge된 $\Theta$의 사실당 한계 decode 비용이 0이라는 사실을 Rosetta 사전의 "모델 재배포 ↔ (U-$\Theta$) 1회"·"batch로 weight 공유 ↔ $\Theta$-경로에서 깨짐" 두 행으로 옮기고, 후자가 이 논문의 설정에서 **발동하지 않는** 이유를 설명할 수 있다.
