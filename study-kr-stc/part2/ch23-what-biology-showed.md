# ch23. 생물학이 실제로 보인 것

## 23.1 Bridge-in: 전작이 남긴 문제

이 장은 Part II에서 유일하게 **뒤를 향해** 열린다. ch12부터 ch22까지 열한 장이 세 경로를 훑는 동안 생물학은 계속 배경에 있었다. ch08이 CLS와 생물학적 sleep을 배경으로 세웠고, 그 위에서 ch17·ch21은 wake/sleep lifecycle을 자기 기제의 이름으로 썼으며, ch14는 같은 단어를 전혀 다른 뜻으로 썼다. 이제 그 배경 문헌을 논문장의 엄밀함으로 다시 읽는다. 물음은 하나다 — **이 두 편은 정확히 무엇의 증거인가.**

전작의 open question을 복원하는 이 절의 규약(→ ch11)을 두 논문에 적용하면 결과가 갈린다.

PLOS 논문(Golden·Delanois·Sanda·Bazhenov, *Sleep prevents catastrophic forgetting in spiking neural networks by forming a joint synaptic weight representation*, PLoS Comput Biol 18(11):e1010628, 2022)은 상속을 **명시한다**. 문장은 이것이다.

> "While proposed approaches are capable of mitigating catastrophic forgetting in certain circumstances, a general solution which can achieve human level performance for continual learning is still an open question [9]." [PLOS Sleep Introduction, L103-105]

그리고 자기 작업 질문을 한 문장으로 적는다: "Can we implement a sleep like phase to our model to protect an old task and still accomplish new task learning without explicit re-training of the old task?" [PLOS Sleep Results, L290-292].

그런데 상속의 성격을 밝혀야 한다. 인용된 [9]는 Hayes·Krishnan·Bazhenov·Siegelmann·Sejnowski·Kanan의 replay 리뷰(*Neural Computation* 2021)이고, **이 논문의 교신저자가 그 리뷰의 공저자다** [PLOS Sleep References, L1257-1260]. 이차 상속으로 지목된 Gonzalez et al. 2020 (*eLife*)도 같은 그룹의 논문이다 [ref 41, L1438-1441]. 공백을 선언한 문헌과 그 공백을 메우는 문헌이 저자를 공유한다.

PAD 논문(Deperrois·Petrovici·Senn·Jordan, *Learning cortical representations through perturbed and adversarial dreaming*, eLife 2022;0:e76384)에는 상속 진술이 **없다**. 선행 저자가 적은 open question을 한 문장도 인용하지 않으며, 계보 진술은 전부 저자 자신이 진단한 선행 문헌의 한계다 — Wake-Sleep 계열은 "do not leverage offline states to improve their generative model but are explicitly trained to reproduce sensory inputs during wakefulness" [PAD Introduction, L82-86], trace transformation 이론은 "lack a mechanistic implementation compatible with cortical structures" [L87-89], 기존 dreaming 이론들은 표현 형성이라는 더 기본적인 기능을 고려하지 않는다 [L104-105]. 그러므로 PAD에 대해서는 bridge-in을 **공동체의 공백**에서 연다: CLS 전통은 sleep이 일어난 일을 다시 돈다고 말하고, dream 현상학은 REM의 꿈이 일어난 일이 아니라고 말한다. PAD는 두 번째 사실을 잡음이 아니라 계산의 원료로 삼은 첫 기능 모델이다.

> **[평가]** ch21에서 상속이 교과서적으로 성립한 이유를 이 책은 저자군의 동일성으로 설명했다. 생물학 쪽도 같다 — 두 편 중 하나는 자기 그룹의 리뷰에서 문제를 받았고 다른 하나는 아무에게서도 받지 않았다. corpus에서 상속 진술이 성립하는 자리는 대체로 **경로가 이어진 자리가 아니라 연구 그룹이 이어진 자리**다. 흠이 아니라 전파 방식에 대한 사실이며, §23.8의 반대 방향 침묵과 짝을 이룬다.

---

## 23.2 문제의식

두 논문이 스스로 세운 문제는 다르다. 그리고 둘 다 이 책의 corpus가 나중에 쓰게 될 문제와도 다르다.

**PLOS의 문제는 저장이다.** 순차 학습이 옛 과제의 가중치를 덮어쓴다는 진단에서 출발하되, 고전적 처방인 interleaved training이 "imposes the stringent constraint that the original training data be perpetually stored for later use and combined with new data to retrain the network"라는 점을 문제의 본체로 잡는다 [PLOS Sleep Introduction, L109-111]. 그래서 논문이 한 문장으로 적은 과제는 정확도가 아니다: "Thus, the challenge is to understand how the biological brain enables memory reactivation during sleep **without access to past training data**" [L111-113, 강조는 이 책]. 이 문장이 이 논문의 모든 결과의 해석 범위를 정한다.

**PAD의 문제는 표현이다.** 라벨 없는 감각 경험에서 어떻게 선형 분리 가능하고 섭동에 강건한 표현이 생기는가, 그리고 두 sleep 상태가 각각 무엇을 기여하는가 [PAD Abstract, L11-14]. 가설은 dream이 재생이 아니라 생성이라는 데 있다: "generating new but realistic sensory experiences, instead of merely reconstructing previous observations, requires the brain to understand the composition of its sensorium" [PAD Introduction, L113-115].

여기서 이 장이 소유하는 첫 정의를 못 박는다.

> **정의.** **생물학 실증의 정확한 범위**란, 한 생물학·계산신경과학 논문이 실제로 뒷받침하는 진술들의 집합을 네 좌표로 닫아 적은 것이다 — **(기제 클래스, 규모, 측정 지표, 반복 라운드 수)**. 네 좌표 중 하나라도 밖으로 나간 진술은 그 논문이 지지하지 않는다. 예를 들어 "sleep이 망각을 막는다"는 범위 밖 진술이고, "국소 spike-timing 가소성과 열 합 보존 정규화를 갖춘 3층 spiking 망에서, 6272개 가소성 시냅스와 두 개의 상보 과제에 대해, pattern discriminability 지표로, 한 번의 interleaved 위상 동안, 옛 과제 성능이 0.70 ± 0.02에서 0.70 ± 0.03으로 유지된다"는 범위 안 진술이다.

이 정의는 사후 트집이 아니라 **인용 가능성의 계약**이다. §23.6이 두 논문의 네 좌표를 채우고, §23.8이 그 좌표 밖으로 나간 인용을 목록으로 잘라낸다.

> **[해설]** 계약이 필요한 이유는 corpus의 인용 패턴에 있다. LLM 쪽 논문이 생물학을 인용할 때 옮겨지는 것은 거의 언제나 **결론 문장**이지 좌표가 아니다. "sleep은 catastrophic forgetting을 막는다"는 한 줄은 네 좌표를 전부 지운 상태로 이송된다. 지워진 좌표를 복원하는 것이 이 장의 실용적 산출물이다.

---

## 23.3 Core mechanism (통일 표기)

두 논문의 기제는 ch08 §08.3–§08.4가 서술했다. 이 절은 그것을 반복하지 않고 **표준형과의 차이**만 적는다.

### 23.3.1 PLOS — (U-$\Theta$)를 두 시계로 두 번

PLOS 논문은 **수식에 번호를 붙이지 않는다.** 따라서 원문 인용은 Methods의 절 이름과 vendored 원문 줄번호로만 한다. 이 책은 이 논문에 원문 수식 번호를 발명해 붙이지 않는다.

읽기는 식 (R)의 극단적 붕괴다.

$$
\hat y_t \;=\; f\big(q_t;\ \Theta,\ \varnothing,\ \varnothing\big)
\;=\;\arg\max_{o}\ \#\mathrm{sp}_o\Big(q_t;\ \Theta^{I\to H},\ \Theta^{H\to O}_k,\ \Theta^{H\to O,-}_k\Big)
\tag{23-1}
$$

$q_t$는 movement cycle $t$의 7×7 이진 시야이고 $\#\mathrm{sp}_o$는 출력 뉴런 $o$의 전반부 스파이크 수다 [PLOS Sleep Methods: Policy, L784-789]. **$W = \varnothing$이고 $E = \varnothing$이다** — movement cycle 사이에 넘어가는 상태는 가중치와 항상성 스칼라뿐이다. 이 논문에는 fast weight도 external store도 없다.

쓰기는 (U-$\Theta$) 하나뿐인데, **wake와 sleep이 같은 가중치 블록에 같은 국소 규칙을 쓴다.** 갱신 대상은 은닉→출력 흥분성 시냅스 $\Theta^{H\to O}$ 6272개다 [L995, L1035]. 표준형과의 차이는 네 가지다.

첫째, **gradient가 없다.** $\nabla_\Theta \mathcal{L}$ 자리에 국소 eligibility trace를 낀 세 인자 Hebbian 곱이 들어간다 [Methods: Synaptic plasticity, L879-901]. loss 함수도, 도함수도, optimizer도, backward pass도 논문 전체에 없다. $\eta_\Theta$는 독립된 존재조차 아니며 보상 $R$, trace 최대치 $K = \pm 0.04$, 항상성 보폭 $\Delta_{\mathrm{tar}}$에 흡수되어 있다.

둘째, **갱신이 사영된다.** 매 STDP 사건 후 heterosynaptic 정규화가 각 출력 뉴런의 입력 합을 고정값으로 되돌린다 [L946-963]. 제약 집합은

$$
\sum_i \Theta^{H\to O}_{ij} \;=\; \Theta^{0}_{\cdot j}\quad \text{(모든 } j\text{)}
\tag{23-2}
$$

이고, 이 사영을 이 책은 $\mathrm{nrm}(\cdot)$로 쓴다(장-국소 기호). **이것이 논문에서 가장 하중을 받는 조각이다.** 열 합이 고정되므로 모든 강화는 같은 열 안의 다른 시냅스의 약화로 지불된다 — 갱신이 제로섬이다. sleep이 단순 동결보다 나은 이유에 대한 논문 자신의 설명이 여기에 기댄다: "the sleep phase results in a large cluster of weights being renormalized around an intermediate value of synaptic strength in the network" [L511-512].

셋째, sleep 라운드의 $\mathcal{R}_k$가 데이터셋이 아니다. 감각 수용기를 끄고, 환경에서 입자를 모두 없애고, 출력층을 운동 제어에서 분리하고, 은닉층을 Poisson 잡음으로 구동한 채 망 자신을 돌린다 [Methods: Simulated sleep, L966-985]. 즉 $\mathrm{gen}(\mathcal{H}_k; B_s)$가 문자 그대로 "$B_s$ movement cycle 동안 잡음 위에서 망을 전진시켜라"이고, **$\mathcal{H}_k$는 $\Theta_k$ 안 말고는 어디에도 저장되지 않는다.**

$$
\Theta^{H\to O}_{k+1} \;=\; \mathrm{nrm}\Big[\ \mathrm{hebb}\big(\Theta^{H\to O}_{k};\ R \equiv 0.5,\ q \equiv \varnothing,\ H \sim \mathrm{Poisson}(\bar f)\big)\Big]
\tag{23-3}
$$

넷째, **wake도 (U-$\Theta$)다.** 배포 가중치가 각성 중에도 계속 움직인다. 이 구조적 사실이 §23.7의 serving 논의를 지배한다.

논문 안의 어휘 불일치도 그대로 옮긴다. Results는 sleep의 규칙을 "the rewarded STDP rule was replaced by unsupervised STDP"라고 쓰고 [L294-295], Methods는 "During periods of sleep the network received a constant reward of Srp = 0.5 on each movement cycle"이라고 쓴다 [L936-937]. 앞 문장만 읽은 독자는 sleep 중 모든 trace에 양의 스칼라가 곱해진다는 것을 예상하지 못한다.

### 23.3.2 PAD — (U-E) 한 번 뒤 (U-$\Theta$) 세 번

PAD는 한 사이클 $k$가 mini-batch 하나이고($b = 64$) [PAD Methods, L1283], 순서는 wake → NREM → REM이며 위상마다 outer loss가 다르다. wake의 쓰기는 (U-E)의 가장 값싼 사례다.

$$
E_k \;=\; \{\hat c_k,\ \hat c_{k-1}\},\qquad
\hat c_k \;=\; S(c_k; B_s) \;=\; \mathrm{enc}\big(x_k;\Theta^{\mathrm{enc}}\big)
\tag{23-4}
$$

$S(\cdot; B_s)$가 encoder forward 한 번이고 $\mathrm{wr}$은 2슬롯 shift register이며 **$\mathrm{ret}(E,q)$가 존재하지 않는다** — 읽기는 질의가 아니라 라운드 첨자로만 이루어진다 [PAD Algorithm 1, L1300, L1351, L1423].

세 번의 (U-$\Theta$)는 위상별로 $\mathcal{L}$과 $\mathcal{R}_k$가 갈린다. NREM은 저장된 latent를 **목표**로, generator 출력에 가림을 적용한 것을 **입력**으로 삼는다: $\mathcal{R}_k = \{(\mathrm{occ}(\mathrm{gen}_{\mathrm{pix}}(\hat c)),\ \hat c)\}$ [PAD Eq. 6]. store가 입력을 주고 동결 teacher가 목표를 주는 일반적 replay 도식과 **입출력이 뒤집혀 있다**(→ ch07). REM은 두 슬롯과 잡음을 섞은 latent를 렌더한다.

$$
z^{\mathrm{mix}}_k \;=\; \tfrac14\,\hat c_k \;+\; \tfrac14\,\hat c_{k-1} \;+\; \tfrac12\,\varepsilon,
\qquad \varepsilon \sim \mathcal{N}(0, I)
\tag{23-5}
$$

계수는 $\lambda = \lambda' = 0.5$에서 나온다 [PAD Methods, L1556-1563; Figure 2c, L213-224]. **혼합의 절반이 잡음이다.**

표준형이 표현하지 못하는 것은 REM의 부호다.

$$
\Theta^{\mathrm{disc}}_{k+1} = \Theta^{\mathrm{disc}}_{k} - \eta_\Theta \nabla \mathcal{L}_{\mathrm{rem}},
\qquad
\Theta^{\mathrm{gen}}_{k+1} = \Theta^{\mathrm{gen}}_{k} \;\mathbf{+}\; \eta_\Theta \nabla \mathcal{L}_{\mathrm{rem}}
\tag{23-6}
$$

같은 loss에 대해 한 파라미터 블록만 상승한다 [PAD Algorithm 1, L1409, L1421]. 논문은 이 부호 전환을 자기 생물학적 예측으로 세운다: "Sign switch indicates that identical local errors lead to opposing weight changes between Wake and REM sleep" [PAD Figure 7 caption, L814-815]. 표준형 (U-$\Theta$)는 부호 필드 없이는 이 형태를 담지 못한다.

$W$층은 없다. 평가 시점에 움직이는 것이 아무것도 없다 — $\hat y = W_{\mathrm{ro}}\,\mathrm{enc}(x;\Theta^{\mathrm{enc}})$ 한 번의 forward가 전부이고, 분류기를 학습할 때 "the connectivity of the network (E and G) is fixed" [PAD Results, L531-532].

### 23.3.3 표기 대응표

표 23-1. PLOS Sleep의 원 논문 기호 → 이 책 표기. 이 논문은 수식 번호가 없으므로 출처는 Methods 절 이름으로 준다.

| 원 논문 기호 | 이 책 표기 | 주의 |
|---|---|---|
| $W_{ij}, W_i, W_j, W_{i0}, W_{j0}, WI_{ij}$ (시냅스 가중치) | $\Theta^{I\to H}$, $\Theta^{H\to O}$, $\Theta^{H\to O,-}$ | $W$는 fast weights 예약. 이 논문에 $W$층도 $E$층도 없다 |
| $S_{rp}$ (보상·처벌 배율) | $R$ | 본서 표기 규약상 첨자 없는 맨 $S$는 금지이고, $r$은 LoRA rank이므로 보상은 $R$이다 — 장-국소 정정 |
| $k$ (trace 합의 첨자) | $m$ | $k$는 sleep 라운드 전용 |
| $R = 0.12$ (시냅스 방출 변동계수) | $r_{\mathrm{rel}}$ | 본서 수식에 등장하지 않음 |
| $H$ (은닉층) | 산문의 "layer H" / $\mathcal{H}_k$(누적 경험) | 두 대상이 개념상 거의 같아 충돌이 위험하다. 한 수식에 함께 두지 않는다 |
| $M_{T1}, M_{T2}, M_{T1\cap T2}$ (해 manifold) | $\mathrm{Man}_{T1}$ 등 | 필기체 $\mathcal{M}$은 memory 읽기 함수 전용 |
| $\alpha = 3.65$ (뉴런 비선형성) | $\alpha_{\mathrm{neu}}$ | $\alpha_t$는 retention gate |
| $\beta_n, \sigma_n$ (외부 전류 변수) | $\beta_{\mathrm{neu}}$ | $\beta_t$는 momentum decay |
| $c = 1$ movement cycle (trace 감쇠 상수) | $t_{\mathrm{off}}$ | $C$는 chunk, $C_{\text{cap}}$은 용량 |
| $n$ (0.5 ms 시간 첨자 **및** 가중치 공간 차원 6272) | 시간 첨자 $s$ / 차원 $d_\Theta = 6272$ | 논문 자체가 $n$을 두 뜻으로 쓴다 |
| epoch(= 600 timestep ≈ 300 ms), aeon(= 100 epoch) | "movement cycle", "aeon(= 100 movement cycle)" | ML의 epoch와 무관. 수식어 없는 "epoch" 사용 금지 |
| $I$ (입력층 / 느린 뉴런 변수 / 시냅스 전류) | 산문의 "layer I" / $u_s$ | 논문 내부 삼중 사용 |
| $T_c, K, \gamma, \mu, \delta, \Delta_{\mathrm{tar}}$ | 그대로 인용만 | 본서 수식에 등장하지 않음. $\delta, \Delta_{\mathrm{tar}}$는 vendored 원문에 수치가 없다 |
| (노트 내부 기호) 정규화 사영 | $\mathrm{nrm}(\cdot)$ | $\Pi$는 지속학습 성능 행렬 전용 — 장-국소 정정 |

표 23-2. PAD의 원 논문 기호 → 이 책 표기.

| 원 논문 기호 | 이 책 표기 | 주의 |
|---|---|---|
| $E, E_z, E_d$ (encoder·latent·discriminator 사상) | $\Theta^{\mathrm{enc}}, \Theta^{\mathrm{disc}}$, $\mathrm{enc}(\cdot)$, $\mathrm{disc}(\cdot)$ | **corpus 최대 위험 충돌.** 논문의 $E$는 slow-weight 망이고 이 책의 $E$는 external store다. 예약 $E$는 논문의 hippocampal 버퍼에 준다 |
| $G$ (generator·feedback 경로) | $\Theta^{\mathrm{gen}}$, $\mathrm{gen}_{\mathrm{pix}}(\cdot)$ | 이 망이 문자 그대로 (U-$\Theta$)의 $\mathrm{gen}(\cdot)$이다 |
| $W \in \mathbb{R}^{10\times 256}$ (선형 readout) | $W_{\mathrm{ro}}$ | fast weight가 **아니다**. 동결 후 학습되는 평가 probe |
| $z, z_{old}, Z, Z_{old}, Z_{mix}, z'$ | $z$ 유지; 저장된 것은 $\hat c_k, \hat c_{k-1}$ | 장-국소 선언 |
| $L_{img}, L_{KL}, L_{real}, L_{Wake}, L_{NREM}, L_{REM}, L_{GAN}, L_C$ | $\mathcal{L}_{\mathrm{img}}$ 등 전부 필기체 | 맨 $L$은 sequence 길이. 이 논문에는 inner loss $\ell$이 없다 |
| $\theta_E, \theta_G$ | $\Theta^{\mathrm{enc}}, \Theta^{\mathrm{disc}}, \Theta^{\mathrm{gen}}$ | 모두 $\Theta_k$의 부분 블록 |
| ADAM lr $= 0.0002$ | $\eta_\Theta = 0.0002$ | $\eta_t$는 없다. **세 위상이 같은 값을 쓴다** |
| $\eta = 0.2$ (readout 학습률) | $\eta_{\mathrm{ro}}$ | 평가 probe 소유 |
| $\beta_1 = 0.5, \beta_2 = 0.999$ | $\beta_1^{\mathrm{Adam}}, \beta_2^{\mathrm{Adam}}$ | 시간 첨자를 붙이지 않는다 |
| $\lambda = \lambda' = 0.5$ (혼합 계수) | 그대로, 장-국소 상수 | retention gate가 아니다. 상태 감쇠가 아니라 기록 두 개의 혼합비 |
| $b = 64$ (mini-batch) | $b$ | $C$(chunk)로 쓰지 않는다 |
| $d$ (discriminator 스칼라 출력) | $\mathrm{disc}(x)$ | $d$는 모델 폭 |
| $\epsilon \sim \mathcal{N}(0,I)$; 절제용 $\epsilon \sim \mathcal{N}(0,0.5I)$ | $\varepsilon$, 장-국소 | 두 $\varepsilon$은 분산도 역할도 다르다 |
| $\omega, \Omega$ (가림 마스크) | $\mathrm{occ}(\cdot)$ | $\omega$는 replay 혼합비 예약(ch07) — 장-국소 정정 |
| $n_z = 256$ (latent 차원) | $n_z$ | $n$은 저장 항목 수 |
| epoch(= 데이터셋 1회 통과) | $k$ = wake-sleep 사이클(= mini-batch 1개) | **한 epoch은 많은 $k$다.** 혼동하면 sleep 예산이 세 자릿수만큼 작아 보인다 |

---

## 23.4 어느 층을 언제 쓰는가

표 23-3. PLOS Sleep에서 움직이는 모든 것 [Methods: Network structure L742-758; Synaptic plasticity L861-963; Simulated sleep L965-986].

| 대상 | 기호 | 크기 | 누가 움직이나 | 언제 |
|---|---|---|---|---|
| 입력→은닉 흥분성 시냅스 | $\Theta^{I\to H}$ | 784 × 9 = 7056 | 비지도 STDP | 초기 비지도 기간에만. 강화학습 기간 내내 **동결** [L205-207] |
| 은닉→출력 흥분성 시냅스 | $\Theta^{H\to O}$ | 784 × 8 = 6272 | wake는 보상 STDP, sleep은 $R \equiv 0.5$의 같은 규칙, 양쪽 다 매 사건 후 $\mathrm{nrm}$ | **wake와 sleep 둘 다.** 명목 속도도 같다 |
| 은닉→출력 억제성 시냅스 | $\Theta^{H\to O,-}$ | 6272, 자유도 0 | 아무도. 원천 뉴런의 흥분성 출력 평균의 음수로 **재계산** | 매 STDP 사건 후, wake·sleep 동일 |
| 출력 뉴런별 목표 입력 합 | $\Theta^0_{\cdot j}$ | 스칼라 8개 | 발화율 기반 항상성 scaling | 매 movement cycle |
| 은닉 뉴런별 목표 출력 합 | $\Theta^0_{i\cdot}$ | 스칼라 784개 | 아무도 — 시작값으로 고정 | 없음 |
| trace 합의 이동 평균 | $\mathrm{Avg_{tr}}$ | 스칼라 | 지수 평활 | trace 평가 시마다 |
| sleep 구동용 목표 발화율 | $\bar f_i$ | 784개 (통제에서 1개) | 학습되지 않음 — **wake에서 측정** | 직전 wake에서 계산, sleep 중 고정 |
| 모델 하이퍼파라미터 | — | 약 15개 | 손으로 설정. outer loop 없음, sweep 보고 없음 | 없음 |
| SVM(RBF)·PCA/kPCA | — | 입력 6272차원 | 저자가 **사후 분석 도구로** 적합 | 종료 후. 망은 한 번도 보지 않는다 |

표 23-4. PAD에서 움직이는 모든 것 [Algorithm 1, L1287-1423; Methods L1247-1663].

| 대상 | 기호 | 누가 움직이나 | 언제 |
|---|---|---|---|
| encoder(conv 4층) | $\Theta^{\mathrm{enc}}$ | outer-loop gradient descent, ADAM | **세 위상 전부**. wake는 $\mathcal{L}_{\mathrm{img}}+\mathcal{L}_{\mathrm{KL}}+\mathcal{L}_{\mathrm{real}}$, NREM은 $\mathcal{L}_{\mathrm{nrem}}$, REM은 $\mathcal{L}_{\mathrm{rem}}$ |
| discriminator head | $\Theta^{\mathrm{disc}}$ | 같은 optimizer. encoder와 앞 3층 공유 | wake(목표 1)와 REM(목표 0). NREM에는 **닫혀 있다** [L866-868] |
| generator(deconv 4층) | $\Theta^{\mathrm{gen}}$ | wake는 $\mathcal{L}_{\mathrm{img}}$에 하강, REM은 $\mathcal{L}_{\mathrm{rem}}$에 **상승** | wake와 REM. NREM에는 갱신 줄이 없다 |
| store 현재 슬롯 | $E$의 $\hat c_k$ | 아무도 — 학습이 아니라 **복사** | wake. NREM에서 목표로, REM에서 혼합 항으로 읽힘 |
| store 이전 슬롯 | $E$의 $\hat c_{k-1}$ | 아무도 — REM 끝에서 밀려 들어감 | 보존 **정확히 한 사이클** |
| REM 잡음 | $\varepsilon$ | 아무도 — 매 REM 재표집 | REM만. 혼합의 절반 |
| 가림 마스크 | $\mathrm{occ}(\cdot)$ | 아무도 — mini-batch마다 확률 0–1, 사각 크기 1–8 표집 [L1627-1628] | NREM만 |
| 혼합 계수 | $\lambda, \lambda'$ | 아무도 — 고정, 변주되지 않음 | REM만 |
| 선형 readout | $W_{\mathrm{ro}}$ | 라벨을 쓰는 지도 SGD | **모델 동결 후.** 모델의 일부가 아니다 |
| discriminator 교시 신호 | 파라미터화되지 않음 | 아무도 — 구현에서는 상수 1과 0. 논문은 **가정된 뇌 상태 신호**라고 밝힌다 [L855-856] | wake와 REM |

두 논문의 시계와 순서를 명시한다. PLOS에서 sleep 라운드 첨자 $k$는 **한 번의 interleaved 주기**다 — 새 과제 학습 100 movement cycle 뒤 sleep 100 movement cycle이고, duty ratio가 시뮬레이션 시간 기준 정확히 1:1이며, 두 블록이 같은 파라미터를 쓴다 [Results L300-306; Methods: Simulated sleep L985-986]. PAD에서 $k$는 **mini-batch 하나**이고, 그 안에서 (U-E) 한 번과 (U-$\Theta$) 세 번이 고정 순서로 돌되 위상마다 갱신 블록이 다르다. 논문의 x축은 epoch 단위이므로 $k$와 epoch를 섞으면 오프라인 예산이 세 자릿수만큼 작아 보인다[본서 추론]. 두 편 모두 오프라인 위상이 하나가 아니라는 ch08 §08.2의 판별기를 만족한다 — PLOS는 두 목적, PAD는 세 목적이다.

> **[해설]** 독자의 1번 질문에 대한 이 절의 답은 짧다. **어느 쪽에서도 decode가 바뀌지 않는다.** PLOS의 결정 한 번은 고정된 3층 전방 경로를 600 timestep 도는 것이고, sleep은 $\Theta^{H\to O}$의 **값만** 바꾼다 [Methods: Policy L769-789; Simulated sleep L978-980]. PAD의 평가는 동결된 encoder의 forward 한 번이다. 즉 두 편 모두 상각 논증이 가장 좋아하는 모양인데, **양쪽 다 어느 쪽 비용도 재지 않는다.**

---

## 23.5 비용 4종

표 23-5. 비용 4종. "논문에 없음"은 추정으로 메우지 않는다.

| | PLOS Sleep | PAD |
|---|---|---|
| $B_s$ | **부분.** 계산량이 아니라 시뮬레이션 시간으로만 존재한다. 한 sleep 구간 = 100 movement cycle = 60,000 timestep, timestep당 0.5 ms이므로 시뮬레이션 시간 30초[본서 산술] [L769, L807, L985-986]. duty ratio 1:1 [L301-303]. **FLOPs·wall-clock·CPU 시간·에너지 전부 논문에 없음.** 길이·비율·교대 횟수 sweep 없음 | **논문에 없음.** FLOPs·GPU 시간·wall-clock 어느 것도 없다. 구조적으로 확인되는 것은 사이클당 (U-$\Theta$) 세 번 중 두 번이 오프라인이라는 사실뿐 |
| $L_w$ | **논문에 없음.** 지연·처리량·결정당 비용이 어디에도 없다 | **논문에 없음** |
| $C_{\text{cap}}$ | **바이트로는 논문에 없음**(dtype 없음). 개수로만: 연속 가소성 시냅스 6272 [L995, L1035], 한 번 학습 후 동결된 입력→은닉 7056 [L751, L205-207], 자유도 0인 억제성 6272 [L956-963], 항상성 스칼라 8 [L938-945], 고정 목표합 784 [L928-930], sleep 구동 발화율 784 → 통제에서 1 [L982-985, L1144-1150]. **과거 과제 데이터 0바이트** [L611, L674-677, L727] | **바이트로는 논문에 없음.** 구조적으로 **2슬롯** 고정, 보존 기간 정확히 한 사이클 — `Z_old ← Z`가 매 사이클 덮어쓴다 [Algorithm 1, L1423]. 누적되는 것이 없다 |
| $\rho$ | **부분.** 간섭 과제 **하나**의 종점만: sequential은 0.70 ± 0.02 → 0.52 ± 0.02, sleep 교대는 0.70 ± 0.02 → 0.70 ± 0.03 [L228-229, L275, L304-305]. 라운드당 감쇠율 없음. 세 번째 과제도 두 번째 interleaved 위상도 없음 | **방향성만.** NREM이 없는 조건에서 "linear separability tends to decrease after many training epochs" [PAD Appendix 1, 'Linear classification performance']. 수치화된 라운드당 손실 없음 |

식 (A)의 손익분기 $N_q^*$는 **두 논문 모두에서 계산할 수 없고, 이유가 서로 다르다.** PLOS에는 질의 스트림도 문맥도 사용자도 없다 — 에이전트는 movement cycle마다 결정 하나를 영원히 낸다. 1:1 duty ratio를 식 (A)에 넣으면 주기당 $C_{\text{sleep}} = C_{\text{wake}}$가 되지만 나눌 $N_q$가 없다. PAD의 평가는 고정된 테스트 집합 위의 선형 probe이므로 **한 문맥을 공유하는 질의**라는 개념 자체가 없다.

> **[평가]** 이 라인의 상각 논증은 생물학에서 오지 않았다. **생물학적 조상 두 편은 상각 증거를 하나도 제공하지 않는다.** 채워진 칸조차 계산량이 아니라 시뮬레이션 시간이거나 파라미터 개수다. 그러므로 ch01 §01.5의 회계 규칙대로, 이 두 편을 근거로 "sleep-time compute이 효율적이다"라고 쓸 수 없다. 이 문헌이 실제로 제공하는 것은 **상태 용량 논증** 하나다 — 과거 과제 데이터 0바이트, 경계를 넘는 보조 상태 $O(1)$ 스칼라(→ ch25, ch26).

---

## 23.6 실험과 스케일

이 절이 §23.2의 정의가 요구하는 네 좌표를 채운다. 수치는 원문 정밀도 그대로 옮기고 반올림하지 않는다.

### 23.6.1 PLOS — 조건 전량

과제는 50×50 격자에서 입자 밀도 10%의 foraging 두 개다 [Methods L731, L739-740]. 입자는 네 방향 중 하나이고, Task 1은 수평을 보상하고 음대각을 처벌하며 Task 2는 수직을 보상하고 양대각을 처벌한다 [Results L225-232; Methods L932-935]. 지표는 pattern discriminability이고 chance가 0.5다 [L166-167]. **모든 수치는 최소 10회 시행의 평균 ± 표준편차이며, vendored 원문 전문에 검정 이름도 p-값도 한 건 없다**[본서 관찰] [L168-169]. 그런데 본문은 "significantly"를 최소 세 번 쓴다 [L418, L444, L506]. 이 책은 두 사실을 함께 적고, 게재본의 그림·보충자료까지 포함한 판정은 유보한다.

표 23-6. PLOS Sleep의 전 조건. 굵은 행이 헤드라인 결과다.

| 조건 | Task 1 | Task 2 | 과거 데이터 | 출처 |
|---|---|---|---|---|
| Task 1 단독 학습 후 | 0.70 ± 0.02 | 0.53 ± 0.02 | — | L228-229 |
| Sequential (T1→T2) | 0.52 ± 0.02 | 0.69 ± 0.03 | 불필요 | L275-276 |
| Interleaved$_{T1,T2}$ (고전 interleaving) | 0.68 ± 0.03 | 0.65 ± 0.04 | **필요** | L282 |
| **Interleaved$_{S,T2}$ (sleep 교대)** | **0.70 ± 0.03** | **0.68 ± 0.05** | 불필요 | L304-305 |
| Uniform-Noise Sleep(집단 평균 1스칼라 구동) | 0.67 ± 0.05 | 0.69 ± 0.03 | 불필요 | L1148-1150 |
| 나이브 망 + Task 2 통계로 만든 sleep 잡음 | 0.60 ± 0.03 | 0.49 ± 0.05 | 불필요 | L1136-1137 |
| Task 1 학습 후 Interleaved$_{S,T1}$ | 0.71 ± 0.02 | 0.51 ± 0.02 | 불필요 | L1141-1143 |
| 상위 1% 시냅스 동결 | 0.54 ± 0.02 | 0.68 ± 0.03 | 불필요 | L1184-1187 |
| 상위 5% 시냅스 동결 | 0.65 ± 0.02 | 0.61 ± 0.01 | 불필요 | L1184-1187 |
| 상위 10% 시냅스 동결 | 0.70 ± 0.03 | 0.53 ± 0.03 | 불필요 | L1184-1187 |

마지막 세 행에는 단서가 붙는다. 동결 baseline은 세 비율만, 그리고 "Task 1 학습 후 크기 상위"라는 단일 선택 기준으로만 시험되었고, 논문이 이 계열의 기계학습 사례로 지목한 EWC [ref 7]는 **이름만 불릴 뿐 한 번도 실행되지 않는다** [L546-547]. 즉 이 비교는 중요도 가중 정규화가 제대로 구현되었을 때의 결과가 아니다[본서 추론].

여섯째·일곱째 행이 sleep이 **하지 않는 일**을 정한다. sleep 잡음에 Task 2의 발화 통계를 실어 넣어도 나이브 망은 Task 2를 배우지 못하고(0.49 ± 0.05), Task 1만 학습한 망에 sleep을 돌려도 Task 2는 baseline에 머문다(0.51 ± 0.02). **오프라인 위상은 학습된 적 없는 지식을 만들어내지 않는다**[본서 추론].

**강한 baseline이 방법과 거의 같다.** 고전 interleaving은 0.68/0.65이고 sleep은 0.70/0.68이다. 차이는 0.02와 0.03이며 표준편차는 0.03–0.05다. 논문은 Results에서 sleep이 고전 interleaving을 "exceeding"한다고 쓰지만 [L305-306], Discussion에서 스스로 정정한다.

> "Although classical interleaved training of the old and new tasks showed similar performance results in our model as interleaving new task training with sleep, **we believe the latter to be superior on the following theoretical grounds.**" [PLOS Sleep Discussion, L654-655, 강조는 이 책]

이어지는 근거는 전부 이론이다 — sleep은 두 과제가 섞인 replay를 지원할 수 있다는 것 [L655-670]. 그런데 replay 내용을 분석한 실험은 이 논문에 없고, 혼합 replay 주장은 인용으로 수입된다 [refs 41, 52, 53].

가중치 상태 판독도 기록한다. 6272차원 가중치 벡터를 Task-1형/Task-2형으로 분류하도록 학습시킨 RBF-SVM의 부호 있는 평균 거리는 Task 1이 −0.069, Task 2가 +0.069, sleep 교대가 **−0.0047**이다 [Fig 4D, L394-396]. 데이터 replay 버전에서는 Interleaved$_{T1,T2}$가 0.016이다 [S5, L1165-1167]. 원문 수치를 단일 과제 크기 0.069로 나누면(이 책의 산술) 전자는 6.8%, 후자는 23%인데 **논문은 두 수치를 직접 비교하지 않는다.**

manifold 결과에는 방법론적 단서가 붙는다. 교집합 점집합 $\mathrm{Man}_{T1\cap T2}$가 "defined solely by the synaptic weight states from the last fifth of both InterleavedT1,T2 and InterleavedS,T1 training"으로 구성되므로 [Methods L1013-1019], **그 점집합으로 수렴하는 것으로 그려지는 궤적 자신이 점집합의 재료다.** 저자도 표본 부족을 밝힌다 [L572-573]. 정확도 결과가 틀렸다는 뜻이 아니라 기하 헤드라인이 독립 측정이 아니라는 뜻이다.

원문 내부 불일치 세 건은 어느 쪽으로도 정리하지 않고 양쪽을 그대로 적는다. (i) 은닉 뉴런 제거 실험의 임계는 본문이 "around 70% of the hidden layer was pruned" [L241], 보충 캡션이 "remains high until ~25% of neurons are left" [L1109-1110]로 어긋난다. (ii) uniform-noise 실험의 이름이 본문에서는 Interleaved$_{US,T1}$ [L359], S4 캡션에서는 Interleaved$_{US,T2}$다 [L1146] — 보고된 수치(Task 1 0.67 ± 0.05, Task 2 0.69 ± 0.03)가 학습된 쪽이 Task 2임을 가린다. (iii) 표준편차 한 개가 "0.70 ± 002"로 오식되어 있다 [L305]. 표 23-6은 L228의 올바른 값 0.70 ± 0.02로 적었다.

논문 자신이 못 박는 적용 범위 제한도 옮긴다. "In our model, and indeed in all neural network models, the system begins as a 'blank slate' without knowledge of any previous learning or competing demands"이며, 그래서 저자는 주장을 "the interference phenomena which follow training on an initial task as opposed to initial learning"으로 좁힌다 [Discussion, L685-697]. 즉 이 논문의 결과는 사전 경험이 있는 시스템에 대한 진술이 아니다.

**규모 천장.** 뉴런 842개(입력 49 · 은닉 784 · 기능적 출력 8) [L743-746], 연속 가소성 시냅스 6272개, 과제 **2개**이며 그 둘은 같은 네 방향 어휘에서 뽑은 대칭 상보 과제다(논문 자신이 유사성을 특징으로 강조한다 [L233-237]). 보호되는 성능 밴드는 0.20 폭이고 조건별 표준편차가 그 밴드의 10–25%다. 과제 환경과 두 평가 지표가 모두 저자 제작이고 표준 지속학습 벤치마크는 하나도 쓰이지 않았으며 [L166-167, L1037-1095], 계산 자원은 보고되지 않았다.

### 23.6.2 PAD — 절제와 통제

설정은 conv 4층 encoder(채널 64·128·256·256), deconv 4층 generator, $n_z = 256$, $b = 64$, ADAM $\eta_\Theta = 0.0002$이고 데이터셋은 CIFAR-10과 SVHN이다 [Methods L1247-1283, L1506]. 지표는 모델을 동결한 뒤 latent 위에 학습시킨 선형 분류기의 테스트 정확도다.

표 23-7. epoch 50 시점의 선형 분리도. 초기값 4종에 대한 평균 ± SEM [PAD Appendix 1—table 1].

| 데이터셋 | PAD | w/o memory mix | w/o REM | w/o NREM | Wake only |
|---|---|---|---|---|---|
| CIFAR-10 | 58.25 ± 0.70 | 53.87 ± 0.85 | 46.00 ± 0.43 | 58.00 ± 0.34 | 42.25 ± 0.54 |
| SVHN | 78.92 ± 0.40 | 60.87 ± 5.07 | 42.30 ± 1.51 | 73.25 ± 0.22 | 41.93 ± 0.65 |

표를 정확히 읽는다. `w/o memory mix`는 혼합만 없앤 조건이 **아니다** — REM을 단일 episodic memory로, 잡음 없이 구동한 조건이다 [PAD Results, L548-549]. CIFAR-10에서 `w/o NREM`은 58.00 ± 0.34로 PAD의 58.25 ± 0.70과 오차 안에서 같다(그 함의는 ch08 §08.4.2가 다뤘다)[본서 관찰]. 절제 대조군이 손봐진 상태로 비교된다는 사실도 함께 옮긴다 — `w/o REM` 모델은 encoder 출력에 $\varepsilon \sim \mathcal{N}(0, 0.5I)$를 더한 변형 재구성 손실로 학습된다 [PAD Eq. 12].

이 장의 나머지 세 의무 caveat가 여기서 나온다.

**첫째, 저자 자신의 통제가 논문 제목의 성분을 지지하지 않는다.** "we do not observe significant differences between using a combination of episodic memories with spontaneous activity or only using spontaneous activity" [PAD Discussion, L1034-1036]. 근거 그림은 세 팔을 비교한다 — 혼합 기억 + 잡음, 순수 잡음만, 혼합 기억만 [Appendix 1—figure 4 caption]. 표 23-7의 `w/o memory mix`가 무너지는 것은 잡음까지 뺐기 때문이고, **잡음만 남기고 기억을 빼면 차이가 없다.** 논문 제목이 가리키는 성분(mixed episodes로부터의 adversarial dreaming)이 통제 대비 효과를 보이지 않았다.

**둘째, sleep 단계의 순서가 무관하다.** "our model does not show significant differences in performance when the order of sleep phases is switched" [PAD Discussion, L1046-1047; Appendix 1—figure 5]. 저자는 생물학의 sequential hypothesis(Giuditta et al. 1995)에서 NREM→REM 순서가 중요하다고 가설된다는 점을 인정하면서, 모델의 순서 독립성을 단계별 시냅스 변화가 작기 때문으로 **추정한다** [L1050-1055].

**셋째, 표현 품질의 지표가 하나뿐이고 절대 성능이 낮다.** 측정된 것은 linear readout 정확도 하나이며 저자는 이를 "an obvious simplification with regard to cortical processing"이라고 인정한다 [L1039-1041]. downstream 과제 성능은 측정되지 않았다. 그 하나뿐인 지표의 절대값이 CIFAR-10에서 "around 59% test accuracy"다 [PAD Results, L534; Figure 4c]. 같은 시기 self-supervised 표현학습의 CIFAR-10 linear probe와의 비교는 논문이 수행하지 않는다. **따라서 이 논문은 기제의 존재 증명이지 성능 주장이 아니며, 성능 근거로 인용하면 오용이다.** SVHN은 표본 분포 불균형 때문에 나이브 분류기의 최고 성능이 18.9%여서 baseline 자체가 자명하지 않다 [Figure 5 caption, L643].

> **[평가]** 여기서 검정의 부재가 양날이 된다. vendored PAD 전문에서 p-값·검정 이름을 검색하면 **한 건도 나오지 않는다**[본서 관찰]. 그러므로 "유의한 차이가 없다"는 두 보고 — 기억 혼합 무관, 단계 순서 무관 — **역시 검정에 근거하지 않는다.** 이 책은 두 보고를 저자의 관측으로 귀속시켜 인용하되 "혼합이 무용하다는 증명"으로 격상시키지 않는다. 정확한 진술은 이것이다 — **저자 자신의 통제는 혼합 성분의 효과를 보이지 못했고, 그 통제에는 검정이 없다.** 두 절반을 다 적어야 정직하다.

표 23-8. §23.2 정의가 요구하는 네 좌표.

| 좌표 | PLOS Sleep | PAD |
|---|---|---|
| 기제 클래스 | 국소 보상 STDP + 항상성 scaling + 열 합 보존 정규화. **loss·gradient·optimizer 없음** | ADAM gradient descent, GAN 목적함수, 한 블록의 부호 반전 |
| 규모 | 뉴런 842개, 가소성 시냅스 6272개, 행동 8개 | conv 4층 + deconv 4층, $n_z = 256$, CIFAR-10 / SVHN |
| 측정 지표 | 저자 정의 pattern discriminability(chance 0.5, 천장 0.70) | 선형 probe 정확도 **하나** (저자가 "명백한 단순화"라고 인정) |
| 반복 라운드 수 | 과제 2개, interleaved 위상 **1회**. 3과제 실험 없음 | 사이클마다 반복되나 $\rho$의 수치화 없음. NREM 없는 조건에서만 감소 경향 보고 |

---

## 23.7 Systems/serving 함의

독자의 1번 질문에 대한 답은 §23.4에서 이미 나왔다 — **어느 쪽에서도 decode가 바뀌지 않는다.** 이 절은 그 위에서 이송 가능한 systems 진술 세 개만 남기고 나머지를 명시적으로 금지한다.

**첫째, 배포 아티팩트가 없는 극단.** PLOS에서는 wake도 sleep도 같은 $\Theta^{H\to O}$를 쓴다. 동결된 서빙 가중치가 존재하지 않는다. Rosetta 표(→ ch01)로 옮기면 "shared weights로 배칭한다"는 전제가 **첫 movement cycle부터** 깨진다 — 에이전트마다 가중치가 즉시 갈라진다. Θ-경로 per-user delta 문제의 극단값이고, 그 문제가 LoRA류 구현의 부산물이 아니라 **경로에 내재**한다는 것을 보여 준다(→ ch19, ch29). LLM 어휘로 옮기면 이 논문의 wake-time inference는 서빙 중인 가중치를 끊임없이 fine-tuning하는 것에 해당한다.

**둘째, 오프라인 워크로드의 모양은 스케줄이 정하지 않는다.** PLOS의 sleep 라운드는 (i) 은닉 784유닛에 독립 Poisson 스파이크열을 만들고, (ii) 784×8 조밀 행렬을 60,000 timestep 동안 전진시키고, (iii) 사건마다 같은 행렬의 열 정규화를 다시 도는 것이다. 작고, 조밀하고, 시간축으로 순차적이며, 병목이 행렬 곱이 아니라 **사건당 정규화**다. LLM의 sleep 라운드(크고, 배치되고, gradient가 지배하는)와 정반대의 프로파일이다 — **"오프라인 계산"은 스케줄의 이름이지 워크로드의 모양이 아니다.** 세 경로의 sleep 비용을 한 축에 놓으려면 이 사실을 먼저 인정해야 한다(→ ch25).

**셋째, 유일한 용량 진술.** wake→sleep 경계를 넘는 보조 상태는 은닉 뉴런 784개의 목표 발화율인데, Uniform-Noise Sleep 통제가 이것을 집단 평균 하나로 줄여도 결과가 유지된다(0.67 ± 0.05 / 0.69 ± 0.03) [S4E/F, L1144-1150]. 과거 과제 데이터는 0바이트다.

> **[평가]** 셋째 항목은 **가설로만** 이송된다. 그 성질이 gradient로 학습되는 모델에 살아남는다면 Θ-경로의 sleep 라운드는 사용자별 replay buffer 없이 가중치만으로 돌 수 있다는 뜻이 되고, 그것은 $C_{\text{cap}}$ 회계를 바꾼다. 그러나 이 논문은 그 이송을 보이지 않았다 — 이 논문의 갱신 규칙에는 도함수가 없고, 저자들이 자기 그룹의 feedforward ANN 이식을 인용하지만 여기서 실행하지는 않는다 [Discussion, L677-679]. 이 책은 이것을 결과가 아니라 반증 가능한 가설로 세우고 Part III로 넘긴다(→ ch26, ch29).

금지도 명시한다. 두 논문 어느 쪽도 바이트·대역폭·arithmetic intensity·하드웨어를 보고하지 않는다. 따라서 **이 장에서 memory-hierarchy 논증을 만들 수 없다.** HBM이든 DRAM이든 이 워크로드를 어디에 놓는 진술은 전부 발명이 된다. NM D4의 억지 연결 금지 규칙을 여기서 그대로 적용한다 — 이 책이 memory-centric 논증을 세우는 자리는 생물학 장이 아니라 자체 측정이 있는 자리다(→ ch26, ch29).

> **[해설]** PAD 쪽에서 가져갈 것은 더 적지만 더 날카롭다. 이 논문의 오프라인 위상 두 개는 wake와 **같은 optimizer, 같은 $\eta_\Theta$, 동등 가중의 loss**를 쓰고 [L1618-1621], 유일한 차이가 목적함수의 선택과 한 블록에 대한 부호다. 새 연산도 새 커널도 추가 상태도 없다. 생물학 쪽에서 "각성과 sleep은 다른 종류의 가소성을 쓴다"로 서술되는 것이 계산 쪽에서는 "오프라인 라운드는 목적함수와 부호만 다르다"로 환원된다. 이 라인이 생물학에서 실제로 가져올 수 있는 것은 대체로 **스케줄과 목적함수의 배치**이지 새로운 갱신 primitive가 아니다.

---

## 23.8 한계와 bridge-out

**논문 자신이 남긴 문제.** PLOS는 세 과제 이상으로 확장되는지 모르고, 중단 없는 새 과제 학습이 얼마나 견딜 수 있는지 모르며(실패 레짐을 지목만 하고 경계를 재지 않는다 [L639-642, L649-653]), gradient로 학습되는 망으로의 이송을 자기 인용으로만 주장하고 [L677-679], 우위 논증의 핵심인 혼합 replay의 실재를 분석하지 않았다 [L668-670]. 모델링된 것은 REM류 활동뿐이고 NREM은 아예 없으며 [L292-297], 생물학적 예측도 미검증이다 [L721-722]. PAD는 latent 혼합 전략의 최적성을 미해결로 두고 [L1030-1033], 연속 학습에서 최근 기억의 선호적 replay가 유용하다는 것을 **가설로** 적으며 [L1036-1038], readout 학습이 encoder를 바꾸지 않는다는 자기 가정을 스스로 문제로 지목하고 [L1041-1044], 단계 순서가 영향을 주려면 시냅스 변화가 더 커야 한다는 예측을 검증하지 않았으며 [L1050-1055], 진동·spindle·slow wave를 포함한 회로 모델을 만들지 않았다 [L940-944].

여기서 이 장의 두 번째 정의를 못 박는다. ch08 §08.5가 **은유의 이송 한계**라는 경계를 이름 붙였다면, 이 장은 그 경계에 대한 판정 절차를 소유한다.

> **정의.** **은유의 이송 한계 판정**이란, 생물학 문헌을 근거로 든 임의의 주장 $P$에 대해 §23.2의 네 좌표를 차례로 묻는 결정 절차다. **(1) $P$가 요구하는 기제가 원 논문의 기제 클래스 안에 있는가. (2) $P$가 주장하는 규모가 원 논문의 규모 안에 있는가. (3) $P$가 쓰는 지표가 원 논문이 측정한 지표인가. (4) $P$가 함의하는 반복 라운드 수를 원 논문이 실행했는가.** 네 물음에 모두 "예"라야 $P$는 그 문헌이 지지하는 주장이다. 하나라도 "아니오"면 $P$는 유비이며, 유비임을 표시하지 않고 인용하는 것은 이 책에서 결함이다.

> **[평가] 이 문헌이 지지하는 것.**
> 1. 식 (U-$\Theta$)의 $\mathcal{R}_k$는 데이터셋일 필요가 없다. 한 시스템에서 모델을 잡음 위에 돌려 만든 스트림으로 충분했고, $\mathcal{H}_k$는 $\Theta_k$ 안 말고 어디에도 없었다 [PLOS L300-311].
> 2. 각성→오프라인 경계를 넘는 보조 상태는 $O(1)$ 스칼라로 줄어들 수 있다 [PLOS S4E/F, L1144-1150].
> 3. 오프라인 위상은 옛 파라미터의 정적 보호와 같지 않다. 시험된 동결 비율 세 점 중 1%·5%는 두 축에서 동시에 뒤지고, 10%는 Task 1에서 동률(0.70 ± 0.03)이면서 Task 2를 배우지 못한다(0.53 ± 0.03 대 0.68 ± 0.05) [PLOS S6 L1184-1187 vs L304-305]. 단 이 비교는 세 점·단일 선택 기준이고 EWC는 실행되지 않았다.
> 4. interleaving의 **입도**에 레짐 경계가 있다. 짧은 에피소드를 sleep과 교대하면 성공하고 한 번의 긴 에피소드는 회복 불가능하다 [PLOS L307-311, L639-653].
> 5. 오프라인 위상의 가치는 스케줄이 아니라 거기서 **무엇을 생성하는가**에서 나온다. 단 PAD에서 결정적인 성분은 기억의 혼합이 아니라 자발 활동의 주입이다 [PAD Appendix 1—table 1 + Appendix 1—figure 4].

> **[평가] 이 문헌이 지지하지 않는 것.**
> 1. **gradient로 학습되는 망에 대해 아무것도.** PLOS에는 loss도 도함수도 optimizer도 backward pass도 없다. SGD/AdamW sleep 라운드가 이 동역학을 물려받는다고 볼 근거가 없다.
> 2. **transformer·언어·시퀀스 모델링에 대해 아무것도.** attention도 recurrence도 토큰도 없다.
> 3. **배포 규모에 대해 아무것도.** 연속 가소성 시냅스 6272개 대 $10^9$–$10^{12}$ 파라미터.
> 4. **생성물의 의미적 정확성에 대해 아무것도.** 두 논문 모두 replay·dream의 **내용**을 검사하지 않는다. LLM에서 이에 대응하는 실패 — 유창하지만 거짓인 생성물로 $\Theta$를 갱신하는 것 — 이 Θ-경로의 중심 위험인데(→ ch06, ch21) 두 편 다 침묵한다.
> 5. **sleep이 데이터 replay보다 정확도에서 낫다는 것에 대해 아무것도.** 논문 자신이 우위를 이론적 근거로 돌린다 [PLOS L654-655].
> 6. **반복 라운드의 $\rho$에 대해 아무것도.** 과제 2개, interleaved 위상 1회다.
> 7. **서빙 비용에 대해 아무것도.** FLOPs·지연·바이트·배칭·하드웨어가 전부 없다.
> 8. **혼합 에피소드로부터의 dreaming과 sleep 단계 순서의 필요성에 대해 아무것도.** 저자 자신의 통제가 둘 다 지지하지 않는다 [PAD L1034-1036, L1046-1047].

**경로 간 인용.** PLOS는 E-경로와 W-경로에 완전히 침묵하고, 그 침묵은 연대가 강제한 것이다 — 2022년 11월 18일 게재로 corpus의 모든 E·W 경로 논문보다 앞선다 [L36]. Θ-경로 쪽으로는 2020년 이전 지속학습 부문만 접촉한다: EWC [ref 7], CLS [ref 11], Hayes et al. 2021 리뷰 [ref 9], meta-learning 한 건 [ref 51]. **Deep Generative Replay는 인용되지 않는다** — 저장 없이 $\mathcal{R}_k$를 만드는 두 해법이 서로를 보지 않은 자리이며 연대가 강제하지 않은 공백이다(→ ch08). PAD도 세 경로 전부에 침묵하고 이 역시 연대가 강제한 것이다. vendored 전문 검색 계수가 그 상태를 보여 준다 — transformer 0, language model 0, LoRA 0, test-time 0, fast weight 0, 그리고 replay 38[본서 관찰].

반대 방향이 이 장의 결론이다. **E-경로의 창시 논문은 제목에 sleep을 빌려 쓰면서 생물학·CLS 문헌을 한 편도 인용하지 않는다.** `Sleep-time Compute`(2504.13171)의 참고문헌 22편 [Letta STC References, pp.13-15] 안에 McClelland도, hippocampal replay도, sleep consolidation 신경과학도, dreaming 문헌도 없다. 그 단어를 도입하는 문장은 순전히 운영적이다: "inference is done between interactions with the model while it would otherwise be idle in sleep-time" [Letta STC §1]. **여기서 sleep은 consolidation이 아니라 유휴를 뜻한다.** 은유만 가져오고 근거는 두고 온 것이다.

> **[평가]** 두 방향의 침묵에 같은 등급을 매기지 않는다. 생물학 쪽의 침묵은 연대가 강제했으므로 계보 사실이고, `Sleep-time Compute`의 침묵은 선택이므로 발견이다 — CLS는 1995년, 이 장의 두 편은 2022년, 저 논문은 2025년이다. 그런데 결함은 인용의 부재가 아니다. 저 논문이 하는 일에 CLS의 두 시스템 논증은 필요하지 않다. **결함은 이름의 재사용이다.** 서로 다른 세 경로가 같은 단어를 쓰기 때문에 독자는 그것들이 한 가설의 변형이라고 읽게 되고, 이 책이 판별식을 따로 세워야 했던 이유가 정확히 그것이다(→ ch11, ch24).

**Part III가 받아가는 것.** ch24는 위 판정 절차를 판별식과 나란히 놓고 "무엇이 실제로 sleep-time compute인가"를 닫는다. ch25는 이 장이 비운 비용 칸을 통일 회계의 결측으로 기록하고, ch26·ch29는 `지지하는 것` 2번을 반증 가능한 가설로 세워 자체 측정과 대조한다. ch27은 "생물학이 지지한다"는 문장을 쓸 때마다 이 장의 두 목록을 통과시키고, ch30은 `지지하지 않는 것` 4번을 반증 조건의 재료로 쓴다. Part II는 여기서 끝난다 — 이제 판정으로.

---

## 요약

- PLOS는 상속받은 open question을 명시하지만 그 공백을 선언한 리뷰의 공저자가 이 논문의 교신저자다. PAD에는 상속 진술이 없다. 상속이 성립하는 자리는 경로가 이어진 자리가 아니라 연구 그룹이 이어진 자리다.
- PLOS는 (U-$\Theta$)를 wake와 sleep 두 시계로 같은 6272개 시냅스에 쓴다. gradient가 없고, 갱신이 열 합 보존 집합으로 사영되며, $\mathcal{R}_k$는 잡음 위에서 망 자신을 돌려 만든다.
- PAD는 사이클마다 (U-E) 한 번과 (U-$\Theta$) 세 번을 돌고 REM에서 한 파라미터 블록의 gradient 부호를 뒤집는다. 표준형 (U-$\Theta$)는 부호 필드 없이 이 형태를 담지 못한다.
- 비용 4종 여덟 칸의 절반이 "논문에 없음"이다. 생물학적 조상 두 편은 상각 증거를 제공하지 않으며, 제공하는 것은 상태 용량 논증 하나다.
- PLOS의 강한 baseline(과거 데이터를 저장하는 고전 interleaving)은 0.68/0.65로 sleep의 0.70/0.68과 표준편차 안에서 같고, 논문 자신이 우위를 이론적 근거로 돌린다. 실증된 우위는 정확도가 아니라 저장의 제거다.
- PAD의 저자 통제는 논문 제목이 가리키는 성분(혼합 에피소드)의 효과도, sleep 단계 순서의 필요성도 보이지 못했다. 표현 품질 지표는 선형 probe 하나뿐이고 CIFAR-10 절대값은 약 59%다.
- E-경로의 창시 논문은 제목에 sleep을 쓰면서 생물학·CLS를 한 편도 인용하지 않는다. 이름의 공유는 이론적 약속의 공유가 아니다.

## 자가 점검 체크리스트

- [ ] PLOS와 PAD 각각에 대해 §23.2의 네 좌표를 채워 말할 수 있다.
- [ ] 임의의 "생물학이 지지한다"는 문장에 §23.8의 네 물음을 적용해 지지/유비를 판정할 수 있다.
- [ ] 두 논문의 비용 4종 중 어느 칸이 왜 비어 있고 그 공백이 어떤 서술을 금지하는지 설명할 수 있다.
- [ ] PLOS의 sleep이 고전 interleaving 대비 실증한 것과 이론으로만 주장한 것을 갈라 말할 수 있다.
- [ ] PAD에서 REM의 결정적 성분이 혼합이 아니라 자발 활동의 주입임을, 그 근거의 검정 부재까지 함께 말할 수 있다.
- [ ] 연대가 강제한 침묵과 선택된 침묵을 실제 사례로 각각 하나씩 들 수 있다.
- [ ] 이 두 논문의 오프라인 위상을 inference 어휘로 옮길 수 있다 — warm-up/사전 컴파일 자리인지 모델 재배포 자리인지, 그리고 왜 두 편이 후자인지.
