# ch05. RL for LLM — 보상으로 가중치를 움직이기

> **이 장의 목표** — 독자가 이 장을 마치면 다음을 할 수 있어야 한다.
> (1) 임의의 LLM 학습 루프를 보고 무엇이 state, 무엇이 action, 무엇이 reward인지 지목하고, 그 셋이 decode loop의 어느 부분에 해당하는지 대응시킬 수 있다.
> (2) policy gradient의 최소 골격(rollout → reward → advantage → 확률 이동)을 손으로 한 step 돌리고, 왜 reward가 미분 불가능해도 되는지 설명할 수 있다.
> (3) RLHF 파이프라인 · DPO · RLVR를 "무엇이 채점하는가"와 "어떤 하드웨어를 먹는가"의 두 축으로 구분하고, 각각이 sleep 라운드에 얹힐 때의 비용 모양을 말할 수 있다.
> (4) reward hacking을 정의하고, RL 이득 보고에 어떤 통제군이 붙어 있어야 하는지 판정할 수 있다.
>
> **왜 필요한가** — Part II의 세 장이 이 어휘 없이는 읽히지 않는다. **ch20의 SEAL** (*Self-Adapting Language Models*, arXiv:2506.10943)은 [SEAL §3.1] 전체가 RL 표기로 쓰여 있다 — 목적함수 [SEAL Eq. 1], 이진 보상 [SEAL Eq. 2], gradient 추정량 [SEAL Eq. 3–4], 그리고 알고리즘 선택의 근거("GRPO와 PPO는 불안정했다") [SEAL §3.1, p.4]. 이 식들을 읽으려면 policy·rollout·advantage가 무엇인지 알아야 한다. **ch21의 `Language Models Need Sleep`** (arXiv:2606.03979)의 Dreaming 단계는 semantic reward와 distillation을 계수로 섞는다 — 그 기호들($\rho_r$, $\lambda_{\mathrm{KD}}$, $r_{\mathrm{sem}}$)은 NM에서 이미 만났지만(→ NM ch17), "reward를 섞는다"가 최적화 관점에서 무엇을 하는 일인지는 이 장이 공급한다. (reward 혼합 계수는 ch21에서 $\rho_r$로 쓴다. 맨 $\rho$는 이 책에서 비용 4종의 망각·열화율로 예약되어 있어 덮어쓸 수 없다.) **ch15의 ReasoningBank** (arXiv:2509.25140)은 agent를 $\pi_L(\cdot\mid M, A)$라는 policy 표기로 쓰고 자기 루프가 "RL의 학습 동역학을 닮았다"고 서술하는데 [ReasoningBank §3.1, §5], 정작 gradient는 한 번도 계산하지 않는다. 그 차이를 독자가 스스로 판정하려면 RL이 실제로 무엇을 하는지 먼저 알아야 한다.
>
> **NM과의 관계** — 넘길 것: backward pass와 optimizer state의 객체적 이해는 전제한다(→ NM ch02). stream 위의 단일 패스 최적화, regret, comparator는 → NM ch03. inner/outer loop의 형식화는 → NM ch04. `Sleep`의 GKD 계수 표기는 → NM ch17. 더할 것은 하나다: **NM의 어느 장에도 reward가 없다.** NM ch03의 online protocol에서 loss는 stream이 자기 label을 들고 도착하는 self-supervised 회귀였다($\ell(W;k_t,v_t)=\|\mathcal{M}(k_t;W)-v_t\|_2^2$). RL의 reward는 생성이 **끝난 뒤 밖에서** 도착하는 스칼라 하나다. 이 차이가 이 장의 전부다. 그리고 어휘 충돌 하나를 미리 못 박는다 — NM ch03의 "online" $t$는 token 인덱스이고, 이 장의 episode step도 token이지만, **NM ch03의 regret 보장을 이 장의 policy 성능 주장으로 옮겨 읽으면 안 된다.** 둘은 다른 comparator에 대한 다른 보장이다.

## 05.1 LLM 위에서 무엇이 state이고 무엇이 action이고 무엇이 reward인가

RL을 처음 만나는 systems 엔지니어가 막히는 지점은 알고리즘이 아니라 **대응**이다. "환경과 상호작용하며 누적 보상을 최대화한다"는 문장은 로봇이나 게임을 떠올리게 하지만, LLM에는 로봇도 물리 환경도 없다. 대응은 아주 구체적이고, 한 번 못 박으면 다시 흔들리지 않는다.

표 5-1 — LLM 위의 RL 대응표. 오른쪽 열이 독자가 이미 매일 다루는 대상이다.

| RL의 항 | LLM에서의 정체 | decode 세계의 대응물 |
|---|---|---|
| state $s_t$ | 지금까지의 token 열 = prompt $q$ + 생성된 $z_{<t}$ | KV cache가 담고 있는 바로 그것 |
| action $a_t$ | 다음 token 하나 | 한 번의 decode step의 출력 |
| action space $\mathcal{A}$ | vocabulary 전체 | logits 벡터의 차원 |
| 전이 $s_{t+1}$ | $s_t$에 방금 뽑은 token을 붙인 것 | KV cache append |
| episode | 한 번의 완결된 생성 | 한 request의 decode 전체 |
| horizon | 생성 길이 | max_new_tokens |
| reward $R$ | 생성이 끝난 뒤 밖에서 도착하는 스칼라 하나 | **대응물 없음 — 이것이 새것이다** |

이 표에서 셋을 강조한다.

첫째, **전이가 결정론적이고 완전히 알려져 있다.** $s_{t+1}$은 $s_t$ 뒤에 $a_t$를 이어 붙인 것, 곧 문자열 concat이다. 로봇 RL이 씨름하는 환경 모델 학습이 여기에는 없다. 이것은 LLM RL을 예외적으로 다루기 쉽게 만드는 성질이고, 동시에 "환경"이라는 단어가 오해를 부르는 이유다 — 환경은 KV cache append 연산이다.

둘째, **보상은 거의 항상 terminal이다.** 중간 token에는 보상이 없고, 생성이 끝난 뒤 채점기가 한 번 호출되어 스칼라 하나를 준다. 즉 신호는 극도로 성기다. 500 token을 뽑고 "0"을 받으면 그 500개 중 어느 결정이 잘못이었는지는 알려주지 않는다. RL 알고리즘 설계의 절반은 이 credit assignment 문제를 다루는 장치다.

셋째, **discount factor를 쓰지 않는다.** episode가 유한하고 짧으며, LLM RL의 관행은 undiscounted return이다. 그래서 이 책은 discount 기호를 아예 도입하지 않는다 — $\gamma_{t,i}$는 NM이 Omega rule의 window gate로 이미 예약한 기호이고, 덮어쓰면 두 책이 갈라진다.

> **[해설]** systems 관점에서 이 대응의 실용적 귀결은 하나다. RL의 state가 KV cache와 같은 것이라면, **RL 훈련의 절반은 batch decode job이다.** rollout을 만드는 단계는 독자가 매일 튜닝하는 그 워크로드 — memory-bandwidth bound, KV cache가 HBM을 지배하고, batch size와 prefix 공유가 처리량을 정하는 — 와 문자 그대로 같다. "RL로 후처리한다"는 말을 들었을 때 무엇이 GPU를 먹는지 감이 서지 않았다면, 절반은 서빙이고 절반은 훈련이라고 생각하면 정확하다.

## 05.2 policy, rollout, advantage — policy gradient의 최소 골격

**policy**는 state에서 action으로의 조건부 분포다. LLM에서는 별도의 policy network가 없다 — **모델 자신이 policy다**: $\pi_\Theta(z_t \mid q, z_{<t})$가 곧 softmax 출력이고, policy를 갱신한다는 것은 $\Theta$를 움직인다는 것, 즉 식 (U-$\Theta$)를 실행한다는 것이다. 이 동일시가 LLM RL의 출발점이자, RL을 이 책의 3층 프레임에 놓을 수 있는 이유다.

**rollout**은 현재 policy로 prompt 하나에서 끝까지 sampling한 완결된 생성 하나와, 그것에 매겨진 보상의 쌍이다. rollout 하나 = decode 한 번. 표기는 생성된 token 열을 $z$, 그 $t$번째 token을 $z_t$, 보상을 $R$로 쓴다.

이 장에서만 쓰는 보조 기호를 먼저 선언한다(이 책의 예약 기호와 충돌하지 않도록 전부 개명했다).

표 5-2 — 장-국소 기호 선언

| 이 장의 기호 | 의미 | 왜 이 기호인가 |
|---|---|---|
| $\pi_\Theta$ | policy = 모델 자신 | $\Theta$가 학습 대상임을 기호에 드러냄 |
| $z$, $z_t$ | rollout(생성 token 열)과 그 $t$번째 token | $y_t$(layer 출력)와 충돌 회피 |
| $\hat A_i$ | rollout $i$의 advantage | NM의 보조 accumulator $A_t$와 hat·첨자로 구분 |
| $R_\psi$ | reward model | 첨자 $\psi$로 $\mathcal{R}_k$(학습 집합)·$R_t$(NM의 retention 항)와 구분 |
| $\beta_{\mathrm{KL}}$ | KL penalty 계수 | $\beta_t$(inner momentum decay)와 반드시 구분 |
| $M$ | 한 prompt당 rollout 수 | — |
| $c$ | context(질의 이전에 주어진 문서·시연) | Atlas의 window 길이 $c$는 이 장에 등장하지 않음 |

*표 5-2 각주 — 보상 기호의 대응.* 이 절에서 만나는 원 논문들(PPO, GRPO, ReST$^{EM}$, SEAL 포함)은 보상을 소문자 $r$로 쓴다. 이 책은 $r$을 LoRA rank에 예약했으므로(→ ch03) 보상은 전부 대문자 $R$로 옮겨 쓴다. 원문 직접 인용 안의 $r$은 그대로 둔다. NM에서 승계한 $r_{\mathrm{sem}}$(→ NM ch17)만 예외이며, 승계 기호는 재정의하지 않는다.

policy gradient의 핵심 식은 하나다. 목적은 기대 보상을 최대화하는 것, 즉 outer loss를 그 음수로 놓고,

$$
\nabla_\Theta \mathcal{L}_{\mathrm{RL}}
\;=\;
-\,\mathbb{E}_{z \sim \pi_\Theta(\cdot\mid q)}
\Big[\,\hat A(z)\;\nabla_\Theta \log \pi_\Theta(z \mid q)\,\Big]
\tag{5-1}
$$

이 식이 말하는 것은 한 문장이다: **샘플이 좋았으면 그 샘플을 다시 낼 확률을 올리고, 나빴으면 내린다.** 그리고 결정적으로, gradient는 보상 자체를 미분하지 않는다. 미분되는 것은 $\log \pi_\Theta$뿐이다. 따라서 **보상은 미분 불가능해도 된다** — 사람이어도, 프로그램이어도, 다른 모델이어도, 심지어 "finetune 한 번 돌려서 평가한 점수"여도 된다. 이것이 SFT 대신 RL을 쓰는 유일하고 충분한 이유다: RL은 **미분 불가능한 채점기를 loss로 만드는 장치**다.

식 (5-1)의 gradient는 sequence 하나에 대한 것이지만 실제 계산은 token 단위로 풀린다. $\log\pi_\Theta(z\mid q) = \sum_{t} \log \pi_\Theta(z_t \mid q, z_{<t})$이므로

$$
\nabla_\Theta \log \pi_\Theta(z\mid q)
\;=\;
\sum_{t=1}^{|z|} \nabla_\Theta \log \pi_\Theta\big(z_t \mid q, z_{<t}\big)
\tag{5-1'}
$$

이고, 여기에 스칼라 $\hat A(z)$가 곱해진다. 이 형태가 §05.1에서 말한 credit assignment의 실체다 — **생성 전체에 매겨진 점수 하나가 그 안의 모든 token에 똑같은 계수로 곱해진다.** 어느 token이 잘했는지에 대한 정보는 어디에도 없고, 여러 rollout에 걸쳐 통계적으로만 분리된다. SEAL의 추정량이 정확히 이 token 전개 형태로 쓰여 있다 [SEAL Eq. 4, §3.1, p.4]. systems 관점의 귀결: 갱신 단계의 GEMM shape은 SFT와 완전히 같고(이미 확정된 token 열 위의 forward+backward), 달라지는 것은 per-sequence 스칼라 하나가 loss에 곱해진다는 것뿐이다. **RL의 갱신 단계는 가중치가 붙은 SFT다.**

$\hat A$ 자리에 보상 $R$을 그냥 넣으면 두 가지가 망가진다. 분산이 크고, 더 나쁘게는 모든 rollout이 $R>0$이면 전부 확률을 올려 아무 것도 구분하지 않는다. **advantage**는 그 rollout의 보상에서 같은 prompt에 대한 기준선(baseline)을 뺀 값이다: $\hat A_i = R_i - b(q)$. 기준선을 빼도 gradient의 기댓값은 변하지 않고 분산만 준다 — advantage가 하는 일은 "이 rollout이 **평소보다** 좋았는가"로 질문을 바꾸는 것이다.

baseline을 만드는 방식이 알고리즘 계보를 가른다. 하나는 별도의 value network를 학습해 $b(q)$를 예측하는 것이고(PPO 계열, Schulman et al. 2017, arXiv:1707.06347; advantage 추정은 Schulman et al. 2015, arXiv:1506.02438), 다른 하나는 같은 prompt에서 $M$개의 rollout을 뽑아 그 표본 평균을 baseline으로 쓰는 것이다(group-relative 방식, Shao et al. 2024, arXiv:2402.03300). 후자는 value network를 통째로 없애 HBM에서 모델 하나를 지운다.

여기에 두 개의 안전장치가 붙는다. 하나는 **KL penalty**: 참조 policy $\pi_{\mathrm{ref}}$(보통 RL 시작 시점의 SFT 모델)에서 멀어지는 것을 $\beta_{\mathrm{KL}}$로 벌해, policy가 보상만 좇다가 언어 모델이기를 그만두는 것을 막는다. 다른 하나는 **clipping**: 한 batch의 rollout으로 여러 gradient step $s = 1, 2, \dots$를 밟을 때 policy가 rollout을 뽑은 분포에서 너무 멀어지지 않도록 확률비를 잘라낸다(PPO의 정의적 장치).

여기서 첨자 하나를 못 박는다. **$s$는 optimizer가 밟는 SGD step이고, $k$는 sleep 라운드다.** 한 sleep 라운드 $k$ 하나가 여러 step $s$를 품으며, 식 (U-$\Theta$)의 $\Theta_{k+1}$은 그 step들이 모두 끝난 뒤의 결과물이다. rollout을 다시 뽑아야 하는 주기는 $k$의 시계이고, clipping이 지키는 것은 $s$의 시계다. 이 책은 두 첨자를 섞어 쓰지 않는다.

> **[해설]** RL 한 iteration은 서로 arithmetic intensity가 정반대인 세 단계다. ① **rollout 단계** — decode. memory-bandwidth bound, KV cache가 지배, 배치가 클수록 좋다. ② **채점 단계** — 무엇이 채점하느냐에 따라 GPU forward일 수도, CPU 프로그램일 수도, 외부 API일 수도 있다. ③ **갱신 단계** — 이미 확정된 token 열 위의 forward+backward. prefill 모양의 큰 GEMM, compute bound. 실무 스택이 sampler(vLLM류)와 trainer를 따로 두고 weight를 왕복시키는 이유가 이것이고, ch20이 다룰 SEAL의 구현도 정확히 그 구조다 — 한 sleep 라운드에 vLLM과 DeepSpeed ZeRO-3가 동시에 상주한다 [SEAL §B.5, p.21].

## 05.3 RLHF 파이프라인 — 채점기를 학습해서 사람을 대신 세운다

식 (5-1)은 보상이 있다고 가정한다. 문제는 우리가 원하는 것 대부분에 채점기가 없다는 것이다. "도움이 되는 답", "요약이 원문에 충실한가" 같은 목표에는 정답 문자열도 단위 테스트도 없다. RLHF는 이 공백을 **채점기를 학습해서** 메운다.

**reward model**은 사람이 매긴 선호 비교 데이터로 학습된, (prompt, 응답) 쌍을 스칼라 점수로 보내는 함수 $R_\psi(q,z)$이다. 사람에게 절대 점수를 매기게 하지 않고 **둘 중 어느 쪽이 나은가**만 묻는다는 것이 설계의 핵심이다 — 절대 점수는 평가자 간에 재현되지 않지만 쌍 비교는 상당히 재현된다. 선호를 확률로 바꾸는 표준 모형은 Bradley–Terry이고,

$$
P\big(z^{+} \succ z^{-} \mid q\big)
\;=\;
\sigma\big(R_\psi(q,z^{+}) - R_\psi(q,z^{-})\big)
\tag{5-2}
$$

$R_\psi$는 이 로그가능도를 최대화하도록 학습된다(Christiano et al. 2017, arXiv:1706.03741). 식 (5-2)가 말하는 것: reward model이 배우는 것은 점수의 절대 눈금이 아니라 **차이**뿐이다. 상수를 더해도 식이 변하지 않으므로 눈금 원점은 정의되지 않는다.

**RLHF 파이프라인**은 이 reward model을 가운데 두고 학습을 세 단계로 나누는 후처리 절차다: (1) 사람이 쓴 시연으로 supervised fine-tuning을 해 policy의 형식을 잡고, (2) 그 policy가 뽑은 응답들에 대한 사람의 쌍 비교로 reward model $R_\psi$를 학습하고, (3) $R_\psi$를 보상으로 삼아 식 (5-1)로 policy를 갱신한다(Stiennon et al. 2020, arXiv:2009.01325; Ouyang et al. 2022, arXiv:2203.02155).

표 5-3 — RLHF 세 단계에서 무엇이 움직이는가

| 단계 | 움직이는 것 | 데이터 | 프레임 |
|---|---|---|---|
| SFT | $\Theta$ | 사람이 쓴 (prompt, 응답) | (U-$\Theta$), $\mathcal{R}$은 사람이 만듦 |
| reward model 학습 | $\psi$ (별도 모델) | 사람의 쌍 비교 | (U-$\Theta$)의 형태이나 대상이 policy가 아님 |
| RL | $\Theta$ | policy가 스스로 뽑은 rollout | (U-$\Theta$), $\mathcal{R}$은 policy가 만들고 $R_\psi$가 채점 |

1단계가 먼저인 이유는 실용적이다 — rollout이 최소한의 형식을 갖추어야 reward model이 의미 있는 점수를 준다. 그리고 3단계가 RL이어야 하는 이유는 §05.2의 답 그대로다: $R_\psi$의 출력을 $\Theta$에 대해 미분할 수 없기 때문이 아니라(사실 $R_\psi$는 미분 가능하다), **$z$를 뽑는 sampling이 미분 불가능**하기 때문이다. 식 (5-1)의 score-function 추정량이 정확히 그 벽을 넘는 장치다.

> **[해설]** 3단계의 HBM 회계를 세어 보면 RLHF가 왜 무거운지가 바로 보인다. 동시에 상주해야 하는 모델이 넷이다 — 학습 중인 policy, KL 항을 계산할 frozen reference, frozen reward model, 그리고 학습 중인 value network. 넷을 모두 7B로 잡으면 bf16 가중치만 4×14 GB = 56 GB이고, 여기에 학습 대상 **둘**의 optimizer state가 더 붙는다(→ NM ch02). 이 56 GB는 어느 논문의 보고값도 아니라 자릿수를 잡기 위한 **이 책의 예시 계산**이다 — 7B라는 크기와 bf16 2 bytes/elem를 이 책이 골랐다. 같은 데이터로 SFT만 할 때와 비교하면 몇 배다. 이후 알고리즘 계보가 전부 "이 넷 중 몇 개를 지울 수 있나"의 역사인 것은 우연이 아니다 — group-relative baseline은 value network를 지우고, 다음 절의 DPO는 sampler와 reward model을 함께 지운다.

## 05.4 DPO — reward model을 지우고 preference를 직접 loss로

**DPO**(Direct Preference Optimization)는 reward model을 명시적으로 학습하지 않고, 선호 쌍에 대한 닫힌 형태의 분류 loss로 policy를 직접 갱신하는 방법이다(Rafailov et al. 2023, arXiv:2305.18290).

출발점은 KL로 정규화된 RL 목적의 최적해가 닫힌 형태를 갖는다는 관찰이다: $\pi^{\star}(z\mid q) \propto \pi_{\mathrm{ref}}(z\mid q)\exp\!\big(R(q,z)/\beta_{\mathrm{KL}}\big)$. 이 관계를 보상에 대해 뒤집으면 보상을 policy로 표현할 수 있고, 그것을 식 (5-2)에 대입하면 정규화 상수가 소거되어 $R_\psi$가 통째로 사라진다.

$$
\mathcal{L}_{\mathrm{DPO}}(\Theta)
=
-\,\mathbb{E}_{(q,z^{+},z^{-})}
\left[
\log \sigma\!\left(
\beta_{\mathrm{KL}}\log\frac{\pi_\Theta(z^{+}\mid q)}{\pi_{\mathrm{ref}}(z^{+}\mid q)}
-
\beta_{\mathrm{KL}}\log\frac{\pi_\Theta(z^{-}\mid q)}{\pi_{\mathrm{ref}}(z^{-}\mid q)}
\right)
\right]
\tag{5-3}
$$

식 (5-3)에서 무엇이 없는지가 무엇이 있는지보다 중요하다. **sampling이 없다.** rollout이 없고, 스칼라 보상이 없고, advantage가 없고, value network가 없다. 남은 것은 고정된 선호 쌍 위의 지도학습이며, 필요한 forward는 policy와 reference 각각 두 번씩, 총 네 번이다.

표 5-4 — 같은 선호 데이터를 쓰는 두 경로의 부품 대조

| 부품 | RLHF (3단계) | DPO |
|---|---|---|
| reward model $R_\psi$ | 학습하고 상주 | 없음 |
| sampler(rollout 생성) | 매 iteration | 없음 |
| value network | 있음(또는 group baseline) | 없음 |
| reference 모델 | 있음(KL) | 있음(식 (5-3) 안에) |
| 워크로드 모양 | decode + 채점 + 훈련 | 훈련만 |

대가는 **off-policy**라는 것이다. 선호 쌍이 현재 policy의 분포에서 오지 않으므로, policy가 데이터를 만든 분포에서 멀어질수록 신호가 낡는다. RLHF의 rollout은 정의상 항상 현재 policy의 것이고, DPO의 쌍은 한 번 고정된다.

> **[평가]** 이 책은 RLHF와 DPO 사이에 우열을 매기지 않는다. 두 방법의 선택을 정하는 변수는 목적 도메인과 선호 데이터의 갱신 주기이지 알고리즘의 우아함이 아니며, 그 두 변수를 명시하지 않은 비교는 판정으로 성립하지 않는다.

sleep 라운드의 관점에서 이 절의 요점은 하나다. **DPO의 워크로드 모양은 SFT와 같다.** 고정 배치, sampler 없음, 채점기 없음. 유휴 시간에 얹기가 비교할 수 없이 싸고, 스케줄링이 훈련 job 하나로 끝난다. ch21이 다룰 `LM Need Sleep`의 Dreaming 단계가 보상 항과 distillation 항을 계수로 섞는 형태를 취하는 것도 같은 압력 아래에 있다(그 계수 표기는 → NM ch17).

## 05.5 RLVR — 채점기를 프로그램으로 바꾼다

**RLVR**(Reinforcement Learning with Verifiable Rewards, 검증가능 보상)은 보상을 사람이나 학습된 모델이 아니라 **결정론적 검증 프로그램**이 산출하는 RL이다. 정답 문자열 대조, 단위 테스트 실행, 파서·컴파일러 통과 여부, 출력 형식 정규식이 전형적인 검증기다. 이 이름은 Lambert et al. 2024(Tulu 3, arXiv:2411.15124)가 붙였고, 규칙 기반 보상만으로 대규모 추론 학습을 돌린 사례로 DeepSeek-R1(arXiv:2501.12948)이 널리 인용된다.

식은 새로울 것이 없다 — 식 (5-1) 그대로이고 $R$의 출처만 바뀐다. 바뀌는 것은 전부 회계와 실패 모드다. 검증기의 종류를 먼저 갈라 두면 그 회계가 보인다.

표 5-5 — 검증기의 네 종류와 그 성질

| 검증기 | 예 | 결정론 | 어디서 도는가 | 이 corpus의 사례 |
|---|---|---|---|---|
| 문자열 대조 | 수학 정답, 추출형 QA | 완전 | CPU, 무시 가능 | — |
| 실행 | 단위 테스트, 컴파일 | 완전(샌드박스 고정 시) | 격리 프로세스 | — |
| 형식 검사 | JSON 파싱, 정규식, 스키마 | 완전 | CPU | SEAL의 few-shot self-edit이 JSON 설정이다 [SEAL Fig. 3, p.5] |
| 모델 채점 | LLM-as-a-Judge, 루브릭 | 아님 | GPU 또는 API | ReasoningBank의 $j_k$ [ReasoningBank §3.2]; SEAL의 proxy-reward ablation [SEAL Table 9] |

앞의 셋만이 엄밀한 의미의 RLVR다. 네 번째 행은 "검증기"라는 이름을 쓰되 실제로는 학습된 채점기이므로 §05.6의 첫 번째 실패 모드를 그대로 물려받는다 — §05.7이 다루는 두 자동 채점기가 모두 네 번째 행에 걸린다는 사실은 거기서 다시 만난다.

검증기가 프로그램일 때 얻는 것은 셋이다.

1. **사람 라벨이 필요 없다.** reward model 학습 단계가 통째로 사라진다.
2. **reward hacking의 표면이 좁다.** 채점기가 학습된 함수가 아니므로 policy가 채점기의 일반화 오차를 타고 오를 수 없다(§05.6). 채점기 자체의 허점은 남는다.
3. **보상을 무인으로, 임의 횟수, 거의 공짜로 만들 수 있다.**

이 책의 관심은 3번에 있다.

> **[해설]** sleep 라운드는 정의상 사람이 없는 시간이다. (U-$\Theta$)를 돌리려면 학습 집합 $\mathcal{R}_k$가 쓸모 있었는지 판정할 신호가 필요한데, 그 신호를 사람이 준다면 라운드의 처리량은 사람의 처리량에 묶인다 — 유휴 시간을 쓴다는 sleep-time compute의 전제 자체가 무너진다. RLVR는 그 신호를 프로그램으로 바꾼다. 그래서 "sleep 라운드에서 $\Theta$를 옮긴다"는 설계는 검증 가능한 도메인(수학, 코드, 구조화 출력)에서 먼저 나타난다. **이 연결이 이 장이 Part II에 남기는 예고다.** 실제로 어떤 형태의 자동 채점기가 쓰이는지는 §05.7이 보이고, 그것이 충분한지는 ch20·ch21이, 세 경로 전체에 대한 답이 되는지는 ch27이 판정한다. 이 장은 판정하지 않는다.

한계도 같은 자리에서 나온다. **검증 가능한 것은 corpus의 일부다.** 임의의 문서를 읽고 그것을 가중치에 넣었는지 판정하려면 그 문서에 대한 문제와 답이 있어야 한다. SEAL은 이 벽에 정면으로 부딪히고, 논문 자신이 그것을 명시한다 — 모든 context에 라벨된 downstream task가 짝지어져 있어야 하며 "이 결합이 보상 계산을 단순하게 만들지만 SEAL의 RL 학습이 라벨 없는 코퍼스로 확장되는 것을 막는다" [SEAL §5 Context-dependent evaluation, p.9]. 논문이 제안하는 탈출구는 모델이 문서를 문맥에 두고 있는 동안 **평가 문제까지 스스로 생성**하게 하는 것이다 [SEAL §5, p.9].

> **[평가]** 이 탈출구는 문제를 옮길 뿐 없애지 못한다. 평가 문제를 모델이 만들면 보상의 신뢰도는 검증기가 아니라 문제 생성기의 품질에 걸리고, 그 생성기를 채점할 독립 신호는 다시 없다. 즉 라벨 의존이 사라지는 것이 아니라 라벨 의존이 한 단계 안쪽으로 들어간다. 이 장은 이 관찰만 남기고, 그것이 SEAL의 실증 범위에 실제로 어떤 상한을 거는지는 ch20이 판정한다.

systems 접점: 검증기는 GPU 밖에 산다. 샌드박스, 테스트 러너, 파서는 CPU·IO 서비스이고, 결정론이므로 캐시할 수 있고 batch로 돌릴 수 있으며 실패해도 재시도가 안전하다. RL 루프의 tail latency가 GPU에서 그 서비스로 옮겨간다. reward model 방식(rollout마다 채점 모델의 forward가 한 번 더 붙는다)과 비교하면 비용이 완전히 다른 자리에 선다 — 한쪽은 HBM과 GEMM, 다른 쪽은 프로세스 격리와 타임아웃 정책이다.

## 05.6 reward hacking — 보상이 목적을 대신할 때

**reward hacking**은 policy가 의도한 목적을 달성하지 않은 채 보상 신호의 결함을 최대화하는 현상이다. Goodhart의 법칙의 기계학습 판이며(Amodei et al. 2016, arXiv:1606.06565), 버그가 아니라 구조적 귀결이다 — reward는 목적의 대리(proxy)이고, RL은 그 대리를 최대화하도록 설계된 최적화기다. 대리와 목적이 갈라지는 지점을 policy가 찾아내는 것은 실패가 아니라 **정확히 시킨 일**이다.

두 종류로 나누면 대응책이 달라진다.

첫째, **학습된 채점기의 오차를 타는 것.** reward model은 고정된 분포에서 학습되었는데 policy는 학습하면서 분포를 옮긴다. policy가 reward model의 훈련 분포 밖으로 나가면 $R_\psi$의 점수는 계속 오르는데 실제 품질은 꺾인다 — reward model overoptimization으로 불리며 그 함수적 형태를 측정한 연구가 있다(Gao·Schulman·Hilton 2022, arXiv:2210.10760). 대응은 세 가지다: KL 항 $\beta_{\mathrm{KL}}$로 참조 policy 근처에 묶기, reward model을 새 분포에서 재학습하기, ensemble로 불확실성을 벌하기.

둘째, **결정론적 채점기의 허점을 타는 것.** 단위 테스트만 통과시키는 코드, 형식만 맞춘 빈 답, 채점 스크립트가 참조하는 파일을 건드리는 행위. RLVR도 면역이 아니다. 다만 이 종류는 **재현 가능**하다 — 허점은 프로그램에 있으므로 찾아서 고칠 수 있고, 고치면 사라진다. 학습된 채점기의 오차는 그렇게 국소적이지 않다.

가장 흔한 형태는 따로 이름을 붙일 만하다: **길이 인플레이션**. 길수록 점수가 높아지는 채점기(사람 평가자를 포함해)에서 policy는 길어진다. 그리고 이 corpus 안에 문서화된 사례가 있다.

SEAL은 RL iteration이 진행될수록 self-edit이 길어지는 것을 그림으로 보고하고 [SEAL Fig. 5, p.8], 논문 자신이 "RL이 예시 응답의 길이를 극적으로 늘린 것으로 나타난다. 그래서 더 긴 생성을 요구하는 prompt를 실험한다"고 쓴다 [SEAL §B.11, p.23]. 그 실험의 결과가 표로 남아 있다. RL을 한 라운드도 돌리지 않고 prompt만 "implications-long"으로 바꾸면 49.3, "rewrite"로 바꾸면 49.4이며, 두 값 모두 2라운드 RL을 돌린 "implications"의 47.0보다 높다 [SEAL Table 10, §B.11, p.24].

> **[평가]** 이 표가 보여주는 것은 reward hacking의 교과서적 **진단 절차**다 — 보상과 상관된 표면적 변량(여기서는 길이)을 의심하고, RL 없이 그 변량만 prompt로 직접 조작해 같은 이득이 나오는지 본다. SEAL은 그 검사를 실제로 수행했고 결과를 실었다. 이 장이 못 박는 규범은 그 절차 쪽이다: **RL 이득을 보고할 때는 보상과 상관된 표면 변량을 prompt만으로 재현해 보는 통제군이 있어야 하고, 없으면 그 이득은 기제에 귀속될 수 없다.** 논문 본문이 그 통제군 대신 47.0을 헤드라인으로 계속 쓰는 것은 별개 문제이며, 그 판정은 ch20이 한다. 덧붙여, 47.0이라는 값 자체도 세 열 중 단일 passage 열의 값이고 나머지 두 continued-pretraining 열에서는 GPT-4.1 합성 데이터가 SEAL을 앞선다 [SEAL Table 2, p.8] — 이 범위 조건 없이 47.0을 인용하지 않는다.

systems 접점: 길이 인플레이션은 품질 지표만 흐리는 것이 아니라 **회계를 망가뜨린다**. rollout이 길어지면 decode 시간이 선형으로 늘고 KV cache가 선형으로 늘며, 채점기가 LLM이면 채점 비용까지 함께 늘어난다. 즉 reward hacking은 sleep 예산 $B_s$를 라운드마다 키운다. 상각식 (A)의 분자가 학습이 진행되는 동안 자란다는 뜻이고, 학습 초기에 계산한 손익분기 $N_q^{\star}$는 나중 라운드에서 성립하지 않는다.

## 05.7 이 corpus가 실제로 쓰는 RL 루프 — 하나는 진짜, 하나는 어휘만

Part II에서 만날 두 논문이 RL을 정반대 방식으로 사용한다. 나란히 놓으면 이 장의 어휘가 무엇을 판별하는 도구인지가 드러난다.

### 05.7.1 SEAL — 보상이 "훈련 실행 한 번"인 RL

먼저 소유권을 밝힌다. 이 절이 소유하는 것은 SEAL의 **RL 기제** — 세 항의 대응, 추정량의 형태, 보상이 무엇으로 만들어지는가 — 뿐이다. self-edit이라는 **산출물의 형식과 그 실증**, 즉 [SEAL Table 2]·[SEAL Table 9]·[SEAL Table 10]의 수치가 결국 무엇을 입증하는가는 ch20이 소유한다. 이 절과 §05.6이 그 표들의 값을 인용하는 것은 기제를 설명하고 통제군의 존재를 확인하기 위해서이며, 그 값들에 대한 판정은 여기서 내리지 않는다.

SEAL의 outer loop는 진짜 RL이다. 그러나 PPO도 group-relative 방식도 아니다. 논문은 "GRPO와 PPO 같은 다양한 on-policy 방법을 실험했으나 학습이 불안정했다"고 쓰고 [SEAL §3.1, p.4], 대신 **ReST$^{EM}$**(Singh et al. 2023, arXiv:2312.06585)을 채택한다 — 논문 자신의 표현으로 "filtered behavior cloning에 기반한 더 단순한 접근, 달리 말해 rejection sampling + SFT"다 [SEAL §3.1, p.4].

세 항의 대응은 §05.1의 표준형과 다르다. **state**는 $(c, \Theta_k)$인데 관측은 $c$뿐이다 — 보상이 파라미터 자신에 의존하지만 "$\theta$를 직접 문맥에 넣는 것은 불가능하다" [SEAL §3.1, p.4]. **action**은 token 하나가 아니라 self-edit $\mathcal{R}_k$ **전체**, 즉 한 번의 완결된 생성이다. **reward**는 그 $\mathcal{R}_k$로 (U-$\Theta$)를 돌려 얻은 모델을 downstream 과제 $\mathcal{T}$에서 채점한 이진 점수다 [SEAL Eq. 2, §3.1, p.4].

세 번째 항이 이 corpus의 핵심 장치다. 채점기가 프로그램도, 사람도, reward model도 아니고 **학습 실행 그 자체**다 — finetune 한 번 + 평가 한 번이 보상 함수의 본체다. 그 실행 비용은 단위 테스트가 아니라 SFT 한 번이고, 실행의 마지막 채점은 프로그램이 아니라 GPT-4.1이 greedy decoding으로 내리는 yes/no다 [SEAL §B.4, p.20]. 게다가 한 self-edit의 점수는 3개 seed의 inner-loop 실행 위에서 매겨진다 [SEAL §B.5, p.21].

> **[평가]** 이 보상은 §05.5의 표 5-5의 어느 한 행에도 깨끗이 들어가지 않는다. 이 장의 분류가 여기서 한 칸 모자란다는 뜻이고, 정정은 이렇게 한다 — 뼈대는 두 번째 행(실행)이되 실행의 마지막 채점이 네 번째 행(모델 채점)이므로, SEAL의 보상은 **결정론적 검증기가 아니라 모델 채점을 끝에 단 학습 실행**이다. 따라서 §05.6의 첫 번째 실패 모드(학습된 채점기의 오차를 타는 것)를 물려받으며, RLVR의 "재현 가능한 허점"이라는 이점은 여기에 적용되지 않는다.

gradient 추정량은 [SEAL Eq. 3–4]이고, 이 책의 표기로 옮기면

$$
\nabla_{\Theta_k}\mathcal{L}_{\mathrm{RL}}
\;\approx\;
-\frac{1}{NM}\sum_{i=1}^{N}\sum_{j=1}^{M}
R_{ij}\;\nabla_{\Theta_k}\log \pi_{\Theta_k}\big(\mathcal{R}_{ij}\mid c_i\big)
\tag{5-4}
$$

식 (5-4)를 식 (5-1)과 나란히 놓으면 차이가 정확히 하나다: advantage 자리에 baseline을 빼지 않은 이진 보상 $R_{ij}$가 그대로 들어간다. 그 결과 $R=0$인 항은 전부 사라지고, 남는 것은 "보상 1을 받은 self-edit들에 대한 SFT"다. 이것이 filtered behavior cloning이라는 이름의 내용이며, 결정적 성질이 하나 따라온다 — **확률을 내리는 방향의 gradient가 존재하지 않는다.** 실패한 rollout은 벌받지 않고 그냥 무시된다.

정직성 주석 둘을 붙인다. 첫째, **식에 쓰인 보상과 실제로 쓴 보상이 다르다.** [SEAL Eq. 2]는 성능을 개선한 모든 self-edit에 $R=1$을 주는 임계값 규칙인데, 각주 2는 knowledge incorporation 실험에서 "샘플된 후보 중 가장 크게 개선한 단 하나의 self-edit"에만 보상을 준다고 밝힌다 [SEAL 각주 2, p.4] — 임계값이 아니라 $M$개 중 argmax다. 둘째, **보상이 $\Theta_k$에 의존하는데 그것을 무시한다.** 논문은 "보상 항이 $\theta_t$에 의존하지만 미분 가능하지 않다. 우리는 보상을 $\theta_t$에 대해 고정된 것으로 취급함으로써 이를 처리한다"고 명시한다 [SEAL §3.1, p.4].

> **[평가]** 두 번째 주석의 귀결을 이 책은 논문보다 강하게 읽는다. 식 (5-4)가 최적화하는 것은 보상에 stop-gradient를 적용한 대리 목적이지 [SEAL Eq. 1]에 쓰인 목적 그 자체가 아니다. 두 목적이 같은 최적점을 갖는다는 논증은 논문에 없다. 이 장이 SEAL의 outer loop를 "진짜 RL"이라고 부르는 근거는 $\eta_\Theta\nabla_\Theta$가 실제로 실행된다는 사실이지, 쓰인 목적함수가 그대로 최적화된다는 사실이 아니다.

비용은 채점기가 지배한다. self-edit 하나를 평가하는 데 30–45초가 걸리고 [SEAL §5 Computational overhead, p.9], 한 ReST$^{EM}$ 라운드는 50 passage × 5 completion × 3 seed = 750회의 inner-loop 실행으로 2×H100에서 약 6시간이다 [SEAL §B.5, p.21]. 그리고 논문 자신이 그 채점기를 지우는 ablation을 싣는다 — TTT 기반 보상을 GPT-4.1 루브릭 채점(4개 범주 각 1–5점 합)으로 바꾸면 47.0 대 **45.6**이고, 시간은 약 6시간 대 **약 5분**이다 [SEAL Table 9, §B.10, p.23].

> **[해설]** 이 한 줄이 systems 독자에게 이 장에서 가장 정보량이 크다. 보상을 "훈련 실행"에서 "추론 호출"로 바꾸면 sleep 라운드의 워크로드 모양이 통째로 바뀐다 — 훈련 GPU 예약이 API 호출 예산으로 내려온다. 두 설계의 품질 차이가 그 비용 차이를 정당화하는지의 판정은 ch20이 한다. 이 장이 못 박는 것은 대응 자체다: **보상의 종류가 곧 sleep 라운드가 먹는 하드웨어의 종류다.**

### 05.7.2 ReasoningBank — policy라고 쓰고, gradient는 없다

ReasoningBank은 agent를 policy 표기로 쓴다: "agent policy $\pi_L(\cdot\mid M, A)$는 backbone LLM $L$로 파라미터화되고, memory module $M$과 action space $A$에 조건화된다" [ReasoningBank §3.1]. (원문 기호를 그대로 옮긴 인용이다. 이 책의 표기로는 원문 $L$이 frozen $\Theta$이고 원문 $M$이 $E$다. 이 책에서 맨 $L$은 sequence 길이로, 맨 $M$은 한 prompt당 rollout 수로 이미 예약되어 있으므로 인용 밖에서는 원문 기호를 쓰지 않는다.) [ReasoningBank §5]는 이 루프가 "RL의 학습 동역학을 닮았다"고 서술한다.

그런데 gradient가 한 번도 계산되지 않는다. backbone은 frozen 체크포인트이고(주력 세 모델은 Vertex AI API 뒤에 있다) [ReasoningBank §4.1, App B.1], 이 논문에서 갱신되는 것은 JSON 파일 하나뿐이다 [ReasoningBank App A.2]. 부품별 대응을 세면 이렇다.

표 5-6 — ReasoningBank에서 RL 부품의 실제 정체

| RL 부품 | ReasoningBank의 대응물 | 실제로 있는가 |
|---|---|---|
| policy $\pi_\Theta$ | frozen backbone + 검색된 기억을 붙인 system prompt | 있으나 **학습되지 않음** |
| rollout | MaTTS는 같은 질의에 $m$개의 궤적을 굴린다 [ReasoningBank §3.3] | 있음 |
| reward | LLM-as-a-Judge의 이진 라벨 $j_k$, temperature 0.0 [ReasoningBank §3.2, App A.2] | 있음(단 정답 라벨 없음) |
| advantage | — | 없음 |
| policy 갱신 $\eta_\Theta\nabla_\Theta$ | 이 자리에 (U-E)가 들어가 있다 | **없음** |

> **[평가]** RL 어휘("policy", "self-evolving", "test-time learning")를 쓰면서 어떤 층의 파라미터도 움직이지 않는 것은 이 corpus에서 반복되는 서술 패턴이다. 판별 기준은 단순하다 — **$\eta_\Theta \nabla_\Theta$가 어디에선가 실제로 실행되는가.** ReasoningBank에서는 실행되지 않는다. 이것이 결함이라는 뜻은 아니다. E-경로는 그렇게 작동하도록 설계된 것이고, 그 설계가 가진 이점(체크포인트 관리도, optimizer state도, 망각 대응도 필요 없다)은 실재한다. 결함은 두 개의 다른 기제를 같은 단어로 부르는 데서 생기며, 그 구분이 ch11 판별식의 일이다.

그럼에도 ReasoningBank은 RL의 결정적 부품 하나를 진짜로 가지고 있다: **자동 채점기**. 그리고 그 채점기의 정확도가 측정되어 있다 — ground truth 대비 **72.7%**다 [ReasoningBank §5]. 이 숫자가 §05.5의 RLVR 논의와 만나는 자리다. 검증기가 결정론적 프로그램이 아니라 LLM일 때 보상 신호의 신뢰도가 얼마인지에 대한, 이 corpus에서 유일하게 측정된 값이기 때문이다. (그 견고성 실험이 실제 judge의 편향 대신 대칭 잡음으로 대체되었다는 사실은 ch15가 다룬다.)

systems 접점: ReasoningBank의 오프라인 비용은 토큰으로 보고되어 있다. 과제당 judge 2186.3 tokens + memory extraction 1562.1 tokens = **3748.4 tokens**이고, wake의 action generation은 49306.1 tokens다 [ReasoningBank Table 5, §C.2]. 즉 "채점 + 요약"이 과제당 총 토큰의 7% 남짓이다(3748.4/53054.5 = 7.07%. 이 나눗셈은 이 책의 산술이며, 논문은 비율을 이 형태로 싣지 않는다). SEAL의 self-edit 하나당 30–45초짜리 채점과 자릿수가 다르다 — 한쪽은 추론 호출 두 번, 다른 쪽은 finetune 한 번이다. **보상을 어떻게 만드느냐가 sleep 예산 $B_s$를 자릿수 단위로 가른다**는 것이 이 절의 결론이며, ch25의 통일 비용 회계가 이 자리에서 시작한다.

## (state, update, cost) 정리

이 장의 개념들은 전부 $\Theta$층에서 벌어지는 일이거나, 그 층을 움직이기 위한 보조 장치다. RL의 부품 중 어느 것도 $W$나 $E$에 놓이지 않는다 — §05.7.2에서 본 (U-E)는 policy 갱신이 있어야 할 자리에 대신 들어선 것이지 RL의 부품이 아니다. 그것이 RL을 세 층 프레임에 놓을 때의 첫 사실이다.

표 5-7 — 이 장의 개념을 세 층 프레임에 배치

| 개념 | state (어느 층인가) | update (어느 규칙인가) | cost (어느 예산인가) |
|---|---|---|---|
| policy $\pi_\Theta$ | $\Theta$ — policy는 모델 자신 | (U-$\Theta$). gradient는 식 (5-1) | rollout $M$개 decode + backward |
| rollout $z$ | 어느 층도 아님 — 소모품 | 저장되지 않고 버려짐 | $B_s$의 대부분. 길이에 선형 |
| reward model $R_\psi$ | 별도 모델의 $\Theta$ | 자체 (U-$\Theta$), 선호 데이터 위 1회 | 학습 1회 + rollout마다 forward 1회 |
| advantage $\hat A$ | 상태 아님 — 추정량의 계수 | — | baseline 방식이 value network 유무를 정함 |
| RLHF 파이프라인 | $\Theta$ (+ $\psi$) | (U-$\Theta$) 3단계 | 넷의 모델 상주. HBM이 병목 |
| DPO | $\Theta$ | (U-$\Theta$) 1회, sampling 없음 | SFT와 같은 모양. sampler·채점기 부재 |
| RLVR | $\Theta$ | (U-$\Theta$) | rollout(GPU) + 검증(GPU 밖) |
| reward hacking | 상태가 아니라 실패 모드 | — | $B_s$를 라운드마다 키움 → (A)의 분자 증가 |

Rosetta로 옮기면 한 줄이다. 독자의 세계에서 **RL 후처리 한 라운드는 "offline batch inference job이 training job을 먹여 살리는 파이프라인"**이고, 세 층 프레임에서는 (U-$\Theta$) 한 번, 즉 **배포 아티팩트가 바뀌는 사건**이다(→ ch01 §01.6 표 1-4의 "모델 재배포 ↔ (U-$\Theta$) 1회" 행).

## Worked micro-example — 한 위치의 확률이 어디로 가는가

숫자 네 개와 확률 하나로 이 장 전체를 손으로 확인한다. 계산기 없이 따라갈 수 있다. **이 예제의 설정값과 그로부터 나오는 표 5-8의 수치는 전부 이 책이 만든 예시 계산이며, 어느 논문의 보고값도 아니다.** 4단계에서만 SEAL의 보고 수치를 끌어 쓰고, 그 자리에 출처를 병기한다.

**설정.** prompt $q$ 하나에 rollout $M=4$개를 뽑았다. 채점기는 검증 프로그램이고 보상은 이진이다. 결과는 $R = (1,\,0,\,1,\,0)$. 그리고 어떤 한 decode 위치에서 두 후보 token만 살아 있다고 하자 — $p_A = 0.25$, $p_B = 0.75$. rollout이 그 위치에서 $A$를 뽑았다. sleep learning rate는 $\eta_\Theta = 0.1$.

**1단계 — 두 알고리즘의 계수를 만든다.**

group-relative 방식은 같은 prompt의 표본 평균을 baseline으로 쓴다. $\bar R = 0.5$이고 네 값의 표준편차($1/M$ 정의)는 $\sqrt{\tfrac14\cdot 4\cdot 0.25} = 0.5$이므로

$$
\hat A_i \;=\; \frac{R_i - \bar R}{0.5} \;=\; (+1,\ -1,\ +1,\ -1)
$$

ReST$^{EM}$(SEAL의 선택)은 baseline을 빼지 않고 식 (5-4)처럼 $R_{ij}$를 그대로 쓴다. 계수는 $(1,\,0,\,1,\,0)$이다. **음수가 없다.**

**2단계 — 한 위치의 logit을 옮긴다.**

두 후보만 있을 때 $\log p_A$의 logit에 대한 gradient는 $\partial\log p_A/\partial\,\mathrm{logit}_A = 1-p_A = 0.75$, $\partial\log p_A/\partial\,\mathrm{logit}_B = -p_B = -0.75$이다. 계수 $w$로 한 step 올라가면 두 logit의 차가

$$
\Delta\big(\mathrm{logit}_A - \mathrm{logit}_B\big) \;=\; \eta_\Theta\, w\,\big[(1-p_A) + p_B\big] \;=\; 0.1 \cdot w \cdot 1.5 \;=\; 0.15\,w
$$

만큼 움직인다. 초기 log-odds는 $\ln(0.25/0.75) = -1.0986$이므로 결과는 표 5-8이다.

표 5-8 — 같은 rollout, 같은 위치, 다른 알고리즘

| 그 rollout이 받은 보상 | group-relative 계수 | ReST$^{EM}$ 계수 | 새 log-odds (group) | 새 $p_A$ (group) | 새 $p_A$ (ReST$^{EM}$) |
|---|---|---|---|---|---|
| $R=1$ | $+1$ | $1$ | $-0.9486$ | **0.279** | **0.279** |
| $R=0$ | $-1$ | $0$ | $-1.2486$ | **0.223** | **0.250 (불변)** |

두 알고리즘은 성공한 rollout에 대해 정확히 같은 일을 하고, 실패한 rollout에 대해 완전히 다른 일을 한다. ReST$^{EM}$은 실패를 **벌하지 않고 버린다**. 이것이 §05.7.1에서 말한 "확률을 내리는 gradient가 존재하지 않는다"의 수치적 내용이다.

**3단계 — 퇴화 지점을 확인한다.**

같은 prompt의 rollout 4개가 **전부** $R=1$이면 어떻게 되는가. group-relative에서는 $\bar R = 1$이고 편차가 0이므로 $\hat A_i = 0$ — **gradient가 정확히 0이고 아무것도 배우지 않는다.** 실무에서 prompt를 난이도로 걸러 "전부 맞거나 전부 틀리는" 문제를 빼는 이유가 이것이다. 반면 ReST$^{EM}$에서는 계수가 $(1,1,1,1)$이 되어 네 rollout 모두 확률이 올라간다 — 즉 자기 샘플 위의 평범한 SFT로 퇴화한다. 두 방식은 극단에서 서로 다른 방향으로 무너진다.

**4단계 — 이 한 prompt의 채점 비용을 센다.**

SEAL의 채점기로 이 예제를 돌리면 self-edit 하나 평가에 30–45초이므로 [SEAL §5, p.9], $M=4$에 대해 **2–3분**이 gradient를 한 번도 계산하기 전에 소비된다(이 곱셈은 이 책의 예시 계산이다). 그중 절반($R=0$인 둘)은 식 (5-4)에서 계수 0을 받아 갱신에 전혀 기여하지 않는다 — **채점 비용의 50%가 버려진다.** 논문이 보고한 라운드 비용과 대조해 보면 자릿수가 맞는다: 750회 실행에 약 6시간이면 21600/750 = **28.8초/회**로, 명시된 30–45초와 정합적이다(이 나눗셈은 이 책의 산술이며, 논문은 초당 값을 따로 싣지 않는다) [SEAL §B.5, p.21].

이 4단계가 이 장의 요약이다. 보상이 확률을 얼마나 옮기는지는 $\eta_\Theta$와 계수가 정하고, 그 계수를 어떻게 만드느냐가 알고리즘 이름이며, 계수를 만들기 위해 채점기를 몇 번 부르느냐가 $B_s$다.

## 요약

- LLM RL에서 state는 KV cache가 담고 있는 token 열, action은 다음 token, reward는 생성이 끝난 뒤 밖에서 도착하는 스칼라 하나다. 전이는 결정론적 concat이고 관행상 discount는 쓰지 않는다.
- policy gradient는 보상을 미분하지 않는다. 미분되는 것은 $\log\pi_\Theta$뿐이며, 그래서 RL은 **미분 불가능한 채점기를 loss로 바꾸는 장치**다.
- advantage는 보상에서 baseline을 뺀 값이고, baseline을 만드는 방식(value network냐 같은 prompt의 표본평균이냐)이 RL 단계의 HBM 예산을 정한다.
- RLHF 파이프라인은 SFT → reward model → RL의 3단계이며, RL 단계에서 policy·reference·reward model·value network 넷이 동시에 상주한다. 이후 알고리즘 계보는 그 넷을 줄이는 역사다.
- DPO는 KL 정규화 최적해의 닫힌 형태를 이용해 reward model과 sampler를 함께 지운다. 워크로드 모양이 SFT와 같아 sleep 라운드에 얹기가 구조적으로 단순하며, 대가는 off-policy라는 것이다.
- RLVR는 보상을 결정론적 프로그램에 맡겨 사람 없이 임의 횟수의 보상을 만든다. sleep 라운드가 사람의 처리량에 묶이지 않으려면 이 성질이 필요하다 — 이 연결이 실제로 성립하는지의 판정은 ch20·ch21·ch27이 한다.
- reward hacking은 버그가 아니라 대리 최적화의 구조적 귀결이며, 길이 인플레이션이 가장 흔한 형태다. RL 이득 보고에는 보상과 상관된 표면 변량을 prompt만으로 재현해 보는 통제군이 있어야 한다.
- SEAL의 self-edit 루프는 ReST$^{EM}$(rejection sampling + SFT)이고 PPO·GRPO가 아니며, 실패한 rollout을 벌하는 gradient가 없다. ReasoningBank은 policy 표기와 rollout과 자동 채점기를 모두 갖추고도 $\eta_\Theta\nabla_\Theta$를 한 번도 실행하지 않는다.

## 자가 점검 체크리스트

- [ ] 임의의 LLM 학습 루프를 보고 state·action·reward를 지목하고, 그 reward가 terminal인지 dense인지 판정할 수 있다.
- [ ] 식 (5-1)에서 보상이 미분되지 않는다는 점을 지적하고, 그것이 왜 SFT 대신 RL을 쓰는 이유가 되는지 설명할 수 있다.
- [ ] $M=4$, 보상 $(1,0,1,0)$, $p_A=0.25$, $\eta_\Theta=0.1$에서 group-relative와 ReST$^{EM}$가 $p_A$를 각각 어디로 옮기는지 손으로 계산하고, 전부 정답일 때 두 방식이 어떻게 다르게 퇴화하는지 말할 수 있다.
- [ ] RLHF·DPO·RLVR를 "무엇이 채점하는가"와 "어느 하드웨어를 먹는가"의 두 축으로 구분하고, 각각을 sleep 라운드에 얹었을 때의 비용 모양을 말할 수 있다.
- [ ] 어떤 논문의 RL 이득 주장을 보고, reward hacking을 배제하는 통제군이 있는지 없는지 판정할 수 있다.
- [ ] "policy"·"self-evolving"이라고 쓰인 논문에서 $\eta_\Theta\nabla_\Theta$가 실제로 실행되는지를 확인해 진짜 RL과 어휘 차용을 구분할 수 있다.
- [ ] (Rosetta) RL 후처리 한 라운드를 독자의 어휘로 "offline batch inference job이 training job을 먹여 살리는 파이프라인"이자, 세 층 프레임에서 (U-$\Theta$) 1회 = **배포 아티팩트 교체**로 옮겨 말할 수 있다.

## 다음 장으로

이 장은 보상이 이미 있다고 가정하고, 그것으로 $\Theta$를 옮기는 법을 다뤘다. 그런데 식 (U-$\Theta$)에는 아직 채워지지 않은 자리가 하나 남아 있다 — $\mathcal{R}_k = \mathrm{gen}(\mathcal{H}_k; B_s)$의 $\mathrm{gen}$이다. SEAL에서 policy의 action이 곧 다음 gradient step이 먹을 학습 데이터였다는 사실은 이 장에서 이미 보았지만, **모델이 자기 학습 데이터를 만들 때 무엇이 좋아지고 무엇이 무너지는가**는 이 장의 도구로는 답할 수 없다. reward는 만들어진 데이터를 채점할 뿐, 그 데이터가 몇 라운드까지 자기 자신을 먹여도 되는지는 말해주지 않는다.

ch06이 그 자리를 맡는다. augmentation에서 자기생성 코퍼스까지 $\mathcal{R}$을 만드는 방법들을 정리하고, 자기 출력을 반복해서 먹었을 때의 붕괴가 (U-$\Theta$)를 반복하는 설계에 어떤 상한을 거는지 — 즉 $\rho$의 한 원천을 — 다룬다. self-edit이라는 산출물의 형식과 그 실증은 ch20이 소유한다.
