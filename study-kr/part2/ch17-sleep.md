# ch17. Sleep: Learning to Self-Modify and Consolidate Memories

## 17.1 Bridge-in: [NL]이 남긴 문제 — consolidation의 절반

Part II의 마지막 논문은 [Sleep] (*Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories*, arXiv:2606.03979; Ali Behrouz, Farnoosh Hashemi, Vahab Mirrokni; arXiv v1 2026-06-02)이다. 논문 1면의 각주는 이 작업의 한 버전이 2025년 9월부터 OpenReview에 공개되어 있었다고 명시하는데 [Sleep p.1 각주], 이 시점 주장은 §17.8에서 다룰 2026년 on-policy self-distillation 물결과의 우선권 문제에서 다시 등장한다.

이 장의 출발점은 16장이 멈춘 자리다. [NL] (*Nested Learning*, arXiv:2512.24695)은 아키텍처와 optimizer를 update frequency(→ 16장)의 단일 스펙트럼 위에 재배열했다. attention은 frequency $\infty$의 memory이고 frozen MLP는 frequency $0$의 memory이며, Continuum Memory System(CMS, → 16장)은 그 사이의 빈 구간을 chunk 주기 $C^{(\ell)}$마다 갱신되는 MLP 사슬로 채웠다. Hope(→ 16장)는 self-modifying Titans와 CMS를 결합해 이 구도를 실증했다. 그러나 [NL]이 남긴 것이 네 가지다. 첫째, **catastrophic forgetting(→ 11장)은 해결이 아니라 지연되었다.** 다중 frequency는 덮어쓰기를 미룰 뿐이고, 모든 level의 update 주기가 정렬되는 순간 CF는 일어난다 — [Sleep §2.2]가 [NL]을 인용해 이 진단을 그대로 재확인한다. 둘째, **offline consolidation — replay·sleep 계열의 기제 — 는 [NL] 스스로 명시적으로 범위 밖에 두었다**: "두 번째 단계가 동등하게, 또는 그 이상으로 중요함에도, 이 작업은 첫 단계 — online 과정으로서의 memory consolidation — 에 집중한다" [NL §2]. 셋째, **capacity가 고정이다** — 유한 크기 파라미터로의 압축인 이상 새 지식은 언젠가 옛 지식을 덮어쓴다는 이 진단은 [Sleep §1]이 online-only consolidation에 들이대는 비판이다. 넷째, **모든 적응이 입력이 흐르는 동안 일어난다** — 둘째 항목의 직접적 따름정리로, 모델이 입력을 끊고 자기 내부를 정리하는 시간이 [NL]의 설계에는 존재하지 않는다.

<!-- FIG-REF: ch16/fig-01-cms-spectrum -->

[Sleep §1]은 [NL]의 anterograde amnesia 비유를 이어받아 이 결핍을 다시 조준한다. 배포된 LLM의 지식은 두 곳에만 있다: 세션이 끝나면 소멸하는 context window, 그리고 pre-training 종료 시점에 동결된 MLP·projection weights. 단기 기억과 장기 기억 사이를 잇는 경로가 없으므로, 모델은 새 장기 기억을 형성하지 못하는 환자처럼 "영원한 현재"를 산다. [NL]의 CMS는 이 경로의 절반 — 깨어 있는 동안 fast block에서 slow block으로 지식이 end-to-end로 흘러가는 **online consolidation**(개념은 → 11장, [NL]의 기법은 → 16장) — 을 놓았다. [Sleep §1]은 online-only consolidation의 구조적 결함을 셋으로 정리한다. (1) **추상화 수준이 그대로다**: 추가적인 lossy 압축 없이 같은 표현 수준의 지식을 옮기므로 capacity를 그만큼 다시 소모한다. (2) **선택적이고 retrieval 의존적이다**: 활발히 recall되는 기억만 강화된다. (3) **context에 갇혀 있다**: update가 현재 context에서만 유도되므로, 새 지식과 기존 지식의 상위 수준 통합이 일어나지 않는다.

[Sleep]의 답은 스펙트럼에 축을 하나 더 놓는 것이다 — 수식의 축이 아니라 시간의 축. 인간 기억 연구가 말하는 두 번째 consolidation, 즉 수면 중의 offline consolidation을 LLM의 lifecycle에 이식한다. 이 장에서 보겠지만, 그 이식은 [TNT] (*TNT: Improving Chunkwise Training for Test-Time Memorization*, arXiv:2511.07343)의 reset 장치와 [NL]의 chunk 스케줄을 재료로 쓰되, 라인의 앞 다섯 논문과는 방법론적으로 결이 다른 물건이 된다.

## 17.2 문제의식과 논문의 핵심 주장

논문이 스스로 정의한 문제는 배포 후 정적인(static-after-deployment) LLM이다 [Sleep §1]. 지식 갱신의 기존 처방은 딜레마의 두 뿔에 각각 걸린다: re-pretraining은 효과적이지만 잦은 갱신에는 비용이 불가능한 수준이고, continual fine-tuning이나 LoRA류 경량 갱신은 반복 적용 시 catastrophic forgetting을 부른다 [Sleep §1]. in-context learning은 효율적인 continual learning이지만 context가 끝나면 지식이 소멸한다. 그래서 질문은 "fragile한 short-term memory를 어떻게 stable한 long-term 지식으로 옮기는가"가 된다 [Sleep §1].

<!-- FIG: ch17/fig-01-wake-sleep-lifecycle -->

핵심 주장은 셋이다. 첫째, **continual learner에게는 training time도 test time도 없다.** 모델의 lifecycle은 새 입력을 받아 처리하는 **wake(active) phase**와, 입력을 최소화하거나 끊고 내부 계산으로 기억을 정리하고 자기를 개선하는 **sleep phase**의 주기적 교대로 재정의되어야 한다 [Sleep §3.1]. 이 책은 이것을 **wake/sleep lifecycle**이라고 부른다 — 이 라인이 Titans 이후 유지해 온 "test time"이라는 단어의 마지막 잔재를 지우는 주장이다. 둘째, **CF는 근본적으로 capacity 문제다.** 파라미터 수가 유한하므로 새 지식을 넣으려면 덮어써야 하고, 그래서 잊는다. 논문은 생물학의 offline consolidation을 replay(수면 중 최근 pattern의 재생)와 neuroplasticity(새 연결의 형성)의 **결합**으로 읽고, 그 처방으로 replay 기반 Knowledge Seeding에 점진적 **parameter expansion**을 함께 쓴다 [Sleep §3.2, §3.3](regularization(EWC류, → 11장)만 이 해법군에서 뺀다). 셋째, sleep은 두 단계다: NREM(slow-wave sleep)의 hippocampus→neocortex 기억 이전과 synaptic homeostasis에 대응하는 **Memory Consolidation**, 그리고 REM의 시냅스 강화·통합·미래 시뮬레이션에 대응하는 **Dreaming** [Sleep §1, §3].

> **[해설]** 이 책의 좌표로 옮기면 이렇게 된다. 12–16장의 논문들은 전부 master update (M)의 성분을 바꿨다 — [Miras] (*It's All Connected*, arXiv:2504.13173; 이 책은 framework 이름 Miras로 통칭한다)는 objective를, [Atlas] (*Atlas: Learning to Optimally Memorize the Context at Test Time*, arXiv:2505.23735)는 window와 optimizer를, [TNT]는 훈련 경제학을, [NL]은 층위를 바꿨다. [Sleep]은 (M)을 건드리지 않는다. 바꾸는 것은 그 update들이 살아가는 **lifecycle**이다. inference 어휘로는 1장 Rosetta의 마지막 행이 이 장의 전부다: sleep phase는 서빙 fleet에 붙는 주기적 백그라운드 job, "weights의 background compaction"이다. 낮에는 요청을 처리하며 fast memory에 쓰고, 밤에는 트래픽을 끊고 compaction·GC를 돌린다 — 단지 그 대상이 로그나 cache가 아니라 모델의 파라미터일 뿐이다.

비슷한 이름의 선행물과의 경계도 논문이 직접 긋는다. sleep-time compute(Lin et al. 2025)는 유휴 시간에 과거 상호작용의 **텍스트 요약**을 만들고, Cartridges(Eyuboglu et al. 2025)는 KV 표현을 보조 모델로 압축한다 — 둘 다 token/KV 공간의 압축이다. [Sleep]은 자기생성 데이터를 통한 distillation으로 지식을 **parametric weight 공간**으로 옮긴다는 점에서 다르다고 주장한다 [Sleep §2.3, App. A.2–A.3]. 이 구분의 실증은 §17.6의 Cartridges 비교가 담당한다.

## 17.3 Core mechanism (통일 표기)

이 절은 sleep 한 사이클을 재구현 가능한 수준까지 전개한다. 구조는 기제의 실행 순서를 따른다: wake의 기반 구조(CMS) → sleep의 발화 시점 → Stage 1 Memory Consolidation(expansion → Knowledge Seeding → Learning to Imitate → reset) → Stage 2 Dreaming.

### 17.3.1 기반 구조: CMS와 식 (M5) — wake가 돌리는 것

[Sleep]은 새 sequence layer를 제안하지 않는다. 기반은 [NL]에서 그대로 수입한 CMS다(정의 소유는 16장; 여기서는 이 장에 필요한 만큼만 환기한다). backbone의 한 블록은 sequence-mixing layer(attention이든 Titans류 memory module이든) 뒤에 MLP 블록 사슬 $\mathrm{MLP}^{(f_1)},\dots,\mathrm{MLP}^{(f_k)}$가 이어지는 구조이고, 각 블록은 자기만의 update frequency $f_\ell$을 가진다 [Sleep §2.2, Eq. 1]. 블록 $\ell$의 chunk 크기는 $C^{(\ell)} := \max_{\ell'} C^{(\ell')}/f_\ell$로 정의되며, 일반성을 잃지 않고 $C^{(\ell)}$은 $C^{(\ell-1)}$로 나누어떨어진다고 가정한다 [Sleep §2.2]. 블록 $\ell$의 파라미터 $\theta^{(\ell)}$은 $C^{(\ell)}$ step마다 한 번 갱신된다. 통일 표기의 (M5) 그대로다:

$$
\theta^{(\ell)}_{i+1} = \theta^{(\ell)}_{i} - e_{i,\ell},
\qquad
e_{i,\ell} = \begin{cases}
\displaystyle\sum_{t=i-C^{(\ell)}+1}^{i} \eta^{(\ell)}_t\, \varepsilon\big(\theta^{(\ell)}_t;\, x_t\big) & i \equiv 0 \pmod{C^{(\ell)}}\\[2pt]
0 & \text{otherwise}
\end{cases}
\tag{17-1}
$$

여기서 $\eta^{(\ell)}_t$는 level $\ell$의 learning rate이고, $\varepsilon(\cdot;\cdot)$은 임의 optimizer의 error 항 — gradient descent라면 task objective의 gradient $\nabla_\theta\mathcal{L}(\theta^{(\ell)}_t;x_t)$ — 이다 [Sleep Eq. 2]. 원문은 이 error 항에 $f(\cdot)$ 기호를 쓰지만 frequency $f_\ell$과 충돌하므로 이 책은 $\varepsilon$으로 개명한다(표 17-1). 식 (17-1)이 말하는 것: 각 블록은 token마다 error 기여분을 **누적**하고, 자기 chunk 경계에서만 그 누적을 파라미터에 **적용**한다. chunk 내부에서 $\theta^{(\ell)}$이 상수이므로 chunk 전체를 병렬 처리하고 update 한 번을 미루는 chunkwise 구조(→ 9장)가 level마다 성립하며, 이것이 [TNT]가 훈련 경제학으로 다뤘던 바로 그 구조다. 해석도 [NL]의 것을 잇는다: $\theta^{(\ell)}$은 자기 chunk의 context를 파라미터로 압축한 "그 문맥의 추상적 지식"이고, 앞쪽(고주파) 블록은 short-term memory, 뒤쪽(저주파) 블록은 long-term memory다 [Sleep §2.2].

이 구조의 급소가 [Sleep]의 출발점이다. 다중 frequency는 CF를 미루지만, 식 (17-1)의 갱신 자체가 덮어쓰기다 — 블록의 update 순간, 직전 chunk가 만든 옛 내용 위에 새 내용이 씌워진다. 그러므로 **각 memory 블록의 파라미터가 갱신되기 전에, 그 블록의 지식을 더 안정된 파라미터로 consolidate하는 기제**가 필요하다는 것이 논문의 핵심 관찰이다 [Sleep §2.2].

### 17.3.2 sleep은 언제 오는가: chunk 경계 스케줄

sleep의 발화 시점은 학습되지 않고 chunk 스케줄에 고정된다. chunk 길이 목록 $\{C^{(1)},\dots,C^{(k)}\}$가 주어지면, sleep(그리고 memory consolidation)은 모든 $b\in\mathbb{N}$에 대해 step $\{C^{(1)}\times b,\dots,C^{(k)}\times b\}$에서만 일어난다 [Sleep §3.2]. 즉 어떤 블록이든 자기 갱신 경계에 도달하면, 갱신 직전에 그 블록의 지식이 다음 느린 블록으로 먼저 옮겨진다. frequency가 중첩되어 있으므로 consolidation은 다대일이다: update frequency 1K token의 블록 뒤에 10K token의 블록이 있으면, 느린 블록이 한 번 갱신되는 동안 빠른 블록은 10번 갱신되고, 따라서 빠른→느린 consolidation이 10번 일어난다 [Sleep §3.2]. 실험 구성의 chunk/update-period 사다리는 $C=$1k→5k→10k token이다 [Sleep Fig. 7] — frequency(단위 시간당 갱신 횟수)는 그 역수라 반대 방향으로 감소한다(1k 블록이 10k 블록보다 자주 갱신된다). 같은 고정 크기의 느린 memory에 10번을 반복해서 써 넣는 것 — 바로 그 지점이 CF가 일어날 자리이고, 그래서 다음 소절의 expansion이 필요해진다.

### 17.3.3 Stage 1a — parameter expansion: 덮어쓰지 말고 키워라

**parameter expansion(periodic parameter (de)activation)** 은 consolidation을 받는 쪽 블록에 새 파라미터를 열어 주는 기제다. 일반성을 잃지 않고 각 $\mathrm{MLP}^{(f_\ell)}$은 router $\mathcal{R}^{(\ell)}$을 가진 sparse mixture-of-experts(MoE)라고 가정한다(MoE/router → 10장 용어집): 블록 $\ell$은 현재 $s_\ell\ge 1$개의 expert $\{W^{(f_\ell),1},\dots,W^{(f_\ell),s_\ell}\}$을 가진다 [Sleep §3.2]. 블록 $\ell^*{-}1$의 지식을 바로 다음 느린 블록 $\ell^*$로 consolidate할 때, 전이되는 지식과 $\mathrm{MLP}^{(f_{\ell^*})}$에 이미 저장된 지식의 간섭을 피하기 위해 **새 low-rank expert 하나를 추가**한다: $A^{(\ell^*),\,s_{\ell^*}+1}\in\mathbb{R}^{d\times d_{\mathrm{low}}}$, $B^{(\ell^*),\,s_{\ell^*}+1}\in\mathbb{R}^{d_{\mathrm{low}}\times d}$, $d_{\mathrm{low}}\ll d$로 파라미터화된 low-rank MLP다 [Sleep §3.2]. 전이되는 지식은 오직 이 새 파라미터($2\,d\,d_{\mathrm{low}}$개)에만 저장되고, 그 결과 sleep이 지날 때마다 일부 layer의 파라미터가 자란다.

구현 노트가 infra 독자에게 중요하다 [Sleep §3.3 "Note on the Implementation"]. tensor 차원을 동적으로 바꾸는 대신, **미래에 열릴 expert 전부를 초기화 시점에 미리 할당해 두고 forward·backward에서 mask**한다. sleep에서 "expert를 추가한다"는 것은 mask를 여는 것이다. shape가 정적이므로 컴파일된 그래프·kernel autotuning·checkpoint 포맷이 흔들리지 않는다. 논문은 이것을 뇌의 고정 capacity와 "새 연결 형성에 의한 뉴런 활성화"에 대응시킨다.

### 17.3.4 Stage 1b — Knowledge Seeding: 위로 가는 distillation

**Knowledge Seeding(KS)** 은 이 논문이 이름 붙인 새로운 knowledge transfer의 방향이다: 하나 이상의 **작은** 모델(teacher)의 지식을 **큰** 모델(student)로 옮기는 distillation — 통상의 KD(→ 11장)와 반대 방향이라 논문은 이를 upward distillation이라고도 부른다 [Sleep §3.3]. **Self-Knowledge Seeding(SKS)** 은 teacher와 student가 같은 모델의 두 버전인 특수 사례다. teacher $\mathrm{LM}_{\theta}$는 **expansion 전**의 모델 상태이고, student $\mathrm{LM}_{\theta_{\mathrm{exp}}}$는 (i) parameter expansion과 (ii) 식 (17-1)에 따른 $\theta^{(\ell^*-1)}$의 갱신, 둘 다를 거친 후의 상태다 [Sleep §3.3]. 이 정의가 기제의 요체이므로 풀어 쓴다: teacher는 빠른 블록이 아직 덮어써지기 **전**의 지식을 들고 있고, student는 빠른 블록이 이미 새 chunk로 넘어간 대신 느린 블록에 빈 expert 하나를 새로 받았다. distillation은 student의 출력 분포를 teacher 쪽으로 당기는데, student에서 움직일 수 있는 파라미터는 새 expert뿐이다. 따라서 최적화가 성공하면, 빠른 블록에서 사라진 지식이 느린 블록의 새 expert 안에 재구성된다 — 이것이 "갱신 전에 consolidate한다"의 정확한 수학적 형태다.

이 distillation에는 통상의 KD와 다른 두 난점이 있다 [Sleep §3.3]. (1) student가 teacher보다 capacity가 **크다**. teacher가 미리 생성해 둔 고정 데이터셋으로 student를 supervised 학습(sequence-level KD, Kim & Rush 2016)시키면 student 파라미터가 과소 활용된다. (2) 모델은 잠들어 있다 — 외부 데이터셋이 없으므로, 보유 corpus 위에서 teacher logits을 맞추는 고전적 Hinton식 KD를 쓸 수 없다. 해법은 Generalized Knowledge Distillation(GKD, Agarwal et al. 2024; → 11장)이다: teacher가 생성한 데이터와 student가 스스로 생성한 on-policy 데이터를 섞는다. 먼저 teacher $\mathrm{LM}_{\theta}$에서 sampling해 dataset $\mathcal{D}$를 만들고, on-policy distillation objective를 세운다 [Sleep §3.3]:

$$
\mathcal{L}_{\mathrm{GKD}}(\theta,\theta_{\mathrm{exp}})
= (1-\lambda_{\mathrm{on}})\,\mathbb{E}_{(x,y)\sim\mathcal{D}}\!\big[\mathcal{F}\big(\mathrm{LM}_{\theta}\,\big\|\,\mathrm{LM}_{\theta_{\mathrm{exp}}}\big)(y|x)\big]
\;+\;
\lambda_{\mathrm{on}}\,\mathbb{E}_{x\sim\mathcal{D}}\,\mathbb{E}_{y\sim \mathrm{LM}_{\theta_{\mathrm{exp}}}(\cdot|x)}\!\big[\mathcal{F}\big(\mathrm{LM}_{\theta}\,\big\|\,\mathrm{LM}_{\theta_{\mathrm{exp}}}\big)(y|x)\big]
\tag{17-2}
$$

$\mathcal{F}$는 teacher와 student의 token 출력 분포 사이 divergence이고(구체적 선택 — forward/reverse KL 등 — 은 GKD의 자유도다, → 11장), $\lambda_{\mathrm{on}}\in[0,1]$은 student가 자기 생성 rollout 위에서 teacher logits의 token 단위 feedback을 받는 on-policy 비중을 정한다(원문 기호 $\lambda$; weight decay와의 충돌을 피해 개명, 표 17-1). 두 번째 항이 난점 (1)의 답이다: student는 **자기가 생성한 sequence 위에서** 교정을 받으므로, teacher의 고정 샘플 분포에 갇히지 않는다.

여기에 의도적 제약이 둘 붙는다 [Sleep §3.3]. 첫째, **student의 sampling 분포를 통해서는 backpropagate하지 않는다** — 안정성과 속도를 위한 선택이다(sampling이라는 이산 연산을 미분하는 REINFORCE류 추정의 분산을 피한다; → 11장). 둘째, **student의 모든 파라미터를 동결하고 새로 확장된 expert만 학습한다.** 전이된 지식이 옛 지식을 물리적으로 덮어쓸 수 없으므로, CF 방지가 regularization이 아니라 **구조**로 보장된다. 12–16장의 어휘로 말하면, retention gate(→ 13장)가 "얼마나 지울지"를 배우는 soft한 장치였다면 이것은 "지울 수 없게 만드는" hard한 장치다.

### 17.3.5 Stage 1c — Learning to Imitate: 아는 것과 쓰는 것은 다르다

distillation만으로는 부족하다는 것이 논문의 경험적 관찰이다: student가 지식에 접근할 수 있게 되었음에도 그것을 **쓰는** 법은 배우지 못해서, teacher의 sampling 행동과 성능을 약하게만 모방한다 [Sleep §3.3]. **Learning to Imitate(LTI)** 는 이를 교정하는 RL 단계다. 7장의 LTI(linear time-invariant)와 무관한 약어다. teacher가 생성한 데이터 $\mathcal{D}_T=\{d^{(1)},\dots,d^{(n)}\}$에서 각 $d^{(i)}$의 random prefix를 뽑아 student에게 이어 쓰게 하고, student의 완성 $\hat d^{(i)}$에 보상을 준다 [Sleep Eq. 3]:

$$
r\big(\hat d^{(i)};\,d^{(i)}\big) = \rho\; r_{\mathrm{sem}}\big(\hat d^{(i)};\,d^{(i)}\big) + (1-\rho)\; r_{\mathrm{abs}}\big(\hat d^{(i)};\,d^{(i)}\big)
\tag{17-3}
$$

혼합 계수 $\rho$(원문 $\gamma$; window gate와의 충돌을 피해 개명)가 두 보상을 섞는다. semantic 보상 $r_{\mathrm{sem}}$은 **frozen reward model**이 $\hat d^{(i)}$와 $d^{(i)}$의 의미가 같으면 1, 다르면 0을 주는 이진 신호다. absolute 보상 $r_{\mathrm{abs}}$는 Levenshtein distance $z(\cdot,\cdot)$ 기반의 token 수준 유사도다 [Sleep Eq. 4]:

$$
r_{\mathrm{abs}} =
\begin{cases}
1 - \dfrac{z(\hat d^{(i)},\, d^{(i)})}{\max\{|\hat d^{(i)}|,\,|d^{(i)}|\}} & z(\hat d^{(i)}, d^{(i)}) \le z_0,\\[6pt]
0 & \text{otherwise},
\end{cases}
\tag{17-4}
$$

$z_0$는 유사도 threshold다. LTI를 식 (17-2)의 on-policy distillation과 결합한 최종 **Knowledge Seeding objective**는 확장 파라미터에 대해 다음을 최대화한다 [Sleep §3.3]:

$$
\mathcal{L}_{\mathrm{KS}}(\theta,\theta_{\mathrm{exp}})
= \mathbb{E}_{x\sim\mathcal{D}}\Big[(1-\lambda_{\mathrm{KD}})\,\mathbb{E}_{y\sim \mathrm{LM}_{\theta_{\mathrm{exp}}}(\cdot|x)}\!\big[r(y)\big]
\;-\;
\lambda_{\mathrm{KD}}\,\mathbb{E}_{y\sim \mathrm{LM}_{\theta_{\mathrm{exp}}}(\cdot|x)}\!\big[\mathcal{F}\big(\mathrm{LM}_{\theta}\,\big\|\,\mathrm{LM}_{\theta_{\mathrm{exp}}}\big)(y|x)\big]\Big]
\tag{17-5}
$$

$\lambda_{\mathrm{KD}}\in[0,1]$(원문 $\alpha$; retention gate와의 충돌을 피해 개명)이 distillation 강도(divergence 벌점)와 LTI 보상 사이를 조율한다. consolidation이 끝나면 마지막 동작이 온다: **빠른 블록 $\mathrm{MLP}^{(f_{\ell^*-1})}$에 과거 sleep들에서 추가되었던 low-rank expert 전부를 reset**해 그 capacity를 미래의 wake를 위해 되돌린다 [Sleep §3.3]. 이 책은 이를 **synaptic-pruning reset**이라 부른다 — 뇌가 불필요·중복 연결을 쳐내는 synaptic pruning의 대응물이라는 논문의 해석을 따른 명명이다. 순 효과는 하나의 불변식이다: 빠른 memory는 작고 plastic하게 유지되고, 느린 memory는 (미리 할당된 한도 안에서) 단조 성장하며 오직 신선한 low-rank expert를 통해서만 쓰인다.

> **[해설]** reset의 계보를 짚어 둘 가치가 있다. [TNT]의 periodic reset(→ 15장)은 local memory를 $W_{\mathrm{init}}$으로 되돌려 sequential chain을 끊는 **throughput 장치**였다. [Sleep]의 reset은 "지식을 위로 옮긴 다음에만 지운다"는 **memory-hygiene 원리**다. 같은 연산이 훈련 경제학의 도구에서 기억 체계의 규율로 승격되었다. inference 어휘로: eviction 전에 반드시 상위 계층으로 write-back하는 cache 정책이다.

### 17.3.6 Stage 2 — Dreaming: 자기 생성 데이터에 의한 self-modification

**Dreaming**은 REM 대응 단계로, 모델이 자기 성능을 올려 줄 합성 데이터(dream)를 **스스로 생성하는 법을 배우는** self-modification 과정이다 [Sleep §3.4]. 자리 자체는 플러그인이다 — 어떤 self-improvement용 합성 데이터 기법도 이 단계에 꽂을 수 있다. proof of concept는 SEAL(Zweiger et al. 2025) 위에 세워졌다. SEAL은 모델이 "self-edit"(자기 갱신용 합성 텍스트)를 생성하고, 그것으로 자신을 fine-tune한 결과가 좋으면 그 생성을 RL로 강화하는 loop다. 논문은 continual 설정에서 SEAL의 문제 셋을 지적한다 [Sleep §3.4]: (1) inner-loop의 SFT 비용 때문에 self-edit 수가 적다. (2) sleep 주기마다 반복되는 self-improvement는 CF를 부른다. (3) sampling이 모델의 기존 지식 분포 안에서만 이루어진다 — dreaming의 핵심 역할은 기억들의 **새로운 합성**의 탐색인데도.

절차는 다섯 단계다. task $(x_{\mathrm{ctx}},\tau)$ — task 관련 정보를 담은 context $x_{\mathrm{ctx}}$(원문 기호 $C$; chunk 크기와의 충돌을 피해 개명)와 downstream 성능 측정 $\tau(\cdot)$ — 가 주어졌을 때:

1. **생성**: $x_{\mathrm{ctx}}$를 context로 $m\ge 1$개의 dream을 뽑는다: $\{\mathrm{DREAM}^{(i)}\}_{i=1}^m \sim \mathrm{LM}_{\theta}(\cdot|x_{\mathrm{ctx}})$. 이때 sampling 중 **모든 MoE router가 정규 dispatch에 더해 random expert 하나를 추가로 활성화**한다 — 무관한 지식을 의도적으로 섞어, 모델의 시야에 숨어 있던 기억 간 패턴을 표면화하는 novelty 주입이다 [Sleep §3.4]. SEAL 문제 (3)의 답이 이 randomized routing이다.
2. **선별**: gradient 기반 data selection(G-DIG·GREATS 계열 문헌에서 차용)으로 대부분의 dream을 기각한다. 각 dream의 중요도 점수 $\omega^{(i)}$는 language-modeling objective의 gradient $g_{\mathrm{DR}}^{(i)} = \nabla_{\theta}\, \mathcal{L}_{\mathrm{SFT}}(\mathrm{DREAM}^{(i)}, \theta)$에서 얻고, 점수 상위 Top-$k$에 다양성을 위한 random 샘플 $n_{\mathrm{rand}}$개(원문 기호 $b$; sleep 스케줄 배수와의 충돌을 피해 개명)를 더해 선별 집합 $\mathrm{D}$를 만든다 [Sleep §3.4]. 원문은 점수를 "objective의 gradient"라고만 정의하고 스칼라화 방법(norm 등)을 명시하지 않는다.
3. **적용**: 각 $\mathrm{DREAM}^{(i)}\in\mathrm{D}$마다 모델의 **isolated instance**를 LoRA(→ 11장)로 fine-tune한다: $\theta'^{(i)} \leftarrow \mathrm{SFT}(\theta^{(i)}, \mathrm{DREAM}^{(i)})$.
4. **보상**: dream의 **생성**에 보상을 준다 — fine-tune된 $\mathrm{LM}_{\theta'^{(i)}}$가 $\tau$에서 $\mathrm{LM}_{\theta^{(i)}}$보다 개선되면 1, 아니면 0 [Sleep Eq. 5].
5. **강화**: 이 보상으로 dream 생성 policy를 ReST^EM(Singh et al. 2024)으로 최적화한다 — 생성→이진 보상으로 필터→생존 샘플에 SFT→반복하는, value network가 필요 없는 EM 계열 RL이다(SEAL과 동일한 선택; → 11장).

두 stage의 **순서**가 설계의 논점이다. consolidation이 먼저 와서, 갓 습득된 fragile한 지식을 새로 확장된 저주파 파라미터에 격리해 둔다(전체 파라미터 동결은 KS 최적화에만 적용된다). 그러면 dreaming의 반복적 self-training이 가하는 weight update(dreaming은 별도 LoRA SFT로 모델을 바꾼다; SEAL 문제 (2), §17.8의 OPSD 계열 collapse·forgetting)에 의한 그 지식의 forgetting **위험이 줄어든다** — 논문은 침식이 구조적으로 불가능하다고 보장하지 않고, 두 단계 순서가 더 robust하다는 가설·실험 결과를 제시한다 [Sleep §3.4, App. A.4].

### 17.3.7 한 번의 sleep step: 전체 절차

블록 $\ell^*{-}1$의 chunk 경계 $i \equiv 0 \pmod{C^{(\ell^*-1)}}$에서, 재구현 관점의 전체 절차는 다음과 같다.

1. teacher 스냅샷 확보: 현재 상태를 $\mathrm{LM}_{\theta}$로 고정한다.
2. expansion: 블록 $\ell^*$에 masked pool로부터 새 low-rank expert $\{A,B\}$를 활성화한다.
3. 빠른 블록 갱신: $\theta^{(\ell^*-1)}$에 식 (17-1)의 누적 update를 적용한다. 이 시점의 모델이 student $\mathrm{LM}_{\theta_{\mathrm{exp}}}$다.
4. corpus 생성: teacher에서 sampling해 $\mathcal{D}$(distillation용)와 $\mathcal{D}_T$(LTI용 dream)를 만든다.
5. Knowledge Seeding: 새 expert만 학습 대상으로 식 (17-5)를 최적화한다 — GKD 항은 식 (17-2), LTI 항은 식 (17-3)–(17-4).
6. synaptic-pruning reset: 블록 $\ell^*{-}1$에 과거 sleep들이 추가했던 expert들을 reset한다.
7. Dreaming: task $(x_{\mathrm{ctx}},\tau)$가 주어져 있으면 §17.3.6의 5단계 loop를 돌린다.
8. wake 재개: 갱신된 모델이 다음 chunk의 입력을 받는다.

### 17.3.8 표기 대응표

표 17-1 — [Sleep] 원 표기와 이 책의 통일 표기 대응 (상단은 전역 확정분, 하단은 장-국소 추가분).

| Sleep 원 표기 | 통일 표기 | 의미 / 주의 |
|---|---|---|
| $\boldsymbol\theta^{(f_\ell)}$, $C^{(\ell)}$, $f_W$ | $\theta^{(\ell)}$, $C^{(\ell)}$, $f_\ell$ | (M5) |
| $e_{i,\ell}$ — 누적 error 항 | $e_{i,\ell}$ | 동일 |
| $L$/$T$ — sequence 길이 혼용 | $L$ | |
| $\mathrm{LM}_{\boldsymbol\theta}$ (teacher), $\mathrm{LM}_{\boldsymbol\theta_{exp}}$ (student) | 동일 | teacher = 확장 전 자기 자신 |
| $\lambda$ — GKD on-policy 비율 | $\lambda_{\mathrm{on}}$ | ⚠ weight decay $\lambda$와 구분 |
| $\gamma$ — reward 혼합 계수 (Eq. 3) | $\rho$ | ⚠ window gate $\gamma$와 충돌 방지 |
| $\alpha$ — distill-vs-reward 계수 ($\mathcal{L}_{KS}$) | $\lambda_{\mathrm{KD}}$ | ⚠ retention gate $\alpha_t$와 충돌 방지 |
| $\mathcal{F}$, $\mathcal{D}$ — divergence | 동일 | |
| $r_{sem}, r_{abs}$; $z(\cdot,\cdot)$, $z_0$ — 보상·Levenshtein | 동일 | |
| $\mathbf{A},\mathbf{B}$, $d_{low}$ — low-rank expert | $A^{(\ell),j},B^{(\ell),j}$, $d_{\mathrm{low}}$ | |
| $s_\ell$ — expert 수 | 동일 | |
| $\mathcal{R}^{(f_\ell)}$ — MoE router | $\mathcal{R}^{(\ell)}$ | |
| $\boldsymbol\omega^{(i)}$, $g^{(i)}_{DR}$ — dream 중요도·gradient | 장-국소 유지 | |
| DREAM$^{(i)}$, KS/SKS/LTI, ReST^EM, SEAL | 고유명사 유지 | |
| $f(\cdot\,;\cdot)$ — Eq. 2의 optimizer error 항 | $\varepsilon(\cdot\,;\cdot)$ | ⚠ frequency $f_\ell$과 충돌 방지 (장-국소 개명, (M5)) |
| $C$ — Dreaming의 task context ($(C,\tau)$의 $C$) | $x_{\mathrm{ctx}}$ | ⚠ chunk 크기 $C$와 충돌 방지 (장-국소 개명) |
| $\mathcal{D}(\cdot\Vert\cdot)$ — $\mathcal{L}_{KS}$ 안의 divergence | $\mathcal{F}(\cdot\Vert\cdot)$ | ⚠ dataset $\mathcal{D}$와 충돌 방지 (장-국소 개명) |
| $b$ — Top-$k$ 외 추가 random dream 수 | $n_{\mathrm{rand}}$ | ⚠ sleep 스케줄 배수 $b\in\mathbb{N}$ [Sleep §3.2]와 구분 |
| Eq. 2 합의 하한 $t=i-C^{(\ell)}$ | $t = i-C^{(\ell)}+1$ | 원전 Eq. 2의 off-by-one **교정**(같은 상한 $i$·1-based에서 두 합은 한 항 차이라 수학적으로 동일하지 않다; 원전은 $C^{(\ell)}{+}1$개 항을 더해 첫 경계 $i{=}C^{(\ell)}$에서 없는 $x_0$를 포함). 이 책은 (M5) 기준 |
| $\theta'^{(i)}$ — dream $i$로 fine-tune된 instance | 동일 | 장-국소 |
| $m$ — dream 생성 수; Top-$k$의 $k$ | 동일 | 장-국소($k$는 보조 인덱스 용법) |

## 17.4 Outer-loop training vs inner-loop test-time learning — 그리고 세 번째 regime

이 절이 이 장에서 가장 주의 깊게 읽어야 할 절이다. [Sleep]은 라인에서 유일하게 outer/inner 이분법 **자체를 거부**하는 논문이다 — "continual learner에게 train/test는 없다"가 곧 논문의 제1주장이기 때문이다. 그러나 이 책의 좌표계는 여전히 유효하며, 오히려 이 논문에서 좌표계를 대야만 보이는 사실이 있다: [Sleep]에는 최적화 regime이 둘이 아니라 **셋** 있고, 그중 셋째는 앞 다섯 논문 어디에도 없던 종류다.

표 17-2 — [Sleep]의 세 최적화 regime.

| | regime 1: backbone pre-training | regime 2: wake (배포 중, 입력 흐름) | regime 3: sleep (배포 중, 입력 차단) |
|---|---|---|---|
| 이 책의 좌표 | outer loop | inner loop | 어느 쪽도 아님 — 배포 중의 offline 훈련 job |
| 움직이는 것 | $\Theta$ 전체 (projection, backbone, CMS 초기 상태) | $\theta^{(\ell)}$ (각 level의 파라미터), sequence layer의 state | consolidation: 새 expert $\{A,B\}$만. dreaming: LoRA adapter + dream 생성 policy |
| 갱신 규칙 | AdamW류 표준 훈련 (→ 2장) | 식 (17-1): error 누적 + chunk 경계 적용 | 식 (17-5) 최적화; LoRA SFT; ReST^EM |
| 데이터 | pre-training corpus | 들어오는 context 그 자체 | consolidation corpus는 자기 생성(teacher sampling, rollout); dream은 외부/wake 보존 task context $x_{\mathrm{ctx}}$와 평가 함수 $\tau$에 조건화 |
| cadence | 배포 전 1회 | token마다 누적, $C^{(\ell)}$마다 write | chunk 경계 $\{C^{(\ell)}\times b\}$마다 |

**regime 1 — 무엇이 meta-learn되는가.** [Sleep] 자체는 backbone 훈련을 새로 유도하지 않고 상속한다. Hope를 [NL] 레시피로 pre-train하는 구성이라면 [NL]의 훈련을 그대로 쓰게 된다: gradient가 식 (17-1)의 다중 frequency 갱신을 **관통**해 흐르고, 따라서 outer 최적화는 "각 level의 chunk 단위 자기 갱신이 context를 유용하게 압축하도록" 블록들을 조형한다 — TTT·Titans의 meta-learning-through-inner-loop 구조(→ 4장, 12장)를 frequency 사슬 전체로 일반화한 것이다. 병렬화의 고리도 같다: chunk 안에서 $\theta^{(\ell)}$이 상수이므로 chunk 전체가 한 번의 지연 update로 배치 처리되고(→ 9장의 stale-snapshot 근사), 이 chunkwise 훈련을 경제적으로 만드는 것이 바로 [TNT]의 주제였다(→ 15장). 반면 Llama·Qwen 위의 graft 실험이라면 regime 1은 그냥 기성 checkpoint다 — 아래 caveat 참조.

**regime 2 — wake에서 무엇이 어떤 규칙으로 움직이는가.** 배포된 모델의 상태는 두 겹이다. sequence layer는 자기 관행대로 state를 유지한다(attention이면 KV cache, Titans류 module이면 고정 크기 $W_t$). 그 위에서 모든 CMS 블록 $\ell$이 token마다 error 기여 $\eta^{(\ell)}_t\,\varepsilon(\theta^{(\ell)}_t;x_t)$를 누적하고, $C^{(\ell)}$ token마다 한 번 자기 weights에 적용한다. per-token 비용은 블록마다 (파라미터 크기의 누적 1회/token) + (파라미터 write 1회/$C^{(\ell)}$ token)이고, error 항 계산을 위한 backward류 연산이 필요하다. state 크기는 "각 블록의 파라미터 + 같은 크기의 accumulator"다. chunk/update-period 사다리가 $C=$1k→5k→10k token이므로 [Sleep Fig. 7] 느린(주기 큰) 블록일수록 weights를 건드리는 빈도(frequency)는 급감한다.

**regime 3 — sleep에서 무엇이 학습되는가.** 여기가 신세계다. consolidation은 진짜 gradient 기반 훈련이지만, 배포 **전에 한 번**이 아니라 배포된 삶의 **주기적 일부**로 돈다. 학습 대상은 새로 활성화된 low-rank expert 하나뿐이고, 나머지 전부는 동결이며, 데이터는 전부 자기 생성이다. dreaming은 per-dream LoRA SFT(안쪽)와 ReST^EM policy 최적화(바깥쪽)의 이중 구조이고, 선별 단계는 후보 dream마다 backward pass 한 번($g_{\mathrm{DR}}^{(i)}$ 계산)을 요구한다. 보고된 공통 설정은 LR $5\times10^{-6}$, effective batch 32, Sleep 100 step(GRPO baseline은 500; GRPO = group-relative policy optimization — 그룹 내 상대 보상으로 baseline을 대신하는 RLVR 계열 방법), LoRA rank 64/alpha 128이다 [Sleep Table 5].

그렇다면 "이 gate는 누가 학습하는가?"라는 이 책의 표준 질문에 대한 답은 이 논문에서 이례적이다:

표 17-3 — [Sleep]의 knob 지배 구조: 누가 정하는가.

| knob | 정체 | 누가 정하는가 |
|---|---|---|
| $W_K,W_V,W_Q$, backbone | slow weights $\Theta$ | regime 1 (또는 기성 checkpoint) |
| $C^{(\ell)}$, $f_\ell$, level 수 | frequency 사다리 | **사람이 hand-set** (예: 1k/5k/10k) |
| $\eta^{(\ell)}_t$ | level별 learning rate | Hope 계열이면 regime 1에서 조형; graft에서는 명시 없음 |
| $\lambda_{\mathrm{on}}, \lambda_{\mathrm{KD}}, \rho, z_0$ | KS/LTI의 혼합 계수·threshold | **사람이 hand-set** — 알고리즘의 hyperparameter |
| $r_{\mathrm{sem}}$의 reward model | 외부 동결 모델 | 학습되지 않음 (frozen) |
| Top-$k$, $n_{\mathrm{rand}}$, $m$, $d_{\mathrm{low}}$ | dreaming·expansion 예산 | **사람이 hand-set** |
| dream 생성 policy | 모델 자신 | regime 3에서 ReST^EM으로 학습 |
| sleep 시점 | chunk 경계 $\{C^{(\ell)}\times b\}$ | 학습되지 않음 — 스케줄에 고정 |

이 표가 드러내는 사실이 이 장의 의무 caveat다. **[Sleep]의 sleep 기제 — $\lambda_{\mathrm{on}},\lambda_{\mathrm{KD}},\rho,z_0$, reward model, ReST^EM loop — 는 어느 것도 pre-training에서 end-to-end로 meta-learn되지 않는다.** 실험의 대부분은 pre-trained Llama/Qwen checkpoint 위에 sleep 기계를 **graft**한 것이다 [Sleep §4, App. B]. 이 라인의 정체성은 지금까지 "update rule 자체를 outer loop가 미분해 학습한다"였다 — [Titans] (*Titans: Learning to Memorize at Test Time*, arXiv:2501.00663)의 $\eta_t,\beta_t,\alpha_t$는 $\Theta$가 산출하는 token의 함수였고, [Atlas]의 window gate도, [NL]의 self-modifying 목표 생성도 그랬다. [Sleep]은 라인 최초로 core mechanism이 **미분되어 통과되는 inner loop가 아니라 알고리즘적 wrapper**인 논문이다.

확인된 사실 하나를 못박아 둔다: [Sleep]의 Hope 계열 실험은 NL 레시피로 pre-train한 Hope checkpoint가 아니라 pre-trained Llama-3B/8B에 5개의 dim-64 MLP memory 블록을 얹은 graft 구성이다 [Sleep §4.1, App. B]. 따라서 regime 1 — 식 (17-1)을 관통하는 sleep-aware pre-training — 은 이 논문의 실험 어디에서도 실행되지 않았다; §17.4의 regime 1은 논문이 정의한 가능성이지 실증된 경로가 아니다.

<!-- VERIFIED(2026-07-12): Sleep §4.1·App.B — Hope 실험 = Llama-3B/8B graft(+5×dim-64 MLP, active param 불변). NL-레시피 checkpoint 아님 → regime 1 미실증으로 본문 확정. -->


> **[평가]** 이것은 결함이라기보다 방법론적 이탈이며, 정직하게 표시되어야 할 라인의 이음새(seam)다. 이 이탈에는 대칭적인 두 독법이 있다. 하나: lifecycle 수준의 결정(언제 자고, 얼마나 키우고, 무엇을 꿈꿀지)은 token 수준 gate와 달리 미분 가능한 형태로 만들기 어려우므로, wrapper는 불가피한 첫걸음이다. 둘: [NL]의 논리를 그대로 밀면 sleep의 knob들 역시 "더 느린 level의 학습 대상"이어야 하는데(frequency 스펙트럼에서 sleep 스케줄보다 느린 것은 없다), 논문은 그 일반화를 시도하지 않았다. 어느 쪽이든, "wake/sleep까지 포함해 전부를 nested optimization으로 훈련한다"는 자리는 비어 있고, 이는 §17.8의 open problem으로 넘어간다.

training 무경험 독자를 위한 마지막 정지 지점: 식 (17-5)의 기대값이 student의 sampling에 의존하는데 "sampling 분포를 통해 backpropagate하지 않는다"는 것은, $y$를 뽑는 행위 자체는 상수 취급하고 뽑힌 $y$ 위에서의 divergence·보상만 미분한다는 뜻이다. 편향이 생기지만 분산 폭발을 피한다 — GKD에서 상속한 표준 관행이다(→ 11장).

## 17.5 Concept ledger delta

표 17-4 — [Sleep]이 ledger에 가한 변경 (누적 관점).

| 구분 | 개념 | 내용 |
|---|---|---|
| 신규 | **wake/sleep lifecycle** | train/test 구분의 대체물. wake = 입력 수신·처리, sleep = 입력 차단·내부 처리 [Sleep §3.1]. 이 장이 정의 소유 |
| 신규 | **offline consolidation (기법)** | sleep phase에서 자기 생성 데이터로 수행하는 fast→slow 지식 이전. lossy 추상화 + capacity 확장이 online 판과의 차별점 [Sleep §3.1] (개념 축은 → 11장, online 판은 → 16장) |
| 신규 | **Knowledge Seeding (KS) / Self-KS** | 작은 모델→큰 모델의 upward distillation; SKS는 teacher=확장 전 자기 자신 [Sleep §3.3] |
| 신규 | **Learning to Imitate (LTI)** | prefix 완성 + $\rho\,r_{\mathrm{sem}}+(1-\rho)\,r_{\mathrm{abs}}$ 보상의 RL로 "저장한 지식을 쓰는 법"을 가르침 [Sleep Eq. 3–4] |
| 신규 | **parameter expansion (periodic (de)activation)** | sleep마다 low-rank expert 활성화; masked pre-allocation으로 정적 shape 유지 [Sleep §3.2–3.3] |
| 신규 | **Dreaming + gradient 기반 dream 선별 + random-expert novelty** | 자기 생성 데이터에 의한 self-modification; $g_{\mathrm{DR}}$ 선별; router의 random expert 주입 [Sleep §3.4] |
| 신규 | **synaptic-pruning reset** | consolidation 완료 후 빠른 블록의 과거 expert 일괄 reset [Sleep §3.3] |
| 확장 | retention 계보의 종점 | retention gate(→ 13장) → window gate(→ 14장) → periodic reset(→ 15장) → **"덮어쓰지 말고 키워라"**: CF를 capacity 문제로 재정의하고 구조적 성장으로 답함 — [Atlas]의 formal capacity(→ 14장) 논의가 lifecycle 처방으로 전화 |
| 확장 | reset의 의미 | [TNT]의 reset-to-$W_{\mathrm{init}}$(throughput 장치) → consolidation 후 reset(memory-hygiene 원리) |
| 확장 | self-modification | [NL]의 weights-by-rule(자기 생성 **target**, → 16장) → weights-by-self-generated-**data**(dream) |
| 확장 | meta-learned 초기 상태의 계보 | $W_{\mathrm{init}}$(→ 15장)의 역할 — "돌아갈 좋은 상태" — 를 teacher(확장 전 자기 자신)가 이어받음 |
| 재사용 | CMS, update frequency $f_\ell$, $C^{(\ell)}$ | [NL]에서 축자 수입 (정의 소유 → 16장) |
| 폐기 | "test time" | 라인 명칭(test-time memorization)에 남아 있던 마지막 잔재를 lifecycle 재정의로 소거 |
| 폐기(암묵) | inner/outer 이분법 | 세 regime(wake 갱신 / sleep consolidation / sleep dreaming)으로 대체 — 단 이 책은 분석 좌표로 이분법을 유지(§17.4) |

## 17.6 실험과 스케일

backbone은 전부 1B–8B의 기성 모델이다: class-incremental에 Llama-3B·Llama3-8B, ARC에 Llama-3.2-1B, 수학 추론에 Qwen3-1.7B·Qwen3-8B, 그리고 Hope 계열 변형 [Sleep §4]. 추가 파라미터는 dimension 64의 MLP block 5개이고 active 파라미터 수는 base와 같게 유지되며, 공통 훈련 설정은 §17.4에 적은 대로다(LR $5\times10^{-6}$·batch 32·Sleep 100 step·LoRA $r{=}64$/alpha 128) [Sleep App. B, Table 5].

**(1) class-incremental learning.** CLINC150(150 intent/10 domain/23.7K query), Banking77(77 intent, 13,083 example), DBpedia level-2(70 class, 10K train/1K test)에서 memory consolidation을 얹은 Hope가 ICL·EWC·InCA·순정 Hope를 모두 상회한다 [Sleep §4.1, Fig. 3; 수치는 그림으로만]. 논문의 해석: ICL 대비 이득은 prompt 적응을 지속적 parametric memory로 바꾼 데서, 순정 Hope 대비 이득은 명시적 self-distillation이 반복 in-context 갱신보다 나은 추상을 만든 데서 온다.

**(2) sleep level 수의 효과.** MK-NIAH(RULER), LongHealth(환자 기록 20건, 각 약 5.1K–6.8K 단어, 200문항), QASPER(논문 약 1.6K건 위 약 5K 문항)에서 Hope 변형이 ICL·DuoAttention·Cartridges를 상회하고, 두 경향이 일관된다 [Sleep §4.1, Fig. 4]: consolidation 단계 수가 늘수록 성능이 단조 개선되고, 가장 저주파인 memory의 frequency를 **올리면**(즉 가장 persistent한 기억을 더 adaptive하게 만들면) retention이 약해져 성능이 떨어진다.

**(3) Continual Translation of a Novel Language.** MTOB(Kalamang)+Manchu 조합의 순차 학습에서 ICL은 사전학습 수준으로 붕괴하는 반면, Hope-1/2/3(consolidation 단계 수 순)은 이득을 유지하고 단계 수에 단조 개선되며, Hope-3은 단일 언어 성능을 거의 회복한다 [Sleep §4.1, Fig. 5]. Cartridges와 SFT는 최소 한 언어에서 catastrophic forgetting으로 ICL보다도 나빠져 그림 범위 밖으로 밀려났다 [Sleep §4.1].

**(4) BABILong.** Hope(Sleep)는 10M token까지 거의 만점을 유지한다고 보고된다 [Sleep §4.1, Fig. 6]. GPT-4·GPT-4o-mini는 128K–256K 너머에서 무너지고, Llama-8B+RAG는 길이에 따라 저하되며, fine-tune된 Titans·ARMT·RMT는 약 1M까지는 Hope와 비등하다가 그 뒤 급락한다 [Sleep §B.2]. 정직성 주의 둘: 소형 모델 비교군은 **전부 BABILong 공식 protocol로 fine-tune**된 상태이고, fine-tuning 없이는 Hope를 포함한 모든 소형 모델이 큰 폭으로 하락한다고 논문 스스로 밝힌다 [Sleep §B.2]. 그리고 이 결과 역시 수치 표 없이 그림으로만 제시된다.

**(5) 수학 추론.** avg@16 기준 전 수치를 표로 옮긴다(원문 대조 검증 완료).

표 17-5 — 수학 추론, avg@16 [Sleep Table 2]. 굵은 글씨는 열 최고.

| Method | AIME-24 | AIME-25 | HMMT-25 |
|---|---|---|---|
| *Qwen3-1.7B* | | | |
| Base (Instruct) | 49.8 | 34.5 | 25.7 |
| SFT | 47.3 | 36.1 | 22.9 |
| GRPO | 51.0 | 38.6 | 26.1 |
| OPSD | 51.6 | 40.0 | 28.1 |
| Sleep | **53.2** | **40.2** | **29.3** |
| *Qwen3-8B* | | | |
| Base (Instruct) | 73.8 | 68.1 | 42.4 |
| SFT | 75.5 | 66.4 | 43.7 |
| GRPO | 76.4 | 68.1 | 44.9 |
| OPSD | 76.6 | 67.4 | 45.1 |
| Sleep | **79.2** | **69.0** | **46.1** |

Qwen3-8B에서 Sleep 79.2 vs GRPO 76.4(AIME-24)가 헤드라인이다. ablation(Qwen3-8B)은 [Sleep Table 1]: 완전체 79.2/69.0/46.1에서 Imitation Learning 제거 시 76.8/67.9/45.0, Semantic Reward 제거 시 78.9/**69.2**/44.5, Expansion 제거 시 78.2/67.9/44.9이고, OPSD 76.6/67.4/45.1에 Expansion만 더해도 77.9/68.2/45.9로 오른다. 논문은 모든 구성 요소가 기여한다고 결론짓지만 [Sleep §4.2], 표가 실제로 보여 주는 것은 더 미세하다: **semantic reward를 제거하면 AIME-25에서는 오히려 69.2로 완전체(69.0)보다 좋다.** frozen reward model이라는 외부 의존이 균일하게 이롭지는 않다는 신호이고, §17.8에서 다시 짚는다.

**(6) knowledge incorporation (SQuAD, SEAL protocol).** no-context 정답률, 전 수치 검증 완료.

표 17-6 — SQuAD knowledge incorporation [Sleep Table 3].

| Method | Single Passage (n=1) | Continued Pretraining (n=200) |
|---|---|---|
| Base model | 31.9 | 31.9 |
| Fine-tuned, no dreaming | 33.4 | 32.0 |
| SEAL | 46.7 | 43.2 |
| Sleep (Transformer, 2-level) | 48.1 | 44.3 |
| Sleep (Transformer + four-level) | **48.9** | **46.2** |
| − gradient 기반 선별 | 47.1 | 45.2 |
| − random expert | 48.0 | 44.7 |
| − Dreaming 전체 | 35.7 | 36.2 |

n=200 설정은 passage 200개를 한 번의 continued pretraining으로 흡수하고 연관 974문항 전체로 평가하며, passage당 dream 5개를 모아 합성 데이터셋을 만든다 [Sleep §4.1]. 구성 요소별로 보면 Dreaming 제거의 낙폭(48.9→35.7)이 압도적이다 — 이 task의 이득은 대부분 dreaming에서 온다.

**(7) few-shot ARC.** Llama-3.2-1B, train 11 task/held-out 8 task. sleep 훈련 중 task당 dream 60개를 뽑아 45개를 기각하고, test에서는 미지 task마다 dream 5개를 생성해 **각각 독립적으로 적용한 5개 instance**로 예측하며, 정답을 낸 dream의 비율을 보고한다 [Sleep §B.3]. 결과: ICL 0%, TTT+synthetic updates 10%, SEAL 72.5%, Sleep **80%** [Sleep Table 4].

**(8) 효율.** step당 비용은 Sleep이 SFT의 4×다. 그러나 같은 목표 성능에 도달하는 wall-clock으로 재면 SFT가 AIME-24/AIME-25/HMMT-25에서 각각 4.3×/3.6×/4.8× 더 걸린다 [Sleep App. B.5]. 즉 이 패러다임이 사는 것은 싼 step이 아니라 step·sample 효율이다.

**스케일의 정직한 결산.** [Sleep]의 8B는 이 라인의 from-scratch 사전학습 실증 상한(여전히 1.3B/100B tokens, [TNT]는 150M; → 15장)을 깨는 것이 아니다 — 기성 checkpoint에 100 step짜리 LoRA 규모 최적화를 얹은 graft이다(§17.4). 또한 여러 헤드라인 결과(Fig. 3–6)가 수치 표 없이 그림으로만 제시되고, 수백 번의 wake/sleep 사이클을 도는 장기 배포 시뮬레이션도 없다(decode wall-clock 부재는 라인 공통 — §17.7·§17.8).

## 17.7 Systems/serving 함의

이 논문의 systems 함의는 라인의 어느 논문보다 크다. 앞 논문들은 layer를 바꿨지만, 이것은 **배포 형태**를 바꾼다.

**첫째, inference는 더 이상 forward-only가 아니다.** wake phase의 CMS는 활성 블록마다 token당 optimizer error 항을 계산하고(backward류 kernel 필요), 파라미터 크기의 accumulator를 유지하며, $C^{(\ell)}$ token마다 weights에 write한다. memory 관점의 비용은 두 지점에서 정량화된다. (a) **decode state의 read-modify-write 트래픽**: 블록 $\ell$마다 token당 accumulator RMW 1회(파라미터 크기), 그리고 $C^{(\ell)}$ token에 1회로 상각되는 weight write. KV cache의 append-only 트래픽과 달리 이것은 read-modify-write이고, 그 대역폭 계산이 10장의 cost model에 새 항으로 들어간다. (b) **update frequency와 memory 계층 배치의 대응**: $C=$1k→5k→10k token 사다리 [Sleep Fig. 7]에서 고주파(주기 작은) 블록의 accumulator는 연산 가까이 상주해야 하지만, 저주파 블록은 weights를 수천 token에 한 번 건드리므로 더 느린 계층에 두고도 write 비용을 상각할 수 있다 — frequency 사다리가 곧 storage-tier 사다리의 설계 힌트다. chunk 안에서 파라미터가 상수라는 성질 덕에 chunk 단위 batching·병렬화는 보존된다(→ 9장, 15장).

**둘째, session state의 범주가 바뀔 수 있다.** 순수 Transformer의 세션 상태는 KV cache, $O(L)$이다 — BABILong의 10M token에서 이는 매우 큰 크기이고, GPT-4급 모델의 정확도가 128K–256K 너머에서 하락한다 [Sleep §B.2](단 논문은 이 하락을 KV-cache 용량 고갈로 귀인하지는 않는다 — 인과 주장은 유보한다). [Sleep]은 장기 지식을 parametric memory(CMS)로 옮기지만 sequence-layer state를 **일반적으로 제거하지는 않는다** — CMS의 sequence model이 attention이면 KV cache의 $O(L)$ 상태가 그대로 남는다 [Sleep §2.2]. fixed-state sequence module(Titans류 $W_t$)을 택한 변형에서만 지속 세션 상태가 **파라미터**로 대체된다: consolidation 1회당 low-rank expert 하나, $2\,d\,d_{\mathrm{low}}$개 값($d_{\mathrm{low}}\ll d$; 실험 전체가 dim-64 블록 5개 추가로 수행됨 [Sleep App. B]). 이것은 KV cache 문제가 아니라 **per-user/per-agent weight-delta 서빙 문제**다 — multi-tenant LoRA adapter 서빙과 같은 부류로, adapter의 버전 관리·routing·eviction이 세션 관리의 어휘가 된다. 1장 Rosetta의 "paged KV cache ↔ per-session weight state" 대응이 여기서 문자 그대로 실현된다.

**셋째, 정적 shape는 설계로 보장된다.** masked pre-allocation(§17.3.3) 덕에 tensor 크기 변경·재컴파일이 없다. 대가는 비활성 expert의 죽은 자리인데, sparse MoE dispatch가 masked expert를 건너뛰면 FLOP 낭비는 자연히 사라진다 — router 수준에서 처리 가능한 비용이다.

**넷째, sleep은 서빙 배포에 붙는 스케줄된 훈련 job이다.** 한 번의 sleep이 요구하는 것을 나열하면 그 자체로 인프라 명세서다: teacher corpus 생성 + student on-policy rollout + LTI 완성 생성(전부 generation-heavy, 즉 inference 모양의 부하), reward-model 서비스 호출 + 값싼 Levenshtein 채점, 후보 dream당 backward pass 1회(ARC protocol이면 task당 60회), dream별로 완전히 독립인 LoRA SFT job들(embarrassingly parallel), 그리고 ReST^EM 반복. 서빙 fleet이 fine-tuning capacity와 주기적으로 결합되어야 하고, **sleep 1회마다 새 모델 버전이 태어난다** — checkpoint 관리, cache 무효화, regression test, rollback이 릴리스 시점의 일이 아니라 상시 운영의 일이 된다. ARC의 test-time protocol은 여기에 한 겹을 더한다: 미지 task 하나의 "추론"이 dream 5개로 각각 adapt된 5개 instance로의 fan-out이다 [Sleep §B.3] — gradient update가 serving loop 안에 들어와 있다.

**다섯째, 비용 프로파일.** step당 4×라는 가격과 iso-accuracy 3.6–4.8× 이득 [Sleep App. B.5]은 "고정 capability 목표에는 compute 우위, 대신 복잡도를 선불"로 요약된다 — RL, reward model, MoE 성장 부기, 버전 관리가 그 선불이다.

> **[평가]** 이 절의 그림은 논문이 그린 것이 아니라 논문이 **비운** 자리다. [Sleep]에는 sleep 한 사이클의 wall-clock·에너지 분해가 없고, dream 선별의 60-backward-pass 비용 분석이 없으며, 동일 compute의 replay-buffer baseline 비교도 없다. decode wall-clock은 라인 6편 공통으로 부재하다. "sleep은 서빙에 붙는 훈련 job"이라는 명제의 경제성은 이 책의 Part III가 검증해야 할 가설이지, 논문이 입증한 사실이 아니다.

## 17.8 한계와 bridge-out: 라인의 완성형과 남은 문제

마지막 장이므로 bridge-out은 다음 논문이 아니라 라인 전체의 결산으로 향한다. 먼저 [Sleep] 자신의 한계를 정리한다.

**[Sleep]의 한계.** (1) **graft이지 co-training이 아니다** — sleep 기계는 기성 backbone에 부착되었고, wake/sleep을 처음부터 함께 훈련한 실증이 없다(§17.4의 의무 caveat). (2) **sleep 시점이 학습되지 않는다** — chunk 경계에 hard-wire되어 있고, 불규칙한 실전 트래픽 아래의 거동은 미지수다. (3) **dreaming이 task-aware다** — $(x_{\mathrm{ctx}},\tau)$, 즉 context와 평가 가능한 성능 측정이 sleep 중에 주어져야 한다. 완전 비지도 배포에는 $\tau$가 없으므로 [Sleep Eq. 5]의 이진 개선 보상을 무엇으로 대체할지가 공백이다. (4) **frozen reward model 의존** — 계속 변하는 student에 대한 고정 심판의 편향·drift가 검토되지 않았고, §17.6에서 본 대로 semantic reward 제거가 AIME-25에서는 오히려 이롭다(69.2 vs 69.0 [Sleep Table 1]). (5) **router 동역학 미명세** — 새로 활성화된 expert로 dispatch를 배우고 reset된 expert로의 dispatch를 멈추는 과정이 서술되지 않았다. 실구현의 실질적 공백이다. (6) **장기 안정성** — 수백 번의 sleep에 걸친 expansion+reset의 안정성, pre-allocate된 expert pool의 고갈 시점 모두 미검증. (7) **이론 부재** — CF를 capacity 문제로 재정의했지만, consolidation당 얼마의 capacity가 얼마의 forgetting을 막는지, upward distillation의 lossy "추상화"가 무엇을 보존하는지에 대한 형식적 진술이 없다. (8) **재귀적 self-modification의 안전성** — 매일 밤 자기 weights를 고쳐 쓰고 random expert로 novelty를 주입하는 모델의 안전·거버넌스 논의가 전혀 없다.

동시대 지형에서의 위치도 기록해 둔다. [Sleep App. A.4]는 2026년의 on-policy self-distillation(OPSD) 물결 전체를 지도화하고 네 축으로 차별화한다: (i) 고정 teacher의 re-conditioning이 아니라 새로 키운 capacity로의 upward distillation, (ii) 평평한 teacher/student 쌍이 아니라 frequency로 정렬된 memory 사슬, (iii) consolidation만이 아니라 Dreaming을 포함한 2단계 sleep, (iv) 순수 per-token reverse-KL이 아니라 GKD+imitation learning. 그리고 OPSD의 보고된 실패 모드 — epistemic verbalisation 억제로 인한 최대 40%의 OOD 하락(Kim et al. 2026), 반복 적용 시의 leakage·collapse — 를 2단계 설계의 동기로 인용한다. OpenReview 2025년 9월 공개를 근거로 한 우선권 주장도 이 부록의 일부다.

**라인의 완성형.** 이제 여섯 논문을 겹쳐 놓고 이 책의 결산을 적는다.

> **[평가]** 여섯 논문이 스스로 세운 것만 합성하면 하나의 완성형이 실제로 그려진다. (1) **state는 모든 시간 규모에서 weights다** — per-token fast memory([Titans]·[Atlas]), chunk 주기의 CMS level([NL]), sleep 주기의 grown expert([Sleep]), frequency 0의 persistent weights까지; Titans→Sleep은 $\{\infty,0\}$ 두 점뿐이던 frequency 스펙트럼을 연속체로 넓혀 온 하나의 긴 운동이다. (2) **모든 블록은 같은 객체다** — (architecture, attentional bias, retention, inner optimizer, frequency)로 정의되는 associative memory이고, attention 자체가 그 안의 non-parametric·무한 capacity·frequency-$\infty$ 꼭짓점이다(→ 13장, 14장, 16장). (3) **inner optimizer는 architecture와 대등한 설계 표면이다** — GD(TTT, → 8장)→momentum([Titans])→objective 동물원([Miras])→window+Muon([Atlas])→자기 생성 target([NL]). (4) **훈련은 어디서나 chunk-anchored이고, 필요한 곳에서 계층적이다** — stale-snapshot 근사가 여섯 편 전부의 병렬화 기반이고(→ 9장), 그 경제학과 reset은 [TNT]가 정리했다. (5) **lifecycle은 wake/sleep이다** — 모델은 결코 "훈련이 끝나지" 않으며, 서빙 fleet은 설계상 주기적 fine-tuning job을 돌린다. 제품으로 읽으면: **세션 상태가 mutable weights인 continually-learning LLM에, sleep 서비스가 배포에 붙어 있는 시스템**이다. 이 완성형이 "보이는" 이유는 여섯 논문의 open-questions 절들이 전부 같은 다섯 공백을 가리키기 때문이다 — 남은 일이 텍스트에 의해 과잉 결정되어 있다.
>
> 단, 두 가지 유보 없이 이 결산은 정직하지 않다. 첫째, 완성은 **개념적** 완성이다. from-scratch 실증은 1.3B/100B tokens에서 멈춰 있고, in-context retrieval의 격차는 측정된 채 닫히지 않았다 — attention 53.55 vs 43.70 [Atlas], FDA 67.3 vs 41.9 [NL] (→ 14장, 16장). capacity 이론이 그 이유까지 말해 주므로($\phi^*$의 무한 capacity), 완성형은 attention과의 hybrid일 가능성이 열려 있고, 라인 자신의 MAC/MAG 결과들이 이를 조용히 인정한다. 둘째, "여섯 편이 한 이야기"라는 독법은 부분적으로 소급적이다. 이음새가 실재한다: [TNT]는 이야기가 본질이라 말하는 momentum·gating을 "명료성을 위해" 제거한 훈련 논문이고(→ 15장), [Sleep]은 end-to-end meta-learning을 떠났다(§17.4). 이 책은 그 이음새를 지우지 않고 보여 주는 쪽을 택한다.

**남은 open problems.** 여섯 편의 자체 목록을 중복 제거하면 다섯으로 수렴한다. (1) **스케일**: 모든 품질 주장이 1.3B/100B에서 멈춘다 — momentum·deep memory·self-modification의 우위가 7B+·SFT/RLHF·production 데이터에서 살아남는지가 최대 미지수다. (2) **retrieval 격차**: parametric하게 닫을 수 있는가, 아니면 완성형은 필연적으로 hybrid인가. (3) **serving 경제학과 kernel**: decode throughput/latency 수치 전무, fused deep-memory kernel 부재, per-request mutable weights가 깨뜨리는 shared-weight batching(→ 10장) — grouped-GEMM decode, state snapshot/rollback, optimizer-trajectory state의 수치 drift, multi-tenant 격리, weight로 흡수된 context의 privacy까지. (4) **학습되는 스케줄**: chunk 크기, window $c$, CMS frequency, level 수, sleep 시점 전부가 hand-set이다 — "무엇을 언제 갱신할지"를 배우는 기제가 없다. (5) **task-free이며 안전한 self-modification**: $\tau$ 없는 dreaming, reward model 의존의 해소, 재귀적 weight 자기 편집의 안전성 분석, 그리고 미뤄진 이론 전부 — linear 특수 사례 밖의 regret·capacity·expressivity, chunkwise staleness의 오차 한계는 여섯 편 어디에도 없다(→ 9장).

[Titans]가 "test time에 외우는 법을 배우자"로 열었던 질문은, 여섯 편째에 이르러 "모델은 언제 깨어 있고 언제 자야 하는가"로 바뀌었다. 그 질문의 답이 논문 한 편이 아니라 serving 시스템의 설계 문서처럼 생겼다는 것 — 그것이 이 라인이 inference 엔지니어에게 남긴 초대장이고, 이 책의 Part III가 그 초대에 응한다.
