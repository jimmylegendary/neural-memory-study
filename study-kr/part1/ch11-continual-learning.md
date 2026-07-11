# ch11. Continual learning, complementary learning systems, distillation, RL-lite

> **이 장의 목표** — 이 장을 마친 독자는 다음을 할 수 있다.
> 1. catastrophic forgetting을 공유 weight 위의 gradient 간섭으로 정의하고, 2-parameter 예제에서 forgetting이 일어나는 조건(입력 겹침)과 일어나지 않는 조건(orthogonality)을 손으로 계산할 수 있다.
> 2. replay·regularization(EWC)·parameter isolation 세 완화책 가족을 (state, update, cost) 객체로 비교하고, 새 기법을 만나면 어느 가족의 조합인지 분류할 수 있다.
> 3. complementary learning systems의 fast/slow 분업 구도를 update-frequency spectrum으로 확장해 읽고, online consolidation과 offline consolidation을 구분해 [NL]과 [Sleep]이 각각 어느 반쪽을 지었는지 말할 수 있다.
> 4. Hinton KD, on-policy GKD, REINFORCE, ReST^EM을 구분하고, REINFORCE 한 스텝을 손으로 계산하며, [Sleep]의 sleep phase가 이 부품들의 어떤 조립인지 지도 위에 놓을 수 있다.
>
> **왜 필요한가** — 이 장은 6편 중 오직 [NL] (*Nested Learning*, arXiv:2512.24695)과 [Sleep] (*Language Models Need Sleep*, arXiv:2606.03979) 두 편을 위해 존재한다. [NL §1]의 anterograde-amnesia framing과 continual-learning 실험(CLINC·CTNL, 16장에서 상술), 그리고 [NL §4.3]의 "momentum도 잊는다" 논증은 catastrophic forgetting과 complementary learning systems의 어휘 없이는 수사로만 읽힌다. [Sleep]은 기제 전체가 이 장의 재료로 조립된다: wake/sleep 구분과 online/offline consolidation의 이분법 [Sleep §3.1], catastrophic forgetting을 capacity 문제로 재정의하고 parameter expansion으로 답하는 [Sleep §3.2], Knowledge Seeding = GKD 기반 on-policy distillation + RL 보상 [Sleep §3.3], Dreaming = ReST^EM 기반 self-training [Sleep §3.4]. 반대로 ch12–ch15의 네 논문에는 이 장이 필요 없다 — 그래서 이 장은 Part I의 맨 끝, Part II 직전에 있다.

## 11.1 Catastrophic forgetting: 쓰기가 곧 파괴인 세계

독자의 세계에는 forgetting이 없다. serving 중인 모델의 weight는 움직이지 않고("weight update 없음"이라는 추론 불변식, → 1장), KV cache는 append-only여서 어제 넣은 entry가 오늘의 write 때문에 변질되는 일이 없다. 지식이 파괴되려면 누군가 명시적으로 지워야 한다. 이 라인이 그 불변식을 폐기한 순간(→ 8장), 훈련 쪽 세계의 기본 병리 하나가 함께 들어온다. 이 절은 그 병리를 정의한다.

**catastrophic forgetting**(CF)은 task A를 이미 학습한 network를 새 task B의 데이터만으로 계속 훈련하면, task A의 성능이 점진적이 아니라 급격하게 — 종종 거의 전부 — 무너지는 현상이다(McCloskey & Cohen 1989, *Psychology of Learning and Motivation*; French 1999, *Trends in Cognitive Sciences*). 메커니즘은 두 사실의 합이다. 첫째, neural network의 지식은 국소화되어 있지 않다. 하나의 mapping이 수많은 weight에 분산 저장되고, 하나의 weight가 수많은 mapping에 참여한다. 둘째, gradient descent는 현재 objective만 본다. $\Theta \leftarrow \Theta - \eta\,\nabla_\Theta \mathcal{L}_{\mathrm{B}}(\Theta)$의 어디에도 "task A의 loss를 지켜라"는 항이 없다. 그래서 B를 위한 update가 A가 쓰던 좌표를 지나가면, A의 함수값은 A의 데이터를 한 번도 다시 보지 않은 채로 바뀐다. 파괴는 부작용이 아니라 update rule의 정의 그 자체다.

이 구조는 이미 본 것이다. 5장의 crosstalk은 하나의 matrix memory에 여러 $(k,v)$ 쌍을 겹쳐 쓸 때 모든 read가 점진적으로 오염되는 현상이었다(→ 5장). catastrophic forgetting은 같은 병리의 시간축 확대판이다: crosstalk이 한 시퀀스 안에서 write들끼리 간섭하는 것이라면, CF는 모델의 배포 수명 동안 task들끼리 간섭하는 것이다. 뿌리가 같으므로("겹치는 표현 + 파괴적 write") 해법의 문법도 같다 — 간섭을 피하도록 좌표를 분리하거나(isolation), 지킬 것을 명시하거나(regularization·retention), 옛것을 다시 써 넣거나(replay). 이 세 가족이 §11.2의 전부다.

간섭의 양은 기하가 정한다. linear 모델 $y=\Theta^\top x$에서 task B의 gradient는 B의 입력 방향으로만 뻗으므로, A와 B의 입력이 orthogonal하면 B의 학습은 A가 쓰는 좌표를 건드리지 않고, 겹치면 겹친 만큼 A를 파괴한다 — §11.7에서 숫자로 확인한다. deep network에서는 표현이 학습되면서 겹침 자체가 움직이므로 이렇게 깔끔하게 쪼개지지 않지만, "forgetting = gradient 부분공간의 겹침"이라는 1차 근사는 이 라인의 논증을 읽는 데 충분하다.

[NL]은 여기에 한 겹을 더 얹는다: 잊는 것은 weight만이 아니다. 2장에서 momentum을 "gradient들의 EMA를 유지하는 상태 객체"로 배웠다. [NL §4.3]은 decay $\beta=0.9$일 때 마지막 gradient 6개가 momentum buffer 누적 기여의 50% 이상, 마지막 43개가 99% 이상을 차지한다고 계산하고, 이를 근거로 momentum이라는 "gradient의 memory"가 최근만 기억하는 low-pass filter임을 지적한다. orthogonal한 task가 이어지는 continual learning에서는 optimizer가 옛 task의 gradient 부분공간을 통째로 잊고, [NL §4.3]은 이것을 모델 capacity의 실패가 아니라 optimizer의 memory 관리 실패로 규정한다. 즉 forgetting은 특정 모듈의 결함이 아니라 **모든 EMA류 상태의 공통 성질**이다 — 2장에서 optimizer를 상태 객체로 배운 것이 여기서 처음 배당을 지급한다. [Sleep §1]은 같은 문제를 배포 관점의 딜레마로 요약한다: 갱신하지 않으면 지식이 낡고, 갱신하면 CF가 온다.

systems 어휘로 옮기면 CF는 **policy 없는 eviction**이다. cache의 eviction은 무엇을 버릴지 명시적 정책(LRU, LFU)이 고르지만, 공유 weight 위의 gradient update는 "새 데이터의 loss를 줄이는 방향"이라는 단일 기준으로 낡은 entry를 암묵적으로 덮어쓴다 — 무엇이 지워졌는지 로그도 남지 않는다. retention gate(→ 13장)는 이 eviction을 학습된 정책으로 바꾸지만, 여전히 무엇을 파괴할지 고르는 정책이다. 파괴 없이 새 지식을 넣으려면 저장소가 자라야 한다. 이 직관이 [Sleep §3.2]의 재정의 — CF는 근본적으로 capacity 문제이며, parameter가 덮어써지는 이유는 새 지식을 넣을 자리가 없기 때문이라는 — 로 곧장 이어진다.

## 11.2 고전 완화책 3종: replay, EWC, parameter isolation

30년치 continual-learning 문헌은 세 가족으로 압축된다. 이 책의 관행대로 각 가족을 (state, update, cost)를 갖는 객체로 제시한다(→ 2장). 이 분류의 실용적 가치는 [NL]·[Sleep]의 기제를 만났을 때 "새 발명"과 "고전의 재조합"을 구분하게 해 준다는 데 있다.

**replay**는 옛 데이터의 일부를 buffer에 보관했다가 새 task의 배치에 섞어 넣는 것이다. update rule은 바뀌지 않고 데이터 분포만 교정된다: 매 스텝의 gradient가 옛 task의 loss 항을 계속 포함하므로, §11.1의 "A를 지켜라는 항이 없다"는 문제를 데이터 쪽에서 해결한다. state는 buffer 자체(모델 밖의 byte), update는 배치 구성 규칙, cost는 저장 용량 + 옛 샘플만큼의 추가 FLOP + 원본 데이터를 계속 보관하는 데 따르는 거버넌스 부담이다. 변형으로, buffer 대신 **생성된 데이터로 하는 replay**(pseudo-rehearsal; French 1999가 정리한 계보)가 있다: 모델 자신이나 생성기가 옛 지식을 표본화해 그것으로 되새김한다. [Sleep]의 sleep phase가 teacher(자기 자신의 이전 버전)에게서 corpus를 표본화해 consolidation에 쓰는 것은 정확히 이 계보다 — 원본 데이터가 없어도 되는 replay(→ 17장).

**EWC**(elastic weight consolidation; Kirkpatrick et al. 2017, PNAS, arXiv:1612.00796)는 regularization 가족의 대표다. task A를 끝낸 시점의 parameter $\Theta^{\star\mathrm{A}}$를 닻으로 삼고, task B의 loss에 좌표별 가중 이차 penalty를 더한다:

$$
\mathcal{L}_{\mathrm{EWC}}(\Theta)
\;=\;
\mathcal{L}_{\mathrm{B}}(\Theta)
\;+\;
\sum_i \frac{\lambda_{\mathrm{EWC}}}{2}\, F_i \,\big(\Theta_i - \Theta^{\star\mathrm{A}}_i\big)^2 .
\tag{11-1}
$$

여기서 $F_i$는 task A에서의 diagonal **Fisher information** — "좌표 $i$가 task A의 예측에 얼마나 하중을 받는가"의 국소 추정 — 이고, $\lambda_{\mathrm{EWC}}$는 옛 task 보존과 새 task 학습의 트레이드오프 계수다(둘 다 장-국소 기호). 직관은 단순하다: A에 하중이 큰 좌표는 못 움직이게 잡아 두고, update를 A가 쓰지 않는 좌표로 흘려보낸다. state는 parameter 크기의 buffer 두 개($\Theta^{\star\mathrm{A}}$와 $F$) — 2장에서 Adam이 이미 $m_t, h_t$ 두 buffer를 갖고 다니는 것을 본 독자에게 익숙한 자원 등급이다. cost는 FLOP으로는 미미하고 memory로는 task마다 buffer가 쌓인다는 것, 그리고 더 근본적으로는 $F$가 국소 이차 근사일 뿐이라 task가 여럿 쌓이면 닻들이 서로 충돌한다는 것이다. 식 (11-1)을 이 책의 표기로 다시 읽으면 정체가 드러난다: weight decay가 원점으로 당기는 무차별 retention이라면(→ 2장), EWC는 옛 최적점으로 당기는 **선택적 retention**이다. Miras가 retention gate를 "무엇을 남길지의 regularizer"로 재이론화하는 것(→ 13장)과 같은 문법 위에 있다.

**parameter isolation**은 간섭을 원천 차단한다: 옛 task의 parameter를 동결하고, 새 task에는 새 capacity(새 모듈, mask, adapter)를 배정한다. state는 task별 parameter 구획과 routing 정보, update는 "새 구획만 훈련", cost는 task 수에 비례해 자라는 parameter와 어느 구획을 쓸지 고르는 routing이다. 동결된 구획은 구조적으로 파괴 불가능하므로 CF가 정의상 없다 — 대신 문제가 "무엇을 지킬까"에서 "성장을 어떻게 관리할까"로 이동한다. [Sleep §3.2]의 parameter expansion은 정확히 이 가족이다: consolidation 때마다 저주파 블록에 low-rank expert 하나를 새로 활성화하고, 전이되는 지식은 그 새 expert에만 쓰며(나머지 전부 동결 — 구조적 CF 방지), 모든 expert를 초기화 시점에 미리 할당해 두고 mask만 벗기는 구현으로 tensor shape을 고정한다. 성장 관리 문제는 빠른 블록의 옛 expert를 주기적으로 reset하는 것으로 답한다(synaptic-pruning reset, → 17장).

표 11-1 — 세 완화책 가족의 (state, update, cost)와 이 라인에서의 화신

| 가족 | state | update | cost | 대표 실패 모드 | 이 라인에서의 화신 |
|---|---|---|---|---|---|
| replay | 옛 데이터 buffer (모델 밖 byte) | 새 배치에 옛 샘플 혼합 | 저장 + 추가 FLOP + 데이터 거버넌스 | buffer가 옛 분포를 대표하지 못함 | [Sleep]의 teacher-표본 corpus (생성형 replay, → 17장) |
| regularization (EWC) | 닻 $\Theta^{\star}$ + Fisher $F$ (parameter 크기 buffer 2개/task) | loss에 이차 penalty 추가, 식 (11-1) | memory (task당 buffer), 근사 오차 | 국소 근사 붕괴, 닻 간 충돌, capacity 경합 | retention gate의 사촌 (선택적 retention, → 13장) |
| parameter isolation | task별 구획·mask·adapter | 옛 구획 동결, 새 구획만 훈련 | parameter 성장 + routing | 무한 성장 (pruning 없이는) | [Sleep §3.2] low-rank expert expansion + reset (→ 17장) |

systems 접점을 세 줄로 요약한다. replay buffer는 **data-plane artifact**다 — 모델 파일이 아니라 저장·표본화 대역폭·보존 정책을 갖는 데이터 자산이며, fleet 설계에서 별도 수명주기를 가진다. EWC의 buffer는 optimizer state와 같은 자원 등급이므로 "훈련 상태 = 추가 parameter 몇 벌"이라는 2장의 회계에 그대로 편입된다. parameter isolation의 서빙 형태를 독자는 이미 운영하고 있다: per-tenant LoRA adapter를 base weight 위에 얹어 서빙하는 multi-LoRA 스택이 바로 그것이다 — continual learning의 격리 전략과 serving의 격리 전략이 같은 물건임을 [Sleep]이 이용한다.

## 11.3 Complementary learning systems: 두 학습자의 분업

**complementary learning systems**(CLS)는 하나의 학습 시스템이 "빠른 획득"과 "안정된 보존"을 동시에 잘할 수 없으므로, 뇌가 학습을 두 시스템으로 분업시켰다는 이론이다(McClelland, McNaughton & O'Reilly 1995, *Psychological Review*). hippocampus는 빠른 학습자다: 새 경험을 한 번에, 서로 겹치지 않는 sparse한 표현으로 즉시 기록한다. neocortex는 느린 학습자다: 겹치는 분산 표현 위에서 구조화된 일반 지식을 축적하며, 그래서 갑작스러운 write에 취약하다 — §11.1의 CF가 바로 겹치는 표현 + 빠른 write의 조합에서 왔음을 상기하라. 분업의 핵심 고리는 전이다: hippocampus에 임시 저장된 새 기억이 수면·휴식 중 **replay**되어 옛 지식과 섞여(interleave) neocortex에 천천히 통합된다. 즉 CLS의 처방은 §11.2의 세 가족 중 두 개의 합성이다 — 빠른 쪽과 느린 쪽을 분리(isolation)하고, 둘 사이를 replay로 연결한다.

이 이론이 없으면 [NL]과 [Sleep]의 신경과학 수사는 장식으로 읽힌다. [NL §1]은 배포된 LLM을 anterograde amnesia — 새 장기 기억을 만들지 못해 영원한 현재를 사는 환자 — 에 비유하는데, CLS 어휘로 옮기면 정확한 진단이 된다: context window(즉시, 세션이 끝나면 소멸)와 동결된 weight(pre-training 시점에 고정) 사이에 hippocampus→neocortex 전이에 해당하는 경로가 없다. [NL §1]은 또한 뇌가 brain-wave 주파수별로 여러 시간 척도에서 기억을 굳히는 것을 들어 "deep network의 update 주파수가 $\infty$(attention state)와 0(동결 MLP) 둘뿐"임을 문제 삼는다 — 이 관찰이 §11.4의 spectrum으로 이어진다. [Sleep §1]은 한 단계 더 내려가 수면의 두 단계를 각각 계산 절차로 번역한다: NREM(느린 파형) 수면의 hippocampus→neocortex consolidation과 synaptic homeostasis(전역 downscaling)는 Memory Consolidation 단계로, REM 수면의 통합·시뮬레이션(꿈)은 Dreaming 단계로. 두 논문에서 신경과학은 동기이지 메커니즘이 아니다 — 검증되는 것은 계산 절차 쪽이다.

표 11-2 — CLS ↔ 이 라인의 대응

| CLS (생물학) | 이 라인의 대응물 | 정의 장 |
|---|---|---|
| hippocampus (빠른 학습자, sparse, 즉시 기록) | fast weights $W_t$, 고주파 블록 | → 6·8장, 16장 |
| neocortex (느린 학습자, 분산 표현, 구조화) | slow weights $\Theta$, 저주파 블록 | → 2장, 16장 |
| 수면 중 interleaved replay | self-generated data로 하는 offline consolidation | §11.4, → 17장 |
| synaptic homeostasis / pruning | 빠른 블록의 low-rank expert reset | → 17장 |
| brain-wave 주파수 사다리 | CMS의 update-frequency 사다리 | → 16장 |

systems 관점에서 CLS는 신경과학이기 이전에 낯익은 아키텍처 패턴이다: write에 최적화된 작은 front store와 read에 최적화된 큰 back store를 두고, 주기적 background 작업이 앞에서 뒤로 데이터를 재조직하며 옮긴다. 독자가 storage 스택에서 매일 보는 구도이며, §11.8에서 이 유비를 정색하고 편다.

## 11.4 Online vs offline consolidation, 그리고 update-frequency spectrum

지금까지 이 책은 두 개의 시간 척도만 썼다: token마다 움직이는 fast weights $W_t$(→ 6·8장)와, pre-training이 끝나면 동결되는 slow weights $\Theta$(→ 2장). CLS의 두 학습자와 정확히 포개지는 이분법이다. [NL]의 첫 수는 이 이분법을 **spectrum**으로 바꾸는 것이다: 배포된 Transformer에서 attention의 state는 매 token 다시 만들어지므로 update frequency가 사실상 $\infty$이고, MLP는 0이다 — 그 사이가 통째로 비어 있다 [NL §1]. 그 사이를 채우는 것, 즉 "1K token마다", "10K token마다" 갱신되는 중간 주파수의 memory 블록 사다리를 두는 것이 CMS다(정식 정의와 update rule 식 (M5)는 → 16장; [Sleep]의 실험은 주기 1K→5K→10K 같은 사다리를 쓴다 [Sleep Fig. 7]). update 주기가 곧 기억의 수명 등급이 된다: 고주파 블록은 최근 문맥의 단기 기억, 저주파 블록은 장기 기억.

이 spectrum 위에서 이 장이 소유하는 구분 하나를 정의한다. **online consolidation**은 모델이 입력을 받는 동안(wake), 상시 학습 과정 자체를 통해 고주파 블록의 지식이 저주파 블록으로 전이되는 것이다 — [NL]의 CMS와 Hope가 하는 일이며, backprop이 층위를 관통해 흐르는 end-to-end 갱신의 부산물로 전이가 일어난다. **offline consolidation**은 입력이 멈춘 기간에, self-generated data를 매개로 지식을 명시적으로 전이·추상화하고 필요하면 capacity를 늘리는 것이다 — [Sleep]의 sleep phase가 하는 일이다. 이 이분법 자체는 [Sleep §3.1]이 wake/sleep lifecycle을 정의하면서 정식화했다(그 lifecycle의 상세는 → 17장). [Sleep §1]은 online만으로 부족한 이유를 세 가지로 든다: 전이가 같은 추상화 수준에서 일어나 lossy 압축이 없으므로 capacity를 그대로 소모하고, 능동적으로 retrieval되는 기억만 강화되는 선택적 과정이며, 현재 context에 갇혀 새 지식과 기존 지식의 상위 수준 통합을 놓친다.

spectrum은 CF를 없애지 않고 유예한다. 저주파 블록도 언젠가는 갱신되고, 그 갱신은 여전히 파괴적 write다. [Sleep §3.2]의 관찰이 이 장 전체의 매듭이다: 주기 1K인 블록 뒤에 주기 10K인 블록이 있으면, 느린 블록이 한 번 갱신되기 전에 빠른 블록의 지식이 10번 그 안으로 consolidate되어야 하고, 고정 크기 저장소에 대한 이 반복 write가 정확히 CF가 터지는 지점이다. 그래서 consolidation은 **덮어쓰기 직전에** fire해야 하고([Sleep §3.2]의 스케줄: 각 블록의 update 경계마다), 반복 write를 받는 쪽은 capacity가 자라야 한다 — §11.2의 isolation 가족이 다시 등장하는 이유다.

systems 접점: update frequency는 memory 계층 배치의 언어로 즉시 번역된다. 매 token 갱신되는 상태는 연산 유닛 가까이(register/SRAM 감각의 자리)에 있어야 하고, 1K token마다 갱신되는 상태는 write 트래픽이 3자릿수 낮으므로 더 멀고 싼 계층에 두어도 된다. 주기 $C^{(\ell)}$이 곧 그 블록의 **write-traffic 예산**이다: parameter 크기 $P$ byte인 블록의 상각 write 대역폭은 $P/C^{(\ell)}$ byte/token으로, 사다리의 각 단이 자기 자리를 스스로 말해 준다. 이 계산은 Part II에서 [NL]·[Sleep]의 serving 함의를 논할 때의 표준 도구가 된다(→ 16·17장).

## 11.5 Knowledge distillation: soft target에서 on-policy GKD까지

offline consolidation의 실행 수단이 필요하다. 지식을 한 parameter 덩어리에서 다른 덩어리로 "옮긴다"는 것은 물리적 복사가 아니다 — 두 덩어리는 shape부터 다르다. 옮길 수 있는 것은 행동뿐이다. **knowledge distillation**(KD)은 teacher 모델의 출력 분포를 target으로 삼아 student 모델을 훈련하는 기법이다(Hinton, Vinyals & Dean 2015, arXiv:1503.02531). LM이라면 위치 $t$마다 teacher의 다음-token 분포 $p^{\mathrm{T}}(\cdot\,|\,y_{<t},x)$를 통째로 맞추게 한다:

$$
\mathcal{L}_{\mathrm{KD}}
\;=\;
\mathbb{E}_{(x,y)\sim\mathcal{D}}
\sum_{t}
\mathrm{KL}\!\Big(p^{\mathrm{T}}(\cdot\,|\,y_{<t},x)\;\Big\|\;p^{\mathrm{S}}(\cdot\,|\,y_{<t},x)\Big).
\tag{11-2}
$$

hard label 대비 이득은 분포의 꼬리에 있다: teacher가 세 후보에 $(0.7, 0.2, 0.1)$을 주면, 정답 하나만 남기는 hard label이 버리는 "오답들 사이의 2:1 비율"까지 student가 물려받는다 — Hinton이 dark knowledge라고 부른 정보다. temperature로 분포를 눅여 꼬리를 키우는 변형이 표준이지만, 이 라인을 읽는 데는 "teacher 분포 전체가 target"이라는 뼈대면 충분하다.

식 (11-2)의 급소는 기대값이 걸리는 분포 $\mathcal{D}$다. teacher가 만든(또는 원본 corpus의) 시퀀스 위에서만 KL을 재면, student는 **자기가 생성할 때 실제로 방문하는 상태**에서 한 번도 교정받지 못한다. autoregressive 생성은 오차가 누적되는 루프라서, 훈련 분포(teacher의 궤적)와 서빙 분포(student 자신의 궤적)의 괴리가 성능을 갉아먹는다. 독자의 어휘로 정확히 옮겨진다: 합성 벤치 트래픽으로만 검증한 시스템이 live 트래픽에서 처음 보는 상태를 만나는 문제다. off-policy 훈련/on-policy 서빙의 불일치라는 이 진단에서 **on-policy distillation**이 나온다: student가 직접 생성한 시퀀스 위에서 teacher가 token-level로 채점한다.

**GKD**(generalized knowledge distillation; Agarwal et al., *On-Policy Distillation of Language Models*, ICLR 2024, arXiv:2306.13649)는 두 극단을 하나의 목적함수로 섞는다:

$$
\mathcal{L}_{\mathrm{GKD}}
\;=\;
(1-\lambda_{\mathrm{on}})\;
\mathbb{E}_{(x,y)\sim\mathcal{D}}
\Big[\mathcal{F}\big(p^{\mathrm{T}}\,\|\,p^{\mathrm{S}}\big)(y\,|\,x)\Big]
\;+\;
\lambda_{\mathrm{on}}\;
\mathbb{E}_{x\sim\mathcal{D}}\,
\mathbb{E}_{y\sim p^{\mathrm{S}}(\cdot|x)}
\Big[\mathcal{F}\big(p^{\mathrm{T}}\,\|\,p^{\mathrm{S}}\big)(y\,|\,x)\Big].
\tag{11-3}
$$

$\lambda_{\mathrm{on}}\in[0,1]$은 student 자신의 rollout이 차지하는 비율이고(이 책의 통일 기호; 원문들의 $\lambda$), $\mathcal{F}$는 teacher/student token 분포 사이의 divergence다. $\mathcal{F}$의 선택은 자유 축이다: forward KL $\mathrm{KL}(p^{\mathrm{T}}\|p^{\mathrm{S}})$은 teacher가 질량을 둔 곳을 student가 모두 덮게 만들고(mode-covering), reverse KL $\mathrm{KL}(p^{\mathrm{S}}\|p^{\mathrm{T}})$은 teacher 질량이 희박한 곳에 student가 질량을 두는 것을 강하게 벌해 소수 mode에 집중하게 만든다(mode-seeking). 실무 관행 하나가 중요하다: 둘째 항에서 student의 표본화 분포 자체로는 backprop하지 않는다 — 표본은 상수 취급하고 그 위의 divergence만 미분한다. 분산과 불안정성을 피하는 표준 선택이며, [Sleep §3.3]도 이 관행을 그대로 계승한다.

[Sleep]이 이 위에 얹는 뒤틀림 하나만 예고한다. KD의 통상 방향은 큰 teacher → 작은 student의 압축이다. [Sleep §3.3]의 Knowledge Seeding은 방향을 뒤집는다: 작은 teacher(갱신 직전의 고주파 블록을 품은, 확장 전의 자기 자신)가 더 큰 student(low-rank expert가 새로 열린 확장 후의 자기 자신)에게 가르친다 — 그래서 별칭이 upward distillation이다. 외부 corpus 없이 teacher에게서 표본화한 $\mathcal{D}$로 식 (11-3)형 objective를 돌리고, student 쪽은 새 expert만 훈련 가능하다. 상세와 정식 정의는 → 17장.

systems 접점: distillation job의 자원 프로파일은 훈련보다 추론에 가깝다. 지배 비용은 corpus 생성(teacher 표본화)과 on-policy rollout(student 생성) — 즉 decode-heavy, 독자의 fleet가 가장 잘하는 일 — 이고, backward는 student(Sleep에서는 low-rank expert 하나)에만 걸린다. "데이터셋을 모델이 만든다"는 이 구도가 §11.8의 lifecycle 그림에서 핵심 부품이 된다.

## 11.6 RL-lite: REINFORCE, imitation, 그리고 post-training의 모양

distillation은 target 분포가 있을 때의 이야기다. [Sleep]의 Dreaming은 target 분포가 없는 신호 — "이 합성 데이터로 fine-tune했더니 벤치마크가 올랐는가" 같은, 시퀀스가 끝난 뒤 도착하는 스칼라 — 로 생성 정책을 개선해야 한다. 이것이 RL이 필요한 전부이므로, 이 절은 RL을 엔지니어 컷으로만 자른다: 추정기 하나, 구분 하나, 조립 패턴 하나.

어휘부터. **policy**는 상태를 받아 행동의 분포를 내는 함수다 — LM이 곧 policy다: $\pi_\Theta(y\,|\,x)$는 prompt $x$에서 시퀀스 $y$를 생성할 확률이고, 독자가 매일 돌리는 sampler가 policy의 실행이다. **reward** $r(y)$는 완성된 시퀀스에 대해 환경이 주는 스칼라다. 요점은 $r$이 미분 가능할 필요도, 모델일 필요도 없다는 것: 채점기, Levenshtein distance, "정답 여부" 모두 된다. 목표는 기대 reward $J(\Theta)=\mathbb{E}_{y\sim\pi_\Theta}[r(y)]$의 최대화인데, $r$을 미분할 수 없으니 gradient를 어떻게 얻는가가 유일한 기술적 문제다.

**REINFORCE**(Williams 1992, *Machine Learning*)가 답이다. 항등식 $\nabla_\Theta \pi_\Theta = \pi_\Theta \nabla_\Theta \log \pi_\Theta$를 기대값 안에 넣으면

$$
\nabla_\Theta J(\Theta)
\;=\;
\mathbb{E}_{y\sim\pi_\Theta(\cdot|x)}
\Big[\big(r(y)-\bar r\big)\,\nabla_\Theta \log \pi_\Theta(y\,|\,x)\Big],
\qquad
\log \pi_\Theta(y\,|\,x)=\sum_t \log \pi_\Theta(y_t\,|\,y_{<t},x).
\tag{11-4}
$$

reward의 gradient는 등장하지 않는다 — 표본화한 시퀀스의 log-likelihood gradient에 reward를 곱해 평균할 뿐이다. baseline $\bar r$(보통 표본 평균 reward; 장-국소 기호)은 기대값을 바꾸지 않으면서 분산만 줄이는 상수다. 구현 관점의 번역이 이 절의 요지다: 식 (11-4)는 **reward로 가중된 SFT gradient**다. kernel 수준에서 하는 일은 cross-entropy 훈련과 동일하고, 다른 것은 데이터가 자기 생성물이라는 것과 시퀀스별 가중치가 $(r-\bar r)$이라는 것뿐이다. §11.7에서 두 행동짜리 예제로 한 스텝을 손으로 돌린다.

구분 하나: **imitation learning**은 시연자의 행동 분포를 맞추는 것(target 분포가 있음 — SFT와 KD가 이 가족), RL은 스칼라 reward를 최대화하는 것(target 분포가 없음)이다. 피드백의 밀도로 다시 읽으면 스펙트럼이 된다: KD는 위치마다 분포 전체(가장 진한 피드백), on-policy GKD는 자기 궤적 위에서 위치마다 분포, REINFORCE는 시퀀스 전체에 스칼라 하나(가장 희박한 피드백)다. [Sleep §3.3]의 Learning to Imitate는 이름 그대로 이 사이에 있다: distillation으로 지식을 넣은 student에게 teacher 시퀀스의 prefix를 주고 나머지를 완성하게 한 뒤, 의미 동등성 reward와 Levenshtein 기반 reward의 혼합(이 책의 통일 기호로 $\rho$와 $\lambda_{\mathrm{KD}}$가 관장하는 항들, → 17장)으로 강화한다 — "아는 것"과 "그렇게 행동하는 것"의 간극을 RL로 메우는 부품이다.

조립 패턴: 현대 post-training의 두 축은 reward의 출처다. RLHF는 사람 선호로 학습한 reward model이 채점하고(Ouyang et al. 2022, arXiv:2203.02155; 최적화기로는 PPO 계열), RLVR은 검증기(테스트 통과, 정답 일치)가 채점한다. 이 장에서 이름만 알면 되는 이유는, [Sleep]이 실제로 쓰는 루프가 그보다 훨씬 단순한 **ReST^EM**(Singh et al. 2024, arXiv:2312.06585)이기 때문이다: 생성한다 → binary reward로 거른다 → 생존한 표본에 SFT한다 → 반복한다. value network도, step별 보정도 없다 — "필터링된 자기 생성물에 대한 반복 SFT"라는 EM 풍의 루프이며, gradient 추정기로서는 식 (11-4)의 reward가 0/1이라 생존자만 남는 특수형으로 읽어도 무방하다. [Sleep §3.4]의 Dreaming은 SEAL(Zweiger et al. 2025, arXiv:2506.10943)의 self-edit 루프를 이 스케줄에 얹고, 후보 선별과 novelty 주입을 추가한 것이다(→ 17장).

마지막 부품은 **LoRA**(Hu et al. 2022, arXiv:2106.09685)다. weight 행렬에 low-rank 보정 $\Delta W = AB$ ($A\in\mathbb{R}^{d\times d_{\mathrm{low}}}$, $B\in\mathbb{R}^{d_{\mathrm{low}}\times d}$, $d_{\mathrm{low}}\ll d$)만 학습하는 fine-tuning이다. 독자는 이것을 serving 쪽에서 이미 안다 — multi-LoRA 스택의 그 adapter다. 훈련 관점에서 한 가지만 바로잡자: LoRA가 줄이는 것은 훈련 가능한 parameter 수(따라서 optimizer state와 전송·저장량)이지, backward pass의 연산이 아니다 — gradient는 여전히 network 전체를 관통해 흘러야 $A,B$에 도달한다. [Sleep §3.4]가 dream마다 격리된 LoRA SFT를 쓰는 이유는 훈련이 싸서라기보다 **병렬·격리·폐기가 쉬워서**다: adapter 하나 = 실험 하나, 실패하면 버린다.

systems 접점: RL 루프의 병목은 학습 스텝이 아니라 rollout 생성이다. 식 (11-4)의 기대값을 채우는 표본이 전부 autoregressive 생성이므로, 생성 처리량이 곧 학습 처리량이고, RL 인프라의 대부분은 사실상 대규모 decode 서비스다. distillation(§11.5)과 같은 결론의 다른 얼굴: post-training 시대의 워크로드는 독자의 전문 분야 쪽으로 이미 넘어와 있다.

## 11.7 Worked micro-example: forgetting, EWC, REINFORCE를 손으로

세 부품을 각각 최소 크기로 돌려 보자. 코드 없이 표와 수식만으로 따라갈 수 있다.

**(a) forgetting이 일어나는 계산.** 모델은 $y=\Theta^\top x$, $\Theta=(\theta_1,\theta_2)^\top\in\mathbb{R}^2$ (성분 $\theta_1,\theta_2$는 장-국소 기호), loss는 제곱 오차다. task A는 예제 하나: 입력 $x^{\mathrm{A}}=(1,0)^\top$, 목표 $1$, 즉 $\mathcal{L}_{\mathrm{A}}(\Theta)=(\theta_1-1)^2$. task B도 하나: $x^{\mathrm{B}}=(1,1)^\top$, 목표 $0$, 즉 $\mathcal{L}_{\mathrm{B}}(\Theta)=(\theta_1+\theta_2)^2$. A를 끝내면 $\Theta=(1,0)$이고 $\mathcal{L}_{\mathrm{A}}=0$이다. 이제 B만으로 gradient step 하나를 밟는다. $\nabla_\Theta\mathcal{L}_{\mathrm{B}}=2(\theta_1+\theta_2)\,(1,1)^\top$이므로 $(1,0)$에서 gradient는 $(2,2)^\top$, $\eta=0.25$로

$$
\Theta \;\leftarrow\; (1,0) - 0.25\,(2,2) \;=\; (0.5,\,-0.5).
$$

결과: $\mathcal{L}_{\mathrm{B}}=(0.5-0.5)^2=0$ — B는 한 스텝에 완벽히 풀렸다. 그리고 $\mathcal{L}_{\mathrm{A}}=(0.5-1)^2=0.25$ — A의 오차가 $0$에서 $0.25$로 뛰었다. A의 데이터는 등장한 적도 없다. 원인은 기하다: $x^{\mathrm{A}\top}x^{\mathrm{B}}=1\neq0$이라 B의 gradient가 A의 하중 좌표 $\theta_1$을 관통했다. 대조 실험으로 $x^{\mathrm{B}}=(0,1)^\top$이었다면 B의 gradient는 $\theta_2$축에만 실리고 $\theta_1$은 손끝 하나 안 다친다 — 간섭은 겹침에서만 온다(5장의 crosstalk 조건과 동일). [NL §4.3]이 orthogonal task 시나리오를 표준 예로 쓰는 이유가 이 대비에 있다.

**(b) EWC가 구조하는 계산.** task A의 Fisher를 추정한다. $\nabla_\Theta\, \hat y = x$이므로 empirical Fisher는 $F \propto x^{\mathrm{A}} x^{\mathrm{A}\top}$의 대각, 즉 $F=\mathrm{diag}(1,0)$: "A는 $\theta_1$에만 하중을 싣는다." $\lambda_{\mathrm{EWC}}=2$로 식 (11-1)을 쓰면 목적함수는

$$
(\theta_1+\theta_2)^2 + (\theta_1-1)^2 .
$$

두 편미분을 0으로 놓으면 $\theta_1+\theta_2=0$과 $\theta_1=1$, 즉 $\Theta^\star=(1,-1)$. 검산: $\mathcal{L}_{\mathrm{A}}=(1-1)^2=0$, $\mathcal{L}_{\mathrm{B}}=(1-1)^2=0$ — 두 task 모두 정확히 풀렸다. EWC가 한 일은 update를 A의 null 방향($\theta_2$)으로 흘려보낸 것이다: Fisher가 $\theta_2$를 "공짜 좌표"로 판정했고, penalty가 $\theta_1$을 닻에 묶었다. 단서 하나가 이 장난감에서도 보인다: 구조가 성립한 것은 **남는 capacity($\theta_2$)가 있었기 때문**이다. 두 task가 같은 좌표를 서로 다른 값으로 요구하면 어떤 penalty 계수도 둘 다를 살릴 수 없다 — CF를 capacity 문제로 재정의하고 확장으로 답하는 [Sleep §3.2]의 논리가 2차원에서 이미 작동한다.

**(c) REINFORCE 한 스텝.** 행동이 두 개뿐인 한 상태 문제(bandit)를 보자. policy는 $\pi_\Theta=\mathrm{softmax}(\Theta)$, $\Theta=(0,0)$에서 시작하므로 $p=(0.5,\,0.5)$. reward는 $r(a^{(1)})=1$, $r(a^{(2)})=0$이고 policy는 이를 모른다. 표본 하나를 뽑아 $a^{(1)}$이 나왔다고 하자. softmax의 성질에서 $\nabla_\Theta\log\pi_\Theta(a^{(1)})=(1-p_1,\,-p_2)=(0.5,\,-0.5)$이고, baseline 없이 $\eta=1$로 식 (11-4)를 적용하면

$$
\Theta \;\leftarrow\; (0,0) + 1\cdot 1\cdot(0.5,\,-0.5) \;=\; (0.5,\,-0.5),
\qquad
p_1 \;=\; \frac{1}{1+e^{-1}} \;\approx\; 0.731 .
$$

좋은 행동의 확률이 $0.5\to0.731$로 올랐다 — reward를 미분한 적은 없다. 이번엔 $a^{(2)}$가 뽑혔다면? $r=0$이라 baseline 없는 update는 정확히 0이다: 실패 표본은 아무것도 가르치지 못하고 버려진다. baseline $\bar r=0.5$를 넣으면 달라진다: 가중치가 $r-\bar r=-0.5$, $\nabla_\Theta\log\pi_\Theta(a^{(2)})=(-0.5,\,0.5)$이므로 update는 $(-0.5)\cdot(-0.5,\,0.5)=(0.25,\,-0.25)$, 즉 $p_1=\frac{1}{1+e^{-0.5}}\approx0.622$ — **실패 표본에서도** 좋은 행동 쪽으로 확률이 이동한다. baseline이 분산을 줄인다는 말의 손에 잡히는 형태다. 이제 [Sleep §3.4]의 Dreaming reward — "이 dream으로 LoRA-SFT한 모델이 벤치마크에서 나아졌는가"의 0/1 — 를 보라. 정확히 이 예제의 reward 자리에 들어가는, 미분 불가능한 black-box 스칼라다.

## 11.8 Systems bridge: wake/sleep은 serving fleet의 lifecycle이다

1장의 Rosetta 사전에는 이 장을 위해 예약된 행이 하나 있었다: "서빙 fleet의 백그라운드 job ↔ sleep phase = 주기적 offline consolidation/dreaming job." 이제 그 행을 정색하고 채운다.

> **[해설]** 이 책이 제안하는 가장 압축적인 유비는 LSM-tree다. write에 최적화된 작은 memtable이 실시간 트래픽을 받고, 배경의 compaction job이 주기적으로 그것을 read에 최적화된 큰 하위 계층으로 재조직해 내려보낸다. 대응은 항목별로 성립한다: memtable ↔ fast weights·고주파 블록(빠른 write, 작음, 휘발성), compaction ↔ consolidation(배경에서 재조직·병합, 전면 트래픽과 격리), 계층별 크기·갱신 빈도의 사다리 ↔ update-frequency spectrum(§11.4), compaction의 write amplification ↔ 빠른 블록이 느린 블록에 반복 consolidate되는 비용([Sleep §3.2]의 주기 1K→10K에서 10회). 유비의 한계도 명시한다: LSM compaction은 무손실 재배치지만 consolidation은 distillation을 거치는 **lossy 추상화**이고, 성공 기준이 byte 보존이 아니라 행동 보존이다.

이 그림에서 sleep phase의 실체는 신비로운 것이 아니라 **serving 배포에 부착된, 스케줄된 오프라인 job의 묶음**이다. 프로파일별로 뜯으면 독자에게 전부 낯익다. (1) corpus 생성과 on-policy rollout은 decode-heavy 추론 워크로드다 — 훈련 클러스터가 아니라 fleet의 비수기 용량이 자연 서식지다. (2) backward가 필요한 부분은 low-rank expert 또는 LoRA 크기로 국한된다(§11.5–11.6). (3) dream별 LoRA SFT는 서로 격리된 작은 job이라 embarrassingly parallel하다. (4) reward 채점은 별도 서비스(reward model) 호출 또는 문자열 연산(Levenshtein)이다. 비용의 1차 회계도 논문에 있다: [Sleep App. B.5]는 스텝당으로는 Sleep이 SFT의 4배 비싸지만, 같은 정확도에 도달하는 wall-clock으로는 SFT 쪽이 AIME-24/AIME-25/HMMT-25에서 각각 4.3×/3.6×/4.8× 더 걸린다고 보고한다 — 스텝은 비싸고 스텝 수가 적은 profile이다.

운영 관점의 파생 문제들이 이 lifecycle의 진짜 청구서다. sleep 한 번이 끝날 때마다 fleet에는 **새 모델 버전**이 생긴다 — checkpoint 관리, cache 무효화, 회귀 테스트, rollback이 릴리스 이벤트가 아니라 상시 운영 문제가 된다. replay/teacher corpus는 data-plane artifact로서 저장·보존·감사의 대상이 되고(§11.2), per-session fast-weight state와 per-tenant adapter는 이미 논한 새 cache class의 문제다(→ 10장). 그리고 정직하게: 이 절의 lifecycle 그림은 1B–8B 규모의 실험에서 외삽한 것이다. 6편 전체에 serving 규모의 wall-clock·에너지 회계는 없으며, sleep 스케줄을 실제 트래픽 아래에서 언제 어떻게 돌릴지는 열린 문제로 남아 있다(→ 17장).

## 요약

- catastrophic forgetting은 공유 weight 위에서 gradient update가 현재 objective만 보기 때문에 생기는 파괴적 write이며, 5장 crosstalk의 시간축 확대판이다. 간섭량은 task 간 입력/gradient 겹침의 기하가 정한다.
- 잊는 것은 weight만이 아니다: $\beta=0.9$인 momentum은 마지막 gradient 6개가 기여의 50%, 43개가 99%를 차지하는 low-pass filter라서 optimizer의 memory도 잊는다 [NL §4.3].
- 완화책은 세 가족으로 압축된다 — replay(데이터 재주입), regularization/EWC(선택적 retention, 식 (11-1)), parameter isolation(동결 + 새 capacity). [Sleep]의 parameter expansion은 isolation 가족이고, teacher-표본 corpus는 생성형 replay 계보다.
- CLS는 빠른 학습자(hippocampus)와 느린 학습자(neocortex)의 분업 + replay 전이로 CF를 피하는 구도이며, [NL]·[Sleep]의 신경과학 수사를 해독하는 열쇠다.
- fast/slow 이분법은 update-frequency spectrum으로 일반화된다. online consolidation은 wake 중 end-to-end 학습에 의한 전이([NL]의 CMS), offline consolidation은 입력 없는 기간에 self-generated data로 하는 전이·추상화·확장([Sleep]의 sleep)이다 [Sleep §3.1]. multi-frequency는 CF를 유예할 뿐이므로 consolidation은 덮어쓰기 직전에 fire해야 한다 [Sleep §3.2].
- KD는 teacher 분포를 target으로 하는 훈련(식 (11-2)), GKD는 student 자신의 rollout 위에서 teacher가 token-level로 채점하는 on-policy 혼합(식 (11-3))이다. [Sleep]의 Knowledge Seeding은 GKD의 방향을 뒤집은 upward distillation이다.
- REINFORCE는 reward-가중 SFT gradient다(식 (11-4)): reward는 미분 불가능한 black-box여도 되고, baseline은 분산만 줄인다. ReST^EM은 "생성→binary 필터→SFT→반복"의 단순 루프이며 [Sleep]의 Dreaming이 그 위에 선다.
- systems 렌즈: sleep phase = serving fleet에 부착된 주기적 배경 job(LSM compaction 유비), distillation·RL 워크로드의 지배 비용은 생성(decode-heavy), update frequency는 write-traffic 예산으로서 memory 계층 배치를 스스로 지정한다.

## 자가 점검 체크리스트

- [ ] catastrophic forgetting이 왜 gradient descent의 정의에서 곧바로 따라 나오는지, 그리고 crosstalk(→ 5장)과 어떤 관계인지 설명할 수 있다.
- [ ] §11.7(a)–(b)를 다른 숫자(예: $x^{\mathrm{B}}=(1,2)^\top$)로 다시 계산해 forgetting 양과 EWC 해를 구할 수 있다.
- [ ] replay·EWC·parameter isolation의 state/update/cost를 각각 말하고, [Sleep]의 parameter expansion + expert reset이 어느 가족의 어떤 변형인지 판별할 수 있다.
- [ ] online consolidation과 offline consolidation을 정의하고, [NL]과 [Sleep]이 각각 어느 쪽을 다루며 offline이 왜 별도로 필요한지([Sleep §1]의 세 가지 한계) 설명할 수 있다.
- [ ] 식 (11-2)와 (11-3)의 차이를 "훈련 분포 vs 서빙 분포"의 언어로 설명하고, forward/reverse KL의 행동 차이를 말할 수 있다.
- [ ] REINFORCE 한 스텝(§11.7(c))을 손으로 재현하고, baseline이 무엇을 바꾸고 무엇을 바꾸지 않는지 말할 수 있다.
- [ ] sleep phase를 inference 어휘로 옮길 수 있다: 어떤 sub-job이 decode-heavy이고 어떤 것이 backward를 요구하며, 왜 전체가 "weights의 background compaction"으로 읽히는지.

## 다음 장으로

Part I이 여기서 끝난다. 이 장의 재료 — forgetting, CLS, consolidation, distillation, RL — 는 잠시 서랍에 들어가고, 16장(NL)과 17장(Sleep)에서 회수된다. 당장 다음 장은 라인의 본편이 시작되는 12장이다: 2장의 optimizer 객체(momentum, weight decay), 5장의 associative loss, 8장의 TTT, 9장의 chunkwise 병렬화가 하나의 아키텍처로 조립된 [Titans] (*Titans: Learning to Memorize at Test Time*, arXiv:2501.00663)를, 이 책의 master update (M2)를 기준 삼아 읽는다. 이 장이 심어 둔 질문 하나를 들고 가라: Titans의 retention gate가 하는 "학습된 eviction"은 결국 파괴적 write다 — 파괴 없이 축적하는 시스템은 어떻게 생겼는가. 그 답이 이 라인의 종착지(16·17장)다.
