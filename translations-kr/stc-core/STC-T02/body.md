# Sleep (offline consolidation for LLMs) — 한국어 전문 번역본 (무축약)

> arXiv:2606.03979 「Sleep (offline consolidation for LLMs)」((Google Research))의 **전문 무축약 한국어 번역**입니다. 스터디용이며 공식 번역이 아닙니다. 원문의 모든 문장·수식·표·각주를 빠짐없이 옮기는 것을 목표로 했고, 수식은 PDF 추출 텍스트 기반이라 일부 기호가 손상될 수 있습니다. 그림은 원저자의 것입니다.

---

# 언어 모델에는 잠이 필요하다: 자기 수정과 기억 통합의 학습

Ali Behrouz †,‡, Farnoosh Hashemi ‡, Vahab Mirrokni †

†

‡

arXiv:2606.03979v1 [cs.LG] 2026년 6월 2일

## 초록 (Abstract)

지난 수십 년 동안 기계 학습 알고리즘 설계에서 상당한 진보가 있었다—과제 특화형(task-specific) 얕은 모델에 관한 초기 연구부터 보다 일반적인 심층 대규모 언어 모델(Large Language Models, LLMs)에 이르기까지. 즉각적인 예측이나 문맥 내 학습(in-context learning)을 요구하는 과제에서 유망한 결과를 보였음에도, 기존 모델들은 지속적으로 학습하고 시간적(temporal) 문맥 내 지식을 자신의 장기 파라미터로 효과적으로 전이(transfer)하는 능력이 결여되어 있다. 인간의 학습 과정에서 영감을 받아, 우리는 모델이 지속적으로 학습하고, 취약한 단기 기억을 리플레이(replay)를 통해 안정적인 장기 지식으로 증류(distill)하며, "꿈꾸기(Dreaming)" 과정을 통해 재귀적으로 스스로를 개선하도록 하는 "잠(Sleep)" 패러다임을 소개한다. 더 자세히 말하면, 잠은 두 단계로 구성된다: (1) 기억 통합(Memory Consolidation): 지식 씨앗화(Knowledge Seeding)라 부르는 상향식 증류(upward distillation) 과정으로, 여기서 더 작은 자기 자신(smaller-self)의 기억이 더 큰 네트워크로 증류되어, 지식을 보존하면서도 더 많은 용량(capacity)을 제공한다. 개념 증명(proof of concept)으로, 우리는 지식 씨앗화를 위한 새로운 일반화된 증류(Generalized Distillation) 과정(즉, 온-폴리시 증류(on-policy distillation)와 강화 학습(Reinforcement Learning, RL) 기반 모방 학습(imitation learning)의 결합)을 제시한다; (2) 꿈꾸기(Dreaming): 자기 개선(self-improvement) 단계로, 모델이 RL을 사용해 새로운 지식을 예행연습(rehearse)하고 기존 능력을 인간의 감독 없이 정제하기 위한 합성 데이터의 커리큘럼(curriculum)을 생성한다. 장기 지평(long-horizon), 지속 학습(continual learning), 지식 통합(knowledge incorporation), 소수샷 일반화(few-shot generalization) 과제에 대한 우리의 실험은 잠 단계의 중요성을 뒷받침한다.

## 1. 서론 (Introduction)

대규모 언어 모델(LLMs)의 발전은 기계 학습 연구에서 중요한 이정표를 이룬다: 즉, 과제 특화형 모델에서 다양한 창발적(emergent) 능력을 갖춘 보다 범용적인 시스템으로의 패러다임 전환이다(Brown et al. 2020; Schaeffer et al. 2023). 다양한 과제 집합에서 LLM들이 보여준 놀라운 능력에도 불구하고(Nijkamp et al. 2023; Wang et al. 2023; Comanici et al. 2025), 그것들은 초기 배치(deployment) 이후 대체로 정적(static)이다. 즉, 사전 학습(pre-training) 또는 사후 학습(post-training) 동안 학습된 과제는 성공적으로 수행하지만, 자신의 즉각적인 문맥(immediate context)을 넘어 새로운 능력을 지속적으로 획득할 수는 없다. 이러한 본질적으로 정적인 성질은 결정적인 취약점을 만들어낸다: 모델의 지식과 기술은 점차 낡아지며(stale), 고정된 "지식 컷오프(knowledge cutoff)" 날짜를 기준으로 작동하여 그 이후의 새로운 사실, 사건, 진화하는 정보를 알지 못한다(Cheng et al. 2024).

이러한 한계를 극복하려는 노력은 주로 다음에 집중되어 왔다: (1) 확장된 데이터셋에 대한 재사전학습(re-pretraining), 이는 효과적이긴 하지만 계산 비용이 크고 빈번한 업데이트에는 비실용적이다(Ibrahim et al. 2024); (2) 값비싼 지속적 파라미터 업데이트나 미세조정(fine-tuning) 또는 저랭크 적응(low-rank adaption)과 같은 다른 경량 대안들을 사용하는 것(Hu et al. 2022; Akyurek et al. 2024a), 이는 반복적인 업데이트를 거치면서 흔히 파국적 망각(Catastrophic Forgetting, CF)을 초래한다(Kemker et al. 2018; Shi et al. 2024)—파국적 망각은 모델이 새로운 과제를 학습함에 따라 원래 과제에 대한 숙련도가 파국적으로 저하되는 잘 알려진 현상이다. 한편으로는 지식 노후화(knowledge obsolescence), 다른 한편으로는 파국적 망각과 더불어 업데이트의 막대한 비용 혹은 파괴적 성질 사이의 이 딜레마는, 결정적이면서도 미해결된 과제를 부각시킨다: 바로 LLM들이 그 생애주기(lifecycle) 전반에 걸쳐 점진적이고 효율적으로 학습할 수 있게 하는 것이다.

최근 몇 년간, 문맥 내 학습(In-Context Learning, ICL)(Brown et al. 2020)은 매우 효율적이고 성공적인 형태의 지속 학습으로 주목받아 왔다(Akyurek et al. 2022, 2024b; Dong et al. 2024; Li et al. 2025). 처음에 ICL은 대규모 데이터로 학습된 LLM들의 창발적 능력으로 알려졌으며, 이는 그것들이 문맥에 빠르게 적응하여 제로샷(zero-shot) 또는 소수샷(few-shot) 과제를 수행할 수 있게 한다(Brown et al. 2020). 이후, 더 많은 연구들이 ICL의 역할을 메타 학습(meta-learning) 과정으로서 밝히고 형식화했다. 이 과정에서 모델은 시퀀스를 따라 내부 계산을 수행하여 문맥 지식을 자신의 출력에 통합하되, 그것을 단기 기억(short-term memory)에 유지하거나 압축한다(Behrouz et al. 2025; Dherin et al. 2025). 지속 학습의 한 형태로서 ICL의 효과성/효율성에도 불구하고,

§ 교신 저자(Correspondence to): {alibehrouz, mirrokni}@google.com

및 sh2574@cornell.edu.

∗ 본 연구의 한 버전은 2025년 9월부터 OpenReview에서 공개적으로 이용 가능하였다.

---

**그림 1: (전통적 기계 학습 vs. 지속 학습)** 전통적 기계 학습에서는 흔히 모델의 수명(lifespan)이 테스트 시간(test time)과 학습 시간(training time)으로 나뉘는 반면, 지속 학습 설정에는 이러한 단계들이 없다. 우리는 지속 학습자(continual learner)가 학습에 있어 서로 다른 활성 상태의 단계들을 가질 필요가 있다고 제안하며, 이를 다음과 같이 부른다: (i) 활성(Active) 또는 깨어 있는 시간(Wake Time), 그리고 (ii) 잠 시간(Sleep Time). 잠 시간은 수동적 상태가 아니라, 오히려 내부적으로 데이터를 처리하여 빠르고 불안정한 모듈에서 보다 안정적인 저주파(느린) 구성요소로 기억을 통합한다.

---

지속 학습의 한 형태로서 ICL은 시퀀스 모델의 문맥 창(context-window)에 제한된다. 즉, 새로 획득된 어떤 지식이든 세션/문맥의 끝에서 모델로부터 제거된다는 것을 의미한다. 이 관점은 결정적인 질문을 제기한다: 모델은 어떻게 취약한 단기 기억을 보다 안정적인 장기 지식으로 효과적으로 전이할 수 있는가?

하나의 비유로서, 우리는 Behrouz et al. (2025)에서 제시된 LLM에 대한 순행성 기억상실증(anterograde amnesia)의 예를 사용한다: 인간에게서 정보를 단기 기억으로부터 보다 장기적인 기억으로 전이하는 과정에 손상이 있는 경우를 생각해 보라. 그 예가 순행성 기억상실증—장애 발병 이후에는 새로운 기억을 형성할 수 없지만 기존 기억은 온전히 남아 있는 신경학적 상태이다(Scoville et al. 1957). 이러한 상태는 사람의 지식을 단기 기억에 담기는 즉각적인 현재와, 장애 발병 이전의 먼 과거로 제한할 수 있으며, 그 결과 즉각적인 현재를 마치 언제나 새로운 것처럼 지속적으로 경험하게 된다. 현재의 트랜스포머(Transformer) 기반 LLM들의 기억 처리에서도 유사한 패턴을 발견할 수 있다. LLM들의 지식은 다음 중 하나로 제한된다: (1) 그것들의 문맥 창에 담기는 즉각적인 문맥(다시 말해 문맥 내 학습), 또는 (2) 먼 과거, 즉 "사전 학습의 종료" 발병 이전의 것을 저장하는 MLP 및 투영(projection) 레이어들. 이러한 패턴상의 유사성은 우리에게 다음을 묻도록 동기를 부여한다. 인간 학습 과정에서 기억을 통합하는 결정적 구성요소는 무엇인가?

인간의 뇌는 기억을 통합하는 데 있어 매우 효율적이고, 손실적(lossy)이지만 효과적이며, 이는 흔히 신경가소성(neuroplasticity)—새로운 경험, 기억, 학습에 반응하여 스스로를 변화시키는 뇌의 능력—에 기인한다(Pascual-Leone et al. 2005; Johnston 2009). 최근 연구들은 장기 기억 형성이 적어도 두 개의 구별되면서도 상호 보완적인 통합 과정을 포함한다는 것을 뒷받침한다(Frey et al. 1997; Goto et al. 2021; Yang et al. 2024): (1) 빠른 "온라인(online)" 통합 단계는 학습 직후 또는 곧, 심지어 깨어 있는 동안에도 일어난다. 이때 새로운 취약한 기억 흔적(memory traces)이 안정화되고 단기 기억으로부터 보다 장기적인 저장소로 전이되기 시작한다; (2) "오프라인(offline)" 통합(시스템 통합(systems consolidation)이라고도 알려짐) 과정은 잠자는 동안 최근 부호화된(encoded) 패턴들의 리플레이를 반복하고, 기억을 재조직하며, 피질(cortical) 부위로의 전이를 지원한다(Foster et al. 2006; Ji et al. 2007; Peyrache et al. 2009).

**인간의 온라인 통합 (활성 단계에서)**
인간의 온라인 통합은 새로 학습된 기억을 회상(recall) 동안 재활성화(reactivating)함으로써 그것을 능동적으로, 깨어 있는 상태에서 강화하고 변형하는 것으로, 시간이 지남에 따라 그 기억을 보다 안정적이고 보다 의미론적(semantic-like)으로 만든다.

**인간의 오프라인 통합 (잠 및 조용한 휴식 단계에서)**
인간의 오프라인 통합은 학습 이후 휴식이나 잠자는 동안 일어나며, 이때 새로 부호화된 기억이 안정화되고 복잡한 손실적 증류(lossy distillation) 과정을 통해 점진적으로 보다 분산된 신피질(neocortical) 표상으로 변형된다.

순행성 기억상실증의 비유로 돌아가면, 그 증거는 이 상태가 두 단계 모두에 영향을 미칠 수 있음을 시사한다. Behrouz et al. (2025)은 최근 중첩 학습(Nested Learning, NL) 패러다임을 제시하고 첫 번째 형태의 기억 통합(즉, 온라인 통합)을 겨냥하였다. 특히, NL 기반 Hope 아키텍처(그림 1 참고)는 각 구성요소가 서로 다른 주파수(frequency)로 업데이트되되 종단간(end-to-end) 방식으로 이루어지는 연속체 기억 시스템(continuum memory system) 내에서 기억을 재활성화함으로써, 지식을 빠르고 불안정한 구성요소로부터 보다 안정적인 저주파 모듈로 온라인 방식으로 전이한다. 이 종단간 과정은 지식을 보다 안정적인 구성요소로 직접 전이하여 망각을 지연시킬 수 있지만, 여전히 같은 양의 모델 용량을 사용한다. 이는 주로 모델이 어떠한 추가적인 손실적 압축 없이 같은 추상화 수준(level of abstraction)에서 지식을 유지한다는 사실 때문에 발생한다. 더 나아가, 이 형태의 통합은 견고한 지속 학습에는 충분하지 않다: (i) 선택적이고 인출 의존적(retrieval-dependent)이다: 그것은 능동적 인출에 의존하므로 반복적으로 회상되는 기억의 부분들만 강화한다; (ii) 문맥에 의존한다: 온라인 통합에 의해 야기되는 업데이트는 오직 모델의 문맥에만 기반하므로, 새로운 지식과 기존 지식에 대한 더 높은 수준의 이해를 놓친다.

본 논문에서 우리는 두 번째 유형의 통합, 즉 잠을 통한 오프라인 통합에 초점을 맞추며, 이는 인간의 잠 단계에서 영감을 받았다:

**인간 학습 과정에서 잠의 역할**
잠은 수동적 상태가 아니라 인지 기능에 필수적인 역동적이고 고도로 구조화된 뇌 활동 기간이다(Rasch et al. 2013; Goldstein et al. 2014). 잠자는 동안, 뇌는 학습, 신경 가소성, 자기 개선, 그리고 기억 통합에 근본적인 복잡한 과정들을 조율한다(Wamsley et al. 2011; Rasch et al. 2013; Goldstein et al. 2014). 인간에게서 이러한 과정들은 주로 두 개의 결정적이고 교대하는 잠 단계에 의해 지배된다: 급속 안구 운동(Rapid Eye Movement, REM) 수면과 비-REM(Non-REM, NREM) 수면.

**비급속 안구 운동 수면 (서파 수면, Slow-Wave Sleep):** 이 단계, 특히 서파 수면(slow-wave sleep)으로 알려진 그 가장 깊은 국면은 동기화되고 고진폭이며 저주파인 신경 활동을 특징으로 한다. 서파 수면은 학습에 결정적인 두 가지 주요 기능과 관련된다: 첫째는 시냅스 항상성(synaptic homeostasis)으로, 이는 깨어 있는 경험으로부터 비롯된 연결성의 순 증가를 상쇄하기 위해 시냅스 강도를 전역적으로 하향 조정하는 과정이며, 그럼으로써 대사적 균형을 유지하고 신경 포화(neural saturation)를 방지한다(Tononi et al. 2006).

두 번째 핵심 기능은 기억 통합(memory consolidation)으로, 이는 취약하고 최근의 경험을 안정적이고 장기적인 지식으로 변형하는 것이다(Squire et al. 1995). 이 과정은 해마(hippocampus)와 신피질(neocortex) 사이의 정교한 대화를 통해 조율된다(Squire et al. 2015). 해마는 특정한 일상적 경험을 빠르게 부호화할 수 있는 고충실도(high-fidelity)의 임시 저장 시스템 역할을 한다. 이와 대조적으로, 신피질은 이러한 경험들로부터 일반화된 규칙과 의미론적 지식을 점진적으로 학습하는 데 더 적합한, 방대한 장기 저장소이다(Squire et al. 1995, 2015). 서파 수면 동안, 뇌는 이 구조들 사이에서 정보의 복잡한 전이를 촉진하는 야간의 대화를 개시한다. 특히, 이 전이는 단순히 원시 데이터(raw data)를 리플레이하는 것이 아니다; 오히려 깨어 있는 시간 동안 획득한 지식을 재구조화(re-architects)하여, 추상들을 추출하고 그것들을 응집된 의미론적 네트워크로 통합한다.

**급속 안구 운동 수면:** 깨어 있는 상태를 닮은 고주파, 저진폭 뇌파를 특징으로 하는 REM 수면은 가장 흔히 꿈꾸기와 관련된다. 기능적으로, 이 단계는 새로 형성된 시냅스의 선택적 강화 및 새로운 정보와 기존의 정서적·의미론적 네트워크의 통합과 연결된다. 더 나아가, 그것은 적응적 행동을 개선하기 위해 미래의 시나리오를 시뮬레이션하는 역할을 한다고 가설된다.

요약하면, 밤새 NREM과 REM 단계 사이의 주기적 교대는 결정적이다. NREM 수면은 그날의 경험을 통합하고 가지치기(prune)하여 보다 효율적인 지식 기반을 구축하는 것으로 보인다. 그 뒤에, REM 수면은 이 정제된 기반 위에서 작동하여, 새로운 연결을 탐색하고 두드러진(salient) 신경 경로를 강화하는 것으로 보인다.

**지속 학습에는 학습 시간이나 테스트 시간이 없다**
지속 학습자에게는, 학습 시간과 테스트 시간 사이에 명확한 경계나 구분이 없다. 모델은 오직 두 가지 서로 다른 상태만을 경험한다: 입력으로 정보를 받을 때, 혹은 고립된 학습 시스템일 때.

**지속 학습에서의 활성(깨어 있는) 및 잠 단계**
활성 또는 깨어 있는 시간에, 지속 학습자는 새로운 입력 데이터를 받고 필요에 따라 그것을 처리한다. 그러나 잠 시간에는, 모델이 최소한의(또는 아무런) 입력 데이터를 받으며 기존 지식을 처리하여 기억을 통합하고 스스로를 개선하는 데 집중한다.

### 기여 (Contributions)

인간의 기억 처리에서 영감을 받아, 우리는 지속 학습자에게는 학습 시간이나 테스트 시간이 없다고 주장한다. 대신, 모델은 주기적으로 "활성" 혹은 "잠"이라는 두 단계를 가질 필요가 있다; 활성 상태에서 모델은 새로운 데이터를 받아 처리하고, 잠 단계에서는 초점이 내부 지식과 최근 기억의 통합에 맞춰진다. 이를 위해, 우리는 두 개의 통합된 단계로 구성된 LLM용 "잠" 패러다임의 한 사례를 소개한다:

i. **기억 통합(Memory Consolidation):** 파국적 망각(CF)을 완화하고 더 높은 수준의 추상을 포착하기 위해, 기억 통합 단계에서 모델은 주기적 과정을 사용한다. 이 과정에서 모델은 새로운 파라미터를 활성화/잠금 해제(unlock)하고, 더 높은 주파수(즉, 더 빠르게 업데이트되는) 레이어/모듈로부터 보다 안정적인(더 낮은 주파수) 레이어/모듈 내의 새로 잠금 해제된 파라미터로 지식을 증류한다. 이 과정은 새로운 파라미터에 대해서는 충분한 가소성(plasticity)을 허용하면서도 오래된 파라미터의 안정성을 보장하여, 오래된 지식을 보존한다.

ii. **꿈꾸기를 통한 자기 개선(Self-Improvement via Dreaming):** 앞선 잠의 첫 번째 단계가 지식 추상을 장기 기억으로 전이하는 것을 보장하는 반면, 이 단계는 재귀적 자기 개선(recursive self-improvement) 과정을 담당한다. 특히, 모델의 현재 상태가 주어지면, 그것은 일련의 꿈들(dreams)(즉, 합성적으로 스스로 생성한 데이터)을 생성하여, 특히 최근에 추가된 지식에 대한 더 많은 숙련도를 획득하는 데 초점을 두고 자신의 성능을 개선한다.

기술적 관점에서, 위 각 단계에 대한 우리의 기여는 다음과 같다:

1. **주기적 파라미터 (비)활성화(Periodic Parameter (De)Activation):** 각 구성요소가 자신만의 업데이트 주파수를 가질 수 있게 하는 중첩 학습(Nested Learning, NL) 패러다임(Behrouz et al. 2025)을 기반으로, 우리는 주기적이고 점진적인 파라미터 (비)활성화 과정을 제안한다. 이 과정에서는, 하나의 블록이 주어지고 각 잠 단계마다, 우리는 더 빠른 블록(즉, 더 높은 주파수 블록) 내의 일련의 파라미터를 비활성화하고 그것들을 현재 블록 내에서 새로 활성화된 일련의 파라미터로 대체한다. 이는 이전 파라미터와의 지식 간섭(interference)을 피하면서 가소성을 유지할 수 있게 한다.

2. **지식 씨앗화(상향식 증류)(Knowledge Seeding (Upward Distillation)):** 우리는 지식 씨앗화라 부르는 새로운 형태의 지식 전이를 제시하는데, 여기서 하나 또는 몇 개의 더 작은 모델이 자신의 지식을 더 큰 모델로 증류한다. 이 설계는 더 큰 모델이 자신의 더 큰 용량을 활용하면서도 더 작은 모델들 내의 기존 지식을 보존할 수 있게 한다. 지식 씨앗화(Knowledge Seeding, KS)의 형식화에 기반하여, 우리는 자기 지식 씨앗화(self-Knowledge Seeding, SKS)를 제시한다. 여기서 모델의 더 작은 버전(예: 일부 파라미터가 비활성인)이 모델의 더 큰 버전(예: 파라미터가 활성인)으로 지식을 증류한다. 그런 다음 우리는 SKS를 기억 통합의 해법으로 사용하는데, 이때 모델은 자신의 지식을 고주파 레이어/블록으로부터 저주파 블록으로 증류한다.

3. **모방 학습을 결합한 일반화된 지식 증류(Generalized Knowledge Distillation, GKD):** SKS의 형식화는 일반적이며, 어떤 형태의 목적함수(objective)와 지식 전이 방법이든 사용될 수 있다. 여기서 우리는 온-폴리시 증류(on-policy distillation)와 모방 학습(imitation learning) 과정을 결합하는 새로운 목적함수를 제시한다. 특히, 우리의 모방 학습 과정에서, 우리는 교사(teacher)(즉, (압축된) 특권 정보(privilege information)를 가진 모델의 더 작은 버전)가 합성 데이터를 생성한 다음 시퀀스를 마스킹(mask)할 것을 제안한다; 그러면 학생(student)은 시퀀스의 연속을 예측하는 것을 목표로 하며, 교사가 생성한 데이터와 자신의 예측 사이의 거리(distance)에 기반하여 보상(reward)을 받는다.

4. **실험적 평가(Experimental Evaluation):** 우리는 일련의 도전적인 다운스트림(downstream) 과제들에 대해 잠 패러다임의 효과성을 평가한다: (1) 사실적 지식 통합(Factual Knowledge Incorporation); (2) 소수샷 학습(Few-shot Learning); (3) 장문맥 이해(Long-context Understanding); 그리고 (4) 지속 학습(Continual Learning). 그 결과들은 잠 패러다임의 효과성뿐만 아니라, 지속 학습을 위한 반복적 지식 증류를 통한 파라미터 성장(growing parameters)의 중요성을 뒷받침한다.

## 2. 예비 지식 및 문제 정식화 (Preliminaries and Problem Formulation)

### 2.1 표기법 (Notation)

우리는 벡터(각각 행렬)에 대해 굵은 소문자(각각 대문자)를 사용하며, 시간 t에 해당하는 개체의 상태를 가리키기 위해 아래첨자 t를 사용한다. 모듈의 파라미터(각각 하이퍼파라미터)에 대한 위첨자는 모듈의 업데이트 주파수를 결정하는 데(각각 서로 다른 인스턴스를 구별하는 데) 사용된다. 논문 전반에 걸쳐, 우리는 x ∈ R^{L×din}을 입력으로, K를 키(keys), V를 값(values), Q를 시퀀스 모델에서의 쿼리(query) 행렬로 두며, L은 시퀀스 길이를 나타낸다. 필요할 때, 우리는 언어 모델 LM_θ를 다음과 같이 파라미터화한다:

θ = {W_1^{(f_1)}, . . . , W_{k_1}^{(f_1)}} ∪ {W_1^{(f_2)}, . . . , W_{k_2}^{(f_2)}} ∪ . . . {W_1^{(f_c)}, . . . , W_{k_c}^{(f_c)}},

여기서 파라미터 집합들은 그것들의 가중치 업데이트 주파수 f_1 ≥ . . . ≥ f_c 에 기반하여 정렬된다(정의 1(Definition 1) 참고).

### 2.2 연속체 기억 시스템 (Continuum Memory System)

트랜스포머 아키텍처는 두 가지 결정적 구성요소로 이루어진다: (1) 연상 기억(associative memory) 역할을 하며 출력을 문맥 내의 과거 토큰들에 조건화하는 어텐션(Attention) 모듈, 이는 또한 문맥 내 학습 능력을 낳는다; 그리고 (2) 학습 단계 이후 고정되며 사전 학습에 걸쳐 획득한 지식을 부호화하는 MLP 또는 순방향(feedforward) 레이어들. Behrouz et al. (2025)에서 논의된 바와 같이, 이러한 아키텍처는 두 수준의 기억 시스템(two-level memory systems)으로 해석될 수 있다. 여기서 어텐션의 업데이트 스팬(update span)은 문맥 길이이며—즉, 문맥의 끝에서 그에 해당하는 파라미터가 업데이트되고 획득한 지식이 망각된다—MLP의 업데이트 스팬은 존재하지 않는다—즉, 사전 학습 이후 업데이트가 없음을 나타낸다. 이 관점에서, 이 두 구성요소는 주파수 스펙트럼의 양극단으로서, 어텐션(각각 MLP)은 무한(각각 0)의 업데이트 주파수를 가진다.

이 직관에 기반하여, Behrouz et al. (2025)은 연속체 기억 시스템(Continuum Memory System, CMS)을 제시하였다. 여기서 아키텍처는 어텐션과 같은 시퀀스 모델에, 각각 자신만의 주파수로 업데이트되는 MLP 레이어들의 체인(chain)이 뒤따르는 형태이다. 더 구체적으로, 가장 느린 모듈에서의 한 단계 업데이트에 걸리는 시간을 시간의 단위로 간주하며, 따라서 다른 구성요소들의 업데이트 속도(update rate)는 다음과 같이 정의된다:

**정의 1 (갱신 빈도, Update Frequency).** $W$의 임의의 가중치 성분에 대해, 우리는 그 빈도를 $f_W$로 표기하며, 단위 시간당 갱신 횟수로 정의한다.

이 개념을 더 잘 이해하기 위해, 입력이 길이 $L$의 시퀀스인 Fast-weight Programs(빠른-가중치 프로그램)(Schmidhuber 1992)의 간단한 예를 사용한다. 이 경우, 느린-가중치(slow-weight)의 각 스텝(단위 시간)마다 빠른-가중치(fast-weight)는 $L$번 갱신되며, 결과적으로 갱신 빈도는 $L$이 된다.

빈도에 대한 이 정의를 따르면 — 이는 고수준에서 한 모듈의 파라미터가 시간에 걸쳐 얼마나 자주 갱신되는지를 나타낸다 — CMS는 MLP 블록들의 사슬(chain) $\text{MLP}^{(f_1)}(\cdot), \ldots, \text{MLP}^{(f_k)}(\cdot)$로 형식화되며, 각각은 하나의 청크(chunk)와 연관된다. 그 청크 크기는

$$C^{(\ell)} := \frac{\max_{\ell'} C^{(\ell')}}{f_\ell}$$

로 주어진다. 이때 입력 $x = \{x_1, \ldots, x_T\}$가 주어지면 사슬의 출력은 다음과 같이 계산된다(명료함을 위해 정규화는 무시한다):

$$y_t = \text{MLP}^{(f_k)}(\text{MLP}^{(f_{k-1})}(\cdots \text{MLP}^{(f_1)}(x_t))), $$

여기서 $\ell$번째 MLP 블록의 파라미터, 즉 $\boldsymbol{\theta}^{(f_\ell)}$는 매 $C^{(\ell)}$ 스텝마다 갱신된다: 즉 $\boldsymbol{\theta}_{i+1}^{(f_\ell)} = \boldsymbol{\theta}_i^{(f_\ell)} - \boldsymbol{e}_{i,\ell}$ 이며 여기서

$$\boldsymbol{e}_{i,\ell} = \begin{cases} \sum_{t=i-C^{(\ell)}}^{i} \eta_t^{(\ell)} f(\boldsymbol{\theta}_t^{(f_\ell)}; x_t) & \text{if } i \equiv 0 \pmod{C^{(\ell)}}, \\ 0 & \text{otherwise.} \end{cases} $$

여기서 $\eta_t^{(\ell)}$은 $\boldsymbol{\theta}^{(f_\ell)}$에 대응하는 학습률(learning rate)이며, $f(\cdot)$는 신경망의 종단간(end-to-end) 최적화에서 임의의 옵티마이저(optimizer)의 오차 성분(error component)이다(예: 경사하강법(gradient descent)의 경우 $\nabla \mathcal{L}(\boldsymbol{\theta}_t^{(f_\ell)}; x_t)$). CMS의 형식화는 매우 광범위하며, 과업에 따라 목적함수 $\mathcal{L}(\cdot)$는 변경될 수 있다(Transformer의 MLP 블록과 동일하게). 이러한 형식화의 일반성과, Transformer 설계의 상위집합(superset)이라는 점(즉, 1개의 MLP 블록은 Transformer와 동등해진다) 때문에, 본 논문 전체에 걸쳐 우리는 CMS를 아키텍처의 기본(default) 구성 요소로 사용한다. 또한, 명료함을 위해 그리고 일반성을 잃지 않고, 우리는 $C^{(\ell)}$이 $C^{(\ell-1)}$로 나누어떨어진다고 가정한다. 주목할 점으로, 식 2는 중요한 해석을 제공한다: 파라미터 $\boldsymbol{\theta}_t^{(f_\ell)}$는 자신의 문맥(context)을 자신의 파라미터로 압축하는 역할을 하며, 따라서 이들은 자신의 문맥의 추상적 지식(abstract knowledge)을 대표한다(더 자세한 내용은 Nested Learning 논문(Behrouz et al. 2025)을 참조하라).

요약하면, 이 관점에서 시퀀스 모델(예: 어텐션(attention)(Vaswani et al. 2017) 또는 기타 메모리 모듈이나 RNN(Katharopoulos et al. 2020; Behrouz et al. 2024))은 모델의 단기 기억(short-term memory) 역할을 하는데, 이는 그들의 고빈도(high-frequency) 갱신이 오래된 지식을 밀어내어 잊혀지게 하고 새로운 기억을 위한 공간을 만들 수 있기 때문이다. 반면에, CMS 블록들은 메모리 모듈들의 스펙트럼(spectrum) 역할을 하는데, 여기서 이른(earlier) 블록들(더 높은 빈도)은 더 단기적인 기억이고, 나중(later) 블록들(그리고 궁극적으로 빈도가 0에 가까운 마지막 블록)은 더 장기적인 기억이다. 이 메모리 시스템을 능동적으로 갱신하는 것은 파국적 망각(Catastrophic Forgetting, CF)에 대한 저항성을 강화할 수 있지만, 모든 모델의 갱신 주기가 어느 시점에서 일치할 때 CF가 발생할 수 있다(Behrouz et al. 2025). 따라서, 각 메모리 블록의 갱신 전에, 그 블록의 추상화된 지식을 더 안정적인 파라미터로 통합(consolidate)하는 메커니즘이 있는 것이 매우 중요하다.

### 2.3 문헌에서의 Sleep(수면) 용어

인간의 수면 과정은 문헌의 많은 연구 설계에 영감을 주어 왔다(McClelland et al. 1995; Kumaran et al. 2016; Hassabis et al. 2017; Ha et al. 2018b). 특히, "꿈꾸기(dreaming)"는 문헌의 여러 과거 연구들이 최근 경험/입력 데이터를 재현(replay)하는 방법을 설계하도록 동기를 부여했다(Lin 1992; Mnih et al. 2015; Ha et al. 2018a,b). 여러 연구들은 모델을 장기 시야(long-horizon) 과업에 대해 더 견고하게 만들 수 있는 오프라인 과정을 설계했다(Gonzalez et al. 2020; Hafner et al. 2020; Tadros et al. 2022). Lin et al. (2025)은 "수면-시간 연산(sleep-time compute)"의 한 형태로서, 다음 사용자 세션을 위해 과거 문맥을 요약하는 오프라인 자기-학습(self-study) 과정을 제안한다.

우리가 아는 한, 기존 문헌(수면-영감(sleep-inspired) 연구들을 포함하여)은 훈련(training)과 테스트(testing) 단계라는 관습적 구분에 확고히 정박해 있다. 심지어 지속(continual) 또는 온라인(online) 학습 설정에서도, 모델들은 파라미터 갱신(훈련)과 성능 평가(테스트) 사이를 번갈아 오간다. 우리는 평생 적응(lifelong adaptation)을 위해서는, 정적인 훈련/테스트 패러다임이 연속적이고 주기적인 "각성(wake)"과 "수면(sleep)" 생애주기(lifecycle)로 대체되어야 한다고 주장한다. "각성" 단계에서 모델은 변화하는 입력 데이터와 능동적으로 상호작용하며 일시적 정보를 빠르게 획득하는 반면, "수면" 단계에서 모델은 자신의 기억을 통합하고 기존 지식을 내부적으로 처리한다.

## 3 Sleep 패러다임 (The Sleep Paradigm)

### 3.1 지속 학습에서의 학습 단계 (Learning Phases in Continual Learning)

앞서 논의했듯이, 지속 학습자(continual learner)는 항상 데이터/경험으로부터 학습하고 있다. 따라서, 머신러닝 모델의 생애주기에 대한 관습적 분할(즉, 테스트 시간과 훈련 시간)은 지속 학습 설정에 직접 적용될 수 없다. 우리는 지속 학습자의 생애주기의 두 가지 핵심 단계로서 "능동 또는 각성(active or wake)" 시간과 "수면(sleep)" 시간의 사용을 제안한다. 특히, 능동 또는 각성 시간에는 지속 학습자가 새로운 입력 데이터를 받아 필요에 따라 처리한다. 그러나 수면 시간에는 모델이 최소한의(또는 전혀 없는) 입력 데이터를 받으며 기억을 통합하고 자기-개선(self-improve)하기 위해 기존 지식을 처리하는 데 집중한다.

그러나, 기억 통합(memory consolidation) 과정은 수면 단계에만 국한되지 않는다. 사실, Nested Learning(NL)(Behrouz et al. 2025)의 논의를 따르면, 기억 통합에는 두 가지 형태가 있다: (1) 온라인 통합(Online consolidation); 그리고 (2) 오프라인 통합(Offline consolidation):

**능동(각성) 단계에서 신경망의 온라인 통합 (Online Consolidation in Neural Networks in Active (Wake) Stage)**
온라인 통합은 각성 또는 능동 단계에서 신경 학습 모듈(예: 그림 1의 Hope 아키텍처)의 종단간(end-to-end) 학습 과정을 통해 발생하는데, 여기서 이른 층들(더 높은 빈도의 더 빠른 블록들)이 그들의 지식을 나중 층들(즉, 더 낮은 빈도의 더 느린 블록들)로 전달한다.

**수면 단계에서 신경망의 오프라인 통합 (Offline Consolidation in Neural Networks in Sleep Stage)**
오프라인 통합 단계에서는, 모델이 수면 단계로 진입하여 새로운 데이터의 처리를 멈추지만, 자기-생성(self-generated) 데이터를 사용하여 최근 기억들을 통합한다.

다음 절에서, 우리는 Sleep 패러다임을 제시하는데, 여기서는 모델의 각성 시간(또는 능동 시간)과는 반대로, 모델이 어떠한 외부 입력 데이터도 받지 않고 자기-개선, 과거 기억의 통합, 그리고 지식의 추상화에 자신의 내부 연산을 집중시킨다. 특히, 우리는 수면 과정을 두 가지 핵심 단계로 나눈다: (1) 기억 통합(Memory consolidation); 그리고 (2) 자기-개선을 위한 꿈꾸기(Dreaming for self-improvement).

**그림 2:** 기억 통합(Memory Consolidation)의 개요. 모델은 용량을 강화하기 위해 자신의 파라미터 수를 늘리고(3.2절), 그런 다음 지식 시딩(knowledge seeding)을 사용하여 지식 추상화를 더 높은 빈도의 메모리에서 더 낮은 빈도의 메모리로 전달한다(3.3절).

### 3.2 기억 통합: 파라미터 확장 (Memory Consolidation: Parameter Expansion)

앞서 논의했듯이, 기억 통합에서 우리는 단기적이고 취약한 기억을 더 방대하고 안정적인 파라미터로 전달하는 것을 목표로 한다. CMS 형식화의 중요한 메시지 중 하나는: 기억의 취약성(fragility) 및/또는 안정성(stability)은 상대적이라는 것이다. 즉, 각 메모리 블록에 대해, 다른 더 높은 빈도의 메모리들은 더 단기적이고 더 취약하다. 따라서, 기억 통합은 단순한 2단계 과정이 아니라, 더 높은 빈도의 메모리에 저장된 지식을 더 안정적인 더 낮은 빈도의 파라미터로 반복적으로 전달하는 반복적(iterative) 연산이다.

더 빠르게 갱신되는 블록(그 예로 인-컨텍스트 학습(in-context learning)(Brown et al. 2020)이 있다)의 지식을 잃지 않기 위해, 우리는 그 파라미터를 갱신하기 전에 기억 통합 단계를 수행할 필요가 있다. 따라서, 청크 길이의 리스트 $\{C^{(1)}, \ldots, C^{(k)}\}$가 주어지면, 수면 과정 및 이에 따른 기억 통합은 모든 $b \in \mathbb{N}$에 대해 $\{C^{(1)} \times b, \ldots, C^{(k)} \times b\}$ 스텝에서만 발생한다. MLP 블록들의 갱신 빈도에 따라, 우리는 한 메모리 모듈의 기억을 그 다음 메모리 블록으로 여러 번 통합해야 할 수도 있다. 예를 들어, 갱신 빈도가 1K인 메모리 뒤에 갱신 빈도가 10K인 메모리가 이어지는 경우를 생각해 보자: 이 경우, 더 빠른 메모리는 더 느린 메모리의 갱신 전에 10번 갱신되며, 이는 더 느린 메모리 자신의 갱신 전에 더 빠른 메모리에서 더 느린 메모리로의 10번의 기억 통합 스텝을 의미한다. 이 더 느린 메모리로의 여러 통합 스텝은, 파국적 망각(CF) 때문에 LLM의 지속 학습 능력을 잠금 해제하는 데 있어 중대한 병목이 될 수 있다. 이 현상은 모델의 제한된 용량(예: 파라미터 수)의 본질적 원인으로, 새로운 지식을 통합하기 위해 파라미터가 덮어써져야(overridden) 하는 것이다. 이 도전에 대한 인간 두뇌의 해결책의 근간은 신경가소성(neuroplasticity), 즉 경험에 반응하여 자신의 기능을 수정하고 새로운 연결을 형성하는 두뇌의 본질적 능력이다. 이에 영감을 받아, 우리는 모델이 새로운 연결을 형성하고 이로써 자신의 용량을 늘릴 수 있게 하는, 메모리 블록에서의 효율적인 점진적 파라미터 확장(gradual parameter expansion)을 제시한다.

일반성을 잃지 않고, 우리는 MLP 블록들 $\{\text{MLP}^{(f_\ell)}(\cdot)\}_{\ell=1}^{k}$이 라우터(router) $\mathcal{R}^{(f_\ell)}$를 갖는 희소 전문가 혼합(sparse mixture of experts, MoEs)이라고 가정한다: 즉, 각 $\text{MLP}^{(f_\ell)}(\cdot)$는 전문가 집합 $\{W^{(f_\ell),1}, \cdots, W^{(f_\ell),s_\ell}\}$를 포함하며, 여기서 $s_\ell \geq 1$은 사슬의 $\ell$번째 블록에서 현재 전문가의 수이다. $(\ell^* - 1)$을 우리가 그 지식을 인덱스 $\ell^*$를 갖는 바로 다음의 더 안정적인 메모리 모듈로 통합하고자 하는 메모리(또는 MLP)의 인덱스라고 하자. $\text{MLP}^{(f_{\ell^*})}(\cdot)$에서 전달된 지식과 이전에 저장된 지식의 간섭을 피하기 위해, 우리는 그것의 파라미터 집합에 새로운 저랭크(low-rank) 전문가를 추가한다. 즉, 우리는 $\{A^{(f_{\ell^*}),s_{\ell^*}+1}, B^{(f_{\ell^*}),s_{\ell^*}+1}\}$로 파라미터화된 저랭크 MLP를 전문가 집합에 추가하는데, 여기서 $A^{(f_{\ell^*})} \in \mathbb{R}^{d \times d_{\text{low}}}$이고 $B^{(f_{\ell^*})} \in \mathbb{R}^{d_{\text{low}} \times d}$이다($d_{\text{low}} \ll d$). 이 새로운 파라미터들은 $\text{MLP}^{(f_{\ell^*-1})}(\cdot)$로부터 새로 전달된 지식을 저장하기 위해 할당될 것이다. 이 과정에 의해, 각 수면 시간 이후, 층들의 부분집합의 파라미터가 증가한다.

### 3.3 지식 시딩을 통한 기억 통합 (Memory Consolidation with Knowledge Seeding)

이 단계에서, 우리는 파라미터 $\boldsymbol{\theta}^{(f_{\ell^*-1})}$를 갖는 $\text{MLP}^{(f_{\ell^*-1})}(\cdot)$의 지식을 $\text{MLP}^{(f_{\ell^*})}(\cdot)$의 확장된 파라미터 집합으로 전달하는 것을 목표로 한다. 우리는 $\text{LM}_{\boldsymbol{\theta}}$를 파라미터 확장 전 언어 모델의 상태로, $\text{LM}_{\boldsymbol{\theta}_{\text{exp}}}$를 (i) 파라미터 확장, 그리고 (ii) 식 2에 기반한 $\boldsymbol{\theta}^{(f_{\ell^*-1})}$의 갱신 이후 언어 모델의 상태로 둔다. 주목할 점은, 수면 및 이에 따른 기억 통합이 $\text{MLP}^{(f_{\ell^*-1})}(\cdot)$에 대해 발생하고 있으므로, 과거 스텝 수는 $C^{\ell^*-1}$로 나누어떨어지고 따라서 이 메모리 블록은 갱신된다는 것이다. 우리는 기억 통합 과정을, 더 작은 모델 상태 $\text{LM}_{\boldsymbol{\theta}}$에 저장된 지식을 $\text{LM}_{\boldsymbol{\theta}_{\text{exp}}}$의 더 큰 변형으로 전달하는 것을 목표로 하는 증류(distillation) 문제로 모델링한다.

**지식 시딩 (Knowledge Seeding).** 우리는 지식 시딩(knowledge seeding)이라 불리는 새로운 형태의 지식 전달을 제시하는데, 여기서 하나 또는 몇몇의 더 작은 모델들이 그들의 지식을 더 큰 모델로 증류한다. 이 설계는 더 큰 모델이 더 작은 모델들의 기존 지식을 보존하면서도, 자신의 더 큰 용량의 이점을 취할 수 있게 한다.

**상향 증류 (Upward Distillation) (지식 시딩)**
교사(teacher)로서 작은 모델들의 집합 $\mathcal{S}_1(\cdot), \ldots, \mathcal{S}_k(\cdot)$가 주어지면, 지식 시딩(KS)은 이 모델들의 지식을 더 큰 모델 $\mathcal{M}(\cdot)$으로 전달하는 것을 목표로 한다.

지식 시딩(KS)의 형식화에 기반하여, 우리는 자기-지식 시딩(self-Knowledge Seeding, SKS)을 제시하는데, 여기서 한 모델의 더 작은 버전(위에서 논의한 대로)이 그 모델의 더 큰 버전으로 지식을 증류한다. 이 증류 과정에는 두 가지 중대한 도전이 있다: (1) 관습적 경우와는 반대로, 학생(student)이 더 많은 용량과 따라서 교사보다 더 큰 표현력(expressive power)을 갖는다. 따라서, 교사가 생성한 데이터셋으로 학생을 훈련시키는 것(예: Kim et al. (2016))은 학생 모델에서 파라미터의 준최적(sub-optimal) 사용을 초래할 수 있다; (2) 모델이 수면 단계에 있으므로 외부 정보/데이터셋에 대한 접근이 제한된다. 따라서, Hinton et al. (2015)과 같은 대부분의 인기 있는 방법들은 적용할 수 없다. 이러한 도전들을 극복하기 위해, 우리는 정책 위(on-policy) 학생-생성 데이터와 교사-생성 데이터의 혼합을 허용하는 일반화 지식 증류(Generalized Knowledge Distillation, GKD)(Agarwal et al. 2024)를 기반으로 삼아, 모방 학습(imitation learning)에 기반한 새로운 증류 과정을 제시한다.

앞서 논의했듯이, 기억 통합 단계는 원시(raw) 데이터를 단순히 재현해서는 안 된다; 대신, 능동(각성) 스텝 동안 획득된 지식의 추상화를 탐색하고 추출해야 한다. 이를 위해, 지식 시딩은 두 가지 주요 단계를 갖는다: (1) 증류 과정으로, 여기서 학생은 자기-생성 시퀀스에 대해 교사의 로짓(logits)으로부터 토큰별(token-specific) 피드백을 받는다; 그리고 (2) RL 기반 모방 학습 방법으로, 이는 학생이 교사의 샘플링된 출력을 기억하도록 강제하여, 증류된 지식을 보존하면서 그들의 샘플링 과정을 정렬(align)한다.

우리는 교사 모델, 즉 $\text{LM}_{\boldsymbol{\theta}}$로부터 샘플링하여 데이터셋 $\mathcal{D}$를 구성하는 것으로 시작한다. 다음으로, GKD(Agarwal et al. 2024)와 유사하게, 우리는 정책 위(on-policy) 증류 목적함수를 다음과 같이 정의한다:

$$\mathcal{L}(\boldsymbol{\theta}, \boldsymbol{\theta}_{\text{exp}}) = (1-\lambda)\, \mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\mathcal{F}(\text{LM}_{\boldsymbol{\theta}} \| \text{LM}_{\boldsymbol{\theta}_{\text{exp}}})(y|x)\right] + \lambda\, \mathbb{E}_{x\sim\mathcal{D}}\, \mathbb{E}_{y\sim\text{LM}_{\boldsymbol{\theta}_{\text{exp}}}(\cdot|x)}\left[\mathcal{F}(\text{LM}_{\boldsymbol{\theta}} \| \text{LM}_{\boldsymbol{\theta}_{\text{exp}}})(y|x)\right].$$

여기서 $\mathcal{F}(\text{LM}_{\boldsymbol{\theta}}, \text{LM}_{\boldsymbol{\theta}_{\text{exp}}})(y|x)$는 교사(즉, $\text{LM}_{\boldsymbol{\theta}}$)와 학생(즉, $\text{LM}_{\boldsymbol{\theta}_{\text{exp}}}$)의 출력 분포 사이의 발산(divergence)이며, $\lambda \in [0,1]$은 정책 위 학생-생성 출력의 비율을 제어한다. 이 최적화 과정에서, 우리는 학생의 샘플링 분포를 통해 역전파(backpropagate)하지 않는데, 이는 훈련 안정성 및 속도에도 도움이 될 수 있다. 또한, 우리는 학생 모델의 모든 파라미터를 동결(freeze)하고 확장된 파라미터만 갱신한다. 이는 전달된 지식이 오래된 지식과 간섭하여 파국적 망각을 일으키지 않도록 보장한다.

**모방하는 법 학습하기 (Learning to Imitate).** 위의 증류 과정은 학생의 새로운 파라미터가 더 낮은 빈도의 메모리에 인코딩된 지식을 저장하도록 보장한다. 그러나, 우리는 지식에 대한 접근을 가짐에도 불구하고, 학생 모델이 그것을 사용하는 법을 학습하지 못하여 교사의 샘플링과 성능을 약하게 모방한다는 것을 관찰한다. 이를 위해, 우리는 RL을 통합하여 모델이 교사의 샘플링을 모방하는 법을 가르침으로써 위의 증류 과정을 더욱 개선한다. 교사가 생성한 데이터(꿈, dreams)의 집합 $\mathcal{D}_T = \{d^{(1)}, \ldots, d^{(n)}\}$가 주어지면, 모방하는 법 학습하기(Learning to Imitate, LTI) 과정은 먼저 각 $d^{(i)}$에서 접두부(prefix)를 무작위로 샘플링한 다음 학생 모델에게 그 연속(continuation)을 완성하도록 요청한다. 학생 응답 $\hat{d}^{(i)}$가 주어지면 할당된 보상(reward)은 다음과 같이 정의된다:

$$r(\hat{d}^{(i)}; d^{(i)}; \text{LM}_{\boldsymbol{\theta}_{\text{exp}}}) = \gamma \times r^{\text{sem}}(\hat{d}^{(i)}; d^{(i)}; \text{LM}_{\boldsymbol{\theta}_{\text{exp}}}) + (1-\gamma) \times r^{\text{abs}}(\hat{d}^{(i)}; d^{(i)}; \text{LM}_{\boldsymbol{\theta}_{\text{exp}}}), $$

여기서 $r^{\text{sem}}(\cdot;\cdot;\cdot)$(각각 $r^{\text{abs}}(\cdot;\cdot;\cdot)$)는 의미적 유사도(semantic similarity)(각각 절대 토큰 수준 유사도(absolute token-level similarity))에 기반하여 보상을 할당한다. 의미적 유사도의 경우, 우리는 동결된 보상 모델을 사용하며, $\hat{d}^{(i)}$와 $d^{(i)}$의 의미가 같으면(각각 그렇지 않으면) 학생에게 1(각각 0)의 보상을 준다. 반면에, 절대 보상은 두 시퀀스의 레벤슈타인 거리(Levenshtein distance)($z(\cdot,\cdot)$로 표기)에 기반하여 정의된다: 즉, $r^{\text{abs}}(\hat{d}^{(i)}; d^{(i)}; \text{LM}_{\boldsymbol{\theta}_{\text{exp}}})$는 다음과 같이 정의된다:

$$r^{\text{abs}}(\cdot) = \begin{cases} 1 - \dfrac{z(\hat{d}^{(i)}, d^{(i)})}{\max\{|\hat{d}^{(i)}|, |d^{(i)}|\}} & \text{if } z(\hat{d}^{(i)}, d^{(i)}) \leq z_0, \\ 0 & \text{otherwise,} \end{cases} $$

여기서 $z_0$은 유사도 임계값(similarity threshold)이다. 위의 LTI 과정을 정책 위(on-policy) 증류에 통합함으로써, 지식 시딩(KS) 목적함수는 다음과 같이 정의된다:

$$\mathcal{L}_{\text{KS}}(\boldsymbol{\theta}, \boldsymbol{\theta}_{\text{exp}}) = \mathbb{E}_{x\sim\mathcal{D}}\left[(1-\alpha)\, \mathbb{E}_{y\sim\text{LM}_{\boldsymbol{\theta}_{\text{exp}}}(\cdot|x)}[r(y)] - \alpha\, \mathbb{E}_{y\sim\text{LM}_{\boldsymbol{\theta}_{\text{exp}}}(\cdot|x)}\, \mathcal{D}(\text{LM}_{\boldsymbol{\theta}} \| \text{LM}_{\boldsymbol{\theta}_{\text{exp}}})(y|x)\right].$$

여기서 $\alpha \in [0,1]$은 LTI 목적함수 대비 증류의 강도를 제어한다. 이 목적함수에 기반하여, 우리는 모델의 새로운 확장된 파라미터를 갱신하고 고빈도 메모리의 기억/지식을 저빈도 메모리 블록으로 통합한다. 이제 $\text{MLP}^{(f_{\ell^*-1})}(\cdot)$의 기억들이 $\text{MLP}^{(f_{\ell^*})}(\cdot)$로 통합되었으므로, 우리는 이전에(과거 수면 기간에) $\text{MLP}^{(f_{\ell^*-1})}(\cdot)$에 추가되었던 모든 저랭크 파라미터를 재설정(reset)하여, 그것의 용량을 사용 가능하게 만든다

미래를 위해 유지한다. 이 단계는 인간 뇌에서 일어나는 시냅스 가지치기(synaptic pruning)와 유사한 절차로 해석할 수 있는데, 여기서 뇌는 불필요하고/또는 중복된 연결을 가지치기하여(Li et al. 2017) 그 효율성과 성능을 향상시킨다.

**구현에 관한 주석(Note on the Implementation).** 성장하는 희소 모듈(growing sparse modules)을 구현하는 것은, 만약 구현에서 텐서의 차원(dimensionality)을 직접 변경해야 한다면 극도로 어려울 수 있다. 대안적으로, 우리는 처음부터 그 파라미터들을 모델 안에 갖고 있되, 수면 단계에서의 최초 활성화 이전까지는 순전파(forward)와 역전파(backward) 과정에서 그것들을 마스킹(masking)할 수 있다. 흥미롭게도, 이는 또한 인간 뇌에 대한 우리의 이해와도 부합하는데, 뇌는 (크지만) 고정된 용량을 가지고 있으며 시간이 지남에 따라 새로운 구성 요소가 추가되지는 않는다. 대신, 뇌 영역들 사이의 새로운 연결이 우리의 삶을 통해 형성될 수 있으며, 이는 새로운 뉴런의 활성화를 잠금 해제(unlocking)하고 새로운 과제를 학습하기 위한 더 많은 가소성(plasticity)을 낳는다(Kandell et al. 2021).

### 3.4 꿈꾸기(Dreaming): 자기 수정(Self-Modifying) 과정

더 높은 빈도(higher-frequency)의 파라미터들을 동결(freezing)하고 그 지식을 더 낮은 빈도(lower-frequency)의 기억으로 증류(distilling)하는 것을 포함했던 이전 단계는, 기억 공고화(memory consolidation)를 담당하는 인간의 서파 수면(slow-wave stage of sleep, NREM) 단계와 유사하게 작동한다. 그러나 REM 단계에서는 뇌가 (심지어 깨어 있는 시간에 필적할 정도로) 매우 활성화되어 있으며, 꿈꾸기(dreaming)를 통해 새롭게 형성된 시냅스를 자기 수정하고 강화하는 것을 목표로 한다. 이로부터 영감을 받아, 우리는 시간이 지남에 따라 스스로를 개선하는 데 도움이 될 수 있는 꿈(합성 데이터, synthetic data)을 생성하는 방법을 학습하는 꿈꾸기 과정을 설계하는 것을 목표로 한다.

실제로는, 자기 개선(self-improvement)을 위한 어떠한 합성 데이터 생성 과정(예: Pang et al. (2024a), Huang et al. (2025), Zweiger et al. (2025))도 이 단계에 통합될 수 있다. 그러나 한 가지 중대한 고려 사항은 지속 학습(continual learning) 설정에서 자기 개선을 반복적으로 적용하는 데 따르는 위험으로, 이는 파국적 망각(catastrophic forgetting)을 야기할 수 있다(Zweiger et al. 2025). 우리의 평가에서, 우리는 수면을 기억 공고화로, 그다음에 꿈꾸기를 자기 수정 과정으로 설계한 우리의 2단계 설계가 어떻게 파국적 망각에 대해 더 강건한지를 보인다. 개념 증명(proof of concept)으로서, 우리는 Zweiger et al. (2025)의 연구인 SEAL을 기반으로 삼는다. 그러나 이를 우리의 수면 패러다임에 통합하는 데에는 세 가지 도전 과제가 있다: (1) SEAL의 내부 루프(inner-loop)에서 지도 미세조정(supervised fine-tuning, SFT)의 비용 때문에, 그것은 적은 수의 자기 편집(self-edits, 우리 용어로는 꿈)으로 제한된다. (2) 수면 기간 동안의 반복적 자기 개선의 원인으로서의 잠재적 파국적 망각. (3) 샘플링 과정은 오직 모델의 기존 지식 공간(existing knowledge space)에서만 샘플링하는 반면, 꿈꾸기의 핵심 역할 중 하나는 기억들의 새로운 종합(novel synthesis)을 탐색하는 것이다(Stickgold 2005).

샘플링된 과제 (C, τ)가 주어졌을 때, 여기서 C는 과제와 관련된 정보를 담고 있는 맥락(context)이고 τ(·)는 하위 평가(downstream evaluation)에서의 성능을 측정하는 척도이며, 우리의 "꿈꾸기" 과정은 C를 맥락에 두고 m ≥ 1개의 꿈을 생성하는 것으로 시작한다. 샘플링 과정에서, MoE 블록의 각 라우터(router)는 추가로 무작위 전문가(random expert)를 선택하고, 그리하여 무작위의 무관한 지식(random irrelevant knowledge)을 꿈꾸기에 통합함으로써, 모델의 시야에서 숨겨져 있던 근본 패턴을 학습한다. 이 단계에서, 우리는 {DREAM^(i)}_{i=1}^{m} ∼ LM_θ(·|C)로 둔다. 다음으로, 우리는 생성된 꿈들 중 일부를 기각(reject)하고 모델의 성능 개선에 가장 잠재력이 있는 샘플들만 유지한다. 이를 위해, 우리는 경사 기반 데이터 선택(gradient-based data selection)에 관한 문헌(Pan et al. 2024; Wang et al. 2024a)에서 영감을 얻는다: 각 꿈 DREAM^(i)에 대해, 우리는 중요도 점수(importance score) ω^(i)를 할당하고, 다양성을 유지하기 위한 b개의 무작위 샘플과 함께 가장 높은 중요도 점수를 가진 Top-k개의 꿈을 선택한다. 언어 모델링 목적함수 L_{SFT}(·)가 주어졌을 때, 우리는 DREAM^(i)의 중요도 점수를 g_{DR}^(i)로 표기하고 이를 목적함수의 경사(gradient)로 정의한다: g_{DR}^(i) = ∇_θ L_{SFT}(DREAM^(i), θ). 우리는 위 과정에 의해 선택된 모든 꿈의 집합을 D로 둔다. 각 DREAM^(i) ∈ D에 대해, 우리는 모델의 격리된 인스턴스(isolated instance)를 고려하고 지도 미세조정(LoRA (Hu et al. 2022) 사용)을 통해 그 파라미터를 갱신한다: 즉, θ′^(i) ← SFT(θ^(i), DREAM^(i)). 새롭게 미세조정된 모델이 주어졌을 때, SEAL(Zweiger et al. 2025)을 따라, 우리는 LM_{θ^(i)} 대비 LM_{θ′^(i)}의 성능 개선에 기반하여 DREAM^(i)의 생성에 보상을 준다:

r(DREAM^(i), τ(·), LM_{θ^(i)}) = 1  (개선되면),  0  (그렇지 않으면).   (5)

우리는 SEAL을 따르며 위 과정을 최적화하기 위해 ReST^EM 알고리즘(Singh et al. 2024a)을 사용한다.

**그림 3(Figure 3):** 텍스트 분류를 위한 클래스 증분 학습(class-incremental learning)이 (왼쪽) CLINC 데이터셋(Larson et al. 2019), (가운데) Banking 데이터셋(Casanueva et al. 2020), (오른쪽) DBpedia 데이터셋(Auer et al. 2007)에서 평가되었다. Hope 아키텍처는 다른 지속 학습 접근법들을 일관되게 능가하며, 가장 높은 정확도를 달성한다.

**그림 4(Figure 4):** (왼쪽) RULER의 MK-NIAH(Hsieh et al. 2024), (가운데) LongHealth(Adams et al. 2025), (오른쪽) QASPER(Dasigi et al. 2021)에 대한 인-컨텍스트 학습(in-context learning) 성능에 미치는 기억 수준(memory levels)의 효과. QASPER의 경우 값이 낮을수록 더 나은 성능을 나타낸다.

## 4 실험 결과(Empirical Results)

우리의 실험적 평가에서, 우리는 수면의 각 단계의 효과와 함께 모든 단계를 결합한 효과도 연구한다. 첫 번째 절에서, 우리는 지속 학습 능력과 장문맥(long-context) 이해를 향상시키는 것을 목표로 설계된 기억 공고화(memory consolidation)에 초점을 맞춘다. 추가적인 실험 결과는 부록 B(Appendix B)를 참조하라.

### 4.1 기억 공고화의 효과(The Effect of Memory Consolidation)

기억 공고화 과정의 주요 목표 중 하나는 장문맥 이해를 강화하면서 지속 학습을 향상시키는 것이다. 이 절에서, 우리는 지속 학습 및 장문맥 벤치마크에서 (자기 개선 없이) 기억 공고화 단계만을 평가한다.

**클래스 증분 학습(Class Incremental Learning).** 우리는 먼저 CLINC(Larson et al. 2019), Banking(Casanueva et al. 2020), DBpedia(Auer et al. 2007)의 세 가지 데이터셋에서의 클래스 증분 학습에 초점을 맞춘다(세부 사항은 B.1절 참조). 우리는 백본(backbone)으로 Llama-3B와 Llama3-8B(Dubey et al. 2024)를 사용한다. Hope는 온-폴리시 자기 증류(on-policy self-distillation)를 통해 추상화(abstraction)를 개선하는 우리의 기억 공고화 메커니즘으로 증강(augmented)된다. Momeni et al. (2025)를 따라, 우리는 ICL(수면이 없는 동일한 지속 사전학습 과정), 탄성 가중치 공고화(Elastic Weight Consolidation, EWC)(Kirkpatrick et al. 2017), 그리고 외부 학습기를 이용한 인-컨텍스트 지속 학습(In-context Continual Learning with an External Learner, InCA)(Momeni et al. 2025)과 비교한다. 우리는 또한 명시적 증류 과정 없이 다중 수준 인-컨텍스트 갱신(multi-level in-context updating) 베이스라인으로서 Hope(Behrouz et al. 2025)를 포함한다. 그림 3의 결과는 Hope가 외부 학습기(InCA) 및 정규화(EWC) 접근법을 포함하여 데이터셋 전반에서 가장 우수한 성능을 보임을 나타낸다. ICL 대비, 이득은 프롬프트 수준의 적응(prompt-level adaptation)을 공고화를 통해 지속적인 파라미터 기억(durable parametric memory)으로 변환하는 데서 온다. Hope 대비, 명시적 자기 증류는 반복적인 인-컨텍스트 갱신만으로 얻는 것보다 더 나은 추상화를 산출한다.

**수준의 수(#수면 단계)가 인-컨텍스트 학습에 미치는 효과(The Effect of Levels (#Sleep Phases) on In-context Learning).** Hope의 공고화 스케줄이 인-컨텍스트 학습과 장문맥 이해에 미치는 영향을 더 잘 분리하기 위해, 우리는 장문맥 하에서의 질의응답(question answering)과 다중 키 검색(multi-key retrieval)을 평가한다. 우리는 LongHealth(Adams et al. 2025), QASPER(Dasigi et al. 2021), MK-NIAH(Hsieh et al. 2024) 데이터셋을 사용한다(B.1절 참조). 베이스라인으로는 ICL, DuoAttention(Xiao et al. 2025), Cartridges(Eyuboglu et al. 2025)를 사용한다. Hope 변형들에 대해서는, (i) 얼마나 많은 공고화 단계가 사용되는지와 (ii) 가장 안정적인 기억의 지속성(persistence)—가장 낮은 공고화 빈도(lowest consolidation frequency)로 조작화됨—을 변경함으로써 수면 스케줄을 다양화한다. 직관적으로, 더 낮은 빈도는 더 지속적이지만 덜 적응적인 장기 기억(long-term memory)을 낳는다. 결과는 그림 4에 보고되어 있다. 모든 과제에 걸쳐, Hope는 ICL과 효율적인 DuoAttention 베이스라인을 일관되게 능가하며, 이는 수면 시(sleep-time) 공고화가 프롬프트 전용 적응(prompt-only adaptation)을 넘어 장문맥 행동을 개선함을 보여준다. 우리는 또한 Hope가 Cartridges(Eyuboglu et al. 2025)를 능가함을 관찰한다: Cartridges는 KV 표현을 압축하기 위해 보조 모델(auxiliary model)을 사용하여 효율성을 개선하는 반면, Hope는 대신 수면 동안 온-폴리시 자기 증류를 수행하여 새롭게 획득한 정보를 전이 가능한 파라미터 지식(transferable parametric knowledge)으로 공고화하고 더 강건한 장문맥 이해를 산출한다. Hope 변형들을 서로 비교하면, 우리는 두 가지 일관된 경향을 발견한다: (1) 공고화 단계의 수를 늘리면 인-컨텍스트 학습과 장문맥 이해가 개선되며, 이는 수면이 더 나은 지식 추상화 및 압축을 가능하게 하여 모델이 더 적은 유효 파라미터(effective parameters)로 더 많은 정보를 유지할 수 있게 한다는 관점을 뒷받침한다; 그리고 (2) 가장 낮은 빈도를 높이면 성능이 감소하는데, 이는 가장 지속적인 기억을 더 적응적으로 만드는 것이 유지(retention)를 약화시킨다는 것을 시사한다.

**표 1(Table 1):** 수학적 추론(mathematical reasoning) 벤치마크에 대한 서로 다른 방법들의 성능. 우리는 Qwen 모델의 다양한 변형을 사용하고 average@16을 보고한다.

| Method | AIME-24 | AIME-25 | HMMT-25 |
|---|---|---|---|
| Qwen3-8B | | | |
| Sleep | 79.2 | 69.0 | 46.1 |
| - Imitation Learning | 76.8 | 67.9 | 45.0 |
| - Semantic Reward | 78.9 | 69.2 | 44.5 |
| - w/o Expansion | 78.2 | 67.9 | 44.9 |
| OPSD | 76.6 | 67.4 | 45.1 |
| + Expansion | 77.9 | 68.2 | 45.9 |

**그림 5(Figure 5):** 새로운 언어의 지속 번역(Continual Translation of a Novel Language, CTNL) 과제. 빨간 점은 단일 언어로 훈련할 때의 성능을 나타내며, 파란 점은 지속 학습 하에서의 성능을 나타낸다.

**그림 6(Figure 6):** BABILong 벤치마크에 대한 결과. 빨간 점은 미세조정된 모델에 해당하고, 파란 점은 대규모 모델의 제로샷(zero-shot) 평가에 해당한다.

**인-컨텍스트로 새로운 언어 학습하기(Learning a New Language In-Context).** LLM은 종종 지속 설정(continual settings)에서 실패하는데, 여기서 모델은 이전에 획득한 지식을 덮어쓰지 않으면서 새로운 기술을 순차적으로 습득할 것으로 기대된다. 이 도전 과제를 연구하기 위해, 우리는 Behrouz et al. (2025)에 의해 도입된 과제를 따르는데, 이는 사전학습 동안 보지 못한 언어들에 대한 두 개의 번역 데이터셋인 MTOB(Tanzer et al. 2024)와 Manchu(Pei et al. 2025) 데이터셋을 결합한다. 즉, 모델은 이전에 보지 못한 두 언어에 인-컨텍스트로 노출되고 구문을 영어로 번역해야 한다. 우리는 두 가지 설정을 고려한다: 각 언어를 독립적으로 학습하고 평가하는 것, 그리고 각 언어에 대한 번역 성능을 평가하기 전에 두 언어를 순차적으로 학습하는 것이다. 우리는 표준 ICL을, 수면 시 공고화 단계의 수가 다른 Hope 변형들(Hope-1, Hope-2, Hope-3로 표기)과 비교한다. 결과는 그림 5에 나타나 있으며, 각 점은 Manchu→English(x축)와 Kalamang→English(y축)에 대한 ChRF 점수를 보고한다. 단일 언어 설정에서, 모든 Hope 변형은 ICL과 같거나 그것을 초과하며, 이는 공고화가 인-컨텍스트 적응을 방해하지 않음을 나타낸다. 지속 학습 설정 하에서, ICL은 급격한 성능 하락을 보이며 대체로 사전학습된 행동으로 되돌아가는 반면, Hope는 그 이득의 상당 부분을 유지한다. 성능은 추가적인 공고화 단계에 따라 단조적으로(monotonically) 개선되며, Hope-3은 순차적 노출에도 불구하고 단일 언어 성능을 거의 회복한다. 이러한 결과는 지속 학습에서 수면 시 공고화의 역할을 강조한다. ICL의 프롬프트 수준 갱신과 달리, Sleep은 유용한 추상화를 더 오래 지속되는 파라미터 기억으로 증류하는 명시적 자기 개선 단계를 도입하여, 파국적 망각 없이 효과적인 순차 학습을 가능하게 한다. 또 다른 베이스라인으로, 우리는 또한 Cartridges(Eyuboglu et al. 2025)와 언어들에 대한 지도 미세조정(Supervised Fine-Tuning, SFT)을 평가했다. 놀랍게도, 두 방법 모두 적어도 하나의 언어에서 파국적 망각을 겪었으며, 적어도 하나의 과제에서 ICL보다도 약한 성능을 보였다(그림 5에서 플롯 바깥에 위치함).

**BABILong.** 우리는 BABILong 벤치마크(Kuratov et al. 2024)에서 Hope(Sleep)를 평가하며, 다음과 비교한다: (1) 대규모 모델(GPT-4 및 GPT-4o-mini(Achiam et al. 2023)); (2) RAG를 갖춘 중간 규모 Llama-8B 모델(Dubey et al. 2024); 그리고 (3) RMT(Bulatov et al. 2022), ARMT(Rodkin et al. 2024), Titans(Behrouz et al. 2024)를 포함한 최첨단 소규모 장문맥 모델. 결과에 대한 상세한 논의는 B.2절에서 찾을 수 있다. 요약하면, Hope는 1000만(10M) 토큰으로의 확장에서 거의 완벽한 점수를 달성한다.

**추론을 위한 기억 공고화(Memory Consolidation for Reasoning).** 기억 공고화 단계의 또 다른 중요한 함의는 모델의 추론 능력을 개선하는 것이다. 이 절에서, 우리는 수학적 추론에 대한 그 효과를 평가하고 이를 기본 모델(base model), SFT, GRPO(Shao et al. 2024)의 일반적인 베이스라인과 비교한다. 결과는 표 2(Table 2)에 보고되어 있다. 기억 공고화를 위한 우리의 알고리즘은 기본 모델의 추론 능력을 개선하는 데 있어 SFT와 GRPO보다 더 나은 성능을 보인다.

**표 3(Table 3):** 구절 설정(Passage Settings) 전반에 걸친 지식 통합(Knowledge Incorporation) 성능.

**표 2(Table 2):** 수학적 추론 벤치마크에 대한 서로 다른 방법들의 성능. 우리는 Qwen 모델의 다양한 변형을 사용하고 average@16을 보고한다.

| Method | AIME-24 | AIME-25 | HMMT-25 |
|---|---|---|---|
| Qwen3-1.7B | | | |
| Base (Instruct) | 49.8 | 34.5 | |
| SFT | 47.3 | 36.1 | |
| GRPO | 51.0 | 38.6 | |
| OPSD | 51.6 | 40.0 | |
| Sleep | 53.2 | | |

40.2

25.7
22.9
26.1
28.1
29.3

68.1
66.4
68.1
67.4
69.0

42.4
43.7
44.9
45.1
46.1

단일 지문(Single Passage)
(n = 1)

계속된 사전학습(Continued Pretraining)
(n = 200)

기반 모델(Base model)
Dreaming 없이 미세조정된 모델(Fine-tuned Model with No Dreaming)
SEAL
Sleep (Transformer)
Sleep (Transformer + 4단계)

31.9
33.4
46.7
48.1
48.9

31.9
32.0
43.2
44.3
46.2

기울기 기반 선택 제거(removing gradient-based selection)
무작위 전문가 제거(removing random expert)
Dreaming 제거(removing Dreaming)

47.1
48.0
35.7

45.2
44.7
36.2

방법(Method)

표 4: 소수샷 추상 추론(Few-shot Abstract Reasoning)

Qwen3-8B
Base (Instruct)
SFT
GRPO
OPSD
Sleep

73.8
75.5
76.4
76.6
79.2

방법(Method)

성공률(Success Rate, %)

ICL
TTT
SEAL
Sleep

0
10
72.5
80

**지식 통합(Knowledge Incorporation).** 다음으로, 우리는 자기개선(self-improvement)까지 가능하게 하는 Sleep의 전체 설계에 초점을 맞춘다. 이 과제에서 우리는 모델이 통합된 사실들에 대한 질문에 답할 수 있기를 기대한다. 공정한 비교를 위해, 우리는 모델과 파라미터의 선택을 포함하여 Zweiger et al.(2025)의 실험 설정을 따른다. 우리는 SQuAD 데이터셋(Rajpurkar et al. 2016)의 새로운 사실 정보를 통합하는 것에 대해 우리 모델을 평가한다. 기준선(baseline)으로는 (i) 어떠한 개선도 없거나 지문에 접근할 수 없는 변형인 기반 모델(base model); (ii) dreaming이 없는 미세조정 모델(fine-tuned model with no dreaming); (iii) RL과 자기적응(self-adaption)을 갖춘 SEAL 모델; (iv) 2단계 메모리 시스템을 갖춘 우리의 Transformer 기반 아키텍처; 그리고 (v) 4단계 메모리 시스템을 갖춘 것을 사용한다.

표 3은 단일 지문(single-passage, n = 1)과 계속된 사전학습(continued pretraining, CPT, n = 200) 설정 모두에 대한 평균 무맥락(no-context) SQuAD 정확도를 요약한다. 우리의 sleep 과정은 다른 설정들 및 SEAL과 같은 최첨단(state-of-the-art) 방법들 가운데 최고의 결과를 달성한다. 우리는 이 결과를 다음에 기인한다고 본다: (1) 모델이 자신의 지식을 더 효과적으로 저장하도록 하는 메모리 통합(memory consolidation) 단계들; (2) 3.4절에서 논의한, SEAL 위에 얹은 우리의 개선들. CPT 체제(regime)에서, 모델은 단일 계속된 사전학습 실행 동안 n = 200개의 지문에 노출되며, 연관된 전체 974개 질문 집합에 대해 평가된다. 각 지문에 대해, 우리는 5개의 꿈(dream)을 샘플링하여 이를 하나의 집계된 합성 데이터셋으로 결합해 학습에 사용한다. Sleep 과정이 최고의 성능을 달성한다.

**소수샷 학습(Few-Shot Learning).** 우리는 선행 연구(Akyurek et al. 2024a; Zweiger et al. 2025)의 소수샷 ARC 실험 프로토콜을 따르며, 이를 우리의 Sleep 패러다임에 맞게 조정한다. 백본(backbone)으로는 Llama-3.2-1B를 사용한다. 일반적인 관행을 따라, 우리는 표준 구성에서 여전히 풀 수 없는 과제들을 피하기 위해 데이터의 부분집합을 필터링하여, 학습용 11개 과제와 평가용 8개의 홀드아웃(held-out) 과제를 산출한다. 자세한 내용은 B.3절을 참조하라. 이 설정에서(표 4 참조), Sleep은 80%의 성공률을 달성하여 다른 방법들보다 높다.

### 4.2 설계 선택에 대한 절제 실험(Ablations on the Design Choices)

우리는 수학적 추론(mathematical reasoning)에서 메모리 통합 과정에 대한 설계 선택들을 절제(ablate)한다. 결과는 그림 1에 보고되어 있다. 모든 구성요소가 우리 방법의 성능에 긍정적으로 기여한다. 또한, 우리는 표 3에서 자기개선(self-improvement)에 대한 설계 선택들을 추가로 절제한다.

## 5 결론(Conclusion)

본 연구에서, 우리는 대규모 언어 모델(Large Language Models)을 위한 Sleep 패러다임을 소개하였으며, 이는 다음으로 구성된다: (i) 지식 시딩(knowledge seeding), 이는 단기적인 문맥 내(in-context) 지식을 더 낮은 빈도의(lower-frequency) 장기(long-term) 파라미터로 이전하는 상향 증류(upward distillation)이며, (ii) dreaming, 이는 간섭(interference)을 제어하면서 능력을 향상시키는 자기생성(self-generated) 학습이다. 우리의 실험 결과에서, 장문맥 이해(long-context understanding), 지식 통합(knowledge incorporation), 소수샷 추론(few-shot reasoning), 그리고 지속 학습(continual learning)에 걸쳐 Sleep은 일관된 향상을 낳는다.

## 참고문헌(References)

[1] William Beecher Scoville and Brenda Milner. "Loss of recent memory after bilateral hippocampal lesions". In: Journal of neurology, neurosurgery, and psychiatry 20.1 (1957), p. 11.

[2] Jurgen Schmidhuber. Evolutionary Principles in Self-Referential Learning. 1987. url: https://people.idsia.ch/~juergen/diploma1987ocr.pdf.

[3] Long-Ji Lin. "Self-improving reactive agents based on reinforcement learning, planning and teaching". In: Machine Learning 8.3–4 (1992), pp. 293–321.

[4] Juergen Schmidhuber. "Learning to control fast-weight memories: An alternative to recurrent nets. Accepted for publication in". In: Neural Computation (1992).

[5] James L. McClelland, Bruce L. McNaughton, and Randall C. O'Reilly. "Why there are complementary learning systems in the hippocampus and neocortex: insights from the successes and failures of connectionist models of learning and memory". In: Psychological Review 102.3 (1995), pp. 419–457.

[6] Larry R Squire and Pablo Alvarez. "Retrograde amnesia and memory consolidation: a neurobiological perspective". In: Current opinion in neurobiology 5.2 (1995), pp. 169–177.

[7] Uwe Frey and Richard GM Morris. "Synaptic tagging and long-term potentiation". In: Nature 385.6616 (1997), pp. 533–536.

[8] Alvaro Pascual-Leone, Amir Amedi, Felipe Fregni, and Lotfi B Merabet. "The plastic human brain cortex". In: Annu. Rev. Neurosci. 28.1 (2005), pp. 377–401.

[9] Robert Stickgold. "Sleep-dependent memory consolidation". In: Nature 437.7063 (2005), pp. 1272–1278.

[10] David J Foster and Matthew A Wilson. "Reverse replay of behavioural sequences in hippocampal place cells during the awake state". In: Nature 440.7084 (2006), pp. 680–683.

[11] Giulio Tononi and Chiara Cirelli. "Sleep function and synaptic homeostasis". In: Sleep medicine reviews 10.1 (2006), pp. 49–62.

[12] Soren Auer, Christian Bizer, Georgi Kobilarov, Jens Lehmann, Richard Cyganiak, and Zachary Ives. "Dbpedia: A nucleus for a web of open data". In: international semantic web conference. Springer. 2007, pp. 722–735.

[13] Daoyun Ji and Matthew A Wilson. "Coordinated memory replay in the visual cortex and hippocampus during sleep". In: Nature neuroscience 10.1 (2007), pp. 100–107.

[14] Michael V Johnston. "Plasticity in the developing brain: implications for rehabilitation". In: Developmental disabilities research reviews 15.2 (2009), pp. 94–101.

[15] Adrien Peyrache, Mehdi Khamassi, Karim Benchenane, Sidney I Wiener, and Francesco P Battaglia. "Replay of rule-learning related neural patterns in the prefrontal cortex during sleep". In: Nature neuroscience 12.7 (2009), pp. 919–926.

[16] Erin J Wamsley and Robert Stickgold. "Memory, sleep and dreaming: experiencing consolidation". In: Sleep medicine clinics 6.1 (2011), p. 97.

[17] Bjorn Rasch and Jan Born. "About sleep's role in memory". In: Physiological reviews (2013).

[18] Andrea N Goldstein and Matthew P Walker. "The role of sleep in emotional brain function". In: Annual review of clinical psychology 10.1 (2014), pp. 679–708.

[19] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network". In: arXiv preprint arXiv:1503.02531 (2015).

[20] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, et al. "Human-level control through deep reinforcement learning". In: Nature 518.7540 (2015), pp. 529–533.

[21] Larry R Squire, Lisa Genzel, John T Wixted, and Richard G Morris. "Memory consolidation". In: Cold Spring Harbor perspectives in biology 7.8 (2015), a021766.

[22] Yoon Kim and Alexander M Rush. "Sequence-level knowledge distillation". In: Proceedings of the 2016 conference on empirical methods in natural language processing. 2016, pp. 1317–1327.

[23] Dharshan Kumaran, Demis Hassabis, and James L. McClelland. "What learning systems do intelligent agents need? Complementary learning systems theory updated". In: Trends in Cognitive Sciences 20.7 (2016), pp. 512–534.

[24] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. "SQuAD: 100,000+ Questions for Machine Comprehension of Text". In: Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing. Ed. by Jian Su, Kevin Duh, and Xavier Carreras. Association for Computational Linguistics, 2016. url: https://aclanthology.org/D16-1264/.

[25] Chelsea Finn, Pieter Abbeel, and Sergey Levine. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks". In: Proceedings of the 34th International Conference on Machine Learning. Ed. by Doina Precup and Yee Whye Teh. Proceedings of Machine Learning Research. PMLR, 2017. url: https://proceedings.mlr.press/v70/finn17a.html.

[26] Demis Hassabis, Dharshan Kumaran, Christopher Summerfield, and Matthew Botvinick. "Neuroscience-inspired artificial intelligence". In: Neuron 95.2 (2017), pp. 245–258.

[27] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. "Overcoming catastrophic forgetting in neural networks". In: Proceedings of the national academy of sciences 114.13 (2017), pp. 3521–3526.

[28] Wei Li, Lei Ma, Guang Yang, and Wen-Biao Gan. "REM sleep selectively prunes and maintains new synapses in development and learning". In: Nature neuroscience 20.3 (2017), pp. 427–437.

[29] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser,

(참고문헌 목록 계속 — 서지 항목은 저자명·논문 제목·게재지 정보이므로 원문 표기를 그대로 보존한다.)

...그리고 Illia Polosukhin. "Attention is All you Need". In: Advances in Neural Information Processing Systems. Ed. by I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett. Vol. 30. Curran Associates, Inc., 2017. url: https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.

[30] David Ha and Jurgen Schmidhuber. "Recurrent world models facilitate policy evolution". In: Advances in Neural Information Processing Systems (NeurIPS). Vol. 31. 2018, pp. 2451–2463.

[31] David Ha and Jurgen Schmidhuber. "World models". In: arXiv preprint arXiv:1803.10122 2.3 (2018), p. 440.

[32] Ronald Kemker, Marc McClure, Angelina Abitino, Tyler Hayes, and Christopher Kanan. "Measuring catastrophic forgetting in neural networks". In: Proceedings of the AAAI conference on artificial intelligence. Vol. 32. 1. 2018.

[33] Stefan Larson, Anish Mahendran, Joseph J Peper, Christopher Clarke, Andrew Lee, Parker Hill, Jonathan K Kummerfeld, Kevin Leach, Michael A Laurenzano, Lingjia Tang, et al. "An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction". In: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). 2019, pp. 1311–1316.

[34] Noam Shazeer. "Fast transformer decoding: One write-head is all you need". In: arXiv preprint arXiv:1911.02150 (2019).

[35] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. "Language models are few-shot learners". In: Advances in neural information processing systems 33 (2020), pp. 1877–1901.

[36] Inigo Casanueva, Tadas Temcinas, Daniela Gerz, Matthew Henderson, and Ivan Vulic. "Efficient Intent Detection with Dual Sentence Encoders". In: ACL 2020 (2020), p. 38.

[37] Oscar C. Gonzalez, Yury Sokolov, Giri P. Krishnan, Jean Erik Delanois, and Maxim Bazhenov. "Can sleep protect memories from catastrophic forgetting?" In: eLife 9 (2020), e51005.

[38] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. "Dream to control: Learning behaviors by latent imagination". In: International Conference on Learning Representations (ICLR). 2020.

[39] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Francois Fleuret. "Transformers are rnns: Fast autoregressive transformers with linear attention". In: International conference on machine learning. PMLR. 2020, pp. 5156–5165.

[40] Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A Smith, and Matt Gardner. "A dataset of information-seeking questions and answers anchored in research papers". In: arXiv preprint arXiv:2105.03011 (2021).

[41] Akihiro Goto, Ayaka Bota, Ken Miya, Jingbo Wang, Suzune Tsukamoto, Xinzhi Jiang, Daichi Hirai, Masanori Murayama, Tomoki Matsuda, Thomas J. McHugh, Takeharu Nagai, and Yasunori Hayashi. "Stepwise synaptic plasticity events drive the early phase of memory consolidation". In: Science 374.6569 (2021), pp. 857–863. doi: 10.1126/science.abj9195. eprint: https://www.science.org/doi/pdf/10.1126/science.abj9195. url: https://www.science.org/doi/abs/10.1126/science.abj9195.

[42] Eric R Kandell, Jojhn D Koester, Sarah H Mack, and Steven Siegelbaum. Principles of neural science. McGraw-Hill, 2021.

[43] Brian Lester, Rami Al-Rfou, and Noah Constant. "The power of scale for parameter-efficient prompt tuning". In: arXiv preprint arXiv:2104.08691 (2021).

[44] Xiang Lisa Li and Percy Liang. "Prefix-Tuning: Optimizing Continuous Prompts for Generation". In: Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). Ed. by Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli. Online: Association for Computational Linguistics, Aug. 2021, pp. 4582–4597. doi: 10.18653/v1/2021.acl-long.353. url: https://aclanthology.org/2021.acl-long.353/.

[45] Ekin Akyurek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. "What learning algorithm is in-context learning? investigations with linear models". In: arXiv preprint arXiv:2211.15661 (2022).

[46] Aydar Bulatov, Yury Kuratov, and Mikhail Burtsev. "Recurrent memory transformer". In: Advances in Neural Information Processing Systems 35 (2022), pp. 11079–11091.

[47] Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. "LoRA: Low-Rank Adaptation of Large Language Models". In: International Conference on Learning Representations. 2022. url: https://openreview.net/forum?id=nZeVKeeFYf9.

[48] Kazuki Irie, Imanol Schlag, Robert Csordas, and Jurgen Schmidhuber. "A modern self-referential weight matrix that learns to modify itself". In: International Conference on Machine Learning. PMLR. 2022. url: https://proceedings.mlr.press/v162/irie22b.html.

[49] Timothy Tadros, Giri P. Krishnan, Ramyaa Ramyaa, and Maxim Bazhenov. "Sleep-like unsupervised replay reduces catastrophic forgetting in artificial neural networks". In: Nature Communications 13.1 (2022), p. 7742.

[50] Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. "STaR: Bootstrapping Reasoning With Reasoning". In: Advances in Neural Information Processing Systems. Ed. by S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh. Curran Associates, Inc., 2022. url: https://proceedings.neurips.cc/paper_files/paper/2022/file/639a9a172c044fbb64175b5fad42e9a5-Paper-Conference.pdf.

[51] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. "Gpt-4 technical report". In: arXiv preprint arXiv:2303.08774 (2023).

[52] Joshua Ainslie, James Lee-Thorp, Michiel De Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. "Gqa: Training generalized multi-query transformer models from multi-head checkpoints". In: arXiv preprint arXiv:2305.13245 (2023).

[53] Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. "Adapting language models to compress contexts". In: arXiv preprint arXiv:2305.14788 (2023).

[54] Tao Ge, Jing Hu, Lei Wang, Xun Wang, Si-Qing Chen, and Furu Wei. "In-context autoencoder for context compression in a large language model". In: arXiv preprint arXiv:2307.06945 (2023).

[55] Yunhao Gou, Zhili Liu, Kai Chen, Lanqing Hong, Hang Xu, Aoxue Li, Dit-Yan Yeung, James T Kwok, and Yu Zhang. "Mixture of cluster-conditional lora experts for vision-language instruction tuning". In: arXiv preprint arXiv:2312.12379 (2023).

[56] Albert Gu and Tri Dao. "Mamba: Linear-time sequence modeling with selective state spaces". In: arXiv preprint arXiv:2312.00752 (2023).

[57] Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu Pang, Chao Du, and Min Lin. "Lorahub: Efficient cross-task generalization via dynamic lora composition". In: arXiv preprint arXiv:2307.13269 (2023).

[58] Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. "Llmlingua: Compressing prompts for accelerated inference of large language models". In: arXiv preprint arXiv:2310.05736 (2023).

[59] Yucheng Li. "Unlocking context constraints of llms: Enhancing context efficiency of llms with self-information-based content filtering". In: arXiv preprint arXiv:2304.12102 (2023).

[60] Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. "CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis". In: The Eleventh International Conference on Learning Representations. 2023. url: https://openreview.net/forum?id=iaYcJKpY2B_.

[61] Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G Patil, Ion Stoica, and Joseph E Gonzalez. "Memgpt: Towards llms as operating systems". In: arXiv preprint arXiv:2310.08560 (2023).

[62] Rylan Schaeffer, Brando Miranda, and Sanmi Koyejo. "Are emergent abilities of large language models a mirage?" In: Advances in neural information processing systems 36 (2023), pp. 55565–55581.

[63] Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Yu Qiao, et al. "Visionllm: Large language model is also an open-ended decoder for vision-centric tasks". In: Advances in Neural Information Processing Systems 36 (2023), pp. 61501–61513.

[64] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Re, Clark Barrett, et al. "H2o: Heavy-hitter oracle for efficient generative inference of large language models". In: Advances in Neural Information Processing Systems 36 (2023), pp. 34661–34710.

[65] Marah Abdin, Jyoti Aneja, Harkirat Behl, Sebastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J Hewett, Mojan Javaheripi, Piero Kauffmann, et al. "Phi-4 technical report". In: arXiv preprint arXiv:2412.08905 (2024).

[66] Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes". In: The Twelfth International Conference on Learning Representations. 2024. url: https://openreview.net/forum?id=3zKtaqxLhW.

[67] Ekin Akyurek, Mehul Damani, Adam Zweiger, Linlu Qiu, Han Guo, Jyothish Pari, Yoon Kim, and Jacob Andreas. "The Surprising Effectiveness of Test-Time Training for Few-Shot Learning". In: Forty-second International Conference on Machine Learning. 2024.

[68] Ekin Akyurek, Bailin Wang, Yoon Kim, and Jacob Andreas. "In-context language learning: Architectures and algorithms". In: arXiv preprint arXiv:2401.12973 (2024).

[69] Simran Arora, Sabri Eyuboglu, Michael Zhang, Aman Timalsina, Silas Alberti, James Zou, Atri Rudra, and Christopher Re. "Simple linear attention language models balance the recall-throughput tradeoff". In: Forty-first International Conference on Machine Learning. 2024. url: https://openreview.net/forum?id=e93ffDcpH3.

[70] Maximilian Beck, Korbinian Poppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Gunter Klambauer, Johannes Brandstetter, and Sepp Hochreiter. "xLSTM: Extended Long Short-Term Memory". In: arXiv preprint arXiv:2405.04517 (2024).

[71] Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. "Titans: Learning to memorize at test time". In: arXiv preprint arXiv:2501.00663 (2024).

[72] Jeffrey Cheng, Marc Marone, Orion Weller, Dawn Lawrie, Daniel Khashabi, and Benjamin Van Durme. "Dated Data: Tracing Knowledge Cutoffs in Large Language Models". In: First Conference on Language Modeling. 2024. url: https://openreview.net/forum?id=wS7PxDjy6m.

[73] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Baobao Chang, Xu Sun, Lei Li, and Zhifang Sui. "A Survey on In-context Learning". In: Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. Ed. by Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen. Miami, Florida, USA: Association for Computational Linguistics, Nov. 2024, pp. 1107–1128. doi: 10.18653/v1/2024.emnlp-main.64. url: https://aclanthology.org/2024.emnlp-main.64/.

[74] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. "The llama 3 herd of models". In: arXiv e-prints (2024), arXiv–2407.

[75] Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, and Boris Ginsburg. "RULER: What's the Real Context Size of Your Long-Context Language Models?" In: First Conference on Language Modeling. 2024. url: https://openreview.net/forum?id=kIoBbc76Sy.

[76] Adam Ibrahim, Benjamin Therien, Kshitij Gupta, Mats Leon Richter, Quentin Gregory Anthony, Eugene Belilovsky, Timothee Lesort, and Irina Rish. "Simple and Scalable Strategies to Continually Pre-train Large Language Models". In: Transactions on Machine Learning Research (2024). issn: 2835-8856. url: https://openreview.net/forum?id=DimPeeCxKO.

[77] Kalle Kujanpaa, Harri Valpola, and Alexander Ilin. "Knowledge injection via prompt distillation". In: arXiv preprint arXiv:2412.14964 (2024).

[78] Yury Kuratov, Aydar Bulatov, Petr Anokhin, Ivan Rodkin, Dmitry Sorokin, Artyom Sorokin, and Mikhail Burtsev. "Babilong: Testing the limits of llms with long context reasoning-in-a-haystack". In: Advances in Neural Information Processing Systems 37 (2024), pp. 106519–106554.

[79] Dengchun Li, Yingzi Ma, Naizheng Wang, Zhengmao Ye, Zhiyuan Cheng, Yinghao Tang, Yan Zhang, Lei Duan, Jie Zuo, Cal Yang, et al. "Mixlora: Enhancing large language models fine-tuning with lora-based mixture of experts". In: arXiv preprint arXiv:2404.15159 (2024).

[80] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. "Snapkv: Llm knows what you are looking for before generation". In: Advances in Neural Information Processing Systems 37 (2024), pp. 22947–22970.

[81] Akide Liu, Jing Liu, Zizheng Pan, Yefei He, Gholamreza Haffari, and Bohan Zhuang. "MiniCache: KV Cache Compression in Depth Dimension for Large Language Models". In: Advances in Neural Information Processing Systems 37 (2024).

[82] Fanxu Meng, Zhaohui Wang, and Muhan Zhang. "Pissa: Principal singular values and singular vectors adaptation of large language models". In: Advances in Neural Information Processing Systems 37 (2024), pp. 121038–121072.

[83] Nihal V Nayak, Yiyang Nan, Avi Trost, and Stephen H Bach. "Learning to generate instruction tuning datasets for zero-shot task adaptation". In: arXiv preprint arXiv:2402.18334 (2024).

[84] Xingyuan Pan, Luyang Huang, Liyan Kang, Zhicheng Liu, Yu Lu, and Shanbo Cheng. "G-DIG: Towards Gradient-based DIverse and hiGh-quality Instruction Data Selection for Machine Translation". In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Ed. by Lun-Wei Ku, Andre Martins, and Vivek Srikumar. Bangkok, Thailand: Association for Computational Linguistics, Aug. 2024, pp. 15395–15406. doi: 10.18653/v1/2024.acl-long.821. url: https://aclanthology.org/2024.acl-long.821/.

[85] Jing-Cheng Pang, Pengyuan Wang, Kaiyuan Li, Xiong-Hui Chen, Jiacheng Xu, Zongzhang Zhang, and Yang Yu. "Language Model Self-improvement by Reinforcement Learning Contemplation". In: The Twelfth International Conference on Learning Representations. 2024. url: https://openreview.net/forum?id=38E4yUbrgr.

[86] Jing-Cheng Pang, Pengyuan Wang, Kaiyuan Li, Xiong-Hui Chen, Jiacheng Xu, Zongzhang Zhang, and Yang Yu. "Language Model Self-improvement by Reinforcement Learning Contemplation". In: The Twelfth International Conference on Learning Representations. 2024. url: https://openreview.net/forum?id=38E4yUbrgr.

[87] Ivan Rodkin, Yuri Kuratov, Aydar Bulatov, and Mikhail Burtsev. "Associative recurrent memory transformer". In: arXiv preprint arXiv:2407.04841 (2024).

[88] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. "Deepseekmath: Pushing the limits of mathematical reasoning in open language models". In: arXiv preprint arXiv:2402.03300 (2024).

[89] Haizhou Shi, Zihao Xu, Hengyi Wang, Weiyi Qin, Wenyuan Wang, Yibin Wang, Zifeng Wang, Sayna Ebrahimi, and Hao Wang. "Continual learning of large language models: A comprehensive survey". In: ACM Computing Surveys (2024).

[90] Avi Singh, John D Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Xavier Garcia, Peter J Liu, James Harrison, Jaehoon Lee, Kelvin Xu, Aaron T Parisi, Abhishek Kumar, Alexander A Alemi, Alex Rizkowsky, Azade Nova, Ben Adlam, Bernd Bohnet, Gamaleldin Fathy Elsayed, Hanie Sedghi, Igor Mordatch, Isabelle Simpson, Izzeddin Gur, Jasper Snoek, Jeffrey Pennington, Jiri Hron, Kathleen Kenealy, Kevin Swersky, Kshiteej Mahajan, Laura A Culp, Lechao Xiao, Maxwell Bileschi, Noah Constant, Roman Novak, Rosanne Liu, Tris Warkentin, Yamini Bansal, Ethan Dyer, Behnam Neyshabur, Jascha Sohl-Dickstein, and Noah Fiedel. "Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models". In: Transactions on Machine Learning Research (2024). Expert Certification. issn: 2835-8856. url: https://openreview.net/forum?id=lNAyUngGFK.

[91] Avi Singh, John D Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Xavier Garcia, Peter J Liu, James Harrison, Jaehoon Lee, Kelvin Xu, Aaron T Parisi, Abhishek Kumar, Alexander A Alemi, Alex Rizkowsky, Azade Nova, Ben Adlam, Bernd Bohnet, Gamaleldin Fathy Elsayed, Hanie Sedghi, Igor Mordatch, Isabelle Simpson, Izzeddin Gur, Jasper Snoek, Jeffrey Pennington, Jiri Hron, Kathleen Kenealy, Kevin Swersky, Kshiteej Mahajan, Laura A Culp, Lechao Xiao, Maxwell Bileschi, Noah Constant, Roman Novak, Rosanne Liu, Tris Warkentin, Yamini Bansal, Ethan Dyer, Behnam Neyshabur, Jascha Sohl-Dickstein, and Noah Fiedel. "Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models". In: Transactions on Machine Learning Research (2024). url: https://openreview.net/forum?id=lNAyUngGFK.

[92] Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, et al. "Learning to (learn at test time): Rnns with expressive hidden states". In: arXiv preprint arXiv:2407.04620 (2024).

[93] Sijun Tan, Xiuyu Li, Shishir Patil, Ziyang Wu, Tianjun Zhang, Kurt Keutzer, Joseph E Gonzalez, and Raluca Ada Popa. "Lloco: Learning long contexts offline". In: arXiv preprint arXiv:2404.07979 (2024).

[94] Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song Han. "Quest: Query-aware sparsity for efficient long-context llm inference". In: arXiv preprint arXiv:2406.10774 (2024).

[95] Garrett Tanzer, Mirac Suzgun, Eline Visser, Dan Jurafsky, and Luke Melas-Kyriazi. "A Benchmark for Learning to Translate a New Language from One Grammar Book". In: The Twelfth International Conference on Learning (계속됨)

Representations. 2024. url: https://openreview.net/forum?id=tbVWug9f2h.

[96] Zhongwei Wan, Xinjian Wu, Yu Zhang, Yi Xin, Chaofan Tao, Zhihong Zhu, Xin Wang, Siqi Luo, Jing Xiong, and Mi Zhang. "D2o: Dynamic discriminative operations for efficient generative inference of large language models". In: arXiv preprint arXiv:2406.13035 (2024).

[97] Jiachen Tianhao Wang, Tong Wu, Dawn Song, Prateek Mittal, and Ruoxi Jia. "Greats: Online selection of high-quality data for llm training in every iteration". In: Advances in Neural Information Processing Systems 37 (2024), pp. 131197–131223.

[98] Zheng Wang, Boxiao Jin, Zhongzhi Yu, and Minjia Zhang. "Model tells you where to merge: Adaptive kv cache merging for llms on long-context tasks". In: arXiv preprint arXiv:2407.08454 (2024).

[99] Xun Wu, Shaohan Huang, and Furu Wei. "Mixture of lora experts". In: arXiv preprint arXiv:2404.13628 (2024).

[100] Chaojun Xiao, Zhengyan Zhang, Chenyang Song, Dazhi Jiang, Feng Yao, Xu Han, Xiaozhi Wang, Shuo Wang, Yufei Huang, Guanyu Lin, et al. "Configurable foundation models: Building llms from a modular perspective". In: arXiv preprint arXiv:2409.02877 (2024).

[101] Prateek Yadav, Colin Raffel, Mohammed Muqeeth, Lucas Caccia, Haokun Liu, Tianlong Chen, Mohit Bansal, Leshem Choshen, and Alessandro Sordoni. "A survey on model moerging: Recycling and routing among specialized experts for collaborative learning". In: arXiv preprint arXiv:2408.07057 (2024).

[102] Wannan Yang, Chen Sun, Roman Huszar, Thomas Hainmueller, Kirill Kiselev, and Gyorgy Buzsaki. "Selection of experience for memory by hippocampal sharp wave ripples". In: Science 383.6690 (2024), pp. 1478–1483.

[103] Rongzhi Zhang, Kuang Wang, Liyuan Liu, Shuohang Wang, Hao Cheng, Chao Zhang, and Yelong Shen. "LoRC: Low-Rank Compression for LLMs KV Cache with a Progressive Compression Strategy". In: arXiv preprint arXiv:2410.03111 (2024).

[104] Yuxin Zhang, Yuxuan Du, Gen Luo, Yunshan Zhong, Zhenyu Zhang, Shiwei Liu, and Rongrong Ji. "Cam: Cache merging for memory-efficient llms inference". In: Forty-first International Conference on Machine Learning. 2024.

[105] Ziyu Zhao, Leilei Gan, Guoyin Wang, Wangchunshu Zhou, Hongxia Yang, Kun Kuang, and Fei Wu. "Loraretriever: Input-aware lora retrieval and composition for mixed tasks in the wild". In: arXiv preprint arXiv:2402.09997 (2024).

[106] Ziyu Zhao, Tao Shen, Didi Zhu, Zexi Li, Jing Su, Xuwu Wang, Kun Kuang, and Fei Wu. "Merging loras like playing lego: Pushing the modularity of lora to extremes through rank-wise clustering". In: arXiv preprint arXiv:2409.16167 (2024).

[107] Lisa Adams, Felix Busch, Tianyu Han, Jean-Baptiste Excoffier, Matthieu Ortala, Alexander Loser, Hugo JWL Aerts, Jakob Nikolas Kather, Daniel Truhn, and Keno Bressem. "Longhealth: A question answering benchmark with long clinical documents". In: Journal of Healthcare Informatics Research (2025), pp. 1–17.

[108] Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, and Vahab Mirrokni. "Nested Learning: The Illusion of Deep Learning Architectures". In: The Thirty-ninth Annual Conference on Neural Information Processing Systems. 2025. url: https://openreview.net/forum?id=nbMeRvNb7A.

[109] Lucas Caccia, Alan Ansell, Edoardo Ponti, Ivan Vulic, and Alessandro Sordoni. "Training Plug-n-Play Knowledge Modules with Deep Context Distillation". In: arXiv preprint arXiv:2503.08727 (2025).

[110] Vivek Chari, Guanghui Qin, and Benjamin Van Durme. "KV-Distill: Nearly Lossless Learnable Context Compression for LLMs". In: arXiv preprint arXiv:2503.10337 (2025).

[111] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. "Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities". In: arXiv preprint arXiv:2507.06261 (2025).

[112] DeepSeek-AI. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. 2025. arXiv: 2501.12948 [cs.CL]. url: https://arxiv.org/abs/2501.12948.

[113] Benoit Dherin, Michael Munn, Hanna Mazzawi, Michael Wunder, and Javier Gonzalvo. "Learning without training: The implicit dynamics of in-context learning". In: arXiv preprint arXiv:2507.16003 (2025).

[114] Sabri Eyuboglu, Ryan Ehrlich, Simran Arora, Neel Guha, Dylan Zinsley, Emily Liu, Will Tennien, Atri Rudra, James Zou, Azalia Mirhoseini, et al. "Cartridges: Lightweight and general-purpose long context representations via self-study". In: arXiv preprint arXiv:2506.06266 (2025).

[115] Audrey Huang, Adam Block, Dylan J Foster, Dhruv Rohatgi, Cyril Zhang, Max Simchowitz, Jordan T. Ash, and Akshay Krishnamurthy. "Self-Improvement in Language Models: The Sharpening Mechanism". In: The Thirteenth International Conference on Learning Representations. 2025. url: https://openreview.net/forum?id=WJaUkwci9o.

[116] Tianle Li, Ge Zhang, Quy Duc Do, Xiang Yue, and Wenhu Chen. "Long-context LLMs Struggle with Long In-context Learning". In: Transactions on Machine Learning Research (2025). issn: 2835-8856. url: https://openreview.net/forum?id=Cw2xlg0e46.

[117] Kevin Lin, Charlie Snell, Yu Wang, Charles Packer, Sarah Wooders, Ion Stoica, and Joseph E Gonzalez. "Sleep-time compute: Beyond inference scaling at test-time". In: arXiv preprint arXiv:2504.13171 (2025).

[118] Yansheng Mao, Yufei Xu, Jiaqi Li, Fanxu Meng, Haotong Yang, Zilong Zheng, Xiyuan Wang, and Muhan Zhang. "LIFT: Improving Long Context Understanding of Large Language Models through Long Input Fine-Tuning". In: arXiv preprint arXiv:2502.14644 (2025).

[119] Saleh Momeni, Sahisnu Mazumder, Zixuan Ke, and Bing Liu. "In-context continual learning assisted by an external continual learner". In: Proceedings of the 31st International Conference on Computational Linguistics. 2025, pp. 7292–7306.

[120] Renhao Pei, Yihong Liu, Peiqin Lin, Francois Yvon, and Hinrich Schutze. "Understanding In-Context Machine Translation for Low-Resource Languages: A Case Study on Manchu". In: arXiv preprint arXiv:2502.11862 (2025).

[121] Haris Riaz, Sourav Bhabesh, Vinayak Arannil, Miguel Ballesteros, and Graham Horwood. "MetaSynth: Meta-Prompting-Driven Agentic Scaffolds for Diverse Synthetic Data Generation". In: arXiv preprint arXiv:2504.12563 (2025).

[122] Weihang Su, Yichen Tang, Qingyao Ai, Junxi Yan, Changyue Wang, Hongning Wang, Ziyi Ye, Yujia Zhou, and Yiqun Liu. "Parametric retrieval augmented generation". In: arXiv preprint arXiv:2501.15915 (2025).

[123] Zhaoyang Wang, Weilei He, Zhiyuan Liang, Xuchao Zhang, Chetan Bansal, Ying Wei, Weitong Zhang, and Huaxiu Yao. "CREAM: Consistency Regularized Self-Rewarding Language Models". In: The Thirteenth International Conference on Learning Representations. 2025. url: https://openreview.net/forum?id=Vf6RDObyEF.

[124] Guangxuan Xiao, Jiaming Tang, Jingwei Zuo, junxian guo, Shang Yang, Haotian Tang, Yao Fu, and Song Han. "DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads". In: The Thirteenth International Conference on Learning Representations. 2025. url: https://openreview.net/forum?id=cFu7ze7xUm.

[125] Adam Zweiger, Jyothish Pari, Han Guo, Ekin Akyurek, Yoon Kim, and Pulkit Agrawal. "Self-Adapting Language Models". In: arXiv preprint arXiv:2506.10943 (2025).

[126] Yinghui He, Simran Kaur, Adithya Bhaskar, Yongjin Yang, Jiarui Liu, Narutatsu Ri, Liam Fowl, Abhishek Panigrahi, Danqi Chen, and Sanjeev Arora. Self-Distillation Zero: Self-Revision Turns Binary Rewards into Dense Supervision. 2026. arXiv: 2604.12002 [cs.CL]. url: https://arxiv.org/abs/2604.12002.

[127] Jonas Hubotter, Frederike Lubeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, et al. "Reinforcement Learning via Self-Distillation". In: arXiv preprint arXiv:2601.20802 (2026).

[128] Jeonghye Kim, Xufang Luo, Minbeom Kim, Sangmook Lee, Dohyung Kim, Jiwon Jeon, Dongsheng Li, and Yuqing Yang. Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs? 2026. arXiv: 2603.24472 [cs.CL]. url: https://arxiv.org/abs/2603.24472.

[129] John Kirchenbauer, Abhimanyu Hans, Brian Bartoldson, Micah Goldblum, Ashwinee Panda, and Tom Goldstein. Multi-Token Prediction via Self-Distillation. 2026. arXiv: 2602.06019 [cs.CL]. url: https://arxiv.org/abs/2602.06019.

[130] Yihong Liu, Raoyuan Zhao, Michael A. Hedderich, and Hinrich Schutze. Crosslingual On-Policy Self-Distillation for Multilingual Reasoning. 2026. arXiv: 2605.09548 [cs.CL]. url: https://arxiv.org/abs/2605.09548.

[131] Ruiyang Qin, Qingzhuo Wang, Dongrui Liu, Qiang Li, Zhihua Wei, and Wen Shen. Multilingual Safety Alignment via Self-Distillation. 2026. arXiv: 2605.02971 [cs.LG]. url: https://arxiv.org/abs/2605.02971.

[132] Hejian Sang, Yuanda Xu, Zhengze Zhou, Ran He, Zhipeng Wang, and Jiachen Sun. CRISP: Compressed Reasoning via Iterative Self-Policy Distillation. 2026. arXiv: 2603.05433 [cs.LG]. url: https://arxiv.org/abs/2603.05433.

[133] Idan Shenfeld, Mehul Damani, Jonas Hubotter, and Pulkit Agrawal. "Self-Distillation Enables Continual Learning". In: arXiv preprint arXiv:2601.19897 (2026).

[134] Idan Shenfeld, Mehul Damani, Jonas Hubotter, and Pulkit Agrawal. Self-Distillation Enables Continual Learning. 2026. arXiv: 2601.19897 [cs.LG]. url: https://arxiv.org/abs/2601.19897.

[135] Alex Stein, Furong Huang, and Tom Goldstein. GATES: Self-Distillation under Privileged Context with Consensus Gating. 2026. arXiv: 2602.20574 [cs.LG]. url: https://arxiv.org/abs/2602.20574.

[136] Hao Wang, Guozhi Wang, Han Xiao, Yufeng Zhou, Yue Pan, Jichao Wang, Ke Xu, Yafei Wen, Xiaohu Ruan, Xiaoxin Chen, and Honggang Qi. Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents. 2026. arXiv: 2604.10674 [cs.LG]. url: https://arxiv.org/abs/2604.10674.

[137] Tianzhu Ye, Li Dong, Qingxiu Dong, Xun Wu, Shaohan Huang, and Furu Wei. Online Experiential Learning for Language Models. 2026. arXiv: 2603.16856 [cs.CL]. url: https://arxiv.org/abs/2603.16856.

[138] Tianzhu Ye, Li Dong, Xun Wu, Shaohan Huang, and Furu Wei. On-Policy Context Distillation for Language Models. 2026. arXiv: 2602.12275 [cs.CL]. url: https://arxiv.org/abs/2602.12275.

[139] Ruixiang Zhang, Richard He Bai, Huangjie Zheng, Navdeep Jaitly, Ronan Collobert, and Yizhe Zhang. Embarrassingly Simple Self-Distillation Improves Code Generation. 2026. arXiv: 2604.01193 [cs.CL]. url: https://arxiv.org/abs/2604.01193.

[140] Xinsen Zhang, Zhenkai Ding, Tianjun Pan, Run Yang, Chun Kang, Xue Xiong, and Jingnan Gu. "OPSDL: On-Policy Self-Distillation for Long-Context Language Models". In: arXiv preprint arXiv:2604.17535 (2026).

[141] Xinsen Zhang, Zhenkai Ding, Tianjun Pan, Run Yang, Chun Kang, Xue Xiong, and Jingnan Gu. OPSDL: On-Policy Self-Distillation for Long-Context Language Models. 2026. arXiv: 2604.17535 [cs.CL]. url: https://arxiv.org/abs/2604.17535.

[142] Yaocheng Zhang, Yuanheng Zhu, Wenyue Chong, Songjun Tu, Qichao Zhang, Jiajun Chai, Xiaohan Wang, Wei Lin, Guojun Yin, and Dongbin Zhao. π-Play: Multi-Agent Self-Play via Privileged Self-Distillation without External Data. 2026. arXiv: 2604.14054 [cs.LG]. url: https://arxiv.org/abs/2604.14054.

[143] Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. "Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models". In: arXiv preprint arXiv:2601.18734 (2026).

[144] Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models. 2026. arXiv: 2601.18734 [cs.LG]. url: https://arxiv.org/abs/2601.18734.

**그림 7(Figure 7): 다중 주파수 메모리 계층(Multi-frequency memory hierarchy).** 업데이트는 반복적인 파라미터 확장(Parameter Expansion)을 통해 고주파(High-Frequency) FFN으로 진입한다. 윈도우 $f_W$가 만료되면, 지식은 중간(Mid-) 주파수 FFN으로, 그다음 저주파(Low-Frequency) FFN으로 통합(Consolidated)된다(1k→5k→10k).

**그림 8(Figure 8): 라우팅된 전문가 업데이트(routed expert updates)에 의한 메모리 통합.** Sleep 사이클을 거치면서(왼쪽→오른쪽), 라우터(router)는 소수의 전문가(expert) 집합을 선택하여 업데이트하고(실선), 나머지는 비활성 상태로 둔다(빗금). 이를 통해 간섭을 제한하면서 용량을 확장한다.

## A. 관련 연구 (Related Work)

### A.1 파라미터 효율적 적응 및 합성 (Parameter-Efficient Adaptation and Composition)

파라미터 효율적 미세조정(parameter-efficient fine-tuning, PEFT)은 백본(backbone)을 동결한 채 최소한의 보조 파라미터 집합만 최적화하여 대규모 언어 모델(large language models, LLMs)을 특정 태스크에 적응시킨다. **저랭크 및 프리픽스 적응(Low-Rank and Prefix Adaptation).** 대표적인 방법으로는 선형 투영(linear projection)에 학습 가능한 저랭크 행렬을 주입하는 LoRA(Hu et al. 2022)와, 학습 가능한 가상 토큰(virtual tokens)을 통해 연산을 유도하는 프리픽스/프롬프트 튜닝(prefix/prompt-tuning)(Lester et al. 2021; Li et al. 2021)이 있다. 최근의 변형들은 수렴을 가속하기 위해 초기화를 최적화하거나(예: PiSSA(Meng et al. 2024)), 이러한 메커니즘을 문맥-대-파라미터 증류(context-to-parameter distillation)로 확장한다(Eyuboglu et al. 2025).

**합성 및 라우팅(Composition and Routing).** 단일 태스크 적응을 넘어, 연구는 점차 다수의 어댑터(adapter)를 합성하는 데 초점을 맞추고 있다. 이러한 접근에는 in-context learning을 통한 가중 합성(Huang et al. 2023)과, 입력마다 관련 LoRA 모듈을 동적으로 선택하는 검색 기반 라우팅(retrieval-based routing)(Zhao et al. 2024a)이 포함된다. 진보된 구성들은 전문가 혼합(mixture-of-experts, MoE) 아키텍처를 활용하여 태스크에 따라 특정 저랭크 경로를 활성화하거나 어댑터 파라미터를 병합한다(Gou et al. 2023; Li et al. 2024a; Wu et al. 2024; Xiao et al. 2024; Yadav et al. 2024; Zhao et al. 2024b).

### A.2 지식 주입, 증류, 자기 개선 (Knowledge Injection, Distillation, and Self-Improvement)

**파라미터적 지식 주입(Parametric Knowledge Injection).** 추론 시점(inference time)의 검색 의존도를 줄이기 위해, 최근 연구는 외부 지식을 모델 파라미터에 직접 주입한다. 그 기법은 문서별 LoRA 어댑터를 할당하는 Parametric RAG(Su et al. 2025)에서부터, 학생(student) 모델이 교사(teacher)가 생성한 QA 쌍이나 합성 대화(synthetic conversations)로부터 학습하는 프롬프트 증류(prompt distillation)(Kujanpaa et al. 2024; Caccia et al. 2025; Mao et al. 2025)에 이른다. Cartridges(Eyuboglu et al. 2025)는 자기 학습(self-study) 목표를 통해 재사용 가능한 "KV 캐시 어댑터(KV cache adapter)"를 사전 학습함으로써 PEFT와 증류를 잇고, 서빙 비용을 줄이면서도 in-context learning(ICL) 품질을 달성한다.

**자기 개선 및 메타 학습(Self-Improvement and Meta-Learning).** 증류를 보완하는 것으로 자기 개선을 위한 모델 생성 신호(model-generated signals)의 활용이 있다. 여기에는 추론 능력을 향상시키기 위한 검증 가능한 보상(verifiable rewards)에 대한 강화학습(Reinforcement Learning, RL)(Zelikman et al. 2022; Singh et al. 2024b; DeepSeek-AI 2025)과 자기 보상(self-rewarding) 메커니즘(Pang et al. 2024b; Wang et al. 2025)이 포함된다. SEAL(Zweiger et al. 2025)과 같은 메타 학습 프레임워크는 적응 전략 자체를 최적화—즉 자기 편집(self-edits)을 어떻게 생성할지를 학습—함으로써 이를 확장하며, 이는 자기 참조 학습(self-referential learning)의 보다 넓은 원리(Schmidhuber 1987; Finn et al. 2017; Irie et al. 2022)에 기반한다. 이러한 방법들은 감독(supervision)을 확장하기 위해 고품질 합성 데이터 생성에 크게 의존한다(Abdin et al. 2024; Nayak et al. 2024; Riaz et al. 2025).

최근 동시대의 일군의 연구들은 자기 증류(self-distillation) 과정을 위해(Hubotter et al. 2026; Zhang et al. 2026b; Zhao et al. 2026a) 또는 지속 학습(continual learning)을 위해(Shenfeld et al. 2026a) 온-폴리시 증류(on-policy distillation)를 사용할 것을 제안해 왔다. 본 연구가 그러한 방법들보다 더 오래되었다는 사실에 더하여, Sleep은 근본적으로 다른 메시지들을 전달한다. 즉: (1) Sleep은 파라미터를 잠금 해제(unlock)하는, 자기 자신에 대한 상향식 증류(upward distillation of self)를 사용한다. (2) Sleep은 온-폴리시 방법과 RL 방법을 결합하는 일반화된 증류(Generalized Distillation) 방법에 기반한다. (3) Sleep은 지식이 서로 다른 업데이트 주파수를 갖는 모듈들에 저장되는 주기적 과정(periodic process)을 제안한다.

### A.3 효율적 문맥 처리 (Efficient Context Processing)

긴 문맥(long-context) 처리의 메모리 병목을 해결하는 것은 KV 캐시를 압축하거나 어텐션 아키텍처를 수정하는 것을 포함한다.

**프롬프트 및 캐시 압축(Prompt and Cache Compression).** 메모리 사용량을 줄이는 접근은 두 범주로 나뉜다: 프롬프트 압축(Prompt compression)은 토큰 필터링(하드 토큰, hard-token)을 통해 입력을 단축하거나(Jiang et al. 2023; Li 2023), 또는 학습된 압축적 소프트 토큰(soft-token) 임베딩을 통해

오토인코더(Chevalier et al. 2023; Ge et al. 2023; Tan et al. 2024). KV 캐시 압축은 런타임에 작동하며, 비필수 키를 버리는 축출(eviction) 정책(Zhang et al. 2023; Li et al. 2024b; Tang et al. 2024)이나 유사성과 어텐션 밀도에 기반해 토큰을 병합하는 방식(Liu et al. 2024; Wan et al. 2024; Wang et al. 2024b; Zhang et al. 2024b)을 활용한다. KV 캐시의 저랭크(low-rank) 투영 또한 높은 압축률에서 성능을 유지하는 데 유망함을 보였다(Zhang et al. 2024a; Chari et al. 2025).

**아키텍처 혁신(Architectural Innovations).** 다중 헤드 어텐션(Multi-Head Attention, MHA)에 대한 구조적 변경으로는, KV 헤드를 공유하여 메모리 대역폭을 줄이는 다중 질의 및 그룹 질의 어텐션(Multi-Query and Grouped-Query Attention, MQA/GQA)(Shazeer 2019; Ainslie et al. 2023)과, 소프트맥스를 제거하여 시퀀스 길이와 무관한 고정 크기 상태(K⊤V)를 달성하는 선형화(linearization) 기법(Gu et al. 2023; Arora et al. 2024; Beck et al. 2024)이 있다. Titans와 TTT 같은 최근의 "압축을 학습하는(learning to compress)" 아키텍처는 무한한 컨텍스트를 처리하기 위해 고정 크기 메모리 객체에 대한 경사 기반(gradient-based) 갱신을 활용한다(Behrouz et al. 2024; Sun et al. 2024). 마지막으로, MemGPT와 같은 오케스트레이션 시스템은 아키텍처를 변경하는 대신 가상 메모리 페이징(virtual memory paging)을 통해 컨텍스트를 관리한다(Packer et al. 2023). 토큰 공간에서의 압축과 유사한 방향으로, Lin et al. (2025)은 수면 시간 연산(sleep-time compute)을 제시한다. 이름은 유사하지만, 그들의 방법은 우리의 연구와 근본적으로 다르다. 이 방법은 모델이 유휴 상태일 때 텍스트 공간에서 모델과 사용자의 상호작용에 대한 좋은 요약을 찾는 것을 목표로 하는 반면, 우리의 제안은 증류(distillation)를 사용하여 텍스트 데이터 지식을 파라메트릭 가중치 갱신(parametric weight update)의 형태로 전이하는 데 있다.

### A.4

**온-폴리시 증류(On-Policy Distillation)에 관한 최근의 동시적 및/또는 후속 연구**

**온-폴리시 자기 증류(On-Policy Self-Distillation)에 관한 동시적 및 후속 연구.** 본 연구의 초기 버전과 동시에 그리고 후속으로, 온-폴리시 자기 증류(on-policy self-distillation, OPSD)를 탐구하는 문헌이 급속히 성장해 왔다. OPSD에서는 동일한 모델이 서로 다른 조건화 컨텍스트(conditioning context) 하에서 교사(teacher)와 학생(student)의 역할을 모두 수행하며, 학생은 교사에 대한 토큰별 발산(per-token divergence)을 통해 자신의 롤아웃(rollout)에 대해 감독(supervise)된다. 정준적(canonical) 구현은 OPSD(Zhao et al. 2026b)로, 이는 검증된 추론 궤적(verified reasoning trace)에 교사를 조건화하고 학생 자신의 롤아웃에 대해 토큰별 역-KL(reverse-KL)을 최소화하여, GRPO와 오프-폴리시(off-policy) 증류 양쪽 대비 수학적 추론에서 이득을 얻는다. 일련의 후속 연구들은 특권 컨텍스트(privileged context)의 형태를 다양하게 변주한다. 온-폴리시 컨텍스트 증류(On-Policy Context Distillation)(Ye et al. 2026b)는 학생이 그 정보를 자신의 파라미터로 내재화하도록 일시적인 인-컨텍스트(in-context) 정보(과거 해법 궤적이나 최적화된 시스템 프롬프트)에 교사를 조건화한다. GATES(Stein et al. 2026)는 문서 기반(document-grounded) 특권 컨텍스트를, 여러 튜터 궤적을 샘플링하고 그 합의(agreement)에 따라 학습을 게이팅함으로써 신뢰할 수 없는 감독을 처리하는 합의-게이팅(consensus-gating) 메커니즘과 함께 사용한다. SD-Zero(He et al. 2026)는 "수정자(reviser)"를 학생의 응답과 이진 보상(binary reward)에 조건화한 뒤, 그 수정자를

23

생성기(generator)로 다시 증류하여, 희소한 결과 보상(sparse outcome reward)을 조밀한 토큰 수준 감독(dense token-level supervision)으로 변환한다. 그리고 COPSD(Liu et al. 2026)는 교사에게 영어 번역과 참조 해법을 특권적 교차언어(crosslingual) 컨텍스트로 제공함으로써 추론 행동을 저자원(low-resource) 언어로 전이한다. Apple의 "당혹스러울 정도로 단순한(embarrassingly simple)" 자기 증류(Zhang et al. 2026a)는 이 스펙트럼의 극한에 위치한다. 이는 명시적 교사를 아예 제거하고 서로 다른 디코딩 구성(decoding configuration) 하에서 모델 자신의 샘플에 대해 지도 미세조정(supervised-finetune)하는데, 이 최소한의 레시피만으로도 코드 생성이 유의미하게 개선됨을 보인다.

**지속적, 경험적, 온라인 적응(Continual, Experiential, and Online Adaptation).** 두 번째 부류의 연구들은 자기 증류를 지속적(continual), 경험적(experiential), 또는 온라인 개선에 적용한다. SDFT(Shenfeld et al. 2026b)는 시연 조건화된(demonstration-conditioned) 모델을 온-폴리시 교사로 사용하여 시연으로부터의 지속 학습을 명시적으로 겨냥하며, 표준 SFT 대비 파국적 망각(catastrophic forgetting)을 줄인다. OEL(Ye et al. 2026a)은 컨텍스트 증류를 배포(deployment)와 결합한다. 즉, 상호작용 궤적(interaction trajectory)에서 전이 가능한 경험적 지식을 추출하고 온-폴리시 컨텍스트 증류를 통해 이를 파라미터로 통합하며, 두 단계를 반복하여 닫힌 온라인 학습 루프(closed online-learning loop)를 형성한다. π-Play(Zhang et al. 2026d)는 이 아이디어를 다중 에이전트 자기 대전(multi-agent self-play) 체제로 더 밀어붙여, 과제 제안자(task proposer)가 생성한 질문 구성 경로(question-construction path)를 교사를 위한 특권 컨텍스트로 사용하고, 심층 탐색 에이전트(deep search agent) 훈련에서 외부 데이터나 인간 피드백의 필요성을 제거한다. 이러한 연구들은 LLM이 배포 후 정적으로 머물러서는 안 된다는 우리의 동기와 밀접하게 부합한다. 그러나 이들은 지속적 또는 경험적 개선을 고정 용량 체크포인트(fixed-capacity checkpoint)에 대한 후속 훈련(post-training) 목표로 정식화한다. 이와 대조적으로, 우리의 Sleep 패러다임은 지속 학습을 메모리 시스템 문제로 취급한다. 즉, 새로 획득된 정보는 먼저 빠르고 취약한(fast, fragile) 메모리 모듈에 표현된 뒤, 지식 시딩(Knowledge Seeding)을 통해 주기적으로 더 느리고 안정적인 파라미터로 통합되며, 이는 점진적 용량 확장(gradual capacity expansion)과 통합 후 고빈도(higher-frequency) 메모리의 재설정(resetting)을 동반한다.

**응용 특화 OPSD 레시피(Application-Specific OPSD Recipes).** 세 번째 부류의 논문들은 OPSD 템플릿을 특정 배포 체제에 맞게 조정한다. CRISP/OPSDC(Sang et al. 2026)는 교사를 "간결하게(be concise)" 지시 접두사(instruction prefix)에 조건화하고 학생 롤아웃에 대한 토큰별 역-KL을 사용하여, 엔트로피 붕괴(entropy collapse) 없이 긴 사고 연쇄(chain-of-thought) 궤적을 압축한다. MTP(Kirchenbauer et al. 2026)는 단일 온라인 자기 증류 목표로 사전 훈련된 다음-토큰-예측(next-token-prediction) 모델을 다중 토큰 예측기(multi-token predictor)로 변환하여, GSM8K에서 5% 미만의 정확도 감소로 3배 이상의 추론 가속을 달성한다. OPSDL(Zhang et al. 2026c)은 추출된 관련 구간(extracted relevant span)에 대한 모델 자신의 단문맥(short-context) 행동을 그 장문맥(long-context) 학생을 위한 자기 교사(self-teacher)로 사용함으로써 장문맥 생성을 공략한다. Skill-SD(Wang et al. 2026)는 다중 턴(multi-turn) 에이전트를 겨냥한다. 완료된 궤적은 간결한 자연어 "스킬(skill)"로 요약되어 교사만을 조건화하고, 학생은 평범한 과제 프롬프트 하에서 훈련된다. MSD(Qin et al. 2026)는 다중언어 질의만을 사용하여 고자원(high-resource) 언어에서 저자원 언어로 안전(safety) 행동을 전이함으로써 OPSD를 다중언어(multilingual) 설정으로 옮긴다. 종합적으로, 이러한 연구들은 특권적 조건화(privileged conditioning) 하의 자기 증류가 추론, 효율성, 에이전트, 다중언어 축에 걸쳐 일반적이고 유연한 후속 훈련 레시피임을 보여준다 — 그러나 각 경우 그것은 고정된 모델과 고정된 응용에 대한 단일 목표 절차(single-objective procedure)로 배포된다.

**OPSD의 한계와 Sleep 패러다임에 대한 동기(Limitations of OPSD and Motivation for the Sleep Paradigm).** 이러한 영역들에 걸친 OPSD의 실증적 성공에도 불구하고, 최근의 분석들은 그 실패 양상(failure mode)을 드러내기 시작했다. Kim et al. (2026)은 수학적 추론에서, 교사를 풍부한 특권 정보에 조건화하는 것이 모델의 인식론적 언어화(epistemic verbalisation, 추론 중 불확실성의 표현)를 억제하여, 심각한 분포 외(out-of-distribution) 성능 저하를 대가로 빠른 도메인 내(in-domain) 최적화를 가능하게 함을 보인다. 이때 Qwen3-8B, DeepSeek-Distill-Qwen-7B, Olmo3-7B-Instruct에 걸쳐 최대 40%의 하락이 나타난다. 여러 최근 OPSD 연구들 또한 장기(long-horizon) 훈련 파이프라인에서 자기 증류를 순진하게 반복하면 특권 교사로부터의 정보 누출(information leakage)과 훈련 붕괴(training collapse)로 이어질 수 있음을 보고한다(He et al. 2026; Wang et al. 2026). 이러한 결과들은 자기 증류만으로는 강건한 지속적 개선에 충분하지 않음을 시사한다. 즉, 고정된 모델에 직접 적용된 반복적 자기 개선 루프는 이전 능력을 침식하거나 추론을 불안정하게 만들 수 있다. 우리의 2단계 Sleep 프레임워크는 메모리 통합(memory consolidation)을 자기 개선(self-improvement)과 분리함으로써 이 우려를 직접적으로 다룬다. 통합은 Dreaming이 모델을 추가로 수정하기 전에 새로 획득한 지식을 새롭게 확장된 저빈도(lower-frequency) 파라미터에 먼저 안정화시켜, 반복적 자기 훈련이 유용한 이전 능력을 덮어쓸 위험을 줄인다.

**우리 연구의 위치 설정(Positioning of Our Work).** 우리의 연구는 함께 Sleep 패러다임의 동기가 되는 네 가지 축을 따라 OPSD 문헌과 차별화된다. **(i) 메모리 통합으로서의 상향 증류(Upward distillation as memory consolidation).** 위에 나열된 모든 자기 증류 연구들은 교사와 학생 사이에 동일한 백본(backbone)을 공유하며 오직 조건화 컨텍스트에서만 차이가 난다 — 특권 궤적(privileged trace)(Kim et al. 2026; Zhao et al. 2026b), 시연(demonstration)(Shenfeld et al. 2026b), 문서(document)(Stein et al. 2026), 컨텍스트 또는 시스템 프롬프트(Ye et al. 2026a,b), "간결(concise)" 접두사(Sang et al. 2026), 디코딩 구성(Zhang et al. 2026a), 스킬 요약(skill summary)(Wang et al. 2026), 수정자 컨텍스트(reviser context)(He et al. 2026), 질문 구성 경로(question-construction path)(Zhang et al. 2026d), 단문맥 부분 문자열(short-context substring)(Zhang et al. 2026c), 또는 영어 참조(English reference)(Liu et al. 2026; Qin et al. 2026). 이와 대조적으로, 우리의 지식 시딩(Knowledge Seeding) 단계는 상향 증류(upward distillation)이다. 즉, 더 작고 고빈도인 메모리 모듈이 저빈도 메모리 모듈 내의 새롭게 확장된 엄격히 더 큰 저랭크 전문가(low-rank experts) 집합으로 증류된다. 이는

24

파국적 망각을 샘플링 분포(sampling distribution)의 문제가 아니라 불충분한 용량(insufficient capacity)의 문제로 재구성하고, 고정된 교사를 재조건화하는 대신 통합 단계 사이의 점진적 파라미터 성장(gradual parameter growth)으로 이를 다룬다. **(ii) 메모리 빈도의 연속체(Continuum of memory frequencies).** 위 연구들이 증류를 평평한(flat) 학생/교사 쌍으로 취급하는 반면, 우리의 방법은 엄격히 순서 지어진 갱신 빈도(update frequency)를 갖는 메모리 블록의 사슬(chain) 위에서 작동하며 연속된 각 블록 쌍 사이에서 통합을 수행한다. 우리가 아는 한, SDFT(Shenfeld et al. 2026b)만이 우리에게 동기를 부여하는 지속 학습 요건(continual-learning desideratum)을 명시적으로 겨냥하지만, 그것은 아키텍처 성장(architectural growth)이나 계층적 메모리(hierarchical memory) 없이 그렇게 한다. **(iii) 통합을 넘어서는 2단계 과정으로서의 Sleep(Sleep as a two-phase process beyond consolidation).** OPSD 문헌은 거의 전적으로 통합 단계에 집중한다. 우리의 프레임워크는 추가로 REM 수면에 비유되는 Dreaming 단계를 포함하는데, 여기서 모델은 합성 데이터(synthetic data)의 커리큘럼을 생성하고, 경사 기반 중요도 점수(gradient-based importance score)로 샘플에 가중치를 부여하며, MoE 라우터를 활용하여 통제된 새로움(controlled novelty)을 주입한다 — 이는 (He et al. 2026; Kim et al. 2026)이 지적한 반복적 자기 개선 실패 양상에 대해 강건하도록 명시적으로 설계되었다. **(iv) 지식 시딩은 온-폴리시 증류를 모방 학습으로 증강한다(Knowledge seeding augments on-policy distillation with imitation learning).** (Liu et al. 2026; Qin et al. 2026; Sang et al. 2026; Stein et al. 2026; Ye et al. 2026a,b; Zhang et al. 2026c; Zhao et al. 2026b)와 같은 연구들의 온-폴리시 증류 목표가 학생 롤아웃에 대한 토큰별 역-KL로 환원되는 반면, 우리의 지식 시딩 목표는 GKD 스타일의 온-폴리시 증류를, 교사의 샘플링 분포와의 의미론적(semantic) 및 레벤슈타인 수준(Levenshtein-level) 정렬을 공동으로 보상하는 RL 기반 모방 학습(Learning-to-Imitate) 항으로 증강한다. 이로써 더 큰 학생이 교사의 지식을 물려받을 뿐만 아니라 교사가 그것을 사용하는 방식까지도 모방할 수 있게 한다.

## B

## 추가 실험 결과 및 세부 사항

우리의 모든 실험에서, 우리는 원래 벤치마크의 설정(각 섹션에 인용됨)을 따른다. 따라서 지면을 아끼고 정보를 중복하지 않기 위해, 모델의 세부 사항에 대해서는 해당 연구들을 참조한다. 우리의 설계에서는 추가 파라미터로 차원 64의 MLP 블록 5개를 사용하며 활성 파라미터 수(active parameter count)는 변경하지 않고 유지한다(즉, 기저 모델과 동일하며, 이는 8B 또는 3B이다).

### B.1

### 데이터셋(Datasets)

**클래스 증분 학습(Class Incremental Learning).** 우리는 세 개의 데이터셋에 대한 클래스 증분 학습에 집중한다:

- **CLINC (Larson et al. 2019):** CLINC150은 과제 지향 대화(task-oriented dialog)를 위한 다중 도메인 의도 분류(intent classification) 벤치마크로, 범위 내(in-scope) 의도 예측과 범위 외(out-of-scope, OOS) 탐지를 모두 평가하는 데 흔히 사용된다. 이는 10개 도메인에 걸친 150개의 범위 내 의도를 포함하며, 총 23.7K개의 질의(범위 내 22.5K, OOS 1.2K)를 갖는다.
- **Banking (Casanueva et al. 2020):** Banking77은 짧은 은행 고객 서비스 질의로 구성된 단일 도메인 의도 데이터셋으로, 77개의 세분화된 의도(예: 카드 문제 또는 PIN 재설정)로 레이블이 달려 있다. 13,083개의 예제를 포함하며, 의도 분포는 눈에 띄게 불균형하다.
- **DBpedia (Auer et al. 2007):** DBpedia 기반 분류는 위키피디아에서 파생된 설명(일반적으로 초록)을 온톨로지 범주(예: 책, 영화, 동물, 장소)로 매핑한다. 우리는 70-클래스, 레벨-2 설정을 사용하며 실험을 위해 10K개의 훈련 인스턴스와 1K개의 테스트 인스턴스를 서브샘플링한다.

**인-컨텍스트 학습에 대한 레벨의 효과(The Effect of Levels on In-context Learning).** Hope의 통합 스케줄(consolidation schedule)이 인-컨텍스트 학습과 장문맥 이해에 어떻게 영향을 미치는지 더 잘 분리해 내기 위해, 우리는 장문맥 하에서 질의응답(question answering)과 다중 키 검색(multi-key retrieval)을 평가한다. 이 평가에는 다음 데이터셋을 사용한다:

- **LongHealth (Adams et al. 2025):** LongHealth는 긴 가상의 환자 사례 기록으로부터 구축된 장문맥 임상 다지선다형 QA 벤치마크이다. 20개의 사례 문서(각각 약 5.1K–6.8K 단어)를 포함하며, 우리는 이 기록들에서 샘플링된 200개의 질문에 대해 평가한다.
- **QASPER (Dasigi et al. 2021):** QASPER는 전문(full-text) NLP 논문에 기반한 정보 탐색형(information-seeking) QA 벤치마크로, 약 1.6K개 논문에 걸쳐 대략 5K개의 질문을 갖는다. 우리는 각 논문의 전문을 컨텍스트로 사용한다.
- **MK-NIAH (Hsieh et al. 2024):** MK-NIAH는 RULER(Hsieh et al. 2024)의 다중 키 건초더미 속 바늘(multi-key needle-in-a-haystack) 과제로, 여러 키–값 사실이 긴 컨텍스트에 삽입되고 모델은 질의된 키에 대한 값을 검색해야 한다.

25

### B.2

### 추가 결과: BABILong

우리는 BABILong 벤치마크(Kuratov et al. 2024)에서 Hope(Sleep)를 평가하며, 다음과 비교한다: (1) 대형 모델(GPT-4 및 GPT-4o-mini (Achiam et al. 2023)); (2) RAG를 갖춘 중간 규모의 Llama-8B 모델(Dubey et al. 2024); 그리고 (3) RMT(Bulatov et al. 2022), ARMT(Rodkin et al. 2024), Titans(Behrouz et al. 2024)를 포함한 최첨단(state-of-the-art) 소형 장문맥 모델. 모든 소형 모델은 공식 BABILong 훈련 프로토콜(Kuratov et al. 2024)을 사용하여 미세조정된다. 그림 6에 나타난 바와 같이, 대형 모델은 컨텍스트 길이가 증가함에 따라 급격한 성능 저하를 보이며 128K–256K 토큰을 넘어서면 실패한다. RAG는 더 긴 컨텍스트에서 강건성을 개선하지만 여전히 시퀀스 길이가 늘어남에 따라 저하된다. 미세조정된 소형 모델들 중에서 Titans, ARMT, Hope는 대략 1M 토큰까지 비슷하게 작동한다. 이 지점을 넘어서면 Titans와 ARMT는 급격히 저하되는 반면, Hope는 최대 10M 토큰의 극단적 길이에서도 안정적으로 유지된다. 이 강건성은 단기 활성화(short-lived activation)를 간결한 파라메트릭 표현으로 변환하는 Sleep의 명시적 통합 및 자기 개선 메커니즘에 의해 주도된다. 통합 과정에서 더 높은 수준의 추상화를 학습함으로써, Hope는 과제 관련 정보를 더 효율적으로 보존하여 증가하는 컨텍스트 길이에 대한 민감도를 줄인다. 마지막으로, 우리는 Hope를 포함한 모든 소형 모델이 미세조정 없이는 상당한 성능 하락을 겪는다는 것을 관찰한다. 초장문맥(ultra-long context)을 압축하려면 중요한 정보를 식별하고 보존하기 위한 저빈도 메모리 레벨에서의 충분한 용량이 필요하다. 미세조정은 이러한 구성요소들이 빠르게 적응하고 추론 중 고빈도 메모리와 효과적으로 협응하도록 해준다.

### B.3

### 추가 실험: ARC

우리는 선행 연구(Akyurek et al. 2024a; Zweiger et al. 2025)의 소수 샷(few-shot) ARC 실험 프로토콜을 따르고 이를 우리의 Sleep 패러다임에 맞게 조정한다. 백본으로는 Llama-3.2-1B를 사용한다. 일반적인 관행을 따라, 표준 구성 하에서 풀 수 없는 과제를 피하기 위해 데이터의 부분집합을 필터링하여, 훈련용 11개 과제와 평가용 8개의 보류(held-out) 과제를 산출한다. 각 Sleep 주기(cycle) 동안, 모델은 먼저 이전 기억들을 통합한 뒤, 소수 샷 데모로부터 합성 경험(synthetic experience)을 생성함으로써 꿈을 꾼다(dreams). 각 과제에 대해 우리는 60개의 꿈을 샘플링하고 그중 45개를 기각한다. 테스트 시점에는, 각 미지의 과제에 대해 모델이 5개의 꿈을 생성하고 보류된 출력을 예측하기 전에 이들을 독립적으로 적용한다. 우리는 정답을 산출하는 꿈의 비율을 보고한다. 기준선(baseline)으로는 Zweiger et al. (2025)을 따라 다음을 사용한다: (i) ICL(In-Context Learning); (ii) TTT + 합성 갱신(synthetic updates)(꿈 없음); 그리고 (iii) SEAL(Zweiger et al. 2025). 이 설정에서 Sleep은 80%의 성공률을 달성하여 다른 방법들보다 높다.

### B.4

### 추가 세부 사항

**표 5: GRPO, SFT, Sleep에 대한 훈련 구성(Training Configuration)**

| 파라미터(Parameter) | GRPO | SFT | Sleep |
|---|---|---|---|
| 학습률(Learning Rate) | 5 × 10−6 | 5 × 10−6 | 5 × 10−6 |
| 유효 배치 크기(Effective Batch Size) | 32 | 32 | 32 |
| 훈련 스텝(Training Steps) | 500 | 100 | 100 |
| LoRA 랭크(LoRA Rank, r) | 64 | 64 | 64 |
| LoRA 알파(LoRA Alpha, α) | 128 | 128 | 128 |

### B.5

### 효율성(Efficiency)

동일한 스텝 수에서, SFT는 우리의 방법보다 4배 더 효율적이다. 그러나 그들의 성능은 비교 가능하지 않다. 이에 따라, 우리는 특정 성능을 목표로 할 때의 효율성 또한 비교한다. 우리는 AIME-24, AIME-25, HMMT-25에서 동일한 성능을 달성하도록 모델들을 훈련하였다. 이 경우, SFT는 우리 설계의 성능에 맞추기 위해 각각 4.3배, 3.6배, 4.8배의 실제 소요 시간(wall-clock time)을 필요로 한다. 따라서 이러한 관점에서, 특정 성능을 목표로 할 때 Sleep은 매우 효율적이다.
