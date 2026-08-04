# 초록 (Abstract)

인공 신경망은 과제를 순서대로 학습할 때 이전 과제를 덮어쓰는 **파국적 망각(catastrophic forgetting)** 을 보인다. 뇌는 지속적으로 학습하며, 새로운 훈련 사이에 기억 통합을 위한 수면이 들어갈 때 학습이 특히 잘 되는 경우가 많다. 이 연구는 스파이킹 신경망(spiking neural network, SNN)을 사용해 파국적 망각의 기전과 수면이 이를 막는 역할을 조사한다.

신경망은 복잡한 foraging task를 학습할 수 있었지만 서로 다른 과제를 순차적으로 훈련하면 이전 과제를 잊었다. 시냅스 가중치 공간에서 새 과제 훈련은 가중치 상태를 이전 과제의 manifold에서 멀어지게 했다. 새 과제 훈련 사이에 생물학적 수면을 모사한 offline reactivation 구간을 넣으면, 가중치 상태가 이전 manifold를 벗어나지 않으면서 새 과제 manifold와의 교차 영역으로 수렴했다. 논문은 수면 중 자발적 재활성화가 과거 원 데이터를 명시적으로 다시 사용하지 않고도 공동 시냅스 표현을 만들 수 있음을 보인다.

# 저자 요약 (Author Summary)

현대 신경망은 많은 과제에서 높은 성능을 내지만, 새 과제를 학습할 때 이전 과제의 성능을 희생하는 경우가 많다. 사람과 동물은 새 정보를 기존 지식에 통합한다. 저자들은 감각 처리와 강화학습을 모사하는 SNN에서 학습과 sleep-like activity를 번갈아 수행하면, 오래된 memory trace가 자발적으로 replay되어 이전 기억을 보호하면서 새 과제를 흡수하는 가중치 표현을 만든다는 것을 보인다.

# 1. 서론 (Introduction)

단순 interleaved training은 이전 과제의 실제 데이터를 계속 보관해 새 데이터와 섞으면 망각을 완화할 수 있다. 그러나 이는 과거 데이터를 영구 저장해야 한다. 생물학적 수면은 과거 입력이 없는 상태에서도 학습 때 형성된 ensemble을 재활성화한다. 연구의 질문은 local learning rule과 spike dynamics만으로 그러한 replay가 일어나고, 그 replay가 sequential learning의 가중치 간섭을 줄일 수 있는가이다.

모델은 다층 SNN과 reinforcement learning을 사용한다. 두 개의 상보적인 foraging task는 pattern discrimination 규칙이 달라 순차 훈련 시 서로 간섭한다. wake에서는 외부 입력과 reward가 들어오고, sleep에서는 외부 입력과 reward가 제거된 채 내부 자발 활동이 진행된다. sleep 중 synaptic plasticity는 계속 작동한다.

# 2. 모델과 과제 (Model and Tasks)

네트워크는 input, hidden, output population으로 이루어지고 spike-timing과 reward 신호에 따라 연결을 갱신한다. foraging 환경에서 agent는 감각 pattern을 보고 두 행동 중 하나를 선택한다. Task 1과 Task 2는 입력–행동 mapping의 서로 다른 부분을 요구하여, 한 과제만 훈련할 때에는 각각 높은 성능에 도달하지만 순차 학습에서는 두 번째 과제가 첫 번째 가중치 구성을 이동시킨다.

저자들은 학습 결과를 단순 accuracy뿐 아니라 고차원 synaptic weight space의 trajectory로 본다. 한 과제를 성공적으로 수행하는 여러 가중치 상태는 하나의 영역 또는 manifold를 이룬다. 두 과제를 함께 수행하려면 가중치가 두 manifold가 양립하는 교차 영역에 있어야 한다.

# 3. 결과 (Results)

## 3.1 순차 학습은 파국적 망각을 만든다

Task 1을 학습한 뒤 Task 2만 계속 학습하면 Task 2 성능은 올라가지만 Task 1 성능은 급격히 낮아진다. 가중치 상태는 Task 1 solution manifold에서 Task 2 방향으로 이동한다. 이는 출력 성능의 손실과 가중치 기하의 이동이 함께 일어남을 보여준다.

## 3.2 Sleep-like activity는 이전 기억을 재활성화한다

sleep 구간에는 과거 task sample을 다시 주입하지 않는다. 외부 입력이 없는 상태에서 네트워크 내부 noise와 학습으로 형성된 recurrent connectivity가 과거 학습 pattern을 자발적으로 재현한다. population spike pattern을 분석하면 wake에서 학습한 memory trace와 닮은 상태가 반복적으로 나타난다.

이 replay는 저장한 과거 dataset을 그대로 재생하는 experience replay와 다르다. 모델 자체의 동역학이 pattern을 복원하며, sleep 동안 local synaptic rule이 그 activity에 반응한다.

## 3.3 새 과제 학습 사이의 수면이 망각을 줄인다

Task 2 wake training을 계속 이어가는 대신 wake 구간과 sleep 구간을 교대로 배치하면 Task 1 성능이 보존되고 Task 2도 학습된다. sleep이 단지 훈련 횟수를 줄여 간섭을 늦춘 것이 아님을 확인하기 위해 wake update 수와 여러 control을 맞춘다. 무작위 activity 또는 plasticity를 끈 sleep은 같은 효과를 내지 못한다.

## 3.4 공동 가중치 표현의 형성

weight trajectory는 sleep이 가중치를 과거 Task 1 manifold 쪽으로만 되돌리는 것이 아님을 보인다. 새 task 학습과 번갈아 작동하면서 두 task의 solution이 겹치는 영역을 찾는다. 결과적으로 하나의 synaptic configuration이 두 기억을 공동으로 표현한다. 논문 제목의 “joint synaptic weight representation”은 이 기하학적 결과를 가리킨다.

## 3.5 과거 원 데이터 없이도 가능한 replay

저자들은 sleep에서 과거 입력을 직접 재사용하지 않았음을 강조한다. 다만 network state와 weights에는 당연히 과거 학습의 흔적이 남아 있다. replay는 그 내부 흔적에서 발생한다. 그러므로 “데이터를 전혀 저장하지 않는다”는 말은 별도 episodic buffer에 원 sample을 보존하지 않는다는 뜻이지, 과거 정보가 물리적으로 사라진다는 뜻이 아니다.

# 4. 가중치 공간 해석

각 과제의 loss 또는 performance surface를 가중치 공간에 투영하면 순차 wake training은 최신 과제의 좋은 영역으로 빠르게 이동하지만 이전 과제의 좋은 영역을 떠난다. sleep reactivation은 과거 pattern에 대한 gradient와 유사한 제약을 생성해 trajectory가 old-task manifold를 따라 움직이게 한다. 새 task wake update와 이 제약이 교대되면서 intersection으로 접근한다.

이 해석은 단순히 synapse 크기를 전역적으로 낮추는 homeostatic scaling만으로는 설명되지 않는다. 어떤 memory ensemble이 재활성화되는지와 그 activity에 의존한 plasticity가 중요하다. 논문은 여러 sleep duration, interleaving schedule, noise 조건을 비교해 이 주장을 뒷받침한다.

# 5. 논의 (Discussion)

결과는 생물학적 뇌 전체의 수면을 재현한 것이 아니라, SNN에서 autonomous replay와 local learning이 sequential interference를 완화할 수 있다는 계산 모델이다. 두 개의 제한된 foraging task와 특정 network/learning rule에서 얻은 proof-of-principle이므로, 대규모 언어 모델이나 임의의 continual-learning 문제에 곧바로 일반화할 수 없다.

모델은 실제 NREM/REM stage의 복잡한 생리, hippocampus–neocortex hierarchy, 장기간 memory selection을 모두 구현하지 않는다. 저자들은 sleep-like activity가 과거 task manifold를 보존하면서 새 task와의 공동 표현을 찾는 기전을 제시한다. 후속 과제 수가 많을 때의 capacity, replay가 어떤 기억을 선택하는지, 실제 생물학적 oscillation과의 대응은 남은 문제다.

# 6. 방법 (Methods)

Methods는 neuron과 synapse dynamics, network connectivity, reward-modulated plasticity, foraging stimulus와 행동 규칙, wake/sleep protocol, performance metric, replay similarity, weight-space projection을 상세히 정의한다. 실험은 여러 seed와 control을 사용하며, 수면 중 외부 task input과 reward를 차단한다. figure 1–7은 architecture, sequential-learning 성능, replay activity, interleaved schedule, manifold trajectory와 control 결과를 순서대로 보여준다.

# 데이터와 재현

원 논문은 PLOS의 CC BY 4.0으로 공개되어 있으며 관련 데이터와 supporting information의 위치를 제공한다. 뒤의 원문 보존 부록에는 31쪽 Version of Record 전체가 포함되어 equation, figure, caption, method parameter를 직접 대조할 수 있다.
