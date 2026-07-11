[
  {
    "severity": "minor",
    "location": "§11.1 및 요약",
    "issue": "β=0.9인 EMA에서 최근 6개가 누적 기여의 50% 이상, 최근 43개가 99% 이상이라는 계산은 틀렸다. 정상상태에서 최근 n개의 비중은 1-β^n이므로 각각 46.86%, 98.92%이며, 임계값을 넘으려면 7개와 44개가 필요하다.",
    "evidence": "papers/2512.24695.txt: §4.3 부근은 기여도를 β^i(1-β)로 정의한 뒤 6개/43개라고 주장한다. 이 원전 수치를 그대로 계산하면 1-0.9^6=0.4686, 1-0.9^43≈0.9892이다.",
    "suggested_fix": "수치를 최근 7개가 50% 이상, 최근 44개가 99% 이상으로 고치고 요약도 함께 수정한다."
  },
  {
    "severity": "major",
    "location": "§11.1 및 §11.7(a)",
    "issue": "직교 task에 관한 설명이 서로 양립하지 않는다. §11.7의 선형 모델에서는 task gradient가 직교하면 새 update가 이전 task의 예측을 바꿀 수 없다고 정확히 계산하지만, §11.1은 동일한 직교-gradient 설정에서 momentum이 과거 subspace를 잊는 것이 catastrophic forgetting을 일으킨다고 단정한다. gradient가 계속 직교한다면 과거 subspace를 기억하지 못한다는 사실만으로 이전 loss가 증가하지 않는다.",
    "evidence": "papers/2512.24695.txt: §4.3, Eq.45 부근은 task gradient가 직교 방향 u_i에 놓인다고 가정한 뒤, momentum이 u_t 방향으로 이동해 과거 gradient subspace를 잊으므로 이전 task가 손상될 수 있다고 서술한다. 그러나 장의 §11.7(a) 계산처럼 이전 입력과 현재 update가 직교하면 이전 예측 변화는 x_A^T ΔΘ=0이다.",
    "suggested_fix": "NL의 주장을 가설로 귀속하고, forgetting이 성립하려면 학습된 표현의 이동, 비직교 update 성분, task 간 Hessian 결합 등 추가 조건이 필요하다고 명시한다. 순수 선형·직교 예제에서는 forgetting이 없다고 유지한다."
  },
  {
    "severity": "major",
    "location": "§11.2 식 (11-1) 뒤 및 표 11-1",
    "issue": "EWC가 이전 task마다 Θ*와 F 두 개의 parameter-size buffer를 반드시 보관하므로 memory가 task 수에 비례한다는 복잡도 설명은 과도하다. 여러 diagonal quadratic penalty의 합은 하나의 quadratic으로 합칠 수 있으므로 누적 precision과 가중 anchor만 유지하는 구현은 O(P) 상태로 가능하다.",
    "evidence": "EWC 원전 arXiv:1612.00796 §2, Eq.3 직후는 세 번째 task로 갈 때 이전 task penalty를 각각 둘 수도 있지만, quadratic penalty의 합 자체가 quadratic이므로 하나로 합칠 수도 있다고 명시한다.",
    "suggested_fix": "기본 상태를 O(P)의 누적 Fisher와 결합 anchor로 설명하고, task별 posterior를 별도로 보존하는 변형에서만 O(TP) memory가 든다고 구분한다."
  },
  {
    "severity": "minor",
    "location": "§11.7(b)",
    "issue": "∇Θ ŷ=x라는 사실만으로 empirical Fisher가 xx^T라고 결론 내린 유도는 성립하지 않는다. Fisher는 예측값의 Jacobian이 아니라 log-likelihood score의 외적 또는 그 기대값이다. 현재처럼 제곱 오차의 최적점에서 관측 label로 empirical Fisher를 계산하면 score가 0이어서 Fisher도 0이 될 수 있다.",
    "evidence": "EWC 원전 arXiv:1612.00796 §2는 F를 posterior의 diagonal precision이자 minimum 근방 loss curvature로 정의한다. 단순히 출력 Jacobian의 외적이라고 정의하지 않는다.",
    "suggested_fix": "고정 분산 Gaussian likelihood y∼N(Θ^T x,σ²)을 명시하고 true Fisher가 xx^T/σ²임을 유도한 뒤, 상수 σ^-2를 λ_EWC에 흡수해 diag(1,0)을 사용한다고 설명한다."
  },
  {
    "severity": "major",
    "location": "§11.2 parameter isolation 문단",
    "issue": "빠른 블록의 expert reset이 parameter-growth 관리 문제를 해결한다고 서술하지만, reset은 지식을 넘긴 source 고주파 블록의 임시 low-rank parameter만 회수한다. 지식을 받은 저주파 target에는 매 consolidation마다 새 expert가 남으므로 전체 성장, 특히 최저주파 블록의 누적 성장은 제한하지 못한다. 사전 할당 mask도 tensor shape만 고정할 뿐 유한 pool 고갈 문제를 없애지 않는다.",
    "evidence": "papers/2606.03979.txt: §3.2는 target MLP에 매 sleep마다 새 low-rank expert를 추가한다고 정의한다. §3.3은 전이 후 source인 MLP^(f_{ℓ*-1})에 과거 추가된 low-rank parameter를 reset한다고 명시하며, 구현 노트는 초기 사전 할당을 가능한 대안으로만 제시한다.",
    "suggested_fix": "reset은 고주파 source capacity를 재활용할 뿐 전체 또는 최저주파 capacity를 bound하지 않는다고 고친다. 사전 할당 방식에는 expert pool 크기와 고갈 시 eviction·merge 정책이 별도로 필요하다고 명시한다."
  },
  {
    "severity": "major",
    "location": "§11.6 LoRA 문단 및 §11.8",
    "issue": "한쪽에서는 LoRA가 backward 연산을 줄이지 않는다고 단정하고, 다른 쪽에서는 backward가 low-rank expert 또는 LoRA 크기로 국한된다고 주장한다. 둘 다 부정확하다. activation gradient는 trainable adapter에 도달하도록 backbone을 통과하지만, frozen weight의 parameter-gradient GEMM은 생략되므로 full fine-tuning보다 backward 비용이 감소한다. 반대로 총 backward 비용이 adapter parameter 수에만 비례하는 것도 아니다.",
    "evidence": "papers/2606.03979.txt: §3.3은 student의 모든 기존 parameter를 freeze하고 expanded parameter만 update한다고 명시한다. LoRA 원전 arXiv:2106.09685 §4는 대부분 parameter의 gradient를 계산하지 않아 GPT-3 175B에서 training throughput이 32.5에서 43.1 tokens/s로 약 25% 향상됐다고 보고한다.",
    "suggested_fix": "LoRA는 optimizer state와 weight-gradient 계산을 크게 줄이지만 activation-gradient 전파는 남는다고 설명한다. Sleep 비용도 'trainable state는 low-rank 크기지만 backward FLOP은 전체 그래프와 adapter 배치 위치에 좌우된다'고 수정한다."
  }
]