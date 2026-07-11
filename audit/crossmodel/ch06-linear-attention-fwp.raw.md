[
  {
    "severity": "major",
    "location": "§6.3 식 (6-2)의 RetNet 설명; 표 6-2 RetNet 행",
    "issue": "RetNet의 retention 계수 α를 데이터와 무관한 '학습된 상수'라고 했지만, 원 모델의 γ는 head 인덱스로 미리 정해 고정하며 학습하지 않는다. 따라서 RetNet을 outer loop가 학습하는 retention gate로 분류한 것은 실제 아키텍처와 다르다.",
    "evidence": "papers/external/2307.08621.txt: §2.2 Eq. (8) 부근에서 γ를 head별 공식으로 설정하고 여러 layer에서 동일하게 유지하며 고정한다고 명시한다. §3.1도 실험용 γ 일정을 별도로 정해 사용한다.",
    "suggested_fix": "RetNet을 'head별로 미리 정해진 고정 decay'로 고치고, 학습되는 상수라는 설명을 삭제한다."
  },
  {
    "severity": "minor",
    "location": "§6.3 첫 문단; 표 6-3 Hebbian 행",
    "issue": "Hebbian additive write에서 state norm이 단조 증가한다고 단정했지만 성립하지 않는다. outer-product 갱신은 부호를 가지므로 이전 항과 상쇄될 수 있다.",
    "evidence": "papers/external/2406.06484.txt: §2.2는 linear attention의 갱신을 S_t = S_{t-1} + v_t k_t^T로 정의한다. 예를 들어 k_2=k_1, v_2=-v_1이면 S_2=0이 되어 norm이 감소한다.",
    "suggested_fix": "'norm이 단조 증가한다'를 '명시적 제거 기제가 없어 norm 증가와 collision이 누적될 수 있다'로 바꾼다."
  },
  {
    "severity": "minor",
    "location": "§6.5 안정성 설명 마지막 문단",
    "issue": "Longhorn의 gate head가 임의로 큰 η_t를 출력할 수 있다고 했지만 실제 Longhorn은 β_t에 sigmoid를 적용해 각 성분을 (0,1)로 제한한다. 모든 양의 step size에 대한 안정성은 closed-form 식의 수학적 성질이지 실제 gate가 무제한 값을 출력하기 때문에 필요한 성질은 아니다.",
    "evidence": "papers/external/2407.14207.txt: Algorithm 1과 §3.2에서 β_t = Sigmoid(W_β x_t) ∈ (0,1)^d로 정의한다. Eq. (6)의 closed-form 자체는 β≥0에 대해 제시된다.",
    "suggested_fix": "'이론적 closed-form은 임의의 양의 β에서 안정하지만 실제 아키텍처의 β는 sigmoid로 제한된다'고 두 층위를 구분한다."
  },
  {
    "severity": "major",
    "location": "§6.6 식 (6-6) 직후의 GDN 환원 주장",
    "issue": "제시한 조건 w_t=α_t1, a_t=α_tη_t1, hat{k}_t=tilde{k}_t=k_t를 식 (6-6)에 넣으면 write 항은 v_t k_t^T로 남는다. 따라서 η_t v_t k_t^T를 갖는 식 (6-4)로 환원되지 않는다.",
    "evidence": "papers/external/2503.14456.txt: §4.1 Eq. (17)의 write 항은 v_t^T tilde{k}_t이며 a_t가 곱해지지 않는다. Eq. (7)은 tilde{k}_t를 별도로 구성한다.",
    "suggested_fix": "추상 식 수준에서는 최소한 tilde{k}_t=η_t k_t로 두어야 한다. 다만 RWKV-7 Eq. (7)의 실제 parameterization이 임의의 GDN을 정확히 포함한다는 추가 증명이 없으므로 'GDN을 부분 경우로 포함한다'는 단정은 삭제하거나 완화한다."
  },
  {
    "severity": "minor",
    "location": "§6.6 '이 gate는 누가 학습하는가' 문단",
    "issue": "두 key 변조의 채널 배율과 보간 계수까지 작은 head가 token마다 산출한다고 했지만, removal-key multiplier ξ와 replacement-rate booster α는 token-independent trainable parameter다. token마다 산출되는 것은 a_t와 w_t 등이다.",
    "evidence": "papers/external/2503.14456.txt: §4.1 Eq. (4), (6), (7)에서 a_t는 입력 기반 low-rank MLP 출력이지만 κ_t=k_t⊙ξ와 tilde{k}_t=k_t⊙lerp(1,a_t,α)의 ξ·α는 학습된 고정 parameter라고 설명한다.",
    "suggested_fix": "w_t와 a_t는 token-dependent head 출력, ξ와 replacement booster는 outer loop에서 학습되는 token-independent vector라고 구분한다."
  },
  {
    "severity": "minor",
    "location": "§6.8(b) DeltaNet 예제 마지막 문단",
    "issue": "같은 쌍들을 반복 제시하면 LMS가 최소제곱 해에 수렴한다고 조건 없이 단정했지만, constant per-token step의 incremental SGD는 일반적으로 정확한 최소제곱 해로 수렴하지 않는다. 예를 들어 k=1, target이 +1과 -1로 번갈아 나오고 η=1이면 W는 ±1 사이를 순환하며 최소제곱 해 0에 수렴하지 않는다.",
    "evidence": "papers/external/2406.06484.txt: §2.2는 DeltaNet을 online regression loss에 대한 단일 SGD step으로 정의하고 β_t=σ(W_βx_t)∈(0,1)을 사용한다. 반복 데이터에 대한 무조건적 최소제곱 수렴 정리는 제시하지 않는다.",
    "suggested_fix": "감소 step size, 일관된 binding, 또는 표준 LMS의 확률적 가정 등 수렴 조건을 명시하거나 해당 수렴 단정을 삭제한다."
  }
]