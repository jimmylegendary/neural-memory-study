[
  {
    "severity": "major",
    "location": "§1.3 표 1-1 및 해설, §1.6 말미, 요약",
    "issue": "“gradient는 (오차)k^T의 rank-1 outer product”라는 설명을 일반 fast-weight memory에 적용한다. 이는 선형 matrix memory에만 성립한다. Titans의 deep MLP memory에서는 각 층의 gradient가 δ_l h_{l-1}^T이며, 전체 gradient는 여러 weight 행렬에 대한 tuple이다.",
    "evidence": "papers/2501.00663.txt: §3.1 Eq. 12–14 부근은 memory를 MLP로 허용하고 그 전체 parameter에 대해 ∇ℓ을 계산한다. 같은 파일 §3.1 ‘Memory Architecture’ 부근은 L_M≥1인 MLP를 사용한다고 명시한다. papers/external/2407.04620.txt: abstract 부근도 TTT-MLP를 two-layer MLP hidden state로 정의한다.",
    "suggested_fix": "outer-product 설명을 “선형 matrix memory W에서”로 한정한다. deep memory에는 “각 층에서 dW_l=δ_l h_{l-1}^T이고, C개 token을 모으면 각 weight 행렬별 inner dimension C의 GEMM이 된다”라고 수정한다."
  },
  {
    "severity": "major",
    "location": "§1.1 slow weights 정의 및 ‘Θ는 얼어 있다’ 불변식",
    "issue": "slow weights Θ가 pretraining 이후 계속 읽기 전용이라는 설명은 이 장이 같은 연구 라인에 포함한 Sleep에는 성립하지 않는다. Sleep 단계는 fast memory의 정보를 더 느린 parameter로 이관하면서 post-pretraining parameter를 갱신하므로, 이를 여섯 논문 전체의 유지되는 불변식으로 선언하면 표 1-2의 Sleep 설명과도 충돌한다.",
    "evidence": "papers/2606.03979.txt: 서론 lines 60–63 부근은 sleep이 fast unstable modules의 memory를 stable low-frequency components로 consolidation한다고 설명한다. §3 도입 lines 294–298 부근은 static train/test paradigm을 wake/sleep lifecycle로 대체하고 sleep 동안 memory를 consolidate한다고 명시한다.",
    "suggested_fix": "Θ의 불변성을 Titans·TTT 계열의 일반적인 wake/inference phase로 한정하고, Sleep에서는 sleep phase가 이 불변식을 깨고 slow parameter를 갱신한다는 예외를 §1.1에서 명시한다."
  },
  {
    "severity": "minor",
    "location": "§1.4 식 (M) 아래 ∇_Wℓ 설명",
    "issue": "∇_Wℓ을 ‘실패를 줄이는 방향’이라고 설명했지만 gradient는 손실의 최대 증가 방향이고, 실패를 줄이는 방향은 −∇_Wℓ이다. 식 (M)의 부호는 올바르지만 바로 아래 해설이 반대로 되어 있다.",
    "evidence": "papers/2501.00663.txt: §3.1 Eq. 8 lines 299–305 부근은 M_t=M_{t-1}−θ_t∇ℓ로 갱신하며, Eq. 14도 momentary surprise gradient 앞에 음의 부호를 둔다.",
    "suggested_fix": "‘∇_Wℓ은 실패가 가장 증가하는 방향이며, −η_t∇_Wℓ이 실패를 줄이는 실제 write/update 방향이다’로 수정한다."
  }
]