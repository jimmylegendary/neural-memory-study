[
  {
    "severity": "major",
    "location": "§5.6, 표 5-1의 softmax attention 행",
    "issue": "KV를 분리 보관해 쓰기 간섭이 없다는 사실을 읽기 crosstalk 부재와 혼동했다. 유한한 softmax에서는 비대상 key의 가중치도 양수이므로 출력에 다른 value가 섞인다. 따라서 'read 오염 없음'과 'exact capacity 무제한'은 일반적으로 거짓이며, 무제한인 것은 원시 KV 저장량이다.",
    "evidence": "papers/external/2008.02217.txt: §2/Theorem 3 부근은 조건부 저장 용량을 유한한 지수 규모로 정리하고, 이어지는 metastable-state 논의는 pattern 분리가 부족하면 softmax가 거의 균등해져 산술평균 근처로 수렴한다고 설명한다.",
    "suggested_fix": "softmax attention은 '쓰기 충돌 없음, 원시 KV 저장량은 L과 함께 증가'로 기술하라. 읽기에는 softmax 혼합에 의한 간섭이 남으며, 정확한 retrieval capacity는 pattern 분리도·온도·허용 오차에 의존한다고 표를 수정하라."
  },
  {
    "severity": "major",
    "location": "§5.5 'delta rule의 보증' 문단",
    "issue": "고정 step size로 sample을 순환하는 LMS가 일반적으로 least-squares 해로 수렴한다고 했지만, 이는 불일치 선형계에서는 성립하지 않는다. 예를 들어 scalar W, k1=k2=1, v1=0, v2=1, η=1이면 W가 0과 1 사이를 영원히 왕복하며 least-squares 해 0.5로 수렴하지 않는다. 또한 이를 Atlas Prop. 1의 증명 논리와 동일시했지만 원전은 cyclic online LMS가 아니라 full-batch gradient descent를 분석한다.",
    "evidence": "papers/2505.23735.txt: Supporting Proofs의 Proposition 1 부근은 목적함수 ‖MK−V‖²_F와 full-batch 갱신을 제시하고, step-size 조건 아래 보간해로의 수렴을 논한다. cyclic per-sample LMS 수렴 정리는 제시하지 않는다.",
    "suggested_fix": "선형독립 key와 m≤d_k인 일관된 보간 문제에서는 normalized cyclic update가 보간해에 수렴할 수 있다고 범위를 제한하라. 일반 least-squares 수렴에는 full-batch GD 또는 감소 step-size 조건이 필요하다고 분리해서 설명하라."
  },
  {
    "severity": "minor",
    "location": "§5.3 energy function 직후 문단",
    "issue": "손상된 입력이 '가장 가까운' 저장 pattern으로 수렴한다고 단정했다. Hopfield dynamics가 보장하는 것은 energy 감소와 어떤 attractor로의 수렴이지, Hamming 또는 Euclidean 거리상 최근접 저장 pattern의 선택이 아니다. spurious attractor나 비정상적인 basin 경계 때문에 더 먼 pattern 또는 저장하지 않은 상태로 갈 수 있다.",
    "evidence": "papers/external/1606.01164.txt: 고전 모델 capacity 설명 부근은 과적재 시 여러 memory가 합쳐져 저장 pattern과 무관한 ground state가 생긴다고 설명한다. 같은 장의 원전 계열 결과도 최근접 검색 보장을 주지 않는다.",
    "suggested_fix": "'초기 상태가 저장 pattern의 attraction basin 안에 있으면 그 pattern으로 수렴한다'로 고치고, 최근접 pattern 보장은 없다고 명시하라."
  },
  {
    "severity": "minor",
    "location": "§5.4 modern Hopfield의 1-step retrieval 설명 마지막 문장",
    "issue": "'energy를 날카롭게 만들수록' 반복 retrieval이 1-step으로 접힌다는 보편적 인과를 제시했지만 원전 정리는 그런 단조 관계를 보장하지 않는다. 1-step 오차 경계는 pattern separation이 충분하고 query가 해당 fixed point의 basin에 있다는 조건에 의존한다.",
    "evidence": "papers/external/2008.02217.txt: Theorem 4 부근은 one-update 오차가 separation Δ_i에 대해 지수적으로 작다고 정리하며, 부록의 trade-off 논의는 작은 separation에서 수렴이 느리고 오차가 커진다고 명시한다.",
    "suggested_fix": "'잘 분리된 pattern과 해당 basin 안의 query에 대해서는 한 번의 update로 fixed point에 가까워진다'로 제한하고, energy sharpness 자체가 충분조건이라는 설명은 삭제하라."
  }
]