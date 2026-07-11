[
  {
    "severity": "major",
    "location": "9.2 분해 및 요약의 nonlinear transition 판정",
    "issue": "비선형 transition이면 exact 병렬 재조직이 없고 chunk-start anchor가 '유일하게 알려진 수'라고 단정한다. 비선형이라는 사실만으로 exact 병렬화 불가능성이 따라오지 않으며, 원전도 이를 불가능성 정리로 제시하지 않는다. 따라서 transition의 linear/nonlinear 여부만으로 병렬화 가능성을 완전히 판정한다는 장의 중심 기준은 지나치게 강하다.",
    "evidence": "papers/2511.07343.txt: §1 부근은 비선형 recurrence의 병렬화를 'largely unsolved outside of ... specialized linear RNNs'라고 한정하며, 참고문헌에는 Gonzalez et al., 'Towards scalable and stable parallelization of nonlinear RNNs'도 수록한다. 이는 일반적 난제라는 주장이지 모든 비선형 recurrence에 exact 재조직이 없다는 주장이 아니다.",
    "suggested_fix": "'일반적인 deep-memory gradient recurrence에 대해 현재 알려진 효율적인 exact sequence-parallel 재조직은 없다'로 범위를 한정하고, 특수 구조를 가진 비선형 recurrence는 별도 분석이 필요하다고 명시한다."
  },
  {
    "severity": "major",
    "location": "9.2 anchor 설명 및 9.6 1단계",
    "issue": "공유 anchor에서 계산한 C개의 per-token gradient를 'batch C짜리 forward+backward 한 번'으로 얻는다고 설명한다. 일반적인 batch backward는 per-token parameter gradient가 아니라 합산 gradient 하나만 반환한다. 특히 momentum scan에는 개별 g_t가 필요하므로 표준 batch backward 한 번으로는 제시된 알고리즘을 구현할 수 없다.",
    "evidence": "papers/external/2407.04620.txt: App. A.2 부근은 표준 backward가 Σ_t G_t만 계산하며 개별 G_t들을 matmul로 함께 계산할 수 없다고 명시한다. 이어 App. A.3은 dual form이 개별 G_t와 중간 W_t를 물화하지 않고 별도의 수식으로 출력과 chunk-end state를 계산한다고 설명한다.",
    "suggested_fix": "TTT dual form은 개별 gradient를 얻는 표준 batch backward가 아니라 특수한 dual 계산임을 명시한다. Titans의 momentum scan처럼 개별 g_t가 필요한 경우에는 vmap/Jacobian 계열의 per-example gradient 계산 또는 이에 상응하는 전용 tensorization이 필요하다고 고친다."
  },
  {
    "severity": "minor",
    "location": "9.4 첫 문단의 DeltaNet 복잡도",
    "issue": "Householder factor의 누적곱을 물화하면 token당 O(d^3)라고 서술한다. 그러나 A(I-ηkkᵀ)=A-η(Ak)kᵀ로 계산하면 factor 하나의 적용·누적은 O(d^2)이다. DeltaNet의 문제는 cubic 연산이 필연적이라는 것이 아니라 d×d 상태의 순차 처리와 물화/I/O 비용이다.",
    "evidence": "papers/external/2406.06484.txt: Eq. 3 부근은 transition을 S_t=S_{t-1}(I-β_tk_tk_tᵀ)+β_tv_tk_tᵀ로 쓰며, §2.2는 recurrent form 전체 복잡도를 O(Ld^2)라고 명시한다.",
    "suggested_fix": "O(d^3)를 O(d^2) per factor로 정정하고, dense d×d 중간 상태를 매 token 순차적으로 갱신·물화해야 해 sequence 병렬성과 메모리 효율이 나쁘다는 점을 실제 병목으로 설명한다."
  },
  {
    "severity": "minor",
    "location": "9.10 식 (9-7)의 적용 범위",
    "issue": "F(C)≈4C²d+4Cd²를 인스턴스 1과 2 모두의 linear-memory kernel 비용으로 제시하지만, DeltaNet에는 앞서 식 (9-5)에 나온 K_nW_ξᵀ 계산과 triangular solve/UT 변환 관련 연산이 추가된다. 따라서 표 9-3의 수치적 arithmetic intensity는 DeltaNet까지 포괄하지 않는다.",
    "evidence": "papers/external/2406.06484.txt: Eq. 10–11 부근은 T=(I+tril(diag(β)KKᵀ,-1))^{-1}diag(β), U=TV, W=TK를 추가로 계산한다. Eq. 8–9 부근의 상태·출력 계산에도 W S 항이 들어가므로 단순 decay kernel의 두 C²d GEMM과 두 Cd² GEMM만으로는 연산량이 완결되지 않는다.",
    "suggested_fix": "식 (9-7)과 표 9-3을 GLA/RetNet형 kernel의 이상화된 roofline으로 한정하거나, DeltaNet의 Gram matrix·triangular solve·추가 state interaction을 포함한 별도 F_Delta(C), B_Delta(C)를 제시한다."
  },
  {
    "severity": "minor",
    "location": "9.7 둘째 문단 및 요약의 C=1 inference 주장",
    "issue": "C=64로 훈련한 모델을 C=1로 실행하면 '무너진다'고 단정하지만, 인용한 TNT Fig. 2는 inference chunk 8, 16, 32, 64, 128, 256, 512만 측정했으며 C=1 결과는 보고하지 않는다.",
    "evidence": "papers/2511.07343.txt: Fig. 2 부근의 x축은 8–512이고, 본문은 C=64에서 최적이며 더 작은 chunk가 항상 우수하다는 직관에 반한다고만 보고한다. C=1의 perplexity는 제시하지 않는다.",
    "suggested_fix": "'C=8에서도 36.45로 크게 악화되어 C=1 serving에 심각한 위험을 시사하지만, C=1 자체는 해당 실험에서 측정되지 않았다'로 수정한다."
  }
]