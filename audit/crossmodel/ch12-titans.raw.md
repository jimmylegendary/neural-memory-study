[
  {
    "severity": "major",
    "location": "§12.3.4, 식 (12-4)·(12-6), §12.4 표 12-2",
    "issue": "식 (12-4)와 본문은 η_t·β_t·α_t를 스칼라로 취급하지만, 식 (12-6)은 세 게이트의 diag 행렬을 사용한다. 벡터 게이트라면 (12-4)는 원소별/행별 작용을 명시해야 하고, 스칼라라면 diag가 불필요하다. 현재 서술은 게이트 차원과 실제 update를 일관되게 정의하지 않는다.",
    "evidence": "papers/2501.00663.txt: Appendix C Eqs. 32–33은 diag(1−α_t), diag(η_t), diag(θ_t)를 명시해 채널별 게이트 일반형을 사용한다.",
    "suggested_fix": "게이트를 스칼라 특수형으로 제한한다고 명시하거나, α_t·β_t·η_t의 벡터 차원과 W·S에 대한 원소별/행별 작용을 (12-4)부터 일관되게 정의한다. 표 12-2의 ‘스칼라 3개’도 이에 맞춘다."
  },
  {
    "severity": "minor",
    "location": "§12.3.7 식 (19), §12.3.8 MAC 입력 결합식",
    "issue": "통일 표기에서 P=[p_1 … p_Np]는 열벡터를 열로 쌓은 d×Np 행렬인데, 토큰을 행으로 쌓는 X∈R^{L×d}와 P∥X로 직접 연결했다. 같은 문제가 MAC의 P∥r∥X에도 있어 행렬 차원이 맞지 않는다.",
    "evidence": "papers/2501.00663.txt: §2는 입력을 x∈R^{N×d_in}인 행-토큰 행렬로 정의하고, §3.3 Eq. 19는 persistent token들을 시퀀스 행 방향으로 앞에 붙인다.",
    "suggested_fix": "시퀀스 결합에는 P_rows=[p_1^T;…;p_Np^T]∈R^{Np×d}를 사용해 X_new=P_rows∥X로 쓰거나, P를 행-쌓기 행렬로 재정의한다."
  },
  {
    "severity": "minor",
    "location": "§12.4 식 (12-8)",
    "issue": "식 (12-1)의 loss가 ||M(k)−v||²로 정의되어 있으므로 linear memory의 gradient는 2(Wk−v)k^T이다. 그런데 (12-8)의 우변에는 계수 2가 없어 등식이 성립하지 않는다.",
    "evidence": "papers/2501.00663.txt: §3.1 Eq. 12는 제곱 loss를 쓰지만, Appendix C는 linear 전개에서 1/2||Mk−v||²를 명시한다. Eq. 17 형태는 이 1/2 관행을 전제로 한 gradient다.",
    "suggested_fix": "식 (12-1)을 1/2||M(k;W)−v||²로 바꾸거나, 식 (12-8)의 우변에 계수 2를 추가한다."
  },
  {
    "severity": "major",
    "location": "§12.3.10 표 12-1, §12.3.8, §12.4",
    "issue": "inner optimizer의 mini-batch 크기 b와 MAC attention segment 크기 C를 같은 hyperparameter로 동일시했다. 원전에서는 전자는 memory-update 병렬화 단위이고 후자는 attention 문맥 분할 단위로 별도로 도입되며, 둘이 같다는 조건은 제시되지 않는다. 이 동일시는 attention 비용과 memory staleness를 불필요하게 결박한다.",
    "evidence": "papers/2501.00663.txt: §3.2는 memory training을 ‘chunks of size b’로 정의한다. §4.1은 별도로 sequence를 fixed-size segments S^(i), 개수 N/C로 나눈다.",
    "suggested_fix": "inner chunk 크기를 C_mem, MAC segment 크기를 C_seg처럼 분리하고, 실제 구현에서 둘을 같게 둔 경우에만 C_mem=C_seg라고 명시한다."
  },
  {
    "severity": "major",
    "location": "§12.3.8 MAC 데이터 흐름",
    "issue": "segment의 attention 출력 전체를 먼저 W_(n+1)C에 쓴 뒤, 모든 위치 τ의 최종 출력을 같은 segment-end 상태 W_(n+1)C에서 읽도록 적었다. 이를 문자 그대로 구현하면 앞쪽 위치가 뒤쪽 token으로 갱신된 memory를 읽어 causal leakage가 발생한다.",
    "evidence": "papers/2501.00663.txt: §3.2 Eq. 16은 chunk 안에서도 각 시간 t의 M_t를 prefix 누적으로 정의한다. §4.1 Eqs. 24–25의 압축 표기는 이 token별 causal state 계산을 제거한다는 근거가 아니다.",
    "suggested_fix": "각 위치 τ의 출력은 해당 위치까지의 prefix update로 얻은 W_τ에서 읽도록 쓰고, chunkwise 구현에서는 causal triangular/prefix dual form으로 모든 W_τ의 read를 계산한다고 명시한다."
  },
  {
    "severity": "major",
    "location": "§12.4 ‘inner loop: inference에서 실제로 움직이는 것’ 및 §12.8 한계 3",
    "issue": "훈련은 C>1에서 chunk 시작 상태에 고정된 stale gradient를 사용하면서 decode는 C=1 online update로 수행한다고 한 뒤 ‘update rule 자체에는 train/inference 불일치가 없다’고 주장한다. C는 계산되는 함수를 바꾸는 semantic hyperparameter이므로 두 궤적은 일반적으로 같지 않으며, §12.8도 동일한 정합 문제를 한계로 인정해 본문이 모순된다.",
    "evidence": "papers/2501.00663.txt: §3.2 Eq. 16과 Eq. 18은 gradient anchor M_t′가 chunk 크기 b에 의해 정해진다. b=1이면 매 step 최신 상태에서 평가하지만 b>1이면 chunk-start 상태에서 평가한다.",
    "suggested_fix": "decode에서도 훈련과 같은 chunk anchor를 순차적으로 유지하거나, C_train과 C_decode가 다르면 train/serve mismatch가 생긴다고 명시하고 ‘불일치가 없다’는 문장을 삭제한다."
  },
  {
    "severity": "minor",
    "location": "§12.4 per-token 비용 해설",
    "issue": "P_M을 parameter 수로 둘 때 약 5–6×P_M ‘FLOP’이라는 수치는 MAC/연산 수와 FLOP를 혼용해 약 2배 과소계상한다. dense forward는 약 2P_M FLOP, backward는 약 4P_M FLOP, read forward는 약 2P_M FLOP이고 상태 갱신과 projection 비용도 추가된다.",
    "evidence": "papers/2501.00663.txt: §3.1 Eqs. 11–15는 write forward, weight gradient, 상태 update, 별도 read 및 projections를 요구한다. §5.8/Fig. 9는 throughput만 보고하며 5–6P_M FLOP 수치를 제시하지 않는다.",
    "suggested_fix": "약 4P_M MAC과 별도 elementwise 연산이라고 표기하거나, FLOP 기준으로 대략 8–10P_M 이상 plus projections라고 고친다."
  },
  {
    "severity": "major",
    "location": "§12.6 Language modeling 및 S-NIAH 해석",
    "issue": "TTT와의 성능 차이를 momentum+forgetting의 값으로, Gated DeltaNet과의 차이를 deep nonlinear memory의 값으로 직접 등치했다. 이 비교들은 objective·architecture·parameterization이 동시에 다른 비통제 비교이므로 개별 성분의 인과 효과를 식별하지 못한다. S-NIAH 표 역시 해당 성분별 ablation이 아니다.",
    "evidence": "papers/2501.00663.txt: Appendix C의 TTT 비교는 ‘different architectural designs and objective functions’도 차이로 명시한다. §5.2는 중요성을 시사한다고 해석할 뿐, 성능 격차가 특정 성분의 순수 효과라는 통제 실험을 제시하지 않는다.",
    "suggested_fix": "‘해당 가설과 일관된 결과’로 한정하고, 개별 성분의 기여는 Table 5 ablation이 직접 측정한 범위에서만 주장한다."
  },
  {
    "severity": "minor",
    "location": "§12.7 표 12-4 뒤 문단",
    "issue": "momentum 때문에 LMM state가 Gated DeltaNet matrix state의 2배라고 했지만, 2배는 LMM의 memory weights 자체와 비교했을 때만 맞는다. 같은 폭의 Gated DeltaNet은 d×d 상태 하나인 반면 deep LMM은 약 L_M d²개의 weights와 같은 크기의 momentum을 가지므로 비율은 약 2L_M배이며, 표의 L_M=2 가정에서는 약 4배다.",
    "evidence": "papers/2501.00663.txt: §3.1은 deep memory를 L_M≥2 MLP로 정의하고 past-surprise S_t를 memory weights와 함께 유지한다. Appendix C Eq. 34의 Gated DeltaNet 상태는 단일 matrix S_t다.",
    "suggested_fix": "‘momentum이 LMM 자체의 fast-weight state를 2배로 만든다’로 고치고, Gated DeltaNet 대비 비율은 architecture와 hidden width를 명시해 별도로 계산한다."
  }
]