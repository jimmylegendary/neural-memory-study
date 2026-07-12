[
  {
    "severity": "major",
    "location": "§2.5 optimizer 객체의 update 서명",
    "issue": "`update(state_t, g_t)`만으로는 이 장의 AdamW를 계산할 수 없다.",
    "evidence": "Decoupled weight decay는 현재 파라미터 Θ_t에 의존하고, Adam의 bias correction은 step t에 의존한다. 따라서 같은 m_t, h_t, g_t라도 Θ_t나 t가 다르면 결과가 달라진다.",
    "suggested_fix": "서명을 `update(Θ_t, state_t, g_t, t) → (Θ_{t+1}, state_{t+1})`로 바꾸거나, Θ_t는 명시적 입력으로 두고 step counter를 optimizer state에 포함하라."
  },
  {
    "severity": "major",
    "location": "§2.1 training loop 4번·표 2-1",
    "issue": "Optimizer step 전체를 GEMM 없는 순수 elementwise pass라고 일반화한 것은 장 후반의 Shampoo·Muon 설명과 모순된다.",
    "evidence": "Shampoo는 통계 갱신과 preconditioning에 GEMM 및 행렬 inverse root를 사용하고, Muon은 Newton–Schulz 반복에서 여러 GEMM을 수행한다. 따라서 이 명제는 SGD·momentum·AdamW·AdaGrad 같은 좌표별 optimizer에만 성립한다.",
    "suggested_fix": "`일반적인 AdamW 계열 optimizer step`으로 범위를 제한하고, Shampoo와 Muon은 GEMM 기반 예외라고 같은 위치에서 명시하라."
  },
  {
    "severity": "major",
    "location": "§2.5.5 AdaGrad 설명",
    "issue": "\"Adam과의 차이는 decay가 없다는 것 하나\"라는 설명은 틀렸다.",
    "evidence": "AdaGrad는 현재 gradient g_t를 분자로 쓰고 제곱 gradient를 누적하지만, Adam은 first-moment EMA m_t를 분자로 쓰며 second-moment EMA와 bias correction도 사용한다. 즉 forgetting 유무 외에도 first-moment momentum과 bias correction이 다르다. [Adam 원 논문](https://arxiv.org/abs/1412.6980), [AdaGrad 원 논문](https://jmlr.org/papers/v12/duchi11a.html)",
    "suggested_fix": "AdaGrad는 first-moment buffer 없이 현재 gradient를 사용하고, second moment를 EMA가 아닌 누적으로 유지한다고 설명하라."
  },
  {
    "severity": "major",
    "location": "§2.5.6 Muon의 Newton–Schulz 설명·식 (2-5)",
    "issue": "Muon의 튜닝된 quintic 반복이 특이값을 1로 수렴시켜 정확한 semi-orthogonal projection을 만든다고 서술한 것은 틀렸다.",
    "evidence": "계수 (3.4445, -4.7750, 2.0315)는 빠른 초기 증폭을 위해 정확한 1 수렴을 포기한 계수다. 원 설명은 점근 특이값을 대략 [0.7, 1.3] 범위에 두며, 공식 구현도 결과가 UVᵀ가 아니라고 명시한다. [Muon 원 설명](https://kellerjordan.github.io/posts/muon/), [공식 구현](https://github.com/KellerJordan/Muon/blob/master/muon.py)",
    "suggested_fix": "UVᵀ는 이상적인 polar factor라고 구분하고, 실제 5-step tuned NS는 특이벡터를 보존하면서 특이값을 1 근방으로 모으는 근사라고 고쳐라."
  },
  {
    "severity": "minor",
    "location": "§2.2 Taylor 근사 직후",
    "issue": "\"내적을 가장 빠르게 음수로 만드는 선택\"이 무조건 −ηg라는 진술에는 크기 제약이 빠져 있다.",
    "evidence": "ΔΘ의 크기를 제한하지 않으면 ⟨g, ΔΘ⟩은 ΔΘ=−cg, c→∞로 하한 없이 감소한다. −ηg가 steepest-descent 방향이 되는 것은 Euclidean norm 제약이나 quadratic step penalty가 있을 때다.",
    "suggested_fix": "\"고정된 Euclidean step 크기에서\" 또는 \"(1/2η)||ΔΘ||² penalty 아래에서\"라는 조건을 추가하라."
  },
  {
    "severity": "minor",
    "location": "§2.5.3 식 (2-3)",
    "issue": "Weight-decay 점화식을 전개하면서 초기 파라미터 항을 누락했다.",
    "evidence": "q=1−ηλ이고 Θ_t=qΘ_{t−1}+u_t이면 정확한 전개는 Θ_t=q^tΘ_0+Σ_{i=1}^t q^{t−i}u_i다. 본문 식은 Θ_0=0을 가정하지 않으면서 첫 항을 생략한다.",
    "suggested_fix": "식에 `(1−ηλ)^t Θ_0`를 추가하거나 Θ_0=0이라는 가정을 명시하라."
  },
  {
    "severity": "minor",
    "location": "§2.5.2 momentum의 기억 길이",
    "issue": "β=0.9에서 최근 gradient 6개가 50% 이상, 43개가 99% 이상을 차지한다는 수치가 각각 하나씩 부족하다.",
    "evidence": "최근 k개가 차지하는 정규화된 질량은 1−0.9^k다. k=6이면 46.86%, k=43이면 98.92%이며, 임계값을 넘는 최소 개수는 각각 7개와 44개다.",
    "suggested_fix": "`최근 7개에 50% 이상, 최근 44개에 99% 이상`으로 고쳐라."
  },
  {
    "severity": "minor",
    "location": "표 2-2·표 2-3 Shampoo state 크기",
    "issue": "fp32 byte 단위 표에서 Shampoo state를 m²+n²로 적어 4배의 단위 오류가 있다.",
    "evidence": "L_t와 R_t에는 각각 m², n²개의 fp32 원소가 있으므로 두 통계의 저장량은 4(m²+n²) byte다. [Shampoo 원 논문](https://proceedings.mlr.press/v80/gupta18a/gupta18a.pdf)",
    "suggested_fix": "byte 열에서는 `4(m²+n²) B/행렬`로 쓰거나, 열 단위를 `fp32 원소 수`로 변경하라."
  },
  {
    "severity": "minor",
    "location": "§2.5.4 Adam cost",
    "issue": "Adam의 step 트래픽이 SGD의 약 3배라는 계산은 본문 자체의 트래픽 모델과 맞지 않는다.",
    "evidence": "모두 fp32라면 본문 기준 SGD는 Θ read+g read+Θ write=12 B/param이고, Adam은 여기에 m,h read+write 16 B가 추가되어 28 B/param, 즉 약 2.33배다. §2.8의 24–28 B 추정도 3배를 지지하지 않는다.",
    "suggested_fix": "`약 2–2.3×`로 수정하거나 어떤 텐서와 dtype을 세는지 별도로 정의하라."
  },
  {
    "severity": "minor",
    "location": "표 2-2·§2.5.6·표 2-3 Muon GEMM 비용",
    "issue": "본문에 제시한 quintic Newton–Schulz 구현에서 κ=5의 비용을 10–15 GEMM이라고 한 것은 부정확하다.",
    "evidence": "각 반복은 X Xᵀ, A A, B X의 행렬곱 3개를 요구하므로 제시된 quintic 식을 5회 수행하면 논리적 GEMM 수는 15개다. [Muon 원 구현](https://kellerjordan.github.io/posts/muon/)",
    "suggested_fix": "제시된 quintic 구현의 비용을 `3κ=15 GEMM`으로 쓰고, 10회가 가능한 별도의 cubic/근사 구현을 뜻한다면 이를 명시적으로 분리하라."
  },
  {
    "severity": "minor",
    "location": "§2.7(d) Adam 첫 step 예제",
    "issue": "ε를 포함한 식 (2-4)를 사용하면서 첫 update가 정확히 −η sign(g)이고 두 좌표의 크기가 정확히 η라고 계산했다.",
    "evidence": "실제 크기는 η|g|/(|g|+ε)이므로 g=−2와 g=−4에서 각각 2η/(2+ε), 4η/(4+ε)다. ε>0이면 두 값은 서로 다르고 η와도 정확히 같지 않다.",
    "suggested_fix": "예제에서 ε=0을 명시하거나 `ε가 매우 작을 때 근사적으로 −η sign(g)`라고 고쳐라."
  }
]