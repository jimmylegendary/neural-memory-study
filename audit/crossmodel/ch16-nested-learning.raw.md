[
  {
    "severity": "major",
    "location": "§16.3.3 예제 (c) — linear attention + GD",
    "issue": "projection의 outer 최적화와 inner memory recurrence 사이에 backpropagation이 없다고 단정하지만, 표준 end-to-end linear attention에서는 task loss의 gradient가 M_t recurrence를 거쳐 W_K와 W_V로 흐른다. stop-gradient를 명시하지 않는 한 두 gradient flow가 서로 관통하지 않는다는 설명은 성립하지 않는다.",
    "evidence": "papers/2512.24695.txt: §3.1 Eq. 14–16은 k_t=W_Kx_t, v_t=W_Vx_t, M_t=M_{t-1}+v_tk_t^T, y_t=M_tq_t로 정의한다. 따라서 연쇄법칙상 ∂L/∂W_K와 ∂L/∂W_V는 M_t를 통과한다. 같은 절의 'no backpropagation through it'이라는 산문은 이 수식과 충돌한다.",
    "suggested_fix": "outer loop에서는 M_t를 독립 parameter로 업데이트하지 않을 뿐, task-loss gradient는 recurrence를 미분해 projection으로 흐른다고 수정한다. 실제 stop-gradient 변형을 뜻한다면 이를 명시하고 일반적인 linear attention 훈련과 구분한다."
  },
  {
    "severity": "major",
    "location": "§16.3.5 Muon 유도, [NL Eq. 43–44] 대응 수식",
    "issue": "직교화 objective ||O^TO-I||_F^2만을 미분해 O-g+2O(O^TO-I)가 나온다고 설명한다. 해당 objective의 gradient에는 O-g 항이 없으므로 제시된 update와 Newton–Schulz 회수 주장이 유도되지 않는다.",
    "evidence": "papers/2512.24695.txt: §4.2 Eq. 43은 L̃=||P(g)^TP(g)-I||_F^2만 정의하지만 Eq. 44에는 O_i-g_t 항이 추가되어 있다. O-g는 별도의 proximity loss에서만 생긴다.",
    "suggested_fix": "objective를 1/2||O-g||_F^2+1/2||O^TO-I||_F^2처럼 수정해 Eq. 44를 유도하거나, Eq. 43만 유지할 경우 O-g 항을 제거하고 Newton–Schulz와의 관계를 다시 서술한다."
  },
  {
    "severity": "major",
    "location": "§16.3.6 식 (16-3)",
    "issue": "proximal objective의 계수와 Sherman–Morrison 역행렬, 최종 step size가 서로 맞지 않는다. 식에 적힌 1/(2η_t) proximal 항과 ||x_t||=λ 조건에서 정확한 계수는 η_t/(1+η_tλ²)이며, η_t/(1+η_t)는 λ=1일 때만 성립한다. 또한 역행렬 대상은 x_tx_t^T+η_t^{-1}I여야 한다.",
    "evidence": "papers/2512.24695.txt: §4.5 Eq. 56과 App. C Eq. 113은 proximal 계수를 1/(2η_t)로 둔다. 같은 Appendix Eq. 117은 ||x_t||=λ에 따른 λ² 의존성을 드러낸다. 장의 η'_t=η_t/(1+η_t)는 일반 λ 조건과 양립하지 않는다.",
    "suggested_fix": "η'_t를 η_t/(1+η_t||x_t||²)로 고치고 역행렬을 (x_tx_t^T+η_t^{-1}I)^{-1}로 유도한다. η_t/(1+η_t)를 유지하려면 ||x_t||=1을 명시한다."
  },
  {
    "severity": "major",
    "location": "§16.3.6 GGD 정의식",
    "issue": "argmin 변수 W가 내부 loss L̃(x_t,u_t)에 나타나지 않아, 현재 식에서는 association loss가 최적해에 아무 영향도 주지 않는다. 따라서 'x_t를 self-generated value에 사상하는 memory'라는 정의를 수식이 표현하지 못한다.",
    "evidence": "papers/2512.24695.txt: §4.5 Def. 5 Eq. 59도 L̃(x_t,u_t)로 인쇄되어 있지만, 바로 뒤 산문은 L̃가 mapping의 품질을 측정한다고 설명한다. 그 의미대로라면 memory parameter W에 대한 의존성이 필요하다.",
    "suggested_fix": "L̃(W;x_t,u_t), L̃(Wx_t,u_t), 또는 L̃(M_W(x_t),u_t)처럼 argmin 변수에 대한 의존성을 명시한다."
  },
  {
    "severity": "major",
    "location": "§16.3.5 Adam 단락",
    "issue": "제시된 additive moment recurrence에서 'Adam이 그대로 나온다'고 단정한다. 그러나 h_{i+1}=h_i+β₂g_{i+1}²와 m̃_{i+1}=m̃_i+β₁g_{i+1}은 Adam의 EMA가 아니며 bias correction도 없다. 이는 누적 제곱합을 쓰는 AdaGrad 계열에 더 가깝고, 원전도 Eq. 105에서 근사 기호를 사용한다.",
    "evidence": "papers/2512.24695.txt: App. B Eq. 102는 두 통계를 additive accumulator로 정의하고, Eq. 105는 근사식 '≈'을 거쳐 Adam과 동치라고 주장한다. 표준 Adam에 필요한 βm_{i-1}+(1-β)g_i 형태와 bias correction은 이 유도에 없다.",
    "suggested_fix": "'Adam과 유사한 정규화 update를 근사적으로 회수한다'로 약화하거나, 실제 Adam EMA와 bias correction을 포함하는 objective 및 유도를 새로 제시한다."
  },
  {
    "severity": "major",
    "location": "§16.3.10 식 (16-4), §16.4 식 (16-5)",
    "issue": "모든 memory가 2-layer residual MLP라고 한 뒤 generic W_□에 (α_tI-η_tk_tk_t^T)를 우측 곱한다. MLP parameter는 두 행렬의 tuple이며 각 행렬의 차원도 달라 이 연산은 일반적으로 정의되지 않는다. 식은 matrix-valued linear memory에만 타입이 맞는다.",
    "evidence": "papers/2512.24695.txt: §8.1 Eq. 88은 M_□에 행렬 우측 곱을 쓰지만, 바로 다음 Eq. 89는 M_□(z)=z+W_{□,1}σ(W_{□,2}z)인 2-layer MLP로 정의한다. §8.2 Eq. 92–93에서야 별도로 'matrix-valued memory' 특수 사례를 유도한다.",
    "suggested_fix": "식 (16-4)와 (16-5)를 linear matrix memory 특수 사례로 제한한다. deep memory에는 parameter tuple Θ_□에 대한 명시적인 layer별 weight decay 및 gradient update를 별도로 적고, 구현이 확인되지 않으면 적용 행렬을 단정하지 않는다."
  },
  {
    "severity": "major",
    "location": "표 16-3, §16.4.2, §16.7 decode 비용·RMW 산수",
    "issue": "chunkwise 병렬 훈련을 autoregressive serving에도 그대로 적용해 fast weights가 C token마다만 갱신되고 update 비용이 1/C로 상환된다고 계산한다. 원전은 이 절차를 훈련 병렬화로만 제시하며, Eq. 90의 memory 상태는 여전히 매 t마다 recurrence로 변한다. decode에서는 미래 chunk의 token을 미리 알 수 없으므로 chunk 전체 gradient를 한 번에 계산할 수도 없다.",
    "evidence": "papers/2512.24695.txt: §8.2 제목은 'Fast and Parallelizable Training'이며, 다음 chunk의 요소와 gradient를 직전 chunk state에서 미리 병렬 계산한다고 설명한다. Eq. 90은 M_{□,t}=M_{□,t-1}(…)-… 형태의 per-token recurrence다. inference에서 동일 batch schedule이나 1/C 상환을 사용한다는 설명은 없다.",
    "suggested_fix": "chunkwise 비용 주장을 training/prefill로 제한한다. decode는 순차 per-token update, 경계 지연 update 등 가능한 구현을 구분하고, 공개 구현이나 측정이 없으면 1/C RMW 및 latency 산수를 미확정으로 남긴다."
  },
  {
    "severity": "major",
    "location": "§16.3.10, 표 16-3, §16.4.2",
    "issue": "q projection의 적응 여부를 서로 다르게 기술한다. §16.3.10은 최종 설계에서 q가 정적이라고 결론내리고 표 16-3도 fast-state 집합에서 q를 제외하지만, §16.4.2와 §16.7은 M_q를 포함한 여섯 fast MLP를 전제로 state와 비용을 계산한다.",
    "evidence": "papers/2512.24695.txt: §8.1 Eq. 83–85는 q_t=x_tW_q를 'only non-adaptive projection'이라고 명시한다. 반면 Eq. 88·96은 □ 집합에 q를 포함하고, Table 6에는 'w/o inner-projection q' ablation이 있다. 원전 자체가 상충하므로 한쪽을 확정할 근거가 없다.",
    "suggested_fix": "원전의 불일치를 명시한 채 구현 확인 전에는 fast memory 수를 확정하지 않는다. 정적 q 해석을 채택하면 전 장의 수를 5개 memory와 약 10개 행렬로 고치고, 적응형 q 해석이면 표 16-3과 '최종 설계는 정적'이라는 결론을 수정한다."
  },
  {
    "severity": "minor",
    "location": "§16.3.5 momentum capacity 산수",
    "issue": "β=0.9에서 마지막 6개 gradient가 최소 50%, 마지막 43개가 최소 99%를 차지한다고 했지만 무한 EMA 기준 합은 각각 1-0.9^6=46.86%, 1-0.9^43=98.92%다. 임계치를 처음 넘는 개수는 7개와 44개다.",
    "evidence": "papers/2512.24695.txt: §4.3은 각 과거 gradient의 가중치를 β^i(1-β)로 정의한 뒤 6/43개라고 주장한다. 해당 기하급수 합을 직접 계산하면 그 주장과 맞지 않는다.",
    "suggested_fix": "수치를 '마지막 7개가 50% 이상, 마지막 44개가 99% 이상'으로 수정한다."
  },
  {
    "severity": "minor",
    "location": "§16.6 in-context recall 단락",
    "issue": "Atlas retrieval gap을 53.6 대 43.7로 적어 원전 Table 5의 정밀 수치 53.55 대 43.70을 변형했다.",
    "evidence": "papers/2505.23735.txt: Table 5의 Average 열은 Transformers 53.55, Atlas 43.70이다.",
    "suggested_fix": "53.55 대 43.70으로 수정한다."
  }
]