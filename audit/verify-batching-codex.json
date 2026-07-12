[
  {
    "claim": "C1",
    "verdict": "imprecise",
    "reason": "의미론적으로는 맞다. 독립 request 상태를 선두 batch 차원으로 쌓아 BMM, strided/grouped GEMM, batched outer-product로 실행할 수 있다. 다만 attention과의 유사성은 상태 격리·스케줄링 측면에 한정된다. decode에서 서로 다른 W_b에 대한 W_b x_b는 사실상 batched GEMV이며 shared W GEMM처럼 batch가 커질수록 weight traffic이 상각되지 않는다. shared W에서는 FLOPs가 B에 비례하지만 W bytes는 거의 고정인 반면, per-request W_b에서는 FLOPs와 state bytes가 모두 B에 비례해 arithmetic intensity가 대체로 B와 무관하다. 단, 'HBM 재사용 0'과 '항상 memory-bound'는 너무 절대적이다. 같은 request의 chunk 내 여러 token, L2/SRAM 상주, kernel fusion에서는 재사용이 가능하다.",
    "evidence": "[2512.24695 §8.2](https://arxiv.org/pdf/2512.24695)는 request 축 batching을 직접 논하지 않지만 각 sequence가 독립적인 M□ 상태를 갖는 수식을 제시한다. HW상 y_b=W_bx_b는 표준 BMM/grouped-GEMM 의미론과 일치하나 shared-weight GEMM과 달리 request 간 weight reuse가 없다."
  },
  {
    "claim": "C2",
    "verdict": "imprecise",
    "reason": "식 자체는 원문의 선형 memory·dot-product objective 특수형과 일치하며 A_t=α_tI−η_tk_tk_tᵀ이다. 제시한 차원도 열벡터 관례에서 맞다. 그러나 이것은 실제 2-layer MLP memory 전체의 일반 update가 아니며, 원문이 사용하는 L2-regression형에는 추가 residual-gradient 항이 있다. 또한 A_t는 일반 dense matrix가 아니라 identity-minus-rank-one이므로 materialize하거나 M A를 dense BMM으로 계산하면 불필요하게 비싸다.",
    "evidence": "[2512.24695 Eq. 92–93](https://arxiv.org/pdf/2512.24695)는 dot-product형에 M_t=M_{t−1}(αI−ηkkᵀ)−ηv̂kᵀ를, L2형에는 −η(M_chunk k−v̂)kᵀ 항을 제시한다. 효율적인 구현은 M k와 outer product를 사용한다."
  },
  {
    "claim": "C3",
    "verdict": "correct",
    "reason": "원문은 memory 본체와 k, v, q, η, α 생성기를 모두 associative-memory module로 만들고 각자의 상태를 context에 따라 갱신한다. 따라서 serving에서 sequence/request별 상태로 분리해 동일 shape tensor로 쌓는 것은 수학적으로 타당하다.",
    "evidence": "[2512.24695 Eq. 83–88, 94–96](https://arxiv.org/pdf/2512.24695)는 □∈{k,v,q,η,α,memory} 각각에 M□,t를 정의하고 갱신한다. α는 retention/forget/weight-decay gate로 설명된다."
  },
  {
    "claim": "C4",
    "verdict": "correct",
    "reason": "원문의 실험 구성은 모든 해당 memory module을 residual 2-layer MLP M□(x)=x+W□,1σ(W□,2x)로 둔다. 열벡터 관례에서 W2:[H,D], W1:[D,H]이며 request 축을 붙인 [B,H,D], [B,D,H] BMM 표현이 맞다.",
    "evidence": "[2512.24695 Eq. 89 및 91](https://arxiv.org/pdf/2512.24695)에 정확히 residual 2-layer MLP가 명시되어 있다."
  },
  {
    "claim": "C5",
    "verdict": "imprecise",
    "reason": "chunk 경계의 상태 의존성 때문에 전체 sequence를 한 번에 독립 계산할 수 없다는 결론은 맞다. 원문의 dual/chunk-wise form은 이전 chunk의 마지막 상태를 기준으로 현재 chunk의 생성값과 gradient를 병렬 계산한다. 다만 chunk 내부 token 병렬성은 training·prefill처럼 chunk의 token들이 이미 주어진 경우에 해당한다. 일반 autoregressive decode에서는 미래 token이 아직 없으므로 batch 축 외의 이 병렬성을 그대로 이용할 수 없다.",
    "evidence": "[2512.24695 §8.2, Eq. 90](https://arxiv.org/pdf/2512.24695)는 다음 chunk를 이전 chunk의 마지막 상태에서 계산한다고 명시한다. [2501.00663 §3.2](https://arxiv.org/pdf/2501.00663)도 chunk-wise inner update를 사용한다."
  },
  {
    "claim": "C6",
    "verdict": "imprecise",
    "reason": "matmul+sum 재구성과 momentum associative scan은 맞지만 적용 범위를 분리해야 한다. Titans의 nonlinear memory recurrence 전체가 associative scan 가능한 것은 아니다. chunk gradient/mini-batch update는 matmul과 sum으로 tensorize하고, 그 안의 선형 momentum recurrence S_t=η_tS_{t−1}−θ_tu_t만 scan한다. NL의 절 제목이 실제로 'Fast and Parallelizable Training'인 것도 맞지만 serving continuous batching을 입증하는 표현은 아니다.",
    "evidence": "[2501.00663 §3.2, Eq. 16–18](https://arxiv.org/pdf/2501.00663)는 mini-batch update를 matmul+sum으로 만들고 momentum에만 parallel associative scan을 적용한다. [2512.24695 §8.2](https://arxiv.org/pdf/2512.24695)는 'Fast and Parallelizable Training'과 chunk-wise dual form을 명시한다."
  },
  {
    "claim": "C7",
    "verdict": "correct",
    "reason": "서로 다른 serving request의 fast state는 독립 trajectory이므로 gradient나 update를 request batch 방향으로 평균하면 모델 의미가 바뀌고 사용자 간 정보가 섞인다. batching은 독립 update를 벡터화하는 것이지 reduction하는 것이 아니다. 반면 공유 초기값·meta-parameter를 학습하는 outer loop에서는 training examples에 대한 gradient 평균이 정상적이다. 단, 논문의 'mini-batch GD'는 request 평균이 아니라 주로 한 sequence 내부 chunk의 token update를 뜻한다.",
    "evidence": "[2501.00663 §3.1](https://arxiv.org/pdf/2501.00663)은 inner loop에서 M의 weights를 갱신하고 outer loop에서 나머지 공유 parameter를 최적화한다고 구분한다. 독립 상태 M_b에 대한 update는 수학적으로 batch reduction이 없는 map 연산이다."
  },
  {
    "claim": "C8",
    "verdict": "correct",
    "reason": "동일 shape의 독립 연산은 각각 batched kernel 한 번으로 dispatch할 수 있지만 여러 단계로 구성된 self-modifying block 전체가 자동으로 하나의 fused kernel이 되는 것은 아니다. 논문은 training 병렬화와 throughput을 보일 뿐, ragged decode, state paging, admission/eviction을 포함한 vLLM급 continuous-serving 구현을 공개했다는 근거는 제공하지 않는다.",
    "evidence": "[2512.24695 §8.2](https://arxiv.org/pdf/2512.24695)는 dual/chunk 병렬 계산을 설명하지만 serving runtime이나 단일 fused kernel을 제시하지 않는다. [2501.00663](https://arxiv.org/pdf/2501.00663)의 효율성 주장도 fast parallelizable training/inference 수준이다."
  },
  {
    "claim": "C9",
    "verdict": "imprecise",
    "reason": "상태 종류와 mutation 관리가 attention KV cache보다 복잡하다는 평가는 타당하며 paged allocator와 유사한 관리 계층이 유용할 수 있다. 다만 '더 크다'까지 일반화할 수는 없다. KV cache는 context length에 비례하지만 fast-weight state는 보통 고정 크기이므로 길이에 따라 크기 우열이 바뀐다. 또한 완전한 Hope에는 열거한 여섯 self-referential memory뿐 아니라 CMS의 다중-frequency mutable blocks와 선택한 optimizer의 momentum state도 고려해야 한다.",
    "evidence": "[2512.24695 Eq. 94–97](https://arxiv.org/pdf/2512.24695)는 여섯 M□ 상태와 후속 CMS chain을 정의하며, §8.2는 memory 본체와 나머지 memories에 서로 다른 chunk size도 사용한다고 밝힌다."
  },
  {
    "claim": "C10",
    "verdict": "imprecise",
    "reason": "TTT layer, batched fast-weight optimizer, recurrent/chunk execution의 결합이라는 요약은 적절하다. 다만 MoE와의 비유는 kernel dispatch 형태에만 가깝다. MoE는 여러 token이 소수의 shared expert weights를 재사용하지만 여기서는 request마다 weight가 달라 decode 시 group당 token 수가 사실상 1일 수 있어 MoE의 expert-weight 상각 이득이 없다.",
    "evidence": "[2512.24695 §8.1–8.2](https://arxiv.org/pdf/2512.24695)는 request-local 2-layer MLP weights와 DGD update를 결합한다. HW 관점에서는 grouped GEMM으로 표현 가능하지만 shared-expert MoE와 데이터 재사용 구조가 다르다."
  },
  {
    "claim": "overall",
    "verdict": "imprecise",
    "reason": "핵심 결론은 맞다. Self-Modifying Titans/Hope는 request-local fast-weight 상태를 batch 차원으로 tensorize해 serving batch로 실행할 수 있으며, request 간 update를 평균해서는 안 된다. 동시에 shared-weight dense layer가 얻는 batch-size 비례 weight 상각은 얻지 못해 one-token decode가 bandwidth-bound일 가능성이 높다는 구분도 정확하다. 주요 과장은 BMM 가능성을 attention과 동일한 효율로 해석하는 것, HBM reuse를 절대적으로 0이라 하는 것, memory-bound를 필연으로 단정하는 것, dot-product 선형 특수식을 전체 2-layer L2 Hope update처럼 제시하는 것, 그리고 training/prefill의 chunk 병렬성을 autoregressive serving decode에도 그대로 적용하는 것이다.",
    "evidence": "[Nested Learning 원문 §8, Eq. 83–97](https://arxiv.org/pdf/2512.24695)과 [Titans 원문 §3.1–3.2](https://arxiv.org/pdf/2501.00663)를 함께 보면 semantic batchability와 shared-weight reuse는 서로 다른 문제다. 전자는 가능하지만 후자는 request-local weights 때문에 batch B로 개선되지 않으며, 개선 요인은 chunk 길이, cache residency, fusion 등에 달려 있다."
  }
]