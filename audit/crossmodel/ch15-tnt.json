[
  {
    "severity": "major",
    "location": "식 (15-4), 식 (15-5)",
    "issue": "1-based 인덱스로 바꾸면서 local-memory reset 경계가 잘못되었다. 식 (15-4)는 t≡0에서 이미 token t를 처리한 뒤의 상태 W_t를 W_init으로 덮어써 해당 token의 write를 누락시키며, projection은 t≡1에서 reset되어 두 상태가 한 token 어긋난다. 또한 shard 첫 token t=mL_s+1에서 식 (15-5)의 왼쪽은 Π_t=k_tk_t^T이지만 오른쪽은 이전 shard의 Π_{mL_s}+k_tk_t^T가 되어 서로 모순되고 이전 shard 정보가 누출된다.",
    "evidence": "papers/2511.07343.txt: §4.1.1 Eq.6 부근은 reset이 각 segment의 beginning에서 일어난다고 설명하고, App. C는 M_t를 t≡1 (mod S_L)에서 reset하며 shard boundary에서 carry-over state를 재초기화한다고 명시한다. §2.1 Eq.1–2에서는 W_t가 token t의 compression 후 retrieval에 쓰이는 상태다.",
    "suggested_fix": "shard 시작 직전의 별도 base state를 정의하라. t≡1 (mod L_s)인 첫 token에서도 W_init에서 출발해 그 token의 gradient update를 적용하고, Π의 carry는 0으로 둬야 한다. 일반 chunk 식에는 shard 첫 chunk일 때 W anchor=W_init, Π carry=0이라는 분기를 넣고 C_l이 L_s를 나눈다는 가정도 명시하라."
  },
  {
    "severity": "major",
    "location": "§15.2 Challenge 3, §15.3.5, §15.5 two-stage training 행, §15.7 State 크기 비교",
    "issue": "chunk-1 serving에 대한 실증 범위를 과장한다. baseline이 C=1에서 무너진다는 측정은 없고, TNT의 C=1이 품질 최적이라는 비교도 없다. Fig. 2의 최소 inference chunk는 8이며, Table 2에서 단일 local {1}은 대응 Stage 1 {8}보다 개선됐을 뿐 다른 단일 chunk와 비교되지 않았다. 전체 최고 평균 perplexity 23.09의 구성은 {2,4,8,16}으로 chunk 1을 포함하지 않는다.",
    "evidence": "papers/2511.07343.txt: Fig. 2/§3은 inference C∈{8,16,32,64,128,256,512}만 보고한다. §4.2는 C'_L=1을 'ideal inference scenario'로 제안하지만 최적성 실험은 제시하지 않는다. Table 2는 {1}=23.99, 최고 결과는 {2,4,8,16}=23.09로 보고한다.",
    "suggested_fix": "baseline에 대해서는 'C=8까지 크게 악화되므로 C=1에도 직접 전환하기 어렵다는 동기를 준다'로 제한하라. TNT에 대해서는 'Stage 2가 {8}→{1} 전환을 성공적으로 적응시켰다'고 쓰고, C=1이 전역 품질 최적이라는 표현은 삭제하라."
  },
  {
    "severity": "minor",
    "location": "§15.2 핵심 주장, §15.3.5 Stage 2, §15.5 two-stage training 행, §15.6 품질",
    "issue": "Stage 2 비용을 일괄적으로 pre-training의 약 5%라고 서술하지만 Table 4의 최종 4-local 구성은 0.46/5.55=8.29%다. 보고된 네 구성의 wall-clock 비율은 약 4.9%, 5.4%, 5.2%, 8.3%다.",
    "evidence": "papers/2511.07343.txt: Table 4 부근은 Stage 1 시간을 3.06, 4.24, 5.00, 5.55시간, 대응 Stage 2 시간을 0.15, 0.23, 0.26, 0.46시간으로 보고한다.",
    "suggested_fix": "'대부분 약 5%, 구성에 따라 약 4.9–8.3%'로 고치고, 최고 품질의 4-local 구성에는 약 8.3%가 든다고 명시하라."
  },
  {
    "severity": "minor",
    "location": "§15.5 Concept ledger delta의 retention 행",
    "issue": "reset-to-W_init을 단순히 α_t∈{0,1}인 hard retention gate로 표현할 수 없다. 책의 master update W_t=α_tW_{t-1}+S_t에서 α_t=0은 상태를 0으로 소거할 뿐 학습된 W_init으로 되돌리지 않는다.",
    "evidence": "papers/2511.07343.txt: §4.1.1 Eq.6은 reset 값을 0이 아니라 shared learnable state W_init으로 정의한다. style/STYLE-NOTATION.md: 식 (M)은 retention을 α_tW_{t-1}로 정의한다.",
    "suggested_fix": "reset을 W_t=α_tW_{t-1}+(1-α_t)W_init+S_t인 affine retention으로 쓰거나, 중심화 상태 W_t-W_init에 대해서만 α_t∈{0,1} gate라고 설명하라."
  },
  {
    "severity": "minor",
    "location": "§15.3.4 App. B two-pass read 수식",
    "issue": "책의 열벡터·matrix-memory 규약에서 W_t=W_{t-1}+φ_tv_t^T는 key-like feature φ를 value v에 대응시키는 write의 전치 방향이 틀렸다. 이후 W_t가 projected feature 열벡터에 왼쪽에서 곱해지므로 저장 update는 v_tφ_t^T여야 한다.",
    "evidence": "papers/2511.07343.txt: App. B Eq.11–12는 원 논문의 표기 관행으로 φ_tv_t^T를 사용한다. style/STYLE-NOTATION.md §1.3은 열벡터를 기본으로 하고 matrix memory W∈R^{d_v×d_k}, write=v_tk_t^T, read=Wq로 고정한다.",
    "suggested_fix": "통일 표기에서는 W_t=W_{t-1}+v_tφ_t^T로 전치해 쓰고, 원 식을 그대로 보일 경우에는 원 논문의 벡터 방향 관행임을 명시하라."
  },
  {
    "severity": "minor",
    "location": "§15.3.3 reset 설명, §15.3.6 두 번째 bullet, §15.7 Kernel 관점",
    "issue": "periodic reset을 비선형 deep-memory recurrence에 대해 '현재 알려진 유일한' sequence-direction 병렬화 수단이라고 단정한다. 원전은 문제를 largely unsolved라고만 표현하며, nonlinear sequential model의 sequence 병렬화를 다루는 기존 연구들도 직접 인용한다.",
    "evidence": "papers/2511.07343.txt: §1은 non-linear recurrence 병렬화가 'largely unsolved'라고 서술한다. References의 Gonzalez et al. 2024는 'Towards scalable and stable parallelization of nonlinear RNNs', Lim et al. 2024는 'Parallelizing non-linear sequential models over the sequence length'이다.",
    "suggested_fix": "'TNT가 제안하는 직접적이고 실용적인 병렬화 수단' 또는 '정확한 recurrence를 유지하는 대신 reset으로 의존성을 끊는 방법'으로 한정하고 유일성 주장은 삭제하라."
  }
]