[
  {
    "severity": "critical",
    "location": "§10.5 식 (10-2), §10.6 Step 3–4",
    "issue": "chunkwise delta/TTT 계산의 intra-chunk 항 O(C²d)를 FLOPs에서 누락한 채 C→∞ 극한을 0.75d로 계산했다. 정확한 chunkwise 출력에는 O(Cd²+C²d)가 필요하므로 식 (10-2)는 C≪d인 제한적 근사일 뿐이며, C=d=64인 예제에서도 누락항은 지배항과 같은 차수다. 따라서 AI(64)=32, 0.75d ceiling, chunk를 아무리 키워도 ridge에 못 간다는 결론은 성립하지 않는다.",
    "evidence": "papers/external/2406.06484.txt §2.1은 exact chunkwise linear-attention 복잡도를 O(LCd+Ld²)로 제시한다(한 chunk당 O(C²d+Cd²)). papers/external/2407.04620.txt §2.5도 dual form에 출력 계산용 O(b²d)가 추가된다고 명시한다.",
    "suggested_fix": "FLOPs 분자에 구현별 계수 a를 둔 aC²d 항을 추가하고, C×C 중간값의 SRAM 유지/오프칩 spill에 따른 트래픽도 모델링하라. 현 식은 C≪d이고 intra-chunk 항을 무시하는 경우로 한정하며 C→∞ ceiling과 §10.6 수치를 다시 계산하라."
  },
  {
    "severity": "major",
    "location": "표 10-2 GLA / Mamba-2 행",
    "issue": "Mamba-2의 recurrent state와 decode 비용을 일반적으로 d² 및 5d²로 적었다. Mamba-2에는 입출력/head 차원 P와 독립적인 SSM state 차원 N이 있으므로 state는 P×N이며 비용도 PN에 비례한다. N=P를 가정하지 않는 한 d²가 아니다.",
    "evidence": "papers/external/2405.21060.txt §2.1 Definitions 2.1–2.2는 입력을 X∈R^(T,P), 단일 채널 hidden state를 h∈R^(T,N)로 정의하고 P개 채널에 broadcast한다고 설명한다. 이어 N을 자유로운 state-size 매개변수라고 명시한다.",
    "suggested_fix": "GLA와 Mamba-2를 별도 행으로 나누고 Mamba-2 state를 PN(다중 head이면 합계 Σ_h P_hN_h), decode FLOPs와 bytes도 PN 기준으로 작성하라."
  },
  {
    "severity": "major",
    "location": "표 10-2 Titans-LMM 행",
    "issue": "Titans의 retention·momentum·learning-rate gate를 chunk-상수라고 서술했지만 원전 실험은 token-dependent gate를 사용한다. chunk별 상수화는 원전이 미래의 속도 최적화 가능성으로만 제시한 미실험 단순화다.",
    "evidence": "papers/2501.00663.txt §3.2 'Parameters as the Function of Chunks' 부근: 원전은 α_t, θ_t, η_t를 chunk 함수로 만들 수 있다고 제안한 뒤, 실험에서는 이 매개변수들을 token의 함수로 사용했다고 명시한다.",
    "suggested_fix": "'token-dependent gate + momentum scan'으로 고치고, chunk-상수 gate는 별도의 미검증 최적화 선택지로만 언급하라."
  },
  {
    "severity": "major",
    "location": "표 10-2 Atlas 행, §10.2 및 요약 첫 bullet",
    "issue": "Atlas의 Newton–Schulz 비용을 단순한 'chunk당 κ회 소형 GEMM 상각분'으로 남기고 모든 고정-state decode가 bandwidth-bound라고 결론냈다. Atlas는 각 시점의 momentum 행렬에 Newton–Schulz를 적용하며, 각 반복은 행렬-행렬 곱을 포함해 행렬 폭에 대해 대략 3차 비용이 난다. C=1 decode에서는 상각할 chunk 축도 없어 compute-bound가 될 수 있다.",
    "evidence": "papers/2505.23735.txt Eq. (32)–(33)은 매 시점 M_t 갱신에 NewtonShulz-k(S_t)를 사용한다. §5.1 Eq. (40)–(41)은 모든 S_t에 Newton-Schulz5를 적용한 뒤 M_t를 갱신한다고 명시한다.",
    "suggested_fix": "memory 행렬별 Newton–Schulz FLOPs와 중간 트래픽을 명시적으로 더하고, Atlas/Muon 계열을 '예외 없이 bandwidth-bound'라는 결론에서 제외하라."
  },
  {
    "severity": "major",
    "location": "표 10-2 TNT 행의 decode FLOPs/token",
    "issue": "state와 트래픽 식은 N개 local memory를 반영하면서 FLOPs는 10P_M+4d²로 두어 N=1 비용만 적었다. 원전의 일반화된 TNT에서는 N개 local memory가 각각 update와 retrieval을 수행하고 N개의 Q-K projection도 계산한다.",
    "evidence": "papers/2511.07343.txt Appendix E Eq. (14)는 i=1,…,N 각각의 W_t^(i)를 갱신하고, Eq. (15)는 N개 local retrieval을 모두 합산한다.",
    "suggested_fix": "동일 크기 memory를 가정하면 대략 2P_global+N(8P_local+4d²)로 고치고, global write 비용은 C_g에 따라 별도로 상각하라."
  },
  {
    "severity": "major",
    "location": "표 10-2 TNT 행의 state 트래픽",
    "issue": "global memory까지 매 token full RMW하는 것으로 4(1+N)P_M bytes를 계산했다. 원전의 global V는 C_g-token chunk 경계에서만 갱신되고 token별 retrieval에서는 읽기만 하므로 global read와 amortized update를 분리해야 한다.",
    "evidence": "papers/2511.07343.txt §4.1.1 Eq. (5)는 V를 global chunk마다 한 번 갱신한다. §4.1.2 Eq. (7) 및 Appendix E Eq. (15)는 각 token에서 V를 retrieval에 사용한다.",
    "suggested_fix": "global retrieval read, C_g당 update RMW, local per-token RMW를 별도 항으로 분리하라. decode에서 global을 고정한다면 global write는 0으로 두고 그 serving 정책을 명시하라."
  },
  {
    "severity": "major",
    "location": "§10.4 항목 2 checkpoint/restore",
    "issue": "TNT의 periodic reset이 프롬프트 replay 분량 전체를 최대 L_s token으로 제한한다고 했지만 reset되는 것은 local memory와 local Q-K projection뿐이다. global memory는 전체 sequence를 따라 순차적으로 누적되므로 global snapshot이 없으면 전체 prefix를 다시 처리해야 한다.",
    "evidence": "papers/2511.07343.txt §4.1.1 Eq. (5)는 global V가 chunk 사이에서 계속 전달됨을 보인다. Eq. (6)과 Appendix C만 local W 및 projection M을 shard 경계에서 reset한다.",
    "suggested_fix": "replay 상한 L_s는 local state 복원에만 적용된다고 고치고, 전체 session 복원에는 global V snapshot 또는 전체 global-prefix replay가 필요하다고 명시하라."
  },
  {
    "severity": "major",
    "location": "§10.5 첫 문단",
    "issue": "chunk 크기가 semantic해지는 조건을 gradient가 '비선형 M을 통과하는' 경우라고 설명했다. TTT-Linear는 M이 선형이어도 gradient 평가점을 chunk 시작 W로 고정하므로 online GD와 다른 함수를 계산한다. 비선형성은 필요조건이 아니다.",
    "evidence": "papers/external/2407.04620.txt §2.4는 선형 f에 대해서도 mini-batch TTT를 G_t=∇ℓ(W_t′;x_t)로 정의하며 b가 속도-품질 trade-off를 제어한다고 명시한다. 같은 절 Fig. 7은 b에 따른 perplexity 변화를 보고한다.",
    "suggested_fix": "exact 여부를 memory 함수의 선형성으로 나누지 말고, 명시적 linear recurrence를 대수적으로 재배열한 linear attention/GLA/DeltaNet과 gradient anchor를 바꾸는 TTT-Linear/TTT-MLP/Titans를 구분하라."
  },
  {
    "severity": "minor",
    "location": "§10.5 TNT 성능 문단",
    "issue": "17.37× speedup과 ppl 23.09를 하나의 설정에서 동시에 얻은 결과처럼 결합했다. 17.37×는 Stage-1 TNT C_L={64}의 target-loss 시간이고, 23.09는 다른 다중-local Stage-2 설정 C_L={2,4,8,16}의 perplexity다.",
    "evidence": "papers/2511.07343.txt Table 1에서 17.37×는 TNT {64}; Table 2에서 avg ppl 23.09는 TNT Stage 2 {2,4,8,16}이다.",
    "suggested_fix": "두 수치가 서로 다른 설정에서 얻어진 별도 최고치임을 명시하거나, 동일 설정의 속도와 품질만 짝지어 비교하라."
  },
  {
    "severity": "minor",
    "location": "§10.5 'Stage 2(추가 ~5% compute)'",
    "issue": "Stage-2 비용을 일반적으로 약 5%라고 했지만 최종 최고 perplexity 설정의 추가 시간은 0.46/5.55≈8.3%다. 약 5%는 앞의 일부 구성에만 해당한다.",
    "evidence": "papers/2511.07343.txt Table 4: Stage-1 {4,8,16,32}=5.55시간, 대응 Stage-2 {2,4,8,16}=0.46시간. 다른 세 구성은 약 4.9–5.4%다.",
    "suggested_fix": "'구성에 따라 약 5–8%; 최고 perplexity 구성은 약 8.3%'로 고쳐라."
  }
]