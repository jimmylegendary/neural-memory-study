[
  {
    "severity": "critical",
    "location": "§14.3.1 Theorem 1",
    "issue": "선형독립인 key가 R^{d_k}에 있다는 정의 아래에서는 항상 m≤d_k이므로, 최소 O(d_k d_v)개의 쌍을 저장한다는 정리와 이를 근거로 한 'depth가 capacity 자체를 올린다'는 결론은 성립하지 않는다. 또한 AK=V에서 rank(V)≤rank(A)는 나오지만 value들의 선형독립을 가정하지 않았으므로 m≤rank(A)는 나오지 않는다.",
    "evidence": "papers/2505.23735.txt:552-584의 Prop. 1과 Thm. 1은 모두 선형독립 key를 전제한다. App. C 부근 2753-2847은 rank(K)=m이면서 m≤rank(A)를 도출하지만 이는 논리적으로 따라오지 않으며, 같은 부록의 Prop. 1 증명(2734-2752)은 선형독립성만으로 m≤d_k임을 이미 보인다.",
    "suggested_fix": "Theorem 1을 검증된 정리처럼 유도하지 말고 원전 자체의 모순·증명 결함으로 명시하라. O(d_k d_v) 하한과 depth-capacity 결론은 삭제하거나, key 선형독립 조건을 제거한 올바른 capacity 정의와 유효한 별도 정리가 있을 때만 다시 제시하라."
  },
  {
    "severity": "minor",
    "location": "§14.3.1 Proposition 1 직후",
    "issue": "capacity가 memory 파라미터 수에 대해 무조건 sub-linear라는 설명은 성립하지 않는다. 파라미터 수 P=d_kd_v이고 capacity가 O(d_k)=O(P/d_v)이므로, d_v를 고정하면 P에 대해 선형이다.",
    "evidence": "papers/2505.23735.txt:552-563은 파라미터 수를 d_kd_v, capacity를 O(d_k)로 둔다. 이 두 식만으로 P에 대한 sub-linear 관계는 나오지 않는다.",
    "suggested_fix": "d_k와 d_v가 함께 증가하는 스케일링을 가정할 때만 sub-linear라고 한정하라. 예를 들어 d_v=Θ(d_k)이면 capacity가 O(√P)라고 명시할 수 있다."
  },
  {
    "severity": "minor",
    "location": "§14.3.1 Proposition 1의 GD 수렴 설명",
    "issue": "정의한 loss가 ||WK−V||_F^2이므로 실제 gradient에는 계수 2가 붙는다. 이 정의 그대로라면 안정적인 step-size 상한은 η<1/λ_max(KK^T)이며, η<2/λ_max는 loss를 1/2||WK−V||_F^2로 정의했을 때의 값이다. 또한 minimum-norm 해로의 수렴에는 보통 0 초기화 조건이 필요하다.",
    "evidence": "papers/2505.23735.txt:2734-2760은 계수 1/2 없는 loss를 쓰면서 gradient의 2를 생략하고 0<η<2/λ_max를 제시한다. 임의 초기화의 null-space 성분이 보존된다는 조건도 다루지 않는다.",
    "suggested_fix": "loss를 1/2||WK−V||_F^2로 고치거나 step-size 상한을 η<1/λ_max로 고쳐라. minimum-norm 수렴에는 W_0=0 같은 초기화 조건을 추가하라."
  },
  {
    "severity": "major",
    "location": "§14.3.2 global 최적화의 효율 설명",
    "issue": "global objective를 최적화하면 반드시 모든 과거 key·value를 cache해야 한다는 단정은 거짓이다. 바로 뒤에서 언급한 선형 least-squares는 Sherman–Morrison 기반 recursive least squares로 충분통계와 inverse state만 유지해 정확히 갱신할 수 있다.",
    "evidence": "papers/2505.23735.txt:675-689는 global objective를 제시하고, 792-803은 Mesa-layer가 Sherman–Morrison으로 inverse를 재귀 계산한다고 설명한다. 이 특수 경우에는 전체 KV cache가 필요하지 않다.",
    "suggested_fix": "전체 KV cache 필요성은 일반적인 비선형 memory나 충분통계가 없는 objective에 한정하라. 선형 least-squares의 실제 결격 사유는 순차적 inverse update, 추가 O(d_k^2) state, 선형 memory 한정이라고 서술하라."
  },
  {
    "severity": "minor",
    "location": "§14.3.2 식 (14-1) 뒤 window gate 설명",
    "issue": "global objective에 gate를 달면 prefix 길이만큼 '파라미터'도 증가한다는 주장은 일반적으로 성립하지 않는다. 공유 gate-producer는 고정된 파라미터로 임의 길이 prefix의 gate 값들을 만들 수 있으며, 증가하는 것은 gate activation·연산·저장량이다.",
    "evidence": "papers/2505.23735.txt:712-720은 γ가 input-dependent라고만 규정한다. 공유 함수가 gate를 생성하지 못한다는 제약이나 prefix 길이에 비례하는 독립 학습 파라미터는 정의하지 않는다.",
    "suggested_fix": "'파라미터·메모리가 함께 자란다'를 'gate 값의 수와 이를 계산·저장하는 비용이 문맥 길이에 따라 자란다'로 고쳐라."
  },
  {
    "severity": "major",
    "location": "§14.3.4 식 (14-4) 뒤 원문 대조 경고",
    "issue": "Table 1, Eq. 32–33, App. D.4의 recurrence가 모두 같은 알고리즘 의미라는 설명은 틀렸다. Table 1은 S_t에 −∇ℓ를 누적하면서 W_t에도 −NS(S_t)를 적용하므로 두 부호가 결합되어 loss ascent 방향이 된다. 반면 Eq. 32–33의 +∇/−NS와 App. D.4의 −∇/+NS만 서로 동등한 descent 관행이다.",
    "evidence": "papers/2505.23735.txt:315-326의 Table 1 Atlas 행은 두 항 모두 음의 부호다. 1173-1184의 Eq. 32–33은 S에 양의 gradient, W에 음의 NS를 쓰고, 3050-3061의 Eq. 57–58은 그 반대 부호 쌍을 쓴다.",
    "suggested_fix": "Table 1의 Atlas 행을 단순한 표기 관행이 아니라 부호 오탈자로 명시하라. 식 (14-4)는 App. D.4 관행을 따른다고 밝히고, 동등성은 Eq. 32–33과 App. D.4 사이에만 주장하라."
  },
  {
    "severity": "major",
    "location": "§14.3.5 DLA 설명",
    "issue": "DLA가 양의 dot-product loss ℓ=⟨M(φ(k)),v⟩를 gradient descent로 최소화하면서 W_t=α_tW_{t-1}+v_tk_t^T가 된다는 유도는 부호가 반대다. 이 loss의 음의 gradient는 −vφ(k)^T를 write한다. 또한 gated linear attention으로 환원되려면 φ가 identity라는 조건도 필요하다.",
    "evidence": "papers/2505.23735.txt:932-960의 Eq. 19와 App. D.1 부근 2920-2965의 Eq. 49–51은 양의 내적을 최소화한다고 정의한 뒤 양의 Hebbian write를 적어 동일한 부호 모순을 보인다.",
    "suggested_fix": "attentional bias를 ℓ=−⟨M(φ(k)),v⟩로 정의하거나, 양의 내적을 gradient ascent로 최대화한다고 써라. gated linear attention 환원에는 φ(x)=x 조건을 명시하라."
  },
  {
    "severity": "major",
    "location": "§14.3.5 DeepTransformers",
    "issue": "식 (26)이 표준 Transformer의 특수 사례이므로 DeepTransformers가 strict generalization이라는 결론은 성립하지 않는다. 식 (26)은 softmax 분모가 없는 unnormalized exponential attention이고, 표준 Transformer의 normalized softmax와는 다른 함수다. 뒤에서 분모 누락을 한계로 인정하는 설명과도 모순된다.",
    "evidence": "papers/2505.23735.txt:977-1038의 Eq. 21은 softmax denominator를 포함한다. 1063-1091의 Eq. 25–26은 명시적으로 unnormalized formulation을 유도한 뒤 Transformer의 strict generalization이라고 주장한다.",
    "suggested_fix": "'unnormalized exponential attention의 strict generalization'으로 한정하라. 표준 Transformer까지 포함한다고 주장하려면 denominator를 계산하는 별도 state와 정규화 연산을 모델 정의에 추가해야 한다."
  },
  {
    "severity": "minor",
    "location": "§14.4 표 14-2 gate-producer 행",
    "issue": "α_t, η_t, β_t와 c개의 γ_{t,i}를 모두 x_t에서 출력하는 작은 함수가 존재한다는 구현 설명은 원전에 없다. 특히 γ_{t,i}가 현재 x_t 하나만의 함수인지, 각 window token이나 pairwise feature에도 의존하는지는 공개되지 않았다.",
    "evidence": "papers/2505.23735.txt:712-720은 γ를 input-dependent parameters라고만 서술한다. §5.1과 App. D.4에도 구체적인 gate-producer 입력·구조는 제시되지 않는다.",
    "suggested_fix": "gate 값은 outer-loop에서 학습된 메커니즘이 생성한다고만 쓰고, producer의 입력과 출력 구조는 원전 미공개라고 표시하라."
  },
  {
    "severity": "major",
    "location": "§14.4 표 14-2 window buffer 및 §14.7 state 크기·decode 트래픽",
    "issue": "최근 φ(k)를 저장한다고 해 놓고 buffer 크기를 O(c(d_k+d_v))로 계산한 것은 차원 오류다. 명시적 polynomial lift의 차원은 D=binom(d_k+p,p)이므로 O(c(D+d_v))가 된다. 같은 이유로 Atlas state를 16d_m^2, DeltaNet 대비 RMW 16배라고 단정하는 계산도 polynomial 입력 차원 또는 sketch 차원이 공개되지 않은 상황에서는 성립하지 않는다.",
    "evidence": "papers/2505.23735.txt:574-587과 2848-2920은 φ_p의 출력 차원을 D=binom(d_k+p,p)로 둔다. App. E 부근 3139-3160은 2층·expansion 4만 밝히며 실제 p, sketch 차원, head별 matrix shape는 주지 않는다.",
    "suggested_fix": "명시적 lift라면 buffer를 O(c(D+d_v))로 고쳐라. memory matrix 크기는 실제 feature/sketch 차원을 포함한 기호로 유지하고, 구현 차원이 공개되기 전에는 16d_m^2, crossover 16K, DeltaNet 대비 16배 같은 Atlas 고유 수치를 제시하지 마라."
  },
  {
    "severity": "minor",
    "location": "§14.6 S-NIAH (RULER)",
    "issue": "Dot이 모든 설정에서 96.8–100을 기록했다는 수치 요약이 틀렸다. S-NIAH-W 16K에서 Dot은 93.2이므로 전체 범위의 최솟값은 93.2다.",
    "evidence": "papers/2505.23735.txt:1872-2053의 Table 3에서 Dot 행의 S-NIAH-W 결과는 99.0, 98.4, 93.2이며, 16K 값이 93.2로 제시된다.",
    "suggested_fix": "'Dot은 전 설정에서 93.2–100'으로 고치거나, 평가 축별 수치를 분리해 적어라."
  }
]