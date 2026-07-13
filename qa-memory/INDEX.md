# QA INDEX — 연상 회상용 색인 (auto-generated)

3건. story line 짤 때 여기서 꺼낸다.

## 1. 안다-4축별 클러스터

### 모른다는 걸 안다 (known_unknown → decomposition) — 3건
- Q001: Titans에는 k,v로 구성된 연상메모리가 있는듯한데 k가 주어졌을때 v가 나오는? 그 weight를 token마다 학습? 그럼 q는 안씀? 
- Q002: decode 기준 흐름: 토큰1개→Wq,k,v로 q,k,v→k를 M에 흘려 v'→v와 비교해 loss→S,M update? 그 다음은? q,k,
- Q003: 선형 메모리(W=d×d)면 update가 rank-1 outer-product write((Wk-v)k^T)로 퇴화, 데이터 의존성=W_{t-1

### 모른다는 것도 모른다 (unknown_unknown → exploration) — 9건
- Q001: Titans에는 k,v로 구성된 연상메모리가 있는듯한데 k가 주어졌을때 v가 나오는? 그 weight를 token마다 학습? 그럼 q는 안씀? 
- Q001: Titans에는 k,v로 구성된 연상메모리가 있는듯한데 k가 주어졌을때 v가 나오는? 그 weight를 token마다 학습? 그럼 q는 안씀? 
- Q002: decode 기준 흐름: 토큰1개→Wq,k,v로 q,k,v→k를 M에 흘려 v'→v와 비교해 loss→S,M update? 그 다음은? q,k,
- Q002: decode 기준 흐름: 토큰1개→Wq,k,v로 q,k,v→k를 M에 흘려 v'→v와 비교해 loss→S,M update? 그 다음은? q,k,
- Q002: decode 기준 흐름: 토큰1개→Wq,k,v로 q,k,v→k를 M에 흘려 v'→v와 비교해 loss→S,M update? 그 다음은? q,k,
- Q003: 선형 메모리(W=d×d)면 update가 rank-1 outer-product write((Wk-v)k^T)로 퇴화, 데이터 의존성=W_{t-1
- Q003: 선형 메모리(W=d×d)면 update가 rank-1 outer-product write((Wk-v)k^T)로 퇴화, 데이터 의존성=W_{t-1
- Q003: 선형 메모리(W=d×d)면 update가 rank-1 outer-product write((Wk-v)k^T)로 퇴화, 데이터 의존성=W_{t-1
- Q003: 선형 메모리(W=d×d)면 update가 rank-1 outer-product write((Wk-v)k^T)로 퇴화, 데이터 의존성=W_{t-1

### 아는데 안 드러남 (unknown_known → prototype/react) — 2건
- Q001: Titans에는 k,v로 구성된 연상메모리가 있는듯한데 k가 주어졌을때 v가 나오는? 그 weight를 token마다 학습? 그럼 q는 안씀? 
- Q002: decode 기준 흐름: 토큰1개→Wq,k,v로 q,k,v→k를 M에 흘려 v'→v와 비교해 loss→S,M update? 그 다음은? q,k,

### 안다는 걸 안다 (known) — 0건

## 2. 개념 key별 클러스터 (연상)

- **associative memory** (1): Q001
- **key-value** (1): Q001
- **query** (1): Q001
- **read vs write** (1): Q001
- **per-token update** (1): Q001
- **inner loop** (1): Q001
- **outer loop projections** (1): Q001
- **surprise** (1): Q001
- **l2 loss** (1): Q001
- **associative regression** (1): Q001
- **memory-as-weights** (1): Q001
- **mac** (1): Q002
- **mag** (1): Q002
- **mal** (1): Q002
- **softmax attention** (1): Q002
- **retrieval** (1): Q002
- **persistent memory** (1): Q002
- **read m*(q)** (1): Q002
- **write from attention output** (1): Q002
- **segment/chunk** (1): Q002
- **short-term vs long-term memory** (1): Q002
- **two projection sets** (1): Q002
- **gate** (1): Q002
- **linear attention** (1): Q003
- **deltanet** (1): Q003
- **delta rule** (1): Q003
- **gated deltanet** (1): Q003
- **ttt** (1): Q003
- **rank-1 outer product** (1): Q003
- **error-correction** (1): Q003
- **forget gate** (1): Q003
- **weight decay** (1): Q003
- **momentum** (1): Q003
- **sequential recurrence** (1): Q003
- **data dependency** (1): Q003
- **deep memory** (1): Q003
- **longhorn** (1): Q003
- **rwkv-7** (1): Q003
- **four-axis generalization** (1): Q003

## 3. 열린 실 (think_about)

- [Q001] Miras가 이 L2 loss를 Lp/Huber/KL 'attentional bias'로 일반화 (G04/ch13) — 왜 다른 loss가 이기나
- [Q001] read M*(q_t)가 MAC/MAG/MAL에서 windowed attention과 어떻게 결합되나
- [Q001] M(k_t)와 M*(q_t)는 같은 네트워크에 다른 입력 — retrieval의 '비슷한 key 일반화'가 어디서 나오나(MLP 비선형성)
- [Q001] surprise(gradient 크기)를 momentum S_t가 어떻게 누적하고 weight decay(1-α)가 어떻게 잊나
- [Q002] decode 시 MAC segment 경계를 1토큰씩 어떻게 처리하나
- [Q002] o=y⊗M*(y)의 gate ⊗가 정확히 뭔지(정규화+σ+곱)
- [Q002] attention이 '무엇을 메모리에 쓸지 필터'한다는 것의 정량 함의(memory overflow 완화, ablation)
- [Q002] persistent memory P=task knowledge(입력무관 파라미터)와 FFN=data-independent attention 관점
- [Q003] delta rule의 (I-βkk^T) 곱셈형 recurrence가 왜 병렬화를 어렵게 하나 → WY/UT chunkwise (G02/ch09, TNT)
- [Q003] momentum이 실제로 얼마나 개선하나 — surprise 이후 under-memorization 정량
- [Q003] deep memory(MLP)가 matrix 대비 용량/표현력을 어디서 얻나 → Atlas 용량 이론(O(d_k^p))과 연결
- [Q003] Titans가 TC^0 넘는다는 Thm 4.1의 의미(상태추적 표현력)

## 4. Storyline seeds

- [Q001] (Titans) Titans 한 줄: 메모리=k→v 회귀를 test-time에 L2로 학습(write=k,v / read=q / surprise=gradient). 투영은 outer-loop 고정. 이후 Miras가 loss 일반화, Atlas가 용량·optimizer.
- [Q002] (Titans) Titans 전체 블록=단기(softmax attention)+장기(LMM)+persistent 결합. MAC=retrieve→attention→write(attention출력을 메모리에)→output; MAG=병렬 게이트; MAL=직렬(최약). q,k,v 두 세트. Q001의 LMM 코어가 이 블록의 한 부품.
- [Q003] (lineage (linear attn→DeltaNet→GDN→TTT→Titans)) 계보 한 줄: linear attn(순수 add,overflow) → DeltaNet(A:국소 error-correction overwrite) → Gated DeltaNet(+B:전역 forget) / TTT(+D:deep memory, but forget·momentum 없음) → Titans(A+B+C:momentum+D:deep = 네 축 통합, GDN(η=0)·TTT를 특수case로). 핵심 교훈: '국소 overwrite vs 전역 forget'을 구분하고, Titans의 새 조각=momentum.
