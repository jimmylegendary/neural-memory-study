# QA INDEX — 연상 회상용 색인 (auto-generated)

7건. story line 짤 때 여기서 꺼낸다.

## 1. 안다-4축별 클러스터

### 모른다는 걸 안다 (known_unknown → decomposition) — 7건
- Q001: Titans에는 k,v로 구성된 연상메모리가 있는듯한데 k가 주어졌을때 v가 나오는? 그 weight를 token마다 학습? 그럼 q는 안씀? 
- Q002: decode 기준 흐름: 토큰1개→Wq,k,v로 q,k,v→k를 M에 흘려 v'→v와 비교해 loss→S,M update? 그 다음은? q,k,
- Q003: 선형 메모리(W=d×d)면 update가 rank-1 outer-product write((Wk-v)k^T)로 퇴화, 데이터 의존성=W_{t-1
- Q004: matrix와 mlp 차이? mlp는 weight matrix 2개+중간 activation? ffn이랑 차이? Titans에서 k->v MLP
- Q005: 1)surprise=k에 대한 v가 전부? 2)3개 gate 어떻게 학습? token축 맞아? backward시 gate값으로 S구하고 W up
- Q006: 1)gate 3개 weight=d 벡터? 2)Persistent memory 정체·논문 위치·학습·정보·shape? 3)MAC 3부품 의존없이 
- Q007: MAC 전체 forward 추적 검증: seq C×d→W_Q로 C query→NM으로 C출력→앞에 concat, persistent도 앞에→(N

### 모른다는 것도 모른다 (unknown_unknown → exploration) — 21건
- Q001: Titans에는 k,v로 구성된 연상메모리가 있는듯한데 k가 주어졌을때 v가 나오는? 그 weight를 token마다 학습? 그럼 q는 안씀? 
- Q001: Titans에는 k,v로 구성된 연상메모리가 있는듯한데 k가 주어졌을때 v가 나오는? 그 weight를 token마다 학습? 그럼 q는 안씀? 
- Q002: decode 기준 흐름: 토큰1개→Wq,k,v로 q,k,v→k를 M에 흘려 v'→v와 비교해 loss→S,M update? 그 다음은? q,k,
- Q002: decode 기준 흐름: 토큰1개→Wq,k,v로 q,k,v→k를 M에 흘려 v'→v와 비교해 loss→S,M update? 그 다음은? q,k,
- Q002: decode 기준 흐름: 토큰1개→Wq,k,v로 q,k,v→k를 M에 흘려 v'→v와 비교해 loss→S,M update? 그 다음은? q,k,
- Q003: 선형 메모리(W=d×d)면 update가 rank-1 outer-product write((Wk-v)k^T)로 퇴화, 데이터 의존성=W_{t-1
- Q003: 선형 메모리(W=d×d)면 update가 rank-1 outer-product write((Wk-v)k^T)로 퇴화, 데이터 의존성=W_{t-1
- Q003: 선형 메모리(W=d×d)면 update가 rank-1 outer-product write((Wk-v)k^T)로 퇴화, 데이터 의존성=W_{t-1
- Q003: 선형 메모리(W=d×d)면 update가 rank-1 outer-product write((Wk-v)k^T)로 퇴화, 데이터 의존성=W_{t-1
- Q004: matrix와 mlp 차이? mlp는 weight matrix 2개+중간 activation? ffn이랑 차이? Titans에서 k->v MLP
- Q004: matrix와 mlp 차이? mlp는 weight matrix 2개+중간 activation? ffn이랑 차이? Titans에서 k->v MLP
- Q004: matrix와 mlp 차이? mlp는 weight matrix 2개+중간 activation? ffn이랑 차이? Titans에서 k->v MLP
- Q005: 1)surprise=k에 대한 v가 전부? 2)3개 gate 어떻게 학습? token축 맞아? backward시 gate값으로 S구하고 W up
- Q005: 1)surprise=k에 대한 v가 전부? 2)3개 gate 어떻게 학습? token축 맞아? backward시 gate값으로 S구하고 W up
- Q005: 1)surprise=k에 대한 v가 전부? 2)3개 gate 어떻게 학습? token축 맞아? backward시 gate값으로 S구하고 W up
- Q006: 1)gate 3개 weight=d 벡터? 2)Persistent memory 정체·논문 위치·학습·정보·shape? 3)MAC 3부품 의존없이 
- Q006: 1)gate 3개 weight=d 벡터? 2)Persistent memory 정체·논문 위치·학습·정보·shape? 3)MAC 3부품 의존없이 
- Q006: 1)gate 3개 weight=d 벡터? 2)Persistent memory 정체·논문 위치·학습·정보·shape? 3)MAC 3부품 의존없이 
- Q007: MAC 전체 forward 추적 검증: seq C×d→W_Q로 C query→NM으로 C출력→앞에 concat, persistent도 앞에→(N
- Q007: MAC 전체 forward 추적 검증: seq C×d→W_Q로 C query→NM으로 C출력→앞에 concat, persistent도 앞에→(N
- Q007: MAC 전체 forward 추적 검증: seq C×d→W_Q로 C query→NM으로 C출력→앞에 concat, persistent도 앞에→(N

### 아는데 안 드러남 (unknown_known → prototype/react) — 6건
- Q001: Titans에는 k,v로 구성된 연상메모리가 있는듯한데 k가 주어졌을때 v가 나오는? 그 weight를 token마다 학습? 그럼 q는 안씀? 
- Q002: decode 기준 흐름: 토큰1개→Wq,k,v로 q,k,v→k를 M에 흘려 v'→v와 비교해 loss→S,M update? 그 다음은? q,k,
- Q004: matrix와 mlp 차이? mlp는 weight matrix 2개+중간 activation? ffn이랑 차이? Titans에서 k->v MLP
- Q005: 1)surprise=k에 대한 v가 전부? 2)3개 gate 어떻게 학습? token축 맞아? backward시 gate값으로 S구하고 W up
- Q006: 1)gate 3개 weight=d 벡터? 2)Persistent memory 정체·논문 위치·학습·정보·shape? 3)MAC 3부품 의존없이 
- Q007: MAC 전체 forward 추적 검증: seq C×d→W_Q로 C query→NM으로 C출력→앞에 concat, persistent도 앞에→(N

### 안다는 걸 안다 (known) — 0건

## 2. 개념 key별 클러스터 (연상)

- **persistent memory** (4): Q002, Q004, Q005, Q006
- **inner loop** (2): Q001, Q005
- **surprise** (2): Q001, Q005
- **mac** (2): Q002, Q005
- **two projection sets** (2): Q002, Q005
- **deep memory** (2): Q003, Q004
- **associative memory** (1): Q001
- **key-value** (1): Q001
- **query** (1): Q001
- **read vs write** (1): Q001
- **per-token update** (1): Q001
- **outer loop projections** (1): Q001
- **l2 loss** (1): Q001
- **associative regression** (1): Q001
- **memory-as-weights** (1): Q001
- **mag** (1): Q002
- **mal** (1): Q002
- **softmax attention** (1): Q002
- **retrieval** (1): Q002
- **read m*(q)** (1): Q002
- **write from attention output** (1): Q002
- **segment/chunk** (1): Q002
- **short-term vs long-term memory** (1): Q002
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
- **longhorn** (1): Q003
- **rwkv-7** (1): Q003
- **four-axis generalization** (1): Q003
- **matrix memory** (1): Q004
- **mlp memory** (1): Q004
- **activation function** (1): Q004
- **ffn** (1): Q004
- **feed-forward network** (1): Q004
- **nonlinearity** (1): Q004
- **hidden dimension** (1): Q004
- **memory shape** (1): Q004
- **p_m** (1): Q004
- **state size** (1): Q004
- **l_m layers** (1): Q004
- **weights-as-state** (1): Q004
- **expansion 4d** (1): Q004
- **rank-1 gradient** (1): Q004
- **gates** (1): Q005
- **learning rate theta** (1): Q005
- **momentum decay eta** (1): Q005
- **weight decay alpha** (1): Q005
- **outer loop** (1): Q005
- **meta-learning** (1): Q005
- **bilevel** (1): Q005
- **differentiate through optimizer** (1): Q005
- **gate shape scalar** (1): Q005
- **core** (1): Q005
- **contextual memory** (1): Q005
- **short-term vs long-term** (1): Q005
- **write-then-read** (1): Q005
- **gate weight shape** (1): Q006
- **sec 3.3** (1): Q006
- **input-independent** (1): Q006
- **task knowledge** (1): Q006
- **attention sink** (1): Q006
- **ffn as fixed-kv attention** (1): Q006
- **augmented sequence** (1): Q006
- **softmax attention input** (1): Q006
- **nm projections w_k/v/q** (1): Q006
- **retrieval uses q not k** (1): Q006
- **one mlp** (1): Q006
- **chunk segment fixed non-overlapping** (1): Q006
- **sliding window mag** (1): Q006
- **retrieved vector c×d** (1): Q006
- **n_p tokens** (1): Q006
- **mac full forward** (1): Q007
- **augmented sequence np+2c** (1): Q007
- **write with y_t** (1): Q007
- **output o_t=y_t⊗m*_t(y_t)** (1): Q007
- **output gate not eltwise mul** (1): Q007
- **sequence length preserved** (1): Q007
- **block=layer** (1): Q007
- **stack l layers** (1): Q007
- **per-layer neural memory** (1): Q007
- **embedding** (1): Q007
- **lm head** (1): Q007
- **no separate ffn** (1): Q007
- **persistent memory as ffn** (1): Q007
- **sec 4.4 block details** (1): Q007
- **depthwise-separable conv** (1): Q007

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
- [Q004] 왜 비선형(MLP)이 matrix보다 용량이 큰가 — 선형분리 불가 연상 구체 예 → Atlas 용량 이론(O(d_k^p))
- [Q004] Titans memory MLP가 width d(확장 없음)인 이유 — state 크기 vs 표현력 trade
- [Q004] 2-layer MLP의 gradient(backprop 전체)가 병렬화를 막는 방식 → chunkwise/TNT(Q003 순차성과 연결)
- [Q004] 메모리 MLP 활성함수 선택(SiLU/GELU)이 test-time 학습에 주는 영향
- [Q005] outer loop가 unrolled 사슬을 backprop할 때 메모리가 rank-1(선형) 아니면 무거워짐 → chunkwise 병렬(Q003/Q004 순차성 실과 연결, TNT)
- [Q005] gate가 scalar per head인지 per channel인지 정확히(원문 v1 미명세) — Atlas/Miras에서 더 정교해짐
- [Q005] θ/η/α의 데이터 의존성이 '문맥 전환' 감지에 쓰이는 방식(η→0=context switch)
- [Q005] Core의 attention이 full causal(MAC) vs sliding-window(MAG)인 차이가 비용/능력에 주는 영향
- [Q006] N_p(persistent 토큰 수)·C(chunk 크기)의 전형값과 증강 시퀀스 N_p+2C가 attention 비용에 주는 영향
- [Q006] 검색 h_t가 segment 토큰당 1개(C×d)인지 요약 1개인지 원문 Fig/식으로 재확인
- [Q006] gate가 head별인지(shape R^{d×H}) 원문 v1 미명세 — Atlas/Miras에서 정교화
- [Q006] chunk 경계에서 메모리 상태 이어짐 = inter-chunk recurrence = 순차성(Q003/Q005 실, TNT가 손대는 지점)
- [Q007] 출력 o_t가 C×d로 잘리는(segment 위치만) 정확한 방식 원문 Fig/식 재확인
- [Q007] 메모리 M·S·persistent가 layer마다 독립인지, head마다 독립인지(멀티헤드 구조)
- [Q007] 표준 FFN 없이 persistent만으로 충분한지 — ablation(persistent 기여가 weight decay>momentum>conv>persistent로 최하위였음, Q005 계열)
- [Q007] layer 쌓을 때 각 layer 메모리가 서로 다른 추상화를 담는지(계층적 기억)

## 4. Storyline seeds

- [Q001] (Titans) Titans 한 줄: 메모리=k→v 회귀를 test-time에 L2로 학습(write=k,v / read=q / surprise=gradient). 투영은 outer-loop 고정. 이후 Miras가 loss 일반화, Atlas가 용량·optimizer.
- [Q002] (Titans) Titans 전체 블록=단기(softmax attention)+장기(LMM)+persistent 결합. MAC=retrieve→attention→write(attention출력을 메모리에)→output; MAG=병렬 게이트; MAL=직렬(최약). q,k,v 두 세트. Q001의 LMM 코어가 이 블록의 한 부품.
- [Q003] (lineage (linear attn→DeltaNet→GDN→TTT→Titans)) 계보 한 줄: linear attn(순수 add,overflow) → DeltaNet(A:국소 error-correction overwrite) → Gated DeltaNet(+B:전역 forget) / TTT(+D:deep memory, but forget·momentum 없음) → Titans(A+B+C:momentum+D:deep = 네 축 통합, GDN(η=0)·TTT를 특수case로). 핵심 교훈: '국소 overwrite vs 전역 forget'을 구분하고, Titans의 새 조각=momentum.
- [Q004] (Titans (memory shape: matrix vs MLP vs FFN)) 메모리 형태 축(D): matrix(L_M=1, 선형 Wk) vs MLP(L_M≥2, 비선형 W_2σ(W_1 k)). MLP=FFN과 같은 구조지만 weight가 '고정 함수'가 아니라 '갱신되는 state'; Titans는 폭 d 유지(P_M≈L_M d²), state=2P_M. L_M=1=linear/DeltaNet 특수case. MLP면 gradient rank-1 아님→학습 비쌈(TNT로 이어짐).
- [Q005] (Titans (gates inner/outer loop; MAC core/contextual/persistent)) Titans 학습의 핵심 구분: inner loop(추론, 매 토큰 M·S만 갱신, gate는 값만 계산) vs outer loop(사전학습, unrolled 사슬 backprop=optimizer 미분=meta-learning으로 gate 생성기·투영·P·M_0 학습). MAC은 write-then-read라 이번 write가 이번 출력에 반영. 세 부품: Core=attention(q,k,v), Contextual=neural memory MLP(memory q,k,v), Persistent=고정 벡터(q,k,v 아님).
- [Q006] (Titans MAC details (persistent memory, NM projections, augmented seq shapes, chunking)) MAC 해부: NM=MLP 하나+투영 3개(쓰기 k,v / 읽기 q, 검색은 q!). 증강 시퀀스=[P(N_p,과제지식,Sec3.3) || 검색 h_t(C) || segment(C)]=N_p+2C 토큰이 softmax attention 입력. chunk=고정 비중첩(sliding 아님, 그건 MAG), 메모리는 chunk 넘어 이어짐(inter-chunk recurrence=순차성 실).
- [Q007] (Titans MAC full forward + layer stacking + no separate FFN) MAC 한 블록 완결: 검색(q)→[P‖h_t‖S] attention(softmax)→write(y_t로 M 갱신)→출력 o_t=y_t⊗M*_t(y_t)(게이트). 이게 한 layer, 모델=임베딩→블록×L→LM head, 각 layer 자기 NM. 별도 FFN 없음(persistent가 FFN 역할). C×d 출력으로 길이 보존.
