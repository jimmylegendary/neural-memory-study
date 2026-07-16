# QA LOG — 공부 질문 기록 (chronological)

총 38건.

## Q001 · 2026-07-14 · Titans

**Q.** Titans에는 k,v로 구성된 연상메모리가 있는듯한데 k가 주어졌을때 v가 나오는? 그 weight를 token마다 학습? 그럼 q는 안씀? loss는 뭐와 뭐의 차이를 어떤 손실함수로 구함?

**A.** 메모리 M=작은 MLP, weight 자체가 state이고 M(k)≈v가 되도록 저장(표 아님, 가중치에 저장). token마다 갱신되는 건 메모리 M의 weight뿐(inner loop); 투영 W_K/W_V/W_Q는 outer-loop 사전학습·inference 고정. q는 씀 — 단 write(학습)가 아니라 read(retrieval)에: write=(k,v)로 loss→메모리 갱신, read=M*(q_t)로 갱신 없이 검색. loss = ||M(k_t)-v_t||_2^2 (L2/MSE 회귀), 메모리 예측 M(k_t) vs 정답 v_t의 차이; 이 gradient 크기가 surprise.

- 축: `known_unknown` → `known` · comprehension: working
- 새로 드러난 것: unknown_unknown: 투영 W_K/W_V/W_Q는 token마다 학습 X — outer-loop 고정, 메모리 M weight만 per-token 갱신; unknown_unknown: q는 안 쓰는 게 아니라 read(retrieval)에 씀; write는 (k,v), read는 q — 두 연산 분리(M vs M*); unknown_known: '메모리=표'가 아니라 '메모리=가중치에 저장된 회귀함수'라는 프레임
- 개념 key: associative memory, key-value, query, read vs write, per-token update, inner loop, outer loop projections, surprise, L2 loss, associative regression, memory-as-weights
- 생각할 것: Miras가 이 L2 loss를 Lp/Huber/KL 'attentional bias'로 일반화 (G04/ch13) — 왜 다른 loss가 이기나; read M*(q_t)가 MAC/MAG/MAL에서 windowed attention과 어떻게 결합되나; M(k_t)와 M*(q_t)는 같은 네트워크에 다른 입력 — retrieval의 '비슷한 key 일반화'가 어디서 나오나(MLP 비선형성); surprise(gradient 크기)를 momentum S_t가 어떻게 누적하고 weight decay(1-α)가 어떻게 잊나
- storyline seed: Titans 한 줄: 메모리=k→v 회귀를 test-time에 L2로 학습(write=k,v / read=q / surprise=gradient). 투영은 outer-loop 고정. 이후 Miras가 loss 일반화, Atlas가 용량·optimizer.

## Q002 · 2026-07-14 · Titans

**Q.** decode 기준 흐름: 토큰1개→Wq,k,v로 q,k,v→k를 M에 흘려 v'→v와 비교해 loss→S,M update? 그 다음은? q,k,v 언제 어떻게 씀? 이걸로 softmax attention? 그 다음은? (Titans 기준)

**A.** 님 트레이스=LMM 코어 write, 맞음(읽기 y=M*(q) 추가). 함정: q,k,v가 두 세트(메모리 자신의 W_Q/K/V + attention 자신의 Q/K/V). 전체 블록=단기(softmax attention)+장기(LMM)+persistent. MAC 흐름: ①세그먼트로 메모리 검색 h=M*_{t-1}(q) ②[P‖h‖S] 증강 ③그 위에서 softmax attention y=Attn ④메모리는 raw토큰 아니라 attention출력 y로 write M_t=M_{t-1}(y) ⑤출력 o=y⊗M*_t(y). 순서=검색→attention→쓰기→출력, attention이 메모리 입력을 필터. 님의 'token→메모리 직접' 트레이스는 MAG(병렬 게이트)/MAL(직렬)/LMM-alone에 해당; MAL이 최약. softmax attention은 어느 변형이든 항상 한 축.

- 축: `known_unknown` → `known` · comprehension: working
- 새로 드러난 것: unknown_unknown: q,k,v가 두 세트 — 메모리 자신의 것 + attention 자신의 것(별개 투영); unknown_unknown: MAC에선 메모리가 raw 토큰이 아니라 'attention 출력 y'로 갱신됨(attention이 무엇을 기억할지 필터); unknown_unknown: MAC은 retrieve-first(검색→attention→write→output); 님 트레이스(token→memory 직접)는 MAG/MAL/LMM-alone에 해당; unknown_known: '메모리+attention 결합 방식'이 곧 MAC/MAG/MAL 세 변형 — MAL이 최약
- 개념 key: MAC, MAG, MAL, softmax attention, retrieval, persistent memory, read M*(q), write from attention output, segment/chunk, short-term vs long-term memory, two projection sets, gate
- 생각할 것: decode 시 MAC segment 경계를 1토큰씩 어떻게 처리하나; o=y⊗M*(y)의 gate ⊗가 정확히 뭔지(정규화+σ+곱); attention이 '무엇을 메모리에 쓸지 필터'한다는 것의 정량 함의(memory overflow 완화, ablation); persistent memory P=task knowledge(입력무관 파라미터)와 FFN=data-independent attention 관점
- storyline seed: Titans 전체 블록=단기(softmax attention)+장기(LMM)+persistent 결합. MAC=retrieve→attention→write(attention출력을 메모리에)→output; MAG=병렬 게이트; MAL=직렬(최약). q,k,v 두 세트. Q001의 LMM 코어가 이 블록의 한 부품.
- 연상: Q001

## Q003 · 2026-07-14 · lineage (linear attn→DeltaNet→GDN→TTT→Titans)

**Q.** 선형 메모리(W=d×d)면 update가 rank-1 outer-product write((Wk-v)k^T)로 퇴화, 데이터 의존성=W_{t-1}→W_t recurrence(순차)인데 이해 안됨. 계보: linear attn(잊는게 없어 overflow)→gated deltanet(잊고 잊은 자리에 새로 씀)? 둘다 잊고 새로쓰는 거 같은데 뭐가 다름? Titans가 왜 TTT와 GDN 사이를 채웠다고 주장? 각 메커니즘·이득·손해와 Titans의 차이는?

**A.** 핵심: '잊기(전역 decay)'와 '덮어쓰기(국소 delta/error-correction)'는 다른 두 축. rank-1 outer product=(Wk-v)k^T는 k 방향 한 슬롯만 v쪽으로 조정(고정크기 W에 GEMM 한 번, KV append와 대비). 순차성=W_t가 W_{t-1}에 의존하는 사슬(W_0→W_1→…), 순서대로만 계산→나중에 chunkwise 병렬(TNT) 필요의 이유. 4손잡이 표: A국소overwrite/B전역forget/C momentum/D deep. linear attn=순수 add(A✗B✗)→overflow; DeltaNet=A✓B✗; Gated DeltaNet=A✓B✓ but 얕은matrix·momentum없음; TTT=A✓(GD step)·deep 가능 but forget·momentum 없음; Titans=A✓B✓C✓D✓ 전부. 원문 App.C: momentum 끄면(η=0) 정확히 Gated DeltaNet, TTT는 forget·momentum 없는 gradient 특수case. 그래서 Titans=GDN의 forget+TTT의 deep+어디에도 없던 momentum을 합친 상위 일반화(네 축). momentum 이득=놀람 직후 토큰 under-memorization을 관성으로 각인.

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_unknown: '잊기(전역 decay)'와 '덮어쓰기(국소 error-correction/delta)'는 독립된 두 축 — 사용자가 한 동작으로 뭉뚱그림; unknown_unknown: DeltaNet은 전역 forget 안 함(국소 overwrite만); GDN이 전역 forget을 추가 — '둘 다 잊는다'가 아님; unknown_unknown: Titans의 구별 조각은 momentum(과거 surprise) — GDN·TTT 어디에도 없던 것; 'forget+write'가 아니라 4축(A/B/C/D); unknown_unknown: 순차 recurrence(W_{t-1}→W_t)가 곧 학습이 비싼 이유 → chunkwise/TNT 복선
- 개념 key: linear attention, DeltaNet, delta rule, Gated DeltaNet, TTT, rank-1 outer product, error-correction, forget gate, weight decay, momentum, sequential recurrence, data dependency, deep memory, Longhorn, RWKV-7, four-axis generalization
- 생각할 것: delta rule의 (I-βkk^T) 곱셈형 recurrence가 왜 병렬화를 어렵게 하나 → WY/UT chunkwise (G02/ch09, TNT); momentum이 실제로 얼마나 개선하나 — surprise 이후 under-memorization 정량; deep memory(MLP)가 matrix 대비 용량/표현력을 어디서 얻나 → Atlas 용량 이론(O(d_k^p))과 연결; Titans가 TC^0 넘는다는 Thm 4.1의 의미(상태추적 표현력)
- storyline seed: 계보 한 줄: linear attn(순수 add,overflow) → DeltaNet(A:국소 error-correction overwrite) → Gated DeltaNet(+B:전역 forget) / TTT(+D:deep memory, but forget·momentum 없음) → Titans(A+B+C:momentum+D:deep = 네 축 통합, GDN(η=0)·TTT를 특수case로). 핵심 교훈: '국소 overwrite vs 전역 forget'을 구분하고, Titans의 새 조각=momentum.
- 연상: Q001, Q002

## Q004 · 2026-07-14 · Titans (memory shape: matrix vs MLP vs FFN)

**Q.** matrix와 mlp 차이? mlp는 weight matrix 2개+중간 activation? ffn이랑 차이? Titans에서 k->v MLP 구조와 shape은?

**A.** matrix=행렬 W 하나(d×d), 읽기 Wk, 선형 사상만. MLP=W_2 σ(W_1 k), 중간 비선형으로 비선형 연상까지(용량↑,비쌈). 2-layer MLP=weight 2개+사이 activation 맞음(=hidden 1개); L_M=1이면 행렬 하나=선형 특수case. FFN=구조는 MLP와 동일(W_2 σ(W_1 x)) — 다른 건 역할: 보통 FFN weight는 inference 고정, Titans 메모리 MLP는 weight가 state라 매 토큰 갱신. 크기도: 표준 FFN hidden=4d 확장, Titans 메모리 MLP는 폭 d 유지→P_M≈L_M·d²(2층≈2d²). 보너스 FFN=W_V softmax(W_K x)=고정 KV attention=persistent memory 역할. Titans k→v MLP: k,v,q,v' 모두 d차원; 2층이면 W_1(d×d)→σ→W_2(d×d), 이 W_1/W_2가 곧 state M_t(매 토큰 갱신); state=2P_M(M_t+momentum S_t). MLP면 gradient가 rank-1 아니라 순차 recurrence 무거워짐→Titans가 DeltaNet보다 학습 비싼 이유(TNT 복선).

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_known: MLP와 FFN은 '같은 구조'다 — 차이는 구조가 아니라 역할(고정 함수 vs 갱신되는 state); unknown_unknown: Titans 메모리 MLP는 폭 d 유지(확장 없음)라 P_M≈L_M·d²(2층≈2d²), 표준 FFN의 4d 확장과 다름; unknown_unknown: FFN=W_V softmax(W_K x)=고정 KV attention → Titans persistent memory의 정체; unknown_unknown: state=2P_M(weights+momentum), L_M=1이 linear/DeltaNet 특수case; MLP면 gradient가 rank-1 아님→순차 recurrence 무거워짐(TNT 복선, Q003 연결)
- 개념 key: matrix memory, MLP memory, deep memory, activation function, FFN, feed-forward network, nonlinearity, hidden dimension, memory shape, P_M, state size, L_M layers, persistent memory, weights-as-state, expansion 4d, rank-1 gradient
- 생각할 것: 왜 비선형(MLP)이 matrix보다 용량이 큰가 — 선형분리 불가 연상 구체 예 → Atlas 용량 이론(O(d_k^p)); Titans memory MLP가 width d(확장 없음)인 이유 — state 크기 vs 표현력 trade; 2-layer MLP의 gradient(backprop 전체)가 병렬화를 막는 방식 → chunkwise/TNT(Q003 순차성과 연결); 메모리 MLP 활성함수 선택(SiLU/GELU)이 test-time 학습에 주는 영향
- storyline seed: 메모리 형태 축(D): matrix(L_M=1, 선형 Wk) vs MLP(L_M≥2, 비선형 W_2σ(W_1 k)). MLP=FFN과 같은 구조지만 weight가 '고정 함수'가 아니라 '갱신되는 state'; Titans는 폭 d 유지(P_M≈L_M d²), state=2P_M. L_M=1=linear/DeltaNet 특수case. MLP면 gradient rank-1 아님→학습 비쌈(TNT로 이어짐).
- 연상: Q001, Q003

## Q005 · 2026-07-14 · Titans (gates inner/outer loop; MAC core/contextual/persistent)

**Q.** 1)surprise=k에 대한 v가 전부? 2)3개 gate 어떻게 학습? token축 맞아? backward시 gate값으로 S구하고 W update하고 그 뒤 gate 3개도 update? 어떻게/shape? 2-1)MAC기준 update 다하고 나온 출력을 이번 추론에 쓰나? 3)이 MLP가 neural memory? MAC의 core/contextual/persistent은 어떻게 구해짐? q,k,v로 설명됨?

**A.** 1)yes: 순간 surprise=∇||M(k)-v||²=k→v 예측오차. 전체=순간+momentum(과거 surprise). 2)핵심: gate는 추론 때 갱신 X. inner loop(매 토큰)=gate 값을 x_t에서 계산(θ=softplus,η/α=sigmoid, linear head)→surprise ∇ℓ(메모리 내부 backward)→S=ηS-θ∇ℓ→M=(1-α)M+S. 바뀌는 건 M,S뿐. outer loop(사전학습)=매 토큰 M 업데이트가 forward 그래프의 일부, LM loss를 이 사슬 전체 관통해 backprop('optimizer를 미분'=meta-learning)해서 gate 생성기 w_θ/η/α, W_K/V/Q, P, attention, M_0을 학습; 추론 땐 고정. gate 값=스칼라(토큰/head당), 생성기=작은 R^d→스칼라 head. 2-1)yes: MAC은 검색→attention y→쓰기 M_t=M_{t-1}(y)→출력 o=y⊗M*_t(y). o가 이번 스텝 출력이고, 출력의 메모리 읽기는 방금 갱신된 M_t(write-then-read). 3)yes 그 MLP=neural(long-term) memory=Contextual Memory. Core=증강세그먼트 위 softmax attention(attention 자신의 q,k,v); Contextual=메모리 자신의 q(검색)/k,v(쓰기); Persistent=q,k,v 아님, 학습된 입력무관 토큰(과제지식).

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_unknown: gate는 추론 중 '학습'되지 않음 — 매 토큰 값만 계산(forward), 갱신되는 건 M,S뿐. gate '생성기 파라미터'는 outer loop에서만 학습; unknown_unknown: outer loop는 unrolled inner loop 전체를 관통해 backprop('optimizer를 미분'=meta-learning); 사용자가 상상한 '토큰마다 gate update'는 outer backprop에서 한 번; unknown_unknown: MAC 출력 o=y⊗M*_t(y)는 방금 갱신된 M_t를 읽음(write-then-read) — 이번 write가 이번 출력에 반영; unknown_known: Persistent memory는 q,k,v가 아니라 그냥 학습된 입력무관 벡터; Core=attention q,k,v, Contextual=memory q,k,v(두 세트)
- 개념 key: surprise, gates, learning rate theta, momentum decay eta, weight decay alpha, inner loop, outer loop, meta-learning, bilevel, differentiate through optimizer, gate shape scalar, MAC, core, contextual memory, persistent memory, short-term vs long-term, write-then-read, two projection sets
- 생각할 것: outer loop가 unrolled 사슬을 backprop할 때 메모리가 rank-1(선형) 아니면 무거워짐 → chunkwise 병렬(Q003/Q004 순차성 실과 연결, TNT); gate가 scalar per head인지 per channel인지 정확히(원문 v1 미명세) — Atlas/Miras에서 더 정교해짐; θ/η/α의 데이터 의존성이 '문맥 전환' 감지에 쓰이는 방식(η→0=context switch); Core의 attention이 full causal(MAC) vs sliding-window(MAG)인 차이가 비용/능력에 주는 영향
- storyline seed: Titans 학습의 핵심 구분: inner loop(추론, 매 토큰 M·S만 갱신, gate는 값만 계산) vs outer loop(사전학습, unrolled 사슬 backprop=optimizer 미분=meta-learning으로 gate 생성기·투영·P·M_0 학습). MAC은 write-then-read라 이번 write가 이번 출력에 반영. 세 부품: Core=attention(q,k,v), Contextual=neural memory MLP(memory q,k,v), Persistent=고정 벡터(q,k,v 아님).
- 연상: Q001, Q002, Q004

## Q006 · 2026-07-14 · Titans MAC details (persistent memory, NM projections, augmented seq shapes, chunking)

**Q.** 1)gate 3개 weight=d 벡터? 2)Persistent memory 정체·논문 위치·학습·정보·shape? 3)MAC 3부품 의존없이 동시에? 시퀀스에 붙임=input? 이걸로 Wqkv 구해 softmax? 3-1)검색은 k만? NM도 q,k,v 별도? k→v MLP만? 3-2)contextual·persistent에서 가져오는 벡터 row 길이(토큰수)? 3-3)sequence=일부 chunk? sliding? EOS까지 고정크기?

**A.** 1)yes 각 gate=linear head w∈R^d→스칼라(+softplus/sigmoid), 3d 파라미터(head별이면 R^{d×H}, v1 미명세). 2)Persistent=학습된 입력무관 벡터 N_p개 P=[p_1..p_Np], x_new=P||x. 논문 Sec 3.3. outer-loop 사전학습으로 backprop(추론 고정, 모든 입력 동일). 정보=과제 지식(task knowledge). 근거3: 메모리시스템/FFN=W_V softmax(W_K x)=고정KV attention/attention-sink 흡수. shape P∈R^{Np×d}. 3)완전병렬 아님: P·segment 즉시, contextual h_t=M*(q)는 이전 메모리 의존→검색먼저. 증강시퀀스 S̃=[P||h_t||S]가 attention 입력, attention 자기 Wq/k/v로 softmax(QK^T/√d)V. 3-1)정정: NM=MLP 하나+투영 3개(W_K/V/W_Q). MLP는 k→v 함수 하나뿐, q/k/v별 네트워크 없음. 쓰기=k,v로 M(k)≈v, 검색=q로 M*(q) — 검색은 k가 아니라 q. q,k,v 두 세트=①메모리 투영(같은 MLP에 넣을 k,v,q)+②attention 투영. 3-2)Persistent=N_p 토큰(고정)×d. Contextual h_t=segment 각 토큰이 query→C개×d. 증강 총 N_p+2C 토큰×d. 3-3)MAC=고정크기 C 비중첩 chunk(sliding 아님), 메모리는 chunk 넘어 상태 이어감, EOS까지 순서대로(마지막만 짧음). sliding은 MAG의 것.

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_unknown: NM=MLP 하나+투영 3개; 검색은 k가 아니라 q로(M*(q)); MLP는 k→v 함수 하나뿐(q/k/v별 네트워크 없음); unknown_unknown: 증강 시퀀스=[P(N_p) || h_t(C) || segment(C)]=N_p+2C 토큰, 이게 attention의 실제 입력; unknown_unknown: MAC=고정 비중첩 chunk(sliding 아님), 메모리가 chunk 넘어 이어짐; sliding은 MAG; unknown_known: Persistent memory=Sec 3.3, 입력무관 학습 벡터 N_p×d=과제지식, FFN=고정KV attention 역할+attention-sink 완화
- 개념 key: gate weight shape, persistent memory, Sec 3.3, input-independent, task knowledge, attention sink, FFN as fixed-KV attention, augmented sequence, softmax attention input, NM projections W_K/V/Q, retrieval uses q not k, one MLP, chunk segment fixed non-overlapping, sliding window MAG, retrieved vector C×d, N_p tokens
- 생각할 것: N_p(persistent 토큰 수)·C(chunk 크기)의 전형값과 증강 시퀀스 N_p+2C가 attention 비용에 주는 영향; 검색 h_t가 segment 토큰당 1개(C×d)인지 요약 1개인지 원문 Fig/식으로 재확인; gate가 head별인지(shape R^{d×H}) 원문 v1 미명세 — Atlas/Miras에서 정교화; chunk 경계에서 메모리 상태 이어짐 = inter-chunk recurrence = 순차성(Q003/Q005 실, TNT가 손대는 지점)
- storyline seed: MAC 해부: NM=MLP 하나+투영 3개(쓰기 k,v / 읽기 q, 검색은 q!). 증강 시퀀스=[P(N_p,과제지식,Sec3.3) || 검색 h_t(C) || segment(C)]=N_p+2C 토큰이 softmax attention 입력. chunk=고정 비중첩(sliding 아님, 그건 MAG), 메모리는 chunk 넘어 이어짐(inter-chunk recurrence=순차성 실).
- 연상: Q002, Q004, Q005

## Q007 · 2026-07-14 · Titans MAC full forward + layer stacking + no separate FFN

**Q.** MAC 전체 forward 추적 검증: seq C×d→W_Q로 C query→NM으로 C출력→앞에 concat, persistent도 앞에→(Np+2C)×d를 attention softmax→(Np+2C)×d 출력→이걸 NM forward/backward update→update된 NM으로 (Np+2C) 다시 forward→attn output과 eltwise mul=출력? gate 3개 값은 따로. 이게 모델 전체 끝? layer 쌓나? 다른 FFN 없나?

**A.** 대부분 정확. 정정1: 마지막 읽기는 raw Np+2C 입력이 아니라 y_t(attention 출력)를 forward — o_t=y_t⊗M*_t(y_t). 정정2: ⊗는 순수 eltwise mul 아니라 학습된 게이트(각 가지 정규화+σ+곱). 부수: 다음 층으로 가는 출력은 segment 위치 C×d(시퀀스 길이 보존, P·h_t 자리는 scratch). 이건 모델 전체가 아니라 '한 블록=한 layer'. layer를 쌓음: 블록ℓ 출력이 블록ℓ+1 입력, 각 layer는 자기만의 NM(M,S,투영,persistent). 전체=임베딩→(Titans 블록)×L→LM head. 별도 표준 FFN 없음 — persistent memory가 FFN 역할(FFN=W_V softmax(W_K x)=고정KV attention, Sukhbaatar). Block Details(Sec4.4)=residual/SiLU/q,k ℓ2정규화/q,k,v 뒤 depthwise-sep conv/출력 전 norm+linear gating, 여기 FFN 없음.

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_unknown: 마지막 출력 읽기는 raw 증강입력이 아니라 y_t(attention 출력)를 갱신된 메모리에 넣음; o_t=y_t⊗M*_t(y_t); unknown_unknown: ⊗는 순수 eltwise mul 아니라 학습된 게이트(정규화+σ+곱); unknown_unknown: 별도 표준 FFN 없음 — persistent memory가 FFN(=고정KV attention) 역할을 대신; unknown_known: 이건 한 블록=한 layer; 모델=임베딩→블록×L→LM head, 각 layer가 자기 NM 보유; 다음 층 출력은 C×d(길이 보존)
- 개념 key: MAC full forward, augmented sequence Np+2C, write with y_t, output o_t=y_t⊗M*_t(y_t), output gate not eltwise mul, sequence length preserved, block=layer, stack L layers, per-layer neural memory, embedding, LM head, no separate FFN, persistent memory as FFN, Sec 4.4 block details, depthwise-separable conv
- 생각할 것: 출력 o_t가 C×d로 잘리는(segment 위치만) 정확한 방식 원문 Fig/식 재확인; 메모리 M·S·persistent가 layer마다 독립인지, head마다 독립인지(멀티헤드 구조); 표준 FFN 없이 persistent만으로 충분한지 — ablation(persistent 기여가 weight decay>momentum>conv>persistent로 최하위였음, Q005 계열); layer 쌓을 때 각 layer 메모리가 서로 다른 추상화를 담는지(계층적 기억)
- storyline seed: MAC 한 블록 완결: 검색(q)→[P‖h_t‖S] attention(softmax)→write(y_t로 M 갱신)→출력 o_t=y_t⊗M*_t(y_t)(게이트). 이게 한 layer, 모델=임베딩→블록×L→LM head, 각 layer 자기 NM. 별도 FFN 없음(persistent가 FFN 역할). C×d 출력으로 길이 보존.
- 연상: Q002, Q005, Q006

## Q008 · 2026-07-14 · Titans MAC shapes + scan/momentum/chunkwise

**Q.** 1)attn 출력은 Np+2C,d 아님(y의 shape)? 그걸 다시 흘리면 Np+2C,d? 2)gate 정체가 뭐야, eltwise 곱 맞아? Np+2C,d가 어케 C,d 되냐? 3)NM update gradient 어케 구함, 미분? 4)scan이 뭔말? momentum도 C크기 g에 대해 한번에? S의 shape? gate값은 Np+2C 크기 scalar? t시점 S는 Np+2C 아니라 하나의 matrix여야 하는거 아님?

**A.** 1)맞음: attn 출력 y_t=(Np+2C)×d(길이 보존). 그 뒤 segment 위치 C개만 잘라 y_t^seg(C×d)가 write+다음층으로, P·h_t 자리는 scratch로 버림. 2)⊗gate=두 텐서를 학습 벡터weight로 정규화+σ+elementwise 곱(eltwise 맞음, shape 보존 C×d). Np+2C→C는 gate가 아니라 '슬라이싱'(prefix 버림)이 함. 두 연산 분리: 슬라이싱(길이축소)+gate(모양보존). 3)미분함: matrix면 닫힌형 (Wk-v)k^T(rank-1), MLP면 진짜 backprop(forward→오차→W1,W2 gradient), 매 토큰. 이건 메모리 자신 loss inner backward(LM loss outer와 다름). 4)핵심: S_t는 메모리와 같은 weight-shaped 버퍼 1개(P_M), Np+2C 아님(사용자 직관 맞음). 시퀀스 차원은 '몇번째 토큰 gradient/gate 적용'만 셈. gate값=토큰당 스칼라(write되는 C개→C개). chunk: C토큰 각각 g_τ(weight-shaped)+gate 스칼라, 재귀 S_t=η_t S_{t-1}-θ_t g_t / M_t=(1-α_t)M_{t-1}+S_t. scan=이 1차 선형 momentum 재귀를 parallel associative scan(prefix-sum 일반화, 감쇠 누적합, affine map 결합)으로 O(log C)에 병렬 계산(S4/S5/Mamba primitive). chunkwise 트릭으로 C개 gradient도 chunk시작 weight에서 matmul 병렬. 순차 대신 matmul+scan.

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_unknown: attn 출력은 (Np+2C)×d 전체; C로 줄이는 건 gate가 아니라 '슬라이싱'(segment 위치만 취함, prefix 버림); unknown_unknown: gate(⊗)는 shape 보존 elementwise 곱(정규화+σ 두름); 길이 축소는 슬라이싱 담당; unknown_unknown: scan=parallel associative scan=선형 momentum 재귀를 O(log C)에 병렬로 푸는 것(S4/S5/Mamba primitive); chunkwise는 gradient를 chunk-start weight에서 matmul 병렬; known 확정(사용자 직관 옳음): S_t는 weight-shaped 버퍼 1개(P_M), Np+2C 아님; 시퀀스 차원은 인덱스일 뿐. gate값은 토큰당 스칼라(C개)
- 개념 key: attention output shape Np+2C, slicing to segment C, gate elementwise mul, shape reduction by slicing not gate, gradient by backprop, rank-1 closed form linear, MLP backprop inner, associative scan, parallel prefix scan, linear recurrence momentum, S is weight-shaped buffer, gate values per-token scalar, chunkwise parallel training, gradients at chunk-start weights, S4/S5/Mamba scan primitive
- 생각할 것: write되는 게 segment C개인지 y_t 전체 Np+2C인지 원문 '틈'(M_t=M_{t-1}(y_t) 'tokens of y_t') 재확인 — 나는 C개(새 정보만) 읽기로 답함; chunk 시작 weight에서 gradient 평가하는 stale 근사가 품질에 주는 영향(chunk 클수록 stale↑) → TNT의 정확한 주제; read M*_t(y_t)에서 y_t를 W_Q로 투영 후 넣는지(query projection) 정확히; associative scan의 combine 연산자 (a1,b1)•(a2,b2)=(a1a2, a2b1+b2) — affine map 합성이 왜 결합법칙인지
- storyline seed: MAC 계산 정밀화: attn 출력 (Np+2C)×d → 슬라이싱으로 segment C×d(길이축소는 슬라이싱, 모양보존은 gate) → NM write(MLP면 backprop으로 gradient) → 출력 y_t^seg⊗M*_t(y_t^seg). S_t·M_t는 weight-shaped 버퍼 1개(시퀀스축 아님), gate는 토큰당 스칼라 C개. scan=선형 momentum 재귀를 O(log C) 병렬(Mamba primitive), chunkwise=gradient를 chunk-start weight matmul 병렬 — 이게 '순차라 비쌈→TNT' 실의 정체.
- 연상: Q003, Q005, Q006, Q007

## Q009 · 2026-07-14 · momentum parallelization via associative scan

**Q.** momentum이 병렬화를 어떻게 유지하는지 associative scan으로 설명해줘

**A.** momentum 재귀 S_t=η_t S_{t-1}-θ_t g_t는 S_{t-1}에 대해 선형(affine): S_t=a_t S_{t-1}+b_t, a_t=η_t, b_t=-θ_t g_t. 결정적으로 a,b는 S에 의존 안 하고 토큰에서 미리 계산됨. 펼치면 감쇠 누적합 S_t=Σ_{i≤t}(∏_{i+1..t}η_j)(-θ_i g_i)=momentum 없는 cumsum에 감쇠 붙은 것. affine map 합성이 또 affine이라 combine (a1,b1)•(a2,b2)=(a1a2, a2b1+b2)이 결합법칙 성립→parallel associative scan으로 prefix[S1..SC]를 이진트리 O(log C) 깊이에 계산(순차 O(C) 대신), 총 O(C) work. prefix-sum 일반화=S4/S5/Mamba SSM primitive. gradient g_i는 chunkwise로 chunk-start weight에서 matmul 병렬 계산해 계수 확보. 그래서 momentum은 선형이라 감쇠 붙은 스캔으로 흡수될 뿐 병렬성 안 깸. 대비: S에 비선형이면 scan 불가 — deep MLP 메모리의 weight 재귀는 g_t가 M_{t-1}을 비선형 통과해 진짜 병목(chunk-start freeze로 우회, chunk 클수록 stale→TNT). weight decay 재귀 M_t=(1-α)M_{t-1}+S_t도 선형이라 scan 가능.

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_unknown: momentum이 병렬화되는 근본 이유=선형(affine) 재귀라는 성질 자체 — scan이 병렬화하는 정확한 부류; unknown_unknown: combine 연산자 (a1,b1)•(a2,b2)=(a1a2,a2b1+b2)=affine 합성, 결합법칙이 스캔의 전제; unknown_unknown: parallel scan=이진트리 O(log C) 깊이(순차 O(C)), cumsum/Mamba와 동일 primitive; unknown_known: 진짜 병목은 nonlinear gradient(deep memory)라 chunk-start freeze로 우회; momentum·weight decay는 선형이라 '공짜'로 scan
- 개념 key: associative scan, parallel prefix scan, linear recurrence, affine map composition, momentum, decay-weighted cumsum, combine operator associativity, O(log C) depth, tree scan, Mamba SSM parallelization, chunkwise, precomputable coefficients, nonlinear is the real bottleneck, weight decay linear recurrence
- 생각할 것: Blelloch(work-efficient) vs Hillis-Steele(depth) scan 차이와 실제 GPU/TPU 구현; chunk-start freeze의 stale 근사가 품질에 주는 영향(chunk↑→stale↑) — TNT의 정확한 주제; 이게 Mamba의 selective scan과 정확히 같은 primitive인지(gate=selective A,B); combine (a2b1+b2)에서 b가 weight-shaped라 메모리·대역폭 비용이 스칼라 SSM보다 큰 점(state size)
- storyline seed: momentum 병렬화의 정체: 선형(affine) 재귀 S_t=a_t S_{t-1}+b_t라 combine (a1a2,a2b1+b2)가 결합적→associative scan O(log C)(=Mamba SSM primitive). gate 계수 precompute+gradient chunkwise matmul. 진짜 병목은 nonlinear deep-memory gradient(chunk-start freeze, TNT로 이어짐). 이 조각이 Q003/Q005/Q006/Q008의 '순차성→chunkwise/scan' 실을 메커니즘으로 마감.
- 연상: Q003, Q008

## Q010 · 2026-07-14 · MAC memory write k,v projections (self-supervised)

**Q.** MAC write 컨펌: attn 출력을 NM에 흘려 update한다는데, NM은 k에 대해 v 학습이잖아? attention block 외부에 W_k,v 있고 그걸로 input/golden 놓고 학습? attn-out을 입력에? golden은? 아니면 attn-out을 golden에 놓고 입력은 W_k 출력 k를 흘려? 뭔데

**A.** 사용자가 빠진 연산 정확히 지적. write M_t=M_{t-1}(y_t) 속: 각 attn-out 토큰 u에 대해 k=u W_K, v=u W_V, loss=||M(k)-v||². W_K,W_V는 메모리 자신 투영(attention Q/K/V와 별개, outer-loop 학습, attention block 외부). MLP 입력=k=uW_K, golden(타깃)=v=uW_V. 3안이 맞음(입력=k), 단 golden은 raw attn-out이 아니라 v=uW_V(한 곳 보정). 2안(attn-out 직접 입력) 틀림. self-supervised: 토큰 u가 자기 자신으로 (k,v) 쌍 생성, 메모리는 k→v 학습. Q001 LMM 코어와 동일 메커니즘, 차이는 x_t 자리에 y_t(attention 출력)=attention이 걸러준 표현을 각인. read는 다른 토큰 q=u'W_Q로 꺼냄(q≈저장된 k면 짝 v 회상).

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_unknown: write는 attn-out u에 메모리 자신의 W_K,W_V를 걸어 k=uW_K(MLP 입력)·v=uW_V(golden) 생성 — 내가 이전에 안 펼친 연산; unknown_known: golden은 raw attn-out이 아니라 v=uW_V; MLP 입력은 k=uW_K(사용자 3안이 맞되 golden 보정); unknown_known: self-supervised — k도 v도 같은 u에서(토큰이 자기 (k,v) 쌍 생성); Q001 LMM 코어와 동일, x_t→y_t만 다름
- 개념 key: memory write projections W_K W_V, self-supervised k-v, k=uW_K v=uW_V, associative loss, MLP input is k target is v, attention output as memory input, same as LMM core Q001, memory projections separate from attention, associative recall q matches k
- 생각할 것: MAC에서 retrieve query는 raw segment(S W_Q)인데 write는 attn-out(y_t W_K/V) — 입력 스테이지가 다른 이유; read M*_t(y_t)도 query 투영(y_t W_Q) 거치는지 정확히; self-supervised (k,v)가 linear attention/DeltaNet/TTT 전부 공유하는 원리(토큰이 자기 key,value 제공); attention이 걸러준 y_t를 각인 vs raw token 각인의 실측 차이(memory overflow ablation)
- storyline seed: MAC write 속살: attn-out u → k=uW_K(MLP 입력)·v=uW_V(golden) → ||M(k)-v||². 메모리 W_K/V는 attention과 별개 투영. self-supervised(토큰이 자기 k,v 생성)=Q001 LMM 코어와 동일, x_t 자리에 y_t. read는 다른 q로 연상 회상. 이게 '연상메모리=k→v 자기지도 학습'의 정확한 형태.
- 연상: Q001, Q002, Q007

## Q011 · 2026-07-14 · Titans synthesis / takeaways checkpoint

**Q.** Titans를 읽고 얻는 것 정리 컨펌: 1)token축 gate·S·g 수식 2)attn-out으로 k,v 구해 NM update 3)MAC 전체 연산 4)chunk 순차/chunk내 병렬. 이렇게 맞을까?

**A.** 4개 다 맞으나 전부 '메커니즘' 축. 보정: point1은 gate '값'이 per-token 계산, '생성기'는 outer loop 학습(token축 학습 아님); batch보다 chunk mini-batch. point3 MAC은 3변형 중 하나. 빠진 '의의' 축: A)패러다임=시퀀스 레이어가 test-time optimizer, 메모리=학습시키는 모델(Q001/Q010, NL/Sleep 씨앗) B)통일=DeltaNet/GDN/TTT/RWKV-7/Longhorn을 특수case로(momentum+forget+deep 동시)-가장 크게 빠진 1순위 takeaway(Q003) C)deep(MLP 비선형) memory가 용량 핵심+gradient rank-1 아님→병렬 어려움(Q004) D)시스템 takeaway: state=2P_M(weights+momentum), per-token≈5-6P_M 문맥무관 상수, 고정 RMW상태(KV와 성격 다름), long-context 2M+. 완성 정리=메커니즘4 + 패러다임/통일/deep+시스템.

- 축: `known` → `known` · comprehension: deep
- 새로 드러난 것: unknown_unknown: 사용자 정리는 메커니즘 축만 — '의의' 축(패러다임·통일·deep·시스템)이 빠짐; unknown_unknown: 가장 큰 누락 takeaway=Titans가 family(DeltaNet/GDN/TTT/RWKV-7/Longhorn)를 특수case로 통일(Q003); unknown_known: 'token축으로 학습되는 gate'는 부정확 — 값은 per-token, 생성기는 outer loop(Q005); unknown_known: 시스템 모델링 목표엔 state=2P_M·per-token 상수비용·고정 RMW가 핵심 takeaway
- 개념 key: Titans takeaways, paradigm optimizer-as-memory, family unification, special cases, deep memory capacity, inner outer loop meta-learning, MAC MAG MAL variants, self-supervised write, chunkwise parallel, state size 2P_M, per-token constant cost, systems modeling takeaway, gate value vs generator
- 생각할 것: 이 8행 재정리표가 곧 story line의 Titans 장 골격 후보 — 세미나 재작성 때 그대로 쓸지; '패러다임(optimizer=memory)'이 NL의 'backprop도 memory'로 확장되는 다리; deep memory의 state 크기(2P_M)가 serving에서 KV와 다른 워크로드가 되는 지점(Part III/G09); 3변형 중 MAC이 왜 대표인지, MAG/MAL의 이득·손해 비교는 아직 안 팠음
- storyline seed: Titans 장 골격(8행): 패러다임(optimizer=memory) → update rule(surprise+momentum+forget gate) → 계보 통일(특수case Q003) → 메모리 형태(matrix vs MLP, state 2P_M) → inner/outer loop(meta-learning) → 아키텍처(단기+장기+persistent, MAC/MAG/MAL) → self-supervised write → chunkwise+scan 병렬 → 시스템(상수비용·고정 RMW·long-context). 사용자 4점=메커니즘, 여기에 패러다임·통일·deep·시스템 얹으면 완성.
- 연상: Q001, Q002, Q003, Q004, Q005, Q007, Q008, Q009, Q010

## Q012 · 2026-07-14 · Titans gate computation (shape/operation)

**Q.** 3개 gate는 어떤 shape의 weight를 어떻게 연산해서 scalar를 구하나? 동일 shape·연산이면 1번만 설명.

**A.** 3개 shape·연산 완전 동일, weight 값+squash만 다름(1번 설명 가능). gate=φ(u·w+b): 입력 u=write되는 토큰(standalone x_t/MAC y_t)∈R^d, weight w∈R^d 벡터 하나(+스칼라 bias), 연산=u·w d차원 내적→스칼라, squash φ=η,α는 sigmoid([0,1]), θ는 sigmoid/softplus(>0, v1 미명세). w_θ/w_η/w_α∈R^d, 총 파라미터≈3d(+bias). per-token cost의 'three tiny gate evaluations'=이 세 d-내적. w들은 outer loop 학습, 추론 땐 고정하고 매 토큰 u에서 스칼라만 계산. 멀티헤드면 w∈R^{d×n_head}→head별 스칼라(v1 미명세).

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_known: 3 gate는 동일 구조(w∈R^d 내적+squash), weight·squash만 다름 → 한 번 설명으로 충분; unknown_unknown: gate 입력 u는 write되는 토큰(standalone x_t / MAC y_t)이고, 출력은 토큰당 스칼라(내적 결과); unknown_known: 총 파라미터 ~3d로 매우 작음; per-token 'tiny gate eval'=세 d-내적; outer-loop 학습·추론 고정(Q005 재확인)
- 개념 key: gate computation, linear head, w in R^d vector, dot product to scalar, sigmoid softplus squash, three gates identical structure, per-token scalar, per-head option, outer-loop learned frozen at inference, tiny gate eval cost
- 생각할 것: gate가 per-token 스칼라인지 per-head(w∈R^{d×H})인지 원문 v1 미명세 — Atlas/Miras에서 벡터/diagonal gate로 정교화되는지; squash 선택(sigmoid vs softplus for θ)이 학습 안정성에 주는 영향; gate 입력이 write 대상 u와 같은 것(x_t/y_t)이라는 점 — gate가 '이 토큰을 얼마나/어떻게 각인할지'를 그 토큰 스스로 정함
- storyline seed: gate 3개=동일 구조: 토큰 u∈R^d에 w∈R^d 내적+bias→squash(sigmoid/softplus)→스칼라. w_θ/η/α만 다름, ~3d 파라미터, outer-loop 학습·추론 고정. 토큰이 자기 gate(각인 강도·momentum·forget)를 스스로 정함. per-token 상수비용의 tiny 부분.
- 연상: Q005, Q006

## Q013 · 2026-07-14 · Miras (Moneta A/W dual state, Yaad Huber, channel-wise vector gate)

**Q.** Miras(easy G04): 1)Moneta W식에 W_{t-1}이 없는데 delta W 어케? A와 W 둘다 state? 2)Yaad는 왜 개념수준 수식·설명 없음? 원논문에 있나? 3)Miras layer block gate에 linear matmul 2개(autoencoder처럼)? 그럼 gate값이 scalar 아냐? Titans와 다른가?

**A.** 1)Moneta는 state 두 벌이나 recurrence는 A_t에만: A_t=α_t A_{t-1}-η_t∇ℓ_p(W_{t-1};k,v), W_t=A_t/||A_t||_q^{q-2}. A=accumulator(진짜 recurrent state, A_{t-1} 씀), W=A를 ℓ_q norm 정규화한 파생값(그래서 W_{t-1} 없음). W_{t-1}은 gradient ∇ℓ_p(W_{t-1};·) 안에 들어감. delta W 직접 안 구함—step은 A에, W는 재계산. 진짜 state는 A 하나. 왜? Titans ℓ2감쇠=W공간 곱셈이면 끝, 일반 ℓ_q는 norm projection이라 쌓기(A)→투영(W) 이중구조 필요(q=2면 Titans로 붕괴). 2)원논문에 있음(Huber, 3형태). 내 easy booklet이 Moneta엔 수식주고 Yaad는 표에만—booklet 빈틈(수정함). 실제 Yaad: W_t=W_{t-1}-{η∇ℓ2 if ||M(k)-v||≤δ_t; η δ_t ∇ℓ1 else}. Huber=작은오차 ℓ2민감/큰오차 ℓ1클립(outlier robust, coping mechanism). Moneta와 달리 W_{t-1} 직접 사용. 3)정확: Miras gate는 channel별 벡터(η,δ,α∈R^d), scalar 아님. low-rank projection(2 matmul d→r→d=autoencoder 모양)으로 뽑음. Titans v1=scalar gate와 다름—Miras는 per-channel 벡터로 일반화(Q012 열린실의 답). booklet에 Yaad식+channel-wise gate 명시 추가·재빌드.

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_unknown: Moneta recurrence는 W가 아니라 accumulator A에 있음(W=A의 ℓ_q정규화 파생, W_{t-1}은 gradient 안에); 일반 ℓ_q retention이 norm projection이라 이중 state 필요; unknown_unknown: Miras gate는 channel별 벡터(∈R^d), Titans scalar와 다름; low-rank 2-matmul(autoencoder)이 그 벡터 gate 생성 — Q012 열린실의 답; known 확정(사용자 지적 옳음): 내 easy booklet이 Yaad를 덜 설명(표에만)했음 → Yaad Huber 수식+A/W 명확화+channel-wise gate 추가·재빌드; unknown_known: Yaad=Huber(작은오차 ℓ2/큰오차 ℓ1 clip)=outlier robust coping mechanism, Moneta와 달리 W_{t-1} 직접 사용
- 개념 key: Moneta, dual accumulator A and W, recurrence in A not W, lq norm retention projection, Yaad, Huber loss, gradient clipping, outlier robust, channel-wise gate vector, low-rank projection gate, autoencoder-shaped gate, scalar vs vector gate, Titans vs Miras gate, retention gate, Memora softmax
- 생각할 것: channel-wise(벡터) gate의 이득이 실측에서 얼마인지 — Miras ablation(retention이 최상위 레버)과 연결; Moneta의 A(accumulator) 추가로 state가 matrix의 16배 되는 비용(booklet §7, G09 시스템); Memora의 softmax/log 정규화가 state를 유계로 만들어 저정밀 저장 우호적인 점; Atlas의 gate/optimizer가 Miras보다 더 정교해지는지(Muon, Omega) — 다음 논문
- storyline seed: Miras 핵심: (a)retention을 forget에서 일반 목적으로 재정의, 4손잡이. (b)Moneta=ℓ_p bias+ℓ_q retention, dual state A(recurrent)/W(정규화 파생); Yaad=Huber(outlier robust); Memora=softmax 유계. (c)gate가 Titans scalar→Miras channel-wise 벡터(low-rank 2-matmul). Titans의 특수화를 4축 설계공간으로 연 것. gate 진화(scalar→벡터)가 Q012→Q013 실.
- 연상: Q003, Q004, Q012

## Q014 · 2026-07-14 · Miras online optimization (FTRL / Learning-Retaining)

**Q.** Miras(easy G04)에서 FTRL과 Learning-Retaining 부분은 왜 빠짐? 중요한 거 아냐?

**A.** 맞음, 논문 제목 'Retention, and Online Optimization'의 이론적 척추이고 내 booklet이 사실상 건너뜀(추가함). 온라인 학습 두 갈래(dual): ①Descent(OMD/OGD) W_t=W_{t-1}-η∇ℓ (Titans/DeltaNet, 로컬); ②FTRL W_t=argmin_W[Σ_{i≤t} ℓ̂_i(W)+ (1/η)R_t(W)] (과거 손실 합+정규화 최소화, 글로벌). 원문 증명: Online GD=FTRL 특수case(W_0=0,선형화)→둘이 같음. 결정적: retention gate=FTRL의 정규화 R(W). R 선택=retention 선택: ℓ2=weight decay(Titans), ℓ1=sparse(elastic net), ℓq=Moneta, KL/Bregman=mirror descent(Memora). Learning-Retaining=retention을 Bregman divergence D_h(W,W')로 일반화, 적절한 h면 FTRL 정확 재현→OGD와 FTRL 통합하는 더 일반 렌즈. Moneta A/W 이중구조=FTRL 실현(A=gradient 누적=leader, W=prox_R(A)=argmin)→Q013의 'W에 W_{t-1} 없음' 답 완성(FTRL은 W_{t-1}에서 내려가는 게 아니라 누적 A에 정규화).

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_unknown: retention gate=FTRL의 정규화 R(W)라는 원리적 정체 — ad-hoc gate 아님(ℓ2/ℓ1/ℓq/KL이 다른 R); unknown_unknown: Online GD가 FTRL의 특수case → descent(W_{t-1}에서)와 leader(누적 합 argmin)가 동등; Learning-Retaining(Bregman)이 둘을 통합; known 확정(사용자 지적 옳음): 내 easy booklet이 논문 이론적 척추(FTRL/Learning-Retaining/Online Optimization)를 건너뜀 → 절 추가·재빌드; unknown_known: Moneta A/W 이중구조=FTRL/dual-averaging 실현(A=leader 누적, W=prox_R) → Q013 'W_{t-1} 없음'의 근본 답
- 개념 key: FTRL, Follow-The-Regularized-Leader, online optimization, OMD OGD descent viewpoint, retention as regularizer R, online GD is FTRL special case, Learning-Retaining, Bregman divergence retention, mirror descent, dual averaging, prox operator, Moneta A/W as FTRL realization, retention=regularizer principled
- 생각할 것: FTRL vs OMD 동등성의 정확한 조건(선형화·W_0=0)과 비선형/deep memory에서 깨지는지; Bregman divergence potential h 선택이 Memora softmax(KL)와 어떻게 연결되는지 구체; 이 온라인 학습 프레임(regret bound)이 Miras에 이론 보증을 주는지 — note는 '새 bias/gate에 regret 이론 없음'이라 했음(열린 문제); G01(학습 기초)에 online learning/FTRL을 넣었어야 하나 — prereq 커리큘럼 B2가 이걸 요구했는데 booklet 반영 점검 필요
- storyline seed: Miras 이론 척추: memory update=online optimization. 두 dual 관점 Descent(OMD, W_{t-1}에서)/FTRL(leader, Σℓ+R argmin), Online GD=FTRL 특수case. retention gate=정규화 R(ℓ2 Titans/ℓ1 sparse/ℓq Moneta/KL Memora). Learning-Retaining(Bregman)이 둘을 통합. Moneta A/W=FTRL 실현. 이게 '연상메모리=정규화된 온라인 학습'이라는 Miras 세계관.
- 연상: Q003, Q013

## Q015 · 2026-07-14 · Miras contribution positioning (framework vs invention)

**Q.** Miras 핵심이 설계축 4개=일반화 형태+실험인가? 특별한 발명보다 '설계축이 있음을 증명'인가?

**A.** 맞음, 중심은 단일 발명보다 통합 프레임워크(설계공간)=map/theory 논문. Titans=단일 아키텍처 발명 vs Miras=거의 모든 시퀀스 모델이 한 템플릿(연상메모리가 attentional bias를 retention 정규화와 online 최적화)의 인스턴스임을 보이는 지도. 단 '축 있음 증명'만은 아니고 4개 함께: ①설계축4(bias/retention/구조/algo)=언어 ②이론=FTRL/Learning-Retaining(retention=정규화 원리, Q014) ③새 instantiation Moneta/Yaad/Memora=빈 축(bias objective, retention) 채워 baseline(hybrid 포함)까지 이김=발명(단 framework 파생) ④ablation=retention>deep memory>… 어느 축 중요한지 지목. 발명 성격: ad-hoc 아니라 framework가 가리킨 자리를 채움(top-down 유도) vs Titans bottom-up. framework 가치=트릭더미→체계적 설계공간→후속 방향(Atlas가 Miras가 비운 optimizer 축을 Muon/Omega로 채움). 정직한 한계: 원문 스스로 새 bias/gate에 capacity/regret 이론 없고 chunkwise 근사오차 미정량 인정—강력한 조직화 렌즈+유망한 좌표지 완결 이론+킬러 단일발명은 아님.

- 축: `known` → `known` · comprehension: deep
- 새로 드러난 것: unknown_known: Miras 발명은 ad-hoc 아니라 framework가 가리킨 빈 축을 채운 top-down 유도(Titans bottom-up과 대비); unknown_unknown: '축 있음 증명'만이 아니라 4요소(설계축+FTRL이론+새 instantiation 승리+ablation 지목)가 함께가 핵심; unknown_unknown: framework 가치=후속 방향 제시 — Atlas가 Miras가 일부러 비운 optimizer 축을 채움(지도로 기능); unknown_known: 정직한 한계 — 원문도 새 bias/gate에 capacity/regret 이론 없음·chunkwise 근사오차 미정량 인정(완결 이론 아님)
- 개념 key: framework paper, design space, unification, 4 design axes, generative use of framework, Moneta Yaad Memora as new instantiations, ablation retention most important, top-down vs bottom-up invention, framework guides Atlas, FTRL theory, honest limitation no regret theory, map for the line
- 생각할 것: framework 논문 vs 발명 논문의 학술적 가치 평가 기준(재현·후속 파생·이론 완결성) — 내 스터디 페이퍼 Part III 논지와 연결; Miras가 비워둔 optimizer 축을 Atlas가 어떻게 채우나(Muon/Omega/capacity) — 다음 논문 예고; ablation의 'retention이 최상위 레버'가 이후 Atlas/TNT 설계 선택에 준 영향; 이 '4축 설계공간'이 나중에 Nested Learning의 '중첩 최적화 레벨'로 어떻게 확장/추상화되나
- storyline seed: Miras 위치: 발명 논문이 아니라 프레임워크(지도) 논문. 핵심=설계축4(bias/retention/구조/algo)+FTRL 이론+빈 축 채운 파생 발명(Moneta/Yaad/Memora 승리)+ablation(retention 최상위). Titans(bottom-up 단일발명)와 대비되는 top-down 유도. 라인의 지도가 되어 Atlas(optimizer 축)를 낳음. 정직한 한계: 새 bias/gate 이론 없음. 세미나에서 'Titans=발명 vs Miras=지도'가 좋은 대비 프레임.
- 연상: Q003, Q011, Q013, Q014

## Q016 · 2026-07-14 · Atlas (Omega gamma gate, capacity theory Prop1/2, softmax unbounded)

**Q.** Atlas(easy G05): 1)Omega rule의 γ_{t,i} 어케 학습·shape·기존 3 gate와 동일? 2)feature map 크기 대비 용량이 root라는 증거 어디? Hopfield 아냐? 맞으면 수식·설명 필요. 2-1)softmax attention 용량 무한? 수식 증거? 논문에 명시? hint라도?

**A.** 1)Omega rule Eq9: min_W Σ_{i=t-c+1..t} γ_{t,i}||M(k_i)-v_i||². γ_{t,i}∈[0,1]=input-dependent per-token admission gate(hard pruning, γ→0=토큰 제거). 학습=기존 3 gate(θ,η,α)와 동일 메커니즘(학습 projection+sigmoid, 생성기 outer loop, 값 추론시 계산). shape=window 토큰당 스칼라→step당 c개. 역할은 새 4번째 gate(admission/pruning vs optimizer 스칼라). 2)root=Atlas Proposition 1(자기 정리, Hopfield 아님-Hopfield는 아이디어 계보). Prop1: 정확저장 m쌍=WK=V, vectorize (K^T⊗I)vec(W)=vec(V)=m·d_v 방정식 vs d_k·d_v 미지수, 임의 V엔 key 선형독립 필요→m≤d_k. 파라미터 d_k d_v인데 용량 d_k, d_v≈d_k면 용량=√파라미터=root. Prop2: φ_p(차수≤p monomial, D=C(d_k+p,p)=Θ(d_k^p))→용량 O(d_k^p). Hopfield(Krotov/Ramsauer)='고차 feature가 용량 올림'은 아이디어 출처, Atlas가 Prop1·2로 정식화. 2-1)논문에 명시(hint 아님)+수식 근거: exp(q^T k)=Σ (q^T k)^n/n!, (q^T k)^n=<q^⊗n,k^⊗n>→=<φ*(q),φ*(k)>, φ*(x)=(x^⊗n/√n!)_n 무한차원. Prop2 논리로 용량=∞. 논문이 'attention이 긴문맥 recall에서 고정state 이기는 root 원인'으로 제시. 스펙트럼: matrix O(d_k)→poly O(d_k^p)→softmax φ* ∞. booklet에 Prop1 증명+exp=φ*φ* 유도 추가·재빌드.

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_unknown: γ_{t,i}는 새 4번째 gate(admission/pruning)지만 학습 메커니즘은 기존 3 gate와 동일(projection+sigmoid, outer-loop 생성기); unknown_known: 'root'(용량=√파라미터)는 Hopfield 아니라 Atlas Prop 1(vectorize→m≤d_k 증명); Hopfield는 아이디어 계보만; unknown_known: softmax 무한용량은 논문 명시+수식(exp=Σ(q^T k)^n/n!=<φ*(q),φ*(k)>, φ* 무한차원)—hint 아님; known 확정(사용자 지적 옳음): 내 booklet이 Prop1 증명·exp=φ*φ* 유도 생략 → 추가·재빌드
- 개념 key: Omega rule, gamma admission gate, in-context pruning, per-token scalar gate, same learning as optimizer gates, capacity theory, Proposition 1 matrix O(d_k), vectorization proof, capacity is sqrt of params, polynomial feature map, O(d_k^p), Proposition 2, exponential feature map phi-star, softmax unbounded capacity, exp=phi*(q)phi*(k), Taylor tensor power derivation, dense Hopfield lineage, capacity spectrum
- 생각할 것: Atlas가 실제 쓴 차수 p·sketch를 미공개 → 용량 절대값 계산 불가(booklet §5, G09 state 회계 반쪽); γ admission(입장)과 Miras retention α eviction(퇴장)이 별개 문 — 둘 다 있는 모델의 상호작용; polynomial feature map이 state 크기·per-token 연산을 D=Θ(d_k^p)로 키우는 시스템 비용; c=1 Omega=Transformers with delta rule(unbounded+error-correcting)라는 note의 흥미로운 특수case
- storyline seed: Atlas 3손잡이: Omega rule(γ admission gate, window c개 문맥 기억)+feature map(용량: matrix O(d_k)=√파라미터[Prop1 vectorize 증명]→poly O(d_k^p)[Prop2]→softmax φ* ∞[exp=<φ*,φ*> 유도])+Muon. 용량 스펙트럼이 'attention이 왜 이기나=용량 천장 차이'를 정리로 확정. Hopfield는 아이디어 계보, Atlas가 정식화. Miras가 비운 optimizer 축(Muon)+용량 축을 채움(Q015 실 회수).
- 연상: Q003, Q005, Q013, Q015

## Q017 · 2026-07-14 · grounding check: Atlas gamma shape (not fully specified) + Titans two projection sets (grounded)

**Q.** 1)gamma weight shape 뭔데, 논문에 언급 없나? 2)Titans에서 Wq,k,v가 attention block 내부/외부 2쌍 존재—논문 어디 보고 알수있나, 근거 있나?

**A.** 1)Atlas 논문은 부분만 명시: γ_i^(t)∈[0,1] input-dependent hard pruning, 토큰당 상수 c개({γ_i}_{i=1}^c). 명시 안 됨=γ 생성 projection의 정확한 weight shape('input-dependent parameters'까지만, w∈R^d+sigmoid 같은 형태 v1 미명세, Titans gate도 동일). 정정: Q016의 '기존 gate와 동일 projection(w∈R^d)'은 값이[0,1]·토큰당 c개까지만 근거, 생성기 weight shape은 관례적 추론이지 논문 근거 아님. 2)근거 있음(추론 아님): ①attention W_Q/K/V=§2 line161 K=xW_K,line180 W_Q,W_K,W_V∈R^{d_in×d_in}(표준 attention). ②neural memory W_K/V/Q=§3.1 line365 'W_K,W_V are hyperparameters in the [inner] loss'(메모리 associative loss 정의). ③Fig2(§4) line484-488 three branches core/contextual/persistent, core parameters=in-context learning(attention), contextual=memory 별개 branch. 두 세트 별개는 이 셋 종합(단일 문장 인용은 아니나 근거 명확: §3.1+§2+Fig2).

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: known 정정: Q016의 γ 생성기 weight shape(w∈R^d)은 논문 근거 아님—관례 추론. 논문 근거=γ∈[0,1]·input-dependent·토큰당 c개까지만; unknown_known: Titans 두 투영 세트는 근거 있음—§2(attention line180)+§3.1(memory line365 'hyperparameters in the loss')+Fig2(core/contextual 별개 branch), 단 단일 문장 인용은 아니고 3근거 종합; unknown_unknown: 논문들이 gate/투영 parameterization을 자주 미명세(input-dependent까지만) — '명시 vs 추론'을 습관적으로 구분해야 함(rigor)
- 개념 key: gamma parameterization not spelled out, input-dependent gate, c gates per token, stated vs inferred, honest grounding, Titans two projection sets, attention W_Q/K/V Sec2, memory W_K/V/Q Sec3.1 inner-loss hyperparameter, MAC Fig2 core contextual persistent branches, line 365 hyperparameters in the loss
- 생각할 것: 논문 미명세 항목 목록화(γ shape, gate head 단위, phi* deep-memory 실현, sketch 차원) — 재현/모델링 시 가정 필요; 메모리 W_K/V/Q와 attention W_Q/K/V가 정말 별개인지(공유 안 하는지) 코드/후속 구현으로 확정 가능한지; 이 'stated vs inferred' 습관이 스터디 페이퍼/세미나의 정직성(TODO-VERIFY)과 같은 규율; Atlas γ가 c개 상수라 global(전체 문맥) gate 대비 파라미터 절약이라는 설계 트레이드오프
- storyline seed: Rigor 노드: 논문이 자주 미명세(γ shape, gate 단위 등)—'명시 vs 추론' 구분 필수. 근거 확인법: Titans 두 투영 세트=§2 attention(line180)+§3.1 memory inner-loss hyperparameter(line365)+Fig2 core/contextual 별개 branch. Atlas γ=[0,1]·토큰당 c개는 근거, 생성기 shape은 미명세. 세미나/페이퍼는 이 구분을 TODO-VERIFY로 지켜야.
- 연상: Q002, Q006, Q012, Q016

## Q018 · 2026-07-14 · grounding rigor: MAC equations in text; two projection sets notation is ambiguous/unspecified

**Q.** 결국 Titans 두 투영 세트는 글 근거 없고 그림+설명 추론? gate 통과나 attn out이 W_k,v로 입력되는 것도 그림에 없는데 설명 어딘가 있나? 애초에 nm용 Wq,k,v와 attn용 Wq,k,v를 다르게 할 이유가 있나?

**A.** 정정3개. 정정1: MAC forward는 그림 아니라 §4.1 식으로 명시—Eq21 h_t=M*_{t-1}(q_t),q_t=S^(t)W_Q / Eq22 augment [P||h_t||S] / Eq23 y_t=Attn(S̃) / Eq24 M_t=M_{t-1}(y_t) / Eq25 o_t=y_t⊗M*_t(y_t) + 'updating weight through forward pass'. Q017에서 '그림 종합'이라 한 건 과소평가. 정정2: 하지만 'y_t가 W_K,W_V로 입력'은 Eq24에 재기술 안 됨—M(·)이 뭔지 §3.1 정의(line365 W_K,W_V=inner loss hyperparameter)를 y_t에 적용한 것. Q010 전개는 Eq24+§3.1 결합한 충실한 풀이지 MAC 식에 그 projection 단계는 안 그려짐(그림에도 없음). 정정3(중요): '두 쌍 별개'는 표기 모호—메모리(§3.1)도 attention(§2)도 같은 기호 W_Q/K/V, 논문이 '다른 행렬'이라 명시 안 함(overload). 확실=각자 투영 가짐(근거), 미명세=별개 행렬 vs 공유. Q017 '근거 있음'은 과했고 정확히는 '각자 투영 가짐 근거+별개 여부 미명세'. 이유: 꼭 달라야 할 하드 이유 없으나 묶을 이유도 없음—역할 다름(attention 유사도 vs 메모리 k→v 저장/검색), 입력·단계 다름(검색 raw segment/쓰기 y_t/attention 증강seq, MAC은 y_t 투영이라 애초에 다른 입력), 공유=자유도만 깎는 제약(투영 d×d 작아 절약 미미). 기본=각자 학습.

- 축: `known` → `known` · comprehension: deep
- 새로 드러난 것: known 정정(과소평가 바로잡음): MAC forward는 그림 아니라 §4.1 Eq21-25로 텍스트 명시—Q017의 '그림 종합' 과소평가; known 정정(과대평가 바로잡음): '두 투영이 별개 행렬'은 명시 안 됨(메모리·attention 같은 기호 W_Q/K/V overload)—Q017 '근거 있음'은 과함, 정확히는 '각자 투영 가짐 근거+별개 여부 미명세'; unknown_known: 'y_t→W_K,W_V 쓰기'는 Eq24에 재기술 안 되고 §3.1 M(·) 정의를 y_t에 적용한 풀이(그림·MAC식엔 그 단계 없음); unknown_known: 투영 분리 이유=역할·입력이 달라 묶을 이유 없음(하드 필연 아님); MAC은 메모리가 y_t 투영이라 애초에 다른 적용
- 개념 key: MAC equations 21-25 in text, retrieve Eq21, write Eq24 M(y_t), output Eq25, implicit projection from Sec3.1 def, notation overload W_Q/K/V, separate vs shared not specified, stated vs inferred correction, reason to keep projections separate, different objectives and inputs, tying is unmotivated constraint
- 생각할 것: 논문 표기 overload(W_Q/K/V 재사용)가 재구현 시 '메모리·attention 투영 공유 여부'를 코드 결정사항으로 남김 — 후속 구현(lucidrains 등) 확인; 명시 vs 추론 3단계로 라벨링: (a)식으로 명시(MAC forward) (b)정의 물려받아 풀이(y_t→k,v) (c)미명세(별개 행렬); 이 rigor 습관이 스터디 페이퍼/세미나에서 각 주장에 근거등급 붙이는 규율(TODO-VERIFY, Q017 실 연장); Q010의 self-supervised write 설명은 여전히 유효(정의 물려받은 충실 풀이)—단 '식에 있다'가 아니라 '정의+Eq24로 유도'로 표현해야
- storyline seed: 근거등급 3단계(rigor 규율): (a)텍스트 식 명시=MAC forward Eq21-25; (b)정의 물려받은 충실 풀이=y_t→k=y_tW_K,v=y_tW_V(§3.1+Eq24); (c)미명세=메모리·attention 투영이 별개 행렬인지(기호 overload). Q017/Q018이 '명시 vs 추론'을 매 주장에 라벨하는 습관을 확립—세미나/페이퍼 정직성의 핵심.
- 연상: Q002, Q006, Q010, Q017

## Q019 · 2026-07-14 · cross-paper grounding: projection separation (NL explicit) + attn-out-to-memory (MAC-specific)

**Q.** 다른 논문에서 attn out이 nm의 k,v input으로 쓰인다거나 attention 내부 W와 밖 W가 별도임을 알 수 있는 근거 찾아봐

**A.** (b)투영 분리: 근거 있음, NL이 가장 명시적. NL이 'Transformer projections W_k,W_v,W_q'를 attention 것으로 따로 명명하고, self-modifying Titans에서 메모리 k/v/q가 각자 별도 memory 모듈에서 나옴(k_t=M_{k,t-1}(x_t), v_t=M_{v,t-1}(x_t), q_t=M_{q,t-1}(x_t), η_t=M_η, α_t=M_α)→메모리 생성기가 서로·attention과 완전 별개 객체임을 대놓고 보임(Titans 안 한 disambiguation을 NL이 함). 또 Titans/Atlas MAG(parallel branch)·MAL(memory block before SWA block)=모듈 분리→투영 각자. Atlas DeepTransformers=attention이 φ* 쓴 메모리→attention층·neural-memory층 각자 k,v,q. 결론: Q018 (c)미명세를 NL이 (a)명시로 승격(단 근거는 Titans 아닌 NL). (a)attn out→메모리 k,v: 근거 있으나 MAC 전용. Titans Eq24 M_t=M_{t-1}(y_t)+Atlas도 MAC 재사용(BABILong). 그러나 MAG는 raw 입력 병렬가지, MAL은 메모리 블록이 raw 시퀀스 먼저 처리→'attn out→메모리'는 Memory-as-Context 배치 선택이지 보편법칙 아님. MAG/MAL/비-MAC은 raw 토큰.

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_unknown: NL이 Titans 모호함을 해소 — self-modifying Titans의 k/v/q가 각자 별도 memory 모듈(M_k/M_v/M_q)이라 투영 분리가 명시적; Q018 (c)미명세→(a)명시 승격; unknown_unknown: 'attn out→메모리 k,v'는 보편이 아니라 MAC 전용 — MAG(병렬 raw입력)/MAL(메모리 블록 먼저)/비-MAC은 raw 토큰을 메모리에; unknown_known: Atlas DeepTransformers가 attention=φ* 메모리임을 증명→attention·neural-memory가 같은 종류의 별개 memory 인스턴스(각자 투영); unknown_known: 계열 cross-reference로 한 논문의 미명세를 다른 논문이 명시(Titans 모호→NL 확정) — 리서치 방법으로 유효
- 개념 key: cross-paper evidence, NL self-modifying Titans, M_k M_v M_q separate memory modules, Transformer projections named separately, projections are distinct objects, MAG parallel branch, MAL separate blocks, Atlas DeepTransformers attention as memory, MAC-specific design, attn out to memory, raw input to memory in MAG/MAL, evidence grade upgrade
- 생각할 것: NL self-modifying의 M_k/M_v/M_q가 각자 memory라는 건 파라미터·비용 폭증(투영도 test-time 갱신) — 시스템 관점 G09/serving; lucidrains titans-pytorch 코드로 Titans 자체의 투영 공유/분리를 (a)명시로 확정 가능 — 아직 코드 미확인; MAC이 attention 필터링을 각인하는 vs MAG/MAL이 raw 각인하는 것의 실측 차이(memory overflow, ablation); 이 '한 논문 미명세→다른 논문 명시' 패턴이 6편을 '하나의 완성형'으로 읽는 근거(스터디 페이퍼 수렴 논지)
- storyline seed: Cross-paper 근거법: Titans가 남긴 모호함(투영 분리)을 NL이 해소(self-modifying Titans의 k/v/q=각자 M_k/M_v/M_q, +Transformer projections 별칭). 'attn out→메모리'는 MAC 전용(MAG/MAL은 raw). Atlas DeepTransformers=attention도 메모리. 교훈: 6편은 서로의 미명세를 채우는 하나의 프로그램이라 cross-reference가 유효(수렴 논지). 세미나: 'Titans가 열고 NL이 명시화'.
- 연상: Q002, Q006, Q010, Q018

## Q020 · 2026-07-14 · Atlas takeaways (3 mechanisms + DeepTransformers) + feature map shape pipeline

**Q.** Atlas 정리: 발명 3개 맞나? 1)Omega rule 2)용량 확장 근거+방법(feature map 투영) 3)GD 대신 Muon. 그리고 feature map 투영 시 연산·shape이 NM 입력 단계까지 미명시—찾아서 명시해둬.

**A.** 3개 맞음(+4번째). 1)Omega rule(γ admission gate, window c개). 2)용량=근거(Prop1 matrix O(d_k)=√파라미터/Prop2 φ_p면 O(d_k^p))+방법(key를 feature map φ로 투영). 3)Muon(GD 대신 Newton-Schulz orthogonalized momentum, 근사 2차). +4)DeepTransformers/Dot: deep memory+φ*가 unnormalized softmax attention 엄밀 일반화(attention을 특수case로). Atlas=Miras가 비운 optimizer/용량 축 채운 invention-heavy(Q015). feature map 파이프라인(원문 §feature map: L(M(K);V)→L(M(φ(K));V) 재정의): ①k_t=x_t W_K, W_K∈R^{d_in×d_k}, k_t∈R^{d_k}(v_t=x_t W_V∈R^{d_v}, q_t=x_t W_Q∈R^{d_k}) ②φ_p(k_t)=[k_t^β]_{|β|≤p}∈R^D, D=C(d_k+p,p)=Θ(d_k^p) ③NM 입력=φ_p(k_t)(raw k_t 아님), M:R^D→R^{d_v}(matrix면 M∈R^{d_v×D}) ④손실 ||M(φ_p(k_t))-v_t||² ⑤읽기 M(φ_p(q_t)). 입력차원 d_k→D가 용량 O(d_k)→O(d_k^p) 원리. 미명세: D=Θ(d_k^p) 폭발적, sketch로 줄이나 정확한 p·sketch 차원 미공개→절대 D 계산불가. booklet에 shape 파이프라인 추가·재빌드.

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_unknown: NM 입력은 raw k_t(d_k)가 아니라 올린 φ_p(k_t)(D=Θ(d_k^p)) — 입력차원 확대가 용량 확대의 원리; unknown_known: Atlas 4번째 기여 DeepTransformers/Dot(deep memory+φ*가 softmax attention 일반화) — 사용자 3개에 빠짐; known 확정(사용자 지적 옳음): feature map 투영 shape이 booklet 미명시 → x_t→W_K→k_t→φ_p→NM 입력 파이프라인 추가; unknown_known: 절대 D 미명세(sketch·p 미공개)=근거등급 (c); L(M(φ(K));V) 재정의는 (a)텍스트 근거
- 개념 key: Atlas 3 mechanisms, Omega rule, capacity theory Prop1 Prop2, feature map method, Muon Newton-Schulz, DeepTransformers Dot, attention as special case, feature map pipeline shapes, k=xW_K d_in×d_k, phi_p(k) in R^D, D=Theta(d_k^p), NM input is phi_p(k) not raw k, memory M R^D to d_v, sketch dimension undisclosed, L(M(phi(K));V) redefinition
- 생각할 것: sketch(PolySketchFormer류)가 D=Θ(d_k^p)를 실제 얼마로 줄이나 — 원문 미공개, 후속/코드 확인; φ_p로 올린 뒤 메모리가 matrix면 M∈R^{d_v×D}로 파라미터 폭증 — Muon/feature map/state 비용의 상호작용(G09); DeepTransformers가 attention을 일반화한다는 게 Q016 용량 스펙트럼(softmax=φ* ∞)과 같은 이야기; Omega의 φ(k) 사용 update W_t=W_{t-1}(αI-Σγφφ^T)+Σγvφ^T이 feature map+admission을 한 식에 결합
- storyline seed: Atlas 정리: 3기계(Omega rule/feature-map 용량[Prop1,2 근거+φ 방법]/Muon)+4번째 DeepTransformers(attention=deep memory+φ* 특수case). feature map 파이프라인: x_t→k_t=x_tW_K(d_k)→φ_p(k_t)(D=Θ(d_k^p))=NM 입력→M:R^D→R^{d_v}. 입력차원 d_k→D가 용량 확대 원리. 절대 D는 sketch 미공개로 계산불가. Miras 빈 축(optimizer/용량)을 채운 invention-heavy.
- 연상: Q013, Q015, Q016, Q017

## Q021 · 2026-07-14 · TNT takeaways + serving (structure in paper, cost not; connects to Part III)

**Q.** TNT 정리: 1)train/inference chunk size 같아야 함 증명 2)chunk 최적+global,local memory 3)2 stage training 4)Q-K projection. 맞나? 근데 serving 시점에 어케 되는지 논문에 없나? global/local memory serving에도 배포? Q-K projection inference에도?

**A.** 4개 맞음(1번 보정). 1)train/serve chunk mismatch 발견·해소: chunk64 학습→chunk8 decode면 ppl 2.6배 악화(Challenge3). 해법=다 같은 chunk가 아니라 train은 큰 chunk(효율)/최종은 decode chunk(=1)에 정렬=train·serve chunk decouple+Stage2 정렬. 2)hierarchical global(C_G=2048 순차)/local(N개 병렬 reset-to-W_init). 3)2-stage(Stage1 효율 pretrain, Stage2 ~5% 고해상도 fine-tune). 4)Q-K projection(key로 쓰고 query로 읽는 mismatch, Challenge2). TNT=아키텍처 아니라 training 레시피. Serving: 구조는 논문에 명시(성능 수치는 없음). Stage2 목표 C_L'=1=prefill-and-decode 패턴: prefill=global memory 큰 병렬 chunk로 프롬프트 흡수, decode=local memory 토큰별(chunk1) 갱신. Q직접답: global·local 둘다 serving 배포(global=prefill/local=decode, +Q-K covariance까지 상태로), Q-K projection은 읽기(Eq7)의 일부라 inference(prefill·decode)마다 함, 메모리는 training·inference 모두 online 갱신(line203). 미측정: decode wall-clock/throughput/footprint 벤치 없음(유일 수치는 32K prefill서 FlashAttn 1.3x, decode latency 아님). 이게 내 스터디 페이퍼 Part III(claim1-6, A1 'decode wall-clock 미공개')가 메우는 지점: serving state=global V+local W+Q-K covariance를 매 decode 토큰 RMW→memory-bound(claim1), KV와 다름(claim2), 대역폭이 배치 조기 닫음(claim3). TNT 구조+내 Part III 비용=완성.

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_unknown: TNT 해법은 '같은 chunk'가 아니라 train(큰)/serve(1) chunk decouple+Stage2 정렬 — 사용자 '같아야 함' 보정; unknown_known: serving 구조는 논문 명시(prefill=global/big-chunk, decode=local/chunk1, Q-K projection은 읽기라 inference마다) — global·local 둘다 배포; unknown_unknown: serving 성능(decode wall-clock/throughput/footprint)은 TNT도 미측정 — 유일 수치는 prefill FlashAttn 1.3x; unknown_known: 이 serving 비용 공백이 정확히 내 스터디 페이퍼 Part III(claim1-6,A1)가 메우는 지점 — TNT 구조+Part III 비용=완성
- 개념 key: TNT training recipe not architecture, train/serve chunk mismatch 2.6x ppl, decouple train chunk from serve chunk, Stage2 align to decode chunk, hierarchical global local memory, reset to W_init, 2-stage training, Q-K projection retrieval, prefill global big-chunk, decode local chunk-1, both memories deployed at serving, Q-K projection at inference, memory updated online at inference, serving structure in paper cost not benchmarked, connects to Part III claim1-6 A1, decode wall-clock gap
- 생각할 것: serving state=global V+local W+Q-K covariance 총합 크기 — G09/Part III state 회계에 TNT 계층 반영; reset-to-W_init가 context parallelism(학습)엔 좋은데 serving decode(순차)엔 어떤 의미인지; prefill(global 큰 chunk)과 decode(local chunk1)의 비용 비대칭 — 내 Part III prefill/decode split과 정합; TNT가 serving 벤치 안 한 것=A2(chunk mismatch 재현·확장) 실험 여지, 내 페이퍼가 이미 지목
- storyline seed: TNT=training 레시피(발명 아님): train/serve chunk mismatch(2.6x ppl) 해소를 위해 train chunk(큰)/serve chunk(1) decouple+Stage2 정렬, hierarchical global(prefill)/local(decode) memory, Q-K projection(읽기 mismatch). Serving 구조는 논문에 있음(global·local 둘다 배포, Q-K projection inference마다, 메모리 online 갱신 계속)—단 decode 비용 수치는 미측정. 이게 내 Part III(serving 비용 실측)가 메우는 지점: TNT 구조+Part III 비용=완성. 세미나: 'TNT가 serving 구조를 정의하고, 내 기여가 그 비용을 잰다'.
- 연상: Q003, Q008, Q009, Q015, Q020

## Q022 · 2026-07-14 · Part III credibility / experiments

**Q.** NL 보는중 — 본서 저자 자체 실험은 얼마나 신뢰성 있지? 우리 HOST에서 실험한거지? HATIR로 실험한 결과야? 신뢰할 만하다는 걸 어케 증명? 실험환경 포함 재현방법도 적어서 신뢰성 증명해야 할듯

**A.** 두 종류다: (a) HATIR roofline @ dgx_h100_x4 twin(analytical) — E1.1/1.2/1.3/1.4/1.5, (b) host CPU numpy 실측 — E2.1/2.2. 이 host엔 NVIDIA GPU가 없어 실제 GPU wall-clock은 0줄. 그래서 신뢰성은 '쟀다'가 아니라 4개다: 결정론 재현(analytical 6개+E1.2 crossover=재실행 bit-identical), 3중 교차검증(E3 closed-form/hatir/plan hand-calc가 anchor에서 <1% 일치), host CPU로 roofline 모양 확인, 모든 숫자 정직 등급화+A100 이월(반증)목록. 유일 비결정성(E1.2 spill sim의 set tie-break=PYTHONHASHSEED 의존)을 찾아 seed=0으로 고정. 단 교차검증은 '우리 계산이 옳다'만 증명하고 twin이 실제 silicon과 맞는지는 증명 못함(같은 twin BW 공유)→A100 runbook 이월.

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_unknown: 이 host엔 GPU가 없어 절대치는 전부 roofline '하한'이지 측정이 아님; unknown_known: 신뢰성의 진짜 근거는 측정이 아니라 결정론적 재현+3중 교차검증+정직한 등급화; known_unknown: twin fidelity(twin이 실제 H100과 얼마나 맞나)는 교차검증으로도 안 풀려 A100 실측으로만 메움
- 개념 key: exploration-grade, roofline lower bound, HATIR twin, cross-validation, determinism, PYTHONHASHSEED, A100 runbook, honesty contract, no-GPU-host
- 생각할 것: 교차검증이 '계산 정확'만 증명하고 twin fidelity는 별개라는 구분 — 세미나에서 이 두 층을 분리해 보여야 과대주장 안 됨; hatir upstream: live_set_spill_bytes tie-break을 안정 정렬로 고쳐 seed 없이도 결정론화(현재는 seed 고정으로 우회); A100 runbook 6개 이월 항목이 곧 '다음에 실제로 재야 할 것' 리스트 = Part III-a 실행 계획
- storyline seed: Part III(저자 기여)의 신뢰성 서사: '측정했다'가 아니라 '결정론적으로 재현되고, 3개 독립 경로가 <1% 수렴하며, 무엇이 아직 증명 안 됐는지를 먼저 밝힌 exploration-grade warrant'. 부록 E/REPRODUCE.md가 그 증거. 세미나에선 '왜 GPU 없이도 이 숫자를 믿나'를 이 4기둥으로 답한다.
- 연상: Q021

## Q023 · 2026-07-14 · Part III credibility / HATIR + HAT spec fidelity

**Q.** 핵심은 HATIR이 믿을만하냐, 거기서 쓴 HAT hw spec이 실제 hw를 얼마나 정교하게 표현하고 얼마나 정확한 spec data이냐일 것 같다. 이 신뢰성 검증을 g07 pdf에도 포함해야 함. 발표에 쓸 거라 성능 수치에 매우 민감함.

**A.** 신뢰성을 두 층위로 분해해야 함. (A) HATIR 계산: ZigZag(KU Leuven 독립 학술 oracle) 대비 operand별 DRAM traffic byte 0.00%(byte-exact) + MAC 정확 + bound 분류 일치로 검증됨 — 단 traffic 계수/구조까지이고 절대 에너지 pJ는 제외. (B) HAT spec: NVIDIA whitepaper/DGX guide/Hopper tuning/CUTLASS 출처 = datasheet peak-nominal이지 실물 측정 아님, 필드별 provenance 태그는 일부만. 결정적 gap: raw roofline은 100% util 가정 → 실측 H100 GEMM 959 vs 716 TFLOPS(~34% 과대), batch-1 decode MBU 26.9%. fit 상수는 '없는 칩에 몰래 수입'이라 방법론적으로 거부하고 provenance-typed(computed/technology/stack/residual)로 분해. 결론: 상쇄되는 양(S*=순수 traffic 등식 GB=GB→BW·MFU/MBU 둘 다 상쇄, bound 분류=2배 오차도 안 뒤집힘, tier 순서)만 단정하고 절대 성능(µs/mJ/GB·s)은 유보.

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_known: 신뢰성은 '두 층위'로 갈림 — 계산 정확(ZigZag로 검증됨) vs spec fidelity(datasheet peak, 미검증); unknown_unknown: raw roofline은 ideal(100% util)이라 실측 대비 GEMM ~34% 과대·decode MBU 26.9% — 절대치는 1.3~1.4배 낙관적; unknown_known: load-bearing 양이 견고한 진짜 이유 = 모델 오차가 비율/crossover에서 상쇄됨(S*는 GB=GB 등식이라 BW값도 MFU/MBU도 상쇄); known_unknown: fit 상수 거부 + provenance-typed 분해가 이 프로젝트의 방법론적 입장(발표에서 이걸 강점으로)
- 개념 key: HATIR, ZigZag crosscheck, byte-exact, HAT twin, datasheet peak, MFU, MBU, ideal vs achieved, provenance-typed calibration, fitted constant 거부, cancellation in ratios, S* traffic equality
- 생각할 것: 발표 핵심 슬라이드: '왜 GPU 없이 낸 숫자를 믿나' → 두 층위(ZigZag byte-exact 검증 vs datasheet spec) + 상쇄 논증(S*는 스펙 오차에 불변)으로 답; 정직한 약점 2개를 먼저 밝히기: twin 필드별 provenance 태그 일부만 + calibrated=no(core 경로는 ideal). 이걸 숨기지 말고 '그래서 절대치는 유보'로 연결; twin fidelity를 실제로 좁히려면: A100/MI300X로 provenance 항 도출 → H100 leave-one-chip-out 예측오차 보고(사내 runbook). 이게 절대치 승격의 유일 경로
- storyline seed: 세미나에서 성능 수치 신뢰성 파트: (1) 두 질문 분리 — 계산이 맞나(HATIR/ZigZag byte-exact O) vs 스펙이 실물이냐(datasheet peak, 미측정). (2) ideal-vs-실측 34% gap을 정직 공개. (3) 그럼에도 결론이 서는 이유 = 우리가 단정하는 양(crossover/bound/tier)은 모델 오차가 상쇄되는 양뿐이고, 상쇄 안 되는 절대 성능은 유보. 이 절제가 곧 신뢰성. G07 §14 + 부록 E가 근거.
- 연상: Q022, Q021

## Q024 · 2026-07-14 · multiarch cross-architecture study

**Q.** 더 다양한 hw twin에서 결과를 뽑자 — 동일 실험을 8개 아키텍처(H100/B100/VeraRubin/최신TPU/Meta/AMD/Groq/Cerebras)에서 동일하게. 각 hw spec을 다중검색으로 100% 신뢰 수치·die/PE 내부까지 구해 schema 만들고, HATIR로 개별 실험. 리뷰 고찰 여러번 걸쳐 진짜 맞나 검증.

**A.** 신뢰도 등급제로 8개 twin 저작(다중검색+이중적대검증 workflow 28 agents, +결과재검증 10 agents). 등급: h100/mtia2 GOLD, b100/mi355x/tpu-v7/groq/wse3 SILVER, vr100 BRONZE(내부 미공개→directional). driver는 E3 closed-form을 twin-무관 파라미터화, H100에서 확정 anchor <0.02% 재현으로 게이트. 3대 발견: (1) S*=65536 전 아키텍처 동일=HW무관 workload 속성(상쇄 논증 실증) (2) decode 7/8 memory-bound(여유 380~1455×), 유일 예외 Cerebras는 wafer SRAM 21PB/s가 dense 12.5PF 대비 커서 roofline knee (3) 절대 decode 0.0003ms(Cerebras)~31.5ms(MTIA LPDDR) 5자릿수=state 배치가 축, SRAM-heavy가 병목 해소. 리뷰가 wse3 sparse(125PF)→dense(12.5PF) 오류를 잡아 knee 발견이 드러남.

- 축: `unknown_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_unknown: bound 분류가 하드웨어 공간 전체에서 견고하되 극단적 BW-rich(wafer-scale)에서는 knee로 이동 — 경계 자체가 결과; unknown_known: S*가 HW 무관(workload 속성)임이 8개 실측으로 실증 = 상쇄 논증의 강한 증거; known_unknown: 절대 성능은 5자릿수 갈리고 축은 'state가 어디 사느냐'(SRAM/HBM/LPDDR); unknown_unknown: sparse/dense FLOPS 혼동이 결론(bound)을 뒤집을 수 있음 — 리뷰가 잡음(신뢰성 절차의 가치)
- 개념 key: hw twin, confidence tier, GOLD SILVER BRONZE, S* HW-invariant, roofline knee, Cerebras wafer SRAM, Groq LPU, MTIA LPDDR, sparse vs dense FLOPS, state placement, provenance, HATIR parameterized
- 생각할 것: 발표 핵심: '동일 실험 8칩'으로 pair thesis의 견고성(S* 불변·bound 견고)과 한계(Cerebras knee)를 동시에 — 견고+경계를 같이 보여야 정직; state 배치 축: SRAM-heavy(Cerebras/Groq)가 decode-state 병목을 구조적으로 해소 = memory-centric 논증의 하드웨어 증거; vr100(BRONZE)은 발표에서 '예비'로만. GOLD/SILVER 7칩으로 결론 서고 Rubin은 미래 참조; 절대치는 여전히 ideal 하한(부록 E) — 8칩에서도 단정 안 함, 비율/crossover/bound만
- storyline seed: 세미나 '신뢰성' 파트의 정점: 성능 수치를 8개 아키텍처에서 뽑아, 우리가 단정하는 양(S* 불변·bound 견고)이 하드웨어 전체에서 성립함을 보이고, 동시에 그 경계(Cerebras knee)와 절대치 유보를 정직히 드러낸다. 2라운드 적대검증(spec 저작+결과)이 신뢰성의 증거. multiarch/REPORT.md + REVIEW-LOG.md가 근거.
- 연상: Q023, Q022

## Q025 · 2026-07-14 · Nested Learning (HOPE) — optimizers + self-mod mechanism

**Q.** NL(G07) 심층: gradient/momentum/M3가 Miras·Atlas(omega)를 참조해 전개되는 쉬운 설명+shape+state / ICL이 우연 아닌 당연? / M3 retention·다주기 조합? / DGD의 dot-product→L2 전개? / HOPE는 3개 다 씀? / self-mod Titans의 k,v,q,η,α 중 α 적용범위·optimizer / self-mod 연산·학습 목표값 메커니즘

**A.** 3 optimizer 계보: DGD←Miras(L2 attentional bias+retention α), DeltaMom, M3←Atlas Muon(NewtonSchulz)+CMS 다주기+Adam2차. state 배수 DGD 1×/DeltaMom 2×/M3 3.5×. M3는 fast(매스텝)+slow(f스텝마다) momentum 조합(님 추측 맞음), theta retention은 M3에 없고 DGD update에 있음. DGD: 내적규칙 ℓ=-⟨Mk,v⟩→L2회귀 ‖Mk-v̂‖², weight decay 붙이면 M_t=M_{t-1}(αI-η kkᵀ)-η∇L. HOPE는 3개 다 쓰는 게 아니라 inner=DGD 하나, M3는 별도 훈련 optimizer 제안. self-mod 메모리 6개 {k,v,q,η,α,mem} 각 2-layer residual MLP, q만 static(line 2116). α·η는 '함수'가 아니라 학습되는 메모리(메타러닝 init+test-time DGD), 출력이 하이퍼파라미터. ICL은 창발 아니라 structural(≥2 레벨의 결과, 논문 명시).

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_known: HOPE inner=DGD 하나, M3는 별도 optimizer 산출물(3개 다 쓴다 아님); unknown_unknown: M3=fast+slow momentum 다주기 조합, retention은 DGD에만; known_unknown: α·η도 완전히 학습되는 메모리(메타러닝+test-time), '함수'라 한 건 부정확
- 개념 key: DGD, Delta Momentum, M3, Muon, NewtonSchulz, Miras attentional bias, retention gate, self-modifying Titans, 6 memories, q static, ICL structural, meta-learned init, associative memory optimizer
- 생각할 것: multiarch attention-side를 Wqkv(3d²)로 잡았는데 q static이라 Wk,Wv(2d²)로 정정 여지; optimizer별 decode state 배수(DGD 1/M3 3.5)가 hope-* 실험의 근거
- storyline seed: NL 세미나: optimizer=memory 프레임이 DGD/DeltaMom/M3를 낳고(Miras/Atlas 이식), self-mod Titans는 투영·게이트까지 메모리화(q만 static). ICL은 ≥2레벨의 structural 결과. G07 §5·§8·§10 반영됨.
- 연상: Q024, Q021

## Q026 · 2026-07-14 · self-mod Titans mechanism + inference execution

**Q.** self-mod Titans 연산 재검토: v_t가 M_v로 얻은 값? 각 기호 정체·shape / 자가목표 각 5개 어떻게 / gradient g는 autograd인가 diff vector인가 / q는 M_q로 self-update 아닌가 / 추론시 forward+update 스케줄: FFN(CMS) 끝나기 전 update 완료 가능? 논문이 어케 했나

**A.** v_t=M_v,t-1(x_t) 맞음(식 83; 보통 x_tW_v를 메모리로 승격). q_t=x_tW_q static(line 2116, 유일 non-adaptive; 식 88 집합에 q 있지만 카브아웃). 자가목표 v̂_□=M_□,t-1(v_t) 각 5개 자기 MLP 통과. gradient=L2 loss를 2-layer MLP 전체에 대해 autograd/backprop(diff 아님; 오차는 backward 시작점, 선형이면 (Mk-v̂)kᵀ로 줄지만 MLP면 두층 관통)+retention 항 동반. 출력 o_t=M_mem,chunk시작(q_t)=pre-update 읽기, 활성 σ는 메모리 MLP 내부, 뒤에 CMS chain. 추론 실행: 출력경로(read+CMS)는 pre-update라 이번 토큰 update와 의존성 없음→겹침 가능(순서 안 걸림). 진짜 임계경로는 토큰 간 update(다음 토큰 read가 M_t 필요). 논문 §8.2는 훈련 chunkwise 병렬화만; decode kernel/latency/wall-clock 전무(inference는 line 458 대조로만)=Part III 공백. 부하: 토큰·layer당 수십 작은 GEMV+backward+의존성, kernel 1개 불가.

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_known: 출력은 pre-update(chunk-start) memory를 읽어 update와 순서 안 걸림; 임계경로는 토큰 간 update; unknown_unknown: 논문은 훈련 병렬화만 풀고 decode 실행은 안 풂 = Part III가 메우는 지점; unknown_known: gradient는 diff가 아니라 MLP full backprop + retention 항
- 개념 key: v_t=M_v(x_t), self-target, full backprop gradient, pre-update read, chunk-start snapshot, training chunkwise parallel, decode sequential, update-CMS overlap, memory-bound RMW, Part III gap, grouped-GEMM batchability
- 생각할 것: decode 임계경로=토큰간 순차 update가 E1.1b memory-bound와 정합; update↔CMS overlap을 pipelining으로 잡으면 decode latency 모델에 반영
- storyline seed: 세미나 핵심 슬라이드: '논문은 훈련 chunkwise 병렬화만, decode 실행은 미해결'. 출력이 pre-update read라 update↔CMS는 겹치지만 토큰 간 update가 순차 임계경로=memory-bound. 이게 내 Part III(decode 비용 계량)가 메우는 지점. G07 §10 '추론 실행' 절 반영.
- 연상: Q025, Q021

## Q027 · 2026-07-15 · Sleep — memory consolidation basics

**Q.** Sleep consolidation 기초: figure의 MLP·low-rank MLP 가로줄이 같은 update 주기 FFN? MLP shape(d×d vs d×d_low)·개수? sleep 과정 1,2,3 단계·high/lower freq 영역·teacher/student·복사 여부·lifecycle? consolidation=wake→sleep 전환이라 추론 불가? 아니면 그 FFN만 빼서 동시 distill?

**A.** CMS=MLP블록 사슬, 각 블록=sparse MoE(router R^(f_ℓ)+experts {W,1..s_ℓ}). 한 가로줄=한 블록(같은 주기 f_ℓ). regular expert=full FFN, 추가 expert=low-rank {A∈R^{d×d_low},B∈R^{d_low×d}, d_low≪d}(d×d 아님). 개수 s_ℓ은 consolidation마다 +1(미리 할당+masking, 활성화 시 unmask). 과정: (1)Param Expansion=느린 블록 ℓ*에 low-rank expert 하나 append(FFN 통째 복사 아님). (2)Knowledge Seeding=upward distillation, teacher=LM_θ(확장전), student=LM_θexp(확장후), GKD(on-policy)+RL LTI(dream prefix 이어쓰기, semantic+Levenshtein reward), 새 expert만 학습·나머지 freeze. (3)Synaptic pruning=ℓ*−1의 과거 low-rank expert reset. nested 주기라 fast→slow consolidation 다중(10K/1K면 10회). wake→sleep은 offline 별도 phase(외부입력 제한)=모드전환, concurrent serve+consolidate는 논문 밖 systems 여지(구조상 새 expert만 학습이라 가능성 있음).

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_known: 새 expert=low-rank(d×d_low)이고 FFN 통째 복사 아님; 한 줄=한 주기 블록; unknown_known: teacher/student는 전체 모델 지칭하되 학습은 새 low-rank expert만(나머지 freeze); unknown_unknown: concurrent serve-while-consolidate는 논문 미규정 = serving-engineering 공백
- 개념 key: CMS, MoE expert, low-rank expert, d_low, Knowledge Seeding, upward distillation, GKD, teacher LM_θ, student LM_θexp, synaptic pruning, masking activation, wake/sleep offline, parameter expansion
- 생각할 것: concurrent serve+consolidate 가능성=새 expert만 학습·한 블록씩이라 구조적으론 백그라운드화 가능, 논문 밖; expert pool 성장이 곧 total param 성장(§5 scaling과 연결)
- storyline seed: Sleep=wake/sleep lifecycle. sleep offline에서 fast block 지식을 느린 블록에 low-rank expert로 seed(GKD+RL), 새 expert만 학습, fast block reset. decode wall-clock처럼 concurrent serve+consolidate도 논문 미규정=systems 공백.
- 연상: Q024, Q021

## Q028 · 2026-07-15 · Sleep — consolidation internals (router/on-policy/optimizer)

**Q.** Sleep 심화: step2 on-policy rollout 설명 없음. teacher/student 출력 다 나와야 학습→router는 어케? teacher는 추가expert mask/student는 unmask? router가 학습대상 low-rank expert 안 고를수도 있는데 강제 선택? step3 reset '빠른 블록 ℓ*−1의 과거 low-rank expert'는 언제 추가됐나(step1은 ℓ*에 추가인데)? 3-1 가장 빠른 1K FFN엔 추가 안되나? 4 low-rank만 학습=거기만 backward? gradient/optimizer? memory 유지?

**A.** 1)on-policy: GKD 목적=(1−λ)teacher-forced + λ E_{y∼LM_θexp}[F] = student가 자기 분포로 rollout한 시퀀스 위에서 teacher divergence 최소화(train/serve mismatch 해소), student 샘플링 분포로는 backprop 안 함. 2)router: 새 expert 미리 할당+forward/backward masking, sleep에 unmask; teacher(masked)/student(unmasked). top-k router가 갓 초기화 expert 안 고를 문제→논문은 router가 expert 강제 추가선택(line479, dreaming서 명시; consolidation의 강제규칙은 c등급)+새 expert만 trainable. 3)(De)Activation(line163-6): faster block 파라미터 deactivate+current block 새 파라미터 activate. ℓ*−1의 low-rank expert는 이전 sleep에 ℓ*−1이 target(ℓ*−2→ℓ*−1)일 때 추가된 것=consolidation-추가분을 reset(님 추측 맞음). 3-1)가장 빠른 MLP(1K)도 그 위 더 빠른 memory(attention/ICL)로부터 low-rank expert 받음; 순수 source는 최상단 attention뿐(formalism은 MLP↔MLP 명시, attention→L1 경계는 함축 b/c). 4)backward는 새 expert만; 증류=backprop, RL/LTI=policy-gradient(샘플링 backprop 안함); optimizer 원문 미명시(c, arbitrary optimizer/Adam급 함축); 이 sleep학습 optimizer momentum은 offline outer 임시버퍼(persistent test-time memory 아님), 남는 건 expert weight.

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_known: on-policy=student 자기 rollout 위 teacher divergence; 샘플링 backprop 없음; unknown_known: router가 새 expert 강제선택해야 gradient 감(dreaming서 명시, consolidation은 c); known_unknown: sleep 학습 optimizer 원문 미명시; momentum은 임시(persistent memory 아님)
- 개념 key: on-policy rollout, GKD lambda, router forced expert, masking/unmask, (de)activation, reset consolidation experts, fastest block boundary, policy gradient LTI, low-rank only backward, transient optimizer state
- 생각할 것: consolidation 시 router 강제선택 규칙이 원문에 dreaming만큼 명시 안됨=구현 재현 시 확인 필요; sleep optimizer(Adam?) 미명시—재현 runbook에 가정으로
- storyline seed: Sleep consolidation 내부: on-policy(student rollout)+RL(policy-grad), router 강제 expert선택으로 새 low-rank expert에 gradient 흘림, 나머지 freeze. reset=(de)activation의 deactivate side. optimizer/momentum은 offline 임시. 세미나: '새 expert만 backward, RL은 policy-gradient, memory는 안 남고 weight만 남는다'.
- 연상: Q025

## Q029 · 2026-07-15 · Sleep data provenance (offline) + NL online consolidation

**Q.** Sleep는 전부 offline 학습? offline시 각 단계(consolidation, dreaming)별로 어떤 data를 어떻게 만들어 주는지 데이터 확보 방식을 논문 근거에 한정해 확실히. 모르는 부분은 모른다고+추정. 그리고 Hope에 online consolidation 언급—NL 다시 보고 CMS의 online consolidation 어떻게 하는지 확인·설명.

**A.** Sleep=전부 offline, 외부입력 안받음(line325), 외부데이터 접근제한(390), raw replay 아니고 self-generated. Consolidation: 증류D=teacher LM_θ 자샘플(399), on-policy=student LM_θexp rollout(λ항), LTI=teacher dream+random prefix→semantic+Levenshtein reward. Dreaming: sampled task(C,τ)에서 DREAM∼LM_θ(·|C), router가 random expert 강제선택해 novel 지식 주입(479), importance g_DR=∇L_SFT로 top-k+b random 선택, 각 dream을 isolated 인스턴스에 LoRA SFT→성능개선시 reward1(SEAL)→ReST^EM. 미명시(c): consolidation teacher 샘플링 prompt 조건, dreaming task(C,τ) 분포 출처(τ=downstream 측정이라 target task 추정). NL online consolidation=명시(line119 'memory consolidation as online process'). 메커니즘(1724-9): CMS 레벨 간 지식전달=빠른→느린 레벨로 memory 초기상태 meta-learning; 느린 레벨이 W_init 학습, context 경계마다 빠른 레벨 reset→지식 consolidate. deep memory(Titans/Atlas/Miras/TTT)는 함, linear은 없음. 고정용량이라 한계→Sleep offline(param 성장+distill)이 확장. 초기상태 학습 자체는 outer backprop(inference 실시간 아님).

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_known: Sleep 데이터는 전부 self-generated(teacher/student 자샘플, dream), 외부 raw 없음; unknown_unknown: consolidation prompt 조건·dreaming task(C,τ) 분포 출처는 원문 미명시(c); unknown_known: NL online consolidation=초기상태 meta-learning+reset(고정용량); Sleep offline=param 성장+distill로 확장
- 개념 key: offline self-generated data, teacher sampling D, on-policy rollout, LTI dream prefix, dreaming task (C,tau), router random expert, gradient importance selection, LoRA SFT ReSTEM, NL online consolidation, meta-learned initial state, W_init reset, fixed capacity limit
- 생각할 것: dreaming task(C,τ) 출처 미명시—재현 시 target downstream task 분포로 가정 필요; NL 초기상태 meta-learning은 outer backprop이라 online consolidation이 inference 실시간 학습은 아님—명명 주의
- storyline seed: 두 consolidation의 대비: NL online=초기상태 meta-learning+context 경계 reset(고정용량, deep memory만), Sleep offline=self-generated distillation으로 low-rank expert 성장(용량 확장). 데이터는 둘 다 self-generated(외부 없음), 유일 앵커는 dreaming task(C,τ)인데 출처 덜 특정. 세미나: wake=online(reset), sleep=offline(성장).
- 연상: Q027, Q028, Q025

## Q030 · 2026-07-15 · CMS multi-frequency update + inter-level transfer (Eq 70-72)

**Q.** CMS low/mid/high 3레벨(low=1K, mid=5K)에서 0~1K 구간 메커니즘과 1K 경계 학습이 어케 되나. 내 추측: low가 0~1K stacking→한번에 학습? 또는 token/chunk별 gradient 구해 1K에 위로 넘기고 mid가 학습? 근데 'low weight 초기상태를 mid로 넘긴다'는 건 초기상태잖아—학습된 변화/diff를 줘야 하는 거 아냐?

**A.** Eq71: 레벨 ℓ은 chunk C^(ℓ) 동안 weight 고정, chunk-시작 weight로 매 토큰 gradient f(θ;x_t) 누적, 경계에서 한 번 θ←θ−η·Σ 업데이트(chunkwise stale-snapshot). low(1K): 0~1K 동안 gradient 누적만(학습X), t=1K에 한 번 자기 weight에 압축. mid(5K): 1~4K 안 건드리고 t=5K에 한번, 그동안 자기 입력(=low 출력)에 대한 gradient 누적. 레벨 간 지식전달=forward 직렬(Eq70 y=MLP^fk(…MLP^f1(x)))로 low의 '출력(activation)'이 mid 입력이 됨 — weight/diff 복사 아님. 사용자 반박 정확: meta-learned initial state(Eq72 θ_0^{s+1}=argmin_Φ, pre-training)는 '리셋할 좋은 출발점'이지 런타임 학습된 diff 전달 아님; 리셋되면 그 레벨 학습변화는 버려짐(이미 forward로 전달됨). 즉 '학습된 diff를 위로 넘김'은 NL online이 안 하는 것=Sleep offline consolidation이 추가(distill→low-rank expert). 사용자가 NL 공백을 스스로 재발견.

- 축: `unknown_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_known: 레벨은 chunk 동안 gradient 누적만 하고 경계에서 1스텝(구간 내내 안 배움); unknown_known: 레벨 간 지식은 forward 출력(activation)으로 흐르지 weight/diff 복사 아님; known_unknown: meta-learned initial state=리셋점(pre-train), 런타임 diff 전달 아님 → 그 diff 전달이 정확히 Sleep이 추가하는 것
- 개념 key: Eq71 accumulate-then-step, chunk boundary update, stale snapshot, Eq70 series forward, activation transfer not weight, Eq72 meta-learned initial state, reset point not diff, Nested CMS, online vs offline consolidation, Sleep fills the gap
- 생각할 것: 사용자가 기대한 'diff 위로 전달'=Sleep distillation의 동기—세미나에서 NL 공백→Sleep 필연으로 연결; Eq72 인덱스 방향(어느 레벨이 어느 θ_0로 리셋) 추출텍스트 모호(b)—원 PDF 대조 필요
- storyline seed: CMS 오해 해소 슬라이드: 레벨은 chunk 동안 gradient 누적→경계 1스텝(자기 context 압축), 레벨 간은 forward activation으로 흐름(weight 복사 아님), 초기상태 meta-learn은 리셋점(diff 아님). 그래서 '학습된 변화를 위로 넘기기'는 NL online에 없고 Sleep offline(low-rank expert distill)이 메움. 청중 직관='diff 줘야지'가 곧 Sleep의 동기.
- 연상: Q029, Q027, Q025

## Q031 · 2026-07-15 · CMS Eq72 direction + optimizer

**Q.** (Eq72 방향 확정) + NL에서 CMS 학습 시 optimizer 뭐 쓰는지 설명.

**A.** Eq72 원PDF 확정: C^(ℓ)=max f_i/f_ℓ·C라 인덱스 낮을수록 느림(C큼,안정). 'level s+1 초기상태가 level s에서 meta-learn'—s+1=빠름(low), s=느림(mid)→느린 mid가 빠른 low의 리셋 초기상태 θ_0 제공(사용자가 말한 low→mid의 반대). low는 mid chunk 끝(=low 5회 갱신)마다 θ_0로 리셋. 지식보존(page28 'CMS Design Helps'): 블록 갱신으로 잊혀도 더 느린 블록 MLP^(f_s'),s'<s에 남고 초기상태 backprop이 circle back→시간축 루프로 안 잊음. 런타임 weight-diff 복사 아님(그게 Sleep이 추가). CMS optimizer 2층: (a)레벨 in-context 갱신 Eq71 f=arbitrary optimizer의 error component, 기본 예시 gradient descent; Hope에선 CMS='simple rule'(GD급), self-mod Titans만 expressive DGD. (b)전체 pre-training(초기상태 θ_0·router·embedding)=outer backprop, 표준 AdamW 또는 제안 M3(Hope 훈련 optimizer 명시는 이 절에 없음, c). page27: learned optimizer 파라미터=최저주파수 레벨(pre-train), vanilla=자기 레벨 자기 gradient flow.

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_known: 느린 블록이 빠른 블록의 리셋 초기상태 제공(사용자 방향 반대), 리셋은 느린 chunk 경계; unknown_known: 지식보존=느린 블록 보유+초기상태 backprop circle back(런타임 diff 복사 아님); known_unknown: CMS inner=arbitrary/GD(simple), self-mod=DGD; 전체훈련 AdamW/M3(Hope 명시 c)
- 개념 key: Eq72 nested init, C^(l) frequency ordering, slow provides fast reset init, circle back backprop, continual learning preservation, Eq71 arbitrary optimizer, CMS simple rule GD, DGD only self-mod Titans, outer AdamW/M3, optimizer per level design axis
- 생각할 것: Hope 실제 훈련 optimizer(AdamW vs M3) 명시 확인 필요(c)—재현 runbook 가정; 'circle back' 루프가 online consolidation의 실체—Sleep offline diff전달과 대비해 세미나에 명확히
- storyline seed: CMS optimizer: 레벨 갱신은 simple GD(arbitrary), self-mod Titans만 expressive DGD, 전체훈련 AdamW/M3. 지식보존은 초기상태 backprop circle-back(고정용량)—런타임 diff 복사 없음→Sleep이 그 diff전달 추가. Eq72: 느린 블록이 빠른 블록 리셋 초기상태 제공.
- 연상: Q030, Q025

## Q032 · 2026-07-15 · Sleep — parameter expansion/synaptic-pruning reset/upward distillation & wake(online CMS)↔sleep consolidation & capacity(Atlas Prop1)

**Q.** Sleep(2606.03979)에서 (1) 가장 빠른(High-Freq) layer에도 low-rank expert가 생기나, 언제? (2) 가장 느린(Low-Freq) layer는 expert가 쌓이기만 하나, pre-allocated mask가 다 소진되면 어떻게 되나? (3) NL online consolidation(CMS weight update)과 sleep consolidation의 관계 — online에선 무엇이 update되고 언제까지? online에서 얻은 정보를 sleep이 소화해야 하는 것 아닌가? d² MLP는 O(d)만 저장 가능(Atlas Prop1)하니 어느 시점엔 flush/reset 되어야 하는 것 아닌가?

**A.** (1) 생긴다. Fig.7: 'Updates enter the High-Frequency FFN via repeated Parameter Expansion' — wake 동안 새 지식을 덮어쓰지 않으려 high-freq 블록에 low-rank expert{A d×d_low,B d_low×d} 반복 추가; sleep 경계에서 mid→low로 upward consolidate 후 그 fast 블록의 이전 expert들은 전부 reset(synaptic pruning, line447-450). high-freq=생성↔증류↔reset 반복(작고 유연). (2) 가장 느린 블록은 consolidation 체인 종착점이라 reset 대상 아님 → 단조 성장(line364). 단 구현은 pre-allocate+mask이고 'brain has (large but) FIXED capacity'(line453-455)라 슬롯 유한. 그 슬롯 소진 시 동작은 논문 미명시(공백) = 큰 유한용량일 뿐 결국 fixed-capacity 벽 잔존(암묵적 한계). (3) 사용자 직관이 논문 설계와 정확히 일치. WAKE=각 블록이 window 누적 gradient로 자기 chunk 경계 1step 자기 파라미터 갱신(Eq2)+expert 추가. SLEEP=backbone 전부 freeze, 새 확장 expert만 학습(GKD, line410). '언제까지'=다음 consolidation 경계(덮어쓰기 직전, line344-346). 소화=Knowledge Seeding(upward distillation, teacher=확장전/student=확장+wake갱신후). 포화→flush: 논문이 CF를 'limited capacity, parameters overridden'(line352-353)로 규정=Atlas Prop1 포화; 해법=(a)expansion 용량↑ +(b)consolidate 후 fast reset(flush). 즉 flush=synaptic-pruning reset, 소화=upward distillation. 불변식: base backbone(meta-learn init)=reset 지점, wake=fast에 expert 추가로 online 학습, sleep=slower로 증류 후 fast reset.

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_known: fast(High-Freq) 블록도 wake 중 repeated parameter expansion으로 expert 생성→consolidate→reset 반복(Fig7 caption); reset은 항상 더 빠른 블록만; unknown_known: Q3 직관(정보포화→flush 필요)이 논문 설계와 정확 일치 — flush=synaptic-pruning reset, 소화=upward distillation(Knowledge Seeding); unknown_unknown→known_unknown: pre-allocated mask 슬롯 소진 시 동작 미명시 = 큰 유한용량일 뿐 fixed-capacity 벽 잔존(논문 암묵적 한계); known_unknown: update 대상 분리 — wake=블록 파라미터 CMS 갱신(Eq2), sleep=backbone freeze 새 expert만(line410)
- 개념 key: parameter expansion (low-rank expert A d×d_low, B d_low×d), high-freq FFN = repeated parameter expansion (Fig7), consolidation = iterative upward (High→Mid→Low, 다대일), synaptic-pruning reset (source/faster 블록만, 종착점은 성장), pre-allocate+mask, FIXED capacity → 슬롯 소진 시 동작 미명시(공백), wake=CMS Eq2 자기 파라미터 갱신+expert 추가, sleep=backbone freeze, 새 expert만 학습(GKD+LTI), flush=reset, 소화=upward distillation (사용자 직관=논문 설계), CF=limited capacity/params overridden = Atlas Prop1(O(d)) 포화, teacher=확장 전 / student=확장+wake갱신 후
- 생각할 것: mask 슬롯 소진 시 실제 대책(확장 중단→덮어쓰기 / 오래된 expert 병합·증류로 슬롯 회수 / 더 느린 레벨 추가) — 전부 논문 밖, 검증 필요; wake에서 정확히 base MLP가 갱신되는가 vs 새 expert만 갱신되는가 — Eq2 대상과 Fig7 'expansion 유입' 프레이밍 정합성 재확인; fast 블록 reset이 '추가된 low-rank expert'만인지 base backbone(meta-learn init)은 불변인지 (불변=reset 지점 해석)
- storyline seed: Sleep 세미나 섹션 보강: 'high-freq=생성/증류/reset 반복, low-freq=단조 성장(종착점)' 다이어그램 + wake(CMS Eq2)↔sleep(GKD, freeze 새 expert만) update 대상 분리 표 + 'flush=reset, 소화=upward distillation, 유한 슬롯=잔존 한계' 명시. Atlas Prop1(O(d)) 포화를 CF의 정보이론적 근거로 연결.
- 연상: Q031, Q030, Q025

## Q033 · 2026-07-15 · Sleep — online(wake) vs offline(sleep) consolidation 종류 구분 & expert 생성=offline 전용 & 유한 슬롯 소진 [Q032 정정]

**Q.** Sleep에서 (1) offline consolidation(self-generated data로 high~low expert 생성·reset·학습)은 online consolidation과 같은 연산의 online/offline 타이밍 차이인가? (2) online에는 상위(느린 MLP)로 올리는 전달이 없나? (3) 100조 token을 하루에 wake로 입력하는 극단 상황에서 online이 이미 low-rank expert를 다 채워버릴 수 있나?

**A.** [Q032 정정: parameter expansion(expert 생성)은 offline(sleep) 전용이지 wake가 아니다]. 원문 §3.1 line314-323이 online/offline을 종류로 구분: (Online, wake) end-to-end 학습으로 fast→slow 지식을 forward chain으로 암묵 전달(Eq2 기존 파라미터 갱신, 고정용량, 실입력, 새 expert 없음). (Offline, sleep) 실입력 끊고 self-generated data로 명시적 distillation(GKD+LTI)+parameter expansion(새 low-rank expert 생성)+reset(용량 성장). (1) 종류가 다름: 데이터(실입력 vs 자기생성)·용량(고정 vs 성장)·전달(암묵 forward chain vs 명시 distillation)·expert(없음 vs 생성/reset) 전부 다름. (2) online에도 상위 전달 있음(line318-320 fast→slow) 단 forward chain 통한 암묵적·고정용량 전달이지 expert 생성 명시 distillation 아님; online만으론 CF 못막음(모든 블록 갱신주기 겹치는 순간 CF, line279-281). (3) 엄밀히 online엔 expert 생성 없어 'online에서 소진'은 불가; wake만이면 고정용량 덮어쓰기 포화(CF)뿐. 그러나 consolidation(offline)은 chunk 경계마다 트리거(line346 {C^k×b} steps)라 100조 token 처리 시 경계를 수없이 넘어 slower 블록 expert 단조 성장 → pre-allocated 유한 슬롯 소진 가능; 소진 시 동작은 논문 미명시(공백, line453-455 fixed capacity). 정확한 그림: 'online이 채운다'가 아니라 '경계 트리거 offline consolidation 반복이 느린 블록 유한 슬롯을 채운다'.

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_known: online/offline은 타이밍이 아니라 KIND 차이(데이터 실입력 vs 자기생성, 용량 고정 vs 성장, 전달 암묵 vs 명시, expert 없음 vs 생성); known_unknown→known: expert 생성/reset은 offline(sleep) 전용 — Q032의 'wake expert 추가' 주장은 오류였음(정정); unknown_known: online에도 fast→slow 상위 전달 존재(line318-320) 단 forward chain 암묵·고정용량; unknown_known: 'online에서 expert 소진'은 개념적 불가(online엔 생성 없음); 소진은 경계 트리거 offline consolidation 반복의 결과
- 개념 key: online consolidation = wake end-to-end 학습(Eq2), fast→slow 암묵 전달, 고정용량, offline consolidation = sleep, self-generated data, 명시 distillation+expansion+reset, parameter expansion(expert 생성)=offline 전용 [Q032 wake-expert 주장 정정], online에도 상위 전달 있음(forward chain 암묵)이나 명시 distillation 아님, consolidation은 chunk 경계마다 트리거(line346), wake만 지속 → 고정용량 덮어쓰기 포화(CF), expert 생성 없음, 100조 token → 경계 다수 → slower 블록 expert 단조 성장 → 유한 슬롯 소진 가능, 슬롯 소진 시 동작 미명시(공백), fixed capacity(line453-455), online만으론 CF 못막음(갱신주기 겹치면 CF, line279-281)
- 생각할 것: consolidation이 chunk 경계마다 트리거(line346)면 wake와 offline sleep이 매 경계에서 교대하는가, 아니면 dedicated 긴 sleep 주기인가 — 스케줄 실제 구현 재확인; online의 fast→slow 암묵 전달(line318-320)과 Q031의 'NL 상위 전달은 초기상태(Eq72)뿐, 런타임 diff는 Sleep이 추가' 정합성 — 암묵 전달(forward chain)과 명시 diff(distillation) 구분 명확화; wake에서 갱신 대상이 base MLP인지(Eq2) — offline seeding은 backbone freeze 새 expert만; 두 update 대상 분리 재확인
- storyline seed: Sleep 세미나: online vs offline consolidation 비교표(데이터/용량/전달/expert/서빙) 슬라이드 추가. 'expert 생성=offline 전용, online은 forward chain 암묵 전달·고정용량' 명시. 100조-token 사고실험으로 '유한 슬롯 소진=잔존 한계' 강조. Q031(Eq72 초기상태 vs 런타임 diff)과 연결.
- 연상: Q032, Q031, Q030

## Q034 · 2026-07-15 · NL HOPE(dense CMS) vs Sleep 모델(MoE CMS+graft) 아키텍처 차이 & online↔offline consolidation의 dense↔MoE interlock 공백(seam)

**Q.** NL(2512.24695)의 HOPE와 Sleep(2606.03979)의 모델은 사실 다른 모델 아닌가? NL은 dense FFN, Sleep은 MoE 구조. 이 차이 때문에 Sleep에서 online consolidation이 offline consolidation으로 어떻게 연결되는지 언급이 없어서(진짜 없음?) online→offline 연결을 알 수 없는 것 아닌가?

**A.** 둘 다 맞음. (1) 아키텍처 다름: NL HOPE=self-mod Titans + CMS(dense MLP 체인; NL 전체에서 MoE/sparse/router/expert 언급 0, CMS='Transformer MLP 대체' dense FFN 일반화, 갱신 Eq70-72). Sleep 모델=CMS를 sparse MoE로 가정(line357 'we assume MLP blocks are sparse MoE with a router')—parameter expansion(사전할당 expert mask/활성화) 가능케 하려고 Sleep이 추가. 게다가 Sleep 실험은 대부분 Llama-3.2-1B/Llama3-8B에 graft(line533/775/1510), from-scratch HOPE 아님. → 같은 이름·계보, 다른 구체 모델. (2) online→offline 연결: objective 수준만 명시(§3.3 teacher=확장 전 상태(빠른 블록 online-갱신 지식 보유), student=확장+Eq2 갱신 후, KS가 teacher를 새 expert로 증류 → online-갱신 지식이 offline 증류 소스). 그러나 dense↔MoE 기계적 연결은 공백: online Eq2(line249)는 θ^(f_ℓ)를 일반적(NL dense 그대로) 갱신, MoE/router/어떤 expert인지 세부 없음; MoE는 offline expansion(§3.2)에서만; Fig7('updates enter High-Freq FFN via repeated Parameter Expansion'=MoE) vs Eq2(dense θ 갱신) 화해 안 됨. → online 어느 expert 갱신? online-갱신 base 가중치 reset/보존? 일반 online 갱신이 MoE expert 증류로 어떻게 매핑? 전부 미명시. 결론: online→offline은 teacher/student 경계에서만 이어지고 dense-online↔MoE-offline 실제 interlock은 공백(seam). Sleep이 online을 NL(dense)에서 수입+MoE-offline만 얹어 interlock 안 품.

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_known: NL HOPE(dense CMS)와 Sleep 모델(MoE CMS+graft)은 구체 아키텍처가 다른 모델—같은 이름/계보일 뿐; unknown_known: MoE 가정은 Sleep이 parameter expansion 위해 추가한 것(NL엔 MoE 전무); unknown_unknown→known: online→offline은 §3.3 teacher/student 경계에서 objective로만 연결, dense-online↔MoE-offline 기계적 interlock은 공백(seam); known_unknown: Fig7(expansion=updates)와 Eq2(dense θ 갱신)의 불일치—online이 MoE에 어떻게 작용하는지 미명시
- 개념 key: NL HOPE CMS=dense MLP 체인 (MoE 언급 0, Transformer MLP 대체), Sleep CMS=sparse MoE 가정(line357)—expansion 위해 추가, Sleep 실험=Llama/Qwen graft(from-scratch HOPE 아님), online Eq2(line249)=θ^(f_ℓ) 일반 갱신(dense 상속), MoE 세부 없음, MoE는 offline expansion(§3.2)에서만 등장, Fig7(updates via expansion=MoE) vs Eq2(dense θ 갱신) 미화해, online→offline 연결=teacher/student 경계 objective만 명시(§3.3), dense-online↔MoE-offline 기계적 interlock=공백(seam), 온라인 갱신 expert 선택/base reset·보존/증류 매핑=전부 미명시
- 생각할 것: online Eq2가 MoE 블록에서 실제로 어느 파라미터(base vs active experts vs router)를 갱신하는지—구현/코드로 확인 필요; online-갱신된 base 가중치가 reset(synaptic pruning) 대상인지 보존인지—reset은 'added low-rank experts'만 명시(line447-448), base는 불명; NL from-scratch HOPE(dense)와 Sleep graft(Llama+MoE) 사이 성능/기여 귀속의 혼입—비통제 비교 주의; 이 seam이 논문의 이음새(knobs meta-learn 안됨, graft)와 같은 계열 한계인지 정리
- storyline seed: 세미나 B5/B6 연결부에 'NL HOPE(dense CMS) vs Sleep 모델(MoE CMS+Llama graft)' 아키텍처 차이 슬라이드 + 'online→offline은 teacher/student 경계 objective로만 연결, dense↔MoE 기계적 interlock은 공백(seam)' 명시. 이 seam을 이 계열의 정직한 한계(graft·knobs 미학습)와 함께 제시.
- 연상: Q033, Q032, Q031

## Q035 · 2026-07-15 · test-time training sequence-layer 방법 기술 계보(2016-2026.07): 5스트림·Miras 수렴·Titans 라인·변주된 축·2026 프론티어

**Q.** Titans처럼 test-time training으로 sequence layer를 처리하는 방법들은 무엇이 있고, optimizer 역사 지도처럼 기술 계보가 어떻게 흘러왔나? (오늘=2026-07 최신)

**A.** 핵심 관점: sequence layer의 state=고정벡터가 아니라 test-time에 학습되는 메모리. 서사: 씨앗(2016 Fast Weights·2022 ICL=GD von Oswald) → 세 스트림이 갱신규칙 개선 → Miras가 4축으로 수렴 → Titans 라인이 각 축을 밀어붙임. 5스트림: [뿌리] FastWeights2016·Hopfield용량2020·LinearAttn2020·FWP(Schlag)2021·ICL=GD2022★. [write-rule] GLA·RetNet2023→DeltaNet병렬2024→GatedDeltaNet·RWKV7 2025→MDN·OSDN·Parallax2026. [SSM] Mamba2023→Mamba2 SSD2024→Mamba3 2026. [명시적TTT] TTT(Sun)2024★·Longhorn2024·TTT-video2025·MesaNet2026★·TTT≈LinearAttn2026. [Titans라인] Titans2501→Miras2504★수렴→Atlas2505→TNT2511→HOPE2512→Sleep2606★. 변주된 축: objective(Hebbian덧쓰기→delta→Lp/KL→self-target)·update(additive→1차GD1스텝→+momentum/decay→2차Muon→국소최적exact solve=Mesa)·retention·구조(→self-modifying)·scope(token→window)·serving(→2-stage)·lifecycle(→wake/sleep). 프론티어(2026.07): Sleep(생애주기)·MesaNet(국소최적 갱신, CG solver)·Mamba-3·delta정제. 산출물: HTML genealogy 아티팩트.

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_known: 이 분야 전체가 'state=test-time 학습 메모리' 한 관점의 변주 — optimizer 역사와 동형 구조; unknown_unknown→known: 2026 최신(repo 6편 너머) — MesaNet(국소최적/Mesa layer, ICLR2026)·Mamba-3·delta정제(MDN/OSDN/Parallax)·TTT≈LinearAttn·TitansRevisited; unknown_known: update 축이 additive→1차GD→2차Muon→'국소최적 exact solve(MesaNet)'로 스펙트럼을 이룸; von Oswald mesa-optimization이 그 이론적 씨앗
- 개념 key: 관점: state=test-time에 학습되는 메모리, 씨앗: Fast Weights(2016)+ICL=GD(von Oswald 2022), write-rule 스트림: linear attn→DeltaNet→GatedDeltaNet→RWKV7, SSM 스트림: Mamba→Mamba2(SSD)→Mamba3, 명시적 TTT: TTT(Sun)·Longhorn·MesaNet, Miras=4축 수렴점, Titans 라인 축별 전진(용량·2차·serving·optimizer=memory·lifecycle), update 축 극단=MesaNet 국소최적 exact solve(CG), 2026 최신: MesaNet·Mamba-3·MDN/OSDN/Parallax·TTT≈LinearAttn·TitansRevisited, Hopfield=용량이론 뿌리(→Atlas)
- 생각할 것: MesaNet(CG exact solve)의 test-time compute 대 품질 트레이드 — HATIR로 비용 모델링 가능?; 각 스트림이 Miras 4축의 어느 점인지 정확 매핑표; 2026 delta정제(MDN/OSDN/Parallax)가 Titans 라인과 어떻게 합쳐질지
- storyline seed: 세미나 도입부(또는 A3 뒤)에 이 genealogy 지도를 1슬라이드로 — optimizer 지도와 짝. '이 분야=한 관점의 변주' 프레임으로 6편을 위치시킴. HTML 아티팩트를 PNG로 렌더해 삽입 가능.
- 연상: Q031, Q025

## Q036 · 2026-07-16 · TTT 계보 4계열 분류(TTT-MLP·LinearAttention·DeltaNet·SSM) + efficiency 포함 + branch graph view; 9-agent 워크플로 검증

**Q.** TTT 계보를 4대 계열[TTT(MLP)·Linear Attention·DeltaNet·SSM]로 나누고 efficiency 논문까지 포함해 시간순 branch graph로 재정리(2016~2026.07)

**A.** 뿌리=Softmax Attention(2017 O(n²))에서 4계열로 분기. [SSM] HiPPO2020→LSSL→S4(2021)→{S4D·S5·GSS·H3·Hyena}→Mamba(2023)→Mamba-2 SSD(2024)→Mamba-3(2026). [Linear Attention] FastWeights2016→LinearTransformers(2020)→{Performer·RetNet·RWKV4/5/6·GLA·HGRN·LightningAttn}→RWKV-7(2025). [DeltaNet] Schlag FWP(2021)→ParallelDeltaNet(Yang 2024)→GatedDeltaNet(2024)→{KDA/Kimi·DeltaProduct·MDN·OSDN·GatedDeltaNet-2} 2026. [TTT-MLP] TTT(Sun 2024)·Longhorn→Titans(2025.01)→Miras→Atlas·MesaNet→TNT→HOPE/NestedLearning(2512)→Sleep(2606). efficiency(sub-quadratic)=⚡로 대부분 노드 태깅; 순수이론/비판(HiPPO·TitansRevisited·Sleep)만 비효율. 교차영향(cross): Mamba-2 SSD=SSM↔LinearAttn 이중성(양방향), DeltaNet⊂LinearAttn, Schlag FWP→TTT, MesaNet/Miras/Atlas→LinearAttn(메모리가 linear), Mamba→Longhorn(SSM→TTT 다리). 4계열 경계사례: MesaNet(TTT지만 linear memory), RWKV-7(RWKV계열이나 delta-rule 이식)→cross로 표시. 43노드/~84엣지. 산출: branch graph 아티팩트(시간축 위→아래, hover 계보 하이라이트, cross 토글).

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_known: 4계열이 별개가 아니라 조밀히 얽힘 — DeltaNet⊂LinearAttn, Mamba-2가 SSM↔LinearAttn을 봉합, TTT가 delta/SSM에서 씨앗을 받음; unknown_unknown→known: DeltaNet 2026 라인(MDN/OSDN/GatedDeltaNet-2)과 KDA/Kimi Linear·DeltaProduct 등 efficiency-delta 최신 클러스터; unknown_known: efficiency(sub-quadratic)가 4계열 전체의 공통 추진력 — 표현력 개선과 계산효율이 같은 논문에서 함께 진행
- 개념 key: 4대 계열 분류: TTT(MLP)/Linear Attention/DeltaNet/SSM, efficiency=sub-quadratic 목적이 계보 전반의 공통 동력(⚡), Mamba-2 SSD = SSM↔Linear Attention 이중성(양방향 다리), DeltaNet은 Linear Attention의 delta-rule 변형(Schlag FWP 발원), Schlag FWP → TTT(계열간 씨앗), SSM 뿌리 HiPPO→LSSL→S4; Mamba=selective SSM, DeltaNet 2026 정제 3종: MDN(momentum)/OSDN(online preconditioning)/GatedDeltaNet-2(채널별 게이트), KDA/Kimi Linear=채널별 게이트 delta, 48B 프로덕션·KV절감, MesaNet 경계: TTT이나 memory가 linear→linear-attention과 cross, 제외한 곁가지: sparse/low-rank efficient-attention(Reformer·Linformer·Longformer·BigBird·Nyströmformer·cosFormer·TransNormer), SSM의 DSS·Zamba
- 생각할 것: MesaNet/Atlas의 국소최적 해법(CG/2차)이 test-time compute를 얼마나 쓰는지 → HATIR 비용모델링; sparse/low-rank efficient-attention 곁가지를 계보에 되살릴지(별도 컬럼); 각 계열이 Miras 4축의 어느 점인지 정합 매핑
- storyline seed: 세미나에서 '이 분야 전체 지형'을 이 4계열 branch graph 1장으로 제시 → 그 다음 우리가 판 TTT-MLP/Titans 라인으로 zoom-in. optimizer 지도(직관)와 이 계보 그래프(지형)를 도입부 한 쌍으로.
- 연상: Q035

## Q037 · 2026-07-16 · efficient/system-level TTT 93편 코퍼스 + 6축 시스템 분석(표현력↔GPU효율); LaCT 기준점, in-place TTT 포함

**Q.** efficient TTT를 정조준해 관련 논문을 싸그리(의미있는 것만) 모아 TTT의 system-level 분석을 하라 (LaCT 2505.23884, in-place TTT 등 누락분 포함)

**A.** 핵심 축=표현력↔GPU효율. 기준점 LaCT(Test-Time Training Done Right, 2505.23884): 기존 TTT는 16~64토큰마다 갱신하는 작은 online minibatch 탓에 GPU FLOPs util<5% → 2K~1M 극단 large chunk로 util 70%대. 93편(2020~2026.07)을 6개 시스템 전선으로 분류: ①update-rule 품질↔비용(delta→gated→Muon/2차(Atlas)→exact-solve MesaNet/EFLA/Kaczmarz/KalmaNet ridge→momentum/precond MDN/OSDN/Parallax) ②병렬화·chunk(WY/householder DeltaNet, chunk크기 tradeoff, 비선형recurrence 병렬화 DEER/ParaRNN/Predictability/Log-Linear/PaTH) ③메모리·용량·in-place(low-rank/sparse LoLA/Lattice/Trellis/SparseDeltaMemory, growing KV-Means/Hippocampus, In-Place TTT 2604.06169=down-proj를 fast weight로 재활용) ④커널·하드웨어(flash-linear-attn/TiledFLA, ThunderKittens, HipKittens AMD, LightningAttn-2 IO-aware, TTQ 양자화) ⑤서빙·긴문맥(hybrid stack Jamba/Zamba2/Nemotron/KimiLinear/Samba/Griffin/Hymba, prefix caching Marconi/HYPIC/SparsePrefix, KVBuffer, RW-TTT request별 배치서빙, E2E-TTT-LongContext 2512.23675, Impossibility Triangle) ⑥이론·등가(ICL=GD/mesa-opt, test-time regression 2501.12352 통일틀, TTT=KV binding≈linear attn 2602.21204, state-tracking 한계 negative-eigenvalue/parity/diagonal, scaling law, Muon>Adam). 산출: system-analysis 아티팩트 + research/ttt-efficient-corpus.{json,md}. 워크플로 synth만 스키마초과 실패→journal에서 kept 추출해 저자가 직접 합성.

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_unknown→known: efficient-TTT 시스템 논문군(LaCT·In-Place TTT·E2E-TTT-LongContext·RW-TTT·TTQ·HipKittens·KVBuffer·Marconi/HYPIC prefix caching) — 이전 계보에서 통째로 빠졌던 층; unknown_known: TTT의 진짜 병목은 '알고리즘'이 아니라 'GPU에서의 순차성/util' — 표현력↔효율이 단일 축; unknown_known: update-rule 강화(exact-solve/precond)와 chunk 병렬화가 정면충돌 — 이 tradeoff가 분야를 조직
- 개념 key: 핵심 축: 표현력(강한 update·큰 memory) ↔ GPU효율(병렬성·util·throughput), LaCT 2505.23884: TTT FLOPs util<5%→큰 chunk로 극복(기준점), In-Place TTT 2604.06169: 기존 down-proj를 fast weight로 재활용(별도 state 없음), update-rule 스펙트럼: additive→delta→gated→2차 Muon→exact-solve→precondition, 병렬화: WY/householder chunk + 비선형 recurrence 병렬화(DEER/ParaRNN), 커널이 O(n)을 실속도로: flash-linear-attn/ThunderKittens/HipKittens(AMD), 서빙 병목: hybrid recurrent state의 prefix caching 미표준(Marconi/HYPIC), RW-TTT: request별 fast weight라 배치가 깨지는 서빙 문제, 이론 상한: test-time regression 통일틀, TTT=KV binding≈linear attention 등가, Impossibility Triangle: 긴문맥 memory·계산·정확도 동시최적 불가?
- 생각할 것: 큰 chunk(LaCT)+강한 update(exact-solve)를 chunk 내 순차성 손실 없이 동시에?; hybrid recurrent state prefix caching 표준화가 서빙의 관건; In-place/sparse memory 용량-정확도 한계 이론(d²에 d 정보) 정합; HATIR로 TTT chunk-update의 util·비용을 하드웨어 모델링
- storyline seed: 세미나에 'TTT의 시스템 층' 파트 신설: 우리가 판 알고리즘 라인(Titans/HOPE) 위에 이 6축 system 분석을 얹어 '왜 아직 프로덕션이 어려운가'(util<5%, prefix caching 미표준, request별 서빙)를 설명. LaCT를 도입 훅으로.
- 연상: Q036, Q035

## Q038 · 2026-07-16 · TTT+Titans 효율 study paper: 3 scaling 병목(model-size/training/serving P·D)·해법·memory device 한계·개선; 근거등급+검증

**Q.** TTT 집중조명 + Titans 효율 전반을 놓고, 3 scaling 병목(model-size / training parallel·batch / serving P/D batch)과 각 해법(무엇을·어떻게·왜 working) + memory device 한계·개선을 풍부·상세한 study paper로 작성 (기존 코퍼스 + 대규모 추가 리서치, 우리 세션 방식)

**A.** 12장 study paper 작성. 뿌리: TTT=추론 중 self-supervised gradient로 fast weight 갱신하는 메모리, 그 '순차적 갱신'이 모든 병목의 근원. 병목①model-size: 문맥을 파라미터 아닌 유한 메모리 용량으로 저장(d²에 ~d, associative/Hopfield), chunk mismatch가 규모에서 열화(Titans Revisited), 고정용량 긴문맥 벽. 병목②training: inner-loop 순차 의존→작은 chunk면 FLOPs util<5%(LaCT), backprop-through-inner-loop 메모리, batch×state footprint, 비선형 recurrence 병렬화 조건부. 병목③serving: request마다 mutable state 소유→batching 불변식(공유 static weight)·prefix-cache 붕괴(RW-TTT), prefill(흡수) vs decode(순차·state-bandwidth-bound) 비대칭, hybrid P/D disaggregation 난점. 해법①: distill/linearize로 규모 상속(MOHAWK·Llamba·LoLCATs 405B·Liger), 용량확장(Atlas), 성장(Sleep param expansion+Knowledge Seeding+Dreaming, HOPE nested). 해법②: 큰 chunk(LaCT, util↑·state를 params 40%까지), chunkwise-parallel(DeltaNet WY·TNT), 비선형 recurrence 병렬화(DEER→ParaRNN·Predictability), 커널(Comba Triton·Tiled FLA). 해법③: RW-TTT(owner/version/RW 태깅·호환 phase batch, 274.61 tok/s·9.31×), hybrid prefix caching(Marconi 34.4×·HYPIC TTFT 2.45×·Sparse Prefix), IO-aware(KVBuffer)·In-Place(별도 state 제거)·hybrid stack(Nemotron·Kimi Linear). memory device 한계: (i)신경 메모리 유한용량·interference·retrieval error (ii)physical HBM 점유·decode 대역폭·batch×state. memory device 개선해법: expandable/sparse state(Sparse State Expansion·Sparse Delta Memory), exact memory(Hippocampus·EFLA), growing(KV-Means·Memory Caching), erase-write 분리, 고용량 feature map(Atlas·sympow), 압축(Lattice·Trellis·LoLA), TTQ 양자화·In-Place·KVBuffer. 근거등급 a/b/c, 섹션별 적대적 검증(20 agent). 산출: repo research/ttt-titans-efficiency-study.md + HTML 아티팩트(MathML).

- 축: `known_unknown` → `known` · comprehension: deep
- 새로 드러난 것: unknown_known: 세 병목(model-size·training·serving)이 별개가 아니라 '순차 갱신+유한 메모리'라는 한 뿌리의 세 그림자; unknown_unknown→known: serving P/D 층위 논문군(RW-TTT·Marconi·HYPIC·Sparse Prefix·KVBuffer)과 In-Place TTT — TTT 서빙 정합성 문제를 명명·해결; unknown_known: 모든 효율 이득이 결국 memory device의 유한 용량·대역폭과 거래된다는 관통 원리; frontier=세 꼭짓점 동시 달성
- 개념 key: 순차적 test-time 갱신 = 모든 scaling 병목의 공통 뿌리, model-size 병목 = 유한 메모리 용량(d²에 ~d), chunk mismatch 규모 열화(Titans Revisited), training 병목 = inner-loop 순차 의존→FLOPs util<5%(LaCT), backprop-through-inner-loop, serving 병목 = request-owned mutable state→batching·prefix-cache 불변식 붕괴, prefill vs decode 비대칭, 해법 = distill로 규모 상속 / 큰 chunk·병렬화로 순차성 접기 / 배치·캐시 프리미티브 재설계, LaCT: 큰 chunk로 util 개선(orders-of-mag) + nonlinear state를 params 40%까지[a]; 하드웨어 A100/H100 표기 상충[b], RW-TTT: request-owned state 배치 서빙(owner/version/RW 태깅), hybrid recurrent state는 KV처럼 prefix-cache 안 됨 → Marconi/HYPIC, memory device 두 얼굴: 신경 메모리 용량 + physical HBM/대역폭, memory device 개선 3축: 더 크게(sparse/expandable)·더 정확히(exact)·더 싸게(quantize/in-place), TTT≈linear attention 등가(2602.21204)가 '진짜 meta-learning인가' 의문 제기
- 생각할 것: 큰 chunk(util)와 세밀한 순차 적응(정확도) 동시 달성의 이론 상한; TTT≈linear attention 등가면 test-time 학습의 표현력 우위는 어디서? 진짜 meta-learning 여부; d²→d 용량 벽을 sparse/expandable이 상수 개선인지 지수 변경인지; memory device 물리(HBM 대역폭)×알고리즘(state 용량) co-design을 HATIR류 비용모델로 예측
- storyline seed: 세미나에 'TTT/Titans는 왜 아직 프로덕션이 어려운가' 파트로 이 3병목×해법×memory device 지도를 그대로 사용. Part I(집중조명)→II(병목)→III(해법)→IV(memory device) 흐름이 곧 강의 흐름. LaCT의 util<5%를 도입 훅.
- 연상: Q037, Q036, Q034, Q033, Q032

