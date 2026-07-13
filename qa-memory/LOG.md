# QA LOG — 공부 질문 기록 (chronological)

총 5건.

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

