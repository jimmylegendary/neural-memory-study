# QA LOG — 공부 질문 기록 (chronological)

총 1건.

## Q001 · 2026-07-14 · Titans

**Q.** Titans에는 k,v로 구성된 연상메모리가 있는듯한데 k가 주어졌을때 v가 나오는? 그 weight를 token마다 학습? 그럼 q는 안씀? loss는 뭐와 뭐의 차이를 어떤 손실함수로 구함?

**A.** 메모리 M=작은 MLP, weight 자체가 state이고 M(k)≈v가 되도록 저장(표 아님, 가중치에 저장). token마다 갱신되는 건 메모리 M의 weight뿐(inner loop); 투영 W_K/W_V/W_Q는 outer-loop 사전학습·inference 고정. q는 씀 — 단 write(학습)가 아니라 read(retrieval)에: write=(k,v)로 loss→메모리 갱신, read=M*(q_t)로 갱신 없이 검색. loss = ||M(k_t)-v_t||_2^2 (L2/MSE 회귀), 메모리 예측 M(k_t) vs 정답 v_t의 차이; 이 gradient 크기가 surprise.

- 축: `known_unknown` → `known` · comprehension: working
- 새로 드러난 것: unknown_unknown: 투영 W_K/W_V/W_Q는 token마다 학습 X — outer-loop 고정, 메모리 M weight만 per-token 갱신; unknown_unknown: q는 안 쓰는 게 아니라 read(retrieval)에 씀; write는 (k,v), read는 q — 두 연산 분리(M vs M*); unknown_known: '메모리=표'가 아니라 '메모리=가중치에 저장된 회귀함수'라는 프레임
- 개념 key: associative memory, key-value, query, read vs write, per-token update, inner loop, outer loop projections, surprise, L2 loss, associative regression, memory-as-weights
- 생각할 것: Miras가 이 L2 loss를 Lp/Huber/KL 'attentional bias'로 일반화 (G04/ch13) — 왜 다른 loss가 이기나; read M*(q_t)가 MAC/MAG/MAL에서 windowed attention과 어떻게 결합되나; M(k_t)와 M*(q_t)는 같은 네트워크에 다른 입력 — retrieval의 '비슷한 key 일반화'가 어디서 나오나(MLP 비선형성); surprise(gradient 크기)를 momentum S_t가 어떻게 누적하고 weight decay(1-α)가 어떻게 잊나
- storyline seed: Titans 한 줄: 메모리=k→v 회귀를 test-time에 L2로 학습(write=k,v / read=q / surprise=gradient). 투영은 outer-loop 고정. 이후 Miras가 loss 일반화, Atlas가 용량·optimizer.

