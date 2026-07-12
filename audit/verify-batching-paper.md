# verify-batching-paper — 사용자 batchability 분석 vs 스터디 페이퍼 정합성 검증

- 검증 대상 사용자 분석: Self-Modifying / Self-Referential Titans & Hope의 **serving-time batchability** (C1–C10)
- 대조 페이퍼: `study-kr/part3/ch20`, `ch23`, `ch24`; `study-kr/part1/ch04`; `experiments/results.json`(claim1/3/6); `experiments/E1.3-kv-vs-ttt/results.json`
- 판정 요지: **사용자 분석은 페이퍼와 모순이 아니라 보완이며, 여러 지점에서 페이퍼가 이미 단정한 것을 다른 각도(수학·커널 form)에서 재확인한다. memory-centric 논지를 약화하지 않고 오히려 강화한다.**

---

## 1. 핵심 긴장 해소 — (b) 서로 다른 층위의 양립·보완

**결론: 모순 아님. 두 진술은 같은 사실의 두 얼굴이며, 페이퍼 본문이 두 얼굴을 이미 모두 명시한다.**

사용자의 주장(C1/C10): request마다 다른 fast-weight를 batch 차원으로 묶어 BMM/grouped-GEMM/batched rank-update로 처리 가능 = **batching feasibility(수학·커널 가능성)**.

페이퍼의 주장(claim3 headline "breaks shared-weight batching", B_max=5.2): batch를 늘려도 대역폭 상각이 안 됨 = **batching economics(대역폭 amortization 경제성)**.

이 둘은 **다른 축**을 잰다. 결정적 증거는 페이퍼가 사용자의 grouped-GEMM을 **자기 제안으로 직접 채택**하고, 그 자리에서 "AI는 B와 무관"임을 명시한다는 것이다.

- `ch24.md` 제안 G (line 106): "per-request mutable fast weights는 shared-weight batching을 **깨뜨린다** … 제안은 request별 state를 batch 축으로 묶어 처리하는 grouped-GEMM decode path다. … request별 unshared weight의 grouped-GEMV가 **근본 arithmetic intensity를 올리지는 못하지만**(각 $W_b$가 한 번 읽혀 한 번 쓰이므로 FLOP·traffic이 함께 $B$에 비례해 AI는 $B$와 무관), kernel-launch·occupancy·scheduling 파편화를 걷어 memory-bound 평원의 **포획 손실**을 줄이기 때문이다."
- `ch24.md` 리스크 G (line 108): "G는 batching **가능성**을 복원하는 것이지 트래픽 하한을 낮추지 않는다 — 그 하한은 소자 D의 몫이며, 그래서 G와 D는 같은 decode step을 나눠 맡는다."
- `ch23.md` §23.4 (line 70, 72): "per-request unshared weights는 shared-weight batching을 깨고, **decode는 grouped-GEMM으로 가야 하며**(→ 24장) … 요청마다 상태가 unshared라, batch $B$의 decode step이 움직이는 상태 트래픽이 $B$에 **선형**으로 자란다."
- `ch04.md` (line 122): "per-request 상태가 shared-weight batching을 깨뜨린다는 함의는 10장과 Part III의 주제다."

**"breaks shared-weight batching"의 정확한 의미 확인.** 페이퍼 본문에서 이 문구는 일관되게 *"공유-weight 서빙의 batch 상각 free-lunch가 사라진다"*로 서술되며, *"수학적으로 batch 불가"*로는 어디에도 서술되지 않는다. 대조 근거:

- `ch23.md` §23.4 (line 72): "shared-weight 서빙에서 batch를 키우는 것은 공짜에 가깝다 — 같은 weights를 여러 요청이 재사용하므로 GEMM이 커질 뿐이다. session-weight에서는 요청마다 상태가 unshared라 … 트래픽이 $B$에 선형." → 깨지는 것은 *재사용(공유) 상각*이지 batch 형성이 아니다.
- 같은 문장이 곧바로 "decode는 grouped-GEMM으로 가야 한다"로 이어진다 → 페이퍼 스스로 batch를 **형성**하는 경로(grouped-GEMM)를 처방한다. 즉 배치는 가능하되 상각이 안 되는 것.

따라서 사용자가 우려할 법한 "batch가 수학적으로 불가능하다"는 오독은 **페이퍼 본문에서는 발생하지 않는다.** 유일한 오독 소지는 `results.json`의 압축 headline 문자열 `"breaks shared-weight batching"`을 문맥 없이 고립해 읽을 때뿐이며, 본문(ch23 §23.4 · ch24 제안 G · ch04 line 122)이 그 문자열을 amortization 경제성으로 명확히 못 박는다.

**요약: 사용자 = 분자(FLOP)와 트래픽이 함께 B로 자라 batch가 form된다. 페이퍼 = 바로 그 때문에 AI가 B에 안 비례해 B_max=5.2에서 대역폭이 배치를 조기에 닫는다. 동일 사실의 두 진술.**

---

## 2. 페이퍼가 이미 다루는 것 vs 사용자 분석이 추가하는 것

| 사용자 항목 | 페이퍼 대응 | 판정 |
|---|---|---|
| C1 batch 차원 추가 → BMM/grouped-GEMM, attention per-request K/V와 구조 유사 | ch24 제안 G(grouped-GEMM decode), ch23 §23.8(continuous batching↔grouped-GEMM) | 이미 다룸(재확인). 사용자는 attention K/V와의 kernel-form 유비를 **명시 추가** |
| C2 M_t = M_{t-1}A_t − η v̂ kᵀ, 텐서 shape M:[B,Dout,Din] 등 | ch20 §20.2.1 delta-rule rank-1 write $W_t=W_{t-1}(I-\eta k k^\top)+\eta v k^\top$; ch04 rank-1 outer-product | 정합. 사용자는 (i) forget을 스칼라 $\alpha_t$→행렬 $A_t$로 일반화, (ii) **명시적 batched 텐서 shape**를 추가 |
| C3 self-referential: k/v/q/lr/decay 생성 memory도 갱신, sequence별 tensorize | ch20 §20.2.4 Hope 6-memory 블록(main+5 projection-memory), ch16 self-modifying | 이미 다룸. 사용자의 "sequence별 별도 상태" = 페이퍼의 per-seq unshared state |
| C4 각 memory module = residual 2-layer MLP, W2:[B,H,D]/W1:[B,D,H] batched forward | ch20 §20.2.2 $\mathcal{M}(z;W)=z+W_1\sigma(W_2 z)$ 정확히 동일 | 정합. 사용자는 batched(weight-per-B) forward shape를 추가 |
| C5 batch ∥ + chunk-내부 token ∥, chunk 경계 recurrence | ch20 §20.5 chunk $C$가 roofline x축, chunkwise scan; ch04/ch09 unrolled static graph | 정합. 사용자는 **batch축과 chunk축의 2D 병렬성 분리**를 명시 articulation으로 추가(페이퍼는 chunk축 위주, batch축은 B_max로만 다룸) |
| C6 inner mini-batch GD → matmul+scan, momentum=associative scan, "Fast&Parallelizable" | ch09 chunkwise-parallel, ch20 §20.5 associative scan | 이미 다룸(재확인) |
| C7 inner update는 request별 개별(평균 금지, leakage), outer meta-param은 batch 평균 OK | ch04 함수/값 분리, 표4-2: $W_t,S_t$=fast(inner, per-seq), $\Theta$(η/β/α 함수, projection, $W_{init}$)=outer(pretraining, batch로 hypergradient 평균) | **정합·강한 확인.** ch04의 per-seq $W_t$ 구조가 곧 "inner 평균 금지"를 함의. serving엔 outer 부재라 평균 자체가 없고(각 request 독립 진화) 그래서 grouped(공유 아님) batch임. C7은 ch04의 함의를 정확히 명시화 |
| C8 각 batched op 1-launch 가능하나 전체 fused single-kernel 아님; Google이 vLLM급 kernel 보유 단정 불가 | ch20 §20.3.1(fused deep-memory kernel 부재, dossier §3.2), ch24 제안 F(fused chunk kernel 미존재, fla가 titans를 naive에 둠) | **정직성 계약과 정합.** 사용자 C8은 페이퍼보다 더 보수적이지도 덜하지도 않게 일치 |
| C9 request별 여러 mutable matrix + optimizer/momentum → "Paged Fast-Weight State Manager" 필요 | claim6 5개 누락 API(update_in_place, mark_dirty/writeback, checkpoint/rollback, bind_to_sequence, free_on_update); ch24 제안 I(per-session-weights serving stack); ch23 §23.8(PagedAttention 무효, fixed-size slab allocator) | **이미 다룸(강한 확인).** C9 = 제안 I 그대로. 사용자는 "Paged Fast-Weight State Manager"라는 명명을 추가 |
| C10 workload = batch별 weight 다른 recurrent MLP + batched optimizer/update; MoE grouped-GEMM + TTT + batched optimizer + recurrent scan 결합 | ch24 제안 D/G/H(RMW device + grouped-GEMM + backward-capable kernel), ch20 §20.3(backward-in-decode primitive), ch20 §20.2.4(Hope≈MoE적 6-stream) | 정합. 사용자는 **깔끔한 workload-class 라벨**(MoE+TTT+optimizer+scan 결합)을 추가 |

**사용자 분석이 실질적으로 추가하는 것(페이퍼에 없거나 암묵적이던 것):**
1. **명시적 batched 텐서 shape/BMM formalization** (C1/C2/C4) — 페이퍼는 "grouped-GEMM으로 간다"까지만, [B,Dout,Din]·[B,Din,Din] 같은 shape 전개는 안 함.
2. **batch축 × chunk축 2D 병렬성의 명시 분리** (C5) — 페이퍼는 chunk축(roofline)과 batch축(B_max)을 각각 다루되 둘을 한 병렬성 다이어그램으로 겹쳐 진술하진 않음.
3. **attention per-request K/V batching과의 kernel-form 유비** (C1) — 페이퍼는 KV를 *append-once/read-many·shareable*로 **대조**하는 데 집중(§23.8), 사용자는 *batched-per-request-state*라는 **공통점**을 부각.

이 셋은 모두 페이퍼 골격 위의 **디테일 보강**이며 어느 것도 페이퍼 주장과 충돌하지 않는다.

---

## 3. 정직성 계약·pair thesis와의 정합 — 어긋남 없음, 오히려 강화

**사용자 분석은 memory-centric 논지를 약화하지 않는다. 강화한다.**

- 페이퍼의 memory-centric 논증은 *"batch가 불가능하다"*에 의존하지 **않는다**. 그것은 (i) state RMW가 GEMV를 394× 압도(claim1), (ii) per-request state라 AI가 B에 무관, (iii) 대역폭이 B_max에서 배치를 조기에 닫음(claim3)에 의존한다. 사용자가 grouped-GEMM 가능성을 입증해도 이 세 기둥은 그대로다 — 리스크 G가 명시하듯 "batching 가능성을 복원하는 것이지 **트래픽 하한을 낮추지 않는다**"(ch24 line 108). 하한은 여전히 소자 D의 몫.
- 즉 사용자의 batchability는 페이퍼의 accelerator 절반(제안 G) 안에 **정확히 예약된 자리**에 앉으며, decode가 memory-bound·B_max-제약이라는 결론을 건드리지 않는다. G와 D는 "같은 decode step을 나눠 맡는다"는 pair thesis의 실천 그 자체.
- 사용자 C8("Google이 vLLM급 continuous-batching kernel 보유를 공개 결과로 단정 불가")은 페이퍼의 정직성 계약(fused kernel 부재, dossier §3.2; NO-WALLCLOCK; 공개 구현 0)과 **동일한 절제**를 지킨다. 과장이 없다.
- C7의 inner/outer 구분(inner 평균 금지 = 정보 오염 방지)은 오히려 "per-request state가 unshared라 shared-weight 상각이 원천적으로 불가능"이라는 페이퍼 논지에 **미시적 근거**를 하나 더 댄다: 배치를 형성하더라도 각 $W_b$를 섞을(평균낼) 수 없으므로 공유가 수학적으로도 금지 — B_max 벽의 정당화 보강.

**유일하게 주의할 미묘한 지점(모순 아님, 강조 축의 차이):** 사용자 C1의 "attention per-request K/V batching과 구조적으로 유사"는 **kernel-form 수준**에서 타당하다(둘 다 per-request state에 batch 차원 추가). 단 페이퍼가 §23.4/§23.8·claim2에서 강조하는 **경제적 비대칭** — KV는 write 0.003%(append-once)·shareable·문맥∝S이고 TTT는 write 100%(RMW)·unshared·폭∝ — 까지 "유사"가 번지면 안 된다. 사용자는 C1을 "구조적 유사"로 한정하고 C7/C10에서 per-request·never-average·write-heavy를 스스로 명시하므로, 이 경계를 이미 지키고 있다. 유비는 form에서 성립, amortization에서는 성립하지 않음 — 이 한 줄만 유지하면 충돌 없음.

---

## 종합 판정

1. **핵심 긴장**: 모순 아님 → 서로 다른 층위(사용자=수학·커널 가능성, 페이퍼=대역폭 상각 경제성)의 **양립·보완**. 페이퍼가 사용자의 grouped-GEMM을 제안 G로 이미 채택하고 "AI는 B에 무관"을 명시. "breaks shared-weight batching"은 본문에서 *상각 free-lunch 소멸*로 정확히 서술되며 *batch 불가*로 오독되지 않음(고립된 results.json headline만이 유일한 소지, 본문이 해소).
2. **추가분**: 사용자는 batched 텐서 shape formalization, batch×chunk 2D 병렬성 분리, attention-K/V kernel-form 유비를 **디테일로 보강**. C7↔ch04, C9↔claim6/제안 I는 페이퍼와 강하게 정합.
3. **정직성/pair thesis**: 어긋남 없음. batchability는 memory-centric 논지를 약화하지 않고(트래픽 하한 불변) 강화(inner-never-average가 unshared 상각불가에 근거 추가). C8은 페이퍼의 절제를 그대로 준수.
