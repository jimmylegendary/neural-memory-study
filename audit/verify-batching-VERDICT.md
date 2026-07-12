# verify-batching-VERDICT — Self-Modifying/Hope serving-time batchability 분석 종합 판정

4-소스 종합: `verify-batching-nl.md` (NL 원문 2512.24695), `verify-batching-titans.md` (Titans 원문 2501.00663), `verify-batching-paper.md` (스터디 페이퍼 정합), `verify-batching-codex.json` (독립 재검토). 모델 간 불일치는 원문 인용으로 재정.

집계: **CONFIRMED 6 / PARTLY 4 / WRONG 0 / UNVERIFIABLE 0** (C1,C4,C6,C7,C8,C10 = CONFIRMED · C2,C3,C5,C9 = PARTLY).

---

## 1. 주장별 판정표 (C1–C10)

### C1 — batch 차원 추가 → BMM/grouped-GEMM/batched rank-update; attention per-request K/V와 구조적 유사
**판정: CONFIRMED** (수학·커널 form). 4-소스 합의: NL=범위밖(구조적 타당), Titans=CONFIRMED(수학)/serving은 타당한 외삽, paper=이미 채택(제안 G), codex=imprecise(경계 조건).
- 근거: Titans Eq.16–18 — per-request inner update가 matmul/sum + rank-1 outer-product `(Wk−v)kᵀ`로 표현됨; 서로 다른 sequence를 결합하는 항이 없어 batch-독립. NL §8.2는 각 sequence가 독립 `M_□` 상태를 갖는 수식 제시.
- **경계(codex 재정)**: "구조적 유사"는 *상태 격리·스케줄링·kernel-dispatch form* 수준에서만 성립. amortization으로 번지면 안 됨 — shared-W는 FLOP∝B·weight bytes≈고정이나, per-request W_b는 FLOP·state bytes 둘 다 ∝B라 AI는 B와 무관. 사용자 C1은 "구조적"으로 한정하고 C7/C10에서 per-request·never-average를 명시하므로 경계를 이미 지킴. codex의 "imprecise"는 원분석의 "HBM재사용 0/항상 memory-bound" 절대화 표현을 겨눈 것이며, 여기 제시된 C1 텍스트에는 그 표현이 없음 → 판정 CONFIRMED 유지.

### C2 — M_t = M_{t−1}A_t − η_t v̂_t k_tᵀ, A_t=data-dependent transition; batch shape M:[B,Dout,Din], A:[B,Din,Din]
**판정: PARTLY.** 4-소스: NL=CONFIRMED(선형/dot-product 특수형, 사용자가 "단순화형" 명시), codex=imprecise, Titans=원본은 diagonal/identity-minus-rank-1(일반 dense A 아님).
- 근거: NL Eq.92 `M_□,t = M_□,t−1(α_t I − η_t k_t k_tᵀ) − η_t v̂_□,t k_tᵀ` 정확히 일치. `A_t = α_t I − η_t k_t k_tᵀ ∈ R^{Din×Din}`, k_t 통해 data-dependent. 차원 자기정합(M:[Dout,Din], A:[Din,Din], v̂kᵀ:[Dout,Din]).
- **보정 필요 2점**: (i) 이 닫힌형은 **선형 memory·dot-product objective 특수형**이며 실제 배치되는 2-layer MLP memory의 일반 update는 Eq.88/90의 full `∇L` 항(L2형은 `−η(Mk−v̂)kᵀ` residual-gradient 추가). 사용자가 "단순화형"으로 라벨해 정직하게 표시함. (ii) A_t는 dense가 아니라 **identity-minus-rank-1**이므로 `A:[B,Din,Din]`를 dense로 materialize하거나 M·A를 dense BMM으로 계산하면 낭비 — 효율 구현은 `Mk`와 outer-product 사용. → 형태는 맞으나 batched-dense-A 텐서 표기는 구현 함정을 유발할 수 있어 PARTLY.

### C3 — self-referential: k/v/q/lr/decay 생성 memory도 context별 갱신, sequence별 tensorize
**판정: PARTLY** (핵심 옳음, query 과다포함). 4-소스: NL=CONFIRMED+query caveat, codex=correct, paper=이미 다룸(Hope 6-memory).
- 근거: NL Eqs.79–88 — fully-adaptive 설계에서 k,v,q,η,α,memory 모두 in-context-updated memory, 초기상태 meta-learned(line 2084). self-modifying step은 각 memory가 `v̂_□,t=M_□,t−1(v_t)` 자체생성.
- **보정(NL 원문 재정, line 2116)**: *최종 self-modifying/Hope 설계에서 q는 static* — "𝒒_t = 𝒙_t W_q is the only non-adaptive projection." ablation도 "freeze inner q"가 ppl 개선(12.19 vs 12.24). 즉 "query가 context-updated memory"는 *중간 fully-adaptive variant*에만 참이고 *배포 Hope*에는 거짓. 사용자가 query를 포함시킨 것은 shipped 모델 기준 과다포함 → PARTLY. 나머지("생성기 memory도 갱신", "sequence별 별도 상태 tensorize")는 CONFIRMED.

### C4 — 각 memory module = residual 2-layer MLP M(x)=x+W1σ(W2x); batched W2:[B,H,D], W1:[B,D,H]
**판정: CONFIRMED.** 4-소스 만장일치 correct.
- 근거: NL Eq.89/91 `M_□(·) = (·) + W_□,1 σ(W_□,2(·))` 축자 일치, "We use a 2-layer MLP block as the architecture of all the memories"(line 2154). 열벡터 관례 W2:[H,D](D→H), W1:[D,H](H→D), residual 위해 in=out=D. `[B,·,·]` 배칭은 analyst 추가지만 차원 정합·TTT-MLP batched-small-GEMM 패턴과 동일.

### C5 — 완전 병렬화 불가; batch축 ∥ + chunk-내부 token ∥, chunk 경계 recurrence; 논문 dual/chunk form이 이것
**판정: PARTLY** (구조 옳음, decode 적용 caveat). 4-소스: NL=CONFIRMED, Titans=CONFIRMED(intra-chunk linear/inter-chunk nonlinear, Appendix C lines 2619–2620), codex=imprecise(decode 경고).
- 근거: NL §8.2 — 이전 chunk의 frozen snapshot `M_□,C×⌈t/C⌉`에서 현 chunk의 생성값·gradient를 병렬 계산, "fast parallelizable dual form"(line 2211). CMS: "for input x_i when i ≢ 0 (mod C) there is no sequential process inside the chunk"(lines 1881–1883). 교차-chunk state carry가 recurrence.
- **보정(codex 재정)**: *chunk-내부 token 병렬성은 token들이 이미 주어진 training·prefill에만 해당*. 일반 autoregressive **decode에서는 미래 token이 없어** batch축 외의 이 병렬성을 그대로 쓸 수 없음 — decode step에는 batch축 병렬성만 남음. 사용자의 2D(batch×chunk) 병렬성 진술은 prefill/training에서 참, decode에서는 chunk축이 붕괴 → PARTLY. (페이퍼가 다루는 것은 memory-bound decode이므로 이 구분이 특히 중요.)

### C6 — inner mini-batch GD → matmul+sum, momentum=associative scan; NL "Fast and Parallelizable Training" 명시
**판정: CONFIRMED.** 4-소스: Titans=CONFIRMED(양 메커니즘 축자), NL=CONFIRMED(§8.2 header + dual form 상속), codex=imprecise(적용범위 분리 주문).
- 근거: Titans §3.2 "reformulated so that it uses only matmuls and sum"(lines 421–424, Eq.16–17); momentum `S_t=η_t S_{t−1}−θ_t u_t`는 "linear recurrence … we can use parallel associative scan"(lines 466–472, Eq.18). NL §8.2 header "Fast and Parallelizable Training"(line 2162), chunk dual form 상속(line 2211).
- **범위 note(codex, 오류 아님)**: nonlinear memory recurrence *전체*가 scan 가능한 게 아니라 chunk gradient는 matmul+sum, 그 안의 *선형 momentum recurrence만* scan. 또한 "Fast and Parallelizable Training"은 training 라벨이지 serving continuous-batching 입증이 아님 — 이는 사용자가 C8에서 스스로 지킨 절제와 일치. 메커니즘 자체는 정확 → CONFIRMED.

### C7 — inner update는 request별 개별(평균 금지, leakage); outer meta-param은 batch 평균 OK
**판정: CONFIRMED.** 4-소스: codex=correct, paper=정합·강한 확인(ch04 표4-2 fast/slow 분리).
- 근거: Titans §3.1 inner-loop가 M weights 갱신 / outer-loop가 공유 parameter 최적화 구분. 독립 상태 M_b에 대한 update는 batch reduction 없는 **map 연산**(벡터화이지 reduction 아님) — request 방향 평균 시 사용자 간 정보 오염. paper ch04: serving엔 outer 부재라 평균 자체가 없고 각 request 독립 진화 → grouped(비공유) batch.
- note: 논문의 "mini-batch GD"는 request 평균이 아니라 *한 sequence 내부 chunk token* update를 뜻함(codex) — 사용자 진술과 상충 없음. 이 never-average가 "per-request state unshared → shared-weight 상각 원천 불가"에 미시 근거 하나를 더 댐(B_max 벽 정당화 보강).

### C8 — 각 batched op 1-launch 가능하나 전체 fused single-kernel 아님; Google vLLM급 continuous-batching kernel 보유 단정 불가
**판정: CONFIRMED.** 4-소스: codex=correct, paper=정직성 계약과 정확 정합(dossier §3.2 fused kernel 부재).
- 근거: NL §8.2는 dual/chunk 병렬 계산·training throughput만 제시, ragged decode·state paging·admission/eviction 포함 serving runtime이나 단일 fused kernel 미제시. 다단계 self-modifying block이 자동 fuse되지 않음. 페이퍼의 NO-WALLCLOCK·공개구현 0 절제와 동일 — 과장 없음.

### C9 — serving state가 attention보다 복잡, 여러 mutable matrix+optimizer/momentum → "Paged Fast-Weight State Manager" 필요 가능성
**판정: PARTLY** (복잡성·필요성 옳음, size 일반화 보정). 4-소스: codex=imprecise, paper=이미 다룸(제안 I·claim6 5개 누락 API·§23.8 PagedAttention 무효).
- 근거: NL Eq.94–97 여섯 `M_□` 상태 + 후속 CMS chain 정의; §8.2는 memory 본체와 나머지 memories에 서로 다른 chunk size 사용. "Paged Fast-Weight State Manager"는 페이퍼 제안 I(per-session-weights serving stack)와 동일 개념에 명명 추가.
- **보정(codex)**: "attention보다 크다"까지 일반화 불가 — KV cache는 context length∝로 자라지만 fast-weight state는 보통 **고정 크기**라 길이에 따라 우열이 뒤바뀜(사용자는 "복잡"이라 표현했으나 크기 오독 방지 필요). 또 완전 Hope는 열거 6개 외 CMS 다중-frequency mutable blocks + optimizer momentum state도 포함(하한 과소). 반대로 M_q는 shipped Hope에서 static(C3 보정)이라 상태 목록에서 뺄 여지 → 열거가 동시에 과다·과소. "필요 가능성"은 적절히 hedge됨 → PARTLY.

### C10 — 결론: batch별 weight 다른 recurrent MLP + batched optimizer/update; MoE grouped-GEMM + TTT + batched optimizer + recurrent scan 결합 workload
**판정: CONFIRMED** (workload-class 라벨 적절). 4-소스: codex=imprecise(MoE 비유 경계), paper=정합(제안 D/G/H, backward-in-decode primitive, Hope≈6-stream).
- 근거: NL §8.1–8.2가 request-local 2-layer MLP weights + DGD(data-dependent GD) update 결합. HW상 grouped-GEMM 표현 가능.
- **경계(codex, C1과 동일 형태)**: MoE 비유는 *kernel-dispatch form*에만 근접. MoE는 다수 token이 소수 shared expert weight 재사용하나 여기선 request마다 weight가 달라 decode 시 group당 token≈1 → expert-weight 상각 이득 없음. 사용자 C10이 "…에 가까움"으로 form-level 한정하므로 경계 준수 → CONFIRMED. C1과 함께 "유비는 form에서 성립, amortization에서 불성립"의 한 줄만 유지하면 됨.

| Claim | 판정 | 4-소스 합의 | 핵심 원문 근거 / 재정 |
|---|---|---|---|
| C1 | CONFIRMED | ✅(경계 조건) | Titans Eq.16–18; 유사는 form-only, amortization 아님 |
| C2 | PARTLY | ⚠️ | NL Eq.92(특수형); dense A materialize 낭비 + MLP-일반 ∇L 항 |
| C3 | PARTLY | ⚠️ | NL line 2116: shipped Hope q는 static(query 과다포함) |
| C4 | CONFIRMED | ✅ 만장일치 | NL Eq.89/91 축자 일치 |
| C5 | PARTLY | ⚠️ | intra-chunk 병렬은 prefill/training-only, decode엔 batch축만 |
| C6 | CONFIRMED | ✅ | Titans Eq.16–18; scan은 선형 momentum에만(범위 note) |
| C7 | CONFIRMED | ✅ | Titans §3.1 inner/outer; map 연산(no reduction) |
| C8 | CONFIRMED | ✅ | NL §8.2 serving runtime 미제시; 정직성 계약 정합 |
| C9 | PARTLY | ⚠️ | 복잡성 옳음, fixed-size vs KV-∝length 우열역전; CMS+momentum 누락 |
| C10 | CONFIRMED | ✅ | NL §8.1–8.2; MoE 비유는 dispatch-form only |

---

## 2. 핵심 긴장 재정 — "batching 가능(사용자)" vs "breaks shared-weight batching(페이퍼 claim3)"

**모순 아님 — 서로 다른 축을 재는 보완이며, 페이퍼 본문이 두 얼굴을 이미 명시한다.** 사용자 = **batching feasibility**(수학·커널: FLOP과 traffic이 함께 B로 자라 grouped-GEMM으로 batch가 *형성*됨). 페이퍼 = **batching economics**(AI가 B와 무관해 대역폭이 B_max=5.2에서 배치를 조기에 닫음). 결정적 증거: 페이퍼가 사용자의 grouped-GEMM을 **제안 G로 직접 채택**하고 그 자리에서 "각 W_b가 한 번 읽혀 한 번 쓰이므로 FLOP·traffic이 함께 B에 비례해 AI는 B와 무관"이라 명시(ch24 line 106), 리스크 G는 "batching 가능성을 복원하는 것이지 트래픽 하한을 낮추지 않는다"(line 108)고 못 박는다. 즉 사용자의 batchability는 accelerator 절반(제안 G) 안 **예약된 자리**에 앉으며 memory-bound 결론을 건드리지 않는다. **"breaks shared-weight batching" 표현은 정확하되 보정 1줄 권고**: 본문(ch23 §23.4·ch24 G·ch04)에서는 일관되게 *"공유-weight 배치 상각 free-lunch 소멸"*로 서술돼 오독 없으나, `results.json`의 고립된 headline 문자열만 "batch 수학적 불가"로 오독될 소지 — headline에 "…batch AMORTIZATION(grouped-GEMM은 여전히 batch를 형성하되 AI는 평평)" gloss 추가 권고. C7(never-average)은 오히려 이 논지를 강화한다.

---

## 3. 정확성 총평

**사용자 분석은 전체적으로 옳다(핵심 결론 정확, WRONG 0).** request-local fast-weight 상태를 batch 차원으로 tensorize해 serving batch로 실행 가능하고 request 간 update를 평균해선 안 되며, 동시에 shared-weight dense layer의 B-비례 weight 상각은 얻지 못해 one-token decode가 bandwidth-bound라는 구분 — 이 골격은 4-소스가 모두 지지한다. 페이퍼와 모순이 아니라 여러 지점에서 페이퍼를 다른 각도(수학·커널 form)로 재확인하며 memory-centric 논지를 강화한다.

**부정확·보정 필요(과장 아닌 정밀도 문제) 4건**: (i) **C2** — dot-product 선형 특수식을 실제 2-layer L2 Hope update의 전형처럼 제시할 위험(사용자가 "단순화형" 라벨로 이미 완화) + dense `A:[B,Din,Din]` 과다-materialize 함정. (ii) **C3** — shipped Hope에서 query는 static인데 context-updated memory에 포함(중간 variant에만 참). (iii) **C5** — training/prefill의 chunk-내부 token 병렬성을 autoregressive decode에 그대로 적용 불가(decode엔 batch축만). (iv) **C9** — "attention보다 복잡"은 타당하나 "더 크다"로 일반화 불가(fast-weight는 고정크기, KV는 길이∝). 원분석의 절대화 표현("HBM 재사용 0", "항상 memory-bound")은 codex가 경고했으나 여기 C1 텍스트엔 없음 — 페이퍼 반영 시 이 절대화만 피하면 됨. **C1/C10의 attention·MoE 유비**는 form-level 한정을 유지하는 한 정확(사용자가 이미 한정).

---

## 4. 반영 권고

**(a) 어느 장/절에**
- ch24 제안 G / ch23 §23.4·§23.8 — batching feasibility의 정식화 자리.
- ch20 §20.2(update form)·§20.5(chunk 병렬)·§20.3(backward-in-decode) — 수학 form 자리.
- claim3 headline(`experiments/results.json`) — 1줄 gloss.

**(b) 무엇을**
1. **BMM/grouped-GEMM 정식화**(C1/C2/C4): `[B,Dout,Din]·[B,Din,Din]·[B,H,D]/[B,D,H]` 명시적 batched 텐서 shape 전개 — 단 **A_t는 identity-minus-rank-1로 factored 유지**(dense materialize 금지) 각주, 그리고 dot-product 특수형↔MLP 일반형(∇L 항) 구분 각주.
2. **batch×chunk 2D 병렬성 분리**(C5): 단 **prefill/training에서만 chunk축 유효, decode step엔 batch축만** 명시 — 페이퍼가 다루는 memory-bound decode에 특히 필요한 경계.
3. **never-average 원칙**(C7): inner=map(no reduction)/outer=meta-average를 B_max 벽의 미시 근거로 추가.
4. **"Paged Fast-Weight State Manager" ↔ claim6/제안 I**(C9): 명명을 제안 I·5개 누락 API(update_in_place, mark_dirty/writeback, checkpoint/rollback, bind_to_sequence, free_on_update)에 연결 — 단 상태 크기는 fixed vs KV-∝length로 명시, 열거에 CMS 다중-frequency blocks+optimizer momentum 추가·M_q(static) 제외.
5. **workload-class 라벨 C10**: "batch별 weight 다른 recurrent MLP + batched optimizer + recurrent scan" 채택 — MoE 유비는 dispatch-form only(group당 token≈1, expert-weight 상각 없음) 경계 한 줄 병기.

**(c) claim3 서술 보정**: 본문 표현("breaks shared-weight batching")은 **유지**(정확). `results.json` headline에만 amortization gloss 추가. 페이퍼가 "batch 수학적 불가"를 주장한 적 없음을 §23.4 각주로 한 번 더 못 박아 고립-headline 오독 차단.

**(d) 정직성 계약·pair thesis 정합**: **어긋남 없음 — 오히려 강화**. batchability는 트래픽 하한을 낮추지 않아(리스크 G) memory-bound·B_max 결론 불변; G와 D가 "같은 decode step을 나눠 맡는다"는 pair thesis의 실천 그 자체. C8이 페이퍼의 NO-WALLCLOCK·fused-kernel-부재 절제를 동일하게 준수해 계약 강화. C7 never-average가 unshared 상각불가에 근거 하나를 추가.

**반영하지 않을 것**: (i) attention/MoE 유비를 *효율(amortization)* 동등으로 확장하는 서술 — form-level에 묶어둘 것. (ii) "HBM 재사용 0/항상 memory-bound"류 절대화 — L2/SRAM 상주·chunk-내 token·fusion에서 재사용 존재. (iii) C2 dense-A 텐서 표기를 구현 지침으로 승격 — factored form만 지침화. (iv) shipped Hope에 query-adaptive 서술(C3) — 중간 variant로만 각주 처리.
