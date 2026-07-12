# Part III 심화판(P3B) cross-model 판정·반영 보고서

**판정자**: Part III 심화판 crossmodel 판정자
**날짜**: 2026-07-12
**입력**: (1) codex 발견 `audit/crossmodel-p3b/{ch19,ch20,ch23,ch24}.json`(총 45건),
(2) 연속성 감사 `audit/p3b-continuity-report.md`(P3B-LOW-1/2 + OPEN-1/2)
**대조 기준**: `experiments/results.json`(claim1–8 + cross_validation + global_caveats) +
`experiments/E4-scaling/results.json`, `style/STYLE-NOTATION.md` v1.1, `dossier/PRE-RESEARCH.md` §3–§5,
현행 `study-kr/part3/ch18–25`.
**판정 원칙**: 각 발견을 실측 warrant·원전 회계·STYLE·dossier 정직 판정과 대조. **stale FP**(이미 REL/
M-ASSUMED/NOVEL-SIM-FALSE/directional 태그로 계약을 지킨 것을 위반으로 오인) 필터. **valid**는 pair-thesis·
정직성 강화 방향으로만, **load-bearing 수치 불변**으로 해당 장에 직접 Edit.

---

## 요약

- 총 판정 = 45 codex + 2 continuity(LOW-1/2). **valid 26 / invalid·defer 21.** (LOW-1 = ch20 matrix 발견과
  동일 항이라 중복 제외.)
- valid 26건 전부를 ch19·ch20·ch23·ch24·ch25에 직접 반영(28 edit; 0.5× 등 일부는 두 장에 걸쳐 각각).
- **load-bearing 수치는 하나도 바뀌지 않았다.** 6.44 GB/token·394×·0.59/2.25·S\*(65k/131k)·B_max(20.8/5.2/
  0.97)·3.9×·d\*≈2896·ρ 통계·retention 6.7× 전부 불변. 반영은 **범위 태그·라벨·회계 분리**뿐이다.
- codex "critical" 다수는 **stale FP**로 판정: NS-5 whole-step, E2.1 일반화, analytic-as-measured, 2k/32k
  등은 이미 §20.2.1/20.4.1/20.5.1·REL·CPU-SHAPE로 계약을 지켰거나(§23.8은 이미 65k/131k 분리) 표준 framing.
- E2.1↔E3 **FLOP 식 불일치**(C=1 AI 1.0 vs 1.5, H100 C\* 337 vs 6Cd² 재계산 ~223)는 **warrant 층 항목**으로
  분리: 절대 C\*·AI는 정직성 계약상 비-load-bearing·directional(CPU-SHAPE)이라 본문 수정 대상 아님. 실험 소유자
  대조 권고(아래 §후공정).

---

## (A) VALID — 반영 완료 (26건 / 28 edit)

### ch19 (3)
1. **[19.6] capacity ablation을 "recall 개선"으로 오명명** (codex major). `22.14→19.97`은 **LM ppl** ablation인데
   본문이 "recall이 오른다"로 불러, 이 장 자신의 ppl≠retention 논지와 충돌. → "LM ppl이 개선되는 ablation"으로
   교정 + poly가 표현·최적화도 함께 바꿔 capacity 단독 분리 불가·retrieval 이득으로 읽지 말 것 명시.
2. **[19.7a 식 19-2] Chinchilla를 ppl에 직접 적용** (codex major). additive power law는 loss $\mathrm{L}=\log\mathrm{ppl}$
   위에서 성립. → (19-2)를 loss-domain($\mathrm{L}=\log\mathrm{ppl}\approx E+A/N^\alpha+B/D^\beta$, $\mathrm{ppl}=\exp\mathrm{L}$)으로
   교정, $E_{\mathrm{class}}$를 데이터/protocol 엔트로피 항(+class 이동)으로 재정의. (제안 형태·미fit이라 수치 무변.)
3. **[19.7c] read-crossover 65k를 full-system crossover처럼 결론** (codex major). → $S^\star_{\mathrm{read}}$(65k)와
   $S^\star_{\mathrm{rmw}}$(≈131k, full RMW) 두 문턱 명시, 그 사이는 회계에 따라 갈리는 혼합 구간(§23.8과 정합).

### ch20 (6)
4. **[20.2] AI 0.59의 "세 방법 1% 합치" 오귀속** (codex major). 1% 합치는 RMW 트래픽(6.44)에 대한 것이고 AI는
   E3=0.59/E1.1=2.25로 4배 차. → 1% 합치를 6.44에 scope, 0.59(E3 단순 회계)와 2.25(E1.1 GEMV-inclusive)를
   분리 병기(둘 다 ≪ ridge 295). results.json 주석과 정합.
5. **[20.2.4 + 표 20-1] HOPE 6-block per-token RMW = resident 혼동** (codex major). E3 주석: "가장 빠른 level만
   매 token RMW($m_{\mathrm{fast}}{\approx}8$), 나머지는 저빈도 상주." → 표 20-1 Hope 행을 "상주 최대·per-token은
   fast-level 지배"로, §20.2.4를 fast-level-RMW/저빈도-resident 분리로 교정(update-frequency 지점 강화).
6. **[표 20-1 matrix 행] $m{\approx}2$ 라벨 충돌** (codex minor = continuity **P3B-LOW-1**). → 행을 "matrix(+momentum)"로
   재라벨 + 캡션에 순수 matrix=GDN $m{\approx}1$(§19.4·§20.2.1·E4 정합) 각주. §20.2 본문도 순수 $m{\approx}1$/+momentum
   $m{\approx}2$로 정정.
7. **[20.2 해설] "1.3B weights 절반쯤" 5배 오류** (codex minor). bf16 1.3B≈2.6 GB → RMW 6.44는 ≈2.5× weights.
   → "state 3.22 GB read+write=6.44 GB, weights 전체를 매 token ≈2.5회 왕복"으로 정정.
8. **[20.2 해설] KV write 0.003% 문맥-무관 제시** (codex minor). 비율은 $O(1/S)$. → "anchor $S{\approx}65\mathrm{k}$;
   $O(1/S)$" scope.
9. **[20.6 + 요약] "RMW=read의 0.5× element throughput 직접 측정"** (codex major). 실측 사실은 element당 2× byte;
   0.5×는 동일-대역폭 환산 corollary(측정 element rate는 ≈동일). → 측정=2× byte(write-back 세), 0.5×는 환산으로
   재framing. 그림 20-3·요약 동시 반영.

### ch23 (8)
10. **[23.2] "세제곱 근처 성장(∝d²)" 자기모순** (codex major). RMW∝$m d^2 L$. → "per-layer $\propto d^2$·whole-model
    $\propto d^2 L_{\mathrm{layer}}$·canonical shape에서 param count에 거의 선형"으로 정정.
11. **[23.2] 70B를 단일 HBM 상주로 100 ms 산출** (codex major). 343.6 GB > 80 GB HBM. → "sharding 전제, 집계
    대역폭 roofline"으로 scope.
12. **[23.5 G1 + 표 23-4] reuse=0 "구조적"** (codex major). 0은 block-hash 재사용에 한정; 결정론적 prefix-snapshot
    +COW는 원리상 가능(새 매니저 근거). → ch24 §24.5가 이미 가진 scope를 ch23에 이식(정합).
13. **[23.5 G3 + 표 23-4] 3.05 mJ/130 µs를 whole-state full writeback으로** (codex major, **검증**: 3053.5 µJ÷7 pJ/B
    =416 MiB, ≠ 3.22 GB whole-session; whole-session=22.5 mJ/962 µs, 7.4×). → "E1.5 416 MiB eviction set; whole-session은
    footprint 비례"로 scope.
14. **[23.6] CMS-mid read cadence=4096 단정 → CXL tier** (codex major). E1.4 caveat: read cadence는 공개 trace 아닌
    가정. → [해설]에 조건 명시: NL Eq.70/Sleep Eq.1은 CMS block을 매 token forward-호출(update만 주기적)이므로,
    read down-sampling은 sparse routing·cached activation일 때만; 매 token forward면 gating rule이 HBM pin.
15. **[23.7] dream fan-out=5 일반화** (codex minor). → "$m{\ge}1$(CPT 실험 $m{=}5$)".
16. **[23.8 해설] 2k/32k full-serve latency를 state-only crossover 증거로** (codex critical). 0.785/2.581은
    full dense-serve; pure KV-read는 65k에 0.96 ms. → crossover는 pure-traffic 축, 2k/32k는 full-serve trend로
    분리 명시(1.923 state-only와 직접 비교 아님).
17. **[23.9] "모든 결론 1% 3-way 합치"** (codex critical). cross_validation은 anchor scalar(RMW/S\*_read/state/
    B_max/C\*-shape)만. → 1% 합치를 그 scalar에 scope; KV-manager·tier·sleep·prefill-bound는 단일/​directional로
    분류(scratchpad 6.8× vs 7.4× 예시).

### ch24 (8)
18. **[24.3.2] NL/Sleep update-cadence를 read cadence로 치환해 CXL 정당화** (codex **critical**). 위 #14와 동일 뿌리.
    → "정직한 경계" 단락 신설: Eq.70/Eq.1의 매 token forward-호출 vs Eq.71/Eq.2의 주기적 update 구분, CXL은
    sparse-routing/cached-activation/offload 입증 시 조건부.
19. **[24.2.2] stale-snapshot을 mismatch의 확정 원인으로** (codex major). TNT Fig.2는 550M·$C_{\mathrm{train}}{=}64$
    단일 설정·인과 미규명. → "유력한 후보 기제·제안 A가 검증할 가설"로 하향.
20. **[24.4 F] "momentum이 chunkwise closed form을 깨뜨린다"** (codex major). Titans Eq.18은 momentum을 parallel
    scan 가능한 선형 recurrence로 명시; fla naive는 구현 공백. → "fused momentum/retention 커널 부재(구현 공백인지
    병렬화 난점인지는 F의 열린 질문)"로 reframe([리스크] F와 정합).
21. **[24.4 G] grouped-GEMM이 AI를 올린다** (codex major). request별 unshared $W$는 FLOP·traffic 함께 $B$에
    비례해 AI 불변. → 이득을 launch·occupancy·scheduling(포획 손실 감소)으로 한정, AI 상승엔 session-shared 재사용
    필요 명시([리스크] G와 정합).
22. **[24.5 free_on_update] retention gate $\alpha_t$를 물리 free 신호로** (codex major). Eq.13 $\alpha_t$는 in-place
    decay; $\alpha{=}0$이어도 활성 sequence는 tensor 유지. → decay-in-place·cold 비용 신호로 재정의, 물리 free는
    sequence lifecycle이 결정(§23.4와 정합).
23. **[24.5 G3] "eviction은 반드시 full writeback"** (codex major). 결정론적 prefix-replay 재계산 대안 존재. →
    writeback OR replay 중 싼 쪽; 3.05 mJ를 416 MiB set으로 scope(#13과 정합).
24. **[24.3.1/24.2.1 그림] 0.5× throughput measured** (codex major). #9와 동일. → 2× byte(측정)/0.5×(환산) 재framing.
25. **[24.3.1] PIM latency "3 MAC/elem 넘겨야 물린다"** (codex minor). 격자 3 MAC/elem에서 이미 2.09×. → "1–3 사이
    뒤집혀 3에서 이미 물림(1 MAC/elem=0.7× 이득)"으로 정정.

### ch25 (1)
26. **[25.4] `<!-- FIG: exp-a -->` 포인터 오류** (continuity **P3B-LOW-2**). E4 fit의 전용 그림은 exp-f. → exp-f로 교체.

---

## (B) INVALID / STALE FP — 반영 안 함 (근거)

| # | 발견 | 판정 근거 |
|---|---|---|
| ch19-4 | 모든 모델이 whole-state를 매 token RMW·context state 없음 | anchor=attention-free canonical LMM($m$-표에 attention $m{=}0$·hybrid 명시), M-ASSUMED tag + §20.2.4·§23.6이 multi-cadence 처리 → 이미 scope |
| ch19-5 | analytic 일치를 "실측/measured"로 승격 | REL·"roofline 하한·A100 이월" 전장 태그, §19.8 [평가]가 analytic-twin vs measured 구분 → stale |
| ch19-1 | 표 19-1 Atlas 760M 19.97 Wiki=18.92 | 외부 table 대조(continuity OPEN-2). 오프라인 확정 불가·비-load-bearing → 서지 이월 |
| ch19-2 | 56.82 앵커 cross-paper ordering | §19.3이 정렬된 tokenizer·재실행 baseline으로 legal 경계 명시, ordinal-only → stale(정직 계약 준수) |
| ch19-3/10 | 760M→1.3B N-dep, hypo-7B/70B·268MB 확정 | 표가 이미 hypo-·M-ASSUMED·directional; §19.5A가 $N,D$ co-scale·compute-axis 명시 → stale |
| ch19-9 | TTT bw-bound/KV cap-bound 보편 반대 | §23.8 [해설]이 KV latency 문맥 상승 인정; 표준 framing(pair 대비) → 표준 |
| ch20-1 | Atlas NS-5까지 C=1 whole-step memory-bound 확정 | §20.4.1이 유효 AI≈6.7(directional)로 상각 + 얇은 matrix+Muon≈53 회색지대 명시 → 이미 정직 |
| ch20-8 | E2.1(DeltaNet)을 전 class 보편 roofline으로 일반화 | §20.2.1–20.2.4가 class별 AI 식 분리, §20.5가 CPU-SHAPE로 shape-only 한정 → stale |
| ch20-9 | 5–10% MFU를 Atlas에 오귀속(실제 TNT) | 외부 인용 대조(OPEN-2), 오프라인 확정 불가 → 서지 이월(플래그) |
| ch20-10, ch24-2 | 3.9× cliff = peak÷DRAM(plateau 아님) | results.json이 3.9를 "boundary headline"으로 canonical화 + 본문이 265/68 endpoint 투명 노출·load-bearing 비율 불변 → 유지(캡션에 "in-cache peak÷DRAM" 명기로 보강) |
| ch23-1, ch24-1 | C=1 AI=1.0(실제 1.5)·H100 C\*=337(6Cd²면 ~223) | E2.1 vs E3 FLOP 식 warrant-층 불일치. 절대 AI·C\*는 비-load-bearing·directional(CPU-SHAPE)·bound class 불변 → 본문 무수정, 실험 소유자 대조 권고 |
| ch23-2 | prefill/training 무조건 compute-bound | 본문이 "큰 $C$"로 조건화·disaggregation을 directional로 제시 → stale |
| ch23-5 | KV 무조건 capacity-bound | [해설]이 KV latency 문맥 상승 인정, pair 대비의 표준 framing → 표준 |
| ch23-10 | 65k/131k 단일 S\* 혼동 | ch23 §23.8은 이미 "read 65k·전체 RMW 131k" 분리 서술 → 이미 정직(ch19에만 이식) |
| ch23-6(G2) | 256×를 구조적 속성으로 | 이미 "append-only·no-eviction" scope(본문 "in-place update 없는"), "no-eviction worst case" 보강만 | 
| ch24-5 | analytic을 "measured structure"로 | REL·"구조 vs 절대" 구분 전장 반복 → stale(문구 유지) |
| ch24-10 | G1 content-addressed reuse 재사용 가능성 누락 | ch24 §24.5가 **이미** prefix-snapshot+COW scope 보유 → stale(ch23에만 이식) |

---

## (C) 후공정 이월(P4·서지, 오프라인 확정 불가)

- **[warrant-consistency]** E2.1(`6Cd^2+4C^2d`) ↔ E3(`4Cd^2+4C^2d`) FLOP 식 불일치 → C=1 AI(1.0 vs 1.5)·H100
  C\*(337 vs ~223) 재계산 필요. 두 값 모두 정직성 계약상 **비-load-bearing directional**(bound class·"곡선 모양"만
  이전)이라 본문은 무수정. 실험 소유자가 두 실험의 FLOP 회계를 통일하면 results.json·그림 갱신 권고.
- **[OPEN-2 서지]** ch19 표 19-1 `19.97`(Atlas Table 6 열)·`5–10% MFU`(Atlas vs TNT 귀속)·table 번호·star·issue·
  arXiv ID = 외부 대조 대상. 본문 내부 일관이라 정직성 결함 아님, P4 검증.
- **[OPEN-1]** GDN 인용(arXiv·affiliation) 미확정 — 서지 공정.

---

## 판정 근거 요약

- **정직성 계약 준수 재확인**: 반영 26건은 전부 범위 태그·회계 분리·라벨 정정이며 load-bearing(비율·crossover·
  tier·bound) 수치 불변. pair thesis 척추·memory-centric 두 지점 규율 강화 방향.
- **stale FP 필터 성공**: codex critical 6건 중 4건(NS-5·E2.1일반화·2k32k·analytic-measured 계열)은 이미 계약
  준수로 stale; 2건(CMS-mid read-cadence·23.9 1%-합치)은 genuine → 반영.
- **가장 값진 반영**: ch24 §24.3.2/ch23 §23.6의 **CMS read-cadence 조건화**(update≠read cadence, gating rule의
  "빠른 쪽"이 실제 down-sample일 때만 CXL admissible)와, ch23 §23.9의 **1%-합치 scope 축소**(anchor scalar에만).
  둘 다 memory-centric tier 주장을 원전 회계에 정확히 다시 못박아 정직성을 높였다.
