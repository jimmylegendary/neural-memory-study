# Part III cross-model 판정·적용 보고서

**판정자**: Part III 검증 판정자
**날짜**: 2026-07-12
**입력**: (1) codex 발견 `audit/crossmodel-p3/ch{18,19,20,23,24}.json` (48건), (2) 연속성 감사 `audit/p3-continuity-report.md` (P3-LOW-1, MED-1, MED-2, LOW-2, OPEN-1/2/3)
**대조 기준**: `experiments/results.json` + `experiments/E1.3-kv-vs-ttt/run.py` (실측 코드), `experiments/RESULTS.md`, `style/STYLE-NOTATION.md` v1.1, `dossier/PRE-RESEARCH.md` §3–§5 정직 판정, `study-kr/part3/ch18–ch25` 실제 본문
**대상 파일**: `study-kr/part3/ch{18,20,21,22,23,24,25}.md`

---

## 0. 총평

codex 발견 48건의 **다수(약 30건)는 stale-snapshot 류 false positive**다. codex가 `results.json`의 `meta.thesis`·caveat만 보고 판정했거나 초기 스냅샷을 본 탓에, **본문이 이미 지킨 정직성 계약을 위반으로 오인**했다. 실제 ch18–25는:

- 모든 절대치(1.92 ms, 46.5 mJ, B_max, GB/s)에 "roofline 하한 / A100 runbook 이월" 꼬리표를 **매 등장마다** 단다(§18.6 정식화 + ch20·23·24·25 재선언).
- 모든 scratchpad/PIM 수치에 `NOVEL-SIM-FALSE` / directional DSE 태그를 단다.
- 두 FORCED 방향(훈련용 새 소자, 일반 PIM)을 **다섯 곳에서 능동적으로 배제**한다(ch19 §19.3, ch20 §20.4/§20.5, ch21 §21.3, ch24 §24.6, ch25 §25.2).

따라서 "절대치를 실측으로 승격" / "scratchpad·PIM을 확정 device로" / "훈련에 새 소자 필요라고 결론" 계열 codex 발견은 **전부 INVALID**(본문이 이미 codex의 suggested_fix를 실행 중).

동시에 codex·연속성 감사가 **진짜로 유효한 결함 12개 클래스**를 짚었다 — 대부분 표기(L_layer/byte factor)·정직성 phrasing 미세 과장·NS-5 복잡도 over-generalization·reuse=0 과잉 단정이다. 이들을 **pair-thesis·정직성 계약 강화 방향으로만** 본문에 적용했다.

---

## 1. VALID → 적용 (13 클래스, 22개 Edit)

| # | 출처 | 위치 | 판정 근거 | 적용 |
|---|---|---|---|---|
| V1 | 연속성 P3-MED-1 + codex ch20 #1·#2 | ch20/21/23/24/25 전역 | STYLE v1.1 §1.2/§1.5는 무첨자 $L$을 sequence 길이 전용으로 예약, layer 수는 $L_{\mathrm{layer}}$. ch19만 준수, 나머지 위반. 추가로 표 20-1 RMW=$2md^2L$은 **byte가 아니라 원소 수**(dtype byte $s$ 누락) | ch20·21·23·24·25의 layer-count $L$ → $L_{\mathrm{layer}}$ 전량 교체; 표 20-1·§20.2·checklist의 byte 식을 $2md^2L_{\mathrm{layer}}s$로($s$=dtype byte 명시). ch18에도 $md^2L$→$L_{\mathrm{layer}}$ |
| V2 | 연속성 P3-MED-2 | ch18 표 18-1, §18.5 | claim 5(cadence→tier)의 실제 본진은 ch23 §23.6·ch24 §24.3.2(부차 ch20 §20.6). ch19에는 frequency-tier 내용도 figure exp-d도 없음 | 표 18-1 claim 5 "주 담당 장" 19,24 → **20,23,24**; §18.5 23장 로드맵에 frequency-tiered placement(claim 5) 추가 |
| V3 | codex ch20 #3 (critical, 핵심만) | ch20 §20.2·§20.3·§20.4 | 전체 $d\times d$ Newton–Schulz 반복은 연산 $O(\kappa d^3)$·트래픽 $O(d^2)$라 AI가 $O(d)$ → **고립 kernel로는 compute-bound**. "연산·트래픽이 함께 $md^2$로 자라 AI 일정"은 GEMV 계열에만 성립. 또 $\kappa$는 backward inner pass 반복이 아니라 gradient 이후 별도 NS 반복 | §20.2에 "함께 $md^2$" 기제를 GEMV 계열로 한정 + NS-5 고립-compute-bound/step-내-traffic-지배 명시; §20.4 line 재정합; §20.3 $\kappa$ 문구를 "별도 행렬 반복"으로 교정 |
| V4 | codex ch20 #(minor) | ch20 §20.2 | "더 무거운 memory class는 더 memory-bound"는 부정확(AI 일정이면 bound class 동일, ridge와 상대거리 동일) | V3 편집에 흡수: "bound class 유지, 총 RMW 트래픽·latency만 커진다"로 |
| V5 | codex ch20 #6 | ch20 §20.6 | per-layer scratchpad 7.4× 이득을 "실재하는 이득"으로 서술 → `simulation_ready=False` directional을 measured fact로 승격하는 어감 | "이 배치의 방향만 load-bearing, NOVEL-SIM-FALSE directional DSE로만 인용"으로 교체. fit→spill "측정되는 물리적 절벽"은 host E2.2가 뒤에서 실측하므로 "host silicon에서 직접 측정되는"으로 명확화, scratchpad 이득은 "analytic twin 예측"으로 분리 |
| V6 | codex ch18 #7 / ch24 #6 | ch24 §24.5 G1 | reuse "구조적으로 0"은 content-addressed(block-hash) 재사용에만 참. 같은 prefix의 서로 다른 요청은 결정론적으로 동일 state snapshot에 도달 → prefix-snapshot+copy-on-write 가능. results.json missing_api(checkpoint/bind_to_sequence)와 정합, 오히려 "새 manager 필요" 논거 강화 | G1에 "content-addressed 한정 + prefix-snapshot/CoW 원리상 가능, append-only manager가 그 API 부재인 것이 새 manager 필요 이유" 추가 |
| V7 | codex ch18 #5·#6 (핵심만) | ch24 §24.3.2 제안 E | cadence→tier는 평균 대역폭·energy를 amortise하지만, 접근이 발생한 token의 **동기 latency**는 prefetch/async 없이는 상각 안 됨. tier admissibility엔 transfer size·link latency·prefetch window·동시 sequence도 필요 | 제안 E [평가]에 "cadence는 한 입력 변수인 directional 가설, 동기 latency는 별도 제약, 전량 directional DSE" 캐비엇 추가 |
| V8 | codex ch20 #5 (핵심만) | ch20 §20.5 | Atlas MFU(5–10%)를 작동점 선택으로 귀속하며 "kernel 비효율 아님"을 단정. MFU(vs peak)와 roofline-attainable efficiency는 다른 지표이고, host $C{=}1$이 attainable의 ~12%라면 memory-bound 상한 아래에 구현 손실도 존재 | "roofline 위에서 읽으면 큰 몫은 작동점" + "Atlas kernel profiling 없어 구조적 몫/구현 overhead 몫을 정량 분해 못함" 캐비엇으로 완화(작동점 = 최대 원인이라는 무근거 순위 제거) |
| V9 | codex ch20 #7 (핵심만) | ch20 §20.7 | "backward-capable decode 소자도 존재하지 않는다"는 "GPU가 backward 못 함"으로 오독 가능. 정확한 부재는 **전용 fused deep-memory decode kernel + 그 backward를 decode 경로에 품는 serving runtime**이지 backward 능력 자체가 아님 | "전용 artifact 부재(기존 GPU의 backward·grouped-GEMM 실행 능력과는 별개)"로 정밀화 |
| V10 | 연속성 P3-LOW-2 | ch22 §22.2 | TNT 배속 17.37×가 ch21/23의 책 표준 17.4×와 불일치 | 17.37× → 17.4×(perplexity 23.09/25.07은 원값 유지) |
| V11 | 연속성 P3-OPEN-3 | ch18 §18.3 | $m$을 정의 없이 anchor 식에서 먼저 사용 | state multiplier $m$ 1줄 정의 삽입(+ 동시에 $md^2L$→$L_{\mathrm{layer}}$) |
| V12 | 연속성 P3-LOW-1 | ch18 §18.1 | ch17 인용 verbatim "생겼다**는 것** —"을 "생겼다 —"로 축약(인용부호는 verbatim 함의) | "생겼다는 것 —"으로 복원 |

적용 Edit 총 22회(ch18×5, ch20×9, ch21×1, ch22×1, ch23×3, ch24×3, ch25×2 중 L_layer 잔여 1 포함). 모든 편집은 v1.1 규약·pair-thesis·정직성 계약 강화 방향으로만 이루어졌고 load-bearing 수치(6.44 GB, 0.59, 394×, S\*=65k/131k, B_max, d\*=2896, 3.9× 등)는 **불변**.

---

## 2. INVALID (필터한 false positive)

### 2.1 정직성 계약 "이미 지킴"을 위반으로 오인 (stale-snapshot) — 최다

| codex 발견 | 이유 |
|---|---|
| ch18 #8, ch19 #2·#3, ch23 #1, ch24 #3 (절대치를 실측/확정 성능으로) | **INVALID.** §18.6이 계약을 정식화하고, 1.92 ms·46.5 mJ·B_max·GB/s는 등장마다 "roofline 하한 / A100 runbook 이월" 태그를 단다(ch20 §20.2 line, §20.7; ch23 §23.9 6항 이월; ch25 §25.2 [평가]). 본문이 codex의 suggested_fix를 이미 실행 |
| ch19 #4, ch23 #4, ch24 #4 (scratchpad/PIM을 확정 device 효과로) | **INVALID.** 6.8×/14.9×/7.4×/1.6×는 전부 `NOVEL-SIM-FALSE`·directional DSE로 태그됨(그림 20-2·23-1 캡션, §24.3.1 제안 D). ※단 §20.6의 "실재하는 이득" 한 구절만 어감 과장 → V5로 교정 |
| ch18 #10, ch19 #7, ch23 #7, ch24 #7 (훈련에 새 메모리 소자 필요라고 결론) | **INVALID — 가장 명백한 stale-snapshot.** 본문은 정반대로 5곳에서 명시 배제. ch24 §24.6 "배제 1"의 제목이 문자 그대로 "훈련이 새 메모리 소자를 요구한다 [배제]". codex가 본문을 읽지 않고 results.json thesis만 본 결과 |
| ch23 #6, ch18 #9 (M-ASSUMED scaling을 경험 법칙으로) | **INVALID.** 표 19-2·19-3·23-1 모두 `REL·M-ASSUMED` 태그 + hypo-7B/70B 표시 + §19.5 "제안·미fit" + §19.6 A3 falsification |

### 2.2 실측 코드·정직 판정과 대조해 기각

| codex 발견 | 이유 |
|---|---|
| ch18 #2, ch23 #2, ch24 #1 (KV read 계수 2 누락 → S\*를 32.8k/65.5k로 반감) | **INVALID.** `E1.3 run.py` line 52 및 caveat: `KV_DIM = 1024 (E3 convention; folds K,V heads x head-dim)` — kv_dim이 **이미 K,V를 folding**한다. read=$S\cdot k_{dim}$이 전체 캐시 1회 읽기로 정합하고 S\*=65536은 E3·hatir·plan **3자 1% 이내 합치**(코드 `_validate` assert). codex는 kv_dim을 한쪽 폭으로 오해. (write append의 미세 내부 이중계상은 0.003% 비율값에만 영향, S\*·load-bearing crossover엔 무영향.) 게다가 본문은 S\*를 directional로 태그하고 실 kernel head-to-head 검증을 A100 runbook carry-forward #4로 이미 이월(ch23 §23.9) |
| ch24 #2 (S\*∝$md^2$ 일반화가 kv_dim 스케일 누락) | **INVALID.** 식 (19-4) $S^*\propto md^2/d_{kv}$로 이미 $d_{kv}$로 나눔; 표 19-3·§23.8이 "fp8은 2배, MHA는 선형 collapse"를 명시 |
| ch18 #1 (C=1 대칭 RMW 일반화, 청크 갱신 미고려) | **INVALID.** C=1 = decode 영역이 pervasively 명시(claim7, §20.5, §19.5, §23.2). 청크 갱신은 본문이 명시적으로 accelerator(큰-C) 영역으로 분류. autoregressive decode는 정의상 per-token |
| ch18 #3, ch19 #1 (CPU/H100 C\* 서로 다른 FLOP 식) | **INVALID.** 본문은 CPU-SHAPE 태그로 곡선 **모양·crossover 존재**만 승격, $C^*$ 절대값(306–430/337)은 ridge 의존·비-load-bearing으로 명시("하드웨어별로 다시 재야 한다", ch23 §23.2). E2/E3 내부 회계 차이는 본문 주장으로 표면화되지 않음 |
| ch18 #4, ch20 #(minor 0.59/2.25) | **INVALID.** 본문은 0.59로 일관, 두 값 모두 ridge 295의 두 자릿수 아래라 bound 결론 불변. 2.25는 results.json 주석에만 존재, 본문 내 모순 없음 |
| ch18 #11, ch19/ch23 (KV capacity-bound vs bandwidth-bound "정반대" 과일반화) | **INVALID.** "정반대"는 **batch 병목 자원**(KV=capacity 상한, TTT=BW 상한)에 한정된 서술이고, 본문은 KV read-many wall·문맥따라 latency 증가를 §23.3·§23.8에서 명시 인정(0.785→2.581 ms) |
| ch20 #4 (20.4 PIM 경계, GEMM형 PIM 배제 과일반화) | **INVALID.** §20.4는 이미 "본 E1.2 twin/평가 범위"로 한정 + epilogue만 directional + "PIM은 못 태우는 GEMM이 대부분"은 이 라인의 dense-matmul 엔지니어링 패턴에 근거한 방어 가능 서술 |
| ch18 #6-scratchpad, ch20 (residency flat-tier vs two-tier) | **INVALID.** 본문은 whole-model이 268MB에 안 들어가 spill, 상주=per-layer/streamed임을 명시(claim3 ondie 불가, residency crossover $d^*=2896$) |
| ch19 #(minor) 교차검증 <1% vs 1.2% | **INVALID/negligible.** 도구 3법(E3/E1.1/E1.3)은 <0.1% 합치, plan 1.9는 반올림 계획값. 연속성 감사도 수치 일관성 PASS |
| ch19 #5, ch18 #5 (cadence→tier 과일반화) | **대부분 INVALID** — 본문은 read/write cadence를 분리(CMS-chunk는 read가 pin해 HBM 유지), 전량 NOVEL-SIM-FALSE·"제안"으로 태그. ※latency-vs-amortization 하위 논점만 유효 → V7로 반영 |

---

## 3. 유효하나 이번 라운드 미적용 (후공정 이월)

- **P3-OPEN-1 (GDN 인용 확정).** ch22 line 49 TODO-VERIFY: Gated DeltaNet arXiv ID·NVIDIA affiliation. 외부 검증 필요(오프라인 확정 불가) → **서지/P4 공정 이월**. ch21 §21.4의 "NVlabs/GatedDeltaNet 공식 구현"과 표면 긴장은 원저자≠호스팅으로 모순 아님.
- **P3-OPEN-2 (figure 임베드 통일).** ch21 §21.3·ch22 §22.1–22.2는 `<!-- FIG -->` 주석+캡션만 두고 이미지 미임베드. 두 장은 figure 소유가 9/15/20장이라 **의도적 중복 회피**로 판단 → 렌더 정책 결정 사항(cosmetic), 본 라운드 보류.

---

## 4. 판정 요약

- codex 48건 + 연속성 7건 = **55건 검토**.
- **VALID 13 클래스 → 22 Edit 적용**(표기 L_layer/byte factor, claim-5 매핑, NS-5 복잡도 scoping, 정직성 phrasing 3건, reuse CoW 좁힘, cadence latency 캐비엇, 17.4×, m 정의, 인용 verbatim).
- **INVALID 약 40건** — 압도적 다수가 "정직성 계약을 이미 지킨 것을 위반으로 오인"(절대치/novel-twin/훈련-새소자)한 stale-snapshot false positive, 그리고 실측 코드(kv_dim folds K,V)·CPU-SHAPE 태그를 오해한 factor-2/C\* 발견.
- **미적용 이월 2건**(GDN 인용=서지, figure 임베드=cosmetic).
- 모든 편집은 pair-thesis 척추·정직성 계약을 **강화**하는 방향이며 load-bearing 수치·서사는 불변. 척추(D4 pair thesis), FORCED 배제, 정직성 3층 태그는 편집 후에도 완전 유지.
