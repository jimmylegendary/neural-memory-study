# Part III 심화판(P3B) 연속성·정직성 감사 보고서

**감사자**: Part III 심화판 연속성·정직성 감사자
**날짜**: 2026-07-12
**대상**: `study-kr/part3/ch18–ch25` (심화 라운드 현행판)
**기준**: `style/STYLE-NOTATION.md` v1.1, `dossier/PRE-RESEARCH.md` §3–§5(정직 판정),
`experiments/{RESULTS.md, results.json}` + `experiments/E4-scaling/results.json`(신규 warrant),
`notes/*.json`, 그리고 선행 라운드 `audit/{p3-continuity-report.md, p3-crossmodel-report.md}`
**판정**: **PASS (조건부)** — 심화가 서사·pair thesis 척추·정직성 계약을 **훼손하지 않았다**. 새 FORCED
crossing·절대치 승격·cross-paper ppl 오비교는 **발견되지 않았다**. E4 인용은 tokenizer 이질성을 명시해
정직하다. 8장 전부 예산 깊이에 **도달(초과)** 했고 심화는 실질(수식·기제·정직 캐비엇)이지 패딩이 아니다.
잔여는 low/cosmetic 2건 + 후공정 이월(bibliography) 2건뿐이다.

---

## 0. 요약

선행 P3 라운드가 지적한 두 medium(L_layer 표기, claim-5 매핑)과 다수 minor는 이미 반영되어 있고, 이번
심화 라운드는 그 위에 (i) 신규 E4-scaling 실험을 §19.5–§19.8 정량 백본으로 통합, (ii) ch20에 NS-5
FLOP-density·MFU 분해·Atlas 세 얼굴·Hope 6-memory 세부, (iii) ch24에 near-memory update engine
3-stage 마이크로아키텍처·migration 스케줄·proposal J(learned schedule), (iv) ch21 systolic-array
역사화·Titans 커널화 기계적 해부, (v) ch22 3-겹 자산·fla governance, (vi) ch23 batching 완화책·
checkpoint format·speculative rollback 정량, (vii) ch25 5-어젠다 확장을 더했다. **모든 신규 절이
정직성 3층 태그(비율 load-bearing / 절대치=roofline 하한·A100 이월 / novel-twin=directional)를 달고
있으며, 두 FORCED 방향은 여전히 다섯 곳에서 능동 배제된다.**

핵심 헤드라인 수치(6.44 GB/token, 0.59 FLOP/B, 394×, S\*=65k/131k, 1.92 ms·46.5 mJ, 6.8×/14.9×,
7.4×, d\*≈2896, 3.9× cliff, 3.05 mJ/130 µs, 0.45→6.44→343, 16k→1.05M, B_max 20.8/5.2/0.97,
104/26/4.87)와 신규 E4 통계(ρ=-0.949 / +0.094 / -0.69 / +1.0 / +0.333, 6.7×/1.5×, 23.53→20.15,
U-curve 36.45/13.78/22.4, r²=0.996–0.999, 18.53/15.28/15.67, 14.97/14.40/15.60/16.42)는 8장 본문·
`results.json`·`E4/results.json`과 **완전 일치**한다.

---

## (a) 서사·pair thesis 척추·정직성 계약 훼손 여부 — **훼손 없음**

**bridge 체인.** ch17→ch18(초대장 인용, verbatim 복원됨)→19→20→21→22→23→24→25가 끊김 없이
이어진다. 각 장 "Bridge-in/다음 장으로"가 앞 장의 결론을 명시 인수하고, 완성형 5성분 소유권을 ch18이
17장에서 정확히 넘겨받는다.

**pair thesis 척추.** 8장 전부가 D4 pair thesis를 명시 축으로 쓴다 — ch18(정의)·ch19(scaling law
안에서 드러남, 식 19-2~19-4의 "하나의 모델, 두(세) cost 영역")·ch20(roofline 두 절반)·ch21
(hardware-lottery 비대칭이 pair split과 "정확히 겹친다", 예측 3)·ch22(player 지도 위 분업 game)·ch23
(청구서 두 장)·ch24(제안을 pair로)·ch25(재확인). 심화가 추가한 절들도 전부 어느 절반에 속하는지로
정당성을 얻는다(예: ch24 §24.4 F·G·H가 "D·E와 경쟁이 아니라 짝").

**memory-centric 두 지점 규율.** 모든 memory-centric 논증이 (①) decode state RMW 트래픽, (②)
update-frequency↔tier 두 지점에만 갇힌다. 신규 ch24 §24.3.1 near-memory engine·multi-tenant
isolation·§24.3.2 migration 스케줄은 모두 이 두 지점의 **직접 귀결**로 도입되고 "얹는 새 주장이 아니라"를
명시하며 falsifier를 단다. ch20 §20.4/§20.4.1 신규 NS-5 해부는 PIM 경계를 오히려 **더 날카롭게** 긋는다
(대부분 FLOP은 GEMM, PIM은 minority-FLOP epilogue 국소).

**정직성 계약.** §18.6이 계약을 정식화하고 ch20 §20.1·§20.2·§20.7, ch23 §23.1·§23.9, ch24 §24.1,
ch25 §25.2가 재선언한다. 심화가 새로 도입한 절대치(NS-5 유효 AI ≈6.7 FLOP/B, matrix+Muon ≈53
FLOP/B, host 4.56→689.5 GFLOP/s)는 전부 "order-of-magnitude·directional" 또는 "host 좌표, 이전
금지, load-bearing은 ~150× swing 비율뿐"으로 태그된다. **절대치를 주장 근거로 승격한 문장은 8장 어디에도
없다.**

---

## (b) 새 FORCED crossing / 절대치 승격 / cross-paper ppl 오비교 — **없음**

| 점검 항목 | 결과 |
|---|---|
| 훈련용 새 소자 옹호(FORCED 3) | 배제 유지 — ch19 §19.3, ch20 §20.5, ch21 §21.3[평가], ch24 §24.6 배제 1, ch25 §25.2 |
| 일반 PIM 옹호(FORCED 4) | 배제 유지 — ch20 §20.4[평가], ch24 §24.6 배제 2; PIM은 epilogue 소수-FLOP에서만 directional |
| 절대치→load-bearing 승격 | 없음 — 신규 절대치 전부 하한/directional/host-shape 태그 |
| novel-twin→shipping 승격 | 없음 — 6.8×/14.9×/7.4×/1.6× 전부 `NOVEL-SIM-FALSE`; 선행 §20.6 "실재하는 이득" 어감은 이미 "analytic twin 예측"으로 교정됨 |
| cross-paper ppl 오비교 | 없음 — 아래 (d) 참조 |

**신규 near-memory 마이크로아키텍처(ch24 §24.3.1)와 migration 스케줄(§24.3.2)이 FORCED를 넘지 않는지
특별 점검.** 두 절 모두 decode RMW 트래픽 + residency + update-frequency라는 인정된 두 지점에서만
파생되고, silicon 미검증 directional로 명시하며, isolation/privacy는 dossier §3.2 serving 공백의 직접
귀결로 falsifier까지 붙인다. **새 지점을 여는 것이 아니라 기존 두 지점을 소자 층으로 내린 것** — 위반 아님.

---

## (c) 장 간 수치·표기(L_layer) 일관성 — **강함 (low 2건)**

**L_layer 표기.** 선행 P3-MED-1이 완전 반영됨: prose·수식·표의 layer-count가 8장 전역에서
$L_{\mathrm{layer}}$이다(무첨자 layer-count $L$의 잔여 0건 — grep 확인). 표 20-1·식 (19-1)·(19-3)·
(19-4)·(20-1)·(23-x)·(24-x) 모두 $m\,d^2\,L_{\mathrm{layer}}$ 또는 byte식 $2\,m\,d^2\,L_{\mathrm{layer}}\,s$.
anchor code-string `neural-mem-1.3B (d=2048, m=16, L=24, GQA-8, bf16)`의 `L=24`는 ch18/20/25에서
**동일 문자열**로 등장하며 `results.json` anchor config와 축자 일치하는 데이터 라벨 — 선행 라운드가 허용한
관용이고 세 장이 균일하므로 결함 아님.

**헤드라인 수치 8장 합치(표본).** RMW 6.44(18/19/20/23/24/25) · AI 0.59(18/20/21/22/24/25) · 394×
(18–25 전부) · state/layer 134 MB(19/20/23/24) · S\* 65k/131k(18/19/22/23/25) · 하한 1.92 ms/46.5 mJ
(18/19/20/23/25) · RMW scaling 0.45→6.44→343(전장) · S\* scaling 16k→1.05M(19/23/25) · B_max@10ms
20.8/9.24/5.2/0.97/0.1(19/23, 신규 760M=9.24·70B=0.1 외삽은 1/RMW 스케일과 내부정합 ✓) · B_max@50ms
104/26/4.87(23) · scratchpad 6.8×/14.9×(18/20/23/24/25) · per-layer 7.4×(20/23/24) · d\*≈2896
(19/20/23/24) · cliff 3.9×·265→68(20/24) · dirty writeback 3.05 mJ/130 µs(23/24) · stale 256×
(20?/23/24/25) · C\* 32 host/306–430 H100(19/20/21/22/23/24) · TNT 17.4×(21/22/23, `17.37` 잔여 0건).

**[P3B-LOW-1] ch20 표 20-1 "matrix/linear memory ≈2" ↔ ch19/23 "matrix ≈1" 라벨 충돌.**
ch20 표 20-1은 첫 행을 "matrix / linear memory | $m$≈2 | ≈17 MB | ≈0.8 GB"로 적고, §20.2.1이 이를
"순수 matrix(GDN)는 momentum 없이 $m$≈1(8.4 MB/layer), momentum 얹으면 $m$≈2(≈17 MB)"로 **장 내부에서
정합**시킨다. 그러나 ch19 §19.4·요약(matrix **1**·deep-MLP 8·+momentum 16·+poly ~24·attention 0)과
ch23 §23.2(matrix $m$≈1) 및 E4(matrix GDN=1)는 "matrix"에 $m$=1을 예약한다. 즉 같은 "matrix" 라벨이
ch20(≈2, momentum 포함)과 ch19/23/E4(≈1, 순수)에서 서로 다른 배수를 이고 있어, 두 장을 교차참조하는
독자에게 표면 충돌로 읽힌다. → **권장 수정**: 표 20-1 첫 행을 "matrix + momentum"으로 재라벨하거나
각주로 "순수 matrix=$m$1, +momentum=$m$2(E4·§19.4와 정합)"를 병기. (severity LOW — ch20 내부는 이미
정합, load-bearing 순서·bound 판정에는 무영향.)

**[P3B-LOW-2] ch25 §25.4 어젠다 1의 `<!-- FIG: exp-a -->` figure 포인터 부정확.** 해당 주석은 E4
scaling-fit 서술("(params, tokens, ppl/score) 점을 digitize … capacity–ppl 역전") 바로 앞에 놓였는데,
그 내용의 전용 figure는 `exp-f-scaling-fits.png`(ch19 그림 19-2로 올바르게 사용됨)이지 KV-TTT crossover의
`exp-a`가 아니다. 어젠다 1이 RMW/crossover scaling(≈exp-a)과 E4 fit(exp-f)을 함께 다루므로 exp-a가
전자엔 맞지만 E4 climax에는 exp-f가 맞다. 렌더되지 않는 숨은 주석이라 영향은 cosmetic. → **권장**:
exp-f로 교체하거나 exp-a/exp-f 병기. (severity LOW.)

---

## (d) E4 인용 정직성(tokenizer 이질성 명시) — **정직 (PASS)**

E4-scaling(신규)은 dossier A3("capacity가 raw state-bytes 대비 예측력을 더하는가")을 공개 점 위에서
검정한 §19.5–§19.8의 정량 백본이다. 정직성은 세 겹으로 지켜진다.

1. **tokenizer 이질성을 정면으로 명문화.** §19.3이 "ppl=exp(per-token CE)의 분모 $T$가 vocabulary가
   정하므로 vocab이 다르면 좌표계가 다르다 → Titans(Llama-2) vs Atlas/TNT(T5) vs NL/Sleep(32K)는 같은
   축의 ppl이 아니다"를 수식으로 못박고, **legal한 fit 경계 = 하나의 setting_group(같은 논문·tokenizer·
   data·metric)** 를 정의한다. 표 19-1은 "tokenizer 열이 비교 가능성의 관문"임을 열로 노출한다.
2. **cross-paper는 ordinal·directional로만.** BABILong retention(§19.5C·§19.6) 등 cross-paper 대조는
   continuous fit이 아니라 Spearman $\rho$로만 읽고, "cross-paper라 continuous fit이 아니라 ordinal $\rho$
   로만"(§19.5C)·"이질적 tokenizer의 digitize된 점 위 rank-correlation으로만"(§19.5[평가])을 반복 태그.
   `E4/results.json`의 `honesty`·`caveats`가 요구한 "orderings/rank-correlations만 load-bearing, 절대
   exponent·state-bytes는 directional"이 본문에 그대로 전사됨.
3. **within-line confound·tiny-N 정직.** §19.5A가 params-only exponent를 "$N,D$ co-scale($15{\to}30
   {\to}100$B) → 순수 param-law 식별 불가, compute $6ND$가 honest axis"로 강등하고, §19.3이 "tiny-N
   (3–8점)에서 최소제곱 CI 무의미 → Spearman $\rho$가 1급 통계, $p$는 지표"임을 선언.

**A3 판정의 정직한 결말**(ch19 §19.8·ch24 §24.2.3·ch25 어젠다 1이 동일 서술): capacity는 **ppl 축에서
잉여(state-bytes와 rank-동치)이고 attention corner에서 부호 반전($\rho$: -0.949→+0.094)해 오도**,
**long-context retention 축에서만 예측력을 더한다($\rho$ 1.0 vs 0.333, Titans→Atlas 6.7× 도약을 1.5×
byte가 underpredict)**. 이는 dossier A3를 반증이 아니라 **sharply-scoped 부분 지지**로 좁힌 정직한 결론
이며, "capacity를 quality 일반 축으로 파는 것·cross-arch ppl-vs-capacity fit"을 명시 금지한다. **cross-
paper ppl 오비교는 발견되지 않았다** — 오히려 그런 오비교를 하지 말라는 것이 §19.3의 핵심 규칙이다.

---

## (e) 예산 깊이 도달 여부 — **도달·초과 (실질 심화, 패딩 아님)**

| 장 | dossier 목표(pp) | 현행 분량(char/1800 근사 pp) | 심화가 추가한 실질 절 |
|---|---|---|---|
| ch18 | 8(arc, Part II 소속) | ~6.6 | pair thesis [평가] 블록, 8-claim 표, 계약 §18.6 |
| ch19 | 9 | ~14.6 | **§19.5–§19.8 E4 정량 백본**(4 fit 묶음·A3 falsifier 3판정) |
| ch20 | 10 | ~16.7 | §20.2.1–20.2.4(class별 roofline·Atlas 세 얼굴·Hope), §20.3.1(fused kernel 요건), §20.4.1(NS-5 $O(\kappa d^3)$), §20.5.1(MFU 천장×포획) |
| ch21 | 7 | ~11.2 | §21.2.1(systolic array 역사화), §21.4.1(Titans 커널화 기계 해부·WY/UT vs companion scan) |
| ch22 | 6 | ~10.9 | §22.2 3-겹 자산(pod·컴파일러 moat·실증), §22.5 fla governance, §22.7 game 정식화 |
| ch23 | 8 | ~13.5 | §23.4 batching 4완화책·checkpoint format·speculative rollback 정량·§23.7 duty-cycle·§23.8 PagedAttn/radix/continuous-batching 대조 |
| ch24 | 9 | ~14.8 | §24.2.5 proposal J, §24.3.1 near-memory 3-stage·isolation, §24.3.2 migration 스케줄 |
| ch25 | 3 | ~8.4 | 5-어젠다(준 것→열린 질문→접근·falsifier), 학술 기여 3좌표 |

**Part III 합계 ≈ 83–97 pp**(char/2100–1800 근사) — 예산 52 pp를 **크게 초과**하며, 시작점 ~37 pp 대비
실질 심화가 확인된다. 각 장이 dossier 목표를 개별적으로 충족(초과)한다. 심화 절 표본을 정독한 결과 **패딩은
발견되지 않았다** — 추가분은 전부 수식(식 20-1·NS-5 유효 AI·MFU 분해·식 19-2~19-4), 기제(WY/UT 접힘 대
companion scan, 3-stage 데이터패스), 또는 정직 캐비엇(NS 경계의 회색지대, isolation falsifier)이다.
유일하게 상대적으로 lean한 ch18(~6.6pp)은 Part III 오프너 역할이고 arc 상세가 Part II에 있어 적정하다
(blocking 아님).

---

## 후공정 이월(bibliography/P4 — 오프라인 확정 불가)

- **[P3B-OPEN-1]** (선행 P3-OPEN-1 계승) GDN 인용 미확정. ch22 line 57·66 `TODO-VERIFY`: Gated
  DeltaNet arXiv ID·NVIDIA affiliation·fla-maintainer 관계 미확정. ch21 §21.4는 "NVlabs/GatedDeltaNet,
  619★, ICLR 2025"를 단정하는데 ch22가 affiliation을 미확정으로 둔 것과 **표면 긴장**(원저자≠호스팅이라
  모순은 아님). 외부 검증 필요 → 서지 공정 이월.
- **[P3B-OPEN-2]** 오프라인 확정 불가한 외부 인용 토큰의 P4 검증: `[Atlas Table 5]`(53.55/43.70)·
  `[Atlas Table 2]`(고정 1.3B cross-arch)·`[Atlas Table 6]`(760M)의 **table 번호**, GitHub star 수(FLA
  5,325★·NVlabs 619★·lucidrains 1,966★·HOPE 84/76★), issue 번호(#107/#214), MesaNet arXiv:2506.05233,
  hardware-lottery arXiv:2009.06489, FlashAttention arXiv:2205.14135. **본문 내부는 8장에 걸쳐 일관**
  하므로 정직성 결함이 아니라 외부 대조 대상.

---

## 판정 근거 요약

- **연속성**: ch17→18 bridge 맞물림, 8장 체인 완결, 완성형 소유권 인계 정확 → PASS.
- **척추/FORCED**: pair thesis 전장 일관, memory-centric 두 지점 규율, FORCED 5곳 능동 배제, 신규
  near-memory/migration 절도 두 지점 파생, **새 crossing 0** → PASS.
- **정직성 계약**: 신규 절대치 전부 하한/directional/host-shape 태그, **승격 사례 0** → PASS.
- **수치·표기**: L_layer 8장 통일(잔여 0), 헤드라인 20+·E4 통계 전부 합치. 결함 2건(LOW: matrix 라벨
  충돌, exp-a→exp-f 포인터) → PASS w/ minor fix.
- **E4 정직성**: tokenizer 이질성 §19.3 수식 명문화 + cross-paper ordinal-only + A3 sharply-scoped
  정직 판정. cross-paper ppl 오비교 0 → PASS.
- **예산 깊이**: 8장 전부 목표 충족·초과, 합계 ≈83–97pp ≫ 52pp, 패딩 0 → PASS.

**수정 우선순위**: [P3B-LOW-1](matrix 라벨)·[P3B-LOW-2](exp-f 포인터)는 즉시 반영 가능한 cosmetic;
OPEN-1/2는 서지/P4 이월. 어느 것도 서사·척추·정직성 계약을 위협하지 않는다.
