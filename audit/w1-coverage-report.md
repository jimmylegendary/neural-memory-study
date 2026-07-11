# W1 감사 보고 — 커버리지·충실성 축

- 감사 대상: `study-kr/part1/ch01–ch11`, `study-kr/part2/ch12–ch17` (17개 장 전량 정독)
- 기준 문서: `notes/prereq-curriculum.md` (B0–B9 모듈 명세), `notes/<arxiv-id>.json` (6편), `style/STYLE-NOTATION.md` §4.3 (분량 가이드), §4.1–4.2 (장 템플릿)
- 감사일: 2026-07-12 / 축: (a) Part I 모듈 명세 충족 (b) Part II 노트 반영도 (c) TODO-VERIFY 인벤토리 (d) 분량

## 0. 총평

**커버리지는 전반적으로 매우 높다.** Part I 11개 장은 커리큘럼 B0–B9의 콘텐츠 항목을 사실상 전부 충족하며(누락은 참고문헌 2건·그림 전무 수준), Part II 6개 장은 notes JSON의 core_mechanism / outer_loop_training / inner_loop_test_time / systems_implications를 세부 수치·표기 슬립 교정까지 충실히 반영한다. 특히 Part II 전 장의 "outer vs inner loop" 절(§N.4)은 형식적 존재가 아니라 표+산문의 실질 절로 작성되어 있다.

주요 결함은 콘텐츠 누락이 아니라 **(i) 분량 초과**(ch01 약 2.4×, ch10 약 1.8×, ch16 약 1.5×; 책 전체 목표 대비 +15–27%), **(ii) 미해소 TODO-VERIFY 37건**(이 중 12건은 본문이 기억 기반 사실 주장을 싣고 있어 다음 라운드 해소 필수), **(iii) 그림 0장**(커리큘럼이 명시 요구한 diagram 다수가 표/산문으로 대체됨)이다.

---

## 1. (a) Part I — prereq-curriculum 모듈별 커버리지

판정 방법: 각 모듈의 "Content" 불릿, "Canonical refs", "Systems bridge" 요구를 장 본문과 1:1 대조.

| 장 | 모듈 | 콘텐츠 충족 | 누락/차이 | canonical refs | systems bridge |
|---|---|---|---|---|---|
| ch01 | B0 | **완전** — 두 세계, 사전(표 1-1, 동일/유비/차이 3등급), inner/outer 비형식 도입, 6편 지도(표 1-2), (M) 선행 제시 | 없음. 오히려 B0 범위 초과(worked example·back-of-envelope는 명세 밖 추가) | Sun 2024 §1 ✓, Titans Eq.6–7 ✓ | ✓ (§1.7) |
| ch02 | B1 | **완전** — backprop/VJP/2×규칙/activation·recomputation, loss surface 2문단, SGD·momentum(선형 recurrence 명시)·WD(forget gate 명시)·AdamW·AdaGrad·Shampoo·Muon(NS 비용=GEMM), batch축 vs sequence축 볼드 경고 | **Goodfellow ch.6–8 미인용**(canonical ref 목록 항목) | 그 외 전부 ✓ (Rumelhart, Robbins-Monro, Polyak, Sutskever, Kingma&Ba, Loshchilov, Duchi, Gupta, Jordan blog, Liu 2502.16982, Li 2018) | ✓ (§2.8: step 비용회계, register file, dW=δxᵀ 동일성) |
| ch03 | B2 | **완전** — protocol(decode 대응표), OGD·regret·sublinear 한계, FTL 실패 손계산, FTRL→OGD worked example, mirror descent/Bregman(Memora 예고), loss geometry 표(ℓp/Huber) | **Orabona(arXiv:1912.13213) 미인용**(canonical ref; "freshest treatment"로 지정된 항목) | Zinkevich ✓, Shalev-Shwartz ✓, Hazan ✓, McMahan ✓ | ✓ (regret vs recall@position, 학습된 eviction) |
| ch04 | B3 | **완전** — bilevel/hypergradient/unrolling vs implicit(1문단)·truncation, Schmidhuber 3부작·Andrychowicz·MAML(→W_init), "skeleton key" 문장 명시, ICL-as-GD(von Oswald×2·Akyürek·Mesa-layer) | 없음 | 전부 ✓ (Franceschi, Hospedales 포함) | ✓ (정적 graph=컴파일 가능, truncation↔chunk 크기) |
| ch05 | B4 | **완전** — k→v 원시, correlation memory/crosstalk/capacity O(d), Hopfield 0.14d, Krotov→Ramsauer 사슬(Atlas φ_p/φ* 발판 명시), delta rule=1-step GD(두 커리큘럼 합류 명시), Bietti attention=AM | 없음 | 전부 ✓ (Kohonen, Anderson, Hopfield, Krotov, Ramsauer, Widrow-Hoff, Bietti) | ✓ (§5.8: 유효 cache 크기, byte 등가≠capacity 등가) |
| ch06 | B5 | **완전** — kernel trick, FWP 동치(1992→2021 timeline **표**), RetNet/GLA/일반형, DeltaNet(Householder·WY 예고), Gated DeltaNet, Longhorn 유도(½p 상당 ✓), RWKV-7(벡터 gate·분리 key·TC⁰ 초과 주장), expressivity 사이드바(Grazzi·DeltaProduct·Merrill) | 콘텐츠 누락 없음. 단 (i) **timeline "figure"가 표로 대체**(커리큘럼: "give it a proper timeline figure"), (ii) 분량이 목표 10pp 대비 ~15% 미달 — B5는 "audience 지렛대 최대" 모듈이라 여유가 오히려 있어야 할 장 | 전부 ✓ (11종) | ✓ (§6.9: crossover L*, GQA 보정, decode=traffic 상한 논증) |
| ch07 | B5b | **완전** — ZOH 2줄, S4/HiPPO 1문단, S5/scan(Blelloch·Martin&Cundy), Mamba selectivity(conv 모드 소멸·recomputation), Mamba-2 SSD(semiseparable·3경로 손계산)·"exact vs semantic 경계" 명시, Titans gate 계보 접속 | 없음 (SSD "1–2 figures" → 표·손계산으로 대체) | 전부 ✓ | ✓ (§7.6: scan vs GEMM, op-mix 논증, 표 7-3 exact 열) |
| ch08 | B6 | **완전** — Sun 2020/TENT 전사, TTT layer 3요소, TTT-Linear≡delta rule(+두 극한 정리), dual form 유도(GEMM 4개 명시), outer 분업표, 스케일 증거+Dalal video, 6편 departure 표(=Part II 목차) | 없음 | 전부 ✓ | ✓ (§8.9: decode 내 backward 비용표, autograd 부재→fused kernel) |
| ch09 | B7 | **완전** — 일반 scheme(freeze-at-boundary, linear/nonlinear 분해), 인스턴스 4종(GLA folding, DeltaNet WY/UT 삼각계 유도, TTT dual(semantic 명제 정식화), Titans retention folding+momentum scan+Atlas NS), TNT frontier 골격, 명제 "C=semantic hyperparameter" 소유 선언 | 없음 ("three-regime diagram"은 표 9-1로 대체) | 전부 ✓ (Hua, RetNet, GLA, 2406.06484, 2407.04620, Titans §3.2, Atlas §3.4, TNT, Blelloch, S5, flash-linear-attention) | ✓ (§9.10: AI(C)≈C 유도, 표 9-3 ridge 대비, TNT 존재이유 재유도) — 커리큘럼 "worked roofline" 요구 그대로 |
| ch10 | B8 | **완전** — 완성 Rosetta(확립 장 병기), cheat sheet 10행(요구 8행 + TNT·Atlas), 4K/64K/1M budget 표, prefill/decode 비대칭+serving 4요구, exact vs semantic 재정리+AI(C) 제2식, glossary 55행 | 없음. 분량 초과가 문제(아래 §4) | Roofline·FlashAttention·FA-2·vLLM 전부 ✓ | 장 전체가 bridge ✓ |
| ch11 | B9 | **완전** — CF 정의·기하(crosstalk 시간축 확대판), replay/EWC(식 포함)/isolation 3가족 객체화, CLS(대응표), update-frequency spectrum+online/offline consolidation 구분, KD→GKD(on-policy 강조=Sleep 요구 그대로), REINFORCE 손계산·RLHF/RLVR 모양·ReST^EM·SEAL·LoRA | 없음 (full-path 초과 충족) | 전부 ✓ (McCloskey, French, Kirkpatrick, McClelland, Hinton, Agarwal, Williams, Ouyang) | ✓ (§11.8: LSM-tree 유비, sleep=background job 비용 회계) |

**Part I 누락 항목 총목록** (콘텐츠 수준):
1. ch02: Goodfellow, *Deep Learning* ch.6–8 인용 부재 — minor.
2. ch03: Orabona 2019 (FTRL 최신 교재) 인용 부재 — minor.
3. 전 장 공통: **책 자체 그림 0장** (`study-kr/figures/` 디렉터리 자체가 없음). 커리큘럼이 명시 요구한 시각물: B1 loss-landscape "one figure's worth", B2 "one figure of loss shapes", B5 "proper timeline figure", B5b "1–2 figures", B7 "three-regime diagram(draw once, reuse everywhere)". 전부 표·산문으로 대체됨. 내용 전달은 되나 명세 미충족 — **major** (1차 초안 특성상 critical은 아님; 라운드 계획에 figure pass를 명시적으로 잡을 것).
4. ch06: 분량 미달(아래 §4) + timeline 그림 — major/minor 경계.

그 외 Part I에서 커리큘럼 "Content" 불릿 단위의 실질 누락은 **발견되지 않았다**. 모듈-장 대응(ch01=B0 … ch11=B9), worked micro-example(11/11장), 자가 점검 체크리스트의 Rosetta 마지막 항목, systems bridge 최소 요구 전부 충족.

---

## 2. (b) Part II — notes/<id>.json 반영도

판정 방법: 각 JSON의 core_mechanism / outer_loop_training / inner_loop_test_time / experiments_and_scale / systems_implications 서술 요소를 장 본문과 대조. "outer vs inner 절"은 STYLE §4.2 요구(독립 절 + 표 + "누가 학습하는가" 전수 응답) 기준.

| 장 | core_mechanism 반영 | outer/inner 절 실질성 | 실험·시스템 반영 | 미반영/부족 |
|---|---|---|---|---|
| ch12 Titans | **충실** — surprise(momentary/past), Eq.8/10/13–14→(M2), 기호 교차(θ↔η, η↔β, α 방향 반전) 표 12-1로 처리, deep memory 논거, persistent memory 3근거, MAC/MAG/MAL(+write filter, block-diagonal mask), App.C 특수사례 회수, Thm 4.1 증명 부재 명기 | **실질** — §12.4 표 12-2 + chunkwise Eq.16–18 유도 + momentum scan + "값은 inference/함수는 pretraining" 구분 + gate head 미명세 공백 명기 | Table 1/2/5, BABILong, Fig.7–9, 보조 도메인, ≤760M/30B 정직성, per-token 비용 산정(JSON의 5–6×P_M 회계 그대로), serving 신질문 4종 | 실질 누락 없음. Table 5=400M 귀속은 TODO-VERIFY로 정직 처리 |
| ch13 Miras | **충실** — Def 3.1/attentional bias, FTRL vs Learning–Retaining(Prop 3.2 + 가정 붕괴 [평가]), 4축, Table 1 재유도(Hebbian/delta/너머), 신규 bias 3종(ℓp·Huber 3형·value-shift), retention 동물원 5종(KL/simplex, elastic net 2형, ℓq, Bregman), Moneta/Yaad/Memora 식 전부, smooth surrogate의 사활성 | **실질** — 표 13-3(두 loop 대비) + hypernetwork·W_init 명시 + chunkwise(식 13-13, lag token, 경계 비선형 처리 3종) + "근사물" 경고 | 1.3B/100B 헤드라인, S-NIAH, p/q sweep·Yaad ablation, 스케일 상한, state 8–16× 회계(JSON 수치와 일치), 원문 내부 스케일 표기 불일치 기록 | 실질 누락 없음 |
| ch14 Atlas | **충실** — capacity 3정리(+증명 뼈대), φ_p 이중 정의 주의, Taylor 계수 a_i, Omega rule/γ gate(admission vs eviction 구분), OmegaNet rank-c, Muon/NS_κ(원문 3중 표기 불일치 경고), DLA/SWLA/DeepTransformers/Dot(unnormalized 한계), Atlas++ | **실질** — 표 14-2(window buffer 포함) + banded mask 병렬화 + momentum 분리 scan + batched NS + backprop-through-NS | Table 2/3/4, BABILong 10M +80%, **의무 caveat 2건**(in-context recall 열세 53.55 vs 43.70, Muon 제거 시 ppl 개선) 충실, learnability micro-study, wall-clock 0건 명기 | 실질 누락 없음. p/sketch 미공개로 state 회계 불능임을 명시(TODO-VERIFY 병기) |
| ch15 TNT | **충실** — 3 challenge, Eq.1–4 정식화, global/local(Eq.5/6/14)+원문 인덱스 슬립 3건 각주 교정, W_init 하중, Q-K projection(Π_t, ABC 족보), two-stage(+abstract/본문 모호성 기록) | **실질** — 표 15-2(7행, 갱신 주기 명시) + "fast weights는 parameter가 아니라 activation" + 훈련 그래프 3분해(파이프 재배관 해설) | Table 1(17.37×)/2/3/4, Fig.2 mismatch 수치 전량, 단순화-Titans 위 검증이라는 의무 caveat, reset의 preemption 이득, 150M/10B 정직성 | 실질 누락 없음 |
| ch16 NL | **충실** — Def 1–4, proximal/FTRL 엔진, LSS·backprop 자기참조성(병렬화 불가 논거), AdaTransformer, transfer 5기제, momentum 6/43 스텝 capacity, Adam 최적성(App B, "존재 논증" [평가]), Muon=내부 level, Delta Momentum/DMGD/DGD(16-3)/GGD, CMS 3변형+(M5)+ad-hoc stacking, M3, self-modifying Titans 3단계(+q 반례), Hope/Hope-Attention | **실질** — §16.4가 "이분법 자체의 심문"으로 시작해 표 16-3(9행) + Eq.90 인덱스 슬립 교정 + "자기참조성을 snapshot으로 절단"이라는 핵심 systems 문장 | Table 1–6 전량(ablation 표 재구성), CTNL, formal language, M3 비용 정직성, retrofit 경로, [평가] 배포 모델 함의 | 실질 누락 없음. Hope momentum 유무 간극은 TODO-VERIFY로 처리 |
| ch17 Sleep | **충실** — CMS 수입(17-1), sleep 스케줄(chunk 경계·다대일), expansion(masked pre-allocation), KS/SKS(GKD 식 17-2), LTI(식 17-3/4), L_KS(17-5), synaptic-pruning reset, Dreaming 5단계(random-expert novelty, g_DR 선별, ReST^EM), 기호 충돌 개명 전수(표 17-1) | **실질이자 백미** — 논문의 이분법 거부를 정면 처리하고 **세 번째 regime**(배포 중 offline 훈련 job)을 좌표에 추가; 표 17-3 "누가 정하는가"로 hand-set knob 전수 노출 + "라인 최초의 알고리즘적 wrapper" 의무 caveat | 8개 실험군 전량(수학 표 avg@16 전수, SQuAD, ARC 80%, App B.5 효율), graft≠co-training 정직성, OPSD 지형, 라인 결산 [평가] | 실질 누락 없음 |

**결론 (b)**: 6개 장 모두 노트의 요구 사항을 충족하며, outer vs inner 절은 전 장에서 실질적(표+산문+"이 gate는 누가 학습하는가" 전수 응답)이다. 반영 누락으로 분류할 항목은 발견하지 못했다. 다만 반영 밀도가 높은 만큼 ch13/15/16/17은 분량 초과로 이어졌다(§4).

---

## 3. (c) TODO-VERIFY 인벤토리 — 총 37건

### 3.1 파일별 목록

| 파일 | 행 | 내용 요지 | 위험도* |
|---|---|---|---|
| part1/ch01 | 19 | Sun 2024 슬로건 원문 wording | 낮음(의역 처리됨) |
| part1/ch01 | 142 | Titans App.C 특수사례 목록에 RWKV-7 포함 여부 | **높음** — 본문(§12.3.9 포함)이 RWKV-7 포함으로 단정 |
| part1/ch04 | 129 | von Oswald 2212.07677 경험 주장 범위 | 중간 |
| part1/ch05 | 74 | Hopfield 0.138d 출처(Amit et al.)·spurious attractor 서술 | 중간 |
| part1/ch05 | 84 | Krotov capacity d^{n-1} 정확한 statement | 중간 |
| part1/ch05 | 90 | Ramsauer exponential capacity·1-step retrieval 조건 | 중간 |
| part1/ch06 | 44 | RetNet 이후 분모 제거+출력 norm 관행 | 중간 |
| part1/ch06 | 117 | DeltaNet 구현 관행(key L2 norm, η_t=sigmoid) | 중간 |
| part1/ch06 | 138 | Longhorn objective의 채널별 η(본문은 스칼라 단순화) | **높음** — 식 (6-5) 유도의 전제 |
| part1/ch06 | 167 | RWKV-7 식 (6-6)의 원문 대응 | **높음** — 카탈로그 기준식 |
| part1/ch06 | 173 | RWKV-7 표현력 정리의 정확한 진술 | 중간 |
| part1/ch07 | 56 | Mamba gate 환원 = Theorem 1 위치 | 낮음 |
| part1/ch07 | 114 | Mamba-2 state 차원 N, "8× state" 출처 | 낮음 |
| part1/ch08 | 42 | TTT learned inner lr 함수형 | **높음**(기억 기반) |
| part1/ch08 | 47 | TTT-MLP f 구조(LN/residual/hidden 배수) | **높음**(기억 기반) |
| part1/ch08 | 65 | 두 동치 정리 번호·전제(W₀=0 등) | **높음** — 본문이 "정리로 제시"라 단정 |
| part1/ch08 | 93 | default inner batch b=16 | 중간(기억 기반, 수치 미기재로 완화) |
| part1/ch08 | 118 | 비교 스케일 125M–1.3B, "Mamba 16k 정체" 그림 번호 | **높음**(기억 기반 주장) |
| part1/ch08 | 123 | Dalal backbone(CogVideo-X 5B) 셋업 | 중간 |
| part1/ch09 | 73 | GLA secondary chunking·log-space decay | 낮음 |
| part1/ch09 | 108 | (9-5) vs UT-transform 기호 대응, UT 원출처 | 중간 |
| part1/ch09 | 120 | TTT-MLP dual form 부록 위치 | 낮음 |
| part1/ch09 | 176 | Miras chunkwise 훈련 서술의 원문 절 번호 | 낮음 |
| part1/ch10 | 75 | Atlas window c 실측값·decode incremental 여부 | 중간 |
| part1/ch10 | 140 | H100 bf16 peak/HBM3 bandwidth 수치 | 낮음(datasheet 확인) |
| part1/ch11 | 147 | ReST^EM arXiv ID(2312.06585?) | 낮음 |
| part1/ch11 | 148 | SEAL arXiv ID(2506.10943?)·첫 인용 위치 | 낮음 |
| part2/ch12 | 139 | MAC N_l = C 여부 | 중간 |
| part2/ch12 | 141 | Eq.25 read의 query 재적용 여부 | 중간 |
| part2/ch12 | 297 | Table 5 = 400M 귀속(4행 수치 일치 확인됨) | 낮음 |
| part2/ch13 | 332 | Miras outer optimizer(AdamW 여부)·batch·스케줄 | **높음** — 표 13-3이 "표준 pre-training/peak 1.5e-3"을 기재 |
| part2/ch14 | 301 | 구현 p·sketch·head 분할 부재 최종 확인 | 중간 |
| part2/ch15 | 258 | TNT Table 3 ppl 열의 corpus 기준 | 낮음(추정 명기됨) |
| part2/ch16 | 297 | NL Eq.88 decay 항 적용 대상(deep memory) | **높음** — 재구현 관건 |
| part2/ch16 | 394 | Hope inner rule의 momentum 포함 여부 | **높음** — ablation 해석과 직결 |
| part2/ch17 | 9 | NL 자체 결산 (2)(3)(4) 귀속의 원문 위치 | 중간 |
| part2/ch17 | 202 | Sleep의 Hope가 NL 레시피 checkpoint인지 | **높음** — §17.4 regime 1 서술의 전제 |

\* 위험도 "높음" = 본문이 해당 사실을 (기억 또는 추론 기반으로) **단정 서술**하고 있어, 틀리면 본문 수정이 필요한 항목. 12건.

### 3.2 해소 방법 제안 (배치 처리 4묶음)

1. **로컬 papers/*.txt grep 배치** (11건, 최저 비용): ch01:142, ch09:176, ch10:75, ch12:139, ch14:301, ch15:258, ch17:9, ch17:202, ch11:147, ch11:148, ch16:394(§9.6 텍스트 확인분). 각 TODO 주석에 이미 검색어가 적혀 있으므로 스크립트 한 번으로 일괄 확인 가능.
2. **외부 arXiv 원문 대조 배치** (외부 논문 18건): 2407.04620(ch08 5건+ch09:120), 2503.14456(ch06 2건), 2407.14207(ch06:138), 2406.06484+2102.11174(ch06:117, ch09:108), 2312.06635(ch06:44, ch09:73), 2312.00752(ch07:56), 2405.21060(ch07:114), 2212.07677(ch04:129), 2008.02217·1606.01164·Amit1985(ch05 3건), 2504.05298(ch08:123). 논문당 1회 fetch로 그룹 해소 — 특히 **2407.04620 1편이 6건을 해소**하므로 최우선.
3. **PDF 직접 확인** (5건): ch12:141/297(Titans PDF p.9·Table 5 캡션), ch13:332(Miras PDF App.C), ch16:297(NL PDF §8.1–8.2), ch14:301(Atlas Fig.3 라벨).
4. **잡건** (1건): ch10:140 — NVIDIA H100 datasheet 수치 확인(웹 1회).

권고: 위험도 "높음" 12건을 W2 라운드 시작 전에 해소하고, 해소 시 TODO 주석을 "확인 완료(출처)" 주석 또는 본문 병기로 치환하는 규약을 정할 것. 현재 37건 중 어떤 것도 해소 이력이 없다.

---

## 4. (d) 분량 — 장별 실제 vs 목표

환산 기준: STYLE §4.3 "1p ≈ 500–550 한국어 단어 + 수식", 실측은 `wc -w`(LaTeX 토큰 포함이므로 실제 페이지는 약간 부풀려질 수 있음 — ±10% 오차 감안).

| 장 | 목표 pp | 목표 단어(500–550/p) | 실측 단어 | 환산 pp | 판정 |
|---|---|---|---|---|---|
| ch01 | 3 | 1,500–1,650 | 3,936 | ~7.5 | **초과 239–262%** — 최대 위반 |
| ch02 | 10 | 5,000–5,500 | 5,110 | ~9.8 | 적정 |
| ch03 | 6 | 3,000–3,300 | 4,485 | ~8.5 | 초과 ~140% |
| ch04 | 6 | 3,000–3,300 | 3,368 | ~6.4 | 적정 |
| ch05 | 8 | 4,000–4,400 | 4,543 | ~8.7 | 적정(+8%) |
| ch06 | 10 | 5,000–5,500 | 4,253 | ~8.1 | **미달 ~81%** — 유일한 미달 장 |
| ch07 | 6 | 3,000–3,300 | 3,758 | ~7.2 | 소폭 초과(+19%) |
| ch08 | 7 | 3,500–3,850 | 3,669 | ~7.0 | 적정 |
| ch09 | 10 | 5,000–5,500 | 4,499 | ~8.6 | 소폭 미달(−12%) |
| ch10 | 5 | 2,500–2,750 | 4,734 | ~9.0 | **초과 172–189%** |
| ch11 | 4–7 | 2,000–3,850 | 4,269 | ~8.1 | 상한 초과(+11%) |
| ch12 | 10 | 5,000–5,500 | 5,430 | ~10.3 | 적정 |
| ch13 | 8 | 4,000–4,400 | 5,537 | ~10.5 | 초과 ~130% |
| ch14 | 10 | 5,000–5,500 | 5,505 | ~10.5 | 적정 |
| ch15 | 7 | 3,500–3,850 | 5,172 | ~9.8 | 초과 ~140% |
| ch16 | 11 | 5,500–6,050 | 8,621 | ~16.4 | **초과 143–157%** |
| ch17 | 9 | 4,500–4,950 | 6,013 | ~11.5 | 초과 ~125% |
| **합계** | **130–133** | 65,000–73,150 | **82,902** | **~158** | **+15–27%** |

분석:
- 초과의 성격이 둘로 갈린다. ch13/15/16/17의 초과는 노트 반영 밀도(원문 표기 슬립 교정, 의무 caveat)의 대가로 **내용 손실 없이 압축이 어려운 초과**다 — 목표 pp 자체의 상향 재배정(예: ch16을 11→14pp)이 현실적일 수 있다. 반면 **ch01(2.4×)과 ch10(1.8×)은 명백한 위반**이다: ch01은 B0 명세(3–4pp, "Before any math") 대비 worked example·수치 계산까지 실었고, ch10은 Rosetta 표 전문 재수록(표 10-1, ch01과 중복)과 55행 glossary가 부피를 만든다.
- 감축 후보(내용 보존): ch01 §1.6–1.7을 축약하거나 ch02/ch10으로 이관; ch10 표 10-1은 "1장 표에 확립-장 열만 추가" 참조로 대체; ch03 §3.8 표 2개 중 1개 축약; ch16 §16.3.5–16.3.6의 유도 일부를 부록화.
- ch06은 유일한 미달 장이며 B5는 "10–12pp, 독자 지렛대 최대" 모듈 — RWKV-7·expressivity 절(각 TODO-VERIFY 해소와 함께)을 보강하면 자연히 채워진다.

---

## 5. 발견 사항 — 심각도별

### Critical (다음 라운드 필수 수정)
1. **[분량] ch01 2.4× 초과, ch10 1.8× 초과** — 두 장 합계로만 약 9pp 초과. 책 전체 +15–27% 초과의 최대 기여자이며, 방치 시 Part III 지면을 잠식한다. (ch16의 1.5× 초과는 목표 재배정 또는 부록화 중 하나를 결정할 것.)
2. **[TODO-VERIFY] 위험도 '높음' 12건 미해소** — 본문이 기억/추론 기반 사실을 단정 서술 중: ch01:142(App.C에 RWKV-7 포함 단정 — ch12 §12.3.9와 연동), ch06:138/167, ch08:42/47/65/118, ch13:332, ch16:297/394, ch17:202. §3.2의 배치 계획(특히 arXiv:2407.04620 1편 fetch로 6건 해소)대로 W2 착수 전 처리.

### Major
3. **[커버리지] 책 그림 0장** — `study-kr/figures/` 부재. 커리큘럼 명시 요구 시각물 5건+(B1 loss landscape, B2 loss shapes, B5 FWP timeline, B5b SSD 1–2장, B7 three-regime diagram)이 전부 표/산문 대체. 표들이 내용은 전달하나 "draw once, reuse everywhere" 류의 재사용 설계는 그림 없이는 성립하지 않음. figure pass를 별도 라운드로 계획할 것.
4. **[분량] ch03(1.4×)·ch13(1.3×)·ch15(1.4×)·ch17(1.25×) 초과, ch06(0.81×) 미달** — ch06 미달은 B5의 중요도(“audience 지렛대 최대”) 대비 역방향.
5. **[프로세스] TODO-VERIFY 총 37건, 해소 이력 0건** — 위험도 중·낮음 25건 포함. 다음 라운드에 해소 규약(확인 완료 주석 형식)과 담당 배치를 확정할 것.

### Minor
6. **[refs] ch02 Goodfellow ch.6–8, ch03 Orabona 2019 미인용** — 커리큘럼 canonical refs 목록 항목.
7. **[슬러그] 파일명 4건이 STYLE §5.1 고정 슬러그와 불일치** (ch01-orientation-rosetta, ch03-online-learning-regret, ch04-meta-learning-bilevel, ch09-chunkwise-parallel-training) — 각 파일 상단 STYLE-ISSUE 주석으로 자기 신고되어 있음. P2 병합 시 일괄 통일 필요(§5.1을 지시서 경로 쪽으로 개정하는 것도 대안).
8. **[중복] ch10 표 10-1이 ch01 표 1-1의 전문 재수록** — 의도된 "완성형 재수록"일 수 있으나 분량 압박 하에서는 차분(확립-장 열)만 싣는 방식 검토.

### 긍정 확인 (수정 불요)
- Part I 11개 장 모두 worked micro-example·자가 점검(Rosetta 마지막 항목 포함)·systems bridge·"왜 필요한가" 박스 완비.
- Part II 6개 장 모두 outer vs inner 절이 실질적(표+산문, hand-set/learned 구분까지). 특히 ch17은 논문의 이분법 거부를 정면 처리하며 세 번째 regime을 좌표화 — 명세 이상의 수행.
- 원문 표기 슬립(TNT Eq.5/6/7, NL Eq.90, Atlas 3중 표기, Miras 스케일 표기 불일치)의 교정·기록이 일관 규약("부호 관행 차이(알고리즘 동일)")으로 처리됨.
- 정직성 규칙(스케일 상한 1.3B/100B, decode wall-clock 부재, Muon ablation, in-context recall 열세) 전 장 반영.
