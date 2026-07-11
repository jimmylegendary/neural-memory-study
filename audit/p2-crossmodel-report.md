# P2 Cross-model adjudication report

**판정자**: Claude (Opus 4.8), cross-model adjudicator
**대상**: GPT-5.6-sol의 독립 기술 검증 발견 (`audit/crossmodel/ch*.json`)
**기준**: 원전(`papers/*.txt`, `papers/external/*.txt`), `notes/*.json`, `style/STYLE-NOTATION.md` v1.1
**일자**: 2026-07-12

## 0. 요약

- 총 발견 **107건** (17개 장; ch02는 실패). 판정: **valid 105 / invalid 2**.
- 조치: **applied 98 / deferred 7 / invalid(무조치) 2**.
- critical 2건(ch10 §10.5–10.6 intra-chunk FLOPs 누락, ch14 §14.3.1 Theorem 1) 모두 valid → 적용.
- 통일 표기 의도를 오류로 오인한 false positive는 소수(ch12 F4, ch01 F1 부분)로, deferred/무조치 처리.
- 원문 자체의 결함(Atlas Table 1 부호 오탈자, TNT Eq.6 reset 경계, NL Eq.43 Muon objective, Sleep Eq.2 off-by-one 등)은 "원문 귀속 + 정정 명시"로 처리 — §6.2/§6.3 준수.

## 1. 실패한 장

- **ch02 (training-as-a-system)**: `ch02-*.json = []`, `ch02-*.raw.md` **부재**. GPT-5.6-sol 실행 자체가 실패(빈 배열 + 원출력 없음). 재실행 대상. 이 장에 대한 cross-model 검증 없음.

## 2. 장별 판정표

표기: A=applied, D=deferred, I=invalid. severity: cr=critical, M=major, m=minor.

### Part I

| 장 | # | sev | 발견 요지 | 판정·근거 | 조치 |
|---|---|---|---|---|---|
| ch01 | F1 | M | "gradient=(오차)k^T rank-1"을 일반 fast-weight에 적용 | 부분 valid — 그러나 outer-product GEMM 구조는 층별 보편($∂L/∂W_l=δ_l h_{l-1}^\top$)이고 §3 Rosetta는 STYLE 소유. worked-example은 linear로 정확히 한정. δh^T 도입은 ch01( backprop 미도입)에 조급 | **D** |
| ch01 | F2 | M | Θ 동결 불변식이 Sleep에 안 성립 | valid — Sleep은 sleep phase에 slow weights 갱신 | A (예외절+cross-ref) |
| ch01 | F3 | m | ∇_Wℓ을 "실패를 줄이는 방향"이라 오기 | valid — ∇는 증가 방향 | A |
| ch03 | F1 | M | (3-1) post-update vs (3-5)/§3.8 play 관행 불일치 | valid but 재인덱싱은 regret+3식+예제 동시 개편(고위험), (M1)이 (3-1) 고정, worked-example 정확 | **D** |
| ch03 | F2 | M | Miras Eq.5=모든 모델의 OGD로 과일반화 | valid — Eq.5는 "one simple approach"(원문 확인); Newton/non-param/implicit 예외 | A |
| ch03 | F3 | M | O(GD√L)에 projection 없음 | valid — Zinkevich는 projected OGD | A |
| ch03 | F4 | m | ½ vs 무-½ 계수 불일치(Longhorn eff. step) | valid but ch06 Longhorn형과 관행 결합; 장-횡단 결정 필요 | **D** |
| ch03 | F5 | m | Sherman-Morrison RLS는 OGD와 동차수 | 챕터는 naive "과거 전체 재최적화"(Θ(d³))를 서술 → "자릿수 다름" 성립; finding은 다른 구현 겨냥 | **D** |
| ch03 | F6 | m | recall 실패=regret 성분 인과 과장 | valid — comparator도 실패하면 국소 기여 ≤0 | A |
| ch03 | F7 | m | softmax 재정규화를 elementwise로 분류 | valid — reduction 필요 | A |
| ch03 | F8 | m | dual accumulator 상주 2배 단정 | "2배"는 방어 가능한 worst-case; 정밀 회계는 ch13 turf | **D** |
| ch04 | F1 | M | chunk C = truncation 길이 동일시 | valid — 기제·방향 반대(C클수록 나쁨, truncation 길수록 좋음) | A |
| ch04 | F2 | M | inner loop를 serving 전용으로 배치 | valid — inner는 pretraining forward에서도 돎 | A (표 주석) |
| ch04 | F3 | M | TNT reset이 정보 폐기 아님 주장 | valid — reset은 local 정보 폐기; global이 보완 | A |
| ch04 | F4 | m | projection·W_init을 token별 함수로 묶음 | valid — 이들은 고정 파라미터 | A |
| ch05 | F1 | M | softmax "read 오염 없음/exact 무제한" | valid — finite softmax는 혼합 read; 무제한은 저장량. STYLE "capacity 무한"(=압축 안함)은 유지 | A |
| ch05 | F2 | M | cyclic LMS→최소제곱 무조건 수렴 | valid — inconsistent계는 발산; Atlas Prop1은 full-batch GD | A |
| ch05 | F3 | m | "가장 가까운 pattern으로 수렴" | valid — basin attractor, spurious 가능(§5.3 자기모순) | A |
| ch05 | F4 | m | energy sharpness→1-step 보편 인과 | separation 조건이 **직전 문장**에 이미 명시 → 방어 가능 | **D** |
| ch06 | F1 | M | RetNet α="학습된 상수" | valid — 원문 확인: γ=1−2^{−5−arange}, head별 고정·비학습 | A |
| ch06 | F2 | m | Hebbian norm 단조 증가 | valid — 부호 상쇄 가능 | A |
| ch06 | F3 | m | Longhorn η_t 임의 큰 값 | valid — 원문 확인: β_t=Sigmoid∈(0,1); 안정성은 closed-form 성질 | A |
| ch06 | F4 | M | (6-6)이 GDN 정확 포함 | valid — write항 계수 불일치(η_t 누락); transition만 일치 | A |
| ch06 | F5 | m | ξ·ν를 token별 산출로 묶음 | valid — ξ·ν는 token-independent 학습 파라미터 | A |
| ch06 | F6 | m | 반복 제시→최소제곱 무조건 수렴 | valid — consistent 경우 한정 | A |
| ch07 | F1 | M | ZOH bar B≈ΔB를 큰 Δ에 적용 | valid — 안정 a<0에서 −B/a로 포화 | A |
| ch07 | F2 | M | Mamba-1 병목=op mix(≠bandwidth) | valid — 원문 확인: scan은 memory-bandwidth-bound + tensor-core 미사용 공존 | A |
| ch07 | F3 | m | semiseparable=임의 부분블록 low-rank | valid — 하삼각 포함 부분행렬 한정(Def 3.1) | A |
| ch07 | F4 | M | Titans 완전소거→W_init 복귀 | valid — 원문 확인: Miras fn2 "cold start"=M_{t-1}서 surprise 측정; M_t=S_t(≠W_init) | A |
| ch08 | F1 | M | TTT-Linear=gate없는 DeltaNet 문자 동일 | valid — C=1+무 LN/residual core에서만; η_t는 gate | A |
| ch08 | F2 | M | dual form을 근사로 서술 | valid — 원문 확인 "equivalent in output"; 근사는 chunk C | A |
| ch08 | F3 | M | GEMM 구현서 η_τ·base read 누락 | valid — D_η, W_ξQ^T 필요 | A |
| ch08 | F4 | m | "gradient가 lr 튜닝" | valid — η_base·warmup은 사람이 설정 | A |
| ch08 | F5 | M | decode 순서 read→update | valid — 원문 확인(train→predict): update-then-read | A |
| ch08 | F6 | m | gradient 손유도 필수 | valid — autograd 가능(App A.4); 최적화 선택 | A |
| ch08 | F7 | m | prefix 공유 불가 | valid — deterministic snapshot 공유 가능; CoW 필요 | A |
| ch09 | F1 | M | 비선형→exact 재조직 "유일" 단정 | valid — TNT §1 "largely unsolved"+Gonzalez/Lim 인용 | A |
| ch09 | F2 | M | batch C forward+backward 한 번 | valid — 합산 backward는 Σg_t; per-token g_t는 dual/vmap | A |
| ch09 | F3 | m | DeltaNet 물화 O(d³) | valid — rank-1 적용은 O(d²); 병목은 순차·IO | A |
| ch09 | F4 | m | AI(C)식이 DeltaNet까지 포괄 | valid — Gram·triangular solve 추가; GLA형으로 한정 | A |
| ch09 | F5 | m | C=1로 "무너진다" | valid — Fig.2 최소 C=8(36.45); C=1 미측정 | A |
| ch10 | F1 | **cr** | (10-2)에서 intra-chunk O(C²d) 누락→0.75d ceiling 오류 | valid — 원문 확인(2406.06484 §2.1 O(C²d+Cd²)); §10.6 Step4가 항을 인지하고도 "결론 불변"이라 오판; line128 자기모순 | A (식·Step3-4·요약 재계산) |
| ch10 | F2 | M | Mamba-2 state=d²(≠PN) | valid — N은 자유 파라미터; d_k=d_v 규약 caveat 명시 | A |
| ch10 | F3 | M | Titans gate=chunk-상수 | valid — 원문 확인: 실험은 token-dependent | A |
| ch10 | F4 | M | 고정-state decode 예외없이 bandwidth-bound | valid — Atlas/Muon NS는 matrix-matrix→compute-bound 가능 | A |
| ch10 | F5 | M | TNT FLOPs=10P_M+4d²(N=1만) | valid — N개 local 반영 필요 | A |
| ch10 | F6 | M | TNT global 매 token full RMW | valid — global은 C_g마다 write, per-token read | A |
| ch10 | F7 | M | reset이 replay를 L_s로 유계 | valid — global은 미reset, snapshot 필요 | A |
| ch10 | F8 | M | semantic=nonlinear M 통과 시 | valid — TTT-Linear(선형 M)도 anchored→semantic | A |
| ch10 | F9 | m | 17.37×·23.09를 한 설정으로 결합 | valid — 원문 확인: 서로 다른 구성 | A |
| ch10 | F10 | m | Stage2 "~5%"(최고 8.3%) | valid — 원문 확인: 0.46/5.55=8.3% | A |
| ch11 | F1 | m | β=0.9 EMA 6/43개 | valid — 1−0.9^6=46.9%<50%; 7/44 필요(NL 근사) | A |
| ch11 | F2 | M | orthogonal task momentum→CF 단정 | valid — 선형·orthogonal에선 task CF 없음(x_A^TΔΘ=0) | A |
| ch11 | F3 | M | EWC memory=O(TP) 과장 | valid — diagonal 이차합은 O(P)로 결합 | A |
| ch11 | F4 | m | Fisher=∇ŷ 외적 | valid — Fisher는 score 외적; Gaussian 가정 명시 | A |
| ch11 | F5 | M | expert reset이 성장 관리 해결 | valid — 원문 확인: reset은 source만; target 누적 | A |
| ch11 | F6 | M | LoRA backward 무감/adapter 한정 상충 | valid — frozen weight-grad GEMM 생략(감소); activation grad는 전체 관통 | A |

### Part II

| 장 | # | sev | 발견 요지 | 판정·근거 | 조치 |
|---|---|---|---|---|---|
| ch12 | F1 | M | (12-4) 스칼라 vs (12-6) diag 게이트 | valid — 원문 App C는 채널별 diag; 스칼라 특수형 명시 | A |
| ch12 | F2 | m | P∥X 차원 불일치 | valid — P는 행-쌓기 N_p×d | A |
| ch12 | F3 | m | (12-8) 계수 2 부재 | valid — ½ 관행 명시로 정합 | A |
| ch12 | F4 | M | inner chunk b = MAC segment C 동일시 | **STYLE §1.7.1이 명시적으로 "동일시함을 명시"로 규정** — 투명한 통일 결정, 장 오류 아님(false-positive류) | **D** |
| ch12 | F5 | M | MAC read를 segment-end서 전부 → causal leakage | valid — prefix 상태 W_τ서 읽어야(Eq16) | A |
| ch12 | F6 | M | "train/inference 불일치 없다"가 §12.8과 모순 | valid — C semantic → mismatch | A |
| ch12 | F7 | m | 5–6×P_M FLOP(MAC 혼동) | valid — ~4–5P_M MAC=8–10P_M FLOP(ch10 12P_M과 정합) | A |
| ch12 | F8 | M | TTT/GDN 격차=성분 값 직접 등치 | valid — 비통제 비교; Table 5 ablation만 인과 | A |
| ch12 | F9 | m | LMM state=GDN의 2배 | valid — ~2L_M배(2-layer면 4배) | A |
| ch13 | F1 | M | 가변 η_t에서 OGD≡FTRL 동치 | valid — 상수 η만 정확 | A |
| ch13 | F2 | m | 발전이 (i)(iii)축만 움직였다 | valid — Hebbian→delta는 bias축; Longhorn 등 algorithm축 | A |
| ch13 | F3 | M | p=1→memory ±1 두 값 | valid — residual만 부호; W·출력은 실수 | A |
| ch13 | F4 | M | Moneta q=4가 변형4(1<q≤2) 범위 밖 | valid — 경험적 확장 명시 | A |
| ch13 | F5 | m | \|x\|가 gradient 죽/폭주 | valid — \|·\| 도함수 ±1 유계; Sign이 0-grad | A |
| ch13 | F6 | m | attention "완벽한 recall" | valid — kernel weighting 의존(=ch05 F1) | A |
| ch13 | F7 | M | 2-layer MLP MAC/FLOP 혼동 | valid — forward 8d² MAC; ~28–32d² MAC 재계산 | A |
| ch13 | F8 | m | Memora softmax=8d² 전역 | valid — 원문 확인 "normalized per slice" | A |
| ch14 | F1 | **cr** | Theorem 1을 검증 정리로 유도 | valid — 선형독립 key 정의상 m≤d_k이나 하한 O(d_k d_v)>d_k 모순; m≤rank(A) 증명 결함. 원문 결함으로 명시([평가] 추가) | A |
| ch14 | F2 | m | capacity가 P에 무조건 sub-linear | valid — d_v 고정 시 선형 | A |
| ch14 | F3 | m | η<2/λ_max·min-norm | valid — 무-½ loss는 η<1/λ_max, W_0=0 | A |
| ch14 | F4 | M | global 최적화=전 KV cache 필요 | valid — linear LS는 Sherman-Morrison(§14.3.2 자기모순) | A |
| ch14 | F5 | m | gate→prefix만큼 파라미터 증가 | valid — 공유 producer 파라미터 고정 | A |
| ch14 | F6 | M | Table1/Eq32-33/App D.4 동일 | valid — Table1은 −∇/−NS=ascent 부호 오탈자; STYLE도 Eq32-33/D.4만 등가 | A |
| ch14 | F7 | M | DLA ℓ=⟨M,v⟩ GD→+vk^T | valid — 원문 확인 부호 모순; ℓ=−⟨·⟩·φ=id 필요 | A |
| ch14 | F8 | M | DeepTransformers=Transformer strict gen | valid — 원문 확인 unnormalized; §14.3.5말미와 모순 | A |
| ch14 | F9 | m | gate-producer x_t서 산출 단정 | valid — 원문 미공개(특히 γ_{t,i} 입력) | A |
| ch14 | F10 | M | window buffer O(c(d_k+d_v)) | valid — lifted φ(k) 차원 D=C(d_k+p,p) | A |
| ch14 | F11 | m | Dot 전설정 96.8–100 | valid — 원문 확인 S-NIAH-W 16K=93.2 | A |
| ch15 | F1 | M | (15-4)/(15-5) 1-based reset 경계 오류 | valid — 원문 확인 "reset at beginning of segment"+"carry re-init at boundaries"; t≡0은 마지막 token read 손실 | A (shard-anchor 재정식) |
| ch15 | F2 | M | chunk-1 serving 실증 과장 | valid — Fig2 최소 C=8; {1}=23.99≠전역최적({2,4,8,16}=23.09) | A |
| ch15 | F3 | m | Stage2 "~5%" | valid — 최고 4-local 8.3% | A |
| ch15 | F4 | m | reset=α_t∈{0,1} gate | valid — α=0은 0 소거≠W_init; affine 필요 | A |
| ch15 | F5 | m | reset "유일한" 병렬화 | valid — TNT §1 "largely unsolved" | A |
| ch16 | F1 | M | linear attn projection에 backprop 없음 | valid — ∂L/∂W_K는 recurrence 관통(stop-grad 아닌 한) | A |
| ch16 | F2 | M | Muon Eq43 gradient에 O−g 항 | valid — ‖O^TO−I‖² grad엔 O−g 없음; proximity항 필요 | A |
| ch16 | F3 | M | (16-3) 계수/Sherman-Morrison/η' 불일치 | valid — η'=η/(1+ηλ²), 역행렬 (xx^T+η^{-1}I) | A |
| ch16 | F4 | M | GGD argmin에 W 부재 | valid — L̃가 M(x_t;W) 통해 W 의존해야 | A |
| ch16 | F5 | M | 가산 recurrence "Adam 그대로" | valid — additive(AdaGrad계), bias-correction 없음; "근사 회수"로 | A |
| ch16 | F6 | M | (16-4) deep MLP에 우측곱 type 미정의 | valid — 각주2+VERIFIED가 이미 flag; inline 포인터 추가 | A |
| ch16 | F7 | M | chunkwise 1/C를 decode에 적용 | valid — 미래 chunk 미리 batch 불가; train/prefill로 한정 | A |
| ch16 | F8 | M | q 적응 여부 불일치(표16-3=5 vs §16.4.2=6) | valid — 내부 모순; q 미확정(5–6) 명시 | A |
| ch16 | F9 | m | β=0.9 6/43개 | valid — 7/44(=ch11 F1) | A |
| ch16 | F10 | m | retrieval gap 53.6/43.7 | **invalid — 현재 본문(line446)이 이미 53.55/43.70** | **I** |
| ch17 | F1 | M | 생물 CF해법=neuroplasticity(≠replay) | valid — 원문 확인 "long-term knowledge with replay"; 결합임 | A |
| ch17 | F2 | M | 1k→5k→10k "frequency 사다리" | valid — 원문 확인 frequency="updates per unit time"(역수); C/period 사다리 | A |
| ch17 | F3 | m | Eq2 하한 +1을 "알고리즘 동일" | valid — 한 항 차이; off-by-one 교정으로 | A |
| ch17 | F4 | M | consolidated 지식 "침식 못한다" | valid — 원문 확인: 전-freeze는 KS만; dreaming LoRA SFT는 갱신; "더 robust" 가설 | A |
| ch17 | F5 | M | sleep 데이터 전부 자기생성 | valid — 원문 확인 Dreaming은 외부 task (C,τ) 조건 | A |
| ch17 | F6 | M | Sleep sequence layer=고정 state | valid — CMS sequence model이 attention이면 KV cache O(L) 잔존; GPT 하락→KV 인과 유보 | A |
| ch17 | F7 | m | retrieval avg 53.6/43.7 | **invalid — 현재 본문(line317)이 이미 53.55/43.70** | **I** |

## 3. Deferred 목록 (valid이나 무조치 — 사유)

1. **ch01 F1** (M): outer-product/GEMM 프레임은 STYLE §3 Rosetta 소유이고 층별 보편($∂L/∂W_l=δ_l h_{l-1}^\top$). worked-example은 linear로 정확히 한정됨. deep-memory δh^T는 backprop을 배우는 ch02 이전이라 조급하며 STYLE과 desync. → 표기 pass에서 재검토.
2. **ch03 F1** (M): (3-1) post-update 관행은 (M1) 고정. 재인덱싱은 regret 정의+(3-3)(3-4)(3-5)+worked-example 동시 개편이라 신규 오류 위험 큼. 하중 예제는 내부 정합. → 전용 표기 pass.
3. **ch03 F4** (m): ½ vs 무-½ 계수는 ch06 Longhorn형과 결합. 장-횡단 관행 결정 필요.
4. **ch03 F5** (m): 챕터가 서술한 naive FTL(과거 전체 재최적화, Θ(d³))에는 "자릿수 다름"이 성립. finding은 recursive RLS(다른 구현)를 겨냥.
5. **ch03 F8** (m): "상주 2배"는 방어 가능한 worst-case. 정밀 resident/checkpoint 회계는 ch13 systems 소유.
6. **ch05 F4** (m): separation 조건이 직전 문장에 이미 명시됨 → 요약문은 문맥상 방어 가능.
7. **ch12 F4** (M): STYLE §1.7.1이 "segment 크기 = chunk 크기로 동일시함을 명시"로 규정한 투명한 통일 결정. finding은 STYLE 결정을 문제 삼는 것이지 장 오류가 아님(통일 표기 의도의 false-positive류).

## 4. 두 모델 간 불일치 패턴 — 교훈

1. **MAC↔FLOP 혼동이 반복 신뢰 오류원.** ch12 F7, ch13 F7, ch10 F1(간접)은 전부 "1 MAC=2 FLOP"를 섞은 데서 온다. GPT-5.6-sol이 이를 잘 잡았다. **교훈**: 비용식은 단위(MAC/FLOP)를 식마다 못박고, 장 간 수치(ch10 표 10-2 vs ch12/13)를 상호 검산해야 한다.

2. **chunk-start staleness의 semantic 성격을 두 모델 다 알지만 적용 경계에서 미끄러진다.** ch10 F8(선형 M도 anchored면 semantic), ch16 F7(chunkwise 1/C를 decode에 오적용), ch09 F5·ch15 F2(C=1 미측정을 "붕괴"로). **교훈**: "$C$=semantic"은 (a) 정의 방식(anchored)이 기준이지 M의 선형성이 아니고, (b) 훈련·prefill의 병렬화 산수를 decode에 그대로 옮기면 안 되며, (c) Fig.2가 측정한 범위(C≥8)를 넘는 외삽은 명시 대상.

3. **원문 자체의 결함을 "통일 표기가 정정을 겸한다"며 과잉 무마하는 경향.** Atlas Table1 부호(ch14 F6), TNT Eq6 reset(ch15 F1), NL Eq43 Muon(ch16 F2), Sleep Eq2 off-by-one(ch17 F3), Titans MAC causal(ch12 F5)은 모두 원문 결함인데 초안은 "관행 차이(알고리즘 동일)"로 눌러 버렸다. GPT-5.6-sol이 "한 항/한 부호 차이는 동일이 아니다"를 정확히 짚었다. **교훈**: §6.2대로 원문 결함은 "원문에 귀속 + 정정 명시"로 처리하고, "알고리즘 동일"이라는 봉인은 실제로 대수적으로 같은 경우로만 제한한다.

4. **인과 과장·비통제 비교.** ch12 F8(모델 간 격차=성분 값), ch11 F2(momentum 망각=task CF), ch17 F6(GPT 하락=KV 고갈), ch03 F6(recall 실패=regret). **교훈**: 상관/일관을 인과로 승격하지 말고, 통제 ablation이 뒷받침하는 범위로 주장을 한정한다.

5. **"유일한/모두/무제한" 절대 단정이 반복 위험.** ch09 F1·ch15 F5(유일한 병렬화), ch03 F2(모든 모델=OGD), ch05 F1·ch13 F6(무제한 capacity/완벽 recall), ch14 F1(검증된 정리). **교훈**: 절대량화어는 원문 근거를 재확인하고(대개 "largely unsolved"·"one approach"·"non-parametric 해"로 하향), 이론 주장은 원문의 증명 상태(Titans Thm4.1 무증명, Atlas Thm1 정의 모순)를 병기한다.

6. **cross-model 자기교정 신호.** ch16 F10·ch17 F7은 GPT가 지적한 수치(53.6/43.7)가 **이미 본문에서 교정된**(53.55/43.70) 사례 — GPT가 stale/misread 스냅샷을 본 것으로, 판정자가 원전+현재 본문 대조로 걸러냈다. **교훈**: cross-model 발견은 현재 본문과 원전 양쪽 대조 없이 자동 적용하면 안 된다(false-positive 2건이 이 경로).

7. **표기 통일 의도 vs 오류의 경계.** ch12 F4(b=C 동일시)는 STYLE이 명시적으로 승인한 통일이라 무조치했고, ch01 F1(outer-product 프레임)도 STYLE-mandated pedagogy라 deferred. **교훈**: cross-model 검증자에게 STYLE-NOTATION 전문을 주더라도, "통일 결정"과 "장 오류"의 구분은 판정자의 최종 몫으로 남는다 — GPT는 수학적 엄밀성을 우선해 통일 단순화를 종종 오류로 본다.

## 5. 무결성 노트

- 모든 수정은 v1.1 통일 표기($W$/$\Theta$, $\eta_t$/$\beta_t$/$\alpha_t$, $\ell$/$\mathcal{L}$, 열벡터, 조사 결합)·용어 규약을 준수하도록 작성했으며 새 위반을 도입하지 않았다.
- 원전 대조로 사실 확인한 항목(RetNet γ, Longhorn sigmoid, Mamba bandwidth/matmul, Miras fn2·per-slice·one-approach, TTT equivalent-in-output·train-then-predict, Titans function-of-chunks, TNT reset-at-beginning·carry-reinit·수치, Atlas 93.2·unnormalized·linearly-independent, Sleep replay·frequency-def·sampled-task)은 §2 표의 "원문 확인"으로 표기.
