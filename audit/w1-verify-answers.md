# W1 검증 이력 장부 (w1-verify-answers.md)

> **역할.** fixplan D9 / STYLE-NOTATION.md v1.1 §6.3(해소 규약 1)이 지정한 **TODO-VERIFY 해소 이력의 SoT(감사추적 요약 장부)**. 각 TODO의 최종 판정과 근거를 (파일:라인 | 판정 | 원전 인용 | 본문 반영 방식) 형식으로 한 줄에 고정한다.
> **원본.** 판정 근거의 전문(대상 주장·원전 발췌·판단 논거·fixer 조치 지시)은 `audit/w1-todo-resolutions.md`에 있고 그 문서가 원본으로 유지된다. 이 장부는 그 요약이며, 세부가 필요하면 원본의 동일 항목을 본다.

- 검증일: 2026-07-12
- 대상: coverage 리포트 §3 인벤토리 + grep 전수 — **총 37건 전량 판정 완료**
- 판정 분포: **CONFIRMED 30 / CORRECTED 6 / UNRESOLVABLE 1**
- 원전 확보: 로컬 6편(papers/*.txt·pdf) + 외부 13편 신규(papers/external/*.{pdf,txt}) + NVIDIA H100 datasheet(웹) + Amit 1985 서지(웹). 상세 내역은 원본 부록 A.
- 표기 규약: "주석→VERIFIED 치환" = 해당 `<!-- TODO-VERIFY: … -->`를 `<!-- VERIFIED(2026-07-12): … -->` 한 줄로 대체(§6.3 해소 규약 1). **★** = coverage 리포트 위험도 '높음' 관련.

## 판정 이력표

| # | 파일:라인 | 판정 | 원전 인용 | 본문 반영 방식 |
|---|---|---|---|---|
| 1 | part1/ch01:19 | CONFIRMED | 2407.04620 abstract·§1 ("make the hidden state a machine learning model itself, and the update rule a step of self-supervised learning") | 주석→VERIFIED 치환. 의역 정확 → 본문 무변경 |
| 2 | part1/ch01:142 | CONFIRMED ★ | Titans 2501.00663 App. C ("Similar approaches such as RWKV-7 (Peng 2021) … LMM is generalizing all such models"; Gated DeltaNet η_t=0 동치·Longhorn·TTT 소절) | 주석→VERIFIED 치환. ch12 §12.3.9와 정합 확인 → 본문 무변경 |
| 3 | part1/ch04:129 | CORRECTED | 2212.07677 abstract·§3–4 (self-attention-only TF + 합성 (비)선형 regression; 단층 LSA weight-일치, 다층 GD++) | 본문 문장 후반부 교체: 주장 범위를 "self-attention-only(linear attention) TF + 합성 linear regression"으로 축소 + 위치 병기. 주석→VERIFIED |
| 4 | part1/ch05:74 | CONFIRMED | AGS PRL 55:1530 (1985, αc≈0.14) + Ann. Phys. 173:30 (1987, 0.138 정밀화); 혼합상태=PRA 32:1007 (1985); 웹 서지 대조 | 주석→VERIFIED 치환. 저자 귀속·서술 정확 → 본문 무변경(연도·게재지 괄호 확장은 선택) |
| 5 | part1/ch05:84 | CONFIRMED | 1606.01164 §2 Eq.5 (K^max=α_n N^{n−1}, 고정 오류율) + Eq.6 (무오류 로그 인자형); n=2 → 0.14N | 주석→VERIFIED 치환. "d^{n−1} 스케일" 정확 → 본문 무변경 |
| 6 | part1/ch05:90 | CONFIRMED | 2008.02217 Thm 3 (N≥√p·c^{(d−1)/4}) + Eq.10/App.A.4 (attention 동치, β=1/√d_k) + Thm 4 (one-update 지수 오차) | 주석→VERIFIED 치환. exp capacity·1-step retrieval 서술 정확 → 본문 무변경 |
| 7 | part1/ch06:44 | CONFIRMED | 2307.08621 §2.2·§3.1 (GroupNorm) + 2312.06635 §2 ("without a normalizer" + head별 LN) | 주석→VERIFIED 치환. 본문 무변경 |
| 8 | part1/ch06:117 | CONFIRMED | 2406.06484 §3.3 (SiLU 후 L2 norm; Schlag는 L1) + §3.1 (β_t=σ(W_β x_t)∈(0,1)); 보조 2102.11174 §4.2 | 주석→VERIFIED 치환. 본문 무변경 |
| 9 | part1/ch06:138 | CORRECTED ★ | 2407.14207 §3.2 Eq.5 (채널별 ‖·‖²_{diag(β_t)}) + Thm 3.1 (행별 ε_{t,i}) + 실전 대각 근사(1−ε_{t,i}k_t^{⊙2}) | (6-5) 아래에 "표기 주의" 각주 삽입(채널별 β_t + 대각 근사판 명기). 스칼라 η_t는 단순화로 명기. 주석→VERIFIED |
| 10 | part1/ch06:167 | CORRECTED ★ | 2503.14456 §4.2 Eq.17 (=(6-6) 전치) / κ̂=L2-norm(k⊙ξ) Eq.6·15 / k̃=k⊙lerp(1,a_t,α) Eq.7 | 식 (6-6)은 CONFIRMED. §6.6 도입부(ch06:160) 산문 교체: "별도 projection 분리"→"같은 key의 채널별 변조". 주석→VERIFIED |
| 11 | part1/ch06:173 | CONFIRMED | 2503.14456 abstract·Thm 2(App.D.1, S5 swap NC1-complete, 1 layer)·Thm 3(App.D.2, 모든 정규 언어 4-layer); 전제 TC0≠NC1, 증명은 c=2 변형 | 주석→VERIFIED 치환. 본문은 이미 "주장한다"+"(표준 복잡도 가정 하)" 헤지 → 무변경. §6.7 확대 인용 시 c=2 병기 조건 |
| 12 | part1/ch07:56 | CONFIRMED | 2312.00752 §3.5.1 Theorem 1 (전제 N=1, A=−1, B=1, s_Δ=Linear, τ_Δ=softplus; 증명 App. C) | 주석→VERIFIED 치환. 위치·정리 번호·식 일치 → 본문 무변경 |
| 13 | part1/ch07:114 | CONFIRMED | 2405.21060 §1 ("8×" — 초록 아님, 초록의 2-8×는 속도); 실험별 N∈{16,64,256} (Fig.8·Table 5), 단일 default N 없음 | 주석→VERIFIED 치환. 본문 문장 정확. (선택)헤지형 수치 문장 추가 권고 |
| 14 | part1/ch08:42 | CONFIRMED ★ | 2407.04620 §2.7 "Learnable η" (η(x)=η_base·σ(θ_lr·x); η_base=1 TTT-Linear/0.1 TTT-MLP; "gate for ∇ℓ") | 주석→VERIFIED 치환. 본문 ch08:41 이미 정합(위치는 §2.7) → 무변경 |
| 15 | part1/ch08:47 | CONFIRMED ★ | 2407.04620 §2.7 "Instantiations of f" (f(x)=x+LN(f_res(x)) 상시; TTT-MLP 2층·hidden 4×·GELU) | ch08:46 문단 끝에 구현 세부 문장 추가(§8.3 유도는 겉옷 벗긴 f_lin). 주석→VERIFIED |
| 16 | part1/ch08:65 | CORRECTED ★ | 2407.04620 §2.6 Thm 1 (f=Wx, batch GD η=1/2, W0=0 → 무정규화 linear attention) + Thm 2 (N-W estimator, κ∝exp → self-attention) | ch08:64 문장 교체: 전제 목록 병기(f=Wx·LN/residual 제거, η=1/2, W0=0, 무정규화 대상) + Thm 2 인용 병기. 주석→VERIFIED |
| 17 | part1/ch08:93 | CONFIRMED | 2407.04620 §2.4 (b=16 전 실험 공통; sweep=Fig.7) | 주석→VERIFIED 치환. (선택)본문 ch08:92에 "(b=16 [§2.4, Fig.7])" 병기 권고 |
| 18 | part1/ch08:118 | CONFIRMED ★ | 2407.04620 abstract·§1 (125M–1.3B; "Mamba cannot after 16k" — 그림 Fig.2 우) + §3.1/3.2 (Pile Fig.10 / Books3 Fig.11) | 주석→VERIFIED 치환. 본문 정확 → 무변경. (선택)Fig.2 번호 병기 |
| 19 | part1/ch08:123 | CONFIRMED | 2504.05298 abstract·§1 (CogVideo-X 5B, 3초→1분, text storyboard, Tom and Jerry, TTT-MLP) | 주석→VERIFIED 치환. 본문 무변경 |
| 20 | part1/ch09:73 | CONFIRMED | 2312.06635 §4.3 (log-space Eq.4 + secondary-level chunking / 2-level tiling) | 주석→VERIFIED 치환. 본문 무변경 |
| 21 | part1/ch09:108 | CONFIRMED | 2406.06484 §3.3 Eq.10–11 (UT transform, T_[t]/U_[t]); 원 출처 Joffrain et al. 2006 (그들의 ref [44], 유도 §B.2) | 주석→VERIFIED 치환. (6-6)계와 대수적 동치(경계항 folding 차이). (선택)본문 ch09:106에 "(Joffrain et al. 2006)" 병기 권고 |
| 22 | part1/ch09:120 | CONFIRMED | 2407.04620 §2.5 → App. A (비선형 f dual form) | 주석→VERIFIED 치환. (선택)본문 ch09:118에 "[Sun et al. 2024 App. A]" 병기 권고 |
| 23 | part1/ch09:176 | CONFIRMED | 2504.13173 §5.3 "Parallelizable Training" 단락 (chunk b=16/64, 직전 chunk 마지막 state 기준 gradient, Memora lag token) — §5.4는 부재 | 주석→VERIFIED 치환. **연동 수정**: ch13:291 "[Miras §5.4]"→"§5.3" (아래 연동 항목 참조) |
| 24 | part1/ch10:75 | CONFIRMED | 2505.23735 Fig.5 (c∈{2,4,8,16}, 클수록 ppl↓) + Table 6 (c=1: 19.97→21.98); headline default c·decode incremental 절차는 원문 미명시 | 주석→VERIFIED 치환(enrichment). 본문 구조적 회계만 서술 → 무변경. (선택)수치 [해설] 추가 |
| 25 | part1/ch10:140 | CONFIRMED | NVIDIA H100 datasheet: SXM bf16 989.4≈989 TFLOPS(dense)/1,979(sparsity), HBM3 3.35 TB/s | 주석→VERIFIED 치환. 본문 무변경 |
| 26 | part1/ch11:147 | CONFIRMED | Sleep 2606.03979 서지 [90/91](중복); ReST^EM = "Beyond Human Data", TMLR 2024 = arXiv:2312.06585 | 주석→VERIFIED 치환. 본문 무변경 |
| 27 | part1/ch11:148 | CONFIRMED | Sleep 2606.03979 서지; SEAL = "Self-Adapting Language Models" = arXiv:2506.10943 (첫 인용 = 11장) | ch11:145 "SEAL(Zweiger et al. 2025)"→"…, arXiv:2506.10943)" 확장. 주석→VERIFIED |
| 28 | part2/ch12:139 | CONFIRMED | 2501.00663 Eq.21 (q_t=S^(t)W_Q가 C행 → N_l=C 차원상 강제); Fig.3a 캡션은 별도 상수 N_l 유지(명시 없음); 공식 코드 부재 | 주석→VERIFIED 치환. 본문 ch12:137 헤지 정확 → 무변경(Eq.21 차원 논증 반영 가능) |
| 29 | part2/ch12:141 | CONFIRMED | 2501.00663 Eq.25 (o_t=y_t⊗M*_t(y_t), W_Q 재적용 없음) + Eq.24 (M_t=M_{t−1}(y_t)); 공식 코드 부재 | 주석→VERIFIED 치환. 본문이 원문 표기 그대로 → 무변경 |
| 30 | part2/ch12:297 | CONFIRMED | 2501.00663 §5.9·Table 5 캡션 (모델 규모 언급 없음, 재확인); 수치는 Table 1 400M 평균과 일치 | 주석→VERIFIED 치환. 부재 진술 정확·400M 귀속은 [해설]에 추론으로 유지 → 본문 무변경 |
| 31 | part2/ch13:332 | UNRESOLVABLE ★ | 2504.13173 App. C PDF(p.26): Table 5(Peak LR 3e-3/1.5e-3/1.25e-3)만; **optimizer 종류·batch·스케줄 부재**. 대조: Atlas App. E는 AdamW/cosine/batch 명시 | §6.2 논문 귀속 전환: 표 13-3 두 칸 교체(outer optimizer/batch/스케줄 "원문 미명시" 헤지) + §13.6(ch13:330)에 부재 문장 추가. 주석→VERIFIED |
| 32 | part2/ch14:301 | CONFIRMED | 2505.23735 전수 + App.E + PDF p.13 Fig.3 라벨: 구현 p·PolySketch 차원·memory head 분할 **전부 부재** 확정. App.E memory=2층·4×·GELU·chunk 끝 LN | 주석→VERIFIED 치환. 본문 [해설](ch14:299) 부재 진술 확정 → 무변경 |
| 33 | part2/ch15:258 | CONFIRMED | 2511.07343 PDF p.9: Table 3 열 머리 무표기, Table 2 C4 열과 6/6 일치(23.53/21.04/20.74/20.47/20.15/20.86, 타 열 불일치) | (선택)본문 문장 교체로 "C4 확정적 독해"로 강화(헤지→6/6 대조). 주석→VERIFIED |
| 34 | part2/ch16:297 | CONFIRMED ★ | 2512.24695 §8.1–8.2 Eq.88/90: 우측 곱(α_t I−η_t k_t k_t^⊤)의 적용 대상이 M=2층 residual MLP(Eq.89/91)일 때 미정의(원문 공백). Eq.92–93(행렬 memory)만 타입 명확; 공식 구현 부재 | 주석→VERIFIED 치환. 본문 "원문 미명시" 진술 정확 → 무변경 |
| 35 | part2/ch16:394 | CONFIRMED ★ | 2512.24695 §9.6 ("removes the momentum term in the self-modifying Titans") + Table 6 (Hope 12.24 vs w/o Momentum 13.58); 수식 Eq.88/90엔 부재 | ch16:393 문장 교체: '시사'→§9.6 인용으로 momentum 항 존재 '명시'로 강화(수식 위치 부재는 원문 사실로 병기). 주석→VERIFIED |
| 36 | part2/ch17:9 | CORRECTED | 2512.24695 §2 ("in this work, we focus on the first stage: memory consolidation as an online process"); (3) capacity 고정은 NL 자체 결산 아님 → [Sleep §1] 비판 | ch17:7 네 항목 교체: (2)=NL §2 직접 인용, (3)=[Sleep §1]로 귀속 이전, (4)=(2)의 따름정리로 재표기. 주석→VERIFIED |
| 37 | part2/ch17:202 | CORRECTED ★ | 2606.03979 §4.1 ("We use Llama-3B and Llama3-8B as the backbones") + App. B (5×dim-64 MLP graft, active param 불변) | §17.4 본문에 확정 사실 반영: Hope 실험 = Llama-3B/8B graft(NL-레시피 checkpoint 아님) → regime 1(식 17-1 관통 pre-train) **미실증** 확정. 주석→VERIFIED |

★ = coverage 리포트 위험도 '높음' 관련.

## 연동·범위 밖 수정 (§6.3 해소 규약 3에 따라 이 장부로 전달)

TODO 해소 과정에서 확인된, 남의 장 파일을 직접 고치지 않고 해당 장 fixer에게 전달할 항목(원본 부록 B):

1. **ch13:291** — "[Miras §5.4]" → **"[Miras §5.3]"**. §5.4는 존재하지 않으며 "Parallelizable Training"은 §5.3 내부 단락(ch09:176 해소 중 확인). #23과 동일 근거.
2. **Sleep 서지 [90]/[91] 중복** — 동일 항목(ReST^EM/"Beyond Human Data")의 중복 수록(원문 서지 오류). 책이 Sleep 참고문헌 번호를 인용할 때 주의(#26 관련).
3. **Titans의 RWKV-7 인용 표기 "(Peng 2021)"** — 원문 서지 슬립(RWKV-LM repo 지시). 책에서 Titans App. C를 재인용할 때 "RWKV-7 (arXiv:2503.14456)"로 바로잡아 인용 권고(#2 관련).

## 원전 확보 요약

로컬 6편: 2501.00663 / 2504.13173 / 2505.23735 / 2511.07343 / 2512.24695 / 2606.03979.
외부 13편(papers/external): 2407.04620, 2503.14456, 2407.14207, 2406.06484, 2102.11174, 2312.06635, 2307.08621, 2312.00752, 2405.21060, 2212.07677, 2008.02217, 1606.01164, 2504.05298.
웹: NVIDIA H100 datasheet(ch10:140), Amit et al. 1985 서지(ch05:74).
arXiv ID↔논문↔해소 항목의 전체 대응은 원본 `w1-todo-resolutions.md` 부록 A 참조.
