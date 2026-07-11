# W1 TODO-VERIFY 해소 리포트 (w1-todo-resolutions.md)

- 검증일: 2026-07-12
- 대상: coverage 리포트 §3 인벤토리 + grep 전수 수집 — **총 37건 전량 판정 완료**
- 판정 결과: **CONFIRMED 30 / CORRECTED 6 / UNRESOLVABLE 1**
- 원전 확보: 로컬 6편(papers/*.txt·pdf) + 외부 13편 신규 다운로드(papers/external/*.{pdf,txt}: 2407.04620, 2503.14456, 2407.14207, 2406.06484, 2102.11174, 2312.06635, 2307.08621, 2312.00752, 2405.21060, 2212.07677, 2008.02217, 1606.01164, 2504.05298) + NVIDIA H100 datasheet(웹) + Amit 1985 서지(웹)
- 표기: 각 항목의 **조치**는 fixer가 그대로 실행할 수 있는 수준으로 기술. "주석 제거 후 확인 주석 치환"은 해당 `<!-- TODO-VERIFY: ... -->`를 `<!-- VERIFIED(2026-07-12): ... -->` 한 줄로 바꾸는 것을 뜻한다.
- 위험도 '높음' 12건: ch01:142, ch06:138, ch06:167, ch08:42, ch08:47, ch08:65, ch08:118, ch13:332, ch16:297, ch16:394, ch17:202 (+ coverage 리포트 집계 기준). 전부 원전 인용 명시.

## 판정 요약표

| # | 파일:행 | 판정 | 한 줄 요지 |
|---|---|---|---|
| 1 | part1/ch01:19 | CONFIRMED | 슬로건 원문 wording 확인, 의역 정확 |
| 2 | part1/ch01:142 | CONFIRMED ★ | Titans App. C에 RWKV-7 **명시적으로 등장** |
| 3 | part1/ch04:129 | CORRECTED | von Oswald 경험 주장 범위 축소 필요 |
| 4 | part1/ch05:74 | CONFIRMED | AGS 1985 PRL 서지 확인 (αc≈0.14; 0.138은 동일 계열 정밀값) |
| 5 | part1/ch05:84 | CONFIRMED | K^max = α_n N^{n−1} (Eq.5) + 로그 인자형 (Eq.6) |
| 6 | part1/ch05:90 | CONFIRMED | Thm 3 (지수 capacity), Thm 4 (1-step retrieval) |
| 7 | part1/ch06:44 | CONFIRMED | RetNet GroupNorm·GLA 무정규화 서술 일치 |
| 8 | part1/ch06:117 | CONFIRMED | k = SiLU 후 L2 norm, β_t = σ(W_β x_t) ∈ (0,1) |
| 9 | part1/ch06:138 | CORRECTED ★ | Longhorn objective는 채널별 β_t + 실전은 대각 근사 |
| 10 | part1/ch06:167 | CORRECTED ★ | (6-6) 구조 일치하나 "별도 projection" 표현 부정확 |
| 11 | part1/ch06:173 | CONFIRMED | 표현력 정리 원문 확인 (단서: 증명은 c=2 변형) |
| 12 | part1/ch07:56 | CONFIRMED | Mamba Theorem 1 = §3.5.1, 위치·내용 일치 |
| 13 | part1/ch07:114 | CONFIRMED | "8×"는 §1(초록 아님); 단일 default N은 논문에 없음 |
| 14 | part1/ch08:42 | CONFIRMED ★ | η(x)=η_base·σ(θ_lr·x) — 기억 정확, §2.7 |
| 15 | part1/ch08:47 | CONFIRMED ★ | f(x)=x+LN(f_res(x)); MLP hidden 4×, GELU |
| 16 | part1/ch08:65 | CORRECTED ★ | Thm 1/2 특정 — 전제(η=1/2, LN/residual 제거) 병기 필요 |
| 17 | part1/ch08:93 | CONFIRMED | b=16 전 실험 공통, sweep = Fig. 7 (§2.4) |
| 18 | part1/ch08:118 | CONFIRMED ★ | 125M–1.3B, Pile/Books3, Mamba 16k 정체 = Fig. 2 |
| 19 | part1/ch08:123 | CONFIRMED | CogVideo-X 5B, storyboard→1분, Tom and Jerry |
| 20 | part1/ch09:73 | CONFIRMED | GLA log-space + secondary-level chunking (§4.3) |
| 21 | part1/ch09:108 | CONFIRMED | UT transform 기호 대응·Joffrain et al. 2006 인용 확인 |
| 22 | part1/ch09:120 | CONFIRMED | TTT-MLP dual form = Appendix A |
| 23 | part1/ch09:176 | CONFIRMED | Miras §5.3 "Parallelizable Training" 단락 (§5.4 아님) |
| 24 | part1/ch10:75 | CONFIRMED | c 스윕 {2,4,8,16} (Fig.5) + c=1 ablation; decode 절차 미명시 |
| 25 | part1/ch10:140 | CONFIRMED | H100 SXM bf16 dense 989 TFLOPS·HBM3 3.35 TB/s |
| 26 | part1/ch11:147 | CONFIRMED | ReST^EM = Beyond Human Data = arXiv:2312.06585 |
| 27 | part1/ch11:148 | CONFIRMED | SEAL = arXiv:2506.10943; 첫 인용은 11장 권고 |
| 28 | part2/ch12:139 | CONFIRMED | Eq. 21이 N_l=C를 사실상 강제; Fig. 3a는 별도 상수 유지 |
| 29 | part2/ch12:141 | CONFIRMED | Eq. 25 = o_t = y_t ⊗ M*_t(y_t), W_Q 재적용 없음 |
| 30 | part2/ch12:297 | CONFIRMED | Table 5 캡션·§5.9에 규모 명시 없음 — 헤지 유지 정당 |
| 31 | part2/ch13:332 | UNRESOLVABLE ★ | optimizer 종류·batch·스케줄 원문 부재 (PDF App.C 직접 확인) |
| 32 | part2/ch14:301 | CONFIRMED | p·sketch·head 분할 부재 최종 확인 (Fig. 3 라벨 포함) |
| 33 | part2/ch15:258 | CONFIRMED | 열 머리 무표기 확인; C4 열과 6개 값 전부 일치 재확인 |
| 34 | part2/ch16:297 | CONFIRMED ★ | Eq. 88의 decay 적용 대상 미명시 — 헤지 정확 |
| 35 | part2/ch16:394 | CONFIRMED ★ | §9.6이 momentum 항 존재를 명시 ("removes the momentum term") |
| 36 | part2/ch17:9 | CORRECTED | (2)는 NL 원문 인용 확보; (3)(4)는 귀속 조정 필요 |
| 37 | part2/ch17:202 | CORRECTED ★ | Sleep의 Hope = **Llama-3B/8B graft** — NL 레시피 checkpoint 아님 |

★ = coverage 리포트 위험도 '높음' 관련.

---

## part1/ch01-orientation-rosetta.md

### ch01:19 — CONFIRMED
- **대상 주장**: Sun et al. 2024 §1 슬로건의 의역("hidden state가 모델 그 자체이고, update rule이 self-supervised learning의 한 step").
- **원전** [2407.04620 abstract]: "The key idea is to make the hidden state a machine learning model itself, and the update rule a step of self-supervised learning." §1에도 동일 문장 재등장: "we make the hidden state a machine learning model itself, and the update rule a step of self-supervised learning."
- **판단**: 의역 정확. 원문은 "the hidden state **is** a model"이 아니라 "**make** the hidden state a machine learning model itself" — 본문이 직접 인용 아닌 의역으로 처리했으므로 문제 없음. 출처는 abstract와 §1 양쪽.
- **조치**: 주석 제거, `<!-- VERIFIED(2026-07-12): 원문 wording = "make the hidden state a machine learning model itself, and the update rule a step of self-supervised learning" (2407.04620 abstract·§1). 의역 처리 유지. -->`

### ch01:142 — CONFIRMED (위험도 높음)
- **대상 주장**: "[Titans]는 자신의 update가 Gated DeltaNet, Longhorn, TTT layer 등 기존 recurrent 모델들을 특수 사례로 포함한다고 논한다 [Titans App. C]" + §12.3.9의 RWKV-7 포함 단정.
- **원전** [2501.00663 App. C "Long-term Memory Module (LMM) as a Sequence Model"]: 세 개의 명시 소절 — "LMM is Generalized Gated DeltaNet"(η_t=0으로 동치), "LMM is Generalized Longhorn"(implicit online learning, forget gate 부재 지적), "LMM is Generalized TTT Layer"(3가지 차이). 그리고 **RWKV-7이 명시적으로 등장한다**: "Similar approaches such as **RWKV-7 (Peng 2021)** are also using the same formulation and loss function, and so LMM is generalizing all such models."
- **판단**: TODO의 우려(RWKV-7이 Titans v1보다 늦어 원문에 없을 가능성)와 달리, 보유 판본(papers/2501.00663.txt)의 App. C에 RWKV-7이 이름으로 등장. ch01 본문과 ch12 §12.3.9의 RWKV-7 포함 서술 모두 지지됨. 부기: Titans의 RWKV-7 인용 표기는 "(Peng 2021)"로 RWKV-LM GitHub 저장소를 가리키는 서지 오류성 표기이나, 본문 텍스트가 "RWKV-7"을 명시하므로 포함 판정에 영향 없음.
- **조치**: 주석 제거, `<!-- VERIFIED(2026-07-12): Titans App. C가 Gated DeltaNet(η_t=0 동치)·Longhorn·TTT를 소절로 다루고 RWKV-7도 "Similar approaches such as RWKV-7 ... LMM is generalizing all such models"로 명시. ch12 §12.3.9와 정합. -->`

## part1/ch04-meta-learning-bilevel.md

### ch04:129 — CORRECTED
- **대상 주장**: "von Oswald et al. 2023 (arXiv:2212.07677)은 linear self-attention layer의 weight를 적절히 두면 그 layer의 forward pass가 ... GD 한 step과 일치함을 보이고, **학습된 transformer가 실제로 그런 해에 도달한다**고 보고한다."
- **원전** [2212.07677 abstract]: "we show empirically that when training **self-attention-only Transformers** on **simple regression tasks** either the models learned by GD and Transformers show great similarity or, remarkably, the weights found by optimization match the construction." §3 제목: "Trained Transformers do mimic gradient descent **on linear regression tasks**". 단일 LSA layer는 construction과 weight 수준 일치(Fig. 2), K-layer는 GD++(iterative curvature correction)와 유사, 비선형은 sine-wave regression에서 "deep data representations 위의 linear model" 학습으로 확장.
- **판단**: 본문 문장은 두 가지를 과대 일반화한다 — (i) "학습된 transformer"(무한정) → 원문은 **self-attention-only(linear attention) transformer**, (ii) 과제 무언급 → 원문은 **합성 linear regression**(noiseless teacher) 계열. 원문 자체도 "great similarity **or** weights match"로 이중 헤지.
- **조치**: 본문 해당 문장 후반부를 다음으로 교체 —
  > "…GD 한 step과 일치함을 보이고, **합성 linear regression 과제에 훈련된 self-attention-only(linear attention) transformer**가 실제로 그 construction과 유사하거나(다층: GD++류 curvature 보정) 단층에서는 weight 수준까지 일치하는 해에 도달한다고 보고한다 [2212.07677 abstract, §3–4]."
  이후 주석을 `<!-- VERIFIED(2026-07-12): 범위 = self-attention-only TF + 합성 (비)선형 regression; 단층 LSA는 weight-수준 일치(Fig.2), 다층은 GD++ 유사. -->`로 치환.

## part1/ch05-associative-memory.md

### ch05:74 — CONFIRMED
- **대상 주장**: 0.138d 임계값의 출처 = Amit, Gutfreund & Sompolinsky 1985 (PRL); spurious attractor(홀수 혼합)·임계 초과 시 retrieval 소멸이 같은 분석 계열의 표준 결과인지.
- **원전 확인**:
  - 서지: Amit, Gutfreund & Sompolinsky, "Storing Infinite Numbers of Patterns in a Spin-Glass Model of Neural Networks", **Phys. Rev. Lett. 55, 1530–1533 (1985)** — 연도·게재지 확인(웹 대조). 이 PRL은 α_c ≈ **0.14**를 보고; 정밀값 **0.138**은 같은 AGS replica 분석 계열(정밀화는 Amit, Gutfreund & Sompolinsky, Ann. Phys. 173, 30 (1987) "Statistical mechanics of neural networks near saturation")의 표준 인용값이다. 참고로 Ramsauer et al. 2021도 "the storage capacity is about 0.138d (Crisanti et al., 1986; Hertz et al., 1991)"로 이 값을 동계열 문헌에 귀속한다 [2008.02217 §1].
  - spurious attractor: 저장 pattern들의 홀수 개 혼합(odd mixture states)이 준안정 상태로 공존한다는 것, 그리고 임계 초과 시 retrieval 상이 소멸한다는 것 모두 AGS 분석 계열의 표준 결과(혼합 상태 분석은 AGS, Phys. Rev. A 32, 1007 (1985); 포화 근처 붕괴는 PRL 1985/Ann. Phys. 1987).
- **판단**: 본문("이후 statistical mechanics 분석은 임계값을 약 0.138d로 확정했다(Amit, Gutfreund & Sompolinsky)")은 저자 귀속·서술 모두 정확. 연도·게재지를 병기하려면 PRL 1985 + Ann. Phys. 1987 쌍으로.
- **조치**: 주석 제거, `<!-- VERIFIED(2026-07-12): AGS = PRL 55:1530 (1985, αc≈0.14 보고) + Ann. Phys. 173:30 (1987, 0.138 정밀화). 혼합상태(spurious)·임계 붕괴 = 동일 AGS 계열(PRA 32:1007, 1985 포함) 표준 결과. -->` (원하면 본문 괄호를 "(Amit, Gutfreund & Sompolinsky; PRL 1985, Ann. Phys. 1987)"로 확장 가능 — 필수 아님.)

### ch05:84 — CONFIRMED
- **대상 주장**: Krotov & Hopfield 2016의 capacity 스케일 d^{n−1}의 정확한 statement.
- **원전** [1606.01164 §2, Eq. 5–6]: 고정 오류율 기준 — "K^max = α_n N^{n−1}, where α_n is a numerical constant, which depends on the (arbitrary) threshold 0.5%." (n=2에서 "the well known result K = 0.14N" 복원). 무오류(perfect recovery, P_error < 1/N) 기준 — "K^max_no errors ≈ N^{n−1} / (2(2n−3)!! ln(N))" (로그 인자·상수 포함형).
- **판단**: 본문 "capacity는 d^{n−1} 스케일로 커진다"는 Eq. 5와 정확히 일치. 로그 인자는 무오류 조건에서만 붙음(Eq. 6).
- **조치**: 주석 제거, `<!-- VERIFIED(2026-07-12): K^max = α_n N^{n-1} (고정 오류율, Eq.5); 무오류는 N^{n-1}/(2(2n-3)!! ln N) (Eq.6). n=2 → 0.14N 복원. -->`

### ch05:90 — CONFIRMED
- **대상 주장**: Ramsauer et al. 2021의 (i) exponential capacity 정리의 형태, (ii) update rule = attention 대응 조건, (iii) 1-step retrieval 조건.
- **원전** [2008.02217]:
  - (i) **Theorem 3**: 실패 확률 p, 반지름 K√(d−1) 구면 위 random pattern에 대해 저장 가능 pattern 수 "N ≥ √p · c^{(d−1)/4}", c는 Lambert W로 정의되는 상수(예시로 c ≥ 3.1546 또는 c ≥ 1.3718 성립) — 지수의 밑은 c^{1/4}, 차수는 d−1. (별도로 §1은 이진 pattern·F=exp에서 2^{d/2}인 Demircigil et al. 결과도 인용.)
  - (ii) update rule ξ^new = X softmax(βX^T ξ)가 transformer attention과 동치 — "The new update rule is the attention mechanism of the transformer" [§, "Hopfield update rule is attention of the transformer" + App. A.4]; 조건은 pattern들을 선형 사상으로 key/value화하고 β=1/√d_k로 두는 것(Eq. 10 주변).
  - (iii) **Theorem 4**: "With query ξ, after one update the distance of the new point f(ξ) to the fixed point x*_i is exponentially small in the separation Δ_i" — 잘 분리된 pattern(separation Δ_i 큼)에 대해 one update로 ε-close retrieval.
- **판단**: 본문 서술("1-step retrieval update가 정확히 softmax attention의 형태가 되고, capacity는 d에 exponential") 모두 원문과 일치.
- **조치**: 주석 제거, `<!-- VERIFIED(2026-07-12): Thm 3 — N ≥ √p·c^{(d−1)/4} (c≥1.37 예시); attention 동치 = Eq.10/App.A.4 (β=1/√d_k); Thm 4 — one-update 오차가 Δ_i에 지수적으로 작음. -->`

## part1/ch06-linear-attention-fwp.md

### ch06:44 — CONFIRMED
- **대상 주장**: "RetNet 이후 분모 누적 제거 + 출력 normalization(GroupNorm 계열) 대체".
- **원전**:
  - [2307.08621 §2.2]: retention은 분모 누적이 없고, 출력에서 "Y = GroupNorm_h(Concat(head_1,…,head_h))"; §3.1 "Retention Score Normalization"은 GroupNorm의 scale-invariance를 수치 안정화에 사용.
  - [2312.06635 §2]: "recent work has found that a linear kernel (i.e., setting φ to be the identity) **without a normalizer** works well in practice (Sun et al., 2023a[=RetNet]). This results in an (unnormalized) linear attention layer"; 출력측은 head별 LayerNorm("LayerNorm (LN) is applied after the output of each [head]") — GroupNorm 계열 동등물.
- **판단**: 본문 서술 그대로 일치.
- **조치**: 주석 제거, `<!-- VERIFIED(2026-07-12): RetNet Eq/§2.2·§3.1(GroupNorm), GLA §2("without a normalizer" + head별 LN). -->`

### ch06:117 — CONFIRMED
- **대상 주장**: DeltaNet 구현 관행 — key L2 normalization, η_t = sigmoid 출력 (0,1).
- **원전** [2406.06484]:
  - §3.3 "Feature map and normalization": "k_t = SiLU(W_K x_t)/‖SiLU(W_K x_t)‖₂ … Schlag et al. [101] used the **L1 norm** to normalize query/key vectors … We instead apply **L2 normalization**" (β_t=1이면 I−k_t k_t^T가 projection이 되는 해석 포함).
  - β_t 정의: "β_t = σ(W_β x_t) ∈ (0,1) is a soft 'writing strength'" (§3.1).
  - 참고 [2102.11174 §4.2]: Schlag et al.은 β = σ(W_β x) (sigmoid) + "sum normalisation"(L1 계열)을 사용.
- **판단**: 본문("key를 ℓ2 normalize하고 η_t를 sigmoid로 (0,1) 범위") 정확 — 단 L2는 Yang et al. 2024의 선택이고 Schlag 2021은 L1이었다는 세부는 이미 "실무 구현"이라는 현재형 서술과 정합.
- **조치**: 주석 제거, `<!-- VERIFIED(2026-07-12): 2406.06484 §3.3 (SiLU 후 L2 norm; Schlag는 L1), β_t=σ(W_β x_t)∈(0,1) "writing strength". -->`

### ch06:138 — CORRECTED (위험도 높음)
- **대상 주장**: 본문 식 (6-5) 직전의 objective를 스칼라 η_t로 적음 — "W_t = argmin ‖W−W_{t−1}‖²_F + η_t‖Wk_t − v_t‖²₂".
- **원전** [2407.14207 §3.2, Eq. 5 + Theorem 3.1]:
  - Objective (Eq. 5): "S_t = argmin_{S∈R^{d×m}} ‖S − S_{t−1}‖²_F + **‖Sk_t − x_t‖²_{diag(β_t)}**", β_t ∈ R^d — **출력 채널별(per-output-channel) 가중 벡터**. "β_{t,i} = 0 implies S_{t,i} = S_{t−1,i}."
  - 닫힌 해 (Thm 3.1): 행별로 "S_{t,i} = (I − ε_{t,i} k_t k_t^⊤) S_{t−1,i} + ε_{t,i} k_t x_{t,i}, where **ε_{t,i} = β_{t,i}/(1 + β_{t,i} k_t^⊤ k_t)** ∈ [0,∞)".
  - 파라미터화: "k_t = W_k x_t, β_t = σ(W_β x_t)" (sigmoid).
  - **추가 사실**: 실전 구현은 parallel scan을 위해 full transition을 **대각 근사**로 치환 — "in practice, we use the diagonal approximation 1_m − ε_{t,i} k_t^{⊙2} in place of I − ε_{t,i} k_t k_t^⊤" (§3.2, Thm 3.1 직후).
- **판단**: 본문의 스칼라 η_t는 단순화(그 자체는 유도의 본질을 보존)이나, (i) 원문은 채널별 β_t이고 (ii) 출하된 Longhorn kernel은 Householder형이 아니라 **대각 근사**라는 두 사실이 명시돼야 한다. (6-5)와 그 아래 안정성 논의(§6.5)는 "원문 그대로"가 아니라 "원문의 행별 식을 스칼라·full-transition으로 단순화한 판"임.
- **조치**: (6-5) 아래(현 주석 자리)에 다음 각주를 삽입하고 주석 치환 —
  > "표기 주의: 원문 objective는 두 번째 항이 채널별 가중 노름 ‖Sk_t − x_t‖²_{diag(β_t)} (β_t ∈ R^{d_v}, β_t = σ(W_β x_t))이고, 닫힌 해도 출력 행별로 ε_{t,i} = β_{t,i}/(1+β_{t,i}‖k_t‖²)를 갖는다 [Longhorn Eq. 5, Thm 3.1]. 위 (6-5)는 β_t를 스칼라로 둔 단순화다. 또한 원문 구현은 parallel scan을 위해 transition I − ε_{t,i}k_tk_t^⊤를 대각 근사 1 − ε_{t,i}k_t^{⊙2}로 치환한다 [Longhorn §3.2] — 즉 출하된 Longhorn은 delta 계열의 '대각 근사판'이다."
  `<!-- VERIFIED(2026-07-12): 원문 = 채널별 diag(β_t) 가중 + 행별 ε_{t,i} + 실전 대각 근사. 스칼라 판은 단순화로 명기. -->`

### ch06:167 — CORRECTED (위험도 높음)
- **대상 주장**: 식 (6-6)의 원문 대응 — 제거 key 정규화 방식, a_t 위치, 기록 key 변조 형태. 그리고 §6.6 도입부의 "지우는 key와 쓰는 key가 **서로 다른 projection**으로 분리된다(제거용 k̂_t, 기록용 k̃_t)".
- **원전** [2503.14456 §4.2]:
  - State evolution (Eq. 17): "wkv_t = wkv_{t−1}(diag(w_t) − κ̂_t^⊤(a_t ⊙ κ̂_t)) + v_t^⊤·k̃_t" — 행벡터 관행. 열벡터로 전치하면 본문 (6-6) W_t = W_{t−1}(Diag(w_t) − k̂_t(a_t⊙k̂_t)^⊤) + v_t k̃_t^⊤와 **구조·기호 배치 완전 일치**. a_t가 곱해지는 위치(제거항 내부의 ⊙) 확인.
  - 제거 key: **κ_t = k_t ⊙ ξ** (Eq. 6; ξ는 학습되는 "removal key multiplier"), 이후 **head별 L2 정규화** "κ̂_t = κ_t/‖κ_t‖₂" (Eq. 15).
  - 기록 key: **k̃_t = k_t ⊙ lerp(1, a_t, α)** (Eq. 7; α는 "replacement rate booster") — "normalized key … decouple a_t from the amount actually added to the state".
  - a_t = in-context learning rate 벡터 (Eq. 4, sigmoid 계열 산출).
- **판단**: 식 (6-6) 자체는 원문과 일치(CONFIRMED). 그러나 도입부 산문의 "**서로 다른 projection으로 분리**"는 부정확 — 둘 다 **같은 key precursor k_t의 채널별 변조**이지 별도 projection 행렬이 아니다(제거용 = k⊙ξ 후 L2 정규화, 기록용 = k⊙lerp(1,a_t,α)).
- **조치**: §6.6 도입부(ch06:160)의 해당 구절을 다음으로 교체 —
  > "…그리고 지우는 key와 쓰는 key가 같은 key에서 서로 다른 채널별 변조로 분리된다: 제거용 k̂_t는 k_t ⊙ ξ(학습된 채널 배율)를 head별 L2 정규화한 것, 기록용 k̃_t는 k_t ⊙ lerp(1, a_t, α)다 [RWKV-7 Eqs. 6–7, 15]."
  (6-6) 아래 주석을 `<!-- VERIFIED(2026-07-12): (6-6) = RWKV-7 Eq.17의 전치. κ̂=L2-norm(k⊙ξ) (Eq.6,15), a_t 위치 일치, k̃=k⊙lerp(1,a_t,α) (Eq.7). -->`로 치환.

### ch06:173 — CONFIRMED
- **대상 주장**: "[RWKV-7] 논문은 generalized delta rule이 (표준 복잡도 가정 하에) transformer의 TC^0 한계를 넘는 state tracking을 가능하게 한다고 주장한다."
- **원전** [2503.14456 abstract·§4.4(요지)·App. D]:
  - abstract: "can perform state tracking and recognize all regular languages, while retaining parallelizability of training … exceeds the capabilities of Transformers under standard complexity conjectures, which are limited to TC0."
  - 정확한 정리: **Theorem 2** (App. D.1) — "RWKV-7 can solve a problem which is NC1-complete under AC0 reductions" (S5 원소 5개 swap tracking, **단일 layer**, Lemma 2). **Theorem 3** (App. D.2) — "For any regular language, there exists a **4-layer** RWKV-7 model that recognizes it." 전제는 TC0 ≠ NC1 conjecture.
  - **핵심 단서**: 증명은 transition을 A_t = diag(w_t) − **c**·κ̂^⊤(a⊙κ̂)에서 **c=2**로 둔 변형(고유값 −1 허용)을 사용 — "Recall that the RWKV-7 wkv state is updated … where c = 1. In the following, we will consider c = 2" [App. D.1]. 즉 출하 아키텍처(c=1)가 아니라 음의 고유값을 허용한 변형에 대한 결과.
- **판단**: 본문은 이미 "주장한다" + "(표준 복잡도 가정 하에)"로 헤지되어 있어 원문과 일치. 정밀 진술을 본문에 넣을 경우 위 정리 번호·layer 수·c=2 단서를 병기할 것.
- **조치**: 주석 제거, `<!-- VERIFIED(2026-07-12): Thm 2(App.D.1) S5 swap-tracking(NC1-complete, 1 layer), Thm 3(App.D.2) 모든 정규 언어 4-layer. 전제 TC0≠NC1. 단 증명은 c=2 변형(음 고유값) 사용 — 출하 c=1과 구분. -->` §6.7에서 이 정리를 인용 확대할 경우 c=2 단서를 반드시 병기.

## part1/ch07-ssm-lineage.md

### ch07:56 — CONFIRMED
- **대상 주장**: gate 환원(h_t = (1−g_t)h_{t−1} + g_t x_t, g_t = σ(Linear(x_t)))이 Mamba 원문의 Theorem 1(§3.5.1)인지.
- **원전** [2312.00752 §3.5.1 "Connection to Gating Mechanisms"]: "Theorem 1. When N = 1, A = −1, B = 1, s_Δ = Linear(x), and τ_Δ = softplus, then the selective SSM recurrence (Algorithm 2) takes the form g_t = σ(Linear(x_t)), h_t = (1 − g_t)h_{t−1} + g_t x_t." (증명은 Appendix C; Gu, Johnson, Goel et al. 2021 Lemma 3.1의 개선으로 자리매김.)
- **판단**: 위치(§3.5.1)·정리 번호(Theorem 1)·식 형태 모두 본문과 일치.
- **조치**: 주석 제거, `<!-- VERIFIED(2026-07-12): Mamba Theorem 1, §3.5.1 (전제 N=1, A=−1, B=1, s_Δ=Linear, τ_Δ=softplus; 증명 App. C). -->`

### ch07:114 — CONFIRMED (수치 추가는 헤지형으로)
- **대상 주장**: 본문 "스칼라 gate 덕에 커널이 단순해져 state 차원도 Mamba-1보다 크게 키울 수 있게 되었다" + TODO의 두 질문(기본 N, "8×" 출처).
- **원전** [2405.21060]:
  - "8× larger state" 출처: **§1 Introduction** "Efficient Algorithms" 문단 — "A dedicated implementation of SSD is 2−8× faster … while simultaneously allowing for much larger recurrent state sizes (**8× the size of Mamba** or even higher, with minimal slowdown)." **초록에는 없다**(초록의 2-8×는 속도).
  - 기본 state 차원: **논문 본문에 단일 default N 명시 없음.** 실험별로 — MQAR(Fig. 8): N ∈ {16, 64, 256}; ablation Table 5: "All models have state expansion factor N = 64 and head size P = 64"; Table 7 비교: 130M은 N=64, 380M은 N=256. §9.2 언어모델링 셋업과 App. D 훈련 레시피에는 N 무언급. (공식 구현의 default d_state=128은 코드 사실이지 논문 사실이 아님.)
- **판단**: 본문 문장 자체는 원문과 일치. TODO가 요구한 "수치 본문 추가"는 다음 헤지형으로만 가능.
- **조치**: 본문 해당 문장 뒤에 추가(선택) — "원문은 이를 'Mamba의 8배 이상'으로 표현하며 [Mamba-2 §1], 실험은 과제별로 N ∈ {16, 64, 256}을 쓴다(단일 default N은 논문에 명시가 없다) [Mamba-2 Fig. 8, Table 5]." 주석 치환: `<!-- VERIFIED(2026-07-12): "8×"=§1(초록 아님); 논문에 단일 default N 없음(실험별 16/64/256). N=128은 코드 default로 논문 밖. -->`

## part1/ch08-ttt-lineage.md

### ch08:42 — CONFIRMED (위험도 높음)
- **대상 주장**: learned inner lr의 함수형(η_t = η_base·σ(θ_lr·x_t)로 기억)과 위치.
- **원전** [2407.04620 §2.7 "Learnable η"]: "Concretely, we design **η(x) = η_base σ(θ_lr · x)**, where the learnable vector θ_lr is an outer-loop parameter, σ is the sigmoid function, and the scalar η_base is the base learning rate, **set to 1 for TTT-Linear and 0.1 for TTT-MLP**. Alternatively, η(x) can also be interpreted as a gate for ∇ℓ." (추가: App. — "We tried η_base ∈ {0.01, 0.1, 1, 10} and used the largest value that does not cause instabilities"; TTT-MLP는 η_base linear warmup.)
- **판단**: 기억한 함수형 정확. 위치는 §2.4가 아니라 **§2.7 Implementation details**.
- **조치**: 주석 제거, `<!-- VERIFIED(2026-07-12): η(x)=η_base·σ(θ_lr·x), §2.7 "Learnable η"; η_base=1(TTT-Linear)/0.1(TTT-MLP); "gate for ∇ℓ" 해석 원문에 있음. -->` 본문(ch08:41)은 이미 정합 — 수정 불요.

### ch08:47 — CONFIRMED (위험도 높음)
- **대상 주장**: 원문 f의 구조 — residual + LayerNorm 포함 여부, TTT-MLP의 hidden 배수·activation.
- **원전** [2407.04620 §2.7 "Instantiations of f"]: "For TTT-Linear, f_lin(x) = Wx, where W is square. For TTT-MLP, f_MLP has two layers similar to the MLPs in Transformers. Specifically, **the hidden dimension is 4× the input dimension, followed by a GELU activation**. For better stability during TTT, **f always contains a Layer Normalization (LN) and residual connection. That is, f(x) = x + LN(f_res(x))**, where f_res can be f_lin or f_MLP."
- **판단**: residual+LN 상시 포함(두 변형 공통), MLP hidden 4×, GELU — 전부 확인. 본문 §8.2는 이 세부를 생략하고 있으므로 아래 한 문장을 추가하면 (6-6)식 계열 유도(§8.3의 M(k;W)=Wk 가정)와의 간극도 정직해진다.
- **조치**: ch08:46 문단 끝에 추가 — "구현 세부: 원문의 f는 안정성을 위해 항상 residual과 LayerNorm을 두른다 — f(x) = x + LN(f_res(x)); TTT-MLP의 hidden은 입력의 4×, activation은 GELU다 [Sun et al. 2024 §2.7]. §8.3의 유도는 이 겉옷을 벗긴 f_lin에 대한 것이다." 주석 치환: `<!-- VERIFIED(2026-07-12): f(x)=x+LN(f_res(x)) 상시; TTT-MLP 2층·4×·GELU (§2.7). -->`

### ch08:65 — CORRECTED (위험도 높음)
- **대상 주장**: "W_0=0일 때 TTT-Linear는 vanilla linear attention과 같은 함수 — 정리로 제시" + "kernel regression → softmax attention"의 정리 번호·전제.
- **원전** [2407.04620 §2.6 "Theoretical equivalences"]:
  - **Theorem 1**: "Consider the TTT layer with **f(x) = Wx** as the inner-loop model, **batch gradient descent with η = 1/2** as the update rule, and **W_0 = 0**. Then … the output rule defined in Equation 5 produces the same output sequence z_1,…,z_T as linear attention." 각주 8: 동치 대상은 normalizer·feature expansion 없는 "simplest formulation of linear attention".
  - **Theorem 2**: "Consider the TTT layer with the **Nadaraya-Watson estimator** … κ(x,x′; θ_K, θ_Q) ∝ e^{(θ_K x)^⊤ θ_Q x′} … produces the same output sequence … as **self-attention**."
  - 관련 확인 [같은 논문, related-work 절]: "TTT-Linear with inner-loop mini-batch size 1, **without the Layer Norm and residual connection**" — LN/residual 제거가 동치의 암묵 전제임을 뒷받침(Thm 1의 f(x)=Wx 자체가 LN/residual 없는 형태).
- **판단**: 본문 서술의 방향은 맞으나 전제 목록이 비어 있다. 정리 번호(1/2)와 전제(η=1/2 — 본문 표기로는 gradient의 상수 2를 η에 흡수한 것과 동치, W_0=0, LN/residual 없는 f=Wx, 대상은 무정규화 linear attention)를 병기해야 "정리로 제시"가 검증 가능한 문장이 된다.
- **조치**: ch08:64의 "…[Sun et al. 2024]가 정리로 제시하는 동치다" 부분을 다음으로 교체 —
  > "…[Sun et al. 2024 Theorem 1]이 정리로 제시하는 동치다(전제: f(x)=Wx — LN/residual 제거 —, batch GD, 원문 표기로 η=1/2 — 이 책은 상수 2를 η에 흡수 —, W_0=0; 대상은 분모 누적 없는 무정규화 linear attention)."
  이어지는 둘째 동치 문장 끝에 "[Sun et al. 2024 Theorem 2 — Nadaraya-Watson + exp kernel]" 인용 병기. 주석 치환: `<!-- VERIFIED(2026-07-12): Thm 1(§2.6): f=Wx, batch GD η=1/2, W0=0 → simplest linear attention. Thm 2(§2.6): N-W estimator, κ∝exp((θ_K x)^⊤θ_Q x′) → self-attention. -->`

### ch08:93 — CONFIRMED
- **대상 주장**: default inner mini-batch b=16, sweep 실험 위치.
- **원전** [2407.04620 §2.4 "Parallelization with mini-batch TTT"]: "Empirically, b controls a trade-off between speed and quality, as shown in **Figure 7**. **We chose b = 16 for all experiments** in this paper." (Fig. 7 캡션: "Ablations on TTT mini-batch size b, where b = 1 is online GD and b = T is batch GD.")
- **판단**: b=16 확인. 위치는 §2.5가 아니라 **§2.4**(dual form이 §2.5).
- **조치**: 주석 제거, `<!-- VERIFIED(2026-07-12): b=16 전 실험 공통(§2.4), sweep=Fig.7. -->` 본문 ch08:92의 "중간 크기의 chunk를 default로 채택"에 "(b=16 [Sun et al. 2024 §2.4, Fig. 7])" 병기 권고.

### ch08:118 — CONFIRMED (위험도 높음)
- **대상 주장**: 비교 스케일 125M–1.3B, 데이터셋 Pile/Books, "Mamba는 16k 이후 정체"의 그림 번호.
- **원전** [2407.04620]:
  - 스케일: abstract — "We evaluate our instantiations at the scale of **125M to 1.3B parameters**, comparing with a strong Transformer and Mamba."
  - 정체 주장: abstract — "TTT-Linear and TTT-MLP can keep reducing perplexity by conditioning on more tokens, **while Mamba cannot after 16k context**"; §1 — "the same metric **plateaus for Mamba after 16k**"; 해당 그림 = **Figure 2**(우측 panel; token index별 mean perplexity).
  - 데이터셋: §3.1 — **Pile** (2k/8k, Figure 10); §3.2 — **Books3** ("a subset of the Pile called Books3"; 2k/32k, Figure 11; 전체 sweep 1k–32k는 Figure 15/16, App. C).
- **판단**: 본문 서술 전부 원전과 일치.
- **조치**: 주석 제거, `<!-- VERIFIED(2026-07-12): 125M–1.3B(abstract); Pile(Fig.10)/Books3(Fig.11); "Mamba cannot after 16k" = abstract·§1, 그림은 Fig.2(우). -->` 본문에 그림 번호를 넣으려면 "…정체한다고 보고한다 [Sun et al. 2024 Fig. 2]"로.

### ch08:123 — CONFIRMED
- **대상 주장**: Dalal et al. 2025의 backbone(CogVideo-X 5B)·셋업(storyboard 조건부 Tom and Jerry 1분 생성).
- **원전** [2504.05298 abstract·§1]: "We start from a pre-trained Diffusion Transformer (**CogVideo-X 5B**) that could only generate 3-second [clips] … to generate **one-minute videos from text storyboards** … we curate a dataset based on **Tom and Jerry cartoons** [with human-annotated storyboards]." TTT-MLP 사용은 논문 전반(28회 언급)에서 확인.
- **판단**: 본문·기억 모두 정확.
- **조치**: 주석 제거, `<!-- VERIFIED(2026-07-12): CogVideo-X 5B(3초→1분), text storyboard 조건, Tom and Jerry 데이터셋, TTT-MLP 삽입 (abstract·§1). -->`

## part1/ch09-chunkwise-parallel-training.md

### ch09:73 — CONFIRMED
- **대상 주장**: GLA의 secondary-level chunking과 log-space decay 처리.
- **원전** [2312.06635]: §3(Eq.4 부근) "we can compute in log space for P"; §4.3 "**Secondary-level chunking**. Unlike in ordinary linear attention, [the intra-chunk part cannot use half-precision matmuls] due to log space computations (Eq. 4). To make better use of tensor cores, we use secondary-level chunking … With this **two-level tiling** strategy…"; 그림 캡션 — "intra-chunk dependencies are modeled via secondary chunking/tiling where the inter-sub-chunk part … [half precision matmul] and the [intra-sub-chunk] part … is computed in full precision in log space."
- **판단**: 본문("log-공간 계산과 chunk 내부의 2차 tile 분할") 정확.
- **조치**: 주석 제거, `<!-- VERIFIED(2026-07-12): GLA §4.3 — log-space(Eq.4) + secondary-level chunking(2-level tiling; inter-sub-chunk는 half-precision matmul, intra-sub-chunk는 full-precision log-space). -->`

### ch09:108 — CONFIRMED
- **대상 주장**: (9-5)의 삼각계가 2406.06484의 UT-transform 정식화와 대응하는지, "UT transform"의 원 출처 인용.
- **원전** [2406.06484 §3.3, Eq. 10–11]: "we further leverage the **UT transform** [44, 23] … **T_[t] = (I + tril(diag(β_[t]) K_[t] K_[t]^⊤, −1))^{−1} diag(β_[t])**, **U_[t] = T_[t] V_[t]** … The inverse of lower triangular matrices could be solved efficiently using forward substitution."
  - 대응: 본문 (9-5) "(I + D_η tril(K_nK_n^⊤,−1)) Ṽ = D_η(V_n − K_nW_ξ^⊤)"는 D_η tril(A,−1) = tril(D_η A,−1)이므로 좌변이 동일한 단위 하삼각계이고, 우변은 원문이 별도 항(Eq. 8–9의 W_[t]S_[t])으로 처리하는 chunk 경계 보정 −K_nW_ξ^⊤을 RHS에 접어 넣은 것 — 대수적으로 동치.
  - 원 출처: ref **[44] = Joffrain, Low, Quintana-Ortí, van de Geijn, Van Zee, "Accumulating Householder transformations, revisited", ACM Trans. Math. Softw. 32:169–179, 2006** — 본문 주석의 추정과 일치 (+ ref [23] Dominguez & Ortí 2018). 유도는 그들의 §B.2.
- **판단**: 기호 배치 대응·원 출처 모두 확인.
- **조치**: 주석 제거, `<!-- VERIFIED(2026-07-12): 2406.06484 Eq.10–11과 동치(경계항 folding 차이만); UT 원 출처 = Joffrain et al. 2006 (그들의 ref [44]), 유도 §B.2. -->` 본문 ch09:106의 "(수치선형대수 문헌의 UT transform)"에 "(Joffrain et al. 2006)" 병기 권고.

### ch09:120 — CONFIRMED
- **대상 주장**: TTT-MLP dual form의 유도 위치(부록 절 번호).
- **원전** [2407.04620 §2.5]: "In **Appendix A**, we show that the dual form still works when f is a neural network with nonlinear layers, except with more complicated notation." (App. A.1 Forward pass / A.2 … / A.4 구성.)
- **조치**: 주석 제거, `<!-- VERIFIED(2026-07-12): TTT-MLP(비선형 f) dual form = Appendix A (§2.5에서 지시). -->` 본문 ch09:118의 "부록에서 보였다"에 "[Sun et al. 2024 App. A]" 병기 권고.

### ch09:176 — CONFIRMED
- **대상 주장**: Miras의 Moneta/Yaad/Memora가 chunkwise 기법으로 훈련된다는 서술의 원문 절 번호.
- **원전** [2504.13173 §5.3 내 "Parallelizable Training" 단락]: "we build upon the work of Behrouz et al. (2024c) and Sun et al. (2024) to make the training parallelizable. The main idea is to divide the sequence into chunks with size b (**usually is 16 or 64**) and calculate the gradient for all tokens in the current chunk with respect to the last state of the memory in the previous chunk. That is, we use ∇ℓ(M_t′; k_t, v_t) instead of ∇ℓ(M_{t−1}; k_t, v_t)…" (Memora의 lag token: "we consider a lag tokens after each chunk (i.e., tokens with index i = kb+1…)").
- **주의**: 절 번호는 **§5.3**("Miras's Variants: Moneta, Yaad, and Memora")의 내부 단락이다 — §5.4는 존재하지 않는다. **ch13:291이 "[Miras §5.4]"로 인용하고 있는 것은 오기이므로 §5.3으로 함께 수정할 것** (TODO 범위 밖이지만 동일 근거로 즉시 수정 가능).
- **조치**: 주석 제거, `<!-- VERIFIED(2026-07-12): Miras §5.3 "Parallelizable Training" 단락 (chunk b=16/64, 직전 chunk 마지막 state 기준 gradient, Memora lag token). -->` + ch13:291의 §5.4 → §5.3 수정.

## part1/ch10-systems-bridge.md

### ch10:75 — CONFIRMED (enrichment 제공)
- **대상 주장**: Atlas 실험의 실제 window 길이 c 값(들), decode 시 Omega rule의 incremental 유지 여부.
- **원전** [2505.23735]:
  - c 값: **Figure 5** (p.14) — "The effect of local context length (i.e. c) on the performance of OmegaNet with different global context length"; 범례가 **C=2, C=4, C=8, C=16** 4개 곡선(global context 2K–16K), c가 클수록 ppl 낮음. **Table 6** ablation (p.18) — "c = 1" 행: ppl 21.98 vs Atlas 19.97 (acc 49.26 vs 52.77). **headline 실험(Table 2 등)의 default c는 논문에 명시가 없다.**
  - decode: 병렬 훈련(§3.4의 sliding-window mask M_s, §5.1)만 기술; **decode 시 incremental 유지 절차(최근 c쌍 buffer 관리 등)는 논문에 없음.**
- **판단**: 본문(ch10:73)은 구조적 회계만 서술하므로 틀린 것 없음. ch14와의 정합화: ch14도 동일 사실(수치 부재) 기준으로 이미 처리됨.
- **조치**: 주석 치환 — `<!-- VERIFIED(2026-07-12): 실측 c: Fig.5 스윕 c∈{2,4,8,16}(클수록 ppl↓), Table 6 ablation c=1(19.97→21.98). headline 설정의 default c와 decode-시 incremental 절차는 원문 미명시. -->` 본문에 수치를 넣으려면 "Atlas의 c 스윕은 {2,4,8,16}이고 c=1로 줄이면 ppl이 19.97→21.98로 나빠진다 [Atlas Fig. 5, Table 6]; headline 설정의 c와 decode 시 유지 절차는 명시가 없다"를 [해설]로 추가.

### ch10:140 — CONFIRMED
- **대상 주장**: H100 SXM bf16 dense peak ≈ 989 TFLOP/s, HBM3 bandwidth 3.35 TB/s.
- **원전**: NVIDIA H100 Tensor Core GPU Datasheet (최신판, nvidia.com) — H100 SXM: "BFLOAT16 Tensor Core 1,979 teraFLOPS*" (*with sparsity → **dense = 989.4 ≈ 989 TFLOPS**), "GPU Memory Bandwidth **3.35TB/s**". (PCIe판은 2TB/s이므로 SXM 명시 유지가 옳다. 구판 datasheet는 3TB/s를 표기했으나 현행 스펙은 3.35TB/s.)
- **조치**: 주석 제거, `<!-- VERIFIED(2026-07-12): NVIDIA H100 datasheet — SXM bf16 1,979 TFLOPS(sparsity)/989 dense, HBM3 3.35 TB/s. -->`

## part1/ch11-continual-learning.md

### ch11:147 — CONFIRMED
- **대상 주장**: ReST^EM(Singh et al. 2024) = arXiv:2312.06585 (*Beyond Human Data: Scaling Self-Training…*)인지.
- **원전** [2606.03979 서지 [90]/[91]]: "Avi Singh, John D Co-Reyes, Rishabh Agarwal, … and Noah Fiedel. '**Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models**'. In: Transactions on Machine Learning Research (2024). … url: openreview.net/forum?id=lNAyUngGFK." 본문 인용은 "We follow SEAL and use ReSTEM algorithm (Singh et al. 2024a)". 이 논문의 arXiv ID는 2312.06585 (제목 일치; TMLR 판).
- **부기**: Sleep 서지의 [90]과 [91]은 **동일 항목의 중복 수록**(원문 서지 오류) — 인용 시 혼동 주의.
- **조치**: 주석 제거, `<!-- VERIFIED(2026-07-12): ReST^EM = Singh et al., "Beyond Human Data…", TMLR 2024 = arXiv:2312.06585 (Sleep 서지 [90]=[91] 중복). -->`

### ch11:148 — CONFIRMED
- **대상 주장**: SEAL(Zweiger et al. 2025) = arXiv:2506.10943 추정; 첫 인용 병기 위치.
- **원전** [2606.03979 서지]: "Adam Zweiger, Jyothish Pari, Han Guo, Ekin Akyürek, Yoon Kim, and Pulkit Agrawal. '**Self-Adapting Language Models**'. In: arXiv preprint **arXiv:2506.10943** (2025)." — 추정 ID 정확.
- **첫 인용 위치 권고**: SEAL이 실질 내용(self-edit 루프)과 함께 처음 등장하는 곳이 ch11:145이므로 **11장에서 전체 서지(arXiv:2506.10943) 병기, 17장은 약식 재인용** — 책의 "첫 등장 전체 인용" 관행과 일치.
- **조치**: ch11:145의 "SEAL(Zweiger et al. 2025)"을 "SEAL(Zweiger et al. 2025, arXiv:2506.10943)"로 확장, 주석 제거·치환 `<!-- VERIFIED(2026-07-12): SEAL="Self-Adapting Language Models", arXiv:2506.10943. 첫 인용 = 이 장. -->`

## part2/ch12-titans.md

### ch12:139 — CONFIRMED
- **대상 주장**: MAC의 retrieved history token 수 N_l = C 여부 (원문 미명시, N_l=C가 자연스러운 독해라는 헤지).
- **원전** [2501.00663]:
  - Eq. 21: "h_t = M*_{t−1}(q_t), where **q_t = S^(t) W_Q**" — query가 segment 전체(C개 token 행렬)이므로 검색 결과 h_t도 **C행** — 즉 식 차원에서 N_l = C가 강제된다.
  - Fig. 3a 캡션: "the first **N_p** tokens are persistent memory and the next **N_l** are long-term memory tokens" — 별도 상수 N_l 표기 유지, N_l=C 명시 없음.
  - 공개 공식 코드 부재로 코드 대조는 불가(비공식 구현은 근거로 쓰지 않음).
- **판단**: 본문(ch12:137)의 헤지 서술("원문 mask 표기로는 N_l개")과 TODO의 독해가 원문과 정확히 일치. Eq. 21의 차원 논증을 근거로 "N_l = C가 식에서 따라 나온다"까지는 본문에 반영 가능.
- **조치**: 주석 치환 — `<!-- VERIFIED(2026-07-12): Eq.21의 q_t=S^(t)W_Q가 C행이므로 N_l=C가 차원상 강제됨; Fig.3a 캡션은 별도 상수 N_l 유지(명시 없음); 공식 코드 부재. -->`

### ch12:141 — CONFIRMED
- **대상 주장**: [Titans Eq. 25]의 최종 read가 y_t를 그대로 query로 쓰는지(W_Q 재적용 없이).
- **원전** [2501.00663 Eqs. 24–25]: "M_t = M_{t−1}(y_t)" (Eq. 24), "**o_t = y_t ⊗ M*_t(y_t)**" (Eq. 25). — read의 인자는 y_t 그대로이며 W_Q 재적용 표기 없음. 직후 문장: "we are updating the weight of M_{t−1} through forward pass."
- **판단**: 본문이 원문 표기 그대로 옮겼음을 확인.
- **조치**: 주석 치환 — `<!-- VERIFIED(2026-07-12): Eq.25 = o_t = y_t ⊗ M*_t(y_t) (W_Q 재적용 없음, 원문 표기 그대로); Eq.24 = M_t = M_{t−1}(y_t). 공식 코드 부재로 코드 대조는 불가. -->`

### ch12:297 — CONFIRMED
- **대상 주장**: "Table 5가 400M 설정이라는 명시가 원문에 없다(수치 일치는 4행 모두 확인됨)."
- **원전** [2501.00663 §5.9 + Table 5]: §5.9 "Ablation Study" 전문과 Table 5 캡션("Ablation Study on Titans. All components of Titans are positively contributing to its performance.")을 재확인 — **모델 규모 언급 없음.** 수치(LMM 27.01/47.83/92.68, MAC 26.67 등)는 본문 [해설]의 Table 1 400M 평균 대조와 일치.
- **판단**: 본문의 부재 진술이 정확하고, 400M 귀속은 [해설]에 추론으로 명기돼 있으므로 처리 방식 그대로 유지.
- **조치**: 주석 치환 — `<!-- VERIFIED(2026-07-12): §5.9·Table 5 캡션에 규모 명시 없음(재확인). 400M 귀속은 Table 1 평균 일치에 기반한 추론으로 [해설]에 유지. -->`

## part2/ch13-miras.md

### ch13:332 — UNRESOLVABLE (위험도 높음; 헤지 문안 제공)
- **대상 주장**: outer optimizer의 종류(AdamW 여부)·batch size·스케줄이 원문에 명시돼 있는지. 표 13-3(ch13:283)은 "표준 pre-training"·"outer η (상수 스케줄, 예: peak 1.5e-3)"을 기재.
- **원전 확인** [2504.13173 — **PDF p.25–26 (App. B–C) 직접 확인 완료**]: App. C "Experimental Setup"의 전체 내용은 (i) 과제·데이터셋 나열, (ii) baseline 나열, (iii) **Table 5 "Architectural Details"(Model 170M/340M/780M × Block/Dim/Head/Peak LR/Token)**가 전부다. **optimizer 종류(AdamW 등)·batch size·LR 스케줄(warmup/decay)은 논문 어디에도 없다.** §6 Setup은 context 4096·데이터셋·모델 크기·token 수만 제공. "Peak LR"이라는 열 이름이 스케줄의 존재를 시사하지만 스케줄 자체는 미명세. (대조: 같은 저자 라인의 Atlas는 App. E에 "AdamW, lr 4e-4, cosine annealing, batch 0.5M tokens, weight decay 0.1"을 명시 — Miras에는 이런 문장이 없다.)
- **판단**: "AdamW인지"는 원전으로 확인 불가. 표 13-3의 "표준 pre-training"은 무해하나, optimizer를 특정하는 인상을 주지 않도록 헤지 병기 필요. "peak 1.5e-3"은 Table 5 소스 있음(340M 행).
- **조치**: 표 13-3의 두 칸을 다음으로 교체 —
  - "update rule" 행 outer 칸: "표준 pre-training (unrolled inner loop 관통 backprop; **optimizer 종류·batch·스케줄은 원문 미명시** [Miras App. C — Table 5 외 훈련 레시피 부재])"
  - "learning rate" 행 outer 칸: "outer η — 원문은 **peak LR만** 명시(3e-3/1.5e-3/1.25e-3 [Miras Table 5]); 스케줄·optimizer 미명시"
  §13.6(ch13:330)의 설정 문단 끝에 한 문장 추가 — "같은 저자 라인의 Atlas가 AdamW·cosine·batch 0.5M tokens를 명시하는 것과 달리 [Atlas App. E], Miras에는 outer optimizer의 종류·batch·스케줄이 없다 — 재현 시도자는 이 부재를 알고 시작해야 한다." 주석 치환: `<!-- VERIFIED(2026-07-12): PDF App.C(p.26) 직접 확인 — Table 5(Peak LR 포함) 외 optimizer/batch/스케줄 부재. AdamW 여부 확인 불가 → 헤지 처리. -->`

## part2/ch14-atlas.md

### ch14:301 — CONFIRMED
- **대상 주장**: 구현된 polynomial 차수 p, PolySketchFormer식 sketch 사용 여부·차원, memory의 head 분할이 원문 어디에도 없는지 최종 확인 (+ PDF Fig. 3 라벨 재확인).
- **원전 확인** [2505.23735 전수 검색 + PDF p.13 Fig. 3 직접 확인]:
  - "degree" 7회 전부 φ_p의 추상 정의·Proposition 2·증명(App. C 모노미얼 계산)에만 등장 — **구현 p 값 없음**.
  - "sketch/PolySketch" 5회 전부 Table 1 카탈로그(PolySketchFormer 행)와 참고문헌 — **자기 구현에 sketch를 썼다는 문장 없음**.
  - App. E: 훈련 레시피(AdamW 4e-4, cosine, batch 0.5M tok, wd 0.1)와 memory 구조("MLP with 2 layers with expansion factor of 4 and GELU … residual connections and layer norm at the end of each chunk: M(x) = x + W₁σ(W₂x)")는 있으나 **memory의 head 분할·p·sketch 차원 없음**.
  - **PDF p.13 Figure 3** "ATLAS Layer Block Design" 라벨 재확인: k/q/v projection + Conv + γ/η/α gate + Norm 블록만 표기 — **수치화된 p·sketch 차원 라벨 없음.**
- **판단**: 본문(ch14:299 [해설])의 "구현된 차수 p와 sketch 차원이 공개되지 않은 한 state 크기는 계산할 수 없다"가 최종 확정됨.
- **조치**: 주석 치환 — `<!-- VERIFIED(2026-07-12): p·sketch·memory head 분할 전부 원문 부재 확정(본문·App.C·App.E·Fig.3 라벨 전수 확인). App.E의 memory 구조(2층·4×·GELU·chunk 끝 LN)는 확보. -->`

## part2/ch15-tnt.md

### ch15:258 — CONFIRMED
- **대상 주장**: Table 3의 "Language Modeling ppl" 열이 어느 corpus 기준인지 — 열 머리로는 미상, 값 일치로 C4 추정.
- **원전 확인** [2511.07343 — **PDF p.9 직접 확인**]: Table 3의 열 머리는 "**Language Modeling ppl ↓**"뿐 — corpus 표기 없음(캡션에도 없음). 동일 페이지 Table 2와 대조하면 Table 3의 값들이 **Table 2의 C4 열과 전부 일치**: Base(Titans C=256) 23.53=C4 23.53; +1 Local 21.04=Stage1 {8} C4 21.04; +2 20.74=Stage1 {8,16} C4 20.74; +3 20.47=Stage1 {4,8,16} C4 20.47; +4 20.15=Stage1 {4,8,16,32} C4 20.15; w Stage 2 20.86=Stage2 {1} C4 20.86 — **6개 값 전부**(FineWeb·PG19·Avg 열과는 불일치).
- **판단**: 본문의 헤지("C4로 추정되나 단정 불가")는 정직하나, 이제 6/6 수치 대조가 완료됐으므로 "열 머리는 무표기이나 값은 Table 2의 C4 열과 전부 일치(6/6) — C4 기준으로 읽는 것이 유일하게 정합적"으로 강화 가능.
- **조치**: 본문 문장을 다음으로 교체(선택) — "Table 3의 ppl 열은 열 머리에 corpus 표기가 없으나, 여섯 값 전부가 Table 2의 C4 열과 일치한다(23.53/21.04/20.74/20.47/20.15/20.86) — C4 기준으로 읽는다." 주석 치환: `<!-- VERIFIED(2026-07-12): PDF p.9 — Table 3 열 머리 무표기 확인; Table 2 C4 열과 6/6 일치(다른 열과 불일치) → C4 확정적 독해. -->`

## part2/ch16-nested-learning.md

### ch16:297 — CONFIRMED (위험도 높음)
- **대상 주장**: deep memory에서 (α_t I − η_t k_t k_t^⊤) 우측 곱의 적용 대상이 [NL Eq. 88]에 미명시라는 본문 진술.
- **원전 확인** [2512.24695 §8.1–8.2 정독]:
  - Eq. 88 (chunkwise판 Eq. 90 동일): "M_{□,t} = M_{□,t−1}(α_t I − η_t k_t k_t^⊤) − η_t ∇L(M_{□,t−1}; k_t, v̂_{□,t}), □ ∈ {k,v,q,η,α,memory}" — 여기서 M_□는 Eq. 89/91에 의해 "**M_□(·) = (·) + W_{□,1} σ(W_{□,2}(·))**" (2층 residual MLP). 함수 M에 행렬을 우측 곱하는 표기가 **어느 weight 행렬(W_{□,1}? W_{□,2}? 둘 다?)에 어떻게 작용하는지 §8.1–8.2 어디에도 정의가 없다.** "the architecture of the memories are arbitrary"라는 문장만 반복.
  - 타입이 닫히는 것은 행렬 memory 특수 사례뿐: Eq. 92 (dot-product) / Eq. 93 (ℓ2) — 본문 진술과 일치.
  - 공개 공식 구현: 확인 시점 기준 부재 — 코드 대조 불가.
- **판단**: 본문의 "원문이 명시하지 않는다"가 정확. 재구현 관건이라는 위험 평가도 유지.
- **조치**: 주석 치환 — `<!-- VERIFIED(2026-07-12): §8.1–8.2 정독 — Eq.88/90의 우측 곱은 M이 2층 residual MLP(Eq.89/91)일 때 적용 대상 미정의(원문 그대로의 공백). Eq.92–93(행렬 memory)만 타입 명확. 공식 구현 부재로 코드 대조 불가. -->`

### ch16:394 — CONFIRMED (위험도 높음)
- **대상 주장**: Hope의 inner rule에 momentum 항이 실제 포함되는지 — 식에는 없고 Table 6 ablation에는 'w/o Momentum' 행이 있다는 간극.
- **원전 확인** [2512.24695 §9.6 + Table 6]:
  - §9.6 원문: "(1) The first row, replaces Delta Gradient Descent with a simple gradient descent in the design of self-modifying Titans; (2) **The second row removes the momentum term in the self-modifying Titans**; (3) removes the weight decay; …" — 즉 **배포된 self-modifying Titans(Hope의 inner rule)에는 제거 가능한 momentum 항이 실재**함을 §9.6 산문이 직접 확인.
  - Table 6: Hope 12.24/58.1 vs w/o Momentum 13.58/56.9.
  - 그러나 §8의 수식(Eq. 88/90/92–93)에는 momentum 항이 등장하지 않음 — 수식-구현 간극은 원문이 해소하지 않음(공식 구현 부재).
- **판단**: 본문의 서술("ablation 행이 momentum 항의 존재를 시사 … 간극은 원문이 명시적으로 해소하지 않는다")이 §9.6 문장으로 한 단계 더 강해진다: '시사'가 아니라 §9.6이 **명시**("removes the momentum term")하되, 수식에는 여전히 없다.
- **조치**: ch16:393의 "…행(13.58 vs 12.24 [NL Table 6])은 배포된 inner rule이 식 (16-4)에 명시되지 않은 Titans식 momentum 항 S_t(→ 12장)도 지니고 있음을 시사한다"를 "…행(13.58 vs 12.24 [NL Table 6])과 §9.6의 산문('removes the **momentum term** in the self-modifying Titans' [NL §9.6])은 배포된 inner rule이 식 (16-4)에 등장하지 않는 Titans식 momentum 항 S_t(→ 12장)를 실제로 지니고 있음을 말해 준다 — 단 그 정확한 수식 위치는 원문 어디에도 없다"로 교체. 주석 치환: `<!-- VERIFIED(2026-07-12): §9.6이 momentum 항 존재를 명시(ablation 행 설명), 수식(Eq.88/90)에는 부재 — 간극 자체가 원문 사실. 공식 구현 부재. -->`

## part2/ch17-sleep.md

### ch17:9 — CORRECTED
- **대상 주장**: 네 항목 중 (2)(3)(4)를 "[NL]의 자체 결산"으로 귀속. ((1) CF 지연은 [Sleep §2.2] 재인용으로 기확인.)
- **원전 확인** [2512.24695 검색: "offline"/"replay"/"capacity"/"future"]:
  - **(2) offline consolidation 범위 밖 — 확인됨(원문 인용 확보)**: NL은 online/offline 두 consolidation을 구분한 뒤 명시적으로 선을 긋는다 — "although the second stage [offline/systems consolidation] is equally, or even more, crucial for the consolidation of memories, and its absence can damage the process and might cause loss of memory …, **in this work, we focus on the first stage: memory consolidation as an online process.**" [NL §2, 신경과학 배경 논의]. → "자체 결산" 귀속 정당.
  - **(4) 모든 적응이 입력이 흐르는 동안 — 부분 확인**: 위 인용의 직접 따름정리(온라인 과정에만 집중한다는 선언 = 입력을 끊는 시간의 부재)이나, NL이 이를 별도의 한계 항목으로 명시한 문장은 없음 → "자체 결산의 따름정리"로 표기 조정.
  - **(3) capacity 고정 — NL 자체 결산으로는 미확인**: "capacity가 고정"을 NL이 자기 한계로 선언하는 문장은 검색되지 않음(관련 문장은 "brain has (large but) fixed capacity and new components are not added over time"으로, 한계 선언이 아니라 Sleep 쪽 설계 정당화 문맥). 이 진단은 [Sleep §1]의 online-only consolidation 비판(capacity 재소모 등)에서 온 것 → 귀속을 [Sleep]으로 옮기거나 헤지.
- **조치**: ch17:7의 "그러나 [NL]의 자체 결산으로도 네 가지가 남았다"를 다음으로 교체 —
  > "그러나 [NL]이 남긴 것이 네 가지다. 첫째, … — [Sleep §2.2]가 [NL]을 인용해 이 진단을 그대로 재확인한다. 둘째, **offline consolidation — replay·sleep 계열의 기제 — 는 [NL] 스스로 명시적으로 범위 밖에 두었다**: '두 번째 단계가 동등하게, 또는 그 이상으로 중요함에도, 이 작업은 첫 단계 — online 과정으로서의 memory consolidation — 에 집중한다' [NL §2]. 셋째, **capacity가 고정이다** — 유한 크기 파라미터로의 압축인 이상 새 지식은 언젠가 옛 지식을 덮어쓴다는 이 진단은 [Sleep §1]이 online-only consolidation에 들이대는 비판이다. 넷째, **모든 적응이 입력이 흐르는 동안 일어난다** — 둘째 항목의 직접적 따름정리로, 모델이 입력을 끊고 자기 내부를 정리하는 시간이 [NL]의 설계에는 존재하지 않는다."
  주석 치환: `<!-- VERIFIED(2026-07-12): (2)=NL §2 직접 인용("in this work, we focus on the first stage: memory consolidation as an online process"); (4)=그 따름정리로 재표기; (3)=NL 자체 결산 아님 → [Sleep §1] 비판으로 귀속 이전. -->`

### ch17:202 — CORRECTED (위험도 높음)
- **대상 주장**: Sleep의 Hope 계열 실험(§4.1)에서 쓰인 Hope가 NL 레시피로 pre-train된 checkpoint인지(regime 1이 식 (17-1)을 관통해 수행되었는지) — 본문은 "원문에 명시가 없다"로 헤지.
- **원전 확인** [2606.03979 §4.1 + App. B]:
  - §4.1 (continual learning, 텍스트 분류): "**We use Llama-3B and Llama3-8B (Dubey et al. 2024) as the backbones.** Hope is augmented with our memory consolidation mechanism…" — Hope baseline도 같은 비교군에 포함("We also include Hope (Behrouz et al. 2025) as a multi-level in-context updating baseline without explicit distillation process").
  - App. B: "In all of our experiments, we follow the settings of the original benchmark … **In our design, we use 5 MLP blocks with dimension 64 as the additional parameters and keep the active parameter count unchanged (i.e., the same as the base model, which is either 8B or 3B).**"
  - 즉 Sleep의 'Hope' 실험은 **NL 레시피로 pre-train한 Hope checkpoint가 아니라, pre-trained Llama-3B/8B 위에 소형 memory 블록(5×dim-64 MLP)을 graft한 구성**이다. 다른 실험들(SQuAD knowledge incorporation은 SEAL 셋업 추종, ARC는 Llama-3.2-1B backbone)도 전부 기성 checkpoint 기반.
- **판단**: "명시가 없다"보다 강한 결론이 원문에서 직접 나온다 — regime 1(식 17-1 관통 pre-train)은 이 실험들에서 **수행되지 않았다**. §17.4의 regime 1 서술은 "Hope 계열이라면 이렇게 된다"는 구성적 서술로 재표시 필요.
- **조치**: ch17:202 주석 자리의 본문(§17.4 관련 서술)에 다음을 반영 —
  > "확인된 사실: [Sleep]의 Hope 계열 실험은 NL 레시피로 pre-train된 Hope checkpoint가 아니라 **pre-trained Llama-3B/8B에 5개의 dim-64 MLP memory 블록을 얹은 graft 구성**이다 [Sleep §4.1('We use Llama-3B and Llama3-8B as the backbones'), App. B]. 따라서 regime 1 — 식 (17-1)을 관통하는 sleep-aware pre-training — 은 이 논문의 실험 어디에서도 실행되지 않았다; §17.4의 regime 1은 논문이 정의한 가능성이지 실증된 경로가 아니다."
  주석 치환: `<!-- VERIFIED(2026-07-12): Sleep §4.1·App.B — Hope 실험 = Llama-3B/8B graft(+5×dim-64 MLP, active param 불변). NL-레시피 checkpoint 아님 → regime 1 미실증으로 본문 확정. -->`

---

## 부록 A — 원전 확보 내역

| arXiv ID | 논문 | 용도 (해소 항목) |
|---|---|---|
| 2407.04620 | TTT (Sun et al. 2024) | ch01:19, ch08:42/47/65/93/118, ch09:120 (7건) |
| 2503.14456 | RWKV-7 | ch06:167, ch06:173 |
| 2407.14207 | Longhorn | ch06:138 |
| 2406.06484 | Parallelizing DeltaNet (Yang et al.) | ch06:117, ch09:108 |
| 2102.11174 | FWP (Schlag et al.) | ch06:117 (보조) |
| 2312.06635 | GLA | ch06:44, ch09:73 |
| 2307.08621 | RetNet | ch06:44 |
| 2312.00752 | Mamba | ch07:56 |
| 2405.21060 | Mamba-2 | ch07:114 |
| 2212.07677 | von Oswald et al. | ch04:129 |
| 2008.02217 | Ramsauer et al. | ch05:90 (+ch05:74 보조) |
| 1606.01164 | Krotov & Hopfield | ch05:84 |
| 2504.05298 | TTT video (Dalal et al.) | ch08:123 |
| (웹) | NVIDIA H100 datasheet | ch10:140 |
| (웹) | Amit et al. 1985 서지 | ch05:74 |
| (로컬) | 2501.00663 / 2504.13173 / 2505.23735 / 2511.07343 / 2512.24695 / 2606.03979 | 나머지 로컬 13건 |

## 부록 B — TODO 범위 밖에서 발견된 수정 대상 (fixer 참고)

1. **ch13:291** — "[Miras §5.4]" → **"[Miras §5.3]"** (§5.4는 존재하지 않음; "Parallelizable Training"은 §5.3 내부 단락 — ch09:176 해소 중 확인).
2. **Sleep 서지 [90]/[91] 중복** — 책이 Sleep의 참고문헌 번호를 인용할 일이 있으면 주의.
3. **Titans의 RWKV-7 인용 표기 "(Peng 2021)"** — 원문 자체의 서지 슬립(RWKV-LM repo를 가리킴). 책에서 Titans App. C를 재인용할 때 "RWKV-7 (arXiv:2503.14456)"로 바로잡아 인용하는 것을 권고.
