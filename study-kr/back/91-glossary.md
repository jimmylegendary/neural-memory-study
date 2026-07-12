# 용어집 색인 (Glossary Index)

> **구성** — STYLE §2.4 concept ledger(공식 명칭 확정표, ~57행) + §1.2 예약 기호 전역표 + 각 장 첫 정의를 종합했다. 각 항목: **용어(영어 원어)** → 한 줄 정의 + **소유 장**(그 개념을 정의하는 단일 소유권, STYLE §2.3). 영어 원어 유지 원칙(STYLE §2.1)에 따라 표제어는 로마자.
> **정렬** — §2는 알파벳순. §1은 inference 엔지니어 어휘와의 Rosetta 대응(STYLE §3). §3은 예약 기호.
> **사용법** — "소유 장"은 그 개념의 정의를 바꿀 수 있는 유일한 장이다. 다른 장은 참조만 한다(재정의 금지).

---

## 1. Rosetta 대응 (inference 어휘 ↔ 이 라인의 어휘)

독자(transformer inference 엔지니어)의 모국어를 이 라인의 어휘로 옮기는 교차 색인. **대응의 성격**이 핵심 — 동일/유비/차이가 논점을 구분하지 않으면 잘못된 직관이 이식된다(STYLE §3). ch01이 이 표를 본문으로 확장한다.

| inference 세계 (독자 어휘) | 이 라인의 어휘 | 대응의 성격 | 소유 |
|---|---|---|---|
| KV cache | non-parametric memory state; softmax attention = capacity 무한($\phi^*$) associative memory | **동일 대상의 재서술** — attention은 압축하지 않는 memory | ch01, ch06 |
| KV cache append | memory **write** (outer-product Hebbian이 최근접; delta rule은 append 아닌 **overwrite**) | 유비 + 차이 | ch01, ch05 |
| attention lookup (q·K) | memory **read** $y_t=\mathcal{M}(q_t;W_t)$ | 동일 추상화 (k→v 연상) | ch01 |
| linear-RNN state ($d\times d$) | matrix memory = 고정 크기 lossy 압축 memory | 동일 | ch06 |
| prefill | 큰 chunk의 병렬 write (compression); TNT에선 global memory | 유비(정확) | ch09, ch15 |
| decode | $C=1$의 per-token online write + read — **backward pass가 decode에 들어온다** | 유비(정확) + 신세계 | ch09 |
| FlashAttention tiling | chunkwise-parallel training의 chunk | ⚠ **차이가 논점**: tiling은 bit-exact, chunk는 함수를 바꾸는 semantic 근사 (M4) | ch09 |
| GEMM shape 감각 | $\nabla_W\ell$의 outer-product 구조: $dW=(\text{오차})k^\top$는 rank-$C$ GEMM | 동일 (이 책의 교수법) | ch01, ch10 |
| scan / prefix-sum kernel | momentum의 associative scan (S5식), $\Pi_t$의 누적 합 | 동일 | ch07, ch09 |
| roofline / arithmetic intensity | chunk 크기 $C$가 arithmetic intensity를 결정; 품질-최적 $C$(작음) vs MFU-최적 $C$(큼)의 긴장 | 동일 도구, 새 독립변수 | ch10 |
| batching (shared weights 전제) | per-request fast-weight state가 shared-weight batching을 **깨뜨림** → grouped-GEMM decode | ⚠ 차이가 논점 | ch10, ch23 |
| paged KV cache / session cache | per-session weight state — 새 cache class(sizing·checkpoint·eviction) | 유비 → Part III | ch23 |
| cache eviction policy | retention gate = **학습된** eviction | 유비(정확) | ch13 |
| speculative decoding rollback | memory state snapshot/rollback (optimizer-trajectory 포함) | 유비 + 미해결 | ch23 |
| optimizer state (경험 없음) | 훈련의 register file / accumulator: $m_t, h_t$ | 도입용 유비 | ch02 |
| weight update 없음(추론 불변식) | **폐기되는 불변식** — 이 라인의 정의적 특징 | 차이 그 자체 | ch01, ch08 |
| sequence 축 | 훈련의 batch 축을 **sequence 축이 대신**한다(inner loop의 mini-batch = chunk) | ⚠ 최대 혼동 지점 | ch02, ch09 |
| distributed training (경험 없음) | context parallelism: reset이 sequential chain을 끊어 shard 병렬화 | 도입용 유비 | ch15 |
| 서빙 fleet의 백그라운드 job | sleep phase = 주기적 offline consolidation/dreaming ("weights의 background compaction") | 유비 → ch17, Part III | ch17 |

---

## 2. 용어집 (알파벳순)

### A
- **arithmetic intensity** — FLOP/byte 비. 이 라인에선 chunk 크기 $C$가 결정하는 새 독립변수. (Rosetta: 독자의 모국어) — **ch10**
- **associative memory** — key→value 연상을 저장·회수하는 memory. 고전 정식화 + Miras의 attentional-bias Def. (Rosetta: KV cache의 read 추상화) — **ch05**(고전), **ch13**(Def. 정식화)
- **attentional bias** — inner objective $\ell$가 이루는 family; memory가 무엇을 "잘 기억하려 하는가"의 축. — **ch13**
- **Atlas** — windowed inner objective(Omega rule) + polynomial feature map + Muon optimizer로 memory를 확장한 논문/모델(OmegaNet 계열). — **ch14** (arXiv:2505.23735)

### B
- **backward pass** — outer loss의 gradient를 얻는 역전파. 이 라인에선 inner loop의 backward가 **decode 안으로** 들어온다. (Rosetta: 독자에게 없던 개념) — **ch02**
- **bilevel optimization** — inner(memory)·outer(parameter) 두 최적화가 중첩된 구조; meta-learning의 형식. — **ch04**

### C
- **catastrophic forgetting** — 새 데이터 학습이 이전 지식을 덮어쓰는 현상. — **ch11**
- **chunk 크기 $C$** — 병렬화 knob이자(TNT 이후) semantic knob. window 크기 $c$와 반드시 구분. — **ch09**(명제화)
- **chunk 크기 = semantic hyperparameter** — $C$는 스케줄이 아니라 계산되는 함수 자체를 바꾼다(FlashAttention tiling의 bit-exact와 결정적 차이). — **ch09** (발견: TNT)
- **chunkwise-parallel training** — sequence를 크기 $C$ chunk로 잘라 chunk 내부는 병렬(GEMM-rich), chunk 사이는 순차/scan으로 재조직하는 훈련 기법의 총칭(축약 "chunkwise training"). — **ch09**
- **chunk-size mismatch (train/serve)** — 훈련 chunk 크기와 서빙 chunk 크기가 달라 생기는 품질/효율 괴리. — **ch15**
- **complementary learning systems (CLS)** — 빠른(해마)·느린(신피질) 두 학습계의 상보 이론; consolidation의 신경과학 근거. — **ch11**
- **context parallelism** — periodic state reset이 sequential chain을 끊어 shard 단위 병렬화를 가능케 함(TNT). — **ch15**
- **contextual memory** — 문맥(입력 sequence)에서 test-time에 채워지는 memory. persistent memory와 대비. — **ch12**
- **Continuum Memory System (CMS)** — level별 update 주기(frequency)를 가진 FFN chain; 여러 시간척도의 memory 연속체 (식 (M5)). — **ch16**
- **crosstalk** — associative memory에서 저장 패턴 간 간섭; capacity 한계의 원인. — **ch05**

### D
- **deep memory** — memory 함수 $\mathcal{M}$가 (linear가 아니라) 다층 MLP인 것. 표준형 = 2-layer residual MLP(expansion 4, GELU). — **ch12**
- **delta rule** — $W\leftarrow W-\eta\nabla\ell$ 형태의 오차-기반 갱신(Widrow–Hoff LMS). 모델 DeltaNet과 구분(rule vs 모델). — **ch05**
- **DeltaNet / Gated DeltaNet (GDN)** — delta rule을 sequence layer로 구현한 linear-memory 모델; GDN은 여기에 retention gate 추가. — **ch06**
- **dual form (TTT)** — inner GD 갱신을 chunk 단위 matmul로 재작성한 병렬 형태(Sun et al.). 일반화는 ch09. — **ch08**(일반화 **ch09**)
- **Dreaming** — self-generated rollout로 자기 update를 개선하는 sleep 2단계 중 self-improvement loop. — **ch17**
- **DGD / GGD / GM / Delta Momentum / DMGD / M3** — NL이 도입한 inner-optimizer 변형군(Deep/Gradient/... Gradient Descent 등). 첫 등장 시 정의. — **ch16**

### F
- **fast weights** — inner loop 상태 = memory 상태 $W_t$. 표기 규약상 책 전체에서 $W$. — **ch06** (Schmidhuber/FWP 계보)
- **fast weight programming (FWP)** — slow net이 fast net의 weights를 만들어내는(프로그래밍) 관점(Schlag et al.). — **ch06**
- **FTRL viewpoint** — Follow-the-Regularized-Leader로 memory 갱신을 보는 Miras의 관점. FTRL 자체는 ch03 소유. — **ch13**(관점), **ch03**(FTRL)
- **forget gate** → **retention gate**의 역사적 별칭. 첫 등장 시 1회 병기만 허용. — **ch13**

### H
- **hardware lottery** — 알고리즘의 성공이 당대 하드웨어와의 궁합에 좌우된다는 명제(Hooker 2020). — **ch21**
- **hierarchical memory (global/local)** — 큰 chunk의 global module + 병렬 local module로 나눈 TNT의 memory 구조. — **ch15**
- **Hope / Hope-Attention** — self-modifying Titans + CMS를 종합한 NL의 대표 아키텍처. — **ch16**
- **Hopfield network** — energy 최소화로 패턴을 저장·회상하는 고전 associative memory. — **ch05**

### I
- **inner loop** — memory 상태 $W_t$를 sequence를 따라 갱신하는 test-time 최적화(inner loss $\ell$). — **ch01**(비형식), **ch04**(형식화)
- **inner loss $\ell$ / outer loss $\mathcal{L}$** — 소문자=inner(per-token memorization), 대문자=outer(task, next-token). 두 loop의 시각적 구분. — **ch01/ch04**

### K
- **knowledge distillation / GKD** — teacher의 출력 분포를 student가 모사하도록 학습; GKD는 on-policy 판본. — **ch11**
- **Knowledge Seeding (KS) / Self-Knowledge Seeding (SKS)** — 작은 self-memory에서 더 큰 network로의 **upward distillation**(Sleep의 consolidation). — **ch17**
- **knowledge-transfer taxonomy (5 mechanisms)** — NL이 정리한 지식 이전 5가지 메커니즘 분류. — **ch16**

### L
- **Learning–Retaining viewpoint** — memory 갱신을 "학습 항 + 유지 항"으로 분해하는 Miras의 관점. — **ch13**
- **Learning to Imitate (LTI)** — Sleep의 imitation-learning 항. ⚠ ch07의 LTI(linear time-invariant)와 무관한 동음 약어 — **ch17 내부 전용**. — **ch17**
- **level / update frequency** — CMS에서 각 memory 성분이 갱신되는 주기; 시간척도 계층. — **ch16** (Sleep 재사용)
- **linear attention** — softmax 없는 $W_t=W_{t-1}+v_tk_t^\top$ 형태; KV-cache 압축 관점. — **ch06**
- **Local Surprise Signal (LSS)** — $u_t=\nabla_{y_t}\mathcal{L}$, layer 출력에 대한 outer-loss gradient(NL). — **ch16**

### M
- **MAC / MAG / MAL** — Memory as Context / Gate / Layer. Titans의 memory 통합 3방식. 첫 등장 시 풀네임 1회 병기. — **ch12**
- **master update (M)** — 이 책의 기준 수식: $S_t=\beta_tS_{t-1}-\eta_t\nabla_W\ell$, $W_t=\alpha_tW_{t-1}+S_t$. 6편은 (M)의 성분 교체로 서술된다. — **ch09/ch12**
- **memory capacity (formal)** — 유한 memory가 저장·회상할 수 있는 쌍의 수의 형식적 한계(Atlas). 고전(Hopfield) capacity는 ch05. — **ch14**
- **meta-learned initial state $W_{\mathrm{init}}$** — outer loop가 학습하는 memory 초기값($W_0$); TNT에서 load-bearing. — **ch04**(개념), **ch15**(역할)
- **meta-learning / MAML** — 학습 알고리즘 자체를 학습; MAML은 초기값을 meta-학습. — **ch04**
- **momentary surprise** — $g_t^{\mathrm{in}}=\nabla_W\ell(W_{t-1};k_t,v_t)$, 현재 token의 inner gradient. (~~놀라움~~ 금지) — **ch12**
- **momentum-as-memory** — momentum buffer $S_t$ 자체가 "과거 surprise를 담은 memory"라는 관점. — **ch02**(객체로서), **ch16**(정리로서)
- **Muon / Newton–Schulz** — semi-orthogonalization optimizer; NS-$\kappa$($\kappa=5$) 반복으로 특이값을 1 근방으로. — **ch02**(객체), **ch14**(inner 사용)

### N
- **Nested Learning (NL)** — 아키텍처를 중첩된 다-level 최적화로 재해석한 논문; CMS·self-modifying·Hope를 종합. — **ch16** (arXiv:2512.24695)
- **NSAM / Neural Learning Module** — NL의 self-modifying memory·학습 모듈 고유명사. — **ch16**

### O
- **Omega rule** — sliding-window 안의 여러 token에 대한 inner objective(window gate $\gamma_{t,i}$ 포함); Atlas의 핵심 (식 (M3)). "Ω-rule" 표기 금지. — **ch14**
- **online vs offline consolidation** — 스트림 중(online) vs 별도 sleep phase(offline)의 memory 통합 구분. — **ch11**(개념), **ch16/ch17**(기법)
- **outer loop** — slow weights $\Theta$를 task loss $\mathcal{L}$로 학습하는 통상적 훈련. — **ch01/ch04**

### P
- **parameter expansion (periodic (de)activation)** — sleep 중 파라미터를 주기적으로 활성/비활성해 capacity를 늘리는 Sleep 기법. — **ch17**
- **periodic state reset** — local memory를 $W_{\mathrm{init}}$로 주기적으로 되돌려 chain을 끊는 것(TNT). — **ch15**
- **persistent memory** — 입력과 무관하게 학습되는 고정 memory token $P$(Titans). — **ch12**

### Q
- **Q-K projection $\Pi_t$** — $\sum_\tau k_\tau k_\tau^\top/\|k_\tau\|^2$, TNT의 누적 projection 행렬(원문 $\mathcal{M}_t$ 개명). — **ch15**

### R
- **regret / FTRL / OGD** — online learning의 성능 척도(comparator $W^\star$ 대비)와 대표 알고리즘. — **ch03**
- **retention gate $\alpha_t$** — memory에서 **남기는 비율**($\alpha_tW_{t-1}$, 1=전부 유지). 공식 용어(forget gate는 별칭). Titans와 방향 반대. — **ch13**
- **roofline / MFU** — 성능 상한 모델·model FLOPs utilization. 독자의 모국어(정의 불필요, 표기만 통일). — **ch10**

### S
- **self-modifying Titans** — memory가 자기 update 규칙을 스스로 생성·수정하는 구조(SRWM 계보). — **ch16**
- **slow weights $\Theta$** — outer loop parameter 전체(projection·gate-producer·backbone). fast weights $W$와 대문자 구분. — **ch06/ch16**
- **SSD (state-space duality)** — SSM과 attention의 이중성(Mamba-2); chunkwise 계산의 형식적 근거. — **ch07**
- **stale-snapshot 근사 (chunk-start anchor)** — chunk 안 모든 gradient를 chunk 시작 상태 $W_{\xi(t,C)}$에서 평가하는 근사("gradient anchor"). — **ch09**
- **surprise (momentary / past)** — momentary $=g_t^{\mathrm{in}}$, past $=S_t$(momentum buffer). Titans 기원. — **ch12**
- **synaptic-pruning reset** — sleep 중 불필요 연결을 잘라내는 reset(Sleep). — **ch17**

### T
- **test-time memorization** — Atlas 이후 용어: test-time에 문맥을 학습(learning)이 아니라 **기억(memorization)**한다는 구분 자체가 논점. — **ch14**
- **test-time training (TTT)** — test 입력에 대해 보조 loss로 weights를 갱신하는 계열의 총칭("TTT 계열"). — **ch08**
- **Titans** — deep neural memory + momentum + retention을 test-time에 갱신하는 라인의 기준점 모델(LMM). — **ch12** (arXiv:2501.00663)
- **two-stage training (train-big / serve-small)** — 큰 chunk로 효율 훈련 후 작은 chunk로 전환하는 TNT의 훈련 경제학. — **ch15**

### W
- **wake/sleep lifecycle** — 스트림 처리(wake)와 주기적 offline consolidation/dreaming(sleep)을 오가는 lifecycle. — **ch17**
- **weight decay = per-token retention** — Titans 문맥에서 weight decay를 매 token의 retention("forgetting mechanism")으로 해석. — **ch12**
- **window gate $\gamma_{t,i}$** — Omega rule의 in-context pruning gate(어느 과거 token을 objective에 넣을지). — **ch14**

---

## 3. 예약 기호 색인 (STYLE §1.2 요약)

| 기호 | 의미 | 소유/정의 |
|---|---|---|
| $W_t$ | fast weights = memory 상태 (inner loop) | §1.2 |
| $\Theta$ | slow weights 전체 (outer loop) | §1.2 |
| $W_{\mathrm{init}}$ | meta-learn된 초기 memory 상태 ($W_0$) | ch04/ch15 |
| $\mathcal{M}(\cdot;W)$ | memory module (함수). $\mathcal{M}^\star$=argmin 최적해(read-only 의미 금지) | §1.2 |
| $\ell(W;k,v)$ / $\mathcal{L}$ | inner(per-token) / outer(task) loss | §1.2 |
| $\eta_t$ / $\eta$ | inner learning rate(gate) / outer lr(상수) | §1.2 |
| $\beta_t$ | inner momentum decay (gate) | §1.2 |
| $\alpha_t\in[0,1]$ | retention gate (남기는 비율) | ch13 |
| $S_t$ | inner momentum buffer (past surprise) | ch12 |
| $g_t^{\mathrm{in}}$ | momentary surprise $=\nabla_W\ell(W_{t-1};k_t,v_t)$ | ch12 |
| $u_t$ | Local Surprise Signal $=\nabla_{y_t}\mathcal{L}$ | ch16 |
| $C$ / $c$ | chunk 크기(병렬화·semantic knob) / Omega window 길이 | ch09 / ch14 |
| $\xi(t,C)$ | chunk 시작 offset $=C\lfloor(t-1)/C\rfloor$ | ch15 |
| $W^{\mathrm{g}}, W^{\mathrm{l}(i)}$ | TNT global / local memory 상태 | ch15 |
| $\Pi_t$ | TNT Q-K projection 행렬 | ch15 |
| $\gamma_{t,i}$ | Omega rule window gate | ch14 |
| $\delta_t$ | Huber threshold (learned) — layer 첨자 $\delta_\ell$(backprop 오차)와 구분 | ch13 |
| $\mathrm{NS}_\kappa$ | Newton–Schulz $\kappa$회 반복($\kappa=5$ 표준) | ch14 |
| $\phi_p, \phi^*$ | polynomial / exponential feature map | ch14 |
| $\theta^{(\ell)}$, $f_\ell$, $C^{(\ell)}$, $e_{i,\ell}$ | CMS의 level parameter·frequency·chunk·누적 error | ch16/ch17 |
| $m_t, h_t$ | outer 1차/2차 moment (Adam; $v_t$는 value와 충돌해 $h_t$) | ch02 |
| $\lambda$ | outer weight decay | §1.2 |
| $L$ / $L_{\mathrm{layer}}$ / $L_{\mathcal{M}}$ | sequence 길이 / 네트워크 깊이 / memory MLP 깊이 | §1.2 (v1.1) |
| $L_{\mathrm{s}}^{(i)}$ | TNT shard 길이(local reset 주기; 원문 $S_{L_i}$ 개명) | ch15 |
| $\rho, \lambda_{\mathrm{on}}, \lambda_{\mathrm{KD}}$ | Sleep의 reward 혼합 / GKD on-policy 비율 / distill-vs-reward 계수(원문 $\gamma,\lambda,\alpha$ 개명) | ch17 |

> **inference 어휘 Rosetta**(재수록): $W_t$≈per-session cache state, $\nabla_W\ell$의 outer-product≈rank-$C$ GEMM, $S_t$ associative scan≈prefix-sum kernel, retention gate $\alpha_t$≈학습된 cache eviction, chunk $C$≈arithmetic-intensity knob(≠bit-exact tiling). 상세는 §1.
