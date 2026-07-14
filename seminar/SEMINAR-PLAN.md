# 세미나 전면 개편 계획 (storyline 반영)

## 0. 의도 분석 (노트에서 추론 — 명시 안 됐지만 확실한 것)

QA 31건 + 세션 이력에서 읽히는 청중·저자 의도:
- **청중 = 학습(training) 경험 0인 inference/systems 엔지니어.** 그래서 Background를 밑바닥(2-layer MLP + head)부터, **shape·연산·화살표로** 눈에 보이게 시작해야 함(storyline 1의 요구).
- **저자(Jimmy) 관심 = "shape 수준 정확한 메커니즘 + 증거 등급(명시 vs 추론) + decode 비용."** QA가 전부 "이 weight의 shape은? token마다 뭐가 update? 논문에 근거 있나?"였음. → 슬라이드는 **정확·근거 중심**, 두루뭉술 금지.
- **framing 진화가 핵심 서사.** optimizer=memory, Miras 4축, Atlas Omega, TNT serving, HOPE wrap-up, Sleep offline. 개별 독립 + 연결 둘 다 보여야 함(storyline 논문설명 4).
- **시스템 모델링은 저자의 기여(Part III)** — 논문설명에선 배제하고 별도 섹션(storyline system modeling).
- 도형 직접 그림, 글/그림 안 겹침, 설명 포함.

## 1. 구성 (3부)

### A. Background (training 밑바닥 + optimizer 역사 + 대안 접근)
- **A1. training 기본** — 2-layer MLP + head DAG. 번호·shape·연산을 화살표 위아래로. forward(x→W1→σ→W2→head→logits) → **L2 loss(batch 누적)** → backward(∇L 체인) → **AdamW(m,v moment)** → weight update. code 함수 기준(loss.backward()/optimizer.step()) 각 단계에서 실제 일어나는 일.
- **A2. optimizer 역사** — GD→SGD→(Momentum)/(AdaGrad→RMSProp)→Adam→AdamW. 각 단계: **문제→다음이 뭘 어떻게 해결→수식**. Adam의 L2-weight-decay 문제 → AdamW decoupled decay.
- **A3. 대안 접근** — linear attention / DeltaNet→Gated DeltaNet / SSM·Mamba2. (논문들의 조상.)

### B. 논문설명 (6편; 핵심발명 먼저 → 구현 → 증명; framing 진화; 시스템모델링 배제)
각 논문: **(1) 핵심 발명/주장 먼저 → (2) 어떻게 구현 → (3) 어떻게 증명(논문 figure/table/실험)**. 논거 전개는 논문 흐름 따름. 쉽게.

### C. System modeling (저자 기여 — storyline 순서로)
pair thesis + 크로스아키텍처 + HOPE-block roofline + zHBM/PIM/SRAM scaling + scaling 청사진.

---

## 2. 논문별 핵심 발명/주장 listup + 논거 흐름 (반드시 전부 담김)

### Titans (2501.00663) — "test-time memorization"
- **핵심주장**: 문맥을 KV에 쌓지 말고 **test-time에 학습되는 deep neural memory**(2-layer MLP)에 압축. 3요소: **surprise(=gradient) + momentum + forgetting(decay)**.
- 논거흐름: linear/KV의 한계(용량·crosstalk) → 메모리를 "학습되는 MLP"로 → 무엇을 배우나? = **k→v 연상**(loss=‖M(k)−v‖²) → 얼마나? = surprise(gradient) + momentum(과거 surprise) + α(forget) → 병렬화? = momentum을 associative scan으로 → 어디에 꽂나? = **MAC/MAG/MAL**(persistent + core + contextual).
- 증명: 언어모델링·NIAH·BABILong figure/table.

### Miras (2504.13173) — "메모리는 online 최적화; 4축 설계공간"
- **핵심주장**: sequence layer = **하나의 online 최적화 문제**. 설계축 **4개**: attentional bias / retention / architecture / algorithm. Titans/DeltaNet 등은 이 축의 점들.
- 논거흐름: Titans 재해석(surprise=attentional bias, forget=retention) → 축을 일반화 → 새 인스턴스 Moneta/Yaad/Memora(FTRL·Learning-Retaining) → 발명이라기보다 **축의 존재 증명 + 실험**.
- 증명: 축별 ablation table.

### Atlas (2505.23735) — "Omega rule로 용량 확장 + Muon"
- **핵심주장 3**: (1) **Omega rule**(sliding window γ_{t,i}) (2) **feature map으로 용량 확장**(matrix √params → φ_p로 d^p; softmax=∞) (3) **GD 대신 Muon**.
- 논거흐름: 메모리 용량이 병목 → Hopfield 용량 이론(root) → feature map 투영으로 확장 → 최적화도 Muon(2차)로 → DeepTransformers/Dot.
- 증명: 용량 실험 + long-context table.

### TNT (2511.07343) — "training 레시피 + serving 구조"
- **핵심주장 4**: (1) train/serve **chunk mismatch**가 품질 붕괴(2.6배) (2) **train chunk(큰)/serve chunk(1) decouple + 2-stage** (3) **hierarchical global(prefill)/local(decode) memory** (4) **Q-K projection**.
- 논거흐름: deep memory를 큰 chunk로 싸게 훈련 → 그런데 serve chunk 다르면 붕괴 → 2-stage로 정렬 → 계층 메모리로 병렬+세밀 → serving 구조 정의.
- 증명: chunk sweep(V자), 속도(17배), 품질 table.

### NL / HOPE (2512.24695) — "전부 중첩된 메모리; 4편 wrap-up"
- **핵심주장 3**: (1) **Expressive Optimizers**(optimizer=gradient의 associative memory → DGD/M3) (2) **Self-Modifying Titans**(투영·게이트까지 memory, q만 static) (3) **CMS**(연속 주파수 = long/short 일반화). 합 = **HOPE**. **ICL은 창발 아니라 ≥2레벨의 structural.**
- **wrap-up (storyline 5)**: 앞 4편을 하나로 — momentum/Adam/Muon(=Miras/Atlas의 algorithm축)이 **memory**임을 보여 optimizer와 architecture를 **하나의 nested system**으로 통합; Titans의 frozen 투영을 self-modify로 풀고; long/short를 CMS 주파수 연속체로. → **HOPE는 "모든 부품이 어떤 주파수의 memory"인 아키텍처.**
- 논거흐름: backprop도 self-referential memory → optimizer도 memory → 그럼 architecture와 optimizer는 같은 것의 다른 레벨 → 레벨을 더 쌓자(higher-order ICL) → self-mod + CMS + HOPE.
- 증명: 760M/1.3B 벤치(Transformer++·RWKV7·DeltaNet·Titans 등 대비), BABILong, formal language, level ablation figure.

### Sleep (2606.03979) — "offline consolidation (diff 전달)"
- **핵심주장**: wake/sleep lifecycle. **offline consolidation = Knowledge Seeding(upward distillation → low-rank expert 성장)** + **Dreaming(SEAL RL)**. NL의 online consolidation이 못 하던 **학습된 지식의 명시적 위쪽 전달**을 sleep이 함.
- 논거흐름: NL online consolidation(초기상태 meta-learn + circle-back)은 고정용량 한계 → offline로 param 성장 → GKD(on-policy)+LTI(RL) → 새 low-rank expert만 학습(freeze) → synaptic pruning → Dreaming으로 자기개선.
- 증명: class-incremental(CLINC/Banking/DBpedia), level 효과 figure, ablation.

---

## 3. 컷 목록 (뺄 것)
- 기존 s09-convergence → HOPE wrap-up으로 흡수(중복 제거).
- 논문설명 안의 시스템모델링(decode 비용 등) → 전부 C(System modeling)로 이동.
- QA 개별 깊은 파생(Q별) → appendix backup만, 본문은 핵심주장으로.
- 145슬라이드 자동레이아웃판 → 폐기, 도형 직접 그림으로 재작성.

## 4. 제작 방식
- python-pptx로 도형 직접 배치(box/arrow/DAG node/tensor label/step 번호). 헬퍼 모듈 `seminar/pptx_lib.py`.
- **글/그림 겹침 금지**: 각 슬라이드 렌더(PNG)해 우측·하단 여백 + 요소 bbox 겹침 검사.
- 논문 figure/table은 reffigs에서 인용(원저자 표기).
- 섹션별 증분 빌드 + 렌더 검증.
