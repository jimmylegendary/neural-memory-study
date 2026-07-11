# 구현 가용성 정찰 (impl-availability) — W1 recon

- 조사일: 2026-07-11 (star 수·push 날짜는 모두 이 날짜 기준 GitHub API 실측값)
- 조사 방법: WebSearch + GitHub REST API(repo 메타데이터, 디렉토리 리스팅, raw 소스 grep) + HuggingFace paper page 코드 링크 확인 + README/소스 구조 기반 충실도 평가. **실행 검증은 하지 않음** (P3 로컬 실험에서 수행).
- 소비처: Part III (a) 사내 A100 80GB x4 x2-node HOPE+Dreaming runbook, (b) 로컬 실험 설계.

---

## 0. 핵심 요약 (TL;DR)

1. **6편 전부 공식 구현 없음.** Google Research는 Titans(2025-01)부터 Sleep(2026-06)까지 단 한 줄의 코드도 공개하지 않았다. 1저자 Ali Behrouz의 개인 GitHub에 `ABehrouz/Titans` repo가 2025-04-09 생성되어 있으나 **완전히 비어 있다** (2026-07-11 확인). 공개 의도는 있었으나 이행되지 않은 것으로 해석된다.
2. 사실상의 reference 구현은 **lucidrains/titans-pytorch (1,966★)** 하나뿐이며, 그마저 Titans의 MAC variant 중심 + 저자 자체 확장이 섞여 있어 "논문 그대로"가 아니다.
3. **flash-linear-attention(FLA, 5,325★)** 이 Part I 계열(DeltaNet/GDN/GLA/RWKV-7/Mamba-2 등)의 production-grade Triton 커널을 전부 커버하고, `fla/ops/ttt`(진짜 Triton chunkwise 커널)와 `fla/ops/titans`(**naive PyTorch 참조 구현만**, Triton 없음)를 ops 레벨로 보유한다. Titans만 naive에 머문 사실 자체가 "chunk-level momentum은 chunkwise 커널화가 어렵다"는 ch09/TNT 서사의 실물 증거다.
4. **Miras·Atlas·TNT는 구현 공백 지대** (별 4개 이하 재현 시도 몇 개), **HOPE는 비공식 2개(84★/76★)가 소규모·단일 GPU 수준으로 존재**, **Sleep(dreaming/consolidation)은 지구상에 구현이 0개**다. OpenReview에 2025-09부터 공개돼 있었음에도 그렇다.
5. 따라서 runbook 전략은 **하이브리드**가 강제된다: FLA+flame 백본 위에 HOPE 자체 구현(비공식 2종은 참조·감사 대상이지 신뢰 기반 아님) + Sleep/Dreaming 완전 자체 구현. Sleep 구현은 이 스터디의 잠재적 기여 포인트다.

---

## 1. 종합 표

| 논문 (arXiv) | 공식 구현 | 최고 비공식 구현 (★, 최근 push) | FLA 커버리지 | JAX | 자체 구현 필요도 |
|---|---|---|---|---|---|
| Titans (2501.00663) | ❌ (`ABehrouz/Titans` 빈 repo) | lucidrains/titans-pytorch (1,966★, 2026-06) | `fla/ops/titans` naive-only (chunk_titans_linear), layer/model 통합 없음 | Titans-NNX (7★) | **낮음** — 참조 구현 풍부, 단 fidelity 감사 필요 |
| Miras (2504.13173) | ❌ | aryateja2106/neural-memory-reproduction (4★, 2025-12) — 수식 단위 재현 | 직접 없음 (인접: comba, mesa_net, mom) | 없음 | **중간** — gate/loss 교체는 쉬우나 검증된 것 없음 |
| Atlas (2505.23735) | ❌ | bhoener/atlas, engineerA314/atlas-rnn (각 0★) | 직접 없음 (가장 가까운 커널: `mesa_net` Triton, 2차 최적화 계열) | 없음 | **중간~높음** — Omega rule + Muon 업데이트 검증된 구현 부재 |
| TNT (2511.07343) | ❌ | **전무** | 없음 | 없음 | **높음** — 2-stage hierarchical 학습 전체 신규 |
| Nested Learning / HOPE (2512.24695) | ❌ (Google blog도 코드 없음) | obekt/HOPE-nested-learning (84★, 2026-06) / erikl2/nested-learning (76★, 2025-12) | 없음 | 없음 | **높음** — 소규모 비공식만 존재, A100급 학습 경로는 신규 |
| Sleep (2606.03979) | ❌ | **전무** (GitHub 검색 0건) | 없음 | 없음 | **최고** — 전 컴포넌트 신규 구현 |

---

## 2. 논문별 상세

### 2.1 Titans (2501.00663)

**공식**: 없음. HuggingFace paper page에도 링크 없음. `ABehrouz/Titans`는 빈 repo(0 KB, 2025-04 생성) — 감시 대상.

**lucidrains/titans-pytorch** — 1,966★ / 206 fork / open issues 36 / 최근 push 2026-06-06. `pip install titans-pytorch`.
- 구현 범위: `NeuralMemory`(핵심), `MemoryAsContextTransformer`(**MAC variant만** 문서화; MAG/MAL 없음), `memory_models.py`(MLP 이외 memory network 탐색), 그 외 저자 자체 실험(`implicit_mlp_attention.py`, `nested_attention.py`)이 섞여 있음.
- 수식 대응 (소스 grep 확인, `neural_memory.py` 1,081줄):
  - momentum ✅ (심지어 `momentum_order` > 1의 고차 momentum, `learned_momentum_combine` 등 논문 밖 확장)
  - data-dependent adaptive lr ✅ (`adaptive_step.sigmoid() * max_lr`)
  - weight decay(forget gate) ✅ (`init_decay_bias`)
  - chunkwise 처리 ✅ — 논문의 "chunk 내 $\theta_t, \eta_t, \alpha_t$ 공유" 구조를 `chunk_size`로 재현, `AssocScan`(accelerated associative scan) 기반 병렬화. dual form(Eq. 18–19의 matmul화)이라기보다 **scan 기반 병렬화**라는 점이 논문과의 구조적 차이.
  - 논문 밖 확장 주의: `spectral_norm_surprises`(Newton-Schulz 5차 반복; Muon 유래, 2025-06 커밋 — 사실상 Atlas 방향 선취), attention pooling, per-parameter lr modulation. **기본값이 논문 세팅이라는 보장이 없으므로** 실험 시 config 감사 필수.
- 평가: 참조 구현으로 최적. 단 "논문 재현체"가 아니라 "논문 + 저자 취향 개선판"이므로, 우리 monograph의 수식과 1:1 대응시키려면 flag를 논문 모드로 고정하는 감사 단계가 필요하다.

**FLA `fla/ops/titans`**: `chunk_titans_linear` — `naive.py` + `log_impl.py`뿐. seq_len x seq_len 행렬을 materialize하는 순수 PyTorch 참조 구현(log-space에서 momentum·decay 곱을 combine). Triton 커널 없음, `fla/layers`·`fla/models` 통합 없음. RFC #107(2025-01, sustcsonglin)에서 TTT·Titans 커널이 함께 발의됐으나 **TTT만 Triton화되고 Titans는 naive에 머묾**(#214에서 갱신 후 정체). 이것이 ch09의 핵심 논점(momentum 항이 chunkwise closed form을 깨뜨림 → TNT의 존재 이유)의 생태계 측 증거다.

**기타**: Aedelon/titans-pytorch-mlx (27★, PyTorch+MLX), Yuan-ManX/Titans-PyTorch (37★, 2025-01 이후 정지), pafos-ai/titans-trainer (11★, HF-style trainer 래퍼), hoshuaclawdbot/titans-cuda (0★).

### 2.2 Miras — Moneta / Yaad / Memora (2504.13173)

**공식**: 없음.

**비공식**: 사실상 공백. 유일하게 이름을 걸고 재현한 것이 **aryateja2106/neural-memory-reproduction** (4★, 2025-12-31 생성 직후 정지): Titans+Miras+NL 3편의 "핵심 수식"을 단위 테스트로 재현 (52 tests, 87% coverage, Moneta $\ell_p$ / Yaad Huber / Memora KL retention 포함). end-to-end 학습 파이프라인은 없음 — **수식 이해 검증용 참조로는 유용, 실험 기반으로는 불가**.

**FLA 관점**: Miras 자체는 없지만, Miras가 프레임워크 논문이라는 점이 중요하다 — Miras의 (attentional bias, retention gate) 좌표계에서 보면 FLA의 `comba`, `mesa_net`, `mom`, `kda`, `gated_deltanet` 등이 이미 그 설계 공간의 점들이다. 즉 **Miras의 "새 인스턴스 3종"만 없는 것이지, 설계 공간의 커널 인프라는 FLA에 있다**.

**자체 구현 평가**: retention gate·loss 교체는 recurrence의 대수 구조를 바꾼다. $\ell_2$ 계열은 FLA DeltaNet/GLA 커널 재활용이 가능하지만, Moneta의 $\ell_p$ norm처럼 **nonlinear한 순간 chunkwise closed form이 사라져** naive 순차 루프(또는 논문의 근사)로 후퇴해야 한다. Part II ch13 집필 시 이 지점을 명시할 것.

### 2.3 Atlas (2505.23735)

**공식**: 없음.

**비공식**: bhoener/atlas (0★, 2026-03까지 push), engineerA314/atlas-rnn (0★, "Modern RNN form of Atlas"), trandat27jk/Mnemo-GPT (0★, Atlas 기반 LLM 시도). 전부 검증 이력 없는 개인 프로젝트 — **참조 신뢰도 최하**.

**부분 흡수**: lucidrains/titans-pytorch가 Atlas를 README에 인용하고 spectral-norm surprise(Newton-Schulz = Muon의 orthogonalization)를 옵션으로 흡수했다. 그러나 Atlas의 본체인 **Omega rule(sliding-window context memorization)과 polynomial feature map의 명시적 구현은 확인되지 않음**.

**FLA 관점**: 가장 가까운 production-grade 커널은 `mesa_net`(MesaNet, 2506.05233 — locally optimal test-time regression, chunkwise CG solver까지 Triton으로 구현: `chunk_cg_solver_fwd/bwd.py`). Atlas의 "2차 정보 활용 + 윈도우 단위 최적화" 방향을 커널 수준에서 보고 싶으면 MesaNet 코드가 실질 교재다.

### 2.4 TNT (2511.07343)

**공식/비공식 모두 전무.** GitHub repo 검색 0건, OpenReview(rajioNWfRs)에도 코드 링크 없음. NeurIPS-track 논문임에도 저자(Zeman Li + Behrouz 라인)가 코드를 내지 않았다.

재현 시 필요한 것: hierarchical memory (large-chunk global module + parallel local modules), 2-stage 학습(효율 지향 pre-training → fine-grained 전환), chunk-size scheduling. **ch09(chunk size = semantic hyperparameter)와 ch15의 수치 주장을 검증하려면 우리 손으로 만들어야 한다.** 다만 Part III fast-path에서는 TNT 재현은 우선순위 낮음 (HOPE+Sleep이 목표물).

### 2.5 Nested Learning / HOPE (2512.24695)

**공식**: 없음. Google Research 블로그(2025-11 "Introducing Nested Learning")도 코드 링크 없이 종료. HuggingFace paper page에도 구현 링크 0. NeurIPS 2025 camera-ready에도 코드 문장 없음.

**비공식 지형** (모두 2025-11 이후 등장, 소규모):

| repo | ★ | push | 범위 | 평가 |
|---|---|---|---|---|
| obekt/HOPE-nested-learning | 84 | 2026-06-11 | self-mod fast weights(delta rule + forget gate + per-token lr), CMS(Fast/Medium/Slow = 1/4/16 step tier), 25M~300M preset, Wikipedia+QA 2-phase 학습 스크립트, full-seq vs token-by-token equivalence test | 가장 활발·가장 완성도 높음. 단 consumer HW(8–16GB) 타깃, 분산 학습 없음, continual-learning 이득 미검증을 스스로 인정. MIT |
| erikl2/nested-learning | 76 | 2025-12-08 | DeepMomentumGD, HOPE+self-modifying attention, CMS, L2RegressionAttention, WikiText-103/LAMBADA eval, 27+ tests(finite-difference gradient check) | **"paper-exact mode vs stable default(surrogate loss)"를 명시 구분** — 충실도 의식이 가장 높고, 이 구분 자체가 우리 감사 체크리스트로 쓸 만함. 분산 미구현. MIT |
| WindOfNature/Nested-Learning | ? | — | (검색 인덱스에 CMS chunking 커스텀 커널 주장으로 잡히나) **계정/repo 현재 접근 불가 — 소실** | 인용 금지 |
| Ray0907/Hope, engichang1467/Open-HOPE, rlaope/HOPE-tensorflow, TASMAYU/HOPE_NL_CMS 등 | ≤2 | — | 부분 재현 | 참고 가치 낮음 |

**공통 한계**: 어느 것도 (i) 논문 스케일(340M–1.3B, 장기 토큰) 학습, (ii) multi-GPU/multi-node, (iii) 논문 벤치마크(BABILong, NIAH, class-incremental) 재현을 하지 않았다. **HOPE의 "결과 재현"은 아직 세상에 없다** — 우리 runbook이 사실상 첫 시도 축에 든다.

### 2.6 Sleep (2606.03979)

**구현 0.** arXiv/HF/OpenReview(iiZy6xyVVE, 2025-09부터 공개) 어디에도 코드 없음. GitHub 검색("knowledge seeding", "dreaming consolidation" 등) 0건. 주의: 동명 이논문 **2605.26099 (Lee et al., "Do Language Models Need Sleep?")와 혼동 금지** — 그쪽은 offline recurrence 논문으로 별개 계열이다.

즉 wake-sleep 파이프라인(Memory Consolidation = Knowledge Seeding 상향 distillation, Dreaming = self-improvement loop, periodic parameter (de)activation) 전체가 **완전 자체 구현 대상**이다. → §5 분해 참조.

---

## 3. flash-linear-attention 생태계 (runbook 기반 인프라)

- **fla-org/flash-linear-attention**: 5,325★ / 575 fork / push 2026-07-09 (활발). Triton 기반, NVIDIA/AMD/Intel 검증, HF transformers 호환 모델 클래스 제공.
- 커버리지 (스터디 관련만): `delta_net`, `gated_deltanet`, `gdn2`, `gated_deltaproduct`, `gla`, `rwkv7`, `mamba2`, `mamba3`, `log_linear_mamba2`, `mesa_net`, `comba`, `kda`, `mom`, `deltaformer`, `path_attn` — Part I ch06–07의 전 계보가 layer+model+Triton kernel로 존재. **Longhorn은 FLA에 없음** (공식: Cranial-XIX/longhorn 57★, 2024-12 이후 정지).
- **ops-only 2종** (layer/model 통합 없음 — 직접 wiring 필요):
  - `fla/ops/ttt`: `chunk_ttt_linear`, `fused_chunk_ttt_linear` — **진짜 Triton 커널** (group_norm 융합, varlen `cu_seqlens`, initial/final state 지원). TTT-Linear 계열 실험의 최적 출발점.
  - `fla/ops/titans`: `chunk_titans_linear` — naive PyTorch만. 위 §2.1 참조.
- **학습 프레임워크**: fla-org/flame (401★, torchtitan 기반) — FLA 모델의 multi-GPU 학습 표준 경로. A100 x4 x2-node에서 baseline(GDN, Mamba-2, Transformer) 학습에 바로 사용 가능.
- **TTT 공식 (Sun et al., ch08용)**: test-time-training/ttt-lm-pytorch (1,382★, inference용 naive), ttt-lm-jax (461★, 학습용 공식), ttt-lm-kernels (91★, inference 커널). Titans 라인은 아니지만 dual form의 공식 구현이 있는 유일한 지점.
- 기타 공식: NVlabs/GatedDeltaNet (619★, ICLR 2025 공식) — ch06 fidelity 대조용.

## 4. JAX vs PyTorch 지형

- **PyTorch + Triton 압승.** 커널(FLA)·참조 구현(lucidrains)·HOPE 비공식 전부 PyTorch.
- JAX 측: ttt-lm-jax(461★, TTT 공식 학습 코드)가 유일한 무게감 있는 자산. Titans는 shayme92/Titans-NNX(7★)뿐, Miras/Atlas/TNT/HOPE/Sleep JAX 구현 0.
- 사내 A100 클러스터 + 우리 팀 역량 고려 시 **PyTorch(+FSDP via flame/torchtitan) 단일 트랙 권고**. JAX는 감시만.

---

## 5. HOPE + Dreaming 자체 구현 범위 분해 (Part III 대비)

전제: 논문 결과의 "정확 재현"이 아니라 **runbook 실측 + scaling 관찰이 목적** → 소형(130M–340M)에서 semantics 검증 후 A100 x8에서 실측. 공수는 1인 풀타임 기준(agent-assisted 시 단축).

| # | 컴포넌트 | 재사용 소스 | 난이도 | 공수 |
|---|---|---|---|---|
| C1 | deep neural memory 코어 (MLP fast weights, momentum, weight decay, adaptive lr, chunkwise inner loop) | lucidrains `NeuralMemory`(논문 모드로 flag 고정) + `fla/ops/titans` naive를 정답 참조로 | 중 | 3–5일 |
| C2 | self-modifying Titans (self-referential update rule 생성; NL §self-mod) | obekt/erikl2 참조 + 논문 수식 직접; erikl2의 paper-exact/surrogate 구분을 감사 체크리스트로 | **상** | 1주 |
| C3 | CMS (multi-frequency FFN chain, tier별 update period + gradient 누적) | obekt 구조 참조; 본질적으로 단순 | 하 | 1–2일 |
| C4 | HOPE block 조립 + LM 골격 (tokenizer/data/config) | flame(torchtitan) 골격에 custom block 주입 | 중 | 3일 |
| C5 | 병렬 학습 경로 (naive 순차 → chunkwise) | `fla/ops/ttt` Triton 커널 패턴 + `fla/ops/titans` log_impl; 1차 목표는 naive+`torch.compile`로 충분, Triton화는 옵션 | 상(옵션) | 1–2주 (Triton +2주) |
| C6 | **Sleep: Memory Consolidation** (Knowledge Seeding 상향 distillation: small-self memory → larger network, replay buffer) | 전부 신규; distillation loss·replay pipeline은 표준 부품 조합 | 상 | 1–1.5주 |
| C7 | **Sleep: Dreaming** (self-generated rollout → self-modification loop) + periodic parameter (de)activation (fast block 파라미터 교체/재활성) | 전부 신규 — 세계 최초 공개 구현이 될 영역 | **최상** | 1–1.5주 |
| C8 | eval harness (WikiText ppl, NIAH, BABILong, continual metrics) | lm-eval-harness + booydar/babilong 재사용 | 중 | 3–5일 |
| C9 | multi-node 실측 배선 (FSDP/DDP, 80GB x4 x2-node, ckpt/재시작/예외 시나리오) | flame/torchtitan 기본 제공 + 사내 스택 인터뷰 필요(PLAN 열린 항목) | 중 | 1주 |

**합계: 약 6–9주** (C5 Triton 제외 시 하한). 로컬 소형 semantics 검증(C1–C4 + C6/C7 축소판)은 **~2–3주**로 분리 가능.

리스크: (i) Sleep 논문의 Dreaming은 수식 대비 자유도가 커서 해석 재량이 필요 — monograph ch17에서 해석을 먼저 고정한 뒤 구현할 것. (ii) HOPE 비공식 2종 모두 surrogate를 섞으므로 그대로 fork하면 "무엇을 측정했는지" 모호해짐 — fork 금지, 참조만.

---

## 6. Runbook 구현 전략 권고 (결론)

**하이브리드 4층**:

1. **베이스라인·백본 = FLA + flame** (공식급 신뢰, multi-GPU 검증됨): Transformer/GDN/Mamba-2 대조군, HF 호환 eval.
2. **Titans neural memory = lucidrains 참조 + 논문 모드 감사**: 논문 밖 확장 flag 전수 점검 후 사용. `fla/ops/titans` naive를 수치 정답(oracle)으로 병용.
3. **HOPE = 자체 구현** (obekt/erikl2는 reference-only): C1–C5. 신뢰 기반이 아니라 대조·감사 대상.
4. **Sleep/Dreaming = 완전 자체 구현** (C6–C7): 공개 구현 0 — 이 monograph/runbook의 차별화 지점이자 잠재적 오픈소스 기여 포인트.

로컬 실험(Part III-b): PyTorch naive + `torch.compile`, ≤130M, single-GPU — chunk size sweep과 semantics 검증에 집중. A100 실측(Part III-a): flame 기반 FSDP, 130M→340M(→760M 여유 시), wake-sleep cycle 실측 + BABILong/NIAH.

**감시 항목** (P2 전 재확인): ① `ABehrouz/Titans` 빈 repo에 코드 등장 여부, ② FLA의 Titans Triton화(#107 후속), ③ TNT/Sleep 공식 코드 출현, ④ lucidrains의 Atlas/HOPE 신규 repo.

---

## Sources

- Titans: https://arxiv.org/abs/2501.00663 · https://github.com/lucidrains/titans-pytorch · https://github.com/ABehrouz/Titans (빈 repo) · https://github.com/Aedelon/titans-pytorch-mlx · https://github.com/Yuan-ManX/Titans-PyTorch · https://github.com/pafos-ai/titans-trainer · https://github.com/shayme92/Titans-NNX
- Miras: https://arxiv.org/abs/2504.13173 · https://huggingface.co/papers/2504.13173 · https://github.com/aryateja2106/neural-memory-reproduction
- Atlas: https://arxiv.org/abs/2505.23735 · https://huggingface.co/papers/2505.23735 · https://github.com/bhoener/atlas · https://github.com/engineerA314/atlas-rnn · MesaNet https://arxiv.org/abs/2506.05233
- TNT: https://arxiv.org/abs/2511.07343 · https://openreview.net/forum?id=rajioNWfRs
- Nested Learning/HOPE: https://arxiv.org/abs/2512.24695 · https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/ · https://github.com/obekt/HOPE-nested-learning · https://github.com/erikl2/nested-learning · https://huggingface.co/papers/2512.24695
- Sleep: https://arxiv.org/abs/2606.03979 · https://huggingface.co/papers/2606.03979 · https://openreview.net/forum?id=iiZy6xyVVE · (혼동 주의) https://arxiv.org/abs/2605.26099
- 인프라: https://github.com/fla-org/flash-linear-attention (issues #107, #214) · https://github.com/fla-org/flame · https://github.com/test-time-training/ttt-lm-pytorch · https://github.com/test-time-training/ttt-lm-jax · https://github.com/test-time-training/ttt-lm-kernels · https://github.com/NVlabs/GatedDeltaNet · https://github.com/Cranial-XIX/longhorn
