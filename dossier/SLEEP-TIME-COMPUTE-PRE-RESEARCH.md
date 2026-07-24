# PRE-RESEARCH — Sleep-Time Compute, Continual Learning, and Long-Term Memory

> 조사 기준일: 2026-07-25
> 목적: sleep-time compute를 **모수적 기억 업데이트**와 **모델 외부 기억 관리**라는 두 축으로
> 재구성하고, 관련 연구를 최초 공개 시점 순으로 읽기 위한 사전 조사 문서
> 범위: 논문, 공식 연구·제품 문서, 그리고 블로그에서 지목한 PLOS·DANN·Letta·Mem0·Zep 계보

---

## 0. 결론부터

이 분야에는 아직 합의된 단일한 “sleep-time compute” 문헌군이 없다. 같은 표현 아래 다음
세 가지가 섞여 있다.

1. **pre-query context compilation**: 미래 질문이 오기 전에 문서·대화·환경을 읽고, 더 짧고
   유용한 context로 다시 쓴다. Letta 연구진의 2025 *Sleep-time Compute* 논문이 여기에
   해당한다. 학습된 모델 weight는 바뀌지 않는다.
2. **external-memory lifecycle**: 깨어 있을 때 쌓인 text/vector/graph/trajectory를
   추출·병합·추상화·검증·삭제한다. Mem0와 Graphiti core는 주로 hot-path/ingest-time,
   Letta sleep agent와 Auto-Dreamer는 명시적 async/offline이라는 차이가 있지만 같은
   외부기억 operator 계보다.
3. **latent/parametric compilation and consolidation**: 경험을 latent KV, adapter, fast weight,
   sparse memory slot, 혹은 slow model weight로 옮긴다. Cartridges, CMU/UMD의 Offline
   Recurrence, Google Research의 `Language Models Need Sleep`는 명시적 offline 경로이고,
   SEAL과 Sparse Memory Finetuning은 sleep system이 차용할 수 있는 adjacent
   continual-update 경로다.

따라서 이 리서치의 중심 질문은 “AI도 잠을 자야 하는가?”가 아니다.

> **배포 후 들어오는 경험을 언제, 어떤 연산으로, 어느 기억 매질에 써야 하며, 유한한 용량과
> 간섭·오염·비용 아래에서 다음 wake의 성능을 어떻게 높일 것인가?**

사용자가 제안한 두 축은 정확하다. 다만 최종 시스템은 둘 중 하나를 택하기보다 다음과 같은
**promotion/demotion ladder**가 될 가능성이 높다.

```text
raw event / exact transcript
        ↓ admission, deduplication
episodic text·vector·temporal graph
        ↓ merge, abstraction, verification
semantic/procedural external memory
        ↓ context distillation / compilation
latent KV·cartridge·per-user adapter·fast weight
        ↓ repeated-use evidence + replay/distillation
slow/shared parametric memory
```

각 승격에는 원본 provenance와 rollback 경로가 남아야 한다. 반대로 오래되거나 불확실한
기억은 slow weight에서 즉시 “삭제”하기 어렵기 때문에, 외부 계층으로 내리거나 retrieval
시점에 차단할 수 있어야 한다.

---

## 1. 조사 프레임

### 1.1 시간축

| 단계 | 입력과 지연 제약 | 주 연산 | 상태 수명 |
|---|---|---|---|
| **pre-/post-training** | 배포 전 대규모 corpus; 높은 latency 허용 | pretraining, SFT, RL, distillation | global checkpoint |
| **wake / test-time** | 실제 요청·환경 stream; 낮은 P99 필요 | inference, retrieval, capture, 제한적 TTT | turn/session/user |
| **sleep-time** | 외부 입력을 잠시 끊거나 비동기 snapshot 사용 | replay, merge, abstraction, synthetic data, SFT/RL, growth/pruning | user/tenant/global |
| **next wake** | 새 query·task | 검색·adapter load·memory application | live working set |

여기서 “sleep”은 반드시 wall-clock 밤이나 완전한 idle을 뜻하지 않는다. 더 정확한 정의는
**사용자 응답의 critical path 밖에서, 미래 요청을 위해 persistent state를 변환하는
비동기 compute**다. traffic이 계속 있는 서비스라면 snapshot·queue·version을 사용해 wake와
sleep이 겹칠 수 있다.

### 1.2 기억 위치축

| 위치 | 예 | 장점 | 주 실패 모드 |
|---|---|---|---|
| raw/exact external | log, transcript, document | 높은 충실도·감사·삭제 용이 | 무한 성장, 검색/보안 비용 |
| symbolic external | summary, profile, skill, temporal KG | 편집·provenance·model portability | 요약 손실, 모순, hallucinated write |
| latent external | KV cache, cartridge, learned vector | 짧은 live context, 빠른 적용 | 해석·삭제·model migration 어려움 |
| session/user parametric | LoRA, sparse slot, fast weight | 반복 query에 낮은 latency | 간섭, GPU residency, version explosion |
| shared slow weight | backbone/MoE expert | 가장 싼 반복 사용, 일반화 가능 | privacy, unlearning, catastrophic forgetting |

### 1.3 “진짜 sleep-time” 판정 기준

한 연구를 아래 다섯 질문으로 읽는다.

1. **wake input**: 무엇을 경험으로 수집하는가?
2. **sleep trigger**: idle, 주기, context pressure, 중요도, budget 중 무엇이 sleep을 시작하는가?
3. **sleep operator**: retain/merge/abstract/rewrite/replay/distill/SFT/RL/grow/prune 중 무엇인가?
4. **destination**: text/vector/graph/KV/adapter/fast weight/slow weight 중 어디에 쓰는가?
5. **capacity policy**: 무엇을 보존·검증·버리고, 가득 차면 어떻게 하는가?

단순한 long-context, RAG, continual fine-tuning, test-time reasoning은 이 다섯 항목을 만족하지
않으면 인접 연구이지 sleep-time compute 자체는 아니다.

### 1.4 증거 등급

| 등급 | 의미 |
|---|---|
| **A** | 동료평가된 논문 또는 공식 학회 게재가 확인됨 |
| **B** | arXiv/OpenReview 연구; 방법·실험이 있으나 동료평가 상태를 단정하지 않음 |
| **C** | 기업 공식 연구/제품 문서; 실제 구현 증거는 있으나 논문과 분리 |
| **D** | SSRN·research preview·단독 제안; 재현 전에는 방향성 증거로만 사용 |

### 1.5 공통 문제로 형식화

wake에서 사건 stream `e₁:t`와 volatile state `hₜ`가 생기고, persistent memory를

```text
M = {Mraw, Mtext, Mvector, Mgraph, MKV, Madapter, Mfast, Mslow}
```

라고 하자. sleep policy `Sφ`는 budget `Bs`, provenance/risk constraint `R` 아래에서

```text
Mₜ₊₁ = Sφ(e₁:t, Mₜ ; Bs, R)
```

를 수행한다. 다음 wake의 answer policy는 live budget `Blive` 아래

```text
a = Aθ(q, retrieve(Mₜ₊₁, q) ; Blive)
```

를 계산한다. 연구 목표는 단순 accuracy 최대화가 아니라 대략 다음 항의 합을 최소화하는
것이다.

```text
future task loss
+ λlive · live latency/tokens
+ λsleep · sleep FLOPs/energy
+ λstore · persistent bytes
+ λmove · bytes moved across tiers
+ λinterfere · forgetting/retrieval competition
+ λstale · stale/conflicting-memory loss
+ λrisk · privacy/corruption/unlearning risk
```

이 관점에서 external과 parametric은 상호 배타적인 정답이 아니라, `Sφ`가 선택할 destination
action이다. 연구의 핵심 학습 대상은 answer model `θ`만이 아니라 **memory operator와
destination을 고르는 policy `φ`**다.

---

## 2. 시간순 계보

날짜는 가능한 경우 최초 공개일을 썼다. “Meta/Google/Microsoft” 표시는 저자 소속이나 공식
연구 페이지로 확인되는 경우에만 붙였다.

### 2.1 1983–2021: 생물학적 basis, replay, continual learning, test-time update

| 시점 | 연구 | 이번 리서치에서의 역할 |
|---|---|---|
| 1983-07 | Crick & Mitchison, [The Function of Dream Sleep](https://doi.org/10.1038/304111a0) **[A, 이론]** | REM에서 기생적 attractor를 약화하는 “reverse learning” 가설. 새 실험이나 AI benchmark는 없다. |
| 1989 | Buzsáki, [Two-stage Model of Memory Trace Formation](https://doi.org/10.1016/0306-4522%2889%2990423-5) **[A, 이론]** | wake/theta의 빠른 hippocampal encoding과 SWS sharp-wave의 cortical replay라는 2-stage 모델. |
| 1994-07 | Wilson & McNaughton, [Reactivation of Hippocampal Ensemble Memories During Sleep](https://doi.org/10.1126/science.8036517) **[A]** | 공간 탐색 후 rat CA1 ensemble correlation이 SWS에서 재등장한 replay의 고전적 실험 증거. 기억 향상의 필요·충분조건까지 증명한 것은 아니다. |
| 1995-05 | Hinton et al., [The “Wake-Sleep” Algorithm](https://pubmed.ncbi.nlm.nih.gov/7761831/) **[A]** | recognition/generative network를 번갈아 학습한 명명상의 조상. 배포 후 lifetime consolidation은 아니다. |
| 1995-07 | McClelland et al., [Complementary Learning Systems](https://web.stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf) **[A]** | 빠른 episodic system과 느린 structured system, replay를 통한 이전이라는 핵심 이론. |
| 1995 | Robins, [Catastrophic Forgetting, Rehearsal and Pseudorehearsal](https://doi.org/10.1080/09540099550039318) **[A]** | random pseudo-input에 old network output을 label로 붙여 새 자료와 interleave. 생성 replay의 공학적 선조지만 toy backprop이며 sleep 구현은 아니다. |
| 1996-03 | Skaggs & McNaughton, [Replay of Neuronal Firing Sequences](https://doi.org/10.1126/science.271.5257.1870) **[A]** | rat SWS에서 wake의 공간 sequence가 압축 재생됨. |
| 2001-01 | Louie & Wilson, [Temporally Structured Replay During REM](https://doi.org/10.1016/S0896-6273%2801%2900186-6) **[A]** | rat REM에서 wake ensemble sequence가 시간 왜곡을 동반해 재현됨. 꿈 내용이나 생성적 AI operator의 증거는 아니다. |
| 2003 | Tononi & Cirelli, [Sleep and Synaptic Homeostasis](https://doi.org/10.1016/j.brainresbull.2003.09.004) **[A, 가설]** | wake의 순 synaptic potentiation을 sleep에서 선택적으로 downscale해 SNR·capacity를 회복한다는 basis. |
| 2007 | Rasch et al., [Odor Cues During SWS](https://doi.org/10.1126/science.1138581) **[A]** | wake 때의 odor cue를 SWS에서 재제시하면 사람의 declarative recall이 향상된 targeted reactivation의 인과 증거. |
| 2007 | Ji & Wilson, [Coordinated Replay in Visual Cortex and Hippocampus](https://doi.org/10.1038/nn1825) **[A]** | rat SWS에서 V1과 hippocampus의 경험 관련 sequence가 조정되어 재생된 상관 증거. 방향성 있는 hippocampus→cortex 전송의 인과 증거는 아니다. |
| 2013 | Ngo et al., [Closed-loop Slow-Oscillation Stimulation](https://doi.org/10.1016/j.neuron.2013.03.006) **[A]** | NREM phase에 맞춘 acoustic stimulation이 slow oscillation/spindle과 다음날 recall을 높임. |
| 2016-07 | Kumaran, Hassabis, McClelland, [CLS Theory Updated](https://pubmed.ncbi.nlm.nih.gov/27315762/) **[A, Google DeepMind]** | replay가 단순 복사가 아니라 goal-weighted statistics를 만들 수 있고, 기존 구조와 정합적인 지식은 빠르게 cortical learning될 수 있다고 확장. |
| 2016-12 / 2017 | Kirkpatrick et al., [EWC](https://arxiv.org/abs/1612.00796) **[A, DeepMind]** | 중요한 weight를 덜 움직이는 regularization 계보. 기억 용량을 늘리지는 않는다. |
| 2017-05 / NeurIPS 2017 | Shin et al., [Continual Learning with Deep Generative Replay](https://arxiv.org/abs/1705.08690) **[A]** | generator가 과거형 sample을 만들어 solver와 interleave. 이후 sleep data synthesis의 강한 공학적 선조지만 generator drift·비용과 MNIST형 benchmark 한계가 있고 sleep state는 아니다. |
| 2017-06 | Lopez-Paz & Ranzato, [Gradient Episodic Memory](https://arxiv.org/abs/1706.08840) **[A, FAIR]** | 작은 episodic buffer의 gradient constraint로 과거 성능을 보호. external replay와 parametric update의 초기 hybrid. |
| 2017-11 | Aljundi et al., [Memory Aware Synapses](https://arxiv.org/abs/1711.09601) **[A, KU Leuven+FAIR]** | unlabeled stream에서 parameter importance를 누적해 선택적으로 보존·덮어쓰기. 유한 용량을 명시적으로 문제화. |
| 2018-10 / 2019-01 | Fachechi et al., [Dreaming Neural Networks](https://doi.org/10.1016/j.neunet.2019.01.006) **[A]** | Hopfield memory의 offline unlearning/consolidation으로 spurious attractor를 제거하고 이상화된 이론 capacity를 높임. 자연 데이터나 deep continual learning은 아니다. |
| 2019-07 | Golden et al., [PLOS 연구의 bioRxiv 선공개판](https://doi.org/10.1101/688622) **[preprint]** | 2022 PLOS 논문의 실제 최초 공개. 따라서 아이디어의 chronology는 2019→2022다. |
| 2019-07 / NeurIPS 2019 | Lample et al., [Large Memory Layers with Product Keys](https://arxiv.org/abs/1907.05242) **[A, FAIR+Sorbonne]** | 최대 10억 parameter의 sparse KV lookup. Meta Memory Layers/SMF의 직접적인 구조적 조상이며, 당시에는 pretrain 후 고정되어 continual sleep은 없다. |
| 2019-08 | Krishnan et al., [Biologically Inspired Sleep Algorithm for ANNs](https://arxiv.org/abs/1908.02240) **[B]** | ANN→SNN 변환, 평균 입력통계 기반 noise+STDP sleep, 다시 ANN 변환. generative replay보다 약하고 dataset별 tuning이 필요했다. |
| 2019-09 / ICML 2020 | Sun et al., [Test-Time Training](https://arxiv.org/abs/1909.13231) **[A]** | unlabeled test sample의 self-supervised loss로 weight를 갱신. distribution-shift adaptation이지 장기 개인 기억은 아니다. |
| 2019-10 | Hayes et al., [REMIND](https://arxiv.org/abs/1910.02509) **[A]** | raw image 대신 압축 latent를 replay하는 continual learning. sleep dataset의 저장비용을 낮추는 전구체. |
| 2019-12 / ICLR 2020 | Tadros et al., [Sleep Algorithm for Generalization and Robustness](https://openreview.net/forum?id=r1xGnA4Kvr) **[A]** | ANN→SNN/STDP sleep이 일부 noise/adversarial robustness를 높임. continual lifetime capacity 실험은 아니다. |
| 2020-08 | González et al., [Can Sleep Protect Memories from Catastrophic Forgetting?](https://doi.org/10.7554/eLife.51005) **[A]** | 생물물리 thalamocortical model에서 input 없는 SWS dynamics가 두 상반 sequence의 부분 trace를 replay. 완전 소실 전만 복구 가능. |
| 2020-08 | van de Ven et al., [Brain-inspired Replay for Continual Learning](https://doi.org/10.1038/s41467-020-17866-2) **[A]** | generative feedback model이 과거 sample을 만들어 새 데이터와 섞는다. 강한 replay baseline이지만 sleep state 자체를 구현하지 않는다. |
| 2021 | Sukhbaatar et al., [Expire-Span](https://arxiv.org/abs/2105.06548) **[A, Meta]** | 각 memory에 학습된 expiration을 부여. “무한히 쌓기” 대신 유한 attention memory의 수명을 학습하는 선행 아이디어. |
| 2021-05 | Hoel, [The Overfitted Brain](https://doi.org/10.1016/j.patter.2021.100244) **[A, 가설]** | 꿈을 distorted/augmented input으로 보아 overfitting을 줄인다는 perspective. 새 계산·생물 실험은 없다. |

생물학에서 비교적 강하게 말할 수 있는 것은 “수면이 기억 통합을 돕고, 경험 관련 활동이
수면 중 재현되며, 일부 cue와 rhythm 조작이 다음날 기억을 바꾼다”까지다.
**NREM=정확 replay, REM=창의적 adversarial generation**이라는 깔끔한 역할 분담은 유용한
계산 가설이지 확정된 생물학적 사실이 아니다.

### 2.2 2021–2022: 생물학적 sleep을 계산 절차로 분해

| 시점 | 연구 | wake → sleep 메커니즘 | 해석 |
|---|---|---|---|
| 2021-09 preprint / 2022-05 eLife | Deperrois et al., [Perturbed and Adversarial Dreaming](https://elifesciences.org/articles/76384) **[A]** | wake에서 episodic latent 저장 → NREM에서 occlusion을 가한 replay/reconstruction → REM에서 기억 혼합+noise를 GAN식 adversarial sample로 생성 | NREM/REM에 서로 다른 objective를 준 명시적 계산 가설. CIFAR-10/SVHN representation learning이며 continual learning·catastrophic forgetting·lifetime capacity 실험은 아니다. |
| 2022-11 | Golden et al., [Sleep Prevents Catastrophic Forgetting in SNNs](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010628) **[A]** | 새 task RL 사이에 input을 끄고 Poisson stimulation과 unsupervised STDP로 자발적 old-trace reactivation | 블로그가 말한 “2022 PLOS”의 정확한 논문. **PLOS Computational Biology 저널 논문이지 PLOS 학회가 아니다.** |
| 2022-12 | Tadros et al., [Sleep-like Unsupervised Replay](https://doi.org/10.1038/s41467-022-34938-7) **[A]** | backprop wake 뒤 binary noise와 local Hebbian update의 sleep-like replay | 여러 vision class split에서 망각을 줄였지만 generative replay보다 약하고 “zero forgetting”은 아니다. |

PLOS 논문의 중요한 결과와 한계:

- feed-forward spiking network가 두 complementary foraging task를 순차 학습한다.
- wake에서는 input→hidden에 STDP, hidden→output에 reward-modulated STDP를 쓴다.
- sleep에서는 외부 input을 끄고 hidden layer를 Poisson spike로 자극하며, reward STDP를
  unsupervised STDP로 바꾼다. 과거 환경의 raw data를 명시적으로 다시 주지 않는다.
- Task 2만 순차 학습하면 Task 1 성능이 `0.52±0.02`, 즉 chance까지 떨어지고 Task 2는
  `0.69±0.03`이다. 새 task와 sleep을 번갈아 수행하면 Task 1/2가 각각
  `0.70±0.03 / 0.68±0.05`가 되며, 분석상 두 task solution manifold의 교집합 쪽으로 이동한다.
- 그러나 **842-neuron SNN, 상보적 foraging task 2개, 최소 10개 random-initialization
  simulation**에 한정된다. old trace가 새 학습으로 이미 지워진 뒤에는 sleep이 복구하지
  못한다. 동물·사람·LLM·장기 다과제 용량을 검증하지 않았다.

### 2.3 2023: LLM 외부기억과 “reflection”이 등장

| 시점 | 연구 | 기억 위치 / operator | sleep-time과의 관계 |
|---|---|---|---|
| 2023-03-20 | Shinn et al., [Reflexion](https://arxiv.org/abs/2303.11366) **[A]** | 이전 trial trajectory+reward에서 자연어 교훈을 만들고 episodic prompt memory에 append | inter-episode consolidation의 선행 사례. 최근 1–3 reflection만 쓰며 별도 sleep scheduler나 weight update는 없다. |
| 2023-04-07 | Park et al., [Generative Agents](https://arxiv.org/abs/2304.03442) **[A]** | text memory stream; recency·importance·relevance 검색; reflection으로 고차 summary 생성 | threshold-triggered reflection은 기능적 consolidation이지만 주로 online이다. |
| 2023-05-17 / AAAI 2024 | Zhong et al., [MemoryBank](https://arxiv.org/abs/2305.10250) **[A]** | daily event/personality summary, global portrait, text/vector retrieval; Ebbinghaus-inspired decay/reinforcement | 기능적 periodic summarization이나 명시적 sleep은 아니다. 10일 simulated dialogue 규모라 lifetime 증거는 약하다. |
| 2023-05-19 | [Zep product launch](https://blog.getzep.com/introducing-zep-memory-ai/) **[C]** | message store에 비동기 summary·embedding·entity enrichment | 외부기억 background processing의 제품 선행 사례지만 cross-memory sleep은 아니다. |
| 2023-06 / NeurIPS 2023 | Wang et al., [LongMem](https://arxiv.org/abs/2306.07174) **[A, Microsoft Research+UCSB]** | frozen backbone/memory encoder + trained adaptive residual SideNet retriever-reader; past context bank | base encoder는 frozen이고 외부 latent memory를 갱신하지만 SideNet은 memory-augmented adaptation으로 학습한다. |
| 2023-10-12 | Packer et al., [MemGPT](https://arxiv.org/abs/2310.08560) **[B, Berkeley]** | in-context working memory + recall DB + archival DB; function call로 self-edit; pressure warning·recursive summary | 원 논문에는 별도 sleep-trained consolidator가 없다. 대화·tool·memory management를 같은 agent가 online으로 수행한다. |
| 2023-12 | Jie et al., [Wake-Sleep Consolidated Learning](https://arxiv.org/abs/2401.08623) **[B]** | wake의 dynamic parameter freezing/STM → NREM replay → REM generative unseen input | vision continual learning이며 LLM agent lifecycle 증거는 아니다. |

MemGPT에서 자주 생기는 오해를 바로잡으면, timed event와 context-pressure interrupt가 있다고
해서 그것이 곧 Letta의 후속 **sleep agent**인 것은 아니다. 원 설계는 raw evicted message를
recall storage에 남기고, main context가 차면 recursive summary를 만든다. 2025년 Letta가
대화 critical path와 memory editing을 두 agent로 분리하면서 비로소 명시적 비동기 sleep
architecture가 된다.

### 2.4 2024: test-time memory layer와 context→parameter compilation

| 시점 | 연구 | 기억 위치 / 학습법 | 의미 |
|---|---|---|---|
| 2024-02 / ICML 2024 | Wang et al., [MEMORYLLM](https://arxiv.org/abs/2402.04624) **[A, UCSD/UCLA/Amazon]** | 1.066B memory parameter의 per-layer latent token pool에 새 chunk를 forward-only로 쓰고 old token을 random drop | 거의 1M update에서도 operational integrity/no performance degradation을 보고했다. 후속 M+가 실제 knowledge retention은 20K update 이내로 제한된다고 지적해 selective forgetting·offload 필요성을 드러냈다. |
| 2024-02-27 | Maharana et al., [LoCoMo](https://arxiv.org/abs/2402.17753) **[A]** | 최대 32 session의 long conversation QA·event summary benchmark | memory system 비교의 대표 benchmark지만 기반 대화가 10개뿐이고 retrieval·consolidation·answering 오류를 분리하지 못한다. |
| 2024-07 | Sun et al., [Learning to (Learn at Test Time)](https://arxiv.org/abs/2407.04620) **[B]** | sequence hidden state 자체를 TTT-Linear/MLP model로 만들고 self-supervised gradient update | per-sequence fast parametric memory. 보통 sequence/reset을 넘는 lifetime 기억은 아니다. |
| 2024-08-28 | [Graphiti open-source release](https://blog.getzep.com/graphiti-knowledge-graphs-for-agents/) **[C, Zep]** | episode→entity/fact temporal graph ingestion | 2025 Zep 논문의 제품 선행판. ingestion은 background일 수 있어도 cross-memory sleep scheduler는 아니다. |
| 2024-09 최초 / 2026-04 v4 | Ni & Liu, [Dreaming Is All You Need → SleepNet and DreamNet: Enriching and Reconstructing Representations for Consolidated Visual Classification](https://arxiv.org/abs/2409.01633) **[B, [TMLR rejected](https://openreview.net/forum?id=vXhsnUIeTl)]** | 매 forward pass의 frozen pretrained-feature fusion과 reconstruction branch | **offline sleep·replay·continual-learning 연구가 아니다.** revision 간 dataset·parameter/FLOP·결과 일관성을 주의한다. |
| 2024-09-23 | [MemGPT project → Letta company/framework naming split](https://www.letta.com/blog/announcing-letta/) **[C]** | 연구 prototype을 stateful-agent framework·company로 확장 | MemGPT 2023 논문, Letta 2025 sleep agent, 현재 Letta 제품을 같은 버전처럼 인용하면 안 된다. |
| 2024-10-14 / ICLR 2025 | Wu et al., [LongMemEval](https://arxiv.org/abs/2410.10813) **[A]** | extraction, multi-session·temporal reasoning, knowledge update, abstention을 다루는 500-question benchmark | 지속 interaction memory 평가에 유용하지만 sleep operator 자체를 격리해 측정하지는 않는다. |
| 2024-11 / ICLR 2025 | Chen et al., [Generative Adapter](https://arxiv.org/abs/2411.05877) **[A, University of Washington+Microsoft Research]** | gradient로 self-supervised pretrain/instruction-tune한 adapter generator가 context를 한 번 읽고 LoRA로 변환 | 배포 시 per-context gradient descent 없이 frozen base LM용 **context→parameter** delta를 forward 생성하는 중요한 중간점. 학습 자체가 gradient-free인 것은 아니다. |
| 2024-12 / ICML 2025 | Berges et al., [Memory Layers at Scale](https://arxiv.org/abs/2412.09764) **[A, Meta FAIR]** | sparse trainable key-value layer; 128B memory parameters, 1T-token pretraining | sparse factual capacity를 backbone FLOPs와 분리. 이후 Sparse Memory Finetuning의 substrate. |
| 2024-12 / NeurIPS 2025 | Behrouz et al., [Titans](https://arxiv.org/abs/2501.00663) **[A, Google Research]** | deep neural memory를 token/chunk마다 surprise gradient, momentum, decay로 update | wake-time neural memory 계보의 시작. 그 자체는 offline sleep이 아니다. |

### 2.5 2025: “sleep-time compute”가 이름을 얻고 두 갈래로 분화

| 시점 | 연구 | 경로 | 핵심 |
|---|---|---|---|
| 2025-01 | Rasmussen et al., [Zep / Graphiti](https://arxiv.org/abs/2501.13956) **[B, Zep]** | external text+temporal KG | raw episode, semantic entity/fact, community summary의 3층 graph; bi-temporal validity와 contradiction invalidation. |
| 2025-02 | Wang et al., [M+](https://arxiv.org/abs/2502.00592) **[A, ICML 2025]** | MEMORYLLM에서 버릴 hidden token을 CPU long-term pool에 옮기고 co-trained retriever가 GPU로 복귀 | 20K→160K+ retention을 보고. 사용자의 hot/cold memory와 CPU↔GPU data movement 질문에 직접 연결되지만 finite cap과 retrieval miss가 남는다. |
| 2025-02 / NeurIPS 2025 | Xu et al., [A-MEM](https://arxiv.org/abs/2502.12110) **[A]** | external notes+links | Zettelkasten식 note attribute/tag/link를 만들고 새 기억이 기존 기억의 context를 진화시킴. |
| 2025-04-02 | [Zep Community Edition support 종료](https://blog.getzep.com/announcing-a-new-direction-for-zeps-open-source-strategy/) **[C]** | Graphiti를 active OSS focus로 전환 | 현재는 unmaintained Zep Community repo, active Graphiti OSS, managed proprietary Zep을 분리해 봐야 한다. |
| 2025-04-17 | Lin et al., [Sleep-time Compute](https://arxiv.org/abs/2504.13171) **[B, Berkeley/Letta]** | external token-space compilation | query 전에 raw context를 `S(c)→c′`로 재표현. weight update가 아니라 prompting을 통한 learned context 생성. |
| 2025-04-21 | [Letta Sleep-Time Agents](https://www.letta.com/blog/sleep-time-compute/) **[C]** | primary는 recall/archival memory를 검색; sleep agent는 primary/shared core-memory block을 비동기 편집 | 대화 critical path와 in-context memory editing을 두 agent로 분리한다. archival store 전체를 sleep agent가 재작성하는 설계는 아니다. |
| 2025-04-28 / ECAI 2025 | Chhikara et al., [Mem0](https://arxiv.org/abs/2504.19413) **[A, Mem0]** | external text/vector, 선택적 graph | extraction 후 top-similar memory에 ADD/UPDATE/DELETE/NOOP; periodic conversation-summary refresh는 background 작업. |
| 2025-04 / ICLR 2026 | Behrouz et al., [Miras](https://arxiv.org/abs/2504.13173) **[A, Google Research]** | wake parametric | memory architecture·attentional bias·retention·learning algorithm의 design space. |
| 2025-05 / ICML 2026 | Behrouz et al., [Atlas](https://arxiv.org/abs/2505.23735) **[A, Google Research]** | wake parametric | windowed Omega objective, Muon inner optimizer, associative-memory capacity 이론. |
| 2025-06-06 | Eyuboglu et al., [Cartridges](https://arxiv.org/abs/2506.06266) **[B]** | latent KV compilation | corpus별 작은 KV cache를 offline 학습. naive NTP 대신 synthetic conversation을 만들고 context distillation. |
| 2025-06-12 / NeurIPS 2025 | Zweiger et al., [SEAL](https://arxiv.org/abs/2506.10943) **[A]** | persistent weight update | model이 finetuning data·augmentation·hyperparameter를 포함한 self-edit를 생성; SFT 후 성능을 RL reward로 사용. |
| 2025-06-29 / v2 2025-08 | Golden et al., [Interleaved Replay of Novel and Familiar Memory Traces During Slow-Wave Sleep](https://doi.org/10.1101/2025.06.25.661579) **[B, bioRxiv]** | phase-separated biological replay | biophysical modeling과 mouse retrosplenial single-unit 분석을 결합한 SCoRe 가설. novel replay는 slow-wave 전환부, familiar replay는 Up-state 중간에 나타나지만 아직 비동료심사이며 AI continual-learning 성능을 직접 시험하지 않았다. |
| 2025-07-07 / ICLR 2026 | Hu et al., [MemoryAgentBench](https://arxiv.org/abs/2507.05257) **[A]** | incremental memory benchmark | accurate retrieval, test-time learning, long-range understanding, selective forgetting을 분리해 평가. offline scheduling 자체는 측정하지 않는다. |
| 2025-08 | Li et al., [MyGO](https://arxiv.org/abs/2508.21296) **[B]** | replay→core weight | wake에서 compact generative memory를 학습하고 sleep에서 pseudo-data를 생성해 knowledge distillation. 작은 continual benchmarks가 중심. |
| 2025-08 작성 / 09 게시 | Tutuncuoglu, [Dream-Augmented Neural Networks](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5402490) **[D]** | latent replay/restructure | raw data 없이 periodic synthetic dream을 한다는 주장. 제목은 “Zero Forgetting”이지만 초록은 vision/NLP sequential benchmark에서 최대 60% forgetting 감소만 주장한다. SSRN 20쪽 단독저자 manuscript이고 “IEEE”는 입력된 affiliation일 뿐 IEEE 출판 논문이 아니며, 공개 record에 구체 dataset·code·독립 재현이 없다. |
| 2025-09-19 OpenReview / 2026-06-02 arXiv v1 / 2026-07-10 v2 | Behrouz et al., [Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories](https://arxiv.org/abs/2606.03979) **[B, Google Research+Cornell]** | parametric sleep | CMS fast→slow expert distillation+RL imitation, synthetic dream+LoRA+ReST^EM. 현재 공개 acceptance는 확인되지 않았다. |
| 2025-09-19 OpenReview / 10-16 arXiv | Lin et al., [Sparse Memory Finetuning](https://arxiv.org/abs/2510.15103) **[B, Meta FAIR]** | sparse parametric memory | 새 batch에 특이적으로 활성화되는 memory slot만 gradient update; dense FT/LoRA보다 기존 능력 손실 감소. ICLR 2026 제출은 [desk rejected](https://openreview.net/forum?id=LGo7U1m24L)이므로 preprint로 취급. |
| 2025-09-20 OpenReview / 2026-02-27 arXiv / ICML 2026 | [Memory Caching](https://arxiv.org/abs/2602.24281) **[A, Google Research+Cornell+USC]** | recurrent neural-memory state의 checkpoint/cache를 선택적으로 보존·혼합 | cached segment-state 수에 따라 capacity가 늘고 `O(NL)` compute와 교환되며 storage tiering에 직접 연결된다. |
| 2025-10 / ICLR 2026 | Zhang et al., [Agentic Context Engineering](https://arxiv.org/abs/2510.04618) **[A, Stanford/SambaNova/UC Berkeley]** | external evolving playbook | generation→reflection→curation의 incremental update로 brevity bias와 context collapse를 줄임. Microsoft 연구가 아니다. |
| 2025-11 | [HaluMem](https://arxiv.org/abs/2511.03506) **[B]** | external-memory safety | extraction·update·QA hallucination이 장기간 누적·전파되는 문제를 측정. |
| 2025-11-10 / ICLR 2026 | Behrouz et al., [TNT](https://arxiv.org/abs/2511.07343) **[A, Google Research+USC]** | parametric systems | local/global memory hierarchy, reset, train-big/serve-small chunking으로 neural-memory 학습을 최대 17배 가속. Nested Learning의 필수 선행 단계가 아니라 병렬적인 training-efficiency branch다. |
| 2025-11 공개 / 12 arXiv | Behrouz et al., [Nested Learning](https://arxiv.org/abs/2512.24695) **[A, Google Research, NeurIPS 2025]** | multi-timescale parametric | model+optimizer를 서로 다른 update frequency의 associative memories로 보고 Continuum Memory System을 제안. |

#### Letta 원 논문의 정확한 주장

- sleep-time은 query가 오기 전 context를 미리 생각하는 compute다.
- Stateful GSM-Symbolic와 Stateful AIME에서 비슷한 accuracy에 필요한 live test-time token을
  약 5배까지 줄였고, sleep compute scaling으로 각각 최대 13%, 18% accuracy gain을 보고했다.
- multi-query 환경에서 같은 context에 대한 sleep 결과를 재사용할 때 평균 query cost를
  2.5배 낮췄다.
- 논문의 직접 평가는 modified Stateful GSM-Symbolic/AIME와 SWE-Features의
  **modified-file-set F1**이다. 후자는 실제 test-suite pass rate가 아니며, conversational
  lifelong-memory 개선은 별도의 Letta 제품 보고 **[C]**와 구분해야 한다.
- 그러나 효과는 **미래 query의 예측 가능성**과 **같은 context의 재사용 횟수**에 의존한다.
  높은 live budget에서는 baseline이 다시 앞설 수 있다.
- 따라서 “항상 2–3배 싸고 정확하다”가 아니라, `offline cost / expected reuse`가 live saving보다
  작을 때만 이득이다.

#### Zep, Mem0는 어디까지 sleep인가

- **Zep/Graphiti**: ingest 때 LLM으로 entity·fact를 추출·resolve하고, 모순된 temporal edge를
  삭제하지 않고 invalidation한다. community는 개발자가 먼저 `build_communities()`를
  실행하고 episode 추가 시 `update_communities=True`를 지정해야 incremental update되며,
  문서도 periodic developer-run rebuild를 권한다. 이것은 강한
  **external consolidation operator**지만 OSS core에는 자동 sleep scheduler가 없고 model
  weight도 고정이다. 2026 managed product의 Observations가 여러 graph evidence를 durable
  pattern으로 background roll-up하고 old observation을 supersede/retire하지만, 이는 2025
  논문에 없던 proprietary 기능이다. 공개 자료로는 explicit idle/periodic schedule보다
  background 또는 ingest-adjacent cross-episode synthesis로 한정해 해석해야 한다.
- **Mem0**: 현재 message pair, 최근 대화, conversation summary에서 후보 기억을 뽑고, 유사
  기억과 비교해 ADD/UPDATE/DELETE/NOOP를 고른다. periodic summary refresh는 sleep-like
  background task다. 역시 learned consolidator training이나 weight update는 없다.
  단, 이것은 2025 논문 기준이다. 현재 V3 문서의 background ingestion은 새 입력을 비동기로
  index하는 것이지 과거 bank 전체를 다시 통합하는 sleep이 아니다.
- 양쪽의 headline benchmark는 각각 다른 model·prompt·retrieval budget의 영향을 받으므로
  “graph가 항상 vector보다 우월”하다는 증거가 아니다. Mem0 자체 결과에서도 graph variant는
  temporal query에 유리하지만 single/multi-hop에서 항상 base를 이기지 않는다.

#### Meta Sparse Memory Finetuning의 정확한 위치

이 연구가 블로그에서 말한 “Meta의 연구”에 가장 잘 부합한다.

- base model에 sparse key-value memory layer를 넣는다.
- 전형적 설정은 1M slot, token당 top-k 32, 4 heads, value dimension 1024이다.
- 새 batch에서 어떤 slot이 많이 사용되는지 세고, 1,000개 DCLM background batch 대비
  TF-IDF가 높은 top-`t` dynamic slot만 갱신한다. 나머지 model과 memory는 동결한다.
- 새 사실 습득 수준을 맞췄을 때 NaturalQuestions F1 하락이 full FT 89%, LoRA 71%,
  SMF 11%라고 보고한다.
- 실험에서는 memory-augmented model의 한 middle FFN을 1M-slot layer로 교체하고 value slot을
  선택적으로 finetune한다. 1,000개 TriviaQA fact와 1,824 document chunk를 순차 학습했으므로
  “반복 update를 전혀 시험하지 않았다”는 평가는 틀리다. 다만 heterogeneous multi-task
  lifetime cycle, slot collision·exhaustion의 포화, consolidation/eviction policy는
  검증하지 않았다. 즉 **저간섭 쓰기 방식**이지 완성된 sleep lifecycle은 아니다.

### 2.6 2026: consolidation policy를 “학습”하고 용량·인프라를 전면화

| 시점 | 연구 | 기억 위치 / 학습법 | 이번 연구에서의 중요성 |
|---|---|---|---|
| 2026-01 / ACL 2026 | [AgeMem](https://arxiv.org/abs/2601.01885) **[A]** | external STM/LTM operation을 policy action으로 만들고 3-stage progressive RL + step-wise GRPO | store/retrieve/update/summarize/discard를 heuristic이 아니라 학습된 controller로 옮김. **runtime은 wake/online이며 sleep이 아니다.** |
| 2026-02 | [Doc-to-LoRA](https://arxiv.org/abs/2602.15902) **[B]** | hypernetwork가 document를 한 번 읽고 LoRA 생성 | per-document gradient distillation의 latency를 한 forward pass로 줄인 parametric compiler. |
| 2026-02-03 / ICML 2026 | Xia et al., [Memora](https://arxiv.org/abs/2602.03315) **[A, Microsoft]** | primary abstraction이 concrete values를 index하고 cue anchors가 검색 경로 확장 | abstraction의 규모 이득과 specificity 손실을 한 구조 안에서 다룸. |
| 2026-02-12 / 03-16 / 04-06 | Letta [Context Repositories](https://www.letta.com/blog/context-repositories/) → [Next Phase](https://www.letta.com/blog/our-next-phase/) → [Code App](https://www.letta.com/blog/introducing-the-letta-code-app/) **[C]** | server sleep agent를 client-side subagent+git-backed MemFS/context repository로 교체하는 중 | 현재 제품은 2025 server sleep-agent 그대로가 아니다. memory subagent가 session을 주기적으로 review/rewrite/refine하는 `dreaming`은 남고 git history가 provenance/rollback을 제공한다. |
| 2026-04 | [Mem0 V3](https://docs.mem0.ai/migration/platform-v2-to-v3) **[C]** | one-pass ADD-only extraction; overwrite/delete 없음 | queued/background 작업도 새 입력의 per-add indexing이지 cross-bank sleep이 아니다. bank는 자라며 expiry·periodic cleanup이 필요하고, 기존 memory를 자동 재처리하지 않는다. 2025 ECAI operator와 분리해야 한다. |
| 2026-04 | Shinde, [Sleep-Consolidated Memory](https://arxiv.org/abs/2604.20943) **[B, research preview]** | importance, NREM/REM, value-based forgetting | 명시적 LLM sleep prototype이나 단독 research preview·작은 평가라 핵심 근거로 쓰기 이르다. |
| 2026-04 | Pan et al., [M★: Every Task Deserves Its Own Memory Harness](https://arxiv.org/abs/2604.11811) **[B]** | schema·storage logic·workflow instruction을 Python memory program으로 만들고 population/reflection으로 진화 | “하나의 범용 기억 정책” 대신 task별 memory system 자체를 offline search. |
| 2026-04 | Hu et al., [When Continual Learning Moves to Memory](https://arxiv.org/abs/2604.27003) **[B]** | external experience representation·organization 실험 | **외부기억도 stability–plasticity를 없애지 않고 retrieval competition으로 옮긴다**는 핵심 반증. |
| 2026-05 | Kerestecioglu et al., [Human-Inspired Memory Architecture for LLM Agents](https://arxiv.org/abs/2605.08538) **[B, Microsoft]** | 6시간 주기 hot cache→episodic vector→semantic KG consolidation; dedup, TTL/interference forgetting, maturation, reconsolidation | Microsoft에서 **명시적으로 sleep-phase**를 부른 외부기억 연구. weight update는 아니다. |
| 2026-05 | Ye et al., [Auto-Dreamer](https://arxiv.org/abs/2605.20616) **[B]** | provenance-linked typed bank를 offline consolidator가 compact replacement로 변환; GRPO | 외부기억 계보에서 가장 직접적인 learned sleep consolidator. downstream agent reward로 학습. |
| 2026-05 | [Zep Observations](https://help.getzep.com/observations) **[C]** | graph evidence에서 durable pattern/decision/state-transition을 background/ingest-adjacent synthesis, supersede/retire | cross-episode 기능적 sleep 등가지만 explicit idle/periodic scheduler의 공개 근거는 없고, proprietary operator이며 독립 평가가 없다. 2025 Zep 논문/Graphiti OSS에는 없던 기능. |
| 2026-05 | CMU/UMD, [Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference](https://arxiv.org/abs/2605.26099) **[B]** | recent context→fixed-size fast weights, KV clear, offline recurrent update `N`회 | 같은 input sequence의 context-window 경계를 넘어 fast state를 유지하는 parametric consolidation. sequence 시작에는 zero-initialize되므로 cross-session/lifetime memory는 아니다. |
| 2026-06 | Hardalov et al., [Cartridges at Scale](https://arxiv.org/abs/2606.04557) **[B]** | document별 KV cartridge, distractor mixing, GPU↔persistent-store budget manager | latent memory의 composability와 offload를 실제 systems 문제로 전환. |
| 2026-06-07 | Bazhenov, Delanois & Krishnan, [Not Just After One: Sleep-Inspired Replay Prevents Catastrophic Forgetting After Sequential Tasks](https://arxiv.org/abs/2606.08447) **[B]** | 여러 task를 학습한 뒤 한 번의 SRC sleep phase | Tadros et al. 2022 *Nature Communications* ANN-SRC 계열의 후속. 모든 과제를 부분 복원하지만 task 수가 늘수록 평균 성능은 하락하며 LLM 증거는 아니다. |
| 2026-06 | Yang et al., [TrustMem](https://arxiv.org/abs/2606.25161) **[B]** | coverage·preservation·faithfulness verifier, preference pair, preference-guided RL | runtime은 chronological chunk마다 하는 online update로 **sleep은 아니지만**, consolidation transition의 왜곡·환각을 학습 목표로 만든다. |
| 2026-06 | Letta, [Memory Models](https://www.letta.com/blog/towards-agents-that-learn/) **[C]** | token-space memory model을 future-task reward의 meta-RL로 학습한다는 연구 방향 | 논문 결과가 아니라 **industry thesis**. token memory→harness/weight distillation까지 사용자의 hybrid 축과 일치. |
| 2026-07 | Elmieh et al., [NSTM](https://arxiv.org/abs/2607.15271) **[B, Google]** | 비싼 neural-memory update와 빠른 per-frame application의 frequency 분리 | CV 논문이지만 wake/sleep cluster의 update/apply decoupling에 강한 infra analogue. |
| 2026-07 | Kang et al., [Retain or Consolidate?](https://arxiv.org/abs/2607.17545) **[B]** | Retain vs Merge/Abstract/Rewrite를 memory budget에 따라 선택; Offline Abstraction-Safety learner | runtime은 query-specific packing이며 persistent sleep은 아니다. 다만 tight/loose budget crossover의 가장 직접적인 theory. |

최초 공개일과 개념 의존 순서는 완전히 같지 않다. 예를 들어 Google Sleep의 초기 OpenReview
원고는 2025-09에 있었고 Nested Learning의 공식 공개는 그 뒤지만, 현재 Sleep 판본의 구조는
Nested Learning/CMS를 명시적으로 상속한다. 따라서 arXiv 번호만 보고 `Nested Learning →
Sleep`의 아이디어 발생 순서를 단정하지 말고, **판본별 공개일과 현재 방법의 dependency**를
각각 기록해야 한다.

### 2.7 2026년의 가장 직접적인 네 구현

#### A. Auto-Dreamer — 외부기억의 learned offline consolidation

1. frozen writer가 session마다 semantic/procedural text와 source trajectory provenance를 typed
   bank에 append한다.
2. 매 `N` session마다 새 entry와 그 기간에 retrieve된 과거 entry로 working region을 만든다.
3. Qwen3-14B consolidator가 bank와 raw trajectory를 bounded tool-use로 조사하고, 여러 session에
   공통인 procedure를 **fresh compact replacement set**으로 쓴다.
4. working region은 교체되지만 retired predecessor는 log/provenance에 남긴다.
5. ScienceWorld trajectory만으로 GRPO를 학습하며 end-to-end task reward와 random-mask
   counterfactual utility, redundancy penalty를 사용한다.

ScienceWorld에서 강한 baseline보다 약 7 point 높고 active bank가 12배 작았으며,
ALFWorld/WebArena에 재학습 없이 transfer했다고 보고한다. 그러나 raw log를 남기므로
**active bank 감소가 total storage 감소와 같지는 않다**. game stream을 개인 대화·기업
knowledge에 일반화한 증거도 아직 없다.

#### B. Offline Recurrence — context를 persistent fast weight로

`Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference`는 KV eviction
직전에 attention+Gated-Delta fast-weight block가 최근 context를 `N`회 재순환해 fixed-size
fast weight에 굳히고, 이후 KV를 비운다. training은 window와 recurrence depth를 통과해
local update rule을 학습한다. `N`을 늘리면 어려운 synthetic graph/GSM-Infinite류 task가
개선되지만 비용이 거의 선형으로 늘며, target은 global backbone이 아니라 **같은 input
sequence의 context-window 경계를 넘어 유지되는 fast state**다. 매 sequence 시작에는
zero-initialize하므로 user/session lifetime memory로 일반화하면 안 된다.

#### C. Google Sleep — fast→slow parameter와 synthetic dream

sleep은 두 stage다.

1. **Knowledge Seeding / NREM 대응**
   CMS의 빠른 block이 덮어써지기 전에 더 느린 block에 새 low-rank MoE expert를 활성화한다.
   pre-expansion smaller self가 teacher corpus를 sampling하고, generalized on-policy knowledge
   distillation과 RL-based Learning to Imitate로 새 expert만 학습한다. 완료 뒤 빠른 block의
   과거 sleep expert를 reset/prune한다.
2. **Dreaming / REM 대응**
   model이 synthetic dream을 생성할 때 각 router에 random expert를 추가 활성화해 기존
   기억의 낯선 조합을 만든다. dream별 SFT gradient로 importance를 매겨 top-k+random
   sample을 고르고, 각 후보를 isolated LoRA instance에 학습한다. downstream metric이
   실제로 개선되면 dream 생성에 binary reward를 주고 ReST^EM으로 policy를 갱신한다.

이것은 distillation, RL, data augmentation, parameter growth를 한 lifecycle에 넣은 가장
직접적인 논문이다. 동시에 다음 공백도 크다.

- sleep-aware하게 처음부터 end-to-end co-training한 것이 아니라 pre-trained
  Llama/Qwen에 알고리즘 wrapper를 graft한 실험이 중심이다.
- sleep 시점과 CMS frequency, rank, dream 수는 hand-set이다.
- task-free deployment에는 downstream metric `τ`가 없는데 dream reward가 무엇이어야
  하는지 해결되지 않았다.
- 새 expert는 masked pre-allocation으로 tensor shape를 고정하지만 pool은 유한하다.
- 수백 번의 wake/sleep, router의 새 expert 적응, full wall-clock·energy·bytes moved,
  self-modification safety가 검증되지 않았다.

#### D. Microsoft Human-Inspired Memory — 명시적이지만 non-parametric

6시간마다 hot event cache를 episodic vector store와 semantic graph로 옮기며 deduplication,
importance, TTL/interference forgetting, engram maturation, retrieval-triggered reconsolidation을
적용한다. threshold는 benchmark에 직접 fit하지 않는 synthetic calibration으로 정한다.
13K VSCode issue/120K event에서 **dedup 기반 consolidation**이 store 58% 감소와 97.2%
retention precision을 냈다고 보고한다. ablation evidence가 있는 것은 네 개 mechanism이고,
maturation·reconsolidation은 benchmark에 반복 retrieval/cross-session contradiction이 없어
아직 design rationale에 가깝다. 이는 “Microsoft도 sleep 관련 연구가 있는가?”에 대한
**예**이지만, model이 distillation/RL로 자기 weight를 배우는 사례는 아니다.

### 2.8 낮은 증거 등급의 side branch

핵심 읽기 목록에는 넣지 않되, “sleep”이라는 이름을 쓴 연구의 범위를 놓치지 않기 위해
다음도 ledger에 남긴다.

| 시점 | 연구 | 판정 |
|---|---|---|
| 2025-06-23 / 2025-10-16 철회 | [Sleep Enhanced Latent Replay for SNNs](https://arxiv.org/abs/2507.02901) **[withdrawn]** | compressed binary latent replay+noisy sleep 아이디어. 저자가 data error와 추가 실험 필요를 밝히며 철회했으므로 기존 정확도·메모리 수치를 근거로 쓰지 않는다. |
| 2025-08 | [Sleep-like and Awake Rehearsal for Equilibrium Propagation](https://arxiv.org/abs/2508.14081) | SRC와 awake rehearsal을 결합한 preprint. long lifetime capacity·독립 재현 없음. |
| 2026-03 | [Slumbering to Precision](https://arxiv.org/abs/2603.07867) | sleep-like post-processing으로 calibration 개선을 주장. 장기기억·catastrophic forgetting 연구는 아니다. |
| 2026-03 | [Learning to Forget](https://arxiv.org/abs/2603.14517) | entropy-triggered microcycle이 KV를 tag/forget/summarize. 793K-parameter toy transformer와 synthetic retrieval에 한정. |
| 2026-06 | [Sleep to Forget](https://www.biorxiv.org/content/10.64898/2026.06.15.732460v1) | slow-wave dynamics가 consolidation과 forgetting을 조절한다는 생물물리 계산 preprint. 실험 데이터가 아니다. |

---

## 3. 세 계보를 같은 표로 비교

아래 표는 진짜 sleep뿐 아니라 sleep system 설계에 필요한 adjacent consolidation operator도
함께 놓는다. `runtime` 열에서 구분한다.

| 연구 | wake에서 모으는 것 | consolidation / adjacent operator | 쓰는 곳 | 학습 신호 | runtime | 가득 찰 때 |
|---|---|---|---|---|---|---|
| PLOS 2022 | 새 task 감각·보상 stream; 과거 raw-trace buffer 없음 | spontaneous H→O replay + unsupervised STDP | H→O synaptic weights | local plasticity | 명시적 offline | old trace가 지워지기 전에 자야 함; 성장 없음 |
| PAD 2022 | episodic latent | perturb/reconstruct; mix+adversarial dream | encoder/generator weight | reconstruction+GAN | 명시적 NREM/REM | 명시적 lifetime budget 없음 |
| MemGPT 2023 | message/tool history | pressure summary | text DB + core context | prompt heuristic | wake/online | bounded main FIFO→recursive summary; raw recall은 무기한 보존되고 archival cap/eviction은 미정 |
| Zep 2025 | episode/entity/fact | resolve, dedup, temporal invalidation, optional community summary | temporal graph | prompted LLM | ingest-time; community는 opt-in/수동 rebuild | hard graph capacity/automatic forgetting 없음; raw episode와 invalidated fact도 보존 |
| Letta 2025 | persistent context/docs | anticipatory rewrite / memory editing | token-space core+external | prompting | 명시적 async sleep | revision; formal eviction theory 없음 |
| Mem0 2025 | message pair + recent history | extract, ADD/UPDATE/DELETE/NOOP, summary refresh | vector text / graph | prompted LLM | hot-path + async summary | 2025 논문에 hard cap 없음; V3는 ADD-only 성장+expiry/manual cleanup |
| Cartridges 2025 | corpus | synthetic QA + context distillation | reusable KV cache | teacher ICL | offline corpus compilation | corpus별 fixed cartridge; CAS에서 modular offload |
| SEAL 2025 | knowledge/task examples | self-edit→SFT | model weight | post-update downstream reward | update-time inner loop | 반복 update의 forgetting 문제 |
| Meta SMF 2025 | new fact batch | TF-IDF-selected sparse gradient update | memory slots | QA/task loss | continual update; sleep 미정 | slot pool 유한; eviction 미해결 |
| AgeMem 2026 | task trajectory | learned memory-tool actions | STM/LTM text store | step-wise GRPO | wake/online | LTM Add/Update/Delete, STM Summary/Filter; hard LTM cap/full trigger 없음 |
| Microsoft 2026 | hot event cache | 6-hour dedup/forget/mature/reconsolidate | vector store + semantic KG | calibrated thresholds | 명시적 periodic sleep | store reduction, TTL/interference |
| Auto-Dreamer 2026 | cross-session typed bank | inspect provenance, synthesize compact replacements | external procedural memory | end-to-end task reward, GRPO | 명시적 offline | old region을 fresh compact set으로 대체 |
| Offline Recurrence 2026 | 한 input sequence의 recent context | learned recurrent passes | sequence-scoped fast weight | meta-learned local update | window 사이 periodic offline | KV clear; fast-weight capacity 유한, 새 sequence에서 reset |
| Google Sleep 2026 | fast CMS knowledge | upward distill+RL imitate; synthetic dreams+LoRA/RL | new slow MoE expert + model | GKD, semantic/edit reward, downstream reward | 명시적 offline | fast expert reset, slow expert growth; pool은 결국 유한 |
| TrustMem 2026 | old state + candidate update | verify/rank Write/Revise/Prune/no-action candidates | external memory | preference-guided RL | chunk-online; sleep 아님 | learned Prune+bloat penalty; hard cap·version/provenance·offline scheduler는 없음 |
| OAS 2026 | raw memory set + budget | retain/merge/abstract/rewrite selection | query context pack | held-out harm calibration | query-time; sleep 아님 | query-pack budget만 다루며 persistent lifecycle/eviction은 없음 |

---

## 4. Sleep-time training은 실제로 무엇을 학습하는가

### 4.1 Replay와 distillation

가장 보수적인 방법은 과거 표본 또는 그 압축본을 다시 보여 주는 것이다.

- **raw replay**: 가장 충실하지만 storage/privacy가 가장 비싸다.
- **latent replay**: REMIND, DANN 주장, MyGO처럼 feature/generative model만 남긴다.
- **teacher sampling**: Google Sleep의 Knowledge Seeding처럼 pre-expansion self가 corpus를
  생성한다. raw history를 보존하지 않아도 되지만 teacher의 오류와 망각도 복제한다.
- **context distillation**: Cartridges가 full-context teacher의 기능을 synthetic dialogue로
  작은 KV state에 옮긴다.
- **cross-session abstraction**: Auto-Dreamer가 여러 trajectory에서 공통 procedure를 추출한다.

핵심 설계 질문은 “무엇을 replay할까?”보다 **어떤 future query distribution을 보존하도록
replay data를 만들까?**다. uniformly replay하면 희귀하지만 중요한 사건을 놓치고, importance
sampling은 잘못된 중요도 estimator를 영구화할 수 있다.

### 4.2 Synthetic dream / dataset augmentation

| 방식 | synthetic data 생성 | 학습 목적 | 위험 |
|---|---|---|---|
| PAD REM | episodic latent 혼합 + spontaneous noise | semantic separation | generator bias, 이미지 한정 |
| PAD NREM | replay latent에서 occluded reconstruction | robustness | 특정 augmentation에 과적합 |
| MyGO | task별 generative memory에서 pseudo-example | old-task retention | generative drift |
| SEAL | model이 self-edit와 augmentation directive 생성 | post-update 성능 | self-training collapse, expensive inner SFT |
| Google Dreaming | random expert routing으로 기억 조합, gradient filter | novel curriculum, capability refinement | 무관한 기억 결합, reward hacking |
| Auto-Dreamer | provenance를 읽고 compact procedural replacement 생성 | future agent success | detail loss, source bias |

좋은 dream dataset에는 최소한 다음이 필요하다.

1. **coverage**: 최근 경험뿐 아니라 rare/old/negative case를 포함한다.
2. **novel recombination**: 단순 복사 외에 counterfactual·cross-episode composition을 넣는다.
3. **grounding**: 원본 provenance 또는 simulator/tool outcome으로 검증한다.
4. **anti-collapse mixture**: raw, teacher, student-on-policy, adversarial, random sample을 섞는다.
5. **holdout future task**: dream 자체의 likelihood가 아니라 sleep 후 미래 성능으로 평가한다.

### 4.3 RL

sleep policy의 RL은 세 층으로 나뉜다.

- **content policy**: 어떤 memory/dream을 만들지. SEAL과 Google Dreaming.
- **operator policy**: retain/merge/abstract/rewrite/delete 중 무엇을 할지. AgeMem, OAS.
- **transition-quality policy**: update가 coverage·preservation·faithfulness를 지켰는지.
  TrustMem.

가장 중요한 reward는 현재 sleep output의 “그럴듯함”이 아니라 **다음 여러 task에서의
downstream utility**다. 단일 next-task reward는 미래를 아는 training 환경에서는 가능하지만,
실제 배포에서는 delay가 길고 non-stationary하며 사용자별이다. off-policy evaluation,
counterfactual replay, verifier reward, human correction을 함께 써야 한다.

### 4.4 Sparse / isolated parametric update

catastrophic forgetting을 낮추는 공통 원리는 공유 parameter 전체를 매번 움직이지 않는 것이다.

- EWC/MAS: 중요한 기존 weight의 이동을 regularize.
- LoRA/adapter: user/task별 delta를 격리.
- SMF: 새 정보에 특이적으로 활성화된 memory row만 갱신.
- Google Sleep: 새 low-rank expert만 활성화해 upward distillation한 뒤 fast expert reset.
- Doc-to-LoRA/Generative Adapter: 매번 optimization하지 않고 hypernetwork가 delta를 생성.

하지만 격리는 capacity 문제를 없애지 않는다. adapter·slot·expert의 수가 늘면 routing,
residency, merge, stale version, privacy deletion이 새 병목이 된다.

---

## 5. “계속 배우면 결국 기억용량을 넘는다”에 대한 답

### 5.1 용량은 하나가 아니라 네 가지다

1. **storage capacity**: bytes, vector count, graph edge, KV slot, adapter/expert 수.
2. **retrieval/application bandwidth**: 다음 wake에서 context·HBM·latency budget 안에 실제로
   읽을 수 있는 양. 외부에 무한히 저장해도 이것이 작으면 기억하지 못한 것과 같다.
3. **representational/interference capacity**: 같은 weight·slot·summary가 몇 개의 pattern을
   분리해 보존할 수 있는가.
4. **governance capacity**: provenance, consent, TTL, unlearning, conflict, access control,
   rollback을 관리할 수 있는 양.

`When Continual Learning Moves to Memory`의 핵심은 외부기억이 2번을 해결하지 못한다는 것이다.
제한된 context 안에서 old/new memory가 retrieval을 두고 경쟁하므로 stability–plasticity
문제가 **weight interference에서 retrieval interference로 이동**한다.

### 5.2 외부기억의 capacity mechanism

| 메커니즘 | 지키는 것 | 잃을 수 있는 것 |
|---|---|---|
| admission / novelty gate | 쓰기 폭주 방지 | 처음엔 사소해 보인 미래 핵심 정보 |
| TTL / learned expiration | stale 정보와 비용 감소 | 장기 선호·희귀 사건 |
| dedup / entity resolution | 중복 제거 | 서로 다른 맥락을 같은 사실로 합칠 위험 |
| temporal invalidation | history와 최신값 동시 보존 | graph 복잡도 |
| merge / abstraction | token budget당 coverage | exact detail |
| rewrite | 한 note의 최신성 | 반복 rewrite에 의한 context collapse |
| provenance-linked raw archive | audit·재구축·rollback | cold storage와 privacy 비용 |
| hot/warm/cold tiering | live latency와 storage 분리 | promotion/demotion policy 오류 |
| learned retrieval/organization | task별 utility | negative transfer와 feedback loop |

OAS가 보인 중요한 crossover는 “항상 요약”도 “항상 raw retain”도 틀렸다는 것이다.
budget이 넉넉하면 exact record가 낫고, 빡빡하면 cross-note merge/abstract가 omitted evidence의
coverage를 늘린다. 따라서 consolidation trigger는 시간만이 아니라 **relative memory
pressure**의 함수여야 한다.

### 5.3 모수적 기억의 capacity mechanism

| 메커니즘 | 예 | 남는 문제 |
|---|---|---|
| replay/distillation | PLOS, MyGO, Cartridges, Google Sleep | teacher 오류·data selection |
| regularization | EWC, MAS | plasticity 감소; 새 capacity 없음 |
| gradient constraint | GEM | replay buffer와 계산 증가 |
| sparse write | SMF | slot collision/exhaustion, routing |
| adapter/expert isolation | LoRA, Google Sleep | adapter/expert 수의 무한 증가 |
| fast→slow hierarchy | CLS, CMS, offline recurrence | promotion schedule·lossy abstraction |
| structural growth | Google Sleep low-rank experts | preallocated pool도 결국 고갈 |
| stochastic replacement | MEMORYLLM latent-token random drop | 중요한 희귀 기억을 임의로 잃을 수 있음 |
| prune/reset | TNT, Google Sleep | 무엇이 안전하게 이전됐는지 검증 필요 |
| cache/checkpoint/offload | Memory Caching, M+, CAS | storage traffic, retrieval miss와 stale cache |

Google Sleep은 catastrophic forgetting을 capacity 문제로 재정의하고 새 expert를 활성화하지만,
그 논문도 **무한 용량을 제공하지 않는다**. prospective experts를 미리 할당해 mask했다가
활성화하므로 tensor shape는 안정적이지만 pool이 다 차면 다시 merge/prune/grow 결정을 해야
한다. 수백 wake/sleep cycle에서의 안정성, pool exhaustion, consolidation당 필요한 capacity에
대한 이론은 열려 있다.

### 5.4 권장되는 hybrid policy

새 사건 `e`를 바로 weight에 쓰지 않고 다음 순서로 다룬다.

```text
1. capture e with provenance, timestamp, privacy scope
2. estimate novelty × importance × expected reuse × confidence
3. retain exact externally while evidence is weak
4. merge/abstract only when budget pressure or repeated pattern is observed
5. compile to KV/adapter when a stable corpus is queried repeatedly
6. promote to slow weight only after cross-session reuse and regression checks
7. keep a reversible external source or tombstone for every promotion
8. demote, invalidate, or block stale/conflicting memories at retrieval
```

즉, **외부기억은 evidence accumulator**, **latent/adapter는 serving cache**, **slow weight는
high-confidence high-reuse compiled knowledge**로 보는 것이 안전하다.

---

## 6. 회사별 실제 연구 지도

### 6.1 Google / DeepMind

직접적인 한 줄은 다음이다.

```text
CLS update / EWC
  → TTT-like neural memory
  → Titans → {Miras, Atlas}
  → Nested Learning / CMS
  → Language Models Need Sleep (parametric consolidation + dreaming)

parallel systems branches: {TNT, Memory Caching, NSTM}
```

- 2016 CLS update는 fast episodic/slow structured learning의 이론적 다리다.
- Titans–Miras/Atlas–Nested Learning은 wake 동안 parameter/state를 서로 다른 frequency로
  업데이트하는 계보다. TNT·Memory Caching·NSTM은 training efficiency, checkpoint/cache,
  update/apply frequency를 다루는 병렬 systems branch다.
- 2026 Google Sleep은 fast memory를 새 slow MoE expert로 upward distill하고, 별도 dreaming
  loop로 self-generated data를 학습한다.
- Memory Caching과 NSTM은 update frequency, cache, offload, application frequency라는 infra
  문제를 전면화한다.

여기서 소속을 섞지 않아야 한다. CLS update와 EWC의 초기 basis는 Google DeepMind이지만,
Titans→Miras/Atlas→Nested Learning→현재 Sleep v2의 직접 계보는 **Google
Research-led**이며 Sleep에는 Cornell 저자도 참여한다. Sleep v1/OpenReview는 Nested
Learning 공개보다 앞서므로, 현재 v2의 명시적 dependency와 최초 아이디어 chronology를
별도로 기록해야 한다.

이 여섯 논문의 세부 연결은 기존 [Google TTT / Neural-Memory PRE-RESEARCH](./PRE-RESEARCH.md)
에 이미 심층 정리되어 있으므로, 본 문서는 외부기억 계보와의 접점에 집중한다.

### 6.2 Meta

현재 확인되는 핵심은 “sleep”이라는 이름보다 **유한 capacity에서 interference를 줄이는
sparse memory**다.

```text
GEM / MAS / Product-Key Memory / Expire-Span
  → Memory Layers at Scale
  → Sparse Memory Finetuning
```

- GEM은 episodic replay buffer로 과거 task gradient를 보호한다.
- MAS는 parameter importance에 따라 덜 중요한 지식을 선택적으로 덮어쓸 수 있게 한다.
- Product-Key Memory는 거대한 sparse KV parameter pool을 저비용 top-k lookup하는 구조를
  만들었다.
- Expire-Span은 기억의 수명을 학습한다.
- Memory Layers는 factual capacity를 sparse lookup으로 크게 늘린다.
- SMF는 새 사실에 특이적인 slot만 고쳐 dense FT보다 forgetting을 줄인다.

따라서 블로그의 Meta 부분은 **모수적 sparse memory update**의 근거로는 유효하지만,
비동기 sleep scheduler나 NREM/REM pipeline까지 Meta가 구현했다고 읽으면 과장이다.

### 6.3 Microsoft

현재 확인되는 연구 portfolio는 explicit “sleep” 명명보다 **외부·context·generated
parameter consolidation**의 병렬 branch다. 아래 항목을 직접적인 논문 계보로 읽으면 안 된다.

```text
external KV: LongMem (Microsoft Research + UCSB)
generated LoRA: Generative Adapter (UW + Microsoft Research)
harmonic external store: Memora
external sleep lifecycle: Human-Inspired Memory Architecture
product memory service: Azure Foundry Agent Memory
```

- LongMem은 frozen backbone과 decoupled long-term bank를 사용한다.
- Generative Adapter는 context를 LoRA로 compile한다.
- Memora는 abstraction과 specificity를 함께 보존하는 구조를 제안한다.
- Human-Inspired Memory Architecture는 6시간 주기의 **sleep-phase consolidation**을
  명시하고 hot cache→episodic vector store→semantic KG, dedup, TTL/interference forgetting,
  maturation, reconsolidation을 설계한다. VSCode 13,127 issues/120K events에서
  **dedup-based consolidation**이 retention precision 97.2%와 store 58% 감소를 보고했지만,
  maturation·reconsolidation은 해당 benchmark가 직접 검증하지 못했고 model weight가 아니라
  외부 store를 다룬다.
- [Microsoft Foundry Agent Memory](https://learn.microsoft.com/en-us/azure/foundry/agents/concepts/what-is-memory)
  **[C]**는 extraction→LLM consolidation→retrieval, profile/chat-summary/procedural memory,
  CRUD·TTL·remember/forget을 제품화한다.

따라서 Microsoft에는 명시적 sleep-phase 외부기억 연구가 존재한다. 다만 Letta처럼
token-space pre-query reasoning을 scaling law로 만들거나 Google처럼 deployed model의 slow
parameter에 지식을 증류·성장시키는 **parametric sleep**은 현재 확인되지 않았다.

### 6.4 Letta, Mem0, Zep

| 조직 | 현재 강점 | 아직 없는 것 |
|---|---|---|
| Letta | 2025 primary/sleep agent 분리와 token-space learned context; 2026 client memory subagent+git-backed MemFS로 provenance/rollback 강화 | 원 논문에는 weight training 없음; server sleep-agent 제품은 교체 중이며 memory-native RL은 아직 공식 vision |
| Mem0 | 2025 ECAI 논문의 extract/update/delete와 vector+graph; vendor-run 평가에서 낮은 live token/latency | memory construction cost와 live serving cost를 분리해야 함; V3 per-add background indexing은 cross-bank sleep이 아니며 ADD-only growth·cleanup 문제와 learned offline consolidator 부재 |
| Zep/Graphiti | provenance, bi-temporal KG, contradiction invalidation, opt-in community abstraction; managed Zep Observations의 background/ingest-adjacent cross-episode roll-up | Zep Community는 지원 종료, Graphiti가 active OSS; core에 scheduler가 없고 managed Observations는 proprietary·독립 ablation 부족; parametric learning 없음 |

---

## 7. 인프라로 번역

### 7.1 세 개의 plane과 하나의 control plane

```text
                         ┌──────────────────────────┐
                         │ control / policy plane   │
                         │ schedule, budget, verify │
                         │ promote, evict, rollback │
                         └────────────┬─────────────┘
                                      │
┌────────────────────┐     deltas     ▼      ┌────────────────────┐
│ wake compute plane │ ────────────────►     │ sleep compute plane│
│ low-P99 inference  │                       │ batch/replay/SFT/RL │
│ retrieve + capture │ ◄──────────────────── │ compact/distill    │
└─────────┬──────────┘   versioned artifact  └─────────┬──────────┘
          │                                             │
          └──────────────────┬──────────────────────────┘
                             ▼
                  ┌────────────────────────┐
                  │ persistent memory plane│
                  │ text/vector/graph/KV   │
                  │ adapters/checkpoints   │
                  │ hot / warm / cold      │
                  └────────────────────────┘
```

- **wake plane**: low P99, retrieval, prompt assembly, hot per-session state, write-ahead log.
- **sleep plane**: batch-friendly generation/backward/RL, large context scan, stronger/slower model,
  snapshot 기반 consolidation.
- **memory plane**: raw immutable event, versioned summary/KG, vector index, cartridge/adapter,
  optimizer/checkpoint를 서로 다른 tier에 둔다.
- **control plane**: sleep trigger, per-tenant budget, promotion, verifier, privacy, lineage,
  canary/rollback을 관리한다.

물리적으로 cluster를 반드시 세 개로 쪼개야 하는 것은 아직 가설이다. 비교해야 할 배치 방식은:

1. 같은 accelerator fleet의 시간 분할,
2. latency fleet와 batch sleep fleet의 물리 분리,
3. memory/storage 근처에서 consolidation,
4. privacy가 강한 경우 device/tenant-local sleep.

### 7.2 data movement를 줄이는 방법

- wake가 raw corpus 전체가 아니라 **append-only delta와 sufficient statistics**만 sleep에 보낸다.
- sleep output도 full checkpoint 대신 **summary delta, graph patch, KV cartridge, LoRA,
  sparse slot diff**로 반환한다.
- raw event는 cold object store, index/summary는 warm memory service, active retrieval set과
  adapter는 HBM/GPU에 둔다.
- per-user adapter를 매 query load하지 않도록 reuse/arrival rate 기반 admission cache를 둔다.
- CAS처럼 hundreds of latent memories를 persistent store와 GPU 사이에서 budget manager가
  회전시키되, cartridge selection과 retrieval miss를 함께 측정한다.
- sleep 중 읽은 snapshot version과 next-wake 적용 version을 명시해 stale write를 막는다.

### 7.3 반드시 계측할 지표

정확도 하나로는 인프라 우열을 결정할 수 없다.

| 계층 | 지표 |
|---|---|
| task utility | next-task accuracy/success, forward transfer, backward transfer, forgetting |
| live serving | P50/P95/P99, prompt tokens, decode tokens, HBM working set |
| sleep cost | FLOPs, wall-clock, energy, generated/rejected dreams, backward pass 수 |
| storage | bytes/user/day, graph/slot/adapter growth, hot/warm/cold residency |
| movement | bytes wake→sleep, sleep→memory, memory→wake; cache hit/miss |
| memory quality | retrieval recall, temporal correctness, compression coverage, detail loss |
| reliability | omission, corruption, hallucination, stale fact, conflict, rollback success |
| economics | amortized cost/query, break-even reuse count, idle-resource utilization |

기본 경제식은 다음처럼 잡을 수 있다.

```text
amortized_cost_per_query
  = live_cost(compiled_memory)
  + sleep_cost / expected_reuse_count
  + storage_and_movement_cost
  + expected_error_and_rollback_cost
```

sleep compute가 이득이려면 미래 query 예측 가능성과 reuse가 충분해야 한다. Letta가 보고한
multi-query 이득을 모든 one-off query에 일반화하면 안 되는 이유다.

### 7.4 현재 benchmark가 무엇을 측정하고 못 측정하는가

| benchmark | 장점 | sleep 연구의 빈칸 |
|---|---|---|
| [LoCoMo, ACL 2024](https://aclanthology.org/2024.acl-long.747/) | 10개 dialogue, 최대 32 session; single/multi-hop·temporal·open-domain-knowledge·adversarial QA, event summary | retrieval, memory write, consolidation, answering error가 한 점수에 섞인다. 작은 dialogue 수와 판본별 turn/token 통계 차이를 함께 봐야 한다. |
| [LongMemEval, ICLR 2025](https://openreview.net/forum?id=UBvm2bIyxz) | 500 curated questions; extraction, multi-session, temporal, knowledge update, abstention | index/retrieve/read 단계는 잘 보지만 sleep scheduler와 operator의 장기 반복 안정성은 직접 측정하지 않음. |
| [MemoryAgentBench, ICLR 2026](https://openreview.net/forum?id=DT7JyQC3MR) | incremental stream, accurate retrieval, test-time learning, long-range understanding, selective forgetting | 외부 memory controller 비교에 좋지만 async/offline 여부와 sleep compute 비용은 별도 계측 필요. |
| [HaluMem](https://arxiv.org/abs/2511.03506) | extraction/update/QA 단계별 hallucination localization, 최대 1M-token user history | memory transition의 안전성에는 좋지만 operator schedule·infra economics는 없다. 세 experimental scoring stage와 shared QA answer model에 GPT-4o를 써 system 차이와 judge/model coupling을 주의해야 한다. |

새 benchmark는 다음을 추가해야 한다.

- 같은 user/tenant가 수백 wake/sleep cycle을 거치는 **lifetime stream**,
- future query가 예측 가능한 경우와 one-off인 경우의 혼합,
- fact·preference·procedure·skill·negative feedback·privacy deletion을 함께 포함,
- store size와 live token, sleep FLOPs, bytes moved를 동시 제한,
- source 수정·철회·모순, poisoned memory, model upgrade와 adapter migration,
- “정답을 맞혔는가” 외에 어떤 기억을 잃고 왜곡했는지 transition-level audit.

---

## 8. 연구 문제와 검증 가능한 가설

### RQ1. 어떤 기억이 어느 매질로 가야 하는가?

가설: exact·low-confidence·privacy-sensitive 정보는 external에, high-reuse procedure는
adapter/weight에 두는 learned router가 한 가지 매질보다 낫다.

반증 조건: 동일 storage/FLOPs/latency에서 단일 external 또는 단일 parametric baseline이
hybrid를 지속적으로 이긴다.

### RQ2. sleep trigger는 주기인가, pressure인가, surprise인가?

가설: fixed nightly schedule보다 `memory pressure × novelty × expected reuse × current load`를
사용한 event-driven scheduler가 낫다.

반증 조건: learned trigger의 overhead·instability가 단순 fixed cadence 이득을 넘지 못한다.

### RQ3. NREM/REM 분리가 실제로 필요한가?

가설: faithful replay/verification(NREM) 뒤에 novel recombination(REM)을 수행해야
catastrophic forgetting과 self-training collapse를 동시에 낮춘다.

필수 ablation: replay only, dream only, dream→replay, replay→dream, raw replay,
latent replay, random augmentation을 compute-matched로 비교한다.

### RQ4. 외부기억을 weight로 언제 승격할 것인가?

가설: 일정 횟수 이상 서로 다른 session에서 retrieval hit와 downstream benefit가 재현된
기억만 compile하면 LoRA/SMF의 간섭과 version 수를 통제할 수 있다.

필수 검증: 잘못 승격된 기억의 unlearning cost와 source deletion propagation까지 포함한다.

### RQ5. memory capacity의 scaling law가 존재하는가?

독립변수는 model parameter뿐 아니라 raw bytes, retrieval token budget, sparse slot/expert 수,
adapter rank, sleep FLOPs, bytes moved여야 한다. 종속변수는 recall뿐 아니라 interference,
staleness, corruption을 포함해야 한다.

### RQ6. sleep-trained memory writer를 어떻게 신뢰할 것인가?

가설: provenance-constrained generation + transition verifier + canary future tasks +
version rollback이 단순 LLM summary보다 낫다.

TrustMem의 coverage/preservation/faithfulness를 parametric update에도 확장해, “이 LoRA/expert가
무엇을 보존·왜곡했는가”를 검증해야 한다.

---

## 9. 읽기 순서

### P0 — 이 framing을 세우는 데 필수

1. [Complementary Learning Systems (1995)](https://web.stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf)
   fast episodic ↔ slow structured learning의 이론.
2. [PLOS SNN Sleep (2022)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010628)
   raw old data 없이 자발적 replay가 weight-manifold를 보존한다는 최소 실험.
3. [Perturbed and Adversarial Dreaming (2022)](https://elifesciences.org/articles/76384)
   NREM replay와 REM novel synthesis를 서로 다른 objective로 분해.
4. [MemGPT (2023)](https://arxiv.org/abs/2310.08560)
   virtual context와 tiered external memory의 시스템 출발점.
5. [Sleep-time Compute (2025)](https://arxiv.org/abs/2504.13171)
   pre-query token-space compilation의 정확한 정의와 amortization 조건.
6. [Memory Layers at Scale (2024)](https://arxiv.org/abs/2412.09764) +
   [Sparse Memory Finetuning (2025)](https://arxiv.org/abs/2510.15103)
   Meta의 sparse parametric capacity와 저간섭 update.
7. [Mem0 (2025)](https://arxiv.org/abs/2504.19413) +
   [Zep/Graphiti (2025)](https://arxiv.org/abs/2501.13956)
   text/vector/temporal graph의 production-style consolidation.
8. [Cartridges (2025)](https://arxiv.org/abs/2506.06266) +
   [SEAL (2025)](https://arxiv.org/abs/2506.10943)
   context→latent와 self-generated data→weight라는 두 compilation 방법.
9. [When Continual Learning Moves to Memory (2026)](https://arxiv.org/abs/2604.27003)
   external memory도 stability–plasticity를 retrieval 문제로 재현한다는 이론적 교정.
10. [Auto-Dreamer (2026)](https://arxiv.org/abs/2605.20616)
    learned external offline consolidation.
11. [Google Language Models Need Sleep (2026)](https://arxiv.org/abs/2606.03979)
    multi-frequency parametric consolidation, distillation, RL, dreaming, growth/reset.
12. [Retain or Consolidate? (2026)](https://arxiv.org/abs/2607.17545) +
    [TrustMem (2026)](https://arxiv.org/abs/2606.25161)
    budget-dependent operator selection과 reliable transition learning.

### P1 — 구현과 학습법을 설계할 때

- [TTT (2020)](https://arxiv.org/abs/1909.13231),
  [TTT Layers (2024)](https://arxiv.org/abs/2407.04620)
- [REMIND (2019)](https://arxiv.org/abs/1910.02509)
- [MEMORYLLM (2024)](https://arxiv.org/abs/2402.04624),
  [M+ (2025)](https://arxiv.org/abs/2502.00592)
- [Generative Agents (2023)](https://arxiv.org/abs/2304.03442),
  [LongMem (2023)](https://arxiv.org/abs/2306.07174)
- [Generative Adapter (2024)](https://arxiv.org/abs/2411.05877),
  [Doc-to-LoRA (2026)](https://arxiv.org/abs/2602.15902)
- [AgeMem (2026)](https://arxiv.org/abs/2601.01885),
  [ACE (2025, Stanford/SambaNova/UC Berkeley)](https://arxiv.org/abs/2510.04618)
- Microsoft line:
  [Memora (2026)](https://www.microsoft.com/en-us/research/publication/memora-a-harmonic-memory-representation-balancing-abstraction-and-specificity/),
  [Microsoft Human-Inspired Memory Architecture (2026)](https://arxiv.org/abs/2605.08538)
- [Cartridges at Scale (2026)](https://arxiv.org/abs/2606.04557),
  [Memory Caching (2026)](https://arxiv.org/abs/2602.24281)
- Google line: [Titans](https://arxiv.org/abs/2501.00663) →
  {[Miras](https://arxiv.org/abs/2504.13173),
  [Atlas](https://arxiv.org/abs/2505.23735)} →
  [Nested Learning](https://arxiv.org/abs/2512.24695)
- Google parallel systems branches:
  [TNT](https://arxiv.org/abs/2511.07343),
  [Memory Caching](https://arxiv.org/abs/2602.24281),
  [NSTM](https://arxiv.org/abs/2607.15271)

### P2 — 아이디어는 흥미롭지만 증거 등급을 낮춰 읽기

- [DANN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5402490):
  “zero forgetting”을 headline으로 인용하지 말고 full method/code/replication을 먼저 확인.
- [Sleep-Consolidated Memory](https://arxiv.org/abs/2604.20943):
  explicit LLM sleep prototype이지만 작은 연구 preview.
- [Dreaming Is All You Need](https://arxiv.org/abs/2409.01633),
  [Wake-Sleep Consolidated Learning](https://arxiv.org/abs/2401.08623),
  [MyGO](https://arxiv.org/abs/2508.21296):
  생물학적 framing과 operator 아이디어는 취하되 LLM production evidence로 일반화하지 않음.
  특히 SleepNet/DreamNet은 offline sleep/continual-learning 구현이 아니며 TMLR rejected
  manuscript라는 상태를 함께 기록.

---

## 10. 블로그 주장에 대한 판정

[검토 대상 원문](https://m.blog.naver.com/simula/224208155832)의 명명과 인용 단서를
논문·공식 제품 자료와 대조한 결과다.

| 블로그 주제 | 판정 | 이 연구에서의 처리 |
|---|---|---|
| 2022 PLOS | **실재, 동료평가 저널 논문** | SNN 두 task 결과로 정확히 한정 |
| DANN | **실재하지만 SSRN manuscript** | 저증거 아이디어로 분류; “zero forgetting” 미확정 |
| Meta sparse memory | **SMF와 Memory Layers로 확인** | parametric sparse update이지 sleep scheduler는 아님 |
| MemGPT/Letta | **둘을 시대별로 분리해야 함** | MemGPT=online single-agent memory; Letta 2025=async split |
| Mem0/Zep | **버전별 external operator가 존재** | Mem0 2025=hot-path update+async summary, V3=ADD-only per-add ingest; Graphiti=per-episode+opt-in/manual community, managed Zep Observations 2026=proprietary background/ingest-adjacent cross-episode synthesis. 모두 weight learning은 아님 |
| Doc-to-LoRA/Cartridges/Generative Adapter | **latent/parameter compilation 계보로 유효** | repeated-query amortization과 model compatibility를 평가 |
| NREM/REM analogy | **PAD와 Google Sleep이 계산적 역할 분담을 구현; PLOS는 REM-like state 하나** | 생물학적으로 확정된 equivalence가 아니라 algorithmic hypothesis로만 사용 |
| Claude/Grok 등의 “sleep” | **공개 근거 없으면 추측** | 회사 제품 마케팅과 논문 증거를 분리 |
| sleep hardware/cluster | **아직 연구 가설** | update/apply decoupling, data movement, tiering으로 검증 |

---

## 11. 다음 deep research의 산출물 제안

pre-research 이후 본 연구는 논문별 요약보다 다음 여섯 개의 공통 artifact를 만드는 편이 좋다.

1. **canonical chronology**: 공개일·소속·peer-review·code·dataset까지 검증한 ledger.
2. **operator matrix**: retain/merge/abstract/rewrite/replay/distill/SFT/RL/grow/prune의
   input, output, loss, compute, reversibility.
3. **memory-promotion policy**: event→external→latent→parametric의 승격/강등 조건.
4. **capacity accounting model**: bytes, context budget, slot/expert, interference, governance를
   함께 세는 model.
5. **wake/sleep infra design**: state ownership, snapshot/version, scheduling, cache/offload,
   bytes moved를 포함한 reference architecture.
6. **matched-budget benchmark**: 동일 model·sleep FLOPs·live token·storage에서
   raw retention, vector, graph, cartridge, LoRA, SMF, hybrid를 비교.

최종 framing은 다음처럼 잡을 수 있다.

> 앞으로의 학습은 배포 전 global learning, wake의 low-latency capture/test-time adaptation,
> sleep의 offline consolidation/dreaming으로 나뉜다. 그러나 이것은 세 개의 고립된 단계가
> 아니라, 경험을 exact external memory에서 점차 압축된 symbolic·latent·parametric
> memory로 옮기는 하나의 lifecycle이다. 핵심 연구 대상은 model 하나가 아니라
> **scheduler + consolidator + memory hierarchy + verifier + serving infra**를 합친 system이다.
