# PRE-RESEARCH — Sleep-Time Compute, Continual Learning, and Long-Term Memory

> 조사 기준일: 2026-07-25
> 목적: sleep-time compute를 **모수적 기억 업데이트**와 **모델 외부 기억 관리**라는 두 축으로
> 재구성하고, 관련 연구를 최초 공개 시점 순으로 읽기 위한 사전 조사 문서
> 범위: 논문, 공식 연구·제품 문서, 그리고 블로그에서 지목한 PLOS·DANN·Letta·Mem0·Zep 계보

---

## 원문 감사 companion

이 문서의 시간순 지도에서 핵심 선행연구는 다음 full-text 감사로 확장했다.

- [생물학·계산적 sleep basis와 PLOS/DANN 감사](../research/sleep-time-compute/PRIMARY-SOURCE-AUDIT-BIOLOGICAL-COMPUTATIONAL-SLEEP.md)
- [2017--2025 wake--sleep 직접 선행연구 감사](../research/sleep-time-compute/PRIMARY-SOURCE-AUDIT-EARLY-PRECURSORS.md)
- [MemGPT/Letta·Mem0·Zep·LangMem 구현 감사](../research/sleep-time-compute/PRIMARY-SOURCE-AUDIT-AGENT-MEMORY-SYSTEMS.md)
- [용량·공고화·망각 이론 계보 감사](../research/sleep-time-compute/PRIMARY-SOURCE-AUDIT-CAPACITY-CONSOLIDATION-THEORY.md)
- [2026년 7월 capacity·compaction 논문 감사](../research/sleep-time-compute/PRIMARY-SOURCE-AUDIT-JULY-2026.md)
- [Google·Meta·Microsoft 및 deployed memory company-lineage 감사](../research/sleep-time-compute/PRIMARY-SOURCE-AUDIT-COMPANY-LINEAGES.md)
- [sleep training·data construction·RL 방법 atlas](../research/sleep-time-compute/TRAINING-DATA-METHODS-ATLAS.md)
- [RQ1–RQ12 answerability·가설·반증 matrix](../research/sleep-time-compute/OPEN-QUESTIONS-ANSWERABILITY-MATRIX.md)
- [wake–sleep–memory system infra blueprint](../research/sleep-time-compute/SYSTEM-INFRA-BLUEPRINT.md)
- [system infra 선행기술 element-by-element 감사](../research/sleep-time-compute/SYSTEMS-INFRA-PRIMARY-SOURCE-AUDIT.md)
- [scaling-law·capacity-knee·queue theory agenda](../research/sleep-time-compute/SCALING-LAWS-THEORY-AGENDA.md)
- [controlled lifetime benchmark·experiment blueprint](../research/sleep-time-compute/BENCHMARK-EXPERIMENT-BLUEPRINT.md)
- [theory·benchmark·infra scientific red-team 감사](../research/sleep-time-compute/SCIENTIFIC-RED-TEAM-AUDIT-2026-07-25.md)

각 감사는 headline claim뿐 아니라 최초 공개일, 실제 operator, 학습법,
capacity/delete/rollback/provenance, 반증, 재현성 경계를 함께 기록한다.

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

세 범주는 다시 **explicit/inspectable ↔ implicit/latent**와
**portable/re-prefillable ↔ execution-bound/stateful** 축으로 나뉜다.
MEMENTO가 보인 것처럼 같은 summary text라도 생성 당시 source를 본 KV와 restart로 만든
KV는 같은 기억이 아닐 수 있다. 따라서 “어디에 저장했는가”뿐 아니라 “완전한 상태가
무엇이며 재시작·이동·삭제 뒤에도 동등한가”를 함께 물어야 한다.

또한 parametric destination도 retrieval-free가 아니다. multi-LoRA는 document/item 대신
adapter를 찾아 load·merge해야 하고, temporal external memory는 session/day/week/month
clock마다 construction call과 derived view를 만든다. 그러므로 최종 비교 단위는
`stored content`가 아니라 **destination route × within-destination route × compatible
composition × total state/service cost**다.

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
| latent external | KV cache, cartridge, learned vector | 짧은 live context, 빠른 적용 | hidden source channel, restart 비동등, 해석·삭제·model migration 어려움 |
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

여기서 \(M_{\mathrm{KV}}\)는 저장 파일만이 아니라 future answer에 기여하는 모든
source-conditioned hidden state와 그 model/mask dependency를 포함한다.

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
| 2017-10 | Kamra et al., [Deep Generative Dual Memory Network](https://arxiv.org/abs/1710.10368) **[B, ICLR 2018 submission]** | 작은 task별 STM이 가득 차면 down-time에 생성 표본을 LTM으로 보내고 deep generative replay로 통합한다. 명시적 wake/sleep 선행 구현이지만 ICLR 제출 기록을 acceptance로 세면 안 되며 MNIST형 분류와 task descriptor 중심이다. |
| 2017-11 | Aljundi et al., [Memory Aware Synapses](https://arxiv.org/abs/1711.09601) **[A, KU Leuven+FAIR]** | unlabeled stream에서 parameter importance를 누적해 선택적으로 보존·덮어쓰기. 유한 용량을 명시적으로 문제화. |
| 2017-11 / ICLR 2018 | Kemker & Kanan, [FearNet](https://openreview.net/forum?id=SJ1Xmf-Rb) **[A]** | 최근 exemplar를 HC에 모으고 10 study session마다 autoencoder pseudo-example과 섞어 mPFC를 60 epoch finetune한 뒤 HC를 비운다. strict periodic sleep의 중요한 선행이지만 pretrained feature 기반 class-incremental vision/audio이며 exact delete·rollback은 없다. |
| 2018-05 / ICML 2018 | Schwarz et al., [Progress & Compress](https://proceedings.mlr.press/v80/schwarz18a.html) **[A, DeepMind]** | active column이 새 task를 배우고, task 뒤 distillation+online EWC로 고정 크기 knowledge base에 압축한다. daytime/nighttime parametric consolidation의 직접 선조지만 task boundary와 current-task data를 가정한다. |
| 2018-10 / 2019-01 | Fachechi et al., [Dreaming Neural Networks](https://doi.org/10.1016/j.neunet.2019.01.006) **[A]** | Hopfield memory의 offline unlearning/consolidation으로 spurious attractor를 제거하고 이상화된 이론 capacity를 높임. 자연 데이터나 deep continual learning은 아니다. |
| 2018-11 preprint / NeurIPS 2019 | Rolnick et al., [Experience Replay for Continual Learning (CLEAR)](https://proceedings.neurips.cc/paper/8327-experience-replay-for-continual-learning) **[A, DeepMind+UPenn]** | recent on-policy trajectory와 replay-buffer의 off-policy trajectory를 계속 섞고 과거 policy/value를 behavioral cloning한다. constrained buffer에서도 random discard가 unbounded buffer에 근접했지만 매 update에 작동하는 wake/online replay이지 분리된 sleep은 아니다. |
| 2019-02 / ICML 2019 | Kaplanis et al., [Policy Consolidation](https://proceedings.mlr.press/v97/kaplanis19a.html) **[A, Imperial+DeepMind]** | 여러 timescale의 hidden policy cascade가 양방향 KL distillation으로 현재 policy를 과거와 regularize한다. task boundary가 필요 없지만 매 training update에 작동하는 wake/online mechanism이지 별도 sleep phase는 아니다. |
| 2019-07 | Golden et al., [PLOS 연구의 bioRxiv 선공개판](https://doi.org/10.1101/688622) **[preprint]** | 2022 PLOS 논문의 실제 최초 공개. 따라서 아이디어의 chronology는 2019→2022다. |
| 2019-07 / NeurIPS 2019 | Lample et al., [Large Memory Layers with Product Keys](https://arxiv.org/abs/1907.05242) **[A, FAIR+Sorbonne]** | 최대 10억 parameter의 sparse KV lookup. Meta Memory Layers/SMF의 직접적인 구조적 조상이며, 당시에는 pretrain 후 고정되어 continual sleep은 없다. |
| 2019-08 | Krishnan et al., [Biologically Inspired Sleep Algorithm for ANNs](https://arxiv.org/abs/1908.02240) **[B]** | ANN→SNN 변환, 평균 입력통계 기반 noise+STDP sleep, 다시 ANN 변환. generative replay보다 약하고 dataset별 tuning이 필요했다. |
| 2019-09 / ICML 2020 | Sun et al., [Test-Time Training](https://arxiv.org/abs/1909.13231) **[A]** | unlabeled test sample의 self-supervised loss로 weight를 갱신. distribution-shift adaptation이지 장기 개인 기억은 아니다. |
| 2019-10 | Hayes et al., [REMIND](https://arxiv.org/abs/1910.02509) **[A]** | raw image 대신 압축 latent를 replay하는 continual learning. sleep dataset의 저장비용을 낮추는 전구체. |
| 2019-12 / ICLR 2020 | Tadros et al., [Sleep Algorithm for Generalization and Robustness](https://openreview.net/forum?id=r1xGnA4Kvr) **[A]** | ANN→SNN/STDP sleep이 일부 noise/adversarial robustness를 높임. continual lifetime capacity 실험은 아니다. |
| 2020-08 | González et al., [Can Sleep Protect Memories from Catastrophic Forgetting?](https://doi.org/10.7554/eLife.51005) **[A]** | 생물물리 thalamocortical model에서 input 없는 SWS dynamics가 두 상반 sequence의 부분 trace를 replay. 완전 소실 전만 복구 가능. |
| 2020-08 | van de Ven et al., [Brain-inspired Replay for Continual Learning](https://doi.org/10.1038/s41467-020-17866-2) **[A]** | generative feedback model이 과거 sample을 만들어 새 데이터와 섞는다. 강한 replay baseline이지만 sleep state 자체를 구현하지 않는다. |
| 2021-05-13 / ICML 2021 | Sukhbaatar et al., [Expire-Span](https://arxiv.org/abs/2105.06548) **[A, Facebook AI Research; Angela Fan also LORIA]** | 각 memory에 학습된 expiration을 부여. “무한히 쌓기” 대신 유한 attention memory의 수명을 학습하는 선행 아이디어. |
| 2021-05 | Hoel, [The Overfitted Brain](https://doi.org/10.1016/j.patter.2021.100244) **[A, 가설]** | 꿈을 distorted/augmented input으로 보아 overfitting을 줄인다는 perspective. 새 계산·생물 실험은 없다. |

생물학에서 비교적 강하게 말할 수 있는 것은 “수면이 기억 통합을 돕고, 경험 관련 활동이
수면 중 재현되며, 일부 cue와 rhythm 조작이 다음날 기억을 바꾼다”까지다.
**NREM=정확 replay, REM=창의적 adversarial generation**이라는 깔끔한 역할 분담은 유용한
계산 가설이지 확정된 생물학적 사실이 아니다.

### 2.2 2021–2022: 생물학적 sleep을 계산 절차로 분해

| 시점 | 연구 | wake → sleep 메커니즘 | 해석 |
|---|---|---|---|
| 2021-09 preprint / 2022-05 eLife | Deperrois et al., [Perturbed and Adversarial Dreaming](https://elifesciences.org/articles/76384) **[A]** | wake에서 episodic latent 저장 → NREM에서 occlusion을 가한 replay/reconstruction → REM에서 기억 혼합+noise를 GAN식 adversarial sample로 생성 | NREM/REM에 서로 다른 objective를 준 명시적 계산 가설. CIFAR-10/SVHN representation learning이며 continual learning·catastrophic forgetting·lifetime capacity 실험은 아니다. |
| 2022-02-01 preprint / 2022-10-24 PNAS | Singh, Norman & Schapiro, [A model of autonomous interactions between hippocampus and neocortex](https://doi.org/10.1073/pnas.2123432119) **[A]** | random seed 뒤 short-term depression으로 attractor를 자율 순회; NREM은 hippocampus--neocortex 결합, REM은 neocortex-only replay; contrastive Hebbian/error-driven cortical update | replay trial 자체를 내부 dynamics가 만드는 직접 선행. 100-initialization synthetic-category simulation이며 stage schedule은 실험자가 고정하고 lifetime capacity·scheduler·governance는 없다. |
| 2022-11-18 | Golden et al., [Sleep prevents catastrophic forgetting in spiking neural networks by forming a joint synaptic weight representation](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010628), *PLOS Computational Biology* 18(11):e1010628 **[A]** | 새 task RL 사이에 input을 끄고 Poisson stimulation과 unsupervised STDP로 자발적 old-trace reactivation | 블로그가 말한 “2022 PLOS”의 정확한 논문. **PLOS Computational Biology 저널 논문이지 PLOS 학회가 아니다.** |
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
| 2023-03-19 / TMLR 2023 | Harun et al., [SIESTA](https://openreview.net/forum?id=MqDVlBWRRV) **[A]** | wake에서는 frozen encoder의 PQ latent를 bounded buffer에 쓰고 running class mean만 갱신; sleep에서는 balanced latent replay로 상위 layers를 backprop | strict wake/sleep continual-learning precursor다. ImageNet-1K augmentation-free top-5 `83.59`와 single-A5000 `1.9 h`를 보고하지만 supervised vision, frozen lower features, 2.02 GB buffer라는 범위가 있다. |
| 2023-03-20 | Shinn et al., [Reflexion](https://arxiv.org/abs/2303.11366) **[A]** | 이전 trial trajectory+reward에서 자연어 교훈을 만들고 episodic prompt memory에 append | inter-episode consolidation의 선행 사례. 최근 1–3 reflection만 쓰며 별도 sleep scheduler나 weight update는 없다. |
| 2023-04-07 | Park et al., [Generative Agents](https://arxiv.org/abs/2304.03442) **[A]** | text memory stream; recency·importance·relevance 검색; reflection으로 고차 summary 생성 | threshold-triggered reflection은 기능적 consolidation이지만 주로 online이다. |
| 2023-05-17 / AAAI 2024 | Zhong et al., [MemoryBank](https://arxiv.org/abs/2305.10250) **[A]** | daily event/personality summary, global portrait, text/vector retrieval; Ebbinghaus-inspired decay/reinforcement | 기능적 periodic summarization이나 명시적 sleep은 아니다. 10일 simulated dialogue 규모라 lifetime 증거는 약하다. |
| 2023-05-19 | [Zep product launch](https://blog.getzep.com/introducing-zep-memory-ai/) **[C]** | message store에 비동기 summary·embedding·entity enrichment | 외부기억 background processing의 제품 선행 사례지만 cross-memory sleep은 아니다. |
| 2023-06 / NeurIPS 2023 | Wang et al., [LongMem](https://arxiv.org/abs/2306.07174) **[A, Microsoft Research+UCSB]** | frozen backbone/memory encoder + trained adaptive residual SideNet retriever-reader; past context bank | base encoder는 frozen이고 외부 latent memory를 갱신하지만 SideNet은 memory-augmented adaptation으로 학습한다. |
| 2023-10-12 | Packer et al., [MemGPT](https://arxiv.org/abs/2310.08560) **[B, Berkeley]** | in-context working memory + recall DB + archival DB; function call로 self-edit; pressure warning·recursive summary | 원 논문에는 별도 sleep-trained consolidator가 없다. 대화·tool·memory management를 같은 agent가 online으로 수행한다. |
| 2023-12-06 preprint / IEEE TNNLS 2025 | Sorrenti et al., [Wake-Sleep Consolidated Learning](https://arxiv.org/abs/2401.08623) **[A]** | wake의 dynamic parameter freezing/STM → NREM의 STM+LTM replay → REM의 disjoint external dream data | vision continual learning이며 LLM agent lifecycle 증거는 아니다. REM data는 주 실험에서 model-generated dream이 아니고 외부의 보지 않은 labeled image다. |

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
| 2024-04-08 | Allen-Zhu & Li, [Physics of Language Models 3.3: Knowledge Capacity Scaling Laws](https://arxiv.org/abs/2404.05405) **[B]** | controlled factual tuples에서 약 2 knowledge bits/parameter와 training·architecture·quantization·MoE·SNR 효과를 측정 | universal fact-capacity가 아니라 해당 생성·readout protocol의 conditional parametric baseline이다. |
| 2024-07 | Sun et al., [Learning to (Learn at Test Time)](https://arxiv.org/abs/2407.04620) **[B]** | sequence hidden state 자체를 TTT-Linear/MLP model로 만들고 self-supervised gradient update | per-sequence fast parametric memory. 보통 sequence/reset을 넘는 lifetime 기억은 아니다. |
| 2024-08-28 | [Graphiti open-source release](https://blog.getzep.com/graphiti-knowledge-graphs-for-agents/) **[C, Zep]** | episode→entity/fact temporal graph ingestion | 2025 Zep 논문의 제품 선행판. ingestion은 background일 수 있어도 cross-memory sleep scheduler는 아니다. |
| 2024-09 최초 / 2026-04 v4 | Ni & Liu, [Dreaming Is All You Need → SleepNet and DreamNet: Enriching and Reconstructing Representations for Consolidated Visual Classification](https://arxiv.org/abs/2409.01633) **[B, [TMLR rejected](https://openreview.net/forum?id=vXhsnUIeTl)]** | 매 forward pass의 frozen pretrained-feature fusion과 reconstruction branch | **offline sleep·replay·continual-learning 연구가 아니다.** revision 간 dataset·parameter/FLOP·결과 일관성을 주의한다. |
| 2024-09-23 | [MemGPT project → Letta company/framework naming split](https://www.letta.com/blog/announcing-letta/) **[C]** | 연구 prototype을 stateful-agent framework·company로 확장 | MemGPT 2023 논문, Letta 2025 sleep agent, 현재 Letta 제품을 같은 버전처럼 인용하면 안 된다. |
| 2024-09-24 preprint / CoLLAs 2024, PMLR 2025 | Taylor et al., [PCMC](https://proceedings.mlr.press/v274/taylor25a.html) **[A]** | wake에서 patch embedding을 STM/LTM centroid로 군집화; periodic sleep에서 raw patch replay로 encoder를 300 epoch contrastive retrain하고 centroid를 재임베딩·prune | strict offline consolidation이다. ImageNet40에서 약 30% memory 절감과 작은 accuracy 비용을 보였지만 sleep이 매우 길고, 40-class vision·raw-patch storage·governance 부재라는 범위가 있다. |
| 2024-10-14 / ICLR 2025 | Wu et al., [LongMemEval](https://arxiv.org/abs/2410.10813) **[A]** | extraction, multi-session·temporal reasoning, knowledge update, abstention을 다루는 500-question benchmark | 지속 interaction memory 평가에 유용하지만 sleep operator 자체를 격리해 측정하지는 않는다. |
| 2024-11 / ICLR 2025 | Chen et al., [Generative Adapter](https://arxiv.org/abs/2411.05877) **[A, University of Washington+Microsoft+Microsoft Research]** | gradient로 self-supervised pretrain/instruction-tune한 adapter generator가 context를 한 번 읽고 LoRA로 변환 | 배포 시 per-context gradient descent 없이 frozen base LM용 **context→parameter** delta를 forward 생성하는 중요한 중간점. 학습 자체가 gradient-free인 것은 아니다. |
| 2024-12-12 / ICML 2025 | Berges et al., [Memory Layers at Scale](https://arxiv.org/abs/2412.09764) **[A, Meta FAIR]** | sparse trainable key-value layer; 128B memory parameters, 1T-token pretraining | sparse factual capacity를 backbone FLOPs와 분리. 이후 Sparse Memory Finetuning의 substrate. |
| 2024-12-31 / NeurIPS 2025 | Behrouz et al., [Titans](https://arxiv.org/abs/2501.00663) **[A, Google Research]** | deep neural memory를 token/chunk마다 surprise gradient, momentum, decay로 update | wake-time neural memory 계보의 시작. 그 자체는 offline sleep이 아니다. |

### 2.5 2025: “sleep-time compute”가 이름을 얻고 두 갈래로 분화

| 시점 | 연구 | 경로 | 핵심 |
|---|---|---|---|
| 2025-01 | Rasmussen et al., [Zep / Graphiti](https://arxiv.org/abs/2501.13956) **[B, Zep]** | external text+temporal KG | raw episode, semantic entity/fact, community summary의 3층 graph; bi-temporal validity와 contradiction invalidation. |
| 2025-02 | Wang et al., [M+](https://arxiv.org/abs/2502.00592) **[A, ICML 2025]** | MEMORYLLM에서 버릴 hidden token을 CPU long-term pool에 옮기고 co-trained retriever가 GPU로 복귀 | 20K→160K+ retention을 보고. 사용자의 hot/cold memory와 CPU↔GPU data movement 질문에 직접 연결되지만 finite cap과 retrieval miss가 남는다. |
| 2025-02 / NeurIPS 2025 | Xu et al., [A-MEM](https://arxiv.org/abs/2502.12110) **[A]** | external notes+links | Zettelkasten식 note attribute/tag/link를 만들고 새 기억이 기존 기억의 context를 진화시킴. |
| 2025-04-02 | [Zep Community Edition support 종료](https://blog.getzep.com/announcing-a-new-direction-for-zeps-open-source-strategy/) **[C]** | Graphiti를 active OSS focus로 전환 | 현재는 unmaintained Zep Community repo, active Graphiti OSS, managed proprietary Zep을 분리해 봐야 한다. |
| 2025-04-17 / ICLR 2026 | Behrouz et al., [Miras](https://arxiv.org/abs/2504.13173) **[A, Google Research]** | wake parametric | memory architecture·attentional bias·retention·learning algorithm의 design space. |
| 2025-04-17 | Lin et al., [Sleep-time Compute](https://arxiv.org/abs/2504.13171) **[B, Berkeley/Letta]** | external token-space compilation | query 전에 raw context를 `S(c)→c′`로 재표현. weight update가 아니라 prompting을 통한 learned context 생성. |
| 2025-04-21 | [Letta Sleep-Time Agents](https://www.letta.com/blog/sleep-time-compute/) **[C]** | primary는 recall/archival memory를 검색; sleep agent는 primary/shared core-memory block을 비동기 편집 | 대화 critical path와 in-context memory editing을 두 agent로 분리한다. archival store 전체를 sleep agent가 재작성하는 설계는 아니다. |
| 2025-04-28 / ECAI 2025 | Chhikara et al., [Mem0](https://arxiv.org/abs/2504.19413) **[A, Mem0]** | external text/vector, 선택적 graph | extraction 후 top-similar memory에 ADD/UPDATE/DELETE/NOOP; periodic conversation-summary refresh는 background 작업. |
| 2025-05-21 / ICLR 2026 | Zhao et al., [Pre-training Limited Memory Language Models with Internal and External Knowledge](https://arxiv.org/abs/2505.15962) **[A]** | external DB call이 반환한 factual value를 loss에서 mask해 model이 사실을 weight에 암기하는 대신 lookup하도록 pretrain | 배포 전 parametric/external allocator의 직접 선행이며 DB edit/delete control을 제공하지만 session sleep은 아니다. |
| 2025-05-29 / ICML 2026 | Behrouz et al., [Atlas](https://arxiv.org/abs/2505.23735) **[A, Google Research]** | wake parametric | windowed Omega objective, Muon inner optimizer, associative-memory capacity 이론. |
| 2025-05-30 | Morris et al., [How much do language models memorize?](https://arxiv.org/abs/2505.24832) **[B, Meta FAIR+Google DeepMind+Cornell+NVIDIA]** | uniform synthetic sequence로 generalization을 제거해 GPT-style model의 memorization을 약 3.6 bits/parameter로 측정 | measured lower bound이며 2-bit tuple 연구와 동일한 capacity 정의가 아니다. 자연어·순차 sleep의 universal constant로 쓰지 않는다. |
| 2025-05-31 arXiv lineage / ICLR 2026 workshop | Hsing, [MIRROR](https://openreview.net/forum?id=IviO4bIZc7) **[B]** | Talker response 뒤 async Thinker+Controller가 bounded first-person narrative를 완전 재생성하고 다음 turn이 읽음 | strict external sleep analogue. `O(1)`은 state/per-turn bound이지 lifetime work bound가 아니며 공개 code에는 generation-ordered publication이 없다. |
| 2025-06-06 | Eyuboglu et al., [Cartridges](https://arxiv.org/abs/2506.06266) **[B]** | latent KV compilation | corpus별 작은 KV cache를 offline 학습. naive NTP 대신 synthetic conversation을 만들고 context distillation. |
| 2025-06-10 LCFM @ ICML 2025 / 2025-07-07 arXiv / ICLR 2026 | Hu et al., [MemoryAgentBench](https://arxiv.org/abs/2507.05257) **[A]** | incremental memory benchmark | accurate retrieval, test-time learning, long-range understanding, selective forgetting을 분리해 평가. offline scheduling 자체는 측정하지 않는다. |
| 2025-06-12 / NeurIPS 2025 | Zweiger et al., [SEAL](https://arxiv.org/abs/2506.10943) **[A]** | persistent weight update | model이 finetuning data·augmentation·hyperparameter를 포함한 self-edit를 생성; SFT 후 성능을 RL reward로 사용. |
| 2025-09-05 ICLR submission / 2025-10-20 arXiv / ICML 2026 | [MemoryBench](https://arxiv.org/abs/2510.17281) **[A, Tsinghua]** | declarative/procedural feedback stream에서 external memory system의 on/off-policy learning을 평가 | wake feedback learning의 강한 비교군이다. 11개 dataset과 explicit/implicit feedback을 포함하지만 sleep phase, lifetime cap, delete·rollback은 없다. ICLR record는 desk-rejected submission이고 최종 venue와 구분한다. |
| 2025-06-29 / v2 2025-08 | Golden et al., [Interleaved Replay of Novel and Familiar Memory Traces During Slow-Wave Sleep](https://doi.org/10.1101/2025.06.25.661579) **[B, bioRxiv]** | phase-separated biological replay | biophysical modeling과 mouse retrosplenial single-unit 분석을 결합한 SCoRe 가설. novel replay는 slow-wave 전환부, familiar replay는 Up-state 중간에 나타나지만 아직 비동료심사이며 AI continual-learning 성능을 직접 시험하지 않았다. |
| 2025-08 작성 / 09 게시 | Tutuncuoglu, [Dream-Augmented Neural Networks](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5402490) **[D]** | latent replay/restructure | raw data 없이 periodic synthetic dream을 한다는 주장. 제목은 “Zero Forgetting”이지만 초록은 vision/NLP sequential benchmark에서 최대 60% forgetting 감소만 주장한다. SSRN 20쪽 단독저자 manuscript이고 “IEEE”는 입력된 affiliation일 뿐 IEEE 출판 논문이 아니며, 공개 record에 구체 dataset·code·독립 재현이 없다. |
| 2025-09-18 OpenReview / 2025-10-01 arXiv / ICML 2026 | Kang et al., [ACON](https://arxiv.org/abs/2510.00615) **[A, KAIST+Microsoft+University of Cambridge; first-author work during Microsoft internship]** | full-context success/compressed failure pair로 natural-language compression guideline을 offline optimize하고 small compressor에 distill | runtime threshold context compression의 control-policy precedent지만 persistent user-memory consolidation·canonical-store capacity·rollback은 아니다. |
| 2025-09-18 / NeurIPS 2025 | Spens, Burgess & Behrens, [Modelling the Control of Offline Processing with Reinforcement Learning](https://papers.nips.cc/paper_files/paper/2025/hash/d7e5870810331da5a8ac8bd16d42e074-Abstract-Conference.html) **[A, Oxford+UCL]** | recurrent PPO meta-controller가 각 toy task의 서로 다른 sleep-action set에서 offline processing을 선택 | adaptive sleep curriculum은 직접 선행이다. image/maze에서만 Shapley-style marginal utility로 episode value estimator를 학습한 뒤 deterministic MMR로 replay를 rerank하며, relational task에는 둘 다 없다. world model의 episode별 reset도 relational task에만 해당한다. Fashion-MNIST 실험은 validation set을 가정한다. LLM, cross-substrate lifetime state, delete·rollback·infra는 다루지 않는다. |
| 2025-09-19 OpenReview / 2026-06-02 arXiv v1 / 2026-07-10 v2 | Behrouz et al., [Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories](https://arxiv.org/abs/2606.03979) **[B, Google Research+Cornell]** | parametric sleep | CMS fast→slow expert distillation+RL imitation, synthetic dream+LoRA+ReST^EM. 현재 공개 acceptance는 확인되지 않았다. |
| 2025-09-19 OpenReview / 10-16 arXiv | Lin et al., [Sparse Memory Finetuning](https://arxiv.org/abs/2510.15103) **[B, FAIR at Meta+UC Berkeley]** | sparse parametric memory | 새 batch에 특이적으로 활성화되는 memory slot만 gradient update; dense FT/LoRA보다 기존 능력 손실 감소. ICLR 2026 제출은 [desk rejected](https://openreview.net/forum?id=LGo7U1m24L)이므로 preprint로 취급. |
| 2025-09-19 OpenReview / 2025-09-29 arXiv / ICLR 2026 | Ouyang et al., [ReasoningBank](https://arxiv.org/abs/2509.25140) **[A, Google-led]** | self-judged success/failure trajectory에서 reasoning strategy를 추출해 external bank에 append하고 다음 task에서 retrieve; MaTTS가 parallel/sequential exploration을 추가 | 배포 후 경험학습과 test-time scaling의 직접 연결이지만 논문 구현은 더 복잡한 bank-wide consolidation을 future work로 남긴 **wake/post-episode append loop**다. strict asynchronous sleep이나 hard capacity·delete·rollback은 아니다. |
| 2025-09-20 OpenReview / 2026-02-27 arXiv / ICML 2026 | [Memory Caching](https://arxiv.org/abs/2602.24281) **[A, ICML 2026 camera-ready: Google Research; arXiv v1: Google Research+Cornell+USC]** | recurrent neural-memory state의 checkpoint/cache를 선택적으로 보존·혼합 | cached segment-state 수에 따라 capacity가 늘고 `O(NL)` compute와 교환되며 storage tiering에 직접 연결된다. |
| 2025-09-20 anonymous record / 2026-03-01 arXiv / ICML 2026 | [Understanding LoRA as Knowledge Memory: An Empirical Analysis](https://arxiv.org/abs/2603.01097) **[A, KAIST+Samsung SDS+NYU]** | raw·QA·summary·rewrite·mixed document supervision으로 single/multi-LoRA를 지식 모듈로 학습 | low-rank capacity knee와 routing·merge 간섭을 직접 측정한다. 64K-token PhoneBook corpus에서 trainable-parameter budget을 맞추면 oracle-routed `8×rank-4`가 rank-32 하나보다 강하지만 실제 router가 이득을 없앨 수 있어, parametric memory에도 retrieval layer가 있음을 보인다. |
| 2025-09-20 OpenReview / 2025-11-10 arXiv / ICLR 2026 | Behrouz et al., [TNT](https://arxiv.org/abs/2511.07343) **[A, Google Research+USC]** | parametric systems | local/global memory hierarchy, reset, train-big/serve-small chunking으로 neural-memory 학습을 최대 17배 가속. Nested Learning의 필수 선행 단계가 아니라 병렬적인 training-efficiency branch다. |
| 2025-10-06 / AAMAS 2026 | Han et al., [LEGOMem](https://arxiv.org/abs/2510.04851) **[A, Microsoft]** | successful OfficeBench trajectory를 full-task와 subtask procedural module로 one-shot offline distill | fixed procedural bank의 sleep-adjacent primitive지만 recurring deployed sleep·online accumulation·forget/delete/rollback은 없다. |
| 2025-10-06 arXiv / ICLR 2026 | Zhang et al., [Agentic Context Engineering](https://arxiv.org/abs/2510.04618) **[A, Stanford/SambaNova/UC Berkeley]** | external evolving playbook | generation→reflection→curation의 incremental update로 brevity bias와 context collapse를 줄임. Microsoft 연구가 아니다. |
| 2025-11-05 arXiv | [HaluMem](https://arxiv.org/abs/2511.03506) **[B]** | external-memory safety | extraction·update·QA hallucination이 장기간 누적·전파되는 문제를 측정. |
| 2025-11-07 Google Research 공개 / 2025-12-31 arXiv / NeurIPS 2025 | Behrouz et al., [Nested Learning](https://arxiv.org/abs/2512.24695) **[A, Google Research]** | multi-timescale parametric | model+optimizer를 서로 다른 update frequency의 associative memories로 보고 Continuum Memory System을 제안. |

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

### 2.6 2025년 말–2026: consolidation policy를 “학습”하고 용량·인프라를 전면화

| 시점 | 연구 | 기억 위치 / 학습법 | 이번 연구에서의 중요성 |
|---|---|---|---|
| 2025-12-08 / ICML 2026 | Li et al., [H‑EPM](https://arxiv.org/abs/2512.07287) **[A, HKUST+Microsoft]** | 성공한 tool trajectory를 state-summary가 달린 episodic/procedural tool graph로 바꾸고, 별도 실험에서는 이 graph로 GRPO rollout을 유도 | ACEBench appendix는 test-time online update만 말하고 성공 판정→task 종료→graph insert→later reuse 순서를 제시하지 않는다. 공개 repo의 가장 가까운 online writer는 tool-call segment마다 inline update하므로 **strict sleep PASS가 아니다**. 외부 기억과 one-epoch GRPO를 잇는 근거지만 주기적 scheduler, cap·merge·delete·rollback은 없다. |
| 2025-12-21 / ICML 2026 | Zhang et al., [MemEvolve](https://arxiv.org/abs/2512.18746) **[A]** | completed task batch의 success·token·latency log로 fixed LLM이 `Encode/Store/Retrieve/Manage` provider code를 Pareto evolution | outer architecture evolution은 다음 round에 재사용되어 experimental boundary PASS지만, inner content update는 task return을 막는 wake path다. 공통 cap이 없고 per-round backup은 total storage를 늘릴 수 있다. |
| 2025-12-27 | Li, [Memento 2](https://arxiv.org/abs/2512.22716) **[D, single-author theory preprint]** | frozen LLM 주위 external case, Parzen policy, \(Q\)를 read→act→feedback→write | 같은 task loop를 바꾸는 wake learning이다. finite-memory/two-timescale value bound는 있으나 실험·fixed-budget coverage/eviction theorem은 없다. |
| 2026-01 / ACL 2026 | [AgeMem](https://arxiv.org/abs/2601.01885) **[A]** | external STM/LTM operation을 policy action으로 만들고 3-stage progressive RL + step-wise GRPO | store/retrieve/update/summarize/discard를 heuristic이 아니라 학습된 controller로 옮김. **runtime은 wake/online이며 sleep이 아니다.** |
| 2026-01-06 / ACL Findings 2026 | [TiMem](https://aclanthology.org/2026.findings-acl.1091/) **[A]** | segment→session→day→week→monthly profile의 5단 external temporal hierarchy; temporal closure 때 higher-level state 생성 | multi-timescale external consolidation의 직접 구현이다. LoCoMo에서 recalled context `−52.20%`지만 L1-only보다 call이 약 `29.5%` 늘고 total store cap·storage-time forgetting은 없다. |
| 2026-02-02 | Zhang et al., [MemSkill](https://arxiv.org/abs/2602.02474) **[B]** | PPO controller가 memory skill을 고르고 fixed-LLM designer가 hard case를 100 step마다 검토해 최대 30개 skill bank를 수정 | predeployment learning이다. bank-full이면 lowest-average-reward used skill 또는 least-used를 교체하고 snapshot rollback도 있지만, 이는 deployed semantic memory의 delete·rollback이 아니다. |
| 2026-02-03 / ICML 2026 | Xia et al., [Memora](https://arxiv.org/abs/2602.03315) **[A, Microsoft]** | primary abstraction이 concrete values를 index하고 cue anchors가 검색 경로 확장 | abstraction의 규모 이득과 specificity 손실을 한 구조 안에서 다룸. |
| 2026-02-06 / ICML 2026 | Yang et al., [PlugMem](https://arxiv.org/abs/2603.03296) **[A, UIUC+Tsinghua+Microsoft]** | raw dialogue/document/action trace를 episodic tuple로 표준화한 뒤 proposition·prescription과 provenance graph로 추상화 | WebArena online path는 매 task 종료 뒤 graph에 삽입하고 다음 task/held-out retrieval-only split이 재사용하므로 **task-boundary experimental sleep**이다. HotpotQA의 merge·soft-deactivate는 active node `−5.0%`를 보이지만 반복 lifetime cap이나 물리적 erase는 아니다. |
| 2026-02-08 | Xiong, Hu & Clune, [ALMA](https://arxiv.org/abs/2602.07755) **[B, UBC+Vector]** | fixed Meta Agent가 executable database/schema, update, retrieval design을 open-ended search; dynamic evaluation은 task reward 뒤 content update | outer design search와 post-task content loop는 experimental PASS다. design의 online evolution은 future work이고 generated store·archive에 공통 cap·provenance·safe promotion이 없다. |
| 2026-02-12 / ICLR 2026 MemAgents workshop / TMLR under review | Ujváry et al., [LaCy](https://arxiv.org/abs/2602.12005) **[B, Cambridge+Apple]** | token loss+spaCy factuality로 factual target을 `<CALL>`로 바꾸어 작은 model이 무엇을 학습하고 외부에 위임할지 pretrain | cascade FactScore는 개선하지만 tested NLU는 유의하게 좋아지지 않는다. 사실 offload가 자동으로 reasoning capacity를 만든다는 주장의 negative control이며 sleep은 아니다. |
| 2026-02-12 / 03-16 / 04-06 | Letta [Context Repositories](https://www.letta.com/blog/context-repositories/) → [Next Phase](https://www.letta.com/blog/our-next-phase/) → [Code App](https://www.letta.com/blog/introducing-the-letta-code-app/) **[C]** | server sleep agent를 client-side subagent+git-backed MemFS/context repository로 교체하는 중 | 현재 제품은 2025 server sleep-agent 그대로가 아니다. memory subagent가 session을 주기적으로 review/rewrite/refine하는 `dreaming`은 남고 git history가 provenance/rollback을 제공한다. |
| 2026-02-13 | [Doc-to-LoRA](https://arxiv.org/abs/2602.15902) **[B]** | hypernetwork가 document를 한 번 읽고 LoRA 생성 | per-document gradient distillation의 latency를 한 forward pass로 줄인 parametric compiler. |
| 2026-02-18 | Liang et al., [Learning Personalized Agents from Human Feedback (PAHF)](https://arxiv.org/abs/2602.16173) **[B, Meta-led]** | pre-action clarification과 post-action correction으로 per-user external preference memory를 즉시 갱신 | live interaction에서 preference drift를 다루는 wake-memory 증거다. offline consolidation·cross-user distillation·bounded capacity·causal erase는 다루지 않는다. |
| 2026-03-10 | Lu et al., [MSSR: Memory-Aware Adaptive Replay for Continual LLM Fine-Tuning](https://arxiv.org/abs/2603.09892) **[B]** | per-sample loss로 memory strength/stability를 갱신하고 replay interval·ratio·sample priority를 조절해 LoRA를 학습 | adaptive replay cadence의 LLM 직접 비교군이지만 continual fine-tuning loop 안의 operator다. off-path sleep service, recoverability threshold 전후 인과비교, delete·rollback은 없다. |
| 2026-03-19 | Lin et al., [MemMA](https://arxiv.org/abs/2603.18718) **[B, Penn State+Amazon+Microsoft]** | 세션 종료 뒤 5개 synthetic probe로 provisional external memory를 검증하고 실패를 fact repair→SKIP/MERGE/INSERT로 write-back | 공개 코드에서 다음 세션 전 실행되므로 **session-boundary experimental sleep**이다. 단 하나의 LoCoMo conversation이고, 논문 quick start의 76-row probe parquet 생성 pipeline이 공개되지 않아 leakage·selection 재현 감사가 불완전하다. |
| 2026-03-19 | Li et al., [Memento-Skills](https://arxiv.org/abs/2603.18743) **[B]** | ground-truth-assisted judge가 실패 skill을 rewrite/create하고 같은 question을 retry; 별도 synthetic-goal InfoNCE router | persistent external procedural learning이지만 paper loop는 wake다. skill growth·held-out gain을 보고하나 seed/CI/cost와 router pipeline/checkpoint가 없고 total bank cap·delete·rollback도 없다. |
| 2026-04 | [Mem0 V3](https://docs.mem0.ai/migration/platform-v2-to-v3) **[C]** | one-pass ADD-only extraction; overwrite/delete 없음 | queued/background 작업도 새 입력의 per-add indexing이지 cross-bank sleep이 아니다. bank는 자라며 expiry·periodic cleanup이 필요하고, 기존 memory를 자동 재처리하지 않는다. 2025 ECAI operator와 분리해야 한다. |
| 2026-04-08 official / 04-10 arXiv | Kontonis et al., [MEMENTO](https://arxiv.org/abs/2604.09852) **[B, Microsoft]** | staged SFT와 CISPO로 reasoning block→text memento를 학습하고 같은 generation 안에서 원 block KV를 물리적으로 compact | `Q/W`, strict sleep FAIL이다. 그러나 text-identical restart가 AIME24 `66.1→50.8`로 하락하고 downstream KV에서 random passcode가 복원되어, 보이는 summary 밖의 **implicit KV state**도 기억·용량·privacy accounting에 포함해야 함을 직접 보인다. |
| 2026-04-10 | Pan et al., [M★: Every Task Deserves Its Own Memory Harness](https://arxiv.org/abs/2604.11811) **[B, CityUHK+Microsoft]** | schema·write/read logic·workflow instruction을 Python memory program으로 만들고 static/rotating validation failure를 이용해 20회 population/reflection 진화 | “하나의 범용 기억 정책” 대신 **sleep operator와 destination policy 자체**를 task별로 offline search하는 control-plane 근거다. base weight나 배포 후 user memory를 학습하는 것은 아니며, ALFWorld 한 run은 약 100시간, 선택된 store의 lifetime cap·revocation은 미해결이다. |
| 2026-04-14 / ACL 2026 main | [GAM: Hierarchical Graph-based Agentic Memory](https://aclanthology.org/2026.acl-long.1600/) **[A]** | 2,048-token live progression buffer가 pause/session-end/overflow에서 topic graph와 raw event archive로 공고화 | live buffer는 bounded지만 3→27 session에서 event node `81→657`, edge `371→3,307`로 늘어난다. bounded hot state가 bounded lifetime state를 뜻하지 않는 직접 반례다. |
| 2026-04-18 / ACL 2026 main | Zhu et al., [HeLa-Mem](https://aclanthology.org/2026.acl-long.625/) **[A]** | episodic graph의 Hebbian coactivation·spreading과 dense hub의 reflective semantic distillation | paper의 episodic→semantic mechanism은 직접 relevant하지만 released LongMemEval default는 consolidation을 켜지 않고 LoCoMo code는 미공개다. decay/forgetting 함수도 실행되지 않아 principal causal path의 재현이 필요하다. |
| 2026-04-22 post-paper code | [Memento-Skills DreamDaemon](https://github.com/Memento-Teams/Memento-Skills/tree/71ac933ea1381d53389a2426f59634e0182071b8) **[C]** | response 뒤 staging summary를 quick async 또는 periodic daemon이 topic Markdown/index로 통합 | 코드상 strict external sleep이지만 03-19 paper 결과의 mechanism은 아니다. Dream 실험, hard cap, provenance, transactional file/index/staging publish, rollback이 없다. |
| 2026-04-22 | Saish Sachin Shinde, [SCM: Sleep-Consolidated Memory with Algorithmic Forgetting for Large Language Models](https://arxiv.org/abs/2604.20943) **[D, research preview]** | importance, NREM/REM, value-based forgetting | 명시적 LLM sleep prototype이나 단독 research preview·작은 평가라 핵심 근거로 쓰기 이르다. |
| 2026-04-25 | Hovagimian, [Evolve](https://arxiv.org/abs/2604.23424) **[D, single-author preprint]** | query-triggered staging section을 별도 manual/03:00 pass가 direct-promote 또는 `.85` overlap teacher compile | strict external sleep이다. 하지만 lifecycle phase마다 동일 250 query를 반복하고 no-sleep control이 없어 exact-key amortization만 지지한다. store/log artifact, hard cap, provenance, cross-vector/SQLite atomicity·durability가 없다. |
| 2026-04 | Hu et al., [When Continual Learning Moves to Memory](https://arxiv.org/abs/2604.27003) **[B]** | external experience representation·organization 실험 | **외부기억도 stability–plasticity를 없애지 않고 retrieval competition으로 옮긴다**는 핵심 반증. |
| 2026-05-08 arXiv | Ding et al., [MemCompiler](https://arxiv.org/abs/2605.07594) **[B, USTC+HUST+Microsoft Research+collaborators]** | teacher trajectory로 compiler를 SFT하고 binary-success GRPO; 매 step Brief State에서 text guidance+16 latent Soft-Mem token 생성 | memory **delivery**를 학습하지만 현재 action의 causal path에서 즉시 쓰므로 `Q/W`, strict sleep FAIL이다. 최대 10-item Brief State와 16 token은 live state만 bound하고 raw task-memory bank는 bound하지 않는다. |
| 2026-05-08 arXiv | Kerestecioglu et al., [Human-Inspired Memory Architecture for LLM Agents](https://arxiv.org/abs/2605.08538) **[B, Microsoft]** | 기본값 6시간 주기의 hot cache→episodic vector→semantic KG consolidation; dedup, TTL/interference forgetting, maturation, reconsolidation | Microsoft에서 **명시적으로 sleep-phase**를 부른 외부기억 연구. weight update는 아니다. |
| 2026-05-11 | Chen, [Mela: Test-Time Memory Consolidation based on Transformation Hypothesis](https://arxiv.org/abs/2605.10537) **[B, MusubiAI]** | surprise-driven high/low-frequency functional neural memory와 MemStack를 5B-token pretraining으로 학습 | 공개 wrapper는 HMM state를 call 밖으로 반환·재주입하지 않는다. within-sequence `Q` architecture이며 cross-session sleep·lifetime retention 증거가 아니다. |
| 2026-05-13 | Zhang et al., [Useful Memories Become Faulty When Continuously Updated by LLMs](https://arxiv.org/abs/2605.12978) **[B]** | natural-language memory를 stream 순서대로 반복 재작성 | WebShop과 ARC-AGI에서 utility가 상승 뒤 하락하고 raw episode retention이 forced consolidation과 같거나 더 강했다. 모든 abstraction의 반증은 아니지만 always-consolidate의 직접 counterevidence다. |
| 2026-05-15 / ACL Findings 2026 | [RecMem: Recurrence-based Memory Consolidation](https://aclanthology.org/2026.findings-acl.1619/) **[A]** | verbatim subconscious store에서 semantic similarity+반복 횟수 threshold를 넘은 내용만 episodic/semantic memory로 승격 | LoCoMo construction token을 Mem0 대비 약 87% 줄이지만 raw store는 unbounded이고 제거 시 한 score가 `81.10→51.88`로 하락한다. recurrence는 admission signal이지 total-capacity 해법이 아니다. |
| 2026-05 | Ye et al., [Auto-Dreamer](https://arxiv.org/abs/2605.20616) **[B]** | provenance-linked typed bank를 offline consolidator가 compact replacement로 변환; GRPO | 외부기억 계보에서 가장 직접적인 learned sleep consolidator. downstream agent reward로 학습. |
| 2026-05 | [Zep Observations](https://help.getzep.com/observations) **[C]** | graph evidence에서 durable pattern/decision/state-transition을 background/ingest-adjacent synthesis, supersede/retire | cross-episode 기능적 sleep 등가지만 explicit idle/periodic scheduler의 공개 근거는 없고, proprietary operator이며 독립 평가가 없다. 2025 Zep 논문/Graphiti OSS에는 없던 기능. |
| 2026-05-25/27 / CTB@ICML·CompLearn 2026 workshop records | [ImprintBench: Measuring the Limits of Continual Learning for LLMs](https://openreview.net/forum?id=QIJgTW3Qd2) **[A; venue record/taxonomy only]** | news·API changelog·personalization에서 context, auxiliary-parameter, model-parameter 방법을 비교 | direct acquisition뿐 아니라 temporal update, reference, composition, implicit relevance, boundary awareness를 분리한다. direct recall만으로 parametric internalization을 판정하지 못하게 하는 핵심 external-validity benchmark 후보지만, exact venue-specific PDF bytes를 고정하기 전에는 item 수와 plot 수치를 release-support 근거로 승격하지 않는다. ICML main-conference/PMLR 논문으로 세지 않는다. |
| 2026-05 | CMU/UMD, [Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference](https://arxiv.org/abs/2605.26099) **[B]** | recent context→fixed-size fast weights, KV clear, offline recurrent update `N`회 | 같은 input sequence의 context-window 경계를 넘어 fast state를 유지하는 parametric consolidation. sequence 시작에는 zero-initialize되므로 cross-session/lifetime memory는 아니다. |
| 2026-06 | Hardalov et al., [Cartridges at Scale](https://arxiv.org/abs/2606.04557) **[B]** | document별 KV cartridge, distractor mixing, GPU↔persistent-store budget manager | latent memory의 composability와 offload를 실제 systems 문제로 전환. |
| 2026-06-04 | Pan et al., [Retrospective Harness Optimization](https://arxiv.org/abs/2606.05922) **[B, CityUHK+Microsoft Research Asia]** | completed unlabeled trajectory에서 difficulty/DPP coreset→group re-solve→self-validation·consistency diagnosis→3개 full-harness 제안→pairwise self-preference로 instruction/skill/executable-tool directory 승격 | held-out task 전에 winner를 publish하므로 **completed-trajectory experimental sleep**이다. 하지만 한 round만 평가했고, SWE-Bench Pro 최적화에 selection 이후 103 agent call이 들며, 같은 GPT-5.5 계열이 solve·diagnose·optimize·rank한다. \(k,G,N\)은 한 번의 검색만 bound하고 harness/version/tool/log lifetime과 독립 safety gate는 미해결이다. |
| 2026-06-04 | Chen et al., [MAGE](https://arxiv.org/abs/2606.06090) **[B, Microsoft+USTC+NJU+UCSD]** | Grow/Compress/Maintain/Revise로 active execution path와 과거 branch를 hierarchical tree에 관리 | context growth와 error isolation을 다루고 MemoryArena에서 평균 task success `+7.8–20.4 pp`, long-context 대비 token `−55.1%`를 보고하지만 long-horizon task **중** 작동하는 wake state manager이며 recurring offline sleep은 아니다. |
| 2026-06-04 | Omri et al., [Agent Memory: Characterization and System Implications of Stateful Long-Horizon Workloads](https://arxiv.org/abs/2606.06448) **[B, Stanford+independent+KU Leuven+MIT]** | 10개 agent-memory system(외부 표현이 없는 long-context baseline 포함)의 construction, retrieval, generation, freshness, footprint, energy profile | construction이 query phase보다 비쌀 수 있고 sync는 latency, async는 staleness를 만든다. local profiling의 각 격리 SLURM job은 H100 80GB 1개와 Intel Xeon Platinum 8480C CPU core 6개를 할당받고, remote-construction run은 OpenAI API를 쓴다. parametric tier·distributed consistency·삭제 감사를 포함하지 않는다. |
| 2026-06-07 | Bazhenov, Delanois & Krishnan, [Not Just After One: Sleep-Inspired Replay Prevents Catastrophic Forgetting After Sequential Tasks](https://arxiv.org/abs/2606.08447) **[B]** | 여러 task를 학습한 뒤 한 번의 SRC sleep phase | Tadros et al. 2022 *Nature Communications* ANN-SRC 계열의 후속. 모든 과제를 부분 복원하지만 task 수가 늘수록 평균 성능은 하락하며 LLM 증거는 아니다. |
| 2026-06 | Yang et al., [TrustMem](https://arxiv.org/abs/2606.25161) **[B]** | coverage·preservation·faithfulness verifier, preference pair, preference-guided RL | runtime은 chronological chunk마다 하는 online update로 **sleep은 아니지만**, consolidation transition의 왜곡·환각을 학습 목표로 만든다. |
| 2026-06 | Letta, [Memory Models](https://www.letta.com/blog/towards-agents-that-learn/) **[C]** | token-space memory model을 future-task reward의 meta-RL로 학습한다는 연구 방향 | 논문 결과가 아니라 **industry thesis**. token memory→harness/weight distillation까지 사용자의 hybrid 축과 일치. |
| 2026-06-23 | [Are We Ready For An Agent-Native Memory System?](https://arxiv.org/abs/2606.24775) **[B]** | 12개 memory system+2개 baseline을 5 workload/11 dataset에서 representation, extraction, retrieval/routing, maintenance로 비교 | 단일 system이 지배하지 않고 utility–latency frontier가 크다. localized maintenance가 global rewrite보다 비용 효율적일 수 있어 scheduler action에 scope를 추가해야 한다. |
| 2026-06-28 / v2 07-21 | Zahn, Evans & Eagleman, [Discovery by Dreaming](https://arxiv.org/abs/2607.16256) **[B]** | cross-domain synthetic replay·counterfactual·edge case를 offline LoRA와 symbolic replay에 사용 | Llama-8B rank-256의 `+5.64±2.31 pp`는 명시적 parametric-dream 증거지만 within-domain·70B/72B 결과는 null/inconclusive이고 overtraining에서 역전된다. |
| 2026-06-30 | Mouchon, [Surprise as a Signal for Plasticity and Metacognition](https://arxiv.org/abs/2606.31495) **[D, single-author research note]** | frozen-encoder prediction error로 episodic write를 gate하고 periodic full replay로 slow linear readout을 학습; 별도 VLM prototype은 fast-store-full/manual sleep 뒤 fast state를 clear | 두 경로 모두 strict experimental sleep이다. 1000-class single-run에서 full replay가 old retention을 `+17.7/+51.3 pp` 회복하지만 recent-five-task replay는 no-replay보다 나쁘다. 코드·완전 설정·slow-store cap·delete/rollback이 없어 방향성 증거로 제한한다. |
| 2026-07-01 | Liu et al., [Procedural Memory Distillation](https://arxiv.org/abs/2607.01480) **[B, Salesforce]** | raw rollout→insight→behavior scaffold를 reverse-KL teacher로 policy weight에 internalize | post-training loop이지 배포 후 idle sleep은 아니다. external scaffold 삭제가 weight unlearning을 뜻하지 않는다. |
| 2026-07-02 | Qin et al., [Episodic-to-Semantic Consolidation Without Identity Drift](https://arxiv.org/abs/2607.01988) **[B]** | per-identity append-only log→deterministic SQLite fact upsert, checkpoint와 provenance | query-independent durable external transform의 primitive지만 synthetic rule-based prototype이고 cap·decay·RTBF·선택적 rollback이 없다. |
| 2026-07-04 | Lu, [ReCoLoRA](https://arxiv.org/abs/2607.07719) **[B]** | task boundary마다 effective weight를 slow principal·frozen residual·fresh fast adapter로 SVD 재분해 | replay 없는 parametric boundary consolidation. six-task/backbone-sensitive evidence이며 autonomous sleep·unlearning·capacity policy는 아니다. |
| 2026-07-07 / v2 07-15 | Yan, Mao & Guo, [MemDefrag](https://arxiv.org/abs/2607.05969) **[B]** | latent prefix를 query마다 reorder/top-k하고 12,800-token overflow에서 low-information state를 영구 prune | bounded latent capacity를 실험하지만 wake query/write-time이며 conflict·provenance·privacy·rollback이 없다. |
| 2026-07-08 | Howe, [Intrinsic-Noise Consolidation](https://arxiv.org/abs/2607.06924) **[B]** | task boundary anchor/Fisher와 다음 task 매 update의 barrier/noise regularizer | offline sleep이 아니다. SplitMNIST positive와 달리 real hardware-in-loop 한 seed에서는 overall net win이 없다. |
| 2026-07-09 | Colaco & Lahjouji, [Rate–Distortion View of Memory Compaction](https://arxiv.org/abs/2607.08032) **[B]** | query-agnostic rate–distortion objective, seven-axis taxonomy, reversible tier와 async sleep 제안 | cross-layer framing은 직접 novelty collision이지만 router/state machine·revocation·rollback·COMPACT-Bench는 구현하지 않았다. |
| 2026-07-13 / v2 07-14 | O'Neill, [Can a Language Model Learn Facts Continually in Its Weights?](https://arxiv.org/abs/2607.11020) **[B]** | 매 20 write마다 누적 fact 전체를 fresh base copy에 batch distill하는 sleep-like intervention | 100-write retention은 `25%`로 no-consolidation `28%`보다 낮고 capability만 12 point 보호했다. weight content와 addressability를 분리하고 external canonical copy 필요성을 보인다. |
| 2026-07-14 | Hao et al., [MemOps](https://arxiv.org/abs/2607.12893) **[B]** | Remember/Forget/Update/Reflect/TrajectoryOps operation benchmark | algorithm이 아니며 behavioral Forget QA는 backend physical erase·replica deletion의 증명이 아니다. |
| 2026-07-15 | Jiang et al., [MemCon](https://arxiv.org/abs/2607.13591) **[B]** | tabular-UCB가 retrieve/consolidate/forget/no-op을 step/query마다 선택 | wake/query-time controller이고 raw trajectory는 남는다. hard capacity·RTBF·rollback이 없다. |
| 2026-07-15 / RSS 2026 FM4RoboPlan oral | [MEMORA: Embodied Action Memory from Egocentric Videos for Reasoning and Planning](https://arxiv.org/abs/2607.14252) **[B]** | 45시간 activity stream의 entity/action state를 online edit하고 participant/video boundary에서 habit·workflow·preference를 offline 공고화 | strict multimodal external sleep이다. 최대 `+20.5 pp`, OOD robot plan `+16.6%`를 보고하고 time-restricted retrieval·snapshot rollback으로 leakage를 제어하지만 최대 15 session/participant이며 lifetime cap·consent·causal erase는 없다. |
| 2026-07-15 | Yu et al., [PReM](https://arxiv.org/abs/2607.14327) **[B, Alibaba]** | 현재 32K prompt KV를 `<m>` token으로 preserve/refresh | within-query compression이며 cross-session persistent memory나 sleep이 아니다. |
| 2026-07-16 | Elmieh et al., [NSTM](https://arxiv.org/abs/2607.15271) **[B, University of Washington+Google]** | live dynamic-NVS fast weight에 1 FPS gradient write, 30 FPS read; running-average cache | query-independent sleep이 아니라 wake/online periodic TTT다. `58.14 ms` update와 `27.01 ms` apply의 분리는 infra analogue지만 finite capacity·missed events·primacy가 남는다. |
| 2026-07-20 / v2 07-21 | Kang et al., [Retain or Consolidate?](https://arxiv.org/abs/2607.17545) **[B, Huawei Noah's Ark Lab+CityU Hong Kong]** | ridge utility estimator만 offline fit; Retain/Merge/Abstract/Rewrite 생성·packing은 매 query | tight budget에서 abstraction, loose budget에서 retention이 우세한 crossover를 보이지만 persistent sleep operator가 아니다. store lifecycle·delete·rollback은 미평가다. |
| 2026-07-20 | Senrayan et al., [EAR](https://arxiv.org/abs/2607.17879) **[B, Fujitsu]** | 매 query explore→feedback→buffer replay→reranker adapter update | wake online retrieval adaptation이며 buffer cap·privacy delete·rollback이 없다. |
| 2026-07-22 | Nijjer, [World Model Remembers, Actor Forgets](https://arxiv.org/abs/2607.19749) **[B]** | 매 2,000 environment step에 prior-task dream self-imitation 50 update | MiniGrid `n=3`에서는 actor를 복구하지만 raw buffer가 무한히 남고 per-cycle cost가 task 수에 선형, lifetime에는 `O(T²)`가 된다. |

최초 공개일과 개념 의존 순서는 완전히 같지 않다. 예를 들어 Google Sleep의 초기 OpenReview
원고는 2025-09에 있었고 Nested Learning의 공식 공개는 그 뒤지만, 현재 Sleep 판본의 구조는
Nested Learning/CMS를 명시적으로 상속한다. 따라서 arXiv 번호만 보고 `Nested Learning →
Sleep`의 아이디어 발생 순서를 단정하지 말고, **판본별 공개일과 현재 방법의 dependency**를
각각 기록해야 한다.

### 2.7 2026년의 핵심 구현과 경계 반례

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

기본값으로 6시간마다 hot event cache를 episodic vector store와 semantic graph로 옮기며 deduplication,
importance, TTL/interference forgetting, engram maturation, retrieval-triggered reconsolidation을
적용한다. threshold는 benchmark에 직접 fit하지 않는 synthetic calibration으로 정한다.
13K VSCode issue/120K event에서 **dedup 기반 consolidation**이 store 58% 감소와 97.2%
retention precision을 냈다고 보고한다. ablation evidence가 있는 것은 네 개 mechanism이고,
maturation·reconsolidation은 benchmark에 반복 retrieval/cross-session contradiction이 없어
아직 design rationale에 가깝다. 이는 “Microsoft도 sleep 관련 연구가 있는가?”에 대한
**예**이지만, model이 distillation/RL로 자기 weight를 배우는 사례는 아니다.

#### E. Microsoft MEMENTO — learned compaction이지만 sleep은 아니며, 기억의 경계를 바꾼다

MEMENTO는 OpenThoughts-v3 reasoning trace를 atom으로 나누고, GPT-5.x가 local boundary를
0–3으로 채점한 뒤 dynamic programming으로 global segment를 정한다. 별도 compressor와
6차원 judge가 block summary를 만들고 최대 두 번 feedback한다. 한 번 생성한 summary의
rubric pass는 `28%`였지만 두 번의 feedback 뒤 `92%`가 되었고, 공개
[OpenMementos](https://huggingface.co/datasets/microsoft/OpenMementos)는 정확히
`228,557`개 trace다. 이미 reasoning 능력이 있는 모델은 full-attention SFT 다음
block-masked SFT를 31K sample, 32K length, stage당 5 epoch로 학습한다. Qwen3-8B에는
DolciMath rule reward의 CISPO를 추가했다.

하지만 deployment transform은 한 answer generation 안에서 일어난다. block을 쓰고 memento를
만든 뒤 원 block KV를 free하며, 그 memento를 바로 다음 reasoning token이 사용한다.
따라서 배포 후 경험을 나중 wake에 넘기는 sleep이 아니라 `Q/W`다. 그럼에도 이 논문은
본 연구의 용량·infra framing에 중요한 반례를 제공한다.

- 보이는 memento text를 똑같이 유지하고 원 block을 보지 않은 KV로 re-prefill하면 AIME24가
  `66.1%→50.8%`로 떨어진다.
- random 5-digit secret은 원 block이 mask된 뒤 downstream memento KV에서도 chance보다 높게
  복원된다.
- uniform-attention model의 peak KV는 대체로 2–3배 줄지만, OLMo-3 AIME'26에서는 peak
  `.91x`인데 KV memory-time AUC는 `1.02x`다.

즉 memory object는 `summary text`가 아니라
`(text, source-conditioned KV, model/tokenizer, mask/position history)`일 수 있다.
원 token을 물리적으로 지웠다는 사실은 semantic erasure나 privacy deletion을 보장하지
않고, text만 옮겨 재시작하면 같은 memory가 아니다. sleep system의 manifest·capacity law·
offload/migration·RTBF benchmark는 이 implicit state를 반드시 포함해야 한다.

#### F. Microsoft RHO — completed trajectory를 실행 가능한 장기 절차로 바꾸는 sleep

RHO는 base weight나 episodic fact를 갱신하지 않는다. 대신 완료된 task trajectory를
사후 분석해 다음 task에서 계속 쓰일 **전체 agent harness**—instruction, natural-language
skill, executable tool script—로 바꾼다.

```text
past trajectories
→ difficulty judge + DPP로 k=10 coreset
→ task당 G=3 재실행
→ self-validation + cross-rollout consistency diagnosis
→ N=3 complete-harness proposal
→ candidate별 coreset 재실행 + pairwise self-preference
→ 평균 score가 양수인 최고 후보만 publish
```

원 trajectory와 task outcome이 이미 끝난 뒤 update하고 held-out task가 새 harness를
읽으므로 `C1=C2=C3=Y`, `B/P`의 **experimental PASS**다. GPT-5.5 high에서 held-out
pass rate는 SWE-Bench Pro `.59→.78`, Terminal-Bench 2 `.71→.76`, GAIA-2
`.29→.37`로 변했다. 다만 선택된 SWE 후보 `.78`은 세 후보 평균 최적 `.79`가 아니었고,
same-model self-preference는 독립 검증이 아니다.

비용도 memory write 한 번으로 숨기면 안 된다. 논문 설정의 selection 이후 한 round는
before-rollout 30, diagnosis 10, proposal 3, after-rollout 30, rank 30으로 **103 agent
invocation**이며, difficulty selection에는 100 judge call과 batched embedding도 든다.
보고된 full RHO run은 합산 agent time `23.1 h`, 10-way 동시 실행 wall time `3.2 h`다.

RHO는 사용자의 framing에서 중요한 네 번째 목적지, 즉 **절차적 control-plane memory**를
명확히 한다. 동시에 한 round의 \(k=10,G=3,N=3\)과 32 KB `AGENTS.md` 제한은 전체
harness·tool·content-addressed version·trajectory·diagnosis·backup·log를 bound하지 않는다.
과거 trajectory의 prompt injection이 영속 executable behavior가 될 수 있으므로, 실제
sleep cluster는 frozen governance kernel, sandbox, 독립 canary, provenance, atomic publish,
exact rollback을 self-preference 밖의 강제 조건으로 가져야 한다.

#### G. Memento 계열 — paper의 wake learning과 나중에 생긴 sleep code를 분리한다

[Memento 2](https://arxiv.org/abs/2512.22716)는 frozen LLM 주위의 case memory,
Parzen policy \(\mu\), \(Q\)를 `read→act→feedback→write`로 갱신한다. 같은 task의
feedback이 같은 task loop를 바꾸므로 `C1=Y,C2=N,C3=이론적 Y`, 즉 wake learning이다.
가치오차 bound와 two-timescale convergence theorem은 유용하지만 finite current memory,
stationary evaluation dynamics, bounded iterate, compact attractor 같은 조건을 둔다. 고정
용량에서 coverage radius를 0으로 보내기 위한 coreset·merge·eviction theorem이나 실험은
없다.

[Memento-Skills](https://arxiv.org/abs/2603.18743)도 논문상 sleep이 아니다. 실패한 현재
질문의 ground truth를 judge에 넣어 skill을 고치고 **같은 질문을 retry**한다. 외부
Markdown/code skill이 다음 task에도 남는 것은 맞지만 transform은 현재 task path에 있다.
별도 router는 약 8K catalog에서 약 3K skill을 sample하고 synthetic goal과 judge filter로
Qwen3-Embedding-0.6B를 InfoNCE 학습한다. 논문은 single-step offline RL이라고 부르지만
실제 objective는 supervised contrastive contextual-bandit fitting에 가깝다. paper-time
repository에는 router train/eval, checkpoint, split/log가 없고 runtime은 모든 local skill
description을 selector prompt에 넣을 수 있어 catalog가 커질수록 wake cost가 증가한다.

중요한 반전은 **논문 34일 뒤**다. 2026-04-22 repository `v0.3.0`에 추가된
DreamDaemon은 response가 끝난 뒤 session summary를 `_staging.md`에 쌓고, quick async
path 또는 600초 daemon scan이 topic Markdown을 create/update/delete해 index를 다시
만든다. 다음 wake에는 index와 최대 3개 topic이 들어가므로 이 코드 경로는
`C1=C2=C3=Y`다. 그러나 논문 benchmark의 원인으로 소급할 수 없고 Dream 성능 실험도 없다.
topic write/delete, index rewrite, staging clear가 하나의 transaction이 아니며 hard cap,
source provenance, tombstone fan-out, rollback도 없다.

#### H. Evolve — strict external sleep이지만 semantic transfer 증거는 아니다

[Evolve](https://arxiv.org/abs/2604.23424)는 Qwen3.5 2B가 42-category section memory를
읽고, miss에서는 GLM-4.7 teacher가 staging section을 만든다. query path의 acquire와
expired-hit refresh는 현재 query에서 바로 사용되므로 `Q/W`다. 별도 sleep은

```text
ephemeral/expired staging 제거
→ canonical overlap이 없으면 promote
→ 같은 category cosine ≥ .85면 teacher가 0/1/many replacement로 compile
→ 같은 cycle 뒤 항목도 새 canonical과 다시 비교
→ 전체 loop 성공 뒤 staging clear
```

를 수행한다. manual sleep과 기본 03:00 scheduler가 있고 다음 query가 canonical을 읽으므로
이 transform은 외부기억 **strict PASS**다. custom/NQ/TriviaQA에서 한 cycle 뒤 canonical
section 수가 각각 `442→300`, `471→324`, `543→361`로 줄고 teacher call/query도
`1.25→.58`, `1.15→.74`, `1.26→.96`으로 감소한다고 보고한다. 다만 section
snapshot·compile log가 없어 compaction 수치를 공개 artifact만으로 재계산할 수 없고,
teacher-call counter는 malformed/API/429 retry와 다른 model call·token·비용을 세지 않는다.

하지만 released resource를 대조하면 cold, warm, post가 네 benchmark 모두 **같은 250개
질문을 같은 순서**로 반복한다. 따라서 낮아진 teacher call은 exact repeated-key cache
amortization을 증명하며, unseen paraphrase·composition·correction으로의 지식 전이를
증명하지 않는다. MMLU에서는 baseline `70.4`가 suppressive memory `66.8`, augmentation
`68.4`로 낮아지는 negative control도 있다.

sleep 전후 score 변화 자체는 custom `+1.0 pp`, NQ `-0.6 pp`, TriviaQA `+1.2 pp`로
작고 mixed-sign이며 parallel no-sleep lifecycle이 없다. 따라서 sleep이 정확도를
인과적으로 높였다고 말할 수 없다. latency도 Evolve condition은 suppress와 augment
generation 두 번을 timer에 넣고 baseline은 한 번만 생성하므로 matched workload가 아니다.

더 큰 system 공백도 있다. vector store와 SQLite metadata는 순차적으로 갱신되어
cross-store atomic transaction이 아니고 foreground query를 멈추는 global lock도 없다.
vector file은 정상 종료 때 snapshot되고 SQLite는 즉시 바뀌므로 crash durability도
갈라진다. scheduler는 기본 disabled이고 persistent interactive app에서만 local 03:00에
붙으며 batch/time/call budget과 checkpoint가 없다.
section schema에는 source URL/span, parent lineage, model/prompt digest, tenant/auth scope가
없다. staging은 지워져도 novel canonical과 unqueried expired canonical은 계속 남을 수
있다. 한 번의 31–34% active compaction은 lifetime equilibrium, causal delete, crash
recovery를 뜻하지 않는다.

### 2.8 낮은 증거 등급의 side branch

핵심 읽기 목록에는 넣지 않되, “sleep”이라는 이름을 쓴 연구의 범위를 놓치지 않기 위해
다음도 ledger에 남긴다.

| 시점 | 연구 | 판정 |
|---|---|---|
| 2025-06-23 / 2025-10-16 철회 | [Sleep Enhanced Latent Replay for SNNs](https://arxiv.org/abs/2507.02901) **[withdrawn]** | compressed binary latent replay+noisy sleep 아이디어. 저자가 data error와 추가 실험 필요를 밝히며 철회했으므로 기존 정확도·메모리 수치를 근거로 쓰지 않는다. |
| 2025-08-29 / 2026-01-07 철회 | Ji & Song, [MyGO](https://arxiv.org/abs/2508.21296) **[withdrawn]** | wake generative memory→sleep pseudo-data distillation을 제안했지만, 저자들이 generative replay의 stability와 high-dimensional scalability 문제를 확인해 fundamental revision 전까지 철회했다. 기존 Split-MNIST/AG News 수치를 근거로 쓰지 않는다. |
| 2025-08 | [Sleep-like and Awake Rehearsal for Equilibrium Propagation](https://arxiv.org/abs/2508.14081) | SRC와 awake rehearsal을 결합한 preprint. long lifetime capacity·독립 재현 없음. |
| 2026-03 | [Slumbering to Precision](https://arxiv.org/abs/2603.07867) | sleep-like post-processing으로 calibration 개선을 주장. 장기기억·catastrophic forgetting 연구는 아니다. |
| 2026-03-15 | Ying Xie, [Learning to Forget: Sleep-Inspired Memory Consolidation for Resolving Proactive Interference in Large Language Models](https://arxiv.org/abs/2603.14517) (SleepGate) | attention-entropy 또는 conflict-density threshold와 periodic budget으로 trigger되는 tag/forget/compress/consolidate architecture를 제안하지만 실험은 soft attention bias만 구현해 실제 eviction·compression·consolidation을 수행하지 않는다. base 793,344 / full 917,313-parameter toy transformer와 synthetic retrieval에 한정. |
| 2026-06 | [Sleep to Forget](https://www.biorxiv.org/content/10.64898/2026.06.15.732460v1) | slow-wave dynamics가 consolidation과 forgetting을 조절한다는 생물물리 계산 preprint. 실험 데이터가 아니다. |

---

## 3. 세 계보를 같은 표로 비교

아래 표는 진짜 sleep뿐 아니라 sleep system 설계에 필요한 adjacent consolidation operator도
함께 놓는다. `runtime` 열에서 구분한다.

| 연구 | wake에서 모으는 것 | consolidation / adjacent operator | 쓰는 곳 | 학습 신호 | runtime | 가득 찰 때 |
|---|---|---|---|---|---|---|
| FearNet 2018 | recent class exemplars | Gaussian pseudo-replay+HC data→mPFC finetune | long-term network weights | classification+reconstruction | 매 10 study-session offline sleep | HC를 clear; class statistics와 weight interference는 남음 |
| Progress & Compress 2018 | current task experience | active column→knowledge-base distillation+online EWC | fixed shared weights | task loss+policy distillation+EWC | task-boundary compress | fixed capacity에서 decay factor로 graceful forgetting |
| SIESTA 2023 | PQ latent features+labels | balanced latent replay로 upper layers backprop | bounded latent buffer+weights | classification/replay | 명시적 periodic sleep | most-populated class에서 random eviction; 2.02 GB |
| WSCL 2023/2025 | STM/LTM examples+external dream data | NREM replay 후 REM unseen-data training | ResNet weights | supervised classification | task-boundary NREM/REM | reservoir memory; external labeled dream set 필요 |
| PCMC 2024 | raw patch exemplars+centroids | contrastive encoder retraining, centroid re-embed, targeted prune | STM/LTM+encoder | patch contrastive loss | periodic 300-epoch sleep | probabilistic prune로 약 30% 절감; raw patches는 필요 |
| Spens et al. 2025 | task별 recent episodes/state | task-specific action set에서 recurrent PPO가 offline action 선택; image/maze만 learned valuation 뒤 deterministic MMR reranking | memory, model, task learner | next-wake reward→PPO; Shapley-style marginal utility→value estimator; MMR reranking(image/maze only) | 명시적 offline sleep | 서로 다른 toy setup; relational만 world-model reset, image는 validation set 가정; lifetime cap·delete·rollback 없음 |
| PLOS 2022 | 새 task 감각·보상 stream; 과거 raw-trace buffer 없음 | spontaneous H→O replay + unsupervised STDP | H→O synaptic weights | local plasticity | 명시적 offline | old trace가 지워지기 전에 자야 함; 성장 없음 |
| PAD 2022 | episodic latent | perturb/reconstruct; mix+adversarial dream | encoder/generator weight | reconstruction+GAN | 명시적 NREM/REM | 명시적 lifetime budget 없음 |
| MemGPT 2023 | message/tool history | pressure summary | text DB + core context | prompt heuristic | wake/online | bounded main FIFO→recursive summary; raw recall은 무기한 보존되고 archival cap/eviction은 미정 |
| Zep 2025 | episode/entity/fact | resolve, dedup, temporal invalidation, optional community summary | temporal graph | prompted LLM | ingest-time; community는 opt-in/수동 rebuild | hard graph capacity/automatic forgetting 없음; raw episode와 invalidated fact도 보존 |
| Letta 2025 | persistent context/docs | anticipatory rewrite / memory editing | token-space core+external | prompting | 명시적 async sleep | revision; formal eviction theory 없음 |
| Mem0 2025 | message pair + recent history | extract, ADD/UPDATE/DELETE/NOOP, summary refresh | vector text / graph | prompted LLM | hot-path + async summary | 2025 논문에 hard cap 없음; V3는 ADD-only 성장+expiry/manual cleanup |
| ReasoningBank 2025 | self-judged success/failure trajectory | generalizable strategy extraction+append; MaTTS contrast/refinement | external reasoning bank | LLM judge+task outcome | wake/post-episode | sophisticated consolidation은 future work; hard cap/delete/rollback 없음 |
| Cartridges 2025 | corpus | synthetic QA + context distillation | reusable KV cache | teacher ICL | offline corpus compilation | corpus별 fixed cartridge; CAS에서 modular offload |
| SEAL 2025 | knowledge/task examples | self-edit→SFT | model weight | post-update downstream reward | update-time inner loop | 반복 update의 forgetting 문제 |
| MemEvolve 2025 | completed task batch+success/token/latency logs | fixed-LLM Pareto evolution of Encode/Store/Retrieve/Manage code | executable provider+content store | task accuracy first, cost/latency Pareto feedback | outer batch-boundary experimental sleep; inner content wake | search round는 finite지만 generated store·timestamp backup에 공통 cap 없음 |
| Memento 2 2025 | current task case+feedback | foreground Parzen policy/\(Q\) update around frozen LLM | external cases+controller state | same-task outcome | wake learning; sleep 아님 | finite-memory theorem은 있지만 fixed-budget coverage/eviction mechanism 없음 |
| MemSkill 2026 | offline episodes+100-step hard-case batches | PPO skill selector+fixed-LLM skill designer | controller weights+≤30 NL skills | delayed task reward | predeployment; episode memory는 wake | reward/usage eviction+snapshot recovery; episode content cap 없음 |
| Memento-Skills paper 2026 | failed attempt+ground-truth-assisted judge | skill rewrite/create 후 같은 question retry; 별도 InfoNCE router | external Markdown/code skills+router | current-task answer+contrastive relevance | wake learning; sleep 아님 | skill `5→41/235`; cap·versioned delete·released router pipeline 없음 |
| ALMA 2026 | completed collection/deployment tasks+design scores | open-ended code search over DB/update/retrieve; post-task content update | executable design+external task memory | deployment success | outer/dynamic loop experimental sleep | design archive/generated store 공통 cap·semantic rollback 없음 |
| Meta SMF 2025 | new fact batch | TF-IDF-selected sparse gradient update | memory slots | QA/task loss | continual update; sleep 미정 | slot pool 유한; eviction 미해결 |
| AgeMem 2026 | task trajectory | learned memory-tool actions | STM/LTM text store | step-wise GRPO | wake/online | LTM Add/Update/Delete, STM Summary/Filter; hard LTM cap/full trigger 없음 |
| PAHF 2026 | live clarification+correction | preference add/revise on drift | per-user external memory | human/simulated feedback | wake/online | bounded capacity·offline abstraction·causal erase 없음 |
| Microsoft 2026 | hot event cache | default 6-hour dedup/forget/mature/reconsolidate | vector store + semantic KG | calibrated thresholds | 명시적 periodic sleep | store reduction, TTL/interference |
| MemMA 2026 | completed dialogue session+5 synthetic probes | frozen-store QA verification→fact repair→semantic skip/merge/insert | external text/vector memory | probe failure+LLM judgment | session-boundary experimental sleep | top-\(k\) view만 bounded; total bytes·lifetime curve·physical erase 없음 |
| MEMENTO 2026 | 현재 response의 completed reasoning block | full-attention SFT→block-masked SFT→optional CISPO로 text+KV state compaction 학습 | 같은 call의 memento text+source-conditioned KV | imitation, attention constraint, rule-based math reward | wake/within-query; sleep 아님 | peak KV는 감소하지만 hidden KV가 source 정보를 보존; restart·AUC·privacy까지 측정해야 함 |
| Memento-Skills DreamDaemon 2026 | post-response summary staging | quick async/600초 scan이 topic create/update/delete+index rebuild | external topic Markdown+index | prompted LLM inference | post-paper code strict sleep | Dream 실험·hard cap·provenance·atomic file/index/staging transaction 없음 |
| Evolve 2026 | query-acquired staging section+TTL/category | `.85` overlap teacher compile, manual/03:00 scheduler | vector section+SQLite metadata | teacher inference+similarity | explicit external strict sleep | 같은 250 query 반복; unseen semantic transfer·cross-store atomicity·lifetime cap 없음 |
| RHO 2026 | completed unlabeled trajectories+outcomes | DPP coreset, group re-solve, self-diagnosis, 3개 full-harness proposal, pairwise self-preference | persistent instruction+skill+executable tools | same-model comparative task score | completed-trajectory experimental sleep | 한 round \(k/G/N\)만 bounded; harness·version·tool·trajectory·log와 독립 promotion gate는 미해결 |
| Surprise-Gated Memory 2026 | high-surprise labeled embeddings 또는 one-shot facts | full interleaved replay→slow linear readout; fast fact→prototype/familiarity migration | frozen backbone+episodic buffer+slow light module/store | prediction error, labels, replay loss | periodic/fast-full experimental sleep | admission은 한 buffer를 절반화하지만 full replay는 lifetime work 증가; recent window는 파괴적 |
| MemCompiler 2026 | teacher trajectories+current Brief State | SFT+GRPO로 per-step text/16 Soft-Mem token compile | transient guidance+compiler parameter | binary episode success | wake/query-time; sleep 아님 | Brief State/soft token만 bounded; read-only trajectory bank는 무제한 |
| Auto-Dreamer 2026 | cross-session typed bank | inspect provenance, synthesize compact replacements | external procedural memory | end-to-end task reward, GRPO | 명시적 offline | old region을 fresh compact set으로 대체 |
| Offline Recurrence 2026 | 한 input sequence의 recent context | learned recurrent passes | sequence-scoped fast weight | meta-learned local update | window 사이 periodic offline | KV clear; fast-weight capacity 유한, 새 sequence에서 reset |
| Google Sleep 2026 | fast CMS knowledge | upward distill+RL imitate; synthetic dreams+LoRA/RL | new slow MoE expert + model | GKD, semantic/edit reward, downstream reward | 명시적 offline | fast expert reset, slow expert growth; pool은 결국 유한 |
| Useful Memories 2026 | streamed text episodes/solutions | repeated LLM rewrite or retain/delete/consolidate | external natural-language memory | downstream task utility | stream/boundary comparison | utility가 상승 후 하락; raw retention이 forced consolidation보다 강함 |
| Facts in Weights 2026 | invented facts+study augmentation | 매 20 write 누적 fact를 fresh model에 batch distill | LoRA-merged model weights | SFT/context distillation | periodic sleep-like experiment | 100-write retention `25%<28%` no-sleep; canonical external copy 필요 |
| Discovery by Dreaming 2026 | old/new cross-domain data | synthetic replay·counterfactual·edge cases→LoRA | adapter weights | supervised offline tuning | 명시적 offline dream | narrow positive; model/rank/domain/dose에 민감 |
| MAGE 2026 | action-observation path+branches | Grow/Compress/Maintain/Revise execution tree | external hierarchical state | prompted validation | wake/within-task | active-path context는 bounded; lifetime store/delete/periodic sleep 없음 |
| TrustMem 2026 | old state + candidate update | verify/rank Write/Revise/Prune/no-action candidates | external memory | preference-guided RL | chunk-online; sleep 아님 | learned Prune+bloat penalty; hard cap·version/provenance·offline scheduler는 없음 |
| OAS 2026 | raw memory set + budget | retain/merge/abstract/rewrite selection | query context pack | offline ridge fit+held-out harm calibration; actions는 query-time | query-time; sleep 아님 | query-pack budget만 다루며 persistent lifecycle/eviction은 없음 |

---

## 4. Sleep-time training은 실제로 무엇을 학습하는가

### 4.1 Replay와 distillation

가장 보수적인 방법은 과거 표본 또는 그 압축본을 다시 보여 주는 것이다.

- **raw replay**: 가장 충실하지만 storage/privacy가 가장 비싸다.
- **mixed on/off-policy replay**: CLEAR는 recent trajectory와 buffer sample을 섞고 과거
  policy/value behavioral cloning으로 drift를 막는다. 효과적인 wake baseline이므로 sleep
  replay는 반드시 같은 replay bytes와 gradient budget으로 이겨야 한다.
- **latent/generative replay**: REMIND와 DANN 주장처럼 feature/generator만 남긴다.
  MyGO도 이 경로를 제안했지만 stability·high-dimensional scalability 문제로 철회됐으므로
  positive evidence가 아니라 failure warning으로만 읽는다.
- **teacher sampling**: Google Sleep의 Knowledge Seeding처럼 pre-expansion self가 corpus를
  생성한다. raw history를 보존하지 않아도 되지만 teacher의 오류와 망각도 복제한다.
- **context distillation**: Cartridges가 full-context teacher의 기능을 synthetic dialogue로
  작은 KV state에 옮긴다.
- **cross-session abstraction**: Auto-Dreamer가 여러 trajectory에서 공통 procedure를 추출한다.
- **periodic full redistillation**: `Facts in Weights`는 매 20 write마다 누적 fact 전체를 fresh
  base에 다시 증류했지만 100-write retention을 개선하지 못했다. “주기적 sleep” 자체가
  addressability를 보장하지 않는 negative control이다.

핵심 설계 질문은 “무엇을 replay할까?”보다 **어떤 future query distribution을 보존하도록
replay data를 만들까?**다. uniformly replay하면 희귀하지만 중요한 사건을 놓치고, importance
sampling은 잘못된 중요도 estimator를 영구화할 수 있다.

### 4.2 Synthetic dream / dataset augmentation

| 방식 | synthetic data 생성 | 학습 목적 | 위험 |
|---|---|---|---|
| PAD REM | episodic latent 혼합 + spontaneous noise | semantic separation | generator bias, 이미지 한정 |
| PAD NREM | replay latent에서 occluded reconstruction | robustness | 특정 augmentation에 과적합 |
| MyGO (withdrawn) | task별 generative memory에서 pseudo-example | old-task retention | 저자 확인 generative-replay instability; 결과를 근거로 사용하지 않음 |
| Cartridges | source corpus에서 synthetic conversation/QA | full-context teacher를 fixed KV state로 distill | teacher coverage와 corpus별 compilation cost |
| SEAL | model이 self-edit와 augmentation directive 생성 | post-update 성능 | self-training collapse, expensive inner SFT |
| Facts in Weights | recall·paraphrase·application·composition·contrast를 포함한 24-item study packet | weight write의 addressability 증가 | 100-write에서도 25–28% plateau; 외부 canonical copy 필요 |
| Google Dreaming | random expert routing으로 기억 조합, gradient filter | novel curriculum, capability refinement | 무관한 기억 결합, reward hacking |
| Auto-Dreamer | provenance를 읽고 compact procedural replacement 생성 | future agent success | detail loss, source bias |
| Discovery by Dreaming | cross-domain replay·counterfactual·edge-case 생성 | offline LoRA에 새로운 연결 학습 | narrow positive와 다수 null; dose 초과 시 역전 |

좋은 dream dataset에는 최소한 다음이 필요하다.

1. **coverage**: 최근 경험뿐 아니라 rare/old/negative case를 포함한다.
2. **novel recombination**: 단순 복사 외에 counterfactual·cross-episode composition을 넣는다.
3. **grounding**: 원본 provenance 또는 simulator/tool outcome으로 검증한다.
4. **anti-collapse mixture**: raw, teacher, student-on-policy, adversarial, random sample을 섞는다.
5. **holdout future task**: dream 자체의 likelihood가 아니라 sleep 후 미래 성능으로 평가한다.

### 4.3 RL

sleep policy의 RL은 네 층으로 나뉜다.

- **content policy**: 어떤 memory/dream을 만들지. SEAL과 Google Dreaming.
- **operator policy**: retain/merge/abstract/rewrite/delete 중 무엇을 할지. AgeMem, OAS.
- **orchestration policy**: 이번 sleep step에서 recent replay, world-model update, generated
  learning, graph operation, no-op 중 무엇을 할지. Spens et al.의 NeurIPS 2025
  meta-controller가 task-specific action set에서 recurrent PPO로 이 문제를 직접 다룬다.
  episode valuation+MMR은 image/maze 실험에만 적용되므로 하나의 범용 controller가 모든
  operator를 동시에 학습했다는 식으로 일반화하지 않는다.
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

consolidation 자체도 용량과 오류를 누적한다. Zhang et al.의 2026 counterevidence에서는
반복 LLM rewrite의 utility가 상승한 뒤 하락했고, forced consolidation보다 raw episodic
retention이 강했다. 따라서 raw episode는 삭제 정책이 허용하는 동안 canonical source of
truth로 남기고, semantic/vector/graph/latent/parametric 결과는 provenance와 version을 가진
재구축 가능한 derived view로 취급해야 한다. “active prompt가 작다”와 “전체 memory가
bounded하다”도 서로 다른 주장이다.

### 5.1 용량은 하나가 아니라 최소 다섯 문제다

1. **storage capacity**: bytes, vector count, graph edge, KV slot, adapter/expert 수.
2. **retention/representational capacity**: 같은 weight·slot·summary가 몇 개의 pattern을
   간섭 없이 보존할 수 있는가.
3. **addressability/application capacity**: 다음 wake에서 context·HBM·latency budget 안에
   필요한 기억을 찾아 실제로 읽고 적용할 수 있는가. 외부에 무한히 저장해도 이것이
   작으면 기억하지 못한 것과 같다.
4. **plasticity/acquisition capacity**: old-task retention이 정상일 때도 고정된 data·step
   budget으로 새 task를 fresh control만큼 배울 수 있는가. retention만 재면 이 축의
   선행 붕괴를 놓친다.
5. **governance capacity**: provenance, consent, TTL, unlearning, conflict, access control,
   rollback을 관리할 수 있는 양.

여기에 sleep service throughput과 hot-state residency가 운영상 별도 병목으로
붙는다. 따라서 “몇 개를 기억하는가”를 단일 scalar로 보고한 결과는 어느 capacity를
측정했는지 분해하지 않으면 scaling law 근거가 될 수 없다.

절차적 기억도 예외가 아니다. RHO의 coreset·rollout·candidate 수는 한 update의 검색
비용만 제한한다. active instruction이 작아도 executable tool, 이전 harness version,
source trajectory, diagnosis, backup, run log까지 포함한 transitive artifact graph는 계속
자랄 수 있다. 따라서 `prompt bytes`와 `procedural lifetime state`를 분리 계측해야 한다.

Evolve와 Memento-Skills는 외부 text/skill memory에서도 같은 착시가 생김을 보인다.
Evolve는 staging을 비우고 한 cycle canonical을 31–34% 줄였지만 novel topic과 query되지
않은 expired canonical은 계속 남을 수 있다. DreamDaemon이 next wake에 최대 3개 topic만
넣어도 전체 topic·skill·index·version·source session은 bounded하지 않다. 특히 모든 skill
description을 selector prompt에 넣는 flat catalog path에서는 실행에 주입되는 artifact 수가
작아도 selection token과 latency가 catalog size에 따라 증가한다. **bounded output,
bounded active view, bounded total state, bounded selection work**는 네 개의 서로 다른
주장이다.

GAM은 이 차이를 더 직접적으로 보인다. 2,048-token progression buffer는 고정돼도
session 3→27에서 event/topic graph와 edge가 계속 늘고 raw graph는 archive된다. RecMem은
construction token을 크게 줄이지만 verbatim subconscious store가 unbounded이며 이를 빼면
한 score가 `81.10→51.88`로 떨어진다. TiMem은 recalled context를 줄이는 대신 higher-level
view와 construction call을 추가한다. 따라서 **bounded hot buffer, cheap construction,
small recalled context** 어느 것도 total-state equilibrium의 대리변수가 아니다.

여기서 `state`는 보이는 memory object만 뜻하지 않는다. MEMENTO는 summary text를 완전히
동일하게 둬도 source-conditioned KV를 re-prefill KV로 바꾸면 utility가 크게 달라지고,
이미 mask된 source의 secret이 downstream KV에서 복원될 수 있음을 보였다. 따라서
`complete retained state = explicit text/graph + implicit KV/recurrent state + model/mask
dependency + recovery copies`로 잡아야 한다. peak resident bytes와
\(\int B_{\mathrm{KV}}(t)dt\)도 별개다. peak가 작아도 generation이 길어지면 총
memory-time은 커질 수 있다.

또한 retained byte와 build 중 peak byte를 분리한다. candidate/staging,
reader-pinned predecessor, dual-index migration, build/merge workspace, temporary
replica를 더한 \(B_{\mathrm{peak}}(t)\)가 실제 hard-cap 대상이다. 반면 누적
rewrite/construction/validation work는 byte에 더하지 않고 별도 resource ledger로
기록한다.

`When Continual Learning Moves to Memory`의 핵심은 외부기억이 3번을 해결하지 못한다는 것이다.
제한된 context 안에서 old/new memory가 retrieval을 두고 경쟁하므로 stability–plasticity
문제가 **weight interference에서 retrieval interference로 이동**한다.

### 5.2 유한 상태가 주는 피할 수 없는 하한

frozen base checkpoint와 공개 protocol을 \(\Theta_0\), 평가 stream을 담을 수 있는
모든 **증분 mutable state**를 \(M_n\)이라 하자. 여기에는 text/vector/graph뿐 아니라
latent, weight delta, generator, index, optimizer/router state, provenance, checkpoint도
포함한다. \(M_n\)은 finite-precision digital state이고
\(H(M_n\mid\Theta_0)\le b_{\mathrm{mutable}}\) bit이며 controlled answer
\(Y_i\)가 선언한 \((Q_i,\Theta_0)\) source model 아래 조건부 독립이라면, 평균
distortion \(D_i\) 이하의 joint recall에는 최소한

\[
\sum_{i=1}^{n}R_{Y_i\mid Q_i,\Theta_0}(D_i)
\le I(Y_{1:n};M_n\mid Q_{1:n},\Theta_0)
\le H(M_n\mid\Theta_0)
\le b_{\mathrm{mutable}}
\]

가 필요하다. frozen base의 물리적 byte는 별도 보고한다. base가 stream을 담도록
변하면 delta 또는 완전한 versioned checkpoint를 \(M_n\)에 과금해야 한다. 자연어 fact
수의 보편 법칙은 아니지만, 양의 새 정보가 계속 들어오는 opaque-payload 실험에서는
명확히 검증할 수 있는 necessary bound다. 따라서 고정 mutable state system이
open-ended stream을 영원히 같은 충실도로 기억하는 것은 불가능하다. 언젠가는

- 전체 상태를 늘리거나 외부 tier로 옮겨 그 비용을 명시하고,
- 실제 중복·prior를 이용해 joint rate를 낮추거나,
- admission/retention horizon을 제한하고,
- 일부 기억의 distortion을 높여 요약·만료·망각해야 한다.

bounded hot set와 무한히 커지는 archive는 serving capacity만 제한할 뿐 lifetime storage를
제한하지 않는다. 실제 system은 이 정보 한계에 닿기 전에도 retrieval 경쟁, interference,
plasticity, governance, sleep backlog 중 하나가 먼저 무너질 수 있다.

Memento 2의 value bound는 external-memory count를 downstream error와 연결할 후보를
준다. local consistency 등 강한 조건 아래 coverage radius \(r_M\)와 retrieval error
\(\delta_M\)이 작아지면 value error upper bound가 줄어든다. 그러나 finite memory를
가정하면서 fixed-budget coreset/eviction으로 \(r_M\)을 어떻게 유지할지는 풀지 않는다.
relevant state support가 실제로 compact하고 covering dimension \(d\)를 가진 제한된
workload에서, codec·precision·prototype당 byte를 고정할 때만
\(r_M=\Theta(M^{-1/d})\)를 순수 cardinality 후보로 둘 수 있다. total bit \(B\)를
고정하고 \(M\)을 바꾸는 별도 arm은 \(cM^{-1/d}+q(B/M)\) 같은
geometry–quantization joint model을 검증해야 한다. open-ended drift,
anisotropic relevance, merge distortion, retrieval competition이 있으면 이 관계는
깨질 수 있으며 transformer factual-capacity law로 일반화할 수 없다.

### 5.3 외부기억의 capacity mechanism

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

### 5.4 모수적 기억의 capacity mechanism

| 메커니즘 | 예 | 남는 문제 |
|---|---|---|
| replay/distillation | PLOS, Cartridges, Google Sleep | teacher 오류·data selection |
| regularization | EWC, MAS | plasticity 감소; 새 capacity 없음 |
| gradient constraint | GEM | replay buffer와 계산 증가 |
| sparse write | SMF | slot collision/exhaustion, routing |
| adapter/expert isolation | LoRA, Google Sleep | adapter/expert 수의 무한 증가 |
| modular low-rank memory | single/multi-LoRA knowledge memory | rank knee, catalog routing, incompatible merge, HBM residency |
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

누적 비용도 함께 포화한다. 매 `k` write마다 현재의 `n`개 fact 전체를 다시 증류하면
lifetime work는 대략 `O(n²/k)`이고, 매 interval 모든 prior task를 dream-replay하면 한
cycle은 `O(T)`, `T`개 task의 lifetime은 `O(T²)`가 된다. task별 adapter/bank를 계속 추가하면
storage는 `O(T)`다. delta-only consolidation, bounded replay/coreset, hierarchy,
promotion threshold, adapter merge+archive가 없으면 sleep cluster가 memory보다 먼저
backlog에 의해 붕괴할 수 있다.

Surprise-Gated Memory는 이 trade-off를 수치로 드러낸다. Novelty gate는 한 finite
buffer를 절반화했지만 full-history replay의 \(O(T^2)\) 누적 차수는 바꾸지 않는다.
최근 5 task만 replay해 \(O(wT)\)로 줄인 arm은 DINOv2에서 `41.2<67.0`,
I-JEPA에서 `0.0<25.9`로 no-replay보다 나빴다. 따라서 fixed bytes 자체가 해법이
아니며, old/rare/correction support geometry를 유지하는 class-balanced/reservoir/utility
coreset과 protected-old canary가 필요하다.

LoRA knowledge-memory 결과는 adapter 수 증가가 단순 storage 문제만은 아님을 보인다.
64K-token PhoneBook corpus에서 trainable-parameter budget을 맞추면 oracle-routed
`8×rank-4`가 rank-32 하나보다 강하지만 learned routing은 이 이득을 없앨 수 있고 naive
merge는 붕괴한다. 여기서 프로젝트의 측정용 heuristic — 논문의 정리나 물리적
capacity 식이 아님 — 을
`normalized isolated-module utility × router hit rate × composition-retention ratio
× serving hit rate`로 둔다. 네 항은 같은 held-out workload에서 `[0,1]`로 정규화하고,
composition-retention은 composed utility/isolated utility, serving hit rate는 latency
deadline 전에 선택 module이 resident인 요청 비율로 측정한다. 이 곱은 stored bits가
아닌 end-to-end 진단 score다. multi-LoRA는 vector search를 없애는 것이 아니라
**module search**로 바꾼다.

### 5.5 권장되는 hybrid policy

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

parallel external/wake branch: ReasoningBank + MaTTS
parallel systems branches: {TNT, Memory Caching, NSTM}
```

- 2016 CLS update는 fast episodic/slow structured learning의 이론적 다리다.
- Titans–Miras/Atlas–Nested Learning은 wake 동안 parameter/state를 서로 다른 frequency로
  업데이트하는 계보다. TNT·Memory Caching·NSTM은 training efficiency, checkpoint/cache,
  update/apply frequency를 다루는 병렬 systems branch다.
- ReasoningBank는 success/failure trajectory를 test time에 reasoning memory로 append하고
  MaTTS가 더 많은 interaction experience를 만드는 외부기억 branch다. Google 공식 설명도
  sophisticated consolidation은 future work라고 명시하므로 strict sleep으로 세지 않는다.
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
continual-stability basis: GEM / MAS
capacity/lifetime basis: Product-Key Memory / Expire-Span
sparse-parametric branch:
  Product-Key Memory → Memory Layers at Scale → Sparse Memory Finetuning

parallel wake-personalization branch: PAHF
```

- GEM은 episodic replay buffer로 과거 task gradient를 보호한다.
- MAS는 parameter importance에 따라 덜 중요한 지식을 선택적으로 덮어쓸 수 있게 한다.
- Product-Key Memory는 거대한 sparse KV parameter pool을 저비용 top-k lookup하는 구조를
  만들었다.
- Expire-Span은 기억의 수명을 학습한다.
- Memory Layers는 factual capacity를 sparse lookup으로 크게 늘린다.
- SMF는 새 사실에 특이적인 slot만 고쳐 dense FT보다 forgetting을 줄인다.
- PAHF는 live clarification/correction으로 explicit per-user preference memory를 갱신하고
  preference drift를 평가하지만, background consolidation이나 parameter sleep은 아니다.

따라서 블로그의 Meta 부분은 **모수적 sparse memory update**의 근거로는 유효하지만,
비동기 sleep scheduler나 NREM/REM pipeline까지 Meta가 구현했다고 읽으면 과장이다.
현재 확인한 Meta 계보에는 strict three-part definition을 모두 만족하는 공개 논문이 없다.

### 6.3 Microsoft

현재 확인되는 연구 portfolio는 explicit “sleep” 명명보다 **외부·context·generated
parameter consolidation**의 병렬 branch다. 아래 항목을 직접적인 논문 계보로 읽으면 안 된다.

```text
external KV: LongMem (Microsoft Research + UCSB)
generated LoRA: Generative Adapter (UW + Microsoft + Microsoft Research)
harmonic external store: Memora
hybrid external graph + RL: H-EPM (HKUST + Microsoft Research Asia)
post-task knowledge graph: PlugMem (UIUC + Tsinghua + Microsoft Research)
post-session probe-and-repair: MemMA (Penn State + Amazon + Microsoft)
learned within-query compaction: MEMENTO (Microsoft)
memory-harness program search: M★ (CityUHK + Microsoft)
retrospective trajectory→full harness: RHO (CityUHK + Microsoft Research Asia)
learned per-step text/latent delivery: MemCompiler (universities + Microsoft Research)
external sleep lifecycle: Human-Inspired Memory Architecture
offline procedural bank: LEGOMem (all Microsoft)
compression control: ACON (KAIST + Microsoft + University of Cambridge;
  first-author work during Microsoft internship)
execution-state tree: MAGE (Microsoft + universities)
stateful-agent benchmark: STATE-Bench software
product memory service: Azure Foundry Agent Memory
```

- LongMem은 frozen backbone과 decoupled long-term bank를 사용한다.
- Generative Adapter는 context를 LoRA로 compile한다.
- Memora는 abstraction과 specificity를 함께 보존하는 구조를 제안한다.
- H-EPM은 성공 trajectory의 tool-transition weight와 state summary를 edge에 저장하고,
  같은 graph로 GRPO rollout을 guide한다. 그러나 ACEBench의 online update는 request/task
  완료 전후 순서를 공개하지 않았고, pinned 공개 코드의 가장 가까운 online writer는
  tool call 뒤 inline 갱신한다. 따라서 외부기억↔모수학습 bridge이지만 strict sleep
  evidence는 아니다.
- PlugMem은 dialogue/document/action trace를 proposition·prescription·episodic graph로
  바꾼다. WebArena 공개 코드에서는 `env.done()` 뒤 이미 정해진 `final_status`를 보존한 채
  `memory.close()`와 graph insert를 실행하므로 `C1+C2+C3`를 만족하는 좁은
  **task-boundary experimental PASS**다. 단, synchronous tail work이며 queue/daemon은
  아니다. HotpotQA merge는 soft-deactivate로 active node를 5.0% 줄였지만 lifetime
  byte cap이나 physical erase가 아니다.
- MemMA는 한 dialogue session의 모든 turn을 처리한 뒤 5개 synthetic probe로 현재
  store를 frozen-state 평가하고, 실패를 evidence-grounded fact로 바꿔
  `SKIP/MERGE/INSERT`한 뒤 재평가한다. pinned 공개 코드에서는 다음 session 전에
  write-back되므로 좁은 **session-boundary experimental PASS**다. 그러나 paper quick
  start는 생성 과정이 아직 공개되지 않은 76-row probe parquet를 사용하며, 평가는
  LoCoMo `conv-26` 하나뿐이고 total-store growth·physical erase·rollback을 측정하지 않는다.
- MEMENTO는 228,557개 OpenMementos trace로 `full-attention SFT→block-masked SFT`를
  학습하고 optional CISPO를 더한다. runtime에는 현재 reasoning block을 summary로 만든
  직후 원 block KV를 free하므로 wake/within-query다. 다만 동일 summary text를 restart한
  ablation과 passcode probe는 summary KV가 원 block 정보를 별도 채널로 운반함을 보인다.
  따라서 text-only 저장량, source-token delete, peak KV만으로 memory capacity·portability·
  privacy를 판정할 수 없다.
- M★는 memory `Schema`, write/read `Logic`, agent `Instruction`을 Python program으로
  만들고 static/rotating validation failure로 20회 진화한다. memory 내용이 아니라
  **memory harness 자체를 학습**하는 control-plane 계보의 한 단계지만, episode ingest와
  program evolution은 held-out query 전의 predeployment search다. 이 계보는 M★보다
  먼저 MemEvolve와 ALMA에서 시작되며, M★의 차별점은 schema·logic·instruction을 함께
  진화시키는 데 있다.
- RHO는 이 계보를 실제 completed session 쪽으로 옮긴다. 과거 trajectory를 재실행·진단해
  instruction/skill/executable tool 전체를 후보화하고, 양의 pairwise self-preference
  winner를 held-out task 전에 적용하므로 좁은 **experimental PASS**다. 하지만 같은
  GPT-5.5 계열이 solve·diagnose·optimize·rank하고, adversarial trajectory가 영속
  behavior가 될 수 있다. 따라서 content-addressed version과 backup은 유용한 복구
  primitive일 뿐 독립 promotion attestation, authorization, source delete, semantic
  rollback을 대신하지 못한다.
- MemCompiler는 successful teacher trajectory로 Qwen2.5-VL-7B compiler를 LoRA SFT한
  뒤 binary episode-success GRPO로 compiler와 Soft-Mem projection만 학습한다. runtime에
  매 step current observation+Brief State+read-only task memory를 읽고 text guidance와
  16 latent token을 즉시 생성하므로 memory-use compiler이지 sleep consolidator가 아니다.
  10-item fold와 16 token은 active delivery state만 제한하고 underlying trajectory bank를
  제한하지 않는다.
- Human-Inspired Memory Architecture는 기본값 6시간 주기의 **sleep-phase consolidation**을
  명시하고 hot cache→episodic vector store→semantic KG, dedup, TTL/interference forgetting,
  maturation, reconsolidation을 설계한다. VSCode 13,127 issues/120K events에서
  **dedup-based consolidation**이 retention precision 97.2%와 store 58% 감소를 보고했지만,
  maturation·reconsolidation은 해당 benchmark가 직접 검증하지 못했고 model weight가 아니라
  외부 store를 다룬다.
- LEGOMem(AAMAS 2026)은 148개 training task의 successful trajectory에서 추출한
  93개 full-task memory와 250개 subtask memory로 고정 procedural bank를 구성하지만 반복
  배포 sleep·online growth·delete/rollback은 없다. ACON(ICML 2026)은 compression
  guideline을 offline 최적화한 뒤 wake path에서
  threshold compression을 수행하므로 durable-memory sleep이 아니다.
- MAGE는 실행 중 Grow/Compress/Maintain/Revise로 active path와 failed branch를 분리하는
  hierarchical state manager다. bounded live context와 복구 경계에는 중요하지만 recurring
  background sleep이 아니다.
- STATE-Bench v0.8.1은 Agent Learning track를 가진 Microsoft OSS benchmark이지 논문이나
  memory algorithm이 아니다. artifact capacity·forget policy를 규정하지 않고 offline
  artifact-building cost가 reported run cost 밖에 남을 수 있다.
- [Microsoft Foundry Agent Memory](https://learn.microsoft.com/en-us/azure/foundry/agents/concepts/what-is-memory)
  **[C]**는 extraction→LLM consolidation→retrieval, profile/chat-summary/procedural memory,
  CRUD·TTL·remember/forget을 제품화한다.

따라서 Microsoft에는 HIMA의 명시적 scheduled sleep architecture, PlugMem의 좁은
task-boundary path, MemMA의 session-boundary path, RHO의
completed-trajectory-to-harness path가 존재한다. 뒤의 세 경로는 periodic production
service가 아니다. H-EPM은 external graph와 GRPO를 연결하고 M★는 harness를
predeployment search하며 MemCompiler는 delivery를 학습하지만, 각각 strict
boundary·deployed lifetime·off-path 조건 중 일부가 없다. Letta처럼 token-space
pre-query reasoning을 scaling law로 만들거나 Google처럼 deployed model의 slow
parameter에 지식을 증류·성장시키는 **parametric sleep**은 현재 확인되지 않았다.

### 6.4 Letta, Mem0, Zep, Memento, Evolve

| 조직 | 현재 강점 | 아직 없는 것 |
|---|---|---|
| Letta | 2025 primary/sleep agent 분리와 token-space learned context; 2026 client memory subagent+git-backed MemFS로 file-level provenance/rollback 강화 | 원 논문에는 weight training 없음; AgentFile checkpoint가 archival passage 전체를 포함하는 whole-agent rollback 증거는 아니고 shared block update는 last-write-wins |
| Mem0 | 2025 ECAI 논문의 extract/update/delete와 vector+graph; vendor-run 평가에서 낮은 live token/latency | API delete/export/expiry는 contract이지 replica·derived-index physical erasure audit가 아님; V3 indexing은 cross-bank sleep이 아니며 ADD-only growth와 learned offline consolidator 부재 |
| Zep/Graphiti | provenance, bi-temporal KG, contradiction invalidation, opt-in community abstraction; managed Zep Observations의 background/ingest-adjacent cross-episode roll-up | episode delete 뒤 shared node summary/name이 재생성되지 않을 수 있어 causal erase가 아님; core scheduler·parametric learning이 없고 managed operator는 proprietary |
| Memento family | Memento 2의 external case/control theory, Memento-Skills의 persistent skill learning, 그리고 나중 DreamDaemon의 실제 post-response daemon path | 두 논문은 wake learning이며 Dream code가 논문 결과를 설명하지 않음; router artifact, Dream experiment, total cap, source lineage, transactional publish/rollback 부재 |
| Evolve | cold→warm→explicit sleep→post lifecycle, manual/03:00 scheduler, 공개 question/result CSV와 one-cycle section compaction | phase별 동일 ordered query라 exact-key reuse만 지지; source provenance, unseen-query transfer, cross-vector/SQLite atomicity, canonical lifetime equilibrium 부재 |

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
- **memory plane**: canonical immutable event log와 그로부터 파생된 versioned
  summary/vector/KG, cartridge/adapter, optimizer/checkpoint view를 서로 다른 tier에 둔다.
- **control plane**: sleep trigger, per-tenant budget, promotion, verifier, privacy, lineage,
  canary/rollback을 관리한다.

sleep job은 snapshot `v`를 읽어 candidate `v+1`과 source-event/model/prompt/config hash를
만든다. remember/update/forget·poison·privacy shadow evaluation을 통과한 뒤 serving pointer를
atomic swap하고, 이전 view/checkpoint는 rollback window 동안 보존한다. 삭제 tombstone은
raw-retention policy, semantic/vector/graph/latent cache와 adapter까지 fan-out한다. exact
parametric unlearning이 불가능하면 adapter를 quarantine하거나 canonical log에서 rebuild해야
한다. migration 중 wake는 versioned dual-read를 사용한다.

`atomic`은 store별 transaction을 각각 성공시키는 뜻이 아니다. vector root, relational
metadata, topic/skill file, index root, source staging range를 하나의 immutable candidate
manifest에 묶고, cross-reference와 canary를 검증한 뒤 **단 하나의 serving pointer**를
CAS해야 한다. 그 receipt가 난 뒤에만 source range를 acknowledge한다. Evolve와
DreamDaemon처럼 store/file/index/staging을 순차 변경하면 crash나 concurrent query가 mixed
generation을 볼 수 있다. recovery 기준은 all-new가 아니라 **all-new or all-old,
no-lost-event, idempotent replay, exact rollback**이다.

latent/KV artifact는 추가로 `model/tokenizer`, attention-mask·position convention,
KV layout, source snapshot, 그리고 `PORTABLE_TEXT / SERIALIZED_LATENT /
LIVE_KV_LINEAGE / REBUILD_ONLY`를 manifest에 기록해야 한다. MEMENTO처럼 text-identical
restart가 동등하지 않은 state를 summary 파일 하나로 offload하거나 model upgrade 뒤
re-prefill하면 조용히 기억이 바뀐다. 삭제 시에도 explicit artifact뿐 아니라 살아남은
KV/recurrent descendant에 opaque-secret residual probe를 실행해야 한다.

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
| live serving | P50/P95/P99, prompt/decode tokens, HBM working set, peak KV와 KV memory-time AUC |
| sleep cost | FLOPs, wall-clock, energy, generated/rejected dreams, backward pass 수 |
| storage | bytes/user/day, graph/slot/adapter growth, hot/warm/cold residency |
| movement | bytes wake→sleep, sleep→memory, memory→wake; exact-key/semantic cache hit/miss |
| memory quality | retrieval recall, temporal correctness, compression coverage, detail loss, exact-repeat↔held-out paraphrase/composition/correction gap, text-identical restart equivalence |
| reliability | omission, corruption, hallucination, stale fact, conflict, latent secret residual, rollback success |
| economics | amortized cost/query, break-even reuse count, flat-catalog selection token/latency, idle-resource utilization |

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
- direct acquisition, temporal update, reference resolution, composition,
  implicit relevance, boundary awareness를 분리한 internalization profile,
- single/routed/merged LoRA에서 oracle↔learned router gap, rank/module/catalog/HBM
  capacity를 분리한 parametric track,
- session/day/week/month clock, recurrence/surprise/utility admission,
  local↔global maintenance, bounded live↔total state의 factorial,
- egocentric action evidence의 time-anchor, participant consent, derived
  habit/workflow/preference 삭제·rebuild를 포함한 embodied track,
- MEMENTO식 stateful compaction 대 text-identical restart, layer/hop별 opaque-secret
  KV probe, peak와 AUC의 동시 계측,
- RHO식 full-harness rewrite 대 frozen governance kernel+task module, 같은 trajectory
  pool에서 반복 round·독립 judge·poison canary·artifact cap·exact rollback, 그리고
  selection 뒤 103 call을 포함한 complete search cost,
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
Surprise residual은 후보 신호지만 base-distribution threshold가 drift하면 rare-known
memory를 잘못 쓰거나 버릴 수 있으므로 calibration holdout과 false-admit/reject cost를
별도로 측정한다.

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

### RQ7. memory artifact의 보이는 내용이 complete state인가?

가설: compression pressure와 stateful reuse hop이 커질수록 explicit text와
answer-relevant/privacy-relevant latent state의 간극이 커진다.

필수 검증: 같은 summary text에 대해 source-conditioned KV, restart/re-prefill KV,
serialized latent, model-migrated latent를 비교하고, source block에만 넣은 opaque secret의
post-delete residual을 layer·hop별로 측정한다. 모두 동등하고 chance-level이면 이 가설은
반증된다.

### RQ8. sleep이 memory content가 아니라 agent harness 자체를 바꿔도 되는가?

가설: workload heterogeneity가 커지면 완전 범용 harness보다 frozen governance kernel과
task/tenant-specific module의 계층형 구조가 낫고, full-harness rewrite보다 poison·rollback
비용이 낮다.

필수 검증: RHO식 complete rewrite, fixed universal harness, governance-kernel+generated-module을
같은 completed trajectory와 held-out budget에서 비교한다. 103-call search, independent
promotion judge, executable sandbox, version/log growth, source deletion, candidate-selection
regret을 모두 charge한다. full rewrite가 모든 scale에서 우월하면 중간 계층 가설은 반증된다.

### RQ9. sleep의 amortization은 cache hit인가 semantic generalization인가?

가설: exact acquisition query를 반복했을 때의 teacher-call 절감은 unseen
paraphrase·composition·correction·distribution shift에서 체계적으로 작아지며, compiler가
query wording/category에 과적합할수록 이 간극이 커진다.

필수 검증: initial write를 만든 `ACQUISITION_KEY`, exact repeat, held-out paraphrase,
held-out composition, correction, unrelated control을 물리적으로 분리한다. 각 family의
quality-floor-conditional hit, teacher call, false merge/split/delete를 따로 보고한다.
Evolve의 동일 250-query lifecycle은 exact-repeat baseline으로만 유지한다. 간극이
compute-matched held-out family에서 사라지면 가설은 반증된다.

### RQ10. parametric memory의 진짜 병목은 rank인가 routing인가?

가설: fixed total adapter parameter에서 low-rank module isolation은 oracle routing에서는
이기지만, module 수가 커질수록 router error·merge interference·HBM miss가 이득을 상쇄하는
knee가 생긴다.

필수 검증: single rank-32와 `8×rank-4`를 같은 data/parameter에서 비교하고,
oracle/lexical/embedding/learned router, top-1/top-k, linear/concat/TIES merge, catalog
residency를 분리한다. learned router가 held-out composition에서도 지속적으로 oracle 이득을
회복하지 못하면 modular advantage는 해당 regime에서 반증된다.

### RQ11. sleep clock을 늘리면 언제 손해가 되는가?

가설: session/day/week/month clock은 처음엔 active context당 utility를 높이지만,
construction arrival와 repeated abstraction distortion가 service capacity를 넘으면
lifecycle utility가 감소한다.

필수 검증: one-clock/full-hierarchy, requested/coalesced/provider call, queue age,
local/global maintenance, exact-evidence loss를 함께 측정한다. 모든 burst/latency/cost
holdout에서 clock 추가가 Pareto improvement면 crossover 가설은 반증된다.

### RQ12. embodied memory의 삭제 단위는 무엇인가?

가설: video/action evidence에서 파생된 entity, habit, workflow, preference가 text memory보다
넓은 consent·delete fan-out을 만들며, query-time filter만으로는 post-anchor 또는 철회된
정보의 영향이 사라지지 않는다.

필수 검증: participant/video/time-span provenance를 모든 derived view에 연결하고,
time-anchor rollback, consent withdrawal, rebuild, behavioral residual을 측정한다. 파생
memory와 adapter까지 완전 삭제·재생성돼 영향이 equivalence bound 안이면 해당 fan-out
위험은 통제된 것으로 판정한다.

---

## 9. 읽기 순서

### P0 — 이 framing을 세우는 데 필수

1. [Complementary Learning Systems (1995)](https://web.stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf)
   fast episodic ↔ slow structured learning의 이론.
2. [DGDMN (2017)](https://arxiv.org/abs/1710.10368),
   [FearNet (2018)](https://openreview.net/forum?id=SJ1Xmf-Rb),
   [Progress & Compress (2018)](https://proceedings.mlr.press/v80/schwarz18a.html),
   [SIESTA (2023)](https://openreview.net/forum?id=MqDVlBWRRV),
   [PCMC (2024)](https://proceedings.mlr.press/v274/taylor25a.html):
   explicit down-time transfer, task-boundary distillation, bounded latent replay, offline
   retraining/pruning의 실제 선행 계보.
3. [PLOS SNN Sleep (2022)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010628)
   raw old data 없이 자발적 replay가 weight-manifold를 보존한다는 최소 실험.
4. [Perturbed and Adversarial Dreaming (2022)](https://elifesciences.org/articles/76384)
   NREM replay와 REM novel synthesis를 서로 다른 objective로 분해.
5. [Modelling the Control of Offline Processing with RL (2025)](https://papers.nips.cc/paper_files/paper/2025/hash/d7e5870810331da5a8ac8bd16d42e074-Abstract-Conference.html)
   learned sleep action, replay selector, adaptive curriculum이 이미 존재한다는 novelty 경계.
6. [MemGPT (2023)](https://arxiv.org/abs/2310.08560)
   virtual context와 tiered external memory의 시스템 출발점.
7. [Sleep-time Compute (2025)](https://arxiv.org/abs/2504.13171)
   pre-query token-space compilation의 정확한 정의와 amortization 조건.
8. [Memory Layers at Scale (2024)](https://arxiv.org/abs/2412.09764) +
   [Sparse Memory Finetuning (2025)](https://arxiv.org/abs/2510.15103)
   Meta의 sparse parametric capacity와 저간섭 update.
9. [Mem0 (2025)](https://arxiv.org/abs/2504.19413),
   [Zep/Graphiti (2025)](https://arxiv.org/abs/2501.13956),
   [Microsoft Human-Inspired Memory (2026)](https://arxiv.org/abs/2605.08538):
   text/vector/temporal graph의 online operator와 explicit external sleep을 비교.
10. [Cartridges (2025)](https://arxiv.org/abs/2506.06266) +
   [SEAL (2025)](https://arxiv.org/abs/2506.10943)
   context→latent와 self-generated data→weight라는 두 compilation 방법.
11. [When Continual Learning Moves to Memory (2026)](https://arxiv.org/abs/2604.27003) +
    [Useful Memories Become Faulty (2026)](https://arxiv.org/abs/2605.12978):
    external memory도 stability–plasticity와 destructive rewrite를 피하지 못한다는 반증.
12. [Auto-Dreamer (2026)](https://arxiv.org/abs/2605.20616)
    learned external offline consolidation.
13. [Google Language Models Need Sleep (2026)](https://arxiv.org/abs/2606.03979)
    multi-frequency parametric consolidation, distillation, RL, dreaming, growth/reset.
14. [Rate–Distortion View of Memory Compaction (2026)](https://arxiv.org/abs/2607.08032) +
    [Can a Language Model Learn Facts Continually in Its Weights? (2026)](https://arxiv.org/abs/2607.11020):
    cross-layer objective와 parametric sleep의 실패·canonical-copy 필요성을 함께 읽는다.
15. [Retain or Consolidate? (2026)](https://arxiv.org/abs/2607.17545) +
    [TrustMem (2026)](https://arxiv.org/abs/2606.25161)
    budget-dependent operator selection과 reliable transition learning.
16. [Retrospective Harness Optimization (2026)](https://arxiv.org/abs/2606.05922)
    completed trajectory를 persistent instruction·skill·tool로 바꾸는 control-plane
    sleep, self-preference promotion의 한계, 그리고 103-call search cost.
17. [Surprise as a Signal for Plasticity and Metacognition (2026)](https://arxiv.org/abs/2606.31495)
    surprise admission과 full/recent replay의 분리, recent-only replay의 파괴적
    counterexample. 핵심 수치는 single-run이고 코드는 없어 낮은 등급으로 읽는다.
18. [Evolve (2026)](https://arxiv.org/abs/2604.23424) +
    [Memento-Skills (2026)](https://arxiv.org/abs/2603.18743):
    strict external section sleep과 wake-time skill rewrite를 대조하고, identical-query
    amortization·catalog selection·multi-store atomicity의 숨은 비용을 읽는다. Memento-Skills
    논문과 34일 뒤 DreamDaemon code evidence는 반드시 분리한다.
19. [ImprintBench (2026)](https://openreview.net/forum?id=QIJgTW3Qd2) +
    [MemoryBench (2026)](https://arxiv.org/abs/2510.17281):
    direct recall을 acquisition/update/reference/composition/relevance/boundary awareness로
    분해하고, explicit/implicit feedback wake-learning을 sleep의 필수 비교군으로 둔다.
    ImprintBench의 수치는 exact venue-specific PDF를 pin하기 전에는 canonical claim으로 승격하지
    않는다.
20. [Understanding LoRA as Knowledge Memory: An Empirical Analysis (2026)](https://arxiv.org/abs/2603.01097):
    raw/QA/summary/rewrite/mixed sleep data, low-rank capacity knee, single/multi-LoRA,
    oracle/learned routing, merge interference를 한 번에 다루는 parametric destination의
    필수 논문.
21. [TiMem](https://aclanthology.org/2026.findings-acl.1091/) +
    [RecMem](https://aclanthology.org/2026.findings-acl.1619/) +
    [GAM](https://aclanthology.org/2026.acl-long.1600/):
    temporal multi-clock, recurrence admission, bounded live buffer→growing graph/archive를
    함께 읽어 trigger와 total capacity를 분리한다.
22. [Agent-Native Memory Systems](https://arxiv.org/abs/2606.24775) +
    [MEMORA](https://arxiv.org/abs/2607.14252):
    utility–latency/local-maintenance frontier와 embodied action evidence의 strict offline
    consolidation·temporal leakage·multimodal governance를 연결한다.

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
  [H-EPM (2026)](https://arxiv.org/abs/2512.07287),
  [PlugMem (2026)](https://arxiv.org/abs/2603.03296),
  [MemMA (2026)](https://arxiv.org/abs/2603.18718),
  [MEMENTO (2026)](https://arxiv.org/abs/2604.09852),
  [M★ (2026)](https://arxiv.org/abs/2604.11811),
  [RHO (2026)](https://arxiv.org/abs/2606.05922),
  [MemCompiler (2026)](https://arxiv.org/abs/2605.07594),
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
- deployed-experience wake branches:
  [ReasoningBank](https://arxiv.org/abs/2509.25140),
  [PAHF](https://arxiv.org/abs/2602.16173),
  [MAGE](https://arxiv.org/abs/2606.06090)
- Memento/Evolve line:
  [Memento 2](https://arxiv.org/abs/2512.22716),
  [Memento-Skills](https://arxiv.org/abs/2603.18743),
  [Evolve](https://arxiv.org/abs/2604.23424);
  case/policy theory, skill-router learning, post-paper Dream daemon, scheduled section
  compilation을 동일한 “sleep”으로 합치지 않는다.

### P2 — 아이디어는 흥미롭지만 증거 등급을 낮춰 읽기

- [DANN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5402490):
  “zero forgetting”을 headline으로 인용하지 말고 full method/code/replication을 먼저 확인.
- [Sleep-Consolidated Memory](https://arxiv.org/abs/2604.20943):
  explicit LLM sleep prototype이지만 작은 연구 preview.
- [Dreaming Is All You Need](https://arxiv.org/abs/2409.01633),
  [Wake-Sleep Consolidated Learning](https://arxiv.org/abs/2401.08623):
  생물학적 framing과 operator 아이디어는 취하되 LLM production evidence로 일반화하지 않음.
  특히 SleepNet/DreamNet은 offline sleep/continual-learning 구현이 아니며 TMLR rejected
  manuscript라는 상태를 함께 기록.
- [MyGO](https://arxiv.org/abs/2508.21296)는 generative-replay stability와
  high-dimensional scalability 문제로 저자 철회됐으므로 방법 아이디어와 failure
  mechanism만 읽고 성능 수치는 근거로 쓰지 않음.

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
| NREM/REM analogy | **PAD, Singh et al., Google Sleep이 서로 다른 계산적 역할 분담을 구현; PLOS는 단일 sleep-like state** | 생물학적으로 확정된 equivalence가 아니라 algorithmic hypothesis로만 사용 |
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
