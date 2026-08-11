# 00 — 어휘 지도: 이 분야가 실제로 쓰는 말

**작성**: 2026-08-11 · **단계**: gap-sweep 00 (뒤 단계 여섯의 입력)
**목적**: 이 책의 자기 용어(`sleep-time compute`·`세 경로`·`판별식`·`B_s`)로는 2026년 문헌을
찾을 수 없다. 공동체가 실제로 쓰는 문자열을 세우고, 그 문자열로만 검색하게 한다.
**근거**: `notes/stc-v2/*.json` 32편, `papers/stc/2607.25380.txt`(서베이 전문),
`dossier/stc-v2/OUTLOOK-REVIEW-2026-08-11.md`, 그리고 이 문서를 쓰며 실행한 검증 검색 10건.

---

## 0. 이 문서의 사용법

- §3의 문자열을 **그대로** 검색창에 넣는다. 번역하거나 다듬지 않는다.
- §4의 반용어는 **쓰지 않는다**. 쓰면 0건이거나 이미 읽은 논문만 나온다.
- §5의 false friend는 **결과를 읽을 때** 본다. 같은 단어가 다른 뜻이므로,
  초록에 `offline`이 있다고 통과 후보로 올리면 안 된다.
- 배제 목록(`corpus-exclusion.md`)은 별도다. 이 문서는 어휘만 다룬다.

---

## 1. 왜 이 책의 말로는 못 찾는가 — 측정된 증거

| 관측 | 값 | 출처 |
|---|---|---|
| 서베이 `Memory for Large Language Models`(2607.25380) 전문의 `sleep` 등장 횟수 | **0** | 본서 grep, 2559줄 20쪽 85 참고문헌 |
| 같은 전문의 `replay` / `distill` / `amortiz` / `idle` / `background` | **0 / 0 / 0 / 0 / 0** | 본서 grep |
| 같은 전문의 `offline` | **28** — 그러나 뜻이 정반대다(§5.1) | 본서 grep |
| `"sleep-time compute"` 검색 결과 | 2504.13171 한 편 + 그 블로그·팟캐스트·요약 사이트뿐. **후속 논문 0건** | 검증 검색 #6 |
| `offline memory consolidation language agents` 검색 결과 | 첫 페이지에서 미읽은 2026년 논문 **6편** 즉시 | 검증 검색 #1 |
| ReasoningBank(2509.25140) 전문의 `sleep` | **0** | `notes/stc-v2/` 침묵 지도 |
| Memory Caching(2602.24281) 전문의 `sleep` | **0** | 〃 |

**결론.** 이 책의 이름은 이 라인의 **바깥에서 붙인 이름**이다. 라인 안쪽에서 쓰는 말은
`offline consolidation`, `idle-time compute`, `memory write policy`, `experience internalization`,
`test-time training`, `sequential model editing`, `KV cache reuse`다. 검색은 그 말로 한다.

---

## 2. 하위 공동체 지도

**일곱 개다.** 과제가 최소로 지목한 넷(에이전트 기억 / test-time·ICL / 지속학습·모델편집 /
시스템·서빙)에 셋을 더한다 — 생물학·신경과학, 평가·감사, 선행계산(proactive/speculative).
마지막 셋을 나누는 이유는 §2.5–2.7에 각각 적었다.

> 공동체를 가르는 기준은 **주제**가 아니라 **인용 폐쇄성**이다. 서로를 인용하지 않고,
> 벤치마크를 공유하지 않고, 같은 단어를 다른 뜻으로 쓰면 다른 공동체다.
> 이 책의 §3.1 "경로 간 침묵 지도"가 관측한 것이 정확히 이 폐쇄성이다.

---

### C1. 에이전트 기억 (agent memory / agentic memory) — **E-경로의 본진**

가장 크고 가장 빠르게 자라며, 이 책의 통과군 대부분이 여기 있다.

| 항목 | 내용 |
|---|---|
| **대표 어휘** | `agent memory`, `agentic memory`, `memory consolidation`, `offline consolidation`, `memory bank`, `memory write policy`, `memory operator`, `store/retrieve/update/summarize/discard`, `admission / eviction`, `memory compaction`, `memory condensation`, `cross-session`, `multi-session`, `long-horizon agent`, `experience`, `procedural memory`, `episodic memory`, `reflection`, `self-evolving`, `memory lifecycle`, `working region`, `memory depth` |
| **대표 venue** | arXiv `cs.CL`/`cs.AI` (사실상 1차 발표지), ACL·EMNLP·NAACL, ICLR, NeurIPS **Datasets & Benchmarks**, COLM. 산업 채널이 실질적 venue다 — Letta 블로그, Mem0 블로그, Zep(getzep) 문서, LangChain/LangMem, Anthropic context-engineering 글 |
| **대표 저자군** | **Berkeley Sky/Letta**: Charles Packer, Sarah Wooders, Kevin Lin, Joseph E. Gonzalez, Ion Stoica (MemGPT → Letta STC) · **Mem0**: Taranjeet Singh, Deshraj Yadav · **Zep**: Daniel Chalef, Preston Rasmussen · **Google DeepMind**: ReasoningBank 팀 · **Rutgers**: Yongfeng Zhang (A-MEM) · **ZJU-NLP**: Ningyu Zhang, Huajun Chen (LightMem 계열) |
| **이 책과의 접점** | 통과군 9편 중 6편(Letta STC·Mem0·Zep·ReasoningBank·SCM·Multi-Timescale)과 Auto-Dreamer가 전부 여기. **F1·F3의 주 사냥터** |
| **이 공동체가 안 쓰는 말** | `sleep`(Auto-Dreamer의 "Dreamer"조차 각주로 world-model Dreamer와 선을 긋는다), `wake-time`, `B_s` |

---

### C2. test-time training · in-context learning · fast weights — **W-경로의 본진**

이 책의 NM 모노그래프 라인이 그대로 여기다. **이 공동체는 C1을 거의 인용하지 않는다.**

| 항목 | 내용 |
|---|---|
| **대표 어휘** | `test-time training (TTT)`, `test-time adaptation`, `in-place TTT`, `fast weights`, `writable parameters`, `optimization-based writing`, `state-transition update`, `delta rule`, `gated delta`, `linear attention`, `recurrent sequence memory`, `chunked recurrence`, `learning to memorize at test time`, `meta-learned memory`, `inner loop / outer loop`, `associative memory`, `nested / multi-timescale update`, `slow–fast parameter decomposition`, `stability–plasticity` |
| **대표 venue** | NeurIPS, ICML, ICLR, arXiv `cs.LG`. TMLR |
| **대표 저자군** | **Google (Behrouz 라인)**: Ali Behrouz, Peilin Zhong, Vahab Mirrokni — Titans → Miras → Atlas → TNT → Nested Learning → `LM Need Sleep` · **Stanford/UCSD (TTT 라인)**: Yu Sun, Xinlei Chen, Xiaolong Wang, Carlos Guestrin, Tatsunori Hashimoto · **MIT**: Songlin Yang, Yoon Kim (gated DeltaNet, linear attention) · **CMU/Princeton**: Albert Gu, Tri Dao (Mamba) · **RWKV**: Bo Peng · **Moonshot**: KDA/Kimi Linear |
| **이 책과의 접점** | 통과군 중 `Do LMs Need Sleep?` 1편. 그러나 **경계·회색지대(F2)의 최대 공급원**이다 — 이 공동체의 논문 대부분이 조건 (1)"질의 도착 전"만 아슬하게 못 맞춘다 |
| **주의** | 이 공동체의 `offline`은 **훈련 시점**을 뜻한다(§5.1). 여기서 `offline` 검색은 역효과다 |

---

### C3. 지속학습 · 모델 편집 · 지식 주입 — **Θ-경로의 본진, 그리고 F3의 본진**

| 항목 | 내용 |
|---|---|
| **대표 어휘** | `continual learning`, `lifelong learning`, `continual pre-training`, `knowledge injection`, `knowledge editing`, `sequential model editing`, `lifelong model editing`, `machine unlearning`, `catastrophic forgetting`, `interference`, `parameter drift`, `representational drift`, `model merging`, `LoRA merging`, `replay / rehearsal`, `synthetic data`, `model collapse`, `self-distillation`, `on-policy distillation`, `experience internalization`, `parametric memory`, `local parametric edits`, `engram` |
| **대표 venue** | ICLR, NeurIPS, ICML, ACL·EMNLP, **CoLLAs**(Conference on Lifelong Learning Agents — 이 축의 전용 venue), TMLR, AAAI |
| **대표 저자군** | **Northeastern**: David Bau, Kevin Meng (ROME/MEMIT) · **ZJU**: Ningyu Zhang (EasyEdit, KnowEdit) · **MIT**: Adam Zweiger, Jyothish Pari, Pulkit Agrawal (SEAL) · **Oxford/DeepMind**: Ilia Shumailov, Yarin Gal (model collapse) · **KU Leuven/Baylor**: Gido van de Ven (brain-inspired replay) · **UNC**: Peter Hase · Vincenzo Lomonaco, Davide Bacciu (CL 커뮤니티 조직) |
| **이 책과의 접점** | 통과군 SEAL·`LM Need Sleep`. **F3(제약·반증)의 압도적 주 공급원** — 반복 갱신 열화, 편집 수 상한, 자기생성 붕괴가 전부 이 공동체의 자기 주제다 |
| **F3 신호어** | `how many edits`, `collapse`, `saturation`, `ceiling`, `capacity limit`, `degradation over rounds`, `muting effect`, `utility degradation` |

---

### C4. 시스템 · 서빙 · 하드웨어 — **F4의 유일한 본진**

이 책의 **가장 큰 공백이 비용 보고**이므로, 이 공동체는 다른 여섯을 합친 것만큼 중요하다.
그리고 **이 공동체는 C1·C2·C3 어느 쪽도 거의 인용하지 않는다.**

| 항목 | 내용 |
|---|---|
| **대표 어휘** | `LLM serving`, `LLM inference`, `KV cache reuse`, `prefix caching`, `KV cache offloading`, `cache admission / eviction policy`, `TTFT`, `TPOT`, `goodput`, `SLO`, `disaggregated prefill-decode`, `multi-LoRA serving`, `adapter swapping`, `memory tiering`, `CXL`, `HBM / DRAM / NVMe tier`, `near-data processing`, `processing-near-memory`, `request trace characterization`, `production trace`, `byte budget`, `memory footprint`, `energy per token` |
| **대표 venue** | **MLSys**, OSDI, SOSP, ASPLOS, ISCA, MICRO, HPCA, EuroSys, NSDI, ATC, VLDB, SIGMOD. arXiv `cs.DC`·`cs.AR`·`cs.OS` |
| **대표 저자군** | **Berkeley Sky/vLLM**: Woosuk Kwon, Zhuohan Li, Ion Stoica · **UChicago (LMCache)**: Junchen Jiang · **UCSD**: Hao Zhang · **CMU**: Zhihao Jia · **SGLang**: Lianmin Zheng · 산업: Alibaba/Tongyi, Moonshot(Mooncake), ByteDance, Microsoft, SK hynix·Samsung(CXL) |
| **이 책과의 접점** | **통과군 0편.** ch25·ch26·ch29의 X1·X2가 이 공동체의 자리를 대신 채우고 있다. 여기서 F4가 하나라도 나오면 **ch26 판정 6("바이트로 보고한 논문 0편")이 흔들린다** |
| **F4 신호어** | `we measure`, `we characterize`, `production trace`, `wall-clock`, `bytes`, `GB per user`, `Joules`, `throughput`, `end-to-end latency breakdown` |

---

### C5. 생물학·신경과학 영감 — 은유의 출처

**따로 세는 이유**: 이 공동체만이 `sleep`·`replay`·`consolidation`·`dreaming`을 문자 그대로
쓴다. 즉 **이 책의 반용어가 유일하게 통하는 곳**이다. 동시에 가장 위험하다 — 이름은 맞는데
기제가 세 층 어디에도 안 닿는 논문이 여기 몰려 있다(SleepGate가 그 표본).

| 항목 | 내용 |
|---|---|
| **대표 어휘** | `memory consolidation`, `systems consolidation`, `complementary learning systems (CLS)`, `hippocampal replay`, `sharp-wave ripple`, `synaptic homeostasis (SHY)`, `synaptic downscaling`, `NREM / REM`, `sleep replay consolidation (SRC)`, `Hebbian plasticity`, `spiking neural network (SNN)`, `dreaming`, `wake-sleep algorithm`, `brain-inspired replay`, `neuro-inspired`, `engram` |
| **대표 venue** | eLife, PLOS Computational Biology, Nature Communications, Neural Computation, Journal of Neuroscience, **CCN**, NeurIPS neuro-AI 워크숍, ICLR |
| **대표 저자군** | Maxim Bazhenov (UCSD — sleep replay in ANN/SNN, PLOS 논문 라인) · Nicolas Deperrois, Mihai Petrovici, Walter Senn (Bern — PAD/eLife) · Gido van de Ven, Andreas Tolias · James McClelland, Randall O'Reilly, Kenneth Norman (CLS 원조) |
| **이 책과의 접점** | ch08·ch23. **F2의 공급원**이며, `wake-sleep algorithm`(Hinton 1995)은 이 책이 아직 다루지 않은 이름이다 |
| **함정** | 이 공동체 논문의 90%는 판별식 조건 (3)"어느 층의 상태를 실제로 바꾼다"에서 떨어진다. **초록에서 무엇이 바뀌는지 명시하지 않으면 버려라** |

---

### C6. 평가 · 벤치마크 · 감사(메타연구) — **F3의 두 번째 본진**

**따로 세는 이유**: 2026년에 새로 생긴 층이다. C1의 주장을 C1 밖에서 재실행하는 논문들이며,
이 책의 §4 caveat 표와 **같은 종류의 일**을 한다. 판정을 바꾸는 힘이 가장 크다.

| 항목 | 내용 |
|---|---|
| **대표 어휘** | `no-memory baseline`, `full-context baseline`, `controlled baseline`, `hidden confound`, `ablation`, `reproduction`, `re-evaluation`, `diagnostic`, `benchmark contamination`, `prequential evaluation`, `write-path cost`, `cross-scenario generality`, `LongMemEval`, `LOCOMO`, `MemoryAgentBench`, `MemoryArena`, `BEAM`, `SCBench`, `KVPress` |
| **대표 venue** | NeurIPS D&B, ACL(resources/analysis 트랙), TMLR, arXiv `cs.CL` |
| **대표 저자군** | 아직 분산. 2026년에 나온 감사 논문들(MemDelta, cross-scenario generality 진단, EvoMemBench, WritePolicyBench)이 각각 다른 그룹 |
| **이 책과의 접점** | ch27 판정 2(FE2 — full-context 팔), ch15 caveat 다수. **여기서 나온 결과는 이 책의 판정을 직접 확증하거나 뒤집는다** |

---

### C7. 선행계산 — proactive · speculative · anticipatory

**따로 세는 이유**: 이 공동체만이 **조건 (1)"질의 도착 전"을 명시적 설계 목표로 삼는다.**
C1은 조건 (3)"상태를 바꾼다"에 강하고 조건 (1)에 약한데, C7은 정확히 반대다.
**두 공동체는 서로를 인용하지 않는다.** 그래서 F1의 미개척 후보가 여기 있을 가능성이 가장 크다.

| 항목 | 내용 |
|---|---|
| **대표 어휘** | `idle-time compute`, `proactive agent`, `anticipatory execution`, `next-query prediction`, `speculative execution`, `speculative decoding`, `shadow run`, `precompute`, `prefetch`, `background thinking`, `proactive thinking`, `pre-launch`, `query anticipation`, `intent memory` |
| **대표 venue** | arXiv `cs.CL`·`cs.HC`, CHI·UIST(일부), **산업 특허·TDCommons**(논문보다 여기 먼저 나온다) |
| **대표 저자군** | 아직 형성 중. Letta 라인 + dialogue/proactive-assistant 라인 + serving 라인이 각각 독립적으로 도달 |
| **이 책과의 접점** | ch14의 "Precompute 열은 여전히 한 편"(OUTLOOK-REVIEW §5.1) — **이 공동체를 검색에 넣으면 그 진술이 바뀔 수 있다.** F1 최우선 |

---

### 2.8 공동체 간 인용 폐쇄성 — 한 표

이 책 §3.1의 침묵 지도를 공동체 수준으로 올린 것이다.

| | C1 agent | C2 TTT | C3 CL·edit | C4 systems | C5 bio | C6 eval | C7 proactive |
|---|---|---|---|---|---|---|---|
| **C1** | — | 거의 없음 | 드묾 | 없음 | 수사적 | 받는 쪽 | 없음 |
| **C2** | 거의 없음 | — | EWC만 | 없음 | 절단 인용 | 없음 | 없음 |
| **C3** | 드묾 | 일부 | — | 없음 | 원조 인용 | 일부 | 없음 |
| **C4** | 없음 | 없음 | 없음 | — | 없음 | 없음 | 일부 |
| **C5** | — | — | 받는 쪽 | 없음 | — | 없음 | 없음 |
| **C6** | 전면 | 없음 | 일부 | **일부** | 없음 | — | 없음 |
| **C7** | 없음 | 없음 | 없음 | 일부 | 없음 | 없음 | — |

**읽을 것 둘.** (a) C4(systems) 행이 전부 "없음"이다 — F4가 비어 있는 이유가 여기 있다.
(b) C6(eval)만이 C1·C3·C4를 동시에 본다. **C6이 가장 높은 수확률을 낸다.**

---

## 3. 검색어 목록 — 실제로 넣을 문자열

**규칙 셋.**
1. 문자열을 그대로 쓴다. 다듬으면 다른 걸 찾는다.
2. 시기 한정은 `2026`을 문자열에 넣거나 도구의 날짜 필터로 건다. `2026-06 이후`가 최우선이다.
3. **[검증됨]** 표시는 이 문서를 쓰며 실제로 실행해 미읽은 2026년 논문을 반환한 문자열이다.

### 3.1 우선순위 A — 최고 수확 (F1·F4를 직접 겨냥, 전부 검증됨)

| # | 문자열 | 겨냥 | 근거 |
|---|---|---|---|
| A1 | `offline memory consolidation language agents` | F1 | [검증됨] 첫 페이지에 미읽은 2026년 논문 6편 |
| A2 | `idle-time compute proactive agents LLM` | F1 | [검증됨] C7의 정문. 조건 (1)을 명시하는 유일한 어휘 |
| A3 | `memory write policy byte budget benchmark` | **F4** | [검증됨] 바이트 예산을 명시하는 유일한 문자열 |
| A4 | `agent memory evaluation controlled baseline confound` | F3 | [검증됨] C6의 정문. ch27 FE2를 직접 시험 |
| A5 | `per-user parametric memory local edits serving footprint` | F1·F4 | [검증됨] Θ-경로 × 시스템 교차점 |
| A6 | `test-time training persistent state across sessions` | F1·F2 | [검증됨] C2에서 조건 (1)을 맞추는 것만 거른다 |
| A7 | `sequential model editing collapse how many edits` | **F3** | [검증됨] Θ-경로 반증 축 직격 |
| A8 | `context compaction long-horizon agent serving latency` | F2·F4 | [검증됨] KV/컨텍스트 회색지대 + 실측 |
| A9 | `continual experience internalization self-evolving agent collapse` | F1·F3 | [검증됨] 자기생성 데이터 붕괴 |
| A10 | `KV cache offloading CXL memory tier LLM inference measurement` | **F4** | [검증됨] C4 정문. 바이트·지연 실측 |

### 3.2 C1 — 에이전트 기억 (E-경로)

| 순위 | 문자열 |
|---|---|
| 1 | `offline consolidation agent memory bank cross-session` |
| 2 | `memory operator selection retain consolidate budget` |
| 3 | `learned memory write policy reinforcement learning agent` |
| 4 | `memory consolidation multi-session conversational agent 2026` |
| 5 | `agentic memory admission eviction rewrite operator` |
| 6 | `memory compaction rate distortion agent what to keep` |
| 7 | `procedural memory reusable experience agent distillation` |
| 8 | `self-evolving agent memory lifecycle store retrieve update forget` |
| 9 | `hierarchical memory consolidation temporal agent long-horizon` |
| 10 | `memory depth parametric consolidation long-running agent` |

### 3.3 C2 — test-time training · fast weights (W-경로)

| 순위 | 문자열 |
|---|---|
| 1 | `test-time training persistent weights across requests` |
| 2 | `fast weights carried across context boundary inference` |
| 3 | `agentic test-time training multi-turn episode policy drift` |
| 4 | `self-guided test-time training long-context adapted weights` |
| 5 | `in-place test-time training writable parameters LLM` |
| 6 | `recurrent state continuity long-running LLM inference` |
| 7 | `state-transition memory update delta rule gated linear attention 2026` |
| 8 | `test-time adaptation reuse across queries amortized` |
| 9 | `multi-timescale nested update schedule language model` |
| 10 | `query-conditioned test-time self-training` |

### 3.4 C3 — 지속학습 · 모델 편집 (Θ-경로 · F3)

| 순위 | 문자열 |
|---|---|
| 1 | `continual knowledge injection LLM weights forgetting 2026` |
| 2 | `lifelong model editing capacity limit degradation` |
| 3 | `sequential knowledge editing collapse spectral mitigation` |
| 4 | `self-distillation loop degradation rounds language model` |
| 5 | `synthetic data self-training collapse accumulate replace` |
| 6 | `on-policy distillation agent experience into weights` |
| 7 | `continual pre-training knowledge retention rate measurement` |
| 8 | `LoRA merging interference many adapters capacity` |
| 9 | `parametric memory capacity bits per parameter facts` |
| 10 | `stability plasticity continual fine-tuning mechanistic analysis` |

### 3.5 C4 — 시스템 · 서빙 · 하드웨어 (**F4 — 이 책의 최대 공백**)

| 순위 | 문자열 |
|---|---|
| 1 | `KV cache reuse production trace characterization cloud provider` |
| 2 | `multi-LoRA serving adapter swapping memory footprint throughput` |
| 3 | `prefix caching hit rate amortization long context serving` |
| 4 | `background batch inference job scheduling GPU idle capacity` |
| 5 | `CXL tiered memory LLM inference KV cache SSD` |
| 6 | `agent memory system write path cost latency breakdown` |
| 7 | `disaggregated prefill decode cost model TTFT measurement` |
| 8 | `energy per token LLM serving measurement joules` |
| 9 | `preemptible low-priority inference workload colocation LLM` |
| 10 | `state size bytes per user LLM personalization storage` |

### 3.6 C5 — 생물학·신경과학

| 순위 | 문자열 |
|---|---|
| 1 | `sleep-inspired replay catastrophic forgetting sequential tasks 2026` |
| 2 | `sleep replay consolidation artificial neural network calibration` |
| 3 | `complementary learning systems operational design principle LLM` |
| 4 | `synaptic downscaling homeostasis deep network offline phase` |
| 5 | `wake-sleep algorithm revisited language model` |
| 6 | `hippocampal replay generative model continual learning 2026` |
| 7 | `NREM REM two-stage consolidation neural network` |

### 3.7 C6 — 평가 · 감사 (**수확률 최고**)

| 순위 | 문자열 |
|---|---|
| 1 | `agent memory benchmark hidden confound controlled baseline` |
| 2 | `re-evaluation memory systems no-memory baseline reproduction` |
| 3 | `cross-scenario generality agentic memory diagnostics strong baseline` |
| 4 | `long-term memory benchmark full context baseline beats memory` |
| 5 | `memory benchmark contamination leakage evaluation protocol LLM` |
| 6 | `write path cost fraction of agent execution time measured` |
| 7 | `prequential streaming evaluation agent memory AUC` |

### 3.8 C7 — 선행계산 (**F1 미개척 가능성 최고**)

| 순위 | 문자열 |
|---|---|
| 1 | `idle-time compute anticipatory agent precompute future queries` |
| 2 | `next-query prediction intent memory multi-turn` |
| 3 | `speculative precomputation before user query LLM assistant` |
| 4 | `proactive thinking precompute response during idle dialogue` |
| 5 | `shadow run anticipatory execution chatbot speculative` |
| 6 | `pre-launch next step while environment idle agent acceleration` |
| 7 | `background thinking between turns language model` |

### 3.9 교차 검색 — 공동체 경계를 넘는 것만 (F2 전용)

경계·회색지대(F2)는 **한 공동체 안에서는 안 보인다.** 아래는 의도적으로 두 공동체의
어휘를 섞은 문자열이다.

| # | 문자열 | 어느 경계 |
|---|---|---|
| X1 | `KV cache state persisted across sessions not context` | C2×C4 — 세 층 어디에도 안 맞는 상태 |
| X2 | `converting context into weights before query arrives` | C2×C7 — W/Θ × 조건 (1) |
| X3 | `offline pass over memory store gradient update agent` | C1×C3 — E-경로가 Θ를 만지는 지점 |
| X4 | `memory system cost accounting bytes FLOPs unified` | C1×C4 — 이 책의 회계를 남이 했는가 |
| X5 | `scheduler decides when to consolidate memory budget` | C1×C4 — ch29의 메타컨트롤러 공백 |
| X6 | `sleep phase for transformer weights not analogy` | C5×C3 — 이름만 sleep을 걸러내기 |

---

## 4. 반(反)용어 — 이 책은 쓰지만 분야는 안 쓴다

**이 문자열로 검색하면 놓친다.** 각 행에 대체 문자열을 병기했다.

| 반용어 | 왜 안 되는가 | 대신 쓸 것 |
|---|---|---|
| `sleep-time compute` | 후속 논문 0건. 2504.13171과 그 블로그만 나온다 | `offline consolidation`, `idle-time compute` |
| `wake-time` | Letta 논문 밖에서 거의 안 쓴다 | `test-time`, `inference-time`, `online`, `serving time` |
| `세 경로` / `three paths` / `E-경로·W-경로·Θ-경로` | 이 책 고유 좌표 | `external memory` / `fast weights` / `parametric memory` |
| `판별식` / `discriminant` | 이 책 고유 | (검색 불가 — 개념으로만 쓴다) |
| `$B_s$` / `$L_w$` / `$C$` / `$\rho$` | 이 책의 기호 | `compute budget`, `latency`, `memory footprint`, `retention rate` |
| `상각식 (A)` / `amortization equation` | 이 책 고유. 서베이 전문에 `amortiz` 0회 | `cache reuse ratio`, `cost per query`, `queries per context` |
| `slow weights` | fast weights는 살아 있으나 slow weights는 거의 안 쓴다 | `base model weights`, `backbone parameters`, `frozen weights` |
| `sleep 라운드` / `sleep round` | 이 책 고유 | `consolidation cycle`, `consolidation cadence`, `offline pass` |
| `dreaming` (자기생성 학습데이터 뜻으로) | world-model `Dreamer`와 충돌. Auto-Dreamer조차 각주로 선을 긋는다 | `synthetic trajectory`, `self-edit`, `self-generated curriculum` |
| `offline recurrence` | 2605.26099 한 편만 쓴다 | `recurrent pass over context`, `state carry-over` |
| `경계 사례` / `회색지대` | 이 책 고유 | `borderline`, `ambiguous case` — 검색어로는 무력 |
| `세 층 Θ/W/E` | 이 책 고유 | `parametric / fast-weight / external memory` |
| `Knowledge Seeding` | `LM Need Sleep` 한 편의 고유명 | `knowledge consolidation objective` |
| `memory-device 기회` | 이 책 고유(NM 승계) | `memory-centric`, `near-data processing`, `hardware-algorithm co-design` |

**한 줄 규칙**: **이 책의 목차에 있는 말은 검색창에 넣지 마라.** 목차는 이 책이 만든 좌표계다.

---

## 5. 동의어 · false friend 표

### 5.1 false friend — **다른 것을 같게 부르는 쌍** (가장 중요)

| 단어 | 이 책의 뜻 | 남의 뜻 | 충돌 정도 |
|---|---|---|---|
| **`offline`** | 질의 도착 **전** 유휴 시간 계산 (= sleep-time) | **서베이 2607.25380 / C2**: **훈련 시점에만 갱신되고 배포 후 고정**. Table I에서 `Offline` = kNN-LM·Switch Transformer·Mixtral·Engram, 즉 **배포 후 아무것도 안 바뀌는 것** | **정면 충돌.** 이 책의 `offline`은 서베이 축에서 `Online`이다 |
| | | **C1(agent)**: `offline consolidation` = 세션 밖 배치 처리 (= 이 책과 **일치**) | 같은 단어가 두 공동체에서 정반대 |
| **`online`** | wake-time, 질의 처리 중 | **서베이/C2**: 추론 중 갱신 = 이 책의 W-경로 **전부** | 이 책의 통과군 `Do LMs Need Sleep?`은 서베이 축에서 `Online`이다 |
| **`consolidation`** | Θ·E 층에 오프라인으로 통합 | **서베이 §V-B-c `working-memory consolidation`**: KV 캐시 토큰 병합. **어느 층도 안 바뀐다** | 판별식 조건 (3)에서 갈린다. 이름만 보고 통과시키면 안 됨 |
| | | **C5(신경과학)**: 시스템 통합(해마→피질). 은유 | |
| **`memory`** | 세 층 Θ/W/E의 총칭 | **서베이/C2**: 아키텍처 substrate(KV·recurrent state·writable params) · **C1**: 텍스트 레코드 저장소 · **C4**: HBM/DRAM 바이트 | 세 뜻이 서로 배타적. `memory` 단독 검색은 무의미 |
| **`test-time compute`** | wake 예산 $B_t$ | **reasoning 공동체**: CoT 길이·샘플 수·verifier 호출 | Letta STC의 비교 대상이 후자다. 후자로 검색하면 STC 계보가 안 나온다 |
| **`state`** | 세 층 전부 | **C4**: KV 캐시 바이트 · **C2**: recurrent hidden state | |
| **`sleep`** | 질의 전 유휴 계산 | **C5/SleepGate**: 이름만 sleep, 기제는 wake-time 게이팅 (판별식 0/4) | **이름이 기제보다 넓다**는 이 책의 척추 논지가 정확히 이 칸이다 |
| **`compaction`** | rate–distortion 압축 결정 | **C1/산업**: 대화 이력 요약(Claude Code/agent harness의 auto-compact) · **C4**: 저장소 조각모음 | |
| **`replay`** | $\mathcal{R}_k$ 구성 | **C4**: 요청 트레이스 재생 · **RL**: replay buffer | |
| **`engram`** | (미사용) | **2026 신조어**: per-user parametric memory row (Engram, `User as Engram`) | 이 책에 대응어가 없다. **새로 들어온 말** |
| **`dreaming`** | 자기생성 학습 데이터 | **Dreamer 계열**: world-model latent imagination | Auto-Dreamer가 각주로 방어할 만큼 흔한 오독 |

### 5.2 동의어 — **같은 것을 다르게 부르는 쌍**

| 이 책의 말 | 분야의 말 (검색에 쓸 것) |
|---|---|
| sleep-time compute | `offline consolidation` · `idle-time compute` · `background compute` · `pre-query computation` · `anticipatory/speculative precomputation` · `proactive thinking` |
| $E$-경로 `wr` | `memory write policy` · `memory operator` · `admission / eviction / consolidation` · `memory management` · `store/update/summarize/discard` |
| $\Theta$-경로 | `parametric memory` · `weight-space memory` · `internalization` · `knowledge injection` · `local parametric edits` · `engram rows` · `writable parameters` |
| $W$-경로 | `fast weights` · `test-time training` · `optimization-based writing` · `persistent recurrent state` · `state continuity` |
| 상각식 (A) | `amortization` · `cache reuse ratio` · `cost per query` · `queries per cached context` |
| 자기생성 $\mathcal{R}$ | `synthetic curriculum` · `self-edit` · `experience internalization` · `on-policy self-distillation` · `self-generated trajectory` |
| 열화율 $\rho$ | `retention rate` · `forgetting curve` · `degradation over rounds` · `execution instability` · `performance collapse` |
| 상태 용량 $C$ | `memory footprint` · `bytes-per-token-of-history (BPT)` · `byte budget` · `active memory size` · `#Tok` · `state size` |
| sleep 예산 $B_s$ | `consolidation cost` · `write-path cost` · `offline compute budget` · `preprocessing cost` |
| 질의 예측가능성 | `query predictability` · `next-query prediction` · `intent memory` · `user intent anticipation` |
| 경로 간 침묵 | `citation gap` · `disconnected literatures` · `siloed communities` |
| 메타컨트롤러 공백 | `adaptive memory orchestration` · `learned controller` · `memory scheduling policy` |

---

## 6. 뒤 단계에 넘기는 지침

1. **A1–A10을 먼저 돌린다.** 전부 검증된 문자열이고 F1·F3·F4를 직접 겨냥한다.
2. **C6(평가·감사)과 C7(선행계산)에 예산을 더 준다.** 전자는 판정을 직접 뒤집고,
   후자는 이 책이 "Precompute 열은 한 편"이라 쓴 진술을 뒤집을 수 있는 유일한 곳이다.
3. **C4(시스템)는 초록에 숫자와 단위가 있는 것만 올린다.** F4의 가치는 실측에 있다.
   `we propose`만 있고 `we measure`가 없으면 버린다.
4. **`offline`이 초록에 있으면 무슨 뜻인지 먼저 확인한다**(§5.1). 훈련 시점을 뜻하면
   판별식 조건 (1)을 못 맞춘다.
5. **이름에 `sleep`/`consolidation`/`memory`가 있다고 올리지 마라.** 판별식은 자기 규정을
   묻지 않는다(ch11 §11.2.1). **무엇이 바뀌는지**를 초록에서 확인한다.
6. 반대로 **이름에 그 말이 없어도 기제가 맞으면 올린다.** 2026년 통과 후보 상당수가
   `sleep`을 한 번도 쓰지 않는다.

---

## 7. 이 문서의 한계

- 검증 검색 10건은 **영어 웹 검색 한 채널**만 썼다. arXiv 리스팅·Semantic Scholar·
  OpenReview·회의 프로시딩 직접 조회는 하지 않았다. 뒤 단계가 채널을 늘려야 한다.
- 저자군은 corpus 32편과 검증 검색에서 확인된 이름만 적었다. **누락이 있다**.
- C6·C7은 2026년에 형성 중이라 대표 저자군을 특정하지 못했다. 이는 관측이지 결함이 아니다 —
  형성 중인 공동체가 미개척 후보를 가장 많이 갖는다.
- 공동체 경계는 인용 폐쇄성으로 그었다. 한 논문이 두 공동체에 걸치는 경우가 있고,
  그 논문들이 대체로 가장 값지다(§3.9의 교차 검색이 그것을 노린다).
