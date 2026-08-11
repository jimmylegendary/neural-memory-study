# gap-sweep — 인용 그래프 채널

**작성**: 2026-08-11 · **채널**: citation-graph (Semantic Scholar citations API) + 보조 WebSearch
**대상**: STC v2 corpus 동결(2026-08) 이후 놓친 논문
**배제 기준**: `scratchpad/corpus-exclusion.md` — deep-read 32편 + vendored arXiv ID 69개
**판정 기준**: `S0-DESIGN.md` §2 (세 층 Θ/W/E, 판별식 네 조건) · `PRE-RESEARCH.md` §2·§5

---

## 1. 방문한 소스

### 1.1 인용 그래프 (주 채널)

앵커 6편에 대해 `api.semanticscholar.org/graph/v1/paper/arXiv:<id>/citations`를 전수 조회했다
(fields = title, year, publicationDate, externalIds, abstract, authors, venue; limit 100, 페이지네이션).

| 앵커 | arXiv | S2 citationCount | 수집된 citing paper |
|---|---|---|---|
| Letta `Sleep-time Compute` | 2504.13171 | 35 | 35 |
| `Do LMs Need Sleep?` | 2605.26099 | **0** | 0 |
| `LM Need Sleep` | 2606.03979 | 3 | 3 |
| Nested Learning | 2512.24695 | (meta 조회 실패, citations는 성공) | 91 |
| SEAL | 2506.10943 | 53 | 53 |
| Auto-Dreamer | 2605.20616 | 2 | 2 |

- **중복 제거 후 총 167편**, 배제목록 히트 10편 → 심사 대상 157편.
- 그중 2026년 111편, **2026-06 이후 25편**.
- 초록은 S2가 반환한 원문 초록을 직접 읽었고, 후보로 올린 것은 전부 arXiv API
  (`export.arxiv.org/api/query?id_list=`)로 ID·날짜·저자·초록을 재확인했다.

### 1.2 보조 채널 (WebSearch → arXiv API 검증)

인용 그래프가 비어 있는 구간(특히 F4 비용·시스템, F3 반증)을 메우기 위해 아래 질의를 돌렸다.

- `arXiv 2026 papers citing "Sleep-time Compute" Letta offline memory consolidation agent`
- `arXiv August 2026 idle-time compute proactive agent precompute before user query`
- `arXiv 2026 measured serving cost per-user LoRA adapter memory footprint agent memory write path latency`
- `arXiv 2026 agent memory benchmark no-memory baseline reproduction confound re-evaluation`
- `arXiv 2026 sequential model editing collapse thousands of edits degradation lifelong knowledge injection`
- `arXiv 2026 self-evolving agent experience internalization collapse degradation repeated self-training rounds`
- `arXiv 2026 context compaction long-horizon agent serving measured latency token cost KV cache offload memory tier`
- `arXiv 2026 learned memory write policy reinforcement learning what to store agent budget bytes`
- `arXiv 2026 CXL memory tier KV cache offloading LLM inference measurement hardware`

### 1.3 이 채널 자체에서 나온 관찰 (그 자체가 발견)

- **`Do LMs Need Sleep?`(2605.26099)의 피인용은 0이다.** 책이 W-경로의 유일한 정면 사례로
  세운 논문을 아무도 인용하지 않았다. §3의 "침묵 지도"는 **인용받는 쪽에서도** 성립한다.
- **Auto-Dreamer(2605.20616)는 피인용 2편**, `LM Need Sleep`(2606.03979)은 3편이다.
  Θ-경로 전체가 아직 인용 그래프상 고립돼 있다.
- 반면 **Nested Learning(91편)·SEAL(53편)은 활발**한데, 그 인용자 대부분은 sleep-time compute
  문헌이 아니라 TTT·continual-learning·아키텍처 라인이다. **책이 §2.2에서 NL을 판별식 불충족으로
  분류한 판단이 인용 그래프에서도 뒷받침된다** — NL을 인용하는 91편 중 오프라인 갱신을 하는 것은
  극소수다.
- **Letta STC(35편)의 인용자 중 E-경로 후속 시스템 논문은 거의 없다.** 대신 idle-time·proactive
  계열(ProAct, IdleSpec)과 평가·서베이가 인용한다. `PRE-RESEARCH` §3의 "ch15가 Letta STC를
  인용하지 않는다"는 관찰이 2026-08까지 유지된다.

---

## 2. 후보 (25편)

`ch` = 이 후보가 판정을 바꿀 수 있는 장. `채널` = CG(citation-graph, 괄호 안은 앵커) / WS(WebSearch).

### 2.1 F3·F4 — 제약과 비용 (이 책의 최대 공백)

| # | arXiv | 제목 / 저자 / 날짜 | F | 왜 판정을 바꾸는가 | ch | 채널 |
|---|---|---|---|---|---|---|
| 1 | **2606.29914** | MemDelta: Controlled Baselines and Hidden Confounds in Agent Memory Evaluation — Kuan Wang — 2026-06-29 | **F3+F4** | write-path 비용을 1급 지표로 요구하며 Mem0가 cloud-RAG와 동률인데 **비용 50배**임을 통제 비교로 보인다. §4 caveat 표의 "품질 1위가 기억 없음" 항목이 **비용 축에서 재진술**되고, ch25의 "어느 논문도 비용을 보고하지 않는다"가 더 이상 참이 아니게 된다 | ch15·ch25·ch27 | WS |
| 2 | **2606.19172** | User as Engram: Internalizing Per-User Memory as Local Parametric Edits — Bojie Li — 2026-06-17 | **F1+F4** | per-user LoRA 대비 **약 33,000× 작은 메모리 풋프린트**를 보고하고, 사용자별 편집이 disjoint hash slot에서 **가산적·무손실로 합성**된다고 주장한다. ch26의 "상태 용량을 바이트로 보고한 논문 0편"이 깨지고 A2(쓰기 용량이 새 병목)의 핵심 반례이자 증거가 된다 | ch26·ch29·ch19 | WS |
| 3 | **2607.18141** | A CXL Memory Rack for Multi-Turn LLM Serving (HyMCache) — Jang, Song, Kim, Noh, Kim — 2026-07-20 (v3 08-05) | **F4** | 실제 CXL-HM 프로토타입에서 multi-turn KV 재사용을 측정 — LMCache 대비 3.0×, Mooncake 대비 **DRAM 16× 절감에 성능 -30%**. ch29의 memory-device 기회 논증이 **추정이 아니라 실측 하드웨어**로 뒷받침되거나 반박된다 | ch29 | WS |
| 4 | **2606.25115** | Forget to Improve: On-Device LLM-Agent Continual Learning via Budget-Curated Memory — Wu, Ding, Huang, Zhao — 2026-06-23 | **F1+F3+F4** | 기억 항목을 **바이트당 net value(value−harm)**로 채점해 KEEP/SHARE/TRUST를 결정하고 Jetson 실기기에서 메모리 2.7×·업링크 2.4× 절감을 측정한다. (U-E)의 `wr`을 **예산 함수로 정의한 최초의 실측 사례** — 식 (A)의 분자를 바이트로 바꾼다 | ch26·ch29·ch15 | WS |
| 5 | **2608.00303** | CrystalMem: Elastic Memory for Self-Evolving LLM Agents via Knowledge Crystallization — Beining Wu, Jun Huang — 2026-07-31 | **F3+F4** | 바이트 예산이 줄었다 회복돼도 능력이 안 돌아오는 **memory hysteresis**를 보이고, keep/drop만 하는 정책에 **residual-deficit floor가 존재함을 증명**한다. E-경로의 `wr`이 파괴적 덮어쓰기라는 ch14 지적에 **정량적 하한**을 준다 | ch14·ch26·ch27 | **CG(letta-stc)** |
| 6 | **2607.21962** | Ground Truth First: A Longitudinal Evaluation Instrument for Agent Memory, and the Tenure Crossover — Quentin Spencer — 2026-07-24 | **F3** | 기억 아키텍처 **순위가 이력 길이에 따라 역전**하고(3주 선두가 9주에 96%→72%), full-history baseline이 단기에는 최고 시스템과 동률이되 **읽기 비용 2배**임을 보인다. "품질 1위가 기억 없음"이 **시간 지평의 함수**였다는 재해석 | ch15·ch27·ch30 | WS |
| 7 | **2606.04703** | Rethinking Continual Experience Internalization for Self-Evolving LLM Agents — Chen, Yang, Fan, Nie, Sun — 2026-06-03 | **F3** | 다회 반복 경험 내재화에서 **누적 개선이 아니라 진행성 능력 붕괴**가 일어남을 체계적으로 보이고(web-reasoning 23.2%→8.5%, 3회), granularity/injection/regime 세 축으로 원인을 분해한다. **X4(반복 consolidation 열화 ρ)의 선행 실측** | ch30·ch22·ch20 | WS |
| 8 | **2601.15313** | Attention Is Not Retention: The Orthogonality Constraint in Infinite-Context Architectures — Zahn, Beton, Chana — 2026-01-14 (v2 02-04) | **F3** | 의미 밀도 ρ가 높으면 **N=5개 사실에서**, 중간이면 N≈20–75에서 공유 연속 파라미터에 쓴 사실이 붕괴한다고 대규모(16,309 fact) 측정한다. 기하학적 상한이므로 Θ-경로 전체에 걸리는 **하드 반증** — ch09의 3.6 bits/param 상한과 다른 축 | ch09·ch22·ch19 | **CG(nested-learning)** |
| 9 | **2607.00368** | Beyond Perplexity: A Behavioral Evaluation Framework for Deployment-Memory Claims in LLM Test-Time Training — Song, Chen, Kong, Xie, Dong — 2026-07-01 | **F3** | TTT의 perplexity·loss 지표가 **배포 기억 주장을 지지하지 못함**을 evidence ladder로 형식화하고, 통제 진단에서 **one-step LoRA가 loss는 낮추는데 free-form recall은 실패**함을 세 규모에서 보인다. W·Θ-경로 증거 기반 전체의 등급을 내린다 | ch17·ch22·ch30 | **CG(seal)** |
| 10 | **2606.32002** | Self-Study Reconsidered: The Hidden Fragility of Learning from Self-Generated QA — Alimaskina, Shveykin, Molodtsov, Shalygin, Kadeishvili — 2026-06-30 | **F3** | 자기생성 QA가 중립 전처리가 아니라 **암묵적 정책**이며 coverage가 조기 포화하고, 텍스트 내 지시문 준수로 injection compliance 88%가 나온다(필터링 후 13%). SEAL·`LM Need Sleep`의 $\mathcal{R}_k$ 생성 단계에 **model collapse와 다른 종류의 붕괴**를 건다 | ch06·ch20·ch21·ch22 | **CG(seal)** |

### 2.2 F1 — 판별식 통과 후보 (통과군 9편이 늘어난다)

| # | arXiv | 제목 / 저자 / 날짜 | F | 왜 판정을 바꾸는가 | 층 | 채널 |
|---|---|---|---|---|---|---|
| 11 | **2605.07076** | Self-Consolidating Language Models (SCoL): Continual Knowledge Incorporation from Context — Wang, Gupta, Dong, MacLellan — 2026-05-08 (v2 05-12) | **F1** | 문맥을 가중치에 쓰되 **어느 층을 갱신할지 텍스트 명령으로 스스로 생성**하고, 갱신이 다음 선택을 바꾸므로 meta-RL로 학습한다. SEAL의 self-edit을 **층 선택으로 확장한 Θ-경로 정면 사례** — ch20의 bridge-out을 실제로 받는 첫 논문 | Θ | **CG(seal)** |
| 12 | **2607.16256** | Discovery by Dreaming: Cross-Domain Recombination in Artificial Memory — Zahn, Evans, Eagleman — 2026-06-28 (v2 07-21) | **F1** | LoRA 통합(DREAMS)과 기호 엔진에서 **교차도메인 replay만 이득을 내고 동일도메인 rehearsal은 못 낸다**를 이중 검증하고, 같은 재료를 671B에 in-context로 넣으면 **이득이 역전**함을 통제로 보인다. Dreaming이 프롬프트 효과가 아니라 가중치 성질임을 처음 분리 | Θ | **CG(lm-need-sleep+seal)** |
| 13 | **2605.25971** | Anticipate and Learn: Unleashing Idle-Time Compute in Proactive Agents (ProAct) — Hu, Lyu, Kong, Liu, Lin — 2026-05-25 | **F1** | 유휴 시간에 **다음 필요를 예측하고 근거를 미리 수집해 persistent memory에 넣는다** — 판별식 (1)(2)(3)(4)를 정면으로 만족하는 E-경로 사례이면서, corpus 어디에도 없는 **"무엇을 미리 계산할지"의 예측 문제**를 도입한다. ch14의 $N_q$ 공백과 직결 | E | **CG(letta-stc)** |
| 14 | **2606.05922** | Evolving Agents in the Dark: Retrospective Harness Optimization via Self-Preference (RHO) — Pan, Liu, Lin, Zeng, Tang — 2026-06-04 (v2 06-10) | **F1** | 라벨 없이 과거 궤적만으로 harness(스킬·툴·워크플로)를 오프라인 재작성해 **SWE-Bench Pro 59%→78%**. ReasoningBank의 "합집합만 하는 `wr`"에 대한 대안이며, E-경로 산출물이 텍스트 레코드가 아니라 **실행 가능한 harness**라는 새 형태 | E | **CG(letta-stc)** |
| 15 | **2601.01885** | Agentic Memory (AgeMem): Learning Unified Long-Term and Short-Term Memory Management — Yu, Yao, Xie, Tan, Feng — 2026-01-05 (v3 07-23) | **F1** | store/retrieve/update/summarize/discard를 **툴 액션으로 노출하고 3단계 RL(step-wise GRPO)로 쓰기 정책 자체를 학습**한다. corpus의 `wr`은 전부 손으로 정한 규칙이었다 — **학습된 `wr`의 첫 사례** | E | WS |
| 16 | **2602.06025** | Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory (BudgetMem) — Zhang, Yue, Feng, Long, Bao — 2026-02-05 (v3 05-27) | **F1+F4** | 기억 모듈마다 Low/Mid/High 예산 티어를 두고 **RL 라우터가 정확도–기억구축비용 프론티어를 명시적으로 제어**한다. $B_s$를 축으로 놓은 **실제 프론티어 데이터** — X3(B_s scaling)의 유일한 외부 재료가 될 수 있다 | E | WS |
| 17 | **2608.02515** | LiveMem: Maintaining Memory State Continuity in Long-Running LLM Inference — Liu, Sun, Yang, Wu, Chen — 2026-08-03 (v2 08-07) | **F1/F2** | KV 창이 넘어갈 때 **고정 용량 상태로 계산을 이어가는 "state continuity under context turnover"**를 정식화하고 state-aware serving까지 붙인다. `Do LMs Need Sleep?`가 W-경로의 유일 사례라는 ch17의 서술을 **두 번째 사례로 깬다** | W | **CG(nested-learning)** |

### 2.3 F2 — 경계·회색지대 (판별식 자체를 시험한다)

| # | arXiv | 제목 / 저자 / 날짜 | F | 왜 판정을 바꾸는가 | 채널 |
|---|---|---|---|---|---|
| 18 | **2603.13875** | GradMem: Learning to Write Context into Memory with Test-Time Gradient Descent — Kuratov, Kairov, Bulatov, Rodkin, Burtsev — 2026-03-14 (v2 05-29) | **F2** | 문맥을 **prefix memory token에 gradient로 쓰고 원문 없이 여러 질의에 답한다**(context removal). 갱신 대상이 Θ도 W도 E도 아닌 **네 번째 자리** — 세 층 프레임의 완결성을 직접 시험한다. 게다가 **쓰기 step 수를 늘리면 용량이 늘어난다**는 $B_s$–$C$ 트레이드오프 측정 | **CG(nested-learning)** |
| 19 | **2605.22154** | IdleSpec: Exploiting Idle Time via Speculative Planning for LLM Agents — Choi, Park, Song, Dingliwal, Jayanthi — 2026-05-21 | **F2** | "idle-time compute"라는 **같은 이름을 쓰지만 산출물이 그 턴 안에서 소멸**한다 — 조건 (4)를 만족하지 않는다. ch11 판별식이 무엇을 걸러내는지 MemGPT(조건 1 위반)와 **대칭인 반대편 사례**로 보여줄 수 있다 | **CG(letta-stc)** |
| 20 | **2608.00902** | Practical Online KV Cache Compaction for LLM Agents: An Empirical Study — Liu, Ji, An, Jain, Polatkan — 2026-08-02 | **F2+F4** | **미래 질의를 모르는 상태에서 압축해야 한다**는 온라인 제약을 명시하고, 즉시 압축은 해가 되며 지연 압축이 회복시킨다는 것을 측정(TE가 KV 80% 절감에 정확도 대부분 유지). 세 층 어디도 아닌 KV 상태의 (U-E)식 대응물 | WS |
| 21 | **2604.27707** | Contextual Agentic Memory is a Memo, Not True Memory — Xu, Dai, Zhang — 2026-04-30 (v2 08-05) | **F3** | 검색 기반 기억은 lookup이지 memory가 아니며 **조합적으로 새로운 과제에 대해 문맥 크기·검색 품질로 넘을 수 없는 일반화 천장이 있다**고 형식화한다. ch27의 E-경로 판정을 "비용 문제"에서 **"원리적 상한 문제"로 격상**시킬 수 있다 | **CG(nested-learning)** |
| 22 | **2605.11836** | More Edits, More Stable: Understanding the Lifelong Normalization in Sequential Model Editing (StableEdit) — Ma, Chen, Liu, Xu, Zheng — 2026-05-12 (v2 07-21) | **F3** | 장기 편집에서 살아남는 편집기들의 공통 성분이 **Lifelong Normalization**임을 찾아내고, 이를 빼면 즉시 붕괴함을 보인 뒤 **점근적 직교성·유계 노름을 증명**한다. ch19가 ROME/MEMIT의 상한만 서술하고 끝난 자리에 **왜 붕괴하고 무엇이 막는지**를 넣는다 | WS |
| 23 | **2605.23296** | Parallel Context Compaction for Long-Horizon LLM Agent Serving — Cim, Topcu, Das, Kandemir — 2026-05-22 | **F4** | 요약 기반 compaction 호출이 **에이전트 추론을 수십 초 블로킹**하고 유지 정보량이 실행마다 요동친다는 것을 8B–120B 4 백본에서 측정한다. $L_w$(wake 지연 기여)를 실제 서빙 단위로 보고한 드문 사례 | WS |
| 24 | **2604.06370** | ForkKV: Scaling Multi-LoRA Agent Serving via Copy-on-Write Disaggregated KV Cache — Wang, Ren, Gui — 2026-04-07 | **F4** | per-agent LoRA가 **KV cache를 발산시켜 prefix caching을 무력화**한다는 것을 시스템 수준에서 규명하고 CoW로 최대 3.0× throughput을 얻는다. Θ-경로가 주류가 될 때 생기는 서빙 비용을 **HBM 바깥이 아니라 KV 쪽에서** 짚는다 — A2 논증의 반대 방향 증거 | WS |
| 25 | **2606.24151** | Metis: Bridging Text and Code Memory for Self-Evolving Agents — Dai, He, Li, Zhou, Li — 2026-06-23 | **F4** | 동일 경험 집합 위에서 **text 기억과 code 기억을 통제 비교한 첫 연구**로, 구축 비용·실행 효율의 상보적 트레이드오프를 측정한다. ch25의 통일 회계가 "E-경로"를 단일 형태로 다루는 것이 부당함을 보인다 | WS |

---

## 3. 버린 것과 이유

157편 중 위 25편을 제외한 나머지. 대표적인 것만 사유별로 묶는다.

### 3.1 판별식과 무관 — 주제가 다름 (약 60편)

`nested-learning` 인용자의 다수가 여기 속한다. NL을 아키텍처 인용으로만 쓴 도메인 논문들:
MuonSSM(2606.30461, SSM 최적화), Tapered LMs(2606.23670, 파라미터 배분), UniFS(2606.22794, VLA),
ORCA(2606.14222, 시계열), FOGO(2606.10406, 옵티마이저), Modular TTT(2608.07110, TTT 성분 ablation —
wake 시계라 조건 (1) 불충족), MemDLM(2603.22241), Mem3R(2604.07279), NL-MambaXCT(2605.27454),
CFNN·ChronoVAE·MSA·AllMem 등. 저널 논문(Plant Phenomics, Nature Genetics, Information Fusion)과
비-LLM 응용(pine wilt UAV, planetary FCL, humanoid control, video anomaly detection)도 전부 여기.

### 3.2 test-time·wake 시계 — 조건 (1) "질의 도착 전" 불충족 (약 20편)

GRACE(2606.19354), NCoTS(2601.11340), MASS/Test-Time Meta-Adaptation(2603.03524),
Absorber LLM(2604.20915), Mela(2605.10537), SECL(2604.09624), Bifrost(2602.05810),
TT-SI / Self-Improving LLM Agents at Test-Time(2510.07841).
— 전부 질의를 본 뒤 움직인다. GradMem(#18)만 예외로 남긴 이유는 **"한 번 읽고 여러 질의에 답한다"**는
상각 구조를 명시적으로 실험 설계에 넣었기 때문이다.

### 3.3 서베이·포지션·거버넌스 — 새 측정도 새 기제도 없음 (약 12편)

Hitchhiker's Guide to Agentic AI(2606.24937, 사실상 교재), Continual Learning in Transition(2608.06216),
Modular Memory is the Key(2603.01761, 포지션), Memory as Metabolism(2604.12034, 규범 제안),
Context Cartography(2603.20578), Memory for Autonomous LLM Agents(2603.07670, 서베이 — 책은 이미
2607.25380을 갖고 있다), Comprehensive AI governance(2606.00047), DAF-SELM(SEAL 감사 프레임워크),
epistemic betrayal 위계.
— **예외 처리**: `Contextual Agentic Memory is a Memo`(#21)는 포지션 논문이지만 **일반화 천장을
형식화**하므로 F3로 남겼다. `Lifelong ICL Requires Parametric Forms of Attention`(2606.25342)도
포지션이지만 W층 용량 천장을 다뤄 **차점**으로 둔다(아래 3.6).

### 3.4 self-evolving RL 일반 — 오프라인 상태 갱신이 아님 (약 15편)

SEAL(2605.24426, **동명이인** — Letta 라인의 SEAL이 아니라 agent–environment 공진화),
iReasoner(2601.05877), CPMobius(2602.02979), SOAR(2601.18778), Evolving-RL, EvoIR-Agent,
Federation over Text(2604.16778), AIP(2606.04781), AutoSci(2605.31468), DOVA(2603.13327),
AgentOdyssey(2606.24893), SWE-Spot(2601.21649), 2604.18131.
— 학습 라운드는 있으나 **"질의 도착 전 유휴 예산"이라는 시계가 없다**. 일반 RL 후속학습이다.

### 3.5 인접하지만 판정을 못 바꿈 (약 20편)

- **ZenBrain(2604.23878)** — sleep 성분과 저장 47.4% 절감·per-query 토큰 1/106을 보고해 F4에
  걸릴 뻔했으나, 1인 저자·15개 기제 동시 주장·재현 코드 부재로 **증거 등급이 corpus 최하위군보다
  낮다**. 인용은 가능하되 판정 재료로는 부적합.
- **LongMemEval-V2(2605.12493)·StreamMemBench(2606.14571)·ProAgentBench(2602.04482)** — 좋은
  벤치마크지만 세 편 모두 **비용을 보고하지 않는다**. 책의 공백(비용 회계)을 메우지 못하고
  기존 caveat를 재확인만 한다. MemDelta(#1)·Ground Truth First(#6)가 같은 자리를 **비용·통제까지
  포함해** 덮으므로 그쪽만 남겼다.
- **Panini(2602.15156)·FluxMem(2605.28773)·MEMORA(2607.14252)** — E-경로 오프라인 통합 사례로
  F1 자격은 있으나, 기제가 Mem0/Zep/ReasoningBank의 변주이고 **바이트·지연 어느 것도 보고하지
  않는다**. ch15의 판정을 바꾸지 못한다. MEMORA는 Auto-Dreamer의 두 인용자 중 하나라는 점만 기록.
- **SHINE(2602.06358)** — context→LoRA 단일 forward. Generative Adapter(ch16 경첩 장)의 정확한
  후속이지만 **시계가 wake라는 점이 GenAdapter와 동일**해 ch16의 판정을 바꾸지 않는다. 차점.
- **MemSFT(2607.25614)·SPA(2603.22213)·Knowledge is Not Enough(2601.11258)** — Θ층 주입이지만
  대상이 **도메인 코퍼스이지 사용자·세션 문맥이 아니다**. per-user 회계로 환산되지 않는다.
- **Memoir(2607.20792)** — read/write 결합의 학습속도 페널티를 통제 실험으로 잰 점은 좋으나
  **81,738 파라미터 장난감 규모**이고 저자 스스로 능력 페널티는 미입증이라고 쓴다.
- **Convergence of Continual Learning in Homogeneous Deep Networks(2606.30559)** — 순차 투영의
  전역 수렴 실패 이론. F3 성격이나 **LLM도 오프라인 시계도 없어** ch22에 접붙일 다리가 없다.
- **Phasor Agents(2601.04362)** — wake tagging과 offline consolidation을 실제로 분리한다.
  기제로는 F1이지만 **LLM이 아니라 Stuart-Landau 진동자 그래프**라 세 층으로 환원되지 않고
  decode 비용으로 번역되지 않는다.

### 3.6 차점 — 지금은 안 올리지만 기록해 둠

판정을 바꿀 가능성이 낮지만, 해당 장을 쓸 때 각주 재료가 될 수 있는 것.

| arXiv | 제목 | 쓸 자리 |
|---|---|---|
| 2606.25342 | Lifelong In-Context Learning with Transformers Requires Parametric Forms of Attention | ch18 W층 용량 천장 |
| 2602.06358 | SHINE: Scalable In-Context Hypernetwork for Mapping Context to LoRA | ch16 경첩 |
| 2607.14252 | MEMORA: Embodied Action Memory from Egocentric Videos | Auto-Dreamer 인용자 기록 |
| 2605.28773 | FluxMem: Rethinking Memory as Continuously Evolving Connectivity | ch15 |
| 2603.05923 | Learning Next Action Predictors from Human-Computer Interaction (LongNAP) | ch14 $N_q$ / 다음 질의 예측 |
| 2605.13360 | Speculative Interaction Agents: Asynchronous I/O and Speculative Tool Calling | ch11 경계, ch25 지연 |
| 2603.27138 · 2604.26557 · 2602.06502 | ScoutAttention / DUAL-BLADE / DualMap | ch29 KV 계층 보조 |
| 2511.04919 | BudgetMem: Learning Selective Memory Policies (2025-11) | ch26, 시기 밖이나 예산-쓰기 원형 |

---

## 4. 이 채널의 한계 (정직하게)

- **Semantic Scholar 색인 지연.** 2026-07~08 arXiv 논문은 S2 인용 그래프에 아직 안 들어온 것이
  많다. CrystalMem(2608.00303)이 Letta STC 인용자로 잡힌 것은 운이 좋은 편이고, MemDelta·
  User as Engram·HyMCache 같은 **최고 가치 후보들이 인용 그래프에서 전혀 나오지 않았다**.
  이 채널만으로는 F4 공백을 못 메운다 — WebSearch 보조가 필수였다.
- **`Do LMs Need Sleep?` 피인용 0 / Auto-Dreamer 2 / `LM Need Sleep` 3.** W·Θ-경로 앵커는
  인용 그래프 채널로 확장할 재료 자체가 거의 없다. 이 세 앵커에서 나온 후보는 #12 한 편뿐이다.
- **Nested Learning·SEAL의 인용자는 대부분 무관.** 91+53편 중 판정에 닿는 것은 6편이었다.
  이 두 앵커는 정밀도가 낮은 채널이다.
- **arXiv ID는 전부 arXiv API로 확인했다.** 추정으로 만든 ID는 하나도 없다. 후보 25편 모두
  `export.arxiv.org/api/query?id_list=`로 제목·날짜·저자·초록을 대조했고, 초록을 읽고 판정했다.
