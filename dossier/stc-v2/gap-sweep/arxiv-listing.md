# gap sweep — arXiv 직접 검색 채널

**작성**: 2026-08-11
**채널**: arXiv API(`export.arxiv.org/api/query`) + arXiv 검색 UI(`arxiv.org/search`) 직접 질의
**배제 기준**: `corpus-exclusion.md`의 deep-read 32편 + vendored arXiv ID 69개
**판정 기준**: `S0-DESIGN.md` §2(세 층 Θ/W/E·갱신식·비용 4종) + `PRE-RESEARCH.md` §2.1(판별식 네 조건)·§5(척추 논지 A1)
**시기 우선순위**: 2026-06 이후 > 2026 상반기 > 2025 이전(F1–F4에 강하게 해당할 때만)

---

## 1. 방문한 소스

### 1.1 arXiv API 질의 (`export.arxiv.org/api/query`, 전부 `sortBy=submittedDate&sortOrder=descending`)

| # | search_query | 반환 | 비고 |
|---|---|---|---|
| Q01 | `all:"sleep-time compute"` | 1 | Letta 원논문 1편뿐. **이 이름을 쓰는 후속 논문이 arXiv에 없다** |
| Q02 | `abs:"memory consolidation" AND abs:"language model"` | 27 | 최대 수확처 |
| Q03 | `abs:"sequential model editing"` | 8 | 대부분 2024 |
| Q04 | `abs:"KV cache" AND abs:"CXL"` | 15 | F4 후보군 |
| Q05 | `abs:"self-evolving" AND abs:"agent" AND abs:"memory"` | 40 | 응용 논문 다수 |
| Q06 | `abs:"idle" AND abs:"agent" AND abs:"LLM"` | 32 | F2·F4 핵심 |
| Q07 | `abs:"before the user" AND abs:"precompute"` | 1 | 무관(SQL) |
| Q08 | `abs:"context compaction"` | 27 | 천체물리 noise 혼입 |
| Q09 | `abs:"lifelong" AND abs:"model editing"` | 17 | |
| Q10 | `abs:"memory" AND abs:"baseline" AND ti:"agent" AND abs:"evaluation"` | 40 | |
| Q11 | `abs:"test-time training" AND abs:"agent"` | 15 | |
| Q12 | `abs:"memory" AND abs:"confound" AND cat:cs.CL` | 13 | F3 핵심 |
| Q13 | `abs:"model collapse" AND abs:"self-generated"` | 8 | 신규 가치 낮음 |
| Q14 | `abs:"per-user" AND abs:"LoRA"` | 9 | **F1·F4 최대 수확** |
| Q15 | `abs:"offline" AND abs:"before the query"` | 2 | 무관 |
| Q16 | `ti:"sleep" AND cat:cs.CL` | 15 | 수면의학 noise |
| Q17 | `abs:"continual" AND abs:"injection" AND abs:"forgetting" AND abs:"weights"` | 15 | |
| Q18 | `abs:"byte budget" OR abs:"bytes per user"` | 7 | **WritePolicyBench 발견** |
| Q19 | `abs:"memory write" AND abs:"latency" AND abs:"agent"` | 0 | 0건 자체가 정보 |
| Q20 | `abs:"anticipate" AND abs:"next query" AND abs:"LLM"` | 1 | OnePred |
| Q21 | `abs:"amortize" AND abs:"offline" AND abs:"inference" AND cat:cs.CL` | 3 | **Cartridges 발견** |
| Q22 | `abs:"fast weights" AND abs:"inference"` | 22 | W-경로 |
| Q23 | `abs:"memory" AND ti:"cost" AND abs:"agent"` | 23 | F4 |
| Q24 | `id_list=` 8건 일괄(초록 전문 검증) | 8 | |

### 1.2 arXiv 검색 UI (`arxiv.org/search`) — API 429 회피용

- `agent memory byte budget` → 3건(**2606.25115 신규 발견**)
- `sleep-time compute offline before queries` → 1건(Letta뿐)
- `idle time precompute LLM serving proactive` → **0건**
- `nightly consolidation user weights LLM` → 1건(2605.24657)
- `proactive agent precompute response during idle dialogue` → **0건**
- `memory system evaluation no-memory baseline reproduction agent` → 2건(무관)

### 1.3 초록 전문 확인 (`arxiv.org/abs/<id>`, 개별 확인)

2602.02574 · 2607.17545 · 2606.19172 · 2605.24657 · 2608.00303 · 2506.06266 · 2601.00821 ·
2605.29630 · 2608.01326 · 2605.22154 · 2607.19214 · 2606.09613 · 2606.11712 · 2606.17628 ·
2608.09819 · 2606.25115 · 2606.15734 (17편 전문 확인)
+ `id_list` 일괄로 초록 전문 확보: 2606.00866 · 2607.03441 · 2608.00101 · 2605.11836 ·
2601.07978 · 2608.05483 · 2606.25161 · 2605.23668 (8편)

**아래 표의 모든 arXiv ID는 실제 응답에서 읽은 값이다. 추측한 ID는 없다.**

### 1.4 채널 자체가 알려준 사실 (본문에 쓸 수 있는 것)

- `all:"sleep-time compute"` 전수 = **1편**. 2504.13171 이후 이 용어를 제목·초록에 쓰는 논문이 arXiv에
  존재하지 않는다. PRE-RESEARCH §5의 "이름이 기제보다 넓다"는 진단은 오히려 **이름이 퍼지지도
  않았다**는 형태로 강화된다 — 후속 문헌은 같은 일을 하면서 다른 이름(consolidation / crystallization /
  internalization / cartridge / engram / gradient bank)을 쓴다.
- `abs:"memory write" AND abs:"latency" AND abs:"agent"` = **0건**. 쓰기 경로의 지연을 초록에서
  명시하는 논문이 없다. ch25 "쓰기 검증 비용이 병목"의 문헌적 공백이 2026-08 시점에도 유지된다.
- 반면 **바이트 보고는 더 이상 0편이 아니다**(§4 참조). PRE-RESEARCH §4의 caveat
  "corpus 전체에서 상태 용량을 바이트로 보고한 논문이 0편이다"는 corpus 한정으로 다시 써야 한다.

---

## 2. 후보 표

정렬 = 판정을 바꿀 수 있는 정도 순. `F`는 과제 정의의 F1(판별식 통과)/F2(경계·회색지대)/
F3(제약·반증)/F4(비용·시스템).

### 2.1 Tier 1 — 판정을 직접 바꾼다

| # | arXiv | 제목 / 1저자 | 날짜 | F | 왜 판정을 바꾸는가 |
|---|---|---|---|---|---|
| 1 | 2605.24657 | Beyond Inference-Only Deployment: Comparing Weight-Based Consolidation Against Cascading Compaction — Simon Dennis | 2026-05-23 | F1·F3 | corpus에 없는 **Θ-경로 대 E-경로 직접 대결**을, no-context 바닥(11.8%)과 full-context 천장(90.1%) 양쪽 통제와 함께 수행한다(야간 LoRA 통합 80.4% vs 3회 압축 36.8%, paired t(9)=14.8) |
| 2 | 2601.00821 | Fidelity Before Structure: Verbatim Chunks Beat Lossy Artifact Extraction in Long-Conversation LLM Memory — Tao An | 2025-12-23 | F3 | E-경로의 전제 자체(구조화 산출물이 원문보다 낫다)를 **저장 표현만 바꾸는 통제 ablation + 6종 confound 통제**로 반증한다(LoCoMo 43.9 vs 28.0, LongMemEval-S 67.4 vs 45.4). ch15·ch27의 "품질 1위가 기억 없음"에 기제를 준다 |
| 3 | 2606.19172 | User as Engram: Internalizing Per-User Memory as Local Parametric Edits — Bojie Li | 2026-06-17 | F1·F4 | Θ-경로 per-user 상태를 **바이트로 보고**하고(per-user LoRA 대비 ~33,000× 작은 footprint) 사용자 간 해시 슬롯 분리로 **가법적 합성**을 보인다. A2(쓰기 용량이 새 병목)의 산술 전제를 바꾼다 |
| 4 | 2506.06266 | Cartridges: Lightweight and general-purpose long context representations via self-study — Sabri Eyuboglu | 2025-06-06 | F1·F4 | 질의 전 오프라인으로 KV cache를 **학습**하고 그 비용을 같은 corpus의 모든 질의에 상각한다 — 식 (A)의 교과서적 사례. 게다가 서빙 비용을 실측(메모리 38.6×↓, 처리량 26.4×↑). corpus에 통과군 + 비용 동시 보고 사례가 없다 |
| 5 | 2606.11712 | Substrate Asymmetry in User-Side Memory: A Diagnostic Framework — Youwang Deng | 2026-06-10 | F3 | per-user LoRA(Θ) 대 dense retrieval(E)을 **세 직교 축**(스타일 일관·사실 존재·사실 부재)에서 갈라 보이고, attention 21–35층 query-projection의 인과 ablation으로 두 효과가 **반대 방향으로 같은 셀에 실린다**는 것을 보인다. "경로마다 다르다"를 축 단위로 정밀화한다 |
| 6 | 2606.25115 | Forget to Improve: On-Device LLM-Agent Continual Learning via Budget-Curated Memory — Beining Wu | 2026-06-23 | F4·F3 | **net-value-per-byte** 단일 척도로 KEEP/SHARE/TRUST를 결정하고, 실제 Jetson 테스트베드에서 RAM·에너지·uplink를 측정한다(메모리 2.7×↓, uplink 2.4×↓). ch26의 바이트 회계에 외부 실측 대조군이 생긴다 |
| 7 | 2608.00303 | CrystalMem: Elastic Memory for Self-Evolving LLM Agents via Knowledge Crystallization — Beining Wu | 2026-07-31 | F3·F4 | 바이트 예산을 줄였다 복구해도 능력이 안 돌아오는 **memory hysteresis**를 보이고, keep/drop만 하는 정책에 **residual-deficit floor**가 있음을 증명한다. E-경로 `wr`의 비가역성에 상한 정리를 준다 — ch30 반증 조건 재료 |
| 8 | 2602.02574 | WritePolicyBench: Benchmarking Memory Write Policies under Byte Budgets — Edgard El Cham | 2026-01-31 | F3·F4 | **byte-accurate cost model**로 쓰기 정책(store/merge/evict)을 drift 있는 스트림에서 평가하는 벤치. corpus 어느 논문도 갖지 못한 통제된 쓰기-예산 축을 제공한다(cs.PF) |
| 9 | 2606.15734 | Retrievable Gradients: Continual Post-Training Without Cumulative Weight Drift — Weihang Su | 2026-06-14 | F2·F1 | 문서별 gradient를 **오프라인 사전계산**해 색인(Gradient Bank)에 넣고 질의 시 관련 gradient만 꺼내 **일시적** 가중치 적응을 한다. Θ 모양의 산출물을 E처럼 저장해 W처럼 쓰는 **세 층 어디에도 정확히 안 맞는 네 번째 자리** — 판별식의 층 분할 자체를 시험한다 |
| 10 | 2608.01326 | Context Compaction Theory — Hayder Tirmazi | 2026-08-02 | F3 | compaction을 **one-way communication complexity와 등가**임을 증명해 하한을 이식하고, selection < generation의 **분리**를 증명한다. 게다가 배포된 상용 compaction 엔드포인트를 최적 대비 평가한다. rate–distortion(2607.08032)보다 강한 형태의 상한 |

### 2.2 Tier 2 — F4(비용·시스템). 이 책의 최대 공백을 직접 메운다

| # | arXiv | 제목 / 1저자 | 날짜 | F | 왜 판정을 바꾸는가 |
|---|---|---|---|---|---|
| 11 | 2608.00101 | Agentic Coding in the Wild: Characterizing GitHub Copilot Traces at Production Scale — Banruo Liu | 2026-07-30 | F4 | 3.2M 사용자·13M 세션·761M LLM 호출·95T 토큰 **프로덕션 트레이스**. 턴 내 KV hit 90% → 턴 경계 55%, 그리고 **턴 경계의 분(minutes) 단위 유휴**를 정량화하고 idle 예측기로 86–90%를 포착한다. 상각식 (A)의 $N_q$·$B_s$ 가용시간을 문헌 최초로 실측 근거 위에 놓는다 |
| 12 | 2606.00866 | Idleness is Relative: Exploiting Tool-Call Idle Windows for Offloading in Agentic Systems with MORI — Tian Xia | 2026-05-30 | F4 | 실제 Claude Code 워크로드에서 유휴를 **이진이 아닌 연속 스펙트럼**으로 재정의하고 HBM/DRAM 계층에 배치한다(처리량 +20–71%, TTFT −18–43%). "유휴 시간에 예산을 쓴다"는 sleep-time의 전제가 하드웨어 용량비에 걸린다는 것을 보인다 — ch29 |
| 13 | 2607.19214 | Keeping the Cache Warm Pays: Keepalive Economics for Agentic Workloads — Maxim Khailo | 2026-07-21 | F4 | Anthropic/OpenAI/Google/DeepSeek **실제 가격·TTL**로 sleep–wake 간극의 손익분기를 유도한다(post-pause 비용 최대 12.5×↓, 손익분기 ~46분). ch14 caveat "$\kappa_{\text{cost}}=10$에 민감도 분석이 없다"에 대응하는 외부 회계 |
| 14 | 2601.07978 | Cost and Accuracy of Long-Term Memory in Distributed Multi-Agent Systems Based on LLMs — Benedict Wolff | 2026-01-12 | F4·F3 | **공급자 아닌 독립 재현 테스트베드**로 정확도·지연·CPU시간·peak RAM·disk I/O·네트워크를 동시 측정. RAG가 mem0와 동급 정확도를 **8.4× 낮은 TCO**로 낸다. ch15·ch27의 E-경로 판정에 비용 축을 붙인다 |
| 15 | 2608.05483 | PLoRA: An NDP-Enhanced Pooled-Memory System for Cost-Efficient Multi-LoRA Serving — Zhongkai Yu | 2026-08-06 | F4 | H100 1장에 **1000개 어댑터**를 CXL/NVLink 풀 메모리 + NDP로 서빙(디코드 지연 평균 6.6×↓). Θ-경로가 주류가 될 때의 per-user delta 서빙을 하드웨어로 실측한 유일 사례 — ch29 memory-device 기회의 직접 대응물 |
| 16 | 2606.09613 | AGENTSERVESIM: A Hardware-aware Simulator for Multi-Turn LLM Agent Serving — Rakibul Hasan Rajib | 2026-06-08 | F4 | 도구 호출 간극·턴 간 캐시 지역성·HBM/DRAM/CXL KV 잔류를 프로그램 단위로 모사하고 실기 대비 6% 오차. X3·X4를 GPU 없이 돌릴 수 있는 도구 — 이 책의 실험 계획을 바꾼다 |

### 2.3 Tier 3 — 기제 추가(F1)와 경계 사례(F2)

| # | arXiv | 제목 / 1저자 | 날짜 | F | 왜 판정을 바꾸는가 |
|---|---|---|---|---|---|
| 17 | 2607.17545 | Retain or Consolidate? Budget-Dependent Operator Selection for Language Agent Memory — Qingcan Kang | 2026-07-20 | F1·F3 | "보존 대 통합"의 우열이 **예산에 따라 교차**함을 coverage/replacement 효과 분해로 설명하고 LongMemEval·LoCoMo에서 crossover를 재현한다(빡빡한 예산에서 통합이 최대 48% 우세). "유망한가"가 예산 레짐 함수임을 보이는 첫 통제 실험 |
| 18 | 2606.17628 | OPD-Evolver: Cultivating Holistic Agent Evolver via On-Policy Distillation — Guibin Zhang | 2026-06-16 | F1 | 느린 루프가 경험을 **정책 가중치로 증류**하고 빠른 루프가 4단 메모리를 읽고 쓴다 — Θ와 E를 한 시스템에서 잇고 ReasoningBank(corpus 통과군)를 11.5% 능가한다. 경로 간 침묵이 깨지는 첫 사례 |
| 19 | 2606.25161 | TRUSTMEM: Learning Trustworthy Memory Consolidation for LLM Agents with Long-Term Memory — Tianyu Yang | 2026-06-23 | F1 | `wr` 연산을 선호 기반 RL로 **학습**하고, 누락·손상·환각을 40.1%/79.1%/50.0% 줄인다. corpus의 `wr`은 전부 프롬프트 규칙이었다 — (U-E)의 갱신 규칙이 학습 대상이 되는 전환점 |
| 20 | 2608.09819 | Macaron-V1: Towards Open Continual Learning with Self-Improvement and Mixture-of-LoRA — Mind Lab | 2026-08-10 | F1·F4 | 744B 베이스 + 사용자 턴마다 LoRA 1개 선택, 배포 후 계속 학습하는 **프런티어 규모 생산 시스템**. corpus의 Θ-경로 실증 상한(소형 연구 모델)을 두 자릿수 끌어올린다 — scale_ceiling 재작성 대상 |
| 21 | 2605.23668 | OnePred: Next-Query Prediction via Recursive Intent Memory in Multi-Turn Conversations — Jiangwang Chen | 2026-05-22 | F1·F2 | Letta STC가 "사용자가 물을 법한 질의를 예상한다"고만 쓰고 넘어간 부분을 **과제로 세우고 벤치(NQP-Bench)를 만든다**. 재귀 갱신 메모리로 턴당 토큰 22×↓. STC 상각의 분모($N_q$)를 예측 가능한 양으로 바꾼다 |
| 22 | 2605.22154 | IdleSpec: Exploiting Idle Time via Speculative Planning for LLM Agents — Daewon Choi | 2026-05-21 | F2 | 유휴에 예산을 쓰지만 산출물이 **같은 에피소드에서 즉시 소모**되고 Θ·W·E 어디도 지속적으로 바뀌지 않는다. 판별식 조건 (3)·(4)가 무엇을 걸러내는지 가장 선명한 반례 — ch11 경계 사례로 MemGPT 옆에 둘 것 |
| 23 | 2607.03441 | No Time Like the Present: Agentic Test-Time Training for LLM Agents — Yanbo Wang | 2026-07-03 | F2·F3 | 에피소드 **안에서** 가중치를 계속 갱신해 조건 (1)을 못 맞추지만, 자기 출력으로 재학습할 때의 **drift 증폭**을 n-gram 반복으로 진단하고 vLLM LoRA API로 오버헤드를 1.9×로 실측한다. $\rho$(반복 갱신 열화)의 첫 온라인 측정 |
| 24 | 2605.11836 | More Edits, More Stable: Understanding the Lifelong Normalization in Sequential Model Editing — Xin Ma | 2026-05-12 | F3 | 장기 편집이 무너지지 않는 편집기들이 공유하는 것이 Lifelong Normalization임을 밝히고, 그것이 **점근적 직교성과 유계 노름**을 준다는 이론을 세운다. ch22(가중치에 사실을 계속 넣을 수 있는가)의 반증 축에 "무엇이 있으면 되는가"를 붙인다 |
| 25 | 2605.29630 | Entity-Collision: A Stratified Protocol for Attributing Retrieval Lift in Agent Memory — Youwang Deng | 2026-05-28 | F3 | 기억 벤치의 hit@k가 **lexical leakage와 tag-mixing을 뒤섞는다**는 것을 보이고, 모든 distractor가 정답 엔티티 토큰을 공유하도록 BM25 바닥을 구성으로 고정한다. "왜 자명한 baseline이 자꾸 이기는가"의 방법론적 답 |

---

## 3. 버린 것과 이유

전체 훑은 항목 약 330편 중 위 25편만 남겼다. 버린 이유별로 묶는다.

| 버린 묶음 | 대표 예 | 버린 이유 |
|---|---|---|
| **에이전트 기억 보안·공격** | 2607.29167(Memory Provenance Laundering), 2608.01679(Authority Collapse), 2608.03509(SkillJack), 2608.01759(Experience Composition), 2606.23075, 2608.06862(SynChain), 2605.18930(OEP) | 상태 변경은 실재하나 논문의 질문이 **적대적 견고성**이다. 판별식 네 조건도 비용 4종도 다루지 않아 판정을 바꾸지 못한다. (다만 "지속 상태가 공격면"이라는 사실은 ch30 한계 절의 한 줄 각주 가치는 있다) |
| **일반 기억 아키텍처(예산·비용 미보고)** | 2601.02845(TiMem), 2602.09712(TraceMem), 2608.08236(LatticeMind), 2608.07438(PsychoAgent), 2608.02113(MemArbiter), 2608.05095, 2608.08253, 2605.03312(MemFlow), 2607.27773(ChronoMem), 2603.15658 | E-경로 변종. `wr`을 재배열할 뿐 $B_s$·$C$·$\rho$ 중 무엇도 새로 보고하지 않는다. ch15에 이미 5편이 있고 6번째를 더해도 논지가 안 움직인다 |
| **self-evolving 응용** | 2607.28692, 2607.21125, 2606.30296(ManimAgent), 2607.26490(EvoPINN), 2608.04872(A-SR), 2606.30949, 2607.24998, 2607.09521, 2607.13940, 2606.24081, 2606.07436, 2606.31537, 2606.18235, 2606.12780, 2608.03392, 2608.05144 | 도메인 응용. 기제는 ReasoningBank/SEAL 계열의 재사용이고 새 제약·새 비용이 없다 |
| **모델 편집 기법 논문** | 2506.17864(QueueEDIT), 2605.08143(HoReN), 2604.11214(HiEdit), 2604.07965, 2603.11239, 2506.07899(MEMOIR), 2505.14679(UltraEdit), 2605.20273 | ch19가 이미 ROME·MEMIT로 primitive와 상한을 다룬다. 점진적 성능 개선은 Θ-경로 판정을 바꾸지 않는다. **예외로 2605.11836만 남겼다** — 붕괴의 *이론*을 주기 때문 |
| **model collapse 후속** | 2608.04268(Fairness Collapse), 2605.24998, 2510.16657, 2506.19262, 2505.08803 | ch06이 2404.01413로 이미 반증 축을 세웠다. 이들은 도메인 확장(공정성·VLM·검증)이지 Θ-경로 sleep 루프의 $\rho$를 재는 것이 아니다 |
| **일반 KV cache·CXL 시스템** | 2607.27187, 2607.18141, 2606.19746(SAC), 2606.12556(ITME), 2604.26968, 2512.18194, 2511.00321, 2509.03377, 2607.12550(JoLT), 2602.16284, 2604.18529 | F4처럼 보이지만 측정 대상이 **일반 장문 추론의 KV cache**다. "이 워크로드"(질의 전 오프라인 갱신, per-user 상태)를 재지 않는다. 에이전트/멀티턴/per-user에 특정된 3편(MORI·AGENTSERVESIM·PLoRA)만 남겼다 |
| **에이전트 speculative 실행** | 2607.03333(SPORK), 2604.16469(B-PASTE), 2607.12236(Speculate with Memory), 2606.07846, 2607.23933(SpecBox), 2506.14852(Agentic Plan Caching) | IdleSpec과 같은 경계에 있다(유휴 예산 → 즉시 소모, 지속 상태 변경 없음). 대표 1편(IdleSpec)만 남기고 중복 제거. Agentic Plan Caching은 계획 템플릿이 세션을 넘어 남아 경계선이 더 모호하나, 2025-06이고 비용 보고가 얕아 제외 |
| **W-경로 fast-weight 아키텍처** | 2605.04651(FAAST), 2601.00671(Fast-weight PKM), 2604.07350, 2604.07279, 2605.02920, 2606.24756, 2608.01672 | NM 모노그래프의 TTT 라인 연장이고 $B_s=0$(질의 전 오프라인 단계가 없다). ch17·ch18의 판정을 바꾸지 않는다. 2608.01672(Learning What to Remember)는 "미래 효용"을 목적에 넣어 아깝지만 여전히 온라인 갱신이라 제외 |
| **평가·벤치 일반** | 2605.20833(MemGym), 2606.17546(SEAGym), 2608.06144(FinEvo-Bench), 2608.00805(AgentSLABench), 2606.16613, 2608.06909 | 판별식을 시험하지 않고 과제 성공률을 잰다. 다만 2608.00805(자원 예산 하 지연·비용·메모리 동시 측정)는 F4로 재검토 여지가 있다 — **차순위 1번**으로 기록 |
| **주제 무관 noise** | 천체물리 "compact" 다수, 수면의학 "sleep" 다수, 2503.00714(SQL 투기 실행) | 질의어 다의성 |

### 3.1 아깝게 떨어뜨린 것 (차순위)

25건 상한 때문에 뺐지만 다음 라운드에서 되살릴 값이 있는 것:

1. **2608.00805** AgentSLABench — 선언된 자원 예산 하에서 정확도와 지연·비용·연산·메모리·네트워크를 동시에 프로파일. F4로는 유효하나 sleep-time 특정성이 없다.
2. **2605.21951** Dynamic Mixture of Latent Memories for Self-Evolving Agents (2026-05-21) — "가중치 갱신은 망각을 부르고 외부 기억은 내재 능력을 못 올린다"는 정확히 Θ/E 딜레마를 문제로 세운다.
3. **2608.02508** RoMeRL — 학습형 기억의 memory-reward trap과 궤적 색인 효용의 상태공간 폭발. $\rho$ 인접.
4. **2603.06642** SR-TTT Does Not Learn Retrieval: A Correction and Mechanistic Post-Mortem — 자기 논문 v1 주장을 저자가 철회·해부한 사례. W-경로 정직성 caveat 재료로 특이하다.
5. **2607.05378** CompactionRL — compaction 정책을 RL로 학습. 17번(Retain or Consolidate?)과 중복도가 높아 제외.
6. **2606.22528** Governance Decay — 반복 compaction이 안전 제약을 조용히 지우는 현상. $\rho$의 비-정확도 형태.

---

## 4. 이 채널이 확정한 것 — 본문 수정이 필요한 지점

1. **PRE-RESEARCH §4 마지막 caveat("corpus 전체에서 상태 용량을 바이트로 보고한 논문이 0편이다")는
   corpus 한정 진술로 다시 써야 한다.** corpus 밖에는 2026 상반기에만 최소 4편이 있다 —
   2606.19172(footprint 배수), 2606.25115(net-value-per-byte, 실기 RAM·에너지), 2608.00303(byte cap 하
   hysteresis), 2602.02574(byte-accurate cost model). "이 책이 동결한 corpus에는 0편이고, 동결 직전·직후에
   바이트 회계가 독립적으로 등장하기 시작했다"가 정확한 서술이다.

2. **"세 경로가 서로를 인용하지 않는다"는 2026-05 이후 깨지기 시작한다.** 2605.24657(Θ vs E 직접 비교),
   2606.11712(Θ vs E 축별 분해), 2606.17628(Θ+E 한 시스템)은 전부 경로를 가로지른다. ch27·ch30은
   "침묵은 연대가 아니라 선택이었고, 선택이 바뀌는 중"으로 시제를 갱신해야 한다.

3. **판별식의 층 3분할에 네 번째 자리가 생겼다.** 2606.15734(ReGrad)는 Θ 모양 산출물을 E처럼 색인해
   W처럼 일시 적용한다. ch11 분기 장은 이 조합을 명시적으로 처리하거나, 처리하지 않는 이유를 밝혀야 한다.

4. **상각식 (A)의 $N_q$는 이제 미보고 양이 아니다.** 2608.00101이 프로덕션 규모에서 턴 구조·유휴 길이·
   캐시 재사용률을 준다. ch14·ch25의 "실서빙 $N_q$ 분포를 보고한 논문이 corpus에 없다"는 유지하되,
   "corpus 밖에서 2026-07에 처음 보고되었다"를 병기해야 한다.
