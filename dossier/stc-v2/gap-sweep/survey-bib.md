# gap-sweep / 채널: 서베이 참고문헌 채굴 + 학계 용어 검색

**작성**: 2026-08-11
**대상**: STC v2 corpus(deep-read 32편 + vendored 69 ID) 동결 이후의 누락 탐색
**판정 기준**: `S0-DESIGN.md` §2.3 판별식 네 조건, `PRE-RESEARCH.md` §2 전수 분류·§5 척추 논지 A1
**분류**: F1 판별식 통과 후보 / F2 경계·회색지대 / F3 제약·반증 / F4 비용·시스템 실측

---

## 0. 한 문단 요약

이 채널의 결론은 **"공백은 기제가 아니라 회계와 반증에 있다"** 이다. 새로 찾은 25편 중
E-경로 기제를 하나 더 얹는 종류는 의도적으로 거의 넣지 않았다(그것은 판정을 바꾸지 않는다).
대신 (a) 이 워크로드의 **쓰기 경로 비용을 실제로 계측한** 시스템 논문 8편,
(b) 판별식을 정면으로 통과하는 **유휴시간 사전계산** 신규 사례 6편,
(c) 반복 갱신의 **붕괴를 정량화**한 제약 논문 9편,
(d) 판별식 자체를 시험하는 경계 사례 2편을 올린다.

특히 세 건은 본서의 확정된 caveat를 **직접 무효화하거나 뒤집을 수 있다**:

| 본서의 현재 단정 | 이를 흔드는 후보 |
|---|---|
| "corpus 전체에서 상태 용량을 **바이트로 보고한 논문이 0편**"(ch26) | 2603.04428(GB 예산·4배 수용), 2606.19172(per-user 33,000× 축소) |
| "상각식 (A)는 $N_q$를 안다고 가정 — 실서빙 $N_q$ 분포 보고 논문 없음"(ch14·ch25) | 2606.06448(construction/retrieval/generation 3분할 프로파일 + "amortization via query volume" 권고) |
| "반복 sleep의 $\rho$는 미측정"(ch29·ch30, X4 예정) | 2606.04703(다회차 경험 내재화의 **진행성 붕괴**), 2605.11836(초기 편집이 후속 편집을 **돕는다**는 역방향 관측) |

---

## 1. 방문한 소스 (채널별)

### 1.1 서베이 참고문헌 전수 파싱 (프로그램 처리)

| 서베이 | 위치 | 참고문헌 | 배제 69ID와의 차집합 | 산출 |
|---|---|---|---|---|
| `2607.25380` Memory for Large Language Models | vendored `papers/stc/2607.25380.txt` | 85항목(84 파싱, [74] 누락 = OCR 결손) | **74항목** | 아키텍처 계열이 지배. 채택 2편(2505.16950, 2607.02574는 여기서 미검출) |
| `2603.07670` Memory for Autonomous LLM Agents (2026-03-08) | 신규 다운로드 | arXiv ID 36개 추출 | **34개** | 대부분 2023–2024 에이전트 기억. 채택 1편(2601.01885), 후보 확인 2편 |
| `2602.19320` Anatomy of Agentic Memory (2026-02-22, v2 05-20) | 신규 다운로드 | arXiv ID 86개 추출 | **84개** | **가장 생산적**. 2025–2026 에이전트 기억이 조밀. 여기서 학습형 쓰기 정책 계열과 pre-storage 계열을 발견 |
| `2607.02574` KV Cache Management 서베이 (2026-06-30) | 신규 다운로드 | arXiv ID 59개 추출 | **57개** | 시스템 계열. 채택 2편(2604.06370, 2607.18141은 검색과 교차확인) |

파싱 스크립트: `parse_refs.py` / `parse2.py` / `mine.py`
(스크래치패드 `…/scratchpad/gap/`). 손으로 세지 않았다.

> **파싱 주의**: `2607.25380.txt`는 pdftotext 산물이라 `arxiv.org/abs/ 2004.05150`처럼 URL이
> 공백으로 끊긴 항목이 14건 있었다. 정규화 후 재파싱해 반영했다. [74]는 원 텍스트에서
> 항목 자체가 결손이다(파서 문제가 아님).

### 1.2 웹 검색 (앞 단계가 세운 학계 용어)

실행한 질의 8건 — 모두 실제 결과를 열어 초록까지 확인했다.

1. `survey memory in LLM agents offline consolidation sleep-time compute 2026`
2. `agent memory write path cost latency bytes measured serving overhead benchmark 2026`
3. `idle-time precompute LLM assistant before user query speculative response latency measured 2026`
4. `proactive precomputation anticipate next user query prefetch answer idle GPU 2026`
5. `sequential knowledge editing collapse limit how many edits lifelong editing degradation measured 2026`
6. `per-user LoRA parametric memory personalization serving thousands of adapters footprint measured 2026`
7. `self-evolving agent internalize experience into weights continual RL collapse over rounds 2026`
8. `repeated self-generated data consolidation rounds degradation model collapse agent memory drift 2026`
9. `KV cache offloading CXL memory tier LLM inference measurement bytes per session 2026`

### 1.3 초록 확인 경로

arXiv API(`export.arxiv.org/api/query`)로 일괄 조회, 레이트리밋 구간은 `arxiv.org/abs/*`
개별 조회로 보완. **표에 오른 25편은 전부 초록을 읽고 판정했다.** 제목만으로 올린 것은 없다.

---

## 2. 후보 (25편)

배제 69 ID와의 교집합 **0건**(프로그램 대조 완료).

### 2.1 F4 — 비용·시스템 실측 (8편) · 본서 최대 공백

| # | arXiv | 제목 / 저자 / 날짜 | 왜 판정을 바꾸는가 | 발견 채널 |
|---|---|---|---|---|
| 1 | **2606.06448** | Agent Memory: Characterization and System Implications of Stateful Long-Horizon Workloads — Omri, Gan, Broveak, Geens, He, Pentland 외 · 2026-06-04 | **에이전트 기억의 첫 시스템 특성화.** construction/retrieval/generation으로 비용을 귀속하는 phase-aware 프로파일러로 10개 시스템×2 벤치 계측하고, 권고에 "amortization via query volume"과 freshness-latency 트레이드오프를 명시한다. ch25의 통일 회계와 ch14 $N_q$ caveat를 **외부 실측으로 대체 가능**하게 만든다 | 검색 2 |
| 2 | **2606.24775** | Are We Ready For An Agent-Native Memory System? — Zhou, Zhou, Han, Xu, Li, Li 외 · 2026-06-23 | 12개 시스템 + 2 baseline × 5 워크로드 · 11 데이터셋을 **모듈 4분할**(표현·추출·검색라우팅·유지보수)로 분해 계측. "어떤 아키텍처도 전 시나리오를 지배하지 않는다", "국소 유지보수가 전역 재구성보다 비용효율적"은 ch27 경로별 판정의 직접 증거 | 검색 2 |
| 3 | **2605.23986** | MemForest: An Efficient Agent Memory System with Hierarchical Temporal Indexing — Chen, Zhang, Pei, He, Wu, Zeng 외 · 2026-05-16 (v2 07-31) | 기억을 **write-efficient 시계열 데이터관리 문제**로 재정식화하고 "memory-freshness latency"와 input-normalized build rate(6.0×/9.5×)를 보고. 본서가 한 번도 다루지 않은 **쓰기 경로 지연**을 1차 지표로 올린 사례 | 검색 2 |
| 4 | **2608.00009** | AgentMemBench — Cherif · 2026-06-16(ID는 2608대, 날짜 불일치 주의) | 5개 기억 전략을 동일 조건에서 **Memory Footprint + Latency까지** 표로 낸다(EKV ~5,100 vs ICW/WAM ~300 토큰). LoCoMo 장거리에서 ICW·WAM·GEM·CBS가 Recall@5 ≤ 0.005로 **붕괴**. ch26의 "바이트 보고 0편"과 ch15의 baseline 서술을 동시에 건드린다 | 검색 2 |
| 5 | **2603.04428** | Agent Memory Below the Prompt: Persistent Q4 KV Cache for Multi-Agent LLM Inference on Edge — Shkolnikov · 2026-02-17 | 에이전트별 KV를 **4bit로 디스크에 영속화**하고 되읽어 TTFT를 22–136× 줄인다. 10.2 GB 예산·에이전트 3개 수용·Q4로 4배 등 **용량을 실제 바이트로** 보고. F2이기도 하다(바뀌는 상태가 Θ·W·E 어디에도 없다) | 검색 2 |
| 6 | **2607.18141** | A CXL Memory Rack for Multi-Turn LLM Serving (HyMCache) — Jang, Song, Kim, Noh, Kim · 2026-07-20 (v3 2026-08-05) | **실물 CXL-HM 프로토타입**에서 TB급 재사용 KV를 서빙. LMCache 대비 3.0×, Mooncake 대비 DRAM 16분의 1. X2가 정량화한 "웜 계층 delta 스테이징"(ch29)의 **하드웨어 실측 대응물** | 검색 9 |
| 7 | **2607.02574** | From Tensor Buffer to Distributed Memory Hierarchy: A Survey of KV Cache Management for LLM Serving — Li, Wang, Chen · 2026-06-30 | 30여 시스템을 locality·lifetime·ownership·substrate 4축으로 분류하고, **현행 평가에 빠진 KV 고유 측정 7종을 지명**한다. ch25·ch29가 쓸 수 있는 기성 측정 결손 목록 | 검색 9 |
| 8 | **2604.06370** | ForkKV: Scaling Multi-LoRA Agent Serving via Copy-on-Write Disaggregated KV Cache — Wang, Ren, Gui · 2026-04-07 | per-agent LoRA가 **KV 발산을 일으켜 prefix caching을 무력화**한다는 것이 진짜 병목임을 보이고 CoW로 3.0× 처리량 회복. A2("per-user 가중치 delta가 상태량을 지배")의 비용 모델을 **어댑터 바이트에서 KV 발산으로 이동**시킨다 | 서베이 `2607.02574` 참고문헌 [128] |

### 2.2 F1 — 판별식 통과 후보 (6편)

| # | arXiv | 제목 / 저자 / 날짜 | 왜 판정을 바꾸는가 | 발견 채널 |
|---|---|---|---|---|
| 9 | **2605.25971** | Anticipate and Learn: Unleashing Idle-Time Compute in Proactive Agents (ProAct) — Hu, Lyu, Kong, Liu, Lin 외 · 2026-05-25 | 조건 넷을 정면으로 만족한다: 질의 **전**, 유휴 예산으로, 대화이력+persistent memory를 갱신하고, 이후 질의에 쓴다. ProActEval(200 시나리오·40 도메인)은 corpus에 없는 **sleep 예산 전용 벤치**. 통과군 9편 → 확장 1순위 | 검색 4 |
| 10 | **2607.12236** | Speculate with Memory: Lossless Acceleration for LLM Agents — Li, Ye, Choubey, Zhang, Wu · 2026-07-14 | 투기 실행기에 온라인 기억 3종을 붙여 **환경 유휴시간에 zero added wall-clock cost로** 갱신. 정확도 19–39%↑, 관측 예측 최대 2.5×, 그리고 **"기억이 쌓일수록 이득이 계속 커진다"** — ReasoningBank의 단조 악화(49.7→44.4)와 정면 충돌 | 검색 3 |
| 11 | **2606.19172** | User as Engram: Internalizing Per-User Memory as Local Parametric Edits — Bojie Li · 2026-06-17 | Θ-경로 per-user 쓰기를 hash-keyed 테이블 행으로 국소화해 per-user LoRA 대비 **약 33,000× 작은 발자국**, 사용자 간 무손실 합성, 그리고 **~100 fact에서 검색 파이프라인을 추월**하는 교차점을 준다. ch26·ch27의 E vs Θ 교차점 논증에 **유일하게 수치가 있는 사례**. 단독저자·자체 벤치라는 점은 caveat로 필수 | 검색 6 |
| 12 | **2601.01885** | Agentic Memory (AgeMem): Learning Unified Long-Term and Short-Term Memory Management — Yu, Yao, Xie, Tan, Feng, Li 외 · 2026-01-05 (v3 07-23) | 저장·검색·갱신·요약·폐기를 **도구 행동으로 노출하고 3단계 RL(step-wise GRPO)로 학습**한다. corpus의 E-경로 전부가 휴리스틱/프롬프트 `wr`인데, **학습된 쓰기 정책**은 §2.2 분류표에 없는 기제 계열이다 | 서베이 `2603.07670` 참고문헌 |
| 13 | **2509.10852** | PREMem: Pre-Storage Reasoning for Episodic Memory — Kim, Lee, Kim, Kim, Cho · 2025-09-13 | 초록이 조작적 정의를 **글자 그대로** 쓴다: "shifts complex reasoning processes from inference to memory construction". Letta STC와 같은 것을 하면서 다른 커뮤니티에 있다 — ch11·ch23의 **침묵 지도에 새 좌표**를 찍는다 | 서베이 `2602.19320` 참고문헌 |
| 14 | **2605.23668** | OnePred: Next-Query Prediction via Recursive Intent Memory — Chen, Zhang, Song, Kang, Yang 외 · 2026-05-22 (개정 2026-08-02) | 턴 사이에 **재귀 갱신되는 intent memory만**을 문맥으로 유지하고, "무엇을 예측할지 → 무엇을 압축할지"를 2단계 RL로 학습. per-turn 토큰 **22× 절감**. 예측(anticipation)을 상태 갱신으로 정식화한 사례 + NQP-Bench | 검색 4 |

### 2.3 F3 — 제약·반증 (9편)

| # | arXiv | 제목 / 저자 / 날짜 | 왜 판정을 바꾸는가 | 발견 채널 |
|---|---|---|---|---|
| 15 | **2606.04703** | Rethinking Continual Experience Internalization for Self-Evolving LLM Agents — Chen, Yang, Fan, Nie, Sun, Lin 외 · 2026-06-03 | 다회차 경험 내재화에서 **누적 개선이 아니라 진행성 능력 붕괴**가 일어난다고 보고하고, 원인을 (경험 입도 / 주입 패턴 / 내재화 레짐) 셋으로 분해한다. off-policy context-distillation이 on-policy보다 안정적이라는 결론은 SEAL·`LM Need Sleep` 계열의 $\mathcal{R}$ 생성 전략을 **직접 반박 가능**하게 만든다. X4의 사전 연구 | 검색 7 |
| 16 | **2608.01679** | When Memory Becomes Authority: Benchmarking Authority Collapse at the Memory Consolidation Boundary — Zhan, Zhang, Guo, Zhao, Liu · 2026-08-03 | 통합(consolidation)이 주장은 보존하면서 **출처의 권한 제약을 지운다**. 7 consolidator × 7 backbone = **49 구성 중 48에서 붕괴**, 무권한 행동률 평균 50.3%. E-경로 `wr`의 실패 양식으로 본서에 **전혀 없는 축**. corpus 동결 직후 발표 | 검색 8 |
| 17 | **2601.11042** | Spectral Characterization and Mitigation of Sequential Knowledge Editing Collapse (REVIVE) — Zhang, Zhang, Ye, Cheng, Zhou 외 · 2026-01-16 (v2 05-09) | 순차 편집 붕괴를 **사전학습 가중치의 지배적 특이방향 훼손**으로 기제화하고 최대 20,000 편집까지 밀어본다. ch19·ch22가 ROME/MEMIT의 상한을 "실용 아님"이라는 저자 진술로만 처리하는 자리에 **기제 설명과 스케일 수치**를 넣는다 | 검색 5 |
| 18 | **2607.20433** | Moir: Let the Model Direct Its Own Story for Robust Cross-Domain Knowledge Editing — Kwon, Kim, Kim, Cha · 게재일 표기 2026-05-10 (ID는 2607대 — **날짜 불일치, unverified**) | **비대칭 열화**: 편집이 수학·코드 추론을 무너뜨리는 동안 백과사전적 회상은 멀쩡하다. Qwen3-8B에 AlphaEdit 20,000 배치편집 후 GSM8K가 79.9%(Moir) vs 10.9%(Wikipedia baseline). Θ-경로가 **무엇을 파괴하는지**를 과제별로 분해한 첫 수치 | 검색 5 |
| 19 | **2605.11836** | More Edits, More Stable: Understanding the Lifelong Normalization in Sequential Model Editing — Ma, Chen, Liu, Xu, Zheng, Xu 외 · 2026-05-12 (v2 07-21) | LN 제거 시 즉시 붕괴 + **"초기 편집이 이후 편집의 성공을 촉진한다"는 반직관적 양의 누적효과**를 보고한다. 사실이면 $\rho$는 단조 감소가 아니다 — ch29·ch30의 "반복 갱신은 열화한다"는 전제를 **뒤집을 수 있는** 유일한 후보 | 검색 5 |
| 20 | **2606.04315** | Exploring Cross-Scenario Generality of Agentic Memory Systems: Diagnostics and a Strong Baseline — Chen, Gu, Yin, Long, Zeng, Liu 외 · 2026-06-03 | 8개 기억 시스템을 5개 시나리오에서 재평가한 결과 **평범한 에이전트 하네스(평문 파일 자율 관리)가 교차과제 1위**. 본서의 "1위가 반복적으로 기억 없음"을 "1위가 설계된 파이프라인이 아니라 **에이전트 자율 제어**"로 **정밀화**한다 — ch27 판정문의 형태가 바뀐다 | 검색 1·2 |
| 21 | **2602.19320** | Anatomy of Agentic Memory: Taxonomy and Empirical Analysis of Evaluation and System Limitations — Jiang, Li, Wei, Yang, Kishore, Zhao 외 · 2026-02-22 (v2 05-20) | 벤치 포화, 지표-효용 불일치, **judge 민감도**, backbone 의존 정확도, 유지보수의 지연·처리량 오버헤드를 **명명된 목록**으로 정리한다. ch15·ch22가 개별 논문마다 따로 지적하는 결함들의 상위 범주를 외부 근거로 제공 | 검색 2 |
| 22 | **2606.28438** | When AI Reviews Its Own Code: Recursive Self-Training Collapse in Code LLMs — Song, Cai, Zhao · 2026-06-26 | 자기 신호로 거르는 AI-self-gate가 **"rubber-stamp 레짐"**에 빠져 수락 점수는 오르고 정확도는 떨어진다. "안정적 재귀 학습은 **외생적 검증**을 요구한다"는 결론은 SEAL·SCM의 자기 검증 루프에 대한 직접 반증 축. ch06이 2404.01413(2024)에만 의존하는 상태를 갱신 | 검색 8 |
| 23 | **2605.30260** | How LoRA Remembers? A Parametric Memory Law for LLM Finetuning — Xu, Hong, Yu, Cui, Huang, Xue 외 · 2026-05-28 | LoRA를 **용량 프로브**로 써서 $\Delta L$–유효파라미터–시퀀스길이의 멱법칙을 세우고, 토큰 수준에서 $p>0.5$의 **결정론적 상전이**(verbatim recall 충분조건)를 보인다. ch09의 용량 상한 라인(2505.24832 / 2603.01097)에 **정량 법칙**을 추가 | 검색 6 |

### 2.4 F2 — 경계·회색지대 (2편)

| # | arXiv | 제목 / 저자 / 날짜 | 왜 판정을 바꾸는가 | 발견 채널 |
|---|---|---|---|---|
| 24 | **2505.16950** | Bottlenecked Transformers: Periodic KV Cache Consolidation for Generalised Reasoning — Oomerjee, Fountas, Bou-Ammar, Wang · 2025-05-22 (v4 2026-03-25) | 스스로를 **memory consolidation/reconsolidation**으로 명명하면서 바꾸는 것은 KV 캐시다 — Θ·W·E 어디도 아니다. 게다가 쓰기가 **생성 도중** 일어나 조건 (1)도 못 맞춘다. **판별식의 조건 (1)과 (3)을 동시에 시험**하는 가장 깨끗한 경계 사례. 서베이 `2607.25380` [51]로 이미 우리 시야 안에 있었으나 deep-read되지 않았다 | 서베이 `2607.25380` 참고문헌 [51] |
| 25 | **2605.22154** | IdleSpec: Exploiting Idle Time via Speculative Planning for LLM Agents — Choi, Park, Song, Dingliwal, Jayanthi, Shin 외 · 2026-05-21 | 유휴시간에 예산을 써서 계획 후보를 만들고 관측 도착 후 집계한다 — 조건 (1)(2)(4)는 만족하지만 **산출물이 지속 상태가 아니다**(에피소드 내 소멸). "sleep-time compute이라는 이름이 기제보다 넓다"는 A1 논지의 **에이전트 서빙판 사례**. GAIA/FRAMES +5.1%p, MLE-Bench +9.1%p | 검색 3 |

---

## 3. 버린 것과 이유

### 3.1 초록까지 읽고 뺀 것 (2군 — 필요하면 승격 가능)

| arXiv | 제목 | 뺀 이유 |
|---|---|---|
| 2606.05698 | Rethinking LoRA Memory Through the Lens of KV Cache Compression (2026-06-04) | **아깝다.** 문서 LoRA가 KV를 공격적으로 압축할 때만 13–21 ROUGE-L 회복 → Θ와 문맥의 교환율을 실측한 F2. 25편 상한에서 밀렸을 뿐, 승격 1순위 |
| 2602.16313 | MemoryArena (2026-02-18) | LoCoMo 포화 에이전트가 상호의존 다세션에서 무너진다는 결론이 #20·#21과 중복. 벤치 자체는 유용하나 판정을 **추가로** 바꾸지 않는다 |
| 2605.12493 | LongMemEval-V2 (2026-05-12) | accuracy-latency Pareto를 다루지만 대상이 web-agent 환경경험 내재화로 좁고, 비용 보고가 #1·#2보다 얕다 |
| 2601.02845 | TiMem (2026-01-06) | 판별식은 통과하나 E-경로 6번째 시스템일 뿐. recall 길이 52.2% 감소는 $C$의 대리지표로만 쓸 수 있어 ch15 각주 재료 |
| 2503.05683 | WikiBigEdit (2025-03) | 500K 실세계 편집으로 편집 vs 검색 vs continual-FT를 맞붙인 강한 F3. **2025년이고**, #17·#18·#19가 같은 축을 2026 수치로 덮는다 |
| 2603.11768 | Governing Evolving Memory in LLM Agents (SSGM) (2026-03) | semantic/procedural drift 분류는 유용하나 프레임워크 제안이고 계측이 얇다. #16이 같은 축을 통제 벤치로 덮는다 |
| 2604.19089 | LightEdit (2026-04-21) | 편집 **방법** 논문. 상한·열화를 다루지 않아 F3이 아니고, 질의 시점 편집이라 F1도 아니다 |
| 2603.27138 | ScoutAttention (2026-03-28) | "layer-ahead CPU pre-computation"은 요청 **내부** 파이프라이닝이다. 질의 전 예산이 아니라 F1·F2 어디에도 안 걸린다 |
| 2607.20433 외 편집 계열 다수 | — | Moir 1편만 남기고 나머지 knowledge-editing 방법론은 전부 제외(기제 개선이지 제약이 아님) |

### 3.2 범주로 잘라낸 것

- **아키텍처 계열(서베이 `2607.25380` 참고문헌 74항목 중 다수)** — Mamba-3, Gated DeltaNet-2, Kaczmarz/Kalman linear attention, log-linear attention, MoM, Hymba, Falcon-H1 등. 전부 (U-W) 안쪽의 읽기·상태전이 설계이고 **$B_s=0$**. 본서가 이미 `nested-learning`·`memory-caching`으로 이 범주의 처리를 확정했다(§2.2 B). 재방문 불필요.
- **장문맥 벤치·확장 기법** — RULER, ∞Bench, SCROLLS, L-Eval, LongRoPE, positional interpolation. 기억 갱신이 없다.
- **E-경로 "또 하나의 기억 시스템"** — 서베이 `2602.19320` 참고문헌에서만 40편 이상 확인(LightMem, SimpleMem, EverMemOS, Nemori, MIRIX, A-Mem, SGMem, HiMem, MemBox, Bi-Mem, LiCoMemory, PISA, Synapse …). **의도적으로 전량 배제한다.** 판별식은 통과하겠지만 §2.2 A 표에 행을 하나 더 넣을 뿐 A1 판정을 바꾸지 못한다. 예외로 올린 것은 (a) 학습된 쓰기 정책(#12), (b) 쓰기 시점 추론(#13), (c) 쓰기 경로 비용 계측(#3)뿐이다.
- **학습형 기억 정책 계열의 나머지** — Memory-R1(2508.19828), Mem-α(2509.25911), MemRL(2601.03192), MemSearcher(2511.02805), Memory-as-Action(2510.12635), MemSkill(2602.02474). #12 하나로 계열을 대표시켰다. 계열 자체가 corpus에 없다는 **사실**은 ch24·ch27에 기록할 가치가 있다.
- **일반 KV 압축·양자화·서빙** — PyramidKV, KIVI, GEAR, ShadowKV, Quest, SparQ, LMCache, MemServe, Sarathi 등. 이 워크로드(질의 전 상태 축적)의 계측이 아니라 장문맥 디코드 일반의 계측이다. #5·#6·#7·#8만 남긴 기준은 **세션·에이전트를 가로지르는 지속 상태를 다루는가**이다.

### 3.3 이 채널이 **못** 준 것 (정직한 공백 보고)

1. **$B_s$–품질 scaling 형태(X3)를 채워줄 논문은 나오지 않았다.** 유휴 예산을 축으로 품질 곡선을 그린 것은 여전히 Letta STC와 `Do LMs Need Sleep?`뿐이고, ProAct(#9)·IdleSpec(#25)도 예산 축 곡선을 내지 않는다. X3는 **자체 실험으로만** 채울 수 있다.
2. **W-경로의 신규 정면 사례는 0편이다.** 서베이 세 편의 참고문헌을 다 뒤져도 `2605.26099` 외에 "질의 전 오프라인으로 fast weight를 갱신"하는 논문이 없다. ch17의 "유일 사례" 서술은 **동결 이후에도 유지된다** — 이것은 억지로 채우지 않은 결과이자 그 자체로 보고할 발견이다.
3. **세 경로 상호인용의 개선 신호도 없다.** 새로 찾은 25편 중 Letta STC를 인용한 것으로 확인된 것은 없다(초록 수준 확인). ch11·ch27의 침묵 지도는 2026-08 시점에도 유효하다.

---

## 4. 후속 처리 제안

| 우선 | 대상 | 무엇을 할 것인가 |
|---|---|---|
| P0 | #1 2606.06448, #2 2606.24775 | **deep-read 정식 진행.** ch25(통일 비용 회계)를 이 둘의 실측과 대조하고, 어긋나면 X1의 등급 규칙에 따라 본문 단정문을 낮춘다 |
| P0 | #15 2606.04703, #19 2605.11836 | X4(반복 consolidation 열화 $\rho$) **설계 전에** 읽는다. 특히 #19의 양의 누적효과가 사실이면 X4의 가설 자체를 다시 세워야 한다 |
| P1 | #5 2603.04428, #11 2606.19172 | ch26의 "바이트 보고 0편" caveat를 **문장 단위로 재작성**. 두 편 다 자체 벤치·단독저자라는 약점이 있으므로 caveat와 함께 인용 |
| P1 | #9 2605.25971, #10 2607.12236 | §2.2 통과군 표에 행 추가 여부 판정. 추가되면 "통과 9 + 경계 1 + 불충족 19 = 29" 카운트 규약(§2.2)을 **전 장 일괄 갱신**해야 한다 — 조립 QA 항목 |
| P2 | #24 2505.16950, #25 2605.22154 | ch11 분기 장의 경계 사례 절에 추가. MemGPT 하나뿐이던 경계 사례가 셋이 되면 판별식 서술이 더 단단해진다 |
| P2 | #21 2602.19320, #20 2606.04315 | ch22·ch27이 개별 논문 결함을 나열하는 대신 **상위 범주 + 외부 근거**로 재구성할 수 있는지 검토 |

---

## 부록 A — 파싱 산출물 위치

```
<scratchpad>/gap/
  parse_refs.py  parse2.py  mine.py  fetch.py  fx.sh
  2603.07670.pdf/.txt   2602.19320.pdf/.txt   2607.02574.pdf/.txt
```

배제 대조는 세 스크립트 모두에 69 ID 상수로 하드코딩되어 있다. 후보 25편과 배제 목록의
교집합이 0건임을 프로그램으로 확인했다.
