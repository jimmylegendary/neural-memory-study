# gap-sweep — 커뮤니티·뉴스레터·제품 채널

**작성**: 2026-08-11
**채널**: Reddit(r/MachineLearning, r/LocalLLaMA), 뉴스레터(Turing Post, AI News/smol.ai, The Batch,
Ahead of AI), 제품 블로그·릴리스(Anthropic, OpenAI, Mem0, Letta, Zep, Samsung), 서드파티 분석 블로그
**배제 기준**: `corpus-exclusion.md`의 노트 32편 + vendored arXiv ID 69개
**판정 기준**: `S0-DESIGN.md` §2.3 판별식 / `PRE-RESEARCH.md` §2 — F1(판별식 통과) · F2(경계) ·
F3(제약·반증) · F4(비용·시스템)

---

## 1. 방문한 소스

### 1.1 접근 실패 (사실만 기록)

| 소스 | 시도 | 결과 |
|---|---|---|
| `www.reddit.com/r/MachineLearning/search/…` | WebFetch | **차단** — "unable to fetch from www.reddit.com" |
| `www.reddit.com/…/search.json` | WebFetch | 차단 (동일 호스트) |
| `old.reddit.com` | WebFetch | 차단 |
| `redlib.catsarch.com` (redlib 미러) | WebFetch | HTTP 403 |
| `openai.com/index/chatgpt-memory-dreaming/` | WebFetch | HTTP 403 — **2차 출처로 대체 확인** |
| `platform.claude.com/docs/en/docs/agents/dreaming` | WebFetch | HTTP 404 (docs.anthropic.com → 301 리다이렉트 후 소실) |
| X/Twitter | 미시도 | 지시대로 생략 |

> **Reddit은 이 세션에서 직접 읽지 못했다.** 대신 `site:reddit.com` 질의를 검색엔진에 걸어
> 우회했는데, 반환된 것은 Reddit 스레드가 아니라 **스레드가 인용하던 arXiv 논문들**이었다.
> 아래 후보 중 2607.00368·2607.05202·2605.09315·2602.03224는 그 경로로 나왔다. 채널 자체는
> 실패했으나 채널이 가리키던 대상은 회수했다.

### 1.2 실제로 읽은 소스

| 소스 | 날짜 | 무엇을 얻었나 |
|---|---|---|
| Turing Post **FOD#155 — Continual Learning in LLMs: Why AI Models Need Sleep** | 2026-06-08 | 논문 16편 목록. 2606.04703·2606.02437 회수. **OpenAI Dreaming** 존재 확인 |
| Turing Post — Agentic Memory: 10 Essential Resources | 2026 | 서베이 6편 + MemEvolve. 2602.19320 회수 |
| **Mem0 공식 블로그 — "Dream: Background memory consolidation for AI agents"** | 2026-08-05 (2026-08-11 갱신) | 제품 사양 원문. merge/supersede/synthesize, 주 1회 스케줄 |
| **Samsung 백서 — Optimizing KV Cache Offloading to CMM-D in a CXL Switch-based Memory Pool** (PDF 14pp 전문) | 2026-06 | 실측 TPS·대역폭·용량 절벽. F4 최대 수확 |
| Arize AI — "Two labs started dreaming, and they built two different architectures" | 2026-06-17 | Anthropic vs OpenAI dreaming 아키텍처 대조. 2605.12978 존재 단서 |
| Ken Huang (Substack) — "Why AI Agents Are Starting to Dream" | 2026-06-13 | dreaming 계보 서술. 정량치 없음(그 자체가 기록거리) |
| AI News (smol.ai) 이슈 목록 + 2026-06-05 이슈 | 2026-05~06 | δ-mem 단서. sleep 관련 커버리지 빈약 확인 |
| digitalapplied — AI Agent Memory 2026 | 2026-05 | 제품 릴리스 연표(Anthropic/Google/OpenAI), 벤더 보고 벤치 수치, Harvey 6× |
| aiworkflowlab / spheron — Mem0·Letta·Zep 운영 비교 | 2026 | **Letta 동기 core-memory 쓰기 150–400ms** 등 운영 지연 수치 |
| Ahead of AI (Raschka) 아카이브·노트 | 2026 | 이 주제 전용 호 **없음**. "2026 inference-time scaling / 2027 continual learning" 시대구분만 |
| The Batch (DeepLearning.AI) | 2026 | 이 주제 전용 기사 **찾지 못함**. Oracle 공동 Agent Memory 코스만 |
| arXiv abs 페이지 14편 | — | 초록 직접 확인 (아래 표의 verified 항목) |

---

## 2. 후보

**18건.** 전부 초록 또는 원문을 실제로 읽고 판정했다. arXiv ID는 전부 실제 abs 페이지 또는
공식 배포처에서 확인했다(추정 ID 없음).

### 2.1 F3 — 제약·반증 (이 채널의 최대 수확)

| # | ID | 제목 / 저자 / 날짜 | F | 왜 판정을 바꾸는가 | 출처 채널 |
|---|---|---|---|---|---|
| 1 | **2605.12978** | *Useful Memories Become Faulty When Continuously Updated by LLMs* — Dylan Zhang 외 — 2026-05 | **F3** | GPT-5.4가 기억 없이 100% 푸는 ARC-AGI 19문항을 **정답이 주어진 상태로** 통합 루프에 흘리면 **54%로 붕괴**하고, 기억 효용이 상승 후 no-memory baseline **아래로** 내려간다 — 책이 §5에서 "품질 1위가 반복적으로 기억 없음"이라 **관찰**만 한 현상에 기제와 수치를 붙인다 | Arize 블로그 → 검색 |
| 2 | **2607.00368** | *Beyond Perplexity: A Behavioral Evaluation Framework for Deployment-Memory Claims in LLM Test-Time Training* — Song, Chen, Kong, Xie, Dong, Chen, Zhang — 2026-07-01 | **F3** | Qwen3에 nonce fact를 1-step LoRA로 써넣으면 loss는 내려가는데 **자유생성 recall이 정확히 0**이다. Θ-경로의 "썼다"를 perplexity로 판정하는 corpus 전체 관행을 무효화한다 — ch22 반증 축의 두 번째 기둥 | site:reddit 우회 검색 |
| 3 | **2606.04703** | *Rethinking Continual Experience Internalization for Self-Evolving LLM Agents* — Jingwen Chen 외 — 2026-06-03 | **F3** | 다회 반복 경험학습에서 기존 방법이 **누적 개선이 아니라 progressive capability collapse**를 겪는다고 보고하고 off-policy context-distillation을 처방 — 책의 미실행 실험 **X4($\rho$)**가 묻는 바로 그 질문에 이미 답이 있다 | Turing Post FOD#155 |
| 4 | **2601.11042** | *Spectral Characterization and Mitigation of Sequential Knowledge Editing Collapse* — Chi Zhang 외 (ACL 2026 Main) — 2026-01-16 | **F3** | 붕괴 원인을 **pretrained 가중치의 dominant singular direction 파괴**로 특정하고 **최대 20,000 edit**까지 검증 — ch19가 MEMIT Specificity 26.6 < 미편집 27.0 하나로 지탱하던 Θ-경로 쓰기 상한에 기제와 자릿수를 준다 | 검색(편집 붕괴 축) |
| 5 | **2605.11836** | *More Edits, More Stable: Understanding the Lifelong Normalization in Sequential Model Editing* (ICML 2026) — 2026-05 | **F3** | **반대 방향의 반증**: LN + ridge 정규화 시 갱신이 asymptotic orthogonality·bounded norm을 가져 "이른 편집이 이후 편집 성공을 돕는" 양의 누적효과가 난다. #4와 정면 충돌 — ch22를 "붕괴한다"가 아니라 **"어떤 조건에서 붕괴하는가"**로 다시 쓰게 만든다 | 검색(편집 붕괴 축) |
| 6 | **2605.09315** | *Do Self-Evolving Agents Forget? Capability Degradation and Preservation in Lifelong LLM Agent Adaptation* — 2026-05 | **F3** | workflow·skill·model·memory 네 진화 축 전부에서 capability erosion을 측정하고 CPE로 단순과제 유지성능 **41.8%→52.8%** 회복. $\rho$를 층별로 분해한 첫 사례 | site:reddit 우회 검색 |
| 7 | **2607.05202** | *EvoAgentBench: Benchmarking Agent Self-Evolution via Ability Transfer* — Gao 외 — 2026-07-06 | **F3** | 수작업 curated ability는 모델 계열을 넘어 전이되지만 **"어떤 자동 방법도 모든 설정에서 양의 이득을 유지하지 못한다"** — ReasoningBank 계열 자기진화 기억의 일반화 주장에 대한 통제된 반례 | site:reddit 우회 검색 |
| 8 | **2602.19320** | *Anatomy of Agentic Memory: Taxonomy and Empirical Analysis of Evaluation and System Limitations* — Jiang 외 — 2026-02-22 (v2 2026-05-20) | **F3+F4** | benchmark saturation, judge sensitivity, backbone 의존성, **memory maintenance가 유발하는 지연·처리량 오버헤드**를 한 문서에서 다룬다. 책이 "아무도 비용을 보고하지 않는다"고 쓰기 전에 반드시 대조해야 할 문헌 | Turing Post 큐레이션 |

### 2.2 F1 — 판별식 통과 후보 (전부 **제품**, 논문 아님)

> 이 채널의 고유 가치가 여기 있다. **2026년 상반기에 세 벤더가 sleep-time compute의 E-경로를
> 각각 출시했는데 corpus에는 한 건도 없다.** 판별식 네 조건(질의 전 · 예산>0 · $E$ 상태 변경 ·
> 이후 질의에 사용)을 문면 그대로 만족한다.

| # | ID | 제품 / 날짜 | F | 왜 판정을 바꾸는가 | 출처 채널 |
|---|---|---|---|---|---|
| 9 | Anthropic **Dreaming** (Claude Managed Agents) | 2026-05-06 (Code with Claude) | **F1+F4** | 세션 **사이**에 비동기 실행, 기존 memory store + **최대 100개 과거 세션** 전사를 읽어 **입력 store를 건드리지 않고 별도의 재조직된 store를 생성**(가역). **표준 API 토큰 요율로 과금** = $B_s$가 실제 청구 가능한 양으로 노출된 첫 사례. 지연 "minutes to tens of minutes". Harvey 과제완수 6× (단일 배포·벤더 보고) | 검색 + Arize + digitalapplied |
| 10 | OpenAI **Dreaming V3** (ChatGPT memory) | 2026-06-04 | **F1+F4** | 유휴 시간 백그라운드 통합이 다수 대화를 동시 합성해 **in-place로 canonical state를 덮어쓴다**(비가역). "you're going to Singapore in July" → "you went to Singapore in July 2026" 식 **시간적 무효화**를 자동 수행 — Zep이 knowledge-update에서 **퇴행**했던(76.9→74.4) 바로 그 연산을 소비자 규모로 돌린다. Plus/Pro 게이팅 | Turing Post FOD#155 → 검색 |
| 11 | Mem0 **Dream** | 2026-08-05 (2026-08-11 갱신) | **F1** | **corpus 동결 이후**. 논문 Mem0(2504.19413)의 `wr`은 메시지쌍마다 온라인 {ADD,UPDATE,DELETE,NOOP}였는데, 제품은 **요청 경로 밖에서 주 1회 스케줄로 도는 synthesis**와 supersede 포인터 기반 lifecycle state를 추가했다 — **ch15의 Mem0 서술이 더 이상 제품과 일치하지 않는다**. 정량치는 여전히 0건("median active project가 수백 개의 중복·모순 기억을 진다"는 관찰뿐) | smol.ai 검색 → 공식 블로그 |
| 12 | **2604.23878** | *ZenBrain: A Neuroscience-Inspired 7-Layer Memory Architecture* — Alexander Bering — 2026-04-26 (v3 **2026-08-09**) | **F1+F3+F4** | (a) Simulation-Selection **sleep loop**로 오프라인 통합 → F1. (b) **cooperative masking**: 중간 부하에서 15개 중 14개 ablation이 무해해 보이나 decay 0.25/day·60일로 올리면 9개가 개별적으로 치명(ΔQ −93.7%, Wilcoxon 10 seed) → **corpus의 모든 단일조건 ablation 해석(NL Table 6, LM Need Sleep, Memory Caching Table 5)을 방법론적으로 흔든다**. (c) full-context oracle 정확도의 91.3%를 **질의당 토큰비용 1/106**로 → 책이 못 채운 $C_{avg}$ 비 | 검색(dreaming 축) |

### 2.3 F4 — 비용·시스템 (책의 최대 공백)

| # | ID | 제목 / 날짜 | F | 왜 판정을 바꾸는가 | 출처 채널 |
|---|---|---|---|---|---|
| 13 | **Samsung 백서** (arXiv 아님) `download.semiconductor.samsung.com/resources/white-paper/Optimizing_KV_Cache_Offloading_to_CMM-D_in_a_CXL_Switch-based_Memory_Pool.pdf` | *Optimizing KV Cache Offloading to CMM-D in a CXL Switch-based Memory Pool* — 2026-06 | **F4** | **실기 실측**: 8×RTX PRO 6000 Blackwell + H3P Falcon C5022 CXL 스위치 + MD220 256GB×4 = **1TB 풀**, vLLM 0.13.0 + LMCache 0.3.14. TP=1에서 DRAM(512GB)은 KV 470GB까지 ~380K TPS를 유지하다 512GB를 넘는 순간 재계산으로 **203K→108K→64,961 TPS로 붕괴**, CXL 풀은 700GB에서도 **363,281 TPS 유지 = 5.6×**. TP=8은 DRAM 대비 **92.3%**(243,439 vs 263,551), CXL 대역폭 **~47GB/s**(Intel PCM 실측). ch29의 웜 계층 논증(X2)에 붙일 수 있는 **corpus 최초의 바이트·TPS 동시 실측** | 검색(CXL 축) → PDF 전문 |
| 14 | **2606.02437** | *On the Scaling of PEFT* — 2026-06 | **F4** | adapter를 **"persistent local state"**로 규정하고 Scale Up/Down/**Out**으로 나눈 뒤, 다수 adapter의 identity·provenance·serving을 다루는 MinT 인프라를 제시 — 책의 A2(per-user 가중치 delta가 상태량을 지배)를 **서빙 인프라 문헌으로** 잇는 고리. 초록에 수치 법칙은 없음 | Turing Post FOD#155 |
| 15 | **2604.08426** | *KV Cache Offloading for Context-Intensive Tasks* — Bocharnikov 외 — 2026-04-09 | **F4+F2** | Text2JSON 벤치에서 **최신 KV 오프로딩이 Llama 3·Qwen 3 양쪽에서 유의하게 열화**한다고 보고(원인: key의 low-rank projection, 신뢰할 수 없는 landmark) — #13의 낙관치에 대한 워크로드 의존 반례. 초록에 수치 없음(본문 확인 필요) | 검색(CXL 축) |
| 16 | (비논문) Letta·Mem0·Zep 운영 지연 | 2026 | **F4** | **Letta의 동기 core-memory 갱신이 턴당 150–400ms를 더한다**; Mem0/Zep은 비동기 반환. 쓰기 경로가 wake 지연에 들어가는지 여부가 시스템마다 갈린다는 1차 근사 — $L_w$ 칸이 corpus 전체에서 "논문에 없음"인 상태를 벤더 외 출처로 부분적으로 메운다. **2차 출처(운영 블로그)이므로 본문 수치로는 부적격, [본서 산술] 아닌 참고치로만** | aiworkflowlab / spheron |

### 2.4 F2 — 경계·회색지대

| # | ID | 제목 / 날짜 | F | 왜 판정을 바꾸는가 | 출처 채널 |
|---|---|---|---|---|---|
| 17 | **2605.12357** | *δ-mem: Efficient Online Memory for Large Language Models* — Lei 외 (declare-lab) — 2026-05-12 | **F2** | frozen backbone 옆에 **8×8 online state**를 두고 delta-rule로 갱신, attention에 low-rank correction을 준다. 시계가 **wake**이므로 조건 (1) 탈락 — Generative Adapter(ch16 경첩)와 **정확히 같은 자리에서 같은 이유로** 떨어지는 두 번째 사례이자, $W$층 용량을 8×8이라는 극단적 소수로 못박은 값(ch18 천장 논증 재료) | smol.ai → 검색 |
| 18 | **2602.03224** | *TAME: A Trustworthy Test-Time Evolution of Agent Memory with Systematic Benchmarking* — 2026-02 | **F2+F3** | executor-evaluator 루프로 기억을 선택적 강화·재사용·확장. **연속 갱신이 유발하는 신뢰성 위험(memory misevolution)을 명시적으로 규제**하려는 첫 시도이자, 갱신 시점이 질의 전인지 중인지 초록만으로 판정 불가 — 판별식 조건 (1)의 해상도를 시험하는 사례. GPT-5.2 AIME에서 최강 기존법 대비 +14.6%p | site:reddit 우회 검색 |

---

## 3. 버린 것과 이유

| 버린 것 | 이유 |
|---|---|
| **2601.04362** *Phasor Agents: Oscillatory Graphs with Three-Factor Plasticity and Sleep-Staged Learning* (2026-01-07) | 판별식은 문면상 통과한다(wake tagging ↔ offline consolidation 분리, deep-sleep gated capture가 가중치 커밋). 그러나 **LLM이 아니라 Stuart-Landau 진동자 그래프**다. 저자 스스로 생물 집단을 모델링하지 않는다고 명시. 세 층 $\Theta/W/E$로 환원해도 서빙·decode 함의가 없어 **판정을 못 바꾼다** |
| **2512.18746** MemEvolve (2025-12-21) | 기억 아키텍처를 메타진화시키는 흥미로운 축이나, 초록이 **"환경 상호작용 안에서 on the fly"** 진화라고 명시 → 조건 (1) 불충족. 2026 이전이고 F1–F4 강도 부족 |
| **2601.02845** TiMem (ACL 2026 Findings, 2026-01/04) | E-경로 계층 통합. LoCoMo 75.30%, **recalled memory length 52.20% 감소**는 rate–distortion(2607.08032)이 이미 점유한 축의 반복이고, 감소치가 **저장 바이트가 아니라 읽기 길이**여서 ch26의 $C$ 공백을 못 메운다. 오프라인 여부도 초록에서 불확정. 경계선에서 탈락 |
| 서베이 8편 — 2602.06052, 2604.16548, 2602.05665, 2601.09113, 2512.13564, 2606.24937, 2603.12658, 2501.07278 | 전부 taxonomy·roadmap. corpus는 이미 2607.25380(Memory for LLMs)을 보유. **새 제약도 새 실측도 없다** |
| 2606.03458 KVarN (KV-cache 양자화) | 읽기식 최적화. 어떤 층의 상태도 오프라인에 안 바꾼다 |
| 2606.06492 Code2LoRA, 2605.18071 KVDrive, 2605.22850 ObjectCache, 2511.00321 CXL PNM | 제목·URL만 확인, **초록 미확인 → unverified**. F4/F2 가능성은 있으나 규정상 후보로 올리지 않는다. 후속 스윕에서 초록 확인 권장 |
| Harvey "6× 과제완수", Mem0 41K stars / 14M DL, Letta LoCoMo 83.2%, Zep LongMemEval 63.8% | 전부 **벤더 자기보고, 단일 배포, 통제군 없음**. 책의 정직성 caveat 기준(§4)으로는 인용 불가. 다만 **"제품은 출시됐는데 통제된 수치가 없다"는 사실 자체**가 ch27 판정 재료 |
| Ken Huang / PromptHub / 각종 요약 블로그 | 1차 정보 0. Letta·MemGPT 재서술 |
| Ahead of AI, The Batch | 이 주제 전용 호를 **찾지 못했다**. Raschka의 시대구분("2026 inference-time scaling, 2027 continual learning?")은 인용 가치 없음 |

---

## 4. 이 채널이 실제로 밝힌 것 (요약 3항)

1. **판별식을 통과하는 시스템이 2026년 상반기에 최소 셋 출시됐고, corpus에는 하나도 없다.**
   Anthropic Dreaming(2026-05-06), OpenAI Dreaming V3(2026-06-04), Mem0 Dream(2026-08-05).
   더구나 셋은 `wr` 설계가 서로 다르다 — Anthropic은 **비파괴·병렬 산출**(입력 store 불변),
   OpenAI는 **in-place 덮어쓰기**, Mem0는 **supersede 포인터 + lifecycle state**. 책이 ch14에서
   "Letta의 `wr`은 파괴적 덮어쓰기"라고 단정한 축이 **제품 층위에서 3분岐했다**.

2. **세 제품 모두 정량 결과를 내지 않았다.** 지연은 "minutes to tens of minutes", 효과는 Harvey
   6× 단일 배포. §4의 caveat "corpus 전체에서 상태 용량을 바이트로 보고한 논문이 0편"은
   **제품 문서까지 포함해도 여전히 참**이다 — 이 확인 자체가 ch27·ch30의 근거가 된다.

3. **오프라인 통합이 성능을 깎는다는 정량 반증이 나왔다(2605.12978).** 정답을 주고도
   100%→54%. 벤더 둘이 같은 기제를 소비자 규모로 출시한 그 달에 나온 결과다. 책의 척추 논지
   A1은 "경로마다 다르다"인데, 이 논문은 **E-경로 안에서도 통합 유무로 갈린다**고 말한다.
   ch27의 E-경로 판정을 쓰기 전에 반드시 흡수해야 한다.
