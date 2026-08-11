# Gap sweep — 채널: GitHub 큐레이션 목록 (awesome-*)

**작업일**: 2026-08-11
**배제 기준**: `scratchpad/corpus-exclusion.md` (deep-read 32편 + vendored arXiv ID 69개)
**판정 기준**: `S0-DESIGN.md` §2.3 판별식 4조건, `PRE-RESEARCH.md` §2·§5
**원칙**: 초록을 실제로 읽고 판단했다. 읽지 못한 것은 후보에 넣지 않고 §4 "미검증 단서"로 내렸다.

---

## 1. 방문한 소스

### 1.1 GitHub 검색 (최근 갱신순 정렬)

| 검색 URL | 수확 |
|---|---|
| `search?q=awesome+llm+memory&s=updated` | 목록 10개 식별 |
| `search?q=awesome+KV+cache&s=updated` | 목록 10개 식별 |
| `search?q=awesome+test-time+training&s=updated` | 목록 6개 — **전부 vision/robot TTT. LLM 없음** |
| `search?q=awesome+self-evolving+agent&s=updated` | 목록 10개 식별 |
| `search?q=awesome+proactive+agent+LLM&s=updated` | 목록 3개 — **이 채널의 최대 수확처** |
| `search?q=awesome+LLM+inference+serving&s=updated` | 목록 4개 (2개는 2025년 정지) |
| `search?q=awesome+knowledge+editing+LLM&s=updated` | **1개, 무관** |
| `search?q=awesome+model+editing+continual+learning+LLM&s=updated` | **0건** |
| `search?q=model+editing+paper+list&s=updated` | **2개, 전부 이미지/3D 편집. 무관** |

### 1.2 README를 실제로 읽은 목록 (14개)

| 저장소 | 갱신 | 이 작업에서의 값 |
|---|---|---|
| `TeleAI-UAGI/Awesome-Agent-Memory` (575★) | 7시간 전 | 후보 4건 (RecMem, AutoMem, 에이전트네이티브, StreamMemBench) |
| `Snseam/awesome-agent-memory` (989편) | 어제 | **`docs/cost-savings-landscape.md`가 F4 전용 목록** — MEMAUDIT·BudgetMem·DimMem·A2RAG 발굴 |
| `selfimproving-agent/Awesome-Self-Improving-Agents` (338★) | 2일 전 | **최대 단일 수확**. 200+ 항목, L1 Reliability 절이 F3 전용 |
| `wkqdzkd/Awesome-Reliable-Self-Evolving-Agents` | 1시간 전 | F3 3건 (망각·붕괴·drift) |
| `yxf203/Awesome-Efficient-Agents` (294★) | 어제 | F2/F4 3건 (KV-Skill, Zero-Mem, ACM) |
| `jjiantong/Awesome-KV-Cache-Optimization` (374★) | 4일 전 | F2/F4 메모리 계층 5건 |
| `LowEntropyAI/awesome-proactive-agent` (9★) | 어제 | **ProAct(idle-time compute) 유일 출처** |
| `tao-hpu/awesome-proactive-agents` | 26일 전 | 중복. 신규 0 |
| `EvaxHe/Awesome-LLM-Agent-Experience-Lifecycle` | 23시간 전 | S4 Consolidation 25건 — 전부 프롬프트공간 E-경로 |
| `Shichun-Liu/Agent-Memory-Paper-List` | — | 신규 0 (E-경로 일변도) |
| `TsinghuaC3I/Awesome-Memory-for-Agents` | — | 신규 0 |
| `RUC-NLPIR/Awesome-Long-Horizon-Agents` | — | 신규 0 |
| `VoltAgent/awesome-ai-agent-papers` (1,700★) | 3일 전 | **별 수 1위인데 2026-05 이후 항목이 하나도 없음**. 죽은 목록 |
| `EvoMap/awesome-agent-evolution` (184★) | 21시간 전 | 신규 0 |
| `sad-and-bad1231/awesome-LLM-inference-systems` | 15시간 전 | arXiv ID를 안 적어 대조 불가 |

### 1.3 ID 확인용 WebSearch (4회)

목록에 제목만 있거나 ID가 틀린 항목을 확인하려고 썼다. 이 과정에서 **어느 목록에도 없던 3편**이
나왔다(2606.06448, 2606.29914, 2602.19320). 출처를 후보표에 정직하게 표기했다.

### 1.4 arXiv 초록 직접 확인 (22건)

후보로 올린 것은 전부 `arxiv.org/abs/` 원문 초록을 읽었다. 1건(2606.00830)은 목록의 ID가
**틀렸고**(입자물리 논문) 재검색으로 2604.00830임을 확인했다 — 큐레이션 목록의 ID를 믿으면 안 된다.

---

## 2. 후보 (19편) — 전부 초록 확인 완료

### Tier 1 — 판정을 실제로 바꿀 수 있는 것 (5편)

| # | arXiv | 제목 · 저자 · 날짜 | F | 왜 판정을 바꾸는가 | 출처 |
|---|---|---|---|---|---|
| 1 | **2606.06448** | *Agent Memory: Characterization and System Implications of Stateful Long-Horizon Workloads* — Omri, Gan, Broveak, Geens, He, Pentland, Verhelst, Weissman, Tambe · 2026-06-04 | **F4** | 이 책이 "공백"이라 선언한 바로 그것 — agent memory의 **최초 시스템 캐릭터리제이션**. construction/retrieval/generation 3단계에 비용을 귀속시키는 phase-aware 프로파일링 하네스로 10개 시스템×2 벤치 측정, 그리고 **"amortization via query volume"을 권고 항목으로 도출**한다. 즉 ch25의 상각식 (A)와 "실서빙 $N_q$ 분포를 보고한 논문이 corpus에 0편"이라는 caveat가 **이 논문 하나로 무효화될 수 있다**. construction이 user-facing decode path 밖에 있어 "background indexing에 가깝다"는 관찰은 sleep/wake 분리의 시스템적 실증이다. | WebSearch(ID추적) |
| 2 | **2605.12978** | *Useful Memories Become Faulty When Continuously Updated by LLMs* — D. Zhang, Lin, Wu, Sun, Li, Li, Peng · 2026-05-13 | **F3** | (U-E)의 반복 적용이 **효용을 올렸다가 내리고 결국 no-memory baseline 아래로 떨어진다**는 것을 통제 환경에서 보인다. GPT-5.4가 **ground-truth 해답으로 통합했는데도** 이전에 기억 없이 풀었던 ARC-AGI 문제의 **54%를 실패**한다. Retain/Delete/Consolidate를 노출한 ARC-AGI Stream에서 에이전트는 raw episode를 보존하는 쪽을 택하고 강제 통합 대비 **정확도가 2배**. ch27의 "품질 1위가 반복적으로 기억 없음"을 일화에서 **기제 규명으로 승격**시키고, 아무도 보고하지 않은 $\rho$(열화율)를 E-경로에 대해 처음 준다. | selfimproving-agent (L2 Memory) |
| 3 | **2605.02199** | *MEMAUDIT: An Exact Package-Oracle Evaluation Protocol for Budgeted Long-Term LLM Memory Writing* — Bhargava, Barrento · 2026-05-04 | **F4**(+F3) | 초록 첫 문장이 판별식 조건 (1)(2)(3)(4)를 그대로 적는다: "**질의를 알기 전에** 경험 스트림을 지속 기억으로 압축해야 한다". 그 위에 **저장 예산(storage cost)을 명시 제약으로 건** 쓰기-선택 최적화를 세우고 branch-and-bound + MILP로 **인증된 최적해**를 낸다. Mem0·A-Mem·Letta 스토어를 실제로 export해 채점한다. ch26의 "corpus 전체에서 상태 용량을 바이트로 보고한 논문이 0편"과 ch27의 write/retrieval 교락 문제를 **동시에** 겨눈다. | Snseam `cost-savings-landscape.md` |
| 4 | **2606.29914** | *MemDelta: Controlled Baselines and Hidden Confounds in Agent Memory Evaluation* — Kuan Wang · 2026-06-29 | **F3** | 한 번에 한 성분만 바꾸는 통제 프로토콜. **agent self-memory 42% < 기본 retrieval 47%**; embedding 모델만 바꿔도 +6.2pp(p=0.004)로 **결론이 뒤집히고**; Mem0는 cloud RAG와 동률인데 **비용이 50배**. 결론 권고가 "**아키텍처에 이득을 귀속시키기 전에 write-path 비용을 보고하라**" — 이 책의 A1 척추 논지와 같은 말을 정량으로 한다. ch15·ch27의 caveat 다수가 이 논문으로 **외부 인용 근거**를 얻는다. | WebSearch(ID추적) |
| 5 | **2605.25971** | *Anticipate and Learn: Unleashing Idle-Time Compute in Proactive Agents* — Hu, Lyu, Kong, W. Liu, Lin, Guo, Xu, Wang, Zhang, Yu · 2026-05-25 | **F1** | "상호작용 사이의 **idle time이 낭비되고 있다**"를 문제로 세우고, ProAct가 대화이력+지속기억을 분석해 다음 필요를 예측하고 **사용자가 질의를 시작하기 전에 증거를 준비**한다. 네 조건 전부 통과 → **통과군이 9편에서 늘어난다**. 더 중요한 것: Letta STC와 **독립적으로 같은 개념을 재발명**한 두 번째 사례이므로 §3.1 침묵 지도에 새 행이 생긴다. 턴 -14.8%, 사용자 노력 -11.7%, 환각 -28.1%. | LowEntropyAI/awesome-proactive-agent |

### Tier 2 — 층 구분과 비용 회계를 시험하는 것 (7편)

| # | arXiv | 제목 · 저자 · 날짜 | F | 왜 | 출처 |
|---|---|---|---|---|---|
| 6 | **2606.04536** | *Scaling Self-Evolving Agents via Parametric Memory* (TMEM) — Ren, Luo, Yang, Zhu, Huang, Wu, Chou, Ye, Liang, Li, Peng · 2026-06-03 | **F1/F2** | 기존 memory agent가 "경험을 **오로지 prompt 공간에만** 저장하고 파라미터를 얼린 채 둔다"를 문제로 삼고, 추출 행동이 만든 supervision을 **fast LoRA 가중치에 온라인으로 흡수**시킨다. Θ 모양 상태를 에이전트 루프 안에서 바꾸는 첫 사례 — ch16(Generative Adapter, 경첩 장)의 후계이며, **에피소드당 LoRA delta**는 ch26·A2가 묻는 per-user 상태량 질문 그 자체다. 단, 갱신이 에피소드 **내부**라 조건 (1)은 경계. | selfimproving-agent §2.2.3 |
| 7 | **2607.03441** | *No Time Like the Present: Agentic Test-Time Training for LLM Agents* — Y. Wang, Hao, Shi, Yuan, Sun · 2026-07-03 | **F2/F3/F4** | 다중턴 에피소드에 걸친 **연속 TTT**: 각 파라미터 갱신이 다음 학습 텍스트를 생성하는 정책을 바꾼다 → 정체 시 **오류가 복리로 누적**. update-text 반복(n-gram)을 그 분기점으로 지목하고 토큰 단위 재가중으로 막는다. **vLLM LoRA runtime API로 동시 서빙 구현·오버헤드 측정** — corpus 인접 문헌 중 가중치 갱신을 루프에 넣고 서빙 비용을 실측한 유일한 사례. ch29 재료. | WebSearch(ID추적) |
| 8 | **2606.04703** | *Rethinking Continual Experience Internalization for Self-Evolving LLM Agents* — J. Chen, Yang, Fan, Nie, Sun, Zheng, Hu, Pan, Zeng, Lin · 2026-06-03 | **F3** | 기존 방법을 **여러 사이클** 돌리면 "복리 개선이 아니라 **점진적 능력 붕괴**"가 온다. 그리고 "**자기 자신의 결함 있는 시도**로 배우는 것보다 교사 시연으로 배우는 것이 안정적"이라는 대조를 준다 — SEAL(ch20)의 self-edit와 ch06 model collapse를 **하나의 실험에서 잇는** 결과다. instance-level 대비 principle-level 경험의 내구성 비교는 (U-Θ)의 $\mathcal{R}_k$ 구성 문제에 직접 답한다. | wkqdzkd L1 Reliability |
| 9 | **2606.24775** | *Are We Ready For An Agent-Native Memory System?* — W. Zhou, X. Zhou, Han, Xu, G. Li, Z. Li, Xiong, Wu · 2026-06-23 | **F3/F4** | 기억 시스템을 **데이터관리 관점**에서 4모듈(표현·저장 / 추출 / 검색·라우팅 / 유지보수)로 분해해 **12개 시스템 + 기준선 2개 × 5 워크로드 × 11 데이터셋**을 측정. "**단일 아키텍처가 모든 시나리오를 지배하지 않는다**", "**국소 유지보수가 전역 재구성보다 비용효율적**". ch27의 경로별 판정에 필요한 외부 증거를 통째로 공급하고, ch15 caveat들(Zep의 community 미사용, Mem0 graph 퇴행)의 일반화 여부를 시험한다. | TeleAI-UAGI |
| 10 | **2605.16045** | *RecMem: Recurrence-based Memory Consolidation for Efficient and Effective Long-Running LLM Agents* — Dai, Deng, Guan, Tian, Yao, Yan, Cheng · 2026-05-15 | **F1/F4** | **"통합을 언제 할 것인가"를 설계 변수로 승격**시킨다. 들어오는 상호작용을 subconscious layer에 임베딩으로만 두고, 의미 유사 상호작용의 **재발(recurrence)이 관측될 때만** LLM 추출을 부른다. 3개 SOTA 시스템의 **memory construction 토큰 비용을 최대 87% 절감하면서 정확도는 상회**. corpus가 보고하지 않는 $B_s$를 E-경로에 대해 실제 숫자로 주고, (U-E)의 `wr` 호출 스케줄이 자유 변수임을 보인다. | TeleAI-UAGI |
| 11 | **2605.30152** | *Do Proactive Agents Really Need an LLM to Decide When to Wake and What to Anchor?* — X. Liu, R. Zhang, Abdi, Galley, Z. Chen, Xiong, X. Wang, Gao · 2026-05-28 | **F3/F4** | **"질의 전 예산을 LLM 추론에 쓸 필요가 있는가"**를 정면으로 반증한다. 활동 스트림을 텍스트화해 LLM에 먹이는 대신 temporal-graph 인코더로 per-event trigger 확률을 내고 필요할 때만 LLM을 부른다. **14개 backbone 전부에서 F1 향상(평균 +16.7, 최대 +46.0)**, GPU 4–7배 / 소비자 하드웨어 **12–83배** 빠르고 **~220MB로 온디바이스**. sleep 예산의 단위가 FLOPs가 아니라 "어떤 모델의 FLOPs"임을 강제한다 — ch25·ch28 회계의 전제를 흔든다. | LowEntropyAI |
| 12 | **2608.05475** | *KV-Skill: Forging Expertise in the Model's Native Language* — Han, X. Zhang, B. Han, K. Liu, Hu, J. Liu · **2026-08-05** | **F2** | 지식이 놓이는 자리를 "prompt text냐 weight update냐"의 **이분법 밖**에 새로 만든다 — frozen LM이 경량 인터페이스로 읽는 **external factorized KV operator**. prompt를 늘리지 않고, 저작된 텍스트 스킬을 operator로 등록하거나 reward로 latent operator를 학습한다. 10 벤치 4 backbone, Qwen3.5-4B LiveMath에서 **77.2% vs 텍스트 스킬 23.4%**, **동일 조건에서 LoRA·prefix-tuning을 이긴다**. 여러 KV-Skill을 성능 저하 없이 동시 적재 — ch19·ch26의 용량 논의에 $\Theta$·$W$·$E$ 어디에도 안 들어가는 **네 번째 저장 자리**를 들이민다. 이번 스윕에서 **가장 최신**. | yxf203 |

### Tier 3 — 보조 증거·경계 사례 (7편)

| # | arXiv | 제목 · 저자 · 날짜 | F | 왜 | 출처 |
|---|---|---|---|---|---|
| 13 | **2602.19320** | *Anatomy of Agentic Memory: Taxonomy and Empirical Analysis of Evaluation and System Limitations* — Jiang 외 11인 · 2026-02-22 (v2 2026-05-20) | F3/F4 | "**system-level cost가 자주 간과된다**", "기억 유지보수가 유발하는 **지연·처리량 오버헤드**", benchmark saturation, judge sensitivity, backbone 의존성. ch27이 열거한 결함 유형과 독립적으로 겹친다 — 이 책의 진단이 자기만의 관점이 아님을 보증하거나, 반대로 이미 알려진 것임을 드러낸다. | WebSearch(ID추적) |
| 14 | **2607.29377** | *Zero-Mem: Zero-Token Memory Operations for LLM Agents* — Xiao, Zhu, Y. Zhang, J. Chen, Hong, Zhuang, Q. Zhang, S. Chen, Ouyang, Ren, Huang · 2026-07-31 | F4/F2 | 기억 연산에서 **LLM 생성을 완전히 제거**하고(entity-context graph + temporal hierarchy) 최종 응답에만 LLM을 쓴다. **memory-operation 시간비용을 최속 baseline 대비 57.6% 절감**. $B_s \to 0$에서도 경쟁력이 나온다면 **상각 논증의 분자 자체가 사라진다** — ch25·ch27 판정의 하한 케이스. | yxf203 |
| 15 | **2605.13438** | *CogniFold: Always-On Proactive Memory via Cognitive Folding* — S. Wang, Duan, Deng, Zhao, Shi, M. Deng, C. Chen, Y. Wang, X. Zhou · 2026-05-13 (v4 2026-08-05) | F1/F2 | 파편 이벤트 스트림을 자기창발 구조로 **계속 접는** always-on 기억. **CLS를 명시적으로 확장**해 prefrontal intent layer를 추가하고, 구조가 의미적으로 가까우면 병합·낡으면 감쇠·밀도가 차면 활성화된다. §3.1 침묵 지도의 "**E-경로는 생물학을 인용하지 않는다**"에 대한 **반례 후보** — 성립하면 그 행을 고쳐야 한다. v4가 corpus 동결 이후(2026-08-05). | LowEntropyAI |
| 16 | **2605.09315** | *Do Self-Evolving Agents Forget? Capability Degradation and Preservation in Lifelong LLM Agent Adaptation* — Yu, Yuan, Jin, H. Liu, Y. Yu, H. Wang · 2026-05-10 | F3 | workflow·skill·model·**memory** 네 진화 채널 전부에서 "capability erosion"을 보이고, 안정화 원리(CPE)로 단순과제 성능 유지를 41.8%→52.8%로 올린다. $\rho$가 $\Theta$뿐 아니라 **$E$·워크플로에도 존재**함을 계열별로 분리한다 — ch30의 반증 조건에 채널 축을 추가한다. | wkqdzkd |
| 17 | **2607.01224** | *AutoMem: Automated Learning of Memory as a Cognitive Skill* — S. Wu, H. Zhu, Y. Zhang, X. Wang, Yeung-Levy · 2026-07-01 | F1 | 두 개의 **오프라인 루프**: (a) 강한 LLM이 **완결된 전체 궤적을 사후 검토**해 기억 구조(프롬프트·파일 스키마·행동 어휘)를 개정하고, (b) 여러 에피소드에서 골라낸 자기 기억결정을 **학습 신호로 써서 모델 자체를 갱신**한다. E와 Θ를 **동시에** 오프라인으로 바꾸는 드문 사례. task-action은 그대로 두고 기억만 최적화해 32B 오픈웨이트를 **2–4배** 끌어올려 프론티어급과 경쟁 — "기억 관리는 독립적으로 학습 가능한 스킬"이라는 주장은 ch27의 경로별 판정에 새 축을 넣는다. | TeleAI-UAGI |
| 18 | **2606.14571** | *StreamMemBench: Streaming Evaluation of Agent Memory for Future-Oriented Assistance* — G. Liu, Ren, Gu, P. Zhang, W. Wang, J. Liu, N. Gu, Lu · 2026-06-12 | F3 | 판별식 **조건 (4)("그 변화가 이후 질의에 쓰이는가")를 직접 측정 대상으로 삼은** 유일한 벤치. evidence recall / initial use / feedback incorporation / follow-up reuse 4축으로 8개 시스템을 재니, **저장은 됐는데 쓰이지 않는** 실패가 지배적이다. 조건 (4)가 자동으로 만족되지 않는다는 실증. | TeleAI-UAGI |
| 19 | **2605.11836** | *More Edits, More Stable: Understanding the Lifelong Normalization in Sequential Model Editing* — Ma, W. Chen, Q. Liu, D. Xu, Zheng, T. Xu, E. Chen · 2026-05-12 (v2 2026-07-21, **ICML 2026**) | F3 | 순차 편집이 왜 붕괴하는지에 대한 이론: 견고한 편집기가 공유하는 "Lifelong Normalization"이 갱신에 **점근적 직교성과 유계 노름**을 부여해 망각·계통적 붕괴를 막는다. ch19(ROME·MEMIT의 상한)와 ch22(반증 축)의 **기제적 설명**을 처음 준다. "이른 편집이 이후 편집의 성공을 돕는다"는 반직관 결과 포함. | WebSearch(ID추적) — **어느 GitHub 목록도 이 축을 담지 않았다** |

### 2.1 F 분포

| F | 편수 | 논평 |
|---|---|---|
| F1 (판별식 통과 후보) | 4 (#5, #6, #10, #17) + 경계 #15 | ProAct(#5)는 순수 통과. TMEM(#6)·CogniFold(#15)는 조건 (1)에서 경계 |
| F2 (경계·회색지대) | 4 (#6, #7, #12, #14) | KV-Skill(#12)이 **세 층 밖 네 번째 저장 자리**로 가장 강함 |
| F3 (제약·반증) | 8 (#2, #4, #8, #9, #11, #13, #16, #18, #19) | 이번 스윕의 **주 수확**. 특히 #2가 E-경로 (U-E) 자체의 반증 |
| F4 (비용·시스템) | 6 (#1, #3, #4, #9, #10, #11, #14) | #1·#3이 책이 선언한 공백을 정면으로 메움 |

---

## 3. 버린 것과 이유

### 3.1 목록에는 많았으나 전부 버린 대분류

| 버린 것 | 규모 | 이유 |
|---|---|---|
| self-evolving agent **응용** 논문 | ~180편 (selfimproving-agent 목록) | EvoDrive·RFAmpDesigner·TabClaw·SEMA-RAG 등 도메인 응용. 기제가 프롬프트/스킬 텍스트 축적이라 (U-E)의 재탕이고 비용도 안 잰다. **판정을 바꿀 수 없다** |
| skill library / skill-evolution | ~60편 (SkillOS·SkillDAG·SkillForge·SkillWiki…) | 스킬 = 텍스트 레코드. $E$층 `wr`의 변주일 뿐 새 층도, 새 제약도, 새 수치도 없다 |
| graph memory 변주 | ~15편 (HyperMem·Mnemis·MAGMA·G-Memory·SAGE·EXG…) | Zep(ch15)의 자료구조 교체. 판별식 판정 동일, 비용 보고 없음 |
| 멀티모달·로봇 기억 벤치 | ~10편 (MBench·RoboMemArena·RMBench·Persona-MME) | 워크로드가 이 책의 대상(텍스트 LLM 서빙)이 아니다 |
| self-play / RL 일반 | ~50편 | 질의 전 오프라인 개념 없음. PRE-RESEARCH §2.2 B의 "지속학습 일반"과 같은 사유로 배제 |
| vision/robot TTT 목록 전체 | 목록 6개 | `awesome test-time training` 검색 결과가 **전부 CV/로봇**. LLM TTT 큐레이션은 이 채널에 **존재하지 않는다** |

### 3.2 개별로 검토했으나 버린 것

| 논문 | 버린 이유 |
|---|---|
| 2606.13177 MemRefine / 2608.03463 LeanMem / 2601.02553 SimpleMem | E-경로 압축 변주. RecMem(#10)이 같은 축을 더 강하게(87%, 3시스템) 커버 |
| 2607.23809 ACM / 2606.23525 Self-Compacting LM Agents | context compaction. **ch22의 rate–distortion(2607.08032)이 이미 배제 목록**에 있고 그것이 이론적으로 상위 |
| 2605.11325 Structured Belief State (PrecisionMemBench) | 두 목록이 **제목을 서로 다르게** 적어 동일성 불확실. §4로 내림 |
| 2606.24535 Governed Shared Memory for Multi-Agent | 멀티에이전트 거버넌스. 층 갱신 아님 |
| 2601.07978 Cost and Accuracy of Long-Term Memory in Distributed MAS | 2026-01, 분산 MAS 한정. #1(2606.06448)이 더 넓고 더 최신 |
| 2602.06025 BudgetMem / 2603.13017 Structured Distillation / 2605.15759 DimMem / 2601.21162 A2RAG | 전부 2026 상반기 토큰 절감. #3 MEMAUDIT이 **인증된 최적해로 상위 호환** (초록 미확인분 포함, §4 참조) |
| 2606.24861 First-Order Recoverability Collapse / 2604.12128 Matrix-Level Dynamics | 자기참조 디코더 일반론. LLM 기억 갱신과의 연결이 초록에서 확인되지 않음 |
| 2605.22217 Survive or Collapse (self-play RL) | 붕괴를 다루나 대상이 self-play RL 데이터 게이팅. ch06 model collapse의 인접이지 신규 제약 아님 |
| 2604.00830 Learning to Learn-at-Test-Time | 2026-04이고 **abs 페이지 초록을 못 읽음**(검색 요약만). §4로 내림 |

### 3.3 이 채널에 대한 발견 자체

1. **agent-memory 계열 awesome 목록은 F1을 못 준다.** 14개 목록의 2026 항목이 압도적으로 $E$-경로
   텍스트/그래프 저장소이고, 서로 90% 중복이며, **가중치를 건드리는 항목을 거의 싣지 않는다**.
   `TsinghuaC3I`·`Shichun-Liu`·`RUC-NLPIR`·`EvoMap` 4개 목록은 **신규 후보 0건**이었다.
2. **별 수와 유용성이 반비례했다.** 1,700★ `VoltAgent`는 2026-05 이후가 하나도 없어 죽었고,
   9★ `LowEntropyAI/awesome-proactive-agent`가 이 스윕 최대 발견(#5 ProAct)의 유일한 출처였다.
   프롬프트의 "**어느 목록에도 없다가 한 목록에만 있는 것**" 지침이 그대로 맞았다.
3. **비용(F4)을 카테고리로 가진 목록은 단 하나** — `Snseam`의 `docs/cost-savings-landscape.md`.
   여기서 #3 MEMAUDIT이 나왔다. 나머지 13개 목록은 비용을 축으로 두지 않는다.
4. **Θ-경로(모델 편집·순차 편집 붕괴)는 이 채널에 존재하지 않는다.** `awesome model editing`류
   검색이 0–2건이고 전부 이미지 편집이었다. #19는 목록이 아니라 검색으로 잡았다. **이 책의 ch19·ch22
   축은 GitHub 큐레이션으로는 갱신할 수 없다** — 다른 채널(학회 프로시딩·저자 추적)이 필요하다.
5. **큐레이션 목록의 arXiv ID를 신뢰하면 안 된다.** 확인한 것 중 1건이 완전히 틀린 ID(입자물리
   논문)였고, 같은 논문을 목록마다 다른 제목으로 싣는 사례가 최소 2건이었다.

---

## 4. 미검증 단서 (초록을 못 읽었으므로 후보에서 제외 — 후속 확인 대상)

| 식별자 | 제목(목록 기재) | 왜 남기는가 |
|---|---|---|
| 2604.00830 | Learning to Learn-at-Test-Time: Language Agents with Learnable Adaptation Policies | 적응 정책 자체를 학습 = 학습된 쓰기 정책. 2026-04이나 F1 가능. **abs 미확인** |
| 2601.11042 | Spectral Characterization and Mitigation of Sequential Knowledge Editing Collapse | #19와 같은 축의 F3. PDF 링크만 봄, **abs 미확인** |
| 2511.20172 | Beluga: A CXL-Based Memory Architecture for Scalable and Efficient LLM KVCache Management | ch29 웜계층 재료(F4). 2025-11이라 우선순위 낮춤, **abs 미확인** |
| 2512.18194 | TraCT: Disaggregated LLM Serving with CXL Shared Memory KV Cache at Rack-Scale | 〃 |
| — | LightMem (ACL 2026) / ConvoMem / TencentDB Agent Memory L0–L3 | `cost-savings-landscape.md`가 **arXiv ID를 안 적음**. ConvoMem은 "정확도/비용/지연 교차점"이라 F4 유망 |
| 2605.11325 | Structured Belief State / PrecisionMemBench (제목 2종) | 목록 간 제목 불일치. 동일 논문 여부 확인 필요 |

### 4.1 초록 확인했으나 F 귀속이 애매해 보류한 것 (참고)

| arXiv | 제목 | 보류 사유 |
|---|---|---|
| 2605.03375 | *Tutti: Making SSD-Backed KV Cache Practical for Long-Context LLM Serving* (Qiu 외 · 2026-05-05) | **F2/F4로는 강함** — SSD 계층에서 SLO 하 TTFT 78.3% 감소, 요청률 2배, 서빙 비용 27% 절감. 다만 대상 상태가 **단일 세션 KV cache**라 sleep-time 워크로드와 직결되지 않는다. ch29(memory-device 기회)가 웜계층 실측치를 필요로 하면 **즉시 승격 대상** |
| 2605.22850 | *ObjectCache: Layerwise Object-Storage Retrieval for KV Cache Reuse* (Y. Zhu, Dhakal, Xiao, Milojicic, Alonso · 2026-05-16) | 〃. S3 호환 객체저장에 KV를 두고 **64K 문맥에서 local DRAM 대비 지연 +5.6%**. X2의 "웜 계층 delta 스테이징"과 **같은 자리를 재는 유일한 실측**이지만 대상이 delta가 아니라 prefix KV |

이 둘은 **판정 기준상 F2·F4에 형식적으로 해당**하나, 프롬프트의 "기억·에이전트 논문은 무수히 많고
가치는 판정을 바꿀 수 있는 것만 고르는 데 있다"는 제약을 존중해 **본 후보표에서 뺐다**.
ch29를 쓸 때 웜계층 숫자가 필요해지면 이 두 편이 첫 번째로 볼 것이다.

---

## 5. 요약

- 방문: GitHub 검색 9회, README 정독 14개 목록, arXiv 초록 직접 확인 22건.
- **후보 19편** — 전부 초록 확인. F3 8편 / F4 6편 / F1 4편 / F2 4편 (중복 계산).
- **최우선 3편**: `2606.06448`(F4, 시스템 실측 — ch25·ch26·ch29 공백 정면), `2605.12978`(F3,
  E-경로 반복 갱신의 붕괴 — ch27 판정 근거), `2605.02199`(F4, 바이트 예산 쓰기 회계 — ch26 공백).
- 이 채널의 한계가 곧 결과다: **agent-memory awesome 목록은 $E$-경로에 편향돼 있고, $\Theta$-경로
  반증 축과 비용 회계는 거의 담지 않는다.** 두 축은 다른 채널로 다시 훑어야 한다.
