# TRIAGE — gap-sweep 6채널 병합·중복제거·선별

**작성**: 2026-08-11
**입력**: `gap-sweep/{arxiv-listing,citation-graph,community,github-awesome,hf-papers,survey-bib}.md`
**배제 대조**: `scratchpad/corpus-exclusion.md` — deep-read 32편 + vendored arXiv ID 69개
**판정 기준**: `S0-DESIGN.md` §2.3 판별식 · `PRE-RESEARCH.md` §2·§4·§5 · `OUTLOOK-REVIEW-2026-08-11.md` §5·§7
**채널 약호**: AX(arXiv 직접검색) · CG(인용그래프) · CM(커뮤니티·제품) · GH(GitHub 큐레이션) · HF(HF Papers) · SB(서베이 참고문헌)

---

## 0. 요약

| 단계 | 수 |
|---|---|
| 6채널이 후보표에 올린 항목(합) | **137** |
| 중복 제거 후 고유 항목 | **106** (arXiv 101 + 비-arXiv 5) |
| arXiv ID 실재·제목 검증 통과 | **101 / 101** |
| 배제목록 위반 | **0** (파서 오탐 1건 — 아래 §3) |
| Tier 1 (판정을 바꿀 수 있음) | **10항목** (arXiv 9 + 제품 3종 1행) |
| Tier 2 (읽을 값은 있으나 걸린 판정 없음) | **43** |
| 버림 | **51** (+ 채널들이 이미 훑고 버린 약 700편) |

**한 문단 판정.** 137건 중 판정을 실제로 건드리는 것은 10건이다. 그 10건은 세 자리에 몰린다 — (a) **반복 갱신의 부호**(ρ가 단조 감소인가): 2606.04703과 2605.11836이 **반대 방향**을 가리키고 둘 다 5채널이 독립적으로 올렸다. X4는 설계 전에 이 둘을 읽어야 하고, 읽고 나면 가설 자체가 달라진다. (b) **통제군이 붙은 첫 경로 대결**: 2605.24657이 no-context 바닥과 full-context 천장을 **둘 다** 세운 채 Θ와 E를 맞붙였다 — `OUTLOOK-REVIEW` §7.6이 "full-context 팔을 실행한 논문이 아직 0편"이라 쓴 그 문장이 이것 하나로 거짓이 된다. (c) **회계의 외부 실측**: 2606.06448(phase-aware 프로파일 + "amortization via query volume")과 2608.00101(3.2M 사용자 프로덕션 트레이스, 턴 경계 분 단위 유휴)이 ch25가 자기 기여로 세운 자리와 ch14·ch25의 $N_q$ caveat를 동시에 건드린다.

**채널들이 놓치거나 틀린 것 셋.** ① CG가 `2608.02515 LiveMem`을 "ch17의 W-경로 유일사례 서술을 깬다"고 올렸으나 **초록을 읽으면 시계가 wake다** — 스트림이 흐르는 동안 고정 용량 상태를 이어받는 구조이고 질의 전 유휴 갱신 단계가 없다. Memory Caching 옆 자리이지 ch17의 두 번째 표본이 아니다. **ch27 §27.5 FW1의 "W-경로 표본은 1로 유지된다"는 살아남는다.** ② HF가 `2606.04351`을 *Video2LoRA*로 적었으나 arXiv 원제는 **Frames2LoRA**다. ③ CM이 "세 제품 모두 정량 결과를 내지 않았다"고 썼으나 OpenAI는 시간민감 기억 정확도 9.4%→75.1%를 공개했다(통제군 없는 벤더 자기보고이므로 인용 불가라는 결론은 유지되지만, "정량치 0건"은 틀렸다).

---

## 1. Tier 1 — 읽으면 판정이 바뀔 수 있는 것 (10편)

`ch/판정문` 칸은 **이 후보가 무효화하거나 조건화하는 인쇄된 문장**을 지목한다. 지목하지 못하면 Tier 2로 내렸다.

| # | arXiv | 제목 | v1 날짜 | F | 걸린 장과 판정문 | 채널 |
|---|---|---|---|---|---|---|
| 1 | **2605.24657** | Beyond Inference-Only Deployment: Comparing Weight-Based Consolidation Against Cascading Compaction | 2026-05-23 | **F1(Θ)+F3** | `OUTLOOK` §7.6 — "다세션 레짐에서 $E$-경로가 memoryless 팔을 이기지만 **full-context 팔을 실행한 논문이 아직 0편이다**". 이 논문은 no-context 바닥 11.8%·full-context 천장 90.1%를 **둘 다** 세우고 그 사이에서 야간 LoRA 통합 80.4%와 3회 cascading compaction 36.8%를 맞붙인다(paired t(9)=14.8). ch27 판정 2의 "미검증" 절반이 검증된다. 동시에 `PRE-RESEARCH` §3.1 침묵 지도 — corpus 최초의 **통제된 Θ↔E 직접 대결** | AX |
| 2 | **2605.12978** | Useful Memories Become Faulty When Continuously Updated by LLMs | 2026-05-13 | **F3** | ch27 판정 2("$E$-경로는 품질 방법으로서 문헌이 지지하지 않는다") — 관찰이 **기제**가 된다. 효용이 상승 후 no-memory baseline **아래로** 내려가고, ground-truth 해답으로 통합해도 GPT-5.4가 이전에 기억 없이 푼 ARC-AGI의 54%를 실패한다. 결정적으로 **episodic-only 통제군이 통합기들과 대등**하고 통합을 아예 끄면 auto 레짐과 같다 — (U-E)의 `wr` 자체가 순손실이라는 통제 증명. ch25·ch29가 "아무도 안 잰다"고 쓴 $E$-경로 $\rho$의 첫 값 | CM, GH |
| 3 | **2606.04703** | Rethinking Continual Experience Internalization for Self-Evolving LLM Agents | 2026-06-03 | **F3** | **X4(반복 consolidation 열화 $\rho$)의 선행 실측.** 다회차에서 기존 방법이 복리 개선이 아니라 **progressive capability collapse**를 겪고, 원인을 입도(principle > instance)·주입(step-wise > global)·레짐(off-policy > on-policy)으로 분해한다. ch20(SEAL self-edit)·ch06($\mathcal{R}$ 자기생성)에 동시에 걸리고, X4를 이 논문 뒤에 설계하지 않으면 중복 실험이 된다 | CG, CM, GH, HF, SB (**5채널**) |
| 4 | **2605.11836** | More Edits, More Stable: Understanding the Lifelong Normalization in Sequential Model Editing | 2026-05-12 | **F3(반전)** | **#3과 정면으로 반대 방향.** "counter-intuitive **positive cumulative effect where early edits can promote the success of future edits**" — 사실이면 $\rho$는 단조 감소가 아니다. ch29 판정 4("반복이 성립하려면 replay가 보존된 원본에 앵커되어야 한다")와 ch30·X4의 열화 전제가 조건부가 된다. LN + ridge가 **점근적 직교성과 유계 노름**을 준다는 정리까지 붙어 있어 "언제 붕괴하지 않는가"의 조건절이 생긴다 | AX, CG, CM, GH, SB (**5채널**) |
| 5 | **2606.06448** | Agent Memory: Characterization and System Implications of Stateful Long-Horizon Workloads | 2026-06-04 | **F4** | ch25(통일 비용 회계)는 이 책이 **자기 기여로 세운 자리**다. 이 논문은 construction/retrieval/generation 3단계에 비용을 귀속하는 phase-aware 프로파일러로 10개 시스템×2 벤치를 계측하고, 10개 권고 중에 **"amortization via query volume"과 freshness-latency 트레이드오프**를 명시한다. ch14·ch25의 "$N_q$를 안다고 가정 / 실서빙 분포 보고 없음" caveat와 ch25의 신규성 주장이 함께 재작성 대상 | GH, SB |
| 6 | **2608.00101** | Agentic Coding in the Wild: Characterizing GitHub Copilot Traces at Production Scale | 2026-07-30 | **F4** | 같은 caveat의 다른 절반. **3.2M 사용자·13M 세션·761M LLM 호출·95T 토큰**(2026-06 트레이스)에서 턴 내 KV hit 90% → 턴 경계 55%로 떨어지고, **턴 경계에 분(minutes) 단위 사용자 유휴**가 존재하며 경량 예측기가 총 유휴의 86–90%를 포착한다. 이 책 전체가 전제하면서 한 번도 근거를 대지 않은 명제 — **"쓸 수 있는 유휴 시간이 실제로 있다"** — 의 첫 프로덕션 증거이자 $N_q$·$B_s$ 가용창의 실측 | AX |
| 7 | **2605.25971** | Anticipate and Learn: Unleashing Idle-Time Compute in Proactive Agents (ProAct) | 2026-05-25 | **F1(E)** | `OUTLOOK` §5.1의 인쇄된 문장 — "**Precompute 열은 여전히 한 편이다. 2026-08 기준으로도 그 열은 그 한 편뿐이다**". ProAct는 유휴시간에 대화이력+persistent memory를 분석해 다음 필요를 예측하고 **사용자가 질의를 시작하기 전에** 증거를 준비한다. Precompute×$E$ 칸이 둘이 되고, ch11·ch24·부록 C의 통과군 카운트가 또 바뀐다. Letta STC를 인용하지 않는 **독립 재발명**이므로 침묵 지도에도 행이 하나 는다. ⚠ 조건 (3)의 지속성(준비한 증거가 persistent memory에 쓰이는가)은 초록에서 확정되지 않는다 — deep-read 첫 확인 항목 | CG, GH, SB |
| 8 | **2607.12227** | Rethinking the Evaluation of Harness Evolution for Agents | 2026-07-14 | **F3** | ch28의 핵심 질문("$B_s$를 써서 얻은 이득이 그냥 $B_t$를 더 쓴 것보다 나은가")에 대한 **예산을 맞춘 부정 답변**. 피드백·추론 예산을 통제하면 자동 harness 진화가 단순 test-time scaling을 **일관되게 못 이기고**, held-out에서 일반화도 제한적이며, 탐색과 최종 평가가 같은 벤치를 공유하는 과적합이 있다(Terminal-Bench 2.1 / GPT-5.4 / Claude Opus 4.6). ch27의 "유망한가"에 직접 걸린다. ⚠ 대상이 harness(프롬프트·툴·워크플로)이지 기억 층이 아니다 — $E$-인접으로 한정해 인용해야 한다 | HF |
| 9 | **2601.11042** | Spectral Characterization and Mitigation of Sequential Knowledge Editing Collapse (REVIVE) | 2026-01-16 | **F3** | ch19·ch22의 Θ 쓰기 상한은 현재 **수치 하나**로 지탱된다(MEMIT zsRE Specificity 26.6 < 미편집 GPT-J 27.0). 이 논문은 붕괴를 **사전학습 가중치의 지배적 특이방향 훼손**으로 기제화하고, 그 부분공간을 보존하면 **최대 20,000 편집**까지 간다는 완화책을 붙인다. #4와 합치면 ch22의 반증 축이 "쌓을 수 없다"에서 **"갱신 기하를 통제하지 않으면 쌓을 수 없다"**로 형태가 바뀐다 | CM, SB |
| 10 | **제품 3종** (비-arXiv) | Anthropic **Dreaming** (Claude Managed Agents, 2026-05-06) · OpenAI **Dreaming V3** (2026-06-04) · Mem0 **Dream** (2026-08-05) | — | **F1(E)+F4** | ch24("무엇이 실제로 sleep-time compute인가")는 논문만 열거한다. 판별식 네 조건을 문면 그대로 만족하는 시스템 셋이 소비자·기업 규모로 **이미 배포됐고 corpus에 하나도 없다.** 더구나 셋의 `wr`이 서로 다르다 — Anthropic은 **비파괴·병렬 산출**(입력 store 불변, 승인 후 반영), OpenAI는 **in-place 덮어쓰기 + 시간적 무효화**, Mem0는 **supersede 포인터 + lifecycle state**. ch14가 "Letta의 `wr`은 파괴적 덮어쓰기"를 축으로 세운 자리가 제품 층위에서 3분기했다. Anthropic은 **표준 API 토큰 요율로 과금** = $B_s$가 청구 가능한 양으로 노출된 첫 사례 | CM (본 검토가 Anthropic·OpenAI를 독립 재확인) |

### 1.1 Tier 1이 서로에게 하는 일

- **#3 ↔ #4는 충돌한다.** 하나는 반복 내재화가 붕괴한다 하고, 하나는 이른 편집이 이후 편집을 돕는다 한다. 둘 다 5채널이 독립적으로 올렸다. **이 충돌이 X4의 진짜 질문이다** — "열화하는가"가 아니라 "무엇이 있으면 열화하지 않는가". #9(REVIVE의 특이방향 보존)가 #4와 같은 방향의 두 번째 기제다.
- **#1 ↔ #2도 충돌한다.** 하나는 가중치 통합이 압축을 두 배 넘게 이긴다 하고, 하나는 텍스트 통합이 raw episode 보존보다 못하다 한다. 두 결과는 양립 가능하다 — **둘 다 "통합(consolidation)이라는 연산 자체가 손실"이라는 같은 말을 다른 층에서 한다.** ch27 경로별 판정을 쓰기 전에 이 대응을 세워야 한다.
- **#5 + #6은 같은 caveat의 두 절반**이다. #5는 회계 방법을, #6은 실서빙 분포를 준다. 둘을 함께 읽지 않으면 ch25가 어느 쪽으로 고쳐져야 하는지 정해지지 않는다.

---

## 2. Tier 2 — 읽을 값은 있으나 걸린 판정이 없는 것 (43편)

전부 arXiv ID·제목 검증 완료. 묶음 안의 순서는 값 순.

### 2.1 $E$-경로 품질 진단의 외부 재현 (ch15·ch27) — 판정을 **확증**한다

| arXiv | 제목 | 무엇을 주는가 | 채널 |
|---|---|---|---|
| 2606.29914 | MemDelta: Controlled Baselines and Hidden Confounds in Agent Memory Evaluation | 한 번에 한 성분만 바꾸는 통제 프로토콜. self-memory 42% < 기본 retrieval 47%, embedding만 바꿔도 결론 역전(+6.2pp, p=0.004), Mem0가 cloud-RAG와 동률인데 비용 50배. 권고가 "아키텍처에 이득 귀속 전 write-path 비용을 보고하라" | CG, GH |
| 2602.19320 | Anatomy of Agentic Memory | 벤치 포화·judge 민감도·backbone 의존·**유지보수의 지연/처리량 오버헤드**를 명명된 목록으로. ch15·ch22가 개별 논문마다 따로 지적하는 결함의 상위 범주. **"아무도 비용을 안 본다"고 쓰기 전에 반드시 대조** | CM, GH, SB |
| 2607.24368 | Keep It InMind: Implicit-Association Blind Spot in Agent Memory | 같은 기억을 문맥에 넣으면 84.0%, 검색시키면 6개 시스템 최대 14.4%. 임베딩 8배로도 격차 불변 → 용량 문제 배제. ch27 A2를 인터페이스의 구조적 실패로 | HF |
| 2606.24595 | MEMPROBE: Probing Long-Term Agent Memory via Hidden User-State Recovery | 과제 완수는 memoryless에서도 포화하는데 범주균형 복원은 ~0.6. "성공했다"≠"기억이 작동했다"의 분리 증명 | HF |
| 2606.04315 | Exploring Cross-Scenario Generality of Agentic Memory Systems | 8시스템×5시나리오 재평가에서 **평범한 에이전트 하네스(평문 파일 자율관리)가 교차과제 1위**. ch27 판정문의 형태를 "1위가 기억 없음"에서 "1위가 자율 제어"로 정밀화 | SB |
| 2601.00821 | Fidelity Before Structure: Verbatim Chunks Beat Lossy Artifact Extraction | 저장 표현만 바꾼 통제 ablation + 6종 confound 통제(LoCoMo 43.9 vs 28.0). $E$-경로의 전제(구조화가 원문보다 낫다)를 정면 반증 | AX |
| 2607.21962 | Ground Truth First: Longitudinal Evaluation Instrument, and the Tenure Crossover | 기억 아키텍처 **순위가 이력 길이에 따라 역전**(3주 선두가 9주에 96%→72%). full-history baseline이 단기 동률이되 읽기 비용 2배 | CG |
| 2606.24775 | Are We Ready For An Agent-Native Memory System? | 12시스템+2 baseline × 5 워크로드 × 11 데이터셋. "단일 아키텍처가 모든 시나리오를 지배하지 않는다", "국소 유지보수가 전역 재구성보다 비용효율적" | GH, SB |

### 2.2 경로 간 교차 — 침묵 지도의 시제 갱신 재료 (ch11·ch27)

| arXiv | 제목 | 무엇을 주는가 | 채널 |
|---|---|---|---|
| 2607.21051 | Sample-Efficient Learning from Agent Experience | 같은 과제에서 잰 **경로 간 환산 계수**: 문맥에서 가중치로 옮길 때 ICL 이득의 64.8%가 남고 단순 SFT는 3.8%만 남는다(SWE 749과제 + 텍스트게임 6종) | HF |
| 2606.19172 | User as Engram: Internalizing Per-User Memory as Local Parametric Edits | per-user LoRA 대비 ~33,000× 작은 발자국, 해시 슬롯이 disjoint라 사용자 편집이 **가산적·무손실 합성**, ~100 fact에서 검색 파이프라인 추월. ⚠ 단독저자·자체 벤치·Engram 전용 아키텍처라 임의 transformer로 이전되지 않는다 | AX, CG, SB |
| 2606.17628 | OPD-Evolver: Holistic Agent Evolver via On-Policy Distillation | 느린 루프가 경험을 정책 가중치로 증류하고 빠른 루프가 4단 기억을 읽고 쓴다 — $E$와 $\Theta$를 한 시스템에서 잇는다 | AX |
| 2606.11712 | Substrate Asymmetry in User-Side Memory: A Diagnostic Framework | per-user LoRA($\Theta$) 대 dense retrieval($E$)을 세 직교 축(스타일 일관·사실 존재·사실 부재)에서 가르고 21–35층 query-projection 인과 ablation | AX |

### 2.3 $\Theta$-경로 용량·반증 (ch09·ch19·ch22)

| arXiv | 제목 | 무엇을 주는가 | 채널 |
|---|---|---|---|
| 2607.19604 | Scaling Laws for Hypernetwork-Based Knowledge Injection | 하이퍼네트워크가 사실 코퍼스로부터 고정 LoRA를 생성. 깊이·폭·타깃 크기 축 멱법칙 + OOD에서 더 가파른 지수. ⚠ **$B_s$ 축 곡선이 아니다**(아키텍처 축이다). 그리고 시계가 train-time 1회이므로 Memory Layers와 같은 이유로 **판별식 통과가 아니다** — HF의 F1 표기는 정정 대상 | HF |
| 2601.15313 | Attention Is Not Retention: The Orthogonality Constraint | 의미 밀도가 높으면 N=5개 사실에서, 중간이면 N≈20–75에서 공유 연속 파라미터 위 사실이 붕괴(16,309 fact). ch09의 3.6 bits/param과 **다른 축의** 기하학적 상한 | CG |
| 2607.08393 | Why Memorized Knowledge Fails to Generalize in LLM Finetuning | **Knowing–Using Gap**: 넣는 데 성공해도 하류 추론에 못 쓰고, 정확도 격차와 시간 지연이 함께 온다. ch22(2607.11020)의 행동적 반증에 기제적 짝 | HF |
| 2605.30260 | How LoRA Remembers? A Parametric Memory Law for LLM Finetuning | $\Delta L$–유효파라미터–시퀀스길이 멱법칙 + 토큰 수준 $p>0.5$ 결정론적 상전이. ch09 용량 라인에 정량 법칙 추가 | SB |

### 2.4 자기생성 데이터의 붕괴 (ch06·ch20·ch21)

| arXiv | 제목 | 무엇을 주는가 | 채널 |
|---|---|---|---|
| 2606.32002 | Self-Study Reconsidered: Hidden Fragility of Learning from Self-Generated QA | 자기생성 QA가 중립 전처리가 아니라 암묵 정책이고 coverage가 조기 포화. injection compliance 88%(필터 후 13%). SEAL·`LM Need Sleep`의 $\mathcal{R}_k$ 생성 단계에 **model collapse와 다른 종류의** 붕괴 | CG |
| 2607.01763 | Denser ≠ Better: Limits of On-Policy Self-Distillation for Continual Post-Training | self-distillation이 망각을 키우고 붕괴시키며 GRPO가 오히려 보수적. 자기강화 교사–학생 루프의 고주파 형식 인공물 증폭 | HF |
| 2606.28438 | When AI Reviews Its Own Code: Recursive Self-Training Collapse | AI-self-gate가 rubber-stamp 레짐에 빠져 수락 점수는 오르고 정확도는 떨어진다. "안정적 재귀 학습은 **외생적 검증**을 요구한다" — SEAL·SCM의 자기검증 루프에 직접 반증 축 | SB |

### 2.5 비용·시스템 F4 (ch25·ch26·ch29)

| arXiv | 제목 | 무엇을 주는가 | 채널 |
|---|---|---|---|
| 2607.18141 | A CXL Memory Rack for Multi-Turn LLM Serving (HyMCache) | 실물 CXL-HM 프로토타입, LMCache 대비 3.0×, Mooncake 대비 DRAM 16× 절감에 성능 −30%. ch29 웜계층 논증의 하드웨어 대응물 | CG, SB |
| 2608.05483 | PLoRA: NDP-Enhanced Pooled-Memory System for Multi-LoRA Serving | H100 1장에 **1000개 어댑터**, CXL/NVLink 풀 + NDP, 디코드 지연 평균 6.6×↓. $\Theta$-경로가 주류일 때의 per-user delta 서빙 실측 | AX |
| 2604.06370 | ForkKV: Multi-LoRA Agent Serving via Copy-on-Write Disaggregated KV | per-agent LoRA가 **KV를 발산시켜 prefix caching을 무력화**한다는 것이 진짜 병목. A2 논증의 **반대 방향** 증거(비용이 어댑터 바이트가 아니라 KV 발산에 있다) | CG, SB |
| 2607.29377 | Zero-Mem: Zero-Token Memory Operations for LLM Agents | 기억 연산에서 LLM 생성을 완전히 제거, memory-operation 시간비용 최속 baseline 대비 57.6%↓. $B_s\to0$ 하한 케이스 | GH, HF |
| 2607.19214 | Keeping the Cache Warm Pays: Keepalive Economics for Agentic Workloads | Anthropic/OpenAI/Google/DeepSeek 실제 가격·TTL로 sleep–wake 간극 손익분기 유도(post-pause 최대 12.5×↓, 손익분기 ~46분). ch14의 "$\kappa_{\text{cost}}=10$ 민감도 분석 없음"에 외부 회계 | AX |
| 2602.02574 | WritePolicyBench: Memory Write Policies under Byte Budgets | **byte-accurate cost model** + drift 있는 스트림 + 명시적 action interface. corpus 어느 논문도 갖지 못한 통제된 쓰기-예산 축(cs.PF) | AX |
| 2605.02199 | MEMAUDIT: Exact Package-Oracle Protocol for Budgeted Long-Term Memory Writing | 저장 예산을 명시 제약으로 걸고 branch-and-bound + MILP로 **인증된 최적해**. Mem0·A-Mem·Letta 스토어를 실제 export해 채점 | GH |
| 2606.25115 | Forget to Improve: On-Device Continual Learning via Budget-Curated Memory | **net-value-per-byte**로 KEEP/SHARE/TRUST 결정, Jetson 실기 RAM·에너지·uplink 측정(메모리 2.7×↓) | AX, CG |
| 2606.09613 | AGENTSERVESIM: Hardware-aware Simulator for Multi-Turn LLM Agent Serving | 도구호출 간극·턴 간 캐시 지역성·HBM/DRAM/CXL KV 잔류를 모사, 실기 대비 6% 오차. **X3·X4를 GPU 없이 돌릴 수 있는 도구 — 실험 계획을 바꾼다** | AX |
| 2605.30571 | Memory-Bound but Not Bandwidth-Limited: Physical AI Inference Gap in Batch-1 Decode | 4 GPU×3 모델×문맥 44셀 실측 — **피크 대역폭이 높을수록 달성률이 떨어진다**(L4 81% vs H100 27%). "바이트를 줄이면 지연이 비례해 준다"가 batch-1에서 거짓. ch29·A2가 통과해야 할 관문. ⚠ 단독저자·소규모 | HF |

### 2.6 판별식 경계 — ch11 회색지대 재작성 재료

`OUTLOOK` §7.5가 이미 회색지대를 셋으로 늘리기로 했다. 아래는 그 절을 쓸 때의 표본이다.

| arXiv | 제목 | 어느 조건을 시험하는가 | 채널 |
|---|---|---|---|
| 2605.22154 | IdleSpec: Exploiting Idle Time via Speculative Planning | 유휴 예산을 쓰지만 산출물이 **에피소드 안에서 소멸** → 조건 (4) 불충족. MemGPT(조건 1 위반)와 **대칭인 반대편** 사례 | AX, CG, SB |
| 2606.15734 | Retrievable Gradients (ReGrad) | gradient를 **오프라인 사전계산**해 Gradient Bank에 색인하고 질의 시 관련분만 꺼내 **일시** 적응. 지속 상태는 $E$(뱅크), $\Theta$ 변경은 wake 일시 → 판별식은 $E$로 옳게 판정하되 **$E$의 비용 모델(프롬프트 토큰)이 깨진다** | AX |
| 2607.23693 | Compute Globally, Materialize Locally: Memory Contract of Sparse Event-KV | KV 행 자체에 의도적으로 쓴다(답 미명시 문장 하나로 donor-정렬 복원 6%→51%). 압축 상태는 살아남고 큰 payload는 우연 수준으로 붕괴. 그리고 방법론: **"원본을 지웠는데 정확도가 안 떨어졌다고 그 원본이 불필요했다는 증명은 아니다"** — eviction/compaction 문헌의 추론 형식에 걸린다 | HF |
| 2505.16950 | Bottlenecked Transformers: Periodic KV Cache Consolidation | 스스로를 memory consolidation이라 부르면서 바꾸는 것은 KV이고 쓰기가 **생성 도중** 일어난다 — 조건 (1)과 (3)을 동시에 시험하는 가장 깨끗한 경계 사례 | SB |
| 2606.14571 | StreamMemBench: Streaming Evaluation for Future-Oriented Assistance | 조건 (4)("그 변화가 이후 질의에 쓰이는가")를 **직접 측정 대상으로 삼은 유일한 벤치**. 8시스템에서 **저장은 됐는데 쓰이지 않는** 실패가 지배적 | GH |
| 2608.02515 | LiveMem: Memory State Continuity in Long-Running LLM Inference | ⚠ CG가 "W-경로 두 번째 사례"로 올렸으나 **시계가 wake다**(문맥 회전 중 고정용량 상태 이어가기). Memory Caching 옆 B군. **ch27 FW1은 유지된다** | CG |

### 2.7 예산 레짐과 포화 (ch28)

| arXiv | 제목 | 무엇을 주는가 | 채널 |
|---|---|---|---|
| 2607.17545 | Retain or Consolidate? Budget-Dependent Operator Selection | "보존 대 통합"의 우열이 **예산에 따라 교차**함을 coverage/replacement 분해로 설명하고 LongMemEval·LoCoMo에서 crossover 재현(빡빡한 예산에서 통합 최대 48% 우세) | AX |
| 2607.05155 | EdgeBench: Scaling Laws of Learning from Real-World Environments | 실환경 상호작용 38,000시간에서 배치 후 경험학습이 **log-sigmoid 법칙(R²=0.998)** — 즉 **포화한다**. A3의 "$B_s$ 지수와 포화점"에 대응하는 유일한 대규모 실측 | HF |
| 2605.30152 | Do Proactive Agents Really Need an LLM to Decide When to Wake? | 질의 전 예산을 **LLM 추론에 쓸 필요가 있는가**를 반증. temporal-graph 인코더로 14 backbone 전부 F1 향상(평균 +16.7), GPU 4–7배·소비자 HW 12–83배 빠르고 ~220MB 온디바이스. **sleep 예산의 단위가 FLOPs가 아니라 "어떤 모델의 FLOPs"임을 강제** | GH |
| 2605.30621 | Harness Updating Is Not Harness Benefit | 쓸모 있는 갱신을 만드는 능력이 기반 능력에 대해 **평평**하고(9B ≈ Opus 4.6), 그것으로 이득 보는 능력은 비단조. 오프라인 갱신 품질이 병목이 아니라는 뜻 | HF |
| 2608.00303 | CrystalMem: Elastic Memory via Knowledge Crystallization | 바이트 예산을 줄였다 복구해도 능력이 안 돌아오는 **memory hysteresis** + keep/drop 정책의 **residual-deficit floor**. $E$-경로 `wr` 비가역성의 정량 하한 | AX, CG |

### 2.8 새 실패 양식 — ch30 한계 절 재료

| arXiv | 제목 | 무엇을 주는가 | 채널 |
|---|---|---|---|
| 2608.01679 | When Memory Becomes Authority: Authority Collapse at the Consolidation Boundary | 통합이 주장은 보존하면서 **출처의 권한 제약을 지운다**. 7 consolidator × 7 backbone = **49 구성 중 48에서 붕괴**, 무권한 행동률 평균 50.3% | SB |
| 2608.03509 | SkillJack: Persistent Skill Backdoors in Self-Evolving Agents | consolidation이 **출처를 세탁**한다: 안전 탐지율 98.5% → 추출된 스킬 11.4%, **80.0%가 원본 레코드를 지운 뒤에도 생존**. (U-E)의 `wr`이 비가역적 승격이라는 뜻 | HF |
| 2608.04570 | The Personalization Mirage: How LLMs Fabricate User Profiles | 12개 모델 **전부**가 사용자 속성의 35–49%를 근거 없이 날조(143,616 판정, κ=0.863), 다회차에서 거의 선형 누적, 자기평가와 실측이 음의 상관(ρ=−0.60). `wr`의 **입력 자체**가 오염 | HF |

---

## 3. 검증 결과

### 3.1 arXiv ID 실재·제목 검증 — 101/101 통과

`export.arxiv.org/api/query?id_list=`로 6배치 일괄 조회. **환각 ID 0건.** 반환된 제목을 채널 기재 제목과 토큰 중복으로 대조한 결과 84편이 완전 일치, 나머지는 약어(LLMs↔Large Language Models)·부제 생략·저자명 혼입에 의한 정상 편차였다. 다만 셋을 기록한다.

| 항목 | 사실 | 처리 |
|---|---|---|
| **2606.04351** | arXiv 원제는 **Frames2LoRA**: Parametric Video Internalization for Vision-Language Models. HF 채널이 **Video2LoRA**로 적었다 | 논문은 실재·주제 일치. **이름 정정 필요.** Tier 밖(도메인이 VLM) |
| **2607.20433** (Moir) | ID는 2607대인데 v1 제출일이 **2026-05-10**. SB가 스스로 "날짜 불일치, unverified"로 표기했다. ID·제목은 실재 확인됨 | 인용 전 원문에서 판 이력 확인 필요. Tier 밖(편집 *기법* 논문) |
| **2608.00009** (AgentMemBench) | ID는 2608대인데 v1 제출일이 **2026-06-16**(6주 간극). SB가 스스로 표기했다. ID·제목 실재 확인됨 | 같음. Tier 밖(#2.1 그룹이 같은 자리를 더 강하게 덮는다) |

### 3.2 배제목록 재대조 — 위반 0건

101개 후보 ID를 vendored 69 ID + deep-read 32편과 프로그램 대조했다. **히트 1건**이 나왔으나 파서 오탐이다 — CM의 후보 #11이 제품 *Mem0 Dream*이고 그 설명문 안에 논문 Mem0(`2504.19413`)를 대조 대상으로 언급했을 뿐이다. 6채널의 배제 규율은 지켜졌다.

### 3.3 비-arXiv 후보의 검증 상태

| 항목 | 검증 |
|---|---|
| Anthropic Dreaming (2026-05-06) | **본 검토가 독립 재확인.** 공식 블로그 + 다수 2차 보도. 스케줄 실행, 기존 memory store + 대량 과거 세션 전사 입력, **재조직된 memory layer를 산출해 승인/거부/수정** — 입력 store 비파괴 |
| OpenAI Dreaming V3 (2026-06-04) | **본 검토가 독립 재확인.** 공식 페이지 + 다수 2차 보도. 유휴 시간 백그라운드 통합, in-place canonical 갱신, 시간적 무효화. **시간민감 기억 정확도 9.4%→75.1%(벤더 자기보고, 통제군 없음)** — CM의 "정량 결과 0건"은 정정 대상 |
| Mem0 Dream (2026-08-05) | CM이 공식 블로그 원문 직독. 주 1회 스케줄 synthesis + supersede 포인터 |
| Samsung CMM-D 백서 (2026-06) | CM이 PDF 14pp 전문 직독. **본 검토는 재확인하지 않았다.** ch29에 인용하려면 원문 재확인 필요 |
| Letta 동기 core-memory 쓰기 150–400ms | 2차 출처(운영 블로그). CM 스스로 "본문 수치로는 부적격"으로 표기. **동의** — 인용 불가 |

---

## 4. 버린 것 — 유형별

채널들이 훑은 총량은 약 700편(AX 330 · CG 157 · HF 1,771 census 중 182 수기분류 · GH ~350 목록항목 · SB 서베이 참고문헌 249 + 검색 · CM 뉴스레터·블로그)이고, 후보표에 오른 137건(고유 106)에서 이 단계가 **51건을 더 버렸다.**

| 버린 묶음 | 규모 | 버린 이유 |
|---|---|---|
| **$E$-경로 "또 하나의 기억 시스템"** | 채널 후보 중 ~14, 채널이 이미 버린 것 ~80 | `wr`을 재배열할 뿐 $B_s$·$C$·$\rho$ 중 무엇도 새로 보고하지 않는다. ch15에 행을 하나 더 넣을 뿐 A1 판정이 안 움직인다. Tier 2로 올린 예외는 (a) 학습된 `wr`, (b) 쓰기 시점을 설계변수로 승격, (c) 쓰기 경로 비용 계측 셋뿐 |
| **self-evolving 도메인 응용** | ~200 | 기제가 ReasoningBank/SEAL 재사용. 새 제약도 새 비용도 없다 |
| **skill library / skill evolution** | ~60 | 스킬 = 텍스트 레코드. $E$층 `wr`의 변주 |
| **모델 편집 *기법* 개선** | ~15 | ch19가 ROME·MEMIT로 primitive와 상한을 이미 다룬다. 점진 성능 개선은 Θ 판정을 안 바꾼다. **붕괴의 이론·기제를 주는 것만 남겼다**(Tier 1 #4·#9) |
| **일반 KV cache 압축·양자화·오프로딩** | ~35 | F4처럼 보이나 측정 대상이 **일반 장문 추론**이다. "질의 전 오프라인 갱신 / per-user 지속 상태"를 재지 않는다. 에이전트·멀티턴·per-user에 특정된 것만 남겼다 |
| **wake 시계 TTT·fast-weight 아키텍처** | ~25 | $B_s=0$, 조건 (1) 불충족. NM 모노그래프의 소관이고 ch17·ch18 판정을 안 바꾼다 |
| **에이전트 speculative 실행** | ~7 | 유휴 예산 → 즉시 소모, 지속 상태 없음. 대표 1편(IdleSpec)만 남기고 중복 제거 |
| **서베이·포지션·거버넌스** | ~20 | corpus가 이미 2607.25380을 보유. 새 측정도 새 기제도 없다. 예외 셋은 결함 목록·측정 결손 목록·계측 동반 서베이 |
| **기억 벤치 일반** | ~15 | 판별식을 시험하지 않고 과제 성공률을 잰다. 조건 (4)를 직접 재는 것과 비용까지 재는 것만 예외 |
| **에이전트 기억 보안·공격** | ~8 | 질문이 적대적 견고성이다. 예외 2편(SkillJack·Authority Collapse)은 **공격이 아니라 consolidation의 비가역성**을 보이므로 남겼다 |
| **model collapse 도메인 확장** | ~6 | ch06이 2404.01413으로 이미 반증 축을 세웠다. 공정성·VLM 확장은 sleep 루프의 $\rho$를 재지 않는다 |
| **멀티모달·로봇·비-LLM** | ~25 | 워크로드가 이 책 대상(텍스트 LLM 서빙)이 아니다. Phasor Agents(진동자 그래프)·Echo-Memory(영상 world model)·Frames2LoRA(VLM) 포함 |
| **벤더 자기보고·요약 블로그** | — | 통제군 없음 → 정직성 caveat 기준으로 인용 불가(Harvey 6×, Letta LoCoMo 83.2%, Zep LongMemEval 63.8%, Mem0 stars). **다만 "제품은 나왔는데 통제 수치가 없다"는 사실 자체는 Tier 1 #10에 흡수했다** |
| **증거 등급 미달** | 2 | ZenBrain(2604.23878) — 1인 저자·15개 기제 동시 주장·재현 코드 부재로 corpus 최하위군보다 낮다. Byte-Exact KV Grafting(2607.14431) — 단독저자·엔진 비공개·6,574× 등 이례적 주장, 해시 커밋만 근거 |
| **질의어 다의성 noise** | — | 천체물리 "compact", 수면의학 "sleep", SQL 투기 실행 |

---

## 5. 채널 수율

### 5.1 계량

| 채널 | 후보 | Tier 1 | 그중 **단독 발굴** | Tier 2 | 그중 단독 | 채택률 |
|---|---|---|---|---|---|---|
| **AX** arXiv 직접검색 | 25 | 3 | **2** (2605.24657, 2608.00101) | 13 | 9 | **64%** |
| **CG** 인용그래프 | 25 | 3 | 0 | 11 | 4 | 56% |
| **CM** 커뮤니티·제품 | 18 | 5 | **1** (제품 3종) | 1 | 0 | 33% |
| **GH** GitHub 큐레이션 | 19 | 5 | 0 | 7 | 3 | 63% |
| **HF** HF Papers | 25 | 2 | **1** (2607.12227) | 13 | **12** | 60% |
| **SB** 서베이 참고문헌 | 25 | 5 | 0 | 11 | 5 | **64%** |

(Tier 1 열의 합이 10을 넘는 것은 5채널이 같은 논문을 올렸기 때문이다. 단독 발굴 4 + 다중발굴 6 = 10항목.)

### 5.2 무엇이 값을 했나

- **AX(arXiv 직접검색)가 최고 수율이다.** Tier 1 단독 발굴 2건이 전부 AX이고, 그 둘이 이번 스윕에서 가장 무거운 후보다(#1 통제군 붙은 경로 대결, #6 프로덕션 트레이스). 이유가 분명하다 — `abs:"per-user" AND abs:"LoRA"`, `abs:"amortize" AND abs:"offline"` 같은 **기제 조합 질의**는 큐레이션 목록도 인용그래프도 도달하지 못하는 자리를 짚는다. 그리고 AX의 **0건 결과가 정보였다**: `abs:"memory write" AND abs:"latency" AND abs:"agent"` = 0건, `all:"sleep-time compute"` = 1편. 후자는 §5의 "이름이 기제보다 넓다"를 **"이름이 퍼지지도 않았다"**로 강화한다.
- **HF는 전수 census(1,771편/72일)라서 유일하게 negative 증거를 만든다.** Tier 1은 1편뿐이지만 Tier 2 단독 발굴이 12편으로 최다이고, 무엇보다 **"이 채널에 0편"이라는 확인**(편집 붕괴 0, CXL 0, bytes/user 0)이 §4 caveat를 corpus 밖으로도 방어해 준다. 판정을 바꾸는 채널이 아니라 **caveat를 지켜 주는 채널**이다.
- **CM은 다른 어느 채널도 못 주는 축 하나를 준다** — 제품. Reddit은 전면 차단으로 실패했고 뉴스레터 수확도 얇았지만, 판별식을 통과하는 배포 시스템 셋은 CM 없이는 발견되지 않았다. 그리고 그 축은 논문 채널로는 영원히 안 나온다.
- **SB는 프로그램 검증이 가장 단단했다.** 서베이 4편의 참고문헌을 파싱해 69 ID와 프로그램 대조하고 교집합 0을 확인했다. 그리고 **못 준 것을 정직하게 보고했다** — X3를 채울 논문 0편, W-경로 신규 정면 사례 0편, 경로 상호인용 개선 신호 0건. 그 세 개의 0이 본 검토에서 전부 유지됐다(§5.3).

### 5.3 헛수고에 가까웠던 것 — 그리고 그것도 결과다

- **CG(인용그래프)는 Tier 1 단독 발굴 0건이다.** 채널 스스로 원인을 적었고 그 진단이 맞다: `Do LMs Need Sleep?` 피인용 **0**, Auto-Dreamer **2**, `LM Need Sleep` **3** — W·Θ 앵커에 확장할 재료 자체가 없다. Nested Learning 91편·SEAL 53편은 인용자 대부분이 무관해 정밀도가 낮았고(144편 중 6편), 최고가 후보들(MemDelta·User as Engram·HyMCache)은 **그래프가 아니라 곁다리 WebSearch에서 나왔다**. **다만 이 채널의 진짜 산출은 후보가 아니라 관측이다** — W·Θ-경로 전체가 인용 그래프상 고립돼 있다는 사실이 `PRE-RESEARCH` §3.1 침묵 지도를 **인용받는 쪽에서도** 성립시킨다.
- **GH(GitHub 큐레이션)는 채택률은 높은데 단독 발굴이 0이다.** 14개 목록 중 4개는 신규 0건이었고, 1,700★ `VoltAgent`는 2026-05 이후 항목이 하나도 없어 죽어 있었다. 반대로 9★ `LowEntropyAI`가 ProAct의 유일 출처였고 비용을 카테고리로 가진 목록은 `Snseam/cost-savings-landscape.md` 하나뿐이었다. **별 수와 유용성이 반비례한다.** 그리고 채널 스스로 확인한 대로 **Θ-경로 반증 축은 이 채널에 존재하지 않는다**(awesome model editing 검색 0–2건, 전부 이미지 편집).
- **CM의 Reddit 경로는 완전 실패했다**(www/old/redlib 전부 차단·403). `site:reddit.com` 우회로 나온 것은 스레드가 아니라 스레드가 인용하던 arXiv 논문이었다. 다음 라운드에서 Reddit에 시간을 쓸 이유가 없다.

---

## 6. 다음 라운드 권고

1. **AX부터 돌린다.** 기제 조합 `abs:` 질의가 단위 시간당 가장 무거운 것을 준다. 이번에 안 돌린 조합: `abs:"full context" AND abs:"ceiling" AND abs:"memory"`(통제군 보유 논문 사냥), `abs:"consolidation" AND abs:"rounds" AND abs:"degrad*"`, `abs:"per-user" AND abs:"bytes"`, `abs:"idle" AND abs:"predictor" AND abs:"serving"`.
2. **CG의 앵커를 교체한다.** Letta STC·`Do LMs Need Sleep?`·`LM Need Sleep`는 피인용이 35/0/3이라 더 나올 것이 없다. 새 앵커는 이번 Tier 1 자체다 — **2606.04703 · 2605.11836 · 2606.06448 · 2605.24657**. 이 넷의 인용자·피인용자가 다음 라운드의 이웃이다.
3. **저자 추적을 새 채널로 연다.** 이번 후보에서 같은 1저자가 복수로 나온 사례가 셋이다 — Beining Wu(2606.25115, 2608.00303), Youwang Deng(2606.11712, 2605.29630), Zahn(2601.15313, 2607.16256). 그리고 GH·HF 둘 다 **아키텍처 레벨 F4와 편집 용량 F3는 자기 채널에 없다**고 보고했다. 저자 추적 + 학회 프로시딩(MICRO/ASPLOS/ISCA/MLSys/OSDI)이 그 두 축의 유일한 경로다.
4. **GH는 목록 하나만 다시 본다** — `Snseam/cost-savings-landscape.md`. 나머지 13개는 $E$-경로 중복이고 90% 겹친다. 그리고 **큐레이션 목록의 arXiv ID를 믿지 않는다**(GH가 확인한 것 중 1건이 입자물리 논문이었다).
5. **CM은 Reddit을 버리고 벤더 1차 소스로 간다** — 릴리스 노트·가격 페이지·API 문서. Anthropic Dreaming이 **표준 토큰 요율로 과금된다**는 사실은 $B_s$를 청구 가능한 양으로 만드는 유일한 외부 근거이고, 가격 페이지는 논문보다 정확하다.
6. **다음 스윕 전에 확정할 두 ID.** `OUTLOOK-REVIEW` 한계 6이 지목한 공백 — **LightMem**과 **UMEM**. Auto-Dreamer가 직전 계보로 지목하는데 corpus에 없고, LightMem이 "online writer + periodic offline consolidation"이면 판별식을 정면 통과해 $E$-경로 계수가 또 바뀐다. 어느 채널도 이 둘의 ID를 확정하지 못했다.
7. **Tier 1을 읽는 순서를 정한다.** #3·#4(ρ의 부호 충돌) → #1·#2(통합 연산의 손실) → #5·#6(회계) → #7·#10(통과군·제품) → #8·#9. **#3과 #4는 반드시 붙여서 읽는다** — 따로 읽으면 각각이 결론처럼 보이고, 붙여 읽어야 X4의 질문이 "열화하는가"에서 "무엇이 있으면 열화하지 않는가"로 바뀐다.
