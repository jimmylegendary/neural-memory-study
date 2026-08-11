# gap-sweep — 채널: Hugging Face Papers

**작성**: 2026-08-11 · **범위**: 2026-06-01 ~ 2026-08-11 (72일 전수) · **판정 기준**: `S0-DESIGN.md` §2.3 판별식 + F1–F4
**배제**: `corpus-exclusion.md`의 노트 32편 + vendored arXiv ID 69개

---

## 1. 방문한 소스

### 1.1 일자별 전수 census (표본 아님)

`https://huggingface.co/papers?date=YYYY-MM-DD` 를 **2026-06-01부터 2026-08-11까지 72일 전부** 수집했다.
지시는 주 2~3일 표본이었으나, 페이지 수집 비용이 낮아 표본 대신 전수를 택했다(누락 위험 제거).

| 항목 | 값 |
|---|---|
| 수집 일자 | 72일 (2026-06-01 … 2026-08-11), 결번 0 |
| 수집 레코드 | 2,415 |
| 고유 논문 | **1,771** |
| 일평균 | 33.5편 (최소 13, 최대 72) |

수집 방식: 각 날짜 페이지의 `data-target="DailyPapers"` 노드에 실린 `data-props` JSON을 파싱해
`id`(=arXiv ID) · `title` · `summary`(초록 전문) · `publishedAt`을 추출. 제목이 아니라 **초록 본문**으로 걸렀다.

### 1.2 월별 뷰 — 사용 불가

`https://huggingface.co/papers/month/2026-07` 은 헤더와 구독 배너만 반환하고 논문 목록을 싣지 않는다.
일자별 뷰가 이 채널의 유일한 열거 경로다.

### 1.3 `?q=` 검색 엔드포인트 — **작동하지 않음 (중요)**

`https://huggingface.co/papers?q=<질의>` 는 HTTP 200에 140개 링크를 반환해 검색처럼 보이지만,
**질의와 무관하게 항상 동일한 최신 20편을 돌려준다.** 앞 단계가 세운 학계 용어 20개를 각각 넣어
전부 확인했다: 400개 결과행의 고유 ID가 **20개**, 전부 2026-08 발행.
`q=sleep-time compute` 의 1위가 `SWE-Bench ProMax`(코드 리팩터링 벤치)인 것으로 확정했다.

→ **이 채널에는 키워드 검색이 없다.** 일자별 전수 census가 유일한 커버리지이며, 위 census가
2026-06-01 이후 구간을 빠짐없이 덮으므로 이 채널에 대한 탐색은 완결됐다.

### 1.4 후보 검증

최종 후보 25편은 전부 `https://huggingface.co/papers/<id>` 개별 페이지를 열어
**ID 해석 · 제목 · 저자 · 발행일을 확인**했다(25/25 OK). 추측으로 만든 ID는 없다.
arXiv API(`export.arxiv.org`)는 이 환경에서 타임아웃이라 HF 논문 페이지를 검증 경로로 썼다.

### 1.5 필터 재작업 이력 (자체 결함 기록)

1차 필터는 `memory`를 보조어(Tier-B)로 두어 190편만 남겼고, **회수율이 심각하게 낮았다**
(`Metis: Memory Foundation Model`, `Memory Decoder at Scale`, `Zero-Mem`, `InMind`, `MEMPROBE`,
`GateMem`, `Code2LoRA`, `Video2LoRA` 등 최상위 후보가 전부 탈락). census 1,771편 전체에
회수율 점검 정규식을 다시 걸어 717편을 재추출하고, 그중 제목이 강한 182편을 수기 재분류했다.
**아래 후보의 과반은 1차 필터가 놓친 것이다.**

---

## 2. 후보 (25편)

★ = 이 책의 판정을 실제로 바꿀 수 있다고 보는 것. 전부 초록 본문을 읽고 판정했다.

### 2.1 F1 — 판별식 통과 후보 (통과군 9편이 늘어난다)

| # | arXiv | 제목 / 저자 / 날짜 | F | 왜 판정을 바꾸는가 |
|---|---|---|---|---|
| 1 ★ | **2607.19604** | Scaling Laws for Hypernetwork-Based Knowledge Injection in LLMs — Nischay Dhankhar, Dos Baha 외 · 2026-07-21 | F1+F3 | 질의 전에 사실 코퍼스를 하이퍼네트워크 1회 forward로 **고정 LoRA에 써 넣고** 이후 질의에 쓴다 — 네 조건 전부 충족이며, corpus에 전무한 **쓰기 쪽 scaling law**(깊이·폭·타깃 크기 축 멱법칙 + OOD 일반화)를 준다. ch28/X3의 유일한 실측 공급원이고, ch22의 Θ-반증 축에 정면으로 맞선다. |
| 2 ★ | **2606.06492** | Code2LoRA: Hypernetwork-Generated Adapters for Code LMs under Software Evolution — Liliana Hotsko, Yinxi Li 외 · 2026-06-04 | F1 | **per-repo 파라메트릭 기억을 GRU 은닉상태로 유지하며 커밋 diff마다 증분 갱신**한다(Evo 트랙 215K 커밋). A2("per-user/per-session 가중치 delta가 상태량을 지배")가 사변이 아니라 실측 대상이 되는 첫 사례. 추론 시 토큰 오버헤드 0. |
| 3 ★ | **2606.04351** | Video2LoRA: Parametric Video Internalization for VLMs — Manan Suri, Sarvesh Baskar 외 · 2026-06-03 | F1+F4 | ch16(Generative Adapter, 경첩 장)의 후속에 해당하는데 **비용 4종 중 $L_w$를 실제로 보고한다**: 답변 시 시각토큰 부하 최대 1,500× 감소, **TTFT 6–80× 단축**. 게다가 비중첩 구간의 어댑터가 rank 공간에서 **합성된다** — Θ 쓰기의 결합법칙 문제를 처음 건드린다. |
| 4 ★ | **2607.21051** | Sample-Efficient Learning from Agent Experience — Chenhui Gou, Haoqin Tu 외 · 2026-07-23 | F1 | 경험을 문맥에서 가중치로 옮길 때 **ICL 이득의 64.8%가 남고 단순 SFT는 3.8%만 남는다**(SWE 749과제 + 텍스트게임 6종). E-경로 이득이 Θ-경로로 얼마나 넘어가는지를 같은 과제에서 잰 **최초의 경로 간 환산 계수** — ch27 경로별 판정의 빈칸이 정확히 이것이다. |
| 5 | **2608.02508** | RoMeRL: …Memory-Reward Trap in Self-Evolving Agent Memory — Yi Yang, Zhennan Chen 외 · 2026-08-04 | F1+F4 | corpus의 `wr`은 전부 수기 규칙(Letta=파괴적 덮어쓰기, Mem0=4연산, RB=순수 합집합)인데 이건 **학습된 쓰기 정책**이고, 공동검색된 기억에 보상이 잘못 귀속되는 **memory-reward trap**을 기제로 짚는다. 유지 기억량 84.4%↓·LLM 호출 21.1%↓로 쓰기 비용까지 보고. |
| 6 | **2607.05155** | EdgeBench: Unveiling Scaling Laws of Learning from Real-World Environments — Deyao Zhu, Xin Zhou 외 · 2026-07-06 | F1+F4 | 실환경 에이전트 상호작용 **38,000시간**에서 배치 후 경험학습 성능이 **log-sigmoid 법칙(R²=0.998)** 을 따른다 — 즉 **포화한다**. A3의 "$B_s$ 지수와 포화점"에 대응하는 유일한 대규모 실측이며, 포화가 사실이면 ch27의 낙관은 상한을 갖는다. |

### 2.2 F2 — 경계·회색지대 (판별식 자체를 시험한다)

| # | arXiv | 제목 / 저자 / 날짜 | F | 왜 판정을 바꾸는가 |
|---|---|---|---|---|
| 7 ★★ | **2607.23693** | Compute Globally, Materialize Locally: The Memory Contract of Sparse Event-KV — Zefeng Cai, Zerui Cai · 2026-07-26 | F2+F3 | **Θ·W·E 어디도 아닌 KV 행 자체에 의도적으로 쓴다**(답을 명시하지 않은 문장 하나로 donor-정렬 복원 6%→51%). 동시에 **압축 상태는 살아남고 큰 payload는 우연 수준으로 붕괴**한다는 용량 상한을 준다. 결정타는 방법론: *"원본 이벤트를 지웠는데 정확도가 안 떨어졌다고 해서 그 원본이 불필요했다는 증명은 아니다"* — eviction·compaction 문헌 전체(ch18, rate–distortion 포함)의 ablation 논법을 무효화한다. |
| 8 ★ | **2607.26760** | Metis: Memory Foundation Model — Zeyu Zhang, Ziliang Guo 외 · 2026-07-29 | F2 | 백본 안에 **지속·진화하는 native 기억 상태**를 두고, 갱신은 **gradient-free·forward 1회**, 추론 시 가중치는 전부 동결. 정확히 ch16 경첩(GenAdapter)의 구조인데 mid-training으로 기억 절차 자체를 심었다. W-경로 정면 사례가 corpus에 **단 1편**(2605.26099)뿐인 상태를 바꾼다. |
| 9 ★ | **2607.27919** | Memory Decoder at Scale: A Pretrained, Parametric Long-Term Memory — Rubin Wei, Jiaqi Cao 외 · 2026-07-30 | F2 | 파라메트릭 기억 모듈을 **6.9B / 300B 토큰**까지 키워 "기억에 파라미터를 더 주는 편이 백본을 키우는 것보다 낫다"를 보인다(6.9B 기억+Pythia-410M = 37.34 > Pythia-12B 37.24, 총 파라미터 39% 적음). ch09 용량 장의 Memory Layers 데이터를 한 자릿수 넘겨 갱신한다. |
| 10 ★ | **2607.14431** | Byte-Exact KV-Cache Grafting Turns a Frozen Small Model into a Verified-Knowledge Flywheel — Sietse Schelpe · 2026-07-15 | F1+F2+F4 | 가중치를 안 바꾸고 **검증된 지식을 byte-exact KV 아티팩트로 한 번 적립해 이후 문맥에 이식**한다 — 상각식 (A)의 교과서적 사례이고 수치가 극단적이다(재발 문항 401,026토큰 → 61토큰 = **6,574×**, 사용가능 문맥 32,768→2,854,766, 추가 가속기 메모리 0). ⚠ **신뢰도 경고**: 단독 저자, 엔진 비공개(proprietary), 주장 크기가 이례적. 해시 커밋만 근거다 — 인용하려면 별도 검증 필수. |

### 2.3 F3 — 제약·반증 (기제에 거는 한계)

| # | arXiv | 제목 / 저자 / 날짜 | F | 왜 판정을 바꾸는가 |
|---|---|---|---|---|
| 11 ★★ | **2607.24368** | Keep It InMind: Benchmarking the Implicit-Association Blind Spot in Agent Memory — Ruizhe Li, Mingxuan Du 외 · 2026-07-27 | F3 | 통제된 짝 대조로 **결정적 기억을 문맥에 넣으면 84.0%, 같은 기억을 검색하게 하면 6개 기억시스템 최대 14.4%** — 같은 사실을 요구하면 100%까지 회상하는데도 그렇다. 임베딩 차원 8배로도 격차가 안 줄어 **용량 문제가 아님을 배제**한다. 이 책의 "품질 1위가 반복적으로 기억 없음"이 일화가 아니라 **인터페이스의 구조적 실패**임을 확정한다. |
| 12 ★★ | **2606.04703** | Rethinking Continual Experience Internalization for Self-Evolving LLM Agents — Jingwen Chen, Wenkai Yang 외 · 2026-06-03 | F3 | 다회 반복 경험 내재화에서 기존 방법들이 **누적 개선이 아니라 점진적 능력 붕괴**를 일으킨다. 이것이 곧 $\rho$이고 **X4(반복 consolidation 열화)가 묻는 바로 그 질문에 대한 외부 실측**이다. 입자도(principle > instance)·주입 패턴(step-wise > global)·레짐(off-policy > on-policy)으로 붕괴 조건까지 분해한다. |
| 13 ★★ | **2607.12227** | Rethinking the Evaluation of Harness Evolution for Agents — Yike Wang, Huaisheng Zhu 외 · 2026-07-14 | F3 | **피드백·추론 예산을 맞춘 비교**에서 자동 harness 진화가 단순 test-time scaling을 일관되게 못 이기고 일반화도 제한적이다(Terminal-Bench 2.1 / GPT-5.4 / Claude Opus 4.6). 게다가 탐색과 최종 평가가 같은 벤치를 공유하는 과적합을 지적한다. **"$B_s$를 써서 얻은 이득이 그냥 $B_t$를 더 쓴 것보다 나은가"** — ch28의 핵심 질문에 대한 부정 증거. |
| 14 ★★ | **2606.24595** | MEMPROBE: Probing Long-Term Agent Memory via Hidden User-State Recovery — Enze Ma, Yufan Zhou 외 · 2026-06-23 | F3 | 기억을 **사후 감사 가능한 아티팩트**로 직접 복원 평가한다(50 사용자 × 31 은닉차원 = 1,550 타깃, 5개 기억시스템). 결과: **과제 완수는 memoryless baseline에서도 포화**하는데 범주균형 복원은 ~0.6에 머문다 — "성공했다"가 "기억이 작동했다"의 증거가 아님을 분리 증명. §4 caveat 표를 일반 명제로 승격시킨다. |
| 15 ★ | **2608.04570** | The Personalization Mirage: How LLMs Fabricate User Profiles… — Yushi Sun, Yanjie Zhang 외 · 2026-08-05 | F3 | 12개 모델 **전부**가 사용자 속성 주장의 **35–49%를 근거 없이 날조**하고(143,616 판정, κ=0.863), 다회차에서 **거의 선형으로 누적되며 수정되지 않는다.** 게다가 자기평가와 실측이 **음의 상관**(ρ=−0.60). E-경로의 `wr` 입력 자체가 오염돼 있다는 뜻 — 검색 품질을 논하기 전 단계의 반증. |
| 16 ★ | **2607.08393** | Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in LLM Finetuning — Lu Dai, Ziyang Rao 외 · 2026-07-09 | F3 | **Knowing–Using Gap**: 가중치에 사실을 넣는 데는 성공해도 하류 추론에 쓰지 못하며, 정확도 격차와 **시간 지연**이 함께 나타난다. self-patching으로 "기억은 있으나 계산에 유효한 층으로 라우팅되지 않는다"는 회로 오정렬 가설을 세운다. ch22(2607.11020)의 행동적 반증에 **기제적 짝**을 붙인다. |
| 17 ★ | **2606.18829** | GateMem: Benchmarking Memory Governance in Multi-Principal Shared-Memory Agents — Zhe Ren, Yibo Yang 외 · 2026-06-17 | F3 | 다주체 공유 기억에서 **효용·접근통제·명시적 삭제 후 망각을 동시에 만족하는 방법이 하나도 없다**. 그리고 또: 거버넌스 점수 1위가 **장문맥 프롬프팅**(=기억 시스템 없음, 대신 토큰 비용 높음)이고, 외부기억 계열은 비용은 줄이되 **삭제된 정보를 여전히 흘린다**. 배치 가능성 판정(ch27)의 새 축. |
| 18 ★ | **2608.03509** | SkillJack: Persistent Skill Backdoors in Self-Evolving Agents — Zonghao Ying, Xiangfan Wu 외 · 2026-08-04 | F3 | 경험→스킬 consolidation이 **출처를 세탁**한다: 안전 탐지율이 오염 궤적 98.5% → 추출된 스킬 **11.4%** 로 붕괴하고, **80.0%가 원본 레코드를 지운 뒤에도 살아남는다.** (U-E)의 `wr`이 비가역적 승격이라는 뜻 — 삭제·감사·규제 관점에서 Θ/E 경로에 거는 실질 제약. |
| 19 ★ | **2607.26637** | Filesystem-Based Memory for LLM Agents: Organization, Evolution, and Sustainability — Sizhe Zhou, Sheldon Yu 외 · 2026-07-29 | F3+F4 | 실배치 기본값(마크다운 디렉터리 트리)의 **암묵 가정 둘을 처음으로 통제 검증**한다: 성장하는 저장소를 정리할 수 있는가, 그 정리가 값을 하는가. **verbatim dump 대조군**을 두고 품질·**비용**·store health를 함께 추적하며, 정리가 확실히 사주는 것은 "검색 경제성(대형에서 검색비용 약 절반)"뿐이라고 답한다. |
| 20 ★ | **2605.30621** | Harness Updating Is Not Harness Benefit — Minhua Lin, Juncheng Wu 외 · 2026-05-28 | F3 | **쓸모 있는 갱신을 만드는 능력은 기반 능력에 대해 평평하다** — Qwen3.5-9B의 갱신이 Claude Opus 4.6의 갱신과 비슷한 이득을 낸다. 반면 그것으로 이득 보는 능력은 **비단조**(중간 티어 최대). 오프라인 갱신 품질이 병목이 아니라는 뜻이라 세 경로 전부의 투자 논리를 뒤집는다. |
| 21 ★ | **2607.01763** | Denser ≠ Better: Limits of On-Policy Self-Distillation for Continual Post-Training — Meng Wang, Haohan Zhao 외 · 2026-07-02 | F3 | 지속 post-training에서 self-distillation이 **망각을 키우고 붕괴하기까지** 하며(GRPO가 오히려 보수적으로 보존), 조밀할수록 파라미터·응답 공간 drift가 커지고 **자기강화 교사–학생 루프로 고주파 형식 인공물이 증폭**된다. ch06(model collapse)를 자기생성 $\mathcal{R}$의 지속학습 레짐으로 확장하는 직접 증거. |

### 2.4 F4 — 비용·시스템 실측 (이 책의 최대 공백)

| # | arXiv | 제목 / 저자 / 날짜 | F | 왜 판정을 바꾸는가 |
|---|---|---|---|---|
| 22 ★★ | **2605.30571** | Memory-Bound but Not Bandwidth-Limited: The Physical AI Inference Gap in Batch-1 LLM Decode — Josef Chen · 2026-05-28 | F4 | 4종 GPU(H100/A100/L40S/L4) × 3모델 × 문맥 2048–16384의 **44셀 실측**: **피크 대역폭이 높을수록 달성률이 떨어진다**(L4 81% vs H100 27%). CUDA Graphs A/B로 launch-side 항을 분리(H100 1.259×, CI [1.253,1.267] / L4 1.028×). **"바이트를 줄이면 지연이 비례해 준다"는 전제가 batch-1에서 거짓** — ch29·A2의 memory-centric 논증이 통과해야 할 관문. (단독 저자, 소규모지만 프로토콜은 통제됨) |
| 23 ★★ | **2607.29377** | Zero-Mem: Zero-Token Memory Operations for LLM Agents — Yilin Xiao, Zhehan Zhu 외 · 2026-07-31 | F4 | **쓰기 경로의 비용을 정면으로 측정하고 0으로 만든다** — 최종 QA 외 어떤 단계도 LLM을 호출하지 않으며 **기억연산 시간비용을 최속 baseline 대비 57.6% 절감**. corpus 전체가 `wr`의 대가를 보고하지 않는데(ch25 공백), 이 논문은 그 대가가 지배적이었음을 역으로 증명한다. |
| 24 ★ | **2608.08097** | OasisKV: Scaling In-Decode KV Cache Beyond HBM with Lookahead Sparse Prefetching — Can Xiao, Sukmin Cho 외 · 2026-08-08 | F4+F2 | vLLM 위에서 **HBM 밖 계층(host/remote)으로 KV를 내리고 speculative lookahead로 미리 stage** 한다 — 처리량 1.69–2.1×, 요청당 KV 6.5–9.7× 감소, 디코드 노드 host 메모리 2.2–2.6× 감소. X2의 **웜 계층 delta 스테이징**과 같은 자리를 실제 시스템에서 잰 값이고, "질의 전에 상태를 옮겨 둔다"는 점에서 경계 사례이기도 하다. |
| 25 ★ | **2607.19058** | Where Should Optimizer State Live? Tiered State Allocation for Memory-Efficient MoE Training — Nuemaan Malik · 2026-07-21 | F4 | **6.78B MoE에서 12.6GB 가중치를 갱신하려고 AdamW가 50.6GB 상태를 든다**(=델타의 약 4배). Θ-경로가 per-user/per-session 가중치 갱신을 하려면 이 옵티마이저 상태가 진짜 용량 항인데 X2가 세지 않았을 항목이다. 계층 배분으로 1.29GB(2.6%)·피크 81.4→31.3GB까지 줄인 실측을 함께 준다. (단독 저자) |

---

## 3. 버린 것과 이유

### 3.1 초록까지 읽고 버린 것 (주요 항목)

| arXiv | 제목 | 버린 이유 |
|---|---|---|
| 2608.07110 | Modular TTT | 갱신이 전부 wake-time 온라인 학습이라 $B_s=0$. 조건 (1)·(2) 동시 불충족 → STC 아님(NM 계열 자료). |
| 2607.15275 | RoboTTT | 위와 동일(wake-time fast weight). 로보틱스 도메인이고 바이트 미보고. |
| 2607.09415 | Self-Guided TTT for Long-Context | 적응이 **질의를 안 뒤** 일어나 조건 (1) 불충족. 다만 "무작위 span TTT는 성능을 **떨어뜨린다**"는 쓰기 품질 민감도는 Θ/W 경로 제약으로 기록해 둘 값어치가 있음(2차 후보). |
| 2606.02461 / 2608.00155 / 2608.03874 / 2608.04003 | AgentCL, AgentStream, ContinualSkillBench, PAST-Bench | 전부 양질의 F3 평가 논문이나 §2.3의 InMind·MEMPROBE·Harness-Evaluation과 **같은 명제를 더 약하게** 말한다(ContinualSkillBench의 "ICL ≈ 명시적 스킬 유지", PAST-Bench의 경험 on/off 통제는 특히 아깝다 — 2차 후보로 승계 권장). |
| 2605.29463 / 2608.04574 / 2607.01071 / 2606.24428 | Honest Lying, When Memory Lies, MemSyco-Bench, EDV | 같은 계열(기억이 **해롭다**)의 좋은 증거. 특히 When Memory Lies의 "raw memory를 믿은 에이전트가 **기억 없는 같은 에이전트보다 2배 이상 자주 죽는다**"와 Honest Lying의 "16개 환경에서 121개 reflection 중 정답 객체 언급 0개"는 §2.3 17·19번과 중복도가 높아 25편 상한에서 밀렸을 뿐, 폐기가 아니다. |
| 2605.30159 / 2605.31075 / 2607.01224 / 2606.03197 / 2607.28272 | MMPO, TaskMem, AutoMem, MemTrain, MemHarness | **학습된 쓰기 정책** 군집. RoMeRL(§2.1-5)을 대표로 남기고 나머지는 접었다. MMPO의 "재귀 요약이 점진적으로 정보를 버리고 semantic noise를 넣는다"와 TaskMem의 "배치 **후** 어댑터 튜닝으로 무엇을 기억할지 학습"은 각각 $\rho$·F1 근거로 2차 후보. |
| 2606.29961 / 2608.07169 / 2606.17628 / 2605.27276 / 2608.09819 | DuoMem, Agent Memory Distillation, OPD-Evolver, SIA, Macaron-V1 | 전부 F1 통과 가능. DuoMem은 사전계산 교사 기억이 **"수 MB"** 라는 드문 바이트 진술 + 3× 벽시계 이득이 있고, OPD-Evolver는 E-쓰기와 Θ-증류를 **한 시스템에서 잇는다**(경로 간 침묵 지도의 반례!). 상한 때문에 밀렸다 — **2차 후보 1순위**. |
| 2606.06302 / 2606.09079 / 2606.04511 | Tangram, FlashMemory-DS-V4, SparDA | 좋은 F4 실측(Tangram: 멀티턴 KV가 **모델 가중치를 넘어선다**, 오프라인 50샘플 보정; FlashMemory: 500K에서 물리 KV 90%↓ + LongMemEval; SparDA: PCIe prefetch, 5.3× 처리량). OasisKV와 층위가 겹쳐 대표 1편만 남겼다. |
| 2607.21503 | Agentic Context Management | 토큰비용 점근(누적=2차, 조잡한 요약=선형+정확도 절벽, 검증된 compaction=선형)은 ch25에 쓸모 있으나 **벤더 자사 구현 홍보 문서**이고 92%/93.2%가 자사 설정 수치. 독립 검증 없이 인용 불가. |
| 2606.24775 | Are We Ready For An Agent-Native Memory System? | 12개 기억시스템 × 11 데이터셋 비교 + 비용-성능 트레이드오프로 F3+F4 자격은 충분하나, 결론("단일 아키텍처 지배 없음", "국소 유지가 전역 재구성보다 저렴")이 §2.3 16·21번과 겹친다. **2차 후보 2순위.** |
| 2607.25614 | MemSFT: …External Parametric Memory | 백본 동결 + 검색기를 모사하도록 학습한 **별도 파라메트릭 기억**을 디코딩 단계마다 라우터로 융합 — Θ·W·E 어느 것도 아닌 네 번째 형태라 F2 자격은 있으나, 같은 역할을 §2.2-8·9가 더 큰 규모로 한다. **2차 후보 3순위.** |
| 2608.06216 | Continual Learning in Transition | When/How/**Where**(내부 파라미터 vs 외부 구조) 3축 서베이 — 이 책 3층 분할과 겹치는 **경쟁 분류 체계**가 5일 전에 나왔다. 기제·측정이 없어 후보에서 뺐으나, **ch11·ch24가 자기 판별식을 이것 대비 위치시키지 않으면 공백이 된다**(집필 시 필독). |
| 2607.07386 | Sparse Delta Memory | W층 용량-회상 isoFLOP 데이터로 ch18/ch26에 유용하나 오프라인 예산이 없어 판별식 밖. |
| 2606.09803 | Echo-Memory | 방법론은 모범적(백본·옵티마이저·샘플러 고정, 저장/읽기만 변주해 capacity·compression·read-out·recurrence 분리, "raw context가 강한 baseline"). 도메인이 action world model(영상)이라 이 책의 워크로드가 아님. |
| 2607.02255 | AgenticSTS | "기억 = 미래 결정이 무엇을 볼 수 있는지에 대한 계약"이라는 틀은 좋으나 표본이 10판 규모라 저자 스스로 directional이라고 씀. |
| 2606.04056 | Token Budgets (63 incidents) | 실비용 사고 카탈로그지만 폭주 루프 거버넌스 문제이지 sleep/wake 상각 회계와 무관. |
| 2608.09096 | Evo-Bench | harness 진화를 긍정 평가(최대 +16.6점)해 §2.3-15와 **정면 배치**된다. 문헌이 갈린다는 사실 자체는 ch27에 유용하나, 예산 통제가 15번만큼 엄격하지 않아 밀었다. |
| 2606.14502 / 2605.26112 / 2607.13104 | Digital Colleague, System Scaling, Self-Improvement Survey | 입장·서베이 문서로 측정 없음. (단 2607.13104의 "self-induced update operator" 형식화는 §2.2-12와 함께 경쟁 분류 체계로 기록.) |
| 2606.06036 / 2607.11523 / 2608.02392 / 2607.05511 / 2607.10350 | MRAgent, Vinci2, GROVE, Light-Omni, ABot-AgentOS | 검색 시점 추론 개선이거나 영상/로보틱스 도메인. 판별식·비용표 어느 쪽도 못 바꾼다. |
| 2606.09707 | BrainSurgery | 체크포인트 조작 도구. 기제·제약·비용 어느 것도 아님. |
| 2608.09119 / 기타 기술보고서 | Motif 3 등 | 일반 LM 사전학습 보고서. 주제 밖. |

### 3.2 채널 자체의 공백 — **이것도 결과다**

앞 단계가 준 검색어 중 아래 축은 이 채널 2026-06~08 구간에 **사실상 존재하지 않는다.**
census 1,771편 전문에 정규식을 걸어 확인했다.

| 축 | census 히트 | 판정 |
|---|---|---|
| sequential/lifelong **model editing collapse**, "몇 번의 편집까지" | **0편** (오탐 1건뿐) | ch19·ch22의 편집 용량 상한은 이 채널에서 보강 불가 |
| **LoRA/adapter 병합 간섭·용량** | 유효 0편 (weight-space merging 1편은 전문가 읽기 예산 문제) | A2의 다중 delta 축적 논증 보강 불가 |
| **CXL / 메모리 계층 / PIM** LLM 서빙 | **0편** (7건 전부 오탐) | ch29 memory-device 논증은 이 채널로 못 채운다 |
| **bytes/user·bytes/session 용량 보고** | **0편** | "corpus 전체에서 상태 용량을 바이트로 보고한 논문 0편"이라는 §4 caveat가 **2026-08까지도 유효**하다 — 유일한 근접 진술이 DuoMem의 "수 MB"뿐 |

→ **판정**: HF Papers는 에이전트·평가·멀티모달에 강하게 편향된 채널이다. F1·F2·F3와
**서빙 레벨 F4**(KV/오프로딩)는 풍부하지만, **아키텍처·메모리 소자 레벨 F4와 편집 용량 F3는
이 채널에 없다.** 그 둘은 arXiv cs.AR/cs.DC나 학회(MICRO/ASPLOS/ISCA/MLSys) 채널로 따로 훑어야 한다.

---

## 4. 이 sweep이 §4 caveat 표에 거는 것 (요약)

- "corpus 전체에서 상태 용량을 바이트로 보고한 논문이 0편" → **2026-08까지 유효**(§3.2).
- "상각식 (A)는 $N_q$를 안다고 가정, 실서빙 $N_q$ 분포 보고 없음" → 여전히 유효. 다만
  2607.14431이 $N_q$ 대신 **재발 문항의 토큰 감소비 6,574×** 라는 다른 형태의 상각 증거를 내놓았다.
- "$E$-경로에서 품질 1위가 반복적으로 기억 없음" → **강화된다.** 2607.24368(84.0 vs 14.4),
  2606.24595(memoryless가 과제 완수 포화), 2606.18829(장문맥이 거버넌스 1위)가 독립적으로 재현.
- "세 경로가 서로를 인용하지 않는다" → **반례 등장.** 2606.17628(OPD-Evolver)은 E-쓰기와 Θ-증류를
  한 시스템에서 잇고, 2607.21051은 두 경로 사이 **환산 계수(64.8% vs 3.8%)** 를 실측했다.
  ch27은 "단절"을 시제 있게(2026 중반부터 이어지기 시작) 서술해야 한다.
- **새로 붙일 caveat 후보**: 반복 갱신은 붕괴한다(2606.04703, 2607.01763) · 예산을 맞추면
  오프라인 진화가 test-time scaling을 못 이긴다(2607.12227) · 쓰기 입력 자체가 35–49% 날조다
  (2608.04570) · 소거 실험으로 기억의 필요성을 증명할 수 없다(2607.23693).
