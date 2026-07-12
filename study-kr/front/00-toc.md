# 목차

여섯 편의 test-time memorization / neural-memory 논문([Titans], [Miras], [Atlas], [TNT], [NL], [Sleep])을 하나의 연구 프로그램으로 읽고, 그 완성형을 systems 엔지니어의 도구로 계량한다. 척추 명제는 **D4 workload-split pair thesis**(18장). 페이지 번호는 빌드가 채운다.

## 앞부분 (front matter)

- **서문** — 누구를 위한 책인가, 왜 이 여섯 편인가, 어떻게 읽는가(training 무경험 독자를 위한 우회로 포함), pair thesis 예고, 정직성 계약, 표기와 3층 구분.
- **목차** — 이 문서.
- **Part I 도입** — 배경 장들이 어떻게 이어지는가.
- **Part II 도입** — 여섯 편 정밀 독해의 8절 격자와 서사.
- **Part III 도입** — 원저 기여의 전개(척추 명제는 18장에 위임).

## Part I — inference 엔지니어를 위한 배경

training이라는 낯선 세계를 독자의 inference 어휘로 건설한다. 모든 개념은 (state, update, cost) 객체로 도입되고, 각 장은 systems bridge를 통과한다.

- **1장 · Orientation: inference 세계와 learning 세계의 Rosetta Stone** — KV cache/linear-RNN state를 memory write/read로 재서술하고 두 loop($W$ vs $\Theta$)와 기준 수식 (M)을 세운다.
- **2장 · Training as a system: backprop과 optimizer를 (state, update, cost) 객체로 읽는다** — backward pass와 momentum·Adam·weight decay·Muon을 상태 기계로 도입한다(이 독자의 최대 공백).
- **3장 · Online learning: OGD, regret, FTRL** — inner loop가 최소화하는 것이 무엇인지, regret과 FTRL 관점으로.
- **4장 · Meta-learning과 bilevel optimization: inner loop vs outer loop** — outer loop가 학습하는 것($W_{\mathrm{init}}$, data-dependent gate)과 MAML 계열의 형식화.
- **5장 · Associative memory: Hopfield에서 delta rule까지** — key→value 연상 저장, crosstalk, 고전 capacity.
- **6장 · Linear attention과 fast-weight programming: DeltaNet, Gated DeltaNet, Longhorn, RWKV-7** — 고정 크기 행렬 memory와 FWP 계보.
- **7장 · SSM 계보: S4 → Mamba → Mamba-2, 그리고 SSD duality** — state-space 계열과 state-space duality.
- **8장 · TTT lineage: test-time adaptation에서 TTT-Linear/TTT-MLP와 dual form까지** — 이 라인의 직계 원형과 dual form.
- **9장 · Chunkwise-parallel training: 하나의 일반 scheme, 네 개의 인스턴스** — stale-snapshot 근사와 "chunk = semantic knob" 명제; 라인 전체의 실행 기반.
- **10장 · 통합 systems bridge: cost model cheat sheet, roofline vs chunk size, glossary** — 흩어진 cost model을 한 장으로 묶은 참조점.
- **11장 · Continual learning, complementary learning systems, distillation, RL-lite** — [Sleep] 직전의 배경.

## Part II — 여섯 편의 정밀 독해

각 장은 8절 격자(bridge-in → 문제의식 → 통일 표기 core mechanism + 대응표 → outer vs inner → ledger delta → 실험·스케일 → systems 함의 → bridge-out)를 따른다. 모든 수식은 통일 표기로 (M)에 환원되고, 정직성 caveat이 본문에 박혀 있다.

- **12장 · Titans: Learning to Memorize at Test Time** — deep memory + GD-with-momentum-and-decay가 sequence layer; MAC/MAG/MAL.
- **13장 · Miras: It's All Connected — sequence model 설계 공간의 통일 이론** — 4축 설계 공간과 retention 재이론화; Moneta/Yaad/Memora.
- **14장 · Atlas: Learning to Optimally Memorize the Context at Test Time** — capacity 이론, Omega rule, inner Muon; retrieval 격차를 정직하게.
- **15장 · TNT: Improving Chunkwise Training for Test-Time Memorization** — chunk 경제학, reset·context parallelism, Q-K projection, train/serve mismatch; 라인 유일의 wall-clock.
- **16장 · Nested Learning: deep learning 아키텍처라는 착시, 그리고 Hope** — update frequency로 색인된 nested memory, optimizer-as-memory, CMS, self-modifying Titans, Hope.
- **17장 · Sleep: Learning to Self-Modify and Consolidate Memories** — wake/sleep lifecycle, Knowledge Seeding, Dreaming; graft-vs-cotrain seam.

## Part III — 원저 기여

완성형이 만드는 배포 workload를 D4 pair thesis로 가르고 8개 실측(exploration-grade)으로 계량한다.

- **18장 · 여섯 편의 수렴과 Part III의 기여 프로그램** — 완성형 재구성과 두 유보, D4 pair thesis 정식 진술, 8개 claim의 실험·figure·장 매핑, 정직성 계약.
- **19장 · Scaling 분석과 memory-centric 모델의 후보 scaling law** — state-bytes와 capacity를 params·tokens와 나란한 scaling 축으로 올려 후보 law를 맞춘다.
- **20장 · 하드웨어 병목 분석: decode roofline과 새 serving primitive** — 각 아키텍처의 decode roofline과 backward-pass-at-decode.
- **21장 · Transformer/NVIDIA scaling 시대의 교훈: hardware lottery** — attention이 GEMM density로 이긴 역사가 이 라인의 채택·kernel에 예고하는 것.
- **22장 · Player-strategy 분석: 누가 TTT scaling에 어떻게 참여하는가** — Google·NVIDIA·open-source kernel 생태계의 합리적 다음 수.
- **23장 · 대규모 학습·서빙 projection** — TNT 경제학의 7–70B 외삽, per-session weight state를 새 cache class로 설계.
- **24장 · 제안: 알고리즘 레벨과 하드웨어 레벨** — 소규모 실험 제안과 frequency-tiered memory·fused kernel·grouped-GEMM decode의 pair.
- **25장 · 결론과 연구 어젠다** — 남은 다섯 공백과 연구 프로그램.

## 뒷부분 (back matter)

- **참고문헌** — Veridraft / Semantic Scholar 검증 풀에서 후공정으로 구축.
- **용어 색인(glossary)** — 개념의 정의 소유 장으로 연결.
