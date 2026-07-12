# 서문

이 책은 여섯 편의 논문 — [Titans], [Miras], [Atlas], [TNT], [NL], [Sleep] — 을 하나의 연속된 연구 프로그램으로 읽고, 그 프로그램이 만들어 낸 배포 형태를 systems 엔지니어의 도구로 측정한 결과다. 논문들은 Google Research의 Ali Behrouz 등이 2024년 12월부터 2026년 6월까지 순차 공개한 test-time memorization / neural-memory 계열이며, 저마다 "다음 논문의 abstract가 이 논문의 future work"인 방식으로 연결된다. 이 책은 그 연결을 복원하고, 마지막에 그 완성형을 decode roofline·GEMM shape·memory 계층 배치라는 독자의 언어로 계량한다.

## 누구를 위한 책인가

이 책은 **transformer inference를 잘 아는 한 명의 엔지니어**를 상정하고 쓰였다. 그 독자는 KV cache, paged attention, FlashAttention tiling, GEMM shape, MFU, roofline, prefill/decode 비대칭, continuous batching, scan kernel을 운영 감각으로 안다. efficient-transformer와 linear-attention 계열(DeltaNet, Gated DeltaNet, Mamba-2)도 최소한 이름과 골격은 안다.

그러나 그 독자는 **training 경험이 전무하다.** backward pass, optimizer state, momentum과 weight decay가 무엇을 하는지, loss surface 위에서 gradient step이 무엇을 움직이는지 — 이 어휘는 독자의 것이 아니다. 이 라인의 정의적 사건이 바로 그 어휘가 **serving 경로 안으로 들어오는 것**이라서, 이 간극이 이 책의 존재 이유다. decode 한 step이 작은 신경망의 forward·loss·backward·update가 되는 순간, 추론 엔지니어는 자기가 한 번도 다뤄 본 적 없는 객체 — 움직이는 weights, optimizer trajectory 상태, gradient가 만드는 트래픽 — 를 serving 시스템 안에서 마주하게 된다. 이 책은 그 마주침을 준비시킨다.

이 독자를 위해 두 가지 원칙을 관철한다. 첫째, 모든 training 개념은 처음부터 **(state, update, cost)를 갖는 객체**로 제시한다 — optimizer는 "register file과 accumulator를 가진 상태 기계"로, backprop은 "surprise를 저장하는 memory"로 도입된다. 둘째, 새 개념은 반드시 독자의 inference 어휘에 접속시킨다: memory write는 KV cache append로, retention gate는 학습된 eviction policy로, chunk는 (그러나 bit-exact가 아닌) FlashAttention tile로 — 이 대응의 성격(동일/유비/차이가 논점)까지 매번 밝힌다.

## 왜 이 여섯 편인가

이 여섯 편은 임의로 묶은 reading list가 아니라 **하나의 설계 공간을 한 축씩 열고 닫아 온 프로그램**이다. 먼저 optimizer를 sequence layer로 만들고([Titans]), 그 수가 여는 설계 공간에 이름을 붙이고([Miras]), 그 공간의 각 축을 최적으로 밀어붙이고([Atlas]), 훈련 비용을 계산해 지불하고([TNT]), 그 수가 보편적임을 — 모델도 optimizer도 backprop도 전부 같은 associative memory임을 — 선언하고([NL]), 마지막으로 train/test 경계 자체를 지운다([Sleep]). 여섯 편의 open-questions 절이 서로 다른 다섯 공백(스케일, retrieval 격차, serving 경제학, 학습되는 스케줄, 안전한 self-modification)으로 **수렴한다**는 사실이, 이 라인의 다음 수가 텍스트에 의해 과잉 결정되어 있음을 보여 준다. 그 다음 수 — 세션 상태가 mutable weights인 continually-learning LLM — 를 읽어 내는 것이 Part II의 결산이다.

## 어떻게 읽는가

책은 세 Part로 나뉜다.

- **Part I (1–11장) — inference 엔지니어를 위한 배경.** training이라는 낯선 세계를 독자의 어휘로 건설한다. optimizer·online learning·meta-learning·associative memory·chunkwise 병렬화가 여기서 (state, update, cost) 객체로 도입된다.
- **Part II (12–17장) — 여섯 편의 정밀 독해.** 각 논문을 통일 표기로 해부하고, 무엇을 계승·일반화·폐기했는지 bridge로 잇는다. 여섯 장을 겹치면 하나의 완성형이 떠오른다.
- **Part III (18–25장) — 원저 기여.** 그 완성형을 척추 명제(아래)로 정식화하고, 8개 실측 실험으로 계량하며, scaling·하드웨어·serving·player-strategy·제안으로 전개한다.

**training 무경험 독자를 위한 우회로.** Part I을 전부 정독하는 것이 정도이지만, 이 독자에게 진짜 새로운 장과 이미 아는 장은 다르다. 급행 경로는 이렇다: **1·2·8·9장을 정독**하라 — 1장(Rosetta 지도), 2장(optimizer를 객체로; 이 독자의 최대 공백), 8장(TTT 원형), 9장(chunkwise 병렬화; 이 라인 전체의 실행 기반). **3·4장은 Part II가 호출할 때 당겨 읽어도** 된다(regret/FTRL는 13장이, meta-learning/$W_{\mathrm{init}}$은 15장이 호출한다). **5·6·7장은 이미 linear attention과 SSM을 아는 독자라면 훑고**(delta rule의 crosstalk과 §1.6 카탈로그만 확인) 지나가도 좋다. **10·11장은 참조 장**이다 — 10장은 cost model cheat sheet라 Part III에서 되돌아오고, 11장은 continual learning으로 17장 직전에 읽으면 된다. 요컨대 이 독자에게 Part I의 무게중심은 익숙한 memory 계보(5–7장)가 아니라 낯선 training 기계(2–4·8–9장)에 있다.

각 장은 목표 상자("이 장을 마친 독자는 ~할 수 있다")로 열고, Part I 장은 손으로 따라갈 수치 예제(worked micro-example)와 자가 점검 체크리스트로 닫는다. 앞선 장을 참조할 때는 "→ 9장"처럼 장 번호로 가리킨다.

## 이 책의 척추: pair thesis 예고

Part III 전체는 하나의 명제를 검증한다. 완성형이 만드는 배포 workload는 **질적으로 다른 두 부하로 갈라지며, 각각의 최적 하드웨어 전략이 다르다**는 것이다(정식 진술은 18장).

- **decode/serving-state 관리 = memory-centric 기회.** decode step은 token마다 fast-weight state 전체를 읽고·갱신하고·되쓴다(read-modify-write). 이 트래픽은 sequence마다 unshared이고 write-heavy이며 content로 주소 지정되지 않는다 — append-once/read-many이고 prefix로 공유 가능한 KV cache와 질적으로 다른 memory 부하다.
- **training/prefill = accelerator 영역.** 같은 알고리즘도 chunk 크기 $C$를 키우면 compute-bound로 옮겨 간다. chunk $C$는 문자 그대로 roofline의 x축이며, 승부는 fused chunk kernel과 grouped-GEMM에서 난다.

흥미로운 하드웨어 제안은 둘 중 하나가 아니라 **쌍(pair)** 이다 — 어느 한쪽만 옹호하는 것은 부하의 절반을 무시하는 것이다. 이 pair thesis(책의 명명 D4)가 memory-centric 논증을 펴는 유일한 두 지점이며, "이 라인의 훈련이 새 memory 소자를 요구한다"거나 "일반 PIM이 답이다" 같은 주장은 논문들의 자체 증거가 반대 방향을 가리키므로 이 책은 펴지 않는다.

## 정직성 계약

이 책은 논문에 불리한 사실의 완곡화·누락을 결함으로 취급한다. 여섯 편의 실증 상한은 **1.3B params / 100B tokens**이며([TNT]는 150M), decode wall-clock 수치는 여섯 편 어디에도 없다 — 이 두 사실은 Part II·III 전반에서 반복 명시된다. Atlas의 Muon 제거가 perplexity를 오히려 개선한 점, attention과의 in-context retrieval 격차(53.55 vs 43.70)가 측정된 채 닫히지 않은 점, TNT가 momentum·gating을 제거한 단순화 Titans로 검증한 점, Sleep이 pre-trained backbone 위의 graft라 라인 최초로 end-to-end meta-learn되지 않은 점 — 이런 유보는 감추지 않고 해당 장 본문에 담는다.

Part III의 실측도 같은 계약을 따른다. **이 책이 돌린 8개 실험은 exploration-grade다** — 순수 analytic cost model(hatir + hat-schema twin)과 CPU micro-benchmark에서 나온 값이다. 본문으로 승격되는 것은 **비율, crossover 위치, tier 순서, bound 분류**이지 silicon 정확 절대치가 아니다. 여섯 논문이 H100 decode wall-clock을 하나도 공개하지 않았으므로, µs/token·mJ/token 같은 절대치는 roofline **하한**일 뿐이며 **사내 A100 runbook(Part III-a)으로 이월**한다. novel device twin(scratchpad, PIM)은 `simulation_ready=False`로 directional DSE에만 인용한다. 세 독립 방법이 anchor 설정에서 1% 이내로 일치한다는 교차검증이, 절대치가 아니라 구조(비율·순서·crossover)를 본문에 올리는 근거다.

## 표기와 3층 구분

기술 용어는 **영어 원어(로마자)를 그대로** 쓴다 — momentum, weight decay, retention gate, surprise, fast weights, chunk, associative memory는 번역도 음차도 하지 않는다. 조사는 그 용어의 한글 발음을 기준으로 붙인다("chunk가", "momentum이"). 동사·서술어만 한국어를 허용한다("memory에 쓴다", "state를 갱신한다"). 통일 표기의 핵심은 두 loop의 대문자 구분이다: **$W$ = fast weights(inner loop 상태), $\Theta$ = slow weights(outer loop parameter).** inner learning rate는 $\eta_t$, momentum decay는 $\beta_t$, retention gate는 $\alpha_t\in[0,1]$(남기는 비율)로 예약된다. 전체 규약은 별도 기준 문서 `style/STYLE-NOTATION.md`에 있다.

서술은 세 층위를 표기로 구분한다. **논문의 주장**은 항상 논문에 귀속하고 위치를 병기한다("…이다. [Atlas §5.2]"). **이 책의 해설**(논문 내용을 독자의 어휘로 재서술)은 필요 시 블록으로 표시한다:

> **[해설]** inference 관점에서 이 식은 KV cache append를 rank-1 GEMM write로 바꾼 것이다.

**이 책의 평가**(논문과 다른 판단)는 반드시 블록으로 표시한다:

> **[평가]** 이 ablation은 Muon 축의 가치가 760M 스케일에서 미결임을 뜻한다.

독자가 어디까지가 논문이고 어디부터가 이 책인지 매 문단에서 알 수 있게 하는 것이, 정직성 계약의 표기 차원이다.

---

*저자: 이승호 (Seungho "Jimmy" Lee).*
*판본: 조립본 (P2 라운드), 2026-07-12. 기준 문서 `style/STYLE-NOTATION.md` v1.1을 따른다.*
