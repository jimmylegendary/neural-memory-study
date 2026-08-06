# ch13. RAG에서 MemGPT로 — 검색에서 운영체제로

## 13.1 Bridge-in: 전작이 남긴 문제

MemGPT(arXiv:2310.08560)가 명시적으로 상속한 문제는 **긴 문맥을 넓혀도 모델이 가운데를 쓰지 못한다**는 것이다. 논문 §1이 그 상속을 인용으로 적는다.

> "even if we could overcome the computational challenges of context scaling, recent research shows that long-context models struggle to utilize additional context effectively (Liu et al., 2023a). As consequence, given the considerable resources needed to train state-of-the-art LLMs and diminishing returns of context scaling, there is a critical need for alternative techniques to support long context." [MemGPT §1]

같은 진술이 실험 절에서 한 번 더 반복된다. "Recent research (Liu et al., 2023a) also raises doubts about the utility of simply scaling contexts, since they find uneven attention distributions in large context models ... To enable reasoning across documents, more flexible memory architectures like MemGPT are needed." [MemGPT §3.2]

상속은 수사에 그치지 않는다. 네 개 벤치마크 중 둘이 그 선행 논문에서 그대로 왔다 — retriever-reader document QA [MemGPT §3.2.1]와 합성 key-value retrieval [MemGPT §3.2.2]. 문제를 물려받고 그 문제를 정의한 측정 도구까지 물려받았다.

다만 정확히 적어 둔다. MemGPT는 어떤 선행 연구에 대해서도 "open question", "future work of", "left open by"라는 표현을 쓰지 않는다. 위 상속은 두 구절과 벤치마크 재사용으로부터 이 책이 복원한 것이다.

두 번째 상속은 이 책 안에서 온다. ch10의 결론은 RAG의 쓰기 쪽 공백이었다 — 그 기억은 논문 자신의 표현으로 "human-readable"이고 "human-writable"이며 [RAG §5], 빠진 세 번째 성질이 machine-writable이다(→ ch10 §10.2). Liu et al.의 문제가 "창을 키우지 마라"라면 ch10의 공백은 "그럼 무엇이 쓰는가"이고, MemGPT는 두 질문에 같은 답을 낸다: **모델 자신이 쓴다.**

## 13.2 문제의식

MemGPT의 문제 진술은 한 문장이다 — "In this paper, we study how to provide the illusion of an infinite context while continuing to use fixed-context models" [MemGPT §1]. 고정 창을 그대로 두고 무한 문맥의 **착시**를 만들겠다는 것이다.

그 착시를 만드는 방법이 이 장의 첫 번째 소유 개념이다.

> **정의.** **계층화된 외부 기억**은 $E$를 하나의 평평한 레코드 집합으로 두지 않고, **프롬프트 안에 상주하는 층**과 **프롬프트 밖에 사는 층**으로 나눈 뒤 두 층 사이의 이동을 명시적 연산으로 관리하는 관점이다. 이동을 촉발하는 것은 질의의 내용이 아니라 **상주 층의 용량 초과**이며, 따라서 $E$ 설계의 1차 변수가 검색 품질이 아니라 "어느 층에 무엇을 남길 것인가"로 바뀐다.

RAG의 $E$와 대비하면 세 축이 전부 뒤집힌다. RAG의 $E$는 corpus당 전역·읽기 전용·평평이라 승격도 강등도 연산으로 존재하지 않고, 검색을 촉발하는 것은 언제나 질의다. MemGPT의 $E$는 사용자당·읽기쓰기·최소 2계층이며, 쓰기를 촉발하는 것은 질의가 아니라 프롬프트 길이다.

최적화 대상도 함께 갈아엎인다. RAG가 올린 것은 $\mathrm{ret}(E,q)$의 정확도였지만(→ ch10 §10.4), MemGPT는 $\mathrm{ret}$을 손대지 않는다 — text-embedding-ada-002를 그대로 쓰고 [MemGPT §3.2.1] 검색 품질에 어떤 기여도 하지 않는다. 대신 검색 결과를 언제 어떻게 받아 **어디에 남길지**를 모델의 결정으로 옮긴다.

> **[해설]** 익숙한 대비로 옮기면, RAG는 캐시 **적중률**을 올리는 논문이고 MemGPT는 캐시 **교체 정책**을 만드는 논문이다. 그리고 그 교체 정책을 하드웨어도 런타임도 아닌 **모델 자신**이 실행한다. 이 계보의 실질적 시작은 저장소가 커진 것이 아니라 저장소를 운영하는 주체가 처음으로 모델이 된 것이다.

논문이 이 관점에 붙인 이름이 이 장의 두 번째 소유 개념이고, 그 이름은 곧바로 한계를 데리고 온다.

> **정의.** **virtual context management**는 고정 크기 프롬프트를 OS의 main memory로, 모델 밖 저장소를 disk로 놓고 둘 사이의 데이터 이동을 시스템이 관리해 무한 문맥의 착시를 만드는 기법이며, 논문 자신이 붙인 이름이다 [MemGPT Abstract]. **이 이름이 빌려 온 OS의 기제와 논문이 실제로 구현한 것이 갈라지는 자리가 셋이다**(판정은 아래 [평가]). 첫째, MemGPT의 page fault는 비자발적 트랩이 아니라 프롬프트 안에 텍스트로 배달되는 **경고**이고, 모델이 그것을 무시하면 아무 일도 일어나지 않는다 [MemGPT §2.2]. 둘째, 주소 변환도 보호도 상주 보장도 없다 — 이동은 전부 자발적이다. 셋째, 논문이 인용하는 유일한 OS·아키텍처 문헌은 Patterson, Gibson & Katz 1988이며 [MemGPT §1], 그것은 가상 메모리 논문이 아니라 RAID 논문이다.

> **[평가]** 은유는 검증되지 않고 주장된다. 그럼에도 이 장이 은유를 지우지 않는 이유는 새는 자리가 정확히 이후 $E$-경로가 채우려 한 자리이기 때문이다. "자발적 page fault"는 **정책이 학습되지 않았다**의 다른 표현이고, "상주 보장 없음"은 **쓰기의 성공이 보장되지 않는다**의 다른 표현이다. ch14·ch15가 손대는 지점이 이 둘이며, 은유의 결함 목록이 그대로 후속 연구 목록이 된다.

## 13.3 Core mechanism (통일 표기)

먼저 못 박아야 할 사실이 있다. **이 논문에는 번호 붙은 수식이 하나도 없고, 수학 표기 자체가 없다.** 순수 systems 논문이며 갱신 규칙 전부가 영어 산문으로 §2.1–§2.4에 쓰여 있다. 따라서 아래 식은 전부 이 책이 그 산문을 통일 표기로 옮긴 것이고, 원문 수식 번호 대신 **절 위치**를 병기한다. 이 부재 자체가 보고 대상이다 — $E$-경로의 조상이 자기 갱신 규칙을 수식으로 진술하지 않는다.

상태 분해부터 한다. $\Theta$는 동결된 상용 API 세 개다: gpt-4-1106-preview(문맥 128,000), gpt-4-0613(8,192), gpt-3.5-turbo-1106(16,385) [MemGPT §3]. $W$는 아예 존재하지 않는다 — Neural Memory가 통째로 다룬 층이 여기서는 빈칸이다(→ NM ch12). $E$와 그 프롬프트 내 투영은 다섯 블록으로 나뉘며, 결정적인 분할은 **프롬프트 안이냐 밖이냐**다 [MemGPT §2, §2.1].

프롬프트 **안**의 셋은 $I_{\mathrm{sys}}$(제어 흐름 설명과 함수 스키마를 담은 읽기 전용 정적 블록), $E^{\mathrm{wc}}$(working context — "a fixed-size read/write block of unstructured text, writeable only via MemGPT function calls"), $\mathcal{Q}$(FIFO queue — 롤링 메시지 이력이며 0번 자리에 축출 메시지들의 재귀 요약 $\sigma$가 들어간다)다 [MemGPT §2.1, §2.3]. 프롬프트 **밖**의 둘은 $E^{\mathrm{rec}}$(recall storage — 들어온 모든 메시지와 나간 모든 출력을 담는 메시지 DB)와 $E^{\mathrm{arc}}$(archival storage — "a read/write database storing arbitrary length text objects")다 [MemGPT §2.2].

앞의 셋은 이 책의 표기에서 $E$가 아니라 **$E$의 프롬프트 내 투영**, 즉 식 (R)에서 $f$의 인자로 들어가는 텍스트다. 실제 $E$는 $E^{\mathrm{rec}}$와 $E^{\mathrm{arc}}$뿐이다. 이 구분을 놓치면 MemGPT의 "main memory"를 $W$로 오독하게 된다.

프롬프트 조립은 세 블록의 연접이다 [MemGPT §2.1].

$$
P_t \;=\; \big[\, I_{\mathrm{sys}} \;;\; E^{\mathrm{wc}}_t \;;\; \mathcal{Q}_t \,\big],
\qquad
\mathcal{Q}_t = \big[\,\sigma_t;\ m_i,\dots,m_t\,\big]
\tag{13-1}
$$

읽기는 식 (R)에 $W=\varnothing$을 넣고 $\mathrm{ret}$을 **반복**으로 바꾼 형태다 [MemGPT §2.3, §2.4].

$$
\hat y \;=\; f\big(q;\ \Theta,\ \varnothing,\ \mathrm{ret}(E,q)\big),
\qquad
\mathrm{ret}(E,q) \;=\; \bigcup_{j=0}^{J-1}\ \mathrm{search}\big(E^{\mathrm{rec}}\!\cup\!E^{\mathrm{arc}},\ \tilde q_j\big)
\tag{13-2}
$$

식 (13-2)에서 두 가지가 RAG와 다르다. 검색 인자 $\tilde q_j$가 고정 encoder로 $q$에서 유도되지 않고 **모델이 디코딩하며**, $J$가 하이퍼파라미터가 아니다 — 모델이 `request_heartbeat=true`를 내는 동안 루프가 돌고 모델이 멈추면 끝난다. 검색 결과는 페이지 단위로 잘려 들어오므로 $\tilde q_j$가 같은 질의의 다음 페이지일 수도 있다 [MemGPT Fig. 6].

쓰기는 **성격이 정반대인 두 갈래**가 공존한다. 모델이 지시하는 갈래는 손실이 있고 확률적이다 [MemGPT §2.3].

$$
\big(E^{\mathrm{wc}}_{t},\,E^{\mathrm{arc}}_{t}\big) \;=\; \mathrm{wr}\big(E^{\mathrm{wc}}_{t-1},\,E^{\mathrm{arc}}_{t-1};\ a_t\big),
\qquad
a_t \sim p_\Theta(\cdot \mid P_t)
\tag{13-3}
$$

큐 관리자가 실행하는 갈래는 결정론적이고 무손실이다 [MemGPT §2.2].

$$
E^{\mathrm{rec}}_{t} \;=\; E^{\mathrm{rec}}_{t-1} \,\cup\, \{m_t,\ \hat y_t\}
\tag{13-4}
$$

식 (13-3)과 (13-4)의 대비가 이 논문의 구조적 핵심이다. 전자에서 쓰기 행위 $a_t$는 **샘플링된 토큰열**이고 $\mathrm{wr}$은 그것을 해석하는 인터프리터다 — `append`, `replace`, `archival_storage.insert`. 파싱에 실패하면 런타임 오류가 텍스트로 모델에게 되돌아간다 [MemGPT §2.3]. 즉 표준형 (U-E)에 없는 **실패 모드**가 있다. 후자는 모델이 개입하지 않는 순수 로그이고 예산도 요약도 없다. 이후 $E$-경로 논문들은 이 둘 중 어느 쪽을 지배적으로 만들 것인가를 다시 설계한 것이다(→ ch15).

쓰기를 촉발하는 것은 질의가 아니라 프롬프트 길이다 [MemGPT §2.2].

$$
|P_t| > N_{\text{warn}} \;\Rightarrow\; \mathcal{Q}_t \leftarrow \mathrm{append}\big(\mathcal{Q}_t,\ \text{“memory pressure”}\big),
\qquad
N_{\text{warn}} = 0.70\,N_{\mathrm{ctx}}
\tag{13-5}
$$

$$
|P_t| > N_{\text{flush}} \;\Rightarrow\; \mathcal{Q}_t \leftarrow \big[\sigma_t;\ \mathcal{Q}^{\mathrm{keep}}_t\big],
\qquad
\sigma_t = S\big(\sigma_{t-1},\, \mathcal{Q}^{\mathrm{evict}}_t;\ \Theta\big),
\qquad
N_{\text{flush}} = 1.00\,N_{\mathrm{ctx}}
\tag{13-6}
$$

축출량은 문맥창의 약 50%다 [MemGPT §2.2]. 위 세 백분율(70%, 100%, 50%)은 원문이 전부 "e.g."를 달아 적은 예시값이지 실험에서 쓴 설정값이 아니다 — 그대로 인용하되 측정된 구성으로 읽으면 안 된다.

식 (13-6)의 $S(\cdot)$는 표준형 (U-E)의 $S(c;B_s)$와 **방향이 반대**다. (U-E)에서 $S(c;B_s)$는 문맥을 압축해 저장소 **안으로** 넣지만, MemGPT의 $S(\cdot)$는 축출된 프롬프트 내용을 압축해 프롬프트 **밖으로** 밀어내고 그동안 $E^{\mathrm{rec}}$는 식 (13-4)에 의해 아무것도 잃지 않는다. 손실은 저장소가 아니라 프롬프트에서만 일어난다. 그리고 이 압축의 예산은 계산 예산이 아니라 **토큰 개수 목표**다 — MemGPT에 압축 예산이 존재하는 유일한 자리이고, 그것이 $B_s$가 아니라는 사실이 §13.5의 결론을 미리 정한다.

마지막으로 제어 흐름이다 [MemGPT §2.4].

$$
\text{사이클 } j:\quad o_j \sim p_\Theta\big(\cdot\mid P^{(j)}\big),
\qquad
P^{(j+1)} = \big[P^{(j)};\ \mathrm{exec}(o_j)\big]\ \ (\texttt{heartbeat}{=}\texttt{true}),\ \ \text{아니면 정지}
\tag{13-7}
$$

식 (13-7)이 말하는 것은 **사용자 턴 하나가 forward pass 하나가 아니라는 것**이다. 연쇄된 함수 호출 하나하나가 직전보다 긴 프롬프트 위에서 도는 완전한 추론 사이클이며, 논문은 이 사이클 수를 한 번도 세지 않는다.

### 13.3.1 표기 대응표

표 13-1 — MemGPT 원 논문 기호 → 이 책 기호. 이 논문에는 번호 붙은 수식이 없으므로 원문 위치는 절·그림 번호로 병기한다.

| 원 논문 표기 | 원문 위치 | 이 책 표기 | 처리 |
|---|---|---|---|
| "memory" / "main context" / "external context" | §2, §2.1 | $E$ (프롬프트 밖만) / $f$의 인자(프롬프트 안) | 논문은 프롬프트 블록과 DB를 둘 다 memory라 부른다. 이 책에서 $E$는 프롬프트 밖 부분만이다 |
| working context | §2.1 | $E^{\mathrm{wc}}$ | 프롬프트 내 투영. 이후 Letta 계열의 "core memory" 개명은 이 논문에 없다(→ ch14) |
| FIFO queue | §2.1 | $\mathcal{Q}_t$ | 원문의 $Q$는 attention의 query 행렬로 오독되므로 필기체로 옮긴다 |
| recursive summary | §2.1, §2.2 | $\sigma_t$ | (U-E)의 $S(\cdot;B_s)$에 대응하나, $S_t$(momentum)와 절대 혼동하지 않는다. 이 장은 $W=\varnothing$이므로 $S_t$가 등장하지 않는다 |
| "context window"(토큰 단위) | Table 1, §3 | $N_{\mathrm{ctx}}$ | 예약 기호 $C$(chunk)·$C_{\text{cap}}$(용량)과 절대 혼용하지 않는다 |
| warning / flush token count | §2.2 | $N_{\text{warn}}$ / $N_{\text{flush}}$ | 예약 기호 $\tau$(세션 첨자)와 충돌하므로 $N$ 계열로 옮긴다. 셋 다 토큰 수다 |
| $K$ (회수 문서 수) | §3.2.1 | $K_{\text{ret}}$ | ch10 §10.4가 도입한 회수 폭 기호를 그대로 쓴다. 예약 기호 $k$(sleep 라운드)·$K$(key 행렬)와 구분 |
| "function"(MemGPT의 호출 가능 객체) | §2.3 | $\mathrm{search}$, $\mathrm{append}$, $\mathrm{replace}$, $\mathrm{insert}$ | 예약 기호 $f$(식 (R)의 읽기 사상)로 축약하지 않는다 |
| "page" / "paging" / "page fault" | §1, §2.2 | (번역하지 않음) | 기호 충돌은 없으나 의미 함정이 있다 — MemGPT의 page fault는 자발적이고 모델이 발행한다. §13.2 정의 참조 |
| 70% / 100% / 50% | §2.2 | 그대로 | 원문이 "e.g."로 적은 예시값. 측정된 설정으로 제시 금지 |

## 13.4 어느 층을 언제 쓰는가

이 절의 답은 한 줄이다. **$\Theta$는 한 번도 움직이지 않고, $W$는 존재하지 않으며, $E$만 매 턴 움직인다.** 그런데 더 중요한 사실이 하나 더 있다.

**MemGPT에서는 아무것도 학습되지 않는다.** gradient가 하나도 계산되지 않고 loss가 없으며 이 방법에 속하는 파라미터가 없다 — 논문에 training 절 자체가 없다 [MemGPT §3]. 학습 가능해 보이는 모든 객체는 셋 중 하나다: 남이 사전학습해 동결한 것, 사람이 손으로 쓴 영어, wake time에 디코딩으로 만들어진 텍스트. 독자의 1번 질문 "이 값은 누가 언제 학습하는가"에 대한 답은 전 항목에서 **아무도, 한 번도**다.

표 13-2 — MemGPT의 모든 상태 객체와 그 갱신 주체. "학습"은 gradient 갱신을 뜻한다.

| 객체 | 층 | 누가 정하는가 | 언제 | test time에 움직이는가 | 출처 |
|---|---|---|---|---|---|
| 기반 LLM 가중치 | $\Theta$ | OpenAI의 사전학습·후처리 | 논문 이전. 전 구간 동결 | 아니오 | §3 |
| fast weights | $W$ | 없음 — 객체 자체가 없다 | — | 해당 없음 | 전 논문에 부재 |
| $I_{\mathrm{sys}}$ (기억 계층 설명 + 함수 스키마) | 프롬프트 | 저자가 손으로 작성 | 저작 시점. "read-only (static)" | 아니오 | §2.1, §2.3, App. §6.1 |
| $E^{\mathrm{wc}}$ (working context) | $E$의 프롬프트 내 투영 | LLM이 `append`/`replace`를 발화 | **wake time**, 사용자 턴의 임계 경로 위 | 예 — 이 논문의 중심 상태 | §2.1, §2.3, Fig. 1, Fig. 4 |
| $\mathcal{Q}$와 $\sigma$ | 프롬프트 | 큐 관리자(결정론적 정책)가 요약 텍스트 생성에만 $\Theta$ 호출 | wake time, flush 시점 | 예 | §2.2 |
| $E^{\mathrm{rec}}$ (recall storage) | $E$ | 큐 관리자가 무조건. 모델 결정 없음 | wake time, 매 메시지 | 예 (append-only, 압축 없음) | §2.2 |
| $E^{\mathrm{arc}}$ (archival storage) | $E$ | 쓰기는 LLM의 함수 호출, 벌크 적재는 운영자 | wake time / 실험 전 사전 적재 | 예 | §2.2, §3.2.1 |
| embedding 모델 + HNSW index | $E$의 읽기 경로 | OpenAI(모델), 빌드(index) | 동결. embedding 사전 계산 | 아니오 | §3.2.1 |
| **기억 관리 정책** (언제 쓰고 언제 검색하고 언제 멈추는가) | — | **아무도** — 학습되지 않고 **프롬프트로 지시된다** | 해당 없음 | 매 사이클 디코딩으로 처음부터 재유도 | §2.3, §5 |

표의 마지막 행이 이 장의 무게중심이다 — "We implement self-directed editing and retrieval by providing explicit instructions within the system instructions" [MemGPT §2.3]. 정책은 지시되지 학습되지 않는다.

한 턴 안의 순서는 이렇다. 메시지가 $\mathcal{Q}$에 붙는다 → 식 (13-4)의 로그 쓰기 → 필요하면 식 (13-5)의 압력 경고 → 식 (13-2)의 읽기와 식 (13-3)의 쓰기가 식 (13-7)의 연쇄 안에서 **자유롭게 섞이며** 반복 → 필요하면 식 (13-6)의 flush와 요약 → 답변. 읽기 단계와 쓰기 단계의 분리가 없고, 이 무단계성이 비용을 세지 못하게 만드는 직접 원인이다.

이 장의 의무 caveat 하나를 여기서 명시한다. **MemGPT는 이 책의 판별식 조건 (1)을 만족하지 않는다.** 판별식은 질의 $q$가 도착하기 **전에** 예산을 써서 상태를 바꿀 것을 요구하는데(→ ch11), MemGPT의 쓰기는 전부 대화 **중**에, 사용자 턴의 임계 경로 위에서, 답을 만드는 것과 같은 디코딩 예산으로 일어난다(식 (13-3), (13-6)). 그럼에도 $E$-경로의 출발점으로 두는 근거는 둘이다 — 그 쓰기의 산출물이 **이후 질의**에 쓰이고(조건 (4)는 만족), ch14가 이 계보를 자기 조상으로 지목한다. MemGPT는 sleep-time compute의 **구조적** 조상이되 sleep-time compute이 아니다.

> **[해설]** 유휴 예산의 자리는 논문 안에 이미 뚫려 있다. §2.4가 이벤트 종류를 열거하며 "timed events that are run on a regular schedule (allowing MemGPT to run 'unprompted' without user intervention)"을 적는다 [MemGPT §2.4]. 정기 스케줄로 도는 사용자 개입 없는 실행 — 이것이 $B_s$가 들어갈 구멍이다. 그런데 어떤 실험도 이 훅을 쓰지 않고 어떤 계산도 여기에 귀속되지 않는다. ch14가 하는 일을 한 문장으로 줄이면 **이 훅을 실제로 실행하고 그 비용에 이름을 붙인 것**이다.

## 13.5 비용 4종

표 13-3 — MemGPT의 비용 4종. 논문이 침묵한 칸은 "논문에 없음"으로 적었고, 추정치는 넣지 않았다.

| 기호 | 값 | 논문이 대신 말하는 것 |
|---|---|---|
| $B_s$ | **논문에 없음** | sleep 단계가 없으므로 예산도 없다. 훅만 있다 — "timed events that are run on a regular schedule" [MemGPT §2.4]. 어떤 실험도 이 훅을 쓰지 않고 계산이 귀속되지 않는다 |
| $L_w$ | **논문에 없음** | TTFT도, ms/token도, wall-clock도, 토큰 수도, 달러 비용도 논문 전체에 없다. 지연에 관한 유일한 문장은 시스템이 아니라 벡터 index에 대한 것이다 — "which uses an HNSW index to enable approximate, sub-second query times" [MemGPT §3.2.1]. 그것은 archival 검색 **1회**의 상한이고, MemGPT는 한 턴에 여러 번 호출한다(식 (13-7)) |
| $C_{\text{cap}}$ | **논문에 없음**(바이트 기준) | 주어진 것은 상태 용량이 아니라 문맥창 $N_{\mathrm{ctx}}$뿐이다: 8,192 / 16,385 / 128,000 토큰 [MemGPT §3]. $E^{\mathrm{wc}}$는 "fixed-size"라고만 하고 **그 크기를 끝내 밝히지 않는다** [MemGPT §2.1]. $E^{\mathrm{arc}}$·$E^{\mathrm{rec}}$에는 용량 정책이 없고, recall storage는 명시적으로 무한하다("stored indefinitely") [MemGPT §2.2]. corpus 규모 수치는 공개 아티팩트 하나뿐 — Wikipedia 문서 2천만 건의 embedding [MemGPT §3] |
| $\rho$ | **논문에 없음** | 반복 consolidation 실험도, 라운드별 열화 측정도, staleness 연구도 없다. Fig. 4가 `working_context.replace`로 "Boyfriend named James"를 "Ex-boyfriend named James"로 고치는 장면을 보여 주지만 [MemGPT Fig. 4] 지표가 붙지 않은 예시 대화다. DMR 과제는 이전 5개 세션 + 새 세션 1개를 쓰고 [MemGPT §3.1], 세션 수를 변화시켜 열화를 드러내지 않는다 |
| $N_q$ | **논문에 없음** | 한 문맥을 몇 개 질의가 공유하는지에 대한 분포가 없다 |

$N_q$가 없다는 사실의 귀결은 명확하다. **식 (A)를 이 논문으로 채울 수 없다** — 분자에 넣을 $C_{\text{sleep}}$이 없고($B_s$ 부재) 분모에 넣을 $N_q$도 없다. 실서빙 $N_q$ 분포를 보고한 논문이 corpus에 없다는 사실의 한 사례다(→ ch14, ch25).

$C_{\text{cap}}$ 칸에는 특히 주의가 필요하다. ch10이 RAG에서 채운 $C_{\text{cap}}$은 36 GB / 약 100 GB였고 단위가 **corpus당**이었다(→ ch10 §10.1). MemGPT의 상태는 **사용자당**인데 그 값이 바이트로 보고되지 않는다. $E$-경로는 단위가 corpus당에서 사용자당으로 바뀌는 바로 그 논문에서 숫자를 잃는다. 두 논문의 $C_{\text{cap}}$을 나란히 놓아 비교할 수 없다는 뜻이며, 이 결손은 corpus 전체로 확대된다(→ ch26).

> **[평가]** 비용 4종이 전부 비어 있으므로 이 책은 MemGPT에 대해 "효율적"이라는 서술을 쓰지 않는다. 그런데 이 침묵은 중립적이지 않고 방향이 있다 — 식 (13-7)이 사용자 턴 하나를 여러 번의 추론 사이클로 만들고 각 사이클은 직전보다 긴 프롬프트 위에서 돈다. 따라서 MemGPT는 답 하나당 **반드시** baseline보다 비싸다. 얼마나 비싼지를 묻는 질문 자체가 논문에 등장하지 않는다.

## 13.6 실험과 스케일

이 절은 두 논문의 실험을 차례로 감사한다.

### 13.6.1 RAG의 평가설계 — 전수

ch10은 RAG를 $E$층의 **형태와 비용**으로만 읽었고 학습된 검색기가 FEVER에서 BM25에 지는 사실까지만 실었다(→ ch10 §10.6). 남은 평가설계 쪽 사실 여덟 항을 여기서 전부 적는다.

**(1) TriviaQA의 SotA 주장은 split 의존적이고, 유리한 split이 쉬운 쪽이다.** 헤드라인 문장부터 단서가 붙어 있다 — "On all four open-domain QA tasks, RAG sets a new state of the art (only on the T5-comparable split for TQA)" [RAG §4.1]. 부록은 그 이유를 official Wiki test set의 문제가 Wikipedia에서 답하기 더 쉽기 때문이라고 적는다 [RAG App. D]. 헤드라인 68.0은 그 쉬운 쪽 split의 값이다. 관행적인 open-domain split에서는 RAG-Sequence가 56.8이고 같은 표의 DPR이 57.9다 [RAG Table 1] — 이 split에서는 진다.

**(2) MS-MARCO 비교가 like-for-like가 아니며, 논문도 그렇게 쓴다.** 저자들은 과제가 제공하는 gold passage 열 개를 버리고 open-domain 설정으로 푼 뒤 [RAG §3.2], 캡션에 "*Uses gold context/evidence"라고 표시된 SotA와 비교한다 [RAG Table 2]. 논문 자신이 어떤 질문은 gold passage 없이는 참조 답과 일치하게 답할 수 없고 어떤 질문은 Wikipedia만으로 답할 수 없다고 덧붙인다 [RAG §3.2]. 조건이 다른 두 시스템의 점수가 한 표에 나란히 인쇄된다.

**(3) Jeopardy는 저자가 제안한 과제이고, 평가도 baseline도 저자가 만들었다.** "As this is a new task, we train a BART model for comparison" [RAG §3.3]. 인간 평가는 BART와 RAG-Token 생성물 **452쌍**의 쌍대 비교이며 선택지는 A 우세/B 우세/둘 다 좋음/둘 다 나쁨 네 개다 [RAG §3.3, §4.3]. 임베드된 gold 문장 검사를 통과하지 못한 평가자 두 명이 제거되었고 [RAG App. B], 평가자 수도 평가자 간 일치도도 신뢰구간도 보고되지 않는다.

**(4) 그 인간 평가 표의 한 열이 100%가 되지 않는다.** Table 4의 Specificity 열은 BART better 16.8%, RAG better 37.4%, Both good 11.8%, Both poor 6.9%, No majority 20.1%로 합이 **93.0%**다 [RAG Table 4]. 같은 표의 Factuality 열은 7.1 + 42.7 + 11.7 + 17.7 + 20.8 = 100.0%로 맞는다. 논문은 이 차이를 언급하지 않는다.

**(5) 같은 표에 대해 산문과 표가 어긋난다.** §4.3은 "both RAG and BART were factual in a further 17% of cases"라고 쓰지만 [RAG §4.3], Table 4의 Factuality "Both good"은 11.7%이고 17.7%는 "Both poor"다 [RAG Table 4]. 어느 쪽을 옮긴 것인지 원문에서 결정되지 않는다. 이 책은 표 값을 인용하고 불일치를 그대로 기록한다.

**(6) null-document 기제는 세 변형을 만들고 전부 버렸는데, 숫자가 하나도 없다.** 학습되는 null embedding, 정적 학습 bias 항, logit을 예측하는 작은 신경망을 시도하고 "We did not find that these improved performance, so in the interests of simplicity, we omit them"으로 끝낸다 [RAG App. F]. 수치가 없으므로 답할 수 없는 질의에서 손해가 얼마인지 알 수 없다.

**(7) RAG-Sequence와 RAG-Token의 순서가 과제마다 뒤집히고, 고르는 규칙이 없다.** RAG-Token이 이기는 곳은 Jeopardy B-1 17.3 대 14.7, QB-1 22.2 대 21.4, WebQuestions 45.5 대 45.2이고, RAG-Sequence가 이기는 곳은 MS-MARCO R-L 40.8 대 40.1과 B-1 44.2 대 41.5, Natural Questions 44.5 대 44.1, CuratedTrec 52.2 대 50.0이다 [RAG Table 1, Table 2]. 논문의 설명은 Jeopardy에 대한 사후 이야기 — RAG-Token이 여러 문서의 내용을 결합할 수 있어서일 수 있다는 것 — 뿐이다 [RAG §4.3].

**(8) 모든 증거가 단일 corpus에서 나온다.** 2018년 12월 Wikipedia dump 하나이고 2016년 12월 dump는 index 교체 프로브에만 쓰인다 [RAG §3, §4.5]. 웹 corpus도 도메인 corpus도 다중 corpus 혼합도 사용자별 파티션도 시험되지 않는다 — Broader Impact 절이 "by endowing it with a medical index" 같은 용도를 직접 제안하는데도 그렇다 [RAG Broader Impact].

> **[평가]** 여덟 항을 묶으면 성격이 드러난다. 넷은 **비교 조건**의 문제이고(1·2·3·8), 둘은 **표의 내적 정합성** 문제이며(4·5), 둘은 **보고되지 않은 음의 결과**다(6·7). 어느 것도 RAG의 기제를 반증하지 않는다. 반증되는 것은 그 기제의 이득 **크기**를 이 논문에서 읽어낼 수 있다는 믿음이다. 이 결손은 비용 축의 공백과 독립이 아니다(→ ch10 §10.5) — 비용을 세지 않으면 이득을 어떤 척도에도 걸 수 없으므로 비교 조건을 조이려는 압력이 생기지 않는다.

### 13.6.2 MemGPT의 실험

평가 대상은 폐쇄 API 세 개뿐이고 [MemGPT §3], 파라미터 수는 어디에도 없다 — 전부 비공개 endpoint라 셀 수가 없다. open-weights 모델은 하나도 돌리지 않는데 정작 문제를 동기화하는 Table 1은 Llama, Llama 2, Mistral 7B, Yi-34B-200k로 채워져 있다. 그 Table 1의 메시지 수 열도 측정값이 아니다 — "preprompt of 1k tokens, and an average message size of ~50 tokens (~250 characters)"를 가정해 유도한 값이며 [MemGPT Table 1], 논문의 동기 주장("이만큼밖에 안 들어간다")을 지탱하는 것이 바로 그 열이다.

**Deep Memory Retrieval (DMR).** Multi-Session Chat 위에 저자들이 세션 6을 만들고 QA 쌍을 "generated using a separate LLM"으로 생성한 저자 제작 벤치다 [MemGPT §3.1, §3.1.1].

표 13-4 — DMR 정확도와 ROUGE-L recall [MemGPT Table 2]. 정확도는 GPT-4 판정기 채점이다.

| 시스템 | 정확도 | ROUGE-L recall |
|---|---|---|
| GPT-3.5 Turbo | 38.7% | 0.394 |
| GPT-3.5 Turbo + MemGPT | 66.9% | 0.629 |
| GPT-4 | 32.1% | 0.296 |
| GPT-4 + MemGPT | 92.5% | 0.814 |
| GPT-4 Turbo | 35.3% | 0.359 |
| GPT-4 Turbo + MemGPT | 93.4% | 0.827 |

이 표에는 세 가지가 함께 붙어야 한다. 첫째, **비교가 정보 동등하지 않다.** 논문 자신의 문장이 "The baselines are able to see a lossy summarization of the past five conversations ... while MemGPT instead has access to the full conversation history"다 [MemGPT §3.1.1]. 92.5% 대 32.1%의 격차는 방법의 효과와 입력 접근권의 효과를 **합쳐서** 잰 값이고, 접근권을 맞춘 baseline은 돌지 않았다. 둘째, **baseline이 뒤집혀 있다.** GPT-4 baseline(32.1%, 0.296)이 GPT-3.5 Turbo baseline(38.7%, 0.394)보다 낮다. 논문은 이 역전을 언급하지 않으며, 역전은 GPT-4의 상승폭(+60.4점)을 GPT-3.5의 상승폭(+28.2점)보다 커 보이게 만든다. 셋째, **판정기가 관대하게 지시받았다** — "you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT" [MemGPT App. §6.1.2]. 판정기는 피험 시스템과 같은 모델 계열이며, DMR 평가 집합의 크기는 논문 어디에도 없다.

**Conversation Opener.** 저자가 정의한 지표로 저자가 만든 과제다 [MemGPT §3.1.2].

표 13-5 — Conversation Opener 유사도 [MemGPT Table 3]. SIM-1·SIM-3은 gold persona 라벨과의 유사도, SIM-H는 사람이 쓴 opener와의 유사도다.

| 시스템 | SIM-1 | SIM-3 | SIM-H |
|---|---|---|---|
| Human | 0.800 | 0.800 | 1.000 |
| MemGPT (GPT-3.5 Turbo) | 0.830 | 0.812 | 0.817 |
| MemGPT (GPT-4) | 0.868 | 0.843 | 0.773 |
| MemGPT (GPT-4 Turbo) | 0.857 | 0.828 | 0.767 |

**이 표에는 MemGPT를 쓰지 않은 LLM 행이 아예 없다.** 비교 대상이 사람이 쓴 opener 하나뿐이므로 이 실험은 MemGPT에 귀속되는 상승을 하나도 확립하지 못한다. "사람을 넘어선다"는 주장을 지탱하는 것도 persona 라벨 대비 유사도인 SIM-1·SIM-3뿐이다 — SIM-H에서는 사람이 정의상 1.000이라 모든 변형이 지고, **가장 강한 backend가 가장 낮다**(0.767 < 0.773 < 0.817). 저자들 자신이 MemGPT의 opener가 "more verbose and cover more aspects of the persona information"라고 적는데 [MemGPT §3.1.2], 그것은 persona 라벨 유사도를 부풀리는 바로 그 성질이지 opener가 더 낫다는 증거가 아니다.

<!-- TODO-VERIFY: SIM-1과 SIM-3의 정확한 정의. 원문은 Table 3 캡션과 §3.1.2에서 "similarity scores to the gold persona labels (SIM-1/3)"라고만 쓰고 1과 3의 의미를 정의하지 않는다. 확인 방법: research.memgpt.ai에 공개된 벤치마크 코드에서 SIM-1/SIM-3/CSIM 계산부 확인. 확정 전까지 본문에 정의를 쓰지 않는다. -->

**Document QA와 nested KV.** 둘 다 **표가 없고 결과가 그림으로만 존재한다** [MemGPT Fig. 5, Fig. 7]. Document QA는 Liu et al. 2023a의 과제를 그대로 쓰고 Natural Questions-Open에서 질문 **50개**를 표본으로 뽑았으며 [MemGPT §3.2.1], nested KV는 UUID 키-값 쌍 140개(약 8k 토큰)에 대해 중첩 수준마다 순서 배치 **30개**를 뽑았다 [MemGPT §3.2.2]. 이 표본 크기에서 분산도 오차 막대도 보고되지 않은 것은 중대한 결손이다.

Document QA에서 "MemGPT's performance is unaffected by increased context length"이고 [MemGPT Fig. 5 캡션] GPT-4는 회수 문서가 늘수록 계속 좋아진다 [MemGPT §3.2.1] — 고정 문맥 baseline은 우상향하고 MemGPT는 평평하다. **큰 $K_{\text{ret}}$에서 baseline이 MemGPT를 추월하는지, 교차점이 있다면 어디인지를 논문은 말하지 않는다.** nested KV에서는 GPT-3.5가 중첩 1에서 0%로 떨어지고 GPT-4와 GPT-4 Turbo는 중첩 3에서 0%에 도달하며 MemGPT+GPT-4는 중첩 수에 영향받지 않는다 [MemGPT §3.2.2].

<!-- TODO-VERIFY: Fig. 5 곡선의 수치값과, 큰 K_ret에서 고정 문맥 baseline이 MemGPT를 추월하는지 여부. Fig. 7의 중첩 수준 4가 실제로 그려졌는지도 함께(본문은 0-4를 시험했다고 쓰는데 축 눈금은 0-3이다). 확인 방법: arXiv:2310.08560v2 PDF의 Figure 5·Figure 7 원본, 또는 research.memgpt.ai 공개 저장소의 플로팅 데이터. -->

불리한 사실 셋이 이 두 그림에 붙는다. 첫째, **더 강한 기반 모델이 MemGPT를 더 나쁘게 만든다** — "While GPT-4 Turbo performs better as a baseline, MemGPT with GPT-4 Turbo performs worse than MemGPT with GPT-4" [MemGPT Fig. 7 캡션]. 논문은 설명하지 않는다. 둘째, **약한 모델에서는 기제가 작동하지 않는다** — "MemGPT has significantly degraded performance using GPT-3.5, due to its limited function calling capabilities" [MemGPT §3.2.1]. 셋째, **중심 이론적 장점이 실현되지 않는다.** 무한 페이징으로 검색기 한계를 넘는다는 것이 논지인데 관측은 그 반대다 — "we observe that MemGPT will often stop paging through retriever results before exhausting the retriever database" [MemGPT §3.2.1]. nested KV에서 MemGPT+GPT-4 Turbo와 MemGPT+GPT-3.5가 중첩 2에서 떨어지는 것도 "failing to perform enough lookups"로 귀속된다 [MemGPT §3.2.2]. 적용된 유일한 대책은 프롬프트에 대문자로 지른 고함이다 — "DO NOT STOP SEARCHING UNTIL YOU VERIFY THAT THE VALUE IS NOT A KEY" [MemGPT App. §6.1.6].

**ablation은 하나도 없다.** working context·recall·archival·압력 인터럽트·페이지네이션 중 어느 것도 제거하고 다시 재지 않았고, 구성요소에 대한 유일한 주장은 수치 없는 산문이다 — "we can see the storing information in working context is key to generating engaging openers" [MemGPT §3.1.2].

**실증 상한.** 가장 긴 대화는 MSC의 5개 세션에 저자가 만든 세션 6을 더한 것이다. 원문이 인쇄하는 값은 세션당 "roughly a dozen messages"뿐이고 [MemGPT §3.1], 메시지 60–70개라는 수는 그 표현에 5를 곱한 **이 책의 산수**이지 논문이 보고한 값이 아니다. 논문의 동기는 "weeks, months, or even years"에 걸친 대화이고 문서 쪽 동기는 "million token mark"를 넘는 SEC 10-K인데 [MemGPT §3.1, §3.2], 실제로 평가된 문서 작업량은 상위 200개 Wikipedia passage다 [MemGPT Fig. 5]. 저장소 상한은 archival storage에 벌크 적재된 Wikipedia 문서 2천만 건의 embedding이다 [MemGPT §3].

## 13.7 Systems/serving 함의

논문이 직접 진술한 사실부터 정리한다. **모든 기억 트래픽이 디코딩된다.** LLM은 프롬프트를 하나의 문자열로 받아 출력 문자열을 내고, MemGPT가 그것을 파싱하며, 런타임 오류를 포함한 결과가 다시 텍스트로 모델에게 들어간다 [MemGPT §2.3]. DMA도 별도 제어 평면도 없다. **archival 계층은 가속기 메모리가 아니라 데이터베이스다** — PostgreSQL + pgvector, HNSW index, "approximate, sub-second query times", text-embedding-ada-002로 사전 계산한 embedding [MemGPT §3.2.1]. **검색은 토큰 인식이다** — 페이지네이션으로 회수 결과가 문맥창을 넘치지 않게 한다 [MemGPT §2.3]. **압력 임계값은 프롬프트 길이 임계값이고 대역 내 텍스트로 배달된다** — 경고와 flush 사이에 $N_{\mathrm{ctx}}$의 약 30%를 여유로 두고 flush에서 창의 약 50%를 버린다 [MemGPT §2.2].

> **[해설]** 독자의 질문 — 그래서 decode에서 뭘 바꾸는가. **커널 안에서는 아무것도 바꾸지 않는다.** 새 상태도 새 GEMM도 attention 변경도 없다. 바뀌는 것은 모델 **위**다. 논리적 턴 하나가 예측 불가능한 횟수의 forward pass가 되고(식 (13-7)), 그 pass들이 도는 프롬프트의 앞부분을 모델 자신이 다시 쓴다. $E$-경로는 긴 문맥을 **가속기 상태 바이트**로 사지 않고 **decode 사이클과 prefix 캐시 지역성**으로 산다.

사이클 수의 하한은 논문 그림의 trace에서 읽힌다. Fig. 8의 nested KV 해결 과정은 archival 검색 네 번에 답변 한 번, 즉 질문 하나에 최소 다섯 사이클이고 각 사이클의 프롬프트가 직전보다 길다 [MemGPT Fig. 8]. Fig. 6은 Wikipedia 질문 하나에 페이지네이션된 검색 두 번을 보인다 [MemGPT Fig. 6]. **논문은 사이클 수를 세지 않으므로 이 값들은 그림의 trace에서 읽은 것이지 보고된 수치가 아니다.**

> **[평가]** 세 가지를 이 책의 판단으로 덧붙인다. 첫째, **shared-weight batching은 온전하다.** 어떤 파라미터도 사용자마다 움직이지 않으므로($\Theta$는 동결 API, $W=\varnothing$) 서빙 fleet은 가중치 사본 하나로 에이전트 사이를 평범한 채팅 세션처럼 배치한다. 사용자별 상태는 Postgres 행과 프롬프트 토큰에 살고 둘 다 기존 스택의 일급 객체다. 이것이 $E$-경로가 $W$·$\Theta$-경로에 대해 갖는 **구조적** 이점이며, Rosetta 표의 "batch로 weight 공유" 행이 왜 경로별로 갈리는지가 여기서 처음 드러난다.
>
> 둘째, **재귀 요약을 FIFO 0번에 두는 것은 prefix KV cache 재사용에 최악의 배치다.** 식 (13-1)의 레이아웃에서 flush마다 다시 쓰이는 $\sigma$가 앞쪽에 있으므로 뒤따르는 전부의 KV가 무효화되고, $E^{\mathrm{wc}}$에 대한 `append`·`replace`도 메시지 꼬리 전체에 같은 일을 한다. ch10 §10.5는 RAG가 prefix cache의 경계를 앞으로 민다고 했는데, MemGPT는 한 걸음 더 간다 — **경계 앞쪽 구간 자체를 모델이 다시 쓴다.** 논문은 KV cache를 한 번도 언급하지 않는다.
>
> 셋째, **턴당 이동 바이트는 저장소가 아니라 re-prefill이 지배한다.** Fig. 6에서 회수된 한 페이지는 텍스트 결과 열 개, 즉 킬로바이트 규모다 [MemGPT Fig. 6]. 그것을 삽입한 결과 — 다음 사이클에서 최대 $N_{\mathrm{ctx}}$ 토큰짜리 프롬프트에 attention을 다시 도는 일 — 은 자릿수가 다른 가속기 트래픽이다. $E$-경로는 메모리 계층을 거의 건드리지 않으면서 HBM 쪽 작업을 배로 늘린다. 논문은 두 양 중 어느 것도 보고하지 않는다.

운영 결손 둘을 마지막으로 적는다. **실패 양식이 서빙에서는 tail latency 문제로 번역된다.** 보고된 실패는 "페이징을 너무 일찍 멈춘다"이고 [MemGPT §3.2.1], 처방인 "더 오래 찾게 만든다"는 정확도 실패를 **무한 사이클 수** 실패로 바꾼다. 사이클 상한도 타임아웃도 기술되지 않는다. **그리고 용량 계획이 구조적으로 정의되지 않는다.** recall storage는 모든 것을 무기한 보관하므로 사용자당 저장 비용에 상한이 없고 [MemGPT §2.2], 정작 **쓸 수 있는** 사용자당 상태는 크기가 공개되지 않은 고정 $E^{\mathrm{wc}}$가 제한한다 [MemGPT §2.1].

## 13.8 한계와 bridge-out

논문 자신이 남긴 문제는 여덟 개다. (a) external context를 무슨 저장 기술로 받칠 것인가 — "integrating different memory tier technologies like databases or caches" [MemGPT §5]. (b) 기억 관리 **정책**을 어떻게 개선할 것인가 — 같은 문장이 "further improving control flow and memory management policies"를 적지만 [MemGPT §5] 정책을 **학습**시킨다는 선택지는 검토되지 않는다. (c) 에이전트가 충분히 오래 찾게 만드는 법 [MemGPT §3.2.1, §3.2.2]. (d) 왜 더 강한 기반 모델이 더 약한 MemGPT를 주는가 [MemGPT Fig. 7]. (e) 기억 계층의 **비용** — 제기조차 되지 않는다. (f) $E$가 무한히 커지면 어떻게 되는가 — 압축·축출·용량 정책이 없다 [MemGPT §2.2]. (g) 사실 정정 기제가 작동하는가 — `working_context.replace`는 예시될 뿐 측정되지 않는다 [MemGPT Fig. 4]. (h) open-weights 모델로 옮겨 가는가 — GPT-3.5의 실패를 "limited function calling capabilities"로 귀속시켜 놓고 검증하지 않는다 [MemGPT §3.2.1].

> **[평가]** **아무것도 학습되지 않는다는 사실을 이 장은 완곡화하지 않는다.** MemGPT의 기여 전부가 동결된 폐쇄 API 세 개 위의 prompt-engineered function calling이며 gradient도 loss도 이 방법에 속하는 파라미터도 없다 [MemGPT §3]. 따라서 이 논문의 모든 능력 주장은 MemGPT 아키텍처에 대한 주장인 만큼 gpt-4-0613의 지시 따르기 능력에 대한 주장이기도 하다.
>
> 그럼에도 이 장이 $E$-경로의 출발점인 이유는 넷이다. 첫째, **$\mathrm{wr}$에 처음으로 행위자가 생겼다** — ch10이 남긴 machine-writable 공백을 이 논문이 메운다. 둘째, **쓰기의 촉발 조건이 질의에서 용량으로 옮겨 갔다.** 식 (13-5)의 인터럽트는 이후 모든 $E$-경로 시스템이 어떤 형태로든 갖는 부품이다. 셋째, **$S(\cdot)$가 처음 등장한다.** 방향은 (U-E)와 반대지만(§13.3) 문맥을 압축해 다른 층으로 옮기는 연산이 파이프라인에 들어온 것은 여기가 처음이다. 넷째, **$B_s$가 들어갈 구멍이 뚫려 있다** — §2.4의 timed events 훅이다. 학습이 없다는 사실과 계보의 출발점이라는 사실은 모순되지 않는다. 이 논문이 공급한 것은 학습된 부품이 아니라 **인터페이스**이고, 이후 연구는 그 인터페이스 뒤에 무엇을 넣을 것인가를 다툰다.

**경로 간 인용.** MemGPT는 $W$-경로와 $\Theta$-경로 어느 쪽과도 접촉하지 않는다. 참고문헌 약 40개가 전부 장문맥 transformer 아키텍처, retrieval-augmented generation, LLM 에이전트, 평가 인프라다. fast weights, test-time training, 모델 편집, PEFT, 지속학습, catastrophic forgetting, complementary learning systems, 생물학적 sleep·consolidation 인용이 **0건**이다. 유일한 근접 사례는 §4가 장문맥 문단 안에서 인용하는 "neural memory (Lee et al., 2019)"인데, 그 참조는 Set Transformer이고 fast-weight·TTT 의미의 neural memory가 아니라 효율 기제다.

연대가 강제한 침묵과 선택된 침묵은 갈라야 한다. MemGPT는 2023년 10월(vendored 판본은 2024년 2월 v2)이므로 이후의 sleep-time 계열 논문을 인용할 수 없었고 그 부재는 흠이 아니다. 그러나 **그 시점에 이미 존재했고 인용할 수 있었던** $W$·$\Theta$-경로 문헌 — fast weight programmer 계열, test-time training, LoRA, ROME/MEMIT, EWC, CLS — 은 하나도 등장하지 않는다. RAG 쪽도 같은 방향으로 어긋난다. RAG의 참고문헌에는 elastic weight consolidation을 제목에 단 논문이 들어 있지만 [RAG §3.4, Table 2], 그것은 FEVER-2-way SotA 92.2라는 **표의 한 칸**으로만 쓰이고 catastrophic forgetting 내용은 아무 데도 쓰이지 않는다.

> **[평가]** 세 경로의 침묵은 이 책이 사후에 부과한 분류가 아니라 첫 경로의 첫 논문에서 이미 관측되는 사실이다. $E$-경로는 같은 정보를 $W$나 $\Theta$에 넣는 선택지를 **반박한 뒤** 출발한 것이 아니라 그 선택지를 **한 번도 제기하지 않은 채** 출발했다. 문제 정의가 "there is a critical need for alternative techniques to support long context"라는 형태이기 때문이다 [MemGPT §1] — *context*가 유일한 축으로 놓이면 파라미터 기억은 설계 공간에 들어오지 않는다. 이 관측이 Part III 판정의 재료가 된다(→ ch27).

**ch14가 받아 가는 것.** 다음 장은 같은 저자들이 그다음에 무엇을 했는가를 다루고, 넘어가는 물건은 셋이다. 첫째, §2.4의 timed events 훅이 실제로 실행되고 그 계산에 $B_s$라는 이름이 붙는다. 둘째, 식 (13-3)의 쓰기가 사용자 턴 밖으로 나가면서 판별식 조건 (1)이 해소된다. 셋째, $\hat c$의 내용이 바뀐다 — MemGPT의 $\hat c$는 대화에서 골라낸 사실 문장이지만 다음 장의 $\hat c$는 질의를 보기 전에 수행한 **추론의 산출물**이다. 물려받지 **못하는** 것도 있다: 비용 4종이 전부 비어 있으므로 상각 논증을 세울 기준선이 이 장에는 없다.

## 요약

- MemGPT가 명시적으로 상속한 문제는 Lost in the Middle이다 — "there is a critical need for alternative techniques to support long context" [MemGPT §1]. 상속은 수사에 그치지 않고 벤치마크 두 개를 함께 물려받는다.
- 이 장의 관점은 **계층화된 외부 기억**이다. 이동을 촉발하는 것이 질의가 아니라 상주 층의 용량 초과가 되면 $E$ 설계의 1차 변수가 검색 품질에서 잔류 정책으로 바뀐다. 논문이 붙인 OS 은유는 세 곳에서 샌다 — page fault가 자발적 경고이고, 주소 변환·보호·상주 보장이 없으며, 인용된 유일한 OS 문헌이 RAID 논문이다 [MemGPT §1, §2.2].
- **아무것도 학습되지 않는다.** 동결된 폐쇄 API 세 개 위의 prompt-engineered function calling이며 gradient도 loss도 이 방법의 파라미터도 없다 [MemGPT §3].
- 쓰기가 대화 **중** 임계 경로 위에서 일어나므로 판별식 조건 (1)을 만족하지 않는다. 구조적 조상이되 sleep-time compute은 아니다 — $\mathrm{wr}$·$S(\cdot)$·인터럽트를 공급하고 $B_s$를 공급하지 않는다.
- 비용 4종이 **전부** 비어 있고 $N_q$도 없어 식 (A)를 채울 수 없다. 그럼에도 식 (13-7)이 턴 하나를 여러 사이클로 만들므로 답 하나당 비용은 반드시 baseline보다 크다.
- 실험은 ablation 0건이고, DMR 비교가 정보 동등하지 않으며, GPT-4 baseline이 GPT-3.5 baseline보다 낮고(32.1% 대 38.7%), 판정기는 관대하게 지시받은 동계열 모델이며, Table 3에는 MemGPT를 쓰지 않은 LLM 행이 없다 [MemGPT Table 2, Table 3, §3.1.1].
- RAG의 평가설계 결함 여덟 항 — TriviaQA split 의존성(56.8 대 DPR 57.9), MS-MARCO 비대칭 비교, 저자 제안 Jeopardy와 452쌍 인간 평가, Table 4 Specificity 합 93.0%, 산문-표 불일치, null-document 3변형 무수치 폐기, RAG-Seq/Tok 순서 비일관, 단일 corpus 증거 — 은 기제가 아니라 **이득의 크기**를 읽어낼 수 있다는 믿음을 반증한다.
- 서빙 관점에서 커널 안은 아무것도 바뀌지 않는다. 바뀌는 것은 턴당 forward pass 수와 prefix cache 지역성이다. 재귀 요약이 프롬프트 앞쪽에 있어 flush마다 뒤따르는 KV 전부가 무효화되고, 논문은 KV cache를 한 번도 언급하지 않는다.

## 자가 점검 체크리스트

- [ ] MemGPT의 상속을 원문 인용으로 복원할 수 있고, 그것이 "open question"이라는 표현 없이 두 구절과 벤치마크 재사용으로 복원된 것임을 밝힐 수 있다.
- [ ] 식 (13-3)과 (13-4)가 왜 성격이 정반대인 두 개의 $\mathrm{wr}$인지, 식 (13-6)의 $S(\cdot)$가 왜 (U-E)의 $S(c;B_s)$와 방향이 반대인지 말할 수 있다.
- [ ] 표 13-2의 전 행에 대해 "누가 언제 학습하는가"를 답할 수 있고, 답이 전부 같다는 사실이 왜 계보의 출발점 자격을 무효화하지 않는지 네 가지 이유로 말할 수 있다.
- [ ] MemGPT가 판별식 네 조건 중 무엇을 만족하지 못하는지 지목하고, 그럼에도 $E$-경로에 두는 근거 둘을 댈 수 있다.
- [ ] 비용 4종이 전부 "논문에 없음"인 상태에서 단정할 수 있는 비용 명제(턴당 사이클 수 증가)와 단정할 수 없는 명제(그 배수)를 가를 수 있다.
- [ ] RAG 평가설계 여덟 항 중 넷을 수치와 함께 인용하고, 비교 조건 문제와 표 내적 정합성 문제를 구분할 수 있다.
- [ ] Rosetta: "prefix cache ↔ $\hat c$"가 왜 무효화 문제로 나타나고 "KV cache ↔ $W$"가 왜 아예 걸리지 않는지($W=\varnothing$) 각각 한 문장으로 설명할 수 있다.
