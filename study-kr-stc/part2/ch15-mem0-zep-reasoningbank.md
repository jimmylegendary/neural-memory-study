# ch15. Mem0 · Zep · ReasoningBank · SCM — 프로덕션과 그 대가

## 15.1 Bridge-in: 전작이 남긴 문제

이 장의 다섯 시스템 중 어느 것도 ch14를 인용하지 않는다.

사실부터 고정한다. Zep(arXiv 2501.13956)은 2025년 1월 20일자이고 ch14의 논문(arXiv
2504.13171)보다 석 달 앞선다. Mem0(arXiv 2504.19413)은 2025년 4월 28일자로 열흘 뒤에
나왔다. ReasoningBank(arXiv 2509.25140)은 이 책이 읽은 판본이 v2(2026년 3월 16일)이고,
SCM(arXiv 2604.20943)은 2026년 4월, Memini(arXiv 2605.05097)는 v3가 2026년 6월이다.
앞의 둘은 ch14를 인용하지 못할 연대적 이유가 있다. 뒤의 셋은 그렇지 않다. ReasoningBank
전문에 "sleep"이라는 문자열은 한 번도 나오지 않고, Memini 9쪽에도 나오지 않으며, SCM은
제목에 sleep을 걸고도 ch14를 참고문헌에 넣지 않았다.

그렇다면 이들은 무엇을 상속했는가. 다섯 논문의 어느 것도 선행 논문이 **명시적으로 남긴
open question을 인용하지 않는다.** Zep은 MemGPT가 남긴 질문이 아니라 MemGPT의 *점수*를
다툰다 — "In the DMR benchmark, which the MemGPT team established as their primary
evaluation metric, Zep demonstrates superior performance (94.8% vs 93.4%)" [Zep Abstract].
Mem0은 MemGPT의 계층화 질문에 답하되 그 사실을 밝히지 않고, 학습된 정책을 쓰자는 선택지는
명시적으로 거절한다 — "Rather than using a separate classifier, we leverage the LLM's
reasoning capabilities to directly select the appropriate operation" [Mem0 §2.1].
ReasoningBank은 AWM과 Synapse를 자기 목소리로 진단하고, SCM과 Memini는 각각 §2에서
선행 연구 세 갈래를 저자 자신의 판정으로 반박한다.

> **[평가]** 이것은 인용 누락이 아니라 사슬의 종류가 다른 것이다. ch14까지의 $E$-경로는
> 질문을 주고받았고, 이 장의 다섯은 **리더보드를 주고받는다.** 대화 상대가 경쟁 *질문*이
> 아니라 경쟁 *제품*이면, 상속할 것은 벤치마크와 점수뿐이고 비용 회계는 상속되지 않는다.
> 다섯 논문이 서로의 $B_s$를 인용하지 못하는 이유가 여기 있다 — 인용할 수 있는 $B_s$를
> 아무도 인쇄한 적이 없기 때문이다.

ch14가 남긴 문제는 두 가지였다. 하나는 저자 제작 벤치마크에 걸린 이득을 어떻게 읽을
것인가이고, 다른 하나는 생성 토큰만 세는 비용 모델이 무엇을 감추는가였다. 이 장의 다섯은
첫 번째 문제를 절반쯤 개선하고(Mem0·Zep·ReasoningBank는 남이 만든 벤치마크를 쓴다)
두 번째 문제를 그대로 물려받는다. 정확히는, 더 나빠진 형태로 물려받는다. ch14는 적어도
sleep 예산을 세려고 시도했다. 이 장에서 $B_s$를 숫자로 인쇄한 논문은 다섯 중 하나뿐이다.

## 15.2 문제의식

다섯 논문의 문제 진술은 놀랄 만큼 같은 모양이다. 고정된 문맥창, 계속 자라는 스트림,
그리고 창을 키워도 해결되지 않는다는 주장이다.

- "these systems are fundamentally limited by their reliance on fixed context windows,
  which severely restrict their ability to maintain coherence over extended interactions"
  [Mem0 §1]. 창을 키우는 것은 지연일 뿐이라는 근거로 두 가지를 든다 — 관계는 어떤 창보다
  오래 지속되고, 실제 대화는 주제적으로 연속하지 않는다 [Mem0 §1].
- "Current approaches using RAG have focused on broad domain knowledge and largely static
  corpora—that is, document contents added to a corpus seldom change" [Zep §1].
- "By approaching each new task in isolation, they are doomed to (i) repeat similar errors
  observed in the past, (ii) discard valuable insights gained from related problems"
  [ReasoningBank §1].
- LLM은 "remain fundamentally amnesic. Each conversation begins anew" [SCM §1].
- "A system whose memory merely grows is accumulating. A system whose memory consolidates,
  forgets, and reshapes its organization in response to experience is learning"
  [Memini §1].

같은 모양이지만 답은 다섯 방향으로 갈라진다. Mem0은 **무엇을 얼마나 적게 다시 읽을
것인가**를 묻고, Zep은 **어떤 사실이 언제 참이었는지를 어떻게 기록할 것인가**를 묻고,
ReasoningBank은 **경험을 어떤 형태로 저장해야 재사용되는가**를 묻고, SCM은 **무엇을 지울
것인가**를 묻고, Memini는 **저장소 자체가 동역학계이면 어떻게 되는가**를 묻는다.

여기서 한 가지가 빠져 있다. 다섯 중 어느 것도 이 문제를 **계산을 언제 쓸 것인가**의
문제로 세우지 않는다. Mem0 전문에서 "sleep", "offline", "test-time", "inference-time"은
0회 등장하고, Zep 전문에서 "sleep", "consolidat", "offline", "amortiz"는 0회이며,
Memini 전문에서 "sleep"과 "offline compute"는 0회다. ch01의 조작적 정의는 예산 $B_s$를
언제 쓰느냐로 이 범주를 가르는데, 이 논문들의 자기 진술은 **표현과 비용의 문제**이지
**시점의 문제**가 아니다. 그러면서도 다섯 모두 판별식은 통과한다 — 질의가 도착하기 전에
$E$의 상태를 실제로 바꾸고 그 변화가 이후 질의에 쓰이기 때문이다(→ ch11).

이 어긋남이 이 장의 진짜 주제다. 다섯 논문은 각자 다른 것을 고치겠다고 선언하지만,
자기 표를 그대로 읽으면 **품질 1위가 반복적으로 "기억 시스템 없음"이다.** Mem0의
Table 2에서 최고 J 점수는 full-context 72.90 ± 0.19%이고 Mem0의 66.88 ± 0.15%보다
6.02점 높다. Zep의 Table 1에서 gpt-4-turbo full-conversation 94.4%는 이미 비교
대상인 MemGPT 93.4%를 넘어선다. SCM의 Table 2에서 아무것도 지우지 않는 No-Forget
Graph는 SCM과 똑같이 22/22를 맞힌다. ReasoningBank은 검색 경험을 하나에서 둘로 늘리면
49.7에서 46.0으로 떨어진다. Memini는 검색을 한 번도 돌리지 않았다.

이 관찰을 다섯 번 반복하는 것이 이 장의 논증이고, §15.8의 [평가]가 그 결론이다.

## 15.3 Core mechanism (통일 표기)

다섯 시스템은 읽기에서 완전히 같은 식을 쓴다. 표준형 (R)에서 $W$ 자리가 비어 있다.

$$
\hat y \;=\; f\big(q;\ \Theta,\ \varnothing,\ \mathrm{ret}(E, q)\big)
\tag{15-1}
$$

$\Theta$는 다섯 논문 전체에서 단 한 번도 움직이지 않는다. Mem0은 모든 호출 지점에
GPT-4o-mini를 쓰고 [Mem0 §2.1], Zep은 그래프 구축에 gpt-4o-mini-2024-07-18을 쓰며
[Zep §4.1], ReasoningBank은 Vertex AI로 Gemini-2.5-Flash/Pro·Claude-3.7-Sonnet을 부르고
[ReasoningBank §4.1], SCM은 "It requires no training or fine-tuning; all components use
existing pretrained models or algorithmic logic" [SCM §3.8]이라고 명시하며, Memini는
"The backbone can be replaced or upgraded without disturbing what the memory has learned"
[Memini §3.1]라고 쓴다. 이 장에는 gradient가 하나도 없다. 손실 함수도 $\ell$도
$\mathcal{L}$도 등장하지 않는다.

쓰기도 모두 (U-E)이고, 차이는 전부 $S(\cdot;B_s)$와 $\mathrm{wr}(\cdot,\cdot)$의 모양에
있다. 다섯의 이탈을 표준형과의 차이로 적으면 이렇다.

**Mem0 — 재귀적 인자.** 요약이 저장소에서 검색되어 추출 프롬프트로 들어가므로 $S$의
인자에 $E$ 자신이 들어간다 [Mem0 §2.1].

$$
E_{k+1} \;=\; \mathrm{wr}\big(E_k,\ S(c_k,\, E_k;\ B_s)\big)
\tag{15-2}
$$

표준형에는 이 되먹임 경로가 없다. 결과적으로 $k$ 라운드의 쓰기가 $k{+}1$ 라운드의 추출
품질을 바꾼다. 트리거 단위는 메시지쌍 $(m_{k-1}, m_k)$이고 최근 창은 $m_{\mathrm{rec}}=10$,
갱신 판단에 참조하는 이웃은 $s_{\mathrm{nn}}=10$이다 [Mem0 §2.1].

**Zep — 타입 있는 그래프 delta.** $\hat c_k$가 문자열이 아니라 다섯 종류의 델타 묶음이다
[Zep §2.2].

$$
\hat c_k \;=\; \big(\Delta\mathcal{N}_e,\ \Delta\mathcal{E}_e,\ \Delta\mathcal{N}_s,\
\Delta\mathcal{E}_s,\ \Delta\mathcal{N}_c\big)
\tag{15-3}
$$

이 델타를 만드는 $S(\cdot;B_s)$는 아홉 단계다 — 개체 추출, reflexion 재확인, 1024차원
임베딩, 개체 해소, 사실 추출, 간선 중복 제거, 시간 추출, 모순 검사·무효화, community
라벨 전파 [Zep §2.2.1–§2.3]. 이 중 여섯에서 일곱 단계가 frozen LLM 호출이다. $S$가 보는
문맥은 현재 메시지와 직전 네 개, 즉 $n_{\mathrm{ctx}}=4$로 고정된다 [Zep §2.2.1].

**ReasoningBank — 라벨로 분기하는 요약.** $S$의 인자가 문맥이 아니라 에이전트 자신의
궤적이고, 자기 판정 라벨 $j_k$가 프롬프트를 고른다 [ReasoningBank §3.2].

$$
\hat c_k \;=\; S\big(\mathcal{H}^{\Delta}_k;\ B_s,\ j_k\big),
\qquad
j_k \in \{\text{success},\ \text{failure}\},
\qquad
|\hat c_k| \le 3
\tag{15-4}
$$

성공 분기는 검증된 전략을, 실패 분기는 예방 교훈을 뽑는다. 두 프롬프트 모두 "Do not
mention specific websites, queries, or string contents"라는 제약을 건다
[ReasoningBank App A.1 Fig. 9].

> **[평가]** 이 논문이 기여로 내세우는 추상화는 아키텍처가 아니라 프롬프트 한 줄에서
> 나온다. (15-4)에서 $S$의 모양을 정하는 것은 라벨 $j_k$가 고르는 프롬프트뿐이고,
> 학습되는 성분도 강제되는 스키마도 없다.

**SCM — 요약이 wake로 옮겨간 경우.** 추출은 매 턴 wake에서 일어나고 [SCM §3.2], sleep
예산은 **이미 쓴 것을 재조직하고 지우는 데만** 쓰인다.

$$
E_{k+1} \;=\; \mathrm{prune}_{\vartheta_{\mathrm{f}}}\Big(
\mathrm{dream}\big(\alpha^{E}\cdot \mathrm{hebb}(E_k,\ E^{\mathrm{wm}}_k)\big)\Big),
\qquad \Theta_{k+1}=\Theta_k
\tag{15-5}
$$

간선 강도의 갱신만 떼어 쓰면 아래 형태가 되는데, 이것은 (U-W)의 retention gate와
대수적으로 같은 모양이다 — 다만 상태가 $W$가 아니라 $E$의 스칼라이고, 쓰기 항이
gradient가 아니라 중요도의 곱이며, $\alpha^E$가 데이터 의존 gate가 아니라 상수 0.8이다
[SCM Eq. 6, Eq. 7].

$$
s_{ij}^{(k+1)} \;=\; \alpha^{E}\Big(s_{ij}^{(k)} \;+\; \eta_{\mathrm{heb}}\,I(c_i)\,I(c_j)\Big),
\qquad \alpha^{E}=0.8,\ \ \eta_{\mathrm{heb}}=0.1
\tag{15-6}
$$

> **[해설]** 반복 강화되는 간선의 고정점은 $s^\ast = 0.08\,I(c_i)I(c_j)/0.2 = 0.4\,I(c_i)I(c_j)$이고,
> 강화되지 않는 간선은 $k$ 사이클 뒤 $0.8^k$로 줄어 20 사이클이면 원래의 약 1.15%가
> 남는다. 이 장에서 유일하게 대수적으로 닫히는 용량 상한인데, 논문은 이 계산을 하지 않는다.

**Memini — 저장소 전체가 선형 필터 뱅크.** 간선마다 읽히는 빠른 성분 $E^{\mathrm{f}}_e$와
읽히지 않는 느린 성분 $E^{\mathrm{s}}_e$가 결합한다 [Memini Eq. 1, Eq. 2].

$$
\dot z_e \;=\; A\,z_e + \begin{bmatrix} I_e(t) \\ 0\end{bmatrix},
\qquad
z_e = \begin{bmatrix} E^{\mathrm{f}}_e \\ E^{\mathrm{s}}_e\end{bmatrix},
\qquad
A = \begin{bmatrix}
-\big(\tfrac{1}{\tau_{\mathrm{f}}}+\mu\big) & \mu \\[2pt]
\mu & -\big(\tfrac{1}{\tau_{\mathrm{s}}}+\mu\big)
\end{bmatrix}
\tag{15-7}
$$

행렬형은 논문에 없다. 논문이 쓴 것은 두 개의 스칼라 미분방정식과 "a forward Euler step
$\Delta t = 1$ corresponding to one document per step"이라는 이산화 규칙뿐이다
[Memini App. A.2]. $A$가 모든 간선에 대해 같고 $I_e$가 임펄스열이므로, Memini의 쓰기
층은 $|\mathcal{E}|$개의 동일한 2극 선형 필터 뱅크다. 여기서도 학습되는 것은 없다.

### 15.3.1 쓰기 연산자의 종류

이 절이 도입하는 첫 번째 정의다. $E$-경로의 쓰기 연산자는 **저장소의 크기를 어떻게
움직일 수 있는가**로 분류된다.

> **정의.** 쓰기 연산자 $\mathrm{wr}(E,\hat c)$가 **확장 전용(append-only)**이라 함은
> 모든 입력에 대해 $|E_{k+1}| \ge |E_k|$가 성립하는 경우를 말하고, **수축 가능
> (contraction-capable)**이라 함은 어떤 입력에 대해 $|E_{k+1}| < |E_k|$가 되는 경우를
> 말한다. 수축 가능한 연산자는 $\rho$를 실재하는 위험량으로 만든다.

이 축 위에서 다섯 시스템의 연산자는 네 가지 형태로 나뉜다.

> **정의.** **네 연산 문법(four-operation grammar)**은 후보 사실 $\omega_i$마다
> frozen LLM이 $\{\textsf{ADD},\ \textsf{UPDATE},\ \textsf{DELETE},\ \textsf{NOOP}\}$
> 중 하나를 함수 호출로 고르는 쓰기 규칙이다 [Mem0 §2.1, Alg. 1]. $\textsf{ADD}$는
> $E \leftarrow E \cup \{(\mathrm{id},\omega)\}$, $\textsf{UPDATE}$는 기존 항목의 교체,
> $\textsf{DELETE}$는 되돌릴 수 없는 제거, $\textsf{NOOP}$는 항등이다.

> **정의.** **합집합 쓰기(union write)**는 $\mathrm{wr}(E_k, \hat c_k) = E_k \cup \hat c_k$,
> 즉 병합·중복 제거·감쇠·축출이 전혀 없는 순수 합집합이다 [ReasoningBank §3.2].
> 논문 자신의 표현으로는 "a minimal consolidation strategy: newly generated items are
> directly added without additional pruning" [ReasoningBank App A.2]이다.

> **정의.** **그래프 delta 쓰기**는 $\hat c$가 문자열이 아니라 노드·타입 간선·메타데이터의
> 델타 묶음이고, 쓰기가 스키마로 강제되는 형태다 [Zep §2.2]. Zep은 이 델타를 LLM이 만든
> 질의가 아니라 미리 정의된 Cypher로 적용한다 — "We chose this approach over LLM-generated
> database queries to ensure consistent schema formats and reduce the potential for
> hallucinations" [Zep §2.2.1].

네 번째는 **동역학 쓰기**다. SCM과 Memini의 $\mathrm{wr}$은 인자에 없는 항목까지 건드린다.
SCM은 사이클마다 모든 간선에 $\alpha^E$를 곱하고 임계 미달 노드를 지우며, Memini는 문서가
하나 도착할 때마다 감쇠와 결합 항을 **모든 간선에** 적용한다 [Memini Eq. 1–2;
App. A.2]. 표준형 (U-E)는 $O(|\hat c_k|)$개 레코드를 건드리지만 이 둘은 $O(|\mathcal{E}_k|)$개를
건드린다.

표 15-1. 다섯 시스템의 쓰기 연산자

| 시스템 | $\mathrm{wr}$의 형태 | 수축 가능성 | 논문의 근거 위치 |
|---|---|---|---|
| Mem0 (base) | 네 연산 문법, frozen LLM이 선택 | **가능** — 하드 $\textsf{DELETE}$ | [Mem0 §2.1, Alg. 1] |
| Mem0$^g$ | 그래프 delta + 무효화 표시 | 불가 — 삭제 대신 INVALID 표시 | [Mem0 §2.2] |
| Zep | 그래프 delta, 스키마 강제 Cypher | 불가 — 전 계층 append-only | [Zep §2, §2.2.3] |
| ReasoningBank | 합집합, 항목 상한 3개(MaTTS 5개) | 불가 | [ReasoningBank §3.2] |
| SCM | 동역학: hebb → 수축 → dream → prune | **가능** — 임계 미달 노드 삭제 | [SCM Alg. 1] |
| Memini | 동역학: 전 간선 forward Euler 1스텝 | 불가 — 가중치만 감쇠, 간선 제거 규칙 없음 | [Memini Eq. 1–2] |

Mem0 한 논문 안에서 두 정책이 화해 없이 공존한다는 사실은 그대로 적어야 한다. base는
모순된 기억을 하드 삭제하고 [Mem0 Alg. 1], 그래프 변형은 "marking them as invalid rather
than physically removing them to enable temporal reasoning" [Mem0 §2.2]라고 쓴다. 논문은
둘의 불일치를 지적하지도, 둘 중 어느 쪽이 옳은지 측정하지도 않는다. 게다가 $\textsf{UPDATE}$
분기의 가드인 `InformationContent(f) > InformationContent(m_i)`에서
$\mathrm{InformationContent}$는 논문 어디에도 정의되어 있지 않다 [Mem0 Alg. 1].

### 15.3.2 시간을 표현하는 방식

두 번째 정의다. 이 장의 시스템들은 시간을 세 가지 방식 중 하나로 표현하며, 셋은 서로
바꿔 쓸 수 없다.

> **정의.** **명시적 타임스탬프 방식**은 저장 레코드에 절대 시각을 필드로 붙이는 방식이다.
> 그 극단이 **이중 시간(bi-temporal) 표현**으로, 사실이 세계에서 참이었던 구간을 다루는
> 타임라인 $T$와 시스템이 그것을 알게 되거나 철회한 순서를 다루는 타임라인 $T'$를
> 독립적으로 유지한다 [Zep §2.1]. Zep은 의미 간선마다 네 개의 시각을 단다 —
> $t^{\mathrm{wall}}_{\mathrm{valid}},\ t^{\mathrm{wall}}_{\mathrm{invalid}} \in T$와
> $t'_{\mathrm{created}},\ t'_{\mathrm{expired}} \in T'$ [Zep §2.2.3].

이 표현이 낳는 쓰기 규칙이 **구간 폐쇄(interval closure)**다. 모순되는 새 간선이 오면
옛 간선을 지우는 대신 그 유효 구간을 닫는다.

$$
\mathrm{contradict}(e_{\mathrm{new}}, e_{\mathrm{old}}) \wedge
\mathrm{overlap}_T(e_{\mathrm{new}}, e_{\mathrm{old}})
\;\Rightarrow\;
t^{\mathrm{wall}}_{\mathrm{invalid}}(e_{\mathrm{old}}) \leftarrow
t^{\mathrm{wall}}_{\mathrm{valid}}(e_{\mathrm{new}})
\tag{15-8}
$$

> **정의.** **감쇠 상수 방식**은 시각을 저장하지 않고, 상태가 시간에 따라 줄어드는
> 속도로 시간을 표현하는 방식이다. SCM은 $\delta(c)=\exp(-\lambda_{\mathrm{rec}}\Delta t)$,
> $\lambda_{\mathrm{rec}}=0.01$을 쓰고 [SCM Eq. 9], Memini는 두 시상수
> $\tau_{\mathrm{f}}=2$, $\tau_{\mathrm{s}}=10$을 **문서 단위로** 쓴다 [Memini App. A.2].
> 이 방식에서 "언제"는 절대 시각이 아니라 "그 뒤로 몇 번의 이벤트가 지났는가"다.

> **정의.** **순서 첨자 방식**은 시간을 라운드 번호로만 표현하고 지속 시간을 아예 모형화하지
> 않는 방식이다. ReasoningBank은 과제 스트림의 순서 $k=1..N$만 갖는다. 항목에 시각도
> 없고 감쇠도 없다 [ReasoningBank §3.1–§3.2].

세 방식은 서로 다른 실패 모드를 갖는다. 명시적 타임스탬프는 정확하지만 **모델이 그것을
읽어야** 쓸모가 있다. Zep의 이중 시간은 결국 프롬프트 안에 문자열로 들어간다 —
템플릿 머리글이 "These are the most relevant facts and their valid date ranges. ...
format: FACT (Date range: from - to)" [Zep §3]이다. 시간 임베딩도, 검색 단계의 유효성
필터링도, 시간 전용 attention도 없다. 그래서 §15.6에서 보게 될 결과 — 약한 모델에서
knowledge-update가 **퇴행한다**는 것 — 은 표현의 결함이 아니라 표현이 추론자에게
전가된 결과다. 감쇠 상수 방식은 절대 시각을 버리는 대신 재언급되지 않은 사실을 조용히
잃고, 순서 첨자 방식은 시간을 아예 다루지 못한다.

### 15.3.3 표기 대응표

다섯 논문의 국소 기호를 이 책의 예약 기호로 옮긴다. 본문에서는 오른쪽 열만 쓴다.

표 15-2. 논문 기호 → 이 책의 표기

| 논문 | 원 기호 (원문에서의 뜻) | 이 책의 표기 |
|---|---|---|
| Mem0 | $S$ (대화 요약 텍스트, §2.1) | $\hat c^{\mathrm{sum}}$ — $E$의 한 원소 |
| Mem0 | $E$ (그래프의 간선 집합, §2.2) | $\mathcal{E}$ |
| Mem0 | $L$ (노드 타입 라벨, §2.2) | $L_{\mathrm{lab}}$ |
| Mem0 | $M$ (기억 저장소, Alg. 1) | $E$ |
| Mem0 | $m$ (메시지 / 최근 창 크기 / 저장 항목, §2.1) | $m_k$ / $m_{\mathrm{rec}}=10$ / $E$ 색인 |
| Mem0 | $t$ (메시지 색인 / 노드 생성 시각 / 병합 임계, §2.1–2.2) | $k$ / $t_v$ / $t_{\mathrm{sim}}$ |
| Mem0 | $k$ (검색 청크 수, §3.3) | $k_{\mathrm{RAG}}$ |
| Mem0 | $s$ (갱신 판단에 쓰는 유사 기억 수, §2.1) | $s_{\mathrm{nn}}=10$ |
| Mem0 | $\phi$ (LLM 추출 함수, §2.1) | $S(\cdot;B_s)$ |
| Mem0 | $\Omega=\{\omega_1..\omega_n\}$ (후보 사실 집합) | $\hat c$ (집합값) |
| Mem0 | $F$ (Alg. 1의 검색된 사실 집합) | $\hat c$ |
| Mem0 | $P$ (추출 프롬프트 튜플, §2.1) | $P_k$ |
| Mem0 | $J$ (LLM-as-a-Judge 점수, §3.2) | $J$ (유지) |
| Zep | $E$ (그래프 간선 집합, §2) | $\mathcal{E}$ |
| Zep | $\rho$ (재순위 함수, §3) | $\mathrm{rrank}$ |
| Zep | $\alpha$ (질의 문자열, §3) | $q$ |
| Zep | $\beta$ (출력 문맥 문자열, §3) | $\mathrm{ret}(E,q)$ |
| Zep | $S$ (문자열 집합, §3) | $\Sigma^{*}$ |
| Zep | $f$ (그래프 검색 API — 생성 아님, §3) | $\mathrm{ret}(E,q)$ |
| Zep | $\varphi$ / $\phi$ (부착 함수 / 검색 단계, §2·§3) | $\varphi$ / $\mathrm{srch}$ |
| Zep | $\chi$ (문맥 구성기, §3) | $\mathrm{cons}$ |
| Zep | $t_{\mathrm{ref}},t_{\mathrm{valid}},t_{\mathrm{invalid}},t'_{\mathrm{created}},t'_{\mathrm{expired}}$ | $t^{\mathrm{wall}}_{\cdot}$, $t'_{\cdot}$ |
| Zep | $T$, $T'$ (두 타임라인, §2.1) | $T$, $T'$ (유지) |
| Zep | $n$ (직전 메시지 수 / BFS 홉 / 리스트 길이) | $n_{\mathrm{ctx}}=4$ / $h$ / $|\cdot|$ |
| Zep | $G, G_e, G_s, G_c$ (그래프와 세 계층) | $E$, $\mathcal{G}_e,\mathcal{G}_s,\mathcal{G}_c$ |
| Zep | $N, N_e, N_s, N_c$ (노드 집합) | $\mathcal{N}_e,\mathcal{N}_s,\mathcal{N}_c$ |
| Zep | $e_i$, $n_i$ (개별 간선·노드) | 유지 |
| ReasoningBank | $M$ (기억 모듈, §3.1) | $E$ |
| ReasoningBank | $L$ (backbone LLM, §3.1) | $\Theta$ (frozen) |
| ReasoningBank | $\pi_L(\cdot\mid M,A)$ (에이전트 정책) | $f(q;\Theta,\cdot,\mathrm{ret}(E,q))$ |
| ReasoningBank | $\mathcal{H}$ (단일 궤적, Fig. 2) | $\mathcal{H}^{\Delta}_k$ (라운드 증분) |
| ReasoningBank | $k$ (MaTTS 스케일 인자, §3.3) | $m$ |
| ReasoningBank | $k$ (검색 top-$k$, 기본 1, App A.2) | $n_{\mathrm{ret}}=1$ |
| ReasoningBank | $t$ (궤적 내부 스텝) | $t$ (틱 = 환경 상호작용 1회) |
| ReasoningBank | $N$ (과제 스트림 길이, §3.1) | $N$ — $N_q$와 다름 |
| ReasoningBank | $q_i$, $Q$ (과제 질의·질의열) | $q_k$ |
| ReasoningBank | $\mathcal{A}/A$ (행동 공간), $T(\cdot)$ (전이) | 유지 |
| ReasoningBank | SR, AS (성공률·평균 스텝, App B.1) | 유지 — AS는 $L_w$가 아니다 |
| SCM | $W$ (7칸 working memory 버퍼, §3.4) | $E_{\mathrm{wm}}$ |
| SCM | $S(c)$ (보존 점수, Eq. 9) | $r_{\mathrm{ret}}(c)$ — 맨 $r$은 rank 예약이므로 첨자를 단다 |
| SCM | $s_{ij}$ (간선 강도, Eq. 6–7) | $s_{ij}$ 유지 — 장-국소 행. 첨자를 둘 달아 step 첨자 $s$와 구분한다 |
| SCM | $\eta = 0.1$ (Hebbian 강화율, Eq. 6) | $\eta_{\mathrm{heb}}$ — inner의 $\eta_t$·sleep의 $\eta_\Theta$와 다르다 |
| SCM | $\alpha = 0.8$ (시냅스 축소, Eq. 7) | $\alpha^{E}$ |
| SCM | $\beta_1=0.8,\ \beta_2=0.2$ (보존 점수 혼합, Eq. 9) | $w_{\mathrm{imp}},\ w_{\mathrm{rec}}$ |
| SCM | $\lambda = 0.01$ (최근성 감쇠, Eq. 9) | $\lambda_{\mathrm{rec}}$ |
| SCM | $\rho(G)$ (모순 간선 밀도, §3.6) | $\rho_{\mathrm{cf}}$ |
| SCM | $\theta_e,\theta_c,\theta_f$ (트리거·삭제 임계, §3.6) | $\vartheta_{\mathrm{e}},\vartheta_{\mathrm{c}},\vartheta_{\mathrm{f}}$ |
| SCM | $e_c$ (384차원 문장 임베딩, Eq. 1) | $x_c \in \mathbb{R}^{384}$ |
| SCM | $e$ (버퍼의 episode 객체, Alg. 1) | 산문 '에피소드' |
| SCM | $M$ (기존 개념 집합, Eq. 1) | $E_{\mathrm{ltm}}$ |
| SCM | $E$ ($|\mathcal{E}_{\text{contradicts}}|/|E|$의 분모, §3.6) | $\mathcal{E}$ |
| SCM | $\sigma_I$ (중요도 표준편차, Eq. 10) / $\sigma$ (활성함수, Eq. 2) | $\mathrm{sd}(I)$ / $\sigma(\cdot)$ |
| SCM | $c$ (개념, Eq. 1–5) | $c$ (장-국소 재정의; 이 장에 $\Omega$ 규칙 없음) |
| SCM | $G$ (장기 기억 그래프) | $E_{\mathrm{ltm}}$, $|G| \to |\mathcal{V}|$ |
| SCM | $H(W)$ (버퍼 중요도 분포의 엔트로피) | $H(E_{\mathrm{wm}})$ |
| SCM | $\tau = 1\text{시간}$ (경과 시간 트리거) | $\tau_{\mathrm{sleep}}$ |
| Memini | $w_{\mathrm{fast}}$ (읽히는 간선 변수) | $E^{\mathrm{f}}_e$ |
| Memini | $w_{\mathrm{slow}}$ (읽히지 않는 간선 변수) | $E^{\mathrm{s}}_e$ |
| Memini | $C$ (두 변수의 결합 강도, Eq. 1–2) | $\mu$ |
| Memini | $E$ (그래프 간선 집합) / $V$ (노드 집합) | $\mathcal{E}$ / $\mathcal{V}$ |
| Memini | $u_i^{(t)}$ (확산 활성값, Eq. 3) | $a_i^{(t)}$ |
| Memini | $\delta$ (반복당 감쇠, $(1-\delta)$로 등장) | $\alpha_{\mathrm{sa}} := 1-\delta$ |
| Memini | $S$ (전역 확산 계수, Eq. 3) | $s_{\mathrm{sp}}$ |
| Memini | $b$ (이진 입력 크기, $=1$) | $b_{\mathrm{in}}$ |
| Memini | $n$ (Benna–Fusi 사슬 길이 / 공기 이벤트 수) | $n_{\mathrm{BF}}=2$ / $N_{\mathrm{ev}}$ |
| Memini | $T$ (확산 반복 수) / top-$k$ | $T_{\mathrm{sa}}$ / $k_{\mathrm{ret}}$ |
| Memini | $\tau_{\mathrm{fast}},\tau_{\mathrm{slow}}$ | $\tau_{\mathrm{f}},\tau_{\mathrm{s}}$ (유지) |

이 표에서 가장 위험한 두 행을 따로 짚는다. 첫째, Zep의 $f$는 (R)의 $f$가 **아니다.**
Zep의 $f(\alpha)=\chi(\rho(\phi(\alpha)))=\beta$는 질의 문자열을 문맥 문자열로 바꾸는
검색 API이고, 답을 만드는 모델은 그 바깥에 있다 [Zep §3]. 둘을 겹쳐 읽으면 Zep이 답을
생성한다고 오해하게 된다. 둘째, Memini의 fast/slow는 이 책의 $W$/$\Theta$가 **아니다.**
Memini의 두 변수는 외부 그래프 간선에 붙은 스칼라 쌍이고, 모델 안에는 아무것도 없다.
같은 낱말이 다른 층을 가리킨다.

## 15.4 어느 층을 언제 쓰는가

다섯 시스템 전부 (U-E)만 실행한다. (U-$\Theta$)는 항등사상이고 (U-W)는 $W$가 존재하지
않으므로 정의되지 않는다. 그래서 층의 *순서* 문제는 발생하지 않는다. 대신 발생하는 것은
**시계(時計)의 문제**다 — 쓰기가 어느 시점에 걸려 있는가.

이 장에서 $k$는 쓰기 라운드 첨자다. 다섯 시스템에서 한 라운드가 무엇인지가 다르고, 그중
실제로 유휴 시간에 도는 것은 일부뿐이다.

표 15-3. 어느 층을, 어느 시계로, 어떤 순서로

| 시스템 | 실행 규칙 | 라운드 $k$의 단위 | 요청 경로 위인가 | 진짜 오프라인 성분 |
|---|---|---|---|---|
| Mem0 | (U-E) | 메시지쌍 $(m_{k-1},m_k)$ | **위** — ingestion 경로에서 동기 실행 | 비동기 요약 갱신 1종 [Mem0 §2.1] |
| Zep | (U-E) | 에피소드 1건 = 메시지 1건 | **위** — 메시지 도착에 결합 | 주기적 community 전면 재계산 [Zep §2.3] |
| ReasoningBank | (U-E) | 과제 1건 | 아래 — 과제 종료 후, 다음 질의 전 | 판정 + 추출 2회 호출 |
| SCM | (U-E) | 트리거 발화 1회 | 추출은 위, 재조직·삭제는 아래 | NREM·REM·prune 전 과정 |
| Memini | (U-E) | 문서 1건 | 아래 — 문서 수집 시점 | 전 간선 Euler 1스텝 |

Mem0과 Zep은 ch01의 조작적 정의 중 "질의 $q$가 도착하기 전"은 만족하지만 "유휴 시간에"는
만족하지 않는다. Zep의 쓰기는 메시지 도착률에 결합되어 있고, Mem0의 추출·갱신은
"incremental processing paradigm, enabling it to operate seamlessly within ongoing
conversations" [Mem0 §2.1]이라고 논문 자신이 밝힌다. 두 시스템에서 유일하게 배치
작업의 모양을 갖는 것은 Mem0의 비동기 요약 모듈과 Zep의 community 전면 재계산인데,
**둘 다 주기가 공개되어 있지 않고 둘 다 어떤 실험에서도 돌지 않았다.** Zep은 그 필요성을
스스로 인정한다 — "the resulting communities gradually diverge from those that would be
generated by a complete label propagation run. Therefore, periodic community refreshes
remain necessary" [Zep §2.3].

SCM은 유일하게 트리거를 상태의 함수로 만든다. 엔트로피 $H(E_{\mathrm{wm}}) > \vartheta_{\mathrm{e}}=0.9$,
모순 밀도 $\rho_{\mathrm{cf}} > \vartheta_{\mathrm{c}}=0.3$, 경과 시간 $\Delta t > \tau_{\mathrm{sleep}}=1$시간,
또는 수동 호출 중 하나면 발화한다 [SCM §3.6]. 서빙 시스템에 맞는 모양이다. 다만 로그의
밑이 명시되어 있지 않고, 7칸 버퍼에서 항목이 균등 중요도일 때 $H=\log k$이므로 nat이면
$k\ge3$, bit이면 $k\ge2$에서 이미 임계를 넘는다. 논문은 어떤 실험에서 sleep이 몇 번
돌았는지 보고하지 않는다.

"이 값은 누가 학습하는가?"에 대한 답은 다섯 논문 전체에서 하나다 — **아무도 학습하지
않는다.** 정확히 세 부류만 존재한다. (a) 남이 사전학습해 동결한 체크포인트, (b) 사람이
손으로 고른 상수, (c) 그 동결 모델이 프롬프트에 따라 뱉은 텍스트. Mem0의 $m_{\mathrm{rec}}=10$과
$s_{\mathrm{nn}}=10$은 한 번 정해지고 한 번도 스윕되지 않았으며, 노드 병합 임계
$t_{\mathrm{sim}}$과 삼중항 관련도 임계는 "configurable"이라고만 적히고 값이 공개되지
않았다 [Mem0 §2.1–2.2]. SCM의 중요도 가중치 $(0.30,\,0.20,\,0.35,\,0.15)$는 "determined
through ablation on the benchmark suite" [SCM §3.3]이고, $w_{\mathrm{rec}}=0.2$는
"grid search over $\beta_2$ in [0.1, 0.5], selecting the value that maximized noise
reduction" [SCM §3.6]이다. Memini의 결정적 파라미터인 시상수 비 $\tau_{\mathrm{s}}/\tau_{\mathrm{f}}=5$는
"chosen so that the three expected regimes are clearly distinguishable within the 13-step
window" [Memini App. A.2]다.

> **[해설]** training을 처음 배우는 독자에게 이 절은 Part II에서 유일하게 "누가
> 학습하는가"의 답이 "사람이 숫자를 골랐다"인 자리다. 이것 자체는 결함이 아니라
> $E$-경로의 정의적 성질이다(→ ch11). 결함이 되는 지점은 §15.6이다 — 그 숫자를 고른
> 벤치마크와 그 숫자로 낸 결과를 보고한 벤치마크가 같을 때다.

ReasoningBank의 MaTTS는 이 표에서 유일하게 예산의 성격이 다르다. 병렬 스케일링은 같은
질의에 대해 $m$개의 궤적을 굴린 뒤 그 대조를 한 번의 프롬프트로 요약해 $E$에 넣는다
[ReasoningBank §3.3]. 통일 표기로 쓰면 $\mathcal{H}^{\Delta}_k = \mathrm{gen}(q_k,
\mathrm{ret}(E_k,q_k);\,B_s^{\mathrm{roll}})$ 다음에 $\hat c_k = S(\mathcal{H}^{\Delta}_k;
B_s^{\mathrm{sum}})$이다. 즉 예산의 일부가 **요약할 경험을 제조하는 데** 쓰인다.
$\mathrm{gen}(\cdot;B_s)$는 (U-$\Theta$)의 부품인데, ReasoningBank은 그것을 만들어놓고
gradient 대신 프롬프트로 보낸다. 이 한 걸음의 차이가 $E$-경로와 $\Theta$-경로를 가른다.

## 15.5 비용 4종

표 15-4. 다섯 시스템의 $B_s$ · $L_w$ · $C_{\mathrm{cap}}$ · $\rho$

| 시스템 | $B_s$ | $L_w$ | $C_{\mathrm{cap}}$ | $\rho$ |
|---|---|---|---|---|
| Mem0 | **논문에 없음** — 유일한 구축 비용 진술은 "graph construction completes in under a minute even in worst-case scenarios" [§4.5] | search p50 0.148 s / p95 0.200 s; total p50 0.708 s / p95 1.440 s. Mem0$^g$ 0.476/0.657, 1.091/2.590. full-context total 9.870/17.117 [Table 2] | 토큰 대용치만: 저장소 ~7k(Mem0) / ~14k(Mem0$^g$) / >600k(Zep) / 원문 ~26k [§4.5]. **바이트·dtype·임베딩 차원 모두 논문에 없음** | **논문에 없음** |
| Zep | **논문에 없음** — 토큰·FLOP·시간·호출 수 전부 없음 | LongMemEval$_s$ end-to-end: gpt-4o-mini 31.3 s → 3.20 s (IQR 8.76 → 1.31), gpt-4o 28.9 s → 2.58 s (IQR 6.01 → 0.684) [Table 2] | **논문에 없음** — 노드·간선 수, 저장 바이트 전무. 1024차원 임베딩만 언급되고 dtype 없음 [§2.2.1] | **논문에 없음** |
| ReasoningBank | 과제당 3748.4 토큰 = judge 2186.3 + extraction 1562.1. 총 53054.5 vs No Memory 50847.4 [Table 5, §C.2]. **MaTTS 구성의 $B_s$는 논문에 없음** | **논문에 없음** — ms·TTFT·시간 전무. 대용치는 평균 스텝 수 AS뿐 | **논문에 없음** — 항목 수·바이트·임베딩 차원 전무. 상한 규칙만 존재(과제당 3개) | **논문에 없음** |
| SCM | **논문에 없음** — 벤치 전체 wall-clock 약 12분(8개 시험 × 5회)이 유일한 수치 [§3.8] | 검색만: 개념 10개에서 0.1 ms 미만, 360개에서 0.3 ms 미만 (Apple M1 MacBook Air, 질의 1000회) [§4.3]. **지배적 비용인 wake 추출 LLM 호출은 측정 안 함** | 개념 수만: SCM 24 / No-Forget 72 / Vector DB 55 / FIFO 7 [Table 2]; 지연 시험 360개; 기본 target_size 100 [§3.6]. **바이트 논문에 없음** | **논문에 없음** — 90.9%는 의도된 삭제율이지 열화율이 아니다 |
| Memini | **논문에 없음** — 실제로 돌린 실험에서는 LLM 추출을 문자열 매칭으로 대체해 토큰 비용이 0이었다 [App. A.2] | **논문에 없음** — 읽기 경로를 한 번도 실행하지 않았다 [App. A.5] | 형태만: 방향 간선당 스칼라 2개; 실험은 개체 20개·고유 쌍 68개 [§3.1, App. A.2]. **바이트·정밀도 논문에 없음** | 감쇠 시상수 $\tau_{\mathrm{f}}=2$, $\tau_{\mathrm{s}}=10$(문서 단위). **성능 손실률은 논문에 없음** |

네 칸 중 $\rho$는 다섯 논문 전부에서 비어 있고, $C_{\mathrm{cap}}$은 바이트 기준으로
다섯 전부에서 비어 있다. $B_s$는 넷에서 비어 있고 ReasoningBank 하나만 토큰으로 채워져
있다. $L_w$는 둘에서 비어 있고 하나(SCM)는 지배적 성분을 빼고 채워져 있다.

$\rho$의 부재가 특히 심각한 두 자리가 있다. Mem0은 이 코퍼스에서 **상태를 되돌릴 수 없이
파괴할 수 있는 첫 $E$-경로 시스템**인데 [Mem0 Alg. 1], ADD/UPDATE/DELETE/NOOP 분류의
정확도, 오삭제율, 회차에 따른 열화 곡선을 하나도 측정하지 않는다. 게다가 추출이 저장소에서
가져온 요약에 조건화되므로 (15-2)의 되먹임 고리가 드리프트를 누적시킬 바로 그 구조인데,
논문은 각 대화를 한 번만 수집하고 재수집 실험을 하지 않는다. SCM은 임계 미달 노드를
삭제하면서, 20 사이클 결과를 유일하게 제시하는 Figure 3을 논문 스스로 "simulated memory
growth over twenty sleep cycles" [SCM §4.3]라고 부른다. 측정된 결과는 전부 sleep 1회
이내다.

상각식 (A)로 손익분기 $N_q^\ast$를 계산할 수 있는 논문은 다섯 중 **하나도 없다.**
$C_{\mathrm{sleep}}=B_s$가 넷에서 비어 있어 분자가 없고, 남은 하나인 ReasoningBank도
$N_q$ — 한 문맥을 공유하는 질의 수 — 를 보고하지 않는다. ReasoningBank에서 각 과제는
자기 자신의 문맥이므로 $N_q=1$이고, 재사용은 과제들 *사이에서* top-1 코사인 검색으로
일어나는데, 저장된 항목이 몇 번 검색되었는지에 대한 분포가 없다. 계산할 수 있는 것은
라운드 1회의 오버헤드 비뿐이다: $3748.4/49306.1 = 7.60\%$ (이 책의 산술, [ReasoningBank
Table 5]의 값에서). 이것은 $N_q$에 대한 상각이 아니라 한 라운드의 회계다.

Zep의 경우는 더 날카롭다. DMR 프로토콜은 "500 multi-session conversations, each containing
5 chat sessions with up to 12 messages per session. Each conversation includes a
question/answer pair" [Zep §4.2]다. 최대 60개 메시지를 아홉 단계 파이프라인으로 전부
수집한 뒤 **질의 하나**로 상각한다. (A)에서 $N_q=1$이면 $C_{\mathrm{avg}} = C_{\mathrm{wake}} + B_s$이고,
$B_s$는 인쇄되지 않았다.

> **[평가]** 비용 4종 표의 빈칸은 우연이 아니다. 다섯 논문 모두 **읽기 쪽 비용은
> 정밀하게 세고 쓰기 쪽 비용은 세지 않는다.** Zep은 이 비대칭을 스스로 선언한다 —
> "we've focused heavily on the accuracy, latency, and scalability of its memory
> **retrieval** mechanisms" [Zep §1, 강조는 이 책]. 그러면서 결론에서는
> "while reducing token costs" [Zep §5]라고 쓰는데, 논문에 있는 토큰 수치는 전부
> 읽기 쪽 프롬프트 크기다. 이 장의 집필 규칙상, 비용 칸이 비어 있는 시스템에 대해
> "효율적"이라고 쓸 수 없다. 쓸 수 있는 것은 "읽기 쪽이 싸다"뿐이다.

## 15.6 실험과 스케일

증거 기반부터 정직하게 적는다. Mem0은 LOCOMO 대화 10건이 전부이고, 각 대화는 평균
약 600턴·약 26000토큰·약 200문항이다 [Mem0 §3.1]. 모든 호출 지점이 GPT-4o-mini
한 모델이고, 판정자 모델은 "a separate, more capable LLM" [Mem0 §3.2]이라고만 적힌 채
끝까지 이름이 나오지 않는다. Zep은 DMR 대화 500건(각 최대 60메시지, 문항 1개)과
LongMemEval$_s$(평균 약 115,000토큰)를 쓴다 [Zep §4.2–4.3]. ReasoningBank은 WebArena
684건(Map 도메인 제외; Shopping 187 / Admin 182 / Gitlab 180 / Reddit 106 / Multi 29),
Mind2Web 1341건, SWE-Bench-Verified 500건을 쓴다 [ReasoningBank App B.1–B.2]. 셋 다
남이 만든 벤치마크를 쓴다 — ch14와 비교할 때 실질적 개선이며, 의무 caveat가 둘을
뭉뚱그리지 않도록 명시해 둔다. 나머지 둘은 자체 제작이다. SCM의 여덟 시험과 세 baseline은
모두 저자가 만든 단순 재구현이고 [SCM §4.2], Memini는 저자가 고른 COVID-19 위키백과
13편에 저자가 미리 정한 개체 20개를 얹었다 [Memini App. A.1–A.2].

**첫 번째 증거 — Mem0.** Table 2의 J 순위는 full-context 72.90 ± 0.19% > Mem0$^g$
68.44 ± 0.17% > Mem0 66.88 ± 0.15% > Zep 65.99 ± 0.16%다. 기억 구조가 전혀 없는 구성이
1위이고, 격차는 base 대비 6.02점, 그래프 변형 대비 4.46점으로 보고된 ±1 표준편차의 바깥에
한참 있다. 논문도 본문에서 인정한다 — "a full-context method ... still achieves the
highest J score (approximately 73%)" [Mem0 §4.3]. 그런데 초록은 순서를 뒤집어
"Beyond accuracy gains, we also markedly reduce computational overhead"라고 쓴다.
더 나아가, 이 논문의 헤드라인 확장인 그래프 기억은 절반의 범주에서 **더 나쁘다**:
single-hop J 65.71 vs 67.13, F1 38.09 vs 38.72; multi-hop J 47.19 vs 51.15, F1 24.32 vs
28.64 [Mem0 Table 1]. 저자들은 이를 그대로 적으면서 원인을 분리하는 ablation을 제시하지
않는다 — "the addition of graph memory in Mem0$^g$ does not provide performance gains here,
indicating potential inefficiencies or redundancies in structured graph representations"
[Mem0 §4.2]. 그리고 그 확장은 검색 지연 p50 0.148 s → 0.476 s, 검색 토큰 1764 → 3616을
치른다 — 이 책의 산술로 각각 3.22배와 2.05배다. 초록이 "consistently outperform all
existing memory systems across four question categories"라고 쓰는 동안, open-domain에서는
Zep이 F1 49.56과 J 76.60으로 Mem0$^g$의 49.27·75.71을 이기고 있고, 결론 §5는 네 범주 중
open-domain만 조용히 빼고 세 범주의 상대 개선만 나열한다.

**두 번째 증거 — Zep.** 이 논문에는 **ablation이 하나도 없다.** 스스로 novelty로 내세운
이중 시간 표현 — "This bi-temporal approach represents a novel advancement in LLM-based
knowledge graph construction" [Zep §2.1] — 을 뺀 변형을 한 번도 돌리지 않았고, reflexion
패스도, 개체 해소 단계도, community 계층도 분리되지 않았다. 그 상태에서 DMR 결과를 보면,
gpt-4-turbo에서 full-conversation baseline 94.4%가 이미 비교 대상 MemGPT 93.4%를 넘고,
Zep은 94.8%로 그 위에 0.4%p를 얹는다. gpt-4o-mini에서는 full-conversation 98.0%,
Zep 98.2%로 0.2%p다. 대화 500건에서 0.2%p는 대화 한 건이다. 신뢰구간도, 시드도, 반복
실행도, 유의검정도 논문 어디에도 없다. 게다가 비교 상대인 MemGPT 93.4%는 인용값이다 —
Table 1이 "† Results reported in [3]"로 표시하고, 본문은 "We were unable to reproduce
MemGPT's results using gpt-4o-mini" [Zep §4.2]라고 적으며, LongMemEval$_s$에서는 MemGPT를
아예 돌리지 못했다 [Zep §4.3.1]. 직접 비교는 존재하지 않는다. 분모도 일관되지 않다 —
Table 3의 상승 델타는 baseline을, 하락 델타는 Zep 자신을 분모로 쓰고, §4.3.2가 gpt-4o에
대해 내세우는 18.5%는 Table 2의 값으로 $(71.2-60.2)/60.2=18.27\%$다(이 책의 산술).
초록은 18.5%를 그대로 옮긴다. 저자들은 "showing marginal improvements over both
MemGPT and the respective full-conversation baselines" [Zep §4.2]라고 적고, 같은 절에서
이 벤치마크가 부적절하다고 스스로 판정한다 — "The high performance achieved by simple
full-context approaches using modern LLMs further highlights the benchmark's inadequacy
for evaluating memory systems" [Zep §4.2]. 더 결정적인 것은 LongMemEval$_s$의 범주별
결과다. gpt-4o-mini에서 knowledge-update는 76.9% → 74.4%로 **떨어진다** [Zep Table 3].
knowledge-update는 사실이 바뀔 때 저장소가 버티는가를 재는 범주, 즉 무효화 기제가 존재하는
바로 그 이유다. 같은 표에서 single-session-assistant는 두 모델 모두에서 퇴행한다
(81.8 → 75.0, 94.6 → 80.4). 논문은 후자만 언급하고 [Zep §4.3.2], 전자는 언급하지 않는다.
그리고 gpt-4o에서는 knowledge-update가 78.2% → 83.3%로 오른다 — **이득의 부호가 생성
모델의 성능에 따라 바뀐다.** 원인은 §15.3.2에서 이미 봤다. 유효 구간이 프롬프트 안의
문자열로 전달되므로, 그것을 쓰는 일은 검색의 보장이 아니라 추론자의 과제다. 논문 자신의
표현으로 "additional development may be needed to improve less capable models'
understanding of Zep's temporal data" [Zep §4.3.2]이다.

**세 번째 증거 — ReasoningBank.** 헤드라인은 실재한다. WebArena 전체 SR이
Gemini-2.5-flash에서 40.5 → 48.8, Gemini-2.5-pro에서 46.7 → 53.9, Claude-3.7-Sonnet에서
41.7 → 46.3이고 [ReasoningBank Table 1], 평균 스텝은 9.7 → 8.3, 8.8 → 7.4, 8.0 → 7.3으로
줄며, SWE-Bench-Verified에서도 34.2 → 38.8(flash), 54.0 → 57.4(pro)다 [Table 2]. 문제는
그 다음이다. §C.1은 검색해 넣는 경험의 개수를 0에서 4까지 늘리며 SR을 잰다: 39.0, 49.7,
46.0, 45.5, 44.4 [Figure 13]. 하나를 넘어가면 **단조 악화**다. 그리고 출하 기본값은
$n_{\mathrm{ret}}=1$, 즉 0이 아닌 최소값이다 [App A.2]. 이 곡선은 "경험이라는 새로운
스케일 축"이라는 논문의 틀과 정면으로 부딪히는데, 논문의 해석은 "excessive experiences
may introduce conflicts or noise"에서 멈춘다. 일반화 주장의 근거도 좁다. §4.2가 내세우는
"최강 baseline 대비 평균 +4.6"은 Multi 부분집합에서 나오고, 그 부분집합은 **29
인스턴스**다 — 한 과제가 3.4점이다. Claude-3.7-Sonnet의 Multi 열은 0.0 / 0.0 / 0.0 /
3.4 / 10.3, 즉 0건·0건·0건·1건·3건이다. Mind2Web의 과제 수준 SR은 바닥에 붙어 있다
(cross-domain flash 1.0 → 1.6, pro 1.4 → 1.7). 이 모든 수치에 시드도, 반복 실행도,
신뢰구간도 없다. 에이전트는 temperature 0.7, 추출자는 1.0으로 디코딩하므로 [ReasoningBank
App A.2, App B.1] 인쇄된 값은 각각 한 번의 확률적 추출이고, 29건·106건짜리 부분집합의
델타는 과제 한두 건 안에 들어간다. 산문이 자기 표와 어긋나는 자리도 있다 — §4.4는
"scaling actually reduces performance for weaker memories, where Synapse slightly increases
from 40.6 to 41.2, and AWM from 44.4 to 45.5"라고 쓰는데 인용된 두 수치는 모두 상승이다.
마지막으로, 모든 쓰기를 여닫는 자기 판정기의
실제 정확도는 **72.7%**다 [ReasoningBank §5]. 그 견고성을 재는 실험은 실제 판정기를
쓰지 않고 정답 라벨을 확률적으로 뒤집어 만든 가상 판정기를 쓴다 — "A 90% accurate
verifier is simulated by using the correct (ground-truth) label 90% of the time and the
incorrect (flipped) label 10% of the time" [§5]. 이 잡음은 독립이고 대칭이다. 실제 판정기의
오류는 그렇지 않다. 게다가 100% 지점은 정답 라벨을 그대로 쓰는 오라클이며, 이는 "no ground
truth is available during test-time" [§3.1]이라는 이 방법의 전제를 그 자리에서 소비한다.

**네 번째 증거 — SCM.** 제목이 consolidation을 말하지만 가중치는 한 번도 바뀌지 않는다 —
"It requires no training or fine-tuning" [SCM §3.8]. ch11의 판별식으로 보면 이 논문은
(U-E)를 실행하므로 sleep-time compute이지만, 제목이 약속하는 $\Theta$ 갱신은 존재하지
않는다. 그리고 자기 표를 그대로 읽으면 여기서도 같은 일이 벌어진다. Table 2에서 아무것도
지우지 않는 No-Forget Graph는 22/22(100%)로 SCM과 **동률**이고, SCM이 이기는 것은 저장소
크기 24 대 72뿐이다. 즉 품질에서 삭제 기제의 기여는 0이고, 기여는 전부 용량 쪽에 있다.
여덟 시험은 모두 정확히 1.00이고 다섯 번 실행에서 분산이 0이다 [SCM Table 3, §4.1] —
포화된 벤치마크는 대안을 순위 매길 수도, 퇴행을 잡아낼 수도 없다. 논문도 난이도를 인정한다
("This represents a lower bound on memory system capability" [§4.3]). 헤드라인 숫자
90.9%는 두 개의 서로 다른 개수에 동시에 붙어 있다 — Table 3은 "50/55 noise concepts
removed (90.9%)"이고 §4.3과 Figure 5는 "removing forty-five of the fifty noise
concepts"인데 45/50은 90.0%다. 이 숫자를 만든 계수는 벤치마크 위에서 격자 탐색으로
정해졌고, 그 이전 값 $w_{\mathrm{rec}}=0.4$에서는 노이즈 제거율이 0%였다 [§3.6, §4.3].
용량을 묶는다는 핵심 기제 Eq. 10은 인쇄된 형태 그대로는 본문의 주장과 반대 방향으로
움직인다 — $\vartheta_{\mathrm{f}} = \overline{I} - \mathrm{sd}(I)\cdot|G|/\text{target\_size}$에서
$\mathrm{sd}(I)\ge 0$이면 $|G|$가 커질수록 임계가 **낮아진다**. Eq. 9도 같은 상태다.
$\delta(c)=\exp(-\lambda_{\mathrm{rec}}\Delta t)$이면 방금 접근한 개념은 $1-\delta=0$인데,
§3.6은 고쳐낸 버그를 "$1-\delta(c)\approx 1$인 새 개념을 부양했다"로 설명한다. 같은 절의
$\lambda_{\mathrm{rec}}=0.01$ 교정("한 시간 전 개념이 최근성 점수의 약 96%를 유지")도
$\Delta t$의 단위가 없어 재현되지 않는다 — 시간 단위면 99.0%, 분 단위면 54.9%다(이 책의
산술). 코드는 공개되지 않았다.

**다섯 번째 증거 — Memini.** 유일한 실험에서 LLM을 한 번도 돌리지 않았다. 설계상 쓰기
경로는 LLM 개체 추출인데 [Memini Fig. 1(a)], 실제로 쓴 것은 "case-insensitive
word-boundary string matching against the 20 predefined terms and a small set of
hand-curated aliases" [App. A.2]다. 그래프도 자라지 않았다 — 노드 20개가 실행 전에
손으로 고정되었다. 읽기 경로는 아예 실행되지 않았고, 확산 활성의 네 하이퍼파라미터
($\alpha_{\mathrm{sa}},\,s_{\mathrm{sp}},\,T_{\mathrm{sa}},\,k_{\mathrm{ret}}$) 중 값이
주어진 것은 하나도 없다. 저자들이 그렇게 적는다 — "we report no retrieval metrics or
comparison against published systems" [App. A.5]. 남은 정량 결과 Table 3에서 헤드라인
"약 30배"는 $N=5$ 쌍에서 0.104 대 0.003이고, 세 구성의 지표가 서로 비교 가능하지 않다
(Uniform 열은 사실상 그룹별 평균 이벤트 수라서 모든 행에서 가장 크다). 검색이 활성값으로
순위를 매긴다는 사실을 함께 놓으면 더 곤란한 결과가 드러난다: 이 기제가 존재하는 이유인
"반복되었으나 최근에 언급되지 않은" 그룹은 0.104로 끝나고, "드물게 언급되었으나 최근인"
그룹은 0.463이다. 논문은 이 순위 역전을 논하지 않는다.

표 15-5. 다섯 시스템의 실증 상한

| 시스템 | 데이터 | 모델 | 저장소 최대 | 반복 라운드 |
|---|---|---|---|---|
| Mem0 | LOCOMO 대화 10건, 단일 데이터셋 | GPT-4o-mini 단일 | ~14k 토큰(Mem0$^g$) | 1회 수집, 재수집 없음 |
| Zep | DMR 500건 + LongMemEval$_s$(~115k 토큰) | gpt-4o-mini / gpt-4o / gpt-4-turbo | 미보고 | 1회 수집, 장기 실행 없음 |
| ReasoningBank | WebArena 684 + Mind2Web 1341 + SWE-Bench-V 500 | Gemini-2.5-flash/pro, Claude-3.7, Gemma-3-12B | 미보고(과제당 항목 ≤3) | 과제 스트림 최대 500 |
| SCM | 저자 제작 8시험, 10턴 대화 22사실 | Llama 3.2(2B, Q4_K_M) + MiniLM-384 | 개념 360개(지연 시험만) | 측정은 1사이클, 20사이클은 시뮬레이션 |
| Memini | 위키백과 13편, 개체 20개, 이벤트 124건 | **없음** — 문자열 매칭 | 고유 쌍 68개 | Euler 13스텝 |

## 15.7 Systems/serving 함의

decode에서 바뀌는 것은 없다. 다섯 시스템 전부 $\Theta$가 동결되어 있으므로 요청별
가중치 상태가 생기지 않고, **shared-weight batching이 그대로 성립한다.** Rosetta 사전의
"batch로 weight 공유 → $\Theta$-경로에서 깨짐" 행은 여기서 적용되지 않는다(→ ch01).
백만 명의 사용자가 한 벌의 가중치를 공유하고, 차이는 데이터베이스가 어떤 행을 돌려주는지에만
있다. 이것이 $E$-경로의 구조적 서빙 이점이고, 다섯 논문 중 아무도 명시하지 않는다.

읽기 쪽에서 실제로 바뀌는 것은 **prefill 길이**다. Mem0은 질의당 문맥을 26031토큰에서
1764토큰으로 줄이고(이 책의 산술로 14.76배), 총 지연 p95는 17.117 s에서 1.440 s로 줄어든다
(11.89배) — 두 비가 가까운 것은 이 길이대에서 지연이 prefill 지배적임을 시사한다. Zep은
115k에서 1.6k로 줄인다(이 책의 산술로 71.9배). 어느 논문도 prefill과 decode를 분해하지 않고, 하드웨어·배치
크기·동시성을 밝히지 않으며, Zep은 보스턴의 소비자용 노트북에서 AWS us-west-2의 호스팅
서비스에 접속해 측정했다고 공개한다 [Zep §4.3]. 네트워크 지연이 Zep 쪽에만 붙었으므로
방향은 Zep에 불리하지만, 두 팔이 프롬프트 길이에서 71.9배 차이 나는 이상 이 비교는
기억 구조의 비교가 아니라 프롬프트 길이의 비교다.

> **[평가]** 이 절에서 반드시 짚어야 할 공백이 하나 있다. Mem0 전문에 "cache"라는
> 단어는 한 번도 나오지 않는다. 그런데 비교 대상인 full-context baseline은 같은 대화에
> 대한 약 200개 질의마다 동일한 26k 프리픽스를 다시 prefill한다 [Mem0 §3.1, §4.3].
> prefix caching이 걸린 서빙 스택에서는 그 prefill이 한 번만 청구된다. 91%/92%라는
> 지연 절감의 얼마가 아키텍처의 몫이고 얼마가 baseline 구성의 몫인지는 이 논문에서
> 판정할 수 없다. 반대로 Mem0·Zep의 검색 결과는 질의마다 달라 prefix cache로 재사용되지
> 않고, ReasoningBank은 검색 항목을 시스템 지시문 맨 앞에 붙이므로 과제 간 공유
> 프리픽스를 위치 0에서 깨뜨린다 [ReasoningBank App A.2]. 셋 다 논문이 다루지 않는다.

쓰기 쪽은 **메시지 속도로 도는 두 번째 추론 워크로드**다. Zep은 메시지 하나마다 여섯에서
일곱 번의 frozen LLM 호출과 임베딩·색인 검색을 돌리고 [Zep §2.2], Mem0은 메시지쌍마다
추출 프롬프트 한 번과 갱신 tool call 한 번을 돌린다 [Mem0 §2.1]. ReasoningBank은 과제당
두 번, 합계 3748.4토큰이다. 호의적인 성질 하나는 어느 논문도 말하지 않는다 — Zep의 $S$는
메시지 다섯 개만 보므로 쓰기 경로는 **짧고 균일하며 배치하기 좋은 프롬프트의 흐름**이고,
이는 그것이 대체하는 115k 단일 프롬프트보다 좋은 prefill 워크로드다. 불리한 성질도 아무도
말하지 않는다 — 쓰기 비용은 대화 트래픽에 비례하고 읽기 절감은 질의 트래픽에 비례하므로,
순이득은 전적으로 $N_q$에 달려 있다.

상태가 앉는 자리도 다르다. Zep의 검색 backend는 Neo4j의 Lucene 구현이고 [Zep §3.1]
배포는 네트워크 너머의 호스팅 서비스다. $E$-경로의 상태는 호스트 DRAM과 SSD에, 벡터
색인과 역색인 뒤에 있고, HBM에는 평범한 모델 가중치만 있다. 그래서 이 경로의 지연 하한은
커널이 아니라 네트워크 왕복이다 — Zep의 상대 산포가 baseline보다 나쁜 것이 그 흔적이다
(IQR/지연 = 1.31/3.20 = 0.409 대 8.76/31.3 = 0.280, 이 책의 산술).

SCM과 Memini는 sleep 작업의 성격을 한 번 더 갈라놓는다. SCM의 sleep 절차에는 **LLM 호출이
하나도 없다** — Hebbian 곱, 전 간선 스칼라 곱, 5스텝 무작위 보행 3회, 임계 검사 한 번이
전부다 [SCM Alg. 1]. 이것은 GPU 학습 작업이 아니라 CPU 배경 압축 작업이고, LSM-tree의
compaction에 훨씬 가깝다. Memini의 쓰기는 문서마다 저장소 전체를 훑는 $O(|\mathcal{E}|)$
작업이다.

> **[해설]** Memini의 (15-7)은 LTI이므로 각 간선을 마지막으로 건드린 시점에서
> $z_e(t) = e^{A(t-t_{\mathrm{last}})} z_e(t_{\mathrm{last}})$로 지연 계산하면 전수 훑기가
> $O(|\hat c_k|)$로 줄어든다. 그리고 "Each pair is integrated independently using the same
> parameters" [Memini App. A.2]이므로 간선 단위 샤딩은 안전하다 — 문서 단위 샤딩은
> 궤적 의존성 때문에 안전하지 않은데도 그렇다. 논문은 비용도 이 완화책도 언급하지 않는다.

마지막으로 운영 위험 하나를 남긴다. 되돌릴 수 없는 삭제가 두 시스템에 있고
(Mem0의 $\textsf{DELETE}$, SCM의 prune), 둘 다 되돌리기·tombstone·감사 로그가 없으며,
둘 다 삭제 판단의 정확도를 측정하지 않는다. SCM의 경우 계수 하나가 잘못 설정되면 삭제
기제 전체가 아무 신호 없이 무력화된 사례가 논문 안에 기록되어 있다 [SCM §4.3, Figure 5].

$C_{\mathrm{cap}}$을 바이트로 환산할 수 있는 논문이 하나도 없으므로, 이 다섯 위에
memory-device 논증을 세울 수는 없다. 격차를 기록하고 여기서 멈춘다(→ ch26, ch29).

## 15.8 한계와 bridge-out

다섯 시스템은 서로를 거의 인용하지 않으면서 같은 뼈대를 세 번씩 다시 만들었다. 그 뼈대를
여기서 못 박는다.

> **정의.** **$E$-경로 프로덕션 삼단 구조**는 (i) 동결 LLM이 도착한 문맥에서 후보
> 레코드를 뽑는 **추출**, (ii) 같은 동결 LLM이 기존 저장소와 대조해 쓰기 연산자를 고르는
> **통합**, (iii) 임베딩·역색인·그래프 탐색으로 후보를 뽑아 프롬프트 문자열로 직렬화하는
> **검색**의 세 단계로 이루어진 구조를 말한다. 학습되는 파라미터는 없고, 시스템의 실질적
> '가중치'는 프롬프트와 사람이 고른 상수다.

> **정의.** **$E$-경로 프로덕션 공통 결함**은 이 구조를 채택한 논문들에서 반복 관찰되는
> 다섯 가지 보고 결함이다. (D1) 쓰기 예산 $B_s$ 미보고. (D2) 상태 용량을 바이트로
> 보고하지 않음. (D3) ablation 부재, 또는 자기 표가 자기 캡션을 반증. (D4) 기억 없는
> baseline이 품질에서 이기는 사실을 본문에서 인정하고 초록·결론에서 누락. (D5) 추출자·판정자·응답자가
> 같은 동결 모델인 닫힌 루프에 외부 검증 없음.

표 15-6. 다섯 시스템의 공통 결함 분포

| 시스템 | D1 | D2 | D3 | D4 | D5 |
|---|---|---|---|---|---|
| Mem0 | ○ | ○ | ○ (제안 시스템 ablation 0건; 스윕은 RAG baseline에만 14구성) | ○ (§4.3 인정, 초록·결론 누락) | ○ (판정자 모델 미공개) |
| Zep | ○ | ○ | ○ (ablation 0건) | ○ (§4.2 인정, 초록은 SOTA 대비만) | ○ (LongMemEval 판정자 GPT-4o, 피평가 모델과 동일 계열) |
| ReasoningBank | — (토큰으로 보고) | ○ | — (Figure 7이 기여를 분리) | 부분 (검색 개수 곡선을 본문에 실음) | ○ (판정자·추출자·에이전트가 같은 backbone) |
| SCM | ○ | ○ | — (Table 4 존재; 단 두 성분이 무효과) | ○ (No-Forget이 품질 동률) | ○ (벤치·baseline·튜닝 모두 저자) |
| Memini | ○ | ○ | 부분 (단일 시상수 대조군 1종) | ○ (Uniform 열이 모든 행에서 최대) | ○ (개체·문서·순서 모두 저자 선정) |

논문들이 스스로 남긴 open question은 서로 겹친다. Mem0 §5는 그래프 지연 최적화, 계층적
기억, 그리고 "developing more sophisticated memory consolidation mechanisms inspired by
human cognitive processes"를 든다 — CLS도, 재생도, 수면 신경과학도 인용하지 않고, "sleep"이라는
단어를 한 번도 쓰지 않은 채 이 책의 주제 쪽으로 손을 뻗는 한 문장이다. Zep §5는 추출기를
fine-tune하면 $E$-경로 파이프라인이 싸지겠는가를 묻고("Similar models fine-tuned for
Graphiti prompts may enhance knowledge extraction"), 이어서 이 책의 비용표와 같은 질문을
제기한 다음 그 절반만 답한다 — "Current literature on LLM memory and RAG systems
insufficiently addresses production system scalability in terms of cost and latency. We
have included latency benchmarks for our **retrieval** mechanisms to begin addressing this
gap" [Zep §5, 강조는 이 책]. ReasoningBank App D는 합성 가능한 기억, 타입별 기억, 그리고
"long-term ... consolidated knowledge with **decay/refresh policies**"를 미래 과제로
든다 — 논문 자신이 $\rho$에 붙인 이름이 미래 과제 목록에 있다. SCM은 하이퍼파라미터
근거와 10,000개 개념 천장을, Memini는 검색을 포함한 전면 검증을 남긴다.

> **[평가]** 이 다섯은 **비용·지연 결과를 성능 결과로 제시하고 있다.** 논증은 §15.6이
> 다섯 번 반복한 그대로다. Mem0의 자기 표에서 품질 1위는 full-context이고, Zep의 DMR에서
> 자명한 full-conversation baseline이 이미 비교 대상을 이기며 남은 마진은 0.4%p이고,
> SCM의 Table 2에서 아무것도 지우지 않는 구성이 품질 동률이며, ReasoningBank은 검색
> 경험을 하나만 넘겨도 단조 악화하고, Memini는 검색을 한 번도 돌리지 않았다. 다섯
> 논문의 초록은 모두 성능의 언어로 쓰여 있고, 다섯 논문의 증거는 모두 비용의 언어로
> 되어 있다.
>
> 이것이 무가치하다는 뜻이 아니다. **비용 절감은 실재한다.** 질의당 문맥이 26031토큰에서
> 1764토큰으로, 115k에서 1.6k로 줄어드는 것은 측정된 사실이고, 그것은 서빙에서 곧바로
> 돈이다. ReasoningBank의 과제당 3748.4토큰 오버헤드로 WebArena 전체 SR이 40.5에서
> 48.8로 오르는 것도 측정된 사실이다. 문제는 **주장의 종류가 바뀌어 있다**는 것이다.
> "우리 시스템이 더 정확하다"와 "우리 시스템은 훨씬 싼 값에 거의 그만큼 정확하다"는
> 다른 주장이고, 후자를 방어하려면 싼 값 쪽 — 즉 $B_s$ — 을 세어야 한다. 다섯 중 넷은
> 그것을 세지 않았다. 그러므로 이 장의 결론은 "이 시스템들이 나쁘다"가 아니라
> **"이 시스템들의 주장은 현재 형태로는 검증 대상이 아니다"** 이며, 검증 가능한 형태로
> 바꾸는 일이 Part III의 과제다(→ ch24, ch25, ch27).

**경로 간 인용.** 침묵은 거의 완전하고, 방향이 셋이다. $\Theta$-경로로: Mem0의 참고문헌에는
LoRA도, adapter·PEFT도, 모델 편집도, 지속학습도, 증류도, 용량 연구도 없다. ReasoningBank은
gradient를 한 번도 취하지 않으면서 $\mathcal{R}_k$에 해당하는 자기 생성·자기 라벨링된
성공/실패 분할 코퍼스를 실제로 만들고, $\Theta$-경로가 존재한다는 사실을 언급하지 않는다.
SCM은 EWC를 "operates at the parameter level rather than the memory architecture level"
[SCM §2.2]로, Memini는 파라미터 지속학습을 서베이 수준에서 각각 한 번 기각한다. $W$-경로로:
Mem0의 유일한 순환 상태 인용(Recurrent Memory Transformer)은 문제의 증거로만 쓰이고 해결책
후보로는 쓰이지 않는다. Zep 전문에서 'test-time', 'fast weight', 'gradient', 'KV', 'cache'는
0회다. Memini는 Titans와 Nested Learning을 한 문장에서 "validated independently"로 부르고
다시 언급하지 않는다. 생물학으로: Mem0은 어휘만 빌리고 문헌은 빌리지 않는다(McClelland도,
CLS도, 수면 신경과학도 없다). 반면 SCM과 Memini에서는 생물학 인용이 실제로 하중을 진다 —
Rasch & Born, Tononi & Cirelli, Benna & Fusi, Kaplanis, McClelland, Tulving이 기제의 출처다.

> **[평가]** 이 침묵의 가장 날카로운 형태는 인용 누락이 아니라 **재발명**이다. SCM의
> Eq. 7은 외부 그래프의 간선 강도에 상수 배율을 곱해 오래된 것을 지우는데, 식 (15-6)이
> 보여주듯 이것은 (U-W)의 retention gate와 대수적으로 같은 형태다. Memini의 두 시상수도
> 같은 계보를 외부 저장소 위에서 다시 세운다. 두 논문 모두 그 형식이 이미 $W$층에서
> 정리되어 있다는 사실을 모르는 채로 도달했다.
>
> 연대와 선택의 구분은 **대상별로** 해야 한다. ch14를 인용하지 않은 것은 Zep과 Mem0에서만
> 연대가 강제한 침묵이다 — Zep은 석 달 앞서 나왔고 Mem0은 열흘 뒤다. 그러나 다섯 논문이
> $W$·$\Theta$ 경로에 대해 지키는 침묵은 어느 논문에서도 연대의 문제가 아니다. LoRA는
> 2021년, ROME·MEMIT은 2022년, TTT 라인 전체가 Zep보다 앞선다. 그리고 선택된 침묵에서
> 대가는 인용 예절이 아니라 중복 노동이다.

**다음 장이 받아가는 것.** 이 장에서 $\mathrm{ret}(E,q)$의 산출물은 언제나 **문자열**이었다.
Zep의 이중 시간은 "FACT (Date range: from - to)"라는 텍스트로 모델에 전달되고, ReasoningBank의
전략은 시스템 지시문에 붙는 몇 문장이며, Mem0의 기억은 1764개의 prefill 토큰이다. 그래서
$E$-경로의 읽기 비용은 **검색된 바이트가 곧 prefill 토큰이 되는** 형태이고, 항목의 개수보다
간결성이 더 중요하다는 §15.6의 관찰(검색 경험 1개 → 4개에서 49.7 → 44.4)이 그 대가다.

ch16은 여기서 층을 옮긴다. Generative Adapter는 같은 문맥을 문자열이 아니라 **한 번의
forward로 만든 파라미터 모양의 산출물**로 바꾼다. 옮겨 가는 질문은 셋이다. 첫째, §15.3.1의
쓰기 연산자 분류 — 확장 전용·수축 가능·그래프 delta·동역학 — 중 어느 것이 $W$층에서
살아남는가. 둘째, §15.3.2의 세 가지 시간 표현 중 무엇이 파라미터 모양의 상태 위에서
표현 가능한가. 셋째, 그리고 가장 중요하게, 이 장이 발견한 패턴 — 비용 결과가 성능 결과의
자리에 놓이는 패턴 — 이 $W$-경로에서도 반복되는가. ch16은 그 검사의 첫 시험대이고,
초록의 헤드라인 수치가 본문 표 어디에도 없는 논문이라는 점에서 이미 예고편을 갖고 있다.

## 요약

- 이 장의 다섯 시스템은 전부 (U-E)만 실행한다. $\Theta$는 한 번도 움직이지 않고 $W$는
  존재하지 않으며, gradient도 손실 함수도 논문 안에 없다.
- 쓰기 연산자는 네 형태로 갈린다: 네 연산 문법(Mem0), 그래프 delta(Zep·Mem0$^g$),
  합집합(ReasoningBank), 동역학(SCM·Memini). 이 중 수축 가능한 것은 Mem0의 $\textsf{DELETE}$와
  SCM의 prune 둘뿐이고, 둘 다 삭제 정확도를 측정하지 않는다.
- 시간은 세 방식으로만 표현된다: 명시적 타임스탬프(Zep의 이중 시간), 감쇠 상수(SCM·Memini),
  순서 첨자(ReasoningBank). Zep의 타임스탬프는 프롬프트 문자열로 전달되므로 그것을 쓰는
  일은 검색의 보장이 아니라 추론자의 과제다.
- 비용 4종에서 $\rho$는 다섯 전부에서, 바이트 기준 $C_{\mathrm{cap}}$도 다섯 전부에서
  비어 있다. $B_s$는 ReasoningBank의 과제당 3748.4토큰이 유일한 수치이고, 그것조차
  MaTTS 구성은 덮지 않는다. 상각식 (A)의 $N_q^\ast$는 다섯 어디에서도 계산되지 않는다.
- 자기 표를 읽으면 품질 1위가 반복적으로 기억 시스템 없는 구성이다: full-context 72.90%
  대 Mem0 66.88%, full-conversation 94.4% 대 MemGPT 93.4%, No-Forget Graph 22/22 대
  SCM 22/22.
- $E$-경로 프로덕션 시스템은 shared-weight batching을 그대로 보존한다. 이것이 $\Theta$-경로
  대비 결정적 서빙 이점이며, 다섯 논문 중 아무도 명시하지 않는다.
- 다섯 논문은 비용·지연 결과를 성능 결과로 제시하고 있다. 비용 절감은 실재하지만
  주장의 종류가 바뀌어 있고, 그 주장을 방어하려면 세지 않은 쪽인 $B_s$를 세어야 한다.
- 침묵의 대가는 인용 예절이 아니라 중복 노동이다. SCM은 (U-W)의 retention gate를 외부
  그래프 위에서 재발명했다.

## 자가 점검 체크리스트

- [ ] 다섯 시스템 각각에 대해 "라운드 $k$가 무엇인가"와 "그 라운드가 요청 경로 위인가
  아래인가"를 표 15-3을 보지 않고 답할 수 있다.
- [ ] Mem0의 (15-2)가 표준형 (U-E)와 다른 점 두 가지를 말할 수 있고, 그 차이가 왜 $\rho$를
  실재하는 위험량으로 만드는지 설명할 수 있다.
- [ ] Zep이 knowledge-update에서 퇴행한 사실(76.9% → 74.4%, gpt-4o-mini)과 같은 범주가
  gpt-4o에서는 오른 사실(78.2% → 83.3%)을 §15.3.2의 시간 표현 방식으로 설명할 수 있다.
- [ ] "이 시스템은 full-context보다 효율적이다"라는 문장이 이 장의 다섯 논문 중 어느 것에
  대해서도 쓸 수 없는 이유를, 비용 4종 표의 빈칸을 지목해 말할 수 있다.
- [ ] ReasoningBank의 판정기 정확도 72.7%와 대칭 잡음 시뮬레이션의 관계를 설명하고,
  왜 후자가 전자의 견고성 근거가 되지 못하는지 말할 수 있다.
- [ ] SCM이 판별식은 통과하면서도 제목이 약속한 $\Theta$ 갱신은 하지 않는다는 진술을,
  ch11의 판별식 네 조건에 비추어 검증할 수 있다.
- [ ] (Rosetta) prefix cache와 $\hat c$의 차이를 이 장의 사례로 말할 수 있다 — Mem0·Zep의
  검색 결과는 질의마다 달라 prefix cache로 재사용되지 않고, ReasoningBank은 검색 항목을
  시스템 지시문 맨 앞에 붙여 공유 프리픽스를 위치 0에서 깨뜨린다. 즉 $E$-경로의 $\hat c$는
  "같은 계산의 재사용"이 아니라 "새 추론의 선행 수행"이고, 그 대가로 캐시 재사용을 잃는다.




